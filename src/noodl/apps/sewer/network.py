"""`SewerNetwork`, its validation, and the (Model, State, Drivers) builder.

Follows `apps/street_aq/network.py`'s builder shape: dataclasses describing the physical
network, one `validate()` that refuses every topology this application does not model BY NAME,
one `build_model` that assembles the `Network`, the layers and the closures and
returns the triple, and one `initial_state` dispatching on each layer's `quantity`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from noodl.apps.sewer.air import (
    F_AIR_DEFAULT,
    F_I_DEFAULT,
    add_air_edges,
    air_layer,
)
from noodl.apps.sewer.controls import ControlSet
from noodl.apps.sewer.dynwave import (
    DynamicWaveHydraulics,
    DynamicWaveOptions,
    air_state,
    build_dynamic_wave_model,
    quality_state,
)
from noodl.apps.sewer.hydraulics import GEOMETRY_ALIASES, SewerHydraulics
from noodl.apps.sewer.inflows import InflowSchedule
from noodl.apps.sewer.pumps import PUMP_KINDS
from noodl.apps.sewer.quality import H2STransfer, LateralLoads, SulfideGeneration
from noodl.apps.sewer.storage import validate_storage_curve
from noodl.apps.sewer.xsect_spec import UNMODELLED_KINDS, Xsect
from noodl.layers.transport import TransportLayer, active_interior
from noodl.model import Model
from noodl.refs import initial_drivers as refs_initial_drivers
from noodl.refs import state_from
from noodl.topology import Network

#: SWMM `FLOW_ROUTING` keywords -> the physical routing names.
ROUTING_ALIASES = {"STEADY": "steady", "KINWAVE": "kinematic_wave", "DYNWAVE": "dynamic_wave"}
ROUTINGS = ("steady", "kinematic_wave", "dynamic_wave")
OUTFALL_KINDS = ("free", "normal", "fixed", "tidal", "timeseries")

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]


@dataclass(frozen=True)
class Manhole:
    """A manhole: `invert` and `ground` elevations (m), a constant lateral `inflow`
    (m3/s), for storage its `surface_area` (m2), and SWMM's `surcharge_depth` (SurDepth,
    m >= 0): how far dynamic-wave routing lets the water rise above the full depth before
    the manhole floods (kinematic and steady routing ignore it). `initial_depth` is SWMM's
    InitDepth (m >= 0), the water depth the dynamic wave starts from (0 lets SWMM average
    the depths of the attached conduits, `DynamicWaveHydraulics.initial_state`). A
    `storage` curve makes it a SWMM storage unit (dynamic wave only).

    The `storage` curve (`storage_unit` builds such a manhole) is its surface area,
    ``("functional", a, b, c)`` for ``A = a y^b + c``
    (m2, y the depth in m), ``("tabular", depths, areas)`` (m, m2) or one of SWMM's closed
    shapes (`noodl.apps.sewer.storage`). Its area adds to the conduits' at the junction, its
    volume is reported (``"sewer.storage_volume"``, m3), its `ground` is the invert plus
    SWMM's MaxDepth, and SWMM's storage rules apply: its full depth is extended to the
    attached conduit crowns only with a `surcharge_depth`, it surcharges (EXTRAN) only above
    its full depth and only with a `surcharge_depth`, and it keeps its own `initial_depth`
    (0 included) at the start."""

    name: str
    invert: float
    ground: float | None = None
    inflow: float = 0.0
    surface_area: float | None = None
    surcharge_depth: float = 0.0
    initial_depth: float = 0.0
    storage: tuple | None = None


def storage_unit(name: str, invert: float, max_depth: float, curve: tuple, *,
                 surcharge_depth: float = 0.0, initial_depth: float = 0.0,
                 inflow: float = 0.0) -> Manhole:
    """A SWMM storage unit (`[STORAGE]`) as a `Manhole` with a `storage` curve: `invert`
    (m), `max_depth` (SWMM's MaxDepth, m >= 0: the manhole's ground is ``invert +
    max_depth``), the surface-area `curve` (`Manhole.storage`), SWMM's SurDepth
    `surcharge_depth` (m) and InitDepth `initial_depth` (m), and a constant lateral `inflow`
    (m3/s). Tabular depths and areas are stored as tuples (the manhole is hashable)."""
    if curve and curve[0] == "tabular" and len(curve) == 3:
        curve = ("tabular", tuple(float(v) for v in curve[1]),
                 tuple(float(v) for v in curve[2]))
    validate_storage_curve(name, curve)
    if not max_depth >= 0.0:
        raise ValueError(f"storage unit {name!r}: max_depth must be >= 0 m, got {max_depth!r}")
    return Manhole(name, invert, ground=invert + max_depth, inflow=inflow,
                   surcharge_depth=surcharge_depth, initial_depth=initial_depth, storage=curve)


@dataclass(frozen=True)
class Outfall:
    """An outfall, where the network discharges: its `invert` elevation (m).

    `kind` is ``"free"``, ``"normal"``, ``"fixed"``, ``"tidal"`` or ``"timeseries"`` (SWMM
    FREE/NORMAL/FIXED/TIDAL/TIMESERIES); ``"fixed"`` needs `stage`, the water-surface
    elevation (m). ``"tidal"`` needs `tide`, SWMM's TIDAL curve as ``(hour, stage)`` pairs
    (hour of the day, stage elevation in m, strictly increasing hours): the stage is read
    from it at the elapsed time of the run, repeating every 24 h
    (`DynamicWaveHydraulics`, `tidal_stage`). ``"timeseries"`` takes its stage elevation
    from a driver, the `outfall_stage` function of `build_model` / `simulate_dynamic_wave`;
    `series` is only the SWMM series' name, for messages. Only dynamic-wave routing reads
    them.
    `flap_gate` (SWMM's Gated) blocks any flow from the outfall into its conduit
    (`link.c::link_setFlapGate`, dynamic wave only).
    """

    name: str
    invert: float
    kind: str = "free"
    stage: float | None = None
    flap_gate: bool = False
    tide: tuple[tuple[float, float], ...] | None = None
    series: str | None = None


@dataclass(frozen=True)
class Pipe:
    """A conduit from manhole `u` to `v`: `length` and `diameter` (m), Manning's `n`
    and bed `slope` (m/m). `inlet_offset` and `outlet_offset` (m) are the heights of the
    pipe invert above the invert of its `u` and `v` node (SWMM's link offsets).

    Dynamic wave only (every other routing refuses them when set): `barrels` identical
    parallel conduits (SWMM's Barrels, 1 to 127; `length`, `diameter`, `n` and `slope` are
    one barrel's, and the flows, volumes and surface areas of the pipe are totals);
    `max_flow` (m3/s, SWMM's MaxFlow: each barrel carries at most this, `dwflow.c:270-273`;
    ``None`` for none);
    `initial_flow` (m3/s, SWMM's InitFlow, a total, positive from `u` to `v`);
    `flap_gate` closes the conduit on reverse flow (SWMM's FlapGate), reverse being from
    `v` to `u` unless `flap_gate_reversed`, where the gate passes `v` to `u` flow instead
    (a conduit the SWMM reader turned round because it was adverse: its file orientation is
    `v` to `u`, and SWMM's ``Link.direction`` is -1).

    `xsect` (dynamic wave only; ``None`` for the circle of `diameter`) is the conduit's SWMM
    cross section (`xsect_spec.Xsect`: a shape keyword and its geometry fields, or an
    IRREGULAR transect, a CUSTOM shape curve); `diameter` is then its full height (m), which
    must equal `Xsect.height`. `loss_inlet`, `loss_outlet` and `loss_average` are SWMM's
    entry, exit and average local loss coefficients (`[LOSSES]`, -, dynamic wave only;
    `dwflow.c::findLocalLosses`). For an IRREGULAR conduit `length` is SWMM's
    ``conduit_getLength`` (the file's length over the transect's length factor) and `n` the
    roughness SWMM then uses (the transect's main-channel n times the root of that factor,
    `link.c::conduit_validate`); `transect_length_factor` is that factor (-, 1 for every other
    conduit): it sets the momentum length when the conduit is not lengthened
    (`link.c:1105-1118`). `culvert` is SWMM's culvert code (`[XSECTIONS]` field 8; 0 for none,
    1 to 57 for the inlet types of FHWA HEC-5, dynamic wave only; `culvert.py`).
    """

    name: str
    u: str
    v: str
    length: float
    diameter: float
    n: float
    slope: float
    inlet_offset: float = 0.0
    outlet_offset: float = 0.0
    barrels: int = 1
    max_flow: float | None = None
    initial_flow: float = 0.0
    flap_gate: bool = False
    flap_gate_reversed: bool = False
    xsect: Xsect | None = None
    loss_inlet: float = 0.0
    loss_outlet: float = 0.0
    loss_average: float = 0.0
    transect_length_factor: float = 1.0
    culvert: int = 0


#: Orifice orientations (SWMM's SIDE, BOTTOM) and opening shapes (CIRCULAR, RECT_CLOSED).
ORIFICE_ORIENTATIONS = ("side", "bottom")
ORIFICE_SHAPES = ("circular", "rectangular")
#: Weir kinds (SWMM's TRANSVERSE, SIDEFLOW, V-NOTCH, TRAPEZOIDAL; ROADWAY is not modelled).
WEIR_KINDS = ("transverse", "sideflow", "v_notch", "trapezoidal")


@dataclass(frozen=True)
class Orifice:
    """An orifice (SWMM `[ORIFICES]`, dynamic wave only) from node `u` to node `v`: a
    ``"side"`` (vertical) or ``"bottom"`` (horizontal) opening of `shape` ``"circular"``
    (`height` its diameter, m) or ``"rectangular"`` (`height` and `width`, m), its crest
    `crest_height` (m above `u`'s invert, SWMM's CrestHt), discharge coefficient
    `discharge_coefficient` (-, SWMM's Cd) and a `flap_gate` (closes on flow from `v` to
    `u`, with SWMM's ARMCO head loss on forward flow). Always fully open (no control
    rules). Flow follows `link.c::orifice_getInflow` (`noodl.apps.sewer.structures`)."""

    name: str
    u: str
    v: str
    orientation: str
    shape: str
    height: float
    crest_height: float
    discharge_coefficient: float
    width: float | None = None
    flap_gate: bool = False
    close_time: float = 0.0
    order: int = 0


@dataclass(frozen=True)
class Weir:
    """A weir (SWMM `[WEIRS]`, dynamic wave only) from node `u` to node `v`. `kind` is
    ``"transverse"`` or ``"sideflow"`` (a rectangular opening `width` m wide),
    ``"v_notch"`` (a triangle `width` m wide at the top) or ``"trapezoidal"`` (bottom
    `width` m, `side_slopes` the left and right side slopes, horizontal per vertical); its
    opening is `height` m tall above the crest `crest_height` (m above `u`'s invert).
    `discharge_coefficient` is SWMM's Cd in SI units (m^0.5/s, the flow ``Cd L h^1.5``,
    ``Cd L^0.83 h^1.67`` sideflow, ``Cd s h^2.5`` V-notch, in m and m3/s),
    `end_discharge_coefficient` a trapezoid's end triangles' Cd (SWMM's Cd2),
    `end_contractions` the number of end contractions (SWMM's EndCon), `can_surcharge` SWMM's
    CanSurcharge (above the opening the weir flows as an orifice; else its head is capped at
    the opening height) and `flap_gate` as for `Orifice`. Always fully open. Flow follows
    `link.c::weir_getInflow` (`noodl.apps.sewer.structures`)."""

    name: str
    u: str
    v: str
    kind: str
    height: float
    width: float
    crest_height: float
    discharge_coefficient: float
    side_slopes: tuple[float, float] = (0.0, 0.0)
    end_contractions: float = 0.0
    end_discharge_coefficient: float = 0.0
    can_surcharge: bool = True
    flap_gate: bool = False
    order: int = 0


@dataclass(frozen=True)
class Outlet:
    """An outlet (SWMM `[OUTLETS]`, dynamic wave only) from node `u` to node `v`, its crest
    `crest_height` m above `u`'s invert, with a rating ``("functional", a, b)`` for
    ``q = a h^b`` (m3/s, h in m) or ``("tabular", heads, flows)`` (m, m3/s; SWMM's RATING
    curve, heads strictly increasing, held at the end flows outside them). `h` is the
    upstream depth above the crest (SWMM's /DEPTH, the default) or, with `head_difference`,
    the head across the outlet (/HEAD). `flap_gate` closes it on flow from `v` to `u`. Flow
    follows `link.c::outlet_getInflow` (`noodl.apps.sewer.structures`)."""

    name: str
    u: str
    v: str
    crest_height: float
    rating: tuple
    head_difference: bool = False
    flap_gate: bool = False
    order: int = 0


@dataclass(frozen=True)
class Pump:
    """A pump (SWMM `[PUMPS]`, dynamic wave only) from node `u` (its inlet, a manhole) to node
    `v`. `kind` is SWMM's pump type: ``"type1"`` (PUMP1 curve: flow steps with the inlet
    node's volume), ``"type2"`` (PUMP2: steps with the inlet depth), ``"type3"`` (PUMP3:
    flow against the head delivered, `v` minus `u`), ``"type4"`` (PUMP4: continuous in the
    inlet depth), ``"type5"`` (PUMP5: the variable-speed PUMP3, head divided by the squared
    speed `setting`) or ``"ideal"`` (no curve: the pump takes its inlet node's inflow).
    `curve` is ``(x values, flows)`` in SI (x in m3 for type1, m otherwise; flows m3/s;
    x strictly increasing); an ideal pump has none. The pump starts ON (`initial_on`, SWMM's
    Status), switches OFF when its inlet depth falls below `shutoff_depth` (m) and back ON
    above `startup_depth` (m); 0 disables either. Control rules can switch it and set its
    speed (`ControlSet`). `order` is the line of the input file (SWMM's link index order)."""

    name: str
    u: str
    v: str
    kind: str = "ideal"
    curve: tuple = ()
    initial_on: bool = True
    startup_depth: float = 0.0
    shutoff_depth: float = 0.0
    order: int = 0

    @property
    def crest_height(self) -> float:
        """A pump has no crest (0, for the shared regulator checks)."""
        return 0.0

    @property
    def flap_gate(self) -> bool:
        """A pump has no flap gate (SWMM, link.c:1469)."""
        return False


def _validate_rating(where: str, rating) -> None:
    """Refuse a malformed outlet rating, BY NAME (`Outlet.rating`)."""
    if not isinstance(rating, tuple) or not rating or rating[0] not in ("functional",
                                                                       "tabular"):
        raise ValueError(f"{where}: rating must be ('functional', a, b) or ('tabular', "
                         f"heads, flows), got {rating!r}")
    if rating[0] == "functional":
        if len(rating) != 3 or not all(math.isfinite(float(x)) for x in rating[1:]):
            raise ValueError(f"{where}: a functional rating takes two finite values (a, b) "
                             f"for q = a h^b, got {rating!r}")
        return
    if len(rating) != 3 or len(rating[1]) != len(rating[2]) or not rating[1]:
        raise ValueError(f"{where}: a tabular rating needs as many flows as heads, at least "
                         f"one, got {rating!r}")
    heads = [float(x) for x in rating[1]]
    if not all(math.isfinite(x) for x in (*heads, *(float(y) for y in rating[2]))):
        raise ValueError(f"{where}: a tabular rating must be finite, got {rating!r}")
    if any(b <= a for a, b in zip(heads, heads[1:], strict=False)):
        raise ValueError(f"{where}: tabular rating heads must be strictly increasing (SWMM's "
                         f"ERR_CURVE_SEQUENCE), got {rating[1]!r}")


@dataclass(frozen=True)
class SewerNetwork:
    """A gravity sewer: manholes, pipes, outfalls. Kinematic-wave and steady routing need a
    dendritic network (one outgoing pipe per manhole); dynamic-wave routing allows loops
    and flat pipes.

    `routing` is ``"steady"``, ``"kinematic_wave"`` or ``"dynamic_wave"`` (SWMM's STEADY,
    KINWAVE and DYNWAVE are accepted and normalised); `dynamic_wave` holds the options of
    the last one. `geometry` selects the circular cross-section: ``"analytic"`` (the exact circle,
    `geometry.py`; the default for networks built in code) or ``"tabulated"`` (SWMM 5.2's own
    tables, lookup rules and unit constants, `xsect_tables.py`; the default of
    `read_swmm_inp`, whose purpose is to reproduce SWMM). ``"tabulated"`` is a steady
    kinematic-wave geometry and is refused with ``storage=True`` by `SewerHydraulics`;
    dynamic-wave routing always uses it and refuses ``"analytic"`` by name.
    `orifices`, `weirs` and `outlets` are flow regulators between two nodes and `pumps` pumps
    (dynamic wave only; SWMM's `[ORIFICES]`, `[WEIRS]`, `[OUTLETS]`, `[PUMPS]`); `controls`
    is the file's `[CONTROLS]` rule set (`ControlSet`) that switches pumps and sets the
    openings of regulators (dynamic wave only).
    `inflows` is the time-varying lateral inflow of a SWMM file (`InflowSchedule`: its
    `[TIMESERIES]`, `[PATTERNS]`, `[INFLOWS]` and `[DWF]`), ``None`` for constant
    `Manhole.inflow` only. When given, each scheduled manhole's `Manhole.inflow` is the value
    at the start of the run, and the model carries the schedule as ``model.inflow_schedule``;
    `simulate_dynamic_wave` uses it when it is given no ``inflow``.
    """

    manholes: tuple[Manhole, ...]
    pipes: tuple[Pipe, ...]
    outfalls: tuple[Outfall, ...]
    routing: str = "kinematic_wave"
    notes: dict[str, str] = field(default_factory=dict)
    geometry: str = "analytic"
    dynamic_wave: DynamicWaveOptions = DynamicWaveOptions()
    orifices: tuple[Orifice, ...] = ()
    weirs: tuple[Weir, ...] = ()
    outlets: tuple[Outlet, ...] = ()
    inflows: InflowSchedule | None = None
    pumps: tuple[Pump, ...] = ()
    controls: ControlSet | None = None

    @property
    def structures(self) -> dict[str, tuple]:
        """The flow regulators and pumps by edge kind: ``{"orifice": ..., "weir": ...,
        "outlet": ..., "pump": ...}``."""
        return {"orifice": self.orifices, "weir": self.weirs, "outlet": self.outlets,
                "pump": self.pumps}

    def __post_init__(self) -> None:
        object.__setattr__(self, "routing", ROUTING_ALIASES.get(self.routing, self.routing))
        object.__setattr__(
            self, "geometry", GEOMETRY_ALIASES.get(self.geometry, self.geometry)
        )

    def validate(self) -> None:
        """Refuse, BY NAME, everything this application does not model."""
        if self.geometry not in ("analytic", "tabulated"):
            raise ValueError(
                f"SewerNetwork: geometry must be 'analytic' or 'tabulated', got "
                f"{self.geometry!r}"
            )
        if self.routing not in ROUTINGS:
            raise ValueError(
                f"SewerNetwork: routing must be one of {ROUTINGS} (or the SWMM keywords "
                f"{tuple(ROUTING_ALIASES)}), got {self.routing!r}"
            )
        dynamic = self.routing == "dynamic_wave"
        if not dynamic:
            self._refuse_dynamic_wave_only()
        for manhole in self.manholes:
            if not manhole.surcharge_depth >= 0.0:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} surcharge_depth must be >= 0 m, "
                    f"got {manhole.surcharge_depth!r}"
                )
        names: set[str] = set()
        for node in (*self.manholes, *self.outfalls):
            if node.name in names:
                raise ValueError(f"SewerNetwork: duplicate node name {node.name!r}")
            names.add(node.name)
        for outfall in self.outfalls:
            if outfall.kind not in OUTFALL_KINDS:
                raise ValueError(
                    f"SewerNetwork: outfall {outfall.name!r} kind must be one of "
                    f"{OUTFALL_KINDS}, got {outfall.kind!r}"
                )
            if outfall.kind == "fixed" and not (
                outfall.stage is not None and math.isfinite(outfall.stage)
            ):
                raise ValueError(
                    f"SewerNetwork: fixed outfall {outfall.name!r} needs a finite stage "
                    f"(m elevation), got {outfall.stage!r}"
                )
            if outfall.kind == "tidal":
                self._validate_tide(outfall)
        pipe_names: set[str] = set()
        outgoing: dict[str, list[str]] = {}
        for pipe in self.pipes:
            if pipe.name in pipe_names:
                raise ValueError(f"SewerNetwork: duplicate pipe name {pipe.name!r}")
            pipe_names.add(pipe.name)
            for end in (pipe.u, pipe.v):
                if end not in names:
                    raise ValueError(
                        f"SewerNetwork: pipe {pipe.name!r} names node {end!r}, which is "
                        f"neither a manhole nor an outfall"
                    )
            if not pipe.slope > 0:
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} has slope {pipe.slope!r}; a gravity "
                    f"sewer pipe must fall from its upstream node to its downstream one "
                    f"(zero and adverse slopes are out of scope)"
                )
            if not pipe.diameter > 0 or not pipe.length > 0 or not pipe.n > 0:
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} must have positive length, diameter "
                    f"and Manning n, got {pipe.length!r}, {pipe.diameter!r}, {pipe.n!r}"
                )
            for label, offset in (("inlet", pipe.inlet_offset), ("outlet", pipe.outlet_offset)):
                if not offset >= 0:
                    raise ValueError(
                        f"SewerNetwork: pipe {pipe.name!r} has {label} offset {offset!r} m; "
                        f"an offset is the invert height above its node's invert and must "
                        f"be >= 0"
                    )
            outgoing.setdefault(pipe.u, []).append(pipe.name)
        self._validate_structures(names, pipe_names)
        if self.controls is not None:
            self.controls.validate(self)
        if self.inflows is not None:
            manhole_names = {m.name for m in self.manholes}
            stray = sorted(set(self.inflows.nodes) - manhole_names)
            if stray:
                raise ValueError(f"SewerNetwork: inflow schedule names node(s) {stray}, "
                                 f"which are not manholes (an outfall takes no lateral "
                                 f"inflow)")
        outfall_names = {o.name for o in self.outfalls}
        if dynamic:
            self._validate_dynamic(outfall_names)
            return
        for manhole in self.manholes:
            found = outgoing.get(manhole.name, [])
            if len(found) != 1:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} has {len(found)} outgoing "
                    f"pipes ({found}); a dendritic sewer gives every manhole exactly one"
                )
        for name in outgoing:
            if name in outfall_names:
                raise ValueError(
                    f"SewerNetwork: outfall {name!r} has an outgoing pipe; an outfall is a "
                    f"sink"
                )
        # Every manhole must reach an outfall by following its own outgoing pipe. A cycle
        # shows up here as a walk that never terminates, so it is bounded by the node count
        # and reported as "does not reach an outfall", which is what a cycle means.
        target = {pipe.u: pipe.v for pipe in self.pipes}
        limit = len(self.manholes) + 1
        stranded = []
        for manhole in self.manholes:
            node, steps = manhole.name, 0
            while node not in outfall_names and steps < limit:
                node = target.get(node)
                if node is None:
                    break
                steps += 1
            if node not in outfall_names:
                stranded.append(manhole.name)
        if stranded:
            raise ValueError(
                f"SewerNetwork: manhole(s) {stranded} -- their component does not reach an "
                f"outfall (a cycle in the pipe subgraph shows up here too)"
            )

    def _validate_structures(self, node_names: set[str], pipe_names: set[str]) -> None:
        """Refuse, BY NAME, a malformed orifice, weir or outlet: a link name used twice, an
        unknown or repeated end node, an unknown kind or shape, a non-positive opening, a
        negative crest height or coefficient, or a malformed rating."""
        link_names = set(pipe_names)
        outfall_names = {o.name for o in self.outfalls}
        for kind, devices in self.structures.items():
            for s in devices:
                where = f"SewerNetwork: {kind} {s.name!r}"
                if s.name in link_names:
                    raise ValueError(f"{where}: duplicate link name (pipes, orifices, weirs, "
                                     f"outlets and pumps share one name space)")
                link_names.add(s.name)
                for end in (s.u, s.v):
                    if end not in node_names:
                        raise ValueError(f"{where} names node {end!r}, which is neither a "
                                         f"manhole nor an outfall")
                if s.u == s.v:
                    raise ValueError(f"{where} joins node {s.u!r} to itself")
                if kind == "pump":
                    self._validate_pump(where, s, outfall_names)
                    continue
                if not (math.isfinite(s.crest_height) and s.crest_height >= 0.0):
                    raise ValueError(f"{where}: crest_height must be >= 0 m (above its first "
                                     f"node's invert), got {s.crest_height!r}")
                if kind == "outlet":
                    _validate_rating(where, s.rating)
                    continue
                if not (math.isfinite(s.discharge_coefficient)
                        and s.discharge_coefficient >= 0.0):                 # link.c:1675, 2040
                    raise ValueError(f"{where}: discharge_coefficient must be >= 0, got "
                                     f"{s.discharge_coefficient!r}")
                if not (math.isfinite(s.height) and s.height > 0.0):          # xsect.c:229
                    raise ValueError(f"{where}: height must be > 0 m, got {s.height!r}")
                if kind == "orifice":
                    if s.orientation not in ORIFICE_ORIENTATIONS:
                        raise ValueError(f"{where}: orientation must be one of "
                                         f"{ORIFICE_ORIENTATIONS}, got {s.orientation!r}")
                    if s.shape not in ORIFICE_SHAPES:
                        raise ValueError(f"{where}: shape must be one of {ORIFICE_SHAPES} "
                                         f"(SWMM: CIRCULAR or RECT_CLOSED), got {s.shape!r}")
                    if s.shape == "rectangular" and not (
                            s.width is not None and math.isfinite(s.width) and s.width > 0.0):
                        raise ValueError(f"{where}: a rectangular opening needs a width > 0 m, "
                                         f"got {s.width!r}")
                    continue
                if s.kind not in WEIR_KINDS:
                    raise ValueError(f"{where}: kind must be one of {WEIR_KINDS} (roadway "
                                     f"weirs are not modelled), got {s.kind!r}")
                slopes = tuple(s.side_slopes)
                if len(slopes) != 2 or not all(math.isfinite(x) and x >= 0.0 for x in slopes):
                    raise ValueError(f"{where}: side_slopes must be two values >= 0, got "
                                     f"{s.side_slopes!r}")
                if s.kind == "trapezoidal":
                    if not (math.isfinite(s.width) and s.width >= 0.0) or (
                            s.width == 0.0 and sum(slopes) == 0.0):           # xsect.c:482, 492
                        raise ValueError(f"{where}: a trapezoid needs a bottom width >= 0 m "
                                         f"and not both it and its side slopes 0, got "
                                         f"{s.width!r}, {s.side_slopes!r}")
                elif not (math.isfinite(s.width) and s.width > 0.0):         # xsect.c:366, 506
                    raise ValueError(f"{where}: width must be > 0 m, got {s.width!r}")
                for what, value in (("end_contractions", s.end_contractions),
                                    ("end_discharge_coefficient", s.end_discharge_coefficient)):
                    if not (math.isfinite(value) and value >= 0.0):          # link.c:2055-2062
                        raise ValueError(f"{where}: {what} must be >= 0, got {value!r}")

    @staticmethod
    def _validate_pump(where: str, pump: Pump, outfall_names: set[str]) -> None:
        """Refuse, BY NAME, a malformed pump: an unknown kind, a curve that does not fit the
        kind, an outfall as its inlet, or startup and shutoff depths out of order (SWMM's
        ERR_PUMP_LIMITS, link.c:1518-1519)."""
        if pump.kind not in PUMP_KINDS:
            raise ValueError(f"{where}: kind must be one of {PUMP_KINDS}, got {pump.kind!r}")
        if pump.u in outfall_names:
            raise ValueError(f"{where}: its inlet {pump.u!r} is an outfall; a pump draws from "
                             f"a junction or storage unit")
        if pump.kind == "ideal":
            if pump.curve:
                raise ValueError(f"{where}: an ideal pump has no curve")
        else:
            curve = pump.curve
            if (len(curve) != 2 or len(curve[0]) != len(curve[1]) or not curve[0]
                    or not all(math.isfinite(float(x)) for x in (*curve[0], *curve[1]))):
                raise ValueError(f"{where}: a {pump.kind} pump needs a curve (x values, flows) "
                                 f"of equal, non-zero length and finite values, got {curve!r}")
            xs = [float(x) for x in curve[0]]
            if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
                raise ValueError(f"{where}: pump curve x values must be strictly increasing "
                                 f"(SWMM's ERR_CURVE_SEQUENCE), got {curve[0]!r}")
        for what, value in (("startup_depth", pump.startup_depth),
                            ("shutoff_depth", pump.shutoff_depth)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{where}: {what} must be >= 0 m, got {value!r}")
        if pump.startup_depth > 0.0 and pump.startup_depth <= pump.shutoff_depth:
            raise ValueError(f"{where}: startup depth {pump.startup_depth} m must exceed "
                             f"the shutoff depth {pump.shutoff_depth} m (SWMM's "
                             f"ERR_PUMP_LIMITS)")

    def _refuse_dynamic_wave_only(self) -> None:
        """Steady and kinematic-wave routing do not model initial conditions, MaxFlow, flap
        gates, barrels, storage units or flow regulators; a set value would be skipped
        silently, so it is refused by name."""
        for kind, devices in self.structures.items():
            if devices:
                raise ValueError(
                    f"SewerNetwork: {kind} {devices[0].name!r}: orifices, weirs, outlets and "
                    f"pumps are modelled with dynamic-wave routing only (routing is "
                    f"{self.routing!r})")
        if self.controls is not None:
            raise ValueError(
                f"SewerNetwork: control rules are modelled with dynamic-wave routing only "
                f"(routing is {self.routing!r})")
        for manhole in self.manholes:
            if manhole.storage is not None:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} is a storage unit; only "
                    f"dynamic-wave routing models storage units (routing is "
                    f"{self.routing!r})")
            if manhole.initial_depth != 0.0:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} has initial_depth "
                    f"{manhole.initial_depth!r} m; only dynamic-wave routing models it "
                    f"(routing is {self.routing!r})")
        for outfall in self.outfalls:
            if outfall.flap_gate:
                raise ValueError(
                    f"SewerNetwork: outfall {outfall.name!r} has a flap gate; only "
                    f"dynamic-wave routing models it (routing is {self.routing!r})")
        for pipe in self.pipes:
            for what, value, default in (
                ("barrels", pipe.barrels, 1), ("max_flow", pipe.max_flow, None),
                ("initial_flow", pipe.initial_flow, 0.0), ("flap_gate", pipe.flap_gate, False),
                ("xsect", pipe.xsect, None), ("culvert", pipe.culvert, 0),
                ("loss_inlet", pipe.loss_inlet, 0.0),
                ("loss_outlet", pipe.loss_outlet, 0.0),
                ("loss_average", pipe.loss_average, 0.0),
            ):
                if value != default:
                    raise ValueError(
                        f"SewerNetwork: pipe {pipe.name!r} has {what} {value!r}; only "
                        f"dynamic-wave routing models it (routing is {self.routing!r})")

    @staticmethod
    def _validate_tide(outfall: Outfall) -> None:
        """A tidal outfall needs a TIDAL curve: finite (hour, stage) pairs with strictly
        increasing hours (SWMM's ERR_CURVE_SEQUENCE, table.c:313)."""
        tide = outfall.tide
        if not tide:
            raise ValueError(
                f"SewerNetwork: tidal outfall {outfall.name!r} needs a tide curve of "
                f"(hour, stage m) pairs, got {tide!r}"
            )
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in tide):
            raise ValueError(
                f"SewerNetwork: tidal outfall {outfall.name!r} tide curve must be finite, "
                f"got {tide!r}"
            )
        if any(b[0] <= a[0] for a, b in zip(tide, tide[1:], strict=False)):
            raise ValueError(
                f"SewerNetwork: tidal outfall {outfall.name!r} tide curve hours must be "
                f"strictly increasing, got {[x for x, _ in tide]}"
            )

    def _validate_xsect(self, pipe: Pipe) -> None:
        """Refuse, BY NAME, a cross section this package does not model, one that SWMM
        rejects, or one that disagrees with the pipe's `diameter`."""
        x = pipe.xsect
        if x.kind in UNMODELLED_KINDS:
            raise ValueError(f"SewerNetwork: pipe {pipe.name!r}: cross section {x.kind}: "
                             f"{UNMODELLED_KINDS[x.kind]}")
        if self.geometry != "tabulated":
            raise ValueError(
                f"SewerNetwork: pipe {pipe.name!r} has a {x.kind} cross section, which only "
                f"SWMM's tabulated geometry (geometry='tabulated') models, not "
                f"{self.geometry!r}")
        try:
            height = x.height
        except ValueError as exc:
            raise ValueError(f"SewerNetwork: pipe {pipe.name!r}: {exc}") from None
        if not math.isclose(pipe.diameter, height, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"SewerNetwork: pipe {pipe.name!r} diameter {pipe.diameter!r} m is not its "
                f"{x.kind} cross section's full height {height!r} m")

    def _validate_dynamic(self, outfall_names: set[str]) -> None:
        """Dynamic-wave topology: loops and any flow direction are allowed, but every
        connected component needs an outfall and each outfall ends exactly one conduit."""
        self.dynamic_wave.validate()
        for manhole in self.manholes:
            if not manhole.initial_depth >= 0.0:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} initial_depth must be >= 0 m, "
                    f"got {manhole.initial_depth!r}")
            if manhole.storage is not None:
                validate_storage_curve(manhole.name, manhole.storage)
                if manhole.surface_area is not None:
                    raise ValueError(
                        f"SewerNetwork: storage unit {manhole.name!r} sets surface_area; its "
                        f"area is its storage curve")
        for pipe in self.pipes:
            if (not isinstance(pipe.barrels, int) or isinstance(pipe.barrels, bool)
                    or not 1 <= pipe.barrels <= 127):
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} barrels must be an integer from 1 to "
                    f"127 (SWMM stores it in a char), got {pipe.barrels!r}")
            if pipe.max_flow is not None and not (
                    math.isfinite(pipe.max_flow) and pipe.max_flow > 0.0):
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} max_flow must be a positive finite "
                    f"flow (m3/s) or None, got {pipe.max_flow!r}")
            if not math.isfinite(pipe.initial_flow):
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} initial_flow must be finite, got "
                    f"{pipe.initial_flow!r} m3/s")
            if not (isinstance(pipe.culvert, int) and not isinstance(pipe.culvert, bool)
                    and pipe.culvert >= 0):
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} culvert code must be an integer >= 0, "
                    f"got {pipe.culvert!r}")
            for what in ("loss_inlet", "loss_outlet", "loss_average"):
                value = getattr(pipe, what)
                if not (math.isfinite(value) and value >= 0.0):
                    raise ValueError(
                        f"SewerNetwork: pipe {pipe.name!r} {what} must be a finite "
                        f"coefficient >= 0, got {value!r}")
            if not (math.isfinite(pipe.transect_length_factor)
                    and pipe.transect_length_factor > 0.0):
                raise ValueError(
                    f"SewerNetwork: pipe {pipe.name!r} transect_length_factor must be > 0, "
                    f"got {pipe.transect_length_factor!r}")
            if pipe.xsect is not None:
                self._validate_xsect(pipe)
        neighbours: dict[str, list[str]] = {}
        touching: dict[str, list[str]] = {name: [] for name in outfall_names}
        links = (*self.pipes, *self.orifices, *self.weirs, *self.outlets, *self.pumps)
        for link in links:
            neighbours.setdefault(link.u, []).append(link.v)
            neighbours.setdefault(link.v, []).append(link.u)
            for end in (link.u, link.v):
                if end in outfall_names:
                    touching[end].append(link.name)
        for name, found in touching.items():
            if len(found) != 1:
                raise ValueError(
                    f"SewerNetwork: outfall {name!r} is attached to {len(found)} links "
                    f"({found}); dynamic-wave routing takes exactly one link (a pipe, "
                    f"orifice, weir or outlet) per outfall (SWMM's ERR_OUTFALL)"
                )
        seen: set[str] = set()
        for manhole in self.manholes:
            if manhole.name in seen:
                continue
            component = {manhole.name}
            frontier = [manhole.name]
            while frontier:
                for other in neighbours.get(frontier.pop(), ()):
                    if other not in component:
                        component.add(other)
                        frontier.append(other)
            seen |= component
            if not component & outfall_names:
                raise ValueError(
                    f"SewerNetwork: manhole {manhole.name!r} and its connected pipes "
                    f"({sorted(component)}) reach no outfall"
                )

    def outgoing(self) -> dict[str, int]:
        """Manhole name -> the position of its outgoing pipe in `self.pipes`."""
        return {pipe.u: i for i, pipe in enumerate(self.pipes)}

    def levels(self) -> list[list[int]]:
        """Leaf-to-root level order over the manholes (positions into `self.manholes`)."""
        from noodl.apps.sewer.hydraulics import _levels

        return _levels(self.pipes, self.manholes)


def tree_steady() -> SewerNetwork:
    """The committed 5-conduit, 6-node fixture (`tests/data/sewer/tree_steady.inp`).

    J1, J2 (invert 12.0) and J5 (11.0) carry the constant inflows 0.05, 0.08 and 0.03 m3/s;
    C1 and C2 join at J3 (10.0), C3 and C4 join at J4 (9.0), C5 discharges to Outfall (8.0).
    Conduits are 200 m with n = 0.013; C1, C2, C4 are 0.30 m at slope 0.010 and C3, C5 are
    0.45 m at slope 0.005. Measured pipe discharges: 0.05, 0.08, 0.03, 0.13, 0.16 m3/s.
    """
    return SewerNetwork(
        manholes=(
            Manhole("J1", 12.0, inflow=0.05),
            Manhole("J2", 12.0, inflow=0.08),
            Manhole("J5", 11.0, inflow=0.03),
            Manhole("J3", 10.0),
            Manhole("J4", 9.0),
        ),
        pipes=(
            Pipe("C1", "J1", "J3", 200.0, 0.30, 0.013, 0.010),
            Pipe("C2", "J2", "J3", 200.0, 0.30, 0.013, 0.010),
            Pipe("C4", "J5", "J4", 200.0, 0.30, 0.013, 0.010),
            Pipe("C3", "J3", "J4", 200.0, 0.45, 0.013, 0.005),
            Pipe("C5", "J4", "Outfall", 200.0, 0.45, 0.013, 0.005),
        ),
        outfalls=(Outfall("Outfall", 8.0),),
    )


def build_model(
    net: SewerNetwork,
    *,
    storage: bool = False,
    air: bool | None = None,
    quality: bool | None = None,
    species: tuple[str, ...] = ("bod", "sulfide"),
    f_air: float = F_AIR_DEFAULT,
    f_i: float = F_I_DEFAULT,
    c_s: float = 1.0,
    leak_area: float = 8e-4,
    leak_cd: float = 0.6,
    fans: tuple = (),
    coupling: str = "pingpong",
    scheme: str = "implicit",
    dt_storage: float | None = None,
    iterate_tol: float | None = None,
    iterate_max: int | None = None,
    outfall_stage: Mapping | None = None,
    iterate_flow_tol: float | None = None,
) -> tuple[Model, State, Drivers]:
    """Assemble the sewer model and return `(model, state, drivers)`.

    `air=False` builds the water-only model the SWMM comparison tests use; `quality=False`
    drops both quality layers. `fans` names the manholes carrying a prescribed extraction
    (m3/s, positive OUT through the `fan` kind). `air` and `quality` default to ``True``
    for steady and kinematic-wave routing.

    A ``routing="dynamic_wave"`` network is built by `build_dynamic_wave_model`: `air` and
    `quality` default to ``False`` there, `coupling` and `dt_storage` (kinematic-wave
    options) are refused by name, and so is a ``geometry="analytic"`` network (dynamic
    wave always uses SWMM's tabulated circle); ``quality=True`` adds the water-quality layer of
    `species` on the solved flows (`scheme` its time scheme); ``air=True`` builds
    the headspace air on the solved conduits (`f_air`, `f_i`, `c_s`, `leak_area`, `leak_cd`
    and `fans` as below; dendritic networks only, loops refused by name), `storage` is
    refused; converged to
    `iterate_tol` (m) and `iterate_flow_tol` (m3/s) within `iterate_max` passes, by default
    the file's HEAD_TOLERANCE and MAX_TRIALS (SWMM's 0.005 ft and 8 where it sets none) and
    the flow tolerance matching them (`dynwave.iterate_tolerances`). Those three
    apply to dynamic wave only and are refused for the other routings. So is `outfall_stage`,
    the stage elevation of each TIMESERIES outfall (``{name: m, or a function of t in s}``;
    `build_dynamic_wave_model`).
    """
    if not (net.pipes or net.orifices or net.weirs or net.outlets):
        raise ValueError("build_model: the network has no links (no conduits, orifices, "
                         "weirs or outlets), so there is nothing to route; add at least one")
    if net.routing == "dynamic_wave":
        if storage:
            raise ValueError("build_model: storage=True is the kinematic path's storage "
                             "sweep; dynamic-wave junctions always store; build it with "
                             "storage=False")
        for name, value, default in (("coupling", coupling, "pingpong"),
                                     ("dt_storage", dt_storage, None)):
            if value != default:
                raise ValueError(
                    f"build_model: {name}={value!r} is a kinematic-wave option; a "
                    f"dynamic-wave network always iterates its passes on the routing step "
                    f"(net.dynamic_wave); leave {name} at its default")
        kwargs = {k: v for k, v in (("iterate_tol", iterate_tol),
                                    ("iterate_max", iterate_max),
                                    ("iterate_flow_tol", iterate_flow_tol)) if v is not None}
        return build_dynamic_wave_model(net, outfall_stage=outfall_stage,
                                        quality=bool(quality), species=species,
                                        scheme=scheme, air=bool(air), f_air=f_air, f_i=f_i,
                                        c_s=c_s, leak_area=leak_area, leak_cd=leak_cd,
                                        fans=fans, **kwargs)
    for name, value in (("iterate_tol", iterate_tol), ("iterate_max", iterate_max),
                        ("iterate_flow_tol", iterate_flow_tol),
                        ("outfall_stage", outfall_stage)):
        if value is not None:
            raise ValueError(
                f"build_model: {name} applies to dynamic-wave routing only; this network's "
                f"routing is {net.routing!r}"
            )
    air = True if air is None else air
    quality = True if quality is None else quality
    net.validate()
    graph = Network(dtype=F64)
    for manhole in net.manholes:
        graph.add_node(manhole.name, invert=manhole.invert, ground=manhole.ground)
    for outfall in net.outfalls:
        graph.add_node(outfall.name, invert=outfall.invert)
    if air:
        graph.add_node("ambient")
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)

    outgoing = net.outgoing()
    manhole_names = [m.name for m in net.manholes]
    manhole_idx = torch.tensor(
        [graph.nodes.index(name) for name in manhole_names], dtype=torch.long
    )
    # Manhole i's outgoing pipe's position in `net.pipes` (per-pipe driver order). On
    # `tree_steady()` this happens to be the identity, but `read_swmm_inp` reads `[CONDUITS]` in
    # its own file order, which need not match `[JUNCTIONS]` order at all -- the committed
    # `tree_steady.inp` gives `out_pipe = [0, 1, 3, 2, 4]`.
    out_pipe = torch.tensor([outgoing[name] for name in manhole_names], dtype=torch.long)
    slope_per_manhole = torch.tensor(
        [net.pipes[outgoing[name]].slope for name in manhole_names], dtype=F64
    )

    closures: list = []
    hydraulics = SewerHydraulics(
        graph, list(net.pipes), list(net.manholes), storage=storage, dt=dt_storage,
        geometry=net.geometry,
        surface_area=(
            torch.tensor(
                [m.surface_area if m.surface_area is not None else 1.167
                 for m in net.manholes], dtype=F64
            )
            if any(m.surface_area is not None for m in net.manholes)
            else None
        ),
        shaft_depth=(
            torch.tensor(
                [m.ground - m.invert for m in net.manholes], dtype=F64
            )
            if all(m.ground is not None for m in net.manholes)
            else None
        ),
    )
    closures.append(hydraulics)

    layers: dict = {}
    drivers: Drivers = {}
    state: State = {}

    if air:
        plan = add_air_edges(graph, net, fans)
        layers["air"] = air_layer(
            graph, net, plan, f_air=f_air, f_i=f_i, c_s=c_s, leak_area=leak_area,
            leak_cd=leak_cd, fans=fans,
        )
        drivers["air.phi_boundary"] = torch.zeros(1, dtype=F64)

    if quality:
        water_cap = torch.ones(len(net.manholes), dtype=F64)
        layers["water_quality"] = TransportLayer(
            graph, "water_quality", capacity=water_cap, flow_kind="pipe",
            boundary=[o.name for o in net.outfalls], n_species=len(species),
            scheme=scheme, quantity="concentration", unit="kg/m3", species_names=species,
            source_unit="kg/s", capacity_unit="m3", flow_unit="m3/s",
        )
        # `out_pipe` above indexes the per-pipe driver vectors by POSITION in `manhole_names`,
        # so the map is correct only if `TransportLayer`'s own active-interior order agrees
        # with `manhole_names` exactly (the same self-check `apps/street_aq/network.py`
        # performs) -- checked ONCE, here, rather than trusted.
        ordered = [graph.nodes[i] for i in layers["water_quality"].interior_idx.tolist()]
        if ordered != manhole_names:
            raise ValueError(
                f"build_model: the water_quality layer's active interior came out as "
                f"{ordered}, not the manhole order {manhole_names}; out_pipe indexes by "
                f"position into manhole_names and assumes they agree"
            )
        drivers["water_quality.x_boundary"] = torch.zeros(
            len(net.outfalls), len(species), dtype=F64
        )
        # The lateral inflow-concentration loads (`bod_in`/`sulfide_in`) are read by
        # `LateralLoads`, which is registered
        # whenever quality is built at all (with or without `air`), and BEFORE `H2STransfer`
        # below (which adds its own transfer term on top rather than overwriting).
        species_drivers = {"bod": "bod_in", "sulfide": "sulfide_in"}
        columns = {
            i: species_drivers[name] for i, name in enumerate(species)
            if name in species_drivers
        }
        closures.append(
            LateralLoads(graph.n, columns=columns, n_species=len(species))
        )
        if air:
            # `active_interior` calls `Network.endpoints` on every kind named, which raises
            # if the network carries no edge of that kind at all -- not merely an empty
            # interior. `fan` edges exist only when `fans` is non-empty, so the kind tuple
            # must match `flow_kind`'s below exactly rather than naming `fan` unconditionally
            # (which would raise `KeyError: unknown edge kind 'fan'` on every fan-less
            # network, including the default fixture).
            air_kinds = ("headspace", "leak", "fan") if fans else ("headspace", "leak")
            # Every pipe, including the one draining to the outfall, gets a PARALLEL
            # headspace edge (Conventions block), so the outfall NODE itself (not just
            # "ambient") is touched by a headspace edge -- it carries the outfall
            # structure's own headspace air. `H2STransfer` reads `air_quality.x` directly as
            # a per-MANHOLE array (matched against `manhole_idx`/`out_pipe`), so the outfall
            # node must be a BOUNDARY of this layer too, exactly like water_quality, or the
            # layer's active interior would be 6 nodes (5 manholes + the outfall) where 5
            # are wanted; `two_film_flux` would then fail on a shape mismatch against
            # water_quality's 5-node interior.
            air_boundary = ["ambient"] + [o.name for o in net.outfalls]
            air_interior, _ = active_interior(graph, air_kinds, air_boundary)
            layers["air_quality"] = TransportLayer(
                graph, "air_quality", capacity=torch.ones(air_interior.numel(), dtype=F64),
                flow_kind=air_kinds,
                boundary=air_boundary, n_species=1, scheme=scheme,
                quantity="concentration", unit="kg/m3", species_names=("h2s",),
                source_unit="kg/s", capacity_unit="m3", flow_unit="m3/s",
            )
            ordered = [graph.nodes[i] for i in layers["air_quality"].interior_idx.tolist()]
            if ordered != manhole_names:
                raise ValueError(
                    f"build_model: the air_quality layer's active interior came out "
                    f"as {ordered}, not the manhole order {manhole_names}; H2STransfer "
                    f"reads 'air_quality.x' as a per-manhole array and assumes they agree"
                )
            drivers["air_quality.x_boundary"] = torch.zeros(len(air_boundary), dtype=F64)
            closures.append(H2STransfer(graph.n, manhole_idx, out_pipe=out_pipe))

    reactions = (
        [(
            "water_quality",
            SulfideGeneration(out_pipe=out_pipe, manhole_idx=manhole_idx, n_nodes=graph.n),
        )]
        if quality else []
    )
    # `reaction_order="before_transport"`: SWMM's `qualrout.c::findLinkQual` reacts a
    # conduit's step-start concentration (`getReactedQual`) and then mixes it with the
    # inflow over the conduit volume (`getMixedQual`), and reports the concentration AFTER
    # mixing. Reacting before the transport step makes the returned `"water_quality.x"` that
    # same sample (the default order, after transport, would return `(1 - k dt)` times it at
    # the fixed point).
    model = Model(
        graph, layers, closures=closures, reactions=reactions, coupling=coupling,
        reaction_order="before_transport",
        iterate_tol=(
            {"water_quality": 1e-12, "air_quality": 1e-12}
            if coupling == "iterate" and quality
            else None
        ),
    )
    # The SAME dict object as the closure's own `notes`, not a one-time copy, so a
    # note the closure adds later (e.g. `capacity_floor`, added only once a step
    # actually hits a dry pipe) is visible on `model.notes` too, rather than frozen at the
    # dict copy this builder made at construction time.
    model.notes = hydraulics.notes

    inflow = torch.zeros(graph.n, dtype=F64)
    inflow[manhole_idx] = torch.tensor([m.inflow for m in net.manholes], dtype=F64)
    drivers.update(
        {
            "inflow": inflow,
            "T_water": torch.tensor(18.0, dtype=F64),
            "T_head": torch.tensor(293.15, dtype=F64),
            "T_amb": torch.tensor(283.15, dtype=F64),
            "pH": torch.tensor(7.0, dtype=F64),
            "bod_in": torch.zeros(graph.n, dtype=F64),
            "sulfide_in": torch.zeros(graph.n, dtype=F64),
            # A construction-time constant, not a state: the SLOPE of each manhole's own
            # outgoing pipe, which the sulfide and two-film correlations read.
            "sewer.q_slope": slope_per_manhole,
        }
    )
    state.update(initial_state(model, drivers))
    model.out_pipe = out_pipe
    model.manhole_idx = manhole_idx
    model.pipe_names = [p.name for p in net.pipes]
    # The inputs the hydraulics, lateral-load and H2S closures read themselves, declared
    # for `model.refs.inputs` and `Model.check` (only those this model was built with).
    specs = {
        "inflow": dict(description="lateral inflow at each manhole", unit="m3/s",
                       over="nodes"),
        "bod_in": dict(description="BOD concentration of the lateral inflow",
                       unit="kg/m3", over="nodes"),
        "sulfide_in": dict(description="sulfide concentration of the lateral inflow",
                           unit="kg/m3", over="nodes"),
        "T_water": dict(description="wastewater temperature", unit="degC"),
        "T_head": dict(description="headspace air temperature", unit="K"),
        "T_amb": dict(description="ambient air temperature", unit="K"),
        "pH": dict(description="wastewater pH", unit="-"),
        "sewer.q_slope": dict(description="slope of each manhole's outgoing pipe",
                              unit="m/m", over=manhole_names,
                              ordering="manhole order"),
    }
    model.input_specs = {k: v for k, v in specs.items() if k in drivers}
    model.driver_template = dict(drivers)
    model.inflow_schedule = net.inflows
    return model, state, drivers


def initial_state(
    model: Model, drivers: Drivers | None = None, *, values: Mapping | None = None
) -> State:
    """All-zero state, dispatching on each layer's `quantity` (the app convention).

    A dynamic-wave model (a ``"head"`` layer and a `DynamicWaveHydraulics` closure) gets
    SWMM's initial state instead (cold and dry, or its InitDepth and InitFlow:
    `DynamicWaveHydraulics.initial_state`; `drivers` is optional there and only validated,
    unless the model has a quality layer, whose clean initial state and capacity
    `dynwave.quality_state` adds and which needs them).

    Also builds every closure-carried state key this application knows --
    `"sewer.H"`, the `SewerHydraulics` closure's manhole levels, when it is running with
    `storage=True` -- so that `step(initial_state(model, drivers))` works exactly as
    `SewerHydraulics`'s own `KeyError` message (raised when `"sewer.H"` is missing) already
    promises, rather than requiring the caller to know to add it separately.

    `SewerHydraulics` writes `"<quality layer>.capacity"` every call (the wetted/headspace
    volume), and `Model` requires the step-start state to carry that key
    whenever the driver is supplied (the storage at the state's own time, for the amount
    form of the transport step). When `drivers` is given,
    `model.initial_capacities(state, drivers)` supplies it -- a QUERY call
    (`Model._apply_closures` with no `ctx`), so `SewerHydraulics` evaluates
    its geometry at this all-zero, dry state (`sewer.H = 0` from the block above) WITHOUT
    advancing it; the wetted volumes come out at the documented floor `CAPACITY_FLOOR`
    (`hydraulics.py`), so the first real step conserves the (negligible) initial amount.
    `drivers` is required for that -- there is no drivers-free way to evaluate a closure --
    so its absence is refused by name whenever it would actually be needed (a transport layer
    exists and some closure is a `SewerHydraulics`), rather than only once the missing key is
    discovered deep inside `Model.step`.
    """
    dynamic = [c for c in model.closures if isinstance(c, DynamicWaveHydraulics)]
    if dynamic and any(layer.quantity == "head" for layer in model.potential.values()):
        state = dynamic[0].initial_state(drivers)
        if "air" in model.potential:
            state.update(air_state(model))
        if model.transport:
            if drivers is None:
                raise ValueError(
                    "apps.sewer.initial_state: drivers are required to evaluate the initial "
                    "quality capacity; call initial_state(model, drivers)")
            state.update(quality_state(model, state, drivers))
        return state_from(model, values, base=state) if values else state
    state: State = {}
    for closure in model.closures:
        if "sewer.H" in getattr(closure, "state_keys", ()):
            state["sewer.H"] = torch.zeros(len(closure.manhole_names), dtype=F64)
    for name, layer in model.potential.items():
        if layer.quantity != "pressure":
            raise ValueError(
                f"apps.sewer.initial_state: potential layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds pressure layers only"
            )
        state[f"{name}.phi"] = torch.zeros(model.net.n, dtype=F64)
        state[f"{name}.q"] = torch.zeros(len(layer.cols), dtype=F64)
    for name, layer in model.transport.items():
        if layer.quantity != "concentration":
            raise ValueError(
                f"apps.sewer.initial_state: transport layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds concentration layers only"
            )
        shape = (layer.n_i,) if layer.n_species == 1 else (layer.n_i, layer.n_species)
        state[f"{name}.x"] = torch.zeros(shape, dtype=F64)
    if values:
        state = state_from(model, values, base=state)
    if drivers is not None:
        state.update(model.initial_capacities(state, drivers))
    elif model.transport and any(isinstance(c, SewerHydraulics) for c in model.closures):
        raise ValueError(
            "apps.sewer.initial_state: drivers are required to evaluate the initial "
            "storage; call initial_state(model, drivers)"
        )
    return state


def initial_drivers(model: Model, *, values: Mapping | None = None) -> Drivers:
    """`build_model`'s driver template (copied), then `values` by name (see
    `noodl.refs.drivers_from`), e.g. by outfall and species name:

        initial_drivers(model, values={"water_quality.x_boundary": {"O1": {"sulfide": 0.0}}})
    """
    return refs_initial_drivers(model, values=values)


def sewer_steady(
    model: Model, state: State, drivers: Drivers, *, reaction=None,
    dt: float = 60.0, max_iter: int = 500, tol: float = 1e-12,
) -> State:
    """The explicit fixed point of transport PLUS reactions (the `street_steady` pattern).

    `Model.steady` never applies reactions (framework behaviour, stated at `Model.steady`),
    so a sewer whose sulfide balance has a source term needs its own fixed point: step until
    every transport state stops changing.
    """
    current = dict(state)
    for _ in range(max_iter):
        new = model.step(current, drivers, dt)
        worst = 0.0
        for name in model.transport:
            key = f"{name}.x"
            worst = max(worst, float((new[key] - current[key]).abs().max()))
        current = new
        if worst < tol:
            return current
    raise RuntimeError(
        f"apps.sewer.sewer_steady: no fixed point within {max_iter} steps of {dt} s "
        f"(largest remaining change {worst:.3e} against tol {tol})"
    )
