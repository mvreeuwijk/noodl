"""SWMM 5.2.4 dynamic-wave routing for the sewer application.

`DynamicWaveOptions` holds the `[OPTIONS]` that change the physics. The model is a
`PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m) whose unknowns are the junction
heads: every conduit is a `DynamicWaveConduit`, q = c0 + k (H1 - H2) with the coefficients of
SWMM's momentum update (`dynwave_terms.conduit_terms`) frozen for one pass, and every junction
carries a `JunctionStorage`, SWMM's trapezoidal continuity
``A_s (y - y_start) = 0.5 dt (N_start + N)``. Orifices, weirs and outlets are edges of their
own kinds, `StructureLink`, their flow SWMM's (`structures`) linearised about the previous
pass. `DynamicWaveHydraulics` (a closure) refreshes
the coefficients, surface areas and outfall heads from the previous pass, and
``Model(coupling="iterate")`` repeats the pass until the heads stop changing: the fixed point
is SWMM's own converged trial.

This module must not import `network.py` at module level: `network.py` imports
`DynamicWaveOptions` and `build_dynamic_wave_model` from here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from noodl.apps.sewer.dynwave_terms import (
    DAMPING,
    FUDGE,
    NORMAL_FLOW_LIMITED,
    SURCHARGE_METHODS,
    ConduitGeometry,
    ConduitTerms,
    conduit_terms,
    lengthening_factor,
)
from noodl.apps.sewer.storage import StorageCurves
from noodl.apps.sewer.structures import (
    CIRCULAR,
    DN_CRITICAL,
    RECT_CLOSED,
    RECT_OPEN,
    TINY,
    TRAPEZOIDAL,
    TRIANGULAR,
    UP_CRITICAL,
    Opening,
    OrificeSet,
    OutletSet,
    WeirSet,
    orifice_flow,
    outlet_dqdh,
    outlet_flow,
    weir_flow,
)
from noodl.apps.sewer.xsect_tables import (
    LCF,
    QCF,
    area_of_depth,
    critical_depth,
    normal_depth_ft,
)
from noodl.elements.base import Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model, StepContext
from noodl.nodesources import NodeSource
from noodl.topology import Network

if TYPE_CHECKING:
    from noodl.apps.sewer.network import SewerNetwork

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: SWMM's dry-node flow cap (`dwflow.c:273-276`): 1e-4 cfs, in m3/s.
FUDGE_FLOW = 1.0e-4 * QCF
#: Bed stiffening factor of `JunctionStorage` (design D3): below its invert a junction's
#: storage slope gains ``(2 A_bed/dt) K_BED``, standing in for SWMM's ``yNew < 0 -> 0``.
K_BED = 1.0e6
#: Stiffness factor of an EXTRAN junction's hold band at its crown (design E1, the tie rule
#: in `JunctionStorage`): the band rises with ``(2 A_bed/dt) K_HOLD``. It trades the band's
#: width ``|w_c| dt / (2 A_bed K_HOLD)`` (0.6 mm at w_c = 0.26 m3/s, SWMM's default surface
#: area floor and a 5 s step) against the head solve's round-off floor on it, ``(2 A_bed/dt)
#: K_HOLD eps |H| / 2`` (3e-13 m3/s at 12 m), which must stay below the 1e-12 m3/s a
#: surcharged run's parity needs; the bed's K_BED would put that floor at 6e-10 m3/s.
K_HOLD = 1.0e3
#: The band is also at least `K_HOLD_STORAGE` times the junction's own storage slope
#: ``2 A_s/dt``, so that above the crown it always rises faster than the storage line it
#: starts from (a large A_s, e.g. a trunk junction or a large user MIN_SURFAREA, would
#: otherwise outgrow ``K_HOLD A_bed``). Its round-off floor then scales with A_s: 6e-12
#: m3/s at A_s = 1200 m2, 5 s and 12 m, above a 1e-12 m3/s head-solve atol there.
K_HOLD_STORAGE = 10.0
#: A surcharged junction's anchor slope floor (design E1): ``K_ANCHOR (2 A_s/dt)`` (m2/s) is
#: added to the conduit conveyance its pass lags (`JunctionStorage`), so the junction's row
#: of the head solve stays nonsingular even where every attached conduit's dependence on its
#: head is lagged or limited. The anchor vanishes at the fixed point, so its size only
#: shapes the iteration: 1e-6 of the storage slope is far below any full conduit's k. Where
#: the floor ALONE grounds the junction (every attached conduit pinned at the far end with
#: no k left, dry or normal-flow limited) nothing bounds the pass's step but the floor:
#: ``dH = dQ / (K_ANCHOR 2 A_s/dt)``, a large jump. Such a pass is still a valid iterate (the
#: fixed point is unchanged), its height is capped by the flood wall at y_max, and if the
#: jumps do not settle the step is refused by name for non-convergence. The
#: common pinned case (an outflow conduit dropping to a DN_CRITICAL end) keeps its lagged k
#: in the anchor, so the step is a Newton step, not the floor's.
K_ANCHOR = 1.0e-6
#: Cap on the slope of the normal-flow slope switch's wall (`DynamicWaveConduit`), as a
#: multiple of the smaller end junction's grounding (its storage slope ``2 A_s/dt`` plus its
#: conduits' conveyance, m2/s). The wall's fixed points do not depend on its slope. A wall
#: steeper than the junction it acts on makes the conduit's flow an S-curve in the head
#: (flat at q_norm, the wall, the momentum line) that the damped Newton head solve cycles
#: on: at 10 x the grounding the tree's drain-down failed the head solve (50 iterations at t
#: = 520 s); at 1 x the slopes a Newton step sees differ by at most 2, which its 0.75
#: damping contracts, and a conduit between two junctions settles on its switch by about
#: half per pass. Its round-off floor, ``S eps |H| / 2``, is then that of the junction's own
#: storage slope. The drop-end area switch's wall (`JunctionStorage`) uses the same factor on
#: its own junction's grounding above the threshold, for the same reason.
K_SWITCH = 1.0
#: Newton's absolute residual tolerance for the head solve (m3/s): a junction balance error
#: of `SOLVE_ATOL` dt m3 at most, far below the head tolerance times any surface area. It is
#: raised to the bed's round-off floor where that is larger (`solve_atol`).
SOLVE_ATOL = 1.0e-12
#: `solve_atol`'s margin over the bed's round-off floor (see there).
_BED_FLOOR_MARGIN = 4.0
#: SWMM's default junction surface-area floor (`dynwave.c:62`), ft2.
DEFAULT_MIN_SURFACE_AREA_FT2 = 12.566
#: SI area per SWMM ft2 (design D2): makes ``A_s dy = 0.5 dt N`` exact in m, m3/s.
_AREA_SI = QCF / LCF
#: SWMM's link velocity is 0 at or below this link depth (`link.c:834`), ft.
_VELOCITY_MIN_DEPTH = 0.01
#: SWMM's shortest time step, MINTIMESTEP (`dynwave.c:60`), s.
MIN_TIME_STEP = 0.001
_OUTFALL_CODES = {"free": 0, "normal": 1, "fixed": 2, "tidal": 3, "timeseries": 4}
#: Outfall kinds whose stage varies in time: it is the ``"outfall_stage"`` driver's.
_STAGED_KINDS = ("tidal", "timeseries")
MSEC_PER_DAY = 86400.0 * 1000.0                                          # consts.h MSECperDAY


def tidal_stage(tide: tuple[tuple[float, float], ...], t: float) -> float:
    """A TIDAL outfall's stage elevation (m) `t` s into the run (`node.c::
    outfall_setOutletDepth`, node.c:1445-1452): the curve `tide` of ``(hour, stage)`` pairs
    is read at ``x = x_first + 24 frac(t / 1 day)`` (the elapsed time, not the clock time:
    the simulation's start is not added, so the tide begins at the curve's first hour and
    repeats every 24 h), by `table.c::table_lookup`: the first stage at or below the first
    hour, linear between entries, the last stage beyond the last hour (the curve is not
    wrapped, so one that stops short of 24 h holds its last stage)."""
    days = 1000.0 * t / MSEC_PER_DAY                                     # node.c:1449
    x = tide[0][0] + (days - math.floor(days)) * 24.0                    # node.c:1450
    x1, y1 = tide[0]
    if x <= x1:                                                          # table.c:420
        return y1
    for x2, y2 in tide[1:]:
        if x <= x2:                                                      # table.c:421
            return _table_interpolate(x, x1, y1, x2, y2)
        x1, y1 = x2, y2
    return y1                                                            # table.c:426


def _table_interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """`table.c::table_interpolate` (table.c:51-63)."""
    dx = x2 - x1
    if abs(dx) < 1.0e-20:
        return (y1 + y2) / 2.0
    return y1 + (x - x1) * (y2 - y1) / dx
#: The flow regulators' edge kinds in the ``"water"`` layer, in the order their columns
#: follow the pipes' in ``"water.q"`` (only the kinds a network has are present).
STRUCTURE_KINDS = ("orifice", "weir", "outlet")
_WEIR_TYPES = {"transverse": 0, "sideflow": 1, "v_notch": 2, "trapezoidal": 3}
_WEIR_SHAPES = {"transverse": RECT_OPEN, "sideflow": RECT_OPEN, "v_notch": TRIANGULAR,
                "trapezoidal": TRAPEZOIDAL}


@dataclass(frozen=True)
class DynamicWaveOptions:
    """SWMM 5.2.4's dynamic-wave [OPTIONS] that change the physics, with SWMM's defaults
    (`project.c:841-880`). `head_tolerance` and `max_trials` are SWMM's own iteration
    controls: recorded so a file round-trips, never used -- noodl solves the same
    equations to `build_model`'s own `iterate_tol` (docs: "Dynamic wave").

    `routing_step` is the fixed routing step (s); `inertial_damping` is ``"none"``,
    ``"partial"`` or ``"full"``; `normal_flow_limited` is ``"slope"``, ``"froude"``,
    ``"both"`` or ``"none"``; `surcharge_method` is ``"extran"`` (end depths capped at the
    crown, a junction above its crown holds continuity without storage) or ``"slot"``
    (Preissmann slot);
    `min_surface_area` is the junction surface-area floor (m2), ``None`` for SWMM's
    12.566 ft2; `head_tolerance` is SWMM's head convergence tolerance (m); `max_trials` is
    SWMM's trial limit (-).

    `variable_step` is SWMM's VARIABLE_STEP, the Courant factor of the variable routing step
    (-, 0 to 2): 0 routes with the fixed `routing_step`, above 0 each step is the smallest
    of `routing_step`, the conduit Courant step and the junction depth-change step
    (`routing_step_size`; SWMM's own default is 0.75, this dataclass's is 0, so a fixed
    step is the default). `minimum_step` is SWMM's MINIMUM_STEP (s), the shortest variable
    step and the very first one.

    `lengthening_step` is SWMM's LENGTHENING_STEP (s, >= 0, 0 for none): a conduit shorter
    than the Courant length ``(sqrt(g yFull) + vFull) t`` is lengthened to it for the
    momentum terms, with ``t`` the smaller of `lengthening_step` and the routing step
    (`conduit_getLengthFactor`, link.c:1217-1254; the volume keeps the true length).
    SWMM takes the routing step before it is lowered to WET_STEP (`project.c:238-250`
    validates the conduits first), so `lengthening_routing_step` (s) is the ROUTING_STEP of
    the file where that differs from `routing_step`; ``None`` means `routing_step`.
    """

    routing_step: float = 20.0
    inertial_damping: str = "partial"
    normal_flow_limited: str = "both"
    surcharge_method: str = "extran"
    min_surface_area: float | None = None
    head_tolerance: float | None = None
    max_trials: int | None = None
    variable_step: float = 0.0
    minimum_step: float = 0.5
    lengthening_step: float = 0.0
    lengthening_routing_step: float | None = None

    @property
    def min_route_step(self) -> float:
        """SWMM's `MinRouteStep` (s): `minimum_step` clipped to [MINTIMESTEP, routing_step]
        (`dynwave.c:184-185`)."""
        step = min(self.minimum_step, self.routing_step)                      # dynwave.c:184
        return max(step, MIN_TIME_STEP)                                       # dynwave.c:185

    def validate(self) -> None:
        """Refuse an unknown keyword or a non-positive step or area, BY NAME."""
        if not self.routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: routing_step must be > 0 s, got {self.routing_step!r}"
            )
        for name, allowed in (
            ("inertial_damping", DAMPING),
            ("normal_flow_limited", NORMAL_FLOW_LIMITED),
            ("surcharge_method", SURCHARGE_METHODS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"DynamicWaveOptions: {name} must be one of {allowed}, got "
                    f"{getattr(self, name)!r}"
                )
        if not 0.0 <= self.variable_step <= 2.0:                              # project.c:698-703
            raise ValueError(
                f"DynamicWaveOptions: variable_step must be in [0, 2], got "
                f"{self.variable_step!r}")
        if not self.lengthening_step >= 0.0:
            raise ValueError(
                f"DynamicWaveOptions: lengthening_step must be >= 0 s, got "
                f"{self.lengthening_step!r}")
        if self.lengthening_routing_step is not None and not self.lengthening_routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: lengthening_routing_step must be > 0 s or None, got "
                f"{self.lengthening_routing_step!r}")
        if not self.minimum_step >= 0.0:
            raise ValueError(
                f"DynamicWaveOptions: minimum_step must be >= 0 s, got {self.minimum_step!r}")
        if self.min_surface_area is not None and not self.min_surface_area > 0:
            raise ValueError("DynamicWaveOptions: min_surface_area must be > 0 m2")


class DynamicWaveConduit(Element):
    """SWMM's dynamic-wave conduit (`dwflow.c`) with its coefficients frozen for one pass.

    ``q = c0 + k dp`` (m3/s, dp = H1 - H2 in m), then ``min(q, q_norm)`` where SWMM's
    normal-flow check passed (`dwflow.c:255-258`), then the user flow limit, the flap gate and
    the dry-node caps in SWMM's order (below; `dwflow.c:270-281`): at most +`FUDGE_FLOW`
    out of a dry upstream node, at least -`FUDGE_FLOW` out of a dry downstream node (each
    times the conduit's barrels).

    The normal-flow slope switch (the tie rule). The check's flag is lagged (design D4):
    evaluated by the closure on the fed heads and flow. Where it is SWMM's slope test ``y1 <
    y2`` alone, with both end depths inside their clamps, the test is ``dp < dz`` (dz = z1 -
    z2, the conduit-end inverts), a switch on this element's own argument, and it is applied
    within the pass instead, as a continuous, non-decreasing curve::

        q = min(lin, max(q_norm, q_last + S (dp - dz)))

    with ``lin = c0 + k dp``, ``q_last`` the fed flow and ``S > 0`` a wall slope. At a
    fixed point (``q = q_last``) this is SWMM's unlimited flow ``lin`` where ``dp > dz``,
    its limited ``min(lin, q_norm)`` where ``dp < dz`` -- every SWMM fixed point is one of
    this curve, unchanged, whatever S -- or a point ON the switching surface ``dp = dz``
    (y1 = y2) with q between q_norm and lin, which is a fixed point here only. That last
    case is where SWMM has none: limited, the solution lies above the switch; unlimited,
    below it. A lagged flag then chatters forever, and SWMM's own trials alternate around
    the surface without converging (`tests/apps/sewer/test_dynwave.py::
    test_ramp_down_converges`). The surface is where SWMM's relaxed trials straddle; noodl
    puts the solution exactly on it, deterministically. The cost against SWMM is confined
    to those steps, where SWMM has no converged answer to compare with: its last trial
    lies within its own oscillation of the surface (~6e-4 m of J4's depth on the tree's
    drain-down). S only shapes the passes (`DynamicWaveHydraulics._switch_slope`).

    Reads the per-pipe drivers ``"water.flow_offset"`` c0 (m3/s), ``"water.conveyance"`` k
    (m2/s), ``"water.normal_flow"`` q_norm (m3/s, +inf where neither the limit nor the
    switch applies), the 0/1 flags ``"water.limited"`` (the lagged limit, where no switch
    applies), ``"water.slope_switch"``, ``"water.dry_upstream"``,
    ``"water.dry_downstream"``, and the switch's ``"water.switch_drop"`` dz (m),
    ``"water.switch_slope"`` S (m2/s) and ``"water.flow_last"`` q_last (m3/s), all written
    by `DynamicWaveHydraulics`.

    The element limits that follow the normal-flow limit, in SWMM's order (`dwflow.c:269-281`;
    SWMM's under-relaxation, :259-267, is not modelled): the user flow limit ``|q| <=``
    ``"water.max_flow"`` (m3/s, +inf for none, the default; `dwflow.c:269-273`), the flap gate
    zeroing flow against it (`link.c::link_setFlapGate`: ``"water.flap_block_reverse"`` 1 zeroes
    ``q < 0``, ``"water.flap_block_forward"`` 1 zeroes ``q > 0``; `dwflow.c:276`), then the
    dry-node caps at ``"water.dry_flow_cap"`` (m3/s, `FUDGE_FLOW` times the barrels;
    `dwflow.c:280-281`). Each is a
    clamp of ``q`` against a constant, so the flow stays continuous and non-decreasing in ``dp``
    (the head solve's Newton step needs both) with ``dflow`` 0 wherever one acts.
    """

    def __init__(self, kind: str = "pipe") -> None:
        super().__init__(kind)

    def _parts(self, dp: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        """The momentum line, the flow and the switch's wall (NaN where no wall acts)."""
        lin = drivers["water.flow_offset"] + drivers["water.conveyance"] * dp
        q_norm = drivers["water.normal_flow"]
        limited = drivers["water.limited"] > 0.5
        switch = drivers["water.slope_switch"] > 0.5
        wall = (drivers["water.flow_last"]
                + drivers["water.switch_slope"] * (dp - drivers["water.switch_drop"]))
        slid = torch.minimum(lin, torch.maximum(q_norm, wall))
        q = torch.where(limited, torch.minimum(lin, q_norm),
                        torch.where(switch, slid, lin))                    # dwflow.c:255-258
        inf = torch.full_like(q, torch.inf)
        zero = torch.zeros_like(q)
        limit = drivers.get("water.max_flow", inf)
        q = torch.minimum(torch.maximum(q, -limit), limit)                 # dwflow.c:269-273
        block_reverse = drivers.get("water.flap_block_reverse", zero) > 0.5
        block_forward = drivers.get("water.flap_block_forward", zero) > 0.5
        q = torch.minimum(torch.maximum(q, torch.where(block_reverse, zero, -inf)),
                          torch.where(block_forward, zero, inf))          # dwflow.c:276
        cap = drivers.get("water.dry_flow_cap", torch.full_like(q, FUDGE_FLOW))
        up = drivers["water.dry_upstream"] > 0.5
        dn = drivers["water.dry_downstream"] > 0.5
        q = torch.where(up, torch.minimum(q, cap), q)                      # dwflow.c:280
        q = torch.where(dn, torch.maximum(q, -cap), q)                     # dwflow.c:281
        on_wall = switch & ~limited
        return lin, q, torch.where(on_wall, wall, torch.full_like(wall, torch.nan))

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Conduit flow (m3/s, positive from the pipe's first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): k where the flow is the momentum line, the wall slope S where it
        is on the switch's wall, else 0 (every other limit is a constant, so ``q == lin``
        and ``q == wall`` are exact tests)."""
        lin, q, wall = self._parts(dp, drivers)
        k = drivers["water.conveyance"].expand_as(q)
        s = drivers["water.switch_slope"].expand_as(q)
        zero = torch.zeros_like(k)
        return torch.where(q == lin, k, torch.where(q == wall, s, zero))


class StructureLink(Element):
    """A flow regulator (orifice, weir or outlet; design E5) with its flow linearised about
    the previous pass: ``q = c0 + g (dp - dp_fed)`` (m3/s, dp = H1 - H2 in m), then the flap
    gate's clamp.

    ``c0`` is SWMM's own regulator flow at the previous pass's heads (`link.c`
    ``orifice_getInflow`` / ``weir_getInflow`` / ``outlet_getInflow``,
    `noodl.apps.sewer.structures`), ``dp_fed`` that pass's head difference, so at a fixed
    point (dp = dp_fed) the flow is SWMM's exactly, whatever ``g``. ``g >= 0`` (m2/s) is
    SWMM's ``dqdh`` for orifices and weirs, a central difference in the upstream head for
    outlets: it only makes each pass a Newton-like step in the device's heads. The flow is
    non-decreasing in dp (``g >= 0``, the clamps are constants), as the head solve needs.

    Reads the per-device drivers ``"water.<kind>.flow_offset"`` c0 (m3/s),
    ``"water.<kind>.slope"`` g (m2/s), ``"water.<kind>.dp_fed"`` (m) and the 0/1 flags
    ``"water.<kind>.flap_block_reverse"`` (its own flap gate, or a gated outfall at its
    second node: ``q >= 0``) and ``"water.<kind>.flap_block_forward"`` (a gated outfall at
    its first node: ``q <= 0``; `link.c::link_setFlapGate`), all written by
    `DynamicWaveHydraulics`. At a fixed point the clamps never act: SWMM's flow already
    respects the gate."""

    def __init__(self, kind: str) -> None:
        if kind not in STRUCTURE_KINDS:
            raise ValueError(f"StructureLink: kind must be one of {STRUCTURE_KINDS}, got "
                             f"{kind!r}")
        super().__init__(kind)
        self.prefix = f"water.{kind}."

    def _parts(self, dp: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        p = self.prefix
        lin = drivers[p + "flow_offset"] + drivers[p + "slope"] * (dp - drivers[p + "dp_fed"])
        zero = torch.zeros_like(lin)
        inf = torch.full_like(lin, torch.inf)
        reverse = drivers.get(p + "flap_block_reverse", zero) > 0.5
        forward = drivers.get(p + "flap_block_forward", zero) > 0.5
        q = torch.minimum(torch.maximum(lin, torch.where(reverse, zero, -inf)),
                          torch.where(forward, zero, inf))                  # link.c:643-670
        return lin, q

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Regulator flow (m3/s, positive from its first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): g where the gate's clamp does not act, else 0."""
        lin, q = self._parts(dp, drivers)
        g = drivers[self.prefix + "slope"].expand_as(q)
        return torch.where(q == lin, g, torch.zeros_like(g))


@dataclass(frozen=True)
class _Structures:
    """One kind of flow regulator in the closure: ``names``; end node indices ``u``, ``v``;
    ``offset1`` (ft, the crest above node 1's invert after SWMM raises a crest below node
    2's invert to it, link.c:425-433) and ``offset2`` (ft, the crest height as given, which
    SWMM keeps at node 2, link.c:366-388); ``y_full`` (ft, the opening height; an outlet's
    is SWMM's DUMMY ``TINY``); ``params`` (`OrificeSet` / `WeirSet` / `OutletSet`);
    ``block_forward``/``block_reverse`` (bool, `structures._gate`); ``extends`` (bool: the
    device extends node 1's full depth to its crown, all but bottom orifices,
    link.c:448-455)."""

    kind: str
    names: list
    u: Tensor
    v: Tensor
    offset1: Tensor
    offset2: Tensor
    y_full: Tensor
    params: object
    block_forward: Tensor
    block_reverse: Tensor
    extends: Tensor


def _build_structures(kind: str, devices, nodes: list, invert: Tensor, gated_outfalls: set,
                      route_step: float) -> _Structures:
    """The closure's view of `devices` of one `kind` (`_Structures`); `route_step` (s) sets
    the orifice and weir equivalent lengths."""
    u = torch.tensor([nodes.index(s.u) for s in devices], dtype=torch.long)
    v = torch.tensor([nodes.index(s.v) for s in devices], dtype=torch.long)
    crest = torch.tensor([s.crest_height for s in devices], dtype=F64)
    inv1, inv2 = invert[u], invert[v]
    # link.c:425-433: under DW a crest below node 2's invert is raised to it (offset1 only)
    offset1 = torch.where(inv1 + crest < inv2, inv2 - inv1, crest) / LCF
    own_gate = torch.tensor([s.flap_gate for s in devices], dtype=torch.bool)
    block_reverse = own_gate | torch.tensor([s.v in gated_outfalls for s in devices],
                                            dtype=torch.bool)
    block_forward = torch.tensor([s.u in gated_outfalls for s in devices], dtype=torch.bool)
    extends = torch.ones(len(devices), dtype=torch.bool)
    if kind == "orifice":
        opening = Opening.from_si(
            [CIRCULAR if s.shape == "circular" else RECT_CLOSED for s in devices],
            [s.height for s in devices],
            [s.height if s.width is None else s.width for s in devices])
        bottom = torch.tensor([s.orientation == "bottom" for s in devices], dtype=torch.bool)
        params = OrificeSet.build(opening, bottom,
                                  [s.discharge_coefficient for s in devices], own_gate,
                                  route_step)
        extends = ~bottom                                                    # link.c:442-444
        y_full = opening.y_full
    elif kind == "weir":
        opening = Opening.from_si([_WEIR_SHAPES[s.kind] for s in devices],
                                  [s.height for s in devices], [s.width for s in devices],
                                  [tuple(s.side_slopes) for s in devices])
        params = WeirSet.build(opening, [_WEIR_TYPES[s.kind] for s in devices],
                               [s.discharge_coefficient for s in devices],
                               [s.end_discharge_coefficient for s in devices],
                               [s.end_contractions for s in devices],
                               [s.can_surcharge for s in devices], own_gate)
        y_full = opening.y_full
    else:
        params = OutletSet.build([s.rating for s in devices],
                                 [s.head_difference for s in devices], own_gate)
        y_full = torch.full((len(devices),), TINY, dtype=F64)                # xsect.c:233
    return _Structures(kind, [s.name for s in devices], u, v, offset1, crest / LCF, y_full,
                       params, block_forward, block_reverse, extends)


class JunctionStorage(NodeSource):
    """SWMM's junction continuity as a node withdrawal: trapezoidal storage over the step,
    with a stiff bed below the invert (design D3), or no storage at all where the junction
    is EXTRAN-surcharged (design E1), capped by flooding at the junction's top (design E3).

    ``w(H) = (2 A_s/dt)(H - H_start) - N_start + (2 A_bed/dt) K_BED min(H - z_bed, 0)``
    (m3/s, positive out of the network), so that with the conduit flows and the lateral
    inflow the node balance reads ``A_s (H - H_start) = 0.5 dt (N_start + N)``
    (`dynwave.c:694-711`). The bed term stands in for SWMM's ``yNew < 0 -> 0``
    (`dynwave.c:743`); the slope is at least ``2 A_s/dt > 0``, so every junction grounds
    itself. The bed stiffness uses the fixed `bed_area` A_bed (m2; the model passes SWMM's
    surface-area floor, which no A_s is below), not A_s, so that the head solve's round-off
    floor on the bed is known before the step (`solve_atol`). Relative to the junction's own
    storage slope the bed is ``K_BED A_bed / A_s`` stiffer: 1e6 at the floor, but only ~1e3
    where A_s is a thousand times a tiny user MIN_SURFAREA.

    EXTRAN surcharge (design E1; `crown` given, the junction crown heads in m). Above its
    crown a junction stores nothing: SWMM's surcharged update ``y_new = y_last + corr
    dQ/denom`` (`dynwave.c:716-741`) is at its fixed point exactly when the net inflow dQ is
    0. With ``w_c = (2 A_s/dt)(H_crown - H_start) - N_start`` the storage withdrawal at the
    crown, ``stiff = (2/dt) max(K_HOLD A_bed, K_HOLD_STORAGE A_s)`` and the band
    ``b(H) = w_c + stiff (H - H_crown)``:

    - a SURCHARGED junction (driver ``"water.surcharged"`` 1, set by `surcharged` from the
      fed head ``H_last``: above the crown and ``b(H_last) > 0``, i.e. above the band where
      ``w_c < 0``) withdraws the anchor ``s (H - H_last)``, zero at the
      fixed point, with ``s`` the driver ``"water.surcharge_slope"`` (m2/s): the
      conveyance of the attached conduits that the pass lags into their c0 (a conduit whose
      other end is pinned), so that the pass is a Newton step in this junction's head, plus
      a floor `K_ANCHOR` (2 A_s/dt) that keeps the head solve nonsingular;
    - any other junction withdraws ``storage(H)`` at and below its crown and ``max(storage(H),
      b(H))`` above it (the band, which is steeper): continuous at the crown, a convex curve
      (one Newton-safe solution per pass). The band never acts below the crown.

    The fixed points are SWMM's: a free-surface solution below the crown; a surcharged one
    (net inflow 0) above it -- the flag is lagged on the fed head as SWMM's test
    ``y_last > y_crown`` (`dynwave.c:688`) is on the previous trial, and a surcharged
    solution below the crown drops it on the next pass, as SWMM's floor ``y_new < y_crown ->
    y_crown - FUDGE`` (`dynwave.c:735-736`) does. Where SWMM has NO fixed point (the storage
    equation ends above the crown, but at the crown the junction already loses water, so the
    surcharged solution lies below it) SWMM's trials alternate between its two branches and
    a flag lagged on the depth alone chatters forever. Here the band takes that case: the
    junction is held at its crown (within the band, ``|w_c|/stiff``, below a millimetre),
    releasing what the balance needs, ``w`` between ``w_c`` and 0 -- the deterministic tie
    rule. The band also moves a surcharged solution that lies within ``|w_c|/stiff`` above
    the crown onto the band; SWMM's surcharged depth lies ``N(crown)/sum(k)`` above it, so
    that needs a net inflow at the crown below ``|w_c| sum(k)/stiff``, a thin sliver.

    Flooding (design E3; `top` given, the junction heads at y_max = full depth + SurDepth,
    m). SWMM holds a junction whose update would end above y_max at y_max and lets the
    excess leave the system (`dynwave.c:743-795`, no ponding). With ``w`` the withdrawal
    above, ``S`` the wall slope (the band's ``stiff``, at least the anchor's ``s`` where
    surcharged) and the lagged excess ``E = max(N_last - w(top), 0)`` (`flood_excess`;
    driver ``"water.flood_excess"``, m3/s, from the fed net inflow N_last), every junction
    withdraws ``max(w(H), w(top) + E + S (H - top))``:

    - the wall caps the junction at the top within the pass (it rises when the head reaches
      y_max, no lag), and the max of two increasing lines stays convex (Newton-safe);
    - at a flooded fixed point the fed net inflow is the solved one, ``N_last = N``, so the
      balance ``w = N`` puts H EXACTLY at the top whatever S is, and the withdrawal ``N`` is
      the flow over the top; S only sets how fast the passes contract, by ``sum(k)/(S +
      sum(k))`` with ``sum(k)`` the junction's conduit conveyance. That is why S is the hold
      band's stiffness (a round-off floor of ~3e-13 m3/s at 12 m, inside a 1e-12 m3/s head
      solve) and not a 1e6 penalty, whose round-off floor would be ~6e-10 m3/s and whose
      offset ``N/S`` would leave the head ~1e-7 m above y_max;
    - E > 0 exactly where SWMM floods: SWMM's update ends above y_max iff the net inflow at
      the top exceeds what the non-flooded form would withdraw there (the storage form's
      ``0.5 dt (N_start + N) > A_s (y_max - y_start)``, the surcharged form's ``dQ > 0``).
      E falls to 0 when that excess would be negative, and the junction leaves the top on
      the next pass: there is no flag to chatter, only a lagged excess that contracts.
      Below the top the wall reaches down only by ``E/(S - slope)``, a transient within the
      passes.

    The drop-end area switch (the tie rule of `DynamicWaveHydraulics`; drivers
    ``"water.area_switch_head"`` H* (m, +inf where none applies), ``"water.area_below"`` and
    ``"water.area_above"`` (m2), ``"water.area_switch_slope"`` S (m2/s) and the fed net
    inflow ``"water.net_inflow_last"`` N_last, all optional). Where H* is finite the
    storage term is, within the pass, ``max(lo(H), min(hi(H), N_last + S (H - H*)))`` with
    lo/hi the storage lines with the area below/above the threshold: continuous and
    non-decreasing. At a fixed point (withdrawal = N = N_last) it is the lower-area storage
    below H* and the upper-area storage above it -- SWMM's own fixed points, unchanged --
    or the threshold itself, H = H*, with the withdrawal inside the jump: the solution where
    SWMM has none.

    Reads the per-junction drivers ``"water.storage_area"`` A_s (m2, design D2's SI area),
    ``"water.head_start"`` (m), ``"water.net_inflow_start"`` (m3/s) and
    ``"water.storage_rate"`` 1/dt (1/s); with `crown` also ``"water.surcharged"`` (0/1)
    and ``"water.surcharge_slope"`` (m2/s), and `surcharged` reads ``"water.head_last"``
    (m); with `top` also ``"water.flood_excess"`` (m3/s), and `flood_excess` reads
    ``"water.net_inflow_last"`` (m3/s). `bed` holds the junction inverts (m); without
    `crown` (SLOT) no junction surcharges, without `top` none floods. `surchargeable`
    (bool per junction, default all) marks the junctions that can surcharge at all: a SWMM
    storage unit only with a SurDepth, and then above its full depth, which is its `crown`
    here (`dynwave.c:683-687`); a plain junction only with a positive crown depth
    (`dynwave.c:690`, ``yCrown > 0``; `DynamicWaveHydraulics` passes both rules). Where it is
    False the band and the flag never act.
    """

    def __init__(self, nodes: Tensor, bed: Tensor, bed_area: float,
                 crown: Tensor | None = None, top: Tensor | None = None,
                 surchargeable: Tensor | None = None) -> None:
        super().__init__(nodes)
        self.register_buffer("bed", torch.as_tensor(bed, dtype=F64))
        self.bed_area = float(bed_area)
        self.extran = crown is not None
        if self.extran:
            self.register_buffer("crown", torch.as_tensor(crown, dtype=F64))
            mask = (torch.ones_like(self.crown, dtype=torch.bool) if surchargeable is None
                    else torch.as_tensor(surchargeable, dtype=torch.bool))
            self.register_buffer("surchargeable", mask)
        self.floods = top is not None
        if self.floods:
            self.register_buffer("top", torch.as_tensor(top, dtype=F64))

    def _storage(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        """Phase A's storage withdrawal (m3/s) and slope (m2/s) at `phi`, and the band's
        stiffness (m2/s)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        area = drivers["water.storage_area"]
        bed = rate2 * self.bed_area * K_BED
        dry = torch.clamp(phi - self.bed, max=0.0)
        rise = phi - drivers["water.head_start"]
        n_start = drivers["water.net_inflow_start"]
        storage = rate2 * area * rise - n_start                              # dynwave.c:694-711
        slope = (rate2 * area).expand_as(storage)
        if "water.area_switch_head" in drivers:
            storage, slope = self._area_switch(phi, rise, storage, slope, drivers)
        storage = storage + bed * dry
        slope = slope + bed * (phi < self.bed).to(phi.dtype)
        return storage, slope, self._stiff(drivers)

    @staticmethod
    def _area_switch(phi: Tensor, rise: Tensor, storage: Tensor, slope: Tensor,
                     drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The drop-end area switch within the pass (class docstring): ``max(lo, min(hi,
        N_last + S (H - H*)))`` where a threshold applies (finite ``H*``), the lagged storage
        and slope elsewhere."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        h_star = drivers["water.area_switch_head"]
        on = torch.isfinite(h_star)
        h_safe = torch.where(on, h_star, phi)
        n_start = drivers["water.net_inflow_start"]
        lo = rate2 * drivers["water.area_below"] * rise - n_start
        hi = rate2 * drivers["water.area_above"] * rise - n_start
        s = drivers["water.area_switch_slope"]
        wall = drivers["water.net_inflow_last"] + s * (phi - h_safe)
        capped = torch.minimum(hi, wall)
        switched = torch.maximum(lo, capped)
        slope_sw = torch.where(
            switched == lo, (rate2 * drivers["water.area_below"]).expand_as(lo),
            torch.where(capped == hi, (rate2 * drivers["water.area_above"]).expand_as(hi),
                        s.expand_as(wall)))
        return torch.where(on, switched, storage), torch.where(on, slope_sw, slope)

    def _stiff(self, drivers: Mapping) -> Tensor:
        """The band's slope (m2/s, per junction): ``(2/dt) max(K_HOLD A_bed,
        K_HOLD_STORAGE A_s)``."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        return rate2 * torch.clamp(K_HOLD_STORAGE * drivers["water.storage_area"],
                                   min=K_HOLD * self.bed_area)

    def _band(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The band ``w_c + stiff (H - H_crown)`` (m3/s) at `phi` (m)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        w_c = (rate2 * drivers["water.storage_area"] * (self.crown - drivers["water.head_start"])
               - drivers["water.net_inflow_start"])
        return w_c + self._stiff(drivers) * (phi - self.crown)

    def surcharged(self, drivers: Mapping) -> Tensor:
        """The EXTRAN surcharge flag (0/1, float64) from the fed heads
        ``drivers["water.head_last"]`` (m): the band positive there (class docstring). With
        a `top` the fed head is taken at most at the top, as SWMM's previous-trial depth
        never exceeds y_max (`dynwave.c:751-754`): a junction whose top IS its crown
        (MaxDepth 0, SurDepth 0) never surcharges."""
        head_last = drivers["water.head_last"]
        if not self.extran:
            return torch.zeros_like(head_last)
        if self.floods:
            head_last = torch.minimum(head_last, self.top)
        return ((head_last > self.crown) & self.surchargeable
                & (self._band(head_last, drivers) > 0.0)).to(F64)        # dynwave.c:683-691

    def _free(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The non-flooded withdrawal (m3/s) and its slope (m2/s) at `phi` (m): storage,
        the hold band above the crown, or the surcharged anchor (class docstring)."""
        storage, slope, stiff = self._storage(phi, drivers)
        if not self.extran:
            return storage, slope
        band = self._band(phi, drivers)
        above = (phi > self.crown) & self.surchargeable
        free = torch.where(above, torch.maximum(storage, band), storage)
        slope = torch.where(above & (band > storage), stiff.expand_as(band), slope)
        surcharged = drivers["water.surcharged"] > 0.5
        anchor_slope = drivers["water.surcharge_slope"]
        anchor = anchor_slope * (phi - drivers["water.head_last"])
        return (torch.where(surcharged, anchor, free),                      # dynwave.c:716
                torch.where(surcharged, anchor_slope.expand_as(slope), slope))

    def _wall_slope(self, drivers: Mapping) -> Tensor:
        """The flood wall's slope (m2/s): the hold band's stiffness, and at least the
        anchor's slope where surcharged, so that above the top the wall always rises
        faster than the withdrawal it caps."""
        stiff = self._stiff(drivers)
        if not self.extran:
            return stiff
        return torch.where(drivers["water.surcharged"] > 0.5,
                           torch.maximum(stiff, drivers["water.surcharge_slope"]), stiff)

    def flood_excess(self, drivers: Mapping) -> Tensor:
        """The lagged flood excess E (m3/s, >= 0): the fed net inflow
        ``drivers["water.net_inflow_last"]`` over the non-flooded withdrawal at the top,
        ``max(N_last - w(top), 0)``; > 0 exactly where SWMM's update would end above y_max
        (`dynwave.c:752`), i.e. where the junction floods."""
        top = self.top.expand(drivers["water.net_inflow_last"].shape)
        w_top, _ = self._free(top, drivers)
        return torch.clamp(drivers["water.net_inflow_last"] - w_top, min=0.0)

    def _wall(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The flood wall ``w(top) + E + S (H - top)`` (m3/s) at `phi` (m)."""
        w_top, _ = self._free(self.top.expand(phi.shape), drivers)
        return (w_top + drivers["water.flood_excess"]
                + self._wall_slope(drivers) * (phi - self.top))

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Withdrawal (m3/s) at junction heads `phi_nodes` (m)."""
        free, _ = self._free(phi_nodes, drivers)
        if not self.floods:
            return free
        return torch.maximum(free, self._wall(phi_nodes, drivers))         # dynwave.c:752

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """d(withdrawal)/dH (m2/s), analytic and strictly positive."""
        free, slope = self._free(phi_nodes, drivers)
        if self.floods:
            wall = self._wall(phi_nodes, drivers)
            wall_slope = self._wall_slope(drivers).expand_as(wall)
            slope = torch.where(wall >= free, wall_slope, slope)
        return slope.expand(torch.broadcast_shapes(slope.shape, phi_nodes.shape))


class DynamicWaveHydraulics:
    """The dynamic-wave closure: the conduit coefficients, junction surface areas and
    outfall heads of one pass, from the previous pass's heads and flows.

    Reads the fed ``"water.phi"`` (m, full-node) and ``"water.q"`` (m3/s, per link: the
    pipes, then the regulators), the
    full-node driver ``"inflow"`` (m3/s, lateral inflow, zero at outfalls) and its own
    carried state, pinned by `Model` to the step start: ``"sewer.junction_depth"`` (m),
    ``"sewer.pipe_flow"`` (m3/s), ``"sewer.mid_area"`` (SWMM's stored mid-depth area, SI
    area m2 per design D2) and ``"sewer.net_inflow"`` (m3/s, per junction). With a
    `StepContext` it writes ``"water.sources"``, ``"water.phi_boundary"`` (outfall heads, m),
    the `DynamicWaveConduit` drivers and the `JunctionStorage` drivers, and the
    reporting key ``"sewer.overflow"`` (below); always the carried keys and the reporting
    keys ``"sewer.q"`` (m3/s), ``"sewer.h"`` (link depth, m), ``"sewer.v"`` (m/s),
    ``"sewer.V_wet"`` (m3), ``"sewer.junction_depth"`` (m) and ``"sewer.surcharged"``
    (per junction, 0/1).

    EXTRAN surcharge (design E1): `crown_depth` is each junction's highest attached conduit
    top (`dynwave.c:137-153`). The closure writes the drivers ``"water.surcharged"`` (0/1,
    from the fed heads: `JunctionStorage.surcharged`), ``"water.surcharge_slope"`` (m2/s)
    and ``"water.head_last"`` (the fed junction heads, m); a surcharged junction holds
    instantaneous continuity, net inflow 0 (see `JunctionStorage` for the tie rule at the
    crown). ``"sewer.surcharged"`` reports SWMM's own test, depth above the crown
    (`dynwave.c:688`): a junction held on the band at its crown (the tie rule) reports 1
    although it is not holding dQ = 0 -- the solve's own flag is the driver
    ``"water.surcharged"``. Under SLOT no junction is ever surcharged (`dynwave.c:677`).

    Flooding (design E3, no ponding): `flood_depth` is SWMM's y_max, the full depth
    `full_depth` (MaxDepth extended to the crown, `link.c:440-463`) plus the junction's
    SurDepth (`Manhole.surcharge_depth`). The closure writes ``"water.net_inflow_last"``
    (the fed net inflow, m3/s), ``"water.flood_excess"`` (m3/s, `JunctionStorage`) and the
    lagged flag ``"water.flooded"`` (0/1, the excess positive); a flooded junction is held
    at y_max. ``"sewer.overflow"`` (m3/s per junction, >= 0) is SWMM's flooding loss,
    ``0.5 (N_start + N)`` at a flooded junction (`dynwave.c:673-674, 783-791`), 0 below
    SWMM's FUDGE flow (1e-4 cfs) and elsewhere. It is lost from the system and is not part
    of the carried ``"sewer.net_inflow"``, which, like SWMM's ``oldNetInflow``
    (`node.c:303`), is the conduit and lateral net inflow alone.

    The drop-end area switch (the tie rule; docs: "Drop-end area switch"). A conduit end
    with an offset gives its junction no surface area while UP_/DN_CRITICAL or dry, and a
    quarter of the conduit above that LOWER threshold (FUDGE, or min(yN, yC) where the flow
    leaves the conduit there); downstream, fasnh takes it back to 0 at max(yN, yC), the
    UPPER threshold, above which it is a quarter again (dwflow.c:357-411, 472-499;
    `conduit_terms` reports each end's threshold nearest its depth). A junction FILLING
    through such a jump can have no fixed point (lower area: the storage equation lands
    above the threshold; upper area: below it), and SWMM's trials do not converge there.
    Per junction the closure takes the attached drop end whose threshold head H* is nearest
    the fed head and, where H* is above the step-start head (the jump in the withdrawal is
    then upward), writes H* and the storage area just below and just above it; the
    junction's storage then switches within the pass (`JunctionStorage`). Where the fed net
    inflow lies inside the jump at H*, at the fraction theta of it, the junction is HELD on
    the threshold: that end is classified as SWMM classifies it AT the threshold (the upper
    side, its tests being strict), and at a lower threshold the conduit's other end gets
    ``(2 - theta)`` quarters of the conduit, sliding between the critical side's half and
    the upper side's quarter (`conduit_terms`' ``hold1/2``), so that its class does not
    chatter through the other end's area either. Only the nearest threshold of each
    junction is switched within the pass.

    Orifices, weirs and outlets (design E5; `net.orifices`, `.weirs`, `.outlets`): each
    kind present is a `StructureLink` edge kind whose ``"water.q"`` columns follow the
    pipes' (`link_names`, `edge_kinds`). The closure writes, per kind, SWMM's regulator flow
    at the fed heads ``"water.<kind>.flow_offset"`` (m3/s), its slope
    ``"water.<kind>.slope"`` (m2/s: SWMM's ``dqdh``, an outlet's a central difference),
    the fed head difference ``"water.<kind>.dp_fed"`` (m) and the flap-gate flags; an
    orifice's end areas join its junctions' storage areas (`_structure_terms`,
    dynwave.c:503-524), the regulators raise crowns and full depths (dynwave.c:137-153,
    link.c:425-463), an outfall on a regulator takes yN = yC = 0 (link.c:753-765), and every
    link's flow enters the junction net inflows.

    Tidal and time-varying outfalls (design C5): a TIDAL or TIMESERIES outfall follows the
    FIXED outfall's stage rule with the stage of the full-outfall driver ``"outfall_stage"``
    (m, one per outfall; only the TIDAL and TIMESERIES entries are read), the stage of the
    END of the step (`outfall_stage` builds it; `simulate_dynamic_wave` sets it each step).
    The initial heads use its value at t = 0.

    The arithmetic is SWMM's own, in feet (design D2); `layer` names the head layer.
    """

    integrates = True
    state_keys = ("sewer.junction_depth", "sewer.pipe_flow", "sewer.mid_area",
                  "sewer.net_inflow")

    def __init__(self, graph: Network, net: SewerNetwork, *, layer: str = "water") -> None:
        self.net = graph
        self.layer = layer
        self.options = net.dynamic_wave
        self.notes: dict[str, str] = {}
        nodes = list(graph.nodes)
        self.pipe_names = [p.name for p in net.pipes]
        self.junction_names = [m.name for m in net.manholes]
        self.outfall_names = [o.name for o in net.outfalls]
        self.key_labels = {
            "sewer.junction_depth": list(self.junction_names),
            "sewer.pipe_flow": list(self.pipe_names),
            "sewer.mid_area": list(self.pipe_names),
            "sewer.net_inflow": list(self.junction_names),
        }
        self.node_invert = torch.tensor(
            [float(graph.graph.nodes[n]["invert"]) for n in nodes], dtype=F64)
        self.junction_idx = torch.tensor(
            [nodes.index(n) for n in self.junction_names], dtype=torch.long)
        self.outfall_idx = torch.tensor(
            [nodes.index(n) for n in self.outfall_names], dtype=torch.long)
        is_outfall = torch.zeros(len(nodes), dtype=torch.bool)
        is_outfall[self.outfall_idx] = True
        self.is_outfall = is_outfall
        self.u = torch.tensor([nodes.index(p.u) for p in net.pipes], dtype=torch.long)
        self.v = torch.tensor([nodes.index(p.v) for p in net.pipes], dtype=torch.long)
        self.diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
        self.roughness = torch.tensor([p.n for p in net.pipes], dtype=F64)
        self.slope = torch.tensor([p.slope for p in net.pipes], dtype=F64)
        self.length = torch.tensor([p.length for p in net.pipes], dtype=F64)
        self.offset1 = torch.tensor([p.inlet_offset for p in net.pipes], dtype=F64)
        self.offset2 = torch.tensor([p.outlet_offset for p in net.pipes], dtype=F64)
        self.barrels = torch.tensor([float(p.barrels) for p in net.pipes], dtype=F64)
        #: per-pipe flow limit (m3/s, total over the barrels: SWMM limits one barrel's flow
        #: to MaxFlow, dwflow.c:270-273; +inf for none)
        self.max_flow = self.barrels * torch.tensor(
            [torch.inf if p.max_flow is None else p.max_flow for p in net.pipes], dtype=F64)
        #: per-pipe InitFlow (m3/s, total, positive from u to v)
        self.initial_flow = torch.tensor([p.initial_flow for p in net.pipes], dtype=F64)
        # link_setFlapGate (link.c:658-667): a conduit's own gate closes on flow against its
        # direction (``q * direction < 0``); an outfall's gate closes on flow that would
        # enter the conduit at the outfall's end.
        outfall_gated = {o.name for o in net.outfalls if o.flap_gate}
        own = [p.flap_gate for p in net.pipes]
        rev = [p.flap_gate_reversed for p in net.pipes]
        self.flap_block_reverse = torch.tensor(
            [(g and not r) or p.v in outfall_gated
             for p, g, r in zip(net.pipes, own, rev, strict=True)], dtype=torch.bool)
        self.flap_block_forward = torch.tensor(
            [(g and r) or p.u in outfall_gated
             for p, g, r in zip(net.pipes, own, rev, strict=True)], dtype=torch.bool)
        if not net.pipes:
            raise ValueError("DynamicWaveHydraulics: the network has no conduit; dynamic-wave "
                             "routing needs at least one")
        self._rebuild_geometry()
        #: the number of pipes: the first columns of ``"water.q"``
        self.n_pipes = len(net.pipes)
        # The flow regulators, one `_Structures` per kind present, their columns after the
        # pipes' in STRUCTURE_KINDS order (the layer's element order).
        route = (self.options.routing_step if self.options.lengthening_routing_step is None
                 else self.options.lengthening_routing_step)                # link.c:1721, 2149
        self.structures: dict[str, _Structures] = {
            kind: _build_structures(kind, net.structures[kind], nodes, self.node_invert,
                                    outfall_gated, route)
            for kind in STRUCTURE_KINDS if net.structures[kind]}
        #: the edge kinds of the ``"water"`` layer, in column order
        self.edge_kinds = ("pipe", *self.structures)
        #: every ``"water.q"`` column's link name: the pipes, then the regulators by kind
        self.link_names = list(self.pipe_names) + [
            name for s in self.structures.values() for name in s.names]
        self.edge_u = torch.cat([self.u, *(s.u for s in self.structures.values())])
        self.edge_v = torch.cat([self.v, *(s.v for s in self.structures.values())])

        # Each outfall's own link and the end it sits at (link.c:740-751: node2 first): a
        # conduit, or a regulator, whose outfall depth SWMM sets from yNorm = yCrit = 0
        # (link.c:753-765: only a conduit computes them).
        pipe_of, at_v, via, reg_offset, link_of = [], [], [], [], []
        for name in self.outfall_names:
            ends = [(i, p.v == name) for i, p in enumerate(net.pipes) if name in (p.u, p.v)]
            regs = [(s, j) for s in self.structures.values() for j in range(len(s.names))
                    if name in (nodes[int(s.u[j])], nodes[int(s.v[j])])]
            if len(ends) + len(regs) != 1:
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall {name!r} is attached to "
                    f"{len(ends) + len(regs)} links; dynamic-wave routing takes exactly one")
            if regs:
                s, j = regs[0]
                if {nodes[int(s.u[j])], nodes[int(s.v[j])]} <= set(self.outfall_names):
                    raise ValueError(
                        f"DynamicWaveHydraulics: {s.kind} {s.names[j]!r} joins two outfalls; "
                        f"SWMM sets the depth of only one of them")
                downstream = nodes[int(s.v[j])] == name
                pipe_of.append(0)                    # a placeholder, masked by `via`
                link_of.append(self.link_names.index(s.names[j]))
                at_v.append(downstream)
                via.append(True)
                reg_offset.append(float(s.offset2[j] if downstream else s.offset1[j]))
                continue
            i, downstream = ends[0]
            pipe = net.pipes[i]
            if pipe.u in self.outfall_names and pipe.v in self.outfall_names:
                raise ValueError(
                    f"DynamicWaveHydraulics: conduit {pipe.name!r} joins two outfalls "
                    f"({pipe.u!r}, {pipe.v!r}); SWMM sets the depth of only one of them")
            pipe_of.append(i)
            link_of.append(i)
            at_v.append(downstream)
            via.append(False)
            reg_offset.append(0.0)
        # per outfall: its conduit's pipe index, internal only (0, a placeholder masked by
        # `outfall_on_structure`, for an outfall on a regulator)
        self._outfall_pipe = torch.tensor(pipe_of, dtype=torch.long)
        #: per outfall: its link's column in ``"water.q"`` (a conduit's, or a regulator's
        #: after the pipes'), positive away from the link's first node
        self.outfall_link = torch.tensor(link_of, dtype=torch.long)
        at_v_t = torch.tensor(at_v, dtype=torch.bool)
        #: per outfall: whether it is its link's downstream (second) node
        self.outfall_at_v = at_v_t
        #: per outfall: whether its link is a regulator (its depth then has yN = yC = 0)
        self.outfall_on_structure = torch.tensor(via, dtype=torch.bool)
        self.outfall_offset_ft = torch.where(
            self.outfall_on_structure, torch.tensor(reg_offset, dtype=F64),
            torch.where(at_v_t, self.offset2[self._outfall_pipe],
                        self.offset1[self._outfall_pipe]) / LCF)
        self.outfall_kind = torch.tensor(
            [_OUTFALL_CODES[o.kind] for o in net.outfalls], dtype=torch.long)
        self.outfall_stage_ft = torch.tensor(
            [(o.stage if o.stage is not None else o.invert) / LCF for o in net.outfalls],
            dtype=F64)
        #: per outfall: whether its stage is the ``"outfall_stage"`` driver's (TIDAL and
        #: TIMESERIES)
        self.outfall_staged = torch.tensor(
            [o.kind in _STAGED_KINDS for o in net.outfalls], dtype=torch.bool)
        self._outfalls = tuple(net.outfalls)
        self.min_area_ft2 = (
            DEFAULT_MIN_SURFACE_AREA_FT2 if self.options.min_surface_area is None
            else self.options.min_surface_area / LCF**2)            # dynwave.c:186-187
        # Each junction's crown: the highest conduit top over the conduit ends it carries,
        # crownElev - invertElev (dynwave.c:137-153); 0 for a junction with none.
        geo = self.geo
        crown = torch.zeros(len(nodes), dtype=F64)
        crown = crown.scatter_reduce(0, self.u, (geo.offset1 + geo.y_full).detach(), "amax")
        crown = crown.scatter_reduce(0, self.v, (geo.offset2 + geo.y_full).detach(), "amax")
        # the full-depth extension (link.c:440-463): both ends of a conduit, node 1 only of
        # a regulator other than a bottom orifice
        reach = crown.clone()
        for s in self.structures.values():
            crown = crown.scatter_reduce(0, s.u, s.offset1 + s.y_full, "amax")  # dynwave.c:148
            crown = crown.scatter_reduce(0, s.v, s.offset2 + s.y_full, "amax")  # dynwave.c:152
            top = torch.where(s.extends, s.offset1 + s.y_full, torch.zeros_like(s.y_full))
            reach = reach.scatter_reduce(0, s.u, top, "amax")                 # link.c:452
        self.crown_ft = crown[self.junction_idx]
        #: per-junction crown depth (m), in junction order
        self.crown_depth = self.crown_ft * LCF
        # The full depth: MaxDepth, extended to every attached conduit's top and to the
        # crown of a regulator leaving it (link.c:440-463) -- at a storage unit only when it
        # has a SurDepth (link.c:450, 458).
        max_depth = torch.tensor([0.0 if m.ground is None else m.ground - m.invert
                                  for m in net.manholes], dtype=F64)
        sur_depth = torch.tensor([m.surcharge_depth for m in net.manholes], dtype=F64)
        #: per junction: a SWMM storage unit (`Manhole.storage`)
        self.is_storage = torch.tensor([m.storage is not None for m in net.manholes],
                                       dtype=torch.bool)
        node_storage = torch.zeros(len(nodes), dtype=torch.bool)
        node_storage[self.junction_idx] = self.is_storage
        #: per node: a storage unit (a regulator's end area is zeroed there, dynwave.c:520)
        self.node_storage = node_storage
        extend = ~self.is_storage | (sur_depth > 0.0)                         # link.c:450
        #: per-junction full depth (m): MaxDepth extended to the crown
        self.full_depth = torch.where(
            extend, torch.maximum(max_depth, reach[self.junction_idx] * LCF), max_depth)
        #: per-junction flood depth y_max (m): the full depth plus SurDepth, above which
        #: the junction floods (no ponding, dynwave.c:746-748)
        self.flood_depth = self.full_depth + sur_depth
        #: per-junction EXTRAN surcharge depth (m): the crown of a junction, the full depth
        #: of a storage unit (dynwave.c:683-690); `surchargeable` is False where SWMM never
        #: surcharges (a storage unit without SurDepth, a junction with no crown)
        self.surcharge_at = torch.where(self.is_storage, self.full_depth, self.crown_depth)
        self.surchargeable = torch.where(self.is_storage, sur_depth > 0.0,
                                         self.crown_depth > 0.0)
        #: the storage units' area and volume curves (`storage.StorageCurves`), per junction
        self.storage_curves = (
            StorageCurves([m.storage for m in net.manholes], self.full_depth,
                          names=self.junction_names)
            if bool(self.is_storage.any()) else None)
        #: per-junction InitDepth (m)
        self.initial_junction_depth = torch.tensor(
            [m.initial_depth for m in net.manholes], dtype=F64)
        too_deep = self.initial_junction_depth > self.flood_depth                 # node.c:216
        if bool(too_deep.any()):
            bad = [self.junction_names[i] for i in too_deep.nonzero().flatten().tolist()]
            raise ValueError(
                f"DynamicWaveHydraulics: junction(s) {bad} start deeper (initial_depth) than "
                f"their full depth plus surcharge depth (SWMM: initial depth exceeds max. "
                f"depth)")
        jinv = self.node_invert[self.junction_idx]
        #: the junctions' `JunctionStorage` (the model's node source; `junction_residual`)
        self.storage = JunctionStorage(
            self.junction_idx, jinv, bed_area=self.min_area_ft2 * _AREA_SI,
            crown=jinv + self.surcharge_at if self.options.surcharge_method == "extran"
            else None, top=jinv + self.flood_depth, surchargeable=self.surchargeable)

    # ------------------------------------------------------------------ geometry
    def _rebuild_geometry(self) -> None:
        opts = self.options
        factor = None
        if opts.lengthening_step > 0.0:                                       # link.c:1105-1108
            route = (opts.routing_step if opts.lengthening_routing_step is None
                     else opts.lengthening_routing_step)
            step = min(route, opts.lengthening_step)                          # link.c:1248
            factor = lengthening_factor(self.diameter, self.roughness, self.slope,
                                        self.length, step)
        self.geo = ConduitGeometry.from_si(
            self.diameter, self.roughness, self.slope, self.length,
            self.node_invert[self.u], self.node_invert[self.v], self.offset1, self.offset2,
            self.is_outfall[self.u], self.is_outfall[self.v], barrels=self.barrels,
            length_factor=factor)

    def set_roughness(self, i: int, value) -> None:
        """Set Manning's n (s m^(-1/3)) of pipe `i` to `value`, which may be a tensor that
        requires grad; the geometry is rebuilt so the gradient reaches it. The other pipes'
        n are detached, so only the latest call's `value` carries a gradient."""
        value = torch.as_tensor(value, dtype=F64).reshape(1)
        self.roughness = torch.cat([self.roughness[:i].detach(), value,
                                    self.roughness[i + 1:].detach()])
        self._rebuild_geometry()

    # ------------------------------------------------------------------- helpers
    def _inflow(self, drivers: Mapping) -> Tensor:
        try:
            inflow = drivers["inflow"]
        except KeyError as exc:
            raise KeyError("DynamicWaveHydraulics: driver 'inflow' (m3/s, full-node lateral "
                           "inflow) is required") from exc
        if not bool(torch.isfinite(inflow).all()):
            bad = ~torch.isfinite(inflow).reshape(-1, inflow.shape[-1]).all(0)
            names = [self.net.nodes[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(f"DynamicWaveHydraulics: non-finite inflow at {names}")
        at_outfalls = inflow[..., self.outfall_idx]
        if bool((at_outfalls != 0).any()):
            bad = (at_outfalls != 0).reshape(-1, len(self.outfall_names)).any(0)
            names = [self.outfall_names[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(
                f"DynamicWaveHydraulics: inflow at outfall(s) {names} must be 0; an outfall "
                f"head is set by its conduit, not by a node balance")
        return inflow

    def _terms(self, state: Mapping, dt: float,
               hold: tuple[Tensor, Tensor] | None = None) -> ConduitTerms:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"][..., :self.n_pipes]
        # SWMM's node depth is never negative (dynwave.c:743); the stiff bed lets a dry
        # junction's head sit a hair below its invert.
        depth_ft = torch.clamp(heads - self.node_invert, min=0.0) / LCF
        return conduit_terms(
            self.geo, depth_ft[..., self.u], depth_ft[..., self.v], q / QCF,
            state["sewer.pipe_flow"] / QCF, state["sewer.mid_area"] / _AREA_SI, dt,
            damping=self.options.inertial_damping,
            normal_flow_limited=self.options.normal_flow_limited,
            surcharge_method=self.options.surcharge_method,
            hold1=None if hold is None else hold[0], hold2=None if hold is None else hold[1])

    def _junction_depth_ft(self, heads: Tensor) -> Tensor:
        """The junction depths (ft, >= 0) at full-node `heads` (m), SWMM's (dynwave.c:743)."""
        jidx = self.junction_idx
        return torch.clamp(heads[..., jidx] - self.node_invert[jidx], min=0.0) / LCF

    def _storage_area_ft2(self, heads: Tensor) -> Tensor:
        """Each junction's own storage area (ft2; 0 at a plain junction) at `heads` (m):
        `node_getSurfArea` (node.c:383-396), the curve in m and m2 (node.c:997-1017)."""
        depth_ft = self._junction_depth_ft(heads)
        if self.storage_curves is None:
            return torch.zeros_like(depth_ft)
        return self.storage_curves.area(depth_ft * LCF) / LCF**2              # node.c:1017

    def _storage_volume(self, heads: Tensor) -> Tensor:
        """Each junction's stored volume (m3; 0 at a plain junction, as SWMM's junction
        volume, node.c:362-379) at `heads` (m): `node_getVolume`, capped at the full volume
        at or above the full depth (node.c:943-945)."""
        depth_ft = self._junction_depth_ft(heads)
        if self.storage_curves is None:
            return torch.zeros_like(depth_ft)
        return self.storage_curves.volume(depth_ft * LCF)

    def _drop_switch(self, state: Mapping, terms: ConduitTerms, net_inflow: Tensor,
                     dt: float, storage_ft2: Tensor) -> dict[str, Tensor]:
        """The drop-end area switch at each junction (`JunctionStorage`, the tie rule), from
        the fed state and one pass's `terms`. Per junction: the threshold head ``"head"`` H*
        (m) of the attached drop end whose threshold is nearest the fed head; the storage
        areas just ``"below"`` and just ``"above"`` it (m2, SWMM's area floor applied);
        ``"applies"``, the threshold lies above the step-start head (the jump in the
        withdrawal is then upward), a storage unit's own area `storage_ft2` (ft2) being on
        both sides; ``"held"``, the fed net inflow `net_inflow` (m3/s) lies
        inside that jump at H*, the junction being on the threshold. Per pipe end:
        ``"hold1"``/``"hold2"``, at the chosen end of a held junction the fraction theta of
        the jump at which the net inflow lies (0 at its bottom, 1 at its top), -1 elsewhere
        (`conduit_terms`)."""
        geo = self.geo
        nodes = torch.cat([self.u, self.v])
        n = self.node_invert.shape[-1]
        heads = state[f"{self.layer}.phi"]
        z = torch.cat([geo.invert1 + geo.offset1, geo.invert2 + geo.offset2])
        depth = torch.cat([terms.drop_depth1, terms.drop_depth2], -1)
        finite = torch.isfinite(depth) & ~self.is_outfall[nodes]
        head_thr = torch.where(finite, (z + torch.where(finite, depth, torch.zeros_like(depth)))
                               * LCF, torch.zeros_like(depth))
        batch = head_thr.shape[:-1]
        index = nodes.expand(batch + nodes.shape)
        inf = torch.full(batch + (n,), torch.inf, dtype=F64)
        dist = torch.where(finite, (heads[..., nodes] - head_thr).abs(),
                           torch.full_like(head_thr, torch.inf))
        nearest = inf.scatter_reduce(-1, index, dist, "amin")
        chosen = finite & (dist == nearest[..., nodes])
        # an exact distance tie (two ends with one threshold head) takes the lowest end index
        order = torch.arange(nodes.shape[-1]).expand(index.shape)
        ranked = torch.where(chosen, order, torch.full_like(order, nodes.shape[-1]))
        first = torch.full(batch + (n,), nodes.shape[-1], dtype=torch.long)
        first = first.scatter_reduce(-1, index, ranked, "amin")
        chosen = chosen & (order == first[..., nodes])

        def pick(values: Tensor, empty: float) -> Tensor:
            base = torch.full(batch + (n,), empty, dtype=F64)
            masked = torch.where(chosen, values, torch.full_like(values, empty))
            return base.scatter_reduce(-1, index, masked, "amax")

        jidx = self.junction_idx
        h_star = pick(head_thr, -torch.inf)[..., jidx]
        jump = pick(torch.cat([terms.drop_area1, terms.drop_area2], -1), 0.0)[..., jidx]
        own = pick(torch.cat([terms.surface_area1, terms.surface_area2], -1), 0.0)[..., jidx]
        above = pick(torch.cat([terms.drop_above1, terms.drop_above2], -1).to(F64),
                     0.0)[..., jidx] > 0.5
        surf = torch.zeros(batch + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)
        surf = surf.index_add(-1, self.v, terms.surface_area2)[..., jidx] + storage_ft2
        rest = surf - own
        lo = torch.clamp(torch.where(above, rest, surf), min=self.min_area_ft2) * _AREA_SI
        hi = torch.clamp(torch.where(above, surf, rest + jump), min=self.min_area_ft2) * _AREA_SI
        has = torch.isfinite(h_star)
        head_start = self.node_invert[jidx] + state["sewer.junction_depth"]
        safe_star = torch.where(has, h_star, head_start)
        applies = has & (safe_star > head_start) & (hi > lo)
        rate2 = 2.0 / dt
        st_lo = rate2 * lo * (safe_star - head_start) - state["sewer.net_inflow"]
        st_hi = rate2 * hi * (safe_star - head_start) - state["sewer.net_inflow"]
        held = applies & (st_lo < net_inflow) & (net_inflow < st_hi)
        gap = torch.where(held, st_hi - st_lo, torch.ones_like(st_hi))
        theta = torch.where(held, (net_inflow - st_lo) / gap, torch.full_like(gap, -1.0))
        theta_node = torch.full(batch + (n,), -1.0, dtype=F64)
        theta_node[..., jidx] = theta
        hold = torch.where(chosen, theta_node[..., nodes], torch.full_like(head_thr, -1.0))
        p = self.u.shape[-1]
        return {"head": safe_star, "below": lo, "above": hi, "applies": applies,
                "held": held, "hold1": hold[..., :p], "hold2": hold[..., p:]}

    def _structure_terms(self, heads: Tensor) -> tuple[dict, Tensor]:
        """The regulators at full-node `heads` (m): per kind ``(StructureFlow, slope)`` (ft
        units: SWMM's flow and its ``dqdh``, an outlet's a central difference,
        `structures.outlet_dqdh`), and every node's regulator surface area (ft2):
        `findNonConduitSurfArea` (dynwave.c:503-524), an orifice's ``surfArea / 2`` at each
        end (weirs and outlets none), zeroed at an UP_CRITICAL first end, a DN_CRITICAL
        second end and at a storage unit."""
        inv = self.node_invert
        h_ft = (inv + torch.clamp(heads - inv, min=0.0)) / LCF               # dynwave.c:743
        z_ft = inv / LCF
        area = torch.zeros(heads.shape[:-1] + inv.shape[-1:], dtype=F64)
        found: dict = {}
        for kind, s in self.structures.items():
            h1, h2 = h_ft[..., s.u], h_ft[..., s.v]
            z1, z2 = z_ft[s.u], z_ft[s.v]
            crest = z1 + s.offset1                                            # link.c:1865
            gates = (s.block_forward, s.block_reverse)
            if kind == "orifice":
                flow = orifice_flow(s.params, h1, h2, z1, z2, crest, *gates)
                slope = flow.dqdh
                half = flow.surf_area / 2.0                                   # dynwave.c:511
                end1 = torch.where((flow.flow_class == UP_CRITICAL) | self.node_storage[s.u],
                                   torch.zeros_like(half), half)              # dynwave.c:518-520
                end2 = torch.where((flow.flow_class == DN_CRITICAL) | self.node_storage[s.v],
                                   torch.zeros_like(half), half)              # dynwave.c:521-523
                area = area.index_add(-1, s.u, end1).index_add(-1, s.v, end2)
            elif kind == "weir":
                flow = weir_flow(s.params, h1, h2, z1, z2, crest, *gates)
                slope = flow.dqdh
            else:
                flow = outlet_flow(s.params, h1, h2, z1, z2, crest, *gates)
                slope = outlet_dqdh(s.params, h1, h2, z1, z2, crest, *gates)
            found[kind] = (flow, slope)
        return found, area

    def _surcharged(self, heads: Tensor) -> Tensor:
        """EXTRAN's surcharge test on the junction heads `heads` (m, full-node): depth above
        a positive crown (`dynwave.c:690`), at a storage unit with a SurDepth depth above its
        full depth (`dynwave.c:683-687`), as a bool per junction."""
        depth_ft = self._junction_depth_ft(heads)
        if self.options.surcharge_method != "extran":
            return torch.zeros_like(depth_ft, dtype=torch.bool)
        at_ft = torch.where(self.is_storage, self.full_depth / LCF, self.crown_ft)
        return self.surchargeable & (depth_ft > at_ft)

    def _carried(self, state: Mapping, terms: ConduitTerms, inflow: Tensor) -> State:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        batch = torch.broadcast_shapes(q.shape[:-1], inflow.shape[:-1])
        through = torch.zeros(batch + (n,), dtype=F64)
        q_b = q.expand(batch + q.shape[-1:])
        # every link's flow, conduits and regulators (dynwave.c:528-541, 382-412)
        through = through.index_add(-1, self.edge_v, q_b).index_add(-1, self.edge_u, -q_b)
        net_inflow = inflow + through                                  # dynwave.c:528-541
        return {
            "sewer.junction_depth": torch.clamp(heads[..., jidx] - self.node_invert[jidx],
                                                min=0.0),
            "sewer.pipe_flow": q[..., :self.n_pipes],
            "sewer.mid_area": terms.mid_area * _AREA_SI,
            "sewer.net_inflow": net_inflow[..., jidx],
        }

    def outfall_stage(self, t: float, series: Mapping | None = None) -> Tensor:
        """Every outfall's stage elevation (m, shape ``(n_outfalls,)``) `t` s into the run:
        a TIDAL one's from its curve (`tidal_stage`), a TIMESERIES one's from `series[name]`
        (a stage in m, or a function of `t` returning one), a FIXED one's its own stage, the
        others' their invert (unused). This is the ``"outfall_stage"`` driver of a step
        whose END is at `t` (SWMM sets the outfall depth with the routing clock already
        advanced: `routing.c:301-302` before `routeFlow`, node.c:1449, 1456)."""
        series = {} if series is None else series
        stage = []
        for o in self._outfalls:
            if o.kind == "tidal":
                value = tidal_stage(o.tide, t)
            elif o.kind == "timeseries":
                source = series[o.name]
                value = float(source(t) if callable(source) else source)
                if not math.isfinite(value):
                    raise ValueError(
                        f"DynamicWaveHydraulics: outfall_stage of timeseries outfall "
                        f"{o.name!r} is {value!r} at t={t} s; it must be finite (m elevation)")
            else:
                value = o.stage if o.stage is not None else o.invert
            stage.append(value)
        return torch.tensor(stage, dtype=F64)

    def check_outfall_stage(self, series: Mapping | None) -> dict:
        """`series` as `outfall_stage` takes it: ``{timeseries outfall name: stage in m or
        function of t}``, covering every TIMESERIES outfall and nothing else (a tidal
        outfall reads its curve, the others their own stage). Returns it as a dict."""
        series = {} if series is None else dict(series)
        wanted = {o.name for o in self._outfalls if o.kind == "timeseries"}
        kinds = {o.name: o.kind for o in self._outfalls}
        for name in series:
            if name not in wanted:
                what = ("is not an outfall" if name not in kinds else
                        f"is a {kinds[name]} outfall, whose stage is "
                        + ("its tide curve's" if kinds[name] == "tidal" else "not time-varying"))
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall_stage names {name!r}, which {what}; it "
                    f"takes the timeseries outfalls {sorted(wanted)}")
        missing = sorted(wanted - set(series))
        if missing:
            raise ValueError(
                f"DynamicWaveHydraulics: timeseries outfall(s) {missing} need an "
                f"outfall_stage (m elevation, or a function of t in s) each")
        return series

    def _stage_ft(self, drivers: Mapping | None) -> Tensor:
        """Each outfall's stage (ft): its own stage, a TIDAL or TIMESERIES outfall's the
        ``"outfall_stage"`` driver's (m, ``(..., n_outfalls)``)."""
        if not bool(self.outfall_staged.any()):
            return self.outfall_stage_ft
        if drivers is None or "outfall_stage" not in drivers:
            raise KeyError(
                f"DynamicWaveHydraulics: driver 'outfall_stage' (m elevation, one per "
                f"outfall) is required by outfall(s) "
                f"{[o.name for o in self._outfalls if o.kind in _STAGED_KINDS]}; "
                f"simulate_dynamic_wave and build_model set it")
        given = torch.as_tensor(drivers["outfall_stage"], dtype=F64)
        if given.shape[-1] != len(self._outfalls):
            raise ValueError(
                f"DynamicWaveHydraulics: outfall_stage must have one value per outfall "
                f"({len(self._outfalls)}), got shape {tuple(given.shape)}")
        if not bool(torch.isfinite(given[..., self.outfall_staged]).all()):
            bad = [o.name for o, k in zip(self._outfalls, self.outfall_staged.tolist(),
                                           strict=True) if k]
            raise ValueError(f"DynamicWaveHydraulics: non-finite outfall_stage at {bad}")
        return torch.where(self.outfall_staged, given / LCF, self.outfall_stage_ft)

    def _outfall_heads(self, q: Tensor, stage_ft: Tensor | None = None) -> Tensor:
        """`link.c::link_setOutfallDepth` + `node.c::outfall_setOutletDepth` (link.c:728-766,
        node.c:1413-1492), from each outfall conduit's flow `q` (m3/s, the pipes' columns)
        and the outfalls' stages `stage_ft` (ft, `_stage_ft`; default the static ones):
        heads (m). An outfall on a regulator takes yN = yC = 0. FIXED, TIDAL and TIMESERIES
        outfalls share the stage rule (node.c:1469-1490); they differ only in the stage."""
        op = self._outfall_pipe
        geo = self.geo
        q_ft = (q[..., op] / QCF / self.barrels[op]).abs()                # link.c:759
        yf, af = geo.y_full[op], geo.a_full[op]
        y_norm = normal_depth_ft(q_ft, yf, af, geo.s_full[op], geo.beta[op])
        y_crit = critical_depth(q_ft, yf, af)
        # a regulator's outfall: yNorm = yCrit = 0 (link.c:753-765)
        reg = self.outfall_on_structure
        y_norm = torch.where(reg, torch.zeros_like(y_norm), y_norm)
        y_crit = torch.where(reg, torch.zeros_like(y_crit), y_crit)
        z = self.outfall_offset_ft
        zero = torch.zeros_like(y_norm)
        free = torch.where(z > 0.0, zero, torch.minimum(y_norm, y_crit))      # node.c:1431
        normal = torch.where(z > 0.0, zero, y_norm)                           # node.c:1436
        invert = self.node_invert[self.outfall_idx] / LCF
        stage = self.outfall_stage_ft if stage_ft is None else stage_ft
        yc = torch.minimum(y_crit, y_norm)                                    # node.c:1469
        fixed = torch.where(
            yc + z + invert < stage, (stage - invert).expand_as(yc),         # node.c:1473
            torch.where(
                z > 0.0,
                torch.where(stage < invert + z,
                            torch.clamp(stage - invert, min=0.0).expand_as(yc),
                            z + yc),                                          # node.c:1481-1485
                yc))                                                          # node.c:1490
        kind = self.outfall_kind
        depth = torch.where(kind == 0, free,
                            torch.where(kind == 1, normal, fixed))    # kinds >= 2 are staged
        return self.node_invert[self.outfall_idx] + depth * LCF

    def _switch_slope(self, q: Tensor, terms: ConduitTerms, area_ft2: Tensor, surf: Tensor,
                      dt: float, stage_ft: Tensor) -> Tensor:
        """The slope switch's wall slope S (m2/s, per pipe; `DynamicWaveConduit`): the
        linearised switching surface ``1/(c + 1/S_max)``. ``c`` (m per m3/s) is how far the
        lagged outfall depth at the conduit's downstream end moves per unit of its flow
        (`_outfall_heads`, a forward difference; 0 for other conduits): on the surface
        ``y1 = y2(q)`` the pass then sees the outfall depth follow the flow, as it does at
        the fixed point. An outfall at the conduit's UPSTREAM end gets no such term (there
        the surface would fall with the flow), so that conduit's S is ``S_max``: its
        switch stays exact at the fixed point but may take more passes to settle. ``S_max``
        is `K_SWITCH` times the smaller end junction's grounding (storage slope ``2 A_s/dt``
        plus its conduits' conveyance)."""
        k_si = terms.conveyance * (QCF / LCF)
        ground = torch.zeros_like(surf).index_add(-1, self.u, k_si).index_add(-1, self.v, k_si)
        ground = ground.index_add(-1, self.junction_idx, 2.0 / dt * area_ft2 * _AREA_SI)
        ground = torch.where(self.is_outfall, torch.full_like(ground, torch.inf), ground)
        s_max = K_SWITCH * torch.minimum(ground[..., self.u], ground[..., self.v])
        op = self._outfall_pipe
        q_abs = q.abs()
        step = 1e-6 * torch.clamp(q_abs[..., op], min=1e-3)
        bumped = q_abs.clone()
        bumped[..., op] = q_abs[..., op] + step
        rise = (self._outfall_heads(bumped, stage_ft)
                - self._outfall_heads(q_abs, stage_ft)) / step
        rise = torch.where(self.outfall_at_v & ~self.outfall_on_structure,
                           torch.clamp(rise, min=0.0), torch.zeros_like(rise))
        compliance = torch.zeros_like(s_max).index_add(-1, op, rise.expand(
            s_max.shape[:-1] + rise.shape[-1:]))
        return 1.0 / (compliance + 1.0 / s_max)

    # ------------------------------------------------------------------ protocol
    def __call__(self, state: Mapping, drivers: Mapping,
                 ctx: StepContext | None = None) -> dict[str, Tensor]:
        """One pass's drivers (see the class docstring); a query (no `ctx`, or
        ``ctx.dt is None``) returns the carried and reporting keys only."""
        dt = None if ctx is None else ctx.dt
        inflow = self._inflow(drivers)
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        out = self._carried(state, terms, inflow)
        heads = state[f"{self.layer}.phi"]
        # a storage unit's own area at the previous pass's depth (dynwave.c:315)
        storage_ft2 = self._storage_area_ft2(heads)
        drop = None
        regulators: dict = {}
        if dt is not None and self.structures:
            # the regulators at the previous pass's heads (dynwave.c:423-454): their flows,
            # slopes and end areas, which join the storage units' own (dynwave.c:503-524)
            regulators, reg_ft2 = self._structure_terms(heads)
            storage_ft2 = storage_ft2 + reg_ft2[..., self.junction_idx]
        if dt is not None:
            # the drop-end area switch (JunctionStorage): a junction held on its threshold
            # has its chosen drop end classified on the upper side, SWMM's class AT it. The
            # hold set is decided once per pass, from the first evaluation's areas and the
            # fed net inflow (which does not depend on the class), and fixed for the pass:
            # the re-evaluation only supplies the held class's areas, so re-deciding on it
            # would let the class it sets feed back into its own decision within one pass.
            drop = self._drop_switch(state, terms, out["sewer.net_inflow"], dt, storage_ft2)
            if bool(drop["held"].any()):
                terms = self._terms(state, dt, (drop["hold1"], drop["hold2"]))
                out = self._carried(state, terms, inflow)
                drop = self._drop_switch(state, terms, out["sewer.net_inflow"], dt,
                                         storage_ft2)
        q_all = state[f"{self.layer}.q"]
        q = q_all[..., :self.n_pipes]
        depth_ft = terms.depth
        a_link = area_of_depth(depth_ft, self.geo.y_full, self.geo.a_full)
        moving = (depth_ft > _VELOCITY_MIN_DEPTH) & (a_link > FUDGE)          # link.c:834-840
        v_ft = (q / QCF / self.barrels) / torch.where(moving, a_link, torch.ones_like(a_link))
        out.update({
            "sewer.q": q,
            "sewer.h": depth_ft * LCF,
            "sewer.v": torch.where(moving, v_ft, torch.zeros_like(v_ft)) * LCF,
            "sewer.V_wet": terms.volume * QCF,
            "sewer.surcharged": self._surcharged(heads).to(F64),
            "sewer.storage_volume": self._storage_volume(heads),
        })
        if dt is None:
            return out
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        surf = torch.zeros(terms.surface_area1.shape[:-1] + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)                # dynwave.c:582-583
        surf = surf.index_add(-1, self.v, terms.surface_area2)
        # the storage unit's own area plus its conduit ends' (not zeroed at a storage node,
        # dwflow.c:546-547; only a regulator's are, dynwave.c:520-523) and its regulators'
        # ends' (in `storage_ft2` above), then the floor
        area_ft2 = torch.clamp(surf[..., jidx] + storage_ft2,
                               min=self.min_area_ft2)                         # dynwave.c:670
        # conveyance the pass lags into c0 (other end pinned), per junction (dynwave.c:578)
        lagged = torch.zeros_like(surf).index_add(-1, self.u, terms.lagged_conveyance1)
        lagged = lagged.index_add(-1, self.v, terms.lagged_conveyance2)[..., jidx]
        anchor = lagged * (QCF / LCF) + K_ANCHOR * 2.0 / dt * area_ft2 * _AREA_SI
        switch = terms.slope_switch
        stage_ft = self._stage_ft(drivers)
        switch_slope = self._switch_slope(q, terms, area_ft2, surf, dt, stage_ft)
        # the drop switch's wall slope: K_SWITCH times the junction's grounding above it
        k_si = terms.conveyance * (QCF / LCF)
        sum_k = torch.zeros_like(surf).index_add(-1, self.u, k_si).index_add(-1, self.v, k_si)
        drop_slope = K_SWITCH * (2.0 / dt * drop["above"] + sum_k[..., jidx])
        drop_head = torch.where(drop["applies"], drop["head"],
                                torch.full_like(drop["head"], torch.inf))
        out.update({
            f"{self.layer}.sources": inflow,
            f"{self.layer}.phi_boundary": self._outfall_heads(q, stage_ft),
            "water.flow_offset": terms.flow_offset * QCF,
            "water.conveyance": terms.conveyance * (QCF / LCF),
            "water.normal_flow": torch.where(switch, terms.switch_normal_flow,
                                             terms.normal_flow) * QCF,
            "water.limited": (terms.limited & ~switch).to(F64),
            "water.slope_switch": switch.to(F64),
            "water.switch_drop": terms.switch_drop * LCF,
            "water.switch_slope": torch.where(switch, switch_slope,
                                              torch.zeros_like(switch_slope)),
            "water.flow_last": q,
            "water.max_flow": self.max_flow,
            "water.flap_block_reverse": self.flap_block_reverse.to(F64),
            "water.flap_block_forward": self.flap_block_forward.to(F64),
            "water.dry_flow_cap": FUDGE_FLOW * self.barrels,
            "water.dry_upstream": terms.dry_upstream.to(F64),
            "water.dry_downstream": terms.dry_downstream.to(F64),
            "water.storage_area": area_ft2 * _AREA_SI,
            "water.storage_rate": torch.tensor(1.0 / dt, dtype=F64),
            "water.head_start": self.node_invert[jidx] + state["sewer.junction_depth"],
            "water.net_inflow_start": state["sewer.net_inflow"],
            "water.surcharge_slope": anchor,
            "water.head_last": state[f"{self.layer}.phi"][..., jidx],
            "water.net_inflow_last": out["sewer.net_inflow"],
            "water.area_switch_head": drop_head,
            "water.area_below": drop["below"],
            "water.area_above": drop["above"],
            "water.area_switch_slope": drop_slope,
        })
        for kind, (flow, slope) in regulators.items():
            s = self.structures[kind]
            out.update({
                f"water.{kind}.flow_offset": flow.flow * QCF,
                f"water.{kind}.slope": slope * (QCF / LCF),
                f"water.{kind}.dp_fed": heads[..., s.u] - heads[..., s.v],
                f"water.{kind}.flap_block_reverse": s.block_reverse.to(F64),
                f"water.{kind}.flap_block_forward": s.block_forward.to(F64),
            })
        out["water.surcharged"] = self.storage.surcharged(out)
        excess = self.storage.flood_excess(out)
        flooded = excess > 0.0
        # SWMM's overflow is the step's trapezoid net inflow dV/dt, zeroed below FUDGE
        # (dynwave.c:673-674, 783-791); it never enters the next step's oldNetInflow
        # (node.c:303 takes inflow - outflow, which exclude it).
        overflow = 0.5 * (state["sewer.net_inflow"] + out["sewer.net_inflow"])
        overflow = torch.where(flooded & (overflow >= FUDGE_FLOW), overflow,
                               torch.zeros_like(overflow))
        out.update({"water.flood_excess": excess, "water.flooded": flooded.to(F64),
                    "sewer.overflow": overflow})
        return out

    def carried(self, state: Mapping, drivers: Mapping, *, dt: float | None = None) -> State:
        """The four carried keys evaluated AT `state` (``water.phi``, ``water.q``), for
        re-synchronising after a step of `dt` s (default: the routing step): the pass's own
        copy is its input's, which equals its output only to the iteration tolerance."""
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        return self._carried(state, terms, self._inflow(drivers))

    def _initial_stage_ft(self, drivers: Mapping | None) -> Tensor:
        """The outfalls' stages (ft) at t = 0 for the initial heads: the driver's if it is
        given, else the tide curves' at t = 0 (a TIMESERIES outfall has no stage without
        its driver)."""
        if drivers is not None and "outfall_stage" in drivers:
            return self._stage_ft(drivers)
        if any(o.kind == "timeseries" for o in self._outfalls):
            raise KeyError("DynamicWaveHydraulics: the initial state needs the "
                           "'outfall_stage' driver for a timeseries outfall")
        return self._stage_ft({"outfall_stage": self.outfall_stage(0.0)})

    def initial_state(self, drivers: Mapping | None = None) -> State:
        """SWMM's initial state (flowrout.c:75-102, 336-507), cold and dry unless the network
        sets InitDepth (`Manhole.initial_depth`) or InitFlow (`Pipe.initial_flow`).

        - Each conduit with an InitFlow ``q0`` (a total, positive from `u` to `v`) starts at
          the normal depth of one barrel's ``q0 / barrels``, capped at the depth of ``qMax``
          (`link.c::link_getYnorm`, link.c:783-804; `conduit_initState`, link.c:1312).
        - A junction with no InitDepth starts at the mean over its attached conduits of that
          depth plus the conduit's FIRST-end offset (zero for a conduit without InitFlow,
          which still counts in the mean): `initNodeDepths`, flowrout.c:337-385 (the first
          offset is used at both ends there, as in SWMM). A junction with an InitDepth keeps
          it, and so does a storage unit, whose InitDepth of 0 is not averaged either
          (flowrout.c:375).
        - Outfalls start at the depth `link_setOutfallDepth` gives from the conduit's initial
          flow (a FIXED outfall at its stage, a TIDAL or TIMESERIES one at its stage at
          t = 0, the others at their invert at zero flow; flowrout.c:383).
        - A conduit with no InitFlow starts at the mean of its end-node depths above its
          offsets, clamped to [0, yFull] and at least FUDGE (`initLinkDepths`,
          flowrout.c:389-420); the area at that depth is its stored area (`initLinks`,
          flowrout.c:497).
        - A junction's net inflow is that of the initial flows through it (`initNodes`,
          flowrout.c:458-466 and node.c:303): 0 with no InitFlow. SWMM zeroes the lateral
          inflow before routing starts (node.c:264, flowrout.c:437), so the first step's
          ``oldNetInflow`` has no lateral part (routing.c:236-242).

        `drivers`, when given, has its ``"inflow"`` validated and supplies the
        ``"outfall_stage"`` at t = 0."""
        if drivers is not None:
            self._inflow(drivers)
        geo = self.geo
        q0 = self.initial_flow.clone()
        # link.c:783-804 / 1311: normal depth of one barrel's initial flow (0 where none)
        y_link = normal_depth_ft(q0.abs() / QCF / self.barrels, geo.y_full, geo.a_full,
                                 geo.s_full, geo.beta)
        # flowrout.c:353-363: depth + the first offset, at both ends, where depth > FUDGE
        y_end = torch.where(y_link > FUDGE, y_link + geo.offset1, torch.zeros_like(y_link))
        total = torch.zeros(self.node_invert.shape, dtype=F64)
        count = torch.zeros_like(total)
        total = total.index_add(0, self.u, y_end).index_add(0, self.v, y_end)
        # every link counts, a regulator with depth 0 (link.c:518, flowrout.c:353-363)
        count = count.index_add(0, self.edge_u, torch.ones(self.edge_u.shape, dtype=F64))
        count = count.index_add(0, self.edge_v, torch.ones(self.edge_v.shape, dtype=F64))
        jidx = self.junction_idx
        given = self.initial_junction_depth / LCF
        average = torch.where(count[jidx] > 0.0, total[jidx] / count[jidx].clamp(min=1.0),
                              torch.zeros_like(given))                        # flowrout.c:378
        # a storage unit keeps its own InitDepth, 0 included (flowrout.c:375-376)
        depth_j = torch.where((given > 0.0) | self.is_storage, given, average)
        phi = self.node_invert.clone()
        phi[jidx] = self.node_invert[jidx] + depth_j * LCF
        phi[self.outfall_idx] = self._outfall_heads(
            q0, self._initial_stage_ft(drivers))                              # flowrout.c:383
        depth_ft = (phi - self.node_invert) / LCF
        y1 = torch.clamp(depth_ft[self.u] - geo.offset1, min=0.0)
        y2 = torch.clamp(depth_ft[self.v] - geo.offset2, min=0.0)
        y_mean = torch.clamp(0.5 * (torch.minimum(y1, geo.y_full)
                                    + torch.minimum(y2, geo.y_full)), min=FUDGE)  # :402-416
        y = torch.where(q0 == 0.0, y_mean, y_link)                            # flowrout.c:407, :418
        through = torch.zeros_like(total).index_add(0, self.v, q0).index_add(0, self.u, -q0)
        n_reg = self.edge_u.shape[0] - self.n_pipes
        return {
            f"{self.layer}.phi": phi,
            f"{self.layer}.q": torch.cat([q0, torch.zeros(n_reg, dtype=F64)]),  # link.c:518
            "sewer.junction_depth": depth_j * LCF,
            "sewer.pipe_flow": q0.clone(),
            "sewer.mid_area": area_of_depth(y, geo.y_full, geo.a_full) * _AREA_SI,  # :497
            "sewer.net_inflow": through[jidx],                                # flowrout.c:458
        }

    def overflow(self, before: Mapping, after: Mapping, drivers: Mapping, *,
                 dt: float | None = None) -> Tensor:
        """SWMM's junction overflow (m3/s, >= 0) over the step from `before` to `after`
        (``dt`` s, default the routing step): ``0.5 (N_before + N_after)`` at each junction
        flooded at `after` (its lagged flood excess positive), 0 below SWMM's FUDGE flow
        and elsewhere (`dynwave.c:783-791`); the pass's ``"sewer.overflow"`` fed `after`."""
        dt = self.options.routing_step if dt is None else float(dt)
        drv = self(after | {k: before[k] for k in self.state_keys}, drivers,
                   StepContext(dt=dt))
        return drv["sewer.overflow"]

    def junction_residual(self, before: Mapping, after: Mapping, drivers: Mapping, *,
                          dt: float | None = None) -> Tensor:
        """``0.5 dt (w(H_after) - N_after)`` per junction (m3): the `JunctionStorage`
        withdrawal w (m3/s) against the net inflow N_after (m3/s), both at `after`, with
        the drivers of a pass fed `after` from the step start `before`; ``dt`` (s) defaults
        to the routing step. For a free-surface junction that is ``A_s (y_after -
        y_before) - 0.5 dt (N_before + N_after)``, for an EXTRAN-surcharged one
        ``-0.5 dt N_after``; at a flooded junction w is the flow over its top, so the
        residual is its balance with that outflow (the overflow), ``0.5 dt S (H_after -
        H_top)``. Zero to the iteration tolerance wherever the junction is wet."""
        dt = self.options.routing_step if dt is None else float(dt)
        drv = self(after | {k: before[k] for k in self.state_keys}, drivers,
                   StepContext(dt=dt))
        now = self.carried(after, drivers, dt=dt)
        heads = after[f"{self.layer}.phi"][..., self.junction_idx]
        withdrawal = self.storage.flow(heads, drv)
        return 0.5 * dt * (withdrawal - now["sewer.net_inflow"])


def build_dynamic_wave_model(net: SewerNetwork, *, iterate_tol: float = 1e-11,
                             iterate_max: int = 2000, relaxation: float = 0.5,
                             acceleration: str | None = "anderson", memory: int = 5,
                             iterate_flow_tol: float | None = 1e-10,
                             outfall_stage: Mapping | None = None
                             ) -> tuple[Model, State, Drivers]:
    """Assemble a dynamic-wave sewer and return ``(model, state, drivers)``.

    One `PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m; flows m3/s) over the pipes,
    with the outfalls as its boundary and a `JunctionStorage` at every junction, and the
    `DynamicWaveHydraulics` closure; ``coupling="iterate"`` converges on the junction heads
    to `iterate_tol` (m) within `iterate_max` passes. The passes are Anderson-accelerated
    over the last `memory` of them by default (``acceleration="anderson"``, `Model`'s
    `iterate_acceleration`), with `relaxation` the damping of the steps it falls back to;
    ``acceleration=None`` is plain relaxation, which reaches the same fixed point in two to
    three times the passes. `outfall_stage` gives each TIMESERIES outfall its stage
    elevation: ``{outfall name: stage (m) or a function of the elapsed time t (s) returning
    it}``, required for every such outfall and refused for any other (a TIDAL outfall reads
    its tide curve; `DynamicWaveHydraulics.outfall_stage`); it sets the initial heads (at
    t = 0) and is the default of `simulate_dynamic_wave`'s own `outfall_stage`.
    `iterate_flow_tol` (m3/s) is the tolerance on the conduit
    flows' own fixed-point residual, judged alongside the heads (`Model`'s
    `iterate_flow_tol`), and is required: ``None`` (heads judged alone, as phase A did) is
    refused by name. On the normal-flow slope switch (`DynamicWaveConduit`) the heads
    bound the flow residual only through the wall slope S over the junction's grounding,
    so a heads-only test can stop a pass short of the fixed point (a flow residual of
    3e-5 m3/s was measured on the tree's drain-down). The state is
    SWMM's initial state (`DynamicWaveHydraulics.initial_state`: cold and dry, or its
    InitDepth and InitFlow). Attached to the model: `junction_idx`, `junction_invert` (m),
    `outfall_link_idx` (each outfall's link's ``"water.q"`` column), `outfall_stage` (the
    TIMESERIES outfalls' stage functions), `dynamic_wave` (the
    closure), `routing_step` (s), `pipe_names`,
    `link_names` (every ``"water.q"`` column: the pipes, then the orifices, weirs and
    outlets, each a `StructureLink` edge of its own kind), `pipe_columns`, `notes`,
    `input_specs`, `driver_template`.
    """
    net.validate()
    if net.routing != "dynamic_wave":
        raise ValueError(
            f"build_dynamic_wave_model: network routing is {net.routing!r}, not "
            f"'dynamic_wave'")
    if iterate_flow_tol is None:
        raise ValueError(
            "build_dynamic_wave_model: iterate_flow_tol (m3/s) is required, got None: on the "
            "normal-flow slope switch a heads-only convergence test can stop short of the "
            "fixed point; plain relaxation is acceleration=None with a flow tolerance")
    graph = Network(dtype=F64)
    for manhole in net.manholes:
        graph.add_node(manhole.name, invert=manhole.invert, ground=manhole.ground)
    for outfall in net.outfalls:
        graph.add_node(outfall.name, invert=outfall.invert)
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)
    # the regulators after the pipes, kind by kind: the layer's column order
    for kind in STRUCTURE_KINDS:
        for device in net.structures[kind]:
            graph.add_edge(device.u, device.v, kind=kind, name=device.name)
    boundary = [o.name for o in net.outfalls]
    closure = DynamicWaveHydraulics(graph, net)
    junction_idx = closure.junction_idx
    junction_invert = closure.node_invert[junction_idx]

    def elements():
        return [DynamicWaveConduit(), *(StructureLink(k) for k in closure.structures)]

    # Built once without the node source to learn the interior (a NodeSource must sit on
    # interior nodes), as the Modelica assembly does.
    probe = PotentialFlowLayer(graph, "water", elements(), boundary=boundary,
                               quantity="head", unit="m", flow_unit="m3/s")
    interior = set(probe.interior.tolist())
    stranded = [m.name for m, i in zip(net.manholes, junction_idx.tolist(), strict=True)
                if i not in interior]
    if stranded:
        raise ValueError(f"build_dynamic_wave_model: junction(s) {stranded} touch no link")
    layer = PotentialFlowLayer(
        graph, "water", elements(), boundary=boundary,
        node_sources=[closure.storage],
        quantity="head", unit="m", flow_unit="m3/s")
    if layer.bound.tolist() != closure.outfall_idx.tolist():
        raise ValueError("build_dynamic_wave_model: the layer's boundary order differs from "
                         "the outfall order the closure writes 'water.phi_boundary' in")
    model = Model(graph, {"water": layer}, closures=[closure], coupling="iterate",
                  iterate_tol={"water": iterate_tol}, iterate_max=iterate_max,
                  iterate_relaxation=relaxation, iterate_acceleration=acceleration,
                  iterate_memory=memory,
                  iterate_flow_tol={"water": iterate_flow_tol})
    inflow = torch.zeros(graph.n, dtype=F64)
    inflow[junction_idx] = torch.tensor([m.inflow for m in net.manholes], dtype=F64)
    stage_functions = closure.check_outfall_stage(outfall_stage)
    initial = {"inflow": inflow}
    if bool(closure.outfall_staged.any()):
        initial["outfall_stage"] = closure.outfall_stage(0.0, stage_functions)
    state = closure.initial_state(initial)
    drivers: Drivers = {
        "inflow": inflow,
        "water.phi_boundary": state["water.phi"][closure.outfall_idx].clone(),
    }
    if "outfall_stage" in initial:
        drivers["outfall_stage"] = initial["outfall_stage"]
    closure.notes["routing"] = (
        f"dynamic wave (SWMM 5.2.4 fixed point): heads converged to {iterate_tol} m"
        + f" and flows to {iterate_flow_tol} m3/s"
        + f" within {iterate_max} passes"
        + ("" if acceleration is None else f" ({acceleration}-accelerated)")
        + "; SWMM's HEAD_TOLERANCE and MAX_TRIALS are not used")
    model.notes = closure.notes
    model.junction_idx = junction_idx
    model.junction_invert = junction_invert
    model.outfall_link_idx = closure.outfall_link
    model.dynamic_wave = closure
    model.outfall_stage = stage_functions
    model.routing_step = float(net.dynamic_wave.routing_step)
    model.pipe_names = list(closure.pipe_names)
    model.link_names = list(closure.link_names)
    model.pipe_columns = ("q", "h", "v", "V_wet")
    model.input_specs = {
        "inflow": dict(description="lateral inflow at each manhole", unit="m3/s",
                       over="nodes"),
    }
    if "outfall_stage" in drivers:
        model.input_specs["outfall_stage"] = dict(
            description="stage elevation of each outfall (TIDAL and TIMESERIES outfalls "
                        "read it; simulate_dynamic_wave sets it each step from the tide "
                        "curve or the outfall_stage functions)",
            unit="m", over=list(closure.outfall_names))
    model.driver_template = dict(drivers)
    return model, state, drivers


def solve_atol(model: Model, dt: float | None = None) -> float:
    """The head solve's default absolute tolerance (m3/s) for steps of `dt` s (default: the
    routing step): `SOLVE_ATOL`, raised to 4x the round-off floor of a junction on its stiff
    bed where that is larger. There the Jacobian slope is ``(2/dt)(A_s + K_BED A_bed)``,
    S ~ (2/dt) K_BED A_bed, and Newton cannot move the head by less than half an ulp,
    ``<= eps |H| / 2``, so the residual cannot go below about ``S eps |H| / 2`` -- 6e-10 m3/s
    on a 12 m invert at SWMM's default 12.566 ft2 floor and a 5 s step, giving an atol of
    2.5e-9 m3/s. A wet junction's head is then solved to within ``atol dt / (2 A_s)`` m, at
    most 5e-9 m in that case."""
    dt = model.routing_step if dt is None else float(dt)
    storage = next(ns for ns in model.layers["water"].node_sources
                   if isinstance(ns, JunctionStorage))
    slope = 2.0 * storage.bed_area * (1.0 + K_BED) / dt
    height = float(storage.bed.abs().max())
    floor = slope * torch.finfo(F64).eps * height / 2.0
    return max(SOLVE_ATOL, _BED_FLOOR_MARGIN * floor)


def routing_step_size(hyd: DynamicWaveHydraulics, before: Mapping | None, after: Mapping,
                      previous_dt: float | None) -> float:
    """SWMM's next routing step (s): `dynwave.c::dynwave_getRoutingStep` (dynwave.c:195-220).

    `after` is the state at the end of the step just taken (``previous_dt`` s) and `before`
    the state at its start; ``before=None`` asks for the very first step of the run. With
    ``options.variable_step == 0`` (or a routing step under MINTIMESTEP) the step is the
    fixed ``options.routing_step``; otherwise the first step is ``min_route_step``, and the
    others the smallest of

    - the routing step;
    - each conduit's Courant step ``V / |q| * Fr / (1 + Fr) * variable_step``
      (`getLinkStep`, dynwave.c:836-878), skipped where ``|q|`` or the mid-depth area is
      within FUDGE or ``Fr <= 0.01`` -- a dry, still or full closed conduit never limits
      the step. Flow and volume are one barrel's, and a lengthened conduit's step is
      scaled by its modified length over its own, ``modLength / length``;
    - each junction's depth-change step ``0.25 (crown - invert) / (|dy|/dt)``
      (`getNodeStep`, dynwave.c:881-921), skipped at outfalls, dry junctions, junctions
      within FUDGE of their crown or above it, and where ``|dy|/dt`` is within FUDGE;

    floored at `min_route_step` (dynwave.c:828) and truncated to a millisecond
    (dynwave.c:218). The Froude number, flow, volume and mid-depth area are those of the
    converged state `after` -- SWMM's last trial of the step, equal to it at its
    convergence; ``|dy|`` is ``|y_after - y_before|`` with the depths in `before` and
    `after` (``sewer.junction_depth``), SWMM's ``Xnode.dYdT`` (dynwave.c:758).
    Not differentiable (the step is a switch of the state). Single network only (the
    state's tensors are 1-D).
    """
    opts = hyd.options
    fixed = opts.routing_step
    if opts.variable_step == 0.0 or fixed < MIN_TIME_STEP:                   # dynwave.c:204-205
        return fixed
    if before is None:                                                       # dynwave.c:209-212
        step = opts.min_route_step
    else:
        step = _variable_step(hyd, before, after, float(previous_dt), fixed)
    return math.floor(1000.0 * step) / 1000.0                                # dynwave.c:218


def _variable_step(hyd: DynamicWaveHydraulics, before: Mapping, after: Mapping,
                   dt: float, max_step: float) -> float:
    """`dynwave.c::getVariableStep` (dynwave.c:799-833), `max_step` s, `dt` the last step."""
    with torch.no_grad():
        terms = hyd._terms(after, dt)
        barrels = hyd.geo.barrels
        q = after[f"{hyd.layer}.q"][..., :hyd.n_pipes].abs() / QCF / barrels  # dynwave.c:852
        fr = terms.froude
        live = (q > FUDGE) & (terms.mid_area > FUDGE) & (fr > 0.01)          # dynwave.c:853-856
        residence = (terms.volume / barrels
                     / torch.where(live, q, torch.ones_like(q)))             # dynwave.c:859
        residence = residence * hyd.geo.mod_length / hyd.geo.length          # dynwave.c:865
        courant = residence * fr / (1.0 + fr) * hyd.options.variable_step    # dynwave.c:861-862
        t_link = torch.where(live, courant, torch.full_like(courant, max_step))
        t_min = min(max_step, float(t_link.min())) if t_link.numel() else max_step
        depth = after["sewer.junction_depth"] / LCF
        rate = (depth - before["sewer.junction_depth"] / LCF).abs() / dt     # dynwave.c:758
        crown = hyd.crown_ft
        allowed = 0.25 * crown                                               # dynwave.c:904
        live = ((depth > FUDGE) & (depth + FUDGE < crown) & (allowed >= FUDGE)
                & (rate >= FUDGE))                                           # dynwave.c:897-910
        t_node = allowed / torch.where(live, rate, torch.ones_like(rate))    # dynwave.c:913
        t_node = torch.where(live, t_node, torch.full_like(t_node, t_min))
        if t_node.numel():
            t_min = min(t_min, float(t_node.min()))
    return max(t_min, hyd.options.min_route_step)                            # dynwave.c:828


def simulate_dynamic_wave(model: Model, state: State, drivers: Drivers, *,
                          steps: int | None = None, duration: float | None = None,
                          dt: float | None = None,
                          inflow: Tensor | Callable[[float], Tensor] | None = None,
                          outfall_stage: Mapping | None = None,
                          record: tuple[str, ...] = ("water.phi", "water.q"),
                          diagnostics: list | None = None, **solve_kwargs
                          ) -> tuple[State, dict[str, Tensor]]:
    """Run the dynamic wave from `state` for `steps` steps, or for `duration` s.

    Exactly one of `steps` and `duration` is required. Under a variable step one call is one
    run: the first step is the minimum step and `t` starts at 0, so a continuation call
    restarts as a new SWMM run would (the step rule needs the previous step's start state).
    The step is `dt` s (default: the model's routing step) when
    ``options.variable_step == 0``; under ``variable_step > 0``
    each step is SWMM's variable step (`routing_step_size`: the first is the minimum step)
    and `dt` is refused, since it would be a second step rule. `duration` (s) is SWMM's
    run length: steps are taken until the elapsed time (kept in milliseconds, as SWMM does,
    `swmm5.c:530-545`) reaches it, and the last step is cut to the time that remains (at
    least 1 ms), so the run ends exactly at `duration` whatever the step rule.

    `inflow` (m3/s, full-node) overrides ``drivers["inflow"]``: ``(n,)`` is one constant
    inflow; ``(B, n)`` a batch of B constant inflows -- a 2-D inflow is ALWAYS a batch,
    never a series; ``(steps, B, n)`` the inflow of each step for a batch of B (one
    network's series is ``(steps, 1, n)``; needs `steps`); or a callable ``inflow(t)``
    returning ``(n,)`` or ``(B, n)`` for the step that starts at `t` s (a float). SWMM
    takes a step's lateral inflow from its time series at the START of the step
    (`routing.c:223, 368`: the date is read before the routing clock advances; the inflow
    enters the trapezoid with the step's end state, and the previous step's final net
    inflow is its start), so ``inflow(t)`` is called once per step, with the step's start
    time. (SWMM's own clock reads 1 ms late, ``getDateTime`` adds 1 ms: `swmm5.c:1551`, so
    a [TIMESERIES] is sampled at ``t + 0.001`` s; a function standing in for one must add
    that, as the verification test does.) A batched inflow expands an unbatched `state` to
    the batch (a batch cannot be combined with a variable step: each member would need its
    own). After every
    step the carried keys are re-evaluated at the returned state. `record` may also name
    ``"sewer.overflow"`` (m3/s per junction, SWMM's flooding loss over each step:
    `DynamicWaveHydraulics.overflow`), which is not a state key, and the closure's
    reporting keys (``"sewer.storage_volume"``, ``"sewer.V_wet"``, ``"sewer.h"``, ...), from
    its query at each step's end state.
    `outfall_stage` ``{outfall name: stage (m) or a function of t (s) returning it}`` gives
    the TIMESERIES outfalls their stage elevation (default: the model's own, the
    `build_model` argument); a TIDAL outfall reads its tide curve (`tidal_stage`). Both
    become the driver ``"outfall_stage"`` (m, per outfall) of each step, and share the
    FIXED outfall's stage rule. SWMM sets the outfall depth with the routing clock already
    advanced (`routing.c:301-302`; node.c:1449, 1456), so the stage is that at the END of
    the step: a function is called once per step with the step's end time in s -- not the
    start time `inflow` gets, and with no 1 ms offset (`NewRoutingTime`, not `getDateTime`).
    It works under fixed and variable steps alike.
    `diagnostics`, when a list, receives each step's `Model.step` diagnostics.
    `solve_kwargs` reach the head solve (Newton); `atol` defaults to `solve_atol` for each
    step's length (m3/s: `SOLVE_ATOL`, or the bed's round-off floor where larger) and
    `rtol` to 0, because Newton's own default (sqrt(eps), 1.5e-8 m3/s) would stop a
    warm-started pass before it moves and leave the junction balance off by that much.
    Returns the final state and the recorded `record` keys stacked over steps (leading
    dimension: the steps), plus ``"t"`` (s, the END time of each step) and ``"dt"`` (s, each
    step's length).
    """
    hyd = model.dynamic_wave
    if (steps is None) == (duration is None):
        raise ValueError("simulate_dynamic_wave: give exactly one of steps and duration "
                         f"(got steps={steps!r}, duration={duration!r})")
    if duration is not None and not duration > 0:
        raise ValueError(f"simulate_dynamic_wave: duration must be > 0 s, got {duration!r}")
    variable = hyd.options.variable_step > 0
    if variable and dt is not None:
        raise ValueError(
            f"simulate_dynamic_wave: dt={dt!r} s given, but options.variable_step = "
            f"{hyd.options.variable_step} chooses every step (SWMM's variable routing step); "
            f"set variable_step to 0 for a fixed dt")
    fixed_dt = model.routing_step if dt is None else float(dt)
    staged = bool(hyd.outfall_staged.any())
    stage_functions = hyd.check_outfall_stage(
        model.outfall_stage if outfall_stage is None else outfall_stage)
    atol_given = "atol" in solve_kwargs
    solve_kwargs.setdefault("rtol", 0.0)
    n = model.net.n
    series = None
    if callable(inflow):
        step_inflow = inflow
    elif inflow is None:
        step_inflow = None
    else:
        series = torch.as_tensor(inflow, dtype=F64)
        if series.dim() not in (1, 2, 3) or series.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow must be (n,), (B, n) or (steps, B, n) with "
                f"n = {n} nodes, got shape {tuple(series.shape)}")
        if series.dim() == 3:
            if steps is None:
                raise ValueError("simulate_dynamic_wave: an inflow series (steps, B, n) "
                                 "needs steps=, not duration=")
            if series.shape[0] < steps:
                raise ValueError(f"simulate_dynamic_wave: inflow has {series.shape[0]} "
                                 f"steps, {steps} asked")
        step_inflow = None

    def inflow_at(k: int, t: float) -> Tensor | None:
        if series is not None:
            return series[k] if series.dim() == 3 else series
        if step_inflow is None:
            return None
        value = torch.as_tensor(step_inflow(t), dtype=F64)
        if value.dim() not in (1, 2) or value.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow(t={t}) must return (n,) or (B, n) with "
                f"n = {n} nodes, got shape {tuple(value.shape)}")
        return value

    first = inflow_at(0, 0.0)    # step 0's inflow, reused in the loop; it also sets the batch
    batch: tuple[int, ...] = ()
    if first is not None and first.dim() > 1:
        batch = tuple(first.shape[:-1])
    if variable and batch:
        raise ValueError("simulate_dynamic_wave: a batched inflow cannot be combined with "
                         "options.variable_step > 0 (one step for every member)")
    state = dict(state)
    if batch:
        state = {k: (v.expand(batch + v.shape[-1:]).clone() if v.dim() == 1 else v)
                 for k, v in state.items()}
    rows: dict[str, list[Tensor]] = {key: [] for key in record}
    ends: list[float] = []
    lengths: list[float] = []
    now_ms = 0.0                                          # swmm5.c: NewRoutingTime, ms
    end_ms = None if duration is None else 1000.0 * float(duration)
    previous: tuple[State, float] | None = None
    k = 0
    while (k < steps) if steps is not None else (now_ms < end_ms):
        if variable:
            step = routing_step_size(hyd, None if previous is None else previous[0], state,
                                     None if previous is None else previous[1])
        else:
            step = fixed_dt
        next_ms = now_ms + 1000.0 * step                                    # swmm5.c:530
        if end_ms is not None and next_ms > end_ms:                         # swmm5.c:531-536
            step = max((end_ms - now_ms) / 1000.0, 1.0 / 1000.0)
            next_ms = end_ms
        t = now_ms / 1000.0
        if inflow is not None:
            drivers = dict(drivers, inflow=first if k == 0 else inflow_at(k, t))
        if staged:                                    # node.c:1449, 1456: the step's END
            drivers = dict(drivers,
                           outfall_stage=hyd.outfall_stage(next_ms / 1000.0, stage_functions))
        kwargs = dict(solve_kwargs)
        if not atol_given:
            kwargs["atol"] = solve_atol(model, step)
        diag: dict = {}
        before = state
        try:
            state = model.step(state, drivers, step, t=t, diagnostics=diag, **kwargs)
        except RuntimeError as err:
            raise RuntimeError(f"dynamic wave step at t={t} s: {err}") from err
        state.update(hyd.carried(state, drivers, dt=step))  # resync, as modelica/run.py does
        previous = (before, step)
        now_ms = next_ms
        k += 1
        ends.append(now_ms / 1000.0)
        lengths.append(step)
        if diagnostics is not None:
            diagnostics.append(diag)
        report = None
        for key in record:
            if key == "sewer.overflow":
                rows[key].append(hyd.overflow(before, state, drivers, dt=step))
            elif key in state:
                rows[key].append(state[key])
            else:
                if report is None:
                    report = hyd(state, drivers)
                if key not in report:
                    raise KeyError(f"simulate_dynamic_wave: record names {key!r}, which is "
                                   f"neither a state key nor a closure reporting key")
                rows[key].append(report[key])
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["t"] = torch.tensor(ends, dtype=F64)
    out["dt"] = torch.tensor(lengths, dtype=F64)
    return state, out
