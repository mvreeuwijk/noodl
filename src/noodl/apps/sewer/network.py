"""`SewerNetwork`, its validation, and the (Model, State, Drivers) builder.

Follows `apps/street_aq/network.py`'s builder shape: dataclasses describing the physical
network, one `validate()` that refuses every topology this application does not model BY NAME,
one `build_model` that assembles the `Network`, the layers and the closures and
returns the triple, and one `initial_state` dispatching on each layer's `quantity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from noodl.apps.sewer.air import (
    F_AIR_DEFAULT,
    F_I_DEFAULT,
    RHO_AIR_REF,
    Drag,
    Headspace,
)
from noodl.apps.sewer.hydraulics import SewerHydraulics
from noodl.apps.sewer.quality import H2STransfer, LateralLoads, SulfideGeneration
from noodl.drives import Stack
from noodl.elements.fixed import FixedFlow
from noodl.elements.powerlaw import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import TransportLayer, active_interior
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]


@dataclass(frozen=True)
class Manhole:
    name: str
    invert: float
    ground: float | None = None
    inflow: float = 0.0
    surface_area: float | None = None


@dataclass(frozen=True)
class Outfall:
    name: str
    invert: float


@dataclass(frozen=True)
class Pipe:
    name: str
    u: str
    v: str
    length: float
    diameter: float
    n: float
    slope: float


@dataclass(frozen=True)
class SewerNetwork:
    """A dendritic gravity sewer: manholes, one outgoing pipe each, outfalls."""

    manholes: tuple[Manhole, ...]
    pipes: tuple[Pipe, ...]
    outfalls: tuple[Outfall, ...]
    routing: str = "KINWAVE"
    notes: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Refuse, BY NAME, everything this application does not model."""
        names: set[str] = set()
        for node in (*self.manholes, *self.outfalls):
            if node.name in names:
                raise ValueError(f"SewerNetwork: duplicate node name {node.name!r}")
            names.add(node.name)
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
            outgoing.setdefault(pipe.u, []).append(pipe.name)
        outfall_names = {o.name for o in self.outfalls}
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
    air: bool = True,
    quality: bool = True,
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
) -> tuple[Model, State, Drivers]:
    """Assemble the sewer model and return `(model, state, drivers)`.

    `air=False` builds the water-only model the pyswmm parity rows use; `quality=False`
    drops both quality layers. `fans` names the manholes carrying a prescribed extraction
    (m3/s, positive OUT through the `fan` kind).
    """
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

    n_pipes = len(net.pipes)
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
    length = torch.tensor([p.length for p in net.pipes], dtype=F64)
    diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
    slope_per_manhole = torch.tensor(
        [net.pipes[outgoing[name]].slope for name in manhole_names], dtype=F64
    )

    closures: list = []
    hydraulics = SewerHydraulics(
        graph, list(net.pipes), list(net.manholes), storage=storage, dt=dt_storage,
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
        # Kind order: `headspace` (one per pipe, then one outfall edge PER outfall pipe),
        # `leak`, `fan`. Each outfall pipe is the one that DRAINS TO an outfall
        # (`p.v in outfall_names`), found by where it drains rather than assumed to be
        # `net.pipes[-1]` (otherwise a network whose outfall conduit is listed
        # first would silently attach the outfall-flagged headspace edge, and its geometry, to
        # the wrong manhole). `SewerNetwork.validate()` guarantees every manhole has exactly
        # one outgoing pipe and every outfall has none, but does not itself forbid two
        # different manholes both draining directly into the same outfall; this builder
        # wires exactly ONE outfall-to-ambient headspace edge per OUTFALL NODE, so that case
        # is refused here by name instead of silently wiring only one of the two. A forest
        # has one outfall pipe per component -- each gets its
        # own appended headspace edge, borrowing its own pipe's own length and diameter.
        outfall_names = {o.name for o in net.outfalls}
        outfall_pipe_candidates = [
            i for i, p in enumerate(net.pipes) if p.v in outfall_names
        ]
        if not outfall_pipe_candidates:
            raise ValueError(
                f"build_model: no pipe drains to an outfall; outfalls are "
                f"{sorted(outfall_names)}"
            )
        drained = [net.pipes[i].v for i in outfall_pipe_candidates]
        if len(drained) != len(set(drained)):
            dupes = sorted({v for v in drained if drained.count(v) > 1})
            raise ValueError(
                f"build_model: outfall(s) {dupes} each have more than one pipe "
                f"draining into them; a dendritic sewer (or forest) gives each outfall "
                f"exactly one"
            )
        pipe_positions = list(range(n_pipes)) + outfall_pipe_candidates
        outfall_pipe_idx = torch.tensor(outfall_pipe_candidates, dtype=torch.long)
        n_outfall_edges = len(outfall_pipe_candidates)
        for pipe in net.pipes:
            graph.add_edge(pipe.u, pipe.v, kind="headspace", name=f"{pipe.name}.air")
        for outfall_pipe in outfall_pipe_candidates:
            graph.add_edge(
                net.pipes[outfall_pipe].u, "ambient", kind="headspace", outfall=True
            )
        for manhole in net.manholes:
            graph.add_edge(manhole.name, "ambient", kind="leak", name=f"{manhole.name}.leak")
        for name in fans:
            graph.add_edge(name, "ambient", kind="fan", name=f"{name}.fan")
        air_lengths = torch.cat([length, length.index_select(-1, outfall_pipe_idx)])
        positions = torch.tensor(pipe_positions, dtype=torch.long)
        # Leak law: `PowerLaw(C = Cd A sqrt(2/rho), n = 0.5, kind="leak")` built DIRECTLY,
        # never through `Orifice`: `Orifice` casts with `torch.get_default_dtype()`
        # (float32 in this repository) regardless of its inputs' own dtype, which would
        # downcast the leak's `C` and floor check C1's residual on the float32 arithmetic of
        # the leak branch alone (measured 1.352e-12 / 1.855e-11 in float32; 4.518e-13 /
        # 6.141e-12 in float64). `leak_cd`/`leak_area` are ordinary floats by
        # default but may also be tensors carrying a leading batch (instance) dimension
        # (`benchmarks/sewer_diurnal.py` varies them per instance without any core or element
        # change, since `PowerLaw._param`'s pass-through rule and ordinary broadcasting
        # do the rest).
        leak_c = (
            torch.as_tensor(leak_cd, dtype=F64)
            * torch.as_tensor(leak_area, dtype=F64)
            * math.sqrt(2.0 / RHO_AIR_REF)
            * torch.ones(len(net.manholes), dtype=F64)
        )
        leak_n = torch.full((len(net.manholes),), 0.5, dtype=F64)
        elements = [
            Headspace(air_lengths, positions, f_air=f_air, kind="headspace"),
            PowerLaw(C=leak_c, n=leak_n, kind="leak", dp_transition=1e-8),
        ]
        if fans:
            elements.append(FixedFlow(torch.zeros(len(fans), dtype=F64), kind="fan"))
        drives = [
            Drag(
                air_lengths, positions, f_i=f_i, c_s=c_s,
                zero_positions=tuple(range(n_pipes, n_pipes + n_outfall_edges)),
                kind="headspace",
            ),
            _stack_for(
                "headspace", graph, net,
                crown=torch.cat([diameter, diameter.index_select(-1, outfall_pipe_idx)]),
            ),
            _stack_for("leak", graph, net),
        ]
        layers["air"] = PotentialFlowLayer(
            graph, "air", elements, drives=drives, boundary=["ambient"],
            quantity="pressure", unit="Pa",
        )
        drivers["air.phi_boundary"] = torch.zeros(1, dtype=F64)

    if quality:
        water_cap = torch.ones(len(net.manholes), dtype=F64)
        layers["water_quality"] = TransportLayer(
            graph, "water_quality", capacity=water_cap, flow_kind="pipe",
            boundary=[o.name for o in net.outfalls], n_species=len(species),
            scheme=scheme, quantity="concentration", unit="kg/m3",
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
                quantity="concentration", unit="kg/m3",
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
    model = Model(
        graph, layers, closures=closures, reactions=reactions, coupling=coupling,
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
    return model, state, drivers


def _ground_of(node_attrs) -> float:
    """A node's ground level: its own `ground`, else its `invert`, else `0.0` (`ambient`'s
    case, carrying neither). `is None` throughout, not an `or`-chain -- `or` treats an
    explicitly given `0.0` ground (or invert) as falsy and falls through to the next
    fallback, silently dropping a real ground-level-zero manhole to whatever its invert (or
    the hard-coded `0.0`) happens to be instead."""
    ground = node_attrs.get("ground")
    if ground is not None:
        return float(ground)
    invert = node_attrs.get("invert")
    return 0.0 if invert is None else float(invert)


def _stack_for(
    kind: str, graph: Network, net: SewerNetwork, *, crown: Tensor | None = None
) -> Stack:
    """The existing buoyancy drive on one air kind.

    On `headspace` edges `z_path` is the pipe CROWN elevation: the mean of the two end
    inverts plus the pipe's own diameter (the mean invert alone is
    the pipe's INVERT at its midpoint, not its crown; `crown` is the per-headspace-edge
    diameter, in the same order as `graph.endpoints("headspace")`, i.e. `net.pipes` order
    then the outfall pipe's diameter again for the extra outfall-to-ambient edge, exactly
    like `air_lengths` at the call site). `z_ref` is the node's ground level, so a warm
    headspace column in a manhole shaft is lighter than the outside column of the same
    height -- the building application's stack effect, unchanged.

    DATUM CONVENTION (a leak DOES carry a stack term). `ground` falls back to the node's own
    `invert` when no `ground` was given, and to `0.0` when neither was given -- which is exactly
    `ambient`'s case, since `ambient` is added as a bare node with no `invert` or `ground` attribute
    at all. On a `leak` edge `z_path` is the manhole's own ground level (`ground[src]`), so the
    MANHOLE term of the Stack law cancels (`z_ref[src] - z_path = 0`), but the AMBIENT term does
    not: `z_ref` at `ambient` is `0.0`, not the manhole's ground level, so every leak in fact
    carries a `g * rho_ambient * ground[manhole]` term, not zero. This is self-consistent rather
    than a bug: `ambient`'s `0.0` datum is used identically at every leak and at the outfall's
    open-air `headspace` edge, and `ambient` is itself a fixed boundary node
    (`air.phi_boundary = 0`), so the datum choice only ever shifts every air pressure by the
    same constant -- `air.phi` is DATUM-REFERENCED to this convention, not to a physical
    zero, and differences between manholes (what every drive and refusal actually reads)
    are unaffected. `crown` is unused (and must be `None`) for `kind='leak'`.
    """
    if kind != "headspace" and crown is not None:
        raise ValueError(
            f"_stack_for: crown is only meaningful for kind='headspace', got kind={kind!r}"
        )
    src, tgt = graph.endpoints(kind)
    invert = torch.tensor(
        [
            0.0 if (v := graph.graph.nodes[n].get("invert")) is None else v
            for n in graph.nodes
        ],
        dtype=F64,
    )
    ground = torch.tensor(
        [_ground_of(graph.graph.nodes[n]) for n in graph.nodes],
        dtype=F64,
    )
    if kind == "headspace":
        if crown is None:
            raise ValueError("_stack_for: kind='headspace' requires the crown diameters")
        z_path = 0.5 * (invert[src] + invert[tgt]) + crown
    else:
        z_path = ground[src]
    return Stack(kind, src=src, tgt=tgt, z_path=z_path, z_ref=ground,
                 rho_key="rho_air_nodes", g=9.80665)


def initial_state(model: Model, drivers: Drivers | None = None) -> State:
    """All-zero state, dispatching on each layer's `quantity` (the app convention).

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
    if drivers is not None:
        state.update(model.initial_capacities(state, drivers))
    elif model.transport and any(isinstance(c, SewerHydraulics) for c in model.closures):
        raise ValueError(
            "apps.sewer.initial_state: drivers are required to evaluate the initial "
            "storage; call initial_state(model, drivers)"
        )
    return state


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
