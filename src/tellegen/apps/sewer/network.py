"""`SewerNetwork`, its validation, and the (Model, State, Drivers) builder.

Follows `apps/street/network.py`'s builder shape: dataclasses describing the physical
network, one `validate()` that refuses every topology this milestone does not model BY NAME,
one `build_sewer_model` that assembles the `Network`, the layers and the closures and
returns the triple, and one `initial_state` dispatching on each layer's `quantity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tellegen.apps.sewer.air import (
    F_AIR_DEFAULT,
    F_I_DEFAULT,
    RHO_AIR_REF,
    Drag,
    Headspace,
)
from tellegen.apps.sewer.hydraulics import SewerHydraulics
from tellegen.apps.sewer.quality import H2STransfer, SulfideGeneration
from tellegen.drives import Stack
from tellegen.elements.fixed import FixedFlow
from tellegen.elements.powerlaw import Orifice
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.layers.transport import TransportLayer, active_interior
from tellegen.model import Model
from tellegen.topology import Network

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
        """Refuse, BY NAME, everything this milestone does not model."""
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
        from tellegen.apps.sewer.hydraulics import _levels

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


def build_sewer_model(
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
    # `tree_steady()` this happens to be the identity, but `read_inp` reads `[CONDUITS]` in
    # its own file order, which need not match `[JUNCTIONS]` order at all -- the committed
    # `tree_steady.inp` gives `out_pipe = [0, 1, 3, 2, 4]` (M4-R4 amendment).
    out_pipe = torch.tensor([outgoing[name] for name in manhole_names], dtype=torch.long)
    length = torch.tensor([p.length for p in net.pipes], dtype=F64)
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
        # Kind order: `headspace` (one per pipe, then the outfall edge), `leak`, `fan`.
        outfall_manhole = net.pipes[-1].u
        pipe_positions = list(range(n_pipes)) + [n_pipes - 1]
        for pipe in net.pipes:
            graph.add_edge(pipe.u, pipe.v, kind="headspace", name=f"{pipe.name}.air")
        graph.add_edge(outfall_manhole, "ambient", kind="headspace", outfall=True)
        for manhole in net.manholes:
            graph.add_edge(manhole.name, "ambient", kind="leak", name=f"{manhole.name}.leak")
        for name in fans:
            graph.add_edge(name, "ambient", kind="fan", name=f"{name}.fan")
        air_lengths = torch.cat([length, length[-1:]])
        positions = torch.tensor(pipe_positions, dtype=torch.long)
        elements = [
            Headspace(air_lengths, positions, f_air=f_air, kind="headspace"),
            Orifice(
                torch.full((len(net.manholes),), leak_cd, dtype=F64),
                torch.full((len(net.manholes),), leak_area, dtype=F64),
                rho=RHO_AIR_REF, kind="leak", dp_transition=1e-8,
            ),
        ]
        if fans:
            elements.append(FixedFlow(torch.zeros(len(fans), dtype=F64), kind="fan"))
        drives = [
            Drag(
                air_lengths, positions, f_i=f_i, c_s=c_s,
                zero_positions=(n_pipes,), kind="headspace",
            ),
            _stack_for("headspace", graph, net),
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
        # with `manhole_names` exactly (the same self-check `apps/street/network.py`
        # performs) -- checked ONCE, here, rather than trusted (M4-R4/M4-R9 amendment).
        ordered = [graph.nodes[i] for i in layers["water_quality"].interior_idx.tolist()]
        if ordered != manhole_names:
            raise ValueError(
                f"build_sewer_model: the water_quality layer's active interior came out as "
                f"{ordered}, not the manhole order {manhole_names}; out_pipe indexes by "
                f"position into manhole_names and assumes they agree"
            )
        drivers["water_quality.x_boundary"] = torch.zeros(
            len(net.outfalls), len(species), dtype=F64
        )
        if air:
            # `active_interior` calls `Network.endpoints` on every kind named, which raises
            # if the network carries no edge of that kind at all -- not merely an empty
            # interior. `fan` edges exist only when `fans` is non-empty, so the kind tuple
            # must match `flow_kind`'s below exactly rather than naming `fan` unconditionally
            # (the brief's own text would otherwise raise `KeyError: unknown edge kind
            # 'fan'` on every fan-less network, including the dictated fixture's default).
            air_kinds = ("headspace", "leak", "fan") if fans else ("headspace", "leak")
            # Every pipe, including the one draining to the outfall, gets a PARALLEL
            # headspace edge (Conventions block), so the outfall NODE itself (not just
            # "ambient") is touched by a headspace edge -- it carries the outfall
            # structure's own headspace air. `H2STransfer` reads `air_quality.x` directly as
            # a per-MANHOLE array (matched against `manhole_idx`/`out_pipe`), so the outfall
            # node must be a BOUNDARY of this layer too, exactly like water_quality, or the
            # layer's active interior would be 6 nodes (5 manholes + the outfall) where 5
            # are wanted; `two_film_flux` would then fail on a shape mismatch against
            # water_quality's 5-node interior (found by running this task's tests).
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
                    f"build_sewer_model: the air_quality layer's active interior came out "
                    f"as {ordered}, not the manhole order {manhole_names}; H2STransfer "
                    f"reads 'air_quality.x' as a per-manhole array and assumes they agree"
                )
            drivers["air_quality.x_boundary"] = torch.zeros(len(air_boundary), dtype=F64)
            closures.append(H2STransfer(graph.n, manhole_idx, out_pipe=out_pipe))

    reactions = (
        [("water_quality", SulfideGeneration(out_pipe=out_pipe))] if quality else []
    )
    model = Model(
        graph, layers, closures=closures, reactions=reactions, coupling=coupling,
        iterate_tol=(
            {"water_quality": 1e-12, "air_quality": 1e-12}
            if coupling == "iterate" and quality
            else None
        ),
    )
    model.notes = dict(hydraulics.notes)

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
    state.update(initial_state(model))
    if storage:
        state["sewer.H"] = torch.zeros(len(net.manholes), dtype=F64)
    model.out_pipe = out_pipe
    model.manhole_idx = manhole_idx
    model.pipe_names = [p.name for p in net.pipes]
    return model, state, drivers


def _stack_for(kind: str, graph: Network, net: SewerNetwork) -> Stack:
    """The existing buoyancy drive on one air kind (spec 3.3).

    On `headspace` edges `z_path` is the pipe crown elevation and `z_ref` the node's ground
    level, so a warm headspace column in a manhole shaft is lighter than the outside column
    of the same height -- the building application's stack effect, unchanged. On `leak`
    edges `z_path` is the cover's own level and `z_ref` the same at both ends, so the leak
    path itself carries no stack term.
    """
    src, tgt = graph.endpoints(kind)
    invert = torch.tensor(
        [graph.graph.nodes[n].get("invert", 0.0) or 0.0 for n in graph.nodes], dtype=F64
    )
    ground = torch.tensor(
        [
            graph.graph.nodes[n].get("ground") or graph.graph.nodes[n].get("invert", 0.0)
            or 0.0
            for n in graph.nodes
        ],
        dtype=F64,
    )
    z_path = 0.5 * (invert[src] + invert[tgt]) if kind == "headspace" else ground[src]
    return Stack(kind, src=src, tgt=tgt, z_path=z_path, z_ref=ground,
                 rho_key="rho_air_nodes", g=9.80665)


def initial_state(model: Model) -> State:
    """All-zero state, dispatching on each layer's `quantity` (the app convention)."""
    state: State = {}
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
