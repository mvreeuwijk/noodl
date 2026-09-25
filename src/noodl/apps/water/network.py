"""`WaterNetwork`, its validation, and the (Model, State, Drivers) builder.

A pressurised distribution network is exactly the framework's own case: every pipe is full,
the head loss is a monotone function of the head difference, and the whole thing is ONE
`PotentialFlowLayer` whose potential is hydraulic HEAD in metres. That is the difference
from the free-surface sewer of `apps.sewer`, whose normal flow is set by slope and upstream
inflow rather than by head difference.

Node order: junctions in file order, then reservoirs, then tanks. Boundary = every reservoir
then every tank, in that order, which is the order `"water.phi_boundary"` is written in.
Edge kinds partition by device type: `pipe`, `pump`, `valve` (a TCV, i.e. a settable minor
loss) and `fcv` (a `FixedFlow`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from noodl.apps.water.demand import PressureDrivenDemand
from noodl.apps.water.elements import (
    HazenWilliams,
    MinorLoss,
    PumpCurve,
    three_point_curve,
)
from noodl.apps.water.tanks import Control, TankLevels
from noodl.elements.duct import Duct
from noodl.elements.fixed import FixedFlow
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: Water at 20 C (spec section 11, verified standard): density (kg/m3) and dynamic
#: viscosity (Pa s). Used only by the Darcy-Weisbach path, which works in PRESSURE.
RHO_W = 998.2
MU_W = 1.002e-3
G = 9.80665

#: Valve types this milestone models, and the ones it refuses (spec 13.5). PRV, PSV and PBV
#: switch status (open / closed / active) with the explicit if/else logic the EPANET 2.2
#: Manual states on p.113, which is not differentiable without a smoothed relaxation; GPV
#: needs a tabulated head-loss curve. All four are recorded follow-ups.
_VALVES_MODELLED = ("TCV", "FCV")
_VALVES_REFUSED = ("PRV", "PSV", "PBV", "GPV")


@dataclass(frozen=True)
class WaterOptions:
    """`[OPTIONS]` keys that change the physics, with EPANET 2.2's own defaults.

    `demand_model` is `"DDA"` or `"PDA"`; `minimum_pressure`/`required_pressure` are in
    METRES of head (this reader converts a US file's native psi at the same boundary as
    every other length, research note section 9's `PSI_TO_M`); `pressure_exponent` is
    dimensionless; `specific_gravity`/`viscosity` are relative to water at 20 C. The
    `required_pressure` default 0.1 is EPANET's own lower limit on the value ("0.1 in psi
    or m", `wntr`'s inp writer), not the manual OPTIONS table's literal 0.0, which only
    describes the no-op DDA case; `wntr`'s `HydraulicOptions.required_pressure = 0.07` is
    that SAME 0.1 converted from psi assuming US units, which does not apply here since
    this reader is always SI.
    """

    demand_model: str = "DDA"
    minimum_pressure: float = 0.0
    required_pressure: float = 0.1
    pressure_exponent: float = 0.5
    specific_gravity: float = 1.0
    viscosity: float = 1.0


@dataclass(frozen=True)
class Junction:
    name: str
    elevation: float
    demand: float = 0.0
    pattern: str | None = None


@dataclass(frozen=True)
class Reservoir:
    name: str
    head: float
    pattern: str | None = None


@dataclass(frozen=True)
class Tank:
    name: str
    elevation: float
    init_level: float
    min_level: float
    max_level: float
    diameter: float


@dataclass(frozen=True)
class WaterPipe:
    name: str
    u: str
    v: str
    length: float
    diameter: float
    roughness: float
    minor_loss: float = 0.0
    status: str = "OPEN"


@dataclass(frozen=True)
class Pump:
    name: str
    u: str
    v: str
    curve: str | None
    power: float | None = None
    speed: float = 1.0


@dataclass(frozen=True)
class Valve:
    name: str
    u: str
    v: str
    diameter: float
    kind: str
    setting: float
    minor_loss: float = 0.0


@dataclass(frozen=True)
class WaterNetwork:
    """A pressurised distribution network, in SI, with everything the builder needs."""

    junctions: tuple[Junction, ...]
    reservoirs: tuple[Reservoir, ...] = ()
    tanks: tuple[Tank, ...] = ()
    pipes: tuple[WaterPipe, ...] = ()
    pumps: tuple[Pump, ...] = ()
    valves: tuple[Valve, ...] = ()
    curves: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    patterns: dict[str, tuple[float, ...]] = field(default_factory=dict)
    controls: tuple[Control, ...] = ()
    demand_pattern: str | None = None
    pattern_timestep: float = 3600.0
    hydraulic_timestep: float = 3600.0
    report_timestep: float = 3600.0
    duration: float = 0.0
    headloss: str = "H-W"
    options: WaterOptions = field(default_factory=WaterOptions)
    notes: dict[str, str] = field(default_factory=dict)

    def nodes(self) -> list[str]:
        """Node names in the builder's own order: junctions, reservoirs, tanks."""
        return (
            [j.name for j in self.junctions]
            + [r.name for r in self.reservoirs]
            + [t.name for t in self.tanks]
        )

    def links(self) -> list:
        """Every link, in the kind order the layer's element list uses."""
        tcv = [v for v in self.valves if v.kind.upper() == "TCV"]
        fcv = [v for v in self.valves if v.kind.upper() == "FCV"]
        return [*self.pipes, *self.pumps, *tcv, *fcv]

    def validate(self) -> None:
        """Refuse, BY NAME, everything this milestone does not model (spec 13.5)."""
        names = self.nodes()
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"WaterNetwork: duplicate node name {name!r}")
            seen.add(name)
        if not self.reservoirs and not self.tanks:
            raise ValueError(
                "WaterNetwork: the network has no reservoir and no tank, so no node's head "
                "is fixed and the nodal system is singular; EPANET requires at least one "
                "fixed-head node too"
            )
        link_names: set[str] = set()
        for link in self.links():
            if link.name in link_names:
                raise ValueError(f"WaterNetwork: duplicate link name {link.name!r}")
            link_names.add(link.name)
            for end in (link.u, link.v):
                if end not in seen:
                    raise ValueError(
                        f"WaterNetwork: link {link.name!r} names node {end!r}, which is "
                        f"neither a junction, a reservoir nor a tank"
                    )
        for pipe in self.pipes:
            if not (pipe.length > 0 and pipe.diameter > 0 and pipe.roughness > 0):
                raise ValueError(
                    f"WaterNetwork: pipe {pipe.name!r} must have positive length, diameter "
                    f"and roughness, got {pipe.length!r}, {pipe.diameter!r}, "
                    f"{pipe.roughness!r}"
                )
            status = pipe.status.upper()
            if status not in ("OPEN", "CLOSED", "CV"):
                raise ValueError(
                    f"WaterNetwork: pipe {pipe.name!r} has status {pipe.status!r}; only "
                    f"OPEN, CLOSED and CV appear in the format"
                )
            if status != "OPEN":
                raise ValueError(
                    f"WaterNetwork: pipe {pipe.name!r} has status {status}; a closed pipe "
                    f"and a check valve are both status-switching devices and are out of "
                    f"scope (recorded follow-up)"
                )
        for pump in self.pumps:
            if pump.power is not None:
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} is a constant-POWER pump; the "
                    f"head-flow relation EPANET uses internally for one is not stated in "
                    f"the manual (research note section 4, UNVERIFIED), so only HEAD-curve "
                    f"pumps are modelled"
                )
            if pump.curve is None:
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} has neither a HEAD curve nor a "
                    f"POWER setting; EPANET requires one of the two"
                )
            if pump.curve not in self.curves:
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} names curve {pump.curve!r}, which "
                    f"has no [CURVES] entry"
                )
            points = self.curves[pump.curve]
            if len(points) not in (1, 3):
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} uses curve {pump.curve!r} with "
                    f"{len(points)} points; EPANET fits the power law h = h0 - r q^n to a "
                    f"single point or to three, and connects four or more with straight "
                    f"segments, which is out of scope (spec 13.5)"
                )
            if not pump.speed > 0.0:
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} has speed {pump.speed!r}; a "
                    f"relative speed setting must be positive"
                )
        for valve in self.valves:
            kind = valve.kind.upper()
            if kind in _VALVES_REFUSED:
                raise ValueError(
                    f"WaterNetwork: valve {valve.name!r} is a {kind}; PRV, PSV and PBV "
                    f"switch status with the if/else logic of EPANET 2.2 Manual p.113 and "
                    f"GPV needs a tabulated curve -- none is differentiable without a "
                    f"smoothed relaxation, and all four are out of scope (spec 13.5)"
                )
            if kind not in _VALVES_MODELLED:
                raise ValueError(
                    f"WaterNetwork: valve {valve.name!r} has type {valve.kind!r}; only "
                    f"{_VALVES_MODELLED} are modelled"
                )
            if kind == "TCV" and not valve.setting >= 0:
                raise ValueError(
                    f"WaterNetwork: TCV {valve.name!r} has a negative loss coefficient "
                    f"{valve.setting!r}"
                )
        if self.headloss not in ("H-W", "D-W"):
            raise ValueError(
                f"WaterNetwork: headloss {self.headloss!r}; only H-W and D-W are modelled "
                f"(Chezy-Manning's SI constant is UNVERIFIED, spec section 11)"
            )
        if self.options.demand_model not in ("DDA", "PDA"):
            raise ValueError(
                f"WaterNetwork: DEMAND MODEL {self.options.demand_model!r}; only DDA and "
                f"PDA are modelled"
            )
        tank_names = {t.name for t in self.tanks}
        pump_names = {p.name for p in self.pumps}
        for control in self.controls:
            if control.node not in tank_names:
                raise ValueError(
                    f"WaterNetwork: control on link {control.link!r} tests node "
                    f"{control.node!r}, which is not a tank"
                )
            if control.link not in pump_names:
                raise ValueError(
                    f"WaterNetwork: control names link {control.link!r}, which is not a "
                    f"pump; pipe controls are a recorded follow-up"
                )


def twoloop() -> WaterNetwork:
    """The committed hand-made SI fixture (`tests/data/water/twoloop_si.inp`).

    One reservoir `R1` at 50 m, six junctions at elevation 0 with constant demands 5, 8, 6,
    10, 7 and 9 L/s, and eight Hazen-Williams pipes (C = 130) forming exactly two
    independent loops. Measured heads at t = 0 (EPANET 2.2 through wntr 1.5.0): J1
    49.267731, J2 48.877316, J3 48.430882, J4 48.213867, J5 48.117119, J6 48.122936 m.
    """
    # `demand * 1e-3` and NOT `demand / 1000.0`: the reader multiplies by the LPS factor
    # 1e-3, and 9.0 / 1000.0 is 0.009 while 9.0 * 1e-3 is 0.009000000000000001. Matching
    # the reader's own arithmetic makes this fixture bit-identical to reading the file,
    # which is what `test_the_reader_reproduces_the_builders_fixture` asserts.
    junctions = tuple(
        Junction(name, 0.0, demand * 1e-3)
        for name, demand in (
            ("J1", 5.0), ("J2", 8.0), ("J3", 6.0),
            ("J4", 10.0), ("J5", 7.0), ("J6", 9.0),
        )
    )
    pipes = tuple(
        WaterPipe(name, u, v, length, diameter, 130.0)
        for name, u, v, length, diameter in (
            ("P1", "R1", "J1", 500.0, 0.300), ("P2", "J1", "J2", 400.0, 0.250),
            ("P3", "J2", "J3", 350.0, 0.200), ("P4", "J3", "J1", 450.0, 0.200),
            ("P5", "J3", "J4", 400.0, 0.250), ("P6", "J4", "J5", 350.0, 0.200),
            ("P7", "J5", "J6", 300.0, 0.150), ("P8", "J6", "J3", 500.0, 0.200),
        )
    )
    return WaterNetwork(
        junctions=junctions, reservoirs=(Reservoir("R1", 50.0),), pipes=pipes
    )


def build_model(
    net: WaterNetwork,
    *,
    headloss: str | None = None,
    pda: bool | None = None,
    p_min: float | None = None,
    p_req: float | None = None,
    exponent: float | None = None,
    quality: float | None = None,
    coupling: str = "pingpong",
    dt: float | None = None,
) -> tuple[Model, State, Drivers]:
    """Assemble the water model and return `(model, state, drivers)`.

    ONE `PotentialFlowLayer("water", ...)` on a network whose edge kinds partition by device
    type, plus the tank/controls closure when the network has tanks, plus an optional
    quality layer.

    `pda`, `p_min`, `p_req` and `exponent` each DEFAULT from `net.options` (the file's own
    `[OPTIONS] DEMAND MODEL`/`MINIMUM PRESSURE`/`REQUIRED PRESSURE`/`PRESSURE EXPONENT`,
    or `WaterOptions`'s EPANET defaults when the network was built by hand) when the caller
    passes nothing; passing any of them explicitly still overrides the file.

    `net.options.specific_gravity`/`viscosity` (relative to water at 20 C) reach the
    Darcy-Weisbach path, which is the only element here parameterised by fluid density and
    viscosity at all (`Duct`'s Reynolds-number friction factor; `scale`, below, uses the
    SAME density so a D-W network's head/pressure conversion stays self-consistent).
    Hazen-Williams has no such parameters -- EPANET's own H-W formula is calibrated for
    water and does not take them either -- so a NON-DEFAULT `SPECIFIC GRAVITY` or
    `VISCOSITY` on an H-W network is refused by name rather than silently ignored.

    `quality` is a bulk decay coefficient in 1/day (EPANET's own `[REACTIONS] Global Bulk`
    units, Manual Table 8.4 p.78), or `None`. TWO facts the plan writer MEASURED go with it
    (spec amendment A15). First, the junction DEMAND must enter the transport layer as a
    first-order removal rate `q_demand,j / V_j` (1/s): without it the advective generator
    has non-zero row sums -- a junction's edge inflow exceeds its edge outflow by exactly
    its demand -- and the steady system is SINGULAR (`torch.linalg.solve` raises on the
    committed two-loop fixture). Second, a junction's capacity on a LOOPED network is HALF
    the volume of every incident pipe; spec 3.4's "the volume of its single outgoing pipe"
    is a TREE property and does not carry over. Both are recorded in `model.notes`.

    The Darcy-Weisbach path reuses the existing `Duct`, which works in PRESSURE, so the
    whole layer does: heads are multiplied by `rho g` on the way in and `model.head_scale`
    carries the factor back out. Hazen-Williams (the default, and this milestone's parity
    formula) works directly in metres of head and `head_scale` is 1.
    """
    net.validate()
    headloss = headloss or net.headloss
    if headloss not in ("H-W", "D-W"):
        raise ValueError(
            f"build_model: headloss must be 'H-W' or 'D-W', got {headloss!r}"
        )
    options = net.options
    if pda is None:
        pda = options.demand_model == "PDA"
    if p_min is None:
        p_min = options.minimum_pressure
    if p_req is None:
        p_req = options.required_pressure
    if exponent is None:
        exponent = options.pressure_exponent
    if headloss == "H-W" and (options.specific_gravity != 1.0 or options.viscosity != 1.0):
        raise ValueError(
            f"build_model: [OPTIONS] SPECIFIC GRAVITY {options.specific_gravity!r} "
            f"/ VISCOSITY {options.viscosity!r} are not the defaults, but Hazen-Williams "
            f"does not take fluid density or viscosity (EPANET's own H-W formula is "
            f"calibrated for water); use headloss='D-W' or leave both at 1.0"
        )
    rho = RHO_W * options.specific_gravity
    mu = MU_W * options.viscosity
    graph = Network(dtype=F64)
    node_names = net.nodes()
    for junction in net.junctions:
        graph.add_node(junction.name, elevation=junction.elevation)
    for reservoir in net.reservoirs:
        graph.add_node(reservoir.name, elevation=reservoir.head)
    for tank in net.tanks:
        graph.add_node(tank.name, elevation=tank.elevation)
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)
    for pump in net.pumps:
        graph.add_edge(pump.u, pump.v, kind="pump", name=pump.name)
    tcv = [v for v in net.valves if v.kind.upper() == "TCV"]
    fcv = [v for v in net.valves if v.kind.upper() == "FCV"]
    for valve in tcv:
        graph.add_edge(valve.u, valve.v, kind="valve", name=valve.name)
    for valve in fcv:
        graph.add_edge(valve.u, valve.v, kind="fcv", name=valve.name)

    notes: dict[str, str] = dict(net.notes)
    scale = rho * G if headloss == "D-W" else 1.0
    elements: list = []
    if net.pipes:
        if headloss == "H-W":
            elements.append(
                HazenWilliams(
                    torch.tensor([p.length for p in net.pipes], dtype=F64),
                    torch.tensor([p.diameter for p in net.pipes], dtype=F64),
                    torch.tensor([p.roughness for p in net.pipes], dtype=F64),
                    minor_loss=torch.tensor([p.minor_loss for p in net.pipes], dtype=F64),
                    kind="pipe",
                )
            )
        else:
            notes["headloss"] = (
                f"Darcy-Weisbach uses the existing Duct element (Colebrook, unrolled "
                f"fixed point) at rho = {rho} kg/m3 and mu = {mu} Pa s (998.2 / 1.002e-3 "
                f"scaled by [OPTIONS] SPECIFIC GRAVITY / VISCOSITY); EPANET uses "
                f"Swamee-Jain above Re = 4000, Hagen-Poiseuille below 2000 and Dunlop's "
                f"cubic between, and spec row D4 RECORDS the resulting residual"
            )
            elements.append(
                Duct(
                    torch.tensor([p.length for p in net.pipes], dtype=F64),
                    torch.tensor([p.diameter for p in net.pipes], dtype=F64),
                    torch.tensor([p.roughness for p in net.pipes], dtype=F64),
                    rho=rho, mu=mu, n_iter=12, kind="pipe",
                )
            )
    if net.pumps:
        h0_list, r_list = [], []
        for pump in net.pumps:
            points = net.curves[pump.curve]
            if len(points) == 1:
                q_design, h_design = points[0]
                h0, r, _ = three_point_curve(
                    torch.tensor([q_design], dtype=F64),
                    torch.tensor([h_design], dtype=F64),
                )
            else:
                h0, r = _fit_three_points(points, pump.name)
            h0_list.append(float(h0) * scale)
            r_list.append(float(r) * scale)
        elements.append(
            PumpCurve(
                torch.tensor(h0_list, dtype=F64),
                torch.tensor(r_list, dtype=F64),
                2.0,
                speed=torch.tensor([p.speed for p in net.pumps], dtype=F64),
                status_key="water.status",
                kind="pump",
            )
        )
    if tcv:
        elements.append(
            MinorLoss(
                torch.tensor([v.setting * scale for v in tcv], dtype=F64),
                torch.tensor([v.diameter for v in tcv], dtype=F64),
                kind="valve",
            )
        )
    if fcv:
        elements.append(
            FixedFlow(torch.tensor([v.setting for v in fcv], dtype=F64), kind="fcv")
        )

    boundary = [r.name for r in net.reservoirs] + [t.name for t in net.tanks]
    node_sources: list = []
    if pda:
        if not net.junctions:
            raise ValueError("build_model: pda=True on a network with no junctions")
        node_sources.append(
            PressureDrivenDemand(
                torch.tensor(
                    [node_names.index(j.name) for j in net.junctions], dtype=torch.long
                ),
                torch.tensor([j.demand for j in net.junctions], dtype=F64),
                torch.tensor([j.elevation * scale for j in net.junctions], dtype=F64),
                p_min=p_min * scale, p_req=p_req * scale, exponent=exponent,
            )
        )
        notes["demand_model"] = (
            f"pressure-driven demand (EPANET's PDA) with MINIMUM PRESSURE {p_min} m, "
            f"REQUIRED PRESSURE {p_req} m and PRESSURE EXPONENT {exponent}"
        )
    layer = PotentialFlowLayer(
        graph, "water", elements, boundary=boundary, node_sources=node_sources,
        quantity="head" if headloss == "H-W" else "pressure",
        unit="m" if headloss == "H-W" else "Pa",
    )

    layers: dict = {"water": layer}
    closures: list = []
    drivers: Drivers = {}
    state: State = {
        "water.phi": torch.zeros(graph.n, dtype=F64),
        "water.q": torch.zeros(len(layer.cols), dtype=F64),
    }

    reservoir_heads = torch.tensor([r.head * scale for r in net.reservoirs], dtype=F64)
    tank_closure = None
    if net.tanks:
        tank_closure = TankLevels(
            graph, list(net.tanks), list(net.controls), [p.name for p in net.pumps],
            dt=net.hydraulic_timestep if dt is None else dt,
            reservoir_heads=reservoir_heads,
            tank_first_index=len(net.reservoirs),
            head_scale=scale,
        )
        closures.append(tank_closure)
        state["water.tank_level"] = torch.tensor(
            [t.init_level for t in net.tanks], dtype=F64
        )
        state["water.link_status"] = torch.ones(len(net.pumps), dtype=F64)
    else:
        drivers["water.phi_boundary"] = reservoir_heads
        if net.pumps:
            drivers["water.status"] = torch.ones(len(net.pumps), dtype=F64)

    sources = torch.zeros(graph.n, dtype=F64)
    if not pda:
        for junction in net.junctions:
            sources[node_names.index(junction.name)] = -junction.demand
    drivers["water.sources"] = sources

    if quality is not None:
        if not net.pipes:
            raise ValueError("build_model: quality on a network with no pipes")
        volume = torch.tensor(
            [torch.pi * p.diameter**2 / 4.0 * p.length for p in net.pipes], dtype=F64
        )
        capacity = torch.zeros(graph.n, dtype=F64)
        for i, pipe in enumerate(net.pipes):
            capacity[node_names.index(pipe.u)] += 0.5 * float(volume[i])
            capacity[node_names.index(pipe.v)] += 0.5 * float(volume[i])
        interior = [node_names.index(j.name) for j in net.junctions]
        cap_i = capacity[interior]
        if bool((cap_i <= 0).any()):
            bad = [
                net.junctions[i].name
                for i, value in enumerate(cap_i.tolist())
                if value <= 0
            ]
            raise ValueError(
                f"build_model: junction(s) {bad} touch no pipe, so their quality "
                f"capacity is zero; a node with no volume cannot hold a concentration"
            )
        demand = torch.tensor([j.demand for j in net.junctions], dtype=F64)
        removal = (demand / cap_i + quality / 86400.0).unsqueeze(-1)
        notes["quality_capacity"] = (
            "on a looped network a junction's capacity is HALF the volume of every "
            "incident pipe; spec 3.4's single outgoing pipe is a tree property"
        )
        notes["quality_removal"] = (
            "the nodal demand enters as a first-order removal rate q_demand / V; without "
            "it the advective generator has non-zero row sums and the steady system is "
            "singular (measured on twoloop_si.inp)"
        )
        layers["quality"] = TransportLayer(
            graph, "quality", capacity=cap_i, flow_kind="pipe",
            boundary=boundary, n_species=1, removal=removal, scheme="implicit",
            quantity="concentration", unit="kg/m3",
        )
        drivers["quality.x_boundary"] = torch.zeros(len(boundary), dtype=F64)
        state["quality.x"] = torch.zeros(len(interior), dtype=F64)

    model = Model(graph, layers, closures=closures, coupling=coupling)
    model.notes = notes
    model.node_names = node_names
    model.link_names = [link.name for link in net.links()]
    model.tank_closure = tank_closure
    model.head_scale = scale
    model.water_network = net
    return model, state, drivers


def _fit_three_points(points, name: str) -> tuple[Tensor, Tensor]:
    """`(h0, r)` for `h = h0 - r q^2` through EPANET's three-point form.

    EPANET fits `h_G = A - B q^C` through the three given points (Manual p.19). This
    application pins `C = 2`, which is what the SINGLE-point construction forces exactly
    (spec amendment A10); for a genuine three-point curve the fit is refused unless the
    third point is consistent with that exponent, rather than silently fitted to another.
    """
    (q1, h1), (q2, h2), (q3, h3) = sorted(points)
    if q1 != 0.0:
        raise ValueError(
            f"build_model: pump {name!r} has a three-point curve whose first point "
            f"is at flow {q1}, not 0; EPANET's fit takes the shut-off head there"
        )
    h0 = torch.tensor(h1, dtype=F64)
    r = torch.tensor((h1 - h2) / q2**2, dtype=F64)
    predicted = h1 - float(r) * q3**2
    if abs(predicted - h3) > 1e-6 * max(abs(h3), 1.0):
        raise ValueError(
            f"build_model: pump {name!r}'s three points are not consistent with the "
            f"exponent 2 this application fits (the third point implies {predicted}, the "
            f"curve gives {h3}); a general exponent is a recorded follow-up"
        )
    return h0, r


def initial_state(model: Model) -> State:
    """All-zero state, dispatching on each layer's `quantity` (the app convention)."""
    state: State = {}
    for name, layer in model.potential.items():
        if layer.quantity not in ("head", "pressure"):
            raise ValueError(
                f"apps.water.initial_state: potential layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds head or pressure layers only"
            )
        state[f"{name}.phi"] = torch.zeros(model.net.n, dtype=F64)
        state[f"{name}.q"] = torch.zeros(len(layer.cols), dtype=F64)
    for name, layer in model.transport.items():
        if layer.quantity != "concentration":
            raise ValueError(
                f"apps.water.initial_state: transport layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds concentration layers only"
            )
        state[f"{name}.x"] = torch.zeros(layer.n_i, dtype=F64)
    if model.tank_closure is not None:
        state["water.tank_level"] = torch.tensor(
            [t.init_level for t in model.tank_closure.tanks], dtype=F64
        )
        state["water.link_status"] = torch.ones(
            len(model.tank_closure.pump_names), dtype=F64
        )
    return state


def water_steady(model: Model, state: State, drivers: Drivers, **solve_kwargs) -> State:
    """The quasi-steady state: one `Model.steady` with tight Newton tolerances.

    `atol`/`rtol` default to 1e-11. MEASURED, sweeping Net1's whole 24 h duty cycle over
    the eight demand multipliers 0.7 to 1.4: 1e-14 does not converge at all (the residual
    floors at 1.3e-14 against a flow scale of 0.1 m3/s), 1e-12 fails at multiplier 1.2
    (floor 5.4e-12), and 5e-12 is the tightest value at which all eight converge. 1e-11 is
    that with a factor of two in hand, and it is still four orders below the parity rows'
    own float32-limited residuals, so it does not move D1, D2, D3 or D5.

    `max_iter` defaults to 200, NOT Newton's own 50. When a `[CONTROLS]` line CLOSES a
    pump, the pipe immediately downstream of it is left carrying exactly zero flow, and
    Hazen-Williams' laminar blend gives that one edge a tangent slope of order
    `(dp_transition / K)^(1/1.852) / dp_transition` -- about 630 on Net1's pipe 10 against
    ~1e-2 elsewhere, a jump of four orders of magnitude in the Jacobian's conditioning at
    the switch. MEASURED on Net1's 24 h duty cycle: the switch step needs 136 Newton
    iterations and every other step needs a handful, so 50 fails outright (residual
    9.6e-4) and 200 clears it with margin. Raising `dp_transition` would reduce the count
    (measured: 50 at 1e-3, 93 at 1e-6, 136 at 1e-9, 179 at 1e-12) but 1e-9 is where the D3
    trajectory has CONVERGED with respect to it, so the iterations are spent rather than
    the fidelity.

    After the solve, every `pump` edge's CONVERGED flow is checked against its own
    `q_max` (ruling M4-R13, spec 13.1: "a demand for more than `q_max` is refused by
    name"). `PumpCurve.flow`/`dflow` extrapolate the fitted curve indefinitely so that
    Newton's intermediate iterates are never aborted mid-solve; this is where that
    extrapolation is finally held to account, once there is a converged answer to check.
    No clamp, no warning: an exceedance raises `RuntimeError` naming the pump, the batch
    instance and the flow against `q_max`.
    """
    solve_kwargs.setdefault("atol", 1e-11)
    solve_kwargs.setdefault("rtol", 1e-11)
    solve_kwargs.setdefault("max_iter", 200)
    result = model.steady(state, drivers, **solve_kwargs)
    _refuse_pump_overflow(model, result)
    return result


def _refuse_pump_overflow(model: Model, state: State) -> None:
    """Ruling M4-R13: refuse, by name, a converged pump flow beyond its own `q_max`.

    FR-13: reads the layer through its public `element_for` (kind -> `(element, slice)`)
    rather than the private `_elements`/`_kind_slices` this used to reach into.
    """
    layer = model.potential.get("water")
    if layer is None or "pump" not in layer.kinds:
        return
    pump, sl = layer.element_for("pump")
    lo, hi = sl.start, sl.stop
    q = state["water.q"][..., lo:hi].detach()
    q_max = pump.q_max().detach()
    over = q.abs() > q_max
    if not bool(over.any()):
        return
    names = model.link_names[lo:hi]
    flat_over = over.reshape(-1, over.shape[-1])
    flat_q = q.abs().reshape(-1, q.shape[-1])
    bad = [
        f"pump {names[j]!r} (instance {b}): flow {float(flat_q[b, j])} exceeds "
        f"q_max {float(q_max[j])}"
        for b in range(flat_over.shape[0])
        for j in range(flat_over.shape[1])
        if flat_over[b, j]
    ]
    raise RuntimeError("water_steady: " + "; ".join(bad))


def tank_inflow(model: Model, state: State) -> Tensor:
    """Net inflow (m3/s) at every tank node, from the layer's solved branch flows.

    `PotentialFlowLayer._accumulate(q)` is `A q`, the net OUTflow at every node, so the
    inflow is its negation restricted to the tank rows -- the same identity
    `Model.ports` uses for `boundary_flows`.
    """
    layer = model.potential["water"]
    first = model.tank_closure.tank_first_index
    return -layer._accumulate(state["water.q"])[..., layer.bound[first:]]
