"""`WaterNetwork`, its validation, and the (Model, State, Drivers) builder.

A pressurised distribution network is exactly the framework's own case: every pipe is full,
the head loss is a monotone function of the head difference, and the whole thing is ONE
`KirchhoffLayer` whose potential is hydraulic HEAD in metres. That is the difference
from the free-surface sewer of `apps.sewer`, whose normal flow is set by slope and upstream
inflow rather than by head difference.

Node order: junctions in file order, then reservoirs, then tanks. Boundary = every reservoir
then every tank, in that order. `"water.phi_boundary"` gives the reservoirs' heads, plus the
tanks' only where they are not held at boundary nodes (`tank_scheme="forward_euler"` holds
them at their heads in the step-start `"water.phi"`), in that order.
Edge kinds partition by device type: `pipe`, `pump`, `valve` (a `ThrottleValve`, TCV),
`fcv` (a `FlowControlValve`), `pbv` (a `BreakerValve`) and `gpv` (a `GeneralPurposeValve`).
Pumps and valves are implicit laws (their flow is an unknown of the solve), and so are pipes in
their resistance form (`pipe_form="flow"`, the default).

Every link's status, setting and allowed flow direction are drivers, one value per link in
`model.link_names` order: `"water.status"`, `"water.setting"` and `"water.one_way"`
(`noodl.apps.water.elements.STATUS_KEY` and its neighbours). `build_model` fills them from
the network: a CLOSED pipe or `[STATUS]` line sets the status, a CV pipe a forward-only
direction, a valve's setting (or a numeric `[STATUS]`) its setting, and every valve starts
ACTIVE.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from noodl._unionfind import UnionFind
from noodl.apps.water.controls import ACTIVE, CLOSED, HTOL, OPEN, Control, Rule, status_code
from noodl.apps.water.demand import EmitterOutflow, PressureDrivenDemand
from noodl.apps.water.elements import (
    CBIG_FT_PER_CFS,
    DEAD_END_KEY,
    DEMAND_FACTOR_KEY,
    EPANET_QCF,
    KW_PER_HP,
    M_PER_FT,
    NU_WATER_REF,
    ONE_WAY_KEY,
    PIN_KEY,
    RESERVOIR_HEAD_KEY,
    RQTOL_FT_PER_CFS,
    SETTING_KEY,
    STATUS_KEY,
    TANK_INFLOW_KEY,
    TIME_KEY,
    BreakerValve,
    ChezyManning,
    CompositeDarcyWeisbach,
    ConstantPowerPump,
    FlowControlledPipe,
    FlowControlledPump,
    FlowControlValve,
    GeneralPurposeValve,
    HazenWilliams,
    PiecewisePumpCurve,
    PressureReducingValve,
    PressureSustainingValve,
    PumpCurve,
    ShutLinks,
    ThrottleValve,
    _joined_groups,
    epanet_minor_coefficient,
    three_point_curve,
)
from noodl.apps.water.rules import evaluate_rules, take_actions_per_link
from noodl.apps.water.tanks import TankLevels, TankStorageControls, TankVolumeCurve
from noodl.elements.base import LawContext
from noodl.elements.duct import Duct
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer, UndefinedSteadyState
from noodl.model import Model
from noodl.refs import drivers_from, state_from
from noodl.refs import initial_drivers as refs_initial_drivers
from noodl.storage import LinearStorage
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: Water at 20 C (standard reference values): density (kg/m3) and dynamic
#: viscosity (Pa s). Used only by the Darcy-Weisbach path, which works in PRESSURE.
RHO_W = 998.2
MU_W = 1.002e-3
G = 9.80665

#: Valve types this application models, in the order of their columns among the links.
_VALVES_MODELLED = ("TCV", "FCV", "PBV", "GPV", "PRV", "PSV")

#: The layer's edge kind of each valve type.
_VALVE_KINDS = {"TCV": "valve", "FCV": "fcv", "PBV": "pbv", "GPV": "gpv", "PRV": "prv",
                "PSV": "psv"}


def _valve_clash(a, b) -> bool:
    """Whether valves `a` and `b` hold one node twice (EPANET 2.2's error 220, `valvecheck`
    in `project.c`): two PRVs sharing their end node or in series, two PSVs sharing their
    start node or in series, a PSV starting at a PRV's or an FCV's end, a PRV ending at an
    FCV's start. Symmetric."""
    for x, y in ((a, b), (b, a)):
        kx, ky = x.kind.upper(), y.kind.upper()
        if kx == ky == "PRV" and (x.v == y.v or x.v == y.u):
            return True
        if kx == ky == "PSV" and (x.u == y.u or x.u == y.v):
            return True
        if kx in ("PRV", "FCV") and ky == "PSV" and x.v == y.u:
            return True
        if kx == "PRV" and ky == "FCV" and x.v == y.u:
            return True
    return False


@dataclass(frozen=True)
class WaterOptions:
    """`[OPTIONS]` keys that change the physics, with EPANET 2.2's own defaults.

    `demand_model` is `"DDA"` or `"PDA"`; `minimum_pressure`/`required_pressure` are in
    METRES of head (this reader converts a US file's native psi at the same boundary as
    every other length, with `PSI_TO_M`); `pressure_exponent` is
    dimensionless; `specific_gravity`/`viscosity` are relative to water at 20 C. The
    `required_pressure` default 0.1 is EPANET's own lower limit on the value ("0.1 in psi
    or m", `wntr`'s inp writer), not the manual OPTIONS table's literal 0.0, which only
    describes the no-op DDA case; `wntr`'s `HydraulicOptions.required_pressure = 0.07` is
    that SAME 0.1 converted from psi assuming US units, which does not apply here since
    this reader is always SI.

    `flow_units` is the `.inp` file's `[OPTIONS] UNITS` (`None` for a network built by
    hand). It changes nothing physical: the EPANET-exact Darcy-Weisbach law uses it only to
    reproduce EPANET's own ROUNDED conversion of that unit to cfs (`EPANET_QCF`).

    `emitter_exponent` is the exponent `n` of every emitter's `q = C p^n` (`[OPTIONS]
    EMITTER EXPONENT`, EPANET's default 0.5).
    """

    demand_model: str = "DDA"
    minimum_pressure: float = 0.0
    required_pressure: float = 0.1
    pressure_exponent: float = 0.5
    specific_gravity: float = 1.0
    viscosity: float = 1.0
    flow_units: str | None = None
    emitter_exponent: float = 0.5


#: The node attributes of a rule premise that this application reads (`noodl.apps.water.rules`).
_NODE_PREMISES = ("DEMAND", "HEAD", "LEVEL", "PRESSURE")


@dataclass(frozen=True)
class Junction:
    """A junction: `elevation` (m) and base `demand` (m3/s, withdrawn), optionally scaled
    by a time `pattern`."""

    name: str
    elevation: float
    demand: float = 0.0
    pattern: str | None = None


@dataclass(frozen=True)
class Reservoir:
    """A fixed-head node: total `head` (m), optionally varied by a time `pattern`."""

    name: str
    head: float
    pattern: str | None = None


@dataclass(frozen=True)
class Tank:
    """A tank: bottom `elevation`, initial, minimum and maximum water levels and
    `diameter` (all m); its level is its node's head less its bottom, stepped by the layer's
    storage (`build_model(tank_scheme=...)`).

    A tank is a cylinder of `diameter`, or, when `volume_curve` names a `curves` entry
    (level m, volume m3), a tank whose volume is that curve of its level (`TankVolumeCurve`;
    the diameter is then unused). `overflow` says the tank may spill when full (`[TANKS]
    Overflow YES`): it is then never closed against filling, its level is held at the
    maximum, and what flows in beyond it is the overflow (`WaterResults.tank_overflow`).
    `min_volume`
    (m3) is the file's MinVol, which moves no level (EPANET's volumes start from it)."""

    name: str
    elevation: float
    init_level: float
    min_level: float
    max_level: float
    diameter: float
    volume_curve: str | None = None
    overflow: bool = False
    min_volume: float = 0.0


@dataclass(frozen=True)
class WaterPipe:
    """A pipe from `u` to `v`: `length` and `diameter` (m), `roughness` (in the
    network's head-loss formula's units), a minor-loss coefficient and its `status`."""

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
    """A pump from `u` to `v` on a head-flow `curve` (by name) or at constant `power` (kW),
    at relative `speed`, optionally varied by `speed_pattern`.

    `efficiency_curve` (a `curves` entry, flow m3/s against percent), `price` ($/kWh) and
    `price_pattern` come from `[ENERGY]` and are `None` when the file gives none (the
    network's `energy` then applies); they do not enter the hydraulics."""

    name: str
    u: str
    v: str
    curve: str | None
    power: float | None = None
    speed: float = 1.0
    speed_pattern: str | None = None
    efficiency_curve: str | None = None
    price: float | None = None
    price_pattern: str | None = None


@dataclass(frozen=True)
class Valve:
    """A valve from `u` to `v`: `diameter` (m), EPANET `kind` (for example `TCV`) and
    its `setting` (TCV: a loss coefficient; FCV: a flow, m3/s; PBV: a head drop, m), with a
    minor-loss coefficient. A GPV's setting is its head-loss `curve` (a `curves` key)."""

    name: str
    u: str
    v: str
    diameter: float
    kind: str
    setting: float
    minor_loss: float = 0.0
    curve: str | None = None


@dataclass(frozen=True)
class Emitter:
    """An emitter at junction `node`: the outflow is `coefficient * p^n` with `p` the
    pressure head (m) and `n` `WaterOptions.emitter_exponent`; `coefficient` is in m3/s per
    m^n."""

    node: str
    coefficient: float


@dataclass(frozen=True)
class EnergyOptions:
    """`[ENERGY]` defaults: `global_efficiency` (percent), `global_price` ($/kWh) with an
    optional `global_pattern` and the monthly `demand_charge` ($/kW). Reporting only: none
    of them changes a head or a flow."""

    global_efficiency: float = 75.0
    global_price: float = 0.0
    global_pattern: str | None = None
    demand_charge: float = 0.0


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
    rules: tuple[Rule, ...] = ()
    emitters: tuple[Emitter, ...] = ()
    initial_status: dict[str, str | float] = field(default_factory=dict)
    energy: EnergyOptions = field(default_factory=EnergyOptions)
    demand_pattern: str | None = None
    pattern_timestep: float = 3600.0
    hydraulic_timestep: float = 3600.0
    report_timestep: float = 3600.0
    duration: float = 0.0
    explicit_rule_timestep: float | None = None
    pattern_start: float = 0.0
    report_start: float = 0.0
    start_clocktime: float = 0.0
    headloss: str = "H-W"
    options: WaterOptions = field(default_factory=WaterOptions)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def rule_timestep(self) -> float:
        """The time step of rule evaluation (s): `[TIMES] RULE TIMESTEP` (`explicit_rule_timestep`)
        capped at the hydraulic step, else a tenth of the hydraulic step, as EPANET."""
        if self.explicit_rule_timestep is None:
            return self.hydraulic_timestep / 10.0
        return min(self.explicit_rule_timestep, self.hydraulic_timestep)

    def nodes(self) -> list[str]:
        """Node names in the builder's own order: junctions, reservoirs, tanks."""
        return (
            [j.name for j in self.junctions]
            + [r.name for r in self.reservoirs]
            + [t.name for t in self.tanks]
        )

    def fixed_head_nodes(self) -> tuple:
        """The nodes that hold a head for the cut-off analysis: the reservoirs, the tanks and
        the junctions with an emitter. An emitter ``q = C p^n`` ties its junction's head to
        the junction's elevation as a reservoir there would (EPANET models it as a
        fictitious pipe to a reservoir at that elevation, `hydcoeffs.c`, `emittercoeffs`), and
        at negative pressure it can supply a demand: a part of the network that holds one is
        neither cut off nor without a head of its own, and the Newton solve finds its heads.
        """
        with_emitter = {e.node for e in self.emitters}
        return (*self.reservoirs, *self.tanks,
                *(j for j in self.junctions if j.name in with_emitter))

    def links(self) -> list:
        """Every link, in the kind order the layer's law list uses."""
        return [*self.pipes, *self.pumps,
                *(v for kind in _VALVES_MODELLED for v in self.valves_of(kind))]

    def valves_of(self, kind: str) -> list:
        """The valves of EPANET type `kind`, in file order."""
        return [v for v in self.valves if v.kind.upper() == kind]

    def _all_links(self) -> list:
        """Every link by name, including valves of a type this application refuses."""
        modelled = {id(v) for v in self.links()}
        return [*self.links(), *(v for v in self.valves if id(v) not in modelled)]

    def check(self) -> None:
        """Refuse a network that is malformed, whatever it models: a duplicate name, no
        fixed-head node, a link, control, status, emitter that names something that
        is not there, a pipe without a positive size. The reader calls this; `validate`
        calls it first and then refuses, BY NAME, each feature not yet modelled."""
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
        for link in self._all_links():
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
            if pipe.status.upper() not in ("OPEN", "CLOSED", "CV"):
                raise ValueError(
                    f"WaterNetwork: pipe {pipe.name!r} has status {pipe.status!r}; only "
                    f"OPEN, CLOSED and CV appear in the format"
                )
        for name in self.initial_status:
            if name not in link_names:
                raise ValueError(f"WaterNetwork: [STATUS] names {name!r}, which is not a link")
        junction_names = {j.name for j in self.junctions}
        for emitter in self.emitters:
            if emitter.node not in junction_names:
                raise ValueError(
                    f"WaterNetwork: emitter at {emitter.node!r}, which is not a junction"
                )
        valve_names = {v.name for v in self.valves}
        pump_names = {p.name for p in self.pumps}
        for control in self.controls:
            if control.link not in link_names:
                raise ValueError(
                    f"WaterNetwork: control names link {control.link!r}, which is not a link"
                )
            if control.status == ACTIVE and control.link not in valve_names:
                raise ValueError(
                    f"WaterNetwork: control on link {control.link!r} sets the status ACTIVE, "
                    f"which only a valve can take, not a pipe or a pump"
                )
            gpvs = {v.name for v in self.valves if v.kind.upper() == "GPV"}
            if control.setting is not None and (
                    control.link not in valve_names | pump_names or control.link in gpvs):
                raise ValueError(
                    f"WaterNetwork: control on link {control.link!r} sets the numeric "
                    f"setting {control.setting!r}, which only a pump (its speed) or a valve "
                    f"other than a GPV takes, not a pipe or a GPV"
                )
            if control.trigger == "node" and control.node in {r.name for r in self.reservoirs}:
                raise ValueError(
                    f"WaterNetwork: control on link {control.link!r} tests reservoir "
                    f"{control.node!r}; EPANET 2.2 tests a reservoir's volume, which has "
                    f"area 0, so the control always holds, and no sensible network means "
                    f"that. Use a tank or a junction"
                )
            if control.trigger == "node" and control.node not in seen:
                raise ValueError(
                    f"WaterNetwork: control on link {control.link!r} tests node "
                    f"{control.node!r}, which is not a node"
                )

        for pump in self.pumps:
            for what, name in (("speed pattern", pump.speed_pattern),
                               ("energy price pattern", pump.price_pattern)):
                if name is not None and name not in self.patterns:
                    raise ValueError(
                        f"WaterNetwork: pump {pump.name!r} names the {what} {name!r}, which "
                        f"has no [PATTERNS] entry"
                    )
        if self.energy.global_pattern is not None and (
            self.energy.global_pattern not in self.patterns
        ):
            raise ValueError(
                f"WaterNetwork: [ENERGY] GLOBAL PATTERN {self.energy.global_pattern!r} has "
                f"no [PATTERNS] entry"
            )
        for rule in self.rules:
            for premise in rule.premises:
                if premise.object == "NODE" and premise.id not in seen:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} names node {premise.id!r}, which "
                        f"is not a node"
                    )
                if premise.object == "LINK" and premise.id not in link_names:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} names link {premise.id!r}, which "
                        f"is not a link"
                    )
                if premise.attribute in ("DAY", "POWER") or (
                        premise.object == "NODE" and premise.attribute not in _NODE_PREMISES):
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} tests {premise.attribute}, which "
                        f"EPANET 2.2 reads in no rule premise on a {premise.object.lower()}"
                        f" (DEMAND, HEAD, LEVEL and PRESSURE of a node; FLOW, STATUS and "
                        f"SETTING of a link; SYSTEM TIME, CLOCKTIME and DEMAND)")
                if (premise.object == "LINK" and premise.attribute == "SETTING"
                        and premise.id not in {v.name for v in self.valves
                                               if v.kind.upper() != "GPV"}
                        | {p.name for p in self.pumps}):
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} tests the SETTING of link "
                        f"{premise.id!r}, which is a pipe or a GPV; only a pump's "
                        f"or another valve's setting is one")
                if (premise.object == "NODE" and premise.attribute == "DEMAND"
                        and premise.id in {r.name for r in self.reservoirs}):
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} tests the DEMAND of reservoir "
                        f"{premise.id!r}; only a junction's and a tank's demand is read "
                        f"here (a tank's is its inflow)")
            gpvs = {v.name for v in self.valves if v.kind.upper() == "GPV"}
            checked = {p.name for p in self.pipes if p.status.upper() == "CV"}
            for action in (*rule.then_actions, *rule.else_actions):
                if action.link not in link_names:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} acts on link {action.link!r}, "
                        f"which is not a link"
                    )
                if action.link in checked:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} acts on pipe {action.link!r}, a "
                        f"check valve, which EPANET 2.2 cannot control (error 207)")
                if (action.attribute == "STATUS" and action.value == ACTIVE
                        and action.link not in {v.name for v in self.valves}):
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} sets the status of link "
                        f"{action.link!r} to ACTIVE, which only a valve can take, not a pipe "
                        f"or a pump (on a valve EPANET's `takeactions` takes no action on "
                        f"it, and neither does this)")
                if action.attribute == "SETTING" and action.value < 0.0:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} sets link {action.link!r} to the "
                        f"negative setting {action.value!r} (EPANET error 202)")
                if action.attribute == "SETTING" and action.link in gpvs:
                    raise ValueError(
                        f"WaterNetwork: rule {rule.name!r} sets a numeric setting on the "
                        f"GPV {action.link!r}, whose head loss is its curve (EPANET error "
                        f"202)")

    def validate(self) -> None:
        """Refuse, BY NAME, everything this application does not model."""
        self.check()
        valve_names = {v.name for v in self.valves}
        pump_names = {p.name for p in self.pumps}
        for name, value in self.initial_status.items():
            if isinstance(value, str):
                if status_code(value) is None:
                    raise ValueError(
                        f"WaterNetwork: [STATUS] gives link {name!r} the status {value!r}; "
                        f"only OPEN, CLOSED, ACTIVE or a numeric setting appear in the format"
                    )
                if status_code(value) == ACTIVE and name not in valve_names:
                    raise ValueError(
                        f"WaterNetwork: [STATUS] sets link {name!r} ACTIVE, which only a "
                        f"valve can take, not a pipe or a pump"
                    )
            elif name in pump_names:
                raise ValueError(
                    f"WaterNetwork: [STATUS] sets pump {name!r} to the speed {value!r}; "
                    f"pump speed settings are out of scope"
                )
            elif name not in valve_names:
                raise ValueError(
                    f"WaterNetwork: [STATUS] gives link {name!r} the setting {value!r}; "
                    f"only a valve or a pump takes a numeric setting"
                )
        for tank in self.tanks:
            # EPANET 2.2 (`input1.c`, `validatetanks`, error 225): the initial level within
            # [min, max], and a volume curve covering both limits
            if not tank.min_level <= tank.init_level <= tank.max_level:
                raise ValueError(
                    f"WaterNetwork: tank {tank.name!r} has the initial level "
                    f"{tank.init_level} m outside its levels [{tank.min_level}, "
                    f"{tank.max_level}] m")
            if tank.volume_curve is None:
                continue
            points = self.curves.get(tank.volume_curve)
            if not points:
                raise ValueError(
                    f"WaterNetwork: tank {tank.name!r} names the volume curve "
                    f"{tank.volume_curve!r}, which is not among the curves")
            if len(points) < 2 or any(b[0] <= a[0] or b[1] < a[1]
                                      for a, b in zip(points, points[1:], strict=False)):
                raise ValueError(
                    f"WaterNetwork: tank {tank.name!r} has the volume curve "
                    f"{tank.volume_curve!r} with {len(points)} point(s); a volume curve needs "
                    f"at least two, with levels strictly increasing and volumes that do not "
                    f"decrease (EPANET accepts a decreasing volume, which is not physical)")
            if tank.min_level < points[0][0] or tank.max_level > points[-1][0]:
                raise ValueError(
                    f"WaterNetwork: tank {tank.name!r} has levels "
                    f"[{round(tank.min_level, 6)}, {round(tank.max_level, 6)}] m outside its "
                    f"volume curve {tank.volume_curve!r} (levels {points[0][0]} to "
                    f"{points[-1][0]} m)")
        for pump in self.pumps:
            if not pump.speed > 0.0:
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r} has speed {pump.speed!r}; a "
                    f"relative speed setting must be positive"
                )
            if pump.power is not None:
                if not pump.power > 0.0:
                    raise ValueError(
                        f"WaterNetwork: pump {pump.name!r} has the power {pump.power!r} kW; "
                        f"a constant-power pump needs a positive power (EPANET error 202)")
                continue
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
            if _pump_kind(self, pump) == "curve":
                try:
                    power_curve(points)
                except ValueError as exc:
                    raise ValueError(
                        f"WaterNetwork: pump {pump.name!r}'s curve {pump.curve!r} has no "
                        f"power-function fit h = h0 - r q^n: {exc}") from None
            elif any(b[0] <= a[0] or b[1] >= a[1]
                     for a, b in zip(points, points[1:], strict=False)):
                raise ValueError(
                    f"WaterNetwork: pump {pump.name!r}'s curve {pump.curve!r} must have its "
                    f"flows rising and its heads falling from point to point (EPANET "
                    f"errors 227, 230)")
        junction_names = {j.name for j in self.junctions}
        for valve in self.valves:
            kind = valve.kind.upper()
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
            if kind == "PBV" and not (valve.setting >= 0 and valve.minor_loss >= 0):
                raise ValueError(
                    f"WaterNetwork: PBV {valve.name!r} needs a head drop and a minor loss "
                    f">= 0, got {valve.setting!r} and {valve.minor_loss!r}"
                )
            if kind in ("PRV", "PSV"):
                if not (valve.setting >= 0 and valve.minor_loss >= 0):
                    raise ValueError(
                        f"WaterNetwork: {kind} {valve.name!r} needs a pressure setting and a "
                        f"minor loss >= 0, got {valve.setting!r} and {valve.minor_loss!r}"
                    )
                held = valve.v if kind == "PRV" else valve.u
                if held not in junction_names:
                    raise ValueError(
                        f"WaterNetwork: {kind} {valve.name!r} controls the head of {held!r}, "
                        f"a reservoir or tank, whose head is fixed already (EPANET refuses "
                        f"it too, error 219)"
                    )
            if kind == "GPV":
                points = self.curves.get(valve.curve)
                if points is None:
                    raise ValueError(
                        f"WaterNetwork: GPV {valve.name!r} names head-loss curve "
                        f"{valve.curve!r}, which has no [CURVES] entry"
                    )
                flows = [x for x, _ in points]
                if len(points) < 2 or any(b <= a for a, b in zip(flows, flows[1:], strict=False)):
                    raise ValueError(
                        f"WaterNetwork: GPV {valve.name!r} uses curve {valve.curve!r}, "
                        f"which needs at least two points with increasing flows"
                    )
        held = [v for v in self.valves if v.kind.upper() in ("PRV", "PSV", "FCV")]
        for k, b in enumerate(held):
            for a in held[:k]:
                if _valve_clash(a, b):
                    raise ValueError(
                        f"WaterNetwork: valve {b.name!r} ({b.kind.upper()} {b.u} -> {b.v}) "
                        f"and valve {a.name!r} ({a.kind.upper()} {a.u} -> {a.v}) hold one "
                        f"node twice: the head a PRV or PSV holds, or the flow an FCV holds, "
                        f"leaves the other's undetermined (EPANET refuses the pair too, "
                        f"error 220)"
                    )
        if self.headloss not in ("H-W", "D-W", "C-M"):
            raise ValueError(
                f"WaterNetwork: headloss {self.headloss!r}; H-W, D-W and C-M are modelled"
            )
        if self.options.demand_model not in ("DDA", "PDA"):
            raise ValueError(
                f"WaterNetwork: DEMAND MODEL {self.options.demand_model!r}; only DDA and "
                f"PDA are modelled"
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


#: Option values of `build_model(friction=...)` named for the reference model, normalised to
#: the physical name: ``"epanet"`` is EPANET 2.2's composite friction law.
_FRICTION_ALIASES = {"epanet": "composite"}


def _rounded_cfs_per_m3s(flow_units: str | None) -> float | None:
    """EPANET's internal cfs per m3/s for a file in `flow_units` (None: exact)."""
    if flow_units is None:
        return None
    from noodl.apps.water.inp import FLOW_UNITS  # the reader imports this module

    return 1.0 / (FLOW_UNITS[flow_units] * EPANET_QCF[flow_units])


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
    friction: str = "composite",
    pipe_form: str | None = None,
    tank_scheme: str = "forward_euler",
    valve_smoothing: float = 0.0,
) -> tuple[Model, State, Drivers]:
    """Assemble the water model and return `(model, state, drivers)`.

    ONE `KirchhoffLayer("water", ...)` on a network whose edge kinds partition by device
    type, plus the tank/controls closure when the network has tanks, plus an optional
    quality layer.

    `pda`, `p_min`, `p_req` and `exponent` each DEFAULT from `net.options` (the file's own
    `[OPTIONS] DEMAND MODEL`/`MINIMUM PRESSURE`/`REQUIRED PRESSURE`/`PRESSURE EXPONENT`,
    or `WaterOptions`'s EPANET defaults when the network was built by hand) when the caller
    passes nothing; passing any of them explicitly still overrides the file.

    `net.options.specific_gravity`/`viscosity` (relative to water at 20 C) reach the
    Darcy-Weisbach path, which is the only element here parameterised by fluid density and
    viscosity at all (`Duct`'s Reynolds-number friction factor, from the KINEMATIC
    viscosity; `scale`, below, uses the density for the head/pressure conversion, and it
    cancels from the head loss exactly as in EPANET).
    Hazen-Williams and Chezy-Manning take neither -- EPANET's own formulas are calibrated for
    water. A NON-DEFAULT `VISCOSITY` on such a network is refused by name rather than silently
    ignored. `SPECIFIC GRAVITY` is accepted: EPANET uses it there only to convert a pressure
    to a head (`Ucf[PRESSURE]`), which the reader applies to every pressure in the file
    (PDA's limits, emitter coefficients, valve settings, control and rule values), so the
    heads and flows do not depend on it (`test_specific_gravity_on_an_si_file_matches_epanet`).

    `quality` is a bulk decay coefficient in 1/day (EPANET's own `[REACTIONS] Global Bulk`
    units, Manual Table 8.4 p.78), or `None`. TWO measured facts go with it.
    First, the junction DEMAND must enter the transport layer as a
    first-order removal rate `q_demand,j / V_j` (1/s): without it the advective generator
    has non-zero row sums -- a junction's edge inflow exceeds its edge outflow by exactly
    its demand -- and the steady system is SINGULAR (`torch.linalg.solve` raises on the
    committed two-loop fixture). Second, a junction's capacity on a LOOPED network is HALF
    the volume of every incident pipe; "the volume of its single outgoing pipe" is a TREE
    property and does not carry over. Both are recorded in `model.notes`.

    `friction` selects the Darcy-Weisbach friction law. `"composite"` (the default) is
    `CompositeDarcyWeisbach`, EPANET 2.2's own composite -- Hagen-Poiseuille below Re = 2000,
    Dunlop's cubic to 4000, Swamee-Jain above -- with EPANET's constants (32.2 ft/s^2,
    0.02517, `VISCOS = 1.1e-5 ft^2/s`) and, when `net.options.flow_units` is known, its
    rounded flow-unit factor; this is the default because the application's reference is
    EPANET, and a `.inp` with `HEADLOSS D-W` then reproduces EPANET to its float32 output.
    `"colebrook"` keeps the generic `Duct` (Colebrook, laminar line below Re = 2000, water
    at 1.002e-3 / 998.2 m^2/s) for users who want the implicit Colebrook law instead.
    Alias: `friction="epanet"` for `"composite"` (EPANET).

    The Darcy-Weisbach path works in PRESSURE (`Duct` is written for it), so the whole
    layer does: heads are multiplied by `rho g` on the way in and `model.head_scale`
    carries the factor back out. Hazen-Williams (the default) works directly in metres of
    head and `head_scale` is 1.

    `pipe_form` is how the pipes enter the solve. `"flow"` writes them in their resistance
    form, `dp = h(q)` (`FlowControlledPipe`), with every pipe's flow an unknown, solved
    from EPANET's own first iterate (every pipe at 1 ft/s,
    `model.initial_flows`): one more unknown per pipe, and a pipe slope that goes to
    EPANET's `RQtol` floor at zero flow. `"potential"` gives each pipe's flow as a function
    of its head loss, `q = g(dp)`, with no flow unknowns, whose slope grows without bound as a
    pipe's flow goes to zero: on Net1's 24 h run the step where a control closes the pump
    and leaves pipe 10 dead-ended needs 136 Newton iterations there and 6 in the flow form
    (`benchmarks/water_eps.py`, which also runs 2.5 times faster). `None` (the default)
    takes `"flow"`, or `"potential"` with `friction="colebrook"`, which gives no head loss
    as a function of the flow.

    Pumps enter in their head-gain form whatever the `pipe_form`: `FlowControlledPump`,
    `phi_discharge - phi_suction = gain(q)` with the pump's flow an unknown, as EPANET writes
    them, each on EPANET's own curve for it (`_pump_kind`: a power function, straight
    segments, or constant power) with EPANET's `RQtol` floor and `CBIG` cap on the curve's
    slope, and shut, with exactly zero flow, where the head asked of it exceeds its
    shut-off head. The inverted curve, `q = g(dp)`, has a zero slope beyond shut-off that
    leaves a pump with a pressure-driven demand downstream singular (`FlowControlledPump`'s
    docstring).

    Every tank is a `LinearStorage` of its cross-section area in the layer
    (`noodl.NodeStorage`), or, when any tank has a volume curve, every tank is one
    `TankVolumeCurve` (a cylinder as the two-point curve of its area), and `tank_scheme` is
    how its level moves. `"forward_euler"` (the
    default) is EPANET's: the tank is a boundary node the solve holds at its step-start head,
    and the step then moves it by `dt * inflow / A` (`KirchhoffLayer.advance_storage`;
    `water_run` takes EPANET's steps, shortened to the next control crossing).
    `"backward_euler"` or `"trapezoidal"` make every tank an interior node instead, its level
    stepped implicitly inside the solve (`water_step`) and differentiable through time; that
    step differs from EPANET's explicit one by the step's own discretisation error, so it is
    not the EPANET comparison mode, and it is not available with `quality`, a volume curve
    or overflow. Under `"forward_euler"` a full or empty tank closes the links that would
    fill or drain it, as EPANET does (`TankLevels`). Either way the
    levels are read from the heads (`"water.phi"`, `tank_levels`), the inflows from
    `"water.storage_flow"` (`tank_inflow`), and the level-triggered controls are closures
    (`TankLevels`, `TankStorageControls`).

    A PRV or PSV is `PressureReducingValve` or `PressureSustainingValve` (edge kinds `"prv"`
    and `"psv"`): one complementarity condition on the head at its controlled node, solved
    exactly by default. `valve_smoothing` (m of head, >= 0) smooths that condition over a
    band of its width instead (`_PressureValve`), which moves the answer by about that much
    near a switch and makes it differentiable across one.
    """
    net.validate()
    if not valve_smoothing >= 0:
        raise ValueError(f"build_model: valve_smoothing must be >= 0 (m), got "
                         f"{valve_smoothing!r}")
    if tank_scheme not in ("forward_euler", "backward_euler", "trapezoidal"):
        raise ValueError(
            f"build_model: tank_scheme must be 'forward_euler', 'backward_euler' or "
            f"'trapezoidal', got {tank_scheme!r}"
        )
    held = tank_scheme == "forward_euler"
    if not held:
        for flag, what in ((lambda t: t.volume_curve is not None, "a volume curve"),
                           (lambda t: t.overflow, "overflow allowed")):
            bad = [t.name for t in net.tanks if flag(t)]
            if bad:
                raise ValueError(
                    f"build_model: tank(s) {bad} have {what}, which only "
                    f"tank_scheme='forward_euler' models (EPANET's step: a tank held through "
                    f"the solve, then moved by volume and kept within its levels)")
    if net.tanks and not held and quality is not None:
        raise ValueError(
            f"build_model: tank_scheme={tank_scheme!r} with quality is not implemented (the "
            f"quality layer treats tanks as boundaries); use tank_scheme='forward_euler'"
        )
    if pipe_form not in (None, "potential", "flow"):
        raise ValueError(
            f"build_model: pipe_form must be 'potential' or 'flow', got {pipe_form!r}"
        )
    headloss = headloss or net.headloss
    if headloss not in ("H-W", "D-W", "C-M"):
        raise ValueError(
            f"build_model: headloss must be 'H-W', 'D-W' or 'C-M', got {headloss!r}"
        )
    friction = _FRICTION_ALIASES.get(friction, friction)
    if pipe_form is None:
        pipe_form = "potential" if headloss == "D-W" and friction == "colebrook" else "flow"
    if friction not in ("composite", "colebrook"):
        raise ValueError(
            f"build_model: friction must be 'composite' or 'colebrook', got {friction!r}"
        )
    if pipe_form == "potential":
        acted = ({a.link for r in net.rules for a in (*r.then_actions, *r.else_actions)}
                 | {c.link for c in net.controls})
        switched = [p.name for p in net.pipes
                    if p.status.upper() != "OPEN" or p.name in net.initial_status
                    or p.name in acted]
        if switched:
            raise ValueError(
                f"build_model: pipe(s) {switched} are closed or check valves (a CLOSED or CV "
                f"status, a [STATUS] line, a control or a rule's action), which a pipe takes "
                f"only in its resistance form: pipe_form='potential' gives each pipe's flow "
                f"from its head loss and reads no status; use pipe_form='flow'"
            )
    options = net.options
    if pda is None:
        pda = options.demand_model == "PDA"
    if p_min is None:
        p_min = options.minimum_pressure
    if p_req is None:
        p_req = options.required_pressure
    if (pda or net.emitters) and any(
            p.attribute == "DEMAND" for r in net.rules for p in r.premises):
        raise ValueError(
            "build_model: a rule premise reads DEMAND, which `apply_rules` takes from the "
            "base demands in 'water.sources'; on a pressure-driven network or one with "
            "emitters EPANET's demand is the delivered one, which is not read here")
    if exponent is None:
        exponent = options.pressure_exponent
    if headloss in ("H-W", "C-M") and options.viscosity != 1.0:
        name = "Hazen-Williams" if headloss == "H-W" else "Chezy-Manning"
        raise ValueError(
            f"build_model: [OPTIONS] VISCOSITY {options.viscosity!r} is not the default, but "
            f"{name} does not take the fluid's viscosity (EPANET's own {headloss} formula "
            f"is calibrated for water); use headloss='D-W' or leave it at 1.0"
        )
    # EPANET 2.2: SPECIFIC GRAVITY is the density ratio and VISCOSITY the KINEMATIC
    # viscosity ratio to water at 20 C, so rho = RHO_W sg and nu = (MU_W / RHO_W) visc.
    # `Duct` is written for MASS flow (F = rho Q, Re = F D / (mu A)); fed rho' = 1 / rho
    # and mu' = nu at dp = rho g h it returns the VOLUMETRIC flow Q = A sqrt(2 g h D /
    # (f L)) with Re = V D / nu, which is what this layer balances (demands in m3/s).
    # Specific gravity then cancels from the head loss, as in EPANET.
    rho = RHO_W * options.specific_gravity
    nu = MU_W / RHO_W * options.viscosity
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
    tcv, fcv, pbv, gpv, prv, psv = (net.valves_of(kind) for kind in _VALVES_MODELLED)
    for kind in _VALVES_MODELLED:
        for valve in net.valves_of(kind):
            graph.add_edge(valve.u, valve.v, kind=_VALVE_KINDS[kind], name=valve.name)
    # Each kind's columns of the per-link drivers, in `net.links()` (= layer column) order.
    link_slice: dict[str, slice] = {}
    start = 0
    for kind, count in (("pipe", len(net.pipes)), ("pump", len(net.pumps)),
                        *((_VALVE_KINDS[k], len(net.valves_of(k))) for k in _VALVES_MODELLED)):
        link_slice[kind] = slice(start, start + count)
        start += count

    notes: dict[str, str] = dict(net.notes)
    scale = rho * G if headloss == "D-W" else 1.0
    # EPANET's internal cfs per m3/s, and its RQtol in the layer's potential per m3/s.
    epanet_cfs = _rounded_cfs_per_m3s(options.flow_units)
    epanet_cfs = 1.0 / M_PER_FT**3 if epanet_cfs is None else epanet_cfs
    rqtol = RQTOL_FT_PER_CFS * M_PER_FT * epanet_cfs * scale
    laws: list = []
    # the check valves and pumps shut on each Newton iterate, for one cut-off test over both
    shut_links = ShutLinks(len(net.links()), pumps=link_slice["pump"])
    if net.pipes:
        if headloss == "H-W":
            laws.append(
                HazenWilliams(
                    torch.tensor([p.length for p in net.pipes], dtype=F64),
                    torch.tensor([p.diameter for p in net.pipes], dtype=F64),
                    torch.tensor([p.roughness for p in net.pipes], dtype=F64),
                    minor_loss=torch.tensor([p.minor_loss for p in net.pipes], dtype=F64),
                    cfs_per_m3s=_rounded_cfs_per_m3s(options.flow_units), kind="pipe",
                )
            )
        elif headloss == "C-M":
            laws.append(
                ChezyManning(
                    torch.tensor([p.length for p in net.pipes], dtype=F64),
                    torch.tensor([p.diameter for p in net.pipes], dtype=F64),
                    torch.tensor([p.roughness for p in net.pipes], dtype=F64),
                    minor_loss=torch.tensor([p.minor_loss for p in net.pipes], dtype=F64),
                    cfs_per_m3s=_rounded_cfs_per_m3s(options.flow_units), kind="pipe",
                )
            )
        else:
            lengths = torch.tensor([p.length for p in net.pipes], dtype=F64)
            diameters = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
            roughness = torch.tensor([p.roughness for p in net.pipes], dtype=F64)
            minor = torch.tensor([p.minor_loss for p in net.pipes], dtype=F64)
            if friction == "composite":
                cfs = _rounded_cfs_per_m3s(options.flow_units)
                nu_ref = NU_WATER_REF * options.viscosity
                notes["headloss"] = (
                    f"Darcy-Weisbach with EPANET 2.2's own friction law "
                    f"(CompositeDarcyWeisbach: Hagen-Poiseuille below Re = 2000, Dunlop's "
                    f"cubic to 4000, Swamee-Jain above) at kinematic viscosity "
                    f"nu = {nu_ref} m2/s (EPANET's 1.1e-5 ft2/s scaled by [OPTIONS] "
                    f"VISCOSITY), flows converted to cfs with "
                    + (
                        f"EPANET's own {options.flow_units} factor ({cfs} cfs per m3/s)"
                        if cfs is not None
                        else "the exact 1 / 0.3048^3 (no file flow unit)"
                    )
                )
                laws.append(
                    CompositeDarcyWeisbach(
                        lengths, diameters, roughness, minor_loss=minor,
                        nu=nu_ref, cfs_per_m3s=cfs, scale=scale, kind="pipe",
                    )
                )
            else:
                notes["headloss"] = (
                    f"Darcy-Weisbach with the generic Duct law (Colebrook, unrolled "
                    f"fixed point) at kinematic viscosity nu = {nu} m2/s (1.002e-3 / "
                    f"998.2 scaled by [OPTIONS] VISCOSITY; SPECIFIC GRAVITY "
                    f"{options.specific_gravity} sets only the head-to-pressure scale); "
                    f"this is NOT EPANET's law -- friction='composite' is"
                )
                laws.append(
                    Duct(
                        lengths, diameters, roughness,
                        # EPANET's minor loss K adds K V^2 / (2 g): Duct's sum_C term,
                        # which at rho' = 1/rho and dp = rho g h is exactly that head.
                        sum_C=minor, rho=1.0 / rho, mu=nu, n_iter=12, kind="pipe",
                    )
                )
    if net.pipes and pipe_form == "flow":
        if headloss == "D-W" and friction != "composite":
            raise ValueError(
                "build_model: pipe_form='flow' needs the pipes' head loss as a function of "
                "the flow, which friction='colebrook' (the generic Duct) does not give; use "
                "friction='composite'"
            )
        pipe = laws[0]
        # kappa: each pipe's own head-loss slope at 1 ft/s, EPANET's first iterate
        # (`hydraul.c`, `initlinkflow`), in the layer's potential per m3/s.
        with torch.enable_grad():
            q_ft = torch.tensor([torch.pi * p.diameter**2 / 4.0 * M_PER_FT for p in net.pipes],
                                dtype=F64, requires_grad=True)
            (slope,) = torch.autograd.grad(pipe.head_loss(q_ft).sum(), q_ft)
        laws[0] = FlowControlledPipe(
            pipe, gradient_floor=RQTOL_FT_PER_CFS * M_PER_FT * epanet_cfs,
            kappa=getattr(pipe, "scale", 1.0) * slope.detach(), links=link_slice["pipe"],
            ends=[(node_names.index(link.u), node_names.index(link.v))
                  for link in net.links()],
            fixed=[node_names.index(n.name) for n in net.fixed_head_nodes()],
            shut_links=shut_links, kind="pipe")
    if net.pumps:
        pump_law, q_design_list = _pump_law(net, link_slice["pump"], scale, epanet_cfs)
        # kappa of a closed pump: 1 m per m3/s, the row `q` a closed pump had before. A pump
        # has no head-loss slope at 1 ft/s; its curve's slope at the design flow
        # (`2 r q_design`) solves to the same answer but, through Newton's damping switch
        # (which reads the residual norm), costs Net1's 24 h run 3246 Newton iterations
        # against 2952 in the potential pipe form (MEASURED, benchmarks/water_eps.py).
        pump_kappa = torch.full((len(net.pumps),), scale, dtype=F64)
        laws.append(FlowControlledPump(
            pump_law, gradient_floor=rqtol,
            max_slope=CBIG_FT_PER_CFS * M_PER_FT * epanet_cfs * scale, kappa=pump_kappa,
            links=link_slice["pump"],
            ends=[(node_names.index(link.u), node_names.index(link.v))
                  for link in net.links()],
            fixed=[node_names.index(n.name) for n in net.fixed_head_nodes()],
            shut_links=shut_links, kind="pump"))
    rounded_cfs = _rounded_cfs_per_m3s(options.flow_units)

    def valve_terms(valves) -> dict:
        """The open-valve minor loss of `valves` (EPANET's `Km`, the layer's potential per
        (m3/s)^2) and their kappa: that loss's slope at 1 ft/s, at least 1 m per m3/s (a
        valve with no minor loss has no head-loss slope of its own)."""
        diameter = torch.tensor([v.diameter for v in valves], dtype=F64)
        minor = scale * epanet_minor_coefficient(
            torch.tensor([v.minor_loss for v in valves], dtype=F64), diameter, rounded_cfs)
        q_ft = torch.pi * diameter**2 / 4.0 * M_PER_FT
        return {"minor": minor, "kappa": torch.clamp(2.0 * minor * q_ft, min=scale),
                "gradient_floor": rqtol}

    if tcv:
        laws.append(ThrottleValve(
            torch.tensor([v.diameter for v in tcv], dtype=F64), cfs_per_m3s=rounded_cfs,
            scale=scale, links=link_slice["valve"], kind="valve", **valve_terms(tcv)))
    if fcv:
        laws.append(FlowControlValve(links=link_slice["fcv"], kind="fcv", **valve_terms(fcv)))
    if pbv:
        laws.append(BreakerValve(None, links=link_slice["pbv"], kind="pbv",
                                 **valve_terms(pbv)))
    if gpv:
        curves = [net.curves[v.curve] for v in gpv]
        width = max(len(c) for c in curves)
        flows = torch.full((len(gpv), width), torch.inf, dtype=F64)
        heads = torch.zeros((len(gpv), width), dtype=F64)
        for i, points in enumerate(curves):
            flows[i, : len(points)] = torch.tensor([x for x, _ in points], dtype=F64)
            heads[i, : len(points)] = torch.tensor([y * scale for _, y in points], dtype=F64)
        laws.append(GeneralPurposeValve(flows, heads, [len(c) for c in curves],
                                        links=link_slice["gpv"], kind="gpv",
                                        **valve_terms(gpv)))
    elevation = {j.name: j.elevation for j in net.junctions}
    for valves, cls, kind, end in ((prv, PressureReducingValve, "prv", "v"),
                                   (psv, PressureSustainingValve, "psv", "u")):
        if valves:
            # the controlled node's elevation: a PRV's end node, a PSV's start node
            controlled = torch.tensor(
                [elevation[getattr(v, end)] * scale for v in valves], dtype=F64)
            laws.append(cls(controlled, smoothing=valve_smoothing * scale,
                            links=link_slice[kind], kind=kind, **valve_terms(valves)))

    storages: list = []
    boundary = [r.name for r in net.reservoirs]
    if net.tanks:
        tank_idx = torch.tensor([node_names.index(t.name) for t in net.tanks], dtype=torch.long)
        if any(t.volume_curve is not None for t in net.tanks):
            # one storage for every tank, in tank order; a cylinder is the two-point curve
            # of its area, linear over every level it can take
            curves = [net.curves[t.volume_curve] if t.volume_curve is not None else
                      ((0.0, 0.0), (2.0 * t.max_level + 1.0,
                                    torch.pi * t.diameter**2 / 4.0 * (2.0 * t.max_level + 1.0)))
                      for t in net.tanks]
            storages.append(TankVolumeCurve(
                tank_idx, torch.tensor([t.elevation for t in net.tanks], dtype=F64),
                [[x for x, _ in c] for c in curves], [[y for _, y in c] for c in curves],
                head_scale=scale))
        else:
            storages.append(LinearStorage(
                tank_idx,
                torch.tensor([torch.pi * t.diameter**2 / 4.0 / scale for t in net.tanks],
                             dtype=F64),
                initial=torch.tensor([(t.elevation + t.init_level) * scale
                                      for t in net.tanks], dtype=F64)))
        if held:
            boundary += [t.name for t in net.tanks]
    withdrawals: list = []
    if pda:
        if not net.junctions:
            raise ValueError("build_model: pda=True on a network with no junctions")
        withdrawals.append(
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
    if net.emitters:
        exponent_e = options.emitter_exponent
        by_node = {j.name: j for j in net.junctions}
        withdrawals.append(
            EmitterOutflow(
                torch.tensor([node_names.index(e.node) for e in net.emitters],
                             dtype=torch.long),
                # in the layer's potential (Pa for D-W): C (p / scale)^n = (C / scale^n) p^n
                torch.tensor([e.coefficient / scale**exponent_e for e in net.emitters],
                             dtype=F64),
                torch.tensor([by_node[e.node].elevation * scale for e in net.emitters],
                             dtype=F64),
                exponent=exponent_e, dp_transition=1e-9 * scale,
            )
        )
        notes["emitters"] = (
            f"emitters at {[e.node for e in net.emitters]} as q = C p^n, n = {exponent_e} "
            f"(EPANET's [EMITTERS] and EMITTER EXPONENT)"
        )
    layer = KirchhoffLayer(
        graph, "water", laws, boundary=boundary, withdrawals=withdrawals,
        quantity="pressure" if headloss == "D-W" else "head",
        unit="Pa" if headloss == "D-W" else "m", flow_unit="m3/s",
        storage=storages, scheme=tank_scheme if net.tanks else "backward_euler",
    )

    layers: dict = {"water": layer}
    closures: list = []
    drivers: Drivers = {}
    state: State = {
        "water.phi": torch.zeros(graph.n, dtype=F64),
        "water.q": torch.zeros(len(layer.cols), dtype=F64),
    }
    if net.tanks:
        state["water.storage_flow"] = torch.zeros(len(net.tanks), dtype=F64)
    if net.pipes and pipe_form == "flow":
        # The first Newton iterate of the resistance form: EPANET's own, every pipe at 1 ft/s
        # (`hydraul.c`, `initlinkflow`). From zero flow every pipe sits on the RQtol floor of
        # its slope and the first Newton step overshoots without bound.
        area = torch.tensor([torch.pi * p.diameter**2 / 4.0 for p in net.pipes], dtype=F64)
        state["water.q"][layer.kind_slice("pipe")] = area * M_PER_FT
    if net.pumps:
        # A pump's first iterate is EPANET's too, its design flow. From zero flow it sits on
        # the `RQtol` tangent of its curve, where it is a near-ideal head source of h0.
        state["water.q"][layer.kind_slice("pump")] = torch.tensor(q_design_list, dtype=F64)
    for valves, kind in ((prv, "prv"), (psv, "psv")):
        if valves:
            # EPANET starts every PRV and PSV ACTIVE (`hydraul.c`, `inithyd`) at 1 ft/s
            # (`initlinkflow`), the pipes' first iterate
            area = torch.tensor([torch.pi * v.diameter**2 / 4.0 for v in valves], dtype=F64)
            state["water.q"][layer.kind_slice(kind)] = area * M_PER_FT

    link_names = [link.name for link in net.links()]
    drivers.update(_link_drivers(net, scale))
    reservoir_heads = torch.tensor([r.head * scale for r in net.reservoirs], dtype=F64)
    control_closure = None
    speed_patterns = [p.name for p in net.pumps if p.speed_pattern is not None]
    if net.tanks or net.controls or net.rules or speed_patterns:
        # the closure of the tanks, the controls, the rules and the pumps' speed patterns
        # (`model.control_closure`, also on a network without tanks)
        junction_names = {j.name for j in net.junctions}
        if net.tanks and not held and any(
                c.trigger == "node" and c.node in junction_names for c in net.controls):
            raise ValueError(
                "build_model: a junction-pressure control acts after a solve, which the "
                "implicit tank schemes' water_step does not repeat; use "
                "tank_scheme='forward_euler' (water_run)")
        kinds = ["PIPE" if isinstance(link, WaterPipe) else
                 "PUMP" if isinstance(link, Pump) else "VALVE" for link in net.links()]
        elevation = {j.name: j.elevation for j in net.junctions}
        control_closure = (TankLevels if (held or not net.tanks) else TankStorageControls)(
            graph, list(net.tanks), list(net.controls), [p.name for p in net.pumps],
            tank_nodes=[node_names.index(t.name) for t in net.tanks],
            dt=net.hydraulic_timestep if dt is None else dt,
            reservoir_heads=reservoir_heads,
            head_scale=scale,
            link_names=link_names,
            link_status=drivers[STATUS_KEY],
            tank_links=_tank_links(net),
            one_way=drivers[ONE_WAY_KEY],
            storage=storages[0] if net.tanks else None,
            link_setting=drivers[SETTING_KEY],
            link_kinds=kinds,
            setting_scale=[scale if (not isinstance(lk, (WaterPipe, Pump))
                                     and lk.kind.upper() in ("PBV", "PRV", "PSV")) else 1.0
                           for lk in net.links()],
            junctions={name: (node_names.index(name), elev)
                       for name, elev in elevation.items()},
            start_clocktime=net.start_clocktime,
            rules=net.rules,
            node_info={name: (node_names.index(name), elev)
                       for name, elev in (*elevation.items(),
                                          *((r.name, r.head) for r in net.reservoirs),
                                          *((t.name, t.elevation) for t in net.tanks))},
            setting_tolerance=[_setting_tolerance(lk, scale) for lk in net.links()],
            speed_patterns=speed_patterns,
        )
        closures.append(control_closure)
        for t in net.tanks:
            state["water.phi"][node_names.index(t.name)] = (t.elevation + t.init_level) * scale
        state["water.link_status"] = drivers[STATUS_KEY].clone()
        state["water.link_setting"] = drivers[SETTING_KEY].clone()
        if any(c.trigger != "node" for c in net.controls):
            drivers[TIME_KEY] = torch.zeros((), dtype=F64)
    else:
        drivers["water.phi_boundary"] = reservoir_heads

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
        # The junctions withdraw whatever the CURRENT hydraulic solution delivers (the
        # requested demand, a scaled or patterned `water.sources`, or the PDA-delivered
        # demand). The layer's `dilution = 1` in "withdrawal" mode adds
        # `max(net_inflow(q), 0) / capacity` to the removal, and at a junction the net inflow
        # of the pass's own flows IS the withdrawal (continuity), so the sink follows the
        # current flows with no closure and no lag. A junction with a net INJECTION
        # (`water.sources > 0`, net pipe inflow < 0) gets no sink: the injected water
        # carries no tracer, so the outflow dilutes it, and any injected tracer is
        # `quality.sources`. Only the bulk decay is a build-time removal.
        removal = (quality / 86400.0) * torch.ones(len(interior), 1, dtype=F64)
        notes["quality_capacity"] = (
            "on a looped network a junction's capacity is HALF the volume of every "
            "incident pipe; a single outgoing pipe is a tree property"
        )
        notes["quality_removal"] = (
            "the nodal withdrawal enters as a first-order removal rate q_withdrawn / V "
            "through the transport layer's dilution = 1 in 'withdrawal' mode (a nodal water "
            "injection carries no tracer and is no sink: injected tracer is "
            "quality.sources), read from the CURRENT flows (so a "
            "scaled or patterned water.sources, or a PDA delivery, is followed exactly); "
            "without it the advective generator has non-zero row sums and the steady "
            "system is singular (measured on twoloop_si.inp). `removal` holds only the "
            "bulk decay"
        )
        # Every hydraulic edge kind carries water, so every one advects the tracer:
        # pipes, pumps and valves alike (a device is a zero-volume edge).
        transport_kinds = tuple(layer.kinds)
        layers["quality"] = TransportLayer(
            graph, "quality", capacity=cap_i, flow_kind=transport_kinds,
            boundary=boundary, n_species=1, removal=removal, scheme="backward_euler",
            quantity="concentration", unit="kg/m3",
            dilution=1.0, dilution_mode="withdrawal",
            source_unit="kg/s", capacity_unit="m3", flow_unit="m3/s",
        )
        drivers["quality.x_boundary"] = torch.zeros(len(boundary), dtype=F64)
        state["quality.x"] = torch.zeros(len(interior), dtype=F64)

    model = Model(graph, layers, closures=closures, coupling=coupling)
    model.notes = notes
    model.node_names = node_names
    model.link_names = link_names
    model.input_specs = {
        STATUS_KEY: dict(
            description="each link's status: CLOSED 0, OPEN 1, ACTIVE 2 (a valve acting on "
                        "its setting)", over=link_names, ordering="model.link_names"),
        SETTING_KEY: dict(
            description="each link's setting: a pump's relative speed, a PBV's head drop, "
                        "a PRV's or PSV's pressure head (the layer's potential unit), an "
                        "FCV's flow (m3/s), a TCV's loss coefficient; unused for pipes and "
                        "GPVs", over=link_names, ordering="model.link_names"),
        ONE_WAY_KEY: dict(
            description="each link's allowed flow direction: 0 both, +1 forward only, -1 "
                        "reverse only", over=link_names, ordering="model.link_names"),
    }
    if net.tanks:
        model.input_specs[TANK_INFLOW_KEY] = dict(
            unit="m3/s",
            description="each tank's inflow (m3/s) of the previous solve, the one-second "
                        "margin of the level controls; taken from the state's "
                        "water.storage_flow when not given",
            over=[t.name for t in net.tanks], ordering="net.tanks", required=False)
    if net.reservoirs and control_closure is not None:
        model.input_specs[RESERVOIR_HEAD_KEY] = dict(
            description="each reservoir's head (the layer's potential unit) in place of the "
                        "built one; water_run sets it from the head patterns",
            over=[r.name for r in net.reservoirs], ordering="net.reservoirs", required=False)
    if pda:
        model.input_specs[DEMAND_FACTOR_KEY] = dict(
            description="each junction's multiplier of its pressure-driven required demand; "
                        "water_run sets it from the demand patterns",
            over=[j.name for j in net.junctions], ordering="net.junctions", required=False)
    if TIME_KEY in drivers:
        model.input_specs[TIME_KEY] = dict(
            description="seconds since the simulation start, which the network's AT TIME and "
                        "AT CLOCKTIME controls read (set it each step)")
    # `initial_state` starts from these flows (resistance-form pipes need a first iterate).
    model.initial_flows = state["water.q"].clone()
    model.control_closure = control_closure
    model.tank_scheme = tank_scheme
    model.head_scale = scale
    model.water_network = net
    model.driver_template = dict(drivers)
    return model, state, drivers


def _tank_links(net: WaterNetwork) -> list[tuple[int, int, bool, str]]:
    """`(link, tank, at_start, kind)` for every link EPANET examines for a full or empty
    tank (`hydstatus.c`, `tankstatus`, `TankLevels`): the start node when it is a tank or a
    reservoir (a reservoir is then skipped, and the end node never examined), else the end
    node when it is a tank."""
    tank = {t.name: i for i, t in enumerate(net.tanks)}
    reservoirs = {r.name for r in net.reservoirs}
    out = []
    for j, link in enumerate(net.links()):
        kind = ("pipe" if isinstance(link, WaterPipe) else
                "pump" if isinstance(link, Pump) else "valve")
        if link.u in tank:
            out.append((j, tank[link.u], True, kind))
        elif link.u not in reservoirs and link.v in tank:
            out.append((j, tank[link.v], False, kind))
    return out


def _pump_kind(net: WaterNetwork, pump: Pump) -> str:
    """EPANET 2.2's type of `pump` (`input2.c`, `updatepumpparams`): `"power"` for a
    constant-power pump (``CONST_HP``), `"curve"` for a power function (``POWER_FUNC``: one
    point, or three with the first at zero flow), `"segments"` for every other head curve
    (``CUSTOM``)."""
    if pump.power is not None:
        return "power"
    points = net.curves[pump.curve]
    if len(points) == 1 or (len(points) == 3 and points[0][0] == 0.0):
        return "curve"
    return "segments"


#: EPANET's ``TINY`` (`types.h`), the tolerance of its pump-curve checks.
_TINY = 1e-6


def power_curve(points) -> tuple[float, float, float]:
    """`(h0, r, n)` of EPANET 2.2's power-function pump curve ``h = h0 - r q^n`` through
    `points` (`(flow, head)` pairs): one point, or three with the first at zero flow
    (`input2.c`, `updatepumpparams`; `input3.c`, `powercurve`).

    One point ``(q1, h1)``: EPANET adds the points ``(0, 1.33334 h1)`` and ``(2 q1, 0)``
    and fits the three (`three_point_curve`, whose exponent is 1.99998). Three points
    ``(0, h0), (q1, h1), (q2, h2)``: ``n = ln((h0 - h2) / (h0 - h1)) / ln(q2 / q1)``,
    ``r = (h0 - h1) / q1^n``. Raises `ValueError` where `powercurve` fails (EPANET's error
    227): heads not falling or flows not rising by more than ``TINY``, or ``n`` outside
    ``(0, 20]``. EPANET applies ``TINY`` to the file's units; here it applies to SI."""
    if len(points) == 1:
        (q1, h1), = points
        h0, r, n = three_point_curve(torch.tensor(q1, dtype=F64), torch.tensor(h1, dtype=F64))
        return float(h0), float(r), float(n)
    (_, h0), (q1, h1), (q2, h2) = points
    if (h0 < _TINY or h0 - h1 < _TINY or h1 - h2 < _TINY or q1 < _TINY
            or q2 - q1 < _TINY):
        raise ValueError("its heads must fall and its flows rise from point to point "
                         "(EPANET error 227)")
    n = math.log((h0 - h2) / (h0 - h1)) / math.log(q2 / q1)
    if not 0.0 < n <= 20.0:
        raise ValueError(f"its fitted exponent {n} is outside (0, 20] (EPANET error 227)")
    return h0, (h0 - h1) / q1**n, n


def _pump_law(net: WaterNetwork, links: slice, scale: float, epanet_cfs: float):
    """`(pump, q_design)`: the curve (or `(curve, columns)` parts) of `FlowControlledPump`
    for every pump of `net`, by EPANET's type (`_pump_kind`), and each pump's first
    iterate, EPANET's own (`hydraul.c`, `initlinkflow`: the speed times ``Q0``, which
    `updatepumpparams` sets to the design flow of a power function, the middle of a
    segmented curve's flows, and 1 cfs for a constant-power pump)."""
    from noodl.apps.water.inp import _US_UNITS, HP_TO_KW  # the reader imports this module

    groups: dict[str, list[int]] = {"curve": [], "segments": [], "power": []}
    q_design = []
    for i, pump in enumerate(net.pumps):
        kind = _pump_kind(net, pump)
        groups[kind].append(i)
        if kind == "power":
            q0 = 1.0 / epanet_cfs
        else:
            points = net.curves[pump.curve]
            q0 = points[len(points) // 2][0] if kind == "curve" else (
                0.5 * (points[0][0] + points[-1][0]))
        q_design.append(q0 * pump.speed)
    single = sum(bool(cols) for cols in groups.values()) == 1
    own = links if single else None
    parts = []
    for kind, cols in groups.items():
        if not cols:
            continue
        pumps = [net.pumps[i] for i in cols]
        speed = torch.tensor([p.speed for p in pumps], dtype=F64)
        if kind == "curve":
            fits = [power_curve(net.curves[p.curve]) for p in pumps]
            curve = PumpCurve(torch.tensor([h0 * scale for h0, _, _ in fits], dtype=F64),
                              torch.tensor([r * scale for _, r, _ in fits], dtype=F64),
                              torch.tensor([n for _, _, n in fits], dtype=F64),
                              speed=speed, links=own, kind="pump")
        elif kind == "segments":
            points = [net.curves[p.curve] for p in pumps]
            width = max(len(c) for c in points)
            flows = torch.full((len(pumps), width), torch.inf, dtype=F64)
            heads = torch.zeros((len(pumps), width), dtype=F64)
            for k, curve_points in enumerate(points):
                flows[k, : len(curve_points)] = torch.tensor([x for x, _ in curve_points],
                                                             dtype=F64)
                heads[k, : len(curve_points)] = torch.tensor(
                    [y * scale for _, y in curve_points], dtype=F64)
            curve = PiecewisePumpCurve(flows, heads, [len(c) for c in points], speed=speed,
                                       links=own, kind="pump")
        else:
            us = net.options.flow_units is None or net.options.flow_units in _US_UNITS
            curve = ConstantPowerPump(
                torch.tensor([p.power for p in pumps], dtype=F64),
                kw_per_hp=HP_TO_KW if us else KW_PER_HP, cfs_per_m3s=epanet_cfs,
                scale=scale, speed=speed, links=own, kind="pump")
        parts.append((curve, cols))
    return (parts[0][0] if single else parts), q_design


def _link_drivers(net: WaterNetwork, scale: float) -> Drivers:
    """`"water.status"`, `"water.setting"` and `"water.one_way"` from the network, one value
    per link in `net.links()` order.

    A pipe is OPEN, or CLOSED by its status column; a CV pipe is OPEN and forward-only. A
    pump is OPEN at its speed. A valve is ACTIVE at its setting (a PBV's, PRV's or PSV's
    scaled to the layer's potential unit). A `[STATUS]` word then sets the status, and a numeric
    `[STATUS]` value a valve's setting, the valve staying ACTIVE (EPANET 2.2 Manual,
    Appendix C.2.17)."""
    links = net.links()
    status = torch.full((len(links),), float(OPEN), dtype=F64)
    setting = torch.zeros(len(links), dtype=F64)
    one_way = torch.zeros(len(links), dtype=F64)
    for i, link in enumerate(links):
        kind = "PIPE" if isinstance(link, WaterPipe) else (
            "PUMP" if isinstance(link, Pump) else link.kind.upper())
        unit = scale if kind in ("PBV", "PRV", "PSV") else 1.0
        if kind == "PIPE":
            if link.status.upper() == "CLOSED":
                status[i] = CLOSED
            elif link.status.upper() == "CV":
                one_way[i] = 1.0
        elif kind == "PUMP":
            setting[i] = link.speed
        else:
            status[i] = ACTIVE
            setting[i] = 0.0 if kind == "GPV" else link.setting * unit
        initial = net.initial_status.get(link.name)
        if isinstance(initial, str):
            status[i] = status_code(initial)
        elif initial is not None:
            setting[i] = initial * unit
            status[i] = ACTIVE
    return {STATUS_KEY: status, SETTING_KEY: setting, ONE_WAY_KEY: one_way}


def initial_state(model: Model, *, values: Mapping | None = None) -> State:
    """All-zero state, dispatching on each layer's `quantity` (the app convention), tank
    levels from the network file; then `values` by node, tank or pump name (see
    `noodl.refs.state_from`).

    A tank's level is its head in `"water.phi"` less its bottom. `values` may set levels by
    tank name under the key `"water.tank_level"`, which is read here only (it is no state
    key) and written into `"water.phi"` at the tanks:
    `initial_state(model, values={"water.tank_level": {"T1": 4.0}})`.
    """
    levels = None
    if values and "water.tank_level" in values:
        values = dict(values)
        levels = values.pop("water.tank_level")
    state: State = {}
    for name, layer in model.kirchhoff.items():
        if layer.quantity not in ("head", "pressure"):
            raise ValueError(
                f"apps.water.initial_state: Kirchhoff layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds head or pressure layers only"
            )
        state[f"{name}.phi"] = torch.zeros(model.net.n, dtype=F64)
        state[f"{name}.q"] = torch.zeros(len(layer.cols), dtype=F64)
        if layer.storage:
            state[f"{name}.storage_flow"] = torch.zeros(
                sum(st.nodes.numel() for st in layer.storage), dtype=F64)
    if getattr(model, "initial_flows", None) is not None:
        state["water.q"] = model.initial_flows.clone()   # `build_model`'s first iterate
    for name, layer in model.transport.items():
        if layer.quantity != "concentration":
            raise ValueError(
                f"apps.water.initial_state: transport layer {name!r} has quantity "
                f"{layer.quantity!r}; this application builds concentration layers only"
            )
        state[f"{name}.x"] = torch.zeros(layer.n_i, dtype=F64)
    closure = model.control_closure
    if closure is not None:
        init = torch.tensor([t.init_level for t in closure.tanks], dtype=F64)
        state["water.phi"][..., closure.tank_nodes] = (closure.bottom + init) * closure.head_scale
        state["water.link_status"] = model.driver_template[STATUS_KEY].clone()
        state["water.link_setting"] = model.driver_template[SETTING_KEY].clone()
    if levels:
        state["water.phi"] = _set_tank_levels(model, state["water.phi"], levels)
    if values:
        state = state_from(model, values, base=state)
    return state


def _set_tank_levels(model: Model, phi: Tensor, levels: Mapping) -> Tensor:
    """`phi` with the head of every tank named in `levels` (`{tank: m above its bottom}`)
    set from its level."""
    closure = model.control_closure
    if closure is None or not isinstance(levels, Mapping):
        raise ValueError(
            "initial_state: 'water.tank_level' takes {tank name: level (m)}"
            + ("" if closure is not None else ", and this network has no tanks")
        )
    unknown = [name for name in levels if name not in closure.tank_of]
    if unknown:
        raise KeyError(
            f"initial_state: 'water.tank_level' names {unknown}, which are not tanks of this "
            f"network ({list(closure.tank_of)})"
        )
    phi = phi.clone()
    for name, level in levels.items():
        i = closure.tank_of[name]
        phi[..., closure.tank_nodes[i]] = (closure.bottom[i] + level) * closure.head_scale
    return phi


def initial_drivers(
    model: Model, *, values: Mapping | None = None, demand: Mapping | None = None,
) -> Drivers:
    """`build_model`'s driver template (copied), then `values` by name (see
    `noodl.refs.drivers_from`), then `demand`: `{junction: m3/s withdrawn}`, positive,
    stored as the negative nodal source the head solve takes.

        initial_drivers(model, demand={"J1": 0.005})
        initial_drivers(model, values={"water.sources": {"J1": -0.005}})   # the same

    A pressure-driven model (`pda=True`) refuses `demand=` and a `water.sources` value: its
    demands are withdrawals the network set at build time. On a model built with `quality=`,
    the quality sink follows the withdrawals of the current hydraulic solution, so a changed
    `demand=` or `water.sources` changes the hydraulics and the quality together.
    """
    drivers = refs_initial_drivers(model, values=values)
    if demand or (values and "water.sources" in values):
        layer = model.kirchhoff["water"]
        if any(isinstance(s, PressureDrivenDemand) for s in layer.withdrawals):
            raise ValueError(
                "initial_drivers: the demands of this model are pressure-driven and were "
                "set by the network at build time, so demand= or a water.sources value "
                "cannot change them"
            )
    if demand:
        key = "water.sources"
        drivers = drivers_from(model, {key: {n: -q for n, q in demand.items()}}, base=drivers)
    return drivers


def _setting_tolerance(link, head_scale: float) -> float:
    """EPANET's tolerance on a setting a rule sets (`rules.c`, `takeactions`: 1e-3 in
    internal units, feet and cubic feet per second) in the unit the closure carries it in: a
    PRV's, PSV's or PBV's head in the layer's potential unit, an FCV's flow in m3/s, a
    pump's speed and a TCV's coefficient as they are."""
    kind = getattr(link, "kind", "").upper()
    if kind in ("PRV", "PSV", "PBV"):
        return 1e-3 * M_PER_FT * head_scale
    if kind == "FCV":
        return 1e-3 * M_PER_FT**3
    return 1e-3


def apply_rules(model: Model, state: State, drivers: Drivers, *, time,
                step: float) -> tuple[State, Tensor]:
    """`(state, changed)`: `state` after the network's `[RULES]` are tested on it at `time` (s
    since the start) and the actions that stand are taken (EPANET 2.2, `rules.c`,
    `checkrules`; `evaluate_rules`, `take_actions`); `changed` says per instance whether any
    action was TAKEN, EPANET's action count (`takeactions`), which asks for a new solve. It is
    not movement of the carried values: a pump the solver shut at its shut-off head is closed
    to `STATUS IS OPEN`, which is then taken (and `changed`) on every check while the pump
    stays shut, though the carried status is OPEN throughout. `water_run` follows
    `ruletimestep`: a taken action ends the step, the next solve is at the action's time, and
    the rules are next checked a rule step later -- never again at the same time until
    nothing changes. `step` (s, positive) is the
    length of the interval since the previous check (the rule time step), which `SYSTEM TIME`
    and `CLOCKTIME` premises test; EPANET checks nothing before the first step has passed. A
    network without rules returns `state` itself.

    `state` is a solution: the premises read its heads (`HEAD`, `PRESSURE`, `LEVEL`), flows,
    the link states as the solve found them (`link_states`: a valve's `ACTIVE`, `OPEN` or
    `CLOSED`; an FCV the flow runs back through reads OPEN, EPANET's `XFCV`), the settings the
    model has (`nan`, never satisfied, for a valve whose status is fixed OPEN or CLOSED, as
    EPANET's MISSING), the junctions' demands (the base demands in `"water.sources"`; the
    system demand sums the junctions' positive totals, where EPANET's `Dsystem` clamps each
    demand category) and a tank's inflow. The actions are tested against those solved states
    (EPANET's `LinkStatus`) and change the carried `"water.link_status"` and
    `"water.link_setting"`, which `TankLevels` writes to the drivers `"water.status"` and
    `"water.setting"` on the next solve.

    ORDER, for a driver of time (EPANET's `ruletimestep` and `runhyd`): the rules are checked
    at the END of a step, at its end time T, and their actions taken; the next solve, at T,
    then applies the simple controls first (`controls`), so a control that fires at T stands
    over a rule's action on the same link. `water_steady` has no time and applies no rule.

    Batched states are tested instance by instance (the rules select: they are not
    differentiable)."""
    out, changed, _ = apply_rules_with_states(model, state, drivers, time=time, step=step)
    return out, changed


def apply_rules_with_states(
    model: Model, state: State, drivers: Drivers, *, time, step: float,
    solved: Tensor | None = None) -> tuple[State, Tensor, Tensor]:
    """`apply_rules`, with the links' solved states `solved` (`solved_link_states`) given:
    within one hydraulic step EPANET checks the rules on the step's one solution, the tanks
    moved on (`hydraul.c`, `ruletimestep`), so `water_run` computes them once per step. Also
    returns `acted` (..., n_links), the links an action was taken on
    (`take_actions_per_link`): `changed` is its any. Internal to `noodl.apps.water` (run
    support)."""
    closure = model.control_closure
    rules = () if closure is None else closure.rules
    if not rules:
        return (state, torch.zeros((), dtype=torch.bool),
                torch.zeros(state["water.q"].shape[:-1] + (len(model.link_names),),
                            dtype=torch.bool))
    premises = [p for r in rules for p in r.premises]
    with torch.no_grad():
        if solved is None:
            solved = solved_link_states(model, state, drivers)
        arrays, names = _rule_readings(closure, premises, state, drivers, solved)
        carried_status = state["water.link_status"].detach()
        carried_setting = state["water.link_setting"].detach()
        time = torch.as_tensor(time, dtype=F64)
        n = len(closure.link_names)
        batch = torch.broadcast_shapes(
            *(a.shape[:-1] for a in arrays.values()), carried_status.shape[:-1],
            carried_setting.shape[:-1], time.shape)

        def rows(t: Tensor) -> Tensor:
            return t.expand(batch + t.shape[-1:]).reshape(-1, t.shape[-1])

        flat = {k: rows(v) for k, v in arrays.items()}
        times = time.expand(batch).reshape(-1)
        status_rows, setting_rows = rows(carried_status), rows(carried_setting)
        solved_rows = rows(solved)
        new_status, new_setting, acted = [], [], []
        for b in range(status_rows.shape[0]):
            snapshot = {a: {i: float(flat[a][b, k]) for k, i in enumerate(ids)}
                        for a, ids in names.items()}
            actions = evaluate_rules(rules, snapshot, time=float(times[b]), step=step,
                                     clocktime=closure.start_clocktime)
            st, se, taken = take_actions_per_link(
                actions, status=status_rows[b], setting=setting_rows[b],
                link_names=closure.link_names, link_kinds=closure.link_kinds,
                setting_scale=closure.setting_scale,
                setting_tolerance=closure.setting_tolerance, solved=solved_rows[b])
            new_status.append(st)
            new_setting.append(se)
            acted.append(taken)
    out = {**state,
           "water.link_status": torch.stack(new_status).reshape(batch + (n,)),
           "water.link_setting": torch.stack(new_setting).reshape(batch + (n,))}
    acted = torch.stack(acted).reshape(batch + (n,))
    return out, acted.any(dim=-1), acted


def solved_link_states(model: Model, state: State, drivers: Drivers) -> Tensor:
    """Every link's state as the solve found it, EPANET's `LinkStatus` (`link_states`), in the
    codes of `noodl.apps.water.controls`, except that an FCV the flow runs back through (the
    head falls along the flow, or the flow is negative: `hydstatus.c`, `fcvstatus`) is OPEN,
    EPANET's `XFCV`, which `checkstatus` reads as open. (EPANET's hysteresis, an FCV that
    stays `XFCV` until the flow regains its setting, needs the last step's state, which is not
    carried.) Internal to `noodl.apps.water` (run support)."""
    closure = model.control_closure
    codes = link_states(model, state, drivers)
    fcv = [model.link_names.index(v.name) for v in model.water_network.valves
           if v.kind.upper() == "FCV"]
    if fcv:
        ends = torch.tensor(_topology(model)[0], dtype=torch.long)
        phi = state["water.phi"].detach()
        dp = phi[..., ends[fcv, 0]] - phi[..., ends[fcv, 1]]
        q = state["water.q"].detach()[..., fcv]
        reversed_ = (dp < -HTOL * closure.head_scale) | (q < -_QTOL)
        codes = codes.clone()
        codes[..., fcv] = torch.where(reversed_ & (codes[..., fcv] == ACTIVE),
                                      torch.full_like(codes[..., fcv], OPEN), codes[..., fcv])
    return codes


#: A flow (m3/s) above which a running pump has not been shut by the solve (`kappa q = 0`).
_PUMP_FLOW = 1e-9

#: EPANET's flow tolerance `Qtol` (0.0001 cfs, `hydraul.c`, `inithyd`) in m3/s.
_QTOL = 1e-4 * M_PER_FT**3


def _rule_readings(closure, premises, state, drivers, solved):
    """`(arrays, names)`: what the rules' premises read of `state`, each attribute as an
    array `(..., len(ids))` over the ids in `names[attribute]` (SI), the snapshot
    `evaluate_rules` takes; only what a premise names is computed."""
    arrays: dict[str, Tensor] = {}
    names: dict[str, list[str]] = {}

    def ids_of(attribute, object_):
        return sorted({p.id for p in premises
                       if p.attribute == attribute and p.object == object_})

    phi = state["water.phi"].detach() / closure.head_scale
    for attribute in ("HEAD", "LEVEL", "PRESSURE"):
        ids = ids_of(attribute, "NODE")
        if ids:
            idx = [closure.node_info[i][0] for i in ids]
            elevation = torch.tensor(
                [0.0 if attribute == "HEAD" else closure.node_info[i][1] for i in ids],
                dtype=F64)
            arrays[attribute], names[attribute] = phi[..., idx] - elevation, ids
    ids = ids_of("DEMAND", "NODE")
    system = any(p.object == "SYSTEM" and p.attribute == "DEMAND" for p in premises)
    if ids or system:
        sources = drivers["water.sources"].detach()
        columns = []
        for i in ids:
            if i in closure.tank_of:
                columns.append(state["water.storage_flow"].detach()[..., closure.tank_of[i]])
            else:
                columns.append(-sources[..., closure.node_info[i][0]])
        if system:
            ids = [*ids, ""]
            junctions = [index for index, _ in closure.junctions.values()]
            columns.append(torch.clamp(-sources[..., junctions], min=0.0).sum(-1))
        arrays["DEMAND"], names["DEMAND"] = torch.stack(
            torch.broadcast_tensors(*columns), -1), ids
    ids = ids_of("FLOW", "LINK")
    if ids:
        idx = [closure.link_names.index(i) for i in ids]
        arrays["FLOW"], names["FLOW"] = state["water.q"].detach()[..., idx].abs(), ids
    ids = ids_of("STATUS", "LINK")
    if ids:
        idx = [closure.link_names.index(i) for i in ids]
        arrays["STATUS"], names["STATUS"] = solved[..., idx].to(F64), ids
    ids = ids_of("SETTING", "LINK")
    if ids:
        idx = [closure.link_names.index(i) for i in ids]
        carried = state["water.link_setting"].detach()
        effective = closure.effective_setting(drivers, carried)
        status = closure._link_status(drivers, state["water.link_status"].detach())
        scale = torch.tensor([closure.setting_scale[j] for j in idx], dtype=F64)
        value = effective[..., idx].to(F64) / scale
        pump = torch.tensor([closure.link_kinds[j] == "PUMP" for j in idx])
        shut = pump & (status[..., idx] <= CLOSED)
        # a valve whose status is fixed (OPEN or CLOSED) has no setting: EPANET's MISSING
        fixed = torch.tensor([closure.link_kinds[j] == "VALVE" for j in idx]) & (
            status[..., idx] != ACTIVE)
        value = torch.where(shut, torch.zeros_like(value), value)
        arrays["SETTING"] = torch.where(fixed, torch.full_like(value, torch.nan), value)
        names["SETTING"] = ids
    return arrays, names


#: The most solves `water_steady` repeats for junction-pressure controls that keep switching.
MAX_PRESSURE_SWITCHES = 10


def water_steady(model: Model, state: State, drivers: Drivers, **solve_kwargs) -> State:
    """The quasi-steady state: one `Model.steady` with tight Newton tolerances.

    A junction-pressure control (`IF NODE <junction> BELOW|ABOVE`) acts after the solve:
    EPANET's `pswitch` tests it on the converged heads, within `Htol` of its setting, and a
    link it switches is solved again, with the step's other controls not tested again
    (`TankLevels.pressure_switch`, `held_controls`), until none switches (at most
    `MAX_PRESSURE_SWITCHES` times, else `RuntimeError` naming the links and junctions).

    `atol`/`rtol` default to 1e-11. MEASURED, sweeping Net1's whole 24 h duty cycle over
    the eight demand multipliers 0.7 to 1.4: 1e-14 does not converge at all (the residual
    floors at 1.3e-14 against a flow scale of 0.1 m3/s), 1e-12 fails at multiplier 1.2
    (floor 5.4e-12), and 5e-12 is the tightest value at which all eight converge. 1e-11 is
    that with a factor of two in hand, and it is still four orders below the comparison tests'
    own float32-limited residuals, so it does not move the two-loop, Net1 single-period,
    Net1 extended-period or pressure-driven-demand comparisons.

    `max_iter` defaults to 200, NOT Newton's own 50. It is the POTENTIAL pipe form that
    needs it (`build_model(pipe_form="potential")`; the default resistance form needs at most
    6 iterations on the same duty cycle). When a `[CONTROLS]` line CLOSES a
    pump, the pipe immediately downstream of it is left carrying exactly zero flow, and
    Hazen-Williams' laminar blend gives that one edge a tangent slope of order
    `(dp_transition / K)^(1/1.852) / dp_transition` -- about 630 on Net1's pipe 10 against
    ~1e-2 elsewhere, a jump of four orders of magnitude in the Jacobian's conditioning at
    the switch. MEASURED on Net1's 24 h duty cycle: the switch step needs 136 Newton
    iterations and every other step needs a handful, so 50 fails outright (residual
    9.6e-4) and 200 clears it with margin. Raising `dp_transition` would reduce the count
    (measured: 50 at 1e-3, 93 at 1e-6, 136 at 1e-9, 179 at 1e-12) but 1e-9 is where the Net1
    extended-period tank trajectory has CONVERGED with respect to it, so the iterations
    are spent rather than the fidelity.

    After the solve, the CONVERGED flow of every power-function pump is checked against its
    own `q_max` at its speed: a demand for more than `q_max` is refused by name. The pump's
    head gain (`PumpCurve.head_gain`) extrapolates the fitted curve indefinitely so that
    Newton's intermediate iterates are never aborted mid-solve; this is where that
    extrapolation is finally held to account, once there is a converged answer to check. No
    clamp, no warning: an exceedance raises `RuntimeError` naming the pump, the batch
    instance and the flow against `q_max`. A segmented curve's last segment and a
    constant-power pump are extended without a limit, as EPANET extends them. A converged
    pump carrying flow backwards is refused by name too (`_refuse_wrong_way`).

    A state whose pipe flows are all exactly zero (a cold start) is solved from
    `model.initial_flows` when the pipes are in their resistance form: from zero flow every
    pipe's slope sits on its `RQtol` floor and Newton's first step overshoots without bound.
    The pumps' flows, likewise, start from their design flows when they are all zero.
    """
    return water_steady_and_cut_off(model, state, drivers, **solve_kwargs)[0]


# ----------------------------------------------------------------------------------------
# Run support. Internal to `noodl.apps.water` (`noodl.apps.water.simulate.water_run` calls
# these; they are not exported from the package): each is a step of EPANET's hydraulic loop
# that `water_steady`, `link_states` and `apply_rules` also take, exposed with the inputs
# the run already has so that it does not compute them twice.


def with_tank_inflow(model: Model, state: State, drivers: Drivers) -> Drivers:
    """`drivers` with the tanks' inflow `"water.tank_inflow"` (`TANK_INFLOW_KEY`), the
    one-second margin of the level controls (EPANET's `controls` reads the tank's
    `NodeDemand`), taken from `state`'s `"water.storage_flow"` when the drivers carry none.
    At a step's start that is exactly the previous solve's inflow, and at `initial_state` it
    is zero, as EPANET's at t = 0. A network without tanks takes no margin."""
    closure = model.control_closure
    if (closure is None or closure.storage is None or TANK_INFLOW_KEY in drivers
            or "water.storage_flow" not in state):
        return drivers
    return {**drivers, TANK_INFLOW_KEY: state["water.storage_flow"].detach()}


def water_steady_and_cut_off(model: Model, state: State, drivers: Drivers,
                             **solve_kwargs) -> tuple[State, tuple]:
    """`(state, (drivers, limit))`: `water_steady`, with the cut-off analysis of its last solve
    (`_cut_off_plan`: the drivers it solved with, the tank inflow margin among them
    (`with_tank_inflow`), and `limit`), which `link_states_from_cut_off` takes instead of
    analysing the solution again."""
    drivers = with_tank_inflow(model, state, drivers)
    plan: list = []
    result = _solve_steady(model, state, drivers, plan=plan, **solve_kwargs)
    closure = model.control_closure
    if closure is None or not closure.has_pressure_controls:
        return result, tuple(plan)
    for _ in range(MAX_PRESSURE_SWITCHES):
        status, setting, changed = closure.pressure_switch(result)
        if not bool(changed.any()):
            return result, tuple(plan)
        with closure.held_controls():
            plan.clear()
            result = _solve_steady(
                model, {**result, "water.link_status": status, "water.link_setting": setting},
                drivers, plan=plan, **solve_kwargs)
    # the last solve may have settled: test it before giving up
    status, setting, changed = closure.pressure_switch(result)
    if not bool(changed.any()):
        return result, tuple(plan)
    moved = ((status != result["water.link_status"])
             | (setting != result["water.link_setting"])).reshape(-1, status.shape[-1]).any(0)
    links = [n for n, m in zip(model.link_names, moved.tolist(), strict=True) if m]
    nodes = sorted({c.node for c in closure.pressure_controls if c.link in links})
    raise RuntimeError(
        f"water_steady: the junction-pressure controls kept switching link(s) {links} "
        f"(tested at junction(s) {nodes}) through {MAX_PRESSURE_SWITCHES} solves; they have "
        f"no consistent state")


def _solve_steady(model: Model, state: State, drivers: Drivers, *, plan: list | None = None,
                  **solve_kwargs) -> State:
    """One `water_steady` solve (see there), the controls as `state` and `drivers` give
    them; `plan`, when given, receives the solve's `(drivers, limit)` (`_cut_off_plan`)."""
    seed = getattr(model, "initial_flows", None)
    layer = model.kirchhoff["water"]
    if seed is not None and layer.implicit_edges.numel():
        q = state["water.q"]
        for kind in ("pipe", "pump", "prv", "psv"):
            if kind in layer.kinds:
                cols = layer.kind_slice(kind)
                if not bool(q[..., cols].any()):
                    q = q.clone()
                    q[..., cols] = seed[cols]
        state = {**state, "water.q": q}
    solve_kwargs.setdefault("atol", 1e-11)
    solve_kwargs.setdefault("rtol", 1e-11)
    solve_kwargs.setdefault("max_iter", 200)
    drivers, limit = _cut_off_plan(model, state, drivers, "water_steady")
    if plan is not None:
        plan.extend((drivers, limit))
    try:
        result = model.steady(state, drivers, **solve_kwargs)
    except UndefinedSteadyState as exc:
        if exc.layer != "quality":
            raise
        raise UndefinedSteadyState(
            f"water_steady: the steady quality of node(s) {exc.nodes} is undefined: they "
            f"are STAGNANT (no inflow, no outflow, no decay; e.g. a dead end whose delivered "
            f"demand is zero because the pressure is below the minimum, or whose demand is "
            f"zero) so the tracer has nowhere to go. Give the water quality a bulk decay "
            f"(quality > 0, which reads such a node as 0), or step the quality in time "
            f"from an initial concentration. Underlying error: {exc}",
            layer=exc.layer, nodes=exc.nodes,
        ) from exc
    _refuse_pump_overflow(model, result, drivers)
    _refuse_wrong_way(model, result, drivers, "water_steady")
    if limit is not None:
        result = {**result, "water.phi": limit(result)[0]}
    return result


def _refuse_pump_overflow(model: Model, state: State, drivers: Drivers) -> None:
    """Refuse, by name, a converged flow of a power-function pump beyond its own `q_max` at
    its speed (`"water.setting"`). A segmented curve's last segment and a constant-power
    pump's law are extended without a limit, as EPANET extends them.

    Reads the layer through its public `law_for` (kind -> `(element, slice)`).
    """
    layer = model.kirchhoff.get("water")
    if layer is None or "pump" not in layer.kinds:
        return
    pump, sl = layer.law_for("pump")
    lo, hi = sl.start, sl.stop
    q = state["water.q"][..., lo:hi].detach()
    speed = drivers.get(SETTING_KEY)
    if speed is not None and pump.link_range is not None:
        speed = speed[..., slice(*pump.link_range)].detach()
    q_max = pump.q_max(speed).detach()
    over = q.abs() > q_max
    if not bool(over.any()):
        return
    names = model.link_names[lo:hi]
    over, flat_q, q_max = torch.broadcast_tensors(over, q.abs(), q_max)
    flat_over = over.reshape(-1, over.shape[-1])
    flat_q = flat_q.reshape(-1, q.shape[-1])
    q_max = q_max.reshape(-1, q.shape[-1])
    bad = [
        f"pump {names[j]!r} (instance {b}): flow {float(flat_q[b, j])} exceeds "
        f"q_max {float(q_max[b, j])}"
        for b in range(flat_over.shape[0])
        for j in range(flat_over.shape[1])
        if flat_over[b, j]
    ]
    raise RuntimeError("water_steady: " + "; ".join(bad))


def _link_status(model: Model, state: State, drivers: Drivers) -> Tensor:
    """The `"water.status"` the solve sees: the drivers' own, or, on a network with tanks,
    the tank closure's (its controls switch the pumps; the closure remembers its last
    answer, so the solve's own call that follows costs nothing), with every pump at speed 0
    (`"water.setting"`) CLOSED."""
    if model.control_closure is not None:
        status = model.control_closure(state, drivers)[STATUS_KEY]
    elif STATUS_KEY not in drivers:
        raise KeyError(
            f"water: driver {STATUS_KEY!r} is required and was not given; build the drivers "
            f"with initial_drivers(model)")
    else:
        status = drivers[STATUS_KEY]
    layer = model.kirchhoff["water"]
    speed = drivers.get(SETTING_KEY)
    if "pump" not in layer.kinds or speed is None:
        return status
    cols = layer.law_for("pump")[1]
    stopped = speed[..., cols].detach() == 0
    if not bool(stopped.any()):
        return status
    # a pump at speed 0 is closed (`hydcoeffs.c`, `pumpcoeff`)
    status = torch.broadcast_to(status, stopped.shape[:-1] + status.shape[-1:]).clone()
    status[..., cols] = torch.where(stopped, float(CLOSED), status[..., cols])
    return status


@dataclass(frozen=True)
class _CutOff:
    """What closed links do to one network for one set of open links and demands
    (`_cut_off_plan`): the junctions cut off with a demand (`fed_none`) and the closed links
    whose row ties a cut-off group to its neighbour during the solve (`ties`)."""

    fed_none: tuple[str, ...]
    ties: tuple[int, ...]


def _cut_off_groups(net: WaterNetwork, open_links: tuple, demand: tuple) -> _CutOff:
    """`_CutOff` for one set of open links (per link) and demands (per junction).

    The groups are the connected parts of the open links. A group with no reservoir or tank
    and no demand has no head of its own in the exact equations (`_limit` gives it EPANET's
    after the solve); so that the solve's Jacobian is not singular, one closed link per
    group, chosen breadth-first from the reservoirs and tanks, ties the group to its
    neighbour during the solve (`ties`, the links `PIN_KEY` marks)."""
    names = net.nodes()
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    links = net.links()
    groups: UnionFind[int] = UnionFind(range(n))
    for link, is_open in zip(links, open_links, strict=True):
        if is_open:
            groups.union(index[link.u], index[link.v])
    group = [groups.find(i) for i in range(n)]
    fed = {group[index[f.name]] for f in net.fixed_head_nodes()}
    cut = [g for g in dict.fromkeys(group) if g not in fed]
    fed_none = tuple(j.name for j, d in zip(net.junctions, demand, strict=True)
                     if d and group[index[j.name]] not in fed)
    if fed_none or not cut:
        return _CutOff(fed_none, ())
    neighbours: dict[int, list] = {}
    for j, (link, is_open) in enumerate(zip(links, open_links, strict=True)):
        gu, gv = group[index[link.u]], group[index[link.v]]
        if not is_open and gu != gv:
            neighbours.setdefault(gu, []).append((j, gv))
            neighbours.setdefault(gv, []).append((j, gu))
    # Every group is reached: the layer refuses, when built, a node no link joins to a
    # reservoir or tank.
    reached, queue, ties = set(fed), deque(fed), []
    while queue:
        for j, other in neighbours.get(queue.popleft(), []):
            if other not in reached:
                reached.add(other)
                ties.append(j)
                queue.append(other)
    return _CutOff((), tuple(ties))


def _groups(model: Model, open_links: tuple, demand: tuple) -> _CutOff:
    """`_cut_off_groups`, remembered per set of open links and demands: an extended-period
    run meets the same few sets again and again."""
    cache = model.__dict__.setdefault("_cut_off", {})
    key = (open_links, demand)
    if key not in cache:
        cache[key] = _cut_off_groups(model.water_network, open_links, demand)
    return cache[key]


def _topology(model: Model) -> tuple[tuple, tuple]:
    """`(ends, fixed)`: the `(start, end)` node indices of every link in `model.link_names`
    order, and the reservoir and tank nodes; built once per model."""
    found = model.__dict__.get("_link_topology")
    if found is None:
        net = model.water_network
        index = {name: i for i, name in enumerate(model.node_names)}
        found = model._link_topology = (
            tuple((index[k.u], index[k.v]) for k in net.links()),
            tuple(index[f.name] for f in net.fixed_head_nodes()))
    return found


def _cut_off_plan(model: Model, state: State, drivers: Drivers, caller: str):
    """`(drivers, limit)` for one solve by `caller`, after refusing what cannot be solved.

    Refused by name: a junction with a demand that closed links cut off from every
    reservoir and tank (EPANET reports it disconnected), one that no water reaches through
    the open links in the directions check valves and full or empty tanks leave them
    (`_one_way_cut_off`), and a status or direction on a
    pipe in its potential form, which reads neither. The demands are the drivers'
    `"water.sources"`, or, under PDA, the build's demands: a pressure-driven demand is not a
    driver, so a PDA demand changed after the build is not seen here.

    A cut-off group without demand is solved, as EPANET solves it: `drivers` gains
    `PIN_KEY` on one closed link per group (`_cut_off_groups`), which ties the group's head
    to that link's far end with every flow exactly zero, and `limit(solution)` then gives
    the heads with every part cut off AT THE SOLUTION at EPANET's limit, and which check
    valves holding no flow are open there (`_limit`). `limit` is `None` when no link is
    closed and none is one-way."""
    net = model.water_network
    layer = model.kirchhoff["water"]
    status = _link_status(model, state, drivers).detach()
    one_way = drivers.get(ONE_WAY_KEY)
    if model.control_closure is not None:
        # a full or empty tank's links (`TankLevels`): the closure's direction, which the
        # solve's own call of it gives again
        given = model.control_closure(state, drivers).get(ONE_WAY_KEY)
        if given is not None:
            one_way = given
            drivers = {**drivers, ONE_WAY_KEY: given}
    if "pipe" in layer.kinds and not isinstance(layer.law_for("pipe")[0], FlowControlledPipe):
        cols = layer.law_for("pipe")[1]
        if bool((status[..., cols] != OPEN).any()) or (
                one_way is not None and bool((one_way[..., cols] != 0).any())):
            raise ValueError(
                f"{caller}: a pipe is closed or one-way in 'water.status'/'water.one_way' (a "
                f"CLOSED or CV pipe, or a pipe to a full or empty tank), which pipes take "
                f"only in their resistance form; build the model with pipe_form='flow'")
    n_j = len(net.junctions)
    n_l = len(model.link_names)
    closed = status == CLOSED
    one_way_any = one_way is not None and bool(one_way.any())
    pressure_valves = "prv" in layer.kinds or "psv" in layer.kinds
    if not n_j or not (bool(closed.any()) or one_way_any or pressure_valves):
        return drivers, None
    if any(isinstance(w, PressureDrivenDemand) for w in layer.withdrawals):
        demand = torch.tensor([j.demand != 0 for j in net.junctions])
    else:
        demand = drivers["water.sources"][..., :n_j].detach() != 0
    if pressure_valves:
        dead = _dead_ends(model, status, demand)
        if dead is not None:
            drivers = {**drivers, DEAD_END_KEY: dead}
        if not (bool(closed.any()) or one_way_any):
            return drivers, None
    if bool(closed.any()):
        open_, wanted = torch.broadcast_tensors(~closed[..., None, :], demand[..., :, None])
        batch = open_.shape[:-2]
        pattern = torch.cat([open_[..., 0, :], wanted[..., 0]], dim=-1).reshape(-1, n_l + n_j)
        rows, inverse = torch.unique(pattern, dim=0, return_inverse=True)
        plans = [_groups(model, tuple(row[:n_l]), tuple(row[n_l:]))
                 for row in rows.tolist()]
        fed_none = sorted({name for p in plans for name in p.fed_none})
        if fed_none:
            raise ValueError(
                f"{caller}: junction(s) {fed_none} are cut off from every reservoir and tank "
                f"by closed links, yet have a demand: no head can deliver it")
        if any(p.ties for p in plans):
            ties = torch.zeros(len(plans), n_l, dtype=F64)
            for k, plan in enumerate(plans):
                ties[k, list(plan.ties)] = 1.0
            drivers = {**drivers, PIN_KEY: ties[inverse].reshape(batch + (n_l,))}
    if one_way_any:
        if any(isinstance(w, PressureDrivenDemand) for w in layer.withdrawals):
            withdrawal = torch.tensor([float(j.demand) for j in net.junctions], dtype=F64)
        else:
            withdrawal = -drivers["water.sources"][..., :n_j].detach()
        unfed = _one_way_cut_off(model, closed, one_way.detach(), withdrawal)
        if unfed:
            raise ValueError(
                f"{caller}: junction(s) {unfed} have a demand that no reservoir, tank or "
                f"inflow reaches through the open links in the direction they let water "
                f"flow: check valves (CV pipes, or the pipes of a full or empty tank) cut "
                f"them off, so no head can deliver it")

    def limit(solution: State) -> tuple[Tensor, Tensor]:
        return _limit(model, solution, closed, demand, one_way, caller)

    return drivers, limit


def _one_way_cut_off(model: Model, closed: Tensor, one_way: Tensor,
                     withdrawal: Tensor) -> list[str]:
    """The junctions with a demand (``withdrawal > 0``, m3/s) that no water can reach, over
    every batch instance (in node order; empty when none): those no path of links that are not
    CLOSED, each taken in a direction its ``one_way`` allows, leads to from a reservoir, a
    tank or a junction with an inflow (``withdrawal < 0``). Every flow that delivers a
    demand runs along such a path, so these demands have no solution; the undirected test
    of `_cut_off_groups` misses them when the only links to a reservoir or tank are check
    valves pointing away (a pipe into an empty tank fills it only). Exact, with pumps and
    valves taken both ways (a superset of their paths), and remembered per set of closed
    links, directions and demand signs."""
    closed, one_way, withdrawal = torch.broadcast_tensors(
        closed[..., None, :], one_way[..., None, :], withdrawal[..., :, None])
    closed, one_way, withdrawal = closed[..., 0, :], one_way[..., 0, :], withdrawal[..., 0]
    n_l = closed.shape[-1]
    pattern = torch.cat([closed.to(F64), one_way.to(F64), torch.sign(withdrawal)],
                        dim=-1).reshape(-1, 2 * n_l + withdrawal.shape[-1])
    cache = model.__dict__.setdefault("_one_way_cut_off", {})
    unfed: set[str] = set()
    for row in torch.unique(pattern, dim=0).tolist():
        key = tuple(row)
        if key not in cache:
            cache[key] = _one_way_unfed(model, row[:n_l], row[n_l:2 * n_l], row[2 * n_l:])
        unfed.update(cache[key])
    return [name for name in model.node_names if name in unfed]


def _one_way_unfed(model: Model, closed: list, one_way: list, sign: list) -> tuple:
    """`_one_way_cut_off` for one set of closed links, directions and demand signs.

    `sign` is per junction, and junction `i` is node `i`: `build_model` puts the junctions
    first in `model.node_names`, in `net.junctions` order (the `"water.sources"` columns
    `[:n_j]` are theirs). Checked here, once per pattern."""
    junctions = [j.name for j in model.water_network.junctions]
    assert list(model.node_names[:len(sign)]) == junctions, "junctions first in node_names"
    ends, fixed = _topology(model)
    ahead: dict[int, list[int]] = {}
    for (u, v), shut, way in zip(ends, closed, one_way, strict=True):
        if shut:
            continue
        if way >= 0:
            ahead.setdefault(u, []).append(v)
        if way <= 0:
            ahead.setdefault(v, []).append(u)
    queue = deque(list(fixed) + [i for i, s in enumerate(sign) if s < 0])
    reached = set(queue)
    while queue:
        for other in ahead.get(queue.popleft(), ()):
            if other not in reached:
                reached.add(other)
                queue.append(other)
    return tuple(model.node_names[i] for i, s in enumerate(sign)
                 if s > 0 and i not in reached)


def _dead_ends(model: Model, status: Tensor, demand: Tensor) -> Tensor | None:
    """`DEAD_END_KEY` for one solve: per link, +1 where a PRV or PSV that is ACTIVE has its
    end node in a dead end, -1 its start node, 0 elsewhere; `None` when no valve has one.
    A valve with both sides dead (in a part without demand that CLOSED links cut off from
    every reservoir and tank) is refused by name.

    A dead end of a valve is the group of junctions on one side of it that only the valve
    joins to a reservoir or tank (through links that are not CLOSED) and that has no
    demand. No water can flow through the valve then, and the exact condition leaves the
    group's head anywhere that keeps the valve shut; the valve's law then ties the group
    (`_PressureValve`, A DEAD END), as EPANET's answer does. Remembered per set of statuses
    and demands."""
    layer = model.kirchhoff["water"]
    status, demand = torch.broadcast_tensors(
        status.detach()[..., None, :], demand[..., :, None])
    status, demand = status[..., 0, :], demand[..., 0]
    batch = status.shape[:-1]
    n_l = status.shape[-1]
    pattern = torch.cat([status, demand.to(status.dtype)], dim=-1).reshape(-1, n_l
                                                                          + demand.shape[-1])
    rows, inverse = torch.unique(pattern, dim=0, return_inverse=True)
    cache = model.__dict__.setdefault("_dead_ends", {})
    cols = [layer.law_for(kind)[1] for kind in ("prv", "psv") if kind in layer.kinds]
    out = []
    for row in rows.tolist():
        key = tuple(row)
        if key not in cache:
            cache[key] = _dead_ends_of(model, row[:n_l], row[n_l:], cols)
        out.append(cache[key])
    if not any(any(r) for r in out):
        return None
    table = torch.tensor(out, dtype=F64)
    return table[inverse].reshape(batch + (n_l,))


def _dead_ends_of(model: Model, status: list, demand: list, cols: list) -> list[float]:
    """`_dead_ends` for one set of link statuses and junction demands (lists)."""
    ends, fixed = _topology(model)
    n_j = len(demand)
    out = [0.0] * len(status)
    for sl in cols:
        for j in range(sl.start, sl.stop):
            if status[j] != ACTIVE:
                continue
            group, fed = _joined_groups(ends, fixed, tuple(
                k != j and status[k] != CLOSED for k in range(len(status))))
            for side, node in ((1.0, ends[j][1]), (-1.0, ends[j][0])):
                g = group[node]
                if g in fed:
                    continue
                if not any(demand[i] for i in range(n_j) if group.get(i) == g):
                    if out[j]:
                        raise ValueError(
                            f"water: valve {model.link_names[j]!r} joins two parts of the "
                            f"network without demand that CLOSED links cut off from every "
                            f"reservoir and tank, so its state and those heads are "
                            f"undetermined; open one of those links, or close the valve")
                    out[j] = side
    return out


#: A flow below this (m3/s) is zero for a check valve: it holds no flow.
_NO_FLOW = 1e-12


def _limit(model: Model, solution: State, closed: Tensor, demand: Tensor,
           one_way: Tensor | None, caller: str) -> tuple[Tensor, Tensor]:
    """`(heads, open)`: `solution`'s heads with every zero-demand part of the network that
    the solve leaves without a head of its own set to EPANET's limit head, and, per link,
    whether a check valve holding no flow is open there.

    A link carrying flow, or open in both directions, ties its ends' heads (it is "solid");
    a node joined by solid links to a reservoir or tank is "reliable" and keeps the solve's
    own head. Every other node is "free": it sits behind closed links (status CLOSED) and
    check valves holding no flow, where the exact equations leave its head anywhere that
    keeps those valves shut. EPANET gives each closed link the conductance `1 / CBIG`, and
    opens a check valve whose upstream head is the higher; the free heads of this model are
    the limit of EPANET's as `CBIG` grows: the solution of the convex quadratic program

        minimise   1/2 sum over the closed links and the check valves holding no flow,
                   (a, b), of (H_a - H_b)^2
        subject to H_u <= H_v for every such check valve u -> v

    (reverse-only pipes with the ends swapped), the heads outside the structure held at the
    solve's. Each link of the structure is a resistor of one conductance (EPANET's shut
    check valve has `1 / CBIG` too), and each check valve is in addition an ideal diode in
    parallel (open, EPANET's is `1 / RQtol`, 1e15 times larger): the KKT conditions are
    the node balances of that resistor-and-diode network, with the constraints' multipliers
    the diodes' flows (in units of `1 / CBIG`), so a check valve is open where its multiplier
    is positive and shut where it is zero. The objective is strictly convex on the free
    heads (every free node reaches a reliable one through the structure), so the heads are
    unique, and `_dual_active_set` finds them in finitely many steps. The heads are a
    linear map of the solve's heads for the active set found, so a gradient flows through
    them. Solid links among free nodes carry no flow and tie their ends.

    The multipliers, and so the states, are unique only where the active constraints'
    normals are independent. Parallel check valves (the same ends, the same way) share one
    constraint row and one multiplier, so they are open or shut together, as EPANET has
    them. Other dependence (a triangle Ra -> J1, J1 -> J2, Ra -> J2, J1 and J2 both tied to
    Ra) leaves a split of the flow among the valves undetermined; the method never adds a
    constraint the active ones already satisfy, so such a valve has multiplier 0 and is
    CLOSED, and the water goes through the valves it added first, the most violated as it
    went (MEASURED: that triangle's states are EPANET's).

    A junction with a demand among the free nodes is refused by name: no head delivers it.
    At a converged solution that cannot happen where EPANET solves (a demand junction is
    reliable: its water reaches it through links carrying flow); it fires where a
    converged check valve holds no flow against a demand behind it, which the solve's
    branch selection refuses first (`_refuse_wrong_way`)."""
    phi = solution["water.phi"]
    net = model.water_network
    n, n_j, n_l = phi.shape[-1], len(net.junctions), len(model.link_names)
    q = solution["water.q"].detach()
    if one_way is None:
        one_way = torch.zeros(n_l, dtype=q.dtype)
    still = (one_way != 0) & (q.abs() <= _NO_FLOW) & ~closed
    solid = ~closed & ~still
    batch = torch.broadcast_shapes(phi.shape[:-1], solid.shape[:-1], demand.shape[:-1],
                                   one_way.shape[:-1])
    flat_in = phi.expand(batch + (n,)).reshape(-1, n)
    out = flat_in.clone()
    open_ = torch.zeros(flat_in.shape[0], n_l, dtype=torch.bool)
    solid = solid.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    still = still.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    closed = closed.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    demand = demand.expand(batch + (n_j,)).reshape(-1, n_j).tolist()
    direction = one_way.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    ends, fixed = _topology(model)
    tol = 1e-9 * model.head_scale
    for b in range(out.shape[0]):
        group, fed = _joined_groups(ends, fixed, tuple(solid[b]))
        free = [i for i in range(n) if group.get(i, i) not in fed]
        if not free:
            continue
        for i in free:
            if i < n_j and demand[b][i]:
                raise ValueError(
                    f"{caller}: junction {net.junctions[i].name!r} has a demand but is "
                    f"joined to every reservoir and tank only through closed links and "
                    f"check valves holding no flow")
        structure = [j for j in range(n_l) if closed[b][j] or still[b][j]]
        rows, far, coefficients, opened = _limit_heads(
            ends, free, structure, [j for j in structure if still[b][j]], direction[b],
            solid[b], flat_in[b].detach(), tol, model, caller)
        out[b, rows] = coefficients.to(phi.dtype) @ flat_in[b][far]
        open_[b, opened] = True
    return out.reshape(batch + (n,)), open_.reshape(batch + (n_l,))


def _limit_heads(ends, free, structure, valves, direction, solid, values, tol, model,
                 caller):
    """`(rows, far, coefficients, opened)` of `_limit` for one instance: the free nodes
    `rows`, whose limit heads are `coefficients @ values[far]` (`far` the reliable nodes
    the structure touches), and the check valves `opened` (the links among `valves` whose
    multiplier is positive).

    The unknowns are the tie groups of the free nodes (joined by solid links, which carry no
    flow there). The program is assembled on them (`G`, `a` from the `structure` links;
    one constraint per check valve whose two ends are not tied or both reliable) and solved
    by `_dual_active_set`; its active set, as ties, then gives the heads exactly: a group
    tied to a reliable node takes that node's head, the others solve the Laplacian over the
    structure links with unit weights."""
    is_free = [False] * len(values)
    for i in free:
        is_free[i] = True
    ties: UnionFind[int] = UnionFind(free)
    for j, (u, v) in enumerate(ends):
        if solid[j] and is_free[u] and is_free[v]:
            ties.union(u, v)
    variables = list(dict.fromkeys(ties.find(i) for i in free))
    k_of = {g: k for k, g in enumerate(variables)}
    of_node = {i: k_of[ties.find(i)] for i in free}
    far: list[int] = []
    far_of: dict[int, int] = {}

    def far_index(node: int) -> int:
        if node not in far_of:
            far_of[node] = len(far)
            far.append(node)
        return far_of[node]

    # the links as (variable or None, variable or None, far index or None): the free ends
    # as variables, a reliable end as a far node
    links = []
    for j in structure:
        u, v = ends[j]
        ku = of_node.get(u) if is_free[u] else None
        kv = of_node.get(v) if is_free[v] else None
        if ku is None and kv is None:
            links.append(None)
        elif ku is not None and kv is not None:
            links.append((ku, kv, None) if ku != kv else None)
        elif ku is not None:
            links.append((ku, None, far_index(v)))
        else:
            links.append((None, kv, far_index(u)))
    n_var = len(variables)
    # one constraint per distinct row (variable +1 or None, variable -1 or None, far +1,
    # far -1), with the check valves that share it: parallel valves give one row, else the
    # multipliers would not be unique and a duplicate would never enter the active set
    constraints: dict[tuple, list[int]] = {}
    for j in valves:
        u, v = ends[j]
        lo, hi = (u, v) if direction[j] > 0 else (v, u)
        k_lo = of_node.get(lo) if is_free[lo] else None
        k_hi = of_node.get(hi) if is_free[hi] else None
        if (k_lo is None and k_hi is None) or (k_lo is not None and k_lo == k_hi):
            continue
        # H_hi - H_lo >= 0, as c x >= b with the fixed heads on the right
        constraints.setdefault((k_hi, k_lo, far_index(lo) if k_lo is None else None,
                                far_index(hi) if k_hi is None else None), []).append(j)
    n_far = len(far)
    g = torch.zeros(n_var, n_var, dtype=F64)
    b_far = torch.zeros(n_var, n_far, dtype=F64)   # the objective's `a` is `-b_far h_far`
    for link in links:
        if link is None:
            continue
        ku, kv, f = link
        if f is None:
            g[ku, ku] += 1.0
            g[kv, kv] += 1.0
            g[ku, kv] -= 1.0
            g[kv, ku] -= 1.0
        else:
            k = ku if ku is not None else kv
            g[k, k] += 1.0
            b_far[k, f] += 1.0
    h_far = values[far]
    active: list[int] = []
    multipliers: list[float] = []
    rows = list(constraints)
    if rows:
        c = torch.zeros(len(rows), n_var, dtype=F64)
        d = torch.zeros(len(rows), n_far, dtype=F64)
        for row, (k_hi, k_lo, f_lo, f_hi) in enumerate(rows):
            if k_hi is not None:
                c[row, k_hi] += 1.0
            if k_lo is not None:
                c[row, k_lo] -= 1.0
            if f_lo is not None:
                d[row, f_lo] += 1.0
            if f_hi is not None:
                d[row, f_hi] -= 1.0
        try:
            _, active, multipliers = _dual_active_set(g, -b_far @ h_far, c, d @ h_far, tol)
        except _Infeasible as exc:
            names = [model.link_names[j] for row in exc.rows for j in constraints[rows[row]]]
            raise ValueError(
                f"{caller}: the check valves {names} hold no flow, yet no heads keep them "
                f"all shut: the solve's state is not a solution") from exc
    opened = [j for row, u in zip(active, multipliers, strict=True) if u > tol
              for j in constraints[rows[row]]]
    # The heads from the active set, as ties: exact where tied to a reliable node.
    merged: UnionFind[int] = UnionFind(range(n_var))
    anchor: dict[int, int] = {}
    for row in active:
        k_hi, k_lo, f_lo, f_hi = rows[row]
        if k_hi is not None and k_lo is not None:
            merged.union(k_hi, k_lo)
        elif k_hi is not None:
            anchor[k_hi] = f_lo
        else:
            anchor[k_lo] = f_hi
    root = [merged.find(k) for k in range(n_var)]
    anchored = {}
    for k, f in anchor.items():
        anchored.setdefault(root[k], f)
    unknown = [r for r in dict.fromkeys(root) if r not in anchored]
    u_of = {r: m for m, r in enumerate(unknown)}
    lap = torch.zeros(len(unknown), len(unknown), dtype=F64)
    rhs = torch.zeros(len(unknown), n_far, dtype=F64)
    for link in links:
        if link is None:
            continue
        ku, kv, f = link
        ru = None if ku is None else root[ku]
        rv = None if kv is None else root[kv]
        if ru is not None and ru == rv:
            continue
        for a, other, f_other in ((ru, rv, f), (rv, ru, f)):
            if a not in u_of:
                continue
            lap[u_of[a], u_of[a]] += 1.0
            if other in u_of:
                lap[u_of[a], u_of[other]] -= 1.0
            else:
                rhs[u_of[a], f_other if other is None else anchored[other]] += 1.0
    per_unknown = torch.linalg.solve(lap, rhs) if unknown else rhs
    coefficients = torch.zeros(n_var, n_far, dtype=F64)
    for k in range(n_var):
        if root[k] in anchored:
            coefficients[k, anchored[root[k]]] = 1.0
        else:
            coefficients[k] = per_unknown[u_of[root[k]]]
    return free, far, coefficients[[of_node[i] for i in free]], opened


class _Infeasible(ValueError):
    """`_dual_active_set`: no point satisfies the constraints `rows`."""

    def __init__(self, rows: list[int]) -> None:
        super().__init__(f"infeasible constraints {rows}")
        self.rows = rows


def _dual_active_set(g: Tensor, a: Tensor, c: Tensor, b: Tensor, tol: float,
                     ) -> tuple[Tensor, list[int], list[float]]:
    """`(x, active, multipliers)` of the strictly convex quadratic program
    ``minimise 1/2 x' g x + a' x subject to c x >= b`` (`g` symmetric positive definite,
    `c` one constraint per row): the dual active-set method of Goldfarb and Idnani
    (Mathematical Programming 27, 1983), in dense arithmetic, for a small program.

    It starts at the unconstrained minimum with no constraint active, and holds throughout:
    `x` minimises the objective with the active constraints as equalities, their
    multipliers are non-negative and their normals independent. Each outer step picks a
    violated constraint `p` and moves `x` towards it along the direction `z` that keeps the
    active equalities (the dual step `r` moves their multipliers); a multiplier that would
    turn negative drops its constraint first (a partial step), and `p` joins the active set
    when it is reached (a full step).

    It ends. With `p` violated and `z` nonzero, a full step has a positive length, so the
    objective, which the method never lets fall, strictly rises over every full step; after
    a full step `x` is the unique minimiser over the active set, whose objective depends on
    that set alone, so no active set returns, and there are finitely many. Between two
    full steps only drops occur, at most one per active constraint. When `z` is zero (the
    normal of `p` lies in the span of the active ones) and no multiplier can fall, no point
    satisfies the constraints (`_Infeasible`, naming `p` and the active set)."""
    chol = torch.linalg.cholesky(g)
    g_inv = torch.cholesky_inverse(chol)
    x = -(g_inv @ a)
    active: list[int] = []
    u: list[float] = []
    for _ in range(100 * (c.shape[0] + 1) * (g.shape[0] + 1)):
        s = c @ x - b
        p = int(torch.argmin(s))
        if float(s[p]) >= -tol:
            return x, active, u
        n_p = c[p]
        u_p = 0.0
        while True:
            if active:
                big_n = c[active].T                           # (n, k)
                m = big_n.T @ g_inv @ big_n                    # (k, k), positive definite
                n_star = torch.linalg.solve(m, big_n.T @ g_inv)
                h = g_inv - g_inv @ big_n @ n_star
                r = (n_star @ n_p).tolist()
            else:
                h = g_inv
                r = []
            z = h @ n_p
            z_n = float(z @ n_p)
            t1, drop = math.inf, None
            for j, (u_j, r_j) in enumerate(zip(u, r, strict=True)):
                if r_j > 1e-12 and u_j / r_j < t1:
                    t1, drop = u_j / r_j, j
            s_p = float(n_p @ x - b[p])
            t2 = math.inf if z_n <= 1e-12 else -s_p / z_n
            t = min(t1, t2)
            if t == math.inf:
                raise _Infeasible([p, *active])
            u = [u_j - t * r_j for u_j, r_j in zip(u, r, strict=True)]
            u_p += t
            if t2 != math.inf:
                x = x + t * z
            if t == t2:
                active.append(p)
                u.append(u_p)
                break
            del active[drop], u[drop]
    raise RuntimeError("_dual_active_set: the dual active-set method did not end")


def _refuse_wrong_way(model: Model, state: State, drivers: Drivers, caller: str) -> None:
    """Refuse a converged one-way pipe carrying flow against its direction, or a pump
    carrying flow backwards. A check valve or pump an iterate would close is kept open
    where closing it would cut a junction off (`FlowControlledPipe`,
    `FlowControlledPump`); where the solution itself needs it closed with the junction
    then cut off, it ends with that wrong-way flow, and the network cannot be solved."""
    layer = model.kirchhoff["water"]
    if "pump" in layer.kinds:
        cols = layer.law_for("pump")[1]
        q = state["water.q"][..., cols].detach()
        wrong = (q < -1e-9).reshape(-1, q.shape[-1]).any(0)
        if bool(wrong.any()):
            names = [model.link_names[cols.start + j] for j in range(len(wrong)) if wrong[j]]
            raise ValueError(
                f"{caller}: pump(s) {names} carry flow backwards: with them shut a junction "
                f"is cut off from every reservoir and tank")
    one_way = drivers.get(ONE_WAY_KEY)
    if one_way is None or "pipe" not in layer.kinds or not bool(one_way.any()):
        return
    cols = layer.law_for("pipe")[1]
    q = state["water.q"][..., cols].detach()
    wrong = (q * one_way[..., cols] < -1e-9).reshape(-1, q.shape[-1]).any(0)
    if bool(wrong.any()):
        names = [model.link_names[cols.start + j] for j in range(len(wrong)) if wrong[j]]
        raise ValueError(
            f"{caller}: check valve(s) {names} carry flow against their direction (a CV "
            f"pipe, or a pipe to a full or empty tank): with them closed a junction is cut "
            f"off from every reservoir and tank")


def link_states(model: Model, state: State, drivers: Drivers) -> Tensor:
    """Every link's state in `state` (a solution), in `model.link_names` order, as the codes
    of `"water.status"` (`noodl.apps.water.controls`): `CLOSED` (0), `OPEN` (1) or `ACTIVE`
    (2). These are also the codes `wntr` reports EPANET's link states in
    (`results.link["status"]`).

    A link whose status is CLOSED is closed. A one-way pipe (a check valve) carrying flow is
    open; one holding no flow is closed, unless EPANET's limit would push water through it
    (`_limit`: the multiplier of its constraint is positive, as for a valve that feeds a
    zero-demand junction from the highest of several reservoirs). A pump is CLOSED where
    its status is, where its speed is 0, or where it is shut beyond its shut-off head
    (`FlowControlledPump.states`), and OPEN elsewhere. An ACTIVE PRV or PSV is in
    the state its residual selects at the solution (`_PressureValve.states`: CLOSED, ACTIVE
    or OPEN). A valve otherwise has its status (ACTIVE or OPEN); every other link is OPEN.

    The status is the control closure's for `state` and `drivers`, so the tank-level controls
    are tested again with their one-second margin: the drivers' `"water.tank_inflow"`, else
    `state`'s `"water.storage_flow"` (`with_tank_inflow`). For a solved state that is the
    solve's own inflow, not the previous one the solve's controls used; pass the solve's
    drivers with `"water.tank_inflow"` set to the state it started from to read its own
    statuses (`water_run` does: `water_steady_and_cut_off`, `link_states_from_cut_off`).
    """
    drivers = with_tank_inflow(model, state, drivers)
    return link_states_from_cut_off(
        model, state, *_cut_off_plan(model, state, drivers, "link_states"))


def link_states_from_cut_off(model: Model, state: State, drivers: Drivers, limit, *,
                             valves: bool = True) -> Tensor:
    """`link_states` with the cut-off analysis `(drivers, limit)` of the solve given
    (`water_steady_and_cut_off`). With `valves=False` an ACTIVE PRV or PSV keeps its code
    ACTIVE instead of the state its residual selects (the pumps' and pipes' codes are the
    same); the run's energy reads only the pumps. Internal to `noodl.apps.water` (run
    support)."""
    status = _link_status(model, state, drivers)
    q = state["water.q"]
    out = torch.broadcast_to(status, q.shape[:-1] + status.shape[-1:]).clone()
    one_way = drivers.get(ONE_WAY_KEY)
    if limit is not None and one_way is not None and bool(one_way.any()):
        _, open_ = limit(state)
        shut = (one_way != 0) & (q.detach().abs() <= _NO_FLOW) & ~open_
        out = torch.where(shut, torch.full_like(out, CLOSED), out)
    layer = model.kirchhoff["water"]
    if "pump" in layer.kinds:
        law, cols = layer.law_for("pump")
        running = out[..., cols] != CLOSED
        # a pump the solve shut carries no flow (its row is `kappa q = 0`): only a running
        # pump with next to none can be one, and only that one asks the law
        undecided = running & (q[..., cols].detach().abs() <= _PUMP_FLOW)
        if bool(undecided.any()):
            ends = torch.tensor(_topology(model)[0], dtype=torch.long)
            phi = state["water.phi"].detach()
            if ("pipe" in layer.kinds and law.shut_links is not None
                    and isinstance(layer.law_for("pipe")[0], FlowControlledPipe)):
                # the check valves this state shuts, as the solve's own evaluation records
                # them for the pumps' cut-off test (`ShutLinks`)
                pipes, pcols = layer.law_for("pipe")
                with torch.no_grad():
                    pipes(phi[..., ends[pcols, 0]] - phi[..., ends[pcols, 1]],
                          q[..., pcols].detach(), drivers, None)
            dp = phi[..., ends[cols, 0]] - phi[..., ends[cols, 1]]
            held = drivers
            closure = model.control_closure
            if closure is not None:
                # the drivers the solve took: the closure's status and speed (a pump a
                # control has just opened or closed, or a speed pattern has moved), memoised
                held = {**drivers, **closure(state, drivers)}
            selected = law.states(dp, q[..., cols].detach(), held)
            out[..., cols] = torch.where(undecided, selected.to(out.dtype), out[..., cols])
    ends = None
    for kind in ("prv", "psv") if valves else ():
        if kind not in layer.kinds:
            continue
        if ends is None:
            ends = torch.tensor(_topology(model)[0], dtype=torch.long)
        law, cols = layer.law_for(kind)
        phi = state["water.phi"].detach()
        phi_from, phi_to = phi[..., ends[cols, 0]], phi[..., ends[cols, 1]]
        ctx = LawContext("dc", phi_from=phi_from, phi_to=phi_to)
        selected = law.states(phi_from - phi_to, q[..., cols].detach(), drivers, ctx)
        active = out[..., cols] == ACTIVE
        out[..., cols] = torch.where(active, selected.to(out.dtype), out[..., cols])
    return out.detach().to(torch.long)


def water_step(model: Model, state: State, drivers: Drivers, dt: float, *,
               scheme: str = "backward_euler", **solve_kwargs) -> State:
    """One step of `dt` (s) of a model built with `tank_scheme="backward_euler"` or
    `"trapezoidal"`: `Model.step` with `water_steady`'s Newton defaults and pump check, the
    tank levels stepped implicitly (`LinearStorage`, `scheme` `"backward_euler"` or
    `"trapezoidal"`; trapezoidal needs a step-start state whose tank flows are solved, not
    the build's zeros). The level controls act on the step-start levels, as EPANET's do.
    `"backward_euler"` is the default because it starts from any state, the build's
    included, and does not ring after the flow jumps that pump and level controls cause.

    The step is the caller's `dt`: it is never shortened, to a control or to anything else
    (`water_run`, which takes EPANET's step rule, refuses these schemes), so an `AT TIME` or
    `AT CLOCKTIME` control fires only if a step starts exactly on its second, and a level
    control at the first step start past its trigger."""
    if not model.water_network.tanks or model.tank_scheme == "forward_euler":
        raise ValueError(
            "water_step: the model's tanks are not stepped inside the solve "
            "(build_model(tank_scheme='backward_euler') or 'trapezoidal'); step a "
            "tank_scheme='forward_euler' model with water_run"
        )
    solve_kwargs.setdefault("atol", 1e-11)
    solve_kwargs.setdefault("rtol", 1e-11)
    solve_kwargs.setdefault("max_iter", 200)
    drivers = with_tank_inflow(model, state, drivers)
    drivers, limit = _cut_off_plan(model, state, drivers, "water_step")
    result = model.step(state, drivers, dt, scheme=scheme, **solve_kwargs)
    _refuse_pump_overflow(model, result, drivers)
    _refuse_wrong_way(model, result, drivers, "water_step")
    if limit is not None:
        result = {**result, "water.phi": limit(result)[0]}
    return result


def settle_tanks(model: Model, storage, phi: Tensor, inflow: Tensor, drivers) -> Tensor:
    """`phi` with every tank that is within one second of filling or emptying put on its
    limit, EPANET 2.2's `tanklevels` (`hydraul.c`), transcribed as it is: `V + q >= Vmax`
    puts it at its maximum, else `V - q <= Vmin` at its minimum (`q` the inflow, m3/s, over
    one second). Every other potential is `phi`'s.

    The head put there is EPANET's `tankgrade(Vmax)` (`tankgrade(Vmin)`): the maximum
    (minimum) level, except on a volume curve whose volume is flat across that level, where
    it is the lower end of the flat stretch (`project.c`, `interp`, takes the first point at
    or above the volume). Internal to `noodl.apps.water` (run support)."""
    closure = model.control_closure
    heads = phi[..., storage.nodes]
    with torch.no_grad():
        top = (closure.bottom + closure.y_max) * closure.head_scale
        low = (closure.bottom + closure.y_min) * closure.head_scale
        volume = storage.amount(heads.detach(), drivers)
        q = inflow.detach()
        v_max, v_min = storage.amount(top, drivers), storage.amount(low, drivers)
        full = volume + q >= v_max
        empty = ~full & (volume - q <= v_min)
        if not bool((full | empty).any()):
            return phi
        if isinstance(storage, TankVolumeCurve):
            top, low = storage.potential_of(v_max), storage.potential_of(v_min)
    settled = torch.where(full, top, torch.where(empty, low, heads))
    batch = torch.broadcast_shapes(phi.shape[:-1], settled.shape[:-1])
    out = phi.expand(batch + phi.shape[-1:]).clone()
    out[..., storage.nodes] = settled.expand(batch + settled.shape[-1:])
    return out


def tank_levels(model: Model, state: State) -> Tensor:
    """Every tank's level (m above its bottom) in `state`: its head less its bottom."""
    return model.control_closure.levels(state)


def tank_inflow(model: Model, state: State) -> Tensor:
    """Net inflow (m3/s) at every tank node: the flow into every tank's storage,
    `"water.storage_flow"`, which a solve writes (`KirchhoffLayer.storage_flows`). Under
    `tank_scheme="forward_euler"` it is the inflow the solve delivered to the held tank,
    the rate over the step `water_run` takes."""
    return state["water.storage_flow"]
