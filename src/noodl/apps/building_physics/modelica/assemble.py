"""Turn a `ComponentGraph` into a noodl `Model`, its initial `State` and its `Drivers`.

Spec sections 5-7. Every number comes from the evaluated JSON (`schema.ModelicaDoc`); this
module never evaluates a Modelica expression. A parameter the export omits takes the MBL
declaration's own default (cited where used); a parameter MBL declares without a default is
required and its absence is refused. All refusals are gathered and raised as ONE
`ModelicaImportError` naming each instance and its class, as `graph.build` does.

What each MBL construct becomes
-------------------------------
* **Nodes.** One network node per zone (`MixingVolume`, `DelayFirstOrder`) and per boundary
  (`Boundary_pT`, `Outside`), in the document's component order, zones first. Junction nodes
  of fused column chains do not appear: each chain is one edge (`graph.FlowPath`).
* **Air layer** ("air", `PotentialFlowLayer`, quantity pressure, Pa). Potentials are GAUGE
  pressures relative to `medium.p_default`. One edge kind per MBL instance, so that each
  element keeps its own `dp_turbulent`, form and drives: `"airpath:<name>"` for a one-way
  element (edge from its `port_a` side to its `port_b` side), `"door_ab:<name>"` and
  `"door_ba:<name>"` for `DoorOpen`/`DoorOperable` (`noodl.elements.mbl.door`),
  `"door_c:<name>"` for the `nCom` compartment edges of a discretised door
  (`noodl.elements.mbl.door_discretized`, with its `DoorCompartmentHead` drive), and
  `"zonal_ab:<name>"`/`"zonal_ba:<name>"` for a zonal flow. A fused column chain adds a
  `_ColumnHead` drive to its path. The air-layer boundary is every boundary node plus, for
  each group of zones joined by pressure-dependent edges with no boundary among them, the
  group's first zone at its own `p_start` (see "Closed zone groups" below).
* **Thermal layer** ("thermal", `TransportLayer`, K) over every edge kind, boundaries =
  boundary nodes plus the zones pinned by a `FixedTemperature` -> `ThermalConductor`
  (`G >= 1e6` W/K) construction. Omitted when no zone is left unpinned.
* **Species layer** ("species", `TransportLayer`, kg/kg) with one species per
  `extraPropertiesNames` entry and, when the model needs it (see "Moisture"), water vapour
  `X_w` as the LAST species. Boundaries = boundary nodes. Omitted when it has no species.
* **Closure** (`_MBLClosure`) writing the full-node drivers `"T"` (K), `"p_abs"` (Pa),
  `"rho"` (buoyancy density, `medium.buoyancy_density`) and, when moisture is carried,
  `"X_w"`; otherwise `"X_w"` is a constant driver at the medium default (0 for SimpleAir).

Capacities (controller ruling; MBL v13 sources)
------------------------------------------------
`ConservationEquation.mo:245-266`: a volume's fluid mass is `m = V rho_start` under
`massDynamics = SteadyState` and `m = V medium.d` (the ACTUAL density) otherwise, with
`U = m u + CSen (T - reference_T)` and `CSen = (mSenFac - 1) rho_default cp_default V`
(`:124-125`); `mC = m C` (`:266`); `der(U) = Hb_flow + Q_flow` (`:303`). For
`Buildings.Media.Air`, `u = h - pStp/dStp` (`Air.mo:782-788`), so `du/dT = cp`, and the
density is pressure-only (`Air.mo:210-215`), so `m` differs from `V rho_start` only by the
relative pressure change (~1e-5). noodl's airflow is quasi-steady (no mass storage), so the
mass is held at its start value: heat capacity `V rho_start cp + CSen`, species capacity
`V rho_start`, with `rho_start = Medium.density(p_start, T_start, X_start)`
(`PartialMixingVolume.mo:92-93`) and `cp = Medium.specificHeatCapacityCp` at `X_default`
(`ConservationEquation.mo:131-132` uses it for `CSen`). For the two ideal-gas media MBL's
`U = m u` with a temperature-dependent `m` has no quasi-steady counterpart; the constant-
pressure capacity `m cp` above is what a fixed-mass open zone at constant pressure has. The
heat carrier on the edges is the same `cp`, so the balance is conservative.

Moisture (controller ruling)
----------------------------
For a medium with moisture, water vapour is carried as a species only when the model needs
it: a volume's `X_start[1]` or a boundary's `X[1]` differs from `X_default[1]`, or a
`MassFlowSource_T` injects air of another composition. Otherwise `"X_w"` is the constant
`X_default[1]` everywhere, which is exactly MBL's value throughout such a model.

`p_abs` (controller ruling)
---------------------------
`"p_abs" = p_default + phi`: at air-layer boundary nodes from `"air.phi_boundary"` (exact),
at interior nodes from the state's last solved `"air.phi"` (seeded with `p_start - p_default`
in the initial state). With `coupling="iterate"` (used whenever a transport layer exists)
each pass reads the previous pass's `"air.phi"`, so within a step the interior pressure
converges with the temperatures; the iteration tests the thermal state (1e-4 K, below the
reference simulation's declared relative tolerance times 300 K), not `phi` itself.

Closed zone groups
------------------
MBL's `ZonalFlow` example has two rooms joined only by prescribed flows and no boundary: the
airflow alone leaves their pressure undetermined, and the potential layer would be singular.
MBL fixes it through mass storage (`der(m) = sum(ports.m_flow)`): with balanced flows the
mass, hence for `Buildings.Media.Air` the pressure, stays at `p_start`. The reader therefore
makes the first zone of every group of zones joined by pressure-dependent edges (paths and
doors, not zonal flows) and holding no boundary node a pressure boundary of the air layer at
its own `p_start`. This is exact for balanced prescribed flows; mass a source injects into
such a zone is absorbed by the reference (reported, not approximated silently: the
docstring of `build` says so and `names.air_references` lists the zones).

Sources
-------
`TraceSubstancesFlowSource` (`TraceSubstancesFlowSource.mo`, equation section): it injects
`m_flow_in` kg/s of fluid (`sum(ports.m_flow) = -m_flow_in`) at `h_default`, `X_default` and
`C = 1` for the named substance (0 for the others). This becomes an air-layer mass source,
a thermal source `cp m T_default`, a species source `m` for the substance and, when moisture
is carried, `m X_default[1]` of water. `MassFlowSource_T` injects `m_flow` at `T`, `X`, `C`
likewise; a negative (extracting) flow is refused, since the extracted enthalpy would depend
on the zone state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from noodl.apps.building_physics.modelica import schema, signals
from noodl.apps.building_physics.modelica.graph import ComponentGraph, FlowPath, TwoWayEdge
from noodl.apps.building_physics.modelica.schema import (
    Component,
    ModelicaDoc,
    ModelicaImportError,
)
from noodl.elements.base import Element
from noodl.elements.mbl import (
    MBLDoorOpen,
    MBLDoorOperable,
    MBLMedium,
    MBLTable,
    mbl_coefficient,
    mbl_discretized_door,
    mbl_discretized_operable_door,
    mbl_ela,
    mbl_orifice,
    mbl_point,
    mbl_points,
    medium,
)
from noodl.elements.mbl.media import _moist_air_buoyancy_density
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import TransportLayer, active_interior
from noodl.model import Drivers, Model, State
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64

G_N = 9.80665  # MSL Modelica/Constants.mo:38
G_PIN_MIN = 1e6  # W/K, spec section 6
THERMAL_ITERATE_TOL = 1e-4  # K
SPECIES_ITERATE_TOL = 1e-10  # kg/kg
ITERATE_MAX = 50

_FREE_ENERGY = ("FixedInitial", "DynamicFreeInitial")


@dataclass(frozen=True)
class ModelicaNames:
    """How MBL instance names map onto the noodl model.

    `edges[name]` lists `(column, sign)` pairs into the air layer's flow vector `"air.q"`
    whose signed sum is the flow from the element's `port_a` (one-way) or side-A (two-way)
    side to its other side: `port_a.m_flow` of a one-way element, `mAB_flow - mBA_flow` of a
    door or zonal flow, the sum over the compartments of a discretised door. For
    `DoorOpen`/`DoorOperable` and the zonal flows the two port flows are listed too:
    `edges["<name>.port_a1"]` is `port_a1.m_flow` and `edges["<name>.port_a2"]` is
    `port_a2.m_flow`. `nodes[name]` is the node position of a zone or boundary. `kinds[name]`
    are the instance's air-layer edge kinds. `times` is the experiment output grid.
    `air_references` lists the zones made air-layer pressure references (module docstring,
    "Closed zone groups").
    """

    edges: dict[str, tuple[tuple[int, int], ...]]
    nodes: dict[str, int]
    kinds: dict[str, tuple[str, ...]]
    times: Tensor
    air_references: tuple[str, ...]


# --------------------------------------------------------------------------- helpers
def _short(value, default: str) -> str:
    return str(value if value is not None else default).rsplit(".", 1)[-1]


def _t(value) -> Tensor:
    return torch.as_tensor(value, dtype=F64)


def experiment_times(doc: ModelicaDoc) -> Tensor:
    """The output grid `StartTime + k Interval`, `k = 0..N` (Modelica's default `Interval`
    is `(StopTime - StartTime)/500`)."""
    start = float(doc.experiment.get("StartTime", 0.0))
    stop = float(doc.experiment.get("StopTime", 1.0))
    interval = float(doc.experiment.get("Interval", (stop - start) / 500.0))
    if not interval > 0.0 or stop < start:
        raise ModelicaImportError(
            f"modelica: experiment needs StopTime >= StartTime and Interval > 0, got "
            f"{doc.experiment}"
        )
    n = int(round((stop - start) / interval))
    return start + interval * torch.arange(n + 1, dtype=F64)


def _stack(values: list[Tensor], n_t: int) -> tuple[Tensor, bool]:
    """Stack scalars (`()`) and series (`(n_t,)`) along a new LAST axis: a series
    `(n_t, len(values))` if any entry is one (flag True), else `(len(values),)`."""
    if not values:
        return torch.zeros(0, dtype=F64), False
    if any(v.ndim for v in values):
        return torch.stack([v.expand(n_t) for v in values], dim=-1), True
    return torch.stack(values), False


def _stack_nested(rows: list[list[Tensor]], K: int, n_t: int) -> tuple[Tensor, bool]:
    """`rows[i][k]` (scalar or series) -> `(n, K)`, or `(n_t, n, K)` if any is a series."""
    if not rows:
        return torch.zeros(0, K, dtype=F64), False
    if any(v.ndim for r in rows for v in r):
        return torch.stack(
            [torch.stack([v.expand(n_t) for v in r], dim=-1) for r in rows], dim=-2
        ), True
    return torch.stack([torch.stack(r) for r in rows]), False


class _Signals:
    """The document's signals keyed by the input they drive, evaluated on the time grid."""

    def __init__(self, doc: ModelicaDoc, times: Tensor, kind_of: Mapping[str, str],
                 errors: list[str]) -> None:
        self.times = times
        self.errors = errors
        self.by_target: dict[str, schema.Signal] = {}
        self.used: set[str] = set()
        self.kind_of = kind_of
        for s in doc.signals:
            if s.drives in self.by_target:
                errors.append(
                    f"{s.name} ({s.cls}): drives {s.drives}, which "
                    f"{self.by_target[s.drives].name} already drives"
                )
                continue
            self.by_target[s.drives] = s

    def get(self, target: str) -> Tensor | None:
        s = self.by_target.get(target)
        if s is None:
            return None
        self.used.add(target)
        try:
            y = signals.evaluate(s, self.times)
        except ModelicaImportError as exc:
            self.errors.append(str(exc).removeprefix("modelica: "))
            return None
        if bool((y == y[0]).all()):
            return y[0].clone()  # a constant signal stays a constant driver
        return y

    def require(self, comp: Component, port: str) -> Tensor | None:
        y = self.get(f"{comp.name}.{port}")
        if y is None and f"{comp.name}.{port}" not in self.by_target:
            self.errors.append(
                f"{comp.name} ({comp.cls}): input {port} is enabled but no supported signal "
                f"drives it"
            )
        return y

    def check_unused(self) -> None:
        for target, s in self.by_target.items():
            if target in self.used:
                continue
            inst = target.split(".", 1)[0]
            if self.kind_of.get(inst) == "observer":
                continue  # feeds a block the reader ignores (spec section 4)
            self.errors.append(
                f"{s.name} ({s.cls}): drives {target}, which this reader does not read "
                f"(the input is disabled or not supported)"
            )


# ------------------------------------------------------------------------- drives
class _ColumnHead:
    """Hydrostatic head of a fused column chain (spec section 6, graph.py docstring):
    `sum(sign h rho g_n)` with `rho = density_pTX(p_default, T, X_w)` of the node each
    column's `densitySelection` names (`MediumColumn.mo`, equation section: `fromTop` reads
    `inStream(port_a...)`, the state beyond the column's top port, `fromBottom` the one beyond
    its bottom port; every medium through `Psychrometrics.Functions.density_pTX` with
    `X_w = 0` when the medium has no moisture). Shape `(..., 1)`."""

    def __init__(self, kind: str, coeff: Tensor, nodes: Tensor) -> None:
        self.kind = kind
        self.coeff = coeff.detach()
        self.nodes = nodes

    def __call__(self, drivers: Mapping[str, Tensor]) -> Tensor:
        T = drivers["T"][..., self.nodes]
        X = drivers["X_w"][..., self.nodes]
        rho = _moist_air_buoyancy_density(T, X)
        return (self.coeff * rho).sum(-1, keepdim=True)


# ------------------------------------------------------------------------ elements
class _ZonalFlowEdge(Element):
    """One direction of a `ZonalFlow_ACS`/`ZonalFlow_m_flow` (spec section 5): a prescribed
    flow, independent of `dp` (`dp_independent`), oriented side A -> side B.

    `ZonalFlow_m_flow.mo:10-11`: `port_a1.m_flow = mAB_flow`, `port_a2.m_flow = mBA_flow`.
    `ZonalFlow_ACS.mo:38-42`: `V_flow = V ACS`, `m_flow = V_flow rho` with `rho = rho_default`
    if `useDefaultProperties` else `(Medium.density(sta_a1_inflow) +
    Medium.density(sta_a2_inflow))/2`, both ports carrying `m_flow`. `sta_a1_inflow` is at
    `port_a1.p` with `port_b1.h_outflow = inStream(port_a1.h_outflow)` (`ZonalFlow.mo`), i.e.
    side A's pressure and state; `sta_a2_inflow` side B's. The `"ab"` edge carries
    `port_a1.m_flow`, the `"ba"` edge `-port_a2.m_flow` (flow from B to A is negative here).

    This is `noodl.elements.fixed.FixedFlow`'s law with the flow read from drivers each call
    rather than held as a fixed parameter: the input signal, and for the ACS form the port
    densities, may change during a simulation.
    """

    dp_independent = True

    def __init__(self, *, kind: str, direction: str, src: int, tgt: int, medium: MBLMedium,
                 key: str, V: float | None = None, use_default: bool = True) -> None:
        super().__init__(kind)
        self.direction = direction
        self.src, self.tgt = int(src), int(tgt)
        self.medium = medium
        self.key = key
        self.V = None if V is None else float(V)
        self.use_default = bool(use_default)

    def _value(self, drivers) -> Tensor:
        u = torch.as_tensor(drivers[self.key])
        if self.V is None:  # ZonalFlow_m_flow
            m = u
        else:
            if self.use_default:
                rho = self.medium.rho_default
            else:
                p, T, X = drivers["p_abs"], drivers["T"], drivers["X_w"]
                i, j = self.src, self.tgt
                rho = 0.5 * (self.medium.density(p[..., i], T[..., i], X[..., i])
                             + self.medium.density(p[..., j], T[..., j], X[..., j]))
            m = self.V * u * rho
        m = m.unsqueeze(-1) if m.ndim else m.reshape(1)
        return m if self.direction == "ab" else -m

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self._value(drivers) + torch.zeros_like(dp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.zeros_like(self.flow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        v = self._value(drivers)
        return v, torch.zeros_like(v)


# ------------------------------------------------------------------------- closure
class _MBLClosure:
    """Writes the full-node drivers `"T"`, `"p_abs"`, `"rho"` (and `"X_w"` when moisture is
    carried) from the transport states, the boundary drivers and the last solved `"air.phi"`
    (module docstring, "p_abs")."""

    def __init__(self, *, medium: MBLMedium, T0: Tensor, phi0: Tensor, X0: Tensor,
                 air_interior: Tensor, air_bound: Tensor, th_interior: Tensor | None,
                 th_bound: Tensor, sp_interior: Tensor | None, sp_bound: Tensor | None,
                 water: int | None, n_species: int) -> None:
        self.medium = medium
        self.T0, self.phi0, self.X0 = T0, phi0, X0
        self.air_interior, self.air_bound = air_interior, air_bound
        self.th_interior, self.th_bound = th_interior, th_bound
        self.sp_interior, self.sp_bound = sp_interior, sp_bound
        self.water = water
        self.n_species = n_species

    @staticmethod
    def _scatter(base: Tensor, parts: list[tuple[Tensor, Tensor]]) -> Tensor:
        shapes = [v.shape[:-1] for v, _ in parts]
        batch = torch.broadcast_shapes(*shapes) if shapes else torch.Size()
        out = base.to(F64).expand(batch + base.shape).clone()
        for value, idx in parts:
            if idx.numel():
                out[..., idx] = value.expand(batch + (idx.numel(),))
        return out

    def temperatures(self, state, drivers) -> Tensor:
        parts = [(drivers["thermal.x_boundary"], self.th_bound)]
        if self.th_interior is not None:
            parts.append((state["thermal.x"], self.th_interior))
        return self._scatter(self.T0, parts)

    def species(self, state, drivers) -> Tensor | None:
        """Full-node species mass fractions `(..., n, K)` (water last when carried)."""
        if self.sp_interior is None:
            return None
        x_i, x_b = state["species.x"], drivers["species.x_boundary"]
        batch = torch.broadcast_shapes(x_i.shape[:-2], x_b.shape[:-2])
        n = self.T0.numel()
        out = torch.zeros(batch + (n, self.n_species), dtype=F64)
        if self.water is not None:
            out[..., :, self.water] = self.X0
        out[..., self.sp_interior, :] = x_i.expand(batch + (self.sp_interior.numel(),
                                                            self.n_species))
        out[..., self.sp_bound, :] = x_b.expand(batch + (self.sp_bound.numel(),
                                                         self.n_species))
        return out

    def pressures(self, state, drivers) -> Tensor:
        parts = [(drivers["air.phi_boundary"], self.air_bound)]
        phi = state.get("air.phi")
        if phi is not None and self.air_interior.numel():
            parts.append((phi[..., self.air_interior], self.air_interior))
        return self.medium.p_default + self._scatter(self.phi0, parts)

    def __call__(self, state, drivers):
        T = self.temperatures(state, drivers)
        out = {"T": T, "p_abs": self.pressures(state, drivers)}
        if self.water is not None:
            X = self.species(state, drivers)[..., self.water]
            out["X_w"] = X
        else:
            X = drivers["X_w"]
        out["rho"] = self.medium.buoyancy_density(T, X)
        return out


# --------------------------------------------------------------------------- build
@dataclass
class _Zone:
    comp: Component
    V: float
    T_start: float
    p_start: float
    X_w: float
    C: list[float]
    rho_start: float
    heat_capacity: float
    pinned: float | None = None


def _param(comp: Component, key: str, errors: list[str], default=None, *, required=False):
    if key in comp.parameters:
        return comp.parameters[key]
    if required:
        errors.append(f"{comp.name} ({comp.cls}): parameter {key!r} is required and was not "
                      f"exported")
        return None
    return default


def _one_way(comp: Component, kind: str, med: MBLMedium, errors: list[str]) -> Element | None:
    cls = comp.cls.rsplit(".", 1)[-1]
    p = comp.parameters
    if p.get("useDefaultProperties", True) is False:
        errors.append(f"{comp.name} ({comp.cls}): useDefaultProperties = false is not "
                      f"supported (the MBL laws are transcribed for the default density)")
        return None
    dpt = float(p.get("dp_turbulent", 0.1))  # PartialOneWayFlowElement.mo:14
    rho = med.rho_default

    def req(key):
        return _param(comp, key, errors, required=True)

    try:
        if cls == "Orifice":
            A = req("A")
            if A is None:
                return None
            return mbl_orifice(A, p.get("CD", 0.65), p.get("m", 0.5), dpt, rho_default=rho,
                               kind=kind)
        if cls == "EffectiveAirLeakageArea":
            L = req("L")
            if L is None:
                return None
            return mbl_ela(L, p.get("dpRat", 4.0), p.get("CDRat", 1.0), p.get("m", 0.65), dpt,
                           rho_default=rho, kind=kind)
        if cls == "Point_m_flow":
            dp, mf = req("dpMea_nominal"), req("mMea_flow_nominal")
            if dp is None or mf is None:
                return None
            return mbl_point(dp, mf, p.get("m", 0.5), dpt, rho_default=rho, kind=kind)
        if cls == "Points_m_flow":
            dp, mf = req("dpMea_nominal"), req("mMea_flow_nominal")
            if dp is None or mf is None:
                return None
            return mbl_points(dp, mf, dpt, rho_default=rho, kind=kind)
        if cls == "Coefficient_V_flow":
            C, m = req("C"), req("m")
            if C is None or m is None:
                return None
            return mbl_coefficient(C, m, "volume", dpt, rho_default=rho, kind=kind)
        if cls == "Coefficient_m_flow":
            k = req("k")
            if k is None:
                return None
            return mbl_coefficient(k, p.get("m", 0.5), "mass", dpt, rho_default=rho,
                                   kind=kind)
        if cls == "Table_V_flow":
            dp, vf = req("dpMea_nominal"), req("VMea_flow_nominal")
            if dp is None or vf is None:
                return None
            return MBLTable(dp, vf, form="volume", rho_default=rho, kind=kind)
        if cls == "Table_m_flow":
            dp, mf = req("dpMea_nominal"), req("mMea_flow_nominal")
            if dp is None or mf is None:
                return None
            return MBLTable(dp, mf, form="mass", kind=kind)
    except (ValueError, TypeError) as exc:
        errors.append(f"{comp.name} ({comp.cls}): {exc}")
        return None
    errors.append(f"{comp.name} ({comp.cls}): one-way flow element not converted")
    return None


def _zone(comp: Component, med: MBLMedium, errors: list[str]) -> _Zone | None:
    p = comp.parameters
    cls = comp.cls.rsplit(".", 1)[-1]
    if cls == "DelayFirstOrder":
        # DelayFirstOrder.mo:4-6,8,11-12: V = m_flow_nominal*tau/rho_default, mSenFac = 1.
        mfn = _param(comp, "m_flow_nominal", errors, required=True)
        if mfn is None:
            return None
        V = float(mfn) * float(p.get("tau", 60.0)) / med.rho_default
        mSenFac = 1.0
    else:
        Vp = _param(comp, "V", errors, required=True)
        if Vp is None:
            return None
        V = float(Vp)
        mSenFac = float(p.get("mSenFac", 1.0))  # PartialMixingVolume / LumpedVolumeDecl.
    if not V > 0.0:
        errors.append(f"{comp.name} ({comp.cls}): volume must be positive, got {V!r}")
        return None
    # LumpedVolumeDeclarations.mo:29-39: start values default to the medium defaults.
    T_start = float(p.get("T_start", med.T_default))
    p_start = float(p.get("p_start", med.p_default))
    X_start = p.get("X_start", list(med.X_default))
    X_w = float(X_start[0]) if med.has_moisture and X_start else 0.0
    C = [float(c) for c in p.get("C_start", [])]
    rho_start = float(med.density(_t(p_start), T_start, X_w))
    cp = med.specific_heat_cp(med.X_default[0] if med.has_moisture else 0.0)
    # ConservationEquation.mo:124-125: CSen = (mSenFac - 1)*rho_default*cp_default*V.
    heat = rho_start * V * cp + (mSenFac - 1.0) * med.rho_default * cp * V
    return _Zone(comp=comp, V=V, T_start=T_start, p_start=p_start, X_w=X_w, C=C,
                 rho_start=rho_start, heat_capacity=heat)


class _Builder:
    def __init__(self, graph: ComponentGraph, doc: ModelicaDoc) -> None:
        self.graph, self.doc = graph, doc
        self.errors: list[str] = []
        try:
            self.med = medium(str(doc.medium.get("class")))
        except KeyError as exc:
            raise ModelicaImportError(f"modelica: {exc.args[0]}") from None
        self.species_names = [str(s) for s in doc.medium.get("extraPropertiesNames", [])]
        self.times = experiment_times(doc)
        self.n_t = int(self.times.numel())
        kind_of = {
            c.name: "observer" if (c.role == "observer" or c.cls in schema.OBSERVERS)
            else "component"
            for c in doc.components
        }
        self.sig = _Signals(doc, self.times, kind_of, self.errors)
        self.by_name = {c.name: c for c in doc.components}
        # Element inputs fed by signals (door `y`, zonal `ACS`/`mAB_flow`/`mBA_flow`), keyed
        # by their driver name "<instance>.<input>"; scalars or series over the grid.
        self._inputs: dict[str, Tensor] = {}

    # ------------------------------------------------------------- boundaries
    def boundary(self, comp: Component) -> dict | None:
        med, p = self.med, comp.parameters
        cls = comp.cls.rsplit(".", 1)[-1]
        out: dict = {}
        if p.get("use_X_in") or p.get("use_Xi_in"):
            self.errors.append(f"{comp.name} ({comp.cls}): composition inputs (use_X_in, "
                               f"use_Xi_in) are not supported")
        if cls == "Outside":
            # Outside.mo: pressure and temperature come from the weather bus only.
            pv = self.sig.get(f"{comp.name}.weaBus.pAtm")
            Tv = self.sig.get(f"{comp.name}.weaBus.TDryBul")
            if pv is None or Tv is None:
                self.errors.append(
                    f"{comp.name} ({comp.cls}): its pressure and temperature come from the "
                    f"weather bus (weaBus.pAtm, weaBus.TDryBul), which no supported signal "
                    f"drives (weather data is not supported)"
                )
                return None
        else:
            # Boundary_pT.mo:6-16: p = p_default, T = T_default unless the inputs are used.
            pv = (self.sig.require(comp, "p_in") if p.get("use_p_in")
                  else _t(float(p.get("p", med.p_default))))
            Tv = (self.sig.require(comp, "T_in") if p.get("use_T_in")
                  else _t(float(p.get("T", med.T_default))))
        X = p.get("X", list(med.X_default))  # PartialSource_Xi_C.mo:15-18
        out["p"], out["T"] = pv, Tv
        out["X_w"] = _t(float(X[0]) if med.has_moisture and X else 0.0)
        C = []
        for k in range(len(self.species_names)):
            if p.get("use_C_in"):
                C.append(self.sig.require(comp, f"C_in[{k + 1}]"))
            else:  # PartialSource_Xi_C.mo:19-22: C = fill(0, nC)
                values = p.get("C", [0.0] * len(self.species_names))
                C.append(_t(float(values[k]) if k < len(values) else 0.0))
        out["C"] = C
        if any(v is None for v in (pv, Tv, *C)):
            return None
        return out

    # ------------------------------------------------------------- build
    def build(self) -> tuple[Model, State, Drivers, ModelicaNames]:
        g, med, errors = self.graph, self.med, self.errors
        zones: dict[str, _Zone] = {}
        for name in g.zones:
            z = _zone(self.by_name[name], med, errors)
            if z is not None:
                zones[name] = z
        bounds: dict[str, dict] = {}
        for name in g.boundaries:
            b = self.boundary(self.by_name[name])
            if b is not None:
                bounds[name] = b

        # Pinned temperatures (spec section 6).
        for pin in g.pins:
            G = _param(pin.conductor, "G", errors, required=True)
            T = _param(pin.source, "T", errors, required=True)
            if G is None or T is None:
                continue
            if float(G) < G_PIN_MIN:
                errors.append(
                    f"{pin.conductor.name} ({pin.conductor.cls}): G = {float(G)!r} W/K is "
                    f"below {G_PIN_MIN:g} W/K; only a fixed temperature through a stiff "
                    f"conductor (G >= {G_PIN_MIN:g}) is supported"
                )
                continue
            z = zones.get(pin.zone)
            if z is not None:
                if z.pinned is not None:
                    errors.append(f"{pin.zone}: pinned by more than one fixed temperature")
                z.pinned = float(T)
        for name, z in zones.items():
            dyn = _short(z.comp.parameters.get("energyDynamics"), "DynamicFreeInitial")
            if dyn not in _FREE_ENERGY and z.pinned is None:
                errors.append(
                    f"{name} ({z.comp.cls}): energyDynamics = {dyn} is not supported (the "
                    f"zone temperature starts at T_start: FixedInitial or DynamicFreeInitial)"
                )

        node_names = [n for n in g.zones] + [n for n in g.boundaries]
        index = {n: i for i, n in enumerate(node_names)}
        net = Network(dtype=F64)
        for n in node_names:
            if n in zones:
                z = zones[n]
                net.add_node(n, volume=z.V, T0=z.T_start, z_ref=0.0)
            else:
                net.add_node(n, volume=0.0, T0=med.T_default, z_ref=0.0)

        elements: list[Element] = []
        drives: list = []
        kinds: dict[str, tuple[str, ...]] = {}
        edge_dirs: dict[str, list[tuple[str, int, int]]] = {}  # name -> (kind, j, sign)
        extra_ports: dict[str, tuple[str, int, int]] = {}
        pressure_edges: list[tuple[int, int]] = []

        for path in g.paths:
            self._path(path, net, index, elements, drives, kinds, edge_dirs, pressure_edges)
        for door in g.doors:
            self._door(door, net, index, elements, drives, kinds, edge_dirs, extra_ports,
                       pressure_edges)
        for zf in g.zonal:
            self._zonal(zf, net, index, elements, kinds, edge_dirs, extra_ports)

        # Moisture (module docstring).
        carry_water = med.has_moisture and (
            any(abs(z.X_w - med.X_default[0]) > 0.0 for z in zones.values())
            or any(abs(float(b["X_w"]) - med.X_default[0]) > 0.0 for b in bounds.values())
            or any(
                c.cls.endswith("MassFlowSource_T")
                and float(c.parameters.get("X", list(med.X_default))[0]) != med.X_default[0]
                for c, _ in g.sources
            )
        )
        K = len(self.species_names) + (1 if carry_water else 0)
        water = len(self.species_names) if carry_water else None

        sources = self._sources(zones, index, water, K)
        self.sig.check_unused()
        if errors:
            unique = sorted(set(errors))
            count = "1 item" if len(unique) == 1 else f"{len(unique)} items"
            raise ModelicaImportError(
                f"modelica: refused {count}:\n" + "\n".join(f"  - {e}" for e in unique)
            )
        if not elements:
            raise ModelicaImportError("modelica: the model has no flow element")

        n = net.n
        # Closed zone groups -> air-layer pressure references (module docstring).
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in pressure_edges:
            parent[find(a)] = find(b)
        has_boundary = {find(index[b]) for b in g.boundaries}
        references: list[str] = []
        seen_roots: set[int] = set()
        for name in g.zones:
            r = find(index[name])
            if r in has_boundary or r in seen_roots:
                continue
            seen_roots.add(r)
            references.append(name)
        air_boundary = list(g.boundaries) + references

        air = PotentialFlowLayer(net, "air", elements, drives=drives, boundary=air_boundary,
                                 quantity="pressure", unit="Pa")
        layers: dict = {"air": air}
        flow_kinds = tuple(air.kinds)

        pinned = [nm for nm, z in zones.items() if z.pinned is not None]
        th_boundary = list(g.boundaries) + pinned
        th_interior, _ = active_interior(net, flow_kinds, th_boundary)
        cp = med.specific_heat_cp(med.X_default[0] if med.has_moisture else 0.0)
        if th_interior.numel():
            cap = torch.tensor([zones[node_names[i]].heat_capacity for i in th_interior.tolist()],
                               dtype=F64)
            layers["thermal"] = TransportLayer(
                net, "thermal", capacity=cap, flow_kind=flow_kinds, boundary=th_boundary,
                carrier=float(cp), scheme="exact", quantity="temperature", unit="K",
            )
        sp_boundary = list(g.boundaries)
        sp_interior = None
        if K:
            sp_int, _ = active_interior(net, flow_kinds, sp_boundary)
            if sp_int.numel():
                sp_interior = sp_int
                cap = torch.tensor([zones[node_names[i]].rho_start * zones[node_names[i]].V
                                    for i in sp_int.tolist()], dtype=F64)
                layers["species"] = TransportLayer(
                    net, "species", capacity=cap, flow_kind=flow_kinds, boundary=sp_boundary,
                    n_species=K, scheme="implicit", quantity="mass_fraction", unit="kg/kg",
                )

        # ---------------------------------------------------------- drivers
        const: dict[str, Tensor] = {}
        series: dict[str, Tensor] = {}

        def put(key: str, packed: tuple[Tensor, bool]) -> None:
            value, is_series = packed
            if is_series:
                series[key] = value
                const[key] = value[0]
            else:
                const[key] = value

        phi_b = [bounds[b]["p"] - med.p_default for b in g.boundaries]
        phi_b += [_t(zones[r].p_start - med.p_default) for r in references]
        put("air.phi_boundary", _stack(phi_b, self.n_t))
        Tb = [bounds[b]["T"] for b in g.boundaries] + [_t(zones[z].pinned) for z in pinned]
        put("thermal.x_boundary", _stack(Tb, self.n_t))
        if "species" in layers:
            rows = [[bounds[b]["X_w"] if k == water else bounds[b]["C"][k] for k in range(K)]
                    for b in g.boundaries]
            put("species.x_boundary", _stack_nested(rows, K, self.n_t))
        for key, value in self._inputs.items():
            put(key, (value, bool(value.ndim)))
        X_const = med.X_default[0] if med.has_moisture else 0.0
        if water is None:
            const["X_w"] = torch.full((n,), X_const, dtype=F64)

        s_air, s_th, s_sp = sources
        air_idx = set(air.interior.tolist())
        vals = [s_air[i] if i in air_idx else _t(0.0) for i in range(n)]
        if any(bool((v != 0).any()) for v in vals):
            put("air.sources", _stack(vals, self.n_t))
        if "thermal" in layers:
            th_idx = set(layers["thermal"].interior_idx.tolist())
            vals = [s_th[i] if i in th_idx else _t(0.0) for i in range(n)]
            if any(bool((v != 0).any()) for v in vals):
                put("thermal.sources", _stack(vals, self.n_t))
        if "species" in layers:
            sp_idx = set(layers["species"].interior_idx.tolist())
            rows = [[s_sp[i][k] if i in sp_idx else _t(0.0) for k in range(K)]
                    for i in range(n)]
            if any(bool((v != 0).any()) for r in rows for v in r):
                put("species.sources", _stack_nested(rows, K, self.n_t))

        drivers: Drivers = dict(const)
        for key, value in series.items():
            drivers[f"series:{key}"] = value
        drivers["series:time"] = self.times.clone()

        # ---------------------------------------------------------- closure + model
        T0 = torch.tensor([zones[nm].T_start if nm in zones else med.T_default
                           for nm in node_names], dtype=F64)
        phi0 = torch.tensor([zones[nm].p_start - med.p_default if nm in zones else 0.0
                             for nm in node_names], dtype=F64)
        X0 = torch.tensor([zones[nm].X_w if nm in zones else X_const for nm in node_names],
                          dtype=F64)
        closure = _MBLClosure(
            medium=med, T0=T0, phi0=phi0, X0=X0, air_interior=air.interior,
            air_bound=air.bound,
            th_interior=layers["thermal"].interior_idx if "thermal" in layers else None,
            th_bound=net.boundary_index(th_boundary),
            sp_interior=sp_interior,
            sp_bound=net.boundary_index(sp_boundary) if sp_interior is not None else None,
            water=water, n_species=K,
        )
        if "thermal" in layers:
            model = Model(net, layers, closures=[closure], coupling="iterate",
                          iterate_tol={"thermal": THERMAL_ITERATE_TOL}, iterate_max=ITERATE_MAX)
        elif "species" in layers:
            model = Model(net, layers, closures=[closure], coupling="iterate",
                          iterate_tol={"species": SPECIES_ITERATE_TOL}, iterate_max=ITERATE_MAX)
        else:
            model = Model(net, layers, closures=[closure])

        # ---------------------------------------------------------- initial state
        phi_init = phi0.clone()
        phi_init[air.bound] = const["air.phi_boundary"]
        state: State = {"air.phi": phi_init}
        if "thermal" in layers:
            state["thermal.x"] = T0[layers["thermal"].interior_idx].clone()
        if "species" in layers:
            rows = []
            for i in layers["species"].interior_idx.tolist():
                z = zones[node_names[i]]
                row = [z.C[k] if k < len(z.C) else 0.0 for k in range(len(self.species_names))]
                if water is not None:
                    row.append(z.X_w)
                rows.append(row)
            state["species.x"] = torch.tensor(rows, dtype=F64).reshape(-1, K)

        # ---------------------------------------------------------- names
        edges: dict[str, tuple[tuple[int, int], ...]] = {}
        for name, entries in edge_dirs.items():
            edges[name] = tuple((air.kind_slice(k).start + j, s) for k, j, s in entries)
        for key, (k, j, s) in extra_ports.items():
            edges[key] = ((air.kind_slice(k).start + j, s),)
        names = ModelicaNames(edges=edges, nodes=dict(index), kinds=kinds,
                              times=self.times.clone(), air_references=tuple(references))
        return model, state, drivers, names

    # ------------------------------------------------------------- edges
    def _path(self, path: FlowPath, net, index, elements, drives, kinds, edge_dirs,
              pressure_edges) -> None:
        comp = path.element
        kind = f"airpath:{comp.name}"
        el = _one_way(comp, kind, self.med, self.errors)
        coeff, nodes = [], []
        for col, sign in path.columns:
            h = float(col.parameters.get("h", 3.0))  # MediumColumn.mo:10
            sel = _short(col.parameters.get("densitySelection"), "")
            # sign = +1: the column's port_a (top) faces the path's src side.
            top, bottom = (path.src, path.tgt) if sign > 0 else (path.tgt, path.src)
            if sel == "fromTop":
                node = top
            elif sel == "fromBottom":
                node = bottom
            else:
                self.errors.append(
                    f"{col.name} ({col.cls}): densitySelection = {sel or 'unset'} is not "
                    f"supported (fromTop and fromBottom are)"
                )
                continue
            coeff.append(sign * h * G_N)
            nodes.append(index[node])
        if el is None:
            return
        net.add_edge(path.src, path.tgt, kind=kind)
        elements.append(el)
        if coeff:
            drives.append(_ColumnHead(kind, _t(coeff), torch.tensor(nodes, dtype=torch.long)))
        kinds[comp.name] = (kind,)
        edge_dirs[comp.name] = [(kind, 0, 1)]
        pressure_edges.append((index[path.src], index[path.tgt]))

    def _door(self, door: TwoWayEdge, net, index, elements, drives, kinds, edge_dirs,
              extra_ports, pressure_edges) -> None:
        comp, med = door.component, self.med
        cls = comp.cls.rsplit(".", 1)[-1]
        p = comp.parameters
        iA, iB = index[door.side_a], index[door.side_b]
        common = dict(wOpe=p.get("wOpe", 0.9), hOpe=p.get("hOpe", 2.1),
                      dp_turbulent=float(p.get("dp_turbulent", 0.01)))
        if cls in ("DoorOpen", "DoorOperable"):
            kab, kba = f"door_ab:{comp.name}", f"door_ba:{comp.name}"
            if cls == "DoorOpen":
                law = dict(common, CD=p.get("CD", 0.65), m=p.get("m", 0.5))
                ctor = MBLDoorOpen
            else:
                LClo = _param(comp, "LClo", self.errors, required=True)
                y = self.sig.require(comp, "y")
                if LClo is None or y is None:
                    return
                self._inputs[f"{comp.name}.y"] = y
                law = dict(common, LClo=LClo, CDOpe=p.get("CDOpe", 0.65),
                           mOpe=p.get("mOpe", 0.5), mClo=p.get("mClo", 0.65),
                           dpCloRat=p.get("dpCloRat", 4.0), CDCloRat=p.get("CDCloRat", 1.0),
                           y_key=f"{comp.name}.y")
                ctor = MBLDoorOperable
            for kind, direction in ((kab, "ab"), (kba, "ba")):
                net.add_edge(door.side_a, door.side_b, kind=kind)
                elements.append(ctor(direction=direction, src=[iA], tgt=[iB], medium=med,
                                     kind=kind, **law))
            kinds[comp.name] = (kab, kba)
            edge_dirs[comp.name] = [(kab, 0, 1), (kba, 0, 1)]
            extra_ports[f"{comp.name}.port_a1"] = (kab, 0, 1)
            extra_ports[f"{comp.name}.port_a2"] = (kba, 0, -1)
        else:
            kind = f"door_c:{comp.name}"
            geo = dict(nCom=int(p.get("nCom", 10)), wOpe=float(common["wOpe"]),
                       hOpe=float(common["hOpe"]), hA=float(p.get("hA", 2.7 / 2)),
                       hB=float(p.get("hB", 2.7 / 2)), dp_turbulent=common["dp_turbulent"],
                       vZer=float(p.get("vZer", 0.001)))
            try:
                if cls == "DoorDiscretizedOpen":
                    el, head = mbl_discretized_door(src=iA, tgt=iB, medium=med, kind=kind,
                                                    CD=p.get("CD", 0.65), **geo)
                else:
                    LClo = _param(comp, "LClo", self.errors, required=True)
                    y = self.sig.require(comp, "y")
                    if LClo is None or y is None:
                        return
                    self._inputs[f"{comp.name}.y"] = y
                    el, head = mbl_discretized_operable_door(
                        src=iA, tgt=iB, medium=med, kind=kind, y_key=f"{comp.name}.y",
                        LClo=float(LClo), CDOpe=p.get("CDOpe", 0.65),
                        CDClo=p.get("CDClo", 0.65), CDCloRat=p.get("CDCloRat", 1.0),
                        dpCloRat=p.get("dpCloRat", 4.0), mOpe=p.get("mOpe", 0.5),
                        mClo=p.get("mClo", 0.65), **geo)
            except ValueError as exc:
                self.errors.append(f"{comp.name} ({comp.cls}): {exc}")
                return
            for _ in range(geo["nCom"]):
                net.add_edge(door.side_a, door.side_b, kind=kind)
            elements.append(el)
            drives.append(head)
            kinds[comp.name] = (kind,)
            edge_dirs[comp.name] = [(kind, j, 1) for j in range(geo["nCom"])]
        pressure_edges.append((iA, iB))

    def _zonal(self, zf: TwoWayEdge, net, index, elements, kinds, edge_dirs,
               extra_ports) -> None:
        comp, med = zf.component, self.med
        cls = comp.cls.rsplit(".", 1)[-1]
        iA, iB = index[zf.side_a], index[zf.side_b]
        kab, kba = f"zonal_ab:{comp.name}", f"zonal_ba:{comp.name}"
        if cls == "ZonalFlow_ACS":
            V = _param(comp, "V", self.errors, required=True)
            acs = self.sig.require(comp, "ACS")
            if V is None or acs is None:
                return
            self._inputs[f"{comp.name}.ACS"] = acs
            # ZonalFlow_ACS.mo:4: useDefaultProperties = false by default.
            use_default = bool(comp.parameters.get("useDefaultProperties", False))
            specs = [(kab, "ab", f"{comp.name}.ACS"), (kba, "ba", f"{comp.name}.ACS")]
        else:
            mab = self.sig.require(comp, "mAB_flow")
            mba = self.sig.require(comp, "mBA_flow")
            if mab is None or mba is None:
                return
            self._inputs[f"{comp.name}.mAB_flow"] = mab
            self._inputs[f"{comp.name}.mBA_flow"] = mba
            V, use_default = None, True
            specs = [(kab, "ab", f"{comp.name}.mAB_flow"), (kba, "ba", f"{comp.name}.mBA_flow")]
        for kind, direction, key in specs:
            net.add_edge(zf.side_a, zf.side_b, kind=kind)
            elements.append(_ZonalFlowEdge(kind=kind, direction=direction, src=iA, tgt=iB,
                                           medium=med, key=key, V=V, use_default=use_default))
        kinds[comp.name] = (kab, kba)
        edge_dirs[comp.name] = [(kab, 0, 1), (kba, 0, 1)]
        extra_ports[f"{comp.name}.port_a1"] = (kab, 0, 1)
        extra_ports[f"{comp.name}.port_a2"] = (kba, 0, -1)

    # ------------------------------------------------------------- sources
    def _sources(self, zones, index, water, K):
        med, n = self.med, len(index)
        zero = _t(0.0)
        s_air = [zero] * n
        s_th = [zero] * n
        s_sp = [[zero] * K for _ in range(n)]
        cp = med.specific_heat_cp(med.X_default[0] if med.has_moisture else 0.0)
        for comp, node in self.graph.sources:
            p = comp.parameters
            cls = comp.cls.rsplit(".", 1)[-1]
            if node not in zones:
                self.errors.append(f"{comp.name} ({comp.cls}): attached to boundary {node}; a "
                                   f"source must feed a volume")
                continue
            if p.get("use_m_flow_in"):
                m = self.sig.require(comp, "m_flow_in")
            else:
                m = _t(float(p.get("m_flow", 0.0)))
            if m is None:
                continue
            if bool((m < 0).any()):
                self.errors.append(f"{comp.name} ({comp.cls}): a negative (extracting) mass "
                                   f"flow is not supported")
                continue
            if cls == "TraceSubstancesFlowSource":
                T_in = _t(med.T_default)  # h_default
                X_in = med.X_default[0] if med.has_moisture else 0.0
                name = str(p.get("substanceName", "CO2"))
                C_in = [_t(1.0 if s == name else 0.0) for s in self.species_names]
            else:  # MassFlowSource_T
                T_in = (self.sig.require(comp, "T_in") if p.get("use_T_in")
                        else _t(float(p.get("T", med.T_default))))
                X = p.get("X", list(med.X_default))
                X_in = float(X[0]) if med.has_moisture and X else 0.0
                C_par = p.get("C", [0.0] * len(self.species_names))
                C_in = []
                for k in range(len(self.species_names)):
                    if p.get("use_C_in"):
                        C_in.append(self.sig.require(comp, f"C_in[{k + 1}]"))
                    else:
                        C_in.append(_t(float(C_par[k]) if k < len(C_par) else 0.0))
                if T_in is None or any(c is None for c in C_in):
                    continue
            i = index[node]
            s_air[i] = s_air[i] + m
            s_th[i] = s_th[i] + cp * m * T_in
            for k, c in enumerate(C_in):
                s_sp[i][k] = s_sp[i][k] + m * c
            if water is not None:
                s_sp[i][water] = s_sp[i][water] + m * X_in
        return s_air, s_th, s_sp


def build(graph: ComponentGraph, doc: ModelicaDoc) -> tuple[Model, State, Drivers,
                                                           ModelicaNames]:
    """Assemble the noodl `Model`, initial `State`, `Drivers` and `ModelicaNames`.

    Drivers that a signal makes time-varying are stored twice: at their first-time value
    under their own key (so `model.step(state, drivers, dt)` runs as is), and as the full
    series over `names.times` under `"series:<key>"` (plus `"series:time"`), which
    `run.simulate`/`run.step_drivers` slice per step. See the module docstring for the
    conversion rules and `ModelicaNames` for the name mapping.
    """
    return _Builder(graph, doc).build()
