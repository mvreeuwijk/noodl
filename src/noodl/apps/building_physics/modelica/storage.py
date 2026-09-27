"""Volume mass storage: MBL's compressible `MixingVolume` balances in noodl physics' layers.

MBL (v13, `Fluid/Interfaces/ConservationEquation.mo:245-310`) gives every volume whose
`massDynamics` is not `SteadyState` a fluid mass `m = V rho_m(p, T, X)` that changes with the
net port inflow, so the zone pressure is a state:

    dm/dt       = w = sum_in q - sum_out q + s                        (mass)
    d(m u)/dt   = sum_in q h_in - sum_out q h + s h_s + Q            (energy, U = m u)
    m dX/dt     = sum_in q X_in - sum_out q X + s X_s                 (water, `der(Xi) = mbXi/m`)
    d(m C)/dt   = sum_in q C_in - sum_out q C + s C_s                 (trace substances)

`rho_m` is `Medium.density` at the actual state for a single-substance medium, and at
`setState_phX(p, h, X_default)` for the two moist-air media (`_simplify_mWat_flow`, `:246-258`):
for `Buildings.Media.Air` that is `dStp p/pStp` either way; for `PerfectGas` it is the ideal
gas at `X_default` and the temperature `T(h(T, X), X_default)` (`PerfectGas.mo:494`). Note the
water balance: MBL integrates the mass FRACTION, so water is not conserved when `m` changes
(`d(m X)/dt = mbXi + X w`), and the energy balance carries the latent enthalpy of that change.

How the four equations map onto noodl physics' layers (implicit in each step: one Newton
airflow solve and the `exact` transport steps per `coupling="iterate"` pass):

* **Mass**: a `NodeSource` on the air layer at every storage zone, the backward-Euler
  withdrawal `(m - m_prev)/dt` (`ZoneStorage`). `rho_m` is proportional to `p` in all three
  media, `rho_m = p k(T, X)`, and the difference is formed as
  `V [k (phi - phi_prev) + (k - k_prev)(p_ref + phi_prev)]` in the gauge pressure `phi`, not
  as a difference of absolute masses, whose round-off (`eps m`, 2.6e-4 kg for the 1e12 m3
  "outside" volumes of some MBL examples) would swamp a step's mass change; for
  `Buildings.Media.Air` `k` is a constant and the second term is exactly zero. `T`, `X` are
  the closure's drivers of the pass; `phi_prev`, `k_prev`, the state at the start of the
  step, are the closure-carried state `"air.storage"`. The zone pressure is then an unknown
  of the air layer, grounded by the storage itself, so closed zone groups need no pressure
  reference.
* **Energy**: substituting the mass and water balances, with `u_T = du/dT`, `u_X = du/dX`,
  the carrier `cp` of the thermal layer (one value at `X_default`, as in the quasi-steady
  route) and `net = sum_in q - sum_out q` of the edges,

      (m u_T + CSen) dT/dt = sum_in cp q T_in - sum_out cp q T - u_T net T
                             + e(T, X) w + cp s T_s - u_T s T + Q,
      e = (h - u) - (cp - u_T) T - X u_X,

  which is the thermal `TransportLayer` with capacity `m u_T + CSen` (a per-step driver),
  `dilution = lam_T` and the source terms, `(lam_T - u_T) T net` restoring the balance's own
  dilution (`thermal_dilution`, "Coupling" below). `e` vanishes for an ideal
  gas without moisture (`SimpleAir`: `h - u = R T`, `cp - u_T = R`); for
  `Buildings.Media.Air` it is `pStp/dStp - X h_X` (`u = h - pStp/dStp`: the flow work `V dp`
  of the pressure-only density, less the latent heat of MBL's water-fraction change). The
  cross term `(h_X - u_X)(X_in - X)` of an inflow of another composition (zero for
  `Buildings.Media.Air`, `R_X T dX` for `PerfectGas`) is neglected, like the per-zone `cp`.
* **Water** is the species layer's conservative flux form (`dilution = 0`) with capacity
  `m`; a **trace substance** has `dilution = 1` and a source correction `- s C`.

`w`, `m`, `e w`, `- u_T s T` and `- s C` use the pass's (previous iterate's) state; at
convergence they are the end-of-step values, like everything else in the step. The carried
`"air.storage"` is likewise the last pass's input state, equal to its output to the coupling
tolerance (`run.simulate` re-evaluates it at each returned state).

Coupling. The step is Hensen's onion (`Model(coupling="iterate")`): closures, airflow, heat
and species, repeated until the temperatures and mass fractions settle. With storage, pressure
and temperature are strongly coupled -- an ideal gas zone's mass depends on its temperature,
and the temperature on the air that mass exchange compresses into it -- and the plain split
converges at `R/cv ~ 0.4` per pass. Three choices, none of which moves the converged step,
make it contract at ~0.06 per pass (measured on ClosedDoors: 6 passes to 1e-8 K):

* the thermal operator carries `- lam_T net T`, `lam_T = u_T - e/T` at the medium default
  (`thermal_dilution`), so that the whole `- u_T net T + e net` is evaluated with the pass's
  own flows, leaving only the small remainder `(lam_T - u_T + e/T) net T` to the previous one;
* the air-layer storage is blended with the previous pass's storage rate with the gain
  `a = -V p k_T (cp - lam_T) T / C` of the temperature that storage rate implies
  (`ZoneStorage`): for an ideal gas `a = R/cv` and the effective compressibility is the
  isentropic `V k cv/cp`, the one fast pressure changes actually follow;
* the passes are not relaxed (`iterate_relaxation = 1`).

Initial state (`ConservationEquation.mo:199-231`): `FixedInitial` (and `DynamicFreeInitial`,
whose start value OpenModelica keeps) fixes `p = p_start`, so the t = StartTime flows are the
element laws at the start pressures, the imbalance going into storage; `SteadyStateInitial` is
`der(p) = 0`, a quasi-steady zone at t = StartTime. The t = StartTime row of `run.simulate`
therefore evaluates the flows at the start pressures when every storage zone is fixed there,
and otherwise solves `init_layer`, the air layer with the fixed zones as pressure boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from noodl.elements.media import (
    _CP_AIR,
    _CP_SIMPLEAIR,
    _CP_STEAM,
    _MM_AIR,
    _MODELICA_CONSTANTS_R,
    _R_AIR,
    _R_H2O,
    MBLMedium,
)
from noodl.nodesources import NodeSource

Tensor = torch.Tensor
F64 = torch.float64

# Buildings/Utilities/Psychrometrics/Constants.mo:12 (h_fg); Air.mo:12 and PerfectGas.mo:10
# (reference_T = 273.15 K).
_H_FG = 2501014.5
_T_REF = 273.15


@dataclass(frozen=True)
class StorageThermo:
    """What the storage balances need from one MBL medium (module docstring).

    `k(T, X)` is the volume mass density per unit pressure (`rho_m = p k`), `k_T` its
    derivative in `T`; `h_minus_u(T, X)`
    and `u_X(T, X)` are `h - u` and `du/dX`; `u_T` is `du/dT` at `X_default` (the thermal
    dilution and the capacity per unit mass, a float); `cp` is the thermal layer's carrier.
    The callables take and return tensors and are differentiable."""

    k: Callable[[Tensor, Tensor], Tensor]
    k_T: Callable[[Tensor, Tensor], Tensor]
    h_minus_u: Callable[[Tensor, Tensor], Tensor]
    u_X: Callable[[Tensor, Tensor], Tensor]
    u_T: float
    cp: float

    def e(self, T: Tensor, X: Tensor) -> Tensor:
        """`e = (h - u) - (cp - u_T) T - X u_X` (J/kg), the storage-rate coefficient of the
        energy source."""
        return self.h_minus_u(T, X) - (self.cp - self.u_T) * T - X * self.u_X(T, X)


def _h_X(T: Tensor) -> Tensor:
    # dh/dX of both moist-air media (Air.mo:123-124, PerfectGas.mo:60-61).
    return (_CP_STEAM - _CP_AIR) * (T - _T_REF) + _H_FG


def thermo(med: MBLMedium) -> StorageThermo:
    """The `StorageThermo` of one of the three supported media."""
    if med.name == "Buildings.Media.Air":
        # Air.mo:43-45 (pStp, dStp), :123-131 (h, u = h - pStp/dStp), :210-215 (density).
        c = med.p_default / 1.2
        k = 1.2 / med.p_default
        cp = med.specific_heat_cp(med.X_default[0])
        return StorageThermo(
            k=lambda T, X: torch.full_like(torch.as_tensor(T, dtype=F64), k),
            k_T=lambda T, X: torch.zeros_like(torch.as_tensor(T, dtype=F64)),
            h_minus_u=lambda T, X: torch.full_like(torch.as_tensor(T, dtype=F64), c),
            u_X=lambda T, X: _h_X(T),
            u_T=cp, cp=cp,
        )
    if med.name == "Buildings.Media.Specialized.Air.PerfectGas":
        # PerfectGas.mo:60-66 (h, R_s(X), u = h - R_s T), :229-231 (density), :494
        # (temperature_phX: the volume mass is at setState_phX(p, h, X_default)).
        X_d = med.X_default[0]
        cp = med.specific_heat_cp(X_d)
        R_d = _R_AIR * (1 - X_d) + _R_H2O * X_d

        def T_d(T, X):
            cp_X = _CP_AIR * (1 - X) + _CP_STEAM * X
            return _T_REF + ((T - _T_REF) * cp_X + _H_FG * (X - X_d)) / cp

        def k_T(T, X):
            cp_X = _CP_AIR * (1 - X) + _CP_STEAM * X
            return -cp_X / (cp * R_d * T_d(T, X) ** 2)

        return StorageThermo(
            k=lambda T, X: 1.0 / (R_d * T_d(T, X)), k_T=k_T,
            h_minus_u=lambda T, X: (_R_AIR * (1 - X) + _R_H2O * X) * T,
            u_X=lambda T, X: _h_X(T) - (_R_H2O - _R_AIR) * T,
            u_T=cp - R_d, cp=cp,
        )
    if med.name == "Modelica.Media.Air.SimpleAir":
        # MSL Media/package.mo PartialSimpleIdealGasMedium: h = cp (T - T0), u = h - R T,
        # d = p/(R T); SimpleAir.mo:6-8 (cp_const, R_gas).
        R = _MODELICA_CONSTANTS_R / _MM_AIR
        return StorageThermo(
            k=lambda T, X: 1.0 / (R * T),
            k_T=lambda T, X: -1.0 / (R * T ** 2),
            h_minus_u=lambda T, X: R * T,
            u_X=lambda T, X: torch.zeros_like(torch.as_tensor(T, dtype=F64)),
            u_T=_CP_SIMPLEAIR - R, cp=_CP_SIMPLEAIR,
        )
    raise KeyError(f"volume mass storage: no thermodynamics for medium {med.name!r}")


def thermal_dilution(th: StorageThermo, T_ref: float, X_ref: float) -> float:
    """The thermal layer's `dilution`: `u_T - e(T_ref, X_ref)/T_ref`.

    The balance's `- u_T net T + e w` (module docstring, "Energy") has `w = net` at a zone
    with no source, and `e` changes little with `T`, so `- (u_T - e/T) net T` is most of it:
    carried by the operator it is implicit in the step's own flows, and only the remainder
    `(lam - u_T + e/T) net T` lags a coupling pass. `u_T` itself for an ideal gas without
    moisture (`e = 0`)."""
    T = torch.tensor(T_ref, dtype=F64)
    return float(th.u_T - th.e(T, torch.tensor(X_ref, dtype=F64)) / T)


def mass_change(V, k, phi, k_prev, phi_prev, p_ref: float) -> Tensor:
    """`m - m_prev = V [k (phi - phi_prev) + (k - k_prev)(p_ref + phi_prev)]` (module
    docstring, "Mass")."""
    return V * (k * (phi - phi_prev) + (k - k_prev) * (p_ref + phi_prev))


class ZoneStorage(NodeSource):
    """The air-layer withdrawal at the storage zones (module docstring, "Mass"):

        w = (rate (m - m_prev) + a w_fed) / (1 + a),

    `rate (m - m_prev)` the backward-Euler storage at the pass's temperature, `w_fed` the
    storage rate of the pass's input state and `a >= 0` the coupling gain (module
    docstring, "Coupling"). At a converged step `w_fed = rate (m - m_prev)` and `w` is
    the backward-Euler storage itself. Reads the full-node drivers `"T"`, `"X_w"` and the
    per-zone drivers `"air.storage_prev"` (`(..., n_s, 2)`: `phi`, `k` at the start of the
    step), `"air.storage_rate"` (1/s), `"air.storage_w_fed"` (kg/s) and `"air.storage_gain"`,
    which `StorageClosure` writes."""

    def __init__(self, nodes, volumes: Tensor, p_ref: float, th: StorageThermo) -> None:
        super().__init__(nodes)
        self.register_buffer("V", torch.as_tensor(volumes, dtype=F64))
        self.p_ref = float(p_ref)
        self.th = th

    def _k(self, drivers: Mapping) -> Tensor:
        return self.th.k(drivers["T"][..., self.nodes], drivers["X_w"][..., self.nodes])

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        prev = drivers["air.storage_prev"]
        dm = mass_change(self.V, self._k(drivers), phi_nodes, prev[..., 1], prev[..., 0],
                         self.p_ref)
        a = drivers["air.storage_gain"]
        return (drivers["air.storage_rate"] * dm + a * drivers["air.storage_w_fed"]) / (1 + a)

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        slope = (drivers["air.storage_rate"] * self.V * self._k(drivers)
                 / (1 + drivers["air.storage_gain"]))
        return slope.expand(torch.broadcast_shapes(slope.shape, phi_nodes.shape))


class StorageClosure:
    """Writes the storage drivers of every pass (module docstring) and carries
    `"air.storage"` (`(..., n, 2)`: gauge pressure and `k` of every node at the state's
    time) across steps.

    Runs after the reader's `_MBLClosure` (`mbl`), whose `"T"`, `"X_w"` and `"p_abs"` drivers
    it reads, and whose `species` gives the full-node mass fractions. With a `StepContext` it
    writes `"air.storage_prev"`/`"air.storage_rate"` (the backward-Euler storage of the
    step), `"thermal.capacity"`, `"species.capacity"` and the storage terms added to
    `"thermal.sources"`/`"species.sources"`; without one (a query) only the carried state.

    A zone wired straight to a boundary (`attached`) is held at the boundary's pressure, so it
    is not an air-layer unknown: the air it stores or releases is exchanged with that boundary,
    `f = w - net - s`, and an inflow carries the boundary's temperature and composition
    (the drivers `"storage.T_attached"` `(n,)` and `"storage.x_attached"` `(n, K)`, zero
    elsewhere)."""

    integrates = True
    state_keys = ("air.storage",)

    def __init__(self, *, mbl, th: StorageThermo, p_ref: float, volumes: Tensor,
                 storage: Tensor, m_fixed: Tensor, air_nodes: Tensor, attached: Tensor,
                 air_src: Tensor, air_tgt: Tensor, csen: Tensor,
                 th_interior: Tensor | None, sp_interior: Tensor | None,
                 sp_dilution: Tensor | None, lam_T: float, init_layer=None) -> None:
        self.mbl = mbl
        self.th = th
        self.lam_T = float(lam_T)       # the thermal layer's dilution (thermal_dilution)
        self.p_ref = float(p_ref)
        self.V = volumes                # (n,) zone volumes, 0 at boundaries
        self.storage = storage          # (n,) bool: zone with a dynamic mass balance
        self.m_fixed = m_fixed          # (n,) V rho_start of a zone without one, else 0
        self.air_nodes = air_nodes      # storage zones that are air-layer unknowns
        self.attached = attached        # storage zones held at a boundary's pressure
        self.air_src, self.air_tgt = air_src, air_tgt
        self.csen = csen                # (n,) CSen (J/K)
        self.th_interior, self.sp_interior = th_interior, sp_interior
        self.sp_dilution = sp_dilution  # (K,)
        # The air layer of the t = StartTime equations when some storage zone is not held
        # at p_start there (module docstring), else None.
        self.init_layer = init_layer
        n = volumes.numel()
        self._th_mask = torch.zeros(n, dtype=torch.bool)
        if th_interior is not None:
            self._th_mask[th_interior] = True
        self._sp_mask = torch.zeros(n, dtype=torch.bool)
        if sp_interior is not None:
            self._sp_mask[sp_interior] = True
        self._att_mask = torch.zeros(n, dtype=torch.bool)
        self._att_mask[attached] = True

    def carried(self, p: Tensor, T: Tensor, X: Tensor) -> Tensor:
        """The `"air.storage"` value `(..., n, 2)` at absolute pressures `p` (full node)."""
        k = self.th.k(T, X)
        return torch.stack(torch.broadcast_tensors(p - self.p_ref, k), dim=-1)

    def masses(self, p: Tensor, T: Tensor, X: Tensor) -> Tensor:
        """Full-node fluid mass: `V k p` at the storage zones, `V rho_start` at a zone
        without a dynamic mass balance, 0 at boundaries."""
        m = self.V * self.th.k(T, X) * p
        return torch.where(self.storage, m, self.m_fixed)

    def _net(self, q: Tensor) -> Tensor:
        full = torch.zeros(q.shape[:-1] + self.V.shape, dtype=q.dtype)
        return full.index_add(-1, self.air_tgt, q).index_add(-1, self.air_src, -q)

    def __call__(self, state, drivers, ctx=None):
        T, X, p = drivers["T"], drivers["X_w"], drivers["p_abs"]
        now = self.carried(p, T, X)
        out: dict[str, Tensor] = {"air.storage": now}
        if ctx is None or ctx.dt is None:
            return out
        prev = state["air.storage"]
        dt = float(ctx.dt)
        out["air.storage_prev"] = prev[..., self.air_nodes, :]
        out["air.storage_rate"] = torch.full((self.air_nodes.numel(),), 1.0 / dt, dtype=F64)
        s = drivers.get("air.sources")
        s = torch.zeros_like(p) if s is None else s.to(F64)
        q = state.get("air.q")
        net = self._net(q) if q is not None else torch.zeros_like(p)
        dm = mass_change(self.V, now[..., 1], now[..., 0], prev[..., 1], prev[..., 0],
                         self.p_ref)
        w = torch.where(self.storage, dm / dt, torch.zeros_like(dm))   # (n,) dm/dt
        m = self.masses(p, T, X)
        th = self.th
        # Coupling (module docstring): the gain `a` of the temperature a storage rate
        # implies, fed back into that rate.
        cap = m * th.u_T + self.csen
        a = -self.V * p * th.k_T(T, X) * (th.cp - self.lam_T) * T / cap
        a = torch.where(self._th_mask & self.storage, a, torch.zeros_like(a))
        w_fed = torch.where(self.storage, net + s, torch.zeros_like(net))
        out["air.storage_gain"] = a[..., self.air_nodes]
        out["air.storage_w_fed"] = w_fed[..., self.air_nodes]
        f_in = f_out = None
        ff = torch.zeros_like(w)
        if self.attached.numel():
            ff = torch.where(self._att_mask, w - net - s, torch.zeros_like(w))
            f_in, f_out = ff.clamp(min=0.0), (-ff).clamp(min=0.0)
        if self.th_interior is not None:
            out["thermal.capacity"] = cap[..., self.th_interior]
            # (thermal_dilution): the operator carries `- lam_T net T`; the balance wants
            # `- u_T net T + e w`, `w = net + s + f` at a storing zone.
            e = torch.where(self.storage, th.e(T, X), torch.zeros_like(T))
            src = ((self.lam_T - th.u_T) * T + e) * net + e * (s + ff) - th.u_T * s * T
            if f_in is not None:
                T_b = drivers["storage.T_attached"]
                src = src + f_in * (th.cp * T_b - th.u_T * T) - f_out * (th.cp - th.u_T) * T
            src = torch.where(self._th_mask, src, torch.zeros_like(src))
            base = drivers.get("thermal.sources")
            out["thermal.sources"] = src if base is None else base + src
        if self.sp_interior is not None:
            out["species.capacity"] = m[..., self.sp_interior]
            x = self.mbl.species(state, drivers)          # (..., n, K)
            lam = self.sp_dilution.to(F64)
            src = -(s.unsqueeze(-1) * x) * lam
            if f_in is not None:
                x_b = drivers["storage.x_attached"]
                src = (src + f_in.unsqueeze(-1) * (x_b - lam * x)
                       - f_out.unsqueeze(-1) * (1.0 - lam) * x)
            src = torch.where(self._sp_mask.unsqueeze(-1), src, torch.zeros_like(src))
            base = drivers.get("species.sources")
            out["species.sources"] = src if base is None else base + src
        return out
