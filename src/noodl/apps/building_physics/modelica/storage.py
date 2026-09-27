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
  `cp(X) = dh/dT`, `g = h - u - X u_X`, the storage rate `w` and the sums over every inflow
  (edges, sources, an attached boundary), MBL's balance is exactly

      (m u_T + CSen) dT/dt = sum_in q [cp(X_in)(T_in - T) + (h_X - u_X)(X_in - X)] + g w + Q.

  The thermal `TransportLayer` carries one `cp` (at `X_default`) and `- lam net T`
  (`dilution = lam`, `thermal_dilution`); with capacity `m u_T(X) + CSen` (a per-step
  driver) the closure adds the rest as sources (`StorageClosure.terms`):
  `(g - (cp - lam) T) net + g (s + f) - cp s T` and, per inflowing edge,
  `|q| [(cp(X_up) - cp)(T_up - T) + (h_X - u_X)(X_up - X)]`. `g` is `R T` for `SimpleAir`,
  `pStp/dStp - X h_X` for `Buildings.Media.Air` (`u = h - pStp/dStp`: the flow work `V dp`
  of the pressure-only density, less the latent heat of MBL's water-fraction change), and
  `h_X - u_X` is `R_X T` for `PerfectGas`, zero for the others.
* **Water** is the species layer's conservative flux form (`dilution = 0`) with capacity
  `m`; a **trace substance** has `dilution = 1` and a source correction `- s C`.

In `Model.step` (`run.simulate(scheme="implicit")`) the storage is backward Euler and
`w`, `m` and the source terms use the pass's (previous iterate's) state; at convergence they
are the end-of-step values, like everything else in the step. `run.simulate(scheme="midpoint")`
takes the mean of the terms at the two ends of the step and the volumes' storage rate by the
L-stable BDF2 formula (`run._Midpoint`). The carried
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
    derivative in `T`; `h_minus_u(T, X)`, `u_X(T, X)` and `u_T(T, X)` are `h - u`, `du/dX`
    and `du/dT`; `cp_X(X)` is `dh/dT`, `hX_minus_uX(T)` is `dh/dX - du/dX`. `cp` is the
    thermal layer's carrier (`cp` at `X_default`) and `u_T0` is `u_T` at `X_default`. The
    callables take and return tensors and are differentiable."""

    k: Callable[[Tensor, Tensor], Tensor]
    k_T: Callable[[Tensor, Tensor], Tensor]
    h_minus_u: Callable[[Tensor, Tensor], Tensor]
    u_X: Callable[[Tensor, Tensor], Tensor]
    u_T: Callable[[Tensor, Tensor], Tensor]
    cp_X: Callable[[Tensor], Tensor]
    hX_minus_uX: Callable[[Tensor], Tensor]
    u_T0: float
    cp: float

    def g(self, T: Tensor, X: Tensor) -> Tensor:
        """`g = h - u - X u_X` (J/kg): the energy a unit of stored mass brings with it beyond
        the zone's own `u` (module docstring, "Energy")."""
        return self.h_minus_u(T, X) - X * self.u_X(T, X)


def _t64(T) -> Tensor:
    return torch.as_tensor(T, dtype=F64)


def _h_X(T: Tensor) -> Tensor:
    # dh/dX of both moist-air media (Air.mo:123-124, PerfectGas.mo:60-61).
    return (_CP_STEAM - _CP_AIR) * (T - _T_REF) + _H_FG


def _cp_moist(X):
    return _CP_AIR * (1 - X) + _CP_STEAM * X


def thermo(med: MBLMedium) -> StorageThermo:
    """The `StorageThermo` of one of the three supported media."""
    if med.name == "Buildings.Media.Air":
        # Air.mo:43-45 (pStp, dStp), :123-131 (h, u = h - pStp/dStp), :210-215 (density).
        c = med.p_default / 1.2
        k = 1.2 / med.p_default
        cp = med.specific_heat_cp(med.X_default[0])
        return StorageThermo(
            k=lambda T, X: torch.full_like(_t64(T), k),
            k_T=lambda T, X: torch.zeros_like(_t64(T)),
            h_minus_u=lambda T, X: torch.full_like(_t64(T), c),
            u_X=lambda T, X: _h_X(T),
            u_T=lambda T, X: _cp_moist(X) + 0.0 * T,
            cp_X=_cp_moist,
            hX_minus_uX=lambda T: torch.zeros_like(_t64(T)),
            u_T0=cp, cp=cp,
        )
    if med.name == "Buildings.Media.Specialized.Air.PerfectGas":
        # PerfectGas.mo:60-66 (h, R_s(X), u = h - R_s T), :229-231 (density), :494
        # (temperature_phX: the volume mass is at setState_phX(p, h, X_default)).
        X_d = med.X_default[0]
        cp = med.specific_heat_cp(X_d)
        R_d = _R_AIR * (1 - X_d) + _R_H2O * X_d

        def T_d(T, X):
            return _T_REF + ((T - _T_REF) * _cp_moist(X) + _H_FG * (X - X_d)) / cp

        return StorageThermo(
            k=lambda T, X: 1.0 / (R_d * T_d(T, X)),
            k_T=lambda T, X: -_cp_moist(X) / (cp * R_d * T_d(T, X) ** 2),
            h_minus_u=lambda T, X: (_R_AIR * (1 - X) + _R_H2O * X) * T,
            u_X=lambda T, X: _h_X(T) - (_R_H2O - _R_AIR) * T,
            u_T=lambda T, X: _cp_moist(X) - (_R_AIR * (1 - X) + _R_H2O * X) + 0.0 * T,
            cp_X=_cp_moist,
            hX_minus_uX=lambda T: (_R_H2O - _R_AIR) * T,
            u_T0=cp - R_d, cp=cp,
        )
    if med.name == "Modelica.Media.Air.SimpleAir":
        # MSL Media/package.mo PartialSimpleIdealGasMedium: h = cp (T - T0), u = h - R T,
        # d = p/(R T); SimpleAir.mo:6-8 (cp_const, R_gas).
        R = _MODELICA_CONSTANTS_R / _MM_AIR
        cv = _CP_SIMPLEAIR - R
        return StorageThermo(
            k=lambda T, X: 1.0 / (R * T),
            k_T=lambda T, X: -1.0 / (R * T ** 2),
            h_minus_u=lambda T, X: R * T,
            u_X=lambda T, X: torch.zeros_like(_t64(T)),
            u_T=lambda T, X: torch.full_like(_t64(T), cv),
            cp_X=lambda X: torch.full_like(_t64(X), _CP_SIMPLEAIR),
            hX_minus_uX=lambda T: torch.zeros_like(_t64(T)),
            u_T0=cv, cp=_CP_SIMPLEAIR,
        )
    raise KeyError(f"volume mass storage: no thermodynamics for medium {med.name!r}")


def thermal_dilution(th: StorageThermo, T_ref: float, X_ref: float) -> float:
    """The thermal layer's `dilution`: `cp - g(T_ref, X_ref)/T_ref`.

    The balance's `- (cp - g/T) net T` (module docstring, "Energy") has `g/T` nearly
    constant (`R` for an ideal gas without moisture, where this is `cv`): carried by the
    operator it is implicit in the step's own flows, and only the remainder
    `(lam - cp + g/T) net T` lags a coupling pass."""
    T = torch.tensor(T_ref, dtype=F64)
    return float(th.cp - th.g(T, torch.tensor(X_ref, dtype=F64)) / T)


def mass_change(V, k, phi, k_prev, phi_prev, p_ref: float) -> Tensor:
    """`m - m_prev = V [k (phi - phi_prev) + (k - k_prev)(p_ref + phi_prev)]` (module
    docstring, "Mass")."""
    return V * (k * (phi - phi_prev) + (k - k_prev) * (p_ref + phi_prev))


class ZoneStorage(NodeSource):
    """The air-layer withdrawal at the storage zones (module docstring, "Mass"):

        w = (rate (m - m_prev) - offset + a w_fed) / (1 + a),

    `rate (m - m_prev) - offset` the discrete storage rate (backward Euler: `rate = 1/h`,
    `offset = 0`; the trapezoidal rule of `run.simulate(scheme="midpoint")`: `rate = 2/h`,
    `offset` the start-of-step rate), `w_fed` the storage rate of the pass's input state and
    `a >= 0` the coupling gain (module docstring, "Coupling"). At a converged step
    `w_fed = rate (m - m_prev) - offset` and `w` is the discrete storage rate itself. Reads
    the full-node drivers `"T"`, `"X_w"` and the per-zone drivers `"air.storage_prev"`
    (`(..., n_s, 2)`: `phi`, `k` at the start of the step), `"air.storage_rate"` (1/s),
    `"air.storage_offset"`, `"air.storage_w_fed"` (kg/s) and `"air.storage_gain"`."""

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
        rate = drivers["air.storage_rate"] * dm - drivers["air.storage_offset"]
        return (rate + a * drivers["air.storage_w_fed"]) / (1 + a)

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
    writes `"air.storage_*"` (the backward-Euler storage of the step), `"thermal.capacity"`,
    `"species.capacity"` and the storage terms added to `"thermal.sources"`/
    `"species.sources"` (`terms`); without one (a query) only the carried state.

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
                 sp_dilution: Tensor | None, lam_T: float, water: int | None,
                 init_layer=None) -> None:
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
        self.water = water              # the water species' index, or None
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
        self._air_mask = torch.zeros(n, dtype=torch.bool)
        self._air_mask[air_nodes] = True

    def carried(self, p: Tensor, T: Tensor, X: Tensor) -> Tensor:
        """The `"air.storage"` value `(..., n, 2)` at absolute pressures `p` (full node)."""
        k = self.th.k(T, X)
        return torch.stack(torch.broadcast_tensors(p - self.p_ref, k), dim=-1)

    def masses(self, p: Tensor, T: Tensor, X: Tensor) -> Tensor:
        """Full-node fluid mass: `V k p` at the storage zones, `V rho_start` at a zone
        without a dynamic mass balance, 0 at boundaries."""
        m = self.V * self.th.k(T, X) * p
        return torch.where(self.storage, m, self.m_fixed)

    def net(self, q: Tensor | None, like: Tensor) -> Tensor:
        """Full-node net inflow of the air-layer edges (kg/s)."""
        if q is None:
            return torch.zeros_like(like)
        full = torch.zeros(q.shape[:-1] + self.V.shape, dtype=q.dtype)
        return full.index_add(-1, self.air_tgt, q).index_add(-1, self.air_src, -q)

    def rate_from_mass(self, now: Tensor, prev: Tensor, h: float) -> Tensor:
        """Full-node `(m - m_prev)/h` from two `"air.storage"` values, 0 off storage."""
        dm = mass_change(self.V, now[..., 1], now[..., 0], prev[..., 1], prev[..., 0],
                         self.p_ref)
        return torch.where(self.storage, dm / h, torch.zeros_like(dm))

    def _inflow_correction(self, q: Tensor | None, T: Tensor, X: Tensor) -> Tensor:
        """Full-node `sum_in |q| [(cp(X_up) - cp)(T_up - T) + (h_X - u_X)(X_up - X)]` over
        the air-layer edges flowing into each node: the part of MBL's moist-air heat carrier
        the thermal layer's one `cp` leaves out (module docstring, "Energy")."""
        if q is None:
            return torch.zeros_like(T)
        up = torch.where(q >= 0, self.air_src, self.air_tgt)
        dn = torch.where(q >= 0, self.air_tgt, self.air_src)
        th = self.th
        T_up, T_dn = T.gather(-1, up), T.gather(-1, dn)
        X_up, X_dn = X.gather(-1, up), X.gather(-1, dn)
        w = q.abs() * ((th.cp_X(X_up) - th.cp) * (T_up - T_dn)
                       + th.hX_minus_uX(T_dn) * (X_up - X_dn))
        out = torch.zeros(q.shape[:-1] + T.shape[-1:], dtype=F64)
        return out.scatter_add(-1, dn.expand(w.shape), w)

    def terms(self, state: Mapping, drivers: Mapping, w: Tensor, net: Tensor,
              s: Tensor, dnet: Tensor | None = None) -> dict[str, Tensor]:
        """The transport layers' storage terms at one state: `"thermal.capacity"`,
        `"species.capacity"` (interior order) and the EXTRA sources `"thermal.extra"`,
        `"species.extra"` (full node) to add to the base ones, for the full-node storage
        rate `w` (kg/s), net edge inflow `net` and air sources `s` (module docstring,
        "Energy" and "Water"). The discrete scheme decides `w` (`__call__`,
        `run._Midpoint`). `dnet`, when given, is the part of the storage rate the transport
        layers' own flows do not carry (`w - s - net` of the flows they are stepped with):
        the terms the balances would otherwise take from those flows are added for it."""
        T, X, p = drivers["T"], drivers["X_w"], drivers["p_abs"]
        th = self.th
        m = self.masses(p, T, X)
        out: dict[str, Tensor] = {}
        f = torch.where(self._att_mask, w - net - s, torch.zeros_like(w))
        f_in, f_out = f.clamp(min=0.0), (-f).clamp(min=0.0)
        if self.th_interior is not None:
            cap = m * th.u_T(T, X) + self.csen
            out["thermal.capacity"] = cap[..., self.th_interior]
            g = torch.where(self.storage, th.g(T, X), torch.zeros_like(T))
            c0, lam = th.cp, self.lam_T
            src = (g - (c0 - lam) * T) * net + g * (s + f) - c0 * s * T
            src = src + self._inflow_correction(drivers.get("_q"), T, X)
            if dnet is not None:
                src = src + torch.where(self.storage, th.g(T, X), torch.zeros_like(T)) * dnet
            if self.attached.numel():
                T_b = drivers["storage.T_attached"]
                x_b = drivers.get("storage.x_attached")
                X_b = X if (x_b is None or self.water is None) else x_b[..., self.water]
                src = src + f_in * (th.cp_X(X_b) * (T_b - T)
                                    + th.hX_minus_uX(T) * (X_b - X))
            out["thermal.extra"] = torch.where(self._th_mask, src, torch.zeros_like(src))
        if self.sp_interior is not None:
            out["species.capacity"] = m[..., self.sp_interior]
            x = self.mbl.species(state, drivers)          # (..., n, K)
            lam = self.sp_dilution.to(F64)
            src = -(s.unsqueeze(-1) * x) * lam
            if dnet is not None:
                src = src + (1.0 - lam) * x * dnet.unsqueeze(-1)
            if self.attached.numel():
                x_b = drivers["storage.x_attached"]
                src = (src + f_in.unsqueeze(-1) * (x_b - lam * x)
                       - f_out.unsqueeze(-1) * (1.0 - lam) * x)
            out["species.extra"] = torch.where(self._sp_mask.unsqueeze(-1), src,
                                               torch.zeros_like(src))
        return out

    def gain(self, drivers: Mapping, cap: Tensor | None = None) -> Tensor:
        """Per air-storage node, the coupling gain `a` (module docstring, "Coupling")."""
        T, X, p = drivers["T"], drivers["X_w"], drivers["p_abs"]
        th = self.th
        cap = self.masses(p, T, X) * th.u_T(T, X) + self.csen
        a = -self.V * p * th.k_T(T, X) * (th.cp - self.lam_T) * T / cap
        a = torch.where(self._th_mask & self.storage, a, torch.zeros_like(a))
        return a[..., self.air_nodes]

    def __call__(self, state, drivers, ctx=None):
        T, X, p = drivers["T"], drivers["X_w"], drivers["p_abs"]
        now = self.carried(p, T, X)
        out: dict[str, Tensor] = {"air.storage": now}
        if ctx is None or ctx.dt is None:
            return out
        prev = state["air.storage"]
        dt = float(ctx.dt)
        n_s = self.air_nodes.numel()
        s = drivers.get("air.sources")
        s = torch.zeros_like(p) if s is None else s.to(F64)
        q = state.get("air.q")
        net = self.net(q, p)
        out["air.storage_prev"] = prev[..., self.air_nodes, :]
        out["air.storage_rate"] = torch.full((n_s,), 1.0 / dt, dtype=F64)
        out["air.storage_offset"] = torch.zeros(n_s, dtype=F64)
        out["air.storage_gain"] = self.gain(drivers)
        out["air.storage_w_fed"] = (net + s)[..., self.air_nodes]
        w = self.rate_from_mass(now, prev, dt)      # backward Euler: (m - m_prev)/dt
        terms = self.terms(state, {**drivers, "_q": q}, w, net, s)
        for name in ("thermal", "species"):
            if f"{name}.capacity" in terms:
                out[f"{name}.capacity"] = terms[f"{name}.capacity"]
                base = drivers.get(f"{name}.sources")
                extra = terms[f"{name}.extra"]
                out[f"{name}.sources"] = extra if base is None else base + extra
        return out
