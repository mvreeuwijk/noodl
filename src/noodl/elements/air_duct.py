"""ContamX's duct airflow laws: a Darcy-Weisbach duct segment and a duct terminal.

Both use the density and the viscosity of the air ENTERING the element: the upstream node,
chosen by the sign of dp. Air viscosity is ContamX's linear law in temperature
(`air_viscosity`). These are the laws ContamX 3.4.1.7 evaluates (measured); `Duct` follows
TN 1887's own printed Colebrook form, which differs by about 0.1 %.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from noodl.elements.base import Element
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor

MU_REF = 1.81625e-5          # Pa s, air at T_MU_REF
T_MU_REF = 293.15            # K
MU_SLOPE = 4.92859e-8        # Pa s / K
LAMINAR_RE = 2300.0          # f = 64 / Re at and below
TURBULENT_RE = 3500.0        # Colebrook at and above; f linear in Re in between
TERMINAL_RE = 30.0           # a terminal is linear below F = 30 mu sqrt(A)
MIN_OPENING = 1e-6
_LN10 = math.log(10.0)


def air_viscosity(T) -> Tensor:
    """Dynamic viscosity of air (Pa s) at temperature `T` (K): ContamX's linear law,
    1.81625e-5 + 4.92859e-8 (T - 293.15), measured between 253 and 333 K."""
    if not isinstance(T, Tensor) or not T.is_floating_point():
        T = torch.as_tensor(T, dtype=torch.float64)
    return MU_REF + MU_SLOPE * (T - T_MU_REF)


def _endpoints(src, tgt, where: str) -> tuple[Tensor, Tensor]:
    s = torch.as_tensor(src).to(torch.long).reshape(-1)
    t = torch.as_tensor(tgt).to(torch.long).reshape(-1)
    if s.shape != t.shape:
        raise ValueError(f"{where}: src and tgt must give one node position per edge, got "
                         f"{tuple(s.shape)} and {tuple(t.shape)}")
    return s, t


def _upstream(values: Tensor, src: Tensor, tgt: Tensor, forward: Tensor) -> Tensor:
    """Per edge, the full-node `values` at the node the air comes from (`src` where
    `forward`)."""
    return torch.where(forward, values[..., src], values[..., tgt])


def _colebrook(Re: Tensor, relative_roughness: Tensor, n_iter: int = 6) -> tuple[Tensor, Tensor]:
    """(f, df/dRe) from 1/sqrt(f) = -2 log10(r / 3.7 + 2.51 / (Re sqrt(f))).

    Newton on g = 1/sqrt(f) from the Swamee-Jain value converges to float64 roundoff in four
    iterations. The derivative is the implicit one of the converged equation."""
    a = relative_roughness / 3.7
    b = 2.51 / Re
    g = -2.0 * torch.log10(a + 5.74 / Re**0.9)
    c = 2.0 / _LN10
    for _ in range(n_iter):
        inner = a + b * g
        g = g - (g + 2.0 * torch.log10(inner)) / (1.0 + c * b / inner)
    dg = c * g * b / (Re * (a + b * g + c * b))
    return 1.0 / (g * g), -2.0 * dg / (g * g * g)


def _friction(Re: Tensor, rel: Tensor) -> tuple[Tensor, Tensor]:
    """ContamX's friction factor above Re 2300 and its derivative in Re: linear in Re up to
    Re 3500, Colebrook from there. (Below 2300 the laminar law is used directly.)"""
    f_edge = 64.0 / LAMINAR_RE
    f_3500, _ = _colebrook(torch.full_like(Re, TURBULENT_RE), rel)
    slope_mid = (f_3500 - f_edge) / (TURBULENT_RE - LAMINAR_RE)
    Re_mid = Re.clamp(min=LAMINAR_RE, max=TURBULENT_RE)
    f_mid = f_edge + (Re_mid - LAMINAR_RE) * slope_mid
    f_turb, df_turb = _colebrook(Re.clamp(min=TURBULENT_RE), rel)
    turbulent = Re >= TURBULENT_RE
    return torch.where(turbulent, f_turb, f_mid), torch.where(turbulent, df_turb, slope_mid)


def _duct_loss(F: Tensor, rho: Tensor, mu: Tensor, L: Tensor, D: Tensor, eps: Tensor,
               C: Tensor, A: Tensor) -> Tensor:
    """dp (Pa) of a duct carrying F >= 0 kg/s, ContamX's friction factor."""
    Re = F * D / (mu * A)
    f, _ = _friction(Re, eps / D)
    dynamic = F * F / (2.0 * rho * A * A)
    laminar = _laminar_resistance(rho, mu, L, D, A) * F
    return torch.where(Re <= LAMINAR_RE, laminar, f * L / D * dynamic) + C * dynamic


def _duct_loss_slope(F: Tensor, rho: Tensor, mu: Tensor, L: Tensor, D: Tensor, eps: Tensor,
                     C: Tensor, A: Tensor) -> Tensor:
    """d dp / d F of `_duct_loss` at F >= 0, in closed form."""
    Re = F * D / (mu * A)
    f, df = _friction(Re, eps / D)
    per_flow = F / (rho * A * A)
    laminar = _laminar_resistance(rho, mu, L, D, A)
    turbulent = L / D * (f * per_flow + 0.5 * F * per_flow * df * D / (mu * A))
    return torch.where(Re <= LAMINAR_RE, laminar, turbulent) + C * per_flow


def _laminar_resistance(rho, mu, L, D, A) -> Tensor:
    """a in dp = a F, the laminar law: no regime loses less (64 / Re is the least friction
    up to Re 2300, and above it the friction is at least 64 / Re)."""
    return 32.0 * mu * L / (rho * A * D * D)


def _least_quadratic_loss(rho, L, D, eps, C, A) -> Tensor:
    """k in dp >= k F^2 for every regime: the friction never falls below the lesser of
    64 / 2300 and the fully rough Colebrook value (2 log10(3.7 D / eps))^-2, the limit its
    friction decreases to as Re grows."""
    f_rough = (2.0 * torch.log10(3.7 * D / eps)) ** -2
    f_least = torch.minimum(f_rough, torch.full_like(f_rough, 64.0 / LAMINAR_RE))
    return (f_least * L / D + C) / (2.0 * rho * A * A)


def _scaled_residual(x, scale, target, rho, mu, L, D, eps, C, A):
    return _duct_loss(scale * x, rho, mu, L, D, eps, C, A) - target


class _UpstreamAir(Element):
    """Shared by the duct laws: per-edge upstream density and viscosity from the drivers."""

    def _set_endpoints(self, src, tgt, rho_key: str, temperature_key: str | None) -> None:
        s, t = _endpoints(src, tgt, f"{type(self).__name__} (kind {self.kind!r})")
        self.register_buffer("src", s)
        self.register_buffer("tgt", t)
        self.rho_key = str(rho_key)
        self.temperature_key = temperature_key

    def _air(self, forward: Tensor, drivers: Mapping[str, Tensor] | None,
             dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """(rho, mu) of the air entering each edge, upstream `src` where `forward`."""
        name = f"{type(self).__name__} (kind {self.kind!r})"
        if drivers is None or self.rho_key not in drivers:
            raise KeyError(f"{name}: driver {self.rho_key!r} not found; it needs the "
                           f"full-node air density")
        rho = _upstream(torch.as_tensor(drivers[self.rho_key]).to(dtype),
                        self.src, self.tgt, forward)
        if self.temperature_key is None:
            return rho, air_viscosity(torch.full_like(rho, T_MU_REF))
        if self.temperature_key not in drivers:
            raise KeyError(f"{name}: driver {self.temperature_key!r} not found; it needs "
                           f"the full-node air temperature for the viscosity")
        T = torch.as_tensor(drivers[self.temperature_key]).to(dtype)
        return rho, air_viscosity(_upstream(T, self.src, self.tgt, forward))


class AirDuct(_UpstreamAir):
    """A duct segment, F (kg/s) from dp (Pa): Darcy-Weisbach with ContamX's friction factor.

        dp = (f L / D + loss) F^2 / (2 rho A^2),   Re = F D / (mu A)
        f = 64 / Re                                                    Re <= 2300
        1 / sqrt(f) = -2 log10(eps / (3.7 D) + 2.51 / (Re sqrt(f)))    Re >= 3500
        f linear in Re in between

    `length` L, hydraulic `diameter` D and `roughness` eps in m; `loss` the sum of the
    segment's dynamic loss coefficients (dimensionless, acting in every regime); `area` A in
    m2, the circle of diameter D (following D) when not given. One value per edge or one for
    all. rho and mu are the upstream node's: `drivers[rho_key]` (full node order), and
    `air_viscosity(drivers[temperature_key])`, or the viscosity at 293.15 K when
    `temperature_key` is None. `src`/`tgt` are the edges' node positions
    (`Network.endpoints(kind)`). `pressure_drop(F, drivers)` is the law itself, dp from F.

    The law is inverted for F with `solve_monotone`. dp grows monotonically with F, and no
    regime loses less than the laminar law `a F`, `a = 32 mu L / (rho A D^2)`, or than
    `k F^2` with the least friction factor the law reaches (the fully rough Colebrook value
    or 64 / 2300): twice the lesser of `dp / a` and `sqrt(dp / k)` brackets the root strictly,
    and the solve runs on F in units of that bracket, so its tolerance is relative. At
    dp = 0 the flow is exactly 0 and the slope is `1 / a`.

    The roughness bound (below 3.7 diameters) is checked at construction only: with
    `learnable=True` a later update can leave it unchecked, so keep a learned roughness inside it.
    """

    def __init__(self, length, diameter, roughness, loss=0.0, *, area=None, src, tgt,
                 rho_key: str = "rho", temperature_key: str | None = None,
                 kind: str = "duct", learnable: bool = False) -> None:
        super().__init__(kind)
        where = f"AirDuct (kind {kind!r})"
        for label, value in (("length", length), ("diameter", diameter),
                             ("roughness", roughness)):
            if not bool((torch.as_tensor(value, dtype=torch.float64) > 0).all()):
                raise ValueError(f"{where}: {label} must be strictly positive, got {value!r}")
        if not bool((torch.as_tensor(loss, dtype=torch.float64) >= 0).all()):
            raise ValueError(f"{where}: loss must be non-negative, got {loss!r}")
        if not bool((torch.as_tensor(roughness, dtype=torch.float64)
                     < 3.7 * torch.as_tensor(diameter, dtype=torch.float64)).all()):
            raise ValueError(f"{where}: roughness must be below 3.7 diameters (the Colebrook "
                             f"equation has no solution beyond), got roughness {roughness!r} "
                             f"and diameter {diameter!r}")
        if area is not None and not bool((torch.as_tensor(area, dtype=torch.float64) > 0)
                                         .all()):
            raise ValueError(f"{where}: area must be strictly positive, got {area!r}")
        self.length = self._param(length, learnable)
        self.diameter = self._param(diameter, learnable)
        self.roughness = self._param(roughness, learnable)
        self.loss = self._param(loss, learnable)
        self.area = None if area is None else self._param(area, learnable)
        self._set_endpoints(src, tgt, rho_key, temperature_key)

    def _geometry(self, dtype: torch.dtype) -> list[Tensor]:
        L, D, eps, C = (p.to(dtype) for p in (self.length, self.diameter, self.roughness,
                                              self.loss))
        A = math.pi * D * D / 4.0 if self.area is None else self.area.to(dtype)
        return [L, D, eps, C, A]

    def pressure_drop(self, F: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """dp (Pa) at mass flow F (kg/s), the air entering from `src` where F >= 0."""
        forward = F >= 0
        rho, mu = self._air(forward, drivers, F.dtype)
        size = torch.where(forward, F, -F)
        dp = _duct_loss(size, rho, mu, *self._geometry(F.dtype))
        return torch.where(forward, dp, -dp)

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        forward = dp >= 0
        rho, mu = self._air(forward, drivers, dp.dtype)
        # `where`, not `abs`: d|dp|/ddp is 0 at dp = 0, which would zero the slope there.
        target = torch.where(forward, dp, -dp)
        parts = torch.broadcast_tensors(target, rho, mu, *self._geometry(dp.dtype))
        target, rho, mu, L, D, eps, C, A = parts
        positive = target > 0
        # The bound is built from a safe target: sqrt(target) at 0 has an infinite derivative,
        # which `where` would turn into NaN in the backward. At dp = 0 the bracket is [0, 0],
        # so F = 0 exactly; the unit scale there only keeps the implicit derivative finite
        # (it multiplies x = 0, and dF/ddp comes out as 1 / a).
        safe = torch.where(positive, target, torch.ones_like(target))
        bound = 2.0 * torch.minimum(safe / _laminar_resistance(rho, mu, L, D, A),
                                    torch.sqrt(safe / _least_quadratic_loss(rho, L, D,
                                                                            eps, C, A)))
        scale = torch.where(positive, bound, torch.ones_like(bound))
        hi = positive.to(dp.dtype)
        x = solve_monotone(_scaled_residual, torch.zeros_like(hi), hi, scale, target, rho, mu,
                           L, D, eps, C, A, tol=4.0 * torch.finfo(dp.dtype).eps)
        F = scale * x
        return torch.where(forward, F, -F)

    def dflow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        F = self.flow(dp, drivers)
        forward = dp >= 0
        rho, mu = self._air(forward, drivers, dp.dtype)
        size = torch.where(forward, F, -F)
        return 1.0 / _duct_loss_slope(size, rho, mu, *self._geometry(dp.dtype))

    def linear_init(self, drivers: Mapping[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        dtype = self._dtype()
        forward = torch.ones(self.src.numel(), dtype=torch.bool)
        rho, mu = self._air(forward, drivers, dtype)
        L, D, _, _, A = self._geometry(dtype)
        k = rho * A * D * D / (32.0 * mu * L)
        return torch.zeros_like(k), k


class DuctTerminal(_UpstreamAir):
    """A duct terminal, the zero-length duct between a junction and its zone, F (kg/s)
    from dp (Pa), as ContamX 3.4.1.7 computes it (measured):

        dp = Ce F^2 / (2 rho A^2)            |F| >= F_t = 30 mu sqrt(A)
        dp = Ce F_t F / (2 rho A^2)          below F_t (the line through that point)
        Ce = loss + (1 - S) / S,             S = clamp(drivers[opening_key], 1e-6, 1)

    `area` A (m2) is the duct's area at the terminal and `loss` its loss coefficient (CONTAM's
    terminal plus balance coefficients); the opening S is one value per edge, and S = 1 when
    `opening_key` is None. A schedule or a control of 0 (or below) nearly closes the terminal
    (Ce ~ 1e6), and nothing opens it beyond S = 1. rho and mu are the upstream node's, as for
    `AirDuct`."""

    def __init__(self, area, loss, *, src, tgt, rho_key: str = "rho",
                 temperature_key: str | None = None, opening_key: str | None = None,
                 kind: str = "duct_terminal", learnable: bool = False) -> None:
        super().__init__(kind)
        where = f"DuctTerminal (kind {kind!r})"
        if not bool((torch.as_tensor(area, dtype=torch.float64) > 0).all()):
            raise ValueError(f"{where}: area must be strictly positive, got {area!r}")
        if not bool((torch.as_tensor(loss, dtype=torch.float64) > 0).all()):
            raise ValueError(f"{where}: loss must be strictly positive, got {loss!r}")
        self.area = self._param(area, learnable)
        self.loss = self._param(loss, learnable)
        self.opening_key = opening_key
        self._set_endpoints(src, tgt, rho_key, temperature_key)

    def _law(self, forward: Tensor, drivers, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """(c, F_t): dp = c F^2 above F_t."""
        rho, mu = self._air(forward, drivers, dtype)
        A, Ce = self.area.to(dtype), self.loss.to(dtype)
        if self.opening_key is not None:
            if drivers is None or self.opening_key not in drivers:
                raise KeyError(f"DuctTerminal (kind {self.kind!r}): driver "
                               f"{self.opening_key!r} not found; it needs the opening, one "
                               f"value per edge")
            S = torch.as_tensor(drivers[self.opening_key]).to(dtype)
            S = S.clamp(min=MIN_OPENING, max=1.0)
            Ce = Ce + (1.0 - S) / S
        return Ce / (2.0 * rho * A * A), TERMINAL_RE * mu * torch.sqrt(A)

    def _regime(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(c, F_t, above, safe): `above` where |dp| is past the laminar switch, and `safe`
        the |dp| there, or the switch pressure elsewhere, so that the square roots of the
        quadratic branch stay finite (value and gradient) where it is not selected."""
        c, F_t = self._law(dp >= 0, drivers, dp.dtype)
        dp_t = c * F_t * F_t
        size = torch.where(dp >= 0, dp, -dp)
        above = size > dp_t
        return c, F_t, above, torch.where(above, size, dp_t)

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        c, F_t, above, safe = self._regime(dp, drivers)
        quadratic = torch.sqrt(safe / c)
        return torch.where(above, torch.where(dp >= 0, quadratic, -quadratic),
                           dp / (c * F_t))

    def dflow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        c, F_t, above, safe = self._regime(dp, drivers)
        return torch.where(above, 0.5 / torch.sqrt(c * safe), 1.0 / (c * F_t))

    def linear_init(self, drivers: Mapping[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        forward = torch.ones(self.src.numel(), dtype=torch.bool)
        c, F_t = self._law(forward, drivers, self._dtype())
        k = 1.0 / (c * F_t)
        return torch.zeros_like(k), k
