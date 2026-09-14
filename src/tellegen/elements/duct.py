"""Colebrook duct element (CONTAM TN 1887r1 section 8.3.3, eq. 50-52).

    F = sqrt( 2 rho A^2 |dp| / (f L/D + sum_C) ),
    1/sqrt(f) = 1.14 - 2 log10(eps/D) - 2 log10( 1 + 9.3 / (Re (eps/D) sqrt(f)) ),
    Re = F D / (mu A).

CONTAM iterates g = f^(-1/2) with  g* = g - [g - a + c ln(1 + g b)] / [1 + c b / (1 + g b)],
a = 1.14 - c ln(eps/D), b = 9.3 / (Re eps/D), c = 2 log10(e), from g = a (fully rough), and
reports 2-3 iterations suffice. Here the update is UNROLLED `n_iter` times with `Re`
re-evaluated from the current `F` on every pass, so autograd differentiates through it;
`dflow` is the base class's autograd default (CONTAM itself uses a secant there).

Laminar regime (TN 1887r1 p. 270): below the transition Reynolds number `Re_t` (default 2000)
the law is the straight line through the origin that meets the turbulent curve at
F_t = mu Re_t A / D, i.e. F = F_t dp / dp_t. `torch.where` selects; the unselected turbulent
branch is fed `dp_t` instead of |dp| so it never sees dp = 0 (the PowerLaw idiom).

`rho` and `mu` are fixed at construction (milestone 2 plan, Task 4 deviation): a per-step
density correction from `drivers` would need the transition pressure `dp_t` recomputed on
every call, since `dp_t` depends on `rho`. The building application passes `rho_0`.

Dtype: every quantity this element derives internally (`A` when not given explicitly, the
Colebrook constants `a`/`b`, and the transition values from `_transition()`) is built from
the already-registered `L`/`D`/`eps`/`sum_C` parameters (or, for `A`, cast with
`Element._dtype()`, which reports those parameters' dtype), never with
`torch.get_default_dtype()`. This matters because `Element._param` casts a plain Python
float with `torch.get_default_dtype()` (float32 in this repo): a caller who wants float64
precision passes float64 tensors for `L`/`D`/`eps`/`sum_C`, and every derived quantity then
naturally inherits that float64 dtype instead of silently downgrading to float32.
"""

from __future__ import annotations

import math

import torch

from tellegen.elements.base import Element

Tensor = torch.Tensor
_C = 2.0 * math.log10(math.e)


class Duct(Element):
    """Colebrook duct: F(dp) with friction from the Colebrook equation, laminar below Re_t."""

    def __init__(
        self,
        L,
        D,
        eps,
        sum_C=0.0,
        *,
        A=None,
        Re_t: float = 2000.0,
        rho: float = 1.2041,
        mu: float = 1.81625e-5,
        n_iter: int = 4,
        kind: str = "duct",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        for name, value in (("L", L), ("D", D), ("eps", eps)):
            if not bool((torch.as_tensor(value, dtype=torch.float64) > 0).all()):
                raise ValueError(f"Duct: {name} must be strictly positive, got {value!r}")
        if int(n_iter) < 1:
            raise ValueError(f"Duct: n_iter must be >= 1, got {n_iter!r}")
        if float(Re_t) <= 0 or float(rho) <= 0 or float(mu) <= 0:
            raise ValueError(
                f"Duct: Re_t, rho and mu must be positive; got {Re_t}, {rho}, {mu}"
            )
        self.L = self._param(L, learnable)
        self.D = self._param(D, learnable)
        self.eps = self._param(eps, learnable)
        self.sum_C = self._param(sum_C, learnable)
        if A is None:
            A = math.pi * torch.as_tensor(D, dtype=self._dtype()) ** 2 / 4.0
        self.A = self._param(A, False)
        self.Re_t = float(Re_t)
        self.rho = float(rho)
        self.mu = float(mu)
        self.n_iter = int(n_iter)

    # ----------------------------------------------------------------- friction
    def _a(self) -> Tensor:
        return 1.14 - _C * torch.log(self.eps / self.D)

    def _update_g(self, g: Tensor, Re: Tensor) -> Tensor:
        b = 9.3 / (Re * (self.eps / self.D))
        gb = 1.0 + g * b
        return g - (g - self._a() + _C * torch.log(gb)) / (1.0 + _C * b / gb)

    def _F_of_g(self, dp_abs: Tensor, g: Tensor) -> Tensor:
        f = 1.0 / g**2
        return torch.sqrt(
            2.0 * self.rho * self.A**2 * dp_abs / (f * self.L / self.D + self.sum_C)
        )

    def _turbulent(self, dp_abs: Tensor) -> Tensor:
        """|F| for |dp| >= dp_t: unrolled Colebrook fixed point, Re from the current F."""
        g = self._a() * torch.ones_like(dp_abs)
        for _ in range(self.n_iter):
            F = self._F_of_g(dp_abs, g)
            Re = F * self.D / (self.mu * self.A)
            g = self._update_g(g, Re)
        return self._F_of_g(dp_abs, g)

    def _transition(self) -> tuple[Tensor, Tensor]:
        """(F_t, dp_t): the turbulent flow and pressure drop at Re = Re_t."""
        F_t = self.mu * self.Re_t * self.A / self.D
        g = self._a()
        Re_t = torch.as_tensor(self.Re_t, dtype=g.dtype)
        for _ in range(self.n_iter):
            g = self._update_g(g, Re_t)
        f_t = 1.0 / g**2
        dp_t = F_t**2 * (f_t * self.L / self.D + self.sum_C) / (2.0 * self.rho * self.A**2)
        return F_t, dp_t

    # ------------------------------------------------------------------ Element
    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        F_t, dp_t = self._transition()
        mask = dp.abs() < dp_t
        dp_safe = torch.where(mask, dp_t * torch.ones_like(dp), dp.abs())
        turbulent = torch.sign(dp) * self._turbulent(dp_safe)
        laminar = F_t * dp / dp_t
        return torch.where(mask, laminar, turbulent)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        F_t, dp_t = self._transition()
        k = F_t / dp_t
        return torch.zeros_like(k), k
