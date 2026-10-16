"""Backdraft damper: CONTAM's PL_BDF/PL_BDQ -- one (C, n) power law per sign of dp.

    F =  C_pos dp^n_pos        dp >= 0
    F = -C_neg (-dp)^n_neg     dp <  0

with `PowerLaw`'s laminar linearisation below `dp_transition` on each side. Two `PowerLaw`
submodules do the work, so parameters register as `pos.C`, `pos.n`, `neg.C`, `neg.n` and
`torch.func.functional_call` substitution in the differentiable solve reaches them. Both
branches are evaluated everywhere and `torch.where` selects; each branch is a `PowerLaw`,
which is finite (value and gradient) at every dp, so no extra guarding is needed here.
"""

from __future__ import annotations

import torch

from noodl.elements.base import ConstitutiveLaw
from noodl.elements.powerlaw import PowerLaw
from noodl.elements.upstream import UpstreamDensityPowerLaw

Tensor = torch.Tensor


class Damper(ConstitutiveLaw):
    """Separate power-law coefficient and exponent for each flow direction: `C_pos`/`n_pos`
    for dp > 0 and `C_neg`/`n_neg` for dp < 0, each as in `PowerLaw` (`C` in the flow unit
    per Pa^n, `n` dimensionless).

    With `src`/`tgt` (node positions) each side's coefficient follows the entering air's
    density, `(rho_up / rho_ref) ** m`, as `UpstreamDensityPowerLaw`; ContamX's duct dampers
    do this (m = 1 for the volume form, 0 for the mass form).
    """

    def __init__(
        self,
        C_pos,
        n_pos,
        C_neg,
        n_neg,
        *,
        dp_transition: float = 1e-3,
        src=None,
        tgt=None,
        m=0.0,
        rho_ref: float = 1.2041,
        rho_key: str = "rho",
        kind: str = "damper",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        if (src is None) != (tgt is None):
            raise ValueError(f"Damper (kind {kind!r}): give both src and tgt (the entering "
                             f"air's density) or neither")
        if src is None:
            self.pos = PowerLaw(
                C_pos, n_pos, dp_transition=dp_transition, kind=kind, learnable=learnable
            )
            self.neg = PowerLaw(
                C_neg, n_neg, dp_transition=dp_transition, kind=kind, learnable=learnable
            )
        else:
            law = dict(src=src, tgt=tgt, m=m, rho_ref=rho_ref, rho_key=rho_key,
                       dp_transition=dp_transition, kind=kind, learnable=learnable)
            self.pos = UpstreamDensityPowerLaw(C_pos, n_pos, **law)
            self.neg = UpstreamDensityPowerLaw(C_neg, n_neg, **law)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.where(dp >= 0, self.pos.flow(dp, drivers), self.neg.flow(dp, drivers))

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.where(dp >= 0, self.pos.dflow(dp, drivers), self.neg.dflow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        _, k_pos = self.pos.linear_init(drivers)
        _, k_neg = self.neg.linear_init(drivers)
        k = 0.5 * (k_pos + k_neg)
        return torch.zeros_like(k), k
