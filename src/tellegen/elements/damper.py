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

from tellegen.elements.base import Element
from tellegen.elements.powerlaw import PowerLaw

Tensor = torch.Tensor


class Damper(Element):
    """Separate power-law coefficient and exponent for each flow direction."""

    def __init__(
        self,
        C_pos,
        n_pos,
        C_neg,
        n_neg,
        *,
        dp_transition: float = 1e-3,
        kind: str = "damper",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.pos = PowerLaw(
            C_pos, n_pos, dp_transition=dp_transition, kind=kind, learnable=learnable
        )
        self.neg = PowerLaw(
            C_neg, n_neg, dp_transition=dp_transition, kind=kind, learnable=learnable
        )

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.where(dp >= 0, self.pos.flow(dp, drivers), self.neg.flow(dp, drivers))

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.where(dp >= 0, self.pos.dflow(dp, drivers), self.neg.dflow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        _, k_pos = self.pos.linear_init(drivers)
        _, k_neg = self.neg.linear_init(drivers)
        k = 0.5 * (k_pos + k_neg)
        return torch.zeros_like(k), k
