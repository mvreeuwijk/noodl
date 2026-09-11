"""Linear conductance branch element: q = g dp (conduction, SIRANE-style exchange)."""

from __future__ import annotations

import torch

from tellegen.elements.base import Element

Tensor = torch.Tensor


class Conductance(Element):
    """q = g dp, the linear branch law used for thermal conduction and passive exchange."""

    def __init__(self, g, *, kind: str = "conduction", learnable: bool = False) -> None:
        super().__init__(kind)
        self.g = self._param(g, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self.g * dp

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return self.g * torch.ones_like(dp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(self.g), self.g
