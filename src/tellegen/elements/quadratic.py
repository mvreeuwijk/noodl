"""Quadratic-drag branch element: dp = a q + b |q| q, inverted for q given dp."""

from __future__ import annotations

import torch

from tellegen.elements.base import Element

Tensor = torch.Tensor


class Quadratic(Element):
    """q = sign(dp) * (sqrt(a^2 + 4 b |dp|) - a) / (2 b), the inverse of dp = a q + b |q| q."""

    def __init__(self, a, b, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        # Store a and b with consistent precision (convert to tensors first)
        a_tensor = torch.as_tensor(a)
        b_tensor = torch.as_tensor(b)
        self.a = self._param(a_tensor, learnable)
        self.b = self._param(b_tensor, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        a, b = self.a, self.b
        # Promote to float64 for computation to avoid precision loss
        a_hp = a.to(torch.float64)
        b_hp = b.to(torch.float64)
        dp_hp = dp.to(torch.float64)
        disc = a_hp**2 + 4 * b_hp * dp_hp.abs()
        result_hp = (
            torch.sign(dp_hp) * (torch.sqrt(disc) - a_hp) / (2 * b_hp)
        )
        return result_hp.to(dp.dtype)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        a, b = self.a, self.b
        # Promote to float64 for computation to avoid precision loss
        a_hp = a.to(torch.float64)
        b_hp = b.to(torch.float64)
        dp_hp = dp.to(torch.float64)
        disc = a_hp**2 + 4 * b_hp * dp_hp.abs()
        result_hp = 1.0 / torch.sqrt(disc)
        return result_hp.to(dp.dtype)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(self.a), 1.0 / self.a
