"""Quadratic-drag branch element: dp = a q + b |q| q, inverted for q given dp."""

from __future__ import annotations

import torch

from tellegen.elements.base import Element

Tensor = torch.Tensor


class Quadratic(Element):
    """q = sign(dp) * 2|dp| / (sqrt(a^2 + 4 b |dp|) + a).

    Inverts the quadratic-drag law dp = a q + b |q| q.

    Precondition: a > 0 and b > 0. Violations produce nan in both forward and backward:
    with a = 0 and dp = 0, the conjugate form evaluates to 0 / 0 = nan in the forward pass.
    With a <= 0, the discriminant a^2 + 4 b |dp| can vanish (when b |dp| = -a^2 / 4), and
    sqrt diverges in gradients. With a = 0 and learnable=True, an optimiser can drive the
    batch toward dp=0 where the forward nan and gradient poisoning propagate to the shared
    a parameter across the whole batch (a.grad = nan).
    """

    def __init__(self, a, b, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        self.a = self._param(a, learnable)
        self.b = self._param(b, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        a, b = self.a, self.b
        # Avoid catastrophic cancellation: use algebraically equivalent form
        # q = sign(dp) * 2|dp| / (sqrt(a^2 + 4 b |dp|) + a)
        # This is identical to sign(dp) * (sqrt(a^2 + 4 b |dp|) - a) / (2 b)
        # (conjugate multiplication), but avoids subtraction of near-equal terms.
        disc = a**2 + 4 * b * dp.abs()
        return torch.sign(dp) * 2 * dp.abs() / (torch.sqrt(disc) + a)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        a, b = self.a, self.b
        disc = a**2 + 4 * b * dp.abs()
        return 1.0 / torch.sqrt(disc)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(self.a), 1.0 / self.a
