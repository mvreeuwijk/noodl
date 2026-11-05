"""Reactions that exist for the tests only."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.layers.reaction import Reaction


class FirstOrderDecay(Reaction):
    """``x <- x * exp(-rate * dt)``; ``rate`` broadcasts against ``x``.

    A generic, driver-free reaction for exercising the operator-split mechanism. Real models
    express first-order decay as the transport layer's `removal`.
    """

    def __init__(self, rate: torch.Tensor | float) -> None:
        self.rate = torch.as_tensor(rate, dtype=torch.float64)

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        rate = self.rate.to(x.dtype)
        return x * torch.exp(-rate * dt)
