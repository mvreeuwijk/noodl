"""Local per-node reactions applied after a transport step by operator splitting."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class Reaction:
    """A local, per-node nonlinear map applied to the state after a transport step."""

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class FirstOrderDecay(Reaction):
    """``x <- x * exp(-rate * dt)``; ``rate`` broadcasts against ``x``."""

    def __init__(self, rate: torch.Tensor | float) -> None:
        self.rate = torch.as_tensor(rate, dtype=torch.float64)

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        rate = self.rate.to(x.dtype)
        return x * torch.exp(-rate * dt)
