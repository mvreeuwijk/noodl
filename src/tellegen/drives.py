"""Drive protocol: additive potential-difference terms added to an element's dp."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Drive(Protocol):
    """A drive must read every differentiable quantity from `drivers`, never own one.

    `PotentialFlowLayer.solve(differentiable=True)` only threads gradients through
    `Function.apply` for values reachable via `phi_boundary`, `sources`, `drivers`, and each
    Element's own registered `nn.Parameter`s. A Drive instance itself is captured by closure
    inside the differentiable solve (it is not, and cannot be, passed through
    `Function.apply`), so any tensor a Drive holds as its own instance attribute -- e.g. a
    learnable coefficient stored at construction time instead of read from `drivers` on every
    call -- is invisible to the backward pass: its gradient would be silently absent rather
    than raising, if `solve` did not explicitly guard against it (it does; see
    `PotentialFlowLayer._check_no_unreachable_differentiable_tensors`). A drive that needs a
    learnable coefficient must have its caller pass that coefficient's current value through
    the `drivers` mapping on every call, not close over it.
    """

    kind: str

    def __call__(self, phi: torch.Tensor, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


class ConstantDrive:
    """A drive that reads a pre-computed, already batched value from `drivers`."""

    def __init__(self, kind: str, key: str) -> None:
        self.kind = kind
        self.key = key

    def __call__(self, phi, drivers):
        try:
            return drivers[self.key]
        except KeyError as exc:
            raise KeyError(f"driver {self.key!r} not found") from exc
