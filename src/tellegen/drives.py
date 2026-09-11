"""Drive protocol: additive potential-difference terms added to an element's dp."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Drive(Protocol):
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
