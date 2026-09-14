"""Drive protocol: additive potential-difference terms added to an element's dp.

A drive is a function of the `drivers` mapping ONLY (milestone 2 spec section 4.1): it never
sees the potential `phi`. The Newton Jacobian `A_I diag(g') A_I^T` is therefore exact and stays
symmetric, which is what the SPD certificate and the conjugate-gradient path rely on. Anything a
drive needs that depends on the state -- zone densities for a stack term -- is computed by a
`Model` closure before the solve and written into `drivers`. CONTAM does the same: densities
from barometric pressure and temperature before the iteration, no density-pressure term in the
Jacobian (TN 1887r1 section 3.18, eq. 8-9).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Drive(Protocol):
    """A drive must read every differentiable quantity from `drivers`, never own one.

    `PotentialFlowLayer.solve(differentiable=True)` only threads gradients through
    `Function.apply` for values reachable via `phi_boundary`, `sources`, `drivers`, and each
    Element's own registered `nn.Parameter`s. A Drive instance is captured by closure inside
    the differentiable solve, so any tensor it holds with `requires_grad=True` is invisible to
    the backward pass; `PotentialFlowLayer._check_no_unreachable_differentiable_tensors`
    raises on that. Geometry and index tensors (no grad) may be held as attributes.
    """

    kind: str

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


def check_drive_signature(drive, *, where: str) -> None:
    """Raise `TypeError` unless `drive` is callable as `drive(drivers)`.

    Loud migration guard for the pre-milestone-2 `(phi, drivers)` form: a two-argument drive
    would otherwise fail deep inside `PotentialFlowLayer.dp` with an unattributed TypeError.
    """
    try:
        sig = inspect.signature(drive.__call__)
    except (TypeError, ValueError):  # builtins without a signature: nothing to check
        return
    positional = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        raise TypeError(
            f"{where}: drive {type(drive).__name__} (kind {getattr(drive, 'kind', '?')!r}) "
            f"must be callable as drive(drivers) -- a Drive is a function of the drivers "
            f"mapping only (milestone 2 spec section 4.1); got parameters {positional}. A "
            f"drive that used to take (phi, drivers) must drop phi."
        )


class ConstantDrive:
    """A drive that reads a pre-computed, already batched value from `drivers`."""

    def __init__(self, kind: str, key: str) -> None:
        self.kind = kind
        self.key = key

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            return drivers[self.key]
        except KeyError as exc:
            raise KeyError(f"driver {self.key!r} not found") from exc
