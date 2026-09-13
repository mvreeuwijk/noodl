"""DenseOperator: the retained dense test oracle every sparse operator is checked against."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


class DenseOperator:
    """A `LinearOperator` backed by an explicit `(..., m, m)` tensor.

    Deliberately the simplest possible operator: `matvec` and `rmatvec` are plain batched
    matrix-vector products, `assemble()` returns the tensor itself, and `spd_certificate()`
    always returns `None` (a dense operator never certifies SPD-ness on its own; see
    `solvers.grounding.spd_certificate` for the graph-aware certificate used by
    `GraphLaplacianOperator`). `symmetric` is never inferred from `A`: it is whatever the
    caller declares at construction, so a nonsymmetric `A` constructed with
    `symmetric=True` is a caller error -- exactly what the symmetry tests in
    `tests/operators/test_base.py` are designed to catch.
    """

    def __init__(self, A: Tensor, *, symmetric: bool = False) -> None:
        if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
            raise ValueError(
                f"DenseOperator requires a square (..., m, m) tensor, got shape "
                f"{tuple(A.shape)}"
            )
        self.A = A
        self.shape = tuple(A.shape)
        self.dtype = A.dtype
        self.device = A.device
        self.symmetric = bool(symmetric)

    def matvec(self, x: Tensor) -> Tensor:
        return (self.A @ x.unsqueeze(-1)).squeeze(-1)

    def rmatvec(self, x: Tensor) -> Tensor:
        return (self.A.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)

    def diagonal(self) -> Tensor:
        return torch.diagonal(self.A, dim1=-2, dim2=-1)

    def assemble(self) -> Tensor | None:
        return self.A

    def spd_certificate(self) -> Tensor | None:
        return None
