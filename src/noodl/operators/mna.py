"""MnaOperator: the Jacobian of a Modified Nodal Analysis system, held as COO triplets.

Unlike `GraphLaplacianOperator`, the operator of a layer with implicit constitutive laws is
neither symmetric nor definite (an ideal source's row has a zero diagonal), so it declares
`symmetric = False` and offers no SPD certificate: `solvers.select.solve` then accepts
`"direct"` (dense batched LU of `assemble()`), `"sparse_direct"` (SuperLU of the triplet,
per instance) and `"gmres"`. The index pattern is fixed by the layer's topology and shared
across the batch; only the values vary per instance. Duplicate `(row, col)` pairs are summed,
the contract of every COO consumer in use here. The values are real: a phasor (complex) layer
is solved in its real view, each complex unknown a (real, imaginary) pair, so its operator is
real as well.
"""

from __future__ import annotations

import torch

from noodl._broadcast import broadcast_shapes

Tensor = torch.Tensor


class MnaOperator:
    """`A[row[k], col[k]] += values[..., k]` for an (n, n) matrix per batch instance."""

    symmetric = False

    def __init__(self, row: Tensor, col: Tensor, values: Tensor, n: int) -> None:
        self.row = row
        self.col = col
        self.values = values
        self._n = n
        self.shape = values.shape[:-1] + (n, n)
        self.dtype = values.dtype
        self.device = values.device

    def _act(self, x: Tensor, out_idx: Tensor, in_idx: Tensor) -> Tensor:
        contrib = self.values * x[..., in_idx]
        batch = broadcast_shapes(self.values.shape[:-1], x.shape[:-1])
        contrib = contrib.expand(batch + contrib.shape[-1:])
        out = torch.zeros(batch + (self._n,), dtype=contrib.dtype, device=contrib.device)
        return out.index_add(-1, out_idx, contrib)

    def matvec(self, x: Tensor) -> Tensor:
        return self._act(x, self.row, self.col)

    def rmatvec(self, x: Tensor) -> Tensor:
        return self._act(x, self.col, self.row)

    def diagonal(self) -> Tensor:
        on = self.row == self.col
        out = torch.zeros(self.values.shape[:-1] + (self._n,), dtype=self.dtype, device=self.device)
        return out.index_add(-1, self.row[on], self.values[..., on])

    def assemble(self) -> Tensor:
        n = self._n
        out = torch.zeros(self.values.shape[:-1] + (n * n,), dtype=self.dtype, device=self.device)
        return out.index_add(-1, self.row * n + self.col, self.values).reshape(self.shape)

    def assemble_sparse(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.row, self.col, self.values

    def spd_certificate(self) -> None:
        return None
