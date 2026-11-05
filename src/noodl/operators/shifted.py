"""ShiftedOperator: a base operator plus a diagonal shift, optionally scaled.

Two systems in the library are "an operator and a diagonal added to it". The potential
network's Jacobian with potential-dependent nodal sources is `A diag(g) A^T + diag(d)`; the
implicit and trapezoidal transport steps solve `S - alpha M`, with `S` the per-row storage
weights (the identity unless some nodes store nothing). One wrapper serves both.
"""

from __future__ import annotations

import torch

from noodl._broadcast import broadcast_shapes

Tensor = torch.Tensor


class ShiftedOperator:
    """`scale * base + diag(shift)`: a wrapped operator with a diagonal shift added, behind
    the same `LinearOperator` duck type `solvers.select.solve` reads (`matvec`, `rmatvec`,
    `diagonal`, `assemble`, `assemble_sparse`, `spd_certificate`, `shape`, `dtype`, `device`,
    `symmetric`).

    `shift` is a `(..., n)` tensor, or the float `1.0` for the plain identity (which skips
    the multiplication). `scale` is `None` for an unscaled base, so that the plain
    `base + diag(shift)` form runs no extra multiply. `symmetric` is the base's own: a
    diagonal never changes it.

    SPD certificate. A non-negative diagonal added to an SPD matrix is SPD, so for an
    unscaled shifted Laplacian the certificate is delegated to the base
    (`base.spd_certificate_with_shift`, which also credits strictly positive shift entries
    with grounding their node; a base without that method gives no certificate) and kept
    ONLY while every shift entry is non-negative; a
    negative entry could destroy definiteness, so the certificate is dropped rather than
    asserted (`solvers.select.solve` then routes to GMRES, which needs none). A scaled
    operator never certifies: `S - alpha M` with an advective `M` is not SPD.

    `shift_first` only fixes where the shift's entries go in `assemble_sparse`'s COO
    triplets: ahead of the base's (the transport step) or after them (the Jacobian).
    Duplicates are summed by the consumer, so the order sets the floating-point summation
    order of the diagonal entries and is kept per use.
    """

    def __init__(
        self,
        base,
        shift: Tensor | float,
        *,
        scale: float | None = None,
        shift_first: bool = False,
    ) -> None:
        self.base = base
        self.shift = shift
        self.scale = scale
        self.shift_first = shift_first
        self.symmetric = base.symmetric
        self._unit_shift = not torch.is_tensor(shift) and shift == 1.0
        batch = base.shape[:-2]
        if torch.is_tensor(shift):
            batch = broadcast_shapes(batch, shift.shape[:-1])
        self.shape = batch + base.shape[-2:]
        self.dtype = base.dtype
        self.device = base.device
        # Cache for `assemble_sparse`'s diagonal row/col index: `m` and the device never
        # change over the life of the object (one `torch.arange` fewer when assembled twice,
        # e.g. forward then backward).
        self._diag_idx: Tensor | None = None

    @classmethod
    def implicit_step(
        cls, base, alpha: float, storage: Tensor | None = None
    ) -> ShiftedOperator:
        """`S - alpha * base`, the system of one implicit transport step (`alpha = dt`) or
        one trapezoidal step (`alpha = dt / 2`).

        `S` is `diag(storage)`, the per-row storage weights: 1 where a row stores, 0 on the
        algebraic row of a storage-free node; `None` is the plain identity.
        """
        return cls(
            base, 1.0 if storage is None else storage, scale=-alpha, shift_first=True
        )

    def _shifted(self, x: Tensor) -> Tensor:
        return x if self._unit_shift else self.shift * x

    def _combine(self, base_part: Tensor, shift_part: Tensor | float) -> Tensor:
        if self.scale is None:
            return base_part + shift_part
        return shift_part + self.scale * base_part

    def matvec(self, x: Tensor) -> Tensor:
        return self._combine(self.base.matvec(x), self._shifted(x))

    def rmatvec(self, x: Tensor) -> Tensor:
        return self._combine(self.base.rmatvec(x), self._shifted(x))

    def diagonal(self) -> Tensor:
        return self._combine(self.base.diagonal(), 1.0 if self._unit_shift else self.shift)

    def assemble(self) -> Tensor:
        dense = self.base.assemble()
        if self._unit_shift:
            m = dense.shape[-1]
            shift = torch.eye(m, dtype=dense.dtype, device=dense.device)
        else:
            shift = torch.diag_embed(self.shift.expand(self.shape[:-1]))
        return self._combine(dense, shift)

    def assemble_sparse(self) -> tuple[Tensor, Tensor, Tensor] | None:
        """COO `(row, col, values)`: the base's own sparse form with the shift's diagonal
        entries added, `None` propagating unchanged if the base has none to give.

        Duplicates are how the shift and the base's diagonal entries combine: the COO
        consumer sums them, so appending one entry per diagonal position is the same
        mechanism the base's own duplicates use, not a special case.
        """
        triplet = self.base.assemble_sparse()
        if triplet is None:
            return None
        row, col, values = triplet
        m = self.shape[-1]
        if self._diag_idx is None or self._diag_idx.device != row.device:
            self._diag_idx = torch.arange(m, dtype=torch.int64, device=row.device)
        idx = self._diag_idx
        if self._unit_shift:
            shift = torch.ones(*values.shape[:-1], m, dtype=values.dtype, device=values.device)
        else:
            batch = broadcast_shapes(values.shape[:-1], self.shift.shape[:-1])
            shift = torch.broadcast_to(self.shift, batch + (m,)).to(values.dtype)
            values = values.expand(batch + values.shape[-1:])
        if self.scale is not None:
            values = self.scale * values
        if self.shift_first:
            return torch.cat([idx, row]), torch.cat([idx, col]), torch.cat([shift, values], dim=-1)
        return torch.cat([row, idx]), torch.cat([col, idx]), torch.cat([values, shift], dim=-1)

    def spd_certificate(self):
        if self.scale is not None or not torch.is_tensor(self.shift):
            return None
        certify = getattr(self.base, "spd_certificate_with_shift", None)
        if certify is None or bool((self.shift < 0).any()):
            return None
        return certify(self.shift)
