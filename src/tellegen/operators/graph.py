"""GraphLaplacianOperator: A_I diag(g) A_I^T, represented WITHOUT ever forming an n x n
matrix in the hot (matvec) path.

`src`/`tgt` are shared endpoint index arrays over one node-index space of size
`n = n_interior + n_boundary`; `interior_of_node`/`boundary_mask` split that space into the
operator's own compact interior domain (size `n_interior`, what `matvec`/`rmatvec` actually
operate on) and the boundary nodes, which this operator always treats as held at potential 0
(a nonzero boundary potential is folded into the right-hand side by the caller, exactly as
`PotentialFlowLayer.residual` folds `phi_boundary` into `lhs - s_I` today).

Every method below is built from two primitives, both O(edges) and loop-free: a GATHER at the
edge endpoints (`torch.gather`) and a SCATTER-ADD back to nodes (`Tensor.scatter_add_`), which
correctly accumulates multiple edges landing on the same node (unlike plain fancy-index
assignment, which would overwrite rather than sum). This mirrors `Network.difference_ep`/
`accumulate` (Task 2) but is reimplemented here directly over the operator's own
`src`/`tgt`/`interior_of_node`, since the operator holds no `Network` reference.
"""

from __future__ import annotations

import torch

from tellegen.solvers.grounding import spd_certificate as _spd_certificate
from tellegen.solvers.grounding import spd_diagnosis as _spd_diagnosis

Tensor = torch.Tensor


class GraphLaplacianOperator:
    """A_I diag(g) A_I^T as a matvec-free LinearOperator. Always symmetric."""

    symmetric = True

    def __init__(
        self,
        src: Tensor,
        tgt: Tensor,
        slopes: Tensor,
        n_interior: int,
        interior_of_node: Tensor,
        *,
        boundary_mask: Tensor,
    ) -> None:
        # Global Constraint: ValueError for bad shapes, naming the offender. The last check
        # is not cosmetic: `interior_nodes = torch.empty(n_interior)` below leaves an
        # UNINITIALISED slot when `n_interior` overstates the mask, and that slot is then
        # used as a gather index -- so before this check the operator constructed happily,
        # reported its claimed shape, and `matvec` returned all zeros (a silent wrong
        # answer, not the opaque torch error the deferral ruling assumed).
        if src.shape != tgt.shape:
            raise ValueError(
                f"GraphLaplacianOperator: src and tgt must have the same shape, got "
                f"src {tuple(src.shape)} and tgt {tuple(tgt.shape)}"
            )
        if slopes.shape[-1] != src.shape[-1]:
            raise ValueError(
                f"GraphLaplacianOperator: slopes has {slopes.shape[-1]} edges in its last "
                f"dimension (shape {tuple(slopes.shape)}) but src/tgt have "
                f"{src.shape[-1]} (shape {tuple(src.shape)})"
            )
        if interior_of_node.shape != boundary_mask.shape:
            raise ValueError(
                f"GraphLaplacianOperator: interior_of_node and boundary_mask must have the "
                f"same shape, got interior_of_node {tuple(interior_of_node.shape)} and "
                f"boundary_mask {tuple(boundary_mask.shape)}"
            )
        n_unmasked = int((~boundary_mask).sum())
        if n_interior != n_unmasked:
            raise ValueError(
                f"GraphLaplacianOperator: n_interior is {n_interior} but boundary_mask "
                f"(shape {tuple(boundary_mask.shape)}) leaves {n_unmasked} interior nodes"
            )

        self.src = src
        self.tgt = tgt
        self.slopes = slopes
        self.n_interior = n_interior
        self.interior_of_node = interior_of_node
        self.boundary_mask = boundary_mask

        n = boundary_mask.shape[-1]
        # Invert interior_of_node (node -> compact index) to interior_nodes (compact index ->
        # node), so matvec can scatter a compact-domain vector into the shared n-node space
        # and gather it back. This is a construction-time, O(n) vectorised computation (no
        # Python loop over edges or nodes): boundary nodes are excluded by boundary_mask
        # before the fancy-index scatter below, so their (unused) compact indices never
        # collide with a real interior one.
        node_ids = torch.arange(n, device=boundary_mask.device)
        interior_positions = node_ids[~boundary_mask]
        compact = interior_of_node[~boundary_mask]
        interior_nodes = torch.empty(n_interior, dtype=torch.long, device=boundary_mask.device)
        interior_nodes[compact] = interior_positions
        self._interior_nodes = interior_nodes
        self._n = n

        self.shape = slopes.shape[:-1] + (n_interior, n_interior)
        self.dtype = slopes.dtype
        self.device = slopes.device

    def _scatter_to_full(self, x: Tensor) -> Tensor:
        """(..., n_interior) -> (..., n), boundary entries implicitly 0."""
        batch_shape = x.shape[:-1]
        phi = torch.zeros(batch_shape + (self._n,), dtype=x.dtype, device=x.device)
        phi[..., self._interior_nodes] = x
        return phi

    def _weighted_difference(
        self, phi_full: Tensor, slopes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Shared body of matvec/rmatvec: w = slopes * (phi[src] - phi[tgt]), plus the
        broadcast src/tgt index tensors used to scatter w back.
        """
        batch_shape = phi_full.shape[:-1]
        b = self.src.shape[-1]
        src = self.src.expand(batch_shape + (b,))
        tgt = self.tgt.expand(batch_shape + (b,))
        d = torch.gather(phi_full, -1, src) - torch.gather(phi_full, -1, tgt)
        return slopes * d, src, tgt

    def _apply(self, x: Tensor) -> Tensor:
        """Shared body of matvec/rmatvec: A_I diag(g) A_I^T is symmetric, so the transpose
        action is the SAME formula as the forward one -- this is that one formula, called
        from two distinct methods. `test_rmatvec_is_a_distinct_method_from_matvec` checks
        `matvec.__func__ is not rmatvec.__func__` (never `rmatvec = matvec`), which this
        satisfies: both remain their own method objects, each merely delegating here.
        """
        batch_shape = torch.broadcast_shapes(x.shape[:-1], self.slopes.shape[:-1])
        x = x.expand(batch_shape + (self.n_interior,))
        slopes = self.slopes.expand(batch_shape + (self.slopes.shape[-1],))
        phi = self._scatter_to_full(x)
        w, src, tgt = self._weighted_difference(phi, slopes)
        out = torch.zeros(batch_shape + (self._n,), dtype=x.dtype, device=x.device)
        out.scatter_add_(-1, src, w)
        out.scatter_add_(-1, tgt, -w)
        return out[..., self._interior_nodes]

    def matvec(self, x: Tensor) -> Tensor:
        return self._apply(x)

    def rmatvec(self, x: Tensor) -> Tensor:
        return self._apply(x)

    def diagonal(self) -> Tensor:
        # Diagonal entry at interior node i is the sum of slopes over every edge incident to
        # i (from either end) -- scatter-add the SAME slopes tensor at both src and tgt.
        batch_shape = self.slopes.shape[:-1]
        b = self.src.shape[-1]
        src = self.src.expand(batch_shape + (b,))
        tgt = self.tgt.expand(batch_shape + (b,))
        diag = torch.zeros(batch_shape + (self._n,), dtype=self.dtype, device=self.device)
        diag.scatter_add_(-1, src, self.slopes)
        diag.scatter_add_(-1, tgt, self.slopes)
        return diag[..., self._interior_nodes]

    def assemble(self) -> Tensor:
        """Dense (..., n_interior, n_interior), the oracle path for small problems only."""
        b = self.src.shape[-1]
        A_full = torch.zeros(self._n, b, dtype=self.dtype, device=self.device)
        ones = torch.ones(b, dtype=self.dtype, device=self.device)
        idx = torch.arange(b, device=self.device)
        # index_put_ with accumulate=True (not plain fancy-index assignment) is required here:
        # a multi-edge graph can have two edges sharing a source node, and assignment would
        # silently drop all but the last such edge's contribution.
        A_full.index_put_((self.src, idx), ones, accumulate=True)
        A_full.index_put_((self.tgt, idx), -ones, accumulate=True)
        A_I = A_full[self._interior_nodes]
        return torch.einsum("ie,...e,je->...ij", A_I, self.slopes, A_I)

    def spd_certificate(self) -> Tensor:
        return _spd_certificate(
            self.src, self.tgt, self.slopes, self.interior_of_node, self.boundary_mask
        )

    def spd_diagnosis(self) -> list[dict]:
        """Delegate to solvers.grounding.spd_diagnosis with the operator's parameters."""
        return _spd_diagnosis(
            self.src, self.tgt, self.slopes, self.interior_of_node, self.boundary_mask
        )
