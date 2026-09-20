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

The hot path (`_apply`, `diagonal`) runs those two primitives over a PADDED COMPACT domain of
width `n_interior + 1` rather than over the full n-node space: `_src_compact`/`_tgt_compact`
carry the endpoints in the operator's own interior indexing, with every boundary node folded
into one shared pad slot whose value is written but never read. See the comment on their
construction for why that is bit-identical to the full-space form.
"""

from __future__ import annotations

import torch

from noodl.solvers.grounding import spd_certificate as _spd_certificate
from noodl.solvers.grounding import spd_diagnosis as _spd_diagnosis

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
        # An interior node is one that is NOT prescribed (`boundary_mask`) AND carries a
        # compact index (`interior_of_node >= 0`). The second half is what a layer's INACTIVE
        # nodes fail (spec 14, 4.5): a node no edge of this layer's kinds touches is neither
        # an unknown of this operator nor a prescribed boundary value, so `boundary_mask`
        # alone would over-count the interior by exactly those nodes. Requiring BOTH also
        # leaves the older, looser calling convention intact, in which `interior_of_node`
        # holds an arbitrary value at a masked (boundary) node and only the mask says so.
        # An inactive node carries no edge here by construction, so nothing below gathers or
        # scatters at its (pad) slot.
        interior_mask = (~boundary_mask) & (interior_of_node >= 0)
        n_marked = int(interior_mask.sum())
        if n_interior != n_marked:
            raise ValueError(
                f"GraphLaplacianOperator: n_interior is {n_interior} but boundary_mask and "
                f"interior_of_node (shape {tuple(interior_of_node.shape)}) mark {n_marked} "
                f"nodes as interior"
            )

        self.src = src
        self.tgt = tgt
        self.slopes = slopes
        self.n_interior = n_interior
        self.interior_of_node = interior_of_node
        self.boundary_mask = boundary_mask

        n = boundary_mask.shape[-1]
        # Invert interior_of_node (node -> compact index) to interior_nodes (compact index ->
        # node), so the dense `assemble` oracle can pick this operator's own interior rows
        # out of the shared n-node space. This is a construction-time, O(n) vectorised
        # computation (no Python loop over edges or nodes): boundary and inactive nodes are excluded
        # by `interior_mask` before the fancy-index scatter below, so their (unused) compact
        # index -1 never collides with -- or overwrites -- a real interior one.
        node_ids = torch.arange(n, device=boundary_mask.device)
        interior_positions = node_ids[interior_mask]
        compact = interior_of_node[interior_mask]
        interior_nodes = torch.empty(n_interior, dtype=torch.long, device=boundary_mask.device)
        interior_nodes[compact] = interior_positions
        self._interior_nodes = interior_nodes
        self._n = n

        # The same endpoints expressed in the operator's OWN compact interior indexing, with
        # EVERY non-interior node (boundary or inactive) mapped to one shared PAD slot at
        # index `n_interior`. `_apply`
        # then works on an `n_interior + 1`-wide vector instead of the full n-node space:
        # what it gathers at a boundary endpoint is the pad's zero (exactly what "boundary
        # nodes are held at potential 0" means here), and what it scatters back at a
        # boundary node lands in the pad slot, which is dropped. This is bit-identical to
        # routing through the full node space -- every INTERIOR slot still accumulates its
        # own incident edges, in the same order, and the pad slot's value is never read --
        # and it removes the two full-width fancy-index operations the old form needed (the
        # scatter of x INTO the node space, and the select of the interior rows back OUT).
        compact_of_node = torch.where(
            interior_mask, interior_of_node, torch.full_like(interior_of_node, n_interior)
        )
        self._src_compact = compact_of_node[src]
        self._tgt_compact = compact_of_node[tgt]

        # Per-(input batch shape, device) cache of everything `_apply` would otherwise
        # recompute on EVERY matvec: the broadcast batch shape and the src/tgt index tensors
        # expanded to it. Both are pure functions of the key and of this operator's own
        # construction-time arrays, so caching them is the "construction-time cost, cached"
        # the plan's Global Constraints allow -- and at the composed reference size the
        # recomputation was 11% of `pcg`'s time (820 `broadcast_shapes` calls per solve).
        # Only INDEX tensors are cached, never an expanded view of `slopes`: `slopes` can
        # require grad, and a cached view of it would carry one autograd graph's
        # `ExpandBackward` into the next forward pass, so a second backward would try to
        # traverse a freed graph. `expand` on a fresh tensor is one dispatch; that is the
        # cheap half anyway.
        #
        # UNBOUNDED BY TYPE, BOUNDED IN FACT, and the lifetime argument is the load-bearing
        # one: a GraphLaplacianOperator is rebuilt at every Newton iterate (see
        # `PotentialFlowLayer.solve`'s `operator_fn`), so one instance sees a handful of
        # solves with one or two distinct (batch shape, device) keys and is then discarded.
        # The entries are stride-0 views of two (edges,) arrays, so even a pathological key
        # set would cost almost nothing; there is deliberately no eviction policy.
        self._index_cache: dict[tuple, tuple[torch.Size, Tensor, Tensor]] = {}

        # Lazily built COO index pattern for `assemble_sparse` (row, col and the gather
        # index into the four-block stencil). Built on first use rather than here: most
        # operators this class builds are never assembled sparsely at all, and the pattern
        # costs a cat and a nonzero over 4 * edges entries.
        self._coo_indices: tuple[Tensor, Tensor, Tensor] | None = None

        self.shape = slopes.shape[:-1] + (n_interior, n_interior)
        self.dtype = slopes.dtype
        self.device = slopes.device

    def _endpoints_for(self, x_batch: torch.Size, device) -> tuple[torch.Size, Tensor, Tensor]:
        """`(batch_shape, src, tgt)` for an input with leading shape `x_batch`, memoised.

        `batch_shape` is the broadcast of `x_batch` against this operator's own slope batch;
        `src`/`tgt` are the COMPACT (pad-slot) endpoint arrays expanded to
        `batch_shape + (edges,)`, which is the form `torch.gather`/`Tensor.scatter_add_`
        need (both want an index of the same rank as the data). Everything here depends only
        on the key and on construction-time state, so a dict cache on the instance turns a
        per-matvec cost into a per-shape one. The expansions are VIEWS (stride 0 on every
        batch dim), so the cache holds no memory beyond the two (edges,) arrays it views.
        """
        key = (tuple(x_batch), device)
        cached = self._index_cache.get(key)
        if cached is None:
            batch_shape = torch.broadcast_shapes(x_batch, self.slopes.shape[:-1])
            edges = self.src.shape[-1]
            cached = (
                batch_shape,
                self._src_compact.expand(batch_shape + (edges,)),
                self._tgt_compact.expand(batch_shape + (edges,)),
            )
            self._index_cache[key] = cached
        return cached

    def _apply(self, x: Tensor) -> Tensor:
        """Shared body of matvec/rmatvec: A_I diag(g) A_I^T is symmetric, so the transpose
        action is the SAME formula as the forward one -- this is that one formula, called
        from two distinct methods. `test_rmatvec_is_a_distinct_method_from_matvec` checks
        `matvec.__func__ is not rmatvec.__func__` (never `rmatvec = matvec`), which this
        satisfies: both remain their own method objects, each merely delegating here.

        The body is deliberately flat: this runs once per PCG iteration, thousands of times
        per solve, so a Python frame removed from it is a real saving.

        The two `scatter_add_` calls are NOT fused into one `index_add` over
        `cat([src, tgt])` with `cat([w, -w])`. That fusion is bit-identical (same `x` to the
        last bit, since each output node still accumulates its incident edges in the same
        order), but it is slower here: building `cat([w, -w])` costs a negate and a copy of a
        (batch, 2 * edges) tensor, which is more than the one `scatter_add_` dispatch it
        saves. The accumulator zeros tensor is already shared by both scatters, so there was
        never a second one to save.

        History: see docs/development-history.md (Milestone 1b).
        """
        n_i = self.n_interior
        batch_shape, src, tgt = self._endpoints_for(x.shape[:-1], x.device)
        x = x.expand(batch_shape + (n_i,))
        slopes = self.slopes.expand(batch_shape + (self.slopes.shape[-1],))
        pad = torch.zeros(batch_shape + (1,), dtype=x.dtype, device=x.device)
        phi = torch.cat([x, pad], dim=-1)
        w = slopes * (torch.gather(phi, -1, src) - torch.gather(phi, -1, tgt))
        out = torch.zeros(batch_shape + (n_i + 1,), dtype=x.dtype, device=x.device)
        out.scatter_add_(-1, src, w)
        out.scatter_add_(-1, tgt, -w)
        # A VIEW of the accumulator, not a copy: the pad slot is simply not part of it.
        return out[..., :n_i]

    def matvec(self, x: Tensor) -> Tensor:
        return self._apply(x)

    def rmatvec(self, x: Tensor) -> Tensor:
        return self._apply(x)

    def diagonal(self) -> Tensor:
        # Diagonal entry at interior node i is the sum of slopes over every edge incident to
        # i (from either end) -- scatter-add the SAME slopes tensor at both src and tgt, in
        # the same padded compact domain `_apply` uses (every edge incident to a boundary
        # node contributes to the pad slot, which is dropped).
        n_i = self.n_interior
        batch_shape, src, tgt = self._endpoints_for(self.slopes.shape[:-1], self.device)
        diag = torch.zeros(batch_shape + (n_i + 1,), dtype=self.dtype, device=self.device)
        diag.scatter_add_(-1, src, self.slopes)
        diag.scatter_add_(-1, tgt, self.slopes)
        return diag[..., :n_i]

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

    def assemble_sparse(self) -> tuple[Tensor, Tensor, Tensor]:
        """COO `(row, col, values)` for A_I diag(g) A_I^T, in O(edges) and with no n x n form.

        The optional `operators.base.SparseAssembling` member. One edge between interior
        nodes i, j with slope g contributes the four-entry stencil (i,i)+=g, (j,j)+=g,
        (i,j)-=g, (j,i)-=g -- which is literally the outer product of that edge's incidence
        column with itself, weighted by g, i.e. the e-th term of A_I diag(g) A_I^T. An edge
        between an interior node i and a BOUNDARY node contributes only (i,i)+=g, because
        the boundary row and column are not part of this operator's domain at all.

        Both cases fall out of ONE vectorised expression, with no branch on which endpoints
        are interior, by reusing `_src_compact`/`_tgt_compact` -- the endpoints in this
        operator's own compact interior indexing, with every boundary node folded into the
        single pad slot at index `n_interior` (see the module docstring). The four blocks
        are emitted for every edge and then filtered by `row < n_interior & col < n_interior`,
        which drops exactly the pad entries: for an interior-boundary edge that leaves the
        (i,i) diagonal entry and nothing else, and for a boundary-boundary edge it leaves
        nothing. The filter reads only INDICES, never slopes, so `row`/`col` are shared
        across the batch as the contract requires, and they are cached: a Newton step builds
        a fresh operator at new slopes but on the same topology, and only the values move.

        DUPLICATES are emitted deliberately and left uncoalesced (a multigraph's parallel
        edges land on the same (i,j) pair, as do the four blocks' own diagonals): the COO
        consumer sums them, and coalescing here would cost a sort for nothing.
        """
        n_i = self.n_interior
        cached = self._coo_indices
        if cached is None:
            sc, tc = self._src_compact, self._tgt_compact
            row = torch.cat([sc, tc, sc, tc]).to(torch.int64)
            col = torch.cat([sc, tc, tc, sc]).to(torch.int64)
            keep = torch.nonzero((row < n_i) & (col < n_i), as_tuple=False).flatten()
            cached = (row[keep], col[keep], keep)
            self._coo_indices = cached
        row, col, keep = cached
        g = self.slopes
        values = torch.cat([g, g, -g, -g], dim=-1).index_select(-1, keep)
        return row, col, values

    def spd_certificate(self) -> Tensor:
        return _spd_certificate(
            self.src, self.tgt, self.slopes, self.interior_of_node, self.boundary_mask
        )

    def spd_diagnosis(self) -> list[dict]:
        """Delegate to solvers.grounding.spd_diagnosis with the operator's parameters."""
        return _spd_diagnosis(
            self.src, self.tgt, self.slopes, self.interior_of_node, self.boundary_mask
        )
