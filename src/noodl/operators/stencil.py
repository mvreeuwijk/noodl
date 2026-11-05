"""The graph-Laplacian edge stencil, written once.

A conductance `g` on an edge between nodes `s` and `t` adds `+g` at `(s, s)` and `(t, t)` and
`-g` at `(s, t)` and `(t, s)` to a symmetric matrix `A diag(g) A^T`. Every operator that
carries a conduction or conductance term (the potential-network Jacobian, the transport
operator's conduction term, its sparse and dense assemblies) spells out that same stencil.
The helpers here are that one spelling, in the three forms the operators need: the action on a
vector, its diagonal, and its COO triplets.

Index tensors follow `torch.gather`/`Tensor.scatter_add_`: they have the same rank as the data
they index, with the edge axis last, and are usually stride-0 expansions of one `(edges,)`
array. Callers own that expansion (and cache it), because it is per-shape work that must not be
repeated inside a solver's inner loop.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def laplacian_apply(x: Tensor, src: Tensor, tgt: Tensor, g: Tensor) -> Tensor:
    """`A diag(g) A^T x` for the edges `src -> tgt`: the Laplacian acting on `x`, batched.

    `x` is `(..., n)`; `src` and `tgt` are `(..., edges)` node indices into its last axis
    (same leading shape as `x`, see the module docstring); `g` broadcasts against `(...,
    edges)`. Returns `(..., n)`: node `i` receives `sum_e g_e (x[src_e] - x[tgt_e])` over
    the edges leaving it and the negative of that over the edges entering it. Edges
    accumulate in edge order, `src` side first.

    A node index may point at a padding slot that the caller drops afterwards (the potential
    network folds every boundary node into one such slot); the slot is written like any
    other.
    """
    w = g * (torch.gather(x, -1, src) - torch.gather(x, -1, tgt))
    out = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    out.scatter_add_(-1, src, w)
    out.scatter_add_(-1, tgt, -w)
    return out


def laplacian_diagonal(src: Tensor, tgt: Tensor, g: Tensor, n: int) -> Tensor:
    """The diagonal of `A diag(g) A^T`, `(..., n)`: node `i` gets the sum of `g` over every
    edge incident to it, from either end.

    `src`, `tgt` and `g` are `(..., edges)` with the same leading shape (see the module
    docstring); the result takes its dtype and device from `g`. The `src` ends accumulate
    first, then the `tgt` ends.
    """
    diag = torch.zeros(src.shape[:-1] + (n,), dtype=g.dtype, device=g.device)
    diag.scatter_add_(-1, src, g)
    diag.scatter_add_(-1, tgt, g)
    return diag


def laplacian_coo_indices(src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
    """`(row, col)` of the four-block stencil of the edges `src -> tgt`, each `(4 * edges,)`.

    The blocks are `(s, s)`, `(t, t)`, `(s, t)`, `(t, s)`, in that order, and
    `laplacian_coo_values` lays its values out the same way. Duplicate positions (parallel
    edges, the diagonals of the four blocks) are left uncoalesced for the COO consumer to
    sum. The indices depend on the topology only, so callers cache them.
    """
    return torch.cat([src, tgt, src, tgt]), torch.cat([src, tgt, tgt, src])


def laplacian_coo_values(g: Tensor) -> Tensor:
    """The values that go with `laplacian_coo_indices`: `[g, g, -g, -g]` along the last axis,
    `(..., 4 * edges)`."""
    return torch.cat([g, g, -g, -g], dim=-1)


def laplacian_coo(src: Tensor, tgt: Tensor, g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """COO `(row, col, values)` of `A diag(g) A^T` for the edges `src -> tgt`, uncoalesced.

    `src`/`tgt` are `(edges,)` node indices, `g` is `(..., edges)`; the triplet has
    `4 * edges` entries and its `values` carry `g`'s leading batch shape. Summing duplicates
    gives the matrix.
    """
    row, col = laplacian_coo_indices(src, tgt)
    return row, col, laplacian_coo_values(g)
