"""Folding storage-free nodes that only conduct into their neighbours.

A node that holds no heat and is joined to the rest of the network by conduction only (the
surface node of a wall, the radiant star node of a zone) has an algebraic balance: what
conducts in conducts out, plus its own source. Its value is the conductance-weighted mean of
its neighbours' values, shifted by its source. Substituting that back gives the neighbours
equivalent conductances among themselves and a share of its source: the star-delta transform
of circuit theory, and the "combined surface coefficient" of building physics (a surface film
in series with the wall conduction behind it). Several such nodes joined to each other (the
surfaces of one zone and their star node) are folded together as one GROUP.

For a group with conduction matrix `K` (its own conductances; `K` is the group's Laplacian
plus, on the diagonal, its conductances to the outside) and the conductances `B` from its
nodes to its NEIGHBOURS (stored nodes or boundaries), the group's values are

    x_group = K^-1 (B x_neighbours + s_group)

(the Schur complement of the group's block), the neighbours `a`, `b` gain the equivalent
conductance `W_ab = (B^T K^-1 B)_ab` between them, and neighbour `a` gains the source
`(B^T K^-1 s_group)_a`. `W` has non-negative entries (`K` is an M-matrix, so `K^-1 >= 0`), so
the folded network is again a network of positive conductances: the maximum principle and
positivity of every scheme are kept, and the exact (exponential) step stays available.

`K` is symmetric positive definite whenever every node of the group reaches a neighbour
through positive conductances. A node whose every conductance is zero this call is HELD (it
keeps its value; its source has nowhere to go and is dropped), as a storage-free node that
nothing passes through always was. A part of a group joined only to itself has no value at all
and is refused by name.

The groups are found once, at construction. Each call assembles one small dense `K` and `B`
per group from the per-call conductances, batched over the groups of one size (a "bucket")
and over any leading batch dimensions, and factors them with a batched Cholesky. Nothing here
forms an `(n, n)` matrix over the layer. A group is as large as the set of folded nodes joined
to each other, which for a building is the surfaces of one zone and its star node; a very
large group (thousands of nodes joined by storage-free conduction) is factored densely, at a
cost cubic in its size.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from noodl.operators.stencil import laplacian_coo_indices, laplacian_coo_values

Tensor = torch.Tensor


class FoldBucket(NamedTuple):
    """The groups of one size `m`, padded to `p` neighbours each. Index tensors only."""

    groups: int          # G
    size: int            # m, nodes per group
    width: int           # p, neighbours per group (padded)
    folded: Tensor       # (G*m,) interior positions of the group nodes, group-major
    edges: Tensor        # (ne,) conduction-edge positions of the edges touching these groups
    k_sel: Tensor        # stencil entries (of `edges`) that land in K ...
    k_pos: Tensor        # ... at these flat positions of (G, m, m)
    b_sel: Tensor        # stencil entries that land in B (negated) ...
    b_pos: Tensor        # ... at these flat positions of (G, m, p)
    neighbours: Tensor   # (G*p,) full node index of each neighbour slot, n for padding
    kept_slot: Tensor    # (G*p,) kept position of each neighbour slot, n_k if not kept
    pairs: Tensor        # flat positions in (G, p, p) of the equivalent conductances kept
    inner_sel: Tensor    # positions in `edges` of the edges inside a group
    inner_row: Tensor    # (2 * inner,) local (g*m + i) ends of those edges, both directions
    inner_col: Tensor


class FoldPlan(NamedTuple):
    """Which nodes are folded, in which groups, and the folded network's edges."""

    kept: Tensor         # (n_k,) interior positions of the nodes the layer still steps
    folded: Tensor       # (n_e,) interior positions of the folded nodes
    kept_edges: Tensor   # conduction-edge positions with no folded end
    touching: Tensor     # conduction-edge positions with a folded end
    src: Tensor          # (b_r,) ends of the folded network's conduction edges: the kept
    tgt: Tensor          # edges, then the equivalent edges of every bucket in order
    buckets: tuple[FoldBucket, ...]


class Folded(NamedTuple):
    """One call's folding: the folded network's conductances and, per bucket, the Cholesky
    factor of `K`, `X = K^-1 B` and the held nodes (`None` when no conductance is zero)."""

    conductance: Tensor                                  # (..., b_r)
    blocks: tuple[tuple[Tensor, Tensor, Tensor | None], ...]


def _long(values) -> Tensor:
    return torch.tensor(values, dtype=torch.long)


def plan_folding(
    *, name: str, nodes: list, interior_idx: Tensor, folded_mask: Tensor,
    src: Tensor, tgt: Tensor, boundary_idx: Tensor, n: int,
) -> FoldPlan:
    """Group the folded nodes (`folded_mask`, over the interior) by the conduction edges
    `src -> tgt` (full node indices) joining them, and lay out the index tensors each call
    uses. Construction-time; the Python loops run over nodes and edges once.

    Raises `ValueError` naming a group joined by conduction only to itself (no neighbour at
    all): nothing could ever set its value."""
    interior = interior_idx.tolist()
    folded_pos = folded_mask.nonzero().flatten().tolist()
    kept_pos = (~folded_mask).nonzero().flatten().tolist()
    pos_of = {node: i for i, node in enumerate(interior)}
    kept_slot_of = {interior[p]: k for k, p in enumerate(kept_pos)}
    folded_nodes = [interior[p] for p in folded_pos]
    is_folded = set(folded_nodes)
    boundary = set(boundary_idx.tolist())
    s_list, t_list = src.tolist(), tgt.tolist()

    parent = {v: v for v in folded_nodes}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for s, t in zip(s_list, t_list, strict=True):
        if s in is_folded and t in is_folded:
            a, b = find(s), find(t)
            if a != b:
                parent[max(a, b)] = min(a, b)

    members: dict[int, list[int]] = {}
    for v in folded_nodes:                       # interior order within each group
        members.setdefault(find(v), []).append(v)
    edges_of: dict[int, list[int]] = {root: [] for root in members}
    neighbours_of: dict[int, list[int]] = {root: [] for root in members}
    kept_edges, touching = [], []
    for e, (s, t) in enumerate(zip(s_list, t_list, strict=True)):
        if s not in is_folded and t not in is_folded:
            kept_edges.append(e)
            continue
        touching.append(e)
        root = find(s if s in is_folded else t)
        edges_of[root].append(e)
        for end in (s, t):
            if end not in is_folded and end not in neighbours_of[root]:
                neighbours_of[root].append(end)
    for root, nbs in neighbours_of.items():
        if not nbs:
            names = [nodes[v] for v in members[root]]
            raise ValueError(
                f"TransportLayer '{name}': storage-free nodes {names} are joined by "
                f"conduction only to each other; no stored node or boundary sets their value"
            )

    by_size: dict[int, list[int]] = {}
    for root in sorted(members, key=lambda r: pos_of[members[r][0]]):
        by_size.setdefault(len(members[root]), []).append(root)

    n_k = len(kept_pos)
    red_src = [s_list[e] for e in kept_edges]
    red_tgt = [t_list[e] for e in kept_edges]
    buckets = []
    for m, roots in sorted(by_size.items()):
        G = len(roots)
        p = max(len(neighbours_of[r]) for r in roots)
        lsrc, ltgt, edges, folded_b = [], [], [], []
        nb_nodes, nb_kept, pairs = [], [], []
        inner_sel, inner_row, inner_col = [], [], []
        for g, root in enumerate(roots):
            mem, nbs = members[root], neighbours_of[root]
            local = {v: g * m + i for i, v in enumerate(mem)}
            local.update({v: G * m + g * p + j for j, v in enumerate(nbs)})
            folded_b.extend(pos_of[v] for v in mem)
            for e in edges_of[root]:
                s, t = s_list[e], t_list[e]
                if s in is_folded and t in is_folded:
                    inner_sel.append(len(edges))
                    inner_row.append(local[s])
                    inner_col.append(local[t])
                edges.append(e)
                lsrc.append(local[s])
                ltgt.append(local[t])
            nb_nodes.extend(nbs + [n] * (p - len(nbs)))
            nb_kept.extend([kept_slot_of.get(v, n_k) for v in nbs] + [n_k] * (p - len(nbs)))
            for a in range(len(nbs)):
                for b in range(a + 1, len(nbs)):
                    if nbs[a] in boundary and nbs[b] in boundary:
                        continue                 # between two prescribed values: no unknown
                    pairs.append(g * p * p + a * p + b)
                    red_src.append(nbs[a])
                    red_tgt.append(nbs[b])
        row, col = laplacian_coo_indices(_long(lsrc), _long(ltgt))
        k_sel, k_pos, b_sel, b_pos = [], [], [], []
        for entry, (r, c) in enumerate(zip(row.tolist(), col.tolist(), strict=True)):
            if r >= G * m:
                continue                         # a neighbour's own row: not needed
            g, i = divmod(r, m)
            if c < G * m:
                k_sel.append(entry)
                k_pos.append(g * m * m + i * m + c % m)
            else:
                b_sel.append(entry)
                b_pos.append(g * m * p + i * p + (c - G * m - g * p))
        buckets.append(FoldBucket(
            groups=G, size=m, width=p, folded=_long(folded_b), edges=_long(edges),
            k_sel=_long(k_sel), k_pos=_long(k_pos), b_sel=_long(b_sel), b_pos=_long(b_pos),
            neighbours=_long(nb_nodes), kept_slot=_long(nb_kept), pairs=_long(pairs),
            inner_sel=_long(inner_sel), inner_row=_long(inner_row + inner_col),
            inner_col=_long(inner_col + inner_row),
        ))
    device = interior_idx.device
    plan = FoldPlan(
        kept=_long(kept_pos), folded=_long(folded_pos), kept_edges=_long(kept_edges),
        touching=_long(touching), src=_long(red_src), tgt=_long(red_tgt),
        buckets=tuple(buckets),
    )
    if device.type != "cpu":
        plan = _to_device(plan, device)
    return plan


def _to_device(plan: FoldPlan, device) -> FoldPlan:
    def move(t):
        return t.to(device) if isinstance(t, Tensor) else t
    buckets = tuple(FoldBucket(*(move(v) for v in b)) for b in plan.buckets)
    return FoldPlan(*(move(v) for v in plan[:-1]), buckets)


def fold_conductance(plan: FoldPlan, g: Tensor, refuse_closed) -> Folded:
    """The folded network's conductances `(..., b_r)` for the per-call conductances
    `g` `(..., b_c)`, and the per-bucket factors the sources and the recovery use.

    Differentiable in `g`. `refuse_closed(bucket, start, linked)` is called (with no
    gradient) for a bucket of groups larger than one node when some conductance touching a
    folded node is zero: it must raise for a group node that cannot reach a neighbour."""
    batch = g.shape[:-1]
    parts = [g.index_select(-1, plan.kept_edges)]
    with torch.no_grad():
        some_zero = bool((g.index_select(-1, plan.touching) == 0).any())
    blocks, infos = [], []
    for b in plan.buckets:
        G, m, p = b.groups, b.size, b.width
        gb = g.index_select(-1, b.edges)
        values = laplacian_coo_values(gb)
        K = g.new_zeros(batch + (G * m * m,)).index_add(
            -1, b.k_pos, values.index_select(-1, b.k_sel)).reshape(batch + (G, m, m))
        B = g.new_zeros(batch + (G * m * p,)).index_add(
            -1, b.b_pos, -values.index_select(-1, b.b_sel)).reshape(batch + (G, m, p))
        idle = None
        if some_zero:
            with torch.no_grad():
                idle = K.diagonal(dim1=-2, dim2=-1) == 0
                if m > 1 and b.inner_sel.numel():
                    start = ((B.sum(-1) > 0) | idle).reshape(-1, G * m)
                    linked = gb.index_select(-1, b.inner_sel) > 0
                    linked = torch.cat([linked, linked], -1).reshape(-1, b.inner_row.numel())
                    refuse_closed(b, start, linked)
            K = K + torch.diag_embed(idle.to(K.dtype))
        L, info = torch.linalg.cholesky_ex(K)
        infos.append(info.reshape(-1))
        X = torch.cholesky_solve(B, L)
        W = B.transpose(-1, -2) @ X
        # W >= 0 exactly; the clamp only removes a rounding-level negative.
        parts.append(W.reshape(batch + (G * p * p,)).index_select(-1, b.pairs).clamp_min(0.0))
        blocks.append((L, X, idle))
    if infos and bool(torch.cat(infos).any()):
        raise RuntimeError(
            "folding storage-free conduction nodes: a group's conduction matrix is not "
            "positive definite (a conductance is not finite?)"
        )
    return Folded(torch.cat(parts, -1), tuple(blocks))


def fold_sources(plan: FoldPlan, folded: Folded, src: Tensor
                 ) -> tuple[Tensor, list[Tensor] | None]:
    """`src` `(..., n_i, K)` (interior, node-major) folded onto the kept nodes:
    `(..., n_k, K)`, and per bucket `Y = K^-1 s_group` for the recovery (`None` when the
    folded nodes carry no source and no gradient asks for one)."""
    kept = src.index_select(-2, plan.kept)
    on_folded = src.index_select(-2, plan.folded)
    if not on_folded.requires_grad and not bool((on_folded != 0).any()):
        return kept, None
    n_k, K = plan.kept.numel(), src.shape[-1]
    ys = []
    for b, (L, X, _idle) in zip(plan.buckets, folded.blocks, strict=True):
        G, m, p = b.groups, b.size, b.width
        s_g = src.index_select(-2, b.folded).reshape(src.shape[:-2] + (G, m, K))
        ys.append(torch.cholesky_solve(s_g, L))
        share = X.transpose(-1, -2) @ s_g                # B^T K^-1 s, (..., G, p, K)
        flat = share.reshape(share.shape[:-3] + (G * p, K))
        padded = flat.new_zeros(flat.shape[:-2] + (n_k + 1, K)).index_add(-2, b.kept_slot, flat)
        kept = kept + padded[..., :n_k, :]
    return kept, ys


def recover(plan: FoldPlan, folded: Folded, ys: list[Tensor] | None, x_kept: Tensor,
            x_boundary: Tensor, *, kept_nodes: Tensor, boundary_idx: Tensor, n: int,
            x_held: Tensor | None, y_scale: float = 1.0) -> Tensor:
    """The interior values `(..., n_i, K)` from the kept nodes' `x_kept` `(..., n_k, K)`
    and the boundary values `(..., n_b, K)`: `x_group = X x_neighbours + y_scale Y`, and a
    held node's value from `x_held` `(..., n_i, K)`. Linear in its inputs, so the same call
    turns the kept nodes' time integral into the folded nodes' (`y_scale = dt`, `x_held` and
    `x_boundary` times `dt`)."""
    K = x_kept.shape[-1]
    batch = torch.broadcast_shapes(x_kept.shape[:-2], x_boundary.shape[:-2])
    values = x_kept.new_zeros(batch + (n + 1, K))
    values = values.index_copy(-2, kept_nodes, x_kept.expand(batch + x_kept.shape[-2:]))
    values = values.index_copy(
        -2, boundary_idx, x_boundary.to(x_kept.dtype).expand(batch + x_boundary.shape[-2:]))
    groups = []
    for i, (b, (_L, X, idle)) in enumerate(zip(plan.buckets, folded.blocks, strict=True)):
        G, m, p = b.groups, b.size, b.width
        x_nb = values.index_select(-2, b.neighbours).reshape(batch + (G, p, K))
        x_g = X @ x_nb
        if ys is not None:
            x_g = x_g + y_scale * ys[i]
        if idle is not None and x_held is not None:
            held = x_held.index_select(-2, b.folded).reshape(x_held.shape[:-2] + (G, m, K))
            x_g = torch.where(idle.unsqueeze(-1), held, x_g)
        groups.append(x_g.reshape(x_g.shape[:-3] + (G * m, K)))
    out_batch = torch.broadcast_shapes(batch, *(x.shape[:-2] for x in groups))
    n_i = plan.kept.numel() + plan.folded.numel()
    out = x_kept.new_zeros(out_batch + (n_i, K))
    out = out.index_copy(-2, plan.kept, x_kept.expand(out_batch + x_kept.shape[-2:]))
    for b, x_g in zip(plan.buckets, groups, strict=True):
        out = out.index_copy(-2, b.folded, x_g.expand(out_batch + x_g.shape[-2:]))
    return out
