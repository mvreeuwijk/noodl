"""Cycle-space utilities: conserved flows from cycle amplitudes and from tree solutions.

``branch_flows`` and ``assert_forward_oriented`` moved here verbatim from
``physics/flows.py``, which now re-exports them for compatibility.
"""

from __future__ import annotations

import torch

from tellegen.topology import Network


def branch_flows(net: Network, amplitudes: torch.Tensor, kind: str | None = None) -> torch.Tensor:
    """Map cycle amplitudes ``(..., l)`` to branch flows ``(..., b_kind)``: ``q = m @ J``.

    ``incidence(kind) @ q == 0`` for every ``m`` because the rows of ``J`` span the
    null space of the incidence matrix.
    """
    J = net.cycle_basis(kind).to(amplitudes.dtype)
    if amplitudes.shape[-1] != J.shape[0]:
        raise ValueError(f"expected {J.shape[0]} amplitudes, got {amplitudes.shape[-1]}")
    return amplitudes @ J


def assert_forward_oriented(net: Network, kind: str | None = None) -> None:
    """Raise unless every cycle-basis entry is 0 or +1.

    A forward-oriented graph is one whose loops all run in the direction of their
    edges, so that non-negative amplitudes give non-negative branch flows and the
    upwind operator is constant.
    """
    J = net.cycle_basis(kind)
    if torch.any(J < 0):
        rows = torch.nonzero(J.min(dim=1).values < 0).flatten().tolist()
        raise ValueError(
            f"graph is not forward oriented: cycle rows {rows} traverse a tree edge backwards; "
            "insert the return edge before the loop edge"
        )


def _expand_index(idx: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    """Broadcast a 1-D index tensor to `(*batch_shape, len(idx))` for scatter_add_/indexing
    against a `(*batch_shape, m)` tensor, without a Python loop over the batch."""
    return idx.reshape((1,) * len(batch_shape) + idx.shape).expand(*batch_shape, *idx.shape)


def _tree_elimination_levels(net: Network, kind: str | None):
    """Levels of net's spanning forest (of edges of `kind`), deepest first, for O(depth)
    elimination. Each level is `(child_nodes, edge_cols, signs, parent_nodes)`, all 1-D
    LongTensors/the edge dtype's signs tensor, describing every node at that BFS depth
    simultaneously: `signs[i]` is `+1` if `child_nodes[i]` is the SOURCE of
    `edge_cols[i]` and `-1` if it is the TARGET (i.e. exactly `incidence(kind)[child, edge]`).
    One root per component (the first node index touched in that component, matching
    particular_flow's existing "drop one reference node" convention) never appears as a
    child and so never receives a level entry. Cached on `net._cache`, exactly like
    `spanning_forest`/`component_labels`; this is the one Python-level (per-node) loop in
    this module, and it runs once per (net, kind), not once per solve.
    """
    key = ("tree_elimination_levels", kind)
    if key in net._cache:
        return net._cache[key]
    tree_cols, _chord_cols = net.spanning_forest(kind)
    cols = net.edge_index(kind).tolist()  # local (kind-filtered) index -> global edge index
    edges = net.edges
    labels = net.component_labels(kind)
    n = net.n
    adjacency: dict[int, list[tuple[int, int, float]]] = {i: [] for i in range(n)}
    for col in tree_cols.tolist():
        u, v, _ = edges[cols[col]]
        ui, vi = net.node_index(u), net.node_index(v)
        adjacency[ui].append((vi, col, -1.0))  # if we move u -> v, child=v is the TARGET
        adjacency[vi].append((ui, col, 1.0))  # if we move v -> u, child=u is the SOURCE
    n_components = net.n_components_of(kind)
    depth = [-1] * n
    parent_of = [-1] * n
    parent_edge = [-1] * n
    parent_sign = [0.0] * n
    order: list[int] = []
    for c in range(n_components):
        node_idx = torch.nonzero(labels == c, as_tuple=False).flatten().tolist()
        if not node_idx:
            continue
        root = node_idx[0]
        depth[root] = 0
        frontier = [root]
        while frontier:
            nxt = []
            for node in frontier:
                for neighbour, col, sign in adjacency[node]:
                    if depth[neighbour] == -1:
                        depth[neighbour] = depth[node] + 1
                        parent_of[neighbour] = node
                        parent_edge[neighbour] = col
                        parent_sign[neighbour] = sign
                        order.append(neighbour)
                        nxt.append(neighbour)
            frontier = nxt
    max_depth = max((depth[node] for node in order), default=0)
    levels = []
    for d in range(max_depth, 0, -1):
        nodes_at_d = [node for node in order if depth[node] == d]
        if not nodes_at_d:
            continue
        levels.append(
            (
                torch.tensor(nodes_at_d, dtype=torch.long, device=net.device),
                torch.tensor(
                    [parent_edge[node] for node in nodes_at_d], dtype=torch.long, device=net.device
                ),
                torch.tensor(
                    [parent_sign[node] for node in nodes_at_d], dtype=net.dtype, device=net.device
                ),
                torch.tensor(
                    [parent_of[node] for node in nodes_at_d], dtype=torch.long, device=net.device
                ),
            )
        )
    net._cache[key] = levels
    return levels


def _tree_solve(net: Network, kind: str | None, rhs: torch.Tensor) -> torch.Tensor:
    """Solve `A_tree @ q_tree = rhs` over `net`'s spanning forest of edges of `kind`, zero
    on chord edges. `rhs` is `(..., n)` and must already sum to (near) zero within every
    connected component (the caller's responsibility -- particular_flow checks this
    explicitly against its own external sources; branch_flows's chord-source construction
    guarantees it by build, since it only ever moves +m/-m between two nodes already in the
    same tree component). Returns `(..., b_kind)`.

    Algorithm: level-synchronous elimination, deepest level first (see this task's header
    note). At each level, every child's excess demand is read off (gather), its parent tree
    edge is solved for directly (`A[child, edge]` is +-1, so dividing is multiplying by the
    same sign), and the child's ENTIRE excess is handed up to its parent (scatter-add) --
    physically, "whatever this child's own subtree could not satisfy internally must now be
    satisfied by the rest of the tree above it." A node with two or more children at the
    same level scatters onto the same parent additively, which is exactly why scatter_add_
    (not a plain index assignment) is used for the handoff.

    The per-level write into `q` uses `Tensor.scatter` (out-of-place, autograd-safe), not
    `q[..., idx] = value` with a batch-shaped `idx`: plain `__setitem__` with an index tensor
    that carries its own leading batch dimensions does not pair each batch row with its own
    row of `idx` -- it broadcasts the last write across every batch row instead (confirmed
    empirically: `q[..., idx] = vals` with `idx`/`vals` shape `(2, k)` leaves both rows of
    `q` equal to the *second* row of `vals`). `scatter` performs the intended per-row paired
    write, and is safe here because every entry of `edge_cols` is already distinct within one
    level (a node has exactly one parent edge).
    """
    levels = _tree_elimination_levels(net, kind)
    batch_shape = rhs.shape[:-1]
    b_kind = net.edge_index(kind).numel()
    q = torch.zeros(*batch_shape, b_kind, dtype=rhs.dtype, device=rhs.device)
    excess = rhs.clone()
    for child_nodes, edge_cols, signs, parent_nodes in levels:
        excess_child = excess[..., child_nodes]
        q_level = signs * excess_child
        q = q.scatter(-1, _expand_index(edge_cols, batch_shape), q_level)
        excess = excess.clone()
        excess.scatter_add_(-1, _expand_index(parent_nodes, batch_shape), excess_child)
    return q


def particular_flow(
    net: Network,
    sources: torch.Tensor,
    kind: str | None = None,
    *,
    atol: float = 1e-10,
) -> torch.Tensor:
    """Tree solution ``A @ q == sources`` on the spanning forest; zero on chord edges.

    ``sources`` has shape ``(..., n)`` and must sum to (near) zero within every
    connected component; ``q`` has shape ``(..., b_kind)``. Raises ``RuntimeError``
    naming the offending components if any component's sources do not sum to zero
    within ``atol``.
    """
    labels = net.component_labels(kind)
    n_components = net.n_components_of(kind)

    bad = []
    for c in range(n_components):
        node_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        total = sources[..., node_idx].sum(dim=-1)
        if torch.any(total.abs() > atol):
            bad.append(c)
    if bad:
        raise RuntimeError(
            f"sources do not sum to zero within atol={atol} on components {bad}; "
            "particular_flow requires a zero net source per connected component"
        )
    return _tree_solve(net, kind, sources)


def project_measured(
    net: Network,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str | None = None,
    sources: torch.Tensor | None = None,
    *,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Constrained least-squares flow: minimise ``||q - target||^2`` subject to
    ``A @ q == sources`` and ``q[mask] == target[mask]``.

    Solved as the KKT system ``[[I, C^T], [C, 0]] @ [q; mu] == [target; rhs]`` with
    ``C = [A_reduced; E_mask]``: ``A_reduced`` drops one row per connected component
    (the rows of ``A`` sum to zero within a component, so one is redundant) and
    ``E_mask`` selects the measured columns. Raises ``RuntimeError`` if the
    measurements and conservation cannot be satisfied simultaneously within ``atol``.
    """
    A = net.incidence(kind)
    n, b = A.shape
    dtype = target.dtype
    labels = net.component_labels(kind)
    n_components = net.n_components_of(kind)

    keep_rows: list[int] = []
    for c in range(n_components):
        node_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        keep_rows.extend(node_idx[1:].tolist())
    keep_rows_t = torch.tensor(sorted(keep_rows), dtype=torch.long)
    A_reduced = A[keep_rows_t].to(dtype)

    if sources is None:
        sources = torch.zeros(n, dtype=dtype)
    measured_idx = torch.nonzero(mask, as_tuple=False).flatten()
    m = measured_idx.numel()
    E = torch.zeros(m, b, dtype=dtype)
    E[torch.arange(m), measured_idx] = 1.0

    C = torch.cat([A_reduced, E], dim=0)  # (k, b)
    k = C.shape[0]

    batch_shape = torch.broadcast_shapes(target.shape[:-1], sources.shape[:-1])
    target_b = target.expand(*batch_shape, b)
    sources_reduced = sources[..., keep_rows_t].expand(*batch_shape, len(keep_rows_t))
    measured_b = target_b[..., measured_idx]
    lower = torch.cat([sources_reduced, measured_b], dim=-1)  # (..., k)
    rhs = torch.cat([target_b, lower], dim=-1)  # (..., b + k)

    top = torch.cat([torch.eye(b, dtype=dtype), C.T], dim=1)
    bottom = torch.cat([C, torch.zeros(k, k, dtype=dtype)], dim=1)
    K = torch.cat([top, bottom], dim=0)  # (b + k, b + k)

    try:
        solution = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError as exc:
        raise RuntimeError(
            "measured-flow constraints are infeasible: the KKT system is singular"
        ) from exc

    q = solution[..., :b]
    residual = torch.einsum("kb,...b->...k", C, q) - lower
    if torch.any(residual.abs() > atol):
        raise RuntimeError(
            f"measured-flow constraints are infeasible beyond atol={atol}: "
            f"max constraint residual {residual.abs().max().item():.3e}"
        )
    return q
