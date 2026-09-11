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
    A = net.incidence(kind)
    tree_cols, chord_cols = net.spanning_forest(kind)
    labels = net.component_labels()
    n_components = net.n_components

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

    col_component = (net.source_selector(kind) @ labels.to(A.dtype)).round().long()
    batch_shape = sources.shape[:-1]
    q = torch.zeros(*batch_shape, A.shape[1], dtype=sources.dtype)
    for c in range(n_components):
        node_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        if node_idx.numel() <= 1:
            continue  # isolated node: no tree edges, nothing to solve
        rest = node_idx[1:]  # drop one reference node per component
        tree_cols_c = tree_cols[col_component[tree_cols] == c]
        A_c = A[rest][:, tree_cols_c]
        rhs_c = sources[..., rest]
        q_tree_c = torch.linalg.solve(A_c, rhs_c.unsqueeze(-1)).squeeze(-1)
        q[..., tree_cols_c] = q_tree_c
    return q
