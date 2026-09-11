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
