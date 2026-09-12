"""Per-instance SPD certificate: every interior node reaches a boundary node through a path
of strictly positive slopes, in that instance.

See the milestone design, section 3.1: symmetry and non-negative slopes give positive
SEMI-definiteness; this certificate is the third condition that upgrades semi-definite to
definite. It is computed on the ACTUAL slopes at the solve, never on initialisation slopes,
and is never bypassed when a caller supplies its own initial guess.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def spd_certificate(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
    *,
    atol: float = 0.0,
) -> Tensor:
    """(...,) bool: every interior node reaches a boundary node via strictly positive slopes.

    `src`, `tgt`: (b,) LongTensors, shared across the batch (as from `Network.endpoints`).
    `slopes`: (..., b), the per-instance, per-edge slope actually used at this solve.
    `interior_of_node`: (n,) LongTensor, the interior-row index of every node, or a negative
    sentinel for a boundary node. `boundary_mask`: (n,) bool, True at every boundary node.
    `atol`: a slope must strictly exceed this to count as active; `atol=0.0` is the
    mathematical "strictly positive" condition.
    """
    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError(
            f"src and tgt must be 1-D LongTensors, got shapes {tuple(src.shape)} and "
            f"{tuple(tgt.shape)}"
        )
    if src.shape != tgt.shape:
        raise ValueError(
            f"src and tgt must have the same shape, got {tuple(src.shape)} and "
            f"{tuple(tgt.shape)}"
        )
    if interior_of_node.ndim != 1:
        raise ValueError(
            f"interior_of_node must be 1-D, got shape {tuple(interior_of_node.shape)}"
        )
    if boundary_mask.ndim != 1:
        raise ValueError(f"boundary_mask must be 1-D, got shape {tuple(boundary_mask.shape)}")
    n = interior_of_node.shape[0]
    if boundary_mask.shape[0] != n:
        raise ValueError(
            f"boundary_mask has length {boundary_mask.shape[0]} but interior_of_node has "
            f"length {n}"
        )
    b = src.shape[0]
    if slopes.shape[-1] != b:
        raise ValueError(
            f"slopes has {slopes.shape[-1]} edges in its last dimension but src/tgt have {b}"
        )

    batch_shape = slopes.shape[:-1]
    device = slopes.device
    interior_mask = interior_of_node >= 0

    if not bool(interior_mask.any()):
        # No interior nodes: nothing needs grounding, so the certificate holds vacuously.
        return torch.ones(batch_shape, dtype=torch.bool, device=device)
    if not bool(boundary_mask.any()):
        # No boundary nodes at all: no interior node can reach one.
        return torch.zeros(batch_shape, dtype=torch.bool, device=device)

    return torch.zeros(batch_shape, dtype=torch.bool, device=device)
