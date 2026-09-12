"""Batched linear solve with clearer error naming for singular systems."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(A, b)
    except (RuntimeError, torch._C._LinAlgError) as exc:
        # Re-solve each batch element on its own to find which one(s) triggered the
        # failure above. A determinant-threshold heuristic was tried first but is
        # dtype-unsafe: comparing against a fixed constant like 1e-300 silently
        # underflows to 0.0 when A is float32 (this project's default; float64 is
        # avoided elsewhere in the package as too costly on GPU), so `det < 1e-300`
        # is always False and no offender is ever reported. Retrying the identical
        # solve per element instead reuses torch's own singularity judgement, so it
        # is consistent with the failure above regardless of dtype.
        n = A.shape[-1]
        batch_shape = torch.broadcast_shapes(A.shape[:-2], b.shape[:-1])
        flat_A = A.expand(*batch_shape, n, n).reshape(-1, n, n)
        flat_b = b.expand(*batch_shape, n).reshape(-1, n)
        bad = []
        for i in range(flat_A.shape[0]):
            try:
                torch.linalg.solve(flat_A[i], flat_b[i])
            except (RuntimeError, torch._C._LinAlgError):
                bad.append(i)
        raise RuntimeError(f"singular system at batch indices {bad}") from exc


def floating_nodes(J: torch.Tensor, names: Sequence) -> list:
    flat = J.reshape(-1, J.shape[-2], J.shape[-1])
    all_zero_row = (flat == 0).all(dim=-1).any(dim=0)
    idx = torch.nonzero(all_zero_row, as_tuple=False).flatten().tolist()
    return [names[i] for i in idx]
