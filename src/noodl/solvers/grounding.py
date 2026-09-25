"""Per-instance SPD certificate: every branch slope is non-negative, AND every interior node
reaches a boundary node through a path of strictly positive slopes, in that instance.

The certificate has three conditions, all per instance: symmetry (structural here -- `A_I diag(g)
A_I^T` is symmetric for any `g`), non-negative slopes (which with symmetry gives positive
SEMI-definiteness), and grounding through strictly positive slopes (which upgrades semi-definite to
definite). Both testable conditions are applied, by `_certified`, to `spd_certificate` and
`spd_diagnosis` alike: see `_certified` for why grounding alone is not sufficient and not
conservative. The certificate is computed on the ACTUAL slopes at the solve, never on initialisation
slopes, and is never bypassed when a caller supplies its own initial guess.

Implementation: label propagation, vectorised over the batch (no Python loop over batch
instances). A node is "grounded" once it is a boundary node or is connected, through some
chain of edges each with slope strictly greater than `atol`, to a node that is already
grounded. An active edge connects its two endpoints UNDIRECTED for this purpose (a
resistor's conductance, unlike its flow, has no preferred direction). Each round is
O(B * E) (one gather and two scatters over the edge list, batched); the worst case over all
rounds is O(B * E * diameter), and the round
count is bounded by the number of nodes regardless of graph shape.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _validate(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
) -> None:
    """Shared argument validation for `spd_certificate` and `spd_diagnosis`."""
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


def _grounded(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
    *,
    atol: float = 0.0,
    extra_grounded: Tensor | None = None,
) -> Tensor:
    """(..., n) bool: every node's grounded status, by label propagation over active edges.

    Assumes `_validate` has already been called. A node is grounded if it is a boundary node,
    if `extra_grounded` marks it grounded directly (a potential-dependent NODAL source
    is itself a virtual connection to ground -- its positive diagonal shift makes
    the operator SPD at that node independent of any edge path), or if it is reachable from a
    grounded node through a chain of edges each with slope strictly greater than `atol` (an
    active edge connects its endpoints undirected). `extra_grounded`, when given, is
    `(..., n)` bool and may be batched independently of `boundary_mask` (a node source's
    slope is a function of the solve point and so can differ per instance, unlike a
    boundary node's fixed set).
    """
    n = interior_of_node.shape[0]
    b = src.shape[0]
    batch_shape = torch.broadcast_shapes(
        slopes.shape[:-1], () if extra_grounded is None else extra_grounded.shape[:-1]
    )
    device = slopes.device
    interior_mask = interior_of_node >= 0

    seed = boundary_mask.to(device=device).expand(batch_shape + (n,))
    if extra_grounded is not None:
        seed = seed | extra_grounded.to(device=device, dtype=torch.bool).expand(batch_shape + (n,))

    if not bool(interior_mask.any()):
        # No interior nodes: nothing needs grounding; the caller's all(dim=-1) over an empty
        # interior selection is vacuously True regardless of this mask's actual values.
        return seed
    if not bool(seed.any()):
        # No boundary node AND no node-source-grounded node at all: no interior node can
        # reach ground.
        return torch.zeros(batch_shape + (n,), dtype=torch.bool, device=device)

    grounded = seed.clone()
    active = slopes > atol
    src_idx = src.to(device=device).expand(batch_shape + (b,))
    tgt_idx = tgt.to(device=device).expand(batch_shape + (b,))

    # Grounded status propagates at most one edge further per round, so it is fully settled,
    # for ANY graph and ANY batch instance, within `n` rounds (the longest possible simple
    # path); a round that changes nothing means every later round would too, hence the early
    # break. Vectorised over the batch throughout -- no Python loop over batch instances.
    for _ in range(n):
        src_grounded = torch.gather(grounded, -1, src_idx)
        tgt_grounded = torch.gather(grounded, -1, tgt_idx)
        propagate = active & (src_grounded | tgt_grounded)
        updated = grounded.scatter_reduce(-1, tgt_idx, propagate, reduce="amax")
        updated = updated.scatter_reduce(-1, src_idx, propagate, reduce="amax")
        if torch.equal(updated, grounded):
            break
        grounded = updated

    return grounded


def _certified(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
    *,
    atol: float = 0.0,
    extra_grounded: Tensor | None = None,
) -> Tensor:
    """(...,) bool: the module docstring's conditions 2 AND 3, per instance.

    Condition 1 (symmetry) is structural -- `A_I diag(g) A_I^T` is symmetric for any `g` --
    so it needs no per-instance test. Condition 2 is `g >= 0` on EVERY branch, which with
    symmetry gives positive SEMI-definiteness; condition 3 is grounding through strictly
    positive slopes, which upgrades semi-definite to definite. Testing condition 3 alone is
    not sufficient and is not conservative: a negative slope makes an edge merely INACTIVE
    for `_grounded`'s propagation, so an instance grounded through other, positive edges
    passed condition 3 while its assembled operator was indefinite (verified: a grounded
    3-node chain with a parallel edge of slope -5 assembles to eigenvalues [-7.53, 0.53]).
    `method="auto"` would then dispatch a definiteness-assuming backend to an indefinite
    system. On the PCG path (which "auto" still takes when no sparse form is available)
    CG's convergence test uses the recursively updated residual, so such a system can report
    CONVERGED on a drifted residual rather than the loud BREAKDOWN it happens to give in
    that example; on the sparse-direct path the certificate is what keeps a matrix SuperLU
    would factor into nonsense from being sent there without a word.

    Note the deliberate asymmetry between this test and `_grounded`'s: `g >= 0` here
    (non-negativity is what semi-definiteness needs, and a zero slope is a legitimately
    open-circuit branch), `slopes > atol` there (only a STRICTLY positive slope actually
    conducts a ground). Assumes `_validate` has already been called.
    """
    interior_mask = interior_of_node >= 0
    grounded = _grounded(
        src, tgt, slopes, interior_of_node, boundary_mask, atol=atol,
        extra_grounded=extra_grounded,
    )
    return (slopes >= 0).all(dim=-1) & grounded[..., interior_mask].all(dim=-1)


def spd_certificate(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
    *,
    atol: float = 0.0,
    extra_grounded: Tensor | None = None,
) -> Tensor:
    """(...,) bool: per instance, every branch slope is non-negative AND every interior node
    reaches ground -- either a boundary node via strictly positive slopes, or a node
    `extra_grounded` marks directly grounded (the module docstring's conditions 2 and 3;
    condition 1, symmetry, is structural).

    `src`, `tgt`: (b,) LongTensors, shared across the batch (as from `Network.endpoints`).
    `slopes`: (..., b), the per-instance, per-edge slope actually used at this solve.
    `interior_of_node`: (n,) LongTensor, the interior-row index of every node, or a negative
    sentinel for a boundary node. `boundary_mask`: (n,) bool, True at every boundary node.
    `atol`: a slope must strictly exceed this to count as active; `atol=0.0` is the
    mathematical "strictly positive" condition. `extra_grounded`: `(..., n)` bool,
    additional per-instance nodes considered grounded independent of any edge path -- a
    potential-dependent nodal source's positive diagonal shift is exactly such a
    connection. `None` (the default) changes nothing for a layer with no node sources.
    """
    _validate(src, tgt, slopes, interior_of_node, boundary_mask)
    return _certified(
        src, tgt, slopes, interior_of_node, boundary_mask, atol=atol,
        extra_grounded=extra_grounded,
    )


def spd_diagnosis(
    src: Tensor,
    tgt: Tensor,
    slopes: Tensor,
    interior_of_node: Tensor,
    boundary_mask: Tensor,
    *,
    atol: float = 0.0,
    extra_grounded: Tensor | None = None,
) -> list[dict]:
    """Per FAILING instance (flat batch index), in ascending instance order:
        {"instance": int,
         "reason": "negative_slope" | "ungrounded",
         "edges": list[int],   # edge indices with slope < 0 in this instance (may be empty)
         "nodes": list[int]}   # NODE indices (not interior rows) of ungrounded interior nodes
    reason is "negative_slope" if any edge has slope < 0, else "ungrounded". Both lists are
    always present. Called ONLY on the failure path; a Python loop over failing instances is
    acceptable here and nowhere else in grounding.py. `extra_grounded`: see `spd_certificate`.
    """
    _validate(src, tgt, slopes, interior_of_node, boundary_mask)
    interior_mask = interior_of_node >= 0
    node_indices = torch.arange(interior_of_node.shape[0])

    grounded = _grounded(
        src, tgt, slopes, interior_of_node, boundary_mask, atol=atol,
        extra_grounded=extra_grounded,
    )
    # The SAME condition `spd_certificate` applies, so an instance the certificate refuses
    # always has a record here (a diagnosis keyed on grounding alone would produce no
    # record for a grounded instance with a negative slope, leaving select.solve's
    # negative-slope message branch unreachable).
    certified = (slopes >= 0).all(dim=-1) & grounded[..., interior_mask].all(dim=-1)
    failing = ~certified

    b = src.shape[0]
    # `grounded`'s own batch shape is the authority (it may be broader than `slopes`'s when
    # `extra_grounded` carries a leading batch `slopes` does not, e.g. a per-instance node
    # source slope over an otherwise unbatched edge set): expand `slopes` to match before
    # flattening, rather than assume the two already agree.
    batch_shape = failing.shape
    flat_slopes = slopes.expand(batch_shape + (b,)).reshape(-1, b)
    flat_grounded = grounded.reshape(-1, grounded.shape[-1])
    flat_failing = failing.reshape(-1)

    records = []
    for idx in torch.nonzero(flat_failing, as_tuple=False).flatten().tolist():
        inst_slopes = flat_slopes[idx]
        edges = torch.nonzero(inst_slopes < 0, as_tuple=False).flatten().tolist()
        reason = "negative_slope" if edges else "ungrounded"
        inst_grounded = flat_grounded[idx]
        nodes = node_indices[interior_mask & ~inst_grounded].tolist()
        records.append({"instance": idx, "reason": reason, "edges": edges, "nodes": nodes})
    return records
