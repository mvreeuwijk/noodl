"""Tests for the per-instance SPD grounding certificate (milestone 1b design, section 3.1):
every interior node reaches a boundary node through a path of STRICTLY POSITIVE slopes, in
that instance. Unweighted connectivity is not sufficient -- this is exactly the defect the
design verifies against `PotentialFlowLayer._floating_group_nodes`'s blind spot (slopes
[[1, 1], [0, 1]] on a grounded three-node chain: instance 0 has minimum eigenvalue 0.382,
instance 1 has 0.0, and an unweighted connectivity check reports nothing wrong with either).
"""

from __future__ import annotations

import pytest
import torch

from tellegen.solvers.grounding import spd_certificate, spd_diagnosis


def _chain():
    """g (boundary) -> a -> b (both interior); src/tgt/interior_of_node/boundary_mask for it."""
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([-1, 0, 1])
    boundary_mask = torch.tensor([True, False, False])
    return src, tgt, interior_of_node, boundary_mask


def test_src_tgt_shape_mismatch_raises_valueerror():
    _, _, interior_of_node, boundary_mask = _chain()
    slopes = torch.tensor([1.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="src and tgt"):
        spd_certificate(
            torch.tensor([[0, 1]]), torch.tensor([1, 2]), slopes, interior_of_node,
            boundary_mask,
        )


def test_interior_of_node_and_boundary_mask_length_mismatch_raises_valueerror():
    src, tgt, _, _ = _chain()
    slopes = torch.tensor([1.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="boundary_mask"):
        spd_certificate(
            src, tgt, slopes, torch.tensor([-1, 0, 1]), torch.tensor([True, False]),
        )


def test_slopes_edge_count_mismatch_raises_valueerror():
    src, tgt, interior_of_node, boundary_mask = _chain()
    slopes = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)  # 3 edges, but src/tgt have 2
    with pytest.raises(ValueError, match="slopes"):
        spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)


def test_result_is_bool_with_the_batch_shape_of_slopes():
    src, tgt, interior_of_node, boundary_mask = _chain()
    slopes = torch.ones(6, 2, dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert result.shape == (6,)
    assert result.dtype == torch.bool


def test_no_interior_nodes_is_vacuously_true():
    src = torch.tensor([0])
    tgt = torch.tensor([1])
    interior_of_node = torch.tensor([-1, -1])  # both nodes are boundary
    boundary_mask = torch.tensor([True, True])
    slopes = torch.tensor([1.0], dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert bool(result)


def test_no_boundary_nodes_certifies_false():
    src = torch.tensor([0])
    tgt = torch.tensor([1])
    interior_of_node = torch.tensor([0, 1])  # both nodes are interior
    boundary_mask = torch.tensor([False, False])
    slopes = torch.tensor([1.0], dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert not bool(result)


def test_an_isolated_interior_component_certifies_false():
    # g (boundary) -- a (interior); b -- c (interior pair, isolated: no path to g at all).
    src = torch.tensor([0, 2])
    tgt = torch.tensor([1, 3])
    interior_of_node = torch.tensor([-1, 0, 1, 2])
    boundary_mask = torch.tensor([True, False, False, False])
    slopes = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert not bool(result)


def test_verified_counterexample_chain_certifies_true_then_false():
    """The exact case the design cites: slopes [[1, 1], [0, 1]] on g->a->b certify
    [True, False] -- unweighted connectivity alone (what `_floating_group_nodes` checked)
    would see BOTH instances as one connected component and report neither as floating.
    """
    src, tgt, interior_of_node, boundary_mask = _chain()
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert result.tolist() == [True, False]


def test_verified_counterexample_matches_explicit_jacobian_eigenvalues():
    """Cross-check against the actual mathematics, not just against the certificate's own
    logic: A_I diag(slopes) A_I^T for this chain has minimum eigenvalue ~0.382 for the
    grounded instance and exactly 0.0 for the ungrounded one.
    """
    A_I = torch.tensor([[-1.0, 1.0], [0.0, -1.0]], dtype=torch.float64)  # rows a, b
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    expected_min_eig = [0.38196601125010515, 0.0]
    for i in range(2):
        J = A_I @ torch.diag(slopes[i]) @ A_I.T
        min_eig = torch.linalg.eigvalsh(J).min().item()
        assert abs(min_eig - expected_min_eig[i]) < 1e-9

    src, tgt, interior_of_node, boundary_mask = _chain()
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert result.tolist() == [True, False]  # grounded iff minimum eigenvalue > 0


def test_fully_grounded_network_certifies_all_true():
    # 4-node star: node 0 boundary, nodes 1-3 interior, all directly grounded.
    src = torch.tensor([0, 0, 0])
    tgt = torch.tensor([1, 2, 3])
    interior_of_node = torch.tensor([-1, 0, 1, 2])
    boundary_mask = torch.tensor([True, False, False, False])
    slopes = torch.ones(5, 3, dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert bool(result.all())


def test_zero_slope_only_path_certifies_false():
    # Same chain as _chain(), but the only path from b to the boundary has slope exactly 0.
    src, tgt, interior_of_node, boundary_mask = _chain()
    slopes = torch.tensor([1.0, 0.0], dtype=torch.float64)
    result = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    assert not bool(result)


def test_atol_excludes_a_tiny_but_nonzero_slope():
    src, tgt, interior_of_node, boundary_mask = _chain()
    slopes = torch.tensor([1.0, 1e-9], dtype=torch.float64)
    assert bool(spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask, atol=0.0))
    assert not bool(
        spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask, atol=1e-6)
    )


def test_batching_matches_looped_single_instance_calls():
    src, tgt, interior_of_node, boundary_mask = _chain()
    torch.manual_seed(0)
    slopes = torch.rand(9, 2, dtype=torch.float64)
    batched = spd_certificate(src, tgt, slopes, interior_of_node, boundary_mask)
    looped = torch.stack(
        [
            spd_certificate(src, tgt, slopes[i], interior_of_node, boundary_mask)
            for i in range(9)
        ]
    )
    assert torch.equal(batched, looped)


def test_spd_diagnosis_names_negative_slope_edges():
    # g(boundary, node 0) -- a -- b ; edge 1 (a-b) negative in instance 1 only
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([-1, 0, 1])
    boundary_mask = torch.tensor([True, False, False])
    slopes = torch.tensor([[1.0, 1.0], [1.0, -0.5]], dtype=torch.float64)
    out = spd_diagnosis(src, tgt, slopes, interior_of_node, boundary_mask)
    assert out == [{"instance": 1, "reason": "negative_slope", "edges": [1], "nodes": [2]}]


def test_spd_diagnosis_names_ungrounded_nodes():
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([-1, 0, 1])
    boundary_mask = torch.tensor([True, False, False])
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)  # verified counterexample
    out = spd_diagnosis(src, tgt, slopes, interior_of_node, boundary_mask)
    assert out == [{"instance": 1, "reason": "ungrounded", "edges": [], "nodes": [1, 2]}]


def test_spd_diagnosis_is_empty_when_every_instance_certifies():
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([-1, 0, 1])
    boundary_mask = torch.tensor([True, False, False])
    slopes = torch.ones(3, 2, dtype=torch.float64)
    assert spd_diagnosis(src, tgt, slopes, interior_of_node, boundary_mask) == []
