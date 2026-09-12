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

from tellegen.solvers.grounding import spd_certificate


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
