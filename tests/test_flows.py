import pytest
import torch

from tellegen.physics import assert_forward_oriented, branch_flows
from tellegen.topology import Network


def two_loop_floor() -> Network:
    """ambient <-> floor with exhaust (tree), supply and infiltration (two loops)."""
    net = Network()
    net.add_node("ambient")
    net.add_node("F")
    net.add_edge("F", "ambient", kind="airpath")  # exhaust: tree edge
    net.add_edge("ambient", "F", kind="airpath")  # supply: loop 1
    net.add_edge("ambient", "F", kind="airpath")  # infiltration: loop 2
    return net


def test_branch_flows_are_divergence_free():
    net = two_loop_floor()
    q = branch_flows(net, torch.tensor([0.6, 0.02]))
    assert q.shape == (3,)
    assert torch.allclose(net.incidence() @ q, torch.zeros(2), atol=1e-6)


def test_exhaust_carries_sum_of_supply_and_infiltration():
    net = two_loop_floor()
    q = branch_flows(net, torch.tensor([0.6, 0.02]))
    assert torch.allclose(q, torch.tensor([0.62, 0.6, 0.02]))


def test_branch_flows_broadcast_over_leading_dimensions():
    net = two_loop_floor()
    m = torch.rand(5, 4, 2)
    q = branch_flows(net, m)
    assert q.shape == (5, 4, 3)
    div = torch.einsum("nb,...b->...n", net.incidence(), q)
    assert torch.allclose(div, torch.zeros(5, 4, 2), atol=1e-6)


def test_forward_oriented_graph_passes_check():
    assert_forward_oriented(two_loop_floor())


def test_backward_edge_fails_orientation_check():
    net = Network()
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")  # tree
    net.add_edge("a", "b", kind="airpath")  # loop closes against the tree edge: -1 entry
    with pytest.raises(ValueError, match="orient"):
        assert_forward_oriented(net)
