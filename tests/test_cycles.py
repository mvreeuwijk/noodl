import torch

from tellegen.cycles import (
    assert_forward_oriented,
    branch_flows,
)
from tellegen.topology import Network


def test_branch_flows_is_importable_from_cycles_and_matches_physics_wrapper():
    from tellegen.physics import branch_flows as physics_branch_flows

    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "a", kind="airpath")
    m = torch.tensor([0.3], dtype=torch.float64)
    assert torch.equal(branch_flows(net, m), physics_branch_flows(net, m))


def test_assert_forward_oriented_is_importable_from_cycles():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")  # tree
    net.add_edge("b", "a", kind="airpath")  # loop closes forward
    assert_forward_oriented(net)  # does not raise
