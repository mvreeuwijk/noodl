import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tellegen.cycles import (
    assert_forward_oriented,
    branch_flows,
    particular_flow,
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


def test_particular_flow_on_a_path_graph_gives_unit_flow():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    sources = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    q = particular_flow(net, sources)
    assert torch.allclose(q, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(net.incidence() @ q, sources, atol=1e-12)


def test_particular_flow_raises_when_sources_do_not_sum_to_zero():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    with pytest.raises(RuntimeError, match="component"):
        particular_flow(net, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_particular_flow_is_zero_on_chord_edges(triangle):
    sources = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    q = particular_flow(triangle, sources)
    tree_cols, chord_cols = triangle.spanning_forest()
    assert torch.all(q[chord_cols] == 0)
    assert torch.allclose(triangle.incidence() @ q, sources, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_particular_flow_satisfies_conservation_on_random_trees(n, seed):
    rng = torch.Generator().manual_seed(seed)
    net = Network(dtype=torch.float64)
    for i in range(n):
        net.add_node(i)
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=rng))
        net.add_edge(i, j, kind="x")
    raw = torch.rand(n, generator=rng, dtype=torch.float64)
    sources = raw - raw.mean()  # zero-sum on the one component
    q = particular_flow(net, sources)
    assert q.shape == (net.b,)
    assert torch.allclose(net.incidence() @ q, sources, atol=1e-10)
