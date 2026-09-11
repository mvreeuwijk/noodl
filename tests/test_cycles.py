import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tellegen.cycles import (
    assert_forward_oriented,
    branch_flows,
    particular_flow,
    project_measured,
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


def test_project_measured_matches_measured_branches_exactly_and_conserves_flow(two_zone):
    mask = torch.tensor([True, False, False])
    target = torch.tensor([0.5, 3.0, -7.0], dtype=torch.float64)  # only index 0 is real
    q = project_measured(two_zone, target, mask)
    assert torch.allclose(q[0], torch.tensor(0.5, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(two_zone.incidence() @ q, torch.zeros(3, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(q, torch.full((3,), 0.5, dtype=torch.float64), atol=1e-8)


def test_project_measured_equals_plain_projection_when_mask_is_all_false(two_zone):
    mask = torch.zeros(3, dtype=torch.bool)
    target = torch.tensor([0.5, 0.2, 0.4], dtype=torch.float64)
    q = project_measured(two_zone, target, mask)

    A_reduced = two_zone.incidence()[1:]  # drop the "ambient" row (one component)
    rhs = A_reduced @ target
    mu = torch.linalg.solve(A_reduced @ A_reduced.T, rhs)
    expected = target - A_reduced.T @ mu

    assert torch.allclose(two_zone.incidence() @ q, torch.zeros(3, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(q, expected, atol=1e-8)


def test_project_measured_batched_equals_looped(two_zone):
    mask = torch.tensor([True, False, False])
    target = torch.stack(
        [
            torch.tensor([0.5, 0.1, 0.2], dtype=torch.float64),
            torch.tensor([0.5, -0.3, 0.4], dtype=torch.float64),
            torch.tensor([0.5, 0.9, -0.9], dtype=torch.float64),
        ]
    )
    batched = project_measured(two_zone, target, mask)
    looped = torch.stack([project_measured(two_zone, target[i], mask) for i in range(3)])
    assert batched.shape == (3, 3)
    assert torch.allclose(batched, looped, atol=1e-10)


def test_project_measured_raises_on_infeasible_measurements(two_zone):
    mask = torch.tensor([True, True, False])
    target = torch.tensor([0.5, 0.9, 0.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="infeasible"):
        project_measured(two_zone, target, mask)
