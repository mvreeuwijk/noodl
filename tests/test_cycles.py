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


def _two_disjoint_airpath_pairs_bridged_by_hydronic() -> Network:
    """4 nodes, two disconnected airpath edges (a-b, c-d) bridged by one hydronic edge.

    Whole-graph component count is 1 (connected via the bridge); the airpath-only
    subgraph has 2 components. This is the topology that exposed the kind-unaware
    component labelling bug.
    """
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("c", "d", kind="airpath")
    net.add_edge("b", "c", kind="hydronic")
    return net


def test_component_labels_kind_restricted_differs_from_whole_graph_on_bridged_network():
    net = _two_disjoint_airpath_pairs_bridged_by_hydronic()
    assert len(set(net.component_labels().tolist())) == 1
    labels_air = net.component_labels(kind="airpath")
    assert len(set(labels_air.tolist())) == 2
    assert labels_air[0] == labels_air[1]
    assert labels_air[2] == labels_air[3]
    assert labels_air[0] != labels_air[2]


def test_particular_flow_is_kind_aware_on_a_bridged_multi_kind_network():
    """Each airpath pair must solve as its own component, independent of the bridge."""
    net = _two_disjoint_airpath_pairs_bridged_by_hydronic()
    sources = torch.tensor([2.0, -2.0, 3.0, -3.0], dtype=torch.float64)
    q = particular_flow(net, sources, kind="airpath")  # must not raise
    assert torch.allclose(q, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)
    A_air = net.incidence(kind="airpath")
    assert torch.allclose(A_air @ q, sources, atol=1e-12)


def _two_rings_bridged_by_hydronic() -> Network:
    """Two independent airpath rings (each like `triangle`) joined by one hydronic edge.

    Whole-graph component count is 1; the airpath-only subgraph has 2 ring
    components, each with 1 cycle, so each ring's flows are only pinned down by
    measuring one edge in that ring (same mechanism as the `two_zone` fixture).
    """
    net = Network(dtype=torch.float64)
    for name in ("a1", "b1", "c1", "a2", "b2", "c2"):
        net.add_node(name)
    net.add_edge("a1", "b1", kind="airpath")
    net.add_edge("b1", "c1", kind="airpath")
    net.add_edge("c1", "a1", kind="airpath")
    net.add_edge("a2", "b2", kind="airpath")
    net.add_edge("b2", "c2", kind="airpath")
    net.add_edge("c2", "a2", kind="airpath")
    net.add_edge("c1", "a2", kind="hydronic")
    return net


def test_project_measured_is_kind_aware_and_does_not_raise_false_infeasible():
    net = _two_rings_bridged_by_hydronic()
    assert net.n_components == 1  # bridged into one whole-graph component
    assert net.n_components_of("airpath") == 2  # but two independent airpath rings
    mask = torch.tensor([True, False, False, True, False, False])
    # only indices 0 and 3 are real measurements; the rest are unused filler values
    target = torch.tensor([0.5, 3.0, -7.0, 0.9, 3.0, -7.0], dtype=torch.float64)
    q = project_measured(net, target, mask, kind="airpath")  # must not raise "infeasible"
    expected = torch.tensor([0.5, 0.5, 0.5, 0.9, 0.9, 0.9], dtype=torch.float64)
    assert torch.allclose(q, expected, atol=1e-8)
    A_air = net.incidence(kind="airpath")
    assert torch.allclose(A_air @ q, torch.zeros(6, dtype=torch.float64), atol=1e-8)


def test_project_measured_still_raises_on_genuinely_infeasible_multi_kind_measurements():
    net = _two_rings_bridged_by_hydronic()
    # two contradictory measurements within the SAME ring: still infeasible after the fix
    mask = torch.tensor([True, True, False, False, False, False])
    target = torch.tensor([0.5, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="infeasible"):
        project_measured(net, target, mask, kind="airpath")
