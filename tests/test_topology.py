"""Tests for the typed network topology: incidence, gradient, cycle basis, upwind, Tellegen."""

import networkx as nx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from noodl.topology import Network


def triangle() -> Network:
    """Three nodes, three directed edges forming one loop: 0->1, 1->2, 2->0."""
    net = Network()
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    net.add_edge("c", "a", kind="airpath")
    return net


def test_counts_on_triangle():
    net = triangle()
    assert net.n == 3
    assert net.b == 3
    assert net.n_components == 1
    assert net.n_cycles == 1  # b - n + components


def test_incidence_has_plus_one_at_source_and_minus_one_at_target():
    net = triangle()
    d = net.incidence()
    assert d.shape == (3, 3)
    # edge 0 is a->b: +1 at a (row 0), -1 at b (row 1)
    assert d[0, 0] == 1 and d[1, 0] == -1 and d[2, 0] == 0
    # edge 2 is c->a: +1 at c (row 2), -1 at a (row 0)
    assert d[2, 2] == 1 and d[0, 2] == -1 and d[1, 2] == 0


def test_incidence_columns_sum_to_zero():
    d = triangle().incidence()
    assert torch.all(d.sum(dim=0) == 0)


def test_gradient_is_negative_transpose_of_incidence():
    net = triangle()
    assert torch.equal(net.gradient(), -net.incidence().T)


def test_gradient_of_nodal_potential_gives_source_minus_target_difference():
    net = triangle()
    phi = torch.tensor([10.0, 4.0, 1.0])
    p = net.gradient() @ phi
    # edge a->b: phi_b - phi_a = -6 ; edge b->c: -3 ; edge c->a: +9
    assert torch.allclose(p, torch.tensor([-6.0, -3.0, 9.0]))


def test_difference_is_transpose_of_incidence():
    net = triangle()
    assert torch.equal(net.difference(), net.incidence().T)


def test_difference_is_the_negative_of_gradient():
    """`difference()` and `gradient()` are opposite sign
    conventions on the same quantity -- `difference()` is source minus target (what
    `PotentialFlowLayer.dp()` actually uses), `gradient()` is target minus source. They must
    be exact negatives of each other on every network, not just on the hand-checked case
    below.
    """
    net = triangle()
    assert torch.equal(net.difference(), -net.gradient())


def test_difference_of_nodal_potential_gives_source_minus_target_difference():
    """Hand-checked graph: triangle a->b->c->a with phi = [10, 4, 1] (a, b, c).

    For each directed edge (source, target), `difference() @ phi` must equal
    `phi[source] - phi[target]`: edge a->b gives phi_a - phi_b = 6; edge b->c gives
    phi_b - phi_c = 3; edge c->a gives phi_c - phi_a = -9. This is the exact negative of
    `test_gradient_of_nodal_potential_gives_source_minus_target_difference` above (which,
    despite its name, computes target minus source -- see `gradient()`'s docstring).
    """
    net = triangle()
    phi = torch.tensor([10.0, 4.0, 1.0])
    p = net.difference() @ phi
    assert torch.allclose(p, torch.tensor([6.0, 3.0, -9.0]))


def test_cycle_basis_spans_nullspace_of_incidence():
    net = triangle()
    J = net.cycle_basis()
    assert J.shape == (1, 3)
    assert torch.all(net.incidence() @ J.T == 0)
    assert torch.linalg.matrix_rank(J) == 1


def test_noodl_power_residual_is_zero_for_consistent_potentials_and_flows():
    net = triangle()
    phi = torch.tensor([3.0, -1.0, 7.0])
    p = net.gradient() @ phi
    q = net.cycle_basis().T @ torch.tensor([2.5])
    assert abs(net.power_residual(p, q).item()) < 1e-6


def test_power_residual_is_nonzero_when_flow_violates_conservation():
    net = triangle()
    phi = torch.tensor([3.0, -1.0, 7.0])
    p = net.gradient() @ phi
    q = torch.tensor([1.0, 0.0, 0.0])  # not divergence free
    assert abs(net.power_residual(p, q).item()) > 1e-3


def test_upwind_selects_upstream_node_value():
    net = triangle()
    phi = torch.tensor([10.0, 20.0, 30.0])  # a, b, c
    q = torch.tensor([1.0, -1.0, 0.0])
    up = net.upwind(q) @ phi
    # edge a->b with q>0: upstream is a (10); edge b->c with q<0: upstream is c (30)
    assert up[0] == 10.0
    assert up[1] == 30.0
    # zero flow defaults to the source node
    assert up[2] == 30.0


def test_edges_are_typed_and_can_be_selected_by_kind():
    net = Network()
    net.add_node("room")
    net.add_node("outside")
    net.add_edge("outside", "room", kind="airpath")
    net.add_edge("room", "outside", kind="conduction")
    assert net.edge_index("airpath").tolist() == [0]
    assert net.edge_index("conduction").tolist() == [1]
    d_air = net.incidence(kind="airpath")
    assert d_air.shape == (2, 1)
    assert d_air[0, 0] == -1  # flow into room (row 0) is -1 at room
    assert d_air[1, 0] == 1  # and +1 at outside (row 1)


def test_parallel_edges_are_distinct_branches():
    net = Network()
    net.add_node("a")
    net.add_node("b")
    k1 = net.add_edge("a", "b", kind="airpath")
    k2 = net.add_edge("a", "b", kind="airpath")
    assert k1 != k2
    assert net.b == 2
    J = net.cycle_basis()
    assert J.shape == (1, 2)
    assert torch.all(net.incidence() @ J.T == 0)


def test_with_ambient_adds_reference_node_joined_to_every_node():
    net = triangle()
    amb = net.with_ambient(name="ambient", kind="storage")
    assert amb.n == 4
    assert amb.b == 6
    assert "ambient" in amb.nodes
    assert amb.edge_index("storage").tolist() == [3, 4, 5]
    # original network is untouched
    assert net.n == 3 and net.b == 3


def test_unknown_node_in_edge_raises():
    net = Network()
    net.add_node("a")
    with pytest.raises(KeyError):
        net.add_edge("a", "zzz", kind="airpath")


def test_disconnected_graph_has_cycle_count_per_component():
    net = Network()
    for name in "abcdef":
        net.add_node(name)
    net.add_edge("a", "b", kind="x")
    net.add_edge("b", "c", kind="x")
    net.add_edge("c", "a", kind="x")  # component 1: one cycle
    net.add_edge("d", "e", kind="x")
    net.add_edge("e", "f", kind="x")  # component 2: a tree
    assert net.n_components == 2
    assert net.n_cycles == 1
    J = net.cycle_basis()
    assert J.shape == (1, 5)
    assert torch.all(net.incidence() @ J.T == 0)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=9),
    extra=st.integers(min_value=0, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cycle_basis_is_divergence_free_on_random_connected_multigraphs(n, extra, seed):
    rng = torch.Generator().manual_seed(seed)
    net = Network()
    for i in range(n):
        net.add_node(i)
    # random spanning tree first so the graph is connected
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=rng))
        net.add_edge(i, j, kind="x")
    for _ in range(extra):
        u = int(torch.randint(0, n, (1,), generator=rng))
        v = int(torch.randint(0, n, (1,), generator=rng))
        if u != v:
            net.add_edge(u, v, kind="x")
    J = net.cycle_basis()
    assert J.shape[0] == net.n_cycles == net.b - net.n + 1
    assert torch.all(net.incidence() @ J.T == 0)
    if net.n_cycles:
        assert torch.linalg.matrix_rank(J) == net.n_cycles


def test_network_exposes_underlying_networkx_graph():
    net = triangle()
    assert isinstance(net.graph, nx.MultiDiGraph)


def test_incidence_and_edge_index_return_the_same_cached_tensor_object():
    net = triangle()
    first = net.incidence()
    second = net.incidence()
    assert first is second
    cols_first = net.edge_index("airpath")
    cols_second = net.edge_index("airpath")
    assert cols_first is cols_second


def test_add_edge_clears_the_incidence_cache():
    net = triangle()
    before = net.incidence()
    net.add_node("d")
    net.add_edge("a", "d", kind="airpath")
    after = net.incidence()
    assert after is not before
    assert after.shape == (4, 4)


def test_node_attr_returns_values_in_node_order():
    net = Network()
    net.add_node("a", elevation=1.0)
    net.add_node("b", elevation=2.5)
    net.add_node("c", elevation=-3.0)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    z = net.node_attr("elevation")
    assert z.shape == (3,)
    assert torch.allclose(z, torch.tensor([1.0, 2.5, -3.0]))


def test_node_attr_raises_keyerror_for_missing_attribute():
    net = Network()
    net.add_node("a", elevation=1.0)
    net.add_node("b")
    with pytest.raises(KeyError, match="b"):
        net.node_attr("elevation")


def test_node_attr_default_fills_missing_values():
    net = Network()
    net.add_node("a", elevation=1.0)
    net.add_node("b")
    z = net.node_attr("elevation", default=0.0)
    assert torch.allclose(z, torch.tensor([1.0, 0.0]))


def test_edge_attr_returns_values_in_branch_order_for_one_kind():
    net = Network()
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath", area=0.02)
    net.add_edge("b", "a", kind="conduction", ua=5.0)
    net.add_edge("a", "b", kind="airpath", area=0.05)
    areas = net.edge_attr("area", kind="airpath")
    assert torch.allclose(areas, torch.tensor([0.02, 0.05]))


def test_node_index_returns_position_and_raises_for_unknown_node():
    net = triangle()
    assert net.node_index("a") == 0
    assert net.node_index("c") == 2
    with pytest.raises(KeyError):
        net.node_index("z")


def test_interior_and_boundary_index_partition_all_nodes():
    net = triangle()  # nodes a, b, c
    interior = net.interior_index(["b"])
    boundary = net.boundary_index(["b"])
    assert interior.tolist() == [0, 2]  # a, c in node order
    assert boundary.tolist() == [1]


def test_boundary_index_preserves_the_given_order():
    net = triangle()
    boundary = net.boundary_index(["c", "a"])
    assert boundary.tolist() == [2, 0]  # order of the argument, not node order


def test_boundary_index_raises_for_unknown_node():
    net = triangle()
    with pytest.raises(KeyError):
        net.boundary_index(["z"])


def test_component_labels_same_label_within_a_component():
    net = triangle()
    labels = net.component_labels()
    assert labels.shape == (3,)
    assert labels[0] == labels[1] == labels[2]


def test_component_labels_different_labels_across_components():
    net = Network()
    for name in "abcdef":
        net.add_node(name)
    net.add_edge("a", "b", kind="x")
    net.add_edge("b", "c", kind="x")
    net.add_edge("c", "a", kind="x")
    net.add_edge("d", "e", kind="x")
    net.add_edge("e", "f", kind="x")
    labels = net.component_labels()
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert set(labels.tolist()) == {0, 1}


def test_component_labels_kind_restricted_can_have_more_components_than_whole_graph():
    """a-b and c-d are disconnected in the airpath subgraph but bridged by hydronic."""
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("c", "d", kind="airpath")
    net.add_edge("b", "c", kind="hydronic")
    labels_air = net.component_labels(kind="airpath")
    labels_all = net.component_labels()
    assert labels_all[0] == labels_all[1] == labels_all[2] == labels_all[3]
    assert labels_air[0] == labels_air[1]
    assert labels_air[2] == labels_air[3]
    assert labels_air[0] != labels_air[2]


def test_n_components_of_is_kind_aware_and_leaves_n_components_property_unchanged():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("c", "d", kind="airpath")
    net.add_edge("b", "c", kind="hydronic")
    assert net.n_components_of("airpath") == 2
    assert net.n_components_of() == 1
    assert net.n_components == 1


def test_component_labels_raises_keyerror_for_unknown_kind():
    net = triangle()
    with pytest.raises(KeyError, match="airpaths"):
        net.component_labels("airpaths")


def test_source_and_target_selector_are_onehot_and_kind_filtered():
    net = Network()
    net.add_node("room")
    net.add_node("outside")
    net.add_edge("outside", "room", kind="airpath")
    net.add_edge("room", "outside", kind="conduction")
    S = net.source_selector("airpath")
    T = net.target_selector("airpath")
    assert S.shape == (1, 2) and T.shape == (1, 2)
    assert S.tolist() == [[0.0, 1.0]]  # source is "outside" (index 1)
    assert T.tolist() == [[1.0, 0.0]]  # target is "room" (index 0)


def test_source_selector_minus_target_selector_recovers_incidence():
    net = triangle()
    S = net.source_selector()
    T = net.target_selector()
    assert torch.equal((S - T).T, net.incidence())


def _random_connected_multigraph(n: int, extra: int, seed: int, dtype=torch.float64) -> Network:
    """Random connected multigraph: a random spanning tree plus `extra` random edges."""
    rng = torch.Generator().manual_seed(seed)
    net = Network(dtype=dtype)
    for i in range(n):
        net.add_node(i)
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=rng))
        net.add_edge(i, j, kind="x")
    for _ in range(extra):
        u = int(torch.randint(0, n, (1,), generator=rng))
        v = int(torch.randint(0, n, (1,), generator=rng))
        if u != v:
            net.add_edge(u, v, kind="x")
    return net


def test_upwind_unbatched_matches_original_edge_by_edge_selection():
    net = triangle()
    phi = torch.tensor([10.0, 20.0, 30.0])
    q = torch.tensor([1.0, -1.0, 0.0])
    up = net.upwind(q) @ phi
    assert up[0] == 10.0
    assert up[1] == 30.0
    assert up[2] == 30.0  # zero flow defaults to the source


def test_downwind_is_the_complement_of_upwind():
    net = triangle()
    q = torch.tensor([1.0, -1.0, 0.3])
    up = net.upwind(q)
    down = net.downwind(q)
    assert torch.equal(up + down, net.source_selector() + net.target_selector())
    assert torch.all((up + down).sum(dim=-1) == 2)  # each edge marks source and target


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    extra=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
    batch=st.integers(min_value=1, max_value=4),
)
def test_upwind_rows_are_one_hot_at_the_upstream_node_for_random_signed_batched_q(
    n, extra, seed, batch
):
    net = _random_connected_multigraph(n, extra, seed)
    torch.manual_seed(seed)
    q = torch.rand(batch, net.b, dtype=torch.float64) - 0.5
    up = net.upwind(q)
    assert up.shape == (batch, net.b, net.n)
    assert torch.all(up.sum(dim=-1) == 1)  # one-hot per edge
    S = net.source_selector()
    T = net.target_selector()
    expected = torch.where((q >= 0).unsqueeze(-1), S, T)
    assert torch.equal(up, expected)


def test_spanning_forest_partitions_columns_into_tree_and_chords():
    net = triangle()
    tree_cols, chord_cols = net.spanning_forest()
    assert tree_cols.tolist() == [0, 1]  # a->b, b->c form the tree
    assert chord_cols.tolist() == [2]  # c->a closes the loop


def test_cycle_basis_matches_spanning_forest_tree_edges():
    net = triangle()
    tree_cols, chord_cols = net.spanning_forest()
    J = net.cycle_basis()
    assert J.shape == (chord_cols.numel(), net.b)
    for r, j in enumerate(chord_cols.tolist()):
        assert J[r, j] == 1  # each chord row has a unit entry on its own column
    assert torch.all(net.incidence() @ J.T == 0)


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    extra=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_incidence_tree_columns_has_rank_n_minus_components(n, extra, seed):
    net = _random_connected_multigraph(n, extra, seed)
    tree_cols, chord_cols = net.spanning_forest()
    A = net.incidence()
    assert tree_cols.numel() + chord_cols.numel() == net.b
    assert torch.linalg.matrix_rank(A[:, tree_cols]) == net.n - net.n_components


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    extra=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_spanning_forest_tree_and_chords_cover_all_columns_exactly_once(n, extra, seed):
    net = _random_connected_multigraph(n, extra, seed)
    tree_cols, chord_cols = net.spanning_forest()
    all_cols = torch.cat([tree_cols, chord_cols]).sort().values
    assert torch.equal(all_cols, torch.arange(net.b, dtype=torch.long))


def test_to_changes_dtype_and_clears_the_cache():
    net = triangle()
    before = net.incidence()
    assert before.dtype == torch.float32
    result = net.to(dtype=torch.float64)
    assert result is net
    after = net.incidence()
    assert after.dtype == torch.float64
    assert after is not before


@settings(max_examples=20, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    extra=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_to_float64_changes_tensor_dtypes_on_random_connected_multigraphs(n, extra, seed):
    net = _random_connected_multigraph(n, extra, seed, dtype=torch.float32)
    assert net.incidence().dtype == torch.float32
    net.to(dtype=torch.float64)
    assert net.incidence().dtype == torch.float64
    assert net.source_selector().dtype == torch.float64
    assert net.cycle_basis().dtype == torch.float64


def test_edge_index_raises_keyerror_for_unknown_kind():
    net = triangle()
    with pytest.raises(KeyError, match="airpaths"):
        net.edge_index("airpaths")


def test_edge_index_kind_none_returns_all_edges_when_all_edges_share_one_kind():
    net = triangle()
    assert net.edge_index(None).tolist() == [0, 1, 2]


def test_edge_index_kind_none_on_empty_network_returns_empty_tensor():
    net = Network()
    net.add_node("a")
    assert net.edge_index(None).tolist() == []


def test_incidence_raises_keyerror_for_unknown_kind():
    net = triangle()
    with pytest.raises(KeyError, match="airpaths"):
        net.incidence("airpaths")


def test_source_selector_raises_keyerror_for_unknown_kind():
    net = triangle()
    with pytest.raises(KeyError, match="airpaths"):
        net.source_selector("airpaths")


def test_edge_attr_raises_keyerror_for_unknown_kind():
    net = Network()
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath", area=0.02)
    with pytest.raises(KeyError, match="airpaths"):
        net.edge_attr("area", kind="airpaths")


def test_upwind_raises_keyerror_for_unknown_kind():
    net = triangle()
    q = torch.tensor([1.0, -1.0, 0.0])
    with pytest.raises(KeyError, match="airpaths"):
        net.upwind(q, kind="airpaths")


def test_with_ambient_preserves_dtype_and_device():
    net = triangle()
    net.to(device=torch.device("meta"), dtype=torch.float64)
    amb = net.with_ambient()
    assert amb.dtype == torch.float64
    assert amb.device == torch.device("meta")
