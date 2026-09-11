"""Tests for the typed network topology: incidence, gradient, cycle basis, upwind, Tellegen."""

import networkx as nx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tellegen.topology import Network


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


def test_cycle_basis_spans_nullspace_of_incidence():
    net = triangle()
    J = net.cycle_basis()
    assert J.shape == (1, 3)
    assert torch.all(net.incidence() @ J.T == 0)
    assert torch.linalg.matrix_rank(J) == 1


def test_tellegen_power_residual_is_zero_for_consistent_potentials_and_flows():
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
