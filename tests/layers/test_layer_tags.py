"""Layer surface added for Model (spec 4.3, 4.4, 14): kind slices, tags, multi-kind
transport, inactive nodes."""

from __future__ import annotations

import pytest
import torch

from noodl.elements import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import TransportLayer, active_interior
from noodl.topology import Network

F64 = torch.float64


def _two_kind_net():
    """ambient -> z1 (airpath), z1 -> z2 (door), z2 -> ambient (airpath)."""
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="door")
    net.add_edge("z2", "ambient", kind="airpath")
    return net


def _one_kind_net():
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    return net


def test_kind_slice_and_flows_of_kind_follow_element_order():
    net = _two_kind_net()
    layer = PotentialFlowLayer(
        net, "air",
        [PowerLaw(torch.tensor([0.01, 0.01], dtype=F64), 0.65, kind="airpath"),
         PowerLaw(torch.tensor([0.5], dtype=F64), 0.5, kind="door")],
        boundary=["ambient"],
    )
    assert layer.kind_slice("airpath") == slice(0, 2)
    assert layer.kind_slice("door") == slice(2, 3)
    q = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    torch.testing.assert_close(layer.flows_of_kind(q, "door"), torch.tensor([3.0], dtype=F64))
    torch.testing.assert_close(
        layer.flows_of_kind(q, ("door", "airpath")), torch.tensor([3.0, 1.0, 2.0], dtype=F64)
    )
    with pytest.raises(KeyError, match=r"'air'.*'hydronic'.*airpath"):
        layer.kind_slice("hydronic")


def test_layers_carry_quantity_and_unit_tags():
    net = _one_kind_net()
    el = PowerLaw(torch.tensor([0.01, 0.01, 0.01], dtype=F64), 0.65)
    default = PotentialFlowLayer(net, "air", [el], boundary=["ambient"])
    assert (default.quantity, default.unit) == ("potential", "")
    tagged = PotentialFlowLayer(
        net, "air", [el], boundary=["ambient"], quantity="pressure", unit="Pa"
    )
    assert (tagged.quantity, tagged.unit) == ("pressure", "Pa")
    t = TransportLayer(
        net, "thermal", capacity=torch.ones(2, dtype=F64), flow_kind="airpath",
        boundary=["ambient"], quantity="temperature", unit="K",
    )
    assert (t.quantity, t.unit) == ("temperature", "K")
    t0 = TransportLayer(
        net, "x", capacity=torch.ones(2, dtype=F64), flow_kind="airpath", boundary=["ambient"]
    )
    assert (t0.quantity, t0.unit) == ("scalar", "")


def test_transport_over_two_flow_kinds_matches_the_single_kind_layer():
    one = TransportLayer(
        _one_kind_net(), "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
        flow_kind="airpath", boundary=["ambient"], scheme="implicit",
    )
    two = TransportLayer(
        _two_kind_net(), "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
        flow_kind=("airpath", "door"), boundary=["ambient"], scheme="implicit",
    )
    assert two.flow_kinds == ("airpath", "door")
    q_one = torch.tensor([1.0, 0.7, 1.0], dtype=F64)          # ambient->z1, z1->z2, z2->ambient
    q_two = torch.tensor([1.0, 1.0, 0.7], dtype=F64)          # airpath edges first, then door
    s = torch.tensor([0.0, 3.0, 0.0], dtype=F64)
    xb = torch.zeros(1, dtype=F64)
    torch.testing.assert_close(two.steady(q_two, s, xb), one.steady(q_one, s, xb),
                               rtol=1e-12, atol=1e-12)
    x0 = torch.tensor([1.0, 2.0], dtype=F64)
    torch.testing.assert_close(two.step(x0, q_two, s, xb, dt=30.0),
                               one.step(x0, q_one, s, xb, dt=30.0), rtol=1e-12, atol=1e-12)
    # The dense oracle agrees too.
    M2, N2 = two.operator(q_two)
    M1, N1 = one.operator(q_one)
    torch.testing.assert_close(M2, M1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(N2, N1, rtol=1e-12, atol=1e-12)


def _wall_net():
    """z1 -> ambient (airpath); z1 -> wall (wall); wall -> ambient (wall).

    The single airpath edge is the zone's EXHAUST (z1 -> ambient), not its supply. With a
    supply-only edge (ambient -> z1) the zone is a dead end: z1 is the upwind node of no
    edge, so its own advective row is identically zero and `species.steady` below would be
    singular at z1 for a reason that has nothing to do with the inactive `wall` node this
    test is about. Oriented outwards, z1's row carries the outflow, and the only zero row
    left in the full-node generator is the `wall` one the inactive rule removes.
    """
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "wall"):
        net.add_node(name)
    net.add_edge("z1", "ambient", kind="airpath")
    net.add_edge("z1", "wall", kind="wall", ua=5.0)
    net.add_edge("wall", "ambient", kind="wall", ua=5.0)
    return net


def test_potential_layer_treats_nodes_untouched_by_its_kinds_as_inactive():
    net = _wall_net()
    layer = PotentialFlowLayer(
        net, "air", [PowerLaw(torch.tensor([0.01], dtype=F64), 0.65)], boundary=["ambient"]
    )
    assert layer.interior.tolist() == [net.node_index("z1")]
    assert layer.inactive.tolist() == [net.node_index("wall")]
    s = torch.tensor([0.0, 2e-3, 0.0], dtype=F64)
    phi, q = layer.solve(torch.zeros(1, dtype=F64), {}, s, differentiable=False)
    assert phi.shape == (3,)
    assert phi[net.node_index("wall")].item() == 0.0
    assert q.shape == (1,)
    bad = torch.tensor([0.0, 0.0, 1e-3], dtype=F64)
    with pytest.raises(ValueError, match=r"'air'.*inactive.*wall"):
        layer.solve(torch.zeros(1, dtype=F64), {}, bad, differentiable=False)


def test_transport_layer_treats_nodes_untouched_by_its_kinds_as_inactive():
    net = _wall_net()
    species = TransportLayer(
        net, "species", capacity=torch.tensor([10.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit",
    )
    assert species.n_i == 1
    assert species.inactive_idx.tolist() == [net.node_index("wall")]
    thermal = TransportLayer(
        net, "thermal", capacity=torch.tensor([10.0, 100.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], conduction_kind="wall",
        conductance=torch.tensor([5.0, 5.0], dtype=F64), scheme="implicit",
    )
    assert thermal.n_i == 2
    assert thermal.inactive_idx.numel() == 0
    # steady of the species layer is not singular: the wall row is simply absent.
    x = species.steady(torch.tensor([0.5], dtype=F64), torch.tensor([0.0, 1.0, 0.0], dtype=F64),
                       torch.zeros(1, dtype=F64))
    assert x.shape == (1,)
    with pytest.raises(ValueError, match=r"'species'.*inactive.*wall"):
        species.steady(torch.tensor([0.5], dtype=F64), torch.tensor([0.0, 0.0, 1.0], dtype=F64),
                       torch.zeros(1, dtype=F64))


def test_active_interior_splits_the_non_boundary_nodes_by_the_kinds_that_touch_them():
    """`active_interior` is the one place the inactive rule is decided: `TransportLayer`,
    `SpeciesTransport` and the composed benchmark all size their per-node vectors with it.
    """
    net = _wall_net()
    i_air, inactive_air = active_interior(net, ("airpath",), ["ambient"])
    assert i_air.tolist() == [net.node_index("z1")]
    assert inactive_air.tolist() == [net.node_index("wall")]
    # Two kinds together touch every node; the boundary node is in neither half.
    i_both, inactive_both = active_interior(net, ("airpath", "wall"), ["ambient"])
    assert i_both.tolist() == [net.node_index("z1"), net.node_index("wall")]
    assert inactive_both.numel() == 0
    # With no boundary at all, the would-be boundary node is classified like any other.
    i_none, inactive_none = active_interior(net, ("airpath",), [])
    assert i_none.tolist() == [net.node_index("ambient"), net.node_index("z1")]
    assert inactive_none.tolist() == [net.node_index("wall")]
