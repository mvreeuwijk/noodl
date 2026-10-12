"""Storage-free interior nodes: a node that holds no amount (a zero-volume air-handler
plenum). Its balance is algebraic in every step, and with no flow through it, it keeps its
value, as ContamX 3.4.1.7 holds a zero-volume plenum of a switched-off air handler."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.building_physics.sources import (
    ConstantSource,
    DepositionRateSink,
    SpeciesSources,
)
from noodl.apps.building_physics.thermal import CP_AIR, RHO_0, species_layer, thermal_layer
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
DT = 300.0
M_ROOM = 120.0                       # kg of air in the room


def _plenum(scheme="implicit", storage_free=("plenum",)):
    """ambient -> plenum (outdoor air), room -> plenum (recirculation), plenum -> room
    (supply), room -> ambient (exhaust)."""
    net = Network(dtype=F64)
    for node in ("ambient", "plenum", "room"):
        net.add_node(node)
    net.add_edge("ambient", "plenum", kind="air")
    net.add_edge("room", "plenum", kind="air")
    net.add_edge("plenum", "room", kind="air")
    net.add_edge("room", "ambient", kind="air")
    layer = TransportLayer(net, "c", capacity=torch.tensor([0.0, M_ROOM], dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme=scheme,
                           storage_free=storage_free)
    return net, layer


def _q(outdoor, recirculated):
    return torch.tensor([outdoor, recirculated, outdoor + recirculated, outdoor], dtype=F64)


def _step(layer, x0, q, **kw):
    sources = torch.zeros(q.shape[:-1] + (3,), dtype=F64)
    xb = torch.full(q.shape[:-1] + (1,), 1e-3, dtype=F64)
    return layer.step(x0, q, sources, xb, DT, **kw)


def _room_and_plenum(x0_room=2e-4, outdoor=0.03, recirculated=0.07):
    # room: M/dt (x - x0) = q_o (x_amb - x)   (the plenum passes q_o x_amb + q_r x on)
    room = (M_ROOM / DT * x0_room + outdoor * 1e-3) / (M_ROOM / DT + outdoor)
    plenum = (outdoor * 1e-3 + recirculated * room) / (outdoor + recirculated)
    return plenum, room


def test_a_storage_free_node_mixes_its_inflows_within_the_step():
    _, layer = _plenum()
    assert layer.storage_free_mask.tolist() == [True, False]
    x = _step(layer, torch.tensor([0.0, 2e-4], dtype=F64), _q(0.03, 0.07))
    plenum, room = _room_and_plenum()
    torch.testing.assert_close(x, torch.tensor([plenum, room], dtype=F64), rtol=1e-13, atol=0)


def test_a_storage_free_node_without_through_flow_keeps_its_value():
    _, layer = _plenum()
    x0 = torch.tensor([3e-4, 2e-4], dtype=F64)
    x = _step(layer, x0, _q(0.0, 0.0))
    torch.testing.assert_close(x, x0, rtol=1e-15, atol=0)


def test_a_source_on_a_storage_free_node_enters_the_mix_or_is_held_without_flow():
    """A heat input at a plenum (a heating coil) is carried on by the flow through it; with
    no flow it has nowhere to go and the node keeps its value."""
    _, layer = _plenum()
    x0 = torch.tensor([3e-4, 2e-4], dtype=F64)
    src = torch.tensor([0.0, 1e-5, 0.0], dtype=F64)
    xb = torch.tensor([1e-3], dtype=F64)
    held = layer.step(x0, _q(0.0, 0.0), src, xb, DT)
    torch.testing.assert_close(held, x0, rtol=1e-15, atol=0)
    x = layer.step(x0, _q(0.03, 0.07), src, xb, DT)
    # room: M/dt (x - x0) = q_o x_amb + S - q_o x   (everything the plenum receives passes on)
    room = (M_ROOM / DT * 2e-4 + 0.03 * 1e-3 + 1e-5) / (M_ROOM / DT + 0.03)
    plenum = (0.03 * 1e-3 + 0.07 * room + 1e-5) / 0.1
    torch.testing.assert_close(x, torch.tensor([plenum, room], dtype=F64), rtol=1e-13, atol=0)


def test_a_batch_with_the_handler_on_in_one_member_and_off_in_the_other():
    _, layer = _plenum()
    x0 = torch.tensor([3e-4, 2e-4], dtype=F64).expand(2, 2)
    x = _step(layer, x0, torch.stack([_q(0.03, 0.07), _q(0.0, 0.0)]))
    assert x.shape == (2, 2)
    torch.testing.assert_close(x[1], x0[1], rtol=1e-15, atol=0)
    _, room = _room_and_plenum()
    assert x[0, 1].item() == pytest.approx(room, rel=1e-13)


def test_steady_keeps_the_algebraic_balance():
    _, layer = _plenum()
    x = layer.steady(_q(0.03, 0.07), torch.zeros(3, dtype=F64),
                     torch.tensor([1e-3], dtype=F64))
    torch.testing.assert_close(x, torch.tensor([1e-3, 1e-3], dtype=F64), rtol=1e-12, atol=0)


def test_removal_does_not_act_on_a_storage_free_node():
    _, layer = _plenum()
    x0 = torch.tensor([0.0, 2e-4], dtype=F64)
    plain = _step(layer, x0, _q(0.03, 0.07))
    removed = _step(layer, x0, _q(0.03, 0.07), removal=torch.tensor([5.0, 0.0], dtype=F64))
    torch.testing.assert_close(removed, plain, rtol=1e-14, atol=0)


def test_a_non_finite_rate_on_a_storage_free_row_does_not_reach_the_solve():
    """A rate of 0 / 0 at a zero-capacity node is replaced, not multiplied by zero."""
    _, layer = _plenum()
    x0 = torch.tensor([0.0, 2e-4], dtype=F64)
    plain = _step(layer, x0, _q(0.03, 0.07))
    nan = _step(layer, x0, _q(0.03, 0.07), removal=torch.tensor([float("nan"), 0.0], dtype=F64))
    torch.testing.assert_close(nan, plain, rtol=1e-14, atol=0)


def test_a_per_step_capacity_may_be_zero_at_the_storage_free_node():
    _, layer = _plenum()
    x0 = torch.tensor([0.0, 2e-4], dtype=F64)
    plain = _step(layer, x0, _q(0.03, 0.07))
    zero = _step(layer, x0, _q(0.03, 0.07), capacity=torch.tensor([0.0, M_ROOM], dtype=F64),
                 capacity_prev=torch.tensor([0.0, M_ROOM], dtype=F64))
    torch.testing.assert_close(zero, plain, rtol=1e-15, atol=0)


def test_the_step_is_differentiable_through_the_flows():
    _, layer = _plenum()
    x0 = torch.tensor([0.0, 2e-4], dtype=F64)

    def room_after(q):
        return _step(layer, x0, q)

    q = _q(0.03, 0.07).requires_grad_(True)
    assert torch.autograd.gradcheck(room_after, (q,), eps=1e-7, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("scheme", ["exact", "trapezoidal"])
def test_only_the_implicit_scheme_takes_storage_free_nodes(scheme):
    with pytest.raises(ValueError, match=r"storage-free nodes \['plenum'\].*implicit"):
        _plenum(scheme=scheme)


def test_zero_capacity_is_still_refused_where_a_node_stores():
    _, layer = _plenum()
    with pytest.raises(ValueError, match=r"strictly positive.*\['room'\]"):
        _step(layer, torch.zeros(2, dtype=F64), _q(0.03, 0.07),
              capacity=torch.zeros(2, dtype=F64))
    with pytest.raises(ValueError, match=r"storage_free nodes \['ambient'\].*not active interior"):
        _plenum(storage_free=("ambient",))


def test_a_layer_without_storage_free_nodes_has_an_all_false_mask():
    net = Network(dtype=F64)
    for node in ("ambient", "room"):
        net.add_node(node)
    net.add_edge("ambient", "room", kind="air")
    net.add_edge("room", "ambient", kind="air")
    layer = TransportLayer(net, "c", capacity=torch.tensor([M_ROOM], dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme="exact")
    assert layer.storage_free_mask.tolist() == [False]


def _building_net():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("plenum", volume=0.0)
    net.add_node("room", volume=100.0)
    net.add_edge("ambient", "plenum", kind="airpath")
    net.add_edge("plenum", "room", kind="airpath")
    net.add_edge("room", "ambient", kind="airpath")
    return net


def test_species_layer_takes_zero_volume_only_where_declared_storage_free():
    net = _building_net()
    with pytest.raises(ValueError, match=r"\['plenum'\] have zero volume"):
        species_layer(net)
    layer = species_layer(net, storage_free=["plenum"])
    assert layer.storage_free_mask.tolist() == [True, False]


def test_thermal_layer_with_storage_free_nodes_uses_the_implicit_scheme():
    net = _building_net()
    with pytest.raises(ValueError, match=r"\['plenum'\] have no heat capacity"):
        thermal_layer(net)
    layer = thermal_layer(net, storage_free=["plenum"])
    assert layer.scheme == "implicit"
    assert layer.storage_free_mask.tolist() == [True, False]
    with pytest.raises(ValueError, match=r"storage-free nodes \['plenum'\].*implicit"):
        thermal_layer(net, storage_free=["plenum"], scheme="exact")
    # supply air at 280 K mixes through the plenum into the room within the step
    q = torch.tensor([0.1, 0.1, 0.1], dtype=F64)
    T0 = torch.tensor([300.0, 300.0], dtype=F64)
    T = layer.step(T0, q, torch.zeros(3, dtype=F64), torch.tensor([280.0], dtype=F64), DT)
    C = RHO_0 * CP_AIR * 100.0
    room = (C / DT * 300.0 + CP_AIR * 0.1 * 280.0) / (C / DT + CP_AIR * 0.1)
    torch.testing.assert_close(T, torch.tensor([280.0, room], dtype=F64), rtol=1e-13, atol=0)


def test_a_storage_free_node_joined_only_by_conduction_takes_its_conduction_balance():
    """A storage-free node with conduction edges and no flow through it is algebraic in
    its conduction: its temperature is the conductance-weighted mean of its neighbours'."""
    net = Network(dtype=F64)
    for node in ("ambient", "core", "room"):
        net.add_node(node)
    net.add_edge("ambient", "room", kind="air")
    net.add_edge("room", "ambient", kind="air")
    net.add_edge("room", "core", kind="wall")
    net.add_edge("core", "ambient", kind="wall")
    layer = TransportLayer(net, "T", capacity=torch.tensor([0.0, 1e5], dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme="implicit",
                           conduction_kind="wall",
                           conductance=torch.tensor([3.0, 1.0], dtype=F64),
                           storage_free=["core"])
    T = layer.step(torch.tensor([0.0, 300.0], dtype=F64), torch.zeros(2, dtype=F64),
                   torch.zeros(3, dtype=F64), torch.tensor([280.0], dtype=F64), DT)
    torch.testing.assert_close(T[0], (3.0 * T[1] + 1.0 * 280.0) / 4.0, rtol=1e-13, atol=0)


# ---------------------------------------------------------------- sources on the layer


def _sources_drivers(layer, n_sources):
    return {
        "time": 60.0,
        "species.x_boundary": torch.zeros(1, dtype=F64),
        "species.source_multiplier": torch.ones(n_sources, dtype=F64),
        "species.capacity": torch.tensor([0.0, RHO_0 * 100.0], dtype=F64),
    }


def test_a_sink_in_the_room_stays_finite_beside_a_zero_volume_plenum():
    net = _building_net()
    layer = species_layer(net, storage_free=["plenum"])
    room = net.node_index("room")
    closure = SpeciesSources([DepositionRateSink(room, 1e-3)], species=[0], layer=layer)
    state = {"species.x": torch.tensor([1e-4, 2e-4], dtype=F64)}
    out = closure(state, _sources_drivers(layer, 1))
    removal = out["species.removal"]
    assert bool(torch.isfinite(removal).all())
    assert removal[0].item() == 0.0
    x = layer.step(state["species.x"], torch.tensor([0.1, 0.1, 0.1], dtype=F64),
                   out["species.sources"], torch.zeros(1, dtype=F64), DT,
                   capacity=_sources_drivers(layer, 1)["species.capacity"], removal=removal)
    assert bool(torch.isfinite(x).all())


@pytest.mark.parametrize("source", [
    lambda node: DepositionRateSink(node, 1e-3),
    lambda node: ConstantSource(node, 1e-6),
])
def test_a_source_or_sink_on_a_storage_free_node_is_refused_by_name(source):
    net = _building_net()
    layer = species_layer(net, storage_free=["plenum"])
    with pytest.raises(ValueError, match=r"source 1 .*'plenum'.*holds no air"):
        SpeciesSources([source(net.node_index("plenum"))], species=[0], layer=layer)
