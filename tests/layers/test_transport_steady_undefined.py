"""A steady transport solve is undefined at a stored node that nothing leaves (no outflow, no
conduction, no removal) and at a closed group of stored nodes whose contents never reach a
boundary outflow, a removal or a filter loss: refused by name, never a singular solve."""

from __future__ import annotations

import pytest
import torch

from noodl.layers import UndefinedSteadyState
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
M_ROOM = 120.0                       # kg of air in each room


def _room(removal=None):
    """ambient -> room -> ambient."""
    net = Network(dtype=F64)
    for node in ("ambient", "room"):
        net.add_node(node)
    net.add_edge("ambient", "room", kind="air")
    net.add_edge("room", "ambient", kind="air")
    layer = TransportLayer(net, "c", capacity=torch.tensor([M_ROOM], dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme="implicit",
                           removal=removal)
    return layer


def _steady(layer, q, source):
    sources = torch.tensor([0.0, source], dtype=F64)
    return layer.steady(torch.tensor(q, dtype=F64), sources, torch.tensor([1e-3], dtype=F64))


@pytest.mark.parametrize("source", [0.0, 1e-6])
def test_a_stored_node_nothing_leaves_is_refused_with_or_without_a_source(source):
    with pytest.raises(UndefinedSteadyState, match=r"nothing leaves.*\['room'\]") as err:
        _steady(_room(), [0.0, 0.0], source)
    message = str(err.value)
    assert "undefined without a source" in message and "unbounded with one" in message
    assert "transient" in message
    assert isinstance(err.value, ValueError)


def test_a_removal_alone_defines_the_steady_state():
    k, s = 2e-4, 1e-6
    layer = _room(removal=torch.full((1, 1), k, dtype=F64))
    x = _steady(layer, [0.0, 0.0], s)
    torch.testing.assert_close(x, torch.tensor([s / (k * M_ROOM)], dtype=F64),
                               rtol=1e-12, atol=0)


def test_a_flowing_room_solves():
    q, s = 0.05, 1e-6
    x = _steady(_room(), [q, q], s)
    torch.testing.assert_close(x, torch.tensor([1e-3 + s / q], dtype=F64), rtol=1e-12, atol=0)


def test_a_closed_loop_with_no_exit_is_refused_naming_its_rooms():
    """Two rooms that exchange air with each other only: each has an outflow (a nonzero
    diagonal), but what leaves one enters the other, so nothing ever leaves the pair."""
    net = Network(dtype=F64)
    for node in ("ambient", "a", "b", "c"):
        net.add_node(node)
    net.add_edge("a", "b", kind="air")
    net.add_edge("b", "a", kind="air")
    net.add_edge("ambient", "c", kind="air")
    net.add_edge("c", "ambient", kind="air")
    layer = TransportLayer(net, "species", capacity=torch.full((3,), M_ROOM, dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme="implicit")
    q = torch.tensor([0.1, 0.1, 0.05, 0.05], dtype=F64)
    with pytest.raises(UndefinedSteadyState, match=r"\['a', 'b'\] form a closed group") as err:
        layer.steady(q, torch.zeros(4, dtype=F64), torch.tensor([1e-3], dtype=F64))
    assert "'c'" not in str(err.value)


def test_a_filter_loss_is_an_exit():
    """The same closed loop with a filter on one of its edges: the filter removes what
    passes it, so the loop's steady state is defined (the pass fraction 0.5 sets it)."""
    net = Network(dtype=F64)
    for node in ("ambient", "a", "b"):
        net.add_node(node)
    net.add_edge("a", "b", kind="air")
    net.add_edge("b", "a", kind="air")
    net.add_edge("ambient", "a", kind="air")    # inactive at q=0; keeps 'ambient' a boundary
    layer = TransportLayer(net, "c", capacity=torch.full((2,), M_ROOM, dtype=F64),
                           flow_kind="air", boundary=["ambient"], scheme="implicit",
                           transmission=torch.tensor([0.5, 1.0, 1.0], dtype=F64))
    q = torch.tensor([0.1, 0.1, 0.0], dtype=F64)
    s = 1e-6
    x = layer.steady(q, torch.tensor([0.0, s, 0.0], dtype=F64),
                     torch.tensor([1e-3], dtype=F64))
    # a: q x_a = q x_b + s ; b: q x_b = 0.5 q x_a  ->  x_a = 2 s / q, x_b = s / q
    torch.testing.assert_close(x, torch.tensor([2 * s / 0.1, s / 0.1], dtype=F64),
                               rtol=1e-12, atol=0)


def test_one_idle_batch_member_is_refused():
    q = torch.tensor([[0.05, 0.05], [0.0, 0.0]], dtype=F64)
    with pytest.raises(UndefinedSteadyState, match=r"\['room'\]"):
        _room().steady(q, torch.zeros(2, 2, dtype=F64), torch.full((2, 1), 1e-3, dtype=F64))


def test_the_refusal_also_guards_the_returning_path():
    with pytest.raises(UndefinedSteadyState):
        _room().steady(torch.zeros(2, dtype=F64), torch.zeros(2, dtype=F64),
                       torch.tensor([1e-3], dtype=F64), on_failure="return")


def _chain(n):
    """ambient -> 0 -> 1 -> ... -> n-1 -> ambient, plus a return edge n-1 -> 0."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    for i in range(n):
        net.add_node(i)
    net.add_edge("ambient", 0, kind="air")
    for i in range(n - 1):
        net.add_edge(i, i + 1, kind="air")
    net.add_edge(n - 1, "ambient", kind="air")
    net.add_edge(n - 1, 0, kind="air")
    return TransportLayer(net, "c", capacity=torch.full((n,), M_ROOM, dtype=F64),
                          flow_kind="air", boundary=["ambient"], scheme="implicit")


def test_a_deep_chain_is_searched_per_batch_member():
    """Member 0 flows through a 400-room chain to the ambient; member 1 recirculates the
    same chain through its return edge with no way out, a closed group as deep as the
    chain. Only member 1 is refused, naming every room."""
    n, f = 400, 0.1
    layer = _chain(n)
    through = [f] * (n + 1) + [0.0]
    loop = [0.0] + [f] * (n - 1) + [0.0, f]
    sources = torch.zeros(n + 1, dtype=F64)
    xb = torch.tensor([1e-3], dtype=F64)
    x = layer.steady(torch.tensor(through, dtype=F64), sources, xb)
    torch.testing.assert_close(x, torch.full((n,), 1e-3, dtype=F64), rtol=1e-12, atol=0)
    q = torch.tensor([through, loop], dtype=F64)
    with pytest.raises(UndefinedSteadyState, match=r"form a closed group") as err:
        layer.steady(q, sources.expand(2, -1), xb.expand(2, -1))
    assert f"{n - 1}]" in str(err.value) and "[0, 1, 2," in str(err.value)
