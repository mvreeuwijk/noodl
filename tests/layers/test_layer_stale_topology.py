"""Transport and allocation layers refuse a network that changed after construction."""

import pytest
import torch

from noodl.layers.allocation import AllocatedFlowLayer
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
MATCH = r"topology.*changed"


def _net() -> Network:
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def _grow(net: Network) -> None:
    net.add_node("W")
    net.add_edge("W", "Z", kind="airpath")


def _transport(net):
    return TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )


_CALLS = {
    "step": lambda L, q, s, xb: L.step(torch.zeros(1, dtype=F64), q, s, xb, 1.0),
    "step_with_transfer": lambda L, q, s, xb: L.step_with_transfer(
        torch.zeros(1, dtype=F64), q, s, xb, 1.0),
    "steady": lambda L, q, s, xb: L.steady(q, s, xb),
    "rate": lambda L, q, s, xb: L.rate(torch.zeros(1, dtype=F64), q, s, xb),
    "operator": lambda L, q, s, xb: L.operator(q),
    "net_inflow": lambda L, q, s, xb: L.net_inflow(q),
}


@pytest.mark.parametrize("entry", sorted(_CALLS))
def test_a_transport_layer_refuses_a_changed_network(entry):
    net = _net()
    layer = _transport(net)
    _grow(net)
    q = torch.full((2,), 0.5, dtype=F64)
    s = torch.zeros(2, dtype=F64)
    xb = torch.zeros(1, dtype=F64)
    with pytest.raises(RuntimeError, match=MATCH):
        _CALLS[entry](layer, q, s, xb)


def test_a_transport_layer_still_runs_on_an_unchanged_network():
    net = _net()
    layer = _transport(net)
    x = layer.step(
        torch.zeros(1, dtype=F64), torch.full((2,), 0.5, dtype=F64),
        torch.zeros(2, dtype=F64), torch.ones(1, dtype=F64), 10.0,
    )
    assert x.shape == (1,)


def test_an_allocation_layer_refuses_a_changed_network():
    net = Network(dtype=F64)
    for n in "AB":
        net.add_node(n)
    net.add_edge("A", "B", kind="link")
    layer = AllocatedFlowLayer(
        net, "cap", "link", s_max=torch.full((2,), 10.0, dtype=F64),
        c_arc=torch.ones(1, dtype=F64),
    )
    net.add_node("C")
    with pytest.raises(RuntimeError, match=MATCH):
        layer.step(
            torch.zeros(2, dtype=F64), {"cap.requests": torch.ones(1, dtype=F64)}, 1.0
        )
