"""Parity tests between TransportLayer's new operator-based paths and its retained
dense oracle (`operator()`, unchanged since milestone 1). Every test here compares
the SPARSE result against the DENSE one on the same problem; `tests/layers/
test_transport.py` is untouched and re-verifies the dense/analytic behaviour on
its own.
"""

import torch
from torch.autograd import gradcheck

from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


def flow_through_zone() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def three_node_chain() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("ambient", "A", kind="airpath")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "ambient", kind="airpath")
    return net


def test_advection_operator_matvec_matches_dense_operator():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)
    op = layer._advection_operator(q)
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_advection_operator_boundary_forcing_matches_dense_N():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    _, N = layer.operator(q)
    op = layer._advection_operator(q)
    x_b = torch.tensor([420.0], dtype=torch.float64)
    torch.testing.assert_close(op.boundary_forcing(x_b), N @ x_b, rtol=1e-9, atol=1e-12)
