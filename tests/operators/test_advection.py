"""Tests for AdvectionOperator: the nonsymmetric transport spatial operator.

Every test compares the operator's ACTION against TransportLayer's existing dense
assembly (the retained oracle), never against a second copy of the gather/scatter
formulas -- a bug shared between the operator and its own test would otherwise be
invisible.
"""

import torch

from tellegen.operators.advection import AdvectionOperator
from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


def _interior_of_node(net: Network, boundary: list) -> torch.Tensor:
    interior_idx = net.interior_index(boundary)
    out = torch.full((net.n,), -1, dtype=torch.long)
    out[interior_idx] = torch.arange(interior_idx.shape[0], dtype=torch.long)
    return out


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


def test_matvec_matches_dense_transport_operator_forward_flow():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, 0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)

    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_matvec_matches_dense_transport_operator_reversed_and_mixed_sign_flow():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    for q in (
        torch.tensor([-0.3, -0.2, -0.25], dtype=torch.float64),
        torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64),
    ):
        M, _ = layer.operator(q)
        src, tgt = net.endpoints("airpath")
        op = AdvectionOperator(
            src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
            capacity=cap, n_interior=layer.n_i,
            interior_of_node=_interior_of_node(net, ["ambient"]),
        )
        x = torch.tensor([7.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_matvec_batched_matches_looped():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    n = 6
    q = 0.2 + 0.6 * torch.rand(n, 2, dtype=torch.float64) - 0.3
    src, tgt = net.endpoints("airpath")
    interior_of_node = _interior_of_node(net, ["ambient"])
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=interior_of_node,
    )
    x = torch.rand(n, 1, dtype=torch.float64)
    out = op.matvec(x)
    ref = []
    for i in range(n):
        op_i = AdvectionOperator(
            src, tgt, flow=layer.carrier.to(q.dtype) * q[i], transmission=layer.transmission,
            capacity=cap, n_interior=layer.n_i, interior_of_node=interior_of_node,
        )
        ref.append(op_i.matvec(x[i]))
    torch.testing.assert_close(out, torch.stack(ref), rtol=1e-10, atol=1e-12)
