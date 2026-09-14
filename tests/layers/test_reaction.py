"""Tests for local reactions applied after a transport step by operator splitting."""

import math

import torch

from tellegen.layers.reaction import FirstOrderDecay
from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


def sealed_zone() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def test_zero_flow_decay_matches_analytic_exponential_exactly():
    reaction = FirstOrderDecay(0.05)
    x = torch.tensor([200.0], dtype=torch.float64)
    dt = 30.0
    for _ in range(10):
        x = reaction.apply(x, dt)
    expected = 200.0 * math.exp(-0.05 * 10 * dt)
    torch.testing.assert_close(
        x, torch.tensor([expected], dtype=torch.float64), rtol=1e-10, atol=1e-12
    )


def test_operator_splitting_matches_removal_matrix_route_to_first_order():
    net = sealed_zone()
    rate = 0.01
    cap = torch.tensor([1000.0], dtype=torch.float64)
    q = torch.tensor([0.2, 0.2], dtype=torch.float64)
    src = torch.zeros(net.n, dtype=torch.float64)  # FULL node order (spec 4.2); all zero
    x_b = torch.tensor([420.0], dtype=torch.float64)
    dt = 5.0  # small step so the splitting error stays within first-order tolerance

    layer_removal = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([rate], dtype=torch.float64),
    )
    layer_plain = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                  boundary=["ambient"])
    reaction = FirstOrderDecay(rate)

    x_removal = torch.tensor([300.0], dtype=torch.float64)
    x_split = torch.tensor([300.0], dtype=torch.float64)
    for _ in range(20):
        x_removal = layer_removal.step(x_removal, q, src, x_b, dt)
        x_split = layer_plain.step(x_split, q, src, x_b, dt)
        x_split = reaction.apply(x_split, dt)

    torch.testing.assert_close(x_removal, x_split, rtol=2e-3, atol=1e-3)
