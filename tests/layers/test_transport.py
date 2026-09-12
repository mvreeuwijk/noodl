"""Tests for TransportLayer: exact multi-species advection-diffusion on graphs."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


def flow_through_zone() -> Network:
    """ambient <-> Z, exhaust (tree) then supply (loop), both forward-oriented."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")  # exhaust
    net.add_edge("ambient", "Z", kind="airpath")  # supply
    return net


def two_sealed_zones() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    return net


def test_single_zone_flow_through_matches_analytic_exponential():
    net = flow_through_zone()
    V, Q = 1000.0, 0.5
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([V]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([Q, Q], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    source = torch.tensor([1.0], dtype=torch.float64)
    dt = 300.0
    c = c0.clone()
    for _ in range(20):
        c = layer.step(c, q, source, c_out, dt)
    t = 20 * dt
    tau = V / Q
    c_ss = c_out + source / Q
    expected = c_ss + (c0 - c_ss) * math.exp(-t / tau)
    torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)


def test_two_sealed_zones_conserve_total_amount():
    net = two_sealed_zones()
    cap = torch.tensor([100.0, 300.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.05, 0.05], dtype=torch.float64)
    c = torch.tensor([1000.0, 400.0], dtype=torch.float64)
    total0 = (cap * c).sum()
    for _ in range(10):
        c = layer.step(c, q, torch.zeros(2, dtype=torch.float64), torch.tensor([420.0]), 900.0)
    total = (cap * c).sum()
    torch.testing.assert_close(total, total0, rtol=1e-9, atol=1e-9)


def test_reversed_flow_transports_in_reverse_direction():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q_fwd = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    forward = layer.step(c0, q_fwd, torch.zeros(1, dtype=torch.float64), c_out, 300.0)

    reversed_net = Network(dtype=torch.float64)
    reversed_net.add_node("ambient")
    reversed_net.add_node("Z")
    reversed_net.add_edge("ambient", "Z", kind="airpath")  # was Z->ambient
    reversed_net.add_edge("Z", "ambient", kind="airpath")  # was ambient->Z
    rev_layer = TransportLayer(
        reversed_net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"],
    )
    q_rev = torch.tensor([-0.5, -0.5], dtype=torch.float64)
    backward = rev_layer.step(c0, q_rev, torch.zeros(1, dtype=torch.float64), c_out, 300.0)
    torch.testing.assert_close(forward, backward, rtol=1e-10, atol=1e-10)


def test_transmission_half_halves_steady_state():
    net = flow_through_zone()
    V, Q = 1000.0, 0.5
    full = TransportLayer(net, "co2", capacity=torch.tensor([V]), flow_kind="airpath",
                           boundary=["ambient"])
    filtered = TransportLayer(
        net, "co2", capacity=torch.tensor([V]), flow_kind="airpath", boundary=["ambient"],
        transmission=torch.tensor([1.0, 0.5], dtype=torch.float64),  # halve the supply edge
    )
    q = torch.tensor([Q, Q], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    zero_source = torch.zeros(1, dtype=torch.float64)
    x_full = full.steady(q, zero_source, c_out)
    x_filtered = filtered.steady(q, zero_source, c_out)
    torch.testing.assert_close(x_filtered, 0.5 * x_full, rtol=1e-6, atol=1e-6)


def test_boundary_inflow_enters_interior():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c = torch.tensor([100.0], dtype=torch.float64)
    c_out_high = torch.tensor([800.0], dtype=torch.float64)
    out = layer.step(c, q, torch.zeros(1, dtype=torch.float64), c_out_high, 300.0)
    assert out.item() > c.item()


def test_steady_equals_long_time_step():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    x_steady = layer.steady(q, source, c_out)
    x_long = layer.step(c0, q, source, c_out, 1e6)
    torch.testing.assert_close(x_long, x_steady, rtol=1e-6, atol=1e-6)


def test_batched_step_equals_looped():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    n = 5
    q = 0.3 + 0.4 * torch.rand(n, 2, dtype=torch.float64)
    c = 100 + 300 * torch.rand(n, 1, dtype=torch.float64)
    source = torch.rand(n, 1, dtype=torch.float64)
    c_out = 400 + 40 * torch.rand(n, 1, dtype=torch.float64)
    out = layer.step(c, q, source, c_out, 300.0)
    ref = torch.stack(
        [layer.step(c[i], q[i], source[i], c_out[i], 300.0) for i in range(n)]
    )
    torch.testing.assert_close(out, ref, rtol=1e-8, atol=1e-8)


def test_wrong_shape_raises_value_error_naming_argument():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="sources"):
        layer.step(
            torch.tensor([100.0], dtype=torch.float64), q, torch.zeros(3, dtype=torch.float64),
            torch.tensor([420.0], dtype=torch.float64), 300.0,
        )


def test_gradcheck_step_wrt_x_q_sources_boundary():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        return layer.step(x, q, sources, x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)
