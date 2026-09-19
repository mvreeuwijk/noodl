"""Every coefficient family gets a gradient under every scheme (R4), checked against central
differences by torch.autograd.gradcheck in float64."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _t(values, **kwargs):
    return torch.tensor(values, dtype=F64, **kwargs)


def _two_species_net():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    return net


def _conduction_net():
    """ambient <- a (flow); a -- b (wall). Both a and b are interior."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "ambient", kind="flow")
    net.add_edge("a", "b", kind="wall")
    return net


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal", "exact"])
def test_kinetics_gradient(scheme):
    """Species 0 turns into species 1 at rate k inside a zone with unit outflow."""
    net = _two_species_net()

    def f(k):
        kinetics = torch.stack([torch.stack([-k, 0 * k]), torch.stack([k, 0 * k])])  # (2, 2)
        layer = TransportLayer(
            net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
            n_species=2, kinetics=kinetics, scheme=scheme,
        )
        x0 = _t([[1.0, 0.0]])          # (n_i, K)
        sources = torch.zeros(2, 2, dtype=F64)
        return layer.step(x0, _t([1.0]), sources, torch.zeros(1, 2, dtype=F64), 1.0)

    k = torch.tensor(0.5, dtype=F64, requires_grad=True)
    assert gradcheck(f, (k,), eps=1e-6, atol=1e-7, rtol=1e-6)


def test_kinetics_gradient_steady():
    net = _two_species_net()

    def f(k):
        kinetics = torch.stack([torch.stack([-k, 0 * k]), torch.stack([k, 0 * k])])
        layer = TransportLayer(
            net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
            n_species=2, kinetics=kinetics, scheme="implicit",
        )
        sources = _t([[0.0, 0.0], [1.0, 0.0]])   # full node order: species 0 fed at zone
        return layer.steady(_t([1.0]), sources, torch.zeros(1, 2, dtype=F64))

    k = torch.tensor(0.5, dtype=F64, requires_grad=True)
    assert gradcheck(f, (k,), eps=1e-6, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal", "exact"])
def test_conductance_gradient(scheme):
    net = _conduction_net()

    def f(g):
        layer = TransportLayer(
            net, "heat", capacity=_t([1.0, 1.0]), flow_kind="flow", boundary=["ambient"],
            conduction_kind="wall", conductance=g.reshape(1), scheme=scheme,
        )
        return layer.step(_t([1.0, 0.0]), _t([1.0]), torch.zeros(3, dtype=F64), _t([0.0]), 1.0)

    g = torch.tensor(0.7, dtype=F64, requires_grad=True)
    assert gradcheck(f, (g,), eps=1e-6, atol=1e-7, rtol=1e-6)


def test_conductance_gradient_steady():
    net = _conduction_net()

    def f(g):
        layer = TransportLayer(
            net, "heat", capacity=_t([1.0, 1.0]), flow_kind="flow", boundary=["ambient"],
            conduction_kind="wall", conductance=g.reshape(1), scheme="implicit",
        )
        return layer.steady(_t([1.0]), _t([0.0, 0.0, 1.0]), _t([0.0]))

    g = torch.tensor(0.7, dtype=F64, requires_grad=True)
    assert gradcheck(f, (g,), eps=1e-6, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_shared_coefficient_across_two_layers_accumulates_both_gradients(scheme):
    """One learnable removal tensor used by two layers: the gradient is the sum of both
    layers' contributions (the shared-parameter case the review asks for)."""
    r = _t([0.5], requires_grad=True)
    nets = []
    for _ in range(2):
        net = Network(dtype=F64)
        net.add_node("ambient")
        net.add_node("zone")
        net.add_edge("zone", "ambient", kind="flow")
        nets.append(net)
    layers = [
        TransportLayer(n, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
                       removal=r, scheme=scheme)
        for n in nets
    ]
    total = sum(
        layer.step(_t([1.0]), _t([1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0).sum()
        for layer in layers
    )
    (dr,) = torch.autograd.grad(total, (r,))
    single = {"implicit": -1.0 / 2.5**2, "trapezoidal": -1.0 / 1.75**2}[scheme]
    assert dr.item() == pytest.approx(2 * single, rel=1e-8)
