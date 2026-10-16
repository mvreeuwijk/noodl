"""Refusals a KirchhoffLayer owes its caller: a stale topology and unreachable gradients."""

import pytest
import torch

from noodl.elements import Conductance
from noodl.elements.base import ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64


def _net() -> Network:
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="conduction")
    return net


def _layer(net: Network) -> KirchhoffLayer:
    return KirchhoffLayer(
        net, "heat", [Conductance(torch.tensor(1.0, dtype=F64))], boundary=["a"]
    )


# ------------------------------------------------------------------ R08
def test_a_layer_refuses_to_solve_after_an_edge_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.solve(torch.ones(1, dtype=F64), differentiable=False)


def test_a_layer_refuses_component_calls_after_a_node_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.flows(torch.zeros(2, dtype=F64), {})


def test_a_new_layer_over_the_changed_network_works():
    net = _net()
    _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    fresh = KirchhoffLayer(
        net, "heat", [Conductance(torch.ones(2, dtype=F64))], boundary=["a"]
    )
    phi, q = fresh.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(phi, torch.ones(3, dtype=F64))
    assert q.shape == (2,)


def test_re_adding_an_existing_node_with_attributes_is_not_a_topology_change():
    net = _net()
    layer = _layer(net)
    net.add_node("b", volume=3.0)
    phi, q = layer.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(q, torch.zeros(1, dtype=F64))


# ------------------------------------------------------------------ H01
class _BufferLaw(ConstitutiveLaw):
    def __init__(self, g):
        super().__init__("conduction")
        self.register_buffer("g", g)

    def flow(self, dp, drivers=None):
        return self.g * dp

    def dflow(self, dp, drivers=None):
        return self.g * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        return torch.zeros_like(self.g), self.g


class _ContainerLaw(_BufferLaw):
    def __init__(self, g):
        ConstitutiveLaw.__init__(self, "conduction")
        self.coeffs = {"g": [g]}
        self.g = g.detach()

    def flow(self, dp, drivers=None):
        return self.coeffs["g"][0] * dp

    def dflow(self, dp, drivers=None):
        return self.coeffs["g"][0] * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        g = self.coeffs["g"][0]
        return torch.zeros_like(g), g


@pytest.mark.parametrize("law_class", [_BufferLaw, _ContainerLaw])
def test_a_gradient_tensor_in_a_buffer_or_container_is_refused(law_class):
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [law_class(g)], boundary=["a"])
    with pytest.raises(ValueError, match="not a registered nn.Parameter"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64))
