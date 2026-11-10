"""Implicit laws that read the heads at their end nodes (`ConstitutiveLaw.reads_heads`).

A pressure-reducing valve holds the head downstream of it, not the drop across it: its
residual reads `ctx.phi_to`, the absolute head of its edge's end node. These tests use an
ideal downstream-head source, `phi_to - setting = 0`, on the line

    R (head 50) --pipe--> J1 --law--> J2 --pipe--> J3 (demand 0.01 m3/s).
"""

import pytest
import torch

from noodl.elements import Conductance, ConstitutiveLaw, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
DEMAND = 0.01


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


class HeldDownstreamHead(ImplicitLaw):
    """An ideal downstream-head source: the head at the edge's end node is `setting`."""

    reads_heads = True

    def __init__(self, setting, kind="held", learnable=False):
        super().__init__(kind)
        self.setting = self._param(setting, learnable)

    def residual(self, dp, q, drivers=None, ctx=None):
        return ctx.phi_to - self.setting


class Mixed(ImplicitLaw):
    """A relation nonlinear in its drop, its flow and both end heads, so that each of the
    four partials (and the layer's combination of them) is non-zero."""

    reads_heads = True

    def __init__(self, kind="held"):
        super().__init__(kind)

    def residual(self, dp, q, drivers=None, ctx=None):
        return (q * q.abs() + 0.3 * dp**3 + 0.02 * ctx.phi_from * ctx.phi_to
                - 0.01 * ctx.phi_to**2)


def _layer(law):
    net = Network(dtype=F64)
    for n in ("R", "J1", "J2", "J3"):
        net.add_node(n)
    net.add_edge("R", "J1", kind="pipe")
    net.add_edge("J1", "J2", kind=law.kind)
    net.add_edge("J2", "J3", kind="pipe")
    layer = KirchhoffLayer(net, "water", [Conductance(torch.tensor([0.002, 0.004]), kind="pipe"),
                                          law], boundary=["R"])
    sources = torch.zeros(net.n)
    sources[net.nodes.index("J3")] = -DEMAND
    return net, layer, torch.tensor([50.0]), sources


def test_the_end_head_is_held():
    net, layer, phi_b, sources = _layer(HeldDownstreamHead(40.0))
    phi, q = layer.solve(phi_b, {}, sources, differentiable=False)
    assert abs(float(phi[net.nodes.index("J2")]) - 40.0) < 1e-10
    torch.testing.assert_close(q, torch.full((3,), DEMAND), rtol=0, atol=1e-12)


@pytest.mark.parametrize("make", [lambda: HeldDownstreamHead(40.0), Mixed], ids=["held", "mixed"])
def test_jacobian_matches_autograd_of_the_residual(make):
    _, layer, phi_b, sources = _layer(make())
    # a point that is not the solution: interior heads (J1, J2, J3), then the law's flow
    x = torch.tensor([47.0, 41.5, 38.0, 0.013])

    def residual(x):
        return layer._residual(x, phi_b, {}, sources)[0]

    assert residual(x).abs().max() > 1e-3
    expected = torch.autograd.functional.jacobian(residual, x)
    got = layer._operator(x, phi_b, {}).assemble()
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)


def test_gradient_of_the_held_head():
    setting = torch.tensor(40.0)
    net, layer, phi_b, sources = _layer(HeldDownstreamHead(setting, learnable=True))
    law = layer._laws[1]
    j1, j3 = net.nodes.index("J1"), net.nodes.index("J3")
    phi, _ = layer.solve(phi_b, {}, sources)
    (d_j3,) = torch.autograd.grad(phi[j3], law.setting)
    assert abs(float(d_j3) - 1.0) < 1e-8

    phi, _ = layer.solve(phi_b, {}, sources)
    (d_j1,) = torch.autograd.grad(phi[j1], law.setting)
    h = 1e-4

    def j1_at(value):
        _, other, pb, src = _layer(HeldDownstreamHead(value))
        return float(other.solve(pb, {}, src, differentiable=False)[0][j1])

    central = (j1_at(40.0 + h) - j1_at(40.0 - h)) / (2 * h)
    # The flow through the law is the demand whatever the setting, so J1's head does not
    # move with it: both are zero, compared with an absolute floor.
    torch.testing.assert_close(float(d_j1), central, rtol=1e-6, atol=1e-12)


def test_reads_heads_on_a_flow_law_is_refused():
    class HeadFlow(ConstitutiveLaw):
        reads_heads = True

        def flow(self, dp, drivers=None):
            return dp

    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="f")
    with pytest.raises(TypeError, match="reads_heads"):
        KirchhoffLayer(net, "water", [HeadFlow("f")], boundary=["a"])


def test_reads_heads_on_a_multi_port_law_is_refused():
    class TwoPort(HeldDownstreamHead):
        ports = 2

    net = Network(dtype=F64)
    for n in ("a", "b", "c"):
        net.add_node(n)
    net.add_edge("a", "b", kind="held")
    net.add_edge("b", "c", kind="held")
    with pytest.raises(TypeError, match="TwoPort.*reads_heads"):
        KirchhoffLayer(net, "water", [TwoPort(1.0)], boundary=["a"])
