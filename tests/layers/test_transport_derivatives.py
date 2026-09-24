"""Independent derivative references for the transport findings R3 and R4.

One compartment: node `zone` with unit capacity, a single edge zone->ambient carrying q=1
(pure outflow) unless `circulation=True` adds ambient->zone with q=1 as well. Every expected
value below is derived in its docstring from the scalar ODE dx/dt = -(q + r) x + s.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _t(values, **kwargs):
    return torch.tensor(values, dtype=F64, **kwargs)


def _layer(scheme="exact", *, circulation=False, **kwargs):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    if circulation:
        net.add_edge("ambient", "zone", kind="flow")
    return TransportLayer(
        net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
        scheme=scheme, **kwargs,
    )


@pytest.mark.parametrize(
    "x0",
    [
        pytest.param(0.0, id="exact-zero"),
        pytest.param(1e-12, id="near-zero"),
        pytest.param(1.0, id="control"),
    ],
)
def test_exact_step_tangent_matches_the_matrix_exponential_at_equilibrium(x0):
    """dx/dt = -x + s with s = 0: x(dt) = e^{-dt} x0 + (1 - e^{-dt}) s.

    d x(1)/d x0 = e^{-1}, d x(1)/d s = 1 - e^{-1}, whatever x0 is -- including 0, where the
    forward value is exactly 0 and the first Taylor increment vanishes.
    """
    layer = _layer("exact")
    x = _t([x0], requires_grad=True)
    sources = _t([0.0, 0.0], requires_grad=True)
    y = layer.step(x, _t([1.0]), sources, _t([0.0]), 1.0)
    dx, ds = torch.autograd.grad(y.sum(), (x, sources))
    assert dx.item() == pytest.approx(math.exp(-1.0), rel=1e-8, abs=1e-10)
    assert ds[1].item() == pytest.approx(1.0 - math.exp(-1.0), rel=1e-8, abs=1e-10)


@pytest.mark.parametrize(
    "scheme, expected",
    [
        pytest.param("exact", -math.exp(-1.5), id="exact-control"),
        pytest.param("implicit", -1.0 / 2.5**2, id="implicit"),
        pytest.param("trapezoidal", -1.0 / 1.75**2, id="trapezoidal"),
    ],
)
def test_step_gradient_with_respect_to_removal(scheme, expected):
    """x0 = 1, q = 1, V = 1, r = 0.5, dt = 1, no source: dx/dt = -(1 + r) x.

    exact:       x1 = e^{-(1+r)}            -> dx1/dr = -e^{-1.5}
    implicit:    x1 = 1/(1 + (1+r))         -> dx1/dr = -1/2.5^2
    trapezoidal: x1 = (1 - a)/(1 + a), a = (1+r)/2 -> dx1/dr = -1/(1+a)^2 = -1/1.75^2
    """
    removal = _t([0.5], requires_grad=True)
    layer = _layer(scheme, removal=removal)
    y = layer.step(_t([1.0]), _t([1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0)
    (dr,) = torch.autograd.grad(y.sum(), (removal,), allow_unused=True)
    assert dr is not None, "removal received no gradient"
    assert dr.item() == pytest.approx(expected, rel=1e-8)


def test_steady_gradient_with_respect_to_removal():
    """0 = -(q + r) x + s with q = 1, s = 1: x = 1/(1 + r) = 2/3 at r = 0.5,
    dx/dr = -1/(1 + r)^2 = -4/9."""
    removal = _t([0.5], requires_grad=True)
    layer = _layer("implicit", removal=removal)
    x = layer.steady(_t([1.0]), _t([0.0, 1.0]), _t([0.0]))
    assert x.item() == pytest.approx(2 / 3, rel=1e-10)
    (dr,) = torch.autograd.grad(x.sum(), (removal,), allow_unused=True)
    assert dr is not None, "removal received no gradient"
    assert dr.item() == pytest.approx(-4 / 9, rel=1e-8)


def test_implicit_gradient_survives_an_outer_transform_of_the_coefficient():
    """removal = exp(theta): d x1/d theta = d x1/d r * r, with d x1/d r = -1/2.5^2 at r = 0.5."""
    theta = _t([math.log(0.5)], requires_grad=True)
    layer = _layer("implicit", removal=torch.exp(theta))
    y = layer.step(_t([1.0]), _t([1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0)
    (dtheta,) = torch.autograd.grad(y.sum(), (theta,), allow_unused=True)
    assert dtheta is not None
    assert dtheta.item() == pytest.approx(-0.16 * 0.5, rel=1e-8)


def test_implicit_gradient_with_respect_to_carrier():
    """carrier c scales the flow: dx/dt = -c q x; implicit x1 = 1/(1 + c) -> dx1/dc = -1/(1+c)^2."""
    carrier = torch.tensor(1.0, dtype=F64, requires_grad=True)
    layer = _layer("implicit", carrier=carrier)
    y = layer.step(_t([1.0]), _t([1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0)
    (dc,) = torch.autograd.grad(y.sum(), (carrier,), allow_unused=True)
    assert dc is not None
    assert y.item() == pytest.approx(0.5)
    assert dc.item() == pytest.approx(-0.25, rel=1e-8)


def test_implicit_gradient_with_respect_to_transmission():
    """Circulating compartment, x_boundary = 1: dx/dt = t_in q x_b - q x.

    implicit x1 = (x0 + dt t_in) / (1 + dt) = (1 + t_in)/2 -> d x1/d t_in = 1/2, and the
    outflow edge's transmission (index 0) does not enter the interior balance at all.
    """
    transmission = _t([1.0, 1.0], requires_grad=True)
    layer = _layer("implicit", circulation=True, transmission=transmission)
    y = layer.step(_t([1.0]), _t([1.0, 1.0]), _t([0.0, 0.0]), _t([1.0]), 1.0)
    (dt_,) = torch.autograd.grad(y.sum(), (transmission,), allow_unused=True)
    assert dt_ is not None
    assert y.item() == pytest.approx(1.0)
    torch.testing.assert_close(dt_, _t([0.0, 0.5]), rtol=1e-8, atol=1e-12)
