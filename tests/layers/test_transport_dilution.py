"""`TransportLayer(dilution=...)`: the balance of a quantity stored in a volume whose content
changes with the net flow into it,

    V dx/dt = (In(q) - Out(q)) x - dilution * net(q) x + N x_b + sources.

Independent references: closed-form solutions of the frozen-coefficient ODE, the amount
balance of a filling tank, and finite differences for the gradients.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def _tank(*, dilution, scheme="exact", carrier=1.0, n_species=1):
    """`zone` fed from `ambient` by one edge, drained by nothing: net(q) = q."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("ambient", "zone", kind="flow")
    return TransportLayer(net, "c", capacity=_t([2.0]), flow_kind="flow",
                          boundary=["ambient"], carrier=carrier, dilution=dilution,
                          scheme=scheme, n_species=n_species)


def test_advective_form_relaxes_to_the_inflow_exponentially():
    """dilution = carrier: V dx/dt = c q (x_b - x), so x = x_b + (x0 - x_b) e^{-c q t/V},
    although the zone has no outflow at all (the flux form would add c q x_b forever)."""
    c, q, V, dt = 3.0, 0.5, 2.0, 1.7
    layer = _tank(dilution=c, carrier=c)
    x = layer.step(_t([1.0]), _t([q]), _t([0.0, 0.0]), _t([4.0]), dt)
    assert float(x) == pytest.approx(4.0 + (1.0 - 4.0) * math.exp(-c * q * dt / V), rel=1e-13)
    # The plain flux form, for contrast: V dx/dt = c q x_b.
    flux = _tank(dilution=None, carrier=c).step(_t([1.0]), _t([q]), _t([0.0, 0.0]), _t([4.0]),
                                                dt)
    assert float(flux) == pytest.approx(1.0 + c * q * 4.0 * dt / V, rel=1e-13)


def test_partial_dilution_is_an_amount_whose_capacity_grows_by_dilution_times_net():
    """dilution = lam < carrier: V dx/dt = c q x_b - lam q x, the balance of the amount
    V x whose capacity grows at lam q. With x_b = 0 and V frozen the solution is
    x0 e^{-lam q t/V}; the implicit scheme gives x0 / (1 + lam q dt/V)."""
    lam, q, V, dt = 0.4, 0.5, 2.0, 1.3
    x_exact = _tank(dilution=lam, carrier=1.0).step(
        _t([1.0]), _t([q]), _t([0.0, 0.0]), _t([0.0]), dt)
    assert float(x_exact) == pytest.approx(math.exp(-lam * q * dt / V), rel=1e-13)
    x_impl = _tank(dilution=lam, scheme="implicit").step(
        _t([1.0]), _t([q]), _t([0.0, 0.0]), _t([0.0]), dt)
    assert float(x_impl) == pytest.approx(1.0 / (1.0 + lam * q * dt / V), rel=1e-13)


def test_balanced_flows_are_unchanged_by_dilution():
    """With net(q) = 0 at every interior node the dilution term vanishes identically."""
    net = Network(dtype=F64)
    for nm in ("in", "a", "b", "out"):
        net.add_node(nm)
    for s, t in (("in", "a"), ("a", "b"), ("b", "out")):
        net.add_edge(s, t, kind="flow")
    kw = dict(capacity=_t([1.0, 2.0]), flow_kind="flow", boundary=["in", "out"], carrier=2.0)
    q = _t([0.3, 0.3, 0.3])
    xb, src, x0 = _t([5.0, 0.0]), _t([0.0, 0.0, 0.0, 0.0]), _t([1.0, 2.0])
    plain = TransportLayer(net, "c", **kw).step(x0, q, src, xb, 2.0)
    diluted = TransportLayer(net, "c", dilution=2.0, **kw).step(x0, q, src, xb, 2.0)
    assert torch.equal(plain, diluted)


def test_capacity_prev_is_a_coefficient_not_an_amount_with_dilution():
    """With dilution the capacity is the rate coefficient at the end of the step, so the
    exact scheme accepts a changing capacity (it refuses one without dilution) and ignores
    `capacity_prev`."""
    layer = _tank(dilution=1.0)
    args = (_t([1.0]), _t([0.5]), _t([0.0, 0.0]), _t([4.0]), 1.0)
    a = layer.step(*args, capacity=_t([3.0]), capacity_prev=_t([2.0]))
    b = layer.step(*args, capacity=_t([3.0]))
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="changing-capacity"):
        _tank(dilution=None).step(*args, capacity=_t([3.0]), capacity_prev=_t([2.0]))


def test_dilution_is_per_species():
    """dilution (K,): species 0 advective (relaxes to x_b), species 1 flux form."""
    q, V, dt = 0.5, 2.0, 1.0
    layer = _tank(dilution=_t([1.0, 0.0]), n_species=2)
    x = layer.step(_t([[1.0, 1.0]]), _t([q]), _t([[0.0, 0.0], [0.0, 0.0]]), _t([[4.0, 4.0]]),
                   dt)
    assert float(x[0, 0]) == pytest.approx(4.0 - 3.0 * math.exp(-q * dt / V), rel=1e-13)
    assert float(x[0, 1]) == pytest.approx(1.0 + q * 4.0 * dt / V, rel=1e-13)
    with pytest.raises(ValueError, match="dilution must be"):
        _tank(dilution=_t([1.0, 0.0, 1.0]), n_species=2)


def test_dense_operator_matches_the_sparse_one():
    """`operator()` (the dense reference) and the stepping operator carry the same term."""
    net = Network(dtype=F64)
    for nm in ("b", "z1", "z2"):
        net.add_node(nm)
    net.add_edge("b", "z1", kind="flow")
    net.add_edge("z1", "z2", kind="flow")
    net.add_edge("z2", "b", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([1.5, 0.7]), flow_kind="flow",
                           boundary=["b"], carrier=1.2, dilution=0.8)
    q = _t([0.9, 0.4, -0.2])      # unbalanced: net(z1) = 0.5, net(z2) = 0.6
    M, _ = layer.operator(q)
    op = layer._advection_operator(q)
    eye = torch.eye(2, dtype=F64)
    sparse = torch.stack([op.matvec(eye[j]) for j in range(2)], dim=-1)
    assert torch.allclose(M, sparse, rtol=1e-14, atol=1e-15)
    assert torch.allclose(layer.net_inflow(q), _t([0.5, 0.6]), rtol=0, atol=1e-15)


@pytest.mark.parametrize("scheme", ["exact", "implicit"])
def test_step_gradcheck_through_the_flows(scheme):
    """The dilution term is differentiable in `q`: gradcheck of the step's output, for an
    unbalanced network (net inflow at both interior nodes)."""
    net = Network(dtype=F64)
    for nm in ("b", "z1", "z2"):
        net.add_node(nm)
    net.add_edge("b", "z1", kind="flow")
    net.add_edge("z1", "z2", kind="flow")
    net.add_edge("b", "z2", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([1.5, 0.7]), flow_kind="flow",
                           boundary=["b"], carrier=1.2, dilution=0.8, scheme=scheme)

    def f(q):
        return layer.step(_t([1.0, 2.0]), q, _t([0.0, 0.1, 0.0]), _t([3.0]), 0.9)

    q0 = _t([0.6, 0.2, 0.3]).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (q0,), eps=1e-6, atol=1e-8)
