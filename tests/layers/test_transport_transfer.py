"""`step_with_transfer`: the layer's own discrete balance closes with its boundary transfer."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer, TransportStep
from noodl.topology import Network

F64 = torch.float64


def _t(v, **kw):
    return torch.tensor(v, dtype=F64, **kw)


def _two_zone(scheme, *, transmission=None):
    net = Network(dtype=F64)
    for n in ("amb", "a", "b", "out"):
        net.add_node(n)
    for u, v in [("amb", "a"), ("a", "b"), ("b", "out"), ("b", "a"), ("out", "b")]:
        net.add_edge(u, v, kind="flow")
    kw = {} if transmission is None else {"transmission": transmission}
    return TransportLayer(net, "c", capacity=_t([2.0, 0.5]), flow_kind="flow",
                          boundary=["amb", "out"], scheme=scheme, **kw)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal", "exact"])
@pytest.mark.parametrize("q", [[1.0, 0.7, 0.4, 0.3, 0.6], [-1.0, -0.7, 0.4, 0.3, -0.6]])
def test_amount_balance_closes_with_the_boundary_transfer(scheme, q):
    layer = _two_zone(scheme)
    x0 = _t([1.5, 0.25])
    xb = _t([3.0, 0.5])
    sources = _t([0.0, 0.2, -0.1, 0.0])           # full node order: amb, a, b, out
    dt = 0.8
    out = layer.step_with_transfer(x0, _t(q), sources, xb, dt)
    assert isinstance(out, TransportStep)
    torch.testing.assert_close(
        out.x, layer.step(x0, _t(q), sources, xb, dt), rtol=1e-12, atol=1e-14
    )
    V = layer.capacity
    gained = (V * (out.x - x0)).sum()
    sourced = dt * sources[1:3].sum()
    tol = 1e-9 if scheme == "exact" else 1e-12
    assert (gained - sourced + out.boundary_transfer.sum()).abs().item() < tol


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_balance_closes_across_a_changing_capacity(scheme):
    layer = _two_zone(scheme)
    x0, xb = _t([1.5, 0.25]), _t([3.0, 0.5])
    q = _t([1.0, 0.7, 0.4, 0.3, 0.6])
    sources = torch.zeros(4, dtype=F64)
    v_old, v_new = _t([2.0, 0.5]), _t([2.6, 0.4])
    out = layer.step_with_transfer(x0, q, sources, xb, 0.8, capacity=v_new, capacity_prev=v_old)
    gained = (v_new * out.x - v_old * x0).sum()
    assert (gained + out.boundary_transfer.sum()).abs().item() < 1e-12


def test_batched_flows_give_per_instance_transfers():
    layer = _two_zone("implicit")
    q = torch.stack([_t([1.0, 0.7, 0.4, 0.3, 0.6]), _t([0.2, 0.1, 0.9, 0.0, 0.05])])
    x0, xb, sources = _t([1.5, 0.25]), _t([3.0, 0.5]), torch.zeros(4, dtype=F64)
    out = layer.step_with_transfer(x0, q, sources, xb, 0.8)
    assert out.boundary_transfer.shape == (2, 2)
    for i in range(2):
        single = layer.step_with_transfer(x0, q[i], sources, xb, 0.8)
        torch.testing.assert_close(
            out.boundary_transfer[i], single.boundary_transfer, rtol=1e-12, atol=1e-14
        )


def test_stacked_single_species_layout_is_preserved():
    layer = _two_zone("implicit")
    out = layer.step_with_transfer(_t([[1.5], [0.25]]), _t([1.0, 0.7, 0.4, 0.3, 0.6]),
                                   torch.zeros(4, 1, dtype=F64), _t([[3.0], [0.5]]), 0.8)
    assert out.x.shape == (2, 1) and out.boundary_transfer.shape == (2, 1)


def test_exact_scheme_transfer_matches_the_dense_integral():
    """The exact transfer is the boundary functional of int_0^dt x(tau) dtau plus dt * x_b."""
    layer = _two_zone("exact")
    x0, xb, dt = _t([1.5, 0.25]), _t([3.0, 0.5]), 0.8
    q = _t([1.0, 0.7, 0.4, 0.3, 0.6])
    out = layer.step_with_transfer(x0, q, torch.zeros(4, dtype=F64), xb, dt)
    op = layer._advection_operator(q)
    M = op.assemble()
    m = M.shape[-1]
    Z = torch.zeros(3 * m, 3 * m, dtype=F64)
    eye = torch.eye(m, dtype=F64)
    Z[:m, :m] = M * dt
    Z[:m, m:2 * m] = eye * dt
    Z[m:2 * m, 2 * m:] = eye * dt
    E = torch.linalg.matrix_exp(Z)
    b0 = op.boundary_forcing(xb)
    integral = E[:m, m:2 * m] @ x0 + E[:m, 2 * m:] @ b0
    expected = op.boundary_net_inflow(integral, dt * xb)
    torch.testing.assert_close(out.boundary_transfer, expected, rtol=1e-9, atol=1e-12)
