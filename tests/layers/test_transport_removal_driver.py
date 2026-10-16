"""A per-step first-order removal rate (the driver "<layer>.removal") replaces the layer's
construction-time `removal`, in every scheme, in steady and in rate, batched and
differentiably."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def _layer(scheme="implicit", removal=None, K=1):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("a")
    net.add_node("b")
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    return TransportLayer(net, "c", capacity=_t([2.0, 3.0]), flow_kind="flow",
                          boundary=["ambient"], n_species=K, scheme=scheme, removal=removal)


Q = _t([0.5, 0.5, 0.5])
SRC = _t([0.0, 0.1, 0.0])
XB = _t([0.2])
X0 = _t([0.3, 0.1])
R = _t([[0.05], [0.2]])
SCHEMES = ["exact", "implicit", "trapezoidal"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_per_call_removal_matches_the_construction_time_one(scheme):
    built = _layer(scheme, removal=R)
    bare = _layer(scheme)
    expected = built.step(X0, Q, SRC, XB, 10.0)
    got = bare.step(X0, Q, SRC, XB, 10.0, removal=R)
    torch.testing.assert_close(got, expected, rtol=1e-14, atol=0)
    tr_exp = built.step_with_transfer(X0, Q, SRC, XB, 10.0)
    tr_got = bare.step_with_transfer(X0, Q, SRC, XB, 10.0, removal=R)
    torch.testing.assert_close(tr_got.x, tr_exp.x, rtol=1e-14, atol=0)
    torch.testing.assert_close(tr_got.boundary_transfer, tr_exp.boundary_transfer,
                               rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_steady_and_rate_take_it_too(scheme):
    built, bare = _layer(scheme, removal=R), _layer(scheme)
    torch.testing.assert_close(bare.steady(Q, SRC, XB, removal=R), built.steady(Q, SRC, XB),
                               rtol=1e-14, atol=0)
    torch.testing.assert_close(bare.rate(X0, Q, SRC, XB, removal=R),
                               built.rate(X0, Q, SRC, XB), rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_per_call_removal_replaces_the_layers_own_and_accepts_a_species_free_shape(scheme):
    built = _layer(scheme, removal=_t([[5.0], [5.0]]))
    reference = _layer(scheme, removal=R)
    torch.testing.assert_close(built.step(X0, Q, SRC, XB, 10.0, removal=R[:, 0]),
                               reference.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_batched_removal_steps_each_instance_with_its_own(scheme):
    bare = _layer(scheme)
    batch = torch.stack([R, 2 * R])                                   # (2, n_i, 1)
    out = bare.step(X0, Q, SRC, XB, 10.0, removal=batch)
    assert out.shape == (2, 2)
    for i in range(2):
        torch.testing.assert_close(out[i], bare.step(X0, Q, SRC, XB, 10.0, removal=batch[i]),
                                   rtol=1e-13, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_step_is_differentiable_in_the_removal_rate(scheme):
    """A sealed zone (q = 0, no source) has x1 = x0 g(k dt), with g = exp(-k dt) (exact),
    1 / (1 + k dt) (implicit) or (1 - k dt / 2) / (1 + k dt / 2) (trapezoidal); the
    gradient is x0 dt g'(k dt)."""
    bare = _layer(scheme)
    k = _t([[0.05], [0.2]]).requires_grad_()
    x1 = bare.step(X0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64), XB, 10.0,
                   removal=k)
    (g,) = torch.autograd.grad(x1.sum(), k)
    z = k.detach()[:, 0] * 10.0
    dg = {"exact": -torch.exp(-z),
          "implicit": -1 / (1 + z) ** 2,
          "trapezoidal": -1 / (1 + z / 2) ** 2}[scheme]
    # the exact scheme differentiates through a matrix exponential: a few 1e-12 of roundoff
    rtol = 1e-10 if scheme == "exact" else 1e-12
    torch.testing.assert_close(g[:, 0], X0 * 10.0 * dg, rtol=rtol, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_each_species_takes_its_own_removal_rate(scheme):
    """Two species with no flow-coupled difference: species one removed at the rate of a
    one-species layer, species two untouched, and the cast to float64 of a float32 rate."""
    two = _layer(scheme, K=2)
    one = _layer(scheme)
    x0 = torch.stack([X0, X0], dim=-1)                                # (n_i, 2)
    sources = torch.stack([SRC, SRC], dim=-1)                         # (n, 2)
    xb = torch.stack([XB, XB], dim=-1)                                # (n_b, 2)
    rate = torch.cat([R, torch.zeros_like(R)], dim=-1)                # (n_i, 2)
    out = two.step(x0, Q, sources, xb, 10.0, removal=rate.to(torch.float32))
    assert out.dtype == F64
    torch.testing.assert_close(out[:, 0], one.step(X0, Q, SRC, XB, 10.0, removal=R),
                               rtol=1e-6, atol=0)
    torch.testing.assert_close(out[:, 1], one.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)


def test_a_misshapen_removal_is_refused_naming_the_layer():
    with pytest.raises(ValueError, match=r"'c'.*removal must have trailing shape \(2, 1\)"):
        _layer().step(X0, Q, SRC, XB, 10.0, removal=_t([0.1, 0.2, 0.3]))


def test_the_model_passes_the_removal_driver_a_closure_writes():
    layer = _layer()
    net = layer.net

    def sinks(state, drivers):
        return {"c.q": Q, "c.removal": R}

    model = Model(net, {"c": layer}, closures=[sinks])
    state = {"c.x": X0}
    drivers = {"c.x_boundary": XB, "c.sources": SRC}
    torch.testing.assert_close(model.step(state, drivers, 10.0)["c.x"],
                               _layer(removal=R).step(X0, Q, SRC, XB, 10.0),
                               rtol=1e-14, atol=0)
    torch.testing.assert_close(model.steady(state, drivers)["c.x"],
                               _layer(removal=R).steady(Q, SRC, XB), rtol=1e-14, atol=0)
    field = model.refs.field("c.removal")
    assert field.unit == "1/s" and field.ordering == "active interior order"
    assert field.required == "optional"
