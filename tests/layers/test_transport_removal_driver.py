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


@pytest.mark.parametrize("scheme", ["exact", "implicit", "trapezoidal"])
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


def test_steady_and_rate_take_it_too():
    built, bare = _layer(removal=R), _layer()
    torch.testing.assert_close(bare.steady(Q, SRC, XB, removal=R), built.steady(Q, SRC, XB),
                               rtol=1e-14, atol=0)
    torch.testing.assert_close(bare.rate(X0, Q, SRC, XB, removal=R),
                               built.rate(X0, Q, SRC, XB), rtol=1e-14, atol=0)


def test_the_per_call_removal_replaces_the_layers_own_and_accepts_a_species_free_shape():
    built = _layer(removal=_t([[5.0], [5.0]]))
    reference = _layer(removal=R)
    torch.testing.assert_close(built.step(X0, Q, SRC, XB, 10.0, removal=R[:, 0]),
                               reference.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)


def test_a_batched_removal_steps_each_instance_with_its_own():
    bare = _layer()
    batch = torch.stack([R, 2 * R])                                   # (2, n_i, 1)
    out = bare.step(X0, Q, SRC, XB, 10.0, removal=batch)
    assert out.shape == (2, 2)
    for i in range(2):
        torch.testing.assert_close(out[i], bare.step(X0, Q, SRC, XB, 10.0, removal=batch[i]),
                                   rtol=1e-13, atol=0)


def test_the_step_is_differentiable_in_the_removal_rate():
    """A sealed zone (q = 0, no source): implicit x1 = x0 / (1 + k dt), so
    dx1/dk = -x0 dt / (1 + k dt)**2."""
    bare = _layer()
    k = _t([[0.05], [0.2]]).requires_grad_()
    x1 = bare.step(X0, torch.zeros(3, dtype=F64), torch.zeros(3, dtype=F64), XB, 10.0,
                   removal=k)
    (g,) = torch.autograd.grad(x1.sum(), k)
    expected = -X0 * 10.0 / (1 + k.detach()[:, 0] * 10.0) ** 2
    torch.testing.assert_close(g[:, 0], expected, rtol=1e-12, atol=0)


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
