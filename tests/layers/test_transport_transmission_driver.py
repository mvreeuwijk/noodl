"""A per-step transmission (the driver "<layer>.transmission") replaces the layer's
construction-time `transmission`, in every scheme, in steady and in rate, batched and
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


def _layer(scheme="implicit", transmission=None, K=1):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("a")
    net.add_node("b")
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    return TransportLayer(net, "c", capacity=_t([2.0, 3.0]), flow_kind="flow",
                          boundary=["ambient"], n_species=K, scheme=scheme,
                          transmission=transmission)


Q = _t([0.5, 0.5, 0.5])
SRC = _t([0.0, 0.1, 0.0])
XB = _t([0.2])
X0 = _t([0.3, 0.1])
TAU = _t([0.6, 0.9, 1.0])
SCHEMES = ["exact", "implicit", "trapezoidal"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_per_call_transmission_matches_the_construction_time_one(scheme):
    built = _layer(scheme, transmission=TAU)
    bare = _layer(scheme)
    torch.testing.assert_close(bare.step(X0, Q, SRC, XB, 10.0, transmission=TAU),
                               built.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)
    tr_exp = built.step_with_transfer(X0, Q, SRC, XB, 10.0)
    tr_got = bare.step_with_transfer(X0, Q, SRC, XB, 10.0, transmission=TAU)
    torch.testing.assert_close(tr_got.x, tr_exp.x, rtol=1e-14, atol=0)
    torch.testing.assert_close(tr_got.boundary_transfer, tr_exp.boundary_transfer,
                               rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_steady_and_rate_take_it_too(scheme):
    built, bare = _layer(scheme, transmission=TAU), _layer(scheme)
    torch.testing.assert_close(bare.steady(Q, SRC, XB, transmission=TAU),
                               built.steady(Q, SRC, XB), rtol=1e-14, atol=0)
    torch.testing.assert_close(bare.rate(X0, Q, SRC, XB, transmission=TAU),
                               built.rate(X0, Q, SRC, XB), rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_per_call_transmission_replaces_the_layers_own(scheme):
    built = _layer(scheme, transmission=_t([0.1, 0.1, 0.1]))
    reference = _layer(scheme, transmission=TAU)
    torch.testing.assert_close(built.step(X0, Q, SRC, XB, 10.0, transmission=TAU[:, None]),
                               reference.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_batched_transmission_steps_each_instance_with_its_own(scheme):
    bare = _layer(scheme)
    batch = torch.stack([TAU, 0.5 * TAU])                             # (2, b_flow)
    out = bare.step(X0, Q, SRC, XB, 10.0, transmission=batch)
    assert out.shape == (2, 2)
    for i in range(2):
        torch.testing.assert_close(out[i], bare.step(X0, Q, SRC, XB, 10.0,
                                                     transmission=batch[i]),
                                   rtol=1e-13, atol=0)


def test_the_steady_state_is_differentiable_in_the_transmission():
    """Steady chain: x_a = tau_1 x_b(ambient), x_b = tau_2 x_a + s / q, so
    dx_b / dtau_1 = tau_2 x_amb."""
    bare = _layer()
    tau = TAU.clone().requires_grad_()
    x = bare.steady(Q, SRC, XB, transmission=tau)
    (g,) = torch.autograd.grad(x[1], tau)
    torch.testing.assert_close(g[0], TAU[1] * XB[0], rtol=1e-12, atol=0)


def test_each_species_takes_its_own_transmission():
    two = _layer(K=2)
    one = _layer()
    x0 = torch.stack([X0, X0], dim=-1)
    sources = torch.stack([SRC, SRC], dim=-1)
    xb = torch.stack([XB, XB], dim=-1)
    tau = torch.stack([TAU, torch.ones(3, dtype=F64)], dim=-1)        # (b_flow, 2)
    out = two.step(x0, Q, sources, xb, 10.0, transmission=tau)
    torch.testing.assert_close(out[:, 0], one.step(X0, Q, SRC, XB, 10.0, transmission=TAU),
                               rtol=1e-14, atol=0)
    torch.testing.assert_close(out[:, 1], one.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)


def test_a_misshapen_transmission_is_refused_naming_the_layer():
    with pytest.raises(ValueError, match=r"'c'.*transmission must have trailing shape \(3,\)"):
        _layer().step(X0, Q, SRC, XB, 10.0, transmission=_t([0.1, 0.2]))


def test_the_model_passes_the_transmission_driver_a_closure_writes():
    layer = _layer()
    net = layer.net

    def filters(state, drivers):
        return {"c.q": Q, "c.transmission": TAU}

    model = Model(net, {"c": layer}, closures=[filters])
    state = {"c.x": X0}
    drivers = {"c.x_boundary": XB, "c.sources": SRC}
    reference = _layer(transmission=TAU)
    torch.testing.assert_close(model.step(state, drivers, 10.0)["c.x"],
                               reference.step(X0, Q, SRC, XB, 10.0), rtol=1e-14, atol=0)
    torch.testing.assert_close(model.steady(state, drivers)["c.x"],
                               reference.steady(Q, SRC, XB), rtol=1e-14, atol=0)
    torch.testing.assert_close(model.residuals({**state, "c.x": X0}, drivers)["c"],
                               reference.rate(X0, Q, SRC, XB), rtol=1e-14, atol=0)
    field = model.refs.field("c.transmission")
    assert field.ordering == "edge order of kinds ('flow',)" and field.required == "optional"


def test_a_closure_that_reads_the_flows_it_feeds_needs_the_iterated_coupling():
    class FlowReader:
        requires_coupling = "iterate"
        reads_flows_of = "air"

        def __call__(self, state, drivers):
            return {}

    layer = _layer()
    with pytest.raises(ValueError, match=r"FlowReader reads the flows of layer 'air'"):
        Model(layer.net, {"c": layer}, closures=[FlowReader()])
