"""Cross-cutting invariants of the core, asserted over feature combinations.

These are physical or structural identities (ensemble equivalence, species permutation,
constant-tracer preservation, composed == standalone, warm-start independence, finite-
difference gradients), not snapshots. Application-level counterparts live in
`tests/apps/<app>/test_invariants.py`.
"""

from __future__ import annotations

import pytest
import torch

from noodl.couple import union
from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.layers.allocation import AllocatedFlowLayer
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
RTOL, ATOL = 1e-9, 1e-15

# ambient -> z1 -> z2 -> ambient: 3 nodes, 2 interior nodes, 3 edges. The batch sizes below
# include each of those counts, so a batch axis mistaken for a node or edge axis (a
# "coincidental size") is caught: 1, 2 (interior), 3 (nodes AND edges), 4.
BATCH_SIZES = [1, 2, 3, 4]


def _net() -> Network:
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    return net


def _model(*, n_species=1, closures=(), reactions=(), learnable=False, scheme="implicit", **kw):
    net = _net()
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65, learnable=learnable)
    air = KirchhoffLayer(net, "air", [el], drives=[ConstantDrive("airpath", "wind")],
                         boundary=["ambient"])
    gas = TransportLayer(net, "gas", capacity=torch.tensor([50.0, 80.0], dtype=F64),
                         flow_kind="airpath", boundary=["ambient"], scheme=scheme,
                         n_species=n_species)
    model = Model(net, {"air": air, "gas": gas}, closures=closures, reactions=reactions, **kw)
    k = (n_species,) if n_species > 1 else ()
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "gas.x_boundary": torch.tensor([1e-3, 2e-3, 4e-3][:n_species], dtype=F64).reshape(1, *k),
        "gas.sources": torch.zeros(3, *k, dtype=F64),
    }
    state = {"gas.x": torch.zeros(2, *k, dtype=F64)}
    return model, state, drivers, el


def _winds(b):
    w = torch.zeros(b, 3, dtype=F64)
    w[:, 0] = torch.linspace(2.0, 9.0, b, dtype=F64)
    return w


def _unbatch(d, ref, i):
    """Instance `i` of every driver that carries more dims than its unbatched reference."""
    return {k: (v[i] if v.dim() > ref[k].dim() else v) for k, v in d.items()}


# ---------------------------------------------------------------- ensemble equivalence


@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("batched", ["wind", "boundary", "sources", "all"])
def test_model_batch_equals_stack_of_independent_runs(b, batched):
    """A batched driver with an UNBATCHED initial state equals the stack of single runs."""
    model, state, drivers, _ = _model()
    d = dict(drivers)
    if batched in ("wind", "all"):
        d["wind"] = _winds(b)
    if batched in ("boundary", "all"):
        d["gas.x_boundary"] = torch.linspace(1e-3, 5e-3, b, dtype=F64).reshape(b, 1)
    if batched in ("sources", "all"):
        src = torch.zeros(b, 3, dtype=F64)
        src[:, 1] = torch.linspace(1e-6, 3e-6, b, dtype=F64)
        d["gas.sources"] = src
    out = model.step(state, d, 600.0)
    # A batch of one transport driver leaves the (unbatched) flow solution unbatched, and a
    # (1, 1) boundary is the stacked (boundary, species) layout: compare broadcast shapes.
    xs = out["gas.x"].reshape(b, 2)
    for i in range(b):
        single = model.step(state, _unbatch(d, drivers, i), 600.0)
        torch.testing.assert_close(xs[i], single["gas.x"], rtol=1e-8, atol=1e-14)
        for key in ("air.q", "air.phi"):
            full = out[key].expand(b, -1) if out[key].dim() == 1 else out[key]
            torch.testing.assert_close(full[i], single[key], rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("b", BATCH_SIZES)
def test_model_batched_state_with_unbatched_drivers(b):
    model, _, drivers, _ = _model()
    x0 = torch.linspace(0.0, 3e-3, b * 2, dtype=F64).reshape(b, 2)
    out = model.step({"gas.x": x0}, drivers, 600.0)["gas.x"]
    assert out.shape == (b, 2)
    for i in range(b):
        single = model.step({"gas.x": x0[i]}, drivers, 600.0)["gas.x"]
        torch.testing.assert_close(out[i], single, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("b", BATCH_SIZES)
def test_model_steady_batch_equals_stack_of_independent_runs(b):
    model, state, drivers, _ = _model()
    d = dict(drivers, wind=_winds(b))
    out = model.steady(state, d)["gas.x"]
    for i in range(b):
        single = model.steady(state, dict(d, wind=d["wind"][i]))["gas.x"]
        torch.testing.assert_close(out[i], single, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("mode", ["hard", "smooth", "projection"])
def test_allocation_batch_equals_stack_with_unbatched_state(b, mode):
    net = Network()
    for n in "ABC":
        net.add_node(n)
    net.add_edge("A", "B", kind="link")
    net.add_edge("B", "C", kind="link")
    layer = AllocatedFlowLayer(
        net, "cap", "link", mode=mode, tau=0.01 if mode == "smooth" else None,
        s_max=torch.full((3,), 10.0, dtype=F64), c_arc=torch.ones(2, dtype=F64),
    )
    s = torch.tensor([0.0, 1.0, 2.0], dtype=F64)
    r = torch.linspace(0.1, 3.0, b * 2, dtype=F64).reshape(b, 2)
    s_new, f = layer.step(s, {"cap.requests": r}, 1.0)
    assert s_new.shape == (b, 3) and f.shape == (b, 2)
    for i in range(b):
        s_i, f_i = layer.step(s, {"cap.requests": r[i]}, 1.0)
        torch.testing.assert_close(s_new[i], s_i)
        torch.testing.assert_close(f[i], f_i)


@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("warm", [False, True])
def test_kirchhoff_batch_equals_stack_with_optional_unbatched_warm_start(b, warm):
    net = _net()
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)
    layer = KirchhoffLayer(net, "air", [el], drives=[ConstantDrive("airpath", "wind")],
                           boundary=["ambient"])
    wind = _winds(b)
    phi_b = torch.zeros(1, dtype=F64)
    phi0 = torch.tensor([1.0, -2.0], dtype=F64) if warm else None
    phi, q = layer.solve(phi_b, {"wind": wind}, None, phi0=phi0, differentiable=False)
    assert phi.shape[0] == b and q.shape[0] == b
    for i in range(b):
        p_i, q_i = layer.solve(phi_b, {"wind": wind[i]}, None, phi0=phi0, differentiable=False)
        torch.testing.assert_close(q[i], q_i, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(phi[i], p_i, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("b", BATCH_SIZES)
def test_transport_layer_batched_flows_equal_stack(b):
    net = _net()
    layer = TransportLayer(net, "gas", capacity=torch.tensor([50.0, 80.0], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"], scheme="implicit")
    q = 0.01 + 0.02 * torch.rand(b, 3, dtype=F64)
    x = torch.rand(b, 2, dtype=F64)
    xb = torch.rand(b, 1, dtype=F64)
    src = torch.zeros(b, 3, dtype=F64)
    out = layer.step(x, q, src, xb, 100.0)
    for i in range(b):
        torch.testing.assert_close(out[i], layer.step(x[i], q[i], src[i], xb[i], 100.0),
                                   rtol=1e-9, atol=1e-14)
    # an unbatched state against batched flows
    out2 = layer.step(x[0], q, src, xb, 100.0)
    assert out2.shape == (b, 2)
    for i in range(b):
        torch.testing.assert_close(out2[i], layer.step(x[0], q[i], src[i], xb[i], 100.0),
                                   rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------- species permutation


@pytest.mark.parametrize("perm", [[1, 0], [2, 0, 1], [1, 2, 0]])
@pytest.mark.parametrize("with_decay", [False, True])
def test_model_species_permutation(perm, with_decay):
    """Permuting the species axis of state, boundary, sources and reaction rates permutes
    the result."""
    n = len(perm)
    rates = torch.tensor([0.0, 2e-3, 5e-4][:n], dtype=F64)
    p = torch.tensor(perm)

    def run(order):
        rx = [("gas", FirstOrderDecay(rates[order]))] if with_decay else []
        model, _, drivers, _ = _model(n_species=n, reactions=rx)
        src = torch.zeros(3, n, dtype=F64)
        src[1] = torch.tensor([1e-6, 2e-6, 3e-6][:n], dtype=F64)
        x0 = torch.tensor([[1e-3, 5e-4, 2e-3][:n]] * 2, dtype=F64)
        d = dict(drivers, **{"gas.x_boundary": drivers["gas.x_boundary"][..., order],
                             "gas.sources": src[..., order]})
        return model.step({"gas.x": x0[..., order]}, d, 600.0)["gas.x"]

    ident = torch.arange(n)
    torch.testing.assert_close(run(p), run(ident)[..., p], rtol=RTOL, atol=ATOL)


def test_passive_species_in_a_decaying_mix_stays_unreacted():
    """A zero-rate species is the identity: its columns equal the model without any decay."""
    rates = torch.tensor([0.0, 2e-3, 0.0], dtype=F64)
    model, state, drivers, _ = _model(n_species=3)
    plain = model.step(state, drivers, 600.0)["gas.x"]
    decaying, _, _, _ = _model(n_species=3, reactions=[("gas", FirstOrderDecay(rates))])
    out = decaying.step(state, drivers, 600.0)["gas.x"]
    torch.testing.assert_close(out[..., [0, 2]], plain[..., [0, 2]], rtol=RTOL, atol=ATOL)
    assert (out[..., 1] < plain[..., 1]).all()


# ---------------------------------------------------------------- constant tracer


@pytest.mark.parametrize("n_species", [1, 3])
@pytest.mark.parametrize("b", [None, 3])
@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_uniform_conservative_tracer_stays_uniform(n_species, b, scheme):
    """x == c in the interior, c at the boundary, no sources: x stays c, whatever the flow,
    batch shape or species count."""
    model, _, drivers, _ = _model(n_species=n_species, scheme=scheme)
    k = (n_species,) if n_species > 1 else ()
    c = torch.tensor([1.5e-3, 7e-4, 3e-3][:n_species], dtype=F64)
    d = dict(drivers, **{"gas.x_boundary": c.reshape(1, *k).clone()})
    if b:
        d["wind"] = _winds(b)
    state = {"gas.x": c.expand(2, *k).clone()}
    for _ in range(3):
        state = model.step(state, d, 600.0, atol=1e-15, rtol=1e-13)
    expect = c.expand(*(([b] if b else []) + [2] + list(k)))
    torch.testing.assert_close(state["gas.x"], expect, rtol=1e-9, atol=1e-13)


# ---------------------------------------------------------------- composed vs standalone


def _gain_closure(state, drivers):
    """Wind that grows with z1's concentration: a genuine flow <- transport feedback."""
    x1 = state["gas.x"][..., 0]
    zero = torch.zeros_like(x1)
    return {"wind": drivers["wind"] + torch.stack([2e3 * x1, zero, zero], dim=-1)}


def _conditional_closure(state, drivers):
    """The gain switches on z1's concentration: a conditional closure."""
    x1 = state["gas.x"][..., 0]
    zero = torch.zeros_like(x1)
    gain = torch.where(x1 > 5e-4, torch.full_like(x1, 3e3), torch.full_like(x1, 1e3))
    return {"wind": drivers["wind"] + torch.stack([gain * x1, zero, zero], dim=-1)}


@pytest.mark.parametrize("closure", [None, _gain_closure])
@pytest.mark.parametrize("b", [None, 3])
def test_union_of_one_model_equals_standalone(closure, b):
    kw = {} if closure is None else {"closures": [closure]}
    model, state, drivers, _ = _model(**kw)
    d = dict(drivers, wind=_winds(b)) if b else drivers
    alone = model.step(state, d, 600.0)
    coupled, c_state, c_drivers = union({"m": (model, state, d)}, [])
    out = coupled.step(c_state, c_drivers, 600.0)["m"]
    for key in ("gas.x", "air.q"):
        torch.testing.assert_close(out[key], alone[key], rtol=1e-10, atol=1e-16)


def test_unlinked_union_of_two_models_equals_each_standalone():
    """Two unrelated models in one union, one carrying a closure and a batch: each member's
    result is its standalone result, and neither disturbs the other."""
    m1, s1, d1, _ = _model(closures=[_conditional_closure])
    m2, s2, d2, _ = _model(n_species=2)
    d1 = dict(d1, wind=_winds(3))
    alone1, alone2 = m1.step(s1, d1, 600.0), m2.step(s2, d2, 600.0)
    coupled, cs, cd = union({"one": (m1, s1, d1), "two": (m2, s2, d2)}, [])
    out = coupled.step(cs, cd, 600.0)
    torch.testing.assert_close(out["one"]["gas.x"], alone1["gas.x"], rtol=1e-10, atol=1e-16)
    torch.testing.assert_close(out["two"]["gas.x"], alone2["gas.x"], rtol=1e-10, atol=1e-16)


def test_union_of_one_electrical_model_equals_standalone_over_a_time_dependent_law():
    from noodl.apps.electrical import Ramp
    from noodl.apps.electrical.analysis import initial_state
    from noodl.apps.electrical.circuit import (
        Capacitor,
        Circuit,
        Resistor,
        VoltageSource,
        build_model,
    )

    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0, Ramp(1.0)),
                       Resistor("R1", "a", "b", 1.0), Capacitor("C1", "b", "0", 0.1)])
    model, _, drivers = build_model(circuit)
    state = initial_state(model, drivers, t=0.0)
    for t in (0.0, 0.2, 0.7):
        alone = model.step(state, drivers, 0.1, t=t)["electrical.phi"]
        coupled, cs, cd = union({"c": (model, state, drivers)}, [])
        out = coupled.step(cs, cd, 0.1, t=t)["c"]["electrical.phi"]
        torch.testing.assert_close(out, alone, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------- warm start / gradient


def _value_and_grad(model, el, state, drivers, **kw):
    el.C.grad = None
    out = model.step(state, drivers, 600.0, **kw)
    out["gas.x"].sum().backward()
    return out["gas.x"].detach(), el.C.grad.clone()


@pytest.mark.parametrize("closure", [_gain_closure, _conditional_closure])
def test_iterate_value_and_gradient_are_independent_of_the_initial_guess(closure):
    """A coupled (iterate) step reaches the same fixed point and the same derivative from a
    cold start, from a warm start at the converged potentials and from a distant one."""
    model, state, drivers, el = _model(
        closures=[closure], learnable=True, coupling="iterate",
        iterate_tol={"gas": 1e-13}, iterate_max=80,
    )
    ref_x, ref_g = _value_and_grad(model, el, state, drivers, warm_start=False)
    converged = model.step(state, drivers, 600.0)
    guesses = {
        "converged": dict(state, **{"air.phi": converged["air.phi"].detach()}),
        "distant": dict(state, **{"air.phi": torch.tensor([0.0, 40.0, -30.0], dtype=F64)}),
    }
    for name, st in guesses.items():
        x, g = _value_and_grad(model, el, st, drivers, warm_start=True)
        torch.testing.assert_close(x, ref_x, rtol=1e-6, atol=1e-12, msg=f"{name} start: value")
        torch.testing.assert_close(g, ref_g, rtol=1e-5, atol=1e-12, msg=f"{name} start: grad")


@pytest.mark.parametrize("closure", [None, _gain_closure])
def test_step_gradient_matches_central_differences(closure):
    kw = {} if closure is None else {"closures": [closure], "coupling": "iterate",
                                     "iterate_tol": {"gas": 1e-13}, "iterate_max": 80}
    model, state, drivers, el = _model(learnable=True, **kw)
    _, grad = _value_and_grad(model, el, state, drivers)
    h = 1e-6
    vals = []
    for sgn in (1, -1):
        with torch.no_grad():
            el.C[0] += sgn * h
            vals.append(model.step(state, drivers, 600.0)["gas.x"].sum().item())
            el.C[0] -= sgn * h
    assert grad[0].item() == pytest.approx((vals[0] - vals[1]) / (2 * h), rel=1e-4, abs=1e-12)
