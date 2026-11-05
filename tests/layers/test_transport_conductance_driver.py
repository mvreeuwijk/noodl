"""Several conduction kinds on one `TransportLayer`, and the per-step conductance driver
`"<layer>.conductance"` that replaces the construction-time conductance for one call."""

from __future__ import annotations

import pytest
import torch
from scipy.optimize import brentq
from torch.autograd import gradcheck

from noodl.layers.transport import TransportLayer, UndefinedSteadyState
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
SCHEMES = ["implicit", "trapezoidal", "exact"]
SIGMA = 5.670374419e-8


def _t(values, **kwargs):
    return torch.tensor(values, dtype=F64, **kwargs)


def _net(kinds=("wall", "film", "rad")):
    """ambient <- a by a flow edge; conduction edges in this order:
    a-b `kinds[0]`, b-ambient `kinds[1]`, a-ambient `kinds[2]`."""
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("a", "ambient", kind="flow")
    net.add_edge("a", "b", kind=kinds[0])
    net.add_edge("b", "ambient", kind=kinds[1])
    net.add_edge("a", "ambient", kind=kinds[2])
    return net


G = [0.7, 1.3, 0.4]


def _layer(scheme="implicit", kinds=("wall", "film", "rad"), g=None, **kwargs):
    net = _net(kinds if len(set(kinds)) == 3 else ("c", "c", "c"))
    return TransportLayer(
        net, "heat", capacity=_t([2.0, 3.0]), flow_kind="flow", boundary=["ambient"],
        conduction_kind=kinds if len(set(kinds)) > 1 else kinds[0],
        conductance=_t(G if g is None else g), scheme=scheme, **kwargs,
    )


X0 = _t([1.0, 0.5])
Q = _t([0.5])
SRC = _t([0.0, 0.3, 0.1])
XB = _t([0.2])


def _run(layer, scheme, **kw):
    if scheme == "steady":
        return layer.steady(Q, SRC, XB, **kw)
    return layer.step(X0, Q, SRC, XB, 0.8, **kw)


@pytest.mark.parametrize("scheme", [*SCHEMES, "steady"])
def test_a_tuple_of_kinds_is_the_merged_kind(scheme):
    split = _layer("implicit" if scheme == "steady" else scheme)
    merged = _layer("implicit" if scheme == "steady" else scheme, kinds=("c", "c", "c"))
    assert split.conduction_kinds == ("wall", "film", "rad") and merged.conduction_kind == "c"
    assert torch.equal(_run(split, scheme), _run(merged, scheme))
    assert torch.equal(split.interior_idx, merged.interior_idx)


def test_conduction_edges_are_the_kinds_in_order():
    layer = _layer()
    blocks = layer.conduction_edges
    assert [b[0] for b in blocks] == ["wall", "film", "rad"]
    nodes = layer.net.nodes
    ends = [[(nodes[int(s)], nodes[int(t)]) for s, t in zip(src, tgt, strict=True)]
            for _, src, tgt in blocks]
    assert ends == [[("a", "b")], [("b", "ambient")], [("a", "ambient")]]
    # reversed kinds: the conductance is the kinds concatenated in the order GIVEN
    rev = TransportLayer(
        _net(), "heat", capacity=_t([2.0, 3.0]), flow_kind="flow", boundary=["ambient"],
        conduction_kind=("rad", "wall"), conductance=_t([0.4, 0.7]),
    )
    assert [b[0] for b in rev.conduction_edges] == ["rad", "wall"]
    assert _layer().net.edge_kinds().count("flow") == 1
    no_conduction = TransportLayer(_net(), "h", capacity=_t([2.0]), flow_kind="flow",
                                   boundary=["ambient"])
    assert no_conduction.conduction_edges == ()


def test_every_conduction_kind_is_a_touch_for_active_interior():
    net = Network(dtype=F64)
    for n in ("ambient", "a", "film_only", "rad_only", "alone"):
        net.add_node(n)
    net.add_edge("a", "ambient", kind="flow")
    net.add_edge("a", "film_only", kind="film")
    net.add_edge("a", "rad_only", kind="rad")
    net.add_edge("alone", "ambient", kind="other")
    layer = TransportLayer(
        net, "h", capacity=_t([1.0, 1.0, 1.0]), flow_kind="flow", boundary=["ambient"],
        conduction_kind=("film", "rad"), conductance=_t([1.0, 1.0]),
    )
    names = [net.nodes[int(i)] for i in layer.interior_idx]
    assert names == ["a", "film_only", "rad_only"]
    assert [net.nodes[int(i)] for i in layer.inactive_idx] == ["alone"]


@pytest.mark.parametrize("scheme", [*SCHEMES, "steady"])
def test_the_driver_equal_to_the_static_conductance_is_bitwise_the_absent_driver(scheme):
    layer = _layer("implicit" if scheme == "steady" else scheme)
    assert torch.equal(_run(layer, scheme), _run(layer, scheme, conductance=_t(G)))


@pytest.mark.parametrize("scheme", [*SCHEMES, "steady"])
def test_the_driver_replaces_the_static_conductance(scheme):
    base = "implicit" if scheme == "steady" else scheme
    other = [1.1, 0.2, 2.0]
    driven = _run(_layer(base), scheme, conductance=_t(other))
    rebuilt = _run(_layer(base, g=other), scheme)
    assert torch.equal(driven, rebuilt)
    assert not torch.allclose(driven, _run(_layer(base), scheme))


@pytest.mark.parametrize("scheme", ["implicit", "exact"])
def test_a_batched_driver_matches_one_call_per_instance(scheme):
    layer = _layer(scheme)
    gs = _t([G, [1.1, 0.2, 2.0], [0.0, 0.5, 0.5]])
    out = layer.step(X0, Q, SRC, XB, 0.8, conductance=gs)
    assert out.shape == (3, 2)
    for i in range(3):
        one = layer.step(X0, Q, SRC, XB, 0.8, conductance=gs[i])
        torch.testing.assert_close(out[i], one, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gradient_flows_through_the_driver(scheme):
    layer = _layer(scheme)

    def f(g):
        return layer.step(X0, Q, SRC, XB, 0.8, conductance=g)

    assert gradcheck(f, (_t([0.9, 1.1, 0.6], requires_grad=True),),
                     eps=1e-6, atol=1e-7, rtol=1e-6)


def test_gradient_flows_through_the_driver_steady():
    layer = _layer("implicit")

    def f(g):
        return layer.steady(Q, SRC, XB, conductance=g)

    assert gradcheck(f, (_t([0.9, 1.1, 0.6], requires_grad=True),),
                     eps=1e-6, atol=1e-7, rtol=1e-6)


def test_the_driver_is_refused_by_name_when_malformed():
    layer = _layer()
    with pytest.raises(ValueError, match=r"conductance must have trailing shape \(3,\)"):
        layer.step(X0, Q, SRC, XB, 0.8, conductance=_t([1.0, 1.0]))
    with pytest.raises(ValueError, match="conductance must have trailing shape"):
        layer.steady(Q, SRC, XB, conductance=_t(1.0))
    negative = r"zero or positive; negative on conduction edge.*b.*ambient"
    with pytest.raises(ValueError, match=negative):
        layer.step(X0, Q, SRC, XB, 0.8, conductance=_t([1.0, -0.1, 1.0]))
    with pytest.raises(ValueError, match="zero or positive"):
        layer.step(X0, Q, SRC, XB, 0.8, conductance=_t([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]))
    plain = TransportLayer(_net(), "h", capacity=_t([2.0]), flow_kind="flow",
                           boundary=["ambient"])
    with pytest.raises(ValueError, match="no conduction_kind"):
        plain.step(X0[:1], Q, SRC, XB, 0.8, conductance=_t([1.0]))


def test_a_zero_driver_isolates_a_node_for_that_call():
    """Both of `a`'s conduction edges vanish and `a` has no flow through it of its own
    conduction: a transient step still solves (the node only gains its source), a steady
    solve names the node as undefined."""
    layer = _layer("implicit")
    g = _t([0.0, 1.0, 0.0])               # a-b, a-ambient zero; only b-ambient conducts
    out = layer.step(X0, Q, SRC, XB, 0.8, conductance=g)
    assert torch.isfinite(out).all()
    with pytest.raises(UndefinedSteadyState, match=r"\['a'\]|'a'"):
        # the flow edge a->ambient is carried by q = 0 here, so `a` has no way out
        layer.steady(_t([0.0]), SRC, XB, conductance=g)
    # the same zeros, but with a flow out of `a`, is a well-posed steady state
    layer.steady(Q, SRC, XB, conductance=g)


def test_a_zero_driver_leaves_a_storage_free_node_idle():
    """A storage-free node whose only conduction edge is zero this call holds its value in
    a transient step and is refused (named) in a steady solve."""
    net = Network(dtype=F64)
    for n in ("ambient", "s", "plenum"):
        net.add_node(n)
    net.add_edge("s", "ambient", kind="flow")
    net.add_edge("s", "ambient", kind="wall")
    net.add_edge("s", "plenum", kind="film")
    layer = TransportLayer(
        net, "heat", capacity=_t([2.0, 0.0]), flow_kind="flow", boundary=["ambient"],
        conduction_kind=("wall", "film"), conductance=_t([1.0, 2.0]), scheme="implicit",
        storage_free=["plenum"],
    )
    x0, q, xb, src = _t([1.0, 0.7]), _t([0.0]), _t([0.0]), _t([0.0, 0.0, 1.0])
    diag: dict = {}
    out = layer.step(x0, q, src, xb, 1.0, conductance=_t([1.0, 0.0]), diagnostics=diag)
    assert out[1].item() == 0.7 and diag["held_sources_dropped"] == 1
    live = layer.step(x0, q, src, xb, 1.0, conductance=_t([1.0, 2.0]))
    assert live[1].item() != 0.7
    with pytest.raises(ValueError, match="nothing passes through storage-free"):
        layer.steady(q, src, xb, conductance=_t([1.0, 0.0]))


# ---------------------------------------------------------------- the T^4 iteration
T_AMB, T_ENV, UA, EPS_A, POWER, CAP = 300.0, 250.0, 2.0, 0.9 * 1.5, 120.0, 1500.0


class _RadiativeCoefficient:
    """Closure: `h_r = sigma eps A (T_i^2 + T_j^2)(T_i + T_j)` from the state's own surface
    temperature `T_i` and the radiating boundary `T_j`, written as the driver of the layer's
    conduction edges (wall UA first, then the radiative edge)."""

    def __call__(self, state, drivers):
        t_i = state["thermal.x"].reshape(-1)[0]
        t_j = drivers["thermal.x_boundary"][1]
        h_r = SIGMA * EPS_A * (t_i**2 + t_j**2) * (t_i + t_j)
        return {"thermal.conductance": torch.stack([torch.full_like(h_r, UA), h_r])}


def _radiative_model(scheme):
    net = Network(dtype=F64)
    for n in ("amb", "env", "s"):
        net.add_node(n)
    net.add_edge("s", "amb", kind="flow")
    net.add_edge("s", "amb", kind="wall")
    net.add_edge("s", "env", kind="rad")
    layer = TransportLayer(
        net, "thermal", capacity=_t([CAP]), flow_kind="flow", boundary=["amb", "env"],
        conduction_kind=("wall", "rad"), conductance=_t([UA, 1.0]), scheme=scheme,
    )
    model = Model(net, {"thermal": layer}, closures=[_RadiativeCoefficient()],
                  coupling="iterate", iterate_tol={"thermal": 1e-12}, iterate_max=200)
    drivers = {"thermal.x_boundary": _t([T_AMB, T_ENV]), "thermal.q": _t([0.0]),
               "thermal.sources": _t([0.0, 0.0, POWER])}
    return model, drivers


def _fourth_power_root():
    def balance(t):
        return POWER + UA * (T_AMB - t) + SIGMA * EPS_A * (T_ENV**4 - t**4)

    return brentq(balance, 200.0, 600.0, xtol=1e-13, rtol=1e-15)


def test_iterating_the_radiative_coefficient_converges_to_the_fourth_power_steady_state():
    root = _fourth_power_root()
    model, drivers = _radiative_model("implicit")
    diag: dict = {}
    out = model.steady({"thermal.x": _t([T_AMB])}, drivers, diagnostics=diag)
    assert diag["passes"] >= 3                       # the coefficient really was iterated
    assert out["thermal.x"].item() == pytest.approx(root, abs=1e-7)
    # a linearised-once balance (h_r frozen at the start temperature) is NOT the root
    h0 = SIGMA * EPS_A * (T_AMB**2 + T_ENV**2) * (T_AMB + T_ENV)
    frozen = (POWER + UA * T_AMB + h0 * T_ENV) / (UA + h0)
    assert abs(frozen - root) > 1e-3


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal", "exact"])
def test_stepping_the_radiative_coefficient_settles_on_the_same_root(scheme):
    root = _fourth_power_root()
    model, drivers = _radiative_model(scheme)
    state = {"thermal.x": _t([T_AMB])}
    for _ in range(60):
        state = model.step(state, drivers, 100.0)
    assert state["thermal.x"].item() == pytest.approx(root, abs=1e-6)


def test_a_negative_construction_time_conductance_is_refused_by_name():
    message = r"zero or positive; negative on conduction edge.*'b', 'ambient'"
    with pytest.raises(ValueError, match=message):
        _layer(g=[0.7, -1.0, 0.4])
