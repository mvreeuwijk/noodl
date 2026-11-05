"""A transport step's per-call scheme, the trapezoidal step's sources at the step start
(`sources_prev`), and a conduction-only layer (`flow_kind=()`).

The references are closed forms of a single lumped node, `C dT/dt = -G (T - Ta) + s(t)`,
stepped by hand with backward Euler and the trapezoidal rule."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer, UndefinedSteadyState
from noodl.topology import Network

F64 = torch.float64
SCHEMES = ["exact", "implicit", "trapezoidal"]
C, G, TA, DT = 4.0, 3.0, 280.0, 0.7
EMPTY = torch.zeros(0, dtype=F64)


def _t(values):
    return torch.tensor(values, dtype=F64)


def _lump(scheme="exact", *, folded=False):
    """One stored node `m` joined to the ambient by `G`; with `folded`, through a
    storage-free surface node `s` (m --2G-- s --2G-- ambient, in series G)."""
    net = Network(dtype=F64)
    for n in ("ambient", "m", "s"):
        net.add_node(n)
    if folded:
        net.add_edge("m", "s", kind="wall")
        net.add_edge("s", "ambient", kind="wall")
        return TransportLayer(net, "T", flow_kind=(), boundary=["ambient"],
                              capacity=_t([C, 0.0]), conduction_kind="wall",
                              conductance=_t([2 * G, 2 * G]), storage_free=["s"],
                              scheme=scheme)
    net.add_edge("m", "ambient", kind="wall")
    return TransportLayer(net, "T", flow_kind=(), boundary=["ambient"], capacity=_t([C]),
                          conduction_kind="wall", conductance=_t([G]), scheme=scheme)


def _source(layer, node, value):
    s = torch.zeros(layer.net.n, dtype=F64)
    s[layer.net.nodes.index(node)] = value
    return s


# ---------------------------------------------------------------- conduction only
def test_a_conduction_only_layer_has_no_flows():
    layer = _lump()
    assert layer.flow_kinds == () and layer._flow_src.numel() == 0
    assert layer.interior_idx.tolist() == [1]          # "s" has no edge: inactive
    out = layer.step(_t([300.0]), EMPTY, _source(layer, "m", 0.0), _t([TA]), DT)
    expected = TA + (300.0 - TA) * torch.exp(torch.tensor(-G * DT / C, dtype=F64))
    torch.testing.assert_close(out, expected.reshape(1), rtol=1e-12, atol=0)
    M, N = layer.operator(EMPTY)
    torch.testing.assert_close(M, _t([[-G / C]]))
    torch.testing.assert_close(N, _t([[G / C]]))


# ---------------------------------------------------------------- per-call scheme
@pytest.mark.parametrize("own", SCHEMES)
@pytest.mark.parametrize("call", SCHEMES)
def test_a_per_call_scheme_steps_as_a_layer_built_with_it(own, call):
    for folded in (False, True):
        layer, built = _lump(own, folded=folded), _lump(call, folded=folded)
        x = _t([300.0, 0.0]) if folded else _t([300.0])
        src = _source(layer, "s" if folded else "m", 5.0)
        assert torch.equal(layer.step(x, EMPTY, src, _t([TA]), DT, scheme=call),
                           built.step(x, EMPTY, src, _t([TA]), DT))
        a = layer.step_with_transfer(x, EMPTY, src, _t([TA]), DT, scheme=call)
        b = built.step_with_transfer(x, EMPTY, src, _t([TA]), DT)
        assert torch.equal(a.x, b.x) and torch.equal(a.boundary_transfer, b.boundary_transfer)
        assert layer.scheme == own


def test_backward_euler_and_trapezoidal_by_hand():
    layer = _lump("exact")
    x0, s = 300.0, 5.0
    be = (C * x0 + DT * (G * TA + s)) / (C + DT * G)
    cn = ((C - DT * G / 2) * x0 + DT * (G * TA + s)) / (C + DT * G / 2)
    src = _source(layer, "m", s)
    for scheme, expected in (("implicit", be), ("trapezoidal", cn)):
        out = layer.step(_t([x0]), EMPTY, src, _t([TA]), DT, scheme=scheme)
        assert float(out[0]) == pytest.approx(expected, rel=1e-14)


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="unknown scheme 'euler'"):
        _lump().step(_t([300.0]), EMPTY, _source(_lump(), "m", 0.0), _t([TA]), DT,
                     scheme="euler")


def test_a_storage_free_node_with_throughflow_refuses_any_scheme_but_implicit():
    net = Network(dtype=F64)
    for n in ("in", "p", "z"):
        net.add_node(n)
    net.add_edge("in", "p", kind="air")
    net.add_edge("p", "z", kind="air")
    net.add_edge("z", "in", kind="air")
    layer = TransportLayer(net, "T", flow_kind="air", boundary=["in"], capacity=_t([0.0, 1.0]),
                           storage_free=["p"], scheme="implicit")
    q, x, src = _t([1.0, 1.0, 1.0]), _t([0.0, 0.0]), torch.zeros(3, dtype=F64)
    layer.step(x, q, src, _t([1.0]), DT, scheme="implicit")
    for scheme in ("exact", "trapezoidal"):
        with pytest.raises(ValueError, match="keep an algebraic row.*needs scheme='implicit'"):
            layer.step(x, q, src, _t([1.0]), DT, scheme=scheme)


# ---------------------------------------------------------------- sources at the step start
def test_trapezoidal_integrates_the_sources_at_both_ends():
    layer = _lump("trapezoidal")
    x0, s0, s1 = 300.0, 2.0, 8.0
    cn = ((C - DT * G / 2) * x0 + DT * G * TA + DT / 2 * (s0 + s1)) / (C + DT * G / 2)
    out = layer.step(_t([x0]), EMPTY, _source(layer, "m", s1), _t([TA]), DT,
                     sources_prev=_source(layer, "m", s0))
    assert float(out[0]) == pytest.approx(cn, rel=1e-14)
    # the same on a per-call trapezoidal step of an exact layer
    out = _lump("exact").step(_t([x0]), EMPTY, _source(layer, "m", s1), _t([TA]), DT,
                              scheme="trapezoidal", sources_prev=_source(layer, "m", s0))
    assert float(out[0]) == pytest.approx(cn, rel=1e-14)


@pytest.mark.parametrize("scheme", ["exact", "implicit"])
def test_sources_prev_is_refused_by_every_other_scheme(scheme):
    layer = _lump(scheme)
    src = _source(layer, "m", 1.0)
    with pytest.raises(ValueError, match="sources_prev.*trapezoidal' only"):
        layer.step(_t([300.0]), EMPTY, src, _t([TA]), DT, sources_prev=src)
    with pytest.raises(ValueError, match="sources_prev.*trapezoidal' only"):
        layer.step_with_transfer(_t([300.0]), EMPTY, src, _t([TA]), DT, sources_prev=src)


def test_a_folded_node_takes_the_end_sources_and_the_kept_node_both_ends():
    # m --2G-- s --2G-- ambient with the heat on s: the kept node sees G to the ambient and
    # half the heat of s (the folding's share), and s is recovered from the END heat.
    layer = _lump("trapezoidal", folded=True)
    x0, s0, s1 = 300.0, 2.0, 8.0
    cn = ((C - DT * G / 2) * x0 + DT * G * TA + DT / 2 * (s0 + s1) / 2) / (C + DT * G / 2)
    out = layer.step(_t([x0, 0.0]), EMPTY, _source(layer, "s", s1), _t([TA]), DT,
                     sources_prev=_source(layer, "s", s0))
    assert float(out[0]) == pytest.approx(cn, rel=1e-14)
    assert float(out[1]) == pytest.approx((cn + TA) / 2 + s1 / (4 * G), rel=1e-14)


def test_the_transfer_with_sources_at_both_ends_closes_the_energy_balance():
    # stored change + what reached the boundary = the heat put in, dt/2 (s0 + s1)
    for folded in (False, True):
        layer = _lump("trapezoidal", folded=folded)
        node = "s" if folded else "m"
        x0 = _t([300.0, 0.0]) if folded else _t([300.0])
        out = layer.step_with_transfer(x0, EMPTY, _source(layer, node, 8.0), _t([TA]), DT,
                                       sources_prev=_source(layer, node, 2.0))
        stored = C * float(out.x[0] - x0[0])
        assert stored + float(out.boundary_transfer.sum()) == pytest.approx(
            DT / 2 * (2.0 + 8.0), rel=1e-12)


def test_a_stored_node_with_only_a_folded_neighbour_has_no_steady_state():
    # m --G-- s, s storage-free and no boundary: the folded network leaves m with no edge
    # at all; its steady state is refused by name, not a failed reshape.
    net = Network(dtype=F64)
    for n in ("m", "s"):
        net.add_node(n)
    net.add_edge("m", "s", kind="wall")
    layer = TransportLayer(net, "T", flow_kind=(), boundary=[], capacity=_t([C, 0.0]),
                           conduction_kind="wall", conductance=_t([G]), storage_free=["s"])
    src = _t([0.0, 2.0])
    with pytest.raises(UndefinedSteadyState, match=r"nothing leaves stored node\(s\) \['m'\]"):
        layer.steady(EMPTY, src, EMPTY)
    out = layer.step(_t([300.0, 0.0]), EMPTY, src, EMPTY, DT)       # all heat stays in m
    assert float(out[0]) == pytest.approx(300.0 + DT * 2.0 / C, rel=1e-14)
    assert float(out[1]) == pytest.approx(float(out[0]) + 2.0 / G, rel=1e-14)


# ---------------------------------------------------------------- the initial point
def test_steady_holding_the_stored_nodes_solves_only_the_folded_ones():
    layer = _lump(folded=True)
    src = _source(layer, "s", 6.0)
    out = layer.steady(EMPTY, src, _t([TA]), stored=_t([350.0, 0.0]))
    # s between m (2G) and the ambient (2G) with 6 W of its own
    assert out.tolist() == pytest.approx([350.0, (350.0 + TA) / 2 + 6.0 / (4 * G)],
                                         rel=1e-14)
    plain = _lump()
    held = _t([350.0])
    assert plain.steady(EMPTY, _source(plain, "m", 1.0), _t([TA]), stored=held) is held


def test_steady_holding_the_stored_nodes_refuses_a_plenum():
    net = Network(dtype=F64)
    for n in ("in", "p", "z"):
        net.add_node(n)
    net.add_edge("in", "p", kind="air")
    net.add_edge("p", "z", kind="air")
    net.add_edge("z", "in", kind="air")
    layer = TransportLayer(net, "T", flow_kind="air", boundary=["in"], capacity=_t([0.0, 1.0]),
                           storage_free=["p"], scheme="implicit")
    with pytest.raises(NotImplementedError, match="storage-free nodes with throughflow"):
        layer.steady(_t([1.0, 1.0, 1.0]), torch.zeros(3, dtype=F64), _t([1.0]),
                     stored=_t([0.0, 0.0]))
