"""A transport layer's per-step options through `Model` (`layer_options={"T": {"scheme":
...}}`), the trapezoidal step's sources at the step start from a closure that carries them,
and a conduction-only transport layer (`flow_kind=()`) in a model."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
C, G, TA, DT = 4.0, 3.0, 280.0, 0.7


def _t(values):
    return torch.tensor(values, dtype=F64)


def _layer(scheme="trapezoidal"):
    net = Network(dtype=F64)
    for n in ("ambient", "m"):
        net.add_node(n)
    net.add_edge("m", "ambient", kind="wall")
    return TransportLayer(net, "T", flow_kind=(), boundary=["ambient"], capacity=_t([C]),
                          conduction_kind="wall", conductance=_t([G]), scheme=scheme)


class _RampHeat:
    """Writes `"T.sources"` = `rate * t_end` at node `m` and carries it as state."""

    integrates = True
    state_keys = ("T.sources",)

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, state, drivers, ctx):
        t_end = ctx.t + (ctx.dt or 0.0)
        return {"T.sources": _t([0.0, self.rate * t_end])}


def test_a_conduction_only_layer_needs_no_flow_driver():
    layer = _layer()
    model = Model(layer.net, [layer])
    assert model.flow_layer_of["T"] is None
    assert "flows" not in model.refs["T"].inputs
    drivers = {"T.x_boundary": _t([TA]), "T.sources": _t([0.0, 6.0])}
    out = model.steady({"T.x": _t([300.0])}, drivers)
    assert float(out["T.x"][0]) == pytest.approx(TA + 6.0 / G, rel=1e-14)
    out = model.step({"T.x": _t([300.0])}, drivers, DT)
    assert torch.isfinite(out["T.x"]).all()
    assert not model.check({"T.x": _t([300.0])}, drivers).errors


def test_layer_options_give_a_transport_layer_the_scheme_of_the_step(monkeypatch):
    layer = _layer("exact")
    seen = []

    def spy(*args, _step=layer.step, **kwargs):
        seen.append(kwargs.get("scheme"))
        return _step(*args, **kwargs)

    monkeypatch.setattr(layer, "step", spy)
    model = Model(layer.net, [layer])
    state, drivers = {"T.x": _t([300.0])}, {"T.x_boundary": _t([TA])}
    model.step(state, drivers, DT)
    model.step(state, drivers, DT, layer_options={"T": {"scheme": "implicit"}})
    assert seen == [None, "implicit"]
    model.steady(state, drivers, layer_options={"T": {"scheme": "implicit"}})   # ignored
    out = model.step(state, drivers, DT, layer_options={"T": {"scheme": "implicit"}})
    assert float(out["T.x"][0]) == pytest.approx(
        (C * 300.0 + DT * G * TA) / (C + DT * G), rel=1e-14)


def test_layer_options_refuse_a_transport_keyword_other_than_the_scheme():
    layer = _layer()
    model = Model(layer.net, [layer])
    state, drivers = {"T.x": _t([300.0])}, {"T.x_boundary": _t([TA])}
    with pytest.raises(ValueError, match=r"transport layer 'T' \['atol'\].*only \['scheme'\]"):
        model.step(state, drivers, DT, layer_options={"T": {"atol": 1e-9}})
    with pytest.raises(ValueError, match=r"names \['X'\], not Kirchhoff or transport"):
        model.step(state, drivers, DT, layer_options={"X": {"scheme": "implicit"}})


@pytest.mark.parametrize("scheme", ["trapezoidal", "implicit"])
def test_a_closure_carrying_the_sources_gives_the_trapezoidal_step_both_ends(scheme):
    # sources r t at node m: trapezoidal integrates dt/2 (r t0 + r t1); backward Euler r t1
    rate, t0 = 5.0, 2.0
    layer = _layer()
    model = Model(layer.net, [layer], closures=[_RampHeat(rate)])
    state = {"T.x": _t([300.0]), "T.sources": _t([0.0, rate * t0])}
    out = model.step(state, {"T.x_boundary": _t([TA])}, DT, t=t0,
                     layer_options={"T": {"scheme": scheme}})
    s0, s1 = rate * t0, rate * (t0 + DT)
    if scheme == "trapezoidal":
        expected = (((C - DT * G / 2) * 300.0 + DT * G * TA + DT / 2 * (s0 + s1))
                    / (C + DT * G / 2))
    else:
        expected = (C * 300.0 + DT * (G * TA + s1)) / (C + DT * G)
    assert float(out["T.x"][0]) == pytest.approx(expected, rel=1e-14)
    assert out["T.sources"].tolist() == [0.0, s1]


def test_substeps_interpolate_the_sources_between_both_ends():
    rate, t0 = 5.0, 2.0
    layer = _layer()
    model = Model(layer.net, [layer], closures=[_RampHeat(rate)], substeps={"T": 2})
    state = {"T.x": _t([300.0]), "T.sources": _t([0.0, rate * t0])}
    out = model.step(state, {"T.x_boundary": _t([TA])}, DT, t=t0)
    h, x = DT / 2, 300.0
    for j in range(2):
        a, b = rate * (t0 + j * h), rate * (t0 + (j + 1) * h)
        x = ((C - h * G / 2) * x + h * G * TA + h / 2 * (a + b)) / (C + h * G / 2)
    assert float(out["T.x"][0]) == pytest.approx(x, rel=1e-14)


# ---------------------------------------------------------------- iterate_mixed
def _coupled_resistor(**coupling):
    """A 220 V resistor R(T) = 100 (1 + 1e-3 (T - 293.15)) whose heat port (storage-free)
    conducts 50 W/K to 293.15 K, coupled by `JouleHeating`."""
    from noodl.apps.electrical import HeatedEdge, JouleHeating, PrescribedVoltage, Resistance
    from noodl.layers.kirchhoff import KirchhoffLayer

    net = Network(dtype=F64)
    for n in ("0", "s", "h", "amb"):
        net.add_node(n)
    net.add_edge("s", "0", kind="voltage_source")
    net.add_edge("s", "0", kind="resistor")
    net.add_edge("h", "amb", kind="conductor")
    electrical = KirchhoffLayer(
        net, "electrical",
        [PrescribedVoltage(_t(220.0)), Resistance(_t(100.0), alpha=_t(1e-3), T_ref=_t(293.15),
                                                  temperature="electrical.resistor.T")],
        boundary=["0"])
    thermal = TransportLayer(net, "thermal", flow_kind=(), boundary=["amb"],
                             capacity=_t([0.0]), conduction_kind="conductor",
                             conductance=_t([50.0]), storage_free=["h"])
    joule = JouleHeating([HeatedEdge(edge=1, p=1, n=0, heat_node=2,
                                     temperature_key="electrical.resistor.T", slot=0)],
                         thermal, base_temperatures={"electrical.resistor.T": _t([293.15])})
    model = Model(net, [electrical, thermal], closures=[joule], **coupling)
    state = {"electrical.phi": _t([0.0, 220.0, 0.0, 0.0]), "electrical.q": _t([2.2, 2.2]),
             "thermal.x": _t([293.15]), "thermal.sources": torch.zeros(4, dtype=F64)}
    drivers = {"electrical.phi_boundary": _t([0.0]), "thermal.x_boundary": _t([293.15]),
               "electrical.resistor.T": _t([293.15])}
    return model, state, drivers


ANDERSON = dict(coupling="iterate", iterate_tol={"electrical": 1e-10, "thermal": 1e-10},
                iterate_flow_tol={"electrical": 1e-10}, iterate_acceleration="anderson",
                iterate_max=100)


def test_a_mixed_heat_state_lands_on_the_coupled_fixed_point():
    # the device's temperature driver and the heat layer's temperature agree at the end
    model, state, drivers = _coupled_resistor(**ANDERSON, iterate_mixed=["thermal"])
    assert model.iterate_mixed == ("thermal",)
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag)
    T = float(out["thermal.x"][0])
    assert abs(T - 302.73806893411466) <= 1e-9
    i = 220.0 / (100.0 * (1.0 + 1e-3 * (T - 293.15)))
    assert abs(float(out["electrical.q"][1]) - i) <= 1e-11


def test_iterate_mixed_is_refused_without_anderson_or_for_an_unjudged_layer():
    with pytest.raises(ValueError, match="needs iterate_acceleration='anderson'"):
        _coupled_resistor(coupling="iterate", iterate_tol={"electrical": 1e-10,
                                                           "thermal": 1e-10},
                          iterate_mixed=["thermal"])
    with pytest.raises(KeyError, match=r"iterate_mixed names \['electrical'\]"):
        _coupled_resistor(**ANDERSON, iterate_mixed=["electrical"])
    with pytest.raises(KeyError, match=r"iterate_mixed names \['thermal'\]"):
        _coupled_resistor(**{**ANDERSON, "iterate_tol": {"electrical": 1e-10}},
                          iterate_mixed=["thermal"])
