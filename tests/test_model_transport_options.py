"""A transport layer's per-step options through `Model` (`layer_options={"T": {"scheme":
...}}`), the trapezoidal step's sources at the step start from a closure that carries them,
and a conduction-only transport layer (`flow_kind=()`) in a model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parent / "data" / "modelica_electrical"
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


# ---------------------------------------------------------------- Anderson mixing of a heat state
def test_anderson_mixing_lands_the_circuit_on_the_heat_layers_temperature():
    """MSL's Resistor (220 V sine, R(T), heat port folded into G = 50 W/K to 293.15 K) at
    t = 0.49 s: the iteration must stop with the circuit solved at the temperature the heat
    layer reports. With the heat state relaxed instead of mixed, two passes of the heat layer
    agreed to 1.8e-11 K while the circuit was solved at a temperature 4.8e-7 K away, and the
    current came out 6.7e-11 A (4.8e-10 relative) off the one at the reported temperature."""
    from noodl.apps.electrical.modelica import read_modelica_circuit
    from noodl.apps.electrical.modelica.run import STEADY_ATOL, STEADY_RTOL
    from noodl.modelica.timeline import step_drivers

    model, state, drivers, names = read_modelica_circuit(
        DATA / "Analog.Examples.Resistor.json")
    t = 0.49
    diag: dict = {}
    out = model.steady(state, step_drivers(drivers, names.grid, t), t=t, atol=STEADY_ATOL,
                       rtol=STEADY_RTOL, diagnostics=diag)
    assert bool(diag["converged"])
    T = float(out["thermal.x"][0])
    v = 220.0 * math.sin(2.0 * math.pi * t)
    i = v / (100.0 * (1.0 + 1e-3 * (T - 293.15)))
    assert abs(float(out["electrical.q"][1]) - i) <= 1e-13
    # and both on OpenModelica's fixed point (its CSV row at 0.49 s)
    assert abs(T - 293.1881633891936) <= 1e-11
