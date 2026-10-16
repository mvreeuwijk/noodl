"""The electrical Modelica reader's runner: steady sequences and time stepping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from noodl.apps.electrical.modelica import extrapolate, read_modelica_circuit, simulate_circuit
from noodl.apps.electrical.modelica.run import STEADY_ATOL, STEADY_RTOL, THERMAL_STEADY_ATOL

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica_electrical"
MSL = "Modelica.Electrical.Analog."


def _rc_fixture(tmp_path):
    """A 1 V step at t = 0.25 s through 1 ohm into 1 F, on an output grid of 0.1 s."""
    components = [
        {"name": "V", "class": MSL + "Sources.StepVoltage",
         "parameters": {"V": 1.0, "offset": 0.0, "startTime": 0.25}},
        {"name": "R1", "class": MSL + "Basic.Resistor",
         "parameters": {"R": 1.0, "T_ref": 300.15, "alpha": 0.0, "useHeatPort": False,
                        "T": 300.15}},
        {"name": "C1", "class": MSL + "Basic.Capacitor", "parameters": {"C": 1.0},
         "initial": {"v": 0.0}},
        {"name": "G", "class": MSL + "Basic.Ground", "parameters": {}},
    ]
    doc = {
        "format": "noodl-modelica-electrical/1", "model": "RC", "mbl_commit": "x",
        "openmodelica": "omc", "msl": "4.1.0",
        "experiment": {"StartTime": 0.0, "StopTime": 1.0, "Interval": 0.1, "Tolerance": 1e-12},
        "components": components,
        "connections": [["V.p", "R1.p"], ["R1.n", "C1.p"], ["C1.n", "G.p"], ["V.n", "G.p"]],
        "signals": [], "compared": ["V.i", "R1.i", "C1.i", "C1.v"],
    }
    path = tmp_path / "rc.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _recurrence(r):
    """Backward Euler on the first step and the step after the event, trapezoidal otherwise,
    on the grid {0, 0.1, ..., 1} U {0.25} split r ways; source 0 up to and including 0.25."""
    base = sorted({round(0.1 * i, 12) for i in range(11)} | {0.25})
    grid = [base[0]] + [a + (b - a) * k / r for a, b in zip(base, base[1:], strict=False)
                        for k in range(1, r + 1)]
    v = i = 0.0
    out = {}
    for k in range(len(grid) - 1):
        t0, t1 = grid[k], grid[k + 1]
        h, vs = t1 - t0, (0.0 if t1 <= 0.25 + 1e-12 else 1.0)
        if k == 0 or abs(t0 - 0.25) < 1e-12:
            v1 = (v + h * vs) / (1 + h)
        else:
            v1 = (vs + i + 2 / h * v) / (1 + 2 / h)
        i, v = vs - v1, v1
        out[round(t1, 9)] = v
    return out


@pytest.mark.parametrize("r", [1, 4])
def test_rc_step_between_output_times_matches_the_recurrence(tmp_path, r):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path), substeps=r)
    assert 0.25 in [round(x, 12) for x in names.grid.tolist()]
    out = simulate_circuit(model, state, drivers, names)
    v = names.value(out, "C1.v")
    ref = _recurrence(r)
    for t, x in zip(out["time"].tolist(), v.tolist(), strict=True):
        if t > 0:
            assert abs(x - ref[round(t, 9)]) <= 1e-12


def test_richardson_of_8_and_16_substeps_is_third_order_accurate(tmp_path):
    runs = []
    for r in (8, 16):
        model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path), substeps=r)
        runs.append(simulate_circuit(model, state, drivers, names))
    best = extrapolate(*runs, order=2)
    t = best["time"]
    exact = torch.where(t > 0.25, 1 - torch.exp(-(t - 0.25)), torch.zeros_like(t))
    err = (names.value(best, "C1.v") - exact).abs()[1:]
    # measured with the recurrence above: 7.8e-9 at t = 0.5, 4.8e-9 at t = 1.0
    assert float(err.max()) <= 1e-8


def test_algebraic_model_is_solved_at_each_requested_time():
    model, state, drivers, names = read_modelica_circuit(
        DATA / "Analog.Examples.ShowVariableResistor.json")
    out = simulate_circuit(model, state, drivers, names, times=names.times[:5])
    assert out["electrical.q"].shape[0] == 5


def test_off_grid_time_is_refused():
    model, state, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.Resistor.json")
    with pytest.raises(ValueError, match="not a grid time"):
        simulate_circuit(model, state, drivers, names, times=[0.0123])


# ------------------------------------------------------------------ further behaviour
def test_history_holds_time_and_every_layer_state_on_the_output_grid(tmp_path):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path), substeps=2)
    out = simulate_circuit(model, state, drivers, names)
    assert set(out) == {"time", "electrical.phi", "electrical.q"}
    torch.testing.assert_close(out["time"], names.times, rtol=0, atol=0)
    assert out["electrical.phi"].shape[0] == names.times.numel() == 11
    # Row 0 is the start state itself.
    torch.testing.assert_close(out["electrical.phi"][0], state["electrical.phi"], rtol=0, atol=0)


def test_a_dynamic_run_may_start_after_the_first_grid_time(tmp_path):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path))
    full = simulate_circuit(model, state, drivers, names)
    part = simulate_circuit(model, state, drivers, names, times=names.times[3:6])
    torch.testing.assert_close(part["electrical.q"], full["electrical.q"][3:6], rtol=0, atol=0)


def test_non_increasing_times_are_refused(tmp_path):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path))
    with pytest.raises(ValueError, match="increasing"):
        simulate_circuit(model, state, drivers, names, times=[0.2, 0.1])
    with pytest.raises(ValueError, match="no times"):
        simulate_circuit(model, state, drivers, names, times=[])


def test_steady_solves_default_to_round_off_and_accept_solve_keywords():
    # 12 V; res1 (2 ohm) across the source; res2 (2 ohm) behind a 2 ohm line: 6 A and 3 A.
    model, state, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.Resistor.json")
    out = simulate_circuit(model, state, drivers, names, times=names.times[:2])
    assert names.value(out, "res1.terminal.i[1]").tolist() == pytest.approx([6.0, 6.0],
                                                                              rel=1e-15)
    # Solve keywords reach the layer's solve.
    with pytest.raises(ValueError, match="on_failure='return'"):
        simulate_circuit(model, state, drivers, names, times=names.times[:1],
                         on_failure="return")


class _Recording:
    """A model wrapper that records the solve keywords of every call."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def steady(self, state, drivers, **kwargs):
        self.calls.append(("steady", kwargs))
        return self.model.steady(state, drivers, **kwargs)

    def step(self, state, drivers, **kwargs):
        self.calls.append(("step", kwargs))
        return self.model.step(state, drivers, **kwargs)


def test_steady_default_tolerances_and_step_keywords_reach_the_model(tmp_path):
    model, state, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.Resistor.json")
    rec = _Recording(model)
    simulate_circuit(rec, state, drivers, names, times=names.times[:1])
    simulate_circuit(rec, state, drivers, names, times=names.times[:1], rtol=1e-10)
    assert [(c[1]["atol"], c[1]["rtol"]) for c in rec.calls] == [
        (STEADY_ATOL, STEADY_RTOL), (STEADY_ATOL, 1e-10)]
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path))
    rec = _Recording(model)
    simulate_circuit(rec, state, drivers, names, times=names.times[:4], max_iter=7)
    # 0 -> 0.1 -> 0.2 -> 0.25 (the step event) -> 0.3: backward Euler again after 0.25.
    assert [c[1]["integration"] for c in rec.calls] == [
        "backward_euler", "trapezoidal", "trapezoidal", "backward_euler"]
    assert all(c[1]["max_iter"] == 7 and "atol" not in c[1] for c in rec.calls)


def test_a_heat_port_circuit_is_solved_at_the_thermal_round_off_tolerance():
    # MSL Resistor: 220 V sine heating R(T) through G = 50 W/K to 293.15 K. The electrical
    # absolute tolerance would stall the heat balance (round-off ~ eps G T, W), so the
    # runner gives the thermal layer THERMAL_STEADY_ATOL (and the electrical one keeps
    # STEADY_ATOL); the coupled state is the hand fixed point at the 220 V peak (t = 0.25 s).
    model, state, drivers, names = read_modelica_circuit(DATA / "Analog.Examples.Resistor.json")
    calls = []
    steady = model.steady

    def recording(*args, **kwargs):
        calls.append((kwargs["atol"], kwargs["rtol"], kwargs["layer_options"]))
        return steady(*args, **kwargs)

    model.steady = recording
    out = simulate_circuit(model, state, drivers, names, times=[0.0, 0.25])
    assert calls == [(STEADY_ATOL, STEADY_RTOL, {"thermal": {"atol": THERMAL_STEADY_ATOL}})] * 2
    T = names.value(out, "resistor.T_heatPort")
    assert float(T[0]) == pytest.approx(293.15, abs=1e-9)
    assert abs(float(T[1]) - 302.73806893411466) <= 1e-9


def test_a_single_time_may_be_a_zero_dimensional_tensor(tmp_path):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path))
    out = simulate_circuit(model, state, drivers, names, 
                           times=torch.tensor(0.3, dtype=torch.float64))
    assert out["time"].tolist() == pytest.approx([0.3]) and out["electrical.q"].shape[0] == 1


def test_returned_time_is_the_grid_time_and_never_an_input_tensor(tmp_path):
    model, state, drivers, names = read_modelica_circuit(_rc_fixture(tmp_path))
    out = simulate_circuit(model, state, drivers, names)
    assert out["time"] is not names.times
    out["time"][0] = -1.0
    assert float(names.times[0]) == 0.0 and float(names.grid[0]) == 0.0
    asked = torch.tensor([0.1 + 1e-12, 0.2], dtype=torch.float64)
    part = simulate_circuit(model, state, drivers, names, times=asked)
    assert part["time"] is not asked and float(part["time"][0]) == float(names.times[1])


def test_extrapolate_is_the_shared_richardson_step(tmp_path):
    from noodl.apps.building_physics.modelica import extrapolate as building_extrapolate
    from noodl.modelica.timeline import extrapolate as shared

    assert extrapolate is shared is building_extrapolate
    coarse = {"time": torch.tensor([0.0, 1.0]), "x": torch.tensor([1.0, 4.0])}
    fine = {"time": torch.tensor([0.0, 1.0]), "x": torch.tensor([1.0, 7.0])}
    assert extrapolate(coarse, fine)["x"].tolist() == [1.0, 8.0]
    with pytest.raises(ValueError, match="same times"):
        extrapolate(coarse, {**fine, "time": torch.tensor([0.0, 2.0])})
