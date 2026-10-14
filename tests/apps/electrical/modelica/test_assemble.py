"""The electrical Modelica reader's assembly: laws, drivers, step grid, initial state, names."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.modelica.run import step_drivers
from noodl.apps.electrical.modelica import (
    CircuitNames,
    Column,
    ModelicaImportError,
    read_modelica_circuit,
)
from noodl.apps.electrical.modelica.schema import load

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica_electrical"
MSL = "Modelica.Electrical.Analog."
DC = "Buildings.Electrical.DC."
AC = "Buildings.Electrical.AC.OnePhase."


def _hist(state):
    return {k: v.unsqueeze(0) for k, v in state.items()}


# ------------------------------------------------------------------ the brief's tests
def test_dc_resistor_example_hand_values():
    # 12 V; res1 (2 ohm) across the source; res2 (2 ohm) behind a 2 ohm line.
    model, state, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.Resistor.json")
    assert not names.dynamic and not names.phasor
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    val = {c: float(names.value({k: v.unsqueeze(0) for k, v in out.items()}, c)[0])
           for c in ("res1.terminal.i[1]", "res2.terminal.i[1]", "res2.terminal.v[1]")}
    assert val == pytest.approx({"res1.terminal.i[1]": 6.0, "res2.terminal.i[1]": 3.0,
                                 "res2.terminal.v[1]": 6.0}, rel=1e-13)


def test_terminal_current_is_the_current_into_the_device():
    # The line res carries 3 A from the source (at terminal_n) to res2 (at terminal_p).
    model, state, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.Resistor.json")
    h = _hist(model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0))
    assert float(names.value(h, "res.terminal_n.i[1]")[0]) == pytest.approx(3.0, rel=1e-13)
    assert float(names.value(h, "res.terminal_p.i[1]")[0]) == pytest.approx(-3.0, rel=1e-13)
    assert float(names.value(h, "sou.terminal.i[1]")[0]) == pytest.approx(-9.0, rel=1e-13)


def test_test_impedance_hand_values():
    model, state, drivers, names = read_modelica_circuit(
        DATA / "AC.OnePhase.Loads.Examples.TestImpedance.json")
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    hist = {k: v.unsqueeze(0) for k, v in out.items()}
    expect = {"Z1": -120j, "Z2": 120j, "Z3": 120.0, "Z4": 60 - 60j, "Z5": 60 + 60j}
    for z, i in expect.items():
        re = float(names.value(hist, f"{z}.terminal.i[1]")[0])
        im = float(names.value(hist, f"{z}.terminal.i[2]")[0])
        assert abs(complex(re, im) - i) <= 1e-11


def test_cauer_history_reproduces_the_capacitor_voltages():
    model, state, drivers, names = read_modelica_circuit(
        DATA / "Analog.Examples.CauerLowPassAnalog.json")
    assert names.dynamic and 0 in names.restarts
    assert any(abs(t - 1.0) <= 1e-12 for t in names.grid.tolist())   # the step at t = 1 s
    k = min(range(len(names.grid)), key=lambda j: abs(float(names.grid[j]) - 1.0))
    assert k in names.restarts                                         # BE after the jump
    hist = {k: v.unsqueeze(0) for k, v in state.items()}
    for c in ("C1", "C2", "C3", "C4", "C5"):
        assert float(names.value(hist, f"{c}.v")[0]) == pytest.approx(0.0, abs=1e-15)


def test_variable_frequency_phase_is_the_exact_integral():
    # SeriesResonance: f ramps 0 -> 200 Hz over 1 s, so phi(t) = 2 pi 100 t^2.
    _, _, drivers, names = read_modelica_circuit(DATA / "Analog.Examples.SeriesResonance.json")
    series = drivers["series:electrical.voltage_source.value"]
    t = names.grid
    # cumulated interval means: round-off grows with the phase (up to 628 rad), hence 1e-10
    torch.testing.assert_close(series[:, names.sources["sineVoltage"]],
                               torch.sin(2 * math.pi * 100.0 * t ** 2), rtol=0, atol=1e-10)
    torch.testing.assert_close(series[:, names.sources["cosineVoltage"]],
                               torch.cos(2 * math.pi * 100.0 * t ** 2), rtol=0, atol=1e-10)


def test_grid_halves_with_twice_the_substeps():
    *_, n1 = read_modelica_circuit(DATA / "Analog.Examples.CauerLowPassAnalog.json", substeps=1)
    *_, n2 = read_modelica_circuit(DATA / "Analog.Examples.CauerLowPassAnalog.json", substeps=2)
    torch.testing.assert_close(n2.grid[::2], n1.grid, rtol=0, atol=0)


def test_every_compared_column_maps():
    for path in sorted(p for p in DATA.glob("*.json") if p.name != "parity-electrical.json"):
        omit = ("RL_3", "load_sc_3") if "ACLine_RL" in path.name else ()
        *_, names = read_modelica_circuit(path, omit=omit)
        doc = load(path)
        skipped = {c for c in doc.compared if c.split(".")[0] in omit}
        assert set(names.columns) == set(doc.compared) - skipped, path.name


# ------------------------------------------------------------------ further behaviour
def _doc(tmp_path, components, connections, signals=(), compared=(), experiment=None):
    doc = {
        "format": "noodl-modelica-electrical/1", "model": "M", "mbl_commit": "x",
        "openmodelica": "omc", "msl": "4.1.0",
        "experiment": experiment or {"StartTime": 0.0, "StopTime": 1.0, "Interval": 0.1},
        "components": list(components), "connections": [list(c) for c in connections],
        "signals": list(signals), "compared": list(compared),
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _c(name, cls, initial=None, **parameters):
    comp = {"name": name, "class": cls, "parameters": parameters}
    if initial is not None:
        comp["initial"] = initial
    return comp


def _at(grid, t):
    return min(range(len(grid)), key=lambda j: abs(grid[j] - t))


def _resistor(name, R=1.0, **extra):
    return _c(name, MSL + "Basic.Resistor", R=R, T_ref=300.15, alpha=0.0, useHeatPort=False,
              T=300.15, **extra)


def test_msl_sources_become_signal_series_with_their_phase_and_offset():
    # CompareTransformers: 0.1 + 0.1 sin(2 pi 10 t + pi/2).
    _, _, drivers, names = read_modelica_circuit(DATA / "Analog.Examples.CompareTransformers.json")
    series = drivers["series:electrical.voltage_source.value"]
    t = names.grid
    expect = 0.1 + 0.1 * torch.sin(2 * math.pi * 10.0 * t + math.pi / 2)
    for source in ("sineVoltage1", "sineVoltage2"):
        torch.testing.assert_close(series[:, names.sources[source]], expect, rtol=0,
                                   atol=1e-15)
    assert names.devices["basicTransformer"][0] == "transformer"
    assert len(names.devices["basicTransformer"][1]) == 2


def test_step_and_ramp_sources_and_a_signal_voltage(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("st", MSL + "Sources.StepVoltage", V=2.0, offset=1.0, startTime=0.35),
        _c("ra", MSL + "Sources.RampVoltage", V=4.0, duration=0.5, offset=0.0, startTime=0.15),
        _c("sv", MSL + "Sources.SignalVoltage"),
        _c("pu", MSL + "Sources.PulseVoltage", V=1.0, width=50.0, period=0.4, offset=0.0,
           startTime=0.0),
        _resistor("r1"), _resistor("r2"), _resistor("r3"), _resistor("r4"),
    ], [["st.p", "r1.p"], ["r1.n", "g.p"], ["st.n", "g.p"], ["ra.p", "r2.p"], ["r2.n", "g.p"],
        ["ra.n", "g.p"], ["sv.p", "r3.p"], ["r3.n", "g.p"], ["sv.n", "g.p"],
        ["pu.p", "r4.p"], ["r4.n", "g.p"], ["pu.n", "g.p"]],
        signals=[{"name": "k", "class": "Modelica.Blocks.Sources.Constant",
                  "parameters": {"k": 3.0}, "drives": "gain.u"},
                 {"name": "gain", "class": "Modelica.Blocks.Math.Gain",
                  "parameters": {"k": 2.0}, "drives": "sv.v"}])
    _, _, drivers, names = read_modelica_circuit(path)
    grid = names.grid.tolist()
    for event in (0.35, 0.15, 0.65):
        assert any(abs(t - event) <= 1e-12 for t in grid), event
    jump = min(range(len(grid)), key=lambda j: abs(grid[j] - 0.35))
    assert jump in names.restarts
    kink = min(range(len(grid)), key=lambda j: abs(grid[j] - 0.15))
    assert kink not in names.restarts                 # a ramp's corner is no restart
    series = drivers["series:electrical.voltage_source.value"]
    step = series[:, names.sources["st"]]
    assert float(step[jump]) == 1.0 and float(step[jump + 1]) == 3.0   # left limit at 0.35
    ramp = series[:, names.sources["ra"]]
    assert float(ramp[_at(grid, 0.5)]) == pytest.approx(4.0 * 0.35 / 0.5, rel=1e-14)
    assert torch.all(series[:, names.sources["sv"]] == 6.0)
    pulse = series[:, names.sources["pu"]]
    assert float(pulse[_at(grid, 0.1)]) == 1.0 and float(pulse[_at(grid, 0.3)]) == 0.0


def test_variable_resistor_scale_series():
    _, _, drivers, names = read_modelica_circuit(
        DATA / "Analog.Examples.ShowVariableResistor.json")
    scale = drivers["series:electrical.resistor.scale"]
    first = min(edges[0] for kind, edges in names.devices.values() if kind == "resistor")
    k = names.devices["VariableResistor"][1][0] - first
    torch.testing.assert_close(scale[:, k], 2.0 + 5.0 * names.grid / 2.0, rtol=0, atol=1e-15)
    others = [i for i in range(scale.shape[1]) if i != k]
    assert torch.all(scale[:, others] == 1.0)
    assert torch.all(drivers["electrical.resistor.T"] == 300.15)


def test_load_scales_follow_the_mbl_modes():
    _, _, drivers, names = read_modelica_circuit(DATA / "DC.Loads.Examples.VariableLoad.json")
    scale = drivers["series:electrical.load.scale"]
    first = names.devices["loa1"][1][0]
    col = {n: names.devices[n][1][0] - first for n in ("loa1", "loa2", "loa3")}
    t = names.grid
    y = torch.clamp((t - 0.3) / 0.5, 0.0, 1.0) * 0.8
    torch.testing.assert_close(scale[:, col["loa1"]], torch.ones_like(t), rtol=0, atol=0)
    torch.testing.assert_close(scale[:, col["loa2"]], 1e-10 + (1 - 1e-10) * y, rtol=0,
                               atol=1e-15)
    torch.testing.assert_close(scale[:, col["loa3"]], -20.0 + 120.0 * y / 0.8, rtol=0,
                               atol=1e-13)


def test_variable_load_is_solved_with_the_mbl_power_sign():
    # At t = 0: loa1 consumes 50 W behind its line, loa2 50 W scaled by 1e-10, and loa3
    # 20 W (its input Pow = -20: MBL counts consumption negative).
    model, state, drivers, names = read_modelica_circuit(
        DATA / "DC.Loads.Examples.VariableLoad.json")
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    h = _hist(out)
    v3 = float(names.value(h, "loa3.terminal.v[1]")[0])
    i3 = float(names.value(h, "loa3.terminal.i[1]")[0])
    assert v3 == pytest.approx(12.0) and v3 * i3 == pytest.approx(20.0, rel=1e-12)
    i2 = float(names.value(h, "loa2.terminal.i[1]")[0])
    assert 12.0 * i2 == pytest.approx(50.0 * 1e-10, rel=1e-9)


def test_phasor_source_carries_its_angle_and_the_frequency(tmp_path):
    path = _doc(tmp_path, [
        _c("V", AC + "Sources.FixedVoltage", f=50.0, V=230.0, phiSou=0.5),
        _c("Z", AC + "Loads.Impedance", R=10.0, L=0.0, C=0.0, inductive=True,
           use_R_in=False, use_L_in=False, use_C_in=False),
    ], [["V.terminal", "Z.terminal"]], compared=["Z.terminal.v[1]", "Z.terminal.v[2]",
                                                  "Z.terminal.i[1]", "Z.terminal.i[2]"])
    model, state, drivers, names = read_modelica_circuit(path)
    assert names.phasor and not names.dynamic
    assert float(drivers["electrical.frequency"]) == 50.0
    assert "series:electrical.voltage_source.value" not in drivers
    assert state["electrical.phi"].dtype == torch.complex128
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    h = _hist(out)
    v = complex(float(names.value(h, "Z.terminal.v[1]")[0]),
                float(names.value(h, "Z.terminal.v[2]")[0]))
    i = complex(float(names.value(h, "Z.terminal.i[1]")[0]),
                float(names.value(h, "Z.terminal.i[2]")[0]))
    expect = 230.0 * complex(math.cos(0.5), math.sin(0.5))
    assert abs(v - expect) <= 1e-12 and abs(i - expect / 10.0) <= 1e-13


def test_flat_start_of_an_algebraic_model():
    _, state, _, names = read_modelica_circuit(DATA / "DC.Loads.Examples.VariableLoad.json")
    phi = state["electrical.phi"]
    electrical = list(names.nodes.values())
    assert float(phi[names.nodes["0"]]) == 0.0
    assert all(float(phi[k]) == 12.0 for k in electrical if k != names.nodes["0"])
    assert torch.all(state["electrical.q"] == 0.0)


def test_initial_state_propagates_capacitor_voltages_and_inductor_currents(tmp_path):
    # V -- r -- C1 (v = 2) -- C2 (v = -0.5) -- ground, an inductor at 0.25 A, and a
    # capacitor C3 (v = 4) whose group does not reach node 0 (grounded through r3 and r4):
    # rooted at 0 V on its first node.
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("V", MSL + "Sources.ConstantVoltage", V=1.0),
        _resistor("r"),
        _c("C1", MSL + "Basic.Capacitor", {"v": 2.0}, C=1.0),
        _c("C2", MSL + "Basic.Capacitor", {"v": -0.5}, C=1.0),
        _c("L", MSL + "Basic.Inductor", {"i": 0.25}, L=1.0),
        _c("C3", MSL + "Basic.Capacitor", {"v": 4.0}, C=1.0),
        _resistor("r3"), _resistor("r4"),
    ], [["V.p", "r.p"], ["V.n", "g.p"], ["r.n", "C1.p"], ["C1.n", "C2.p"], ["C2.n", "g.p"],
        ["L.p", "C1.n"], ["L.n", "g.p"], ["C3.p", "r3.p"], ["r3.n", "g.p"], ["C3.n", "r4.p"],
        ["r4.n", "g.p"]],
        compared=["C1.v", "C2.v", "L.i", "C3.v"])
    _, state, _, names = read_modelica_circuit(path)
    h = _hist(state)
    for column, value in (("C1.v", 2.0), ("C2.v", -0.5), ("L.i", 0.25), ("C3.v", 4.0)):
        assert float(names.value(h, column)[0]) == value, column
    phi = state["electrical.phi"]
    assert float(phi[names.nodes["r.n"]]) == 1.5
    assert float(phi[names.nodes["C3.p"]]) == 0.0 and float(phi[names.nodes["C3.n"]]) == -4.0
    assert names.dynamic


def test_rc_line_capacitor_starts_at_its_recorded_voltage():
    _, state, _, names = read_modelica_circuit(DATA / "DC.Lines.Examples.RCModel.json")
    assert names.dynamic
    assert float(names.value(_hist(state), "RC_dyn.Vc")[0]) == 50.0
    kind, edges = names.devices["RC_dyn"]
    assert kind == "impedance" and len(edges) == 3


def test_heat_port_columns_map_to_the_thermal_nodes():
    *_, names = read_modelica_circuit(DATA / "Analog.Examples.HeatingRectifier.json")
    col = names.columns["HeatingDiode1.T_heatPort"]
    assert col.key == "thermal.phi"
    assert col.terms == ((names.thermal_nodes["HeatingDiode1.heatPort"], 1.0),)
    assert names.columns["HeatCapacitor1.T"].terms == (
        (names.thermal_nodes["HeatCapacitor1.port"], 1.0),)
    assert set(names.thermal_nodes.values()).isdisjoint(names.nodes.values())


def test_resistor_example_has_a_thermal_layer_coupled_by_iteration():
    model, state, drivers, names = read_modelica_circuit(DATA / "Analog.Examples.Resistor.json")
    assert list(model.potential) == ["electrical", "thermal"]
    assert model.coupling == "iterate"
    assert names.columns["resistor.T_heatPort"].key == "thermal.phi"
    thermal = model.potential["thermal"]
    assert thermal.kinds == ["conductor", "heat_source"]
    assert (thermal.quantity, thermal.unit, thermal.flow_unit) == ("temperature", "K", "W")
    assert not names.dynamic
    # boundaries: the 0 K reference and the fixed temperature (20 degC), in that order
    torch.testing.assert_close(drivers["thermal.phi_boundary"],
                               torch.tensor([0.0, 293.15], dtype=torch.float64))
    heat_port = names.thermal_nodes["resistor.heatPort"]
    assert float(state["thermal.phi"][heat_port]) == 293.15
    assert float(drivers["thermal.heat"][0]) == 0.0
    # at t = 0.25 s the source is at its 220 V peak: the hand fixed point of
    # alpha x^2 + x - V^2/(G R0) = 0 (R0 = 100, alpha = 1e-3, G = 50 W/K, T_ref = 293.15 K)
    t = float(names.grid[25])
    assert t == pytest.approx(0.25)
    out = model.steady(state, step_drivers(drivers, names.grid, t), t=t)
    assert abs(float(names.value(_hist(out), "resistor.T_heatPort")[0])
               - 302.73806893411466) <= 1e-9
    assert abs(float(names.value(_hist(out), "resistor.i")[0]) - 2.179106575935151) <= 1e-11


def test_heating_rectifier_thermal_start_and_temperature_drivers():
    model, state, drivers, names = read_modelica_circuit(
        DATA / "Analog.Examples.HeatingRectifier.json")
    thermal = model.potential["thermal"]
    assert thermal.kinds == ["conductor", "heat_source", "heat_capacitor"]
    assert names.dynamic
    # no fixed temperature: the 0 K reference is the only boundary
    torch.testing.assert_close(drivers["thermal.phi_boundary"],
                               torch.tensor([0.0], dtype=torch.float64))
    port = names.thermal_nodes["HeatingDiode1.heatPort"]
    capacitor = names.thermal_nodes["HeatCapacitor1.port"]
    assert float(state["thermal.phi"][capacitor]) == 293.15
    assert float(state["thermal.phi"][port]) == 293.15      # the mean of the known ones
    assert float(state["thermal.phi"][names.thermal_nodes["thermal.reference"]]) == 0.0
    assert state["thermal.q"].shape == (3,)
    # the heated diode reads its heat port; the resistor without one keeps its T
    assert drivers["electrical.diode.T"].tolist() == [293.15]
    assert drivers["electrical.resistor.T"].tolist() == [300.15]
    closure = model.closures[0]
    out = closure(state, drivers)
    assert set(out) == {"thermal.heat", "electrical.diode.T"}


def test_thermal_components_without_a_heated_device_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("V", MSL + "Sources.ConstantVoltage", V=1.0),
        _resistor("r"),
        _c("G", MSL + "Basic.Ground"),
        _c("hc", "Modelica.Thermal.HeatTransfer.Components.HeatCapacitor", {"T": 300.0},
           C=1.0),
    ], [["V.p", "r.p"], ["V.n", "G.p"], ["r.n", "G.p"]])
    with pytest.raises(ModelicaImportError, match="the thermal components heat no circuit "
                                                  "device"):
        read_modelica_circuit(path)


def test_column_value_takes_the_part_and_the_weighted_sum():
    names = CircuitNames(
        model="M", times=torch.zeros(1), grid=torch.zeros(1), restarts=frozenset({0}),
        dynamic=False, phasor=True, nodes={}, thermal_nodes={}, devices={}, sources={},
        columns={"v": Column("electrical.phi", ((1, 1.0), (2, -1.0)), "imag")})
    hist = {"electrical.phi": torch.tensor([[0, 3 + 4j, 1 + 1j], [0, 1j, 0]],
                                           dtype=torch.complex128)}
    torch.testing.assert_close(names.value(hist, "v"), torch.tensor([3.0, 1.0],
                                                                    dtype=torch.float64))


def test_unmapped_column_and_unsupported_settings_are_named_together(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("V", MSL + "Sources.ConstantVoltage", V=1.0),
        _c("r", MSL + "Basic.Resistor", useHeatPort=False),
        _c("d1", MSL + "Semiconductors.Diode", Ids=1e-6, Vt=0.04, Maxexp=15.0, R=1e8,
           useTemperatureDependency=False, useHeatPort=False, T=300.15),
    ], [["V.p", "r.p"], ["V.n", "g.p"], ["r.n", "d1.p"], ["d1.n", "g.p"]],
        signals=[{"name": "k", "class": "Modelica.Blocks.Sources.Constant",
                  "parameters": {"k": 1.0}, "drives": "r.R"}],
        compared=["r.i", "r.bogus", "nobody.i"])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica_circuit(path)
    msg = str(exc.value)
    assert "r.bogus" in msg and "nobody.i" in msg
    assert "parameter R" in msg              # a fixed resistor needs its R
    assert "drives r.R" in msg               # a fixed resistor reads no input R


def test_ac_load_near_the_smoothing_band_and_power_factor_input_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("V", AC + "Sources.FixedVoltage", f=50.0, V=230.0, phiSou=0.0),
        _c("a", AC + "Loads.Inductive", P_nominal=-100.0, V_nominal=230.0, pf=0.99999,
           linearized=False, use_pf_in=False,
           mode="Buildings.Electrical.Types.Load.FixedZ_steady_state"),
        _c("b", AC + "Loads.Inductive", P_nominal=-100.0, V_nominal=230.0, pf=0.9,
           linearized=False, use_pf_in=True,
           mode="Buildings.Electrical.Types.Load.FixedZ_steady_state"),
        _c("z", AC + "Loads.Impedance", R=1.0, L=0.0, C=0.0, inductive=False,
           use_R_in=False, use_L_in=False, use_C_in=False),
    ], [["V.terminal", "a.terminal"], ["V.terminal", "b.terminal"],
        ["V.terminal", "z.terminal"]])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica_circuit(path)
    msg = str(exc.value)
    assert "a (Buildings.Electrical.AC.OnePhase.Loads.Inductive): pf = 0.99999" in msg
    assert "b (Buildings.Electrical.AC.OnePhase.Loads.Inductive): use_pf_in = True" in msg
    assert ("z (Buildings.Electrical.AC.OnePhase.Loads.Impedance): a capacitive impedance "
            "needs C > 0") in msg


def test_algebraic_phasor_model_with_a_converter_and_zero_demand_converges():
    model, state, drivers, names = read_modelica_circuit(
        DATA / "AC.OnePhase.Conversion.Examples.ACACTransformer.json")
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    assert torch.isfinite(out["electrical.q"]).all()
    model, state, drivers, names = read_modelica_circuit(
        DATA / "AC.OnePhase.Conversion.Examples.ACDCConverter.json")
    out = model.steady(state, step_drivers(drivers, names.grid, 0.0), t=0.0)
    v_dc = float(names.value(_hist(out), "conversion.terminal_p.v[1]")[0])
    assert v_dc == pytest.approx(0.5 * 120.0, rel=1e-9)


def test_signal_wiring_and_start_value_problems_are_named(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("sv", MSL + "Sources.SignalVoltage"),
        _c("vr", MSL + "Basic.VariableResistor", T_ref=300.15, alpha=0.0, useHeatPort=False,
           T=300.15),
        _c("C", MSL + "Basic.Capacitor", C=1.0),
    ], [["sv.p", "vr.p"], ["vr.n", "C.p"], ["C.n", "g.p"], ["sv.n", "g.p"]],
        signals=[{"name": "add", "class": "Modelica.Blocks.Math.Add", "parameters": {},
                  "drives": "sv.v"},
                 {"name": "one", "class": "Modelica.Blocks.Sources.Constant",
                  "parameters": {"k": 1.0}, "drives": ["add.u1", "add.u1"]},
                 {"name": "fb", "class": "Modelica.Blocks.Math.Gain",
                  "parameters": {"k": 1.0}, "drives": ["vr.R"]},
                 {"name": "fb2", "class": "Modelica.Blocks.Math.Gain",
                  "parameters": {"k": 1.0}, "drives": ["fb.u"]},
                 {"name": "fb3", "class": "Modelica.Blocks.Math.Gain",
                  "parameters": {"k": 1.0}, "drives": ["fb2.u"]}])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica_circuit(path)
    msg = str(exc.value)
    assert "already drives" in msg                 # add.u1 driven twice
    assert "input u2 is not driven" in msg         # add.u2 undriven
    assert "input u is not driven" in msg          # fb3.u undriven
    assert "start value of v" in msg               # C has no recorded start value


def test_transformer_ratings_and_power_factors_are_checked(tmp_path):
    rating = {"Zperc": 0.03, "XoverR": 8.0, "ground_1": False, "ground_2": True}
    load = {"P_nominal": -100.0, "V_nominal": 60.0, "linearized": False, "use_pf_in": False,
            "mode": "Buildings.Electrical.Types.Load.FixedZ_steady_state"}
    path = _doc(tmp_path, [
        _c("V", AC + "Sources.FixedVoltage", f=60.0, V=120.0, phiSou=0.0),
        _c("t1", AC + "Conversion.ACACTransformer", VHigh=120.0, VLow=0.0, **rating),
        _c("t2", AC + "Conversion.ACACTransformer", VHigh=120.0, VLow=60.0, **rating),
        _c("t3", AC + "Conversion.ACACTransformer", VHigh=-1.0, VLow=60.0, VABase=4000.0,
           **rating),
        _c("t4", AC + "Conversion.ACACTransformer", VHigh=120.0, VLow=60.0, VABase=1e400,
           **rating),
        _c("t5", AC + "Conversion.ACACTransformer", VHigh=120.0, VLow=60.0, VABase=10 ** 400,
           Zperc=10 ** 400, XoverR=8.0, ground_1=False, ground_2=True),
        _c("l1", AC + "Loads.Inductive", pf=-2.0, **load),
        _c("l2", AC + "Loads.Inductive", pf=0.0, **load),
        _c("l3", AC + "Loads.Inductive", pf=1.5, **load),
    ], [["V.terminal", "t1.terminal_n"], ["V.terminal", "t2.terminal_n"],
        ["V.terminal", "t3.terminal_n"], ["V.terminal", "t4.terminal_n"],
        ["V.terminal", "t5.terminal_n"],
        ["t1.terminal_p", "l1.terminal"],
        ["t2.terminal_p", "l2.terminal"], ["t3.terminal_p", "l3.terminal"]])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica_circuit(path)
    msg = str(exc.value)
    tra = AC + "Conversion.ACACTransformer"
    assert f"t1 ({tra}): parameter VLow must be a finite number > 0, found 0.0" in msg
    assert f"t1 ({tra}): parameter VABase must be a finite number > 0, found None" in msg
    assert f"t2 ({tra}): parameter VABase must be a finite number > 0, found None" in msg
    assert f"t3 ({tra}): parameter VHigh must be a finite number > 0, found -1.0" in msg
    assert f"t4 ({tra}): parameter VABase must be a finite number > 0, found inf" in msg
    # a 400-digit integer overflows a float: named, not an OverflowError
    assert f"t5 ({tra}): parameter VABase must be a finite number > 0, found 1000000" in msg
    assert f"t5 ({tra}): parameter Zperc must be a finite number, found 1000000" in msg
    for name, pf in (("l1", -2.0), ("l2", 0.0), ("l3", 1.5)):
        assert (f"{name} (Buildings.Electrical.AC.OnePhase.Loads.Inductive): pf = {pf!r}: a "
                f"power factor must lie in (0, 1]") in msg


def test_a_signal_the_evaluator_cannot_read_is_named(tmp_path):
    # A table column out of range raises IndexError inside the evaluator.
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("sv", MSL + "Sources.SignalVoltage"),
        _resistor("r"),
    ], [["sv.p", "r.p"], ["r.n", "g.p"], ["sv.n", "g.p"]],
        signals=[{"name": "tab", "class": "Modelica.Blocks.Sources.CombiTimeTable",
                  "parameters": {"table": [[0.0, 1.0], [1.0, 2.0]], "columns": [5]},
                  "drives": "sv.v"}])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica_circuit(path)
    expected = "tab (Modelica.Blocks.Sources.CombiTimeTable): cannot be evaluated (IndexError"
    assert expected in str(exc.value)
