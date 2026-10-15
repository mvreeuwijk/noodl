"""The electrical Modelica reader's schema and netlist: class roles, nodes, refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noodl.apps.electrical.modelica.netlist import build
from noodl.apps.electrical.modelica.schema import ModelicaImportError, domain, load

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica_electrical"
_EXPERIMENT = {"StartTime": 0.0, "StopTime": 1.0, "Interval": 0.1, "Tolerance": 1e-12}


def _doc(tmp_path, components, connections, signals=(), experiment=None) -> Path:
    doc = {
        "format": "noodl-modelica-electrical/1",
        "model": "M",
        "mbl_commit": "x",
        "openmodelica": "omc",
        "msl": "4.1.0",
        "experiment": dict(experiment or _EXPERIMENT),
        "components": list(components),
        "connections": [list(c) for c in connections],
        "signals": list(signals),
        "compared": [],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(doc))
    return path


def _c(name, cls, **parameters):
    return {"name": name, "class": cls, "parameters": parameters}


def _refusal(path, **kwargs) -> str:
    with pytest.raises(ModelicaImportError) as exc:
        build(load(path), **kwargs)
    return str(exc.value)


MSL = "Modelica.Electrical.Analog."
DC = "Buildings.Electrical.DC."
AC = "Buildings.Electrical.AC.OnePhase."
AC3 = "Buildings.Electrical.AC.ThreePhasesBalanced."
HEAT = "Modelica.Thermal.HeatTransfer."


# ------------------------------------------------------------------ the brief's tests
def test_cauer_netlist_from_the_fixture():
    nl = build(load(DATA / "Analog.Examples.CauerLowPassAnalog.json"))
    assert not nl.phasor and nl.frequency is None
    # nodes: reference + R1.p/V.p, R1.n/C1.p/L1.p/C2.p, L1.n/C2.n/C3.p/C4.p/L2.p,
    #        L2.n/C4.n/C5.p/R2.p
    assert len(nl.node_names) == 5
    c1 = next(d for d in nl.devices if d.name == "C1")
    assert c1.nodes["n"] == 0 and c1.initial == {"v": 0.0}


def test_dc_terminals_are_two_nodes_and_sensors_are_wires():
    nl = build(load(DATA / "DC.Loads.Examples.Resistor.json"))
    res1 = next(d for d in nl.devices if d.name == "res1")
    sou = next(d for d in nl.devices if d.name == "sou")
    assert res1.nodes["c1"] == sou.nodes["c1"] != 0      # sen1 is a wire
    assert res1.nodes["c2"] == sou.nodes["c2"] == 0      # sou.n is grounded
    assert not any(d.role == "wire" for d in nl.devices)


def test_ac_model_is_phasor_with_the_source_frequency():
    nl = build(load(DATA / "AC.OnePhase.Loads.Examples.TestImpedance.json"))
    assert nl.phasor and nl.frequency == 60.0
    assert {d.nodes["a"] for d in nl.devices} == {1}


def test_dynamic_phasor_line_is_refused_unless_omitted():
    path = DATA / "AC.OnePhase.Lines.Examples.ACLine_RL.json"
    with pytest.raises(ModelicaImportError, match=r"RL_3 .*dynamic phasor"):
        build(load(path))
    nl = build(load(path), omit=("RL_3", "load_sc_3"))
    assert "RL_3" not in {d.name for d in nl.devices}


def test_refusals_are_gathered_and_name_every_instance(tmp_path):
    path = _doc(tmp_path, [
        _c("bat", "Buildings.Electrical.DC.Storage.Battery"),
        _c("pv", "Buildings.Electrical.DC.Sources.PVSimple"),
        _c("d", "Modelica.Electrical.Analog.Ideal.IdealDiode"),
        _c("x", "Some.Unknown.Class"),
    ], [])
    with pytest.raises(ModelicaImportError) as exc:
        build(load(path))
    msg = str(exc.value)
    for name in ("bat", "pv", "d", "x"):
        assert name in msg
    assert "4 problem" in msg


def test_connection_errors_are_named(tmp_path):
    path = _doc(tmp_path, [
        _c("g", "Modelica.Electrical.Analog.Basic.Ground"),
        _c("r", "Modelica.Electrical.Analog.Basic.Resistor", useHeatPort=True, R=1.0),
        _c("s", "Buildings.Electrical.DC.Sources.ConstantVoltage", V=12.0),
        _c("z", "Buildings.Electrical.AC.OnePhase.Loads.Impedance", R=1.0),
        _c("l", "Buildings.Electrical.DC.Loads.Resistor", R=1.0),
    ], [["r.p", "g.p"], ["r.n", "l.terminal"], ["s.terminal", "z.terminal"]])
    msg = _refusal(path)
    assert "r.heatPort" in msg            # enabled, unconnected
    assert "r.n" in msg and "l.terminal" in msg   # pin to terminal
    assert "s.terminal" in msg and "z.terminal" in msg  # DC to AC
    assert "s.n" in msg                    # DC source n pin not grounded


def test_phasor_model_refuses_dc_storage_and_mixed_frequencies(tmp_path):
    path = _doc(tmp_path, [
        _c("a", "Buildings.Electrical.AC.OnePhase.Sources.FixedVoltage", f=60.0, V=120.0),
        _c("b", "Buildings.Electrical.AC.OnePhase.Sources.FixedVoltage", f=50.0, V=120.0),
        _c("z", "Buildings.Electrical.AC.OnePhase.Loads.Impedance", R=1.0),
        _c("rc", "Buildings.Electrical.DC.Lines.TwoPortRCLine", R=1.0, C=1.0, use_C=True),
    ], [["a.terminal", "z.terminal"], ["b.terminal", "z.terminal"]])
    msg = _refusal(path)
    assert "rc" in msg and "phasor" in msg
    assert "50.0" in msg and "60.0" in msg


def _line(name, cls, derived, **parameters):
    comp = _c(name, cls, **parameters)
    comp["derived"] = derived
    return comp


def test_line_rule_is_checked_against_the_simulated_values(tmp_path):
    one = AC + "Lines.Line"
    params = {"R": 0.9, "L": 3e-3, "use_T": False, "modelMode": "Buildings.Electrical.Types.Load."
              "FixedZ_steady_state", "mode": "Buildings.Electrical.Types.CableMode.commercial"}
    components = [
        _c("E", AC + "Sources.FixedVoltage", f=60.0, V=120.0),
        _line("good", one, {"line.R": 0.9 / 3, "line.L": 3e-3 / 3, "line.C": 0.0}, **params),
        _line("bad", one, {"line.R": 0.9, "line.L": 3e-3 / 3, "line.C": 0.0}, **params),
        _c("Z", AC + "Loads.Impedance", R=1.0),
    ]
    connections = [["E.terminal", "good.terminal_n"], ["good.terminal_p", "bad.terminal_n"],
                   ["bad.terminal_p", "Z.terminal"]]
    msg = _refusal(_doc(tmp_path, components, connections))
    assert "1 problem" in msg and "bad" in msg and "good" not in msg
    assert repr(0.9) in msg and repr(0.9 / 3) in msg
    nl = build(load(_doc(tmp_path, [c for c in components if c["name"] != "bad"],
                         [["E.terminal", "good.terminal_n"], ["good.terminal_p", "Z.terminal"]])))
    good = next(d for d in nl.devices if d.name == "good")
    assert good.derived["line.R"] == pytest.approx(0.3)


def test_balanced_line_rule_uses_the_full_values(tmp_path):
    bal = AC3 + "Lines.Line"
    components = [
        _c("E", AC3 + "Sources.FixedVoltage", f=60.0, V=480.0),
        _line("ln", bal, {"line.R": 0.724, "line.L": 4e-4, "line.C": 0.0},
              R=0.724, L=4e-4, C=0.0, use_C=False, use_T=False),
        _c("Z", AC3 + "Loads.Impedance", R=10.0),
    ]
    nl = build(load(_doc(tmp_path, components,
                         [["E.terminal", "ln.terminal_n"], ["ln.terminal_p", "Z.terminal"]])))
    ln = next(d for d in nl.devices if d.name == "ln")
    assert ln.role == "ac_line" and set(ln.nodes) == {"p", "n", "mid"}
    assert nl.node_names[ln.nodes["mid"]] == "ln.mid"
    components[1]["derived"]["line.R"] = 0.724 / 3
    components[1]["derived"]["line.C"] = 1e-9
    msg = _refusal(_doc(tmp_path, components,
                        [["E.terminal", "ln.terminal_n"], ["ln.terminal_p", "Z.terminal"]]))
    assert "line.R" in msg and "line.C" in msg and "2 problems" in msg


# ------------------------------------------------------------------ further behaviour
def test_resistor_heat_port_joins_the_thermal_network_in_kelvin():
    nl = build(load(DATA / "Analog.Examples.Resistor.json"))
    assert nl.thermal_names[0] == "thermal.reference" and nl.node_names[0] == "0"
    res = next(d for d in nl.devices if d.name == "resistor")
    cond = next(d for d in nl.devices if d.name == "thermalConductor")
    fixed = next(d for d in nl.devices if d.name == "fixedTemperature")
    assert res.heat_node == cond.nodes["port_a"] != 0
    assert cond.nodes["port_b"] == fixed.nodes["port"] != 0
    assert nl.fixed_temperatures == {fixed.nodes["port"]: pytest.approx(293.15, abs=1e-12)}
    assert fixed.parameters == {"T": 20.0}   # parameters stay as exported (degC)


def test_heating_rectifier_diode_heat_port_and_heat_capacitor():
    nl = build(load(DATA / "Analog.Examples.HeatingRectifier.json"))
    diode = next(d for d in nl.devices if d.name == "HeatingDiode1")
    cap = next(d for d in nl.devices if d.name == "HeatCapacitor1")
    assert diode.role == "diode" and diode.heat_node is not None
    assert cap.role == "heat_capacitor" and cap.initial == {"T": 293.15}
    assert len(nl.thermal_names) == 3 and nl.fixed_temperatures == {}
    r = next(d for d in nl.devices if d.name == "R")
    assert r.heat_node is None


def test_rc_line_mid_node_and_shared_second_conductor():
    nl = build(load(DATA / "DC.Lines.Examples.RCModel.json"))
    rc = next(d for d in nl.devices if d.name == "RC_dyn")
    sc = next(d for d in nl.devices if d.name == "sc_dyn")
    assert rc.role == "dc_rc_line" and rc.initial == {"Vc": 50.0}
    assert nl.node_names[rc.nodes["mid"]] == "RC_dyn.mid"
    assert rc.nodes["c2"] == 0 and sc.nodes["c2"] == 0
    assert rc.nodes["n1"] == sc.nodes["c1"] != rc.nodes["p1"]


def test_converter_dc_side_and_frequency():
    nl = build(load(DATA / "AC.OnePhase.Conversion.Examples.ACDCConverter.json"))
    conv = next(d for d in nl.devices if d.name == "conversion")
    res = next(d for d in nl.devices if d.name == "res")
    assert nl.phasor and nl.frequency == 60.0
    assert conv.nodes["p2"] == 0 and conv.nodes["p1"] == res.nodes["c1"] != 0
    assert len(nl.signals) == 1 and nl.signals[0].drives == ("load.Pow",)


def test_converter_without_dc_ground_is_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("sou", AC + "Sources.FixedVoltage", f=60.0, V=120.0),
        _c("conv", AC + "Conversion.ACDCConverter", conversionFactor=0.5, eta=0.9,
           ground_AC=False, ground_DC=False),
        _c("res", DC + "Loads.Resistor", R=1.0),
    ], [["sou.terminal", "conv.terminal_n"], ["conv.terminal_p", "res.terminal"]])
    msg = _refusal(path)
    assert "conv" in msg and "ground_DC" in msg and "1 problem" in msg


def test_every_fixture_builds():
    for path in sorted(p for p in DATA.glob("*.json") if p.name != "parity-electrical.json"):
        omit = ("RL_3", "load_sc_3") if "ACLine_RL" in path.name else ()
        nl = build(load(path), omit=omit)
        assert nl.devices, path.name
        assert len(set(nl.node_names)) == len(nl.node_names)
        for d in nl.devices:
            assert all(0 <= n < len(nl.thermal_names if d.role in _THERMAL else nl.node_names)
                       for n in d.nodes.values()), (path.name, d.name)


_THERMAL = {"thermal_conductor", "heat_capacitor", "fixed_temperature"}


def test_omitted_components_and_their_signals_are_dropped(tmp_path):
    ramp = {"name": "ramp", "class": "Modelica.Blocks.Sources.Ramp",
            "parameters": {"height": 1.0}, "drives": ["l2.Pow"]}
    keep = {"name": "keep", "class": "Modelica.Blocks.Sources.Ramp",
            "parameters": {"height": 1.0}, "drives": ["l1.Pow", "l2.Pow"]}
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("s", DC + "Sources.ConstantVoltage", V=12.0),
        _c("l1", DC + "Loads.Conductor", mode="Buildings.Electrical.Types.Load.VariableZ_P_input"),
        _c("l2", DC + "Loads.Conductor", mode="Buildings.Electrical.Types.Load.VariableZ_P_input"),
        _c("bat", DC + "Storage.Battery"),
    ], [["s.n", "g.p"], ["s.terminal", "l1.terminal"], ["s.terminal", "l2.terminal"],
        ["s.terminal", "bat.terminal"]], signals=[ramp, keep])
    assert "bat" in _refusal(path)
    nl = build(load(path), omit=("l2", "bat"))
    assert {d.name for d in nl.devices} == {"s", "l1"}
    assert [s.name for s in nl.signals] == ["keep"]
    assert nl.signals[0].drives == ("l1.Pow",)
    assert "nothing" in _refusal(path, omit=("nothing",))


def test_two_fixed_temperatures_on_one_node_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("a", HEAT + "Sources.FixedTemperature", T=300.0),
        _c("b", HEAT + "Celsius.FixedTemperature", T=20.0),
    ], [["a.port", "b.port"]])
    msg = _refusal(path)
    assert "a" in msg and "b" in msg and "fixed temperature" in msg


def test_mbl_temperature_dependence_and_three_phase_unbalanced_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("ln", AC3 + "Lines.Line", use_T=True, R=1.0, L=0.0, C=0.0),
        _c("r", DC + "Loads.Resistor", useHeatPort=True, R=1.0),
        _c("u", "Buildings.Electrical.AC.ThreePhasesUnbalanced.Loads.Impedance"),
        _c("cl", AC + "Lines.Line", use_T=False,
           modelMode="Buildings.Electrical.Types.Load.FixedZ_dynamic",
           mode="Buildings.Electrical.Types.CableMode.commercial"),
        _c("t", MSL + "Ideal.IdealTransformer", n=2.0, considerMagnetization=True),
        _c("grid", AC + "Sources.Grid"),
    ], [])
    msg = _refusal(path)
    for needle in ("ln ", "use_T", "r ", "heat port", "u ", "unbalanced", "cl ",
                   "dynamic phasor", "t ", "considerMagnetization", "grid ", "grid sources"):
        assert needle in msg, needle


def test_unknown_connectors_and_instances_are_named(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("r", MSL + "Basic.Resistor", R=1.0, useHeatPort=False),
    ], [["r.p", "g.p"], ["r.q", "g.p"], ["nobody.p", "r.n"]])
    msg = _refusal(path)
    assert "r.q" in msg and "nobody.p" in msg


def test_observers_are_ignored_and_do_not_join_nodes(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("v", MSL + "Sources.ConstantVoltage", V=1.0),
        _c("r", MSL + "Basic.Resistor", R=1.0, useHeatPort=False),
        _c("vs", MSL + "Sensors.VoltageSensor"),
        {"name": "probe", "class": "Some.Probe", "parameters": {}, "role": "observer"},
    ], [["v.n", "g.p"], ["v.p", "r.p"], ["r.n", "g.p"], ["vs.p", "r.p"], ["vs.n", "g.p"],
        ["probe.x", "r.p"]])
    nl = build(load(path))
    assert {d.name for d in nl.devices} == {"v", "r"}
    assert len(nl.node_names) == 2


def test_phasor_model_refuses_msl_components(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("E", AC + "Sources.FixedVoltage", f=50.0, V=230.0),
        _c("r", MSL + "Basic.Resistor", R=1.0, useHeatPort=False),
        _c("Z", AC + "Loads.Impedance", R=1.0),
    ], [["E.terminal", "Z.terminal"], ["r.p", "g.p"], ["r.n", "g.p"]])
    msg = _refusal(path)
    assert "r " in msg and "phasor" in msg and "1 problem" in msg


def test_domains_of_the_roles():
    assert domain("resistor") == "analog" and domain("heat_capacitor") == "thermal"
    assert domain("dc_rc_line") == "dc" and domain("ac_dc_converter") == "ac"


def test_load_refuses_malformed_documents(tmp_path):
    path = _doc(tmp_path, [_c("a", MSL + "Basic.Ground"), _c("a", MSL + "Basic.Ground")], [])
    with pytest.raises(ModelicaImportError, match="duplicate"):
        load(path)
    raw = json.loads(path.read_text())
    raw["format"] = "noodl-modelica/1"
    path.write_text(json.dumps(raw))
    with pytest.raises(ModelicaImportError, match="format"):
        load(path)
    raw["format"] = "noodl-modelica-electrical/1"
    del raw["msl"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ModelicaImportError, match="msl"):
        load(path)
    raw["msl"] = "4.1.0"
    raw["components"] = [{"name": "a", "class": "X", "parameters": {}, "initial": [1]}]
    path.write_text(json.dumps(raw))
    with pytest.raises(ModelicaImportError, match=r"components\[0\] \(a\)\.initial"):
        load(path)
    raw["components"] = []
    raw["compared"] = "x"
    path.write_text(json.dumps(raw))
    with pytest.raises(ModelicaImportError, match="compared"):
        load(path)


def test_line_without_simulated_values_is_refused(tmp_path):
    components = [
        _c("E", AC + "Sources.FixedVoltage", f=60.0, V=120.0),
        _c("nod", AC + "Lines.Line", R=1.0, L=1e-3, use_T=False),
        _c("nor", AC3 + "Lines.Line", L=1e-3, C=0.0, use_T=False),
        _line("part", AC + "Lines.Line", {"line.R": 1.0 / 3}, R=1.0, L=1e-3, use_T=False),
    ]
    msg = _refusal(_doc(tmp_path, components, []))
    assert "nod " in msg and "'derived'" in msg
    assert "nor " in msg and "parameter(s) R missing" in msg
    assert "part " in msg and "line.L is missing" in msg and "3 problems" in msg


# ------------------------------------------------------------------ review follow-ups
def test_connection_to_a_disabled_heat_port_is_dropped(tmp_path):
    # Modelica removes a connect() to a disabled conditional connector.
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("v", MSL + "Sources.ConstantVoltage", V=1.0),
        _c("r", MSL + "Basic.Resistor", R=1.0, useHeatPort=False),
        _c("k", HEAT + "Components.ThermalConductor", G=1.0),
        _c("amb", HEAT + "Sources.FixedTemperature", T=300.0),
    ], [["v.p", "r.p"], ["v.n", "g.p"], ["r.n", "g.p"], ["r.heatPort", "k.port_a"],
        ["k.port_b", "amb.port"]])
    nl = build(load(path))
    r = next(d for d in nl.devices if d.name == "r")
    assert r.heat_node is None and "r.heatPort" not in nl.thermal_of


def test_connection_with_a_signal_block_end_is_skipped(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("v", MSL + "Sources.SignalVoltage"),
        _c("r", MSL + "Basic.Resistor", R=1.0, useHeatPort=False),
    ], [["v.p", "r.p"], ["v.n", "g.p"], ["r.n", "g.p"], ["ramp.y", "v.v"]],
        signals=[{"name": "ramp", "class": "Modelica.Blocks.Sources.Ramp",
                  "parameters": {"duration": 1.0}, "drives": "v.v"}])
    nl = build(load(path))
    assert {d.name for d in nl.devices} == {"v", "r"}


def test_missing_or_malformed_deciding_parameters_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("E", AC + "Sources.FixedVoltage", f="sixty", V=120.0),
        _c("ln", AC3 + "Lines.Line", R=1.0, L=0.0, C=0.0),
        _c("t", MSL + "Ideal.IdealTransformer", n=2.0),
        _c("amb", HEAT + "Sources.FixedTemperature", T="warm"),
    ], [])
    msg = _refusal(path)
    assert "E " in msg and "f must be a finite number" in msg
    assert "ln " in msg and "use_T must be true or false" in msg
    assert "t " in msg and "considerMagnetization must be true or false" in msg
    assert "amb " in msg and "T must be a finite number" in msg


def test_rc_line_capacitor_off_the_reference_is_refused(tmp_path):
    # The line's conductor 2 reaches only a load's conductor 2: not the reference.
    path = _doc(tmp_path, [
        _c("rc", DC + "Lines.TwoPortRCLine", R=1.0, C=1.0, use_C=True, T=298.15,
           T_ref=298.15),
        _c("a", DC + "Loads.Resistor", R=1.0, T=300.15, T_ref=300.15),
        _c("b", DC + "Loads.Resistor", R=1.0, T=300.15, T_ref=300.15),
    ], [["a.terminal", "rc.terminal_n"], ["rc.terminal_p", "b.terminal"]])
    msg = _refusal(path)
    assert "rc " in msg and "conductor 2" in msg
    build(load(DATA / "DC.Lines.Examples.RCModel.json"))  # grounded through the source


def test_mbl_resistive_parts_off_their_reference_temperature_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("s", DC + "Sources.ConstantVoltage", V=12.0),
        _c("res", DC + "Lines.TwoPortResistance", R=1.0, T=310.0, T_ref=298.15),
        _c("l", DC + "Loads.Resistor", R=1.0, T=320.0, T_ref=300.15),
        _c("ok", DC + "Loads.Resistor", R=1.0, T=300.15, T_ref=300.15),
    ], [["s.n", "g.p"], ["s.terminal", "res.terminal_n"], ["res.terminal_p", "l.terminal"],
        ["s.terminal", "ok.terminal"]])
    msg = _refusal(path)
    assert "res " in msg and "l " in msg and "ok " not in msg
    assert "T = 310.0" in msg and "T_ref = 298.15" in msg


def test_load_reads_utf8(tmp_path):
    path = _doc(tmp_path, [_c("g", MSL + "Basic.Ground")], [])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model"] = "Modèle"
    path.write_bytes(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    assert load(path).model == "Modèle"


def test_diode_heat_port_without_temperature_dependency_is_refused(tmp_path):
    # the heat would flow, but the diode law would ignore the heat-port temperature
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("v", MSL + "Sources.ConstantVoltage", V=1.0),
        _c("d", MSL + "Semiconductors.Diode", useHeatPort=True,
           useTemperatureDependency=False),
        _c("amb", HEAT + "Sources.FixedTemperature", T=300.0),
    ], [["v.p", "d.p"], ["v.n", "g.p"], ["d.n", "g.p"], ["d.heatPort", "amb.port"]])
    msg = _refusal(path)
    assert "d (" in msg and "useHeatPort = true" in msg
    assert "useTemperatureDependency = false" in msg


def test_floating_thermal_network_is_refused_naming_its_nodes(tmp_path):
    # r1's heat port reaches only a conductor (no fixed temperature, no heat capacitor);
    # r2's is held by a heat capacitor, r3's by a fixed temperature through a conductor
    path = _doc(tmp_path, [
        _c("g", MSL + "Basic.Ground"),
        _c("v", MSL + "Sources.ConstantVoltage", V=1.0),
        _c("r1", MSL + "Basic.Resistor", R=1.0, useHeatPort=True),
        _c("r2", MSL + "Basic.Resistor", R=1.0, useHeatPort=True),
        _c("r3", MSL + "Basic.Resistor", R=1.0, useHeatPort=True),
        _c("k1", HEAT + "Components.ThermalConductor", G=1.0),
        _c("cap", HEAT + "Components.HeatCapacitor", C=1.0),
        _c("k3", HEAT + "Components.ThermalConductor", G=1.0),
        _c("amb", HEAT + "Sources.FixedTemperature", T=300.0),
    ], [["v.p", "r1.p"], ["v.p", "r2.p"], ["v.p", "r3.p"], ["v.n", "g.p"], ["r1.n", "g.p"],
        ["r2.n", "g.p"], ["r3.n", "g.p"], ["r1.heatPort", "k1.port_a"],
        ["r2.heatPort", "cap.port"], ["r3.heatPort", "k3.port_a"], ["k3.port_b", "amb.port"]])
    msg = _refusal(path)
    assert "floating" in msg and "r1.heatPort" in msg and "k1.port_b" in msg
    assert "r2.heatPort" not in msg and "r3.heatPort" not in msg


def test_non_finite_frequency_and_fixed_temperature_are_refused(tmp_path):
    path = _doc(tmp_path, [
        _c("E", AC + "Sources.FixedVoltage", f=float("nan"), V=120.0),
        _c("amb", HEAT + "Sources.FixedTemperature", T=float("inf")),
    ], [])
    msg = _refusal(path)
    assert "E " in msg and "f must be a finite number, found nan" in msg
    assert "amb " in msg and "T must be a finite number, found inf" in msg
