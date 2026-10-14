"""Contract test for `scripts/modelica_export_electrical.py`.

Runs without OpenModelica: the exporter's pure-Python writer is fed synthetic
`getModelInstance` trees (reduced to the fields it reads). The script is imported by path, so it
stays out of the `noodl` package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "modelica_export_electrical.py"


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("modelica_export_electrical", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelica_export_electrical"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ synthetic instance tree
def _param(name, value=None, binding=None, *, public=True, type_="Real"):
    prefixes = {"variability": "parameter"}
    if not public:
        prefixes["public"] = False
    v = {}
    if binding is not None:
        v["binding"] = binding
    if value is not None:
        v["value"] = value
    return {"$kind": "component", "name": name, "type": type_, "value": v,
            "prefixes": prefixes}


def _var(name, type_name="Real"):
    return {"$kind": "component", "name": name, "type": type_name}


def _conn(name, type_name):
    return {"$kind": "component", "name": name,
            "type": {"name": type_name, "restriction": "connector"}}


def _cref(ref):
    return {"$kind": "cref", "parts": [{"name": p} for p in ref.split(".")]}


def _cls(name, elements, restriction="model", bases=(), connections=()):
    ext = [{"$kind": "extends", "baseClass": b} for b in bases]
    return {"name": name, "restriction": restriction, "elements": ext + list(elements),
            "connections": [{"lhs": _cref(a), "rhs": _cref(b)} for a, b in connections]}


def _comp(name, type_):
    return {"$kind": "component", "name": name, "type": type_}


def _one_port(cls, *params):
    return _cls(cls, [_conn("p", "Modelica.Electrical.Analog.Interfaces.PositivePin"),
                      _conn("n", "Modelica.Electrical.Analog.Interfaces.NegativePin"),
                      _var("v"), _var("i"), *params])


def _rc_instance():
    return _cls("RC", [
        _comp("G", _cls("Modelica.Electrical.Analog.Basic.Ground",
                        [_conn("p", "Modelica.Electrical.Analog.Interfaces.Pin")])),
        _comp("V", _one_port("Modelica.Electrical.Analog.Sources.StepVoltage",
                             _param("V", 1.0), _param("startTime", 0.25))),
        _comp("R1", _one_port("Modelica.Electrical.Analog.Basic.Resistor", _param("R", 1.0),
                              _param("useHeatPort", False, type_="Boolean"))),
        _comp("C1", _one_port("Modelica.Electrical.Analog.Basic.Capacitor", _param("C", 1.0))),
        _comp("A", _one_port("Modelica.Electrical.Analog.Sensors.CurrentSensor")),
    ], connections=[("V.p", "R1.p"), ("R1.n", "A.p"), ("A.n", "C1.p"), ("C1.n", "G.p"),
                    ("V.n", "G.p")])


def _document(exporter, inst, model="M"):
    return exporter.build_circuit_document(inst, model=model, experiment={"StartTime": 0.0},
                                           experiment_declared={}, mbl_commit="x",
                                           openmodelica="omc", msl="4.1.0")


# ------------------------------------------------------------------------------- tests
def test_msl_columns_states_and_wires(exporter):
    inst = _rc_instance()
    doc = exporter.build_circuit_document(inst, model="RC", experiment={"StartTime": 0.0},
                                          experiment_declared={}, mbl_commit="x",
                                          openmodelica="omc", msl="4.1.0")
    assert doc["format"] == "noodl-modelica-electrical/1" and "medium" not in doc
    assert exporter.compared_variables(inst, doc) == ["V.i", "R1.i", "C1.i", "C1.v"]
    assert exporter.initial_variables(inst, doc) == {"C1": ["v"]}


def test_mbl_terminal_columns_follow_the_connector_type(exporter):
    dc = "Buildings.Electrical.DC.Interfaces.Terminal_"
    ac = "Buildings.Electrical.AC.OnePhase.Interfaces.Terminal_"
    inst = _cls("M", [
        _comp("conv", _cls("Buildings.Electrical.AC.OnePhase.Conversion.ACDCConverter",
                           [_conn("terminal_n", ac + "n"), _conn("terminal_p", dc + "p")])),
        _comp("line", _cls("Buildings.Electrical.AC.OnePhase.Lines.Line",
                           [_conn("terminal_n", ac + "n"), _conn("terminal_p", ac + "p")])),
    ], connections=[("conv.terminal_n", "line.terminal_p")])
    doc = exporter.build_circuit_document(inst, model="M", experiment={"StartTime": 0.0},
                                          experiment_declared={}, mbl_commit="x",
                                          openmodelica="omc", msl="4.1.0")
    assert exporter.compared_variables(inst, doc) == [
        "conv.terminal_n.v[1]", "conv.terminal_n.v[2]", "conv.terminal_n.i[1]",
        "conv.terminal_n.i[2]", "conv.terminal_p.v[1]", "conv.terminal_p.v[2]",
        "conv.terminal_p.i[1]",
        "line.terminal_n.v[1]", "line.terminal_n.v[2]", "line.terminal_n.i[1]",
        "line.terminal_n.i[2]", "line.terminal_p.v[1]", "line.terminal_p.v[2]",
        "line.terminal_p.i[1]", "line.terminal_p.i[2]"]
    assert exporter.derived_variables(doc) == {"line": ["line.R", "line.L", "line.C"]}


def test_output_stem_strips_the_library_prefix(exporter):
    assert exporter.stem("Modelica.Electrical.Analog.Examples.Resistor") == \
        "Analog.Examples.Resistor"
    assert exporter.stem("Buildings.Electrical.DC.Loads.Examples.Resistor") == \
        "DC.Loads.Examples.Resistor"


def test_start_time_other_than_zero_is_refused(exporter):
    with pytest.raises(exporter.ExportError, match="StartTime"):
        exporter.check_start_time({"StartTime": 1.0})
    exporter.check_start_time({"StartTime": 0.0})


def test_document_fields(exporter):
    doc = exporter.build_circuit_document(
        _rc_instance(), model="RC",
        experiment={"StartTime": 0.0, "StopTime": 1.0, "Interval": 0.01, "Tolerance": 1e-12},
        experiment_declared={"StopTime": 1.0}, mbl_commit="c", openmodelica="omc 1.27.1",
        msl="4.1.0")
    assert doc["msl"] == "4.1.0" and doc["experiment_declared"] == {"StopTime": 1.0}
    assert doc["experiment"]["Tolerance"] == 1e-12
    assert [c["name"] for c in doc["components"]] == ["G", "V", "R1", "C1", "A"]
    assert ["C1.n", "G.p"] in doc["connections"]
    r1 = next(c for c in doc["components"] if c["name"] == "R1")
    assert r1["parameters"] == {"R": 1.0, "useHeatPort": False}


def test_heat_port_temperature_and_heat_capacitor_columns(exporter):
    port = _cls("Modelica.Electrical.Analog.Interfaces.ConditionalHeatPort", [
        _param("useHeatPort", True, type_="Boolean"), _var("T_heatPort")])
    heated = _cls("Modelica.Electrical.Analog.Semiconductors.Diode",
                  [_conn("p", "Modelica.Electrical.Analog.Interfaces.PositivePin"),
                   _var("i"), _param("Ids", 1e-6)], bases=[port])
    cap = _cls("Modelica.Thermal.HeatTransfer.Components.HeatCapacitor",
               [_param("C", 2.0), _var("T")])
    cond = _cls("Modelica.Thermal.HeatTransfer.Components.ThermalConductor",
                [_param("G", 1.0), _var("Q_flow")])
    inst = _cls("H", [_comp("D", heated), _comp("HC", cap), _comp("TC", cond)])
    doc = _document(exporter, inst)
    assert exporter.compared_variables(inst, doc) == ["D.i", "D.T_heatPort", "HC.T"]
    assert exporter.initial_variables(inst, doc) == {"HC": ["T"]}


def test_rc_line_state_only_with_capacitance(exporter):
    dc = "Buildings.Electrical.DC.Interfaces.Terminal_"

    def line(use_c):
        return _cls("Buildings.Electrical.DC.Lines.TwoPortRCLine", [
            _conn("terminal_n", dc + "n"), _conn("terminal_p", dc + "p"),
            _param("use_C", use_c, type_="Boolean"), _var("Vc")])

    inst = _cls("L", [_comp("a", line(True)), _comp("b", line(False))])
    doc = _document(exporter, inst)
    cols = exporter.compared_variables(inst, doc)
    assert cols[:7] == ["a.terminal_n.v[1]", "a.terminal_n.v[2]", "a.terminal_n.i[1]",
                        "a.terminal_p.v[1]", "a.terminal_p.v[2]", "a.terminal_p.i[1]", "a.Vc"]
    assert "b.Vc" not in cols and len(cols) == 13
    assert exporter.initial_variables(inst, doc) == {"a": ["Vc"]}


def test_recorded_values_are_written_into_the_components(exporter):
    inst = _rc_instance()
    doc = _document(exporter, inst, model="RC")
    exporter.record_values(doc, {"C1": ["v"]}, {}, {"C1.v": 0.5})
    c1 = next(c for c in doc["components"] if c["name"] == "C1")
    assert c1["initial"] == {"v": 0.5}
    doc2 = {"components": [{"name": "line_1",
                            "class": "Buildings.Electrical.AC.ThreePhasesBalanced.Lines.Line"}]}
    values = {"line_1.line.R": 0.724, "line_1.line.L": 1e-3, "line_1.line.C": 0.0}
    exporter.record_values(doc2, {}, exporter.derived_variables(doc2), values)
    assert doc2["components"][0]["derived"] == {"line.R": 0.724, "line.L": 1e-3,
                                                "line.C": 0.0}


def test_missing_result_variables_are_named(exporter):
    with pytest.raises(exporter.ExportError, match=r"C1\.v.*G\.x"):
        exporter.check_result_variables(["C1.v", "R1.i", "G.x"], ["R1.i", "time"])
    exporter.check_result_variables(["R1.i"], ["R1.i"])


def test_experiment_as_simulated(exporter):
    model_experiment = {"values": {"StartTime": 0.0, "StopTime": 60.0, "Interval": 0.12,
                                   "Tolerance": 1e-6},
                        "numberOfIntervals": 500,
                        "declared": ["StopTime", "Tolerance", "__Dymola_Algorithm"]}
    sim, n, declared = exporter.simulated_experiment(model_experiment, 1e-12, 600)
    assert n == 600
    assert sim == {"StartTime": 0.0, "StopTime": 60.0, "Interval": 0.1, "Tolerance": 1e-12}
    assert declared == {"StopTime": 60.0, "Tolerance": 1e-6}
    sim, n, _ = exporter.simulated_experiment(model_experiment, 1e-12, None)
    assert n == 500 and sim["Interval"] == 0.12


def test_class_sets(exporter):
    assert exporter.FORMAT == "noodl-modelica-electrical/1"
    assert exporter.GROUNDS == {"Modelica.Electrical.Analog.Basic.Ground"}
    assert exporter.STATES["Buildings.Electrical.DC.Lines.TwoPortRCLine"] == ("Vc",)
    assert exporter.LINES == {"Buildings.Electrical.AC.OnePhase.Lines.Line",
                              "Buildings.Electrical.AC.ThreePhasesBalanced.Lines.Line"}


def test_unevaluated_flags_are_refused(exporter):
    expr = {"$kind": "cref", "parts": [{"name": "x"}]}
    inst = _cls("U", [_comp("R1", _one_port(
        "Modelica.Electrical.Analog.Basic.Resistor", _param("R", 1.0),
        _param("useHeatPort", binding=expr, type_="Boolean")))])
    with pytest.raises(exporter.ExportError, match=r"R1\.useHeatPort"):
        _document(exporter, inst)


def test_csv_header_keys_and_formatting(exporter):
    header = exporter.csv_header(
        model="Modelica.Electrical.Analog.Examples.Resistor", mbl_commit="c",
        openmodelica="OpenModelica 1.27.1", msl="4.1.0",
        experiment={"StartTime": 0, "StopTime": 50, "Interval": 50 / 1003,
                    "Tolerance": 1e-12})
    assert header == {
        "model": "Modelica.Electrical.Analog.Examples.Resistor", "mbl_commit": "c",
        "openmodelica": "OpenModelica 1.27.1", "msl": "4.1.0", "tolerance": "1e-12",
        "start_time": "0.0", "stop_time": "50.0", "interval": repr(50 / 1003),
        "solver": "dassl"}


def test_main_exit_code_follows_the_csv(exporter, monkeypatch):
    model = "Modelica.Electrical.Analog.Examples.Resistor"
    monkeypatch.setattr(exporter, "export", lambda model, **_kw: {"model": model})
    assert exporter.main([model]) == 1
    assert exporter.main([model, "--no-simulate"]) == 0
    monkeypatch.setattr(exporter, "export", lambda model, **_kw: {"model": model, "csv": "x"})
    assert exporter.main([model]) == 0


def test_reader_and_exporter_agree_on_class_sets(exporter):
    from noodl.apps.electrical.modelica import schema

    assert schema.FORMAT == exporter.FORMAT
    assert schema.GROUNDS == exporter.GROUNDS
    assert schema.WIRES == exporter.WIRES
    assert schema.STATES == exporter.STATES
    assert schema.LINES == exporter.LINES


def test_every_fixture_loads():
    from noodl.apps.electrical.modelica import schema

    data = Path(__file__).resolve().parents[3] / "data" / "modelica_electrical"
    paths = sorted(data.glob("*.json"))
    assert len(paths) == 14
    for path in paths:
        doc = schema.load(path)
        assert doc.compared and doc.components, path.name
