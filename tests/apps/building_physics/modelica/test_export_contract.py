"""Contract test for `scripts/modelica_export.py` (plan Task 7).

Runs without OpenModelica: it feeds the exporter's pure-Python writer a synthetic
`getModelInstance` tree (the JSON OpenModelica's scripting API returns, reduced to the fields
the exporter reads), writes the `noodl-modelica/1` document, and checks that the reader's own
`schema.load` and `graph.build` accept it. The script is imported by path, so it stays out of
the `noodl` package (spec section 3: the package never imports it).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from noodl.apps.building_physics.modelica import schema
from noodl.apps.building_physics.modelica.graph import build
from noodl.apps.building_physics.modelica.schema import load

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "modelica_export.py"


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("modelica_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelica_export"] = module
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


def _cls(name, elements, restriction="model", bases=()):
    ext = [{"$kind": "extends", "baseClass": b} for b in bases]
    return {"name": name, "restriction": restriction, "elements": ext + list(elements)}


def _comp(name, type_):
    return {"$kind": "component", "name": name, "type": type_,
            "modifiers": {"Medium": {"$value": {"$kind": "class", "name": "Medium",
                                                "baseClass": "Medium"}}}}


def _cref(ref):
    parts = []
    for p in ref.split("."):
        if "[" in p:
            n, idx = p[:-1].split("[")
            parts.append({"name": n, "subscripts": [int(idx)]})
        else:
            parts.append({"name": p})
    return {"$kind": "cref", "parts": parts}


def _connect(a, b):
    return {"lhs": _cref(a), "rhs": _cref(b)}


_ENUM_FIXED = {"$kind": "enum", "name": "Modelica.Fluid.Types.Dynamics.FixedInitial",
               "index": 2}
_OUT = _var("y", "Modelica.Blocks.Interfaces.RealOutput")

_PARTIAL_VOLUME = _cls("Buildings.Fluid.MixingVolumes.BaseClasses.PartialMixingVolume", [
    _param("energyDynamics", binding=_ENUM_FIXED),
    _param("T_start", value=298.15,
           binding={"$kind": "binary_op", "lhs": 273.15, "op": "+", "rhs": 25}),
    _param("p_start", value=101325, binding={"$kind": "cref", "parts": [{"name": "p"}]}),
    _param("rho_start", binding={"$kind": "call", "name": "Medium.density"}),  # unevaluated
    _param("secret", binding=1.0, public=False),  # protected: not exported
    _var("T"), _var("p"), _var("Xi"), _var("C"),
])
VOLUME = _cls("Buildings.Fluid.MixingVolumes.MixingVolume", [
    _param("V", value=62.5, binding={"$kind": "binary_op", "lhs": 2.5, "op": "*", "rhs": 25}),
    _param("nPorts", binding=2, type_="Integer"),
    _param("m_flow_nominal", binding=0.001),
], bases=[_PARTIAL_VOLUME])
BOUNDARY = _cls("Buildings.Fluid.Sources.Boundary_pT", [
    _param("use_p_in", binding=True, type_="Boolean"), _param("p", value=101325),
    _param("T", binding=293.15), _param("nPorts", binding=1, type_="Integer"),
    _var("p_in", "Modelica.Blocks.Interfaces.RealInput"),
])
ORIFICE = _cls("Buildings.Airflow.Multizone.Orifice", [
    _param("A", binding=0.01), _param("CD", binding=0.65), _param("m", binding=0.5),
    _param("dp_turbulent", binding=0.1), _var("m_flow"),
])
COLUMN = _cls("Buildings.Airflow.Multizone.MediumColumn", [
    _param("h", binding=1.5),
    _param("densitySelection", binding={
        "$kind": "enum", "name": "Buildings.Airflow.Multizone.Types.densitySelection.fromTop",
        "index": 1}),
])
DOOR_BASE = _cls("Buildings.Fluid.Interfaces.PartialFourPortInterface",
                 [_var("m1_flow"), _var("m2_flow")])
DOOR = _cls("Buildings.Airflow.Multizone.DoorOperable", [
    _param("LClo", value=0.002, binding={"$kind": "binary_op"}), _param("wOpe", binding=1),
    _param("hOpe", binding=2.2), _var("y", "Modelica.Blocks.Interfaces.RealInput"),
], bases=[DOOR_BASE])
FIXED_T = _cls("Buildings.HeatTransfer.Sources.FixedTemperature", [_param("T", binding=298.15)])
CONDUCTOR = _cls("Modelica.Thermal.HeatTransfer.Components.ThermalConductor",
                 [_param("G", value=1e9, binding=1e9)])
RAMP = _cls("Modelica.Blocks.Sources.Ramp", [
    _param("height", binding=10), _param("duration", binding=30),
    _param("offset", binding=101325), _param("startTime", binding=10), _OUT,
], restriction="block")
CONSTANT = _cls("Modelica.Blocks.Sources.Constant", [_param("k", binding=1), _OUT],
                restriction="block")
GAIN = _cls("Modelica.Blocks.Math.Gain", [
    _param("k", binding=2), _var("u", "Modelica.Blocks.Interfaces.RealInput"), _OUT,
], restriction="block")
REAL = "Modelica.Units.SI.Pressure"  # a plain variable of the model, not a component

BASE_MODEL = _cls("test.Base", [
    _comp("volA", VOLUME), _comp("volB", VOLUME), _comp("bouB", BOUNDARY),
    _comp("colA", COLUMN), _comp("oriA", ORIFICE), _comp("doo", DOOR),
    _comp("TA", FIXED_T), _comp("conA", CONDUCTOR), _comp("ramp", RAMP),
    {"$kind": "component", "name": "dp", "type": REAL},
])
BASE_MODEL["connections"] = [
    _connect("volA.ports[1]", "colA.port_b"),
    _connect("colA.port_a", "oriA.port_a"),
    _connect("oriA.port_b", "bouB.ports[1]"),
    _connect("volA.ports[2]", "doo.port_a1"),
    _connect("doo.port_b1", "volB.ports[1]"),
    _connect("volB.ports[2]", "doo.port_a2"),
    _connect("doo.port_b2", "volA.ports[3]"),
    _connect("TA.port", "conA.port_a"),
    _connect("conA.port_b", "volA.heatPort"),
    _connect("ramp.y", "bouB.p_in"),
]
MODEL = _cls("test.Derived", [_comp("open1", CONSTANT), _comp("gai", GAIN)],
             bases=[BASE_MODEL, _cls("Modelica.Icons.Example", [])])
MODEL["connections"] = [_connect("open1.y", "doo.y"), _connect("open1.y", "gai.u")]

MEDIUM = {"class": "Buildings.Media.Air", "extraPropertiesNames": ["CO2"],
          "p_default": 101325.0, "T_default": 293.15, "X_default": [0.01, 0.99], "nXi": 1,
          "dStp": 1.2, "pStp": 101325.0}
EXPERIMENT = {"StartTime": 0.0, "StopTime": 60.0, "Interval": 1.0, "Tolerance": 1e-6}


def _document(exporter):
    return exporter.build_document(
        MODEL, model="test.Derived", medium=MEDIUM, experiment=EXPERIMENT,
        mbl_commit="55abf579598ca81cae0a82f337350375958e6722",
        openmodelica="OpenModelica 1.27.1~2-g6db4671",
    )


# ------------------------------------------------------------------------------- tests
def test_written_document_loads_and_builds(exporter, tmp_path: Path) -> None:
    path = tmp_path / "Derived.json"
    exporter.write_json(_document(exporter), path)

    doc = load(path)
    graph = build(doc)

    assert doc.model == "test.Derived"
    assert doc.mbl_commit == "55abf579598ca81cae0a82f337350375958e6722"
    assert doc.openmodelica.startswith("OpenModelica 1.27")
    assert doc.experiment == EXPERIMENT
    assert doc.medium["class"] == "Buildings.Media.Air"
    assert graph.zones == ("volA", "volB")
    assert graph.boundaries == ("bouB",)
    assert [p.element.name for p in graph.paths] == ["oriA"]
    assert [d.component.name for d in graph.doors] == ["doo"]
    assert [p.conductor.name for p in graph.pins] == ["conA"]


def test_inherited_components_and_connections_are_included(exporter) -> None:
    doc = _document(exporter)
    names = [c["name"] for c in doc["components"]]
    assert names == ["volA", "volB", "bouB", "colA", "oriA", "doo", "TA", "conA", "gai"]
    assert ["volA.ports[1]", "colA.port_b"] in doc["connections"]
    assert "dp" not in names  # a plain variable of the model is not a component


def test_parameters_are_the_evaluated_values(exporter) -> None:
    by_name = {c["name"]: c for c in _document(exporter)["components"]}
    vol = by_name["volA"]["parameters"]
    assert vol["V"] == 62.5  # OpenModelica's evaluated value, not the binding expression
    assert vol["T_start"] == 298.15  # inherited from the base class
    assert vol["energyDynamics"] == "Modelica.Fluid.Types.Dynamics.FixedInitial"
    assert vol["nPorts"] == 2
    assert "rho_start" not in vol  # OpenModelica gave no value: not guessed
    assert "secret" not in vol  # protected parameters are not exported
    assert by_name["colA"]["parameters"]["densitySelection"].endswith(".fromTop")
    assert by_name["bouB"]["parameters"]["use_p_in"] is True
    assert by_name["doo"]["class"] == "Buildings.Airflow.Multizone.DoorOperable"


def test_signals_observers_and_drives(exporter) -> None:
    doc = _document(exporter)
    signals = {s["name"]: s for s in doc["signals"]}
    assert set(signals) == {"ramp", "open1"}
    assert signals["ramp"]["drives"] == "bouB.p_in"
    assert signals["open1"]["drives"] == ["doo.y", "gai.u"]
    assert signals["ramp"]["parameters"] == {"height": 10, "duration": 30,
                                             "offset": 101325, "startTime": 10}
    gai = next(c for c in doc["components"] if c["name"] == "gai")
    assert gai["role"] == "observer"
    # Signal wiring is carried by `drives`, never by `connections`.
    assert not any(ref.split(".")[0] in signals for pair in doc["connections"] for ref in pair)


def test_compared_variables(exporter) -> None:
    variables = exporter.compared_variables(MODEL, _document(exporter))
    assert variables == [
        "oriA.m_flow", "doo.m1_flow", "doo.m2_flow",
        "volA.T", "volA.p", "volA.Xi[1]", "volA.C[1]",
        "volB.T", "volB.p", "volB.Xi[1]", "volB.C[1]",
    ]


def test_script_class_sets_match_the_reader(exporter) -> None:
    assert exporter.VOLUMES == schema.ZONES
    assert exporter.ONE_WAY == schema.ONE_WAY
    assert exporter.FOUR_PORT == schema.TWO_WAY | schema.ZONAL


def test_csv_is_exact_and_headed(exporter, tmp_path: Path) -> None:
    out = tmp_path / "Derived.csv"
    series = [[0.0, 7.199999999999999], [0.1, 1 / 3], [101325.0, 101324.99990336655]]
    rows = exporter.write_csv(out, {"model": "test.Derived", "tolerance": 1e-6},
                              ["oriA.m_flow", "volA.p"], series)
    lines = out.read_text().splitlines()
    assert rows == 2
    assert lines[:3] == ["# model: test.Derived", "# tolerance: 1e-06",
                         '"time","oriA.m_flow","volA.p"']
    assert [float(x) for x in lines[4].split(",")] == [7.199999999999999, 1 / 3,
                                                        101324.99990336655]
    with pytest.raises(exporter.ExportError, match="strictly increasing"):
        exporter.write_csv(out, {}, ["x"], [[0.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    with pytest.raises(exporter.ExportError, match="columns"):
        exporter.write_csv(out, {}, ["x", "y"], [[0.0], [1.0]])


def test_reals_are_replaced_by_the_simulation_result(exporter) -> None:
    doc = _document(exporter)
    refs = exporter.exact_value_refs(MODEL, doc)
    # Reals only (Integer nPorts, Boolean use_p_in and enumerations are exact already).
    assert refs["volA.V"] == ("volA", "V", ())
    assert refs["ramp.height"] == ("ramp", "height", ())
    assert "volA.nPorts" not in refs and "bouB.use_p_in" not in refs
    values = {r: 0.123456789012345 for r in refs if r != "doo.LClo"}
    missing = exporter.apply_exact_values(doc, refs, values)
    by_name = {c["name"]: c for c in doc["components"]}
    assert missing == ["doo.LClo"]
    assert by_name["volA"]["parameters"]["V"] == 0.123456789012345
    assert by_name["doo"]["approximate"] == ["LClo"]
    assert by_name["doo"]["parameters"]["LClo"] == 0.002  # kept, flagged


def test_array_parameters_are_indexed_like_the_result_file(exporter) -> None:
    doc = {"components": [{"name": "tab", "parameters": {"table": [[1, 2], [3, 4]]}}],
           "signals": []}
    table = _cls("Modelica.Blocks.Tables.CombiTable1Dv", [_param("table", binding=[[1, 2]])],
                 restriction="block")
    instance = _cls("test.T", [_comp("tab", table)])
    refs = exporter.exact_value_refs(instance, doc)
    assert list(refs) == ["tab.table[1,1]", "tab.table[1,2]", "tab.table[2,1]",
                          "tab.table[2,2]"]
    exporter.apply_exact_values(doc, refs, {"tab.table[2,1]": 3.5, "tab.table[1,1]": 1.0,
                                            "tab.table[1,2]": 2.0, "tab.table[2,2]": 4.0})
    assert doc["components"][0]["parameters"]["table"] == [[1.0, 2.0], [3.5, 4.0]]


def test_library_paths_are_written_relative_to_the_mbl_root(exporter) -> None:
    mbl = "/opt/mbl"
    doc = {"components": [{"parameters": {
        "filNam": "/opt/mbl/Buildings/Resources/weatherdata/x.mos",
        "other": "/elsewhere/y.mos", "n": 1.0, "names": ["/opt/mbl/Buildings/z.txt"]}}]}
    out = exporter.relative_library_paths(doc, mbl)["components"][0]["parameters"]
    assert out["filNam"] == "modelica://Buildings/Resources/weatherdata/x.mos"
    assert out["other"] == "/elsewhere/y.mos"
    assert out["n"] == 1.0
    assert out["names"] == ["modelica://Buildings/z.txt"]


def test_omc_helpers(exporter) -> None:
    assert exporter.parse_omc_matrix("{{0.0, 0.5}, {1.0, NaN}}")[0] == [0.0, 0.5]
    assert exporter.medium_reference(MODEL) == "Medium"
    assert exporter.model_classes(MODEL) == ["test.Derived", "test.Base",
                                             "Modelica.Icons.Example"]


# ------------------------------------------------------------------------------ exit code
def test_main_exits_non_zero_when_a_requested_simulation_fails(exporter, monkeypatch) -> None:
    """A requested simulation that fails still gets a JSON-only `export()` (no "csv" key in
    the summary: `export`'s own `if run_simulation and not simulated` branch, which only
    warns) -- `main` must not report that as success (final review, Minor 9). Checked without
    OpenModelica by monkeypatching `export` itself, since the failure path needs `omc`."""
    monkeypatch.setattr(
        exporter, "export",
        lambda model, out, mbl, *, run_simulation, keep: {"model": model, "json": "x.json"})
    assert exporter.main(["test.Whatever"]) == 1


def test_main_exits_zero_when_the_simulation_succeeds(exporter, monkeypatch) -> None:
    monkeypatch.setattr(
        exporter, "export",
        lambda model, out, mbl, *, run_simulation, keep: {"model": model, "json": "x.json",
                                                          "csv": "x.csv"})
    assert exporter.main(["test.Whatever"]) == 0


def test_main_exits_zero_with_no_simulate_even_without_a_csv(exporter, monkeypatch) -> None:
    """`--no-simulate` never produces a "csv" key by design; that is not a failure."""
    monkeypatch.setattr(
        exporter, "export",
        lambda model, out, mbl, *, run_simulation, keep: {"model": model, "json": "x.json"})
    assert exporter.main(["test.Whatever", "--no-simulate"]) == 0
