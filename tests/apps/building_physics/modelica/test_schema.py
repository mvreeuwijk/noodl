"""Tests for `noodl.apps.building_physics.modelica.schema` (spec section 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noodl.apps.building_physics.modelica.schema import (
    FORMAT,
    ModelicaImportError,
    load,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_parses_a_well_formed_document() -> None:
    doc = load(FIXTURES / "two_zones_orifice.json")

    assert doc.model == "test.TwoZonesOrifice"
    assert doc.mbl_commit == "55abf579598ca81cae0a82f337350375958e6722"
    assert doc.openmodelica == "OpenModelica 1.23.0"
    assert doc.experiment == {"StartTime": 0, "StopTime": 500, "Interval": 1, "Tolerance": 1e-6}
    assert doc.medium["class"] == "Buildings.Media.Specialized.Air.PerfectGas"
    assert [c.name for c in doc.components] == ["bouA", "bouB", "ori"]
    assert doc.components[2].cls == "Buildings.Airflow.Multizone.Orifice"
    assert doc.components[2].parameters["A"] == 0.01
    assert doc.components[0].role is None
    assert doc.connections == (("bouA.ports[1]", "ori.port_a"), ("ori.port_b", "bouB.ports[1]"))
    assert doc.signals == ()


def test_load_accepts_a_path_string() -> None:
    doc = load(str(FIXTURES / "two_zones_orifice.json"))
    assert doc.model == "test.TwoZonesOrifice"


def _write(tmp_path: Path, obj: object) -> Path:
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(obj))
    return path


def _base_doc(**overrides: object) -> dict:
    doc = {
        "format": FORMAT,
        "model": "test.Minimal",
        "mbl_commit": "55abf579598ca81cae0a82f337350375958e6722",
        "openmodelica": "OpenModelica 1.23.0",
        "experiment": {},
        "medium": {},
        "components": [],
        "connections": [],
        "signals": [],
    }
    doc.update(overrides)
    return doc


def test_not_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text("{not json")
    with pytest.raises(ModelicaImportError, match="not valid JSON"):
        load(path)


def test_top_level_must_be_an_object(tmp_path: Path) -> None:
    path = _write(tmp_path, [1, 2, 3])
    with pytest.raises(ModelicaImportError, match="must be an object"):
        load(path)


def test_wrong_format_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc(format="something-else/1"))
    with pytest.raises(ModelicaImportError, match="format is 'something-else/1'"):
        load(path)


def test_missing_format_raises(tmp_path: Path) -> None:
    doc = _base_doc()
    del doc["format"]
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match="format is None"):
        load(path)


@pytest.mark.parametrize("field", ["model", "mbl_commit", "openmodelica"])
def test_missing_required_string_field_raises(tmp_path: Path, field: str) -> None:
    doc = _base_doc()
    del doc[field]
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match=field):
        load(path)


def test_components_must_be_an_array(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc(components={"not": "a list"}))
    with pytest.raises(ModelicaImportError, match="'components' must be an array"):
        load(path)


def test_component_missing_name_raises(tmp_path: Path) -> None:
    doc = _base_doc(components=[{"class": "Buildings.Fluid.Sources.Boundary_pT"}])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match=r"components\[0\]\.name"):
        load(path)


def test_component_missing_class_raises(tmp_path: Path) -> None:
    doc = _base_doc(components=[{"name": "bouA"}])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match=r"components\[0\] \(bouA\)\.class"):
        load(path)


def test_component_parameters_must_be_an_object(tmp_path: Path) -> None:
    doc = _base_doc(components=[
        {"name": "bouA", "class": "Buildings.Fluid.Sources.Boundary_pT", "parameters": [1, 2]}
    ])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match="parameters must be an object"):
        load(path)


def test_component_role_must_be_a_string(tmp_path: Path) -> None:
    doc = _base_doc(components=[
        {"name": "bouA", "class": "Buildings.Fluid.Sources.Boundary_pT", "role": 3}
    ])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match="role must be a string"):
        load(path)


def test_component_role_defaults_to_none_and_parameters_default_to_empty(
    tmp_path: Path,
) -> None:
    doc = _base_doc(components=[{"name": "bouA", "class": "Buildings.Fluid.Sources.Boundary_pT"}])
    path = _write(tmp_path, doc)
    parsed = load(path)
    assert parsed.components[0].role is None
    assert parsed.components[0].parameters == {}


def test_duplicate_component_names_raise(tmp_path: Path) -> None:
    doc = _base_doc(components=[
        {"name": "bouA", "class": "Buildings.Fluid.Sources.Boundary_pT"},
        {"name": "bouA", "class": "Buildings.Fluid.Sources.Outside"},
    ])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match="duplicate component name"):
        load(path)


def test_connections_must_be_an_array(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc(connections={"a": 1}))
    with pytest.raises(ModelicaImportError, match="'connections' must be an array"):
        load(path)


def test_connection_must_have_two_elements(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc(connections=[["a.port_a"]]))
    with pytest.raises(ModelicaImportError, match=r"connections\[0\] must be a 2-element array"):
        load(path)


def test_signals_must_be_an_array(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc(signals="nope"))
    with pytest.raises(ModelicaImportError, match="'signals' must be an array"):
        load(path)


def test_signal_missing_drives_raises(tmp_path: Path) -> None:
    doc = _base_doc(signals=[
        {"name": "ramp", "class": "Modelica.Blocks.Sources.Ramp", "parameters": {}}
    ])
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match=r"signals\[0\] \(ramp\)\.drives"):
        load(path)


def test_signal_is_parsed(tmp_path: Path) -> None:
    doc = _base_doc(signals=[
        {
            "name": "ramp", "class": "Modelica.Blocks.Sources.Ramp",
            "parameters": {"height": 100, "duration": 500}, "drives": "bouB.p_in",
        }
    ])
    path = _write(tmp_path, doc)
    parsed = load(path)
    assert parsed.signals[0].drives == "bouB.p_in"
    assert parsed.signals[0].parameters == {"height": 100, "duration": 500}


def test_refusal_reason_for_known_and_unknown_classes() -> None:
    from noodl.apps.building_physics.modelica import schema

    assert schema.refusal_reason("Buildings.Fluid.Sources.Outside_CpLowRise") == (
        "wind pressure is not supported"
    )
    assert schema.refusal_reason(
        "Buildings.BoundaryConditions.WeatherData.ReaderTMY3"
    ) == "weather data is not supported"
    assert schema.refusal_reason("Buildings.Airflow.Multizone.Orifice") is None
