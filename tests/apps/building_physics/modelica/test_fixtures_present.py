"""The 23 OpenModelica exports of the MBL multizone models and the 5 of MBL's
`Buildings.Fluid.Sources` wind-boundary examples are present and well-formed, with the weather
file they read.

Fast: no simulation, so this runs on every test invocation. It only checks that the fixtures
exist, record their provenance (the MBL commit and an OpenModelica version) and parse with
`schema.load`; exercising `read_modelica`/`run.simulate` (20 models), the
`ModelicaImportError` refusal (4 models) and the weather and wind pressure alone (4 models)
against them is `tests/verification/test_modelica_parity.py`'s job, not repeated here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from noodl.apps.building_physics.modelica.schema import load

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "tests" / "data" / "modelica"

VALIDATION = [
    "DoorOpenClosed",
    "OneWayFlow",
    "OpenDoorBuoyancyDynamic",
    "OpenDoorBuoyancyPressureDynamic",
    "OpenDoorPressure",
    "OpenDoorTemperature",
    "ThreeRoomsContam",
    "ThreeRoomsContamDiscretizedDoor",
]
EXAMPLES = [
    "CO2TransportStep",
    "ChimneyShaftNoVolume",
    "ChimneyShaftWithVolume",
    "ClosedDoors",
    "NaturalVentilation",
    "OneEffectiveAirLeakageArea",
    "OneOpenDoor",
    "OneRoom",
    "Orifice",
    "PowerLaw",
    "PressurizationData",
    "ReverseBuoyancy",
    "ReverseBuoyancy3Zones",
    "TrickleVent",
    "ZonalFlow",
]
MODELS = VALIDATION + EXAMPLES


def test_all_23_json_fixtures_are_present() -> None:
    assert len(MODELS) == 23
    missing = [m for m in MODELS if not (DATA / f"{m}.json").is_file()]
    assert missing == [], f"missing JSON fixtures: {missing}"


def test_each_fixture_records_its_provenance() -> None:
    for name in MODELS:
        doc = load(DATA / f"{name}.json")
        assert doc.mbl_commit == "55abf579598ca81cae0a82f337350375958e6722", name
        assert doc.openmodelica.startswith("OpenModelica "), name


def test_each_fixture_loads_with_schema() -> None:
    for name in MODELS:
        doc = load(DATA / f"{name}.json")
        assert doc.model.startswith("Buildings.Airflow.Multizone."), name
        assert doc.components or doc.signals, name


SOURCES = [
    "Outside_Cp",
    "Outside_CpData",
    "Outside_CpData_Angles",
    "Outside_CpData_Directions",
    "Outside_CpLowRise",
]
WEATHER_FILE = DATA / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.mos"


def test_the_fluid_sources_fixtures_are_present_and_load() -> None:
    for name in SOURCES:
        doc = load(DATA / f"{name}.json")
        assert doc.mbl_commit == "55abf579598ca81cae0a82f337350375958e6722", name
        assert doc.model.startswith("Buildings.Fluid.Sources."), name
        assert (DATA / f"{name}.csv").is_file(), name


def test_every_weather_file_entry_names_the_committed_file() -> None:
    digest = hashlib.sha256(WEATHER_FILE.read_bytes()).hexdigest()
    entries = [c["files"]["filNam"] for f in DATA.glob("*.json")
               for c in json.loads(f.read_text()).get("components", []) if "files" in c]
    assert entries, "no fixture records a weather file"
    assert all(e == {"path": WEATHER_FILE.name, "sha256": digest} for e in entries), entries


def test_weather_files_are_kept_from_line_ending_conversion() -> None:
    """The sha256 above is of the file's bytes: git must not normalise its line endings."""
    lines = (ROOT / ".gitattributes").read_text().splitlines()
    assert "tests/data/modelica/*.mos -text" in lines
