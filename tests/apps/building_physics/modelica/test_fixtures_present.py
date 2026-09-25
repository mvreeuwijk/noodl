"""The 23 OpenModelica exports of the MBL multizone models are present and well-formed.

Fast: no simulation, so this runs on every test invocation. It only checks that the fixtures
exist, record their provenance (the MBL commit and an OpenModelica version) and parse with
`schema.load`; exercising `read_modelica`/`run.simulate` (18 models) and the
`ModelicaImportError` refusal (5 models) against them is Task 8's own verification step, not
repeated here as a permanent test.
"""

from __future__ import annotations

from pathlib import Path

from noodl.apps.building_physics.modelica.schema import load

DATA = Path(__file__).resolve().parents[4] / "tests" / "data" / "modelica"

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
