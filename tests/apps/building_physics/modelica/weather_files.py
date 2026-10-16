"""Hand-written TMY3 weather files and `noodl-modelica/1` documents for the weather tests.

Not a test module (no `test_` prefix): `test_weather.py` and `test_weather_reader.py` import it
as `tests.apps.building_physics.modelica.weather_files`. A `.mos` file here holds table `tab1`
with the 30 columns of MBL's TMY3 files (`Buildings/Resources/weatherdata/*.mos`); only the
columns `ReaderTMY3` reads for airflow are filled, the others are 0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

READER = "Buildings.BoundaryConditions.WeatherData.ReaderTMY3"
MBL_COMMIT = "55abf579598ca81cae0a82f337350375958e6722"
MEDIA = {
    "Buildings.Media.Air": {
        "class": "Buildings.Media.Air", "p_default": 101325.0, "T_default": 293.15,
        "X_default": [0.01, 0.99], "nXi": 1, "extraPropertiesNames": [], "dStp": 1.2,
        "pStp": 101325.0,
    },
    "Modelica.Media.Air.SimpleAir": {
        "class": "Modelica.Media.Air.SimpleAir", "p_default": 101325.0, "T_default": 293.15,
        "X_default": [1.0], "nXi": 0, "extraPropertiesNames": [],
    },
}


def tmy3_row(t: float, T: float = 10.0, rh: float = 50.0, p: float = 101000.0,
             wd: float = 270.0, ws: float = 4.0) -> list[float]:
    """One row: column 1 time (s), 2 dry bulb (degC), 4 relative humidity (%), 5 pressure (Pa),
    16 wind direction (deg, N = 0, E = 90), 17 wind speed (m/s); the others 0."""
    row = [0.0] * 30
    row[0], row[1], row[3], row[4], row[15], row[16] = t, T, rh, p, wd, ws
    return row


def write_mos(path: Path, rows) -> Path:
    lines = ["#1", f"double tab1({len(rows)},30)",
             "#C1 Time in seconds. Beginning of a year is 0s."]
    lines += ["\t".join(repr(float(v)) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", newline="\n")
    return path


def reader(parameters: dict | None = None) -> dict:
    """The component `weaDat`, a ReaderTMY3 reading `w.mos`."""
    return {"name": "weaDat", "class": READER,
            "parameters": {"filNam": "modelica://Buildings/Resources/weatherdata/w.mos",
                           **(parameters or {})}}


def document(components, connections=(), signals=(), *, medium: str = "Buildings.Media.Air",
             experiment: dict | None = None) -> dict:
    return {
        "format": "noodl-modelica/1", "model": "test.Weather", "mbl_commit": MBL_COMMIT,
        "openmodelica": "OpenModelica 1.27.1~2-g6db4671",
        "experiment": experiment or {"StartTime": 0.0, "StopTime": 10800.0, "Interval": 600.0,
                                     "Tolerance": 1e-6},
        "medium": dict(MEDIA[medium]), "components": list(components),
        "connections": [list(c) for c in connections], "signals": list(signals),
    }


def write_model(tmp_path: Path, doc: dict, rows, *, files: bool = True) -> Path:
    """Write `w.mos` (`rows`) and the document `model.json` into `tmp_path`, recording the
    file's sha256 on every reader as `scripts/modelica_export.py` does (unless `files` is
    false); return the JSON's path."""
    mos = write_mos(tmp_path / "w.mos", rows)
    digest = hashlib.sha256(mos.read_bytes()).hexdigest()
    for comp in doc["components"]:
        if comp["class"] == READER and files:
            comp["files"] = {"filNam": {"path": "w.mos", "sha256": digest}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(doc))
    return path
