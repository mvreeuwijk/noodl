"""Regenerates the committed WSIMOD parity fixtures under tests/data/wsimod/.

Run manually: `.venv/Scripts/python scripts/regenerate_wsimod_fixtures.py`. Never run by
CI or by the test suite (design spec section 4.4) -- the committed fixtures ARE the
reference from pytest's point of view; this script is how they get produced in the first
place, or refreshed against a newer `wsimod` release.

Downloads WSIMOD's own demo forcing data on demand (design spec amendment A2: not
shipped in the `wsimod` PyPI wheel) rather than committing it -- only the COMPUTED
fixture outputs (topology, captured events, WSIMOD's own realised flows) belong in this
repository.

What it does, in order: downloads the one forcing file both demos share into a temp
directory; builds `quickstart_demo`'s model inline (design spec amendment A3) and
`oxford_demo`'s via WSIMOD's own packaged `create_oxford_model`; runs each under the
capture harness (`tests/verification/_wsimod_reference.py`); and writes four fixtures --
`{quickstart,oxford}_topology.json` and `{quickstart,oxford}_events.csv` -- under
`tests/data/wsimod/`. Both demos are fully implemented here (they were stubs raising
`NotImplementedError` while milestone 4b's Task 5 only built the harness; Tasks 6 and 7
filled them in).

Each `_events.csv` holds one row per (arc, DIRECTION, timestep), aggregated from the
harness's raw per-event rows. Push and pull are deliberately NOT merged into one row:
whether to sum them is the replay's decision, and `tests/verification/
test_wsimod_parity.py` does sum them (it must -- four oxford arcs see both a push and a
pull in the same timestep). Each build prints its raw-event and aggregated-row counts
and warns when they differ, i.e. when some arc saw two events of the SAME direction in
one timestep; neither demo currently does (quickstart 7464/7464, oxford 33880/33880),
which is what makes the aggregation here a pure relabelling rather than a summation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Run directly as `python scripts/regenerate_wsimod_fixtures.py` (this file's own
    # docstring above), sys.path[0] is `scripts/`, not the repo root -- pytest's own
    # `pythonpath = ["."]` (pyproject.toml) doesn't apply outside pytest, so the
    # `tests.verification._wsimod_reference` import below needs this inserted explicitly.
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "wsimod"
DATA_URL = (
    "https://raw.githubusercontent.com/ImperialCollegeLondon/wsi/main/docs/demo/"
    "data/processed/timeseries_data.csv"
)


def _download_data_folder() -> str:
    """Downloads timeseries_data.csv into a temp dir laid out as `create_oxford_model`
    and `quickstart_demo`'s inline build both expect (`<data_folder>/processed/
    timeseries_data.csv`, design spec amendments A2 and A3 -- both demos read the same
    file)."""
    tmp = tempfile.mkdtemp(prefix="wsimod_data_")
    processed = Path(tmp) / "processed"
    processed.mkdir()
    urllib.request.urlretrieve(DATA_URL, processed / "timeseries_data.csv")
    return tmp


def _build_and_capture_quickstart(data_folder: str) -> None:
    """Builds `quickstart_demo`'s model inline (design spec amendment A3: five node
    dicts, six arc dicts, `Model.add_nodes`/`add_arcs`), runs it under `capture_events`
    (`tests/verification/_wsimod_reference.py`), and writes `quickstart_topology.json` and
    `quickstart_events.csv` (one row per (arc, direction, timestep), aggregated from the
    harness's raw per-event rows -- see the print-out below) under `FIXTURE_DIR`."""
    import pandas as pd
    from wsimod.core import constants
    from wsimod.orchestration.model import Model

    from tests.verification._wsimod_reference import capture_events, extract_topology

    input_fid = os.path.join(data_folder, "processed", "timeseries_data.csv")
    input_data = pd.read_csv(input_fid)
    input_data.loc[input_data.variable == "precipitation", "value"] *= constants.MM_TO_M
    input_data.date = pd.to_datetime(input_data.date)
    input_data = input_data.loc[input_data.site == "oxford_land"]
    dates = input_data.date.drop_duplicates()
    land_inputs = input_data.set_index(["variable", "date"]).value.to_dict()

    sewer = {"type_": "Sewer", "capacity": 0.04, "name": "my_sewer"}
    surface1 = {
        "type_": "ImperviousSurface",
        "surface": "urban",
        "area": 10,
        "pollutant_load": {"phosphate": 1e-7},
    }
    surface2 = {
        "type_": "PerviousSurface",
        "surface": "rural",
        "area": 100,
        "depth": 0.5,
        "pollutant_load": {"phosphate": 1e-7},
    }
    land = {
        "type_": "Land",
        "data_input_dict": land_inputs,
        "surfaces": [surface1, surface2],
        "name": "my_land",
    }
    gw = {"type_": "Groundwater", "area": 100, "capacity": 100, "name": "my_groundwater"}
    node = {"type_": "Node", "name": "my_river"}
    waste = {"type_": "Waste", "name": "my_outlet"}

    urban_drainage = {
        "type_": "Arc", "in_port": "my_land", "out_port": "my_sewer", "name": "urban_drainage"
    }
    percolation = {
        "type_": "Arc", "in_port": "my_land", "out_port": "my_groundwater", "name": "percolation"
    }
    runoff = {"type_": "Arc", "in_port": "my_land", "out_port": "my_river", "name": "runoff"}
    storm_outflow = {
        "type_": "Arc", "in_port": "my_sewer", "out_port": "my_river", "name": "storm_outflow"
    }
    baseflow = {
        "type_": "Arc", "in_port": "my_groundwater", "out_port": "my_river", "name": "baseflow"
    }
    catchment_outflow = {
        "type_": "Arc", "in_port": "my_river", "out_port": "my_outlet", "name": "catchment_outflow"
    }

    quickstart_model = Model()
    quickstart_model.dates = dates
    quickstart_model.add_nodes([sewer, land, gw, node, waste])
    quickstart_model.add_arcs(
        [urban_drainage, percolation, runoff, storm_outflow, baseflow, catchment_outflow]
    )

    topology = extract_topology(quickstart_model)
    with capture_events(quickstart_model) as events:
        quickstart_model.run(verbose=False)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "quickstart_topology.json").write_text(json.dumps(topology, indent=2))
    raw = pd.DataFrame(events)
    aggregated = raw.groupby(["arc", "direction", "t"], as_index=False)[
        ["requested", "realised"]
    ].sum()
    n_events = len(raw)
    n_rows = len(aggregated)
    print(
        f"quickstart: {n_events} raw push/pull events aggregated into {n_rows} "
        f"(arc, direction, timestep) rows"
    )
    if n_events != n_rows:
        print(
            "quickstart: at least one arc saw more than one event in a single "
            "timestep -- inspect before trusting the aggregation (harness docstring)"
        )
    aggregated.to_csv(FIXTURE_DIR / "quickstart_events.csv", index=False)


def _build_and_capture_oxford(data_folder: str) -> None:
    """Builds `oxford_demo`'s model via the packaged
    `wsimod.demo.create_oxford.create_oxford_model` (design spec amendment A3: 18 nodes,
    21 arcs), runs it under `capture_events` (`tests/verification/_wsimod_reference.py`),
    and writes `oxford_topology.json` and `oxford_events.csv` under `FIXTURE_DIR`."""
    import pandas as pd
    from wsimod.demo.create_oxford import create_oxford_model

    from tests.verification._wsimod_reference import capture_events, extract_topology

    oxford_model = create_oxford_model(data_folder)
    oxford_topology = extract_topology(oxford_model)
    with capture_events(oxford_model) as oxford_events_raw:
        oxford_model.run(verbose=False)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "oxford_topology.json").write_text(json.dumps(oxford_topology, indent=2))
    oxford_raw = pd.DataFrame(oxford_events_raw)
    oxford_aggregated = oxford_raw.groupby(
        ["arc", "direction", "t"], as_index=False
    )[["requested", "realised"]].sum()
    n_events = len(oxford_raw)
    n_rows = len(oxford_aggregated)
    print(f"oxford: {n_events} raw events aggregated into {n_rows} rows")
    if n_events != n_rows:
        print(
            "oxford: at least one arc saw more than one event in a single "
            "timestep -- inspect before trusting the aggregation (harness docstring)"
        )
    oxford_aggregated.to_csv(FIXTURE_DIR / "oxford_events.csv", index=False)


if __name__ == "__main__":
    data_folder = _download_data_folder()
    print(f"Downloaded WSIMOD demo data to {data_folder}")
    _build_and_capture_quickstart(data_folder)
    _build_and_capture_oxford(data_folder)
