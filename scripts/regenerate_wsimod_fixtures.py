"""Regenerates the committed WSIMOD parity fixtures under tests/data/wsimod/.

Run manually: `.venv/Scripts/python scripts/regenerate_wsimod_fixtures.py`. Never run by
CI or by the test suite (design spec section 4.4) -- the committed fixtures ARE the
oracle from pytest's point of view; this script is how they get produced in the first
place, or refreshed against a newer `wsimod` release.

Downloads WSIMOD's own demo forcing data on demand (design spec amendment A2: not
shipped in the `wsimod` PyPI wheel) rather than committing it -- only the COMPUTED
fixture outputs (topology, captured events, WSIMOD's own realised flows) belong in this
repository.

Milestone 4b Task 5 skeleton: this script downloads the one forcing file both demos
share and leaves the quickstart_demo and oxford_demo build/capture/fixture-write steps
as explicit stubs -- building either full demo model is out of Task 5's scope (Task 6
fills in quickstart, Task 7 fills in oxford). Both stubs currently raise
`NotImplementedError` rather than silently doing nothing.
"""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "wsimod"
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
    """TODO (Task 6): build quickstart_demo's model inline -- five node dicts
    (`my_sewer` a `Sewer` capacity=0.04, `my_land` a `Land` with two surfaces,
    `my_groundwater` a `Groundwater`, `my_river` a plain `Node`, `my_outlet` a `Waste`)
    and six arc dicts (`urban_drainage`, `percolation`, `runoff`, `storm_outflow`,
    `baseflow`, `catchment_outflow`), per design spec amendment A3 -- run it under
    `capture_events` (`tests/verification/_wsimod_oracle.py`), and write
    `quickstart_topology.{csv,json}`, `quickstart_requests.csv` and
    `quickstart_reference.csv` (WSIMOD's own realised flows and storage) under
    `FIXTURE_DIR`."""
    raise NotImplementedError("Task 6 fills this in")


def _build_and_capture_oxford(data_folder: str) -> None:
    """TODO (Task 7): call `wsimod.demo.create_oxford.create_oxford_model(data_folder)`
    (18 nodes, 21 arcs, design spec amendment A3), run it under `capture_events`
    (`tests/verification/_wsimod_oracle.py`), and write `oxford_topology.{csv,json}`,
    `oxford_requests.csv` and `oxford_reference.csv` under `FIXTURE_DIR`."""
    raise NotImplementedError("Task 7 fills this in")


if __name__ == "__main__":
    data_folder = _download_data_folder()
    print(f"Downloaded WSIMOD demo data to {data_folder}")
    _build_and_capture_quickstart(data_folder)
    _build_and_capture_oxford(data_folder)
