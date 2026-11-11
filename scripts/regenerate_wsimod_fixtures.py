"""Regenerates the committed WSIMOD fixtures under tests/data/allocation/ and tests/data/wsimod/.

`write_demo_configs` (further down) writes the two demo configurations, see
`tests/data/wsimod/README.md`.

Run manually: `.venv/Scripts/python scripts/regenerate_wsimod_fixtures.py`. Never run by
CI or by the test suite -- the committed fixtures ARE the
reference from pytest's point of view; this script is how they get produced in the first
place, or refreshed against a newer `wsimod` release.

Downloads WSIMOD's own demo forcing data on demand (it is not shipped in the `wsimod`
PyPI wheel) rather than committing it -- only the COMPUTED
fixture outputs (topology, captured events, WSIMOD's own realised flows) belong in this
repository.

What it does, in order: downloads the one forcing file both demos share into a temp
directory; builds `quickstart_demo`'s model inline (twice: as shipped, and as
`quickstart_tight` with `QUICKSTART_TIGHT`'s lowered arc capacities) and `oxford_demo`'s
via WSIMOD's own packaged `create_oxford_model`; runs each under the capture harness
(`tests/verification/_wsimod_reference.py`); and writes six fixtures --
`{quickstart,quickstart_tight,oxford}_topology.json` and `..._events.csv` -- under
`tests/data/allocation/`.

Each `_events.csv` holds one row per (arc, DIRECTION, timestep), SUMMED over the
harness's raw per-event rows. Push and pull are deliberately NOT merged into one row:
whether to sum them is the replay's decision, and `tests/verification/
test_allocation_wsimod_replay.py` does sum them (it must -- four oxford arcs see both a push and a
pull in the same timestep). Same-direction events ARE summed here, and both demos have
them: a pass-through `Node` forwards each push it receives as its own push, so e.g.
quickstart's `catchment_outflow` sees two or three pushes every timestep (10184 raw
events -> 7464 rows) and oxford seven arcs (48995 -> 33880). Summing is exact for the
per-timestep replay: WSIMOD clips each event against the arc's accumulated `flow_in`, so
the per-timestep total is `min(sum of requests, capacity)` whatever the order (see
`tests/verification/test_allocation_wsimod_capacity.py`).
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

ALLOCATION_DIR = REPO_ROOT / "tests" / "data" / "allocation"
WSIMOD_DIR = REPO_ROOT / "tests" / "data" / "wsimod"
DATA_URL = (
    "https://raw.githubusercontent.com/ImperialCollegeLondon/wsi/main/docs/demo/"
    "data/processed/timeseries_data.csv"
)

# `quickstart_demo` with three arcs' capacities lowered below their peak requests:
# `baseflow`'s own arc clip then cuts 1085 of its daily requests, and `runoff` and
# `percolation` run at capacity on 866 and 552 days (via `Land`'s check-sized
# requests, see `tests/verification/test_allocation_wsimod_capacity.py`). Chosen on arcs
# whose receivers accept everything offered (a pass-through `Node` onto a `Waste`, and
# `Groundwater` well below its capacity), so no other bound interferes;
# `tests/verification/test_allocation_wsimod_capacity.py` explains why a bottleneck on
# `catchment_outflow` would not be a fair case.
QUICKSTART_TIGHT = {"runoff": 0.03, "baseflow": 0.04, "percolation": 0.08}


def _download_data_folder() -> str:
    """Downloads timeseries_data.csv into a temp dir laid out as `create_oxford_model`
    and `quickstart_demo`'s inline build both expect (`<data_folder>/processed/
    timeseries_data.csv` -- both demos read the same file)."""
    tmp = tempfile.mkdtemp(prefix="wsimod_data_")
    processed = Path(tmp) / "processed"
    processed.mkdir()
    urllib.request.urlretrieve(DATA_URL, processed / "timeseries_data.csv")
    return tmp


def _build_quickstart_model(data_folder: str, capacities: dict | None = None):
    """Builds `quickstart_demo`'s model inline (five node dicts, six arc dicts,
    `Model.add_nodes`/`add_arcs`).

    `capacities` overrides named arcs' WSIMOD `capacity` (default: every arc at WSIMOD's
    `UNBOUNDED_CAPACITY`, i.e. the demo as shipped). `quickstart_tight` uses it to make
    arcs genuinely bind -- see `QUICKSTART_TIGHT`."""
    import pandas as pd
    from wsimod.core import constants
    from wsimod.orchestration.model import Model

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
    arcs = [urban_drainage, percolation, runoff, storm_outflow, baseflow, catchment_outflow]
    for name, capacity in (capacities or {}).items():
        (arc,) = [a for a in arcs if a["name"] == name]
        arc["capacity"] = capacity
    quickstart_model.add_arcs(arcs)
    return quickstart_model


def _build_and_capture_quickstart(
    data_folder: str, prefix: str = "quickstart", capacities: dict | None = None
) -> None:
    """Runs `_build_quickstart_model` under `capture_events`
    (`tests/verification/_wsimod_reference.py`), and writes `{prefix}_topology.json` and
    `{prefix}_events.csv` (one row per (arc, direction, timestep), aggregated from the
    harness's raw per-event rows -- see the print-out below) under `ALLOCATION_DIR`."""
    import pandas as pd

    from tests.verification._wsimod_reference import capture_events, extract_topology

    quickstart_model = _build_quickstart_model(data_folder, capacities)

    topology = extract_topology(quickstart_model)
    with capture_events(quickstart_model) as events:
        quickstart_model.run(verbose=False)

    ALLOCATION_DIR.mkdir(parents=True, exist_ok=True)
    (ALLOCATION_DIR / f"{prefix}_topology.json").write_text(
        json.dumps(topology, indent=2), encoding="utf-8", newline="\n")
    raw = pd.DataFrame(events)
    aggregated = raw.groupby(["arc", "direction", "t"], as_index=False)[
        ["requested", "realised"]
    ].sum()
    n_events = len(raw)
    n_rows = len(aggregated)
    print(
        f"{prefix}: {n_events} raw push/pull events aggregated into {n_rows} "
        f"(arc, direction, timestep) rows"
    )
    if n_events != n_rows:
        print(
            f"{prefix}: some arcs saw more than one same-direction event in a "
            "timestep; their rows are sums (harness docstring)"
        )
    aggregated.to_csv(ALLOCATION_DIR / f"{prefix}_events.csv", index=False)


def _build_and_capture_oxford(data_folder: str) -> None:
    """Builds `oxford_demo`'s model via the packaged
    `wsimod.demo.create_oxford.create_oxford_model` (18 nodes,
    21 arcs), runs it under `capture_events` (`tests/verification/_wsimod_reference.py`),
    and writes `oxford_topology.json` and `oxford_events.csv` under `ALLOCATION_DIR`."""
    import pandas as pd
    from wsimod.demo.create_oxford import create_oxford_model

    from tests.verification._wsimod_reference import capture_events, extract_topology

    oxford_model = create_oxford_model(data_folder)
    oxford_topology = extract_topology(oxford_model)
    with capture_events(oxford_model) as oxford_events_raw:
        oxford_model.run(verbose=False)

    ALLOCATION_DIR.mkdir(parents=True, exist_ok=True)
    (ALLOCATION_DIR / "oxford_topology.json").write_text(
        json.dumps(oxford_topology, indent=2), encoding="utf-8", newline="\n")
    oxford_raw = pd.DataFrame(oxford_events_raw)
    oxford_aggregated = oxford_raw.groupby(
        ["arc", "direction", "t"], as_index=False
    )[["requested", "realised"]].sum()
    n_events = len(oxford_raw)
    n_rows = len(oxford_aggregated)
    print(f"oxford: {n_events} raw events aggregated into {n_rows} rows")
    if n_events != n_rows:
        print(
            "oxford: some arcs saw more than one same-direction event in a "
            "timestep; their rows are sums (harness docstring)"
        )
    oxford_aggregated.to_csv(ALLOCATION_DIR / "oxford_events.csv", index=False)


def _first_difference(original, reloaded) -> str | None:
    """Names the first recorded series on which two `WsimodReference`s differ, or None."""
    import numpy as np

    # the reloaded arcs are in registration order (`_registration_order`): align them by name
    rows = [reloaded.arc_names.index(name) for name in original.arc_names]
    series = [("arc_flow", original.arc_flow, reloaded.arc_flow[:, rows]),
              ("arc_quality", original.arc_quality, reloaded.arc_quality[:, rows]),
              ("tank_storage", original.tank_storage, reloaded.tank_storage),
              ("tank_quality", original.tank_quality, reloaded.tank_quality),
              ("surface_quality", original.surface_quality, reloaded.surface_quality)]
    series += [(f"surface.{k}", v, reloaded.surface[k]) for k, v in original.surface.items()]
    for name, a, b in series:
        a, b = a.numpy(), b.numpy()
        if a.shape != b.shape:
            return f"{name}: shape {a.shape} != {b.shape}"
        bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if bad.any():
            index = tuple(int(i) for i in np.argwhere(bad)[0])
            day = original.dates[index[0]]
            return f"{name}{list(index)}: {a[index]!r} != {b[index]!r} (day {day})"
    return None


def _registration_order(model) -> list[str]:
    """Arc names in an order that reproduces every node's `in_arcs`/`out_arcs` order on reload.

    A node's `in_arcs`/`out_arcs` fill in the order the arcs were constructed
    (`arcs.py:70-71`), and the pushes and pulls of a node walk them in that order, so it
    changes the float summation order. `create_oxford_model` constructs its arcs in a
    different order from the list it passes to `add_arcs`, so `Model.save`, which writes
    `model.arcs` in that list's order, loses it (the reloaded oxford run differs from the
    original by ulps, then by 1e-6 on near-empty flows). The saved arcs are re-sorted:
    a topological order of the precedences every node's two arc lists give, ties broken by
    `model.arcs`' own order."""
    names = list(model.arcs)
    after: dict[str, set[str]] = {name: set() for name in names}
    for node in model.nodes.values():
        for ordered in (list(node.in_arcs), list(node.out_arcs)):
            for first, second in zip(ordered, ordered[1:], strict=False):
                after[first].add(second)
    indegree = {name: 0 for name in names}
    for successors in after.values():
        for name in successors:
            indegree[name] += 1
    order: list[str] = []
    ready = [name for name in names if indegree[name] == 0]
    while ready:
        name = ready.pop(0)
        order.append(name)
        for successor in sorted(after[name], key=names.index):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort(key=names.index)
    if len(order) != len(names):
        sys.exit("the nodes' arc lists impose a cyclic arc order; cannot reproduce it on reload")
    return order


def _corrected_config(model, config_path: Path) -> dict:
    """Corrects the saved YAML: arcs in `_registration_order`, pervious depths physical.

    `Model.save` writes `surface.depth`, the simulated depth (`depth * total_porosity`), under
    the key `depth` for each `PerviousSurface` (`model.py:419-420`), so a reload shrinks the
    depth by the porosity again (0.5 -> 0.2 written -> 0.08 reloaded). The YAML is read back
    and corrected to `surface.depth / surface.total_porosity` (spec W12).

    `Model.save` writes `surface.depth`, the simulated depth (`depth * total_porosity`), under
    the key `depth` (`model.py:419-420`), so a reload shrinks the depth by the porosity again
    (0.5 -> 0.2 written -> 0.08 reloaded). The YAML is read back and corrected to
    `surface.depth / surface.total_porosity` (spec W12)."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name, node in model.nodes.items():
        for surface in getattr(node, "surfaces", []):
            if hasattr(surface, "total_porosity"):
                saved = config["nodes"][name]["surfaces"][surface.surface]
                saved["depth"] = surface.depth / surface.total_porosity
    config["arcs"] = {name: config["arcs"][name] for name in _registration_order(model)}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return config


def _neutral_overrides(model) -> dict:
    """The `overrides` section that makes WSIMOD's water quality neutral.

    Every `WWTW`/`FWTW`: `process_parameters[p] = {constant: c_v, exponent: 1}` (c_v the
    volume constant) and `liquor_multiplier[p] = liquor_multiplier["volume"]` for every
    additive `p`. Every surface of every `Land`: `pollutant_load` zero for every key it has,
    `soil_temp_w_prev 1, soil_temp_w_air 0, soil_temp_w_deep 0` (pervious surfaces only, the
    keys `PerviousSurface.apply_overrides` accepts, `land.py:845-880`)."""
    from wsimod.core import constants

    nodes = {}
    for name, node in model.nodes.items():
        kind = type(node).__name__
        if kind in ("WWTW", "FWTW"):
            c_v = node.process_parameters["volume"]["constant"]
            volume = node.liquor_multiplier["volume"]
            nodes[name] = {
                "type_": kind,
                "name": name,
                "process_parameters": {
                    p: {"constant": c_v, "exponent": 1} for p in constants.ADDITIVE_POLLUTANTS
                },
                "liquor_multiplier": {p: volume for p in constants.ADDITIVE_POLLUTANTS},
            }
        elif kind == "Land":
            surfaces = {}
            for surface in node.surfaces:
                override = {"pollutant_load": {k: 0.0 for k in surface.pollutant_load}}
                if hasattr(surface, "total_porosity"):
                    override.update(soil_temp_w_prev=1, soil_temp_w_air=0, soil_temp_w_deep=0)
                surfaces[surface.surface] = override
            nodes[name] = {"type_": kind, "name": name, "surfaces": surfaces}
    return {"nodes": nodes}


def write_demo_configs(data_folder) -> None:
    """Writes `tests/data/wsimod/{quickstart,oxford}/` from the demo forcing in `data_folder`.

    Per demo: build the model; `Model.save(dir, "config.yml", compress=True)`; correct the
    saved config (`_corrected_config`); write `config_neutral.yml` (the same
    mapping plus `_neutral_overrides`); reload `config.yml` with `wsimod_model`, run both it
    and the original model, and exit non-zero naming the first difference if they disagree."""
    import yaml
    from wsimod.demo.create_oxford import create_oxford_model

    from tests.verification._wsimod_reference import (
        record_wsimod_run,
        reset_wsimod_globals,
        wsimod_model,
    )

    for case in ("quickstart", "oxford"):
        folder = WSIMOD_DIR / case
        folder.mkdir(parents=True, exist_ok=True)
        reset_wsimod_globals()
        model = (
            _build_quickstart_model(data_folder)
            if case == "quickstart"
            else create_oxford_model(data_folder)
        )
        model.save(str(folder), "config.yml", compress=True)
        config = _corrected_config(model, folder / "config.yml")
        neutral = dict(config, overrides=_neutral_overrides(model))
        (folder / "config_neutral.yml").write_text(
            yaml.safe_dump(neutral, sort_keys=False), encoding="utf-8", newline="\n"
        )
        original = record_wsimod_run(model)
        reloaded = record_wsimod_run(wsimod_model(folder))
        difference = _first_difference(original, reloaded)
        if difference is not None:
            sys.exit(f"{case}: the reloaded config.yml run differs from the original: {difference}")
        print(f"{case}: reloaded config.yml reproduces the original run "
              f"({len(original.dates)} days)")


if __name__ == "__main__":
    data_folder = _download_data_folder()
    print(f"Downloaded WSIMOD demo data to {data_folder}")
    _build_and_capture_quickstart(data_folder)
    _build_and_capture_quickstart(data_folder, "quickstart_tight", QUICKSTART_TIGHT)
    _build_and_capture_oxford(data_folder)
    write_demo_configs(data_folder)
