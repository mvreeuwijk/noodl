"""The committed WSIMOD demo configurations and the reference harness that runs them.

`tests/data/wsimod/{quickstart,oxford}/` are written by `scripts/regenerate_wsimod_fixtures.py`
through WSIMOD's own `Model.save` (see `tests/data/wsimod/README.md`).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

pytest.importorskip("wsimod")

from wsimod.core import constants  # noqa: E402

from tests.verification._wsimod_reference import (  # noqa: E402
    reset_wsimod_globals,
    wsimod_model,
    wsimod_reference,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "wsimod"


@pytest.mark.parametrize(("case", "n_nodes", "n_arcs"), [("quickstart", 5, 6), ("oxford", 18, 21)])
def test_the_demo_configurations_load_in_wsimod(case, n_nodes, n_arcs):
    model = wsimod_model(DATA / case)
    assert len(model.nodes) == n_nodes
    assert len(model.arcs) == n_arcs
    assert len(model.dates) == 1456


def test_the_saved_pervious_depth_is_the_physical_depth():
    config = yaml.safe_load((DATA / "oxford" / "config.yml").read_text(encoding="utf-8"))
    land = next(n for n in config["nodes"].values() if n["type_"] == "Land")
    assert land["surfaces"]["rural"]["depth"] == 0.5
    model = wsimod_model(DATA / "oxford")
    surface = next(n for n in model.nodes.values() if n.__class__.__name__ == "Land").get_surface(
        "rural"
    )
    assert surface.depth == pytest.approx(0.5 * 0.4, rel=1e-15)


@pytest.mark.parametrize("case", ["quickstart", "oxford"])
def test_the_neutral_configuration_differs_only_by_its_overrides(case):
    def load(name):
        return yaml.safe_load((DATA / case / name).read_text(encoding="utf-8"))

    neutral = load("config_neutral.yml")
    assert "overrides" in neutral
    del neutral["overrides"]
    assert neutral == load("config.yml")


def test_reset_wsimod_globals_restores_the_defaults():
    wsimod_model(DATA / "oxford")
    assert constants.FLOAT_ACCURACY == 1e-8
    reset_wsimod_globals()
    assert constants.FLOAT_ACCURACY == 1e-11
    assert constants.POLLUTANTS == [
        "do", "org-phosphorus", "phosphate", "ammonia", "solids", "bod", "cod", "ph",
        "temperature", "nitrate", "nitrite", "org-nitrogen",
    ]  # fmt: skip


def test_the_reference_records_wsimods_run():
    ref = wsimod_reference(DATA / "quickstart", days=10)
    assert ref.arc_flow.shape == (10, 6)
    assert ref.arc_quality.shape == (10, 6, len(ref.pollutants))
    assert len(ref.dates) == 10
    impervious = ref.surface_names.index(("my_land", "urban"))
    assert all(math.isnan(x) for x in ref.surface["tank_recharge"][:, impervious].tolist())
    assert not any("mass balance error" in line for line in ref.printed)
