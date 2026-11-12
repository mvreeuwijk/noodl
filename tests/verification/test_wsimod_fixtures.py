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
def test_the_no_deposition_configuration_differs_only_by_its_overrides(case):
    def load(name):
        return yaml.safe_load((DATA / case / name).read_text(encoding="utf-8"))

    config = load("config.yml")
    no_deposition = load("config_no_deposition.yml")
    overrides = no_deposition.pop("overrides")
    assert no_deposition == config
    lands = {n: node for n, node in config["nodes"].items() if node["type_"] == "Land"}
    assert set(overrides["nodes"]) == set(lands)
    for name, land in lands.items():
        entry = overrides["nodes"][name]
        assert entry == {
            "type_": "Land",
            "name": name,
            "surfaces": {
                s: {"pollutant_load": {k: 0.0 for k in surface["pollutant_load"]}}
                for s, surface in land["surfaces"].items()
            },
        }
    model = wsimod_model(DATA / case, config_name="config_no_deposition.yml")
    for node in model.nodes.values():
        if type(node).__name__ == "Land":
            for surface in node.surfaces:
                assert all(v == 0 for v in surface.pollutant_load.values())


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


@pytest.mark.parametrize("case", ["quickstart", "oxford"])
def test_the_saved_arc_order_gives_every_node_its_arcs_in_file_order(case):
    """A node's arc lists fill in the file's arc order (`arcs.py:70-71`): a hand edit of the
    arc order would change WSIMOD's summation order (README, "Two corrections")."""
    config = yaml.safe_load((DATA / case / "config.yml").read_text(encoding="utf-8"))
    position = {name: i for i, name in enumerate(config["arcs"])}
    model = wsimod_model(DATA / case)
    for node in model.nodes.values():
        for arcs in (list(node.in_arcs), list(node.out_arcs)):
            assert [position[a] for a in arcs] == sorted(position[a] for a in arcs)
