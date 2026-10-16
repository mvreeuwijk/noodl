"""MediumColumnDynamic expanded into a volume between two half columns (MBL v13.0.0
`Airflow/Multizone/MediumColumnDynamic.mo`)."""

import dataclasses
from pathlib import Path

import pytest

from noodl.apps.building_physics.modelica import expand, graph, schema
from noodl.apps.building_physics.modelica.schema import ModelicaImportError

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica"
WITH_VOLUME = DATA / "ChimneyShaftWithVolume.json"


def test_a_dynamic_column_becomes_a_volume_between_two_half_columns():
    doc = expand.expand(schema.load(WITH_VOLUME))
    by_name = {c.name: c for c in doc.components}
    assert "sha" not in by_name
    vol, top, bot = by_name["sha|vol"], by_name["sha|colTop"], by_name["sha|colBot"]
    assert vol.cls == "Buildings.Fluid.MixingVolumes.MixingVolume"
    assert vol.parameters["V"] == 3.0 and vol.parameters["nPorts"] == 2
    assert vol.parameters["m_flow_nominal"] == pytest.approx(3.0 / 360.0)
    assert vol.parameters["massDynamics"] == vol.parameters["energyDynamics"]
    assert vol.parameters["T_start"] == 293.15
    assert (top.parameters["h"], bot.parameters["h"]) == (1.5, 1.5)
    assert top.parameters["densitySelection"].endswith("fromBottom")
    assert bot.parameters["densitySelection"].endswith("fromTop")
    pairs = {frozenset(c) for c in doc.connections}
    for a, b in (("sha|colTop.port_a", "oriChiTop.port_a"),
                 ("sha|colBot.port_b", "oriChiBot.port_b"),
                 ("sha|colTop.port_b", "sha|vol.ports[2]"),
                 ("sha|vol.ports[1]", "sha|colBot.port_a")):
        assert frozenset((a, b)) in pairs


def test_a_dynamic_column_with_heat_transfer_is_refused():
    doc = schema.load(WITH_VOLUME)
    comps = tuple(dataclasses.replace(c, parameters={**c.parameters, "use_HeatTransfer": True})
                  if c.name == "sha" else c for c in doc.components)
    with pytest.raises(ModelicaImportError, match=r"sha \(.*MediumColumnDynamic\): "
                                                  r"use_HeatTransfer = true"):
        expand.expand(dataclasses.replace(doc, components=comps))


def test_the_expanded_shaft_is_two_ordinary_column_paths():
    """oriChiTop: from the shaft volume up through the top half column to the ambient;
    oriChiBot: from the room up through the bottom half column into the volume. Walking
    either path from its source, the half column's bottom port comes first: sign -1."""
    doc = expand.expand(schema.load(WITH_VOLUME))
    doc = dataclasses.replace(            # without the controller, which is still refused
        doc, signals=tuple(x for x in doc.signals if x.name != "con"),
        connections=tuple(c for c in doc.connections
                          if not (c[0].startswith("con.") or c[1].startswith("con."))))
    g = graph.build(doc)
    paths = {p.element.name: p for p in g.paths}
    top, bot = paths["oriChiTop"], paths["oriChiBot"]
    assert (top.src, top.tgt) == ("sha|vol", "bou0")
    assert (bot.src, bot.tgt) == ("roo", "sha|vol")
    assert [(c.name, s) for c, s in top.columns] == [("sha|colTop", -1)]
    assert [(c.name, s) for c, s in bot.columns] == [("sha|colBot", -1)]
    assert "sha|vol" in g.zones
    assert [(c.name, z) for c, z in g.heat_sensors] == [("temSen", "roo")]
