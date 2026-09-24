"""Tests for `noodl.apps.building_physics.modelica.graph` (spec section 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from noodl.apps.building_physics.modelica.graph import build
from noodl.apps.building_physics.modelica.schema import ModelicaImportError, load

FIXTURES = Path(__file__).parent / "fixtures"


def _graph(name: str):
    return build(load(FIXTURES / name))


def test_two_zones_orifice_has_no_junctions() -> None:
    g = _graph("two_zones_orifice.json")

    assert g.nodes == {"bouA": "boundary", "bouB": "boundary"}
    assert g.zones == ()
    assert g.boundaries == ("bouA", "bouB")
    assert len(g.paths) == 1
    path = g.paths[0]
    assert path.element.name == "ori"
    assert path.src == "bouA"
    assert path.tgt == "bouB"
    assert path.columns == ()
    assert g.doors == ()
    assert g.zonal == ()
    assert g.pins == ()
    assert g.sources == ()


def test_stack_chain_fuses_the_west_stack_with_column_signs() -> None:
    g = _graph("stack_chain.json")

    assert g.zones == ("volWes", "volTop")
    assert g.boundaries == ()
    # Two plain wires (colWesBot<->oriWesTop, oriWesTop<->colWesTop) hold no zone/boundary
    # port, so they become junctions; every other port belongs to a zone.
    junctions = {n: k for n, k in g.nodes.items() if k == "junction"}
    assert len(junctions) == 2
    assert set(junctions) == {"_j0", "_j1"}

    assert len(g.paths) == 1
    path = g.paths[0]
    assert path.element.name == "oriWesTop"
    # Oriented from the element's port_a side (which leads, through colWesTop, to volTop) to
    # its port_b side (through colWesBot, to volWes) -- see MediumColumn.mo and the
    # ThreeRoomsContam.mo west-stack wiring the fixture copies.
    assert path.src == "volTop"
    assert path.tgt == "volWes"
    columns = [(c.name, sign) for c, sign in path.columns]
    # Both columns keep the flow direction (the whole stack runs straight top to bottom from
    # volTop, through colWesTop and colWesBot, to volWes), so both signs are +1, and
    # colWesTop -- the one nearer the src (port_a) end -- comes first.
    assert columns == [("colWesTop", 1), ("colWesBot", 1)]


def test_door_wiring_pairs_port_a1_b2_and_port_b1_a2() -> None:
    g = _graph("door_wiring.json")

    assert g.zones == ("volWes", "volEas")
    assert len(g.doors) == 1
    door = g.doors[0]
    assert door.component.name == "dooOpeClo"
    assert {door.side_a, door.side_b} == {"volWes", "volEas"}
    # port_a1 and port_b2 both land on volWes; port_b1 and port_a2 both land on volEas.
    assert door.side_a == "volWes"
    assert door.side_b == "volEas"
    assert g.paths == ()


def test_bad_door_wiring_is_refused_naming_the_door() -> None:
    with pytest.raises(ModelicaImportError, match="dooOpeClo"):
        _graph("bad_door_wiring.json")


def test_two_elements_in_a_chain_is_refused_naming_both() -> None:
    with pytest.raises(ModelicaImportError) as excinfo:
        _graph("two_elements_in_chain.json")
    message = str(excinfo.value)
    assert "oriA" in message
    assert "oriB" in message


def test_no_flow_element_chain_is_refused_naming_the_column() -> None:
    with pytest.raises(ModelicaImportError, match="colOnly"):
        _graph("no_flow_element_chain.json")


def test_refused_classes_are_named_together() -> None:
    with pytest.raises(ModelicaImportError) as excinfo:
        _graph("refused.json")
    message = str(excinfo.value)
    assert "bouOut" in message
    assert "wind pressure is not supported" in message
    assert "ctrl" in message
    assert "feedback controllers are not supported" in message


def test_thermal_pin_and_source_are_resolved() -> None:
    g = _graph("thermal_and_source.json")

    assert g.zones == ("volA",)
    assert g.boundaries == ("bouB",)
    assert len(g.paths) == 1
    assert g.paths[0].src == "volA"
    assert g.paths[0].tgt == "bouB"

    assert len(g.pins) == 1
    pin = g.pins[0]
    assert pin.source.name == "TA"
    assert pin.conductor.name == "conA"
    assert pin.zone == "volA"

    assert len(g.sources) == 1
    source, node = g.sources[0]
    assert source.name == "souA"
    assert node == "volA"
