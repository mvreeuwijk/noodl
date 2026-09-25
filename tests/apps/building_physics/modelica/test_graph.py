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


def _head(path, rho: float, g: float) -> float:
    """`(phi_src - phi_tgt) + sum(sign * h * rho * g)`'s column term (module docstring)."""
    return sum(sign * col.parameters["h"] * rho * g for col, sign in path.columns)


def test_stack_chain_head_matches_hand_derived_hydrostatic_balance() -> None:
    """Physics check (fix round 1, review item 1), not mere self-consistency.

    `MediumColumn.mo`'s own relation is `port_a.p - port_b.p = -h*rho*g_n` (the bottom port,
    `port_b`, sits at the HIGHER pressure). Chasing that through the west stack's wiring by
    hand (also written out in `graph.py`'s module docstring) gives, at zero flow through the
    orifice (`dp_element = 0`),

        p(volWes) - p(volTop) = 2 * h * rho * g

    with `h = 1.5` for both `colWesTop` and `colWesBot`. This test computes the SAME quantity
    from the graph's signs (`FlowPath.columns`) and checks it against that independently
    hand-derived number, not merely that the code agrees with itself.
    """
    g = _graph("stack_chain.json")
    path = g.paths[0]
    assert path.src == "volTop"
    assert path.tgt == "volWes"

    rho, g_n = 1.2, 9.81
    head = _head(path, rho, g_n)
    # dp_element = (phi_src - phi_tgt) + head = 0  =>  phi_tgt - phi_src = head.
    implied_p_volWes_minus_p_volTop = head
    assert implied_p_volWes_minus_p_volTop == pytest.approx(2 * 1.5 * rho * g_n)


def test_reverse_column_gets_sign_minus_one_and_matches_hand_derivation() -> None:
    """A column wired the OTHER way along its path (its own `port_a` facing the `tgt` end,
    `port_b` facing the flow element) gets sign -1.

    Hand derivation (`reverse_column.json`: `volA -- oriX -- colRev -- volB`, `colRev.port_a`
    wired to `volB`, `colRev.port_b` wired to `oriX.port_b`): `MediumColumn.mo` gives
    `p(colRev.port_a) - p(colRev.port_b) = -h*rho*g_n`, i.e. `p(volB) - p(oriX.port_b) =
    -h*rho*g_n`, so `p(oriX.port_b) = p(volB) + h*rho*g_n`; `oriX.port_a` is wired straight to
    `volA` with no column, so `p(oriX.port_a) = p(volA)`. Hence

        dp_oriX = p(volA) - p(volB) - h*rho*g_n = (phi_src - phi_tgt) + (-1)*h*rho*g_n

    matching the module docstring's formula with sign -1, and at zero flow
    `p(volA) - p(volB) = h*rho*g_n`.
    """
    g = _graph("reverse_column.json")
    path = g.paths[0]
    assert path.src == "volA"
    assert path.tgt == "volB"
    assert [(c.name, sign) for c, sign in path.columns] == [("colRev", -1)]

    rho, g_n = 1.2, 9.81
    head = _head(path, rho, g_n)
    assert head == pytest.approx(-1 * 2.0 * rho * g_n)
    # dp_element = 0  =>  phi_src - phi_tgt = -head = h*rho*g_n.
    implied_p_volA_minus_p_volB = -head
    assert implied_p_volA_minus_p_volB == pytest.approx(2.0 * rho * g_n)


def test_refusals_are_gathered_across_every_phase_into_one_error() -> None:
    """A doc combining an unsupported class (found in the earliest phase) and a bad door
    wiring (found in a much later phase) must name BOTH in the one raised error -- refusals
    are gathered across every phase, not just the first one that finds something."""
    with pytest.raises(ModelicaImportError) as excinfo:
        _graph("refused_and_bad_door.json")
    message = str(excinfo.value)
    assert "bouOut" in message
    assert "wind pressure is not supported" in message
    assert "dooOpeClo" in message


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


# ------------------------------------------------------------------ in-line flow sensors
def test_inline_flow_sensors_are_transparent_wires() -> None:
    """A two-port flow sensor (`Fluid/Sensors/BaseClasses/PartialFlowSensor.mo:14-16`:
    `port_b.m_flow = -port_a.m_flow`, `port_a.p = port_b.p`) joins its two ports into one
    node and counts for nothing in a junction's degree, so a sensor in series with a flow
    element or inside a column chain leaves the network unchanged."""
    g = _graph("inline_sensors.json")

    assert g.zones == ("vol",)
    assert g.boundaries == ("bouA", "bouB")
    # One junction: oriCol.port_b ... senC1 ... senC2 ... col.port_a (degree two once the
    # sensor ports are discounted). The one-port TraceSubstances sensor and the
    # RelativePressure sensor (no flow through it) are observers, not wires.
    assert {n for n, k in g.nodes.items() if k == "junction"} == {"_j0"}
    paths = {p.element.name: p for p in g.paths}
    assert set(paths) == {"ori", "oriCol", "oriOut"}
    assert (paths["ori"].src, paths["ori"].tgt) == ("bouA", "bouB")
    assert (paths["oriCol"].src, paths["oriCol"].tgt) == ("bouA", "vol")
    assert [(c.name, s) for c, s in paths["oriCol"].columns] == [("col", 1)]
    assert [(d.component.name, d.side_a, d.side_b) for d in g.doors] == [("doo", "bouA", "bouB")]


def test_inline_sensors_record_the_element_port_they_measure() -> None:
    """Each sensor's `port_a.m_flow` equals `sign * m_flow` into one flow-element port: the
    port reached from the sensor's `port_a` side (sign -1, the flow leaves that element port
    into the sensor) or, failing that, from its `port_b` side (sign +1), walking through
    other in-line sensors."""
    g = _graph("inline_sensors.json")
    got = {s.component.name: (s.node, s.port, s.sign) for s in g.sensors}
    assert got == {
        "senOri": ("bouB", "ori.port_b", -1),
        "senDoo": ("bouB", "doo.port_a2", -1),
        "senC1": ("_j0", "col.port_a", -1),
        "senC2": ("_j0", "col.port_a", -1),
    }


def test_inline_sensor_without_flow_reversal_is_refused(tmp_path) -> None:
    import json

    doc = json.loads((FIXTURES / "inline_sensors.json").read_text())
    for c in doc["components"]:
        if c["name"] == "senOri":
            c["parameters"]["allowFlowReversal"] = False
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelicaImportError, match=r"senOri \(Buildings.Fluid.Sensors.MassFlowRate\)"
                       r": allowFlowReversal = false"):
        build(load(path))


# ----------------------------------------------------------------- prescribed heat flow
def test_prescribed_heat_flow_into_a_zone_is_resolved() -> None:
    g = _graph("heat_flow.json")
    assert [(c.name, zone) for c, zone in g.heat_sources] == [("preHea", "vol")]


def test_prescribed_heat_flow_not_into_a_zone_is_refused(tmp_path) -> None:
    import json

    doc = json.loads((FIXTURES / "heat_flow.json").read_text())
    doc["connections"][-1] = ["preHea.port", "bou.heatPort"]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelicaImportError, match=r"preHea \(Modelica.Thermal.HeatTransfer"
                       r".Sources.PrescribedHeatFlow\): unsupported heat-port wiring"):
        build(load(path))


# ------------------------------------------------------- boundary wired straight to a volume
def test_boundary_wired_straight_to_a_volume_shares_its_node() -> None:
    """`Validation/OpenDoorBuoyancyDynamic.mo` connects `bou.ports[1]` to `bouA.ports[3]`:
    the boundary fixes the volume's pressure. The two are one node, named after the volume,
    and the boundary is recorded as attached to it rather than listed as a boundary node."""
    g = _graph("attached_boundary.json")
    assert g.nodes == {"volA": "zone", "volB": "zone"}
    assert g.zones == ("volA", "volB")
    assert g.boundaries == ()
    assert [(z, b.name) for z, b in g.attached] == [("volA", "bou")]
    assert [(d.side_a, d.side_b) for d in g.doors] == [("volA", "volB")]


def test_two_boundaries_or_two_volumes_wired_together_stay_refused(tmp_path) -> None:
    import json

    doc = json.loads((FIXTURES / "attached_boundary.json").read_text())
    doc["connections"].append(["bou.ports[1]", "volB.ports[3]"])
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelicaImportError, match="bou and volA and volB: connected directly"):
        build(load(path))
