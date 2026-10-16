"""Tests for `noodl.apps.building_physics.modelica.graph`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from noodl.apps.building_physics.modelica import expand, graph, schema
from noodl.apps.building_physics.modelica.graph import build
from noodl.apps.building_physics.modelica.schema import ModelicaImportError, load

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).resolve().parents[3] / "data" / "modelica"


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


def test_two_elements_in_series_meet_at_a_junction() -> None:
    """`volA - oriA - oriB - volB`: two paths joined at one junction, and one series chain
    from volA to volB carrying oriA's flow (the first element in name order)."""
    g = _graph("two_elements_in_chain.json")
    (j,) = g.junctions
    assert g.nodes[j] == "junction"
    by = {p.element.name: p for p in g.paths}
    assert (by["oriA"].src, by["oriA"].tgt) == ("volA", j)
    assert (by["oriB"].src, by["oriB"].tgt) == (j, "volB")
    (chain,) = g.series
    assert (chain.src, chain.tgt, chain.junctions) == ("volA", "volB", (j,))
    assert [(p.element.name, s) for p, s in zip(chain.paths, chain.signs, strict=True)] == [
        ("oriA", 1), ("oriB", 1)]


def test_a_series_chain_takes_the_orientation_of_its_first_element_by_name() -> None:
    """oriB turned round (`volA - oriA - oriB.port_b ... oriB.port_a - volB`): the chain
    runs along oriA, oriB against it."""
    doc = load(FIXTURES / "two_elements_in_chain.json")
    doc = dataclasses.replace(doc, connections=(
        ("volA.ports[1]", "oriA.port_a"), ("oriA.port_b", "oriB.port_b"),
        ("oriB.port_a", "volB.ports[1]")))
    g = build(doc)
    (chain,) = g.series
    assert (chain.src, chain.tgt) == ("volA", "volB")
    assert [s for s in chain.signs] == [1, -1]
    doc = dataclasses.replace(doc, connections=(
        ("volB.ports[1]", "oriA.port_a"), ("oriA.port_b", "oriB.port_b"),
        ("oriB.port_a", "volA.ports[1]")))
    (chain,) = build(doc).series
    assert (chain.src, chain.tgt) == ("volB", "volA")
    assert [p.element.name for p in chain.paths] == ["oriA", "oriB"]


def test_a_ring_of_flow_elements_with_no_volume_is_refused() -> None:
    doc = load(FIXTURES / "two_elements_in_chain.json")
    doc = dataclasses.replace(doc, connections=(
        ("oriA.port_b", "oriB.port_a"), ("oriB.port_b", "oriA.port_a"),
        ("volA.ports[1]", "volB.ports[1]")))
    with pytest.raises(ModelicaImportError) as excinfo:
        build(doc)
    message = str(excinfo.value)
    assert "oriA" in message and "oriB" in message
    assert "a ring of flow elements" in message


def test_a_door_in_a_column_chain_is_refused() -> None:
    doc = load(FIXTURES / "door_wiring.json")
    comps = {c.name: c for c in doc.components}
    ori = dataclasses.replace(comps["dooOpeClo"], name="ori",
                              cls="Buildings.Airflow.Multizone.Orifice",
                              parameters={"A": 0.01, "m": 0.5})
    doc = dataclasses.replace(doc, components=(*doc.components, ori), connections=(
        *[c for c in doc.connections if "dooOpeClo.port_a1" not in c],
        ("dooOpeClo.port_a1", "ori.port_b"), ("ori.port_a", "volWes.ports[3]")))
    with pytest.raises(ModelicaImportError,
                       match="ori .*a door or zonal flow in a hydrostatic-column chain"):
        build(doc)


def test_two_orifices_in_series_through_a_column_meet_at_a_junction():
    """ChimneyShaftNoVolume: roo - oriChiBot - sha (MediumColumn, actual) - oriChiTop -
    bou0. The column belongs to the element whose name sorts first (oriChiBot), whose path
    ends at the junction next to oriChiTop; oriChiTop's path starts there. Walking oriChiBot
    from roo, the column's bottom port comes first: sign -1."""
    doc = schema.load(DATA / "ChimneyShaftNoVolume.json")
    ctx = graph._classify_components(doc)
    graph._process_fluid_connections(doc, ctx)
    graph._assign_nodes(ctx)
    paths, junctions = graph._fuse_paths(ctx)
    by = {p.element.name: p for p in paths}
    (j,) = junctions
    assert (by["oriChiBot"].src, by["oriChiBot"].tgt) == ("roo", j)
    assert (by["oriChiTop"].src, by["oriChiTop"].tgt) == (j, "bou0")
    assert [(c.name, s) for c, s in by["oriChiBot"].columns] == [("sha", -1)]
    assert by["oriChiTop"].columns == ()
    assert ctx.uf.find("oriChiTop.port_a") == ctx.uf.find("sha.port_a")
    assert ctx.root_to_name[ctx.uf.find("oriChiTop.port_a")] == j


def test_no_flow_element_chain_is_refused_naming_the_column() -> None:
    with pytest.raises(ModelicaImportError, match="colOnly"):
        _graph("no_flow_element_chain.json")


def test_refused_classes_are_named_together() -> None:
    with pytest.raises(ModelicaImportError) as excinfo:
        _graph("refused.json")
    message = str(excinfo.value)
    assert "bouOut" in message
    assert "component class is not supported" in message
    assert "wea" in message
    assert "weather data other than a ReaderTMY3 is not supported" in message


def test_malformed_connection_ref_is_gathered_not_raised_immediately() -> None:
    """`graph._split` must append a malformed (no '.') connection reference to the error list
    rather than raising immediately, so gather-all keeps running and names the OTHER error
    (`bouOut`'s refusal as an unsupported class, found in an earlier phase) in the same
    `ModelicaImportError`. `schema.load` does not itself check the
    '<instance>.<port>' format, so this is reachable from a malformed document even though the
    exporter itself never writes one."""
    with pytest.raises(ModelicaImportError) as excinfo:
        _graph("malformed_connection_ref.json")
    message = str(excinfo.value)
    assert "not_a_dotted_reference" in message
    assert "is not an '<instance>.<port>' reference" in message
    assert "bouOut" in message
    assert "component class is not supported" in message


def _head(path, rho: float, g: float) -> float:
    """`(phi_src - phi_tgt) + sum(sign * h * rho * g)`'s column term (module docstring)."""
    return sum(sign * col.parameters["h"] * rho * g for col, sign in path.columns)


def test_stack_chain_head_matches_hand_derived_hydrostatic_balance() -> None:
    """Physics check, not mere self-consistency.

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
    assert "component class is not supported" in message
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


def test_attached_boundary_with_further_connected_ports_is_refused(tmp_path) -> None:
    """`bou.ports[2] -> ori -> volB` beside `bou.ports[1] -> volA`. In
    MBL the air leaving `bou.ports[2]` carries the boundary's own state and mass; merged onto
    volA's node it would carry volA's. Refused by name instead of re-wired."""
    import json

    doc = json.loads((FIXTURES / "attached_boundary.json").read_text())
    for c in doc["components"]:
        if c["name"] == "bou":
            c["parameters"]["nPorts"] = 2
        if c["name"] == "volB":
            c["parameters"]["nPorts"] = 3
    doc["components"].append({"name": "ori", "class": "Buildings.Airflow.Multizone.Orifice",
                              "parameters": {"A": 0.01}})
    doc["connections"] += [["bou.ports[2]", "ori.port_a"], ["ori.port_b", "volB.ports[3]"]]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelicaImportError) as exc:
        build(load(path))
    msg = str(exc.value)
    assert "refused 1 item" in msg
    assert ("bou (Buildings.Fluid.Sources.Boundary_pT): wired straight to volume volA and "
            "also through bou.ports[1], bou.ports[2]") in msg


def test_ports_beside_a_refused_component_add_no_follow_on_message(tmp_path) -> None:
    """An orifice wired to a refused `FlowControlled_m_flow` (a mover) is not also
    reported as "not connected"; the refusal names the root cause once."""
    import json

    doc = json.loads((FIXTURES / "two_zones_orifice.json").read_text())
    doc["components"][0]["class"] = "Buildings.Fluid.Movers.FlowControlled_m_flow"
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelicaImportError) as exc:
        build(load(path))
    msg = str(exc.value)
    assert "refused 1 item" in msg
    assert "bouA (Buildings.Fluid.Movers.FlowControlled_m_flow)" in msg
    assert "not connected" not in msg


def test_a_temperature_sensor_on_a_volume_is_wired_into_its_controller():
    doc = expand.expand(schema.load(DATA / "ChimneyShaftWithVolume.json"))
    g = graph.build(doc)
    assert g.signal_wires == (("temSen.T", "con.u_m"),)
    assert [(c.name, zone) for c, zone in g.heat_sensors] == [("temSen", "roo")]


def test_a_temperature_sensor_wired_to_anything_but_a_block_input_is_refused():
    doc = schema.load(DATA / "ChimneyShaftWithVolume.json")
    rewired = dataclasses.replace(doc, connections=tuple(
        ("temSen.T", "gain.y") if tuple(c) == ("temSen.T", "con.u_m") else c
        for c in doc.connections))
    with pytest.raises(ModelicaImportError,
                       match=r"temSen\.T -> gain\.y: y is not an input of gain "
                             r"\(Modelica\.Blocks\.Math\.Gain\); its inputs are u"):
        graph.build(expand.expand(rewired))


def test_a_component_output_into_a_signal_must_come_from_a_temperature_sensor():
    doc = schema.load(DATA / "ChimneyShaftWithVolume.json")
    wired = dataclasses.replace(doc, connections=(*doc.connections, ("roo.T", "con.u_m")))
    with pytest.raises(ModelicaImportError, match=r"roo\.T -> con\.u_m"):
        graph.build(expand.expand(wired))


def test_a_temperature_sensor_must_sit_on_a_volume_heat_port():
    doc = schema.load(DATA / "ChimneyShaftWithVolume.json")
    loose = dataclasses.replace(doc, connections=tuple(
        c for c in doc.connections if "temSen.port" not in c))
    with pytest.raises(ModelicaImportError, match=r"temSen \(.*TemperatureSensor\): "
                                                  r"unsupported heat-port wiring"):
        graph.build(expand.expand(loose))
