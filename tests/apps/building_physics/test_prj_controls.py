"""Control nodes in the `.prj` reader (TN 1887 Appendix A section 12b).

The behaviour each node type, sensor and actuator has was measured on ContamX 3.4.1.7 (the
controls work's probes); the ContamX values quoted here are from those runs.
"""

import re
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.controls import (
    DriverReading,
    FunctionReading,
    Sensor,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj, steady
from noodl.apps.building_physics.prj_controls import Actuation
from noodl.apps.building_physics.simulation import simulate, simulate_project
from noodl.elements import Damper, FixedFlow

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
DCV = DATA / "dcv_three_zone.prj"
PRESSURE = DATA / "pressure_exhaust_three_zone.prj"
WINDOW = DATA / "hysteresis_window_three_zone.prj"
DOORWAY = DATA / "doorway_damper_fan.prj"
SCHEDULED = DATA / "scheduled_three_zone.prj"
FAN_CURVE = DATA / "fan_curve_three_zone.prj"
DUCTS = DATA / "duct_three_zone.prj"
AHS = DATA / "ahs_three_zone.prj"
F64 = torch.float64
TIGHT = dict(atol=1e-14, rtol=1e-14)
P4 = ("   4    1  -1   3   3   0   3   0   0   0   1  30.000   0.000   1.500 1 0 0.1225 90"
      " 0 0 0  23  5 -1 0 0 0 0 0 0\n")


def _text(path):
    return path.read_bytes().decode().replace("\r\n", "\n")


def _write(tmp_path, text, name="variant.prj"):
    p = tmp_path / name
    p.write_bytes(text.replace("\n", "\r\n").encode())
    return p


def _section(text, name, count, body):
    pat = re.compile(r"^\d+ ! " + re.escape(name) + r":\n.*?^-999\n",
                     re.DOTALL | re.MULTILINE)
    m = pat.search(text)
    assert m, name
    return text[:m.start()] + f"{count} ! {name}:\n{body}-999\n" + text[m.end():]


def _replace(text, *pairs):
    for old, new in pairs:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def _with_fan(text, control_nr, nodes_body, count):
    """NIST's three-zone sample with path 5, its constant-flow fan (element 2, 0.100342
    kg/s) from the ambient into zone one, under control node `control_nr`."""
    fan = (f"   5    0  -1   1   2   0   0   0   0   {control_nr}   1   0.000   0.000"
           f"   1.500 1 0 0 -1 0 0 0  23  2 -1 0 0 0 0 0 0\n")
    text = text.replace("4 ! flow paths:", "5 ! flow paths:", 1).replace(P4, P4 + fan, 1)
    return _section(text, "control nodes", count, nodes_body)


def _schedule_nodes(*schedules):
    """One schedule node per week schedule number, numbered 1, 2, ... in that order."""
    body = "! # typ seq f n  c1  c2 name\n"
    for nr, ps in enumerate(schedules, 1):
        body += f"  {nr} sch   {nr} 0 0   0   0 <none>\nschedule\n {ps}\n"
    return len(schedules), body


def test_the_control_section_is_read_node_by_node():
    p = read_prj(WINDOW)
    nodes = p.control_nodes
    assert len(nodes) == 49
    assert (nodes[1].type, nodes[1].flags, nodes[1].data[:3]) == ("sns", 1, ("293.15", "0.1",
                                                                           "0.0"))
    assert nodes[5].type == "max" and nodes[5].inputs == (2, 4)
    assert nodes[8].type == "sub" and nodes[8].inputs == (1, 3)
    assert nodes[26].data == ("1",) and nodes[27].name == "o1"
    assert len(read_prj(THREE).control_nodes) == 33
    assert read_prj(THREE).control_plan is None          # 33 nodes, nothing uses them


def test_a_schedule_node_is_an_input_on_the_networks_input():
    p = read_prj(WINDOW)
    assert p.control_plan.schedule_slots == {7: 0, 26: 1}
    keyed = [i for i in p.inputs if i.key == "controls.input"]
    assert sorted(i.index for i in keyed) == [(0,), (1,)]


def test_named_report_nodes_are_the_control_outputs_in_node_order():
    p = read_prj(DCV)
    assert [name for name, _ in p.control_plan.outputs] == ["co2", "pi", "fan", "excess"]
    order = p.control_plan.order
    assert order == tuple(sorted(order))                 # seq = node number in the fixtures
    history = torch.arange(2.0 * len(order), dtype=F64).reshape(2, len(order))
    positions = [pos for _, pos in p.control_plan.outputs]
    assert p.control_outputs(history).tolist() == history[:, positions].tolist()


def test_the_evaluation_order_is_the_files_sequence_numbers(tmp_path):
    """Sequence numbers, not node numbers, order the network: here the pass node (seq 1)
    comes before the constant it reads (seq 2)."""
    body = ("! # typ seq f n  c1  c2 name\n  1 pas   2 0 1   2   0 out\npass\n"
            "  2 set   1 0 0   0   0 <none>\nconstant\n 0.5\n")
    p = read_prj(_write(tmp_path, _with_fan(_text(THREE), 1, body, 2)))
    assert p.control_plan.order == (2, 1)
    assert p.control_plan.outputs == (("out", 1),)


def test_each_record_writes_what_its_schedule_writes():
    zone = read_prj(WINDOW)
    acts = {a.record: a for a in zone.control_plan.actuations}
    one = zone.net.node_index("one")
    assert acts["zone 1"] == Actuation(26, "T", (one,), "zone 1")
    assert acts["path 5"] == Actuation(6, "flow_multiplier.pl_3", (4,), "path 5")  # 5th orifice
    dcv = read_prj(DCV)
    acts = {a.record: a for a in dcv.control_plan.actuations}
    assert acts["source 1"] == Actuation(10, "species.source_multiplier", (0,), "source 1")
    assert acts["path 5"].key == "flow_multiplier.fan_2"


def test_sensors_map_to_state_and_driver_readings():
    def sensors(path):
        p = read_prj(path)
        model, state, drivers = project_to_model(p)
        network = model.closures[0]
        return p, model, state, {n: b for n, b in zip(p.control_plan.order, network.blocks,
                                                      strict=True) if isinstance(b, Sensor)}

    p, model, _, s = sensors(PRESSURE)
    assert (s[1].reading.key, s[1].reading.index) == ("air.phi",
                                                      (model.net.node_index("three"),))
    assert s[5].reading.key == "air.dp" and s[5].reading.index == (p.paths[1].edge_columns[0],)
    assert s[7].reading.key == "air.q" and s[7].reading.index == (p.paths[5].edge_columns[0],)
    assert (s[1].offset, s[1].scale) == (0.0, 1.0)       # flags 0: no conversion
    p, model, state, s = sensors(DCV)
    reading = s[1].reading
    assert isinstance(reading, FunctionReading) and reading.key == "species.x"
    x = torch.tensor([[1e-3], [2e-3], [3e-3]], dtype=F64)  # zones one, two, three; CO2
    assert float(reading({**state, "species.x": x}, {})) == 1e-3
    assert s[1].scale == 1e6
    p, model, _, s = sensors(WINDOW)
    assert isinstance(s[1].reading, DriverReading) and s[1].reading.key == "T"
    assert s[1].reading.index == (model.net.node_index("one"),)
    assert (s[1].offset, s[1].scale) == (293.15, 0.1)


def test_unused_unsupported_nodes_are_ignored_and_a_needed_one_is_refused(tmp_path):
    """NIST's three-zone sample: 33 nodes (integrations, reports) that nothing uses, so it
    loads (ContamX runs it too). Point path 4 at node 2, an integration: refused, naming the
    path, the node and why."""
    text = _text(THREE)
    used = text.replace(P4, P4.replace("   0   0   0   1  30.000", "   0   0   2   1  30.000"),
                        1)
    with pytest.raises(ValueError, match=r"path 4 depends on control node 2 \(int\): "
                                         r"integration over time"):
        read_prj(_write(tmp_path, used))


def test_an_unknown_node_type_stops_the_reader(tmp_path):
    text = _text(THREE).replace(" 22 mul  25", " 22 xyz  25", 1)
    with pytest.raises(ValueError, match=r"control node 22 has type 'xyz'"):
        read_prj(_write(tmp_path, text))


def test_a_missing_node_and_a_wrong_input_count_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"path 5 references control node 3, which the "
                                         r"'control nodes' section does not define"):
        read_prj(_write(tmp_path, _with_fan(_text(THREE), 3, *_schedule_nodes(1)[::-1])))
    body = "! # typ seq f n  c1  c2 name\n  1 add   1 0 1   2   0 <none>\nadd\n\n"
    body += "  2 set   2 0 0   0   0 <none>\nconstant\n 1\n"
    with pytest.raises(ValueError, match=r"control node 1 \(add\) needs 2 input\(s\), "
                                         r"found \[2\]"):
        read_prj(_write(tmp_path, _with_fan(_text(THREE), 1, body, 2)))


def test_a_control_on_a_path_scales_its_flow_without_clipping(tmp_path):
    """ContamX: control 2.0 on the fan path -> 0.200684 kg/s, -0.5 -> -0.050171, 0 -> 0
    (a multiplier, as a schedule; not clipped to 0..1)."""
    for value, flow in ((2.0, 0.200684), (-0.5, -0.050171), (0.0, 0.0)):
        body = ("! # typ seq f n  c1  c2 name\n  1 set   1 0 0   0   0 <none>\nconstant\n"
                f" {value}\n")
        p = read_prj(_write(tmp_path, _with_fan(_text(THREE), 1, body, 1)))
        model, state, drivers = project_to_model(p, ambient={"Ws": 0.0})
        out = simulate(model, state, drivers, [0.0, 300.0], **TIGHT)
        got = p.path_flows(out["air.q"])[:, 4]
        assert got.tolist() == pytest.approx([flow, flow], rel=1e-12, abs=1e-15)


def _same_runs(tmp_path, path, direct, controlled, schedules, *, period=None, tight=True):
    """`simulate_project` on `path` with its records as `direct`, and with them as
    `controlled` under schedule nodes of `schedules`: bit-identical (ContamX: a control
    through a schedule node is the schedule attached directly, max |difference| 0)."""
    text = _text(path)
    if period is not None:
        text = _replace(text, period)
    plain = _replace(text, *zip(direct, direct, strict=True))       # each line present once
    nodes = _section(_replace(text, *zip(direct, controlled, strict=True)), "control nodes",
                     *_schedule_nodes(*schedules))
    for suffix in (".wth", ".ctm"):                     # the run control's files
        if path.with_suffix(suffix).exists():
            (tmp_path / path.with_suffix(suffix).name).write_bytes(
                path.with_suffix(suffix).read_bytes())
    kw = TIGHT if tight else {}
    a = simulate_project(read_prj(_write(tmp_path, plain, "plain.prj")), **kw)
    project = read_prj(_write(tmp_path, nodes, "controlled.prj"))
    assert project.control_plan is not None and project.control_plan.actuations
    b = simulate_project(project, **kw)
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=0, atol=0, msg=key)
    return project, b


def test_zone_path_and_source_controls_act_as_their_schedules(tmp_path):
    """`scheduled_three_zone.prj` 07:00-11:30: zone two's temperature ramp (schedule 3),
    path 1's window multiplier (schedule 4), path 5's fan (schedule 1) and source 1's
    release (schedule 2), each through a schedule node instead."""
    project, _ = _same_runs(
        tmp_path, SCHEDULED,
        ["   2  3   3   0   0   1   0.000   600 293.15 0 two",
         "   1    1  -1   1   3   0   3   0   4   0   1",
         "   5    1   3  -1   2   0   0   0   1   0   1",
         "  1   1   1   2   0     1     0"],
        ["   2  3   0   1   0   1   0.000   600 293.15 0 two",
         "   1    1  -1   1   3   0   3   0   0   2   1",
         "   5    1   3  -1   2   0   0   0   0   3   1",
         "  1   1   1   0   4     1     0"],
        (3, 4, 1, 2),
        period=("Jan01 00:00:00  Jan01 24:00:00", "Jan01 07:00:00  Jan01 11:30:00"))
    keys = sorted(a.key for a in project.control_plan.actuations)
    assert keys == ["T", "flow_multiplier.fan_2", "flow_multiplier.pl_3",
                    "species.source_multiplier"]


def test_a_fan_curve_control_is_its_speed_ratio(tmp_path):
    """`fan_curve_three_zone.prj`: path 5's speed schedule through a schedule node."""
    project, _ = _same_runs(tmp_path, FAN_CURVE,
                            ["   5    0   3  -1   4   0   0   0   1   0   1"],
                            ["   5    0   3  -1   4   0   0   0   0   1   1"], (1,))
    (act,) = project.control_plan.actuations
    assert act.key == "fan_speed.fan_4" and project.speed_edges == {"fan_4": 1}


def test_duct_fan_and_terminal_controls_act_as_their_schedules(tmp_path):
    """`duct_three_zone.prj`: segment 2's fan speed (schedule 1) and terminal 6's opening
    (schedule 2) through schedule nodes, to 01:10 (both change before)."""
    project, _ = _same_runs(tmp_path, DUCTS,
                            ["  2  0   2   3   3   0   1   0   4 1",
                             "  6  0 1  2   0   0   2   0   1"],
                            ["  2  0   2   3   3   0   0   1   4 1",
                             "  6  0 1  2   0   0   0   2   1"], (1, 2),
                            period=("Jan01 02:00:00", "Jan01 01:10:00"), tight=False)
    acts = {a.record: a for a in project.control_plan.actuations}
    assert acts["segment 2"] == Actuation(1, "fan_speed.duct_3", (0,), "segment 2")
    assert acts["terminal 6"] == Actuation(2, "terminal_opening.duct_terminal", (2,),
                                           "terminal 6")


def test_air_handler_controls_act_as_their_schedules(tmp_path):
    """`ahs_three_zone.prj`: the supply point to zone one (path 5, schedule 2) and the
    recirculation path's outdoor-air fraction (path 8, schedule 1) through schedule nodes."""
    project, _ = _same_runs(tmp_path, AHS,
                            ["   5    8   5   1   0   0   0   1   2   0   1",
                             "   8   16   4   5   0   1   0   0   1   0   1"],
                            ["   5    8   5   1   0   0   0   1   0   1   1",
                             "   8   16   4   5   0   1   0   0   0   2   1"], (2, 1))
    acts = {a.record: a for a in project.control_plan.actuations}
    assert acts["path 5"] == Actuation(1, "flow_multiplier.ahs_1", (0,), "path 5")
    assert acts["path 8"] == Actuation(2, "outdoor_air_fraction.ahs_1", None, "path 8")


def test_controls_on_records_a_schedule_cannot_drive_are_refused(tmp_path):
    nodes = _schedule_nodes(1)
    cases = [
        (DUCTS, "  2  0 0  1   0   0   0   0   1", "  2  0 0  1   0   0   0   1   1",
         r"junction 2 references control node 1: a control on a junction that is not a "
         r"terminal is not supported"),
        (DUCTS, "  1  0   1   2   1   1   0   0   4 4", "  1  0   1   2   1   1   0   1   4 4",
         r"segment 1 references control node 1 on Darcy-Colebrook duct element 1"),
        (AHS, "   9   32  -1   5   0   2   0   0   0   0   1",
         "   9   32  -1   5   0   2   0   0   0   1   1",
         r"path 9 \(an AHS outdoor-air or exhaust path\) references control node 1"),
        (DCV, "1 CO2 ccf occupants\n\n 8e-06 0 0 0", "1 CO2 brs occupants\n\n 8e-06 0",
         r"source 1 references control node 10: a control on a triggered source is not "
         r"supported"),
    ]
    for path, old, new, message in cases:
        text = _replace(_text(path), (old, new))
        if path is not DCV:
            text = _section(text, "control nodes", *nodes)
        with pytest.raises(ValueError, match=message):
            read_prj(_write(tmp_path, text))


def test_refused_sensors_name_the_node(tmp_path):
    def body(sensor):
        return ("! # typ seq f n  c1  c2 name\n  1 sns   1 0 0   0   0 <none>\nsensor\n"
                f" {sensor}\n  2 pas   2 0 1   1   0 out\npass\n")

    cases = [("0 1 0 0 1 1 5 0 0 0 0 none", r"control node 1 \(sns\).*occupancy"),
             ("0 1 0 0 1 5 0 0 0 0 0 none", r"control node 1 \(sns\).*exposure"),
             ("0 1 0 0 1 1 0 0 0 0 0 CO2", r"control node 1 \(sns\).*species 'CO2'"),
             ("0 1 0 0 7 1 0 0 0 0 0 sarin", r"control node 1 \(sns\) senses zone 7"),
             ("0 1 0 0 9 2 2 0 0 0 0 none", r"control node 1 \(sns\) senses path 9"),
             ("0 1 0 0 1 2 4 0 0 0 0 none", r"control node 1 \(sns\).*path sensor of "
                                            r"measure 4")]
    zone_one = ("   1  3   0   0   0   1", "   1  3   0   2   0   1")  # zone 1 on node 2
    for sensor, message in cases:
        text = _section(_replace(_text(THREE), zone_one), "control nodes", 2, body(sensor))
        p = read_prj(_write(tmp_path, text))
        with pytest.raises(ValueError, match=message):
            project_to_model(p)
    door = _section(_replace(_text(DOORWAY), zone_one), "control nodes", 2,
                    body("0 1 0 0 3 2 2 0 0 0 0 none"))
    p = read_prj(_write(tmp_path, door))
    with pytest.raises(ValueError, match=r"control node 1 \(sns\).*path 3.*two"):
        project_to_model(p)


def test_duct_sensors_read_their_segment_and_terminal_columns(tmp_path):
    body = ("! # typ seq f n  c1  c2 name\n"
            "  1 sns   1 0 0   0   0 <none>\nsensor\n 0 1 0 0 4 4 2 0 0 0 0 none\n"
            "  2 sns   2 0 0   0   0 <none>\nsensor\n 0 1 0 0 6 6 3 0 0 0 0 none\n"
            "  3 sns   3 0 0   0   0 <none>\nsensor\n 0 1 0 0 4 3 1 0 0 0 0 none\n"
            "  4 pas   4 0 1   1   0 seg\npass\n  5 pas   5 0 1   2   0 term\npass\n"
            "  6 pas   6 0 1   3   0 junction\npass\n"
            "  7 set   7 0 0   0   0 <none>\nconstant\n 1\n")    # segment 2's fan speed
    text = _replace(_text(DUCTS), ("  2  0   2   3   3   0   1   0   4 1",
                                   "  2  0   2   3   3   0   1   7   4 1"))
    p = read_prj(_write(tmp_path, _section(text, "control nodes", 7, body)))
    model, _, _ = project_to_model(p)
    s = model.closures[0].blocks
    seg = next(x for x in p.segments if x.nr == 4)
    term = next(x for x in p.terminals if x.junction_nr == 6)
    assert (s[0].reading.key, s[0].reading.index) == ("air.q", (seg.edge_columns[0],))
    assert (s[1].reading.key, s[1].reading.index) == ("air.dp", (term.edge_columns[0],))
    assert (s[2].reading.key, s[2].reading.index) == ("T", (model.net.node_index(
        p.junction_nr_to_name[4]),))


def test_without_a_species_layer_a_source_control_is_not_built(tmp_path):
    """As the source schedules (`Project.inputs_for`): a model without contaminants has no
    source multipliers to write."""
    text = _replace(_text(SCHEDULED), ("  1   1   1   2   0     1     0",
                                       "  1   1   1   0   1     1     0"))
    p = read_prj(_write(tmp_path, _section(text, "control nodes", *_schedule_nodes(2))))
    model, _, _ = project_to_model(p, species=False)
    assert model.closures[0].actuators == []


def test_a_steady_run_with_controls_is_refused(tmp_path):
    text = _text(DCV).replace("     1      1     30   1e-05", "     0      1     30   1e-05", 1)
    with pytest.raises(ValueError, match="control nodes need a transient airflow"):
        simulate_project(read_prj(_write(tmp_path, text)))


def test_project_to_model_builds_a_new_network_for_every_model():
    p = read_prj(DCV)
    first, _, _ = project_to_model(p)
    second, _, _ = project_to_model(p)
    assert first.closures[0] is not second.closures[0]
    assert p.closures == []


def test_simulate_project_reports_the_named_outputs():
    p = read_prj(PRESSURE)
    out = simulate_project(p, **TIGHT)
    assert out["control"].shape == (289, 4)
    assert float(out["control"][0, 0]) == 0.0           # zone gauge pressure before a solve
    torch.testing.assert_close(out["control"], p.control_outputs(out["controls.outputs"]),
                               rtol=0, atol=0)


def test_report_nodes_alone_build_no_network(tmp_path):
    """Named report nodes act on nothing: a project whose control nodes only report (here
    through an integration, which the reader does not build) loads, builds no network and
    solves steady as without them."""
    body = ("! # typ seq f n  c1  c2 name\n"
            "  1 sns   1 0 0   0   0 <none>\nsensor\n 0 1 0 0 1 1 0 0 0 0 0 sarin\n"
            "  2 int   2 0 2   1   3 <none>\nintegrate\n\n"
            "  3 set   3 0 0   0   0 <none>\nconstant\n 1\n"
            "  4 pas   4 0 1   2   0 dose\npass\n")
    p = read_prj(_write(tmp_path, _section(_text(THREE), "control nodes", 4, body)))
    assert p.control_plan is None and len(p.control_nodes) == 4
    model, state, drivers = project_to_model(p)
    plain, pstate, pdrivers = project_to_model(read_prj(THREE))
    assert len(model.closures) == len(plain.closures)
    out, ref = steady(model, state, drivers), steady(plain, pstate, pdrivers)
    torch.testing.assert_close(out["air.q"], ref["air.q"], rtol=0, atol=0)


def test_a_super_node_is_refused_where_it_is_read(tmp_path):
    body = ("! # typ seq f n  c1  c2 name\n  1 sup   1 0 0   0   0 <none>\nsuper\n"
            " 1 2 3 4\n")
    with pytest.raises(ValueError, match=r"control node 1 is a super node"):
        read_prj(_write(tmp_path, _section(_text(THREE), "control nodes", 1, body)))


def test_a_control_overrides_a_schedule_on_the_same_record(tmp_path):
    """ContamX: schedule 0.25 and control 0.5 on the fan path -> 0.050171 kg/s."""
    schedules = ("1 ! day-schedules:\n! # npts shap utyp ucnv name\n"
                 "  1    2    0    1    0 quarter\n\n 00:00:00 0.25\n 24:00:00 0.25\n-999\n"
                 "1 ! week-schedules:\n! # utyp ucnv name\n  1    1    0 week\n\n"
                 + " ".join(["1"] * 12) + "\n-999\n")
    fan = ("   5    0  -1   1   2   0   0   0   1   1   1   0.000   0.000"
           "   1.500 1 0 0 -1 0 0 0  23  2 -1 0 0 0 0 0 0\n")
    text = _replace(_text(THREE), ("0 ! day-schedules:\n-999\n0 ! week-schedules:\n-999\n",
                                   schedules),
                    ("4 ! flow paths:", "5 ! flow paths:"), (P4, P4 + fan))
    body = "! # typ seq f n  c1  c2 name\n  1 set   1 0 0   0   0 <none>\nconstant\n 0.5\n"
    p = read_prj(_write(tmp_path, _section(text, "control nodes", 1, body)))
    model, state, drivers = project_to_model(p, ambient={"Ws": 0.0})
    assert drivers["flow_multiplier.fan_2"].tolist() == [0.25]   # the schedule alone
    out = simulate(model, state, drivers, [0.0, 300.0], inputs=p.inputs, **TIGHT)
    assert p.path_flows(out["air.q"])[:, 4].tolist() == pytest.approx([0.050171] * 2,
                                                                       rel=1e-5)


def test_one_fan_or_damper_element_on_several_paths_is_one_kind_per_edge(tmp_path):
    """A second path on the constant-flow fan (mult 2) and one on the backdraft damper
    (mult 3): one element per kind with a coefficient per edge, in the kind order a single
    path gives, and each path its own flow."""
    text = _text(DOORWAY)
    p6 = ("   6    0   1  -1   5   0   0   0   0   0   1  10.000   0.000   1.500 1 0 0 -1 0 0 0"
          "  23  2 -1 0 0 0 0 0 0\n")
    extra = ("   7    0   1  -1   1   0   0   0   0   0   1  10.000   0.000   2.000 2 0 0 -1 0"
             " 0 0  23  2 -1 0 0 0 0 0 0\n"
             "   8    0  -1   1   4   0   0   0   0   0   1 -10.000   0.000   1.000 3 0 0 -1 0"
             " 0 0  23  2 -1 0 0 0 0 0 0\n")
    two = _replace(text, ("6 ! flow paths:", "8 ! flow paths:"), (p6, p6 + extra))
    one, both = read_prj(DOORWAY), read_prj(_write(tmp_path, two))
    assert both.kinds == one.kinds                     # same kinds, same order
    (fan,) = [e for e in both.laws if isinstance(e, FixedFlow)]
    (damper,) = [e for e in both.laws if isinstance(e, Damper)]
    assert fan.q0.tolist() == pytest.approx([0.200683, 0.401366])
    assert damper.pos.C.tolist() == pytest.approx([0.02, 0.06])
    model, state, drivers = project_to_model(both, ambient={"Ws": 0.0})
    flows = both.path_flows(steady(model, state, drivers)["air.q"])
    assert flows[4].item() == pytest.approx(0.200683)
    assert flows[6].item() == pytest.approx(0.401366)
