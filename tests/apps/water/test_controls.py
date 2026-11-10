"""EPANET's simple controls: `apply_controls` and `next_control_time` as functions, and the
closure that carries a link's status and setting across steps (`TankLevels`)."""

import pytest
import torch

from noodl.apps.water.controls import (
    ACTIVE,
    CLOSED,
    HTOL,
    OPEN,
    Control,
    apply_controls,
    next_control_time,
)
from noodl.apps.water.network import (
    Junction,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterPipe,
    build_model,
    water_event_step,
    water_steady,
)

F64 = torch.float64
LINKS = ["P1", "PU", "V1"]
KINDS = ["PIPE", "PUMP", "VALVE"]


def _t(x):
    return torch.tensor(x, dtype=F64)


def _apply(controls, **kw):
    kw.setdefault("status", _t([OPEN, OPEN, ACTIVE]))
    kw.setdefault("setting", _t([0.0, 1.0, 30.0]))
    return apply_controls(controls, link_names=LINKS, link_kinds=KINDS, **kw)


def test_time_control_fires_at_its_time():
    control = Control("P1", CLOSED, None, "time", value=3600.0)
    for now, closed in ((0.0, False), (3599.0, False), (3600.0, True), (3601.0, False)):
        status, _, changed = _apply([control], time=_t(now))
        assert (status[0].item() == CLOSED) is closed
        assert bool(changed) is closed
    # EPANET shortens the step to the control: the time to it
    assert float(next_control_time([control], time=_t(0.0), levels={}, rates={})) == 3600.0
    assert float(next_control_time([control], time=_t(1800.0), levels={}, rates={})) == 1800.0
    assert next_control_time([control], time=_t(3600.0), levels={}, rates={}) == torch.inf


def test_clocktime_control_respects_start_clocktime():
    """`AT CLOCKTIME 6:00` with the simulation starting at 20:00 fires 10 h in, and again 24
    h later (`(Htime + Tstart) % 86400 == Time`)."""
    control = Control("P1", CLOSED, None, "clocktime", value=6 * 3600.0)
    start = 20 * 3600.0
    for now, closed in ((0.0, False), (9 * 3600.0, False), (10 * 3600.0, True),
                        (34 * 3600.0, True), (11 * 3600.0, False)):
        status, _, _ = _apply([control], time=_t(now), clocktime=start)
        assert (status[0].item() == CLOSED) is closed, now
    ahead = next_control_time([control], time=_t(0.0), clocktime=start, levels={}, rates={})
    assert float(ahead) == 10 * 3600.0
    # past it, the next one is a day after
    ahead = next_control_time([control], time=_t(11 * 3600.0), clocktime=start, levels={},
                              rates={})
    assert float(ahead) == 23 * 3600.0
    # without a start clock time, `AT CLOCKTIME 6:00` is `AT TIME 6:00`
    assert float(next_control_time([control], time=_t(0.0), levels={}, rates={})) == 21600.0


def test_level_control_both_directions():
    """BELOW opens when the level is under the trigger, ABOVE closes when it is over it, and
    a level between them leaves the status alone (the pair is a hysteresis)."""
    controls = [Control("PU", OPEN, None, "node", "T", False, 10.0),
                Control("PU", CLOSED, None, "node", "T", True, 14.0)]
    for level, expected in ((9.0, OPEN), (12.0, CLOSED), (15.0, CLOSED)):
        status, _, changed = _apply(controls, levels={"T": _t(level)},
                                    status=_t([OPEN, CLOSED, ACTIVE]))
        assert status[1].item() == (OPEN if level < 10.0 else CLOSED)
        assert status[1].item() == expected or level == 12.0
        assert bool(changed) is (level < 10.0)
    # a falling level reaches BELOW 10 in (12 - 10) / 0.5 s; a rising one ABOVE 14 in 4 s
    falling = next_control_time(controls, time=_t(0.0), levels={"T": _t(12.0)},
                                rates={"T": _t(-0.5)})
    rising = next_control_time(controls, time=_t(0.0), levels={"T": _t(12.0)},
                               rates={"T": _t(0.5)})
    assert float(falling) == pytest.approx(4.0)
    assert float(rising) == pytest.approx(4.0)
    # a level moving away from a trigger does not reach it
    away = next_control_time(controls[:1], time=_t(0.0), levels={"T": _t(12.0)},
                             rates={"T": _t(0.5)})
    assert away == torch.inf


def test_batch_instances_are_independent():
    control = Control("P1", CLOSED, None, "node", "T", True, 10.0)
    status, _, changed = _apply([control], levels={"T": _t([9.0, 11.0])})
    assert status[:, 0].tolist() == [OPEN, CLOSED] and changed.tolist() == [False, True]


def test_pressure_control_needs_a_resolve():
    """A junction-pressure control is tested after a solve, against its setting within EPANET's
    `Htol` (`pswitch`), and a change asks for another solve (`changed`). In a network, the
    solve that follows leaves the link shut with no flow."""
    control = Control("P1", CLOSED, None, "node", "J", False, 20.0)
    for pressure, fires in ((20.0 + 0.5 * HTOL, True), (20.0 + 2 * HTOL, False)):
        status, _, changed = _apply([control], pressures={"J": _t(pressure)})
        assert bool(changed) is fires and (status[0].item() == CLOSED) is fires
    above = Control("P1", CLOSED, None, "node", "J", True, 20.0)
    assert bool(_apply([above], pressures={"J": _t(20.0 - 0.5 * HTOL)})[2])
    assert not bool(_apply([above], pressures={"J": _t(20.0 - 2 * HTOL)})[2])
    # without pressures the control is not tested: it acts after a solve, never before
    assert not bool(_apply([control], time=_t(0.0), levels={"T": _t(0.0)})[2])

    net = _two_source_network(controls=(
        Control("P2", CLOSED, None, "node", "J1", True, 40.0),))
    model, state, drivers = build_model(net)
    assert model.control_closure is not None
    free, free_state, free_drivers = build_model(_two_source_network())
    open_flows = water_steady(free, free_state, free_drivers)["water.q"]
    assert float(open_flows[1]) != 0.0   # P2 carries water without the control
    out = water_steady(model, state, drivers)
    assert out["water.link_status"][1].item() == CLOSED
    assert float(out["water.q"][1]) == 0.0
    assert float(out["water.q"][0]) == pytest.approx(0.005, abs=1e-9)


def _two_source_network(**kwargs):
    """R1 (50 m) - P1 - J1 (5 L/s) - P2 - R2 (45 m)."""
    pipes = (WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),
             WaterPipe("P2", "J1", "R2", 500.0, 0.3, 130.0))
    return WaterNetwork(junctions=(Junction("J1", 0.0, 0.005),),
                        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 45.0)),
                        pipes=pipes, **kwargs)


def test_setting_action_activates_a_valve():
    """A numeric setting on a valve makes it ACTIVE at that setting (an OPEN or CLOSED valve
    keeps its setting); on a pump it is the relative speed, and 0 closes it; OPEN runs a pump
    at speed 1."""
    controls = [Control("V1", None, 55.0, "time", value=10.0),
                Control("PU", None, 0.0, "time", value=20.0),
                Control("PU", None, 0.8, "time", value=30.0),
                Control("PU", OPEN, None, "time", value=40.0),
                Control("V1", CLOSED, None, "time", value=50.0)]
    start = dict(status=_t([OPEN, OPEN, OPEN]), setting=_t([0.0, 0.5, 30.0]))
    status, setting, changed = _apply(controls, time=_t(10.0), **start)
    assert (status[2].item(), setting[2].item(), bool(changed)) == (ACTIVE, 55.0, True)
    status, setting, _ = _apply(controls, time=_t(20.0), **start)
    assert (status[1].item(), setting[1].item()) == (CLOSED, 0.0)
    status, setting, _ = _apply(controls, time=_t(30.0), **start)
    assert (status[1].item(), setting[1].item()) == (OPEN, 0.8)
    status, setting, _ = _apply(controls, time=_t(40.0), **start)
    assert (status[1].item(), setting[1].item()) == (OPEN, 1.0)
    status, setting, changed = _apply(controls, time=_t(50.0), **start)
    assert (status[2].item(), setting[2].item()) == (CLOSED, 30.0)
    # nothing changes when the link already is as the control says
    _, _, changed = _apply(controls[3:4], time=_t(40.0), status=_t([OPEN, OPEN, OPEN]),
                           setting=_t([0.0, 1.0, 30.0]))
    assert not bool(changed)
    with pytest.raises(ValueError, match="pump or a valve"):
        _apply([Control("P1", None, 3.0, "time", value=0.0)], time=_t(0.0))


def test_time_controls_act_in_a_network_over_time():
    """`AT TIME` through the closure: the status is carried, so the link stays closed after
    the control's step."""
    net = _two_source_network(controls=(Control("P2", CLOSED, None, "time", value=3600.0),))
    model, state, drivers = build_model(net)
    assert "water.time" in drivers and "water.time" in model.input_specs
    at = lambda t: {**drivers, "water.time": _t(t)}  # noqa: E731
    state = water_steady(model, state, at(0.0))
    assert float(state["water.q"][1]) != 0.0
    state = water_steady(model, state, at(3600.0))
    assert float(state["water.q"][1]) == 0.0 and state["water.link_status"][1] == CLOSED
    state = water_steady(model, state, at(7200.0))
    assert float(state["water.q"][1]) == 0.0


def test_the_event_step_ends_at_the_next_time_control():
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.002),),
        reservoirs=(Reservoir("R1", 50.0),),
        tanks=(Tank("T1", elevation=40.0, init_level=5.0, min_level=0.0, max_level=20.0,
                    diameter=10.0),),
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),
               WaterPipe("P2", "J1", "T1", 100.0, 0.3, 130.0)),
        controls=(Control("P2", CLOSED, None, "time", value=600.0),),
    )
    model, state, drivers = build_model(net)
    at = lambda t: {**drivers, "water.time": _t(t)}  # noqa: E731
    state, dt = water_event_step(model, state, at(0.0), 3600.0)
    assert dt == 600.0 and float(state["water.q"][1]) > 0.0
    state, dt = water_event_step(model, state, at(600.0), 3600.0)
    assert dt == 3600.0 and float(state["water.q"][1]) == 0.0
    assert state["water.link_status"][1] == CLOSED


def test_a_time_control_without_the_time_driver_is_refused_by_name():
    net = _two_source_network(controls=(Control("P2", CLOSED, None, "time", value=3600.0),))
    model, state, drivers = build_model(net)
    drivers = {k: v for k, v in drivers.items() if k != "water.time"}
    with pytest.raises(KeyError, match="water.time"):
        water_steady(model, state, drivers)


def test_a_valve_setting_control_in_a_network():
    """A TCV given a new loss coefficient by a control: the loss across it follows, from the
    step the control fires in, and the setting is carried on."""
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.005)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),),
        valves=(Valve("V1", "A", "B", 0.2, "TCV", 5.0),),
        controls=(Control("V1", None, 500.0, "time", value=600.0),),
    )
    model, state, drivers = build_model(net)
    at = lambda t: {**drivers, "water.time": _t(t)}  # noqa: E731
    state = water_steady(model, state, at(0.0))
    drop = lambda s: float(s["water.phi"][model.node_names.index("A")]  # noqa: E731
                           - s["water.phi"][model.node_names.index("B")])
    before = drop(state)
    state = water_steady(model, state, at(600.0))
    assert state["water.link_status"][1].item() == ACTIVE
    assert state["water.link_setting"][1].item() == 500.0
    assert drop(state) == pytest.approx(before * 100.0, rel=1e-6)
    state = water_steady(model, state, at(1200.0))   # carried on
    assert drop(state) == pytest.approx(before * 100.0, rel=1e-6)


# ------------------------------------------------------------------ review round 1
def _prv_network(setting, controls=()):
    return WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.002)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),),
        valves=(Valve("V1", "A", "B", 0.2, "PRV", setting),),
        controls=controls, headloss="D-W",
    )


def _head(model, state, node):
    return float(state["water.phi"][model.node_names.index(node)])


def test_a_prv_setting_control_on_darcy_weisbach_matches_a_prv_built_at_that_setting():
    """The control's setting is in metres; the layer's potential on D-W is pressure
    (`rho g` times the head), which the closure puts it in."""
    built, bstate, bdrivers = build_model(_prv_network(30.0))
    want = _head(built, water_steady(built, bstate, bdrivers), "B")
    net = _prv_network(40.0, (Control("V1", None, 30.0, "time", value=0.0),))
    model, state, drivers = build_model(net)
    got = _head(model, water_steady(model, state, {**drivers, "water.time": _t(0.0)}), "B")
    assert got == pytest.approx(want, rel=1e-9)
    assert got == pytest.approx(30.0 * model.head_scale, rel=1e-6)


def test_a_no_op_setting_control_on_darcy_weisbach_does_not_end_a_step():
    """A setting control that sets the value the valve has is no change, on a pressure layer
    too, so it ends no step (`controltimestep`)."""
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.002)),
        reservoirs=(Reservoir("R1", 50.0),),
        tanks=(Tank("T1", elevation=40.0, init_level=5.0, min_level=0.0, max_level=20.0,
                    diameter=10.0),),
        pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),
               WaterPipe("P2", "B", "T1", 100.0, 0.3, 130.0)),
        valves=(Valve("V1", "A", "B", 0.2, "PRV", 30.0),),
        controls=(Control("V1", None, 30.0, "time", value=600.0),),
        headloss="D-W",
    )
    model, state, drivers = build_model(net)
    _, dt = water_event_step(model, state, {**drivers, "water.time": _t(0.0)}, 3600.0)
    assert dt == 3600.0


def test_a_control_on_a_reservoir_or_a_numeric_setting_on_a_gpv_is_refused():
    with pytest.raises(ValueError, match="reservoir 'R1'"):
        _two_source_network(controls=(Control("P2", CLOSED, None, "node", "R1", True,
                                              1.0),)).validate()
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.002)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),),
        valves=(Valve("V1", "A", "B", 0.2, "GPV", 0.0, curve="C"),),
        controls=(Control("V1", None, 1.0, "time", value=0.0),),
    )
    with pytest.raises(ValueError, match="not a pipe or a GPV"):
        net.validate()


def test_the_closure_needs_the_kinds_of_the_links_it_controls_unless_they_are_pumps():
    from noodl.apps.water.tanks import TankLevels
    tank = Tank("T1", elevation=0.0, init_level=1.0, min_level=0.0, max_level=2.0,
                diameter=1.0)
    kwargs = dict(tank_nodes=[0], dt=3600.0, reservoir_heads=torch.zeros(0, dtype=F64),
                  link_names=["P1", "V1"], link_status=torch.ones(2, dtype=F64))
    control = Control("V1", CLOSED, None, "node", "T1", True, 1.0)
    with pytest.raises(ValueError, match="link_kinds"):
        TankLevels(None, [tank], [control], [], **kwargs)
    TankLevels(None, [tank], [control], [], link_kinds=["PIPE", "VALVE"], **kwargs)


def test_the_pressure_switch_cap_names_the_links_and_junctions():
    """Two pressure controls that undo each other never settle."""
    net = _two_source_network(controls=(
        Control("P2", CLOSED, None, "node", "J1", False, 49.0),
        Control("P2", OPEN, None, "node", "J1", True, 49.0),))
    model, state, drivers = build_model(net)
    with pytest.raises(RuntimeError, match=r"\['P2'\].*\['J1'\]"):
        water_steady(model, state, drivers)


def test_gradient_through_a_setting_driver_with_a_control_that_does_not_fire():
    """d (head at B) / d (PRV setting) through `"water.setting"`, against central
    differences, on a network whose closure also carries a control that never fires."""
    net = _prv_network(30.0, (Control("P1", CLOSED, None, "time", value=1e9),))
    model, state, drivers = build_model(net)
    j = model.link_names.index("V1")

    def head(setting):
        setting_driver = drivers["water.setting"].clone()
        setting_driver[j] = setting
        out = water_steady(model, state, {**drivers, "water.setting": setting_driver,
                                          "water.time": _t(0.0)})
        return out["water.phi"][model.node_names.index("B")]

    s0 = torch.tensor(30.0 * model.head_scale, dtype=F64, requires_grad=True)
    value = head(s0)
    (grad,) = torch.autograd.grad(value, s0)
    h = 1e-4
    fd = (head(s0.detach() + h) - head(s0.detach() - h)) / (2 * h)
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
    assert float(grad) == pytest.approx(1.0, rel=1e-4)   # the held head follows its setting
