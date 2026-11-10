"""`[RULES]`: `evaluate_rules` and `take_actions` (EPANET 2.2 `rules.c`) as functions of plain
numbers, and the closure's `apply_rules` on a solved network."""

import dataclasses
import itertools
from pathlib import Path

import pytest
import torch

from noodl.apps.water.controls import (
    ACTIVE,
    CLOSED,
    OPEN,
    Action,
    Control,
    Premise,
    Rule,
    apply_controls,
)
from noodl.apps.water.inp import PSI_TO_M, read_epanet_inp
from noodl.apps.water.network import (
    Junction,
    Pump,
    Reservoir,
    Valve,
    WaterNetwork,
    WaterPipe,
    apply_rules,
    build_model,
    water_steady,
)
from noodl.apps.water.rules import evaluate_rules, take_actions

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "water"


def _t(x):
    return torch.tensor(x, dtype=F64)


def _premise(conjunction, attribute, relation, value, *, object="NODE", id="J1", tolerance=1e-3):
    return Premise(conjunction, object, id, attribute, relation, value, tolerance)


def _rule(name, premises, then=(), otherwise=(), priority=0.0):
    return Rule(name, tuple(premises), tuple(then), tuple(otherwise), priority)


CLOSE_P2 = Action("P2", "STATUS", float(CLOSED))
OPEN_P2 = Action("P2", "STATUS", float(OPEN))
SNAPSHOT = {"PRESSURE": {"J1": 50.0}, "LEVEL": {"T1": 10.0}, "HEAD": {"J1": 50.0},
            "FLOW": {"P1": 0.01}, "STATUS": {"P1": float(OPEN), "V1": float(ACTIVE)},
            "SETTING": {"V1": 30.0}, "DEMAND": {"J1": 0.005, "": 0.005}}


def _fires(premises, snapshot=SNAPSHOT, **kw):
    kw.setdefault("time", 0.0)
    return evaluate_rules([_rule("R", premises, then=[CLOSE_P2])], snapshot, **kw) == [CLOSE_P2]


# ------------------------------------------------------------------ the premises' logic
TRUE = ("PRESSURE", ">", 0.0)
FALSE = ("PRESSURE", "<", 0.0)

#: EPANET 2.2 itself (wntr 1.5.0, a rule closing a link when its premises hold), for
#: `IF a <op1> b <op2> c` over the truth values (a, b, c): the rows of its truth table.
EPANET_AND_OR = {(0, 0, 0): 0, (0, 0, 1): 0, (0, 1, 0): 0, (0, 1, 1): 0,
                 (1, 0, 0): 0, (1, 0, 1): 1, (1, 1, 0): 1, (1, 1, 1): 1}
EPANET_OR_AND = {(0, 0, 0): 0, (0, 0, 1): 0, (0, 1, 0): 0, (0, 1, 1): 1,
                 (1, 0, 0): 0, (1, 0, 1): 1, (1, 1, 0): 0, (1, 1, 1): 1}


def _chain(values, ops):
    premises = [_premise(conj, *(TRUE if v else FALSE))
                for conj, v in zip(("IF", *ops), values, strict=True)]
    return _fires(premises)


def test_premises_follow_epanets_own_logic_not_the_textbook_precedence():
    """`evalpremises` (`rules.c`) is NOT "AND binds tighter than OR", whatever the manual says:
    a false result at an AND ends the rule false, so `IF a AND b OR c` with a false is false
    (c is never reached), and `IF a OR b AND c` is `(a OR b) AND c`. The tables are EPANET's,
    measured on wntr 1.5.0's EPANET 2.2 (a probe network, a rule closing a pipe); the rule
    here is transcribed, so it reproduces them."""
    for values in itertools.product((0, 1), repeat=3):
        assert _chain(values, ("AND", "OR")) is bool(EPANET_AND_OR[values]), values
        assert _chain(values, ("OR", "AND")) is bool(EPANET_OR_AND[values]), values
        assert _chain(values, ("OR", "OR")) is any(values), values
        assert _chain(values, ("AND", "AND")) is all(values), values


def test_the_cases_where_epanet_and_the_precedence_rule_agree():
    # a AND b OR c with a true: (a AND b) OR c
    assert _chain((1, 0, 1), ("AND", "OR")) and not _chain((1, 0, 0), ("AND", "OR"))
    # a OR b AND c with a false: a OR (b AND c)
    assert _chain((0, 1, 1), ("OR", "AND")) and not _chain((0, 1, 0), ("OR", "AND"))


def test_else_actions():
    rule = _rule("R", [_premise("IF", "PRESSURE", ">", 60.0)], then=[CLOSE_P2],
                 otherwise=[OPEN_P2])
    assert evaluate_rules([rule], SNAPSHOT, time=0.0) == [OPEN_P2]
    assert evaluate_rules([rule], {**SNAPSHOT, "PRESSURE": {"J1": 70.0}}, time=0.0) == [CLOSE_P2]
    # no ELSE: nothing
    bare = _rule("R", [_premise("IF", "PRESSURE", ">", 60.0)], then=[CLOSE_P2])
    assert evaluate_rules([bare], SNAPSHOT, time=0.0) == []


def test_priority_resolves_conflicts():
    """`updateactionlist`/`onactionlist`: among actions on one link the rule of the highest
    PRIORITY wins, a tie goes to the rule listed first, and a rule's own second action on a
    link it already acts on is dropped."""
    yes = [_premise("IF", "PRESSURE", ">", 0.0)]
    a = _rule("A", yes, then=[CLOSE_P2], priority=1.0)
    b = _rule("B", yes, then=[OPEN_P2], priority=2.0)
    c = _rule("C", yes, then=[CLOSE_P2], priority=2.0)
    assert evaluate_rules([a, b], SNAPSHOT, time=0.0) == [OPEN_P2]
    assert evaluate_rules([b, a], SNAPSHOT, time=0.0) == [OPEN_P2]
    assert evaluate_rules([b, c], SNAPSHOT, time=0.0) == [OPEN_P2]   # tie: first listed
    assert evaluate_rules([c, b], SNAPSHOT, time=0.0) == [CLOSE_P2]
    twice = _rule("D", yes, then=[CLOSE_P2, OPEN_P2])
    assert evaluate_rules([twice], SNAPSHOT, time=0.0) == [CLOSE_P2]
    # actions on different links do not compete
    other = Action("P3", "STATUS", float(CLOSED))
    assert evaluate_rules([a, _rule("E", yes, then=[other])], SNAPSHOT, time=0.0) == [
        CLOSE_P2, other]
    # a rule whose premises fail takes no part in the conflict
    no = _rule("F", [_premise("IF", "PRESSURE", "<", 0.0)], then=[OPEN_P2], priority=9.0)
    assert evaluate_rules([a, no], SNAPSHOT, time=0.0) == [CLOSE_P2]


def test_control_and_rule_on_one_link():
    """EPANET's order within a step: the simple controls, then the rules; a rule's action
    stands over a control's on the same link."""
    status, setting = _t([OPEN, OPEN]), _t([0.0, 1.0])
    control = Control("P1", CLOSED, None, "time", value=0.0)
    status, setting, _ = apply_controls(
        [control], time=_t(0.0), status=status, setting=setting, link_names=["P1", "PU"],
        link_kinds=["PIPE", "PUMP"])
    assert status[0].item() == CLOSED
    rule = _rule("R", [_premise("IF", "PRESSURE", ">", 0.0)],
                 then=[Action("P1", "STATUS", float(OPEN))])
    (action,) = evaluate_rules([rule], SNAPSHOT, time=0.0)
    status, setting, changed = take_actions(
        [action], status=status, setting=setting, link_names=["P1", "PU"],
        link_kinds=["PIPE", "PUMP"])
    assert status[0].item() == OPEN and changed


# ------------------------------------------------------------------ premises on values
def test_value_premises_use_epanets_tolerance_of_a_thousandth():
    """`checkvalue`: `=` within 1e-3 of the value, `<>` beyond it, `<` and `>=` with the
    margin on the lenient side of the test, in the premise's own unit."""
    at = lambda relation, x, value=10.0: _fires(  # noqa: E731
        [_premise("IF", "LEVEL", relation, value, id="T1")],
        {**SNAPSHOT, "LEVEL": {"T1": x}})
    assert at("=", 10.0009) and not at("=", 10.0011)
    assert at("<>", 10.0011) and not at("<>", 10.0009)
    assert at("<", 10.0009) and not at("<", 10.0011)      # x > v + tol fails
    assert at("<=", 9.9989) and not at("<=", 9.9991)      # x > v - tol fails
    assert at(">", 9.9991) and not at(">", 9.9989)        # x < v - tol fails
    assert at(">=", 10.0011) and not at(">=", 10.0009)    # x < v + tol fails
    # the tolerance is the premise's: a thousandth of a file unit, in SI
    wide = Premise("IF", "NODE", "T1", "LEVEL", "=", 10.0, 0.3048e-3 * 100)
    assert evaluate_rules([_rule("R", [wide], then=[CLOSE_P2])],
                          {**SNAPSHOT, "LEVEL": {"T1": 10.03}}, time=0.0) == [CLOSE_P2]


def test_flow_is_absolute_status_has_three_values_and_a_missing_setting_is_false():
    flow = lambda value, x: _fires(  # noqa: E731
        [_premise("IF", "FLOW", ">", value, object="LINK", id="P1")],
        {**SNAPSHOT, "FLOW": {"P1": x}})
    assert flow(0.005, -0.01) and not flow(0.005, -0.001)
    status = lambda relation, code, link="P1": _fires(  # noqa: E731
        [_premise("IF", "STATUS", relation, float(code), object="LINK", id=link)])
    assert status("=", OPEN) and not status("=", CLOSED) and status("<>", CLOSED)
    assert status("=", ACTIVE, "V1") and status("=", OPEN, "V1") is False
    # a valve in any state but CLOSED or ACTIVE reads as OPEN; relations other than IS and
    # NOT never hold for a status
    assert not status("<", CLOSED)
    setting = lambda x: _fires(  # noqa: E731
        [_premise("IF", "SETTING", "=", 30.0, object="LINK", id="V1")],
        {**SNAPSHOT, "SETTING": {"V1": x}})
    assert setting(30.0) and not setting(float("nan"))


def test_system_demand_and_node_attributes():
    small = 1e-9   # the premise's tolerance in m3/s: a thousandth of a file unit is not 1e-3
    assert _fires([Premise("IF", "SYSTEM", "", "DEMAND", ">", 0.004, small)])
    assert not _fires([Premise("IF", "SYSTEM", "", "DEMAND", ">", 0.006, small)])
    assert _fires([_premise("IF", "HEAD", "<", 51.0)])
    assert _fires([_premise("IF", "DEMAND", "=", 0.005, tolerance=small)])
    with pytest.raises(ValueError, match="DAY"):
        _fires([Premise("IF", "SYSTEM", "", "DAY", ">", 3.0)])


def _time_fires(relation, value, *, time, step, clock="TIME", start=0.0):
    return _fires([Premise("IF", "SYSTEM", "", clock, relation, value)], time=time,
                  step=step, clocktime=start)


def test_time_premises_test_the_interval_the_rule_step_covers():
    """`checktime`: the interval is `(Htime - dt + 1, Htime]`; `=` holds when the value falls
    in it, an inequality tests its end."""
    assert _time_fires("=", 3600.0, time=3600.0, step=300.0)
    assert _time_fires("=", 3500.0, time=3600.0, step=300.0)
    assert not _time_fires("=", 3600.0, time=3900.0, step=300.0)    # the next interval
    assert not _time_fires("=", 3300.0, time=3600.0, step=300.0)    # 3301..3600
    assert _time_fires("<>", 3300.0, time=3600.0, step=300.0)
    assert _time_fires(">", 3000.0, time=3600.0, step=300.0)
    assert not _time_fires(">", 3600.0, time=3600.0, step=300.0)
    assert _time_fires("<", 3601.0, time=3600.0, step=300.0)
    assert _time_fires("<=", 3600.0, time=3600.0, step=300.0)
    assert _time_fires(">=", 3600.0, time=3600.0, step=300.0)
    assert not _time_fires(">=", 3601.0, time=3600.0, step=300.0)


def test_clocktime_premises_wrap_midnight():
    """`checktime`: the clock time is `(t + Tstart) % 86400`; an interval across midnight
    holds an equality at either side of it."""
    start = 20 * 3600.0
    # six hours after a start at 20:00 it is 02:00; the interval's equality at 01:59 holds
    # when the interval 01:55..02:00 holds it
    assert _time_fires("=", 2 * 3600.0 - 60.0, time=6 * 3600.0, step=300.0, clock="CLOCKTIME",
                       start=start)
    assert not _time_fires("=", 3 * 3600.0, time=6 * 3600.0, step=300.0, clock="CLOCKTIME",
                           start=start)
    # across midnight: the interval 23:55:01..00:00 of a 5-minute step ending at 00:00
    assert _time_fires("=", 23 * 3600.0 + 59 * 60.0, time=4 * 3600.0, step=300.0,
                       clock="CLOCKTIME", start=start)
    assert _time_fires("=", 0.0, time=4 * 3600.0, step=300.0, clock="CLOCKTIME", start=start)
    assert not _time_fires("=", 12 * 3600.0, time=4 * 3600.0, step=300.0, clock="CLOCKTIME",
                           start=start)


# ------------------------------------------------------------------ taking the actions
NAMES, KINDS = ["P1", "PU", "V1", "F1"], ["PIPE", "PUMP", "VALVE", "VALVE"]


def _take(actions, status, setting, **kw):
    out = take_actions(actions, status=_t(status), setting=_t(setting), link_names=NAMES,
                       link_kinds=KINDS, **kw)
    return out[0].tolist(), out[1].tolist(), out[2]


def test_status_actions_as_takeactions_applies_them():
    """`takeactions`: OPEN acts only on a closed link (a pump then runs at speed 1), CLOSED
    only on a link that is not closed (a pump's speed 0); an action that finds the link as
    it says is no change; a pipe's setting is its status."""
    start = ([OPEN, CLOSED, ACTIVE, ACTIVE], [0.0, 0.5, 30.0, 0.1])
    status, setting, changed = _take([Action("PU", "STATUS", float(OPEN))], *start)
    assert (status[1], setting[1], changed) == (OPEN, 1.0, True)
    status, setting, changed = _take([Action("P1", "STATUS", float(OPEN))], *start)
    assert not changed
    status, setting, changed = _take([Action("V1", "STATUS", float(OPEN))], *start)
    assert not changed    # an ACTIVE valve is "not closed": OPEN does nothing
    status, setting, changed = _take([Action("V1", "STATUS", float(CLOSED))], *start)
    assert (status[2], setting[2], changed) == (CLOSED, 30.0, True)
    status, setting, changed = _take([Action("PU", "STATUS", float(CLOSED))], *start)
    assert not changed
    status, setting, changed = _take([Action("P1", "SETTING", 0.0)], *start)
    assert (status[0], changed) == (CLOSED, True)
    status, setting, changed = _take([Action("P1", "SETTING", 1.0)], *start)
    assert not changed


def test_setting_actions_with_epanets_tolerance_and_units():
    start = ([OPEN, OPEN, ACTIVE, ACTIVE], [0.0, 1.0, 30.0, 0.1])
    tolerance = [1e-3, 1e-3, 3.048e-4, 2.83e-5]
    scale = [1.0, 1.0, 9806.65, 1.0]
    # a valve's head setting, to the layer's potential unit; ACTIVE at it
    status, setting, changed = _take([Action("V1", "SETTING", 40.0)], *start,
                                     setting_scale=scale, setting_tolerance=tolerance)
    assert (status[2], setting[2], changed) == (ACTIVE, 40.0 * 9806.65, True)
    # within EPANET's 1e-3 (internal units) of the present setting: no change
    _, _, changed = _take([Action("V1", "SETTING", 30.0 + 1e-4)], *start,
                          setting_scale=[1.0] * 4, setting_tolerance=tolerance)
    assert not changed
    _, _, changed = _take([Action("F1", "SETTING", 0.1 + 1e-5)], *start,
                          setting_tolerance=tolerance)
    assert not changed
    _, _, changed = _take([Action("F1", "SETTING", 0.1 + 1e-4)], *start,
                          setting_tolerance=tolerance)
    assert changed
    # a pump: the speed; 0 closes it, a positive speed opens a closed one
    status, setting, _ = _take([Action("PU", "SETTING", 0.8)], *start)
    assert (status[1], setting[1]) == (OPEN, 0.8)
    status, setting, _ = _take([Action("PU", "SETTING", 0.0)], *start)
    assert (status[1], setting[1]) == (CLOSED, 0.0)
    status, setting, _ = _take([Action("PU", "SETTING", 0.8)],
                               [OPEN, CLOSED, ACTIVE, ACTIVE], [0.0, 1.0, 30.0, 0.1])
    assert (status[1], setting[1]) == (OPEN, 0.8)
    # a rule that sets what is already set changes nothing
    _, _, changed = _take([Action("PU", "SETTING", 1.0)], *start)
    assert not changed


def test_a_valve_given_a_setting_after_a_close_is_active_again():
    status, setting, _ = _take([Action("V1", "SETTING", 25.0)],
                               [OPEN, OPEN, CLOSED, ACTIVE], [0.0, 1.0, 30.0, 0.1])
    assert (status[2], setting[2]) == (ACTIVE, 25.0)


# ------------------------------------------------------------------ in a network
def _two_source(rules, **kw):
    """R1 (50 m) - P1 - J1 (5 L/s) - P2 - R2 (45 m)."""
    pipes = (WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),
             WaterPipe("P2", "J1", "R2", 500.0, 0.3, 130.0))
    return WaterNetwork(junctions=(Junction("J1", 0.0, 0.005),),
                        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 45.0)),
                        pipes=pipes, rules=tuple(rules), **kw)


def test_a_rule_closes_a_link_after_a_solve():
    rule = _rule("R1", [_premise("IF", "PRESSURE", ">", 40.0)], then=[CLOSE_P2])
    net = _two_source([rule])
    net.validate()
    model, state, drivers = build_model(net)
    assert model.control_closure is not None
    state = water_steady(model, state, drivers)
    assert float(state["water.q"][1]) != 0.0
    state, changed = apply_rules(model, state, drivers, time=0.0)
    assert bool(changed) and state["water.link_status"][1].item() == CLOSED
    state = water_steady(model, state, drivers)
    assert float(state["water.q"][1]) == 0.0
    assert float(state["water.q"][0]) == pytest.approx(0.005, abs=1e-9)
    # nothing more to do: the link is closed
    _, changed = apply_rules(model, state, drivers, time=0.0)
    assert not bool(changed)


def test_rules_read_heads_levels_flows_and_the_status_a_solve_found():
    """HEAD, PRESSURE, FLOW and STATUS premises read the solved state; the junction is 0 m
    up and held near 47 m."""
    net = _two_source([
        _rule("A", [_premise("IF", "HEAD", ">", 46.0),
                    _premise("AND", "FLOW", ">", 0.001, object="LINK", id="P1"),
                    _premise("AND", "STATUS", "=", float(OPEN), object="LINK", id="P2")],
              then=[CLOSE_P2])])
    model, state, drivers = build_model(net)
    state = water_steady(model, state, drivers)
    head = float(state["water.phi"][model.node_names.index("J1")])
    assert head > 46.0
    state, changed = apply_rules(model, state, drivers, time=0.0)
    assert bool(changed) and state["water.link_status"][1].item() == CLOSED


def test_a_rule_that_does_not_hold_changes_nothing_and_batches_are_independent():
    rule = _rule("R1", [_premise("IF", "PRESSURE", ">", 40.0)], then=[CLOSE_P2])
    model, state, drivers = build_model(_two_source([rule]))
    state = water_steady(model, state, drivers)
    j = model.node_names.index("J1")
    low = {k: v.clone() for k, v in state.items()}
    low["water.phi"][j] = 10.0
    batch = {k: torch.stack([state[k], low[k]]) for k in state}
    out, changed = apply_rules(model, batch, drivers, time=0.0)
    assert changed.tolist() == [True, False]
    assert out["water.link_status"][:, 1].tolist() == [CLOSED, OPEN]


def test_a_rule_setting_on_a_darcy_weisbach_prv_is_in_the_layers_unit():
    """The rule's setting is in metres; the layer's potential on D-W is pressure (`rho g`
    times the head), which the setting is put in (Task 8's control bug, for rules)."""
    def network(setting, rules=()):
        return WaterNetwork(
            junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.002)),
            reservoirs=(Reservoir("R1", 50.0),),
            pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),),
            valves=(Valve("V1", "A", "B", 0.2, "PRV", setting),), rules=rules, headloss="D-W")
    rule = _rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)],
                 then=[Action("V1", "SETTING", 30.0)])
    built, bstate, bdrivers = build_model(network(30.0))
    want = float(water_steady(built, bstate, bdrivers)["water.phi"][built.node_names.index("B")])
    model, state, drivers = build_model(network(40.0, (rule,)))
    state = water_steady(model, state, drivers)
    state, changed = apply_rules(model, state, drivers, time=0.0)
    assert bool(changed)
    assert state["water.link_status"][1].item() == ACTIVE
    got = float(water_steady(model, state, drivers)["water.phi"][model.node_names.index("B")])
    assert got == pytest.approx(want, rel=1e-9)
    assert got == pytest.approx(30.0 * model.head_scale, rel=1e-6)
    # the same setting again is within EPANET's tolerance: no change
    _, changed = apply_rules(model, state, drivers, time=0.0)
    assert not bool(changed)


def test_a_rule_runs_a_pump_at_its_setting_in_a_network():
    pipes = (WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),
             WaterPipe("P2", "B", "J1", 400.0, 0.3, 130.0))
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.0), Junction("J1", 10.0, 0.004)),
        reservoirs=(Reservoir("R1", 10.0),), pipes=pipes,
        pumps=(Pump("PU", "A", "B", "C1"),),
        curves={"C1": ((0.0, 60.0), (0.01, 50.0), (0.02, 30.0))},
        rules=(_rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)],
                     then=[Action("PU", "SETTING", 0.5)]),))
    model, state, drivers = build_model(net)
    state, changed = apply_rules(model, state, drivers, time=0.0)
    j = model.link_names.index("PU")
    assert bool(changed) and state["water.link_setting"][j].item() == 0.5
    slow = water_steady(model, state, drivers)
    base_model, base_state, base_drivers = build_model(dataclasses.replace(net, rules=()))
    fast = water_steady(base_model, base_state, base_drivers)
    # the demand is fixed, so the flow is the same: the pump's head gain is what differs
    at = model.node_names.index("J1")
    assert float(slow["water.phi"][at]) < float(fast["water.phi"][at]) - 1.0


def test_apply_rules_without_rules_is_a_no_op():
    model, state, drivers = build_model(_two_source([]))
    out, changed = apply_rules(model, state, drivers, time=0.0)
    assert out is state and not bool(changed)


# ------------------------------------------------------------------ refusals, by name
def test_the_network_names_what_a_rule_cannot_say():
    yes = [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)]
    cases = [
        (_rule("R", yes, then=[Action("P1", "STATUS", float(ACTIVE))]), "ACTIVE"),
        (_rule("R", yes, then=[Action("V1", "STATUS", float(ACTIVE))]), "ACTIVE"),
        (_rule("R", yes, then=[Action("P1", "SETTING", -1.0)]), "negative"),
        (_rule("R", yes, then=[Action("G1", "SETTING", 1.0)]), "GPV"),
        (_rule("R", [_premise("IF", "DEMAND", ">", 0.0, id="R1")], then=[CLOSE_P2]),
         "reservoir"),
        (_rule("R", [Premise("IF", "SYSTEM", "", "DAY", ">", 1.0)], then=[CLOSE_P2]), "DAY"),
        (_rule("R", [_premise("IF", "POWER", ">", 1.0, object="LINK", id="P1")],
               then=[CLOSE_P2]), "POWER"),
    ]
    for rule, word in cases:
        net = WaterNetwork(
            junctions=(Junction("A", 0.0, 0.0), Junction("J1", 0.0, 0.005)),
            reservoirs=(Reservoir("R1", 50.0),),
            pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),
                   WaterPipe("P2", "A", "J1", 100.0, 0.3, 130.0)),
            valves=(Valve("V1", "A", "J1", 0.2, "TCV", 5.0),
                    Valve("G1", "A", "J1", 0.2, "GPV", 0.0, curve="C")),
            curves={"C": ((0.0, 0.0), (1.0, 1.0))}, rules=(rule,))
        with pytest.raises(ValueError, match=word):
            net.validate()


def test_a_check_valve_cannot_be_the_object_of_a_rule_action():
    """EPANET error 207: a CV cannot be controlled."""
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.005),), reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "J1", 100.0, 0.3, 130.0, status="CV"),),
        rules=(_rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)],
                     then=[Action("P1", "STATUS", float(CLOSED))]),))
    with pytest.raises(ValueError, match="check valve"):
        net.validate()


def test_a_demand_premise_is_refused_on_a_pressure_driven_model():
    rule = _rule("R", [_premise("IF", "DEMAND", ">", 0.0)], then=[CLOSE_P2])
    with pytest.raises(ValueError, match="DEMAND"):
        build_model(_two_source([rule]), pda=True)


def test_a_rule_that_acts_on_a_pipe_of_the_potential_form_is_refused():
    rule = _rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)], then=[CLOSE_P2])
    with pytest.raises(ValueError, match="pipe_form"):
        build_model(_two_source([rule], headloss="D-W"), friction="colebrook")


# ------------------------------------------------------------------ the reader
def _edit(tmp_path, name, *pairs):
    text = (DATA / name).read_text()
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new, 1)
    path = tmp_path / name
    path.write_text(text)
    return path


def test_trailing_tokens_after_a_premise_value_are_refused(tmp_path):
    for premise in ("NODE J1 PRESSURE ABOVE 20 PSI", "LINK P2 STATUS IS OPEN NOW",
                    "SYSTEM DEMAND > 1 GPM"):
        rules = f"[RULES]\nRULE R\nIF {premise}\nTHEN LINK P3 STATUS IS OPEN\n\n[CURVES]"
        with pytest.raises(ValueError, match="after the value"):
            read_epanet_inp(_edit(tmp_path, "twoloop_si.inp", ("[CURVES]", rules)))


def test_a_premise_carries_epanets_tolerance_in_si(tmp_path):
    """A thousandth of a file unit: of a psi in a US file, of a metre in an SI one."""
    rules = ("[RULES]\nRULE R\nIF NODE J1 PRESSURE ABOVE 20\nAND SYSTEM DEMAND > 1\n"
             "AND LINK P2 STATUS IS OPEN\nTHEN LINK P3 STATUS IS OPEN\n\n[CURVES]")
    net = read_epanet_inp(_edit(tmp_path, "twoloop_si.inp", ("[CURVES]", rules)))
    pressure, demand, status = net.rules[0].premises
    assert pressure.tolerance == pytest.approx(1e-3 * (pressure.value / 20.0))
    assert demand.tolerance == pytest.approx(1e-3 * demand.value)
    us = ("[RULES]\nRULE R\nIF NODE 10 PRESSURE ABOVE 20\nTHEN LINK 11 STATUS IS OPEN\n\n"
          "[CURVES]")
    net = read_epanet_inp(_edit(tmp_path, "Net1.inp", ("[CURVES]", us)))
    assert net.rules[0].premises[0].tolerance == pytest.approx(1e-3 * PSI_TO_M)
