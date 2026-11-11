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
    next_control_time,
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
from noodl.apps.water.rules import evaluate_rules, take_actions, take_actions_per_link

# A closed link's flow is an unknown of the Newton solve whose row is `kappa q = 0`, so it
# is zero to round-off (flows here are ~1e-2 m3/s, eps ~ 2e-16: well below 1e-15), not
# bit-exactly, and the last bit depends on the BLAS.
ZERO_FLOW = 1e-15

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
    kw.setdefault("step", 300.0)
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
    assert evaluate_rules([rule], SNAPSHOT, time=0.0, step=300.0) == [OPEN_P2]
    high = {**SNAPSHOT, "PRESSURE": {"J1": 70.0}}
    assert evaluate_rules([rule], high, time=0.0, step=300.0) == [CLOSE_P2]
    # no ELSE: nothing
    bare = _rule("R", [_premise("IF", "PRESSURE", ">", 60.0)], then=[CLOSE_P2])
    assert evaluate_rules([bare], SNAPSHOT, time=0.0, step=300.0) == []


def test_priority_resolves_conflicts():
    """`updateactionlist`/`onactionlist`: among actions on one link the rule of the highest
    PRIORITY wins, a tie goes to the rule listed first, and a rule's own second action on a
    link it already acts on is dropped."""
    yes = [_premise("IF", "PRESSURE", ">", 0.0)]
    a = _rule("A", yes, then=[CLOSE_P2], priority=1.0)
    b = _rule("B", yes, then=[OPEN_P2], priority=2.0)
    c = _rule("C", yes, then=[CLOSE_P2], priority=2.0)
    assert evaluate_rules([a, b], SNAPSHOT, time=0.0, step=300.0) == [OPEN_P2]
    assert evaluate_rules([b, a], SNAPSHOT, time=0.0, step=300.0) == [OPEN_P2]
    assert evaluate_rules([b, c], SNAPSHOT, time=0.0, step=300.0) == [OPEN_P2]   # tie: first listed
    assert evaluate_rules([c, b], SNAPSHOT, time=0.0, step=300.0) == [CLOSE_P2]
    twice = _rule("D", yes, then=[CLOSE_P2, OPEN_P2])
    assert evaluate_rules([twice], SNAPSHOT, time=0.0, step=300.0) == [CLOSE_P2]
    # actions on different links do not compete
    other = Action("P3", "STATUS", float(CLOSED))
    assert evaluate_rules([a, _rule("E", yes, then=[other])], SNAPSHOT, time=0.0, step=300.0) == [
        CLOSE_P2, other]
    # a rule whose premises fail takes no part in the conflict
    no = _rule("F", [_premise("IF", "PRESSURE", "<", 0.0)], then=[OPEN_P2], priority=9.0)
    assert evaluate_rules([a, no], SNAPSHOT, time=0.0, step=300.0) == [CLOSE_P2]


def test_control_and_rule_on_one_link():
    """EPANET's order (`ruletimestep` at the end of a step at T, then `runhyd` at T):
    the rules' action is taken first, the simple controls of the next solve after it, so a
    control that fires at T stands over a rule's action on the same link."""
    names, kinds = ["P1", "PU"], ["PIPE", "PUMP"]
    status, setting = _t([CLOSED, OPEN]), _t([0.0, 1.0])
    rule = _rule("R", [_premise("IF", "PRESSURE", ">", 0.0)],
                 then=[Action("P1", "STATUS", float(OPEN))])
    (action,) = evaluate_rules([rule], SNAPSHOT, time=3600.0, step=300.0)
    status, setting, changed = take_actions(
        [action], status=status, setting=setting, link_names=names, link_kinds=kinds)
    assert status[0].item() == OPEN and changed
    control = Control("P1", CLOSED, None, "time", value=3600.0)
    status, setting, _ = apply_controls(
        [control], time=_t(3600.0), status=status, setting=setting, link_names=names,
        link_kinds=kinds)
    assert status[0].item() == CLOSED
    # a control that does not fire at T leaves the rule's action standing
    status, _, _ = apply_controls(
        [Control("P1", CLOSED, None, "time", value=7200.0)], time=_t(3600.0),
        status=_t([OPEN, OPEN]), setting=setting, link_names=names, link_kinds=kinds)
    assert status[0].item() == OPEN


def test_a_pump_closed_by_a_rule_is_at_speed_zero_as_a_closing_control_leaves_it():
    """EPANET's `setlinkstatus` (`hydraul.c`) puts a pump a rule CLOSES at speed 0, as a
    CLOSED control does (`input3.c`, `controldata`): a CLOSED control that follows on the
    same pump then changes nothing and ends no step (`controltimestep`), and an OPEN, from a
    rule or a control, runs it at speed 1 again."""
    names, kinds = ["P1", "PU"], ["PIPE", "PUMP"]
    close = Action("PU", "STATUS", float(CLOSED))
    status, setting, changed = take_actions(
        [close], status=_t([OPEN, OPEN]), setting=_t([0.0, 1.0]), link_names=names,
        link_kinds=kinds)
    assert (status[1].item(), setting[1].item(), changed) == (CLOSED, 0.0, True)
    control = [Control("PU", CLOSED, None, "time", value=1800.0)]
    ahead = next_control_time(control, time=_t(0.0), levels={}, rates={}, status=status,
                              setting=setting, link_names=names, link_kinds=kinds)
    assert ahead == torch.inf
    _, _, changed = apply_controls(control, time=_t(1800.0), status=status, setting=setting,
                                   link_names=names, link_kinds=kinds)
    assert not bool(changed)
    opened, speed, _ = take_actions([Action("PU", "STATUS", float(OPEN))], status=status,
                                    setting=setting, link_names=names, link_kinds=kinds)
    assert (opened[1].item(), speed[1].item()) == (OPEN, 1.0)
    opened, speed, _ = apply_controls([Control("PU", OPEN, None, "time", value=0.0)],
                                      time=_t(0.0), status=status, setting=setting,
                                      link_names=names, link_kinds=kinds)
    assert (opened[1].item(), speed[1].item()) == (OPEN, 1.0)


def test_the_rule_step_must_be_positive():
    """EPANET never checks the rules over an empty interval (`ruletimestep`: a zero first
    increment is made a whole one): with no step a time premise would hold for any time."""
    rule = [_rule("R", [Premise("IF", "SYSTEM", "", "TIME", "=", 99999.0)], then=[CLOSE_P2])]
    assert evaluate_rules(rule, SNAPSHOT, time=3600.0, step=300.0) == []
    for step in (0.0, -1.0):
        with pytest.raises(ValueError, match="step"):
            evaluate_rules(rule, SNAPSHOT, time=3600.0, step=step)
    with pytest.raises(TypeError):
        evaluate_rules(rule, SNAPSHOT, time=3600.0)
    model, state, drivers = build_model(_two_source(rule))
    with pytest.raises(TypeError):
        apply_rules(model, state, drivers, time=3600.0)


def test_a_premise_built_in_python_has_epanets_tolerance_in_si_units():
    """A thousandth of the unit of an SI file (LPS, metres), not 1e-3 m3/s."""
    tol = lambda attribute: Premise("IF", "NODE", "J1", attribute, "=", 0.0).tolerance  # noqa: E731
    assert tol("DEMAND") == tol("FLOW") == 1e-6
    assert tol("HEAD") == tol("LEVEL") == tol("PRESSURE") == tol("SETTING") == 1e-3
    assert Premise("IF", "NODE", "J1", "HEAD", "=", 0.0, 0.5).tolerance == 0.5


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
                          {**SNAPSHOT, "LEVEL": {"T1": 10.03}}, time=0.0, step=300.0) == [CLOSE_P2]


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


def test_the_links_an_action_is_taken_on_are_marked_though_nothing_carried_moved():
    """`take_actions_per_link` marks each link an action was taken on, EPANET's per-action
    `flag` in `takeactions`, which tests the SOLVED status: an OPEN on a pump the solve shut at
    its shut-off head (carried OPEN at speed 1, solved CLOSED) is taken and marked though it
    moves neither the carried status nor the speed, as is a changed positive SETTING on it; a
    CLOSED on it is not. `take_actions`' `changed` is the mask's any."""
    start = dict(status=_t([OPEN, OPEN, ACTIVE, ACTIVE]), setting=_t([0.0, 1.0, 30.0, 0.1]),
                 link_names=NAMES, link_kinds=KINDS, solved=_t([OPEN, CLOSED, ACTIVE, ACTIVE]))
    status, setting, acted = take_actions_per_link([Action("PU", "STATUS", float(OPEN))],
                                                   **start)
    assert acted.tolist() == [False, True, False, False]
    assert status.tolist() == start["status"].tolist() and setting.tolist() == [0.0, 1.0, 30.0,
                                                                                0.1]
    assert take_actions([Action("PU", "STATUS", float(OPEN))], **start)[2] is True
    status, setting, acted = take_actions_per_link([Action("PU", "SETTING", 0.9)], **start)
    assert acted.tolist() == [False, True, False, False] and setting[1].item() == 0.9
    _, _, acted = take_actions_per_link([Action("PU", "STATUS", float(CLOSED))], **start)
    assert not acted.any() and take_actions([Action("PU", "STATUS", float(CLOSED))],
                                            **start)[2] is False
    # two actions, two links; a valve already ACTIVE at its setting is left alone
    _, _, acted = take_actions_per_link(
        [Action("P1", "STATUS", float(CLOSED)), Action("V1", "SETTING", 30.0),
         Action("F1", "SETTING", 0.2)], **start)
    assert acted.tolist() == [True, False, False, True]


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
    state, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
    assert bool(changed) and state["water.link_status"][1].item() == CLOSED
    state = water_steady(model, state, drivers)
    assert abs(float(state["water.q"][1])) <= ZERO_FLOW
    assert float(state["water.q"][0]) == pytest.approx(0.005, abs=1e-9)
    # nothing more to do: the link is closed
    _, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
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
    state, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
    assert bool(changed) and state["water.link_status"][1].item() == CLOSED


def test_a_rule_that_does_not_hold_changes_nothing_and_batches_are_independent():
    rule = _rule("R1", [_premise("IF", "PRESSURE", ">", 40.0)], then=[CLOSE_P2])
    model, state, drivers = build_model(_two_source([rule]))
    state = water_steady(model, state, drivers)
    j = model.node_names.index("J1")
    low = {k: v.clone() for k, v in state.items()}
    low["water.phi"][j] = 10.0
    batch = {k: torch.stack([state[k], low[k]]) for k in state}
    out, changed = apply_rules(model, batch, drivers, time=0.0, step=300.0)
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
    state, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
    assert bool(changed)
    assert state["water.link_status"][1].item() == ACTIVE
    got = float(water_steady(model, state, drivers)["water.phi"][model.node_names.index("B")])
    assert got == pytest.approx(want, rel=1e-9)
    assert got == pytest.approx(30.0 * model.head_scale, rel=1e-6)
    # the same setting again is within EPANET's tolerance: no change
    _, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
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
    state, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
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
    out, changed = apply_rules(model, state, drivers, time=0.0, step=300.0)
    assert out is state and not bool(changed)


# ------------------------------------------------------------------ refusals, by name
def test_the_network_names_what_a_rule_cannot_say():
    yes = [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)]
    cases = [
        (_rule("R", yes, then=[Action("P1", "STATUS", float(ACTIVE))]), "ACTIVE"),
        (_rule("R", [_premise("IF", "SETTING", "=", 1.0, object="LINK", id="P1")],
               then=[CLOSE_P2]), "pipe"),
        (_rule("R", [_premise("IF", "SETTING", "=", 1.0, object="LINK", id="G1")],
               then=[CLOSE_P2]), "GPV"),
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


def test_status_active_on_a_valve_is_accepted_and_does_nothing_as_in_epanet():
    """EPANET's `takeactions` has no branch for IS_ACTIVE: no change, on an OPEN valve too
    (measured on EPANET 2.2)."""
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("J1", 0.0, 0.005)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "A", 100.0, 0.3, 130.0),),
        valves=(Valve("V1", "A", "J1", 0.2, "TCV", 5.0),),
        rules=(_rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)],
                     then=[Action("V1", "STATUS", float(ACTIVE))]),),
        initial_status={"V1": "OPEN"})
    net.validate()
    model, state, drivers = build_model(net)
    out, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
    assert not bool(changed)
    assert out["water.link_status"].tolist() == state["water.link_status"].tolist()


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


def test_a_rule_or_control_on_a_pipe_of_the_potential_form_is_refused_at_build():
    rule = _rule("R", [Premise("IF", "SYSTEM", "", "TIME", ">=", 0.0)], then=[CLOSE_P2])
    with pytest.raises(ValueError, match=r"pipe_form.*potential|potential.*pipe_form"):
        build_model(_two_source([rule], headloss="D-W"), friction="colebrook")
    control = Control("P2", CLOSED, None, "time", value=3600.0)
    with pytest.raises(ValueError, match=r"\['P2'\].*control"):
        build_model(_two_source([], headloss="D-W", controls=(control,)),
                    friction="colebrook")
    build_model(_two_source([], headloss="D-W"), friction="colebrook")


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


# ------------------------------------------------------------------ against EPANET 2.2
def _epanet(tmp_path, text, *, links=(), nodes=()):
    """The last hydraulic step's `{"flow": {link: file unit}, "status": {...}, "head": {...}}`
    of EPANET 2.2 (wntr's toolkit) on the `.inp` `text`."""
    wntr_toolkit = pytest.importorskip("wntr.epanet.toolkit")
    from wntr.epanet.util import EN
    path = tmp_path / "epanet.inp"
    path.write_text(text)
    en = wntr_toolkit.ENepanet()
    en.ENopen(str(path), str(tmp_path / "epanet.rpt"), str(tmp_path / "epanet.bin"))
    en.ENopenH()
    en.ENinitH(0)
    out = {}
    while True:
        en.ENrunH()
        out = {
            "flow": {k: en.ENgetlinkvalue(en.ENgetlinkindex(k), EN.FLOW) for k in links},
            "status": {k: en.ENgetlinkvalue(en.ENgetlinkindex(k), EN.STATUS) for k in links},
            "head": {k: en.ENgetnodevalue(en.ENgetnodeindex(k), EN.HEAD) for k in nodes},
        }
        if en.ENnextH() <= 0:
            break
    en.ENcloseH()
    en.ENclose()
    return out


def _inp(body, rules="", status="", duration="1:00"):
    return (f"[TITLE]\n[JUNCTIONS]\n{body['junctions']}\n[RESERVOIRS]\n{body['reservoirs']}\n"
            f"[PIPES]\n{body.get('pipes', '')}\n[PUMPS]\n{body.get('pumps', '')}\n"
            f"[VALVES]\n{body.get('valves', '')}\n[CURVES]\n{body.get('curves', '')}\n"
            f"[STATUS]\n{status}\n[RULES]\n{rules}\n[TIMES]\nDURATION {duration}\n"
            f"HYDRAULIC TIMESTEP 1:00\nRULE TIMESTEP 0:05\n[OPTIONS]\n"
            f"{body.get('options', 'UNITS LPS')}\n[END]\n")


def _noodl(tmp_path, text):
    path = tmp_path / "noodl.inp"
    path.write_text(text)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    return model, water_steady(model, state, drivers), drivers


def test_a_setting_premise_on_a_valve_with_a_fixed_status_is_never_satisfied(tmp_path):
    """EPANET sets the setting of a valve to MISSING when its status is OPEN or CLOSED
    (`changestatus`, `setlinkstatus`), and `checkvalue` then returns false."""
    body = {"junctions": "A 0 0\nB 0 0\nC 0 1", "reservoirs": "R 50",
            "pipes": "P1 R A 100 300 130 0 Open\nP3 R C 100 300 130 0 Open",
            "valves": "V A B 200 PRV 30 0"}
    rule = "RULE R1\nIF LINK V SETTING = 30\nTHEN LINK P3 STATUS IS CLOSED"
    for status, fires in (("", True), ("V OPEN", False), ("V CLOSED", False)):
        text = _inp(body, rule, status)
        shut = abs(_epanet(tmp_path, text, links=["P3"])["flow"]["P3"]) < 1e-9
        assert shut is fires, status
        model, state, drivers = _noodl(tmp_path, text)
        out, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
        assert bool(changed) is fires, status
        assert (out["water.link_status"][model.link_names.index("P3")].item() == CLOSED) is fires


def test_an_open_action_reopens_a_prv_the_solver_closed(tmp_path):
    """EPANET tests `s <= CLOSED` on the solved `LinkStatus`: a PRV closed by a reversed head
    is fixed open by `STATUS IS OPEN` and the water runs back through it."""
    body = {"junctions": "A 0 0\nB 0 0", "reservoirs": "R1 40\nR2 60",
            "pipes": "P1 R1 A 100 300 130 0 Open\nP2 B R2 100 300 130 0 Open",
            "valves": "V1 A B 200 PRV 30 0"}
    rule = "RULE R1\nIF SYSTEM TIME >= 0\nTHEN LINK V1 STATUS IS OPEN"
    want = _epanet(tmp_path, _inp(body, rule), links=["V1"])["flow"]["V1"] / 1000.0
    none = _epanet(tmp_path, _inp(body), links=["V1"])["flow"]["V1"] / 1000.0
    assert none == pytest.approx(0.0, abs=1e-9) and abs(want) > 1e-3
    model, state, drivers = _noodl(tmp_path, _inp(body, rule))
    j = model.link_names.index("V1")
    assert float(state["water.q"][j]) == pytest.approx(0.0, abs=1e-9)
    state, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
    assert bool(changed) and state["water.link_status"][j].item() == OPEN
    state = water_steady(model, state, drivers)
    # 0.44 m3/s through a 300 mm pipe: EPANET stops at its own accuracy (the two differ by
    # 5e-6 relative, measured), so the bound is 1e-5 relative, not 1e-6 absolute
    assert float(state["water.q"][j]) == pytest.approx(want, rel=1e-5)


PUMP_BODY = {"junctions": "A 0 0", "reservoirs": "R1 0\nR2 45",
             "pipes": "P2 A R2 100 300 130 0 Open", "pumps": "PU R1 A HEAD C1 SPEED 0.8",
             "curves": "C1 0 60\nC1 10 50\nC1 20 30"}


def test_an_open_action_runs_a_pump_the_solver_shut_at_shut_off_at_speed_one(tmp_path):
    """The pump's shut-off head at speed 0.8 is 38.4 m, below the 45 m to lift: EPANET's
    `LinkStatus` is XHEAD (closed); `STATUS IS OPEN` sets its speed to 1 and it runs."""
    rule = "RULE R1\nIF SYSTEM TIME >= 0\nTHEN LINK PU STATUS IS OPEN"
    got = _epanet(tmp_path, _inp(PUMP_BODY, rule), links=["PU"])["flow"]["PU"] / 1000.0
    assert got > 1e-3
    model, state, drivers = _noodl(tmp_path, _inp(PUMP_BODY, rule))
    j = model.link_names.index("PU")
    assert float(state["water.q"][j]) == pytest.approx(0.0, abs=1e-9)
    state, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
    assert bool(changed) and state["water.link_setting"][j].item() == 1.0
    state = water_steady(model, state, drivers)
    assert float(state["water.q"][j]) == pytest.approx(got, abs=1e-6)


def test_a_closed_action_does_nothing_to_a_pump_the_solver_shut(tmp_path):
    """`takeactions` closes a link only if `LinkStatus > CLOSED`: a pump at shut-off is no
    change (EPANET counts no action, and the solver may restart the pump)."""
    rule = "RULE R1\nIF SYSTEM TIME >= 0\nTHEN LINK PU STATUS IS CLOSED"
    model, state, drivers = _noodl(tmp_path, _inp(PUMP_BODY, rule))
    j = model.link_names.index("PU")
    assert float(state["water.q"][j]) == pytest.approx(0.0, abs=1e-9)
    out, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
    assert not bool(changed) and out["water.link_status"][j].item() == OPEN


def test_a_flow_control_valve_the_flow_runs_back_through_reads_open(tmp_path):
    """EPANET's `fcvstatus`: a head that falls along the flow is `XFCV`, which `checkstatus`
    reads as OPEN though the valve's status is ACTIVE otherwise."""
    body = {"junctions": "A 0 0\nB 0 0\nC 0 1", "reservoirs": "R1 30\nR2 50",
            "pipes": ("P1 R1 A 100 300 130 0 Open\nP2 B R2 100 300 130 0 Open\n"
                      "P3 R1 C 100 300 130 0 Open"),
            "valves": "F A B 200 FCV 5 0"}
    for relation, fires in (("IS OPEN", True), ("IS ACTIVE", False)):
        rule = f"RULE R1\nIF LINK F STATUS {relation}\nTHEN LINK P3 STATUS IS CLOSED"
        text = _inp(body, rule)
        shut = abs(_epanet(tmp_path, text, links=["P3"])["flow"]["P3"]) < 1e-9
        assert shut is fires, relation
        model, state, drivers = _noodl(tmp_path, text)
        _, changed = apply_rules(model, state, drivers, time=300.0, step=300.0)
        assert bool(changed) is fires, relation


US_SG = """[JUNCTIONS]
A 0 0
B 0 0
[RESERVOIRS]
R 200
[PIPES]
P1 R A 100 12 100 0 Open
[VALVES]
V A B 8 PRV 30 0
[RULES]
RULE R1
IF NODE B PRESSURE ABOVE 20
THEN LINK P1 STATUS IS CLOSED
[TIMES]
DURATION 0
[OPTIONS]
UNITS GPM
HEADLOSS D-W
SPECIFIC GRAVITY 1.2
[END]
"""


def test_a_us_files_pressures_carry_the_specific_gravity(tmp_path):
    """EPANET's `Ucf[PRESSURE]` is psi per foot times the specific gravity: a PRV set at 30
    psi holds `30 / (0.4333 SG)` ft of head, and a PRESSURE premise or setting is read so."""
    no_rule = US_SG.replace("[RULES]\nRULE R1\nIF NODE B PRESSURE ABOVE 20\n"
                            "THEN LINK P1 STATUS IS CLOSED\n", "")
    want = _epanet(tmp_path, no_rule, nodes=["B"])["head"]["B"] * 0.3048
    assert want == pytest.approx(30.0 / (0.4333 * 1.2) * 0.3048, rel=1e-3)
    path = tmp_path / "sg.inp"
    path.write_text(US_SG)
    net = read_epanet_inp(path)
    assert net.valves[0].setting == pytest.approx(want, rel=1e-3)
    model, state, drivers = build_model(net)
    state = water_steady(model, state, drivers)
    head = float(state["water.phi"][model.node_names.index("B")]) / model.head_scale
    assert head == pytest.approx(want, rel=1e-3)
    assert net.rules[0].premises[0].value == pytest.approx(20.0 * 0.3048 / (0.4333 * 1.2))
