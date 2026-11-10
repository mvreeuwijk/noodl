"""EPANET 2.2's rule-based controls (`rules.c`) as functions of plain numbers.

`evaluate_rules` is `checkrules`: it tests every rule's premises on a snapshot of a solved
hydraulic state and returns the actions that stand, one per link; `take_actions` is
`takeactions`: it applies them to the status and setting of the links. Both are pure and
unbatched (`TankLevels` and `apply_rules` in `noodl.apps.water.network` call them for each
instance, for the state they read); the module imports only the data classes of
`noodl.apps.water.controls`.

EPANET's rules do NOT follow the textbook precedence of AND over OR, whatever its manual says:
`evalpremises` runs left to right and a false result meeting an AND ends the rule as false, so
`IF a AND b OR c` with `a` false is false and `IF a OR b AND c` is `(a OR b) AND c`. It is
transcribed as it is (and checked against EPANET 2.2 in the tests), since EPANET is the
reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from noodl.apps.water.controls import (
    ACTIVE,
    CLOSED,
    OPEN,
    SECONDS_PER_DAY,
    Action,
    Premise,
    Rule,
)

Tensor = torch.Tensor

#: The attributes a premise reads, by the kind of object; the snapshot holds each by id.
NODE_ATTRIBUTES = ("DEMAND", "HEAD", "LEVEL", "PRESSURE")
LINK_ATTRIBUTES = ("FLOW", "STATUS", "SETTING")
SYSTEM_ATTRIBUTES = ("DEMAND", "TIME", "CLOCKTIME")

#: EPANET's tolerance on a setting that a rule sets (`takeactions`: `tol = 1.e-3`, in the
#: internal units: feet, cubic feet per second, a relative speed).
SETTING_TOLERANCE = 1e-3


def evaluate_rules(
    rules: Sequence[Rule], snapshot: Mapping[str, Mapping[str, float]], *, time, step,
    clocktime=0.0,
) -> list[Action]:
    """The actions that stand after every rule is tested at `time`, in the order their
    links first appear: one per link. EPANET 2.2, `rules.c`, `checkrules`, `evalpremises`,
    `updateactionlist` and `onactionlist`.

    A rule whose premises hold contributes its `then_actions`, else its `else_actions`. Among
    actions on one link the rule of the highest PRIORITY wins, a tie goes to the rule listed
    first (a later rule replaces an earlier one only with a strictly higher priority), and a
    rule's own second action on a link it already acts on is dropped.

    `snapshot` maps an attribute to `{id: value}` in SI: node `"DEMAND"` (m3/s, positive out
    of the network), `"HEAD"`, `"LEVEL"` and `"PRESSURE"` (m), link `"FLOW"` (m3/s, a sign is
    ignored), `"STATUS"` (a code of `noodl.apps.water.controls`) and `"SETTING"` (in SI; `nan`
    where the link has none: a valve whose status is fixed, OPEN or CLOSED), and
    `"DEMAND"[""]` the system demand (the sum of the junctions' positive demands).
    Only what the rules read need be present.

    `time` is the simulation time (s) the rules are checked at, `clocktime` the time of day
    (s after midnight) the simulation starts at, `step` the length (s, positive: EPANET never
    checks the rules over an empty interval, `ruletimestep`) of the interval since the
    previous check, which a `SYSTEM TIME` or `CLOCKTIME` premise tests (EPANET's
    `Time1 = Htime - dt + 1`)."""
    if not float(step) > 0.0:
        raise ValueError(
            f"evaluate_rules: step is the length of the rule interval and must be positive, "
            f"got {step!r}")
    time1 = round(float(time)) - round(float(step)) + 1
    now, start = round(float(time)), round(float(clocktime))
    standing: list[tuple[Action, Rule]] = []
    for rule in rules:
        holds = _premises_hold(rule, snapshot, time1, now, start)
        for action in (rule.then_actions if holds else rule.else_actions):
            for k, (other, owner) in enumerate(standing):
                if other.link == action.link:
                    if rule.priority > owner.priority:
                        standing[k] = (action, rule)
                    break
            else:
                standing.append((action, rule))
    return [action for action, _ in standing]


def _premises_hold(rule: Rule, snapshot, time1: int, now: int, start: int) -> bool:
    """`evalpremises`: left to right, an OR tested only while the result is false, an AND (or
    the first IF) ending the rule false at once when the result so far is false."""
    result = True
    for premise in rule.premises:
        if premise.conjunction == "OR":
            if not result:
                result = _premise_holds(premise, snapshot, time1, now, start)
        else:
            if not result:
                return False
            result = _premise_holds(premise, snapshot, time1, now, start)
    return result


def _premise_holds(p: Premise, snapshot, time1: int, now: int, start: int) -> bool:
    """`checkpremise`."""
    if p.object == "SYSTEM" and p.attribute in ("TIME", "CLOCKTIME"):
        return _check_time(p, time1, now, start)
    if p.object == "LINK" and p.attribute == "STATUS":
        return _check_status(p, _read(snapshot, "STATUS", p))
    return _check_value(p, _value(p, snapshot))


def _read(snapshot, attribute: str, p: Premise) -> float:
    try:
        return float(snapshot[attribute][p.id])
    except KeyError:
        raise ValueError(
            f"evaluate_rules: the snapshot has no {attribute} of {p.object.lower()} "
            f"{p.id!r}, which a rule premise reads") from None


def _value(p: Premise, snapshot) -> float:
    """The quantity a numerical premise tests, as `checkvalue` finds it (in SI)."""
    if p.object == "SYSTEM":
        if p.attribute != "DEMAND":
            raise ValueError(f"evaluate_rules: SYSTEM {p.attribute} is not a rule premise "
                             f"(TIME, CLOCKTIME and DEMAND are)")
        return _read(snapshot, "DEMAND", p)
    if p.object == "NODE" and p.attribute in NODE_ATTRIBUTES:
        return _read(snapshot, p.attribute, p)
    if p.object == "LINK" and p.attribute == "FLOW":
        return abs(_read(snapshot, "FLOW", p))
    if p.object == "LINK" and p.attribute == "SETTING":
        return _read(snapshot, "SETTING", p)
    raise ValueError(f"evaluate_rules: {p.object} {p.attribute} is not a rule premise")


def _check_value(p: Premise, x: float) -> bool:
    """`checkvalue`'s comparison: EPANET's tolerance (`p.tolerance`) on the lenient side of
    every relation. A missing value (`nan`: a valve whose status was set carries no setting)
    holds for none."""
    if math.isnan(x):
        return False
    tol, v, relation = p.tolerance, p.value, p.relation
    if relation == "=":
        return not abs(x - v) > tol
    if relation == "<>":
        return not abs(x - v) < tol
    if relation == "<":
        return not x > v + tol
    if relation == "<=":
        return not x > v - tol
    if relation == ">":
        return not x < v - tol
    if relation == ">=":
        return not x < v + tol
    raise ValueError(f"evaluate_rules: unknown relation {relation!r}")


def _check_status(p: Premise, code: float) -> bool:
    """`checkstatus`: a link is CLOSED at or below `CLOSED`, ACTIVE at `ACTIVE`, OPEN else;
    only `=` (IS) and `<>` (NOT) are tested, any other relation never holds."""
    code = int(code)
    state = CLOSED if code <= CLOSED else ACTIVE if code == ACTIVE else OPEN
    if p.relation == "=":
        return state == int(p.value)
    if p.relation == "<>":
        return state != int(p.value)
    return False


def _check_time(p: Premise, time1: int, now: int, start: int) -> bool:
    """`checktime`: the interval `(time1, now)` of the rule step, in simulation time or, for
    CLOCKTIME, in the time of day; an inequality tests the interval's end, an equality
    whether the value lies in it (across midnight, on either side of it)."""
    if p.attribute == "TIME":
        t1, t2 = time1, now
    else:
        t1 = (time1 + start) % round(SECONDS_PER_DAY)
        t2 = (now + start) % round(SECONDS_PER_DAY)
    x = int(p.value)
    relation = p.relation
    if relation == "<":
        return not t2 >= x
    if relation == "<=":
        return not t2 > x
    if relation == ">":
        return not t2 <= x
    if relation == ">=":
        return not t2 < x
    inside = (x >= t1 or x <= t2) if t2 < t1 else (t1 <= x <= t2)
    return inside if relation == "=" else not inside


def take_actions(
    actions: Sequence[Action], *, status: Tensor, setting: Tensor, link_names: Sequence[str],
    link_kinds: Sequence[str], setting_scale: Sequence[float] | None = None,
    setting_tolerance: Sequence[float] | None = None, solved: Tensor | None = None,
) -> tuple[Tensor, Tensor, bool]:
    """`(status, setting, changed)`: the links' status and setting (`(n_links,)`, in
    `link_names` order, of `link_kinds` `"PIPE"`, `"PUMP"` or `"VALVE"`) after `actions`,
    and whether any link changed. EPANET 2.2, `rules.c`, `takeactions`, with `hydraul.c`,
    `setlinkstatus` and `setlinksetting`.

    STATUS OPEN acts on a closed link only (a pump then runs at relative speed 1), STATUS
    CLOSED on a link that is not closed (a pump's speed is then 0); an action that finds its
    link as it says is no change. "Closed" is the link's state as the last solve found it
    (`solved`, as `link_states` gives it: EPANET's `LinkStatus`; the carried `status` when not
    given), so a PRV the solver closed or a pump it shut at shut-off is closed to an OPEN, and
    a pump shut at shut-off is not closed again by a CLOSED. The action itself is written to
    the carried `status`. STATUS ACTIVE is no action at all in EPANET, and none here. SETTING
    on a pipe is its status (0 closes it); on a
    pump its relative speed (positive opens a closed pump, 0 closes it); on a valve it makes
    it ACTIVE at the setting. A setting is a change only beyond `setting_tolerance` (per
    link, default 1e-3 as EPANET's `tol`, in the carried unit) of the present one, a closed
    pump counting as set to 0. `setting_scale` (per link, default 1) puts an action's SI
    setting in the carried unit (`head_scale` for a PRV, PSV or PBV on a pressure layer).

    A pump CLOSED keeps its carried speed here, where EPANET sets it to 0 (the speed changes
    no flow of a shut pump, and `TankLevels` carries the build's speed on)."""
    status, setting = status.clone(), setting.clone()
    found = status if solved is None else solved
    index = {name: j for j, name in enumerate(link_names)}
    changed = False
    for action in actions:
        j = index[action.link]
        kind = link_kinds[j]
        s = int(found[j])
        wanted: int | None = None
        if action.attribute == "STATUS":
            wanted = int(action.value)
        elif kind == "PIPE":
            wanted = CLOSED if action.value == 0.0 else OPEN
        if wanted is not None:
            if wanted == OPEN and s <= CLOSED:
                status[j] = OPEN
                if kind == "PUMP":
                    setting[j] = 1.0
                changed = True
            elif wanted == CLOSED and s > CLOSED:
                status[j] = CLOSED
                changed = True
            continue
        x = action.value * (1.0 if setting_scale is None else setting_scale[j])
        tol = SETTING_TOLERANCE if setting_tolerance is None else setting_tolerance[j]
        if kind == "PUMP":
            present = 0.0 if int(status[j]) <= CLOSED else float(setting[j])
            if abs(x - present) > tol:
                setting[j] = x
                if x > 0.0 and s <= CLOSED:
                    status[j] = OPEN
                if x == 0.0:
                    status[j] = CLOSED   # a speed of 0 is a shut pump (never a 0-speed curve)
                changed = True
        elif int(status[j]) != ACTIVE or abs(x - float(setting[j])) > tol:
            # a valve set after its status was fixed (EPANET's MISSING setting) is ACTIVE
            status[j], setting[j] = ACTIVE, x
            changed = True
    return status, setting, changed
