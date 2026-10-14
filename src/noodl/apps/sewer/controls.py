"""SWMM 5.2.4's rule-based controls (``[CONTROLS]``, `controls.c`) for the dynamic wave.

A rule is ``RULE name / IF premise / AND|OR premise ... / THEN action / AND action ... /
ELSE action ... / PRIORITY p``. A premise compares a variable (``NODE n DEPTH|HEAD|VOLUME|INFLOW``,
``LINK|CONDUIT|PUMP|ORIFICE|WEIR|OUTLET l FLOW|STATUS|SETTING|TIMEOPEN|TIMECLOSED``, or
``SIMULATION TIME|DATE|CLOCKTIME|DAYOFYEAR|DAY|MONTH``) with a number or another variable; an
action sets a pump's STATUS (ON, OFF) or SETTING (the speed), an orifice's, weir's or outlet's
SETTING (the fraction open) to a number, to a ``CURVE`` of the controlling premise's value, to
a ``TIMESERIES`` of the clock, or to a ``PID`` controller's output.

`ControlSet` is the immutable rule set (what the reader makes of ``[CONTROLS]``, in the
file's own units, which is how SWMM compares: a premise is judged on the value in the units of
the file, `ControlSet.length_factor` and `flow_factor` convert noodl's SI state to them).
`ControlEngine` evaluates it the way `controls_evaluate` does (controls.c:458-512):

- the premises of a rule are evaluated in file order, strictly left to right: an AND premise is
  skipped (and stops the rest) once the result is false, an OR premise is evaluated only while
  it is false -- there is no operator precedence (controls.c:469-483);
- the actions of the branch taken (THEN if true, else ELSE) are visited in REVERSE of the order
  written (SWMM links them at the head, controls.c:944-956) and each updates its value (a curve
  or time series lookup, a PID step) before it competes: per controlled link, the first action
  visited stays unless a later one comes from a rule of STRICTLY higher priority
  (`updateActionList`, controls.c:1188-1211), so of equal priority the earlier rule wins, and
  within a rule the action written last;
- a CURVE setting is the curve at the value of the LAST premise comparison made (SWMM's global
  ``ControlValue``; for TIMEOPEN and TIMECLOSED in hours), a PID's set point and process value
  are that comparison's right and left sides (controls.c:1099-1113, 1295-1311).

Not modelled, refused BY NAME when the file is read: ``VARIABLE`` and ``EXPRESSION`` lines,
rain-gauge premises, the premise attributes MAXDEPTH, FULLFLOW, FULLDEPTH, LENGTH, SLOPE and
VELOCITY and the link DEPTH, and the conduit actions (a closed conduit). Rule evaluation is
discrete state, not a differentiable function of the model.

Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

import bisect
import datetime
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from noodl.apps.sewer.inflows import TimeSeries, parse_clock, parse_date

#: SWMM's relational operators (controls.c:73), in its order.
RELATIONS = ("=", "<>", "<", "<=", ">", ">=")
#: SWMM's ``MISSING`` (consts.h): a variable that does not apply to its object.
MISSING = -1.0e10
#: SWMM's ``TINY`` (consts.h:33), the PID's error floor.
TINY = 1.0e-6
#: seconds per day
SECONDS_PER_DAY = 86400.0
#: SWMM's ``DateDelta`` (datetime.c:40): days from 01/01/0000 to its DateTime zero, 12/30/1899.
DATE_DELTA = 693594
_OBJECTS = ("GAGE", "NODE", "LINK", "CONDUIT", "PUMP", "ORIFICE", "WEIR", "OUTLET",
            "SIMULATION")
_ATTRIBUTES = ("DEPTH", "MAXDEPTH", "HEAD", "VOLUME", "INFLOW", "FLOW", "FULLFLOW",
               "FULLDEPTH", "STATUS", "SETTING", "LENGTH", "SLOPE", "VELOCITY", "TIMEOPEN",
               "TIMECLOSED", "TIME", "DATE", "CLOCKTIME", "DAYOFYEAR", "DAY", "MONTH")
_STATUS_WORDS = ("OFF", "ON")
_CONDUIT_WORDS = ("CLOSED", "OPEN")
_SETTING_WORDS = ("CURVE", "TIMESERIES", "PID")
#: the premise attributes this implementation evaluates
SUPPORTED_ATTRIBUTES = {
    "node": ("depth", "head", "volume", "inflow"),
    "link": ("flow", "status", "setting", "timeopen", "timeclosed"),
    "simulation": ("time", "date", "clocktime", "dayofyear", "day", "month"),
}
#: the link types of `ControlSet.validate` / the reader: SWMM's link kinds
LINK_TYPES = ("conduit", "pump", "orifice", "weir", "outlet")


def _match(token: str, words: Sequence[str]) -> int:
    """SWMM's ``findmatch``: the index of the first keyword that is a (case-insensitive)
    prefix of `token`, -1 if none."""
    upper = token.upper()
    for i, word in enumerate(words):
        if upper.startswith(word):
            return i
    return -1


def _exact(token: str, words: Sequence[str]) -> int:
    """SWMM's ``findExactMatch`` (controls.c:1536): the index of the keyword equal to
    `token` (case-insensitive), -1 if none."""
    upper = token.upper()
    return words.index(upper) if upper in words else -1


@dataclass(frozen=True)
class Variable:
    """A premise variable: `object` ``"node"``, ``"link"`` or ``"simulation"``, the object's
    `name` (``None`` for ``"simulation"``) and the lower-case `attribute`."""

    object: str
    name: str | None
    attribute: str


@dataclass(frozen=True)
class Premise:
    """One premise clause: `join` ``"and"`` (also the IF) or ``"or"`` joins it to the result
    so far; it compares `lhs` with the number `value` or, when `value` is None, with the
    variable `rhs`, by one of the `RELATIONS` (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``).
    `value` is in the file's units (clock quantities in days)."""

    join: str
    lhs: Variable
    relation: str
    value: float | None = None
    rhs: Variable | None = None


@dataclass(frozen=True)
class Action:
    """One action clause on link `link` (of kind `link_type`, one of `LINK_TYPES`):
    `attribute` ``"status"`` (ON/OFF = 1/0 for a pump) or ``"setting"``; `mode` ``"value"``
    (the number `value`), ``"curve"`` (the CONTROL curve `source` at the controlling premise's
    value), ``"timeseries"`` (the time series `source` at the clock) or ``"pid"`` (gains
    `gains` = Kp, Ki (minutes), Kd (minutes))."""

    link: str
    link_type: str
    attribute: str
    mode: str = "value"
    value: float = 0.0
    source: str | None = None
    gains: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Rule:
    """A control rule: `name`, its `premises` in order, the actions `then_actions` and
    `else_actions` in the order written and its `priority` (SWMM's default 0; the higher
    wins)."""

    name: str
    premises: tuple[Premise, ...]
    then_actions: tuple[Action, ...]
    else_actions: tuple[Action, ...] = ()
    priority: float = 0.0


@dataclass(frozen=True)
class ControlSet:
    """The ``[CONTROLS]`` of a SWMM file.

    `rules` in file order; `curves` the CONTROL curves ``{name: (x values, y values)}`` and
    `series` the time series ``{name: TimeSeries}`` that actions name (both as in the file);
    `length_factor` and `flow_factor` the metres per length unit and m3/s per flow unit of the
    file (`inp.unit_factors`), by which noodl's SI values are brought back to the file's
    units for the comparisons; `start` the run's start date and time (SWMM's
    START_DATE + START_TIME, which the clock premises and the open and closed times count
    from) and `rule_step` the RULE_STEP option in seconds (0: the rules are evaluated at every
    routing step, otherwise every `rule_step` s, which also shortens the routing steps so that
    one ends at each, routing.c:189-195)."""

    rules: tuple[Rule, ...]
    curves: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        default_factory=dict)
    series: Mapping[str, TimeSeries] = field(default_factory=dict)
    length_factor: float = 1.0
    flow_factor: float = 1.0
    start: datetime.datetime = datetime.datetime(2004, 1, 1)
    rule_step: float = 0.0

    def validate(self, net) -> None:
        """Refuse, BY NAME, a rule that names an unknown node or link, an action on a link
        of another kind, an unsupported variable or an unknown curve or series."""
        nodes = {m.name for m in net.manholes} | {o.name for o in net.outfalls}
        links = {p.name: "conduit" for p in net.pipes}
        for kind, devices in net.structures.items():
            links.update({d.name: kind for d in devices})
        if not self.rule_step >= 0.0:
            raise ValueError(f"ControlSet: rule_step must be >= 0 s, got {self.rule_step!r}")
        for rule in self.rules:
            where = f"ControlSet: rule {rule.name!r}"
            for premise in rule.premises:
                for var in (premise.lhs, premise.rhs):
                    if var is None:
                        continue
                    if var.attribute not in SUPPORTED_ATTRIBUTES.get(var.object, ()):
                        raise ValueError(f"{where}: premise variable {var.object} "
                                         f"{var.attribute} is not modelled")
                    known = nodes if var.object == "node" else links
                    if var.object != "simulation" and var.name not in known:
                        raise ValueError(f"{where}: premise names {var.object} {var.name!r}, "
                                         f"which does not exist")
                if premise.relation not in RELATIONS:
                    raise ValueError(f"{where}: relation must be one of {RELATIONS}, got "
                                     f"{premise.relation!r}")
                if (premise.value is None) == (premise.rhs is None):
                    raise ValueError(f"{where}: a premise needs exactly one of a value and a "
                                     f"right-hand variable")
            for action in (*rule.then_actions, *rule.else_actions):
                if links.get(action.link) != action.link_type:
                    raise ValueError(f"{where}: action names {action.link_type} "
                                     f"{action.link!r}, which is not one")
                if action.link_type == "conduit":
                    raise ValueError(f"{where}: an action on conduit {action.link!r} (a "
                                     f"closed conduit) is not modelled")
                if action.attribute not in ("status", "setting") or (
                        action.attribute == "status" and action.link_type != "pump"):
                    raise ValueError(f"{where}: {action.link_type} {action.link!r} has no "
                                     f"{action.attribute} to set")
                if action.mode == "curve" and action.source not in self.curves:
                    raise ValueError(f"{where}: names CONTROL curve {action.source!r}, which "
                                     f"does not exist")
                if action.mode == "timeseries" and action.source not in self.series:
                    raise ValueError(f"{where}: names time series {action.source!r}, which "
                                     f"does not exist")
                if action.mode not in ("value", "curve", "timeseries", "pid"):
                    raise ValueError(f"{where}: unknown action mode {action.mode!r}")


# --------------------------------------------------------------------------------- reading
def parse_rules(lines, *, links: Mapping[str, str], nodes: set[str], path) -> list[Rule]:
    """The `Rule` list of the ``[CONTROLS]`` `lines` (`inpfile.InpLine`), following
    ``controls_addRuleClause`` / ``addPremise`` / ``addAction`` (controls.c:432-880): `links`
    maps every link name to its kind (one of `LINK_TYPES`) and `nodes` holds the node names.
    A malformed rule, an unknown object or a clause SWMM rejects is refused with its line; so
    is everything this implementation does not model (module docstring)."""
    rules: list[Rule] = []
    current: dict | None = None
    state = "priority"

    def refuse(line, why):
        raise ValueError(f"{path}: line {line.number} of [CONTROLS]: {why}")

    def close():
        if current is not None:
            rules.append(Rule(current["name"], tuple(current["premises"]),
                              tuple(current["then"]),
                              tuple(current["else"]), current["priority"]))

    for line in lines:
        tok = list(line.fields)
        word = tok[0].upper()
        if word in ("VARIABLE", "EXPRESSION"):
            refuse(line, f"{word} (a named variable or math expression in a rule) is not "
                         f"modelled")
        if word == "RULE":
            if len(tok) != 2:
                refuse(line, "RULE takes exactly one name")
            close()
            current = {"name": tok[1], "premises": [], "then": [], "else": [], "priority": 0.0}
            state = "rule"
        elif word == "IF":
            if state != "rule":
                refuse(line, "IF must follow RULE")
            state = "if"
            current["premises"].append(_premise(line, "and", tok, links, nodes, path))
        elif word == "AND":
            if state == "if":
                current["premises"].append(_premise(line, "and", tok, links, nodes, path))
            elif state in ("then", "else"):
                current[state].append(_action(line, tok, links, path))
            else:
                refuse(line, "AND is out of place")
        elif word == "OR":
            if state != "if":
                refuse(line, "OR must continue the IF premises")
            current["premises"].append(_premise(line, "or", tok, links, nodes, path))
        elif word == "THEN":
            if state != "if":
                refuse(line, "THEN must follow the IF premises")
            state = "then"
            current["then"].append(_action(line, tok, links, path))
        elif word == "ELSE":
            if state != "then":
                refuse(line, "ELSE must follow the THEN actions")
            state = "else"
            current["else"].append(_action(line, tok, links, path))
        elif word == "PRIORITY":
            if state not in ("then", "else") or len(tok) != 2:
                refuse(line, "PRIORITY must follow the actions and take one number")
            try:
                current["priority"] = float(tok[1])
            except ValueError:
                refuse(line, f"PRIORITY {tok[1]!r} is not a number")
            state = "priority"
        else:
            refuse(line, f"unknown rule keyword {tok[0]!r}")
    close()
    return rules


def _variable(line, tok, k, links, nodes, path) -> tuple[Variable, int]:
    """``getPremiseVariable`` (controls.c:632-770): the variable starting at token `k` and the
    index of its last token."""
    def refuse(why):
        raise ValueError(f"{path}: line {line.number} of [CONTROLS]: {why}")

    obj = _match(tok[k], _OBJECTS)
    if obj < 0:
        refuse(f"unknown object type {tok[k]!r}")
    name = _OBJECTS[obj]
    if name == "GAGE":
        refuse("a rain gage in a premise is not modelled")
    n = k
    if name != "SIMULATION":
        n += 1
        if n >= len(tok):
            refuse("the premise names no object")
        ident = tok[n]
        if name == "NODE":
            if ident not in nodes:
                refuse(f"node {ident!r} is unknown")
        elif ident not in links:
            refuse(f"link {ident!r} is unknown")
    else:
        ident = None
    n += 1
    if n >= len(tok):
        refuse("the premise names no attribute")
    attr = _match(tok[n], _ATTRIBUTES)
    if attr < 0:
        refuse(f"unknown attribute {tok[n]!r}")
    attribute = _ATTRIBUTES[attr]
    valid = {
        "NODE": ("DEPTH", "MAXDEPTH", "HEAD", "VOLUME", "INFLOW"),
        "LINK": ("STATUS", "DEPTH", "FULLFLOW", "FULLDEPTH", "FLOW", "LENGTH", "SLOPE",
                 "VELOCITY"),
        "PUMP": ("FLOW", "SETTING", "STATUS"),
        "ORIFICE": ("FLOW", "SETTING"),
        "SIMULATION": ("TIME", "DATE", "CLOCKTIME", "DAY", "MONTH", "DAYOFYEAR"),
    }
    valid["CONDUIT"] = valid["LINK"]
    valid["WEIR"] = valid["OUTLET"] = valid["ORIFICE"]
    if not (name not in ("NODE", "SIMULATION") and attribute in ("TIMEOPEN", "TIMECLOSED")
            or attribute in valid[name]):
        refuse(f"{name} {attribute} is not a valid variable")
    kind = {"NODE": "node", "SIMULATION": "simulation"}.get(name, "link")
    if attribute.lower() not in SUPPORTED_ATTRIBUTES[kind]:
        refuse(f"the premise variable {name} {attribute} is not modelled")
    return Variable(kind, ident, attribute.lower()), n


def _premise_value(line, token: str, attribute: str, path) -> float:
    """``getPremiseValue`` (controls.c:775-848): the number in a premise, in days for a clock
    quantity."""
    def bad(what):
        raise ValueError(f"{path}: line {line.number} of [CONTROLS]: {token!r} is not {what}")

    if attribute == "status":
        i = _match(token, _STATUS_WORDS)
        if i < 0:
            i = _match(token, _CONDUIT_WORDS)
        if i < 0:
            bad("ON, OFF, OPEN or CLOSED")
        return float(i)
    if attribute in ("time", "clocktime", "timeopen", "timeclosed"):
        seconds = parse_clock(token)
        if seconds is None:
            bad("a time")
        return seconds / 24.0 / 3600.0
    if attribute == "date":
        try:
            day = parse_date(token)
        except ValueError:
            day = None
        if day is None:
            bad("a date")
        return float(date_serial(day))
    if attribute == "dayofyear":                       # MM/DD, or a day number (controls.c:830)
        try:
            day = parse_date(token + "/1947")
        except ValueError:
            day = None
        if day is not None:
            return float(day.timetuple().tm_yday)
    try:
        value = float(token)
    except ValueError:
        bad("a number")
    if attribute == "day" and not 1.0 <= value <= 7.0:
        bad("a day of the week (1-7)")
    if attribute == "month" and not 1.0 <= value <= 12.0:
        bad("a month (1-12)")
    if attribute == "dayofyear" and not 1.0 <= value <= 365.0:
        bad("a day of the year (1-365)")
    return value


def _premise(line, join, tok, links, nodes, path) -> Premise:
    """``addPremise`` (controls.c:550-630)."""
    def refuse(why):
        raise ValueError(f"{path}: line {line.number} of [CONTROLS]: {why}")

    if len(tok) < 4:
        refuse("a premise needs a variable, an operator and a value")
    lhs, n = _variable(line, tok, 1, links, nodes, path)
    n += 1
    if n >= len(tok):
        refuse("the premise has no operator")
    rel = _exact(tok[n], RELATIONS)
    if rel < 0:
        refuse(f"unknown relational operator {tok[n]!r}")
    n += 1
    if n >= len(tok):
        refuse("the premise has no right-hand side")
    value, rhs = None, None
    if _match(tok[n], _OBJECTS) >= 0:
        rhs, n = _variable(line, tok, n, links, nodes, path)
    else:
        value = _premise_value(line, tok[n], lhs.attribute, path)
    if n + 1 < len(tok):
        refuse("more than one clause on a line")
    return Premise(join, lhs, RELATIONS[rel], value, rhs)


def _action(line, tok, links, path) -> Action:
    """``addAction`` (controls.c:890-1005) and ``setActionSetting``."""
    def refuse(why):
        raise ValueError(f"{path}: line {line.number} of [CONTROLS]: {why}")

    if len(tok) < 6:
        refuse("an action needs an object, a name, an attribute, = and a setting")
    obj = _match(tok[1], _OBJECTS)
    kind = _OBJECTS[obj].lower() if obj >= 0 else ""
    if kind not in ("link", "conduit", "pump", "orifice", "weir", "outlet"):
        refuse(f"unknown action object {tok[1]!r}")
    name = tok[2]
    if name not in links:
        refuse(f"link {name!r} is unknown")
    if kind != "link" and links[name] != kind:
        refuse(f"{name!r} is not a {kind}")
    if kind == "link":
        refuse("an action on a LINK object is not valid in SWMM; name the pump, orifice, weir "
               "or outlet")
    attrib = _match(tok[3], _ATTRIBUTES)
    if attrib < 0:
        refuse(f"unknown attribute {tok[3]!r}")
    attribute = _ATTRIBUTES[attrib].lower()
    if kind == "conduit":
        refuse(f"the conduit action {attribute.upper()} (a closed conduit) is not modelled"
               if attribute == "status" else
               f"conduit {name!r}: only STATUS can be set (SWMM's ERR_KEYWORD)")
    if kind == "pump" and attribute == "status":
        i = _match(tok[5], _STATUS_WORDS)
        if i < 0:
            refuse(f"pump STATUS must be ON or OFF, got {tok[5]!r}")
        return Action(name, kind, "status", "value", float(i))
    if attribute != "setting":
        refuse(f"{kind} {name!r}: only SETTING can be set (SWMM's ERR_KEYWORD)"
               if kind != "pump" else f"pump {name!r}: only STATUS and SETTING can be set")
    mode = _match(tok[5], _SETTING_WORDS)
    if mode >= 0 and len(tok) < 7:
        refuse("CURVE, TIMESERIES and PID settings need a name or gains")
    word = _SETTING_WORDS[mode] if mode >= 0 else ""
    if word == "CURVE":
        return Action(name, kind, "setting", "curve", source=tok[6])
    if word == "TIMESERIES":
        return Action(name, kind, "setting", "timeseries", source=tok[6])
    if word == "PID":
        if len(tok) < 9:
            refuse("a PID setting needs Kp, Ki and Kd")
        try:
            gains = (float(tok[6]), float(tok[7]), float(tok[8]))
        except ValueError:
            refuse("PID gains must be numbers")
        return Action(name, kind, "setting", "pid", gains=gains)
    try:
        value = float(tok[5])
    except ValueError:
        refuse(f"setting {tok[5]!r} is not a number")
    if kind != "pump" and not 0.0 <= value <= 1.0:
        refuse(f"{kind} setting {tok[5]} must be between 0 and 1")
    return Action(name, kind, "setting", "value", value)


# -------------------------------------------------------------------------------- clock
def date_serial(day: datetime.date) -> int:
    """SWMM's DateTime of a calendar date (days since 12/30/1899, `datetime_encodeDate`)."""
    return day.toordinal() - DATE_DELTA


def swmm_datetime(start: datetime.datetime, elapsed_ms: float) -> float:
    """`swmm5.c::getDateTime` (swmm5.c:1551): the DateTime (days) `elapsed_ms` ms into a run
    that starts at `start`, read 1 ms late as SWMM's clock is, with SWMM's own arithmetic
    (`datetime_addSeconds`, datetime.c:381-393: the start's time of day rounded to whole
    seconds, then the day fraction summed)."""
    day = float(date_serial(start.date()))
    seconds = start.hour * 3600 + start.minute * 60 + start.second
    return day + (float(seconds) + (elapsed_ms + 1.0) / 1000.0) / SECONDS_PER_DAY


def start_serial(start: datetime.datetime) -> float:
    """SWMM's ``StartDateTime``: the start's DateTime (days)."""
    seconds = start.hour * 3600 + start.minute * 60 + start.second
    return float(date_serial(start.date())) + seconds / SECONDS_PER_DAY


# ------------------------------------------------------------------------------ evaluation
class RuleContext(Protocol):
    """What `ControlEngine.evaluate` reads of the simulation, in the file's units."""

    def node_value(self, name: str, attribute: str) -> float:
        """A node's ``depth``, ``head``, ``volume`` or ``inflow``."""

    def link_flow(self, name: str) -> float:
        """A link's flow, positive in the file's orientation."""

    def link_setting(self, name: str) -> tuple[str, float, float, float]:
        """``(kind, setting, target setting, time last set)`` of a link: its kind, current and
        target settings and the DateTime (days) it was last opened or closed."""


class ControlEngine:
    """The stateful evaluation of a `ControlSet` over a run: SWMM's ``controls_evaluate``.

    Holds what SWMM keeps in globals between and within evaluations: the last comparison's two
    sides (``ControlValue``, ``SetPoint``) and each PID action's two previous errors. Make a
    new engine for a new run."""

    def __init__(self, controls: ControlSet) -> None:
        self.controls = controls
        self.control_value = 0.0
        self.set_point = 0.0
        self._pid: dict[tuple[int, int, bool], list[float]] = {}
        self._tables = {name: (list(xs), list(ys)) for name, (xs, ys) in controls.curves.items()}

    def evaluate(self, ctx: RuleContext, current: float, elapsed: float, step: float,
                 ) -> dict[str, float]:
        """The new target setting of each link the fired actions change (a link whose action
        value equals its current target is left out, as in ``executeActionList``), at the
        DateTime `current` (days), `elapsed` days into the run, with the routing step `step`
        (days)."""
        self._current, self._elapsed, self._ctx = current, elapsed, ctx
        chosen: dict[str, tuple[Action, float, float]] = {}
        for r, rule in enumerate(self.controls.rules):
            result = True
            for premise in rule.premises:
                if premise.join == "or":
                    if not result:
                        result = self._premise(premise, step)
                else:
                    if not result:
                        break
                    result = self._premise(premise, step)
            actions = rule.then_actions if result else rule.else_actions
            for i in range(len(actions) - 1, -1, -1):                      # controls.c:944-956
                action = actions[i]
                value = self._action_value(action, (r, i, result), step)
                held = chosen.get(action.link)
                if held is None or rule.priority > held[2]:
                    chosen[action.link] = (action, value, rule.priority)
        changed = {}
        for link, (_, value, _) in chosen.items():
            if ctx.link_setting(link)[2] != value:
                changed[link] = value
        return changed

    # -- premises ------------------------------------------------------------------------
    def _premise(self, p: Premise, step: float) -> bool:
        lhs = self._value(p.lhs)
        rhs = p.value if p.value is not None else self._value(p.rhs)
        if lhs == MISSING or rhs == MISSING:
            return False
        attribute = p.lhs.attribute
        if attribute in ("time", "clocktime"):
            return self._compare_times(lhs, p.relation, rhs, step / 2.0)
        if attribute in ("timeopen", "timeclosed"):
            result = self._compare_times(lhs, p.relation, rhs, step / 2.0)
            self.control_value = lhs * 24.0                               # controls.c:1109
            return result
        return self._compare(lhs, p.relation, rhs)

    def _compare_times(self, lhs: float, relation: str, rhs: float, half: float) -> bool:
        if relation == "=":
            return rhs - half <= lhs < rhs + half
        if relation == "<>":
            return lhs < rhs - half or lhs >= rhs + half
        return self._compare(lhs, relation, rhs)

    def _compare(self, lhs: float, relation: str, rhs: float) -> bool:
        self.set_point = rhs                                              # controls.c:1295
        self.control_value = lhs
        return {"=": lhs == rhs, "<>": lhs != rhs, "<": lhs < rhs, "<=": lhs <= rhs,
                ">": lhs > rhs, ">=": lhs >= rhs}[relation]

    def _value(self, v: Variable) -> float:
        """``getVariableValue`` (controls.c:1135-1279); MISSING where it does not apply."""
        a = v.attribute
        if v.object == "simulation":
            day = math.floor(self._current)
            if a == "time":
                return self._elapsed
            if a == "date":
                return float(day)
            if a == "clocktime":
                return self._current - day
            when = datetime.date.fromordinal(int(day) + DATE_DELTA)
            if a == "day":
                return float((int(day) + DATE_DELTA) % 7 + 1)             # datetime.c:476
            if a == "month":
                return float(when.month)
            return float(when.timetuple().tm_yday)                        # dayofyear
        if v.object == "node":
            return self._ctx.node_value(v.name, a)
        if a == "flow":
            return self._ctx.link_flow(v.name)
        kind, setting, _, last = self._ctx.link_setting(v.name)
        if a == "status":
            return setting if kind in ("conduit", "pump") else MISSING
        if a == "setting":
            return setting if kind in ("pump", "orifice", "weir") else MISSING
        if a == "timeopen":
            return self._current - last if setting > 0.0 else MISSING
        return self._current - last if setting <= 0.0 else MISSING       # timeclosed

    # -- action values ------------------------------------------------------------------
    def _action_value(self, a: Action, key: tuple[int, int, bool], step: float) -> float:
        """``updateActionValue`` (controls.c:1058-1075)."""
        if a.mode == "curve":
            xs, ys = self._tables[a.source]
            return _table_lookup(xs, ys, self.control_value)
        if a.mode == "timeseries":
            series = self.controls.series[a.source]
            t = (self._current - start_serial(self.controls.start)) * SECONDS_PER_DAY
            return _series_lookup(series, t)
        if a.mode == "pid":
            return self._pid_setting(a, key, step)
        return a.value

    def _pid_setting(self, a: Action, key: tuple[int, int, bool], step: float) -> float:
        """``getPIDSetting`` (controls.c:1077-1131)."""
        kp, ki, kd = a.gains
        dt = step * 1440.0
        e = self._pid.setdefault(key, [0.0, 0.0])                         # e1, e2
        e0 = self.set_point - self.control_value
        if abs(e0) > TINY:
            e0 = e0 / (self.set_point if self.set_point != 0.0 else self.control_value)
        if abs(e0 - e[0]) < 0.0001:
            e[1] = e[0] = 0.0
        p = e0 - e[0]
        i = 0.0 if ki == 0.0 else e0 * dt / ki
        d = kd * (e0 - 2.0 * e[0] + e[1]) / dt
        update = kp * (p + i + d)
        if abs(update) < 0.0001:
            update = 0.0
        setting = self._ctx.link_setting(a.link)[2] + update
        e[1], e[0] = e[0], e0
        setting = max(setting, 0.0)
        if a.link_type != "pump":
            setting = min(setting, 1.0)
        return setting


def _table_lookup(xs: list[float], ys: list[float], x: float) -> float:
    """`table.c::table_lookup` (table.c:395-426): linear, the first y at or below the first
    x and the last y beyond the last x."""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    k = bisect.bisect_left(xs, x)
    if k >= len(xs):
        return ys[-1]
    x1, x2, y1, y2 = xs[k - 1], xs[k], ys[k - 1], ys[k]
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def _series_lookup(series: TimeSeries, t: float) -> float:
    """`table.c::table_tseriesLookup` with ``extend`` (table.c:745-804): linear in time, the
    first value before the series and the last after it."""
    times, values = series.times, series.values
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    return _table_lookup(list(times), list(values), t)
