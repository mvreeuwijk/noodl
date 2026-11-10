"""Link status codes and the data classes of EPANET's `[CONTROLS]` and `[RULES]`.

The data classes hold what the reader produces and what the builder acts on. `apply_controls`
and `next_control_time` are EPANET's simple controls as functions of plain tensors (the
tank closure `TankLevels` calls them); this module imports nothing from the rest of the
application.

`CLOSED`, `OPEN` and `ACTIVE` are the codes of a link's status: `CLOSED = 0`, `OPEN = 1`,
`ACTIVE = 2` (a valve acting on its setting).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

CLOSED, OPEN, ACTIVE = 0, 1, 2

#: EPANET 2.2's head tolerance `Htol` (0.0005 ft, `hydraul.c`, `inithyd`) in metres.
HTOL = 0.0005 * 0.3048

SECONDS_PER_DAY = 86400.0

Tensor = torch.Tensor


@dataclass(frozen=True)
class Control:
    """One line of `[CONTROLS]`: set `link` to `status` (`CLOSED`, `OPEN`, `ACTIVE`) or to a
    numeric `setting` (exactly one of the two is not `None`) when the trigger holds.

    `trigger` is `"time"` (`AT TIME t`, `value` in seconds of simulation time),
    `"clocktime"` (`AT CLOCKTIME t`, `value` in seconds after midnight) or `"node"`
    (`IF NODE n ABOVE|BELOW v`: `node` names it, `above` is True for ABOVE, `value` is in
    metres: the LEVEL of a tank, the PRESSURE head of a junction, the HEAD of a reservoir,
    as EPANET reads it).
    """

    link: str
    status: int | None
    setting: float | None
    trigger: str
    node: str | None = None
    above: bool | None = None
    value: float = 0.0


@dataclass(frozen=True)
class Premise:
    """One `IF`/`AND`/`OR` clause of a rule: `object id attribute relation value`.

    `conjunction` is `"IF"`, `"AND"` or `"OR"`; `object` is `"NODE"`, `"LINK"`, `"SYSTEM"`
    (`id` is `""` for the last); `relation` one of `=`, `<>`, `<`, `<=`, `>`, `>=` (`IS`,
    `NOT`, `BELOW` and `ABOVE` are normalised to these). `value` is in SI (m, m3/s, kW,
    seconds); a status is its code (`CLOSED`, `OPEN`, `ACTIVE`).

    `tolerance` is EPANET's equality tolerance of a numerical premise (`rules.c`, `checkvalue`:
    1e-3 in the unit of the file) in the premise's own SI unit; the reader sets it from the
    file's units. Built in Python it defaults to 1e-3 of the unit of an SI file (`LPS`, metres):
    1e-6 m3/s for `FLOW` and `DEMAND`, 1e-3 m for `HEAD`, `LEVEL` and `PRESSURE`, 1e-3 for a
    `SETTING`. It takes no part in comparing two premises.
    """

    conjunction: str
    object: str
    id: str
    attribute: str
    relation: str
    value: float
    tolerance: float | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.tolerance is None:
            small = 1e-6 if self.attribute in ("FLOW", "DEMAND") else 1e-3
            object.__setattr__(self, "tolerance", small)


@dataclass(frozen=True)
class Action:
    """A `THEN`/`ELSE` action: set `attribute` (`"STATUS"`: a status code, or `"SETTING"`:
    a number, in SI) of `link` to `value`."""

    link: str
    attribute: str
    value: float


@dataclass(frozen=True)
class Rule:
    """One `[RULES]` entry: the `premises` (in file order, joined by their conjunctions),
    the actions when they hold (`then_actions`) and when they do not (`else_actions`)."""

    name: str
    premises: tuple[Premise, ...]
    then_actions: tuple[Action, ...]
    else_actions: tuple[Action, ...]
    priority: float = 0.0


_STATUS_WORDS = {"CLOSED": CLOSED, "OPEN": OPEN, "ACTIVE": ACTIVE}


def status_code(word: str) -> int | None:
    """The status code of the word `OPEN`, `CLOSED` or `ACTIVE` (any case), else `None`."""
    return _STATUS_WORDS.get(word.upper())


def _fires_on_clock(control: Control, time, clocktime) -> Tensor | None:
    """Whether a time or clock-time `control` fires at `time` (s since the start), the
    simulation starting at `clocktime` (s after midnight): EPANET's `controls`
    (`hydraul.c`): `Htime == Time` (`AT TIME`) and `(Htime + Tstart) % SECperDAY == Time`
    (`AT CLOCKTIME`), on whole seconds. `None` when `time` is not given."""
    if time is None:
        return None
    t = torch.round(torch.as_tensor(time, dtype=torch.float64))
    if control.trigger == "time":
        return t == round(control.value)
    start = 0.0 if clocktime is None else float(clocktime)
    return torch.remainder(t + round(start), SECONDS_PER_DAY) == round(control.value)


def _fires_on_node(control: Control, levels, pressures, tested) -> Tensor | None:
    """Whether a node-triggered `control` fires. A tank's level is tested strictly
    (`level > value` ABOVE, `level < value` BELOW; `tested(control)` replaces that test where
    it returns a mask, as for a tank on a volume curve), a junction's PRESSURE head against
    the setting with EPANET's `Htol` (`hydsolver.c`, `pswitch`: `h <= hset + Htol` BELOW,
    `h >= hset - Htol` ABOVE). `None` when the node's value is not among those given."""
    node = control.node
    if node in levels:
        if tested is not None:
            mask = tested(control)
            if mask is not None:
                return mask
        value = torch.as_tensor(levels[node])
        return value > control.value if control.above else value < control.value
    if node in pressures:
        value = torch.as_tensor(pressures[node])
        if control.above:
            return value >= control.value - HTOL
        return value <= control.value + HTOL
    return None


def _action(control: Control, kind: str) -> tuple[int | None, float | None]:
    """`(status, setting)` a firing `control` imposes on a link of `kind` (`"PIPE"`,
    `"PUMP"` or `"VALVE"`); `None` keeps the link's own. The setting is in the control's
    units (SI), before `setting_scale`. EPANET 2.2 (`input3.c`,
    `controldata`; checked against wntr 1.5.0's EPANET): a pump OPENed runs at relative speed
    1, a pump given a setting runs at it and is CLOSED at 0, a valve given a setting is
    ACTIVE at it. A pump CLOSED keeps its speed here (EPANET sets it to 0, which changes no
    flow: the pump is shut)."""
    if control.status is not None:
        if kind == "PUMP" and control.status == OPEN:
            return OPEN, 1.0
        return control.status, None
    if kind == "PUMP":
        return (OPEN if control.setting > 0 else CLOSED), control.setting
    if kind == "VALVE":
        return ACTIVE, control.setting
    raise ValueError(
        f"control on link {control.link!r} sets a numeric setting, which only a pump or a "
        f"valve takes")


def apply_controls(
    controls: Sequence[Control], *, time=None, clocktime=0.0,
    levels: Mapping[str, Tensor] | None = None, pressures: Mapping[str, Tensor] | None = None,
    status: Tensor, setting: Tensor, link_names: Sequence[str], link_kinds: Sequence[str],
    setting_scale: Sequence[float] | None = None,
    tested: Callable[[Control], Tensor | None] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """`(status, setting, changed)`: the links' status and setting (`(..., n_links)`, in
    `link_names` order, of `link_kinds` `"PIPE"`, `"PUMP"` or `"VALVE"`) after every control
    whose condition holds is applied, in file order (a later control on a link wins), and per
    instance whether any status or setting changed. EPANET 2.2, `hydraul.c`, `controls`, and
    `hydsolver.c`, `pswitch`.

    A control is tested only when its inputs are given: `time` (s since the start; `clocktime`
    is the start's time of day, s after midnight) for `AT TIME` and `AT CLOCKTIME`; `levels`
    (tank name to level, m) and `pressures` (junction name to pressure head, m) for
    `IF NODE` (a reservoir has no control: its EPANET test compares volumes of area 0 and
    always holds, so `WaterNetwork.check` refuses it).
    `setting_scale` (per link, default 1) puts a control's numeric setting in the units of
    the `setting` it acts on (a PBV's, PRV's or PSV's head in the layer's potential unit:
    `head_scale` on a Darcy-Weisbach pressure layer). EPANET tests the time, tank and reservoir
    controls at the start of a step and the junction-pressure ones after a converged solve
    (`pswitch`, within `Htol` of the setting); `changed` then asks for a new solve, the
    caller's to make. `tested` replaces the tank-level test (see `_fires_on_node`)."""
    levels, pressures = levels or {}, pressures or {}
    index = {name: j for j, name in enumerate(link_names)}
    parts = [status.shape[:-1], setting.shape[:-1]]
    parts += [torch.as_tensor(v).shape for m in (levels, pressures) for v in m.values()]
    if time is not None:
        parts.append(torch.as_tensor(time).shape)
    batch = torch.broadcast_shapes(*parts)
    status = status.expand(batch + status.shape[-1:]).clone()
    setting = setting.expand(batch + setting.shape[-1:]).clone()
    status0, setting0 = status.clone(), setting.clone()
    for control in controls:
        fires = (_fires_on_clock(control, time, clocktime) if control.trigger != "node"
                 else _fires_on_node(control, levels, pressures, tested))
        if fires is None:
            continue
        j = index[control.link]
        new_status, new_setting = _action(control, link_kinds[j])
        if new_setting is not None and setting_scale is not None:
            new_setting = new_setting * setting_scale[j]
        old_status, old_setting = status[..., j], setting[..., j]
        s = old_status if new_status is None else torch.where(
            fires, torch.full_like(old_status, float(new_status)), old_status)
        k = old_setting if new_setting is None else torch.where(
            fires, torch.full_like(old_setting, float(new_setting)), old_setting)
        status[..., j], setting[..., j] = s, k
    # a change is a link left otherwise than it was found: controls that undo one another
    # (an OPEN and a CLOSED that both hold) change nothing and ask for no new solve
    changed = ((status != status0) | (setting != setting0)).any(-1)
    return status, setting, changed


def next_control_time(
    controls: Sequence[Control], *, time, clocktime=0.0,
    levels: Mapping[str, Tensor], rates: Mapping[str, Tensor],
    crossing: Callable[[Control], Tensor | None] | None = None,
    status: Tensor | None = None, setting: Tensor | None = None,
    link_names: Sequence[str] = (), link_kinds: Sequence[str] = (),
    setting_scale: Sequence[float] | None = None,
) -> Tensor:
    """Seconds from `time` to the next control that would fire: the next time or clock-time
    control, or the tank level that, changing linearly at `rates` (m/s), reaches a
    level control's trigger; `inf` where there is none. EPANET 2.2, `hydraul.c`,
    `controltimestep`: a time control `Time - Htime` ahead (a clock-time one at the next
    `Time` of the day, a whole day later if it has passed; a distance of 0 is skipped), a
    level control `(grade - h) / rate` ahead when the level moves towards it (rising to an
    ABOVE, falling to a BELOW), ROUNDED to whole seconds as EPANET rounds it
    (`t = ROUND(v / q)`); a crossing that rounds to 0 is skipped, as EPANET skips it. The
    rounded step may end up to half a second short of the crossing: EPANET's level test
    (`controls`, by volume with one second's inflow as margin; `TankLevels`) fires the
    control at the next step's start all the same. `crossing(control)` replaces the
    straight-line time of a level control (a tank on a volume curve) by its time in
    seconds, or `None` to keep it.

    Given the links' `status` and `setting` (as `apply_controls` takes them), a control that
    would change nothing, its link already as it says, does not end a step: EPANET revises
    the step only "if control action causes a status or setting change" (`controltimestep`).
    Without them every control counts."""
    t = torch.as_tensor(time, dtype=torch.float64)
    best = torch.full_like(t, torch.inf)
    for control in controls:
        if control.trigger == "node":
            if control.node not in levels:
                continue
            level = torch.as_tensor(levels[control.node], dtype=torch.float64)
            rate = torch.as_tensor(rates[control.node], dtype=torch.float64)
            ahead = None if crossing is None else crossing(control)
            if ahead is None:
                safe = torch.where(rate == 0, torch.ones_like(rate), rate)
                ahead = (control.value - level) / safe
            toward = ((rate > 0) & (level < control.value) if control.above
                      else (rate < 0) & (level > control.value))
            # EPANET's ROUND (`types.h`): (long)(x + 0.5) for x >= 0
            ahead = torch.where(toward, torch.floor(ahead + 0.5), torch.inf)
        elif control.trigger == "time":
            ahead = round(control.value) - t
        else:
            now = torch.remainder(t + round(float(clocktime)), SECONDS_PER_DAY)
            ahead = torch.remainder(round(control.value) - now, SECONDS_PER_DAY)
        ahead = torch.where(ahead > 0, ahead, torch.inf)
        if status is not None:
            j = link_names.index(control.link)
            new_status, new_setting = _action(control, link_kinds[j])
            if new_setting is not None and setting_scale is not None:
                new_setting = new_setting * setting_scale[j]
            acts = None
            if new_status is not None:
                acts = status[..., j] != new_status
            if new_setting is not None:
                moved = setting[..., j] != new_setting
                acts = moved if acts is None else acts | moved
            if acts is None:
                continue                      # it changes nothing: it never ends a step
            ahead = torch.where(acts, ahead, torch.inf)
        best = torch.minimum(best, ahead)
    return best
