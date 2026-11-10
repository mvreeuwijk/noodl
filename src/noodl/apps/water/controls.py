"""Link status codes and the data classes of EPANET's `[CONTROLS]` and `[RULES]`.

Data classes only: this module holds what the reader produces and what the builder will
act on, and imports nothing from the rest of the application.

`CLOSED`, `OPEN` and `ACTIVE` are the codes of a link's status: `CLOSED = 0`, `OPEN = 1`,
`ACTIVE = 2` (a valve acting on its setting).
"""

from __future__ import annotations

from dataclasses import dataclass

CLOSED, OPEN, ACTIVE = 0, 1, 2


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
    """

    conjunction: str
    object: str
    id: str
    attribute: str
    relation: str
    value: float


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
