"""Time-varying lateral inflows of a SWMM model: `[TIMESERIES]`, `[PATTERNS]`, `[INFLOWS]`
and `[DWF]`, as one `InflowSchedule` a dynamic-wave run consumes.

``schedule.driver(names)`` is the ``inflow(t)`` callable of `simulate_dynamic_wave` (``(n,)``
m3/s over the nodes `names`); `simulate_dynamic_wave` uses the network's own schedule when
it is given no ``inflow``. What follows is SWMM 5.2.4's behaviour, each point checked against
the engine through pyswmm (the tests in ``tests/verification/test_sewer_inflows_parity.py``).

Sampling time. SWMM adds a step's lateral inflow with the date at the START of the step plus
1 ms (`getDateTime`, swmm5.c:1551; `routing.c` addExternalInflows / addDryWeatherInflows), so
``inflow(t)`` is evaluated at the simulation clock ``START_DATE + START_TIME + t + 0.001 s``
(`SAMPLE_OFFSET`). The month, day of week and hour of a pattern are those of that clock.

Time series (`table.c` ``table_tseriesLookup``, ``extend`` false; `input.c` readTimeseries).
Values are linearly interpolated between entries and are 0 before the first and after the last
entry. A series with no date on any entry gives times in hours (decimal, or ``HH:MM``) since
the START of the simulation (START_TIME included); with dates, a date applies to its entry
and to the entries after it that carry only a time, and the times are calendar times. The
times must strictly increase. ``name FILE path`` reads the entries from a file (``[date]
time value`` per line, ``;`` comments), the path relative to the ``.inp`` file's folder.

External inflow (`inflow.c`). ``Node FLOW Series Type Mfactor Sfactor Baseline Pattern``: the
flow is ``Sfactor * series(t) + Baseline * pattern(t)`` in the file's flow units (the pattern
factor is 1 without a pattern; Sfactor scales the series only, the pattern the baseline
only). SWMM IGNORES Mfactor for a FLOW inflow (it converts pollutant mass flows), so it is
read and has no effect here either. A repeated line for the same node and constituent
replaces the earlier one. Flows are converted to m3/s with the file's flow unit.

Dry-weather flow (`inflow.c` inflow_getDwfInflow). ``Node FLOW Average Pat1..Pat4``: the average
flow times the monthly, daily and hourly (or, on Saturday and Sunday, weekend) pattern
factors. SWMM reads the first four pattern fields only and keeps ONE pattern per type (a
later pattern of the same type replaces an earlier one, they do not multiply); a weekend
pattern takes the place of the hourly one on Saturday and Sunday, and is 1 on other days.
A pattern with fewer factors than its type needs has 1 for the missing ones. DAILY index 0 is
Sunday.

Both sources are dropped when their flow is smaller than SWMM's ``FLOW_TOL`` of 1e-5 cfs
(`routing.c`): the external flow of one node (series plus baseline) and its dry-weather flow
are each tested alone, on their absolute value.
"""

from __future__ import annotations

import bisect
import dataclasses
import datetime
import re
from collections.abc import Callable, Sequence

import torch

from noodl.apps.sewer.xsect_tables import QCF

F64 = torch.float64

#: SWMM reads a step's lateral inflow at the step's start plus 1 ms (swmm5.c:1551), s.
SAMPLE_OFFSET = 0.001
#: routing.c `FLOW_TOL`: an inflow smaller than this (1e-5 cfs, in m3/s) is zero.
FLOW_TOL = 1e-5 * QCF
#: The factors a pattern of each type holds (SWMM's ``PatternTypeWords``).
PATTERN_SIZES = {"monthly": 12, "daily": 7, "hourly": 24, "weekend": 24}

_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_TOKEN = re.compile(r'"[^"]*"|\S+')


def tokens(raw: str) -> list[str]:
    """The fields of an `.inp` line with its ``;`` comment removed, a double-quoted field
    being one field with its quotes removed (SWMM's tokenizer: ``""`` is an empty field)."""
    return [t[1:-1] if t.startswith('"') and t.endswith('"') and len(t) >= 2 else t
            for t in _TOKEN.findall(raw.split(";", 1)[0])]


def parse_date(text: str) -> datetime.date | None:
    """The date `text` (``M/D/Y``, ``M-D-Y``, ``Y-M-D``, ``D-MON-Y`` or ``MON-D-Y``, the
    formats `datetime.c` reads), or ``None`` when `text` is not three fields joined by ``/``
    or ``-``. A three-field text that is not a calendar date raises ValueError."""
    parts = re.split(r"[/-]", text)
    if len(parts) != 3 or not all(parts):
        return None
    month_at = [i for i, p in enumerate(parts) if p.isalpha()]
    try:
        if month_at:
            m_i = month_at[0]
            if len(month_at) != 1 or m_i == 2 or parts[m_i][:3].upper() not in _MONTHS:
                raise ValueError
            month = _MONTHS[parts[m_i][:3].upper()]
            day, year = (int(p) for i, p in enumerate(parts) if i != m_i)   # MON-D-Y, D-MON-Y
        elif len(parts[0]) == 4:                         # Y-M-D
            year, month, day = (int(p) for p in parts)
        else:                                            # M/D/Y
            month, day, year = (int(p) for p in parts)
        return datetime.date(year, month, day)
    except ValueError:
        raise ValueError(f"{text!r} is not a date (M/D/Y, Y-M-D or D-MON-Y)") from None


def parse_clock(text: str) -> float | None:
    """Seconds for a SWMM time of day or interval: decimal hours (``1.5``) or ``H:M[:S]``;
    ``None`` when `text` is neither."""
    try:
        return float(text) * 3600.0
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = values[0] * 3600.0 + values[1] * 60.0 + (values[2] if len(parts) == 3 else 0.0)
    return seconds if all(v >= 0.0 for v in values) else None


@dataclasses.dataclass(frozen=True)
class TimeSeries:
    """A SWMM time series: `times` (s since the simulation START, strictly increasing) and
    `values`, in the unit of the series. ``series(t)`` is the linear interpolation, 0 outside
    the series (before its first and after its last time)."""

    name: str
    times: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.times or len(self.times) != len(self.values):
            raise ValueError(f"time series {self.name!r}: needs as many values as times, "
                             f"at least one")
        if any(b <= a for a, b in zip(self.times, self.times[1:], strict=False)):
            raise ValueError(f"time series {self.name!r}: times must strictly increase "
                             f"(SWMM's ERR_CURVE_SEQUENCE)")

    def __call__(self, t: float) -> float:
        times = self.times
        if t < times[0] or t > times[-1]:
            return 0.0
        k = bisect.bisect_right(times, t)
        if k >= len(times):
            return self.values[-1]
        t0, t1 = times[k - 1], times[k]
        y0, y1 = self.values[k - 1], self.values[k]
        return y0 + (y1 - y0) * (t - t0) / (t1 - t0)

    def held(self, t: float) -> float:
        """The value at `t` with the series extended on both ends: its first value before
        its first time and its last after its last (SWMM's ``table_tseriesLookup`` with
        ``extend`` TRUE, table.c:779, which a TIMESERIES outfall's stage uses, node.c:1457)."""
        times = self.times
        if t <= times[0]:
            return self.values[0]
        if t >= times[-1]:
            return self.values[-1]
        return self(t)


@dataclasses.dataclass(frozen=True)
class TimePattern:
    """A SWMM time pattern of `kind` ``monthly`` (12 factors), ``daily`` (7, Sunday first),
    ``hourly`` (24) or ``weekend`` (24, used on Saturday and Sunday only). `factors` holds
    exactly the kind's count (missing ones 1, as in SWMM)."""

    name: str
    kind: str
    factors: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in PATTERN_SIZES:
            raise ValueError(f"pattern {self.name!r}: kind must be one of "
                             f"{sorted(PATTERN_SIZES)}, got {self.kind!r}")
        if len(self.factors) != PATTERN_SIZES[self.kind]:
            raise ValueError(f"pattern {self.name!r}: a {self.kind} pattern has "
                             f"{PATTERN_SIZES[self.kind]} factors, got {len(self.factors)}")

    def factor(self, clock: datetime.datetime) -> float:
        """The factor at the calendar time `clock` (month, day of week or hour of it; the
        weekend pattern is 1 on a weekday)."""
        if self.kind == "monthly":
            return self.factors[clock.month - 1]
        sunday_first = (clock.weekday() + 1) % 7
        if self.kind == "daily":
            return self.factors[sunday_first]
        if self.kind == "weekend" and sunday_first not in (0, 6):
            return 1.0
        return self.factors[clock.hour]


@dataclasses.dataclass(frozen=True)
class ExternalInflow:
    """A node's `[INFLOWS]` FLOW entry: ``scale * series(t) + baseline * pattern(t)`` (m3/s;
    `scale` is SWMM's Sfactor times the flow-unit conversion, `baseline` is in m3/s)."""

    node: str
    series: TimeSeries | None = None
    scale: float = 1.0
    baseline: float = 0.0
    pattern: TimePattern | None = None

    def flow(self, t: float, clock: datetime.datetime) -> float:
        """The flow (m3/s) at `t` s since START and the calendar time `clock`."""
        q = self.baseline * (1.0 if self.pattern is None else self.pattern.factor(clock))
        if self.series is not None:
            q += self.scale * self.series(t)
        return q


@dataclasses.dataclass(frozen=True)
class DryWeatherInflow:
    """A node's `[DWF]` FLOW entry: `average` (m3/s) times the factors of its `patterns`
    (at most one per kind: monthly, daily, hourly and weekend)."""

    node: str
    average: float
    patterns: tuple[TimePattern, ...] = ()

    def __post_init__(self) -> None:
        kinds = [p.kind for p in self.patterns]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"dry-weather flow at {self.node!r}: one pattern per kind "
                             f"(SWMM keeps the last of a kind), got {kinds}")

    def flow(self, clock: datetime.datetime) -> float:
        """The flow (m3/s) at the calendar time `clock`."""
        by_kind = {p.kind: p for p in self.patterns}
        f = 1.0
        for kind in ("monthly", "daily"):
            if kind in by_kind:
                f *= by_kind[kind].factor(clock)
        hourly = by_kind.get("hourly")
        weekend = by_kind.get("weekend")
        if weekend is not None and (clock.weekday() + 1) % 7 in (0, 6):
            f *= weekend.factor(clock)
        elif hourly is not None:
            f *= hourly.factor(clock)
        return self.average * f


@dataclasses.dataclass(frozen=True)
class InflowSchedule:
    """The lateral inflows of a model: `external` and `dry_weather` entries on the
    calendar that starts at `start` (SWMM's START_DATE and START_TIME).

    ``flows(t)`` is ``{node: m3/s}`` for the routing step that starts `t` s into the run,
    ``driver(names)`` the ``inflow(t)`` callable `simulate_dynamic_wave` takes, and
    ``at(t, names)`` one such vector. Each source of a node is dropped when its flow is below
    `tolerance` (`FLOW_TOL`), as SWMM does (module docstring)."""

    start: datetime.datetime
    external: tuple[ExternalInflow, ...] = ()
    dry_weather: tuple[DryWeatherInflow, ...] = ()
    tolerance: float = FLOW_TOL

    @property
    def nodes(self) -> tuple[str, ...]:
        """The nodes that receive an inflow, in order of first appearance."""
        return tuple(dict.fromkeys([e.node for e in self.external]
                                   + [d.node for d in self.dry_weather]))

    def flows(self, t: float) -> dict[str, float]:
        """`{node: m3/s}` of the step that starts `t` s after START (sampled at
        ``t + SAMPLE_OFFSET``, see the module docstring)."""
        ts = t + SAMPLE_OFFSET
        clock = self.start + datetime.timedelta(seconds=ts)
        out: dict[str, float] = {}
        for entry in self.external:
            q = entry.flow(ts, clock)
            out[entry.node] = out.get(entry.node, 0.0) + (q if abs(q) >= self.tolerance
                                                          else 0.0)
        for dwf in self.dry_weather:
            q = dwf.flow(clock)
            out[dwf.node] = out.get(dwf.node, 0.0) + (q if abs(q) >= self.tolerance else 0.0)
        return out

    def at(self, t: float, names: Sequence[str]) -> torch.Tensor:
        """The ``(len(names),)`` m3/s inflow of the step that starts at `t`; a node of the
        schedule missing from `names` raises ValueError."""
        index = {n: i for i, n in enumerate(names)}
        missing = sorted(set(self.nodes) - set(index))
        if missing:
            raise ValueError(f"InflowSchedule: node(s) {missing} are not among the "
                             f"model's nodes")
        return self._vector(self.flows(t), index, len(index))

    @staticmethod
    def _vector(flows: dict[str, float], index: dict[str, int], n: int) -> torch.Tensor:
        q = torch.zeros(n, dtype=F64)
        for node, value in flows.items():
            q[index[node]] = value
        return q

    def driver(self, names: Sequence[str]) -> Callable[[float], torch.Tensor]:
        """``inflow(t)`` for `simulate_dynamic_wave`: the ``(n,)`` m3/s vector over `names`
        (the model's nodes, ``list(model.net.nodes)``) of the step that starts at `t`. A
        schedule node missing from `names` raises ValueError here."""
        index = {n: i for i, n in enumerate(names)}
        missing = sorted(set(self.nodes) - set(index))
        if missing:
            raise ValueError(f"InflowSchedule: node(s) {missing} are not among the "
                             f"model's nodes")
        return lambda t: self._vector(self.flows(float(t)), index, len(index))


def start_datetime(date: datetime.date, seconds: float) -> datetime.datetime:
    """`date` at `seconds` after midnight (START_DATE and START_TIME)."""
    return datetime.datetime.combine(date, datetime.time()) + datetime.timedelta(seconds=seconds)


def series_from_entries(name: str, entries: Sequence[tuple[datetime.date | None, float, float]],
                        start: datetime.datetime) -> TimeSeries:
    """A `TimeSeries` from `(date, seconds, value)` entries: `date` is the calendar date of
    the entry (``None`` for a series without dates, whose `seconds` are then since START);
    dated times become seconds since `start`. Mixing dated and undated entries raises
    ValueError."""
    dated = [d is not None for d, _, _ in entries]
    if any(dated) and not all(dated):
        raise ValueError(f"time series {name!r}: a time before the first date; give the "
                         f"date on its first entry or on none")
    times = [s if d is None else (start_datetime(d, s) - start).total_seconds()
             for d, s, _ in entries]
    return TimeSeries(name, tuple(times), tuple(v for _, _, v in entries))


def patterns_from_lines(lines: Sequence[tuple[int, list[str]]], path) -> dict[str, TimePattern]:
    """`{name: TimePattern}` from the tokenized `[PATTERNS]` lines `(line number, tokens)`:
    ``Name Type f1 f2 ...`` and continuation lines ``Name f...`` (SWMM's readPattern). A
    pattern may give fewer factors than its type (the rest are 1) and extra ones are
    ignored."""
    kinds: dict[str, str] = {}
    values: dict[str, list[float]] = {}
    for number, toks in lines:
        name = toks[0]
        rest = toks[1:]
        if name not in kinds:
            if not rest or rest[0].lower() not in PATTERN_SIZES:
                raise ValueError(f"{path}: line {number}: pattern {name!r} needs its type "
                                 f"({', '.join(k.upper() for k in PATTERN_SIZES)}) first")
            kinds[name] = rest[0].lower()
            values[name] = []
            rest = rest[1:]
        for tok in rest:
            try:
                values[name].append(float(tok))
            except ValueError:
                raise ValueError(f"{path}: line {number}: pattern {name!r} factor must be a "
                                 f"number, got {tok!r}") from None
    out = {}
    for name, kind in kinds.items():
        size = PATTERN_SIZES[kind]
        factors = (values[name] + [1.0] * size)[:size]
        out[name] = TimePattern(name, kind, tuple(factors))
    return out


def timeseries_entries(toks: Sequence[str], number: int, path,
                       last: datetime.date | None = None
                       ) -> tuple[list[tuple[datetime.date | None, float, float]],
                                  datetime.date | None]:
    """`(entries, last)`: the `(date, seconds, value)` entries of the inline fields `toks`
    after the series name, ``[date] time value`` repeated, a date applying to the entries
    after it, also on later lines (`last` is the date in force before this line, and the
    one in force after it is returned; SWMM's readTimeseries, whose state machine this
    follows)."""
    entries: list[tuple[datetime.date | None, float, float]] = []
    k = 0
    while k < len(toks):
        try:
            date = parse_date(toks[k])
        except ValueError as err:
            raise ValueError(f"{path}: line {number}: {err}") from None
        if date is not None:
            last = date
            k += 1
        if k + 1 >= len(toks):
            raise ValueError(f"{path}: line {number}: a time series entry needs a time and a "
                             f"value, got {' '.join(toks)!r}")
        seconds = parse_clock(toks[k])
        if seconds is None:
            raise ValueError(f"{path}: line {number}: time series time must be hours or "
                             f"HH:MM, got {toks[k]!r}")
        try:
            value = float(toks[k + 1])
        except ValueError:
            raise ValueError(f"{path}: line {number}: time series value must be a number, "
                             f"got {toks[k + 1]!r}") from None
        entries.append((last, seconds, value))
        k += 2
    return entries, last


def read_series_file(name: str, file, path, start: datetime.datetime) -> TimeSeries:
    """The `TimeSeries` `name` stored in the file `file` (``[date] time value`` per line,
    ``;`` comments; `path` is the `.inp` it was named in, for messages). A missing file is
    refused by name."""
    try:
        text = file.read_text()
    except OSError as err:
        raise ValueError(f"{path}: time series {name!r} reads the file {str(file)!r}, which "
                         f"cannot be read ({err.strerror or type(err).__name__})") from None
    entries: list[tuple[datetime.date | None, float, float]] = []
    last: datetime.date | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        toks = tokens(raw)
        if not toks:
            continue
        where = f"{file}: line {number}: time series {name!r}"
        if len(toks) not in (2, 3):
            raise ValueError(f"{where} needs 'time value' or 'date time value' per line, got "
                             f"{len(toks)} fields")
        try:
            date = parse_date(toks[0]) if len(toks) == 3 else None
        except ValueError as err:
            raise ValueError(f"{where}: {err}") from None
        if len(toks) == 3 and date is None:
            raise ValueError(f"{where}: {toks[0]!r} is not a date")
        last = date if date is not None else last
        seconds = parse_clock(toks[-2])
        if seconds is None:
            raise ValueError(f"{where}: bad time {toks[-2]!r}")
        try:
            value = float(toks[-1])
        except ValueError:
            raise ValueError(f"{where}: value must be a number, got {toks[-1]!r}") from None
        entries.append((last, seconds, value))
    if not entries:
        raise ValueError(f"{path}: time series {name!r}: the file {str(file)!r} holds no data")
    try:
        return series_from_entries(name, entries, start)
    except ValueError as err:
        raise ValueError(f"{path}: {err} (file {str(file)!r})") from None
