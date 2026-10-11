"""Time in a building simulation: the calendar, CONTAM's day and week schedules, time series.

Time `t` is seconds since the simulation start. A `Calendar` places t = 0 on a day of the year
(1-365; CONTAM's weather and ambient files carry no year and no 29 February) at a time of day,
and gives every day a CONTAM day type (1-12), which selects one of a week schedule's twelve
day schedules. With a weather file, the day type comes from the file's per-day `Dtype` column.
Without one, it starts at the steady weather's day type and cycles through types 1-7: the day
after a day of type d has type (d mod 7) + 1. This was measured on ContamX 3.4.1.7 (start 6
gives 6, 7, 1, ...; start 9 gives 9, 3, 4, ...).

Evaluation instant (ContamX, measured): every value is taken at the END of a step, and where a
schedule jumps, the value AT the jump is the one before it (the left limit). A fan scheduled on
at 08:00 is off over the step ending 08:00 and on over the step ending 08:05. At t = 0 (the
initial airflow solve) the value at the start time itself is used. Every `Signal.value(t,
calendar)` follows this rule: left limit for t > 0, right value at t == 0.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

Tensor = torch.Tensor
F64 = torch.float64
DAY = 86400.0
N_DAY_TYPES = 12
_WEEK = 7
_SHAPES = ("rectangular", "trapezoidal")
#: A time of day within this many ulps (of the seconds since new year) of a midnight or a
#: schedule knot is taken as that instant.
_ROUND_OFF_ULPS = 8


def _as_f64(values) -> Tensor:
    """float64 values, keeping a caller's autograd graph."""
    if isinstance(values, Tensor):
        return values if values.dtype == F64 else values.to(F64)
    return torch.as_tensor(values, dtype=F64)


def _piecewise(
    times: list[float], values: Tensor, s: float, *, linear: bool, left: bool
) -> Tensor:
    """`values` (rows at the increasing `times`) evaluated at `s` within [times[0], times[-1]].

    `linear` interpolates between rows (continuous, so `left` does not matter); otherwise each
    row is held until the next time, and at a jump `left` takes the earlier row and `not left`
    the later one. Differentiable in `values`. Shared by `DaySchedule` and `TimeSeries`, and by
    any other signal built from tabulated values.
    """
    n = len(times)
    if linear:
        i = min(max(bisect.bisect_right(times, s) - 1, 0), n - 2)
        w = (s - times[i]) / (times[i + 1] - times[i])
        return values[i] * (1.0 - w) + values[i + 1] * w
    i = (bisect.bisect_left(times, s) if left else bisect.bisect_right(times, s)) - 1
    return values[min(max(i, 0), n - 1)]


@runtime_checkable
class Signal(Protocol):
    """A time-varying input: `value(t, calendar)`, a float64 tensor at `t` s since the
    simulation start (left limit at a jump for t > 0, right value at t == 0), and
    `breakpoints(lo, hi, calendar)`, the times in (lo, hi) where it jumps or kinks."""

    def value(self, t: float, calendar: Calendar) -> Tensor: ...

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]: ...


class Calendar:
    """Where t = 0 sits in the year, and the CONTAM day type (1-12) of every day.

    `start_day` is the day of the year (1-365) of t = 0, `start_time` its time of day (s,
    [0, 86400)), `start_day_type` the day type of the start day. `day_types` maps {day of
    year: day type} when a weather file supplies them; every simulated day must then be in
    it. Otherwise the types cycle through 1-7 from `start_day_type` (module docstring).
    """

    def __init__(
        self,
        *,
        start_day: int = 1,
        start_time: float = 0.0,
        start_day_type: int = 1,
        day_types: Mapping[int, int] | None = None,
    ) -> None:
        if not 1 <= int(start_day) <= 365:
            raise ValueError(
                f"Calendar: start_day must be a day of the year 1-365, got {start_day}"
            )
        if not 0.0 <= float(start_time) < DAY:
            raise ValueError(f"Calendar: start_time must be in [0, 86400) s, got {start_time}")
        if not 1 <= int(start_day_type) <= N_DAY_TYPES:
            raise ValueError(f"Calendar: start_day_type must be 1-12, got {start_day_type}")
        types = {int(d): int(k) for d, k in (day_types or {}).items()}
        bad = {d: k for d, k in types.items() if not 1 <= k <= N_DAY_TYPES}
        if bad:
            raise ValueError(f"Calendar: day types must be 1-12; got {bad}")
        self.start_day, self.start_time = int(start_day), float(start_time)
        self.start_day_type, self.day_types = int(start_day_type), types

    def __repr__(self) -> str:
        return (
            f"Calendar(start_day={self.start_day}, start_time={self.start_time}, "
            f"start_day_type={self.start_day_type}, day_types={self.day_types})"
        )

    def seconds_from_new_year(self, t: float) -> float:
        """Seconds from 00:00 on 1 January to simulation time `t`."""
        return (self.start_day - 1) * DAY + self.start_time + float(t)

    def round_off(self, t: float) -> float:
        """The round-off (s) of a time of day computed at `t`: a time of day this close to a
        midnight or a schedule knot IS that instant. A breakpoint is simulation time
        midnight + knot, which need not survive the sum `seconds_from_new_year` exactly; within
        this tolerance it still evaluates to its left limit."""
        return _ROUND_OFF_ULPS * math.ulp(max(abs(self.seconds_from_new_year(t)), DAY))

    def day_and_time(self, t: float, *, left: bool) -> tuple[int, float]:
        """(day of year, seconds after midnight) at `t`. With `left`, an exact midnight is the
        END of the previous day, (day - 1, 86400.0): a left limit sees the old day. A time
        within `round_off(t)` of a midnight is that midnight."""
        s = self.seconds_from_new_year(t)
        day = math.floor(s / DAY) + 1
        tod = s - (day - 1) * DAY
        tol = self.round_off(t)
        if tod <= tol:
            tod = 0.0
        elif DAY - tod <= tol:
            day, tod = day + 1, 0.0
        if left and tod == 0.0 and day > 1:
            day, tod = day - 1, DAY
        if day > 365:
            raise ValueError(
                f"Calendar: t = {float(t)} s falls after 31 December; CONTAM's weather and "
                f"ambient files carry no year, so a run cannot cross the new year"
            )
        return day, tod

    def day_type(self, day: int) -> int:
        """The CONTAM day type (1-12) of `day` (day of the year)."""
        if self.day_types:
            try:
                return self.day_types[int(day)]
            except KeyError as exc:
                raise KeyError(
                    f"Calendar: day {day} has no day type; the weather file covers days "
                    f"{min(self.day_types)}..{max(self.day_types)}"
                ) from exc
        i = int(day) - self.start_day
        if i < 0:
            raise ValueError(f"Calendar: day {day} is before the start day {self.start_day}")
        if i == 0:
            return self.start_day_type
        return (self.start_day_type - 1 + i) % _WEEK + 1


class DaySchedule:
    """One day of values at times of day: CONTAM's day schedule.

    `times` are seconds after midnight, strictly increasing from 0 (00:00:00) to 86400
    (24:00:00). `values` are in the scheduled quantity's own unit: dimensionless for a
    multiplier, K for a zone temperature. `shape="rectangular"` holds each value until the
    next time. `"trapezoidal"` interpolates linearly between them.
    """

    def __init__(self, times, values, *, shape: str = "rectangular", name: str = "") -> None:
        t = torch.as_tensor(times, dtype=F64).reshape(-1)
        v = _as_f64(values).reshape(-1)
        label = f"DaySchedule {name!r}" if name else "DaySchedule"
        if shape not in _SHAPES:
            raise ValueError(f"{label}: shape must be one of {_SHAPES}, got {shape!r}")
        if t.numel() < 2 or float(t[0]) != 0.0:
            raise ValueError(f"{label}: the first time must be 00:00:00 (0 s)")
        if float(t[-1]) != DAY:
            raise ValueError(f"{label}: the last time must be 24:00:00 (86400 s)")
        if bool((t[1:] <= t[:-1]).any()):
            raise ValueError(f"{label}: times must be strictly increasing, got {t.tolist()}")
        if v.numel() != t.numel():
            raise ValueError(f"{label}: {t.numel()} times but {v.numel()} values")
        self.times, self.values, self.shape, self.name = t, v, shape, name
        self._t = t.tolist()

    def __repr__(self) -> str:
        return f"DaySchedule({self.name!r}, {self.shape}, {len(self._t)} points)"

    def at(self, seconds: float, *, left: bool) -> Tensor:
        """The value at `seconds` after midnight (0..86400); `left` takes the left limit."""
        s = float(seconds)
        if not 0.0 <= s <= DAY:
            raise ValueError(f"DaySchedule {self.name!r}: {s} s is not a time of day")
        return _piecewise(self._t, self.values, s, linear=self.shape == "trapezoidal", left=left)

    def knots(self) -> list[float]:
        """The interior times (s after midnight) where the value jumps or kinks."""
        return self._t[1:-1]


class WeekSchedule:
    """CONTAM's week schedule: one `DaySchedule` per day type (1-12).

    `day_schedules` are the distinct day schedules. `day_types` has 12 entries; entry k is
    the 1-based position in `day_schedules` used on day type k + 1 (CONTAM's `j1 ... j12`).
    The default uses the first day schedule every day.
    """

    def __init__(
        self,
        day_schedules: Sequence[DaySchedule],
        day_types: Sequence[int] = (1,) * N_DAY_TYPES,
        *,
        name: str = "",
    ) -> None:
        days = list(day_schedules)
        js = [int(j) for j in day_types]
        label = f"WeekSchedule {name!r}" if name else "WeekSchedule"
        if len(js) != N_DAY_TYPES:
            raise ValueError(f"{label}: needs 12 day types (CONTAM's j1..j12), got {len(js)}")
        for k, j in enumerate(js, start=1):
            if not 1 <= j <= len(days):
                raise ValueError(
                    f"{label}: day type {k} names day schedule {j}, but only {len(days)} given"
                )
        self.day_schedules, self.day_types, self.name = days, js, name

    def __repr__(self) -> str:
        return f"WeekSchedule({self.name!r}, day_types={self.day_types})"

    def day(self, day_type: int) -> DaySchedule:
        """The day schedule used on `day_type` (1-12)."""
        return self.day_schedules[self.day_types[int(day_type) - 1] - 1]

    def value(self, t: float, calendar: Calendar) -> Tensor:
        left = float(t) > 0.0
        day, tod = calendar.day_and_time(t, left=left)
        schedule = self.day(calendar.day_type(day))
        tol = calendar.round_off(t)
        knot = min(schedule.knots(), key=lambda k: abs(k - tod), default=None)
        if knot is not None and abs(knot - tod) <= tol:
            tod = knot                      # a breakpoint, up to round-off: the knot itself
        return schedule.at(tod, left=left)

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]:
        first = calendar.day_and_time(lo, left=False)[0]
        last = calendar.day_and_time(hi, left=True)[0]
        origin = calendar.seconds_from_new_year(0.0)
        out: set[float] = set()
        for day in range(first, last + 1):
            midnight = (day - 1) * DAY - origin
            out.add(midnight)
            out.update(midnight + k for k in self.day(calendar.day_type(day)).knots())
        return sorted(p for p in out if lo < p < hi)


class TimeSeries:
    """Values at times since the simulation start (s): a measured or computed input.

    `values` has shape `(len(times), *trailing)` and may carry gradients (a calibrated input).
    `interpolation="linear"` interpolates; `"step"` holds each value until the next time (the
    left limit at a jump, as for a rectangular schedule). Never extrapolated: a time outside
    `times` raises.
    """

    def __init__(self, times, values, *, interpolation: str = "linear", name: str = "") -> None:
        t = torch.as_tensor(times, dtype=F64).reshape(-1)
        v = _as_f64(values)
        label = f"TimeSeries {name!r}" if name else "TimeSeries"
        if interpolation not in ("linear", "step"):
            raise ValueError(f"{label}: interpolation must be 'linear' or 'step'")
        if t.numel() < 2 or bool((t[1:] <= t[:-1]).any()):
            raise ValueError(f"{label}: needs at least two strictly increasing times")
        if v.dim() == 0 or v.shape[0] != t.numel():
            raise ValueError(f"{label}: values need {t.numel()} rows, got {tuple(v.shape)}")
        self.times, self.values, self.interpolation, self.name = t, v, interpolation, name
        self._t = t.tolist()

    def __repr__(self) -> str:
        return f"TimeSeries({self.name!r}, {len(self._t)} points, {self.interpolation})"

    def value(self, t: float, calendar: Calendar | None = None) -> Tensor:
        s = float(t)
        if not self._t[0] <= s <= self._t[-1]:
            raise ValueError(
                f"TimeSeries {self.name!r}: t = {s} s is outside its times "
                f"[{self._t[0]}, {self._t[-1]}] s; a time series is never extrapolated"
            )
        return _piecewise(
            self._t, self.values, s, linear=self.interpolation == "linear", left=s > 0.0
        )

    def breakpoints(self, lo: float, hi: float, calendar: Calendar | None = None) -> list[float]:
        return [p for p in self._t if lo < p < hi]
