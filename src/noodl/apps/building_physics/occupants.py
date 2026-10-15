"""Occupants: where people are over the day.

An `OccupancyDay` gives an occupant's zone at times of day (seconds after midnight, from
00:00:00 to 24:00:00); `None` means out of the building. An `Occupancy` holds one occupancy
day per CONTAM day type (1-12) and follows a `Calendar`, like a week schedule. The occupant
is in a listed zone FROM the listed time, also in the middle of a step: a move at 01:02:30
splits the step from 01:00 to 01:05 there. This is how ContamX accounts for it, both in the
exposure it reports and in the contaminants an occupant generates (measured on ContamX
3.4.1.7).

`Presence(occupancy, net)` is the occupant's location as a signal over the network's nodes:
1 at the occupied node at an instant (the left limit at a move, as every schedule), and
over a step (t0, t1] the fraction of the step spent at each node (`step_mean`), which
`apply_inputs` writes when it is given the step's start.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from noodl.apps.building_physics.schedules import (
    N_DAY_TYPES,
    Calendar,
    _check_day_times,
    _check_time_of_day,
    _DayTypeCycle,
    _piecewise,
)

Tensor = torch.Tensor
F64 = torch.float64


class OccupancyDay:
    """An occupant's zone over one day: `zones[i]` from `times[i]` (s after midnight) until
    the next time. `times` run from 0 (00:00:00) to 86400 (24:00:00), strictly increasing.
    A zone is a node name, or None for out of the building. ContamX needs the 24:00:00 point
    too: without it the run stops."""

    def __init__(self, times, zones: Sequence[Hashable | None], *, name: str = "") -> None:
        t = [float(v) for v in times]
        z = list(zones)
        label = f"OccupancyDay {name!r}" if name else "OccupancyDay"
        _check_day_times(label, t, len(z), "zones")
        self.times, self.zones, self.name = t, z, name

    def __repr__(self) -> str:
        return f"OccupancyDay({self.name!r}, {len(self.times)} points)"

    def zone_at(self, seconds: float, *, left: bool) -> Hashable | None:
        """The zone at `seconds` after midnight (0..86400); `left` takes the one before a move
        listed at exactly that time."""
        s = _check_time_of_day(f"OccupancyDay {self.name!r}", seconds)
        return _piecewise(self.times, self.zones, s, linear=False, left=left)

    def knots(self) -> list[float]:
        """The interior times (s after midnight) where the occupant may move."""
        return self.times[1:-1]


class Occupancy(_DayTypeCycle):
    """An occupant's movements: one `OccupancyDay` per CONTAM day type (1-12).

    `day_types` has 12 entries; entry k is the 1-based position in `day_schedules` used on
    day type k + 1 (CONTAM's twelve occupancy-schedule numbers). The default uses the first
    occupancy day every day.
    """

    _KIND = "Occupancy"
    _DAY = "occupancy day"

    def __init__(self, day_schedules: Sequence[OccupancyDay],
                 day_types: Sequence[int] = (1,) * N_DAY_TYPES, *, name: str = "") -> None:
        super().__init__(day_schedules, day_types, name=name)

    @property
    def zones(self) -> set[Hashable]:
        """Every zone the occupant ever enters (None, out of the building, excluded)."""
        return {z for d in self.day_schedules for z in d.zones if z is not None}

    def zone_at(self, t: float, calendar: Calendar, *, left: bool) -> Hashable | None:
        """The zone at `t` s since the start; `left` (ignored at t = 0) takes the zone before
        a move at exactly `t`, and at midnight the previous day's last zone. A time within
        the calendar's round-off of a move is that move."""
        left = left and float(t) > 0.0
        day, tod = self._day_and_time(t, calendar, left=left)
        return day.zone_at(tod, left=left)

    def segments(self, t0: float, t1: float, calendar: Calendar
                 ) -> list[tuple[float, float, Hashable | None]]:
        """(a, b, zone) for every piece of (t0, t1] (s since the start) spent in one zone, in
        time order; cut at every move and every midnight."""
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError(f"Occupancy {self.name!r}: segments need t1 > t0, got "
                             f"({t0}, {t1})")
        cuts = [t0, *self.breakpoints(t0, t1, calendar), t1]
        return [(a, b, self.zone_at(a, calendar, left=False))
                for a, b in zip(cuts, cuts[1:], strict=False)]


class Presence:
    """An occupant's location as a signal over `net`'s nodes (full node order, float64,
    dimensionless).

    `value(t, calendar)` is 1 at the occupied node (the left limit at a move for t > 0, as
    every schedule) and all zeros out of the building. `step_mean(t0, t1, calendar)` is the
    fraction of the step (t0, t1] spent at each node; `apply_inputs` writes it when it knows
    the step's start, which is how an occupant's generation is split at a mid-step move.

    Occupancy is a step-mean signal, not an instantaneous one: ContamX averages an occupant's
    location over each step, as it averages its schedules, and splits the step at a move
    inside it. The instant at the step's end would put the whole step in the new zone.

    A zone the occupancy names that is not a node of `net` is refused (KeyError).
    """

    def __init__(self, occupancy: Occupancy, net, *, name: str = "") -> None:
        self.occupancy, self.name, self.n = occupancy, name, int(net.n)
        self.index: dict[Hashable, int] = {}
        for zone in sorted(occupancy.zones, key=str):
            try:
                self.index[zone] = int(net.node_index(zone))
            except KeyError as exc:
                raise KeyError(f"Presence {name!r}: zone {zone!r} is not a node of the "
                               f"network") from exc

    def __repr__(self) -> str:
        return f"Presence({self.name!r})"

    def value(self, t: float, calendar: Calendar) -> Tensor:
        """1 at the occupied node at `t` s since the start, 0 elsewhere, `(n,)`."""
        out = torch.zeros(self.n, dtype=F64)
        zone = self.occupancy.zone_at(t, calendar, left=float(t) > 0.0)
        if zone is not None:
            out[self.index[zone]] = 1.0
        return out

    def step_mean(self, t0: float, t1: float, calendar: Calendar) -> Tensor:
        """The fraction of (t0, t1] (s since the start) spent at each node, `(n,)`; sums to
        the fraction spent in the building."""
        out = torch.zeros(self.n, dtype=F64)
        span = float(t1) - float(t0)
        for a, b, zone in self.occupancy.segments(t0, t1, calendar):
            if zone is not None:
                out[self.index[zone]] += (b - a) / span
        return out

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]:
        """The times in (lo, hi) (s since the start) where the occupant may move."""
        return self.occupancy.breakpoints(lo, hi, calendar)
