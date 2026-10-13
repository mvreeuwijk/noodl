"""Source waveforms with the semantics of the Modelica Standard Library sources.

A source's value is `offset + amplitude * unit(t)`, where the waveform supplies the
dimensionless `unit(t)` and the source law (`PrescribedVoltage`, `PrescribedCurrent`)
supplies `amplitude` and `offset`, as in `Modelica.Electrical.Analog.Sources.*Voltage`, which
pass `offset` and `startTime` to the `Modelica.Blocks.Sources` signal of the same name. Every
waveform is a frozen dataclass; `breakpoints(t0, t1)` lists the times in `(t0, t1]` where
the value jumps or its slope does, so a fixed-step integrator can restart there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class _Waveform:
    """Shared interface: `unit(t)`, `time_dependent`, `breakpoints(t0, t1)`."""

    time_dependent: bool = True

    def unit(self, t: float) -> float:  # pragma: no cover - every waveform overrides it
        raise NotImplementedError

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return []


def _inside(times, t0: float, t1: float) -> list[float]:
    return sorted({t for t in times if t0 < t <= t1})


@dataclass(frozen=True)
class Constant(_Waveform):
    """A constant: unit value 1 at every time (a DC source: value `offset + amplitude`)."""

    time_dependent = False

    def unit(self, t: float) -> float:
        return 1.0


@dataclass(frozen=True)
class Step(_Waveform):
    """A step: 0 before `start_time` (s), 1 from `start_time` on (MSL
    `Blocks.Sources.Step`: `if time < startTime then 0 else height`)."""

    start_time: float = 0.0

    def unit(self, t: float) -> float:
        return 0.0 if t < self.start_time else 1.0

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return _inside([self.start_time], t0, t1)


@dataclass(frozen=True)
class Ramp(_Waveform):
    """A ramp: 0 before `start_time` (s), rising linearly to 1 over
    `duration` (s), then 1 (MSL `Blocks.Sources.Ramp`; `duration = 0` is a step)."""

    duration: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.duration >= 0.0:
            raise ValueError(f"Ramp: duration must be >= 0 s, got {self.duration!r}")

    def unit(self, t: float) -> float:
        if t < self.start_time:
            return 0.0
        if t < self.start_time + self.duration:
            return (t - self.start_time) / self.duration
        return 1.0

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return _inside([self.start_time, self.start_time + self.duration], t0, t1)


@dataclass(frozen=True)
class Sine(_Waveform):
    """A sine wave: 0 before `start_time` (s), then
    `sin(2 pi frequency (t - start_time) + phase)` (`frequency` in Hz, `phase` in rad; MSL
    `Blocks.Sources.Sine` with `continuous = false`)."""

    frequency: float
    phase: float = 0.0
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.frequency >= 0.0:
            raise ValueError(f"Sine: frequency must be >= 0 Hz, got {self.frequency!r}")

    def unit(self, t: float) -> float:
        if t < self.start_time:
            return 0.0
        return math.sin(2.0 * math.pi * self.frequency * (t - self.start_time) + self.phase)

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        return _inside([self.start_time], t0, t1)


@dataclass(frozen=True)
class Pulse(_Waveform):
    """A pulse train: 1 for the first `width` percent of
    every `period` (s) from `start_time`, else 0; `n_periods < 0` repeats for ever, otherwise
    the pulse train stops after `n_periods` periods (MSL `Blocks.Sources.Pulse`).

    Period `k` starts at `start_time + k * period`, computed as MSL computes its time events,
    and the pulse is on while `t < start_time + k * period + width / 100 * period`.

    `k` is the largest period index with `start_time + k * period <= t`, the count MSL
    reaches through its time events. MSL initialises its count at the simulation start `t0`
    as `integer((t0 - startTime) / period)` instead; where that quotient and the event grid
    disagree in floating point (`t0 = 1.7`, `period = 0.1`: the quotient floors to 17 but
    `17 * 0.1 > 1.7`), an MSL run started at `t0` counts one period ahead until its next
    event, and its value at `t0` can differ from `unit(t0)`.
    """

    width: float
    period: float
    start_time: float = 0.0
    n_periods: int = -1

    def __post_init__(self) -> None:
        if not 0.0 < self.width <= 100.0:
            raise ValueError(f"Pulse: width must be in (0, 100] percent, got {self.width!r}")
        if not self.period > 0.0:
            raise ValueError(f"Pulse: period must be > 0 s, got {self.period!r}")

    def _on_time(self) -> float:
        return self.period * self.width / 100.0

    def _start(self, k: int) -> float:
        return self.start_time + k * self.period

    def _count(self, t: float) -> int:
        """The largest `k` with `start_time + k * period <= t` (t >= start_time)."""
        k = math.floor((t - self.start_time) / self.period)
        while self._start(k) > t:
            k -= 1
        while self._start(k + 1) <= t:
            k += 1
        return k

    def unit(self, t: float) -> float:
        if t < self.start_time or self.n_periods == 0:
            return 0.0
        k = self._count(t)
        if 0 < self.n_periods <= k:
            return 0.0
        return 1.0 if t < self._start(k) + self._on_time() else 0.0

    def breakpoints(self, t0: float, t1: float) -> list[float]:
        if self.n_periods == 0 or t1 < self.start_time:
            return []
        last = self.n_periods if self.n_periods > 0 else math.inf
        k = max(0, math.floor((t0 - self.start_time) / self.period) - 1)
        full = self.width >= 100.0
        times = []
        while k < last and self._start(k) <= t1:
            if not full or k == 0:
                times.append(self._start(k))
            if not full:
                times.append(self._start(k) + self._on_time())
            k += 1
        if full and last < math.inf:
            times.append(self._start(last))
        return _inside(times, t0, t1)


@dataclass(frozen=True)
class Signal(_Waveform):
    """An external signal: the unit value is read from the driver `drivers[key][..., index]`
    at evaluation time (batched like any driver), so a source can follow an external input
    (MSL `SignalVoltage`, `SignalCurrent`). It needs no clock and has no breakpoints."""

    key: str
    index: int = 0
    time_dependent = False

    def unit(self, t: float) -> float:
        raise TypeError(
            f"Signal({self.key!r}, index={self.index}) has no unit value of its own: it is "
            f"read from drivers[{self.key!r}][..., {self.index}] by the source law"
        )
