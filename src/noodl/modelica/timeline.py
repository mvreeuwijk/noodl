"""The time grid a Modelica model is stepped on, and its drivers on it.

The output grid comes from the document's `experiment` (`experiment_times`). The driver grid
(`driver_grid`) adds every event strictly inside the run, so that no step straddles one:
where a signal's output jumps (a `Step`'s start, a `Pulse`'s edges) or kinks (a `Ramp`'s
corners, a linearly interpolated table's knots), `signal_events`, plus whatever further step
boundaries a reader supplies (the building reader's weather file). `grid_events` places the
jumps and kinks on that grid, where a run restarts its step sequence. The drivers of a run
hold their time series as `"series:<key>"` entries on the grid; `step_drivers` slices them
at one grid time, `grid_index` finds requested times on the grid, and `extrapolate`
combines two runs at `substeps` `r` and `2r` by Richardson extrapolation.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Mapping

import torch

from noodl.modelica import signals
from noodl.modelica.errors import ModelicaImportError
from noodl.modelica.schema import Signal

Tensor = torch.Tensor
F64 = torch.float64

__all__ = ["SERIES", "driver_grid", "experiment_times", "extrapolate", "grid_events",
           "grid_index", "signal_events", "step_drivers"]

#: The prefix of a driver that holds a series over the driver grid.
SERIES = "series:"
# Driver-grid metadata, not per-time drivers: the grid, its signal events (`grid_events`)
# and its base grid (`driver_grid` at one substep).
_GRID_KEYS = ("series:time", "series:jumps", "series:kinks", "series:base")

#: `events(lo, hi) -> (jumps, kinks, breaks)`: the event times strictly inside `(lo, hi)`.
#: Breaks are step boundaries that are not events.
Events = Callable[[float, float], tuple[set[float], set[float], set[float]]]


def experiment_times(experiment: Mapping) -> Tensor:
    """The output grid `StartTime + k Interval`, `k = 0..N` (Modelica's default `Interval`
    is `(StopTime - StartTime)/500`)."""
    start = float(experiment.get("StartTime", 0.0))
    stop = float(experiment.get("StopTime", 1.0))
    interval = float(experiment.get("Interval", (stop - start) / 500.0))
    if not interval > 0.0 or stop < start:
        raise ModelicaImportError(
            f"modelica: experiment needs StopTime >= StartTime and Interval > 0, got "
            f"{experiment}"
        )
    n = int(round((stop - start) / interval))
    return start + interval * torch.arange(n + 1, dtype=F64)


def signal_events(sigs: Iterable[Signal], lo: float, hi: float
                  ) -> tuple[set[float], set[float]]:
    """`(jumps, kinks)` of the source signals in `(lo, hi)` (`signals.events`); a signal
    that cannot be evaluated adds none here (it is refused by name when it is evaluated)."""
    jumps: set[float] = set()
    kinks: set[float] = set()
    for sig in sigs:
        if signals.is_math(sig):
            continue
        try:
            j, k = signals.events(sig, lo, hi)
        except ModelicaImportError:
            continue
        jumps |= set(j)
        kinks |= set(k)
    return jumps, kinks - jumps


def driver_grid(experiment: Mapping, events: Events, substeps: int = 1) -> Tensor:
    """The time grid the drivers are evaluated on and a run steps over: the BASE grid, the
    output grid (`experiment_times`) plus every event and break of `events` strictly inside
    the run, with each of its intervals split into `substeps` equal steps. A step then never
    straddles an event. Splitting the base intervals, not the output intervals, makes the
    grid at `2 r` substeps the grid at `r` with every step halved, which `extrapolate`
    relies on."""
    if int(substeps) != substeps or substeps < 1:
        raise ValueError(f"modelica: substeps must be a positive integer, got {substeps!r}")
    out = experiment_times(experiment)
    lo, hi = float(out[0]), float(out[-1])
    jumps, kinks, breaks = events(lo, hi)
    # An event within round-off of an output time is that time.
    tol = 1e-9 * max(1.0, abs(hi))
    outs = out.tolist()
    kept: list[float] = []
    for t in sorted(set(outs) | jumps | kinks | breaks):
        k = bisect.bisect_left(outs, t)
        near = [outs[i] for i in (k - 1, k) if 0 <= i < len(outs)]
        if near and min(abs(o - t) for o in near) <= tol:
            t = min(near, key=lambda o: abs(o - t))
        if kept and t - kept[-1] <= tol:
            continue
        kept.append(t)
    base = torch.tensor(sorted(set(kept)), dtype=F64)
    if substeps == 1:
        return base
    frac = torch.arange(substeps, dtype=F64) / substeps
    a, b = base[:-1], base[1:]
    inner = (a.unsqueeze(1) + (b - a).unsqueeze(1) * frac).reshape(-1)
    return torch.cat([inner, base[-1:]])


def grid_events(grid: Tensor, events: Events) -> tuple[Tensor, Tensor]:
    """`(jumps, kinks)`: the times of `grid` (a `driver_grid`) at which `events` jumps, and
    those at which it only kinks, each moved to the grid time it was merged into. Breaks
    are neither."""
    grid = torch.as_tensor(grid, dtype=F64)
    jumps, kinks, _ = events(float(grid[0]), float(grid[-1]))

    def on_grid(ts: set[float]) -> Tensor:
        if not ts:
            return torch.zeros(0, dtype=F64)
        t = torch.tensor(sorted(ts), dtype=F64)
        k = (t.unsqueeze(1) - grid.unsqueeze(0)).abs().argmin(dim=1)
        return torch.unique(grid[k])

    j, k = on_grid(jumps), on_grid(kinks)
    return j, k[~torch.isin(k, j)]


def step_drivers(drivers: Mapping[str, Tensor], grid: Tensor, t: float) -> dict[str, Tensor]:
    """`drivers` at grid time `t`: every `"series:<key>"` entry sliced into `<key>`, and the
    series entries themselves dropped. Raises `ValueError` if `t` is not on the grid."""
    out = {k: v for k, v in drivers.items() if not k.startswith(SERIES)}
    series = {k[len(SERIES):]: v for k, v in drivers.items()
              if k.startswith(SERIES) and k not in _GRID_KEYS}
    if not series:
        return out
    grid = torch.as_tensor(grid, dtype=F64)
    match = torch.isclose(grid, torch.tensor(float(t), dtype=F64), rtol=0.0,
                          atol=1e-9 * max(1.0, abs(float(t))))
    if not bool(match.any()):
        raise ValueError(
            f"simulate: time {t!r} is not on the experiment grid the driver series were "
            f"evaluated on ({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    k = int(match.nonzero()[0])
    for key, value in series.items():
        out[key] = value[k]
    return out


def grid_index(grid: Tensor, times: Tensor, *, caller: str) -> list[int]:
    """The position of every entry of `times` on the driver grid; raises `ValueError`, its
    message starting with `caller`, for a time off the grid or a non-increasing one."""
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    if bool(off.any()):
        k = int(torch.nonzero(off)[0])
        raise ValueError(
            f"{caller}: times[{k}] = {float(times[k])!r} is not a grid time of the drivers "
            f"({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    out = idx.tolist()
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError(f"{caller}: times must be increasing")
    return out


def extrapolate(coarse: Mapping[str, Tensor], fine: Mapping[str, Tensor],
                order: int = 2) -> dict[str, Tensor]:
    """Richardson extrapolation of two run histories on the same `"time"`s, `fine` with
    half the step of `coarse` (twice the `substeps`), of a method of order `order`:
    `fine + (fine - coarse)/(2**order - 1)` for every entry but `"time"`. For a symmetric
    second-order method (the building reader's midpoint steps; the circuits' trapezoidal
    ones, backward Euler after each restart) `order=2` removes the leading step-size error.
    Raises `ValueError` when the two histories are not on the same times."""
    if not torch.equal(coarse["time"], fine["time"]):
        raise ValueError("extrapolate: the two histories are not on the same times")
    f = 1.0 / (2 ** order - 1)
    return {k: v if k == "time" else v + (v - coarse[k]) * f for k, v in fine.items()}
