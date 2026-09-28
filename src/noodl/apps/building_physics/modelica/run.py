"""Run a model `read_modelica` built over its driver grid.

`simulate(model, state, drivers, times)` steps the model from `times[0]` over every later
driver-grid time (`assemble.driver_grid`: the output grid, optionally split into substeps,
plus the signals' event times) and returns, at `times`, the time histories MBL's reference
CSV holds: every air-layer edge flow, every node's absolute pressure, temperature, water mass
fraction and trace-substance mass fractions. Time-varying drivers are taken at the step's END
time, except sources, which are the MEAN of their value over the step (`assemble.py`,
"Sources"). Two schemes: `"implicit"` (`Model.step`, flows at the step's end, first order)
and `"midpoint"` (`_Midpoint`, symmetric, second order; `extrapolate` combines two of its
runs into a fourth-order one). A model without zones (no transport layer) is algebraic in its
boundary values and is solved with `model.steady` at every time asked for instead. Row 0 is
the initial state with its quasi-steady flows.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping

import torch

from noodl.apps.building_physics.modelica.assemble import _ZoneStateClosure
from noodl.apps.building_physics.modelica.storage import StorageClosure, mass_change
from noodl.elements import Element
from noodl.model import Drivers, Model, State

Tensor = torch.Tensor
F64 = torch.float64
_SERIES = "series:"
# Driver-grid metadata, not per-time drivers: the grid, its signal events
# (`assemble.grid_events`) and its base grid (`assemble.driver_grid` at one substep).
_GRID_KEYS = ("series:time", "series:jumps", "series:kinks", "series:base")
# Newton tolerances for the airflow solves (kg/s). The layer default, sqrt(eps) = 1.5e-8 in
# absolute and relative terms, is 1e-4 of a small crack flow; the reference simulation's
# declared tolerance is 1e-6 relative, so the airflow is solved to round-off instead.
# Newton converges quadratically, so this costs about one more iteration.
AIR_ATOL = 1e-13
AIR_RTOL = 1e-12
# With volume mass storage (`storage` module) the zones leave the quasi-steady pressures:
# ReverseBuoyancy's start 1325 Pa above its boundary, and there the round-off of a door's
# `dp`, `eps |phi|` = 1.5e-13 Pa at 650 Pa, times an open door's slope (~3 kg/s/Pa) is a
# residual floor of 3e-13 to 8e-13 kg/s (measured), above AIR_ATOL. 1e-11 kg/s is still
# 1e-7 of ClosedDoors' smallest reported crack flows (~1e-4 kg/s).
STORAGE_AIR_ATOL = 1e-11
# With storage, a short step's storage terms (`V k dphi/h`) can hold the airflow residual a
# little above that: OneEffectiveAirLeakageArea's first 1.4e-4 s step stalled at 1.2e-11
# kg/s. `_Midpoint` accepts a Newton solve that stalls within STORAGE_STALL times its
# tolerance (1e-9 kg/s, 1e-7 of the smallest reported crack flows).
STORAGE_STALL = 100.0


def step_drivers(drivers: Mapping[str, Tensor], grid: Tensor, t: float) -> Drivers:
    """`drivers` at grid time `t`: every `"series:<key>"` entry sliced into `<key>`, and the
    series entries themselves dropped. Raises `ValueError` if `t` is not on the grid."""
    out: Drivers = {k: v for k, v in drivers.items() if not k.startswith(_SERIES)}
    series = {k[len(_SERIES):]: v for k, v in drivers.items()
              if k.startswith(_SERIES) and k not in _GRID_KEYS}
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


def _grid_index(grid: Tensor, times: Tensor) -> list[int]:
    """The position of every entry of `times` on the driver grid; raises `ValueError` for a
    time off the grid or a non-increasing one."""
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    if bool(off.any()):
        k = int(torch.nonzero(off)[0])
        raise ValueError(
            f"simulate: times[{k}] = {float(times[k])!r} is not a grid time of the drivers "
            f"({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    out = idx.tolist()
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("simulate: times must be increasing")
    return out


def _closure(model: Model) -> _ZoneStateClosure:
    for c in model.closures:
        if isinstance(c, _ZoneStateClosure):
            return c
    raise TypeError("simulate: the model was not built by read_modelica (no MBL closure)")


def _storage(model: Model) -> StorageClosure | None:
    return next((c for c in model.closures if isinstance(c, StorageClosure)), None)


def _initial_air(model: Model, store: StorageClosure, state: State, drivers: Drivers,
                 **solve_kwargs) -> State:
    """The t = StartTime airflow of a model with volume mass storage (`storage` module
    docstring, "Initial state"): every storing zone at its start pressure (the state's own
    `"air.phi"`) and the flows the elements give there, or, when some zone's initial
    equation is `der(p) = 0`, the air layer solved with the others held at their start
    pressures. The carried `"air.storage"` is re-evaluated at the result."""
    drv = dict(drivers)
    drv.update(_closure(model)(state, drv))
    new = dict(state)
    air = model.potential["air"]
    if store.init_layer is None:
        new["air.q"] = air.flows(state["air.phi"], drv)
    else:
        init = store.init_layer
        phi_b = state["air.phi"][..., init.bound]
        new["air.phi"], new["air.q"] = init.solve(phi_b, drv, drv.get("air.sources"),
                                                  phi0=None, **solve_kwargs)
    drv.update(_closure(model)(new, drv))
    new["air.storage"] = store(new, drv)["air.storage"]
    return new


# Volume mass storage (`storage` module docstring, "Initial state"): MBL starts every
# `FixedInitial` volume at `p_start`, and the imbalance relaxes in the volumes' own time
# `tau = V k / (dq/dp)`, a fraction of a second to a few seconds, much less than a grid
# interval. A step with `h >> tau` does not resolve that: backward Euler reports the mean
# release rate over the step (CO2TransportStep: 1.5e-4 kg/s at its first 172.8 s row
# against OpenModelica's ~0). So the run starts on a graded sub-grid,
# `h = max(h0, (ratio - 1) t)` from `h0 = GRADING_H0` of the output interval, cut at every
# base-grid time (`assemble.driver_grid` at one substep), for `GRADING_WINDOW` output
# intervals, and then doubles up to the base step (`_graded`); so does every stretch after a
# signal event. At `r` substeps every one of these steps is split into `r` equal ones, so the
# runs `extrapolate` combines (1 and 2 substeps) take the same steps, halved (`simulate`).
# `h` passes `tau` only at `t = 50 tau`. A start in balance (`_balanced`: ClosedDoors,
# OneOpenDoor, OneRoom, ZonalFlow, OneEffectiveAirLeakageArea) has no release to resolve and
# only doubles up from `h0`. The window resolves
# ReverseBuoyancy's release of its 1325 Pa start imbalance over its first ~30 s: at a
# ratio of 1.1 over one interval it was off by 1.4 Pa and 2.9e-5 relative in T, at 1.02 over
# five by 8e-10 in p and 3.7e-8 in T (measured, extrapolated; 1.05: 3.8e-7 and 1.5e-7).
# OpenDoorBuoyancyPressureDynamic (a 5 Pa imbalance across a 1.9 m2 open door, `tau` of a
# few ms) needs a finer first step: at `h0 = 1e-4` its first sub-step (2.9 ms) left 1.5e-5 K
# in both rooms' T and 1.1e-5 in the door flows at 28.8 s (not removed by `extrapolate`, and
# only 4x smaller at 2 and 4 substeps); at 1e-5 8e-9 kg/s (6e-8 K). 1e-6 (3e-10 kg/s) is
# too fine: ReverseBuoyancy's airflow Newton then stalls at a 1.7e-11 kg/s residual, the
# round-off of its storage terms at a 7 us step.
# The window lasts until the graded step `(ratio - 1) t` has grown to the grid step
# (`1/(ratio - 1)` output intervals), so that no doubling ramp follows it: with a window of 5
# the ramp from 2.9 s to the 28.8 s grid step at 144 s left 4.1e-6 in the door flows of
# OpenDoorBuoyancy(Pressure)Dynamic at 172.8 s (the extrapolation's remainder there), with
# the full window 3e-7.
# The sources over a sub-step are the means of their quadratic reconstruction from the
# grid's step means (`_StepDrivers`).
GRADING_H0, GRADING_RATIO = 1e-5, 1.02
GRADING_WINDOW = round(1.0 / (GRADING_RATIO - 1.0))


def _graded(t_start: float, t0: float, t1: float, h0: float, grading: bool,
            h_last: float | None) -> list[float]:
    """Sub-step ends in `(t0, t1]` (offsets from `t0`, the last exactly `t1 - t0`): inside
    the grading window (`grading`) `h = max(h0, (GRADING_RATIO - 1)(t - t_start))`; after
    it, `h` doubles from the last sub-step (`h_last`) until it is the grid step, so that the
    BDF2 step ratio of `_Midpoint` never exceeds 2. The last sub-step takes the remainder, at
    most 2 h (a sliver followed by a full step would make that ratio large)."""
    ends, t = [], t0
    while True:
        if grading:
            h = max(h0, (GRADING_RATIO - 1.0) * (t - t_start))
        elif h_last is not None:
            h = 2.0 * h_last
        else:
            h = t1 - t0
        if t + 2.0 * h >= t1 - 1e-12 * max(1.0, abs(t1)):
            ends.append(t1 - t0)
            return ends
        t += h
        h_last = h
        ends.append(t - t0)


def _cell_mean(v0: Tensor | None, m: Tensor, v2: Tensor | None, a: float, b: float,
               first: bool) -> Tensor:
    """Mean over `(a, b)` (fractions of the step) of `s(x) = m + c1 (x - 1/2) +
    c2 ((x - 1/2)^2 - 1/12)`, whose mean over the step is `m`, fitted to the mean `v2` of the
    next step and either the mean `v0` of the step before or (`first`) the point value `v0`
    at the step's start; linear (`c2 = 0`) without `v2` or without `v0` (then through `m`
    and `v2`, or constant without both)."""
    if v0 is None:
        c1 = torch.zeros_like(m) if v2 is None else v2 - m
        c2 = torch.zeros_like(m)
    elif v2 is None:
        left = v0 if first else 0.5 * (v0 + m)  # s(0)
        c1, c2 = 2.0 * (m - left), torch.zeros_like(m)
    elif first:  # s(0) = v0, mean over (1, 2) = v2
        c2 = 1.5 * (v0 - m) + 0.75 * (v2 - m)
        c1 = v2 - m - c2
    else:        # means over (-1, 0) and (1, 2)
        c1, c2 = 0.5 * (v2 - v0), 0.5 * (v2 - 2.0 * m + v0)
    sq = ((b - 0.5) ** 3 - (a - 0.5) ** 3) / (3.0 * (b - a))
    return m + c1 * (0.5 * (a + b) - 0.5) + c2 * (sq - 1.0 / 12.0)


class _StepDrivers:
    """The drivers of `simulate`'s steps. On a driver-grid step `(grid[j-1], grid[j])` they
    are the grid's own (`step_drivers` at its two ends). On any other interval `(u, v)` (a
    sub-step of the graded start, `_graded`, or the parts of a step split at a switch,
    `_Switches`): point values linearly interpolated in the grid step holding each end, and
    every source (`"<layer>.sources"`, a step mean on the grid) the mean over `(u, v)` of its
    reconstruction, grid step by grid step (`_cell_mean`: through the step's own mean, the
    next step's and the one before, or at the run's first time (`start`) the point value;
    its mean over each grid step is the grid's, and it is exact for a quadratic signal). After
    a signal's kink (`grid_events`) the mean before is not used: the source is the linear one
    through the step's and the next step's means (constant without a next step of the same
    length). After a jump the source is held at its step mean and the point values at the
    step's end, as on the grid. Without the event lists (drivers not built by `read_modelica`)
    a jump is where the grid step changes (a source pulse is its own step)."""

    def __init__(self, drivers: Mapping[str, Tensor], grid: Tensor, start: int) -> None:
        self.drivers, self.grid, self.start = drivers, grid, start
        self.g = grid.tolist()
        self.tol = 1e-9 * max(1.0, abs(self.g[0]), abs(self.g[-1]))
        jumps, kinks = _event_indices(drivers, grid)
        if jumps is None:
            h = [b - a for a, b in zip(self.g, self.g[1:], strict=False)]
            jumps = {i for i in range(1, len(h))
                     if abs(h[i] - h[i - 1]) > 1e-9 * max(h[i], h[i - 1])}
        self.jumps, self.kinks = jumps, kinks
        self._at: dict[int, Drivers] = {}

    def at(self, k: int) -> Drivers:
        """`step_drivers` at grid position `k`."""
        if k not in self._at:
            self._at[k] = step_drivers(self.drivers, self.grid, self.g[k])
        return self._at[k]

    def index(self, t: float) -> int | None:
        """The grid position of time `t`, or `None` if `t` is not a grid time."""
        k = bisect.bisect_left(self.g, t - self.tol)
        return k if k < len(self.g) and abs(self.g[k] - t) <= self.tol else None

    def _cell(self, j: int):
        """Grid step `j` = `(grid[j-1], grid[j])`: its two ends' drivers, the next step's
        (or `None`), and whether a jump or a kink starts it."""
        g = self.g
        h = g[j] - g[j - 1]
        d2 = None
        if (j + 1 < len(g) and abs((g[j + 1] - g[j]) - h) <= 1e-9 * h
                and j not in self.jumps and j not in self.kinks):
            d2 = self.at(j + 1)
        return self.at(j - 1), self.at(j), d2, (j - 1) in self.jumps, (j - 1) in self.kinks

    def _point(self, t: float, j: int, out: Drivers) -> None:
        """The point values of `out` (a copy of grid step `j`'s end drivers) at `t`."""
        d0, d1, _, jump, _ = self._cell(j)
        if jump:
            return
        a = (t - self.g[j - 1]) / (self.g[j] - self.g[j - 1])
        for key, v1 in d1.items():
            v0 = d0.get(key)
            if (key.endswith(".sources") or not isinstance(v1, Tensor)
                    or not isinstance(v0, Tensor) or v0.shape != v1.shape
                    or torch.equal(v0, v1)):
                continue
            out[key] = v0 + (v1 - v0) * a

    def span(self, u: float, v: float) -> tuple[Drivers, Drivers]:
        """The drivers at the two ends of the step `(u, v)` (class docstring)."""
        ku, kv = self.index(u), self.index(v)
        if ku is not None and kv == ku + 1:
            return self.at(ku), self.at(kv)
        g = self.g
        ju = bisect.bisect_right(g, u + self.tol)       # the grid step holding u's right
        jv = bisect.bisect_left(g, v - self.tol)        # ... and v's left
        du, dv = dict(self.at(ju)), dict(self.at(jv))
        self._point(u, ju, du)
        self._point(v, jv, dv)
        sums: dict[str, Tensor] = {}
        for j in range(ju, jv + 1):
            d0, d1, d2, jump, kink = self._cell(j)
            h = g[j] - g[j - 1]
            a, b = (max(u, g[j - 1]) - g[j - 1]) / h, (min(v, g[j]) - g[j - 1]) / h
            for key, v1 in d1.items():
                if not key.endswith(".sources"):
                    continue
                v0 = d0.get(key)
                if jump or not isinstance(v0, Tensor) or v0.shape != v1.shape:
                    mean = v1
                else:
                    v2 = None if d2 is None else d2.get(key)
                    mean = _cell_mean(None if kink else v0, v1, v2, a, b,
                                      j - 1 == self.start)
                piece = mean * ((b - a) * h)
                sums[key] = piece if key not in sums else sums[key] + piece
        for key, total in sums.items():
            du[key] = dv[key] = total / (v - u)
        return du, dv


def _event_indices(drivers: Drivers, grid: Tensor) -> tuple[set[int] | None, set[int]]:
    """The grid positions of the jumps and of the kinks (`assemble.grid_events`), or
    `(None, set())` when the drivers carry no event lists."""
    if "series:jumps" not in drivers:
        return None, set()

    def where(ts) -> set[int]:
        ts = torch.as_tensor(ts, dtype=F64)
        if ts.numel() == 0:
            return set()
        return set(_grid_index(grid, ts))

    return where(drivers["series:jumps"]), where(drivers.get("series:kinks", ()))


def _resync_storage(model: Model, store: StorageClosure, state: State,
                    drivers: Drivers) -> State:
    """`"air.storage"` re-evaluated at the returned state itself. The model carries the
    last coupling pass's INPUT state (`StorageClosure`), equal to the returned one to the
    coupling tolerance; the difference, `V k dphi` per step, would otherwise accumulate as a
    mass drift (measured 1.2e-11 of ClosedDoors' mass over 40 steps, growing with the run)."""
    drv = dict(drivers)
    drv.update(_closure(model)(state, drv))
    return {**state, "air.storage": store(state, drv)["air.storage"]}


def _solve_air(model: Model, state: State, drivers: Drivers, **solve_kwargs) -> State:
    """The potential layers alone at `state` (closures first): the quasi-steady flows of the
    initial state, without advancing any transport layer.

    Newton starts from the layer's own linear initial guess (`phi0=None`,
    `PotentialFlowLayer.linear_init`), not from the state's `p_start` seed: every zone at
    `p_start` is far from a stack's hydrostatic solution, and from there Newton cycles on
    MBL's discretised doors (`Validation/ThreeRoomsContamDiscretizedDoor.mo`: residual
    0.084 kg/s after 50 iterations), while the linear guess converges in three."""
    drv = dict(drivers)
    for c in model.closures:
        drv.update(c(state, drv))
    new = dict(state)
    for name, layer in model.potential.items():
        phi, q = layer.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"),
                             phi0=None, **solve_kwargs)
        new[f"{name}.phi"], new[f"{name}.q"] = phi, q
    return new


def simulate(model: Model, state: State, drivers: Drivers, times, *,
             scheme: str = "implicit", **step_kwargs) -> dict[str, Tensor]:
    """Time histories at `times` (increasing times of the driver grid `"series:time"`).

    Returns `"time"` `(N,)`, `"air.q"` `(N, b)` in the air layer's edge order (see
    `ModelicaNames.edges`), `"air.phi"` `(N, n)` gauge pressure (relative to
    `ModelicaNames.p_ref`), `"p"` `(N, n)` absolute
    pressure (Pa), `"T"` `(N, n)` (K), `"X_w"` `(N, n)`, and `"C"` `(N, n, K)` for the species
    layer's mass fractions (water last when carried) when the model has one. `step_kwargs`
    reach `Model.step`/`Model.steady` (and so the airflow solves); the airflow Newton
    tolerances default to `atol=AIR_ATOL` (`STORAGE_AIR_ATOL` with volume mass storage),
    `rtol=AIR_RTOL`.

    The model is stepped over EVERY driver-grid time from `times[0]` to `times[-1]` (the
    grid `read_modelica(..., substeps=r)` builds: the base grid of output times and signal
    events, each of its steps split into `r`), and the rows at `times` are returned. Each
    source driver is its mean over the one grid interval ending at its row (`assemble` module
    docstring, "Sources"), so the injected amounts are exact. With volume mass storage the
    run starts on a graded sub-grid (`GRADING_WINDOW`), after every signal event too. The
    steps are made on the base grid and split into `r` alike, so that a run at `2 r` substeps
    takes the steps of the run at `r`, halved.

    `scheme="midpoint"` also puts a step boundary on every switch (`_Switches`): where an
    element's law changes piece (the edges of its regularisation band,
    `Element.switching`) or an edge flow changes sign (the transport layers upwind on it). A
    step across one is split there, the crossing located on the step's own runs
    (`_advance_located`), so that the right-hand side is smooth within every step: a door's
    flows reversing through its bands across a 14.4 s step were off by 1.1e-4 of the flow
    floor at OneOpenDoor's reversals (2131 s), 8.3e-6 with the switches located (measured,
    extrapolated from 1 and 2 substeps).

    `scheme`: `"implicit"` (the default) is `Model.step` with `coupling="iterate"`: the
    transport layers advance with the flows held at their END-of-step values, first order
    in the step. `"midpoint"` (`_Midpoint`) advances them with the MEAN of the start- and
    end-of-step flows, boundary values and state-dependent heat sources: symmetric and
    second order in the step, so two runs at `r` and `2 r` substeps combine by `extrapolate`
    to fourth order. It is forward-only (no gradient) and for unbatched models.
    """
    if scheme not in ("implicit", "midpoint"):
        raise ValueError(f"simulate: unknown scheme {scheme!r}; 'implicit' or 'midpoint'")
    atol = AIR_ATOL if _storage(model) is None else STORAGE_AIR_ATOL
    step_kwargs = {"atol": atol, "rtol": AIR_RTOL, **step_kwargs}
    times = torch.as_tensor(times, dtype=F64)
    grid = torch.as_tensor(drivers.get("series:time", times), dtype=F64)
    idx = _grid_index(grid, times)
    closure = _closure(model)
    store = _storage(model)
    dynamic = bool(model.transport)
    rows: dict[str, list[Tensor]] = {"air.q": [], "air.phi": [], "p": [], "T": [], "X_w": []}
    if closure.sp_interior is not None:
        rows["C"] = []

    def record(state: State, d: Drivers) -> None:
        extra = closure(state, d)
        rows["air.q"].append(state["air.q"])
        rows["air.phi"].append(state["air.phi"])
        rows["p"].append(closure.pressures(state, d))
        rows["T"].append(extra["T"])
        rows["X_w"].append(extra["X_w"] if "X_w" in extra else d["X_w"])
        if "C" in rows:
            rows["C"].append(closure.species(state, d))

    if not dynamic:  # algebraic: the steady solve at each time asked for
        for k in idx:
            d = step_drivers(drivers, grid, float(grid[k]))
            state = model.steady(state, d, **step_kwargs)
            record(state, d)
    else:
        sd = _StepDrivers(drivers, grid, idx[0])
        d = sd.at(idx[0])
        state = (_solve_air(model, state, d, **step_kwargs) if store is None
                 else _initial_air(model, store, state, d, **step_kwargs))
        record(state, d)
        mid = (_Midpoint(model, closure, step_kwargs, store) if scheme == "midpoint"
               else None)
        switches = _Switches(model, closure, SWITCH_FLOW) if mid is not None else None
        g = sd.g
        # The steps: the base grid's (`assemble.driver_grid` at one substep: the output
        # times and the signal events) between the times asked for, each split into the
        # driver grid's substeps, so that the runs `extrapolate` combines (1 and 2 substeps)
        # take the same steps, halved. The graded start and the splits at switches are
        # made on the base step and split alike.
        base = drivers.get("series:base")
        base = g if base is None else torch.as_tensor(base, dtype=F64).tolist()
        lo, hi = g[idx[0]], g[idx[-1]]
        stops = {idx[0], idx[-1], *idx}
        for t in base:
            k = sd.index(t)
            if k is not None and lo <= g[k] <= hi:
                stops.add(k)
        stops = sorted(stops)
        record_at = set(idx)
        spacing = [b - a for a, b in zip(base, base[1:], strict=False)]
        out_dt = max(spacing) if spacing else 0.0
        h0 = GRADING_H0 * out_dt
        window = GRADING_WINDOW * out_dt
        t_start, grading = g[idx[0]], True
        h_last: float | None = None  # the last base (sub-)step taken
        if store is not None and _balanced(model, store, state, d):
            # A start in balance has no release to resolve: steps doubling up from h0.
            grading, h_last = False, 0.5 * h0

        def advance(state: State, pts: list[float]) -> tuple[State, Drivers]:
            dv = sd.at(idx[0])
            for u, v in zip(pts, pts[1:], strict=False):
                du, dv = sd.span(u, v)
                if mid is None:
                    state = model.step(state, dv, v - u, t=u, **step_kwargs)
                    if store is not None:
                        state = _resync_storage(model, store, state, dv)
                else:
                    state = mid.step(state, du, dv, v - u)
            return state, dv

        for k0, k1 in zip(stops, stops[1:], strict=False):
            p0, p1 = g[k0], g[k1]
            H, n = p1 - p0, k1 - k0
            if k0 != idx[0] and (k0 in sd.jumps or k0 in sd.kinks):
                # A signal event: the graded start again, and a fresh storage rate
                # (`_Midpoint.restart`), so that no step's history straddles it.
                t_start, grading = p0, True
                if mid is not None:
                    mid.restart()
            if grading and p0 >= t_start + window - 1e-9 * max(1.0, window):
                grading = False  # then the ramp up to the base step (`_graded`)
            ramp = h_last is not None and h_last < H * (1 - 1e-9)
            # In absolute time, whatever the base step's length: a base step shortened by a
            # signal event (CO2TransportStep's 3.6 s pulse) grades like any other. Graded in
            # proportion to its length (h0 3.6e-5 s, growing 50 times more slowly), the pulse
            # left 6e-14 kg/kg (2.1e-5 of the floor) in the trace substance after it at 1 and
            # at 2 substeps alike, against 1.1e-15 kg/kg graded in absolute time (measured).
            cuts = (_graded(t_start, p0, p1, h0, grading, h_last)
                    if store is not None and (grading or ramp) else [H])
            if len(cuts) == 1:
                pieces = [g[k0:k1 + 1]]
            else:
                ends = [p0, *(p0 + c for c in cuts[:-1]), p1]
                pieces = [_split(a, b, n) for a, b in zip(ends, ends[1:], strict=False)]
            for pts in pieces:
                if switches is None:
                    state, _ = advance(state, pts)
                else:
                    state = _advance_located(state, pts, advance, switches, mid, sd, h0)
                h_last = pts[-1] - pts[0]
            if k1 in record_at:
                record(state, sd.at(k1))
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["time"] = times.clone()
    return out


def _split(a: float, b: float, n: int) -> list[float]:
    """`a`, `n - 1` equally spaced times, `b`."""
    return [a, *(a + (b - a) * i / n for i in range(1, n)), b]


# A start in balance (`_balanced`): no storing zone's net inflow at t = StartTime above this
# fraction of the largest edge flow (or 1e-15 kg/s with no flow).
BALANCE_RTOL = 1e-10


def _balanced(model: Model, store: StorageClosure, state: State, d: Drivers) -> bool:
    """Whether the volumes start in balance: every storing zone's net inflow at the initial
    state (the flows at the start pressures, `_initial_air`, plus the air sources) is zero to
    `BALANCE_RTOL` of the largest edge flow. Then there is no start imbalance to release
    (`GRADING_WINDOW`) and `simulate` starts with steps doubling up from `GRADING_H0`
    instead."""
    q = state["air.q"]
    s_air = d.get("air.sources")
    s_air = torch.zeros(store.V.shape, dtype=F64) if s_air is None else s_air.to(F64)
    net = store.net(q, s_air)[..., store.air_nodes]
    scale = max(float(q.abs().max()) if q.numel() else 0.0, 1e-5)
    return bool((net.abs() <= BALANCE_RTOL * scale).all())


class _Switches:
    """The switching values at a state: every value changes sign where the step's right-hand
    side stops being smooth. These are the air layer's elements' own switches
    (`Element.switching`: a regularisation band's edge, where the law changes piece) and
    every edge flow's sign, in units of `q_scale` (the transport layers upwind on it: the
    advected state switches from one end's to the other's where a flow reverses). `parts`
    are the elements that have switches."""

    def __init__(self, model: Model, closure: _ZoneStateClosure, q_scale: float) -> None:
        ((self.name, self.air),) = model.potential.items()
        self.closure = closure
        self.q_scale = q_scale
        self.parts = []
        for kind in self.air.kinds:
            el, sl = self.air.element_for(kind)
            if type(el).switching is not Element.switching:
                self.parts.append((el, sl))

    def __call__(self, state: State, d: Drivers) -> Tensor:
        drv = dict(d)
        drv.update(self.closure(state, drv))
        dp = self.air.dp(state[f"{self.name}.phi"], drv)
        vals = [el.switching(dp[..., sl], drv) for el, sl in self.parts]
        vals.append(state[f"{self.name}.q"] / self.q_scale)
        return torch.cat([v.reshape(-1) for v in vals if v is not None])


# State-event location (`_advance_located`): a switch is crossed in a step when its value
# changes sign between the step's ends and was not within SWITCH_SKIP of zero at the start
# (a switch just located stays there); it is located to |value| <= SWITCH_TOL (in units of
# the band's width, `Element.switching`) in at most SWITCH_MAX_ITER trial steps.
SWITCH_TOL = 1e-7
SWITCH_SKIP = 1e-5
SWITCH_MAX_ITER = 40
# An edge flow's switch is its sign, in units of SWITCH_FLOW kg/s: located to 1e-10 kg/s,
# and a sign change counts only between flows of 1e-8 kg/s or more (the airflow solve's
# own resolution of a door's flow, its slope times `MIDPOINT_ATOL`'s 1e-9 Pa, is ~3e-9).
SWITCH_FLOW = 1e-3
# A switch within SWITCH_MARGIN of a step's length (at least GRADING_H0 of the output
# interval) from either end does not split the step: a sliver step would sit at the
# storage's round-off, and a band edge that close to the end of a step moves its error by
# the cube of that fraction.
SWITCH_MARGIN = 1e-3
# The storage rate's BDF2 history is kept across a located switch (the mass is smooth there):
# restarting it (backward Euler after each switch) left 1.2e-7 kg/s in ReverseBuoyancy's
# door flow at 612 s against 9.6e-9 (measured, extrapolated from 2 and 4 substeps).


def _crossed(g0: Tensor, g1: Tensor) -> Tensor:
    """The switches crossed between two ends with values `g0` and `g1`, as a mask: a sign
    change with both ends clear of the switch (within SWITCH_SKIP it is at that end)."""
    return (g0 * g1 < 0) & (g0.abs() > SWITCH_SKIP) & (g1.abs() > SWITCH_SKIP)


def _first(g0: Tensor, g1: Tensor, mask: Tensor) -> int:
    """The switch of `mask` crossed first, by the linear interpolation of its values."""
    theta = torch.where(mask, g0 / (g0 - g1), torch.full_like(g0, 2.0))
    return int(torch.argmin(theta))


def _advance_located(state: State, pts: list[float], advance, switches: _Switches,
                     mid: _Midpoint, sd: _StepDrivers, min_step: float) -> State:
    """`advance(state, pts)` (steps of the midpoint scheme over the times `pts`), split at
    every switch crossed (`_Switches`): the crossing time is located by the Illinois
    variant of regula falsi on the switch's value at the end of trial runs from `pts[0]`,
    each in as many equal steps as `pts` has, and the run is taken to it and then on from it,
    again in as many equal steps. Every step then lies on one side of every switch, where
    the element laws are smooth, and the runs `extrapolate` combines (1 and 2 substeps) split
    alike, their steps in ratio 2."""
    n = len(pts) - 1
    a, b = pts[0], pts[-1]
    skip: set[int] = set()  # switches located (or within the margin): not split at again
    while True:
        g_a = switches(state, sd.span(pts[0], pts[1])[0])
        snap = mid.snapshot()
        end, d_end = advance(state, pts)
        g_b = switches(end, d_end)
        cross = _crossed(g_a, g_b)
        if skip:
            cross[list(skip)] = False
        if not bool(cross.any()):
            return end
        k = _first(g_a, g_b, cross)
        margin = max(SWITCH_MARGIN * (b - a), min_step)
        if b - a <= 4 * margin:
            return end  # a step this short is not split (SWITCH_MARGIN)
        lo, hi, g_lo, g_hi, side = a, b, float(g_a[k]), float(g_b[k]), 0
        best = None
        for _ in range(SWITCH_MAX_ITER):
            t = lo + (hi - lo) * g_lo / (g_lo - g_hi)
            t = min(max(t, a + margin), b - margin)
            mid.restore(snap)
            trial, d_t = advance(state, _split(a, t, n))
            g_t = switches(trial, d_t)
            best = (t, trial, mid.snapshot())
            before = _crossed(g_a, g_t) & (g_t.abs() > SWITCH_TOL)
            if skip:
                before[list(skip)] = False
            if bool(before.any()):
                j = _first(g_a, g_t, before)
                if j != k:  # another switch is crossed first: locate that one
                    k, lo, hi, g_lo, g_hi, side = j, a, t, float(g_a[j]), float(g_t[j]), 0
                    continue
            gk = float(g_t[k])
            if abs(gk) <= SWITCH_TOL:
                break
            if (t <= a + margin * (1 + 1e-9) and gk * g_hi > 0) or (
                    t >= b - margin * (1 + 1e-9) and gk * g_lo > 0):
                break  # the switch lies within the margin of an end
            if hi - lo <= 1e-9 * (b - a):
                break  # located to the time resolution
            if gk * g_hi > 0:
                hi, g_hi = t, gk
                if side == -1:
                    g_lo *= 0.5
                side = -1
            else:
                lo, g_lo = t, gk
                if side == 1:
                    g_hi *= 0.5
                side = 1
        t, trial, post = best
        skip.add(k)
        if t <= a + margin * (1 + 1e-9) or t >= b - margin * (1 + 1e-9):
            # The switch lies within the margin of an end: the step is not split there.
            mid.restore(snap)
            continue
        state = trial
        mid.restore(post)
        a, skip = t, {k}
        pts = _split(a, b, n)


def extrapolate(coarse: Mapping[str, Tensor], fine: Mapping[str, Tensor],
                order: int = 2) -> dict[str, Tensor]:
    """Richardson extrapolation of two `simulate` histories on the same `times`, `fine`
    with half the step of `coarse`, of a method of order `order`:
    `fine + (fine - coarse)/(2**order - 1)`. For `scheme="midpoint"` (`order=2`, symmetric,
    so its error expands in even powers of the step) the result is fourth order."""
    if not torch.equal(coarse["time"], fine["time"]):
        raise ValueError("extrapolate: the two histories are not on the same times")
    f = 1.0 / (2 ** order - 1)
    return {k: v if k == "time" else v + (v - coarse[k]) * f for k, v in fine.items()}


# Convergence of `_Midpoint`'s step iteration, per unknown: |change| <= ATOL + RTOL |value|.
MIDPOINT_ATOL = {"thermal": 1e-10, "species": 1e-16, "air": 1e-9}  # K, kg/kg, Pa
MIDPOINT_RTOL = 1e-13
MIDPOINT_MAX_ITER = 100
_ANDERSON_DEPTH = 5


# The order of the storage rate's BDF formula (`_Midpoint`): the derivative at the step's end
# of the polynomial through the mass at the step's end and at the last STORAGE_ORDER step
# ends, built up from backward Euler after a restart. 2: BDF3 cut ClosedDoors' flow error at
# one substep per output interval a hundredfold (4.4e-11 against 6.0e-9 kg/s) but made
# ReverseBuoyancy's through its graded start eight times larger (3.6e-5 against 4.6e-6 of
# the flow floor at 21.6 s, extrapolated from 2 and 4 substeps; measured).
STORAGE_ORDER = 2


def _bdf_weights(nodes: list[float]) -> list[float]:
    """The weights `l_i'(t)` at `t = nodes[0]` of the Lagrange polynomials on `nodes`: the
    derivative at `nodes[0]` of the polynomial through values at `nodes` is their weighted
    sum (variable-step BDF)."""
    t = nodes[0]
    out = [sum(1.0 / (t - x) for x in nodes[1:])]
    for i, xi in enumerate(nodes[1:], start=1):
        w = 1.0 / (xi - t)
        for j, xj in enumerate(nodes[1:], start=1):
            if j != i:
                w *= (t - xj) / (xi - xj)
        out.append(w)
    return out


class _Midpoint:
    """The symmetric step of `simulate(scheme="midpoint")`.

    Over a step of length `h` from state 0 to state 1, each transport layer advances with
    its own exact scheme (`TransportLayer.step`, the flows held constant over the step) at
    the MEAN flows `(q0 + q1)/2`, the mean boundary values `(x_b0 + x_b1)/2` and its step-mean
    source driver plus the mean of the closure's state-dependent sources (the moist-air heat
    carrier, `_ZoneStateClosure.cp_correction`). State 1 is the fixed point of that map together
    with the airflow solve at state 1 (and the interior pressures `p_abs`, which the closure
    feeds back): an Anderson-accelerated fixed-point iteration (depth `_ANDERSON_DEPTH`) on
    the transport states and the air layer's interior potentials, started from the linear
    extrapolation of the previous step and converged to `MIDPOINT_ATOL`/`MIDPOINT_RTOL`.
    The step map is symmetric in states 0 and 1, so the scheme is second order and its
    global error expands in even powers of the step (`extrapolate`).

    With volume mass storage (`store`, `storage` module) the volumes' storage rate at state
    1 is the variable-step BDF2 formula in their mass (the air layer's storage node source
    with `rate` and `offset` from the last two steps; backward Euler on the first): second
    order and L-stable. The trapezoidal rule `m1 - m0 = h (w0 + w1)/2`, the symmetric choice,
    keeps the volumes' fast pressure relaxation (`tau` well under a second against steps of
    seconds) as an undamped mode that flips sign every step: ThreeRoomsContam's door flow
    carried +-2.1e-5 kg/s of it at every row (measured), where BDF2 leaves 1.7e-7, the same
    as the quasi-steady route. The transport layers take the mean of the storage terms
    (`StorageClosure.terms`: capacity and sources) at the two states, with the step's mass
    change `(m1 - m0)/h` as the storage rate."""

    def __init__(self, model: Model, closure: _ZoneStateClosure, solve_kwargs: dict,
                 store: StorageClosure | None = None) -> None:
        if len(model.potential) != 1:
            raise ValueError("simulate(scheme='midpoint'): one potential layer expected")
        ((self.air_name, self.air),) = model.potential.items()
        self.layers = dict(model.transport)
        self.closure = closure
        self.store = store
        # (h, "air.storage" at the start, at the end) of the last step, for BDF2.
        # The storage history for the BDF rate: the carried "air.storage" at the last few
        # step ends (the last is the next step's start) and the steps between them.
        self.storage_hist: tuple[tuple[Tensor, ...], tuple[float, ...]] = ((), ())
        self.kw = dict(solve_kwargs, differentiable=False)
        self.last: tuple[float, Tensor, Tensor] | None = None  # (h, z at start, z at end)

    def restart(self) -> None:
        """Forget the storage rate's BDF2 history (at a signal event, `simulate`): the next
        step takes backward Euler, and BDF2 then builds up again from the event on."""
        self.storage_hist = ((), ())

    def snapshot(self) -> tuple:
        """The step history (`restore` returns to it: a trial step leaves no trace)."""
        return self.last, self.storage_hist

    def restore(self, snap: tuple) -> None:
        self.last, self.storage_hist = snap

    # z = [each transport layer's x, flattened; the air layer's interior potentials]
    def _pack(self, state: State) -> Tensor:
        parts = [state[f"{n}.x"].reshape(-1) for n in self.layers]
        parts.append(state[f"{self.air_name}.phi"][..., self.air.interior].reshape(-1))
        return torch.cat(parts)

    def _unpack(self, z: Tensor, like: State) -> State:
        out, i = dict(like), 0
        for n in self.layers:
            x = like[f"{n}.x"]
            out[f"{n}.x"] = z[i:i + x.numel()].reshape(x.shape)
            i += x.numel()
        phi = like[f"{self.air_name}.phi"].clone()
        phi[..., self.air.interior] = z[i:]
        out[f"{self.air_name}.phi"] = phi
        return out

    def _tolerance(self, z: Tensor, like: State) -> Tensor:
        parts = [torch.full((like[f"{n}.x"].numel(),), MIDPOINT_ATOL.get(n, 1e-12), dtype=F64)
                 for n in self.layers]
        parts.append(torch.full((self.air.interior.numel(),), MIDPOINT_ATOL["air"],
                                dtype=F64))
        return torch.cat(parts) + MIDPOINT_RTOL * z.abs()

    def _extra_sources(self, state: State, d: Drivers) -> dict[str, Tensor]:
        """The closure's state-dependent sources: what it adds to `"<layer>.sources"`."""
        written = self.closure(state, d)
        return {n: written[f"{n}.sources"] - d.get(f"{n}.sources", 0.0)
                for n in self.layers if f"{n}.sources" in written}

    def _zero_sources(self, n: str, x: Tensor) -> Tensor:
        layer = self.layers[n]
        n_nodes = len(self.closure.T0)
        shape = (n_nodes,) if x.ndim == 1 else (n_nodes, layer.n_species)
        return torch.zeros(shape, dtype=F64)

    def _storage_drivers(self, s0: State, d0: Drivers, d1: Drivers, h: float):
        """The trapezoidal storage's air drivers for the step, and state 0's terms."""
        store = self.store
        drv0 = dict(d0)
        drv0.update(self.closure(s0, drv0))
        q0 = s0[f"{self.air_name}.q"]
        s_air = d1.get(f"{self.air_name}.sources")
        s_air = torch.zeros(store.V.shape, dtype=F64) if s_air is None else s_air.to(F64)
        net0 = store.net(q0, s_air)
        n_s = store.air_nodes.numel()
        # The storage rate at state 1 by the variable-step BDF2 formula
        # `w1 = [(1 + 2 om)/(1 + om) (m1 - m0) - om^2/(1 + om) (m0 - m_-1)] / h`,
        # `om = h/h_prev` (backward Euler on the first step): second order and L-stable,
        # where the trapezoidal rule keeps the volumes' fast pressure relaxation as a mode
        # that flips sign every step (class docstring).
        prev = s0["air.storage"]
        states, steps = self.storage_hist
        if not states or not torch.equal(states[-1], prev):
            states, steps = (prev,), ()
        # Node times relative to the step's start: t1 = h, t0 = 0, then the earlier ends.
        nodes = [h, 0.0]
        for dt in reversed(steps):
            nodes.append(nodes[-1] - dt)
        weights = _bdf_weights(nodes)
        rate = weights[0]
        offset = torch.zeros(store.V.shape, dtype=F64)
        for j, before in enumerate(reversed(states[:-1]), start=2):
            dm = mass_change(store.V, before[..., 1], before[..., 0], prev[..., 1],
                             prev[..., 0], store.p_ref)  # m_j - m0
            offset = offset - weights[j] * dm
        air_drv = {
            "air.storage_prev": prev[..., store.air_nodes, :],
            "air.storage_rate": torch.full((n_s,), rate, dtype=F64),
            "air.storage_offset": offset[..., store.air_nodes],
            "air.storage_gain": torch.zeros(n_s, dtype=F64),
            "air.storage_w_fed": torch.zeros(n_s, dtype=F64),
        }
        return drv0, q0, net0, s_air, air_drv

    def _storage_terms(self, s: State, drv: Drivers, q: Tensor, net: Tensor, s_air: Tensor,
                       w_mass: Tensor, net_mean: Tensor) -> dict[str, Tensor]:
        """`StorageClosure.terms` at one end of the step, with the step's mass-consistent
        storage rate `w_mass = (m1 - m0)/h` and, at the air-layer storage zones, the part of
        it the mean flows `net_mean` do not carry (`dnet`): with the BDF2 storage rate the
        mean of the two ends' flows is not the step's mass change."""
        store = self.store
        w = torch.where(store.storage, w_mass, torch.zeros_like(w_mass))
        dnet = torch.where(store._air_mask, w_mass - s_air - net_mean,
                           torch.zeros_like(w_mass))
        return store.terms(s, {**drv, "_q": q}, w, net, s_air, dnet=dnet)

    def _solve_air(self, drv: Drivers, sources, phi0: Tensor, h: float):
        """The air layer at state 1 (`air.solve`); with storage a solve that stalls within
        STORAGE_STALL times its tolerance is accepted (the round-off of the storage terms)."""
        name = self.air_name
        if self.store is None:
            return self.air.solve(drv[f"{name}.phi_boundary"], drv, sources, phi0=phi0,
                                  **self.kw)
        diag: dict = {}
        phi, q = self.air.solve(drv[f"{name}.phi_boundary"], drv, sources, phi0=phi0,
                                diagnostics=diag, **{**self.kw, "on_failure": "return"})
        if not bool(torch.as_tensor(diag["converged"]).all()):
            res = float(torch.as_tensor(diag["residual_norm"]).max())
            if not res <= STORAGE_STALL * self.kw.get("atol", STORAGE_AIR_ATOL):
                raise RuntimeError(
                    f"simulate(scheme='midpoint'): the airflow solve of a {h!r} s step did "
                    f"not converge (residual {res:.3g} kg/s)")
        return phi, q

    def step(self, s0: State, d0: Drivers, d1: Drivers, h: float) -> State:
        air, name = self.air, self.air_name
        q0 = s0[f"{name}.q"]
        extra0 = self._extra_sources(s0, d0)
        z0 = self._pack(s0)
        store = self.store
        if store is not None:
            drv0, _, net0, s_air, air_drv = self._storage_drivers(s0, d0, d1, h)
        # The air balance at state 1: with storage, the BDF2 storage rate there is the
        # derivative of the mass at the step's END, so it balances the sources' point values
        # (`"air.sources_point"`), not their step mean. The step mean would lag the stored
        # mass by half a step of a varying source: OneEffectiveAirLeakageArea's ramped
        # injection, 0.07 kg of 36 kg at 1 substep, first order in the step (measured).
        s_end = d1.get(f"{name}.sources")
        if store is not None:
            s_end = d1.get(f"{name}.sources_point", s_end)

        def G(z: Tensor) -> tuple[Tensor, State]:
            s = self._unpack(z, s0)
            drv = dict(d1)
            drv.update(self.closure(s, drv))
            if store is not None:
                drv.update(air_drv)
            phi, q = self._solve_air(drv, s_end, s[f"{name}.phi"][..., air.interior], h)
            s[f"{name}.phi"], s[f"{name}.q"] = phi, q
            extra1 = self._extra_sources(s, d1)
            qm = 0.5 * (q0 + q)
            cap: dict[str, Tensor] = {}
            if store is not None:
                drv1 = dict(d1)
                drv1.update(self.closure(s, drv1))
                now1 = store(s, drv1)["air.storage"]
                w_mass = store.rate_from_mass(now1, s0["air.storage"], h)
                net_mean = store.net(qm, s_air)
                t0 = self._storage_terms(s0, drv0, q0, net_mean, s_air, w_mass, net_mean)
                t1 = self._storage_terms(s, drv1, q, net_mean, s_air, w_mass, net_mean)
                for n in self.layers:
                    if f"{n}.capacity" in t0:
                        cap[n] = 0.5 * (t0[f"{n}.capacity"] + t1[f"{n}.capacity"])
                        extra = 0.5 * (t0[f"{n}.extra"] + t1[f"{n}.extra"])
                        extra1 = {**extra1, n: extra1.get(n, 0.0) + 2.0 * extra}
                s["air.storage"] = now1
            for n, layer in self.layers.items():
                x0 = s0[f"{n}.x"]
                src = d1.get(f"{n}.sources")
                if src is None:
                    src = self._zero_sources(n, x0)
                if n in extra0 or n in extra1:
                    src = src + 0.5 * (extra0.get(n, 0.0) + extra1.get(n, 0.0))
                xb = 0.5 * (d0[f"{n}.x_boundary"] + d1[f"{n}.x_boundary"])
                s[f"{n}.x"] = layer.step(x0, air.flows_of_kind(qm, layer.flow_kinds), src, xb,
                                         h, capacity=cap.get(n))
            return self._pack(s), s

        z = z0.clone()
        if self.last is not None and torch.equal(self.last[2], z0):
            h_prev, z_prev, _ = self.last  # linear extrapolation of the previous step
            z = z0 + (z0 - z_prev) * (h / h_prev)
        dF: list[Tensor] = []
        dG: list[Tensor] = []
        f_old = g_old = None
        for _ in range(MIDPOINT_MAX_ITER):
            g, s1 = G(z)
            f = g - z
            tol = self._tolerance(g, s0)
            if bool((f.abs() <= tol).all()):
                self.last = (h, z0, self._pack(s1))
                if store is not None:
                    states, steps = self.storage_hist
                    if not states or not torch.equal(states[-1], s0["air.storage"]):
                        states, steps = (s0["air.storage"],), ()
                    self.storage_hist = ((*states, s1["air.storage"])[-STORAGE_ORDER:],
                                         (*steps, h)[-(STORAGE_ORDER - 1):]
                                         if STORAGE_ORDER > 1 else ())
                return s1
            w = 1.0 / tol  # weights of the least-squares mixing
            if f_old is not None:
                dF = [*dF, (f - f_old) * w][-_ANDERSON_DEPTH:]
                dG = [*dG, g - g_old][-_ANDERSON_DEPTH:]
            f_old, g_old = f, g
            if dF:
                A = torch.stack(dF, dim=1)
                gamma = torch.linalg.lstsq(A, (f * w).unsqueeze(1)).solution.squeeze(1)
                z = g - torch.stack(dG, dim=1) @ gamma
            else:
                z = g
        raise RuntimeError(
            f"simulate(scheme='midpoint'): a step of {h!r} s did not converge in "
            f"{MIDPOINT_MAX_ITER} iterations (max |change|/tolerance "
            f"{float((f.abs() / tol).max()):.3g})"
        )
