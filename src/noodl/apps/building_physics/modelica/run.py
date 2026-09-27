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

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.modelica.assemble import _MBLClosure
from noodl.apps.building_physics.modelica.storage import StorageClosure, mass_change
from noodl.model import Drivers, Model, State

Tensor = torch.Tensor
F64 = torch.float64
_SERIES = "series:"
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


def step_drivers(drivers: Mapping[str, Tensor], grid: Tensor, t: float) -> Drivers:
    """`drivers` at grid time `t`: every `"series:<key>"` entry sliced into `<key>`, and the
    series entries themselves dropped. Raises `ValueError` if `t` is not on the grid."""
    out: Drivers = {k: v for k, v in drivers.items() if not k.startswith(_SERIES)}
    series = {k[len(_SERIES):]: v for k, v in drivers.items()
              if k.startswith(_SERIES) and k != "series:time"}
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


def _closure(model: Model) -> _MBLClosure:
    for c in model.closures:
        if isinstance(c, _MBLClosure):
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
# `h = max(h0, (ratio - 1) t)/r` from `h0 = GRADING_H0` of the output interval, cut at
# every grid time, for `GRADING_WINDOW` output intervals, and then doubles up to the grid
# step (`_graded`); so does every stretch after a signal event. The `1/r` (grid step over
# output interval) makes the runs `extrapolate` combines (1 and 2 substeps) grade in the
# ratio of their steps too. `h` passes `tau` only at `t = 50 tau`. The window resolves
# ReverseBuoyancy's release of its 1325 Pa start imbalance over its first ~30 s: at a
# ratio of 1.1 over one interval it was off by 1.4 Pa and 2.9e-5 relative in T, at 1.02 over
# five by 8e-10 in p and 3.7e-8 in T (measured, extrapolated; 1.05: 3.8e-7 and 1.5e-7).
# The sources over a sub-step are the means of their quadratic reconstruction from the
# grid's step means (`_sub_drivers`).
GRADING_H0, GRADING_RATIO, GRADING_WINDOW = 1e-4, 1.02, 5


def _graded(t_start: float, t0: float, t1: float, h0: float, grading: bool,
            h_last: float | None, scale: float = 1.0) -> list[float]:
    """Sub-step ends in `(t0, t1]` (offsets from `t0`, the last exactly `t1 - t0`): inside
    the grading window (`grading`) `h = max(h0, (GRADING_RATIO - 1)(t - t_start))`; after
    it, `h` doubles from the last sub-step (`h_last`) until it is the grid step, so that the
    BDF2 step ratio of `_Midpoint` never exceeds 2. The last sub-step takes the remainder, at
    most 2 h (a sliver followed by a full step would make that ratio large)."""
    ends, t = [], t0
    while True:
        if grading:
            h = max(h0, (GRADING_RATIO - 1.0) * (t - t_start) * scale)
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


def _sub_drivers(d0: Drivers, d1: Drivers, u: float, v: float, h: float, *, first: bool,
                 constant: bool, d2: Drivers | None = None) -> tuple[Drivers, Drivers]:
    """The drivers at the two ends of the sub-step `(t0 + u, t0 + v)` of a grid step of
    length `h` (the graded start, `GRADING_RATIO`): point values linearly interpolated
    between the step's own `d0` and `d1`, and every source (`"<layer>.sources"`, a step mean
    on the grid) the mean over the sub-step of its quadratic reconstruction from the step
    means (`d0`: the mean before, or at the run's first time (`first`) the point value;
    `d1`: the step's; `d2`: the next step's, when there is one, else a linear one): its mean
    over the whole step is the grid's, and it is exact for a quadratic signal. After an
    event (`constant`, the grid step changed there) the source is held at its step mean and
    the point values at the step's end, as on the grid."""
    a, b = u / h, v / h
    du, dv = dict(d1), dict(d1)
    for key, v1 in d1.items():
        v0 = d0.get(key)
        if not isinstance(v1, Tensor) or not isinstance(v0, Tensor) or v0.shape != v1.shape:
            continue
        if key.endswith(".sources"):
            if constant:
                continue
            v2 = None if d2 is None else d2.get(key)
            du[key] = dv[key] = _cell_mean(v0, v1, v2, a, b, first)
        elif not constant and not torch.equal(v0, v1):
            du[key] = v0 + (v1 - v0) * a
            dv[key] = v0 + (v1 - v0) * b
    return du, dv


def _cell_mean(v0: Tensor, m: Tensor, v2: Tensor | None, a: float, b: float,
               first: bool) -> Tensor:
    """Mean over `(a, b)` (fractions of the step) of `s(x) = m + c1 (x - 1/2) +
    c2 ((x - 1/2)^2 - 1/12)`, whose mean over the step is `m`, fitted to the mean `v2` of the
    next step and either the mean `v0` of the step before or (`first`) the point value `v0`
    at the step's start; linear (`c2 = 0`) without `v2`."""
    if v2 is None:
        left = v0 if first else 0.5 * (v0 + m)  # s(0)
        c1, c2 = 2.0 * (m - left), torch.zeros_like(m)
    elif first:  # s(0) = v0, mean over (1, 2) = v2
        c2 = 1.5 * (v0 - m) + 0.75 * (v2 - m)
        c1 = v2 - m - c2
    else:        # means over (-1, 0) and (1, 2)
        c1, c2 = 0.5 * (v2 - v0), 0.5 * (v2 - 2.0 * m + v0)
    sq = ((b - 0.5) ** 3 - (a - 0.5) ** 3) / (3.0 * (b - a))
    return m + c1 * (0.5 * (a + b) - 0.5) + c2 * (sq - 1.0 / 12.0)


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
    grid `read_modelica(..., substeps=r)` builds: `r` steps per output interval plus the
    signals' event times), and the rows at `times` are returned. Each source driver is its
    mean over the one grid interval ending at its row (`assemble` module docstring,
    "Sources"), so the injected amounts are exact.

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
        d = step_drivers(drivers, grid, float(grid[idx[0]]))
        state = (_solve_air(model, state, d, **step_kwargs) if store is None
                 else _initial_air(model, store, state, d, **step_kwargs))
        record(state, d)
        mid = (_Midpoint(model, closure, step_kwargs, store) if scheme == "midpoint"
               else None)
        # The grading window (`GRADING_RATIO`): one output interval from the start and from
        # every change of the grid step, which marks a signal event
        # (`assemble.driver_grid`: a source pulse is its own step). In absolute time, so
        # that the runs `extrapolate` combines (1 and 2 substeps) grade alike.
        out_dt = float(times[1] - times[0]) if times.numel() > 1 else 0.0
        h0 = GRADING_H0 * out_dt
        t_start, h_prev, grading = float(grid[idx[0]]), None, False
        h_last: float | None = None  # the last sub-step taken
        for a, b in zip(idx, idx[1:], strict=False):
            for j in range(a + 1, b + 1):
                t0, t1 = float(grid[j - 1]), float(grid[j])
                d0, d1 = step_drivers(drivers, grid, t0), step_drivers(drivers, grid, t1)
                h = t1 - t0
                event = h_prev is not None and abs(h - h_prev) > 1e-9 * max(h, h_prev)
                if h_prev is None or event:
                    t_start, grading = t0, True
                h_prev = h
                window = GRADING_WINDOW * out_dt
                if grading and t0 >= t_start + window - 1e-9 * max(1.0, window):
                    grading = False  # then the ramp up to the grid step (`_graded`)
                ramp = h_last is not None and h_last < h * (1 - 1e-9)
                d2 = (step_drivers(drivers, grid, float(grid[j + 1]))
                      if j + 1 < grid.numel()
                      and abs(float(grid[j + 1] - grid[j]) - h) <= 1e-9 * h else None)
                # Scaled with the grid step (1/r of the output interval at r substeps):
                # the runs `extrapolate` combines then grade with steps in ratio 2 too.
                scale = h / out_dt if out_dt > 0 else 1.0
                cuts = (_graded(t_start, t0, t1, h0 * scale, grading, h_last, scale)
                        if store is not None and (grading or ramp) else [t1 - t0])
                for u, v in zip([0.0, *cuts[:-1]], cuts, strict=True):
                    du, dv = ((d0, d1) if len(cuts) == 1 else
                              _sub_drivers(d0, d1, u, v, h, first=j == idx[0] + 1,
                                           constant=event, d2=d2))
                    if mid is None:
                        state = model.step(state, dv, v - u, t=t0 + u, **step_kwargs)
                        if store is not None:
                            state = _resync_storage(model, store, state, dv)
                    else:
                        state = mid.step(state, du, dv, v - u)
                    h_last = v - u
                d = d1
            record(state, d)
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["time"] = times.clone()
    return out


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


class _Midpoint:
    """The symmetric step of `simulate(scheme="midpoint")`.

    Over a step of length `h` from state 0 to state 1, each transport layer advances with
    its own exact scheme (`TransportLayer.step`, the flows held constant over the step) at
    the MEAN flows `(q0 + q1)/2`, the mean boundary values `(x_b0 + x_b1)/2` and its step-mean
    source driver plus the mean of the closure's state-dependent sources (the moist-air heat
    carrier, `_MBLClosure.cp_correction`). State 1 is the fixed point of that map together
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

    def __init__(self, model: Model, closure: _MBLClosure, solve_kwargs: dict,
                 store: StorageClosure | None = None) -> None:
        if len(model.potential) != 1:
            raise ValueError("simulate(scheme='midpoint'): one potential layer expected")
        ((self.air_name, self.air),) = model.potential.items()
        self.layers = dict(model.transport)
        self.closure = closure
        self.store = store
        # (h, "air.storage" at the start, at the end) of the last step, for BDF2.
        self.storage_last: tuple[float, Tensor, Tensor] | None = None
        self.kw = dict(solve_kwargs, differentiable=False)
        self.last: tuple[float, Tensor, Tensor] | None = None  # (h, z at start, z at end)

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
        if self.storage_last is None or not torch.equal(self.storage_last[2], prev):
            rate, offset = 1.0 / h, torch.zeros(store.V.shape, dtype=F64)
        else:
            h_prev, before, _ = self.storage_last
            om = h / h_prev
            dm_prev = mass_change(store.V, prev[..., 1], prev[..., 0], before[..., 1],
                                  before[..., 0], store.p_ref)
            rate = (1 + 2 * om) / ((1 + om) * h)
            offset = om ** 2 / ((1 + om) * h) * dm_prev
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

    def step(self, s0: State, d0: Drivers, d1: Drivers, h: float) -> State:
        air, name = self.air, self.air_name
        q0 = s0[f"{name}.q"]
        extra0 = self._extra_sources(s0, d0)
        z0 = self._pack(s0)
        store = self.store
        if store is not None:
            drv0, _, net0, s_air, air_drv = self._storage_drivers(s0, d0, d1, h)

        def G(z: Tensor) -> tuple[Tensor, State]:
            s = self._unpack(z, s0)
            drv = dict(d1)
            drv.update(self.closure(s, drv))
            if store is not None:
                drv.update(air_drv)
            phi, q = air.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"),
                               phi0=s[f"{name}.phi"][..., air.interior], **self.kw)
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
                    self.storage_last = (h, s0["air.storage"], s1["air.storage"])
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
