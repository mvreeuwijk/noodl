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

from noodl.apps.building_physics.modelica.assemble import (
    _HeldMover,
    _ZoneStateClosure,
    boundary_rate_key,
    is_boundary_rate_key,
)
from noodl.apps.building_physics.modelica.feedback import FeedbackHeat
from noodl.apps.building_physics.modelica.storage import StorageClosure, mass_change
from noodl.elements import ConstitutiveLaw, LawContext
from noodl.model import Drivers, Model, State
from noodl.modelica.timeline import extrapolate as extrapolate  # this reader's public API
from noodl.modelica.timeline import grid_index
from noodl.modelica.timeline import step_drivers as step_drivers
from noodl.solvers.errors import ConvergenceError

Tensor = torch.Tensor
F64 = torch.float64
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


def _closure(model: Model) -> _ZoneStateClosure:
    for c in model.closures:
        if isinstance(c, _ZoneStateClosure):
            return c
    raise TypeError("simulate: the model was not built by read_modelica (no MBL closure)")


def _storage(model: Model) -> StorageClosure | None:
    return next((c for c in model.closures if isinstance(c, StorageClosure)), None)


def _feedback(model: Model) -> FeedbackHeat | None:
    return next((c for c in model.closures if isinstance(c, FeedbackHeat)), None)


# Movers with volume mass storage. A `FlowControlled_dp` (`assemble._MoverHead`, SteadyState
# energy dynamics) has no inertia and no volume of its own: its mass flow is an ALGEBRAIC
# variable, not a state. Within a step the airflow solve fixes it (MNA unknown); the value
# carried in `"air.q"` is only the last solve's, read where noodl evaluates the flows WITHOUT
# solving: the t = StartTime row (`_initial_air`) and state 0 taken again with other flow
# directions (`_Midpoint._start_with_directions`, which passes state 0's own solved `q`).
# At t = StartTime every storing zone is held at its start pressure, so the rise relation
# `phi_b - phi_a = dp` does not involve the flow, and "the element laws at the start
# pressures" leave it undetermined. MBL's DAE fixes it by index reduction: the rise ties the
# pressures of its two ends, so their derivatives are tied too, `der(p_b) = der(p_a)` at a
# held rise, and the mover's flow is the one that makes the two volumes' storage rates
# agree (`_start_mover_flows`). For two volumes between orifices that is
# `m_fan = (V_b q_a + V_a q_b) / (V_a + V_b)`. This is what OpenModelica reports at
# t = StartTime from a start that satisfies the rise, the only start it accepts (one that
# violates it is an inconsistent over-determined initialisation there); neither zero nor
# the storage-free steady flow is. From a start that violates the rise the first step
# projects the pressures onto it (mass-conserving, an impulse through the mover), and
# `_balanced` reports that start as out of balance so that the step is graded.


def _start_mover_flows(air, store: StorageClosure, phi: Tensor, drv: Drivers) -> Tensor:
    """Every air-layer edge's flow at t = StartTime with volume mass storage, the zones at
    the pressures `phi` (full node): the element laws at `phi` on the potential-controlled
    edges and, on the implicit (mover) edges, the flows of the index-reduced balance
    (comment above): unknowns the interior nodes' pressure rates `v` and the mover flows
    `q_F`, with `C v = net(q) + s` at every storing zone and every node a mover touches
    (`C = V k / (1 + a)`, the storage's own pressure capacity, `ZoneStorage.dflow` per unit
    rate; 0 at a junction, a zone without a mass balance or one whose initial equation is
    `der(p) = 0`) and `v_a = v_b` across every mover (the rise held over the instant: a
    rising `dp_in` signal's own rate is not added; a boundary's `v` is 0). Without implicit
    edges this is `air.flows(phi, drv)`."""
    if not air.mna:
        return air.flows(phi, drv)
    F = air.implicit_edges
    n_e = len(store.air_src)
    q_p = air.flows(phi, drv, torch.zeros(phi.shape[:-1] + (n_e,), dtype=F64))
    s = drv.get("air.sources")
    b = store.net(q_p, store.V) + (0.0 if s is None else s.to(F64))
    # The rows: every storing zone, and every other interior node a mover touches (there
    # `C = 0`, its KCL fixes the mover's flow, and its rate follows the mover's other end);
    # the rest of the interior is algebraic in `phi` and does not enter.
    ends = {int(n) for e in F.tolist() for n in (store.air_src[e], store.air_tgt[e])}
    storing = set(store.air_nodes.tolist())
    interior = [n for n in air.interior.tolist() if n in storing or n in ends]
    pos = {node: i for i, node in enumerate(interior)}
    n_i, n_f = len(interior), int(F.numel())
    # A zone whose initial equation is der(p) = 0 (solved by `init_layer`) has no storage
    # rate at t = StartTime: like a junction, C = 0 there.
    free = (set(store.init_layer.interior.tolist()) if store.init_layer is not None
            else set())
    k = store.th.k(drv["T"], drv["X_w"]).expand(store.V.shape)
    a = store.gain(drv)
    a = torch.zeros(a.shape[:-1] + store.V.shape, dtype=F64).index_copy(-1, store.air_nodes, a)
    c_s = store.V * k / (1.0 + a)
    cap = torch.stack([c_s[..., n] if n in storing and n not in free
                       else torch.zeros_like(c_s[..., n]) for n in interior], dim=-1)
    M = torch.zeros(cap.shape[:-1] + (n_i + n_f, n_i + n_f), dtype=F64)
    M[..., range(n_i), range(n_i)] = cap
    for j, e in enumerate(F.tolist()):
        a, t = int(store.air_src[e]), int(store.air_tgt[e])
        if a in pos:  # the mover's flow leaves port_a ...
            M[..., pos[a], n_i + j] = 1.0
            M[..., n_i + j, pos[a]] = 1.0
        if t in pos:  # ... and enters port_b
            M[..., pos[t], n_i + j] = -1.0
            M[..., n_i + j, pos[t]] = -1.0
    rhs = torch.cat([b[..., interior], torch.zeros(b.shape[:-1] + (n_f,), dtype=F64)],
                    dim=-1)
    x = torch.linalg.solve(M, rhs.expand(M.shape[:-1]))
    return q_p.index_copy(-1, F, x[..., n_i:])


def _carry_held_movers(layer, q: Tensor, carried: Tensor) -> Tensor:
    """`q` with the columns of every mover `layer` holds at both ends (`assemble._HeldMover`,
    a placeholder zero there) taken from `carried`, the state's own solved flows."""
    for kind in dict.fromkeys(layer.kinds):
        law, sl = layer.law_for(kind)
        if isinstance(law, _HeldMover):
            q = torch.cat([q[..., :sl.start], carried[..., sl], q[..., sl.stop:]], dim=-1)
    return q


def _initial_air(model: Model, store: StorageClosure, state: State, drivers: Drivers,
                 **solve_kwargs) -> State:
    """The t = StartTime airflow of a model with volume mass storage (`storage` module
    docstring, "Initial state"): every storing zone at its start pressure (the state's own
    `"air.phi"`) and the flows the elements give there, or, when some zone's initial
    equation is `der(p) = 0`, the air layer solved with the others held at their start
    pressures. The carried `"air.storage"` is re-evaluated at the result."""
    closure = _closure(model)
    air = model.kirchhoff["air"]
    new = dict(state)
    for _ in range(_DIRECTION_PASSES):
        drv = dict(drivers)
        drv.update(closure(new, drv))
        before = new.get("air.q")
        if store.init_layer is None:
            new["air.q"] = _start_mover_flows(air, store, state["air.phi"], drv)
        else:
            init = store.init_layer
            phi_b = state["air.phi"][..., init.bound]
            new["air.phi"], new["air.q"] = init.solve(phi_b, drv, drv.get("air.sources"),
                                                      phi0=None, **solve_kwargs)
            if air.mna:  # the movers' flows of the whole layer, at the solved pressures
                new["air.q"] = _start_mover_flows(air, store, new["air.phi"], drv)
        if _same_directions(closure, before, new["air.q"]):
            break
    else:
        _unsettled(closure, before, new["air.q"], "the start")
    drv.update(closure(new, drv))
    new["air.storage"] = store(new, drv)["air.storage"]
    return new


# The airflow at a state whose flows decide an `actual` column's density (`assemble`, "Flow
# elements in series") is solved again with the directions of its own flows until they agree,
# at most this many times (a single reversal settles in two); then `_unsettled` raises.
_DIRECTION_PASSES = 4


def _changed(closure: _ZoneStateClosure, before: Tensor | None, after: Tensor,
             at_switch: float | None = None) -> list[str]:
    """The elements whose flow `after` has another direction than `before` had, among those
    whose direction an `actual` column's density reads (none without such a column). With
    `at_switch`, only those where `|before|` is within it (a reversal at the start of a
    step, `_Midpoint.step`)."""
    if not closure.heads:
        return []
    if before is None:
        before = torch.ones_like(after)  # the closure's directions before any solve
    names = []
    for head in closure.heads:
        b, a = before[..., head.edge], after[..., head.edge]
        changed = (b >= 0) != (a >= 0)
        if at_switch is not None:
            changed = changed & (b.abs() <= at_switch)
        if bool(changed.any()):
            names.append(head.kind.split(":", 1)[1])
    return names


def _same_directions(closure: _ZoneStateClosure, before: Tensor | None, after: Tensor,
                     at_switch: float | None = None) -> bool:
    """Whether no direction an `actual` column reads changed (`_changed`)."""
    return not _changed(closure, before, after, at_switch)


def _unsettled(closure: _ZoneStateClosure, before: Tensor | None, after: Tensor,
               where: str) -> None:
    """Raise `ConvergenceError` naming the elements whose flow direction did not settle in
    `_DIRECTION_PASSES` solves: each direction's density drives the flow the other way."""
    names = _changed(closure, before, after) or [h.kind.split(":", 1)[1]
                                                 for h in closure.heads]
    raise ConvergenceError(
        f"simulate: the flow direction of {', '.join(names)} does not settle at {where} "
        f"(densitySelection = actual): with the density of the air coming from either end, "
        f"the flow runs the other way; after {_DIRECTION_PASSES} solves it still reverses"
    )


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
    `_Switches`): point values interpolated in the grid step holding each end (a boundary
    value with its rates, `assemble.boundary_rate_key`, by the cubic Hermite
    through the step's ends, `_hermite`, which also gives its rate there; any other linearly),
    and
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
        h = self.g[j] - self.g[j - 1]
        a = (t - self.g[j - 1]) / h
        for key, v1 in d1.items():
            v0 = d0.get(key)
            if (key.endswith(".sources") or is_boundary_rate_key(key)
                    or not isinstance(v1, Tensor)
                    or not isinstance(v0, Tensor) or v0.shape != v1.shape):
                continue
            r0 = d0.get(boundary_rate_key(key, "right"))
            r1 = d1.get(boundary_rate_key(key, "left"))
            if isinstance(r0, Tensor) and isinstance(r1, Tensor):
                # A boundary value with its rates: the cubic Hermite through both ends.
                out[key], rate = _hermite(v0, v1, r0, r1, h, a)
                out[boundary_rate_key(key, "left")] = rate
                out[boundary_rate_key(key, "right")] = rate
                continue
            if torch.equal(v0, v1):
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


# The drivers holding a boundary value's rate of change on either side of a grid time
# (`assemble.boundary_rate_key`; `assemble._Builder.boundary_rates`) are interpolated with the
# value itself (`_hermite`), not on their own (`is_boundary_rate_key`).


def _hermite(v0: Tensor, v1: Tensor, r0: Tensor, r1: Tensor, h: float,
             a: float) -> tuple[Tensor, Tensor]:
    """The cubic Hermite through `(0, v0)` with slope `r0` and `(h, v1)` with slope `r1`,
    and its slope, at `a h`."""
    h00, h10, h01, h11 = (2 * a**3 - 3 * a**2 + 1, a**3 - 2 * a**2 + a, -2 * a**3 + 3 * a**2,
                          a**3 - a**2)
    d00, d10, d01, d11 = (6 * a**2 - 6 * a, 3 * a**2 - 4 * a + 1, -6 * a**2 + 6 * a,
                          3 * a**2 - 2 * a)
    value = h00 * v0 + h10 * h * r0 + h01 * v1 + h11 * h * r1
    rate = (d00 * v0 + d01 * v1) / h + d10 * r0 + d11 * r1
    return value, rate


def _event_indices(drivers: Drivers, grid: Tensor) -> tuple[set[int] | None, set[int]]:
    """The grid positions of the jumps and of the kinks (`assemble.grid_events`), or
    `(None, set())` when the drivers carry no event lists."""
    if "series:jumps" not in drivers:
        return None, set()

    def where(ts) -> set[int]:
        ts = torch.as_tensor(ts, dtype=F64)
        if ts.numel() == 0:
            return set()
        return set(grid_index(grid, ts, caller="simulate"))

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
    """The Kirchhoff layers alone at `state` (closures first): the quasi-steady flows of the
    initial state, without advancing any transport layer.

    Newton starts from the layer's own linear initial guess (`phi0=None`,
    `KirchhoffLayer.linear_init`), not from the state's `p_start` seed: every zone at
    `p_start` is far from a stack's hydrostatic solution, and from there Newton cycles on
    MBL's discretised doors (`Validation/ThreeRoomsContamDiscretizedDoor.mo`: residual
    0.084 kg/s after 50 iterations), while the linear guess converges in three."""
    closure = next((c for c in model.closures if isinstance(c, _ZoneStateClosure)), None)
    new = dict(state)
    for _ in range(_DIRECTION_PASSES):
        drv = dict(drivers)
        for c in model.closures:
            drv.update(c(new, drv))
        before = new.get("air.q")
        for name, layer in model.kirchhoff.items():
            phi, q = layer.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"),
                                 phi0=None, **solve_kwargs)
            new[f"{name}.phi"], new[f"{name}.q"] = phi, q
        if closure is None or _same_directions(closure, before, new["air.q"]):
            break
    else:
        _unsettled(closure, before, new["air.q"], "the start")
    return new


def simulate(model: Model, state: State, drivers: Drivers, times, *,
             scheme: str = "implicit", **step_kwargs) -> dict[str, Tensor]:
    """Time histories at `times` (increasing times of the driver grid `"series:time"`).

    Returns `"time"` `(N,)`, `"air.q"` `(N, b)` in the air layer's edge order (see
    `ModelicaNames.edges`), `"air.phi"` `(N, n)` gauge pressure (relative to
    `ModelicaNames.p_ref`), `"p"` `(N, n)` absolute
    pressure (Pa), `"T"` `(N, n)` (K), `"X_w"` `(N, n)`, `"C"` `(N, n, K)` for the species
    layer's mass fractions (water last when carried) when the model has one, and
    `"controls.outputs"` `(N, n_blocks)` (the network's `outputs_key`, the key
    `simulation.simulate` uses too), every feedback block's output (`ModelicaNames.controls`),
    when it has a feedback network (`feedback.FeedbackHeat`). `step_kwargs`
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
    `ConstitutiveLaw.switching`), a feedback controller's limiter reaches a limit
    (`ControlNetwork.switching`) or an edge flow changes sign (the transport layers upwind
    on it). A step across one is split there, the crossing located on the step's own runs
    (`_advance_located`), so that the right-hand side is smooth within every step: a door's
    flows reversing through its bands across a 14.4 s step were off by 1.1e-4 of the flow
    floor at OneOpenDoor's reversals (2131 s), 8.3e-6 with the switches located (measured,
    extrapolated from 1 and 2 substeps).

    `scheme`: `"implicit"` (the default) is `Model.step` with `coupling="iterate"`: the
    transport layers advance with the flows held at their END-of-step values, first order
    in the step. `"midpoint"` (`_Midpoint`) advances each exact transport layer over two
    halves, the first with the start-of-step flows, capacity and state-dependent sources, the
    second with the end-of-step ones, the boundary values through both ends (the cubic
    Hermite with their rates where they vary, else the straight line): symmetric and second
    order in the step, so two runs at `r` and `2 r` substeps combine by `extrapolate` to
    fourth order. It is forward-only (no gradient) and for unbatched models.
    """
    if scheme not in ("implicit", "midpoint"):
        raise ValueError(f"simulate: unknown scheme {scheme!r}; 'implicit' or 'midpoint'")
    atol = AIR_ATOL if _storage(model) is None else STORAGE_AIR_ATOL
    step_kwargs = {"atol": atol, "rtol": AIR_RTOL, **step_kwargs}
    times = torch.as_tensor(times, dtype=F64)
    grid = torch.as_tensor(drivers.get("series:time", times), dtype=F64)
    idx = grid_index(grid, times, caller="simulate")
    closure = _closure(model)
    store = _storage(model)
    control = _feedback(model)
    dynamic = bool(model.transport)
    rows: dict[str, list[Tensor]] = {"air.q": [], "air.phi": [], "p": [], "T": [], "X_w": []}
    if closure.sp_interior is not None:
        rows["C"] = []
    if control is not None:
        rows[control.network.outputs_key] = []

    def record(state: State, d: Drivers) -> None:
        extra = closure(state, d)
        rows["air.q"].append(state["air.q"])
        rows["air.phi"].append(state["air.phi"])
        rows["p"].append(closure.pressures(state, d))
        rows["T"].append(extra["T"])
        rows["X_w"].append(extra["X_w"] if "X_w" in extra else d["X_w"])
        if "C" in rows:
            rows["C"].append(closure.species(state, d))
        if control is not None:
            rows[control.network.outputs_key].append(control.network.evaluate(state, d))

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
    instead. A start whose pressures violate a mover's rise (run module, "Movers with
    volume mass storage") is out of balance too: the first step moves them onto it."""
    q = state["air.q"]
    s_air = d.get("air.sources")
    s_air = torch.zeros(store.V.shape, dtype=F64) if s_air is None else s_air.to(F64)
    net = store.net(q, s_air)[..., store.air_nodes]
    scale = max(float(q.abs().max()) if q.numel() else 0.0, 1e-5)
    if not bool((net.abs() <= BALANCE_RTOL * scale).all()):
        return False
    air = model.kirchhoff["air"]
    if not air.mna:
        return True
    drv = dict(d)
    drv.update(_closure(model)(state, drv))
    dp = air.dp(state["air.phi"], drv)
    ctx = LawContext(mode="dc")
    for kind in air.kinds:
        law, sl = air.law_for(kind)
        if law.law_form == "implicit":
            r = law.residual(dp[..., sl], q[..., sl], drv, ctx)
            if not bool((r.abs() <= BALANCE_RTOL * (dp[..., sl].abs() + 1.0)).all()):
                return False
    return True


def _version(v) -> int | None:
    return v._version if isinstance(v, Tensor) else None


def _same(a, b) -> bool:
    """Whether two driver values are the same values (the same object, or equal tensors)."""
    if a is b:
        return True
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    return False


class _Switches:
    """The switching values at a state: every value changes sign where the step's right-hand
    side stops being smooth. These are the air layer's laws' own switches
    (`ConstitutiveLaw.switching`: a regularisation band's edge, where the law changes
    piece), every edge flow's sign, in units of `q_scale` (the transport layers upwind on
    it: the advected state switches from one end's to the other's where a flow reverses)
    and, with a feedback network, its limiters' corners (`ControlNetwork.switching`, in
    units of each limiter's range), last. `parts` are the laws that have switches. `jumps`
    holds the positions of the switches where the right-hand side jumps rather than
    changes piece: the signs of the flows an `actual` column's density reads, and of the
    flows in series with them (`assemble`, "Flow elements in series")."""

    def __init__(self, model: Model, closure: _ZoneStateClosure, q_scale: float) -> None:
        ((self.name, self.air),) = model.kirchhoff.items()
        self.closure = closure
        self.control = _feedback(model)
        self.q_scale = q_scale
        self._last: tuple | None = None  # (state, point drivers, values) of the last call
        self.parts = []
        self.jump_edges = list(closure.jump_edges)
        self.jumps: set[int] = set()
        for kind in self.air.kinds:
            el, sl = self.air.law_for(kind)
            if type(el).switching is not ConstitutiveLaw.switching:
                self.parts.append((el, sl))

    def __call__(self, state: State, d: Drivers) -> Tensor:
        # A step's start is the last step's end (`_advance_located` evaluates both): the
        # same state objects and the same point drivers give the same values. The sources
        # (step means, which differ from step to step) do not enter the switches.
        point = {k: v for k, v in d.items() if "sources" not in k}
        last = self._last
        if (last is not None and last[0].keys() == state.keys()
                and all(state[k] is v and _version(v) == ver
                        for k, (v, ver) in last[0].items())
                and last[1].keys() == point.keys()
                and all(_same(point[k], v) for k, v in last[1].items())):
            return last[2]
        vals = self._values(state, d)
        self._last = ({k: (v, _version(v)) for k, v in state.items()}, point, vals)
        return vals

    def _values(self, state: State, d: Drivers) -> Tensor:
        drv = dict(d)
        drv.update(self.closure(state, drv))
        dp = self.air.dp(state[f"{self.name}.phi"], drv)
        vals = [el.switching(dp[..., sl], drv) for el, sl in self.parts]
        offset = sum(v.numel() for v in vals if v is not None)
        self.jumps = {offset + e for e in self.jump_edges}
        vals.append(state[f"{self.name}.q"] / self.q_scale)
        if self.control is not None:
            vals.append(self.control.network.switching(state, drv))
        return torch.cat([v.reshape(-1) for v in vals if v is not None])


# State-event location (`_advance_located`): a switch is crossed in a step when its value
# changes sign between the step's ends and was not within SWITCH_SKIP of zero at the start
# (a switch just located stays there); it is located to |value| <= SWITCH_TOL (in units of
# the band's width, `ConstitutiveLaw.switching`) in at most SWITCH_MAX_ITER trial steps.
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
    alike, their steps in ratio 2.

    A JUMP (`_Switches.jumps`: a flow reversal an `actual` column's density follows) is located
    by bisection and the run taken to the last trial before it; the step from there starts at
    the jump (`_Midpoint.jump_from`). A jump within the margin of the run's end, or in a run too
    short to split, is moved to the end: the run keeps its starting directions throughout
    (`_Midpoint.frozen`) and the next run starts at the jump. One within the margin of the
    run's start is moved to the start: the run is made with the directions it ends with. That
    approximation moves the jump by at most the margin (`max(SWITCH_MARGIN (b - a), min_step)`,
    `min_step` independent of the substeps), an error of the margin times the jump in the
    flow."""
    n = len(pts) - 1
    a, b = pts[0], pts[-1]
    skip: set[int] = set()  # switches located (or within the margin): not split at again

    def done(end: State) -> State:
        if mid.frozen is not None:  # a jump moved to the end: the next run starts at it
            # `end` was solved with the run's starting directions, though its flows may have
            # crossed zero by then: the next step compares against those (`_Midpoint.step`).
            mid.jump_from, mid.jump_directions = end, mid.frozen
            mid.frozen = None
        return end

    while True:
        g_a = switches(state, sd.span(pts[0], pts[1])[0])
        snap = mid.snapshot()
        end, d_end = advance(state, pts)
        g_b = switches(end, d_end)
        cross = _crossed(g_a, g_b)
        if skip:
            cross[list(skip)] = False
        if not bool(cross.any()):
            return done(end)
        k = _first(g_a, g_b, cross)
        margin = max(SWITCH_MARGIN * (b - a), min_step)
        if b - a <= 4 * margin:
            jumps = [j for j in switches.jumps if bool(cross[j])]
            if jumps and mid.frozen is None:  # whichever switch is crossed first
                mid.restore(snap)
                mid.frozen = state["air.q"]  # the jump to the end of this short run
                skip.update(jumps)
                continue
            return done(end)  # a step this short is not split (SWITCH_MARGIN)
        lo, hi, g_lo, g_hi, side = a, b, float(g_a[k]), float(g_b[k]), 0
        best = before_k = None
        for _ in range(SWITCH_MAX_ITER):
            # Across a jump the switch's value is no guide to where it lies: bisection.
            t = (0.5 * (lo + hi) if k in switches.jumps
                 else lo + (hi - lo) * g_lo / (g_lo - g_hi))
            t = min(max(t, a + margin), b - margin)
            mid.restore(snap)
            trial, d_t = advance(state, _split(a, t, n))
            g_t = switches(trial, d_t)
            best = (t, trial, mid.snapshot())
            if float(g_t[k]) * float(g_a[k]) > 0:
                before_k = best  # the latest trial ending before switch k
            before = _crossed(g_a, g_t) & (g_t.abs() > SWITCH_TOL)
            if skip:
                before[list(skip)] = False
            if bool(before.any()):
                j = _first(g_a, g_t, before)
                if j != k:  # another switch is crossed first: locate that one
                    k, lo, hi, g_lo, g_hi, side = j, a, t, float(g_a[j]), float(g_t[j]), 0
                    before_k = None
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
        jump = k in switches.jumps
        if jump and before_k is not None:
            # Where the right-hand side jumps, the run is taken to the trial nearest the
            # switch on its near side (to SWITCH_TOL or the time resolution): one beyond it
            # was stepped across the jump.
            best = before_k
        t, trial, post = best
        skip.add(k)
        if jump and before_k is None:
            # A jump within the margin of the start: the step starts at it.
            mid.restore(snap)
            mid.jump_from, mid.jump_directions = state, None
            continue
        if jump and t >= b - margin * (1 + 1e-9):
            # A jump within the margin of the end: the run keeps its directions to the end.
            mid.restore(snap)
            mid.frozen = state["air.q"]
            continue
        if t <= a + margin * (1 + 1e-9) or t >= b - margin * (1 + 1e-9):
            # The switch lies within the margin of an end: the step is not split there.
            mid.restore(snap)
            continue
        state = trial
        mid.restore(post)
        mid.frozen = None
        if jump:
            # The step from here starts at the jump (`_Midpoint.step`).
            mid.jump_from, mid.jump_directions = state, None
        a, skip = t, {k}
        pts = _split(a, b, n)


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


def _plus(a: Tensor | None, b: Tensor | None) -> Tensor | None:
    """`a + b`, either of which may be absent (`None`)."""
    if a is None:
        return b
    return a if b is None else a + b


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

    Over a step of length `h` from state 0 to state 1, each transport layer with an exact
    scheme advances over two halves (`TransportLayer.step`, the flows held constant over
    each): the first with state 0's flows, storage capacity and state-dependent sources (the
    moist-air heat carrier, `_ZoneStateClosure.cp_correction`; the storage terms), the
    second with state 1's, and both with the step-mean source driver. The boundary values
    run through both ends' values over the step: the cubic Hermite with their rates
    (`assemble.boundary_rate_key`, `assemble._Builder.boundary_rates`) where they vary,
    else the straight line (`TransportLayer.step`'s `x_boundary_end`, `x_boundary_rates`).
    Both matter for a zone that flushes much faster than the step (PressurizationData's
    room: about 10 s against steps of minutes), which ends each step at the boundary value
    less its lag, its flushing time times the boundary's rate there. The step-mean boundary
    value left it half a step behind (0.035 K of a 0.07 K-per-step outdoor-air ramp at 396 s
    steps); the straight line, behind by the chord's slope instead of the end's (2.0e-2 K on
    a 5 K sine over 60 s steps, 2.1e-5 K with the Hermite); the mean flows, with the mean
    flow's flushing time instead of the end's, off by about `tau h q'/(2 q)` times the
    boundary's rate. Errors first order in the step that `extrapolate` cannot remove. With
    state 1's flows over the second half, what is left of the flows' change is about
    `tau^2 q'/q` times the rate, independent of the step while it is many flushing times
    long (2e-5 K on PressurizationData, measured). A layer
    whose scheme is not exact advances over the whole step at the mean flows, capacity,
    sources and boundary values. State 1 is the fixed point of that map together
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
    change `(m1 - m0)/h` as the storage rate.

    A feedback network's heat flows (`feedback.FeedbackHeat`) are affine in their zone's
    temperature on each piece of its limiters, `A + B T`. Both halves take the piece of the
    whole step (`FeedbackHeat.pieces_between`: a step lies on one side of every corner,
    `_Switches`), each with `A` and `B` at its own end, `A` as a source and `-B/C` as the
    removal rate of the half's exact transport step: implicit in the zone temperature, so
    that a controller far faster than the step (`k gain/C` of 0.2 1/s against 7.2 s steps)
    stays exact. A removal rate given as the driver `"<layer>.removal"` adds to it, each half
    taking the driver's value at its own end."""

    def __init__(self, model: Model, closure: _ZoneStateClosure, solve_kwargs: dict,
                 store: StorageClosure | None = None) -> None:
        if len(model.kirchhoff) != 1:
            raise ValueError("simulate(scheme='midpoint'): one Kirchhoff layer expected")
        ((self.air_name, self.air),) = model.kirchhoff.items()
        self.layers = dict(model.transport)
        self.closure = closure
        self.store = store
        self.control = _feedback(model)
        # (h, "air.storage" at the start, at the end) of the last step, for BDF2.
        # The storage history for the BDF rate: the carried "air.storage" at the last few
        # step ends (the last is the next step's start) and the steps between them.
        self.storage_hist: tuple[tuple[Tensor, ...], tuple[float, ...]] = ((), ())
        self.kw = dict(solve_kwargs, differentiable=False)
        self.last: tuple[float, Tensor, Tensor] | None = None  # (h, z at start, z at end)
        # The state a located jump was taken to (`_advance_located`): a step from it starts
        # at the jump (`step`). While `frozen` is set (the flows at the start of a run whose
        # jump was moved to its end), the steps take their directions from it.
        self.jump_from: State | None = None
        self.frozen: Tensor | None = None
        # The flows whose directions `jump_from` was solved with, when they are not its own
        # (a jump moved to the end of a run), else None.
        self.jump_directions: Tensor | None = None

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

    def _extra_sources(self, state: State, d: Drivers,
                       written: Drivers | None = None) -> dict[str, Tensor]:
        """The closure's state-dependent sources: what it adds to `"<layer>.sources"`
        (`written`: the closure's output at `state`, `d`, when already evaluated)."""
        if written is None:
            written = self.closure(state, d)
        return {n: written[f"{n}.sources"] - d.get(f"{n}.sources", 0.0)
                for n in self.layers if f"{n}.sources" in written}

    def _flows(self, layer, q: Tensor) -> Tensor:
        """The transport layer's flows from the air layer's `q`: its own kinds', or with
        elements in series the closure's columns (`assemble`, "Flow elements in series")."""
        series = self.closure.series
        if series is None:
            return self.air.flows_of_kind(q, layer.flow_kinds)
        return q[..., series.flow_cols]

    def _start_with_directions(self, s0: State, d0: Drivers, q1: Tensor) -> State:
        """State 0 with its algebraic pressures and flows solved again with the directions of
        the flows `q1` at the step's end (an `actual` column's density, `step`)."""
        name = self.air_name
        s = dict(s0)
        s[f"{name}.q"] = q1
        drv = dict(d0)
        drv.update(self.closure(s, drv))
        layer = self.closure.state0_layer
        if self.store is None:
            phi, q = self._solve_air(drv, drv.get(f"{name}.sources"),
                                     s0[f"{name}.phi"][..., self.air.interior], 0.0)
        elif layer is None:
            # A mover's flow (an implicit edge) is not a function of the pressures: state 0's
            # own solved flow is carried (run module, "Movers with volume mass storage").
            phi, q = s0[f"{name}.phi"], self.air.flows(s0[f"{name}.phi"], drv,
                                                       s0[f"{name}.q"])
        else:
            phi, q = self._solve_air(drv, drv.get(f"{name}.sources"),
                                     s0[f"{name}.phi"][..., layer.interior], 0.0, layer,
                                     s0[f"{name}.phi"][..., layer.bound])
            q = _carry_held_movers(layer, q, s0[f"{name}.q"])
        s[f"{name}.phi"], s[f"{name}.q"] = phi, q
        return s

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

    def _solve_air(self, drv: Drivers, sources, phi0: Tensor | None, h: float, layer=None,
                   phi_b: Tensor | None = None):
        """The air layer (`layer`, default the model's, with boundary values `phi_b`,
        default `"air.phi_boundary"`) at state 1 (`air.solve`, with the caller's solve
        options). With storage a solve that stalls within STORAGE_STALL times its tolerance is
        accepted (the round-off of the storage terms). With an `actual` column a solve from
        `phi0` that fails is made again from the layer's linear initial guess: from the
        pressures before a run of elements in series reversed (its junction 37 Pa off,
        ChimneyShaftNoVolume at 0.19 s) Newton did not converge, from the linear guess it
        does. A solve that still fails raises `ConvergenceError`, unless the caller asked
        for `on_failure="return"`."""
        name = self.air_name
        layer = self.air if layer is None else layer
        pb = drv[f"{name}.phi_boundary"] if phi_b is None else phi_b
        retry = bool(self.closure.heads) and phi0 is not None
        if self.store is None and not retry:
            return layer.solve(pb, drv, sources, phi0=phi0, **self.kw)
        limit = 0.0 if self.store is None else STORAGE_STALL * self.kw.get("atol",
                                                                           STORAGE_AIR_ATOL)

        def attempt(start):
            diag: dict = {}
            phi, q = layer.solve(pb, drv, sources, phi0=start, diagnostics=diag,
                                 **{**self.kw, "on_failure": "return"})
            res = float(torch.as_tensor(diag["residual_norm"]).max())
            return phi, q, bool(torch.as_tensor(diag["converged"]).all()) or res <= limit, res

        phi, q, ok, res = attempt(phi0)
        if not ok and retry:
            phi, q, ok, res = attempt(None)
        if not ok and self.kw.get("on_failure", "raise") != "return":
            raise ConvergenceError(
                f"simulate(scheme='midpoint'): the airflow solve of a {h!r} s step did "
                f"not converge (residual {res:.3g} kg/s)")
        return phi, q

    def step(self, s0: State, d0: Drivers, d1: Drivers, h: float) -> State:
        """State 1 after a step of `h` from state 0 (class docstring). Where a flow whose
        direction decides an `actual` column's density reverses at the step's start (located
        there, `_advance_located`: `jump_from`, or within SWITCH_SKIP of its switch at state 0,
        of the other direction at state 1), the density jumps there: state 0's algebraic
        pressures and flows are taken with the end's directions, and the step is made again
        from there. A reversal inside the step is left to `_advance_located`, which splits
        the step at it, or moves one within its margin of either end to that end (an error of
        the margin times the jump in the flow; while `frozen`, the step keeps the directions
        it starts with). Directions that do not settle raise `ConvergenceError`."""
        name = self.air_name
        strict = s0 is self.jump_from  # every direction counts, not only those at a switch
        # The directions state 0 was solved with: its own flows', or, at a jump moved to the
        # end of the run before, that run's starting ones (its end flows may have crossed
        # zero under the old density).
        solved = (self.jump_directions if strict and self.jump_directions is not None
                  else s0[f"{name}.q"])
        s1 = self._step(s0, d0, d1, h)
        if self.frozen is not None:
            return s1
        for _ in range(_DIRECTION_PASSES):
            q0, q1 = solved, s1[f"{name}.q"]
            if _same_directions(self.closure, q0, q1,
                                at_switch=None if strict else SWITCH_SKIP * SWITCH_FLOW):
                return s1
            # The flows jump there, and with them the volumes' storage rate: the BDF2
            # history before the jump does not carry over (`restart`).
            self.restart()
            s0 = self._start_with_directions(s0, d0, q1)
            solved = s0[f"{name}.q"]
            s1 = self._step(s0, d0, d1, h)
            strict = True  # state 0 now has the end's directions: they must stay
        if _same_directions(self.closure, s0[f"{name}.q"], s1[f"{name}.q"]):
            return s1
        _unsettled(self.closure, s0[f"{name}.q"], s1[f"{name}.q"], f"a step of {h!r} s")

    def _step(self, s0: State, d0: Drivers, d1: Drivers, h: float) -> State:
        air, name = self.air, self.air_name
        control = self.control
        # Every step-mean driver (`"<...>.sources"`: the sources and the air drawn out of a
        # zone, `assemble.EXTRACTION`) is the mean over THIS step, at both ends: `d0` at a grid
        # time holds the previous step's means, which the closure's state-dependent sinks at
        # state 0 would otherwise read.
        d0 = {**d0, **{k: v for k, v in d1.items() if k.endswith(".sources")}}
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

        # The last solved flows: the closure's view of the flows at state 1 (the directions of
        # an `actual` column, a junction's upwind end), from state 0's until the first solve.
        q_last = [q0]

        def G(z: Tensor) -> tuple[Tensor, State]:
            s = self._unpack(z, s0)
            if self.closure.heads or self.closure.series is not None:
                s[f"{name}.q"] = q_last[0] if self.frozen is None else self.frozen
            drv = dict(d1)
            drv.update(self.closure(s, drv))
            if store is not None:
                drv.update(air_drv)
            phi, q = self._solve_air(drv, s_end, s[f"{name}.phi"][..., air.interior], h)
            s[f"{name}.phi"], s[f"{name}.q"] = phi, q
            q_last[0] = q
            written1 = self.closure(s, d1)  # at state 1 (the closure reads, never writes, d1)
            extra1 = self._extra_sources(s, d1, written1)
            qm = 0.5 * (q0 + q)
            # Per end of the step: the storage capacity and extra sources (`StorageClosure.terms`)
            # and the closure's state-dependent sources.
            cap0: dict[str, Tensor] = {}
            cap1: dict[str, Tensor] = {}
            ext0, ext1 = dict(extra0), dict(extra1)
            if store is not None:
                drv1 = dict(d1)
                drv1.update(written1)
                now1 = store(s, drv1)["air.storage"]
                w_mass = store.rate_from_mass(now1, s0["air.storage"], h)
                net_mean = store.net(qm, s_air)
                t0 = self._storage_terms(s0, drv0, q0, net_mean, s_air, w_mass, net_mean)
                t1 = self._storage_terms(s, drv1, q, net_mean, s_air, w_mass, net_mean)
                for n in self.layers:
                    if f"{n}.capacity" in t0:
                        cap0[n], cap1[n] = t0[f"{n}.capacity"], t1[f"{n}.capacity"]
                        ext0[n] = ext0.get(n, 0.0) + t0[f"{n}.extra"]
                        ext1[n] = ext1.get(n, 0.0) + t1[f"{n}.extra"]
                s["air.storage"] = now1
            removal0 = removal1 = None
            if control is not None and control.heat:
                # The heat flows' affine form at each end, on the step's pieces (class
                # docstring).
                pieces = control.pieces_between(s0, d0, s, d1)
                A0, B0 = control.affine(s0, d0, pieces)
                A1, B1 = control.affine(s, d1, pieces)
                removal0 = control.removal(B0, cap0.get("thermal"))
                removal1 = control.removal(B1, cap1.get("thermal"))
                ext0["thermal"] = ext0.get("thermal", 0.0) + A0
                ext1["thermal"] = ext1.get("thermal", 0.0) + A1
            for n, layer in self.layers.items():
                x0 = s0[f"{n}.x"]
                src = d1.get(f"{n}.sources")
                if src is None:
                    src = self._zero_sources(n, x0)
                xb0, xb1 = d0[f"{n}.x_boundary"], d1[f"{n}.x_boundary"]
                rem0, rem1 = (removal0, removal1) if n == "thermal" else (None, None)
                # A removal rate given as the driver "<layer>.removal" adds to the feedback
                # network's, as `FeedbackHeat` adds its own to it in a `Model.step`.
                rem0 = _plus(d0.get(f"{n}.removal"), rem0)
                rem1 = _plus(d1.get(f"{n}.removal"), rem1)
                if layer.scheme == "exact":
                    # Two halves, the first with state 0's flows, capacity and sources,
                    # the second with state 1's; the boundary values through both ends,
                    # with their rates where they vary (class docstring).
                    r0 = d0.get(boundary_rate_key(f"{n}.x_boundary", "right"))
                    r1 = d1.get(boundary_rate_key(f"{n}.x_boundary", "left"))
                    if r0 is None or r1 is None:
                        xbm, rates0, rates1 = 0.5 * (xb0 + xb1), None, None
                    else:
                        xbm, rm = _hermite(xb0, xb1, r0, r1, h, 0.5)
                        rates0, rates1 = (r0, rm), (rm, r1)
                    xm = layer.step(x0, self._flows(layer, q0),
                                    src + ext0.get(n, 0.0), xb0, 0.5 * h,
                                    capacity=cap0.get(n), x_boundary_end=xbm,
                                    x_boundary_rates=rates0, removal=rem0)
                    s[f"{n}.x"] = layer.step(xm, self._flows(layer, q),
                                             src + ext1.get(n, 0.0), xbm, 0.5 * h,
                                             capacity=cap1.get(n), x_boundary_end=xb1,
                                             x_boundary_rates=rates1, removal=rem1)
                else:
                    cap = (0.5 * (cap0[n] + cap1[n])) if n in cap0 else None
                    if n in ext0 or n in ext1:
                        src = src + 0.5 * (ext0.get(n, 0.0) + ext1.get(n, 0.0))
                    removal = None if rem0 is None else 0.5 * (rem0 + rem1)
                    s[f"{n}.x"] = layer.step(x0, self._flows(layer, qm), src,
                                             0.5 * (xb0 + xb1), h, capacity=cap,
                                             removal=removal)
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
