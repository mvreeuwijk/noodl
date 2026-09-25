"""Run a model `read_modelica` built over its experiment grid.

`simulate(model, state, drivers, times)` steps the model from `times[0]` through every later
grid time, with each step's time-varying drivers taken at the step's END time (the implicit
end-of-interval convention of the transport schemes), except sources, which are the MEAN of
their value over the step instead (`assemble.py`, "Sources"), and returns the time histories
MBL's reference CSV holds: every air-layer edge flow, every node's absolute pressure,
temperature, water mass fraction and trace-substance mass fractions. A model without zones (no
transport layer) is algebraic in its boundary values and is solved with `model.steady` at every
grid time instead. Row 0 is the initial state with its quasi-steady flows.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.modelica.assemble import _MBLClosure
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


def _require_consecutive(drivers: Mapping[str, Tensor], grid: Tensor, times: Tensor) -> None:
    """Refuse `times` that skip grid intervals when a source driver is a step-mean series."""
    sources = sorted(k[len(_SERIES):] for k in drivers
                     if k.startswith(_SERIES) and k.endswith(".sources"))
    if not sources or times.numel() < 2:
        return
    grid = torch.as_tensor(grid, dtype=F64)
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    gaps = (idx[1:] - idx[:-1]) != 1
    if bool(off.any()):
        k = int(torch.nonzero(off)[0])
        why = "is not a grid time"
    elif bool(gaps.any()):
        k = int(torch.nonzero(gaps)[0]) + 1
        why = f"is not the grid time after {float(times[k - 1])!r}"
    else:
        return
    raise ValueError(
        f"simulate: time-varying sources ({', '.join(sources)}) are step means over the "
        f"driver grid's intervals, so `times` must be consecutive grid times; times[{k}] = "
        f"{float(times[k])!r} {why} (grid interval {float(grid[1] - grid[0])!r} s)"
    )


def _closure(model: Model) -> _MBLClosure:
    for c in model.closures:
        if isinstance(c, _MBLClosure):
            return c
    raise TypeError("simulate: the model was not built by read_modelica (no MBL closure)")


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


def simulate(model: Model, state: State, drivers: Drivers, times,
             **step_kwargs) -> dict[str, Tensor]:
    """Time histories over `times` (a subset of the experiment grid, increasing).

    Returns `"time"` `(N,)`, `"air.q"` `(N, b)` in the air layer's edge order (see
    `ModelicaNames.edges`), `"air.phi"` `(N, n)` gauge pressure (relative to
    `ModelicaNames.p_ref`), `"p"` `(N, n)` absolute
    pressure (Pa), `"T"` `(N, n)` (K), `"X_w"` `(N, n)`, and `"C"` `(N, n, K)` for the species
    layer's mass fractions (water last when carried) when the model has one. `step_kwargs`
    reach `Model.step`/`Model.steady` (and so the airflow solves); the airflow Newton
    tolerances default to `atol=AIR_ATOL`, `rtol=AIR_RTOL`.

    When a source driver varies in time, `times` must be CONSECUTIVE grid times (any run of
    the grid, e.g. a prefix): each series row of a source is its mean over the one grid
    interval ending there (`assemble` module docstring, "Sources"), so a step spanning
    several grid intervals would inject only the last interval's amount. Anything else
    raises `ValueError`.
    """
    step_kwargs = {"atol": AIR_ATOL, "rtol": AIR_RTOL, **step_kwargs}
    times = torch.as_tensor(times, dtype=F64)
    grid = drivers.get("series:time", times)
    _require_consecutive(drivers, grid, times)
    closure = _closure(model)
    dynamic = bool(model.transport)
    rows: dict[str, list[Tensor]] = {"air.q": [], "air.phi": [], "p": [], "T": [], "X_w": []}
    if closure.sp_interior is not None:
        rows["C"] = []
    for k in range(times.numel()):
        t = float(times[k])
        d = step_drivers(drivers, grid, t)
        if not dynamic:
            state = model.steady(state, d, **step_kwargs)
        elif k == 0:
            state = _solve_air(model, state, d, **step_kwargs)
        else:
            state = model.step(state, d, t - float(times[k - 1]), t=float(times[k - 1]),
                               **step_kwargs)
        extra = closure(state, d)
        rows["air.q"].append(state["air.q"])
        rows["air.phi"].append(state["air.phi"])
        rows["p"].append(closure.pressures(state, d))
        rows["T"].append(extra["T"])
        rows["X_w"].append(extra["X_w"] if "X_w" in extra else d["X_w"])
        if "C" in rows:
            rows["C"].append(closure.species(state, d))
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["time"] = times.clone()
    return out
