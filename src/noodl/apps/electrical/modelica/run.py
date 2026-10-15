"""Run a circuit `read_modelica_circuit` built, over its step grid.

`simulate_circuit(model, state, drivers, names)` returns, at the output times, the layers'
states (`"electrical.phi"`, `"electrical.q"`, and the thermal ones when the circuit has a
thermal network) stacked over time, plus `"time"`.

An algebraic circuit (no capacitor, no inductor) is solved with `model.steady` at each
requested time, warm-started from the previous solution. A dynamic one is stepped over the
step grid `names.grid` from its first time: backward Euler on the steps that start at a
restart (`names.restarts`: the first step and every step starting at a signal jump, which
read only the step-start capacitor voltages and inductor currents) and trapezoidal
otherwise; each step reads its drivers at its END time, so the step ending at a signal
event uses the signal's left limit there. Row 0 of a dynamic run is the start state itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from noodl.apps.building_physics.modelica.run import step_drivers
from noodl.apps.electrical.modelica.assemble import THERMAL, CircuitNames
from noodl.model import Model, State

Tensor = torch.Tensor
F64 = torch.float64

# Newton tolerances of the steady solves (in the units of the residual, A or W): converged
# when the residual's max-norm is within STEADY_ATOL + STEADY_RTOL times the first one's.
# The layer's default, sqrt(eps) = 1.5e-8 for both, leaves MBL's ACACTransformer example up
# to 29 times the comparison bound 1e-9 |x| + 1e-12 from OpenModelica (VariableLoad 0.012 of
# it); at these values the eight algebraic examples without a heat port stay within 0.004 of
# it (measured). An absolute 1e-14 is below the round-off floor of the residual: Newton
# stalls at 3.4e-13 on ACDCConverter, at 1.1e-13 on the balanced ACLine.
STEADY_ATOL = 1e-12
STEADY_RTOL = 1e-14
# With a thermal network the same absolute tolerance also applies to its heat balance (W),
# whose round-off floor is eps G T: Newton stalls at 1.8e-12 W on MSL's Resistor example
# (G = 50 W/K at 300 K). 1e-11 keeps that example within 0.03 of the comparison bound.
THERMAL_STEADY_ATOL = 1e-11


def _grid_index(grid: Tensor, times: Tensor) -> list[int]:
    """The position of every entry of `times` on the step grid; raises `ValueError` for a
    time off the grid or a non-increasing one."""
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    if bool(off.any()):
        k = int(torch.nonzero(off)[0])
        raise ValueError(
            f"simulate_circuit: times[{k}] = {float(times[k])!r} is not a grid time of the "
            f"circuit ({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    out = idx.tolist()
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("simulate_circuit: times must be increasing")
    return out


def simulate_circuit(model: Model, state: State, drivers: Mapping[str, Tensor],
                     names: CircuitNames, *, times: Sequence[float] | Tensor | None = None,
                     **solve_kwargs) -> dict[str, Tensor]:
    """The circuit's history at `times` (default `names.times`; each must be a time of
    `names.grid`, increasing; a single time may be a scalar): `"time"` `(T,)` (the grid times
    found, a new tensor) and every `".phi"` / `".q"` state entry the model returns, stacked
    on dim 0.

    `solve_kwargs` (Newton's `atol`, `rtol`, `max_iter`, ...) are passed to every
    `model.steady` / `model.step`. The steady solves of an algebraic circuit default to
    `atol=STEADY_ATOL, rtol=STEADY_RTOL` (round-off; `atol=THERMAL_STEADY_ATOL` when the
    circuit has a thermal network); the steps of a dynamic one to the layer's own
    tolerances.
    """
    grid = names.grid
    if times is None:
        times = names.times
    else:
        times = torch.atleast_1d(torch.as_tensor(times, dtype=F64))
    if times.numel() == 0:
        raise ValueError("simulate_circuit: no times asked for")
    want = _grid_index(grid, times)
    rows: list[State] = []
    if not names.dynamic:
        atol = THERMAL_STEADY_ATOL if f"{THERMAL}.phi" in state else STEADY_ATOL
        kwargs = {"atol": atol, "rtol": STEADY_RTOL, **solve_kwargs}
        for k in want:
            t = float(grid[k])
            state = model.steady(state, step_drivers(drivers, grid, t), t=t, **kwargs)
            rows.append(state)
    else:
        wanted = set(want)
        if want[0] == 0:
            rows.append(state)
        for k in range(want[-1]):
            t0, t1 = float(grid[k]), float(grid[k + 1])
            scheme = "backward_euler" if k in names.restarts else "trapezoidal"
            state = model.step(state, step_drivers(drivers, grid, t1), dt=t1 - t0, t=t0,
                               integration=scheme, **solve_kwargs)
            if k + 1 in wanted:
                rows.append(state)
    keys = [key for key in rows[0] if key.endswith((".phi", ".q"))]
    out = {key: torch.stack([row[key] for row in rows]) for key in keys}
    out["time"] = grid[want].clone()
    return out
