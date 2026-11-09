"""The time-march loop behind `Model.march`: one `Model.step` per interval of a time grid,
backward Euler on the restart steps, the layers' own schemes elsewhere."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from noodl.model import Model

#: The state keys `march` keeps: potentials, flows, stored amounts, storage flows, capacities.
RECORDED_SUFFIXES = (".phi", ".q", ".x", ".storage_flow", ".capacity")


def march(model: Model, state, drivers, times, *, restarts: Collection[int] = (),
          drivers_at: Callable[[float, float], Mapping] | None = None,
          record: Collection[int] | None = None, diagnostics: dict | None = None,
          layer_options: Mapping[str, Mapping] | None = None, **solve_kwargs) -> dict:
    """Step `model` through `times`; see `Model.march`."""
    grid = [float(t) for t in (times.tolist() if isinstance(times, torch.Tensor) else times)]
    if not grid:
        raise ValueError("Model.march: times is empty; give the start time and the step ends")
    if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError(f"Model.march: times must be strictly increasing, got {grid}")
    # a restart index past the last step is ignored (one restart set serves any prefix of a grid)
    restart_steps = {int(k) for k in restarts}
    rows = list(range(len(grid))) if record is None else [int(k) for k in record]
    if not rows or any(not 0 <= k < len(grid) for k in rows) or rows != sorted(set(rows)):
        raise ValueError(f"Model.march: record must be increasing indices into the {len(grid)} "
                         f"times, got {rows}")
    kept = set(rows)
    caller_options = {k: dict(v) for k, v in (layer_options or {}).items()}
    # transport layers that a restart step must take by backward Euler
    trapezoidal = [name for name, layer in model.transport.items()
                   if layer.scheme == "trapezoidal"]
    restart_options = {**caller_options,
                       **{name: {**caller_options.get(name, {}), "scheme": "backward_euler"}
                          for name in trapezoidal}}
    schemes: list[str | None] = []
    history = []
    if 0 in kept:
        history.append(state)
    for k in range(rows[-1]):
        t0, t1 = grid[k], grid[k + 1]
        d = drivers if drivers_at is None else drivers_at(t0, t1)
        if k in restart_steps:
            kwargs = {**solve_kwargs, "scheme": "backward_euler"}
            options = restart_options
        else:
            kwargs, options = solve_kwargs, caller_options
        schemes.append(kwargs.get("scheme"))
        state = model.step(state, d, dt=t1 - t0, t=t0, layer_options=options or None, **kwargs)
        if k + 1 in kept:
            history.append(state)
    if diagnostics is not None:
        diagnostics["schemes"] = schemes
    keys = [key for key in history[0] if key.endswith(RECORDED_SUFFIXES)]
    out = {key: torch.stack(torch.broadcast_tensors(*[row[key] for row in history]))
           for key in keys}
    out["time"] = torch.tensor([grid[k] for k in rows], dtype=torch.float64)
    return out
