"""The time-march loop behind `Model.march`: one `Model.step` per interval of a time grid,
backward Euler on the restart steps for the layers that step inside their solve, the layers'
own schemes elsewhere."""

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
    negative = sorted(k for k in restart_steps if k < 0)
    if negative:
        raise ValueError(f"Model.march: restarts must be step indices >= 0, got {negative}")
    rows = list(range(len(grid))) if record is None else [int(k) for k in record]
    if not rows or any(not 0 <= k < len(grid) for k in rows) or rows != sorted(set(rows)):
        raise ValueError(f"Model.march: record must be increasing indices into the {len(grid)} "
                         f"times, got {rows}")
    kept = set(rows)
    caller_options = {k: dict(v) for k, v in (layer_options or {}).items()}
    # The layers a restart step takes by backward Euler: the Kirchhoff layers that step
    # something inside their solve (storage at interior nodes, dynamic laws) and the
    # trapezoidal transport layers. A Kirchhoff layer with storage at boundary nodes steps it
    # by forward Euler only, and keeps its scheme.
    restarted = [name for name, layer in model.kirchhoff.items() if layer.dynamic]
    restarted += [name for name, layer in model.transport.items()
                  if layer.scheme == "trapezoidal"]
    restart_options = {**caller_options,
                       **{name: {**caller_options.get(name, {}), "scheme": "backward_euler"}
                          for name in restarted}}
    schemes: list[str | None] = []
    history = []
    if 0 in kept:
        history.append(state)
    for k in range(rows[-1]):
        t0, t1 = grid[k], grid[k + 1]
        d = drivers if drivers_at is None else drivers_at(t0, t1)
        if k in restart_steps:
            options = restart_options
            schemes.append("backward_euler")
        else:
            options = caller_options
            schemes.append(solve_kwargs.get("scheme"))
        state = model.step(state, d, dt=t1 - t0, t=t0, layer_options=options or None,
                           **solve_kwargs)
        if k + 1 in kept:
            history.append(state)
    if diagnostics is not None:
        diagnostics["schemes"] = schemes
    out = {key: torch.stack(torch.broadcast_tensors(*column))
           for key, column in _columns(model, history).items()}
    out["time"] = torch.tensor([grid[k] for k in rows], dtype=torch.float64)
    return out


def _columns(model: Model, history: list) -> dict[str, list]:
    """Every recorded key's value in every row. Recorded are the keys of the first row (the
    start state, or the first row kept) that end in `RECORDED_SUFFIXES` or are closure-carried
    state, and the `"<layer>.storage_flow"` of every Kirchhoff layer with storage, which the
    start state need not carry: a row without it takes the layer's own at that row's
    potentials and flows with no step (zero for storage at interior nodes, what the flows
    deliver for storage at boundary nodes). A key the first row does not carry (a potential
    the caller left to a cold start) has no value there and is not recorded. A recorded key
    missing from a later row is refused by name."""
    closure_keys = set(model.closure_state_keys)
    flows = {f"{name}.storage_flow": name for name, layer in model.kirchhoff.items()
             if layer.storage}
    keys = [key for key in history[0]
            if key.endswith(RECORDED_SUFFIXES) or key in closure_keys]
    keys += [key for key in flows if key not in keys and f"{flows[key]}.phi" in history[0]]
    columns: dict[str, list] = {}
    for key in keys:
        column = []
        for i, row in enumerate(history):
            if key in row:
                column.append(row[key])
            elif key in flows:
                name = flows[key]
                if model.kirchhoff[name].holds_storage and f"{name}.q" not in row:
                    raise ValueError(
                        f"Model.march records {key!r} from the start state; on a layer "
                        f"that holds its storage at boundary nodes that needs every edge's "
                        f"flow too: pass {name + '.q'!r} in the start state"
                    )
                column.append(model.kirchhoff[name].storage_flows(
                    row[f"{name}.phi"], {}, dt=None, q=row.get(f"{name}.q")))
            else:
                raise ValueError(
                    f"Model.march: recorded row {i} has no {key!r}, which the first row "
                    f"carries"
                )
        columns[key] = column
    return columns
