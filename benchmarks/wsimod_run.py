"""WSIMOD application demonstration: time per day and autograd graph size of `WsimodDay`.

Runs WSIMOD's two demos, `quickstart` and `oxford` (`tests/data/wsimod/`, quantity-only,
`pollutants=False`), for 60 days at batch sizes 1, 10 and 100, and then `oxford` once more on
`config.yml` with `pollutants=True`, where every ported process is live: the surface loads,
decay, treatment and the soil temperature (oxford has no `River`). A last row, `growing_spring`
(built from `tests.verification._wsimod_cases.CASES`, with the eight nutrient pollutants), runs
the growing surface with its nutrient pool for 120 days, which end before its sowing on run day
182 (the pool runs every day, the crop not yet). The script imports the test package for that
case (`tests.verification._wsimod_cases`), so it is a developer script: run it from the
repository root with the root on the path.
The batch is the leading dimension of `wsimod.Arc.capacity` (every instance has the
same value, so every instance takes the same branches: the cost measured is the batch's tensor
width, not divergence). Two rows per configuration:

- **no grad**: the run under `torch.no_grad()` (what `wsimod_run` does when nothing requires
  grad), seconds per day;
- **grad**: the same run with `wsimod.Arc.capacity` requiring grad (W9), seconds per day, and
  the number of nodes of the autograd graph behind the last day's storage (counted through
  `grad_fn.next_functions`). The graph grows linearly with the days (hence the advice to fit in
  windows) and is independent of the batch size, which only widens each tensor.

The per-instance Python math kept for bitwise equality with WSIMOD (`pow_half`, `exp_python`,
`pow_python`: a Python `float ** 0.5`, `math.exp` and `**` per instance) is a loop over the
batch. On the demos it is not what the batch-100 rows pay for (neither demo has a `River`, and
the soil's `** 0.5` is a small part of a day), so the batch rows stay close to batch 1; a model
with many rivers or time-area groundwaters would show it. A growing surface adds its own: about 30
per-instance Python calls (`pow_python`, `exp_python`, `log_python`, `log10_python`,
`sin_python`) per growing surface per day, plus 3 per Newton iteration of its adsorption, so the
`growing_spring` rows are where a batch pays for them.

Each time is a single shot (no repeats, no warm-up). Prints and asserts nothing about the
numbers (times are measured, never asserted).

Run, from the repository root: `PYTHONPATH=src:. .venv/Scripts/python benchmarks/wsimod_run.py`
(on Windows `src;.`)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from noodl.apps.wsimod import build_model, initial_drivers, initial_state, read_wsimod_config
from tests.verification._wsimod_cases import CASES

F64 = torch.float64
DAYS = 60
GROWING_DAYS = 120  # growing_spring is sown on run day 90
BATCHES = (1, 10, 100)
DEMOS = ("quickstart", "oxford")
DATA = Path(__file__).resolve().parent.parent / "tests" / "data" / "wsimod"
DAY = 86400.0


def graph_size(root: torch.Tensor) -> int:
    """Number of distinct autograd nodes behind `root`."""
    seen: set[int] = set()
    stack = [root.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return len(seen)


def run(
    name: str,
    batch: int,
    *,
    grad: bool,
    pollutants: bool = False,
    config: str = "config.yml",
    days: int = DAYS,
) -> tuple[float, int]:
    """`(seconds per day, autograd nodes)` of a `days`-day run at `batch` instances. A `name` in
    `CASES` is that purpose-built case, otherwise the demo of that name."""
    if name in CASES:
        network = read_wsimod_config(CASES[name].build())
    else:
        network = read_wsimod_config(DATA / name, config_name=config)
    model, _, _ = build_model(network, pollutants=pollutants)
    state = initial_state(model)
    base = initial_drivers(model)["wsimod.Arc.capacity"]
    capacity = base.expand(batch, *base.shape).clone().requires_grad_(grad)
    drivers = initial_drivers(model, values={"wsimod.Arc.capacity": capacity})
    start = time.perf_counter()
    with torch.set_grad_enabled(grad):
        for k in range(days):
            state = model.step(state, drivers, dt=DAY, t=k * DAY)
    seconds = (time.perf_counter() - start) / days
    nodes = graph_size(state["wsimod.storage"]) if grad else 0
    return seconds, nodes


def main() -> None:
    print(f"{DAYS}-day runs; seconds per day, autograd nodes (grad rows)")
    print(f"{'demo':<22}{'batch':>6}{'no grad s/day':>15}{'grad s/day':>13}{'graph nodes':>13}")
    rows = [(name, name, False, "config.yml", DAYS) for name in DEMOS]
    rows.append(("oxford", "oxford+pollutants", True, "config.yml", DAYS))
    rows.append(("growing_spring", "growing_spring", True, "config.yml", GROWING_DAYS))
    for name, label, pollutants, config, days in rows:
        for batch in BATCHES:
            plain, _ = run(name, batch, grad=False, pollutants=pollutants, config=config, days=days)
            taped, nodes = run(
                name, batch, grad=True, pollutants=pollutants, config=config, days=days
            )
            print(f"{label:<22}{batch:>6}{plain:>15.4f}{taped:>13.4f}{nodes:>13}")


if __name__ == "__main__":
    main()
