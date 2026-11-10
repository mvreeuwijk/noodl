"""Time of the electrical transient march: `Model.march` against the hand-rolled loop it
replaced (the same `Model.step` call sequence), on a two-stage RC ladder, 400 steps.

Both loops are run once to warm up, then timed in interleaved rounds (the order swapped every
round), so a drift of the machine over the run falls on both alike.

    PYTHONPATH=src python benchmarks/march_time.py [rounds]
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

from noodl.apps.electrical import (
    Capacitor,
    Circuit,
    Resistor,
    VoltageSource,
    build_model,
    initial_state,
)

ROUNDS, STEPS, DT = 8, 400, 1e-5


def _old(model, state, drivers, grid):
    for k in range(len(grid) - 1):
        rule = "backward_euler" if k == 0 else "trapezoidal"
        state = model.step(state, drivers, grid[k + 1] - grid[k], t=grid[k], scheme=rule)
    return state


def _new(model, state, drivers, grid):
    return model.march(state, drivers, grid, restarts={0}, scheme="trapezoidal")


def main(rounds: int = ROUNDS) -> None:
    torch.set_default_dtype(torch.float64)
    model, _, drivers = build_model(Circuit([
        VoltageSource("V1", "in", "0", 1.0), Resistor("R1", "in", "a", 1e3),
        Capacitor("C1", "a", "0", 1e-6), Resistor("R2", "a", "b", 1e3),
        Capacitor("C2", "b", "0", 1e-6)]))
    state = initial_state(model, drivers)
    grid = [k * DT for k in range(STEPS + 1)]
    loops = {"hand loop": _old, "Model.march": _new}
    for run in loops.values():
        run(model, state, drivers, grid)            # warm-up, not timed
    times: dict[str, list[float]] = {name: [] for name in loops}
    order = list(loops)
    for _ in range(rounds):
        for name in order:
            t0 = time.perf_counter()
            loops[name](model, state, drivers, grid)
            times[name].append(time.perf_counter() - t0)
        order.reverse()
    for name, t in times.items():
        print(f"{name:12s} median {statistics.median(t):.3f} s  min {min(t):.3f} s  "
              f"({1e3 * statistics.median(t) / STEPS:.2f} ms/step, {rounds} rounds)")
    ratio = statistics.median(times["Model.march"]) / statistics.median(times["hand loop"])
    print(f"Model.march / hand loop: {ratio:.3f} (median)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else ROUNDS)
