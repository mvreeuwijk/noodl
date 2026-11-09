"""Time of the electrical transient march: `Model.march` against the hand-rolled loop it
replaced (the same `Model.step` call sequence), on a two-stage RC ladder, 400 steps.

    PYTHONPATH=src python benchmarks/march_time.py
"""

from __future__ import annotations

import statistics
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

RUNS, STEPS, DT = 5, 400, 1e-5


def _old(model, state, drivers, grid):
    for k in range(len(grid) - 1):
        rule = "backward_euler" if k == 0 else "trapezoidal"
        state = model.step(state, drivers, grid[k + 1] - grid[k], t=grid[k], scheme=rule)
    return state


def _new(model, state, drivers, grid):
    return model.march(state, drivers, grid, restarts={0}, scheme="trapezoidal")


def main() -> None:
    torch.set_default_dtype(torch.float64)
    model, _, drivers = build_model(Circuit([
        VoltageSource("V1", "in", "0", 1.0), Resistor("R1", "in", "a", 1e3),
        Capacitor("C1", "a", "0", 1e-6), Resistor("R2", "a", "b", 1e3),
        Capacitor("C2", "b", "0", 1e-6)]))
    state = initial_state(model, drivers)
    grid = [k * DT for k in range(STEPS + 1)]
    for name, run in (("hand loop", _old), ("Model.march", _new)):
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            run(model, state, drivers, grid)
            times.append(time.perf_counter() - t0)
        print(f"{name:12s} median {statistics.median(times):.3f} s  min {min(times):.3f} s  "
              f"({1e3 * statistics.median(times) / STEPS:.2f} ms/step)")


if __name__ == "__main__":
    main()
