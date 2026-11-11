"""Water demonstration: Net1's 24 h extended period, over 8 demand multipliers.

Runs EPANET's own Example Network 1 for a full day with its two tank-level pump controls
through `water_run` (EPANET 2.2's step rule: the hydraulic step shortened to the next pattern
period, report time, tank fill or drain and control crossing, `hydraul.c`, `timestep`), run
SEQUENTIALLY over 8 demand multipliers -- one run per multiplier, not a batched leading
dimension. Each multiplier's pump switches at its own times, so its own run takes only its
own step boundaries; a batched run (`water_run` with a leading dimension in the drivers)
would share the shortest step of all eight, which oversamples the instances that did not
need shortening and changes their explicit-Euler trajectories by millimetres
(`tests/apps/water/test_simulate.py::test_batched_run_equals_single_runs`), so the benchmark
keeps the sequential form it has always timed. Prints the wall time, the tank-level
trajectory of multiplier 1.0, the number of hydraulic steps the event shortening cost, and
the Newton iterations of the hardest step (the step where a control closes the pump and
leaves pipe 10 dead-ended).

`--pipe-form flow` runs the pipes in their resistance form (`build_model(pipe_form="flow")`,
solved by Modified Nodal Analysis) instead of the default potential form. `--check-valve 21`
makes Net1's pipe 21 a check valve (CV); its water runs backwards for 11 of the 26 steps of
the plain run, so the check valve is closed for much of the day (flow form only).

Run: `.venv/Scripts/python benchmarks/water_eps.py [--pipe-form potential|flow]
[--check-valve PIPE]`
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.simulate import water_run

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "water"
MULTIPLIERS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)


def _run(net, multiplier: float, pipe_form: str = "potential") -> tuple[list[float], int, int]:
    """One instance's 24 h trajectory, its step count and its most Newton iterations."""
    model, _, drivers = build_model(net, pipe_form=pipe_form)
    drivers = {**drivers, "water.sources": drivers["water.sources"] * multiplier}
    diagnostics: dict = {}
    results = water_run(model, drivers=drivers, quality=False, diagnostics=diagnostics)
    steps = len(diagnostics["times"]) - 1
    return results.tank_level[:, 0].tolist(), steps, max(diagnostics["newton_iterations"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Net1's 24 h extended period")
    parser.add_argument("--pipe-form", choices=("potential", "flow"), default="potential")
    parser.add_argument("--check-valve", metavar="PIPE", default=None,
                        help="make this pipe a check valve (CV)")
    args = parser.parse_args(argv)
    net = read_epanet_inp(DATA / "Net1.inp")
    if args.check_valve is not None:
        net = dataclasses.replace(net, pipes=tuple(
            dataclasses.replace(p, status="CV") if p.name == args.check_valve else p
            for p in net.pipes))
    started = time.perf_counter()
    trajectories = []
    total_steps = 0
    most = 0
    for multiplier in MULTIPLIERS:
        levels, steps, worst = _run(net, multiplier, args.pipe_form)
        trajectories.append(levels)
        total_steps += steps
        most = max(most, worst)
    elapsed = time.perf_counter() - started
    print(
        f"water_eps: Net1, {int(net.duration / 3600)} h, pipe_form={args.pipe_form}, "
        f"{'check valve ' + args.check_valve + ', ' if args.check_valve else ''}"
        f"{len(MULTIPLIERS)} sequential demand multipliers, {total_steps} hydraulic "
        f"steps in {elapsed:.2f} s; at most {most} Newton iterations in one step"
    )
    print("tank 2 level (m) at each report step, demand multiplier 1.0:")
    baseline = trajectories[MULTIPLIERS.index(1.0)]
    for hour, level in enumerate(baseline):
        print(f"  t = {hour:2d} h   {level:9.6f}")
    final = [row[-1] for row in trajectories]
    print("final level (m) per multiplier:")
    for multiplier, level in zip(MULTIPLIERS, final, strict=True):
        print(f"  x{multiplier:.1f}   {level:9.6f}")


if __name__ == "__main__":
    main()
