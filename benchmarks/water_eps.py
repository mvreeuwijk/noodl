"""Water demonstration: Net1's 24 h extended period, over 8 demand multipliers.

Runs EPANET's own Example Network 1 for a full day with its two tank-level pump controls,
using the EVENT-SHORTENED hydraulic step (EPANET 2.2 Manual section 13.1 item 17, p.113),
run SEQUENTIALLY over 8 demand multipliers -- one model per multiplier, not a batched leading
dimension. `water_event_step` (one hydraulic solve per sub-step, the tank moved by
forward Euler afterwards) shortens the step to each multiplier's OWN next
control crossing, so the eight instances take a different number of sub-steps and land on
different report times mid-run; a genuinely batched rollout would need a per-instance step
(the sub-stepping loop below only ever advances one scalar `moment` at a time) or a
global-minimum step shared by all eight, which oversamples the seven instances that did not
need shortening on that particular sub-step -- undesirable either way, so the multipliers run
sequentially. Prints the wall time, the tank-level trajectory of instance 0 and
the number of hydraulic sub-steps the event shortening cost, and the Newton iterations of the
hardest sub-step (the step where a control closes the pump and leaves pipe 10 dead-ended).

`--pipe-form flow` runs the pipes in their resistance form (`build_model(pipe_form="flow")`,
solved by Modified Nodal Analysis) instead of the default potential form.

Run: `.venv/Scripts/python benchmarks/water_eps.py [--pipe-form potential|flow]`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, tank_levels, water_event_step

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "water"
MULTIPLIERS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)


def _pattern_factor(net, seconds: float) -> float:
    if net.demand_pattern is None or net.demand_pattern not in net.patterns:
        return 1.0
    values = net.patterns[net.demand_pattern]
    return values[int(seconds // net.pattern_timestep) % len(values)]


def _run(net, multiplier: float, pipe_form: str = "potential") -> tuple[list[float], int, int]:
    """One instance's 24 h trajectory, its sub-step count and its most Newton iterations."""
    model, state, drivers = build_model(net, pipe_form=pipe_form)
    base = drivers["water.sources"].clone() * multiplier
    levels = [float(tank_levels(model, state)[0])]
    moment = 0.0
    sub_steps = 0
    most = 0
    while moment < net.duration - 1e-9:
        report_end = moment + net.report_timestep
        while moment < report_end - 1e-9:
            step_drivers = dict(drivers)
            step_drivers["water.sources"] = base * _pattern_factor(net, moment)
            diagnostics: dict = {}
            state, step = water_event_step(model, state, step_drivers, report_end - moment,
                                           diagnostics=diagnostics)
            most = max(most, diagnostics["layers"]["water"]["newton_iterations"])
            moment += step
            sub_steps += 1
        levels.append(float(tank_levels(model, state)[0]))
    return levels, sub_steps, most


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Net1's 24 h extended period")
    parser.add_argument("--pipe-form", choices=("potential", "flow"), default="potential")
    args = parser.parse_args(argv)
    net = read_epanet_inp(DATA / "Net1.inp")
    started = time.perf_counter()
    trajectories = []
    total_sub_steps = 0
    most = 0
    for multiplier in MULTIPLIERS:
        levels, sub_steps, worst = _run(net, multiplier, args.pipe_form)
        trajectories.append(levels)
        total_sub_steps += sub_steps
        most = max(most, worst)
    elapsed = time.perf_counter() - started
    print(
        f"water_eps: Net1, {int(net.duration / 3600)} h, pipe_form={args.pipe_form}, "
        f"{len(MULTIPLIERS)} sequential demand multipliers, {total_sub_steps} hydraulic "
        f"sub-steps in {elapsed:.2f} s; at most {most} Newton iterations in one sub-step"
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
