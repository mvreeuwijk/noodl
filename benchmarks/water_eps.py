"""Milestone 4 demonstration: Net1's 24 h extended period, over 8 demand multipliers.

Runs EPANET's own Example Network 1 for a full day with its two tank-level pump controls,
using the EVENT-SHORTENED hydraulic step (EPANET 2.2 Manual section 13.1 item 17, p.113),
run SEQUENTIALLY over 8 demand multipliers -- one model per multiplier, not a batched leading
dimension (N15). `TankLevels.event_step` shortens the step to each multiplier's OWN next
control crossing, so the eight instances take a different number of sub-steps and land on
different report times mid-run; a genuinely batched rollout would need a per-instance step
(the sub-stepping loop below only ever advances one scalar `moment` at a time) or a
global-minimum step shared by all eight, which oversamples the seven instances that did not
need shortening on that particular sub-step -- undesirable and, either way, not "straightforward"
enough to fold into a fix-wave item, so this docstring records reality instead of the plan's
original "batched" wording. Prints the wall time, the tank-level trajectory of instance 0 and
the number of hydraulic sub-steps the event shortening cost.

Run: `.venv/Scripts/python benchmarks/water_eps.py`
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, tank_inflow, water_steady

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "water"
MULTIPLIERS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)


def _pattern_factor(net, seconds: float) -> float:
    if net.demand_pattern is None or net.demand_pattern not in net.patterns:
        return 1.0
    values = net.patterns[net.demand_pattern]
    return values[int(seconds // net.pattern_timestep) % len(values)]


def _run(net, multiplier: float) -> tuple[list[float], int]:
    """One instance's 24 h trajectory and its sub-step count."""
    model, state, drivers = build_model(net)
    closure = model.tank_closure
    base = drivers["water.sources"].clone() * multiplier
    levels = [float(state["water.tank_level"][0])]
    moment = 0.0
    sub_steps = 0
    while moment < net.duration - 1e-9:
        report_end = moment + net.report_timestep
        while moment < report_end - 1e-9:
            step_drivers = dict(drivers)
            step_drivers["water.sources"] = base * _pattern_factor(net, moment)
            solved = water_steady(model, state, step_drivers)
            inflow = tank_inflow(model, solved)
            rate = inflow / closure.area
            step = closure.event_step(
                state["water.tank_level"], rate, report_end - moment
            )
            state = dict(solved)
            state["water.tank_level"] = closure.advance(
                state["water.tank_level"], inflow, step
            )
            moment += step
            sub_steps += 1
        levels.append(float(state["water.tank_level"][0]))
    return levels, sub_steps


def main() -> None:
    net = read_epanet_inp(DATA / "Net1.inp")
    started = time.perf_counter()
    trajectories = []
    total_sub_steps = 0
    for multiplier in MULTIPLIERS:
        levels, sub_steps = _run(net, multiplier)
        trajectories.append(levels)
        total_sub_steps += sub_steps
    elapsed = time.perf_counter() - started
    print(
        f"water_eps: Net1, {int(net.duration / 3600)} h, "
        f"{len(MULTIPLIERS)} sequential demand multipliers, {total_sub_steps} hydraulic "
        f"sub-steps in {elapsed:.2f} s"
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
