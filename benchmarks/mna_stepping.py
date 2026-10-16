"""Where the time of a Modified Nodal Analysis (MNA) solve goes.

Three cases from the electrical tests:

- `RCModel`: MBL's `DC.Lines.Examples.RCModel`, a dynamic RC line, stepped over its grid
  (`read_modelica_circuit` + `simulate_circuit`);
- `HeatingRectifier`: MSL's `Analog.Examples.HeatingRectifier`, a diode rectifier with a
  heat-port thermal network, coupled electro-thermal steps;
- `case118`: MATPOWER's 118-bus AC power flow, one steady phasor solve.

Each case runs `--steps` steps (one solve for `case118`), warmed once, over `--rounds`
rounds; the median wall time per step is reported. In a separate, instrumented run the
time and the number of calls of the three parts of a Newton iteration are counted: the
Jacobian (`MnaSystem.operator`, the forward-mode colouring), the linear solve
(`solvers.select._sparse_direct`, SciPy's SuperLU) and the residual (`MnaSystem.residual`).
A third run under `cProfile` counts the calls that reach `torch._refs` / `torch._prims`,
PyTorch's Python fallbacks for operations without a native kernel (complex elementwise
ops among them).

Timings depend on the machine and its load, so they are recorded here and in
`mna_stepping_results.txt`, never asserted in a test.

    PYTHONPATH=src .venv/Scripts/python -m benchmarks.mna_stepping --quick
    PYTHONPATH=src .venv/Scripts/python -m benchmarks.mna_stepping \
        --out benchmarks/mna_stepping_results.txt
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import platform
import pstats
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

import noodl.solvers.select as select
from noodl.apps.electrical.modelica import read_modelica_circuit, simulate_circuit
from noodl.apps.electrical.power import (
    MatpowerCaseUnavailableError,
    build_power_flow,
    matpower_case,
    read_matpower,
    solve_power_flow,
)
from noodl.layers._mna import MnaSystem

ROOT = Path(__file__).resolve().parent.parent
CIRCUITS = ROOT / "tests" / "data" / "modelica_electrical"
DEFAULT_STEPS = 50
DEFAULT_ROUNDS = 3


# ------------------------------------------------------------------------- cases
def _circuit(name: str) -> Callable[[int], Callable[[], object]]:
    def make(steps: int):
        model, state, drivers, names = read_modelica_circuit(CIRCUITS / f"{name}.json")
        n = min(steps, names.grid.numel() - 1)
        t = float(names.grid[n])

        def run():
            with torch.no_grad():
                return simulate_circuit(model, state, drivers, names, times=t)

        run.steps = n
        return run
    return make


def _power_flow(case: str) -> Callable[[int], Callable[[], object]]:
    def make(steps: int):
        model, state, drivers = build_power_flow(read_matpower(matpower_case(case)))

        def run():
            with torch.no_grad():
                return solve_power_flow(model, state, drivers)

        run.steps = 1
        return run
    return make


CASES = {
    "RCModel": _circuit("DC.Lines.Examples.RCModel"),
    "HeatingRectifier": _circuit("Analog.Examples.HeatingRectifier"),
    "case118": _power_flow("case118"),
}


# ------------------------------------------------------------------- instrumentation
class _Counter:
    """Time and calls of the wrapped functions, by part."""

    def __init__(self) -> None:
        self.time: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, part: str, fn):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.time[part] = self.time.get(part, 0.0) + time.perf_counter() - t0
                self.calls[part] = self.calls.get(part, 0) + 1
        return timed


@contextlib.contextmanager
def _instrumented(counter: _Counter):
    saved = (MnaSystem.operator, MnaSystem.residual, select._sparse_direct)
    MnaSystem.operator = counter.wrap("jacobian", MnaSystem.operator)
    MnaSystem.residual = counter.wrap("residual", MnaSystem.residual)
    select._sparse_direct = counter.wrap("lu", select._sparse_direct)
    try:
        yield
    finally:
        MnaSystem.operator, MnaSystem.residual, select._sparse_direct = saved


def _refs_calls(run) -> dict[str, int]:
    """Calls into `torch._refs` / `torch._prims` during `run()`, by function."""
    profile = cProfile.Profile()
    profile.enable()
    run()
    profile.disable()
    stats = pstats.Stats(profile).stats
    out: dict[str, int] = {}
    for (filename, _, name), (_, ncalls, *_rest) in stats.items():
        path = filename.replace("\\", "/")
        if "/torch/_refs/" in path or "/torch/_prims" in path:
            key = f"{path.split('/torch/', 1)[1]}:{name}"
            out[key] = out.get(key, 0) + ncalls
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------- run
def measure(name: str, steps: int, rounds: int) -> dict:
    run = CASES[name](steps)
    run()  # warm: forward-mode rules, SciPy import, caches
    walls = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        run()
        walls.append(time.perf_counter() - t0)
    counter = _Counter()
    with _instrumented(counter):
        t0 = time.perf_counter()
        run()
        total = time.perf_counter() - t0
    return {
        "case": name,
        "steps": run.steps,
        "ms_per_step": 1e3 * statistics.median(walls) / run.steps,
        "spread_ms_per_step": [1e3 * min(walls) / run.steps, 1e3 * max(walls) / run.steps],
        "instrumented_ms_per_step": 1e3 * total / run.steps,
        "parts_ms_per_step": {k: 1e3 * v / run.steps for k, v in counter.time.items()},
        "calls_per_step": {k: v / run.steps for k, v in counter.calls.items()},
        "refs_calls": _refs_calls(run),
    }


def machine_block() -> dict:
    return {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def report(results: list[dict], machine: dict) -> str:
    lines = ["# MNA stepping", ""]
    lines += [f"- {k}: {v}" for k, v in machine.items()]
    lines += ["", "| case | steps | ms/step (median) | min-max | jacobian | lu | residual | "
              "jacobians/step | LUs/step | residuals/step |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['case']} | {r['error']} |")
            continue
        p, c = r["parts_ms_per_step"], r["calls_per_step"]
        lo, hi = r["spread_ms_per_step"]
        lines.append(
            f"| {r['case']} | {r['steps']} | {r['ms_per_step']:.2f} | {lo:.2f}-{hi:.2f} | "
            f"{p.get('jacobian', 0):.2f} | {p.get('lu', 0):.2f} | {p.get('residual', 0):.2f} | "
            f"{c.get('jacobian', 0):.1f} | {c.get('lu', 0):.1f} | {c.get('residual', 0):.1f} |")
    lines += ["", "Parts are inclusive wall times per step from the instrumented run (the",
              "Jacobian includes its law evaluations; the residual's calls inside the Jacobian",
              "are not double counted, since the Jacobian does not call it).", ""]
    for r in results:
        if r.get("refs_calls"):
            lines.append(f"`torch._refs` / `_prims` calls, {r['case']} ({r['steps']} steps):")
            lines += [f"- {k}: {v}" for k, v in list(r["refs_calls"].items())[:10]]
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="2 steps, 1 round, the two circuits only (a smoke run)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    steps, rounds, cases = args.steps, args.rounds, args.cases
    if args.quick:
        steps, rounds = 2, 1
        cases = [c for c in cases if c != "case118"]
    results = []
    for name in cases:
        try:
            results.append(measure(name, steps, rounds))
        except MatpowerCaseUnavailableError as exc:
            results.append({"case": name, "error": f"unavailable ({exc})"})
    text = report(results, machine_block())
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
