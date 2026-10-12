"""Dynamic-wave speed against SWMM 5.2.4 (pyswmm): the dynwave fixtures and synthetic trees.

For each network it times SWMM (`pyswmm.Simulation`, the whole run) and noodl
(`build_model` with its defaults, then `simulate_dynamic_wave` over the same duration) and
reports noodl's fixed-point passes per step, its milliseconds per pass, the steps that used
up their trials (accepted, as SWMM accepts them), both wall times and their ratio. Two
option sets per fixture: ``file``, the committed file (HEAD_TOLERANCE 1e-12 m, MAX_TRIALS
500: the parity tests' tight convergence), and ``swmm``, the same file with HEAD_TOLERANCE
and MAX_TRIALS removed, i.e. SWMM's own defaults (0.005 ft, 8 trials), which noodl's
`build_model` now follows (`dynwave.iterate_tolerances`). The synthetic trees are complete
binary trees of ``2**depth - 1`` conduits, lateral inflow at the leaves, SWMM defaults,
5 min.

The per-pass cost is a fixed overhead of a few thousand small PyTorch operations, nearly
independent of the network size, so the ratio to SWMM falls as the network grows; one pass
costs about the same for a batch of B scenarios (`simulate_dynamic_wave` with a ``(B, n)``
inflow), which amortises it across an ensemble.

MEASURED (one core pinned, `--core 2`; Windows 11, PyTorch 2.14 CPU, SWMM 5.2.4 via
pyswmm 2.2; the machine was shared, so wall times vary ~10 %):

before (commit 26fa6b4: tolerances 1e-11 m / 1e-10 m3/s / 2000 passes whatever the file)

| case | SWMM s | noodl s | passes/step | ms/pass | ratio |
|---|---|---|---|---|---|
BEFORE_ROWS

after (this benchmark's defaults: the file's HEAD_TOLERANCE / MAX_TRIALS)

| case | SWMM s | noodl s | passes/step | ms/pass | non-conv. steps | ratio |
|---|---|---|---|---|---|---|
AFTER_ROWS

Run: `.venv/Scripts/python benchmarks/sewer_dynwave_speed.py [--core N] [--trees 5 7 9]
[--fixtures tree_dynwave.inp ...] [--modes file swmm]`
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "sewer"
FIXTURES = ("tree_dynwave.inp", "backwater_dynwave.inp", "loop_dynwave.inp",
            "structures_dynwave.inp", "storage_dynwave.inp")
#: The [OPTIONS] lines removed for SWMM's own iteration defaults.
_ITERATION_OPTIONS = re.compile(r"(?im)^\s*(MAX_TRIALS|HEAD_TOLERANCE)\s+.*$")


def pin_to_core(core: int) -> None:
    """Pin this process to one logical core (Windows only; elsewhere a no-op): on a hybrid
    or shared CPU the per-op cost differs severalfold between cores."""
    if sys.platform != "win32":
        return
    import ctypes

    kernel = ctypes.windll.kernel32
    kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.c_size_t(1 << core))


def binary_tree(depth: int) -> str:
    """A complete binary tree of ``2**depth - 1`` circular conduits on the tree fixture's
    options (5 min, SWMM's iteration defaults): node k drains to k // 2, the root to the
    outfall, 10 L/s of lateral inflow at each leaf."""
    template = (DATA / "tree_dynwave.inp").read_text()
    options = _ITERATION_OPTIONS.sub("", template.split("[EVAPORATION]")[0])
    options = options.replace("00:30:00", "00:05:00")
    n = 2**depth - 1
    junctions, conduits, xsections, inflows = [], [], [], []
    for k in range(1, n + 1):
        level = k.bit_length()
        junctions.append(f"J{k} {8.0 + 0.4 * level:.3f} 5 0 0 0")
        conduits.append(f"C{k} J{k} {f'J{k // 2}' if k > 1 else 'Out'} 100 0.013 0 0 0 0")
        xsections.append(f"C{k} CIRCULAR {0.3 + 0.15 * (depth - level):.2f} 0 0 0 1")
        if level == depth:
            inflows.append(f'J{k} FLOW "" FLOW 1.0 1.0 0.01')
    return (options + "[JUNCTIONS]\n" + "\n".join(junctions)
            + "\n\n[OUTFALLS]\nOut 7.6 FREE NO\n\n[CONDUITS]\n" + "\n".join(conduits)
            + "\n\n[XSECTIONS]\n" + "\n".join(xsections) + "\n\n[INFLOWS]\n"
            + "\n".join(inflows) + "\n")


def time_swmm(path: Path) -> tuple[float, int]:
    """SWMM's wall time (s) and routing steps for the whole run of `path`."""
    import pyswmm

    started = time.perf_counter()
    steps = 0
    with pyswmm.Simulation(str(path)) as sim:
        for _ in sim:
            steps += 1
    return time.perf_counter() - started, steps


def time_noodl(path: Path, duration: float) -> dict:
    """noodl's wall time (s), steps, passes per step, ms per pass and non-converged steps
    for `duration` s of `path` at `build_model`'s defaults."""
    import torch

    from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_model(net)
    diagnostics: list = []
    started = time.perf_counter()
    simulate_dynamic_wave(model, state, drivers, duration=duration, diagnostics=diagnostics)
    elapsed = time.perf_counter() - started
    passes = sum(d["passes"] for d in diagnostics)
    stuck = sum(1 for d in diagnostics if not bool(torch.as_tensor(d["converged"]).all()))
    return {"s": elapsed, "steps": len(diagnostics), "passes": passes / len(diagnostics),
            "ms_pass": 1e3 * elapsed / passes, "nonconv": stuck}


def report(name: str, path: Path, duration: float) -> None:
    swmm_s, _ = time_swmm(path)
    got = time_noodl(path, duration)
    print(f"| {name} | {swmm_s:.3f} | {got['s']:.2f} | {got['passes']:.2f} | "
          f"{got['ms_pass']:.2f} | {got['nonconv']}/{got['steps']} | "
          f"{got['s'] / swmm_s:,.0f}x |", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--core", type=int, default=None, help="pin to this logical core")
    parser.add_argument("--fixtures", nargs="*", default=list(FIXTURES))
    parser.add_argument("--modes", nargs="*", default=["file", "swmm"],
                        help="option sets: file (as committed), swmm (SWMM's defaults)")
    parser.add_argument("--trees", nargs="*", type=int, default=[3, 6, 9, 11],
                        help="binary-tree depths (2**depth - 1 conduits)")
    args = parser.parse_args()
    if args.core is not None:
        pin_to_core(args.core)
    print("| case | SWMM s | noodl s | passes/step | ms/pass | non-conv. steps | ratio |")
    print("|---|---|---|---|---|---|---|")
    work = Path(tempfile.mkdtemp())
    for name in args.fixtures:
        text = (DATA / name).read_text()
        end = re.search(r"(?m)^END_TIME\s+(\d+):(\d+):(\d+)", text)
        duration = 3600.0 * int(end[1]) + 60.0 * int(end[2]) + int(end[3])
        for label, body in (("file", text), ("swmm", _ITERATION_OPTIONS.sub("", text))):
            if label not in args.modes:
                continue
            path = work / f"{label}_{name}"
            path.write_text(body)
            report(f"{name} ({label})", path, duration)
    for depth in args.trees:
        path = work / f"tree{depth}.inp"
        path.write_text(binary_tree(depth))
        report(f"binary tree, {2**depth - 1} conduits", path, 300.0)


if __name__ == "__main__":
    main()
