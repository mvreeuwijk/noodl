"""cProfile (and fast timing) of the composed model's NON-differentiable forward solve.

This is the measurement instrument for the spec's section 6.2 forward-time follow-up, step 1
(dispatch overhead): at the reference composed size the forward solve is ~94% `pcg`, and the
question this script answers is how much of that `pcg` time is spent in the SOLVER'S OWN
Python/dispatch overhead -- the per-iteration `torch.where` masking, the index re-broadcasting
inside the operator matvecs -- rather than in the floating-point work.

Run it as a SCRIPT, never as a test:

    .venv/Scripts/python -m benchmarks.profile_forward             # profile, ensembles 1 + 100
    .venv/Scripts/python -m benchmarks.profile_forward --timing    # median-of-5 workload timing
    .venv/Scripts/python -m benchmarks.profile_forward --ensembles 1
    .venv/Scripts/python -m benchmarks.profile_forward --compare-solvers

`--timing` is the before/after number the follow-up is judged on: the median over 5 warm runs
of `workload_forward`'s own already-warm timed section (its model build and one warm-up step
happen outside that section, so the median is the steady per-step cost).

The "dispatch share" line is a LOWER bound on how much of `pcg` is overhead, and its
components are process-wide totals: `torch.where`, `Tensor.expand` and `broadcast_shapes` are
counted wherever they were called, not only under `pcg`. During `layer.solve` essentially all
of them are under `pcg` (the residual/Jacobian assembly calls each a handful of times per
Newton step against `pcg`'s thousands), so the figure is reported as-is and read as an
order-of-magnitude, not to three digits.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys

from benchmarks.composed_model import build_composed, workload_forward
from benchmarks.measure import time_call

# Profile entries counted as "dispatch overhead" inside `pcg`. cProfile records a C function
# under its whole repr ("<built-in method torch.where>", "<method 'expand' of
# 'torch._C.TensorBase' objects>"), not a bare name, so these are SUBSTRINGS, not names.
# `broadcast_shapes` matches both `torch.functional.broadcast_shapes` and the
# `torch._refs._broadcast_shapes` it calls; summing TOTTIME (never cumtime) counts each
# frame's own time exactly once.
_WHERE_MARKERS = ("torch.where",)
_EXPAND_MARKERS = ("'expand' of", "broadcast_shapes", "_bcast_index")


def _solve_once(model) -> None:
    model.layer.solve(
        model.phi_boundary,
        model.drivers,
        model.sources,
        differentiable=False,
    )


def _tottime_by_marker(stats: pstats.Stats, markers: tuple[str, ...]) -> tuple[float, int]:
    """Total tottime and call count over every profiled function matching any marker."""
    total_time = 0.0
    total_calls = 0
    for (_file, _line, func), (_cc, nc, tt, _ct, _callers) in stats.stats.items():
        if any(marker in func for marker in markers):
            total_time += tt
            total_calls += nc
    return total_time, total_calls


def _cumtime_of(stats: pstats.Stats, name: str) -> tuple[float, float]:
    """(cumtime, tottime) of the profiled function called `name` (0.0 if never called)."""
    for (_file, _line, func), (_cc, _nc, tt, ct, _callers) in stats.stats.items():
        if func == name:
            return ct, tt
    return 0.0, 0.0


def profile_ensemble(ensemble: int) -> None:
    """cProfile one `layer.solve` at the given ensemble size, after one warm-up solve."""
    model = build_composed(ensemble=ensemble)
    _solve_once(model)  # warm every construction-time cache before the profiled call

    profiler = cProfile.Profile()
    profiler.enable()
    _solve_once(model)
    profiler.disable()

    stats = pstats.Stats(profiler)
    print(f"\n{'=' * 78}\nensemble {ensemble}: top 15 by cumulative time\n{'=' * 78}")
    stats.sort_stats("cumulative").print_stats(15)

    pcg_cum, pcg_tot = _cumtime_of(stats, "pcg")
    where_time, where_calls = _tottime_by_marker(stats, _WHERE_MARKERS)
    expand_time, expand_calls = _tottime_by_marker(stats, _EXPAND_MARKERS)
    overhead = pcg_tot + where_time + expand_time
    share = overhead / pcg_cum if pcg_cum > 0 else float("nan")
    print(
        f"ensemble {ensemble}: pcg cumtime {pcg_cum:.3f} s; pcg tottime {pcg_tot:.3f} s; "
        f"torch.where {where_time:.3f} s ({where_calls} calls); "
        f"index expansion {expand_time:.3f} s ({expand_calls} calls); "
        f"dispatch share of pcg = {share:.1%}"
    )


def time_solve(ensemble: int, repeats: int = 15) -> float:
    """Median wall time of `repeats` warm `layer.solve` calls on ONE model, in seconds.

    The low-variance companion to `time_ensemble`: it times exactly what the profile above
    covers, in one process, on one already-built model, so run-to-run scatter is the
    machine's rather than the model build's. On this (noisy, 14-thread) box the
    `workload_forward` median of 5 moved by 1.6x between two runs of IDENTICAL code, which is
    far more than the effect being measured -- this figure moves by a few percent instead.
    """
    model = build_composed(ensemble=ensemble)
    for _ in range(3):
        _solve_once(model)
    samples = [time_call(lambda: _solve_once(model))[0] for _ in range(repeats)]
    median = statistics.median(samples)
    print(
        f"ensemble {ensemble}: layer.solve median {median * 1e3:.1f} ms of {repeats} warm "
        f"calls [min {min(samples) * 1e3:.1f} ms, max {max(samples) * 1e3:.1f} ms]"
    )
    return median


def time_ensemble(ensemble: int, repeats: int = 5) -> float:
    """Median of `repeats` warm `workload_forward` runs (its own timed section), in seconds."""
    samples = [workload_forward(ensemble=ensemble)["elapsed_s"] for _ in range(repeats)]
    median = statistics.median(samples)
    formatted = ", ".join(f"{s:.4f}" for s in samples)
    print(f"ensemble {ensemble}: median {median:.4f} s of 5 warm runs [{formatted}]")
    return median


# --- section 6.2 step 2: the in-process auto-vs-sparse_direct comparison ---------------------
#
# The budget report (`benchmarks.report_composed_scaling`) measures the same two solvers from
# a fresh child process per sample, which is right for a MEMORY peak and for an end-to-end
# wall clock but costs ~45 minutes and buries a 2x solver difference under the process start
# and the model build. This comparison is the same question asked in ONE process, on ONE
# already-warm model per configuration, so what it reports is the solve itself. It is what the
# `method="auto"` default was chosen on (spec section 2: "selected on evidence, per platform").

# "auto" now resolves to sparse_direct for a certified-SPD operator (Task C), so comparing it
# against "sparse_direct" would measure one backend against itself; "cg" is the PCG backend
# the evidence table in `solvers.select`'s docstring actually compares sparse_direct against.
COMPARE_SOLVERS = ("cg", "sparse_direct")
COMPARE_FORWARD_ENSEMBLES = (1, 100, 1000)
COMPARE_BACKWARD_ENSEMBLES = (1, 100)


def _forward_with_diagnostics(model) -> dict:
    """One non-differentiable `layer.solve`, returning its diagnostics dict."""
    diagnostics: dict = {}
    model.layer.solve(
        model.phi_boundary,
        model.drivers,
        model.sources,
        differentiable=False,
        diagnostics=diagnostics,
    )
    return diagnostics


def compare_forward(ensemble: int, solver: str, repeats: int = 3) -> dict:
    """Median warm `layer.solve` time under one `linear_solver`, plus its iteration counts."""
    model = build_composed(ensemble=ensemble, linear_solver=solver)
    diagnostics = _forward_with_diagnostics(model)  # also warms every construction cache
    _forward_with_diagnostics(model)
    samples = [time_call(lambda: _forward_with_diagnostics(model))[0] for _ in range(repeats)]
    linear = diagnostics.get("linear_iterations")
    return {
        "ensemble": ensemble,
        "solver": solver,
        "seconds": statistics.median(samples),
        "newton_iterations": int(diagnostics["newton_iterations"]),
        "linear_iterations_max": None if linear is None else int(linear.amax()),
    }


def _timed_backward(model) -> float:
    """Build a fresh differentiable forward (UNTIMED) and time `loss.backward()` alone.

    The graph has to be rebuilt for every sample: `backward` frees it, and a second
    `backward` on the same graph would raise rather than re-measure. Only the backward pass
    is timed, matching `benchmarks.composed_model.workload_backward`.
    """
    sources = model.sources.detach().clone().requires_grad_(True)
    phi, _q = model.layer.solve(
        model.phi_boundary, model.drivers, sources, differentiable=True
    )
    loss = phi[..., model.layer.interior].sum()
    elapsed, _ = time_call(loss.backward)
    return elapsed


def compare_backward(ensemble: int, solver: str, repeats: int = 3) -> dict:
    """Median warm `loss.backward()` time through the implicit adjoint, under one solver."""
    model = build_composed(ensemble=ensemble, linear_solver=solver)
    _forward_with_diagnostics(model)  # warm the construction-time caches
    _timed_backward(model)
    samples = [_timed_backward(model) for _ in range(repeats)]
    return {"ensemble": ensemble, "solver": solver, "seconds": statistics.median(samples)}


def compare_solvers(repeats: int = 3, *, forward: bool = True, backward: bool = True) -> list:
    """The comparison table, printed row by row as it is measured.

    `forward`/`backward` select halves of it, because the whole table is several minutes of
    wall clock (the ensemble-1000 forward alone is tens of seconds per sample under either
    solver) and the two halves are independent measurements.
    """
    rows = []
    for ensemble in COMPARE_FORWARD_ENSEMBLES if forward else ():
        for solver in COMPARE_SOLVERS:
            row = {"direction": "forward", **compare_forward(ensemble, solver, repeats)}
            print(
                f"forward  ensemble={row['ensemble']:>4} solver={row['solver']:<13} "
                f"{row['seconds'] * 1e3:9.1f} ms  newton={row['newton_iterations']} "
                f"linear_iterations_max={row['linear_iterations_max']}"
            )
            rows.append(row)
    for ensemble in COMPARE_BACKWARD_ENSEMBLES if backward else ():
        for solver in COMPARE_SOLVERS:
            row = {"direction": "backward", **compare_backward(ensemble, solver, repeats)}
            print(
                f"backward ensemble={row['ensemble']:>4} solver={row['solver']:<13} "
                f"{row['seconds'] * 1e3:9.1f} ms"
            )
            rows.append(row)
    for direction, ensemble in sorted({(r["direction"], r["ensemble"]) for r in rows}):
        by_solver = {
            r["solver"]: r["seconds"]
            for r in rows
            if r["direction"] == direction and r["ensemble"] == ensemble
        }
        speedup = by_solver["auto"] / by_solver["sparse_direct"]
        print(
            f"{direction:<8} ensemble={ensemble:>4}: sparse_direct is {speedup:.2f}x "
            f"{'faster' if speedup > 1 else 'SLOWER'} than auto"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true", help="median-of-5 timing, no profile")
    parser.add_argument(
        "--solve-timing", action="store_true", help="median layer.solve timing, no profile"
    )
    parser.add_argument(
        "--compare-solvers",
        action="store_true",
        help="auto vs sparse_direct, forward and backward, in process (section 6.2 step 2)",
    )
    parser.add_argument(
        "--forward-only", action="store_true", help="--compare-solvers: forward rows only"
    )
    parser.add_argument(
        "--backward-only", action="store_true", help="--compare-solvers: backward rows only"
    )
    parser.add_argument("--ensembles", type=int, nargs="+", default=[1, 100])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    if args.compare_solvers:
        compare_solvers(
            args.repeats,
            forward=not args.backward_only,
            backward=not args.forward_only,
        )
        return 0

    for ensemble in args.ensembles:
        if args.solve_timing:
            time_solve(ensemble)
        elif args.timing:
            time_ensemble(ensemble, repeats=args.repeats)
        else:
            profile_ensemble(ensemble)
    return 0


if __name__ == "__main__":
    sys.exit(main())
