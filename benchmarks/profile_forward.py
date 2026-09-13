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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true", help="median-of-5 timing, no profile")
    parser.add_argument(
        "--solve-timing", action="store_true", help="median layer.solve timing, no profile"
    )
    parser.add_argument("--ensembles", type=int, nargs="+", default=[1, 100])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

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
