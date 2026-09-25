"""Interleaved benchmark of the transport layer's four linear solvers.

Compares `TransportLayer(linear_solver=...)` in `("gmres", "gmres_jacobi", "gmres_ilu",
"sparse_direct")` on the implicit transport step's FORWARD time (a plain `.step(...)` call)
and BACKWARD time (`.backward()` through a loss on the summed returned state, with `q` --
the flow driving the step -- as the differentiable leaf, so the transposed solve genuinely
runs through `_LinearSolve.backward`), at ensemble sizes `1`, `32` (the `sparse_direct` batch
cap) and `100` (gmres variants only -- `sparse_direct` is not attempted there, see
`_applicable_solvers`), on `benchmarks.composed_model.build_composed`'s reference topology.
Then the same four solvers on the street/building coupling demo
(`benchmarks.coupling_street_building`), batch 1, one coupled hour.

The protocol is INTERLEAVED, not solver-by-solver: for a given ensemble, one model is built
ONCE, warmed once per solver, then `--rounds` rounds each cycle round-robin over every
solver still applicable, timing one forward-or-backward call per (round, solver). This is
what cancels ambient load out of the ratios between solvers -- on this machine a single `.step`
timing can drift 15-25 s over a 50 s run of unrelated work, which would otherwise look like a
solver difference. Reports median, min, max (not mean: the spread is exactly what interleaving
is trying to keep honest, so it is reported, not averaged away).

This script measures; the default `"auto"` rule in `TransportLayer._resolve_solver` was
chosen on these medians.

Run as a SCRIPT, with `PYTHONPATH` pointed at the checkout's `src` (so that an editable
install elsewhere does not shadow it):

    PYTHONPATH=src .venv/Scripts/python -m benchmarks.transport_solver_bench --quick
    PYTHONPATH=src .venv/Scripts/python -m benchmarks.transport_solver_bench

Writes `benchmarks/transport_solver_bench.json` and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import noodl  # noqa: F401  (printed below; also proves PYTHONPATH resolved the right one)
from benchmarks.composed_model import ComposedModel, _step_state, build_composed
from benchmarks.coupling_street_building import build_city
from benchmarks.measure import time_call
from noodl.layers.transport import TransportLayer

SOLVERS: tuple[str, ...] = ("gmres", "gmres_jacobi", "gmres_ilu", "sparse_direct")
SPARSE_DIRECT_MAX_BATCH = 32
COMPOSED_ENSEMBLES: tuple[int, ...] = (1, 32, 100)
DEFAULT_ROUNDS = 5
DEFAULT_COUPLING_ROUNDS = 3
DT = 60.0

OUT_PATH = Path(__file__).resolve().parent / "transport_solver_bench.json"


def _applicable_solvers(ensemble: int) -> tuple[tuple[str, ...], str | None]:
    """`(solvers to attempt, note about anything excluded)` for one ensemble size."""
    if ensemble > SPARSE_DIRECT_MAX_BATCH:
        note = (
            f"sparse_direct excluded at ensemble={ensemble}: its batch cap is "
            f"{SPARSE_DIRECT_MAX_BATCH} (SuperLU factorises one instance at a time in a "
            f"Python loop; not attempted, not an error)"
        )
        return tuple(s for s in SOLVERS if s != "sparse_direct"), note
    return SOLVERS, None


def _stats(samples: list[float]) -> dict:
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "n": len(samples),
    }


# --- composed-model rows -------------------------------------------------------------------


def _prepare_composed(
    model: ComposedModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(q_slice, x0, x_boundary, zero_sources)` for the co2 transport layer's implicit step.

    `q_slice` comes from ONE untimed, non-differentiable potential-layer solve (the airpath
    slice of its flows) -- a realistic flow field to drive the transport step, built once and
    reused, detached, across every solver and every round so what differs between rows is
    only the transport layer's own linear solve, never the potential layer's.
    """
    lo, hi = model.layer._kind_slices["airpath"]
    _phi, q = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    q_slice = q[..., lo:hi].detach()
    x0, x_boundary, zero_sources = _step_state(model)
    return q_slice, x0, x_boundary, zero_sources


def _forward_once(model: ComposedModel, inputs) -> tuple[float, dict | None]:
    q_slice, x0, x_boundary, zero_sources = inputs
    diag: dict = {}
    elapsed, _ = time_call(
        lambda: model.transport.step(
            x0, q_slice, zero_sources, x_boundary, dt=DT, diagnostics=diag
        )
    )
    return elapsed, diag.get("linear")


def _backward_once(model: ComposedModel, inputs) -> tuple[float, dict | None]:
    q_slice, x0, x_boundary, zero_sources = inputs
    q_grad = q_slice.detach().clone().requires_grad_(True)
    diag: dict = {}
    x = model.transport.step(x0, q_grad, zero_sources, x_boundary, dt=DT, diagnostics=diag)
    loss = x.sum()
    elapsed, _ = time_call(loss.backward)
    return elapsed, diag.get("linear")


def bench_composed_ensemble(ensemble: int, rounds: int) -> list[dict]:
    """Interleaved rows for one ensemble size: one row per solver, forward and backward."""
    print(f"\n--- composed model, ensemble={ensemble} ---", flush=True)
    model = build_composed(ensemble=ensemble)
    inputs = _prepare_composed(model)
    solvers, note = _applicable_solvers(ensemble)
    if note:
        print(f"  note: {note}", flush=True)

    rows: dict[str, dict] = {}
    live: list[str] = []
    for solver in solvers:
        model.transport.linear_solver = solver
        try:
            _forward_once(model, inputs)  # warm every construction-time cache
            _forward_once(model, inputs)
            _backward_once(model, inputs)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal to the whole run
            print(f"  {solver}: FAILED during warm-up: {exc!r}", flush=True)
            rows[solver] = {
                "ensemble": ensemble,
                "solver": solver,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        live.append(solver)
        rows[solver] = {
            "ensemble": ensemble,
            "solver": solver,
            "forward": {"samples": []},
            "backward": {"samples": []},
        }

    for r in range(rounds):
        for solver in list(live):
            model.transport.linear_solver = solver
            try:
                elapsed, linear = _forward_once(model, inputs)
            except Exception as exc:  # noqa: BLE001
                rows[solver]["error"] = f"{type(exc).__name__}: {exc}"
                live.remove(solver)
                continue
            rows[solver]["forward"]["samples"].append(elapsed)
            if linear:
                rows[solver]["forward"]["backend"] = linear.get("backend")
                rows[solver]["forward"]["iterations"] = (
                    int(linear["iterations"]) if linear.get("iterations") is not None else None
                )
        for solver in list(live):
            model.transport.linear_solver = solver
            try:
                elapsed, linear = _backward_once(model, inputs)
            except Exception as exc:  # noqa: BLE001
                rows[solver]["error"] = f"{type(exc).__name__}: {exc}"
                live.remove(solver)
                continue
            rows[solver]["backward"]["samples"].append(elapsed)
            if linear:
                rows[solver]["backward"]["backend"] = linear.get("backend")
                rows[solver]["backward"]["iterations"] = (
                    int(linear["iterations"]) if linear.get("iterations") is not None else None
                )
        print(f"  round {r + 1}/{rounds} done", flush=True)

    out = []
    for solver in solvers:
        row = rows[solver]
        if "error" in row and "forward" not in row:
            out.append(row)
            continue
        for direction in ("forward", "backward"):
            samples = row[direction]["samples"]
            entry = {
                "ensemble": ensemble,
                "solver": solver,
                "direction": direction,
                "backend": row[direction].get("backend"),
                "iterations": row[direction].get("iterations"),
            }
            if samples:
                entry.update(_stats(samples))
            if "error" in row:
                entry["error"] = row["error"]
            out.append(entry)
    if not solvers or "sparse_direct" not in solvers:
        out.append({
            "ensemble": ensemble,
            "solver": "sparse_direct",
            "not_applicable": True,
            "reason": note,
        })
    return out


# --- coupling-demo rows ---------------------------------------------------------------------


def _set_transport_solver(city, solver: str) -> None:
    for m in city.models.values():
        for layer in m.layers.values():
            if isinstance(layer, TransportLayer):
                layer.linear_solver = solver


def bench_coupling(rounds: int) -> list[dict]:
    """Interleaved rows for the street/building coupling demo, batch 1, one coupled hour."""
    print("\n--- coupling demo, batch=1, one coupled hour ---", flush=True)
    cities: dict[str, tuple] = {}
    live: list[str] = []
    rows: dict[str, dict] = {}
    for solver in SOLVERS:
        city, state0, drivers = build_city(1)
        _set_transport_solver(city, solver)
        try:
            city.step(state0, drivers, dt=3600.0, diagnostics={})  # warm every cache
            city.step(state0, drivers, dt=3600.0, diagnostics={})
        except Exception as exc:  # noqa: BLE001
            print(f"  {solver}: FAILED during warm-up: {exc!r}", flush=True)
            rows[solver] = {"solver": solver, "error": f"{type(exc).__name__}: {exc}"}
            continue
        cities[solver] = (city, state0, drivers)
        live.append(solver)
        rows[solver] = {"solver": solver, "samples": []}

    for r in range(rounds):
        for solver in list(live):
            city, state0, drivers = cities[solver]

            def _one_hour(city=city, state0=state0, drivers=drivers):
                return city.step(state0, drivers, dt=3600.0, diagnostics={})

            try:
                elapsed, _ = time_call(_one_hour)
            except Exception as exc:  # noqa: BLE001
                rows[solver]["error"] = f"{type(exc).__name__}: {exc}"
                live.remove(solver)
                continue
            rows[solver]["samples"].append(elapsed)
        print(f"  round {r + 1}/{rounds} done", flush=True)

    out = []
    for solver in SOLVERS:
        row = rows[solver]
        entry = {"solver": solver}
        if row["samples"]:
            entry.update(_stats(row["samples"]))
        if "error" in row:
            entry["error"] = row["error"]
        out.append(entry)
    return out


# --- machine block, table, main ---------------------------------------------------------------


def _git_commit() -> str | None:
    """The short commit the benchmark ran at, so a future run (or a stale one straddling a
    behaviour change, as F-B2 recorded) is attributable without cross-referencing a timestamp.
    `None` when git itself is unavailable rather than raising -- attributability is a nicety,
    not a benchmark precondition."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_block() -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        noodl_file = str(Path(noodl.__file__).resolve().relative_to(repo_root))
    except ValueError:
        noodl_file = noodl.__file__  # outside the repo (e.g. an unrelated install)
    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "noodl_file": noodl_file,
        "git_commit": _git_commit(),
    }


def _fmt_ms(entry: dict) -> str:
    if "error" in entry and "median_s" not in entry:
        return "ERROR"
    if entry.get("not_applicable"):
        return "n/a"
    if "median_s" not in entry:
        return "-"
    return (
        f"{entry['median_s'] * 1e3:.1f} "
        f"[{entry['min_s'] * 1e3:.1f}, {entry['max_s'] * 1e3:.1f}]"
    )


def print_table(composed_rows: list[dict], coupling_rows: list[dict]) -> None:
    print("\n## Composed-model transport step (forward / backward), median [min, max] ms\n")
    print("| ensemble | solver | direction | median [min, max] ms | iterations |")
    print("|---|---|---|---|---|")
    for row in composed_rows:
        if row.get("not_applicable"):
            print(f"| {row['ensemble']} | {row['solver']} | - | n/a | - |")
            continue
        it = row.get("iterations")
        print(
            f"| {row['ensemble']} | {row['solver']} | {row.get('direction', '-')} | "
            f"{_fmt_ms(row)} | {it if it is not None else '-'} |"
        )
    print("\n## Coupling demo (street + building), batch=1, one coupled hour, ms\n")
    print("| solver | median [min, max] ms |")
    print("|---|---|")
    for row in coupling_rows:
        print(f"| {row['solver']} | {_fmt_ms(row)} |")


def main(argv: list[str] | None = None) -> int:
    print(f"noodl.__file__ = {noodl.__file__}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="1 round, ensemble=1 only, coupling 1 round -- for shaking the script out",
    )
    parser.add_argument("--rounds", type=int, default=None, help="override composed rounds")
    parser.add_argument(
        "--coupling-rounds", type=int, default=None, help="override coupling rounds"
    )
    parser.add_argument(
        "--ensembles", type=int, nargs="+", default=None, help="override composed ensembles"
    )
    parser.add_argument("--skip-coupling", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.quick:
        rounds = args.rounds or 1
        coupling_rounds = args.coupling_rounds or 1
        ensembles = args.ensembles or (1,)
    else:
        rounds = args.rounds or DEFAULT_ROUNDS
        coupling_rounds = args.coupling_rounds or DEFAULT_COUPLING_ROUNDS
        ensembles = tuple(args.ensembles) if args.ensembles else COMPOSED_ENSEMBLES

    machine = machine_block()
    print(f"machine: {machine}")

    composed_rows: list[dict] = []
    for ensemble in ensembles:
        composed_rows.extend(bench_composed_ensemble(ensemble, rounds))
        result = {
            "machine": machine,
            "quick": bool(args.quick),
            "composed": composed_rows,
            "coupling": [],
        }
        args.out.write_text(json.dumps(result, indent=2))

    coupling_rows: list[dict] = []
    if not args.skip_coupling:
        coupling_rows = bench_coupling(coupling_rounds)

    result = {
        "machine": machine,
        "quick": bool(args.quick),
        "composed": composed_rows,
        "coupling": coupling_rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")

    print_table(composed_rows, coupling_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
