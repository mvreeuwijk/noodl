"""The milestone-1b acceptance measurements: the section 6.1 budget table and the shape gates.

Every measurement lives here, and `tests/verification/test_composed_scaling.py` imports these
functions rather than re-implementing them, so the numbers the gate asserts on and the numbers
the JSON report records are the same numbers, produced by the same code.

Times and memory both come from `benchmarks.measure.isolated_peak_rss`, i.e. from a fresh
child process per measurement: `tracemalloc` cannot see PyTorch's allocator at all (amendment
A4), and a peak taken in a process that has already run three other configurations is not a
peak for any of them.

Iteration counts travel beside the times because design section 6.1 requires it: on a machine
with enough threads, a conditioning regression can leave wall-clock unchanged while the inner
solver iteration count doubles.

Every budget row is measured under EVERY entry of `SOLVERS` (spec section 6.2 step 2: the
`method="auto"` default is selected on evidence, so the evidence lives in the report), and
each row carries a `solver` field saying which. The row keys are otherwise unchanged, so a
consumer that ignores `solver` reads the same schema it always did.

Run it explicitly (it takes minutes -- nothing runs it automatically):

    .venv/Scripts/python -m benchmarks.report_composed_scaling

(as a module, not as a path: it imports its siblings through the `benchmarks` package)
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import torch

from benchmarks.composed_model import build_composed
from benchmarks.measure import isolated_peak_rss, time_call
from tellegen.operators.graph import GraphLaplacianOperator

# The reference composed model of design section 6: 8 buildings x 120 nodes, a 40-node street
# and a 30-node sewer network -- 1030 nodes, about 2200 edges.
REFERENCE_KWARGS = {
    "n_buildings": 8,
    "building_nodes": 120,
    "street_nodes": 40,
    "sewer_nodes": 30,
}

# (ensemble, steps, forward_budget_s, backward_budget_s, peak_memory_budget_bytes), exactly
# design section 6.1's table. These are the gate; they are not adjustable here.
BUDGET_TABLE = [
    (1, 1, 0.050, 0.100, 100 * 1_000_000),
    (100, 1, 0.500, 1.000, 1_000 * 1_000_000),
    (100, 24, 12.0, 25.0, 2_000 * 1_000_000),
    (1000, 1, 5.0, 10.0, 8_000 * 1_000_000),
]


# The inner linear solvers the budget table is measured under (spec section 6.2 step 2: the
# default for `method="auto"` is chosen ON EVIDENCE, so the evidence has to exist). "auto" is
# the shipped default; "sparse_direct" is the SciPy SuperLU reference path. The FIRST entry is
# the one whose numbers the gate reads -- `measure_budget_row`'s own default -- so the
# committed report keeps meaning what `tests/verification/test_composed_scaling.py` asserts.
#
# NOTE, since Task C decided the default: for a certified-SPD `PotentialFlowLayer` with SciPy
# installed, "auto" now RESOLVES to sparse_direct (see `solvers.select`'s eligibility table),
# so these two rows measure the same backend and differ only in whether it was chosen or
# pinned. That makes the second row a check that the routing rule really lands where it says,
# at the cost of doubling the report's wall clock. Swap the second entry to "cg" to recover
# the PCG comparison the Task C measurement was made on -- the evidence itself is recorded in
# `solvers.select`'s module docstring and in the Task C report, not only here.
SOLVERS = ("auto", "sparse_direct")


def measure_budget_row(
    ensemble: int,
    steps: int,
    forward_budget: float,
    backward_budget: float,
    memory_budget: int,
    samples: int = 3,
    solver: str = "auto",
) -> dict:
    """Measure one budget-table row: forward time, backward time, peak RSS, iteration counts.

    `samples` child processes per direction. The reported TIME is the median over samples,
    because a single sample is not a measurement at this size: the ensemble-1 forward figure
    was seen to range over 0.315-1.621 s across runs on the development machine, a 5x spread,
    which is wider than several of the budget margins being judged. The shape gates already
    take a median for the same reason; this brings the budget rows into line with them.

    The reported PEAK is the MAXIMUM over samples, not the median: a peak is a high-water
    mark, and the question the memory budget asks is how much this configuration can need.

    `peak_memory_bytes` is the larger of the forward and backward peaks. The row's memory
    budget is a budget for the configuration, and a backward pass that needs more than the
    forward pass is still this row's peak.

    `solver` is the layer's inner linear solver (`PotentialFlowLayer(linear_solver=...)`),
    carried into the child process and recorded in the row. It defaults to `"auto"`, the
    shipped default, so the gate keeps measuring what ships; the report additionally measures
    every entry of `SOLVERS`, which is the section 6.2 evidence the `"auto"` default is
    chosen on. The BUDGETS do not vary with it -- they are the design's, not the backend's.
    """
    kwargs = {
        **REFERENCE_KWARGS,
        "ensemble": ensemble,
        "steps": steps,
        "linear_solver": solver,
    }

    def measure(workload: str) -> tuple[int, float, dict]:
        peaks, results = [], []
        for _ in range(samples):
            peak, result = isolated_peak_rss("benchmarks.composed_model", workload, kwargs)
            peaks.append(peak)
            results.append(result)
        return max(peaks), median(r["elapsed_s"] for r in results), results[0]

    peak_forward, forward_seconds, forward = measure("workload_forward")
    peak_backward, backward_seconds, backward = measure("workload_backward")
    peak = max(peak_forward, peak_backward)
    return {
        "ensemble": ensemble,
        "steps": steps,
        "samples": samples,
        "solver": solver,
        "n_nodes": forward["n_nodes"],
        "n_edges": forward["n_edges"],
        "forward_seconds": forward_seconds,
        "forward_budget_seconds": forward_budget,
        "forward_within_budget": forward_seconds <= forward_budget,
        "backward_seconds": backward_seconds,
        "backward_budget_seconds": backward_budget,
        "backward_within_budget": backward_seconds <= backward_budget,
        "peak_memory_bytes": peak,
        "peak_memory_forward_bytes": peak_forward,
        "peak_memory_backward_bytes": peak_backward,
        "peak_memory_budget_bytes": memory_budget,
        "peak_memory_within_budget": peak <= memory_budget,
        "newton_iterations": forward["newton_iterations"],
        "linear_iterations_max": forward["linear_iterations_max"],
        "newton_iterations_backward": backward["newton_iterations"],
        "linear_iterations_max_backward": backward["linear_iterations_max"],
        "method": forward["method"],
    }


def format_budget_row(row: dict) -> str:
    """One line per row, with every measured number and its verdict, for `-s` output."""

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return (
        f"budget[ensemble={row['ensemble']}, steps={row['steps']}, "
        f"solver={row.get('solver', 'auto')}]: "
        f"forward {row['forward_seconds']:.3f}s / {row['forward_budget_seconds']}s "
        f"{verdict(row['forward_within_budget'])}, "
        f"backward {row['backward_seconds']:.3f}s / {row['backward_budget_seconds']}s "
        f"{verdict(row['backward_within_budget'])}, "
        f"peak {row['peak_memory_bytes'] / 1e6:.1f} MB "
        f"(fwd {row['peak_memory_forward_bytes'] / 1e6:.1f}, "
        f"bwd {row['peak_memory_backward_bytes'] / 1e6:.1f}) "
        f"/ {row['peak_memory_budget_bytes'] / 1e6:.0f} MB "
        f"{verdict(row['peak_memory_within_budget'])}, "
        f"newton_iterations={row['newton_iterations']}, "
        f"linear_iterations_max={row['linear_iterations_max']}, "
        f"method={row['method']}, "
        f"samples={row['samples']} (median time, max peak)"
    )


# The doubled-node configuration of shape gate 1: every submodel's node count doubled, which
# doubles the builder's derived extra-edge counts with it, so the edge-to-node ratio is held
# fixed and the only thing that changes is size.
DOUBLED_NODE_KWARGS = {
    "n_buildings": 8,
    "building_nodes": 240,
    "street_nodes": 80,
    "sewer_nodes": 60,
}


def measure_memory_shape_gate() -> dict:
    """Shape gate 1: peak RSS at ~1030 nodes against ~2060, fixed ensemble 1, one step."""
    small_kwargs = {**REFERENCE_KWARGS, "ensemble": 1, "steps": 1}
    large_kwargs = {**DOUBLED_NODE_KWARGS, "ensemble": 1, "steps": 1}
    peak_small, small = isolated_peak_rss(
        "benchmarks.composed_model", "workload_forward", small_kwargs
    )
    peak_large, large = isolated_peak_rss(
        "benchmarks.composed_model", "workload_forward", large_kwargs
    )
    return {
        "name": "peak RSS vs nodes",
        "quantity": "peak RSS",
        "small_label": f"{small['n_nodes']} nodes / {small['n_edges']} edges",
        "large_label": f"{large['n_nodes']} nodes / {large['n_edges']} edges",
        "small": peak_small,
        "large": peak_large,
        "small_nodes": small["n_nodes"],
        "large_nodes": large["n_nodes"],
        "small_edges": small["n_edges"],
        "large_edges": large["n_edges"],
        "ratio": peak_large / peak_small,
        "budget": 2.5,
        "within_budget": peak_large / peak_small <= 2.5,
        "units": "bytes",
    }


def _scaled_edge_operator(layer, slopes: torch.Tensor, multiplier: int, seed: int = 0):
    """A `GraphLaplacianOperator` on the SAME nodes as `layer`, with `multiplier`x the edges.

    `build_composed` derives each submodel's extra-edge count from its node count, so it
    cannot be asked for more edges at the same node count. The extra edges are drawn here
    instead, by the same rule the builder's own `_random_tree_plus_extra` uses for its extra
    edges -- a uniformly random ordered pair of distinct nodes -- and given slopes resampled
    from the layer's own, so the scaled operator differs from the reference in edge count and
    nothing else. `matvec` is a gather/scatter over edges against a fixed node-space buffer,
    which is exactly what this gate is about.
    """
    n = layer._boundary_mask.shape[-1]
    b = int(layer._src.numel())
    rng = torch.Generator().manual_seed(seed)
    src, tgt, slope_blocks = [layer._src], [layer._tgt], [slopes]
    for _ in range(multiplier - 1):
        src_extra = torch.randint(0, n, (b,), generator=rng)
        tgt_extra = torch.randint(0, n, (b,), generator=rng)
        # No self-loops, matching the builder's `if u != v` rejection, without changing count.
        self_loops = src_extra == tgt_extra
        tgt_extra[self_loops] = (tgt_extra[self_loops] + 1) % n
        src.append(src_extra)
        tgt.append(tgt_extra)
        slope_blocks.append(slopes[..., torch.randperm(b, generator=rng)])
    return GraphLaplacianOperator(
        torch.cat(src),
        torch.cat(tgt),
        torch.cat(slope_blocks, dim=-1),
        len(layer.interior),
        layer._interior_of_node,
        boundary_mask=layer._boundary_mask,
    )


# Edge multipliers at which the same doubling is re-measured, purely as a diagnostic. At the
# reference edge count a matvec is dominated by fixed per-call dispatch overhead rather than
# by the edges themselves (measured: flat at ~0.10 ms/matvec from 2193 up to 17544 edges), so
# the headline ratio would sit near 1.0 even for an implementation that was not linear in
# edges at all. These two points are in the regime where edge work dominates, and their ratio
# is what shows the operator is genuinely linear in edge count.
EDGE_SENSITIVITY_MULTIPLIERS = (16, 32)


def measure_matvec_shape_gate(repetitions: int = 200, medians_of: int = 5) -> dict:
    """Shape gate 2: `matvec` time at the reference edge count against twice that.

    The median of `medians_of` batches of `repetitions` matvecs each, per operator, because
    a single batch on a loaded machine is noise; the median is taken over batches rather
    than over individual calls so per-call timer resolution never enters the figure.

    `edge_sensitivity_ratio` repeats the same doubling at 16x and 32x the reference edge
    count, where the measurement is actually edge-bound rather than dispatch-bound -- see
    EDGE_SENSITIVITY_MULTIPLIERS. It is reported, not asserted; the gate is the ratio at the
    reference size, as the design specifies.
    """
    model = build_composed()
    layer = model.layer
    phi, _q = layer.solve(model.phi_boundary, model.drivers, model.sources, differentiable=False)
    slopes = layer.dflows(phi, model.drivers)
    operator = _scaled_edge_operator(layer, slopes, 1)
    doubled = _scaled_edge_operator(layer, slopes, 2)
    big, bigger = (
        _scaled_edge_operator(layer, slopes, m) for m in EDGE_SENSITIVITY_MULTIPLIERS
    )
    x = torch.zeros(
        slopes.shape[:-1] + (len(layer.interior),), dtype=slopes.dtype, device=slopes.device
    )

    def batch(op) -> float:
        elapsed, _ = time_call(lambda: [op.matvec(x) for _ in range(repetitions)])
        return elapsed

    for op in (operator, doubled, big, bigger):
        batch(op)  # warm every operator up before any of them is timed
    t_small = median(batch(operator) for _ in range(medians_of))
    t_large = median(batch(doubled) for _ in range(medians_of))
    t_big = median(batch(big) for _ in range(medians_of))
    t_bigger = median(batch(bigger) for _ in range(medians_of))
    b_small = int(layer._src.numel())
    return {
        "name": "matvec time vs edges",
        "quantity": "matvec time",
        "small_label": f"{b_small} edges",
        "large_label": f"{2 * b_small} edges",
        "small": t_small,
        "large": t_large,
        "small_edges": b_small,
        "large_edges": 2 * b_small,
        "nodes": int(model.net.n),
        "repetitions": repetitions,
        "medians_of": medians_of,
        "ratio": t_large / t_small,
        "budget": 2.5,
        "within_budget": t_large / t_small <= 2.5,
        "units": f"seconds per {repetitions} matvecs",
        "edge_sensitivity_multipliers": list(EDGE_SENSITIVITY_MULTIPLIERS),
        "edge_sensitivity_seconds": [t_big, t_bigger],
        "edge_sensitivity_ratio": t_bigger / t_big,
    }


def format_shape_gate(gate: dict) -> str:
    """One line per shape gate, with both measured points and the verdict."""
    scale = 1e6 if gate["units"] == "bytes" else 1.0
    unit = "MB" if gate["units"] == "bytes" else "s"
    return (
        f"shape gate [{gate['name']}]: {gate['small_label']} -> {gate['small'] / scale:.3f}{unit}, "
        f"{gate['large_label']} -> {gate['large'] / scale:.3f}{unit}, "
        f"ratio {gate['ratio']:.2f} / budget {gate['budget']} "
        f"{'PASS' if gate['within_budget'] else 'FAIL'}"
        + (
            f" [edge sensitivity: same doubling at "
            f"{gate['edge_sensitivity_multipliers'][0]}x -> "
            f"{gate['edge_sensitivity_multipliers'][1]}x the reference edge count gives "
            f"ratio {gate['edge_sensitivity_ratio']:.2f}]"
            if "edge_sensitivity_ratio" in gate
            else ""
        )
    )


def machine_info() -> dict:
    """What the numbers in this report depend on besides the code itself."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> None:
    """Measure everything and write `benchmarks/composed_scaling_report.json`.

    Run explicitly after a change that could affect conditioning or scaling; nothing runs
    this automatically, because the ensemble-1000 and 24-step rows take minutes.
    """
    budget_rows = []
    for solver in SOLVERS:
        for row_spec in BUDGET_TABLE:
            row = measure_budget_row(*row_spec, solver=solver)
            print(format_budget_row(row))
            budget_rows.append(row)

    shape_gates = [measure_memory_shape_gate(), measure_matvec_shape_gate()]
    for gate in shape_gates:
        print(format_shape_gate(gate))

    report = {
        "machine": machine_info(),
        "reference_configuration": REFERENCE_KWARGS,
        "budget_table": budget_rows,
        "shape_gates": shape_gates,
        "solvers": list(SOLVERS),
        # The verdict is the DEFAULT solver's, not the best of the two: `all_budgets_met`
        # answers "does what ships meet its budgets", and a row measured under an
        # alternative backend that nothing selects cannot change that answer. The
        # alternative's rows are in `budget_table` beside it, tagged by `solver`.
        "all_budgets_met": all(
            row["forward_within_budget"]
            and row["backward_within_budget"]
            and row["peak_memory_within_budget"]
            for row in budget_rows
            if row["solver"] == SOLVERS[0]
        )
        and all(gate["within_budget"] for gate in shape_gates),
    }
    out = Path(__file__).parent / "composed_scaling_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
