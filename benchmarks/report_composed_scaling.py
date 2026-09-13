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
"""

from __future__ import annotations

from benchmarks.measure import isolated_peak_rss

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


def measure_budget_row(
    ensemble: int,
    steps: int,
    forward_budget: float,
    backward_budget: float,
    memory_budget: int,
) -> dict:
    """Measure one budget-table row: forward time, backward time, peak RSS, iteration counts.

    Two child processes, one per direction. `peak_memory_bytes` is the LARGER of the two
    peaks: the row's memory budget is a budget for the configuration, and a backward pass
    that needs more than the forward pass is still this row's peak.
    """
    kwargs = {**REFERENCE_KWARGS, "ensemble": ensemble, "steps": steps}
    peak_forward, forward = isolated_peak_rss(
        "benchmarks.composed_model", "workload_forward", kwargs
    )
    peak_backward, backward = isolated_peak_rss(
        "benchmarks.composed_model", "workload_backward", kwargs
    )
    peak = max(peak_forward, peak_backward)
    return {
        "ensemble": ensemble,
        "steps": steps,
        "n_nodes": forward["n_nodes"],
        "n_edges": forward["n_edges"],
        "forward_seconds": forward["elapsed_s"],
        "forward_budget_seconds": forward_budget,
        "forward_within_budget": forward["elapsed_s"] <= forward_budget,
        "backward_seconds": backward["elapsed_s"],
        "backward_budget_seconds": backward_budget,
        "backward_within_budget": backward["elapsed_s"] <= backward_budget,
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
        f"budget[ensemble={row['ensemble']}, steps={row['steps']}]: "
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
        f"method={row['method']}"
    )
