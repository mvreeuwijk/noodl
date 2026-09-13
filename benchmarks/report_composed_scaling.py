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


def _doubled_edge_operator(layer, slopes: torch.Tensor, seed: int = 0):
    """A `GraphLaplacianOperator` on the SAME nodes as `layer`, with twice as many edges.

    `build_composed` derives each submodel's extra-edge count from its node count, so it
    cannot be asked for twice the edges at the same node count. The extra edges are drawn
    here instead, by the same rule the builder's own `_random_tree_plus_extra` uses for its
    extra edges -- a uniformly random ordered pair of distinct nodes -- and given slopes
    resampled from the layer's own, so the doubled operator differs from the reference in
    edge count and nothing else. `matvec` is a gather/scatter over edges against a fixed
    node-space buffer, which is exactly what this gate is about.
    """
    n = layer._boundary_mask.shape[-1]
    b = int(layer._src.numel())
    rng = torch.Generator().manual_seed(seed)
    src_extra = torch.randint(0, n, (b,), generator=rng)
    tgt_extra = torch.randint(0, n, (b,), generator=rng)
    # No self-loops, matching the builder's `if u != v` rejection, without changing the count.
    self_loops = src_extra == tgt_extra
    tgt_extra[self_loops] = (tgt_extra[self_loops] + 1) % n
    perm = torch.randperm(b, generator=rng)
    return GraphLaplacianOperator(
        torch.cat([layer._src, src_extra]),
        torch.cat([layer._tgt, tgt_extra]),
        torch.cat([slopes, slopes[..., perm]], dim=-1),
        len(layer.interior),
        layer._interior_of_node,
        boundary_mask=layer._boundary_mask,
    )


def measure_matvec_shape_gate(repetitions: int = 200, medians_of: int = 5) -> dict:
    """Shape gate 2: `matvec` time at the reference edge count against twice that.

    The median of `medians_of` batches of `repetitions` matvecs each, per operator, because
    a single batch on a loaded machine is noise; the median is taken over batches rather
    than over individual calls so per-call timer resolution never enters the figure.
    """
    model = build_composed()
    layer = model.layer
    phi, _q = layer.solve(model.phi_boundary, model.drivers, model.sources, differentiable=False)
    slopes = layer.dflows(phi, model.drivers)
    operator = GraphLaplacianOperator(
        layer._src,
        layer._tgt,
        slopes,
        len(layer.interior),
        layer._interior_of_node,
        boundary_mask=layer._boundary_mask,
    )
    doubled = _doubled_edge_operator(layer, slopes)
    x = torch.zeros(
        slopes.shape[:-1] + (len(layer.interior),), dtype=slopes.dtype, device=slopes.device
    )

    def batch(op) -> float:
        elapsed, _ = time_call(lambda: [op.matvec(x) for _ in range(repetitions)])
        return elapsed

    batch(operator)  # warm up both before either is timed
    batch(doubled)
    t_small = median(batch(operator) for _ in range(medians_of))
    t_large = median(batch(doubled) for _ in range(medians_of))
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
    )
