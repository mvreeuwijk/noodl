"""The Milestone 1b acceptance gate: composed-model correctness and scaling (design section 6).

Fast tests (no marker) run in CI by default; `@pytest.mark.slow` tests are the full section 6.1
budget table and the two shape gates, deselected by the existing `-m "not slow"` addopts.

Memory here is measured with `benchmarks.measure.isolated_peak_rss`, never `tracemalloc`:
`tracemalloc` does not see PyTorch's own allocator at all (amendment A4), so a
`tracemalloc`-based memory gate would measure Python bookkeeping overhead and pass vacuously.
"""

from __future__ import annotations

from benchmarks.measure import isolated_peak_rss, time_call


def test_time_call_returns_a_nonnegative_elapsed_seconds_and_the_callables_result():
    elapsed, value = time_call(lambda: 2 + 2)
    assert elapsed >= 0.0
    assert value == 4


def test_isolated_peak_rss_sees_pytorch_allocations_a_200_mib_tensor_makes():
    # Non-vacuous by construction: the workload's only allocation is a torch tensor, which
    # `tracemalloc` reports as 0 bytes (amendment A4). A subprocess RSS measurement must see
    # it. 150 MiB, not 200, leaves room for the allocator returning pages in large blocks.
    peak_bytes, result = isolated_peak_rss(
        "benchmarks.composed_model", "workload_alloc", {"mb": 200}
    )
    assert result == 200
    assert peak_bytes >= 150 * 2**20
