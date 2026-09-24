"""Smoke test for `benchmarks.transport_solver_bench`: `--quick` runs in-process and produces
one row per solver with the expected keys.

Run as a normal test (not a script): pytest's own `pythonpath = ["src", "."]` setting
(`pyproject.toml`) already resolves `noodl` to this worktree, so no `PYTHONPATH` juggling is
needed here the way the module's own docstring requires for a standalone run.
"""

from __future__ import annotations

import json

from benchmarks.transport_solver_bench import (
    SOLVERS,
    bench_composed_ensemble,
    machine_block,
    main,
)

# Measured at ~9.5 s (pytest-reported) on this machine -- under the brief's 20 s threshold for
# the `slow` marker (registered in `pyproject.toml`, excluded by default via
# `-m "not slow"`), so this test is left unmarked and runs in the default suite.


def test_quick_run_writes_one_row_per_solver_with_expected_keys(tmp_path) -> None:
    """`main(["--quick"])` (redirected to a scratch JSON) reports every solver at ensemble 1,
    each with forward and backward rows carrying `median_s`/`min_s`/`max_s`/`iterations`, and
    a machine block with `noodl_file` pointing at THIS worktree's `noodl` (not the main
    checkout's), which is the whole point of the benchmark's `PYTHONPATH` requirement for a
    standalone run -- in-process, pytest's own `pythonpath` setting already gets this right,
    so this assertion is the regression guard that a future refactor cannot silently break it.
    """
    out = tmp_path / "quick.json"
    rc = main(["--quick", "--skip-coupling", "--out", str(out)])
    assert rc == 0

    result = json.loads(out.read_text())
    assert set(result) == {"machine", "quick", "composed", "coupling"}
    assert result["quick"] is True
    assert result["coupling"] == []

    machine = result["machine"]
    expected_machine_keys = (
        "torch_version", "num_threads", "platform", "timestamp", "noodl_file", "git_commit",
    )
    for key in expected_machine_keys:
        assert key in machine
    assert "noodl" in machine["noodl_file"]

    rows = result["composed"]
    seen_solvers = {row["solver"] for row in rows}
    assert seen_solvers == set(SOLVERS)
    for solver in SOLVERS:
        solver_rows = [r for r in rows if r["solver"] == solver]
        if any(r.get("not_applicable") for r in solver_rows):
            continue
        directions = {r.get("direction") for r in solver_rows if "direction" in r}
        if any("error" in r for r in solver_rows if "direction" not in r):
            continue
        assert directions == {"forward", "backward"}
        for r in solver_rows:
            if "direction" not in r:
                continue
            if "error" in r:
                continue
            for key in ("median_s", "min_s", "max_s", "n"):
                assert key in r
            assert r["n"] == 1  # --quick: one round


def test_machine_block_has_the_expected_keys() -> None:
    """Unit-level check of `machine_block` alone, independent of a full `--quick` run."""
    machine = machine_block()
    assert set(machine) == {
        "torch_version", "num_threads", "platform", "python_version", "timestamp", "noodl_file",
        "git_commit",
    }
    assert isinstance(machine["num_threads"], int)
    # `git_commit` is `None` only when git itself is unavailable; this worktree has git, so a
    # short hash is expected here.
    assert machine["git_commit"] is None or isinstance(machine["git_commit"], str)


def test_bench_composed_ensemble_reports_a_row_per_solver_and_direction() -> None:
    """One round at ensemble 1, called directly (no JSON round trip), for a faster, more
    targeted check than the full `--quick` smoke test above."""
    rows = bench_composed_ensemble(1, rounds=1)
    seen = {(r["solver"], r.get("direction")) for r in rows if "direction" in r}
    expected = {(solver, direction) for solver in SOLVERS for direction in ("forward", "backward")}
    # sparse_direct is applicable at ensemble=1 (below the batch cap), so all four should
    # produce both directions -- unless a solver genuinely failed, which would show as an
    # "error" row with no "direction" key instead, so a strict equality here is the right
    # assertion, not merely a subset check.
    assert seen == expected
