# Solver selection: measurements and rationale

This page holds the measurements behind the default linear-solver choices in
`noodl.solvers.select`, `noodl.layers.transport` and `noodl.operators.graph`. The code keeps
the rule and the reason; the numbers live here, with how to reproduce them. Every number is a
CPU measurement (torch 2.14 CPU, 14 threads, Windows 11, Python 3.12, SciPy SuperLU) on the
reference composed model of `benchmarks/composed_model.py`: 1028 unknowns, about 5300 nonzeros,
float64. Re-measure when the platform or the problem size changes.

## Sparse-direct versus PCG

`method="auto"` sends a certified-SPD operator that offers a sparse form to sparse-direct
(SciPy SuperLU, one factorisation per instance), and otherwise to Jacobi-preconditioned CG.
Median of 3 warm runs, in process:

| Direction | Ensemble | pcg | sparse_direct | Ratio |
| --- | --- | --- | --- | --- |
| forward | 1 | 128.4 ms | 28.0 ms | 4.59x faster |
| forward | 100 | 2399 ms | 2182 ms | 1.10x (a tie) |
| forward | 1000 | 29489 ms | 19263 ms | 1.53x faster |
| backward | 1 | 44.9 ms | 13.8 ms | 3.25x faster |
| backward | 100 | 1741 ms | 476 ms | 3.66x faster |

At ensemble 100 the forward result is a tie: repeated, interleaved runs gave pcg 2068-2434 ms
against sparse_direct 2113-2182 ms. PCG needs 168-180 iterations per Newton step at this
conditioning; the factorisation needs one. The cost of sparse-direct is the per-instance Python
loop (SuperLU has no batched entry point), about 21 ms per instance on this model.

Reproduce: `python -m benchmarks.profile_forward --compare-solvers` (add `--forward-only` for
the forward rows).

### The ensemble threshold

`_SPARSE_DIRECT_MAX_BATCH = 32` caps the flat batch for which `auto` still factorises. Ratio
sparse_direct / cg for the forward solve (below 1.0 sparse-direct wins):

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ratio | 0.24 | 0.14 | 0.26 | 0.30 | 0.54 | 0.68 | 1.00 |

In the child-process acceptance gate (`benchmarks/composed_scaling_report.json`, whole forward
pass) cg is already ahead of an unthresholded `auto` at ensemble 100 (1.593 s against 1.686 s),
100x24 (39.4 s against 48.5 s) and 1000 (12.9 s against 15.2 s), while `auto` wins 2.2x at
ensemble 1 (0.097 s against 0.210 s). 32 is the last measured point at which the factorisation
is clearly ahead and 64 is the tie. The whole 32-64 band is within 1.0-1.5x, so choosing 32
rather than 64 costs little.

Reproduce: `python -m benchmarks.report_composed_scaling` regenerates the acceptance-gate
report.

## Transport solver choice

`TransportLayer(linear_solver="auto")` resolves to `sparse_direct` at or under
`_SPARSE_DIRECT_MAX_BATCH` instances and to plain `gmres` above it. `gmres_jacobi` and
`gmres_ilu` are never chosen by `auto`. The decision rests on the interleaved benchmark
`benchmarks/transport_solver_bench.py`, which cycles round-robin over the solvers within each
round so that ambient load cancels out of the ratios. It reports median, min and max rather
than the mean. Forward and backward medians in ms, from
`benchmarks/transport_solver_bench.json` (2026-09-24, 5 rounds per cell):

| Ensemble | Solver | Forward | Backward |
| --- | --- | --- | --- |
| 1 | gmres | 230 | 278 |
| 1 | gmres_jacobi | 272 | 316 |
| 1 | gmres_ilu | 288 | 285 |
| 1 | sparse_direct | 12 | 14 |
| 32 | gmres | 556 | 460 |
| 32 | gmres_jacobi | 365 | 540 |
| 32 | gmres_ilu | 712 | 937 |
| 32 | sparse_direct | 252 | 254 |
| 100 | gmres | 805 | 1093 |
| 100 | gmres_jacobi | 898 | 789 |
| 100 | gmres_ilu | 2048 | 1839 |

`sparse_direct` is not attempted at ensemble 100 (above the threshold). Reading the table:

- sparse_direct wins at ensembles 1 and 32 by a margin larger than the spread.
- `gmres_ilu` is the clear loser at every ensemble: a differentiable step calls `gmres` from
  both `_LinearSolve.forward` and `.backward`, so the per-instance ILU factorisation is paid
  twice per step and never reused, and GMRES still iterates afterwards.
- `gmres` against `gmres_jacobi` is ambiguous within the spread (each wins some cells), so
  there is no evidence to prefer a preconditioner as the fallback.

The JSON file carries a note: its run straddled a commit that made the reported `iterations`
of preconditioned gmres cycle-granular, so the `iterations` field of the `gmres_jacobi` and
`gmres_ilu` rows is stale. Timings are unaffected.

Reproduce: `PYTHONPATH=src python -m benchmarks.transport_solver_bench` (add `--quick` for a
short run). It writes the JSON file and prints a markdown table.

## Operator caches

`GraphLaplacianOperator` caches the broadcast batch shape and the source/target index tensors
expanded to it, per key. At the composed reference size, recomputing them on every matvec cost
about 11% of `pcg`'s time (820 `broadcast_shapes` calls per solve). Only index tensors are
cached, never an expanded view of `slopes`: `slopes` can require grad, and a cached view would
carry one autograd graph's `ExpandBackward` into the next forward pass.

Reproduce: profile `pcg` on the composed model with `python -m benchmarks.profile_forward`.

## Other empirical facts kept in the code

These are small and stay next to the code that depends on them; they are listed here so that
they can be found.

- `solvers/iterative.py`: the Givens breakdown threshold `eps ** 0.75` sits about 3 orders
  above every measured breakdown ratio (2e-16 to 1e-14) and about 2 orders below the smallest
  legitimate ratio over 1000 random dense systems of size 4 to 64 (2.6e-10).
- `solvers/iterative.py`: an ill-conditioned but nonsingular system (cond 1e8, m = 4, 8, 16)
  routinely ends its last Arnoldi step under the relative threshold, which is why `SINGULAR`
  is only reported for a breakdown that truncates the cycle.
- `solvers/newton.py`: the iterative solvers' own floor on the 2x2 float32 test system is
  3e-8, which is why an explicit dense Jacobian keeps `method="direct"`.
