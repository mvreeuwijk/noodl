# Sparse-by-default operators for `tellegen`: research and benchmark report

> Status (12 Sep 2026): drafted by an AI agent (Claude) from web sources, direct
> introspection of this repo's `.venv` (torch 2.14.0+cpu), and a benchmark script at
> `benchmarks/sparse_scaling.py`. Answers the open question in
> `docs/superpowers/specs/2026-09-11-tellegen-framework-design.md` section 4.0: "the
> choice between a formed sparse Jacobian and a matrix-free operator is being settled by
> measurement." Sections 4-5 are measured on this machine (CPU only, no CUDA); GPU claims
> elsewhere are from cited literature, marked as such.
>
> **Corrected 12 Sep 2026** following an independent review
> (`docs/sparse_operators_review.md`) that found several conclusions below overstated
> what the benchmark actually showed. See the Errata section immediately below for what
> changed and why; the corrections are applied in place throughout the body so this
> document is internally consistent — the errata is a map for readers of the prior
> version, not a patch that needs to be read alongside contradicting text.

## Errata (12 Sep 2026)

1. **Withdrawn: "never `torch.sparse`."** The original benchmark only tested a *formed,
   batched COO Jacobian* against gather/scatter. It never tested a shared, non-batched
   2-D sparse incidence matrix applied to dense per-instance values — which needs no
   index replication, agrees with gather/scatter to floating point, and passes
   `gradcheck` against both conductances and potentials. See sections 1, 4-5, 7(b).
2. **Corrected: the dense "OOM" was a self-imposed skip, not a measured failure.**
   `n=300, B=1000` was never run; it was estimated against a 400 MB budget and skipped.
   No allocation failure was ever captured. Independent counter-measurements (chunked
   dense, sequential sparse-direct) are added. See sections 4-5, 7(c).
3. **Corrected: the `spsolve` CPU gap is not "documented."** Upstream dispatch registers
   CUDA and XPU kernels only; there is no CPU registration at all, which is the opposite
   of a documented CPU-only restriction — and it means a sparse direct solve may well
   exist on a CUDA/XPU wheel (untested here). See section 1.
4. **Corrected: index replication does not negate sparse's scaling advantage**, and
   replication is not intrinsic to vendor sparse-direct batching (cuDSS supports a
   shared pattern with per-instance value buffers). See section 6.
5. **Narrowed: CG is validated only for confirmed-SPD potential blocks**, not adopted as
   the framework's general solver; transport and other nonsymmetric/coupled operators
   need their own forward/transpose actions and solver choice. See sections 3(a), 7(b)-(c).
6. **Corrected: Jacobi-preconditioned CG converged with zero failed instances** at 6 and
   8 conductance decades (445 and 813 iterations) where plain CG failed on 9/100 and
   100/100 instances — the original text implied Jacobi also broke down. See section 4-5.
7. **Corrected: the "7.7 s / 26 GB" transport figures are attributed to the earlier
   design-spec review**, not reproduced by this sparse benchmark. See section 7(c).

Everything not listed above — the sparse requirement itself, shared edge indices, the
transport-first priority, the assembly crossover measurements, and the raw benchmark
numbers — stands as originally reported.

## 1. The state of `torch.sparse` today

PyTorch ships five sparse layouts — COO, CSR, CSC, and the block variants BSR/BSC, plus a
prototype 2:4 semi-structured format for NVIDIA GPUs — and the whole module is still
officially beta: "The PyTorch API of sparse tensors is in beta and may change in the near
future" ([torch.sparse docs](https://docs.pytorch.org/docs/stable/sparse.html)). Autograd
support is narrow: the docs list `torch.sparse.mm`, `torch.sparse.addmm`, and the
low-rank `pca_lowrank`/`svd_lowrank` as carrying gradients through sparse COO tensors;
almost everything else does not.

Batching is where the gap between the documented model and this repo's needs is widest.
The docs describe batched sparse compressed tensors as requiring "the same number of
specified elements per batch entry" — tellegen's shared-topology case satisfies this
trivially. Verified directly in this `.venv`, that is not the binding constraint:

- A batched CSR tensor built the documented way (`crow_indices`/`col_indices` given an
  explicit leading batch dimension, `values` of shape `(B, nnz)`) constructs without
  error, but `S @ vec`, `torch.bmm(S, vec)`, `torch.sparse.mm(S, vec)` and
  `torch.matmul(S, vec)` **all four** raise `RuntimeError: expand is unsupported for
  SparseCsr tensors` — no working batched CSR matvec exists in this build.
- Batched COO does work: `torch.sparse_coo_tensor` with an explicit batch dimension,
  `torch.bmm` as the matvec, gradients flowing through the values. But COO batching is
  not "shared indices, batched values" — every batch entry's `(row, col)` pairs must be
  listed, so a `(B, n, n)` batched COO Laplacian with `nnz` structural nonzeros costs
  `B * nnz` index entries, not `nnz`. This is exactly the "block-diagonal embedding"
  independently used by a 2026 GPU power-flow accelerator, SABLE, to batch sparse
  Jacobians for cuDSS ([arXiv:2606.07099](https://arxiv.org/abs/2606.07099)) — the
  standard workaround, but it forfeits the "store indices once" saving that motivates
  going sparse at all.
- `torch.sparse.spsolve` is CSR-only. The cited docs do **not** say CPU-only — an
  earlier version of this report claimed that, and it was wrong. Checked against
  upstream dispatch registration (`aten/src/ATen/native/native_functions.yaml`):
  `_spsolve` is registered for `SparseCsrCUDA` (`_sparse_csr_linear_solve`) and
  `SparseCsrXPU` (`_sparse_csr_linear_solve_xpu`) — there is **no CPU registration at
  all**. That is exactly why it does not run, forward or backward, batched or not, in
  this build: `NotImplementedError: Could not run 'aten::_spsolve' with arguments from
  the 'SparseCsrCPU' backend.` This matches a tracked upstream bug,
  [pytorch/pytorch#160813](https://github.com/pytorch/pytorch/issues/160813). The
  consequence is the opposite of what a "CPU-only" reading would suggest: a sparse
  direct solve may well be available on a CUDA or XPU wheel — which matters, since this
  framework targets GPU — but that was not tested here; no GPU was available on this
  machine, so this is a documented-dispatch fact, not a measured GPU result. Where it
  does run, it is non-differentiable by construction — no registered backward.

Net: a `torch.sparse`-first design built on **batched, per-instance sparse tensors**
gives tellegen a beta API, no batched CSR matvec that runs, no batched or
differentiable direct solve, and a `spsolve` broken in the exact environment this
project ships — using natural calling conventions, not edge cases. This does **not**
rule out `torch.sparse` altogether: sections 4-5 and 7(b) show a *shared, non-batched*
2-D sparse incidence matrix applied to dense per-instance values is a distinct, working,
differentiable path that this report originally failed to test.

## 2. Sparse linear solves in PyTorch: the honest gap

`torch.sparse.spsolve` is the only first-party sparse direct solve: registered only for
CUDA/XPU backends (no CPU kernel at all — section 1), non-differentiable, unbatched, and
non-functional on this CPU-only build as a direct consequence. `torch.linalg`
has no sparse-aware entry points — `torch.linalg.solve` always materialises a dense matrix.

GPU sparse-direct solving exists as a *library*, not a PyTorch feature: NVIDIA cuDSS
([docs](https://docs.nvidia.com/cuda/archive/12.9.1/cudss/index.html)) and CuPy's
`cupyx.scipy.sparse.linalg.spsolve` (wrapping cuSolverSp) interoperate with PyTorch
tensors at zero copy via `__cuda_array_interface__`/DLPack, but neither plugs into
autograd: a PyTorch forum thread attempting exactly this reports "unsupported tensor
layout: Sparse" inside `gradcheck`
([discuss.pytorch.org/t/141309](https://discuss.pytorch.org/t/differentiable-sparse-linear-solver-with-cupy-backend-unsupported-tensor-layout-sparse-in-gradcheck/141309)).
Differentiating through such a solve means hand-writing a `torch.autograd.Function` that
calls the external solver forward and does its own adjoint solve backward — precisely
tellegen's own `implicit_solve` pattern (`src/tellegen/solvers/implicit.py`), with the
linear algebra swapped out.

A brand-new project, torch-sla ([arXiv:2601.13994](https://arxiv.org/html/2601.13994v2)),
exists because of this gap: "`torch.sparse.spsolve` is non-differentiable, CPU-only, and
restricted to square systems" (their "CPU-only" phrasing repeats the same imprecision
section 1 corrects above — dispatch is CUDA/XPU-only, with no CPU kernel registered at
all — but their operative claim, non-differentiability, is independently confirmed
here); naive backprop through k CG iterations costs `O(k)` graph
memory (~80 GB for a 1M-DOF problem at 1000 iterations); "batched solves over shared
sparsity patterns are unsupported" natively. Its fix is the one this project already
committed to: implicit-function adjoint (one extra solve, `O(1)` graph nodes) —
corroborating tellegen's own section-4.0 note almost verbatim, from an unrelated project
three months old.

**Conclusion:** no differentiable, batched, sparse *direct* solve exists in stock
PyTorch today. Third-party wrappers (torch-sla,
[torchsparsegradutils](https://pypi.org/project/torchsparsegradutils/)) close part of
the gap but add a dependency of unknown maturity. The matrix-free route below avoids the
problem rather than working around it.

## 3. The matrix-free alternative

For `J = A_I diag(g) A_I^T`, forming `J` is never necessary — only its action `J @ x`:
gather (`x[src] - x[dst]`), scale (`* g`), scatter (`index_add` at `src`, negated at
`dst`). Four claims, checked:

**(a) Positive semidefinite in general; positive definite needs strictly positive
slopes, per instance.** For potential-driven branch laws with `g >= 0`, `J` is a
weighted graph Laplacian restricted to interior nodes (boundary/ground rows dropped):
symmetric by construction, and positive SEMI-definite whenever every interior node has
a path to a boundary node in the unweighted graph. Positive DEFINITENESS is a stronger,
per-instance condition: it additionally requires every interior component to reach a
prescribed potential through STRICTLY POSITIVE slopes IN THAT INSTANCE — unweighted
connectivity is not sufficient, since a zero slope on a bridging edge (e.g. a fully
closed damper) can leave a component effectively floating even though the graph itself
is connected. Tellegen's existing diagnostic, `PotentialFlowLayer._floating_group_nodes`,
is narrower than "detects and rejects the floating case" as originally stated here: it
checks `(k_flat != 0).any(dim=0)`, the union of active edges **across the whole batch**,
using linear-initialisation slopes, and is bypassed entirely when the caller supplies
`phi0`. A three-node chain grounded at node 0 with per-instance slopes `[[1,1],[0,1]]`
reports no floating nodes even though the second instance's Jacobian has an exactly-zero
eigenvalue (independently verified). CG is appropriate where SPD is actually established
for the instance at hand; a genuine per-instance, not per-batch-union, singularity check
is needed before relying on it, and this does not extend to operators that are not SPD in
the first place (section 7(b)-(c)).

**(b) Batches perfectly.** The action is elementwise: `edge_index` is `(2, b)`, shared;
`g` is `(B, b)`; gather is fancy indexing (`x_full[..., src]`), scatter is `index_add`
along the last axis with a `(B, b)` source — both apply uniformly across the batch.

**(c) Trivially differentiable, same operator for the adjoint.** Every op in the matvec
carries ordinary autograd; the backward solve is `J^T lam = grad_x`, and since `J` is
symmetric, `J^T = J` — the same matvec serves the backward CG unmodified, Tellegen
reciprocity doing structural work.

**(d) Preconditioning.** A Jacobi (diagonal) preconditioner is a one-line scatter of `g`
into node degrees, costing nothing extra to batch; measured below, it cuts iteration
counts substantially but does not fix ill-conditioning from *heterogeneous* conductances
(a near-closed damper beside an open one raises the condition number by the conductance
ratio regardless of diagonal scaling — a topological, not per-node, disagreement about
scale). Incomplete Cholesky and algebraic multigrid (AMG) are the standard next steps for
graph Laplacians — PyAMG ([github](https://github.com/pyamg/pyamg),
[OSTI](https://www.osti.gov/pages/biblio/2433956)) and the Lean AMG family
([arXiv:1108.1310](https://arxiv.org/pdf/1108.1310),
[arXiv:2606.24791](https://arxiv.org/html/2606.24791)) are mature — but neither is a
PyTorch operation: both are SciPy/NumPy CPU libraries with no autograd or batching, so
using either means building the hierarchy once per topology, outside the autograd graph,
and applying it as a fixed preconditioner inside an otherwise-differentiable CG — a
defensible, common pattern (exactly how "deep learning of preconditioners for CG" for
urban water networks frames it, [arXiv:1906.06925](https://arxiv.org/pdf/1906.06925)),
but real engineering, not a drop-in. Jacobi is measured below to already be enough for
tellegen's near-term sizes; AMG earns its complexity past that, mainly for extreme
conductance ratios, not size alone.

## 4-5. Benchmark results

Script: `benchmarks/sparse_scaling.py`. Raw logs in `benchmarks/`:
`sparse_scaling_results_partA.txt` (full `n x B` sweep), `_rest.txt` (conductance-spread
sweep, `spsolve` attempt, matvec-cost micro-benchmark), `_partC.txt` (assembly cost). CPU
only, torch 2.14.0+cpu, float64. Graphs: random spanning tree plus `n` extra edges (~`2n`
total), node 0 grounded, conductances log-uniform over 6 decades unless stated. Columns
are wall-clock seconds; iterations in parentheses; "(capped)" = iteration cap was hit.
Iteration caps differ between methods and between this sweep and the spread experiment
below, so cross-method iteration/time comparisons in this table are not apples-to-apples
(see the conditioning paragraph below). "Skipped" `dense` cells are pre-run ESTIMATES
from the assembly-size formula, gated by a fixed memory budget, not measured allocation
failures — see below.

| n | B | dense s (mem) | cg s (it) | cgJ s (it) | sparse s (it) |
|---|---|---|---|---|---|
| 50 | 1 | 0.0005 (211 MB) | 0.017 (132, converged) | 0.005 (43) | 0.012 (139) |
| 50 | 1000 | 0.011 (283 MB) | 0.80 (223, converged) | 0.30 (62) | 1.22 (capped) |
| 300 | 1 | 0.007 (469 MB) | 0.16 (800, **not converged**) | 0.05 (369, converged) | 0.28 (800) |
| 300 | 1000 | **skipped, 1.4 GB** | 1.48 (150, capped) | 5.61 (250, capped) | 3.87 (capped) |
| 1000 | 1 | 0.11 (571 MB) | 0.50 (800, **not converged**) | 0.56 (861, converged) | 0.47 (800) |
| 1000 | 1000 | **skipped, 16 GB** | 6.82 (150, capped) | 16.3 (250, capped) | 15.3 (capped) |
| 3000 | 1 | 1.65 (1.4 GB) | 0.46 (400, **not converged**) | 0.66 (600, capped) | 1.09 (800) |
| 3000 | 100 | **skipped, 14.4 GB** | 1.53 (150, capped) | 2.78 (250, capped) | 5.50 (capped) |
| 3000 | 1000 | **skipped, 144 GB** | not measured (see below) | not measured | not measured |

Full table (all `B` at every `n`) is in the raw log. Findings below are MEASURED unless
marked ESTIMATED or SKIPPED:

**Dense was SKIPPED past a self-imposed ESTIMATE threshold on `B * n_i * b`, not
measured to OOM.** The benchmark estimates the dominant `(B, n_i, b)` broadcast
temporary inside assembly (not the final `(B, n_i, n_i)` Jacobian) and SKIPS the dense
path whenever that estimate exceeds `DENSE_MEM_BUDGET_BYTES = 4e8` (400 MB;
`benchmarks/sparse_scaling.py`). No allocation failure was ever captured: at `n=300,
B=1000` the "1.4 GB" figure is an ESTIMATE of one intermediate, not a measured process
peak, and dense was never actually run there. The script's own comments confirm this
was a deliberate, tightened-down gate to avoid long wall time on this machine, not a
response to an observed crash — an earlier, looser 1.5 GB budget "produced a single
combo that took an unreasonable amount of wall time," not an OOM. The `dense_mem_MB`
figures reported elsewhere in the table are also PROCESS-LIFETIME peaks that include
earlier benchmark paths (cg, sparse, etc.) sharing the same process, so even the
MEASURED entries cannot attribute peak memory to dense alone. Two independent
counter-measurements at exactly `n=300, B=1000` show the estimate does not establish a
hard wall: the ORIGINAL dense assembly, run in chunks of 32 instead of all 1000 at once,
solved the full ensemble in 3.92 s with a maximum relative residual of 1.84e-10 and only
a 45.8 MB weighted temporary per chunk; a sequential SciPy sparse-direct loop over the
same 1000 systems (assembly, factorisation and solve per instance, no shared setup, no
custom backward implemented) took 1.15 s with a maximum relative residual of 7.78e-11.
Dense's `O(B * N^2)` storage still makes it unsuitable as the large-scale default — that
conclusion stands, and by `n=3000` the ESTIMATED figures (14.4 GB at `B=100`, 144 GB at
`B=1000`) are large enough that no counter-measurement is needed to doubt dense there —
but the specific `n=300, B=1000` OOM/threshold claim was not established by this
benchmark.

**Unpreconditioned CG stops converging past `n~300` at any batch size; Jacobi-
preconditioned CG converged in every case tested here, including at 6 and 8 conductance
decades — an earlier "Jacobi degrades too" reading of this same data was wrong.** The
conductance-spread sweep (`n=300, B=100`, 100 instances per decade) shows the mechanism:
plain CG needs 56 iterations at 0 decades of spread and 182 at 2, then hits its
3000-iteration cap and FAILS outright (recomputed true relative residual >= 1e-6) on
9/100 instances at 6 decades and 100/100 at 8. Jacobi-preconditioned CG needs only 35,
76, 445 and 813 iterations at those same four spread levels, respectively, and converged
with ZERO failed instances at every one of them, including 6 and 8 decades. So Jacobi did
not degrade to the point of failing in this sweep — it converged. What actually happens
in the main `n x B` table above is that `cgJ`'s cap there (250) is far lower than the 813
iterations Jacobi needed at 8 decades in the spread sweep, so its "capped" entries above
`n=1000` or `B=1000` reflect an insufficient cap chosen for that experiment, not a
demonstrated Jacobi failure — and because plain CG's and Jacobi's iteration caps differ
between experiments (and from each other), none of these are directly comparable
times-to-solution. This still confirms the design spec's underlying prediction that
iterative solves degrade on extreme conductance ratios — but conductance ratio alone does
not determine difficulty: an 8-decade GROUNDED STAR converges in a single Jacobi
iteration (independently reproduced), because every interior node sits one hop from
ground. What would actually establish robustness is testing chains, spatial networks,
weak interfaces between submodels, changing active-edge sets, realistic branch-law
derivative ranges, and equal stopping criteria across methods — not a random-tree
topology under a single uniform cap. This is where AMG (section 3d) earns its
complexity, and where stronger preconditioning or a direct fallback should be evaluated
before milestone 2.

**`torch.sparse`'s BATCHED, PER-INSTANCE-REPLICATED COO matvec is correct but
consistently slower than gather/scatter here, by a margin growing with size — and that
is why the sweep above has gaps.** A controlled micro-benchmark (30 raw matvec calls, no
CG) gives per-call cost directly for this specific construction — one sparse COO tensor
of shape `(B, n_i, n_i)` with `B * nnz` index entries, i.e. topology replicated across
the batch: within 5x either way at `n=50` (overhead dominates), reaching 15x slower at
`n=300, B=1000` (5.86ms vs 88.8ms) and 9-15x from `n=1000` up. `n=3000, B=1000` — 24
million nonzeros once COO's per-batch index replication is paid for — did not complete
even 30 iterations within this benchmark's time budget and was abandoned; that gap is
itself the measurement. Where both methods reached real convergence (`n<=100`), answers
agreed to `1e-6`-`1e-9`.

**This benchmark never tested the other `torch.sparse` construction — a SHARED,
non-replicated 2-D sparse incidence matrix applied to dense batch columns — and an
independent check found it competitive, and sometimes faster.** Instead of forming a
per-instance `(B, n_i, n_i)` Jacobian, `A_I` and `A_I^T` can be kept as ordinary,
non-batched 2-D sparse tensors (shared indices, `O(E)` storage, never replicated) and
applied to the batch as dense right-hand-side columns:

```python
differences = torch.sparse.mm(A_I_T, X.T)
JX = torch.sparse.mm(A_I, G.T * differences).T
```

Here `G` (the per-instance conductances) varies by batch member, but the sparse INDICES
never do, and the Jacobian is never formed. Independently verified: this agrees with
gather/scatter to floating point, and `torch.autograd.gradcheck` passes with respect to
both `G` and `X`. MEASURED medians (milliseconds per matvec, construction excluded):

| n | B | threads | gather/scatter | shared CSR incidence |
|---|---|---|---|---|
| 300 | 100 | 1 | 0.4474 | 0.2778 |
| 300 | 100 | 14 | 0.4539 | 0.6893 |
| 1000 | 100 | 1 | 3.3825 | 1.2114 |
| 1000 | 100 | 14 | 0.7659 | 1.8058 |

Shared CSR beats gather/scatter by roughly 1.6-2.8x at 1 thread (both sizes shown), but
loses by roughly 1.5-2.4x at 14 threads — performance depends on thread count, size and
layout, not a fixed ranking either way. **Conclusion: shared-index sparse is a viable,
gradient-correct backend whose performance is workload-dependent, not a categorically
inferior or banned one; gather/scatter remains the sensible initial default because it
is simpler and is the one exercised end-to-end (through CG and the adjoint) in this
report, not because `torch.sparse` fails outright.**

**`torch.sparse.spsolve` never ran at all** — trivial 4x4 case, forward-only, no
autograd — the identical `NotImplementedError` from section 1 every time.

**Assembly (section 5) has a real, large crossover.** Comparing `A @ q` (dense matmul vs.
`index_add`) and `A^T @ phi` (dense matmul vs. gather): within 2x either way at
`n=100-300` (small-matrix overhead dominates); 20-60x faster for gather/scatter by
`n=3000`; 70-370x by `n=10000`, at both `B=1` and `B=1000` — matching theory (`O(n*b)` vs
`O(b)`) and landing close to tellegen's target scale, not a large-`n` curiosity.

## 6. What other differentiable-physics projects do

The dominant pattern, from several independent directions, is exactly tellegen's own:
**shared edge-index arrays, batched dense values, gather/scatter as the only sparse
primitive, implicit-function differentiation for anything solved iteratively.**

- **PyTorch Geometric**: "PyG makes heavy usage of gather and scatter operations to map
  node and edge information into edge and node parallel space"
  ([Creating Message Passing Networks](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)).
  A GNN layer and tellegen's `A_I q`/`A^T phi` are the same primitive on the same data
  structure; PyG found scatter/gather competitive with optimised SpMM up to average node
  degree ~128 ([arXiv:1903.02428](https://arxiv.org/pdf/1903.02428)) — well above
  tellegen's ~4 (a 300-node, 600-branch building).
- **JAX**'s `jax.experimental.sparse` (BCOO) is explicitly batchable, jittable,
  differentiable ([docs](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)) —
  ahead of `torch.sparse` on paper — but serious sparse *solves* still route through
  `lineax` with a custom operator (e.g. `splineax`'s `BCOOLinearOperator`), not a native
  batched sparse solve: "wrap an external solver, differentiate via IFT" again.
- **cvxpylayers** differentiates via the implicit function theorem on the KKT residual
  map, using LSQR for sparse problems to avoid forming a dense KKT Jacobian
  ([Differentiable Convex Optimization Layers](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf))
  — IFT plus a matrix-free solve, not a direct factorisation.
- **Differentiable power flow**, tellegen's closest analogue, converged on the same
  shape: SABLE is "an implicit power flow layer," batching Jacobians via a block-diagonal
  sparse template shared across PyTorch/CuPy/cuDSS, citing up to 206x training throughput
  over dense batching ([arXiv:2606.07099](https://arxiv.org/abs/2606.07099));
  Differentiable Power-Flow Optimization reports the same batching-plus-IFT recipe
  ([arXiv:2603.28203](https://arxiv.org/abs/2603.28203)). No differentiable-EPANET
  project of comparable maturity was found; closest is learned CG preconditioners for
  urban water networks ([arXiv:1906.06925](https://arxiv.org/pdf/1906.06925)), which
  assumes tellegen's own matrix-free-CG structure.

Most projects surveyed that both batch and differentiate over shared topology land on
gather/scatter (or a hand-rolled matrix-free operator) plus IFT — but not all: SABLE
itself uses genuine sparse-direct LU via cuDSS, not gather/scatter, so this is not a
universal rule and should not be read as "every differentiable batched project chooses
matrix-free iterative solves." SABLE and torch-sla batch by replicating the sparse
pattern per instance (the block-diagonal embedding of section 1), which costs a constant
memory factor — but that replication is a choice these two projects made, not a
requirement of vendor sparse-direct solving in general: NVIDIA cuDSS documents uniform
batching with a SHARED sparsity pattern and a buffer of per-instance values, keeping
analysis and numerical factorisation as separate phases
([cuDSS docs](https://docs.nvidia.com/cuda/cudss/types.html)) — so index-replicated
batching is one option for reaching GPU sparse-direct solvers, not an unavoidable cost.
Either way, batch-replicated indices remain `O(B*E)` on sparse graphs — a constant
factor above `O(E)`, but still far better than dense's `O(B*N^2)` — so replication does
not negate the main advantage of going sparse; it only forfeits the "store indices once"
saving for the batch dimension specifically.

## 7. Recommendation

**(a) Topology operators — switch now, unconditionally.** Represent
`incidence()`/`difference()`/`upwind()` as a shared `edge_index: (2, b)` tensor plus
per-kind slices, as the design spec's section 4.0 table already specifies. `A_I q` and
`A^T phi` become `index_add`/fancy-indexing gather. Section 5 (20-60x faster by `n=3000`,
70-370x by `n=10000`) shows this wins well before target scale, since dense does `O(n*b)`
work for what gather/scatter does in `O(b)`. Pitfall: use out-of-place
`index_add`/`scatter_add` into a freshly created zero tensor, never `index_add_` on a
live leaf — confirmed directly in this venv, the in-place form on a leaf raises
immediately; on a fresh non-leaf buffer it is fine.

**(b) The potential-flow Jacobian and its solve — matrix-free CG as the initial backend,
not a mandated formed sparse matrix.** For *batched, per-instance* sparse tensors,
`torch.sparse` is not ready: no working batched CSR matvec, no differentiable, batched,
or even functional direct solve on CPU. But a *shared, non-batched* 2-D sparse incidence
matrix applied to dense per-instance values (`torch.sparse.mm(A_I, ...)`) is a working,
gradient-correct alternative — see sections 1 and 4-5 — whose performance relative to
gather/scatter depends on thread count, size and layout, not a universal winner or
loser. Matrix-free CG needs no dependency, batches like the topology operators, and
reuses the *same* matvec for the implicit-function adjoint since `J` is symmetric **for
confirmed-SPD potential blocks** — this does not extend to transport or other
nonsymmetric operators (sections 3(a), 7(c)). Ship Jacobi preconditioning from day one
(section 4-5: it converged with zero failed instances at 6 and 8 conductance decades,
where plain CG failed on 9/100 and 100/100 instances respectively); treat AMG as a later
addition gated on extreme conductance ratios actually appearing, not on node count. Keep
the shared-index sparse-mm backend and sparse-direct solving (section 2) as evaluated
alternatives behind the same operator interface, not excluded options.

**(c) The transport layer — matters most, and first.** The 7.7 s / 26 GB dense
transport projection is attributed to the earlier design-spec review, not to this
sparse benchmark: this benchmark did not run the transport operator at all, so those
figures are cited here, not reproduced. `TransportLayer.operator()`
(`src/tellegen/layers/transport.py`) builds `Out`, `In`, `L` as dense `(..., n, n)` (or
`(..., K, n, n)`) tensors via `einsum` over one-hot selectors — this should become
gather/scatter like the potential-flow Jacobian, with `_van_loan_step`'s matrix
exponential restricted to sparsity-permitting cases or replaced by a Krylov-subspace
exponential-action method (`expm(M) @ v` without forming `expm(M)`), since
`torch.linalg.matrix_exp` has no sparse path either. Unlike the potential-flow block,
signed advection is generally NONSYMMETRIC, as are general species kinetics and general
multiphysics coupling: transport's implicit, trapezoidal and steady solves cannot
inherit the potential-flow CG assumption or reuse the forward matvec as its own
transpose. They need their own forward and transpose actions and their own solver
choice — e.g. GMRES, or a sparse direct solve — evaluated separately from
potential-flow PCG. This is milestone 1b's real payload: the potential-flow Jacobian
solve is the number this benchmark actually characterises, but transport is the
component believed to dominate cost at scale, on the earlier spec's projection, not
this benchmark's measurement.

**What can stay dense.** Anything genuinely `O(n)`-small and not batched at scale:
boundary selectors, single-instance debugging, tests. `torch.linalg.solve` remains right
while `B * n_i * b` stays small — fine at `n<=100` for any `B` up to 1000, and at `n=300`
up to `B=100`; at `n=300, B=1000` and beyond, this benchmark did not run dense at all —
it was skipped against a self-imposed, estimate-based 400 MB budget, not a measured
allocation failure (section 4-5) — though its `O(B * N^2)` storage still makes it
unsuitable as a large-scale default regardless of exactly where a measured limit would
fall.

**What would break the adjoint.** Nothing structural: `implicit_solve` only needs
`jacobian(x, *params)` to return something `adjoint()` can solve against, and `adjoint()`
is just `torch.linalg.solve(J.T, grad_x)`. An earlier draft called swapping that call for
matrix-free CG "like-for-like"; that was too simple, and the milestone 1b design corrects it.
Three reasons: the Newton update, the linear initialisation and the backward solve all consume
the same dense `J` and must migrate TOGETHER; the forward matvec may serve as the transpose
only where the operator is symmetric, which transport is not; and CG requires positive
definiteness, which symmetry alone does not establish. The one real risk: `linear.py`'s `floating_nodes`
diagnostic inspects `J`'s dense rows to name disconnected nodes; with no formed `J` it
must move to the edge-index level, where `_floating_group_nodes` in `potential.py`
partially does this today — though as section 3(a) notes, that check is narrower than a
full per-instance singularity diagnostic (it unions active edges across the whole batch
and is bypassed when `phi0` is supplied), so `linear.py`'s dense-only version should be
retired only once replaced by a genuine per-instance check, not simply ported as-is.
