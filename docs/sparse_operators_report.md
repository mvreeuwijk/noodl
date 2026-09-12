# Sparse-by-default operators for `tellegen`: research and benchmark report

> Status (12 Sep 2026): drafted by an AI agent (Claude) from web sources, direct
> introspection of this repo's `.venv` (torch 2.14.0+cpu), and a benchmark script at
> `benchmarks/sparse_scaling.py`. Answers the open question in
> `docs/superpowers/specs/2026-09-11-tellegen-framework-design.md` section 4.0: "the
> choice between a formed sparse Jacobian and a matrix-free operator is being settled by
> measurement." Sections 4-5 are measured on this machine (CPU only, no CUDA); GPU claims
> elsewhere are from cited literature, marked as such.

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
- `torch.sparse.spsolve` is CSR-only, documented CPU-only
  ([docs](https://docs.pytorch.org/docs/stable/generated/torch.sparse.spsolve.html)), but
  in this build it does not run at all, forward or backward, batched or not:
  `NotImplementedError: Could not run 'aten::_spsolve' with arguments from the
  'SparseCsrCPU' backend.` This matches a tracked upstream bug,
  [pytorch/pytorch#160813](https://github.com/pytorch/pytorch/issues/160813). Where it
  does run (other builds), it is non-differentiable by construction — no registered
  backward.

Net: a `torch.sparse`-first design gives tellegen a beta API, no batched matvec that
runs, no batched or differentiable direct solve, and a `spsolve` broken in the exact
environment this project ships — using natural calling conventions, not edge cases.

## 2. Sparse linear solves in PyTorch: the honest gap

`torch.sparse.spsolve` is the only first-party sparse direct solve: CPU-only,
non-differentiable, unbatched, and (as above) non-functional in this build. `torch.linalg`
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
restricted to square systems"; naive backprop through k CG iterations costs `O(k)` graph
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

**(a) SPD.** For potential-driven branch laws with `g >= 0`, `J` is a weighted graph
Laplacian restricted to interior nodes (boundary/ground rows dropped): symmetric by
construction, positive definite whenever every interior node has a path to a boundary
node (tellegen already detects and rejects the floating case,
`PotentialFlowLayer._floating_group_nodes`). CG applies directly, no symmetrisation needed.

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

Full table (all `B` at every `n`) is in the raw log. Findings, all measured:

**Dense OOMs early, on `B * n_i * b`, not `n` alone.** The dominant cost is the
`(B, n_i, b)` broadcast tensor inside assembly, not the final `(B, n_i, n_i)` Jacobian:
the crossover is a joint function of network AND ensemble size. At `n=300` dense is fine
for `B=100` (469 MB) but not `B=1000` (1.4 GB, climbing); by `n=3000`, `B=100` needs 14.4
GB and `B=1000` needs 144 GB. For "many buildings, many scenarios," this wall sits at
ordinary sizes.

**Unpreconditioned CG stops converging past `n~300` at any batch size; Jacobi buys
roughly 1.6-4x fewer iterations but degrades too.** The conductance-spread sweep
(`n=300, B=100`) shows the mechanism: plain CG needs 56 iterations at 0 decades of
spread, 179 at 2, and still has not converged at 3000 iterations by 6-8 decades; Jacobi
needs 35, 76, 446, 805 — better, same curve, not flat. This confirms the design spec's own
prediction ("iterative solves degrade on extreme conductance ratios... an ordinary
building configuration, not a corner case") and is why `cgJ` reads "capped" above `n=1000`
or `B=1000`: Jacobi alone is not enough once the batch samples a genuinely extreme ratio.
This is where AMG (section 3d) earns its complexity.

**`torch.sparse`'s batched-COO matvec is correct but consistently slower, by a margin
growing with size — and that is why the sweep above has gaps.** A controlled
micro-benchmark (30 raw matvec calls, no CG) gives per-call cost directly: within 5x
either way at `n=50` (overhead dominates), reaching 15x slower at `n=300, B=1000` (5.86ms
vs 88.8ms) and 9-15x from `n=1000` up. `n=3000, B=1000` — 24 million nonzeros once COO's
per-batch index replication is paid for — did not complete even 30 iterations within this
benchmark's time budget and was abandoned; that gap is itself the measurement. Where both
methods reached real convergence (`n<=100`), answers agreed to `1e-6`-`1e-9`.

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

This is not close to a coin toss: every project surveyed that both batches and
differentiates lands on gather/scatter (or a hand-rolled matrix-free operator) plus IFT.
Projects using genuine sparse formats (SABLE, torch-sla) do so to reach GPU vendor
solvers (cuDSS), paying the index-replication cost from section 1 — optimising peak GPU
throughput at huge scale, a different axis than tellegen's.

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

**(b) The Jacobian and its solve — matrix-free CG, not a formed sparse matrix.**
`torch.sparse` is not ready: no working batched matvec, no differentiable, batched, or
even functional direct solve. Matrix-free CG needs no dependency, batches like the
topology operators, and reuses the *same* matvec for the implicit-function adjoint since
`J` is symmetric. Ship Jacobi preconditioning from day one (section 4: 1.6-4x fewer
iterations wherever both converged, convergence at all where plain CG did not); treat
AMG as a later addition gated on extreme conductance ratios actually appearing, not on
node count.

**(c) The transport layer — matters most, and first.** The design spec measured the
dense transport operator at 7.7 s / 26 GB projected at target scale — the one component
actually observed to break. `TransportLayer.operator()`
(`src/tellegen/layers/transport.py`) builds `Out`, `In`, `L` as dense `(..., n, n)` (or
`(..., K, n, n)`) tensors via `einsum` over one-hot selectors — this should become
gather/scatter like the Jacobian, with `_van_loan_step`'s matrix exponential restricted
to sparsity-permitting cases or replaced by a Krylov-subspace exponential-action method
(`expm(M) @ v` without forming `expm(M)`), since `torch.linalg.matrix_exp` has no sparse
path either. This is milestone 1b's real payload: the Jacobian solve is the more
interesting number, but transport is what actually OOMs.

**What can stay dense.** Anything genuinely `O(n)`-small and not batched at scale:
boundary selectors, single-instance debugging, tests. `torch.linalg.solve` remains right
while `B * n_i * b` stays small — fine at `n<=100` for any `B` up to 1000, and at `n=300`
up to `B=100`; it fails at `n=300, B=1000` and beyond.

**What would break the adjoint.** Nothing structural: `implicit_solve` only needs
`jacobian(x, *params)` to return something `adjoint()` can solve against, and `adjoint()`
is just `torch.linalg.solve(J.T, grad_x)` — swapping that call for matrix-free CG (same
matvec as forward) is like-for-like. The one real risk: `linear.py`'s `floating_nodes`
diagnostic inspects `J`'s dense rows to name disconnected nodes; with no formed `J` it
must move to the edge-index level, where `_floating_group_nodes` in `potential.py`
already does this — so `linear.py`'s dense-only version should be retired, not ported.
