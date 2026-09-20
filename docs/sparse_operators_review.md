# Independent review of the sparse-network decision

12 September 2026. Reviewed commits `a26a645` and `ba42b9a`, their report,
benchmark source and raw logs, and the current topology, potential, transport,
Newton and implicit-gradient implementations. The user's clarification makes
scaling to composed network models and digital twins a primary acceptance
criterion. Those two commits change documentation and benchmarks; they do not
implement the sparse production path.

**Verdict: keep the sparse requirement and shared edge-index foundation; revise
the claim that measurement has settled the solver architecture.** Gather/scatter
is a defensible default. The evidence does not justify banning `torch.sparse`,
making CG the general solver, or asserting a measured dense failure threshold.
The transport priority and early investigation of conditioning are justified.

Independent, bounded reproductions are in
[`sparse_review_checks.py`](../benchmarks/sparse_review_checks.py), with output in
[`sparse_review_results.txt`](../benchmarks/sparse_review_results.txt). These use
the repository's torch 2.14.0+cpu and float64. No production code or original
specification/report was changed by this review.

The existing default test suite passed: **241 passed, 1 slow test deselected,
97.60% coverage**. Command: `.venv/Scripts/python -m pytest -q -p
no:cacheprovider --basetemp=<workspace>/.review-pytest-20260912-sparse`.
A first run encountered the sandbox's inaccessible default temporary directory;
using a fresh workspace temporary directory resolved it. These tests validate
the existing dense implementation, not the proposed sparse migration.

1. **[P1] The prohibition on `torch.sparse` excludes an untested, working shared-pattern path.**

   Locations: design spec lines 171–190; original report lines 26–50 and 234–240.
   The benchmark compares gather/scatter with a *formed, batched COO Jacobian*.
   It omits applying shared two-dimensional sparse incidence matrices with the
   batch as dense right-hand-side columns:

   ```python
   differences = torch.sparse.mm(A_I_T, X.T)
   JX = torch.sparse.mm(A_I, G.T * differences).T
   ```

   `G` differs per batch member. Neither incidence matrix has batch-replicated
   indices. Storage remains O(E + B(E + N)), and the Jacobian is never formed.
   This directly refutes the spec's assertion that a shared sparse matrix cannot
   help when Jacobian values vary by instance. Both COO and CSR versions agreed
   with gather/scatter and passed `gradcheck` with respect to `G` and `X`.

   Representative independent medians, milliseconds per matvec, excluding setup:

   | Nodes | Batch | CPU threads | Gather/scatter | Shared CSR incidence |
   |---|---|---|---|---|
   | 300 | 100 | 1 | 0.4474 | 0.2778 |
   | 300 | 100 | 14 | 0.4539 | 0.6893 |
   | 1000 | 100 | 1 | 3.3825 | 1.2114 |
   | 1000 | 100 | 14 | 0.7659 | 1.8058 |

   These small CPU measurements establish a missing viable candidate, not a
   universal winner. Layout, thread count, batch and hardware matter. Keep
   gather/scatter as the baseline, with backend selection behind the operator
   interface. [PyTorch's current sparse.mm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)
   explicitly supports COO/CSR times dense and gradients through both inputs.

2. **[P1] “Dense fails at 300 nodes and batch 1000” is not a measured OOM result.**

   Locations: benchmark lines 280–290 and 328–346; report lines 149–155; spec
   lines 176–177. The benchmark skips this case when its estimated weighted
   incidence temporary exceeds a self-imposed **400 MB budget**. The comments
   describe earlier slow runs, not a captured allocation failure. The 1.4 GB
   figure estimates one `(B, n_i, E)` intermediate, not total process memory or
   even the final Jacobian. Reported `dense_mem_MB` is the process lifetime peak,
   including preceding benchmark paths, so it cannot identify each solve's peak.

   A bounded independent run solved all 1000 instances of the same graph and
   conductance construction at n=300 using the original dense assembly in
   chunks of 32: **3.92 s**, maximum relative true residual **1.84e-10**, with a
   **45.8 MB weighted temporary** per chunk. This is not a rerun of the unchunked
   allocation, and it does not make dense appropriate for large composed models.
   It demonstrates that the claimed ensemble threshold is not fundamental.

   A sequential SciPy sparse-direct comparison on those same systems took
   **1.15 s**, with maximum relative true residual **7.78e-11**. This includes
   per-instance assembly/factorization/solve but not initial incidence setup; it
   is a CPU reference, with no custom backward implemented in this check.
   Direct solving is therefore a concrete fallback candidate. Dense Jacobians
   can also be assembled from edge contributions without the weighted-incidence
   temporary, although their O(B N²) storage still makes them unsuitable as the
   default at large scale. Keep dense for bounded references/fallbacks and
   measure sparse-direct fill and memory before selecting production limits.

3. **[P1] CG eligibility and the connectivity diagnostic are narrower than stated.**

   Locations: report lines 92–105; spec lines 164–169 and 202–208;
   `layers/potential.py:197–254,365–366`.
   For nonnegative branch slopes, `A_I diag(g) A_I.T` is positive semidefinite.
   Positive definiteness requires every interior component to reach a fixed
   potential through **strictly positive current slopes in that instance**.
   Connectivity in the unweighted graph is insufficient.

   `_floating_group_nodes` uses `(k_flat != 0).any(dim=0)`: the union of active
   edges across the batch. On a three-node chain grounded at node 0, slopes
   `[[1,1], [0,1]]` produce no reported floating nodes, although the second
   Jacobian has an exactly zero eigenvalue. The independent script reproduces
   this. The check uses linear-initialization slopes and is bypassed when the
   caller supplies `phi0`; it is not performed before every solve as claimed.
   The current dense solve can still reject a singular system. Removing it
   without replacing that protection would be a regression.

   Require per-instance singularity/breakdown diagnostics, zero-slope handling,
   and an explicit SPD contract. Nonmonotone/custom elements need validation or
   another solver. The current Drive protocol admits dependence on `phi`;
   derivatives of that dependence generally invalidate the stated Laplacian
   Jacobian. The spec already records this unresolved contract in section 9.
   Resolve it before extending the symmetry assumption to coupled models.

4. **[P1] The transport rewrite needs its own solver and gradient design.**

   Locations: report lines 242–252; `layers/transport.py:147–190,231–312`.
   The production code confirms the scaling problem: dense upwind/downwind
   selectors, dense spatial generators, then a dense `(K n_i)²` species matrix.
   The default exponential step doubles that dimension again. All three time
   schemes and `steady()` consume dense matrices, so replacing assembly alone
   does not finish the migration.

   Signed advection is generally nonsymmetric, as are general species kinetics.
   Therefore its implicit, trapezoidal and steady solves cannot inherit the
   potential-flow CG assumption or use the forward matvec as the transpose.
   Keep state as `(batch, nodes, species)`, edge transport as `(batch, edges,
   species)`, and local kinetics as node-local species blocks. That permits
   O(B(E K + N K²)) operator work for dense local kinetics, without dense
   cross-node species blocks. Evaluate a nonsymmetric iterative method such as
   GMRES and sparse direct solving separately from potential-flow PCG.

   Exponential action is a promising route for the default scheme, but must
   preserve the affine forcing, not only compute `exp(dt*M) @ x`. For a fixed
   forcing vector `b`, the augmented action
   `exp(dt * [[M,b],[0,0]]) @ [x,1]` gives the required update without a `2m`
   block exponential. A numerical action method needs error control and a
   checked backward path. Validate flow reversal, capacity scaling, boundary
   forcing, reactions, conservation, applicable positivity properties and
   gradients against the existing dense reference. The 7.7 s / 26 GB transport
   claim comes from the earlier spec; the sparse benchmark does not reproduce
   or independently document that transport OOM.

5. **[P2] Conditioning is demonstrated, but the stopping data do not settle robustness or scaling.**

   Locations: benchmark lines 184–210,348–410 and 419–442; report lines 157–164.
   Part A reduces plain CG's cap from 800 to 150 and Jacobi CG's from 1200 to
   250 as batch/size grows; sparse COO often gets only 30 iterations. These are
   unequal workloads, not comparative times to a correct answer. Hitting a cap
   is not proof that the method cannot converge at that size. The script lacks
   per-instance iteration/status output, an independent true-residual check,
   Jacobi solution-error reporting and solve/adjoint gradient checks. It also
   continues updating converged members and clamps nonpositive denominators
   instead of reporting loss of positive definiteness.

   Independent checks of the spread experiment support its main trend:

   | Conductance decades | Plain CG iterations | Plain failed instances / 100 | Jacobi CG iterations | Jacobi failed instances / 100 |
   |---|---|---|---|---|
   | 0 | 56 | 0 | 35 | 0 |
   | 2 | 182 | 0 | 76 | 0 |
   | 6 | 3000 cap | 9 | 445 | 0 |
   | 8 | 3000 cap | 100 | 813 | 0 |

   Failure here means a separately recomputed relative infinity residual at
   least 1e-6. The topology/conductance seeds match Part A2; the RHS seed differs,
   so identical iteration counts are not expected. Original `SKIP_PART_A=1`
   itself changes the RNG history and hence the A2 RHS. All tested Jacobi solves
   passed that residual criterion; this is not evidence of accurate gradients
   at stricter tolerances or acceptable time at digital-twin scale.

   Conductance ratio alone does not determine difficulty. An eight-decade
   grounded star converges in one Jacobi iteration (also reproduced). Conversely,
   long chains and weakly connected model interfaces can be difficult without
   extreme coefficient spread. A random tree with random shortcut edges does
   not represent all building/street/sewer compositions. The claimed physical
   six-decade damper range is plausible motivation, not established by the
   random-conductance experiment or a measured branch-law calibration.

   Test PCG with real derivative ranges, chains, spatial graphs, weak bridges,
   changing active edges, multiple boundaries and composed submodels. Evaluate
   stronger preconditioning/direct fallback before milestone 2 as proposed.
   A good AMG hierarchy is coefficient-dependent: “build once per topology”
   needs a reuse/update policy and measurement, especially across different
   damper states. [PyAMG's API](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html)
   builds from the matrix and exposes strength-of-connection and smoothing choices.

6. **[P2] The literature/API summary contains factual overstatements relevant to future backends.**

   Locations: report lines 16–19,40–56,193–221. The local CPU `spsolve` failure
   and unavailable direct batched-CSR matvec were reproduced, although the
   different CSR operations do not all produce the same error claimed in the
   report. “No batched matvec” contradicts its own working COO example.
   “Documented CPU-only” is incorrect: the cited
   [spsolve page](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.spsolve.html)
   says no such thing, and
   [current upstream dispatch](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/native_functions.yaml)
   registers `_spsolve` for SparseCsrCUDA and SparseCsrXPU, not CPU. This does not
   establish availability in a particular GPU wheel; no GPU was tested here.

   Batch-replicated sparse indices incur a constant-factor memory cost, but
   still retain O(B E) storage on sparse graphs. They do not negate the main
   advantage over O(B N²) dense storage. Nor is duplication intrinsic to vendor
   sparse direct solving: [cuDSS documents uniform batching](https://docs.nvidia.com/cuda/cudss/types.html)
   with shared sparsity and a buffer of per-instance values. Its analysis and
   numerical factorization are separate phases. Integration and gradient work
   remain, but this is relevant to the stated scaling goal, not an irrelevant
   huge-scale exception. The report's own [SABLE source](https://arxiv.org/abs/2606.07099)
   explicitly uses sparse-direct LU; it cannot support a universal conclusion
   that differentiable batched projects choose matrix-free iterative solves.

For implementation, I recommend this order:

1. Establish a small operator/solver contract and shared endpoint primitives.
   Include `matvec`, `rmatvec`, diagonal/block-diagonal access, optional CSR
   assembly, shape/device/dtype metadata and per-instance solver results.
   Preserve a bounded dense reference. `Network.edge_index(kind)` currently
   means a one-dimensional list of edge columns, so add a distinct endpoint
   accessor rather than silently changing that existing meaning.
2. Complete the transport migration first using those primitives, including
   time stepping, steady solves and backward. Then migrate all remaining
   topology consumers. Audit cycle utilities too: `particular_flow` currently
   uses a dense tree solve, and `branch_flows` materializes a dense cycle basis.
   Merely changing the named topology methods will not remove these costs.
3. Migrate potential-flow initialization, Newton updates and adjoints to the
   solver interface. PCG/Jacobi is the initial SPD backend; keep sparse-direct
   and stronger preconditioners open. Newton currently accesses dense shapes,
   forms an identity and calls `torch.linalg.solve`, so this is a coordinated
   change, not just replacing the one call in `adjoint()`.
4. Add a composed-model scaling gate before treating milestone 1b as complete:
   join multiple building blocks through street/sewer interfaces; vary nodes,
   edges, ensemble, species and simulation length; test interface conservation
   and gradients across the join. Report operator cost separately from solver
   setup, iterations, forward/backward latency and peak memory. O(E) matvec cost
   does not imply an O(E) solve or simulation. Retain submodel/interface maps
   for future block preconditioners or domain decomposition, and group different
   topologies rather than requiring all digital twins to share one giant batch
   shape. Same-physics graph union can preserve an SPD potential block;
   general multiphysics feedback need not. External black-box models retain
   their port/co-simulation contracts and do not acquire gradients automatically.
   [As delivered in milestone 5, `noodl.couple.union` does not merge graphs or
   remap interface nodes at all: it orchestrates two independently-built `Model`s,
   exchanging named driver/state values between their `step` calls each outer step
   and rechecking grounding within each model unchanged. The graph-merging design
   sketched in this paragraph was not the one built; see
   docs/development-history.md, Milestone 5 status, "Why orchestration, not graph
   merging".] For long differentiable simulations,
   budget timestep-history memory separately: the existing implicit-function
   backward removes Newton-iteration history, not the history of all timesteps.
   Checkpointing or a separate time-adjoint strategy may be needed.

**Proposed replacement decision:** Sparse by default means shared graph topology
and operator application whose storage and work scale with the represented
connections and local state. Gather/scatter is the initial backend. Solver
selection depends on the operator's mathematical properties and measured
end-to-end performance; PCG/Jacobi applies to validated SPD potential blocks,
with sparse direct and stronger preconditioning available as evaluated options.
Transport and coupled operators need their own stepping, transpose and solver
paths. Scalability is accepted on composed models, including gradients and
repeated timesteps, rather than inferred from one Laplacian microbenchmark.
