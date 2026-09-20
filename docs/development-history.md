# Development history

The per-milestone engineering records for noodl, moved here out of the README. Each section
was written as the milestone landed: what it added, the decisions behind it, what was
measured, and what was left open. They are a record of how the package reached its current
shape, not a guide to using it — for that, start at the
[published documentation](https://mvreeuwijk.github.io/noodl/).

The design specs and implementation plans these milestones were executed from are kept
under `docs/superpowers/`. Like this page, they are part of the repository but not of the
published site, and they still refer to the package by its former name, `tellegen`.

## Milestone 1b status

Milestone 1b moved every hot path onto a matvec-free linear-operator contract: no `n x n`
or `n x b` object is formed on any solve path, and the dense forms that remain
(`jacobian()`, `TransportLayer.operator()`, `*.assemble()`, `solvers.linear.solve`) are
retained deliberately as test oracles.

**What passes.** All of section 6.1's correctness gates: parity with the dense reference,
interface conservation at the joins, cross-join gradients against finite differences, and
per-instance solver honesty. All four peak-memory budgets, under both `auto` and `cg`. Both
shape gates -- peak memory vs nodes at 1.01x and matvec time vs edges at 1.13x (edge
sensitivity 1.64x at 16x/32x the reference edge count), against a 2.5x budget each. **Every
backward-pass budget under the shipped default `auto`, and three of the four under `cg`** --
`cg`'s ensemble-1 backward misses at 1.20x, and that row's verdict is not settled (see below).
Before the sparse-direct evaluation two of the four backward rows failed.

**What does not.** The forward-pass wall-clock budgets, on every row, under both solvers, per
[`benchmarks/composed_scaling_report.json`](benchmarks/composed_scaling_report.json)
(`all_budgets_met: false`, keyed to the shipped default, `auto`):

| Configuration | Solver | Forward | Backward | Peak memory |
|---|---|---|---|---|
| ensemble 1, 1 step | `auto` (sparse_direct) | 0.104 s vs 0.05 s (2.08x) FAIL | 0.080 s vs 0.1 s (0.80x) PASS | 69.1 MB vs 100 MB (0.69x) PASS |
| | `cg` (PCG) | 0.228 s vs 0.05 s (4.56x) FAIL | 0.120 s vs 0.1 s (1.20x) FAIL | 46.4 MB vs 100 MB (0.46x) PASS |
| ensemble 100, 1 step | `auto` (= PCG, above the threshold) | 1.499 s vs 0.5 s (3.00x) FAIL | 0.668 s vs 1.0 s (0.67x) PASS | 129.9 MB vs 1000 MB (0.13x) PASS |
| | `cg` (PCG) | 1.733 s vs 0.5 s (3.47x) FAIL | 0.641 s vs 1.0 s (0.64x) PASS | 128.5 MB vs 1000 MB (0.13x) PASS |
| ensemble 100, 24 steps | `auto` (= PCG, above the threshold) | 40.7 s vs 12 s (3.39x) FAIL | 14.4 s vs 25 s (0.58x) PASS | 426.6 MB vs 2000 MB (0.21x) PASS |
| | `cg` (PCG) | 39.2 s vs 12 s (3.27x) FAIL | 14.1 s vs 25 s (0.56x) PASS | 425.4 MB vs 2000 MB (0.21x) PASS |
| ensemble 1000, 1 step | `auto` (= PCG, above the threshold) | 13.3 s vs 5 s (2.66x) FAIL | 3.90 s vs 10 s (0.39x) PASS | 746.9 MB vs 8000 MB (0.09x) PASS |
| | `cg` (PCG) | 13.0 s vs 5 s (2.60x) FAIL | 3.81 s vs 10 s (0.38x) PASS | 755.2 MB vs 8000 MB (0.09x) PASS |

Forward misses its budget on every row under both solvers, by 2.08x-3.39x under `auto` and
2.60x-4.56x under `cg` -- 2.08x, the shipped default's ensemble-1 row, being the smallest
margin anywhere in the table. Before the sparse-direct evaluation the same rows missed by
roughly 6-9.7x. At the reference row it was measured on (ensemble 1, 1 step; the older
committed figures were all under what `auto` then meant, plain PCG): forward 0.483 s ->
0.104 s under `auto`, 0.228 s under `cg`; backward 0.290 s -> 0.080 s / 0.120 s.

`method="auto"` selects sparse-direct SciPy SuperLU for a per-instance operator that certifies
SPD, declares a sparse form, whose solve is grad-safe, with SciPy importable, **and whose flat
batch is at most 32 instances**; Jacobi-PCG above that threshold and in every other certified
case; GMRES for the non-symmetric transport block (`TransportLayer`), unchanged. The threshold
is why the `auto` and `cg` rows above coincide from ensemble 100 up: there they are the same
backend, and the report shows it rather than merely asserting it -- `auto`'s
`linear_iterations_max` is 1 at ensemble 1 and 180 at ensembles 100, 100x24 and 1000, matching
`cg`'s. SuperLU has no batched entry point, so its cost is linear in the ensemble
(~21 ms/instance) while PCG's batched arithmetic is sub-linear; measured in process, the ratio
sparse_direct/cg runs 0.24 at batch 1, 0.68 at 32 and 1.00 at 64. This rule and its threshold
were decided on, and measured at, the reference size only -- 1028 unknowns, CPU, SciPy present;
SuperLU's fill-in at much larger `n` is unprobed, and revisiting the rule there is the
documented condition in section 6.2. A vendor sparse-direct route -- one batched call rather
than a Python loop, and separately the CUDA/XPU `_spsolve` route -- stays admissible but
unbuilt, per spec section 2; it would remove the loop and with it the threshold.

**Under the shipped default, `linear_iterations` is 1 wherever sparse-direct runs** (one
factorisation per Newton step), so on those rows the number carries no conditioning
information at all. Section 6.2's "Jacobi-PCG exceeds 500 iterations -> revisit
preconditioning" trigger, and `_iteration_counts`'s stated purpose ("a conditioning regression
shows up in it first"), therefore live on the rows where PCG actually runs: the `cg` rows at
every size, and `auto`'s own rows from ensemble 100 up. Those read 168-180 iterations, well
under the 500 trigger, so it did not fire -- evaluated on the reference configuration only,
not across section 6's robustness cases. Read the `cg` rows as the conditioning monitor, not
as a legacy comparison.

**Two rows are not settled results, and section 6.2's <=2x rule applies to both.** That rule
says any §6.1 budget missed by 2x or less is re-measured with `samples >= 5` and, if still
missed, recorded as an open budget failure with its measured spread. Neither has been
re-measured at that sample count; the committed figures are medians of `samples: 3`.

- `auto`, ensemble 1, **forward**. The same configuration and code has measured 0.097 s
  (1.94x), 0.104 s (2.08x) and -- under the acceptance gate itself
  (`pytest tests/verification/test_composed_scaling.py -m slow`) -- 0.164 s (3.29x). The row straddles the 2x boundary depending on which run is read, so
  whether the rule even fires on it is itself unsettled. Spread 0.097-0.164 s, 1.94x-3.29x,
  against a 0.050 s budget.
- `cg`, ensemble 1, **backward**. 0.120 s (1.20x, FAIL) here against ~0.100 s (~1.00x, PASS)
  in the previous report on the same code. `auto`'s equivalent row has been as wide:
  0.080 s (0.80x) and 0.089 s (0.89x) in two report runs against 0.141 s (1.41x, FAIL) in the
  gate run. Run-to-run spread on this machine is wide enough to flip these verdicts.

**The two instruments disagree systematically, and that is the bigger open item.** The
acceptance gate (`test_composed_scaling.py -m slow`) and the report script call the SAME
measurement functions in `benchmarks/report_composed_scaling.py`, so they should agree. On the
tree they were both run against (`f3d2ba4`, before the batch threshold existed, so `auto` was
sparse-direct on all four rows) the gate's forward figures were 1.2-1.7x worse on every row:
3.29x / 5.16x / 4.96x / 5.13x from the gate against the report's
1.94x / 3.37x / 4.04x / 3.03x for the identical configurations. A systematic gap of
that size between two instruments matters more than either number, because it is the gap that
decides the verdict on the two rows above. The gate has not been re-run since the threshold
landed, so the table here is the report script's alone. Both are carried into the next
milestone's gate, the discrepancy itself as a measurement defect to understand, per section
6.2.

The committed report sweeps ensemble size and simulation length, now under two solvers;
section 6's species count and joined-submodel sweeps, and its composed robustness cases, are
not measured. Nodes and edges are covered by the two shape gates.

### Docstring narratives moved from source

The paragraphs below were review-history narrative embedded in source docstrings and
comments; they were moved here (Task 26) to keep the contract beside the code concise, and
are reproduced verbatim.

From `src/noodl/layers/potential.py::PotentialFlowLayer._grounding_check`:

> The certificate (Task 3's `solvers.grounding.spd_certificate`) tests both of spec
> section 3.1's testable conditions -- every branch slope non-negative, and every
> interior node grounded through strictly positive slopes -- so the batched message
> below ("do not certify a grounded, positive-slope system") states exactly what was
> checked. It is run on the slopes
> GIVEN -- the caller decides whether those are `linear_init`'s tangent-at-zero slopes
> or the actual `dflows` at a solve point -- and it is per instance. That is the whole
> point: the pre-Task-11 check ORed "is this edge's slope nonzero" across the WHOLE
> batch before testing connectivity, so an edge closed in one instance but open in
> another counted as present for both, and a genuinely ungrounded instance sailed
> through to a dense factorisation that could only report "singular", if it reported
> anything at all.
>
> The message is built from `solvers.grounding.spd_diagnosis` (amendment A2), which
> names, per failing instance, either the negative-slope EDGES or the ungrounded
> interior NODES. Node indices are rendered as node NAMES here, because this layer --
> unlike the raw operator -- knows them, and because the error text every existing
> (unbatched) test in test_potential.py asserts on is exactly those names. A batched
> failure additionally leads with the failing BATCH INDICES, since node-level detail
> alone is not attributable across unrelated per-instance failures. Every message
> names the offending LAYER first: a model composes several layers over one network,
> and "solve: floating nodes ... ['z']" alone does not say which of them failed.

From `src/noodl/layers/potential.py::PotentialFlowLayer.solve`:

> This branch used to run Newton's closures under ORDINARY autograd, so with a
> `learnable=True` Element (or a grad-requiring `phi_boundary`/`drivers`/`sources`) the
> result carried an UNROLLED graph through the converged iterate. Those gradients were
> real but were never the implicit-function ones `differentiable=True` computes, and
> nothing asked for them. Worse, they made the INNER SOLVER'S CHOICE depend on the
> caller's ambient grad mode: `solvers.select.solve`'s `"auto"` will not hand a
> grad-requiring solve to the non-differentiable sparse-direct backend, so the same
> call factorised through SuperLU from a plain call site and fell back to PCG -- 4.6x
> slower -- from inside `torch.enable_grad()` with a grad-requiring `sources`. A
> backend must not be a function of who is calling. A caller who wants gradients calls
> `differentiable=True`, which is unchanged (`solvers.implicit._Implicit.forward`
> already solved under `no_grad` and takes its gradients from the adjoint).

From `src/noodl/layers/potential.py::PotentialFlowLayer.solve` (body comment, the
detached-`phi0` `no_grad` block):

> What it costs is everything `linear_init` does -- a whole preconditioned-CG loop, its
> int64 gather indices and every iterate -- retained until backward(). Measured on
> the composed model at ensemble 100 (Task 14 review): 2567 MB of the 2571 MB saved
> per differentiable step came from here; with a detached guess the same step saves
> 74.7 MB.

From `src/noodl/layers/transport.py` (module-level comment on the `_TransposeView` alias):

> `_TransposeView` used to be its own LinearOperator-shaped adjoint-view class, duplicating
> `solvers.implicit.TransposeOperator` method for method except for `spd_certificate` (this
> module's version always returned None; `TransposeOperator`'s forwards the wrapped
> operator's certificate iff it declares itself symmetric). Both this layer's operators
> (`AdvectionOperator`, `_AffineSystemOperator` below) declare `symmetric = False`, so
> `TransposeOperator.spd_certificate()` returns None for them exactly as the old local class
> did -- this alias changes nothing observable here, it only removes the duplicate.

From `src/noodl/layers/transport.py::TransportLayer.__init__` (conduction-edges comment):

> The conduction edges' ENDPOINTS, never the (n, b_c) incidence matrix and never
> the (n, n) Laplacian it used to build here: since Task 15 this tuple is the
> layer's whole representation of its conduction topology. `_advection_operator`
> (Task 9) already consumed exactly this; `operator()`, the dense oracle, now
> forms its (n, n) `L` from it on demand (`_conduction_matrix`). The (n, n)
> matrix was 8.5 MB at the composed model's reference size, grew 4x per node
> doubling, and -- with no conduction configured, as in that model -- was a block
> of ZEROS that `operator()` subtracted for nothing.

From `src/noodl/layers/transport.py::TransportLayer._step` (`on_failure` validation comment):

> This one used to sit inside the `scheme == "exact"` branch below, after
> `_to_stacked` had already validated and reshaped `x`, so a caller who passed
> both a bad shape and this unusable combination was told about the shape (final
> review M9).

From `src/noodl/operators/graph.py::GraphLaplacianOperator._apply`:

> The body is deliberately flat (the scatter into the full node space and the weighted
> endpoint difference were their own helpers until the milestone-1b follow-up): this
> runs once per PCG iteration, thousands of times per solve, and at ensemble 1 the two
> extra Python frames alone were measurable against the ~50 us the whole call takes.
>
> The two `scatter_add_` calls are NOT fused into one `index_add` over
> `cat([src, tgt])` with `cat([w, -w])`. That fusion IS bit-identical (measured: same
> `x` to the last bit, since each output node still accumulates its incident edges in
> the same order), but it is SLOWER -- interleaved medians at the composed reference
> size: 34.7 vs 36.6 us at ensemble 1, 474.8 vs 529.8 us at ensemble 100 -- because
> building `cat([w, -w])` costs a negate and a copy of a (batch, 2 * edges) tensor,
> which is more than the one `scatter_add_` dispatch it saves. The accumulator zeros
> tensor is already shared by both scatters, so there was never a second one to save.

From `src/noodl/solvers/implicit.py` (module docstring):

> `@torch.autograd.function.once_differentiable` was tried first, as it is the standard idiom
> for this, but was found NOT to catch this case here and was dropped again: its guard fires
> only when the incoming `grad_x` itself already `requires_grad`, which is false for the
> ordinary implicit unit-seed `torch.autograd.grad(x.sum(), theta, create_graph=True)` produces
> -- confirmed empirically (with and without the decorator, a mixed loss
> `(dx/dtheta**2).sum() + (theta**2).sum()` silently returns only the second term's gradient,
> identically, in both cases). It also cannot be layered underneath a manual check of its own,
> since its wrapper forces `torch.no_grad()` before calling the wrapped body, hiding the very
> signal (`torch.is_grad_enabled()`) that would otherwise reveal a `create_graph=True` request.

From `src/noodl/solvers/newton.py` (module docstring):

> This is what replaces the old identity-substitution trick for a converged-but-
> singular instance: a dense ``torch.linalg.solve`` raises for the WHOLE batched
> call if any one instance's matrix is singular, which is why the old code had to
> substitute an identity for converged rows before the call ever happened. An
> operator-based solve (PCG/GMRES, or a per-instance-safe direct path) is batched
> elementwise over the leading dimensions with no cross-instance coupling in its
> own arithmetic, so one instance being exactly singular cannot make the call
> fail for its siblings; that instance's own step is simply garbage (possibly
> NaN), and it is discarded by the ``torch.where`` on ``step`` below exactly as
> it always was, without needing to keep the solve well-posed first.

From `src/noodl/solvers/newton.py::inner_solve_rtol`:

> measured: the float32 CONTAM series case in ``tests/verification`` floors at 3.2e-8 and
> was reported as a ``linear_init`` failure until this floor was applied, and the 128-zone
> leaky chain in ``tests/solvers/test_newton_operator_contract.py`` ran every inner PCG to
> its full ``max_iter = 128`` ceiling. That second symptom is the quieter one: Newton
> passes ``on_failure="return"``, so the ``MAX_ITER`` status is swallowed and only
> ``NewtonResult.linear_iterations`` -- the number the composed-model report publishes --
> carries the damage, as the ceiling rather than the work actually done.

## New public API in milestone 1b

Every one of these is optional and defaults to the pre-milestone behaviour.

| Entry point | Keyword | Meaning |
|---|---|---|
| `PotentialFlowLayer(...)` | `linear_solver=` | inner linear solver for this layer, used on both the forward and the backward pass: `"auto"` (default; sparse-direct SciPy SuperLU when the per-instance SPD certificate holds, the operator has a sparse form, SciPy is importable, the solve is grad-safe **and the flat batch is at most 32 instances**; PCG in every other certified case, including larger ensembles; GMRES when the operator cannot certify), `"cg"`, `"gmres"`, `"sparse_direct"` (SciPy SuperLU on the assembled sparse form), or `"direct"` (LU of the assembled dense operator -- the retained milestone-1 numerics). The sparse-direct branch of `"auto"`, and `"sparse_direct"` itself, need SciPy: `pip install noodl[sparse]`. Without it `"auto"` falls back to PCG (correct, and 4.6x slower at ensemble 1) and warns once per process; an explicit `"sparse_direct"` raises `ImportError` naming scipy |
| `PotentialFlowLayer.solve(...)` | `diagnostics=` | a dict, filled with `newton_iterations`, `linear_iterations`, `method`, `backend`, `converged` and `residual_norm`. `method` is the solver **requested** (e.g. `"auto"`); `backend` is the one that **ran** (`"sparse_direct"`, `"pcg"`, `"gmres"` or `"direct"`), which is the only way to see which side of the batch threshold -- or of the SciPy check -- a given solve landed on |
| | `on_failure=` | `"raise"` (default) or `"return"`. `"return"` requires `diagnostics=` (the status must land somewhere) and `differentiable=False` (a non-converged forward has no defined adjoint) |
| | `method=` | overrides the layer's `linear_solver` for this call |
| `TransportLayer.step(...)` / `.steady(...)` | `on_failure=` | `"raise"` (default) returns a `Tensor`; `"return"` returns the raw `SolveResult` with its per-instance status. `step` accepts it only for the `"implicit"` and `"trapezoidal"` schemes |
| `solvers.newton.newton(...)` | `operator=` | a callable returning a `LinearOperator` at the current iterate (a plain dense `(..., m, m)` tensor is still accepted and auto-wrapped) |
| | `method=` | forwarded to `solvers.select.solve` for every inner linear solve |
| | `on_failure=` | `"raise"` (default) or `"return"`, which returns the `NewtonResult` with its true per-instance `converged` |
| | `where=` | the caller's name for this solve, used to prefix Newton's own error and every inner-solve refusal |
| `solvers.implicit.implicit_solve(...)` | `diagnostics=` | as above, filled from the forward Newton solve. `on_failure="return"` is refused here |

`solvers.select.solve` is the single place where a numerical failure becomes an exception;
`solvers.iterative.pcg`/`gmres` never raise, and an eligibility refusal (asking for `"cg"`
on an operator that cannot certify SPD, or letting `"auto"` see a partly-certifying batch)
raises regardless of `on_failure`.

The SPD certificate enters at two different levels, and it is easy to attribute it to the
wrong one. At `solvers.select.solve`, only `"cg"` and `"auto"` consult it: `"sparse_direct"`
and `"direct"` are direct methods that make no SPD or symmetry assumption and never look at
it. At `PotentialFlowLayer`, the grounding check in front of every solve requires the
certificate for **every** `linear_solver` -- `"direct"` and `"sparse_direct"` included --
because an ungrounded instance is a singular system whichever backend is asked to solve it,
and the layer would rather name the floating nodes than hand back a plausible wrong answer.

## Milestone 2 status

Milestone 2 put HEAT on the same footing as species and airflow -- a transport layer over the
same typed graph -- and added the building application and the CONTAM interoperability that
makes the result checkable against an engine other than itself.

**What it adds.**

- A `Drive` is a function of the DRIVERS only, never of the state, which is what keeps the
  assembled Jacobian exactly symmetric; `sources` is in FULL-node order everywhere.
- Per-layer inactive nodes: a node no edge of a layer's kinds touches has no row in that
  layer at all, exported as `noodl.layers.transport.active_interior`. A caller sizing a
  `capacity` must use it.
- `kind_slice`/`flows_of_kind` on the potential layer, `quantity`/`unit` tags on a transport
  layer, and multi-kind transport (one layer advected by several edge kinds).
- Elements: `Duct` (Colebrook friction, unrolled iteration) and `Damper`.
- Drives: `Stack` and `Wind`, with `WindProfile`; profiles are identified by CONTAM's profile
  NUMBER.
- `Model`: several layers on one network stepped together, with Hensen's two couplings --
  `"pingpong"` (one pass) and `"iterate"` (successive substitution at 0.5 relaxation,
  per-instance convergence, diagnostics `passes`/`converged`/`max_change`/`layers`;
  `iterate_max < 2` is refused).
- The building application `apps/building`: `Zone`, `WallMass`, `thermal_layer`,
  `species_layer`, `IdealGasDensity`, `LinearDensity`, `build_model`, `add_large_opening`,
  `mass_orifice`.
- Readers: the CONTAM `.prj` reader (a documented subset -- multi-species, 30-field path
  records -- which REFUSES on a record referencing an unsupported section) with
  `project_to_model`, the `.wth` weather reader, and the four CONTAM source types
  (`CutoffSource` clamped at zero above the cutoff).
- The ContamX driver over `contamxpy` (marked `external`, installed by the `contam` extra).
- `UpstreamDensityPowerLaw`, the upstream-density correction on CONTAM power-law elements
  that the spec required.

**Two remediation tasks were inserted mid-milestone.** `solvers.iterative.gmres` mishandled
Arnoldi near-breakdown -- it tested against `finfo.tiny` where real near-breakdown is about
1e-16 RELATIVE -- which made multi-species transport steady solves fail for about one flow
magnitude in seven; it now tests a relative threshold and freezes the instance at breakdown.
The second was the upstream-density correction above.

**What passes, and at what tolerance.**

| Case | Tolerance | Measured |
|---|---|---|
| Brown-Solvason doorway | 1e-12 relative | 1.04e-15 |
| Ventilated zone, `T = T_o + S/(c_p q + UA)` | closed form | holds |
| Wall RC time constant `m c/(UA)` | 1e-10 | holds |
| Li and Delsante buoyancy, envelope-loss, assisting-wind closed forms | 1e-6 | worst 1.8e-10 |
| Two-zone doorway against an independent scipy oracle | 1e-5 K | holds |
| Hensen's ping-pong versus iterate table | qualitative | holds |
| Gradients through `Model.step`, `Model.steady` and `build_model`-built models to element parameters, drivers, sources and boundary values | finite differences | hold |
| ContamX three-zone steady flows | 1e-3 relative | 9.1e-8 |
| ContamX 24-step transient concentrations | 1e-3 relative (atol 1e-7) | 6.5e-6 |
| ContamX single-zone stack flows at 273.15/283.15/303.15/313.15 K ambient | 1e-3 | 4.4e-5 |

The ContamX figures are against ContamX 3.4.1.7 through `contamxpy` 0.0.9, Windows x86-64
only. The stack row is 4.4e-5 **after** the upstream-density correction; before it, the same
case was off by 1.7e-2. The transient row's 6.5e-6 is the POINTWISE MAXIMUM RELATIVE ERROR
`max |ours - ref| / |ref|` over the whole (25, 3, 1) trace of zone mass fractions -- all 25
steps and all three zones -- taken over the entries where `ref` is nonzero, the first row
being identically zero on both sides. (An earlier version of this table gave 3.4e-6 there,
which is not that metric and is not reproducible as one; 6.5e-6 is the re-measured pointwise
figure. That row's tolerance was also 2e-3, against the spec's 1e-3, and is now 1e-3.)

**The section 6.1 budget table with a heat layer.** One new gate row,
`tests/verification/test_composed_scaling.py::test_composed_model_with_thermal_layer_24_steps_within_budget`
(`slow`): the reference composed model at ensemble 100 for 24 steps, with air, species AND
heat stepped together through `Model.step`, judged against the SAME section 6.1 budgets as
the 100x24 row above -- 12 s forward, 25 s backward, 2000 MB -- because it differs from that
row in the added layer and the `Model.step` dispatch around it. Solver `auto`, `samples: 3`,
medians of three child processes as every other row is. **It misses both time budgets.**

| Measured | Forward | Backward | Peak memory |
|---|---|---|---|
| the gate, run in isolation | 91.189 s vs 12 s (7.60x) FAIL | 38.258 s vs 25 s (1.53x) FAIL | 510.9 MB vs 2000 MB (0.26x) PASS (fwd 118.4, bwd 510.9) |
| the same row inside the report script, back to back with every other row | 170.872 s (14.24x) FAIL | 75.533 s (3.02x) FAIL | 508.5 MB (0.25x) PASS |

`newton_iterations` 4, `linear_iterations_max` 180, `method` `auto`, per
[`benchmarks/composed_scaling_report.json`](benchmarks/composed_scaling_report.json), which
now carries two `(100, 24, "auto")` rows distinguished only by `"thermal"`.

**Today's absolute times are not comparable to the 1b table above, and the machine is why.**
Every PRE-EXISTING row in the new report is 1.8x-4.4x slower than the committed 1b report --
(100, 24, `auto`) forward 40.7 s then against 71.869 s now, (1000, 1, `auto`) 13.3 s then
against 58.340 s now -- which on its own would be indistinguishable from a regression on this
branch. A discriminating experiment separates them: rows (1, 1, `auto`) and (100, 1, `auto`)
were measured at this branch's HEAD and at the merge base `e982efe` in the same venv,
interleaved HEAD/BASE/HEAD, giving forward 0.457 / 0.439 / 0.392 s and 6.735 / 6.211 /
6.608 s. HEAD and base are within noise of each other, so **the branch has not regressed**
at the shapes that experiment measured -- and it measured (1, 1) and (100, 1) only, so it
establishes no regression THERE and does not directly measure the 24-step thermal shape,
which has no counterpart at the base to be compared against: the machine is 2-4x slower
today than when the 1b table was recorded, and noisy within the morning -- (100, 1, `auto`) measured 2.900 s inside the report run and 6.7 s an hour later.
So the figures above cannot be read against the 1b table, and reading them against the
budgets says as much about the machine as about the code.

**The meaningful figure is the WITHIN-RUN ratio.** The report run measured the thermal row
and the plain 100x24 `auto` row back to back on the same machine, and there the thermal
configuration costs **2.38x forward** (170.872 s / 71.869 s) and **3.06x backward**
(75.533 s / 24.714 s), for 508.5 MB of peak against 424.9 MB. That is the cost of the second
transport layer and its `Model.step` dispatch, and it is the number to carry forward.

**The two instruments disagree more widely than they did in 1b.** The isolated gate run
measured this row 1.9x FASTER than the report script did (91.189 s against 170.872 s) --
opposite in sign to milestone 1b, where the gate was 1.2-1.7x SLOWER than the report on
every row. Both call the same measurement functions. The discrepancy is carried into the next
milestone as a measurement defect to understand, as 1b already carried it.

**Spec section 13's trigger has fired.** Its condition -- the 24-step composed gate missing
by more than 2x with the thermal layer -- is met on the forward budget under both instruments
(7.60x and 14.24x), so the milestone-1b spec's section 6.2 routes apply. That is RECORDED
here as a follow-up; nothing was fixed, re-measured to a friendlier verdict, or re-budgeted
for it. The milestone-1b forward-time misses are likewise carried into this milestone
unchanged, and no budget was edited for this row or any other.

One measurement note belongs with the row: the composed
model's transport layer now has 960 active rows rather than 1028, because the street and sewer
nodes carry no airpath edge and per-layer inactive nodes removed them, so the transport half of
a step now measures about 6.6% less work than the milestone-1b table's rows did.

**What is open, recorded rather than resolved.**

1. The opposing-wind three-root case is `xfail(strict=True)`: `coupling="iterate"` converges
   to the wrong (down) root, because the up root is a REPELLING fixed point of the successive
   substitution map at the hard-coded 0.5 relaxation (measured slope -7.756). Any relaxation
   below 0.228 would contract, so adaptive or user-settable under-relaxation is a candidate
   remedy ALONGSIDE the spec's monolithic Newton. Ping-pong TIME STEPPING resolves all three
   roots, and a test asserts that it does.
2. The remaining 4.4e-5 stack residual against ContamX is the `Stack` drive's constant
   node-density hydrostatic column against ContamX's variable-density integration. By
   construction of that column the discrepancy is linear in opening height and independent of
   the temperature difference -- that is what the constant-density form implies, not a fitted
   or measured scaling.
3. `dp_transition`, quadratic, damper and `fan_cvf` elements still use the REFERENCE density.
4. Newton at `atol=rtol=1e-14` hard-fails on a three-zone stack after 50 iterations (residual
   1.22e-13), so coupling tolerances that tight are reachable on one- and two-zone cases only.
5. The `.prj` reader refuses duct networks, AHS, filters, schedules, kinetics, constant wind
   pressure, and `fan_fan` with `mult != 1`. NIST's `test_OneZoneSsStack-UseApi.prj` carries a
   filter and is therefore refused, so the stack parity case uses
   `test_OneZoneWthCtmStack-UseApi.prj`.
6. `build_model` defaults to `coupling="pingpong"`, which is ONE pass, so a default `steady()`
   returns the airflow solved at the INITIAL temperatures. Documented on `build_model`.
7. The spec's proposed `ReferenceCorrection` closure mechanism cannot work -- a closure runs
   before the solve and cannot see flow direction -- so the correction lives inside the
   element, and spec sections 5 and 6.2 need amending.
8. The monolithic coupled Newton and an implicit-function adjoint of the coupling fixed point
   (rather than the unrolled one `iterate` uses today) are both section-13 follow-ups.

## Milestone 3 status

Milestone 3 built a street-network dispersion application ON TOP of the core package
(spec section 1), to check the framework against a domain neither `apps/building` nor the
core's own tests exercise: driver-prescribed transport, junction elimination, and a
non-negative directed routing whose combinatorics have to stay differentiable.

**What it adds.**

- Driver-prescribed transport flows in `Model` (a `TransportLayer` whose flows are written
  by a closure from the drivers, not solved for) and `Photostationary`, the Leighton
  NO/NO2/O3 reaction.
- `apps/street/`: `canyon.py` (the boundary layer, canyon wind and exchange-velocity
  closures), `routing.py` (the north-west-corner non-crossing router and the roof
  closure), `network.py` (`StreetNetwork`, `build_street_model`), `chemistry.py`
  (`photostationary_for_streets`, `street_steady`), `loader.py` (the AQ_DT reader),
  `impaq.py` (the IMPAQ prototype ported as a comparison oracle) and `report.py` (units
  and the NetCDF product).
- The AQ_DT reader (`read_aqdt`), the IMPAQ port, and the MUNICH checks
  (`tests/verification/test_munich.py`, `test_street_parity.py`).

**What passes, and at what tolerance.**

| Case | Tolerance | Measured |
|---|---|---|
| Mass conservation (steady-state nodal balance, atmosphere balance) | 1e-12 | 3.0e-16 |
| Junction elimination against a hand-written dense linear system | 1e-12 | 2.1e-16 |
| `exchange` edge pair against `TransportLayer`'s conduction term | 1e-12 | 1.2e-16 |
| Gradients through the whole model against central differences | 1e-6 | 4e-10 to 1.3e-8 |
| IMPAQ parity 1, four-node network, noodl against the port at `fix_a=True`, `fix_b=True` | 1e-9 | 4.2e-16 |
| The thirteen MUNICH formula pairs | the precision each source publishes | hold |
| MUNICH linearity in wind speed (210/240 degrees, all streets) | < 1e-9 | exactly 2 |
| MUNICH 270-degree canyon-wind-floor fingerprints (street 11, street 9) | against the paper's 1.99451 / 4.00 | 4.5e-4, 3.2e-3 |
| M3-R8 Photostationary chemistry on the synthetic 12-street network | NOx/Ox conservation < 1e-12, PSS < 1e-10 | 9.65e-20, 7.38e-17, 6.80e-16 |
| `leiden_small` parity, noodl against the ported IMPAQ oracle, median street at three sampled steps | asserted per-street `1e-9` (median only, since a routing defect below leaves a worst-case tail) | 3.6e-16, 3.2e-11, 1.9e-11 |

The full suite passes **934 passed, 10 skipped, 9 deselected, 1 xfailed** (coverage
96.65 %).

**THE ISSUE-C RETRACTION.** The prototype's `u_d = sigma_w/(sqrt(2) pi)` is CORRECT, and is
what SIRANE, MUNICH and all three papers use; `sigma_w/sqrt(2 pi)` appears in no source.
Both the framework spec's earlier text and IMPAQ's own prototype docstring called this an
error ("issue C") and proposed a `fix_c`; that reading came from plain-text PDF extraction
flattening the radical over the whole fraction rather than just the 2, and it is retracted
here against BOTH of those sources. It was checked at glyph level in Soulhac et al. 2011
Eq. (5) (p. 7386), Kim et al. 2018 Eq. (3) (p. 613) and Kim et al. 2022 Eq. (B10) (p. 7387),
and against MUNICH's own source, `StreetNetworkTransport.cxx:3273`, and the
`beta = 0.45` Schulte mixing length MUNICH derives from it. There is no `fix_c` in this
codebase.

**What else was found and is NOT a noodl defect.** IMPAQ's `flow_route` mis-permutes its
routing matrix at three-way junctions whose angular sort is a proper cycle -- 12, 8 and 12
of 162 `leiden_small` roads at three sampled forcing steps (worst factor 13.95, 6.78, 5.92)
-- so its matrix does not conserve each street's own flux there; this is a property of the
prototype, not of the ported oracle or of the model. It is exactly why the `leiden_small`
parity row above is asserted on the MEDIAN street (which agrees at machine/solver
precision) rather than on every street: 15, 21 and 12 of the 162 streets at those same
three steps disagree by more than `1e-9`, with a worst case up to a factor of 4.5, tracking
the same mis-permuted junctions. IMPAQ's `fix_b` is only meaningful together with `fix_a`
(it corrects a term `fix_a` introduces). The saved `network_concentration_2024.nc` shipped
with AQ_DT describes a geometry that was rewritten after it was produced (160 edges against
162 `network_transport` features, 94 of the 160 rows naming an osmid that does not match
the feature its own `edge_feature_index` points at), so the saved-product comparison is
skipped with that measurement recorded rather than compared against stale rows.

**The measured runs (17 September 2026, on this machine).** Spec section 11 budgets nothing
for this milestone: both runs are RECORDED, and neither of the two conditional follow-up
triggers fired.

| run | measured | trigger |
|---|---|---|
| `leiden_small`, 162 streets, 230 junctions, the whole of 2024 (2928 forcing steps, chunks of 96) | load 0.381 s, build 0.019 s, solve 27.832 s (0.00951 s/step) | a street row in the composed scaling gate if it exceeds 10 minutes -- it does not |
| `leiden`, 2943 streets, 3829 junctions, one step | load 1.824 s, build 0.803 s, solve 1.524 s | the same row if one step exceeds 60 s -- it does not |

Both runs completed with no refusals: the loader's `emission_key` alignment matched all
selected features, and `pblh_floor=True` kept every `sigma_w` positive on both domains.

**Data facts recorded by the loader (Task 9), because they decide what any of the above
numbers mean.** The AQ_DT products are out of step: 905 canyon features against 904
emission rows, and the `edge_index` alignment misplaces 515 of those 904 rows, so the
loader aligns by the `(osmid, u, v)` emission key instead and refuses `edge_index` on a
mismatch. `edge_emission_rate_nox_kg_per_year` is entirely NaN and is refused by name.
The wind is ERA5 10 m labelled 30 m in the file; the loader defaults to 10 m and refuses
a file whose label disagrees unless `trust_file_height=True` accepts the disagreement (the
label is recorded in `notes`). The background field is a CAMS
mixing ratio (kg/kg) converted to kg/m3 with `RHO_AIR = 1.2041`. `leiden_small` has 162
`network_transport` streets and 230 junctions.

**What is open.**

- The MUNICH idealised case's absolute concentrations are not reproduced -- Kim et al.
  2022's own inputs are unpublished, so only a fitted uniform geometry is available (a
  wind-speed/lidar or WSL follow-up would supply the real one).
- The relative pattern across the twelve streets sits at a worst residual of 50.7 % (7 of
  19 sampled ratios inside 15 %) against the spec's 5 % target, recorded under that same
  stated uniform-geometry assumption rather than absorbed into a widened tolerance.
- `leiden` (2943 streets) is loaded and stepped but not run for a full year; only
  `leiden_small` has a year-long recorded run.
- Chemistry is exercised analytically and on the synthetic 12-street network only; the
  Leiden AQ_DT data is NOx-only, so no real-data chemistry run exists in this milestone.
- `apps/street/impaq.py`, the comparison oracle, keeps the prototype's per-edge Python
  loops by design -- it exists to be compared against, not to be a model path, and nothing
  else in `apps/street/` calls it.

## Milestone 4 status

Milestone 4 built two applications ON TOP of the core package (spec section 1): a gravity
sewer, `noodl.apps.sewer` (water hydraulics on a dendritic tree, a headspace air network
driven by drag and buoyancy, water quality and H2S coupled by two-film transfer), and,
folded in during the spec review, a pressurised water-distribution network,
`noodl.apps.water`, with EPANET 2.2 (via `wntr`) as its oracle. The two interfacial
coefficients that calibrate the sewer's air side, `f_i` (drag) and `f_air` (wall friction),
are CALIBRATED to a single laboratory source (Pescod and Price's Test 8, as tabulated by
Edwini-Bonsu and Steffler 2004), not literature-pinned, so rows A1 and A2 below are
consistency checks against that calibration source, not independent validation.

**What it adds.**

- `apps/sewer/`: `geometry.py` (exact circular geometry and the batched Manning normal-
  depth inversion), `hydraulics.py` (`SewerHydraulics`, the closure-first tree flow, depth,
  and the optional level-synchronous implicit-Euler storage sweep), `air.py` (`Headspace`,
  `Drag`, air density), `quality.py` (Henry's law, the two-film flux, Pomeroy-Parkhurst
  sulfide generation, BOD decay and, since the fix wave, `LateralLoads` -- the spec 4.2
  inflow-concentration drivers `bod_in`/`sulfide_in`, wired as a source term the
  water-quality layer actually reads, FR-21), `network.py` (`SewerNetwork`,
  `build_sewer_model`, `sewer_steady`), `inp.py` (the SWMM `.inp` reader, a documented
  subset) and `report.py`.
- `apps/water/`: `network.py` (`Junction`/`Reservoir`/`Tank`/`WaterPipe`/`Pump`/`Valve`,
  `WaterNetwork`, `build_water_model`, `water_steady`), `elements.py` (`HazenWilliams`,
  `PumpCurve`, `MinorLoss`), `tanks.py` (`TankLevels`, the tank-level and simple-control
  closure), `demand.py` (`PressureDrivenDemand`), `inp.py` (the EPANET `.inp` reader) and
  `report.py`. The two applications share `apps/inpfile.py`, a section-keyed tokenizer, and
  nothing else.
- Two small core extensions: `NodeSource` (`src/noodl/nodesources.py`), a
  potential-dependent nodal withdrawal added to `PotentialFlowLayer` (`node_sources=`) --
  used by `PressureDrivenDemand` and re-exported from the package root -- and the
  closure-carried state keys/per-step transport capacity driver the sewer app's
  continuity-first hydraulics needs (spec section 4.6).
- A core fix (M4-R22, `fix(solvers)`): the batched Newton solver now falls back to its
  damped step for any instance whose residual did not shrink under a full step, rather
  than locking its relaxation factor at 1 -- a dead-end square-root-law headspace edge
  otherwise cycled `dp -> -dp` for the full iteration budget.
- A second core fix (N1): closure-carried state (the sewer's storage sweep, the water
  application's `TankLevels`) is now evaluated from the STEP-START state in every pass
  under `coupling="iterate"`, not fed forward from the previous pass's own output -- one
  `model.step(dt)` used to advance such state by `passes x dt` rather than by `dt`.

**What passes, and at what tolerance.**

| Row | Check | Tolerance | Measured |
|---|---|---|---|
| W1 | pipe flows vs pyswmm KINWAVE | 1e-9 relative | 1.370e-14 |
| W2 | normal depths vs pyswmm KINWAVE | 1e-3 relative | 5.464e-4 (SWMM's own 51-point circular lookup table, not solver noise) |
| W3 | velocities vs pyswmm's binary FLOW_VELOCITY | 1e-3 relative | 6.257e-4 |
| W4 | tracer concentration vs pyswmm KINWAVE / tank-in-series closed form | 3e-5 relative (amendment A4) | 7.811e-6 |
| W5 | Manning inversion round trip, h/D in [0.01, 0.938] | 1e-12 relative | 7.5e-14 worst |
| W6 | surcharge refusal, naming the pipe | exact | holds |
| W7 | storage dynamics reach the quasi-steady fixed point, 200 steps of 60 s | 1e-9 relative | 5.7e-15 (flows), 4.4e-15 (depths) |
| C1 | air nodal residual / power identity / water continuity | 1e-11 / 1e-10 (M4-R20b) / exact | 4.518e-13 / 6.141e-12 / < 1e-15, with the manhole leak built float64 (N2) -- the earlier 1.352e-12 / 1.855e-11 (still inside tolerance) was mostly an arithmetic floor from the leak's default-dtype `Orifice` cast, not a stopping criterion |
| C2 | cross-phase sulfide conservation, moles of S | 1e-12 | equal to rtol 1e-12 |
| C3 | gradients vs Richardson-extrapolated central differences (inflows, `T_head`, leak area) | 1e-6 (M4-R20a) | inflows 3.17e-8 / 3.16e-9 / 1.16e-8, `T_head` 3.57e-8, leak area 9.67e-9, all relative; `f_i` is a `Drag` `Drive` attribute and structurally NOT differentiable (refused by name); `f_air` finite and correctly signed (a learnable-friction test, not a Richardson row) |
| A1 | air/water velocity ratio vs Pescod and Price Table 1 (three points) | inside 20-40 % | 24.139 % / 24.995 % / 25.149 %; closed form to 1.12e-8 relative |
| A2 | Tyneside field range, bracketed (amendment A5) | inside 105-315 m3/h | open-both-ends 1253.78 m3/h, vented (8 cm2) 0.24 m3/h; band reproduced at leak areas 0.355/1.097 m2 |
| A3 | leak-and-fan flow balance / nodal residual / power residual (amendment A6) | 1e-12 / 1e-11 / 1e-11 | 1.234e-14 / 6.3e-15 / 8.06e-13 |
| H1 | Henry's constant vs Sander 2023 | 0.36 +/- 0.01 at 293.15 K | 0.363854 (293.15 K), 0.403418 (298.15 K) |
| H2 | two-film flux, analytic closed form | exact | 0.0 |
| H3 | Henry-equilibrium fixed point, a genuine `Model.step` run (600 steps, dt = 60 s) on one CLOSED manhole | 1e-10 | 1.4e-16 relative gap; the assembled tree cannot reach equilibrium since its leaks and outfall vent H2S to a zero-concentration boundary |
| S1 | Pomeroy-Parkhurst rate, closed form | exact | 0.0 |
| FR-21 | lateral BOD load wiring, one step: `bod_in = 0.3` kg/m3 at a single headwater manhole, against the transport layer's own backward-Euler closed form | exact | 0.0 relative (bit-exact); every other manhole's source stays exactly zero |
| FR-22 | `sewer_diurnal.py`'s tracer, run through the model's own quality layer (`LateralLoads` + `SulfideGeneration`) rather than hand-resolved, Richardson over dt = 10 s / 20 s vs SWMM | 3e-5 relative | 8.05e-6 (single-dt operator-split error 3.47e-3 at dt = 60 s, 5.79e-5 at dt = 1 s) |
| G1 | sewer golden regression | 1e-10 | 0.0 |
| D1 | `twoloop_si.inp` heads and flows vs EPANET 2.2 (wntr) | 1e-6 relative | 4.361e-7 (heads), 8.090e-8 (flows); nodal continuity 2.093e-14 against 1e-13 |
| D2 | Net1 single period: heads, flows, pump head gain vs EPANET | 1e-6 relative | 7.059e-8 (heads), 2.868e-6 (flows, worst pipe 113, against 1e-5), 1.189e-7 (pump gain) |
| D3 | Net1 24 h tank level with tank-level pump controls | 2e-4 m absolute | 8.181e-5 m worst of 25 reported steps (26 hydraulic sub-steps) |
| D4 | Darcy-Weisbach pipe vs EPANET D-W (friction-factor formulae differ) | recorded band | 3.822e-2 (heads, band 3.8e-3..3.8e-1), 4.446e-1 (flows, band 4.4e-2..4.4) |
| D5 | pressure-driven demand vs EPANET `DEMAND MODEL PDA`, now read from the file's own `[OPTIONS]` (N5) rather than re-supplied by hand | 1e-5 | 3.521e-7 (heads), 2.184e-7 (delivered demands) |
| D6 | loop consistency, head loss around every cycle-basis loop | 1e-12 | 0.0 |
| D7 | gradients vs central differences (nodal demands, Hazen-Williams roughness, pump `h0`, tank area through `TankLevels.advance`; a single steady solve does not itself reach tank area, N6) | 1e-6 x scale | sources 3.212e-6 (allowance 4.444e-4); roughness 2.505e-10 (allowance 6.259e-8); pump `h0` 1.007e-6 relative; tank area 2.753e-9 relative |
| D8 | TRACE water quality on `twoloop_trace.inp` (single-source smoke row) | 1e-3 | 6.438e-12 percentage points |
| G2 | water golden regression | 1e-10 | 0.0 |

The full suite passes **1225 passed, 10 skipped, 9 deselected, 1 xfailed** (coverage
96.41 %), ruff clean.

**The measured runs (18 September 2026, on this machine).**

| run | measured | budget |
|---|---|---|
| `benchmarks/sewer_diurnal.py`: 24 h at 60 s, 8 instances (`f_i` and leak area varied 0.5x-1.5x across the batch, FR-18), storage on | 1440 steps x 8 instances in 235.28 s; peak headspace H2S per manhole, instance 0: 9.336 / 7.702 / 12.108 / 11.680 / 13.509 ppm | 60 s -- **FAIL**, recorded as FR-19 rather than loosened |
| `benchmarks/water_eps.py`: Net1, 24 h extended period, 8 SEQUENTIAL demand multipliers (N15: one model per multiplier, not a batched leading dimension) | 210 hydraulic sub-steps in 11.85 s | recorded, no budget set (spec section 7 sets one for the sewer benchmark only) |

**Data facts.** SWMM's KINWAVE (kinematic wave) routing, not its dynamic-wave engine, is
the parity target for the sewer rows -- the theory section explains why. EPANET 2.2 is
reached through `wntr`'s `EpanetSimulator`, whose result arrays are float32 (a measured
~4e-7 heads / ~8e-8 flows floor that the D1/D2 tolerances above are set around), and `wntr`
itself drags in a measured ~351 MB of mandatory dependencies (scipy, pandas, numpy,
matplotlib and friends); both `pyswmm` and `wntr` are dev-extra, test-only, and neither
sewer nor water applications need anything beyond torch and the existing `sparse` extra to
run.

**Coefficient register (unverified defaults; spec section 11).** `f_air` (air wall
friction, default 0.02, reported range 0.015-0.045, Edwini-Bonsu and Steffler 2006,
paywalled); `C_d`/`A_leak` (manhole leak orifice, 0.6 / 8 cm2, generic/scenario values);
the two-film `K_L a` form's constants 0.86 and 0.20 (form corroborated, Yongsiri et al.
2004, paywalled); the Pomeroy-Parkhurst `M'` and `m` (vendor defaults, unverified against
the 1977 paper); `k_BOD` (generic bulk-water value, Metcalf and Eddy); `k_gas`, the
gas-phase H2S sink (no verified value, default 0); `pKa`'s temperature dependence
(approximated as constant). `f_i` is calibrated, not unverified in this sense -- see the
framing paragraph above.

**What is open.**

- FR-19: `benchmarks/sewer_diurnal.py` takes 235.28 s against its 60 s budget -- recorded,
  not loosened (profiled: the storage sweep's own Newton root-finds are ~56 % of one step,
  the two transport GMRES solves ~31 %, the air Newton solve ~12 %).
- FR-20: the outfall node is a dead end in the air graph (physically inert edge; a design
  tidy-up, not a correctness defect).
- FR-26: `LateralLoads` overwrites a caller-supplied `water_quality.sources` driver while
  `H2STransfer` adds to it, so an industrial discharge given as a source driver is silently
  dropped when `quality=True`; add to it or refuse the key by name (found by the wave
  re-review, not yet fixed).
- FR-27: `TankLevels.event_step` indexes its level/rate vectors positionally on the first
  dimension, so a batched `(B, n_tanks)` rollout would index the batch; no batched caller
  exists yet.
- FR-31: on a Darcy-Weisbach file with both `SPECIFIC GRAVITY` and `VISCOSITY` non-default,
  the effective kinematic viscosity is divided by the specific gravity once too often
  (EPANET's `VISCOSITY` is already relative kinematic); no fixture sets both.
- FR-25: `tank_inflow` (`apps/water/network.py`) still reads `PotentialFlowLayer`'s private
  `_accumulate` directly; `element_for` (FR-13) covers looking up an element by kind, not
  this per-node flow accumulation, so a second public accessor is a recorded follow-up.
- N15: `benchmarks/water_eps.py` runs its 8 demand multipliers SEQUENTIALLY, one model per
  multiplier, not as a batched leading dimension -- `TankLevels.event_step` shortens each
  instance's step to its own next control crossing, so a genuinely batched rollout would
  need a per-instance step or an oversampling global-minimum step; recorded as a design
  limitation rather than fixed.
- The diffusive-wave sewer formulation (a surface-elevation potential layer with
  depth-dependent conveyance, Newton-solved) and the MIXED regime (some pipes free-surface,
  others surcharged, with transitions) are parked, not implemented -- the natural
  generalisation if a dynamic, looped or backwater sewer case is ever needed.
- Headspace heat and moisture layers are not modelled; headspace temperature is a driver.
  The UWO (Western Ontario) dataset is not used. Surcharge, backwater, dynamic-wave
  hydraulics, pumps, weirs, orifices and force mains in the sewer are out of scope; the
  relative-velocity drag form is a recorded follow-up requiring an element-side
  formulation (the theory section explains the `Drive`-cannot-see-`phi` constraint that
  rules it out here).
- In the water application: PRV, PSV, PBV and GPV valves, time-based controls and
  `[RULES]`, variable-speed pumps, volume curves, emitters, leakage models, energy and cost
  reports, and Chezy-Manning (SI constant unverified) are refused by name, not implemented.
  D8's trace row is a single-source smoke test; a genuinely discriminating two-source trace
  (EPANET's Net3-style "percent of Lake water") is a recorded follow-up.
- The paywalled coefficient sources in the register above (Edwini-Bonsu and Steffler 2006;
  Yongsiri et al. 2004; Pomeroy and Parkhurst 1977) have not been obtained; if supplied, the
  plan pins the corresponding defaults and upgrades their status without any code change,
  since every entry is already a parameter.

## Milestone 4b status

Milestone 4b added `CapacitatedTransferLayer` (`src/noodl/layers/capacitated.py`), the
fourth layer type in `src/noodl/layers/` alongside `PotentialFlowLayer`, `TransportLayer`
and `Reaction` -- the framework spec's fourth way of determining edge flows (alongside
potential-flow Newton solves, driver-prescribed flows and closure-computed flows):
clipping per-edge requests against arc capacity and receiver storage headroom rather than
solving for a potential, with proportional sharing when more than one edge competes for one
node's headroom. It is validated against **WSIMOD 0.8.1's own output** -- WSIMOD
(Dobson, Liu and Mijic; JOSS 2023, GMD 2024) is a published Python water-systems model whose
own `Arc`/`Node` push/pull semantics this layer's hard-clip mode reproduces -- not against
an independent measurement, exactly the same oracle relationship milestone 4 has with
pyswmm and EPANET.

**What it adds.**

- `CapacitatedTransferLayer`: hard-clip, `"smooth"` (softmin/softplus at temperature `tau`)
  and `"projection"` (a per-node QP solved via `noodl.solvers.scalar.solve_monotone`)
  modes, selected at construction. A real bug was found and fixed in `"projection"`
  mode's sharing site: the first implementation reused the same preference-proportional
  share VALUE formula hard-clip mode uses, which is a function of preference weights and
  total headroom alone and therefore has a provably zero cross-gradient between competing
  edges -- delivering none of the mode's actual purpose (gradients flowing through which
  arc absorbs a constraint). The fix routes the sharing site through a real coupled QP whose
  KKT stationarity reduces to one scalar monotone equation per node, shared by every
  competing edge; the genuine cross-gradient this produces, `d(f_BD)/d(r_CD) = -0.5` on this
  specific SYMMETRIC-preference diamond fixture, was hand-derived and is checked directly
  (sign and magnitude) by `test_projection_mode_sharing_has_nonzero_cross_gradient`. The
  general sharing mechanism -- the KKT derivation `d(f_i)/d(r_j) = -(1/preference_i) /
  sum_k(1/preference_k)`, of which -0.5 is the symmetric-weight special case -- was
  separately hand-verified against DIFFERENT, asymmetric-preference fixtures during Task 4's
  review (`pref=[1,3]` and `pref=[1,2,5]`, confirmed to 4 decimal places; see the ledger at
  `.superpowers/sdd/2026-09-18-milestone-4b-wsimod/progress.md`, lines ~241-244). That
  confirms the mechanism generalises correctly, not that -0.5 itself does -- for asymmetric
  weights the value is generically different (e.g. `pref=[1,3]` gives -0.25, not -0.5), and
  no committed test asserts the literal -0.5 value under asymmetric weights.
- `tests/verification/_wsimod_oracle.py`: a request-capture harness that monkeypatches
  `wsimod.arc.Arc.send_push_request`/`send_pull_request` to record every per-arc
  `requested`/`realised` pair WSIMOD itself computes while running its own
  `quickstart_demo` and `oxford_demo`, and the committed fixtures those runs produced
  (`tests/data/wsimod/{quickstart,oxford}_{topology.json,events.csv}`) -- WSIMOD need not
  be installed at all to run the parity tests themselves, only to regenerate the fixtures.
- `benchmarks/wsimod_oxford.py`: batched throughput on the `oxford_demo` topology (below).

**What passes, and at what tolerance.**

| Row | Check | Tolerance | Measured |
|---|---|---|---|
| W1 | Hard-clip mode vs WSIMOD's own realised flows, `quickstart_demo`, all 1,456 timesteps | 1e-9 absolute | 1.11e-16 |
| W2 | Hard-clip mode vs WSIMOD's own realised flows, `oxford_demo`, full 1,456-timestep run, 20 of 21 arcs (see limitation below) | 1e-6 absolute | 1.86e-9 |
| W3 | `"smooth"` mode (`tau=1e-3`) vs W1's own hard-clip noodl output, `quickstart_demo` | 3e-3 absolute | 1.79e-3 |
| W4 | `"projection"` mode vs W1's own hard-clip noodl output, `quickstart_demo` | 1e-9 absolute | 9.10e-13 |
| W5 | Conservation (`sum(f) + overflow` in equals out plus `ds`), exact by construction, both demos, all three modes | exact | holds (no oracle needed) |
| W6 | `torch.autograd.gradcheck` through `"smooth"` and `"projection"` on the diamond fixture | analytic finite-difference | holds (`test_gradcheck_smooth_and_projection_on_diamond`) |
| W7 | Fixed `n_passes=5` sharing loop vs a hand-converged reference on a synthetic multi-out-arc node | exact once converged | holds (`test_step_conserves_with_n_passes_1_vs_5`) |

**A significant limitation of what W1/W2 actually demonstrate (amendment A7, design spec
section 9) -- read this before citing W1/W2 as validating the capacity clip itself.**
Neither `quickstart_demo` nor `oxford_demo` ever exercises a genuine arc-capacity clip: of
`quickstart_demo`'s 6 arcs and `oxford_demo`'s 21 arcs, all but one sit at WSIMOD's own
`UNBOUNDED_CAPACITY` (1e15) for their entire run, and the one finite-capacity arc
(`abstraction_to_farmoor`, capacity 50000.0) never sees its request exceed ~30934 across
oxford's full 1,456-day run. W1 and W2 -- the two rows this milestone's WSIMOD-oracle
strategy rests on -- therefore validate `CapacitatedTransferLayer`'s clip arithmetic only on
the identity path (`min(x, c_arc) == x`), never on the branch where `c_arc` actually binds.
**This is NOT a gap in the layer's own correctness**: the clip-against-`c_arc` mechanism is
directly, rigorously unit-tested on synthetic fixtures with a deliberately tight capacity
(e.g. `test_step_hard_clip_above_arc_capacity_is_capped` and the proportional-sharing
tests). The gap is narrower and specific to what the WSIMOD comparison itself has shown:
WSIMOD's own numbers have never been used to cross-check the layer's behaviour AT the exact
point a capacity binds, because neither reference demo happens to push any arc that far --
the same kind of distinction the milestone 4 section above draws between a coefficient
CALIBRATED to one source and one INDEPENDENTLY VALIDATED. The same applies, for a separate
reason, to the clip's OTHER bound: both fixtures set `s_max = inf` at every node (the harness
captures per-arc capacity only; WSIMOD's node science, not its arcs, decides what a node
accepts), so the receiver-headroom clip is the identity everywhere and the proportional-
sharing branch never runs against WSIMOD's numbers either -- neither half of
`min(r, c_arc, h_receiver)` is cross-checked against WSIMOD at a binding point, and both are
covered instead by the synthetic fixtures with finite bounds. A separate, unrelated exclusion in
the same W2 fixture: `oxford_demo`'s `sewer_to_wwtw` arc is excluded from the strict W2
comparison (20 of 21 arcs compared) because it shows 185/1456 mismatched timesteps
root-caused to WSIMOD's own `WWTW` node applying an internal
`treatment_throughput_capacity`/stormwater-tank constraint -- a NODE-level throughput cap
this milestone's harness does not extract and `CapacitatedTransferLayer` does not model
(`s_max` here is a storage-headroom bound, not a per-step throughput-rate cap), not that
arc's own capacity (still 1e15, unbounded, throughout).

**The measured runs (19 September 2026, on this machine).**

| run | measured |
|---|---|
| `benchmarks/wsimod_oxford.py`: `oxford_demo` topology (21 arcs, 18 nodes), hard-clip mode, 1,456 steps, random per-instance requests scaled to each arc's own capacity | batch_size=1: 1.004 s; batch_size=10: 1.338 s; batch_size=100: 1.439 s (WSIMOD's own single-instance run: ~4 s -- NOT a fair batched-vs-unbatched comparison, spec section 7; no budget is set for this row) |

The full suite passes **1250 passed, 10 skipped, 9 deselected, 1 xfailed** (coverage
96.39 %), ruff clean.

**Out of scope.**

- The WSIMOD `Node` wrapper (embedding a noodl `Model` as a live WSIMOD `Node` via
  `push_set`/`pull_set`/`push_check`/`pull_check`, so a noodl model could sit inside a
  running WSIMOD orchestration) -- named in the framework spec's roadmap for 4b, deferred
  per Maarten's own scope decision (design spec section 1).
- The other nine WSIMOD pollutants (do, org-phosphorus, phosphate, ammonia, solids, cod,
  ph, nitrate, nitrite, org-nitrogen) -- the layer is species-count-agnostic, so this is
  purely a matter of widening a future `TransportLayer`'s species list and the fixture
  capture, not a `CapacitatedTransferLayer` change.
- Species/quality transport (volume, temperature, BOD) is not wired up despite an earlier
  scope note to the contrary (amendment A5): the WSIMOD oracle harness captures per-arc
  volume only, not WSIMOD's `temperature`/`bod` VQIP fields, and no task attaches a
  `TransportLayer` riding on a capacitated layer's `"<name>.q"`.
- Time-varying arc capacities and storage bounds: construction-time buffers only, since
  WSIMOD's own capacities are static within a run.
- Recorded follow-up closing the A7 gap above: a small synthetic 2-3 node fixture with a
  deliberately tight `c_arc`, captured the same way (WSIMOD's own push/pull on that
  fixture, not just noodl's own unit tests) -- not built in 4b to avoid unilaterally
  expanding an already-approved 8-task plan.

## Milestone 5 status

Milestone 5 covers the first three items of framework spec section 8's milestone list entry
"graph-union demonstration, inverse examples, benchmarks" (the remaining item of that entry
is editorial rather than engineering work and is not part of this repository). It couples a real street segment (the `leiden_small` AQ_DT
domain milestone 3 already reads) to a real CONTAM building (a milestone 2 `.prj` fixture)
without editing either application, differentiates through the coupled solve for three
inverse examples, and measures -- rather than assumes -- both the building's own back-effect
on the street and the error a loosely-coupled file-exchange workflow would make in its place.
Union scope is deliberately narrow (design spec section 1): the street-ambient/building-
boundary pair only, not the wider sewer+street+building union framework spec section 8 also
names.

**What it adds.**

- `src/noodl/couple.py` (new module): `ValueLink` (one shared-node value relationship --
  `from_model`/`from_key`/`from_index`, `to_model`/`to_key`/`to_index`, `convert`, `two_way`,
  `convert_back`, `sources_key`), `DriverAlias` (one driver value aliased across models, each
  target through its own registered conversion), `CoupledModel` (`.step(state, drivers, dt,
  diagnostics=)`; no `.steady` -- see "What is open"), `union(models, shared=, *,
  relaxation=0.5, iterate_rtol=1e-8, iterate_atol=0.0, iterate_max=50, substeps=)`,
  `transport_boundary_inflow` (the net mass inflow at a transport layer's boundary node, built
  from `net.accumulate`/`net.upwind` alone, since `Model.ports()` reports boundary flows only
  for potential layers), and the unit/angle conversion registry
  (`concentration_to_mass_fraction`, its inverse, and the two street-radians/CONTAM-degrees
  wind-direction conversions), each entry carrying the units it maps between so that a
  `ValueLink` whose `convert` does not carry the FROM layer's `unit` to the TO layer's -- or a
  `convert=None` link between unequal units -- is refused at construction. `iterate_max`
  defaults to 50 because the demo itself needs 22-28 passes to `rtol=1e-10` and ~21 passes per
  coupled hour at the default `rtol=1e-8`; `iterate_atol=0.0` makes the criterion purely
  relative, so a legitimately-zero shared value needs a positive `iterate_atol` to converge.
- `Model.current_flows(name, state, drivers)`: a transport layer's branch flows for a given
  state/drivers without stepping the model -- closures run first, then the layer's own owner
  is consulted; when the owner is a potential layer and the state carries no `"<owner>.q"`
  yet (a step's first pass), the owning layer is solved fresh for these drivers, warm-started
  from `"<owner>.phi"` when available. Needed because a transport layer's flow is never a
  plain driver-dict entry before `Model.step` runs: it is either closure-written (visible
  only inside `Model._pass`) or potential-owned (sitting in the OUTPUT state, not the input).

**Why orchestration, not graph merging (design spec section 3).** `union` never copies,
rebuilds or mutates either model's `Network`, layers or closures; it exchanges named
driver/state VALUES between two ordinary `Model.step` calls each outer step, converged to a
fixed point when a link is two-way -- the standard Dirichlet-Neumann style flux/value
exchange used to couple independently-solved subdomains, applied here across two
`noodl.Model`s rather than two mesh partitions of one solver. Two corrections came from
reading the apps rather than the framework spec's illustrative pseudocode: the street
application has **no potential layer at all** -- `build_street_model` builds exactly one
`TransportLayer` whose flows on every kind are closure-written by `StreetFlows`, so there is
no Newton-solved state for a merged graph to expose; and closures are net-bound and
application-specific (`StreetFlows` is constructed bound to the specific graph object
`build_street_model` built, with its own canyon-wind arguments), so a literal graph merge
would need to rebuild closures from inside `couple.py`, which would require importing
`noodl.apps.*` and breaks "neither model is modified" as badly as editing the app itself.
PyTorch's autograd tracks the computation graph, not Python object structure, so two ordinary
`.step` calls sharing tensors between them are exactly as differentiable in one `.backward()`
as a merged model would be -- there is no `CoSim`/ports-style differentiability boundary here.
Two convention facts sit at the join: the forward value needs a unit conversion (street
`kg/m3` to the building's CONTAM-convention `kg/kg` mass fraction, via the building's own
`rho_amb` driver) while the backward value needs none (both sides are already a flux, kg/s of
pollutant, once the inflow is computed in the transport layer's own units -- design spec A2);
and the two models' wind-direction drivers use different conventions entirely -- the street's
radians counter-clockwise from east, the direction the wind blows TOWARD, against CONTAM's
`Wd`, degrees clockwise from north, the direction the wind blows FROM -- so `DriverAlias`
carries a registered conversion per target rather than aliasing the raw number (design spec
A3). Ambient temperature is not among the aliased drivers in this milestone's demo (A4).

**A core fix, found while building the second inverse example.** `PotentialFlowLayer.solve`
threads every key of a shared `drivers` mapping into `implicit_solve`'s adjoint params,
including keys a given layer's own residual never reads -- for a coupled model, that is
routinely another layer's driver (here, the building's `air` layer receiving
`species.x_boundary`, which requires grad because it is two-way coupled to the street's
emissions, but which only the `species` layer consumes). When the only grad-requiring params
are such functionally-unused ones, the layer's residual has no autograd graph at all, and
`torch.autograd.grad` refuses to start from an output that does not itself require grad --
`allow_unused=True` excuses individual unused INPUTS, not an output with no graph whatsoever.
`_Implicit.backward` (`src/noodl/solvers/implicit.py`) now short-circuits to all-`None`
gradients whenever the residual itself does not require grad, which is exact when the
graph-less residual is provably independent of every param (as here), but is, by
construction, indistinguishable from a param whose graph was accidentally severed upstream
(a stray `.detach()`) -- the same trade-off `allow_unused=True` already makes one level down
for the partially-connected case. A `params_may_be_unused` precondition on `implicit_solve`
naming which params a residual is allowed to ignore is a recorded follow-up (see "What is
open"). Covered by two unit tests in `tests/solvers/test_implicit.py`:
`test_implicit_solve_ignores_a_grad_requiring_but_functionally_unused_parameter` and
`test_implicit_solve_mixed_used_and_unused_parameters`.

**What passes, and at what tolerance.**

| Case | Tolerance | Measured |
|---|---|---|
| Two-way step is a fixed point of one step from the start state | rtol 1e-9, atol 1e-14 | holds |
| Gradient across the join vs. central differences | rel 1e-5 | holds |
| `substeps={"building": k}` calls the fast model exactly `k` times per one slow step, glue-derived boundary held constant across them | exact | holds (`k=6`, calls = `k x passes`) |
| Synthetic back-coupling, 2x3 m canyon, segment `r2`, one-way (`4.16974e-08`) vs. two-way (`4.14518e-08` kg/m3) | measured, not budgeted | 0.589 % relative change, 28 passes to converge |
| Real `leiden_small` back-coupling, segment `783` (busiest street, forcing step 1000), steady (`2.07911e-07`) vs. coupled (`2.07896e-07` kg/m3) | measured, not budgeted | 7.19e-3 % relative change (7.2e-5), 22 passes to converge -- correctly signed (building is a sink) but negligible for one real building, as anticipated |
| Loose sequential file-exchange vs. the two-way coupled result, building indoor mass fraction | measured, not budgeted | 0.589 % discrepancy -- equal to the street-side back-coupling change, as expected for a boundary response linear in the shared value |
| Inverse example 1: leakage coefficient calibration through the join, gradient running through the coupled solve and back through the street model | relative error < 0.05 | 1.29e-4; final loss 3.4574e-08 |
| Inverse example 2: source attribution by one adjoint pass vs. central differences | rel 1e-4 | r1 3.6e-7, r2 5.3e-7; r3 structurally zero (both attribution and FD below 1e-12) |
| Inverse example 3: latent infiltration, one measured path recovers all four branch flows (`project_measured`) | rtol 1e-10, atol 1e-14 | exact recovery |
| Inverse example 3: 200-sample cycle-amplitude ensemble, seeded from the chord edge's own solved flow, mean vs. the hand-solved flows | rtol 0.05 | holds, correctly signed |

**The measured runs (19 September 2026, on this machine).**

| run | measured |
|---|---|
| `benchmarks/coupling_street_building.py`: the headline union, 6 coupled hours (60 building sub-steps per street hour), batch sizes 1/10/100 | batch_size=1: 41.221 s, 116 outer passes; batch_size=10: 38.541 s, 128 outer passes; batch_size=100: 134.772 s, 128 outer passes -- 10 -> 100 is 3.5x the time for 10x the batch (sub-linear); no budget is set (spec section 6) |

The full suite passes **1306 passed, 10 skipped, 10 deselected, 1 xfailed** (coverage
96.47 %), ruff clean.

**What is open, and out of scope.**

- `CoupledModel.steady` is not built, though design spec section 3 point 6 names it in the
  public surface (R2-11).
- Adaptive relaxation: `relaxation` IS a `union`/`CoupledModel` parameter (default 0.5), but
  it is never adapted during the iteration -- at 0.5 the demo takes 22-28 passes to
  `rtol=1e-10`. README milestone 2's open item 1 already names under-relaxation as a candidate
  remedy for the same successive-substitution risk (a repelling fixed point at a given value),
  and this milestone's own numbers are consistent with 0.5 being slow rather than wrong here.
- The `params_may_be_unused` precondition on `implicit_solve` (R2-14, above): the
  `_Implicit.backward` short-circuit cannot distinguish a legitimately-unused driver key from
  an accidental upstream `.detach()`.
- Single-species scope: `transport_boundary_inflow` assumes a single-species transport layer
  (milestone 5's global constraint); a multi-species boundary inflow is unbuilt.
- Two-way scope, refused at construction (R2-20): a two-way link requires a SINGLE flow kind
  on its TO layer and `n_species == 1` on BOTH layers. The first is the feedback flux's own
  limit (`transport_boundary_inflow`; a typical `.prj` has several edge kinds, so this is a
  real pairing to refuse); the second is a layout ambiguity, since a stacked `(n_i, 1)` and a
  reduced `(n_i, K)` with `K == n_i` are the same shape.
- The coupling fixed point is differentiated by UNROLLING: every pass stays on the autograd
  graph, so memory grows with the pass count. An implicit-function treatment (one adjoint
  solve at the converged state, as `implicit_solve` does for Newton) is a follow-up -- it is
  also the main cost of the 4-minute calibration example, which unrolls the whole iteration on
  every one of its optimiser steps.
- Held-constant sub-stepping: the fast model's glue-derived boundary value is held constant
  across the slow model's inner steps rather than interpolated -- a follow-up if that policy
  proves too coarse for a real pairing (design spec section 3 point 4, section 9).
- `Model.current_flows`'s first-pass branch re-solves the owning potential layer from scratch
  when a step's state carries no `"<owner>.q"` yet, rather than reusing the solve the pass
  itself is about to perform -- a reuse of the pass's own solve is a recorded follow-up.
- Caching the owner's solve across sub-steps (R2-21): with `substeps={"building": 60}` the
  building's `air` potential layer is Newton-solved 60 times per pass at IDENTICAL drivers
  (~1500 identical solves per coupled hour, at 25 passes). Caching the owning potential
  solve across sub-steps while its drivers are unchanged is the highest-leverage optimisation
  available here; it is not implemented.
- The street's canyon AIRFLOW field is unaffected by the building's ventilation (R2-21): the
  street's flows are closure-prescribed from the wind, so only the POLLUTANT balance is
  coupled -- the building removes and returns mass at the shared node, it does not change how
  the canyon ventilates.
- Ambient temperature is not coupled in this milestone's demo (A4): the building's
  temperature enters only through `rho`/`rho_amb`, computed once from the `.prj`'s `Ta`; the
  AQ_DT forcing carries no temperature field.
- Species identity is not coupled either: the demo joins the street's `nox` to the CONTAM
  fixture's only species, `sarin`. `union` relates two layers by UNIT, not by species
  identity, and both are passive tracers here -- a real pairing of named species is a
  modelling decision the glue does not make.
- The wider sewer+street+building three-way union is deferred (decision 1) -- the same
  orchestration mechanism is expected to extend to it, but it is not built or tested here.
- `CoSim`/ports and the WSIMOD `Node` wrapper stay out of scope (decision 2).
- The leakage-calibration test (inverse example 1) is marked `slow` (~4.3 minutes; R2-12),
  deselected by default like the milestone-1b and milestone-2 acceptance gates.


## Performance re-baseline after framework hardening (20 Sep 2026)

The framework-hardening branches changed the transport layer's exponential action, the
closure contract and the coupler after the milestone-1b section 6.1 numbers above were
recorded (17 September). `benchmarks/report_composed_scaling.py` was re-run unmodified on
this code, same settings (`samples: 3`, `torch_num_threads: 14`, both `auto` and `cg`, the
reference composed model: 1030 nodes, 2193 edges). The old report is kept as
[`benchmarks/composed_scaling_report_2026-09-17.json`](benchmarks/composed_scaling_report_2026-09-17.json);
the new one is [`benchmarks/composed_scaling_report.json`](benchmarks/composed_scaling_report.json)
(generated 2026-09-20T14:17 UTC). `all_budgets_met` is `false` in both.

**Machine state.** The machine was not idle: another session's Python process was active at
launch and stayed active through the run (a ~2.4 GB resident process, unrelated to this
benchmark, present before it started and never exited). Both reports otherwise share the same
thread count (`torch_num_threads: 14`). Because of this, absolute times are **not** a clean
before/after of the code change -- the largest single row, `cg`/ensemble 100/24 steps
forward, moved from 149.9 s to 221.3 s, a 48% increase that is at least partly contention, not
regression. The comparison that isolates the code change from machine noise is the controlled
A/B that Task 28 runs; this re-baseline's job is only to re-measure on the current code and
record the new absolute numbers under the noisy conditions they were actually taken under.

**Result: every latency budget in the table remains unmet, on both solvers, on every row.**
Two rows that passed their backward budget on 17 September (`auto`, ensemble 100, both step
counts) now fail it as well -- the table has strictly more failures than before, not fewer,
though under a machine state this run cannot separate from noise. All four peak-memory
budgets still pass, on both solvers, and **both shape gates still pass** (peak RSS vs nodes
1.06x, matvec time vs edges 0.32x, budget 2.5x each).

| Ensemble | Steps | Solver | Thermal | Forward: old -> new | Budget | Backward: old -> new | Budget | Peak memory: old -> new | Budget |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `auto` | no | 0.205 s -> 0.512 s (10.24x) | 0.05 s FAIL (was 4.09x FAIL) | 0.157 s -> 0.452 s (4.52x) | 0.1 s FAIL (was 1.57x FAIL) | 66.3 MB -> 67.5 MB (0.67x) | 100 MB PASS |
| 100 | 1 | `auto` | no | 2.900 s -> 8.899 s (17.80x) | 0.5 s FAIL (was 5.80x FAIL) | 0.912 s -> 3.194 s (3.19x) | 1.0 s **FAIL (was PASS, 0.91x)** | 128.7 MB -> 127.7 MB (0.13x) | 1000 MB PASS |
| 100 | 24 | `auto` | no | 71.869 s -> 212.044 s (17.67x) | 12 s FAIL (was 5.99x FAIL) | 24.714 s -> 119.951 s (4.80x) | 25 s **FAIL (was PASS, 0.99x)** | 424.9 MB -> 420.1 MB (0.21x) | 2000 MB PASS |
| 1000 | 1 | `auto` | no | 58.340 s -> 50.679 s (10.14x) | 5 s FAIL (was 11.67x FAIL) | 12.375 s -> 14.688 s (1.47x) | 10 s FAIL (was 1.24x FAIL) | 719.5 MB -> 726.0 MB (0.09x) | 8000 MB PASS |
| 1 | 1 | `cg` | no | 0.482 s -> 1.099 s (21.98x) | 0.05 s FAIL (was 9.64x FAIL) | 0.283 s -> 0.435 s (4.35x) | 0.1 s FAIL (was 2.83x FAIL) | 46.1 MB -> 45.1 MB (0.45x) | 100 MB PASS |
| 100 | 1 | `cg` | no | 6.747 s -> 8.322 s (16.64x) | 0.5 s FAIL (was 13.49x FAIL) | 2.370 s -> 2.998 s (3.00x) | 1.0 s FAIL (was 2.37x FAIL) | 127.6 MB -> 128.0 MB (0.13x) | 1000 MB PASS |
| 100 | 24 | `cg` | no | 149.870 s -> 221.318 s (18.44x) | 12 s FAIL (was 12.49x FAIL) | 50.835 s -> 79.755 s (3.19x) | 25 s FAIL (was 2.03x FAIL) | 422.4 MB -> 419.0 MB (0.21x) | 2000 MB PASS |
| 1000 | 1 | `cg` | no | 53.673 s -> 48.894 s (9.78x) | 5 s FAIL (was 10.73x FAIL) | 14.262 s -> 13.511 s (1.35x) | 10 s FAIL (was 1.43x FAIL) | 726.2 MB -> 725.7 MB (0.09x) | 8000 MB PASS |
| 100 | 24 | `auto` | **yes** | 170.872 s -> 248.178 s (20.68x) | 12 s FAIL (was 14.24x FAIL) | 75.533 s -> 68.121 s (2.72x) | 25 s FAIL (was 3.02x FAIL) | 508.5 MB -> 497.6 MB (0.25x) | 2000 MB PASS |

The four `auto`, non-thermal forward rows -- the shipped default, the numbers a reader of
this table cares about first -- went 0.205 s -> 0.512 s (ensemble 1), 2.900 s -> 8.899 s
(ensemble 100, 1 step), 71.869 s -> 212.044 s (ensemble 100, 24 steps) and 58.340 s -> 50.679 s
(ensemble 1000, 1 step, the one row that got faster). **Unmet budgets, stated plainly: every
forward budget and every backward budget in the table, on both solvers, on every row and
including the thermal row.** Only the four peak-memory budgets and the two shape gates are
met, unchanged from 17 September.

**Exponential-action work counts (R3).** The norm-scheduled exponential action with the
mean-diagonal shift (commit `e451872`, PR-1) changed the work three cases take:

- Pure decay, `dx/dt = -500x`, `x(0) = 10`, `dt = 50`: 184,459 matvecs before the fix (the
  review's instrumented count) -> 1 matvec after (the mean-diagonal shift makes `M - mu*I`
  vanish for this case).
- The forced mixed-stiffness batch in `tests/layers/test_transport_sparse.py`
  (`test_expm_action_mixed_stiffness_batch_matches_standalone_within_tolerance`, removal
  rates 0.01 and 500.0 batched together so the whole batch halves its step whenever either
  instance hasn't converged): 107,030 matvecs (1946 substeps x 55 terms) (re-measured after
  the P1-1 schedule; the forced case pays for the derivative bound -- 104,280 matvecs (1896
  substeps x 55 terms) before P1-1). There is no instrumented pre-fix count from the original
  review.
- The smaller mixed-stiffness case in `tests/layers/test_expm_schedule.py`
  (`test_a_mixed_stiffness_batch_shares_one_schedule_and_matches_the_reference`, a 4-node
  chain, removal-free capacities 1.0 and 1e-3): 4,180 matvecs after the fix (76 substeps x 55
  terms), measured directly on this branch for this re-baseline.

## Decision record: what the re-baseline says (20 Sep 2026)

**What PR-5 measured.** PR-5 re-ran `benchmarks/report_composed_scaling.py` on the current
code (above) and added a controlled A/B against `main` (`8fdea28`, part 1 of the hardening)
and pre-hardening (`f1d8177`), plus a cProfile of one coupled street/building run, to separate
code-attributable change from the machine load both sessions ran under. The A/B, interleaved
three ways per measurement so ambient load cancels out of the ratio rather than the absolute
time, found no code-attributable slowdown: `coupling_street_building.py 1`, wall seconds,
median of 3 interleaved rounds -- head 65.79 s (107 outer passes) vs main 66.03 s (116 passes)
vs pre-hardening 65.60 s (116 passes), ratios head/main 0.996 and main/pre 1.007, both inside
the ~1.2x noise band the machine's ambient load requires. The composed ensemble-1 forward
step (implicit-scheme transport only, no `exact`-scheme closures) gave the same verdict:
0.733 s / 0.711 s / 0.686 s median, ratios 1.031 / 1.036, again inside noise. The 1.5-3x
deltas the 20 September re-baseline table shows against the 17 September numbers are
therefore machine load (another session's ~3 GB resident Python process, active at launch and
throughout both runs), not a regression introduced by this branch. Head does converge the
coupled iteration in fewer outer passes than main and pre-hardening (107 vs 116, repeatable
every round) -- a real, deterministic effect of part 1 of the hardening already on `main`
before this branch -- but it happens to cost slightly more per pass, so it produces no
wall-time gain. **All latency budgets in the re-baseline table remain unmet, on both solvers,
on every row; all four peak-memory budgets and both shape gates pass**, unchanged from 17
September.

**A1 verdict.** The prepared-execution refactor (A1) is **not justified now.** The review's
own gate was orchestration exceeding "about a fifth" of the coupled step; the profile of one
coupled batch-1 run (81.2 s cumulative inside `couple.py:step`) puts the named orchestration
functions -- `_apply_closures`, `_write_at`, `_forward_value`, `apply_conversion`, plus
`_step_model`'s and `model.py`'s `step`/`_advance`/`_pass` self-times, and `couple.py`'s
`_iterate` -- at roughly 0.70 s combined self-time, **under 1% of the run**, two orders of
magnitude below the gate. It stays a recorded option, to revisit only if a future extension
contract genuinely needs the prepared structure (fixed endpoint indices, explicit
flow-provider bindings, an execution order) independent of any wall-time argument -- the
profile gives no wall-time case for it today. (Those per-function self-times come from the
profile's full pstats listing, not from the committed `benchmarks/profile_coupled_2026-09.txt`
-- that file keeps only the cumulative-time top 25 of 8824 profiled functions, below which
these orchestration functions fall; the full numbers are recorded in Task 28's report.)

**Where the time goes, and what that points at.** 93.7% of the profiled run (76.05 s of
81.17 s) is inside two numerical-solve subtrees, sibling calls from `model.py:_pass`: the
implicit-scheme transport step (`layers/transport.py:_step` through `_linear_solve` and
`solvers/select.py:solve` into `solvers/iterative.py:gmres`), 45.1 s cumulative with 13.25 s
of that inside GMRES's own iterative-solve loop; and the building-airflow Newton solve
(`layers/potential.py:solve` through `implicit_solve`/`newton`), 30.9 s cumulative. The
transport solve is routed to GMRES rather than a direct method because `AdvectionOperator`
(`src/noodl/operators/advection.py`) answers both eligibility questions the way that forces
it there: `spd_certificate()` returns `None` unconditionally, and `assemble_sparse()` also
returns `None` (a sparse COO form is derivable but was left unimplemented in Task C's scope,
per that method's own docstring, precisely because it would not change the routing -- see
below). `solvers/select.py`'s `method="auto"` eligibility table (module docstring, confirmed
by reading `solve`) picks the backend in this order: a certificate mixed across a batch
raises; a uniformly-certified-SPD operator with a usable sparse form goes to `sparse_direct`;
a certified-SPD operator without one goes to `pcg`; and a certificate of `None`, or uniformly
`False`, goes to `gmres` -- the last row is `AdvectionOperator`'s row, decided by
`spd_certificate()` alone, so giving it an `assemble_sparse()` would not move it off GMRES
without also revisiting nonsymmetric eligibility in `select.py`. This is where the review's
solver/preconditioner list and the profile agree, and it names three concrete candidates to
benchmark -- not commitments made here:

- a COO/CSR assembly for `AdvectionOperator`, paired with a nonsymmetric sparse-direct
  (or sparse-LU) eligibility path in `select.py`, so small-ensemble implicit transport steps
  can take a direct route instead of GMRES (the doubled-nnz cost of carrying both edge
  orientations, noted in `assemble_sparse`'s docstring, would need to be measured against
  the 45.1 s this profile shows GMRES costing);
- a preconditioner for GMRES on the advection system: `select.solve`'s `preconditioner`
  argument is pcg-only (forwarded only to `pcg`, never to `gmres`), and `iterative.gmres`
  itself takes no preconditioner parameter at all, so both of `select.solve`'s `gmres` call
  sites -- the explicit `method="gmres"` branch and the `auto` fallback -- run GMRES on the
  advection system fully unpreconditioned today. Adding one is the candidate; a Jacobi
  diagonal is the natural first step, but A2 already flags a diagonal as a starting point,
  not the complete strategy for a large ill-conditioned network, so a better one is the real
  target;
- for the potential layer's Newton solve, preconditioner quality under high conductance
  contrast (the review's own item), which this profile's 30.9 s Newton share is consistent
  with but does not by itself isolate from the reference model's ordinary conditioning.

**Exponential-action work counts.** The mean-diagonal-shift fix changed matvec counts as
recorded above: the pure-decay case collapses from 184,459 matvecs to 1 (the shift makes
`M - mu*I` vanish for a case that is structurally an exact-shift-eligible pure decay); the
committed forced mixed-stiffness batch test needs 107,030 matvecs (re-measured after the
P1-1 schedule; the forced case pays for the derivative bound -- 104,280 before P1-1), with
no instrumented pre-fix count to compare against; the smaller (zero-forcing, shift-eligible)
mixed-stiffness case is unaffected by P1-1 and still needs 4,180.
Separately, and by design, `_expm_action` (`src/noodl/layers/transport.py`) now enforces a
work budget rather than relying only on the recursion-depth limit the review flagged: it
raises `RuntimeError` naming the predicted substep x Taylor-term matvec count when that count
exceeds `max_matvecs` (200,000 by default), with the message advising `scheme='implicit'` or
`'trapezoidal'`, or raising `max_matvecs` deliberately -- a stiff exact step is refused rather
than silently left to run to a possibly much larger matvec count.

## Appendix: the source tree

A module-by-module map of the repository, as it stood at the end of milestone 5.


```
src/noodl/
  topology.py    typed multigraph; incidence, gradient, cycle basis, selectors, sign-aware
                 upwind/downwind, spanning forest, interior/boundary indices, caching,
                 to(device, dtype)
  cycles.py      branch_flows, assert_forward_oriented, particular_flow, project_measured
  elements/      Element base class and built-in branch laws: powerlaw.py (PowerLaw, Orifice),
                 quadratic.py (Quadratic), fixed.py (FixedFlow), conductance.py (Conductance),
                 fan.py (FanCurve), duct.py (Duct: Colebrook friction, unrolled), damper.py
                 (Damper), upstream.py (UpstreamDensityPowerLaw, the upstream-density
                 correction on CONTAM power-law elements)
  drives.py      Drive protocol (a function of the DRIVERS alone), ConstantDrive, Stack,
                 Wind, WindProfile (profiles identified by CONTAM's profile number)
  model.py       Model: several physics layers on one network, stepped together;
                 coupling="pingpong" (one pass) or "iterate" (Hensen's onion, 0.5
                 relaxation, per-instance convergence); current_flows(name, state,
                 drivers), a transport layer's flows without stepping the model
  couple.py      orchestration-level coupling between two independently-built Models:
                 ValueLink, DriverAlias, CoupledModel (.step, diagnostics=), union, the
                 unit/angle conversion registry -- never merges networks, layers or
                 closures (milestone 5)
  operators/     the matvec-free linear-operator contract: base.py (LinearOperator protocol,
                 SolveResult, SolverStatus), dense.py (DenseOperator, the retained dense
                 oracle), graph.py (GraphLaplacianOperator), advection.py (AdvectionOperator)
  solvers/       scalar.py (solve_monotone), linear.py (solve, the dense reference),
                 grounding.py (spd_certificate, spd_diagnosis), iterative.py (pcg, gmres;
                 never raise), select.py (solve: the method="auto" eligibility table and the
                 single raise/return failure boundary), newton.py (newton, NewtonResult),
                 implicit.py (implicit_solve, adjoint)
  layers/        potential.py (PotentialFlowLayer), transport.py (TransportLayer),
                 reaction.py (Reaction, FirstOrderDecay)
  apps/building/ the building application: thermal.py (Zone, WallMass, thermal_layer,
                 species_layer, IdealGasDensity, LinearDensity, build_model), elements.py
                 (mass_orifice, add_large_opening), prj.py (the CONTAM .prj reader, a
                 documented subset, and project_to_model), wth.py (the .wth weather reader),
                 sources.py (the four CONTAM source types), contamx.py (the ContamX driver
                 over contamxpy)
  apps/street/   the street application: canyon.py (BoundaryLayer, canyon_velocity,
                 exchange_velocity, the soulhac/macdonald closures), routing.py
                 (StreetGeometry, StreetFlows, routing_matrix, node_closure,
                 direction_offsets -- the north-west-corner routing), network.py
                 (StreetNetwork, build_street_model, street_geometry, street_index,
                 munich_idealised), chemistry.py (photostationary_for_streets,
                 street_steady), loader.py (read_aqdt, the AQ_DT GeoJSON/NetCDF reader),
                 impaq.py (the IMPAQ prototype ported as a numpy/scipy comparison oracle,
                 kept as per-edge Python loops by design -- it is the oracle, not the
                 model path), report.py (to_ug_m3, from_ug_m3,
                 write_network_concentration)
  apps/sewer/    the gravity-sewer application: geometry.py (exact circular geometry, the
                 batched Manning normal-depth inversion), hydraulics.py (SewerHydraulics --
                 closure-first tree flow, depth, the optional implicit-Euler storage
                 sweep), air.py (Headspace, Drag, air_density), quality.py (Henry's law,
                 the two-film flux, Pomeroy-Parkhurst sulfide generation, BOD decay,
                 LateralLoads), network.py (SewerNetwork, build_sewer_model, sewer_steady),
                 inp.py (a documented SWMM .inp subset) and report.py
  apps/water/    the pressurised water-distribution application: network.py (Junction/
                 Reservoir/Tank/WaterPipe/Pump/Valve, WaterNetwork, build_water_model,
                 water_steady), elements.py (HazenWilliams, PumpCurve, MinorLoss),
                 tanks.py (TankLevels), demand.py (PressureDrivenDemand), inp.py (a
                 documented EPANET 2.2 .inp subset) and report.py
  apps/inpfile.py  the section-keyed `.inp` tokenizer shared by apps/sewer/inp.py and
                 apps/water/inp.py, and nothing else
  physics/       flows.py, species.py: thin wrappers so downstream code runs unchanged
docs/superpowers/  design spec and implementation plans
tests/
  conftest.py, test_topology.py, test_endpoints.py, test_cycles.py, test_cycles_sparse.py,
  test_drives.py, test_flows.py, test_import.py, test_model.py, test_species.py,
  test_species_compat.py, test_couple.py (the union mechanism: conversions, one- and
  two-way links, aliases, substeps, gradients across the join)
  elements/      test_base.py, test_powerlaw.py, test_quadratic.py, test_fixed.py,
                 test_conductance.py, test_fan.py
  operators/     test_base.py, test_graph.py, test_advection.py, test_assemble_sparse.py
  solvers/       test_scalar.py, test_linear.py, test_grounding.py, test_iterative.py,
                 test_select.py, test_newton.py, test_newton_operator_contract.py,
                 test_implicit.py, test_implicit_operator_contract.py
  layers/        test_potential.py, test_potential_sparse.py, test_transport.py,
                 test_transport_sparse.py, test_reaction.py
  apps/building/ test_thermal.py, test_elements.py, test_prj.py, test_wth.py,
                 test_sources.py; tests/data/contam holds the sample projects
  apps/street/   test_canyon.py, test_routing.py, test_network.py, test_chemistry.py,
                 test_loader.py, test_impaq_port.py, test_conservation.py, test_report.py;
                 tests/data/street holds the AQ_DT and MUNICH fixtures
  verification/  CONTAM-style closed-form airflow cases, batched against scipy roots;
                 test_composed_model.py (parity, interface conservation, cross-join
                 gradients); test_natural_ventilation.py (Li and Delsante closed forms, a
                 two-zone scipy oracle, Hensen's ping-pong/onion table, the golden);
                 test_contam_parity.py (ContamX through contamxpy, skipped when absent);
                 CPU performance budgets and test_composed_scaling.py, the milestone-1b
                 and milestone-2 acceptance gates (both marked slow, skipped by default);
                 test_munich.py (13 MUNICH formula pairs and the idealised 12-street case);
                 test_street_parity.py (the IMPAQ port, four-node and leiden_small parity);
                 test_coupling_demo.py (the headline street+building union: one-way vs
                 two-way back-coupling on a synthetic 2x3 m canyon and the real
                 leiden_small/CONTAM pairing, plus the loose sequential file-exchange
                 comparison); test_coupling_inverse.py (the three inverse examples:
                 calibration through the join, source attribution by one adjoint pass,
                 latent infiltration via the cycle space; the calibration test is `slow`)
  golden/        stored reference results (contam_airflow.json, natural_ventilation.json)
                 and load_golden/save_golden
benchmarks/
  newton_scaling.py           batched Newton solve timing vs network size and batch size
  composed_model.py           the composed reference model: 8 buildings joined through a
                              street and a sewer network, optionally (thermal=True) with a
                              heat layer beside the species one, stepped through Model
  natural_ventilation.py      the coupled airflow-heat demo: two rooms, a doorway, a daily
                              ambient sinusoid, under both coupling modes
  measure.py                  isolated peak-RSS (one child process per figure) and
                              saved-tensor-bytes measurement
  profile_forward.py          per-stage forward/backward profile of the composed model, and
                              --compare-solvers, the in-process backend comparison the
                              method="auto" default and its batch threshold were decided on
  sparse_scaling.py           gather/scatter vs shared-CSR matvec timing across thread counts
  sparse_review_checks.py     standalone numerical checks used during the sparse-path review
  report_composed_scaling.py  writes benchmarks/composed_scaling_report.json
  street_leiden.py            load/build/solve timing for the street model on the real
                              AQ_DT `leiden_small` and `leiden` domains
  coupling_street_building.py batched throughput of `city.step` across batch sizes, on the
                              headline street+building coupled demo (milestone 5)
  regenerate_golden.py        rewrites tests/golden/contam_airflow.json and
                              tests/golden/natural_ventilation.json (explicit action)
```
