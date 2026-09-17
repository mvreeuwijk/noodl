# tellegen

Network topology and conservative physics on graphs, in PyTorch. Named after
Tellegen's theorem: for any potentials in the cut space and any flows in the
cycle space of a graph, the branch power sum is zero.

Origin: John Craske's 2019 `Tellegen` package (autograd, Brayton and Moser's
formulation of nonlinear networks), kept unchanged in `legacy/`. This
repository ports the topology layer to PyTorch and rebuilds the physics as
nodal state-space modules with storage at nodes, typed edges, and batching
over buildings.

## Layout

```
src/tellegen/
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
                 relaxation, per-instance convergence)
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
  physics/       flows.py, species.py: thin wrappers so downstream code runs unchanged
legacy/          the original 2019 package, for reference
docs/superpowers/  design spec and implementation plans
tests/
  conftest.py, test_topology.py, test_endpoints.py, test_cycles.py, test_cycles_sparse.py,
  test_drives.py, test_flows.py, test_import.py, test_model.py, test_species.py,
  test_species_compat.py
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
  verification/  CONTAM-style closed-form airflow cases, batched against scipy roots;
                 test_composed_model.py (parity, interface conservation, cross-join
                 gradients); test_natural_ventilation.py (Li and Delsante closed forms, a
                 two-zone scipy oracle, Hensen's ping-pong/onion table, the golden);
                 test_contam_parity.py (ContamX through contamxpy, skipped when absent);
                 CPU performance budgets and test_composed_scaling.py, the milestone-1b
                 and milestone-2 acceptance gates (both marked slow, skipped by default)
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
  regenerate_golden.py        rewrites tests/golden/contam_airflow.json and
                              tests/golden/natural_ventilation.json (explicit action)
```

Licence: MIT (the original code with the agreement of its author).

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

## New public API in milestone 1b

Every one of these is optional and defaults to the pre-milestone behaviour.

| Entry point | Keyword | Meaning |
|---|---|---|
| `PotentialFlowLayer(...)` | `linear_solver=` | inner linear solver for this layer, used on both the forward and the backward pass: `"auto"` (default; sparse-direct SciPy SuperLU when the per-instance SPD certificate holds, the operator has a sparse form, SciPy is importable, the solve is grad-safe **and the flat batch is at most 32 instances**; PCG in every other certified case, including larger ensembles; GMRES when the operator cannot certify), `"cg"`, `"gmres"`, `"sparse_direct"` (SciPy SuperLU on the assembled sparse form), or `"direct"` (LU of the assembled dense operator -- the retained milestone-1 numerics). The sparse-direct branch of `"auto"`, and `"sparse_direct"` itself, need SciPy: `pip install tellegen[sparse]`. Without it `"auto"` falls back to PCG (correct, and 4.6x slower at ensemble 1) and warns once per process; an explicit `"sparse_direct"` raises `ImportError` naming scipy |
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
  layer at all, exported as `tellegen.layers.transport.active_interior`. A caller sizing a
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
| ContamX 24-step transient concentrations | 2e-3 | 3.4e-6 |
| ContamX single-zone stack flows at 273.15/283.15/303.15/313.15 K ambient | 1e-3 | 4.4e-5 |

The ContamX figures are against ContamX 3.4.1.7 through `contamxpy` 0.0.9, Windows x86-64
only. The stack row is 4.4e-5 **after** the upstream-density correction; before it, the same
case was off by 1.7e-2.

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
6.608 s. HEAD and base are within noise of each other, so **the branch has not regressed**:
the machine is 2-4x slower today than when the 1b table was recorded, and noisy within the
morning -- (100, 1, `auto`) measured 2.900 s inside the report run and 6.7 s an hour later.
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

## Installation

`pip install -e .[dev]`, or on Windows x86-64 `pip install -e .[dev,contam]`, which adds
NIST's `contamxpy` (it bundles the ContamX 3.4.1.7 engine) and so enables the ContamX parity
tests in `tests/verification/test_contam_parity.py`; `-m external` is not needed, they run
when the package imports and skip themselves otherwise, including on every non-Windows CI.

## Quick start

```python
import torch

from tellegen.elements.powerlaw import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network

net = Network(dtype=torch.float64)
net.add_node("ambient")
net.add_node("room")
net.add_edge("room", "ambient", kind="airpath")

leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65))
layer = PotentialFlowLayer(net, "air", [leak], boundary=["ambient"])

phi_boundary = torch.zeros(1, dtype=torch.float64)
# sources is indexed in full node order (ambient, room); 0.05 m3/s injected into "room"
sources = torch.tensor([0.0, 0.05], dtype=torch.float64)
phi, q = layer.solve(phi_boundary, sources=sources, differentiable=False)

print("pressures (Pa):", dict(zip(net.nodes, phi.tolist())))
print("flows (m3/s):  ", dict(zip(("room->ambient",), q.tolist())))
```
