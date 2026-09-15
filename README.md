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
                 fan.py (FanCurve)
  drives.py      Drive protocol, ConstantDrive
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
  physics/       flows.py, species.py: thin wrappers so downstream code runs unchanged
legacy/          the original 2019 package, for reference
docs/superpowers/  design spec and implementation plans
tests/
  conftest.py, test_topology.py, test_endpoints.py, test_cycles.py, test_cycles_sparse.py,
  test_drives.py, test_flows.py, test_import.py, test_species.py, test_species_compat.py
  elements/      test_base.py, test_powerlaw.py, test_quadratic.py, test_fixed.py,
                 test_conductance.py, test_fan.py
  operators/     test_base.py, test_graph.py, test_advection.py, test_assemble_sparse.py
  solvers/       test_scalar.py, test_linear.py, test_grounding.py, test_iterative.py,
                 test_select.py, test_newton.py, test_newton_operator_contract.py,
                 test_implicit.py, test_implicit_operator_contract.py
  layers/        test_potential.py, test_potential_sparse.py, test_transport.py,
                 test_transport_sparse.py, test_reaction.py
  verification/  CONTAM-style closed-form airflow cases, batched against scipy roots;
                 test_composed_model.py (parity, interface conservation, cross-join
                 gradients); CPU performance budgets and test_composed_scaling.py, the
                 milestone-1b acceptance gate (both marked slow, skipped by default)
  golden/        stored reference results (contam_airflow.json) and load_golden/save_golden
benchmarks/
  newton_scaling.py           batched Newton solve timing vs network size and batch size
  composed_model.py           the composed reference model: 8 buildings joined through a
                              street and a sewer network
  measure.py                  isolated peak-RSS (one child process per figure) and
                              saved-tensor-bytes measurement
  profile_forward.py          per-stage forward/backward profile of the composed model, and
                              --compare-solvers, the in-process backend comparison the
                              method="auto" default and its batch threshold were decided on
  sparse_scaling.py           gather/scatter vs shared-CSR matvec timing across thread counts
  sparse_review_checks.py     standalone numerical checks used during the sparse-path review
  report_composed_scaling.py  writes benchmarks/composed_scaling_report.json
  regenerate_golden.py        rewrites tests/golden/contam_airflow.json (explicit action)
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
