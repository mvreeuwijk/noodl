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
  operators/     test_base.py, test_graph.py, test_advection.py
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
shape gates -- peak memory vs nodes at 1.05x and matvec time vs edges at 0.72x (edge
sensitivity 1.03x at 16x/32x the reference edge count), against a 2.5x budget each. **Every
backward-pass budget, under both solvers** -- new since the sparse-direct evaluation below;
previously two of the four backward rows failed. Jacobi-PCG takes 168-180 iterations at the
reference size, well under section 6.2's 500-iteration preconditioning trigger, so that
trigger did not fire -- evaluated on the reference configuration only, not across section 6's
robustness cases.

**What does not.** The forward-pass wall-clock budgets, on every row, under both solvers, per
[`benchmarks/composed_scaling_report.json`](benchmarks/composed_scaling_report.json)
(`all_budgets_met: false`, keyed to the shipped default, `auto`):

| Configuration | Solver | Forward | Backward | Peak memory |
|---|---|---|---|---|
| ensemble 1, 1 step | `auto` (sparse_direct) | 0.097 s vs 0.05 s (1.94x) FAIL | 0.089 s vs 0.1 s (0.89x) PASS | 68.3 MB vs 100 MB (0.68x) PASS |
| | `cg` (PCG) | 0.210 s vs 0.05 s (4.20x) FAIL | 0.100 s vs 0.1 s (~1.00x) PASS | 46.6 MB vs 100 MB (0.47x) PASS |
| ensemble 100, 1 step | `auto` (sparse_direct) | 1.686 s vs 0.5 s (3.37x) FAIL | 0.566 s vs 1.0 s (0.57x) PASS | 153.5 MB vs 1000 MB (0.15x) PASS |
| | `cg` (PCG) | 1.593 s vs 0.5 s (3.19x) FAIL | 0.554 s vs 1.0 s (0.55x) PASS | 128.8 MB vs 1000 MB (0.13x) PASS |
| ensemble 100, 24 steps | `auto` (sparse_direct) | 48.5 s vs 12 s (4.04x) FAIL | 15.0 s vs 25 s (0.60x) PASS | 456.5 MB vs 2000 MB (0.23x) PASS |
| | `cg` (PCG) | 39.4 s vs 12 s (3.29x) FAIL | 13.9 s vs 25 s (0.56x) PASS | 426.8 MB vs 2000 MB (0.21x) PASS |
| ensemble 1000, 1 step | `auto` (sparse_direct) | 15.2 s vs 5 s (3.03x) FAIL | 4.43 s vs 10 s (0.44x) PASS | 783.6 MB vs 8000 MB (0.10x) PASS |
| | `cg` (PCG) | 12.9 s vs 5 s (2.58x) FAIL | 7.74 s vs 10 s (0.77x) PASS | 753.9 MB vs 8000 MB (0.09x) PASS |

Before -> after, at the reference row the sparse-direct evaluation was measured on (ensemble
1, 1 step; old committed figures were all under what `auto` then meant, plain PCG): forward
0.483 s -> 0.097 s under `auto`, 0.210 s under `cg`; backward 0.290 s -> 0.089 s / 0.100 s.
Every row's forward time dropped by roughly 2-3x and every row's backward budget now passes
under both solvers; forward still misses its budget everywhere, by a narrower margin than
before (was ~6-9.7x, now 2.6-4.2x).

`method="auto"` now selects sparse-direct SciPy SuperLU for a per-instance operator that
certifies SPD, declares a sparse form, and whose solve is grad-safe, with SciPy importable;
PCG when certified but not sparse-direct-eligible; GMRES for the non-symmetric transport
block (`TransportLayer`), unchanged. This rule was decided on, and measured at, the reference
size only -- 1028 unknowns -- per this gate; SuperLU's fill-in at much larger `n` is unprobed,
and revisiting the rule there is the documented condition in section 6.2.

Forward time stays dominated by the per-instance loop, not the backend: `solvers.select.solve`
calls SciPy's sparse-direct solver once per ensemble instance inside a Python loop, so the
loop's own dispatch cost scales with ensemble size under either solver. A vendor sparse-direct
route -- one batched call rather than a Python loop, and separately the CUDA/XPU `_spsolve`
route -- stays admissible but unbuilt, per spec section 2.

Backward time is recorded as passing everywhere in the committed report, but the reference-size
row is close enough to budget that it is not a settled result: re-running the same gate
(`test_composed_scaling.py -m slow`, not the report script, three subprocess samples apart on
the same machine) measured `auto`'s ensemble-1 backward at 0.141 s (1.41x, FAIL) against the
committed report's 0.089 s (0.89x, PASS) for the identical configuration and code. Run-to-run
spread on this machine remains wide enough to flip that row's verdict, and the committed
figures are still medians of `samples: 3`. Section 6.2's new rule ("any budget missed by 2x or
less: re-measure with `samples >= 5`") applies to it; it has not been re-measured at that
sample count, so it is left here as an open item with its measured spread (0.089-0.141 s
against a 0.100 s budget) rather than a settled pass.

The committed report sweeps ensemble size and simulation length, now under two solvers;
section 6's species count and joined-submodel sweeps, and its composed robustness cases, are
not measured. Nodes and edges are covered by the two shape gates.

## New public API in milestone 1b

Every one of these is optional and defaults to the pre-milestone behaviour.

| Entry point | Keyword | Meaning |
|---|---|---|
| `PotentialFlowLayer(...)` | `linear_solver=` | inner linear solver for this layer, used on both the forward and the backward pass: `"auto"` (default; sparse-direct SciPy SuperLU when the per-instance SPD certificate holds, the operator has a sparse form, SciPy is importable and the solve is grad-safe; PCG when certified but not sparse-direct-eligible; GMRES otherwise), `"cg"`, `"gmres"`, `"sparse_direct"` (SciPy SuperLU on the assembled sparse form; requires the SPD certificate), or `"direct"` (LU of the assembled dense operator -- the retained milestone-1 numerics) |
| `PotentialFlowLayer.solve(...)` | `diagnostics=` | a dict, filled with `newton_iterations`, `linear_iterations`, `method`, `converged` and `residual_norm` |
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
