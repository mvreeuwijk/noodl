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
per-instance solver honesty. All four peak-memory budgets. Both shape gates -- peak memory
vs nodes at 1.06x and matvec time vs edges at 1.12x, against a 2.5x budget each. Jacobi-PCG
takes 168-180 iterations at the reference size, well under section 6.2's 500-iteration
preconditioning trigger, so that trigger did not fire -- evaluated on the reference
configuration only, not across section 6's robustness cases.

**What does not.** The wall-clock budgets, per
[`benchmarks/composed_scaling_report.json`](benchmarks/composed_scaling_report.json)
(`all_budgets_met: false`):

| Configuration | Forward | Backward | Peak memory |
|---|---|---|---|
| ensemble 1, 1 step | 0.48 s vs 50 ms (9.7x) | 0.29 s vs 100 ms (2.9x) | ok |
| ensemble 100, 1 step | 3.36 s vs 500 ms (6.7x) | 1.05 s vs 1.0 s (1.05x) | ok |
| ensemble 100, 24 steps | 70.4 s vs 12 s (5.9x) | ok | ok |
| ensemble 1000, 1 step | 31.0 s vs 5 s (6.2x) | ok | ok |

Forward time misses on all four rows by roughly 6x, which fires section 6.2's "missed by
more than 2x" trigger: *evaluate a sparse-direct path*. Profiling puts the miss squarely in
the solver (PCG is 94% of forward time; the SPD certificate is 1.5%), but about 36% of PCG
time is Python/dispatch overhead inside the iteration rather than matvec work, so the
follow-up should measure the cheap wins -- caching the operator's expanded index tensors
per batch shape, narrowing rather than masking in PCG's freezing -- before concluding that
a sparse-direct path is the only route.

Backward time misses on the two smallest rows. These are recorded as open failures rather
than resolved: run-to-run spread on this machine is 3-5x and the committed figures are
medians of `samples: 3`, so the same two rows measured 1.71x and 1.10x in the final
review's own run, straddling section 6.2's 2x trigger. They should be re-measured as part
of the sparse-direct evaluation above. Section 6.2 defines no follow-up for a miss of 2x or
less, which is a gap in the spec, not a finding about the implementation.

The committed report sweeps ensemble size and simulation length only; section 6's species
count and joined-submodel sweeps, and its composed robustness cases, are not measured.
Nodes and edges are covered by the two shape gates.

## New public API in milestone 1b

Every one of these is optional and defaults to the pre-milestone behaviour.

| Entry point | Keyword | Meaning |
|---|---|---|
| `PotentialFlowLayer(...)` | `linear_solver=` | inner linear solver for this layer, used on both the forward and the backward pass: `"auto"` (default; PCG when the per-instance SPD certificate holds, GMRES otherwise), `"cg"`, `"gmres"`, or `"direct"` (LU of the assembled operator -- the retained milestone-1 numerics) |
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
