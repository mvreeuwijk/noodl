# Core orchestration: measurements and rationale

This page holds the measurements behind small implementation choices in the core
orchestration (`noodl._broadcast`, `noodl.layers.kirchhoff`, `noodl.solvers.anderson`). The
code keeps the rule and the reason; the numbers live here, with how to reproduce them. All
figures are CPU measurements (torch CPU, Windows 11, Python 3.12, float64) on the reference
composed model of `benchmarks/composed_model.py` unless stated; they move with hardware and
only their ordering is a claim.

## Shape broadcasting

`noodl._broadcast.broadcast_shapes` replaces `torch.broadcast_shapes` for plain integer
shapes. The torch function goes through `torch._refs` (symbolic-shape checks) and costs about
150-190 us a call on CPU, more than most of the small tensor operations it sizes. The airflow
and transport steps call it about a hundred times per step; in the Modelica parity runs this
was about 12 % of a step (profiled).

Reproduce: profile a building-physics Modelica parity run (`tests/verification/
test_modelica_parity.py`, `-m slow`) with `cProfile` and look at the cumulative time of
`torch.broadcast_shapes`.

## Dense incidence matrix

`KirchhoffLayer.A`, the dense `(n, b_layer)` incidence matrix, is built on first access only
(`functools.cached_property`). At the composed model's reference size it is 18.1 MB per layer
and grows 4x per node doubling. Held eagerly, it broke the memory-scaling budget of the
composed model (measured growth 3.16x against a 2.5x budget).

Reproduce: `python -m benchmarks.report_composed_scaling` (memory scaling of the composed
model).

## Residual incidence product

`KirchhoffLayer.residual` forms the interior row sums `A_I q` by scatter-add over the layer's
edges, not by an einsum against the dense `(n_I, b)` slice of the incidence. At ensemble 100
the einsum form alone cost 14.2 ms of a 41.8 ms residual evaluation (profiled on the composed
model).

Reproduce: `python -m benchmarks.profile_forward` and compare the residual's share of the
profile.

## Backend of the classic solve

For the sparse-direct versus PCG ratios (about 4.6x slower PCG on the single-instance
forward solve) and the ensemble crossover at `select._SPARSE_DIRECT_MAX_BATCH`, see
[Solver selection](solvers.md).

## Anderson mixing

`noodl.solvers.anderson` writes Gram-Schmidt and the back substitution out in elementwise
torch over the (at most `M`, in practice a handful of) history columns. A LAPACK QR or
triangular solve there costs far more than its arithmetic: after one, the OpenMP worker
threads it wakes slowed every later small elementwise op of the pass about threefold
(measured on the sewer dynamic wave).

Reproduce: `python -m benchmarks.sewer_dynwave_speed`, swapping the hand-written
factorisation for `torch.linalg.qr`.
