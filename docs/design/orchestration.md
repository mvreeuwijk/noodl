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

The least-squares columns (newest first) are kept while two bounds hold, and the first column
that fails either is dropped together with every older one:

- relative: its orthogonalised norm is at least `1 / _ANDERSON_COND_MAX` (1e-8) of the largest
  one so far, so an ill-conditioned history loses its oldest columns;
- absolute: its orthogonalised norm is at least `_ANDERSON_NOISE_MIN` (10) times its rounding
  noise, the norm of `eps (|g| + |x|)` of the two passes it differences, scaled by the
  tolerances exactly as the residuals are (`_AndersonHistory` keeps it per pass).

The relative bound alone does not see the residuals' rounding floor. Near convergence the
leading column shrinks with the residual, and a column that holds nothing but rounding passes
it; on the golden toy (a tree, whose flows the sources fix, so their residuals are rounding
only) such a column took a coefficient of order one and moved the heads by its large output
difference. The pass count then followed the last ulp: 101 source perturbations of up to 50
ulp at a head tolerance of 1e-10 took 10 to 15 passes, and three of them stepped into a head
the conductance closure has no solution for and raised. The reversal safeguard of
`_AndersonHistory` does not catch it: the step that throws the iteration off (from a
residual of 0.1 to one of 1e1 to 1e3, in units of the tolerance) is taken when every residual
is already below its tolerance, where the safeguard does not look, and the head solve fails two
passes later. With the noise bound the runs all converge, in 10 to 13 passes, and near
convergence the mixing degrades to fewer columns and in the end to the relaxed step (`tests/test_model.py::test_anderson_near_convergence_does_not_follow_rounding`).

The residuals' actual floor sits somewhat above one ulp (the head solve's own tolerance, and
earlier mixed steps that carry it into the flows), so a larger multiple narrows the toy's
spread further: at 100 those runs take 10 or 12 passes. But it also starts to cut real secant
information where the tolerance is close to rounding, as the sewer fixtures' HEAD_TOLERANCE of
1e-12 m is. Over the dynamic-wave fixtures (`tests/data/sewer/*_dynwave.inp` with their own
options, 60 steps each, passes counted from the diagnostics) the average was 14.40 passes per
step without the bound, 14.37 at 10, 14.53 at 100 (up to 10 % more on the `shapes` fixtures)
and, at 1000, up to 30 % more on single fixtures. Hence 10: far from convergence every column
is many orders above it, and nothing changes.
