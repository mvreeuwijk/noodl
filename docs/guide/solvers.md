# Choosing solvers and coupling schemes

Most models run well on the defaults. This page says when to change them: the linear solver of a
layer, what to do when Newton does not converge, and how to couple layers or models. How the
solvers work is explained in [Solvers](../concepts/solvers.md); the measurements behind the defaults
are in the design notes on [solver selection](../project/design/solvers.md).

## Choose a linear solver

Every potential layer solves a nonlinear system by Newton's method, and each Newton step solves a
linear system. `KirchhoffLayer(linear_solver=...)` takes `"auto"` (the default), `"cg"`, `"gmres"`,
`"direct"` or `"sparse_direct"`; `TransportLayer(linear_solver=...)` takes `"auto"`, `"gmres"`,
`"gmres_jacobi"`, `"gmres_ilu"`, `"sparse_direct"` or `"direct"`. An explicit choice is honoured at
every batch size.

Leave `"auto"` unless you have a measured reason. It picks by what the operator can prove:

| Situation | `auto` uses |
|---|---|
| Symmetric positive definite system (a grounded network of positive conductances), up to 32 batch instances, SciPy installed | `sparse_direct` (SuperLU, one factorisation per instance) |
| The same system, a larger batch or no SciPy | `pcg` (Jacobi-preconditioned conjugate gradients) |
| A system that cannot be proved symmetric positive definite | `gmres` |

Consequences:

- Install the `sparse` extra if you solve small batches: without SciPy the same model runs on PCG
  at several times the cost. `diagnostics` reports `method` (what you asked for) and `backend`
  (what ran) separately, so you can see which one you got.
- Do not ask for `"cg"` on a system that is not symmetric positive definite: it is refused, not
  run. A layer with flow unknowns, a phasor layer or a held initial point refuses it.
- A batch in which some instances certify symmetric positive definite and others do not raises: a
  batch is never split between backends.
- `gmres_ilu` is the slowest transport choice at every ensemble size measured; `auto` never picks
  it.

## When Newton does not converge

Try these in order, stopping at the first that works.

1. **Retry from the linear guess.** `model.step(..., warm_start=False)` starts each potential
   solve from the layer's linear initial guess instead of the last potentials. Catch
   `noodl.solvers.ConvergenceError` to retry on that error alone.
2. **Give the layer a fallback ladder.** `KirchhoffLayer(..., fallback=("linear_guess",
   "storage_ramp"))` starts again, only when Newton did not converge, from the linear guess and then
   from a pseudo-time ramp of fictitious storage that is removed in steps. A solve that a rung
   rescued is an ordinary converged one, and the gradient is the implicit one at the exact root. The
   default is no ladder. `diagnostics["fallback"]` names the rung that converged.
3. **Kinked laws.** A law with kinks (a flat-steep-flat curve, a stiff bed meeting a soft storage
   line) can send the damped Newton into a cycle. `line_search="convex"` cuts a step that overshoots
   back to the minimum along it; it needs every branch flow non-decreasing in its own potential
   difference and is refused on a layer with flow unknowns. `step_floor=k` ends an instance as
   converged once its step is within `k` units in the last place of the iterate, for an `atol`
   below the round-off of a stiff law. Both are off by default; set them for a layer through
   `newton_defaults`.
4. **Check the tolerances against the dtype.** Newton's default `atol` and `rtol` are the square
   root of the machine epsilon of the working dtype (about 1.2e-4 for float32, 1.5e-8 for float64).
   A tighter tolerance than the dtype can reach is never met. See
   [Troubleshooting](troubleshooting.md#float32-or-float64).
5. **Look for a lost ground.** A potential problem needs one grounded (boundary) node per connected
   component. A fan at shutoff, a closed damper or an isolated component makes the operator
   singular; the error names the component. See [Troubleshooting](troubleshooting.md#a-singular-or-ungrounded-system).

## Choose a coupling

Within one model, layers that feed back on each other (airflow carries heat, and temperature drives
the airflow) are stepped under one of two schemes:

| Scheme | Passes per step | Use when |
|---|---|---|
| `coupling="pingpong"` (default) | one, from the step-start state | the coupling is weak, or you want the cheapest step; splitting error is first order in `dt` |
| `coupling="iterate"` with `iterate_tol` | repeated until the named states stop changing, at most `iterate_max` | the flow depends on what it carries, you call `steady()`, or you need the converged answer |

`iterate_tol` is a per-layer mapping, absolute, in each layer's own units
(`{"thermal": 0.01}` in kelvin, `{"species": 1e-6}` in kg/kg). Name the layers whose change decides
convergence, and name a potential layer (`{"water": 1e-6}`) when a closure computes its
coefficients from its own previous solution, as the sewer's dynamic wave does. Add
`iterate_flow_tol` when the flows keep moving while the potentials do not. If plain relaxation is
slow, `iterate_acceleration="anderson"` mixes the last passes (it needs an `iterate_flow_tol` for
each potential layer it mixes); it changes the sequence of states, not the fixed point. The
[layers page](../concepts/layers.md) explains the schemes; [Time stepping](time-stepping.md) shows
the calls.

Across two models, use `union` and a `ValueLink`: see [Coupling two models](coupling.md). Its
two-way iteration has a fixed relaxation (`relaxation=0.5`); lower it if a coupled step oscillates.

## Choose a time scheme

A potential layer takes `scheme="trapezoidal"` (default) or `"backward_euler"`; a transport layer
takes `"exact"`, `"backward_euler"` or `"trapezoidal"`. `"forward_euler"` is only for a potential
layer whose storage sits at boundary nodes (an EPANET tank). Trapezoidal is second order but rings
after a discontinuity: `model.march(..., restarts={0})` takes the first step by backward Euler to
damp it.
