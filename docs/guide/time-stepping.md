# Time stepping and schedules

How to advance a model in time, hold inputs that change, and decide how layers that depend on each
other are stepped. The examples assume `model`, `state` and `drivers` as built in
[the workflow](workflow.md).

## Step, or solve the steady state

`model.step(state, drivers, dt)` returns the new state; call it in a loop, changing the drivers
between steps as the inputs change. `model.steady(state, drivers)` returns the quasi-steady state
instead. An allocation layer is a time-stepping rule and has no steady state: a model that owns
one refuses a steady pass.

Time-dependent laws and integrating closures read the clock: pass `t=`, the time at the start of
the step, to `step`. Without `t` a time-dependent law refuses to evaluate.

## Drivers that change in time

A driver is a tensor, and you hold it for as long as you pass it. To change an input in time,
build the drivers for each step (or replace a key in the dictionary) before calling `step`:

```py
for k in range(n_steps):
    drivers["thermal.x_boundary"] = outdoor_temperature[k]     # a schedule, read per step
    state = model.step(state, drivers, dt, t=k * dt)
```

Applications that carry their own schedules take them as drivers: the sewer's inflows, the
meteorology of the street model, the forcing series of a WSIMOD model (a `(T, n)` series, from
which a coupled run takes the day's row). Each application's page says which.

## Marching over a time grid

`model.march(state, drivers, times)` is the loop over `model.step` for a time grid: step `k` goes
from `times[k]` to `times[k+1]`. The result is a dictionary holding `"time"` and every layer state
the start state carries, every state key a closure carries (a pump's status, a manhole level), and
the storage flow of every potential layer with storage, stacked on a new first axis with the start
state as row 0. It is differentiable through time as `step` is.

```py
history = model.march(
    state, drivers, times,
    restarts={0},                                # steps taken by backward Euler
    drivers_at=lambda t0, t1: drivers_of(t1),    # drivers of the step t0 -> t1
    record=[0, 10, 20],                          # keep only these indices into times
)
```

- `drivers_at` gives the drivers of each interval when they change in time.
- `restarts` names the steps taken by backward Euler. The trapezoidal rule rings after a first
  step or a source discontinuity; a backward Euler step damps it. Layers with the `"exact"`
  scheme are untouched, and a potential layer whose storage sits at boundary nodes keeps its
  forward Euler. On the other steps each layer takes its own scheme.
- `record` keeps only the listed indices, so a long run does not hold every step.
- `diagnostics["schemes"]` reports the scheme of each step.

The electrical `transient` and the Modelica electrical reader's `simulate_circuit` are this loop.
The building application's `simulate` is its own march (step-mean inputs, controls held over
passes, breakpoint sub-steps).

## Layers that depend on each other

When the layers feed back on each other (airflow carries heat, and temperature drives the
airflow), build the model with `coupling="iterate"` and an `iterate_tol`, which repeats each step
until the named states stop changing. The default, `coupling="pingpong"`, takes one pass per step
and is first-order accurate in `dt`. Calling `steady()` on a two-way coupled model at the default
returns a plausible number that has not converged: see
[Troubleshooting](troubleshooting.md#the-default-coupling-returns-a-plausible-wrong-answer) and
[Layers and models](../concepts/layers.md). Which to choose is in
[Choosing solvers and coupling schemes](solvers.md#choose-a-coupling).

## Repeating a pass until it settles

`model.fixed_point(pass_fn, state, watch=[...], tol=...)` repeats any function of a state (a
`steady`, or a `step` plus a reaction, or a boundary read off the previous pass) until the watched
keys stop changing, and returns the last pass's state. It is what the street and sewer steady
solves run on. Its gradient is that of the fixed point (the implicit adjoint), not of the unrolled
passes, including through an unwatched key that feeds back from one pass to the next.
`relaxation`, a per-instance `measure` and `diagnostics` are described in its docstring.

## Warm starts and retries

Each potential solve starts from the state's last potentials, and a layer with flow unknowns from
its last flows too. `warm_start=False` starts from the layer's linear initial guess instead, while
closures still read those potentials; use it as a retry when a warm start does not converge.
Non-convergence raises `noodl.solvers.ConvergenceError` under either coupling, so a retry can
catch it alone and let every other error through.
