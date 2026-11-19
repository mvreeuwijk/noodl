# Troubleshooting

What to do when a run raises, does not converge or returns a number you do not believe. Each
section starts with the symptom.

## The default coupling returns a plausible wrong answer

**Symptom.** A model whose flow depends on what it carries (airflow and temperature, a ventilated
room and a species) gives a steady state that is not the converged one, with no warning.

**Cause.** `coupling="pingpong"`, the default, takes one pass per step with the state at the start
of the step. `model.steady()` at the default solves the airflow at the temperatures the closures
saw, the initial ones, and never re-converges it against the temperatures it produced.

**Fix.** Build the model with `coupling="iterate"` and an `iterate_tol`, as the
[workflow example](workflow.md#a-complete-example) does:

```py
model = build_model(net, ..., coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50)
```

Use ping-pong only when the coupling is weak. See [Layers and models](../concepts/layers.md) and
[Choosing solvers and coupling schemes](solvers.md#choose-a-coupling).

## `coupling='iterate' did not converge`

**Symptom.** `RuntimeError: Model: coupling='iterate' did not converge within N passes ... for
instances [...]; largest change per key {...}, tolerances {...}`. When the last pass's potential
solve also failed, the error is `noodl.solvers.ConvergenceError`, a `RuntimeError` subclass.

**Check, in order.**

1. The message lists the failing batch instances and the largest change per key. One instance may
   be badly posed (an extreme input) while the rest converge.
2. The tolerance may be below what the solve can deliver. `iterate_tol` is absolute, in each
   layer's units, and has a floor set by the potential solve's own residual: a tighter tolerance
   needs matching `atol` and `rtol` on the `step` call.
3. Raise `iterate_max`, or let a batch return its last pass with `iterate_on_limit="accept"`
   (`diagnostics["converged"]` is then False for those instances; a gradient is refused).
4. If the message says the potential solve did not converge, treat it as a Newton failure
   (below).

## Newton does not converge

**Symptom.** `ConvergenceError`, or `converged=False` in `diagnostics`.

Retry with `warm_start=False`, add a `fallback` ladder to the layer, and for kinked laws use
`line_search="convex"`. These steps, and what each costs, are in
[Choosing solvers and coupling schemes](solvers.md#when-newton-does-not-converge). Catch
`noodl.solvers.ConvergenceError` to retry on non-convergence alone; every other error is a bug or a
setup mistake and should pass through.

## A singular or ungrounded system

**Symptom.** A solve raises `SingularSystemError`, `GroundingError` or `LinearSolveError`
(all `RuntimeError`s, importable from `noodl.solvers`), sometimes in mid-run.

**Cause.** A potential problem needs one grounded (boundary) node per connected component. A
network loses its grounding when a fan reaches shutoff, a damper closes or a part turns out to be
isolated; a loop of ideal potential sources, a cut set of flow sources, or a part reached only
through flow-prescribed edges is singular in itself. At the consistent initial point of a circuit
every storage law is held at its initial value and acts as a source, so the error suggests
starting from the operating point (`mode="dc"`).

**Fix.** The error names the layer and the component. Add a boundary node to the component,
or reconnect the part that was cut off.

## float32 or float64

`Network()` defaults to float32. Pass `dtype=torch.float64` (as the examples do), or move a built
model with `model.to(dtype=...)` (see [Devices, dtype, copying and saving](devices.md)). The
precision changes two things.

- **Tolerances.** Newton's default tolerances follow the dtype (about 1.2e-4 relative for float32,
  1.5e-8 for float64). A tolerance tighter than the dtype can reach never converges. An application
  that pins its own tight defaults (`water_steady`'s 1e-11, the building and electrical solves)
  needs looser ones in float32, `atol=rtol=1e-5` or so.
- **Mixed dtypes.** State and drivers are yours: pass them in the model's dtype and on its device.
  Drivers in another dtype than the model's are a mismatch. Moving the network alone with `Network.to`
  makes the layers refuse to run; move the whole model with `Model.to`.

Use float64 for calibration against measurements and for gradient checks; float32 suits large
ensembles on a GPU once the answer is validated against float64.

## Check messages

`model.check(state, drivers)` returns a report whose `issues` are `[error]`, `[warning]` or
`[info]` lines with a code. `.raise_for_errors()` raises `ValueError` for errors;
`.raise_for_errors(strict=True)` raises for warnings too. The codes:

| Code | Level | Meaning and fix |
|---|---|---|
| `type` | error | A value is not a `torch.Tensor`. Build it with `initial_drivers` or `torch.tensor`. |
| `shape` | error | The trailing shape does not match the key's order. Build the tensor by name. |
| `missing-input`, `missing-state` | error | A required driver or state key was not given, or a closure-carried state key is missing. A closure that writes a capacity driver needs `Model.initial_capacities` in the state. When closures might write the missing driver, `missing-input` is a warning and `probe=True` finds out. |
| `batch-shape` | error | Batch shapes of the keys do not broadcast. |
| `value-on-fixed-node` | error | A nonzero value at a boundary or inactive node, which the layer would drop or refuse. |
| `wrong-dictionary` | error | A state key given in the drivers, or a driver key in the state. |
| `two-flow-sources` | error | A transport layer's flows were given as a driver, but a potential layer already provides them. Remove the driver. |
| `attributes-changed` | error | A node or edge attribute was edited after the layer was built; the layer copied the old values. Rebuild the laws, layers and model. |
| `probe-needs-inputs`, `probe-failed` | error | `probe=True` needs both state and drivers, or the probe step itself raised; the message gives the exception. |
| `unknown-layer-key`, `unknown-layer-prefix` | warning | A key with a layer's prefix but no such input, or a near-miss of a layer name. Usually a typo; the message suggests the name. |
| `unused-edge-kind` | warning or info | No layer uses an edge kind. A warning when it is a near-miss of a kind a layer uses (`"airpth"`), else information (a closure may work on it). |
| `value-range` | warning | A temperature below 150 K: a Celsius value where kelvin is expected. |
| `overwritten-output` | warning | A closure replaces what an earlier closure wrote without declaring it among its `inputs`. |
| `declared-vs-probed` | warning | A closure returned a key it does not declare among its `outputs`. |
| `unit-mismatch` | warning | A closure or input declares a unit different from the field's. |
| `unused-key` | warning | With a probe: a key nothing read. |
| `closure-query-failed` | warning | Evaluating the closures once at the given state raised; the message gives the exception. |
| `names-mismatch` | warning | The `names=` map was flattened into a different network than the model's. |
| `undeclared-closure`, `unit-undeclared`, `unconnected-port` | info | A closure declares no inputs or outputs, so some checks cannot see it; a field has no unit; a component port is connected to nothing and is closed. |

A key the model knows nothing about, such as a custom closure's own input, is not rejected without
a probe. Run `model.check(state, drivers, dt=600.0, probe=True)` to learn which keys the model
actually reads.

## A layer raises after the network changed

**Symptom.** `RuntimeError: the network topology has changed` from `step`, `steady` or any solve.

**Cause.** A node or edge was added after the layers were built; a layer keeps the edge indices
and node count of construction time. **Fix.** Build the network completely, then build the layers
and the model again. See [Devices, dtype, copying and saving](devices.md).
