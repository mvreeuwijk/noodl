# Model options

Reference for the `Model` keys, coupling options and closure declarations. For how the pieces fit together see [Layers and models](../concepts/layers.md); for the four ways a flow is determined see [The four flow-determination modes](../concepts/four-modes.md).

## `Model`

```py
from noodl.model import Model

model = Model(
    net,
    layers={"air": air_layer, "thermal": thermal_layer},
    closures=[density],
    reactions=[("species", decay)],
    coupling="pingpong",            # or "iterate"
    iterate_tol={"thermal": 0.01},  # REQUIRED when coupling="iterate"; transport or Kirchhoff layers
    iterate_max=20,
    substeps={"species": 4},
)

state = model.step(state, drivers, dt=60.0)
state = model.steady(state, drivers)
```

`iterate_tol` may name a transport layer (converged on its state `.x`) or a Kirchhoff layer
(converged on the largest change of its potential `.phi` over its interior and boundary nodes, in
the potential's units, per instance). A Kirchhoff layer is named when a closure computes that
layer's element coefficients, or its boundary potentials, from its own previous solution, as the
sewer dynamic wave does with its conduit coefficients and outfall heads: for example
`iterate_tol={"water": 1e-6}` (the sewer's head layer) repeats the pass until no head moves by
more than 1 micrometre.
Its `.phi` and `.q` are relaxed with `iterate_relaxation` like a transport `.x`. Kirchhoff layers not
named in `iterate_tol` are not iterated.

`iterate_flow_tol={"water": 1e-9}` (optional, flow units) also judges such a layer's flows: a pass
converges only when its returned `.q` is within that of the `.q` it was fed. It matters when the
potential solve stops on its warm start (its residual already within `atol`), which hands the fed
potential back unchanged while the flows the closures read are still moving.

`iterate_acceleration="anderson"` (opt-in; default `None`) replaces the relaxation of those
Kirchhoff layers' `.phi` and `.q` by Anderson mixing (type II, Walker & Ni 2011) over the last
`iterate_memory` passes (default 5), and needs an `iterate_flow_tol` for each of them: the mixing
minimises the fixed-point residual of `.phi` and `.q`, each divided by its tolerance, and the
convergence test then also asks that residual of `.phi` to be within `iterate_tol`. A
conduction-only transport layer (`flow_kind=()`, a heat network) named in `iterate_tol` is mixed
and judged on its fixed-point residual like a Kirchhoff layer's `.phi`; advective transport
layers stay relaxed. Mixing changes only the sequence of fed states, never the fixed point, and the
fixed-point adjoint is unchanged. Two safeguards: an ill-conditioned history drops its oldest
passes, and a mixed step that would move any entry against its own residual is replaced by the
relaxed step, because the relaxed iteration only reaches fixed points that attract it and a secant
step can head for one that repels it (a closure may have both; the sewer's flat, near-dry conduits
do). The sewer dynamic wave builds with it and needs two to four times fewer passes.

### Key conventions

State and driver keys are namespaced by layer name:

| Key | Meaning |
|---|---|
| `"<layer>.phi"` | Nodal potentials, full node order |
| `"<layer>.q"` | Branch flows, the layer's kind order |
| `"<layer>.storage_flow"` | A Kirchhoff layer with [node storage](kirchhoff-options.md#node-storage): the flow into every storage entry at the end of the step (`layer.storage_flows`), zeros after a steady solve; the trapezoidal scheme's step-start flow, zero where it is missing. Under `scheme="forward_euler"` (storage at boundary nodes) the flow the solve delivered to every storage node, on a steady solve too |
| `"<layer>.x"` | Transport state, interior order, `(n_i,)` or `(n_i, K)` |
| `"<layer>.s"` | An allocation layer's per-node storage |
| `"<layer>.phi_boundary"` | Prescribed boundary potentials (driver) |
| `"<layer>.x_boundary"` | Prescribed boundary transport state (driver) |
| `"<layer>.sources"` | Nodal sources, **full node order** (driver, optional) |
| `"<layer>.capacity"` | Per-step capacity override (driver, optional) |
| `"<layer>.removal"` | Per-step first-order removal rate, 1/s, `(n_i, K)`, replacing the layer's construction-time `removal` (driver, optional; not carried in the state, so a closure that sets it must write it every step) |
| `"<layer>.conductance"` | Per-step conduction conductance of each conduction edge, `(b_c,)` in the order of `layer.conduction_edges`, replacing the layer's construction-time `conductance` (driver, optional; zero or positive; not carried in the state, so a closure that sets it must write it every step) |
| `"<layer>.transmission"` | Per-step pass fraction of each flow edge, `(b_flow,)` or `(b_flow, K)` in the layer's flow order, replacing the layer's construction-time `transmission` (driver, optional; not carried in the state, so a closure that sets it must write it every step) |
| `"<layer>.requests"` | An allocation layer's per-edge request (driver, required) |
| `"<layer>.frequency"` | A phasor layer's frequency in Hz, one per instance (driver, required) |

### The order within a pass

Closures → potential solves → allocation steps → transport steps → reactions (reactions before
the transport steps under `reaction_order="before_transport"`).

The allocation step sits between the potential solves and the transport steps so that a
transport layer reading `"<layer>.q"` sees a freshly written flow, whichever kind of layer wrote
it.

### Layers with flow unknowns, dynamic laws or laws that read the context

A Kirchhoff layer with implicit or multi-port laws is solved with the flows of its implicit-law
edges as unknowns (`layer.implicit_edges`); one whose laws are dynamic or read the context
(without either) is solved on the fast path. `Model` hands such a layer what its laws need:

- the warm start of those flows from `"<layer>.q"` in the state, when it is there;
- the time the laws are evaluated at: the **end** of a step (`t + dt` for `step(..., t=t)`),
  the `t` of `steady(..., t=t)`, or `None` when no time is given (every Kirchhoff layer is
  given it, whichever path solves it);
- on a step of a layer with dynamic laws (`layer.dynamic`: a capacitance, an inductance), the
  step-start `"<layer>.phi"` and `"<layer>.q"` as its history, and `dt`. Both keys are then
  required in the state (`state_from` and `check` say so); give the first step a consistent
  initial point, e.g. `layer.solve(..., mode="initial")`. The history is always the
  step-start state, so `coupling="iterate"` re-steps every pass from the same start.

`scheme=` (`"trapezoidal"` or `"backward_euler"`) travels through `**solve_kwargs`; the
keywords the Model sets itself (`phi0`, `q0`, `history`, `dt`, `t`, `storage_flow`) are refused
there by name.

A Kirchhoff layer with [node storage](kirchhoff-options.md#node-storage) gets the same step context (`t`, and on a
step `dt` and the step-start `"<layer>.phi"` as `history`; `"<layer>.q"` is not read, so it
is not required), and also the step-start `"<layer>.storage_flow"`, which the trapezoidal
scheme reads (a missing one is zero: the storage at rest at the step start). After every solve the Model writes the
new `"<layer>.storage_flow"`, zeros on a steady pass, so a run started from `steady` carries
it from the first step. A state that is not at rest but carries zero `storage_flow` (built by
hand, missing, or with the `refs` default of 0) starts the trapezoidal scheme assuming no storage flow
at the step start, which can ring; take the first step with `scheme="backward_euler"`, as
`water_step` does by default.
`steady` solves such a layer at its operating point (a capacitance open, an inductance
shorted). A layer with laws that read the context but no dynamic one (a time-dependent source and
resistors, no storage) takes no history: on a step it is solved at its operating point at the
step end, `t + dt`, with `ctx.mode == "dc"` and `ctx.dt` `None`, which is exact for a layer that
stores nothing. `residuals` reports `layer.balance(phi, q, sources)`, the interior balance of the
state's own flows, for a layer with flow unknowns or a law that reads the context (whose flows
need a time and a step start `residuals` does not have). A phasor layer (`phasor=True`) has complex `"<layer>.phi"` and
`"<layer>.q"` and reads the driver `layer.frequency_key`; the iterated coupling judges its
convergence on the modulus of the change and differentiates its fixed point in the real view.
Anderson mixing (`iterate_acceleration="anderson"`) is refused on a phasor layer.
`line_search="convex"` (by keyword or through `newton_defaults`) is refused on a layer with flow
unknowns: its residual is not the gradient of a convex function.

### Closures

A closure is `(state, drivers) -> driver updates`. It is the escape hatch for anything that is a
function of solved state: air density from temperature, canyon velocities from wind, tank levels
from the previous step.

```py
def density(state, drivers):
    T = state["thermal.x"]
    return {"rho": P_REF / (R_AIR * T)}
```

Closures may **not** write layer state keys. A closure may carry *its own* state across steps by
declaring `state_keys`, which `Model` copies from its return into the returned state — a sewer
manhole level, or a water tank level and its controlled links' status. Two closures claiming one
key is refused at construction, naming both, rather than resolved by whichever ran last.

### Closures and the clock

A closure that only computes coefficients or forcing terms keeps the two-argument form `closure(state, drivers)`.
A closure that integrates some state over the step (such as a storage sweep that advances manhole levels)
declares `integrates = True` and receives a third argument, `ctx = StepContext(dt, t)`, where `dt` is
the integration interval and `t` the step's start time if tracked. Inside `step(...)` the context carries the
interval; inside `steady(...)` it refuses the closure by name because there is no interval to integrate over.
Queries (`residuals`, `current_flows`, and the sewer report) call closures with `ctx = None`, where an
integrating closure evaluates its outputs at the given state without advancing it. A closure declaring `state_keys`
must also declare `integrates` (True or False) so the model knows whether its carried state changes step to step;
omitting it is refused by name.

### What a closure may declare

Everything a closure says about itself is an optional attribute, listed in
`noodl.contracts.CLOSURE_ATTRIBUTES` and read once, at construction, into a `ClosureSpec`
(`integrates`, `state_keys`, `inputs`, `outputs`, `input_specs`, `key_labels`, `edge_kinds`,
`requires_coupling`, `reads_flows_of`, `species_columns`). A malformed declaration is refused
at construction by name:

- `inputs`, `outputs`, `state_keys` and `edge_kinds` must be tuples of names; a bare string
  (`outputs = "rho"`) is a `TypeError`;
- `integrates` must be `True` or `False`;
- a declared output may not be a layer's state key (`"<layer>.phi"`, `.q`, `.x`, `.s`), apart
  from `"<layer>.q"` of a transport layer no Kirchhoff layer owns;
- `species_columns = {layer: {species: column}}`, for a closure or reaction that addresses
  species by column index, must agree with the layer's `species_names` and `n_species`.

`model.refs.ownership()` lists, for every key, who writes it and who reads it (layers,
closures by their declarations, the caller); `model.check` reports closures that overwrite
each other's declared outputs (`overwritten-output`), declare nothing (`undeclared-closure`),
return keys they did not declare when probed (`declared-vs-probed`), or declare a unit that
differs from the field's (`unit-mismatch`, with `unit-undeclared` for a field with no unit).

### Introspection

```py
model.residuals(state, drivers)   # per-layer nodal balance residual
model.ports(state)                 # what a coupled or external model may prescribe and read
```

`Ports` reports boundary nodes, prescribable keys, boundary flows, and each transport layer's
flow driver key. `Model.flow_layer_of[name] is None` tells you whether a transport layer's flows
are yours to prescribe. This is the surface the [coupling application](../guide/coupling.md)
builds on.

### Differentiability

Ping-pong is one pass of differentiable operations. Iterate runs its passes to convergence
without a graph, then re-runs the certified pass once more on the graph and attaches the
implicit adjoint of the interface equations — memory is one pass, not all of them, and the
gradient error is of the order of the primal residual rather than tied to the pass count. The
convergence *decision* is still made on detached copies, so it never appears in the graph. See
[Differentiability](../concepts/differentiability.md#what-is-guaranteed) and
[Coupling](../guide/coupling.md) for the full contract.
`**solve_kwargs` on `step`/`steady` reach the **potential solves only** (`differentiable`,
`on_failure`, `method`, Newton options); transport steps always raise on failure.
