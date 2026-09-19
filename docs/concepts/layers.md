# Layers and models

A **layer** is one physical process on one network. A **`Model`** holds several layers on one
network and steps them together.

This page covers the part of noodl least like other network solvers: there are four different
ways an edge flow can be determined, and choosing the right one for your problem matters more
than any other modelling decision you will make.

## The four flow-determination modes

Most network tools support exactly one of these. noodl supports all four, and they compose on
one graph.

| # | Mode | The flow comes from | Implemented by |
|---|---|---|---|
| 1 | **Potential solve** | Newton's method on a nodal conservation residual: $A_I\,g(A^\top\phi) = s_I$ | `PotentialFlowLayer` |
| 2 | **Driver-prescribed** | Handed in directly as a driver value | any `TransportLayer` with no potential owner |
| 3 | **Closure-computed** | Computed from other state by an arbitrary function | a closure writing `"<layer>.q"` |
| 4 | **Clip-and-allocate** | A *request*, clipped against arc capacity and receiver headroom — no potential variable anywhere | `CapacitatedTransferLayer` |

![The four flow-determination modes: potential solve, driver-prescribed, closure-computed, clip-and-allocate](../assets/four-modes.svg)

Worked examples of each, from the applications:

- **1** — A building's airflow network. Pressures are unknown; flows follow from them.
- **2** — Not usually chosen alone; it is the escape hatch for a flow you measured or imposed.
- **3** — A street canyon. The along-canyon velocity is a closed-form function of the wind aloft
  and the canyon geometry, so there is no potential to solve for; `StreetFlows` computes every
  flow and writes it. The sewer's water side likewise: on a tree the cycle space is empty, so
  continuity alone fixes every pipe discharge in closed form.
- **4** — A water-resources network. An arc *requests* a transfer; the receiving reservoir
  accepts what its remaining headroom allows; the rest is unmet. There is no head, no pressure,
  no potential — the rule *is* the physics.

Getting this wrong is expensive. Modelling a gravity sewer as a potential problem, or a
water-allocation rule as a pressure network, produces a model that converges to a confidently
wrong answer.

## `PotentialFlowLayer`

Solves for nodal potentials such that flow is conserved at every interior node.

```python
from noodl.layers.potential import PotentialFlowLayer

layer = PotentialFlowLayer(
    net,
    name="air",
    elements=[leak],            # one Element per edge kind
    drives=[stack],             # additive potential-difference terms
    boundary=["ambient"],       # nodes whose potential is prescribed
    linear_solver="auto",       # 'auto' | 'cg' | 'gmres' | 'direct' | 'sparse_direct'
    quantity="pressure",        # metadata a Model reports
    unit="Pa",
)

phi, q = layer.solve(phi_boundary, drivers, sources=sources, differentiable=True)
```

`quantity` and `unit` are metadata only — nothing numerical reads them. They exist so that a
model composing several potential layers over one network can tell a pressure layer from a
temperature one without matching on names.

Useful accessors: `flows(phi, drivers)` evaluates the branch law; `flows_of_kind(q, kinds)`
extracts one kind's block from the concatenated flow vector; `residual(...)` and `jacobian(...)`
expose the Newton pieces directly; `kind_slice(kind)` and `element_for(kind)` locate a kind.

`diagnostics`, when you pass a dict, is filled with `newton_iterations`, `linear_iterations`,
`method`, `backend`, `converged` and `residual_norm`. `method` is what you asked for; `backend`
is what actually ran. They are deliberately separate — see [Solvers](solvers.md#solve-the-auto-table).

## `TransportLayer`

Advects one or more species (or heat) along the flows of its kinds, with storage at nodes:

$$
\frac{dx}{dt} = M x + N x_b + \frac{s}{\text{capacity}}
$$

```python
from noodl.layers.transport import TransportLayer

layer = TransportLayer(
    net, "thermal",
    capacity=heat_capacities,       # per ACTIVE INTERIOR node
    flow_kind=("airpath", "door"),  # one or several advecting kinds
    boundary=["ambient"],
    n_species=1,
    conduction_kind="wall",         # optional diffusive edges
    conductance=ua,
    scheme="exact",                 # 'exact' | 'implicit' | 'trapezoidal'
    quantity="temperature", unit="K",
)
```

Three things catch people:

- **The active interior.** Nodes that no edge of this layer's kinds touches are *inactive* and
  have no row at all — not a zero row, which would be singular. `capacity` must be indexed by the
  active interior, which `active_interior(net, kinds, boundary)` computes. A wall-mass node with
  conduction edges and no airpath edge *is* an unknown of a thermal layer and is *not* one of a
  species layer over the same network.
- **`sources` is in full node order**, not interior order — zeros on boundary and inactive nodes.
  This differs from `capacity` on purpose: sources are an application-level quantity, capacity a
  layer-level one.
- **`flow_kind` may be several kinds**, and `q` is then their flows *concatenated in that order*.
  That is exactly what `PotentialFlowLayer.flows_of_kind(q, layer.flow_kinds)` produces, so heat
  advected by both `"airpath"` and `"door"` edges needs no new operator.

The three schemes: `"exact"` uses an augmented matrix exponential and controls its own error by
sub-stepping; `"implicit"` is implicit Euler; `"trapezoidal"` is the second-order variant. Use
`"exact"` for heat, `"implicit"` for species — the application defaults reflect this.

The exact scheme bases its sub-stepping on the operator's one-norm `||dt M||_1`, computed from
the state-independent matrix before any arithmetic on the state; if the predicted work exceeds
the budget, the step is refused, raising an error that names the layer and advises the implicit
or trapezoidal schemes. All coefficients the layer owns—carrier, transmission, kinetics, removal,
and conductance—are differentiable under every scheme.

## `Reaction`

Applied to a transport layer's state after its step, operator-split:

```python
from noodl.layers.reaction import FirstOrderDecay

model = Model(net, layers, reactions=[("species", FirstOrderDecay(k))])
```

`FirstOrderDecay` and `Photostationary` (Leighton NO/NO₂/O₃) ship built in; the sewer
application adds `SulfideGeneration`.

**`Model.steady` never applies reactions.** This is deliberate — a steady state under a reaction
is a different fixed-point problem, not the transport steady state. The street and sewer
applications each provide their own `street_steady` / `sewer_steady` helper that iterates
transport and chemistry together to a joint fixed point. If you have a reaction and you call
`Model.steady`, you get the transport-only answer.

## `CapacitatedTransferLayer`

The fourth mode. Each edge carries a *requested* flow; the layer clips it against arc capacity
and the receiver's remaining storage headroom, sharing proportionally when several edges compete
for one node's headroom.

```python
from noodl.layers.capacitated import CapacitatedTransferLayer

layer = CapacitatedTransferLayer(
    net, "cap", kind="link",
    s_max=storage_ceilings,   # per node, full node order; inf for unbounded
    c_arc=arc_capacities,     # per edge
    preference=weights,       # optional sharing weights, strictly positive
    mode="hard",              # 'hard' | 'smooth' | 'projection'
    n_passes=5,
)

s_new, f = layer.step(s, drivers, dt, diagnostics=diag)
```

In the simplest case the realised flow is just $f = \min(r,\; c_{\text{arc}},\; h)$. The three
modes differ in gradient behaviour, which is the whole reason there is more than one — see the
[capacitated transfer application page](../applications/capacitated.md) for the full treatment.

A capacitated layer is inherently discrete-time. A `Model` owning one refuses a steady pass and
refuses `residuals()` outright, because a clip-and-allocate rule has no steady meaning to report.

## `Model`

```python
from noodl.model import Model

model = Model(
    net,
    layers={"air": air_layer, "thermal": thermal_layer},
    closures=[density],
    reactions=[("species", decay)],
    coupling="pingpong",            # or "iterate"
    iterate_tol={"thermal": 0.01},  # REQUIRED when coupling="iterate"
    iterate_max=20,
    substeps={"species": 4},
)

state = model.step(state, drivers, dt=60.0)
state = model.steady(state, drivers)
```

### Key conventions

State and driver keys are namespaced by layer name:

| Key | Meaning |
|---|---|
| `"<layer>.phi"` | Nodal potentials, full node order |
| `"<layer>.q"` | Branch flows, the layer's kind order |
| `"<layer>.x"` | Transport state, interior order, `(n_i,)` or `(n_i, K)` |
| `"<layer>.s"` | A capacitated layer's per-node storage |
| `"<layer>.phi_boundary"` | Prescribed boundary potentials (driver) |
| `"<layer>.x_boundary"` | Prescribed boundary transport state (driver) |
| `"<layer>.sources"` | Nodal sources, **full node order** (driver, optional) |
| `"<layer>.capacity"` | Per-step capacity override (driver, optional) |
| `"<layer>.requests"` | A capacitated layer's per-edge request (driver, required) |

### The order within a pass

Closures → potential solves → capacitated steps → transport steps → reactions.

The capacitated step sits between the potential solves and the transport steps so that a
transport layer reading `"<layer>.q"` sees a freshly written flow, whichever kind of layer wrote
it.

### Closures

A closure is `(state, drivers) -> driver updates`. It is the escape hatch for anything that is a
function of solved state: air density from temperature, canyon velocities from wind, tank levels
from the previous step.

```python
def density(state, drivers):
    T = state["thermal.x"]
    return {"rho": P_REF / (R_AIR * T)}
```

Closures may **not** write layer state keys. A closure may carry *its own* state across steps by
declaring `state_keys`, which `Model` copies from its return into the returned state — a sewer
manhole level, or a water tank level and its controlled links' status. Two closures claiming one
key is refused at construction, naming both, rather than resolved by whichever ran last.

### The two couplings, and the trap in the default

![Ping-pong takes one pass per step; iterate repeats the pass until the named states stop changing](../assets/coupling-modes.svg)

`coupling="pingpong"` (**the default**) takes exactly **one** pass per step, with the state at
the *start* of the step. It is cheap, it is what a weakly coupled model wants, and its splitting
error is first order in `dt`.

`coupling="iterate"` — Hensen's "onion" — repeats that pass within the one step until the
transport states named in `iterate_tol` stop changing, at most `iterate_max` times. Successive
substitution with 0.5 relaxation, applied with a one-pass delay: pass 1 sees the step-start
state, pass 2 sees pass 1's transport states unrelaxed, and from pass 3 on each pass sees the
mean of the two before it. Every pass re-advances from the step-start state, so passes never
compound into `passes × dt`.

**The trap.** Calling `.steady()` at the defaults on a two-way coupled model solves the airflow
at the temperatures the closures saw — the *initial* ones — and never re-converges it against
the temperatures it produced. It does not warn you; it returns a plausible number. Whenever the
flow depends on what it carries, pass `coupling="iterate"` with an `iterate_tol`.

Three more things about `iterate_tol`:

- It is **absolute**, in each layer's own units. A species layer whose mass fractions are
  $\sim10^{-3}$ and a thermal layer in kelvin cannot share a tolerance, which is why it is a
  per-layer mapping. A transport layer left out of it is stepped every pass but not tested.
- It has a **floor**: the potential solve's own residual, propagated through $dT/dF$. A tolerance
  tighter than that can never be met, and the run raises naming the layer, the change and the
  tolerance. A tight `iterate_tol` needs a matching `atol`/`rtol` on the `step`/`steady` call.
- Convergence is decided **per batch instance**. A batch that fails raises naming the instances;
  `on_failure="return"` with a `diagnostics` dict returns instead of raising.

### Introspection

```python
model.residuals(state, drivers)   # per-layer nodal balance residual
model.ports(state)                 # what a coupled or external model may prescribe and read
```

`Ports` reports boundary nodes, prescribable keys, boundary flows, and each transport layer's
flow driver key. `Model.flow_layer_of[name] is None` tells you whether a transport layer's flows
are yours to prescribe. This is the surface the [coupling application](../applications/coupling.md)
builds on.

### Differentiability

Ping-pong is one pass of differentiable operations. Iterate is differentiable by unrolling its
passes — memory grows with pass count — and the convergence *decision* is made on detached
copies, so it never appears in the graph. `**solve_kwargs` on `step`/`steady` reach the
**potential solves only** (`differentiable`, `on_failure`, `method`, Newton options); transport
steps always raise on failure.
