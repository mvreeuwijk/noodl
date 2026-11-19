# Kirchhoff layer options

Reference for the options of `KirchhoffLayer`: node storage, implicit, coupled and dynamic laws, and what a solve takes. For how the pieces fit together see [Layers and models](../concepts/layers.md); for the four ways a flow is determined see [The four flow-determination modes](../concepts/four-modes.md).

## `KirchhoffLayer`

Solves for nodal potentials such that flow is conserved at every interior node.

```py
from noodl.layers.kirchhoff import KirchhoffLayer

layer = KirchhoffLayer(
    net,
    name="air",
    laws=[leak],                # one ConstitutiveLaw per edge kind
    boundary=["ambient"],       # nodes whose potential is prescribed
    linear_solver="auto",       # 'auto' | 'cg' | 'gmres' | 'direct' | 'sparse_direct'
    quantity="pressure",        # metadata a Model reports
    unit="Pa",
    phasor=False,               # True: complex sinusoidal steady state (see below)
)

phi, q = layer.solve(phi_boundary, drivers, sources=sources, differentiable=True)
```

`quantity` and `unit` are metadata only — nothing numerical reads them. They exist so that a
model composing several Kirchhoff layers over one network can tell a pressure layer from a
temperature one without matching on names. They are still checked for consistency: every key
of a layer is a `Field` carrying its `unit` (a transport layer's `unit`, `source_unit`,
`capacity_unit`, `flow_unit` and `conductance_unit` give the units of its state, sources,
capacity, flows and conductances), and
`Model.check()` warns (`unit-mismatch`) when a closure declares a different unit for a key it
reads (`input_specs`) or writes (`output_units`), and notes (`unit-undeclared`) a field with
no unit. Units are compared as strings after trivial respelling (`m3` and `m^3`, `1` and `-`);
nothing is converted. See [The closure contract](../project/extending/index.md).

Useful accessors: `flows(phi, drivers)` evaluates the branch law; `flows_of_kind(q, kinds)`
extracts one kind's block from the concatenated flow vector; `residual(...)` and `jacobian(...)`
expose the Newton pieces directly; `kind_slice(kind)` and `law_for(kind)` locate a kind.

`diagnostics`, when you pass a dict, is filled with `newton_iterations`, `linear_iterations`,
`method`, `backend`, `converged` and `residual_norm`. `method` is what you asked for; `backend`
is what actually ran. They are deliberately separate — see [Solvers](../concepts/solvers.md#solve-the-auto-table).

### Node storage

A node stores an amount; the layer integrates it. A `NodeStorage` (`noodl.storage`, exported as
`noodl.NodeStorage`) says how much is stored at each of its nodes as a function of that node's
own potential, `S(phi)`: the charge of a capacitor to ground, the water in a tank, the air in a
zone. Over a step of length `dt` the layer turns the change of that amount into a withdrawal,
positive into storage, with the step rate of its scheme (`noodl.elements.step_rate`):

$$
w = \frac{S(\varphi) - S(\varphi_{start})}{\Delta t} \quad \text{(backward Euler)}, \qquad
w = \frac{2\,(S(\varphi) - S(\varphi_{start}))}{\Delta t} - w_{start} \quad \text{(trapezoidal)}.
$$

```py
from noodl import LinearStorage

store = LinearStorage(nodes=[1, 2, 3], capacity=[0.4, 1.1, 0.7])   # S = C phi
layer = KirchhoffLayer(net, "heat", [conductance], boundary=["ambient"],
                       storage=[store], scheme="backward_euler")
phi, q = layer.solve(phi_b, drivers, sources, dt=60.0, history=(phi_prev, q_prev))
w = layer.storage_flows(phi, drivers, dt=60.0, history=(phi_prev, q_prev))
```

Storage adds no flow unknowns; a layer of potential laws with storage stays on the nodal
path. Like a withdrawal it adds only a diagonal, `step_rate_factor * capacity` with `capacity = dS/dphi`, to the Jacobian, so a
network of conductances with storage keeps its symmetric positive definite system and its
sparse or CG solve. A node whose only connection is its storage is grounded during a step and
reported floating at a steady solve. Several storages may share a node (their withdrawals add
up); one `(node, storage)` pair is an *entry*, and `storage_flows` returns one flow per entry,
storages in list order, then their nodes.

- Without `dt` storage is absent: the steady point (mode `"dc"`) stores nothing.
- With `dt` the layer needs `history=(phi, q)` at the step start (only `phi` is read; `q`
  may be `None`) and, under the trapezoidal scheme, `storage_flow`, the flow into every entry
  at the step start.
- `residual`, `jacobian` and `adjoint` take the same step keywords (`dt`, `scheme`,
  `history`, `storage_flow`): with `dt` they include the storage of that step, without it
  they are the steady forms, where storage is absent.
- `mode="initial"` with a storage that holds an `initial()` potential holds its node there.
  The held rows make that one solve's system nonsymmetric, so it is solved on the sparse
  operator of a layer with flow unknowns (a layout with no unknown flows, built on the first
  such call); every other solve of the layer keeps the symmetric system, and that solve refuses
  `linear_solver="cg"` and `line_search="convex"`.
- `layer.dynamic` is True for a layer with storage, and `layer.linear` asks every storage too.
- On the differentiable path gradients reach the storages' registered parameters, `history`
  and `storage_flow`, so a chain of steps back-propagates through time.

**Storage at boundary nodes, stepped by forward Euler.** With `scheme="forward_euler"` the
storage sits at boundary nodes instead (an EPANET tank), and is never an unknown. The solve
holds each storage node at its potential at the step start (`layer.holds_storage` is True),
which it reads from `history=(phi, None)`, `phi` every node's potential then: every call
that evaluates the layer at a boundary (`solve`, `residual`, `jacobian`, `adjoint`,
`linear_init`) takes it, steady or not, and refuses by name to run without it.
`"<layer>.phi_boundary"` gives the other boundary nodes only, `layer.prescribed` in boundary
order. A `Model` passes the step-start `"<layer>.phi"` as `history` on every path (`step`,
`steady`, `residuals`, `current_flows`, `check`), as it passes every other layer's step start.
During a step a closure reading `"<layer>.phi"` sees such a node at its step-start potential
too (`hold_potentials`), even in the later passes of `coupling="iterate"`. After the solve
the step moves each storage node by $\Delta t\,w / C$ (`advance_storage(phi, storage_flow,
dt)`), with $w$ the flow the solve delivered to the node and $C$ the storage's capacity.
That is explicit Euler, EPANET's own tank method; the solve itself takes no step and does
not depend on `dt`. A storage whose capacity changes with the potential gives the inverse of
its amount, `potential_of`, and moves to `potential_of(amount(phi) + dt * w)` instead, so its
amount changes by exactly $\Delta t\,w$ (EPANET's tank on a volume curve,
`noodl.apps.water.TankVolumeCurve`).

```py
store = LinearStorage(nodes=[tank], capacity=[area])
layer = KirchhoffLayer(net, "water", laws, boundary=["R1", "T1"],
                       storage=[store], scheme="forward_euler")
phi, q = layer.solve(phi_boundary, drivers, sources,    # phi_boundary: R1 only
                     history=(phi_start, None))         # T1 held at phi_start[tank]
w = layer.storage_flows(phi, drivers, dt=None, q=q)     # the flow into T1
phi_next = layer.advance_storage(phi, w, dt=600.0)      # T1 moved by dt * w / area
```

- `storage_flows` there needs `q` and returns the flow into every storage node, on a steady
  solve as on a step; a `Model` writes it to `"<layer>.storage_flow"` after every solve and
  moves the storage when the pass has a `dt`. `"<layer>.phi"` is then required by every
  solve, steady or not.
- A linear storage (`LinearStorage`, a constant capacity) or one that gives `potential_of` is
  accepted at a boundary node, one entry per node; a nonlinear one without it is refused by
  name.
- A layer whose storage sits at interior nodes, with dynamic laws, or with no storage at
  all refuses `"forward_euler"`; a layer with boundary storage refuses the other schemes, and a storage at
  a boundary node of a layer with another scheme is refused.
- Gradients reach the capacities through `advance_storage`; the solve does not depend on them.

Which method to write on a subclass:

- **`amount`** alone, when the stored amount depends on the node's own potential only. The
  default `change` is `amount(phi) - amount(phi_start)`, both at the *current* drivers (the
  lagged-coefficient form SWMM and EPANET use), and the default `capacity` is its autograd
  derivative.
- **`change`**, when the amount at the step start is not `amount(phi_start)` at the current
  drivers: an amount that depends on a driver that changes over the step (a zone's air mass
  depends on its temperature), or a difference that must be formed without round-off
  (`LinearStorage` forms `C (phi - phi_start)`).
- **`rate`** and **`rate_slope`**, for an application whose withdrawal has rules beyond storage
  (they default to `step_rate(change)` and `step_rate_factor * capacity`). `rate_slope` is
  used by both the non-differentiable and the differentiable solve, whose adjoint it builds,
  so an override that is not the true derivative (a Newton aid) declares
  `exact_slope = False`, and the differentiable solve then takes the derivative of `rate` by
  autograd for that storage (as for `NodeWithdrawal.dflow`).

A withdrawal that is not storage (a demand, a leak) is a `NodeWithdrawal`
([Constitutive laws](../concepts/elements.md)).

## Implicit, coupled and dynamic laws

A `KirchhoffLayer` whose constitutive laws all give their flow from their own edge's potential
difference has the interior potentials as its only unknowns, as above (the fast path). A law
that cannot, an implicit law (an `ImplicitLaw`) or a multi-port law
(`ports > 1`), adds the flows of its edges as unknowns (`layer.implicit_edges`, empty on a layer
without flow unknowns); a layer built with `phasor=True` is always solved this way. The
declarations are described under
[Implicit, coupled and dynamic constitutive laws](../concepts/elements.md#implicit-coupled-and-dynamic-constitutive-laws).
The [electrical application](../applications/electrical/index.md) is built on it; the same machinery
serves any domain with ideal sources, flow-controlled laws or storage on its edges.

**Laws that read the context stay on the fast path.** A law with a `ctx` parameter in `flow`
that still gives its flow from the potential difference (a time-dependent current source; a
storing law written in flow form, `q = C (dp - ctx.dp_prev) / ctx.dt` with `dynamic = True`)
adds no flow unknown. Its flow is evaluated in the solve's context, restricted to its edges, and
its slope is its `jacobian` there (its analytic one where it overrides it, else automatic
differentiation of `flow` with the context), since `dflow` takes no context. The Jacobian stays
the symmetric network matrix, so such a layer accepts `linear_solver="cg"` and
`line_search="convex"`, and its grounding is certified at the start of every solve: a node
reached only through a current source is refused there as floating, by name.

**The system.** The unknowns are the interior potentials and the flows of the implicit-law edges
(`layer.implicit_edges`, in layer edge order). The residual rows are the conservation balance at
the interior nodes, $A_I q - s_I$ (plus withdrawals), where $q$ is the law's flow on a
potential-controlled edge and the unknown on an implicit one, followed by every implicit law's
own residual. The Jacobian is assembled as a sparse, nonsymmetric matrix from each law's local
blocks (forward-mode derivatives, one per port, two on a phasor layer), so a multi-port law
couples the edges of its group exactly. `solve` still returns `(phi, q)`, with `q` covering every edge.

**The methods on a layer with flow unknowns.** `flows`, `dflows`, `residual`, `jacobian`,
`adjoint` and `linear_init` are defined on every layer. On a layer without flow unknowns every
flow is a function of its own edge's potential difference, so the potentials and drivers give
everything. An implicit law's flow is not: a fan with a prescribed rise has a residual that does
not contain $q$, and conservation at the nodes fixes its flow. So on a layer with flow unknowns
the methods that need the flows take `q`, as `solve` returned it (every edge), and refuse a
missing one with a `ValueError` that names the implicit kinds.

- `flows(phi, drivers, q)` returns the implicit edges' columns as given and evaluates the
  potential-controlled edges' columns from `phi` (steady context), differentiably. At `solve`'s
  own `(phi, q)` the result is `q`. A phasor layer has no such accessor (`TypeError`).
- `dflows(phi, drivers, q)` is every edge's slope `d q / d dp`. On an edge of an implicit law
  `r(dp, q) = 0` it is the slope the law implies, `-(d r / d dp) / (d r / d q)` at `q`'s
  flow, and `nan` where `d r / d q` is zero (an ideal source: its flow does not follow from its
  potential difference). Every edge of a law with several ports is `nan`. A phasor layer has no
  such accessor (`TypeError`: its slopes are complex).
- `residual(..., q=q)` is the residual of the unknowns `[phi_I, q_F]`, the interior potentials
  then the flows of `implicit_edges`: the balance rows, then the implicit laws' own residuals; it is
  zero at `solve`'s result. A phasor layer's rows are complex.
- `jacobian(..., q=q)` is the dense matrix of the operator `solve`'s Newton iteration builds, of
  size `n_I + n_F` (twice that on a phasor layer, in the real view: each complex unknown a
  (Re, Im) pair).
- `adjoint(..., grad, q=q)` solves `J^T lambda = grad` with that matrix; `grad` and the result
  have the width of the unknowns (the real view on a phasor layer).

On a layer without flow unknowns `residual` keeps its `n_I` balance rows, and `q` is ignored
by `residual`, `jacobian` and `adjoint`. The balance of a given state is `balance(phi, q,
sources)`.

**One scheme vocabulary for every layer.**
`SCHEMES = ("exact", "backward_euler", "trapezoidal", "forward_euler")`
(`noodl.elements.base`) names the time-integration schemes of both layer kinds, with the keyword
`scheme` everywhere. A `KirchhoffLayer` takes `scheme="trapezoidal"` (the default) or
`"backward_euler"` at construction, and `solve(scheme=None)` means "the layer's own"; it refuses
`"exact"`, which belongs to a `TransportLayer`, by name. `"forward_euler"` is for a
`KirchhoffLayer` whose storage sits at boundary nodes only
([Node storage](#node-storage)); a `TransportLayer` refuses it. A name outside `SCHEMES` is refused with an error
that lists them.

**Construction checks**, each raising `ValueError` naming the layer and the kind or nodes:

- every kind's edge count is a multiple of its law's `ports`;
- every connected part of the layer's edges contains a boundary node (a ground): a part with
  none (a transformer secondary left floating) has undetermined potentials;
- a dynamic kind carries no head (its step-start potential differences are taken without one);
- `linear_solver="cg"` is refused (the system is not symmetric positive definite);
- on a phasor layer, every law is `phasor_capable` and there are no withdrawals and no
  storage (a capacitance is an edge law there).

Such a layer takes [node storage](#node-storage) as well. On a step its withdrawal and its
analytic `rate_slope` enter each node's flow balance and the diagonal of the Jacobian, as on
the fast path (and the same `history`, `scheme` and `storage_flow`); in the steady operating
point (`mode="dc"`) storage is absent. At the consistent initial point (`mode="initial"`) each
node with a storage entry whose `initial()` is not `None` is *held* at that potential: its flow
balance is replaced by `phi = initial` (entries at one node with different initial values are
refused by name). The flow into storage there is whatever balances the node:
`storage_flows(phi, drivers, dt=None, mode="initial", q=q, sources=sources)` returns the
node's flow imbalance (its sources, less the net outflow of `q` and the withdrawals'
withdrawal), split among the node's entries in proportion to their `capacity`; the Model
writes it to `"<layer>.storage_flow"`, the trapezoidal scheme's step-start flow. A
`LinearStorage(node, C, initial=v0)` to ground is then the same circuit as a
`Capacitance(C, v0=v0)` edge to ground, and its storage flow the capacitor current.

A system that is structurally sound but singular (a loop of ideal potential sources, a cut set
of flow sources, a part reached only through flow-prescribed edges) raises a `RuntimeError` at
the solve naming the layer and its implicit kinds. At the initial point (`mode="initial"`) every
storage law is held at its initial value and so acts as a source itself (a capacitance as a
potential source, an inductance as a flow source); the error then says so and suggests starting
from the operating point (`mode="dc"`). Under `on_failure="return"` that singularity shows only
as `converged=False` in `diagnostics`.

`linear_init(phi_boundary, drivers, sources, q=False)` is defined on every layer. On a layer
with flow unknowns it is the linear guess of the potentials and the flow unknowns together: each
law with a flow is replaced by its tangent at zero (its own `linear_init`; a law with several
ports, and every law on a phasor layer, is taken at its flow and Jacobian at the starting
potentials instead: the same for a linear law), each implicit law by its residual and Jacobian at
zero interior potentials and its start flow (`start_flow`, zero unless the law gives one; see
[Implicit, coupled and dynamic constitutive laws](../concepts/elements.md#implicit-coupled-and-dynamic-constitutive-laws)),
the withdrawals and storages by their value and slope there, and the system is solved once. On a layer with flow unknowns the result is detached: no
gradient flows to `phi_boundary`, `sources`, the drivers or the law parameters, unlike on a
layer without flow unknowns. On a phasor layer it is complex. `q=True` returns `(phi_interior, q_implicit)`, the second the flows of
`implicit_edges` (empty on a layer without flow unknowns). A system the guess finds singular is
reported like a singular solve. A solve still starts from zeros, or from the warm start.

### What a solve takes: dc, initial, a step, a phasor

| Keyword | Meaning |
|---|---|
| `mode` | `None` (default): `"transient"` when `dt` is given, `"dc"` otherwise. `"dc"`: the steady operating point (no `dt`). `"initial"`: the consistent initial point (each storage law at its initial value; no `dt`). A phasor layer is always `"phasor"`. |
| `dt` | One implicit step of this length (s). |
| `t` | The evaluation time, the END of the step on a transient (a time-dependent source is evaluated there); `None` when untracked. |
| `history` | `(phi, q)` at the step start, full-node potentials and every edge's flow, as `solve` returns them. Required on a step of a layer with a dynamic law or storage; `q` is read only by a dynamic law and may be `None` otherwise. |
| `scheme` | `"trapezoidal"` or `"backward_euler"`; `None` (default) is the layer's own, `KirchhoffLayer(scheme=...)`, trapezoidal unless built otherwise. |
| `phi0`, `q0` | Warm start of the interior potentials and of the implicit-edge flows; zeros otherwise. |

Every layer checks these keywords the same way, by name, whichever path solves it, and builds
one step description per solve (`_Step`: the law context and what the storage does).

- **dc** (`mode="dc"`, or no `dt`): the steady operating point. Storage is absent, so no
  `history` or `storage_flow` is taken.
- **initial** (`mode="initial"`, no `dt`): the consistent initial point. A storage that holds
  an `initial()` potential holds its node there, on every layer (a layer without flow
  unknowns solves that one call on the sparse operator, see [Node storage](#node-storage)).
- **a step** (`dt > 0`): one implicit step ending at `t`, with the `scheme`. It needs
  `history`, and `storage_flow` under the trapezoidal scheme on a layer with storage. A layer
  with storage, a dynamic law or a law that reads the context takes a step; a layer with none
  of these has static laws, takes no step, and refuses `dt` and `history` (it accepts and
  ignores `t`, a valid `scheme`, `q0` and the modes `"dc"` and `"initial"`). In practice
  only a fast-path layer takes no step: an ideal or implicit law reads the context, so a
  layer with one steps. A layer whose laws read the context but none dynamic steps without
  history, at its operating point at the step end. A layer whose storage is all that steps
  takes `history=(phi, None)`: its laws read no step-start flow. `storage_flow` is refused
  on a layer without storage, and any step keyword without `dt`.
- **a phasor** (always, on a `phasor=True` layer): ignores `scheme` and refuses `dt`,
  `history` and any mode but `"phasor"`.

A scheme outside the two above is refused by name. On the differentiable path (the default)
gradients reach `history` too, so a chain of steps back-propagates through time.

**Linear solver.** `linear_solver="auto"` on such a layer factorises: SciPy's sparse LU
(`"sparse_direct"`, per instance) whenever SciPy is importable, at any batch size, and batched
dense LU (`"direct"`) only when it is not. A dense batched factorisation costs memory and work growing with the square and cube of the system size per instance, which is why the sparse factorisation is the default. A tiny circuit
at a very large batch may be faster with an explicit `method="direct"`. `"auto"` never picks
GMRES (the system is indefinite and badly scaled, with potentials and flows in one vector) and
refuses CG. An explicit `"gmres"`, `"direct"` or `"sparse_direct"` is honoured. `diagnostics["method"]` is the
method requested and `diagnostics["backend"]` the one that ran.

### Phasor layers

`KirchhoffLayer(..., phasor=True)` solves the sinusoidal steady state: `phi`, `q`, the
boundary potentials and the warm starts are complex (complex128). The frequency in Hz comes from
the driver `layer.frequency_key` (`"<name>.frequency"`, one value per instance) and reaches the
laws as `ctx.omega = 2 pi f`. Each complex unknown is solved as its (real, imaginary) pair, so a
law that is not holomorphic in `dp` (a constant-power load, `q = conj(S / dp)`) is solved
exactly, and the real solver machinery serves unchanged. A load is an edge to ground (node
sources are refused). `power_residual` is then the complex power balance, real and imaginary
parts both zero at a solution: Tellegen's theorem for phasors.

The initial guess is zero. A law that is singular or flat at `dp = 0`, such as a constant-power
load, needs a warm start: pass `phi0` (for example every interior potential at the nominal
source voltage).

### On a `Model`

`Model` gives such a layer its history, time and warm start; see
[Layers with flow unknowns, dynamic laws or laws that read the context](model-options.md#layers-with-flow-unknowns-dynamic-laws-or-laws-that-read-the-context).
