# Transport layer options

Reference for the options of `TransportLayer`, `Reaction` and `AllocatedFlowLayer`. The Kirchhoff layer is on [its own page](kirchhoff-options.md) and `Model` on [Model options](model-options.md). For how the pieces fit together see [Layers and models](../concepts/layers.md); for the four ways a flow is determined see [The four flow-determination modes](../concepts/four-modes.md).

## `TransportLayer`

Advects one or more species (or heat) along the flows of its kinds, with storage at nodes:

$$
\frac{dx}{dt} = M x + N x_b + \frac{s}{\text{capacity}}
$$

```py
from noodl.layers.transport import TransportLayer

layer = TransportLayer(
    net, "thermal",
    capacity=heat_capacities,       # per ACTIVE INTERIOR node
    flow_kind=("airpath", "door"),  # one or several advecting kinds
    boundary=["ambient"],
    n_species=1,
    conduction_kind="wall",         # optional diffusive edges: one kind or a tuple of kinds
    conductance=ua,
    scheme="exact",                 # 'exact' | 'backward_euler' | 'trapezoidal'
    quantity="temperature", unit="K",
)
```

Four things catch people:

- **The active interior.** Nodes that no edge of this layer's kinds touches are *inactive* and
  have no row at all — not a zero row, which would be singular. `capacity` must be indexed by the
  active interior, which `active_interior(net, kinds, boundary)` computes. A wall-mass node with
  conduction edges and no airpath edge *is* an unknown of a thermal layer and is *not* one of a
  species layer over the same network.
- **`sources` is in full node order**, not interior order — zeros on boundary and inactive nodes.
  This differs from `capacity` on purpose: sources are an application-level quantity, capacity a
  layer-level one.
- **`conduction_kind` may be several kinds too**: `("wall", "film", "radiation")` carries wall
  conduction, a surface (film) coefficient and a radiative coefficient as separate edges.
  `conductance` is then one tensor over the edges of all those kinds, the first kind's edges in
  network order, then the second kind's, and so on; `layer.conduction_edges` lists the kinds with
  their end nodes in exactly that order, so a closure can write one value per edge, and
  `layer.conductance` reads the construction-time values back. Every
  conduction kind counts as a touch for the active interior.
- **`flow_kind` may be several kinds**, and `q` is then their flows *concatenated in that order*.
  That is exactly what `KirchhoffLayer.flows_of_kind(q, layer.flow_kinds)` produces, so heat
  advected by both `"airpath"` and `"door"` edges needs no new operator. It may also be none,
  `flow_kind=()`: a conduction-only layer (a heat network of conductances and capacities, as the
  electrical application's heat ports use), whose `q` is empty; in a `Model` it has no flow
  driver and no flow owner.

The three schemes: `"exact"` uses an augmented matrix exponential and controls its own error by
sub-stepping; `"backward_euler"` is implicit Euler; `"trapezoidal"` is the second-order variant. Use
`"exact"` for heat, `"backward_euler"` for species — the application defaults reflect this.

`step(..., scheme=...)` takes one step with another scheme than the layer's own, checked as the
constructor checks it (a storage-free node with throughflow still needs `"backward_euler"`); in a
`Model`, `model.step(..., layer_options={"thermal": {"scheme": "backward_euler"}})` does the same for
one step. A layer coupled to a circuit steps this way with the circuit's own rule: backward
Euler on the step after a restart, trapezoidal otherwise. `step(..., sources_prev=...)`
(trapezoidal only) gives the sources at the START of the step; the trapezoidal rule then
integrates them as it integrates the state, `dt/2 (sources_prev + sources)`, instead of holding
`sources` over the step. A `Model` passes it when a closure carries the layer's sources as state
(`state_keys` holding `"<layer>.sources"`, as the electrical application's `JouleHeating` does):
the step-start state's `"<layer>.sources"` are then the heat at the step start.

The exact scheme bases its sub-stepping on the operator's one-norm `||dt M||_1`, computed from
the state-independent matrix before any arithmetic on the state; if the predicted work exceeds
the budget, the step is refused, raising an error that names the layer and advises the implicit
or trapezoidal schemes. A stiff operator (large `||dt M||_1`, e.g. fast removal with a nonzero
forcing) costs the exact scheme thousands of matvecs per step by nature; `scheme="backward_euler"` is
the documented choice there, not a larger budget. All coefficients the layer owns—carrier,
transmission, kinetics, removal, and conductance—are differentiable under every scheme.

**Boundary values linear in time.** By default, the boundary values stay constant over a step.
In the exact scheme, `step(..., x_boundary_end=...)` instead makes them vary linearly over the
step, from `x_boundary` at its start to `x_boundary_end` at its end (a first-order hold). The
step is then exact for a boundary value that ramps linearly while the flows are held fixed. This
matters for a zone that flushes much faster than the step. With a boundary value held at the
step mean of a ramp, such a zone relaxes to that mean and ends about half a step behind the
boundary. With the linear boundary, it ends behind by its own flushing time, as it should. The
Modelica importer's `scheme="midpoint"` runner uses this. The backward Euler and trapezoidal schemes
do not accept the argument and raise `ValueError` if it is given. So does `step_with_transfer`, which returns the boundary transfer: the transfer is defined for a boundary held constant over the step.

`x_boundary_rates=(start, end)`, given together with `x_boundary_end`, adds the boundary's rates
of change at both ends of the step. The boundary then follows the cubic Hermite through both
ends' values and rates instead of the straight line. A fast-flushing zone ends behind its
boundary by its flushing time times the boundary's rate at the step's end. On a curved boundary
the straight line gets that rate wrong: it uses the chord's slope, which is off by about half a
step times the curvature. The Hermite uses the end's own rate.

**Changing storage.** A transport layer's capacity may change over a step: a sewer pipe's wetted
volume, or a headspace volume as the water level changes. The step conserves the stored amount
`V x` in amount form: `V_new x_new - V_old x_old = dt F(x_new)` (implicit) or `= dt/2 (F(x_new) + F(x_old))`
(trapezoidal). In the trapezoidal form the old-time rate `F(x_old)` is evaluated with the OLD
capacity, which matters for removal and kinetics (they act on the amount `V x`), not for advection
or conduction (capacity-free amount rates). The argument `capacity_prev` (the storage at the step's start, when it differs from
the `capacity` argument at the step's end) implements this; a fixed-storage layer omits both and
recovers the classical form. The driver `"<layer>.capacity"` holds the storage at the step's end.
When a closure writes that driver, the step-start state must carry the key `"<layer>.capacity"`,
the storage at the state's own time, which `Model.initial_capacities(state, drivers)` builds by
querying the closures. The `"exact"` scheme has no changing-capacity form and raises by name if
`capacity_prev` differs from `capacity`.

**Dilution.** `dilution=` (per species, `None` by default) adds `dilution * net(q) / capacity`
to the removal, `net(q)` the signed net inflow of the flow edges into each node. Where the flows
do not balance because the stored volume changes, `dilution = carrier` is the advective form
`V dx/dt = sum_in carrier q (x_in - x)` and `dilution = 0` the conservative flux form; the
capacity is then a rate coefficient at the end of the step. `dilution_mode="withdrawal"` instead
reads only `max(net(q), 0)`: a FIXED volume whose flow imbalance is a nodal sink or source of
the carrier. A surplus of inflow is a withdrawal that leaves at the node's own value; a deficit
is an injection of carrier with no content of its own, which the outflow carries away, and
whatever content it brings is the caller's `sources`, never a negative removal. The water
application's quality layer uses this mode. The sink goes through `torch.where`, so at
`net(q) = 0` its derivative is the withdrawal side's. The default `"signed"` mode is unchanged.

**Storage-free nodes.** `storage_free=[...]` names interior nodes that hold no amount, such as a
zero-volume air-handler plenum. Their balance is algebraic in every step: what flows in flows out
within the step, so their value is the flow-weighted mix of what enters. A storage-free node with
nothing passing through it (no flow and no conduction) keeps its value, as ContamX holds the
plenum of a switched-off air handler. A source at such a node (a heating coil in a plenum) is
carried on by the flow through it within the step. At a node nothing passes through the source
has nowhere to go, and the step drops it (no airflow through a plenum: its coil delivers
nothing); `step`'s `diagnostics` reports the number of nonzero source entries dropped under
`"held_sources_dropped"` (in `Model.step`, at `diagnostics["layers"][name]`). Their capacity, at construction or per step, may be 0
(a negative one is refused everywhere, and a zero one at any other node). Removal and kinetics act
on a stored amount, so they do not act on these nodes. Only `scheme="backward_euler"` and `steady` can
hold a storage-free node that flow passes through: the exponential and trapezoidal steps have no
algebraic form, so a layer with either scheme refuses such a node at construction (a node joined
only by conduction is folded instead, see below). A `steady` solve has no value for such a node
with nothing passing through it, and refuses it, naming the node. Whether such a node mixes or holds is a 0/1 choice on the flows and
carries no gradient: a held node's value has zero gradient with respect to the flows, although
any flow through it would switch it to mixing. `layer.storage_free_mask` is the `(n_i,)` boolean mask of these nodes.

**Surface and star nodes: folding a storage-free node into its neighbours.** A storage-free node
joined to the rest of the network only by conduction (the inside surface of a wall, between the
zone air and the first wall layer; a zone's radiant "star" node, joined to each of its surfaces)
stores no heat, so whatever conducts in conducts out. Its temperature is the conductance-weighted
mean of its neighbours' temperatures, plus its own source (absorbed sunlight, the radiant part of
a gain) divided by its total conductance. Putting that back into its neighbours' balances removes
the node: its neighbours are joined directly by *equivalent* conductances, and its source is
shared out among them. This is the "combined surface coefficient" of building physics (a surface
film of conductance $h A$ in series with the wall half behind it, $k A$, is one conductance
$1/(1/(hA) + 1/(kA))$ between the air and the wall) and the star-delta transform of circuit
theory (a star node joined to $n$ surfaces becomes conductances between every pair of them).
Several such nodes joined to each other, the surfaces of a zone with their star node, are folded
together as one group (in linear-algebra terms, the Schur complement or Kron reduction of their
block of the conduction matrix).

The layer does this by itself: a storage-free node with no flow edge, and no removal or kinetics
term at construction, is folded. The folded network has positive conductances again, so every
scheme, `exact` included, steps it, and positivity and the maximum principle hold as before (no
new temperature extremes from the folding). After the step the folded nodes' temperatures are
recovered from their neighbours' new values, the boundary values at the end of the step and the
sources, and returned in the state like any other node's; the boundary transfer counts the heat
that leaves through them. The folding is computed once from the construction-time conductance
(again after an in-place edit of it, and on every call while it requires grad), and on each call
that passes a per-call conductance (the driver below), so a radiative
coefficient re-evaluated each pass is folded with its current value, and gradients reach the
conductances and the sources through it. A folded node all of whose conductances are zero in a
call is held at its value, its source dropped, as above; a group of folded nodes joined that call
only to each other (every conductance to a stored node or boundary zero) has no temperature and is
refused, naming the nodes. Checking for such a group searches the group's edges on the host, so it
costs a little on every call in which some conductance touching a group of two or more folded
nodes is zero (a radiative coefficient that vanishes at night); with every such conductance
positive it costs nothing. Each group is factored as a small dense matrix, batched over the groups
of one size, never as a matrix over the whole network; a group is as large as the set of folded
nodes joined to each other, usually the surfaces of one zone. A layer whose every interior node is
folded (a heat network without any heat capacity) has nothing to step: its step and its steady
state are the folded nodes' balance at the end of the step; under `on_failure="return"` they
return a `SolveResult` that has converged in zero iterations. `layer.steady(..., hold=x0)` holds
the stored nodes at `x0` and solves only the folded ones: the consistent initial point
`Model.steady(..., mode="initial")` uses. A storage-free node that air flows
through (a plenum), or that has a removal or reaction at construction, is not folded and still
needs `scheme="backward_euler"`; a per-call removal (the `"<layer>.removal"` driver) that is nonzero at a
folded node is refused, naming the node. `layer.folded_mask` is the `(n_i,)` boolean mask of the folded nodes. `rate`
and the dense `operator` still describe the whole network, a folded node as its algebraic balance.

**A conductance that changes with the state.** A radiative coefficient depends on the surface
temperatures, a film coefficient on the air speed. A closure writes such a value into the driver
`"<layer>.conductance"`, shape `(..., b_c)` in the order of `layer.conduction_edges`, and it
*replaces* the construction-time `conductance` for that call, under all three schemes and
`steady`. Gradients flow from the solution back to the closure that computed it. Entries must be
zero or positive (a negative one is refused, naming the edge); zero is allowed because a
coefficient can vanish. A stored node whose conducting edges are all zero this call, and with no
flow through it, is isolated for that call: a transient step leaves it at its value plus its own
source, and a steady solve refuses it as an undefined steady state (or, for a storage-free
node, as one nothing passes through). With `Model(coupling="iterate")` the closure reads the
previous pass's temperatures and the passes converge to the steady state of the nonlinear
balance, for instance radiation to a boundary at the fourth power of the temperature. The library
carries no radiation law: the closure owns it.

**Undefined steady states.** A stored node that nothing leaves (no outflow, no conduction, no
removal) keeps everything that enters it, and so does a closed group of stored nodes whose
contents only circulate among themselves, never reaching a boundary outflow, a removal, a filter
loss or a reaction. Their steady state is undefined without a source and unbounded with one, and
the operator is singular. `steady` checks for this on every batch instance before it solves and
raises `UndefinedSteadyState` (a `ValueError`), naming the nodes; its `layer` and `nodes`
attributes say which. Step such a model through time instead, or give the nodes an exchange.

**Linear solver.** The `"backward_euler"` and `"trapezoidal"` schemes each need one linear solve per
step (`"exact"` has none). `linear_solver` names it: `"auto"` (the default), `"gmres"`,
`"gmres_jacobi"`, `"gmres_ilu"`, `"sparse_direct"` or `"direct"`, resolved fresh on every solve
(unlike `KirchhoffLayer.linear_solver`, which is fixed for the layer's lifetime). `"auto"`
picks `"sparse_direct"` (SciPy's sparse LU, one exact factor-and-solve, no outer iteration) at or
under a batch cap of 32 instances, and plain `"gmres"` above it — chosen on an interleaved
forward/backward benchmark across the whole family (`gmres`, `gmres_jacobi`, `gmres_ilu`,
`sparse_direct`), not on iteration counts alone.
`"auto"` never picks
`gmres_jacobi`/`gmres_ilu`: the benchmark found `gmres_ilu`'s per-instance ILU factorisation cost
(paid twice per differentiable step — once in the forward solve, once in the backward adjoint,
since preconditioning is not reused across the two) the clear loser at every batch size measured,
and the gmres-vs-`gmres_jacobi` ordering too close to the run-to-run spread to trust as a
tie-break. Two fall-backs, both silent (never a raise, since `"auto"` is a promise to choose a
backend that works): SciPy absent (`"auto"` would have factorised, but the optional `noodl-physics[sparse]`
extra is not installed) falls back to `gmres` and warns once per process; a grad-requiring solve
on the `on_failure="return"` early-return path (which runs outside the differentiable path's own
`no_grad`, unlike `step`'s ordinary forward/backward) falls back to `gmres` silently, since an
explicit `sparse_direct` there would otherwise raise for a reason the caller did not ask for.
`diagnostics["linear"]["backend"]` reports which backend actually ran on every solve.

## `Reaction`

Applied to a transport layer's state after its step (by default), operator-split. A reaction
is any object with `apply(x, dt, drivers)` returning the new state:

```py
from noodl.layers.reaction import Reaction

class Source(Reaction):
    def apply(self, x, dt, drivers=None):
        return x + dt * drivers["generation"]

model = Model(net, layers, reactions=[("species", Source())])
```

The core ships only this protocol. The applications bring their own reactions: the street
application `Photostationary` (Leighton NO/NO₂/O₃, `noodl.apps.street_aq`), the sewer
application `SulfideGeneration`. `Photostationary`'s default rate is evaluated at the driver
`"temperature"` (K), which it requires by name unless a constant `k_no_o3` rate overrides it.

**First-order decay is not a reaction.** A species that decays as $\mathrm{d}x/\mathrm{d}t = -k x$
is the transport layer's `removal` (a rate per node, 1/s), which is solved together with the
transport in the same step rather than split from it.

`Model(reaction_order="before_transport")` applies the reactions to the step-start state
before the transport step instead, so the returned state is the one after transport. The
sequence of operations is the same; only the point where the state is sampled moves. The sewer
application uses it because SWMM reports its water quality after mixing.

**`Model.steady` never applies reactions.** This is deliberate — a steady state under a reaction
is a different fixed-point problem, not the transport steady state. The street and sewer
applications each provide their own `street_steady` / `sewer_steady` helper that iterates
transport and chemistry together to a joint fixed point. If you have a reaction and you call
`Model.steady`, you get the transport-only answer.

## `AllocatedFlowLayer`

The fourth mode. Each edge carries a *requested* flow; the layer clips it against arc capacity
and the receiver's remaining storage headroom, sharing proportionally when several edges compete
for one node's headroom.

```py
from noodl.layers.allocation import AllocatedFlowLayer

layer = AllocatedFlowLayer(
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
modes differ in gradient behaviour, which is the whole reason there is more than one — see
[Flow allocation](../applications/water-resources/allocation.md) for the full treatment.

An allocation layer is inherently discrete-time. A `Model` owning one refuses a steady pass and
refuses `residuals()` outright, because a clip-and-allocate rule has no steady meaning to report.
