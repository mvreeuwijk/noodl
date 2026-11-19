# State and driver keys

The keys of a model's `state` and `drivers` dictionaries, and the order of each tensor. How to use
them is in [The workflow](../guide/workflow.md) and [Reading results by name](../guide/results.md).

## Key suffixes

Every key is `"<layer>.<suffix>"`, and a `Model` registers each layer under its own `.name`, so a
layer's keys never clash with another's. `model.refs.<layer>` has one attribute per key, named for
the physics the layer carries; each attribute is the plain key string.

| Attribute | Key | Role | Order of the tensor |
|---|---|---|---|
| `.pressure`, `.head` (potential layers) | `"<layer>.phi"` | state | full node order |
| `.flow` | `"<layer>.q"` | state (driver for a transport layer whose flows nothing solves) | edge order of the layer's kinds |
| `.boundary_pressure`, `.boundary_head` | `"<layer>.phi_boundary"` | driver | the layer's boundary order, without the storage nodes a forward-Euler layer holds (`layer.prescribed`) |
| `.temperature`, `.mass_fraction`, `.concentration` | `"<layer>.x"` | state | the layer's active interior order |
| `.boundary_temperature`, ... | `"<layer>.x_boundary"` | driver | the layer's boundary order |
| `.sources` | `"<layer>.sources"` | driver | full node order |
| `.capacity` | `"<layer>.capacity"` | driver | the layer's active interior order |
| `.removal` | `"<layer>.removal"` | driver | the layer's active interior order, `(n_i, K)` |
| `.transmission` | `"<layer>.transmission"` | driver | edge order of the layer's flow kinds, `(b_flow, K)` |
| `.storage` (allocation layers) | `"<layer>.s"` | state | full node order |
| `.requests` (allocation layers) | `"<layer>.requests"` | driver | edge order of the layer's kind |

The solver's short spellings (`.x`, `.phi`, `.s`, `.q`) work as attribute names too. The names come
from each layer's `quantity`, so every application has them. `model.refs.describe()` lists every key
of every layer with what it holds, its order, its shape and its unit.

## Orders

Four orders are in use:

- **full node order**: every node of the network, in insertion order;
- **active interior order**: the nodes the layer solves for. A wall-mass node, for example, is in
  the thermal layer's interior but not in a species layer's, so `"thermal.x"` and `"species.x"`
  index rooms differently on the same network;
- **boundary order**: the nodes whose value you prescribe;
- **edge order of the layer's kinds**: the edges of the kinds the layer uses.

A tensor of the right size in the wrong order is accepted by the solvers without complaint, so
build tensors by name (`field.build({label: value})`) and read them by name (`field.named(tensor)`).
Each attribute's `.ordering` and `.labels` give its order and the labels in it.

## Units

Every key reports a unit: the layer's own for state and boundary values, and the application's
`flow_unit`, `source_unit`, `capacity_unit` and `conductance_unit` for flows, sources, capacity and
conductances. `describe()` shows `thermal.sources` in W and `air.q` in kg/s, for example.

## Per-step removal and transmission

`"<layer>.removal"` is a per-step first-order removal rate (1/s, active interior order, `(n_i, K)`)
that replaces the layer's construction-time `removal`. A closure that owns sinks writes it. It
matters for species layers, where it removes a species from the air (a deposition); the layer's
`dilution`, if it has one, adds its own rate to it. The driver is not carried in the state: a
closure that sets it must write it every step.

`"<layer>.transmission"` is likewise a per-step pass fraction of each flow edge (1 minus a filter's
efficiency), replacing the layer's construction-time `transmission`. A closure whose filters depend
on the step's flows writes it (the building application's `FilterTransmission`, for UVGI filters).
It reads the flows of the previous pass, so such a model iterates each step
(`coupling="iterate"`). A closure that needs this declares `requires_coupling = "iterate"` and
`reads_flows_of = "<potential layer>"`, and `Model` refuses it unless that layer is in
`iterate_tol` and `iterate_flow_tol`.

## Inputs that closures read

Inputs that closures, reactions, laws and head terms read themselves are declared in
`model.refs.inputs`, each with its unit and layout (a street model's wind speed, CONTAM's node
densities, a chemistry's photolysis rate). A custom closure declares its own with an `input_specs`
attribute, `{key: {"description", "unit", "over", "required"}}`.
