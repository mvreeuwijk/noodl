# Setting up a model

Every model in noodl physics, whatever it describes, is set up in the same three steps:

1. **Configure**: build the network, its layers and closures, and assemble them into a
   `Model`. The application builders (`build_model`) do this for you.
2. **Initialise**: give the starting state and the inputs (drivers) **by name** —
   `initial_state(model, values=...)` and `initial_drivers(model, values=...)`.
3. **Check, then run**: `model.check(state, drivers).raise_for_errors(strict=True)` stops on
   anything missing, misspelt, mis-shaped or in the wrong order *before* the expensive run
   (`strict=True` makes warnings, such as a misspelt optional source, fail too);
   `model.step` advances it.

The state and drivers remain plain `{key: tensor}` dictionaries, so existing code keeps
working. What this page adds is a way to build those dictionaries without knowing their
internal tensor orderings, and a way to check them.

## The same calls in every application

| Application | Configure | Initialise | Notes |
|---|---|---|---|
| Building physics | `model = build_model(net, ...)` | `initial_state(model, values=...)`, `initial_drivers(model, values=...)` | Temperatures from each node's `T0`; `species` may be a list of names |
| CONTAM `.prj` | `model, state, drivers = project_to_model(read_prj(path))` | as above, or `model.state_from`/`model.drivers_from` | Species named as in the file |
| Street air quality | `model, state, drivers = build_model(streets, species=(...))` | `initial_state`, `initial_drivers` | Meteorology keys (`U_ref`, ...) are plain tensors |
| Sewers (SWMM) | `model, state, drivers = build_model(sewer_net)` | `initial_state(model, drivers, values=...)`, `initial_drivers` | Manhole levels `"sewer.H"` by manhole name |
| Water (EPANET) | `model, state, drivers = build_model(water_net)` | `initial_state`, `initial_drivers` | Tank levels and pump status by tank/pump name |
| Allocation (WSIMOD) | `model, state, drivers = build_model(topology)` | `initial_state`, `initial_drivers` | Requests by arc name, storage by node name |

In every one of them `initial_state(model)` and `initial_drivers(model)` without `values`
return the application's defaults, and `values` changes only what it names.

## A complete example

Two rooms, one heated, a doorway between them and two openings to the outside.

```python
import torch

from noodl.apps.building_physics import (
    Zone, add_zone, build_model, initial_drivers, initial_state,
)
from noodl.apps.building_physics.elements import add_large_opening, orifice_elements_from_edges
from noodl.drives import Stack
from noodl.topology import Network

# 1. Configure
net = Network(dtype=torch.float64)
net.add_node("ambient", z_ref=0.0, T0=283.15)
add_zone(net, Zone("A", volume=60.0, T0=288.15))
add_zone(net, Zone("B", volume=60.0, T0=285.15))
add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)
model = build_model(
    net, air_elements=[orifice_elements_from_edges(net, "airpath")],
    drives=[Stack.from_network(net, "airpath")], species=("co2",),
    coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50,
)

# 2. Initialise, by name
th, co2 = model.refs.thermal, model.refs.species
state = initial_state(model, values={co2.mass_fraction: {"A": 6e-4, "B": 6e-4}})
drivers = initial_drivers(model, values={
    th.boundary_temperature: {"ambient": 283.15},
    th.sources: {"A": 1000.0},            # W into room A
    co2.boundary_mass_fraction: {"ambient": 6e-4},
})

# 3. Check, then run
model.check(state, drivers).raise_for_errors(strict=True)   # warnings fail too
for _ in range(6):
    state = model.step(state, drivers, dt=600.0)
T = th.temperature.named(state[th.temperature])
print({room: round(float(value), 2) for room, value in T.items()})
```

```text
{'A': 304.13, 'B': 300.93}
```

`model.refs` holds one reference per layer. Its attributes are the layer's keys, named
for the physics the layer carries -- `th.temperature`, `th.boundary_temperature`,
`th.sources`, `co2.mass_fraction`; `.pressure` or `.head` and `.flow` on an airflow or
water layer, `.concentration` on a street or sewer quality layer, `.storage` and
`.requests` on an allocation layer. The names come from each layer's `quantity`, so
every application has them. Each key is a plain string
(`th.temperature == "thermal.x"`, the solver's own spelling, which also works as
`th.x`) that also knows its layout:

```python
print(th.sources.ordering, th.sources.labels)
print(th.temperature.ordering, th.temperature.labels, th.temperature.unit)
```

```text
full node order ('ambient', 'A', 'B')
active interior order ('A', 'B') K
```

`model.refs.describe()` lists every key of every layer, what it holds, its order, its shape
and its unit.

## Orders, and why naming matters

Four orders are in use, and a tensor of the right size in the wrong order is accepted
without complaint by the solvers:

| Key | Order | Example |
|---|---|---|
| `"<layer>.sources"`, `"<layer>.phi"`, `"<layer>.s"` | full node order | `net.nodes` |
| `"<layer>.phi_boundary"`, `"<layer>.x_boundary"` | the layer's boundary order | as given to the layer |
| `"<layer>.x"`, `"<layer>.capacity"` | the layer's active interior order | nodes its edge kinds touch, minus the boundary |
| `"<layer>.q"`, `"<layer>.requests"` | edge order of the layer's kinds | `net.edges` of those kinds |

A wall-mass node, for example, is in the thermal layer's interior but not in a species
layer's, so `"thermal.x"` and `"species.x"` index rooms differently on the same network.
Building by name removes the question:

- `field.build({label: value})` assembles the tensor in that field's order. A label the
  network does not have raises `KeyError` with suggestions; a label the field may not set
  (a source on a boundary or inactive node) raises `ValueError`. Labels not named take the
  field's default (zero for sources, flows and requests) or, over a `base`, keep the base
  value; boundary values and state have no default and must be complete.
- Values may be numbers or tensors. A tensor's shape is the batch shape (followed by the
  species for a multi-species layer, or give `{species: value}`), batch shapes broadcast,
  and the result is built with `torch.stack`, so gradients, dtype and device are kept.
- `field.named(tensor)` reads a tensor back as `{label: value}`.
- Edges are labelled by their `name` attribute when they have one (EPANET links, SWMM
  conduits, WSIMOD arcs), otherwise by `(source, target)`.

## Checking before a run

`model.check(state, drivers)` returns a report; nothing is raised until you call
`.raise_for_errors()` (or `.raise_for_errors(strict=True)` to fail on warnings too). It
reports:

- **Errors**: a required input or state missing, a trailing shape that does not match the
  key's order, batch shapes that do not broadcast, a nonzero source on a boundary or
  inactive node (which a potential layer would otherwise drop silently), a state key put in
  the drivers, a driver `"<layer>.q"` for a layer whose flows a potential layer computes.
- **Warnings**: a key with a layer's prefix but no such input (`"thermal.source"` — *did
  you mean `"thermal.sources"`?*), a near-miss of a layer name (`"therml.sources"`), an edge
  kind no layer uses that is a near-miss of one a layer does use (`"airpth"`). An unused
  kind with no such near-miss is only noted (`info`): a closure may work on it, and may say
  so with `edge_kinds = (...)`.
- A table of every key the model knows: role, required or optional, order, expected
  shape, unit, what was given.

Keys the model does not know the layout of — a custom closure's inputs, a weather
variable — are **not** rejected. `model.check(state, drivers, dt=600.0, probe=True)` runs
one step on copies, under `torch.no_grad()`, recording which keys the closures, layers,
elements and drives actually read; a key nothing read is reported as unused. A closure may
also declare `inputs = (...)` and `outputs = (...)` (driver keys it reads and writes); when
every closure declares `outputs`, a missing required driver is an error even without a
probe. Without a probe the check also evaluates the closures once as a query (no solve,
no time step; the call `Model.initial_capacities` makes) to learn what they write, so a
flow or density a closure provides is not reported missing.

```python
drivers_typo = dict(drivers)
drivers_typo["thermal.source"] = drivers_typo.pop("thermal.sources")
for issue in model.check(state, drivers_typo).issues:
    print(issue)
```

```text
[warning] unknown-layer-key: 'thermal.source' is not a driver key of layer 'thermal' (its driver keys: ['thermal.capacity', 'thermal.sources', 'thermal.x_boundary']); did you mean 'thermal.sources'?
```

## One identity per layer

Every key is `"<layer>.<suffix>"`. A `Model` registers each layer under its own `.name`,
and refuses a layer registered under any other name: the model and the layer's helpers
(the density closure reads `"<thermal layer>.x"`, an allocation layer reads
`"<layer>.requests"`) would otherwise build its keys under two different names. Layers can
be passed as a list, keyed by their names:

```python
from noodl.model import Model

same = Model(net, [model.layers["air"], model.layers["thermal"]], closures=model.closures)
print(list(same.layers))
```

```text
['air', 'thermal']
```

## Migrating from string dictionaries

Existing code needs no change. Hand-built dictionaries:

```python
sources = torch.zeros(net.n, dtype=torch.float64)
sources[net.node_index("A")] = 1000.0          # full node order, by position
old = {
    "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
    "thermal.x_boundary": torch.tensor([283.15], dtype=torch.float64),
    "thermal.sources": sources,
}
```

become, with the same tensors to the last bit:

```python
new = initial_drivers(model, values={"thermal.sources": {"A": 1000.0},
                                     "thermal.x_boundary": {"ambient": 283.15}})
assert all(torch.equal(new[k], old[k]) for k in old)
```

The keys may be written as strings or as `model.refs` fields. `model.drivers_from(values,
base=...)` and `model.state_from(values, base=...)` do the same for any model, including
one assembled by hand from layers; `state_from` also accepts a layer's name for its state
key (`{"thermal": {...}}`) and names any state a step needs that is still missing.
