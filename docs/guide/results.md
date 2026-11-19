# Reading results by name

Results come back in the same dictionaries you gave: `state` after a step, and for a march a
dictionary of stacked states. A tensor is in the order of its layer, and reading by position is
how mistakes happen, so read by name.

The examples on this page use the two-room model of [the workflow](workflow.md#a-complete-example),
built and stepped six times:

```python
import torch

from noodl.apps.building_physics import (
    Zone, add_zone, build_model, initial_drivers, initial_state,
)
from noodl.apps.building_physics.elements import add_large_opening, orifice_law_from_edges
from noodl.heads import Stack
from noodl.topology import Network

net = Network(dtype=torch.float64)
net.add_node("ambient", z_ref=0.0, T0=283.15)
add_zone(net, Zone("A", volume=60.0, T0=288.15))
add_zone(net, Zone("B", volume=60.0, T0=285.15))
add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)
model = build_model(
    net, air_laws=[orifice_law_from_edges(net, "airpath")],
    heads=[Stack.from_network(net, "airpath")], species=("co2",),
    coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50,
)
th, co2 = model.refs.thermal, model.refs.species
state = initial_state(model, values={co2.mass_fraction: {"A": 6e-4, "B": 6e-4}})
drivers = initial_drivers(model, values={
    th.boundary_temperature: {"ambient": 283.15},
    th.sources: {"A": 1000.0},
    co2.boundary_mass_fraction: {"ambient": 6e-4},
})
for _ in range(6):
    state = model.step(state, drivers, dt=600.0)
```

## Names for keys

`model.refs` names the keys of every layer. Each layer's reference has one attribute per key,
named for the physics it carries (`.pressure`, `.temperature`, `.mass_fraction`, `.flow`,
`.sources`, ...). The names come from each layer's `quantity`, so every application has them, and
the solver's short spellings (`.x`, `.phi`, `.s`, `.q`) work too. An attribute is the plain key
string (`th.temperature == "thermal.x"`), so it indexes the dictionaries directly, and it also
knows its layout:

```python
print(th.sources.ordering, th.sources.labels)
print(th.temperature.ordering, th.temperature.labels, th.temperature.unit)
```

```text
full node order ('ambient', 'A', 'B')
active interior order ('A', 'B') K
```

`model.refs.describe()` lists every key of every layer with what it holds, its order, its shape and
its unit. The suffixes and orders are tabulated in the [keys reference](../reference/keys.md).

## Reading a tensor by label

`field.named(tensor)` reads a tensor back as `{label: value}`:

```python
flows = model.refs.air.flow.named(state[model.refs.air.flow])
print(sorted(flows)[:2])
```

```text
[('A', 'B', 0), ('A', 'B', 1)]
```

Nodes are labelled by name. Edges are labelled by their `name` attribute when they have one (EPANET
links, SWMM conduits, WSIMOD arcs, street names), otherwise by `(source, target)`, or `(source,
target, key)` where two edges join the same nodes. Units are those reported by `describe()`: the
layer's own for state, and the application's flow, source and capacity units for the rest.

## Where the pieces came from

A model assembled from [components](components.md) keeps a `NameMap` from every node and edge back
to its component, so results can be read per room or per door.
