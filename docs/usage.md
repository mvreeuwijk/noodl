# Using noodl

Every model in noodl physics is used the same way, whether it is a building, a street
network, a sewer, a water network or an allocation network. This page is that common
pattern. The [catalogue](catalogue/core.md) lists everything each application offers, and
each [application page](applications/index.md) explains its physics.

## The four steps

1. **Configure.** Build the network and its layers into a `Model`. Each application's
   `build_model` does this from its own description (rooms and openings, streets, pipes, a
   file read from another tool).
2. **Initialise, by name.** `initial_state(model, values=...)` and
   `initial_drivers(model, values=...)` give the starting state and the inputs. Values are
   given per node (or per pipe, street, arc) by name; anything not named keeps the
   application's default.
3. **Check.** `model.check(state, drivers).raise_for_errors(strict=True)` stops on anything
   missing, misspelt, of the wrong shape or in the wrong place *before* the run.
4. **Run.** `model.step(state, drivers, dt)` advances one step; `model.steady(state,
   drivers)` solves the quasi-steady state. Results come back in the same dictionaries and
   are read by name.

```py
state   = initial_state(model, values={...})      # starting state, by name
drivers = initial_drivers(model, values={...})    # inputs, by name
model.check(state, drivers).raise_for_errors(strict=True)
for t in times:
    state = model.step(state, drivers, dt)        # or model.steady(state, drivers)
```

## The same calls in every application

| Application | Configure | Layers (`model.refs.<name>`) and their keys | Also give |
|---|---|---|---|
| [Building physics](applications/building_physics.md) | `model, state, drivers = build_model(net, ..., return_inputs=True)` (or just `model = build_model(net, ...)`) | `air`: `pressure`, `flow`; `thermal`: `temperature`; `species`: `mass_fraction` | nothing else (`P_ref` optional) |
| CONTAM `.prj` | `model, state, drivers = project_to_model(read_prj(path))` | `air`: `pressure`, `flow`; `species`: `mass_fraction` | wind and densities are in the returned drivers |
| [Street air quality](applications/street_aq.md) | `model, state, drivers = build_model(streets, ...)` | `street`: `concentration`, `flow` | the meteorology: `U_ref`, `theta_w`, `h_abl` (and `lmo`) |
| [Sewers (SWMM)](applications/sewer.md) | `model, state, drivers = build_model(sewer_net, ...)` | `air`: `pressure`; `water_quality`, `air_quality`: `concentration` | inflows and temperatures are in the returned drivers |
| [Water (EPANET)](applications/water.md) | `model, state, drivers = build_model(water_net, ...)` | `water`: `head`, `flow`; `quality`: `concentration` | tank levels and pump status are state, by tank and pump name |
| [Allocation (WSIMOD)](applications/allocation.md) | `model, state, drivers = build_model(topology)` | `wsimod`: `storage`, `requests`, `flow` | nothing else |

Every application exports `build_model`, `initial_state` and `initial_drivers`, and
`initial_state(model)`/`initial_drivers(model)` without `values` return its defaults.
(The sewer's `initial_state` also takes the drivers, `initial_state(model, drivers=drivers,
values=...)`, because its storage depends on them.)

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

# 3. Check, then 4. run
model.check(state, drivers).raise_for_errors(strict=True)   # warnings fail too
for _ in range(6):
    state = model.step(state, drivers, dt=600.0)
T = th.temperature.named(state[th.temperature])
print({room: round(float(value), 2) for room, value in T.items()})
```

```text
{'A': 304.13, 'B': 300.93}
```

## What every model is made of

- **A network.** Nodes (rooms, streets, manholes, junctions, stores) and directed edges,
  each edge with a `kind` (`"airpath"`, `"pipe"`, ...). The kind decides which layer and
  which law an edge belongs to.
- **Layers.** Each conserves one quantity on the network. A *potential* layer solves for a
  potential (pressure, head) and the flows it drives; a *transport* layer carries a
  quantity (temperature, mass fraction, concentration) on those flows; an *allocation*
  layer moves requested amounts between stores within capacity limits.
- **Elements and drives.** The law on each kind of edge (an orifice, a pipe, a pump) and
  the terms added to its potential difference (stack effect, wind, fans).
- **Closures.** Functions of the state that compute inputs each step: the air density from
  the temperature, the street flows from the wind, the sewer flows from the inflows. Their
  own inputs (a wind speed, an inflow) are ordinary drivers.
- **A model.** The layers and closures on one network, stepped together.
- **Two dictionaries.** `state` (what the model carries from step to step) and `drivers`
  (what you prescribe), each `{key: tensor}`. Every key is `"<layer>.<suffix>"`.

Large networks are easier to assemble from components: pieces of network with ports
(a room, a door, a stairwell) that you place, connect and nest in plain Python, and that
`flatten()` turns into the one network above, with a map from every node and edge back to
its component. The model is built and run exactly as before; see
[Components](concepts/components.md).

`model.refs` names those keys for you. Each layer's reference has one attribute per key,
named for the physics it carries:

| Attribute | Key | Role | Order of the tensor |
|---|---|---|---|
| `.pressure`, `.head` (potential layers) | `"<layer>.phi"` | state | full node order |
| `.flow` | `"<layer>.q"` | state (driver for a transport layer whose flows nothing solves) | edge order of the layer's kinds |
| `.boundary_pressure`, `.boundary_head` | `"<layer>.phi_boundary"` | driver | the layer's boundary order |
| `.temperature`, `.mass_fraction`, `.concentration` | `"<layer>.x"` | state | the layer's active interior order |
| `.boundary_temperature`, ... | `"<layer>.x_boundary"` | driver | the layer's boundary order |
| `.sources` | `"<layer>.sources"` | driver | full node order |
| `.capacity` | `"<layer>.capacity"` | driver | the layer's active interior order |
| `.storage` (allocation layers) | `"<layer>.s"` | state | full node order |
| `.requests` (allocation layers) | `"<layer>.requests"` | driver | edge order of the layer's kind |

Every key reports its unit: the layer's own for state and boundary values, and the
application's `flow_unit`, `source_unit` and `capacity_unit` for flows, sources and
capacity, so `describe()` shows `thermal.sources` in W and `air.q` in kg/s.

The names come from each layer's `quantity`, so every application has them without code
of its own; the solver's short spellings (`.x`, `.phi`, `.s`, `.q`) work too. Each
attribute is the plain key string (`th.temperature == "thermal.x"`), so it indexes the
dictionaries directly, and it also knows its layout:

```python
print(th.sources.ordering, th.sources.labels)
print(th.temperature.ordering, th.temperature.labels, th.temperature.unit)
```

```text
full node order ('ambient', 'A', 'B')
active interior order ('A', 'B') K
```

`model.refs.describe()` lists every key of every layer with what it holds, its order, its
shape and its unit.

## Running and reading results

`model.step(state, drivers, dt)` returns the new state; call it in a loop, changing the
drivers between steps as the inputs change. `model.steady(state, drivers)` returns the
quasi-steady state instead (an allocation layer, which is a time-stepping rule, has none).

When the layers feed back on each other — airflow carries heat, and temperature drives the
airflow — build the model with `coupling="iterate"` and an `iterate_tol`, which repeats
each step until they agree. The default, `coupling="pingpong"`, takes one pass per step;
[Layers and models](concepts/layers.md) explains the difference.

Read any result by name with `named`:

```python
flows = model.refs.air.flow.named(state[model.refs.air.flow])
print(sorted(flows)[:2])
```

```text
[('A', 'B', 0), ('A', 'B', 1)]
```

Edges are labelled by their `name` attribute when they have one (EPANET links, SWMM
conduits, WSIMOD arcs, street names), otherwise by `(source, target)`, or `(source,
target, key)` where two edges join the same nodes.

## Batches and gradients

A value may be a tensor instead of a number. Its shape is the batch shape, and every
value broadcasts against the others, so one call builds a whole ensemble; values that
require a gradient keep it.

```python
power = torch.tensor([500.0, 1000.0, 1500.0], requires_grad=True)
batch = initial_drivers(model, values={
    th.boundary_temperature: {"ambient": 283.15},
    th.sources: {"A": power},
    co2.boundary_mass_fraction: {"ambient": 6e-4},
})
out = model.step(initial_state(model), batch, dt=600.0)
out[th.temperature][:, 0].sum().backward()     # d(T_A)/d(power), all three instances
print(tuple(batch[th.sources].shape), power.grad.shape)
```

```text
(3, 3) torch.Size([3])
```

## Orders, and why naming matters

A tensor of the right size in the wrong order is accepted without complaint by the
solvers, and four orders are in use (the table above). A wall-mass node, for example, is
in the thermal layer's interior but not in a species layer's, so `"thermal.x"` and
`"species.x"` index rooms differently on the same network. Building by name removes the
question:

- `field.build({label: value})` assembles the tensor in that field's order. A label the
  network does not have raises `KeyError` with suggestions; a label the field may not set
  (a source on a boundary or inactive node) raises `ValueError`. Labels not named take the
  field's default (zero for sources, flows and requests) or, over a `base`, keep the base
  value; boundary values and state have no default and must be complete.
- A multi-species layer takes `{node: {species: value}}` with its species' names.
- The result is built with `torch.stack`, so gradients, dtype and device are kept.
- `field.named(tensor)` reads a tensor back as `{label: value}`.

## Checking before a run

`model.check(state, drivers)` returns a report; nothing is raised until you call
`.raise_for_errors()` (or `.raise_for_errors(strict=True)` to fail on warnings too). It
reports:

- **Errors**: a required input or state missing, a value that is not a tensor, a trailing
  shape that does not match the key's order, batch shapes that do not broadcast, a nonzero
  source on a boundary or inactive node (which a potential layer would otherwise drop
  silently), a state key put in the drivers, a closure output of the wrong shape.
- **Warnings**: a key with a layer's prefix but no such input (`"thermal.source"` — *did
  you mean `"thermal.sources"`?*), a near-miss of a layer name (`"therml.sources"`), an
  edge kind no layer uses that is a near-miss of one a layer does use (`"airpth"`), a
  temperature below 150 K (a Celsius value where kelvin is expected).
- A table of every key the model knows: role, required or optional, order, expected
  shape, unit, and what was given.

The inputs the closures, reactions, elements and drives read themselves are declared too,
in `model.refs.inputs`: the street model's wind speed, direction and boundary-layer depth
(one value per instance, or one per street with `meteo="per_street"`), the sewer's inflow
per manhole and its temperatures, CONTAM's node densities and wind, the chemistry's
photolysis rate. Each knows its unit and layout, so the check reports a missing required
one, or one of the wrong shape, before the run, and `model.refs.describe()` lists them:

```python
print(model.refs.inputs["P_ref"].describe())
```

```text
'P_ref' (driver, optional): reference pressure of the ideal-gas density (default 101325 Pa)
  one value per instance
  quantity: ? [Pa]
```

A key the model knows nothing about — a custom closure's own input — is **not** rejected.
`model.check(state, drivers, dt=600.0, probe=True)` runs one step on copies, recording
which keys the closures, layers, elements and drives read, and reports any key nothing
read. Without a probe the check evaluates the closures once (no solve, no time step) to
learn what they write, so an input a closure provides is not reported missing. A custom
closure can declare its own inputs with an `input_specs` attribute,
`{key: {"description", "unit", "over", "required"}}` (see `noodl.refs.input_field`).

```python
drivers_typo = dict(drivers)
drivers_typo["thermal.source"] = drivers_typo.pop("thermal.sources")
for issue in model.check(state, drivers_typo).issues:
    print(issue)
```

```text
[warning] unknown-layer-key: 'thermal.source' is not a driver key of layer 'thermal' (its driver keys: ['thermal.capacity', 'thermal.sources', 'thermal.x_boundary']); did you mean 'thermal.sources'?
```

## Finding what is available

The [catalogue](catalogue/core.md) lists, for the core library and for every application,
each type it offers — builders, network components, elements, drives, closures,
reactions, sources, file readers and writers, functions and model options — with what it
does and how to call it. It is generated from the code, so it is always complete.

## One identity per layer

Every key is `"<layer>.<suffix>"`, and a `Model` registers each layer under its own `.name`.
It refuses a layer registered under any other name: the model and the layer's helpers (the
density closure reads `"<thermal layer>.x"`, an allocation layer reads
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

## Writing the dictionaries by hand

The dictionaries are ordinary Python, and code that builds them directly keeps working:

```python
sources = torch.zeros(net.n, dtype=torch.float64)
sources[net.node_index("A")] = 1000.0          # full node order, by position
old = {
    "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
    "thermal.x_boundary": torch.tensor([283.15], dtype=torch.float64),
    "thermal.sources": sources,
}
```

The same tensors, to the last bit, by name:

```python
new = initial_drivers(model, values={"thermal.sources": {"A": 1000.0},
                                     "thermal.x_boundary": {"ambient": 283.15}})
assert all(torch.equal(new[k], old[k]) for k in old)
```

Keys may be written as strings or as `model.refs` attributes. `model.drivers_from(values,
base=...)` and `model.state_from(values, base=...)` do the same for any model, including
one assembled by hand from layers; `state_from` also accepts a layer's name for its state
key (`{"thermal": {...}}`) and names any state a step needs that is still missing.
