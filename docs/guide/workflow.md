# The workflow

Every model in noodl is used the same way, whether it is a building, a street network, a sewer,
a water network, a WSIMOD catchment or an electrical circuit. This page is that common pattern:
configure, initialise by name, check, run. Each application page explains its own physics, and
the [catalogue](../reference/catalogue/index.md) lists what each one offers.

## The four steps

1. **Configure.** Build the network and its layers into a `Model`. Each application's
   `build_model` does this from its own description (rooms and openings, streets, pipes, a
   file read from another tool).
2. **Initialise, by name.** `initial_state(model, values=...)` and
   `initial_drivers(model, values=...)` give the starting state and the inputs. Values are
   given per node (or per pipe, street, arc) by name; anything not named keeps the
   application's default.
3. **Check.** `model.check(state, drivers).raise_for_errors(strict=True)` stops on anything
   missing, misspelt, of the wrong shape or in the wrong place before the run.
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

State and drivers are plain dictionaries `{key: tensor}`. Every key is
`"<layer>.<suffix>"`, and `model.refs.<layer>.<name>` gives it by name; the
[keys reference](../reference/keys.md) lists the suffixes and the order of each tensor.

## A complete example

Two rooms, one heated, a doorway between them and two openings to the outside.

```python
import torch

from noodl.apps.building_physics import (
    Zone, add_zone, build_model, initial_drivers, initial_state,
)
from noodl.apps.building_physics.elements import add_large_opening, orifice_law_from_edges
from noodl.heads import Stack
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
    net, air_laws=[orifice_law_from_edges(net, "airpath")],
    heads=[Stack.from_network(net, "airpath")], species=("co2",),
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

The stack and wind pressures of the air edges are the driver `"air.edge_head"`, which the
model's `EdgeHeads` closure writes on every pass. `step`, `steady`, `check` and `simulate` run
the closures first; a direct `model.kirchhoff["air"].solve(...)` on the raw drivers has no
head, so apply the model's closures before it.

```text
{'A': 304.13, 'B': 300.93}
```

Build the network completely before constructing layers or models over it. A layer keeps the
edge indices and node count of construction time, so adding a node or edge afterwards makes it
raise `RuntimeError` instead of silently solving the old graph; construct the layers and model
again.

## The same calls in every application

| Application | Configure | Layers (`model.refs.<name>`) and their keys | Also give |
|---|---|---|---|
| [Building physics](../applications/building/index.md) | `model, state, drivers = build_model(net, ..., return_inputs=True)` (or just `model = build_model(net, ...)`) | `air`: `pressure`, `flow`; `thermal`: `temperature`; `species`: `mass_fraction` | nothing else (`P_ref` optional) |
| CONTAM `.prj` | `model, state, drivers = project_to_model(read_prj(path))` | `air`: `pressure`, `flow`; `species`: `mass_fraction` | wind and densities are in the returned drivers |
| [Street air quality](../applications/street/index.md) | `model, state, drivers = build_model(streets, ...)` | `street`: `concentration`, `flow` | the meteorology: `U_ref`, `theta_w`, `h_abl` (and `lmo`) |
| [Sewers (SWMM)](../applications/sewer/index.md) | `model, state, drivers = build_model(sewer_net, ...)`; a `routing="dynamic_wave"` network is run with `simulate_dynamic_wave(model, state, drivers, steps=...)` | `air`: `pressure`; `water_quality`, `air_quality`: `concentration`; dynamic wave: `water`: `head`, `flow` | inflows and temperatures are in the returned drivers |
| [Water (EPANET)](../applications/water/index.md) | `model, state, drivers = build_model(water_net, ...)` | `water`: `head`, `flow`; `quality`: `concentration` | tank levels and pump status are state, by tank and pump name |
| [Allocation](../applications/water-resources/allocation.md) | `model, state, drivers = build_model(topology)` | `wsimod`: `storage`, `requests`, `flow` | nothing else |
| [WSIMOD](../applications/water-resources/wsimod.md) | `model, state, drivers = build_model(read_wsimod_config(directory))` | `wsimod`: `storage`, `flow`; per-class drivers such as `wsimod.River.width` | forcing series are in the returned drivers; step with `t=` and `dt=86400.0` |
| [Electrical circuits](../applications/electrical/index.md) | `model, state, drivers = build_model(circuit, ...)` | `electrical`: `voltage`, `flow` (current) | a transient starts from `initial_state(model, drivers)`; per-device values from `device_values` |

Every application exports `build_model`, `initial_state` and `initial_drivers`, and
`initial_state(model)` and `initial_drivers(model)` without `values` return its defaults. Two
exceptions. The sewer's `initial_state` also takes the drivers, `initial_state(model,
drivers=drivers, values=...)`, because its storage depends on them. The electrical application
has no `initial_drivers`: a circuit has no state to set by name, so its
`initial_state(model, drivers)` solves the consistent initial point from the devices' own initial
values, and `build_model` returns every driver it needs.

## Check before you run

`model.check(state, drivers)` returns a report; nothing is raised until you call
`.raise_for_errors()`, or `.raise_for_errors(strict=True)` to fail on warnings too. It reports:

- **Errors**: a required input or state missing, a value that is not a tensor, a trailing
  shape that does not match the key's order, batch shapes that do not broadcast, a nonzero
  source on a boundary or inactive node (which a potential layer would otherwise drop
  silently), a state key put in the drivers, a closure output of the wrong shape.
- **Warnings**: a key with a layer's prefix but no such input (`"thermal.source"`: *did
  you mean `"thermal.sources"`?*), a near-miss of a layer name (`"therml.sources"`), an
  edge kind no layer uses that is a near-miss of one a layer does use (`"airpth"`), a
  temperature below 150 K (a Celsius value where kelvin is expected).
- A table of every key the model knows: role, required or optional, order, expected
  shape, unit, and what was given.

For example, a misspelt driver key:

```python
drivers_typo = dict(drivers)
drivers_typo["thermal.source"] = drivers_typo.pop("thermal.sources")
for issue in model.check(state, drivers_typo).issues:
    print(issue)
```

```text
[warning] unknown-layer-key: 'thermal.source' is not a driver key of layer 'thermal' (its driver keys: ['thermal.capacity', 'thermal.removal', 'thermal.sources', 'thermal.transmission', 'thermal.x_boundary']); did you mean 'thermal.sources'?
```

The inputs that closures, reactions, laws and head terms read themselves are declared too, in
`model.refs.inputs`: the street model's wind speed, direction and boundary-layer depth, the
sewer's inflow per manhole, CONTAM's node densities and wind, the chemistry's photolysis rate.
Each knows its unit and layout, so the check reports a missing required one, or one of the wrong
shape, before the run. `model.refs.describe()` lists every key and input.

```python
print(model.refs.inputs["P_ref"].describe())
```

```text
'P_ref' (driver, optional): reference pressure of the ideal-gas density (default 101325 Pa)
  one value per instance
  quantity: ? [Pa]
```

A key the model knows nothing about, such as a custom closure's own input, is not rejected.
`model.check(state, drivers, dt=600.0, probe=True)` runs one step on copies, records which keys
the closures, layers, laws and head terms read, and reports any key nothing read. A custom
closure can declare its own inputs with an `input_specs` attribute,
`{key: {"description", "unit", "over", "required"}}` (see `noodl.refs.input_field`). What each
message means is in [Troubleshooting](troubleshooting.md#check-messages).

## Writing the dictionaries by hand

The dictionaries are ordinary Python, so a script may build them directly. Keys may be strings
or `model.refs` attributes, and `model.drivers_from(values, base=...)` and
`model.state_from(values, base=...)` do what `initial_drivers` and `initial_state` do for a model
assembled by hand from layers.

## Where next

- [Time stepping and schedules](time-stepping.md): marching over a time grid, changing drivers.
- [Reading results by name](results.md).
- [Batches and ensembles](batches.md) and [Gradients and optimisation](gradients.md).
- [Assembling from components](components.md) for large networks.
