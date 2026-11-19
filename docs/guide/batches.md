# Batches and ensembles

A value may be a tensor instead of a number. Its shape is the batch shape, and every value
broadcasts against the others, so one call builds a whole ensemble: a sweep of source strengths,
a set of weather cases, the draws of a Monte Carlo study. Values that require a gradient keep it.

The examples use the two-room model of [the workflow](workflow.md#a-complete-example):

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

## Building a batch

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

The leading dimension is the batch; the trailing dimensions are the key's order (here the three
nodes). Batch shapes of different keys must broadcast: `model.check` reports a mismatch before
the run.

## Building tensors by name

A tensor of the right size in the wrong order is accepted by the solvers without complaint, so
build it by name:

- `field.build({label: value})` assembles the tensor in that field's order. A label the network
  does not have raises `KeyError` with suggestions; a label the field may not set (a source on a
  boundary or inactive node) raises `ValueError`. Labels not named take the field's default (zero
  for sources, flows and requests) or, over a `base`, keep the base value. Boundary values and
  state have no default and must be complete.
- A multi-species layer takes `{node: {species: value}}` with its species' names.
- The result is built with `torch.stack`, so gradients, dtype and device are kept.

## Convergence in a batch

Convergence is decided per instance. A batch is converged only when every instance is, so a batch
spanning a wide range of conditions takes as many passes as its slowest instance. With
`coupling="iterate"` a failure raises naming the instances that failed.

For gradients through a batch, see [Gradients and optimisation](gradients.md).
