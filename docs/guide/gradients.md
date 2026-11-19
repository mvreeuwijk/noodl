# Gradients and optimisation

Every solve in noodl is differentiable, so the gradient of any result with respect to a driver, an
initial state or a learnable law parameter comes from one `.backward()`. This page shows how to
get a gradient, and how to use it with `torch.optim` to calibrate or design. How the gradient is
computed, and when it is zero, is explained in [Differentiability](../concepts/differentiability.md).

## Get a gradient

Give a value as a tensor with `requires_grad=True`, run the model, reduce the result to a scalar
and call `backward()`. This is the batch example of [Batches and ensembles](batches.md), where
one backward pass returns the derivative for every instance:

```py
power = torch.tensor([500.0, 1000.0, 1500.0], requires_grad=True)
...                                          # drivers built with th.sources: {"A": power}
out[th.temperature][:, 0].sum().backward()   # d(T_A)/d(power), all three instances
power.grad                                   # shape (3,)
```

Gradients reach:

- the drivers and the initial state (boundary values, sources, inflows, meteorology);
- every law parameter built with `learnable=True`, which is registered on the law so that
  `law.parameters()` lists it;
- across a [coupled model](coupling.md), the parameters of the other model.

Run forward-only work under `torch.no_grad()` or pass `differentiable=False` to a solve: it then
builds no graph.

## Calibrate with `torch.optim`

A learnable parameter, or a driver that you want to fit, is an ordinary tensor for an optimiser.
Build the drivers inside the loop from the current value of the tensor, run the model, compare
with a target and step. This fits the heat input to room A so that its temperature reaches a
target after one hour:

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
```

```python
target = 303.0
power = torch.tensor([800.0], dtype=torch.float64, requires_grad=True)
optimiser = torch.optim.Adam([power], lr=40.0)
losses = []
for _ in range(5):
    optimiser.zero_grad()
    drivers = initial_drivers(model, values={
        th.boundary_temperature: {"ambient": 283.15},
        th.sources: {"A": power},
        co2.boundary_mass_fraction: {"ambient": 6e-4},
    })
    state = initial_state(model, values={co2.mass_fraction: {"A": 6e-4, "B": 6e-4}})
    for _ in range(6):
        state = model.step(state, drivers, dt=600.0)
    loss = ((th.temperature.named(state[th.temperature])["A"] - target) ** 2).sum()
    loss.backward()
    optimiser.step()
    losses.append(loss.item())
print(losses[-1] < losses[0])
```

```text
True
```

For a law parameter the loop is the same: build the law with `learnable=True`, pass
`law.parameters()` to the optimiser, and the step updates the coefficient in place, so the next
step of the model uses it. Rebuild the state and drivers on each iteration, as above, rather
than reusing tensors that already carry a graph. Use `torch.optim.LBFGS` with a closure for
small, smooth problems; it needs fewer evaluations than Adam.

A worked calibration of a real network is in [Calibrate with gradients](../get-started/calibrate.md).

## Cost

- Memory is one solve's graph, not the number of Newton iterations: the backward pass applies the
  implicit-function theorem at the converged solution and costs one extra linear solve.
- A model with `coupling="iterate"` runs its passes to convergence without a graph, then
  re-runs the certified pass once on the graph. The gradient is that of the converged fixed
  point, not of the unrolled passes.
- A batch of instances costs far less than the same instances run one by one.

## When a gradient is zero

A zero gradient is usually structural, not a bug: a hard clip or capacity crossing in an
allocation layer (use its smooth or projection mode), a quantity that cannot reach the result,
or a flow of exactly zero in a sewer conduit. See
[Differentiability](../concepts/differentiability.md#where-gradients-are-zero-and-why).
