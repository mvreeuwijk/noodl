# Calibrate against data

This tutorial fits a model parameter to measurements, using gradients from the model and
`torch.optim`. The measurements are synthetic: we produce them with the model itself at a known
true value, so we can check that the fit recovers it.

The model is the two-loop EPANET network. Its file says junction J4 draws 10 L/s. Suppose the
true demand there is 16 L/s, and we have heads measured at J5 and J6.

## The forward model

```python
import torch
from noodl.apps.water import read_epanet_inp, build_model, water_steady, initial_drivers

net = read_epanet_inp("twoloop_si.inp")
model, state, _ = build_model(net)
water = model.refs.water


def heads_at(demand_j4, nodes):
    """Heads (m) at `nodes` for a demand at J4 given in L/s (a tensor, so it is differentiable)."""
    drivers = initial_drivers(model, demand={"J4": demand_j4 / 1000})
    final = water_steady(model, state, drivers)
    heads = water.head.named(final[water.head])
    return torch.stack([heads[n] for n in nodes])
```

## Synthetic measurements

```python
nodes = ["J5", "J6"]
measured = heads_at(torch.tensor(16.0, dtype=torch.float64), nodes)
print("measured heads (m):", [round(h, 3) for h in measured.tolist()])
```

```text
measured heads (m): [47.537, 47.573]
```

## Fit by gradient descent

Start from the file's 10 L/s. Each evaluation of the loss solves the network, and `backward()`
differentiates through that solve.

```python
demand = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
optimiser = torch.optim.LBFGS([demand], lr=1.0, line_search_fn="strong_wolfe")


def closure():
    optimiser.zero_grad()
    loss = ((heads_at(demand, nodes) - measured) ** 2).sum()
    loss.backward()
    return loss


for step in range(3):
    loss = optimiser.step(closure)
    print(f"step {step}: loss at start {loss.item():.4f}, demand {demand.item():.2f} L/s")
print("recovered demand at J4 (L/s):", round(demand.item(), 2))
```

```text
step 0: loss at start 0.6396, demand 16.00 L/s
step 1: loss at start 0.0000, demand 16.00 L/s
step 2: loss at start 0.0000, demand 16.00 L/s
recovered demand at J4 (L/s): 16.0
```

The optimiser recovers the true value. Here the parameter is a demand, which is a driver. The
roughness of a pipe read from an EPANET file is not learnable through `build_model`; see the
[water limitations](../applications/water/limitations.md). The same recipe fits several parameters at once and any
differentiable parameter of any application; see [Gradients](../guide/gradients.md) for what can
be differentiated and what to watch for, and the [examples](../examples/index.md) for fits to
more realistic data.
