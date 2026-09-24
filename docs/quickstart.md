# Quick start

This page builds a working model in about fifteen lines, then shows the two things that make
noodl different from a conventional network solver: it batches, and it differentiates.

## A two-node network

The smallest interesting network: a room connected to the outside through a leak. Air is
injected into the room at a known rate; we want the pressure that develops and the flow that
escapes.

```python
import torch

from noodl.elements.powerlaw import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

net = Network(dtype=torch.float64)
net.add_node("ambient")
net.add_node("room")
net.add_edge("room", "ambient", kind="airpath")

leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65))
layer = PotentialFlowLayer(net, "air", [leak], boundary=["ambient"])

phi_boundary = torch.zeros(1, dtype=torch.float64)
# sources is indexed in full node order (ambient, room); 0.05 injected into "room"
sources = torch.tensor([0.0, 0.05], dtype=torch.float64)
phi, q = layer.solve(phi_boundary, sources=sources, differentiable=False)

print("pressures (Pa):", dict(zip(net.nodes, phi.tolist())))
print("flows:         ", q.tolist())
```

```text
pressures (Pa): {'ambient': 0.0, 'room': 11.894289986499988}
flows:          [0.04999999999555594]
```

Four things happened there, and they are the whole framework in miniature:

1. **`Network`** holds the graph. Every edge carries a `kind` — here `"airpath"` — which is how
   one network can hold several physical systems side by side.
2. **`PowerLaw`** is an *element*: a constitutive law relating the potential difference across
   an edge to the flow along it, $q = C \, \Delta p^{\,n}$. Elements are the replaceable part.
3. **`PotentialFlowLayer`** assembles nodal conservation from the elements and solves
   $A_I \, g(A^\top \phi) = s_I$ for the interior potentials by Newton's method. `boundary`
   names the nodes whose potential is prescribed rather than solved.
4. The flow out matches the flow in to solver tolerance — conservation is structural, not
   something the element had to be told about.

## Batching over instances

Every solve is batched. Leading dimensions are instances, and they cost far less than a loop:
one Newton iteration serves the whole ensemble.

```python
# five instances, each injecting a different amount into the room
sources = torch.zeros(5, 2, dtype=torch.float64)
sources[:, 1] = torch.linspace(0.01, 0.05, 5, dtype=torch.float64)
phi_boundary = torch.zeros(5, 1, dtype=torch.float64)

phi, q = layer.solve(phi_boundary, sources=sources, differentiable=False)
print(phi.shape, q.shape)
print("room pressures:", phi[:, 1].tolist())
```

```text
torch.Size([5, 2]) torch.Size([5, 1])
room pressures: [0.99999991, 2.90484593, 5.42041753, 8.43812956, 11.89428999]
```

The same mechanism carries a thousand building variants, a Monte Carlo sample over uncertain
leakage, or one network under a year of hourly weather. Nothing in the model changes; only the
leading dimension does.

## Gradients

Drop `differentiable=False` and the solve becomes differentiable. Gradients reach the boundary
potentials, the sources, every value in `drivers`, and any element parameter constructed with
`learnable=True`.

```python
C = torch.tensor(0.01, requires_grad=True)
leak = PowerLaw(C=C, n=torch.tensor(0.65), learnable=True)
layer = PotentialFlowLayer(net, "air", [leak], boundary=["ambient"])

phi, q = layer.solve(
    torch.zeros(1, dtype=torch.float64),
    sources=torch.tensor([0.0, 0.05], dtype=torch.float64),
)
phi[1].backward()

print("d(room pressure)/dC =", next(layer._elements[0].parameters()).grad)
```

```text
d(room pressure)/dC = tensor(-1829.8909)
```

That derivative did not come from finite differences or from unrolling the Newton iteration.
It came from the implicit-function theorem applied at the converged solution, which costs one
extra linear solve regardless of how many Newton steps the forward pass took. See
[Differentiability](concepts/differentiability.md) for what that buys and what it requires of
your elements.

## Several physics on one network

A real model has more than one layer. `Model` steps them together on one network:

```python
from noodl.model import Model

model = Model(
    net,
    layers={"air": air_layer, "thermal": thermal_layer},
    closures=[density_closure],
    coupling="iterate",
    iterate_tol={"thermal": 0.01},
)

state = model.step(state, drivers, dt=60.0)
```

Here the airflow depends on temperature (through air density and the resulting stack effect)
and the temperature depends on the airflow (which advects the heat). `coupling="iterate"`
repeats the pass within one step until the named states stop changing;
`coupling="pingpong"`, the default, takes a single pass. [Layers and models](concepts/layers.md)
covers both, and the trap in the default.

## Where to go from here

- **[Concepts](concepts/index.md)** — the framework proper: what a network, element, layer and
  solver each are, and how they fit.
- **[Applications](applications/index.md)** — six worked systems with real file readers and
  parity with the reference implementation in each field. If your problem is a building, a
  street, a sewer or a water network, start there rather than from `Network`.
- **[Theory](theory.md)** — why any of this is well posed.
