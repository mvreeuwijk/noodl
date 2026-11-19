# Your first model

This tutorial builds a working model in about fifteen lines, then shows the two things that make
noodl physics different from a conventional network solver: it batches, and it differentiates.

## A two-node network

The smallest interesting network: a room connected to the outside through a leak. Air is
injected into the room at a known rate; we want the pressure that develops and the flow that
escapes.

```python
import torch

from noodl.elements.powerlaw import PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

net = Network(dtype=torch.float64)
net.add_node("ambient")
net.add_node("room")
net.add_edge("room", "ambient", kind="airpath")

leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65))
layer = KirchhoffLayer(net, "air", [leak], boundary=["ambient"])

phi_boundary = torch.zeros(1, dtype=torch.float64)
# sources is indexed in full node order (ambient, room); 0.05 injected into "room"
sources = torch.tensor([0.0, 0.05], dtype=torch.float64)
phi, q = layer.solve(phi_boundary, sources=sources, differentiable=False)

print("pressures (Pa):", {n: round(v, 3) for n, v in zip(net.nodes, phi.tolist())})
print("flows:", [round(x, 6) for x in q.tolist()])
```

```text
pressures (Pa): {'ambient': 0.0, 'room': 11.894}
flows: [0.05]
```

Four things happened there, and they are the whole framework in miniature:

1. **`Network`** holds the graph. Every edge carries a `kind` — here `"airpath"` — which is how
   one network can hold several physical systems side by side.
2. **`PowerLaw`** is a *constitutive law*, relating the potential difference across
   an edge to the flow along it, $q = C \, \Delta p^{\,n}$. Constitutive laws are the replaceable part.
3. **`KirchhoffLayer`** assembles nodal conservation from the laws and solves
   $A_I \, g(A^\top \phi) = s_I$ for the interior potentials by Newton's method. `boundary`
   names the nodes whose potential is prescribed rather than solved.
4. The flow out matches the flow in to solver tolerance — conservation is structural, not
   something the law had to be told about.

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
print("room pressures:", [round(v, 3) for v in phi[:, 1].tolist()])
```

```text
torch.Size([5, 2]) torch.Size([5, 1])
room pressures: [1.0, 2.905, 5.42, 8.438, 11.894]
```

The same mechanism carries a thousand building variants, a Monte Carlo sample over uncertain
leakage, or one network under a year of hourly weather. Nothing in the model changes; only the
leading dimension does.

## Gradients

Drop `differentiable=False` and the solve becomes differentiable. Gradients reach the boundary
potentials, the sources, every value in `drivers`, and any law parameter constructed with
`learnable=True`.

```python
C = torch.tensor(0.01, requires_grad=True)
leak = PowerLaw(C=C, n=torch.tensor(0.65), learnable=True)
layer = KirchhoffLayer(net, "air", [leak], boundary=["ambient"])

phi, q = layer.solve(
    torch.zeros(1, dtype=torch.float64),
    sources=torch.tensor([0.0, 0.05], dtype=torch.float64),
)
phi[1].backward()

# gradient of the room pressure with respect to each learnable parameter of the law
print({name: round(p.grad.item(), 1) for name, p in leak.named_parameters()})
```

```text
{'C': -1829.9, 'n': -45.3}
```

That derivative did not come from finite differences or from unrolling the Newton iteration.
It came from the implicit-function theorem applied at the converged solution, which costs one
extra linear solve regardless of how many Newton steps the forward pass took. See
[Differentiability](../concepts/differentiability.md) for what that buys and what it requires of
your elements.

## Several physics on one network

A real model has more than one layer, for example airflow and heat on one network. `Model`
steps them together, and a model is normally built by an application's `build_model` rather
than layer by layer. The [user guide](../guide/workflow.md) is the recipe, the same for every
application, and [Assembling from components](../guide/components.md) shows how to build one by hand.

## Where to go from here

- **[Concepts](../concepts/index.md)** — the framework proper: what a network, law, layer and
  solver each are, and how they fit.
- **[Run a model from a file](run-from-file.md)** — the next tutorial: read an EPANET network and run it.
- **[Applications](../applications/index.md)** — worked systems with file readers and
  comparisons with the reference tool in each field. If your problem is a building, a street,
  a sewer, a water network or an electrical circuit, start there rather than from `Network`.
- **[Theory](../concepts/theory.md)** — why any of this is well posed.
