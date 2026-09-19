# noodl

<p align="center">
  <img src="docs/assets/noodl-logo.png" alt="noodl — a differentiable library for network physics" width="640">
</p>

**noodl** — the **N**etwork-**O**riented **O**pen **D**ifferentiable **L**ibrary — is a generic
framework for solving physics problems on networks, built on PyTorch. It combines graph
topology, conservation laws, and modular descriptions of physical processes to model flows,
storage, transport, and reactions within a common framework. Different physical systems can be
composed and coupled through shared network structures, while differentiable solvers support
simulation, parameter estimation, and optimisation.

**[Documentation](https://mvreeuwijk.github.io/noodl/)** ·
[Installation](https://mvreeuwijk.github.io/noodl/installation/) ·
[Quick start](https://mvreeuwijk.github.io/noodl/quickstart/) ·
[Concepts](https://mvreeuwijk.github.io/noodl/concepts/) ·
[Applications](https://mvreeuwijk.github.io/noodl/applications/) ·
[API reference](https://mvreeuwijk.github.io/noodl/api/) ·
[Theory](https://mvreeuwijk.github.io/noodl/theory/)

## Installation

```
pip install noodl
```

Python 3.11 or newer; PyTorch, NetworkX and NumPy are the only hard dependencies. The optional
extras are genuinely optional — `[sparse]` (sparse-direct linear solver), `[street]` (the street
application's NetCDF I/O and comparison oracle), `[contam]` (ContamX parity, Windows x86-64 only)
and `[dev]` (everything the test suite needs). For a checkout:

```
git clone https://github.com/mvreeuwijk/noodl.git
cd noodl
pip install -e ".[dev]"
```

See [Installation](https://mvreeuwijk.github.io/noodl/installation/) for what each
extra buys you.

## Quick start

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
# sources is indexed in full node order (ambient, room); 0.05 m3/s injected into "room"
sources = torch.tensor([0.0, 0.05], dtype=torch.float64)
phi, q = layer.solve(phi_boundary, sources=sources, differentiable=False)

print("pressures (Pa):", dict(zip(net.nodes, phi.tolist())))
print("flows (m3/s):  ", dict(zip(("room->ambient",), q.tolist())))
```

## Applications

noodl ships six worked applications. Each is a thin layer of domain physics over the shared
core, and each is validated against the standard tool in its field.

| Application | Physical system | Reference model |
|---|---|---|
| Buildings | Multi-zone airflow, heat and contaminant transport | CONTAM / ContamX |
| Street canyons | Urban air quality, canyon exchange and routing | MUNICH, SIRANE, IMPAQ |
| Sewers | Gravity sewer hydraulics, headspace air, sulfide | SWMM |
| Water distribution | Pressurised mains, pumps, tanks, demand | EPANET 2.2 |
| Capacitated transfer | Rule-based water-systems allocation | WSIMOD |
| Coupling | Two independent models exchanging values | — |

## Repository layout

```
src/noodl/        the package: topology, elements, drives, layers, solvers, operators,
                  Model and couple, and apps/ (building, street, sewer, water)
tests/            the suite, including verification/ — the parity cases against CONTAM,
                  MUNICH, SWMM, EPANET and WSIMOD, and the performance gates
benchmarks/       timing and scaling scripts, and the composed reference model
docs/             the published documentation, plus internal specs and plans under
                  docs/superpowers/
legacy/           John Craske's original 2019 package, unchanged, for reference
```

A module-by-module map is in the
[development history appendix](docs/development-history.md#appendix-the-source-tree).

## Status

The framework core and all six applications are built and validated; the suite is 1306 tests
at 96 % coverage. The per-milestone engineering record — what each milestone added, the
decisions behind it, the measured errors and what it left open — is in
[docs/development-history.md](docs/development-history.md). The design specs and
implementation plans behind them are under `docs/superpowers/`.

## The name

*noodl* stands for **N**etwork-**O**riented **O**pen **D**ifferentiable **L**ibrary. The mark is
a single continuous path through two nodes and a junction — flow through a network, drawn in
one stroke.

The package was previously called `tellegen`, after Tellegen's theorem: for any potentials in
the cut space and any flows in the cycle space of a graph, the branch power sum is zero. That
result is still the backbone of the topology layer, and
[the theory page](https://mvreeuwijk.github.io/noodl/theory/) opens with it.

## Credits and licence

noodl grew out of John Craske's 2019 `Tellegen` package, which formulated nonlinear networks
after Brayton and Moser using autograd; that original is kept unchanged under `legacy/` for
reference. This project ports the topology layer to PyTorch and rebuilds the physics as nodal
state-space modules with storage at nodes, typed edges, and batching over instances.

Authors: John Craske and Maarten van Reeuwijk. MIT licensed, with the agreement of the original
author.
