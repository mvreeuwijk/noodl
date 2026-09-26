# noodl

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/noodl-physics-full-dark.svg">
    <img src="docs/assets/noodl-physics-full-light.svg" alt="noodl physics — complex networks, differentiable by design" width="640">
  </picture>
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
extras are genuinely optional — `[sparse]` (sparse-direct linear solver), `[street_aq]` (the
street air quality application's NetCDF I/O), `[contam]` (ContamX parity, Windows
x86-64 only) and `[dev]` (everything the test suite needs). For a checkout:

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

noodl is one generic solver for networks whose edges carry a flow and whose nodes conserve it.
The applications are physical systems modelled with that solver, each a thin layer of domain
physics over the shared core:

| Application | Physical system | Flow determination | Entry point |
|---|---|---|---|
| Building physics | Multi-zone airflow, heat and contaminant transport | Potential: Newton on zone pressure | `build_model` |
| Street air quality | Urban air quality, canyon exchange and routing | Closure: flows computed from the wind aloft | `build_model`, `StreetNetwork` |
| Sewers | Gravity sewer hydraulics, headspace air, sulfide | Continuity on a tree for the water; Newton potential for the headspace air | `build_model` |
| Water distribution | Pressurised mains, pumps, tanks, demand | Potential: Newton on hydraulic head | `build_model` |
| Capacitated allocation | Requested flows clipped to arc capacity and free storage at the receiving node | Capacitated clip | `CapacitatedTransferLayer` |
| Coupling | Two models meeting at a shared boundary | Two models exchanging values, iterated to a fixed point | `union` |

## Reading model files

A model can be built directly in Python or, where a reader exists, read from a file in an
established exchange format. A model read from a file can be batched, differentiated and coupled
exactly like one built by hand.

| Format | Reader | What it describes | Produces a model for |
|---|---|---|---|
| CONTAM `.prj` / `.wth` | `read_prj`, `project_to_model`, `read_wth` | Multi-zone building airflow and contaminants; weather | Building physics |
| Modelica Buildings Library (JSON + CSV export) | `read_modelica` | Multi-zone airflow models built from MBL components | Building physics |
| SWMM `.inp` | `read_swmm_inp` | Gravity sewer and drainage networks | Sewers |
| EPANET `.inp` | `read_epanet_inp` | Pressurised water distribution | Water distribution |

## Repository layout

```
src/noodl/        the package: topology, elements (every branch law, grouped by physics),
                  drives, layers, solvers, operators, Model and couple, and apps/
                  (building_physics with its CONTAM and Modelica readers, street_aq, sewer,
                  water, and inpfile, the tokenizer the two .inp readers share)
tests/            the suite, including verification/ — code-to-code comparisons against
                  reference models and analytical solutions — and the performance gates
benchmarks/       timing and scaling scripts, and the composed model: eight buildings
                  joined through street and sewer networks
docs/             the published documentation
scripts/          fixture-regeneration scripts (Modelica, WSIMOD)
```

The [API reference](docs/api.md) is organised module by module.

## Status

Version 0.1.0. The core and all six applications are implemented, each with its verification
cases in the test suite.

## The name

*noodl* stands for **N**etwork-**O**riented **O**pen **D**ifferentiable **L**ibrary. The mark is
a single continuous path through two nodes and a junction — flow through a network, drawn in
one stroke.

## Credits and licence

The topology layer is built on PyTorch, and the physics is implemented as nodal state-space
modules with storage at nodes, typed edges, and batching over instances.

Copyright (c) 2026, John Craske, Maarten van Reeuwijk and Imperial College London. Licensed under the BSD 3-Clause License; see `LICENSE`.
