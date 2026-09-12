# tellegen

Network topology and conservative physics on graphs, in PyTorch. Named after
Tellegen's theorem: for any potentials in the cut space and any flows in the
cycle space of a graph, the branch power sum is zero.

Origin: John Craske's 2019 `Tellegen` package (autograd, Brayton and Moser's
formulation of nonlinear networks), kept unchanged in `legacy/`. This
repository ports the topology layer to PyTorch and rebuilds the physics as
nodal state-space modules with storage at nodes, typed edges, and batching
over buildings.

## Layout

```
src/tellegen/
  topology.py    typed multigraph; incidence, gradient, cycle basis, selectors, sign-aware
                 upwind/downwind, spanning forest, interior/boundary indices, caching,
                 to(device, dtype)
  cycles.py      branch_flows, assert_forward_oriented, particular_flow, project_measured
  elements/      Element base class and built-in branch laws: powerlaw.py (PowerLaw, Orifice),
                 quadratic.py (Quadratic), fixed.py (FixedFlow), conductance.py (Conductance),
                 fan.py (FanCurve)
  drives.py      Drive protocol, ConstantDrive
  solvers/       scalar.py (solve_monotone), linear.py (solve, floating_nodes),
                 newton.py (newton, NewtonResult), implicit.py (implicit_solve, adjoint)
  layers/        potential.py (PotentialFlowLayer), transport.py (TransportLayer),
                 reaction.py (Reaction, FirstOrderDecay)
  physics/       flows.py, species.py: thin wrappers so downstream code runs unchanged
legacy/          the original 2019 package, for reference
docs/superpowers/  design spec and implementation plans
tests/
  conftest.py, test_topology.py, test_cycles.py, test_drives.py, test_flows.py,
  test_import.py, test_species.py, test_species_compat.py
  elements/      test_base.py, test_powerlaw.py, test_quadratic.py, test_fixed.py,
                 test_conductance.py, test_fan.py
  solvers/       test_scalar.py, test_linear.py, test_newton.py, test_implicit.py
  layers/        test_potential.py, test_transport.py, test_reaction.py
  verification/  CONTAM-style closed-form airflow cases, batched against scipy roots;
                 CPU performance budget (marked slow, skipped by default)
  golden/        stored reference results (contam_airflow.json) and load_golden/save_golden
benchmarks/
  newton_scaling.py     batched Newton solve timing vs network size and batch size
  regenerate_golden.py  rewrites tests/golden/contam_airflow.json (explicit action)
```

Licence: MIT (the original code with the agreement of its author).

## Quick start

```python
import torch

from tellegen.elements.powerlaw import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network

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
