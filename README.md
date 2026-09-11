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
  topology.py    typed multigraph; incidence, gradient, cycle basis, upwind operator, Tellegen residual
  physics/       air network (cycle-space latent flows), thermal network, species transport, plant modules
  dynamics.py    assembly into one right-hand side; batched integration
legacy/          the original 2019 package, for reference
tests/
```

Licence: MIT (the original code with the agreement of its author).
