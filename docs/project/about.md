# About

noodl physics is based on a simple idea: many physical systems that appear very different share the same underlying network structure. noodl physics provides a general, differentiable framework for modelling such systems, allowing the same models to be used for simulation, inference and optimisation.

The idea grew out of discussions between John Craske and Maarten van Reeuwijk at Imperial College London, starting around 2020, about complex physical networks and the surprising amount of mathematical structure shared by models from different disciplines. noodl physics is an attempt to turn that observation into a practical modelling framework: small enough to understand, general enough to extend, and flexible enough to connect different physical systems.

For several years, noodl physics remained largely an idea. The emergence of AI-assisted software development changed what was practical. Using Claude made it possible to turn a concept we had discussed for years into a working framework in a remarkably short time.

## The name

*noodl* stands for **N**etwork-**O**riented **O**pen **D**ifferentiable **L**ibrary; *physics*
says what it is for. The package is installed as `noodl-physics` and imported as `noodl`. The
mark is a single continuous pipe between two nodes: flow through a network, drawn in one
stroke.

## Credits

The topology layer is built on PyTorch, and the physics is implemented as nodal state-space
modules with storage at nodes, typed edges, and batching over instances. The applications are
compared against, and in places transcribe the published rules of, CONTAM, MUNICH, SIRANE,
SWMM, EPANET, WSIMOD, the Modelica Standard and Buildings libraries and pandapower; see
[Verification](../reference/verification/index.md) for the versions.

## Licence

Copyright (c) 2026, Maarten van Reeuwijk, John Craske and Imperial College London. Licensed
under the BSD 3-Clause License; see
[`LICENSE`](https://github.com/mvreeuwijk/noodl/blob/main/LICENSE).

## Citing noodl physics

There is no paper or DOI yet. Until there is, cite the software and its version:

> van Reeuwijk, M. and Craske, J. (2026). *noodl physics: differentiable network modelling of
> physical systems* (version 0.1.0). Imperial College London.
> <https://github.com/mvreeuwijk/noodl>

Print the installed version with `python -c "import noodl; print(noodl.__version__)"`.
