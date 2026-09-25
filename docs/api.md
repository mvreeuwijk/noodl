# API reference

Generated from the package's own docstrings. For the ideas behind these objects, start with
[Concepts](concepts/index.md); for worked usage, [Applications](applications/index.md).

!!! note

    This page renders only on the built documentation site, where `mkdocstrings` reads the
    installed package. On GitHub it shows as a list of module names.

## Topology

::: noodl.topology
    options:
      members: [Network]

::: noodl.cycles
    options:
      members: [branch_flows, particular_flow, project_measured, assert_forward_oriented]

## Elements and drives

::: noodl.elements.base
    options:
      members: [Element]

::: noodl.elements.powerlaw
::: noodl.elements.quadratic
::: noodl.elements.conductance
::: noodl.elements.fixed
::: noodl.elements.fan
::: noodl.elements.duct
::: noodl.elements.damper
::: noodl.elements.upstream

::: noodl.drives
    options:
      members: [Drive, ConstantDrive, Stack, Wind, WindProfile]

::: noodl.nodesources
    options:
      members: [NodeSource]

## Layers

::: noodl.layers.potential
    options:
      members: [PotentialFlowLayer]

::: noodl.layers.transport
    options:
      members: [TransportLayer, active_interior]

::: noodl.layers.reaction
    options:
      members: [Reaction, FirstOrderDecay, Photostationary]

::: noodl.layers.capacitated
    options:
      members: [CapacitatedTransferLayer]

## Model and coupling

::: noodl.model
    options:
      members: [Model, Closure, Ports]

::: noodl.couple
    options:
      members: [union, CoupledModel, ValueLink, DriverAlias, apply_conversion, transport_boundary_inflow]

## Operators

::: noodl.operators.base
    options:
      members: [LinearOperator, SparseAssembling, SolveResult, SolverStatus, as_operator]

::: noodl.operators.dense
::: noodl.operators.graph
::: noodl.operators.advection

## Solvers

::: noodl.solvers.select
    options:
      members: [solve]

::: noodl.solvers.newton
    options:
      members: [newton, NewtonResult, inner_solve_rtol]

::: noodl.solvers.implicit
    options:
      members: [implicit_solve, adjoint, TransposeOperator]

::: noodl.solvers.iterative
    options:
      members: [pcg, gmres]

::: noodl.solvers.scalar
::: noodl.solvers.grounding
    options:
      members: [spd_certificate, spd_diagnosis]

## Applications

### Building physics

::: noodl.apps.building_physics.thermal
::: noodl.apps.building_physics.elements
::: noodl.apps.building_physics.prj
    options:
      members: [read_prj, project_to_model, Project]
::: noodl.apps.building_physics.wth
    options:
      members: [read_wth, Weather]
::: noodl.apps.building_physics.sources

### Street air quality

::: noodl.apps.street_aq.network
::: noodl.apps.street_aq.canyon
::: noodl.apps.street_aq.routing
    options:
      members: [StreetFlows, StreetGeometry, routing_matrix, node_closure, direction_offsets]
::: noodl.apps.street_aq.chemistry
::: noodl.apps.street_aq.loader
    options:
      members: [read_aqdt, AqdtData, Forcing]
::: noodl.apps.street_aq.report

### Sewers

::: noodl.apps.sewer.network
::: noodl.apps.sewer.geometry
::: noodl.apps.sewer.hydraulics
    options:
      members: [SewerHydraulics, resolve_nodal_driver]
::: noodl.apps.sewer.air
::: noodl.apps.sewer.quality
::: noodl.apps.sewer.inp
    options:
      members: [read_swmm_inp]
::: noodl.apps.sewer.report

### Water distribution

::: noodl.apps.water.network
::: noodl.apps.water.elements
::: noodl.apps.water.tanks
::: noodl.apps.water.demand
::: noodl.apps.water.inp
    options:
      members: [read_epanet_inp]
::: noodl.apps.water.report

### Shared

::: noodl.apps.inpfile
    options:
      members: [read_sections, require_fields, as_float, InpLine]
