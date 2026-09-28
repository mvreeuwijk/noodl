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

Elements are grouped below by the physics they express, as in
[Airflow elements](applications/building_physics.md#airflow-elements).

### Power-law openings

::: noodl.elements.powerlaw
    options:
      heading_level: 4
::: noodl.elements.powerlaw_mbl
    options:
      heading_level: 4

### Density upwinding

::: noodl.elements.upstream
    options:
      heading_level: 4

### Quadratic and tabulated laws

::: noodl.elements.quadratic
    options:
      heading_level: 4
::: noodl.elements.table
    options:
      heading_level: 4

### Large openings and doors

::: noodl.elements.door
    options:
      heading_level: 4
::: noodl.elements.door_discretized
    options:
      heading_level: 4

### Ducts, dampers, fans, fixed flows and conductances

::: noodl.elements.duct
    options:
      heading_level: 4
::: noodl.elements.damper
    options:
      heading_level: 4
::: noodl.elements.fan
    options:
      heading_level: 4
::: noodl.elements.fixed
    options:
      heading_level: 4
::: noodl.elements.conductance
    options:
      heading_level: 4

### Air properties

::: noodl.elements.media
    options:
      heading_level: 4

### Drives

::: noodl.drives
    options:
      members: [Drive, ConstantDrive, Stack, Wind, WindProfile]
      heading_level: 4

::: noodl.nodesources
    options:
      members: [NodeSource]
      heading_level: 4

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
      members: [read_prj, project_to_model, steady, ZonePressureDensity, Project]
::: noodl.apps.building_physics.wth
    options:
      members: [read_wth, Weather]
::: noodl.apps.building_physics.sources
::: noodl.apps.building_physics.modelica
    options:
      members: [read_modelica, ModelicaNames, ModelicaImportError, simulate, step_drivers]

### Street air quality

::: noodl.apps.street_aq.network
::: noodl.apps.street_aq.canyon
::: noodl.apps.street_aq.routing
    options:
      members: [StreetFlows, StreetGeometry, routing_matrix, node_closure, direction_offsets]
::: noodl.apps.street_aq.case
    options:
      members: [StreetCase, drivers_at, read_case, write_case]
::: noodl.apps.street_aq.chemistry
::: noodl.apps.street_aq.report

### Sewers

::: noodl.apps.sewer.network
::: noodl.apps.sewer.geometry
::: noodl.apps.sewer.swmm_xsect
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
