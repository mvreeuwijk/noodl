# Model and coupling

The `Model` that steps layers together, the contracts closures follow, the helpers that build and check its state and drivers, and the coupling of several models. Guides: [The workflow](../../guide/workflow.md), [Coupling two models](../../guide/coupling.md).

## Model

::: noodl.model
    options:
      members:
        - Model
        - Ports

## Closures and contracts

::: noodl.contracts
    options:
      members:
        - StepContext
        - AlgebraicClosure
        - IntegratingClosure
        - ClosureAttribute
        - CLOSURE_ATTRIBUTES
        - ClosureSpec
        - closure_spec
        - call_closure
        - reaction_species_columns
        - check_species_columns

## Set-up: names, drivers and state

::: noodl.refs
    options:
      members:
        - LayerRefs
        - LayerRef
        - Field
        - drivers_from
        - state_from
        - initial_drivers
        - KeyOwnership
        - normalize_unit
        - units_agree
        - input_field
        - input_unit_declarations
        - required_state_keys

::: noodl.layout
    options:
      members:
        - batch_shape
        - broadcast_batch
        - per_instance
        - is_stacked
        - as_stacked
        - as_layout
        - reduced_single
        - write_at
        - species_columns

## Checking a set-up

::: noodl.validation
    options:
      members:
        - check_setup
        - SetupReport
        - Issue
        - InputRow

## Coupling

::: noodl.couple
    options:
      members:
        - union
        - CoupledModel
        - ValueLink
        - DriverAlias
        - apply_conversion
        - transport_boundary_inflow
        - Conversion
