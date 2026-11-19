# Constitutive laws and heads

The general-purpose laws that relate the flow through an edge to the potential difference across it, and the terms added to that difference. Subclass `ConstitutiveLaw` or `HeadTerm` to add your own ([Extending noodl](../../project/extending/index.md)). Concepts: [Elements](../../concepts/elements.md).

## Base classes

::: noodl.elements.base
    options:
      members:
        - ConstitutiveLaw
        - ImplicitLaw
        - LawContext
        - step_rate
        - step_rate_factor
        - memo
        - band_edges

## Power laws and orifices

::: noodl.elements.powerlaw
    options:
      members:
        - PowerLaw
        - Orifice

::: noodl.elements.powerlaw_regularized
    options:
      members:
        - MBLPowerLaw
        - mbl_orifice
        - mbl_ela
        - mbl_point
        - mbl_points
        - mbl_coefficient
        - RegularizedPowerLaw
        - regularized_orifice
        - effective_leakage_area
        - power_law_from_point
        - power_law_from_points
        - power_law_coefficient

::: noodl.elements.upstream
    options:
      members:
        - UpstreamDensityPowerLaw

## Quadratic, tabulated and curve laws

::: noodl.elements.quadratic
    options:
      members:
        - Quadratic

::: noodl.elements.table
    options:
      members:
        - MBLTable
        - SplineFlowTable

::: noodl.elements.curve
    options:
      members:
        - SplineCurve
        - clamped_spline_slopes
        - CubicSplineFlow

## Doors

::: noodl.elements.door
    options:
      members:
        - MBLDoorOpen
        - MBLDoorOperable
        - mbl_door_pair
        - mbl_operable_door_pair
        - OpenDoor
        - OperableDoor
        - open_door_pair
        - operable_door_pair

::: noodl.elements.door_discretized
    options:
      members:
        - MBLDoorCompartment
        - MBLDoorCompartmentOperable
        - MBLDoorPortStream
        - mbl_discretized_door
        - mbl_discretized_operable_door
        - DoorCompartmentHead
        - DoorCompartment
        - OperableDoorCompartment
        - discretized_door
        - discretized_operable_door
        - DoorPortStream

## Ducts, dampers and fans

::: noodl.elements.duct
    options:
      members:
        - Duct

::: noodl.elements.air_duct
    options:
      members:
        - air_viscosity
        - AirDuct
        - DuctTerminal

::: noodl.elements.damper
    options:
      members:
        - Damper

::: noodl.elements.fan
    options:
      members:
        - FanCurve
        - AffinityFan

## Fixed, scaled and linear flows

::: noodl.elements.fixed
    options:
      members:
        - FixedFlow
        - FixedVolumeFlow
        - PrescribedFlow

::: noodl.elements.scaled
    options:
      members:
        - Scaled

::: noodl.elements.conductance
    options:
      members:
        - Conductance

## Fluid properties

::: noodl.elements.media
    options:
      members:
        - AirMedium
        - MBLMedium
        - medium

## Heads

::: noodl.heads
    options:
      members:
        - HeadTerm
        - EdgeHeads
        - DriverHead
        - Stack
        - DuctStack
        - Wind
        - WindProfile

## Withdrawals and storage

::: noodl.withdrawals
    options:
      members:
        - NodeWithdrawal

::: noodl.storage
    options:
      members:
        - NodeStorage
        - LinearStorage
