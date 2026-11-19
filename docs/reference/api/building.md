# Building physics

Multi-zone airflow, heat and contaminants. Application guide: [Building physics](../../applications/building/index.md).

::: noodl.apps.building_physics.air_handling
    options:
      members:
        - AirHandlerFlows
        - add_air_handler
        - air_handler_from_edges
        - air_handler_plenums

::: noodl.apps.building_physics.components
    options:
      members:
        - room
        - door
        - crack
        - window
        - shaft
        - fan
        - duct
        - air_handler

::: noodl.apps.building_physics.contamx
    options:
      members:
        - run_steady
        - run_transient
        - run_project

::: noodl.apps.building_physics.controls
    options:
      members:
        - Evaluation
        - StateReading
        - DriverReading
        - FunctionReading
        - Block
        - Actuator
        - ControlNetwork
        - NamedReading
        - Constant
        - DriverValue
        - Identity
        - Sensor
        - Sum
        - Average
        - Maximum
        - Minimum
        - Product
        - Divide
        - Absolute
        - Positive
        - Complement
        - Affine
        - And
        - Or
        - ExclusiveOr
        - BelowLimit
        - AboveLimit
        - Shortfall
        - Excess
        - LowerBandSwitch
        - UpperBandSwitch
        - Hysteresis
        - ProportionalControl
        - IncrementalPI
        - LimitedPID
        - node_mass_fraction
        - node_temperature
        - node_pressure
        - edge_flow
        - edge_pressure_difference

::: noodl.apps.building_physics.ctm
    options:
      members:
        - AmbientConcentrations
        - read_ctm

::: noodl.apps.building_physics.ducts
    options:
      members:
        - DuctNetwork
        - duct_laws_from_edges

::: noodl.apps.building_physics.elements
    options:
      members:
        - mass_orifice
        - add_large_opening
        - orifice_law_from_edges
        - fan_law_from_edges

::: noodl.apps.building_physics.epw
    options:
      members:
        - read_epw
        - write_wth

::: noodl.apps.building_physics.filters
    options:
      members:
        - AirThroughFilter
        - ConstantEfficiencyFilter
        - ParticleFilter
        - UVGIFilter
        - FilterSeries
        - FilterTransmission
        - filter_transmission

::: noodl.apps.building_physics.inputs
    options:
      members:
        - Input
        - bind_inputs
        - scatter
        - apply_inputs
        - input_breakpoints

::: noodl.apps.building_physics.modelica
    options:
      members:
        - read_modelica
        - ModelicaNames
        - simulate

::: noodl.apps.building_physics.occupants
    options:
      members:
        - OccupancyDay
        - Presence
        - Occupant
        - OccupantGeneration
        - Occupancy
        - occupant_inputs

::: noodl.apps.building_physics.outputs
    options:
      members:
        - species_history
        - occupant_exposure
        - cumulative_exposure
        - inhaled_dose
        - infection_probability
        - wells_riley
        - zone_air_change_rate
        - building_air_change_rate
        - local_mean_age

::: noodl.apps.building_physics.prj
    options:
      members:
        - read_prj
        - project_to_model
        - steady
        - ZonePressureDensity
        - Project
        - PrjElement
        - PrjPath
        - PrjTerminal
        - PrjSegment
        - PrjSource
        - RunControl
        - PrjExposure
        - PrescribedTemperatureDensity
        - ContamCompatibilityWarning
        - fan_curve_element
        - spline_element
        - ZoneAirMassChange

::: noodl.apps.building_physics.schedules
    options:
      members:
        - Calendar
        - DaySchedule
        - TimeSeries
        - Signal
        - WeekSchedule

::: noodl.apps.building_physics.simulation
    options:
      members:
        - simulate
        - project_calendar
        - simulate_project

::: noodl.apps.building_physics.sources
    options:
      members:
        - SpeciesSources
        - ConstantSource
        - CutoffSource
        - DecayingSource
        - BurstSource
        - DepositionVelocitySink
        - DepositionRateSink
        - PressureDrivenSource
        - PowerLawDecaySource
        - PeakEmissionSource
        - assemble_sources
        - sources_from_project

::: noodl.apps.building_physics.species
    options:
      members:
        - Species
        - KineticReaction
        - reaction_kinetics

::: noodl.apps.building_physics.thermal
    options:
      members:
        - WallMass
        - Zone
        - add_zone
        - thermal_layer
        - species_layer
        - IdealGasDensity
        - LinearDensity
        - check_flow_balanced
        - check_head_order
        - build_model
        - initial_state
        - initial_drivers

::: noodl.apps.building_physics.wth
    options:
      members:
        - read_wth
        - Weather
