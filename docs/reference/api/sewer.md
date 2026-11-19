# Sewers

Gravity sewers. Application guide: [Sewers](../../applications/sewer/index.md).

::: noodl.apps.sewer.air
    options:
      members:
        - Drag
        - AirPlan
        - Headspace
        - air_density
        - check_air_topology
        - add_air_edges
        - air_layer
        - air_heads

::: noodl.apps.sewer.dynwave
    options:
      members:
        - DynamicWaveOptions
        - DynamicWaveHydraulics
        - tidal_stage
        - DynamicWaveConduit
        - StructureLink
        - JunctionStorage
        - control_start
        - iterate_tolerances
        - build_dynamic_wave_model
        - OverflowLoss
        - air_state
        - quality_state
        - solve_atol
        - routing_step_size
        - simulate_dynamic_wave

::: noodl.apps.sewer.geometry
    options:
      members:
        - flow_area
        - wetted_perimeter
        - top_width
        - hydraulic_radius
        - hydraulic_mean_depth
        - manning_flow
        - capacity_flow
        - normal_depth
        - air_geometry

::: noodl.apps.sewer.hydraulics
    options:
      members:
        - SewerHydraulics
        - resolve_nodal_driver
        - air_density_nodes

::: noodl.apps.sewer.inp
    options:
      members:
        - read_swmm_inp
        - unit_factors

::: noodl.apps.sewer.network
    options:
      members:
        - Exfiltration
        - Manhole
        - Outfall
        - Pipe
        - Orifice
        - Weir
        - Outlet
        - Pump
        - SewerNetwork
        - storage_unit
        - tree_steady
        - build_model
        - initial_state
        - initial_drivers
        - sewer_steady

::: noodl.apps.sewer.quality
    options:
      members:
        - LateralLoads
        - H2STransfer
        - henry_h2s
        - free_fraction
        - kla_h2s
        - sulfide_rate
        - bod_rate
        - two_film_flux
        - ppm_from_concentration
        - SulfideGeneration

::: noodl.apps.sewer.rdii
    options:
      members:
        - Rdii
        - RdiiInflow
        - RdiiModel
        - UnitHydrograph
        - UnitHydrographSet
        - encode_date
        - month_of_year
        - add_seconds

::: noodl.apps.sewer.report
    options:
      members:
        - to_mg_per_litre
        - to_ppm
        - pipe_table

::: noodl.apps.sewer.runoff
    options:
      members:
        - Adjustments
        - CurveNumber
        - Evaporation
        - GreenAmpt
        - Horton
        - Hydrology
        - RainGage
        - RunoffDriver
        - RunoffInflows
        - RunoffModel
        - Subareas
        - Subcatchment
        - evaporation_can_be_positive
        - odesolve_integrate

::: noodl.apps.sewer.storage
    options:
      members:
        - StorageCurves
        - validate_storage_curve

::: noodl.apps.sewer.xsect_spec
    options:
      members:
        - Transect
        - Xsect
        - is_open

::: noodl.apps.sewer.xsect_tables
    options:
      members:
        - lookup_many
        - lookup
        - inv_lookup
        - y_of_a
        - s_of_a
        - a_of_s
        - a_of_y
        - conduit_constants
        - psi_of_flow
        - kinwave_steady
        - full_flow
        - area_of_depth
        - width_of_depth
        - radius_of_depth
        - area_width_of_depth
        - area_radius_of_depth
        - froude_number
        - critical_flow_stations
        - critical_depth
        - normal_depth_ft
