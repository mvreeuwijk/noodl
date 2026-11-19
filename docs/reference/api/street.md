# Street air quality

Street-canyon networks. Application guide: [Street air quality](../../applications/street/index.md).

::: noodl.apps.street_aq.above_roof
    options:
      members:
        - RoofWiring
        - roof_wiring
        - roof_fluxes
        - street_steady_with_plume

::: noodl.apps.street_aq.canyon
    options:
      members:
        - soulhac_residual
        - soulhac_shape
        - BoundaryLayer
        - macdonald_profile
        - bessel_j0
        - bessel_j1
        - bessel_y0
        - bessel_y1
        - bessel_shape_residual
        - bessel_shape_parameter
        - grid_shape_parameter
        - shape_parameter
        - boundary_layer
        - canopy_displacement_roughness
        - roof_wind
        - canyon_velocity
        - exchange_velocity

::: noodl.apps.street_aq.case
    options:
      members:
        - StreetCase
        - drivers_at
        - read_case
        - write_case
        - StreetResults
        - read_results
        - write_sweep

::: noodl.apps.street_aq.chemistry
    options:
      members:
        - photostationary_for_streets
        - street_steady
        - j_no2_zenith_table
        - solar_elevation
        - j_no2_elevation_cloud
        - k_no_o3_soulhac2011

::: noodl.apps.street_aq.exposure
    options:
      members:
        - street_population
        - total_exposure
        - exposure_reduction_adjoint
        - exposure_reduction_forward

::: noodl.apps.street_aq.network
    options:
      members:
        - Street
        - StreetNetwork
        - munich_idealised
        - midlatitude_bearing
        - street_geometry
        - from_test_network
        - twelve_street_grid
        - build_model
        - initial_state
        - initial_drivers
        - street_index

::: noodl.apps.street_aq.plume
    options:
      members:
        - PlumeTable
        - turbulence
        - wind_speed
        - default_tau
        - plume_table
        - table_chunks
        - crosswind_profile
        - vertical_profile
        - StreetSources
        - source_points
        - street_midpoints
        - junction_points
        - junction_sources
        - rows_per_chunk
        - downwind_cutoff
        - plume_kernel
        - street_kernel
        - junction_kernel

::: noodl.apps.street_aq.puff
    options:
      members:
        - PuffClosure
        - PuffField
        - PuffOptions
        - PuffState
        - PuffTables
        - PulseSource
        - puff_tables
        - puff_mass
        - housekeep
        - emit
        - advance
        - roof_concentration
        - concentration_at
        - pulse_mass
        - mass_in_streets
        - pending_roof_mass

::: noodl.apps.street_aq.reactions
    options:
      members:
        - k_no_o3_jpl2003
        - molar_volume
        - Photostationary

::: noodl.apps.street_aq.report
    options:
      members:
        - to_ug_m3
        - from_ug_m3
        - write_network_concentration

::: noodl.apps.street_aq.routing
    options:
      members:
        - StreetFlows
        - StreetGeometry
        - routing_matrix
        - node_closure
        - direction_offsets
        - sigma_theta_munich
        - n_theta_munich
        - sirane_direction_samples
        - sigma_theta_turbulence_intensity
        - n_theta_rectangle_rule
        - exact_gaussian_direction_samples
        - order_slots
