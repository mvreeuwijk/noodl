# Water distribution

Pressurised water networks. Application guide: [Water distribution](../../applications/water/index.md).

::: noodl.apps.water.controls
    options:
      members:
        - Control
        - Premise
        - Action
        - Rule
        - status_code
        - pump_speed_for_status
        - apply_controls
        - next_control_time

::: noodl.apps.water.demand
    options:
      members:
        - PressureDrivenDemand
        - EmitterOutflow

::: noodl.apps.water.elements
    options:
      members:
        - EpanetDarcyWeisbach
        - epanet_friction_factor
        - ShutLinks
        - link_driver
        - link_flags
        - closed_row
        - open_valve_loss
        - area_flow
        - open_start
        - HazenWilliams
        - ChezyManning
        - three_point_curve
        - PumpCurve
        - PiecewisePumpCurve
        - ConstantPowerPump
        - MinorLoss
        - composite_friction_factor
        - CompositeDarcyWeisbach
        - epanet_minor_coefficient
        - ThrottleValve
        - FlowControlValve
        - BreakerValve
        - GeneralPurposeValve
        - PressureReducingValve
        - PressureSustainingValve
        - FlowControlledPipe
        - FlowControlledPump

::: noodl.apps.water.energy
    options:
      members:
        - EnergyMeter
        - PumpEnergy
        - interpolate_curves

::: noodl.apps.water.inp
    options:
      members:
        - read_epanet_inp

::: noodl.apps.water.network
    options:
      members:
        - WaterOptions
        - Junction
        - Reservoir
        - Tank
        - WaterPipe
        - Pump
        - Valve
        - Emitter
        - EnergyOptions
        - WaterNetwork
        - twoloop
        - build_model
        - power_curve
        - initial_state
        - initial_drivers
        - link_setting_scale
        - apply_rules
        - apply_rules_with_states
        - solved_link_states
        - water_steady
        - pattern_factor
        - pattern_period
        - with_tank_inflow
        - water_steady_and_cut_off
        - link_states
        - link_states_from_cut_off
        - water_step
        - settle_tanks
        - tank_levels
        - tank_inflow

::: noodl.apps.water.quality
    options:
      members:
        - QualityMassBalance
        - QualityRouter
        - SegmentPool
        - flow_volume_scale
        - quality_step
        - sort_nodes

::: noodl.apps.water.quality_data
    options:
      members:
        - QualitySource
        - WaterQuality

::: noodl.apps.water.quality_react
    options:
      members:
        - pipe_coefficients
        - tank_coefficients
        - bulk_rate
        - wall_coefficients
        - wall_rate

::: noodl.apps.water.report
    options:
      members:
        - pressure_head
        - to_kilopascal
        - link_table

::: noodl.apps.water.rules
    options:
      members:
        - evaluate_rules
        - take_actions
        - take_actions_per_link

::: noodl.apps.water.simulate
    options:
      members:
        - WaterResults
        - water_run

::: noodl.apps.water.tanks
    options:
      members:
        - LinkData
        - TankLevels
        - TankVolumeCurve
        - TankStorageControls
