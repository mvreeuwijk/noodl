# Water resources

The WSIMOD port. Application guide: [Water resources](../../applications/water-resources/index.md). The allocation layer it uses is on the [Layers](layers.md) page.

::: noodl.apps.wsimod.build
    options:
      members:
        - build_model
        - initial_state
        - initial_drivers

::: noodl.apps.wsimod.config
    options:
      members:
        - read_wsimod_config
        - WsimodNetwork
        - NodeSpec
        - ArcSpec
        - SurfaceSpec
        - parse_date

::: noodl.apps.wsimod.day
    options:
      members:
        - DayRecord
        - WsimodDay

::: noodl.apps.wsimod.run
    options:
      members:
        - WsimodMassBalance
        - WsimodResults
        - wsimod_run

::: noodl.apps.wsimod.structure
    options:
      members:
        - check_network
        - recursion_cycles
        - river_order
        - orchestration_plan
        - forcing_readers
        - surface_forcing_readers
