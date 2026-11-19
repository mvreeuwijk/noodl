# Shared file helpers

What the file readers have in common: the line-oriented `.inp` parser used by the SWMM and EPANET readers, and the Modelica pipeline shared by the building and electrical readers.

## Line-oriented `.inp` files

::: noodl.apps.inpfile
    options:
      members:
        - read_sections
        - require_fields
        - as_float
        - InpLine

## Modelica

::: noodl.modelica.errors
    options:
      members:
        - ModelicaImportError

::: noodl.modelica.timeline
    options:
      members:
        - experiment_times
        - driver_grid
        - grid_events
        - signal_events
        - step_drivers
        - grid_index
        - extrapolate

::: noodl.modelica.inputs
    options:
      members:
        - SignalResolver
        - evaluation_problem
