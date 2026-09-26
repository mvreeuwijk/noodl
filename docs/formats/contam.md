# CONTAM (`.prj`, `.wth`)

Reading a CONTAM project and weather file produces a model for [building
physics](../applications/building_physics.md). Each `.prj` airflow-path component becomes one of
the [`noodl.elements`](../applications/building_physics.md#airflow-elements) objects the building
application assembles into a network by hand; a source record becomes one of the four source
models in `noodl.apps.building_physics.sources`.

## CONTAM `.prj`

`read_prj(path) -> Project` parses a documented subset of TN 1887r1 Appendix A into a `Project`,
building a float64 `Network`, its elements and its drives. `project_to_model(project, *,
ambient=None, species=True, scheme="implicit")` returns `(model, state, drivers)`.

**Supported:** run control and ambient conditions; species; wind pressure profiles; zones and
initial concentrations; airflow paths; the power-law element family
(`plr_orfc/leak1/leak2/leak3/crack/fcn/test1/test2/conn/stair/shaft/qcn`), the quadratic family
(`qfr_*`), doorways (`dor_door`, `dor_pl2`), dampers (`plr_bdq`, `plr_bdf`), constant-flow fans
(`fan_cmf`, `fan_cvf`) and the cubic fan curve (`fan_fan`); source elements `ccf`, `cut`, `eds`,
`brs`.

**Refused by name**, rather than silently dropped: schedules, control nodes, filters, kinetic
reactions and air-handling systems referenced by nonzero index; CFD and 1-D convection/diffusion
zones; continuous-values-file zones and paths; duct networks; a constant wind pressure with no
profile; a fan curve on a path with `mult != 1`; `csf_*` and `sup_afe` elements.

A **filter** is refused emphatically, because a filter is invisible to airflow but *not* to the
species layer — loading one would silently corrupt contaminant results while the airflow looked
perfect.

Control nodes that no zone and no path references are silently skipped rather than refused: NIST's
own sample projects carry dozens of unreferenced sensor and logger nodes, and those projects must
load.

**`project_to_model` never builds a thermal layer**, because a `.prj` carries no thermal data.

Every power-law element is built as `UpstreamDensityPowerLaw` — re-evaluating the density of the
air *entering* the path, per direction, matching ContamX section 3.2. The quadratic, damper and
fan families keep reference-density coefficients, because they are not part of this application's
parity evidence.

```python
from noodl.apps.building_physics import read_prj, project_to_model

project = read_prj("valThreeZonesWthCtm-UseApi.prj")
model, state, drivers = project_to_model(project)
solved = model.steady(state, drivers)

flows = project.path_flows(solved["air.q"])   # in CONTAM path-number order
```

## CONTAM `.wth`

`read_wth(path) -> Weather` reads the `WeatherFile ContamW 2.0` format. Only `Date, Time, Ta, Pb,
Ws, Wd` are used; humidity, radiation and ground-temperature columns are ignored, as is the
per-day header block.

```python
weather = read_wth("year.wth")
drivers.update(weather.drivers_at(t, thermal="thermal"))
```

`at(t)` interpolates linearly between listed times, with wind direction interpolated on its
shortest arc. `drivers_at(t)` returns the driver keys a `build_model`-built model reads:
`"<thermal>.x_boundary"`, `"P_ref"`, `"V_met"`, `"theta_w"`. It handles the single-boundary-node
case; a model with several prescribed temperatures must build its own boundary vector.
