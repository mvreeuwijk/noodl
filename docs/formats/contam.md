# CONTAM (`.prj`, `.wth`, `.ctm`)

Reading a CONTAM project, weather file and ambient-concentration file produces a model for [building
physics](../applications/building_physics.md). Each `.prj` airflow-path component becomes one of
the [`noodl.elements`](../applications/building_physics.md#airflow-elements) objects the building
application assembles into a network by hand; a source record becomes one of the four source
models in `noodl.apps.building_physics.sources`.

## CONTAM `.prj`

`read_prj(path) -> Project` parses a documented subset of TN 1887r1 Appendix A into a `Project`,
building a float64 `Network`, its elements and its drives. `project_to_model(project, *,
ambient=None, species=True, scheme="implicit")` returns `(model, state, drivers)`.

**Supported:** run control (`Project.run_control`, a `RunControl`: the steady simulation's
date and time, the transient run's start and end date and time, time step and output step, day
type, the weather and contaminant file names, and `dodMdt`) and ambient conditions; species and
each species' default ambient concentration (`Project.ambient_mass_fractions`); day and week
schedules referenced by zones (the zone temperature, on `"T"`) and by paths (a flow multiplier,
through `Scaled` on `"flow_multiplier.<kind>"`); wind pressure profiles, and a constant wind
pressure on an envelope path (`"wind_pressure.<kind>"`, positive pushing air from ambient into
the zone, as ContamX); zones and initial concentrations; airflow paths; the power-law element
family
(`plr_orfc/leak1/leak2/leak3/crack/fcn/test1/test2/conn/stair/shaft/qcn`), the quadratic family
(`qfr_*`), doorways (`dor_door`, `dor_pl2`), dampers (`plr_bdq`, `plr_bdf`), constant-flow fans
(`fan_cmf`, `fan_cvf`) and the cubic fan curve (`fan_fan`); source elements `ccf`, `cut`, `eds`,
`brs`.

**Refused by name**, rather than silently dropped: control nodes, filters, kinetic reactions
and air-handling systems referenced by nonzero index; a schedule number that no week schedule
carries; a schedule on a `fan_fan` path; a zone without the variable-pressure flag (which ContamX
holds at constant pressure); CFD and 1-D convection/diffusion zones; continuous-values-file
zones and paths; duct networks; a constant wind pressure on an interior path; a fan curve on a
path with `mult != 1`; `csf_*` and `sup_afe` elements; `stackD = 1` (densities varying with
height inside a zone).

A **filter** is refused emphatically, because a filter is invisible to airflow but *not* to the
species layer — loading one would silently corrupt contaminant results while the airflow looked
perfect.

Control nodes that no zone and no path references are silently skipped rather than refused: NIST's
own sample projects carry dozens of unreferenced sensor and logger nodes, and those projects must
load.

**`project_to_model` never builds a thermal layer**, because a `.prj` carries no thermal data.

`project_to_model` returns drivers at the **steady-simulation instant** of the run control (its
day, time and day type), so a `steady()` on them is CONTAM's steady simulation, schedules
included. `Project.inputs` holds the schedules as `Input`s for `simulate`.

Every power-law element is built as `UpstreamDensityPowerLaw` — re-evaluating the density of the
air *entering* the path, per direction, matching ContamX section 3.2. The quadratic, damper and
fan families keep reference-density coefficients, because they are not part of this application's
parity evidence.

Node densities are computed every step by the `PrescribedTemperatureDensity` closure from two
drivers: `"T"` (K, every node's temperature, the ambient's included) and `"P_ref"` (Pa, the
barometric pressure). The species layer's capacity is each zone's air mass at that density,
`rho V`, as ContamX holds it. A scheduled zone temperature or a weather file therefore changes
both from step to step. When the run control sets **`densZP = 1`**, ContamX evaluates each zone's
density at its absolute pressure `Pb + P_zone`. `project_to_model` then adds a
`ZonePressureDensity` closure that does the same from the previous solve's pressures, zone air
masses included, and `prj.steady(model, state, drivers)` repeats `model.steady` to that fixed
point (a single call when `densZP = 0`). The term is small in the density (|P_zone| / Pb,
~1e-5) but a stack flow is driven by a density *difference*, which amplifies it: on a 20 K stack
it is 4e-5 of the flow.

```python
from noodl.apps.building_physics import read_prj, project_to_model
from noodl.apps.building_physics.prj import steady

project = read_prj("valThreeZonesWthCtm-UseApi.prj")
model, state, drivers = project_to_model(project)
solved = steady(model, state, drivers)

flows = project.path_flows(solved["air.q"])   # in CONTAM path-number order
```

## CONTAM `.wth`

`read_wth(path) -> Weather` reads the `WeatherFile ContamW 2.0` format. `Date, Time, Ta, Pb, Ws,
Wd` drive the model, and `Hr` (g/kg) is stored as `humidity_ratio` in kg/kg. The per-day rows
give each day's **day type**, which selects the day schedules of the week schedules;
`Weather.day_types` maps the day of the year to it, and a `Calendar(day_types=...)` uses it.
Radiation and ground-temperature columns are ignored.

```python
weather = read_wth("year.wth")
drivers.update(weather.drivers_at(t, thermal="thermal"))
```

`at(t)` interpolates linearly between listed times, with wind direction interpolated on its
shortest arc. `drivers_at(t)` returns the driver keys a `build_model`-built model reads:
`"<thermal>.x_boundary"`, `"P_ref"`, `"V_met"`, `"theta_w"`. It handles the single-boundary-node
case; a model with several prescribed temperatures must build its own boundary vector.

`Weather.inputs(model)` writes the weather into a model's drivers at every step of `simulate`:
the ambient entry of `"T"` on the CONTAM route, or every boundary temperature of a thermal
model, plus `P_ref`, `V_met` and `theta_w`. `Weather.signal("Ta")` is one column as a signal.
A run outside the file's rows raises.

## CONTAM `.ctm`

`read_ctm(path) -> AmbientConcentrations` reads the `SpeciesFile ContamW 2.0` ambient
(outdoor) concentration file:

```text
SpeciesFile ContamW 2.0
free-text description
1/1<TAB>1/2<TAB>1                 start date, end date, number of species
sarin                             one species name per line
!Date<TAB>Time<TAB>sarin
1/1<TAB> 0:00:00<TAB>0.0023254    rows: date, time, one mass fraction (kg/kg) per species
1/1<TAB>24:00:00<TAB>0.0
```

Values are interpolated linearly between rows, and a day's `24:00:00` is the next day's
`00:00:00`. ContamX uses the file only when its species match the project's; this reader
refuses a mismatch and names both lists.

```py
outdoor = read_ctm("outdoor.ctm")
inputs = outdoor.inputs(model)      # writes "species.x_boundary" at every step of simulate
```

