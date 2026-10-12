# CONTAM (`.prj`, `.wth`, `.ctm`)

Reading a CONTAM project, weather file and ambient-concentration file produces a model for [building
physics](../applications/building_physics.md). Each `.prj` airflow-path component becomes one of
the [`noodl.elements`](../applications/building_physics.md#airflow-elements) objects the building
application assembles into a network by hand; a source/sink record becomes one of the source
and sink models in `noodl.apps.building_physics.sources`.

## CONTAM `.prj`

`read_prj(path) -> Project` parses a documented subset of TN 1887r1 Appendix A into a `Project`,
building a float64 `Network`, its elements and its drives. `project_to_model(project, *,
ambient=None, species=True, scheme="implicit")` returns `(model, state, drivers)`.

**Supported:** run control (`Project.run_control`, a `RunControl`: the steady simulation's
date and time, the transient run's start and end date and time, time step and output step, day
type, the weather and contaminant file names, and `dodMdt`) and ambient conditions; the simulated species'
records (`Project.species_records`: molar mass, mean diameter, effective density,
diffusivity, default concentration; the decay constant is read but, as in ContamX, not applied)
and each species' default ambient concentration (`Project.ambient_mass_fractions`); kinetic
reactions on zones (`Project.zone_reactions`, the species layer's `kinetics`); day and week
schedules referenced by zones (the zone temperature, on `"T"`) and by paths (a flow multiplier,
through `Scaled` on `"flow_multiplier.<kind>"`) and by sources and sinks (a multiplier of the whole source or sink, on
`"species.source_multiplier"`; for `ccf`, `cut`, `prs`, `dvs` and `drs` records); wind pressure profiles with the
linear or cubic-spline curve fit (`WindProfile(..., curve="linear" | "spline")`; the spline is
periodic in the wind angle, as ContamX evaluates it), and a constant wind pressure on an
envelope path (`"wind_pressure.<kind>"`, positive pushing air from ambient into the zone, as
ContamX); zones and initial concentrations; airflow paths; the power-law element
family
(`plr_orfc/leak1/leak2/leak3/crack/fcn/test1/test2/conn/stair/shaft/qcn`), the quadratic family
(`qfr_*`), doorways (`dor_door`, `dor_pl2`), dampers (`plr_bdq`, `plr_bdf`), constant-flow fans
(`fan_cmf`, `fan_cvf`) and the cubic fan curve (`fan_fan`); source/sink elements `ccf`, `cut`, `eds`,
`brs`, `prs`, `dvs`, `drs`, `plm` and `pkm`, attached by `project_to_model` (scaled by the record's multiplier,
which enters the element's first coefficient, and a `ccf` record's removal coefficient too); filters on airflow paths (`Project.edge_filters`, the species layer's `transmission`):
constant-efficiency (`cef`), particle (`pf0`, interpolated at each species' mean diameter as
ContamX does) and super (`spf`) filter elements; a filter acts on the air crossing its path in
either direction.

**Refused by name**, rather than silently dropped: control nodes and air-handling
systems referenced by nonzero index; boundary-layer diffusion (`bls`), resuspension (`dvr`)
and super (`sup`) source/sink elements, when a source/sink record references one; a source/sink
record whose element or zone the file does not define; gaseous (`gf0`) and UVGI (`uv1`) filter elements,
referenced or not; a filter that names a species the project does not simulate (when a path
references it: an unreferenced filter element is read and unused), a super filter
that contains itself, and a path whose filter number the file does not define; non-trace species; a contaminant list that is not the
simulated species in file order; a source or kinetic reaction that names a species the project
does not simulate (a species the file defines but marks as not simulated included), a kinetic
reaction that gives one product-reactant pair twice, and a zone whose kinetic reaction number the
file does not define; a schedule
number that no week schedule carries; a schedule on a `fan_fan` path; a schedule on an `eds`,
`brs`, `plm` or `pkm` source (CONTAM triggers these once, at the schedule's first nonzero
value); a zone without the variable-pressure flag (which
ContamX holds at constant pressure) or, in a project with species, without the
variable-contaminants flag (ContamX then holds the zone's concentrations at their initial values
and ignores its sources); CFD and 1-D convection/diffusion zones; continuous-values-file
zones and paths; duct networks; a constant wind pressure on an interior path; a fan curve on a
path with `mult != 1`; `csf_*` and `sup_afe` elements; `stackD = 1` (densities varying with
height inside a zone); a wind pressure profile with the trigonometric curve fit; a day schedule whose shape is
neither rectangular nor trapezoidal; zone names with spaces.

Control nodes that no zone and no path references are silently skipped rather than refused: NIST's
own sample projects carry dozens of unreferenced sensor and logger nodes, and those projects must
load.

**`project_to_model` never builds a thermal layer**, because a `.prj` carries no thermal data. A run control that simulates no contaminants (`none`) builds no species layer and binds no source inputs.

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
masses included. When the run control sets **`dodMdt = 1`**, each zone's air-mass change over a
step enters its airflow balance (`simulate` switches the term on by default for such a model),
and ContamX
takes that mass at the zone's absolute pressure even with `densZP = 0`. `project_to_model`
therefore attaches a `ZoneAirMassChange` node source to the air layer (only when the run
control sets `dodMdt` and a zone has volume), which acts only when `simulate` supplies the
step's start mass. For a `dodMdt` project with species it also adds
a `ZonePressureDensity(densities=False)` closure, which scales the species layer's capacity (the
zone air mass) and leaves the densities at `Pb`. Whenever the model carries a
`ZonePressureDensity` closure, `prj.steady(model, state, drivers)` repeats `model.steady` to the
pressure fixed point, and so does each step of `simulate`; they do the same for a closure that
reads the solved pressure (a pressure-driven source); otherwise it is a single call. The
zone-pressure density term is small in the density (|P_zone| / Pb,
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

`simulate_project(project)` runs the project the way ContamX does: its run control (start, end
and time step), its schedules, and the weather and ambient-concentration files it names
(resolved next to the `.prj`). Every input is evaluated at the end of each step, with no step
split at a schedule's jump, and the zone air-mass change enters the airflow balance when the
run control sets `dodMdt = 1`. A steady-airflow run returns the one steady state at the run
control's steady instant. Cyclic contaminant runs, integration other than the trapezoid
method with weight 1 (implicit Euler), steady airflow with transient contaminants, transient
airflow with steady contaminants (ContamX solves those to steady state at every step), and a run
that is not a whole number of time steps are refused by name.

```py
project = read_prj("test_OneZoneWthCtmStack.prj")
out = simulate_project(project)          # the project's run control, schedules, .wth and .ctm
out["path_flow"]                         # (289, 2) kg/s, path-number order
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

