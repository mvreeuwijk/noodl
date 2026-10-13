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
diffusivity, default concentration; the decay constant is read but, as in ContamX, not applied; a nonzero one raises a
`ContamCompatibilityWarning`, which `warnings.filterwarnings("ignore",
category=ContamCompatibilityWarning)` silences)
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
(`fan_cmf`, `fan_cvf`), the cubic-spline elements (`csf_fsp`, `csf_qsp`, `csf_psf`, `csf_psq`; a curve whose fitted spline overshoots and so is not increasing is refused, though ContamX may accept it, because the inversion of the pressure-against-flow forms would be ambiguous), and the cubic fan (`fan_fan`) with its speed schedule (the speed ratio, by the fan laws), its path `mult` (fans in parallel), the rating density and the shut-off orifice below its off ratio; source/sink elements `ccf`, `cut`, `eds`,
`brs`, `prs`, `dvs`, `drs`, `plm` and `pkm`, attached by `project_to_model` (scaled by the record's multiplier,
which enters the element's first coefficient, and a `ccf` record's removal coefficient too); filters on airflow paths (`Project.edge_filters`, the species layer's `transmission`):
constant-efficiency (`cef`), particle (`pf0`, interpolated at each species' mean diameter as
ContamX does) and super (`spf`) filter elements; a filter acts on the air crossing its path in
either direction; duct networks: duct elements `dct_dwc` (ContamX's friction law; its `lam`
field has no effect, as in ContamX), `dct_plr`, `dct_fcn`, `dct_qcn`, `dct_cmf`, `dct_cvf`,
`dct_fan`, `dct_bdq`, `dct_bdf`, `dct_fsp`, `dct_qsp`, `dct_psf`, `dct_psq`; junctions with
their temperatures and initial concentrations and ContamX's volumes (half the length times
the element area of every segment that meets them; a junction without volume, whose segments
all have zero length, holds no air and carries the mix of what flows through it); terminals with their loss and balance
coefficients (the duct-balancing data `bal`, `Fdes` and `Cbmax` is read and, as in ContamX,
has no effect), filters, schedules (the opening, or the junction temperature when the week
schedule's unit type is a temperature) and wind (an ambient terminal whose flags carry the value 1, from
its profile or a constant wind pressure); segments with their dynamic losses, filters and
schedules (a flow multiplier, or a fan's speed ratio); simple air-handling systems
(`Project.closures`, one `AirHandlerFlows` per system, on the edge kind `ahs_<nr>`): their
plenum zones (zero gauge pressure in the airflow solve, `Project.flow_balanced`; storage-free
when their volume is 0, `Project.storage_free`), supply and return points with their design
flows, schedules (on `"flow_multiplier.ahs_<nr>"`) and filters, the recirculation,
outdoor-air and exhaust paths with their filters, the outdoor-air fraction schedule (on
`"outdoor_air_fraction.ahs_<nr>"`) and the minimum outdoor-air flow (both read from the
recirculation path, as ContamX reads them; ContamX finds the implicit paths by their flags,
not by the numbers in the AHS record, which noodl checks against them).

**Refused by name**, rather than silently dropped: control nodes referenced by nonzero
index; boundary-layer diffusion (`bls`), resuspension (`dvr`)
and super (`sup`) source/sink elements, when a source/sink record references one; a source/sink
record whose element or zone the file does not define; gaseous (`gf0`) and UVGI (`uv1`) filter elements,
referenced or not; a filter that names a species the project does not simulate (when a path
references it: an unreferenced filter element is read and unused), a super filter
that contains itself, and a path whose filter number the file does not define; non-trace species; a contaminant list that is not the
simulated species in file order; a source or kinetic reaction that names a species the project
does not simulate (a species the file defines but marks as not simulated included), a kinetic
reaction that gives one product-reactant pair twice, and a zone whose kinetic reaction number the
file does not define; a schedule
number that no week schedule carries; a week schedule that names a day schedule the file
does not define; a schedule on an `eds`,
`brs`, `plm` or `pkm` source (CONTAM triggers these once, at the schedule's first nonzero
value); a zone without the variable-pressure flag (which
ContamX holds at constant pressure), unless an air-handling system names it as a plenum, or,
in a project with species, without the
variable-contaminants flag (ContamX then holds the zone's concentrations at their initial values
and ignores its sources); CFD and 1-D convection/diffusion zones; continuous-values-file
zones and paths; duct elements of any type not listed above; duct leakage (a duct element's
leakage rate); a schedule on a Darcy-Colebrook segment (ContamX ignores it); control nodes,
kinetic reactions or value files on junctions, and control nodes or value files on segments; a
segment, junction or terminal that references a junction, duct element, level or zone the file
never defines; wind data on a terminal that is not
an ambient terminal whose flags carry the value 1; a junction type other than 0 (junction),
1 or 3 (terminal); a non-temperature schedule on a junction that is not a terminal; a junction
that no segment or terminal reaches; initial-concentration rows for a zone or junction the
file does not define; a duct network with `densZP = 1` or `dodMdt = 1` (the junctions' air mass under
either is unmeasured); an air-handling system record that disagrees with its paths' flags and
endpoints, implicit air-handling paths that are not the last paths or not in the order of the
air-handling records (ContamX gives the k-th record the k-th recirculation, outdoor-air and
exhaust path, and mis-assigns them otherwise; this also refuses an implicit path named by two
records), a path that is both an implicit air-handling path and a supply or return point, a
supply or return point without an air-handling system or naming one the file does not
define, or whose path neither leaves that system's supply zone nor enters its return zone, a schedule or a flow on an outdoor-air or exhaust
path or a `mult` on a supply or return point (ContamX ignores them), a flow element or a wind
pressure on an air-handling path, a kinetic reaction in a zero-volume plenum, any other path
or duct terminal joined to a plenum (only the air handler's own flows may touch it), an
air-handling system with `densZP = 1`, and a steady contaminant run with an air handler off
(nothing passing through a plenum); a zone or junction with species rows that no airflow path,
duct segment or terminal reaches (it would have no species state); airflow modes other than steady and transient
(duct balancing, the building airflow and pressurization tests); a constant wind pressure on
an interior path; `sup_afe` super elements (ContamX 3.4.1.7 cannot simulate them); `stackD = 1` (densities varying with
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
air *entering* the path, per direction, matching ContamX section 3.2. The cubic fan (`fan_fan`) runs at the
ratio of its upstream density to its rating density by the fan laws, and its shut-off orifice
takes the upstream density too. The quadratic, damper and constant-flow fan families keep
reference-density coefficients, because they are not part of this application's parity evidence.

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
run control sets `dodMdt = 1`. A step whose airflow solve does not converge from the previous
step's pressures is retried once from the linear initial guess (`diagnostics={}` lists such
steps under `"airflow_retries"`). A steady-airflow run returns the one steady state at the run
control's steady instant. Cyclic contaminant runs, integration other than the trapezoid
method with weight 1 (implicit Euler), steady airflow with transient contaminants, transient
airflow with steady contaminants (ContamX solves those to steady state at every step), and a run
that is not a whole number of time steps are refused by name.

```py
project = read_prj("test_OneZoneWthCtmStack.prj")
out = simulate_project(project)          # the project's run control, schedules, .wth and .ctm
out["path_flow"]                         # (289, 2) kg/s, path-number order
```

A project with a duct network adds its junctions to the network as nodes named
`"junction <nr>"` (`Project.junctions`), after the zones: the species state carries the zones'
rows first and the junctions' after them, and `project.zone_mass_fractions(x)` selects the
zones. `project.terminal_flows(q)` is the flow out of each duct terminal into its zone, in
junction-number order (the order and sign of ContamX's `getDuctTerminalFlow`), and
`project.segment_flows(q)` the flow along each segment in segment-number order.
`simulate_project` and `contamx.run_project` both report `"terminal_flow"`, `(N, n_terminals)`
kg/s; `run_project`'s is empty, `(N, 0)`, without terminals.

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

