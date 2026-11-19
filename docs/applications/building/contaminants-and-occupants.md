# Contaminants, occupants and exposure

How to add contaminant sources, sinks, reactions and filters to a building model, move occupants through it, and report what they breathe, the age of air and air change rates.

## Contaminant sources

CONTAM's source and sink models, all in `noodl.apps.building_physics.sources`:

| Source | Model |
|---|---|
| `ConstantSource(node, G, R=0.0)` | $S = G - R x$; $R$ in kg/s per unit mass fraction |
| `CutoffSource(node, G, x_cut)` | $S = G(1 - x/x_{\text{cut}})$, which removes above the cutoff, as in ContamX |
| `DecayingSource(node, G0, tau, t0=0.0)` | $S = G_0 e^{-(t-t_0)/\tau}$ for $t \ge t_0$ |
| `BurstSource(node, mass, t_burst, dt)` | `mass` delivered over the step(s) covering $[t_{\text{burst}}, t_{\text{burst}} + dt)$ |
| `DepositionVelocitySink(node, velocity, area)` | removal $R = v_d A \rho$: the rate $v_d A / V$ |
| `DepositionRateSink(node, rate)` | removal at the rate $k_d$ (1/s) |
| `PowerLawDecaySource(node, coefficient, exponent, t_p)` | $S = a\,(\max(t, t_p)/1\,\text{h})^{-b}$, $t$ since the start |
| `PeakEmissionSource(node, peak, width, t_peak)` | $S = a \exp(-\tfrac12 (\ln(t/t_p)/b)^2)$ |
| `PressureDrivenSource(node, coefficient, exponent)` | $S = G \max(\Delta P, 0)^n$, $\Delta P$ = ambient static pressure at the zone's height minus the zone's |

A source's concentration-proportional part (the $R$ of `ConstantSource`, $G/x_{\text{cut}}$ of
`CutoffSource`, the two sinks) is handed to the species step as the per-step removal rate
`"species.removal"` (1/s) and held implicitly, as ContamX holds it; `SpeciesSources` divides it
by the zone's air mass, and it replaces the layer's construction-time `removal` (a layer with a
nonzero one is refused). A source on the ambient or on a zone no airflow path touches is refused
by name. A source's schedule multiplies the whole source, sinks included. A
source of your own is a callable `source(t, x_full) -> kg/s per node`, or an object with
`generation(t, x, pressure_difference)` and `removal(t, x, air_mass, density)` methods.
`SpeciesSources(..., heights=net.node_attr("z_ref"), g=...)` supplies the pressure difference;
`simulate` and `prj.steady` repeat a step until the source has seen the step's own pressures,
as ContamX evaluates it.

Place `SpeciesSources` after the closures that write the density and the capacity: it reads
`"species.capacity"` and `"rho_amb"` as they stand when it runs, and a hand-built model that
lists it earlier uses stale values.

`SpeciesSources(sources, species=..., layer=...)` is the closure that adds the sources to the
species layer's `"species.sources"` each step, each on its own species, keeping what a driver
or another closure (an occupant's generation) put there. It evaluates them at
`drivers["time"]` (set by `simulate` to the step's end) and multiplies each by its entry of
`"species.source_multiplier"`, one per source, which is where a schedule or a control acts. A
`BurstSource` releases its mass over the steps that overlap its burst (it reads the step length
from `drivers["time_step"]`, which `simulate` sets), so no step length misses it. The burst's
duration, `dt`, is the run control's time step for a project's sources; a coarser `simulate`
step spreads the same mass over its own length.
`sources_from_project(project, dt=60.0)` builds the source objects from a `.prj`'s own source
records, and `project_to_model` attaches them with `SpeciesSources` automatically, with the
multipliers at one. `assemble_sources(sources, t, x_full)` sums the callable sources outside a
model (a `ConstantSource` or `CutoffSource` at the `x_full` given) and refuses a sink by name.

A `.prj` project's sources and sinks are attached by `project_to_model`. The three-zone
project `sinks_three_zone.prj` has a deposition velocity sink, a scheduled deposition rate
sink, a constant-coefficient source with a removal term, a cutoff source and the two NRCC
emission models. Its species carries a decay constant, which ContamX ignores, so reading it
warns (see [Species and reactions](#species-and-reactions)); the example silences that warning:

```python
import warnings

from noodl.apps.building_physics import ContamCompatibilityWarning, read_prj, simulate_project

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ContamCompatibilityWarning)
    project = read_prj("sinks_three_zone.prj")
out = simulate_project(project)
print([f"{v:.4e}" for v in out["species.x"][-1, :, 0].tolist()])
```

```text
['1.5688e-05', '1.0336e-05', '6.4669e-05']
```

ContamX gives 1.568757e-05, 1.033583e-05 and 6.466919e-05.

## Species and reactions

A `Species` holds CONTAM's species record: name, molar mass, mean diameter (0 for a gas),
effective density, decay constant, diffusivity, default (ambient) mass fraction. The decay
constant is kept as data and not applied, as in ContamX; reading a nonzero one raises a
`ContamCompatibilityWarning` (silence it with `warnings.filterwarnings("ignore",
category=ContamCompatibilityWarning)`). Model a decay with a `DepositionRateSink` in each zone
or with a kinetic reaction, a negative coefficient of the species on itself.

A `KineticReaction` holds first-order reactions between species, `(product, reactant) ->
k` in 1/s, on mass fractions: $\dot x_p \mathrel{+}= k_{pr}\,x_r$. A negative `(s, s)`
coefficient is a decay of `s`. `reaction_kinetics(net, species=..., reactions={zone:
reaction}, ambient=..., flow_kinds=...)` gives the species layer's `kinetics`, which the
transport step integrates implicitly, as ContamX does. A reaction on a zone that no airflow
edge touches is refused by name: the species layer holds no concentration there.

```python
from noodl.apps.building_physics import KineticReaction

fast = KineticReaction({("sarin", "sarin"): -1e-3, ("product", "sarin"): 1e-3})
print(fast.matrix(["sarin", "product"]).tolist())
```

```text
[[-0.001, 0.0], [0.001, 0.0]]
```

`read_prj` reads every species record into `Project.species_records` (the simulated ones, in the
order of `Project.species`) and each zone's kinetic reaction into `Project.zone_reactions`;
`project_to_model` builds the species layer's `kinetics` from them.

## Filters

A filter on an edge removes the fraction *efficiency* of each species from the air passing
through it, in whichever direction the air flows, as in ContamX. `ConstantEfficiencyFilter`
removes a fixed fraction of each species it lists; `ParticleFilter(diameters, efficiencies)`
removes by particle size, interpolating its curve at each species' mean diameter the way
ContamX does (a cubic spline in log-diameter, flat at both ends; a gas passes);
`FilterSeries(stages)` chains filters. `filter_transmission(net, flow_kinds, species,
{edge: filter})` gives the species layer's `transmission` (1 − efficiency per edge and
species). `read_prj` reads a project's constant-efficiency, particle, super and UVGI filters
and the filter on each path (`Project.edge_filters`), and `project_to_model` builds the
`transmission` from them; gaseous filters whose efficiency depends on their loading are
refused by name.

`UVGIFilter` is CONTAM's Penn State ultraviolet germicidal irradiation filter: it inactivates
each species with the efficiency 1 − exp(−k I<sub>t</sub>), k the species'
`uv_susceptibility` (m²/J) and I<sub>t</sub> the UV dose, the design dose
ln(S<sub>design</sub>) / −k<sub>design</sub> scaled by the lamp's output (a polynomial in the
air temperature and velocity, C0..C4 in percent, and optionally a lamp-age factor) and by the
residence time F<sub>design</sub> / F. Because the dose follows the flow, the filter has no
fixed transmission: its `efficiency(species, air)` takes an `AirThroughFilter` (mass flow,
temperature, density and time), and the closure `FilterTransmission` writes the species
layer's per-step `"species.transmission"` driver from the airflow layer's flows. A closure
runs before the airflow solve, so a model with one needs `coupling="iterate"` with the
airflow layer in `iterate_tol` and `iterate_flow_tol`: each pass is fed the previous pass's
flows, and the step converges when the flows the closure read are the flows the pass
produced (two passes, as the airflow does not depend on the contaminants). `Model` refuses
the closure, by name, under any other coupling (it declares `requires_coupling = "iterate"`
and `reads_flows_of`). `project_to_model` builds a project with a UVGI filter that way. Air
crossing the filter against its edge's direction gets ContamX's fixed dose of 1 J/m²
(efficiency 1 − exp(−k)), which `project_to_model` warns about once per filter. The
efficiency is differentiable in every parameter and in the flow; at zero flow it is 1, with a
finite gradient.

```python
from noodl.apps.building_physics import AirThroughFilter, Species, UVGIFilter

lamp = UVGIFilter(design_survival=0.1, design_velocity=2.5, design_mass_flow=0.1,
                  design_rate_constant=0.5, output_coefficients=[50, 0, 1, 0, 0])
bug = Species("bug", uv_susceptibility=0.2)
air = AirThroughFilter(mass_flow=0.1, temperature=293.15, density=1.2041)
print(round(lamp.efficiency([bug], air).item(), 4))
```

```text
0.3834
```

```python
from noodl.apps.building_physics import (
    ConstantEfficiencyFilter, FilterSeries, ParticleFilter, Species,
)

merv = ParticleFilter([0.3e-6, 0.5e-6, 1e-6, 2e-6, 3e-6, 5e-6],
                      [0.1, 0.1, 0.4, 0.65, 0.8, 0.8])
gas = Species("sarin", molar_mass=140.09)
pm = Species("pm", mean_diameter=1.5e-6, effective_density=1000.0)
both = FilterSeries([ConstantEfficiencyFilter({"sarin": 0.3}), merv])
print([round(v, 4) for v in both.efficiency([gas, pm]).tolist()])
```

```text
[0.3, 0.5447]
```

## Occupants

An occupant's movements are an `Occupancy`: one `OccupancyDay(times, zones)` per CONTAM day
type, each listing the zone (a node name, or `None` for out of the building) from each time
of day, 00:00:00 to 24:00:00 (s after midnight). `day_types` picks the occupancy day of each
day type, as for a `WeekSchedule`. The occupant is in a zone from the listed time exactly,
also in the middle of a step, as in ContamX. `Presence(occupancy, net)` is the occupancy as a
dimensionless signal over the nodes: an `Input` on it writes, for each step, the fraction of
the step the occupant spent at each node. A zone that is not a node of the network is
refused by name.

```python
from noodl.apps.building_physics import (
    Calendar, Occupancy, OccupancyDay, Presence, project_to_model, read_prj,
)

model, state, drivers = project_to_model(read_prj("valThreeZonesWthCtm-UseApi.prj"))
visit = OccupancyDay([0.0, 8 * 3600.0, 8 * 3600.0 + 120.0, 17 * 3600.0, 86400.0],
                     [None, "one", "two", None, None])
away = OccupancyDay([0.0, 86400.0], [None, None])
# day type 1 Sunday, 2-6 weekdays, 7 Saturday, 8-12 holidays
occupancy = Occupancy([visit, away], day_types=[2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
presence = Presence(occupancy, model.net, name="visitor")
calendar = Calendar(start_day_type=2)                       # a Monday
step = presence.step_mean(8 * 3600.0, 8 * 3600.0 + 300.0, calendar)
print({n: round(v, 2) for n, v in zip(model.net.nodes, step.tolist()) if v})
```

```text
{'one': 0.4, 'two': 0.6}
```

An `Occupant(occupancy, generation={"CO2": 5.2e-6}, multiplier=2.0, schedules=...,
name="staff")` generates each species at the given rate (kg/s per person, times `multiplier`
persons, times its optional schedule, a signal per species) in the zone it is in. A move in
the middle of a step splits that step's generation by the time spent in each zone, as ContamX
does. Generation reaches the model through the `OccupantGeneration(occupants, layer=...)`
closure on the species layer, which adds it to `"species.sources"` (before or after
`SpeciesSources`, which adds too), and the inputs `occupant_inputs(occupants, net)` gives
`simulate`: each occupant's presence on `"presence.<name>"` and its schedules on
`"generation_multiplier.<name>"`. A zone where the layer holds no air is refused by name when
the closure is built: the ambient, a zone no airflow path reaches, or a storage-free node such
as a zero-volume plenum. Add the closure before the model's first run.

```python
import torch

from noodl.apps.building_physics import (
    Occupancy, OccupancyDay, Occupant, OccupantGeneration, occupant_inputs, project_to_model,
    read_prj, simulate,
)

model, state, drivers = project_to_model(read_prj("valThreeZonesWthCtm-UseApi.prj"))
day = OccupancyDay([0.0, 450.0, 86400.0], ["one", "two", "two"])    # moves at 00:07:30
staff = Occupant(Occupancy([day]), generation={"sarin": 1e-6}, multiplier=2.0, name="staff")
model.closures.append(OccupantGeneration([staff], layer=model.transport["species"]))
out = simulate(model, state, drivers, torch.arange(4, dtype=torch.float64) * 300.0,
               inputs=occupant_inputs([staff], model.net), land_on_breakpoints=False)
print([f"{x:.3e}" for x in out["species.x"][-1, :, 0].tolist()])   # zones one, two, three
```

```text
['2.014e-06', '1.401e-06', '5.376e-08']
```

A `.prj` project's occupants (CONTAM's exposures) are `project.occupants` (`"occupant 1"`,
`"occupant 2"`, ... by exposure number), their inputs are among `project.inputs`, and
`project_to_model` attaches their `OccupantGeneration`; `simulate_project` reports their
exposure as `"exposure"`, `(N - 1, ..., n_occupants, K)`, on the run's own calendar.
`project_calendar(project)` is that calendar: the run control's start, and the day types of
the project's weather file, read next to its `.prj` (a weather file it cannot find there is
refused by name; `weather=` overrides it). A project with occupants needs a transient
contaminant run with at least one simulated species to be simulated, and an occupant who
only breathes is refused by `project_to_model` in a zone where the species layer holds no
air, as one who generates is.

## Outputs

`noodl.apps.building_physics.outputs` turns `simulate`'s histories into what a study
reports. Exposure follows CONTAM's exposure file: `occupant_exposure(history, occupant,
model, calendar=...)` is, for every step, the mean over the step of the mass fraction where
the occupant is, interpolated linearly between the step's start and end values, split at a
mid-step move and zero out of the building; shape `(N - 1, ..., K)`, in kg/kg. The
history may be stacked `(N, ..., n_i, K)` or, for one species, the reduced `(N, ..., n_i)`
that the standard builder's `initial_state` starts from and `simulate` preserves (the result
is then reduced too, `(N - 1, ...)`); `species_history(x, layer)` normalises both.
`calendar` is required: it places t = 0 in the year and so decides the day's occupancy and
the time of every move; pass the run's own. The interpolation is between output rows, so the
result is exact for steps the solver takes whole, as `simulate_project` does; a run with
`land_on_breakpoints=True` splits steps internally at input breakpoints, which its history
does not show. `cumulative_exposure(exposure, times)` is the time integral from the start
(kg/kg s), over the actual step lengths. An occupant whose day schedules name a zone where
the species layer holds no air (the ambient, a zone no airflow path reaches, a storage-free
node) is refused by name, as for its generation.

The inhaled dose and the infection risk are **additions of noodl**, not CONTAM features.
ContamX has no infection model, and CONTAM 3.2 removed the inhalation fields.
`inhaled_dose(integrated_exposure, breathing_rate, density=1.2041)` gives
$p \rho \int x\,dt$ from the cumulative exposure (kg, or quanta for a quanta species);
pass as `density` the air density the species layer's mass fractions refer to (its capacity
over the zone volume; the default is CONTAM's 1.2041 kg/m³).
`infection_probability(dose) = 1 - exp(-dose)` is the Wells-Riley probability in its
quanta-concentration form, $P = 1 - \exp(-p \int c_q\,dt)$, for a species that carries quanta
(generation in quanta/s, an infector being an `Occupant` that generates it).
`wells_riley(I, q, p, Q, t)` is the classic steady single-zone form
$P = 1 - \exp(-I q p t / Q)$. The two agree in a well-mixed zone at steady state. Continuing the run of the previous example, a visitor who follows the staff
member and generates nothing:

```python
from noodl.apps.building_physics import (
    Calendar, Occupancy, Occupant, cumulative_exposure, occupant_exposure, wells_riley,
)

calendar = Calendar()                       # the run above starts 1 January 00:00
visitor = Occupant(Occupancy([day]), name="visitor")
e = occupant_exposure(out, visitor, model, calendar=calendar)
print(f"{cumulative_exposure(e, out['time'])[-1, 0].item():.3e}")    # kg/kg s
print(f"{wells_riley(1, 10 / 3600, 0.5 / 3600, 0.1, 8 * 3600.0).item():.4f}")
```

```text
8.397e-04
0.1052
```

The air change rates and the age of air take the history and the model, and give a value
for every row of the run (or for one state: `prj.steady`'s result is a history of one
instant without the leading axis). They read the airflow layer's flows (`"air.q"`) on the
edges the species layer advects on, and the zones' air mass from `"species.capacity"`, which
`simulate` records whenever a closure writes it: a zone's air mass changes with its
temperature and pressure, and each row's own is used. `capacity=` (kg, the species layer's
interior order) overrides it; a history without it, from a model whose air mass can change,
is refused by name. A capacity given only as a driver to a model run by hand is not seen:
pass it as `capacity=`.

- `local_mean_age(history, model)` is the local mean age of air (s) of every interior row of
  the species layer, as ContamX writes it to its `.age` file (in hours there). noodl solves
  the steady age equation $\sum_{in} q\,\tau_{up} - \sum_{out} q\,\tau + M = 0$ as a transport
  problem with a uniform source of one second per second in every kilogram of air, the
  outdoor air at age zero. A zero-volume plenum holds no air: its age is the flow-weighted
  mix of the ages flowing in. A zone without airflow, or zones that exchange air only among
  themselves, have no age: refused by name with `UndefinedSteadyState`.
- `building_air_change_rate(history, model, conditioned=...)` is the building air change
  rate (1/h), as ContamX writes it to its `.ach` file: every flow entering the conditioned
  space from outside it, over 1.2041 kg/m³ times the space's volume. ContamX counts in that
  space every zone without its "outside the building volume" flag and the whole mechanical
  system: the duct junctions (their duct segments' volume) and the air handlers' plenums. Air
  a supply terminal or an air handler delivers to a zone then moves within the space, and what
  enters is the outdoor air drawn in through the envelope (the `.ach` file's `path` column),
  at an ambient duct terminal or an air handler's intake (its `duct` column). Only the
  total is returned, not the path/duct split. The default space is every interior row of
  the species layer except the zones the CONTAM reader marks as outside the building volume
  (the node attribute `in_building_volume`), which for a `.prj` project is
  `project.building_zones`.
- `zone_air_change_rate(history, model)` returns `(rate, zones)`: each zone's inflow from the
  ambient and from other zones alike over its air mass (1/h), the inverse of its nominal time
  constant, not an outdoor air change rate. It is an **addition of noodl**: ContamX writes no
  zone rate. `zones` names the last axis; a storage-free plenum, with no air to change, is
  left out. `local_mean_age`, by contrast, returns every interior row, plenums included.

Continuing the run above, whose three zones of 300, 600 and 900 m³ are in series:

```python
from noodl.apps.building_physics import (
    building_air_change_rate, local_mean_age, zone_air_change_rate,
)

age = local_mean_age(out, model)                          # s, (N, n_i): every row of the run
print([f"{a / 3600:.3f}" for a in age[-1].tolist()])     # h, zones one, two, three
rate, zones = zone_air_change_rate(out, model)            # 1/h, (N, n_z)
print(zones, [f"{r:.3f}" for r in rate[-1].tolist()])
print(f"{building_air_change_rate(out, model)[-1].item():.4f}")   # 1/h
```

```text
['1.000', '3.000', '6.000']
['one', 'two', 'three'] ['1.000', '0.500', '0.333']
0.1667
```

On a `.prj` project the same functions read `simulate_project`'s result. The fixture
`occupants_three_zone.prj` has two occupants, one of them generating on a schedule and
moving mid-step, and its zone three is outside the building volume, so the building air
change rate counts zones one and two (`project.building_zones`, the default). ContamX's own
result files for this project give cumulative exposures of 0.37964 and 0.13995 kg/kg s,
ages of 0.9044, 2.775 and 5.680 h, and an air change rate of 0.3564 /h at the end of the
run:

```python
from noodl.apps.building_physics import (
    building_air_change_rate,
    cumulative_exposure,
    local_mean_age,
    project_to_model,
    read_prj,
    simulate_project,
)

project = read_prj("occupants_three_zone.prj")
out = simulate_project(project)
model, _, _ = project_to_model(project)
total = cumulative_exposure(out["exposure"], out["time"])[-1, :, 0]   # kg/kg s, per occupant
print([f"{v:.2e}" for v in total.tolist()])
print([f"{v:.3g}" for v in (local_mean_age(out, model)[-1] / 3600).tolist()])   # h
print(f"{building_air_change_rate(out, model)[-1].item():.3g}")                 # 1/h
```

```text
['3.80e-01', '1.40e-01']
['0.904', '2.77', '5.68']
0.356
```
