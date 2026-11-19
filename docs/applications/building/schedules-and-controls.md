# Schedules, time and controls

How to drive a building model over time, from schedules, weather and ambient files, and how to close a loop with a control network.

## Time, schedules and inputs

Time in a building run is seconds since the start. A `Calendar` says where that start sits:
day of the year, time of day, and the CONTAM *day type* (1–12) of each day, which picks one
of a week schedule's twelve day schedules. Day types come from the weather file when there
is one. Otherwise they start at a given type and cycle through 1–7, as ContamX does.

A schedule's value is taken at the **end** of each step, and where it jumps, the value *at*
the jump is the one before it. A fan switched on at 08:00 is therefore off over the step
ending 08:00 and on over the next one. This matches ContamX 3.4 (measured); at the start
time itself the value at that instant is used.

```python
from noodl.apps.building_physics import Calendar, DaySchedule, WeekSchedule

office = DaySchedule([0.0, 8 * 3600.0, 18 * 3600.0, 86400.0], [0.0, 1.0, 0.0, 0.0])
closed = DaySchedule([0.0, 86400.0], [0.0, 0.0])
# day type 1 Sunday, 2-6 weekdays, 7 Saturday, 8-12 holidays
week = WeekSchedule([office, closed], day_types=[2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
calendar = Calendar(start_day=1, start_day_type=2)        # 1 January is a Monday
times = (8 * 3600.0, 9 * 3600.0, 6 * 86400 + 9 * 3600.0)
print([week.value(t, calendar).item() for t in times])
```

```text
[0.0, 1.0, 0.0]
```

`TimeSeries(times, values, interpolation="linear" | "step")` is the same kind of input built
from data (seconds since the start, never extrapolated); its values may carry gradients.

### Running over time

An `Input(signal, key, at=... | index=...)` writes a signal's value into a driver: the whole
driver, the entries named by `at` (node names for the node temperatures `"T"`, source names,
edge labels), or positions `index`. `simulate(model, state, drivers, times, inputs=...)`
evaluates every input at the **end** of each step, sets `drivers["time"]`, and steps the model.
It returns the time histories at `times`, which start at 0 (a `Calendar` places that start
in the year). Steps are split at every jump or kink of an input unless
`land_on_breakpoints=False`. Each step's airflow solve starts from the previous step's
pressures; when it does not converge from there (a fan curve after a large speed change can
swing between the ends of its curve, where its slope is zero), that solve is retried from
the linear initial guess, and a retry that fails too raises, naming the step. The retry is
the air layer's own fallback ladder (`KirchhoffLayer(fallback=("linear_guess",))`, which
`project_to_model` and `build_model` set; see
[When Newton fails](../../concepts/solvers.md#when-newton-fails-the-fallback-ladder)): it
solves only the failed airflow solve again, inside its pass, not the whole step.
`simulate(..., diagnostics={})` lists the end times of the steps with a retried solve under
`"airflow_retries"`. A signal with a step mean, an occupant's `Presence` (see
[Occupants](contaminants-and-occupants.md#occupants)), is written as its mean over each step (or sub-step) instead of its
value at the step's end. The histories have one row per output time either way.

```python
import torch
from noodl.apps.building_physics import (
    Input, TimeSeries, project_to_model, read_prj, simulate,
)

project = read_prj("valThreeZonesWthCtm-UseApi.prj")
model, state, drivers = project_to_model(project)
cold_snap = TimeSeries([0.0, 1800.0, 3600.0], [293.15, 273.15, 273.15])
out = simulate(model, state, drivers, torch.arange(13, dtype=torch.float64) * 300.0,
               inputs=[Input(cold_snap, "T", at="ambient")])
print(out["air.q"].shape, out["species.x"].shape)
```

```text
torch.Size([13, 4]) torch.Size([13, 3, 1])
```

`simulate` adds each zone's air-mass change to its airflow balance (CONTAM's `dodMdt`)
exactly when the model carries a `ZoneAirMassChange` storage, which `project_to_model`
attaches for a project that sets `dodMdt` (the default `air_mass_change=None`;
`air_mass_change=False` opts out, and `True` on a model without the source is refused): a zone whose temperature rises pushes out the air it no longer holds. The mass is
taken at the zone's absolute pressure, (P_b + P_zone) V / (R T), as ContamX's is, so a zone
depressurised by an exhaust fan switching on also gives up a little air. For a project that
sets `dodMdt`, `project_to_model` gives the contaminants that same zone air mass. The term is
part of the airflow solve itself (the `ZoneAirMassChange` storage `project_to_model`
attaches), so a step of any length converges. When the model carries a `ZonePressureDensity`
closure (CONTAM's `densZP`), `simulate` repeats each step with the latest zone pressures until
they no longer move; the same repetition runs when a pressure-driven source is present, so the
source sees the step's own pressures.

`simulate_project(project)` runs a `.prj` the way ContamX does: its run control, its
schedules, and its weather and ambient files, read from next to the project.

```python
from noodl.apps.building_physics import simulate_project

project = read_prj("scheduled_three_zone.prj")
out = simulate_project(project)
fan = out["path_flow"][:, 4]
print(out["time"].shape, f"fan at 07:55 {fan[95].item():.6f} kg/s, at 08:05 {fan[97].item():.6f} kg/s")
```

```text
torch.Size([289]) fan at 07:55 0.000000 kg/s, at 08:05 0.100342 kg/s
```

A weather file and an ambient-concentration file become inputs the same way:

```python
from noodl.apps.building_physics import Calendar, read_ctm, read_wth

weather = read_wth("test_OneZoneWthCtm.wth")
print(weather.day_types, weather.signal("Ta").value(6 * 3600.0 + 300.0, Calendar()).item())
outdoor = read_ctm("test_OneZoneWthCtm.ctm")
print(outdoor.species, outdoor.signal(["sarin"]).value(15 * 3600.0, Calendar()).tolist())
```

```text
{1: 4, 2: 5} 293.15
('sarin',) [0.0]
```

## Controls

A `ControlNetwork` reads the model's state, computes signals and writes them into drivers:
a fan's flow multiplier, a damper's opening, a heat input. It is built from three kinds of
part:

- a **reading** names one value of the state or the drivers: `StateReading(key, index)`,
  `DriverReading(key, index)` or `FunctionReading(fn)`. The sensors by name name what they
  read instead: `node_mass_fraction(node, species)` (kg/kg), `node_temperature(node)` (K),
  `node_pressure(node)` (gauge, Pa), `edge_flow(kind, position)` (kg/s) and
  `edge_pressure_difference(kind, position)` (Pa). `build_model(..., controls=network)`
  resolves them against the model it builds, so they read that model's own rows and
  columns; a name the model lacks is refused there, and so is a network already resolved
  for another model (build one network per model). A temperature is the thermal layer's
  state, a boundary node's boundary temperature, or, without a thermal layer, the
  prescribed temperature driver `"T"`, which the model must then declare (a `.prj` model
  does; `node_temperature` on a model with neither is refused);
- a **block** computes one output from the outputs of the blocks it names by position in the
  network's list, or from a reading (`Sensor`);
- an **actuator**, `Actuator(block, key, index=None, mode="set" | "add")`, writes a block's
  output into a driver: the whole driver, or the positions `index` on its last axis; `"add"`
  adds it to those entries (a heat flow into `"thermal.sources"`, W).

| Block | Output |
|---|---|
| `Constant(value)` | the value |
| `DriverValue(key, index)` | a driver's value at this evaluation, such as a schedule an `Input` writes into `"controls.input"` |
| `Identity(a)` | its input |
| `Sensor(reading, offset, scale, time_constant)` | `(value - offset) * scale`, lagged with `time_constant` (s) when it is > 0 |
| `Sum(*inputs, weights)` | `sum_i w_i u_i` |
| `Average`, `Maximum`, `Minimum`, `Product` | the mean, largest, smallest, product of its inputs |
| `Divide(numerator, denominator)` | the quotient; a zero denominator raises, naming the block |
| `Absolute(a)`, `Complement(a)` | `abs(u)`, `1 - u` |
| `Positive(a)` | 1 where `u > 0`, else 0 |
| `Affine(a, offset, gain)` | `(u - offset) * gain` |
| `And`, `Or`, `ExclusiveOr` | logic on `u > 0` |
| `BelowLimit(a, limit)`, `AboveLimit(a, limit)` | 1 where `u < limit` (or `u > limit`), else 0 |
| `Shortfall(a, limit)`, `Excess(a, limit)` | `max(0, limit - u)`, `max(0, u - limit)` |
| `LowerBandSwitch(a, limit, band)` | on above `limit + band`, off below `limit`, unchanged in between |
| `UpperBandSwitch(a, limit, band)` | on below `limit - band`, off above `limit`, unchanged in between |
| `Hysteresis(a, slack, slope, initial)` | ContamX's hysteresis between a rising and a falling line |
| `ProportionalControl(a, gain)` | `clip(gain u, 0, 1)` on an error `u` |
| `IncrementalPI(a, kp, ki)` | ContamX's PI controller: `y = clip(y_prev + kp (e - e_prev) + ki dt (e + e_prev)/2, 0, 1)` |
| `LimitedPID(setpoint, measurement, gain, y_max, y_min, integral_time)` | MSL's `LimPID` as P or PI: `clamp(gain (u_s - u_m + xi), y_min, y_max)` with anti-windup |

A network is **sampled** (the default) or **continuous** (`sampled=False`). A sampled network
is evaluated once per step, as ContamX evaluates its control nodes; the switches with memory,
the hysteresis, the incremental PI and a lagged sensor need one. A continuous network is
algebraic in the state at every call, and its integrators (`LimitedPID` with an
`integral_time`) advance by one explicit Euler step of their rate per step; `LimitedPID` needs
one. That step holds the controller's output over the whole model step, so a stiff loop
(a large gain on a fast zone) run with long steps can overshoot or oscillate where the
continuous controller would not: take steps short against the loop's time constant.

A block's parameters are differentiable: a gain, a limit, an offset, a weight, a band or a
constant's value may be a float64 tensor, and the outputs carry its gradient as they carry
those of the state and the drivers (`IncrementalPI(0, kp=torch.tensor(2.0,
requires_grad=True), ki=2e-3)` in the example below gives d(CO2)/d(kp) through `simulate`).
A float stays a float. A band or a switch's threshold only moves where the switch flips, so
the gradient through a switch is zero wherever it does not flip; a hysteresis's slope and
slack and a controller's limits enter its output directly and have gradients on the pieces
where they act. A `LimitedPID` output exactly at a limit is on the unlimited piece: it
carries the gradient of the unlimited output and the limit gets none, for a float limit
and a tensor one alike.

The network is a closure of the model, the first one: `build_model(..., controls=network)`
puts it ahead of the density closure, so its writes reach every other closure. Its block
outputs, its blocks' memory and whether it has been evaluated travel in the state as
`"controls.outputs"`, `"controls.memory"` and `"controls.evaluated"`, which
`build_model(..., return_inputs=True)` and `project_to_model` seed. The seed says "not
evaluated" (and its outputs are NaN): until the network is first evaluated it writes nothing,
so a query or `Model.check` on the initial state sees the drivers as the inputs leave them.
In a batch, a member not evaluated yet keeps its drivers the same way.
Inputs are applied before the closures run, so a control overrides a schedule on the same
driver entry, as in CONTAM.

Timing in a sampled network follows ContamX:

- sensors read the state the step starts from, so the value acting over a step comes from
  the end of the previous one;
- a flow or pressure sensor reads the last airflow solve, and 0 before the first;
- a sensor of a driver (a prescribed temperature, an outdoor mass fraction) reads the value
  the previous step used; the first evaluation reads the current value;
- a block that reads a block later in the list gets that block's previous output; before
  the first evaluation it gets 0, or a `Constant`'s value (ContamX presets its constant
  nodes). The network evaluates in the order given and never reorders.

`simulate` evaluates a sampled network once before the initial airflow, with the first
step's length, so a control acts in the first row already while its flow and pressure
sensors read 0 there. It then evaluates it once at the start of every output step, with the
inputs at the step's end, and holds it over the step's zone-pressure passes and the
sub-steps an input breakpoint makes; a sensor of `"air.phi"` therefore reads the pressures
that ended the previous step. The edges' pressure differences are recorded as `"air.dp"`
after every airflow solve when a sensor reads them, a controlled zone temperature enters the
zone air-mass change (`air_mass_change`) as a scheduled one does, and the histories carry
the block outputs as `"controls.outputs"`.

A room of 60 m³ with a CO2 source (4e-5 kg/s) takes its air through a leak and loses it
through an exhaust fan of 0.1 kg/s, wrapped in `Scaled` so that a driver scales its flow. A PI
controller on the room's CO2 above 1.2 g/kg sets the fan's multiplier:

```python
import torch
from noodl.apps.building_physics import (
    Actuator, ControlNetwork, IncrementalPI, Sensor, Zone, add_zone, build_model,
    node_mass_fraction, orifice_law_from_edges, simulate,
)
from noodl.elements import FixedFlow, Scaled
from noodl.heads import Stack
from noodl.topology import Network

F64 = torch.float64
net = Network(dtype=F64)
net.add_node("ambient", z_ref=0.0)
add_zone(net, Zone("room", volume=60.0, T0=293.15))
net.add_edge("ambient", "room", kind="airpath", z_path=1.0, Cd=0.6, area=0.05)
net.add_edge("room", "ambient", kind="fan")
fan = Scaled(FixedFlow(torch.tensor([0.1], dtype=F64), kind="fan"), "flow_multiplier.fan")

network = ControlNetwork(
    [Sensor(node_mass_fraction("room", "CO2"),     # CO2 above 1.2 g/kg, in g/kg
            offset=1.2e-3, scale=1e3),
     IncrementalPI(0, kp=2.0, ki=2e-3)],           # the fan multiplier, 0..1
    [Actuator(1, "flow_multiplier.fan")])
model, state, drivers = build_model(
    net, air_laws=[orifice_law_from_edges(net, "airpath"), fan],
    heads=[Stack.from_network(net, "airpath")], species=("CO2",),
    controls=network, return_inputs=True)
state["species.x"] = torch.tensor([6e-4], dtype=F64)
drivers["species.x_boundary"] = torch.tensor([6e-4], dtype=F64)  # outdoor CO2, kg/kg
drivers["species.sources"] = torch.tensor([0.0, 4e-5], dtype=F64)  # kg/s in the room

times = torch.arange(13, dtype=F64) * 300.0
out = simulate(model, state, drivers, times)
q_fan = out["air.q"][:, model.kirchhoff["air"].kind_slice("fan")][:, 0]
signal = out["controls.outputs"][:, 1]                            # the PI's output
for k in range(len(times)):
    print(f"{times[k] / 60:4.0f} min  CO2 {out['species.x'][k, 0] * 1e3:.4f} g/kg  "
          f"PI {signal[k]:.4f}  fan {q_fan[k]:.4f} kg/s")
```

```text
   0 min  CO2 0.6000 g/kg  PI 0.0000  fan 0.0000 kg/s
   5 min  CO2 0.7661 g/kg  PI 0.0000  fan 0.0000 kg/s
  10 min  CO2 0.9292 g/kg  PI 0.0220  fan 0.0022 kg/s
  15 min  CO2 1.0687 g/kg  PI 0.1368  fan 0.0137 kg/s
  20 min  CO2 1.1655 g/kg  PI 0.2951  fan 0.0295 kg/s
  25 min  CO2 1.2188 g/kg  PI 0.4390  fan 0.0439 kg/s
  30 min  CO2 1.2409 g/kg  PI 0.5409  fan 0.0541 kg/s
  35 min  CO2 1.2454 g/kg  PI 0.6031  fan 0.0603 kg/s
  40 min  CO2 1.2416 g/kg  PI 0.6379  fan 0.0638 kg/s
  45 min  CO2 1.2347 g/kg  PI 0.6563  fan 0.0656 kg/s
  50 min  CO2 1.2274 g/kg  PI 0.6654  fan 0.0665 kg/s
  55 min  CO2 1.2209 g/kg  PI 0.6695  fan 0.0670 kg/s
  60 min  CO2 1.2155 g/kg  PI 0.6710  fan 0.0671 kg/s
```

Each row's PI output is the one its fan flow used: at 0 min the evaluation before the
initial airflow, later the one decided from the state the step ending there started from.
The fan starts before the room reaches the set point: the proportional part answers the
rising CO2 while the integral part still pulls down. It settles where it removes the source:
4e-5 kg/s over the set point's 0.6 g/kg above the outdoor air is 0.067 kg/s.

**From a `.prj`.** `read_prj` reads a project's control nodes (`Project.control_nodes`) and
plans, when some record references a control node, the ones that a zone's, path's,
source's, duct segment's, terminal's or air-handler path's control reference, or a named
report node, needs (`Project.control_plan`), in the file's sequence order (a project whose
control nodes only report plans none: it runs as without them); `project_to_model` builds them into a new sampled network, first among
the closures, for every model. A control reference acts on what a schedule on the same record
acts on (a zone's `"T"`, a path's flow multiplier or fan speed, a source's multiplier, a
terminal's opening, an air handler's flows), and a schedule node is an `Input` on
`"controls.input"`. `project.control_outputs(out["controls.outputs"])` picks the named report
nodes, ContamX's control outputs, from a history; `simulate_project` returns them as
`"control"` (when there is a plan). `dcv_three_zone.prj` is demand-controlled ventilation:
a PI controller on zone one's CO2 above 650 ppm (by mass) drives an exhaust fan, never below
0.2 of its flow, while the occupants are in from 08:00 to 18:00:

```python
from noodl.apps.building_physics import read_prj, simulate_project

project = read_prj("dcv_three_zone.prj")
print([name for name, _ in project.control_plan.outputs])
out = simulate_project(project)
for k in (96, 108, 120, 144, 216, 240):                 # 300 s steps from midnight
    co2, pi, fan, excess = out["control"][k].tolist()
    print(f"{k * 5 // 60:02d}:00  CO2 {co2:6.1f} ppm  PI {pi:.3f}  fan {fan:.3f}")
```

```text
['co2', 'pi', 'fan', 'excess']
08:00  CO2  600.0 ppm  PI 0.000  fan 0.200
09:00  CO2  644.1 ppm  PI 0.003  fan 0.200
10:00  CO2  660.6 ppm  PI 0.313  fan 0.313
12:00  CO2  653.7 ppm  PI 0.846  fan 0.846
18:00  CO2  650.0 ppm  PI 0.898  fan 0.898
20:00  CO2  605.2 ppm  PI 0.000  fan 0.200
```

The fan runs at its 0.2 floor until the CO2 nears the set point, around 09:00; the PI
output then rises until the exhaust holds zone one at 650 ppm, and falls back to 0 once the
occupants leave.

**From Modelica.** `read_modelica` reads a `LimPID` whose measurement is a temperature sensor
on a volume, and the Math blocks between it and a `PrescribedHeatFlow`, into a continuous
network: a `LimitedPID` (a P controller; the integral and derivative parts are refused by
name) and `Affine`/`Sum`/`Product`/`Divide` blocks. Its heat flow does not go through an
actuator. On each piece of the limiter it is affine in the zone temperature, `A + B T`, and
the closure `FeedbackHeat` hands it to the thermal layer as the source `A` and the removal
rate `-B/C`, so that each transport step treats it implicitly: the MBL examples' loop, 15000
W/K on a 75 kJ/K room, settles in 5 s, shorter than their steps. `simulate(...,
scheme="midpoint")` takes both halves of a step on the step's piece and splits a step where the
limiter reaches a limit. A removal rate you pass as the driver `"thermal.removal"` (1/s) adds to
the network's in either scheme, each half of a midpoint step taking the driver's value at its own
end. `names.controls` maps each block to its column of `simulate`'s
`"controls.outputs"` (the block outputs, under the same key as the Python `simulate`). The
blocks are the network's (`FeedbackHeat.network.blocks`, the closure among
`model.closures`): a `LimitedPID`'s `gain` or limits set to a tensor carry their gradient
into the heat flows and the zone's energy balance. MBL's
`ChimneyShaftWithVolume` holds its room at 293.15 K while its volumes release their start
imbalance:

```python
from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import simulate

model, state, drivers, names = read_modelica("ChimneyShaftWithVolume.json", return_names=True)
out = simulate(model, state, drivers, names.times[:3])
print(names.controls)
heat = out["controls.outputs"][:, names.controls["gain"]]  # the heat into the room, W
room = out["T"][:, names.nodes["roo"]]
for k in range(3):
    print(f"{float(names.times[k]):4.1f} s  room {float(room[k]):.4f} K  heat {float(heat[k]):6.1f} W")
```

```text
{'con': 2, 'gain': 3}
 0.0 s  room 293.1500 K  heat    0.0 W
 7.2 s  room 293.1558 K  heat  -86.9 W
14.4 s  room 293.1514 K  heat  -20.9 W
```

The compressed air warms the room at first, and the controller cools it back (MBL's room is
at 293.1558 K at 7.2 s too).
