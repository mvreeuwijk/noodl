# Importing Modelica Buildings Library models

`read_modelica` turns multizone airflow models of the Modelica Buildings Library (MBL) into a [building physics](index.md) model. Reading a Modelica export produces a model whose power-law, tabulated and door components become the [airflow elements](elements.md) of the same physics: an open door, for example, becomes an `OpenDoor` edge pair and a discretised door a `DoorPortStream` pair (its two port streams), while zonal flows are assembled by the importer itself as prescribed-flow edge pairs.

The same pipeline also reads electrical circuits of the Modelica Standard Library and of MBL's `Buildings.Electrical`; that reader is described under [Electrical circuits](../electrical/index.md).

## How to load

`read_modelica(path) -> (model, state, drivers)` imports a multizone airflow model built with MBL v13.0.0 (commit `55abf579598ca81cae0a82f337350375958e6722`) through an OpenModelica JSON export. The reader never parses Modelica source and never evaluates a Modelica expression: every number in the JSON was already evaluated by OpenModelica.

```py
from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import simulate

model, state, drivers, names = read_modelica("OneRoom.json", return_names=True)
result = simulate(model, state, drivers, names.times)

print(result["T"][-1])       # zone temperatures (K) at the last grid time, node order names.nodes
print(result["air.q"][-1])   # edge flows (kg/s); names.edges maps an MBL instance to its columns
```

### The pipeline

The reader follows these steps, which it shares with the circuit reader (`noodl.modelica`):

1. **Export.** A script run next to OpenModelica (`scripts/modelica_export.py`) writes one JSON document per model from
   OpenModelica's evaluated instance tree, every parameter already a number, and the
   reference trajectories as a CSV.
2. **Load.** The reader's `schema.load` checks the document's shape. The signal record, the
   structural checks and the reading of Boolean parameters are shared: a Boolean parameter
   the export does not hold takes its Modelica default, and any value other than `true` or
   `false` is refused.
3. **Resolve.** The components are joined into nodes by their `connect()` pairs , and each class is supported or refused by name.
4. **Signals.** The `Modelica.Blocks.Sources` and `Modelica.Blocks.Math` blocks are
   evaluated on the time grid as MSL 4.1.0 defines them (`noodl.modelica.signals`), and each
   input takes its value from the signal that drives it (`noodl.modelica.inputs`).
5. **Time grid.** The output grid of the document's `experiment`, plus every event of the
   signals strictly inside the run, each interval split into `substeps` equal steps
   (`noodl.modelica.timeline`); `extrapolate` combines two runs at `substeps` `r` and `2r` by Richardson extrapolation.
6. **Refuse or build.** Every problem found is gathered and raised as one
   `ModelicaImportError`, never a partial model; otherwise the reader returns the model, its
   start state, its drivers on the grid and the names that map the reference's columns to
   noodl quantities.

### Reproducing the export

This is needed only to add models: the fixtures shipped with noodl need none of it. It runs in WSL. Tested on
Ubuntu 22.04 (`jammy`) in WSL with OpenModelica 1.27.1. Install OpenModelica from its own apt
repository (needs `sudo`, done once by whoever administers the WSL environment):

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL http://build.openmodelica.org/apt/openmodelica.asc | sudo gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] https://build.openmodelica.org/apt jammy stable" | sudo tee /etc/apt/sources.list.d/openmodelica.list
sudo apt-get update && sudo apt-get install -y omc
```

MBL v13 needs the Modelica Standard Library (MSL) 4.1.0, installed once, without `sudo`, with
`omc`'s own `installPackage` (it writes into `~/.openmodelica/libraries`, not a system path):

```bash
echo 'installPackage(Modelica, "4.1.0");' > /tmp/install_msl.mos && omc /tmp/install_msl.mos
```

Clone MBL at the tag this release was exported against:

```bash
git clone --depth 1 --branch v13.0.0 https://github.com/lbl-srg/modelica-buildings.git ~/modelica/modelica-buildings
```

(`git rev-parse HEAD` there is `55abf579598ca81cae0a82f337350375958e6722`, the commit recorded
in every fixture's JSON.) Then, with `omc` on the WSL `PATH` (this release was exported against
`OpenModelica 1.27.1~2-g6db4671`):

```bash
python3 scripts/modelica_export.py Buildings.Airflow.Multizone.Validation.ThreeRoomsContam --out exports
```

(`<Model>` is the model's fully qualified name; the exporter writes `<Short>.json`/`<Short>.csv`
under `<Short>`, its last component — `ThreeRoomsContam.json`/`ThreeRoomsContam.csv` here.) This
writes `exports/<Short>.json` (the component graph) and `<Short>.csv` (OpenModelica's
own simulated reference) as the fixtures. `scripts/modelica_export.py` is not imported by
the package and is not run by the test suite. It exits non-zero if the simulation fails (the
JSON is still written, with the instance API's Reals instead of the simulated values, so the
export can be inspected); a batch script over every model should check the exit code rather
than assume success. `--tolerance` simulates at a solver tolerance other than the model's
declared one; all fifteen dynamic parity references (and those of `PressurizationData` and
`TrickleVent`, imported but not compared) were exported with `--tolerance 1e-13`
(the two chimney models and `TrickleVent` as described below). At 1e-14 OpenModelica's DASSL returns
no trajectory for these models, so 1e-13 is the finest reference available.

A model with a `ReaderTMY3` also gets its weather file: the exporter copies the `.mos` file
`filNam` names (from the MBL checkout) next to the JSON and records its name and sha256 in the
reader's `"files"` entry. The reader opens that copy and refuses the model, naming the reader,
if the file is missing or its sha256 differs. The CSV of such a model also holds the reader's
`weaBus.TDryBul`, `weaBus.pAtm`, `weaBus.relHum` and `weaBus.winSpe` and every wind boundary's
`pWin`, which the verification compares at round-off.

The exporter keeps a connection from a component's output into a signal block's input (a
temperature sensor's `T` feeding a controller's `u_m`), so that the reader can wire the sensor
into the block or refuse the connection by name. The CSV of a model with a feedback controller
holds its output `<controller>.y`, and that of a model with a `MediumColumnDynamic` the state
of its volume, `<column>.vol.T`, `.p`, `.Xi[1]` and `.C[k]`. `ChimneyShaftNoVolume`,
`ChimneyShaftWithVolume` and `TrickleVent` were re-exported this way at `--tolerance 1e-13`.

The fan and zero-resistance paths are compared with OpenModelica on five models of noodl's
own package `NoodlFanShort.mo` (`FanOrifice`, `FanRamp`, `FanAdverse`,
`ShortConnections`, `FanShortParallel`: a `FlowControlled_dp` with a constant head, a ramped `dp_in`
and a back pressure that reverses the flow; a `LosslessPipe` and a `PressureDrop` at
`dp_nominal = 0` in series; a fan through a lossless pipe beside a leak). They are exported with
`scripts/modelica_export.py NoodlFanShort.<Model> --package <the .mo file> --mbl <mbl>`, and are
algebraic models in the parity set, at the same tolerance as the others.
Three dynamic models cover a fan with volume mass storage (`FanVolumes`, `FanRampVolumes`,
`FanLosslessVolumes`: two `MixingVolume`s of different size joined by a fan and each leaking to an
ambient, with a constant head and a start that meets the rise; the same with `dp_in` ramped from
0 to 20 Pa; a fan into one volume through a `LosslessPipe`). They are in the dynamic set, at the
reference tolerance of 1e-13, with no resolution allowance: the full runs agree with
OpenModelica to 2.5e-7 relative at (1, 2) substeps (the t = 0 row, where the fan flow is set
by the reader, to 3.6e-8).

## What is read

### Supported models

22 models: 8 `Validation` + 13 `Examples` of `Buildings.Airflow.Multizone`, and
`Buildings.Fluid.Sources.Examples.Outside_Cp`. `Buildings.Airflow.Multizone`
elements, `FlowControlled_dp` movers (see [Movers](#movers)), `MixingVolume` zones, pressure/temperature boundaries, trace substances, the outside
boundaries `Outside`, `Outside_Cp`, `Outside_CpLowRise` and `Outside_CpData` and the weather
reader `ReaderTMY3` (see [Weather and wind](#weather-and-wind)) — not `Buildings.ThermalZones`, HVAC or
district networks. `Validation`: `OneWayFlow`,
`DoorOpenClosed`, `OpenDoorPressure`, `OpenDoorTemperature`, `ThreeRoomsContam`,
`ThreeRoomsContamDiscretizedDoor`, `OpenDoorBuoyancyDynamic`, `OpenDoorBuoyancyPressureDynamic`.
`Examples`: `ChimneyShaftNoVolume`, `ChimneyShaftWithVolume`, `CO2TransportStep`, `ClosedDoors`, `NaturalVentilation`,
`OneEffectiveAirLeakageArea`, `OneOpenDoor`, `OneRoom`, `Orifice`, `PowerLaw`, `ReverseBuoyancy`,
`ReverseBuoyancy3Zones`, `ZonalFlow`. The weather and wind pressure of the `Buildings.Fluid.Sources` examples
`Outside_CpLowRise`, `Outside_CpData`, `Outside_CpData_Angles` and
`Validation.Outside_CpData_Directions` (boundaries with no flow element, so not importable as
models) are compared with `assemble.weather_columns`.
`MediumColumnDynamic` is read as MBL builds it: a mixing volume between two half columns, the
top one taking its density from below and the bottom one from above, both the volume's
(`use_HeatTransfer = false` only). Its volume is named `<column>.vol` in `names.nodes`, as in
MBL's results. A `TemperatureSensor` on a volume's `heatPort` reads that volume's
temperature; its `T` may feed a signal block's input.

### Flow elements in series

 One-way flow elements may be wired in series, joined through
columns or directly (`ChimneyShaftNoVolume`: two orifices at the ends of one shaft with no
volume in it). The columns between two elements go with the first in name order; the joint is
a junction without storage, a node of the air layer (`names.junctions`, also in
`names.nodes`), where the elements' flows balance. Every element, column and junction passes
the stream on unchanged, so the transport of heat and species takes the whole run as one edge
from one end to the other; the junction's temperature and composition are those of the run's
upwind end. `MediumColumn` takes `densitySelection = actual` too: the density of the stream
entering the column, which is the upwind end's by the flow's direction (the driver
`"air.forward"`). Where that flow reverses, the column's head jumps; `simulate` locates the
reversal, ends the step there and starts the next one from the airflow of the new direction.
A reversal closer to a step's start or end than the locator's margin (the larger of 1e-3 of
the step and 1e-5 of the output interval) is moved to that start or end: an error of at most
that margin times the jump in the flow. A flow whose direction does not settle (each
direction's density drives it the other way) raises `ConvergenceError`, naming the element.
Under `Model.step` the transport flows and directions are those of the previous coupling
pass, converging with the coupling iteration.

### Movers

 `Buildings.Fluid.Movers.FlowControlled_dp` is a fan with a prescribed pressure
rise, `port_b.p - port_a.p = dp` (`FlowControlled_dp.mo`), whatever the flow: an implicit law,
the flow an unknown of the air layer's solve (as the currents of sources are in the
electrical networks). `inputType = Constant` takes `constantHead` (default
`dp_nominal`); `Continuous` (MBL's default) takes the signal driving `<mover>.dp_in`, evaluated
on the driver grid like any other input. The mover is an edge between its two nodes with no
buoyancy head of its own; it may be wired to a zone or boundary directly, or to a one-way
element through a junction, where it is one of the elements in series (it passes the stream on
unchanged), but a hydrostatic column wired to a mover with no flow element between them is
refused. Only the configuration the law reproduces exactly is accepted, each parameter against
its MBL default, so an export that omits one is read as MBL would:
`use_riseTime = false` (the default, true, is a 30 s rate limiter on `dp`),
`addPowerToMedium = false` (the default, true, heats the air by the fan power),
`energyDynamics = SteadyState` (the default, `DynamicFreeInitial`, adds a mixing volume of
1 s time constant), `prescribeSystemPressure = false` and `inputType` `Constant` or
`Continuous` (`Stages` is refused). Each violation is refused naming the instance and the
parameter. With volume mass storage the mover's flow is algebraic, not a state (it has no
inertia and, with `energyDynamics = SteadyState`, no volume): every step's airflow solve fixes
it. Only the t = StartTime row is not a solve: there every storing zone is at its start
pressure, and the rise relation does not involve the flow. As MBL's DAE does by index
reduction, the reader takes the flow that keeps the rise held, the pressures of the mover's two
ends changing at the same rate: between two volumes `V_a`, `V_b` whose other flows are `q_a`
in and `q_b` out, `m_flow = (V_b q_a + V_a q_b)/(V_a + V_b)` (in general the volumes' pressure
capacities `V k` weight it). This is the value OpenModelica reports from a start that meets the
rise; a start that violates it (both volumes at the same `p_start` across a 5 Pa fan) is an
inconsistent initialisation there, and here the first step moves the pressures onto the rise
through the mover, conserving mass, with the graded start that resolves a release.
`names.edges["<mover>"]` is its `m_flow` column, like a one-way element's.

### Feedback control

 `LimPID` is supported as a P controller, `y = clamp(k (wp u_s - u_m),
yMin, yMax)`, transcribed from MSL 4.1.0 (`Blocks/Continuous.mo`): its measurement from a
temperature sensor, its output, through Math blocks (`Gain`, `Add`, `Add3`, `Sum`, `MultiSum`,
`Feedback`, `Product`, `Division`), into a `PrescribedHeatFlow` on the sensed zone. The
blocks the sensor's reading reaches form a continuous control network
(`ControlNetwork`, see [Controls](schedules-and-controls.md#controls)); the
signals that feed it from outside the loop (the set point) are evaluated on the driver grid.
The heat flow is affine in the zone temperature on each piece of the limiter, and enters the
zone's energy balance in that form, so that the transport step treats it implicitly: MBL's
examples heat a 75 kJ/K room with `k gain` = 15000 W/K, a loop that settles in 5 s. The
limiter's corners are switches, located on step boundaries like an element's. `names.controls`
maps each block of the loop (`con`, `gain`) to its column of `simulate`'s `"controls.outputs"` (the key
`simulate` uses for a control network's outputs),
which holds the blocks' outputs `y`.

### Volume mass storage

 Like MBL, the reader stores each volume's air compressibly unless its
`massDynamics` is `SteadyState`: zone pressures are states, the t = StartTime row holds MBL's
`p_start` initialisation, a closed zone group needs no pressure reference, and a mass source
may feed one (see the [verification tables](../../reference/verification/building.md#against-openmodelica-modelica-buildings-library)).
`read_modelica(path, mass_storage=False)` gives the quasi-steady airflow of the CONTAM route
instead, with its closed-group rules below.

### Weather and wind

 A `ReaderTMY3` supplies the outside boundaries wired to it with the dry-bulb
temperature, the atmospheric pressure, the relative humidity, the wind speed and the wind
direction, as MBL computes them: the TMY3 table interpolated by MSL's Akima spline on the
calendar time (which repeats a one-year file), converted to K, Pa, 1, m/s and rad, and limited
(humidity to 0..1, wind speed to at least 0, direction modulo 2 pi). Each output may come from
the file, a parameter or an input signal (`<output>Sou`); `pAtm` defaults to a 101325 Pa
parameter, as in MBL. Solar and sky outputs are not read. The `.mos` file travels with the
JSON (see [Reproducing the export](#reproducing-the-export)); a model whose file is missing or altered is refused.

An outside boundary is at the atmospheric pressure, the dry-bulb temperature and the water
mass fraction of the relative humidity (`X = phi k / (k phi + p/pSat - phi)`). The wind
boundaries add `pWin = Cs 0.5 Cp d v^2`:

- `Outside_Cp`: a fixed `Cp` (or the `Cp_in` signal), `d` at the atmospheric pressure and the
  default composition;
- `Outside_CpLowRise`: Swami and Chandra's low-rise correlation, `Cp = Cp0 ln(1.248 - 0.703
  sin(a/2) - 1.175 sin^2 a + 0.131 sin^3(2 a G) + 0.769 cos(a/2) + 0.071 G^2 sin^2(a/2) + 0.717
  cos^2(a/2))`, `G = ln s`, blended to `Cp0` below 5 degrees and to its 180-degree value above
  175;
- `Outside_CpData`: a periodic cubic Hermite spline through the `(incAngSurNor, Cp)` table,
  scaled by `Cs`.

The last two take `d` at the boundary's own pressure, an implicit equation solved by
iteration. Angles follow MBL: the wind direction is where the wind comes from (north 0, east
pi/2), a surface azimuth where the surface faces (south 0, west pi/2), and the incidence is
`winDir - (azi - pi)`. These differ from CONTAM's, whose wind pressure acts on the envelope
path rather than on the boundary node.

The boundaries' values are point values at each driver-grid time, like any boundary signal.
The table's knots and the points where a limiter starts cutting are added to the driver grid as
step boundaries (the outputs are once continuously differentiable there); a repeating file's
year wrap is an event. `names.weather` (from `read_modelica(..., return_names=True)`) holds the
reader's outputs, keyed `"<reader>.weaBus.<output>"`, and every wind boundary's `"<boundary>.pWin"`
on the output grid; `assemble.weather_columns(doc, times)` evaluates the same for a model with
no flow element, which `read_modelica` refuses.

```py
from noodl.apps.building_physics import read_modelica

model, state, drivers, names = read_modelica("PressurizationData.json",
                                             return_names=True)
print(names.times[2].item(), names.weather["weaDat.weaBus.winSpe"][2].item())
print(names.weather["west.pWin"][2].item(), names.weather["east.pWin"][2].item())
```

```text
10368.0 2.133143665116279
0.3241156179122303 -1.2724675933675318
```

### Conventions

- `DoorOpen`/`DoorOperable` use MBL's fixed default density (`Door.mo`); a discretised door
  (`DoorDiscretizedOpen`/`Operable`) evaluates density at the actual port pressure
  (`TwoWayFlowElement.mo`) instead — the two door families do not share one convention.
- A door becomes two directional noodl physics edges between the same pair of zones; so does a
  discretised door, whose two edges are its port streams `mAB_flow` and `-mBA_flow`
  (`DoorPortStream`), the compartments and their hydrostatic heads evaluated inside the
  element.
- Zonal flows are four-port, like doors (not the two-port shape a one-way element has), and
  become two directional edges the same way.
- An in-line flow sensor (`Buildings.Fluid.Sensors`, flow-through) is a transparent wire: it adds
  no node and no pressure drop. So is a zero-resistance connection: a
  `Fluid.FixedResistances.LosslessPipe`, or a `Fluid.FixedResistances.PressureDrop` with
  `dp_nominal = 0` (MBL then computes no resistance, `dp = 0`). A `PressureDrop` with any
  other `dp_nominal` is refused, and so is a short between two volumes or boundaries.
- `PrescribedHeatFlow` is supported only at `alpha = 0` (MSL's default: no temperature
  dependence); a nonzero `alpha` is refused.
- A `MassFlowSource_T` with a negative `m_flow` draws air out of its volume. By Modelica's
  stream semantics the air entering the source is the volume's own, so it leaves at the
  zone's temperature and composition, whatever the source's `T`, `X` and `C`
  (`ChimneyShaftWithVolume` extracts 0.05 kg/s for ten minutes).
- A boundary wired straight to one zone's port, and to no other port, is supported: it fixes that
  zone's pressure.
- `"air.phi"` is GAUGE pressure relative to a per-model reference `p_ref` (the first boundary's
  pressure, or an attached boundary's, at the first grid time; the first zone's `p_start` with
  neither) — flows depend only on pressure differences, so the choice changes no result.
- Every source driver (air, heat and species) is the MEAN of the source over each step, not its
  end-of-step value, so that a pulse shorter than the output interval still injects its exact
  mass (e.g. `CO2TransportStep`'s 3.6 s pulse landing between two 172.8 s outputs).
- A signal may drive several inputs (`drives` accepts one name or a list) — MBL's `ZonalFlow`
  example drives two flows from one `Constant`.
- Refused, also with a named error: with `mass_storage=False` (or volumes whose `massDynamics`
  is `SteadyState`), a closed group of zones (joined only by pressure-dependent
  edges or zonal flows, no boundary among them) with a net flow imbalance — an unequal
  `ZonalFlow_m_flow` pair or a mass source into it; and an outside boundary whose weather-bus
  inputs come from no `ReaderTMY3` and no signal. With storage, none of the 22 supported models
  needs either.

### Imported but not compared

`PressurizationData` and `TrickleVent` import (`read_modelica` builds them, and their weather and wind pressures match MBL's to round-off), but they are not compared as dynamic models. The error comes from a transport step that sees the flows only at the step's two ends, and the figures are under [Limitations](limitations.md).

## What is refused

Each of these raises a named error (`ModelicaImportError` lists every offending instance and
its class):

- A door or zonal flow inside a column chain, and a ring of one-way elements in series with
  no volume or boundary.
- A `LimPID` with an integral or derivative part (`controllerType` PI, PD or PID), with
  feed-forward, or with the steady-state or initial-output initialisation; a feedback loop
  whose signal drives anything but a `PrescribedHeatFlow`'s `Q_flow`, a heat flow into a zone
  a fixed temperature holds, or a zone's heat flow read from another zone's temperature; a
  product of two feedback signals or a division by one (not affine in the zone temperature).
- A `MediumColumnDynamic` with `use_HeatTransfer = true` (heat exchanged through its
  `heatPort`).
- A component output other than a temperature sensor's `T` wired into a signal block, or a
  sensor's `T` wired to anything but one of a block's inputs, naming the connection; a
  temperature sensor whose `port` is not wired to a volume's `heatPort`.
- A `TraceSubstancesFlowSource` with a negative (extracting) flow.
- A `FlowControlled_dp` with `use_riseTime`, `addPowerToMedium` or `prescribeSystemPressure`
  true, with `energyDynamics` other than `SteadyState`, or with `inputType = Stages`.
- Every class outside the supported list above ("component class is not supported", or
  "signal source is not supported" for a signal block), the CDL blocks
  (`Buildings.Controls.OBC.CDL`, their PID and hysteresis included) among them.
- Every `Buildings.BoundaryConditions.WeatherData` class but `ReaderTMY3`; a weather bus wired
  to anything but an `Outside` boundary's `weaBus`, or any weather-bus output other than
  `TDryBul`, `pAtm`, `relHum`, `winSpe` and `winDir` read by anything; a `ReaderTMY3` wired
  through any port but its `weaBus` and the `<output>_in` inputs of those five; a signal
  driving one of its inputs that it does not read (an output taken from the file, or an
  input of another output).

For example, an orifice `ori` wired to a door's `port_a1` raises:

```text
ModelicaImportError: modelica: refused 1 item:
  - ori (Buildings.Airflow.Multizone.Orifice): a door or zonal flow in a hydrostatic-column chain
```
