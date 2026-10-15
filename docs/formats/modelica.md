# Modelica Buildings Library

Reading a Modelica export produces a model for [building
physics](../applications/building_physics.md). Power-law, tabulated and door components become
the [`noodl.elements`](../applications/building_physics.md#airflow-elements) classes of the
same physics — an open door, for example, becomes an `OpenDoor` edge pair and a discretised
door an `DoorPortStream` pair, its two port streams — while zonal flows are assembled by the
importer itself as prescribed-flow edge pairs, as set out below.

`read_modelica(path) -> (model, state, drivers)` imports a multizone airflow model built with the
Modelica Buildings Library (MBL) v13.0.0 (commit `55abf579598ca81cae0a82f337350375958e6722`)
through an OpenModelica JSON export. The reader never parses Modelica source and never evaluates
a Modelica expression: every number in the JSON was already evaluated by OpenModelica.
Electrical circuits of the Modelica Standard Library and of MBL's `Buildings.Electrical` have
a reader of their own: see [Electrical circuits](#electrical-circuits) below.

```python
from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import simulate

model, state, drivers, names = read_modelica("tests/data/modelica/OneRoom.json", return_names=True)
result = simulate(model, state, drivers, names.times)

print(result["T"][-1])       # zone temperatures (K) at the last grid time, node order names.nodes
print(result["air.q"][-1])   # edge flows (kg/s); names.edges maps an MBL instance to its columns
```

**Supported (19 models: 8 `Validation` + 11 `Examples`).** `Buildings.Airflow.Multizone`
elements, `MixingVolume` zones, pressure/temperature boundaries and trace substances — not
`Buildings.ThermalZones`, HVAC, wind, weather or district networks. `Validation`: `OneWayFlow`,
`DoorOpenClosed`, `OpenDoorPressure`, `OpenDoorTemperature`, `ThreeRoomsContam`,
`ThreeRoomsContamDiscretizedDoor`, `OpenDoorBuoyancyDynamic`, `OpenDoorBuoyancyPressureDynamic`.
`Examples`: `CO2TransportStep`, `ClosedDoors`, `NaturalVentilation`, `OneEffectiveAirLeakageArea`,
`OneOpenDoor`, `OneRoom`, `Orifice`, `PowerLaw`, `ReverseBuoyancy`, `ReverseBuoyancy3Zones`,
`ZonalFlow`.

**Refused, each with a named error** (`ModelicaImportError` lists every offending instance and
its class):

- `PressurizationData`, `TrickleVent`, `ChimneyShaftNoVolume`, `ChimneyShaftWithVolume` — wind
  pressure, weather data, feedback controllers, or a dynamic (mass- and heat-storing) hydrostatic
  medium column, none of which is in scope.

For example, reading `PressurizationData` raises:

```
ModelicaImportError: modelica: refused 3 items:
  - east (Buildings.Fluid.Sources.Outside_CpLowRise): wind pressure is not supported
  - weaDat (Buildings.BoundaryConditions.WeatherData.ReaderTMY3): weather data is not supported
  - west (Buildings.Fluid.Sources.Outside_CpLowRise): wind pressure is not supported
```

**Conventions:**

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
  no node and no pressure drop.
- `PrescribedHeatFlow` is supported only at `alpha = 0` (MSL's default: no temperature
  dependence); a nonzero `alpha` is refused.
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
  `ZonalFlow_m_flow` pair or a mass source into it; `MediumColumn.densitySelection = "actual"`;
  and `Outside` without a weather-bus signal driving it. With storage, none of the 19 supported models needs
  any of the three.

**Volume mass storage.** Like MBL, the reader stores each volume's air compressibly unless its
`massDynamics` is `SteadyState`: zone pressures are states, the t = StartTime row holds MBL's
`p_start` initialisation, a closed zone group needs no pressure reference, and a mass source
may feed one (see [the Modelica parity
tables](../applications/building_physics.md#against-openmodelica-modelica-buildings-library)).
`read_modelica(path, mass_storage=False)` gives the quasi-steady airflow of the CONTAM route
instead, with its closed-group rules below.

**Reproducing the export (WSL only — the test suite itself needs none of this).** Tested on
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
python3 scripts/modelica_export.py Buildings.Airflow.Multizone.Validation.ThreeRoomsContam --out tests/data/modelica
```

(`<Model>` is the model's fully qualified name; the exporter writes `<Short>.json`/`<Short>.csv`
under `<Short>`, its last component — `ThreeRoomsContam.json`/`ThreeRoomsContam.csv` here.) This
writes `tests/data/modelica/<Short>.json` (the component graph) and `<Short>.csv` (OpenModelica's
own simulated reference) as committed fixtures. `scripts/modelica_export.py` is not imported by
the package and is not run by the test suite. It exits non-zero if the simulation fails (the
JSON is still written, with the instance API's Reals instead of the simulated values, so the
export can be inspected); a batch script over every model should check the exit code rather
than assume success. The dynamic parity tests compare a window of rows of every dynamic model
in the repository's default `pytest` run; every row of every model runs under
`@pytest.mark.slow` (`pytest -m slow`). `--tolerance` simulates at a solver tolerance other than the model's
declared one; the thirteen dynamic parity references were regenerated with
`--tolerance 1e-13` (`tests/data/modelica/NOTICE.md`). At 1e-14 OpenModelica's DASSL returns
no trajectory for these models, so 1e-13 is the finest reference available.

Regenerating the committed parity records (`tests/data/modelica/parity-{algebraic,dynamic}.json`,
in [the building physics parity tables](../applications/building_physics.md#against-openmodelica-modelica-buildings-library))
needs no OpenModelica — they are written by `tests/verification/test_modelica_parity.py`
itself, only when the environment variable `NOODL_RECORD_PARITY=1` is set:

```bash
NOODL_RECORD_PARITY=1 pytest tests/verification/test_modelica_parity.py -m "not slow"
NOODL_RECORD_PARITY=1 pytest tests/verification/test_modelica_parity.py -m slow
```

(two runs, since the default `addopts` excludes `slow`-marked tests and a command-line `-m`
replaces rather than adds to it). Without the variable, the suite reads and checks the
fixtures but never rewrites the records.

## Electrical circuits

A second reader, `noodl.apps.electrical.modelica.read_modelica_circuit`, imports electrical
example models of the Modelica Standard Library (MSL 4.1.0, `Modelica.Electrical.Analog`, with
the lumped heat-transfer components of `Modelica.Thermal.HeatTransfer`) and of the Modelica
Buildings Library (MBL v13.0.0, `Buildings.Electrical`) into an [electrical
circuit](../applications/electrical.md#reading-modelica-circuits). As for airflow models, the
reader never parses Modelica source: OpenModelica evaluates every parameter, and the reader
reads a JSON export.

```py
from noodl.apps.electrical.modelica import read_modelica_circuit, simulate_circuit

model, state, drivers, names = read_modelica_circuit("DC.Loads.Examples.Resistor.json")
history = simulate_circuit(model, state, drivers, names)   # every output time
names.value(history, "res2.terminal.i[1]")                 # a reference column, (T,)
```

`read_modelica_circuit(path, *, substeps=1, omit=())` returns `(model, state, drivers, names)`:
a `Model` with a `PotentialFlowLayer` named `"electrical"` (and a second one, `"thermal"`, when
a device has a heat port), its start state, its drivers on the step grid, and a `CircuitNames`
that maps every compared column of the reference to a noodl quantity (`names.value`).
`substeps` splits every interval of the step grid into that many equal steps; `omit` names
components to drop, with their connections, before the circuit is built.
`simulate_circuit(model, state, drivers, names, *, times=None)` runs it. A model the reader
cannot convert in full raises one `ModelicaImportError` naming every offending instance, never
a partial model.

### The export

`scripts/modelica_export_electrical.py` (run in WSL; standard library only, never imported by
noodl and never run by the test suite) loads MSL 4.1.0 and MBL for every model, simulates it
with DASSL at a relative tolerance of `1e-12` (`--tolerance`; the models declare `1e-6` or
nothing) on the model's own output grid or on `--intervals N`, and writes `<stem>.json` and
`<stem>.csv`, the stem being the model name without `Modelica.Electrical.` or
`Buildings.Electrical.`:

```bash
python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.SeriesResonance --intervals 1000
python3 scripts/modelica_export_electrical.py Buildings.Electrical.DC.Loads.Examples.Resistor --intervals 100
```

OpenModelica, MSL and MBL are installed as described above; the commands for all 14 committed
models are in `tests/data/modelica_electrical/NOTICE.md`. The JSON
(`"format": "noodl-modelica-electrical/1"`) holds `model`, `mbl_commit`, `openmodelica`,
`msl`, `experiment` (`StartTime`, `StopTime`, `Interval`, `Tolerance` as simulated),
`experiment_declared` (the model's own annotation), `components` (each with `name`, `class`,
every public parameter in full double precision, and where they apply `initial` and `derived`,
below), `connections` (pairs of connector names), `signals` (the `Modelica.Blocks` sources and
math blocks, with the inputs each `drives`) and `compared` (the CSV's columns after `time`).
The CSV starts with `# key: value` lines (model, MBL commit, OpenModelica and MSL versions,
tolerance, start and stop time, interval, solver).

**Compared columns.** Per component in document order, skipping sensors and grounds: for an
MSL analog class its current `i` (or `i1`, `i2` for a two-port), plus `v` for a `Capacitor`
and `T_heatPort` when its heat port is enabled; a `HeatCapacitor`'s `T`; for an MBL class, per
connector `terminal`, `terminal_n`, `terminal_p`, the conductor potentials `v[1]`, `v[2]` and
the current `i[1]` of a DC terminal, or the phasor parts `v[1]`, `v[2]`, `i[1]`, `i[2]` of an
AC terminal, plus `Vc` of a `TwoPortRCLine` with capacitance.

### Supported classes

| Library | Classes | noodl law or role |
|---|---|---|
| MSL `Analog.Basic` | `Ground` | node `"0"` at 0 V |
| | `Resistor`, `VariableResistor` | `Resistance` (temperature and scale drivers) |
| | `Capacitor`, `Inductor`, `Transformer` | `Capacitance`, `Inductance`, `MutualInductance` |
| MSL `Analog.Ideal` | `IdealTransformer` (`considerMagnetization = false`) | `TransformerRatio` |
| MSL `Analog.Semiconductors` | `Diode` (with or without temperature dependency) | `JunctionDiode` |
| MSL `Analog.Sources` | `ConstantVoltage`, `StepVoltage`, `SineVoltage`, `RampVoltage`, `PulseVoltage`, `SignalVoltage`, `SineVoltageVariableFrequencyAndAmplitude`, `CosineVoltageVariableFrequencyAndAmplitude` | `PrescribedVoltage` following the equivalent `Modelica.Blocks.Sources` signal |
| MSL `Analog.Sensors` | `CurrentSensor` | a wire (its two pins are one node) |
| | `VoltageSensor`, `PotentialSensor` | ignored, with their connections |
| MSL `Thermal.HeatTransfer` | `Components.ThermalConductor`, `Components.HeatCapacitor` | `Conductance`, `Capacitance` in the thermal layer |
| | `Sources.FixedTemperature`, `Celsius.FixedTemperature` | a thermal boundary node |
| MBL `DC` | `Sources.ConstantVoltage` | `PrescribedVoltage` |
| | `Loads.Resistor`, `Lines.TwoPortResistance` | `Impedance(R)` |
| | `Loads.Conductor` | `ConstantPowerLoad(..., direct_current=True)` |
| | `Lines.TwoPortRCLine` | `Impedance(R/2)` either side of an internal node; with `use_C` a `Capacitance` from it |
| | `Sensors.GeneralizedSensor` | a wire |
| MBL `AC.OnePhase` | `Sources.FixedVoltage` | `PrescribedVoltage` with a complex amplitude |
| | `Loads.Impedance` | `Impedance(R, L)` or `Impedance(R, C)` |
| | `Loads.Inductive`, `Loads.Resistive` | `ConstantPowerLoad` (power factor `pf`, linearised or not) |
| | `Lines.TwoPortResistance`, `TwoPortInductance`, `TwoPortRL`, `Line` | `Impedance` |
| | `Lines.TwoPortRLC` | a T of two `Impedance(R/2, L/2)` and a shunt capacitance |
| | `Conversion.ACACTransformer` | `ImpedanceTransformer` |
| | `Conversion.ACDCConverter` (`ground_DC = true`) | `PowerConverter` |
| | `Sensors.GeneralizedSensor` | a wire |
| MBL `AC.ThreePhasesBalanced` | `Sources.FixedVoltage`, `Loads.Impedance`, `Lines.Line` | as the one-phase classes (an impedance in triangle `R/3`, `L/3`, `3 C`; the line as a T) |

The MBL load modes `FixedZ_steady_state`, `VariableZ_y_input` and `VariableZ_P_input` are read.
The inputs of variable resistors, loads and variable-frequency sources come from the document's
signals, `Modelica.Blocks.Math` chains included.

### Refused, with the reason

- `Storage.Battery` (DC and AC), `Sources.PVSimple` and the `Grid` sources: not implemented.
- `Analog.Ideal.IdealDiode`: a piecewise-linear switch needs event location, which noodl's
  fixed step grid does not have.
- Every `AC.ThreePhasesUnbalanced` class: unbalanced three-phase systems are not supported.
- A component with `mode` or `modelMode` `FixedZ_dynamic` (a dynamic phasor model, including a
  `Line` whose load mode says so), an MBL cable with `use_T`, an MBL component with
  `useHeatPort = true`, an MBL resistive part away from its reference temperature (`T` not
  `T_ref`), an `IdealTransformer` with `considerMagnetization`, an `ACDCConverter` with
  `ground_DC = false`, an MSL `Diode` with `useHeatPort` but without
  `useTemperatureDependency` (its law would ignore the heat-port temperature).
- Inputs the reader does not follow: an AC `Impedance` with `use_R_in`, `use_L_in` or
  `use_C_in` (an impedance following a signal), an `Inductive` load with `use_pf_in` (a power
  factor input), and a power factor `pf` inside the band of MBL's
  `smoothMin(pf, 0.99999, 2.5e-6)`, whose blend is not transcribed (or outside `(0, 1]`).
- Mixtures one law cannot hold, since these settings apply to every edge of a kind: diodes with
  and without `useTemperatureDependency`, linearised and nonlinear loads, DC and AC loads in
  one model.
- A thermal network part (thermal nodes joined by conductors) held by no fixed temperature and
  no heat capacitor: its temperature is undetermined, and the refusal names its nodes.
- A non-finite (NaN or infinite) frequency `f` of an AC source or temperature `T` of a fixed
  temperature.
- In a phasor model: an MSL analog component other than `Ground`, a DC `TwoPortRCLine` with
  capacitance (a phasor layer would give it the admittance `j omega C`), AC sources at
  different frequencies.
- Connection errors: a pin connected to a terminal, a DC terminal to an AC terminal, an enabled
  heat port left unconnected, a DC `ConstantVoltage` whose pin `n` is not connected to a
  `Ground`, an unknown component or connector.
- Any class not in the table above.

A connection to a disabled conditional heat port is dropped (legal Modelica: it disappears with
the connector), and a connection with a signal-block end is skipped (signal wiring is carried by
the signals' `drives`). MBL's `ACLine_RL` example, for instance, is refused:

```
ModelicaImportError: modelica: 1 problem:
  - RL_3 (Buildings.Electrical.AC.OnePhase.Lines.TwoPortRL): mode = Buildings.Electrical.Types.Load.FixedZ_dynamic: dynamic phasor models are not supported
```

`omit=("RL_3", "load_sc_3")` reads it without that branch and its load (the remaining lines are
steady). Omitting a component states that the rest of the model stands without it: here the
source `E` also feeds the omitted branch, so its current is not comparable with the reference.

### Connectors and nodes

- An MSL pin (`p`, `n`, `p1`, ...) is one real node.
- An MBL DC terminal carries two conductors (`v[2]`, `i[2]`) and becomes two nodes, one per
  conductor, named `<instance>.<terminal>[1]` and `[2]`; a `connect` joins conductor k with
  conductor k. `v[k]` is that conductor's potential against the absolute reference, so a DC
  `ConstantVoltage` (`terminal.v[1] = V`) must have its pin `n` grounded.
- An MBL AC terminal (one-phase, also used by the balanced three-phase package) is one complex
  node, its `v = {v[1], v[2]}` the phasor `v[1] + j v[2]`; a device with one AC terminal is an
  edge from that node to the reference.
- MSL's `Ground`, MBL's implicit AC neutral and the absolute DC reference are all node `"0"` at
  potential 0. Heat ports are nodes of the thermal layer, whose reference node
  `"thermal.reference"` is held at 0 K.

### Real or phasor

A model with any `Buildings.Electrical.AC` component is solved as one phasor layer at its AC
sources' common frequency; its DC parts are complex nodes whose imaginary parts come out zero
(every DC law used there is real and frequency-independent). Every other model (MSL analog, MBL
DC) is one real layer, stepped in time, or solved at each output time when it has no capacitor
or inductor. MBL's AC models are quasi-static phasor equations and MSL's are instantaneous ones,
and no example mixes the two. The phasors are MBL's: root-mean-square values in the frame
rotating with the source angle, written directly as complex source amplitudes, so the power
laws (which assume RMS phasors) hold.

### Start state

A model with storage starts from the reference's own initialisation. The exporter records the
value at the start time of every state variable (capacitor `v`, inductor `i`, transformer
`i1`, `i2`, heat capacitor `T`, RC line `Vc`) in the component's `initial`, and the reader
builds the start state from those: node voltages propagated from node 0 over the capacitor
edges, inductor and transformer currents as recorded. The first step, and every step starting
at a signal jump, is backward Euler, which reads only those capacitor voltages and inductor
currents; the other steps are trapezoidal. No consistent initial solve is attempted: it is
singular for capacitor loops (as in `CauerLowPassAnalog`), and OpenModelica itself falls back
on start values where its initialisation is underdetermined. The row at the start time of a
dynamic model is therefore not compared.

### MBL one-phase lines: `R/3` and `L/3`

MBL's one-phase `AC.OnePhase.Lines.Line` instantiates its inner line as
`TwoPortRL(R=R/3, L=L/3)`, where `R = l RCha` and `L = l XCha/(2 pi 50)` come from the cable
(`Transmission/BaseClasses/PartialBaseLine.mo`). The division by 3 is the convention of the
unbalanced three-phase lines ("each line has a resistance equal to R/3",
`ThreePhasesUnbalanced/Lines/TwoPortResistance_N.mo`), carried into the one-phase line without
a word in its documentation. The balanced three-phase `Line` redeclares the inner line as
`TwoPortRLC(R=R, L=L, C=C)`, and there the redeclaration's values hold: in
`ThreePhasesBalanced.Lines.Examples.ACLine`, `line_1` (2000 m) simulates with
`line_1.line.R = line_1.R = 0.724` ohm (and `line.L = 4.647e-4` H, `line.C = 0`), although
OpenModelica's instance API still prints `R/3` for the inner line. The reader reproduces MBL as
it is (one-phase `R/3`, `L/3`; balanced `R`, `L`, `C`). The exporter writes each `Line`'s
simulated `line.R`, `line.L`, `line.C` into its `derived`, and the reader refuses a file in
which the class rule and the simulated value differ by more than `1e-12` relative.

### Solver tolerances

`simulate_circuit` solves a model without storage at each output time with Newton's
`atol=1e-12`, `rtol=1e-14` (`atol=1e-11` with a heat-port thermal network, whose heat balance
in W stalls at about `2e-12` W). The layer's default, `1.5e-8`, leaves MBL's `ACACTransformer`
example up to 29 times the comparison bound from OpenModelica. Pass the same tolerances to
`model.steady` when solving such a model by hand. The steps of a dynamic model use the layer's
own tolerances. The comparison and the measured agreement are on [the verification
page](../verification.md#modelica-electrical-circuits-model-by-model).
