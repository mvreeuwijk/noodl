# Importing Modelica circuits

How to load electrical example models of the Modelica Standard Library (MSL 4.1.0,
`Modelica.Electrical.Analog`, with the lumped heat-transfer components of
`Modelica.Thermal.HeatTransfer`) and of the Modelica Buildings Library (MBL v13.0.0,
`Buildings.Electrical`: DC, one-phase AC and balanced three-phase AC) into an
[electrical circuit](index.md) model, and what is read and refused. For the airflow models of
the same library see [Modelica airflow](../building/modelica.md).

The reader never parses Modelica source: OpenModelica evaluates every parameter, and the
reader reads a JSON export of the evaluated model.

## How to load

```py
from noodl.apps.electrical.modelica import read_modelica_circuit, simulate_circuit

model, state, drivers, names = read_modelica_circuit("DC.Loads.Examples.Resistor.json")
history = simulate_circuit(model, state, drivers, names)   # every output time
names.value(history, "res2.terminal.i[1]")                 # a reference column, (T,)
```

`read_modelica_circuit(path, *, substeps=1, omit=())` returns `(model, state, drivers, names)`:
a `Model` with a `KirchhoffLayer` named `"electrical"` (and a heat layer, a `TransportLayer`
named `"thermal"`, when a device has a heat port), its start state, its drivers on the step
grid, and a `CircuitNames` that maps every compared column of the reference to a noodl
quantity (`names.value`). `substeps` splits every interval of the step grid into that many
equal steps; `omit` names components to drop, with their connections, before the circuit is
built. `simulate_circuit(model, state, drivers, names, *, times=None)` runs it. A model the
reader cannot convert in full raises one `ModelicaImportError` naming every offending
instance, never a partial model.

MBL's `DC.Loads.Examples.Resistor` feeds two 2 ohm loads from a 12 V source, `res1` directly
and `res2` through a 2 ohm line `res`:

```python
from noodl.apps.electrical.modelica import read_modelica_circuit, simulate_circuit

model, state, drivers, names = read_modelica_circuit("DC.Loads.Examples.Resistor.json")
history = simulate_circuit(model, state, drivers, names)

print(f"{names.model}: {len(names.times)} output times, "
      f"dynamic {names.dynamic}, phasor {names.phasor}")
for load in ("res1", "res2"):
    v = float(names.value(history, f"{load}.terminal.v[1]")[-1])
    i = float(names.value(history, f"{load}.terminal.i[1]")[-1])
    print(f"{load}: v = {v:.6f} V, i = {i:.6f} A")
print(f"res: i = {float(names.value(history, 'res.terminal_n.i[1]')[-1]):.6f} A")
```

```text
Buildings.Electrical.DC.Loads.Examples.Resistor: 101 output times, dynamic False, phasor False
res1: v = 12.000000 V, i = 6.000000 A
res2: v = 6.000000 V, i = 3.000000 A
res: i = 3.000000 A
```

OpenModelica's own values for these columns are the same to the last digit.

`CircuitNames` holds `times` (the reference's output times), `grid` (the step grid), `nodes`
and `devices` (where each node and device sits in `"electrical.phi"` and `"electrical.q"`),
and `value(history, column)`, which reads any column of the reference CSV from a history. A
column of an MBL terminal is a conductor potential or current (`v[1]`, `v[2]`, `i[1]`) or, in
an AC model, the real or imaginary part of a phasor.

`simulate_circuit` solves a model without storage at each output time and steps a dynamic one
over `names.grid`. Unlike [`transient`](circuits.md#time-stepping-and-breakpoints), the
reader's step grid contains every jump and kink of the sources and signals as well as the
output times, so no edge falls inside a step: the step ending at a jump uses the signal's
value just before it, and the step starting there is backward Euler (as is the first).
`read_modelica_circuit(path, substeps=r)` splits each interval into `r` steps; two runs at
`r` and `2 r` combined by `noodl.apps.electrical.modelica.extrapolate(coarse, fine,
order=2)` remove the leading time-step error, which is how the comparison with OpenModelica
reaches `1e-6` of each quantity's range. A model with heat ports gets the heat layer and the
coupling of [Electro-thermal coupling](electro-thermal.md); its heat-port and heat-capacitor
temperatures are in `"thermal.x"`.

## Make an export

The repository's `scripts/modelica_export_electrical.py` (run in WSL; standard library only,
not part of the noodl package) loads MSL 4.1.0 and MBL for every model, simulates it
with DASSL at a relative tolerance of `1e-12` (`--tolerance`; the models declare `1e-6` or
nothing) on the model's own output grid or on `--intervals N`, and writes `<stem>.json` and
`<stem>.csv`, the stem being the model name without `Modelica.Electrical.` or
`Buildings.Electrical.`:

```bash
python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.SeriesResonance --intervals 1000
python3 scripts/modelica_export_electrical.py Buildings.Electrical.DC.Loads.Examples.Resistor --intervals 100
```

OpenModelica, MSL 4.1.0 and MBL must be installed in WSL. The JSON
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

## What is read

The supported classes:

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
| MSL `Thermal.HeatTransfer` | `Components.ThermalConductor`, `Components.HeatCapacitor` | a conduction edge, a node's heat capacity in the heat layer (a heat port without one holds no heat: a storage-free node) |
| | `Sources.FixedTemperature`, `Celsius.FixedTemperature` | a boundary node of the heat layer |
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

## Connectors and nodes

- An MSL pin (`p`, `n`, `p1`, ...) is one real node.
- An MBL DC terminal carries two conductors (`v[2]`, `i[2]`) and becomes two nodes, one per
  conductor, named `<instance>.<terminal>[1]` and `[2]`; a `connect` joins conductor k with
  conductor k. `v[k]` is that conductor's potential against the absolute reference, so a DC
  `ConstantVoltage` (`terminal.v[1] = V`) must have its pin `n` grounded.
- An MBL AC terminal (one-phase, also used by the balanced three-phase package) is one complex
  node, its `v = {v[1], v[2]}` the phasor `v[1] + j v[2]`; a device with one AC terminal is an
  edge from that node to the reference.
- MSL's `Ground`, MBL's implicit AC neutral and the absolute DC reference are all node `"0"` at
  potential 0. Heat ports are nodes of the heat layer, which has no reference node: its
  temperatures are absolute (K), in its state `"thermal.x"`.

## Real or phasor

A model with any `Buildings.Electrical.AC` component is solved as one phasor layer at its AC
sources' common frequency; its DC parts are complex nodes whose imaginary parts come out zero
(every DC law used there is real and frequency-independent). Every other model (MSL analog, MBL
DC) is one real layer, stepped in time, or solved at each output time when it has no capacitor
or inductor. MBL's AC models are quasi-static phasor equations and MSL's are instantaneous ones,
and no example mixes the two. The phasors are MBL's: root-mean-square values in the frame
rotating with the source angle, written directly as complex source amplitudes, so the power
laws (which assume RMS phasors) hold.

## Start state

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

## MBL one-phase lines: `R/3` and `L/3`

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

## What is refused

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
- A thermal node that no `ThermalConductor` reaches and no fixed temperature holds (a heat
  capacitor directly on a heat port): the heat layer conducts heat only along conductors; put a
  `ThermalConductor` between them.
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

## Solver tolerances

`simulate_circuit` solves a model without storage at each output time with Newton's
`atol=1e-12`, `rtol=1e-14` (a heat-port thermal network is a linear solve, with no tolerance
of its own). The layer's default, `1.5e-8`, leaves MBL's `ACACTransformer`
example up to 29 times the comparison bound from OpenModelica. Pass the same tolerances to
`model.steady` when solving such a model by hand. The steps of a dynamic model use the layer's
own tolerances; the heat layer of a model with heat ports takes the circuit's rule on every
step (`layer_options={"thermal": {"scheme": "backward_euler"}}` after a restart, `"trapezoidal"`
otherwise), which `simulate_circuit` gives it. The comparison and the measured agreement are on the [verification
page](../../reference/verification/electrical.md#modelica-electrical-circuits-model-by-model).
