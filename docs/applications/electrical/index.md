# Electrical circuits

Lumped electrical circuits (DC, transient and AC) and balanced power flow, built on one
Kirchhoff layer: node voltages are the potential, and the currents of voltage sources,
inductors, transformers and controlled sources are unknowns beside them. The same layer
type, in phasor form, solves the power flow of a network read from a MATPOWER case file.
Circuits can be batched, differentiated and coupled to a heat network.

**Matches.** The device laws transcribe the Modelica Standard Library
(`Modelica.Electrical.Analog`) and the electrical models of the Modelica Buildings Library
(`Buildings.Electrical`). The power flow follows MATPOWER 8.1 and is compared with
pandapower.

**Maturity: verified on a stated subset.** Fourteen Modelica example models agree with
OpenModelica on every column of the reference, to `1e-9` relative plus `1e-12`
absolute (algebraic models) or `1e-6` of the column range (dynamic models). The full
comparison covers every row, except the start row of the dynamic models; the default test run
compares a subset of the rows (every tenth row, or the first rows of a dynamic model). Five MATPOWER
cases (`case9`, `case14`, `case30`, `case118` and a feature-rich variant of `case9`) agree
with pandapower to round-off, five or more orders inside the tolerances. Simpler circuits
are checked against closed forms. The evidence is on the
[verification page](../../reference/verification/electrical.md). Outside the listed
subset, the application refuses rather than guesses.

## Supported and not supported

| Supported | Not supported |
|---|---|
| DC operating point, transient on a fixed time grid, AC by phasors | Adaptive time steps, event location, breakpoints inside a step |
| Resistors, conductors, capacitors, inductors, sources with waveforms, diodes, switches, ideal and coupled transformers, controlled sources | The piecewise-linear ideal diode (MSL `IdealDiode`) |
| Constant-power loads, impedances, transformers and AC/DC converters (MBL laws) | Dynamic phasors, harmonic balance, small-signal linearisation of nonlinear points |
| Balanced power flow, AC and DC, from MATPOWER case files | Reactive-power limits, DC lines, isolated buses, distributed slack |
| Joule heating coupled to a heat network | Unbalanced three-phase systems, distributed transmission lines |
| Batches (as drivers) and gradients through every analysis | MBL batteries, photovoltaics and grid sources |
| Reading MSL and MBL examples from an OpenModelica export | Modelica source files (only the JSON export is read) |

The full list, with the reasons, is on the [limitations page](limitations.md).

## A worked example: a voltage divider

A circuit is a `Circuit` of devices between named nodes, with a ground node (`"0"` by
default) held at 0 V. `build_model` turns it into a model; `operating_point` solves it.

```python
from noodl.apps.electrical import (
    Circuit, Resistor, VoltageSource, build_model, device_values, operating_point,
)

divider = Circuit([
    VoltageSource("V1", "in", "0", 10.0),
    Resistor("R1", "in", "out", 3.0),
    Resistor("R2", "out", "0", 7.0),
])
model, state, drivers = build_model(divider)
op = operating_point(model, drivers)

voltage = model.refs.electrical.voltage
print({node: round(float(v), 6) for node, v in voltage.named(op[voltage]).items()})
for name, (v, i) in device_values(model, op).items():
    print(f"{name}: v = {float(v):+.6f} V, i = {float(i):+.6f} A")
```

```text
{'0': 0.0, 'in': 10.0, 'out': 7.0}
V1: v = +10.000000 V, i = -1.000000 A
R1: v = +3.000000 V, i = +1.000000 A
R2: v = +7.000000 V, i = +1.000000 A
```

The source's current is negative: it delivers power (see
[sign conventions](physics.md#sign-conventions)).

**Units.** Voltage V, current A, resistance ohm, conductance S, capacitance F, inductance H,
time s, frequency Hz, temperature K. The power flow is in per unit.

## Pages in this section

| Page | Kind | What it covers |
|---|---|---|
| [Build and analyse a circuit](circuits.md) | How-to | Devices, waveforms, DC, transient, time stepping, batches, gradients, input checks |
| [AC analysis and power laws](ac.md) | How-to | Frequency sweeps, phasor values, impedances, constant-power loads, converters |
| [Electro-thermal coupling](electro-thermal.md) | How-to | Joule heating and a heat network; a heat pump example |
| [Power flow](power-flow.md) | How-to | Build, solve and differentiate an AC or DC power flow |
| [Physics](physics.md) | Explanation | The nodal formulation, device laws, sign and phasor conventions, the power-flow model |
| [Modelica circuits](modelica.md) | How-to and reference | Load OpenModelica exports of MSL and MBL circuits; what is read and refused |
| [MATPOWER files](matpower.md) | How-to and reference | Import a MATPOWER `.m` case; columns, rules, standard cases |
| [Limitations](limitations.md) | Reference | What is not supported |

Every type the application offers, including the constitutive laws behind the devices, is in
the [catalogue](../../reference/catalogue/electrical.md) and the
[API reference](../../reference/api/index.md).
