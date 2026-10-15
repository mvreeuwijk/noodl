# Electrical circuits

Lumped electrical circuits: nodes carry voltages, devices between them carry currents, and
Kirchhoff's current law holds at every node. The whole circuit is **one** `PotentialFlowLayer`
whose potential is the node voltage, solved by Modified Nodal Analysis: the currents of
voltage sources, inductors, transformers and controlled sources are unknowns alongside the
node voltages (see
[Modified Nodal Analysis](../concepts/layers.md#modified-nodal-analysis-implicit-coupled-and-dynamic-laws)).
A circuit is analysed at its DC operating point, as a transient on a fixed time grid, or in the
sinusoidal steady state (AC, by phasors). The device laws transcribe the Modelica Standard
Library (MSL, `Modelica.Electrical.Analog`).

**Units.** Voltage V, current A, resistance ohm, conductance S, capacitance F, inductance H,
time s, frequency Hz, temperature K.

**Precision.** A constitutive law stores a parameter given as a Python float in torch's default
dtype, which is float32 unless changed. `build_model` passes float64 values, but a circuit
assembled by hand from laws (as in the examples from
[Impedances, constant-power loads and converters](#impedances-constant-power-loads-and-converters)
on) should pass float64 tensors, `torch.tensor(x, dtype=torch.float64)`, or call
`torch.set_default_dtype(torch.float64)` first: a float32 parameter next to float64 drivers
limits the result to about 7 significant digits.

```py
from noodl.apps.electrical import (
    Circuit, Resistor, Conductor, Capacitor, Inductor, VoltageSource, CurrentSource, Diode,
    Switch, IdealTransformer, Transformer, ControlledSource,
    Constant, Step, Ramp, Sine, Pulse, Signal,
    build_model, operating_point, initial_state, transient, ac_sweep, device_values,
)
```

Every type this application offers is listed in its [catalogue](../catalogue/electrical.md),
including the constitutive laws behind the devices (`Resistance`, `Capacitance`,
`PrescribedVoltage`, ...), which can be used in any `PotentialFlowLayer`.

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

The source's current is negative: it delivers power (see [Sign conventions](#sign-conventions)).

## Devices

Each device is a frozen dataclass. A one-port device has pins `p` and `n`; a two-port device
has `p1`, `n1` (port 1) and `p2`, `n2` (port 2). Parameters are scalars (a float or a 0-d
tensor).

| Device | Law (`v = v_p - v_n`, `i` entering `p`) | Constitutive law (kind) | MSL model | AC |
|---|---|---|---|---|
| `Resistor(name, p, n, R, alpha=0, T_ref=300.15)` | `v = R (1 + alpha (T - T_ref)) i` | `Resistance` (`resistor`) | `Basic.Resistor` | yes |
| `Conductor(name, p, n, G)` | `i = G v` | `Conductance` (`conduction`) | `Basic.Conductor` | yes |
| `Capacitor(name, p, n, C, v0=0)` | `i = C dv/dt` | `Capacitance` (`capacitor`) | `Basic.Capacitor` | yes |
| `Inductor(name, p, n, L, i0=0)` | `v = L di/dt` | `Inductance` (`inductor`) | `Basic.Inductor` | yes |
| `VoltageSource(name, p, n, V, waveform, offset=0)` | `v = offset + V unit(t)` | `PrescribedVoltage` (`voltage_source`) | `Sources.*Voltage` | `Constant`, `Sine` |
| `CurrentSource(name, p, n, I, waveform, offset=0)` | `i = offset + I unit(t)` | `PrescribedCurrent` (`current_source`) | `Sources.*Current` | `Constant`, `Sine` |
| `Diode(name, p, n, Ids, Vt, Maxexp, R)` | `i = Ids (exlin(v/Vt) - 1) + v/R` | `JunctionDiode` (`diode`) | `Semiconductors.Diode` | no |
| `Switch(name, p, n, control, Ron, Goff)` | closed `v = Ron i`, open `i = Goff v` | `SwitchedConductance` (`switch`) | `Ideal.IdealClosingSwitch` | yes |
| `IdealTransformer(name, p1, n1, p2, n2, n)` | `v1 = n v2`, `i2 = -n i1` | `TransformerRatio` (`ideal_transformer`) | `Ideal.IdealTransformer` | yes |
| `Transformer(name, p1, n1, p2, n2, L1, L2, M, i0)` | `v1 = L1 di1/dt + M di2/dt`, `v2 = M di1/dt + L2 di2/dt` | `MutualInductance` (`transformer`) | `Basic.Transformer` | yes |
| `ControlledSource(name, p1, n1, p2, n2, gain, control, output)` | port 2 imposes `gain` times port 1's voltage or current | `DependentSource` (`vcvs`, `vccs`, `ccvs`, `cccs`) | `Basic.VCV`, `VCC`, `CCV`, `CCC` | yes |

A `ControlledSource` senses on port 1 (`control="voltage"`: an open circuit; `"current"`: a
short) and imposes on port 2 (`output="voltage"` or `"current"`). A `Switch` is closed where
its control driver (a key, read at index 0 of its last dimension, or a `(key, index)` pair)
exceeds 0.5. A `Resistor` reads its temperature from the driver named by
`build_model(..., temperature=...)`, one value per resistor in device order; without it every
resistor is at `T_ref`.

### Waveforms

A source's value is `offset + amplitude * unit(t)`, where the waveform supplies `unit(t)` with
the semantics of the MSL `Blocks.Sources` signal of the same name:

| Waveform | `unit(t)` | Breakpoints |
|---|---|---|
| `Constant()` | 1 (the default: a DC source) | none |
| `Step(start_time=0)` | 0 before `start_time`, then 1 | `start_time` |
| `Ramp(duration, start_time=0)` | 0, rising linearly to 1 over `duration`, then 1 | both ends |
| `Sine(frequency, phase=0, start_time=0)` | 0 before `start_time`, then `sin(2 pi f (t - start_time) + phase)` | `start_time` |
| `Pulse(width, period, start_time=0, n_periods=-1)` | 1 for the first `width` percent of every `period`, else 0 | every edge |
| `Signal(key, index=0)` | read from the driver `drivers[key][..., index]` | none |

`Signal` is how a source follows an external input or how a batch of circuits differs (see
[Batches and gradients](#batches-and-gradients)).

## The three analyses

### DC operating point

`operating_point(model, drivers, t=0.0)` solves the circuit with capacitors open and inductors
shorted, sources at their value at time `t` (`Model.steady(..., t=t)` underneath).
`Model.steady` called without `t` on a circuit with a time-dependent source raises, naming the
source kind.

### Transient

`initial_state(model, drivers, t=0.0)` gives the consistent initial point: every capacitor at
its `v0`, every inductor and transformer winding at its `i0`, and the other voltages and
currents consistent with those. `transient(model, state, drivers, times)` then takes one
`Model.step` per interval of the grid `times`.

At the initial point a capacitor acts as a voltage source at its `v0` and an inductor as a
current source at its `i0`. A capacitor directly across a voltage source or another
capacitor, two inductors in series, or an inductor in series with a current source therefore
make the initial point singular, and `initial_state` raises a
`RuntimeError` saying so. Start such a transient from `operating_point(model, drivers)`
instead: the operating point is a consistent state, with each capacitor at its DC voltage
and each inductor at its DC current rather than at `v0` and `i0`.

An RC circuit charged by a step:

```python
import math

from noodl.apps.electrical import Capacitor, Step, initial_state, transient

rc = Circuit([
    VoltageSource("V1", "in", "0", 1.0, waveform=Step(start_time=0.0)),
    Resistor("R1", "in", "out", 1e3),
    Capacitor("C1", "out", "0", 1e-6),
])
model, state, drivers = build_model(rc)
run = transient(model, initial_state(model, drivers), drivers,
                [k * 1e-5 for k in range(301)])

out = model.net.nodes.index("out")
tau = 1e3 * 1e-6
for k in (100, 200, 300):
    t = float(run["time"][k])
    print(f"t = {t * 1e3:.1f} ms: v = {float(run['electrical.phi'][k, out]):.6f} V, "
          f"exact {1.0 - math.exp(-t / tau):.6f} V")
print(run["integration"][:3])
```

```text
t = 1.0 ms: v = 0.632105 V, exact 0.632121 V
t = 2.0 ms: v = 0.864660 V, exact 0.864665 V
t = 3.0 ms: v = 0.950212 V, exact 0.950213 V
['backward_euler', 'trapezoidal', 'trapezoidal']
```

The result holds `"time"`, `"electrical.phi"` and `"electrical.q"` with the start state as row
0, and `"integration"`, the rule each step used. `operating_point`, `initial_state`,
`transient` and `ac_sweep` pass any further keyword (Newton's `atol` and `rtol`, `method`, ...)
to every layer solve.

### AC (phasor) analysis

`ac_sweep(circuit, frequencies)` solves the sinusoidal steady state at every frequency in one
batched phasor solve and returns `{"frequency": (F,), "electrical.phi": (F, n),
"electrical.q": (F, b)}`, complex phasors with the frequency as the leading dimension. A
series RLC circuit at, and either side of, its resonance `f0 = 1 / (2 pi sqrt(L C))`:

```python
import torch

from noodl.apps.electrical import Inductor, Sine, ac_sweep

R, L, C = 10.0, 10e-3, 1e-6
series = Circuit([
    VoltageSource("V1", "in", "0", 1.0, waveform=Sine(frequency=50.0)),
    Resistor("R1", "in", "a", R),
    Inductor("L1", "a", "b", L),
    Capacitor("C1", "b", "0", C),
])
f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
f = torch.tensor([100.0, f0, 1e4], dtype=torch.float64)
sweep = ac_sweep(series, f)

model, _, _ = build_model(series, phasor=True, frequency=f)
current = sweep["electrical.q"][:, model.circuit_index["R1"][1].start]
w = 2.0 * math.pi * f
exact = 1.0 / (R + 1j * w * L + 1.0 / (1j * w * C))
for k in range(3):
    print(f"f = {float(f[k]):7.1f} Hz: |I| = {float(current[k].abs()) * 1e3:8.4f} mA, "
          f"arg I = {round(math.degrees(float(current[k].angle())), 3) + 0.0:+.3f} deg, "
          f"exact: {bool((current[k] - exact[k]).abs() < 1e-12 * exact[k].abs())}")
```

```text
f =   100.0 Hz: |I| =   0.6308 mA, arg I = +89.639 deg, exact: True
f =  1591.5 Hz: |I| = 100.0000 mA, arg I = +0.000 deg, exact: True
f = 10000.0 Hz: |I| =   1.6327 mA, arg I = -89.064 deg, exact: True
```

The source's `Sine(frequency=50.0)` gives its amplitude and phase; the sweep frequency
governs, not the waveform's own (an AC sweep varies the frequency by design). For per-device
values at one frequency, build the phasor model directly and read `device_values`. At
resonance the resistor absorbs `P = (1/2) |V| |I| = 0.05 W` and the inductor stores reactive
power only:

```python
model, _, drivers = build_model(series, phasor=True, frequency=f0)
op = operating_point(model, drivers)
values = device_values(model, op)
for name in ("R1", "L1"):
    v, i = values[name]
    S = 0.5 * v * i.conj()
    print(f"{name}: P = {round(float(S.real), 9) + 0.0:.6f} W, "
          f"Q = {round(float(S.imag), 9) + 0.0:.6f} var")
```

```text
R1: P = 0.050000 W, Q = 0.000000 var
L1: P = 0.000000 W, Q = 0.500000 var
```

## Sign conventions

- A one-port device `p -> n` is one edge of the network. Its voltage is `v = v_p - v_n` and its
  current `i` is the current entering pin `p` (and leaving pin `n`): the MSL convention, under
  which every device absorbs the power `v i`.
- A source that holds node `a` at `+V` relative to ground is the device `p = a`, `n = ground`.
  A source delivering power carries a negative current, as `V1` does in the divider.
- A two-port device is two consecutive edges, port 1 `p1 -> n1` then port 2 `p2 -> n2`;
  `device_values` gives its voltage and current as pairs `(port 1, port 2)`.
- `"electrical.q"` is ordered by kind (kinds in order of first appearance), then by device;
  `model.circuit_index[name]` is the device's `(kind, slice)` in that order.

**Phasors** are peak-amplitude and sine-referenced: the complex value `X` stands for the time
signal `x(t) = |X| sin(omega t + arg X)`, matching the `Sine` waveform. The complex power a
device absorbs is therefore `S = (1/2) V conj(I)` (W), as in the example above. A source has one
phase, its waveform's: a `Sine(frequency, phase, start_time)` source is the phasor
`amplitude * exp(j (phase - omega start_time))`, the steady state of its time signal; a
`Constant` source is `amplitude` at phase 0 (the AC magnitude, as SPICE's `AC` value; its
`offset` has no phasor). Other waveforms are refused in an AC analysis, by name, as are
nonlinear devices (`Diode`).

The power laws below (`ConstantPowerLoad`, `PowerConverter`) are the exception: they define
complex power as `S = dp conj(q)`, without the factor 1/2, so they assume root-mean-square
phasors. A circuit that mixes a power law with a `Sine` source must give that source its RMS
amplitude (`V_peak / sqrt(2)`): otherwise the source's phasor is its peak value, which the power
laws read as an RMS value, a factor `sqrt(2)` in voltage. Linear laws hold for either scaling. The [Modelica
reader](#reading-modelica-circuits) uses MBL's RMS phasors throughout.

## Impedances, constant-power loads and converters

Four further constitutive laws transcribe the building-scale electrical models of the
Modelica Buildings Library (MBL, `Buildings.Electrical`). They are not circuit devices: use
them directly as laws of a `PotentialFlowLayer`, alongside any of the laws above.

| Law (default kind) | Relation (`dp` across the edge, `q` entering its first node) | MBL model |
|---|---|---|
| `Impedance(R, L=0, C=inf)` (`impedance`) | `dp = (R + jX) q`, `X = omega L - 1/(omega C)`; `dp = R q` on a real layer (a reactance is refused there); `R = 0` is a short | `AC.OnePhase.Loads.Impedance`, `DC.Loads.Resistor` |
| `ConstantPowerLoad(P, Q=0, V_nominal=None, linearized=False, direct_current=False, scale=None)` (`load`) | `q = conj(S / dp)` with `S = P + jQ` absorbed (positive = consumption), or MBL's linearised forms | `DC.Loads.Conductor`, `AC.OnePhase.Loads.Inductive`, `Resistive` |
| `ImpedanceTransformer(n, impedance)` (`impedance_transformer`) | two edges `[primary -> 0, secondary -> 0]`: `dp1 = Z q1 + n dp2`, `q2 = -n q1` | `AC.OnePhase.Conversion.ACACTransformer` |
| `PowerConverter(voltage_ratio, efficiency)` (`power_converter`) | two edges `[ac -> 0, dc 1 -> dc 2]`: `dp_dc = k \|dp_ac\|`, power balance with loss `(1 - eta)` of the feeding side, unity power factor | `AC.OnePhase.Conversion.ACDCConverter` |

`ImpedanceTransformer` and `PowerConverter` work in phasor layers only; a converter's DC side is
a node of the same phasor layer whose imaginary part comes out zero. A `ConstantPowerLoad`
given `V_nominal` returns MBL's homotopy start form below `1e-3 V_nominal`, so a solve may start
from zero potentials; without it the load is singular at zero voltage and the solve needs a
nonzero start (`phi0`). A `PowerConverter`'s power row has no slope at zero potentials: start
it from a flat guess, every node at the source voltage. A `ConstantPowerLoad`'s `scale` names
a driver that multiplies the power per edge (a load profile).

**Phasor scaling.** The power laws (`ConstantPowerLoad`, `PowerConverter`) define complex
power as `S = dp conj(q)`, so they assume root-mean-square phasors, as MBL's are; a
converter's `voltage_ratio` is the DC voltage over the AC RMS voltage, and `PowerConverter` is
valid only with RMS phasors. In a circuit with a power law, give source amplitudes as RMS
values (a `Sine` source's phasor is otherwise its peak, as above). `Impedance` and
`ImpedanceTransformer` are linear and hold for either scaling. At zero frequency an
`Impedance`'s inductance is a short and a finite capacitance is refused (an open circuit). A
120 V (RMS) feeder with a
0.1 ohm, 0.5 ohm-reactance line and a 1 kW, 300 var load:

```python
import math

import torch

from noodl.apps.electrical import ConstantPowerLoad, Impedance, PrescribedVoltage
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

f64 = torch.float64
net = Network(dtype=f64)
for node in ("source", "load", "0"):
    net.add_node(node)
net.add_edge("source", "0", kind="voltage_source")
net.add_edge("source", "load", kind="impedance")
net.add_edge("load", "0", kind="load")
laws = [PrescribedVoltage(torch.tensor(120.0, dtype=f64)),
        Impedance(torch.tensor(0.1, dtype=f64),
                  L=torch.tensor(0.5 / (2 * math.pi * 60.0), dtype=f64)),
        ConstantPowerLoad(torch.tensor(1000.0, dtype=f64), torch.tensor(300.0, dtype=f64),
                          V_nominal=torch.tensor(120.0, dtype=f64))]
feeder = PotentialFlowLayer(net, "electrical", laws, boundary=["0"], phasor=True)
phi, q = feeder.solve(torch.zeros(1, dtype=torch.complex128),
                      {"electrical.frequency": torch.tensor(60.0, dtype=f64)},
                      differentiable=False)
v = phi[net.nodes.index("load")]
S = v * q[2].conj()
print(f"|V| = {float(v.abs()):.4f} V, arg V = {math.degrees(float(v.angle())):+.4f} deg")
print(f"absorbed: P = {float(S.real):.6f} W, Q = {float(S.imag):.6f} var")
```

```text
|V| = 117.8116 V, arg V = -1.9052 deg
absorbed: P = 1000.000000 W, Q = 300.000000 var
```

Two options of the circuit laws serve the same models: `Resistance(..., scale=key)` multiplies
`R` by a per-edge driver (MSL `Basic.VariableResistor`), and
`JunctionDiode(..., temperature=key, N, TNOM, XTI, EG)` makes the diode current depend on the
junction temperature read from a driver (MSL `Semiconductors.Diode` with
`useTemperatureDependency = true`: `Vt = k T / e`, saturation current scaled by
`(T/TNOM)^(XTI/N) exp((T/TNOM - 1) EG / (N Vt))`).

## Reading Modelica circuits

`noodl.apps.electrical.modelica.read_modelica_circuit` reads an OpenModelica export of an
electrical example model, from the Modelica Standard Library (`Modelica.Electrical.Analog`,
with its heat ports) or from the Modelica Buildings Library (`Buildings.Electrical`, DC, one-phase
AC and balanced three-phase AC), into a model of the laws above. The export format, the
supported and refused classes, how connectors become nodes and how the start state is chosen
are described under [File formats: Modelica](../formats/modelica.md#electrical-circuits).

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
`read_modelica_circuit` returns, besides the model, its start state and drivers, a
`CircuitNames`: `times` (the reference's output times), `grid` (the step grid), `nodes` and
`devices` (where each node and device sits in `"electrical.phi"` and `"electrical.q"`), and
`value(history, column)`, which reads any column of the reference CSV from a history. A
column of an MBL terminal is a conductor potential or current (`v[1]`, `v[2]`, `i[1]`) or, in
an AC model, the real or imaginary part of a phasor.

`simulate_circuit(model, state, drivers, names, *, times=None)` solves a model without
storage at each output time and steps a dynamic one over `names.grid`. Unlike
[`transient`](#time-integration-and-breakpoints), the reader's step grid contains every jump
and kink of the sources and signals as well as the output times, so no edge falls inside a
step: the step ending at a jump uses the signal's value just before it, and the step starting
there is backward Euler (as is the first). `read_modelica_circuit(path, substeps=r)` splits each
interval into `r` steps; two runs at `r` and `2 r` combined by
`noodl.apps.electrical.modelica.extrapolate(coarse, fine, order=2)` remove the leading
time-step error, which is how the comparison with OpenModelica reaches `1e-6` of each
quantity's range. A model with heat ports gets the thermal layer and the coupling of the next
section; its heat-port and heat-capacitor temperatures are in `"thermal.phi"`.

## Electro-thermal coupling

A device that heats up and whose resistance follows its temperature couples the circuit to a
thermal network. noodl puts the thermal side on the same `Network` as a second
`PotentialFlowLayer`, conventionally named `"thermal"`: temperature (K) is the potential against
a reference node held at 0 K (a boundary node), heat flow (W) the flow.

| Thermal part | Law | Edge | Modelica (MSL `Thermal.HeatTransfer`) |
|---|---|---|---|
| heat conductance `G` (W/K) | `Conductance(G)` | between the two nodes | `Components.ThermalConductor` |
| heat capacity `C` (J/K) | `Capacitance(C, v0=T0)` | node `->` 0 K reference | `Components.HeatCapacitor` |
| fixed temperature | a boundary node | | `Sources.FixedTemperature` |
| a device's Joule heat | `PrescribedCurrent(1, waveforms=[Signal("thermal.heat", k)])` | 0 K reference `->` heat-port node | the device's `heatPort` |

Give each thermal law a kind of its own (`kind="conductor"`, `"heat_capacitor"`,
`"heat_source"`): the electrical kinds `capacitor` and `current_source` belong to the circuit
layer. Both layers use the same companion forms, so a heat capacity is stepped by backward
Euler or trapezoidal steps together with the circuit.

`JouleHeating(heated, base_temperatures=...)` is the closure that couples them. Each
`HeatedEdge(edge, p, n, heat_node, temperature_key, slot)` names a heated device: its edge in
the circuit layer's `q`, its end nodes, the thermal node of its heat port, and the per-edge
temperature driver its law reads (`Resistance(..., temperature=key)`,
`JunctionDiode(..., temperature=key)`) with its position in it. From the state the closure
writes the dissipated power `p = Re((phi[p] - phi[n]) conj(q[edge]))` (MSL `LossPower = v i`)
into `"thermal.heat"`, and each temperature driver as the driver you give (or
`base_temperatures[key]` when the drivers lack it) with the heated devices' entries replaced
by their heat-port temperatures (devices without a heat port keep the temperature you give). Build the model with `coupling="iterate"`, so that every step (or
steady solve) repeats circuit and thermal solves until both stop changing.

A resistor `R(T) = R0 (1 + alpha (T - T_ref))` on 220 V, cooled through `G = 50` W/K to
293.15 K, settles where its Joule heat `V^2 / R(T)` equals `G (T - 293.15)`, a quadratic in
the temperature rise `x`: `alpha x^2 + x - V^2 / (G R0) = 0`.

```python
from noodl.apps.electrical import (
    HeatedEdge, JouleHeating, PrescribedCurrent, PrescribedVoltage, Resistance, Signal,
)
from noodl.elements import Conductance
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network


def f64(x):
    """A float64 parameter or driver (a Python float would be stored as float32)."""
    return torch.tensor(x, dtype=torch.float64)


V, R0, alpha, T_ref, G, T_amb = 220.0, 100.0, 1e-3, 293.15, 50.0, 293.15
net = Network(dtype=torch.float64)
for name in ("0", "s", "thermal.reference", "h", "ambient"):
    net.add_node(name)
net.add_edge("s", "0", kind="voltage_source")
net.add_edge("s", "0", kind="resistor")
net.add_edge("h", "ambient", kind="conductor")
net.add_edge("thermal.reference", "h", kind="heat_source")
node = {name: k for k, name in enumerate(net.nodes)}

electrical = PotentialFlowLayer(
    net, "electrical",
    [PrescribedVoltage(f64(V)),
     Resistance(f64(R0), alpha=f64(alpha), T_ref=f64(T_ref),
                temperature="electrical.resistor.T")],
    boundary=["0"])
thermal = PotentialFlowLayer(
    net, "thermal",
    [Conductance(f64(G), kind="conductor"),
     PrescribedCurrent(f64(1.0), waveforms=[Signal("thermal.heat", 0)], kind="heat_source")],
    boundary=["thermal.reference", "ambient"])
joule = JouleHeating(
    [HeatedEdge(edge=1, p=node["s"], n=node["0"], heat_node=node["h"],
                temperature_key="electrical.resistor.T", slot=0)],
    base_temperatures={"electrical.resistor.T": f64([T_ref])})
model = Model(net, [electrical, thermal], closures=[joule], coupling="iterate",
              iterate_tol={"electrical": 1e-10, "thermal": 1e-10},
              iterate_flow_tol={"electrical": 1e-10, "thermal": 1e-8},
              iterate_acceleration="anderson", iterate_max=100)

zeros = torch.zeros(len(net.nodes), dtype=torch.float64)
state = {"electrical.phi": zeros, "electrical.q": torch.zeros(2, dtype=torch.float64),
         "thermal.phi": zeros, "thermal.q": torch.zeros(2, dtype=torch.float64)}
drivers = {"electrical.phi_boundary": f64([0.0]), "thermal.phi_boundary": f64([0.0, T_amb]),
           "thermal.heat": f64([0.0]), "electrical.resistor.T": f64([T_ref])}
out = model.steady(state, drivers)

rise = (-1.0 + math.sqrt(1.0 + 4.0 * alpha * V**2 / (G * R0))) / (2.0 * alpha)
print(f"resistor at {float(out['thermal.phi'][node['h']]):.9f} K "
      f"(closed form {T_amb + rise:.9f} K)")
print(f"current {float(out['electrical.q'][1]):.9f} A")
```

```text
resistor at 302.738068934 K (closed form 302.738068934 K)
current 2.179106576 A
```

`"thermal.phi_boundary"` holds the boundary temperatures in the order of `boundary=`, the 0 K
reference first. The Modelica reader builds the same structure for MSL models whose resistors
or diodes have an enabled heat port (`Analog.Examples.Resistor`, `HeatingRectifier`).

## Example: a heat pump's electrical load

The coupling closure need not be `JouleHeating`: any function of the state that writes
drivers couples the two layers. A heat pump delivers a fixed 6 kW of heat to a zone (heat
capacity 1e7 J/K, loss 150 W/K to 273.15 K outdoors) and draws the electrical power
`P = Q_th / COP` from a 400 V DC bus through a 0.2 ohm feeder, with a coefficient of
performance `COP = 0.45 T_zone / (T_zone - T_out)` (45 % of the Carnot value) that falls as
the zone warms. The heat pump is a `ConstantPowerLoad` of 1 W whose `scale` driver the closure
sets to `P`; the heat input is a `PrescribedCurrent` reading the driver `"thermal.heat"`. One
`Network` holds both layers; the model steps a day in hours, backward Euler first:

```python
from noodl.apps.electrical import Capacitance, ConstantPowerLoad, Impedance

net = Network(dtype=torch.float64)
for name in ("0", "bus", "hp", "thermal.reference", "zone", "outdoor"):
    net.add_node(name)
net.add_edge("bus", "0", kind="voltage_source")
net.add_edge("bus", "hp", kind="impedance")
net.add_edge("hp", "0", kind="load")
net.add_edge("zone", "outdoor", kind="conductor")
net.add_edge("thermal.reference", "zone", kind="heat_source")
net.add_edge("zone", "thermal.reference", kind="heat_capacitor")
node = {name: k for k, name in enumerate(net.nodes)}
zone = node["zone"]

electrical = PotentialFlowLayer(
    net, "electrical",
    [PrescribedVoltage(f64(400.0)), Impedance(f64(0.2)),
     ConstantPowerLoad(f64(1.0), V_nominal=f64(400.0), direct_current=True,
                       scale="electrical.load.scale")],
    boundary=["0"])
thermal = PotentialFlowLayer(
    net, "thermal",
    [Conductance(f64(150.0), kind="conductor"),
     PrescribedCurrent(f64(1.0), waveforms=[Signal("thermal.heat", 0)], kind="heat_source"),
     Capacitance(f64(1e7), v0=f64(293.15), kind="heat_capacitor")],
    boundary=["thermal.reference", "outdoor"])


class HeatPumpLoad:
    """Delivers Q_th to the zone; draws P = Q_th / COP from the bus, COP = 0.45 T_zone/(T_zone - T_out)."""

    def __init__(self, q_th, t_out, zone):
        self.q_th, self.t_out, self.zone = q_th, t_out, zone

    def __call__(self, state, drivers):
        t_zone = state["thermal.phi"][..., self.zone]
        cop = 0.45 * t_zone / (t_zone - self.t_out)
        return {"thermal.heat": self.q_th.expand(t_zone.shape + (1,)),
                "electrical.load.scale": (self.q_th / cop).unsqueeze(-1)}


model = Model(net, [thermal, electrical],
              closures=[HeatPumpLoad(f64(6000.0), 273.15, zone)], coupling="iterate",
              iterate_tol={"thermal": 1e-9, "electrical": 1e-9})

phi = torch.zeros(len(net.nodes), dtype=torch.float64)
phi[[node["bus"], node["hp"]]] = 400.0
phi[[node["zone"], node["outdoor"]]] = f64([293.15, 273.15])
state = {"electrical.phi": phi, "electrical.q": torch.zeros(3, dtype=torch.float64),
         "thermal.phi": phi.clone(), "thermal.q": torch.zeros(3, dtype=torch.float64)}
drivers = {"electrical.phi_boundary": f64([0.0]),
           "thermal.phi_boundary": f64([0.0, 273.15]),
           "thermal.heat": f64([6000.0]), "electrical.load.scale": f64([1.0])}

for hour in range(24):
    state = model.step(state, drivers, dt=3600.0, t=3600.0 * hour,
                       integration="backward_euler" if hour == 0 else "trapezoidal")

v = state["electrical.phi"]
print(f"zone after 24 h: {float(state['thermal.phi'][zone]):.2f} K")
print(f"heat pump draws {float(v[node['hp']] * state['electrical.q'][2]):.1f} W")
print(f"bus to heat pump drop: {float(v[node['bus']] - v[node['hp']]):.3f} V")
```

```text
zone after 24 h: 307.67 K
heat pump draws 1496.0 W
bus to heat pump drop: 0.749 V
```

The zone approaches its steady 313.15 K with the time constant `C / G` = 18.5 h; at 307.67 K
the COP is 4.01, so the 6 kW of heat cost 1.50 kW, carried as 3.75 A through the feeder. The
same closure on a phasor layer would set an AC load's power, with source amplitudes given as
RMS values (see [Phasors](#sign-conventions)).

## Time integration and breakpoints

`transient(..., integration="trapezoidal")` (the default) uses the trapezoidal rule, second
order, except on the first step and on every step whose interval contains a source breakpoint
(a step, ramp or pulse edge, the start of a sine), which use backward Euler. The trapezoidal rule
rings after a discontinuity (it is A-stable, not L-stable); one backward Euler step damps that.
This is the practice of SPICE. `integration="backward_euler"` uses backward Euler throughout.
A pulse-driven RC circuit restarts at every edge:

```python
from noodl.apps.electrical import Pulse

pulsed = Circuit([
    VoltageSource("V1", "in", "0", 1.0, waveform=Pulse(width=50.0, period=2e-3)),
    Resistor("R1", "in", "out", 1e3),
    Capacitor("C1", "out", "0", 1e-6),
])
model, state, drivers = build_model(pulsed)
run = transient(model, initial_state(model, drivers), drivers,
                [k * 1e-5 for k in range(401)])
print([k for k, rule in enumerate(run["integration"]) if rule == "backward_euler"])
```

```text
[0, 99, 199, 299, 399]
```

Step 99 covers 0.99 ms to 1.00 ms and contains the falling edge at 1 ms.

The grid is **fixed**: a breakpoint inside a step is not inserted as a time point, so the step
that contains it sees the source at the step end. Put the breakpoints on the grid (as above)
when the edge timing matters.

A circuit with no capacitor or inductor has no storage, and each step solves it at its operating
point at the step end. `drivers` is one mapping for the whole run, so a `Signal` source holds
its driver value throughout a transient; a time-varying input is a waveform.

## Batches and gradients

Device parameters are scalars, so a batch of circuits is a batch of **drivers**: a `Signal`
source, a switch control or a resistor temperature with a leading batch dimension. Three supply
voltages in one solve:

```python
from noodl.apps.electrical import Signal

batched = Circuit([
    VoltageSource("V1", "in", "0", 1.0, waveform=Signal("supply")),
    Resistor("R1", "in", "out", 3.0),
    Resistor("R2", "out", "0", 7.0),
])
model, _, drivers = build_model(batched)
drivers["supply"] = torch.tensor([[5.0], [10.0], [12.0]], dtype=torch.float64)
op = operating_point(model, drivers)
print(op["electrical.phi"][:, model.net.nodes.index("out")].tolist())
```

```text
[3.5, 7.0, 8.4]
```

`build_model(..., learnable=True)` registers each law's principal parameters (resistance,
conductance, capacitance, inductance, source amplitudes, turns ratio, `L1`, `L2`, `M`, gain;
one entry per device) as `nn.Parameter`s, and every analysis is differentiable through the
implicit-function adjoint, a transient through every step. The divider's output voltage
`v_out = V R2 / (R1 + R2)` has `d v_out / d R1 = -0.7` and `d v_out / d R2 = 0.3` V/ohm:

```python
model, _, drivers = build_model(divider, learnable=True)
op = operating_point(model, drivers)
v_out = op["electrical.phi"][model.net.nodes.index("out")]
resistors = model.potential["electrical"].element_for("resistor")[0]
(grad,) = torch.autograd.grad(v_out, resistors.R)
print([round(g, 12) for g in grad.tolist()])
```

```text
[-0.7, 0.3]
```

## Checks

`Circuit` refuses, naming the device: an unknown device type, a duplicate or empty name, a port
whose two pins are one node, a non-scalar parameter, a waveform that is not one, a malformed
switch control or controlled-source mode, and nodes with no path to ground (an isolated
transformer winding needs a resistor to ground). A loop of voltage sources or a cut set of
current sources passes those checks and is refused at the solve with a `RuntimeError` naming the
layer and the implicit kinds.

## Verification

Every check is against a closed form or a hand Modified Nodal Analysis (`tests/apps/electrical/`,
`tests/layers/test_mna*.py`):

| Circuit | Compared | Tolerance |
|---|---|---|
| Divider; balanced and unbalanced Wheatstone bridge; floating voltage source between two non-ground nodes | Node voltages, bridge current | 1e-12 V or A abs |
| Controlled sources (all four), ideal transformer, coupled windings at DC | Port voltages and currents against hand MNA | 1e-12 abs |
| RC charge, RL, backward Euler | The exact backward Euler recurrence at every step | 1e-13 abs |
| RC charge and series RLC ringing, trapezoidal | Convergence order over `dt`, `dt/2`, `dt/4` | order 2 within 0.1 |
| RC charge through `transient` (backward Euler first step) | Error at 1 s with `dt = 0.01` and order from `dt = 0.02` | error < 1e-4, order >= 1.9 |
| Series RLC, `ac_sweep` | Current modulus against `1 / sqrt(R^2 + (w L - 1/(w C))^2)` at seven frequencies from `f0/2` to `2 f0`, peak at `f0` | 1e-12 relative |
| Series RLC, phasor layer | Complex current against `1 / (R + j w L + 1 / (j w C))` at three frequencies | 1e-12 relative |
| RC with a `Sine` source of phase 0.3 rad, `ac_sweep` | Complex node voltage at three sweep frequencies | 1e-12 relative |
| RC phasor model, `device_values` | Complex device voltages and currents | 1e-13 relative |
| Constant-power load behind an impedance, phasor | Load voltage against a fixed-point iteration; drawn power | 1e-12 relative (voltage); 1e-12 abs (power, of magnitude 0.54) |
| Tellegen identity, real and complex | `power_residual` | 1e-12 |
| Diode with a resistor and capacitor, transient | KCL at the diode node; the diode current against its law | 1e-15 A abs; 1e-12 relative |
| Diode behind a resistor, DC | The resistor and diode laws at the solution | 1e-10 and 1e-9 relative |
| Gradients: a capacitance and a source value through three transient steps; a conductance, a source and a boundary potential at DC; a conductance and a source through a phasor magnitude | `torch.autograd.gradcheck` against finite differences | gradcheck defaults |
| Gradients of the learnable divider resistances | Closed form | 1e-10 relative |
| Gradients of an inductance, a mutual inductance, a controlled-source gain and four resistances through `initial_state` and three `transient` steps | Central differences | 1e-8 relative |

Measured: the divider is exact, the backward Euler recurrence holds to 3.3e-16 V, the
trapezoidal order through `transient` is 1.96 to 1.99 (RC) and 2.01 to 2.03 (RLC) over three
halvings of the step, and the series RLC phasor current agrees to 4.9e-14 relative.

### Against OpenModelica

Fourteen example models of the Modelica Standard Library 4.1.0 and the Modelica Buildings Library
v13.0.0, read with [`read_modelica_circuit`](#reading-modelica-circuits), are compared with
OpenModelica 1.27.1 (DASSL at 1e-12) at every row and every column of the reference
(`tests/verification/test_modelica_electrical_parity.py`):

| Models | Bound | Measured (every row) |
|---|---|---|
| 9 algebraic: MSL `Resistor` (heat port, temperature-dependent resistance), `ShowVariableResistor`; MBL DC `Resistor`, `VariableLoad`, AC `TestImpedance`, `ACLine_RL` (dynamic branch omitted), `ACDCConverter`, `ACACTransformer`, balanced three-phase `ACLine` | `1e-9 \|omc\| + 1e-12` | at most 6.2e-13 A or V; 9.5e-12 A and 1.0e-10 K on the thermally coupled `Resistor` |
| 5 dynamic: MSL `SeriesResonance`, `CauerLowPassAnalog`, `CompareTransformers`, `HeatingRectifier`; MBL `RCModel` | `1e-6` of each column's range after the start row, Richardson extrapolation of runs at `r` and `2 r` substeps | 1.7e-8 to 3.1e-7 of the range |

Model by model, with the step counts and the worst columns, on the [verification
page](../verification.md#modelica-electrical-circuits-model-by-model).

## Limitations

- **Fixed time grid.** No adaptive step, and `transient` inserts no breakpoints: a source edge
  inside a step is seen at the step end (the Modelica reader's step grid holds every source
  event).
- **No event location.** A `Switch` is a discontinuous law: it changes state only where its
  control driver is evaluated, at a step end, and the step in which it switches is not refined.
  (The `Diode` is continuously differentiable, its exponential continuing linearly above
  `Maxexp`, and needs no event.)
- **No switching inside a transient.** `transient` takes one `drivers` mapping for the whole
  run, so a `Switch` keeps the state its control driver gives it throughout. To switch at a
  time, step the model yourself: call `model.step(state, drivers, dt, t=t)` per step and pass a
  new `drivers` mapping, with the changed control, from the step that should see it.
- **No ideal diode.** The piecewise-linear ideal diode (MSL `Ideal.IdealDiode`) is not provided;
  `Diode` is the exponential junction diode with its linear continuation.
- **One frequency per phasor model.** No harmonic balance, no small-signal linearisation of a
  nonlinear operating point: `ac_sweep` takes linear devices only. The nonlinear phasor laws
  (constant-power loads, converters) are solved by Newton at one frequency.
- **No dynamic phasors.** A phasor model is quasi-static; MBL's `FixedZ_dynamic` models are
  refused.
- **Lumped, balanced.** No distributed transmission lines; balanced three-phase models are
  read as their one-phase equivalent, and unbalanced three-phase systems are not supported.
- **No batteries, photovoltaics or grid sources** from MBL (refused by name).
