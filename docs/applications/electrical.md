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
`Model.step` per interval of the grid `times`. An RC circuit charged by a step:

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
0, and `"integration"`, the rule each step used.

### AC (phasor) analysis

`ac_sweep(circuit, frequencies)` solves the sinusoidal steady state at every frequency in one
batched phasor solve. A series RLC circuit at, and either side of, its resonance
`f0 = 1 / (2 pi sqrt(L C))`:

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
| Series RLC, phasor | Current against `1 / (R + j w L + 1 / (j w C))` at three frequencies, resonance included | 1e-12 relative |
| Constant-power load behind an impedance, phasor | Load voltage against a fixed-point iteration; drawn power | 1e-12 relative |
| Tellegen identity, real and complex | `power_residual` | 1e-12 |
| Diode with a resistor and capacitor | KCL and the diode law at the solution | 1e-15 abs, 1e-9 relative |
| Gradients: a capacitance and a source value through three transient steps; a conductance, a source and a boundary potential at DC; a conductance and a source through a phasor magnitude | `torch.autograd.gradcheck` against finite differences | gradcheck defaults |
| Gradients of the learnable divider resistances | Closed form | 1e-10 relative |

Measured: the divider is exact, the backward Euler recurrence holds to 3.3e-16 V, the
trapezoidal order through `transient` is 1.96 to 1.99 (RC) and 2.01 to 2.03 (RLC) over three
halvings of the step, and the series RLC phasor current agrees to 4.9e-14 relative.

No reference circuit simulator has been run against this application; the comparison with
OpenModelica on MSL models is planned.

## Limitations

- **Fixed time grid.** No adaptive step and no inserted breakpoints: a source edge inside a step
  is seen at the step end.
- **No event location.** A switch or a diode changes state at a step end only; the step in which
  a switching law changes branch is not refined.
- **No ideal diode.** The piecewise-linear ideal diode (MSL `Ideal.IdealDiode`) is not provided;
  `Diode` is the exponential junction diode with its linear continuation.
- **One frequency per phasor model.** No harmonic balance, no small-signal linearisation of a
  nonlinear operating point: an AC analysis takes linear devices only.
- **Lumped, single-phase.** No transmission lines, no three-phase power flow.
