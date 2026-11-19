# Build and analyse a circuit

How to describe a circuit, solve it at DC, step it in time, batch it and differentiate it.
For AC see [AC analysis and power laws](ac.md); the equations are in
[Physics](physics.md).

**Precision.** A constitutive law stores a parameter given as a Python float in torch's
default dtype, which is float32 unless changed. `build_model` passes float64 values, but a
circuit assembled by hand from laws (as in the [AC](ac.md) and
[electro-thermal](electro-thermal.md) guides) should pass float64 tensors,
`torch.tensor(x, dtype=torch.float64)`, or call `torch.set_default_dtype(torch.float64)`
first: a float32 parameter next to float64 drivers limits the result to about 7 significant
digits.

```python
from noodl.apps.electrical import (
    Circuit, Resistor, Conductor, Capacitor, Inductor, VoltageSource, CurrentSource, Diode,
    Switch, IdealTransformer, Transformer, ControlledSource,
    Constant, Step, Ramp, Sine, Pulse, Signal,
    build_model, operating_point, initial_state, transient, ac_sweep, device_values,
)
```

## Devices

Each device is a frozen dataclass. A one-port device has pins `p` and `n`; a two-port device
has `p1`, `n1` (port 1) and `p2`, `n2` (port 2). Parameters are scalars (a float or a 0-d
tensor).

| Device | Law (`v = v_p - v_n`, `i` entering `p`) | Constitutive law (kind) | MSL model | AC |
|---|---|---|---|---|
| `Resistor(name, p, n, R, alpha=0, T_ref=300.15)` | `v = R (1 + alpha (T - T_ref)) i` | `Resistance` (`resistor`) | `Basic.Resistor` | yes |
| `Conductor(name, p, n, G)` | `i = G v` | `Conductance` (`conduction`) | `Basic.Conductor` | yes |
| `Capacitor(name, p, n, C, v0=0)` | `i = C dv/dt` | `LinearStorage` at the other pin when one pin is the ground, else `Capacitance` (`capacitor`) | `Basic.Capacitor` | yes (an edge) |
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
[Batches](#batches-and-gradients)).

## Solve at DC

`operating_point(model, drivers, t=0.0)` solves the circuit with capacitors open and inductors
shorted, sources at their value at time `t`. `Model.steady` called without `t` on a circuit
with a time-dependent source raises, naming the source kind. The
[voltage divider](index.md#a-worked-example-a-voltage-divider) is the example.

## Step in time

`initial_state(model, drivers, t=0.0)` gives the consistent initial point: every capacitor at
its `v0`, every inductor and transformer winding at its `i0`, and the other voltages and
currents consistent with those. `transient(model, state, drivers, times)` then takes one
`Model.step` per interval of the grid `times`.

At the initial point a capacitor acts as a voltage source at its `v0` and an inductor as a
current source at its `i0`. A capacitor directly across a voltage source or another
capacitor, two inductors in series, or an inductor in series with a current source therefore
make the initial point singular, and `initial_state` raises a `RuntimeError` saying so. Start
such a transient from `operating_point(model, drivers)` instead: the operating point is a
consistent state, with each capacitor at its DC voltage and each inductor at its DC current
rather than at `v0` and `i0`.

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
print(run["scheme"][:3])
```

```text
t = 1.0 ms: v = 0.632105 V, exact 0.632121 V
t = 2.0 ms: v = 0.864660 V, exact 0.864665 V
t = 3.0 ms: v = 0.950212 V, exact 0.950213 V
['backward_euler', 'trapezoidal', 'trapezoidal']
```

The result holds `"time"`, `"electrical.phi"` and `"electrical.q"` with the start state as row
0, and `"scheme"`, the rule each step used. `operating_point`, `initial_state`, `transient`
and `ac_sweep` pass any further keyword (Newton's `atol` and `rtol`, `method`, ...) to every
layer solve. The ordering of `"electrical.q"` and the treatment of a grounded capacitor are
in [sign conventions](physics.md#sign-conventions).

### Time stepping and breakpoints

`transient(..., scheme="trapezoidal")` (the default) uses the trapezoidal rule, second order,
except on the first step and on every step whose interval contains a source breakpoint (a
step, ramp or pulse edge, the start of a sine), which use backward Euler. The trapezoidal
rule rings after a discontinuity (it is A-stable, not L-stable); one backward Euler step
damps that. This is the practice of SPICE. `scheme="backward_euler"` uses backward Euler
throughout. `transient` finds the restart steps and hands them to
[`Model.march`](../../guide/time-stepping.md#marching-over-a-time-grid). A pulse-driven RC circuit
restarts at every edge:

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
print([k for k, rule in enumerate(run["scheme"]) if rule == "backward_euler"])
```

```text
[0, 99, 199, 299, 399]
```

Step 99 covers 0.99 ms to 1.00 ms and contains the falling edge at 1 ms.

The grid is **fixed**: a breakpoint inside a step is not inserted as a time point, so the step
that contains it sees the source at the step end. Put the breakpoints on the grid (as above)
when the edge timing matters.

A circuit with no capacitor or inductor has no storage, and each step solves it at its
operating point at the step end. `drivers` is one mapping for the whole run, so a `Signal`
source holds its driver value throughout a transient; a time-varying input is a waveform. To
switch a `Switch` at a time, step the model yourself: call `model.step(state, drivers, dt,
t=t)` per step and pass a new `drivers` mapping, with the changed control, from the step that
should see it.

## Batches and gradients

Device parameters are scalars, so a batch of circuits is a batch of **drivers**: a `Signal`
source, a switch control or a resistor temperature with a leading batch dimension. Three
supply voltages in one solve:

```python
import torch

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
divider = Circuit([
    VoltageSource("V1", "in", "0", 10.0),
    Resistor("R1", "in", "out", 3.0),
    Resistor("R2", "out", "0", 7.0),
])
model, _, drivers = build_model(divider, learnable=True)
op = operating_point(model, drivers)
v_out = op["electrical.phi"][model.net.nodes.index("out")]
resistors = model.kirchhoff["electrical"].law_for("resistor")[0]
(grad,) = torch.autograd.grad(v_out, resistors.R)
print([round(g, 12) for g in grad.tolist()])
```

```text
[-0.7, 0.3]
```

## What a circuit refuses

`Circuit` refuses, naming the device: an unknown device type, a duplicate or empty name, a
port whose two pins are one node, a non-scalar parameter, a waveform that is not one, a
malformed switch control or controlled-source mode, and nodes with no path to ground (an
isolated transformer winding needs a resistor to ground). A loop of voltage sources or a cut
set of current sources passes those checks and is refused at the solve with a `RuntimeError`
naming the layer and the implicit kinds.
