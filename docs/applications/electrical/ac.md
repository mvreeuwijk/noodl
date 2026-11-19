# AC analysis and power laws

How to solve the sinusoidal steady state of a circuit, read complex powers, and use the
building-scale laws (impedances, constant-power loads, converters). The conventions behind
the numbers are in [Physics](physics.md#phasor-conventions). The devices and waveforms are
described in [Build and analyse a circuit](circuits.md).

## Sweep the frequency

`ac_sweep(circuit, frequencies)` solves the sinusoidal steady state at every frequency in one
batched phasor solve and returns `{"frequency": (F,), "electrical.phi": (F, n),
"electrical.q": (F, b)}`, complex phasors with the frequency as the leading dimension. A
series RLC circuit at, and either side of, its resonance `f0 = 1 / (2 pi sqrt(L C))`:

```python
import math

import torch

from noodl.apps.electrical import (
    Capacitor, Circuit, Inductor, Resistor, Sine, VoltageSource, ac_sweep, build_model,
    device_values, operating_point,
)

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
governs, not the waveform's own (an AC sweep varies the frequency by design). Only linear
devices are accepted: other waveforms and the `Diode` are refused by name.

## Read device values at one frequency

For per-device values at one frequency, build the phasor model directly and read
`device_values`. At resonance the resistor absorbs `P = (1/2) |V| |I| = 0.05 W` and the
inductor stores reactive power only:

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

## Impedances, constant-power loads and converters

Four further constitutive laws transcribe the building-scale electrical models of the
Modelica Buildings Library (MBL, `Buildings.Electrical`). They are not circuit devices: use
them directly as laws of a `KirchhoffLayer`, alongside any of the laws in
[Build and analyse a circuit](circuits.md#devices).

| Law (default kind) | Relation (`dp` across the edge, `q` entering its first node) | MBL model |
|---|---|---|
| `Impedance(R, L=0, C=inf)` (`impedance`) | `dp = (R + jX) q`, `X = omega L - 1/(omega C)`; `dp = R q` on a real layer (a reactance is refused there); `R = 0` is a short | `AC.OnePhase.Loads.Impedance`, `DC.Loads.Resistor` |
| `ConstantPowerLoad(P, Q=0, V_nominal=None, linearized=False, direct_current=False, scale=None)` (`load`) | `q = conj(S / dp)` with `S = P + jQ` absorbed (positive = consumption), or MBL's linearised forms | `DC.Loads.Conductor`, `AC.OnePhase.Loads.Inductive`, `Resistive` |
| `ImpedanceTransformer(n, impedance)` (`impedance_transformer`) | two edges `[primary -> 0, secondary -> 0]`: `dp1 = Z q1 + n dp2`, `q2 = -n q1` | `AC.OnePhase.Conversion.ACACTransformer` |
| `PowerConverter(voltage_ratio, efficiency)` (`power_converter`) | two edges `[ac -> 0, dc 1 -> dc 2]`: `dp_dc = k \|dp_ac\|`, power balance with loss `(1 - eta)` of the feeding side, unity power factor | `AC.OnePhase.Conversion.ACDCConverter` |

`ImpedanceTransformer` and `PowerConverter` work in phasor layers only; a converter's DC side
is a node of the same phasor layer whose imaginary part comes out zero. A `ConstantPowerLoad`
given `V_nominal` returns MBL's homotopy start form below `1e-3 V_nominal`, so a solve may
start from zero potentials; without it the load is singular at zero voltage and the solve
needs a nonzero start (`phi0`). A `PowerConverter`'s power row has no slope at zero
potentials: start it from a flat guess, every node at the source voltage. A
`ConstantPowerLoad`'s `scale` names a driver that multiplies the power per edge (a load
profile). At zero frequency an `Impedance`'s inductance is a short and a finite capacitance
is refused (an open circuit).

**RMS phasors.** The power laws (`ConstantPowerLoad`, `PowerConverter`) assume root-mean-square
phasors, as MBL's are; a converter's `voltage_ratio` is the DC voltage over the AC RMS
voltage. In a circuit with a power law, give source amplitudes as RMS values (see
[phasor conventions](physics.md#phasor-conventions)). `Impedance` and `ImpedanceTransformer`
are linear and hold for either scaling.

A 120 V (RMS) feeder with a 0.1 ohm, 0.5 ohm-reactance line and a 1 kW, 300 var load:

```python
import math

import torch

from noodl.apps.electrical import ConstantPowerLoad, Impedance, PrescribedVoltage
from noodl.layers.kirchhoff import KirchhoffLayer
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
feeder = KirchhoffLayer(net, "electrical", laws, boundary=["0"], phasor=True)
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

Two options of the circuit laws serve the same models: `Resistance(..., scale=key)`
multiplies `R` by a per-edge driver (MSL `Basic.VariableResistor`), and
`JunctionDiode(..., temperature=key, N, TNOM, XTI, EG)` makes the diode current depend on the
junction temperature read from a driver (MSL `Semiconductors.Diode` with
`useTemperatureDependency = true`: `Vt = k T / e`, saturation current scaled by
`(T/TNOM)^(XTI/N) exp((T/TNOM - 1) EG / (N Vt))`).
