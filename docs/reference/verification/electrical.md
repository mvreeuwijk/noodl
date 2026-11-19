# Verification: electrical circuits

Evidence for the [electrical application](../../applications/electrical/index.md); the summary is on the [verification overview](index.md).

## Modelica electrical circuits, model by model

Fourteen example models of the Modelica Standard Library 4.1.0 (`Modelica.Electrical.Analog`)
and the Modelica Buildings Library v13.0.0 (`Buildings.Electrical`), simulated by OpenModelica
1.27.1 with DASSL at a relative tolerance of 1e-12 and committed as CSV
(`tests/data/modelica_electrical/`), are read with `read_modelica_circuit` and compared column
by column (`tests/verification/test_modelica_electrical_parity.py`). The columns are the
currents of the MSL devices (with capacitor voltages, heat-port and heat-capacitor
temperatures) and the conductor potentials and currents of every MBL terminal, real and
imaginary parts for AC; see [the reader's page](../../applications/building/modelica.md#electrical-circuits).

- **Algebraic models** (no capacitor or inductor, so each output time is the same algebraic
  problem for both codes): every column at every row within `|noodl - omc| <= 1e-9 |omc| +
  1e-12` (in V, A or K). The steady solves run at Newton `atol = 1e-12`, `rtol = 1e-14` (the
  heat layer of a thermal network is a linear solve).
- **Dynamic models**: every column at every row after the start time within `1e-6` of the
  column's range (its largest magnitude over the run), from two runs at `r` and `2 r` steps per
  interval of the step grid combined by Richardson extrapolation (order 2). The start row is
  the reference's own initial state and is not compared. `r` is the smallest power of 2 that
  meets the bound with margin, checked against the next finer pair (the two extrapolations
  agree to 1.5e-8 to 2.7e-7 of the range, and the error falls by 8 to 16 per halving). No
  model needed a bound of its own.

Measured, from the committed record `tests/data/modelica_electrical/parity-electrical.json`
(every row of every model): the worst absolute error over all columns, and the worst relative
error (algebraic: of the reference value; dynamic: of the column's range). For the dynamic
models the last column is the Richardson correction, the finer run's own distance from the
extrapolated result, as a fraction of the range: the error a single run at `2 r` would have.

| Model | Kind | Columns | Rows | `(r, 2r)` | Worst absolute error | Worst relative error | Correction |
|---|---|---|---|---|---|---|---|
| MSL `Analog.Examples.Resistor` (heat port, `R(T)`) | algebraic | 3 | 501 | | 1.6e-12 K (`resistor.T_heatPort`); 7.3e-13 A on the currents | 2.4e-5 (`SineVoltage1.i`, near a zero crossing) | |
| MSL `ShowVariableResistor` | algebraic | 7 | 501 | | 4.4e-16 A | 4.3e-7 (near a zero crossing) | |
| MBL `DC.Loads.Examples.Resistor` | algebraic | 15 | 101 | | 0 | 0 | |
| MBL `DC.Loads.Examples.VariableLoad` | algebraic | 18 | 101 | | 2.7e-15 A | 1.2e-13 | |
| MBL `AC.OnePhase.Loads.Examples.TestImpedance` | algebraic | 24 | 101 | | 0 | 0 | |
| MBL `AC.OnePhase.Lines.Examples.ACLine_RL` (dynamic branch omitted) | algebraic | 34 | 101 | | 7.1e-15 V | 1.2e-16 | |
| MBL `AC.OnePhase.Conversion.Examples.ACDCConverter` | algebraic | 17 | 101 | | 2.8e-14 A | 1.8e-14 | |
| MBL `AC.OnePhase.Conversion.Examples.ACACTransformer` | algebraic | 40 | 101 | | 4.5e-13 A | 1.2e-9 (`shortCircuit.terminal.v[1]`, a value near zero) | |
| MBL `AC.ThreePhasesBalanced.Lines.Examples.ACLine` | algebraic | 68 | 101 | | 6.2e-13 A | 2.1e-13 | |
| MSL `SeriesResonance` (0 to 200 Hz sweep, resonance at 100 Hz) | dynamic | 10 | 1001 | (32, 64) | 2.5e-6 V | 1.9e-7 (`capacitor2.v`, past the resonance) | 2.2e-4 |
| MSL `CauerLowPassAnalog` (step at 1 s) | dynamic | 15 | 601 | (8, 16) | 3.4e-8 A | 1.2e-7 (`C5.i`, just after the step) | 2.4e-5 |
| MSL `CompareTransformers` (10 Hz) | dynamic | 14 | 1004 | (16, 32) | 3.0e-6 A | 3.1e-7 (`resistor11.i`, first period) | 4.0e-5 |
| MSL `HeatingRectifier` (diode with heat port, heat capacitor) | dynamic | 7 | 501 | (16, 32) | 7.0e-7 A | 1.5e-7 (`Capacitor1.i`) | 8.1e-6 |
| MBL `DC.Lines.Examples.RCModel` | dynamic | 22 | 501 | (2, 4) | 4.3e-7 V | 1.7e-8 (`RC_dyn.terminal_n.i[1]`, first row after the start) | 6.9e-6 |

A relative error above 1e-9 on an algebraic model is at a value close to zero, where the
absolute part (1e-12) of the bound governs; the bound holds at every row. In `ACLine_RL` the
source `E` also feeds the omitted dynamic branch, so its two current columns are not
compared. Every other column of every CSV is.

**What runs by default.** The algebraic models run in full, except MSL's `Resistor`, whose
coupled thermal solve at each of its 501 rows takes about 48 s: by default every tenth row.
Each dynamic model compares its first rows (6 to 100, each window holding the model's worst
early row: Cauer's step, the first period of the transformers, the diode's first conduction;
`SeriesResonance`'s window ends at 7 ms, well before its resonance at 0.5 s, which only the
full run reaches); `pytest -m slow` compares every row, between about 1 and 42 minutes a model
(73 s for `RCModel` to 2,507 s for `SeriesResonance`; each step costs 10 to 30 ms, a coupled
one about 100 ms). The record is rewritten only by a run with
`NOODL_RECORD_PARITY=1`.

## Power flow against pandapower

The AC and DC power flows of the electrical application are compared with pandapower 3.5.6
results stored under `tests/data/power_flow/` (`tests/verification/test_power_flow_parity.py`).
The cases are MATPOWER 8.1's `case9`, `case14`, `case30` and `case118`, and `case9_features`, a
variant of `case9` with off-nominal taps, phase shifters, parallel and out-of-service branches,
a shunt conductance, two generators on one bus, a generator on a PQ bus, a PV bus left without
generation and a slack angle of 10°. MATPOWER's case files are not redistributed:
`matpower_case` downloads each on first use from MATPOWER's repository at release 8.1, checks it
against the SHA-256 recorded with its reference result and caches it
(`$NOODL_CACHE_DIR/matpower/8.1/`, by default `~/.cache/noodl/matpower/8.1/`); `case9_features`
is derived from the fetched `case9` and checked the same way. `scripts/power_flow_reference.py`
produced the results once, from the same files through pandapower's `from_ppc`, and
cross-checked every case against PYPOWER 5.1.21; the script is never run by the test suite, and
neither package is a dependency of noodl physics. pandapower's own `pandapower.networks.case9`
and `case118` are not MATPOWER's files: their generator setpoints differ and `case118` loses its
30° slack angle, which moves voltages by up to 0.04 p.u. and angles by up to 0.52 rad. They are
not used. The per-case agreement is on the [electrical
page](../../applications/electrical/index.md#power-flow-against-pandapower).
