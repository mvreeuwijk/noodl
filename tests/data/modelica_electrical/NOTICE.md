The JSON and CSV files in this directory are derived from the Modelica Standard Library (MSL)
4.1.0 (the `Modelica 4.1.0+maint.om` package OpenModelica installs with
`installPackage(Modelica, "4.1.0")`, loaded with `loadModel(Modelica, {"4.1.0"})`) and the
Modelica Buildings Library (MBL) v13.0.0, commit `55abf579598ca81cae0a82f337350375958e6722`,
by OpenModelica 1.27.1 (`OpenModelica 1.27.1~2-g6db4671`) with
`scripts/modelica_export_electrical.py`. Both libraries are loaded for every model. Each
`.json` is the model's structure and parameters (format `noodl-modelica-electrical/1`) and each
`.csv` is a reference simulation trajectory (`# key: value` header lines, then `time` and one
column per compared variable). Every JSON records the MBL commit, the OpenModelica and MSL
versions and the experiment as simulated; every CSV repeats them in its header.

Every reference was simulated with DASSL at a relative tolerance of 1e-12 (the script's
default `--tolerance`; the models declare 1e-6 or nothing), `-noEventEmit -emit_protected`,
from `StartTime = 0`. Real parameters are re-read from the simulation result in full double
precision (`"parameter_precision"`); each state variable's value at t = 0 is recorded in its
component's `"initial"` (capacitor `v`, inductor `i`, transformer `i1`/`i2`, heat capacitor
`T`, MBL `TwoPortRCLine` `Vc` when `use_C`), and each MBL `Line`'s simulated inner
`line.R`, `line.L`, `line.C` in its `"derived"`.

The 14 models and the export commands (run in WSL from the repository root, one omc at a
time; the output grid is the model's own experiment unless `--intervals` is given):

```bash
for m in \
 'Modelica.Electrical.Analog.Examples.Resistor' \
 'Modelica.Electrical.Analog.Examples.ShowVariableResistor' \
 'Modelica.Electrical.Analog.Examples.HeatingRectifier' \
 'Buildings.Electrical.DC.Lines.Examples.RCModel'; do python3 scripts/modelica_export_electrical.py $m; done
python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.SeriesResonance --intervals 1000
python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.CauerLowPassAnalog --intervals 600
python3 scripts/modelica_export_electrical.py Modelica.Electrical.Analog.Examples.CompareTransformers --intervals 1003
for m in \
 'Buildings.Electrical.DC.Loads.Examples.Resistor' 'Buildings.Electrical.DC.Loads.Examples.VariableLoad' \
 'Buildings.Electrical.AC.OnePhase.Loads.Examples.TestImpedance' \
 'Buildings.Electrical.AC.OnePhase.Lines.Examples.ACLine_RL' \
 'Buildings.Electrical.AC.OnePhase.Conversion.Examples.ACDCConverter' \
 'Buildings.Electrical.AC.OnePhase.Conversion.Examples.ACACTransformer' \
 'Buildings.Electrical.AC.ThreePhasesBalanced.Lines.Examples.ACLine'; do \
 python3 scripts/modelica_export_electrical.py $m --intervals 100; done
```

| file stem | stop time (s) | intervals | output interval (s) |
|---|---|---|---|
| `Analog.Examples.Resistor` | 5 | 500 | 0.01 |
| `Analog.Examples.ShowVariableResistor` | 1 | 500 | 0.002 |
| `Analog.Examples.HeatingRectifier` | 5 | 500 | 0.01 |
| `DC.Lines.Examples.RCModel` | 15 | 500 | 0.03 |
| `Analog.Examples.SeriesResonance` | 1 | 1000 | 0.001 |
| `Analog.Examples.CauerLowPassAnalog` | 60 | 600 | 0.1 |
| `Analog.Examples.CompareTransformers` | 50 | 1003 | 0.04985 (50/1003) |
| `DC.Loads.Examples.Resistor` | 1 | 100 | 0.01 |
| `DC.Loads.Examples.VariableLoad` | 1 | 100 | 0.01 |
| `AC.OnePhase.Loads.Examples.TestImpedance` | 1 | 100 | 0.01 |
| `AC.OnePhase.Lines.Examples.ACLine_RL` | 1 | 100 | 0.01 |
| `AC.OnePhase.Conversion.Examples.ACDCConverter` | 1 | 100 | 0.01 |
| `AC.OnePhase.Conversion.Examples.ACACTransformer` | 1 | 100 | 0.01 |
| `AC.ThreePhasesBalanced.Lines.Examples.ACLine` | 1 | 100 | 0.01 |

`CompareTransformers` uses 1003 intervals rather than a round number: its sources run at
10 Hz, and an output interval commensurate with the 0.1 s period (0.05 s) would sample only
two phases of each period; 50/1003 s walks the samples through the whole period at the same
CSV size.

Compared columns, per component in document order, skipping observers, ideal current sensors
(MSL `CurrentSensor`, MBL `GeneralizedSensor`) and grounds: for `Modelica.Electrical.Analog.*`
classes the current `i` (or `i1`, `i2` for two-ports), plus `v` for a `Capacitor` and
`T_heatPort` when `useHeatPort` is true; a `HeatCapacitor`'s `T`; for `Buildings.Electrical.*`
classes, per connector `terminal`, `terminal_n`, `terminal_p`, the conductor potentials
`v[1]`, `v[2]` and the current `i[1]` (DC terminal) or the phasor parts `v[1]`, `v[2]`,
`i[1]`, `i[2]` (AC terminal), plus `Vc` of a `TwoPortRCLine` with `use_C`. Thermal
conductors and fixed temperatures have no column.

MBL is licensed under a 3-clause BSD licence (with an added paragraph on accepting
enhancements), reproduced below from `Buildings/legal.html` in the MBL source tree:

---

Modelica Buildings Library. Copyright (c) 1998-2026
Modelica Association,
International Building Performance Simulation Association (IBPSA),
The Regents of the University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy) and
contributors.
All rights reserved.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and
the U.S. Government consequently retains certain rights.
As such, the U.S. Government has been granted for itself and others acting on its behalf
a paid-up, nonexclusive, irrevocable, worldwide license in the Software
to reproduce, distribute copies to the public, prepare derivative works, and
perform publicly and display publicly, and to permit other to do so.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the distribution.
3. Neither the names of the Modelica Association,
   International Building Performance Simulation Association (IBPSA),
   the University of California,
   Lawrence Berkeley National Laboratory,
   U.S. Dept. of Energy,
   nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.

---

MSL is licensed under the 3-clause BSD licence, reproduced below from `LICENSE` in the MSL
source tree (tag `v4.1.0`, commit `8ae3d35c24e519cb2996cab20f3b13daf2b0c50a`):

---

BSD 3-Clause License

Copyright (c) 1998-2025, Modelica Association and contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---
