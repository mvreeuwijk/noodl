The JSON and CSV files in this directory are derived from the Modelica Buildings Library
(MBL) v13.0.0, commit `55abf579598ca81cae0a82f337350375958e6722`, by OpenModelica 1.27.1
(`OpenModelica 1.27.1~2-g6db4671`) with `scripts/modelica_export.py`. They export the 8
`Buildings.Airflow.Multizone.Validation.*` and 15 `Buildings.Airflow.Multizone.Examples.*`
models: each `.json` is the model's structure and parameters (format `noodl-modelica/1`,
defined in `src/noodl/apps/building_physics/modelica/schema.py`) and each `.csv` is a
reference simulation trajectory (`# key: value` header lines, then `time` and one column per
compared variable). Every JSON records the MBL commit and the
OpenModelica version it was generated with; every CSV repeats them in its header.

The reference CSVs of the 13 dynamic models (`ClosedDoors`, `CO2TransportStep`,
`NaturalVentilation`, `OneEffectiveAirLeakageArea`, `OneOpenDoor`, `OneRoom`,
`OpenDoorBuoyancyDynamic`, `OpenDoorBuoyancyPressureDynamic`, `ReverseBuoyancy`,
`ReverseBuoyancy3Zones`, `ThreeRoomsContam`, `ThreeRoomsContamDiscretizedDoor`, `ZonalFlow`)
were regenerated with `scripts/modelica_export.py --tolerance 1e-13` (DASSL at a relative
tolerance of 1e-13 instead of each model's declared 1e-6 or 1e-8; the CSV header's
`tolerance` line records it; the JSON, which keeps the declared experiment, came out
byte-identical). At the declared tolerance the references' own integration error was
comparable to the parity tolerance (ZonalFlow: 5.6e-5 K, 2.8e-8 kg/kg; CO2TransportStep's
door flow 5e-5 relative after its source pulse, still visible at 1e-10). From 1e-12 to 1e-13
the references still moved by up to 1.1e-4 of the flow floor (`ClosedDoors`' crack flows,
1.2e-11 kg/s), 3.7e-5 (`NaturalVentilation`'s orifices at their reversal, 3.7e-9 kg/s),
4.7e-5 (`OneOpenDoor`'s door at 28.8 s, 7.9e-9 kg/s) and 1.2e-6 in `CO2TransportStep`'s
trace substance (6e-14 kg/kg); at 1e-14 DASSL returns no trajectory for these models.

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

Four of the 23 models are refused by `read_modelica` (`ModelicaImportError`, naming the
offending instances); their JSON is exported and kept here for the refusal tests:

- `PressurizationData.json` — wind-pressure boundaries (`Outside_CpLowRise`) and weather data
  (`ReaderTMY3`) are not supported.
- `TrickleVent.json` — the same wind-pressure/weather-data boundaries, plus a feedback
  controller (`LimPID`) and a temperature sensor wired as a component.
- `ChimneyShaftNoVolume.json` — a chain of more than one flow element between two nodes and a
  feedback controller are not supported.
- `ChimneyShaftWithVolume.json` — a dynamic (mass- and heat-storing) hydrostatic column
  (`MediumColumnDynamic`) and a feedback controller are not supported.

All four still simulate cleanly in OpenModelica, so their `.csv` was exported too (this
export script does not know which models `noodl`'s reader refuses); they are kept here for
completeness even though noodl's tests only read the JSON of these four.

Also derived from MBL, elsewhere in this repository: `tests/apps/building_physics/modelica/
fixtures/three_rooms_discretized_door.json` reuses the instance names of
`Buildings.Airflow.Multizone.Validation.ThreeRoomsContam` (`volWes`, `volEas`, `volOut`,
`colWesBot`, `colWesTop`, `oriWesTop`, `colOutBot`, `colOutTop`, `oriOutBot`, `oriOutTop`,
`col1EasBot`, `colEasInBot`, `colEasInTop`, `oriEasTop`) and of its `DiscretizedDoor` variant
(`dooOpeClo`), hand-written to a smaller, hand-checkable size for the reader's unit tests
rather than exported by `scripts/modelica_export.py`. The same is true of, and the same MBL
licence covers:

- `tests/apps/building_physics/modelica/fixtures/bad_door_wiring.json`
- `tests/apps/building_physics/modelica/fixtures/door_wiring.json`
- `tests/apps/building_physics/modelica/fixtures/refused_and_bad_door.json`
- `tests/apps/building_physics/modelica/fixtures/stack_chain.json`

each of which reuses a subset of `ThreeRoomsContam`'s instance names (`volWes`/`volEas`/
`dooOpeClo`, or the `colWesBot`/`oriWesTop`/`colWesTop`/`volTop` stack). No other fixture in
that directory reuses an MBL instance name or model name.
