The JSON and CSV files in this directory are derived from the Modelica Buildings Library
(MBL) v13.0.0, commit `55abf579598ca81cae0a82f337350375958e6722`, by OpenModelica 1.27.1
(`OpenModelica 1.27.1~2-g6db4671`) with `scripts/modelica_export.py`. They export the 8
`Buildings.Airflow.Multizone.Validation.*` and 15 `Buildings.Airflow.Multizone.Examples.*`
models: each `.json` is the model's structure and parameters (`noodl-modelica/1`, spec
section 4) and each `.csv` is a reference simulation trajectory (`# key: value` header lines,
then `time` and one column per compared variable). Every JSON records the MBL commit and the
OpenModelica version it was generated with; every CSV repeats them in its header.

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

Five of the 23 models are refused by `read_modelica` (`ModelicaImportError`, naming the
offending instances); their JSON is exported and kept here for the refusal tests:

- `PressurizationData.json` — wind-pressure boundaries (`Outside_CpLowRise`) and weather data
  (`ReaderTMY3`) are not supported.
- `TrickleVent.json` — the same wind-pressure/weather-data boundaries, plus a feedback
  controller (`LimPID`) and a temperature sensor wired as a component.
- `ChimneyShaftNoVolume.json` — a chain of more than one flow element between two nodes and a
  feedback controller are not supported.
- `ChimneyShaftWithVolume.json` — a dynamic (mass- and heat-storing) hydrostatic column
  (`MediumColumnDynamic`) and a feedback controller are not supported.
- `OneEffectiveAirLeakageArea.json` — a mass flow source into a zone group with no boundary
  node (the injected mass could only be stored by compressing a volume, which is not
  modelled) is not supported.

All five still simulate cleanly in OpenModelica, so their `.csv` was exported too (this
export script does not know which models `noodl`'s reader refuses); they are kept here for
completeness even though noodl's tests only read the JSON of these five.

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
