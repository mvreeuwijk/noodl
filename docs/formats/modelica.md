# Modelica Buildings Library

Reading a Modelica export produces a model for [building
physics](../applications/building_physics.md). Power-law, tabulated and door components become
the [`noodl.elements`](../applications/building_physics.md#airflow-elements) classes of the
same physics — an open door, for example, becomes an `OpenDoor` edge pair and a discretised
door an `DoorPortStream` pair, its two port streams — while zonal flows are assembled by the
importer itself as prescribed-flow edge pairs, as set out below.

`read_modelica(path) -> (model, state, drivers)` imports a multizone airflow model built with the
Modelica Buildings Library (MBL) v13.0.0 (commit `55abf579598ca81cae0a82f337350375958e6722`)
through an OpenModelica JSON export. The reader never parses Modelica source and never evaluates
a Modelica expression: every number in the JSON was already evaluated by OpenModelica.

```python
from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import simulate

model, state, drivers, names = read_modelica("tests/data/modelica/OneRoom.json", return_names=True)
result = simulate(model, state, drivers, names.times)

print(result["T"][-1])       # zone temperatures (K) at the last grid time, node order names.nodes
print(result["air.q"][-1])   # edge flows (kg/s); names.edges maps an MBL instance to its columns
```

**Supported (20 models: 8 `Validation` + 11 `Examples` of `Buildings.Airflow.Multizone`, and
`Buildings.Fluid.Sources.Examples.Outside_Cp`).** `Buildings.Airflow.Multizone`
elements, `MixingVolume` zones, pressure/temperature boundaries, trace substances, the outside
boundaries `Outside`, `Outside_Cp`, `Outside_CpLowRise` and `Outside_CpData` and the weather
reader `ReaderTMY3` (see "Weather and wind" below) — not `Buildings.ThermalZones`, HVAC or
district networks. `Validation`: `OneWayFlow`,
`DoorOpenClosed`, `OpenDoorPressure`, `OpenDoorTemperature`, `ThreeRoomsContam`,
`ThreeRoomsContamDiscretizedDoor`, `OpenDoorBuoyancyDynamic`, `OpenDoorBuoyancyPressureDynamic`.
`Examples`: `CO2TransportStep`, `ClosedDoors`, `NaturalVentilation`, `OneEffectiveAirLeakageArea`,
`OneOpenDoor`, `OneRoom`, `Orifice`, `PowerLaw`, `ReverseBuoyancy`, `ReverseBuoyancy3Zones`,
`ZonalFlow`. The weather and wind pressure of the `Buildings.Fluid.Sources` examples
`Outside_CpLowRise`, `Outside_CpData`, `Outside_CpData_Angles` and
`Validation.Outside_CpData_Directions` (boundaries with no flow element, so not importable as
models) are compared with `assemble.weather_columns`.
`MediumColumnDynamic` is read as MBL builds it: a mixing volume between two half columns, the
top one taking its density from below and the bottom one from above, both the volume's
(`use_HeatTransfer = false` only). Its volume is named `<column>.vol` in `names.nodes`, as in
MBL's results. A `TemperatureSensor` on a volume's `heatPort` reads that volume's
temperature; its `T` may feed a signal block's input.

**Imported, not yet compared.** `PressurizationData` imports (`read_modelica` builds it, and its weather and
wind pressures match MBL's to round-off), but it is not yet compared as a dynamic model. It is a
62.5 m3 room between two `Outside_CpLowRise` walls over 30 days of San Francisco weather. The room
flushes in about 10 s, against steps of 50 to 3000 s. Where the flow fades, in a calm or with the
wind parallel to both walls, it can fall by orders of magnitude within one step, and the room
freezes at the state it had when its flushing stopped. The run does not resolve that decay: the
transport step sees the flows only at the step's two ends. It therefore freezes the room at the
wrong state, and this error, first order in the step, stays for the whole calm. Over every row at 1
and 2 substeps, the room temperature is 2.9e-3 relative from the reference (0.82 K, at the end of
the first calm, 25920 s) and its humidity 3.4e-2 (at 93312 s). Flows are within 1e-6 beyond the
reference's resolution and the pressure within 2e-11. Refining the grid where the boundary pressures
fade reduced the temperature error to 1.6e-4 and the humidity error to 2.0e-3 in an experiment. That
refinement, together with a transport step that treats the flows' change within a step implicitly,
is the direction for the follow-up. The wind boundaries are verified on the
`Buildings.Fluid.Sources` models and `Outside_Cp`.

**Refused, each with a named error** (`ModelicaImportError` lists every offending instance and
its class):

- `TrickleVent`, `ChimneyShaftNoVolume`, `ChimneyShaftWithVolume` — feedback controllers
  (`LimPID`), and in `ChimneyShaftNoVolume` a chain of two flow elements.
- A `MediumColumnDynamic` with `use_HeatTransfer = true` (heat exchanged through its
  `heatPort`).
- A component output other than a temperature sensor's `T` wired into a signal block, naming
  the connection; a temperature sensor whose `port` is not wired to a volume's `heatPort`.
- Every `Buildings.BoundaryConditions.WeatherData` class but `ReaderTMY3`; a weather bus wired
  to anything but an `Outside` boundary's `weaBus`, or any weather-bus output other than
  `TDryBul`, `pAtm`, `relHum`, `winSpe` and `winDir` read by anything; a `ReaderTMY3` wired
  through any port but its `weaBus` and the `<output>_in` inputs of those five; a signal
  driving one of its inputs that it does not read (an output taken from the file, or an
  input of another output).

For example, reading `TrickleVent` raises:

```text
ModelicaImportError: modelica: refused 1 item:
  - con (Modelica.Blocks.Continuous.LimPID): feedback controllers are not supported
```

**Conventions:**

- `DoorOpen`/`DoorOperable` use MBL's fixed default density (`Door.mo`); a discretised door
  (`DoorDiscretizedOpen`/`Operable`) evaluates density at the actual port pressure
  (`TwoWayFlowElement.mo`) instead — the two door families do not share one convention.
- A door becomes two directional noodl physics edges between the same pair of zones; so does a
  discretised door, whose two edges are its port streams `mAB_flow` and `-mBA_flow`
  (`DoorPortStream`), the compartments and their hydrostatic heads evaluated inside the
  element.
- Zonal flows are four-port, like doors (not the two-port shape a one-way element has), and
  become two directional edges the same way.
- An in-line flow sensor (`Buildings.Fluid.Sensors`, flow-through) is a transparent wire: it adds
  no node and no pressure drop.
- `PrescribedHeatFlow` is supported only at `alpha = 0` (MSL's default: no temperature
  dependence); a nonzero `alpha` is refused.
- A boundary wired straight to one zone's port, and to no other port, is supported: it fixes that
  zone's pressure.
- `"air.phi"` is GAUGE pressure relative to a per-model reference `p_ref` (the first boundary's
  pressure, or an attached boundary's, at the first grid time; the first zone's `p_start` with
  neither) — flows depend only on pressure differences, so the choice changes no result.
- Every source driver (air, heat and species) is the MEAN of the source over each step, not its
  end-of-step value, so that a pulse shorter than the output interval still injects its exact
  mass (e.g. `CO2TransportStep`'s 3.6 s pulse landing between two 172.8 s outputs).
- A signal may drive several inputs (`drives` accepts one name or a list) — MBL's `ZonalFlow`
  example drives two flows from one `Constant`.
- Refused, also with a named error: with `mass_storage=False` (or volumes whose `massDynamics`
  is `SteadyState`), a closed group of zones (joined only by pressure-dependent
  edges or zonal flows, no boundary among them) with a net flow imbalance — an unequal
  `ZonalFlow_m_flow` pair or a mass source into it; `MediumColumn.densitySelection = "actual"`;
  and an outside boundary whose weather-bus inputs come from no `ReaderTMY3` and no signal. With
  storage, none of the 20 supported models needs any of the three.

**Weather and wind.** A `ReaderTMY3` supplies the outside boundaries wired to it with the dry-bulb
temperature, the atmospheric pressure, the relative humidity, the wind speed and the wind
direction, as MBL computes them: the TMY3 table interpolated by MSL's Akima spline on the
calendar time (which repeats a one-year file), converted to K, Pa, 1, m/s and rad, and limited
(humidity to 0..1, wind speed to at least 0, direction modulo 2 pi). Each output may come from
the file, a parameter or an input signal (`<output>Sou`); `pAtm` defaults to a 101325 Pa
parameter, as in MBL. Solar and sky outputs are not read. The `.mos` file travels with the
JSON (see "Reproducing the export"); a model whose file is missing or altered is refused.

An outside boundary is at the atmospheric pressure, the dry-bulb temperature and the water
mass fraction of the relative humidity (`X = phi k / (k phi + p/pSat - phi)`). The wind
boundaries add `pWin = Cs 0.5 Cp d v^2`:

- `Outside_Cp`: a fixed `Cp` (or the `Cp_in` signal), `d` at the atmospheric pressure and the
  default composition;
- `Outside_CpLowRise`: Swami and Chandra's low-rise correlation, `Cp = Cp0 ln(1.248 - 0.703
  sin(a/2) - 1.175 sin^2 a + 0.131 sin^3(2 a G) + 0.769 cos(a/2) + 0.071 G^2 sin^2(a/2) + 0.717
  cos^2(a/2))`, `G = ln s`, blended to `Cp0` below 5 degrees and to its 180-degree value above
  175;
- `Outside_CpData`: a periodic cubic Hermite spline through the `(incAngSurNor, Cp)` table,
  scaled by `Cs`.

The last two take `d` at the boundary's own pressure, an implicit equation solved by
iteration. Angles follow MBL: the wind direction is where the wind comes from (north 0, east
pi/2), a surface azimuth where the surface faces (south 0, west pi/2), and the incidence is
`winDir - (azi - pi)`. These differ from CONTAM's, whose wind pressure acts on the envelope
path rather than on the boundary node.

The boundaries' values are point values at each driver-grid time, like any boundary signal.
The table's knots and the points where a limiter starts cutting are added to the driver grid as
step boundaries (the outputs are once continuously differentiable there); a repeating file's
year wrap is an event. `names.weather` (from `read_modelica(..., return_names=True)`) holds the
reader's outputs, keyed `"<reader>.weaBus.<output>"`, and every wind boundary's `"<boundary>.pWin"`
on the output grid; `assemble.weather_columns(doc, times)` evaluates the same for a model with
no flow element, which `read_modelica` refuses.

```python
from noodl.apps.building_physics import read_modelica

model, state, drivers, names = read_modelica("tests/data/modelica/PressurizationData.json",
                                             return_names=True)
print(names.times[2].item(), names.weather["weaDat.weaBus.winSpe"][2].item())
print(names.weather["west.pWin"][2].item(), names.weather["east.pWin"][2].item())
```

```text
10368.0 2.133143665116279
0.3241156179122303 -1.2724675933675318
```

**Volume mass storage.** Like MBL, the reader stores each volume's air compressibly unless its
`massDynamics` is `SteadyState`: zone pressures are states, the t = StartTime row holds MBL's
`p_start` initialisation, a closed zone group needs no pressure reference, and a mass source
may feed one (see [the Modelica parity
tables](../applications/building_physics.md#against-openmodelica-modelica-buildings-library)).
`read_modelica(path, mass_storage=False)` gives the quasi-steady airflow of the CONTAM route
instead, with its closed-group rules below.

**Reproducing the export (WSL only — the test suite itself needs none of this).** Tested on
Ubuntu 22.04 (`jammy`) in WSL with OpenModelica 1.27.1. Install OpenModelica from its own apt
repository (needs `sudo`, done once by whoever administers the WSL environment):

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL http://build.openmodelica.org/apt/openmodelica.asc | sudo gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] https://build.openmodelica.org/apt jammy stable" | sudo tee /etc/apt/sources.list.d/openmodelica.list
sudo apt-get update && sudo apt-get install -y omc
```

MBL v13 needs the Modelica Standard Library (MSL) 4.1.0, installed once, without `sudo`, with
`omc`'s own `installPackage` (it writes into `~/.openmodelica/libraries`, not a system path):

```bash
echo 'installPackage(Modelica, "4.1.0");' > /tmp/install_msl.mos && omc /tmp/install_msl.mos
```

Clone MBL at the tag this release was exported against:

```bash
git clone --depth 1 --branch v13.0.0 https://github.com/lbl-srg/modelica-buildings.git ~/modelica/modelica-buildings
```

(`git rev-parse HEAD` there is `55abf579598ca81cae0a82f337350375958e6722`, the commit recorded
in every fixture's JSON.) Then, with `omc` on the WSL `PATH` (this release was exported against
`OpenModelica 1.27.1~2-g6db4671`):

```bash
python3 scripts/modelica_export.py Buildings.Airflow.Multizone.Validation.ThreeRoomsContam --out tests/data/modelica
```

(`<Model>` is the model's fully qualified name; the exporter writes `<Short>.json`/`<Short>.csv`
under `<Short>`, its last component — `ThreeRoomsContam.json`/`ThreeRoomsContam.csv` here.) This
writes `tests/data/modelica/<Short>.json` (the component graph) and `<Short>.csv` (OpenModelica's
own simulated reference) as committed fixtures. `scripts/modelica_export.py` is not imported by
the package and is not run by the test suite. It exits non-zero if the simulation fails (the
JSON is still written, with the instance API's Reals instead of the simulated values, so the
export can be inspected); a batch script over every model should check the exit code rather
than assume success. The dynamic parity tests compare a window of rows of every dynamic model
in the repository's default `pytest` run; every row of every model runs under
`@pytest.mark.slow` (`pytest -m slow`). `--tolerance` simulates at a solver tolerance other than the model's
declared one; the thirteen dynamic parity references (and `PressurizationData`'s) were regenerated with
`--tolerance 1e-13` (`tests/data/modelica/NOTICE.md`). At 1e-14 OpenModelica's DASSL returns
no trajectory for these models, so 1e-13 is the finest reference available.

A model with a `ReaderTMY3` also gets its weather file: the exporter copies the `.mos` file
`filNam` names (from the MBL checkout) next to the JSON and records its name and sha256 in the
reader's `"files"` entry. The reader opens that copy and refuses the model, naming the reader,
if the file is missing or its sha256 differs. The CSV of such a model also holds the reader's
`weaBus.TDryBul`, `weaBus.pAtm`, `weaBus.relHum` and `weaBus.winSpe` and every wind boundary's
`pWin`, which the parity tests compare at round-off.

The exporter keeps a connection from a component's output into a signal block's input (a
temperature sensor's `T` feeding a controller's `u_m`), so that the reader can wire the sensor
into the block or refuse the connection by name. The CSV of a model with a feedback controller
holds its output `<controller>.y`, and that of a model with a `MediumColumnDynamic` the state
of its volume, `<column>.vol.T`, `.p`, `.Xi[1]` and `.C[k]`. `ChimneyShaftNoVolume`,
`ChimneyShaftWithVolume` and `TrickleVent` were re-exported this way at `--tolerance 1e-13`.

Regenerating the committed parity records (`tests/data/modelica/parity-{algebraic,dynamic}.json`,
in [the building physics parity tables](../applications/building_physics.md#against-openmodelica-modelica-buildings-library))
needs no OpenModelica — they are written by `tests/verification/test_modelica_parity.py`
itself, only when the environment variable `NOODL_RECORD_PARITY=1` is set:

```bash
NOODL_RECORD_PARITY=1 pytest tests/verification/test_modelica_parity.py -m "not slow"
NOODL_RECORD_PARITY=1 pytest tests/verification/test_modelica_parity.py -m slow
```

(two runs, since the default `addopts` excludes `slow`-marked tests and a command-line `-m`
replaces rather than adds to it). Without the variable, the suite reads and checks the
fixtures but never rewrites the records.
