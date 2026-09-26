# Modelica Buildings Library

Reading a Modelica export produces a model for [building
physics](../applications/building_physics.md). Each imported instance becomes one of the
[`noodl.elements`](../applications/building_physics.md#airflow-elements) objects the building
application assembles into a network by hand — a door becomes an orifice pair, a zonal-flow
component becomes a directed edge pair, and so on, as set out below.

`read_modelica(path) -> (model, state, drivers)` imports a multizone airflow model built with the
Modelica Buildings Library (MBL) v13.0.0 (commit `55abf579598ca81cae0a82f337350375958e6722`)
through an OpenModelica JSON export — a second reference implementation for this application,
independent of CONTAM. The reader never parses Modelica source and never evaluates a Modelica
expression: every number in the JSON was already evaluated by OpenModelica.

```python
from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import simulate

model, state, drivers, names = read_modelica("tests/data/modelica/OneRoom.json", return_names=True)
result = simulate(model, state, drivers, names.times)

print(result["T"][-1])       # zone temperatures (K) at the last grid time, node order names.nodes
print(result["air.q"][-1])   # edge flows (kg/s); names.edges maps an MBL instance to its columns
```

**Supported (18 models: 8 `Validation` + 10 `Examples`).** `Buildings.Airflow.Multizone`
elements, `MixingVolume` zones, pressure/temperature boundaries and trace substances — not
`Buildings.ThermalZones`, HVAC, wind, weather or district networks. `Validation`: `OneWayFlow`,
`DoorOpenClosed`, `OpenDoorPressure`, `OpenDoorTemperature`, `ThreeRoomsContam`,
`ThreeRoomsContamDiscretizedDoor`, `OpenDoorBuoyancyDynamic`, `OpenDoorBuoyancyPressureDynamic`.
`Examples`: `CO2TransportStep`, `ClosedDoors`, `NaturalVentilation`, `OneOpenDoor`, `OneRoom`,
`Orifice`, `PowerLaw`, `ReverseBuoyancy`, `ReverseBuoyancy3Zones`, `ZonalFlow`.

**Refused, each with a named error** (`ModelicaImportError` lists every offending instance and
its class):

- `PressurizationData`, `TrickleVent`, `ChimneyShaftNoVolume`, `ChimneyShaftWithVolume` — wind
  pressure, weather data, feedback controllers, or a dynamic (mass- and heat-storing) hydrostatic
  medium column, none of which is in scope.
- `OneEffectiveAirLeakageArea` — a mass source feeding two boundary-less volumes; the injected
  air can only go into compressing them, which needs the compressible volume storage this
  release does not model.

For example, reading `PressurizationData` raises:

```
ModelicaImportError: modelica: refused 3 items:
  - east (Buildings.Fluid.Sources.Outside_CpLowRise): wind pressure is not supported
  - weaDat (Buildings.BoundaryConditions.WeatherData.ReaderTMY3): weather data is not supported
  - west (Buildings.Fluid.Sources.Outside_CpLowRise): wind pressure is not supported
```

**Conventions:**

- `DoorOpen`/`DoorOperable` use MBL's fixed default density (`Door.mo`); a discretised door
  (`DoorDiscretizedOpen`/`Operable`) evaluates density at the actual port pressure
  (`TwoWayFlowElement.mo`) instead — the two door families do not share one convention.
- A door becomes two directional noodl edges between the same pair of zones; a discretised door
  becomes one edge per compartment, each with its own hydrostatic head.
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
  mass (found from `CO2TransportStep`'s 3.6 s pulse landing between two 172.8 s outputs).
- A signal may drive several inputs (`drives` accepts one name or a list) — MBL's `ZonalFlow`
  example drives two flows from one `Constant`.
- Refused, also with a named error: a closed group of zones (joined only by pressure-dependent
  edges or zonal flows, no boundary among them) with a net flow imbalance — an unequal
  `ZonalFlow_m_flow` pair or a mass source into it; `MediumColumn.densitySelection = "actual"`;
  and `Outside` without a weather-bus signal driving it. None of the 18 supported models needs
  any of the three.

**Quasi-steady airflow.** Like the CONTAM route, a volume's air mass is not stored: the airflow
is quasi-steady at every step. MBL's volumes do store mass, so a model whose dynamics are
dominated by that storage — a closed, heated room expanding through its leakage, or an initial
pressure imbalance draining away — parts company with noodl by more than round-off (see [the
Modelica parity tables](../applications/building_physics.md#against-openmodelica-modelica-buildings-library)).
Adding volume mass storage would close this gap; it is a possible extension, not implemented in
this release.

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
than assume success. 9 of the 12 dynamic parity tests take 30 s–5 min each and are marked
`@pytest.mark.slow`, excluded by the repository's default `pytest` run; `pytest -m slow` runs
them.

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
