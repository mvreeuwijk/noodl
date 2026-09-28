# Reading and writing street-network cases

A street-network case bundles a network with the meteorology, emissions and background
concentrations that drive it hour by hour. `read_case(path) -> StreetCase` reads one from
disk: a directory holding a `munich.cfg` is read as a MUNICH case; a `.dat` file is read as
a SIRANE master file (`Donnees_*.dat`), with the deck it names; anything else raises
`ValueError` naming both expectations.

```python
from noodl.apps.street_aq import read_case

case = read_case("tests/data/street/munich_paris_excerpt")
case.street_ids           # ['1', '3', '8', '11'], the emissions/background street axis
case.species               # ['NO2']
case.meteo["u_star"].shape  # (3, 4): 3 hours, 4 streets
```

A SIRANE master file names a whole deck (the network Shapefile, the site, meteo, emission,
background and species files) relative to its own directory:

```python
from noodl.apps.street_aq import read_case

case = read_case("tests/data/street/sirane_south_kensington/Donnees_SouthKensington.dat")
case.source                # 'sirane'
len(case.street_ids)        # 46
case.species                # ['NO2', 'NO', 'O3']
case.native["one_sided"]    # ['3', '14', '42']: streets with buildings on one side only
```

**The SIRANE network mapping.** Street `i` is the network Shapefile's record `i`, named
`str(i)` -- the record order, not the DBF's own `ID` column, is the street's id in every
SIRANE output. Each street's width is `WG + WD` (the half-widths either side of the
centreline) and its height `mean(HG, HD)`; a street with `HG = 0` or `HD = 0` (buildings on
one side only) is listed in `native["one_sided"]`, and its height is still that mean, i.e.
half the one side's. `z0_b` is `Z0D_BAT`, SIRANE's building-surface roughness. A street's
two ends are the DBF's `NDDEB` -> `NDFIN` node ids, which are also the junction names and
`case.junction_ids`.

`native["physics"]` carries the master-file switches that change SIRANE's own results --
`background_on`, `plume_on` (the street-plume model above the roofs), `retro_on` (the
retrotrajectory model), `dispersion_model`, `chemistry_on` -- recorded rather than acted on:
noodl physics has no above-roof dispersion model. `native["options"]` holds every master-file
setting by its SIRANE keyword, `native["site_disp"]`/`native["site_meteo"]` the two site
files, `native["streets"]` each record's SIRANE-consumed DBF fields, and
`native["meteo_raw"]` the precipitation and cloud-cover columns `read_case` itself does not
use.

`read_case` refuses, by name, a SIRANE deck that exercises anything the South Kensington
deck does not: a network file that is not a PolyLine Shapefile, meteorology that is not one
measured station, user-supplied street velocities (`B_STREET_U_SIGMA_W = 1`), NO emitted as
NO2-equivalent, a per-hour meteo override file, a non-empty source-groups file, a non-empty
surface-emission file, and a point source with non-zero emission (noodl physics' street
model has no point source) -- see `read_case`'s docstring for the complete list.

`wind_dir_from_deg` is degrees clockwise from north, the direction the wind blows FROM, in
both formats. MUNICH's own `WindDirection` is radians clockwise from north, the direction
the wind blows TOWARD (MUNICH's `preprocessing/meteo.py`, `compute_wdir`); `read_case` and
`write_case` convert at the file boundary, and `drivers_at` converts degrees-FROM into noodl
physics' `theta_w` (radians counter-clockwise from east, TOWARD). SIRANE's own meteo file is
already in the FROM convention, so no direction conversion happens at that boundary -- only
the unit conversions (g/s, micrograms/m3) do. Every mass in `StreetCase` is SI (kg/s, kg/m3).

## `StreetCase`

| Field | Holds |
|---|---|
| `source` | Which reader produced the case: `"munich"`, `"sirane"`, or `"synthetic"` for one built with `StreetCase.synthetic`. |
| `network` | The case's `StreetNetwork`, in metres. |
| `times` | `(n_hours,)`, seconds since `start`. |
| `street_ids` | The streets' names, in `network.streets` order -- the axis `emissions` and `background` use. |
| `junction_ids` | The source model's own node ids for `network.junctions`, in that order (MUNICH: `intersection.dat`'s ids; SIRANE: the network Shapefile's `NDDEB`/`NDFIN` node ids). |
| `species` | The case's species names, the last axis of `emissions` and `background`. |
| `meteo` | One `(n_hours, n_streets)` array per key: `wind_dir_from_deg` and `wind_speed` always; `h_abl`, `u_star`, `lmo`, `temperature` and `sigma_theta` (the direction spread, radians) wherever the source provides them (a case read from SIRANE has neither `h_abl`, `u_star` nor `lmo` -- SIRANE derives them itself; see [`StreetCase.model_options()`](#streetcasemodel_options)). |
| `meteo_junction` | The same keys, `(n_hours, n_junctions)`, in `network.junctions` order -- may be empty or partial when the source has no genuine per-junction meteorology (always empty for SIRANE: one meteorological station for the whole network). |
| `emissions` | `(n_hours, n_streets, n_species)`, kg/s per street. |
| `background` | `(n_hours, n_streets, n_species)`, kg/m3 per street. |
| `native` | The source's own options, as read (MUNICH: one dict per `munich.cfg` section, plus the lon/lat projection the reader used; SIRANE: the master file's options by SIRANE keyword, the two site files, each street's network fields, the one-sided streets and the raw meteo columns; the refusals are listed above). `model_options` translates these into `build_model` keywords, and `write_case` writes them back when the format matches. |
| `start` | The absolute date and time of `times[0]`, or `None` for a synthetic case built without one (writing such a case then raises). |

## `read_results` / `StreetResults`

`read_results(path, *, case=None, hours="case") -> StreetResults` reads one street model's
own results, format-neutral: a SIRANE result directory (holding `RUES_PAR_HEURE/` and
`METEO/Resul_Meteo.dat`) or a MUNICH `results/` directory of `<species>.bin` files -- a
MUNICH directory always needs `case` (its binaries carry no street order, species or
absolute time of their own); a path with neither `RUES_PAR_HEURE/` nor a `case` is refused,
naming both expectations.

```python
from noodl.apps.street_aq import read_case, read_results

case = read_case("tests/data/street/sirane_south_kensington/Donnees_SouthKensington.dat")
results = read_results(
    "tests/data/street/sirane_south_kensington/RESULT_SOUTHKENSINGTON", case=case,
)
results.source              # 'sirane'
sorted(results.meteo)        # ['h_abl', 'lmo', 'sigma_theta', 'temperature', 'u_star', ...]
results.u_canyon.shape       # (2, 46): SIRANE's own U_moy, m/s
```

| Field | Holds |
|---|---|
| `source` | `"sirane"` or `"munich"`. |
| `times` | The absolute time of each output hour. |
| `street_ids` | What every per-street field's street axis indexes (SIRANE: its own record-order ids; MUNICH: `case.street_ids`). |
| `species` | `c_in`/`c_above`'s dict keys, in file order. |
| `c_in` | One `(n_hours, n_streets)` array per species, kg/m3: the in-canyon concentration (SIRANE `Cint`; MUNICH's own street concentration). |
| `c_above` | The same, kg/m3, for the concentration just above the canyon (SIRANE `Cext`) -- `{}` for MUNICH, which has no such output. |
| `u_canyon` | `(n_hours, n_streets)`, m/s, the mean in-canyon velocity (SIRANE `U_moy`) -- `None` for MUNICH. |
| `sigma_w_roof` | `(n_hours, n_streets)`, m/s, the vertical-velocity fluctuation at roof height (SIRANE `Sigma_wH`) -- `None` for MUNICH. |
| `u_exchange` | `(n_hours, n_streets)`, m/s, the roof-level exchange velocity, read AS PRINTED (SIRANE `u_d`), not recomputed -- `None` for MUNICH. |
| `meteo` | `u_star`, `sigma_theta`, `h_abl`, `lmo`, `wind_speed`, `wind_dir_from_deg`, `temperature` (K), each `(n_hours, n_streets)`, network-wide (SIRANE's `Resul_Meteo.dat`, one meteorological station for the whole case) -- `{}` for MUNICH, whose own meteorology is the case's `meteo`, not a result. |

`hours="case"` (the default) keeps only the hours inside `case`'s own period; a SIRANE
result directory can otherwise mix hours from more than one run, so this is the safer
default -- a case hour missing from the directory raises `ValueError` naming it. `hours="all"`
returns every hour the directory holds, in time order, with or without a `case`.

A case read from SIRANE carries none of `h_abl`, `u_star` or `lmo` in its own `meteo` (see
[`StreetCase.model_options()`](#streetcasemodel_options)): a SIRANE case is driven with these
results' own meteorology instead, through the `u_star` driver directly -- never through
noodl physics' log law from the measured wind speed.

## `drivers_at`

`drivers_at(case, model, k, *, species=None) -> dict` is the driver mapping at time index `k`
for `model`, built from `case`. It follows the model's own shape: `meteo="uniform"` reduces
every meteorology array to one network-wide value (circular mean for direction, through the
reciprocal for the Obukhov length, a plain mean otherwise); `meteo="per_street"` keeps every
driver's trailing street axis and adds the `"<key>_junction"` drivers junction routing needs,
from `case.meteo_junction` when the source has it, otherwise the same street-to-junction
reduction. `u_star` is supplied whenever `case.meteo` has it, and drives the friction
velocity directly rather than through noodl physics' log law. `background` follows the
model's own boundary count the same way: one `"<layer>.x_boundary"` row per street, or one
network-wide mean. `species` (default `case.species`) selects and orders which of the case's
species end up on the emissions and background drivers. The model's own street order
(`street_index(model)`) must equal `case.street_ids` -- build the model on `case.network`
itself.

`drivers_at` supplies `sigma_theta` (radians) when the case's `meteo` carries it and the model
takes its direction spread from the driver (`direction_spread="driver"`): one network-wide
mean under `meteo="uniform"`, one value per junction under `meteo="per_street"` (from
`meteo_junction["sigma_theta"]` when the case has it, else the mean over the streets meeting
there). A case read from a SIRANE deck has no spread of its own -- SIRANE derives it hour by
hour (`results.meteo["sigma_theta"]`, from `read_results`) -- so it is added as a driver, see
the [SIRANE worked example](#sirane-worked-example).

## `StreetCase.model_options()`

`case.model_options()` reads the closure options `case`'s own source model implies: a
`preset` plus the options the source's own files set, all as `build_model` keywords.

For `source="munich"`, that is `preset="munich"` (whose Monin-Obukhov turbulence,
turbulence-intensity direction spread and hard-coded `u_d_min=0.001` every MUNICH case
implies) plus `munich.cfg`'s `[street]` section, translated:

| `[street]` key | Value | `build_model` keyword |
|---|---|---|
| `Mean_wind_speed_parameterization` | `Exponential` / `Sirane` | `canyon_wind="exponential_profile"` / `"bessel_profile"` |
| `Transfer_parameterization` | `Schulte` / `Sirane` | `roof_exchange="aspect_ratio_scaled"` / `"turbulent_velocity"` |
| `Building_height_wind_speed_parameterization` | `Sirane` / `Macdonald` | `roof_wind="bessel_canyon_mean"` / `"canopy_log_law"` |
| `With_horizontal_fluctuation` | `yes` / `no` | `direction_averaging="rectangle_rule"` / `"none"` |
| `Zref` | m | `z_ref` |
| `Minimum_Street_Wind_Speed` | m/s | `canyon_wind_min` |

A missing `Minimum_Street_Wind_Speed` defaults to MUNICH's own `0.1` m/s
(`canyon_wind_min=0.1`); `Zref`'s absence still raises, and a value with no counterpart (e.g.
the `Wang` transfer) raises `NotImplementedError`.

For `source="sirane"`, the master file does not switch SIRANE's closures, so `model_options`
returns `preset="sirane"`: the Bessel canyon wind, `u_d = sigma_w / (sqrt(2) pi)`, the
non-crossing-streamline junction routing, the exact Gaussian average of the junction routing
over the direction spread (Soulhac et al. 2011, Eq. 7) with the spread from the `sigma_theta`
driver, and the three-branch stable/neutral/unstable `sigma_w` (`stability="monin_obukhov"`).
The deck's turbulence floors `SIGMA_W_MIN` and `SIGMA_V_MIN` become `sigma_w_min` and
`sigma_v_min` when the master file sets them; otherwise the preset's 0.30 and 0.5 m/s,
SIRANE's own defaults, apply. `z_ref` is left at `build_model`'s default: noodl
physics has no SIRANE meteorological preprocessor (SIRANE derives u*, the boundary-layer
height, the Obukhov length and the direction spread from the meteo site's wind, temperature
and cloud cover, over that site's own roughness). A SIRANE case is therefore driven with
SIRANE's own u* -- e.g. from `read_results(...).meteo["u_star"]` -- through the `u_star`
driver, never through noodl physics' log law from the measured wind speed. Likewise the
dispersion site's `Z0D` and `ZDISPL` are recorded in `native["site_disp"]` but not used:
noodl physics' canopy takes `d = 2 h_mean / 3` and `z0 = h_mean / 10` from the network.

A `synthetic` case raises `NotImplementedError` -- pass `build_model`'s keywords directly
instead.

## `StreetCase.synthetic`

`StreetCase.synthetic(network, *, species, times, meteo, emissions, background,
meteo_junction=None, start=None)` builds a case in Python -- an idealised network to drive
directly, or to write out with `write_case`. `meteo` needs `wind_dir_from_deg` and
`wind_speed`; the rest are optional. `meteo_junction` takes any subset of the same keys, or
none at all. Each value is a scalar, an `(n_hours,)` series, or the full `(n_hours, n_streets)`
(`n_junctions` for `meteo_junction`) array. `emissions` and `background` broadcast the same
way, with an optional trailing species axis. Writing the case out with
`write_case(format="munich")` additionally needs `meteo` to carry `h_abl`, `u_star` and
`lmo`.

## `write_case`

`write_case(out_dir, case, *, format="munich", options=None) -> Path` writes `case` under
`out_dir` and returns `out_dir`; `format` is `"munich"` or `"sirane"`, and `case.start` must
be set: both formats date every input.

**`format="munich"`**: `case.meteo` must have `h_abl`, `u_star` and `lmo` -- MUNICH needs
their `PBLH`, `UST`, `LMO` fields whenever transport is on, which this writer always turns
on. A case read from MUNICH files writes its own `[street]` section and projection back
(`read_case` then `write_case` round-trips those two); `[options]` itself always turns
chemistry, photolysis, deposition and scavenging off, and the six `[meteo]` fields MUNICH
always requires (`Rain`, `SolarRadiation`, `SpecificHumidity`, `SurfacePressure`,
`SurfaceTemperature`, `Attenuation`) come from the case where it has one (only
`SurfaceTemperature`, from `meteo["temperature"]`), else a constant default -- either way,
`options` overrides them. `options` are the format's own overrides -- for MUNICH, any
`[street]` closure key, any of those six `[meteo]` fields, or `lat0_deg`/`lon0_deg` (the
lon/lat a synthetic network's `(0, 0)` is anchored to). Missing per-junction meteorology is
derived from the streets meeting at each junction (circular mean for direction, through the
reciprocal for the Obukhov length, a plain mean otherwise).

**`format="sirane"`**: a complete SIRANE v2.1 deck, master file `out_dir / "Donnees.dat"` in
French labels. `times` must be whole consecutive hours; meteorology, background and the
streets' `z0_b` must each be one value for the network (SIRANE has one meteo station, one
background and one building roughness); the wind speed a multiple of 0.1 m/s and the
direction whole degrees (the meteo file's format); the species must be SIRANE species (a
passive tracer is written as an existing one, e.g. NO2, with chemistry and deposition off).
Masses are written in SIRANE's units (g/s, micrograms/m3). The deck's input folder is
`out_dir` relative to its parent (SIRANE's working directory) and its results folder
`<out_dir>/RESULT` (created in advance, with SIRANE's own result subfolders). Every numeric
setting is checked against SIRANE's own range and refused by name outside it. A case read
from a SIRANE deck writes its own physics and numerical settings, site files, street fields
and species flags back.

`options` for `format="sirane"`: `chapman` (0/1, SIRANE's Chapman NO-NO2-O3 chemistry;
default 0), `plume` (0/1, SIRANE's street-plume model above the roofs; default 1),
`deposition` (0/1; default 0), `latitude` (deg; default 51.5), `measurement_height` (m, the
height of the wind speed; default 10), `input_dir`/`result_dir` (SIRANE's two folders,
relative to its working directory), and SIRANE's own keyword for any physics or numerical
setting it writes back (e.g. `U_MIN`).

```python
import tempfile
from pathlib import Path

from noodl.apps.street_aq import read_case, write_case

case = read_case("tests/data/street/sirane_south_kensington/Donnees_SouthKensington.dat")

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp) / "deck"
    write_case(out_dir, case, format="sirane")
    (out_dir / "Donnees.dat").is_file()      # True: the master file write_case writes

    back = read_case(out_dir / "Donnees.dat")   # round trip
    back.source, len(back.street_ids)            # ('sirane', 46)
```

## `write_sweep`

`write_sweep(out_dir, base_case, *, directions_deg, speeds, sources="unit_impulse",
species="NO2", variants=None, chapman=0) -> Path` writes a sweep of SIRANE decks under
`out_dir` -- one per (direction, speed, source street, variant) -- plus a manifest of each
deck's parameters.

Each deck covers two hours -- the first a warm-up -- with the wind from `direction`
at `speed`, zero background, and a unit emission (1 g/s) of `species` on the source street
only. `sources="unit_impulse"` (the default) takes every street in turn; a sequence of
street ids takes those. `variants` names `write_case` option overrides for each variant
(default `{"plume_on": {"plume": 1}, "plume_off": {"plume": 0}}`); `chapman` applies to
every variant that does not set its own.

```python
import tempfile
from pathlib import Path

from noodl.apps.street_aq import read_case, write_sweep

case = read_case("tests/data/street/sirane_south_kensington/Donnees_SouthKensington.dat")

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp) / "sweep"
    write_sweep(
        out_dir, case, directions_deg=[0.0, 90.0], speeds=[3.0], sources=["0", "4"],
        species="NO2",
    )
    sorted(p.name for p in out_dir.iterdir())
    # ['decks', 'runs.csv']
```

**Layout**: `decks/<run-id>/` (each deck, with its results folder `decks/<run-id>/RESULT/`)
and `runs.csv` (every deck's run id, direction, speed, source street, variant and options).
A deck reads back with `read_case("decks/<run-id>/Donnees.dat")`, and SIRANE output in its
results folder with `read_results`.

## Worked example

```python
from noodl.apps.street_aq import build_model, drivers_at, read_case

case = read_case("tests/data/street/munich_paris_excerpt")

model, state, _ = build_model(
    case.network, species=case.species, meteo="per_street", background="per_street",
    **case.model_options(),
)

for k in range(len(case.times)):
    drivers = drivers_at(case, model, k)
    state = model.steady(state, drivers)

print(state["street.x"])          # kg/m3 per street, at the case's last hour
```

*(`munich_paris_excerpt` is a four-street excerpt of MUNICH's own published test case -- see
`tests/data/street/munich_paris_excerpt/NOTICE.md`.)*

## SIRANE worked example

A SIRANE case's own `meteo` has no friction velocity, boundary-layer height or Obukhov
length: [`model_options()`](#streetcasemodel_options) leaves `z_ref` at `build_model`'s
default because noodl physics has no SIRANE meteorological preprocessor. Driving the case
therefore takes these, and the hourly direction spread, from `read_results`:

```python
from dataclasses import replace

import numpy as np
import torch

from noodl.apps.street_aq import build_model, drivers_at, read_case, read_results

FIXTURE = "tests/data/street/sirane_south_kensington"
case = read_case(f"{FIXTURE}/Donnees_SouthKensington.dat")
results = read_results(f"{FIXTURE}/RESULT_SOUTHKENSINGTON", case=case)

# Drive with the RESULTS' own meteorology (u*, h_abl, lmo, direction), not the deck's:
# read_case's own case.meteo has none of h_abl, u_star or lmo.
driven = replace(
    case, meteo={k: v for k, v in results.meteo.items() if k != "sigma_theta"},
    meteo_junction={},
)

# The deck zeroes SIRANE's turbulence floors (model_options() gives sigma_w_min=0.0); the
# archived results use SIRANE's default sigma_w floor of 0.30 m/s, so that is set here.
model, state, _ = build_model(
    driven.network, species=driven.species, meteo="per_street", background="per_street",
    **dict(driven.model_options(), sigma_w_min=0.30),
)

drivers = drivers_at(driven, model, 0, species=driven.species)
# SIRANE's own direction spread (one meteorological station for the whole network),
# broadcast onto every junction -- the exact Gaussian direction average takes it as a
# driver, not a constructor value, because it varies hour by hour.
spread = np.unique(results.meteo["sigma_theta"][0])
drivers["sigma_theta"] = torch.full(
    (len(driven.network.junctions),), float(spread[0]), dtype=torch.float64,
)

state = model.steady(state, drivers)
print(state["street.x"])          # kg/m3 per street, at hour 0
```

*(`sirane_south_kensington` is SIRANE's own South Kensington deck with part of its archived
results -- see `tests/data/street/sirane_south_kensington/NOTICE.md`.)*
