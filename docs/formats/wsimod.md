# WSIMOD `config.yml`

Reading a WSIMOD model description produces a `WsimodNetwork`: the pollutants, the orchestration,
the dates, and the nodes and arcs with their parameters and forcing, as plain dataclasses.
`noodl.apps.wsimod.config.read_wsimod_config(source)` reads the format that WSIMOD 0.8.1's
`Model.load` reads (`config.yml` plus one CSV or CSV.GZ forcing file per node), or the same
structure as a Python mapping. WSIMOD is never imported: a configuration directory needs only
PyYAML (`pip install "noodl-physics[wsimod]"`), a mapping nothing.

The reader checks keys, classes and formats and computes no physics. Quantities that WSIMOD
derives in a constructor (a surface's capacity, a pervious surface's simulated depth, a river's
area, a treatment works' volume constant) are computed when the model is built, from the
parameters read here.

```python
from noodl.apps.wsimod.config import read_wsimod_config

net = read_wsimod_config(
    {
        "pollutants": [], "additive_pollutants": [], "non_additive_pollutants": [],
        "dates": ["2000-01-01", "2000-01-02"],
        "nodes": [
            {"name": "inflow", "type_": "Catchment",
             "data_input_dict": {("flow", "2000-01-01"): 120.0, ("flow", "2000-01-02"): 110.0}},
            {"name": "river", "type_": "River", "length": 200, "width": 20},
            {"name": "foul_outlet", "type_": "Waste"},
        ],
        "arcs": [
            {"name": "inflow_to_river", "type_": "Arc", "in_port": "inflow", "out_port": "river"},
            {"name": "river_to_outlet", "type_": "Arc", "in_port": "river",
             "out_port": "foul_outlet"},
        ],
        "overrides": {"nodes": {"river": {"type_": "River", "name": "river", "length": 300}}},
    }
)
print([(n.name, n.cls, n.slot_key) for n in net.nodes])
print(net.nodes[1].parameters["length"], net.nodes[1].parameters["width"])
print(net.pollutants.K, net.dates[-1])
```

```text
[('inflow', 'Catchment', 'Catchment'), ('river', 'River', 'River'), ('foul_outlet', 'Waste', 'Foul')]
300 20
0 2000-01-02 00:00:00
```

The `small` model in `tests/data/wsimod/small/` (a catchment, a river and a sink, ten days, no
pollutants) is a directory in this format that WSIMOD's own `Model.load` reads as well.

## What is read, ignored and refused

`nodes` and `arcs` are mappings (WSIMOD uses their values; the keys are free) or lists, and are
kept in the configuration's order: a node's in-arcs and out-arcs are filled in the arcs' order,
which fixes the order of every push and pull.

| | Keys |
|---|---|
| **Read** | `nodes` (required), `arcs`, `overrides`, `orchestration`, `pollutants`, `additive_pollutants`, `non_additive_pollutants`, `float_accuracy`, `dates` |
| **Ignored** | `river_discharge_order` (`save` writes it, `load` recomputes it) |
| **Refused by name** | `extensions` when non-empty (WSIMOD would import and run the files); `unified_data_file` (the parquet file needs pandas and pyarrow; per-node CSV and CSV.GZ are read instead); any other top-level key (WSIMOD ignores it silently) |

Absent pollutant lists default to WSIMOD's twelve pollutants, `float_accuracy` to `1e-11` and
`orchestration` to WSIMOD's default order. `read_wsimod_config(..., overrides={...})` replaces
top-level keys first, as `Model.load`'s own `overrides` argument does. A pollutant that is in
neither or both of the additive and non-additive lists, or in a list but not in `pollutants`, is
refused by name.

### Nodes

- `name` and `type_` are required. `node_type_override`, when present, is the class; `type_` is
  then only the node's slot key.
- The **slot key** (`NodeSpec.slot_key`) is the key under which WSIMOD files the node in
  `Model.nodes_type` and which the orchestration names: `"Foul"` for any node whose name contains
  `foul` (WSIMOD's own hack, `model.py:658-660`), otherwise `type_`. A `ResidentialDemand` saved
  by `Model.save` has `type_: Demand` and `node_type_override: ResidentialDemand`, so its slot is
  `"Demand"`.
- `geometry` is dropped, as WSIMOD drops it.
- `filename` (CSV or CSV.GZ, relative to the configuration directory) or `data_input_dict`. A
  file has the columns `variable`, `time`, `value` (other columns are ignored). Inline,
  `data_input_dict` is `{(variable, date): value}`. A node whose class takes no forcing (a
  `Waste`, a treatment works other than `FWTW`) is refused if it names one, as WSIMOD raises
  `TypeError`.
- `surfaces` (a `Land` only): a mapping keyed by name or a list. Each surface has `type_`
  (`ImperviousSurface`, `PerviousSurface`, `GrowingSurface`, `IrrigationSurface`,
  `GardenSurface` or `VariableAreaSurface`), `surface` (its name), its
  parameters, and an optional `filename` or `data_input_dict`. A `GrowingSurface` takes
  `rooting_depth` (its physical depth; `depth` is refused), `ET_depletion_factor`,
  `crop_factor_stages`, `crop_factor_stage_dates`, `sowing_day`, `harvest_day` and
  `initial_soil_storage`, and its other 25 attributes (`crop_cover_max`, `kfr`, ...) only as
  overrides. `crop_factor_stages` and `crop_factor_stage_dates` are lists of one length,
  `sowing_day` and `harvest_day` whole days of the year; the crop calendar and its refusals are
  in the [WSIMOD application](../applications/wsimod.md#the-growing-surface). An
  `IrrigationSurface` takes a `GrowingSurface`'s keys and `irrigation_coefficient` (default 0.1,
  also an override; see [the irrigation surface](../applications/wsimod.md#the-irrigation-surface)).
  A `GardenSurface` takes a `GrowingSurface`'s keys and no other; its `ResidentialDemand` asks it
  for water (see [the garden surface](../applications/wsimod.md#the-garden-surface-and-the-demands-garden-link)).
  With two or more on a `Land`, the last one takes the demand and the earlier ones are listed in
  `model.unread` as `<land>.surfaces.<name>.garden_handlers`.
  A `VariableAreaSurface` takes a `GrowingSurface`'s keys and `current_surface_area` (m2, default
  0; a constructor key only, refused as an override; see
  [the variable-area surface](../applications/wsimod.md#the-variable-area-surface)).
- Every other key must be a constructor keyword of the class's chain (its own `__init__` and
  those its `**kwargs` reach, `noodl.apps.wsimod.classes`). Otherwise it is refused by name,
  where WSIMOD raises `TypeError`. A surface's `capacity` (and an impervious surface's `depth`)
  is refused too: the constructor computes it, so WSIMOD raises `TypeError` for a duplicate.
- `NodeSpec.parameters` holds the class's defaults, then the configuration's values, then the
  overrides, per node (WSIMOD's shared mutable defaults are not reproduced). A treatment works
  with an empty `process_parameters` or `liquor_multiplier` gets WSIMOD's defaults for the
  model's additive pollutants (`wtw.py:73-85`): constant 0.01, exponent 1.001, multiplier 0.7 and
  volume 0.03. A non-empty `process_parameters` must name every additive pollutant with its
  `constant` and `exponent`, and a non-empty `liquor_multiplier` every additive pollutant and
  `volume`; WSIMOD raises `KeyError` otherwise, and so the reader refuses it by name. Entries for
  a pollutant that is not additive or not listed, a `volume` `exponent` and an `exponent` of 0 or
  below are refused when the model is built (WSIMOD never reads the first two, and `0 **
  negative` raises).
- `decays` is `None` (a plain tank) or a dict (a `DecayTank`, `{}` included) on `Storage`,
  `Reservoir`, `RiverReservoir` and `River`: a key per decaying pollutant, each with exactly
  `constant` and `exponent` (numbers, `exponent > 0`); a dict needs the node's `temperature`
  forcing on every date. A `Groundwater` loses its configured forcing (`storage.py:202-203`), so
  its dict needs a `filename` override that gives `temperature` on every date, and so does a
  `QueueGroundwater`'s (`storage.py:291-292`), which builds a `DecayQueueTank`. A surface takes
  `decays` too (default `{}`, a key per decaying pollutant as above); it decays with its land's
  air `temperature`.
- Classes WSIMOD has that are not read are refused by name with where they go:
  `EnfieldFoulSewer`, whose `storm_exchange` is undefined in WSIMOD (`sewer.py:288-291`).

### Arcs

`name`, `type_`, `in_port`, `out_port` (node names), `capacity`, `preference` and, for the queue
arcs (`QueueArc`, `AltQueueArc`, `DecayArc`, `DecayArcAlt`), `number_of_timesteps`. WSIMOD's
`Arc` accepts any extra keyword silently; the reader refuses it by name.

- `decays` is read on the user decay arcs `DecayArc` (a `QueueArc`, `arcs.py:668-728`) and
  `DecayArcAlt` (an `AltQueueArc`, `arcs.py:731-805`) only, with the storage family's form (a
  key per decaying pollutant, each with exactly `constant` and `exponent`, `exponent > 0`) and
  WSIMOD's default `{}`; `decays: None` is refused (WSIMOD iterates it). On every other arc
  class `decays` is refused: WSIMOD would store it unread (`arcs.py:52`). An arc override of
  `decays` is refused too: an arc's `apply_overrides` takes `capacity` and `preference` only
  (`arcs.py:87-96`).
- A decay arc decays at its **in-port's** temperature (`core.py:464-470`), for either
  direction, so that node needs a `temperature` forcing on every date, in both modes; a missing
  date is refused when the model is built, naming the arc and the node. A plain `Node` takes the
  forcing as `data_input_dict` or a `filename` override (which is then read, not listed in
  `unread`). A `Catchment` in-port reads its `temperature` series, whether or not `temperature`
  is a pollutant.
- `parent` is refused on every arc: a decay object reads its temperature from a `parent`
  attribute when it has one (`core.py:465-468`), and a configured value would be read as a
  node.

### Dates

`dates` are read in WSIMOD's formats and order (`to_datetime`, `model.py:139-162`):
`YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD`, `YYYY-MM` (the first of the month). Anything else is
refused, naming the string (`01/02/2000` is). `WsimodNetwork.dates` holds the parsed
`datetime`s and `date_labels` the strings as written. The forcing's `time` column and the keys of
an inline `data_input_dict` are parsed the same way, so `2000-01`, `2000-01-01` and
`2000-01-01 00:00:00` name one day.

## Overrides

The `overrides` section (`nodes` and `arcs`, each a mapping or a list of entries with `type_` and
`name`) is merged into the parameters as WSIMOD's `apply_overrides` consumes it. It does not
compute derived quantities.

- A node override names its node by **slot key and name** (`model.py:819-825`): `"Foul"` for a
  foul sewer, `"Demand"` for a `ResidentialDemand`. An unknown slot key, node, surface or arc is
  refused by name.
- Only the keys the class's chain consumes are accepted; any other is refused by name, where
  WSIMOD only prints "No override behaviour defined". A surface's `capacity` and an impervious
  surface's `depth` are deprecated in WSIMOD, which ignores them with a printed warning; they
  are refused here.
- `pollutant_load` and `liquor_multiplier` merge by update (the other keys are kept),
  `process_parameters` merges per pollutant and per key (an override of one `constant` keeps the
  `exponent`; a partial override is allowed where a partial configured entry is not), and `surfaces` (a `Land`) merges per named surface.
  Every other value replaces the old one. A pervious surface's `depth` is the physical depth, as
  in the configuration.
- A pervious surface's soil-temperature overrides (`soil_temp_w_prev`, `soil_temp_w_air`,
  `soil_temp_w_deep`, `soil_temp_deep`) are state attributes in WSIMOD, not constructor keywords.
  They are stored in `SurfaceSpec.parameters` all the same, which is where the model builder
  reads them.
- A `River`'s biochemistry parameters are set by overrides: `uptake_PNratio`, `denpar_w`,
  `T_wdays`, `halfsatINwater`, `hsatTP`, `limpppar`, `prodNpar`, `prodPpar`, `muptNpar`,
  `muptPpar` (numbers; the defaults are WSIMOD's), and the structural `max_temp_lag` and
  `max_phosphorus_lag` (whole numbers: the sizes of the two lag registers). `bulk_density` is
  accepted and never read (`WsimodNetwork.unread`). The biochemistry runs when `nitrate` is among
  the pollutants and needs `temperature`, `phosphate`, `org-phosphorus`, `ammonia`, `nitrate` and
  `org-nitrogen`; `T_wdays` of 0, a river of zero area and `halfsatINwater <= 0` are refused.
- A `QueueGroundwater` override of `number_of_timesteps` (a key of its tank, not a
  constructor keyword of the node) is accepted on a node with `decays`: it sets the
  `DecayQueueTank`'s and its internal arc's (`tanks.py:682-685`), for example to 0 instead of
  WSIMOD's default 1. On a node without `decays` it is refused, and so is a configured
  (non-override) `number_of_timesteps` on either.
- An override of `decays` on the Storage family merges per pollutant, as `dict.update` does
  (an entry replaces the pollutant's whole dict, so it needs both keys); it is refused on a node
  built with `decays: None` (`storage.py:96-100`). On a surface it merges per surface
  (WSIMOD's shared default `{}` would reach every surface built without a `decays` key).
- A node's `filename` override is read as WSIMOD reads it (`nodes.py:93-99`, `model.py:1252-1269`):
  the path as written, opened relative to the process's **working directory**, not the
  configuration's directory (use absolute paths); the file's data replaces the node's forcing.
  It is not the constructor's `filename` key, which is relative to the configuration. A falsy
  non-string (`None`, `0`, `[]`) is ignored. The empty string is refused as a missing file
  (WSIMOD opens `""` and raises `FileNotFoundError`), and any other non-string value is refused
  by name (WSIMOD raises `RuntimeError`). On a class that reads no forcing (a `Waste`, say) the file is
  still read and the override is listed in `WsimodNetwork.unread`. A `Groundwater`'s override is
  its only forcing (X11). A node's `data_input_dict` override is refused for good
  (`Node.apply_overrides` consumes only `filename`, `nodes.py:92-100`, and would store it unread);
  so is a surface's `filename` override (no surface consumes it). Override a node's forcing by
  `filename`, a surface's by `data_input_dict`.
- A surface's `data_input_dict` override **replaces** its data (`land.py:413-421`), as a mapping in
  the inline `{(variable, date): value}` form or as a path. A path is read as written, relative
  to the process's **working directory** (X27: use absolute paths). `None`, a number or any other
  type is refused by name (`ValueError` in WSIMOD), as is a missing file. The constructor fixed
  the surface's inflows (`land.py:375-382`) and an override does not rebuild them (L15, Y21):
  an override that adds deposition data to a surface built without `nhx-dry` is read and never
  used, and each variable is listed in `WsimodNetwork.unread`; an override that drops `nhx-dry`
  from a surface built with it (WSIMOD raises `KeyError` at the first run) is refused, and, when `nitrate`
  is a pollutant (the only case in which a growing surface reads them, `land.py:1276-1280`), so is
  one that drops a fertiliser or manure variable of a growing surface; without `nitrate` the drop
  is accepted and the run is unchanged.
- A `pollutant_load` override on a surface built **without** a load is ignored, as in WSIMOD, whose
  inflows were fixed at construction (L14): each of its run-pollutant keys is listed in
  `WsimodNetwork.unread` as `<land>.surfaces.<surface>.pollutant_load.<key>` (a key that is no
  run pollutant is listed as for a configured load). WSIMOD's override also mutates a shared
  default `{}`, so every later surface built without a load gets it; noodl's loads are per
  surface. On a surface built with a load the override merges per key and is live.
- A surface's `data_input_dict` (or constructor `filename`) holds its monthly deposition data:
  the six variables `nhx-dry`, `noy-dry`, `srp-dry`, `nhx-wet`, `noy-wet` and `srp-wet`, in kg/m2/d.
  The data are read when `nhx-dry` is among the constructor's variables (`land.py:375-382`), and
  then all six are needed (a missing companion is refused by name, `KeyError` in WSIMOD). Day `t`
  reads the row dated the first of its month (`get_data_input_surface`, `land.py:462-471`): a row
  dated `"2004-01"` or `"2004-01-01"` serves every January day, one dated `"2004-01-15"` is
  never read and the run is refused. Without `nhx-dry` no variable is read and each is listed in
  `WsimodNetwork.unread` as `<land>.surfaces.<surface>.data_input_dict.<variable>`; so is a
  deposition of a species (`ammonia`, `nitrate`, `phosphate`) that is not a pollutant. Any other
  variable, `*-residue` data (residue is not ported) and fertiliser or manure data on a surface
  that is not a growing one are refused.

## Units, order and what runs

The reader converts nothing (W20): volumes are m³, flows and capacities m³/d, additive
pollutants kg, non-additive ones by name (`temperature` in °C), precipitation and
evapotranspiration m/d. The order of `nodes` and of `arcs` is the order in which WSIMOD
constructs them and noodl runs them, so it decides the order of every push and pull and, where
WSIMOD's river order has a tie, the tie rule (configuration order; WSIMOD's own is set-hash
order, which changes with `PYTHONHASHSEED`). The two committed demos, `quickstart` and `oxford`
(`tests/data/wsimod/`, WSIMOD's BSD-3 data in its `Model.save` format), are read by this reader
and run bit for bit as WSIMOD runs them; every ported process runs on them: each demo's
`config.yml` runs unmodified, with no second configuration, the surfaces' `pollutant_load`
included. See the [WSIMOD application](../applications/wsimod.md).
