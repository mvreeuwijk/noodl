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
  (`ImperviousSurface` or `PerviousSurface`), `surface` (its name), its parameters, and an
  optional `filename` or `data_input_dict`.
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
  its dict needs a `filename` override that gives `temperature` on every date. On a
  `QueueGroundwater` a dict is refused until a later task of sub-project 3b, and a surface's must
  be empty.
- Classes WSIMOD has that are not read are refused by name with where they go:
  `DecayArc` and `DecayArcAlt` (sub-project 3b), `GrowingSurface`, `IrrigationSurface`,
  `GardenSurface` and `VariableAreaSurface` (3c), and `EnfieldFoulSewer`, whose
  `storm_exchange` is undefined in WSIMOD (`sewer.py:288-291`).

### Arcs

`name`, `type_`, `in_port`, `out_port` (node names), `capacity`, `preference` and, for the queue
arcs (`QueueArc`, `AltQueueArc`), `number_of_timesteps`. `decays` is refused (it belongs to the
`Decay*` arcs of sub-project 3b). WSIMOD's `Arc` accepts any extra keyword silently; the reader
refuses it by name.

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
- A `QueueGroundwater` override of `number_of_timesteps` (a key of its `QueueTank`, not a
  constructor keyword of the node) is refused.
- An override of `decays` on the Storage family merges per pollutant, as `dict.update` does
  (an entry replaces the pollutant's whole dict, so it needs both keys); it is refused on a node
  built with `decays: None` (`storage.py:96-100`) and on a surface (sub-project 3b).
- A node's `filename` override is read as WSIMOD reads it (`nodes.py:93-99`, `model.py:1252-1269`):
  the path as written, opened relative to the process's **working directory**, not the
  configuration's directory (use absolute paths); the file's data replaces the node's forcing.
  It is not the constructor's `filename` key, which is relative to the configuration. A falsy
  non-string (`None`, `0`, `[]`) is ignored. The empty string is refused as a missing file
  (WSIMOD opens `""` and raises `FileNotFoundError`), and any other non-string value is refused
  by name (WSIMOD raises `RuntimeError`). On a class that reads no forcing (a `Waste`, say) the file is
  still read and the override is listed in `WsimodNetwork.unread`. A `Groundwater`'s override is
  its only forcing (X11). A surface's `filename` and any `data_input_dict` override are still
  refused (deposition data, sub-project 3c; `land.py:413-421`).

## Units, order and what runs

The reader converts nothing (W20): volumes are m³, flows and capacities m³/d, additive
pollutants kg, non-additive ones by name (`temperature` in °C), precipitation and
evapotranspiration m/d. The order of `nodes` and of `arcs` is the order in which WSIMOD
constructs them and noodl runs them, so it decides the order of every push and pull and, where
WSIMOD's river order has a tie, the tie rule (configuration order; WSIMOD's own is set-hash
order, which changes with `PYTHONHASHSEED`). The two committed demos, `quickstart` and `oxford`
(`tests/data/wsimod/`, WSIMOD's BSD-3 data in its `Model.save` format), are read by this reader
and run bit for bit as WSIMOD runs them; a configuration with pollutant processes not yet ported
(decay, river biochemistry, deposition) needs `pollutants=False` until sub-projects 3b and 3c. See the [WSIMOD application](../applications/wsimod.md).
