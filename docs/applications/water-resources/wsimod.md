# Running a WSIMOD model

WSIMOD (the Water Systems Integrated Modelling framework, Imperial College London) models an
urban water cycle as nodes (land, groundwater, rivers, reservoirs, sewers, treatment works,
demands) joined by capacity-limited arcs, and steps it one day at a time by a fixed
orchestration of push and pull requests. noodl runs the same calls in the same order with the
same arithmetic as WSIMOD 0.8.1, so a day computed here equals WSIMOD's bit for bit, and it does
so on batched tensors: each instance of a batch follows its own WSIMOD path, and gradients flow
through every value.

This page is the user guide: load a configuration, run it, read the results, set state and
drivers. Related pages:

- [What is ported](wsimod-ported.md): the node and arc classes and the processes in each.
- [Gradients, batches and long runs](wsimod-gradients.md).
- [Differences from WSIMOD](differences.md): where noodl refuses, records or does something
  different from WSIMOD.
- [Importing a `config.yml`](wsimod-config.md): the file format.
- [Evidence](../../reference/verification/water-resources.md), and the
  [catalogue](../../reference/catalogue/wsimod.md) of every type the application offers.

**Maturity: verified on a stated subset.** Every WSIMOD 0.8.1 node and arc class is ported
except `EnfieldFoulSewer`, and noodl equals WSIMOD exactly (a worst difference of 0) on both of
WSIMOD's demonstration models and on 61 purpose-built cases that cover the features the demos
skip. The subset is stated below.

| Supported | Not supported |
| --- | --- |
| Reading WSIMOD's `config.yml` with per-node CSV or CSV.GZ forcing, or the same structure as a Python mapping | Writing a configuration; `extensions` and the `unified_data_file` (parquet) format are refused |
| Every node and arc class of WSIMOD 0.8.1 except `EnfieldFoulSewer`, with all twelve pollutants | `EnfieldFoulSewer` (undefined in WSIMOD itself); data named `*-residue` |
| Quantity-only runs and runs with pollutants | Devices other than the CPU, dtypes other than float64 |
| Batches of instances that each take their own WSIMOD branch | Checkpointing of long gradient runs (fit in windows) |
| Gradients with respect to every numeric parameter, forcing series and initial store | Gradients across a branch change (they are those of the branch taken) |
| Embedding a WSIMOD model in a larger noodl model through `union` | WSIMOD models that rely on recursion between checks (refused by name) |

## Load and build

```py
from noodl.apps.wsimod import build_model, initial_drivers, initial_state, read_wsimod_config
```

`read_wsimod_config` reads the model from WSIMOD's own description, a `config.yml` with one
forcing file per node. `build_model` turns it into a noodl `Model` whose one closure,
`WsimodDay`, is the day: `model.step(state, drivers, dt=86400, t=k * 86400)` runs day `k` of
the configuration's dates. A WSIMOD model composes with other noodl models through `union`; a
forcing driver is a whole `(T, n)` series, so a coupled run takes the day's row from it.

**Units.** As in WSIMOD, never converted: volumes m³, flows and capacities m³/d, additive
pollutants kg, non-additive ones by name (`temperature` in °C), precipitation and
evapotranspiration m/d.

## A worked example

The `small` model is a catchment's inflow routed through a river to a sink, ten days, quantity
only:

```python
from noodl.apps.wsimod import build_model, initial_drivers, initial_state, read_wsimod_config

network = read_wsimod_config(".")  # config.yml, and the forcing file it names
model, state, drivers = build_model(network)
print(model.wsimod_day.state_labels("wsimod.storage"))
print(sorted(drivers))

for k in range(len(network.dates)):
    state = model.step(state, drivers, dt=86400.0, t=k * 86400.0)
record = model.wsimod_day.last_record
print([round(x, 4) for x in record.arc_flow.tolist()])
print(round(float(state["wsimod.storage"][0, 0]), 4))
```

```text
[('river', 'tank')]
['wsimod.Arc.capacity', 'wsimod.Arc.preference', 'wsimod.River.damp', 'wsimod.River.datum', 'wsimod.River.length', 'wsimod.River.mrf', 'wsimod.River.velocity', 'wsimod.River.width', 'wsimod.flow']
[130.0, 130.0232]
0.1507
```

The state is the stores WSIMOD carries from day to day; the drivers are every numeric parameter,
`"wsimod.<Class>.<parameter>"` with one row per node of that class, the arcs' capacities and
preferences, and the forcing series `(T, n)`. `model.wsimod_day.last_record` is the last day's
record: arc flows and qualities, tank and surface storages, events and the mass-balance terms;
`wsimod_run` records every day.

Values are given by name. Any driver or state key may carry leading batch dimensions, which
broadcast; each instance runs its own WSIMOD day and equals its own single run bit for bit.
Here the river starts with 500 m³ and its outlet arc is a batch of three capacities:

```python
import torch

state = initial_state(model, values={"wsimod.storage": {("river", "tank"): 500.0}})
capacity = torch.tensor([1e15, 60.0, 30.0], dtype=torch.float64)
drivers = initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlet": capacity}})
for k in range(3):
    state = model.step(state, drivers, dt=86400.0, t=k * 86400.0)
print([round(x, 4) for x in state["wsimod.storage"][:, 0, 0].tolist()])
```

```text
[0.1101, 645.0, 735.0]
```

`model.step` needs `dt=86400` and a `t` that is a whole number of days inside the dates.
`model.check(...)` cannot step this model, because its probe steps without `t`; use
`model.step(state, drivers, dt=86400, t=0)` as the check. Gradients of this model are the subject
of [the gradients page](wsimod-gradients.md).

## Running over the dates

`wsimod_run(model, state, drivers, days=...)` is WSIMOD's loop: it steps the same `WsimodDay`
closure over the configuration's dates (or the first `days` of them, `1 <= days <= len(dates)`)
and records each day as WSIMOD's `Model.run(record_all=True)` does. `state` and `drivers`
default to the configured ones and may carry batch dimensions. The run is under
`torch.no_grad()` unless a state or driver tensor requires grad. Stepping with `model.step` and
running with `wsimod_run` give identical values.

```python
from noodl.apps.wsimod import EVENT_NAMES, wsimod_run

model, state, drivers = build_model(network)
diagnostics = {}
results = wsimod_run(model, state, drivers, days=5, diagnostics=diagnostics)
print(results.dates[:2], results.arc_names, tuple(results.arc_flow.shape))
print(results.tank_names, [round(x, 4) for x in results.tank_storage[:, 0].tolist()])
balance = results.mass_balance
print(sorted(balance.terms))
print(float(balance.residual.abs().max()) < 1e-12)
print(set(diagnostics["events"]) == set(EVENT_NAMES), tuple(diagnostics["events"]["demand_deficit"].shape))
print(round(float(balance.inflow[0, 0]), 4), round(float(balance.storage_change[0, 0]), 4))
```

```text
['2000-01-01', '2000-01-02'] ['inflow_to_river', 'river_to_outlet'] (5, 2)
[('river', 'tank')] [0.1389, 0.1275, 0.1101, 0.1043, 0.2316]
['Catchment.flow', 'Catchment.unrouted', 'Waste.total_in', 'dropped']
True
True (5, 3)
120.0 0.1389
```

### Results

`WsimodResults` holds what WSIMOD records, with the batch dimensions in front and the days next:

- `dates` (as written in the configuration), `arc_names`, `arc_flow` `(..., T, n_arcs)` (each
  arc's `vqip_out` volume) and `arc_quality` `(..., T, n_arcs, K)`;
- `tank_names`, `tank_storage` `(..., T, n_tanks)` and `tank_quality`: the attributes WSIMOD's
  `record_all` reports (`Tank`, `QueueTank` and `ResidenceTank`), nodes in configuration order
  and each node's attributes alphabetically;
- `surface_names`, `surface` (`storage`, `evaporation`, `precipitation`, `percolation`,
  `subsurface_r`, `surface_r`, `tank_recharge` and `et0_coef`, each `(..., T, n_surfaces)`, and
  noodl's own `crop_factor`) and `surface_quality`: the surfaces of the nodes under the type key
  `"Land"`, as WSIMOD records them. A field WSIMOD does not record for an impervious surface is
  NaN, and `crop_factor` is NaN except on a growing surface;
- `mass_balance`, below.

The records are taken at the state the day reached, before `end_timestep`, as WSIMOD records
them. Nothing is converted. A tank with `decays` does not appear in the `tank_*` results, as in
WSIMOD, though its storage stays in the state and in the mass balance.

### Diagnostics

A dict passed as `diagnostics` receives:

- `"maxiter"`: for every distribute loop `(node, method, call index)` a `(..., T)` boolean, true
  on the days the loop ended at WSIMOD's iteration cap (WSIMOD prints "Maxiter reached in
  <node> at <date>");
- `"events"`: for every name in `EVENT_NAMES` a `(..., T, n_nodes)` boolean (`(..., T, n_arcs)`
  for the arc-keyed `"decay_arc_order_unknown"`), recording WSIMOD's warning prints as facts.
  `WsimodDay.event_columns(condition)` names the columns. The events are listed below;
- `"decisions"`, only when `diagnostics["decisions"] = True` on entry: a list over the days of
  that day's branch records `Decision(site, call, round, active, outcome, margin)` in call
  order, per instance. The margin says how near a run is to a branch change;
- `"mass_balance"`, the same object as `results.mass_balance`.

Events that reproduce a WSIMOD print: `"storage_unable_to_push"`, `"demand_deficit"`,
`"reservoir_spill"`, `"wwtw_couldnt_push"`, `"service_reservoirs_not_filled"`,
`"sludge_nowhere"`, `"river_cant_push"`, `"ponded_water_cant_reenter"` (never fires),
`"groundwater_couldnt_push_all"`, `"miscalculated_tank_storage"`,
`"environmental_not_able_to_push"`, `"demand_not_able_to_push"`, `"leakage_not_going_to_gw"`,
`"more_treated_than_input"`, `"excess_treated_water"`, `"more_backflow_than_vqip"`,
`"erosion_pool_short"`, `"adsorption_shortcut"` and `"freundlich_flow_adjusted"`.

Events that are noodl's own, recorded where WSIMOD divides by zero, loses mass or cannot report:
`"zero_priority"`, `"wtw_liquor_zero_denominator"`, `"inaccurate_evaporation"`,
`"wtw_negative_solids"`, `"decay_arc_order_unknown"`, `"river_nitrogen_lost"`,
`"crop_water_stress_zero_denominator"`, `"pull_storage_no_din"`,
`"nutrient_pool_not_tracking"`, `"transformation_no_din"` and `"crop_uptake_dry_soil"`. The
[differences page](differences.md) says what each one means.

### The mass balance

WSIMOD's own system check runs but cannot report, because it compares the system's inflow with
itself. noodl computes what it meant to. Per day, `inflow` is the sum of the nodes'
mass-balance-in terms (catchment flows, precipitation, `FWTW` deficits, `UnlimitedDistribution`
supply, demand loads), `outflow` that of their mass-balance-out terms (Waste, evaporation,
demand consumption and backup, unrouted catchment water, unpushed sludge, treatment solids)
plus the water WSIMOD loses without a trace, and `storage_change` the change of every carried
store; `residual = inflow - outflow - storage_change`. Each is `(..., T, 1 + K_additive)`: the
volume, then the additive pollutants' masses. `terms` lists every term by `"<Class>.<term>"`
(arcs' flows cancel and are not listed).

The lost water is `terms["dropped"]`: pushes below WSIMOD's tolerance that a `QueueArc` or
`AltQueueArc` discards, waiting `QueueArc` requests removed on update, the `Land` percolation
push-back, the `WWTW` storm-tank rejection and `make_discharge` push-back, and a `Sewer`'s ponded
water below the tolerance. `terms["WWTW.unprocessed_input"]` is the water pushed into a `WWTW`
after its `calculate_discharge` (a custom orchestration), which WSIMOD empties at `end_timestep`
without treating it. Per-day values (a `Demand`'s totals, a `Land`'s running flows) are terms,
not storage.

A day's window runs from the state it loads to the state it leaves after `end_timestep`, which
is the state the next day loads, so consecutive windows meet exactly. The day's record
(`WsimodDay.last_record`) holds `storage_start`, `storage_total` (at the record point, where
WSIMOD records) and `storage_end` (after `end_timestep`, which the balance uses).

The residual closes to rounding on every day, with one known gap: a `Land`'s outflow in
WSIMOD's balance is IHACRES's evaporation, and when the soil cannot supply it `evaporate`
removes less (the `"inaccurate_evaporation"` event). On every day the residual equals that gap,
`E_actual - E_ihacres` summed over the lands, which WSIMOD's own node check prints as "mass
balance error for volume of ...". The gap is 0 on days without a clamp.

## State and drivers

The state keys, each with a leading batch shape and labelled rows:

| Key | Rows | Holds |
|---|---|---|
| `"wsimod.storage"` | `(node, tank)`, `(node, "surface:<name>")` | every tank's storage, `(..., n, 1+K)`: nodes in configuration order, each node's tanks in alphabetical order, a land's surfaces after its tanks |
| `"wsimod.active_storage"` | `(node, tank)` | each `QueueTank`'s water that can be pulled |
| `"wsimod.travel_queue"`, `"wsimod.travel_queue_present"` | `(arc, i)` | the travel registers of the `QueueTank`s and `AltQueueArc`s, one row per travel time, and their key sets |
| `"wsimod.arc_queue"`, `"wsimod.arc_queue_since"` | `(arc, direction, i, j)`, `(arc, direction)` | every `QueueArc`'s waiting requests and the days since its last delivery |
| `"wsimod.liquor"`, `"wsimod.previous_pulled"`, `"wsimod.treated"` | node | the treatment works' liquor, an `FWTW`'s previous day's pull, a `WWTW`'s treated water |
| `"wsimod.river_temperature_lag"`, `"wsimod.river_phosphorus_lag"` | `(river, i)` | the two registers of every River with biochemistry (degC, kg) |
| `"wsimod.total_decayed"` | `(node, "tank.internal_arc")`, `(node, "surfaces.<name>")` | decayed mass carried to the next day (see [What is ported](wsimod-ported.md)) |
| `"wsimod.nutrient_pool"` | `(node, "surfaces.<name>.nutrient_pool", pool, nutrient)` | every growing surface's five nutrient pools, `N` then `P` of each (kg), when `nitrate` is a pollutant and pollutants are on. Setting a growing surface's tank in `"wsimod.storage"` does not derive the dissolved pools from it, as WSIMOD's constructor does, so set both together |
| `"wsimod.days_after_sow"`, `"wsimod.crop_cover"`, `"wsimod.ground_cover"` | `(node, "surfaces.<name>")` | every growing surface's days since sowing (`-1` before sowing) and its crop and ground cover |
| `"wsimod.surface_climate"` | `(node, "surfaces.<name>", "evaporation" or "precipitation")` | every irrigation and garden surface's last evaporation and precipitation volumes (m³) |

A store is a VQIP (volume, quality, in WSIMOD's terms): the volume, then one column per
pollutant. In `initial_state` a number is a volume with no pollutants and a mapping a VQIP by
name, `{"volume": 5.0, "phosphate": 0.1}`; a `QueueTank` given a storage starts with all of it
active, as in WSIMOD.

Parameters keyed by pollutant (`pollutant_load`) have one column per pollutant, absent ones 0;
time-area diagrams (`"wsimod.Sewer.pipe_timearea"`, `"wsimod.QueueGroundwater.timearea"`) have
one column per travel time. Their keys are structure and stay as configured; the fractions are
drivers. Quantities WSIMOD derives in a constructor (a river's area, a surface's capacity) are
derived from the day's parameters, never given. A driver left out of the dictionary takes its
configured value.

## Quantity only or with pollutants

`build_model(network, pollutants=False)` runs on the volumes alone (`K = 0`). The parameters such
a run does not read are listed by name in `model.ignored_by_quantity_only`: surface deposition
loads, a demand's `pollutant_load`, the treatment works' pollutant entries, the growing-surface
parameters for uptake, erosion, adsorption and denitrification, every `decays` value, the
River's biochemistry parameters and a catchment's pollutant series. The parameters that set
volumes (`ET_depletion_factor`, `rooting_depth`, `crop_factor_stages`, the crop calendar) are
still read.

What WSIMOD accepts and never reads is `model.unread`: a `Groundwater`'s or `QueueGroundwater`'s
configured forcing, and a `filename` override on a class that reads no forcing (a `River` reads
it when `nitrate` is among the pollutants). With `pollutants=True` (the default) every ported
process runs. A pollutant key on a surface that is not a run pollutant is listed in
`model.unread` as `<land>.surfaces.<surface>.pollutant_load.<key>`.

### Surface deposition

A surface built with `nhx-dry` in its forcing runs atmospheric and precipitation deposition: each
reads `nhx`, `noy` and `srp` (kg/m²/d), multiplies by the area and forces the amounts into the
tank as `ammonia`, `nitrate` and `phosphate` at volume 0. A species that is not a run pollutant
is read and dropped (listed in `model.unread`). Day `t` reads the row of the first of its month,
so the data are monthly; the forcing is checked for every run date in both modes. The data are
the drivers `wsimod.nhx-dry` to `wsimod.srp-wet`, `(T, n_surfaces)`, rows
`(land, "surfaces.<name>")`, filled from the monthly table. A batched driver may vary within a
month, which WSIMOD cannot express. Dry data without the wet ones, `*-residue` data, and
fertiliser or manure data on a surface that is not a growing one are refused.

### The demonstration models

Each demo (`quickstart`, `oxford`) has one configuration, `config.yml`, as WSIMOD saved it. Each
runs with `pollutants=False` on the volumes, and with every pollutant and every process the demo
contains live. On `oxford` that is the `FWTW`'s and `WWTW`'s treatment of 11 additive pollutants
and the rural surface's soil temperature; on `quickstart` the soil temperature and the nitrite
carry-over. Neither demo has a non-empty `decays`, a `DecayQueueTank`, a user decay arc or a
`River`; purpose-built cases cover those. Both demos run unmodified, bitwise as WSIMOD does.

## What is refused

Everything that cannot run is refused by name when the model is built, where WSIMOD would fail
mid-run, hang or silently do something else:

- out-of-scope classes and keys (by the reader);
- River biochemistry that WSIMOD cannot run; `decays` on an arc class other than `DecayArc` and
  `DecayArcAlt`; a decaying `Groundwater` or `QueueGroundwater` without a `filename` override
  that gives the temperature; a `decays` key that is not a configured pollutant, an entry
  without exactly `constant` and `exponent`, a non-numeric value, an `exponent` of 0 or below;
- a `total_porosity` not above `field_capacity`, a residence time of 0, `leakage` outside
  `[0, 1)`, preferences that are negative or all zero, travel times that are not whole days;
- a cycle that makes WSIMOD's checks recurse into each other, which makes WSIMOD retry at every
  level and never finish: every cycle of pushes, a cycle of pulls only when it lies upstream of a
  node whose orchestrated step pulls (`create_demand`, `treat_water`, `make_abstractions`), and a
  base `WTW` in a pull cycle. The rule over-approximates: it may refuse a model WSIMOD runs,
  never the reverse. WSIMOD's own `Land -> Sewer -> Land` ponding is not a cycle;
- river arcs without a `Waste`, an arc into a `Catchment`, an orchestration slot whose nodes
  lack the method;
- missing forcing, naming the node, the variable and the first missing date (a `Land` needs
  `temperature` even in a quantity-only run, as WSIMOD reads it);
- a `leakage` batch that mixes zero and positive values on one `Distribution`, because WSIMOD
  decorates the node's handlers only for a positive configured leakage.

## Limitations

- **Scope.** Every WSIMOD 0.8.1 class is ported except `EnfieldFoulSewer`. Not ported:
  `GrowingSurface.residue` (surface data named `*-residue` is refused), the `NutrientPool`
  override hook (the pool's parameters are drivers instead), the `("RiparianBuffer", "volume")`
  check (no such class exists in WSIMOD 0.8.1) and `Model.change_impervious_area`.
- **Input and output.** A configuration is read, not written. `extensions` and
  `unified_data_file` are refused.
- **Speed.** The day is a transcription of many small operations, with no compilation, so it is slower per day than a compiled model; a batch amortises the cost.
- **Not re-entrant.** One `WsimodDay` holds one engine that every step reuses, so two steps of
  one model must not run at once (threads, or a step that calls another).
- **`Model.to`.** `model.to(dtype=torch.float64)` (or a CPU device) works and changes nothing.
  Another dtype or a device move is refused by name (`TypeError`): the port is float64 on the
  CPU, as bitwise equality with WSIMOD needs.
- **River ties.** WSIMOD breaks a tie in the river order by set-hash order, which varies with
  `PYTHONHASHSEED`; noodl takes the configuration order, so a model whose river ties matter may
  differ from a WSIMOD run in the last bit. Neither demo has a `River`.
- **A gapped time-area** (`QueueGroundwater`) may differ from WSIMOD by an ulp: WSIMOD walks the
  queue in dict insertion order, noodl in ascending time.
- **The recursion rule over-approximates** and may refuse a model WSIMOD runs.
