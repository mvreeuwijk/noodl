# WSIMOD

WSIMOD 0.8.1's daily step, transcribed into noodl. WSIMOD (the Water Systems Integrated
Modelling framework, Imperial College London) models an urban water cycle as nodes — land,
groundwater, rivers, reservoirs, sewers, treatment works, demands — joined by capacity-limited
arcs, and steps it one day at a time by a fixed orchestration of push and pull requests. noodl
runs the same calls in the same order with the same arithmetic, so a day computed here equals
WSIMOD's bit for bit, and it does so on batched tensors: each instance of a batch follows its
own WSIMOD path, and gradients flow through every value.

The model is read from WSIMOD's own description, a `config.yml` with one forcing file per node
(see [WSIMOD `config.yml`](../formats/wsimod.md)). `build_model` turns it into a noodl `Model`
whose one closure, `WsimodDay`, is the day: `model.step(state, drivers, dt=86400, t=k * 86400)`
runs day `k` of the configuration's dates. A WSIMOD model composes with other noodl models
through `union`; a forcing driver is a whole `(T, n)` series, so a coupled run takes the day's
row from it.

**Units.** As WSIMOD's own (W20), never converted: volumes m³, flows and capacities m³/d,
additive pollutants kg, non-additive ones by name (`temperature` in °C), precipitation and
evapotranspiration m/d.

```python
from noodl.apps.wsimod import build_model, initial_drivers, initial_state, read_wsimod_config
```

Every type this application offers is listed in its [catalogue](../catalogue/wsimod.md). It is
set up and run like every other model; see [Using noodl](../usage.md).

## What is ported

The node classes `Node`, `Waste`, `Catchment`, `Storage`, `Groundwater`, `QueueGroundwater`,
`River` (with its nitrate biochemistry), `Reservoir`, `RiverReservoir`, `Demand`,
`NonResidentialDemand`, `ResidentialDemand`, `Distribution` (with leakage),
`UnlimitedDistribution`, `Sewer`, `WTW`, `WWTW`, `FWTW` and `Land` with `Surface`,
`ImperviousSurface` and `PerviousSurface` (IHACRES); the arcs `Arc`, `QueueArc`, `AltQueueArc`,
`PullArc`, `PushArc`, `SewerArc` and `WeirArc`. The soil temperature of a `PerviousSurface` is
live (below), and so are the treatment works' pollutant treatment and the first-order decay of
the storage family's tanks, a `QueueGroundwater`'s queue and a land's surfaces, and so is a
`River`'s biochemistry (below), and the user decay arcs `DecayArc` and `DecayArcAlt`. What
still changes a pollutant and is not ported (deposition and the land's nutrient pools,
sub-project 3c) is refused unless it is neutral.

| Class | Process ported (sub-project 3b) |
|---|---|
| `Storage`, `Reservoir`, `RiverReservoir`, `Groundwater` | first-order decay of the tank at the end of the day, when `decays` is given (`DecayTank`) |
| `QueueGroundwater` | the same, in a `DecayQueueTank` whose internal queue is a `DecayArcAlt` |
| `River` | the decaying tank, and the biochemistry: temperature relaxation, denitrification, nitrogen and phosphorus production and mineralisation, macrophyte uptake, with the two lag registers as carried state |
| `WTW`, `WWTW`, `FWTW` | the per-pollutant treatment (`process_parameters`) and liquor multipliers |
| `Land` surfaces | decay at the land's air temperature, and the nitrite carry-over of the day's decayed ammonia and nitrite |
| `PerviousSurface` | the soil temperature |
| `DecayArc`, `DecayArcAlt` (user arcs) | decay of the requests in transit, at the in-port's temperature |

### Decay in the storage family

A `Storage`, `Reservoir`, `RiverReservoir` or `River` built with a `decays` dict (an empty one
included) owns a `DecayTank` instead of a `Tank` (`storage.py:53-59`). At the end of every day,
after the orchestration and the record, the tank decays its storage (`tanks.py:442-447`,
`core.py:474-521`): for every pollutant `p` in `decays`,

```
diff[p]  = storage[p] * min(constant[p] * exponent[p] ** (T - 20), 1)
storage[p] -= diff[p]
```

with `T` the node's own `("temperature", date)` forcing. The pollutant may be non-additive (`do`
decays like any other); the volume and every pollutant not in `decays` are untouched. The power
is Python's, evaluated once per node and day, so every value is bitwise WSIMOD's. The decayed
mass is the mass-balance term `"DecayTank.decayed"` (an outflow) and the per-day store
`tank.total_decayed`, so the balance closes over a window that includes the end of the day
(X8).

Where `T` comes from:

| Node | Temperature |
|---|---|
| `Storage`, `Reservoir`, `RiverReservoir`, `River` | the node's own forcing, given as the constructor's `data_input_dict` or `filename` key (a constructor key, read relative to the configuration's directory), or replaced by a `filename` override (an override key, read relative to the working directory); a missing date is refused by name when the model is built, whatever `decays` lists (an empty dict reads it too, `core.py:484-487`) |
| `Groundwater` | a `filename` **override** only: WSIMOD loses the constructor's `data_input_dict` and `filename` (`storage.py:202-203`), so a configured forcing is listed in `model.unread` and a `decays` dict without the override's `("temperature", date)` on every date is refused by name (X11). Use an absolute path: the override is opened relative to the working directory, as in WSIMOD |
| `QueueGroundwater` | the same (`storage.py:291-292`): a `filename` override only |
| a node with `decays: None` | none |

The kinks. The cap `min(factor, 1)` is a decision, `node:<name>/tank/make_decay/cap:<pollutant>`,
true where the factor exceeds 1, evaluated at the day's bind (it depends on the temperature and
the parameters only), with the margin `factor - 1`; a factor of exactly 1 is not capped, and
gives the same decayed mass either way. The factor is non-differentiable at the cap, where its
gradient jumps from the exponential's to 0.

Drivers: `"wsimod.<Class>.decays.constant"` and `".exponent"`, `(..., n, K)`, for every node of
the class, nodes built with `decays: None` included (all-inert rows), columns the `K` pollutants.
The configured key set is structure: an entry outside it is inert (`constant` 0, `exponent` 1)
and `bind` refuses any other value by name, as it does for a node with `decays: None`. Both are
absent in a quantity-only run, which decays nothing (it carries no pollutant) but still reads
the temperature. An `exponent` of 0 or below is refused (WSIMOD raises or returns a complex
number), and so is a `constant` or `exponent` that is not a number.

A `decays` override merges per pollutant, as `dict.update` does: an entry replaces that
pollutant's whole dict, so `{phosphate: {constant: 0.3}}` is refused after the merge for the
missing `exponent`. An override on a node built with `decays: None` is refused (WSIMOD raises
`ValueError`, `storage.py:96-100`).

WSIMOD's `record_all` lists only tanks whose class is exactly `Tank`, `QueueTank` or
`ResidenceTank` (`model.py:1123`): a `DecayTank` drops out of the `tank_*` results, and a user
who adds `decays` to a node sees its tank disappear from them. Its storage stays in the state
(`"wsimod.storage"`) and in the mass balance. The same holds for a `QueueGroundwater`'s
`DecayQueueTank` (below), whose `QueueTank` row a user who adds `decays` sees disappear (X40).

### Decay in a `QueueGroundwater`

A `QueueGroundwater` with a `decays` dict owns a `DecayQueueTank` instead of a `QueueTank`
(`storage.py:307-315`, `tanks.py:645-696`): its internal queue is a `DecayArcAlt`
(`arcs.py:731-805`), whose water decays by the rule above on entry and at every end of day, with
the node's temperature from its `filename` override. The cap decision is
`node:<name>/tank/internal_arc/make_decay/cap:<pollutant>`. WSIMOD's class is transcribed as
written (X13), which differs from a non-decaying `QueueGroundwater` in four ways:

- **One day longer.** `DecayQueueTank`'s `number_of_timesteps` defaults to **1**
  (`tanks.py:648`), against its docstring and `QueueTank`'s 0, and `QueueGroundwater` passes
  none, so adding `decays` delays all of the node's water by one day: a time-area `{0: 0.5, 1:
  0.5}` queues under keys 1 and 2, and nothing is active at once (X12). The override
  `number_of_timesteps` (`DecayQueueTank.apply_overrides`, `tanks.py:682-685`) sets it, for
  example to 0 (X28); it is refused on a node without `decays`.
- **The decay reaches `storage` a day late.** A push adds the **undecayed** water to `storage`
  (`tanks.py:536-538`) and the decayed water to the queue. At the end of day d, `storage` loses
  the internal arc's `total_decayed` (`tanks.py:694`): that day's entry decay plus the end-of-day
  decay of day d - 1; the decay of day d's end waits in `total_decayed` until the end of day d +
  1. So `total_decayed` is carried state, row `(node, "tank.internal_arc")` of
  `"wsimod.total_decayed"`, and what `storage` loses is the mass-balance outflow
  `"DecayQueueTank.decayed"`. The balance counts the tank by its `storage`, so it closes every
  day. Only the pollutant columns lag (decay changes no volume), and no pull reads them: the
  pulls take their water from `active_storage` or, `pull_set_active`, pro rata from the queue.
  Its order of addition is WSIMOD's: reset, slots 1 to L ascending, slot 0, then the next day's
  entries in push order (X30).
- **`active_storage` never decays** (the TODO at `tanks.py:692-693`): water decays only while
  it is in the queue.
- **No flush at the end of the day.** `QueueTank`'s end of day ends with `update_queue`;
  `DecayQueueTank`'s does not. Water that the end of day d moves to slot 0 becomes active only at
  the next push (every push flushes slot 0) or at `distribute`'s `update_queue`
  (`storage.py:352`), and decays again at each end of day it waits. The new slot 0 is
  `sum_vqip(decay(slot 1), decay(slot 0))`, the other order from `AltQueueArc`'s; the blend is
  symmetric, so the order shows only where the two slots hold no water and the first one's
  non-additive values are kept.

Because the timing differs, a quantity-only run keeps the `DecayQueueTank` structure (X25): its
delay, its end of day and its `total_decayed` row (all zero), with no `decays.*` driver.

### The user decay arcs `DecayArc` and `DecayArcAlt`

A `DecayArc` (`arcs.py:668-728`) is a `QueueArc` whose requests decay; a user `DecayArcAlt`
(`arcs.py:731-805`) is an `AltQueueArc` with the `DecayQueueTank` internal arc's entry and end of
day (above). Both decay by the storage family's rule with their own `decays`, at the
temperature of their **in-port** (`core.py:464-470`; a WSIMOD TODO, `arcs.py:700-702`, notes it
is not the water's own), for pushes and pulls alike. Their drivers are
`"wsimod.DecayArc.decays.constant"`/`".exponent"` and `"wsimod.DecayArcAlt.decays.*"`,
`(..., n_arcs_of_class, K)`, one row per arc in configuration order, inert `(0, 1)` where
`decays` lists no pollutant (absent in a quantity-only run). They are bound after every node,
so the factor is the in-port's temperature of the same day (D15). The cap decision is
`arc:<name>/make_decay/cap:<pollutant>`.

- **Entry.** The request passes `enter_arc` undecayed, so `vqip_in` and `flow_in` count it as
  sent, then decays, then waits. A pulled request decays after it left the sender, which loses
  the undecayed water. A push's backflow is the decayed water, returned to the sender and taken
  off `vqip_in` (`arcs.py:399-405`). With `number_of_timesteps: 0` a request decays once and is
  delivered at once.
- **End of the day.** Every waiting request decays in place, in both directions; a
  `DecayArcAlt` shifts its register as above (slots 1 to L, then slot 0).
- **Mass balance.** Each decayed mass is the outflow `"DecayArc.decayed"` or
  `"DecayArcAlt.decayed"`: the day's entries in call order, then the end of the day, in
  WSIMOD's order. For a `DecayArc` that order is WSIMOD's list order, entry age descending and
  then the call order of the entry day. noodl's rows know a request's age only while it is not
  yet due and waiting (the days-since counter of a direction saturates at
  `number_of_timesteps`), so the term's order of addition is exact while no end of day finds
  due-and-waiting rows in both directions. Where one does, the event
  `"decay_arc_order_unknown"` is recorded at the arc, and only the term's last bit may differ;
  the delivered water and the queued requests are WSIMOD's in every case.
- **A catchment in-port** (X29). A catchment that a decay arc leaves has its own
  `"wsimod.temperature"` row, which the arc's decay reads; the catchment's outflow reads its
  `"wsimod.concentration"` columns. Where `temperature` is a pollutant the same value is in
  both drivers: changing one does not change the other, so a user who changes the catchment's
  temperature changes both. The arc also runs where `temperature` is not a pollutant, and in a
  quantity-only run.

### Decay on a land's surfaces and the nitrite carry-over

A `Surface` is a `DecayTank`: with a `decays` dict it decays its storage at the end of every
day, by the same rule as the storage family (`land.py:303-581`, `tanks.py:442-447`), with the
`Land`'s **air** temperature (`DecayObj` reads its parent's `("temperature", date)`,
`core.py:462-468`), not the soil temperature a `PerviousSurface` works out. A surface's
`decays` default to `{}`, which decays nothing. The cap decision is
`node:<land>/surfaces.<surface>/make_decay/cap:<pollutant>`. Drivers:
`"wsimod.<ImperviousSurface|PerviousSurface>.decays.constant"` and `".exponent"`, `(..., n, K)`,
labelled `(node, surface)`, inert for a pollutant the surface does not list (refused by name
otherwise) and absent in a quantity-only run.

The carry-over. With `"nitrite"` among the pollutants, `Surface.run` starts with yesterday's
decayed mass (`land.py:427-439`):

```
storage["nitrate"] += total_decayed["nitrite"];  running_inflow_mb["nitrate"] += total_decayed["nitrite"]
storage["nitrite"] += total_decayed["ammonia"];  running_inflow_mb["nitrite"] += total_decayed["ammonia"]
```

so decayed ammonia becomes nitrite and decayed nitrite becomes nitrate, and both count as
inflows of the land. Decayed nitrate, phosphate or anything else is lost: it is the
mass-balance outflow `"DecayTank.decayed"`. Without `"nitrite"` the block is skipped and all of
the decayed mass is lost. The block reads `nitrate`, `nitrite` and `ammonia`: a `Land` with
`nitrite` but not the other two is refused by name. `total_decayed` starts empty, so day 0
converts nothing, and is carried state, `"wsimod.total_decayed"` (below). WSIMOD converts at
every `Surface.run`, so a custom orchestration that runs a `Land` twice a day converts the same
mass twice (transcribed; case `land_run_twice`).

Two label forms. `"wsimod.total_decayed"` labels a surface row with its attribute path,
`(node, "surfaces.<name>")` (X35), while `"wsimod.storage"` keeps `(node, "surface:<name>")`.

WSIMOD's default `decays={}` is one dict shared by every surface built without a `decays` key
(a mutable default argument), so a `decays` override of one of them reaches all of them. noodl
merges per surface: the override reaches only the surface it names (D13).

### Treatment works

`WTW.treat_current_input` (`wtw.py:163-214`) splits the day's input into treated water, liquor and
solids. Every additive pollutant `k` has its own `process_parameters[k] = {constant, exponent}` and
`liquor_multiplier[k]`; the volume's constant is derived, `1 - percent_solids -
liquor_multiplier["volume"]`. With `T` the influent's temperature:

```
treated[k] = influent[k] * constant[k] * exponent[k] ** (20 - T)     (volume: no factor)
liquor[k]  = influent[k] * liquor_multiplier[k]                       (overwritten, not blended)
solids[k]  = (influent[k] - treated[k]) - liquor[k]                   (the remainder, unclamped)
```

The power is Python's, evaluated once per element, so every value is bitwise WSIMOD's. Three
quirks are transcribed. The factor *rises* as the temperature falls when `exponent > 1`
(`20 - T` grows). Nothing caps `constant * factor + liquor_multiplier`: where it exceeds 1 the
solids go negative and the works creates mass (WSIMOD's lists still close, the solids being the
remainder), and noodl records the event `"wtw_negative_solids"` on that day. And with no
influent the temperature column is whatever `current_input` holds, 0 after a reset; `exponent ** 20`
then multiplies a zero mass.

The drivers, one row per node of the class (`WTW`, `WWTW` or `FWTW`):

| Key | Shape | Columns |
|---|---|---|
| `"wsimod.<Class>.process_parameters.constant"`, `"....exponent"` | `(..., n, K_additive)` | the additive pollutants; absent in a quantity-only run |
| `"wsimod.<Class>.liquor_multiplier"` | `(..., n, 1 + K_additive)` | `volume`, then the additive pollutants; `(..., n, 1)` in a quantity-only run |

`liquor_multiplier` was `(..., n)`, the volume entry alone, in 3a: a driver of that shape is
refused, naming the key and the new shape. An `exponent` of 0 or below is refused (also in a
batched driver, when the day is bound): `0 ** negative` raises where `T > 20` and a negative base
gives a complex number. A configured `process_parameters` or `liquor_multiplier` must name every
additive pollutant (WSIMOD raises `KeyError` otherwise), and every `constant`, `exponent` and
liquor multiplier must be a real number (a string, `None` or a boolean is refused by name);
an override may be partial and merges.
Entries for a pollutant that is not additive, for one the model does not list, and a `volume`
`exponent` are refused: WSIMOD never reads them.

### The soil temperature

With a `temperature` pollutant, a `PerviousSurface` moves its storage's temperature every day
(`land.py:1060-1079`), after the IHACRES step:

`T = (T_storage * soil_temp_w_prev + T_air * soil_temp_w_air + soil_temp_deep * soil_temp_w_deep) / (soil_temp_w_air + soil_temp_w_deep + soil_temp_w_prev)`

WSIMOD's defaults are `w_prev = 0.1`, `w_air = 0.6`, `w_deep = 0.1`, `soil_temp_deep = 10`; any
weights are accepted, and the port is bitwise equal to WSIMOD's. Weights that sum to 0 are
refused by name (`"total weight"`), in the configuration and, for a batched parameter, at `bind`,
where WSIMOD would divide by zero. A quantity-only run does not read the temperature.

A day is WSIMOD's (`model.py:993-1176`): the orchestration slot by slot, each slot's nodes in
configuration order; `River.distribute` in the river order, most upstream first; the record;
then `end_timestep` on every node and every arc. Where WSIMOD breaks a tie in the river order by
Python's set-hash order, which changes with `PYTHONHASHSEED`, noodl takes the configuration
order.

### River biochemistry

A `River` runs WSIMOD's in-river biochemistry (`storage.py:738-898`) when `nitrate` is among the
pollutants, in the orchestration slot of `calculate_discharge`, before the day's catchment
inflow. It needs `temperature`, `phosphate`, `org-phosphorus`, `ammonia`, `nitrate` and
`org-nitrogen` among the pollutants and a `temperature` forcing on every date; the River reads
that forcing in a quantity-only run too, and a quantity-only run ignores the parameters below.
The steps, in WSIMOD's order and operation order, each cited in the code:

1. the depth `V / area`;
2. the water temperature relaxes towards the air: `T = (1 - 1/T_wdays) T + (1/T_wdays) T_air`,
   on dry and frozen days too;
3. both registers shift in `T` and `phosphate + org-phosphorus`, before the guards, so every
   call appends;
4. the guards `V < FLOAT_ACCURACY` (dry) and `T <= 0` (frozen) end the call with nothing moved;
5. denitrification, `min(denpar_w * area * 2^((T-20)/10) * confcn, 0.5 din)` (times `T/5` below
   5 degC), taken from ammonia and nitrate in proportion;
6. the 10-day and 20-day temperature means, the phosphorus "mean", `tempfcn` and `totalphosfcn`;
7. production (`minprodN > 0`: dissolved nitrogen and phosphate into the organic pools) or
   mineralisation (the reverse), each capped at half its pool;
8. macrophyte uptake of dissolved nitrogen and phosphate, capped at half the pools.

The ten parameters are drivers `"wsimod.River.<p>"`, set by overrides only: `uptake_PNratio` (-),
`denpar_w` (kg/m2/d), `T_wdays` (d), `halfsatINwater`, `hsatTP`, `limpppar` (kg/m3), `prodNpar`,
`prodPpar` (kg/m3/d), `muptNpar` and `muptPpar` (kg/m2/d); their defaults are WSIMOD's Python
expressions (`1 / 7.2`, `0.0015`, `5`, `1.5 * 1e-3`, ...). `max_temp_lag` (20) and
`max_phosphorus_lag` (365) are structural: the registers have `max_temp_lag + 1` and
`max_phosphorus_lag + 1` slots. `bulk_density` is set by an override and never read
(`model.unread`). `"River.bio_in"` and `"River.bio_out"` are mass-balance terms that add every
call's gains and losses.

**Quirks, reproduced (X14, X15).**

- Each window sum divides by its full length from the first day (a cold start): the 10-day
  mean of one temperature is that temperature over 10.
- The "365-day" phosphorus mean adds 366 entries (the list is cut after the append) and divides
  by 365.
- `totalphosfcn` compares the phosphorus **mass** (kg) with the concentrations `limpppar` and
  `hsatTP` (kg/m3), and is negative for `limpppar - hsatTP < TP < limpppar`.
- Mineralisation with no dissolved nitrogen removes nitrogen from `org-nitrogen` and adds it
  nowhere: the loss is in `bio_out` and the event `"river_nitrogen_lost"` says so.
- `minprodN` of `-0.0` takes the mineralisation branch and moves nothing.
- `calculate_discharge` **assigns** `bio_in`/`bio_out` (`storage.py:923-924`), so an
  orchestration that calls it twice a day leaves WSIMOD's own node balance missing the first
  call's terms, and WSIMOD prints a mass-balance error for the river on every such day. The
  processes run twice in noodl too (X16); its terms add both calls and its balance closes.

**Verified by two whole-model cases** (`tests/verification/_wsimod_cases.py`). `river_nutrients`
is two rivers in a chain fed by catchments with all six pollutants, one river overridden to
`max_temp_lag = 5`, `max_phosphorus_lag = 300` and raised rates, and a reservoir abstracting
from the upstream river. Its forcing places every branch inside its first 400 days: a frost spell
(`T <= 0` after relaxation), days between 0 and 5 degC, a drought in which the abstraction drains
the upstream river to exactly 0 (the dry guard), a DIN-free refill with a cold snap (mineralisation
with `din = 0`: `"river_nitrogen_lost"`), a phosphate pulse that carries the low-P river's
365-day mean across the `totalphosfcn` guard, warming and cooling seasons (production and
mineralisation), and every 50 % cap binding on some days and not on others. The default suite
runs its first 400 days at value parity, bitwise, without decisions (X39); `-m slow` runs all
1100 days (three turns of the 366-slot register; the pulse leaves it on days 426-455 and the
`totalphosfcn` guard is False again from day 448) and asserts each river's branches from the
recorded decisions on the 400-day prefix and on the full run (D24). `river_double_discharge`
calls `calculate_discharge` twice a day: the values are at parity, WSIMOD prints its River
mass-balance error on every day and noodl's balance closes within rounding (2e-16 of scale);
the parity harness takes those pollutant prints as the one named exception to its print check
(D17, X32).

**The registers and warm starts.** The state keys `"wsimod.river_temperature_lag"` (degC) and
`"wsimod.river_phosphorus_lag"` (kg) hold the two registers, rows `(river, i)`, slot 0 the
oldest, zero-padded at the front. All zeros is WSIMOD's empty lists: no sum notices the padding
(`+0.0` leaves a sum unchanged), and each window sum is the interpreter's own `sum` over the
slots as Python floats (compensated from CPython 3.12), so the sums equal Python's over WSIMOD's
lists, bitwise, with the plain sum's gradient. A run can start from another run's registers. Parity assumes
WSIMOD's lists hold Python floats, as `read_csv` produces; `numpy.float64` items get CPython's
plain sum. They are not water held, so the mass balance skips them. A river with `max_temp_lag = 5` has 6 slots,
and both temperature windows are the whole register.

**Refused by name:** a missing pollutant of the six, a missing temperature on a date,
`T_wdays = 0`, `length * width = 0`, `halfsatINwater <= 0` (WSIMOD divides by `din / V +
halfsatINwater`, which is 0 where `din` is), `max_temp_lag` that is not a non-negative integer and
`max_phosphorus_lag` that is not a positive integer. The three drivers are also refused at
`bind`.

**Kinks and the singularity.** The four 50 % caps, the `T < 5` factor, the production and
mineralisation switch, and the `max(0, .)` of the macrophyte factors are kinks; the dry guard is
a jump; `totalphosfcn` is a singularity as `TP - limpppar + hsatTP` tends to 0 from above (it
tends to minus infinity and jumps to 0 at the guard). The two powers, `2 ** ((T-20)/10)` and
`(T/20) ** 0.3`, are Python's, per instance.

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

Every driver and every initial store can carry `requires_grad`. The river's storage after three
days, with respect to its velocity:

```python
velocity = torch.tensor([17280.0], dtype=torch.float64, requires_grad=True)
model, state, drivers = build_model(network)
drivers = {**drivers, "wsimod.River.velocity": velocity}
for k in range(3):
    state = model.step(state, drivers, dt=86400.0, t=k * 86400.0)
state["wsimod.storage"][0, 0].backward()
print(f"{velocity.grad.item():.3e}")
```

```text
-6.380e-06
```

WSIMOD's branches (`if x > FLOAT_ACCURACY`, `min`, `max`, the distribute loops) are decided on
detached values per instance, so the gradient is that of the branch each instance took; it jumps
where a branch changes.

## Running over the dates

`wsimod_run(model, state, drivers, days=...)` is WSIMOD's loop: it steps the same `WsimodDay`
closure over the configuration's dates (or the first `days` of them, `1 <= days <= len(dates)`)
and records each day as WSIMOD's `Model.run(record_all=True)` does. `state` and `drivers`
default to the configured ones and may carry batch dimensions. The run is under
`torch.no_grad()` unless a state or driver tensor requires grad (a long run with gradients
builds a large graph, so fit in windows). Stepping with `model.step` and running with
`wsimod_run` give identical values.

```python
from noodl.apps.wsimod import wsimod_run

model, state, drivers = build_model(network)
diagnostics = {}
results = wsimod_run(model, state, drivers, days=5, diagnostics=diagnostics)
print(results.dates[:2], results.arc_names, tuple(results.arc_flow.shape))
print(results.tank_names, [round(x, 4) for x in results.tank_storage[:, 0].tolist()])
balance = results.mass_balance
print(sorted(balance.terms))
print(float(balance.residual.abs().max()) < 1e-12)
print(sorted(diagnostics["events"])[:3], tuple(diagnostics["events"]["demand_deficit"].shape))
print(round(float(balance.inflow[0, 0]), 4), round(float(balance.storage_change[0, 0]), 4))
```

```text
['2000-01-01', '2000-01-02'] ['inflow_to_river', 'river_to_outlet'] (5, 2)
[('river', 'tank')] [0.1389, 0.1275, 0.1101, 0.1043, 0.2316]
['Catchment.flow', 'Catchment.unrouted', 'Waste.total_in', 'dropped']
True
['decay_arc_order_unknown', 'demand_deficit', 'demand_not_able_to_push'] (5, 3)
120.0 0.1389
```

**Results.** `WsimodResults` holds what WSIMOD records, with the batch dimensions in front and
the days next:

- `dates` (as written in the configuration), `arc_names`, `arc_flow` `(..., T, n_arcs)` (each
  arc's `vqip_out` volume) and `arc_quality` `(..., T, n_arcs, K)`;
- `tank_names`, `tank_storage` `(..., T, n_tanks)` and `tank_quality`: the attributes WSIMOD's
  `record_all` reports (`Tank`, `QueueTank` and `ResidenceTank`), nodes in configuration order and
  each node's attributes alphabetically;
- `surface_names`, `surface` (`storage`, `evaporation`, `precipitation`, `percolation`,
  `subsurface_r`, `surface_r` and `tank_recharge`, each `(..., T, n_surfaces)`) and
  `surface_quality`: the surfaces of the nodes under the type key `"Land"`, as WSIMOD records
  them. A field WSIMOD does not record for an impervious surface is NaN;
- `mass_balance`, below.

The records are taken at the state the day reached, before `end_timestep`, as WSIMOD records
them. Nothing is converted: volumes are m³, flows m³/d, additive pollutants kg.

**Diagnostics.** A dict passed as `diagnostics` receives:

- `"maxiter"`: for every distribute loop `(node, method, call index)` a `(..., T)` bool, true on
  the days the loop ended at `MAXITER` (WSIMOD prints "Maxiter reached in <node> at <date>");
- `"events"`: for every name in `EVENT_NAMES` a `(..., T, n_nodes)` bool (`(..., T, n_arcs)`
  for `"decay_arc_order_unknown"`, which is recorded at an arc), WSIMOD's warning
  prints as recorded facts: `"storage_unable_to_push"`, `"demand_deficit"` ("demand deficit of
  ... at <node> on <date>"), `"reservoir_spill"`, `"wwtw_couldnt_push"`,
  `"service_reservoirs_not_filled"`, `"sludge_nowhere"`, `"river_cant_push"`,
  `"ponded_water_cant_reenter"` (never fires: the forced push before WSIMOD's print returns
  empty), `"groundwater_couldnt_push_all"`, `"miscalculated_tank_storage"`,
  `"environmental_not_able_to_push"`, `"demand_not_able_to_push"`, `"leakage_not_going_to_gw"`,
  `"more_treated_than_input"`, `"excess_treated_water"` and `"more_backflow_than_vqip"`, and
  noodl's own `"zero_priority"` (a distribute round with priority 0 where WSIMOD divides by
  zero), `"wtw_liquor_zero_denominator"`, `"inaccurate_evaporation"` and
  `"wtw_negative_solids"` (an additive solids column of a treatment works is negative) and
  `"decay_arc_order_unknown"` (an end of day of a `DecayArc` found due-and-waiting rows in both
  directions, so its decayed term's order of addition may differ from WSIMOD's in the last
  bit, D21) and `"river_nitrogen_lost"` (a River's mineralisation with no dissolved nitrogen
  destroyed nitrogen, X15). The arc-keyed event, `"decay_arc_order_unknown"`, is keyed by arc,
  `(..., T, n_arcs)`, and `WsimodDay.event_columns(condition)` names the columns (X48);
- `"decisions"`, only when `diagnostics["decisions"] = True` on entry: a list over the days of
  that day's branch records `Decision(site, call, round, active, outcome, margin)` in call
  order, per instance. The gradient tests and the parity admission rule use it;
- `"mass_balance"`, the same object as `results.mass_balance`.

**The mass balance.** WSIMOD's own system check runs but cannot report: it compares the
system's inflow with itself (`model.py:1043-1058`). noodl computes what it meant to.
Per day, `inflow` is the sum of the nodes' `mass_balance_in` terms (catchment flows,
precipitation, `FWTW` deficits, `UnlimitedDistribution` supply, demand loads), `outflow` that of
their `mass_balance_out` terms (Waste, evaporation, demand consumption and backup, unrouted
catchment water, unpushed sludge, treatment solids) plus the water WSIMOD loses without a
trace, and `storage_change` the change of every carried store; `residual = inflow - outflow -
storage_change`. Each is `(..., T, 1 + K_additive)`: the volume, then the additive pollutants'
masses. `terms` lists every term by `"<Class>.<term>"` (arcs' flows cancel and are not listed).
The lost water is `terms["dropped"]`: the pushes below `FLOAT_ACCURACY` that a `QueueArc` or
`AltQueueArc` discards, the waiting `QueueArc` requests removed on update, the `Land` percolation
push-back, the `WWTW` storm-tank rejection and `make_discharge` push-back and a `Sewer`'s ponded
water below `FLOAT_ACCURACY`; and `terms["WWTW.unprocessed_input"]`, the water pushed into a
`WWTW` after its `calculate_discharge` (a custom orchestration), which WSIMOD empties at
`end_timestep` without ever treating it. A push below `FLOAT_ACCURACY` into a `QueueTank` stays
in the tank's storage and never enters its queue, so it is storage and not a drop. Per-day
values (a `Demand`'s totals, a `Land`'s running flows) are terms, not storage.

**The balance window.** A day's window runs from the state it loads to the state it leaves
after `end_timestep`, the state the next day loads, so consecutive windows meet exactly and
nothing happens between two days' balances. A term recorded during `end_timestep` counts in
its day. The day's record (`WsimodDay.last_record`, a `DayRecord`) holds `storage_start`, the
water held when the day began; `storage_total`, the water held at the record point, before
`end_timestep`, where WSIMOD records; and `storage_end`, the water held after `end_timestep`,
which the balance uses. `storage_end` is set when the day completes: at the yield point of
`WsimodDay.run_days`, which comes before `end_timestep`, it is still `None`.

The residual closes to rounding on every day, with one known gap, and one rule for it: a
`Land`'s outflow in WSIMOD's balance is IHACRES's evaporation, and when the soil cannot supply
it `evaporate` removes less (the `"inaccurate_evaporation"` event). On every day the residual
equals that gap, `E_actual - E_ihacres` summed over the lands, which WSIMOD's own node check
prints as "mass balance error for volume of ...". The gap is 0 on days without a clamp, and a
shortfall below `FLOAT_ACCURACY` fires no event but is in the residual all the same.

## State and drivers

The state keys, each with a leading batch shape and labelled rows:

| Key | Rows | Holds |
|---|---|---|
| `"wsimod.storage"` | `(node, tank)`, `(node, "surface:<name>")` | every tank's storage, `(..., n, 1+K)`: nodes in configuration order, each node's tanks in alphabetical order, a land's surfaces after its tanks |
| `"wsimod.active_storage"` | `(node, tank)` | each `QueueTank`'s water that can be pulled |
| `"wsimod.travel_queue"`, `"wsimod.travel_queue_present"` | `(arc, i)` | the travel registers of the `QueueTank`s and `AltQueueArc`s, one row per travel time, and their key sets |
| `"wsimod.arc_queue"`, `"wsimod.arc_queue_since"` | `(arc, direction, i, j)`, `(arc, direction)` | every `QueueArc`'s waiting requests and the days since its last delivery |
| `"wsimod.liquor"`, `"wsimod.previous_pulled"`, `"wsimod.treated"` | node | the treatment works' liquor, an `FWTW`'s previous day's pull, a `WWTW`'s treated water |
| `"wsimod.river_temperature_lag"`, `"wsimod.river_phosphorus_lag"` | `(river, i)` | the two registers of every River with biochemistry, `max_temp_lag + 1` and `max_phosphorus_lag + 1` slots (degC, kg) |
| `"wsimod.total_decayed"` | `(node, "tank.internal_arc")`, `(node, "surfaces.<name>")` | each decaying `QueueGroundwater`'s queue decay that its `storage` loses at the next end of day, and each surface's decayed mass of the last end of day, which the next `Surface.run` converts (surface rows only when `nitrite` is a pollutant; in the state layout's order) |

A store is a VQIP: the volume, then one column per pollutant. In `initial_state` a number is a
volume with no pollutants and a mapping a VQIP by name, `{"volume": 5.0, "phosphate": 0.1}`; a
`QueueTank` given a storage starts with all of it active, as in WSIMOD.

Parameters keyed by pollutant (`pollutant_load`) have one column per pollutant, absent ones 0;
time-area diagrams (`"wsimod.Sewer.pipe_timearea"`, `"wsimod.QueueGroundwater.timearea"`) one
column per travel time. Their keys are structure and stay as configured; the fractions are
drivers. Quantities WSIMOD derives in a constructor (a river's area, a surface's capacity) are
derived from the day's parameters, never given. A driver left out of the dictionary takes its
configured value.

## Quantity only or with pollutants

`build_model(network, pollutants=False)` runs on the volumes alone (`K = 0`). The parameters such
a run does not read — surface deposition loads, a demand's `pollutant_load`, the treatment works'
pollutant entries, every `decays` value (nodes, surfaces and user decay arcs; the
`DecayQueueTank` structure stays), the River's biochemistry parameters, a catchment's pollutant series — are listed by name in
`model.ignored_by_quantity_only`. What WSIMOD accepts and never reads is `model.unread`
(`WsimodNetwork.unread`): a `Groundwater`'s or `QueueGroundwater`'s configured `data_input_dict`
or `filename`, and a `filename` override on a class that reads no forcing (WSIMOD reads the file
and ignores it; a `River` reads it when `nitrate` is among the pollutants). With `pollutants=True` (the default) the same configuration is
refused if a process that is not ported yet would change the pollutants (deposition and the
land's nutrient pools, 3c): until then, such configurations need `pollutants=False` or a
neutralised copy, below. The soil temperature, the treatment, the decays and the River's
biochemistry are ported; their parameters are only listed as ignored in a quantity-only run.

### The demo configurations

Each demo (`tests/data/wsimod/quickstart`, `oxford`) has two configurations: `config.yml`, as
WSIMOD saved it, and `config_no_deposition.yml`, the same configuration plus an `overrides`
section that sets every surface's `pollutant_load` to 0 (written by
`scripts/regenerate_wsimod_fixtures.py`). `config.yml` runs with `pollutants=False`; the second
runs with every pollutant, every 3b process the demo contains live: on `oxford` the `FWTW`'s and
`WWTW`'s treatment on 11 additive pollutants and the rural surface's soil temperature; on
`quickstart` the soil temperature and the nitrite carry-over, which adds zero with empty
`decays`. Neither demo has a non-empty `decays`, a `DecayQueueTank`, a user decay arc or a
`River`: the purpose-built cases cover them. What still needs neutralising for 3c is each
surface's `pollutant_load`, in both demos.

## Refusals

Everything that cannot run is refused by name when the model is built, where WSIMOD would fail
mid-run, hang or silently do something else:

- out-of-scope classes and keys (by the reader);
- River biochemistry that WSIMOD cannot run (above); `decays` on an arc class other than
  `DecayArc` and `DecayArcAlt`, `decays: None` on those two, an arc override of `decays`, and a
  `parent` on an arc; a decaying `Groundwater` or `QueueGroundwater` without a `filename` override's
  temperature; a `number_of_timesteps` override on a `QueueGroundwater` without `decays`; a
  `decays` key that is not
  a configured pollutant, an entry without exactly `constant` and `exponent`, a non-numeric
  value, an `exponent` of 0 or below, and an override of `decays` on a node built without;
- a `total_porosity` not above `field_capacity`, a residence time of 0, `leakage` outside
  `[0, 1)`, preferences that are negative or all zero, travel times that are not whole days;
- a cycle that makes WSIMOD's checks recurse into each other, which makes WSIMOD retry at
  every level and never finish: every cycle of pushes, a cycle of pulls only when it lies
  upstream of a node whose orchestrated step pulls (`create_demand`, `treat_water`,
  `make_abstractions`), and a base `WTW` in a pull cycle (its pull handlers keep recursing). The
  rule over-approximates: it may refuse a model WSIMOD runs, never the reverse. WSIMOD's own
  `Land -> Sewer -> Land` ponding is not a cycle: no check enters it;
- river arcs without a `Waste`, an arc into a `Catchment`, an orchestration slot whose nodes
  lack the method;
- missing forcing, naming the node, the variable and the first missing date (a `Land` needs
  `temperature` even in a quantity-only run, as WSIMOD reads it).

`model.step` needs `dt=86400` and a `t` that is a whole number of days inside the dates.
`model.check(state, drivers, dt=86400, probe=True)` cannot step this model: the probe steps
without `t`, which the closure refuses, so it reports `probe-failed` ("model.step needs t=",
W82). `model.step(state, drivers, dt=86400, t=0)` is the check. A `leakage` batch that mixes
zero and positive values on one `Distribution` is refused when it is bound: WSIMOD decorates the
node's handlers only for a positive configured leakage.

## Where WSIMOD is silent or wrong

Where WSIMOD's source is defective, noodl does the evident intent or refuses by name; where
WSIMOD divides by zero or loses water, noodl records it. Each row is pinned by a test.

| WSIMOD | noodl | Test |
|---|---|---|
| `Land.push_set_sewer` lacks `self.surface`, and `Land` has no `"Sewer"` check | evident intent: ponded sewer water is forced into the first impervious surface | `test_sewer_ponding_into_land_is_forced_into_the_first_impervious_surface`; `test_case_runs_where_wsimod_crashes_land_sewer_ponding` |
| `EnfieldFoulSewer.make_discharge` (`storm_exchange` undefined) | refused by name | `test_refusals_name_the_key[mapping7-EnfieldFoulSewer]` |
| `NonResidentialDemand.get_demand` recurses without end | evident intent: its constant demand, as a house | `test_non_residential_demand_is_its_constant_demand_as_house` |
| `GrowingSurface.residue` is unwired | sub-project 3c; refused as a class | `test_refusals_name_the_key[mapping8-3c]` |
| the system mass balance compares `sys_in` with itself | a real balance (above) | `test_the_mass_balance_closes_by_hand`; every parity test |
| `Distribution.apply_overrides` decorates the handlers again | once | `test_an_overridden_leakage_is_applied_once` |
| `UnlimitedDistribution` loses its leakage | refused | `test_leakage_refusals` |
| overrides update the shared default dicts | per node | `test_an_override_of_pollutant_load_reaches_only_its_node` |
| river arcs but no `Waste` | refused | `test_structural_refusals[river-without-waste]` |
| an arc into a `Catchment` | refused | `test_structural_refusals[arc-into-catchment]` |
| WTW solids are the unclamped remainder, negative where `constant * factor + liquor_multiplier > 1` | transcribed; event `"wtw_negative_solids"` | `test_cold_influent_makes_negative_solids_and_says_so`; case `wtw_treatment` |
| the WTW liquor blend divides by zero | the previous value is kept, event `"wtw_liquor_zero_denominator"` | `test_zero_input_on_day_one_keeps_the_previous_liquor_value`; `test_case_runs_where_wsimod_crashes_wtw_zero_input` |
| a partial zero priority in a distribute round divides by zero, depending on the state (W43) | nothing is distributed in that round, event `"zero_priority"` | `test_a_zero_priority_with_room_is_an_event_not_a_division_by_zero` |
| `QueueArc` drops a push below `FLOAT_ACCURACY` | **reproduced** (a tolerance-scale loss that real models' delivery depends on), recorded as dropped | `test_queue_arc_delivery_matches_wsimod_on_irregular_pushes`; `test_the_tiny_push_is_recorded_as_dropped` |
| `QueueArc` zero-volume pull with additive keys | the rescale is skipped, as `Arc` does | `test_a_zero_volume_pull_with_pollutants_skips_the_rescale` |
| `WWTW.make_discharge` rejection at or below `FLOAT_ACCURACY` | a tolerance, recorded as dropped | `test_a_tiny_discharge_rejection_vanishes_and_is_recorded_as_dropped` |
| `decays: {}` on the `Storage` family builds a `DecayTank`, which reads the temperature and decays nothing | ported; the temperature forcing is required | `test_storage_with_decays_reads_its_temperature` |
| WSIMOD's `record_all` skips a `DecayTank` | the tank drops out of `tank_*` (X40); its storage stays in the state | `test_a_decaying_storage_decays_at_the_end_of_the_day_and_closes_the_balance` |
| `Groundwater` stores its `data_input_dict` before `Node.__init__` resets it (`storage.py:154`, `:202-203`, `nodes.py:26`): a decaying one raises `TypeError` at its first end of day | the configured data is lost (`model.unread`); the temperature comes from a `filename` override, and a decaying `Groundwater` without one is refused by name (X11) | `test_case_runs_where_wsimod_crashes_decay_groundwater`; `test_case_matches_wsimod[decay_groundwater-quantity]` and `[decay_groundwater-pollutants]` |
| the same in `QueueGroundwater` (`storage.py:252`, `:291-292`): a decaying one raises `TypeError` at its first push or end of day | the same (X11) | `test_case_runs_where_wsimod_crashes_decay_queue_groundwater`; `test_case_matches_wsimod[decay_queue_groundwater-quantity]` and `[decay_queue_groundwater-pollutants]` |
| `DecayQueueTank`'s default `number_of_timesteps=1` (`tanks.py:648`) contradicts its docstring and `QueueTank`'s 0 | reproduced: `decays` delays a `QueueGroundwater` by one day; the `number_of_timesteps` override sets it (X12, X28) | `test_decaying_delays_a_queue_groundwater_by_one_day`; `test_number_of_timesteps_override`; `test_case_matches_wsimod[decay_queue_groundwater_n0-quantity]` and `[decay_queue_groundwater_n0-pollutants]` |
| `River.calculate_discharge` assigns `bio_in`/`bio_out` (`storage.py:923-924`): a second call a day drops the first call's terms from WSIMOD's node check, which prints an error | the processes run twice as in WSIMOD; noodl's terms add both calls and its balance closes (X16, X32) | `test_case_matches_wsimod[river_double_discharge-pollutants]` (WSIMOD prints on every day, the named exception to the print check, D17; noodl's balance closes within rounding, 2e-16 of scale); `test_river_double_discharge_runs_the_biochemistry_twice_a_day` |
| River mineralisation with `din = 0` (`storage.py:866-873`) destroys the mineralised nitrogen | reproduced (X14, X15); the loss is in `bio_out` and the event `"river_nitrogen_lost"` says so | `test_mineralisation_without_din_destroys_nitrogen_and_says_so`; `test_case_matches_wsimod[river_nutrients-pollutants]` (the DIN-free stretch) |
| the River's "365-day" phosphorus mean adds 366 entries and divides by 365; every window divides by its full length from day 1 (cold start) | reproduced (X14) | `test_the_registers_reproduce_pythons_sums_from_a_cold_start`; `test_case_matches_wsimod[river_nutrients-pollutants]` |
| `totalphosfcn` compares the River's phosphorus mass (kg) with the concentrations `limpppar` and `hsatTP` (kg/m3) | reproduced and documented: the evident fix has no single form (X14) | `test_river_nutrients_reaches_every_branch` (`-m slow`: both `totalphosfcn` outcomes of the low-P river, in the first 400 days and over 1100) |
| no check of a decay's or a treatment's `exponent` (`core.py:513-515`, `wtw.py:191-193`): `0 ** negative` raises `ZeroDivisionError`, a negative base gives a complex number | an `exponent` of 0 or below is refused by name | `test_read_decays_refusals[decays4-exponent]` and `[decays5-exponent]` (an `exponent` of 0 and of -1), among the ten refusals of that test; `test_wtw_treatment_refusals` (9 refusals) |
| `decays` may name `volume` (`core.py:510-519`): the water itself would decay | refused by name | `test_read_decays_refusals[decays0-volume]` |
| the default `decays={}` of a surface is one dict shared by every surface built without one, so an override reaches all of them | per surface | `test_a_decays_override_reaches_only_its_surface` |
| `Surface.run` converts `total_decayed` at every call: a second `Land.run` on one day converts the same mass twice | transcribed (X16) | `test_case_matches_wsimod[land_run_twice-quantity]` and `[land_run_twice-pollutants]` (and the carry-over's own cases, `surface_decay_nitrification`) |
| `total_porosity` not above `field_capacity` or `wilting_point` | refused | `test_porosity_not_above_field_capacity_is_refused` |

WSIMOD's `land_sewer_ponding` case does not crash with fixed parameters (W73): the sewer's own
check clips every push. The crash test lowers the sewer's capacity between `Land.run` and
`make_discharge` in both engines, asserts WSIMOD's exception, and that noodl ponds on some days,
returns the water and balances. WSIMOD's `NonResidentialDemand` raises `RecursionError`
(`test_case_runs_where_wsimod_crashes_nonresidential_demand`).

## Differentiability

- **Learnable.** Every numeric parameter (`"wsimod.<Class>.<parameter>"`), the arcs' capacities
  and preferences, the forcing series and the initial stores may require grad. The gradient
  tests check 49 of them. From 3a, 22: arc capacity and preference, groundwater and land
  residence times, impervious and pervious surface coefficients, `ihacres_p`, river `mrf` and
  `velocity`, a sewer's time-area fractions, `Distribution.leakage`, the `WWTW` and `FWTW`
  throughput capacities, `ResidentialDemand.per_capita`, a demand's `pollutant_load`, a
  catchment's flow and concentration, precipitation and an initial storage. From 3b, 27: the
  decay `constant` and `exponent` of a `Storage`, a `RiverReservoir`, a `QueueGroundwater`
  (`DecayQueueTank`) and a `DecayArc`; the temperature forcing through a decay and through the
  treatment; the `WWTW`'s `process_parameters` (`constant`, `exponent`) and `liquor_multiplier`;
  the three soil-temperature weights; and, through the River's biochemistry, `denpar_w`,
  `halfsatINwater`, `prodNpar`, `muptNpar`, `T_wdays` and the air temperature, each read through
  the nitrate and the organic-nitrogen outputs.
- **Structural.** The topology, the classes, the pollutant lists, the orchestration, the travel
  times (whole days: they size the queues) and whether a `Distribution` has leakage at all
  (WSIMOD decorates its handlers only then) are fixed when the model is built.
- **Detached.** Every branch (`if x > FLOAT_ACCURACY`, `min`, `max`, the loop exits, the tie
  rules) is decided on detached values, per instance. The values stay in the graph, so the
  gradient is that of the branch the instance took.
- **Jumps and kinks.** The gradient is not defined where a branch changes: at a volume that
  crosses `FLOAT_ACCURACY`, at a capacity that just clips a push, at the day a tank starts to
  spill, at a distribute loop that ends one round earlier. A decision's margin (`lhs - rhs`, in
  `diagnostics["decisions"]`) says how near a run is to such a point. A residence time near
  one day leaves a recession tail of about 1e-9 that a central difference sees as a kink.

Against central differences at a relative step of 1e-6, all 49 gradients agree within 1.5e-7
relative (the tolerance is 1e-6), and the `"decisions"` outcomes of the two perturbed runs are
identical. The worst are the River's (`T_wdays [nitrate]` 1.5e-7, `halfsatINwater [nitrate]`
1.0e-7 and `denpar_w [org-nitrogen]` 6.8e-8, small gradients through 120 days of two rivers),
then `Sewer.pipe_timearea` 5.9e-8; the decay entries are within 9e-10, the treatment ones within
1.2e-10 and the soil-temperature weights within 1.7e-8. The `window_sum` of the River's
registers carries the plain sum's gradient (X55). The tests move a point off a kink where the default one lies on it (the preference
1.0 to 1.37, the surface residence time 1 to 3), and say so.

## Batches

Any driver or state key may carry leading batch dimensions. Every instance follows its own
WSIMOD path, so one instance may clip an arc where another does not, run its distribute loop to
`MAXITER` where another stops, or pond a sewer: each equals its own single run bit for bit (every
case, both modes, in `tests/apps/wsimod/test_build.py`; `batch_divergence` is a case whose
instances take different branches). What cannot differ between instances is structure: the
sequence of calls (fixed by a tracing day at build time and asserted on every later day), the
travel times, the pollutant lists and whether a `Distribution` has leakage.

## Graph size and fitting in windows

Autograd is on only when a state or driver tensor requires grad (W9); otherwise the run is under
`torch.no_grad()`. The graph is large because the day is thousands of small operations. The
benchmark (`benchmarks/wsimod_run.py`) times 60 days with `wsimod.Arc.capacity` requiring grad
and counts the autograd nodes behind the last day's storage:

| Demo | Batch | `no_grad`, s/day | grad, s/day | Autograd nodes (60 days) |
|---|---|---|---|---|
| quickstart (quantity only) | 1 | 0.0468 | 0.0888 | 240,999 |
| quickstart | 10 | 0.0706 | 0.0922 | 240,999 |
| quickstart | 100 | 0.0711 | 0.0986 | 240,999 |
| oxford (quantity only) | 1 | 0.2776 | 0.3720 | 936,765 |
| oxford | 10 | 0.2751 | 0.3759 | 936,765 |
| oxford | 100 | 0.2810 | 0.3948 | 936,765 |
| oxford, `config_no_deposition.yml`, pollutants | 1 | 0.2843 | 0.3892 | 942,436 |
| oxford, pollutants | 10 | 0.3012 | 0.4161 | 942,436 |
| oxford, pollutants | 100 | 0.3317 | 0.4400 | 942,436 |

**These times were measured on a shared machine** (other jobs were running, every figure is 2.6
to 3.9 times its unloaded value). The unloaded medians, measured before 3b on the same machine
(quantity only), were, no grad / grad in s/day: quickstart 0.0179 / 0.0255 (batch 1), 0.0187 /
0.0283 (10), 0.0188 / 0.0289 (100); oxford 0.0709 / 0.1070, 0.0723 / 0.1115, 0.0725 / 0.1179.
The pollutant rows were not measured unloaded. What the shared run does show, because the rows
ran back to back, is the cost of 3b's processes on `oxford`: the pollutant run takes 1.02 to
1.18 times the quantity-only run (no grad) and 1.05 to 1.12 times (grad), and its graph is 0.6 %
larger (942,436 nodes against 936,765). Oxford has no `River`; the `river_nutrients` case (two rivers with the biochemistry) runs 400
days in about 3 s unloaded, so the biochemistry is not free, and a model with many rivers costs
more.

That is about 4,000 nodes a day for `quickstart` and 15,600 for `oxford`, independent of the
batch, which only widens the tensors. Gradients through the full 1,456 days of `oxford` would
build about 23 million nodes, so fit gradients in windows: run some days, take the gradient,
`detach` the state and carry on. noodl does not checkpoint in this version. The times are from
one measurement on the development machine; no budget is asserted. The per-instance Python
arithmetic that keeps the result bitwise equal to WSIMOD's (`pow_half`, `exp_python` and
`pow_python`: `** 0.5`, `exp` and `**` per instance, in a loop over the batch) costs time
proportional to the batch; on these demos it is small next to the many small tensor operations
of the day, and in the shared run above a batch of 100 costs 1 % to 6 % more than a batch of 1 on
`oxford` (no grad, grad), 17 % and 13 % more with pollutants. On `quickstart` the batch-1 time
is the outlier (0.0468 against 0.0706 at batch 10, no grad: noise from the other jobs; the
unloaded medians differ by 5 %).

## Limitations

- **Sub-project 3c.** The processes that change a pollutant (deposition, the land's nutrient pools, `GrowingSurface`, the garden and
  irrigation surfaces) are refused unless neutral. The soil temperature, the treatment works'
  pollutant treatment and the decay of the storage family's tanks (a `QueueGroundwater`'s
  `DecayQueueTank` included) and of a land's surfaces and a River's biochemistry are ported. Until 3c, a configuration with surface deposition or nutrient pools needs `pollutants=False` or neutralised loads.
- **Input and output.** A configuration is read, not written. `extensions` and
  `unified_data_file` are refused.
- **Speed (W8).** The day is a transcription of many small operations, with no compilation:
  `oxford` takes about 0.07 s a day without gradients (unloaded; the pollutant run costs 2 to 18 % more, measured on a shared machine).
- **The closure is not re-entrant.** One `WsimodDay` holds one engine that every step reuses, so
  two steps of one model must not run at once (threads, or a step that calls another).
- **`Model.to`.** `model.to(dtype=torch.float64)` (or a CPU device) works and changes nothing
  (tested). Another dtype or a device move is refused by name (`TypeError`): the port is float64
  on the CPU, as bitwise equality with WSIMOD needs.
- **River ties.** WSIMOD breaks a tie in the river order by set-hash order, which varies with
  `PYTHONHASHSEED`; noodl takes the configuration order (W13, W64), so a model whose river ties
  matter may differ from a WSIMOD run in the last bit. Neither demo has a `River`.
- **A gapped time-area** (`QueueGroundwater`) may differ from WSIMOD by an ulp: WSIMOD walks the
  queue in dict insertion order, noodl in ascending time (W54).
- **The recursion rule over-approximates** and may refuse a model WSIMOD runs.

## Verification

Against live WSIMOD 0.8.1 (`pytest.importorskip("wsimod")`; WSIMOD is never imported by
`src/`), in four layers. Every comparison is bitwise where the arithmetic is `+ - * /`, `min`,
`max` and comparisons: the ceiling is `1e-12 * scale` per compared series (the largest magnitude
over the run), but the tests assert the class's own bound, which is 0, so a last-bit difference
has to be debugged, not admitted.

- **Per-node replay** (`tests/verification/test_wsimod_replay.py` and the class tests under
  `tests/apps/wsimod`):
  WSIMOD runs a case with every call into the chosen nodes captured; noodl replays each node
  alone on those calls and compares every arc call it makes, every reply and every store, in both
  modes. On the two demos every node class (Land, Sewer, Groundwater, Node, Waste, WWTW, FWTW,
  Reservoir, ResidentialDemand, Catchment) has worst difference exactly 0, with both IHACRES
  outcomes sampled.
- **Whole model** (`tests/verification/test_wsimod_parity.py`): WSIMOD's two demos, `quickstart`
  and `oxford`, each in two modes, against WSIMOD's own `Model.run`, and 49 purpose-built cases
  (every feature the demos skip: queue arcs, a leaky `Distribution` and the reservoir spill,
  sewers with time-areas and foul sewers, storm tanks, an `FWTW` deficit, an evaporation clamp,
  a porous `QueueGroundwater`, rivers and the River biochemistry, decays, treatment, a
  `float_accuracy` override and others) in two modes.
  Every arc's flow and pollutants, every recorded tank and every surface field, on every day. The
  worst difference is exactly 0 for every node of both demos and for all 49 cases in both modes,
  the River biochemistry cases (`river_nutrients`, `river_double_discharge`) included, and no
  admission was needed. WSIMOD's printed warnings are compared with noodl's events day by
  day (dated prints) or per day and kind (undated ones); WSIMOD's own "mass balance error" lines
  equal noodl's residual plus the dropped water, with one named exception (W90/W91, X32):
  `river_double_discharge`, where WSIMOD's River balance misses the first of two daily
  `calculate_discharge` calls and prints on every day while noodl's balance closes within
  rounding (2e-16 of scale); the harness requires those pollutant prints on every day and
  compares them with nothing (D17). The system mass
  balance closes within `1e-12 * scale` every day, and every day the residual equals the
  evaporation clamp's gap (0 without a clamp). A quantity-only run's volumes equal the pollutant
  run's exactly (`torch.equal`). The River biochemistry cases: `river_nutrients` runs its first
  400 days by default and all 1100 under `-m slow`, when the decisions are recorded and every
  branch is asserted (X39, D24); `river_double_discharge` runs its 40 days by default.
- **Crashes** (`test_case_runs_where_wsimod_crashes_*`): five cases where WSIMOD raises. Three
  run in noodl, finite and balanced (`land_sewer_ponding` driven by a capacity, `wtw_zero_input`,
  `nonresidential_demand`). For the other two, a decaying `Groundwater` and `QueueGroundwater`
  without a `filename` override (`decay_groundwater`, `decay_queue_groundwater`), noodl refuses by
  name where WSIMOD raises `TypeError`; with the override both run and agree bitwise.
- **Gradients** (`tests/verification/test_wsimod_gradients.py`): 49 entries (22 from 3a, 27 from
  3b), central differences at a relative step of 1e-6, the two perturbed runs' decision outcomes
  identical. The worst relative disagreement is 1.5e-7 (`River.T_wdays [nitrate]`; then
  `River.halfsatINwater [nitrate]` 1.0e-7, `River.denpar_w [org-nitrogen]` 6.8e-8, `Sewer.pipe_timearea`
  5.9e-8 and `River.mrf` 3.6e-8) against a tolerance of 1e-6. Nineteen of the 49 are `-m slow`:
  the six quickstart ones (200 days), two soil weights and eleven of the River's (120 days of two
  rivers, 10 s and more each); each 3b family keeps its cheapest entry in the default selection,
  and a mutation of `pow_python` or of `window_sum` fails a default entry. The 49 take 229 s
  with `-n 4` on a shared machine.

| Run | Days | Modes | Worst difference | Default | Time (`-n 4`) |
| --- | --- | --- | --- | --- | --- |
| quickstart | 1456 | quantity-only, pollutant mode | 0 | yes | 41 s, 38 s |
| oxford | 365 | quantity-only, pollutant mode | 0 | yes | 36 s, 37 s |
| oxford | 1456 | quantity-only, pollutant mode | 0 | `-m slow` | 142 s, 146 s |
| 47 cases (the 3a ones and the decay, treatment, soil-temperature and surface cases of 3b) | 3 to 40 | quantity-only, all pollutants | 0 | yes | at most 2 s a case |
| `river_nutrients` | 400 | quantity-only, all pollutants | 0 | yes | 3 s a mode |
| `river_nutrients` | 1100, with decisions | all pollutants | 0 | `-m slow` | about 10 s a mode unloaded (22 s, 27 s and 31 s for the branch assertions measured on a shared machine) |
| `river_double_discharge` | 40 | quantity-only, all pollutants | 0 (WSIMOD's own balance prints on every day; noodl's closes within 2e-16 of scale) | yes | not timed separately |
| 49 gradients | 1 to 200 | quantity-only values, pollutant runs | 1.5e-7 relative | 30 of 49 (19 `-m slow`) | 229 s in all |

The quantity-only mode runs `config.yml` with `pollutants=False` and compares the volumes; the
pollutant mode runs `config_no_deposition.yml` (an `overrides` section that sets the surfaces'
`pollutant_load` to zero) with every pollutant and compares all of them. Neither demo has a `River` node, so
WSIMOD's river-order ties (set-hash order of the names, which varies with `PYTHONHASHSEED`)
cannot arise in the demos, and neither has a `QueueGroundwater`; the river and groundwater
cases cover those classes, with the ties kept independent. WSIMOD's source files are not
modified; the committed forcing is WSIMOD's (BSD-3, `tests/data/wsimod/LICENSE-WSIMOD`).

**Suite composition.** The wsimod row of the test table (the default selection) takes
about 315 s with `-n 4` on an unloaded machine (5,798 tests, measured in Task 13), within the 10-minute
budget. At the end of 3b, with the machine shared and every item 2 to 3 times slower (the four
demo parity items 91 to 97 s each), the row took 675 s (11 min 14 s, 5,835 tests); earlier shared runs took 11 to 15 min. One case is cut
(X39): `river_nutrients` runs 400 of its 1100 days by default, which already turns its 366-slot
register over and holds every branch; its default items take about 3 s a mode (with the register
sums as CPython's own `sum`, X55; 8 s before), and the 1100-day run with the branch assertions
from the recorded decisions is `-m slow` (about 10 s a mode on an unloaded machine). `pytest -m slow` on the row also
adds the full `oxford` run, the full per-node replays and the 19 slow gradients.
