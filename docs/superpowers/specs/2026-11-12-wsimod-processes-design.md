# WSIMOD in-water processes in `apps.wsimod` (3b)

Status: designed 2026-11-12 under the user's standing instruction ("do not wait for me and make
sensible decisions"). Rulings X1–X6 are the brainstorming rulings, which stand in for the user's
approval. X7–X26 are made here (Rulings made in this spec). A Fable review checks both sets
before the plan is written (X6). This document covers sub-project 3b of 3 of the WSIMOD port
(W1). It builds on 3a, specified in `2026-11-11-wsimod-core-design.md` and merged as PR #41
(main `a2fb7d0`). Rulings W1–W95 still bind and are cited by number.

## Goal

3a carries pollutants without changing them. 3b ports every WSIMOD 0.8.1 process that changes
a pollutant inside a body of water:

- the temperature-dependent first-order decay (`DecayObj`) in tanks, storage nodes, surfaces,
  queue tanks and arcs;
- the WTW pollutant treatment;
- the river biochemistry;
- the soil-water temperature of the pervious surface.

The port is faithful, batched, branch-free and differentiable, on the 3a substrate (X2). It is
judged against live WSIMOD 0.8.1 in the same three ways as 3a (X3):

- node by node on replayed inputs;
- on both demos, with every 3b process they contain running live;
- on purpose-built cases for the processes the demos lack.

Authority: the installed WSIMOD 0.8.1 source, `.venv/Lib/site-packages/wsimod/`. Every
`file:line` below is relative to that directory. Where this spec and the source disagree, the
source wins and the spec is corrected (W2).

## Constraints

The constraints are 3a's:

- branch `wsimod-processes` from main `a2fb7d0`, with one PR and a Fable review before the
  merge (X6);
- physical names and no aliases;
- code, tests and docs change together;
- no wall-clock assertions;
- `ruff check .` passes;
- every transcribed behaviour cites its WSIMOD `file:line`;
- WSIMOD is imported only by tests;
- `torch.float64` on the CPU only (W94).

## Scope (X1)

### In: what 3b ports

| WSIMOD class or method (file:lines) | What 3b ports |
|---|---|
| `DecayObj` (`core/core.py:432-521`) | `__init__`'s choice of temperature source (`:464-470`), `make_decay` (`:474-492`), `generic_temperature_decay` (`:494-521`) |
| `DecayTank` (`nodes/tanks.py:403-457`) | `end_timestep_decay` (`:442-447`) and the `decays` override (`:430-440`). The tank is the storage family's when `decays` is a dict (`storage.py:53-68`), and every surface's (`land.py:303`) |
| `DecayQueueTank` (`tanks.py:645-696`) | all of it: the internal `DecayArcAlt` (`:663-669`), `_end_timestep` (`:689-696`) and the override (`:673-687`). It is a decaying `QueueGroundwater`'s tank (`storage.py:307-315`) |
| `DecayArc` (`arcs/arcs.py:668-728`) | as a user arc: decay on entry (`:686-708`) and at the end of the day (`:710-728`) |
| `DecayArcAlt` (`arcs.py:731-805`) | as a user arc and as `DecayQueueTank`'s internal arc: entry (`:752-778`) and `_end_timestep` (`:780-805`) |
| `Storage`, `Groundwater`, `QueueGroundwater`, `Reservoir`, `RiverReservoir`, `River` with `decays` (`storage.py:24`, `:53-68`, `:96-101`, `:300-315`) | the decaying tank, the `decays` override, and the temperature forcing they read |
| `River.biochemical_processes` (`storage.py:738-898`) with `update_depth` (`:723-725`), `get_din_pool` (`:727-736`), `calculate_discharge` (`:917-924`), the biochemistry parameters and lags (`:539-566`), their overrides (`:620-644`) and `bio_in`/`bio_out` (`:592-596`, `:951-955`) | all of it |
| `WTW.treat_current_input` (`nodes/wtw.py:163-214`) with any `process_parameters` and per-pollutant `liquor_multiplier` (`:72-85`, `:121-128`, `:130-153`) | the general process. 3a accepted only the neutral one |
| `Surface.run`'s nitrite carry-over (`nodes/land.py:427-439`) | the ammonia→nitrite→nitrate conversion of the decayed mass |
| `PerviousSurface.calculate_soil_temperature` (`land.py:1060-1079`) with the weights (`:807-812`, overrides `:843-884`) | any weights. 3a accepted only `(w_prev > 0, w_air = 0, w_deep = 0)` |

The only temperature process that `land.py` runs outside the growing surfaces is
`calculate_soil_temperature`. It sets the temperature of the water held in the pervious
surface's soil store, which then leaves with the subsurface and percolation flows. So it
belongs to the water and to 3b (X1). The other temperatures in `land.py` are inputs that 3a
already ports: rain at air temperature (`:685`, `:991`, `:1028`). The temperature dependence
factors (`:1813-1856`) belong to the growing surfaces and so to 3c.

### Out: what stays 3c, and what is never ported

- **3c:** `GrowingSurface`, `IrrigationSurface`, `GardenSurface` and `VariableAreaSurface`
  (`land.py:1082-2461`), with their own decay and temperature use (`:1276`, `:1813-1856`,
  `:1878`). Also `NutrientPool`, and surface deposition: `simple_deposition`,
  `atmospheric_deposition` and `precipitation_deposition` (`land.py:375-380`, `:505-581`).
  This includes a surface's non-zero `pollutant_load` and its `nhx-dry`-style data.
- **Never ported:**
  - `DecayObj.generic_temperature_decay_c` (`core.py:523-549`), which nothing calls;
  - `River.pull_check_fp` (`storage.py:938-949`), which is still refused as 3c (3a);
  - the unused River attributes `bulk_density` and `T_10_days`, `T_20_days`, `TP_365_days`
    (`:540`, `:548-550`). An override of `bulk_density` is accepted and listed as unread.

### Every 3a "3b" refusal, decided

`git grep -n 3b src/noodl/apps/wsimod` on `a2fb7d0` gives the list below. Each item is
either ported or still refused, with the reason.

| 3a refusal (noodl file:line) | 3b decision |
|---|---|
| `Storage.validate`: a `decays` dict (`storage.py:15`, `:279-288`), and `config._read_node`/`_refuse_decays` (`config.py:246-249`, `:378-379`) | **ported** (Decay). `decays: None` still builds a plain `Tank` |
| `_read_surface`: non-empty surface `decays` (`config.py:393-394`) | **ported** |
| `classes.py:278`: the `DecayTank` override `decays` | **ported**: the override merges, per element (Interaction › Overrides) |
| `classes.py:246`: the `Storage` override `decays` | **ported** when the node was built with a dict. With `decays: None` it is **still refused**, by name, as WSIMOD refuses it (`storage.py:96-100`, `ValueError`) |
| `classes.py:383-384`: `OUT_OF_SCOPE` `DecayArc`, `DecayArcAlt` | **ported** (user arcs) |
| `config._read_arc`: `decays` on an arc (`config.py:403-404`) | **ported** on `DecayArc` and `DecayArcAlt`. **Still refused** on every other arc class, which never reads the key (`Arc.__init__` stores any keyword unread, `arcs.py:52`); the message no longer names 3b |
| `classes.py:253`, `River.validate`/`calculate_discharge` (`storage.py:689`, `:794-796`, `:808-822`): River with `nitrate` | **ported** (River biochemistry). The remaining refusals are the new ones in Interaction › Validation |
| `WTW.validate`: non-neutral `process_parameters` and `liquor_multiplier` (`wtw.py:241-286`) | **ported** (WTW treatment). These refusals stay, because WSIMOD crashes there: additive pollutants without `temperature` (`wtw.py:191-193`, `KeyError`), and a configured volume constant other than the derived one (W70) |
| `Land.validate`: non-neutral soil-temperature weights (`land.py:507`, `:844`, `:904-916`) | **ported**. A weight sum of 0 is refused (`ZeroDivisionError`, `land.py:1072-1077`). The `temperature` pollutant that a `PerviousSurface` needs with pollutants on stays required (`:1070`, `KeyError`) |
| `Land.validate`: deposition data and non-zero `pollutant_load` (`land.py:859`, `:870`) | **still refused**: they are 3c. The message changes from "until sub-projects 3b and 3c" to "until sub-project 3c" |
| `tanks.py:277`: `DecayTank` with empty `decays` only | **ported** |

## Approach: transcribed and batched, as in 3a (X2)

Everything in 3a's Approach section holds. That covers masks, `MAXITER` rounds, safe
denominators, Python's tie rule for `min`/`max`, WSIMOD's order of reduction, 1e15 rather than
`inf`, and the static call structure. 3b adds four rules.

1. **Python's transcendentals, per instance.** Every power in these processes is evaluated with
   Python's `float.__pow__`, once per instance (`pow_python`, `land.py` in noodl; W53/W62
   pattern). The powers are:
   - the decay factor `exponent ** (T − 20)` (`core.py:513-515`);
   - the treatment factor `exponent ** (20 − T)` (`wtw.py:191-193`);
   - the river's `2 ** ((T − 20)/10)` (`storage.py:775`);
   - the macrophyte factor `(T/20) ** 0.3` (`:878`).

   `torch.pow` differs from them in the last bit (Differentiability). Every other operation is
   `+ − × ÷` in WSIMOD's operand order, so the replay bound is 0 (bitwise, W41/W47) for every
   3b class.
2. **Structural switches.** The following are fixed at build time:
   - `"nitrate" in POLLUTANTS` (biochemistry, `storage.py:919`);
   - `"nitrite" in POLLUTANTS` (the carry-over, `land.py:427`);
   - whether a node's `decays` is a dict;
   - each `decays` dict's key set;
   - which arcs are `Decay*` classes;
   - the river's lag lengths.

   Python control flow follows these switches and nothing else.
3. **Decay is a pure function of `(vqip, constants, exponents, T)`.** WSIMOD calls `make_decay`
   several times per element per day with the same temperature (`arcs.py:798-805`). noodl
   evaluates the factor `min(c · e^(T−20), 1)` once per element and day, and applies it at every
   call. The values are bitwise the same, because the inputs are identical.
4. **Decay's arithmetic is column-wise.** `generic_temperature_decay` walks the `decays` dict and
   touches only its keys (`core.py:510-519`). noodl applies the decay to the dense VQIP and
   restores the untouched columns with a structural `present` mask (`where(present, decayed,
   t)`). An absent key is therefore exactly WSIMOD's untouched value, and its driver entries have
   no effect (X9).

## The equations, transcribed

Every formula below is WSIMOD's, in its operation order. `T` is the temperature that the
process reads, `FA` is `FLOAT_ACCURACY`, and `REF = DECAY_REFERENCE_TEMPERATURE = 20`
(`constants.py:66`).

### Decay: `DecayObj` (`core.py:432-521`)

**Temperature source** (`:464-470`). Construction picks `data_input_object`:

- `self.parent` when the object has a `parent` attribute;
- otherwise `self.in_port`.

`make_decay` then reads `data_input_object.data_input_dict[("temperature",
data_input_object.t)]` (`:485-487`), that object's own forcing on the day its clock shows. The
choice per class:

| Object | `data_input_object` | Why |
|---|---|---|
| `DecayTank` in a storage-family node | the node | `parent=self` (`storage.py:67`, `:313`); `DecayTank` stores it before `DecayObj.__init__` (`tanks.py:418`, `:424`) |
| `DecayTank` that is a surface | the `Land` | `surface["parent"] = self` (`land.py:103`) |
| `DecayQueueTank`'s internal `DecayArcAlt` | the `QueueGroundwater` | the `parent=` keyword reaches `Arc.__init__`, which keeps it (`arcs.py:52`; `tanks.py:663-669`) |
| a user `DecayArc` or `DecayArcAlt` | its `in_port` | no `parent` keyword, so `"parent" not in dir(self)` (`core.py:465-468`); the sending node's temperature, for either direction |

So the temperature is never the water's own: a WSIMOD TODO says so (`arcs.py:700-702`). noodl
reads the same node's `"temperature"` driver row (Inputs).

**Arithmetic** (`generic_temperature_decay`, `:494-521`). The input VQIP is copied (`:507`),
`diff` is empty (`:508`), and then for each `pol` in the dict, in any order (each column is
independent):

```
factor   = constant * (exponent ** (T - 20))      # :513-515, ** binds first
diff[pol] = t[pol] * min(factor, 1)                # :512-517, Python min: factor unless 1 < factor
t[pol]    = t[pol] - diff[pol]                     # :519
```

Then `total_decayed = sum_vqip(total_decayed, diff)` (`make_decay`, `:491`). `diff` has volume
0, so `sum_vqip` adds its additive columns and keeps `total_decayed`'s non-additive ones
(`core.py:104-114`): a decayed non-additive value never enters `total_decayed`.

Quirks, all transcribed:

- **No volume guard.** A zero-volume VQIP decays its pollutant columns. Mass at zero volume can
  exist, for example on a surface after deposition, which is 3c. Zero mass gives `diff = 0`.
- **No temperature guard.** The decay continues below 0 °C, with `factor < constant` below
  20 °C when `exponent > 1`.
- **The cap.** `min(factor, 1)` removes everything once `factor ≥ 1`. When the cap binds, the
  Python result is the integer `1`, and `t * 1` is exactly `t`, so `t − t = 0`. noodl writes
  `where(cap, 1.0, factor)`, which gives the same value.
- **Non-additive keys decay too.** For `do` or `temperature`, the intensive value is multiplied
  by `1 − min(factor, 1)`. This is accepted and transcribed (X21). It has no mass-balance term.
- **Exponent domain.** With `exponent = 0`, `0.0 ** negative` raises `ZeroDivisionError` when
  `T < 20`. With `exponent < 0`, a non-integer power gives a `complex`, and `min(complex, 1)`
  raises `TypeError`. noodl refuses `exponent ≤ 0` by name (defect table).
- **Forcing type.** Forcing values are floats, so `exponent ** (T − 20)` is a float power even
  for an integer `exponent`; `float.__pow__` converts it, as in 3a's `pow_python`.

### `DecayTank` (`tanks.py:403-457`)

At the node's `end_timestep`, which happens after the day's record (`model.py:1172-1173`),
`end_timestep_decay` runs these steps (`:442-447`):

1. `total_decayed = empty`;
2. `storage_ = copy(storage)` (only for WSIMOD's own balance; not ported);
3. `storage = make_decay(storage)`, with the day's temperature.

The tank's other methods are `Tank`'s. Classes and orchestration:

- `Storage.end_timestep`, `Groundwater`, `QueueGroundwater` and `Reservoir` call
  `tank.end_timestep()` (`storage.py:134-136`).
- `RiverReservoir` and `River` call it from their own `end_timestep_` (`:1130-1133`,
  `:951-955`).
- `Land.end_timestep` calls each surface's, then the residence tanks' (`land.py:266-275`). The
  residence tanks never decay; a WSIMOD TODO notes this (`:136-137`).

A decaying `QueueGroundwater` first builds a `DecayTank` in `Storage.__init__`
(`storage.py:61-68`) and then replaces `self.tank` with a `DecayQueueTank` (`:307-315`). The
first tank stays only as the frozen push check of W52 (`check_tank`). It is never ended, so it
never decays.

`record_all` reports only the classes `Tank`, `QueueTank` and `ResidenceTank`, matched exactly
(`model.py:1123`). A `DecayTank` or `DecayQueueTank` therefore drops out of
`WsimodResults.tank_*`, as in WSIMOD. Its storage stays in the state (`"wsimod.storage"`).

### `DecayQueueTank` and its `DecayArcAlt` (`tanks.py:645-696`, `arcs.py:731-805`)

The tank is a `QueueTank` (3a) whose internal arc is a `DecayArcAlt` with `parent` set to the
node (`:663-669`).

**Entry.** `push_storage` (`tanks.py:513-539`) calls the internal arc's `send_push_request`,
which runs as in 3a, including the early return below `FA` (`arcs.py:378-379`).
`DecayArcAlt.enter_queue` (`:752-778`) runs these steps:

1. `enter_arc` counts the request before decay in `vqip_in` and `flow_in` (`:766`,
   `:409-430`).
2. `request["vqip"] = make_decay(request["vqip"])` (`:769`).
3. The request is summed into, or assigned to, `queue[time]` as in `AltQueueArc` (`:772-778`;
   3a's `present` register, W48).

`update_queue` then moves the decayed `queue[0]` into `active_storage` (`:603-638`). The tank's
`storage` gains the **undecayed** accepted part: `sum_vqip(storage, v_change(vqip, V −
reply.V))` (`tanks.py:536-538`).

**End of the day.** `_end_timestep` (`tanks.py:689-696`) runs these steps:

1. `storage = extract_vqip(storage, internal_arc.total_decayed)` (`:694`).
2. `storage_ = copy(storage)` (`:695`).
3. `internal_arc.end_timestep()`, which is `DecayArcAlt._end_timestep` (`:748`, `:780-805`):
   1. reset `vqip_in`, `vqip_out`, `total_decayed`, `flow_in` and `flow_out` (`:787-791`);
   2. `queue_ = copy(queue)`;
   3. for `i` in `0 … max_travel − 1`, if key `i + 1` exists, set `queue[i] =
      make_decay(queue_[i + 1])` and `queue[i + 1] = empty` (`:798-803`);
   4. `queue[0] = sum_vqip(queue[0], make_decay(queue_[0]))` (`:805`).

Four consequences follow. All are transcribed (X13).

- **The decay reaches `storage` a day late.** At the end of day d, `storage` loses what
  `total_decayed` holds at that moment: day d's entry decay plus the end-of-day decay of day
  d − 1. The end-of-day decay of day d is computed afterwards and waits in `total_decayed` until
  the end of day d + 1. Only the pollutant columns lag, because decay changes no volume.
  `storage`'s pollutant columns are WSIMOD's bookkeeping: no pull reads them
  (`tanks.py:553-562` takes its reply from `active_storage`), and `record_all` does not report
  the class. Running WSIMOD while writing this spec confirmed the lag: after a push of 5 kg of phosphate, `storage`
  read 5.0, then 4.608 and 4.247 on the next two days, while the queue held 4.247 − 0.333 = 3.914
  after the second day. So `total_decayed` is carried state (State).
- **`active_storage` never decays** (TODO, `tanks.py:692-693`). Water decays only while it is in
  the queue, both on entry and at the end of each day it waits.
- **No `update_queue` at the end of the day.** `QueueTank._end_timestep` ends with
  `update_queue` (`tanks.py:633-634`). `DecayQueueTank._end_timestep` does not. Water that
  reaches `queue[0]` at the end of day d enters `active_storage` only at the next push (every
  push flushes `queue[0]`) or at `QueueGroundwater.distribute`'s `update_queue`
  (`storage.py:352`). Until then it is decayed again at each end of day. The timing differs from
  a non-decaying `QueueGroundwater`, so a quantity-only run keeps this structure (X25).
- **The merge order is swapped.** The new `queue[0]` is `sum_vqip(decay(old[1]),
  decay(old[0]))`, while `AltQueueArc` computes `sum_vqip(old[0], old[1])` (`arcs.py:660`).
  The non-additive blend `(t2·V2 + t1·V1)/V` (`core.py:112-114`) therefore adds in the other
  order, and noodl follows it.

Order of `total_decayed` (`core.py:491`, sequential `sum_vqip`):

1. the day's entries, in push order;
2. after the reset, the moved slots `1 … L` in ascending order;
3. slot 0.

noodl adds in that order. An absent key adds an empty `diff`, which is an exact zero. `L` is
3a's register length; WSIMOD's growing `max_travel` covers only keys that exist, and the rest
are absent.

### User `DecayArc` and `DecayArcAlt` (`arcs.py:668-805`)

**`DecayArc`** is a `QueueArc` (3a's request rows) with two changes.

- **Entry** (`enter_queue`, `:686-708`): the request passes `enter_arc` undecayed (`:698`),
  then `request["vqip"] = make_decay(request["vqip"])` (`:705`), then it is appended (`:708`).
  - This holds for pushes (`:396`) and pulls (`:354`). A pulled request decays after
    `in_port.pull_set` has removed it from the sender.
  - A push's backflow is the decayed water, returned to the sender and taken off `vqip_in`
    (`:399-405`).
  - With `number_of_timesteps = 0` the request still decays once, on entry, and is then
    delivered at once.
- **End of the day** (`end_timestep`, `:710-728`, replacing `QueueArc.end_timestep`):
  1. reset the accumulators and `total_decayed` (`:717-721`);
  2. for **every** waiting request, in list order and in both directions, `vqip =
     make_decay(vqip)` and then `time = max(time − 1, 0)` (`:726-728`).

  In noodl the decay applies in place to every present row of `"wsimod.arc_queue"`, and the
  `since` counters advance as in 3a.

`total_decayed` feeds only WSIMOD's arc balance (`:684`), never a state. noodl sums the term in
its row order, which is not WSIMOD's list order across the two directions. The term is
therefore compared at 1e-12 × scale with that named cause (X24). The rows themselves are
bitwise.

**`DecayArcAlt`** as a user arc is 3a's user `AltQueueArc` (shifted at every end of day,
flushed only by its next push) with `DecayQueueTank`'s entry and end-of-day rules above. Its
temperature is its `in_port`'s. Its `total_decayed` feeds only its own balance (`:750`), so it
is not carried.

### `Surface.run`: the nitrite carry-over (`land.py:425-449`)

When `"nitrite"` is a pollutant (`:427`), each surface's `run` starts by converting the mass
that decayed at the previous end of day (`total_decayed`, set by `end_timestep_decay`):

```
storage["nitrate"]  += total_decayed["nitrite"]     # :434
running_inflow_mb["nitrate"] += total_decayed["nitrite"]   # :435 (the Land's)
storage["nitrite"]  += total_decayed["ammonia"]     # :438
running_inflow_mb["nitrite"] += total_decayed["ammonia"]   # :439
```

The storage is updated in place, without `sum_vqip`; noodl adds to the columns in this order.
Then the `inflows + processes + outflows` loop runs (`:441-449`, 3a). Details:

- Decayed nitrate, phosphate or anything else is lost, and becomes the out-term
  `"DecayTank.decayed"` (Mass balance).
- `total_decayed` starts empty (`core.py:472`). On day 0 the conversion therefore adds zero.
  After that it holds the previous day's decay, so it is carried state.
- Without `"nitrite"` the block is skipped. The surface still decays, and the decayed mass is
  then simply lost.
- The block reads `nitrate`, `nitrite` and `ammonia` (`KeyError` otherwise). noodl refuses
  `nitrite` without the other two by name.
- WSIMOD converts at every `Surface.run`. A second `Land.run` on one day therefore converts the
  same decayed mass again and counts it as an inflow again. noodl transcribes this (X16); that
  orchestration already applies the day's rain twice (3a).

3a omitted the block as "adds zero without decays". 3b ports it. `x + 0.0` is exactly `x` for
every `x` except `−0.0`, so the results are unchanged.

### `PerviousSurface.calculate_soil_temperature` (`land.py:1060-1079`)

This is the surface's only process (`:833-835`). It runs after `ihacres` and before `route`
(`:830`, `:841`):

```
auto  = storage["temperature"] * w_prev                       # :1070
air   = T_air * w_air                                         # :1071 (the Land's forcing)
total_weight = w_air + w_deep + w_prev                        # :1072-1074, that order
storage["temperature"] = (auto + air + T_deep * w_deep) / total_weight   # :1075-1077
```

It returns empty mass-balance VQIPs (`:1079`). The weights are attributes set in `__init__`
(`:809-812`: 0.1, 0.6, 0.1 and `soil_temp_deep` 10), so they reach a configuration only through
overrides (`:843-884`). 3a already transcribes the general formula. The decay of the surface's
`DecayTank` uses the **air** temperature, not this soil temperature (WSIMOD TODO, `:837-839`),
and noodl does the same. A total weight of 0 raises `ZeroDivisionError` and is refused by name,
both at build and at `bind`.

### WTW treatment (`wtw.py:163-214`)

`influent = copy(current_input)` (`:167`). Then:

1. **Non-additive columns** (`:174-182`, 3a, unchanged):
   - `discharge[k] = influent[k]`;
   - `liquor[k] = (liquor[k]·V_liq + influent[k]·V_in·m_v) / (V_liq + V_in·m_v)`.

   Where that denominator is 0, the old value is kept (3a defect row).
2. **Volume and additive columns**, in the order `ADDITIVE_POLLUTANTS + ["volume"]`
   (`:188-201`). The columns are independent.

   ```
   temp_factor = exponent[k] ** (20 - influent["temperature"])   # :191-193, note 20 − T
   (volume: temp_factor = 1, :195)
   discharge[k] = influent[k] * constant[k] * temp_factor        # :197-199, left to right
   liquor[k]    = influent[k] * liquor_multiplier[k]             # :201, overwritten, not blended
   ```

   - The volume's constant is the derived `c_v = 1 − percent_solids − m_v` (`:92`, `:104-110`).
   - `x · c · 1` is exactly `x · c`, and `pow_python(1, y) = 1.0` exactly. So 3b's general code
     reproduces 3a's neutral split bitwise.
3. `solids["volume"] = V_in · percent_solids` (`:204`). For each additive `k`, `solids[k] =
   (influent[k] − discharge[k]) − liquor[k]` (`:207-208`).
4. `treated = sum_vqip(treated, discharge)` (`:211`). The "more treated than input" event
   (`:213-214`) is 3a's.

Quirks:

- The factor rises as the temperature falls, when `exponent > 1`. Nothing caps
  `constant · factor` or the sum `constant · factor + liquor_multiplier`.
- When that sum exceeds 1, `solids[k]` goes negative and the works creates mass. WSIMOD's lists
  still close, because the solids are the remainder. noodl transcribes this and records the
  event `"wtw_negative_solids"` (X22).
- The temperature is the influent's after blending. A `WWTW`'s `current_input` already holds
  the recycled liquor (`:340`).
- With zero influent, the temperature column is whatever `current_input` holds, which is 0 after
  a reset. `exponent ** 20` then multiplies a zero mass.
- `process_parameters[k]` and `liquor_multiplier[k]` are read for every additive `k`
  (`KeyError` otherwise). 3a's `_fill_wtw_defaults` fills an empty dict as WSIMOD does
  (`:74-85`: constant 0.01, exponent 1.001, multiplier 0.7, volume 0.03). A dict that names only
  some pollutants is refused by name.
- Entries for non-additive pollutants, and a volume `exponent`, are never read (`:188`, `:195`).
  noodl refuses them by name.
- `exponent ≤ 0` is refused, as for decay: `0 ** negative` raises when `T > 20`, and a negative
  base gives a `complex`.

`FWTW` pushes `liquor + solids` to the sewer (`:607-611`) and `WWTW` recycles its liquor
(`:340`). Both are 3a's wiring, and now they carry treated masses.

### River biochemistry (`storage.py:738-898`)

`calculate_discharge` (`:917-924`) runs `biochemical_processes` only when `"nitrate"` is a
pollutant. It then **assigns** `bio_in, bio_out = in_, out_` (`:923-924`); the River's
`end_timestep_` empties them (`:951-955`). In the default orchestration it runs in slot 9,
before `Reservoir.make_abstractions`, `Catchment.route` and the river order's
`River.distribute`. It therefore acts on the storage before the day's catchment inflow.

The parameters are attributes set after `Storage.__init__` (`:539-566`), so they reach a
configuration only through overrides (`:620-644`). The defaults are the Python expressions,
evaluated as written:

| Parameter | Default (source) |
|---|---|
| `uptake_PNratio` | `1 / 7.2` (`:539`) |
| `denpar_w` | `0.0015` (`:541`) |
| `T_wdays` | `5` (`:543`) |
| `halfsatINwater` | `1.5 * 1e-3` (`:545-547`) |
| `hsatTP` | `0.05 * 1e-3` (`:551`) |
| `limpppar` | `0.1 * 1e-3` (`:552`) |
| `prodNpar` | `0.001` (`:553`) |
| `prodPpar` | `0.0001` (`:554-556`) |
| `muptNpar` | `0.001` (`:557`) |
| `muptPpar` | `0.0001` (`:558`) |
| `max_temp_lag` | `20` (`:560`) |
| `max_phosphorus_lag` | `365` (`:563`) |

`area = length · width` (`:524`, 3a's derived value). `S` below is `tank.storage`. The DIN pool
is `din = sum([S["ammonia"], S["nitrate"]])`, Python's `sum` from 0 (`:733-735`;
`din_components`, `:566`; nitrite is not included, `:567-568`).

1. **Depth** (`:746`, `:723-725`): `current_depth = S.V / area`.
2. **Temperature relaxation** (`:748-750`):
   `S["temperature"] = (1 − 1/T_wdays)·S["temperature"] + (1/T_wdays)·T_air`. `T_air` is the
   River's own forcing (`get_data_input`, `nodes.py:635`). This step runs even on a dry day.
   Below, `T` is the relaxed `S["temperature"]`.
3. **Lags** (`:752-763`). These run before the guards, so every call appends.
   - If `len(lagged_temperatures) > max_temp_lag`, delete the oldest; then append `T`.
   - If `len(lagged_total_phosphorus) > max_phosphorus_lag`, delete the oldest; then append
     `S["phosphate"] + S["org-phosphorus"]`.

   The check comes before the append, so the lists hold `max_lag + 1` entries: 21 and 366.
4. **Guards**, which return `(empty, empty)`:
   - `S.V < FA` (`:766-768`);
   - `T ≤ 0` (`:770-772`).
5. **Denitrification** (`:775-801`):

   ```
   tempfcn = 2 ** ((T - 20) / 10)                 # :775
   if T < 5: tempfcn *= T / 5                     # :776-777
   din = get_din_pool();  c = din / S.V           # :779-780
   confcn = c / (c + halfsatINwater)              # :781-783
   denitri_water = denpar_w * area * tempfcn * confcn          # :784-786, left to right
   river_denitrification = min(denitri_water, 0.5 * din)       # :788-790, the 50 % cap
   din_ = din - river_denitrification                          # :791
   if din > 0: for pol in (ammonia, nitrate):                  # :796-801
       loss = (din - din_) / din * S[pol];  out_[pol] += loss;  S[pol] -= loss
   ```
6. **Averages and coefficients** (`:803-819`). `din` is recomputed. Then:

   ```
   temp_10_day = sum(lagged_temperatures[-10:]) / 10           # :806
   temp_20_day = sum(lagged_temperatures[-20:]) / 20           # :807
   total_phos_365_day = sum(lagged_total_phosphorus) / max_phosphorus_lag   # :808
   tempfcn = T / 20 * (temp_10_day - temp_20_day) / 5          # :811-813, left to right
   totalphosfcn = (TP - limpppar) / (TP - limpppar + hsatTP) if (TP - limpppar + hsatTP) > 0 else 0
   ```

   Here `TP` is `total_phos_365_day`.

   - Each `sum` runs from 0, oldest entry first.
   - Each mean divides by the full window even before the list has filled: a cold start.
   - The phosphorus "mean" adds 366 entries and divides by 365.
   - `totalphosfcn` is negative for `limpppar − hsatTP < TP < limpppar`.
   - `TP` is a **mass** (kg), compared with the concentrations `limpppar` and `hsatTP` (kg/m³).
   - The four quirks above are reproduced (X14).
7. **Production or mineralisation** (`:822-873`):

   ```
   minprodN = prodNpar * totalphosfcn * tempfcn * area * current_depth                  # :822-824
   minprodP = prodPpar * totalphosfcn * tempfcn * area * current_depth * uptake_PNratio # :825-832
   ```

   - **Production** if `minprodN > 0` (`:833`):
     1. `minprodN = min(0.5·din, minprodN)` and `minprodP = min(0.5·S["phosphate"], minprodP)`;
     2. `out_["phosphate"] = minprodP`, `S["phosphate"] −= minprodP`;
     3. `in_["org-phosphorus"] = minprodP`, `S["org-phosphorus"] += minprodP`;
     4. if `din > 0`, then for each DIN `pol`: `loss = minprodN · S[pol] / din`, `out_[pol] +=
        loss`, `S[pol] −= loss`;
     5. `in_["org-nitrogen"] = minprodN`, `S["org-nitrogen"] += minprodN` (`:835-854`).

     The assignments are equivalent to `+=` here, because those entries are still 0.
   - **Mineralisation** otherwise (`:856-873`):
     1. `minprodN = min(0.5·S["org-nitrogen"], −minprodN)` and
        `minprodP = min(0.5·S["org-phosphorus"], −minprodP)`;
     2. `in_["phosphate"] = minprodP`, `S["phosphate"] += minprodP`;
     3. `out_["org-phosphorus"] = minprodP`, `S["org-phosphorus"] −= minprodP`;
     4. if `din > 0`, then for each DIN `pol`: `gain = minprodN · S[pol] / din`, `in_[pol] +=
        gain`, `S[pol] += gain`;
     5. `out_["org-nitrogen"] = minprodN`, `S["org-nitrogen"] −= minprodN`.

     When `din = 0`, the mineralised nitrogen leaves `org-nitrogen` and is added nowhere. The
     loss is reproduced and recorded (X15).
   - `minprodN = −0.0` (from `totalphosfcn = 0` and `tempfcn < 0`) takes the mineralisation
     branch and moves nothing.
8. **Macrophyte uptake** (`:875-895`). `din` is recomputed. Then:

   ```
   tempfcn1 = (max(0, T) / 20) ** 0.3          # :878; T > 0 here, so max returns T
   tempfcn2 = (T - temp_20_day) / 5            # :879
   tempfcn  = max(0, tempfcn1 * tempfcn2)      # :880
   macrophyte_uptake_N = min(0.5 * din, muptNpar * tempfcn * area)       # :882-883
   if din > 0: for pol in DIN: loss = uptake_N * S[pol] / din; out_[pol] += loss; S[pol] -= loss
   macrophyte_uptake_P = min(0.5 * S["phosphate"], muptPpar * tempfcn * max(0, totalphosfcn) * area)
   out_["phosphate"] += uptake_P;  S["phosphate"] -= uptake_P            # :890-895
   ```

The four 50 % caps are Python `min`s with two operand orders:

- `min(denitri_water, 0.5·din)` takes the first unless the second is smaller;
- `min(0.5·pool, x)`, used at every other cap, takes the pool half unless `x` is smaller.

noodl's decision margins follow each site's order. The integer `0` of `max(0, ·)` and of
`totalphosfcn = 0` multiplies as `0.0`, so noodl's float zeros give the same values, including
the sign of zero.

**Pollutants read.** The step reads `temperature`, `phosphate`, `org-phosphorus`, `ammonia`,
`nitrate` and `org-nitrogen`. A configuration with `nitrate` but without any of the others makes
WSIMOD raise `KeyError`. noodl refuses it by name.

## State

The new carried state follows 3a's dict-of-tensors layout (W68/W77: each class declares its
`carried_paths`; per-day values are not carried):

| Key | Shape | What it holds (WSIMOD attribute) | Labels |
|---|---|---|---|
| `"wsimod.total_decayed"` | (..., n_rows, 1+K) | `total_decayed` of each `DecayQueueTank`'s internal arc (`tanks.py:694`, `arcs.py:789`), then, when `"nitrite"` is a pollutant, of every surface (`land.py:434-439`) | `(node, "tank")` for a decaying `QueueGroundwater`; `(node, "surface:<name>")` for a surface |
| `"wsimod.river_temperature_lag"` | (..., n_slots) | `lagged_temperatures` of every River with biochemistry (`storage.py:561`, `:753-755`), as a fixed register of `max_temp_lag + 1` slots per river, flattened as 3a's `travel_queue` is | `(river, i)`, `i = 0 … max_temp_lag`, with slot 0 the oldest; unit `degC` |
| `"wsimod.river_phosphorus_lag"` | (..., n_slots) | `lagged_total_phosphorus` (`:564`, `:758-763`), `max_phosphorus_lag + 1` slots per river | `(river, i)`, `i = 0 … max_phosphorus_lag`; unit `kg` |

The changes to existing keys:

- A decaying storage-family node's `DecayTank` and every surface stay in `"wsimod.storage"`.
- A `DecayQueueTank` uses 3a's `"wsimod.storage"`, `"wsimod.active_storage"`,
  `"wsimod.travel_queue"` and `"wsimod.travel_queue_present"` rows, labelled as for a
  `QueueTank`.
- A user `DecayArc` uses `"wsimod.arc_queue"` and `"wsimod.arc_queue_since"`. A user
  `DecayArcAlt` uses `"wsimod.travel_queue"` and its `present` rows. The queued water in all of
  these is already decayed: decay rewrites the rows in place, and nothing new is carried.
- `bio_in`, `bio_out`, `current_depth`, the treatment's `solids` and the end-of-day
  `total_decayed` of a `DecayTank` or a user `Decay*` arc live within a day. They are terms or
  per-day stores, not state.

**The registers (X5, X17).** A call shifts a river's register left by one slot and writes the
new value into the last slot. WSIMOD's list, by contrast, grows to `max_lag + 1` entries and
then deletes the oldest before each append. After call `n`, the register's last `min(n, size)`
slots equal WSIMOD's list, and the slots in front of them are `+0.0`. The sums reproduce
Python's exactly:

- the windows are the last `min(10, size)`, `min(20, size)` and `size` slots;
- they are added one slot at a time, oldest first, starting from `+0.0`;
- Python's `sum` starts from the integer 0, and `0 + x`, `+0.0 + x` and `+0.0 + 0.0 + … + x`
  are all exactly `x`, including when `x = −0.0`.

So no presence mask is needed. The initial state is all zeros, which matches WSIMOD's empty
lists. A run can be warm-started from another run's registers. Unlike WSIMOD, noodl stores no
list length, but zero-padded slots are indistinguishable from it in every sum. The registers
are not water held, so the mass balance skips them; they join `_FLAT_KEYS`.

A decaying river with a small `max_temp_lag` (for example 5) has 6 slots, and both temperature
windows are then the whole register. This is WSIMOD's `[-10:]` of a 6-entry list.

## Inputs

**Drivers, parameters** (3a's convention `"wsimod.<Class>.<parameter>"`, rows in configuration
order, D6):

| Driver | Shape | Rows, columns | Notes |
|---|---|---|---|
| `"wsimod.<Class>.decays.constant"`, `"….decays.exponent"` | (..., n, K) | every node of a storage-family class `Class`; every surface of class `Class` (`ImperviousSurface`, `PerviousSurface`); every arc of class `DecayArc` or `DecayArcAlt`. Columns: the pollutants | an entry outside the element's `decays` key set is inert: its template is `(0, 1)` and `bind` refuses any other value (X9; W65's pattern) |
| `"wsimod.<Class>.process_parameters.constant"`, `"….exponent"` | (..., n, K_add) | every `WWTW`, `FWTW` or `WTW`; columns: the additive pollutants | absent in a quantity-only run, as `pollutant_load` is |
| `"wsimod.<Class>.liquor_multiplier"` | (..., n, 1 + K_add) | same rows; columns `volume`, then the additive pollutants | **changed from 3a's (..., n)**, which was the volume entry alone (X10) |
| `"wsimod.River.<p>"` for `uptake_PNratio`, `denpar_w`, `T_wdays`, `halfsatINwater`, `hsatTP`, `limpppar`, `prodNpar`, `prodPpar`, `muptNpar`, `muptPpar` | (..., n_rivers) | every `River` | present only when biochemistry runs |
| `"wsimod.PerviousSurface.soil_temp_*"` | (..., n) | already 3a's drivers | 3b lifts the neutral requirement; `bind` refuses a weight sum of 0 |

New units: `decays.*` is `-`. The River parameters' units are WSIMOD's comments:

| Parameter | Unit |
|---|---|
| `denpar_w` | `kg/m2/d` |
| `T_wdays` | `d` |
| `halfsatINwater`, `hsatTP`, `limpppar` | `kg/m3` |
| `prodNpar`, `prodPpar` | `kg/m3/d` |
| `muptNpar`, `muptPpar` | `kg/m2/d` |
| `uptake_PNratio` | `-` |

**Drivers, forcing.** `"wsimod.temperature"` `(..., T, n_readers)` gains these readers, and the
forcing check at build time refuses a missing `(temperature, date)` by name, as in 3a:

- every storage-family node with a `decays` dict, including `Groundwater` and `QueueGroundwater`
  through their configured `data_input_dict` (defect table: WSIMOD loses it);
- every `River` when `"nitrate"` is among the configured pollutants;
- the `in_port` of every user `DecayArc` or `DecayArcAlt`, each node counted once.

An `in_port` that is a `Catchment` reads its temperature from `"wsimod.concentration"`. That
requires `temperature` to be a pollutant, and without one the arc is refused by name.

These readers are required in quantity-only runs too, because WSIMOD reads them whatever the
pollutant list (3a's rule).

**Learnable.** Every new real-valued driver above can be batched and differentiated, together
with the temperature forcing through every new process.

**Structural** (cannot be batched or learnt):

- the pollutant lists, including whether `nitrate` (biochemistry) and `nitrite` (carry-over)
  are among them;
- whether a node's `decays` is `None` or a dict, which decides `Tank` or `DecayTank`, and
  `QueueTank` or `DecayQueueTank` with its different end of day;
- each `decays` dict's key set;
- the arc classes;
- `max_temp_lag` and `max_phosphorus_lag`, which are non-negative and positive integers, set by
  override, and size the registers (`max_phosphorus_lag` is also the divisor);
- the DIN components `ammonia` and `nitrate` (`storage.py:566`).

**Quantity-only runs** (`pollutants=False`, W10):

- Decay values, the WTW pollutant entries and the River biochemistry parameters are not read.
  They are listed in `model.ignored_by_quantity_only`.
- The structure stays: a decaying `QueueGroundwater` keeps `DecayQueueTank`'s end of day,
  because WSIMOD's volumes depend on it (X25).
- Volumes still never depend on any pollutant value or parameter. With X12 (the
  `QueueGroundwater` delay) this holds for every 3b class, and a test asserts it.

## Mass balance

3a's balance (W84/W86/W91) gains these terms, each `(..., T, 1 + K_add)` with the direction
declared beside it. Every term covers additive columns only; the non-additive changes (decayed
`do`, relaxed or soil temperature) have none.

| Term | Dir. | When | Value |
|---|---|---|---|
| `"DecayTank.decayed"` | out | each `DecayTank`'s end of day (storage family and surfaces) | `diff` (`tanks.py:447`) |
| `"DecayQueueTank.decayed"` | out | each `DecayQueueTank`'s end of day | what `storage` loses, `total_decayed` at `tanks.py:694` |
| `"DecayArc.decayed"`, `"DecayArcAlt.decayed"` | out | each user arc's entry and end-of-day decay | the `diff`s, summed |
| `"River.bio_in"`, `"River.bio_out"` | in, out | each `calculate_discharge` call | `in_` and `out_`, **summed over the day's calls** |
| `"Land.running_inflow_mb"` (3a) | in | as in 3a | now includes the nitrite carry-over (`land.py:435`, `:439`) |

Arcs declare terms as nodes do (`mass_balance`), and `_balance` collects them from both.

**The window becomes contiguous (X8).** In 3a, a day's storage change runs from the state
loaded at its start to the water held at the record point, which is before `end_timestep`. The
end-of-day decays happen in `end_timestep`, after the record. Under 3a's window they would fall
between two days and appear in no residual, which is WSIMOD's own blindness (below). In 3b:

- `storage_change(d) = held(after end_timestep of d) − held(start of d)`;
- the terms recorded during `end_timestep` belong to day d;
- `held(start of d + 1)` equals `held(after end_timestep of d)` exactly, because it is the same
  state, so consecutive windows meet.

In 3a, `end_timestep` changes the held water only by rounding: the shift of a user
`AltQueueArc` re-adds rows, and a `QueueTank`'s queue and active storage are inside its storage.
3a's closure test therefore passes unchanged under the new window, and the plan re-runs it.

`held` counts a `DecayQueueTank` by its `storage`, as 3a counts a `QueueTank`. So the term is
the lagged extraction, and it closes the balance exactly where `storage` changes. The queue
(already decayed) and `"wsimod.total_decayed"` are not held. The surfaces' `total_decayed` is
mass that leaves on day d (`"DecayTank.decayed"`) and returns on day d + 1 as converted nitrite
or nitrate (`"Land.running_inflow_mb"`).

**WSIMOD's own lists, and where they break.**

- **`DecayTank`.** `decay_ds` adds `total_decayed` back into `ds` (`tanks.py:449-457`). Each
  node's lists close, but the decayed mass appears in no in, out or ds list, so the system loses
  it unseen. WSIMOD's system check would not notice even if it worked; it compares `sys_in` with
  itself (3a defect row).
- **`DecayQueueTank`.** The tank's `ds` is `Tank.ds` across a window in which `storage` does not
  decay; the extraction falls between WSIMOD's windows. The internal arc lists `total_decayed`
  as an outflow (`arcs.py:750`), but internal arcs are not in `Model.arcs` (`model.py:1016`), so
  the list is never summed. The decay is invisible.
- **User `DecayArc` and `DecayArcAlt`.** These lists close. `vqip_in` is undecayed, `out`
  includes `total_decayed` (`:684`, `:750`), and `queue_storage_` is captured before the end-of-
  day decay (`:723`, `:793`).
- **River.** The lists are `bio_in` and `bio_out` (`storage.py:595-596`).
  - `calculate_discharge` assigns them rather than adding to them (`:923-924`). A second call on
    one day replaces the first call's terms, and WSIMOD's node check then prints a mass-balance
    error. noodl's own balance adds every call (defect table).
  - A mineralisation with `din = 0` lists the nitrogen loss under `bio_out` and no gain. The
    lists close, and the nitrogen is destroyed (X15).
- **Surface carry-over.** The conversion is listed as an inflow (`land.py:435`, `:439`), while
  the decay that produced it is hidden in `ds`. Each node closes, but the system's totals would
  show converted nitrite or nitrate entering from outside. noodl lists both sides.
- **WTW.** The lists close by construction, because the solids are the remainder
  (`wtw.py:207-208`), including when they are negative.

Every 3b process changes no volume, so the volume column's balance is 3a's.

## Interaction with 3a

### Classes

- `tanks.DecayTank` gets the real end-of-day decay. A new `tanks.DecayQueueTank` subclasses
  `QueueTank` with a `DecayArcAlt` internal arc and its own end of day.
- `arcs.DecayArc(QueueArc)` and `arcs.DecayArcAlt(AltQueueArc)` are new. `ARC_SCHEMAS` gains
  them, with the keys `name`, `type_`, `in_port`, `out_port`, `capacity`, `preference`,
  `number_of_timesteps` and `decays`. `parent` is refused by name: in WSIMOD a string `parent`
  would become the temperature source and crash.
- `storage.Storage` and its family build the decaying tanks. `River` gets `calculate_discharge`,
  the biochemistry, its parameters and the registers.
- `wtw.WTW.treat_current_input` becomes general.
- `land.Surface.run` gets the carry-over. `land.Land.validate` relaxes the soil-temperature
  check.
- `day.py` gets the state keys, the drivers and the contiguous window. `run.py` gets the terms.
- `structure.forcing_readers` gets the new temperature readers.

### Validation

The refusals 3a lifts are listed under Scope. New refusals, each by name and each with a test:

- **Decays.**
  - A `decays` key that is not a configured pollutant, including `volume`: WSIMOD would
    `KeyError`, or would decay the water itself.
  - A `decays` entry without `constant` or `exponent`, or with any other key.
  - `exponent ≤ 0`, at build and at `bind`.
- **WTW.**
  - `process_parameters` or `liquor_multiplier` missing an additive pollutant.
  - Entries for non-additive pollutants or unknown names.
  - A volume `exponent`.
  - `exponent ≤ 0`.
- **River with biochemistry.**
  - Any of the six pollutants missing.
  - `T_wdays = 0` (the step divides by it).
  - `area = length · width = 0` (`update_depth` divides by it).
  - `halfsatINwater ≤ 0`: with `din = 0` the confcn denominator is 0.
  - `max_temp_lag` not a non-negative integer: a negative one makes `del` on an empty list
    raise `IndexError`.
  - `max_phosphorus_lag` not a positive integer: it is a divisor.
  - The `bind`-time checks repeat for batched drivers.
- **Surfaces.** `nitrite` without `nitrate` and `ammonia` (`land.py:434-439`).
- **Soil temperature.** A total weight of 0.
- **Arcs.** `decays` on a non-`Decay*` arc, and `parent` on any arc.

### Loading and overrides

- A configuration's `decays` on a storage-family node, or on a surface, is read.
- Overrides (W5, 3a's per-class transcription):
  - `Storage.apply_overrides` merges `decays` into a node built with a dict, and refuses one on
    a node built with `None` (`storage.py:96-101`).
  - `DecayTank.apply_overrides` and `DecayQueueTank.apply_overrides` merge it (`tanks.py:439`,
    `:686`).
  - `River.apply_overrides` sets the 13 biochemistry attributes (`storage.py:620-644`).
- **Shared default dicts.** WSIMOD's mutable default `decays={}` is shared by every `DecayTank`
  built without a `decays` key (`tanks.py:406`), so an override on one surface reaches all of
  them. 3a's defect row already makes the merge per element. In 3b the row is live, and it gets
  a test with two surfaces.
- An arc `decays` override is refused, as WSIMOD consumes none (`Arc.apply_overrides`,
  `arcs.py:93-96`).
- The 3a rule stands: an override key that the class does not consume is refused by name.
  `QueueTank`'s `number_of_timesteps` override (`tanks.py:499-502`, `:682-685`) stays refused,
  as in 3a; it is not a 3b process.

### Decision vocabulary (W44), events and diagnostics

The new sites follow 3a's grammar, `"<node|arc>:<name>/<WSIMOD method>/<branch>"`.

- **Decay cap**, `…/make_decay/cap:<pollutant>`, evaluated once per element and day (Approach
  rule 3, `call` 0). One site per decaying pollutant:
  - `node:N/tank/make_decay/cap:<p>` for a storage-family tank;
  - `node:N/surfaces.<s>/make_decay/cap:<p>` for a surface;
  - `node:N/tank/internal_arc/make_decay/cap:<p>` for a `DecayQueueTank`;
  - `arc:A/make_decay/cap:<p>` for a user arc.

  The margin is `factor − 1`.
- **River**, `node:N/biochemical_processes/<branch>`, with `call` the index of the day's
  `calculate_discharge` call:
  - `dry` (`:766`) and `frozen` (`:770`);
  - `cold` (`:776`);
  - `denitrification_cap` (`:788`) and `din:denitrification` (`:796`);
  - `totalphosfcn` (`:814`);
  - `production` (`:833`), `production_cap:N` (`:835`) and `production_cap:P` (`:838`);
  - `din:production` (`:847`);
  - `mineralisation_cap:N` (`:858`), `mineralisation_cap:P` (`:859`) and
    `din:mineralisation` (`:866`);
  - `macrophyte_temperature` (`:878`) and `macrophyte_factor` (`:880`);
  - `macrophyte_cap:N` (`:883`) and `din:macrophyte` (`:884`);
  - `macrophyte_phosphorus` (`:891`) and `macrophyte_cap:P` (`:893`).

  The branches after `dry` and `frozen` are evaluated under the mask that those guards leave.
- The WTW treatment and the soil temperature have no branches.

**Events** (D13). WSIMOD prints nothing in these processes. noodl adds two events of its own
(X22):

- `"wtw_negative_solids"`: an additive solids column below 0;
- `"river_nitrogen_lost"`: mineralisation with `din = 0` and `minprodN > 0` after the cap.

### The replay harness (W41/W47)

`capture_node_calls` (`tests/verification/_wsimod_reference.py`) also wraps:

- `end_timestep` of the target nodes and their tanks, with the stores before and after;
- `River.calculate_discharge`. The capture records `lagged_temperatures` and
  `lagged_total_phosphorus`, padded into registers by a helper, and the River's
  `data_input_dict` value at `t`;
- `total_decayed` and the queue of each surface and `DecayQueueTank`.

A new `capture_arc_calls(model, arcs)` records a user `Decay*` arc's request calls, its
`update_queue` and its `end_timestep`, with the replies of its `out_port`. Its replay builds the
noodl arc alone between `ScriptedPort`s that return the captured replies.

The bound is 0 (bitwise) for every 3b class. The one exception is the `DecayArc` mass-balance
term, at 1e-12 × scale, with its cause named in the test (X24).

A helper `patch_groundwater_reference(model, network)` applies the two corrections of X11 and
X12 to the live WSIMOD model before `run`:

- it sets each `Groundwater`'s and `QueueGroundwater`'s `data_input_dict` from the
  configuration;
- it sets a decaying `QueueGroundwater`'s `tank.number_of_timesteps` and
  `tank.internal_arc.number_of_timesteps` to 0.

These are the only cases that compare against a patched WSIMOD. Each one also has a crash test
against the unpatched WSIMOD.

### The demos

- **What they contain** (`tests/data/wsimod/*/config.yml`).
  - **quickstart**: a `Sewer`, a `Land` and a `Groundwater` with `decays: null`, a `Node` as its
    river, and a `Waste`.
    - The `Land` has an `ImperviousSurface` and a `PerviousSurface`, both with `decays: {}` and
      a phosphate `pollutant_load`, and default soil-temperature weights.
    - It uses WSIMOD's 12 pollutants, so `nitrite` is among them.
  - **oxford**:
    - four `Catchment`s;
    - a `ResidentialDemand` and a `Reservoir` with `decays: null`;
    - an `FWTW` and a `WWTW`, both with WSIMOD's default process parameters (constant 0.01,
      exponent 1.001, multiplier 0.7, volume 0.03);
    - a `Land` with two surfaces carrying `decays: {}` and non-zero `pollutant_load`s, and
      default soil temperature;
    - a `Groundwater` with `decays: null`, and the river junctions as `Node`s.
    - It has 12 pollutants with `nitrate` but no `nitrite`, and no `River`.
- **The 3b-live configuration** (X20). Each demo's `config_neutral.yml` is replaced by
  `config_no_deposition.yml`. It differs from `config.yml` only by an `overrides` section that
  sets every surface's `pollutant_load` to 0, the one 3c process the demos contain.
  `scripts/regenerate_wsimod_fixtures.py` writes it, and the README says so.
- **What runs live.** Both demos then run every 3b process they contain:
  - oxford: the FWTW and WWTW treatment on 11 additive pollutants, and the rural surface's soil
    temperature;
  - quickstart: the soil temperature, and the nitrite carry-over with empty decays, which adds
    zero.
- **What neither demo contains:** decay with a non-empty `decays`, a `DecayQueueTank`, a user
  `Decay*` arc, and a `River`. The purpose-built cases cover them.
- **What still needs neutralising for 3c:** each surface's `pollutant_load`, in both demos.

## WSIMOD defects found in these files (X4)

Rule as in 3a: where the intent is evident, noodl does it; otherwise it refuses by name or, for
a state-dependent case, reproduces it and records an event. Each row has a test. Nothing is
filed upstream without the user (X4). Issue #258 exists.

| Defect | WSIMOD | noodl | Test |
|---|---|---|---|
| `Groundwater` takes `data_input_dict` and stores it before `super().__init__`, which lets `Node.__init__` reset it to `None` (`storage.py:154`, `:202-203`, `nodes.py:26`, `:58`) | a decaying `Groundwater` raises `TypeError: 'NoneType' object is not subscriptable` at its first decay (`core.py:485`); run while writing this spec | reads the configured data, the evident intent (X11) | crash test against WSIMOD; parity against the patched WSIMOD (`decay_groundwater`) |
| The same in `QueueGroundwater` (`storage.py:252`, `:291-292`) | `TypeError` at the first push or end of day; run likewise | the same (X11) | `decay_queue_groundwater` |
| `DecayQueueTank`'s default `number_of_timesteps=1` (`tanks.py:648`) contradicts its docstring ("Defaults to 0", `:656-658`) and `QueueTank` (`:463`). `QueueGroundwater` passes none (`storage.py:308-315`) | adding `decays` delays all of a `QueueGroundwater`'s water by one day; run likewise (keys 1 and 2 for a time-area {0, 1}) | 0, the evident intent, so that `decays` changes only pollutants (X12) | `decay_queue_groundwater` against the patched WSIMOD; a unit test that WSIMOD's default gives the delay |
| `River.calculate_discharge` assigns `bio_in`/`bio_out` (`storage.py:923-924`) | a second call per day drops the first call's terms; the node check prints an error | the processes run twice as in WSIMOD, and noodl's balance adds both calls' terms | `river_double_discharge`: WSIMOD prints, noodl's residual is 0 |
| River mineralisation with `din = 0` (`storage.py:866-873`) | the mineralised nitrogen is destroyed, and the lists show it as `bio_out` | reproduced, event `"river_nitrogen_lost"` (X15) | `river_nutrients` (DIN-free stretch): value parity and the event |
| River phosphorus "365-day mean" over 366 entries ÷ 365 (`storage.py:758-763`, `:808`), and the windows that divide by their full length from day 1 (`:806-808`) | biased mean; cold start | reproduced (X14) | `river_nutrients`, more than 366 days |
| River `total_phosphorus` is a mass compared with concentrations (`storage.py:551-552`, `:760-762`, `:814-817`) | `totalphosfcn` saturates near 1 for any river holding more than about 1e-4 kg of P | reproduced and documented: the evident fix (a concentration) has no single form, and every WSIMOD river model computes this (X14) | `river_nutrients` with a low-P river crossing both `totalphosfcn` branches |
| No check of the decay or treatment `exponent` (`core.py:513-515`, `wtw.py:191-193`) | `0 ** negative` raises `ZeroDivisionError`; a negative base gives a `complex`, and then `TypeError` | `exponent ≤ 0` refused | refusal tests |
| WTW solids are the remainder, unclamped (`wtw.py:207-208`) | negative solids create mass when `constant·f + m > 1` | transcribed; event `"wtw_negative_solids"` (X22) | `wtw_treatment` (cold days) |
| `decays` may name `volume` (`core.py:510-519`) | the water itself decays | refused | refusal test |
| Shared `decays={}` default (`tanks.py:406`), now reachable (3a row) | one surface's override decays every surface built without a `decays` key | per element | `surface_decay_nitrification` override test |
| `Surface.run` converts `total_decayed` at every call (`land.py:427-439`) | a second `Land.run` on one day converts the same mass twice | transcribed with the doubled run (X16); documented | `land_run_twice` |

Two behaviours are transcribed and are not defects (X13): `DecayQueueTank._end_timestep` omits
`update_queue`, and its `active_storage` never decays (`tanks.py:689-696`). WSIMOD does not
state the intent, nothing crashes, and no mass is lost. They are documented on the application
page.

## Differentiability

- **Learnable.** Every new real-valued driver (Inputs) and the temperature forcing.
- **Detached.** Every new decision (vocabulary above).
- **Kinks and jumps.** They are documented on the application page, and the gradient tests
  avoid them.
  - The decay cap `min(c·e^(T−20), 1)` is a kink. Above it, the gradient with respect to `c`,
    `e` and `T` is zero.
  - The four 50 % caps of the River are kinks. At them, the gradient moves to the pool.
  - The production/mineralisation switch at `minprodN = 0` is continuous, because both
    branches give 0 there, and it is a kink.
  - `T < 5` is continuous, because the factor is 1 at `T = 5`, and it is a kink.
  - `T ≤ 0` is continuous in value, because every process vanishes as `T → 0⁺`. The slope of
    `(T/20)^0.3` is unbounded at `0⁺`.
  - The dry guard `V < FA` is a jump.
  - The `totalphosfcn` guard is a singularity. As `TP − limpppar + hsatTP → 0⁺`, the function
    tends to `−∞`, and it jumps to 0 at the guard.
  - `max(0, ·)` in the macrophyte factor and in `max(0, totalphosfcn)` are kinks.
  - The `din > 0` guards are continuous, because every loss is bounded by `0.5·din`.
  - The WTW treatment and the soil temperature are smooth. The treatment can make solids
    negative without a kink.
- **Per-instance Python math (X18).** The four powers use `pow_python`. Measured on the
  developer machine (torch 2.14.1+cpu, Python 3.12.10, 10⁵ samples, `T ∈ [−5, 35]`), the share
  of results where `torch.pow` differs from Python's `**` in the last bit is:

  | Power | Differs |
  |---|---|
  | `1.005 ** (T−20)` | 0.98 % |
  | random `e ∈ [1, 1.1]` | 1.57 % |
  | `1.001 ** (20−T)` | 0.11 % |
  | `2 ** ((T−20)/10)` | 1.69 % |
  | `(T/20) ** 0.3` | 1.23 % |
  | `1 ** x` | 0 |

  The gradients of `pow_python` are the existing ones: `y·x^(y−1)` and `x^y·ln x` for `x > 0`.
  The bases here are positive:
  - `exponent > 0` by refusal;
  - 2;
  - `T/20 > 0` behind the `frozen` guard.

  `exp` and `sqrt` do not occur.
- **Cost.** These are Python calls per instance, per decaying pollutant and element per day
  (once, X18), per additive pollutant and treatment call, and twice per river call. The
  phosphorus register adds 366 sequential additions per river and call.
  `benchmarks/wsimod_run.py` measures oxford with 3b live at batch sizes 1, 10 and 100, with no
  asserts (W8).

## Verification (X3)

The harness is 3a's: live WSIMOD 0.8.1, `reset_wsimod_globals()` before every build, and the
cases in `tests/verification/_wsimod_cases.py`.

### 1. Per-class replay on captured inputs

- **Unit tests against WSIMOD's own objects**, on random sequences, asserted bitwise:
  - `generic_temperature_decay` against `DecayObj`. The inputs include a binding cap, `T`
    either side of 20, `exponent = 1`, zero volume with mass, a non-additive key, and absent
    keys.
  - `DecayTank` against a dummy parent with `data_input_dict`.
  - `DecayQueueTank`, built with an explicit `number_of_timesteps=0`, which needs no patch. The
    sequences mix pushes, pulls, `pull_set_active` and ends of day, and assert the one-day lag
    of `storage`, `total_decayed`, the swapped merge and the missing `update_queue`.
  - `DecayArc` with the W21 driver: irregular pushes, pulls, backflow and `n = 0`.
  - `DecayArcAlt` with irregular pushes and a non-additive blend.
  - `treat_current_input` against `WTW` objects, with non-neutral parameters and temperatures
    either side of 20.
- **Node replays** (`capture_node_calls`), bound 0:
  - FWTW and WWTW on oxford's `config_no_deposition.yml`, live;
  - `Land` with the soil temperature live, on both demos;
  - and, on the purpose-built cases: `Land` with decaying surfaces and the carry-over, `River`
    (`calculate_discharge`, both registers), and the storage family with decays.

  The default suite samples days as 3a does (W15); every day is replayed under `-m slow`.
- **Arc replays** (`capture_arc_calls`), bound 0 for the rows. The `DecayArc` term is compared
  at 1e-12 × scale (X24).

### 2. Whole-model parity

- Each demo runs on `config_no_deposition.yml` with pollutants on, compared bitwise:
  - every arc's flow and pollutants;
  - every recorded tank;
  - every surface field.

  The composition is 3a's: quickstart in full (1456 days) and oxford for 365 days by default,
  with full oxford under `slow`.
- The quantity-only run on `config.yml` stays.
- A new test asserts that the quantity-only volumes equal the 3b-live volumes exactly, on both
  demos and on every purpose-built case.
- Bounds and admission are W19's: 1e-12 × scale, and admissions only when named, measured and
  asserted. The expectation is bitwise with no admissions, as in 3a's Task 16.
- WSIMOD's per-day prints are checked as in W90/W91. The only new expected print is the River
  mass-balance error of `river_double_discharge`.
- noodl's balance closes within 1e-12 × scale on every case and every day, under the contiguous
  window (X8). The test is 3a's, re-run.

### 3. Purpose-built cases

Each case runs in both modes. Unless marked, it is compared with the unpatched WSIMOD.

- `decay_storage`:
  - a `Storage`, a `Reservoir`, a `RiverReservoir` and a `River` without nitrate, each with
    `decays` and node temperature forcing that crosses 20 °C;
  - a constant large enough that the cap binds on warm days;
  - a decaying `do` (non-additive);
  - a pollutant that is not decayed;
  - a `decays` override merged into a node built with a dict.
- `decay_groundwater`: a `Groundwater` with `decays`. It is compared with the patched WSIMOD
  (X11), and the unpatched one must raise.
- `decay_queue_groundwater`:
  - a `QueueGroundwater` with time-area {0: 0.5, 2: 0.5} and `decays`;
  - abstraction pulls through `pull_set_active`;
  - days on which water reaches slot 0 with no push before `distribute`, which makes the missing
    `update_queue` observable.

  It is compared with the patched WSIMOD (X11, X12).
- `decay_arc`: a user `DecayArc` with `n = 2` and a second one with `n = 0`, with push and pull
  requests, a rejecting receiver (decayed backflow), and an `in_port` with temperature forcing.
- `decay_arc_alt`: a user `DecayArcAlt` with `n = 2`, irregular pushes, and temperature as a
  blended pollutant.
- `surface_decay_nitrification`:
  - impervious and pervious surfaces with polluted VQIP initial storages (`tanks.py:36-42`) and
    decays on ammonia, nitrite and phosphate;
  - the full 12 pollutants;
  - live soil temperature;
  - the shared-default override test (two surfaces, one override).
- `soil_temperature`: pervious surfaces with non-default weights, including `w_prev = 0`.
- `wtw_treatment`:
  - an `FWTW` and a `WWTW` with per-pollutant constants and exponents other than the defaults;
  - liquor multipliers other than the volume's;
  - temperatures either side of 20 °C;
  - cold days on which the solids go negative (the event);
  - the storm tank filling and draining.
- `river_nutrients`:
  - two rivers in a chain, fed by catchments that carry all six pollutants;
  - 1100 days, so that the 366-slot register fills and turns over;
  - frost, so that `T ≤ 0` after relaxation;
  - days between 0 and 5 °C;
  - a dry day (V < FA, from an upstream abstraction);
  - warming and cooling seasons, giving production and mineralisation;
  - a low-P river that crosses both `totalphosfcn` branches;
  - a DIN-free stretch (`"river_nitrogen_lost"`);
  - one river overridden to `max_temp_lag = 5`, with other biochemistry overrides;
  - every 50 % cap binding at least once, which is asserted from the decisions.
- `river_double_discharge`: a custom orchestration with `River.calculate_discharge` twice.
  WSIMOD prints its mass-balance error. Values must be at parity and noodl's residual must be 0.
- `land_run_twice`: `Land.run` twice with decaying surfaces, at value parity.

### 4. Gradients

The protocol is 3a's: central differences at relative step 1e-6 and agreement 1e-6, with the
`"decisions"` records asserted identical between ±h, and the kink rule of W92/W93. The
derivatives checked are those of arc pollutant masses, final storages and the River's state
with respect to:

- the decay `constant` and `exponent`, below the cap;
- the WTW `process_parameters` constant and exponent and a pollutant `liquor_multiplier`;
- `denpar_w`, `halfsatINwater`, `prodNpar`, `muptNpar` and `T_wdays`;
- the soil-temperature weights;
- the temperature forcing, through decay, treatment and the river.

`torch.autograd.gradcheck` also runs on the decay function and on one `biochemical_processes`
call away from its guards.

### Refusals and structure

Each new refusal (Interaction › Validation) gets a by-name test. A batch of 4 instances with
different 3b parameters equals the 4 single runs. `model.step` day by day equals `wsimod_run`
with the new state keys. A state round-trip through `initial_state` and the registers is exact.
The no-`wsimod`-import subprocess test is unchanged.

## Documentation

- `docs/applications/wsimod.md`:
  - the ported processes;
  - the decay's temperature sources;
  - the River's quirks (X14, X15);
  - the `DecayQueueTank` behaviours (X13);
  - the new refusals, kinks and events;
  - the verification rows with measured values;
  - the demo configurations.
- `docs/formats/wsimod.md`: `decays`, the `Decay*` arcs, the River overrides, the WTW entries,
  and the soil-temperature overrides.
- `tests/data/wsimod/README.md`: `config_no_deposition.yml`.
- `docs/verification.md`: the new rows.
- AGENTS.md and `select_tests.py` keep the wsimod row (3a). The paths do not change.

## Rulings made in this spec

Each ruling states what was decided, why, and the cost if it is wrong.

- **X7, scope and the refusal table.** 3b ports every class and method in Scope › In. Every 3a
  "3b" refusal is decided in the table. The deposition refusals now name only 3c.
  - Why: X1's list, checked against `git grep 3b`.
  - Cost if wrong: a refusal left behind, or a class moved between sub-projects.
- **X8, a contiguous balance window.** A day's storage change runs from its loaded state to its
  dumped state, after `end_timestep`, and the end-of-day terms count in that day.
  - Why: the end-of-day decays otherwise fall between windows and disappear unseen, which is
    WSIMOD's blindness. The 3a numbers change only by rounding.
  - Cost if wrong: a re-baselined balance test.
- **X9, the `decays` key set is structural.** Drivers are dense `(…, n, K)`, absent entries are
  inert with template `(0, 1)`, and `bind` refuses a changed absent entry.
  - Why: `where(present, …)` is exactly WSIMOD's untouched column, and dense drivers keep 3a's
    layout.
  - Cost if wrong: a user cannot learn a decay for a pollutant without first configuring it.
- **X10, the WTW drivers.**
  - `liquor_multiplier` becomes `(…, n, 1 + K_add)`, with columns volume then the additive
    pollutants; this is WSIMOD's one dict.
  - `process_parameters.constant` and `.exponent` are `(…, n, K_add)`.
  - Why: physical names, and no second key for one WSIMOD parameter.
  - Cost if wrong: 3a's `(…, n)` `liquor_multiplier` callers break. They are refused by name,
    and the 3a tests and docs are updated.
- **X11, the `Groundwater` and `QueueGroundwater` forcing.** noodl reads the configured
  `data_input_dict`. The reference is WSIMOD patched the same way, with a crash test for the
  unpatched one.
  - Why: the evident intent. Without it the classes cannot decay at all.
  - Cost if wrong: parity rests on a patched reference for these two classes.
- **X12, a decaying `QueueGroundwater` uses `number_of_timesteps = 0`.**
  - Why: the class's docstring says 0, `QueueTank` uses 0, and no WSIMOD model can run the
    defective value (X11's crash). The volumes then never depend on `decays`.
  - Cost if wrong: a one-day timing difference against a WSIMOD with the data defect fixed by
    another route.
- **X13, `DecayQueueTank` is transcribed as written.** That covers the missing `update_queue`,
  the `active_storage` that never decays, the storage that lags by a day, and the swapped merge
  order.
  - Why: no stated intent, no crash and no mass loss, so faithfulness wins (X2).
  - Cost if wrong: a later fix changes timing and pollutant columns.
- **X14, River phosphorus quirks reproduced.** These are the mean of 366 entries over 365, the
  cold-start windows, and the mass compared with concentrations.
  - Why: every WSIMOD river model computes them; the fix for the units has no single evident
    form; parity is the evidence.
  - Cost if wrong: noodl inherits a modelling error that its users must know about. It is
    documented.
- **X15, mineralisation without DIN.** The loss is reproduced and recorded as
  `"river_nitrogen_lost"`.
  - Why: it is state-dependent, so it cannot be refused statically, and the intended home of the
    nitrogen (ammonia or nitrate) is not evident.
  - Cost if wrong: destroyed nitrogen in noodl results, which is visible in the event and the
    balance.
- **X16, repeated calls.**
  - The River's processes run at every call, and noodl's terms add over calls.
  - The surface carry-over is transcribed at every `Land.run`.
  - Why: state stays faithful, and noodl's own balance must be right. The carry-over doubling
    lives only in an orchestration that already doubles the rain.
  - Cost if wrong: one of these custom orchestrations differs from what a user expects.
- **X17, the lag registers.** They are fixed-size and zero-padded at the front, and summed from
  `+0.0`, oldest first, over WSIMOD's windows. They have no presence mask.
  - Why: zero padding is exact in Python's `sum`, so the registers are bitwise WSIMOD's lists.
  - Cost if wrong: a state-format change.
- **X18, Python powers, once per element and day.** `pow_python` is used for the four powers,
  and the decay factor is evaluated once per element and day.
  - Why: the measured `torch.pow` differences, and the factor is a pure function of that day's
    inputs.
  - Cost if wrong: speed at large batches, measured in the benchmark.
- **X19, the new refusals.** They are the list in Interaction › Validation.
  - Why: each is a WSIMOD crash, an unread key, or water decaying.
  - Cost if wrong: a configuration that WSIMOD runs is refused, and must be edited.
- **X20, the demo configurations.** `config_neutral.yml` is replaced by
  `config_no_deposition.yml`, which neutralises only the surfaces' `pollutant_load`. A test
  asserts that the quantity-only volumes equal the live volumes.
  - Why: X3. The 3a neutral file no longer names what it neutralises.
  - Cost if wrong: fixture churn.
- **X21, decaying non-additive pollutants.** They are accepted and transcribed, with no balance
  term.
  - Why: WSIMOD runs them, and decaying an intensive value is meaningful (dissolved oxygen).
  - Cost if wrong: a user's typo decays a temperature.
- **X22, two noodl events.** They are `"wtw_negative_solids"` and `"river_nitrogen_lost"`.
  - Why: they turn silent mass creation and loss into recorded facts, as D13 does.
  - Cost if wrong: two more diagnostic keys.
- **X23, `"wsimod.total_decayed"` as state.** It covers the `DecayQueueTank` internal arcs, and
  the surfaces when `nitrite` is a pollutant.
  - Why: only these two read the previous day's value. The storage family's `DecayTank`s and the
    user arcs reset it before every use.
  - Cost if wrong: a state-format change.
- **X24, the `DecayArc` mass-balance term.** It is compared at 1e-12 × scale rather than
  bitwise.
  - Why: WSIMOD sums it in list order across directions, while noodl keeps its rows by
    direction. The rows themselves, which are state, are bitwise.
  - Cost if wrong: none to the values.
- **X25, quantity-only runs keep the decay structure.** That means the `DecayQueueTank`'s end of
  day and the temperature forcing; the values are listed as ignored.
  - Why: WSIMOD's volumes depend on the structure, not on the values (W10).
  - Cost if wrong: a quantity-only run of a decaying `QueueGroundwater` differs from WSIMOD in
    timing.
- **X26, the state keys and labels.** They are as tabled under State:
  - `"wsimod.total_decayed"` rows `(node, "tank")` and `(node, "surface:<name>")`;
  - `"wsimod.river_temperature_lag"` and `"wsimod.river_phosphorus_lag"` rows `(river, i)`,
    with slot 0 the oldest.
  - Why: 3a's labelling (D10).
  - Cost if wrong: a state-format change before release.
