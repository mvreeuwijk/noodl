# WSIMOD core and water quantity in `apps.wsimod`

Status: designed 2026-11-11 under the user's standing instruction ("do not wait for me and make
sensible decisions"). Reviewed by Fable in place of the user (W7). The verdict was "approve with
changes", no ruling was overturned, and this revision applies every change (W22). Rulings W1–W22
are binding and are cited by number where they apply. W8–W21 were this spec's open questions,
and the review accepted each as recommended. The document covers sub-project 3a of 3 of the
WSIMOD port (W1). 3b (in-water processes) and 3c (land) get their own specs.

## Goal

`apps.wsimod` runs WSIMOD 0.8.1's daily simulation natively in noodl: batched, in float64
tensors, differentiable, with no WSIMOD import at runtime. It reads WSIMOD's own model
description (W5). For every class in scope it reproduces WSIMOD's water volumes, storages and
arc flows, and it carries pollutants without changing them: additive pollutants as mass,
non-additive ones blended by volume. Success is measured against live WSIMOD 0.8.1 in three
ways: node by node on replayed inputs, on the quickstart and oxford demos, and on purpose-built
cases (Verification).

Authority: the installed WSIMOD 0.8.1 source, `.venv/Lib/site-packages/wsimod/`. Every
`file:line` below is relative to that directory.

## Constraints

- Branch `wsimod-core` from `main`, one PR, a Fable review before it is merged (W7). Never push
  from an agent session without the user.
- Physical names and no aliases. Code, tests and docs change together. No wall-clock
  assertions. `ruff check .` passes.
- Every transcribed behaviour cites its WSIMOD `file:line` in a docstring or comment.
- WSIMOD stays a test-only reference (the `reference` extra, pinned `==0.8.1`). A test proves
  that `import noodl.apps.wsimod` and a full run never import `wsimod`.
- `torch.float64` throughout.
- WSIMOD's values are transcribed, not reformulated (W2). Where this spec and the WSIMOD source
  disagree, the source wins and the spec is corrected.

## Scope (W1)

### In: every class 3a ports

| WSIMOD class (file:lines) | What 3a ports |
|---|---|
| `WSIObj` VQIP algebra (`core/core.py:13-368`) | `sum_vqip`, `extract_vqip`, `v_change_vqip`, `v_distill_vqip`, `ds_vqip`, `copy`/`empty` |
| `constants` (`core/constants.py`) | `FLOAT_ACCURACY` (per model, from the config), `MAXITER = 5`, `UNBOUNDED_CAPACITY = 1e15`, default pollutant lists |
| `Model.run` (`orchestration/model.py:893-1200`) | the daily order, the river order, the recording, `end_timestep` |
| `Model.load` and the methods it calls (`model.py:216-348`, `:642-834`) | the reader (W5) |
| `to_datetime` (`model.py:41-170`) | date parsing |
| `Node` (`nodes/nodes.py:15-646`) | handlers, `pull/push_distributed`, `get_connected`, `check_basic`, deny/accept |
| `Arc`, `QueueArc`, `AltQueueArc`, `PullArc`, `PushArc`, `SewerArc`, `WeirArc` (`arcs/arcs.py:18-665`, `:808-905`) | all of each |
| `Tank`, `ResidenceTank`, `QueueTank` (`nodes/tanks.py:10-400`, `:460-642`) | all of each |
| `DecayTank` (`tanks.py:403-457`) | only as `Surface`'s base, with empty `decays` |
| `Storage`, `Groundwater`, `QueueGroundwater`, `Reservoir`, `RiverReservoir` (`nodes/storage.py`) | all of each |
| `River` (`storage.py:452-955`) | everything except `biochemical_processes` (`:738-898`) |
| `Sewer` (`nodes/sewer.py:14-247`) | all; foul sewers by name (`model.py:658-660`) |
| `WTW`, `WWTW`, `FWTW` (`nodes/wtw.py`) | volumes, capacities, storm and service tanks, liquor and sludge. Pollutants only in the neutral split (Representation) |
| `Demand`, `ResidentialDemand`, `NonResidentialDemand` (`nodes/demand.py`) | all, including the garden-demand query and push. `NonResidentialDemand` per W4 |
| `Distribution`, `UnlimitedDistribution` (`nodes/distribution.py`) | all, including the leakage decorators |
| `Catchment` (`nodes/catchment.py`), `Waste` (`nodes/waste.py`) | all |
| `Land` (`nodes/land.py:17-300`) | all. `apply_irrigation` only calls the no-op `lambda: None` (`land.py:97`), because irrigation surfaces are 3c. The demand side of garden and irrigation water is 3a. `GardenSurface` and `IrrigationSurface` are `GrowingSurface`s and so belong to 3c; without them, `Land` answers zero (W16) |
| `Surface`, `ImperviousSurface`, `PerviousSurface` (`land.py:303-1079`) | water balance (IHACRES, pores, routing). The deposition inflows and soil-temperature process only in neutral form |

### Out

- **3b (in-water processes):** `DecayObj` (`core.py:432-549`), `DecayArc`, `DecayArcAlt`,
  `DecayQueueTank`, any `decays` dict on the `Storage` family (`{}` included; see Nodes ›
  Storage), `River.biochemical_processes`, WTW pollutant
  treatment (`process_parameters`, per-pollutant `liquor_multiplier`), and the soil temperature
  `PerviousSurface.calculate_soil_temperature` (`land.py:1060-1079`).
- **3c (land):** `GrowingSurface`, `IrrigationSurface`, `GardenSurface`, `VariableAreaSurface`
  (`land.py:1082-2461`; their water balance depends on the crop calendar, `:1514`),
  `NutrientPool`/`NutrientStore` (`nodes/nutrient_pool.py`), and surface deposition
  (`Surface.simple_deposition`, `atmospheric_deposition`, `precipitation_deposition`,
  `land.py:505-581`).
- **IO and CLI:** `Model.save`, pickling, `csv2yaml`/`yaml2csv`, the unified parquet file
  (`model.py:261-322`, `:350-640`, `:1432-1571`), `validation.py`, `__main__.py`.
- **Extensions:** `extensions.py`, the `extensions:` key (`model.py:248-249`), and runtime
  class renaming or monkey-patching by users (W5).
- **Never ported:** `Model.debug_node_mb`, `change_runoff_coefficient`, `reinit`, the
  `objectives` callbacks of `run`, `general_distribute` (dead code, `nodes.py:649-713`), and
  `WSIObj.mass_balance` printing. A noodl mass balance replaces the last (Diagnostics).
- **Refused by name (W4):** `EnfieldFoulSewer` (`sewer.py:250-310`).

Each out-of-scope class or key is refused by name when a configuration uses it. The message
names the node, the class or key, and the sub-project that will port it. `pollutants=False` is
the one exception (Loading).

## Approach: WSIMOD's procedure, transcribed and batched (W2)

noodl runs the same calls in the same order as WSIMOD:

- the orchestration list;
- within each slot, nodes in configuration order;
- within each method, WSIMOD's own sequence of arc requests, checks and tank updates.

Each node is a Python object of a transcribed class with WSIMOD's handler dictionaries keyed by
tag. The values the objects hold are batched tensors.

**The structure is static; the data is batched.** Which method calls which, on which arcs, with
which tag and `of_type`, depends only on the model's structure. noodl's Python control flow
follows that structure and nothing else. Every branch that depends on data in WSIMOD, such as
`if x > FLOAT_ACCURACY`, a loop's exit condition, or `storage == 0`, is evaluated per instance
as a boolean tensor and applied with `torch.where`. Three rules make this exact:

1. **Masks.** Every transcribed method takes the caller's `active` mask, a bool tensor over the
   batch shape. A branch ANDs its own condition into the mask for the writes it makes. Every
   state write, to tank storages, arc accumulators and queues, is `where(active, new, old)`.
   A call that WSIMOD would not make for an instance is computed and discarded for that
   instance. So it is a no-op whatever its arguments are, including zero volumes.
2. **Loops.** `while` loops bounded by `MAXITER` (`nodes.py:386`, `:453`) run exactly
   `MAXITER` rounds. A round runs under `active & condition`, so an instance that WSIMOD would
   have left the loop for leaves every later round unchanged.
3. **Safe denominators.** A division WSIMOD guards with a branch, or reaches only in a branch,
   divides by `where(den != 0, den, 1)`, and the branch's mask removes the result. No NaN or
   infinity enters a value or its gradient. This is the double-`where` pattern.

**Ties.** `min(a, b)` and `max(a, b)` are `where` on a detached comparison with Python's tie
rule: `min` returns `a` unless `b < a`, and `max` returns `a` unless `b > a`. Values stay in
the autograd graph.

**Reductions run in WSIMOD's order.** Where WSIMOD loops over arcs or slots and adds, noodl adds
in that order, one term at a time. It never uses `torch.sum` over an axis WSIMOD walks
sequentially, so the rounding follows WSIMOD's.

**Unbounded is the finite 1e15.** "Unbounded" is WSIMOD's `UNBOUNDED_CAPACITY = 1e15`
(`constants.py:18`), never `inf`. WSIMOD sizes its preference shares from sums that include
1e15 (`get_connected`, `nodes.py:307-315`).

**Check recursion and cycles.** WSIMOD's checks recurse depth-first, and noodl recurses the
same way. Cycles of arcs are legal and ordinary: `Land → Sewer` runoff together with
`Sewer → Land` ponding (`sewer.py:233-237`) is WSIMOD's own pattern. Only handlers that call
`get_direction_arcs` or `get_connected` recurse, so noodl builds the recursion graph explicitly,
per direction and tag.

- **Push edges.** An arc is a recursion edge for (push, tag) when its `out_port`'s push-check
  handler for that tag is one of:
  - `Node.push_check_basic` (`nodes.py:68-72`, `of_type` [Node, River, Waste, Reservoir]);
  - `RiverReservoir.push_check_river_reservoir` (`storage.py:1086-1110`, `of_type` [Node,
    River, Waste]).
- **Pull edges.** An arc is a recursion edge for (pull, tag) when its `in_port`'s pull-check
  handler is one of:
  - `Node.pull_check_basic` (`nodes.py:67`, over all in-arcs). This is the default that
    `Node`, `Sewer`, `Land`, `WTW` and `WWTW` keep for pulls without a handler of their own.
    `WWTW` overrides it with `pull_check_reuse` (`wtw.py:274`), which terminates.
  - `Distribution`'s leakage-decorated `pull_check_basic` (`distribution.py:132-139`).
  - `River.pull_check_river` (`storage.py:660-686`, `of_type` [River, Node]).
- **Terminating handlers.** Every other check handler terminates. These are the tanks, `Waste`,
  `Catchment`, `Demand`, `FWTW`, `UnlimitedDistribution`, `QueueGroundwater`, the `Sewer` push
  check, the `River` push check and `Land`.

Only a cycle in this graph is refused by name at build time, naming the cycle. WSIMOD does not
fail cleanly on such a cycle. `query_handler` (`nodes.py:335-343`) catches the exception and
calls the handler again at every level, so the work doubles per level and the run hangs. Fable
confirmed this by running it: a two-`Node` `push_check` cycle did not finish in 120 s.

## Representation

### VQIP

A VQIP is a tensor `(..., 1 + K)`. Column 0 is the volume, and columns `1..K` hold the model's
pollutants in configuration order. `...` is the batch shape.

- Additive pollutants hold total mass and non-additive ones hold the intensive value, exactly as
  WSIMOD stores them (`core.py:56-76` comment). Two boolean masks of length `K`, `additive` and
  `non_additive`, select the columns.
- The operations transcribe `core.py` column-wise:
  - `sum_vqip` (`:83-116`): volume and additive columns add. Non-additive ones become
    `(t2·V2 + t1·V1)/V` when `V > 0` (that operand order), otherwise they keep `t1`'s values.
  - `extract_vqip` (`:150-178`): volume and additive columns subtract. Non-additive ones keep
    `t1`'s values.
  - `v_change_vqip` (`:251-284`): when `V > 0`, `ratio = v/V`, and both the volume and the
    additive columns are multiplied by `ratio`. The volume becomes `V·(v/V)`, not `v`, as
    WSIMOD computes it. When `V = 0` only the volume is set to `v`.
  - `v_distill_vqip` (`:209-223`): subtracts volume only.
  - `ds_vqip` (`:305-328`).
- `K = 0` is allowed and is the quantity-only model (`pollutants=False`).

### The module-global pollutant lists

WSIMOD keeps `POLLUTANTS`, `ADDITIVE_POLLUTANTS` and `NON_ADDITIVE_POLLUTANTS` as module
globals (`constants.py:21-50`), replaced by `Model.load` (`model.py:232-238`) and by
`create_oxford_model` (`demo/create_oxford.py:172-180`). The same applies to `FLOAT_ACCURACY`
(`model.py:239-241`).

In noodl these are per-model fields of the description. The defaults are WSIMOD's 12-pollutant
lists and `1e-11`. Two models with different pollutant sets can coexist in one process.

The reader refuses inconsistent lists by name: every pollutant must be in exactly one of the
two lists, and the lists may name nothing else. WSIMOD would `KeyError`, or silently neither
add nor blend such a pollutant.

Some code writes `temperature` into a VQIP when temperature is not a pollutant (`land.py:685`,
`:991`, `:1028`; `demand.py:320`). WSIMOD carries that key but never reads it, so noodl drops
the write.

### Pollutants in 3a are carried without change

Every in-scope node and arc moves, mixes, splits and stores pollutants with the VQIP algebra
above. Pollutants enter only from these sources:

- `Catchment` series (`catchment.py:59-74`: the data's concentration × flow for additive ones);
- `Demand` and `ResidentialDemand` `pollutant_load` (`demand.py:121-131`, `:303-324`, including
  the mixed foul temperature `T_air·(1 − w) + T_const·w`);
- the air temperature given to rainfall (`land.py:685`, `:991`, `:1028`);
- a store's initial VQIP;
- `FWTW`'s deficit water, which takes the quality of the previous day's pulled water
  (`wtw.py:588-590`, `:650`).

With pollutants on, a process that would change a pollutant inside a water body is accepted
only in its neutral form, in which its effect is identity. Any other form is refused by name,
naming the sub-project that will port it:

- **WTW pollutants (3b).** For every additive pollutant, `process_parameters[p]` must be
  `{constant: c_v, exponent: 1}`, with `c_v = 1 − percent_solids − liquor_multiplier["volume"]`
  (`wtw.py:104-110`), and `liquor_multiplier[p]` must equal `liquor_multiplier["volume"]`. Each
  pollutant then splits into treated, liquor and solids in the same proportions as the volume
  (`wtw.py:163-214`). WSIMOD's defaults (`:73-85`: constant 0.01, multiplier 0.7) are not
  neutral, and they are refused.
  Two further rules apply only with pollutants on. Volume-only runs (`K = 0`) are unaffected.
  - `treat_current_input` reads `influent["temperature"]` for every additive pollutant
    (`wtw.py:191-193`). A model with additive pollutants but no `temperature` pollutant is
    therefore refused by name, where WSIMOD raises `KeyError`.
  - The non-additive liquor blend divides by `V_liq + V_in·m_v` (`wtw.py:174-182`). That sum is
    0 on a day with no input and no previous liquor, and on every day when
    `liquor_multiplier["volume"] = 0`. WSIMOD then raises `ZeroDivisionError`; Fable ran it on
    `FWTW.treat_water` and `WWTW.calculate_discharge` with zero input on day 1. noodl keeps the
    previous liquor value through a double `where`, and records the event (defect table).
- **Surface deposition (3c).** `pollutant_load` must be zero for every pollutant, with no
  non-additive entries. The surface's own `data_input_dict` must hold no `nhx-dry`-style keys
  (`land.py:375-380`).
  "Neutral" means an identity only to rounding for the non-additive pollutants, and the same
  holds for the dense register's absent keys (below). `sum_vqip` recomputes `(t·V)/V`
  (`core.py:111-114`), which can move a value by one ulp. The 1e-12 bound absorbs this, so no
  test asserts these steps exactly.
- **Soil temperature (3b).** When `temperature` is a pollutant, `PerviousSurface` must have
  `soil_temp_w_air = soil_temp_w_deep = 0` and `soil_temp_w_prev > 0` (`land.py:1060-1079`).
- **River biochemistry (3b).** `River` with `"nitrate"` among the pollutants runs biochemistry
  (`storage.py:917-924`), so it is refused. Without nitrate, WSIMOD skips biochemistry and so
  does noodl.
- **Decay (3b).** `decays` must be `None` on the `Storage` family; a dict, even `{}`, is
  refused (Nodes › Storage). On `Surface`, `decays` must be empty. Arcs may not carry `decays`,
  which belongs to the `Decay*` arc classes of 3b.

**Volumes never depend on pollutant values in the classes of 3a.** The plan confirms this by
the replay tests. With `pollutants=False` the model is quantity-only (`K = 0`). Pollutant
parameters are then not read, the run's volumes equal WSIMOD's whatever those parameters are,
and the ignored parameters are listed on the model (`model.ignored_by_quantity_only`). This is
how the demos, with their non-neutral WTW defaults and surface loads, run unchanged in 3a
(W10). Until 3b, real configurations need `pollutants=False`.

### Node and arc state, and the state dict

The state carried from one day to the next is a dict of tensors. Each key has a leading batch
shape, and the model's labels give each row a name.

| Key | Shape | What it holds (WSIMOD attribute) |
|---|---|---|
| `"wsimod.storage"` | (..., n_tanks, 1+K) | every `Tank`/`ResidenceTank`/`QueueTank` `.storage` and every surface's `.storage`, labelled `(node, attribute)` in WSIMOD's attribute names: `tank`, `sewer_tank`, `stormwater_tank`, `service_reservoir_tank`, `surface_runoff`, `subsurface_runoff`, `percolation`, `surface:<name>` |
| `"wsimod.active_storage"` | (..., n_queue_tanks, 1+K) | `QueueTank.active_storage` (`tanks.py:478`) |
| `"wsimod.travel_queue"` | (..., n_alt_queues, L+1, 1+K) | the `{time: vqip}` queue of every `AltQueueArc` as a dense shift register: each `QueueTank`'s internal arc, and each user `AltQueueArc`. `L` is the largest travel time the structure can enter (pipe time, time-area keys, `number_of_timesteps`). A user `AltQueueArc` shifts at `end_timestep` but is flushed only by its next push, because nothing calls its `update_queue` at the end of the day (`arcs.py:603-660`). A `QueueTank`'s arc is also flushed at the end of the day (`tanks.py:633-634`) |
| `"wsimod.travel_queue_present"` | (..., n_alt_queues, L+1) | each register's key set (bool; Shift register below): WSIMOD assigns a first entry and sums later ones, which differ by rounding |
| `"wsimod.arc_queue"` | (..., n_rows, 1+K) | `QueueArc` request rows (below), labelled `(arc, direction, i, j)` |
| `"wsimod.arc_queue_since"` | (..., n_anchors) | per `(arc, direction)`, the days since the arc's last call in that direction that reached `update_queue`, saturating at `n` (below); `n` at the start |
| `"wsimod.liquor"` | (..., n_wtw, 1+K) | `WTW.liquor` of every `WWTW` and `FWTW`, empty at the start. Neither `end_timestep` resets it (`wtw.py:447-453`, `:646-653`), and every treatment blends with it (`:174-182`). For a `WWTW` it is also an inflow to the next day (`:340`). For an `FWTW` only its non-additive entries and volume matter. `solids` needs no state, because each treatment overwrites it completely (`:204-208`) |
| `"wsimod.previous_pulled"` | (..., n_fwtw, 1+K) | `FWTW.previous_pulled` (`wtw.py:650`) |
| `"wsimod.treated"` | (..., n_wwtw, 1+K) | `WWTW.treated`. Only `make_discharge` empties it (`wtw.py:346`). `end_timestep_` does not (`:447-453`), so a custom orchestration without `WWTW.make_discharge` lets it accumulate from day to day |

Arc accumulators (`flow_in`, `flow_out`, `vqip_in`, `vqip_out`), the WTW per-day stocks and the
surfaces' per-day flows reset every day (`arcs.py:259-265`, `wtw.py:216-220`, `:447-453`,
`:646-653`). They live only inside a day.

`Tank.storage_` and the other lagged copies exist only for WSIMOD's mass balance and are not
ported.

**Shift register.** `AltQueueArc.end_timestep` (`arcs.py:640-660`) is a shift:
`new[i] = old[i+1]` for `1 ≤ i < L`, and `new[0] = sum_vqip(old[0], old[1])` in that order. A
key absent from WSIMOD's dict is an empty VQIP in the register. A first entry under an absent
key is assigned (`arcs.py:599-601`), a later one summed (`:595-598`), and `sum_vqip(empty, v)`
differs from `v` by rounding in the non-additive columns (`(c·V + 0)/V`). So each register
carries WSIMOD's key set as a bool `present` (..., L+1): keys 0 and 1 exist from the start
(`:567`), an entry adds its key, the shift adds key `i` when key `i+1` exists (`:655-658`), and
no key is removed. With it the register equals WSIMOD's dict bitwise (`test_tanks.py`); without
it the temperature column differed in the last bit within days.

**`QueueArc` slots.** WSIMOD keeps `QueueArc.queue` as a Python list of requests
(`arcs.py:290`, `:432-446`). Each request is delivered one at a time, in list order, at the
next `send_push_request` on that arc after its `time` reaches 0 (`update_queue`,
`arcs.py:448-528`; `end_timestep` only decrements, `:544-545`). noodl keeps fixed rows indexed by
`(i, j)`, with `n = number_of_timesteps`: `i ∈ [0, n−1]` is the request's entry day counted back
from the arc's last call (in that direction) that reached `update_queue`, and `j` is the
request call's index in the arc's static daily call order, which merges the call site and the
`push_distributed` round in their static order (WSIMOD's list order). A per-(arc, direction)
counter `since` holds the days since that call, saturating at `n`.

An earlier revision indexed the slots by (entry day modulo `n + 1`, site, round). That cannot
tell a request's age once it waits more than `n + 1` days for a push: with `n = 2` and pushes
on days 1 and 5, day 1's request (age 4, due) shares its residue with day 4 (age 1, not due).
The anchored rows are exact: after a call on day `d0` that reached `update_queue`, every
request still waiting was entered on a day `e` with `e + n > d0`, because due requests are
delivered or returned as backflow (`backflow_enabled` is always true on these paths,
`arcs.py:393-396`, `:466-512`) and sub-`FLOAT_ACCURACY` ones are removed.

- **Why the round index.** One call site can send up to `MAXITER` requests to the same arc on
  one day. Each round of `push_distributed` (`nodes.py:453-473`) is its own
  `send_push_request`, and so its own list entry, with its own tag and concentration.
  `update_queue` delivers and clips the entries one at a time (`arcs.py:479-512`), and each
  rejected part keeps its own request's concentration, which merging would lose. Rounds beyond
  the first really happen on a `QueueArc`: the backflow of due requests is added to the
  rejection (`:399-400`), so `sent = extract(to_send, reply)` goes negative, `not_pushed_`
  grows, and the loop polls again.
- **Delivery order.** A row is due at a later call iff `since + i ≥ n`. Due rows are delivered
  in the order `i = n−1 … 0`, then `j`, which is WSIMOD's list order. After delivery the
  waiting rows move to `i + since` (a masked sum over `i`, static operations) and `since`
  becomes 0. A row's `average_flow` is `V/(n+1)`, because a waiting row's volume never changes.
- **Absence and the update-time drop.** A row with zero volume is absent. WSIMOD removes a
  waiting request below `FLOAT_ACCURACY` on update (`:483-485`). This is reachable, because the
  clip at entry (`:409-412`) can leave a request between 0 and `FLOAT_ACCURACY`, which
  `enter_arc` has already added to `vqip_in`. noodl removes such rows and records their water
  as dropped.
- **The early return.** A push below `FLOAT_ACCURACY` returns before both `enter_queue` and
  `update_queue` (`:378-379`). On such a call nothing is enqueued and nothing due is delivered.
  Fable ran it: a tiny push on a due day left the due 10 units queued with `vqip_out` 0, and the
  next real push delivered them. noodl reproduces this by masking that call's delivery with
  `V ≥ FLOAT_ACCURACY` of the triggering push (defect table).
- **Bound.** The number of rows is `n·J` per direction, bounded by the structure. Pushes that
  returned early do not reset `since`. A pull always reaches `update_queue` (no early return),
  so `since` for pulls resets on every pull call.
- **Order of enqueue and delivery.** noodl delivers the waiting requests in list order before
  it enqueues the new one. For `n ≥ 1` this matches WSIMOD's "enqueue then deliver" order
  (`:393-399`), because the new request cannot be due. For `n = 0`, noodl delivers the new
  request last.
- **Directions.** Push and pull requests have separate rows and counters (`direction`, `:480`).

The W21 test proves this layout against WSIMOD's delivery order. It drives the arc with pushes
on irregular days, with several pushes and several distribute rounds on one day, and with a
tiny push on a due day, and with a gap between pushes longer than `n + 1` days.

### Batches

Every driver and state key may carry leading batch dimensions, and they broadcast. Each
instance follows its own WSIMOD path through the shared call structure. A batch of B instances
equals B single runs to rounding; the operations are elementwise, so they are expected to be
bitwise equal. Integer and structural inputs cannot be batched:

- topology, `of_type`, tags, the orchestration;
- travel times and time-area keys;
- the pollutant lists;
- `FLOAT_ACCURACY`;
- whether a `Distribution` is decorated. WSIMOD wraps the pull handlers only when the configured
  `leakage > 0` (`distribution.py:132-139`). noodl decorates exactly when the configured value
  is positive. `leakage` stays a learnable, batchable driver within `(0, 1)` on a decorated node.
  A batch that would mix zero and non-zero leakage on one node is refused by name, and so is
  `leakage ≥ 1`, which divides by zero or reverses the request (`distribution.py:40`).

Real-valued parameters can be batched, including time-area fractions.

## The daily driver

### Order of a day (`model.py:993-1176`)

For each date `d` of the run:

1. Every node's clock is set to `d` (`:995-997`). In noodl this is the day index into the
   forcing tensors.
2. The orchestration list is walked (`:1000-1003`). The default list is `model.py:191-205`:
   1. `FWTW.treat_water`
   2. `Demand.create_demand`
   3. `Land.run`
   4. `Groundwater.infiltrate`
   5. `Sewer.make_discharge`
   6. `Foul.make_discharge`
   7. `WWTW.calculate_discharge`
   8. `Groundwater.distribute`
   9. `River.calculate_discharge`
   10. `Reservoir.make_abstractions`
   11. `Land.apply_irrigation`
   12. `WWTW.make_discharge`
   13. `Catchment.route`

   A configuration's `orchestration` key replaces it (`:251-253`).

   Each slot's key is a **configuration type key**, not a class:
   - It is the node's `type_` as written (`add_nodes`, `:650-673`).
   - A node whose name contains `"foul"` gets the key `"Foul"` (`:658-660`).
   - `node_type_override` chooses the class but not the key (`:653-657`).

   So a node written `type_: ResidentialDemand`, without the override, never runs
   `create_demand` under the default list. noodl transcribes this. Nodes of one key run in
   configuration order.
3. `River.distribute` runs for each name in the river order (`:1006-1007`). The order is
   computed from the arcs whose ends are both of WSIMOD type River, Node, Waste or Reservoir
   (`add_arcs`, `:697-735`). `assign_upstream` (`:780-807`) gives Waste nodes 0 and gives each
   upstream layer one more. Rivers are then taken most upstream first. Three details are
   transcribed:
   - A river that no river path links to a Waste is left out of the order and never
     distributes, as in WSIMOD.
   - Only names under the type key `"River"` enter the order (`:730-735`). A `River` written
     under another `type_`, or one whose name contains `"foul"`, is left out.
   - WSIMOD iterates a `set` (`:800`), so rivers at the same level come in hash order. noodl
     breaks the tie by configuration order and documents it. The parity cases keep tied rivers
     independent (W13).
4. Recording (below), at the state the day has reached (`:1063-1170`).
5. `end_timestep` on every node in configuration order, then on every arc (`:1172-1176`). The
   per-class `end_timestep` behaviours are transcribed, including:
   - `QueueTank._end_timestep`: the internal arc shifts, then `update_queue` moves the arrived
     water into `active_storage` (`tanks.py:631-635`);
   - `QueueArc`: times are decremented (`arcs.py:544-545`);
   - `FWTW`: `previous_pulled = total_pulled` (`wtw.py:650`);
   - `WWTW`: the liquor is kept (`wtw.py:447-453`);
   - `RiverReservoir`: `total_environmental_satisfied` is reset (`storage.py:1130-1133`).

   Surfaces are `DecayTank`s. Their decay-at-end-of-day (`tanks.py:442-447`) reads the parent's
   `("temperature", d)` (`core.py:485-487`) even with empty `decays`. So a `Land` needs
   temperature forcing on every day, as in WSIMOD (Loading). The decay itself is an identity.

A date list with gaps is still one step per listed date, as in WSIMOD. The step is always one
day (`DT_DAYS = 1`, `constants.py:20`).

### Methods an orchestration slot may name

The transcribed zero-argument step methods:

| Classes | Methods |
|---|---|
| `FWTW` | `treat_water` |
| `Demand` family | `create_demand` |
| `Land` | `run`, `apply_irrigation` |
| `Groundwater`, `QueueGroundwater` | `infiltrate`, `distribute` |
| `Storage`, `River`, `Reservoir`, `RiverReservoir` | `distribute` |
| `Sewer`, `WWTW` | `make_discharge` |
| `WWTW` | `calculate_discharge` |
| `River` | `calculate_discharge` (a no-op without biochemistry) |
| `Reservoir`, `RiverReservoir` | `make_abstractions` |
| `RiverReservoir` | `satisfy_environmental` (not in the default list, so environmental flow happens only when a configuration names it) |
| `Catchment` | `route` |

A slot whose key names nodes lacking the method is refused by name. WSIMOD raises
`AttributeError` there mid-run. A key that names no node is skipped, as WSIMOD does with
`nodes_type.get(...)`.

### Flushing a `QueueTank` on push

`QueueTank.push_storage` (`tanks.py:513-539`) calls `internal_arc.send_push_request`. After
`enter_queue`, that calls `update_queue` (`arcs.py:393-399`), which pushes `queue[0]` into
`active_storage` at once. So:

- Water entered with total time 0 is active at once. Examples are a `Sewer` with
  `pipe_time = 0`, a time-area key 0, and a `QueueGroundwater` key 0.
- That water is discharged by the same day's `make_discharge` or `distribute`.
- Every push also flushes whatever is already in `queue[0]`.

This decides the demos' sewer volumes. `pull_ponded` on a `QueueTank` is `Tank.pull_ponded`
(`tanks.py:78-100`) calling `QueueTank.pull_storage` (`:541-564`), so the ponded volume
`max(storage − capacity, 0)` is computed on the total storage but limited to `active_storage`.

### The pull/push protocol (`nodes.py`, `arcs.py`)

- **Handlers.** Each node has four dicts keyed by tag: `pull_set`, `push_set`, `pull_check`,
  `push_check` (`nodes.py:61-72`). Each class overrides entries exactly where WSIMOD does, and
  the tables are cited per class below. A `*_set` push returns the VQIP **not** accepted; a pull
  returns the VQIP **obtained**. A check changes no state. A missing tag raises by name. WSIMOD
  prints, retries the call and then raises (`query_handler`, `:319-343`).
- **Arc push** (`send_push_request`, `arcs.py:112-158`). Without `force`:
  `excess = get_excess` (`:230-257`), that is `min(capacity − flow_in, out_port.push_check(v,
  tag).volume)`, with the reply carrying the node check's concentration. The arc then sends
  `extract(v, v_change(v, max(V − excess, 0)))` to `out_port.push_set`. It accumulates
  `flow_in += sent volume` and `vqip_in = sum(vqip_in, sent)`, sets `vqip_out = vqip_in`, and
  returns the rejection plus the clip. `force` skips the clip.
- **Arc pull** (`send_pull_request`, `arcs.py:160-197`). The volume is clipped to the excess,
  and only when the clipped volume is positive are the request's additive entries (if any)
  rescaled. The arc calls `in_port.pull_set`, and the reply is accumulated.
- **First come, first served.** `flow_in` accumulates over every event of the day and resets
  only in `end_timestep`. Capacity is therefore served in call order.
- **Directional arcs.** `PullArc` and `PushArc` deny one direction (`arcs.py:808-893`).
  `SewerArc` and `WeirArc` are plain arcs (`:896-905`).
- **`get_connected`** (`nodes.py:271-317`). It loops over the arcs of `of_type` (`get_direction_arcs`, `:218-269`; a
  string `of_type` is a one-element list, which is equivalent to WSIMOD at every call site in
  scope). For each arc it computes `avail = send_*_check(tag).volume`, zeroed below
  `FLOAT_ACCURACY`, then `allocation[arc] = avail·preference`, `priority = Σ allocation` and
  `capacity[arc] = avail`.
- **`push_distributed`** (`nodes.py:409-478`).
  - With exactly one out-arc, it pushes straight through that arc, or returns the VQIP
    unpushed when the arc's type is not in `of_type`. There is no check and no loop
    (`:424-438`).
  - Otherwise, if the request exceeds `avail`, the **first** round's shares switch to
    `capacity`. This happens only before the loop (`:446-450`); every later re-poll restores
    preference shares.
  - Then up to `MAXITER` rounds, while `not_pushed > FLOAT_ACCURACY` and
    `avail > FLOAT_ACCURACY`:
    1. `amount = min(avail, not_pushed)`.
    2. Each arc, in order, is sent `v_change(not_pushed_, amount·alloc/priority)`. What it
       accepts is extracted from `not_pushed_`.
    3. Re-poll with `get_connected`.
- **`pull_distributed`** (`nodes.py:345-407`). The single-arc case is the same. Otherwise, up to
  `MAXITER` rounds while `deficit > FLOAT_ACCURACY` and `avail > FLOAT_ACCURACY`:
  1. Each arc is asked for `{volume: deficit·alloc/priority}`. The deficit stays fixed within
     the round.
  2. The replies are summed in order.
  3. `deficit = request − pulled`, then re-poll.
- **Defaults of a plain `Node`** (`nodes.py:61-72`). Push to `["Node", "River", "Waste",
  "Reservoir"]`. Pulls are `pull_distributed` over all in-arcs. Checks are `check_basic`: the
  ordered `sum_vqip` of the neighbour checks, then `v_change` to `min(avail, request)`.
- **A check clips only when it is given a request.** `check_basic` clips only `if vqip is not
  None` (`nodes.py:505-506`). `get_connected` (`:308`) and the recursion itself (`:503`) call
  `send_*_check(tag=tag)` with no VQIP. So `Arc.get_excess(vqip=None)` calls `push_check(None,
  tag)` or `pull_check(None, tag)`, and the reply is unclipped. Only `send_*_request` passes the
  request (`arcs.py:134`, `:176`). The same two cases appear in:
  - `Tank.get_excess` and `get_avail` (`tanks.py:123-130`, `:151-158`);
  - `Sewer.push_check_sewer` (`sewer.py:156-161`) and `WWTW.push_check_sewer`
    (`wtw.py:367-372`);
  - `RiverReservoir.push_check_river_reservoir` (`storage.py:1105-1106`);
  - `QueueGroundwater.pull_check_active` (`:380-384`) and `River.pull_check_river`
    (`:681-684`);
  - `Waste`'s `push_check_accept` (`nodes.py:620-623`);
  - `Catchment.pull_check_abstraction`, which tests `if vqip:` (`catchment.py:116-117`).

  noodl transcribes the cases separately; the `None` case has no request argument.
- **A forced push can make the pipe excess negative.** `force` skips the clip but still adds to
  `flow_in` (`arcs.py:131-152`). `pipe_excess = capacity − flow_in` can then go negative, and
  `get_excess` returns a negative volume. `get_connected` zeroes it (`nodes.py:309-310`), but
  `check_basic` sums it as it is (`:502-503`). This is transcribed, not clamped.
- **Type of an arc's end.** An arc's end has the WSIMOD class name **after** WSIMOD's renames:
  `QueueGroundwater` → `Groundwater` (`storage.py:294`), `RiverReservoir` → `Reservoir`
  (`:1043`), and `ResidentialDemand` → `Demand` (`demand.py:214`). `NonResidentialDemand` keeps
  its own name. `of_type` filters and the river order use these names.
- **No zero preferences.** A node with two or more arcs of one direction whose preferences are
  all zero is refused by name, for the directions in which its class distributes (push and pull
  for `Node`, push for `Catchment`; a `Waste` distributes neither). WSIMOD divides by
  `priority = 0` when such a loop runs. Negative preferences are refused too. Some zero
  preferences beside full positive-preference arcs give `priority = 0` only in some states; that
  is an event, not a refusal (defect table, `"zero_priority"`).
- **Request mutation.** WSIMOD mutates request dicts in place (`arcs.py:180-185`;
  `distribution.py:40`, `:91`). One mutated dict is read again.
  1. `decorate_leakage_check` divides the request's volume by `1 − leakage` inside
     `Arc.get_excess` (`distribution.py:90-91`).
  2. `Arc.send_pull_request` then rescales the request's additive entries by
     `volume / vqip["volume"]` with that inflated denominator (`arcs.py:181-183`). Fable ran it:
     a request of volume 10 and phosphate 7 left with volume 20 and phosphate 3.5.

  The effect is confined to the additive entries of a pull request, and no in-scope `pull_set`
  reads those. `Tank.pull_storage`, `Catchment`, `FWTW`, `River`, `QueueGroundwater` and
  `UnlimitedDistribution` all use the volume only. So the functional transcription, which
  passes volumes, is equivalent. The replay harness deep-copies arguments at call entry, so it
  compares the requests as they were before mutation.

### Entry points (`water_run` style)

- `read_wsimod_config(source, *, config_name="config.yml", overrides=None) -> WsimodNetwork`.
  `source` is a directory, read as `Model.load(address, config_name, overrides)`, or the parsed
  mapping. `overrides` replaces top-level keys as WSIMOD's does (`model.py:229-230`).
  `WsimodNetwork` is a plain, editable description: pollutants and their masks,
  `float_accuracy`, orchestration, dates, nodes (name, class, slot key, parameters, surfaces)
  and arcs, in configuration order. Its forcing tables are keyed by variable and date. The
  config's `overrides` section is already applied.
- `build_model(network, *, pollutants=True) -> (model, state, drivers)`. `model` is a noodl
  `Model` over a `Network`: WSIMOD nodes are network nodes, and arcs are edges of kind `"arc"`
  named as in the config. The day is one integrating closure, `WsimodDay` (`integrates =
  True`), that owns every state key above. `model.step(state, drivers, dt=86400, t=k·86400)`
  advances day `k`, which lets the model couple with other noodl models. Keys, units and labels
  are registered so that `model.refs`, `model.check` and the units table work. The closure sits
  on a `Model` with no layers. If `Model` refuses an empty layer set, the plan adds that support
  to `Model`, with a test, instead of creating a separate model class (W11).
- `initial_state(model, *, values=None)` and `initial_drivers(model, *, values=None)`: the
  configuration's values, then `values` by name, for example
  `initial_state(model, values={"wsimod.storage": {("farmoor", "tank"): 9e6}})`.
- `wsimod_run(model, state=None, drivers=None, *, days=None, diagnostics=None) ->
  WsimodResults`. It runs WSIMOD's loop over the configuration's dates, or the first `days` of
  them, through the same `WsimodDay` closure, and records each day.

### Inputs

- **Drivers, parameters.** Each numeric WSIMOD parameter is a driver
  `"wsimod.<Class>.<parameter>"`, shape `(..., n_nodes_of_class)` in configuration order, under
  WSIMOD's own parameter name. Examples: `"wsimod.Groundwater.residence_time"`,
  `"wsimod.Arc.capacity"`, `"wsimod.Arc.preference"`, `"wsimod.PerviousSurface.field_capacity"`
  (rows are surfaces). Time-area fractions are `(..., n, L+1)`.
- **Drivers, forcing.** The full series: `"wsimod.precipitation"`, `"wsimod.et0"` and
  `"wsimod.temperature"`, shape `(..., T, n_readers)` over the nodes that read them, plus
  `"wsimod.flow"` `(..., T, n_catchments)` and `"wsimod.concentration"`
  `(..., T, n_catchments, K)`. The closure takes day `k`'s slice, so `model.step` and
  `wsimod_run` read the same drivers.
- **State.** As tabled, from the configuration's `initial_storage` values: a number means a
  volume with zero pollutants, and a dict means a VQIP (`tanks.py:36-50`). `QueueTank` starts
  with `active_storage = storage` (`tanks.py:478`).

### Outputs: `WsimodResults`

What WSIMOD's `Model.run` records (`model.py:1063-1170`), as tensors with a leading batch
shape:

- `dates` (list of str), `arc_names`.
- `arc_flow` `(..., T, n_arcs)`: `vqip_out` volume per day. `arc_quality`
  `(..., T, n_arcs, K)` holds the other entries of `vqip_out`.
- `tank_names` `[(node, attribute)]`, `tank_storage` `(..., T, n_tanks)` and `tank_quality`.
  These cover the attributes WSIMOD's `record_all` reports: exactly the classes `Tank`,
  `QueueTank` and `ResidenceTank` (`:1119-1133`).
- `surface_names`, and `surface` (a dict of field → `(..., T, n_surfaces)`). The fields are
  `storage`, `evaporation`, `precipitation`, `percolation`, `subsurface_r`, `surface_r` and
  `tank_recharge` (`:1135-1170`). A field WSIMOD does not record for impervious surfaces is NaN
  there. `surface_quality` holds the storage's pollutant entries. Only the surfaces of nodes
  under the type key `"Land"` are recorded (`:1135`), as in WSIMOD.
- `mass_balance: WsimodMassBalance` (Diagnostics).
- Volumes are in the configuration's own unit, labelled `m3` and `m3/d` (per daily step).
  Additive pollutants are labelled `kg`, and non-additive ones by name (`temperature` in °C).
  Values are never converted, and the docs say so (W20).

### Diagnostics

`diagnostics`, when given a dict, receives the following.

- `"maxiter"`: a dict keyed by call site, `(node, method, call index)`. Each value is a
  `(..., T)` bool saying whether that `pull_distributed` or `push_distributed` loop ended at
  `MAXITER`. A node runs several such loops a day: `Land.run` up to two, and `create_demand` one
  pull and up to three pushes. WSIMOD prints "Maxiter reached" for these (`nodes.py:405-406`,
  `:475-476`).
- `"events"`: a dict from each of WSIMOD's warning conditions to a `(..., T, n_nodes)` bool. The
  conditions are those behind "Storage unable to push", "demand deficit", "Spill at reservoir",
  "WWTW couldnt push", "Service reservoirs not filled", "nowhere for sludge to go", "river cant
  push" and "ponded water cant reenter". This turns each print into a recorded fact. noodl adds
  conditions of its own where WSIMOD would raise instead: `"zero_priority"` (a distribute
  round with `priority = 0` while `avail > FLOAT_ACCURACY`, defect table),
  `"wtw_liquor_zero_denominator"` and `"inaccurate_evaporation"`.
- `"decisions"` (only when `diagnostics["decisions"] = True` on entry): every branch site's
  per-instance outcome per day, in call order. The gradient tests and the parity admission rule
  use it. Each record is `Decision(site, call, round, active, outcome, margin)`:
  - `site` is `"<node|arc>:<name>/<WSIMOD method>/<branch>"`, with the direction in the branch
    for methods that serve both directions: `node:N/get_connected/<push|pull>:<arc>/below_fa`,
    `node:N/push_distributed/{oversubscribed,not_pushed,avail,amount}`,
    `node:N/pull_distributed/{deficit,avail}`, `node:N/check_basic/<push|pull>/clip`,
    `arc:A/get_excess/<push|pull>/min`, `arc:A/send_push_request/clip`,
    `arc:A/send_pull_request/clip`, `node:N/pull_check_abstraction/clip`,
    `node:N/pull_set_abstraction/clip`, `arc:A/send_push_request/below_fa` (`AltQueueArc`).
    A tank is not a node: its sites are its owner's with the tank's attribute path,
    `node:N/<tank>/get_excess/{headroom,clip}`, `node:N/<tank>/get_avail/clip`,
    `node:N/<tank>/push_storage/clip`, `node:N/<tank>/pull_storage/{empty,clip}`,
    `node:N/<tank>/pull_ponded/excess`, `node:N/<tank>/evaporate/clip`, and for a `QueueTank`
    `node:N/<tank>/push_check/clip` and `node:N/<tank>/pull_storage_exact/clip:<column>`; its
    internal arc's are `node:N/<tank>/internal_arc/{get_excess/push/min,
    send_push_request/below_fa, send_push_request/clip}`. Later classes extend the pattern.
  - `call` is noodl's static per-day index of the enclosing (element, method) call. It counts
    the calls noodl's structure makes, masked ones included, so it is the same every day; it is
    not WSIMOD's dynamic count.
  - `round` is the distribute loop's round, 1 to `MAXITER`, or none outside a loop.
  - `active` is the instance mask the branch was evaluated under; only its true entries are
    WSIMOD's decisions. A loop condition at round `r` is evaluated under the mask round `r − 1`
    ran with, the branches inside round `r` under round `r`'s own mask.
  - `outcome` is the comparison and `margin` is `lhs − rhs`.

  Tests key the records by `(site, call, round)`.
- `"mass_balance"`, also on `WsimodResults`: per day, the system's inflow (catchment flows,
  precipitation, `FWTW` deficits, `UnlimitedDistribution` supply, demand loads), its outflow
  (Waste, evaporation, demand consumption, unpushed sludge, WTW solids, unrouted catchment
  water, and the water WSIMOD drops below `FLOAT_ACCURACY`), and its storage change. WSIMOD
  drops that water in six places: `QueueArc` pushes (`arcs.py:378-379`), waiting `QueueArc`
  requests removed on update (`arcs.py:483-485`), the
  `Land.run` percolation push-back (`land.py:213-215`), the `WWTW` storm-tank rejection
  (`wtw.py:403-404`), the `WWTW.make_discharge` push-back (`:347-348`), and the `QueueTank`
  storage-without-queue entry (Nodes › QueueTank). All are
  `(..., T, 1 + K_additive)`, with the residual. This is what WSIMOD's system check
  (`model.py:1011-1060`) meant to compute (W4).

## Nodes and arcs: equations, state, quirks

Each entry gives the WSIMOD lines it transcribes. Every quirk listed is reproduced unless it is
marked W4.

- **Tank** (`tanks.py:10-356`).
  - `get_excess = v_change(storage, max(cap − V, 0))`, clipped to the request (`:132-158`).
  - `push_storage` (`:160-205`): without `force` it accepts `min(V_in, excess)` and returns the
    rest. With `force` it sums everything.
  - `pull_storage` (`:207-239`): returns empty when `V == 0`, an exact comparison. Otherwise it
    takes `v_change(storage, min(req, V))`.
  - `pull_ponded = pull_storage(max(V − cap, 0))` (`:78-100`).
  - `evaporate` (`:303-317`): takes `min(evap, V)` by `v_distill`, which concentrates the
    pollutants.
  - `get_head` is not used in 3a.
- **ResidenceTank** (`:359-400`): `pull_outflow = pull_storage(v_change(storage,
  V/residence_time))`.
- **QueueTank** (`:460-642`).
  - `push_storage(v, time)` (`:513-539`) goes through the internal `AltQueueArc` with request
    time `time + number_of_timesteps`. `storage` gains the accepted part and `active_storage`
    gains water only on arrival.
  - `push_check` is `get_excess` (`:588-608`), and `push_set` adds to `active_storage`
    (`:610-629`).
  - `pull_storage` (`:541-564`) and `pull_storage_exact` (`:566-586`, a per-column `min` with
    `active_storage`) act on `active_storage` and `storage` together.
  - Every push flushes the queue's due slot at once, and `pull_ponded` is limited to
    `active_storage` (Daily driver › Flushing a `QueueTank` on push).
  - Quirk: a push below `FLOAT_ACCURACY` returns empty, so the water counts as accepted, but
    it never enters the queue (`arcs.py:378-379`), while `storage` keeps it. The difference is
    recorded as dropped water in the mass balance.
- **Arc, QueueArc, AltQueueArc** as in the protocol.
  - `QueueArc` counts `average_flow = V/(time + 1)` against capacity (`arcs.py:409-430`).
  - On push it returns the backflow of rejected due requests to the current sender
    (`:399-406`), and the arc's `vqip_in` loses that backflow (`:405`).
  - The pull path (`:318-358`) enqueues what it pulled now. The caller receives only the
    requests that are due, so a delay `≥ 1` hands the puller water pulled on earlier days.
    `tag` is not passed through on a pull (`:337`, `:348`).
  - The pull path rescales the request's additive entries without the `volume > 0` guard that
    `Arc` has (`:341-343`). WSIMOD therefore divides by zero only when a zero-volume request
    carries additive keys. A full-VQIP request can, for example `Reservoir.make_abstractions`
    passing `tank.get_excess()` (`storage.py:985`) when the reservoir is full. A zero-volume
    request without additive keys is harmless, because the loop body never runs. Only the
    first case is a defect. noodl does what `Arc` does and skips the rescale at zero volume
    (defect table). Nothing reads the rescaled entries (Protocol › Request mutation).
  - `AltQueueArc.update_queue` pushes `queue[0]` without a tag (`:603-638`).
  - Travel times and time-area keys must be non-negative integers. Any other key would never
    shift in WSIMOD, so it is refused by name.
- **Node** as in the protocol. **Waste** (`waste.py:10-49`): accepts any push, and a check
  returns the request or `1e15`. It denies pulls.
- **Storage** (`storage.py:15-143`).
  - Handlers `:70-73`: the tank's `push_storage`, `get_excess`, `pull_storage` and `get_avail`.
  - `distribute` (`:120-127`): pull everything, `push_distributed` with `of_type=None`, and
    force back the rest.
  - `decays: None` and `decays: {}` are two different behaviours. The node tests `decays is
    None` (`storage.py:53`, `:97`, `:300`), so any dict, `{}` included, builds a `DecayTank` or
    `DecayQueueTank`. Such a tank reads the node's own `("temperature", t)` at every
    `end_timestep` (`core.py:485-487`), raising `KeyError` without it, and it is left out of
    `record_all`, which matches the classes `Tank`, `QueueTank` and `ResidenceTank` exactly
    (`model.py:1123`). noodl refuses a `decays` dict on the `Storage` family by name (3b).
    Nothing in 3a needs one, and WSIMOD itself refuses decay overrides on a `None` node
    (`storage.py:96-100`).
- **Groundwater** (`:146-245`).
  - `distribute` (`:222-229`): `V/residence_time` to Node, River and Waste, with the rest
    forced back.
  - `infiltrate` (`:231-245`): `(max(V − cap·threshold, 0)·pct) ** 0.5` pushed to `Sewer`.
    Note the square root. noodl uses a safe root: the value is exact, the gradient is 0 where
    the argument is 0, and the infinite slope at 0⁺ is documented.
- **QueueGroundwater** (`:248-433`).
  - `push_set_timearea` (`:330-348`) splits the inflow over `timearea` into the `QueueTank`.
  - `distribute` (`:350-364`): `update_queue`, then push all of `active_storage`, then pull
    what was sent.
  - `infiltrate` is a no-op (`:366-368`).
  - `pull_check_active` reports `active_storage` (`:370-384`). `pull_set_active` instead pulls
    from **total** storage, pro rata over every queue slot and the active store
    (`:386-433`). That inconsistency, which WSIMOD marks with a TODO, is transcribed.
- **River** (`:452-955`), without biochemistry.
  - `area = length·width`, `capacity = 1e15` (`:524-533`); `depth ≠ 2` warns (`:514-518`).
  - `riverrc = 1 − kt + kt·exp(−1/kt)`, with `kt = damp·length/velocity`, and 1 when `kt = 0`
    (`:900-915`).
  - `distribute` (`:926-936`): `V·riverrc` to River, Node and Waste, with the rest forced back.
  - Push accepts everything by force, and its check returns `1e15` (`:582-583`, `:711-721`).
  - `pull_check_river` (`:660-686`): the tank's VQIP with its volume raised by the
    upstream `avail`, then rescaled to `max(V + upstream − mrf/riverrc, 0)`. The rescale uses
    the raised volume as its base, so the tank's mass is spread over the upstream water too.
    Transcribed.
  - `pull_set_river` (`:688-709`): from the tank first, then `pull_distributed` upstream from
    River and Node.
  - The `("RiparianBuffer", "volume")` check (`:589`, `:938-949`) reads a nonexistent
    `river_tank`. It belongs to 3c and is refused by name if anything asks for it.
- **Reservoir** (`:958-989`). `make_abstractions` pulls the tank's excess from upstream, then
  pushes it into the tank without force and forces the spill back in too. The tank can
  therefore exceed its capacity, and nothing spills downstream (`:983-989`).
- **RiverReservoir** (`:992-1133`).
  - Push: force in, pull the ponded water, `push_distributed` it to Node, River and Waste, and
    count it as environmental flow (`:1060-1084`).
  - Check: the tank's excess plus downstream `avail` (`:1086-1110`).
  - `satisfy_environmental` (`:1112-1128`).
- **Sewer** (`sewer.py:14-247`).
  - `QueueTank` with `number_of_timesteps = 0` (`:108-113`).
  - Handlers (`:96-104`): the `"default"` and `"Sewer"` tags enter at `pipe_time`. The `"Land"`
    and `"Demand"` tags spread over `pipe_timearea`. `Land`'s impervious runoff is pushed with
    the default tag, so it uses `pipe_time`.
  - `make_discharge` (`:201-239`):
    1. `update_queue`.
    2. `push_distributed(active_storage)` over all types.
    3. `pull_storage_exact` of what was sent.
    4. Push the ponded water to `Land` with tag `"Sewer"`, forcing back the rest at time 0.
  - A `pipe_timearea` that does not sum to 1 warns (`warnings.warn`) and runs, as WSIMOD prints
    and runs (`:135-141`).
- **WTW** (`wtw.py:14-220`).
  - Volume constant `1 − percent_solids − liquor_multiplier["volume"]` (`:92`, `:104-110`).
  - `treat_current_input` (`:163-214`):
    - treated `= V·c_v`, liquor `= V·m_v`, solids `= V·percent_solids`;
    - pollutants by the neutral split;
    - the non-additive liquor blend with the old liquor (`:174-182`), keeping the previous
      value where the denominator is 0 (Representation › neutral WTW);
    - `treated` accumulates (`:211`);
    - with pollutants on, each additive pollutant reads the influent's temperature
      (`:191-193`), so `temperature` must be a pollutant.
  - `get_excess_throughput = max(cap − current_input.V, 0)` (`:155-161`).
  - A bare `WTW` node denies pushes and is never orchestrated, so it is accepted and inert.
- **WWTW** (`:223-453`).
  - `push_set_sewer` (`:374-407`): direct to `current_input` up to the spare throughput, with
    the exact-equality early return (`:394`). The rest goes to the storm tank without force,
    and the rejection is returned only when it is at least `FLOAT_ACCURACY`.
  - `push_check_sewer`: storm-tank excess plus spare throughput (`:351-372`).
  - `calculate_discharge` (`:323-341`): drain the storm tank into the spare throughput, add
    yesterday's liquor, then treat.
  - `make_discharge` (`:343-349`): push `treated` and set it empty. The rejection is forced
    into the storm tank only when it exceeds `FLOAT_ACCURACY` (`:347-348`); a smaller one
    vanishes and is recorded as dropped water.
  - `treated` is reset only here, not in `end_timestep_` (`:447-453`), so it is carried state
    (`"wsimod.treated"`).
  - Reuse pulls come from `treated` (`:409-445`).
- **FWTW** (`:456-657`). `treat_water` (`:573-617`):
  1. `target = min(service excess, throughput)`, then `pull_distributed`.
  2. The deficit `max(target − pulled, 0)` is **created** water with the quality of yesterday's
     `total_pulled` (`:588-596`). It is a mass-balance inflow.
  3. Treat. Push liquor + solids to `Sewer`; any rejection is unpushed sludge, which leaves the
     system.
  4. Treated water goes into the service reservoir, without force first and then forced.

  Pulls come from the service reservoir and accumulate `total_pulled` (`:619-644`).
- **Demand** (`demand.py:14-137`).
  - `create_demand` (`:79-119`) pulls the total over all in-arcs. It then pushes each
    component in dict order: `garden` to `Land` with tag `("Demand", "Garden")`, `house` to
    `Sewer` with tag `"Demand"`, and `default` to any node.
  - The pulled water's quality is discarded. What is pushed is the demand's own VQIP.
  - `get_demand`: `constant_demand` with `pollutant_load` entries set as given (`:121-131`).
- **ResidentialDemand** (`:153-324`).
  - Garden: `get_connected(push, "Land", ("Demand", "Garden")).avail × gardening_efficiency`
    (`:250-301`). `Land` answers 0 unless it has a `GardenSurface` (`land.py:93-94`), which is
    3c. So in 3a the garden demand is 0, but it still goes through the transcribed calls.
  - House: `population·per_capita`, with `pollutant_load·population` for the additive entries
    and the mixed temperature (`:303-324`). A `pollutant_load` lacking an additive pollutant
    `KeyError`s in WSIMOD and is refused by name.
- **NonResidentialDemand** (`:140-150`). W4: WSIMOD recurses forever. noodl does the evident
  intent `{"house": Demand.get_demand(self)["default"]}`.
- **Distribution** (`distribution.py:101-153`).
  - The leakage decorators (`:13-98`): the request is divided by `1 − leakage`. The reply loses
    `leakage` of its volume, which is pushed to `Groundwater`. Water that cannot leak is
    returned to the caller.
  - Push is denied.
  - Decoration is structural: the handlers are wrapped when the configured `leakage > 0`
    (`:132-139`), and `leakage ≥ 1` is refused (Representation › Batches).
  - W17: `apply_overrides` decorates again (`:152`), so an overridden leakage is applied twice.
    Fable confirmed this by running it: a request of 10 at leakage 0.2 became 15.625 = 10/0.8².
    noodl applies it once.
- **UnlimitedDistribution** (`:156-200`): pulls return clean water of the requested volume,
  counted as supply, and its check is `1e15`. Its own `__init__` replaces the handlers that the
  base class decorated (`:172-177`), so a leakage given at construction is silently lost.
  noodl refuses `leakage > 0` on it by name rather than guess.
- **Catchment** (`catchment.py:12-138`).
  - `get_flow`: the data's flow, plus every pollutant series (additive ones × flow).
  - `get_avail`: the flow less the `vqip_in` volume of **every** out-arc so far (`:87-101`).
  - `route` pushes the remainder to Node, River and Waste, and accumulates the unrouted water
    (`:76-85`).
  - Abstraction pulls come from `get_avail` (`:103-134`).
  - Its push check is `push_set_deny` (`:52`), which returns its argument. A check given a VQIP
    therefore accepts all of it, and the following `push_set_deny` (`:51`) rejects all of it.
    A check without a VQIP (from `get_connected` or `check_basic`) returns `None` and fails with
    a `TypeError`; Fable ran this. A push into a catchment can never deliver, so an arc
    **into** a catchment is refused by name.
- **Land** (`land.py:17-300`).
  - Residence tanks: `surface_runoff`, `subsurface_runoff` and `percolation`, each of capacity
    `1e15` (`:138-149`).
  - `run` (`:200-244`):
    1. Surfaces run in order.
    2. Percolation outflow goes to `Groundwater`. The rejection is forced back only when it
       exceeds `FLOAT_ACCURACY`, so smaller remainders vanish (recorded as dropped water).
    3. Surface plus subsurface outflow goes to River and Node, but only if positive. A
       positive rejection is split pro rata by volume and forced back into each tank.
  - W4: `push_set_sewer` (`:246-264`) reads a nonexistent `self.surface`, and `Land` has no
    `"Sewer"` push-check handler. Any ponding from a sewer to land therefore crashes WSIMOD,
    whether through the single-arc push or the check of a multi-arc one. noodl does the evident
    intent: the `"Sewer"` check accepts everything when the land has an `ImperviousSurface`
    (empty otherwise), and the set forces the water into the first such surface.
- **Surface** (`:303-581`).
  - `capacity = area·depth` (`:369-371`).
  - `run` walks `inflows + processes + outflows` (`:425-449`). In 3a these hold only the
    neutral deposition and the water functions below. The `nitrite` carry-over (`:427-439`)
    adds zero without decays.
- **ImperviousSurface** (`:584-721`).
  - `capacity = area·pore_depth`. The default pore depth is 0, so everything ponds at once.
  - `precipitation_evaporation` (`:645-697`):
    - If `P < E·et0_to_e`, it evaporates `(E − P)·area`, limited to the store.
    - Otherwise it forces `(P − E)·area` in, at air temperature.
  - `push_to_sewers` (`:699-721`) pushes the ponded water to `Sewer` and forces the rejection
    back.
  - The oxford urban surface starts at 5e6 with capacity 0, and all of it leaves on day 1.
    Transcribed.
- **PerviousSurface** (`:724-1079`).
  - Depths: the simulated depth is `depth·total_porosity`, while `field_capacity_m` and
    `wilting_point_m` use the physical depth (`:796-816`; overrides `:843-884`, including the
    order of the porosity update).
  - The denominators `D − fc_m = depth·(porosity − field_capacity)` and
    `D − wp_m = depth·(porosity − wilting_point)` (`:942`, `:950`) must both be positive.
    Otherwise Python raises `ZeroDivisionError`, or raises a negative base to a non-integer
    power `ihacres_p` and gets a complex number. noodl therefore refuses
    `total_porosity ≤ field_capacity` and `total_porosity ≤ wilting_point` by name, after
    overrides are applied.
  - IHACRES (`ihacres`, `:913-1044`):
    - `P_inf = min(P, infiltration_capacity)`;
    - `evap = E·et0_coefficient·min(1, exp(2(1 − cmd/(D − wp))))`;
    - `outflow = P_inf·(1 − min(1, (cmd/(D − fc))^p))`;
    - `evap = min(evap, P + smc)` (`:957`);
    - the split into surface, percolation and subsurface (`:961-973`);
    - `tank_recharge` (`:974`).
  - If water passes through the soil store (`> 0`), it is forced in at air temperature, and
    the subsurface and percolation flows are pulled out (`:985-996`). Otherwise
    `evaporate(−total)` runs and those flows are empty (`:997-1001`). The `evaporate` clamp can
    break the balance; WSIMOD prints when it does (`:1003-1021`), and noodl records the event.
  - `route` forces the three flows into the parent's tanks (`:1046-1058`).
  - The soil-temperature process is neutral only (3b).

### WSIMOD defects (W4, W17): what noodl does instead

Each item below has its own test of the documented behaviour. The parity cases avoid each one,
except the reproduced `QueueArc` row.

Rule: where the intent is clear, noodl does it; otherwise it refuses by name (W17). One row
departs from the rule of not reproducing a defect: the `QueueArc` drop is reproduced. The
water lost is below `FLOAT_ACCURACY`, and the delayed delivery is what every WSIMOD model with
such an arc computes, so parity with real models needs it. The review asked for it (item 2).
WSIMOD's other sub-`FLOAT_ACCURACY` drops are tolerances, not defects. They are transcribed and
recorded in the mass balance (Diagnostics).

| Defect | WSIMOD | noodl |
|---|---|---|
| `Land.push_set_sewer` (`land.py:262`) + no `"Sewer"` check | `AttributeError`/`KeyError` | evident intent (above) |
| `EnfieldFoulSewer.make_discharge` (`sewer.py:288-291`) | `AttributeError` (`storm_exchange`) | refused by name |
| `NonResidentialDemand.get_demand` (`demand.py:143-150`) | infinite recursion | evident intent |
| `GrowingSurface.residue` unwired (`land.py:1281`) | not called | 3c; nothing in 3a |
| System mass balance compares `sys_in` with itself (`model.py:1040`, `:1046-1052`) | check is a no-op | real balance in diagnostics |
| `Distribution.apply_overrides` re-decorates (`distribution.py:152`) | leakage applied twice | once |
| `UnlimitedDistribution` loses its leakage (`:172-177`) | silently ignored | `leakage > 0` refused |
| Overrides `update` shared default dicts (`pollutant_load={}`, `decays={}`: `demand.py:76`, `land.py:397`, `tanks.py:439`) | an override reaches every node built with the default | per-node |
| River arcs but no `Waste` (`model.py:799`, `max([])`) | `ValueError` | refused by name |
| An arc into a `Catchment` (`catchment.py:51-52`) | a check with a VQIP accepts and the push then rejects everything; a check without one raises `TypeError` | refused by name |
| WTW liquor blend divides by zero on a zero-input day with no previous liquor, or always when `liquor_multiplier["volume"] = 0` (`wtw.py:176-182`) | `ZeroDivisionError` | previous liquor value kept (double `where`), event recorded |
| A `QueueArc` push below `FLOAT_ACCURACY` returns before `update_queue` (`arcs.py:378-379`) | the water is lost, counted as accepted, and due requests wait for the next real push | reproduced; the water is recorded as dropped |
| A `QueueArc` pull rescales a zero-volume request that carries additive keys (`arcs.py:341-343`) | `ZeroDivisionError` | the rescale is skipped at zero volume, as `Arc` does (`:180-183`) |
| A distribute round with `priority = 0` while `avail > FLOAT_ACCURACY`: some arcs have preference 0 and the others no room (`nodes.py:393`, `:462`). State-dependent, so a static refusal would refuse configurations WSIMOD runs | `ZeroDivisionError` | nothing is distributed that round (a safe denominator over zero allocations), so the amount stays unpushed or unpulled; event `"zero_priority"` |
| The `WWTW.make_discharge` rejection at or below `FLOAT_ACCURACY` is not forced back (`wtw.py:347-348`) | the water vanishes | transcribed as a tolerance, not a defect; recorded as dropped |
| `decays: {}` on the `Storage` family builds a `DecayTank` (`storage.py:53`, `:300`) | needs node temperature forcing, and the tank is not recorded (`model.py:1123`) | refused by name (3b) |
| No check that `total_porosity` exceeds `field_capacity` and `wilting_point` (`land.py:942`, `:950`) | `ZeroDivisionError` or a complex number | refused by name |

## Loading WSIMOD's model description (W5)

The format is WSIMOD's `config.yml` as `Model.load` reads it (`model.py:216-274`), with
per-node CSV files (`_load_individual_files`, `:324-348`; `read_csv`, `:1252-1269`, `.gz`
included). The parsed mapping is also accepted in the same structure, with `data_input_dict`
given inline as `{(variable, date): value}`, which is what the in-memory `add_nodes` form takes.

### Top-level keys

| Key | What noodl does |
|---|---|
| `nodes` | Required (`:255-256`). |
| `arcs` | Read. |
| `overrides` | Read. |
| `orchestration` | Read. |
| `pollutants`, `additive_pollutants`, `non_additive_pollutants` | Read. |
| `float_accuracy` | Read. |
| `dates` | Read: WSIMOD's date formats (`:139-162`), one step per date. |
| `river_discharge_order` | Ignored (written by `save`; `load` recomputes it). |
| `extensions` (non-empty) | Refused by name (out of scope). |
| `unified_data_file` | Refused by name: the parquet file needs pandas and pyarrow at runtime, and IO is out of scope. Per-node CSV and CSV.GZ are supported (W18). |
| Anything else | Refused by name. |

`wsimod_run` refuses a model without dates by name. WSIMOD raises `AttributeError` there.

### Node keys

- `name`, `type_`, and `node_type_override` (the class; the slot key stays `type_`).
- `geometry` is dropped, as WSIMOD drops it (`:661-662`).
- `filename` or `data_input_dict`.
- For `Land`, `surfaces`, a dict keyed by name (`:339-346`) or a list. Each surface has `type_`,
  `surface`, its parameters and an optional `filename` or `data_input_dict`.
- Every other key is a constructor argument of the class's chain. noodl refuses an argument the
  chain does not accept by name, where WSIMOD raises `TypeError`.

### Arc keys

`name`, `type_`, `in_port`, `out_port`, `capacity`, `preference`, `number_of_timesteps`
(`QueueArc` family). `decays` is refused (3b), because it belongs to the `Decay*` arc classes.
WSIMOD's `Arc` accepts any extra keyword silently (`arcs.py:52`). noodl refuses an extra key by
name.

### Overrides

The `overrides` section is applied after construction, per class, by transcribing that class's
`apply_overrides`. The transcription keeps every derived recomputation:

- `Surface` capacity (`land.py:399-401`);
- `PerviousSurface` depths (`:858-884`);
- `River` area and capacity (`storage.py:643-658`);
- `ImperviousSurface` `pore_depth` (`land.py:633-643`);
- the WTW setters (`wtw.py:116-128`, `:142-152`);
- `Sewer` tank fields (`sewer.py:127-142`);
- the `Land` residence times (`land.py:174-185`);
- the per-surface `surfaces` overrides (`:187-188`).

A node override names its node by `type_` and `name`, and WSIMOD looks it up as
`nodes_type[type_][name]` (`model.py:819-825`). So the override's `type_` must be the **slot
key**: `"Foul"` for a foul sewer, or `"Demand"` for a saved `ResidentialDemand`. `save` writes
`type_` as the renamed class and puts the real class in `node_type_override`
(`model.py:398-401`), so the regenerated demos keep their slot keys. noodl uses the same lookup.

An override for an unknown node or arc is refused by name, as WSIMOD refuses it (`:819-834`).
An override key the class does not consume is refused by name. WSIMOD only prints "No
override behaviour defined" (`nodes.py:101-102`).

A `filename` (or a surface `data_input_dict` path) in an override is refused by name in 3a.
`Node.apply_overrides` (`nodes.py:93-99`) and `Surface.apply_overrides` (`land.py:413-421`)
open it relative to the current working directory, not the config directory.

### Forcing checks at build time

Each node's forcing is checked over the run's dates when the model is built, and each missing
`(variable, date)` is refused by name, with the node, the variable and the first missing date.
WSIMOD raises `KeyError` mid-run instead. The variables each node reads are:

- `Land`: `precipitation`, `et0` and `temperature` (the last also for the end-of-day decay
  read);
- `ResidentialDemand`: `temperature`;
- `Catchment`: `flow` and every pollutant.

These are required unconditionally, as WSIMOD reads them. That includes quantity-only runs
(`K = 0`), and models where `temperature` is not a pollutant: `Land` reads it in
`precipitation_evaporation`, `ihacres` and the surface decay (`land.py:685`, `:924`;
`core.py:485-487`), and `ResidentialDemand` reads it in `get_house_demand` (`demand.py:321`),
whatever the pollutant list.

## Differentiability

- **Learnable.** Every real-valued parameter driver, every forcing driver and every initial
  storage (volume and pollutants). Examples: arc `capacity` and `preference`, tank
  capacities and areas, residence times, `Groundwater` infiltration threshold and percent,
  River `length`, `width`, `velocity`, `damp` and `mrf`, WTW throughputs, `percent_solids`,
  `liquor_multiplier["volume"]`, demand `population`, `per_capita`, `constant_demand` and
  `pollutant_load`, `leakage` (within `(0, 1)` on a decorated node), the `RiverReservoir`
  environmental flow, time-area fractions, and every IHACRES and impervious parameter.
- **Not learnable (structural).** Topology, tags, `of_type`, orchestration, the river order,
  integer travel times and time-area keys, the pollutant lists and `FLOAT_ACCURACY`.
- **Detached.** Every decision: comparisons, `min`/`max` choices, loop exits, the over-subscription
  switch, the single-arc shortcut (structural), and `V == 0` and `V > 0` guards.
- **Known jumps and kinks.** These are documented on the application page, and the gradient
  tests avoid them:
  - `get_connected` zeroes `avail < FLOAT_ACCURACY`.
  - The loop exit thresholds.
  - `push_distributed`'s first-round switch to capacity shares (`nodes.py:446-450`) is a jump
    in the shares.
  - Truncation at `MAXITER`.
  - `Land.run`'s `> FLOAT_ACCURACY` push-back threshold.
  - The sub-`FLOAT_ACCURACY` drop in `QueueArc`.
  - The IHACRES `min(1, ·)` kinks.
  - The `Groundwater.infiltrate` root at its threshold.
  - The gradient with respect to a clipped request is zero wherever a capacity binds.

  The branch switches of IHACRES (at zero recharge) and of the impervious surface (at `P = E`)
  are continuous.
- **No smoothed variants in 3a.** WSIMOD's hard decisions only. The smooth and projection modes
  of `AllocatedFlowLayer` are a different model (W2).
- **Graph size (W9).** Autograd is enabled for a run only when an input requires grad.
  Otherwise the run is under `torch.no_grad()`. A 1456-day run with grad builds a large graph,
  so the docs describe fitting in windows. 3a has no checkpointing.
- **Speed (W8).** A Python-level transcription makes thousands of small tensor operations a
  day, and batches amortise that cost. This is accepted for 3a. `benchmarks/wsimod_run.py`
  measures the time per day and the graph size for both demos at batch sizes 1, 10 and 100,
  with no asserts. Compiling the static call tree into vectorised levels is left to its own PR.

## Verification (W3)

References come from live WSIMOD 0.8.1 (`pytest.importorskip("wsimod")`). WSIMOD's globals
(pollutant lists, `FLOAT_ACCURACY`) are reset before every WSIMOD build by a harness helper,
`reset_wsimod_globals()` in `tests/verification/_wsimod_reference.py`. `create_oxford_model`
and `Model.load` mutate them for the whole process, which would make results depend on test
order and on xdist workers.

### 1. Per-node equations on replayed inputs

`capture_node_calls(model, nodes)` (in `_wsimod_reference.py`) wraps, for each target node:

- its orchestration methods and its four handler entry points, recording the arguments, the
  reply, and the node's stores before and after;
- every arc call the node makes (`send_*_request`, `send_*_check`), recording the arguments and
  the reply, in order.

Every argument is deep-copied at call entry, before WSIMOD can mutate it (Protocol › Request
mutation), so the replay compares the requests as they were sent.

The replay builds the noodl node alone. Its stores are set to the captured pre-state. Its arcs
are `ScriptedArc`s that return WSIMOD's captured replies in order and assert that noodl made
the same calls, in the same order, with the same arguments to rounding. The replay then
compares the reply and the post-state.

Every in-scope class is replayed:

- `Land` (impervious and pervious, both IHACRES branches), `Sewer`, `WWTW`, `FWTW`,
  `Reservoir`, `Groundwater`, `ResidentialDemand`, `Catchment`, `Node` and `Waste`: on
  quickstart and oxford. The default suite replays a sample of days: the first 60, then every
  20th, plus every day on which the capture shows a branch outcome the sample has not yet
  covered. Every day is replayed under `-m slow`;
- `QueueGroundwater`, `River`, `RiverReservoir`, `Storage`, `Demand`, `Distribution`,
  `UnlimitedDistribution` and the `QueueArc` family: on the purpose-built cases.

Bound: 1e-12 × the case scale (below). The VQIP algebra and the arcs also get unit tests
against WSIMOD's `WSIObj` and `Arc` on random inputs, including zero volumes and empty
non-additive blends.

### 2. Whole-model parity

Each case is a WSIMOD configuration directory, committed under `tests/data/wsimod/<case>/`.
WSIMOD runs it through `Model.load` + `run(verbose=False)`, and noodl through
`read_wsimod_config` + `build_model` + `wsimod_run`. Compared: every arc's `flow` and pollutant
entries, every recorded tank, and every surface field, on every day.

**Demos.** `quickstart` and `oxford`, written by `scripts/regenerate_wsimod_fixtures.py`
through WSIMOD's own `Model.save(..., compress=True)` with the shared demo forcing. That forcing
is WSIMOD's BSD-3-Clause `timeseries_data.csv`, subset to the sites used and credited in
`tests/data/wsimod/README.md`. It is committed in WSIMOD's save format, about 1 MB gzipped, so
test runs never download it (W14).

The script fixes one WSIMOD `save` defect before writing. `save` writes a `PerviousSurface`'s
simulated depth (`depth·porosity`) as `depth`, because `model.py:419-420` keeps that key for
it. A reload then shrinks the depth by the porosity again; this was run and confirmed
(0.5 → 0.2 written → 0.08 reloaded). The script then checks that the reloaded WSIMOD model's
`run` equals the original model's, and fails otherwise. noodl itself reads configurations
exactly as `Model.load` does, with no correction (W12).

Each demo runs twice:

- quantity-only (`pollutants=False`), as shipped;
- **neutralised**, a second `config_neutral.yml` that differs only by an `overrides` section.
  The section sets the WTW process parameters and liquor multipliers neutral, the surface
  `pollutant_load` to zero, and the soil-temperature weights to `(prev 1, air 0, deep 0)`. This
  run compares every pollutant.

**Suite composition (W15, as amended by the review).** The candidate default is quickstart in
full (1456 days), oxford for its first 365 days, the purpose-built cases and the sampled
replays. Full oxford and the full replays run under `-m slow`. The suite time cannot be
measured until the driver exists, so the plan must apply this rule:

1. Measure first. Once `wsimod_run` runs both demos, the plan records the wall time of each
   default candidate: quickstart in each mode, oxford-365 in each mode, the sampled replays, and
   the case set. It uses `pytest --durations=0 -n 4 --no-cov` on the wsimod row and writes the
   numbers into the plan's ledger. Nothing asserts a time (no wall-clock tests).
2. Then choose the composition. If the wsimod row's default run exceeds 10 minutes on the
   developer machine, make these changes in order until it fits:
   1. move oxford-365 to `slow`, keeping oxford's first 60 days in the default;
   2. halve the replay sample (every 40th day);
   3. run quickstart's quantity-only mode only for its first 365 days, keeping the full run in
      the neutralised mode.
3. Record and document. The chosen composition is recorded in `AGENTS.md`'s wsimod row and in
   the application page's verification table.

**Purpose-built cases** (small configs with inline forcing, 30–120 days, run in both modes
where pollutants matter). Each exercises a feature the demos skip:

- `QueueArc` push and pull with delay ≥ 1, a rejecting receiver (backflow), and days without a
  push (due requests waiting). The W21 test is part of this case: pushes on irregular days,
  several pushes and several distribute rounds on one day, and a tiny push on a due day;
- `AltQueueArc` as a user arc (shifted every day, flushed only by a push); `PullArc` and
  `PushArc`;
- `Land → Sewer` runoff with a `Sewer → Land` ponding arc, which is a legal topological cycle.
  The model is built, and the ponding is noodl-only per the defect table;
- forced pushes that make an arc's pipe excess negative, seen through `check_basic`;
- a `QueueTank` entry with total time 0, active on the same day;
- a `WWTW` and an `FWTW` with zero input on day 1. These are noodl-only: WSIMOD divides by zero;
- a `Sewer` with `pipe_time > 0` and a `pipe_timearea`, reached through the `"Demand"` and
  default tags;
- foul sewers by name;
- `Groundwater.infiltrate` with `infiltration_pct > 0`;
- `QueueGroundwater` with a time-area and pulls through `pull_set_active`;
- a river chain: `riverrc` with `damp = 0`, `mrf > 0`, upstream pulls, the river order with
  three levels and a tie, and a river not linked to the Waste;
- `Reservoir` spill;
- `RiverReservoir` with a custom orchestration that includes `satisfy_environmental`, and
  `Storage.distribute` in a custom orchestration;
- `Distribution` leakage, with and without a `Groundwater` to take it;
- `UnlimitedDistribution`; `Demand` with `constant_demand` and `pollutant_load`;
  `ResidentialDemand` with a `Land` (garden 0);
- `WWTW` storm tank filling and draining past its throughput;
- an `FWTW` deficit, and sludge to `Sewer`;
- an abstraction pulled from a `Catchment`;
- a node with three out-arcs and unequal preferences hitting the over-subscription switch;
- a `pull_distributed` that needs more than `MAXITER` rounds;
- binding arc capacities with first-come-first-served order across orchestration slots;
- a chain of three pass-through `Node`s (check recursion);
- surfaces alone (impervious-only, pervious-only), the IHACRES evaporation clamp, and
  non-additive blending (temperature);
- the simple pollutant set (`set_simple_pollutants`); a `float_accuracy` override; a config
  using `node_type_override`.

**Bounds and admission (W19).** Bitwise equality is not required, because torch's `pow`, `exp`
and `sqrt` may differ from CPython's by an ulp. For each compared series, the scale is its
largest magnitude over the
run (an arc's largest flow, a tank's largest storage, a pollutant's largest mass or value).
Every (day, element) must lie within 1e-12 × scale. A difference beyond it is admitted only
when the noodl run's `"decisions"` record shows a branch whose operand was within 1e-9 × scale
of its threshold, flipped against WSIMOD on that day. The test must name the admission (case,
element, day, branch), and it asserts how many admissions it allows. A bound is loosened only
with a measured cause written in the test, as in the quality spec.

**Mass balance.** noodl's system balance closes within 1e-12 × scale on every case, every day.
WSIMOD's per-node checks, captured from stdout, print no error on the parity cases.

### 3. Defects, refusals and structure

- Each row of the defect table has a test of noodl's behaviour against a hand-computed
  expectation.
- Each refusal has a test that it is raised by name:
  - every out-of-scope class and key;
  - the non-neutral pollutant processes;
  - additive pollutants without `temperature`;
  - cycles in the check-recursion graph, with a topological cycle accepted;
  - zero or negative preferences and non-integer travel times;
  - `decays: {}` on a `Storage`;
  - `leakage ≥ 1`, and a batch mixing zero and non-zero leakage;
  - `total_porosity ≤ field_capacity`;
  - `filename` overrides and unknown override keys;
  - missing forcing (including `temperature` with `K = 0`);
  - inconsistent pollutant lists.
- A quantity-only run equals the neutralised run's volumes exactly.
- A batch of 4 instances with different parameters equals the 4 single runs.
- `model.step` stepped day by day equals `wsimod_run`.
- No `wsimod` import at runtime, checked in a subprocess.

### 4. Gradients

Short runs (quickstart 60 days and the purpose-built cases) are compared with central
differences, at relative step 1e-6 and agreement 1e-6 relative, with the ±h runs' `"decisions"`
records asserted identical. The derivatives checked are those of arc flows, final storages and
pollutant masses with respect to:

- arc capacity (binding) and preference (shares);
- `Groundwater.residence_time`;
- `field_capacity`, `ihacres_p`, `et0_coefficient` and `et0_to_e`;
- the `Land` residence times;
- River `velocity` and `mrf`;
- WWTW and FWTW throughput;
- `per_capita`, `leakage` and a time-area fraction;
- the precipitation and catchment-flow forcing;
- an initial storage;
- a catchment concentration and a demand `pollutant_load`.

`torch.autograd.gradcheck` also runs on the VQIP operations and on one IHACRES day.

## What happens to the existing `apps/wsimod` (W6)

- `AllocatedFlowLayer` (`src/noodl/layers/allocation.py`) is a core layer, not part of the
  application. It stays, with its tests and its page `docs/applications/allocation.md`.
- `apps.wsimod.build_model(topology, ...)`, `initial_state` and `initial_drivers`
  (`src/noodl/apps/wsimod/network.py`) are superseded. They are deleted, with no aliases, and
  the names now belong to the new application.
- The layer's WSIMOD replay tests are renamed, not deleted, because they verify the layer and
  not the application:
  - `test_wsimod_parity.py` → `test_allocation_wsimod_replay.py`;
  - `test_wsimod_capacity.py` → `test_allocation_wsimod_capacity.py`.

  They build the layer directly from the topology JSON. The fixtures
  (`tests/data/wsimod/*_topology.json`, `*_events.csv`) stay and move under
  `tests/data/allocation/`. Their generator stays in
  `scripts/regenerate_wsimod_fixtures.py`, beside the new config writer. The two pending
  docstring edits in the working tree (which drop "WSIMOD is never reimplemented") go into the
  renamed files.
- The callers of the old builder move to the new application on a small committed config:
  - `tests/apps/test_initialisation.py` (the wsimod block);
  - `tests/test_refs.py` (units);
  - `tests/test_units.py` (`_wsimod`);
  - `tests/test_docs_usage.py` (fixture globs).

  `tests/apps/test_entry_points.py` keeps `"wsimod"`.
- `AGENTS.md` and `.github/scripts/select_tests.py` change together:
  - a new "wsimod" row: `tests/apps/wsimod tests/verification/test_wsimod_*.py
    tests/verification/test_allocation_wsimod_*.py`, triggered by `src/noodl/apps/wsimod/`,
    `src/noodl/layers/allocation.py`, `tests/data/wsimod/` and `tests/data/allocation/`;
  - the water row drops its `wsimod` entries.

## Documentation

- A new `docs/applications/wsimod.md`: what is ported, running (`read_wsimod_config`,
  `build_model`, `wsimod_run`), quantity-only versus pollutants, refusals, differentiability and
  its jumps, verification rows with measured values, and limitations.
- A new `docs/formats/wsimod.md`: the configuration keys read.
- `docs/applications/allocation.md`: its "From a WSIMOD topology" section is rewritten to build
  the layer directly, and it links to the new page.
- `docs/applications/index.md`: a WSIMOD row; the allocation row names `AllocatedFlowLayer`.
- The catalogue entry in `scripts/gen_catalogue.py` becomes "WSIMOD", pointing at the new page,
  and is regenerated.
- `mkdocs.yml` nav.
- `docs/verification.md` rows.

## Rulings made in this spec

The review accepted these, with S4 amended as below. The former open questions are now W8–W21,
recorded in the ledger and folded into the sections above:

| Ruling | Subject | Section |
|---|---|---|
| W8 | speed | Differentiability |
| W9 | graph size | Differentiability |
| W10 | pollutants and quantity-only runs | Representation |
| W11 | the closure on a `Model` | Entry points |
| W12 | the `save` depth fix | Verification |
| W13 | river ties | Order of a day |
| W14 | committed forcing | Verification |
| W15 | suite composition | Verification |
| W16 | garden and irrigation | Scope |
| W17 | further defects | Defect table |
| W18 | parquet | Loading |
| W19 | bounds | Verification |
| W20 | units | Outputs |
| W21 | `QueueArc` proof | Representation |

W22 is this revision.

- S1 (scope table above): DecayTank only as Surface's base; RiparianBuffer check refused.
  Cost if wrong: re-homing a class between sub-projects.
- S2: pollutants on means neutral processes only; `pollutants=False` is quantity-only. Cost if
  wrong: a demo would need 3b before it runs.
- S3: masks, fixed MAXITER rounds, and `where` with safe denominators. WSIMOD's summation
  order. 1e15, not inf. Cost if wrong: a re-layout of the transcription.
- S4: the state keys and representations above:
  - one shift register for every `AltQueueArc`, internal and user;
  - `QueueArc` rows by (entry day counted back from the last delivering call, daily call
    index), with a per-direction `since` counter and delivery masked by the early return;
  - liquor carried for every WTW, and `WWTW.treated` carried.

  Cost if wrong: a state-format change before release.
- S5: the interfaces (`read_wsimod_config`, `build_model(pollutants=)`, `initial_state`,
  `initial_drivers`, `wsimod_run(days=, diagnostics=)`, `WsimodResults`, the driver names, and
  the `WsimodDay` closure on `Model`). Cost if wrong: an API change.
- S6: the verification design (replay by `ScriptedArc`, config directories run by both sides,
  1e-12 × scale with named admissions, neutralised demos). Cost if wrong: weaker evidence.
- S7: W6 as above (the layer stays and its replay tests are renamed; the old builder is
  deleted). Cost if wrong: the layer's WSIMOD evidence is lost or kept twice.
