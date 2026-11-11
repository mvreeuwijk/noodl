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
runs day `k` of the configuration's dates, so a WSIMOD model couples with other noodl models
like any other.

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
`River` (without its nitrate biochemistry), `Reservoir`, `RiverReservoir`, `Demand`,
`NonResidentialDemand`, `ResidentialDemand`, `Distribution` (with leakage),
`UnlimitedDistribution`, `Sewer`, `WTW`, `WWTW`, `FWTW` and `Land` with `Surface`,
`ImperviousSurface` and `PerviousSurface` (IHACRES); the arcs `Arc`, `QueueArc`, `AltQueueArc`,
`PullArc`, `PushArc`, `SewerArc` and `WeirArc`. Pollutants are carried without change: the
treatment, decay and deposition processes that would change them are later sub-projects and
are refused unless they are neutral.

A day is WSIMOD's (`model.py:993-1176`): the orchestration slot by slot, each slot's nodes in
configuration order; `River.distribute` in the river order, most upstream first; the record;
then `end_timestep` on every node and every arc. Where WSIMOD breaks a tie in the river order by
Python's set-hash order, which changes with `PYTHONHASHSEED`, noodl takes the configuration
order.

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
['demand_deficit', 'demand_not_able_to_push', 'environmental_not_able_to_push'] (5, 3)
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
- `"events"`: for every name in `EVENT_NAMES` a `(..., T, n_nodes)` bool, WSIMOD's warning
  prints as recorded facts: `"storage_unable_to_push"`, `"demand_deficit"` ("demand deficit of
  ... at <node> on <date>"), `"reservoir_spill"`, `"wwtw_couldnt_push"`,
  `"service_reservoirs_not_filled"`, `"sludge_nowhere"`, `"river_cant_push"`,
  `"ponded_water_cant_reenter"` (never fires: the forced push before WSIMOD's print returns
  empty), `"groundwater_couldnt_push_all"`, `"miscalculated_tank_storage"`,
  `"environmental_not_able_to_push"`, `"demand_not_able_to_push"`, `"leakage_not_going_to_gw"`,
  `"more_treated_than_input"`, `"excess_treated_water"` and `"more_backflow_than_vqip"`, and
  noodl's own `"zero_priority"` (a distribute round with priority 0 where WSIMOD divides by
  zero), `"wtw_liquor_zero_denominator"` and `"inaccurate_evaporation"`;
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
a run does not read — surface deposition loads, a demand's `pollutant_load`, non-neutral
treatment processes, a catchment's pollutant series — are listed by name in
`model.ignored_by_quantity_only`. With `pollutants=True` (the default) the same configuration is
refused if any of them is not neutral, since the process that would change the pollutants is not
ported yet.

## Refusals

Everything that cannot run is refused by name when the model is built, where WSIMOD would fail
mid-run, hang or silently do something else:

- out-of-scope classes and keys (by the reader);
- non-neutral pollutant processes, `decays`, River biochemistry (nitrate among the pollutants);
- a `total_porosity` not above `field_capacity`, a residence time of 0, `leakage` outside
  `[0, 1)`, preferences that are negative or all zero, travel times that are not whole days;
- a cycle of checks that recurse into each other, which makes WSIMOD retry at every level and
  never finish (WSIMOD's own `Land -> Sewer -> Land` ponding is not one: no check enters it);
- river arcs without a `Waste`, an arc into a `Catchment`, an orchestration slot whose nodes
  lack the method;
- missing forcing, naming the node, the variable and the first missing date (a `Land` needs
  `temperature` even in a quantity-only run, as WSIMOD reads it).

`model.step` needs `dt=86400` and a `t` that is a whole number of days inside the dates.
`model.check(state, drivers, dt=86400, probe=True)` therefore reports `probe-failed`: the probe
steps without `t`, which the closure refuses; `model.step(state, drivers, dt=86400, t=0)` is the
check. A
`leakage` batch that mixes zero and positive values on one `Distribution` is refused when it is
bound: WSIMOD decorates the node's handlers only for a positive configured leakage.

## Verification

Against live WSIMOD 0.8.1, on purpose-built cases that reach each class's branches:

- **Per-node replay** (`tests/verification/test_wsimod_replay.py`): WSIMOD runs each case with
  every call into the chosen nodes captured; noodl replays each node alone on those calls and
  compares every arc call it makes, every reply and every store, in both modes. The bound is the
  class's own, zero (bitwise) for every class whose arithmetic is `+ - * /` and comparisons.
- **Day by day** (the same file and `tests/apps/wsimod`): whole cases stepped by `WsimodDay`
  against WSIMOD's own run, every arc's flow and quality, every tank and surface storage, bitwise
  on every day; the `QueueArc` delivery order on random schedules; WSIMOD's printed warnings
  against noodl's events, day by day.
- **Batches**: every case, both modes, a batch of two arc capacities equals its two single runs
  bitwise (`tests/apps/wsimod/test_build.py`).

Whole-model parity on WSIMOD's own demos, with measured bounds, follows in a later release of
this page.
