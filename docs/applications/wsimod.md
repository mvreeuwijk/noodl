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
record: arc flows and qualities, tank and surface storages, events and the mass-balance terms.

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
