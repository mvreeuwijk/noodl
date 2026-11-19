# Water distribution: extended-period simulation

How to run a network over time with `water_run`: patterns, controls, tank moves, report times, results, batches and pump energy. Tanks, controls and rules are described on [Tanks, controls and rules](tanks-controls.md), and water quality on [Water quality](quality.md). The overview is [Water distribution](index.md).

## Running over time

`water_run(model, state=None, drivers=None, *, duration=None, quality=None, diagnostics=None, **solve_kwargs)`
is EPANET 2.2's extended-period run (`hydraul.c`, `runhyd` and `nexthyd`), on a model built
with the default `tank_scheme="forward_euler"`. From `state` (`initial_state(model)` when
`None`) it repeats, at each step's start time $t$ in whole seconds:

1. **patterns** (`demands`): each junction's demand is its base demand in `"water.sources"`
   times its pattern's multiplier `pattern_factor(net, pattern, t)` (its own pattern, else
   `[OPTIONS] PATTERN`, else a pattern named "1"), with the pattern period
   $\lfloor (t + \text{PATTERN START}) / \text{PATTERN TIMESTEP} \rfloor$; a reservoir's head
   follows its head pattern; a pump's speed pattern sets its speed, 0 closing it and a positive
   speed opening it again (`setlinksetting`);
2. **controls and solve**: the simple controls are applied and the network solved
   (`water_steady`), the junction-pressure controls solving it again up to 10 times;
3. **the step**: the hydraulic step, ended at the next pattern period, the next report time,
   the time a tank fills or empties and the time to the next control that would change a link
   (`timestep`, `tanktimestep`, `controltimestep`), the last two rounded to whole seconds as
   EPANET rounds them, and at the run's end;
4. **the tanks** move by volume over the step, $V \mathrel{+}= q\,\Delta t$, a tank within one
   second's inflow of a limit put on it (`tanklevels`). With `[RULES]` the step is walked in rule
   steps: the first to the next multiple of `RULE TIMESTEP`, the tanks moved and the rules
   checked after each (`apply_rules`); a rule that takes an action ends the step there
   (`ruletimestep`), and the next solve applies the simple controls first.

`quality=None` (the default) also runs EPANET's water quality (`QualityRouter`, after every solve)
when the network defines a mode other than NONE in `[OPTIONS] QUALITY` and the duration is
positive; `quality=False` never does, which is what a hydraulics-only run (a benchmark, a parity
check against EPANET's heads and flows) asks for, and `quality=True` demands it, raising a
`ValueError` naming `[OPTIONS] QUALITY` when there is no mode and `DURATION 0` at a zero
duration. The quality never feeds back: the hydraulics are bit-identical with it on and off.
A batched run routes each instance on its own. The steady tracer (a model built with
`build_model(quality=...)`) is a different thing, a steady layer of the model; `water_run`
refuses it by name ("steady tracer") when EPANET's quality runs, so give `quality=False`.

On a file with a quality mode, `water_run(model)` therefore also routes the quality, runs it
attached when the run is `differentiable=True` (its graph grows with the quality steps, see
[Water quality](quality.md#differentiability)), refuses the steady tracer, and refuses a quality
step under one second. `quality=False` gives the hydraulics-only run.

The results are kept at the report times, `REPORT START` and every `REPORT TIMESTEP` after it
to the duration, as EPANET writes them: each the solve at that time (a tank's level is its
level at the step's start, the flows are those of the step that starts there).

```python
from noodl.apps.water import read_epanet_inp, build_model, water_run

net = read_epanet_inp("Net1.inp")       # 24 h, two tank-level pump controls
model, state, drivers = build_model(net)
run = water_run(model)

print(run.time[:3].tolist(), tuple(run.tank_level.shape))
print([round(float(level), 3) for level in run.tank_level[:4, 0]])   # tank 2, m
pump = run.link_names.index("9")
print([int(t // 3600) for t, s in zip(run.time, run.status[:, pump]) if s == 0])
```

```text
[0.0, 3600.0, 7200.0] (25, 1)
[36.576, 37.511, 38.425, 39.056]
[13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
```

The pump fills the tank until it reaches the 140 ft (42.67 m) trigger, between 12 and 13 h,
and is closed until the level falls to 110 ft; the step that crosses a trigger ends at the
crossing (to the whole second), so each switch is on time. `WaterResults` holds, at every
report time:

| Field | Shape | Meaning |
| --- | --- | --- |
| `time` | `(T,)` | seconds from the start |
| `head`, `pressure` | `(..., T, n_nodes)` | m; the pressure is the head less the elevation (a reservoir's base head, a tank's bottom) |
| `flow` | `(..., T, n_links)` | m³/s |
| `status` | `(..., T, n_links)` | `CLOSED`, `OPEN` or `ACTIVE` as the solve found the link (`link_states`) |
| `setting` | `(..., T, n_links)` | a pump's speed (0 when closed), a PRV's, PSV's or PBV's head (m), an FCV's flow, a TCV's coefficient |
| `tank_level`, `tank_overflow` | `(..., T, n_tanks)` | m above the bottom; m³/s spilt by a tank that may overflow |
| `demand`, `emitter_flow` | `(..., T, n_junctions)` | m³/s delivered (the requested demand, or the pressure-driven delivery) and the emitters' outflow; EPANET reports their sum as the demand |
| `energy` | | the pumps' energy and cost over the run (`PumpEnergy`, [below](#pump-energy-and-cost); `None` for a network without pumps) |

`node_names` and `link_names` give the columns. `duration=` overrides `DURATION`, and
`diagnostics={}` receives `"times"` (every solve's time) and `"newton_iterations"`. The run owns
the driver `"water.time"`; the drivers passed are the base values the patterns scale.

**Batches.** A leading dimension in the state or the drivers is a batch of instances that share
one step: the shortest any of them needs (a rule that acts in one ends it for all). Each
instance then takes the others' step boundaries as well as its own, and so differs from its
own single run by exactly those extra boundaries (each an extra solve of EPANET's explicit
step): two Net1 instances at 0.8 and 1.25 times the demand are each their single run, forced
onto the batch's boundaries, to 1e-9 m, and up to 6.5e-3 m from their plain single runs.

```python
import torch

base = drivers["water.sources"]
batch = {**drivers, "water.sources": torch.stack([0.8 * base, 1.25 * base])}
runs = water_run(model, drivers=batch)
print(tuple(runs.tank_level.shape))
```

```text
(2, 25, 1)
```

Gradients reach the heads, flows, levels and demands through the solves and the tank moves
(a tank's area or volume curve, a demand, a roughness, a setting given as a driver); not
through the step lengths or a switch. The implicit tank schemes (`tank_scheme="backward_euler"`
or `"trapezoidal"`) are refused by name: their step is `water_step`'s, which no rule shortens.

## Pump energy and cost

`WaterResults.energy` is a `PumpEnergy`, the pumps' energy as EPANET 2.2 reports it (`hydraul.c`
`addenergy` and `getenergy`, `output.c` `saveenergy`). Per pump, in `WaterNetwork.pumps` order
(`pump_names`; each field `(..., n_pumps)`):

| Field | Meaning |
| --- | --- |
| `utilisation` | percent of the run's duration the pump is on line |
| `efficiency` | average efficiency over the time on line, percent |
| `energy_per_volume` | kWh per m³ pumped, averaged over the time on line (EPANET prints kWh per million gallons for a file in US units: multiply by 3785.411784 m³ per million gallons) |
| `average_power`, `peak_power` | kW, the first over the time on line |
| `cost` | cost per day |

and `demand_charge` `(...,)`, the peak total power times `[ENERGY] DEMAND CHARGE`.

Every hydraulic step that starts before the duration adds its hours to each pump that is not
closed: the efficiency is the pump's `[ENERGY] EFFICIENCY` curve at the flow `q / speed`
(EPANET's `interp`, the end values beyond the curve, then the speed adjustment
`100 - (100 - e)(1/speed)^0.1`) or the global efficiency, limited to 1 to 100 percent; the power
is `dh q SpGrav / 8.814 / e` in horsepower (feet, cfs; times 0.7457 kW per horsepower), from the
solved head gain, also for a constant-power pump (it works out as its power over the
efficiency); the price is the pump's own `PRICE` (when positive) else the global price, times the
pump's `PATTERN` else the global price pattern, at the pattern period `(t + PATTERN START) /
PATTERN TIMESTEP`. A pump is not on line when its status is CLOSED: closed by a control or a
rule, at speed 0, or shut by the solve at its shut-off head (the energy reads the status, not
the carried speed). EPANET does not shorten the last step to the duration, and neither does the
energy: with a duration of 1:30 and hourly steps the second step counts a full hour (the
utilisation is then 200/1.5 percent, as EPANET reports), and a run of duration 0 counts one
hour. The reported heads and flows are unchanged by that.

A step is charged as EPANET's `addenergy` sees it (`hydraul.c`, `nexthyd`: after `timestep`):
with the solve's flows and junction heads, the tank heads moved to the step's end (`tanklevels`
runs inside `timestep`: a pump next to a tank is charged with the tank's end-of-step head; Net1
with pump 9 discharging into tank 2 agrees to 3.8e-7, 1.9e-2 with the solve's head), and each
pump's status and speed as the rules checked at the step's end leave them. A pump a rule closes
at the end of a step is off over that step (on Net1 with a rule closing pump 9 at 2 h and an
OPEN at 4 h, 3 h of 6 on line, as EPANET). A pump a rule opens there (a STATUS OPEN or a
positive SETTING) is on line over it with the `LinkFlow` EPANET then has: a constant-power pump
its `Q0` of 1 cfs, which `setlinkstatus` and `setlinksetting` write when they reopen it
(`resetpumpflow`); any other pump the flow its closed link carried in the solve, `dh / CBIG`,
which sets its energy per volume (3.9e-7 for a curve, 1.6e-7 for a constant-power pump). Which pumps a rule acted on is read from the actions taken
(`take_actions_per_link`, EPANET's per-action `flag` in `takeactions`), not from the movement of
the carried status: `takeactions` tests the solved status (`LinkStatus`), so a STATUS OPEN or a
changed positive SETTING on a pump the solve shut at its shut-off head (`XHEAD`) is taken, and
charges the step, though the pump was carried OPEN throughout (the four rules' network with its
pump at shut-off under `IF SYSTEM TIME >= 2 THEN PUMP PU1 STATUS IS OPEN`: 95.8 % on line, as
EPANET, 7.4e-7; before, 67.6 %). A step the duration cuts is charged whole, as EPANET does,
with its tanks moved and its rules checked to its own end: a rule that acts after the
duration ends that step there and is charged. A simple control or a pattern acts at the start
of the next step, before its solve, and does not reach the energy of the step before.

