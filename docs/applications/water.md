# Water distribution

Pressurised water distribution: reservoirs and tanks at fixed or slowly varying head, and
junctions with demand, are nodes; full pipes, pumps and valves are edges. Water is conserved at
every junction, and flow is driven by differences in hydraulic head.

Every pipe is full and its head loss is a monotone function of the head difference, so the whole
network is **one** `KirchhoffLayer` whose potential is hydraulic head. That is exactly what
distinguishes it from the free-surface [sewer](sewer.md), whose normal flow is set by slope and
upstream inflow rather than by head difference.

**Units.** Head and elevation m, flow and demand m³/s, pressure head m (with a helper to kPa).

```python
from noodl.apps.water import (
    Junction, Reservoir, Tank, WaterPipe, Pump, Valve, WaterNetwork, WaterOptions,
    build_model, water_steady, initial_state, twoloop,
    pressure_head, to_kilopascal, link_table, tank_inflow,
)
```

![A reservoir and pump feeding a looped distribution network with a tank](../assets/app-water.svg)

Every type this application offers — what it does and how to call it — is listed in its [catalogue](../catalogue/water.md). It is set up and run like every other model; see [Using noodl](../usage.md).

## A worked example

```python
from noodl.apps.water import build_model, water_steady, twoloop

model, state, drivers = build_model(twoloop())
model.check(state, drivers).raise_for_errors(strict=True)
final = water_steady(model, state, drivers)

water = model.refs.water
heads = water.head.named(final[water.head])     # m, by node name
flows = water.flow.named(final[water.flow])     # m3/s, by link name
print(round(float(heads["J1"]), 6), round(float(flows["P1"]), 6))
```

```text
49.267725 0.045
```

To change an input, give it by name over the builder's own values. A demand is a
negative source (m³/s withdrawn; `demand=` does the sign for you), and junctions not named
keep their demand. On a model built with `quality=` the quality sink is the
withdrawal of the current hydraulic solution (a scaled or patterned `water.sources`, or the
delivered demand under `pda=True`), so a tracer entering at the reservoir stays at its
concentration at every demand level. A junction with a positive `water.sources` (an injection
of water) is no sink: the injected water carries no tracer of its own and dilutes what arrives
through the pipes, and any tracer injected with it is `quality.sources` (kg/s). On R → A → B
with the reservoir at 100, an injection of 0.005 m³/s and 0.1 kg/s at A and a withdrawal of
0.01 m³/s at B, both junctions read (0.005 × 100 + 0.1) / 0.01 = 60, and the tracer withdrawn
equals the tracer supplied. A STAGNANT node (no inflow and no outflow, e.g. a dead end whose delivered demand is zero under `pda=True` because the pressure is below the minimum) has no steady concentration without decay: `water_steady` raises `UndefinedSteadyState` naming the node and the cause, as the transport layer does for any stagnant node (a steady solve never invents a value). With a bulk decay (`quality > 0`) the node reads 0; or step the quality in time from an initial concentration. Quality advects through every hydraulic edge kind: pipes, pumps and valves. A pressure-driven model (`pda=True`)
refuses both `demand=` and a `water.sources` value, since its demands are withdrawals set
at build time:

```python
from noodl.apps.water import initial_drivers

more = initial_drivers(model, demand={"J2": 0.012})      # m3/s withdrawn at J2
final = water_steady(model, state, more)
print(round(float(water.head.named(final[water.head])["J2"]), 6))
```

```text
48.641165
```

In the dictionaries the heads are `"water.phi"` (full node order) and the flows `"water.q"`
(link order); a tank's level is its head in `"water.phi"` less its bottom (`tank_levels`),
and the status and setting that the controls carry across steps are the state keys `"water.link_status"` and
`"water.link_setting"` (per link). Both are set by tank and pump name with
`initial_state(model, values=...)`, the levels under the key `"water.tank_level"`, which
`initial_state` reads and writes into the tank heads (it is no state key).
See [Using noodl](../usage.md).

`twoloop()` is a hand-built network: reservoir `R1` at 50 m, six junctions with demands
5/8/6/10/7/9 L/s, eight Hazen-Williams pipes at $C=130$ forming two independent loops. Its heads
are 49.267731, 48.877316, 48.430882, 48.213867, 48.117119 and 48.122936 m, the same as EPANET 2.2
gives for the equivalent `twoloop_si.inp`.

The same network can be read from an EPANET `.inp` file (see [File formats](../formats/epanet.md)):

```python
from noodl.apps.water import read_epanet_inp, build_model, water_steady

net = read_epanet_inp("twoloop_si.inp")
model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)
```

## The objects

| Object | Fields |
|---|---|
| `Junction(name, elevation, demand=0.0, pattern=None)` | A demand node. |
| `Reservoir(name, head, pattern=None)` | Fixed-head boundary. |
| `Tank(name, elevation, init_level, min_level, max_level, diameter, volume_curve=None, overflow=False, min_volume=0.0)` | A cylinder of `diameter`, or the volume curve `volume_curve` (a `curves` key, level m against volume m³); `overflow` lets it spill when full. |
| `WaterPipe(name, u, v, length, diameter, roughness, minor_loss=0.0, status="OPEN")` | A full pipe. |
| `Pump(name, u, v, curve, power=None, speed=1.0)` | Edge oriented **suction → discharge**; on a head `curve` (a `curves` key) or at a constant `power` (kW), at relative `speed`. |
| `Valve(name, u, v, diameter, kind, setting, minor_loss=0.0, curve=None)` | TCV, FCV, PBV, GPV, PRV or PSV (`curve`: a GPV's head-loss curve). |
| `WaterOptions(demand_model="DDA", minimum_pressure=0.0, required_pressure=0.1, pressure_exponent=0.5, ...)` | Carried from the file's `[OPTIONS]`. |

`WaterNetwork` collects them plus `curves`, `patterns`, `controls` and the timing options.
`nodes()` returns junctions, then reservoirs, then tanks; `links()` returns pipes, pumps, TCVs,
FCVs, PBVs, GPVs, PRVs, PSVs in that order; `validate()` refuses anything out of scope by name.

## `build_model`

```py
model, state, drivers = build_model(
    net,
    headloss=None,      # 'H-W' (default) or 'D-W'; None takes the network's own
    pda=None,           # pressure-driven demand; None takes net.options
    p_min=None, p_req=None, exponent=None,
    quality=None,       # bulk decay coefficient, 1/day; adds a quality TransportLayer
    coupling="pingpong",
    dt=None,            # overrides net.hydraulic_timestep for the tank closure
    friction="composite",  # D-W friction law: 'composite' (default) or 'colebrook'
    pipe_form=None,     # 'flow' (resistance form; the default) or 'potential'
    tank_scheme="forward_euler",
    valve_smoothing=0.0,  # m: smooth the PRV/PSV switch over a band of this width
)
```

`pipe_form` is how the pipes enter the solve. In the **resistance form** (`"flow"`, the default
except with `friction="colebrook"`) each pipe is `dp = h(q)` (`FlowControlledPipe`) with its
flow an unknown, solved from EPANET's own first iterate (every pipe
at 1 ft/s). In the **potential form** (`"potential"`) each pipe's flow is the inverted law
$q = g(dp)$, with no flow unknowns. Both give the same solution to round-off; they differ in how
Newton gets there. The potential form's slope grows without bound as a pipe's flow goes to
zero, so on Net1's 24 h run the step where a control closes the pump and leaves pipe 10
dead-ended needs 136 Newton iterations; the resistance form's slope goes to EPANET's `RQtol`
floor instead, and no step needs more than 6. The whole 24 h run over eight demand
multipliers (`benchmarks/water_eps.py`) takes 2.1 times less time (1.84 s against 3.94 s, medians
of five interleaved runs).

`water_steady(model, state, drivers, **solve_kwargs)` wraps `model.steady` with defaults
`atol=rtol=1e-11`, `max_iter=200` (both empirically tuned, the iteration cap for the
potential form; caller overrides win), starts a cold start of the resistance form (every pipe
flow exactly zero) from `model.initial_flows`, and the pumps from their design flows (EPANET's
own first iterate), then checks
every converged power-function pump's flow against its own `q_max()` at its speed and raises
naming the pump and batch instance if it is exceeded — rather than silently accepting an
extrapolated point off the end of the curve. (A segmented curve's last segment and a
constant-power pump are extended without a limit, as EPANET extends them.) A converged pump
carrying flow backwards is refused by name too. The flows are seeded, as EPANET seeds them,
rather than left to the layer's linear guess or its fallback ladder
([When Newton fails](../concepts/solvers.md#when-newton-fails-the-fallback-ladder)): at zero
flow every resistance-form pipe sits on its `RQtol` slope floor, and the layer's linear guess,
taken there, carries the head difference between a reservoir and a tank over that floor (on
Net1 a flow of 1.9e7 m³/s, from which the storage ramp settles nowhere; the two-loop network,
with one reservoir, has a sane guess). From the seeded flows Newton needs 5 iterations on
both.

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

This default changes what an existing call does on a file with a quality mode: `water_run(model)`
now also routes the quality (on Net1, 0.84 s more for 24 h), runs it attached when the run is
`differentiable=True` (its graph grows with the quality steps, see Differentiability below),
refuses the steady tracer, and refuses a quality step under one second. `quality=False` restores
the hydraulics-only run.

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
onto the batch's boundaries, to 1e-9 m, and up to 6.5e-3 m from their plain single runs
(`tests/apps/water/test_simulate.py::test_batched_run_equals_single_runs`).

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

### Pump energy and cost

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
which sets its energy per volume (`test_energy_under_rules_matches_epanet`: 3.9e-7 for a curve,
1.6e-7 for a constant-power pump). Which pumps a rule acted on is read from the actions taken
(`take_actions_per_link`, EPANET's per-action `flag` in `takeactions`), not from the movement of
the carried status: `takeactions` tests the solved status (`LinkStatus`), so a STATUS OPEN or a
changed positive SETTING on a pump the solve shut at its shut-off head (`XHEAD`) is taken, and
charges the step, though the pump was carried OPEN throughout (the four rules' network with its
pump at shut-off under `IF SYSTEM TIME >= 2 THEN PUMP PU1 STATUS IS OPEN`: 95.8 % on line, as
EPANET, 7.4e-7; before, 67.6 %). A step the duration cuts is charged whole, as EPANET does,
with its tanks moved and its rules checked to its own end: a rule that acts after the
duration ends that step there and is charged. A simple control or a pattern acts at the start
of the next step, before its solve, and does not reach the energy of the step before.

## Water quality

`water_run` carries EPANET 2.2's water-quality simulation along with the hydraulics: the
Lagrangian procedure of `quality.c`, `qualroute.c` and `qualreact.c`, transcribed, so it
reproduces EPANET's answer and not merely the advection-reaction equation's. EPANET moves
volumes of water through pipes as segments, merges neighbours that differ by less than a
tolerance and reacts each by an explicit Euler step; an Eulerian discretisation would differ
from it however fine. The hydraulics are untouched: quality reads each step's flows, link
states and delivered demands and never feeds back, so a run is bit-identical with it on and
off. The quality is read from the `.inp` file ([format](../formats/epanet.md)) into
`WaterNetwork.quality` (a `WaterQuality`, in SI) and run by `QualityRouter`
(`noodl.apps.water.quality`).

**Modes** (`[OPTIONS] QUALITY`). `CHEMICAL` (named, with its units, `Chlorine mg/L`;
concentrations stay in the file's unit), `AGE` (hours: every segment gains the quality step as it
moves) and `TRACE <node>` (percent of the water that came from that node: the node is held at
100 and every other starts at 0 whatever the file's initial qualities say, except a tank's own
segments). `NONE`, or a duration of 0, runs no quality.

**Reactions** (`[REACTIONS]`). In the bulk water, order `n` with a limiting concentration
(`bulkrate`: growth towards or decay to the limit; a negative order is Michaelis-Menten). At
the wall, order 0 (mass per area per day) or 1 (length per day), limited by mass transfer from
the Sherwood number (laminar, transitional, turbulent) with the diffusivity relative to
chlorine's, `DIFFUSIVITY 0` switching the limitation off; a pipe with no wall coefficient of
its own takes one from the roughness correlation (Hazen-Williams, Darcy-Weisbach,
Chezy-Manning). In tanks, one global order and a coefficient per tank (default: the global bulk
coefficient). A reaction step is `c = max(0, c + (rbulk + rwall) dt)`; pumps and valves never
react. The wall coefficients are recomputed once per hydraulic step from that step's flow.

**Sources** (`[SOURCES]`). `CONCEN` (at a junction only where its demand is negative), `MASS`
(per minute), `SETPOINT` and `FLOWPACED`, each with an optional pattern. A reservoir whose
source goes to zero keeps its last quality, as in EPANET.

**Tank mixing** (`[MIXING]`). `MIXED`, `2COMP` (a mixing zone and a stagnant zone, the zone
fraction given), `FIFO` and `LIFO`. A tank's quality tracks its own segment volumes, not the
hydraulic tank volume.

**Running.** `quality=None` (the default) runs it when the network defines a mode and the
duration is positive, [as above](#running-over-time). The results carry:

| Field | Shape | Meaning |
| --- | --- | --- |
| `node_quality` | `(..., T, n_nodes)` | each node's quality at the report times, in the file's units; a tank's is the quality of the water it releases |
| `link_quality` | `(..., T, n_links)` | the volume-weighted mean over the link's segments (the mean of its end nodes for a link with none) |
| `quality_units` | `str` | `mg/L`, `hrs`, `%` or the file's |
| `mass_balance` | `QualityMassBalance`, fields of shape `(...,)` | initial, inflow, outflow, reacted and final mass and their ratio, as EPANET's report writes them, in physical units (a chemical in its mass unit, AGE and TRACE as quality times m³) |

All are `None` when no quality ran. `diagnostics={}` additionally receives
`diagnostics["quality"]`, one entry per instance: the node order of every sort, the cycle
breaks, the flow reversals, the segment counts per link and quality step, and the per-node
inflow and outflow.

```python
from noodl.apps.water import read_epanet_inp, build_model, water_run

net = read_epanet_inp("Net1.inp")        # chlorine, bulk -0.5 per day, wall -1 ft per day
model, state, drivers = build_model(net)
run = water_run(model, duration=6 * 3600)

print(run.quality_units, tuple(run.node_quality.shape), tuple(run.link_quality.shape))
print([round(float(c), 3) for c in run.node_quality[:4, run.node_names.index("11")]])
print(round(float(run.mass_balance.ratio), 4))
```

```text
mg/L (7, 11) (7, 13)
[0.5, 0.444, 0.866, 0.864]
1.0
```

**Learnable inputs.** Optional overrides of what the file says: `state["quality.initial"]`
(n_nodes, quality units) and the drivers `"quality.bulk_coefficient"` and
`"quality.wall_coefficient"` (n_pipes, in `net.pipes` order; (quality unit)^(1-n)/s for a
bulk reaction of order n, 1/s for order 1; m/s for a wall coefficient of order 1, quality
units per m² per s of order 0), `"quality.tank_coefficient"` (n_tanks, (quality
unit)^(1-n)/s for the bulk order n, 1/s for order 1) and `"quality.source_strength"`
(n_nodes, scaled by the source's pattern; a MASS source in mass per second). The file gives
its coefficients per day (and in its own length units); the reader converts them to these
per-second SI values. A wrong shape, or any of them on a run without quality, is refused
by name.

**Differentiability.** Volumes and concentrations stay in the autograd graph; only the
decisions are made on detached values: whether a segment is used up, whether two segments merge
(`|c1 - c2| < Ctol`), the flow direction and the node order. Segment volumes are `|q| dt` from
the solve's flows, so the quality is piecewise differentiable with respect to the reaction
coefficients, the source strengths, the initial quality and the hydraulic parameters (demands,
roughnesses, a pump's speed) through the travel times. It jumps where a decision flips: where a
merge, a reversal or the node order differs between `p - h` and `p + h`, a central difference
is not the derivative. Autograd is on only when a quality input or the solve's flows require
gradients (`differentiable=True` for the hydraulics); otherwise the router runs detached, since
the graph grows linearly with the quality steps (about 60 nodes per step for a coefficient, 135
for a hydraulic parameter, on the verification network).

**Batches.** A batched run builds one router per instance (each has its own segments and node
order) and stacks the results `(..., T, n)`; the `mass_balance` fields have shape `(...,)`. The
hydraulic step is shared by the batch, [as above](#running-over-time): an instance takes the
others' step boundaries, so its quality is its own single run's only when no instance has an
event (a tank filling, a control) that the others lack. Run instances one at a time to
reproduce EPANET's sequence for each.

**EPANET behaviours transcribed as they are**, not corrected:

- A check-valve pipe (`CV`) holds no initial segment and never reacts; a pump's or valve's
  chain is filled and emptied within the step and never reacts.
- A flow below `Q_STAGNANT` (0.005 gpm) has no direction: the link is left out of the node
  order and its water moves from its first node. A closed link has zero flow. Segments are
  reversed only when the direction changes from positive to negative or the reverse, not
  through a stagnant interval.
- A `2COMP` tank with no net flow loses the mass of its stagnant zone (the zone's volume is
  set to 0), so the mass balance ratio is below 1 there, in EPANET and here alike.
- The last hydraulic step is transported only if it ends within the duration: EPANET does not
  cut it at `DURATION`, so a last step that overruns it is not routed.
- A reservoir's `[QUALITY]` is its quality, kept until a source overwrites it; a reservoir
  without a source adds `quality x outflow` to the mass balance.
- EPANET's rounded unit factors. It divides a file's flows by a rounded factor (28.317 L/s per
  cfs) but its lengths by the exact foot, so the volume its quality moves is 5.4e-6 smaller than
  the exact SI volume (in LPS files; other units differ), and a `MASS` source and an order 0 wall
  coefficient use 28.317 L per ft³ against 1000 L per m³. These are transcribed. The mass
  balance reports physical mass, so it carries no unit factor; EPANET's terms are divided by it
  when compared.
- Segments merge when their quality differs by less than `Ctol` (`TOLERANCE`, default 0.01 in
  the quality unit). The decision is made in EPANET's internal unit (a chemical divided by
  28.317 L per ft³), where a tie at exactly `Ctol` can be decided otherwise than in mg/L.

**Refused by name:** a bulk or tank reaction order between 0 and 1 (EPANET's `bulkrate` raises
zero to a negative power at zero concentration and gives NaN or infinity: there is no finite
behaviour to reproduce), `ORDER WALL` other than 0 or 1, a Darcy-Weisbach roughness not below
the diameter under a roughness correlation (EPANET runs it, with an unphysical value), a
quality step that rounds to 0, a quality input on a run without quality, and the steady tracer
alongside the EPANET quality (give `quality=False` for that model).

## The physics

### Hazen-Williams

$$
h_L = K q^{1.852} + m q^2, \qquad
K = \frac{10.666829500036\, L}{C^{1.852} d^{4.871}}, \qquad
m = \frac{K_{\text{minor}}}{2 g A^2}
$$

The SI constant was re-derived from EPANET's own US constant 4.727 and matches `wntr`'s `hw_k` to
1.04e-9 relative. For a network read from a file, $K$ carries EPANET's own rounded flow-unit
factor, as Chezy-Manning and Darcy-Weisbach do: EPANET divides a file's flows by `LPSperCFS =
28.317`, `GPMperCFS = 448.831`, ... (`types.h`; exact 28.3168466 and 448.8311688) and evaluates
the law in those cfs, so $K$ is multiplied by $(c \cdot 0.3048^3)^{1.852}$ with $c$ EPANET's cfs
per m³/s (`HazenWilliams(cfs_per_m3s=...)`), and the minor loss is EPANET's $0.02517\,K/d^4$
(ft, cfs) with the same factor. A network built by hand has no flow unit and keeps the exact
conversion and $m = K_{\text{minor}}/(2 g A^2)$. With the exact factor an LPS file's heads are
1.0e-5 of the losses off EPANET's (3.1e-5 relative on a network with 110 m of losses); with the
rounded one the two-loop heads agree to 5.3e-8. With no minor loss the inversion is closed form; with one, it is a batched
monotone root solve bracketed by the pure Hazen-Williams flow as an upper bound.

Below `dp_transition` (default $10^{-9}$ m) the law blends to a laminar linear form to keep the
Jacobian finite. That value is empirically chosen: at 1e-9 the 24-hour tank trajectory is
converged to 8.18e-5 m against EPANET, and Newton iteration counts run 50/93/136/179 at
1e-3/1e-6/1e-9/1e-12.

### Chezy-Manning

`HEADLOSS C-M` is EPANET 2.2's resistance (`hydcoeffs.c`, `resistcoeff`; `input1.c`, `Hexp = 2`),
in feet and cfs with Manning's roughness $n$ (dimensionless):

$$
h_L = (R + m) q^2, \qquad R = \left(\frac{4 n}{1.49\,\pi d^2}\right)^{2} \left(\frac{d}{4}\right)^{-1.333} L
$$

The exponent $-1.333$ is EPANET's literal, not $-4/3$. `ChezyManning` converts $R$ to metres per
$(\mathrm{m^3/s})^2$ with the file's own flow factor (as Darcy-Weisbach does), takes EPANET's
`RQtol` line below the gradient floor, and carries the minor loss $m$ as `K / d^4` with EPANET's
rounded constant. It works in metres of head, so, like Hazen-Williams, it refuses a
non-default `VISCOSITY` by name; a `SPECIFIC GRAVITY` only converts the file's pressures to
heads, which the reader does. It enters the solve through
`FlowControlledPipe` like Hazen-Williams, so check valves and closed pipes work unchanged.

### Darcy-Weisbach

`CompositeDarcyWeisbach` (`friction="composite"`) is EPANET 2.2's own composite law:
Hagen-Poiseuille, then Dunlop's transition cubic, then Swamee-Jain. It is transcribed from its source
(`hydcoeffs.c`, `DWpipecoeff` and `frictionFactor`), in EPANET's internal feet and cfs:

$$
h_L = \begin{cases}
(16 \pi \nu d R + m \lvert q \rvert)\, q & Re \le 2000 \quad \text{(Hagen-Poiseuille)} \\
(f(Re) R + m)\, \lvert q \rvert q & Re > 2000
\end{cases}
\qquad R = \frac{L}{2 \cdot 32.2\, d A^2}, \quad m = \frac{0.02517\, K_{\text{minor}}}{d^4}
$$

with $f$ Dunlop's cubic interpolation for $2000 < Re < 4000$ and Swamee-Jain above. The
composite is $C^1$: the cubic matches $64/Re$ and Swamee-Jain in value and slope at its ends.
EPANET's constants are kept as EPANET has them, because they are part of what it computes:
$g = 32.2$ ft/s² (not 32.174), the rounded minor-loss factor 0.02517, water at
$\nu = 1.1 \times 10^{-5}$ ft²/s times `[OPTIONS] VISCOSITY`, and the flow-unit factor EPANET
divides a file's flows by (`LPSperCFS = 28.317`, `GPMperCFS = 448.831`, ...;
`WaterOptions.flow_units` carries the file's unit). EPANET has no low-flow linearisation for D-W:
the laminar branch is already linear through $q = 0$, so the element needs no transition blend.
The flow is found by a safeguarded Newton inversion of the odd law $h_L(q)$ with EPANET's own
analytic gradient, and differentiated by the implicit-function rule.

This is the default whenever the head loss is D-W, including every `.inp` read with
`HEADLOSS D-W`, because the application's reference is EPANET. `friction="colebrook"` selects
the framework's generic `Duct` instead (implicit Colebrook, a straight laminar line to its
Re = 2000 value, water at $1.002\times10^{-3}/998.2$ m²/s, standard gravity); it is a different
law and does not reproduce EPANET (3.6e-4 on the two-loop heads, up to 81 % on a pipe's head
loss near Re = 2000).

### Pumps

Each pump takes the curve EPANET 2.2 gives it (`input2.c`, `updatepumpparams`), and its head
gain at flow $q$ and relative speed $w$ follows EPANET's `pumpcoeff` (`hydcoeffs.c`), with the
affinity laws for the speed:

| Pump in the file | Curve | Head gain |
|---|---|---|
| `HEAD` curve of one point $(q_d, H)$ | `PumpCurve`, `three_point_curve`: EPANET's points $(0, 1.33334 H)$, $(q_d, H)$, $(2 q_d, 0)$ fitted as three points: $h_0 = 1.33334 H$, $n = 1.99998$, $r = 0.33334 H / q_d^n$ | $w^2 h_0 - r w^{2-n} q^n$ |
| `HEAD` curve of three points, the first at zero flow | `PumpCurve`, `power_curve` (EPANET's `powercurve`): $h_0$ the first head, $n = \ln\frac{h_0 - h_2}{h_0 - h_1} / \ln\frac{q_2}{q_1}$, $r = (h_0 - h_1)/q_1^n$ | $w^2 h_0 - r w^{2-n} q^n$ |
| any other `HEAD` curve | `PiecewisePumpCurve`: straight segments (`curvecoeff`), the end segments extended | $w^2 h(q/w)$ |
| `POWER` $P$ | `ConstantPowerPump`: $8.814\,P/q$ in feet, horsepower and cfs | $8.814\,P w^3 / q$, converted to SI |

EPANET's own rounded 1.33334, not the manual's 133 % with exact thirds (which would force
$n = 2$, as wntr's `get_head_curve_coefficients` has it): with $n = 2$ a pump at zero flow
sat $6.7 \cdot 10^{-6} H$ below EPANET's (1.05e-6 relative at 63 m) and Net1's flows
missed EPANET's by 2.9e-6 relative instead of 2.6e-7 (MEASURED).
`q_max = w (h_0/r)^{1/n}` is a power function's flow at zero head gain.

Every pump enters the solve in its **head-gain form**, whatever the `pipe_form`, as EPANET and
wntr write it: `FlowControlledPump`, $H_{\text{discharge}} - H_{\text{suction}} =
h_{\text{gain}}(q)$ with the pump's flow an unknown. One law holds every pump of a network,
each on its own curve. The gain is strictly decreasing in $q$, so every head difference has
exactly one flow on the curve:

- where a power function's slope falls below EPANET's `RQtol` (at low flow when $n > 1$) or
  rises above its `CBIG` (at low flow when $n < 1$), and where a constant-power pump's slope
  $8.814 P w^3/q^2$ leaves `[RQtol, CBIG]`, the gain is the curve's tangent there (wntr's
  `get_pump_line_params` construction), continuous in value and slope. EPANET's own lines
  there are its iteration's linearisation, not a decreasing curve; the two agree wherever a
  running pump works;
- a segmented curve whose first point is above zero flow shuts at that point's head, as
  EPANET's `pumpstatus` does (below the point's flow the gain is that head plus a line of
  slope `RQtol`).

The pump's speed is the per-link driver `"water.setting"` (its `SPEED` in the file at the
build), so a calibration can differentiate through it. Its status is `"water.status"`. The
residual of each pump, with $r = dp + h_{\text{gain}}(q)$, is

- $\kappa q$ (no flow) when its status is `CLOSED` or its speed is 0;
- otherwise $\max(-\kappa q, r)$: the pump runs on its curve with $q \ge 0$, or it is shut,
  $q = 0$, where the head asked of it is at or above its shut-off head $h_{\text{gain}}(0)$.
  EPANET closes such a pump (`pumpstatus`, `XHEAD`) and leaves it the closed-link leak
  `excess / CBIG`; here there is no leak: its row is $\kappa q = 0$, so its flow is zero to round-off, and `link_states` reports it `CLOSED`.

The branch is selected as for a check valve (`FlowControlledPipe`): the no-flow row only where
$r < -\kappa q$ and the open row's own Newton step would reverse the flow, never at a tie (a
pump with no demand behind it runs at zero flow on its curve, `OPEN`, as EPANET reports it),
and never where shutting the pump, together with the check valves the same iterate shuts
(`ShutLinks`, which the pipes' law fills first), would cut a group of nodes off from every
reservoir and tank. $\kappa$ is 1 m per m³/s. The Jacobian is the closed form `(1, -h'_{\text{gain}})` on
the curve's row.

`PumpCurve.flow` still gives the inverted curve, $q = w((h_0 + dp/w^2)/r)^{1/n}$, with a laminar
blend near shut-off, but the solve does not use it: its slope is infinite at shut-off and
exactly zero beyond it. A Newton iterate beyond shut-off then has no Jacobian entry between the
pump's two sides, and with a pressure-driven demand downstream only the demand's own small slope
ties those heads to a boundary; the next step throws them far outside the curve and the system
goes singular. That was measured on a reservoir, a pump and a pipe feeding a PDA junction
(singular from the fifth iterate on). In the head-gain form the pump's row ties its two heads
together on every iterate, and the same network converges quadratically, in 5 Newton
iterations (19 with `pipe_form="potential"`).

### Valves

A **TCV** (`ThrottleValve`) is a minor loss whose coefficient is its setting, and an **FCV**
(`FlowControlValve`) a fixed flow. A **PBV** (pressure breaker) and a **GPV** (general purpose
valve) follow EPANET 2.2's own coefficient routines (`hydcoeffs.c`, `pbvcoeff` and `gpvcoeff`)
as `BreakerValve` and `GeneralPurposeValve`. All four are implicit laws, whose flow is an
unknown of the layer:

- **PBV:** the head drop from the valve's start node to its end node is its setting,
  whichever way the water flows. Where the open valve's minor loss $m q^2$ exceeds the setting,
  or the setting is 0, it is an open valve instead, $h = m q |q|$, with EPANET's minor-loss
  coefficient $0.02517 K / d^4$ (ft, cfs) and its rounded flow-unit factor.
- **GPV:** $h = \operatorname{sgn}(q)\,(h_0 + r |q|)$ on the segment of its head-loss curve
  that brackets $|q|$ (the first or last segment, extended, outside the curve), with
  $\operatorname{sgn}(0) = +1$.

A valve's flow is an unknown of the solve (sparse LU).

A **PRV** (`PressureReducingValve`) holds the head at its end node at no more than
$H_\text{set} = z + s$ (the node's elevation $z$ plus the setting $s$); a **PSV**
(`PressureSustainingValve`) holds the head at its start node at no less than that node's
$z + s$. Each is one exact complementarity condition on the absolute head at its controlled
node (the law reads its end nodes' heads, `ConstitutiveLaw.reads_heads`), solved in the same
Newton iteration as everything else, with no outer loop over states and no memory of the
previous one (`docs/plans/water-prv-psv-design.md`):

$$\min\bigl(\kappa q,\ \max(\text{held},\ K_m q|q| - \Delta\phi)\bigr) = 0,$$

with $\text{held} = H_v - H_\text{set}$ for a PRV and $H_\text{set} - H_u$ for a PSV, the
controlled head beyond its setting. The selected row is the valve's state: $\kappa q$
(no flow) is `CLOSED`, the held row `ACTIVE` (the controlled head at its setting), the
minor-loss row `OPEN` (the controlled head short of its setting, the valve fully open).
That is EPANET 2.2's three states (`hydstatus.c`, `prvstatus` and `psvstatus`) without
their hysteresis. Inside EPANET's `HTOL` band (1.524e-4 m below the switch) EPANET keeps an
active PRV active with the downstream head at its setting above the upstream head, a
negative drop through the valve; noodl gives the ideal valve there, `OPEN` with no drop, and
the heads differ by less than `HTOL`.

The rows are selected with `torch.where` and fixed tie rules: at a tie of the held and open
rows the held row; the no-flow row only where the flow is not forward. With the fixed
$\kappa$ alone (1 m per m³/s for a valve without a minor loss) an iterate carrying 20 L/s
forward through a PRV whose downstream head is 7 m above its setting picks the no-flow row
and cuts off the junctions downstream, with their demand (MEASURED: a singular step). A
side of the valve with no demand and no other way to a reservoir or tank (a dead end)
carries no flow; the valve then takes a row that ties that side's head, with EPANET's
state: a dead end on a PRV's end leaves it `ACTIVE` at its setting (or `OPEN` at the
upstream head below it), on a PSV's end `OPEN` at its upstream head (or `CLOSED` where that
is below its setting); a dead end on either valve's start closes it (no water flows out of
a group without demand). A closed valve's dead end is held at the head of its other side,
the `1 / CBIG` limit. A valve with both sides dead (a part without demand that closed
links cut off) is refused by name.

The gradient is the implicit-function one of the selected row: through an `ACTIVE` PRV the
downstream head moves one for one with the setting, through an `OPEN` or `CLOSED` one not at
all. The solution is continuous across a switch and its derivative jumps there.
`build_model(valve_smoothing=eps)` (metres) replaces the `min` by the Fischer-Burmeister
function $a + b - \sqrt{a^2 + b^2 + 2\epsilon^2}$ and the `max` by its smooth counterpart, for an
optimiser that must cross a switch: the derivative is then continuous, and the answer moves
by $O(\epsilon)$ near a switch (within 2e-3 m of the exact one at $\epsilon$ = 1 mm,
`test_smoothed_prv_converges_to_exact`).

`validate()` refuses, by name, a PRV or PSV whose controlled node is a reservoir or tank,
whose setting or minor loss is negative, and two valves that hold one node twice: two PRVs
sharing their end node or in series, two PSVs sharing their start node or in series, a PSV
starting at a PRV's or an FCV's end, a PRV ending at an FCV's start (EPANET's own errors 219
and 220, `valvecheck`); the error names both valves. Unlike EPANET, a PRV or PSV may start or
end at a reservoir or tank on its other side.

### Link status, settings and check valves

Every link's status, setting and allowed direction are drivers, one value per link in
`model.link_names` order (pipes, pumps, then TCVs, FCVs, PBVs, GPVs, PRVs, PSVs):

| Driver | Values |
|---|---|
| `"water.status"` | `CLOSED` 0, `OPEN` 1, `ACTIVE` 2 (`noodl.apps.water.controls`); `ACTIVE` only for a valve, which then acts on its setting |
| `"water.setting"` | a pump's relative speed, a PBV's head drop, a PRV's or PSV's pressure head (m; the layer's potential unit), an FCV's flow (m³/s), a TCV's loss coefficient; unused for pipes and GPVs |
| `"water.one_way"` | 0 both directions, +1 forward flow only, -1 reverse flow only |

`build_model` fills them from the file: a `CLOSED` pipe or a `[STATUS]` word sets the status, a
`CV` pipe is forward-only, a valve's setting (or a numeric `[STATUS]` value) is its setting,
and every valve starts `ACTIVE`. On a pump `[STATUS]` acts as EPANET's `changestatus`: a number is
its speed (the pump `OPEN` at it, `CLOSED` at 0), `OPEN` runs it at speed 1, `CLOSED` keeps its
speed. They are set by link name like any other driver:

```python
from noodl.apps.water import CLOSED, link_states

model, state, drivers = build_model(twoloop())
closed = initial_drivers(model, values={"water.status": {"P7": CLOSED}})
final = water_steady(model, state, closed)
print(link_states(model, final, closed).tolist())
```

```text
[1, 1, 1, 1, 1, 1, 0, 1]
```

A `CLOSED` link carries no flow: its residual is $\kappa q$. A valve whose status is `OPEN`
is its minor loss, $h = K_m q|q|$, whatever its setting. A check valve is one exact
complementarity condition, solved in the same Newton iteration as everything else, with no
outer loop over statuses: $\max(-\kappa q,\ \Delta\phi - h(q)) = 0$, either the pipe flows
forward with $\Delta\phi = h(q)$, or it holds $q = 0$ against a reverse head (the design is
in `docs/plans/water-prv-psv-design.md`). $\kappa$, a pipe's own head-loss slope at 1 ft/s,
only scales the row. Newton picks the no-flow branch only where the open equation's residual
is below both $-\kappa q$ and $-h'(q)\,q$: with $\kappa$ alone, an iterate that overshoots
forward closes an open check valve, sets every pipe in series with it to zero flow (to round-off),
and cycles (`FlowControlledPipe`). An iterate that would close every check valve feeding a
junction, which leaves the step singular, keeps open the one whose open equation predicts the
most forward flow, so the step is taken; `water_steady` then refuses a converged check valve
carrying flow the wrong way. Closed links and check valves need the pipes' resistance form
(`pipe_form="flow"`, the default); `pipe_form="potential"` refuses them.

`link_states(model, state, drivers)` gives every link's state in a solution: `CLOSED`, `OPEN`
or `ACTIVE`, the codes `wntr` reports EPANET's in (an `ACTIVE` PRV or PSV: the row its
condition selects there). `water_steady` refuses, before it solves, a
junction with a demand that closed links cut off from every reservoir and tank, and one that
no water reaches through the open links in the directions check valves and full or empty tanks
leave them (a pipe into an empty tank only fills it). A group of
junctions without demand that is cut off at the solution, by closed links or by check valves
holding no flow, has no head of its own in the exact equations: any head that keeps those
valves shut solves them. EPANET gives each closed link the conductance $1/C_\text{BIG}$ and
opens a check valve whose upstream head is the higher, and as $C_\text{BIG}$ grows its
heads tend to the solution of the convex program

$$\min_H \tfrac12 \sum_{(a,b)} (H_a - H_b)^2 \quad \text{subject to} \quad H_u \le H_v
\text{ for every check valve } u \to v,$$

the sum over every closed link and every check valve holding no flow, with the heads outside
the group at the solve's own. It is the resistor-and-diode network with one conductance per
link and an ideal diode across each check valve: the constraints' multipliers are the diodes'
flows, so a check valve is `OPEN` where its multiplier is positive (it ties its two ends) and
`CLOSED` where it is zero. `water_steady` sets the group's heads to that solution, with every
flow into it zero to round-off (`noodl.apps.water.elements.PIN_KEY`), by the dual active-set
method of Goldfarb and Idnani on the group's nodes alone (`noodl.solvers.qp.solve_qp`: the objective
rises strictly over every full step and no active set returns, so it ends). For a single
junction behind shut valves that is the plain average of the far ends; a valve the average
would drive is open and the junction takes its far end's head; and a valve that opened in an
intermediate active set is shut again where a later one leaves it with nothing to carry. The
heads are a linear map of the solved heads around them for the active set found, so a
gradient reaches them.

The TCV's coefficient is EPANET's own, $0.02517\,K/d^4$ in feet and cfs: $8/(32.2\,\pi^2)
= 0.025173$ (EPANET's 32.2 ft/s² gravity) rounded to four figures, 9.2e-4 below the exact
$8/(g\pi^2) = 0.0251933$ a `MinorLoss` uses.

### Tanks and controls

Every tank is a `LinearStorage` of its cross-section area $A$ in the water layer
(`noodl.NodeStorage`), or, when any tank of the network has a volume curve, every tank is one
`TankVolumeCurve` (a cylinder as the two-point curve of its area); a tank is an ordinary node
of `"water.phi"`, its level is its head less
its bottom (`tank_levels`), and the flow into it is `"water.storage_flow"`, which `tank_inflow`
returns. `build_model(tank_scheme=...)` chooses how the level moves.

With the default `tank_scheme="forward_euler"` the tank steps by EPANET's own method. It is a
boundary node of the layer, held at its step-start head through the hydraulic solve, and the
step then moves it by explicit Euler, $H \mathrel{+}= q\,\Delta t / A$
(`KirchhoffLayer.advance_storage`). The held head is the tank's in the step-start
`"water.phi"`, which `Model` passes to the solve as `history`, so `"water.phi_boundary"` gives
the reservoirs' heads only. `TankLevels` is a `Model` closure declaring
`state_keys = ("water.link_status", "water.link_setting")`. Its `__call__` applies the controls
and returns updated drivers, among them every link's `"water.status"`: the pumps' and the
controlled links' from `"water.link_status"`, every other link's from the drivers' own (and
`"water.setting"` for a link whose setting a control has changed). `TankLevels.control_time`
implements EPANET's adaptive step-shortening to the next control.

Every form of `[CONTROLS]` is modelled (`noodl.apps.water.controls.apply_controls`, EPANET's
`controls` and `pswitch`), on any link:

| Form | When it acts |
| --- | --- |
| `AT TIME t` | at the step that starts at `t` seconds; the driver `"water.time"` is the step's start (`water_run` sets it; the builder creates it when the network has such a control) |
| `AT CLOCKTIME t` | when `(time + START CLOCKTIME) mod 24 h` equals `t` |
| `IF NODE <tank> ABOVE or BELOW v` | at the start of a step, on the tank's volume, with one second of the previous solve's inflow as margin (EPANET's `controls`: $V \ge V(v) - |q|$ for ABOVE) |
| `IF NODE <junction> ABOVE or BELOW v` | after a converged solve, on the pressure head within EPANET's `Htol` (0.0005 ft) of `v`: a link it switches is solved again (`water_steady`; at most 10 times) |

The action is `OPEN`, `CLOSED` or a number. A number on a valve makes it `ACTIVE` at that
setting (a PRV's or PSV's head, an FCV's flow, a TCV's coefficient, in the units of
`"water.setting"`); on a pump it is the relative speed, and 0 closes it; `OPEN` runs a pump at
speed 1, as EPANET does. The controls are tested in file order and the last one that holds wins.
A step ends at the next time control and at the next tank level that reaches a control's trigger
(`next_control_time`, rounded to whole seconds), but only for a control that would change its
link's status or setting:
a pump already running does not split a step at its `OPEN AT TIME`, as in EPANET. Junction-pressure
controls need `tank_scheme="forward_euler"` (or no tanks): `water_step` does not repeat the solve.

That shortening is not optional. A fixed one-hour step was measured to diverge from EPANET by
**2.07 m** on Net1, because the pump switches an hour late; with event shortening the same
trajectory matches to **2.5e-5 m**. The margin of the level test matters as much: a crossing
rounded to whole seconds may end half a second short, and with a strict level test instead
the pump switches a step late (MEASURED: 0.76 m off on Net1 with time controls added). The
previous solve's inflow is the driver `"water.tank_inflow"`, which `water_steady` takes
from the state it starts from (its `"water.storage_flow"`: the previous solve's at a step
start, zero at `initial_state` as at EPANET's t = 0) when the drivers carry none.
`link_states` reads it the same way, so for a solved state it tests the controls against that
solve's own inflow; `water_run` reads its statuses with the drivers the solve took.

[Running over time](#running-over-time) is the run that takes these steps. Gradients with
respect to a tank's area (the storage's capacity `C`) reach the levels through the move of
each step.

**Volume curves.** A tank with a volume curve is a `TankVolumeCurve`: its volume is EPANET's
`tankvolume`, linear between the curve's points and the end values beyond them (`interp`,
`project.c`), and its step is by volume, $V \mathrel{+}= q\,\Delta t$, $H = $ `tankgrade`$(V)$
(`hydraul.c`), through `NodeStorage.potential_of`, the inverse of the stored amount, which
`KirchhoffLayer.advance_storage` reads for a storage that is not linear. The control crossings
and the time to fill are found by volume too, as EPANET's `controltimestep` and
`tanktimestep` find them. Gradients reach the curve's volumes through the steps (against
central differences: 1.4e-9 relative). `validate()` refuses a curve that does not cover the
tank's minimum and maximum levels, as EPANET does (`input1.c`, error 225).

A curve may have flat segments (two points with the same volume), as EPANET allows: such a
segment has no area, and the level jumps across it. EPANET's `tankgrade` gives the flat volume
the segment's lower level and any volume above it a level above its upper one, and
`potential_of` is the same transcription. A tank whose maximum (minimum) level lies on a flat
stretch has that stretch's volume as its `Vmax` (`Vmin`) and, filled (drained) to it, is put
at `tankgrade(Vmax)`, the stretch's lower end, as `tanklevels` puts it. Every tank tests its
level controls by volume, as EPANET's `controls` (`hydraul.c`) does,
$V(y) \ge V(y_c) - |q|$ (ABOVE) and $V(y) \le V(y_c) + |q|$ (BELOW) with $q$ one second of
the previous solve's inflow, so on a flat stretch every control level on it is reached at once.
The times to a control or a limit use the inflow, not an area.

**Full and empty tanks.** A tank within EPANET's `HTOL` (0.0005 ft) of its maximum level is
full and one within `HTOL` of its minimum is empty (`hydstatus.c`, `tankstatus`). At the step
start `TankLevels` then gives every pipe to it the one direction that drains a full tank or
fills an empty one (`"water.one_way"`, the check-valve condition, solved exactly), closes a
pump discharging into a full tank or drawing from an empty one, and closes a pipe both
directions are then forbidden (a check valve into a full tank); all of it for that step
only, in `"water.status"` and `"water.one_way"`, not in the carried state. As in EPANET a
link whose start node is a tank or reservoir is examined at its start node only, so a pump
from a reservoir into a full tank keeps running (EPANET then puts the tank back at its maximum
each step, and the water is lost). A valve at a full or empty tank is refused by name.
`water_run` ends a step when a tank fills or empties (`TankLevels.time_to_limit`, rounded
to whole seconds as `tanktimestep` rounds it) and puts a tank within one second's inflow of a
limit on it (`tanklevels`). A tank with `overflow=True` is never full: its inlet stays open,
its level is held at the maximum, and `WaterResults.tank_overflow` is the flow it spills over
the step from each report time. Volume curves and overflow are refused with the implicit schemes below,
which also refuse a level outside the limits by name, since a tank moved inside the solve has
no step start to close links at.

**Tanks stepped inside the solve.** `build_model(net, tank_scheme="backward_euler")` (or
`"trapezoidal"`) makes every tank an interior node instead, its level stepped
implicitly inside the solve: backward Euler, $q = A\,(H - H_{\text{prev}})/\Delta t$, or
trapezoidal. `water_step(model, state, drivers, dt)` takes one step (`Model.step`, with
`water_steady`'s Newton defaults); `tank_levels(model, state)` reads the levels from the solved
heads; the level-triggered controls stay a closure (`TankStorageControls`), acting on the
step-start levels. The step is differentiable through time (the gradient of a level after
several steps matches central differences). It is **not** the EPANET comparison mode: the
implicit step differs from EPANET's explicit one by the step's own discretisation error. Both
converge to the same trajectory, at first order: on Net1, halving the storage step halves its
distance from a fine explicit run. Its step is the caller's and is never shortened (`water_run`
refuses these schemes), so with steps as long as an hour a switch can land up to a step late,
and sub-steps of a few minutes are the useful range. Not with `quality`: the combination is refused with a
`ValueError` (a storage tank is an interior node with node storage, which
the quality layer does not model as a well-mixed volume); use the default `tank_scheme` for
water quality, where a tank is a quality boundary whose concentration the caller sets.
A constant tracer is preserved (every junction stays at the source concentration) across both
`pipe_form`s, TCV, PBV and GPV valves, tanks and `pda`, tested in `tests/apps/water`.

### Rules

`[RULES]` are modelled as EPANET 2.2's `rules.c` has them (`noodl.apps.water.rules`):
`IF`, `AND` and `OR` premises, `THEN` and `ELSE` actions, `PRIORITY`. A premise reads a node's
`DEMAND`, `HEAD` (`GRADE`), `LEVEL` or `PRESSURE`, a link's `FLOW` (its absolute value),
`STATUS` (`OPEN`, `CLOSED`, `ACTIVE`: a valve as its last solve found it, `link_states`) or
`SETTING`, or `SYSTEM` `TIME`, `CLOCKTIME` and `DEMAND`. A numerical premise holds within
EPANET's tolerance of a thousandth of a file unit (`checkvalue`); a `TIME` or `CLOCKTIME` premise
tests the interval since the previous check (`checktime`: `=` when its time lies in it, across
midnight too, an inequality at its end). An action is a link's `STATUS` (`OPEN` or `CLOSED`) or
`SETTING`, with the same meaning as a control's (a number on a valve makes it `ACTIVE`, on a
pump it is the speed, `OPEN` runs a pump at speed 1); as in `takeactions`, `OPEN` acts on a
closed link only, `CLOSED` on one that is not, and a setting within EPANET's 1e-3 (feet, cubic
feet per second, a relative speed) of the present one is no change. Among the actions on one
link the rule of the highest `PRIORITY` stands, a tie goes to the rule listed first. A valve
setting is converted like a control's, to the layer's potential unit (`head_scale` on
Darcy-Weisbach).

**EPANET's premise logic is not the textbook's.** Its manual says `AND` binds tighter than `OR`,
but `evalpremises` runs left to right and a false result meeting an `AND` ends the rule false:
`IF a AND b OR c` with `a` false is false whatever `c` is, and `IF a OR b AND c` is
`(a OR b) AND c`. The rule is transcribed as it is, and its truth tables are the ones EPANET 2.2
gives (`tests/apps/water/test_rules.py`, measured on `wntr`'s EPANET).

`apply_rules(model, state, drivers, *, time, step)` tests the rules on a solved `state`
and returns `(state, changed)`: the state with the carried `"water.link_status"` and
`"water.link_setting"` after the actions, and per instance whether an action was taken
(EPANET's action count, which asks for a new solve: a pump the solver shut at shut-off is
reopened by `STATUS IS OPEN` at every check while it stays shut). `step`, the interval since the previous check, is required and positive: EPANET
never checks the rules over an empty interval, and a `TIME` or `CLOCKTIME` premise tests that
interval. The actions are tested against the link states of the solve (`link_states`, EPANET's
`LinkStatus`): `OPEN` reopens a PRV the solver closed and a pump it shut at shut-off (at speed
1), `CLOSED` does nothing to a pump already at shut-off. A `SETTING` premise on a valve whose
status is fixed `OPEN` or `CLOSED` is never satisfied (EPANET's missing setting), and an FCV
the flow runs back through reads `OPEN` (`XFCV`; its hysteresis needs the last step's state,
which is not carried). The order within a time point is EPANET's: the rules are checked at the
end of a step, at its time T, and the next solve at T applies the simple controls first, so a
control that fires at T stands over a rule's action on the same link. `water_steady` has no
time and applies no rule; `water_run` calls `apply_rules` at each rule time step inside a
hydraulic step (`WaterNetwork.rule_timestep`, a tenth of the hydraulic step unless
`RULE TIMESTEP` says otherwise), never at t = 0, and a rule that acts ends the step. A batched state is tested instance by instance. `SYSTEM DEMAND` is the sum of the
junctions' positive demands; EPANET clamps each demand category of a junction, which are
added up here.

A rule action `STATUS IS ACTIVE` on a valve is accepted and does what EPANET's `takeactions`
does with it: nothing. Refused by name: `STATUS IS ACTIVE` on a pipe or a pump, a `SETTING`
premise on a pipe or a GPV, an action on a check valve (EPANET error 207), a negative setting or a setting on a GPV
(error 202), a `DEMAND` premise on a reservoir, and the `DAY` and `POWER` premises EPANET does
not read; a `DEMAND` premise on a pressure-driven or emitter network (the delivered demand is
not read); a rule's or control's action on a pipe when the pipes are in the
`pipe_form="potential"` form, which reads no status (refused at `build_model`). A premise built
in Python has EPANET's tolerance of a thousandth of the unit of an SI file (1e-6 m3/s, 1e-3 m)
unless it is given; the reader sets it from the file's units.

### Pressure-driven demand

EPANET's Wagner function (Manual eq. 13.6), with $p = H - z$:

$$
d(p) = \begin{cases}
D & p \ge P_{\text{req}} \\
D \left(\dfrac{p - P_{\min}}{P_{\text{req}} - P_{\min}}\right)^{e} & P_{\min} < p < P_{\text{req}} \\
0 & p \le P_{\min}
\end{cases}
$$

Implemented with a guarded kink so both `where` branches stay finite and differentiable.
`PressureDrivenDemand` refuses `p_req <= p_min` — a zero-span demand curve.

### Emitters

An emitter is a flow that leaves a junction as a power of its pressure head,
$q = \operatorname{sign}(p)\, C |p|^{n}$ with $p = H - z$ and $n$ the `EMITTER EXPONENT` (default
0.5), `EmitterOutflow` (a potential-dependent `NodeWithdrawal`). EPANET models it as a fictitious
pipe to a reservoir at the junction's elevation (`hydcoeffs.c`, `emittercoeffs`,
`emitterheadloss`); that loss is odd in the flow and not clipped, so **at negative pressure an
emitter is an inflow** (verified against EPANET). Below `dp_transition` (1e-9 m) the law is its
tangent line, which keeps the slope finite at $p = 0$. The coefficient is differentiable
(`learnable`), and matches central differences to 4.8e-11 relative. Emitters add no work to a
network without them.

## Verification

Against the real EPANET 2.2 engine through `wntr` 1.5.0, which bundles `epanet22.dll`, driven
through its toolkit on the very file noodl reads (not on `EpanetSimulator`'s rewrite of it) and at
a true `ACCURACY 1e-7`, set with `EN_setoption`: EPANET's reader clamps an `ACCURACY` in the file
to at least 1e-5. A test that keeps 1e-5 says why (a network with no demand at all, whose
relative stopping test runs against EPANET's closed-link leaks alone; a group whose head is
the last thing EPANET's iteration settles; an active PBV, at 1e-6). A run on which EPANET does
not converge is refused. Note the reference implementation's own floor: EPANET's binary output
has **float32** reals on disk, so roughly 1e-7 relative is EPANET's own precision, not noodl
physics'.

| Check | Tolerance | Measured |
|---|---|---|
| `twoloop_si.inp` heads and flows | rel 1e-6 | **5.3e-8** heads, **8.090e-8** flows |
| `twoloop_si.inp`, noodl physics' own nodal continuity | 1e-13 | **2.093e-14** |
| Net1 single period: heads, flows, pump head gain, pump flow | 1e-6 / 1e-6 / 1e-6 / 1e-8 m³/s | 7.3e-8, 1.1e-7, 1.0e-7, 5.4e-9 m³/s |
| Net1 24 h tank level with level-triggered pump controls (`water_run`) | 2e-4 m | **3.9e-5 m** worst of 25 steps (26 hydraulic steps) |
| Net1 24 h tank level with `AT TIME`, `AT CLOCKTIME` (START CLOCKTIME 8 am) and junction-pressure pump controls added | 1e-4 m | 3.8e-5 m (time and clock time; 1.8e-4 m before EPANET's whole-second crossings and volume margin), 3.5e-5 m (pressure) |
| Net1 24 h with REPORT START and TIMESTEP 2 h; PATTERN START 1 h; a pump speed pattern (with a 0); a reservoir head pattern: report times, heads, levels, flows, states, pump setting | 1e-5 / 2e-4 m / 1e-6 m³/s / equal | heads 9.0e-8, 9.6e-8, 7.7e-8, 9.7e-8; levels 2.0e-5, 2.5e-5, 2.1e-5, 2.9e-5 m; flows 5.2e-8 to 5.9e-8 m³/s |
| Four rules (ELSE, PRIORITY, CLOCKTIME, a PRESSURE premise) on a tank, a pump and a PRV, 24 h, RULE TIMESTEP 5 min, against EPANET's toolkit; PRIORITY flipped too | 2e-4 m / 1e-5 / 1e-6 m³/s / equal | 3.2e-5 m, 6.9e-7, 4.0e-8 m³/s; pump states and PRV settings equal at every hour |
| Net1 with REPORT START 1:30 (not a multiple of the report step), against EPANET's toolkit with `runqual`'s rule applied: report times, levels, heads, flows | equal / 2e-4 m / 1e-5 / 1e-6 m³/s | equal, 1.1e-5 m, 3.9e-8, 5.3e-8 m³/s |
| A rule acting in one instance of a batch: the shared step ends for both, each instance its forced single run | 1e-9 m / equal | exact to 1e-9 m, states equal |
| A batch of two Net1 instances (0.8 and 1.25 times the demand) against each run alone on the batch's step boundaries; against the plain single runs | 1e-9 m / 1e-2 m | exact to 1e-9 m; 6.5e-3 m (the extra boundaries' explicit-step difference, see Limitations) |
| SPECIFIC GRAVITY 1.2 on an LPS file with pressure-driven demand and an emitter, 2 h of a pattern: heads, flows, demand plus emitter outflow | 1e-6 / 1e-6 m³/s / 1e-6 m³/s | 1.3e-7, 1.9e-8, 1.0e-8 |
| Pump energy and cost (`WaterResults.energy`): Net3 over 7 days (two control-switched pumps; a global price on a price pattern, a pump price, a demand charge), Anytown 24 h (efficiency curves, a pump at speed 1.2 for the speed adjustment, shut-off), Net1 with a duration of 1:30 (the last step counted whole) and of 0, against the energy section of EPANET's binary output: on-time, efficiency, kWh per volume, average and peak kW, cost per day, demand charge | 1e-5 relative | 5.0e-8 (Net3), 2.6e-6 (Anytown), 3.8e-8 and 2.3e-8 (Net1) |
| Net1 24 h, tank on a five-point volume curve: levels, pipe 110 flow | 2e-4 m / 1e-6 m³/s | 2.8e-5 m, 5.5e-8 m³/s |
| Net1 24 h, tank on a volume curve with a flat segment: below the range, crossed by the level (110-120 ft), the tail from 110 ft: levels, pipe 110 flow | 2e-4 m / 1e-6 m³/s | 2.8e-5, 2.7e-5, 2.4e-5 m; 5.5e-8, 5.6e-8, 5.1e-8 m³/s |
| Net1 24 h without controls: the tank fills, its inlet closes; levels, pipe 110 and pump flows | 2e-4 m / 1e-6 m³/s | 2.6e-5 m, 3.7e-8 m³/s |
| Net1 24 h at 1.5 times the demand: the tank empties and refills | 2e-4 m / 1e-6 m³/s | 2.4e-5 m, 3.2e-8 m³/s |
| A pump into a tank that starts full (closed, then filling it to full many times an hour), 12 h | 2e-4 m / 1e-6 m³/s | 5.3e-6 m, 1.4e-9 m³/s |
| Net1 24 h, the tank allowed to overflow: held at its maximum with its inlet open | 2e-4 m / 1e-6 m³/s | 2.6e-5 m, 9.8e-9 m³/s |
| Darcy-Weisbach, `twoloop_si.inp` heads and flows | rel 1e-6 | 5.819e-8 heads, 6.482e-8 flows |
| Darcy-Weisbach, Net1 single period: heads, flows | 1e-6 / 1e-6 | 7.1e-8, 9.6e-8 |
| Darcy-Weisbach, laminar + transitional + turbulent pipes (`lowflow_dw_si.inp`): heads, flows, per-pipe head loss | 1e-6 / 1e-6 / 2.65e-7 m | 3.5e-8, 9.4e-8, 9.6e-8 m |
| Emitters at J4 and J6 vs EPANET, the file's demands and coefficients (heads, emitter outflows, flows) | 1e-6, 1e-6 m3/s, 1e-6 m3/s | 8.6e-8, 7.3e-10, 4.0e-9 |
| An emitter at negative pressure (J4 at 60 m, an inflow) vs EPANET | 1e-6, 1e-6 m3/s, 1e-6 m3/s | 7.0e-8, 7.3e-10, 2.8e-9 |
| `HEADLOSS C-M` (n = 0.011) vs EPANET | 1e-6, 1e-6 m3/s | 7.6e-8 heads, 1.8e-9 m3/s flows |
| A check valve on the `HEADLOSS C-M` two-loop (P4 closes, P5 open, P7 closed): heads, flows, every link's state | 1e-6 / 1e-6 m³/s / equal | 7.6e-8, 2.2e-9 m³/s, all equal |
| Check valve and closed pipe on the two-loop: heads, flows, every link's state | 1e-6 / 1e-6 m³/s / equal | 6.6e-8, 2.2e-9 m³/s, all equal |
| A junction without demand cut off by closed links (held at the average of its far ends, flows into it zero to round-off) | 1e-6 / 1e-6 m³/s | 7.5e-8, 2.2e-9 m³/s |
| Every pipe a check valve at five times the demand (two close) | 2e-6 (EPANET's closed-link leak, see Limitations) / 1e-6 m³/s / equal | 1.4e-6, 5.7e-8 m³/s |
| A junction fenced off by two check valves (held at 75 m, the `1 / CBIG` limit) | 1e-6 | 3e-9, both valves CLOSED |
| TCV on the two-loop: heads, flows, the valve's head loss | 1e-6 / 1e-6 m³/s / 1e-5 m | 5.3e-8, 1.8e-9 m³/s, 3.2e-6 m |
| Emitters behind a closed pipe; Darcy-Weisbach emitters (heads, emitter outflows) | 1e-6 relative to the 50 m reservoir (one case 1e-3 m: EPANET's closed-link leak) / 1e-6 m³/s | 5.4e-8, 3.1e-9 m³/s (Darcy-Weisbach); the leak case 6.9e-4 m |
| Emitter coefficient gradient vs central differences | 1e-8 | 4.8e-11 |
| Pressure-driven demand vs EPANET's `DEMAND MODEL PDA` | 1e-5 | 5.5e-8 heads, 8.4e-8 demands |
| A pump feeding pressure-driven demand (two-loop with a pump, every junction pressure-limited), both `pipe_form`s | 1e-5 | 6.3e-8 heads, 7.9e-8 flows, 9.8e-8 demands |
| PBV (active and open) and GPV on the two-loop: heads, flows, the valve's head loss (EPANET at 1e-6 for the active PBV) | 1e-6 / 1e-6 m³/s / 1e-5 | 6.7e-8, 6.4e-8 m³/s, 2.0e-6 |
| PRV and PSV, the eleven cases below: heads, flows, states, controlled head | 1e-6 / 1e-6 m³/s / equal / 1e-6 m | 9.2e-8, 3.7e-8 m³/s, all equal, 0.0 m |
| PRV inside EPANET's `HTOL` band (EPANET `ACTIVE`, noodl `OPEN`): heads | `HTOL` = 1.524e-4 m | below `HTOL` at three points in the band |
| Pumps on the two-loop (heads, flows, states): Net3's three-point curve ($n = 1.77$); Anytown's five-point curve on its first and second segments; Net6's constant-power pump | 1e-6 / 1e-6 m³/s / equal | 6.4e-8, 1.8e-9; 7.1e-8 and 5.7e-8, 7.2e-9; 5.4e-8, 1.8e-9 |
| A pump shut beyond its shut-off head; a zero-demand group behind a pump (shut and running at zero flow); a pump and a check valve shutting together around a group; a segmented curve starting above zero flow, shut at its first point's head | 1e-6 / 1e-6 m³/s / equal | 8.5e-8, 8.1e-8 m³/s (1.8e-8 at the shut pump, EPANET's leak); all states equal |
| Pump speed 0.8 through `"water.setting"` against `SPEED 0.8` in the file, every curve kind | 1e-6 / 1e-6 m³/s / equal | 3.3e-8 to 6.6e-8, 1.8e-9 |
| ky10 at time zero (13 constant-power pumps, PRVs, tanks), EPANET's demands; EPANET at `DAMPLIMIT 1e-2` (Limitations, two builds) | 1e-6 / 1e-6 m³/s / equal | 1.2e-7, 5.4e-8 |
| ky4 at time zero (two constant-power pumps, one closed, an empty tank, 959 junctions), demand pattern at t = 0 | 1e-6 / 1e-6 m³/s / equal | 8.2e-8, 9.1e-8 (two parallel-pipe pairs: 2.6e-7) |
| Anytown at time zero, pumps running, tanks at mid-level, the five placeholder pipes feeding junctions 5-7 at 12 in (three parallel pumps on the five-point curve; pipe 125 still 0.0001 in) | 1e-6 / 1e-6 m³/s / equal | 9.3e-8, 3.1e-8 |
| Whole networks through `water_run` for their full DURATION (`test_water_networks.py`): heads (relative to the larger of the head and 1 m), tank levels, flows, states at every report time, and the energy section | 1e-5 / 2e-4 m / 1e-6 m³/s / equal / 1e-5, except where noted | the next five rows, in that order |
| Net3, 168 h (AT TIME and level controls, a closed pipe, three-point pumps); 6 s | as above | 1.2e-7, 5.2e-6 m, 1.3e-7 m³/s, equal, 4.2e-8 |
| Net6, 96 h (3323 junctions, 61 pumps, PRVs, a check valve, 124 level controls); 120 s | heads 2e-5, levels 3e-3 m, flows 5e-4 m³/s, energy 3e-4 (EPANET's closed-link leak, see Limitations: the heads at 90 h follow a tank level 1.5e-3 m apart) | 1.1e-5, 1.94e-3 m, 3.1e-4 m³/s, equal, 1.95e-4 |
| ky4, DURATION 0 | as above | 8.2e-8, 6.0e-6 m, 9.1e-8 m³/s (pairs 2.6e-7), equal, 1.1e-8 |
| ky10, DURATION 0 (level controls on `~@`-named pumps); EPANET at `DAMPLIMIT 1e-2` | as above | 1.2e-7, 1.3e-5 m, 5.5e-8 m³/s, equal, 9.0e-7 |
| Anytown edited (`_anytown`), 24 h at a 1 min step; 11 s | heads 1e-4 (EPANET's closed-link leak out of the empty tanks, at a head of 0.55 m; see Limitations) | 5.1e-5 (1.8e-4 m; 9.3e-8 at t = 0), 1.8e-6 m, 2.9e-7 m³/s, equal, 3.7e-7 |
| Net1 6 h, pump 9 CLOSED at 2 h and OPENed at 4 h by time controls; closed at 2 h by a rule, given a CLOSED control at 2:30, then OPENed: the hydraulic steps, heads, level, flows, states, the pump's speed, the energy | equal / 1e-5 / 2e-4 m / 1e-6 m³/s / equal / 1e-5 | steps equal (no step at 2:30), 7.3e-8, 2.0e-5 m, 5.1e-8 m³/s, speed 0 then 1, energy within 1e-5 |
| The four rules' network with a price and a demand charge (both priorities), its pump on a curve or at constant power: the pump's energy section against EPANET's toolkit | 1e-5 relative | 3.9e-7 (curve), 1.6e-7 (power) |
| The four rules' network with its pump at shut-off (curve `25 20` under a tank head of 41 m) and one rule opening it from 2 h, `STATUS IS OPEN` or `SETTING IS 0.9`, with a price: energy section (EPANET counts the shut pump on line from 2 h; the reported state stays CLOSED) | 1e-5 relative | 7.4e-7 (open), 9.9e-7 (setting) |
| Net1 with pump 9 into tank 2, 24 h: energy section | 1e-5 relative | 5.4e-8 |
| Net1 1:30 with a rule closing the pump at 1:45 (past the duration, inside EPANET's whole last step): energy section | 1e-5 relative | 3.8e-8 |
| Every network `wntr` ships (37 files, the two malformed ones aside) builds, or is refused by name | as listed | 36 build; `bad_times.inp` refused ("is not a time of the form HH:MM[:SS]") |
| Head loss sums to zero around every cycle-basis loop | 1e-12 | **exactly 0.0** |
| Autodiff vs central differences | 1e-6 × scale | 3.212e-6 / 2.505e-10 / 9.2e-10 / 2.753e-9 |
| d(junction heads)/d(demands) through a closed check valve, vs central differences | 1e-6 × scale | 2.7e-11 × scale (2.6e-8 at scale 983) |
| d(junction heads)/d(pump speed), every curve kind, vs central differences | 1e-6 relative | 3.1e-10 or better |
| steady tracer, single source | 1e-3 | 2.8e-14 percentage points |
| Golden regression | 1e-10 | **0.0** |

A few rows deserve comment.

**Continuity** is machine-precision exact for noodl physics (2.093e-14), while EPANET itself misses
continuity at Net1 node 13 by 4.5e-10 m³/s. The ~1e-7 residuals on `twoloop_si.inp` and the
Net1 single period are attributable to EPANET's float32 output path and its own mild continuity
violation, not to this solver.

**Darcy-Weisbach reaches the same floor as Hazen-Williams.** With EPANET's own composite law
(above) the D-W rows sit at the float32 output floor. The per-pipe loss bound is derived, not
fitted: EPANET stores its heads as float32 feet in its hydraulics file and reports them as
float32 metres, so each reported head can be off by half a spacing in each, and a loss by twice
that, $2(0.3048 \cdot 2^{-22} + 2^{-24})$ m for heads between 1 and 2 m. Two EPANET details
matter at this level: the rounded flow-unit factor (with the exact one the two-loop heads miss
by 4.3e-7 instead of 5.8e-8) and, for the low-flow network, an accuracy beyond EPANET's
default 1e-3, at which its own loop flows stop 5e-7 short of converged.

**The valve row compares flows absolutely.** EPANET holds an active PBV's head drop with a
penalty ($10^8$, `CBIG`), so its own flows are not continuous through the valve: P2 and the
valve are in series through a demand-free junction, yet EPANET reports them 5.9e-8 m³/s apart,
and the drop across the valve as 5.000002 m for a 5 m setting. noodl physics' flows agree to
round-off and its drop is the setting. The penalty also limits EPANET's own convergence: at
`ACCURACY 1e-7` its relative flow change stalls at 1.9e-6 and it reports the system
unbalanced, so the active PBV is compared with EPANET at 1e-6.

**PRV and PSV.** The eleven cases of `docs/plans/water-prv-psv-design.md` (section 4),
`test_prv_psv_match_epanet`. The two-loop cases add junction J7 (P2 ends there) and the valve
V1 from J7 to J2 (250 mm); the series cases have pipes of 1000 m, 200 mm, C = 100 and valves
of 200 mm, with a demand of 20 L/s at the end of the PRV line. Heads are compared with the
file's demands (the pipes carry EPANET's rounded LPS factor; with the exact one the worst was
2.4e-6, on the C = 100 lines), and EPANET at `ACCURACY 1e-7`: at the two-loop fixture's
0.001 it stops with an active PRV's flow 5.8e-6 m³/s short of its node balance, and its
heads 5.8e-6 relative from converged (MEASURED).

| Case | Network | EPANET state | noodl state | Heads (rel) | Flows (m³/s) |
|---|---|---|---|---|---|
| PRV active | two-loop, PRV 48 m | ACTIVE, $H_{J2}$ = 48 | ACTIVE, $H_{J2}$ = 48 exactly | 4.2e-8 | 1.8e-9 |
| PRV open | two-loop, PRV 49.9 m | OPEN | OPEN | 5.3e-8 | 1.8e-9 |
| PRV open with minor loss | R1 40 m, PRV 50 m, $K$ = 10 | OPEN (0.206 m drop) | OPEN | 6.7e-8 | 1.4e-9 |
| PRV closed, reverse flow | two-loop, V1 from J2 to J7, 49.9 m | CLOSED | CLOSED | 9.2e-8 | 3.6e-9 |
| PRV closed, downstream above setting | two-loop, PRV 45 m | CLOSED, $H_{J2}$ = 45.298 | CLOSED | 9.2e-8 | 3.8e-9 |
| PSV active | R1 100 m, PSV 70 m, R2 20 m | ACTIVE, $H_{J1}$ = 70 | ACTIVE, $H_{J1}$ = 70 exactly | 0 | 1.6e-9 |
| PSV open | the same, 30 m | OPEN | OPEN | 3.2e-10 | 5.4e-9 |
| PSV closed, upstream below setting | R1 60 m, 70 m | CLOSED | CLOSED | 0 | 3.7e-8 |
| PSV closed, reverse | R1 60 m, 30 m, R2 80 m | CLOSED | CLOSED | 0 | 1.9e-8 |
| PRV and PSV in series | R1 100, PRV 60, J3 10 L/s, PSV 55, R2 10 | both ACTIVE | both ACTIVE, 60 and 55 exactly | 3.2e-8 | 5.8e-10 |
| Near the switch | PRV 50 m, R1 53.80 to 53.85 m (switch 53.8214) | OPEN below, ACTIVE above | the same | 5.3e-8 | 1.4e-9 |

Also compared: a PRV or PSV feeding a dead end without demand (EPANET's states and heads), a
PRV directly downstream of a reservoir (against EPANET with a 1 mm pipe between them, which
EPANET needs), the PRV-and-PSV line with Darcy-Weisbach (1.3e-8), and gradients through an
active PRV and PSV (demands, roughness, setting) against central differences
(`max|analytic - fd| <= 1e-6 max|fd|`), with $dH_{J2}/ds = 1$ exactly downstream of an
active PRV, 0 through an open one, and the jump from 1 to 0 across the switch.

**The loop head-loss row is a formulation identity, not a comparison** — head loss summing to zero around every
independent loop is a property of the cycle-space formulation, and it holds exactly.

### Water quality

The reference is EPANET 2.2 through the toolkit on the committed file, with the same settings as
above (`ACCURACY 1e-7` through `EN_setoption`, `TRIALS 400`, 5000 and `DAMPLIMIT 1e-2` for ky10; any warning fails the
test unless it is named). Quality is read at every hydraulic step (the time of each `EN_runQ`; `EN_nextQ`
advances the quality to the next one, the loop `EN_solveQ` runs; `EN_stepQ` splits a step at a hydraulic event, which EPANET's own
runs do not). The bound, over every node and link at every hydraulic step time: each point within 1e-5 of
the quality's scale (1 for a chemical, 100 for a trace, the largest age of the run for an
age: 24 h on the base network, 6.92 h on ky10) and the worst within `Ctol`; mass-balance terms within 2e-5 relative and the ratio within
1e-6. A point beyond 1e-5 is admitted only with a measured cause asserted in the test. EPANET reads its hydraulics back as
float32 reals, a floor of about 6e-8 relative.

Per feature, on a small base network (`quality_base.inp`: a reservoir, a tank with a pump line,
a dead end), each case compared at 25 hourly times; the 54 cases also have their mass balance
compared:

| Feature | Measured (worst, of the quality's scale, away from the admitted points) |
|---|---|
| Base, CHEMICAL, TRACE at a junction, a reservoir and a tank; a dead end below `Q_STAGNANT`; a tank at its limit; a check valve; a closed link; a flow cycle | 1.3e-7 to 4.4e-7 |
| Bulk order 0, 1, 2, Michaelis-Menten, limiting potential (growth and decay) | 4.9e-8 to 2.2e-7 |
| Wall order 0 (SI and US), order 1, mass-transfer limited, `DIFFUSIVITY 0`, roughness correlation (Hazen-Williams; Darcy-Weisbach in US units) | 5.7e-8 to 2.8e-7; 2.3e-6 for Darcy-Weisbach (the hydraulics of that file, 1e-6 class) |
| Tank reaction; water age; reactive dead end and check valve (`noflowqual`) | 2.0e-7 to 2.2e-7 |
| Sources: each type, with and without a pattern, at a reservoir, a junction (positive and negative demand), a tank; SETPOINT below the node's quality; a reservoir source whose pattern reaches 0 | 1.7e-5 to 2.5e-5 at the admitted dead-end points, none beyond elsewhere |
| Mixing: MIXED, 2COMP, FIFO, LIFO; 2COMP crossing its mixing-zone volume; FIFO and LIFO emptying through several segments; 2COMP with no net flow | 3.3e-8 to 4.2e-7 |
| Mass balance, all 54 cases | terms 3.8e-6 (2e-5 bound), ratio 8.3e-8 (1e-6) |
| A duration past the last whole hydraulic step (5:30; the last step not routed) | 1.98e-6; terms 2.9e-7, ratio 2.6e-8 |
| A batch against its single runs (bulk batch bit-equal; demands 0.8 and 1.2 over 23 h) | equal; 1e-8 of scale |
| Gradients of a node's quality with respect to bulk, wall, source, initial quality (a junction and a tank) and a pipe roughness, against central differences (h = 1e-4 of the value; the +h, -h and base runs have identical segment counts and node order) | relative difference 8.5e-12 to 1.1e-8 (bound 1e-5) |

The admitted points on the base network are all at its dead-end pipe P7 (1.2e-5 to 6.3e-5 of the
scale): EPANET's solver leaves a residual flow of at most 3.7e-9 m³/s in it, and it pushes the
junction's water into the pipe while noodl's P7 carries 1e-25 m³/s. Each point is checked
against a bound computed from EPANET's cumulative flow through P7, and admitted only while that flow stays
below 1e-8 m³/s.

EPANET's example networks (`test_water_quality_networks.py`):

| Network | Compared | Worst | Notes |
|---|---|---|---|
| Net1 (chlorine, 24 h) | all nodes and links, 27 times | 1.6e-6 mg/L (p99.9 1.4e-6); mass terms 7.7e-7, ratio 1.8e-8 | |
| Net2 (fluoride, a `CONCEN` source on a pattern, 55 h, one tank), hydraulics | 56 steps | heads 1.1e-7 relative, levels 8.9e-6 m, flows 1.9e-8 m³/s | states and step times equal |
| Net2, `Tolerance` 0.01 moved by 1e-8 either way | all nodes and links | 4.2e-6 mg/L (p99.9 6.6e-7) | |
| Net2 as shipped (`Tolerance` 0.01) | | 2.83e-3 mg/L, within `Ctol`; 667 of 4256 points beyond 1e-5, all from 32 h | a tie at exactly `Ctol`, below |
| Net3 (TRACE from the lake, 168 h, 1226 flow reversals, 154 node sorts) | 183 times | 2.5e-4 % (p99.9 9.7e-5 %); mass terms 1.4e-6, ratio 1.3e-8 | |
| Net6 as shipped (quality identically 0) | smoke | exactly 0 in both; ratio exactly 1 | |
| Net6 with a source added, before 18922 s | 40 steps | 5.9e-6 mg/L; mass terms (whole run) 1.9e-6 | hydraulic cause below |
| Net6 with a source added, noodl's router on EPANET's saved hydraulics | all 609 times | 5.2e-11 | |
| ky4 (TRACE, DURATION edited to 24 h) | 65782 points, 31 times | 9.35e-3 % (p99.9 5.5e-4 %); terms 1.8e-6, ratio 9.0e-7; router on EPANET's hydraulics 2.0e-9 % | 33 points admitted, below |
| ky10 (AGE, DURATION edited to 6:54:59) | 35928 points, 18 times | 3.53e-3 h (p99.9 1.5e-5 h); terms 3.2e-6, ratio 6.1e-9; router on EPANET's hydraulics 1.4e-12 h | 12 points admitted (10 beyond on Windows, 11 on Linux), below |

The causes admitted, each asserted in its test:

- **Net2 as shipped: a decision at an exact tie.** The pattern steps from 0.09 to 0.08 mg/L at 31 h,
  and pipe 1's last segment then meets the new outflow exactly `Ctol` = 0.01 apart. noodl computes
  `0.09 - 0.08 = 0.009999999999999995`, below `Ctol`, and merges; EPANET, in its internal unit, gets a
  difference one ulp above `Ctol` and starts a new segment. The test checks the decision (the
  segment counts, and that EPANET at 0.01 equals EPANET at 0.00999999) and does not count it as accuracy.
  noodl's own run flips those 667 points; its router fed EPANET's saved hydraulics in mg/L (`_router_on_epanets_hydraulics`) flips 791. Accuracy is tested with `Tolerance` moved by 1e-8.
  A tie at 49 h does resolve as EPANET's, since the decisions are made in EPANET's unit.
- **Net6 with a source: closed-link leaks.** The step plans part at 18922 s (EPANET's PUMP-3845
  switch; noodl's is at 18921 s, see Limitations, closed links), after which 252 of
  EPANET's 609 step times are 1 to 55 s from noodl's and the fronts arrive at shifted times (a 1 s
  shift of one late switch moves EPANET's own quality by 2e-2). The test asserts the bounds up to 18922 s and
  the mass balance over the whole run; after it the comparison is the router on EPANET's
  hydraulics, 5.2e-11.
- **ky4: closed-link leaks and EPANET's own convergence.** 30 of the 33 points are leaks of closed links, which EPANET
  conducts at `1/CBIG` and noodl does not: the closed pump's 9e-8 m³/s back to the reservoir through
  P-977 (below `Q_STAGNANT`, so EPANET routes it from its first node and the reservoir's 100 % enters the pipe) and the
  closed tank pipes' 4e-9 to 2e-8 m³/s. With those leaks injected into noodl as sources, 3 points
  remain (J-703 and P-696 at 83882 and 86400 s) against EPANET at `ACCURACY 1e-7`, and none against
  EPANET at 1e-8: EPANET's own convergence moves its flow by 3e-4 relative there. With wntr's
  Linux build two of the three are beyond at 1e-7 and J-703 still at 1e-8 (2.4e-3 %, from
  4.3e-3 %): every one shrinks with EPANET's accuracy, on both builds.
  The mass-balance ratio differs by 9.0e-7: the router on EPANET's saved hydraulics reproduces EPANET's
  ratio to 1e-15, and 89 % of the deficit is the pump's leak routed into dead junctions, whose
  mass counts as neither outflow nor storage.
- **ky10: EPANET's unconverged continuity, and closed-link leaks.** 10 points are P-310, the only link into four
  dead-end junctions: noodl's flow equals their demand to 1e-12, EPANET's is up to 0.5 % off
  (a 1.3e-9 to 5.0e-8 m³/s residual at its accuracy, `DAMPLIMIT 1e-2`), and 1e-8 is unbalanced at the start even
  with 5000 trials; nine of them are beyond the bound with wntr's Windows build, all ten with
  its Linux build. Two points (J-433 at 23050 s; J-145 at 14400 s, Windows only) are closed-link leaks.
  The run stops at 6:54:59 because the hydraulics diverge after it (Limitations).

## Limitations

- **Not modelled, refused by name:** a valve at a full or empty tank. A file using it raises an
  error naming the valve.
- **Rules.** `FILLTIME` and `DRAINTIME` premises and the `DAY` and `POWER` ones are refused. A
  pump closed by a rule is at speed 0 in `"water.link_setting"`, as EPANET (`setlinkstatus`)
  and as a closing control leaves it; a `SETTING` premise on it reads 0, as EPANET.
- **Controls.** `validate()` refuses a control on a reservoir (`IF NODE <reservoir>`): EPANET
  2.2 compares tank volumes and a reservoir has area 0, so such a control always holds. A pump
  closed by a control is at speed 0 in `"water.link_setting"`, as in EPANET, so a `CLOSED`
  control reaching a pump closed by `[STATUS]` (which keeps its speed 1) ends a step at its
  trigger as EPANET's does (Net6, PUMP-3881 at 1050 s); a pump closed by `[STATUS]` or by the
  solve keeps its speed (read `"water.status"` too). A control that sets a pump's
  built speed leaves the `"water.setting"` driver in force (a calibration driver at 0.9 stays
  0.9 after an `OPEN`, EPANET gives 1.0). `water_step` and the implicit schemes never shorten
  a step to a time control, which fires only if a step starts exactly on its second
  (`water_run` refuses them).
- **Closed links carry no flow; EPANET's carry `dh / CBIG`.** EPANET gives a closed link the
  conductance `1 / CBIG` (`CBIG = 1e8` ft per cfs), so a head drop across it leaks: on Net6 at
  3600 s, 2.56e-7 m³/s through three closed pumps from a dead-ended suction (noodl's head drop
  over CBIG gives the same 2.5592e-7 m³/s). Integrated over hours that is 1.7e-6 m an hour in
  a tank's level, enough to move a level trigger that falls near a half second by one second
  (EPANET rounds the time to it); over Net6's 96 h the 609 steps all match, up to a minute
  apart late in the run, and the tank levels differ by up to 1.95e-3 m. On Anytown the empty
  tanks' closed pipes leak 1.4e-7 to 2.7e-7 m³/s out of them, EPANET's pumps deliver about as
  much less, and the heads far from the pumps differ by up to 1.8e-4 m (1.3e-7 relative with
  those leaks added as demands); on the two-loop with every pipe a check valve at five times
  the demand the two closed ones leak 5.7e-8 and 3.1e-8 m³/s (heads 1.4e-6 relative; 7.6e-8
  with the leaks added as demands). The leak is not modelled in the hydraulics; the energy
  reproduces it where EPANET's energy reads it (a pump other than a constant-power one that a
  rule opens at the end of a step, see the pump energy above).
- **A zero-demand group of several junctions joined by a link inside it, fenced off by
  check valves**, is held at the `1 / CBIG` limit head (75 m between reservoirs at 100 and
  50 m). EPANET's head there is not that limit. MEASURED, for J - P5 - L fenced off between
  those reservoirs: 73.2097 m with P5 5 m long, 75.0213 m at 500 m, 74.9940 m at 5000 m
  (not monotonic in the length, so not a converged limit); the 2.3e-8 m³/s EPANET reports in
  P5 is the `1 / CBIG` flow through the closed check valves (25 m over `CBIG`). The group's
  head is the last thing EPANET's iteration settles, its flows all below the relative-flow
  test: at `ACCURACY 1e-7` EPANET stops one iteration later than at 1e-5, on 74.9122 m, and
  driven on (FLOWCHANGE 1e-10, a relative change of 8e-17) it returns to 75.0213 m, so these
  cases are compared with EPANET at 1e-5, its fixed point. It is not the limit because
  EPANET resolves this head only to steps of 0.0546 m: eliminating J leaves L the pivot
  `(P5 + 1/CBIG) - P5^2 / (P5 + 1/CBIG)` (2e-8 exactly, computed from P5's conductance of
  8.1e4), and one ulp of 8.1e4 is 7.3e-4 of it; every head EPANET gave the group, over
  accuracies, options and both of wntr's builds, is one of 74.9122, 74.9667, 75.0213 and 75.0759 m
  (75.0759 m on Linux at 1e-5). Tests: `test_a_group_fenced_off_by_check_valves_is_held_at_the_limit_head`
  (two steps, 1.5e-3 relative; MEASURED 2.8e-4 Windows, 1.0e-3 Linux) and the internal-check-valve case of
  `test_check_valves_fencing_a_junction_match_epanet` (4.1e-5: J1 = J2 = 56.6667 m, the
  limit, against EPANET's 56.6690; MEASURED, EPANET's head there moves with the internal
  pipe's length as well, 55.3140 m at 5 m, 56.6690 at 500 m, 56.6655 at 5000 m, and with
  the accuracy, 56.6827 m at 1e-7). A single fenced-off junction, with no link inside the group, matches
  EPANET to 3e-9, and a junction fed from the highest of several reservoirs through one
  check valve, to 8.5e-8 (EPANET's float32 head).
- **PRV and PSV inside EPANET's `HTOL` band.** Within 1.524e-4 m below a PRV's switch EPANET
  keeps it `ACTIVE` with a negative drop (its status rules have hysteresis); noodl gives the
  ideal valve, `OPEN`. The heads differ there by less than `HTOL`
  (`test_prv_inside_htol_band`). EPANET's `XPRESSURE` (an active valve that cannot deliver its
  setting) has no counterpart: such a valve is `OPEN` in the exact condition.
- **A group behind a closed link, fed only through it, has EPANET's head only in the limit.**
  A PRV (or any valve) `CLOSED` by `[STATUS]` in front of junctions without demand holds them
  at the `1 / CBIG` limit, the head on the other side (60 m); EPANET reports 59.930 m for the
  same network at `ACCURACY 1e-7` and 1e-8 (53.687 m at 1e-5, where it stops after two
  iterations), with a PRV or a TCV alike
  (`test_a_closed_prv_cutting_off_a_junction_is_held_at_the_limit_head`, 1.2e-3 relative).
  A PRV or PSV that closes itself in front of a dead end (no demand anywhere) is held at the
  limit head too. With the dead end on the valve's start, EPANET converged (`ACCURACY` 1e-7
  and 1e-8 alike) holds it at exactly that head, 40 m for a PRV under 40 m and 60 m for a PSV
  under 60 m, but reports the valve OPEN where noodl reports it CLOSED: at zero flow between
  equal heads both states satisfy every link, so the state is not determined there (at 1e-5
  EPANET stops early, at 39.997 and 53.687 m, CLOSED). On three other dead-end networks
  EPANET does not converge at 1e-7 at all (its relative stopping test runs against its
  closed-link leaks alone) and on one, a PRV under 60 m with the dead end on its start, its
  head moves with its accuracy (60.0006, 64.42, 61.36 and 59.93 m at 1e-5, 1e-6, 1e-7 and
  1e-8); those are compared with EPANET at 1e-5, its early stop (35.791 m behind a PSV whose
  upstream head of 40 m is below its setting), and so are the zero-demand groups behind a
  pump below (`test_a_valve_feeding_a_dead_end_without_demand_matches_epanet`). These are wntr's
  Windows build; its Linux build marks the valve `XPRESSURE` on four of the cases (a PSV with
  the dead end on its end, a PRV on its start) and holds the dead end OPEN at the limit head,
  noodl's head (below, two builds).
- **A zero-demand group between a pump and a check valve to a higher reservoir has no
  unique head.** Every head from R1 plus the pump's shut-off head up to R2 satisfies every
  link with no flow anywhere. EPANET's value is set by the ratio of its closed-link leaks
  and its status path: with R1 at 50 m and R2 at 100 m it shuts the pump and reports
  74.48 m. noodl returns the lower end, the pump running at zero flow at its shut-off head
  (63.3334 m), which EPANET also returns when a CLOSED pipe replaces the check valve. With
  any demand, or R2 below that shut-off head, the two agree
  (`test_a_pump_and_a_check_valve_shutting_together_match_epanet`).
- **A zero-demand group between a stopped pump and a check valve has no unique head
  either.** With the pump CLOSED or at speed 0 every head up to R2 satisfies every link;
  noodl gives the `1 / CBIG` limit with the check valve shut (the average of R1 and R2:
  75 m for 50 and 100 m), the same for a CLOSED pump and one at speed 0, and EPANET keeps
  the check valve OPEN from its first iterate (its `cvstatus` holds a status inside `HTOL`)
  and reports R2's head (`test_a_stopped_pump_and_a_check_valve_fencing_a_group`).
- **The Colebrook option is not EPANET below Re = 4000.** With `friction="colebrook"` a laminar
  D-W pipe's head loss follows a straight line to the Colebrook value at Re = 2000, not
  Hagen-Poiseuille, and Dunlop's transition cubic is not used. The default `friction="composite"`
  reproduces both.
- **The steady tracer is verified on a single source only.** The check uses a
  network with one source, where the answer is 100 % everywhere it reaches; mixing of several
  traced sources has not been compared against EPANET.
- **Tank area does not enter a single `water_steady` call.** Only the tank's head enters the
  steady solve, so gradients with respect to tank area come from a time-stepped run
  (`water_run`), not from the steady state.
- **An empty tank that was a part's only supply** leaves that part cut off: the pipe to it
  then refuses to drain it, and the solve is refused by name before it starts ("junction(s)
  ... have a demand that no reservoir, tank or inflow reaches"), where EPANET reports negative
  pressures (wntr's Anytown as shipped: pumps at speed 0, tanks at their minimum).
- **A volume that decreases along the curve is refused.** EPANET 2.2 accepts it (wntr 1.5.0
  solves Net1 with `(110, 80000)` after `(60, 90000)`), but a volume that falls as the level
  rises is not physical; `validate()` and `TankVolumeCurve` refuse it by name. Flat segments
  are accepted.
- **Links of extreme resistance that must carry a demand are not solved.** Anytown's 0.0001 in
  placeholder pipes alone feed three junctions; EPANET reports heads of -9.9e24 m there, and
  noodl's Newton cannot bring a row whose two terms are 9.9e24 m below its tolerance (one
  rounding step there is 2^30 m): it raises the "singular" error of a numerically singular
  Jacobian. The Anytown comparison gives those pipes 12 in.
- **A batch shares one step.** `water_run` ends the step of every instance at the earliest
  event of any (spec: "Batches"), so a batched instance is its own single run only up to the
  extra step boundaries, each an extra solve of EPANET's explicit step (6.5e-3 m on Net1 at
  0.8 and 1.25 times the demand; exact to 1e-9 m against its single run forced onto the same
  boundaries). Run instances one at a time to reproduce EPANET's own sequence for each.
- **The run ends at the duration.** `water_run` solves at every time EPANET's whole steps reach
  within `DURATION`, past the last report time too (a tank or control event there; the quality
  is kept and routed there as EPANET's is). EPANET's last step is not cut at `DURATION` (it
  solves once more past it, and reports nothing there); `water_run` ends the run at the
  duration without that solve.
  A REPORT START that is not a multiple of the REPORT TIMESTEP follows EPANET's rule (the first
  solve at or after each report time, `runqual`). wntr 1.5.0 cannot read that output file, so
  it is compared through EPANET's toolkit with `runqual`'s rule applied to EPANET's solves
  (`test_an_unaligned_report_start_follows_epanets_output_rule`), not against the file.
- **PATTERN START is transcribed with EPANET's own step rule.** EPANET looks for the next
  pattern period at `n * PATTERN TIMESTEP - t` (`timestep`), which ignores PATTERN START: a
  step that starts between two of the shifted period boundaries runs to the unshifted one,
  and takes the new period's multipliers on only there. `water_run` does the same
  (`test_pattern_start_shifts_the_pattern_as_epanet_does`).
- **One demand pattern per junction.** Several `[DEMANDS]` categories of one junction must
  share their pattern (refused by name otherwise).
- **Water quality: not modelled.** EPANET-MSX (multi-species), EPANET 2.3 features, an Eulerian
  discretisation, the per-link average reaction-rate output and the `Wbulk`, `Wwall`, `Wtank` and
  `Wsource` summary of EPANET's report (cosmetics). A reaction order between 0 and 1, `ORDER WALL`
  other than 0 or 1, and a Darcy-Weisbach roughness not below the diameter under a roughness
  correlation are refused by name ([Water quality](#water-quality)).
- **Water quality inherits the hydraulics' closed-link leaks.** EPANET's closed links conduct
  `1/CBIG` and noodl's do not (above), so a quality front that crosses a closed link's leak, or a
  switch that falls a second apart (Net6's PUMP-3845), is carried differently: ky4's 33 admitted
  points, ky10's J-433 and Net6 after 18922 s. They are measured and asserted in the tests; the leak
  is not modelled.
- **ky10's quality is verified to 6:54:59 only.** After it EPANET's `pumpcoeff` caps the slope of
  the constant-power pump `~@Pump-10` (`CBIG`), which dead-heads against its closed CV and
  takes it to zero flow, where noodl keeps it on its power curve (5.4e-4 m³/s, 1396 m): the
  hydraulics, and so the quality, differ from 7 h. Forcing the pump closed in noodl matches EPANET to
  10 h. The 24 h comparison is an expected failure in `test_water_networks.py` with these values;
  EPANET itself needs `TRIALS 5000` at `ACCURACY 1e-7` on this file, and `DAMPLIMIT 1e-2` (below).
  Left for a later change to the pump physics.
- **wntr's two builds of EPANET differ in their last bits, and where EPANET's answer rests on
  those bits the reference is measured on both.** wntr 1.5.0 ships `libepanet22` built with MSVC
  (Windows) and with GCC 7.4 (Linux), both version 2.2 with the same toolkit. MEASURED where they
  part: (1) ky10 at time zero, at EPANET's default `DAMPLIMIT 0`, the Linux build stops on a
  second branch that is no solution, `~@RV-4` CLOSED and the constant-power pump `~@Pump-11` at
  2.1e-22 m³/s with 7.9 m of head (its law asks 7e21 m: below `sqrt(r / CBIG)` the `CBIG` slope cap
  makes it a closed link, as for `~@Pump-10` above), from which its 24 h is another run; at
  `DAMPLIMIT 1e-2` (PRV and PSV status checks deferred to a relative flow change of 1e-2, the
  flow updates damped there) both builds take Windows' branch, and noodl's, at time zero to
  1.2e-7, and the same 50 steps over 24 h, so every ky10 reference runs at 1e-2. (2) A fenced
  zero-demand group's head (above) is resolved only to steps of 0.0546 m, one ulp of its link's
  conductance in EPANET's pivot: 75.0213 m on Windows, 75.0759 m on Linux. (3) A PRV or PSV
  feeding a dead end on the side it does not fix: the dead end's Cholesky pivot is zero up to
  its last bit, and where the Linux build's is not positive `badvalve` marks the valve
  `XPRESSURE` (warning 5, "open but cannot deliver pressure"): the valve OPEN and the dead end at
  the limit head, on four of the eight dead-end cases (above). (4) The age and trace points that
  follow EPANET's unconverged flows (ky4's J-703 and P-696, ky10's P-310) differ in which pass the
  bound. Each test names its build's values.
- **EPANET's own accuracy bounds the comparison.** A file's `ACCURACY` is clamped to 1e-5 by
  EPANET's reader; the reference sets 1e-7 through the toolkit. At 1e-8 EPANET leaves Net2 and Net3
  hydraulically unbalanced at several steps, so 1e-7 is the tightest reference that converges
  everywhere. noodl's quality on EPANET's own saved hydraulics matches EPANET to 5.2e-11 (Net6)
  and 1.4e-12 h (ky10), so the differences above are in the hydraulics' inputs, not the router.
- **A tie at exactly `Ctol` cannot always be reproduced.** The decision is made in EPANET's
  internal unit, but the last bit of the compared values depends on EPANET's merge history in
  float32-read volumes and ft³; Net2 as shipped (31 h) is decided otherwise than EPANET's (above).

## Install

Nothing beyond the base dependencies, plus optionally `sparse`. `wntr` — the EPANET reference
implementation — is test-only and deliberately **not** a runtime extra, because it pulls in a
measured 351 MB of mandatory dependencies.

```bash
pip install "noodl-physics[dev]"     # only if you want to run the EPANET parity tests
```
