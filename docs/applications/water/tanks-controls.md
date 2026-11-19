# Water distribution: tanks, controls and rules

How tanks move, how `[CONTROLS]` and `[RULES]` act on links, and what the water distribution application refuses. The run that takes these steps is [Extended-period simulation](extended-period.md); the overview is [Water distribution](index.md).

## Tanks and controls

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
the pump switches a step late (measured: 0.76 m off on Net1 with time controls added). The
previous solve's inflow is the driver `"water.tank_inflow"`, which `water_steady` takes
from the state it starts from (its `"water.storage_flow"`: the previous solve's at a step
start, zero at `initial_state` as at EPANET's t = 0) when the drivers carry none.
`link_states` reads it the same way, so for a solved state it tests the controls against that
solve's own inflow; `water_run` reads its statuses with the drivers the solve took.

[Extended-period simulation](extended-period.md) is the run that takes these steps. Gradients with
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
`pipe_form`s, TCV, PBV and GPV valves, tanks and `pda`.

## Rules

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
gives (measured on `wntr`'s EPANET).

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

