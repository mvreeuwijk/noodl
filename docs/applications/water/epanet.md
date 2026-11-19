# Importing EPANET files

Reading an EPANET input file produces a model for [water distribution](index.md). The reader shares its lexical tokenizer (`noodl.apps.inpfile`) with the SWMM reader; each format's own section names, column layouts and units are read by `noodl.apps.water.inp`.

## How to load

```python
from noodl.apps.water import read_epanet_inp, build_model, water_steady

net = read_epanet_inp("twoloop_si.inp")
model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)
```

`final["water.phi"]` is the head (m) at every node and `final["water.q"]` the flow (m³/s) in every link. For a run over time use `water_run` ([Extended-period simulation](extended-period.md)). Everything is converted to SI on read.

## What is read

### Network sections

`[JUNCTIONS]`, `[RESERVOIRS]`, `[TANKS]` (with a volume curve and the overflow flag), `[PIPES]` (with their `OPEN`, `CLOSED` or `CV` status), `[PUMPS]` (`HEAD`, `POWER`, `SPEED` and `PATTERN`), `[VALVES]` (every type, PRV and PSV included), `[DEMANDS]`, `[PATTERNS]`, `[CURVES]`, `[EMITTERS]`, `[STATUS]`.

`[CONTROLS]` is read in every form (`AT TIME`, `AT CLOCKTIME`, `IF NODE`) and `[RULES]` in full ([Tanks, controls and rules](tanks-controls.md)). `[ENERGY]` (efficiency, price, price pattern and demand charge) gives the pumps' energy and cost, `WaterResults.energy`; no head or flow depends on it.

**Demands.** Several `[DEMANDS]` categories of one junction add up (they replace the `[JUNCTIONS]` demand). A category with no pattern uses the default pattern, so it equals one naming that default explicitly. Categories with different patterns are refused, since one demand pattern per junction is modelled.

**Status.** A numeric `[STATUS]` on a pump is its initial speed (0 closes it), and a pump speed pattern is applied by `water_run` (0 closes the pump).

### Options and units

`[OPTIONS]` is read including `EMITTER EXPONENT` and `HEADLOSS C-M`. Unrecognised `[OPTIONS]` lines are recorded verbatim in `notes["unrecognised_options"]` rather than dropped.

Units follow `[OPTIONS] UNITS`: `CFS GPM MGD IMGD AFD` are US, `LPS LPM MLD CMH CMD` are SI, and everything is converted to SI on read. D-W roughness is millimetres (SI) or millifeet (US). A `VISCOSITY` above 1e-3 is a multiplier on water's viscosity, anything else the actual kinematic viscosity (m²/s or ft²/s), as in EPANET. A `HEADLOSS D-W` file is modelled with EPANET's own composite friction law by default ([Darcy-Weisbach](physics.md#darcy-weisbach)).

**Pressures.** A pressure in the file (`PRESSURE` premises, PRV/PSV/PBV settings, `MINIMUM` and `REQUIRED PRESSURE`, emitter coefficients) is converted as EPANET's `Ucf[PRESSURE]` does, including `SPECIFIC GRAVITY` (psi per foot of head times it; metres of head times it in an SI file).

**Curves by role.** `[CURVES]` has no type of its own, so each curve is converted by the role that uses it: a pump head or GPV head-loss curve (flow, length), a pump efficiency curve (flow, percent, kept as written) or a tank volume curve (length, volume). A curve used in two roles with different units is refused by name when the file is read.

**Pumps** are typed as EPANET 2.2 types them (`input2.c`, `updatepumpparams`): a curve of one point, or of three with the first at zero flow, is a power function `h = h0 - r q^n` (`n` fitted to the three points; one point is fitted with EPANET's added points `(0, 1.33334 H)` and `(2 q, 0)`); every other curve is joined by straight segments. `POWER` is a constant-power pump (its power is read in kW: horsepower in a US file, kilowatts in an SI one, which EPANET divides by its 0.7457 kW per hp). `SPEED` is the pump's initial relative speed, the driver `"water.setting"`. See [Pumps](physics.md#pumps).

### Times

`[TIMES]` is read including `RULE TIMESTEP`, `PATTERN START`, `REPORT START` and `START CLOCKTIME` (which reads `12 am`, `6:30 PM` and `18`). `water_run` honours ([Extended-period simulation](extended-period.md#running-over-time)):

- `DURATION`: the run's end.
- `HYDRAULIC TIMESTEP`: no longer than the pattern or report step, as EPANET caps it.
- `PATTERN TIMESTEP` and `PATTERN START`: the pattern period `(t + PATTERN START) / PATTERN TIMESTEP`.
- `REPORT TIMESTEP` and `REPORT START`: the report times.
- `RULE TIMESTEP`: a tenth of the hydraulic step unless given.
- `START CLOCKTIME`: for `AT CLOCKTIME` controls and `CLOCKTIME` premises.
- `STATISTIC` is read past and changes nothing (`net.notes["unrecognised_times"]`); `QUALITY TIMESTEP` goes to `WaterNetwork.quality` (below).

`[TIMES]` values take an optional unit (any token beginning SEC, MIN, HOU or DAY, case-insensitive, as EPANET reads it: MINS, SECS, HOU; a bare number is hours, `HH:MM[:SS]` is a clock time); other units are refused.

### Water quality

Water quality is read, in full, into `WaterNetwork.quality` (a `WaterQuality`; always set by the reader, mode `NONE` when the file has none) in SI: `[QUALITY]`, `[SOURCES]`, `[REACTIONS]`, `[MIXING]`, `[OPTIONS] QUALITY` (`NONE`, `CHEMICAL`, `AGE`, `TRACE <node>`, and EPANET's "any other word [units]", a chemical so named: `Chlorine mg/L`), `DIFFUSIVITY`, `TOLERANCE`, and `[TIMES] QUALITY TIMESTEP`. `water_run` then runs the quality by default ([Water quality](quality.md)). `WaterQuality.node_order` and `link_order` carry EPANET's own node and link indices.

**Keywords.** They are recognised as EPANET's `match` does, on the strings of its `text.h`: a case-insensitive prefix test of the keyword against the token, so `QUAL AGE`, `DIFF 2`, `TOLER 0.5`, `GLOB BULK -1`, `LIMIT POTENTIAL 0.5`, `ROUG CORRELATION 0.5` and `chem` read as `QUALITY AGE`, `DIFFUSIVITY`, `TOLERANCE`, `GLOBAL`, `LIMITING` and `ROUGHNESS` do, and any `[TIMES]` line whose first token begins `QUAL` is the quality step (the second word is not tested); a later line overrides an earlier.

**Chemical name and units.** They are overridden field by field, as EPANET does: `NONE` and `CHEMICAL` change neither (so `QUALITY CHEMICAL ug/L` keeps `Chemical` in `mg/L`, and `Chlorine ug/L` followed by `QUALITY CHEMICAL` keeps `Chlorine ug/L`); another word sets the name and, only if a units token follows, the units; `AGE` and `TRACE` set both.

**Ranges and special forms.** The range forms `id1 id2 value` of `[QUALITY]` and of `[REACTIONS] BULK|WALL|TANK` are numeric when the C `atol` of both ids is positive (`10x` reads as 10) and in `strcmp` order otherwise. A legacy 3-token `[SOURCES]` line is a `CONCEN` source and `*` is no pattern. A `[MIXING]` fraction of 0 means 1.

**Unit conversion.**

- Bulk and tank coefficients: per day to per second.
- A wall coefficient of order 1: length per day (feet in a US file) to m/s; of order 0: mass per area per day (ft² to m²) to per second, whatever the order line's position. The roughness correlation likewise.
- `DIFFUSIVITY`: a value above 1e-4 multiplies water's own; anything else is the value in m²/s (ft²/s in a US file).
- A `MASS` source: mass per minute to per second.
- `QUALITY TIMESTEP 0` is EPANET's default (a tenth of the hydraulic step, `None`); its value is the line's last token, or its last two when the last is a unit, whatever words lie between (`timedata`: `QUAL FOO 10` is 10 hours). EPANET's `hour` would also take an `AM`/`PM` after the value, which the reader refuses: meaningless for a time step.

## What is ignored

Sections nothing in the hydraulics references: `[TITLE]`, `[REPORT]`, `[COORDINATES]`, `[VERTICES]`, `[LABELS]`, `[BACKDROP]`, `[TAGS]` and `[END]`.

Skipped as EPANET does, and recorded in `net.notes["ignored_quality"]`: a `[MIXING]` line on a junction (whatever its model, which EPANET does not read) or a reservoir, a `TANK` coefficient on a junction, a `BULK` or `WALL` coefficient on a pump or valve, and a line too short to mean anything.

## What is refused

The reader converts every feature above to SI, and `WaterNetwork.validate()` (called by `build_model`) refuses, naming the feature, each one the builder does not model, so a file never loads into a network that quietly means something else.

### By the reader

A malformed record or a name that is not in the network. A `TIMES` unit other than seconds, minutes, hours or days. A curve used in two roles with different units. Two `[DEMANDS]` categories of one junction with different patterns.

In `[RULES]`: a `STATUS IS ACTIVE` action on a pipe or a pump, an action on a `CV` pipe, a `DAY` or `POWER` premise, and anything after a premise's value (but a time unit) are refused by name when the file is read or validated. `GRADE` reads as `HEAD`.

### By `validate()`

- A numeric `[CONTROLS]` setting on a pipe. A control on a reservoir (`IF NODE <reservoir>`): EPANET tests its volume of area 0, so the control always holds.
- Valves, from EPANET's `valvecheck`: a PRV or PSV whose controlled node (a PRV's end, a PSV's start) is a reservoir or tank, and two valves that hold one node twice (EPANET's errors 219 and 220), naming the valves.
- Tanks, from EPANET's `validatetanks` (error 225): an initial level outside the minimum and maximum, and a volume curve that does not cover both. A tank volume curve and overflow are built (`TankVolumeCurve`, `WaterResults.tank_overflow`) under the default `tank_scheme` and refused by name under the implicit ones.
- Pumps: a curve EPANET refuses (error 227), that is a one- or three-point curve with no power-function fit (`powercurve`), and any other curve whose flows do not rise or heads do not fall from point to point; and a constant-power pump without a positive power.

### Water quality

`water_run` refuses by name what EPANET 2.2's own `bulkrate` cannot answer: a bulk or tank reaction order between 0 and 1 (zero concentration gives NaN or infinity), and a Darcy-Weisbach roughness not below the diameter under `ROUGHNESS CORRELATION`.

The reader refuses `ORDER WALL` other than 0 or 1, `GLOBAL TANK` (EPANET error 213), `QUALITY TRACE` without a node (201) or with an unknown one (212), a negative initial quality (209), an unknown mixing model on a tank (213) or source type, a negative `TOLERANCE` or `DIFFUSIVITY`, an unknown reaction keyword, and a node, link or pattern that is not in the network (EPANET skips or refuses these; the reader refuses).

The full list of limitations is on [Limitations](limitations.md).
