# EPANET `.inp`

Reading an EPANET input file produces a model for [water
distribution](../applications/water.md). It shares its lexical tokenizer
(`noodl.apps.inpfile`) with the SWMM reader; each format's own section names, column layouts and
units are read by `apps/water/inp.py`.

**Read:** `[JUNCTIONS]`, `[RESERVOIRS]`, `[TANKS]` (with a volume curve and the overflow flag),
`[PIPES]` (with their `OPEN`, `CLOSED` or `CV` status), `[PUMPS]` (`HEAD`, `POWER`, `SPEED` and
`PATTERN`), `[VALVES]` (every type, PRV and PSV included), `[DEMANDS]`, `[PATTERNS]`, `[CURVES]`,
`[CONTROLS]` (every form: `AT TIME`, `AT CLOCKTIME`, `IF NODE`), `[RULES]`, `[STATUS]`,
`[EMITTERS]`, `[ENERGY]` (efficiency, price, price pattern and demand charge: the pumps' energy and cost,
`WaterResults.energy`; no head or flow depends on it), `[OPTIONS]` (including `EMITTER EXPONENT` and
`HEADLOSS C-M`) and
`[TIMES]` (including `RULE TIMESTEP`, `PATTERN START`, `REPORT START` and `START CLOCKTIME`, which
reads `12 am`, `6:30 PM` and `18`).

**Times honoured by `water_run`** ([Running over time](../applications/water.md#running-over-time)):
`DURATION` (the run's end), `HYDRAULIC TIMESTEP` (no longer than the pattern or report step, as
EPANET caps it), `PATTERN TIMESTEP` and `PATTERN START` (the pattern period
`(t + PATTERN START) / PATTERN TIMESTEP`), `REPORT TIMESTEP` and `REPORT START` (the report
times), `RULE TIMESTEP` (a tenth of the hydraulic step unless given) and `START CLOCKTIME` (for
`AT CLOCKTIME` controls and `CLOCKTIME` premises). `QUALITY TIMESTEP` and `STATISTIC` are read
past and change nothing (`net.notes["unrecognised_times"]`).

**Ignored** because nothing in the hydraulics references them: `[TITLE]`, `[REPORT]`,
`[COORDINATES]`, `[VERTICES]`, `[LABELS]`, `[BACKDROP]`, `[TAGS]`, `[END]`, and the
quality sections (a quality run is configured through `build_model(quality=...)` instead).

**Aggregated / interpreted:** several `[DEMANDS]` categories of one junction add up (they replace
the `[JUNCTIONS]` demand); categories with different patterns are refused (a category with no pattern uses the default pattern, so it equals one naming that default explicitly), since one demand pattern
per junction is modelled. `[TIMES]` values take an optional unit (any token beginning SEC, MIN, HOU or DAY,
case-insensitive, as EPANET reads it: MINS, SECS, HOU; a bare number is hours, `HH:MM[:SS]` is a clock time); other units are refused.

**Curves by role.** `[CURVES]` has no type of its own, so each curve is converted by the role
that uses it: a pump head or GPV head-loss curve (flow, length), a pump efficiency curve (flow,
percent, kept as written) or a tank volume curve (length, volume). A curve used in two roles
with different units is refused by name when the file is read.

**Pressures.** A pressure in the file (`PRESSURE` premises, PRV/PSV/PBV settings, `MINIMUM` and
`REQUIRED PRESSURE`, emitter coefficients) is converted as EPANET's `Ucf[PRESSURE]` does,
including `SPECIFIC GRAVITY` (psi per foot of head times it; metres of head times it in an SI file).

**Rules.** `[RULES]` are modelled (`apply_rules`): premises on node and link attributes and
`SYSTEM` `TIME`, `CLOCKTIME` and `DEMAND`, `THEN`/`ELSE` actions and `PRIORITY`, with EPANET's own
premise logic (not the textbook precedence of `AND` over `OR`). `GRADE` reads as `HEAD`; a
`STATUS IS ACTIVE` action on a pipe or a pump, an action on a `CV` pipe, a `DAY` or `POWER` premise, and anything
after a premise's value (but a time unit) are refused by name when the file is read or
validated.

**Read, then refused by name.** The reader converts every feature above to SI, and
`WaterNetwork.validate()` (called by `build_model`) refuses, naming the feature, each one the
builder does not model yet, so a file never loads into a network that quietly means something
else: a numeric `[STATUS]` pump speed and a numeric `[CONTROLS]` setting on a pipe. A pump
speed pattern is applied by `water_run` (0 closes the pump). Every other `[CONTROLS]` form is
modelled: `AT TIME`, `AT CLOCKTIME` and `IF NODE` on a tank or a junction (a reservoir is refused: EPANET tests its volume of area 0, so
the control always holds), with an
`OPEN`, `CLOSED` or numeric action on a pipe, pump or valve. Of EPANET's own valve checks
(`valvecheck`), `validate()` refuses a PRV or PSV whose controlled node (a PRV's end, a PSV's
start) is a reservoir or tank, and two valves that hold one node twice (EPANET's errors 219 and
220), naming the valves. Of its tank checks (`validatetanks`, error 225) it refuses an initial level outside the minimum and maximum, and a volume curve that does not cover both. A tank volume curve and overflow are built (`TankVolumeCurve`, `WaterResults.tank_overflow`) under the default `tank_scheme` and refused by name under the implicit ones. Of its pump checks it refuses a curve EPANET refuses (error 227): a
one- or three-point curve with no power-function fit (`powercurve`), and any other curve whose
flows do not rise or heads do not fall from point to point; and a constant-power pump without a
positive power. The reader itself refuses what it cannot read: a malformed record or a name
that is not in the network.

**Pumps** are modelled as EPANET 2.2 types them (`input2.c`, `updatepumpparams`): a curve of one
point, or of three with the first at zero flow, is a power function `h = h0 - r q^n` (`n` fitted
to the three points; one point is fitted with EPANET's added points `(0, 1.33334 H)` and
`(2 q, 0)`); every other curve is joined by straight segments;
`POWER` is a constant-power pump (its power is read in kW: horsepower in a US file, kilowatts in
an SI one, which EPANET divides by its 0.7457 kW per hp). `SPEED` is the pump's initial relative
speed, the driver `"water.setting"`. See
[Water distribution: pumps](../applications/water.md#pumps).

**Units** follow `[OPTIONS] UNITS`: `CFS GPM MGD IMGD AFD` are US, `LPS LPM MLD CMH CMD` are SI,
and everything is converted to SI on read. D-W roughness is millimetres (SI) or millifeet (US).
A `VISCOSITY` above 1e-3 is a multiplier on water's viscosity, anything else the actual kinematic
viscosity (m²/s or ft²/s), as in EPANET. A `HEADLOSS D-W` file is modelled with EPANET's own
composite friction law by default (see [Darcy-Weisbach](../applications/water.md#darcy-weisbach)). Unrecognised `[OPTIONS]` lines are recorded verbatim in
`notes["unrecognised_options"]` rather than dropped.

```python
from noodl.apps.water import read_epanet_inp, build_model, water_steady

net = read_epanet_inp("twoloop_si.inp")
model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)

# final["water.phi"] is head (m) at every node
# final["water.q"] is flow (m3/s) in every link
```
