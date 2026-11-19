# Verification: building physics

Evidence for the [building physics application](../../applications/building/index.md); the summary is on the [verification overview](index.md). Three groups of evidence: the ContamX engine run live, OpenModelica simulations of Modelica Buildings Library models, and closed-form solutions.

## Against ContamX

The comparison is `tests/verification/test_contam_parity.py`. The reference is NIST's ContamX 3.4.1.7, driven through `contamxpy` (the `contam` extra, see [Installation](../../get-started/installation.md); Windows
x86-64 only, as the wheel bundles the engine). `noodl.apps.building_physics.contamx` provides `run_steady`
and `run_transient`, which run the engine on a scratch copy of the project with the ambient
conditions set through the API, and `run_project`, which runs a project as written: its own run
control, schedules, and weather and ambient-concentration files. None of them mutates your
fixture directory; `airflow_convergence=(rcnvg, acnvg)` tightens ContamX's airflow tolerances in
the copy. `simulate_project(project)` is the noodl physics side of `run_project`.

ContamX holds project input data in single precision (the fan rated 0.200683 kg/s comes back as
float32(0.200683) exactly), so the reference is only as precise as a float32 roundoff,
$u = 2^{-24} \approx 6\times10^{-8}$, in each input. Every tolerance below is a first-order budget
of such roundoffs through the compared quantity, not a fit to the measured error; ContamX's
airflow iteration is run at `(1e-10, 1e-12)` (`(1e-13, 1e-16)` for the whole-project runs) and
noodl physics' Newton at 1e-14 kg/s so that neither solver contributes. The whole-project rows also
allow an absolute floor of 64 $u$ times the largest flow (or mass fraction), for the near-zero
values of calm and isothermal hours; their measured column gives the relative error of every entry
above 0.1 % of the largest.

| Check | Tolerance | Measured |
|---|---|---|
| Stack project, flow directions | exact signs | match |
| Stack project, flow magnitudes, over a ±20 K ambient sweep | rel 17.2–32.8 $u$ (1.02e-6 – 1.96e-6) | **7.2e-8 – 9.4e-8** |
| Stack residual without / with the zone-pressure density | 3e-5 – 6e-5 flat / < 1 % of that | 4.4e-5 / 9.4e-8 |
| Constant-mass-flow fan delivers its rating (0.200683 kg/s) | engine rel $u$, noodl physics rel 1e-12 | holds |
| Three-zone project, steady flows | rel 16 $u$ (9.5e-7) | **9.1e-8** |
| Three-zone project, transient concentrations, 24 steps at 300 s | rel 64 $u$ (3.8e-6) | **2.0e-7** |
| Three-zone project, constant wind pressure 5 Pa on path 1 | rel 16 $u$ (9.5e-7) | **9.9e-8** |
| One-zone project with its own `.wth` and `.ctm`, 288 steps | flow rel 33 $u$, mf rel 64 $u$, floor 64 $u$ × max | **1.3e-8** flow, **1.1e-7** mf |
| One-zone stack with a scheduled zone temperature (`densZP`, `dodMdt`), 288 steps | rel 64 $u$, floor 64 $u$ × max | **4.5e-8** flow, **1.9e-7** mf |
| Scheduled stack project over two days, across midnight | rel 64 $u$, floor 64 $u$ × max | **7.8e-8** flow, **1.9e-7** mf |
| Scheduled three-zone project (fan, source, zone temperature, window, weather, ambient file), 24 h | rel 64 $u$, floor 64 $u$ × max | **1.6e-8 kg/s** flow (the floor governs: 2 % of 64 $u$ × max; 3.6e-5 relative on path 4 at its flow reversal), **1.4e-7** mf |
| Three-zone project with kinetic reactions in two zones, 24 steps | flow rel 16 $u$, mf rel 80 $u$, floor 64 $u$ × each species' max | **9.1e-8** flow, **2.6e-7** mf |
| One-zone stack with a filtered path, steady, Ta = 273.15 / 313.15 K | rel stack budget + 3 $u$ (20.2 / 21.2 $u$) | **6.6e-8 / 5.6e-8** |
| Three-zone project with constant-efficiency, particle and series filters, 24 steps | flow rel 16 $u$, mf rel 72 $u$, floor 64 $u$ × each species' max | **9.1e-8** flow, **3.7e-7** mf |
| Three-zone project with UVGI filters (one with a lamp-age model on a path whose flow a schedule cuts to 0.4 for 45 minutes, one in series with a constant-efficiency filter, one crossed against its path's direction), 24 steps | flow rel 16 $u$, floor 64 $u$ × max; mf: the implicit step's first-order error bound, derived on the run with each step's UVGI pass fractions and their float32 errors (at most 7.4e-6 relative) | **8.0e-8** flow (1.4 % of the allowance), **5.2e-7** mf (12 %) |
| Three-zone project with deposition sinks, removal terms and NRCC emissions, 24 steps | flow rel 16 $u$, mf rel 80 $u$, floor 64 $u$ × each species' max | **9.1e-8** flow, **1.3e-7** mf |
| One-zone stack with a pressure-driven source, 288 steps | flow rel 64 $u$, mf rel 88 $u$, floor 64 $u$ × each species' max | **4.5e-8** flow, **7.4e-8** mf |
| Three-zone project with a fan curve (mult 1.5) on a speed schedule 1, 0.8, off (its shut-off orifice) and 1.2, 24 steps | flow rel 32 $u$, mf rel 64 $u$, floor 64 $u$ × each species' max | **1.6e-7** flow, **5.8e-8** mf |
| Three-zone project with a duct network (wind terminal, filtered Darcy-Colebrook duct, fan curve on a speed schedule, vertical rise, split to two terminals, one half closed from 01:00, damper and constant-volume exhaust), 24 steps | flow rel 64 $u$, floor 64 $u$ × max; mf: the implicit step's first-order error bound, derived on the run (at most 8.3e-6 relative) | **1.4e-7** flow, **1.0e-8** terminal flow, **5.4e-8** mf (2.9 % of the allowance) |
| Three-zone project with a simple air-handling system (zero-volume return and 2 m³ supply plenum, two supply points, one halved from 00:30, a filtered return point, outdoor-air fraction 0.3 then 0.6 from 01:00, minimum outdoor air 0.02 kg/s, filtered recirculation and outdoor air), 24 steps | flow rel 32 $u$, floor 64 $u$ × max; mf: the implicit step's first-order error bound, derived on the run, round the return-air loop (at most 8.6e-6 relative) | **7.9e-7** flow (2 % of the allowance), **3.7e-7** mf (6.0 %) |
| NIST's `test_GetPrjInfo.prj` without its controls and duct leakage: two air-handling systems with zero-volume plenums (all outdoor air) and two fan-driven duct systems, 12 steps (noodl physics' Newton at 1e-12 kg/s, the float64 floor of its 28 kg/(s Pa) terminals) | flow rel 64 $u$, floor 64 $u$ × max; mf: the implicit step's first-order error bound, derived on the run | **1.1e-7** flow, **3.6e-8** terminal flow, **1.0e-7** mf (80 % of the allowance: the zones fill with outdoor air, where the bound is the ambient 1e-3's own float32 roundoff, 0.797 $u$) |
| Three-zone project with the four cubic-spline elements (one crossed against its declared direction, zone two at 303.15 K, a 0.02 kg/s supply), 24 steps | flow rel 32 $u$, mf rel 64 $u$, floor 64 $u$ × each species' max | **1.1e-8** flow, **2.7e-8** mf |
| Demand-controlled ventilation: a PI controller on zone one's CO2 drives an exhaust fan (never below 0.2), occupants 08:00–18:00 through a schedule node, 288 steps; the named control outputs (CO2 reading, PI output, fan signal, excess) too | float32 control arithmetic and noodl's own CO2 reading propagated through the linearised loop, its response sums maximised over the visited operating range (including the open loop while the fan holds its 0.2 floor): PI output and fan signal absolute 38 539 $u$; flow rel 64 $u$ + 38 539 $u$ × the fan rating; mf rel 64 $u$ + 222 $u$ (1.71e-5) | **5.4e-6** flow (0.1 % of the allowance), **4.8e-8** mf, control outputs 0.7 % of the allowance |
| Pressure-controlled exhaust: zone three's gauge pressure above −0.5 Pa through a P controller (Kp 0.8) drives the exhaust fan; a path pressure difference and a flow ratio reported, 288 steps | rel 64 $u$, floor 64 $u$ × max (control outputs: × each output's max) | **5.3e-7** flow, control outputs 3 % of the allowance (clean air) |
| Hysteresis window: zone one's temperature ramped through a schedule node, a window opened by hysteresis and a band switch, every logical and arithmetic node reported, 288 steps | the sensed $x = 0.1\,(T - 293.15)$ is off by an absolute $0.1\,(T + 293.15)\,u + 2u$ = 61.6 $u$ in float32 (offset cancellation); control outputs 2.67 × that, flows 1.67 × that × the window's flow per unit opening, both on top of rel 64 $u$ and the floor | **1.6e-4** flow on the window at opening 0.028 (41 % of the allowance), control outputs 38 % (clean air) |
| The simple-AHS and duct-network projects with the outdoor-air fraction, a supply point, a fan's speed and a terminal's opening driven through schedule nodes instead of schedules | as the scheduled rows (mass fractions: the derived bound); noodl physics' runs equal the scheduled runs exactly | **7.9e-7** / **1.4e-7** flow, **3.7e-7** / **5.4e-8** mf, **1.0e-8** terminal flow |
| Three-zone project with two occupants (one moving mid-step, out of the building for an hour, generating on a schedule), zones at 303.15 / 293.15 / 283.15 K in a 263.15 K ambient, zone three outside the building volume, 48 steps: mass fractions, exposure, age of air, building air change rate | flow rel 33 $u$, mf rel 107 $u$ (three zones in series at 35 $u$ each, the generation rate and the initial mass fraction), floor 64 $u$ × max; exposure 107 $u$ + print (5e-7); age 35 $u$ + print (at most 2.6e-6); air change rate 34 $u$ + print (at most 2.5e-6) | **4.6e-8** flow (0.8 % of the allowance), **1.3e-7** mf (0.3 %), **4.0e-7** exposure (ContamX's print precision; 1.2 %), **1.1e-7** age (4.3 %), **1.5e-7** air change rate (5.9 %) |
| The duct-network and simple-AHS projects, each also with zone two outside the building volume, and the duct project with a return duct (zone three → junctions → zone one): age of air and building air change rate, 24 steps | the flows' rows above; mass fractions: the derived bound (≤ 8.6e-6 relative; measured 1.2–6.0 % of it); age: the first-order bound $3u\,\tau + A^{-1} r$ of the age equation $A\tau = M$ under the flow budget, evaluated each step ($r_i = \sum_{in} \lvert\delta q\rvert\,\lvert\tau_i - \tau_{up}\rvert$), at most 6.1e-6 – 9.9e-6 relative + print; air change rate: the entering flows' budgets over $1.2041\,V$, plus 2 $u$, + print (at most 3.1e-6 – 6.1e-6) | age **2.6e-7 – 5.0e-7** (≤ 9 % of the allowance), air change rate **5.7e-8 – 3.0e-7** (≤ 12 %) |

Sinks and the concentration-proportional parts of sources are implicit in ContamX's species
step, as are kinetic reactions and filters; noodl's implicit step reproduces it to float32
roundoff. ContamX evaluates the NRCC power law with time in hours, does not clamp a cutoff
source above its cutoff, and does not apply a species' decay constant; noodl follows all three
(measured). A pressure-driven source sees the pressure difference of the step's own airflow
solution, so the step is repeated until the zone pressure it read is the solved one.

The stack rows rely on two density effects matching ContamX's. Orifice coefficients are
corrected for the density of the air entering the path (freezing them at reference density
cost 1.7e-2, growing with $\Delta T$). And the project sets `densZP = 1`, zone density at the
zone's absolute pressure: ignoring it left a residual of 4.1e-5 – 4.4e-5, flat across the sweep,
because the stack pressure is a density difference that amplifies the |P_zone| / Pb ≈ 6e-6
density change about fourteenfold. The three-zone transient was 6.5e-6 while zone air mass was
held at the reference density 1.2041 kg/m³ rather than ContamX's `Pb / (R T)` (2.2e-6 apart at
293.15 K). The scheduled three-zone project was 1e-3 off until two more ContamX rules were
matched: its wind pressure profiles are cubic splines, periodic in the wind angle (linear
interpolation left 7.3e-5 kg/s once the wind turned away from a tabulated angle), and with
`dodMdt` the zone air mass is taken at the zone's absolute pressure even when the densities are
not (5.9e-6 kg/s when the fan switches on, 4.4e-9 in the mass fractions when the window
reopens). What remains on these rows is one or two float32 roundoffs. One gap is known and its
cause is not identified: a variant of the scheduled three-zone project with `dodMdt = 0` differs from
ContamX by 4.4e-9 in mass fraction when the window reopens, 2.3 times the floor.

The whole-project rows pin how ContamX runs a project over time, each convention measured on
the engine before it was implemented. Every input (schedule, weather, ambient file) is evaluated
at the **end** of each step, and at a jump the step ending on it still sees the old value (the
left limit): the zone schedule's switch to 303.15 K at 13:00 first acts in the step ending at
13:05, and so does a switch moved to 13:02:30. At midnight the left limit is the previous day's
value, and the new day's schedule acts from the first step after it, with the day type from the
weather file's per-day record. With `dodMdt = 1` each zone's air-mass change enters its airflow
balance: in the step after the 10 K jump the 600 m³ zone's net inflow is
$P_b V / R \,(1/303.15 - 1/293.15) / 300\,\mathrm{s} = -0.0795$ kg/s, and `simulate` adds the
same $-(M_1 - M_0)/\Delta t$ term; noodl physics matches ContamX's imbalance in that step to 0.4 $u$. A
weather file without a 00:00 row on its second day is interpolated linearly from the first day's
24:00 row, as ContamX does. ContamX holds a day schedule's values in single precision but the
weather's temperatures more precisely: with zone and weather both at 293.15 K, it drives a
2.9e-5 kg/s exchange flow, exactly the stack flow of a zone at float32(293.15), 6.1e-6 K colder
(it vanishes at 293.125 K, which float32 represents exactly). The comparison gives noodl physics
the schedule values ContamX holds; otherwise the mass fractions drift 1.4e-4 apart over that hour.

ContamX writes the occupants' exposure, the age of air and the building air change rate to
result files (`.ebw`, `.age`, `.ach`) when the project's output flags ask for them. `run_project`
reads them, and noodl's `occupant_exposure`, `local_mean_age` and `building_air_change_rate` are
compared with them. ContamX splits a step at an occupant's move, both for the exposure and for
what the occupant generates, takes the occupant's presence as a step mean and the generation
schedule at the step's end; noodl does the same (taking the schedule as a step mean instead
was 7.2e-3 off). It computes the age with each zone's own air mass, a zero-volume plenum
holding none, and the building rate with 1.2041 kg/m³ over the conditioned volume: every zone
without the "outside the building volume" flag plus the duct junctions and the air handlers'
plenums. It counts every flow that enters that space, including from zones left out of it,
and none that moves between a zone and a junction: with a return duct carrying 0.048 kg/s
from zone three to zone one, taking the zones alone as the conditioned space (the junctions
outside it, their terminals' flows entering and the duct volume left out) is 15–22 % off
ContamX's rate. noodl does the same (measured). The age and air-change budgets are
first-order bounds of the flows' and air masses' float32 budgets through the age equation
and the sum of entering flows, evaluated on the run; the age is not amplified along a chain
of zones (each zone adds its own $M_i/q$), and the AHS's recirculation raises the bound to
about 2.7 times the flow budget. The mass-fraction budgets of the duct and AHS rows are
derived the same way: the first-order error of the implicit species step under the flows',
filters', air masses' and ambient mass fraction's float32 roundoff, carried from step to
step through the step matrix (an M-matrix, so its inverse is non-negative), which follows the
return air round its loop instead of assuming a chain of zones.

The path-flow sign convention was verified independently against two cases rather than assumed,
since `contamxpy`'s own documentation does not pin the sign of `getPathFlow`.

## Against OpenModelica (Modelica Buildings Library)

**Algebraic models (7, no volumes).** Every CSV row and column against OpenModelica's own
simulation, `|noodl - omc| <= 1e-6 |omc| + 1e-9` kg/s:

| Model | Worst relative error | Rows compared |
|---|---|---|
| OneWayFlow | 2.4e-16 | 501 |
| DoorOpenClosed | 1.2e-16 | 501 |
| OpenDoorPressure | 2.4e-16 | 49 |
| OpenDoorTemperature | 1.8e-11 | 49 |
| Orifice | 4.0e-16 | 501 |
| PowerLaw | 6.1e-16 | 501 |
| Outside_Cp | 5.0e-16 | 501 |

All at round-off except `OpenDoorTemperature`, whose discretised-door port flows sit at 1.8e-11
relative — still five orders of magnitude inside the 1e-6 bound, and attributed to an
OpenModelica nonlinear-solver residual on the door's inflow-density loop, not a formula
difference (likely, not diagnosed: not shown by a tighter `omc` tolerance).

The **CONTAM cross-check** on `OneWayFlow` (13 pressure-difference knots × 8 elements, from
CONTAM's own validation table, `OneWayFlow.mo`'s `contamData`): noodl physics differs from CONTAM by at
most 0.74 % relative (7.68e-4 kg/s absolute) — exactly the amount MBL itself differs from CONTAM.
The assertion is `|noodl - contam| <= |omc - contam| + 1e-6 |omc| + 1e-9` at every entry: noodl physics
adds nothing to MBL's own departure from the table (41 of 104 entries miss the table's 3
significant figures, in MBL as much as in noodl physics).

**Weather and wind pressure.** The weather a `ReaderTMY3` puts on the bus (dry-bulb temperature,
pressure, relative humidity, wind speed) and every wind boundary's `pWin` are compared at every row,
separately from any time integration (`assemble.weather_columns`), on MBL's
`Buildings.Fluid.Sources` examples (four orientations of `Outside_CpLowRise` and of `Outside_CpData`
over a week from day 200, a full sweep of the wind direction over symmetric, asymmetric and narrow
profiles), on `Outside_Cp`, `PressurizationData` and `TrickleVent`. Being the same arithmetic as MBL's, they are
held to round-off, `|noodl - omc| <= 1e-12 |omc|` with no absolute floor; the weather comes out
identical to the last bit, the calendar time included (MBL's `t - k lenWea + lenWea` rounds the
model time to the spacing of doubles near a year, 3.7e-9 s, and noodl rounds it the same way):

| Model | Worst relative error (weather) | Worst relative error (pWin) | Rows |
|---|---|---|---|
| Outside_Cp | 0.0e+00 | 3.4e-16 | 501 |
| Outside_CpLowRise | 0.0e+00 | 5.4e-16 | 501 |
| Outside_CpData | 0.0e+00 | 6.0e-16 | 501 |
| Outside_CpData_Angles | 0.0e+00 | 4.6e-16 | 501 |
| Outside_CpData_Directions | 0.0e+00 | 4.3e-16 | 501 |
| PressurizationData | 0.0e+00 | 9.1e-16 | 501 |
| TrickleVent | 0.0e+00 | 1.6e-15 | 4321 |

`PressurizationData` and `TrickleVent` import, but they are not compared as dynamic models (see
[Limitations](../../applications/building/limitations.md)). Their weather and wind pressures are compared, in the last rows above.

**Dynamic models (15, with volumes).** Every volume stores its air compressibly, as MBL's
`MixingVolume` does (see "Volume mass storage" below), and every model is compared at the
reference's own precision. The reference CSVs were regenerated with DASSL at a tolerance of
1e-13; at 1e-14 DASSL returns no trajectory. At the models' declared 1e-6 the reference's
own error was of the order of the comparison: 5.6e-5 K on `ZonalFlow`, and a 5e-5 relative
door-flow wiggle after `CO2TransportStep`'s pulse.

noodl physics runs `scheme="midpoint"` and combines two runs, with steps in ratio 2, by Richardson
extrapolation (`run.extrapolate`). The midpoint scheme is second order. The transport layers advance
over two halves of each step, the first with the start-of-step flows, capacities and sources and the
second with the end-of-step ones. Over the whole step the boundary values follow the cubic Hermite
through their values and rates at both ends (the straight line where they are constant). The
volumes' mass advances with the L-stable BDF2 rate. A zone that flushes much faster than a step then
ends the step behind its boundary by its own flushing time at the step's end. Held at the step's
mean flows and boundary values, it fell half a step behind, an error of first order that
extrapolation cannot remove. Extrapolation needs the error to expand smoothly in the step, so the
steps are laid out with care:

- Every signal event (a step, a pulse edge, a ramp corner) is a step boundary.
- Every **switch** is a step boundary too. A switch is where an element's law changes piece
  (the edges of its regularisation band, `ConstitutiveLaw.switching`) or where an edge flow changes
  sign, since the transport layers upwind on it. A step across one is split there, the
  crossing located on the step's own runs. `OneOpenDoor`'s door flows, reversing through their
  bands across a 14.4 s step, were off by 1.1e-4 of the flow floor before; with the switches
  located they are at the reference's resolution.
- The run at two substeps takes exactly the steps of the run at one, halved. The volumes start
  at MBL's `p_start`, so a start that releases an imbalance begins on a graded sub-grid; a
  start in balance only doubles its steps up from a tiny first one.

`ClosedDoors`, `ReverseBuoyancy`, `ChimneyShaftWithVolume` and `ChimneyShaftNoVolume` combine
two and four substeps per output interval: at one step per interval their crack flows switching
on (`ClosedDoors`), their orifices reversing at 18 s (`ReverseBuoyancy`) and the room controller
answering the end of the injection at 601 s (both chimney shafts) are not yet in the asymptotic
range of the extrapolation.

`ChimneyShaftWithVolume` adds a room heated by a P controller on its temperature
(`LimPID`, through a `Gain` into a `PrescribedHeatFlow`), a `MediumColumnDynamic` shaft and a
source that injects and then extracts 0.05 kg/s. Its controller output `con.y` (column `y`)
is compared like any other column; it is five times the room's deviation from its set point, a
difference of two temperatures, so its relative error is the strictest of the model's. It is at
7.4e-7 (7.1e-7 beyond the reference's resolution), 4 s after the injection stops: its 3.6e-8
there is 7e-9 K of room temperature (2.5e-11 relative).

`ChimneyShaftNoVolume` is the same room with its shaft a plain column between two orifices in
series, joined at a junction without storage. The shaft's density is that of the air entering
it (`densitySelection = actual`): the room's while the flow goes up, the outside air's while it
comes down, so the shaft's head jumps where its flow reverses, at 0.192 s and at 1800.25 s
(the extraction's onset). The run locates each reversal, ends the step there and starts the
next from the airflow of the new direction; a reversal within the locator's margin of a step's
start or end (the larger of 1e-3 of the step and 1e-5 of the output interval) is moved to it,
an error of that margin times the jump in the flow (the first reversal comes within the
margin of a band switch the run has just located, the flow 6.1e-5 kg/s short of zero there at
one substep and 3.0e-5 at two, and is taken at it). Its controller output is within
1.3e-7 of the reference beyond its resolution, its flows within 2.4e-7, its room temperature
within 8e-12. OpenModelica chatters at the first reversal: the 1e-13 export holds six rows at
0.192 s in place of the output times 7.2 to 43.2 s. The 1e-12 export has two such rows, in
place of 7.2 and 14.4 s, and holds 21.6 to 43.2 s; those four rows are compared against it
at 1e-6 with no resolution allowance (con.y 8.3e-7 at 28.8 s, the closest). Only 7.2 and
14.4 s are compared against neither.

The metric is `|noodl - omc| / max(|omc|, floor)` at every row after `t = StartTime`. The floor
is 1e-3 of the model's largest flow for flows, and 1e-3 of the column's largest value
otherwise. The tolerance is 1e-6, applied to the difference beyond the reference's own
resolution. That resolution is, per model and variable kind, the largest amount the reference
moved when its tolerance was tightened from 1e-12 to 1e-13: a difference the reference itself
moves by under a tenfold tighter tolerance is not resolved by it. It decides only a few flows,
where noodl physics' runs at 1, 2 and 4 substeps agree to 1e-10 kg/s and the reference does not.
`OneOpenDoor`'s door flows at 28.8 s are 1.2e-8 kg/s from the reference, which moved by up to
2.9e-8 kg/s over its first rows. `NaturalVentilation`'s orifices at their reversal are
5.0e-10 kg/s from it, where it moved by up to 3.7e-9 kg/s. `ClosedDoors`' crack flows are
at most 1.3e-12 kg/s from it (the reference moved by 1.2e-11 kg/s), and `ReverseBuoyancy`'s
door flow at 612 s 1.7e-8 kg/s (it moved by 2.2e-8 kg/s). `ReverseBuoyancy3Zones`' door flow now passes
through the same allowance, since the two-halves scheme: its worst difference is 1.03e-6 relative
(4.9e-9 kg/s absolute) against a reference resolution of 1.8e-8 kg/s.

The `t = StartTime` row, MBL's `p_start` initialisation, is reproduced too. Worst relative
error per variable, with the part beyond the reference's resolution in brackets where the two
differ and the first is above 1e-6:

| Model | flow | T | p | Xi | C | substeps |
|---|---|---|---|---|---|---|
| ThreeRoomsContam | 4.3e-07 | 7.1e-09 | 1.3e-14 | 1.1e-11 | 0 | 1, 2 |
| ThreeRoomsContamDiscretizedDoor | 4.1e-07 | 7.0e-09 | 6.6e-13 | 8.9e-12 | 0 | 1, 2 |
| OneRoom | 1.5e-11 | 1.3e-13 | 1.4e-16 | 1.7e-16 | — | 1, 2 |
| ZonalFlow | 0 | 2.3e-12 | 1.4e-16 | 3.3e-12 | — | 1, 2 |
| CO2TransportStep | 4.5e-07 | 7.1e-09 | 2.7e-14 | 3.3e-11 | 3.8e-07 | 1, 2 |
| OpenDoorBuoyancyDynamic | 5.8e-07 | 1.2e-10 | 1.4e-14 | 1.6e-14 | — | 1, 2 |
| OpenDoorBuoyancyPressureDynamic | 1.8e-07 | 1.6e-10 | 4.2e-15 | 4.8e-12 | — | 1, 2 |
| NaturalVentilation | 1.1e-06 (0) | 1.8e-13 | 1.5e-14 | 1.6e-14 | — | 1, 2 |
| ReverseBuoyancy3Zones | 1.0e-06 (0) | 3.3e-10 | 6.9e-14 | 1.1e-12 | — | 1, 2 |
| ClosedDoors | 1.2e-05 (4.9e-07) | 6.7e-12 | 3.7e-12 | 1.3e-11 | — | 2, 4 |
| OneOpenDoor | 7.2e-05 (0) | 1.2e-10 | 2.1e-11 | — | — | 1, 2 |
| ReverseBuoyancy | 8.1e-06 (0) | 1.5e-10 | 4.7e-10 | 4.7e-10 | — | 2, 4 |
| OneEffectiveAirLeakageArea | 4.4e-07 | 7.4e-13 | 1.8e-13 | 3.0e-13 | — | 1, 2 |
| ChimneyShaftWithVolume | 1.1e-08 | 2.7e-11 | 9.1e-14 | 1.3e-12 | — | 2, 4 |
| ChimneyShaftNoVolume | 2.7e-07 | 8.6e-12 | 2.2e-11 | — | — | 2, 4 |

No model needs a bound above 1e-6. The storage-dominated models
also assert the storage mechanism on noodl physics' own history. `ClosedDoors` and
`OneOpenDoor` heat at constant volume (`sum V dp = (R/cv) int Q`): MBL's temperature rise over
noodl physics' is 1 to 1e-4, where the quasi-steady route was off by cp/cv = 1.40.
`ReverseBuoyancy` starts at MBL's `p_start` against a 100000 Pa boundary and cools as MBL
does: 0.8301 K in both by 21.6 s. `OneEffectiveAirLeakageArea` stores what its source injects,
to 3.5e-14 of the 36 kg injected.

**Volume mass storage.** MBL's `MixingVolume` (`ConservationEquation.mo`) integrates the volume's
mass `m = V rho(p, T, X)`, its energy `U = m u`, its water mass fraction (`der(Xi) = mbXi/m`,
not the water mass) and its trace-substance masses. The reader writes these balances into the
existing layers: the air layer gets a node storage per volume (`NodeStorage`, the volume's
air mass; so a zone's pressure is an unknown even with no boundary in its group), the thermal
and species layers get the zone's actual mass as a per-step capacity and the storage-rate form
of `TransportLayer(dilution=...)`,
and the energy balance gets MBL's flow-work, latent and moist-air carrier terms as sources
. `read_modelica(..., mass_storage=False)`
gives the quasi-steady airflow of the CONTAM route instead.

The comparison is `tests/verification/test_modelica_parity.py`. Every column's numbers (not just the worst) are recorded in the parity records the repository ships (`parity-algebraic.json` and `parity-dynamic.json`). The default run compares the first rows of every dynamic model; the `slow` marker compares every row of every model.

## Against analytical solutions

Separately from ContamX parity, `tests/verification/test_natural_ventilation.py` and `tests/verification/test_airflow_closed_forms.py` check the
coupled airflow-heat physics against the closed-form solutions of Li and Delsante (2001) for a
single ventilated zone — buoyancy-only, envelope-loss, assisting-wind and opposing-wind cubics —
plus an independent `scipy.optimize.fsolve` reference for a two-zone doorway case and a golden
regression at rtol 1e-8.

| Case | Tolerance | Measured |
|---|---|---|
| Airflow closed forms: series and parallel power laws, a fan-driven zone, a fan curve against a leak, a three-zone stack (`scipy.optimize.brentq` or `fsolve` where no closed form exists), each once alone and as a batch of 64 random instances: flows, zone pressures | 1e-9 relative, no absolute allowance, Newton solved to 1e-12 | Single instances at round-off (at most 8.2e-16); batches of 64 up to 9.7e-11 (stack), 1.6e-11 (fan-driven zone), 1.1e-11 (series) |
| Same, nodal residual | 1e-11 kg/s, ten times Newton's stopping bound | within the bound |
| Natural ventilation, single zone (buoyancy only, envelope loss, assisting wind, opposing wind with three steady states): ventilation flow | 1e-6 (1e-4 on the three-root opposing-wind case) | at most 1.8e-10; the stable opposing-wind states are found by time stepping |
| Natural ventilation, two-zone doorway: zone temperatures | 1e-5 K absolute | 7.1e-10 K |

The airflow closed forms check the solver and the element laws, not CONTAM itself. The residual error is Newton's stopping tolerance. At the default tolerance (the square root of machine epsilon, about 1.5e-8) the error reaches 1.7e-6 on small flows. That is the per-instance stopping rule, not batching: on the series case the batch and 64 separate solves stop at the same 9.8e-7. Coupled steady states converge to the Newton tolerance.

`tests/apps/building_physics/test_outputs.py` checks the outputs of occupants, age of air and air change rates against closed forms. The
local mean age of air: one zone ($\rho V / q$), three zones in series (the ages add up) and
two zones with recirculation, to 1e-12, and an air handler's zero-volume plenum, whose age
is that of the zone it draws from. The exposure of an occupant in a zone filling from clean
air: the analytic integral less the implicit-Euler lag $x_s\,\Delta t / 2$, to 1e-9. The
Wells-Riley probability in its quanta-concentration form from a simulated exposure: the
classic steady form, to 1e-12. The zone and building air change rates: inflow over air mass,
and the flows entering a conditioned space over 1.2041 kg/m³ times its volume, on one zone,
two zones exchanging air and three zones in series with one left out of the space, to 1e-14.
