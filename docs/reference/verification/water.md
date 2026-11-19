# Verification: water distribution

Evidence for the [water distribution application](../../applications/water/index.md) (hydraulics; water quality is on [its own page](water-quality.md)); the summary is on the [verification overview](index.md). What is not covered, and where noodl differs from EPANET on purpose, is on [Limitations](../../applications/water/limitations.md).

## Method

The comparisons run in `tests/verification/test_water_parity.py` (hydraulics, rules, energy) and `tests/verification/test_water_networks.py` (whole networks). Water quality is on [Verification: water quality](water-quality.md).

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
| A batch of two Net1 instances (0.8 and 1.25 times the demand) against each run alone on the batch's step boundaries; against the plain single runs | 1e-9 m / 1e-2 m | exact to 1e-9 m; 6.5e-3 m (the extra boundaries' explicit-step difference, see [Limitations](../../applications/water/limitations.md)) |
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
| Every pipe a check valve at five times the demand (two close) | 2e-6 (EPANET's closed-link leak, see [Limitations](../../applications/water/limitations.md)) / 1e-6 m³/s / equal | 1.4e-6, 5.7e-8 m³/s |
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
| ky10 at time zero (13 constant-power pumps, PRVs, tanks), EPANET's demands; EPANET at `DAMPLIMIT 1e-2` (two builds, see [Where EPANET's answer is not unique](#where-epanets-answer-is-not-unique)) | 1e-6 / 1e-6 m³/s / equal | 1.2e-7, 5.4e-8 |
| ky4 at time zero (two constant-power pumps, one closed, an empty tank, 959 junctions), demand pattern at t = 0 | 1e-6 / 1e-6 m³/s / equal | 8.2e-8, 9.1e-8 (two parallel-pipe pairs: 2.6e-7) |
| Anytown at time zero, pumps running, tanks at mid-level, the five placeholder pipes feeding junctions 5-7 at 12 in (three parallel pumps on the five-point curve; pipe 125 still 0.0001 in) | 1e-6 / 1e-6 m³/s / equal | 9.3e-8, 3.1e-8 |
| Whole networks through `water_run` for their full DURATION: heads (relative to the larger of the head and 1 m), tank levels, flows, states at every report time, and the energy section | 1e-5 / 2e-4 m / 1e-6 m³/s / equal / 1e-5, except where noted | the next five rows, in that order |
| Net3, 168 h (AT TIME and level controls, a closed pipe, three-point pumps) | as above | 1.2e-7, 5.2e-6 m, 1.3e-7 m³/s, equal, 4.2e-8 |
| Net6, 96 h (3323 junctions, 61 pumps, PRVs, a check valve, 124 level controls) | heads 2e-5, levels 3e-3 m, flows 5e-4 m³/s, energy 3e-4 (EPANET's closed-link leak, see [Limitations](../../applications/water/limitations.md): the heads at 90 h follow a tank level 1.5e-3 m apart) | 1.1e-5, 1.94e-3 m, 3.1e-4 m³/s, equal, 1.95e-4 |
| ky4, DURATION 0 | as above | 8.2e-8, 6.0e-6 m, 9.1e-8 m³/s (pairs 2.6e-7), equal, 1.1e-8 |
| ky10, DURATION 0 (level controls on `~@`-named pumps); EPANET at `DAMPLIMIT 1e-2` | as above | 1.2e-7, 1.3e-5 m, 5.5e-8 m³/s, equal, 9.0e-7 |
| Anytown edited, 24 h at a 1 min step | heads 1e-4 (EPANET's closed-link leak out of the empty tanks, at a head of 0.55 m; see [Limitations](../../applications/water/limitations.md)) | 5.1e-5 (1.8e-4 m; 9.3e-8 at t = 0), 1.8e-6 m, 2.9e-7 m³/s, equal, 3.7e-7 |
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
([Darcy-Weisbach](../../applications/water/physics.md#darcy-weisbach)) the D-W rows sit at the float32 output floor. The per-pipe loss bound is derived, not
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

**PRV and PSV.** The eleven cases are below. The two-loop cases add junction J7 (P2 ends there) and the valve
V1 from J7 to J2 (250 mm); the series cases have pipes of 1000 m, 200 mm, C = 100 and valves
of 200 mm, with a demand of 20 L/s at the end of the PRV line. Heads are compared with the
file's demands (the pipes carry EPANET's rounded LPS factor; with the exact one the worst was
2.4e-6, on the C = 100 lines), and EPANET at `ACCURACY 1e-7`: at the two-loop fixture's
0.001 it stops with an active PRV's flow 5.8e-6 m³/s short of its node balance, and its
heads 5.8e-6 relative from converged (measured).

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

## Where EPANET's answer is not unique

These notes explain the comparisons that are bounded or compared at a looser accuracy than the rest. In each, the exact equations have many solutions or EPANET's own iteration has not converged, so the difference is in the reference and not in noodl.

### A zero-demand group fenced off by check valves

A zero-demand group of several junctions joined by a link inside it, fenced off by check valves, is held at the `1 / CBIG` limit head (75 m between reservoirs at 100 and
50 m). EPANET's head there is not that limit. Measured for J - P5 - L fenced off between
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
(75.0759 m on Linux at 1e-5). The tests accept 1.5e-3 relative over two steps (measured 2.8e-4 Windows, 1.0e-3 Linux) and, for the internal-check-valve case, 4.1e-5: J1 = J2 = 56.6667 m, the
limit, against EPANET's 56.6690; EPANET's head there moves with the internal
pipe's length as well, 55.3140 m at 5 m, 56.6690 at 500 m, 56.6655 at 5000 m, and with
the accuracy, 56.6827 m at 1e-7. A single fenced-off junction, with no link inside the group, matches
EPANET to 3e-9, and a junction fed from the highest of several reservoirs through one
check valve, to 8.5e-8 (EPANET's float32 head).

### A group behind a closed link

A PRV (or any valve) `CLOSED` by `[STATUS]` in front of junctions without demand holds them
at the `1 / CBIG` limit, the head on the other side (60 m); EPANET reports 59.930 m for the
same network at `ACCURACY 1e-7` and 1e-8 (53.687 m at 1e-5, where it stops after two
iterations), with a PRV or a TCV alike (1.2e-3 relative).
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
pump in [Limitations](../../applications/water/limitations.md#cases-with-no-unique-head). These are wntr's
Windows build; its Linux build marks the valve `XPRESSURE` on four of the cases (a PSV with
the dead end on its end, a PRV on its start) and holds the dead end OPEN at the limit head,
noodl's head (see the two builds, below).

### EPANET's two builds

wntr 1.5.0 ships `libepanet22` built with MSVC
(Windows) and with GCC 7.4 (Linux), both version 2.2 with the same toolkit. Measured where they
part: (1) ky10 at time zero, at EPANET's default `DAMPLIMIT 0`, the Linux build stops on a
second branch that is no solution, `~@RV-4` CLOSED and the constant-power pump `~@Pump-11` at
2.1e-22 m³/s with 7.9 m of head (its law asks 7e21 m: below `sqrt(r / CBIG)` the `CBIG` slope cap
makes it a closed link, as for `~@Pump-10` above), from which its 24 h is another run; at
`DAMPLIMIT 1e-2` (PRV and PSV status checks deferred to a relative flow change of 1e-2, the
flow updates damped there) both builds take Windows' branch, and noodl's, at time zero to
1.2e-7, and the same 50 steps over 24 h, so every ky10 reference runs at 1e-2. (2) A fenced
zero-demand group's head (see above) is resolved only to steps of 0.0546 m, one ulp of its link's
conductance in EPANET's pivot: 75.0213 m on Windows, 75.0759 m on Linux. (3) A PRV or PSV
feeding a dead end on the side it does not fix: the dead end's Cholesky pivot is zero up to
its last bit, and where the Linux build's is not positive `badvalve` marks the valve
`XPRESSURE` (warning 5, "open but cannot deliver pressure"): the valve OPEN and the dead end at
the limit head, on four of the eight dead-end cases (see above). (4) The age and trace points that
follow EPANET's unconverged flows (ky4's J-703 and P-696, ky10's P-310) differ in which pass the
bound. Each test names its build's values.

### EPANET's own accuracy

A file's `ACCURACY` is clamped to 1e-5 by
EPANET's reader; the reference sets 1e-7 through the toolkit. At 1e-8 EPANET leaves Net2 and Net3
hydraulically unbalanced at several steps, so 1e-7 is the tightest reference that converges
everywhere. noodl's quality on EPANET's own saved hydraulics matches EPANET to 5.2e-11 (Net6)
and 1.4e-12 h (ky10), so the differences in the comparison are in the hydraulics' inputs, not the router.
