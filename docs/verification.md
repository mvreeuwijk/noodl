# Verification against reference models

Each noodl physics application reimplements the physics of an established tool. This page
summarises how closely each one reproduces that tool: one row per code-to-code comparison,
with the tolerance, how it was justified, the measured agreement, and the stated cause of any
difference that remains. The details, and every intermediate number, are on the linked
application pages.

*Verification* here means comparison with another code or with a closed-form solution. It
says nothing about whether either code matches reality. *Validation*, comparison with
measurements, is reported separately where it exists (for example the Pescod and Price
ventilation measurements on the [sewer page](applications/sewer.md#verification)).

## How to read the table

- **Tolerances are set from the reference, not fitted to the result.** Most are derived from
  the reference's own precision: EPANET's and SWMM's float32 binary output, ContamX's
  single-precision inputs, OpenModelica's resolution at a DASSL tolerance of 1e-13. Where a
  tolerance is only a regression guard (some SIRANE checks), the row says so.
- **The street air-quality rows are under revision.** A change to the street chemistry is
  pending, and these numbers are the ones the street page states today.
- **Relative unless marked otherwise.** "abs" is an absolute bound in the quantity's own units.
- **Live engine** means the test runs the reference code during the test. **Stored output**
  means the test reads results the reference produced earlier, committed under `tests/data/`.

## Summary

| App | Reference model and version | How the reference runs | Cases | Quantities compared | Tolerance and justification | Measured agreement | Cause of remaining difference |
|---|---|---|---|---|---|---|---|
| [Water distribution](applications/water.md#verification), Hazen-Williams | EPANET 2.2 (`epanet22.dll` bundled in `wntr` 1.5.0) | Live engine, `wntr.sim.EpanetSimulator` | `twoloop_si.inp`; Net1 single period; Net1 24 h with level-triggered pump controls | Heads, flows, pump head gain, tank level | 1e-6 heads, flows and pump gain (1e-5 on Net1 flows); 2e-4 m abs on tank level. EPANET's output is float32, about 1e-7 | Two-loop 4.4e-7 heads, 8.1e-8 flows; Net1 7.1e-8 / 2.9e-6 / 1.2e-7; tank 8.2e-5 m worst of 25 steps | EPANET's float32 output, and EPANET's own continuity miss of 4.5e-10 m³/s at Net1 node 13. noodl physics' continuity holds to 2.1e-14 |
| [Water distribution](applications/water.md#darcy-weisbach), Darcy-Weisbach | EPANET 2.2, as above | Live engine | `twoloop_si.inp`; Net1 single period; `lowflow_dw_si.inp` (laminar, transitional and turbulent pipes) | Heads, flows, per-pipe head loss | 1e-6 heads and flows (1e-5 on Net1 flows); 2.65e-7 m abs on head loss, derived from EPANET writing heads as float32 feet, then float32 metres | Two-loop 5.8e-8 / 6.5e-8; Net1 7.0e-8 / 4.0e-6; low-flow 3.5e-8 / 9.4e-8 / 9.6e-8 m | Float32 output floor. Reached only with EPANET's own composite law and its rounded unit factors; `friction="colebrook"` is a different law (3.6e-4 on heads) |
| [Water distribution](applications/water.md#verification), demand and quality | EPANET 2.2, as above | Live engine | Two-loop with `DEMAND MODEL PDA`; single-source TRACE | Heads, delivered demands; traced percentage | 1e-5 (PDA); 1e-3 percentage points (TRACE) | 3.5e-7 heads, 2.2e-7 demands; 6.4e-12 points | Float32 output. Several traced sources are not compared |
| [Sewers](applications/sewer.md#verification), hydraulics | SWMM 5.2.4 (engine version and zero routing error asserted) | Live engine, `pyswmm` | Kinematic-wave trees: 5, 13 (EPA *Example 1*, converted) and 32 conduits | Pipe flows; depths and volumes (live, float64); velocities (binary output, float32) | Flows 1e-9; depths and volumes 1e-10, three decades above a ~1e-13 rounding bound; velocities 1e-7, above float32's half-ulp of 5.96e-8 | Flows ≤ 6.2e-14; depths ≤ 1.7e-14; volumes ≤ 2.3e-14; velocities ≤ 5.8e-8 | None beyond rounding, with `geometry="tabulated"` (SWMM's own tabulated circle; `"swmm"` is an alias). The exact circle differs by up to 5.2e-3 in depth and 0.18 in velocity |
| [Sewers](applications/sewer.md#dynamic-wave-against-swmm), dynamic wave (core; the other fixtures are in [the next section](#sewer-dynamic-wave-fixture-by-fixture)) | SWMM 5.2.4, `HEAD_TOLERANCE` 1e-12 m, `MAX_TRIALS` 500, fixed 5 s step; noodl `iterate_tol` 1e-13 m, `iterate_max` 2000 | Live engine, `pyswmm`; 0 % of steps not converging asserted | Tree (5 conduits), backwater line, loop with an adverse conduit; 30 min | Node depths and link flows at every 5 s step | Depths 1e-7 m; flows 1e-6 relative + 1e-9 m3/s | Depths 1.2e-9 / 4.5e-9 / 3.4e-9 m (tree / backwater / loop); flows within the target at every step | Iteration tolerances on both sides. SWMM at its default tolerance (0.0015 m, 8 trials) is off the fixed point by 2.3e-3 / 1.3e-2 / 3.6e-3 m in depth and 0.8 / 7.4 / 3.6 % of maximum flow, while reporting 0 % non-convergence |
| [Sewers](applications/sewer.md#verification), steady quality | SWMM 5.2.4, as above | Live engine | Decaying tracer on the 5-conduit tree | Link concentration: tank-in-series closed form, and the model's quality layer | 1e-9 | 4.5e-15 (closed form); 6.3e-13 (quality layer) | None. Both codes react, then mix, and report after mixing |
| [Sewers](applications/sewer.md#verification), transient quality | SWMM 5.2.4, as above | Live engine, sampled every 5 s routing step | Time-varying tracer load on the 5-conduit tree | Link concentration at every step | 1e-9 on the conduit fed by a lateral inflow; downstream conduits not asserted | Head conduit 1.1e-14; downstream 1.1e-2 and 1.7e-2 of peak at 5 s (2.2e-3 and 3.5e-3 at 1 s) | **Intentional:** SWMM's one-step junction lag (see below). SWMM equals the lagged recursion on noodl physics' own volumes and flows to 5e-15 |
| [Flow allocation](applications/allocation.md#verification), shipped demos | WSIMOD 0.8.1 (pinned exactly) | Stored output: every per-arc request and realised flow captured from WSIMOD's own run | `quickstart_demo` (1,456 steps); `oxford_demo`, 20 of 21 arcs | Realised arc flows, hard clip | 1e-9 abs (quickstart); 1e-6 abs (oxford, flows up to about 1e6) | 1.1e-16; 1.9e-9 | Float64 noise. The demos never bind a capacity, so this checks only the pass-through path. `sewer_to_wwtw` is excluded: WSIMOD caps it at a node, not an arc |
| [Flow allocation](applications/allocation.md#binding-capacities-and-headroom), binding capacities | WSIMOD 0.8.1 | Live engine, WSIMOD's own `Arc`, `Node`, `Storage`, `Waste` classes | 6 scripted networks with binding arcs and tanks; `quickstart_tight` (3 capacities lowered, 1,456 steps) | Realised volumes, storage | 1e-12 × case scale: both sides are float64 doing the same arithmetic up to association | ≤ 4.4e-16; 1.0e-16 | **Intentional** where arcs compete for one node's headroom: preference sharing, not first-come (see below). The totals and storage agree |
| [Building physics](applications/building_physics.md#against-contamx) | NIST ContamX 3.4.1.7 | Live engine, `contamxpy` (Windows x86-64 only) | Stack project over a ±20 K ambient sweep; constant-mass-flow fan; three-zone project, steady and 24 transient steps of 300 s | Path flow directions and magnitudes; zone concentrations | First-order budget of float32 input roundoffs, $u = 2^{-24}$: 17.2 to 32.8 $u$ on stack flows (1.02e-6 to 1.96e-6, growing as the temperature difference shrinks), 16 $u$ on three-zone flows, 64 $u$ on concentrations | Stack 7.2e-8 to 9.4e-8; three-zone flows 9.1e-8; concentrations 2.0e-7; flow signs match | One or two float32 roundoffs in ContamX's single-precision inputs |
| [Building physics](applications/building_physics.md#against-contamx), controls | NIST ContamX 3.4.1.7 | Live engine, `contamxpy` (Windows x86-64 only) | Demand-controlled ventilation (a PI controller on CO2 drives an exhaust fan), 288 steps; pressure-controlled exhaust (a P controller on a zone pressure), 288 steps; hysteresis window (a window opened by hysteresis and a band switch on a ramped temperature), 288 steps; the simple-AHS and duct-network projects with their schedules driven through schedule nodes | Path flows; zone mass fractions; every named control output | Derived through the closed loop from ContamX's float32 control arithmetic, so looser on the control outputs than on the plant. DCV: PI output and fan signal 38 539 $u$ absolute; flows 64 $u$ + 38 539 $u$ × the fan rating; mass fractions 64 $u$ + 222 $u$ (1.71e-5). Pressure exhaust: 64 $u$, floor 64 $u$ × each output's max. Hysteresis: the float32 sensed value off by an absolute 61.6 $u$; control outputs 2.67 × that, flows 1.67 × that × the window's flow per unit opening, on top of 64 $u$. Schedule nodes: equal to the scheduled runs | DCV 5.4e-6 flow (0.1 % of the allowance), 4.8e-8 mf, control outputs 0.7 % of the allowance; pressure exhaust 5.3e-7 flow, control outputs 3 %; hysteresis 1.6e-4 flow on the window at opening 0.028 (41 % of the allowance), control outputs 38 %; schedule nodes equal to the scheduled runs exactly | ContamX's float32 control signals, carried through the loop (the PI output's stored value, the sensed temperature's offset cancellation) |
| [Building physics](applications/building_physics.md#against-contamx), occupants and outputs | NIST ContamX 3.4.1.7 | Live engine, `contamxpy` (Windows x86-64 only); ContamX's own result files (`.ebw`, `.age`, `.ach`) | Three-zone project with two occupants (one moving mid-step, out of the building for an hour, generating on a schedule) and a zone outside the building volume, 48 steps; the duct-network and simple-AHS projects, each also with a zone outside the building volume, and the duct project with a return duct, 24 steps | Occupant exposure; local mean age of air; building air change rate; with the flows and zone mass fractions | First-order budgets of float32 input roundoffs, $u = 2^{-24}$, derived through each quantity, plus ContamX's print precision (5e-7). Occupants project: flows 33 $u$, mass fractions and exposure 107 $u$, age 35 $u$ (at most 2.6e-6), air change rate 34 $u$ (at most 2.5e-6). Duct and AHS projects: the age equation's first-order bound under the flow budget, evaluated each step (6.1e-6 to 9.9e-6); the entering flows' budgets over $1.2041\,V$ for the air change rate (3.1e-6 to 6.1e-6); mass fractions the implicit step's error bound derived on the run (at most 8.6e-6) | Occupants project: exposure 4.0e-7 (1.2 % of the allowance), age 1.1e-7 (4.3 %), air change rate 1.5e-7 (5.9 %), mass fractions 1.3e-7 (0.3 %). Duct and AHS projects: age 2.6e-7 to 5.0e-7 (at most 9.1 %), air change rate 5.7e-8 to 3.0e-7 (at most 12 %), mass fractions at most 6.0 % | ContamX's float32 inputs and its printed result files. The rules matched were measured on the engine: presence a step mean, the generation schedule at the step's end (a step mean is 7.2e-3 off), plenums and duct junctions inside the conditioned volume, junction-to-zone flows internal (a zones-only space is 15 to 22 % off on the return duct) |
| [Building physics](applications/building_physics.md#against-openmodelica-modelica-buildings-library), algebraic | Modelica Buildings Library v13.0.0, simulated by OpenModelica 1.27.1 | Stored output (OpenModelica CSV, committed) | 6 models without volumes; plus CONTAM's own validation table on `OneWayFlow` (104 entries) | Every row and column of mass flows | $\lvert\Delta\rvert \le$ 1e-6 $\lvert$ref$\rvert$ + 1e-9 kg/s; for the table, no more than MBL's own departure from it | ≤ 6.1e-16 on five models, 1.8e-11 on `OpenDoorTemperature`; table: 0.74 %, the same as MBL's own | `OpenDoorTemperature`: likely an OpenModelica nonlinear-solver residual (not diagnosed). The CONTAM table's 3 significant figures |
| [Building physics](applications/building_physics.md#against-openmodelica-modelica-buildings-library), dynamic | Modelica Buildings Library v13.0.0, OpenModelica 1.27.1, DASSL at 1e-13 | Stored output (OpenModelica CSV, committed) | 15 models with compressible volumes, plus `PressurizationData` and `TrickleVent` imported but not compared (see the building physics page's limitations) | Flows, temperatures, pressures, water and trace-substance fractions, feedback controller outputs, every row from `StartTime` | 1e-6, beyond the reference's own resolution (`REFERENCE_RESOLUTION`: how far the reference moves between DASSL 1e-12 and 1e-13); floor 1e-3 of the largest value | Worst beyond resolution 7.1e-7 (controller output, `ChimneyShaftWithVolume`), 4.9e-7 for flows (`ClosedDoors`); raw worst 7.2e-5 (flows, `OneOpenDoor`, 0 beyond resolution); temperatures ≤ 7.1e-9 | The reference's resolution at flow reversals and switch points, where noodl physics' runs at 1, 2 and 4 substeps agree to 1e-10 kg/s |
| [Building physics](applications/building_physics.md#against-analytical-solutions), airflow closed forms | Closed form; `scipy.optimize.brentq` or `fsolve` where no closed form exists | Computed in the test | Series and parallel power laws; fan-driven zone; fan curve against a leak; three-zone stack. Each once alone and as a batch of 64 random instances | Flows, zone pressures, nodal residual | 1e-9, no absolute allowance, with Newton solved to 1e-12; nodal residual 1e-11 kg/s, ten times Newton's stopping bound | Single instances at round-off (≤ 8.2e-16); batches of 64 up to 9.7e-11 (stack), 1.6e-11 (fan-driven zone), 1.1e-11 (series) | Newton's stopping tolerance. At the default tolerance ($\sqrt{\epsilon}$, about 1.5e-8) the error reaches 1.7e-6 on small flows. That is the per-instance stopping rule, not batching: on the series case the batch and 64 separate solves stop at the same 9.8e-7. These check the solver and the element laws, not CONTAM itself |
| [Building physics](applications/building_physics.md#against-analytical-solutions), natural ventilation | Li and Delsante (2001) closed forms; `fsolve` for the two-zone case | Computed in the test | Single zone: buoyancy only, envelope loss, assisting wind, opposing wind (three steady states); two-zone doorway | Ventilation flow; zone temperatures | 1e-6 on flows (1e-4 on the three-root opposing-wind case); 1e-5 K abs on the two-zone temperatures | Flows ≤ 1.8e-10; two-zone temperatures 7.1e-10 K; the stable opposing-wind states are found by time stepping | None expected. Coupled steady states converge to the Newton tolerance |
| [Street air quality](applications/street_aq.md#munich-formulas), formulas (**under revision**) | MUNICH source code | Transcribed input/output pairs | 13 formula pairs: exchange constant and velocity, `bessel_shape_parameter` root, Macdonald $d_c$, $z_{0c}$, quadrature weight sums | Scalar outputs | At the precision MUNICH's source publishes: < 1e-15 to 1e-6 | All exact to the published precision | **Intentional:** MUNICH's weights are not normalised to 1, and noodl physics reproduces that. MUNICH's root is quantised to 0.01; `bessel_shape_parameter` solves it continuously (4e-4 in $u_M$, up to 2 % in $u_H$), and `preset="munich"` reproduces the grid with `shape_constant="grid_search"` |
| [Street air quality](applications/street_aq.md#munich-idealised-12-street-case), idealised network (**under revision**) | MUNICH, as published by Kim et al. (2022), Fig. 1 | Values transcribed from the paper | 12-street idealised network | Linearity in wind speed; 270° canyon-wind ratio; concentrations relative to one street (19 ratios) | < 1e-9 (linearity); the paper's value (ratio); 5 % target (pattern) | Linearity holds; ratio 1.99451, as in the paper; pattern **not met**: worst ratio 51 % off, 7 of 19 within 15 % | The paper does not give the street geometry, so the pattern is compared under a fitted uniform geometry. Open discrepancy |
| [Street air quality](applications/street_aq.md#sirane) (**under revision**) | SIRANE v2.1 rev 128 | Stored output (archived results for its South Kensington network) | 46 streets, 36 junctions, hours 00 and 01 of 7 January 2014; 10 single-street kernel probes | Roof exchange velocity; in-canyon wind; above-roof concentration from SIRANE's own fluxes; street concentration | Exchange velocity: SIRANE's printed half-step, 0.005 m/s abs. Wind: every street inside SIRANE's printing envelope (half-step + 3.6 %), plus median and 90th-percentile regression guards. Above-roof: regression guards | Exchange velocity worst 0.0025 m/s; wind median 4.5 %, 90th percentile 12.8 %; above-roof median 0.3 %, 90th percentile 1.4 %; street concentration not comparable (residual 0.888 recorded) | SIRANE prints two decimals. The above-roof check uses a fitted $u_*$ and $\theta_*$. The archive's emission field is not the deck's. The exchange's $\sigma_w$ is evaluated at SIRANE's canopy height `H_R` for every street, not at each street's height: SIRANE's backed-out $u_d$ follows `H_R` (10, 20, 30 m) and not the street heights, to within its printed digits |

## Sewer dynamic wave, fixture by fixture

Every row below is a live comparison with SWMM 5.2.4 through `pyswmm`
(`tests/verification/test_sewer_dynwave_parity.py`; fixtures in `tests/data/sewer/`), at every
routing step, with SWMM at `HEAD_TOLERANCE` 1e-12 m and `MAX_TRIALS` 500 and its report asserted
to say 0 % of steps not converging (except where a row says SWMM does not converge). noodl runs at `iterate_tol` 1e-13 m,
`iterate_max` 2000 and the default Anderson acceleration. The targets are depths 1e-7 m, flows 1e-6
relative plus 1e-9 m3/s, and overflows at the flow target; the measured column is the worst depth
difference. SWMM's default tolerance cannot be the reference: it stops centimetres short of its own
fixed point (see the [sewer page](applications/sewer.md#dynamic-wave-against-swmm)). The slow rows
run under `pytest -m slow`; each has a short fast case for the default run.

| Fixture and what it exercises | Steps | Measured worst depth | Notes |
|---|---|---|---|
| `tree_dynwave`, `backwater_dynwave`, `loop_dynwave`: filling tree, backwater line (drop, flat conduit, `FIXED` outfall), loop with an adverse conduit | 359 | 1.2e-9 / 4.5e-9 / 3.4e-9 m | Flows within target at every step |
| `tree_slot_dynwave`: `SURCHARGE_METHOD SLOT`, 2.5x inflow, C4 pressurised from step 95 | 359 | 8.3e-10 m | |
| `tree_extran_dynwave`: `EXTRAN`, 3x inflow, J2, J4 and J3 surcharged at steps 21, 139 and 270 | 359 | 1.6e-10 m | Head solve at `atol=1e-12`: a surcharged junction has no storage to damp a residual |
| `flooding_dynwave` (EXTRAN, J2 floods steps 21-27, J3 from 270), `flooding_slot_dynwave` (SLOT, J3 from 307) | 359 each | 1.2e-10 m; overflow 1.4e-11 m3/s | Overflow against `Node.flooding` |
| `flooding_outfall_dynwave`: J4 floods into its outfall conduit from step 236 | 359 | 1.2e-10 m; overflow 1.4e-11 m3/s | C5's flow, the outfall depth and J4's overflow on steps 236-258 are held to SWMM's onset error only: its stopping rule ignores C5 while J4 is clamped (3.2e-5 m3/s from its own fixed point) |
| Ramp-down of every inflow on `tree_dynwave` (held 10 min, ramped to nothing over 1 min, off for 10 min) | 252 (compared to 144) | 6.7e-10 m to SWMM's first non-convergence (step 145) | **Tie rule** (normal-flow slope switch). SWMM fails to converge on 82 of the 108 later steps of the ramp, and noodl is held to its oscillation: 5.7e-4 m, 3.6e-4 m3/s (bounds 1e-3 m, 5e-4 m3/s) |
| `tree_variable_dynwave`: `VARIABLE_STEP 0.75`, 30 min | 33 | 5.7e-10 m | Step sequence equal to the millisecond, including the clipped last step |
| the same with a ramping `[TIMESERIES]` inflow at J1 | 33 | 5.7e-10 m | The inflow read at the step start with SWMM's 1 ms clock |
| `ROUTING_STEP 120` with `WET_STEP 60` | 5 | step sequence only | The step is lowered to `WET_STEP` as SWMM does (equal to the millisecond) |
| `options_dynwave`: `InitDepth`, `InitFlow`, `MaxFlow`, outlet offsets, two barrels, flap gates (one reversed), a gated outfall, lengthening | 359 | 6.4e-9 m | The same fixture under `VARIABLE_STEP 0.75` (304 steps): step sequence equal to the millisecond, states at the targets |
| `options_dynwave` with a 0.03 m or 0.1 m offset on C3, or C6 without its gate (drop-end area threshold) | 359 each | 2.0e-9 m restarted from SWMM's state, SWMM-converged steps | **Tie rule** (area switch). SWMM fails on steps 60, 62, 167 / 107 / 106, 129; there noodl is within SWMM's oscillation (3.1e-4 m, 2.5e-5 m3/s) |
| `storage_dynwave`: J5 `CONICAL`, J3 `FUNCTIONAL`, J4 `TABULAR`, filling from dry | 359 | 7.6e-10 m (all four storage rows) | Depths, flows, overflow and volume against `Node.volume` |
| `storage_extran_dynwave`: J3 floods at its `MaxDepth`, J4 with `SurDepth` surcharges above its crown | 359 | within the 7.6e-10 m above | Same targets |
| the same under SLOT | 359 | within the 7.6e-10 m above | |
| the same, J4 `MaxDepth` 0.55 m above its crown (EXTRAN) | 359 | within the 7.6e-10 m above | SWMM's surcharged update uses the crown for its blend and floor |
| `structures_dynwave`: side orifice (weir to orifice regime), transverse weir with end contractions, V-notch weir, rated outlet into an outfall | 359 | 2.9e-9 m | Every link flow, regulators included |
| the same with the outfall stage between the junction invert and the weir crest (reversal, then Villemonte submergence for 200 steps) | 359 | 3.4e-9 m | A variant with the stage above the crest from the start is not compared: SWMM fails on 3 steps (0.84 %) |
| the same with the orifice's downstream junction filling through the orifice's crest | 359 | 4.8e-9 m | End-area jump at the crest, converged without an in-pass switch |
| `tidal_dynwave`: `TIDAL` outfall backing the outfall conduit up to 0.38 m of 0.45 m | 359 | 8.4e-10 m | The stage at the step end (SWMM's clock) |
| the same stage as a `TIMESERIES` outfall (driver) | 359 | 8.4e-10 m | |
| the tide on the variable-step tree | 33 | 8.4e-10 m | Step sequence equal to the millisecond |
| Conservative tracer on the tree (J1 ramps 0 to 100 mg/L), noodl driven with SWMM's J1 concentration | 3 h | steady 4.2e-11 relative | **By design** in transit: the reactors are junctions, SWMM's are conduits. Areas under the curves match the delay difference to 1.5e-11 / 5.7e-11 / 2.6e-10; the pointwise difference stays within the bound (25 % / 18 % / 9 % of the plateau against 57 % / 49 % / 61 %). SWMM's concentrations equal `qualrout.c` ported on SWMM's flows to 4.7e-16 kg/m3 |
| Conservative tracer on the loop, 100 mg/L at J1 and 40 mg/L at J3 | 3.5 h | steady 2.5e-10 relative | Slow test, about 11 min |

Not compared live (implemented and checked against closed forms): the cases listed under
[Limitations](applications/sewer.md#limitations).

## What runs by default

The default `pytest` run (`addopts` deselects `slow`) runs every comparison above. The
live-engine rows skip themselves when their engine is not installed. `wntr`, `pyswmm` and
`wsimod` come with the `dev` extra. `contamxpy` comes with the `contam` extra, which exists
only on Windows x86-64, so the ContamX rows are marked `external` and do not run on Linux CI.
The stored-output rows, the WSIMOD demo replay and every closed-form row need no external
engine.

Only the dynamic Modelica comparison is split. By default it compares the first few rows of
each of the 15 models (between 2 and 21 rows each). `pytest -m slow` compares every row and adds
a check that the time-integration error falls with the order of the scheme. The full runs take
between 21 s and 1,180 s per model; the timings are on the
[building physics page](applications/building_physics.md#against-openmodelica-modelica-buildings-library).
The measured values for every Modelica column are committed in
`tests/data/modelica/parity-algebraic.json` and `parity-dynamic.json`.

## Golden files are not verification

`tests/golden/*.json` hold noodl physics' own earlier output (for example the two-loop water
network, the sewer tree, the CONTAM airflow case and the MUNICH idealised case). Comparing
with them detects that a result has changed. It does not show that the result is right: a
golden file matches whatever the code computed when it was recorded, right or wrong.

## Intentional differences

Two differences from the reference are deliberate, and each is tested so that it stays the
only difference.

- **SWMM's one-step junction lag is not replicated.** SWMM mixes the inflow to a junction from
  the upstream conduits' concentrations at the *start* of the step (`findLinkMassFlow` reads
  `Link.oldQual`), so each junction delays the signal by one step. noodl physics solves the
  whole tree implicitly within the step. The difference is first order in the time step and
  disappears at steady state. The implicit solve is the transport operator every noodl physics
  application shares; reproducing the lag would need an explicit, upstream-ordered sweep
  specific to SWMM. The test asserts that SWMM equals the lagged recursion built on noodl
  physics' own volumes and flows, so the lag accounts for the whole difference.
- **WSIMOD's first-come sharing is not replicated.** When several arcs push into one node's
  limited headroom, WSIMOD serves them in call order: two 10-unit pushes into 12 units give
  `[10, 2]` or `[2, 10]`. The allocation layer shares by preference weight and ignores order,
  giving `[6, 6]`. Order-dependent allocation has no gradient with respect to the requests. The
  preference-weighted share is a smooth function of them, with genuine cross-gradients between
  competing arcs in projection mode. The totals and the storage agree with WSIMOD. The other
  known departures from WSIMOD (same-step outflow, bottlenecks behind a pass-through `Node`,
  node-level limits) are listed on the
  [flow allocation page](applications/allocation.md#where-the-layer-differs-from-wsimod).

The MUNICH quadrature weights and exchange constant are the opposite case: MUNICH's own
choices, reproduced on purpose even where they look like artefacts, because "fixing" them would
bias the results relative to MUNICH.
