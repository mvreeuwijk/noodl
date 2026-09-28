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
| [Sewers](applications/sewer.md#verification), hydraulics | SWMM 5.2.4 (engine version and zero routing error asserted) | Live engine, `pyswmm` | Kinematic-wave trees: 5, 13 (EPA *Example 1*, converted) and 32 conduits | Pipe flows; depths and volumes (live, float64); velocities (binary output, float32) | Flows 1e-9; depths and volumes 1e-10, three decades above a ~1e-13 rounding bound; velocities 1e-7, above float32's half-ulp of 5.96e-8 | Flows ≤ 6.2e-14; depths ≤ 1.7e-14; volumes ≤ 2.3e-14; velocities ≤ 5.8e-8 | None beyond rounding, with `geometry="swmm"` (SWMM's own tabulated circle). The exact circle differs by up to 5.2e-3 in depth and 0.18 in velocity |
| [Sewers](applications/sewer.md#verification), steady quality | SWMM 5.2.4, as above | Live engine | Decaying tracer on the 5-conduit tree | Link concentration: tank-in-series closed form, and the model's quality layer | 1e-9 | 4.5e-15 (closed form); 6.3e-13 (quality layer) | None. Both codes react, then mix, and report after mixing |
| [Sewers](applications/sewer.md#verification), transient quality | SWMM 5.2.4, as above | Live engine, sampled every 5 s routing step | Time-varying tracer load on the 5-conduit tree | Link concentration at every step | 1e-9 on the conduit fed by a lateral inflow; downstream conduits not asserted | Head conduit 1.1e-14; downstream 1.1e-2 and 1.7e-2 of peak at 5 s (2.2e-3 and 3.5e-3 at 1 s) | **Intentional:** SWMM's one-step junction lag (see below). SWMM equals the lagged recursion on noodl physics' own volumes and flows to 5e-15 |
| [Capacitated allocation](applications/capacitated.md#verification), shipped demos | WSIMOD 0.8.1 (pinned exactly) | Stored output: every per-arc request and realised flow captured from WSIMOD's own run | `quickstart_demo` (1,456 steps); `oxford_demo`, 20 of 21 arcs | Realised arc flows, hard clip | 1e-9 abs (quickstart); 1e-6 abs (oxford, flows up to about 1e6) | 1.1e-16; 1.9e-9 | Float64 noise. The demos never bind a capacity, so this checks only the pass-through path. `sewer_to_wwtw` is excluded: WSIMOD caps it at a node, not an arc |
| [Capacitated allocation](applications/capacitated.md#binding-capacities-and-headroom), binding capacities | WSIMOD 0.8.1 | Live engine, WSIMOD's own `Arc`, `Node`, `Storage`, `Waste` classes | 6 scripted networks with binding arcs and tanks; `quickstart_tight` (3 capacities lowered, 1,456 steps) | Realised volumes, storage | 1e-12 × case scale: both sides are float64 doing the same arithmetic up to association | ≤ 4.4e-16; 1.0e-16 | **Intentional** where arcs compete for one node's headroom: preference sharing, not first-come (see below). The totals and storage agree |
| [Building physics](applications/building_physics.md#against-contamx) | NIST ContamX 3.4.1.7 | Live engine, `contamxpy` (Windows x86-64 only) | Stack project over a ±20 K ambient sweep; constant-mass-flow fan; three-zone project, steady and 24 transient steps of 300 s | Path flow directions and magnitudes; zone concentrations | First-order budget of float32 input roundoffs, $u = 2^{-24}$: 17 to 33 $u$ on stack flows (1.0e-6 to 2.0e-6, growing as the temperature difference shrinks), 16 $u$ on three-zone flows, 64 $u$ on concentrations | Stack 7.2e-8 to 9.4e-8; three-zone flows 9.1e-8; concentrations 2.0e-7; flow signs match | One or two float32 roundoffs in ContamX's single-precision inputs |
| [Building physics](applications/building_physics.md#against-openmodelica-modelica-buildings-library), algebraic | Modelica Buildings Library v13.0.0, simulated by OpenModelica 1.27.1 | Stored output (OpenModelica CSV, committed) | 6 models without volumes; plus CONTAM's own validation table on `OneWayFlow` (104 entries) | Every row and column of mass flows | $\lvert\Delta\rvert \le$ 1e-6 $\lvert$ref$\rvert$ + 1e-9 kg/s; for the table, no more than MBL's own departure from it | ≤ 6.1e-16 on five models, 1.8e-11 on `OpenDoorTemperature`; table: 0.74 %, the same as MBL's own | `OpenDoorTemperature`: likely an OpenModelica nonlinear-solver residual (not diagnosed). The CONTAM table's 3 significant figures |
| [Building physics](applications/building_physics.md#against-openmodelica-modelica-buildings-library), dynamic | Modelica Buildings Library v13.0.0, OpenModelica 1.27.1, DASSL at 1e-13 | Stored output (OpenModelica CSV, committed) | 13 models with compressible volumes | Flows, temperatures, pressures, water and trace-substance fractions, every row from `StartTime` | 1e-6, beyond the reference's own resolution (`REFERENCE_RESOLUTION`: how far the reference moves between DASSL 1e-12 and 1e-13); floor 1e-3 of the largest value | Worst beyond resolution 4.7e-7 (flows, `ClosedDoors`); raw worst 7.2e-5 (flows, `OneOpenDoor`, 0 beyond resolution); temperatures ≤ 7.1e-9 | The reference's resolution at flow reversals and switch points, where noodl physics' runs at 1, 2 and 4 substeps agree to 1e-10 kg/s |
| [Building physics](applications/building_physics.md#against-analytical-solutions), airflow closed forms | Closed form; `scipy.optimize.brentq` or `fsolve` where no closed form exists | Computed in the test | Series and parallel power laws; fan-driven zone; fan curve against a leak; three-zone stack. Each once alone and as a batch of 64 random instances | Flows, zone pressures, nodal residual | 1e-6 (1e-5 for the batched stack); residual at the Newton tolerance | Within tolerance; the exact values are not recorded | None expected. These check the solver and the element laws, not CONTAM itself |
| [Building physics](applications/building_physics.md#against-analytical-solutions), natural ventilation | Li and Delsante (2001) closed forms; `fsolve` for the two-zone case | Computed in the test | Single zone: buoyancy only, envelope loss, assisting wind, opposing wind (three steady states); two-zone doorway | Ventilation flow; zone temperatures | 1e-6 on flows (1e-4 on the three-root opposing-wind case); 1e-5 K abs on the two-zone temperatures | Flows ≤ 1.8e-10; two-zone temperatures 7.1e-10 K; the stable opposing-wind states are found by time stepping | None expected. Coupled steady states converge to the Newton tolerance |
| [Street air quality](applications/street_aq.md#munich-formulas), formulas (**under revision**) | MUNICH source code | Transcribed input/output pairs | 13 formula pairs: exchange constant and velocity, `soulhac_shape` root, Macdonald $d_c$, $z_{0c}$, quadrature weight sums | Scalar outputs | At the precision MUNICH's source publishes: < 1e-15 to 1e-6 | All exact to the published precision | **Intentional:** MUNICH's weights are not normalised to 1, and noodl physics reproduces that. MUNICH's root is quantised to 0.01; noodl physics solves it continuously (4e-4 in $u_M$) |
| [Street air quality](applications/street_aq.md#munich-idealised-12-street-case), idealised network (**under revision**) | MUNICH, as published by Kim et al. (2022), Fig. 1 | Values transcribed from the paper | 12-street idealised network | Linearity in wind speed; 270° canyon-wind ratio; concentrations relative to one street (19 ratios) | < 1e-9 (linearity); the paper's value (ratio); 5 % target (pattern) | Linearity holds; ratio 1.99451, as in the paper; pattern **not met**: worst ratio 51 % off, 7 of 19 within 15 % | The paper does not give the street geometry, so the pattern is compared under a fitted uniform geometry. Open discrepancy |
| [Street air quality](applications/street_aq.md#sirane) (**under revision**) | SIRANE v2.1 rev 128 | Stored output (archived results for its South Kensington network) | 46 streets, 36 junctions, hours 00 and 01 of 7 January 2014; 10 single-street kernel probes | Roof exchange velocity; in-canyon wind; above-roof concentration from SIRANE's own fluxes; street concentration | Exchange velocity: SIRANE's printed half-step, 0.005 m/s abs. Wind: every street inside SIRANE's printing envelope (half-step + 3.6 %), plus median and 90th-percentile regression guards. Above-roof: regression guards | Exchange velocity worst 0.0025 m/s; wind median 4.5 %, 90th percentile 12.8 %; above-roof median 0.3 %, 90th percentile 1.4 %; street concentration not comparable (residual 0.888 recorded) | SIRANE prints two decimals. The above-roof check uses a fitted $u_*$ and $\theta_*$. The archive's emission field is not the deck's |

## What runs by default

The default `pytest` run (`addopts` deselects `slow`) runs every comparison above. The
live-engine rows skip themselves when their engine is not installed. `wntr`, `pyswmm` and
`wsimod` come with the `dev` extra. `contamxpy` comes with the `contam` extra, which exists
only on Windows x86-64, so the ContamX rows are marked `external` and do not run on Linux CI.
The stored-output rows, the WSIMOD demo replay and every closed-form row need no external
engine.

Only the dynamic Modelica comparison is split. By default it compares the first few rows of
each of the 13 models (between 2 and 21 rows each). `pytest -m slow` compares every row and adds
a check that the time-integration error falls with the order of the scheme. The full runs take
between 75 s and 8,112 s per model; the timings are on the
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
  `[10, 2]` or `[2, 10]`. The capacitated layer shares by preference weight and ignores order,
  giving `[6, 6]`. Order-dependent allocation has no gradient with respect to the requests. The
  preference-weighted share is a smooth function of them, with genuine cross-gradients between
  competing arcs in projection mode. The totals and the storage agree with WSIMOD. The other
  known departures from WSIMOD (same-step outflow, bottlenecks behind a pass-through `Node`,
  node-level limits) are listed on the
  [capacitated allocation page](applications/capacitated.md#where-the-layer-differs-from-wsimod).

The MUNICH quadrature weights and exchange constant are the opposite case: MUNICH's own
choices, reproduced on purpose even where they look like artefacts, because "fixing" them would
bias the results relative to MUNICH.
