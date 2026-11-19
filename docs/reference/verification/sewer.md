# Verification: sewers

Evidence for the [sewer application](../../applications/sewer/index.md) (its [limitations](../../applications/sewer/limitations.md) list what is not compared); the summary is on the [verification overview](index.md).

## Sewer dynamic wave, fixture by fixture

Every row below is a live comparison with SWMM 5.2.4 through `pyswmm`
(`tests/verification/test_sewer_dynwave_parity.py`; fixtures in `tests/data/sewer/`), at every
routing step, with SWMM at `HEAD_TOLERANCE` 1e-12 m and `MAX_TRIALS` 500 and its report asserted
to say 0 % of steps not converging (except where a row says SWMM does not converge). noodl runs
at `iterate_tol` 1e-13 m, `iterate_max` 2000 and the default Anderson acceleration. The targets
are depths 1e-7 m, flows 1e-6 relative plus 1e-9 m3/s, and overflows at the flow target; the
measured column is the worst depth difference. SWMM's default tolerance cannot be the reference:
it stops centimetres short of its own fixed point (see the [dynamic-wave
runs](#dynamic-wave-against-swmm-in-detail)). The slow rows run under
`pytest -m slow`; each has a short fast case for the default run.

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
| `shapes_dynwave`, `shapes_drop_dynwave` (offsets at both ends): 25 chains (every modelled SWMM shape: the `xsect.dat` tables, the closed forms, `CUSTOM`, `FILLED_CIRCULAR`, `FORCE_MAIN`), one chain each | 40 | 3.0e-9 / 1.2e-9 m | Flows within target at every step; `RECT_ROUND` omitted from the offset fixture (see the [sewer page](../../applications/sewer/dynamic-wave-structures.md#cross-sections-losses-transects-and-culverts)) |
| `shapes_surcharge_dynwave` (EXTRAN), `shapes_slot_dynwave` (SLOT): the closed shapes past their crowns under a `FIXED` stage | 60 | 7.2e-8 / 2.1e-8 m | |
| `shapes_us_dynwave`: CFS, standard ellipse and arch sizes by code, a rectangular and a trapezoidal channel | 40 | 1.6e-9 m | |
| `force_main_hw_dynwave`, `force_main_dw_dynwave`: full force mains, Hazen-Williams and Darcy-Weisbach | 60 | 1.8e-8 / 2.7e-8 m | |
| `losses_dynwave`: entry, exit and average loss coefficients | 60 | 5.6e-9 m | |
| `irregular_dynwave`: three `[TRANSECTS]` channels (overbank n, `Lfactor`, `Xfactor`, `Yfactor`) | 40 | 5.8e-10 m | |
| `culvert_dynwave`: six inlet-control types from unsubmerged to submerged | 60 | 1.1e-8 m | `tests/verification/test_sewer_shapes_parity.py` for these rows |

Not compared live (implemented and checked against closed forms): the cases listed under
[Limitations](../../applications/sewer/limitations.md).

## Kinematic wave against SWMM

Against SWMM 5.2.4 through `pyswmm`, with `geometry="tabulated"`, on three kinematic-wave networks
of circular conduits:

- `tree_kinwave.inp`, the hand-built 5-conduit tree.
- `example1_kinwave.inp`, the 13-conduit network of EPA SWMM's published *Example 1*. It is
  converted to metres, and its runoff is replaced by rational-method constant inflows (the
  runoff itself is compared separately, [below](../../applications/sewer/swmm-hydrology.md#subcatchment-runoff-from-a-swmm-file)).
- `tree_kinwave_large.inp`, a synthetic 32-conduit tree with fills from 8e-5 to 0.81 of `Qfull`.

`tests/data/sewer/make_kinwave_fixtures.py` regenerates the last two. The engine identity is
pinned: `engine_version == "5.2.4"` and `flow_routing_error == 0.0`.

| Check | Tolerance | Measured worst (5 / 13 / 32 conduits) | Exact circle, for comparison |
|---|---|---|---|
| Pipe flows | rel 1e-9 | 1.4e-14 / 3.1e-15 / 6.2e-14 | the same |
| Depths (live, float64) | rel 1e-10 | 7.5e-15 / 1.2e-15 / 1.7e-14 | 5.5e-4 / 1.8e-3 / 5.2e-3 |
| Volumes (live, float64) | rel 1e-10 | 8.6e-15 / 1.8e-15 / 2.3e-14 | 6.5e-4 / 1.7e-3 / 4.8e-3 |
| Velocities (binary output, float32) | rel 1e-7 | 3.7e-8 / 5.8e-8 / 5.3e-8 | 6.3e-4 / 2.5e-3 / 1.8e-1 |
| Tracer, tank-in-series closed form | rel 1e-9 | 4.5e-15 | 7.8e-6 |
| Tracer, the model's quality layer, steady | rel 1e-9 | 6.3e-13 | 3.5e-3 at dt = 60 s if sampled after the reaction |
| Tracer, time-varying load, every 5 s step, head conduit | rel 1e-9 | 1.1e-14 | |
| Tracer, same run, downstream conduits (SWMM's junction lag, see below) | not asserted | 1.1e-2 / 1.7e-2 of peak at 5 s | 2.2e-3 / 3.5e-3 at 1 s |

The tolerances are not fitted to these numbers. Depth and volume are read live, as float64, and
both sides run the same float64 operations. 1e-10 is three decades above a rounding bound of
about 1e-13. Velocity has no live accessor under KINWAVE. The binary output stores it as float32,
whose half-ulp is 5.96e-8.

The quality layer uses SWMM's scheme and reports where SWMM reports. Both codes split a step
into an explicit first-order decay $c \to c(1 - k\Delta t)$ followed by an implicit upwind mixing
over the conduit volume (SWMM `qualrout.c`: `getReactedQual`, then `getMixedQual`), and both
return the concentration after the mixing: `build_model` sets
`Model(reaction_order="before_transport")`. The fixed point satisfies
$c(q + kV) = \sum q_u c_u + \text{load}$ at any $\Delta t$, which is SWMM's own. Sampled after the
reaction instead (the framework's default order), the fixed point would be $(1 - k\Delta t)$
times SWMM's, 3.47e-3 at 60 s.

The transient check drives the model with the inflow concentration SWMM applied at each step
(its `J1` node quality) and compares every routing step. A conduit fed only by a lateral inflow
takes exactly SWMM's step. A conduit fed by upstream conduits does not: SWMM mixes a junction's
inflow from the upstream links' start-of-step concentrations (`findLinkMassFlow` reads
`Link.oldQual`), a one-step lag per junction, where the model solves the whole tree implicitly.
That difference is first order in $\Delta t$. The test asserts that SWMM equals exactly this
lagged recursion on the model's own volumes and flows (5e-15), so the lag is the whole
difference. Transient hydraulics are not compared: the model's water side is quasi-steady (or
manhole storage), not a kinematic wave. SWMM also mixes over the step-start volume,
$(cV_1 + c_{in} q_{in}\Delta t)/(V_1 + q_{in}\Delta t)$, where the model's amount form divides by
$V_2 + q_{out}\Delta t$; the two agree when $V_2 - V_1 = (q_{in} - q_{out})\Delta t$.

## Dynamic wave against SWMM, in detail

`tests/verification/test_sewer_dynwave_parity.py` runs live SWMM 5.2.4 through `pyswmm` with a
fixed 5 s routing step for 30 min and compares every step's node depths and link flows
(files `tests/data/sewer/{tree,backwater,loop}_dynwave.inp`):

- a 5-conduit tree, filling from dry;
- a backwater line, with a drop at J2, a flat conduit C2 and a `FIXED` outfall stage above the outfall invert;
- a loop with an adverse conduit.

SWMM is run with `HEAD_TOLERANCE` 1e-12 m and `MAX_TRIALS` 500; noodl with `iterate_tol` 1e-13 m
and `iterate_max` 2000. Each test asserts that SWMM's report says 0 % of steps not converging.
The two sides compare the same fixed point only if SWMM has reached it, and SWMM's trials are
relaxed, so a loose `HEAD_TOLERANCE` stops it short (second table). Tightening it to 1e-12 m
brought the differences below the targets without changing either code's equations.

| Check | Target | Measured worst (tree / backwater / loop) |
|---|---|---|
| Node depths, every 5 s step | abs 1e-7 m | 1.2e-9 / 4.5e-9 / 3.4e-9 m |
| Link flows, every 5 s step | rel 1e-6 + abs 1e-9 m³/s | within the target at every step |

The options are compared together on `options_dynwave.inp`: J1 starts at 0.25 m (`InitDepth`)
and C3 at 0.03 m³/s (`InitFlow`); C1 is limited to 0.012 m³/s (`MaxFlow`) and has a 0.05 m
outlet offset; C3 has two barrels; C4 has a flap gate that closes on the water backing up from
J4; C6 is adverse in the file (it is reversed, with its `InitFlow` negated, and the offsets
swap), has 0.003 m³/s of `InitFlow` and a flap gate that, being reversed, passes only the
file's J5-to-J3 direction, so the downhill flow is closed; the `FIXED` outfall (stage 9.3 m,
above J4's invert and C5's crown) is gated, so C5 carries nothing until J4 rises over the
stage; C2 is 10 m long and `LENGTHENING_STEP 5` lengthens it. The fixture is therefore not
free-surface throughout. SWMM converges on every step, and noodl meets the targets above:
worst depth 6.4e-9 m over the 359 steps. Under `VARIABLE_STEP 0.75` the same fixture's 304
steps match SWMM's step sequence to the millisecond (the Courant step reads one barrel's flow
and volume and the lengthened conduit's `modLength / length`) and its states to the same
targets.

Three variants of that fixture take a junction through a drop end's area threshold while it
fills (see [Drop-end area switch](../../applications/sewer/physics-dynamic-wave.md#drop-end-area-switch)): a 0.03 m or 0.1 m outlet offset on C3 (J4 fills behind
the closed outfall gate) and C6 without its flap gate (J5 fills). SWMM fails to converge on
steps 60, 62 and 167, on step 107, and on steps 106 and 129 respectively, found by cutting
the run after each step (SWMM floors a run's duration, so each cut run ends at 5n + 1 to
3 s). noodl converges on every step. Its own run meets the targets on every step before
SWMM's first failure; after it the two histories differ by what each did on that step
(SWMM's last trial against noodl's held solution), up to 3e-4 m stored behind the gate. Each
step is therefore also run from SWMM's own state before it: on every step SWMM converged
from a converged start the targets hold (worst 2.0e-9 m), and on SWMM's failing steps and
the steps after them noodl is within SWMM's own oscillation (at most 3.1e-4 m and
2.5e-5 m³/s; bounded at 1e-3 m and 5e-5 m³/s). The fixture itself keeps C1's offset, where
nothing fills through a threshold.

Surcharge is compared on the same tree at higher inflows. `tree_slot_dynwave.inp` (2.5x,
`SURCHARGE_METHOD SLOT`) pressurises C4 from step 95. `tree_extran_dynwave.inp` (3x,
`SURCHARGE_METHOD EXTRAN`) surcharges J2, J4 and J3 above their crowns at steps 21, 139 and
270; there noodl's head solve runs at `atol=1e-12` m3/s, because a surcharged junction has no
storage to damp a balance residual. SWMM converges on every step of both runs. Every step meets
the targets above: the worst depth difference is 8.3e-10 m (SLOT) and 1.6e-10 m (EXTRAN).
SWMM's EXTRAN iteration has no fixed point when a junction would fall back through its crown
while the storage equation keeps it above (see [Dynamic-wave physics](../../applications/sewer/physics-dynamic-wave.md#surcharge-slot-and-extran)). Such steps have no SWMM
reference, and none occurs in this fixture.

Flooding is compared on depths, flows and `sewer.overflow` against pyswmm's `Node.flooding`
(SWMM's overflow rate of the step), at the same targets (the overflow at the flow target):

- `flooding_dynwave.inp` (EXTRAN, 3x): J2, with `SurDepth` 1.7 m, floods at 2.0 m on steps
  21-27 and is then released; J3 floods from step 270.
- `flooding_slot_dynwave.inp` (the same junctions under SLOT): J3 floods from step 307.

SWMM converges on every step of both, and the worst differences are 1.2e-10 m and
1.4e-11 m³/s of overflow.

`flooding_outfall_dynwave.inp` floods J4, whose only outflow conduit C5 ends at the free
outfall, from step 236. There SWMM's answer is not converged for one quantity, although
SWMM reports 0 % of steps not converging:

- SWMM's stopping rule tests only non-outfall depths. J4 is clamped at `y_max`, so once J4
  is clamped on two trials in a row nothing tested depends on C5's flow.
- SWMM therefore stops while C5's relaxed flow is still 3.2e-5 m³/s from its fixed point.
  A reproduction of SWMM's trials at that step lands on SWMM's value to 1e-10 m³/s and
  converges to noodl's.
- That error decays over the next 22 steps.

The test therefore leaves out C5's flow, the outfall depth and J4's overflow on steps
236-258, where they are held only to the onset error's scale. Every other quantity on every
step meets the targets.

Storage units are compared on depths, flows, overflow and each unit's volume (against
pyswmm's `Node.volume`, at the depth target times the unit's area):

- `storage_dynwave.inp`: the tree with J5 `CONICAL`, J3 `FUNCTIONAL` and J4 `TABULAR`,
  filling from dry below every crown;
- `storage_extran_dynwave.inp` (EXTRAN, 3x): J3 (`MaxDepth` 0.4 m below its 0.45 m crown, no
  `SurDepth`) floods at 0.4 m, its full depth not extended; J4 (`MaxDepth` 0.3 m, `SurDepth`
  1.0 m) has its full depth extended to its 0.45 m crown and surcharges above it, its volume
  held at the full volume;
- the same under SLOT;
- the same under EXTRAN with J4's `MaxDepth` 0.55 m, above its crown: its full depth stays
  0.55 m and it surcharges only above that (to 0.84 m), where SWMM's surcharged update still
  uses the crown for its blend and its floor.

SWMM converges on every step of all four, and every step meets the targets (worst depth
7.6e-10 m). Without the storage area the first fixture misses by 1.2e-2 m within 40 steps,
and with the conduit end areas zeroed at the storage units by 0.17 m.

Regulators are compared on every node depth and every link flow, conduits and regulators
(`structures_dynwave.inp`): a side orifice that opens from dry and passes from its weir
regime to its orifice regime, a transverse weir with two end contractions, a V-notch weir,
and an outlet with a `RATING` curve from a storage unit into an outfall. A second variant
fixes the outfall stage between the junction's invert and the transverse weir's crest: the
outfall's backflow first overshoots the crest and the weir flows in reverse into its dry
upstream junction, then closes, and later runs submerged (Villemonte) for over 200 steps. SWMM converges on every step of
both, and every step meets the targets (worst depth 2.9e-9 m and 3.4e-9 m over 359 steps).
With the downstream stage above the crest from the start, water flows back over that weir
and its flow passes through zero under full submergence, where the flow's slope in the
heads is infinite: SWMM does not converge on 3 of those steps (0.84 %), so that variant is
not compared.
A third variant raises the orifice's downstream junction so that it fills through the
orifice's crest while the orifice flows: the orifice's end area there jumps from 0
(DN_CRITICAL) to half its equivalent area (~9 m²), a threshold like a drop end's. SWMM
converges on every step, and noodl converges without an in-pass switch for it and meets
the targets (worst depth 4.8e-9 m).

What is implemented but has no live SWMM comparison yet is listed under [Limitations](../../applications/sewer/limitations.md#what-is-not-supported).

Water quality is compared with a conservative tracer (a `[POLLUTANTS]` entry without
decay, written into copies of the fixtures; noodl takes the tracer as a driver):

- the tree, J1's lateral inflow carrying none for an hour while the hydraulics settle, then
  ramping to 100 mg/L over 10 min and holding it to 3 h; noodl is driven step by step with
  SWMM's own J1 concentration (J1 has only its lateral inflow, so that is the inflow
  concentration SWMM applied over the step);
- the loop, 100 mg/L entering at J1 and 40 mg/L at J3 from the start, for 3.5 h (its
  hydraulics settle slowly: at 2 h both codes are still 1e-6 from steady state). J3's water
  reaches J4 through the reversed conduit Cd.

SWMM converges on every step of both. In transit the codes differ by design (see
[Quality on the dynamic wave](../../applications/sewer/headspace-sulfide.md#quality-on-the-dynamic-wave)), so the transit checks test
SWMM against its own scheme and the difference against what the two schemes predict. At
steady flow both are linear chains with the same gain $g$, so the area between a conduit's
concentration and its final value is $g\,u\,T$, $T$ the chain's mean delay: SWMM's
$\sum V_{\text{conduit}}/q$ plus one step per junction passed, noodl's
$\sum V_j/q_{\text{out}}$. For step responses rising monotonically to $g$, the pointwise
difference is at most $\max|\Delta u|\,g\,(T_{\text{noodl}} + T_{\text{SWMM}})/\Delta t$.

| Check | Target | Measured worst |
|---|---|---|
| Tracer at steady state, every conduit (tree / loop) | rel 1e-8 | 4.2e-11 / 2.5e-10 |
| SWMM's tree concentrations against `qualrout.c` ported on SWMM's own flows and volumes, every step | abs 1e-12 of the plateau | 4.7e-16 kg/m³ |
| Area between SWMM's and noodl's concentration in transit (C1 / C3 / C5) against $g\,u\,(T_{\text{noodl}} - T_{\text{SWMM}})$ | rel 1e-6 | 1.5e-11 / 5.7e-11 / 2.6e-10 |
| Pointwise difference in transit (C1 / C3 / C5), of each conduit's plateau | the bound above: 57 % / 49 % / 61 % | 25 % / 18 % / 9 % |

The mean delays to C1 / C3 / C5 are $T_{\text{SWMM}}$ = 259 / 456 / 628 s and
$T_{\text{noodl}}$ = 82 / 306 / 542 s. The quality runs are slow tests (the loop's takes
about 11 min).

The same networks with SWMM at its **default** tolerance (`HEAD_TOLERANCE` 0.0015 m,
`MAX_TRIALS` 8), against noodl's converged answer:

| SWMM at default tolerance | Worst depth difference (m) | Worst flow difference (% of the network's maximum flow) |
|---|---|---|
| tree | 2.3e-3 | 0.8 |
| backwater | 1.3e-2 | 7.4 |
| loop | 3.6e-3 | 3.6 |

SWMM reports 0 % of steps not converging at those defaults too. Its stopping rule bounds the
change between relaxed trials, not the distance to the fixed point, so a result that SWMM
calls converged can sit centimetres away from the equations' solution on a backwater.

**The SWMM corpus.** `benchmarks/swmm_corpus.py` runs every file of a folder of SWMM models
through both engines for a short window and compares the node depths at its end (the corpus
is an argument, never part of the repository). On the 1,646 files of the SWMMEnablement
collection, 107 run in noodl (the others are refused by name, mostly for hydrology, pumps and
non-circular shapes), with a 600 s window. At each file's own `HEAD_TOLERANCE` and
`MAX_TRIALS` the largest depth difference per file is under 1 mm on 12 files, 1-10 mm on 22,
1-5 cm on 34, 5-20 cm on 29, 0.2-1 m on 8 and over 1 m on 2. That spread is SWMM's own
iteration error, not a difference of model: run four ways (each engine at the file's
tolerances and at `HEAD_TOLERANCE` 1e-12, `MAX_TRIALS` 500) on the 87 distinct files, noodl at
the file's tolerances is within 1 mm of its own converged answer on 58 of them and within
1 cm on 71, while SWMM at the file's tolerances is 1 cm or more from its own converged answer
on 57; on 54 of the 60 files that differ by 1 cm or more, SWMM's own error is at least half
the difference (`95_Linear_H_Elements`, 3.18 m between the engines, is 3.18 m between SWMM's
two runs and 2 mm between noodl's). With `--tight`, both engines converge each step:

| Max depth difference | < 1 mm | 1-10 mm | 1-5 cm | 5-20 cm | 0.2-1 m | > 1 m |
|---|---|---|---|---|---|---|
| file's tolerances (107 files) | 12 | 22 | 34 | 29 | 8 | 2 |
| `--tight` (107 files) | 79 | 9 | 7 | 5 | 6 | 1 |
| `--tight`, SWMM converging on every step (37 files) | 37 | 0 | 0 | 0 | 0 | 0 |

Every file whose tight difference exceeds 1 mm has steps on which SWMM itself does not
converge in 500 trials (its report's "% of Steps Not Converging", which the harness records
with noodl's own count): there SWMM's trials alternate around a switch with no fixed point
(the tie rules under [Intentional differences](../../applications/sewer/physics-dynamic-wave.md#intentional-differences-from-swmm)), and the
histories differ from SWMM's first such step on. Up to that step, at the instants both engines
share (a variable step is the same sequence only while the states agree), they agree to below
0.6 mm on every file but two: `2079_Nodes`, whose 5 mm conduits let noodl's passes land on a
second (surcharged) fixed point of the first step, and `STEEP_drop_example_2`, 2.7 cm on a
step where noodl's own iteration does not converge either. No option the reader accepts was
found ignored.

## Checks for physics SWMM does not model

The headspace air and the sulfide chemistry have no SWMM reference. They are checked against
closed forms, balances and one laboratory data set:

| Check | Result |
|---|---|
| Pescod & Price Tests 7–9, air-to-water velocity ratio in a 300 mm UPVC pipe, 15 m, open at both ends (20–40 % band) | 24.14 %, 25.00 %, 25.15 % against measured 35 %, 25 %, 27.5 % (Test 8 calibrates `f_i`); each inside the band and equal to the closed form to round-off (asserted at rel 1e-6) |
| Pescod & Price Tyneside ventilation band (105–315 m³/h) | bracketed; open-both-ends gives 1253.78 m³/h, a 17.27% velocity ratio, inside their 5–30% envelope |
| Fan draws exactly through the leaks | balance 1.234e-14, nodal residual 6.3e-15, power residual 8.1e-13 |
| Transfer-dominated state reaches Henry equilibrium | rel 1e-10, **measured 1.4e-16** |
| Air-layer nodal residual, and Tellegen power balance | 4.518e-13 m3/s, 6.141e-12 W |
| Cross-phase sulfide conservation | rtol 1e-12 |
| Adjoint gradients vs Richardson-extrapolated central differences | rel 1e-6, holds |
| Storage sweep vs the closed-form steady state | **5.7e-15** on flows, **4.4e-15** on depths |
