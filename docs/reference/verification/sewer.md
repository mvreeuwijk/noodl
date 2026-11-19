# Verification: sewers

Evidence for the [sewer application](../../applications/sewer/index.md); the summary is on the [verification overview](index.md).

## Sewer dynamic wave, fixture by fixture

Every row below is a live comparison with SWMM 5.2.4 through `pyswmm`
(`tests/verification/test_sewer_dynwave_parity.py`; fixtures in `tests/data/sewer/`), at every
routing step, with SWMM at `HEAD_TOLERANCE` 1e-12 m and `MAX_TRIALS` 500 and its report asserted
to say 0 % of steps not converging (except where a row says SWMM does not converge). noodl runs
at `iterate_tol` 1e-13 m, `iterate_max` 2000 and the default Anderson acceleration. The targets
are depths 1e-7 m, flows 1e-6 relative plus 1e-9 m3/s, and overflows at the flow target; the
measured column is the worst depth difference. SWMM's default tolerance cannot be the reference:
it stops centimetres short of its own fixed point (see the [sewer
page](../../applications/sewer/index.md#dynamic-wave-against-swmm)). The slow rows run under
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
| `shapes_dynwave`, `shapes_drop_dynwave` (offsets at both ends): 25 chains (every modelled SWMM shape: the `xsect.dat` tables, the closed forms, `CUSTOM`, `FILLED_CIRCULAR`, `FORCE_MAIN`), one chain each | 40 | 3.0e-9 / 1.2e-9 m | Flows within target at every step; `RECT_ROUND` omitted from the offset fixture (see the [sewer page](../../applications/sewer/index.md#cross-sections-losses-transects-and-culverts)) |
| `shapes_surcharge_dynwave` (EXTRAN), `shapes_slot_dynwave` (SLOT): the closed shapes past their crowns under a `FIXED` stage | 60 | 7.2e-8 / 2.1e-8 m | |
| `shapes_us_dynwave`: CFS, standard ellipse and arch sizes by code, a rectangular and a trapezoidal channel | 40 | 1.6e-9 m | |
| `force_main_hw_dynwave`, `force_main_dw_dynwave`: full force mains, Hazen-Williams and Darcy-Weisbach | 60 | 1.8e-8 / 2.7e-8 m | |
| `losses_dynwave`: entry, exit and average loss coefficients | 60 | 5.6e-9 m | |
| `irregular_dynwave`: three `[TRANSECTS]` channels (overbank n, `Lfactor`, `Xfactor`, `Yfactor`) | 40 | 5.8e-10 m | |
| `culvert_dynwave`: six inlet-control types from unsubmerged to submerged | 60 | 1.1e-8 m | `tests/verification/test_sewer_shapes_parity.py` for these rows |

Not compared live (implemented and checked against closed forms): the cases listed under
[Limitations](../../applications/sewer/index.md#limitations).
