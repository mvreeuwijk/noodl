# Verification: water resources

Evidence for [flow allocation](../../applications/water-resources/allocation.md) and the
[WSIMOD port](../../applications/water-resources/wsimod.md); the summary is on the
[verification overview](index.md). Both are checked against WSIMOD 0.8.1 (Dobson, Liu and
Mijic), pinned exactly rather than with `>=`, because the committed fixtures freeze one specific
WSIMOD run and a newer release could silently change the demos' forcing or node parameters.
WSIMOD is the reference implementation here, as EPANET and SWMM are for the water and sewer
applications, not an independent measurement. WSIMOD's source files are not modified; the
committed forcing is WSIMOD's (BSD-3 licence, kept beside the data).

## Flow allocation against WSIMOD

The harness replaces `Arc.send_push_request` and `send_pull_request` to capture every per-arc
requested and realised pair while WSIMOD runs its own `quickstart_demo` and `oxford_demo`. The
captured fixtures are committed, so these tests run without WSIMOD installed. Tests:
`tests/verification/test_allocation_wsimod_replay.py`.

| Check | Tolerance | Measured |
|---|---|---|
| Hard clip vs WSIMOD's realised flows, `quickstart_demo`, all 1,456 steps | 1e-9 abs | **1.11e-16** |
| Same, `oxford_demo`, 20 of 21 arcs | 1e-6 abs | **1.86e-9** |
| `"smooth"` at $\tau=10^{-3}$ vs the hard clip's own output on `quickstart_demo` | 3e-3 abs | **1.79e-3** |
| `"projection"` vs the hard clip's own output on `quickstart_demo` | 1e-9 abs | **9.10e-13** |
| Conservation, both demos, all three modes | exact | holds, no reference needed |
| `gradcheck` through smooth and projection on the diamond | analytic | holds |
| `n_passes=5` vs a hand-converged reference | exact | holds |
| Scripted WSIMOD networks with binding arcs and tanks (6 cases, `dt` = 86400 s): realised volumes and storage | 1e-12 × case scale | **≤ 4.4e-16** |
| `quickstart_tight` (three arc capacities lowered), all 6 arcs, 1,456 steps | 1e-12 × 0.113 | **1.04e-16** |

The 1.79e-3 discrepancy of smooth mode is the softplus-at-zero bias, not capacity smoothing; see
the [physics page](../../applications/water-resources/allocation-physics.md).

### Binding capacities and headroom

The two shipped demos never bind: all but one of their 27 arcs sit at WSIMOD's
`UNBOUNDED_CAPACITY` of 1e15, the one finite arc (`abstraction_to_farmoor`, 50000) never sees a
request above about 30,934, and the replay sets $s_{\max} = \infty$. On their own they check only
the identity path, $\min(x, c) = x$.

`tests/verification/test_allocation_wsimod_capacity.py` covers the binding branches against
WSIMOD's own numbers. It runs small networks built from WSIMOD's own `Arc`, `Node`, `Storage` and
`Waste` classes, driven by WSIMOD's `push_distributed` and `pull_distributed`, and checks from
WSIMOD's output that each targeted bound really cut a request before it compares:

- a push clipped by arc capacity, with one or several requests in a timestep;
- a push and a pull on the same arc in one timestep, sharing its capacity;
- a tank whose headroom and inflow-arc capacity bind in turn as its storage evolves;
- WSIMOD's preference-weighted `push_distributed` and `pull_distributed`, where a round's request
  exceeds an arc's capacity and WSIMOD's own clip cuts it.

It also replays `quickstart_tight`, which is `quickstart_demo` with `runoff`, `baseflow` and
`percolation` lowered to 0.03, 0.04 and 0.08. WSIMOD cuts `baseflow` to its capacity on 1,085
days. `runoff` and `percolation` run at capacity on 866 and 552 days, but WSIMOD's `Land` node
sizes those requests from the arc's own check, so there request = capacity.

WSIMOD clips each request against `capacity - flow_in`, and `flow_in` accumulates every push and
pull already realised on the arc in that timestep. The split between requests therefore depends
on their order, but the timestep total is $\min(\sum_k r_k, c)$ in any order, and that total is
what one layer step computes from the summed request.

### Differences that are pinned

These differences from WSIMOD are tested, not tolerated; they are described on the
[allocation page](../../applications/water-resources/allocation.md#where-the-layer-differs-from-wsimod):
shared headroom served by preference rather than call order; same-step outflow; bottlenecks behind
a pass-through node; node-level limits (which is why `sewer_to_wwtw` is left out of the
`oxford_demo` comparison: it mismatches on 185 of 1,456 steps); and queue arcs, forced pushes and
pollutants, which are not compared.

### Throughput

| Run | Observation |
|---|---|
| `oxford_demo` topology (21 arcs, 18 nodes), hard clip, 1,456 steps | batch 1, 10 and 100 cost about the same: the batch widens tensors |

The benchmark carries its own caveat: with $s_{\max} = \infty$ everywhere no node is ever
oversubscribed, the sharing branch never runs and every pass after the first is a no-op. It
measures the cheapest path only, and no budget is asserted on it.

## WSIMOD port: whole model and per-node replay

WSIMOD runs live (`pytest.importorskip("wsimod")`; the package never imports WSIMOD) in four
layers. Every comparison is bitwise where the arithmetic is `+ - * /`, `min`, `max` and
comparisons: the ceiling is `1e-12 * scale` per compared series (the largest magnitude over the
run), but the tests assert the class's own bound, which is 0, so a last-bit difference has to be
debugged, not admitted.

- **Per-node replay** (`tests/verification/test_wsimod_replay.py` and the class tests under
  `tests/apps/wsimod`). WSIMOD runs a case with every call into the chosen nodes captured; noodl
  replays each node alone on those calls and compares every arc call it makes, every reply and
  every store, in both modes. On the two demos every node class (Land, Sewer, Groundwater, Node,
  Waste, WWTW, FWTW, Reservoir, ResidentialDemand, Catchment) has worst difference exactly 0,
  with both IHACRES outcomes sampled.
- **Whole model** (`tests/verification/test_wsimod_parity.py`). WSIMOD's two demos, `quickstart`
  and `oxford`, each in two modes (volumes only, and every pollutant), against WSIMOD's own
  `Model.run`, and the purpose-built cases: every feature the demos skip (queue arcs, a leaky
  `Distribution` and the reservoir spill, sewers with time-areas and foul sewers, storm tanks,
  an `FWTW` deficit, an evaporation clamp, a porous `QueueGroundwater`, rivers and the River
  biochemistry, decays, treatment, a `float_accuracy` override and others). There are 69 cases:
  61 run against WSIMOD, 60 of them in both modes and `growing_lean_soil` in pollutant mode
  only; eight are noodl-only (six crash cases, `wtw_zero_input` and `nonresidential_demand`).
  Every arc's flow and pollutants, every recorded tank and every surface field are compared on
  every day. The worst difference is exactly 0 for every node of both demos and for all 61 cases
  in each of their modes. WSIMOD's printed warnings are compared with noodl's events day by day
  (dated prints) or per day and kind (undated ones); WSIMOD's own "mass balance error" lines
  equal noodl's residual plus the dropped water, with one named exception: in
  `river_double_discharge` WSIMOD's River balance misses the first of two daily
  `calculate_discharge` calls and prints on every day, while noodl's balance closes within
  rounding (2e-16 of scale). The system mass balance closes within `1e-12 * scale` every day,
  and every day the residual equals the evaporation clamp's gap (0 without a clamp). A
  quantity-only run's volumes equal the pollutant run's exactly.
- **The growing surface with its nutrient pool.** `growing_spring` (sown on day 90, harvested on
  250, fertiliser, manure, deposition and an `initial_soil_storage`), `growing_autumn` (sown on
  280, harvested on 200: the unsorted calendar and the autumn uptake), `growing_decay` (the 12
  default pollutants and `decays` on ammonia and nitrite) and `growing_lean_soil` (a P-free soil
  that starts below its wilting point beside a P-rich one) agree with WSIMOD bitwise on every
  compared series, and their land replays are bitwise, the pool, its two factors, the crop factor
  and the `et0_coefficient` included. Recorded branch decisions show every site of the
  transformation, uptake, erosion, denitrification, adsorption and the two factors taking both
  outcomes, except eight outcomes that `growing_spring` never shows (the wilted soil, a positive
  dissolved-P increase, the empty pool, the shortcut, no change, desorption, no available water
  and no eroded P), which `growing_lean_soil` shows. `deposition_non_additive` runs the
  deposition surfaces with `ammonia` non-additive: bitwise, no prints.
- **Irrigation, garden and variable area.** `irrigation` and `irrigation_early` (120 days): two
  `IrrigationSurface`s sown on run days 16 and 36 and fed by a `River`, a `Groundwater` and a
  `Reservoir` through the `Land`'s in-arcs, with eight pollutants; soils start above capacity, so
  the days before sowing, the sowing day, dry days with a demand and saturated days all occur
  (asserted from the decisions); `irrigation_early` runs `apply_irrigation` before `Land.run`.
  `garden` (100 days): a `ResidentialDemand` fed through a `Distribution` asks two `Land`s for
  garden water; one has two `GardenSurface`s and a `PerviousSurface`, the other no garden
  surface and answers 0. `variable_area` (100 days): two `VariableAreaSurface`s, `idle` at the
  default area of 0 and `half` at half its area. All are bitwise equal to WSIMOD in both modes,
  and the `Land` replays, handler calls included, are bitwise.
- **Crashes** (`test_case_runs_where_wsimod_crashes*`). Eleven cases where WSIMOD raises. Three
  run in noodl, finite and balanced (`land_sewer_ponding` driven by a capacity,
  `wtw_zero_input`, `nonresidential_demand`). For a decaying `Groundwater` and
  `QueueGroundwater` without a `filename` override, noodl refuses by name where WSIMOD raises
  `TypeError`; with the override both run and agree bitwise. Five are growing surfaces: WSIMOD
  raises `ZeroDivisionError` at a pinned line on a pinned day; noodl equals it bitwise on the
  days before, records the guard's event on the crash day, and runs on with finite values and a
  closed balance. The triggers are ordinary: `growing_default_depletion` (day 11) and, from a
  tank without inorganic N and zero first-month inputs, `growing_no_din`, `growing_zero_n`,
  `growing_nitrite_only` (day 0) and `growing_empty_soil` (day 1). A sixth,
  `irrigation_without_nutrients` (`KeyError` on day 5), runs a surface with the pollutants
  `phosphate` and `temperature` and no event: noodl updates no pool.

### Whole-model and case table

| Run | Days | Modes | Worst difference | Default |
| --- | --- | --- | --- | --- |
| quickstart | 1456 | quantity-only, pollutant mode | 0 | yes |
| oxford | 365 | quantity-only, pollutant mode | 0 | yes |
| oxford | 1456 | quantity-only, pollutant mode | 0 | slow |
| 47 cases (decay, treatment, soil-temperature, surface and earlier structural cases) | 3 to 40 | quantity-only, all pollutants | 0 | yes |
| `river_nutrients` | 400 | quantity-only, all pollutants | 0 | yes |
| `river_nutrients` | 1100, with decisions | all pollutants | 0 | slow |
| `river_double_discharge` | 40 | quantity-only, all pollutants | 0 (WSIMOD's own balance prints on every day; noodl's closes within 2e-16 of scale) | yes |
| `growing_spring` | 365 | quantity-only, the eight nutrient pollutants | 0 (WSIMOD's round-off prints admitted) | yes |
| `growing_spring` | 1096 | quantity-only, the eight nutrient pollutants | 0 (as above) | slow |
| `growing_spring` branches | 1096, with decisions, and `growing_lean_soil` | the eight nutrient pollutants | every site both ways | slow |
| `growing_autumn` | 300 (400 slow) | quantity-only, the eight nutrient pollutants | 0 (as above) | yes |
| `growing_decay` | 100 (200 slow) | quantity-only, the 12 default pollutants | 0 (as above) | yes |
| `growing_lean_soil` | 60 | the eight nutrient pollutants | 0 (as above) | yes |
| `surface_deposition` | 180 (450 slow) | quantity-only, all pollutants | 0 | yes |
| `growing_no_nitrate` | 365 (540 slow) | quantity-only, all pollutants | 0 | yes |
| five growing crash cases | 10 to 20 | the pollutants of each | 0 on the days before WSIMOD's crash | yes |
| `deposition_non_additive` | 90 | quantity-only, all pollutants | 0 | yes |
| `irrigation`, `irrigation_early` | 120 | quantity-only, the eight nutrient pollutants | 0 (no print admission needed) | yes |
| `garden` | 100 | quantity-only, the eight nutrient pollutants | 0 | yes |
| `variable_area` | 100 | quantity-only, the eight nutrient pollutants | 0 (three round-off lines within the store bound) | yes |
| `surface_overrides` | 100 | quantity-only, pollutant mode | 0 | yes |
| `irrigation_without_nutrients` | 40 | pollutants `phosphate` and `temperature` | noodl only (WSIMOD raises on day 5) | yes |
| 81 gradients | 1 to 200 | quantity-only values, pollutant runs | 5.3e-7 relative | 35 of 81 (46 slow) |

The quantity-only mode runs `config.yml` with `pollutants=False` and compares the volumes; the
pollutant mode runs the same `config.yml` with every pollutant and compares all of them. Neither
demo has a `River` node, so WSIMOD's river-order ties (set-hash order of the names, which varies
with `PYTHONHASHSEED`) cannot arise in the demos, and neither has a `QueueGroundwater`; the
river and groundwater cases cover those classes, with the ties kept independent.

### River biochemistry cases

`river_nutrients` is two rivers in a chain fed by catchments with all six pollutants, one river
overridden to `max_temp_lag = 5`, `max_phosphorus_lag = 300` and raised rates, and a reservoir
abstracting from the upstream river. Its forcing places every branch inside its first 400 days:
a frost spell, days between 0 and 5 °C, a drought in which the abstraction drains the upstream
river to exactly 0 (the dry guard), a DIN-free refill with a cold snap (mineralisation with no
dissolved nitrogen), a phosphate pulse that carries the low-P river's 365-day mean across the
`totalphosfcn` guard, warming and cooling seasons, and every 50 % cap binding on some days and
not on others. The default selection runs its first 400 days at value parity, bitwise; the slow
selection runs all 1100 days (three turns of the 366-slot register) and asserts each river's
branches from the recorded decisions. `river_double_discharge` calls `calculate_discharge` twice a
day: values are at parity, WSIMOD prints its River mass-balance error on every day and noodl's
balance closes within rounding.

### The one print admission

On a land with a growing surface and `nitrate` among the pollutants, WSIMOD's node check can
print round-off lines "mass balance error for <pollutant> of x in <land>" (`org-phosphorus` or
`org-nitrogen`, |x| about 1e-17 kg) when a tank holds a large organic store and the day's flows
are small: its change in storage is a difference of two large stores and its normaliser is the
flow scale. noodl's own balance closes. Whether a line falls on a given day depends on the last
bit of WSIMOD's own `math.exp`, `**` and `log`, so on the platform's C library. The admission is
therefore made at every such land of every case, the crash cases' days before their crash
included. The comparison helper admits such a line of an additive pollutant where
`|x| <= c * eps * S`, with `eps = 2.22e-16` and `S` the largest store of that pollutant on the
land's surfaces and its three residence tanks at the end of that day or the day before; a line
beyond it, a volume line and any line of another land still fail. Measured on Windows, the
maximum of `|x| / (eps S)` is 1.277 (`growing_spring`, day 908 of its three years; 0.432 and 43
lines in its first 365 days), 0.889 (`variable_area`, 3 lines), 0.610 (`growing_lean_soil`),
0.479 (`growing_decay`), 0.404 (`growing_autumn`, 400 days) and 0.170 (`surface_overrides`, 1
line); `irrigation`, `irrigation_early` and `garden` print none. The bound constant is therefore
2; a platform whose maximum is larger raises it to its ceiling.

## WSIMOD gradients

`tests/verification/test_wsimod_gradients.py`: 81 entries, central differences at a relative step
of 1e-6, with the two perturbed runs' decision outcomes identical. The worst relative
disagreement is 5.3e-7 (`GrowingSurface.kfr` at 20; then `nhx-fertiliser` 4.5e-7, `nfr` 3.0e-7,
`uptake1` 1.4e-7 and, among the river entries, `River.T_wdays [nitrate]` 1.5e-7 and
`River.halfsatINwater [nitrate]` 1.0e-7) against a tolerance of 1e-6. The irrigation and garden
entries are within 2.5e-8 (`IrrigationSurface.irrigation_coefficient`,
`VariableAreaSurface.current_surface_area`, `ResidentialDemand.gardening_efficiency`); the decay
entries within 9e-10, the treatment ones within 1.2e-10, the soil-temperature weights within
1.7e-8, and `Sewer.pipe_timearea` at 5.9e-8. The tests move a point off a kink where the
default one lies on it (the preference 1.0 to 1.37, the surface residence time 1 to 3), and say
so. Of the 81 entries, 46 run only under `-m slow` (long windows of two rivers or of growing
surfaces); each family keeps its cheapest entry in the default selection, and a mutation of the
Python-power or window-sum primitives fails a default entry.

## What runs by default

Tests marked `slow` are deselected by default and selected with `-m slow`. The default selection
of the WSIMOD tests is quickstart in full (1456 days, both modes), oxford for its first 365 days
(both modes), the sampled replays, the default gradients and every case over its default days.
`river_nutrients` runs 400 of its 1100 days, which already turns its 366-slot register over and
holds every branch; the growing cases run a prefix (`growing_spring` 365 of 1096 days,
`growing_autumn` 300 of 400, `growing_decay` 100 of 200, `surface_deposition` 180 of 450,
`growing_no_nitrate` 365 of 540); the bookkeeping checks run the long cases on a shorter prefix.
The quantity-only volume comparison runs by default only on the three cases WSIMOD does not run.
`pytest -m slow` adds the full horizons, the full `oxford` run, the full per-node replays, the
slow gradients and the other quantity-only comparisons.
