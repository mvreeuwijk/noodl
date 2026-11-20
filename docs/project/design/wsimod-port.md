# WSIMOD port: decision log

For maintainers of `noodl.apps.wsimod`. The user documentation is
[Running a WSIMOD model](../../applications/water-resources/wsimod.md),
[What is ported](../../applications/water-resources/wsimod-ported.md) and
[Differences from WSIMOD](../../applications/water-resources/differences.md). This page keeps
what those omit: the numbered decisions, the upstream source locations that justify them, and the
admitted round-off cases. Source locations are for WSIMOD 0.8.1 (`wsimod/...`). Codes are the
identifiers used in source comments, tests and docstrings: `L<n>` upstream items (L1 to L26 are the growing-surface set), `X<n>`
transcription decisions, `D<n>` design and test-harness decisions, `Y<n>` review items and `W<n>`
earlier workstream decisions.

## Principles

- The day is transcribed: same calls, same order, same arithmetic, so results are bitwise
  WSIMOD's. Python's `**`, `exp`, `sqrt` and `sum` are evaluated per instance (`pow_python`,
  `exp_python`, `pow_half`, `window_sum`), which costs time proportional to the batch (W8).
- Branches are decided on detached values per instance; values stay in the graph (W9).
- At a zero denominator a guard gives a finite value and a named event; other upstream
  behaviour is reproduced; a configuration that cannot run is refused.
- Every refusal names the key, node or date. Every comparison asserts the class's own bound,
  which is 0 for `+ - * /`, `min`, `max` arithmetic.

## Storage-family decay

- `DecayTank` replaces `Tank` when `decays` is given (`storage.py:53-59`); the end-of-day decay
  is `tanks.py:442-447` over `core.py:474-521`. The cap `min(factor, 1)` is the decision
  `node:<name>/tank/make_decay/cap:<pollutant>`, bound at the day's bind (X8 for the balance
  window).
- `core.py:484-487`: an empty `decays` dict still reads the temperature.
- `Groundwater` and `QueueGroundwater` lose the constructor's forcing (`storage.py:154`,
  `:202-203`, `:291-292`, `nodes.py:26`): a decaying one raises `TypeError` at its first end of
  day. Decision X11: list the configured forcing as unread; require a `filename` override.
- `decays` override on a node built with `decays: None` raises `ValueError` upstream
  (`storage.py:96-100`): refused.
- `record_all` lists only tanks whose class is exactly `Tank`, `QueueTank` or `ResidenceTank`
  (`model.py:1123`): decaying tanks drop out of `tank_*` (X40).
- `DecayQueueTank` (`storage.py:307-315`, `tanks.py:645-696`): `number_of_timesteps` defaults to
  1 (`tanks.py:648`) against its docstring (X12); the override is `tanks.py:682-685` (X28).
  A push adds undecayed water to `storage` (`tanks.py:536-538`); `storage` loses the internal
  arc's `total_decayed` (`tanks.py:694`), hence the carried row (X13). `active_storage` never
  decays (TODO at `tanks.py:692-693`). No `update_queue` at end of day; the new slot 0 is
  `sum_vqip(decay(slot 1), decay(slot 0))`, the reverse of `AltQueueArc` (X30 for the order of
  addition). A quantity-only run keeps the structure (X25).
- `DecayArc` (`arcs.py:668-728`), `DecayArcAlt` (`arcs.py:731-805`): decay at the in-port's
  temperature (`core.py:464-470`; upstream TODO at `arcs.py:700-702`); arcs read `parent`
  (`core.py:465-468`), so `parent` on an arc is refused; `decays` on other arc classes is stored
  unread (`arcs.py:52`) and refused; arc `apply_overrides` takes `capacity` and `preference` only
  (`arcs.py:87-96`). Backflow at `arcs.py:399-405`. Bound after every node so the factor uses
  the same day's in-port temperature (D15). Order of addition for a `DecayArc` is list order
  (age descending, then call order); the row model knows ages only while not due, so the term is
  exact unless an end of day finds due-and-waiting rows in both directions, when
  `"decay_arc_order_unknown"` is recorded and only the last bit may differ (D21). A catchment
  in-port carries its own `"wsimod.temperature"` row (X29).
- Surfaces: `land.py:303-581`; `DecayObj` reads its parent's `("temperature", date)`
  (`core.py:462-468`), the Land's air temperature. Nitrite carry-over at `land.py:427-439`;
  `Surface.run` converts at every call (X16 for the run-twice case); labels
  `(node, "surfaces.<name>")` in `total_decayed` against `(node, "surface:<name>")` in `storage`
  (X35). Shared mutable `decays={}` default: per-surface merge (D13).
- Exponent checks: `core.py:513-515` and `wtw.py:191-193` do none; `0 ** negative` raises,
  a negative base gives a complex number. `decays` naming `volume` (`core.py:510-519`) refused.

## Treatment works

`WTW.treat_current_input` is `wtw.py:163-214`; defaults at `wtw.py:73-85`. Solids are the
unclamped remainder and can go negative (`"wtw_negative_solids"`); the liquor blend can divide by
zero (`"wtw_liquor_zero_denominator"`, previous value kept); with no influent the temperature
column is 0 after a reset. The `liquor_multiplier` driver was `(..., n)` before pollutant
treatment was ported; a driver of that shape is refused with the new shape named.

## River

`storage.py:738-898`; `calculate_discharge` assigns `bio_in`/`bio_out` at `storage.py:923-924`
(a second call a day loses the first call's terms in WSIMOD's check; noodl adds both, X16, with
the print exception D17, W90/W91, X32). Mineralisation with no DIN at `storage.py:866-873` destroys
nitrogen (X14, X15). Registers: window sums are CPython's `sum` over Python floats (compensated
from 3.12), `window_sum` carries the plain sum's gradient (X55); parity assumes the lists hold
Python floats as `read_csv` produces. The 365-day mean adds 366 entries; `totalphosfcn` mixes
kg and kg/m3 (X14). River order ties break by set hash upstream; noodl uses configuration order
(W13, W64). The default suite runs `river_nutrients` for 400 days without decisions (X39); the
slow suite runs 1100 with decisions and asserts each river's branches (D24).

## Growing surface (L1 to L26)

Code references: `land.py:1186-1250` (25 attributes set in `__init__`), `:1376-1431` (31 names
with `rooting_depth`), `:1192` (depth), `:1514-1572` (`calc_crop_cover`).

- L1 leap year: `if t.is_leap_year:` is a method in the individual-files loader (always true) and
  a property in the parquet loader; day 59 for 1 March and 364 for 31 December every year.
- L2 autumn calendar (`sowing_day > harvest_day`, or default `harvest_day=365`): unsorted list,
  `bisect_left` returns zero-cover segments.
- L3 `Ks` denominator `(1 - ET_depletion_factor) * TAW` is 0 at the wilting point for the
  default factor (`:1566`): `Ks = 0`, event; crash case `growing_default_depletion` (day 11).
- L4 `TAW <= 0`: refused. L23 `quick_interp` wraps (negative index) and raises past the last
  stage. L22 the configured `et0_coefficient` is overwritten (listed unread). L26 harvest branch
  stores `crop_factor_stages[0]` which the next statement overwrites: not ported. L21/Y36 residue
  unwired: refused.
- L5 `pull_storage` (`:1433-1493`) divides by DIN at `:1460`: three species 0, event
  `"pull_storage_no_din"`; crash case `growing_no_din` day 0. L6 transformation (`:1744-1811`,
  `np.py:396-481`) divides at `:1758`: change to `ammonia`, event; cases `growing_zero_n`,
  `growing_nitrite_only` day 0. L7 `calc_crop_uptake` (`:1858-1926`) divides by volume at
  `:1899`: no uptake, event; `growing_empty_soil` day 1.
- L8 `effective_precipitation_flushing` (`:1605-1647`) extracts outflow nutrients a second time
  (tank phosphate 1.158 kg against 0.052 in the pool after 120 days of `growing_spring`);
  reproduced without a switch. L9 uptake and denitrification take all N as nitrate (`:1917`,
  `:2145`). L10 the "not tracking" check compares a VQIP with itself: event
  `"nutrient_pool_not_tracking"`. L11 the Freundlich warning tests `extracted - request >
  FLOAT_ACCURACY`, impossible: noodl tests `request - extracted`, event
  `"freundlich_flow_adjusted"`. L12 soil temperature factor `T / 5` (`:1813-1831`) jumps at 5 °C.
- L13 `irrigate` reads five pollutant keys whatever the pollutants (`land.py:2373`, `KeyError`):
  pool updated only with `nitrate` (Y20; case `irrigation_without_nutrients`, day 5).
- L14/L15 override defects (`pollutant_load`, `data_input_dict`; `land.py:375-382`, `:413-421`),
  Y21. L16 float `initial_soil_storage` raises `TypeError`. L17 `depth`. L18 two garden surfaces
  (`:120-128`; Y28), L19 garden water at 0 °C (`demand.py:273-287`; Y29). L20 `NutrientPool`
  parameters not configurable (`np.py:16-23`). L24 erosion reads `solids`. L25 dry deposition
  without wet data raises `KeyError`.
- Erosion (`:1928-2094`) uses the raw day of the year in the sine; eroded masses bypass the tank
  (mass-balance part `"GrowingSurface.erosion"`); `"erosion_pool_short"` where WSIMOD prints
  "weird nutrients". Adsorption (`:2154-2265`): Newton iteration unrolled to
  `adsorption_nr_maxiter` passes under a mask; the convergence test lags one pass; the shortcut
  path (`"adsorption_shortcut"`); `kadsdes` below -709.78 overflows `math.exp`: refused.
- Irrigation `land.py:2314-2381` (demand `:2352`; apply order `:192-198`; nutrient pool
  `:2372-2381`); garden `:2384-2437` (decisions `:2411`, `:2416`); variable area `:2440-2461`
  (scaling `:2455-2459`; erosion reads unscaled rain at `:1940`; negative area raises at `:1984`,
  Y50 for the override refusal). Fertiliser and manure `:1649-1710`; `adjust_vqip_to_liquid`
  `:1574-1603`; deposition `:462-581` and `:2267-2311`.

## Mass balance and records

WSIMOD's system check compares `sys_in` with itself (`model.py:1043-1058`); noodl computes the
real one. The orchestration is `model.py:993-1176`; `record_all` surface field set at
`model.py:1150`. Windows: a day's window ends after `end_timestep`; `storage_end` is `None` at the
`run_days` yield point. The evaporation-clamp gap (`"inaccurate_evaporation"`) equals the residual
on every day. Overrides: `model.py:819-825` (slot key and name), `:1252-1269` and `nodes.py:93-99`
(`filename`, working directory, X27), `nodes.py:92-100`. Dates: `model.py:139-162`. Foul slot:
`model.py:658-660`. `EnfieldFoulSewer`: `sewer.py:288-291`. `Land -> Sewer -> Land` ponding is not a
recursion cycle; the `land_sewer_ponding` crash test lowers the sewer's capacity between
`Land.run` and `make_discharge` (W73). `model.check(probe=True)` cannot step the model because the
closure requires `t` (W82).

## Print admission (D12; Y35, Y41, Y91)

WSIMOD's per-node check can print round-off "mass balance error for org-phosphorus/org-nitrogen"
lines on a land with a growing surface when tanks hold a large organic store; whether a line
falls depends on the platform's `exp`, `**` and `log`. `assert_matches_wsimod(...,
store_bounded_prints={land: c})` admits such a line of an additive pollutant where
`|x| <= c * eps * S`; `GROWING_PRINT_C = 2`; the maxima are recorded in `PRINTS_MEASURED`.
Only `growing_spring` must print. The admission is made at every such land of every case, the
crash cases' pre-crash days included.

## Suite composition and decisions

Y68: the growing cases run a prefix by default. Y90: quantity-only volume equality by default
only on the three cases WSIMOD does not run. X39: `river_nutrients` default 400 days. D17: the
pollutant-print exception of `river_double_discharge`. D24: branch assertions from recorded
decisions. D13: five crash cases for the growing surface, pinned lines and days. D21: the
decay-arc order event. D15: decay arcs bind after every node. Y43: a non-additive species in
deposition blends at volume 0 (`deposition_non_additive`). Y48: the crash triggers are
ordinary inputs.

## Source-of-truth locations

Parity fixtures and cases: `tests/verification/_wsimod_cases.py`, `test_wsimod_parity.py`,
`test_wsimod_replay.py`, `test_wsimod_gradients.py`. Class tests: `tests/apps/wsimod`. The
benchmark: `benchmarks/wsimod_run.py` (autograd node counts: about 4,000 a day for `quickstart`,
15,600 for `oxford`; no budget asserted, W8).
