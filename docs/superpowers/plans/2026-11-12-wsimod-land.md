# WSIMOD Land Processes (3c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `apps.wsimod` ports what WSIMOD 0.8.1 does on the land: surface deposition and pollutant loads, the `GrowingSurface` crop calendar and its `NutrientPool` soil nutrient cycle, `IrrigationSurface`, `GardenSurface` with the `Demand` garden link, `VariableAreaSurface` and a surface's `data_input_dict` override. The port is faithful, batched, branch-free and differentiable. It is bitwise equal to live WSIMOD node by node, on both demos running their unmodified `config.yml` with every process live, and on purpose-built cases. After 3c nothing is refused as "3c".

**Architecture:** A transcription on the 3a/3b substrate (Y2). The growing classes live in a new `growing.py`, and the batched pool, a `(..., 5, 2)` tensor per growing surface, lives in a new `nutrient_pool.py`. Every date-only quantity is a Python number computed at build time: the day of year, the sowing and harvest tests, `quick_interp`'s indices and erosion's `sin` (rule 2). Every transcendental with a tensor argument is evaluated per instance by Python (rule 1), so every 3c class keeps `replay_bound = 0`. Structural switches (`nitrate`, the constructor's `nhx-dry` and `pollutant_load`, the class, `adsorption_nr_maxiter`) decide which functions are built. Data decide only masks.

**Tech Stack:** Python 3.12, PyTorch (float64, CPU), PyYAML, pytest (+ xdist), `wsimod==0.8.1` (tests and the fixture script only).

**Spec:** `docs/superpowers/specs/2026-11-12-wsimod-land-design.md` at `e49a8fa`. It is binding: read it whole. It carries rulings Y9–Y50 and the defect table L1–L26, and it is the authority wherever this plan is silent. The scope rulings Y1–Y8 are summarised in its header. The 3a rulings W1–W95 and the 3b rulings X1–X40 still bind. WSIMOD source: `C:/projects/noodl/.venv/Lib/site-packages/wsimod/`; every `file:line` below is relative to it (`land.py` is `nodes/land.py`, `np.py` is `nodes/nutrient_pool.py`).

## Global Constraints

- **Start point.** Worktree `C:/Users/mvr/AppData/Local/Temp/noodl-wt/wsimod-land`, branch `wsimod-land`: main `13b997d`, plus the spec commits `31afc2b` and `e49a8fa`, plus this plan. One PR, opened by the lead after a Fable review (Y6).
- **Never push. Never `git stash`** (the stash list is shared between worktrees). Never `git checkout`, `git reset`, `git restore` or `git checkout -- <path>`.
- **Staging.** The worktree may hold unrelated uncommitted files. Stage files by name only (`git add <path> ...`); never use `git add -A`, `git add .` or a directory. Before each commit, run `git diff --cached --stat` and check that it lists only your task's files.
- **Formatting.** Run `ruff format` only on files you name on its command line, never on a directory. `ruff check .` must pass before every commit. Run it unpiped (no `| head`, no `| tail`), so that its exit status is the one you read.
- **Interpreter.** Use `C:/projects/noodl/.venv/Scripts/python.exe` from the worktree root; pytest puts the worktree's `src` first.
- **The core subset:** `tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py`.
- **The wsimod subset** (the AGENTS.md wsimod row; its paths do not change in 3c): `python -m pytest -q -n 4 --no-cov <core subset> tests/apps/wsimod tests/verification/test_allocation_wsimod_*.py tests/verification/test_wsimod_*.py`.
- **Slow tests.** Add `-m slow` on the wsimod row when a task changes what a slow test compares.
- **mkdocs.** Run `python -m mkdocs build --strict` at the end of every task that touches `docs/`.
- **Catalogue.** A task that adds or changes a public name runs `python scripts/gen_catalogue.py` and commits `docs/catalogue/wsimod.md` (`tests/test_docs_catalogue.py`).
- **Commit.** `git -c user.email=58522531+mvreeuwijk@users.noreply.github.com -c user.name="Maarten van Reeuwijk" commit`. Every message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Project rules.** Physical names and no aliases: a moved name is imported from its new home and never re-exported. Code, tests and docs change together. No wall-clock assertions: times are measured and recorded, never asserted. Coverage is 90 % on the full suite. `src/` never imports `wsimod`.
- **Transcription.** Every transcribed behaviour cites its WSIMOD `file:line`. Where the spec and the source disagree, the source wins, and the spec is corrected in the same commit (W2).
- **Bitwise parity.** Every 3c class has `replay_bound: ClassVar[float] = 0.0`, and every whole-model and replay comparison of a 3c case expects 0. "Bitwise" means equal under `abs(a − b) <= 0` / `torch.equal`, for which `-0.0 == 0.0` (rule 6, Y50). A non-zero bound needs a measured cause, written beside it and in the test (W19). The only planned admission is Y41's store-bounded print (D12).
- **Per-instance Python math.** Every `**`, `exp`, `log`, `log10` and `sin` whose argument is a tensor goes through `pow_python` or the `log_python`, `log10_python` and `sin_python` helpers (`vqip.py`), or through `exp_python` (`storage.py`). Each helper evaluates per instance with Python's `float.__pow__` or `math.*`, wherever torch can differ in the last bit (Approach rule 1, Y26). Date-only values are Python floats computed at build (rule 2). Every other operation is `+ − × ÷`, `min`, `max` or `abs` in WSIMOD's operand order. The tie rule is Python's: `min(a, b)` is `a` unless `b < a`.
- **The end-of-day rule.** 3c adds nothing to `end_timestep` (`day.py:1216`, Y50). Every new register is written during `Land.run`, `apply_irrigation` or the garden handlers, the harvest reset included. No 3c code calls `ctx.decide` or `ctx.call` from an `end_timestep`.
- **Numerics.** Everything in 3a's and 3b's Approach holds: `torch.float64`; `write(active, new, old)`; `MAXITER` masked rounds; safe denominators; WSIMOD's reduction order, one term at a time; `1e15`, never `inf`. Each method takes the caller's `active` mask, and every branch after a guard runs under the mask the guard leaves.
- **Decision sites.** The grammar is `"<node|arc>:<name>/<attribute path>/<WSIMOD method>/<branch>"`, and the 3c sites are the spec's Decision vocabulary list. Write them into `engine.py`'s module docstring in the task that adds them.
- **Reference.** Live WSIMOD 0.8.1 through `pytest.importorskip("wsimod")`, with `reset_wsimod_globals()` before every WSIMOD build. No case patches WSIMOD. A crash case runs WSIMOD as it is and asserts its exception (D13).
- **Suite budget.** The wsimod row's default run stays within 10 minutes with `-n 4` (W15). Every new slow-only test is marked `slow`. No 3c case may be among the default run's 30 slowest items; if one is, its default horizon is cut, and the full horizon stays under `slow`.
- **Progress file.** `.superpowers/sdd/2026-11-12-wsimod-land/progress.md` (git-ignored) holds the baselines, measured times, measured constants (D12) and open causes.
- **Implementer models (Y8).** cheap = Haiku, standard = Sonnet, most capable = Opus. Each task names one.

## Review Focus

1. **A batch whose instances leave the adsorption Newton loop at different iterations, one of them by the shortcut** (`adsorption`): each instance should be bitwise equal to its single run, with the stopped instances' `xn`, `xn_1` and `fxn` frozen (Y25), and no instance should take another's iteration count. Tests in Tasks 8 and 10.
2. **A soil exactly at field capacity or exactly at the wilting point** (`calc_soil_moisture_dependence_factor`, `denitrification`, `calc_crop_cover`): expect WSIMOD's tie branches. `smc == fc_m` gives `smdf = satact` (`>=`), while denitrification takes the ratio branch, giving `((1 − limpar)/(1 − limpar)) ** exppar`, which is `1.0` (`>` is strict). `smc == wp_m` gives `smdf = 0`. A tie must not fall into a neighbouring branch. Tests in Task 7.
3. **Negative fertiliser, manure or deposition values** (a sign error in user data): expect them transcribed. `adjust_vqip_to_liquid` skips its `dep > 0` scaling, the pool receives the negative mass and the tank receives it too, equal to WSIMOD, with no NaN and no refusal. Test in Task 6.
4. **A warm start mid-season** (`days_after_sow > 0`) from another run's last state, whose next days cross the harvest day: expect the harvest to reset `days_after_sow` to −1 and both covers to 0 on the harvest day, equal to the continuation of the first run. It must not be treated as an unsown start. Test in Task 10.
5. **A surface data row dated mid-month** (`"2004-01-15"`): WSIMOD looks up `(variable, first of the month)` and never matches that row (`land.py:462-471`, `model.py:164-170`). Expect the forcing check to refuse it, naming the variable and `2004-01-01`, the date WSIMOD would raise `KeyError` on. The row must not be accepted as January's. Test in Task 3.

## Decisions taken where the spec is silent or ambiguous

Each decision is binding for the implementer.

- **D1 The 3c-ii boundary (Y31).**
  - Tasks 1–11 are 3c-i: deposition, `GrowingSurface`, the pool and their verification.
  - Task 12 opens 3c-ii (the surface `data_input_dict` override), and Tasks 13–16 add `IrrigationSurface`, `GardenSurface`, `VariableAreaSurface` and their structure tests.
  - The commit that ends Task 11 is the boundary. Nothing in Tasks 12–16 changes a 3c-i behaviour or test except by adding to it. A task-12+ edit to a 3c-i file adds a branch or an entry; it never rewrites one.
  - Until its task, each of the three subclasses stays in `OUT_OF_SCOPE` with the reason `"3c"`. Task 17's grep proves that none remains.
  - `GrowingSurface`'s own attribute overrides (`land.py:1376-1431`) are part of the class, so they are 3c-i (Task 5). The 3c-ii "overrides" are the `data_input_dict` override, the unread `pollutant_load` override (L14, L15) and the `surface_overrides` case.
- **D2 Modules and registration.**
  - `growing.py` imports `land.PerviousSurface`, and at import it registers its classes in `land.SURFACE_CLASSES`, as `land.py` registers `NODE_CLASSES["Land"]`.
  - `engine._register_all` imports `growing`. `nutrient_pool.py` imports only `vqip`, `engine` and `torch`.
  - `land.py` never imports `growing`. Where `land.py` needs a class test, it uses `getattr(surface, "<flag>", False)` on a class attribute: `has_nutrients`, `is_garden`, `is_irrigated`.
- **D3 The math helpers.** `log_python(x)`, `log10_python(x)` and `sin_python(x)` live in `vqip.py`, beside `pow_python`, as `torch.autograd.Function`s. The forward pass maps `math.log`/`math.log10`/`math.sin` over `x.detach().reshape(-1).tolist()`, and the backward pass returns `grad / x`, `grad / (x * math.log(10))` and `grad * torch.cos(x)`. Domain errors propagate as Python's `ValueError`, because every guarded call site keeps its argument in the domain. `exp_python` stays in `storage.py`, and `growing.py` imports it from there.
- **D4 State keys with a trailing shape.**
  - `day.py` gains `_TRAILING = {"wsimod.nutrient_pool": (5, 2), "wsimod.surface_climate": (2,)}`, and `_rank(key)` becomes `1 + len(_TRAILING[key])` for those keys.
  - `initial_state`, `_dump`, `_load` and `_batch_shape` use `(rows, *trailing)` where today they use `(rows,)` or `(rows, 1 + K)`.
  - `"wsimod.days_after_sow"`, `"wsimod.crop_cover"` and `"wsimod.ground_cover"` join `_FLAT_KEYS`.
  - All five join `_NOT_HELD` (Y11). The units are `kg`, `d`, `-`, `-` and `m3`.
  - A trailing-shaped key's `Field` has `column=False`, `species=None`, and a description that names the axes: the pools `fast_pool, humus_pool, dissolved_inorganic_pool, dissolved_organic_pool, adsorbed_inorganic_pool`, then `N, P`; or `evaporation, precipitation`.
  - `initial_state(values=...)` accepts:
    - for `nutrient_pool`, `{label: {pool: {"N": x, "P": y}}}`, with pools and nutrients that are not named left unchanged;
    - for `surface_climate`, `{label: {"evaporation": x, "precipitation": y}}`.
- **D5 Store paths.**
  - A growing surface's stores add `days_after_sow`, `crop_cover`, `ground_cover`, `crop_factor` and `et0_coefficient`. With nutrients they also add `nutrient_pool`, `temperature_dependence_factor` and `soil_moisture_dependence_factor`.
  - An Irrigation or Garden surface adds `surface_climate`.
  - Carried: `nutrient_pool` (with nutrients), the three crop registers (always), and `surface_climate`. The others are per-day stores that the replay compares.
  - `surface_climate` is a view. `stores()` stacks the volumes of `evaporation` and `precipitation`. `load_stores` rebuilds both VQIPs as `v_change_vqip(empty, volume)` (`land.py:1029-1031`).
  - The signature changes to `Land.store_paths_of(surfaces, *, nutrients: bool = False)`, where `nutrients` is `"nitrate"` among the run's pollutants. The harness's `_store_paths` passes `"nitrate" in constants.POLLUTANTS`.
  - State rows: `(land, "surfaces.<name>.nutrient_pool")` for the pool, and `(land, "surfaces.<name>")` for the others (spec State).
- **D6 `days_after_sow`.**
  - A float register, with −1 for `None`.
  - The sowing and harvest tests are structural, from the build-time calendar (D7).
  - `is None` is the decision `S/calc_crop_cover/unsown`, `days_after_sow < 0` (margin `days_after_sow`).
  - Each `if days_after_sow:` (`:1550`, `:1872`, `:2350`) is the decision `…/crop`, `days_after_sow > 0`.
  - The harvest sets −1 and the covers 0 under `active & ~unsown`. The sowing sets 0 under `active & unsown`. Otherwise the register adds 1 under `active & ~unsown`.
- **D7 The build-time calendar.**
  - `GrowingSurface.set_dates(dates, labels)` builds one `CalendarDay` per network date, in WSIMOD's arithmetic: `bisect.bisect_left` on the same Python lists, Python indexing (the wrap included), and `math.sin(2 * PI * ((raw - 70) / 365))`. `Engine.__init__` calls it on every growing surface, and unit tests call it directly.
  - The day's entry is `calendar[ctx.day]`.
  - A date whose interpolation would raise `IndexError` or divide by zero is refused at build, naming the surface, the stage list and the date label.
  - **Y47's step-time clause is unreachable:** `WsimodDay._day_index` confines `model.step` to the network's dates (3a D16), all of which are checked at build. A test pins that `model.step` cannot reach a date outside them, so no separate step-time check is written (see Spec gaps).
- **D8 Surface forcing.**
  - `land.py` gains `SURFACE_DEPOSITION = ("nhx-dry", "noy-dry", "srp-dry", "nhx-wet", "noy-wet", "srp-wet")` (replacing `DEPOSITION_KEYS`) and `SURFACE_INPUTS = ("nhx-fertiliser", "noy-fertiliser", "srp-fertiliser", "nhx-manure", "noy-manure", "srp-manure")`.
  - The drivers are `"wsimod.<variable>"` `(..., T, n_readers)`, `forcing=True`, unit `kg/m2/d` in `_FORCING_UNITS`, with rows `(land, "surfaces.<name>")` and targets `(land_node, "surfaces.<name>.<variable>")`.
  - Each day's value is `spec.data[(variable, month_start(date))]`, where `month_start(d) = datetime(d.year, d.month, 1)`.
  - `Land.bind` routes `surfaces.<s>.<variable>` to the surface. `Surface.bind` stores the twelve variables in `self.forcing: dict[str, Tensor]`, not among its parameters.
  - `structure.surface_forcing_readers(network) -> dict[str, list[tuple[str, str]]]` lists `(land, surface)` per variable. `forcing_readers` is unchanged.
- **D9 Constructor flags (Y21).** `SurfaceSpec` gains `constructed_variables: frozenset[str] = frozenset()` and `constructed_load: tuple[str, ...] = ()`. `_read_surface` sets both before any override. `Surface.__init__` reads `has_deposition = "nhx-dry" in spec.constructed_variables` and `has_load = len(spec.constructed_load) > 0`.
- **D10 Events.**
  - `run.EVENT_NAMES` gains the eight events in order of introduction: `"crop_water_stress_zero_denominator"` (Task 5), `"pull_storage_no_din"` (6), `"nutrient_pool_not_tracking"`, `"transformation_no_din"` and `"crop_uptake_dry_soil"` (7), and `"adsorption_shortcut"`, `"erosion_pool_short"` and `"freundlich_flow_adjusted"` (8).
  - Each event is recorded at the `Land`'s name.
  - `_wsimod_parity._UNDATED` gains `"adsorption_shortcut"` and `"erosion_pool_short"`, with patterns copied from WSIMOD's print strings (`land.py:2193`, `:2053`).
- **D11 Mass-balance parts (Y11).**
  - Add `Ctx.part(name: str, direction: str, vqip: Tensor, mask: Tensor) -> None`. It sums into `ctx.parts[direction][name]` as `term` sums, and is not summed into the balance.
  - `DayRecord.parts` and `WsimodMassBalance.parts: dict[str, dict[str, Tensor]]` hold `{"in": {...}, "out": {...}}`, each entry `(..., T, 1 + K_add)`.
  - `Surface.run` records each function's `(in_, out_)` under `PART_NAMES[f.__name__]`, a module dict in `land.py` that maps the method to `"<WSIMOD defining class>.<method>"`. It lists exactly the spec's parts table, plus `"Surface.nitrite_carry_over"` for the carry-over. A function absent from it returns empty VQIPs.
  - A repeated `Land.run` adds to its parts.
- **D12 The Y41 admission.**
  - `assert_matches_wsimod` gains `store_bounded_prints: Mapping[str, float] = {}`, which maps a land's name to `c`.
  - A line `mass balance error for <p> of x in <land>` of a mapped land passes when `|x| <= c * 2.220446049250313e-16 * S`. `S` is the largest storage of `p` over the land's surfaces in WSIMOD's recorded surface quality, at the end of that day and at the end of the day before (the initial storage on day 0). The line must also fall on its day.
  - An unmapped land's line fails as today, and that is how "zero lines" is asserted.
  - Task 9 measures `max |x| / (ε S)` per case and sets `GROWING_PRINT_C = ceil(maximum)` in `test_wsimod_parity.py`, with the measured values in a comment and in the progress file. The expected value is 2, from the review's ≈1.4.
- **D13 Crash cases.**
  - `Case` gains `crash: tuple[str, str] | None = None`: the exception's class name and the `"land.py:<line>"` WSIMOD raises at. A crash case has `wsimod=False`, so the generic parity and replay skip it.
  - `_wsimod_parity.wsimod_until_crash(mapping) -> tuple[WsimodReference, BaseException, int]` runs WSIMOD one day at a time until it raises. It returns the days recorded, the exception and the crash day.
  - `test_case_runs_where_wsimod_crashes[<case>]` asserts:
    - the exception's class, and the line of its innermost frame in `nodes/land.py`;
    - noodl's run equals the reference bitwise on the days before the crash day (vacuous on day 0);
    - noodl's event (where the row has one) is true at the land on the crash day;
    - the results are finite;
    - noodl's balance closes.
- **D14 Context propagation.** `GrowingSurface`'s `ctx` setter also sets its pool's. `NutrientPool` has `ctx` like a `Tank`.
- **D15 Unread entries (strings in `network.unread`):**
  - `"<land>.surfaces.<s>.pollutant_load.<key>"`: a load key that is not a configured pollutant, `volume` included (Y38);
  - `"<land>.surfaces.<s>.data_input_dict.<variable>"`: a wet variable without `nhx-dry`, a deposition variable whose species (`nhx`→`ammonia`, `noy`→`nitrate`, `srp`→`phosphate`) is not a pollutant, and a deposition variable added by an override to a surface built without `nhx-dry`;
  - `"<land>.surfaces.<s>.pollutant_load"`: a load override on a surface built without a load (L14);
  - `"<land>.surfaces.<s>.et0_coefficient"`: every growing surface (L22, Y39);
  - `"<land>.surfaces.<g>.garden_handlers"`: every garden surface but the last on its land (L18).
- **D16 Surface driver columns.**
  - `Surface.driver_columns() -> dict[str, tuple]` (default `{}`) names the columns of its non-pollutant vector parameters. `_driver_layout`'s surface loop consults it before the pollutant-column rule.
  - A growing surface returns `("N", "P")` for the six `nutrient_pool.<p>` pairs.
  - It returns `tuple(range(S))` for `crop_factor_stages`, where `S` is that class's longest list. The layout pads each shorter template row with 0 (Y33), and `GrowingSurface.bind` refuses a non-zero padding entry, naming the surface and the index.
- **D17 Growing parameters.**
  - `GrowingSurface.parameter_defaults` is `PerviousSurface`'s without `depth` and `et0_coefficient`, plus the spec's 25 attributes, `ET_depletion_factor = 1` and `rooting_depth = 1`. `et0_coefficient` is accepted (schema and override) and listed as unread (D15), but it is not a driver.
  - The structural keys `crop_factor_stage_dates`, `sowing_day`, `harvest_day`, `adsorption_nr_maxiter` and `initial_soil_storage` are read from `spec.parameters` at construction and are not drivers.
- **D18 Validation scope.**
  - `_validate_surface` applies `PerviousSurface`'s checks (area, porosity, `ihacres_p`, soil weights, `temperature`) to the four growing classes too. For them the physical depth is `rooting_depth`.
  - The `nitrate` pollutant requirement (eight names) and the fertiliser and manure forcing check (Y49) use the network's pollutant list in both modes.
  - The parameter-domain refusals are checked at build from the merged parameters, and again in `GrowingSurface.bind` over every batched instance.
- **D19 `pull_pollutants` and the tracking event.**
  - `Tank.pull_pollutants(vqip, *, active) -> Tensor` clamps each additive column, in the run's additive order, then the volume, with the decisions `<site>/pull_pollutants/clip:<name>`. It then extracts the result and returns it (`tanks.py:261-266`).
  - The four growing call sites go through `GrowingSurface._pull_tracked(out_, *, active) -> Tensor`. It records `"nutrient_pool_not_tracking"` where `active & any(request − clamped > FA)` and returns the clamped VQIP.
- **D20 Partial process lists.** Tasks 6–8 append each process to `inflows`/`processes` as it lands, in WSIMOD's list order. No whole-model `nitrate` case exists before Task 9: Tasks 6–8 are judged by unit tests against WSIMOD's own objects.
- **D21 Calendar of the cases.**
  - Every 3c case's dates start on 2003-10-01 (`_land_dates(n)`), so that a run crosses 29 February 2004.
  - Monthly surface data come from `_monthly(variables: Mapping[str, Callable[[int, int], float]], start: str, months: int) -> dict`, keyed `"YYYY-MM"`.
  - `growing_spring` has `days=540` and `slow_days=1096` (three years).
- **D22 Orchestration variants.** "`apply_irrigation` before `Land.run`" is its own case, `irrigation_early`, with a custom orchestration. The rest of its mapping is `irrigation`'s.
- **D23 `et0_coef` and `crop_factor` in the results.** `day._SURFACE_FIELDS` and the harness's `_SURFACE_FIELDS` gain `"et0_coef": "et0_coefficient"`, recorded for every non-impervious surface as `record_all` does (`model.py:1150`; a `PerviousSurface`'s is its parameter). `results.surface["crop_factor"]` is noodl's own field, NaN except on growing surfaces, and is never compared with WSIMOD (Y32).
- **D24 The garden handlers.** `Land.__init__`, after building its surfaces, binds the last `GardenSurface`'s `calculate_irrigation_demand` and `receive_irrigation_demand` as `push_check_handler[GARDEN_TAG]` and `push_set_handler[GARDEN_TAG]` (`land.py:120-128`). Without one, 3a's deny and empty handlers stay.

---

### Task 0: Baseline (controller)

**Implementer:** cheap (the controller runs it). **Files:** none changed.

- [ ] **Step 1:** Run `git status --short` and `git log --oneline -3`, and record both in the progress file.
- [ ] **Step 2:** Run the wsimod subset with `--durations=30`. Expected: pass. Record the wall time, the test count and the 30 slowest items under "Baseline".
- [ ] **Step 3:** Run `python -m pytest -q -n 4 --no-cov -m slow tests/apps/wsimod tests/verification/test_wsimod_*.py` and record its wall time and count.
- [ ] **Step 4:** Run `python benchmarks/wsimod_run.py` three times and record the median seconds per day for each (demo, batch, grad) row.
- [ ] **Step 5:** Record `git grep -n 3c src tests scripts docs/applications docs/formats` (the spec's refusal table) as the list Task 17 must empty.

### Task 1: The substrate: Python math, `pull_pollutants`, balance parts, the `RiparianBuffer` refusal

**Implementer:** standard.

**Files:**
- Modify: `src/noodl/apps/wsimod/vqip.py` (D3), `tanks.py` (`Tank.pull_pollutants`, D19), `engine.py` (`Ctx.part`, `Ctx.parts`), `day.py` (`DayRecord.parts`, `_record`, `_finish_record`), `run.py` (`WsimodMassBalance.parts`, `_balance`), `land.py` (`PART_NAMES`; `Surface.run` records parts), `storage.py` (`River.pull_check_fp` message, Y10).
- Test: `tests/apps/wsimod/test_vqip.py`, `test_tanks.py`, `test_run.py`, `test_storage.py`.

**Interfaces:**
- Consumes: `pow_python`, `Ctx.term`, `DayRecord`, `Surface.run` (3a/3b).
- Produces:

```python
# vqip.py
def log_python(x: Tensor) -> Tensor      # math.log per instance; d/dx = 1/x
def log10_python(x: Tensor) -> Tensor    # math.log10 per instance; d/dx = 1/(x ln 10)
def sin_python(x: Tensor) -> Tensor      # math.sin per instance; d/dx = cos x
# tanks.py
class Tank:
    def pull_pollutants(self, vqip: Tensor, *, active: Tensor) -> Tensor   # tanks.py:261-266, D19
# engine.py
class Ctx:
    parts: dict[str, dict[str, Tensor]]                                    # {"in": {}, "out": {}}
    def part(self, name: str, direction: str, vqip: Tensor, mask: Tensor) -> None
# run.py
class WsimodMassBalance:  parts: dict[str, dict[str, Tensor]]              # D11
# land.py
PART_NAMES: dict[str, str]   # "ihacres" -> "PerviousSurface.ihacres", ..., D11
```

- [ ] **Step 1: Write the failing tests.**
  - `test_log_log10_sin_python_are_pythons_bitwise` (`test_vqip.py`): 10⁴ seeded samples each on `log` and `log10` over `[1e-6, 50]` and `sin` over `[0, 0.2]` and `[-7, 7]`. Every element equals `math.<f>(float(x))`, compared as floats. The test also asserts that torch's own function differs from Python's on at least one sample of the `sin` range `[-7, 7]`, so that the helper is needed (spec: 3.0 %).
  - `test_gradcheck_on_the_python_math_helpers`: `torch.autograd.gradcheck` on each helper, inside its domain.
  - `test_pull_pollutants_equals_wsimods` (`test_tanks.py`, `importorskip`): a WSIMOD `Tank` with pollutants `phosphate`, `ammonia`, `temperature` (WSIMOD globals set to match). 200 random storages and requests, a third of the requests above the store in some column. The returned VQIP and the storage equal WSIMOD's. The returned VQIP is clamped column by column, and `volume` is clamped too. The decisions `node:t/tank/pull_pollutants/clip:phosphate`, `…/clip:ammonia` and `…/clip:volume` appear in that order.
  - `test_the_parts_sum_to_the_running_terms[<case>-<mode>]` (`test_run.py`), over every `CASES` entry that has a `Land`. For each direction, the sum of `mass_balance.parts[direction]` over its names equals `terms["Land.running_inflow_mb"]` / `["Land.running_outflow_mb"]` within `1e-12 * scale` on every day. On `surface_decay_nitrification`, `parts["in"]["Surface.nitrite_carry_over"]` is non-zero on some day.
  - `test_the_parts_are_not_in_the_balance`: on `pervious_only`, `residual` equals `inflow − outflow − storage_change`, recomputed from `terms` alone, `torch.equal`. No `parts` name appears in `terms`.
  - Rename `test_the_riparian_buffer_check_is_refused_by_name_with_3c` (`test_storage.py:920`) to `test_the_riparian_buffer_check_is_refused_for_good`. `match="RiparianBuffer"`, and the message names "no RiparianBuffer class in WSIMOD 0.8.1" and does not contain `"3c"`.
- [ ] **Step 2:** Run `python -m pytest -q --no-cov tests/apps/wsimod/test_vqip.py tests/apps/wsimod/test_tanks.py tests/apps/wsimod/test_run.py tests/apps/wsimod/test_storage.py`. Expected: FAIL (names missing).
- [ ] **Step 3:** Implement D3, D11 and D19. The `River.pull_check_fp` message cites `storage.py:588-589`, `:938-949` and Y10.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): Python log/log10/sin, Tank.pull_pollutants and the balance parts (3c)`.

### Task 2: Live `pollutant_load`, and both demos on `config.yml` (Y30, Y38)

**Implementer:** standard.

**Files:**
- Delete (`git rm`): `tests/data/wsimod/quickstart/config_no_deposition.yml`, `tests/data/wsimod/oxford/config_no_deposition.yml`.
- Modify:
  - `src/noodl/apps/wsimod/land.py`: `_validate_surface` accepts any load, and a non-pollutant key goes to unread (D15); the `Surface.simple_deposition` docstring;
  - `config.py` (`unread` for load keys);
  - `scripts/regenerate_wsimod_fixtures.py`: drop `_no_deposition_overrides`, `write_no_deposition_configs` and `--no-deposition-only`;
  - `tests/verification/test_wsimod_parity.py`: `CONFIG = {"quantity": "config.yml", "pollutants": "config.yml"}`;
  - `test_wsimod_replay.py`: the `RUNS` rows, and the comment at `:658`;
  - `test_wsimod_fixtures.py`: delete `test_the_no_deposition_configuration_differs_only_by_its_overrides`;
  - `test_wsimod_gradients.py` (`_model`);
  - `tests/apps/wsimod/test_build.py:355-365`;
  - `tests/apps/wsimod/test_land.py:386-405`, and the load rows of `:780-830`;
  - `tests/data/wsimod/README.md`; `docs/applications/wsimod.md` (the demo paragraph).

**Interfaces:**
- Consumes: `Surface.simple_deposition` (3a), `unread` (3b D10).
- Produces: `_validate_surface` returns no load refusal. `network.unread` holds `"<land>.surfaces.<s>.pollutant_load.<key>"` (D15).

- [ ] **Step 1: Write the failing tests.**
  - `test_simple_deposition_equals_wsimods` (`test_land.py`, `importorskip`): a WSIMOD `Surface` with a stub parent, and `pollutant_load {"phosphate": 0.3, "temperature": 7.0, "nitrite": 1.0}`, with `nitrite` not a pollutant. Tank volumes include 0 and 2.5. After `simple_deposition`, the storage and the returned `in_` equal WSIMOD's. The returned phosphate is `0.3 * area`, and the temperature blends to `(7.0*0 + t*V)/V` (or stays `t` at `V = 0`).
  - `test_a_load_key_that_is_not_a_pollutant_is_unread`: `"land.surfaces.s.pollutant_load.nitrite"` and `"…pollutant_load.volume"` are in `network.unread`, and the results equal those of the same run without the keys (`torch.equal`).
  - Replace `test_deposition_refusals_name_only_3c` with `test_a_non_zero_pollutant_load_builds_with_pollutants_on`.
  - `test_build.py`: quickstart with `pollutants=True` builds, and `"wsimod.Surface.pollutant_load"` (or the class key) holds phosphate's configured value.
  - The parity tests `test_demo_matches_wsimod[quickstart-pollutants]`, `[oxford365-pollutants]` and the slow `[oxford-pollutants]`, the replay `test_demo_nodes_replay_wsimod[pollutants-*]`, and `test_demo_quantity_only_equals_the_pollutant_run_volumes[*]` now run on `config.yml` unchanged in body.
- [ ] **Step 2:** Run `test_land.py` and `test_build.py`. Expected: FAIL (the loads are refused).
- [ ] **Step 3:** Implement. Delete the two files, and clean the script.
- [ ] **Step 4:** Run the wsimod subset, and `-m slow` on `test_wsimod_parity.py test_wsimod_replay.py`. Expected: PASS, bitwise, and neither demo prints anything (spec: none did). `git grep -n config_no_deposition -- . ":!docs/superpowers"` is empty. Docs: the demos run live, and the README says so.
- [ ] **Step 5:** Commit: `feat(wsimod): live pollutant loads; the demos run config.yml with every process (Y30)`.

### Task 3: Surface deposition data (`land.py:375-382`, `:462-581`)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/land.py` (D8, D9): the deposition functions, `_validate_surface`, `Surface.bind` forcing, `Land.bind` routing;
  - `config.py`: `SurfaceSpec`, `_read_surface`;
  - `day.py`: `_driver_layout` surface forcing, `_FORCING_UNITS`;
  - `structure.py`: `surface_forcing_readers`;
  - `engine.py`: sites;
  - `tests/verification/_wsimod_reference.py`: the capture of `t`, `monthyear` and each read `(variable, monthyear)` value;
  - `tests/apps/wsimod/_replay.py`: binds the captured surface data;
  - `tests/verification/_wsimod_cases.py`: `_land_dates`, `_monthly`, case `surface_deposition` (D21);
  - tests: `tests/apps/wsimod/test_land.py`, `test_config.py`, `test_build.py`;
  - docs: `docs/formats/wsimod.md` (surface data), `docs/applications/wsimod.md`.

**Interfaces:**
- Consumes: `Tank.push_storage` (3a), `PART_NAMES` (Task 1).
- Produces:

```python
SURFACE_DEPOSITION: tuple[str, ...]; SURFACE_INPUTS: tuple[str, ...]          # D8
def month_start(date: datetime) -> datetime
@dataclass class SurfaceSpec: ...; constructed_variables: frozenset[str]; constructed_load: tuple[str, ...]
class Surface:
    has_deposition: bool; forcing: dict[str, Tensor]
    def atmospheric_deposition(self, *, active: Tensor) -> tuple[Tensor, Tensor]    # land.py:528-554
    def precipitation_deposition(self, *, active: Tensor) -> tuple[Tensor, Tensor]  # :556-581
    def dry_deposition_to_tank(self, vqip: Tensor, *, active: Tensor) -> Tensor     # :473-487
    def wet_deposition_to_tank(self, vqip: Tensor, *, active: Tensor) -> Tensor     # :489-503
def surface_forcing_readers(network) -> dict[str, list[tuple[str, str]]]            # D8
def _land_dates(n: int) -> list[str]
def _monthly(variables, start: str, months: int) -> dict
```

- [ ] **Step 1: Write the failing tests.**
  - `test_deposition_equals_wsimods` (`importorskip`): a WSIMOD `Surface` and a `PerviousSurface`, each with a stub parent that supplies `t`, `monthyear`, `data_input_dict` and the running balances, given 15 months of `nhx`, `noy` and `srp` dry and wet data. Over 400 days from 2003-10-01:
    - the storages, each function's `(in_, out_)` and the running balances equal WSIMOD's;
    - the inflow order is `[atmospheric_deposition, precipitation_deposition, simple_deposition, <class functions>]`.
  - `test_deposition_reads_the_first_of_the_month` (Review Focus 5): rows dated `"2004-01"` and `"2004-01-01"` both serve every January day. A data set whose January row is dated `"2004-01-15"` is refused with `match="nhx-dry.*2004-01-01"`.
  - `test_surface_data_refusals` (parametrised, `match=`):
    - `nhx-dry` without `nhx-wet` (`"nhx-wet"`, L25), and each other missing companion;
    - variable `flow` (`"not read"`);
    - `nhx-residue` (`"residue"`, Y36, L21);
    - `nhx-fertiliser` on a `PerviousSurface` (`"fertiliser"`);
    - a read date missing (`"2004-03-01"`).
  - `test_wet_data_without_dry_is_unread`: the six `data_input_dict.<variable>` entries are in `unread`, and nothing is read.
  - `test_a_deposition_species_that_is_not_a_pollutant_is_unread`: without `ammonia`, `nhx-dry` and `nhx-wet` are listed (D15), and the forcing check still applies (refused when missing).
  - `test_a_non_additive_deposition_species_blends_at_volume_zero` (Y43): `ammonia` non-additive. The tank value changes only by `(x·0 + t·V)/V`, equal to WSIMOD's.
  - `test_the_constructor_flags_are_kept_apart_from_the_merged_data` (D9): `SurfaceSpec.constructed_variables` and `.constructed_load` equal the configuration's.
  - `test_quantity_only_lists_surface_data_as_ignored`: a quantity-only build lists `"surfaces.<s>.data_input_dict"` in `model.ignored_by_quantity_only`, and the forcing check still applies.
  - `test_surface_forcing_drivers`: `model.driver_template["wsimod.nhx-dry"]` has shape `(T, 2)`, labels `[("land", "surfaces.a"), ("land", "surfaces.b")]` and unit `kg/m2/d`, and every January day holds January's value.
  - Update `test_land.py:780-830`: the deposition-data rows become the refusals above or accepted configurations.
  - Case `surface_deposition` (spec Verification 3): `replay=("land",)`, `days=450` (the 15 months of data from 2003-10 hold 457 days), crossing month and year boundaries. Add it to `test_the_cases_cover_every_listed_feature`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Check that `wsimod_model` passes a surface's `data_input_dict` keyed by `to_datetime` (`_wsimod_reference.py:315-316`), and extend it if it does not.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS, with `test_case_matches_wsimod[surface_deposition-*]` and its replay bitwise. Docs: the deposition functions, the monthly lookup, the refusals and the unread entries. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): surface deposition from monthly nhx/noy/srp data`.

### Task 4: The batched `NutrientPool` (`np.py:11-564`)

**Implementer:** most capable (every method's operand order, the `min` caps, the integer zeros).

**Files:**
- Create: `src/noodl/apps/wsimod/nutrient_pool.py`, `tests/apps/wsimod/test_nutrient_pool.py`.
- Modify: `engine.py` (sites).

**Interfaces:**
- Consumes: `Ctx.decide`, `dmin`, `write`.
- Produces:

```python
POOLS = ("fast_pool", "humus_pool", "dissolved_inorganic_pool", "dissolved_organic_pool", "adsorbed_inorganic_pool")
FAST, HUMUS, DI, DO, ADS = range(5)
NUTRIENTS = ("N", "P")
NUTRIENT_POOL_PAIRS: dict[str, tuple[float, float]]   # np.py:16-23 as written: degrhpar (7 * 1e-5, 7 * 1e-6),
    # dishpar (7 * 1e-5, 7 * 1e-6), minfpar (0.00013, 0.000003), disfpar (0.000003, 0.0000001),
    # immobdpar (0.0056, 0.2866), fraction_manure_to_dissolved_inorganic (0.5, 0.1)
NUTRIENT_POOL_SCALARS: dict[str, float]               # fraction_dry_n_to_dissolved_inorganic 0.9
class NutrientPool:
    def __init__(self, *, site: str) -> None          # storage zeros (5, 2); site ".../surfaces.<s>.nutrient_pool"
    storage: Tensor                                    # (..., 5, 2)
    temperature_dependence_factor: Tensor; soil_moisture_dependence_factor: Tensor   # (...), the day's
    def bind(self, parameters: Mapping[str, Tensor]) -> None
        # the six (..., 2) and the scalar; derives fraction_manure_to_fast = 1 - fmdi and
        # fraction_dry_n_to_fast = 1 - fddi as calculate_fraction_parameters (np.py:146-153)
    def receive(self, pool: int, amount: Tensor, *, active: Tensor) -> None        # (..., 2), N then P
    def extract(self, pool: int, amount: Tensor, *, active: Tensor,
                order: tuple[str, ...] = ("N", "P")) -> Tensor
        # reply = min(storage, amount), storage -= reply, per nutrient in `order` (np.py:551-556);
        # decision "<site>.<POOLS[pool]>/extract/clip:<n>", call = that pool's extract count
    def allocate_inorganic_irrigation(self, amount, *, active) -> Tensor           # np.py:192-206
    def allocate_organic_irrigation(self, amount, *, active) -> Tensor
    def allocate_dry_deposition(self, dep, *, active) -> Tensor                    # np.py:226-248: fast.N, DI.N, ads.P
    def allocate_wet_deposition(self, dep, *, active) -> Tensor
    def allocate_manure(self, dep, *, active) -> Tensor                            # np.py:267-290
    def allocate_fertiliser(self, dep, *, active) -> Tensor                        # np.py:327-328
    def extract_dissolved(self, proportion: Tensor, *, active) -> tuple[Tensor, Tensor]   # (inorganic, organic), np.py:330-356
    def get_erodable_P(self) -> Tensor                                             # ads.P + humus.P
    def erode_P(self, amount: Tensor, *, active) -> tuple[Tensor, Tensor]          # (org, inorg), np.py:379-394
    def soil_pool_transformation(self, *, active) -> tuple[Tensor, Tensor]         # (inc_DI, inc_DO), np.py:409-450
    def temp_soil_process(self, parameter: Tensor, source: int, target: int, *, active) -> Tensor   # np.py:468-481
```

- [ ] **Step 1: Write the failing tests** in `test_nutrient_pool.py`:
  - `test_every_method_equals_wsimods_on_random_sequences` (`importorskip`, seeded). 2000 calls drawn from the method list of Scope › In on one WSIMOD `NutrientPool` and one noodl pool.
    - The draws include zero pools, requests above the pool, proportions in `[0, 1]`, and factors in `[0, 1.5]`.
    - A second run uses non-default parameters set on the WSIMOD object (then `calculate_fraction_parameters()`) and bound on noodl's.
    - After each call, every pool's storage and every return value are equal (`abs(a − b) <= 0`).
  - `test_extract_takes_pythons_min_per_nutrient_in_request_order`. `extract(DI, (3.0, 1.0), order=("P", "N"))` on a pool `(2.0, 1.0)` returns `(2.0, 1.0)`; the decisions are recorded P first, and the tie `min(1.0, 1.0)` keeps the storage.
  - `test_the_transformation_by_hand`: default parameters, `tdf = 0.5`, `smdf = 0.8`, pools `fast (10, 1)`, `humus (100, 10)`, `DI (5, 0.5)`, `DO (2, 0.2)`. The pools and `(inc_DI, inc_DO)` equal the five `temp_soil_process` steps written in the test as Python expressions (`par * tdf * smdf * X`, in `np.py:413-447` order).
  - `test_erode_P_splits_by_the_adsorbed_fraction`: `ads.P 3`, `humus.P 1`, amount 2 → `fa = 3/4`. The adsorbed pool gives `2*0.75`, the humus pool gives `2*(1-0.75)`, the N columns are unchanged, and the return is `(humus removed, adsorbed removed)`.
  - `test_sums_from_the_empty_nutrient_equal_pythons_under_torch_equal` (rule 6): a never-touched pool's `-increase` compares equal.
  - `test_a_batch_equals_its_single_pools_and_masked_instances_are_unchanged`.
  - `test_gradcheck_on_the_transformation_and_extract_dissolved` (with respect to storage, parameters and factors, away from the caps).
- [ ] **Step 2:** Run `python -m pytest -q --no-cov tests/apps/wsimod/test_nutrient_pool.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. Each method is a transcription in WSIMOD's statement order: functional writes (`write(active, …)`), with every request computed before its pool changes. The residue methods are not ported (Y36).
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): the batched NutrientPool`.

### Task 5: `GrowingSurface`: construction, overrides, the crop calendar and `calc_crop_cover`

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/growing.py`, `tests/apps/wsimod/test_growing.py`.
- Modify:
  - `land.py`:
    - `PerviousSurface.get_climate() -> tuple[Tensor, Tensor]` (`land.py:905-911`), which `ihacres` calls, bitwise as now;
    - `PerviousSurface.crop_et0_coefficient()`, overridden by the growing class;
    - `_validate_surface` (D18);
    - `store_paths_of` (D5);
    - `Surface.driver_columns` (D16);
  - `classes.py`:
    - `_CONSTRUCTORS["GrowingSurface"]` with WSIMOD's keywords (`:1085-1094`) and `initial_soil_storage`;
    - `_OVERRIDES["GrowingSurface"]` with the 31 names of `:1376-1408` plus `rooting_depth`;
    - `_NOT_OVERRIDES` and `_DERIVED` gain `depth`;
    - `SURFACE_SCHEMAS`;
    - `OUT_OF_SCOPE` loses `GrowingSurface`;
  - `config.py`: overrides of a growing surface;
  - `engine.py`: `_register_all`, `set_dates` (D7), sites;
  - `day.py` (D4, D23): `STATE_KEYS` gains `"wsimod.days_after_sow"`, `"wsimod.crop_cover"` and `"wsimod.ground_cover"`; `_STATE_TEXT`, `_STATE_UNITS`, `_state_layout`, `_SURFACE_FIELDS`, `_driver_layout` padding;
  - `run.py`: `"crop_water_stress_zero_denominator"`;
  - `tests/verification/_wsimod_reference.py`: `_store_paths`, `_wsimod_store` (`None` → −1), the capture of the crop state before and after each `Land.run`, `_SURFACE_FIELDS`;
  - `tests/apps/wsimod/_replay.py`;
  - `tests/verification/_wsimod_cases.py`: `_growing`, the cases `growing_no_nitrate` and `growing_default_depletion` (D13);
  - `tests/verification/test_wsimod_parity.py`: the crash test of D13;
  - tests: `test_config.py:415`, `test_build.py`;
  - docs.

**Interfaces:**
- Consumes: `PerviousSurface` (3a/3b), `month_start`, `SURFACE_INPUTS` (Task 3).
- Produces:

```python
GROWING_DEFAULTS: dict[str, float]          # the 25 attributes as written (land.py:1186-1250), ET_depletion_factor 1, rooting_depth 1
@dataclass(frozen=True)
class CalendarDay:
    raw: int; adjusted: int                 # t.dayofyear; minus 1 after day 59 every year (L1)
    sowing: bool; harvest: bool             # raw == sowing_day / harvest_day
    factor_index: int; cover_index: int     # bisect_left on crop_factor_stage_dates / harvest_sow_calendar
    erosion_sine: float                     # math.sin(2 * PI * ((raw - 70) / 365))
def quick_interp(x: int, xp: Sequence[float], yp: Tensor, i: int) -> Tensor
    # yp[..., i-1] + (yp[..., i] - yp[..., i-1]) * (x - xp[i-1]) / (xp[i] - xp[i-1]); i, xp Python (land.py:1495-1512)
class GrowingSurface(PerviousSurface):
    surface_class = "GrowingSurface"; has_nutrients: bool
    days_after_sow: Tensor; crop_cover: Tensor; ground_cover: Tensor; crop_factor: Tensor; et0_coefficient: Tensor
    calendar: list[CalendarDay]
    def set_dates(self, dates: Sequence[datetime], labels: Sequence[str]) -> None      # D7, refuses by name
    def calc_crop_cover(self, *, active: Tensor) -> tuple[Tensor, Tensor]              # land.py:1514-1572
    def crop_et0_coefficient(self) -> Tensor                                            # self.et0_coefficient
# state "wsimod.days_after_sow", "wsimod.crop_cover", "wsimod.ground_cover": (..., n_growing), rows (land, "surfaces.<s>")
# drivers "wsimod.GrowingSurface.<p>" (..., n) for GROWING_DEFAULTS and PerviousSurface's; ".crop_factor_stages" (..., n, S)
# Case.crash (D13); wsimod_until_crash (D13)
```

- [ ] **Step 1: Write the failing tests** in `test_growing.py`. Each WSIMOD-side test uses `importorskip`, with a stub parent supplying `t` and `monthyear` from WSIMOD's `to_datetime`.
  - `test_quick_interp_equals_wsimods`: random sorted `xp` and `yp`, the default calendar `[0, 1, 365, 366, 365]`, and the wrap `xp = [1, 365]`, `yp = [0.3, 1]` at `x = 1`, which gives `0.30000000000000004` (L23).
  - `test_calc_crop_cover_equals_wsimods_every_day[spring|autumn|default|wrap]`: calendars (90, 250), (280, 200), (1, 365) and `crop_factor_stage_dates = [1, 365]`, with five-point `crop_factor_stages` and `ET_depletion_factor = 0.5`. Every day of 2003–2005, with `smc` drawn either side of `RAW`. `days_after_sow` (−1 for `None`), both covers, `crop_factor` and `et0_coefficient` are equal every day.
  - `test_every_year_shifts_the_day_after_february` (L1): the adjusted day is 59 on 2003-02-28 and on 2003-03-01, 364 on 2003-12-31, and 60 on 2004-03-01.
  - `test_the_sowing_day_has_no_crop`: on the sowing day `days_after_sow == 0` and the covers are unchanged. The next day they are interpolated.
  - `test_an_autumn_calendar_has_no_cover_all_year` (L2, Y15): the maximum `crop_cover` over days 1–365 is `0.0` for (280, 200) and `0.9` for (90, 250).
  - `test_the_stress_zero_denominator_gives_zero` (L3, Y17): `ET_depletion_factor = 1`, `smc ≤ wp_m`. `Ks = 0`, `et0_coefficient = 0`, and the event is true. WSIMOD raises `ZeroDivisionError` at `land.py:1566`.
  - `test_stress_either_side_of_raw`: `root_zone_depletion` just below `RAW` gives `Ks = 1`, and at `RAW` it gives `max(0, (TAW − RAW)/((1 − p)·TAW))`, which equals 1 there.
  - `test_calendar_refusals` (parametrised, `match=`):
    - stage lengths differ (`"crop_factor_stages"`);
    - `crop_factor_stage_dates = [0, 300]` on dates past day 300 (`"crop_factor_stage_dates.*<first label>"`);
    - a duplicate pair of dates (`"divides by zero"`);
    - `sowing_day: 90.5` (`"sowing_day"`), while `90.0` is accepted;
    - `adsorption_nr_maxiter: -1` and `2.5`.
  - `test_step_cannot_leave_the_network_dates` (D7): `model.step` at `t = len(dates) * 86400` raises 3a's `ValueError`.
  - `test_growing_refusals` (parametrised, `match=`):
    - `depth` configured (`"depth"`, L17), and `depth` as an override (`"depth"`);
    - `wilting_point > field_capacity` and `=` (`"wilting_point"`, L4, Y44);
    - `area: 0`, and `total_porosity` not above `field_capacity` (inherited, D18).
  - `test_fertiliser_and_manure_data_are_needed_with_nitrate` (Y49): a growing surface on a network listing `nitrate` without `nhx-fertiliser` data is refused, `match="nhx-fertiliser"`, with `pollutants=True` and with `False`. Without `nitrate`, no such data are needed. With them, quantity-only lists them as ignored.
  - `test_a_growing_et0_coefficient_is_unread` (L22, Y39).
  - `test_growing_overrides_equal_wsimods`: each of the 31 names and `rooting_depth`, applied to WSIMOD's object with `apply_overrides` and to noodl's spec. The attributes, `capacity`, `field_capacity_m`, `wilting_point_m`, the calendar, `TAW` and `RAW` are equal, and the derived values equal construction's bitwise (spec, Loading and overrides).
  - `test_crop_factor_stages_padding_is_inert` (Y33, D16): two surfaces with 2 and 5 stages give a driver `(2, 5)`, and the shorter row's padding is 0. `initial_drivers` with a padding entry of 0.1 is refused at bind, naming the surface and index 2.
  - `test_the_crop_state_exists_in_quantity_only`: `state_labels("wsimod.days_after_sow")` holds `("land", "surfaces.g")` in both modes, starting at −1.
  - The harvest branch's dead store (L26) is not ported. The docstring of `calc_crop_cover` says so.
  - `test_config.py:415`: the `GrowingSurface` row becomes accepted. `IrrigationSurface`, `GardenSurface` and `VariableAreaSurface` stay refused with `"3c"` (D1).
  - Cases (D21):
    - `growing_no_nitrate`: one `GrowingSurface` (90/250, `ET_depletion_factor = 0.5`, five-point stages) and an `ImperviousSurface`, with pollutants `phosphate` and `temperature`, dry and wet deposition (straight into the tank), `days=540`, and `replay=("land",)`;
    - `growing_default_depletion` (`crash=("ZeroDivisionError", "land.py:1566")`): the default `ET_depletion_factor = 1` on a soil that dries under `et0 = 0.01`.
- [ ] **Step 2:** Run `test_growing.py`, `test_config.py` and `test_build.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. The `calc_crop_cover` decisions are D6's plus `depletion` (`dmax`), `unstressed` and `stress_floor` (`dmax(0.0, ·)`), and `stress_zero_denominator` (`(1 − p)·TAW == 0`, the guard: `Ks = 0`) (spec vocabulary). `ihacres` reads `crop_et0_coefficient()` through `get_climate()`.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS: `growing_no_nitrate` at parity in both modes with its replay bitwise, and the crash test passing. Add the cases to the listed-features test. Docs: the class, the calendar, its quirks (L1 naming the individual-files loader, Y42; L2, L23), the stress guard and the refusals. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): GrowingSurface with WSIMOD's crop calendar and cover`.

### Task 6: The pool on the surface: inputs, `pull_storage` and the flushing (`land.py:1293-1324`, `:1433-1710`, `:2267-2311`)

**Implementer:** most capable (`pull_storage`'s species split and guard, the double flush, the liquid adjustment's ratios).

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/growing.py`: the pool, `pull_storage`, `effective_precipitation_flushing`, `fertiliser`, `manure`, `adjust_vqip_to_liquid`, the deposition overrides, the nitrate validation, `initial_soil_storage`;
  - `day.py` (D4): `"wsimod.nutrient_pool"`, `_TRAILING`, and the pool drivers through `driver_columns` (D16);
  - `classes.py`;
  - `run.py`: `"pull_storage_no_din"`;
  - `engine.py`: sites;
  - `tests/verification/_wsimod_reference.py`: the capture of the pools and `_wsimod_store`'s `nutrient_pool` as `[[pool.storage[n] for n in "NP"] for pool in pools]`;
  - `tests/apps/wsimod/_replay.py`;
  - tests: `test_growing.py`, `test_build.py`, `test_config.py`;
  - docs.

**Interfaces:**
- Consumes: `NutrientPool` (Task 4), `GrowingSurface` (Task 5), `Surface.dry_/wet_deposition_to_tank` and `forcing` (Task 3), `Tank.pull_storage` (3a).
- Produces:

```python
class GrowingSurface:
    nutrient_pool: NutrientPool | None                    # D14; None without nitrate
    def pull_storage(self, vqip: Tensor, *, active: Tensor) -> Tensor                    # land.py:1433-1493
    def effective_precipitation_flushing(self, *, active) -> tuple[Tensor, Tensor]       # :1605-1647
    def fertiliser(self, *, active) -> tuple[Tensor, Tensor]                             # :1649-1679
    def manure(self, *, active) -> tuple[Tensor, Tensor]                                 # :1681-1710
    def adjust_vqip_to_liquid(self, vqip: Tensor, dep: Tensor, in_: Tensor, *, active) -> Tensor   # :1574-1603
    def dry_deposition_to_tank(self, vqip, *, active) -> Tensor                          # :2267-2288
    def wet_deposition_to_tank(self, vqip, *, active) -> Tensor                          # :2290-2311
NUTRIENT_POLLUTANTS = ("ammonia", "nitrite", "nitrate", "phosphate", "org-nitrogen", "org-phosphorus", "solids", "temperature")
# state "wsimod.nutrient_pool" (..., n_growing, 5, 2), rows (land, "surfaces.<s>.nutrient_pool"), unit kg
# drivers "wsimod.<Class>.nutrient_pool.<p>" (..., n, 2), columns ("N", "P"); "...fraction_dry_n_to_dissolved_inorganic" (..., n)
```

- [ ] **Step 1: Write the failing tests** in `test_growing.py` (`importorskip` where WSIMOD is driven):
  - `test_pull_storage_equals_wsimods`: 300 random tanks and pools, including `V = 0` (empty reply), a request above `V` (the cap), and pools larger than the tank in some column. The reply, the storage and the pools are equal.
  - `test_pull_storage_without_din_gives_no_species` (L5, Y18): a tank with no inorganic N and `V > 0`. WSIMOD raises `ZeroDivisionError` at `land.py:1460`. noodl gives `nitrate = ammonia = nitrite = 0` in the reply, the pool has still lost `DI.N · prop`, and `"pull_storage_no_din"` is true.
  - `test_fertiliser_manure_and_deposition_equal_wsimods`: a stub parent with 15 months of the 12 variables and one month of negative values (Review Focus 3). Over 400 days: storage, pools and each `(in_, out_)` are equal, with no NaN. The manure split is `fmdi`/`1 − fmdi`, and the dry deposition sends `P` to the adsorbed pool with the tank's phosphate `x · (0/P) = 0`.
  - `test_the_liquid_ratio_is_evaluated` (`:1595-1602`): with fertiliser, the tank's nitrate is `noy * area * (in_.N / dep.N)` with the ratio exactly `1.0`. With `dep.N = 0` the line is skipped. Each `adjust_vqip_to_liquid/N|P` decision carries `call` in the order dry, wet, fertiliser, manure.
  - `test_flushing_extracts_the_outflows_again` (L8): after `pull_storage` twice and `effective_precipitation_flushing`, `DI` equals WSIMOD's. It is smaller than `DI₀ − (ss + perc)` by the second extraction.
  - `test_the_initial_pool_comes_from_the_storage_and_initial_soil_storage` (`:1293-1324`): the six keys and their sums, in WSIMOD's order.
  - `test_initial_soil_storage_refusals`: a float `0.5` is refused (`"initial_soil_storage"`, L16, Y37; WSIMOD raises `TypeError` at `:1312`). A mapping lacking `org-phosphorus` is refused (`"org-phosphorus"`). `0` and `None` are accepted (falsy).
  - `test_nitrate_needs_the_eight_pollutants` (parametrised over each name, `match=<name>`; `solids` is L24).
  - `test_the_domain_refusals` (parametrised, at build and at bind): `rooting_depth`, `bulk_density`, `kfr` and `nfr` `≤ 0`; `cohesion`, `thetaupp` and `thetalow` `= 0`; `hsatINs ≤ 0`; `limpar < 0`; `field_capacity ≤ 0`.
  - `test_nutrient_pool_parameters_are_drivers_with_wsimods_defaults` (L20, Y24):
    - the template of `"wsimod.GrowingSurface.nutrient_pool.degrhpar"` is `[[7 * 1e-5, 7 * 1e-6]]` with columns `("N", "P")`;
    - a surface key `degrhpar` is refused in the configuration;
    - an override `{"degrhpar": ...}` is refused, naming `NutrientPool`.
  - `test_without_nitrate_there_is_no_pool`: no `"wsimod.nutrient_pool"` rows and no pool drivers, and dry deposition enters the tank whole.
- [ ] **Step 2:** Run `test_growing.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. The `din = 0` guard is the decision `S/pull_storage/no_din`, and the species go through safe denominators under its mask. `inflows += [effective_precipitation_flushing, fertiliser, manure]` (D20).
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Regenerate the catalogue. Docs: the pool, its state and drivers, the pool–tank gap (L8), and "WSIMOD needs initial inorganic N or a non-zero first-month input" (Y48).
- [ ] **Step 5:** Commit: `feat(wsimod): the growing surface's pool inputs, pull_storage and flushing`.

### Task 7: The soil factors, transformation, crop uptake and denitrification (`land.py:1744-1926`, `:2096-2152`)

**Implementer:** most capable.

**Files:**
- Modify: `src/noodl/apps/wsimod/growing.py`, `run.py` (D10), `engine.py`, `tests/apps/wsimod/test_growing.py`, docs.

**Interfaces:**
- Consumes: `NutrientPool.soil_pool_transformation`, `extract` (Task 4); `_pull_tracked` (D19); `pow_python`, `exp_python`.
- Produces:

```python
class GrowingSurface:
    def calc_temperature_dependence_factor(self, *, active) -> tuple[Tensor, Tensor]      # :1813-1831
    def calc_soil_moisture_dependence_factor(self, *, active) -> tuple[Tensor, Tensor]    # :1833-1856
    def soil_pool_transformation(self, *, active) -> tuple[Tensor, Tensor]                # :1744-1811
    def calc_crop_uptake(self, *, active) -> tuple[Tensor, Tensor]                        # :1858-1926
    def denitrification(self, *, active) -> tuple[Tensor, Tensor]                         # :2096-2152
    def _pull_tracked(self, out_: Tensor, *, active: Tensor) -> Tensor                    # D19
```

- [ ] **Step 1: Write the failing tests** (`importorskip` where WSIMOD is driven):
  - `test_the_soil_factors_equal_wsimods` (L12): `T` in `{-1, 0, 2.5, 4.9, 5.0, 5.000001, 15, 25}`.
    - `tdf(4.9) == 4.9 / 5`, `tdf(5.0) == 1.0`, `tdf(5.000001) == 2 ** ((5.000001 - 20) / 10)` and `tdf(15) == 2 ** (-0.5)`, with `tdf(15) < tdf(4.9)` asserted.
    - `smc` at `wp_m`, between, at `fc_m` and above (Review Focus 2): `smdf(fc_m) == satact` and `smdf(wp_m) == 0`.
    - The three-way `min(1, wp_comp, fc_comp)` takes the first of the smallest, with decisions `…/min:wp` and `…/min:fc`.
  - `test_soil_pool_transformation_equals_wsimods`: random pools and tanks.
    - An increase of exactly 0 takes the `else` branches (rule 6).
    - `nitrate + ammonia = 0`: WSIMOD raises `ZeroDivisionError` at `:1758`. noodl uses proportion 0, sending the change to ammonia, and records `"transformation_no_din"` (L6, Y18).
  - `test_a_clamped_pull_records_the_event` (L10): a transformation `out_` larger than the tank's column. `pull_pollutants` clamps, the returned `out_` is the clamped one, and `"nutrient_pool_not_tracking"` is true. With the tank above the request the event is false.
  - `test_calc_crop_uptake_equals_wsimods[spring|autumn]`: each cap binds on some draw. Autumn's `temp_func = max(0, min(1, (T - 5) / 20))` and `days_after_sow − 25`. `uptake_par + uptake2 ≤ 0` skips the shape. All the uptake leaves as nitrate. The extract order is P, then N.
  - `test_crop_uptake_from_an_empty_soil_is_zero` (L7): `V = 0` with a crop. WSIMOD raises `ZeroDivisionError` at `:1899` (with `ET_depletion_factor = 0.5`). noodl gives `N_max = P_max = 0`, and `"crop_uptake_dry_soil"` is true.
  - `test_denitrification_equals_wsimods`: each branch. At `smc == fc_m` the ratio branch gives `f == 1.0`; `smc / fc_m ≤ limpar` returns empty.
  - `test_gradcheck_on_the_factors_and_uptake`, away from 0 °C, 5 °C and the caps.
- [ ] **Step 2:** Run `test_growing.py`. Expected: FAIL.
- [ ] **Step 3:** Implement in WSIMOD's operation order. `** thetapow`, `** exppar`, `** 2` (`:1893`) and `2 ** (...)` use `pow_python`, and `exp(-uptake3 * d)` uses `exp_python`. Append the five functions to `processes` (D20).
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Docs: the factors' jump (L12, Y19), the nitrate-only removals (L9) and the tracking event (L10).
- [ ] **Step 5:** Commit: `feat(wsimod): soil factors, transformation, crop uptake and denitrification`.

### Task 8: Erosion and adsorption (`land.py:1928-2094`, `:2154-2265`)

**Implementer:** most capable (the unrolled Newton loop and its lagged test, erosion's operand order).

**Files:**
- Modify: `src/noodl/apps/wsimod/growing.py`, `run.py` (D10), `engine.py`, `tests/verification/_wsimod_parity.py` (`_UNDATED`, D10), `tests/apps/wsimod/test_growing.py`, docs.

**Interfaces:**
- Consumes: `NutrientPool.get_erodable_P`, `erode_P`, `extract`, `receive` (Task 4); `CalendarDay.erosion_sine` (Task 5); `log_python`, `log10_python`, `sin_python` (Task 1); `exp_python`, `pow_python`.
- Produces:

```python
class GrowingSurface:
    adsorption_nr_maxiter: int                                                # structural (D17)
    def erosion(self, *, active) -> tuple[Tensor, Tensor]                     # :1928-2094; adds to ie/ss/perc
    def adsorption(self, *, active) -> tuple[Tensor, Tensor]                  # :2154-2265; Y25 unrolled
```

- [ ] **Step 1: Write the failing tests** (`importorskip` where WSIMOD is driven):
  - `test_erosion_equals_wsimods`. Draws cover:
    - rain `≤ 5` and `> 5` mm (`pd = precipitation * 1e3`);
    - `ie.volume` 0 and positive;
    - `erodingflow` 0, in `(0, 4]` and above 4;
    - an erodable pool shorter than the request.

    The three outflow VQIPs (their `solids`, `phosphate` and `org-phosphorus`), the pools and `in_` are equal. `erodingflow = 0` returns empty. The short pool records `"erosion_pool_short"` on exactly the draws where WSIMOD prints "weird nutrients" (captured with `capsys`).
  - `test_erosion_uses_the_raw_day_in_its_sine`: on 2003-03-01, the sine is `math.sin(2 * 3.141592653589793 * ((60 - 70) / 365))`, with the raw day 60, not the adjusted 59.
  - `test_adsorption_equals_wsimods` (parametrised):
    - the empty pool (`pool_ == 0`);
    - the shortcut (`ads.P = 0`), where `"adsorption_shortcut"` is true and WSIMOD prints its warning;
    - Newton exits by residual, by step, and by `maxiter`, with `adsorption_nr_maxiter = 2`;
    - the `xn_1 ≤ 0` clamp to `1e-10`, from crafted pools with `kfr = 1e4`;
    - adsorption and desorption;
    - `abs(equi − conc_sol) ≤ 1e-6`, which moves nothing.

    The pools, the tank, `(in_, out_)` and every iteration's decisions are recorded with `…/newton:<j>/residual|step|positive`.
  - `test_the_newton_test_lags_one_iteration`: on a crafted case, WSIMOD's loop and noodl's both run `k + 1` passes, where the residual was below `limit` after `k`. The count is asserted from the decisions.
  - `test_a_short_pool_records_freundlich_flow_adjusted` (L11): a request above `DI.P` (adsorption) or `ads.P` (desorption) gives the event true, and WSIMOD prints nothing.
  - `test_a_batch_of_newton_exits_equals_its_single_calls` (Review Focus 1): four instances exit by residual, by step, by `maxiter`, and by the shortcut. Each is bitwise equal to its single call.
  - `test_gradcheck_on_one_adsorption_and_one_erosion_call`: a fixed iteration count, away from the guards, with respect to `kfr`, `nfr`, `kadsdes`, the pools, `erodibility`, `sreroexp` and `srfilt`.
  - `test_the_adsorption_limit_has_zero_gradient`: the gradient with respect to `adosorption_nr_limit` is exactly 0.
- [ ] **Step 2:** Run `test_growing.py`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - Each power uses `pow_python`, `log10` uses `log10_python`, `sin(slope / 100)` uses `sin_python`, `log(conc_sol)` and `log(kfr)` use `log_python`, and `exp` uses `exp_python`.
  - The Newton loop runs `adsorption_nr_maxiter` iterations under `running = active & ~shortcut & (abs(fxn) > limit)`. The `break` and `xn_1 ≤ 0` are masks inside the iteration. A stopped instance keeps `xn`, `xn_1` and `fxn` (`write`).
  - Append `erosion` and `adsorption` to `processes` (D20).
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Docs: erosion's quirks (eroded masses bypass the tank; raw day), the Newton lag, the shortcut and the two events.
- [ ] **Step 5:** Commit: `feat(wsimod): erosion and Freundlich adsorption with an unrolled Newton loop`.

### Task 9: The growing cases, the land replay and the print admission (Y3, Y35, Y41, Y45, Y48)

**Implementer:** most capable (cases that reach every branch within the default horizon, and the admission's measurement).

**Files:**
- Modify:
  - `tests/verification/_wsimod_cases.py`: the cases below, and `Case.crash` on the crash cases;
  - `tests/verification/_wsimod_reference.py`: the capture of the factors, `crop_factor` and `et0_coefficient` before and after each `Land.run`;
  - `tests/apps/wsimod/_replay.py`;
  - `tests/verification/_wsimod_parity.py`: `store_bounded_prints`, D12;
  - `tests/verification/test_wsimod_parity.py`: `GROWING_PRINT_C`, the branch test, L8, L9;
  - `tests/verification/test_wsimod_replay.py`;
  - docs: the verification rows.

**Interfaces:**
- Consumes: every growing function (Tasks 5–8), `wsimod_until_crash` and `Case.crash` (Task 5), `Case.slow_days` (3b).
- Produces: `assert_matches_wsimod(..., store_bounded_prints: Mapping[str, float] = {})` (D12); `GROWING_PRINT_C: int`; the cases `growing_spring`, `growing_autumn`, `growing_decay`, `growing_no_din`, `growing_zero_n`, `growing_nitrite_only` and `growing_empty_soil`.

- [ ] **Step 1: Write the cases** (spec, Verification 3; D21):
  - `growing_spring`: one `GrowingSurface` (sowing 90, harvest 250, `ET_depletion_factor = 0.5`, five-point `crop_factor_stages`), with fertiliser and manure data, dry and wet deposition, `initial_soil_storage`, and the eight pollutants. It runs for `days=540` with `slow_days=1096`, and `replay=("land",)`. The forcing has rain above 5 mm with runoff, dry spells that stress the crop, and soil temperatures below 0, in `(0, 5]` and above 5 °C, all within the first 540 days.
  - `growing_autumn`: sowing 280, harvest 200, a season crossing the new year, and the autumn uptake temperature function.
  - `growing_decay`: the full 12 pollutants and `decays` on ammonia and nitrite. `"nutrient_pool_not_tracking"` is true on some day.
  - The crash cases (D13):
    - `growing_no_din`: `("ZeroDivisionError", "land.py:1460")`, a zero-N tank, no deposition, rain on day 0;
    - `growing_zero_n`: `(…, "land.py:1758")`, a zero-N tank with zero first-month inputs, crashing on day 0;
    - `growing_nitrite_only`: `(…, "land.py:1758")`;
    - `growing_empty_soil`: `(…, "land.py:1899")`, `ET_depletion_factor = 0.5`, an empty soil and a sown crop.
- [ ] **Step 2: Write the tests:**
  - The generic `test_case_matches_wsimod[<case>-<mode>]`, `test_case_matches_wsimod_on_its_slow_days[growing_spring-*]` (slow), `test_replay_matches_wsimod[<case>-land]`, `test_quantity_only_equals_the_pollutant_run_volumes[<case>]`, the mass-balance test, and the parts test of Task 1 all run on the three cases. The parity calls pass `store_bounded_prints={"land": GROWING_PRINT_C}` where measured.
  - `test_growing_spring_reaches_every_branch` (`slow`). From the decisions, on the 540-day prefix and on the full horizon, each transformation, uptake, erosion, denitrification and adsorption site of the vocabulary is active at least once, with both outcomes where it has two, and so is each factor branch.
  - `test_the_pools_are_flushed_twice` (L8, Y45): on `growing_spring`, noodl's daily tank–pool gap (`tank phosphate − DI.P − DO.P`, and the N analogue) equals WSIMOD's from the replay capture, bitwise. The test asserts that the gap is non-zero; it asserts no hard-coded ratio.
  - `test_the_tank_columns_keep_their_sign` (L9): on `growing_spring` and `growing_decay`, it records whether any tank column went negative, asserts the same answer from WSIMOD's capture, and writes that answer into the assertion message.
  - `test_growing_spring_spans_the_leap_shift` (L1): the run's dates include 2004-02-29 and two common-year 1 March days, and `crop_factor` equals WSIMOD's on each.
  - `test_case_runs_where_wsimod_crashes[<case>]` (D13) over the four crash cases and `growing_default_depletion`.
  - `test_the_store_bounded_prints[<case>]`: a mapped case's admitted lines satisfy D12's bound. A case whose measured count is zero is not mapped, and its parity call proves zero lines.
- [ ] **Step 3: Measure (D12).** For each case, compute `max |x| / (ε S)` over its printed lines and record it in the progress file. Set `GROWING_PRINT_C` to the ceiling of the maximum, with the per-case values in its comment. If it exceeds 4, stop and report the cause before setting it.
- [ ] **Step 4:** Run the wsimod subset, and `-m slow` on `test_wsimod_parity.py test_wsimod_replay.py`. Expected: PASS, bitwise. Run `--durations=30` and record the 3c items; cut a default horizon if any is among the 30 slowest. Add the cases to the listed-features test. Docs: the verification rows with measured values, the admission and its `c`.
- [ ] **Step 5:** Commit: `test(wsimod): growing-surface cases against WSIMOD, every branch and the crash rows`.

### Task 10: Batches, stepping, state and quantity-only structure (3c-i)

**Implementer:** standard.

**Files:**
- Modify: `tests/apps/wsimod/test_run.py`, `tests/apps/wsimod/test_build.py`.

**Interfaces:**
- Consumes: every 3c-i class, driver and state key (Tasks 2–9); `initial_state` and `initial_drivers` (3a, D4).

- [ ] **Step 1: Write the tests:**
  - `test_a_batch_of_four_3c_parameter_sets_equals_its_single_runs[growing_spring|surface_deposition]` (first 100 days). Four instances with different `kfr`, `uptake1`, `ET_depletion_factor`, `crop_factor_stages`, a pool rate, and a fertiliser forcing column. On some day the decisions show two instances taking different Newton iteration counts and different uptake caps (Review Focus 1). Each instance equals its single run.
  - `test_step_by_step_equals_wsimod_run[growing_spring]` (60 days): the new state keys are carried through `model.step`.
  - `test_the_3c_state_round_trips`. `initial_state(values=...)` set by label on `"wsimod.nutrient_pool"` (D4's nested form), `"wsimod.days_after_sow"`, `"wsimod.crop_cover"` and `"wsimod.ground_cover"` reaches day 0.
  - `test_a_warm_start_mid_season_crosses_the_harvest` (Review Focus 4): run `growing_spring` for 300 days (to 2004-07-26, mid-season, `days_after_sow > 0`), then warm-start a second model at day 300 from the last state. Its next 60 days cross the harvest (raw day 250, 2004-09-06, run day 341). The continuation equals days 300–360 of one 360-day run, `torch.equal`, and `days_after_sow == −1` after the harvest day.
  - `test_quantity_only_volumes_never_depend_on_3c_pollutant_values`: on `growing_spring` and `surface_deposition`, vary every driver a quantity-only run lists as ignored (the pool rates, uptake, erosion, adsorption, denitrification, and the deposition, fertiliser and manure forcing). The volumes are `torch.equal`.
  - `test_quantity_only_lists_the_3c_parameters`: those names are in `model.ignored_by_quantity_only`. `ET_depletion_factor`, `rooting_depth` and `crop_factor_stages` are not.
  - `test_wsimod_is_never_imported` re-runs unchanged.
- [ ] **Step 2:** Run them. Expected: PASS. A failure is a defect, fixed in its task's code with a test there.
- [ ] **Step 3:** Commit: `test(wsimod): batches, stepping and state through the growing surface`.

### Task 11: Gradients (3c-i)

**Implementer:** most capable.

**Files:**
- Modify: `tests/verification/test_wsimod_gradients.py`.

**Interfaces:**
- Consumes: `wsimod_run(..., diagnostics={"decisions": True})`, `CASES`, the 3a `Spec` table and the kink rule (W92/W93).

- [ ] **Step 1: Write the tests.** Extend `test_gradient_matches_central_differences[<parameter>]` (relative step `1e-6`, agreement `1e-6`, the ±h decisions identical, the kink rule) with these `Spec` entries, each away from the spec's kinks and jumps (Differentiability):

  | Parameter | Case, horizon | Scalar |
  |---|---|---|
  | `crop_factor_stages` (one stage), `ET_depletion_factor`, `rooting_depth` | `growing_no_nitrate`, 120 days | the land's out-arc volume and the final soil storage |
  | `uptake1`, `uptake3`, `denpar`, `hsatINs`, `exppar`, `satact`, `thetapow` | `growing_spring`, 120 days | the final pool `DI.N` and the out-arc nitrate |
  | `kfr`, `nfr`, `kadsdes` | `growing_spring`, 120 days | the final `ads.P` and the out-arc phosphate |
  | `erodibility`, `sreroexp`, `srfilt` | `growing_spring`, a window with rain above 5 mm | the out-arc solids and org-phosphorus |
  | `nutrient_pool.minfpar`, `.immobdpar`, `.fraction_manure_to_dissolved_inorganic` (N column) | `growing_spring`, 120 days | the final `DI.N` |
  | `wsimod.nhx-fertiliser`, `wsimod.srp-manure`, `wsimod.noy-dry` (one month) | `growing_spring`, 120 days | the final tank nitrate |
  | `pollutant_load` (phosphate) | `surface_deposition` | the out-arc phosphate |
  | the Land's `temperature` forcing | `growing_spring`, a window in 6–20 °C | the final `DI.N`, through `tdf` |

  `test_the_adsorption_limit_gradient_is_zero_through_a_run`: `adosorption_nr_limit`'s gradient on `growing_spring` is exactly 0. Mark slow any entry whose run exceeds 10 s, as 3a's `SLOW` set does.
- [ ] **Step 2:** Run them, and `-m slow`. Expected: PASS. Record the worst relative agreement per parameter in the progress file.
- [ ] **Step 3:** Commit: `test(wsimod): gradients through the growing surface and its pool`.

> **3c-ii boundary (D1, Y31).** Every commit above stands alone as 3c-i. Tasks 12–16 can move to a follow-up PR without rework.

### Task 12: The surface `data_input_dict` override and the override refusals (Y21, Y22)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/config.py`: `_apply_overrides`, where a surface's `data_input_dict` is a path or a mapping and the node form and the surface `filename` are refused; `unread`, D15;
  - `classes.py`: `"Surface"` overrides gain `data_input_dict`;
  - `land.py`: the forcing check reads the merged data;
  - `tests/verification/_wsimod_cases.py`: case `surface_overrides`, written with `_forcing_file`'s directory for its path override;
  - tests: `tests/apps/wsimod/test_config.py`, `test_land.py`;
  - docs: `docs/formats/wsimod.md`.

**Interfaces:**
- Consumes: `SurfaceSpec.constructed_variables`/`constructed_load` (Task 3), `_read_csv`, `_inline_data` (3a), the growing overrides (Task 5).
- Produces: `_apply_overrides(..., surface=True)` replaces `SurfaceSpec.data` with the override's data (`land.py:413-421`).

- [ ] **Step 1: Write the failing tests:**
  - `test_a_surface_data_override_replaces_its_data[path|mapping]` (`importorskip`, `tmp_path`): the replaced `SurfaceSpec.data` equals WSIMOD's `data_input_dict` after `add_overrides` (keys compared as datetimes). A relative path opens from the working directory (`monkeypatch.chdir`).
  - `test_surface_override_refusals` (parametrised, `match=`):
    - a node's `data_input_dict` override (`"data_input_dict"`; `Node.apply_overrides` consumes only `filename`, `nodes.py:92-100`);
    - a surface `filename` override (`"filename"`);
    - a surface `data_input_dict` of `None` or `3` (`"data_input_dict"`; WSIMOD raises `ValueError`, `:419-421`);
    - an override that drops `nhx-dry` from a surface built with it (`"nhx-dry"`, L15);
    - an override that drops fertiliser data from a growing surface (`"nhx-fertiliser"`).
  - `test_deposition_data_added_by_override_are_unread` (L15): the variables are listed (D15), and the run equals the run without the override.
  - `test_a_load_override_without_a_load_is_unread` (L14): the surface's inflows hold no `simple_deposition`, `"<land>.surfaces.<s>.pollutant_load"` is in `unread`, and the results equal those without the override. A load override on a surface with a load merges per key and is live.
  - Replace 3a's `REFUSALS` row for the surface `data_input_dict` override (`"3c"`).
  - Case `surface_overrides`: data overrides as an absolute path and as a mapping, the two `pollutant_load` overrides, and every `GrowingSurface` override including `rooting_depth` and the calendar. `replay=("land",)`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS: `surface_overrides` at parity and its replay bitwise. Add it to the listed-features test. Docs: the override, its working-directory path (X27), and the refused node and `filename` forms.
- [ ] **Step 5:** Commit: `feat(wsimod): surface data_input_dict overrides`.

### Task 13: `IrrigationSurface` and `Land.apply_irrigation` (`land.py:2314-2381`, `:192-198`)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/growing.py`: `IrrigationSurface`;
  - `land.py`: `Land.apply_irrigation`, `store_paths_of`;
  - `classes.py`: `_CONSTRUCTORS`, `_OVERRIDES` (`irrigation_coefficient`), `SURFACE_SCHEMAS`, `OUT_OF_SCOPE`;
  - `day.py`: `"wsimod.surface_climate"` (D4, D5);
  - `engine.py`: sites;
  - `tests/verification/_wsimod_reference.py`: the capture of `Land.apply_irrigation` with its pulls and replies, and `surface_climate`;
  - `tests/apps/wsimod/_replay.py`;
  - `tests/verification/_wsimod_cases.py`: `irrigation`, `irrigation_early` (D22), `irrigation_without_nutrients` (`crash=("KeyError", "land.py:2373")`); `surface_overrides` gains an `IrrigationSurface` with an `irrigation_coefficient` override;
  - tests: `tests/apps/wsimod/test_growing.py`, `test_land.py` (replace `test_apply_irrigation_does_nothing`);
  - docs.

**Interfaces:**
- Consumes: `GrowingSurface` and the pool (Tasks 5–6), `Node.pull_distributed` (3a), D6's `crop` decision.
- Produces:

```python
class IrrigationSurface(GrowingSurface):
    surface_class = "IrrigationSurface"; is_irrigated = True
    def irrigate(self, *, active: Tensor) -> None          # land.py:2347-2381; pool update only with nitrate (Y20)
class Land:
    def apply_irrigation(self, *, active: Tensor) -> None  # every IrrigationSurface.irrigate in surface order
# state "wsimod.surface_climate" (..., n_climate, 2), columns evaporation, precipitation, unit m3
# driver "wsimod.IrrigationSurface.irrigation_coefficient" (..., n), default 0.1
IRRIGATION_TYPES = ("River", "Node", "Groundwater", "Reservoir")
```

- [ ] **Step 1: Write the failing tests:**
  - `test_irrigate_equals_wsimods` (`importorskip`): a WSIMOD `IrrigationSurface` whose parent has a scripted `pull_distributed`. It covers days before sowing, the sowing day (no irrigation), dry days with demand, saturated days (`cmd ≤ FA`: demand 0, and a zero-volume pull is still made), and a supply carrying nutrients. Storage, pools and the pull requests are equal.
  - `test_irrigation_without_nitrate_leaves_the_pool_alone` (L13, Y20).
  - `test_irrigated_nitrite_enters_the_tank_and_not_the_pool`.
  - `test_the_climate_is_carried` (D5): `apply_irrigation` before `Land.run` reads the previous day's flows (0 on day 0), and `state_labels("wsimod.surface_climate")` holds `("land", "surfaces.irr")`.
  - `test_irrigation_coefficient_override_equals_wsimods`.
  - Cases:
    - `irrigation`: an `IrrigationSurface` fed by a `River`, a `Groundwater` and a `Reservoir` through the `Land`'s in-arcs (spec, Verification 3), with `replay=("land",)`;
    - `irrigation_early`;
    - `irrigation_without_nutrients` (pollutants `phosphate`, `temperature`). Its crash test runs through D13's parametrisation.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. The decisions are `irrigate/crop`, `…/deficit` (`dmax`), `…/demand` (`> FA`) and `…/saturated` (`cmd ≤ FA`). `irrigate`'s return is not summed (`:197-198`).
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS: the cases at parity, the replay bitwise, and the balance closing through the arcs. Docs: irrigation, Y20, and the carried climate.
- [ ] **Step 5:** Commit: `feat(wsimod): IrrigationSurface and Land.apply_irrigation`.

### Task 14: `GardenSurface` and the `Demand` garden link (`land.py:2384-2437`, `:120-128`, `demand.py:78-117`, `:250-301`)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/growing.py`: `GardenSurface`;
  - `land.py`: D24, `unread` (D15);
  - `demand.py`: docstrings at `:253`;
  - `classes.py`;
  - `day.py`: `surface_climate` rows for gardens;
  - `tests/verification/_wsimod_reference.py`: the capture of the `("Demand", "Garden")` handler calls and replies;
  - `tests/verification/_wsimod_cases.py`: case `garden`;
  - tests: `tests/apps/wsimod/test_growing.py`, `test_land.py` (`test_garden_demand_is_zero_without_a_garden_surface` stays), `test_demand.py`;
  - docs.

**Interfaces:**
- Consumes: `IrrigationSurface`'s climate register (Task 13), `ResidentialDemand.get_garden_demand` (3a).
- Produces:

```python
class GardenSurface(GrowingSurface):
    surface_class = "GardenSurface"; is_garden = True
    def calculate_irrigation_demand(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor   # :2396-2423
    def receive_irrigation_demand(self, vqip: Tensor, *, active: Tensor) -> Tensor                   # :2425-2437
```

- [ ] **Step 1: Write the failing tests:**
  - `test_garden_handlers_equal_wsimods` (`importorskip`): `calculate_irrigation_demand` gives `max(E − P, 0)`, set to 0 when `cmd ≤ FA`. `receive_irrigation_demand` forces the push in, and its reply is equal.
  - `test_the_last_garden_takes_the_handlers` (L18, Y28): with `g1`, `g2`, the handlers are `g2`'s, and `"land.surfaces.g1.garden_handlers"` is in `unread`.
  - `test_the_garden_water_is_at_zero_degrees` (L19, Y29): a push at 0 °C into a soil at 11.5 °C cools the soil, as in WSIMOD.
  - `test_the_garden_demand_is_yesterdays`: days 0 and 1 answer 0, then the previous day's deficit times `gardening_efficiency` (the spec's run: 40.0 × 0.42 = 16.8).
  - Case `garden`: a `ResidentialDemand` through a `Distribution` to a `Land` with two `GardenSurface`s and a `PerviousSurface`, and a second `Land` with no garden on the same demand, which answers 0. `replay=("land", "land_plain")`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement D24.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Docs: the garden link, its quirks (L18, L19, yesterday's demand) and the `NonResidentialDemand` note.
- [ ] **Step 5:** Commit: `feat(wsimod): GardenSurface and the Demand garden link`.

### Task 15: `VariableAreaSurface` (`land.py:2440-2461`)

**Implementer:** standard.

**Files:**
- Modify: `src/noodl/apps/wsimod/growing.py`, `classes.py` (`OUT_OF_SCOPE` keeps only `EnfieldFoulSewer`; `current_surface_area` is a constructor keyword, refused as an override, Y50), `tests/verification/_wsimod_cases.py` (case `variable_area`), `tests/apps/wsimod/test_growing.py`, `test_config.py`, docs.

**Interfaces:**
- Consumes: `PerviousSurface.get_climate()` (Task 5), erosion's unscaled rain (Task 8).
- Produces:

```python
class VariableAreaSurface(GrowingSurface):
    surface_class = "VariableAreaSurface"
    def get_climate(self) -> tuple[Tensor, Tensor]   # :2455-2459: P, E * et0_coefficient, each * current_surface_area / area
# driver "wsimod.VariableAreaSurface.current_surface_area" (..., n), default 0, unit m2
```

- [ ] **Step 1: Write the failing tests:**
  - `test_get_climate_equals_wsimods` (`importorskip`).
  - `test_a_default_variable_area_surface_gets_no_rain`: its precipitation volume is `0.0`.
  - `test_erosion_reads_the_unscaled_rain`: with half the area, erosion's `pd` is the Land's.
  - `test_current_surface_area_is_refused_as_an_override` (`match="current_surface_area"`).
  - `test_out_of_scope_holds_only_the_enfield_sewer`.
  - Case `variable_area`: one surface at the default 0 and one at half its area. `replay=("land",)`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset. Expected: PASS. Docs.
- [ ] **Step 5:** Commit: `feat(wsimod): VariableAreaSurface`.

### Task 16: 3c-ii structure and gradients

**Implementer:** most capable (gradients).

**Files:**
- Modify: `tests/apps/wsimod/test_run.py`, `tests/verification/test_wsimod_gradients.py`.

**Interfaces:**
- Consumes: Tasks 12–15.

- [ ] **Step 1: Write the tests:**
  - `test_a_batch_of_four_3c_parameter_sets_equals_its_single_runs` gains `[irrigation|garden|variable_area]`, with different `irrigation_coefficient`, `gardening_efficiency` and `current_surface_area`.
  - `test_step_by_step_equals_wsimod_run` gains `[irrigation|garden]`.
  - `test_the_climate_state_round_trips`: `initial_state(values={"wsimod.surface_climate": {("land", "surfaces.irr"): {"evaporation": 3.0, "precipitation": 1.0}}})` sets day 0's garden demand or irrigation to `max(3 − 1, 0) …`, equal to a run warm-started from a state holding those flows.
  - Gradient `Spec` entries:
    - `irrigation_coefficient` on `irrigation` (the soil storage);
    - `current_surface_area` on `variable_area` (the out-arc volume);
    - `gardening_efficiency` on `garden` (the garden soil storage);

    each away from `max(E − P, 0)` and the `FA` switches.
- [ ] **Step 2:** Run them, and `-m slow`. Expected: PASS. Record the agreement figures.
- [ ] **Step 3:** Commit: `test(wsimod): batches, state and gradients through irrigation, gardens and variable areas`.

### Task 17: Documentation, benchmark, final checks

**Implementer:** standard.

**Files:**
- Modify: `docs/applications/wsimod.md`, `docs/formats/wsimod.md`, `docs/verification.md`, `tests/data/wsimod/README.md`, `benchmarks/wsimod_run.py`, `docs/catalogue/wsimod.md` (regenerated), `AGENTS.md` (only the wsimod row's measured-time sentence).

- [ ] **Step 1: Docs** (spec, Documentation). Complete `wsimod.md`:
  - what is ported: the four surface classes, deposition, the pool, irrigation and the garden;
  - sections for deposition, the growing surface (the call-order table, the calendar, the pool, the processes), irrigation, the garden link and the variable-area surface;
  - state and drivers;
  - no "until 3c";
  - `config.yml` live and `config_no_deposition.yml` retired;
  - the refusals;
  - "Where WSIMOD is silent or wrong": rows L1–L25, with L21 replacing the `residue` row;
  - the kinks and jumps;
  - the Limitations line: "every WSIMOD 0.8.1 class in `nodes/` and `arcs/` is ported except `EnfieldFoulSewer` and the never-ported list";
  - the verification rows with measured values;
  - the pool–tank gap for users (L8, L9), with `"nutrient_pool_not_tracking"` to watch.

  `formats/wsimod.md`:
  - the four classes and their keys;
  - the surface variables and monthly dates;
  - the `data_input_dict` override;
  - the `GrowingSurface` and `IrrigationSurface` overrides;
  - `initial_soil_storage`.

  `docs/verification.md` gains the new rows. Regenerate the catalogue.
- [ ] **Step 2: Benchmark.** `benchmarks/wsimod_run.py`:
  - the oxford pollutant row runs on `config.yml`;
  - it gains `growing_spring` (built from `CASES`) at batch sizes 1, 10 and 100, no grad and grad, with no asserts (W8);
  - the docstring names the per-instance Python calls (about 30 per growing surface per day, plus 3 per Newton iteration).

  Run it once and record the results beside Task 0's medians.
- [ ] **Step 3: Final checks.** Run each of these. Expected: all pass.
  - `ruff check .` (unpiped);
  - the full `python -m pytest` with coverage (≥ 90 %);
  - `python -m pytest -q -n 4 --no-cov -m slow tests/apps/wsimod tests/verification/test_allocation_wsimod_*.py tests/verification/test_wsimod_*.py`;
  - `python -m mkdocs build --strict`;
  - `git grep -n 3c src tests scripts docs/applications docs/formats`, which shows no refusal or "until 3c" text (history in `docs/superpowers` is excluded).

  Record the wsimod row's default wall time against the 10-minute budget, and update AGENTS.md's measured-time sentence.
- [ ] **Step 4:** Commit: `docs(wsimod): the land processes, verification and benchmark (3c)`.

## Defect table: each row and its test

| Row | noodl | Test (task) |
|---|---|---|
| L1 leap test always true | reproduced | `test_every_year_shifts_the_day_after_february` (5); `test_growing_spring_spans_the_leap_shift` (9) |
| L2 unsorted calendar | reproduced | `test_an_autumn_calendar_has_no_cover_all_year`, `test_calc_crop_cover_equals_wsimods_every_day` (5); `growing_autumn` (9) |
| L3 `Ks` zero denominator | `Ks = 0`, event | `test_the_stress_zero_denominator_gives_zero` (5); `growing_default_depletion` crash (5, 9) |
| L4 `TAW ≤ 0` | refused | `test_growing_refusals[wilting_point>|=]` (5) |
| L5 `pull_storage` `din = 0` | species 0, event | `test_pull_storage_without_din_gives_no_species` (6); `growing_no_din` (9) |
| L6 transformation `nitrate + ammonia = 0` | proportion 0, event | `test_soil_pool_transformation_equals_wsimods` (7); `growing_zero_n`, `growing_nitrite_only` (9) |
| L7 uptake `V = 0` | 0, event | `test_crop_uptake_from_an_empty_soil_is_zero` (7); `growing_empty_soil` (9) |
| L8 double flush | reproduced | `test_flushing_extracts_the_outflows_again` (6); `test_the_pools_are_flushed_twice` (9) |
| L9 nitrate-only removals, bypasses | reproduced, event | `growing_spring`, `growing_decay`, `test_the_tank_columns_keep_their_sign` (9) |
| L10 dead tracking check | event | `test_a_clamped_pull_records_the_event` (7) |
| L11 reversed Freundlich test | event | `test_a_short_pool_records_freundlich_flow_adjusted` (8) |
| L12 `T/5` factor | reproduced | `test_the_soil_factors_equal_wsimods` (7) |
| L13 irrigation reads five keys | pool only with nitrate | `test_irrigation_without_nitrate_leaves_the_pool_alone`, `irrigation_without_nutrients` (13) |
| L14 load override ignored, shared default | per surface, unread | `test_a_load_override_without_a_load_is_unread`, `surface_overrides` (12) |
| L15 data override not rebuilt | unread; removal refused | `test_deposition_data_added_by_override_are_unread`, `test_surface_override_refusals` (12) |
| L16 float `initial_soil_storage` | refused | `test_initial_soil_storage_refusals` (6) |
| L17 `depth` on a growing surface | refused | `test_growing_refusals[depth]` (5) |
| L18 two gardens | last one; unread | `test_the_last_garden_takes_the_handlers`, `garden` (14) |
| L19 garden water at 0 °C | reproduced | `test_the_garden_water_is_at_zero_degrees`, `garden` (14) |
| L20 pool parameters unreachable | drivers; refused as configuration | `test_nutrient_pool_parameters_are_drivers_with_wsimods_defaults` (6) |
| L21 `allocate_residue` | not ported; data refused | `test_surface_data_refusals[residue]` (3) |
| L22 overwritten `et0_coefficient` | unread | `test_a_growing_et0_coefficient_is_unread` (5) |
| L23 `quick_interp` wrap and crash | wrap reproduced; crash refused | `test_quick_interp_equals_wsimods`, `test_calendar_refusals` (5) |
| L24 erosion needs `solids` | refused | `test_nitrate_needs_the_eight_pollutants[solids]` (6) |
| L25 dry without wet | refused | `test_surface_data_refusals[nhx-wet]` (3) |
| L26 dead store | not ported | the `calc_crop_cover` docstring (5) |

## Self-review

- **Spec coverage, by section.**
  - Goal and Scope › In: deposition (3); `GrowingSurface` (5–8); `NutrientPool` (4); `IrrigationSurface` and `Land.apply_irrigation` (13); `GardenSurface` and the `Demand` handlers (14); `VariableAreaSurface` (15); `Tank.pull_pollutants` (1); the surface `data_input_dict` override (12).
  - Scope › Out: residue (3, L21); `NutrientPool.apply_overrides` (6, L20); `RiparianBuffer` (1, Y10); `change_impervious_area` (not touched).
  - The refusal table, each row: `classes.py:416-420` (5, 13–15); `config.py:332-339` (12); the `land.py` deposition rows (2, 3); the garden rows (14); `apply_irrigation` (13); `storage.py:896`, `:1079-1085` (1); the docs, fixtures and script rows (2, 17); the test rows (1, 2, 3, 5, 12). Task 17's grep closes the table.
  - Approach rules 1–6: Global Constraints; D3 (1); D7 (5); the structural switches (D9, D17, D20); the `(5, 2)` pool (4, D4); the unrolled Newton loop (8); integer zeros (4, 7).
  - The equations, in sections: deposition (3); construction (5, 6); call order (5–8, D20; pinned by `growing_spring`, 9); `calc_crop_cover` and `quick_interp` (5); `pull_storage` (6); flushing, fertiliser and manure (6); transformation, factors, uptake and denitrification (7); erosion and adsorption (8); irrigation (13); garden (14); variable area (15).
  - The rest:
    - State (4, 5, 6, 13; D4, D5);
    - Inputs (3, 5, 6, 13, 15; D8, D16, D17);
    - Mass balance (1, D11; the parts tests in 1 and 9);
    - Interaction › Classes (D2), Validation (3, 5, 6, 12; D18), Loading (5, 12), Vocabulary and events (each task; D10), the harness (3, 5, 6, 9, 13, 14), the demos (2);
    - Defects (the table above);
    - Differentiability (gradchecks in 1, 4, 5–8; the gradient tasks 11 and 16; docs in 17);
    - Verification 1 (4–8, 13–15), 2 (2, 9), 3 (3, 5, 9, 12–15), 4 (11, 16), refusals and structure (10, 16);
    - Delivery (D1);
    - Documentation (each task; 17).
- **Rulings.**
  - Scope and approach: Y1/Y9 (all tasks, grep in 17); Y2 (Global Constraints); Y3 (9 and the per-task cases); Y4 (the defect table); Y5/Y12 (5, 6, 13); Y6 (start point); Y7/Y31 (D1); Y8 (tiers).
  - Y10–Y20: Y10 (1); Y11 (1, D11); Y13/Y42 (5, 17); Y14 (5, D7); Y15 (5, 9); Y16 (6, 7, 9); Y17/Y44 (5); Y18/Y46 (6, 7); Y19 (7); Y20 (13).
  - Y21–Y30: Y21 (3, 12, D9); Y22 (12); Y23 (3, D8); Y24 (6); Y25 (8); Y26 (1, D3); Y27 (5, 6); Y28 (14); Y29 (14); Y30 (2).
  - Y32–Y40: Y32 (5, D23); Y33 (5, D16); Y34 (D10); Y35/Y41 (9, D12); Y36 (3); Y37 (6); Y38 (2, 3); Y39 (5); Y40 (9, `growing_decay`).
  - Y43–Y50: Y43 (3); Y45 (9); Y47 (5, D7); Y48 (6 docs, 9); Y49 (5, `growing_no_nitrate`); Y50 (Global Constraints: the end-of-day rule and torch.equal; Y50's `current_surface_area` clause in 15).
- **Type consistency.**
  - `NutrientPool.extract(pool, amount, *, active, order)` is used by 6, 7 and 8 with the pool indices `FAST…ADS` of Task 4.
  - `_pull_tracked` is defined in 7 and used in 7 and 8. Task 6 has no `pull_pollutants` call.
  - `CalendarDay.erosion_sine` is produced in 5 and read in 8.
  - `get_climate()` is introduced in 5 and overridden in 15.
  - `Case.crash` and `wsimod_until_crash` are introduced in 5 and used in 9 and 13.
  - `store_bounded_prints` is introduced and used in 9.
  - `surface_climate` is introduced in 13 and extended in 14. `Land.store_paths_of(..., nutrients=)` is introduced in 5 and used by the harness from 5 on.
  - `Ctx.part` and `PART_NAMES` are introduced in 1 and extended by 3, 6, 7 and 8.
- **Proportion.** The plan is about 60 % of the spec's length and carries no function bodies, only signatures, the values the spec pins, test names and assertions.

## Spec gaps and contradictions (with the provisional decision)

1. **Y47's step-time refusal cannot happen.** `model.step` is confined to the network's dates (3a D16, `day.py:1042-1062`), and the build checks every one of them. D7 therefore writes no step-time check. A test pins that `model.step` cannot leave the dates. If dates past the configuration ever become steppable, the check moves to `calendar_day` lookup then.
2. **Y41's constant `c` cannot be fixed in a plan.** Task 9 measures it, and D12 sets it to the ceiling of the measured maximum (expected 2, from the review's ≈1.4). Task 9 stops if it exceeds 4.
3. **The spec's GrowingSurface validation does not say whether `PerviousSurface`'s checks apply** (area 0, porosity against field capacity, `ihacres_p`, soil weights). `_validate_surface` returns early for every class but `PerviousSurface`. D18: they apply, with `rooting_depth` as the physical depth, because WSIMOD crashes on them as it does for a pervious surface.
4. **"The overrides" in Y7's 3c-ii is ambiguous.** D1 keeps `GrowingSurface`'s attribute overrides (`land.py:1376-1431`) with the class, in 3c-i. It moves only the `data_input_dict` override, L14/L15 and the `surface_overrides` case to 3c-ii.
5. **`et0_coef` is recorded by WSIMOD for every non-impervious surface** (`model.py:1150`), so recording it (Y32) adds a compared field to every existing pervious-surface case too (D23). These comparisons are expected to be bitwise: a pervious surface's value is its constant parameter.
