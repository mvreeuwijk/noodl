# WSIMOD In-Water Processes (3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `apps.wsimod` runs every WSIMOD 0.8.1 process that changes a pollutant inside a body of water (temperature-dependent decay in tanks, queue tanks, surfaces and arcs; WTW treatment; River biochemistry; the pervious surface's soil temperature) faithfully, batched, branch-free and differentiable on the 3a substrate, bitwise equal to live WSIMOD node by node, on both demos with every 3b process they contain live, and on purpose-built cases.

**Architecture:** A transcription, as in 3a (X2). Decay is one pure function of `(vqip, constant, exponent, T)` whose factor is evaluated once per element and day (Approach rule 3) and applied column-wise under a structural `present` mask (rule 4). Every power goes through `pow_python` (rule 1), so every 3b class keeps `replay_bound = 0`. The structural switches (rule 2) decide which classes are built; data decide nothing but masks. The mass-balance window becomes contiguous (X8) so that end-of-day decay counts in its own day.

**Tech Stack:** Python 3.12, PyTorch (float64), PyYAML, pytest (+ xdist), `wsimod==0.8.1` (tests and the fixture script only).

**Spec:** `docs/superpowers/specs/2026-11-12-wsimod-processes-design.md` at 73e87cd (binding; read it whole; rulings X1–X40; it is the authority wherever this plan is silent). 3a rulings W1–W95 still bind. 3a's spec and plan (`2026-11-11-wsimod-core-*.md`) keep the old `config_neutral.yml` name as history. WSIMOD source: `C:/projects/noodl/.venv/Lib/site-packages/wsimod/`; every `file:line` below is relative to it. This revision applies the coordinator's rulings on the Fable plan review (R1–R5, O1–O5).

## Global Constraints

- **Start point.** Worktree `C:/Users/mvr/AppData/Local/Temp/noodl-wt/wsimod-processes`, branch `wsimod-processes` (main `a2fb7d0` + spec commits `b66dcb0`, `73e87cd` + this plan). One PR, opened by the lead after a Fable review (X6).
- **Never push. Never `git stash`** (the stash list is shared between worktrees). Never `git checkout -- <path>` or `git restore` on files you did not change.
- **Staging (the worktree may hold unrelated uncommitted files).** Stage files by name only (`git add <path> ...`; never `git add -A`, `git add .` or a directory); before each commit run `git diff --cached --stat` and check that it lists only your task's files.
- **Formatting.** Run `ruff format` only on files you name on its command line, never on a directory; `ruff check .` must pass before every commit.
- **Interpreter.** `C:/projects/noodl/.venv/Scripts/python.exe`, `PYTHONPATH=src`, from the worktree root.
- **The core subset:** `tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py`.
- **The wsimod subset:** `python -m pytest -q -n 4 --no-cov <core subset> tests/apps/wsimod tests/verification/test_allocation_wsimod_*.py tests/verification/test_wsimod_*.py` (the AGENTS.md wsimod row; its paths do not change in 3b).
- **Slow tests.** Add `-m slow` on the wsimod row when a task changes what a slow test compares.
- **mkdocs.** `python -m mkdocs build --strict` at the end of every task.
- **Catalogue.** A task that adds or changes a public name runs `python scripts/gen_catalogue.py` and commits `docs/catalogue/wsimod.md` (`tests/test_docs_catalogue.py`).
- **Git identity.** `git -c user.email=58522531+mvreeuwijk@users.noreply.github.com -c user.name="Maarten van Reeuwijk" commit`.
- **Co-Authored-By.** Every commit message ends with a `Co-Authored-By:` line naming the model that wrote the commit (the implementer's own model).
- **Project rules.** Physical names; no aliases (a moved name is imported from its new home, never re-exported); code, tests and docs change together; no wall-clock assertions (times are measured and recorded, never asserted); 90 % coverage on the full suite.
- **Transcription.** Every transcribed behaviour cites its WSIMOD `file:line`; where the spec and the source disagree, the source wins and the spec is corrected in the same commit.
- **Numerics.** 3a's rules (spec Approach): `torch.float64`; `write(active, new, old)`; `MAXITER` masked rounds; safe denominators; `dmin`/`dmax` tie rule; WSIMOD's reduction order one term at a time; `1e15` never `inf`. Every power is `pow_python` (X18).
- **`replay_bound`.** Every 3b class keeps `replay_bound: ClassVar[float] = 0.0` (bitwise, W41/W47/W87). A non-zero bound needs a measured cause written beside it and in the test (W19).
- **Decision sites.** `"<node|arc>:<name>/<WSIMOD method>/<branch>"` (W44), recorded with their `active` mask (W42); 3b's sites are the spec's Decision vocabulary list, documented in `engine.py`'s module docstring.
- **Reference.** Live WSIMOD 0.8.1 through `pytest.importorskip("wsimod")`; `reset_wsimod_globals()` before every WSIMOD build. No case patches WSIMOD. `src/` never imports `wsimod`.
- **Bounds (W19).** Parity per series within `1e-12 * scale`; the expectation is bitwise (0) with no admissions; an admission is named, measured and asserted. Mass balance within `1e-12 * scale` every day.
- **Suite budget.** The wsimod row's default run stays within 10 minutes with `-n 4` (W15/W23); every new slow-only test is marked `slow`.
- **Progress file** `.superpowers/sdd/2026-11-12-wsimod-processes/progress.md` (git-ignored): baselines, measured times, xfail lists, open causes.
- **Implementer models.** cheap = Haiku, standard = Sonnet, most capable = Opus (Opus only for delicate numerics). Each task names one.

## Review Focus

1. **A decaying `QueueGroundwater` whose water reaches slot 0 at the end of a day with no push before `distribute`** (`decay_queue_groundwater`): expect that water to be decayed again at the next end of day and to enter `active_storage` only at `distribute`'s `update_queue` (X13), not to be active at once as in a non-decaying `QueueTank`. Test in Task 9.
2. **A user `DecayArc` holding waiting requests of several ages at the end of a day, inside the R1 precondition** (`decay_arc`): expect the `"DecayArc.decayed"` term to equal WSIMOD's re-windowed `diff`s bitwise, and expect `"decay_arc_order_unknown"` on exactly the days that break the precondition, not a silent last-bit difference. Tests in Tasks 10 and 11.
3. **A River before its registers fill, and one with `max_temp_lag = 5`** (`river_nutrients`): expect every mean to divide by its full window from day 1 (cold start) and the phosphorus mean to add 366 slots and divide by 365, with both temperature windows the whole 6-slot register (X14, X17), not a mean over the entries present. Tests in Tasks 12 and 13.
4. **A decay factor exactly 1, above 1, and below 0 °C, with `exponent = 1`** (`decay_storage` and the unit test): expect Python's `min(factor, 1)` (the factor unless `1 < factor`), the cap leaving exactly `+0.0`, and decay continuing below 0 °C, not a guard or a clamp. Tests in Tasks 5 and 6.
5. **A second `calculate_discharge` or `Land.run` on one day** (`river_double_discharge`, `land_run_twice`): expect the processes to run twice as in WSIMOD, noodl's terms to add both calls and its residual to be 0, while WSIMOD prints its River mass-balance error every day (X16, X32), not a skipped second call. Tests in Tasks 8 and 13.

## Decisions taken where the spec is silent or ambiguous

Each is binding for the implementer. D21 onward come from the plan review (R1–R5, O3, O5).

- **D1 The gap between the rename and the ports.** Task 1 points every pollutant-mode demo test at `config_no_deposition.yml`, which 3a's code refuses: default soil-temperature weights on both demos, and WSIMOD's default WTW processes on oxford. Each test in Task 1's expected set gets `pytest.mark.xfail(strict=True, raises=<the observed exception>, reason="<process> is Task N")`. The quickstart marks name Task 2 and the oxford marks name Task 3 (R2). Task 2 removes the quickstart marks and Task 3 the oxford marks; `strict=True` makes a forgotten mark fail.
- **D2 Mode names.** The demo pollutant mode is `"pollutants"`, as for the cases (`MODES` in `test_wsimod_parity.py`). Ids: `quickstart-pollutants`, `oxford365-pollutants`, `oxford-pollutants` (slow); replay ids `pollutants-<demo>-<class>`. `test_quantity_only_equals_the_neutralised_volumes` becomes `test_demo_quantity_only_equals_the_pollutant_run_volumes`.
- **D3 Regeneration without a download.** `scripts/regenerate_wsimod_fixtures.py` gains `write_no_deposition_configs(folder: Path) -> None`, which reads the committed `config.yml`, loads it with `wsimod_model`, and writes `config_no_deposition.yml` from it. `write_demo_configs` calls it. `python scripts/regenerate_wsimod_fixtures.py --no-deposition-only` runs it alone, so `config.yml` and the `.csv.gz` files stay byte-identical.
- **D4 `pow_python` moves.** `_PowPython` and `pow_python` move from `land.py` to `vqip.py`, beside `safe_sqrt`. `tanks.py`, `arcs.py`, `wtw.py`, `storage.py` and `land.py` import them from there; there is no re-export. Reason: `tanks.py` cannot import `land.py`, which imports it.
- **D5 When the decay factor is evaluated.** Each decaying element owns a `Decay` (Task 5). `Decay.bind` runs once per day, after the element's temperature is bound (D15). It evaluates `factor = where(cap, 1.0, constant * pow_python(exponent, T - 20))`, with one `Ctx.decide(site + f"/make_decay/cap:{p}", raw, ">", 1.0, active=all-true, call=0)` per present pollutant; the margin is `factor − 1` (spec, Decision vocabulary). Every `make_decay` of that day applies the stored factor.
- **D6 Where the `decays` checks live.** The reader (`decay.read_decays`, called from `config.py`) refuses at read time: a key that is not a configured pollutant (including `volume`); an entry whose keys are not exactly `{constant, exponent}`; `exponent ≤ 0`. Overrides merge per pollutant, as WSIMOD's `dict.update` does (`tanks.py:439`, `:686`, `storage.py:101`). The merged dict is checked again in `validate`, because an override entry that lacks `exponent` crashes WSIMOD too. The "names only some pollutants" refusal of X36 concerns `process_parameters`/`liquor_multiplier` only (D20). A partial `decays` dict is the normal case.
- **D7 `"wsimod.total_decayed"` row order.** Rows follow `WsimodDay._state_layout`'s walk (D10 of 3a): nodes in configuration order, each node's carried paths in order. A decaying `QueueGroundwater` therefore contributes `(node, "tank.internal_arc")` at its position, and a `Land` its `(node, "surfaces.<name>")` rows. The spec's State table lists the two kinds, not an order; Task 8 corrects its wording to "in the state layout's order".
- **D8 Store paths.** `"tank.internal_arc.total_decayed"` (a decaying `QueueGroundwater`, carried); `"surfaces.<name>.total_decayed"` (a surface, carried when `nitrite` is a pollutant); `"lagged_temperatures"` and `"lagged_total_phosphorus"` (a River with biochemistry, carried, flat registers). A storage-family `DecayTank`'s `"tank.total_decayed"` is a per-day store: it is reset before use, so it is not carried (X23), but the replay compares it.
- **D9 River biochemistry's inputs exist only when it runs.** The `"wsimod.River.<p>"` drivers and both lag registers exist when `pollutants=True` and `"nitrate"` is a pollutant. The River's `"wsimod.temperature"` row exists whenever `"nitrate"` is among the configured pollutants, in both modes (spec Inputs: WSIMOD reads it whatever the mode).
- **D10 Unread configuration.** `WsimodNetwork.unread: list[str]` lists what WSIMOD accepts and never reads: `"<node>.data_input_dict"`/`"<node>.filename"` of a `Groundwater` or `QueueGroundwater` (`storage.py:202-203`, `:291-292`); `"<node>.filename"` of a class that reads no forcing (D23); and `"<node>.bulk_density"` (`storage.py:540`). `build_model` exposes it as `model.unread`.
- **D11 Cases with a `filename` override.** A case writes its CSV with `_forcing_file(case: str, rows: Mapping[tuple[str, str], float]) -> str` in `_wsimod_cases.py`. The file goes into a per-process temporary directory (`functools.cache`d `tempfile.mkdtemp`, so each xdist worker writes its own), and the case's `overrides` section names it by absolute path. Both engines then read the same file through their own override code (X11).
- **D12 `decay_queue_groundwater` is two CASES entries:** `decay_queue_groundwater` (WSIMOD's default `number_of_timesteps = 1`) and `decay_queue_groundwater_n0` (the override `number_of_timesteps: 0`).
- **D13 The shared-default `decays={}` row is a unit test, not a parity case.** noodl merges per element and WSIMOD's override reaches every surface built without a `decays` key, so the two cannot be at parity. `surface_decay_nitrification` gives each surface its own `decays` dict.
- **D14 The X8 record (R4).** `WsimodDay._day` builds the `DayRecord` at the record point as now, with `storage_end = None`. After `engine.end_day()`, `_finish_record(ctx, shape)` mutates that same object:
  - it sets `storage_end` (`_held` after `end_timestep`);
  - it re-snapshots `terms`, `dropped`, `dropped_arcs` and `events` from `ctx` (D22).

  `run_days` yields before `end_timestep`, so a consumer at that point must not read `storage_end`; the field's docstring says so. `wsimod_run` and `_balance` read the records only after the generator has resumed.
- **D15 Binding order (R5).**
  - Each node binds its own `Decay`s inside its `bind`, from its bound parameters and its `"temperature"` row: `Storage.bind` for its tank, and `Land.bind` for each surface, with the Land's temperature.
  - After the node loop, `WsimodDay._bind` calls the new `Arc.bind_decay(constant, exponent)` on every `Decay*` arc. An arc's temperature is its `in_port`'s bound `params["temperature"]` (spec, Temperature source). This step comes after the nodes, which reverses, for `Decay*` arcs, 3a's arcs-then-nodes order at `day.py:797-808`. 3a's `capacity`/`preference` binding stays where it is.
  - On a node built with `decays: None`, `Storage.bind` refuses a `decays.*` driver entry other than the inert `(0, 1)`, naming the node, the key and the pollutant. `Land.bind` does the same for a surface without decays, and `Arc.bind_decay` for an absent key.
- **D16 Arc replays.** `Case` gains `replay_arcs: tuple[str, ...] = ()`, the user arcs `test_arc_replay_matches_wsimod` captures (Task 11).
- **D17 The W90/W91 exception (X32).** `assert_matches_wsimod` gains `expected_mass_balance_prints: Sequence[str] = ()`, the node names whose `"mass balance error"` prints are expected on every day. A print listed there is asserted present on every day; any other print is checked as before.
- **D18 Event order.** `"wtw_negative_solids"` (Task 3), `"decay_arc_order_unknown"` (Task 10) and `"river_nitrogen_lost"` (Task 12) are appended to `run.EVENT_NAMES`, in that order.
- **D19 3a's driver shape.** A `wsimod.<WTW class>.liquor_multiplier` of 3a's shape `(..., n)` is refused by `bind` with a `ValueError` naming the key and the expected `(..., n, 1 + K_additive)` (X10).
- **D20 Partial WTW dicts.** A configured `process_parameters` or `liquor_multiplier` that lacks an additive pollutant is refused in `config._read_node`, before overrides merge (X36). Entries for non-additive or unknown pollutants, a volume `exponent`, and `exponent ≤ 0` are refused in `WTW.validate` after the merge.
- **D21 The `DecayArc` order precondition (R1, O5).**
  - A waiting row `(d, i, j)` is due when `i + since[d] >= n` (D1 of 3a). `since` saturates at `n`, so a due row that is still waiting has no reconstructible age.
  - The term order of Task 10 is exact while no end of day finds due-and-waiting rows in both directions.
  - At every `DecayArc.end_timestep`, the detached event `"decay_arc_order_unknown"` is recorded at the arc's name (in `Ctx.events`, keyed by the arc) where `active & any(present & due, push) & any(present & due, pull)`.
  - Delivered values and rows are exact in every case; only the summation order of the diagnostic term is affected.
  - 3a's `since` semantics are unchanged.
- **D22 What `end_timestep` records (R4, verified).** No 3a `end_timestep` (nodes, tanks or arcs) records an event, a decision, a maxiter, a term or a drop. In 3b, `end_timestep` records terms (`"DecayTank.decayed"`, `"DecayQueueTank.decayed"`, `"DecayArc.decayed"`, `"DecayArcAlt.decayed"`) and one event (`"decay_arc_order_unknown"`, D21). It records no decision, because the caps are evaluated at bind (D5), and no maxiter. `_finish_record` therefore re-snapshots terms, drops and events. Task 4 adds a test asserting that `ctx.decisions` and `ctx.maxiter` are unchanged across `end_day` on every case, so a later decision recorded during `end_timestep` fails loudly.
- **D23 `filename` edge cases (O3).** These mirror `Node.apply_overrides` (`nodes.py:93-99`):
  - A string is read with `read_csv`.
  - An empty string (falsy) is ignored, and the node's forcing is unchanged.
  - Any other non-string value is refused by name; WSIMOD raises `RuntimeError("Not recognised format for data_input_dict")`.
  - On a class that reads no forcing, the override is accepted (WSIMOD reads the file) and listed in `unread`.
- **D24 The slow horizon of a case (R3).** `Case` gains `slow_days: int | None = None`. The default run uses `days`. A test marked `slow` that names the full horizon uses `slow_days`. `test_case_matches_wsimod` keeps `days`, and a slow companion `test_case_matches_wsimod_on_its_slow_days[<case>-<mode>]` runs every case with `slow_days`.

---

### Task 0: Baseline (controller)

**Implementer:** cheap (the controller runs it). **Files:** none changed.

- [ ] **Step 1:** `git status --short` and `git log --oneline -3` in the worktree; record both in the progress file. Unrelated untracked or modified files are left alone (Global Constraints, Staging).
- [ ] **Step 2:** Run the wsimod subset with `--durations=25`; expected: pass. Record its wall time, the test count and the 25 slowest items under "Baseline".
- [ ] **Step 3:** Run `python -m pytest -q -n 4 --no-cov -m slow tests/apps/wsimod tests/verification/test_wsimod_*.py`; record its wall time.
- [ ] **Step 4:** Run `python benchmarks/wsimod_run.py` three times; record the median seconds per day per (demo, batch, grad) row.

### Task 1: `config_no_deposition.yml` replaces `config_neutral.yml` (X20, X38)

**Implementer:** standard.

**Files:**
- Rename (`git mv`): `tests/data/wsimod/quickstart/config_neutral.yml` → `config_no_deposition.yml`; `tests/data/wsimod/oxford/config_neutral.yml` → `config_no_deposition.yml` (content regenerated).
- Modify: `scripts/regenerate_wsimod_fixtures.py`, `tests/verification/test_wsimod_fixtures.py`, `tests/verification/test_wsimod_parity.py`, `tests/verification/test_wsimod_replay.py`, `tests/verification/test_wsimod_gradients.py`, `tests/data/wsimod/README.md`, `docs/applications/wsimod.md`.

**Interfaces (produced):**

```python
# scripts/regenerate_wsimod_fixtures.py
def _no_deposition_overrides(model) -> dict
    # {"nodes": {land: {"type_": "Land", "name": land, "surfaces": {s: {"pollutant_load": {k: 0.0 for k in loads}}}}}}
    # for every Land and every surface, every key the surface's pollutant_load has; nothing else
def write_no_deposition_configs(folder: Path) -> None   # D3
# tests/verification/test_wsimod_parity.py
CONFIG = {"quantity": "config.yml", "pollutants": "config_no_deposition.yml"}   # D2
```

**The expected set of marked tests (R2).** These are the tests that build or replay noodl in pollutant mode on a demo, with the class that 3a refuses or computes neutrally. Quickstart fails at build (`ValueError`, soil temperature) and is marked "Task 2". Oxford fails at build on soil temperature and then on the WTW processes, and is marked "Task 3".

| Test (file) | Mark | Exception |
|---|---|---|
| `test_demo_matches_wsimod[quickstart-pollutants]` (parity) | Task 2 | `ValueError` |
| `test_demo_quantity_only_equals_the_pollutant_run_volumes[quickstart]` (parity) | Task 2 | `ValueError` |
| `test_a_difference_beyond_the_bound_is_named_by_element_and_day[*]` (parity, every parametrisation; it uses `_perturbed`, the quickstart pollutant run) | Task 2 | `ValueError` |
| `test_different_arc_names_or_day_counts_are_refused` (parity, `_perturbed`) | Task 2 | `ValueError` |
| `test_an_admission_needs_a_decision_within_its_margin_that_day` (parity, `_perturbed`) | Task 2 | `ValueError` |
| `test_a_wrong_node_name_mapping_fails_the_deficit_check` (parity) | Task 2 | `ValueError` |
| `test_a_print_noodl_does_not_record_is_refused` (parity, `_perturbed`) | Task 2 | `ValueError` |
| `test_an_undated_print_needs_noodls_event_and_a_dated_one_its_node_and_day` (parity, `_perturbed`) | Task 2 | `ValueError` |
| `test_demo_matches_wsimod[oxford365-pollutants]` and `[oxford-pollutants]` (slow) (parity) | Task 3 | `ValueError` |
| `test_demo_quantity_only_equals_the_pollutant_run_volumes[oxford]` (parity) | Task 3 | `ValueError` |
| `test_oxfords_urban_surface_holds_no_storage_after_the_first_day` (parity) | Task 3 | `ValueError` |
| `test_demo_nodes_replay_wsimod[pollutants-oxford-WWTW]`, `[pollutants-oxford-FWTW]` and the same ids of `test_demo_nodes_replay_wsimod_on_every_day` (slow) (replay; the replay builds no model, so 3a's neutral split meets WSIMOD's default processes) | Task 3 | `Mismatch` |

Expected to pass unmarked:
- every other `pollutants-*` replay, `Land` included, because 3a's soil-temperature formula is already general;
- `test_a_pollutant_mode_demo_replay_notices_a_difference`, which raises `Mismatch` anyway during the gap: vacuous until Task 3, which re-checks it;
- every gradient test, because the quickstart `Spec`s are quantity-only (`_Q`, `pollutants=False`).

If the observed failures differ from this set, stop and report the difference before marking anything.

- [ ] **Step 1: Write the failing test** in `test_wsimod_fixtures.py`: `test_the_no_deposition_configuration_differs_only_by_its_overrides[quickstart|oxford]` (replaces the neutral test). It asserts:
  - the YAML mapping of `config_no_deposition.yml` with `"overrides"` removed equals `config.yml`'s;
  - `overrides["nodes"]` names exactly the `Land` nodes;
  - every surface entry is exactly `{"pollutant_load": {k: 0.0 ...}}` over the keys of that surface's configured `pollutant_load`;
  - `wsimod_model(dir, config_name="config_no_deposition.yml")` loads, and every surface's `pollutant_load` values are 0.
- [ ] **Step 2:** Run `pytest tests/verification/test_wsimod_fixtures.py -v`; expected: FAIL (no file).
- [ ] **Step 3:** Rename the two files with `git mv`. Replace `_neutral_overrides` with `_no_deposition_overrides` and add `write_no_deposition_configs` and the `--no-deposition-only` flag (D3). Run `python scripts/regenerate_wsimod_fixtures.py --no-deposition-only`. `git diff --stat` must show changes to the two YAML files only, among the data files.
- [ ] **Step 4:** Point every reader at the new file:
  - `test_wsimod_parity.py`: `CONFIG`, `DEMOS` ids and the `"neutral"` mode arguments, including `_perturbed` (D2);
  - rename `test_quantity_only_equals_the_neutralised_volumes` to `test_demo_quantity_only_equals_the_pollutant_run_volumes`;
  - `test_wsimod_replay.py`: `RUNS` rows `("pollutants-", demo, "config_no_deposition.yml", "pollutants")`, and `test_a_pollutant_mode_demo_replay_notices_a_difference`;
  - `test_wsimod_gradients.py` `_model`.

  README: the file neutralises only the surfaces' `pollutant_load` (3c), it is written by the script, and the command is given. `wsimod.md`'s verification text says "pollutant mode on `config_no_deposition.yml`".
- [ ] **Step 5:** Run the wsimod subset and `-m slow` on `test_wsimod_parity.py test_wsimod_replay.py`. Compare the failures with the expected set. Mark them per D1 and the table. Expected after marking: pass. Copy the table, with the observed exceptions, into the progress file. `git grep -n config_neutral -- . ":!docs/superpowers"` is empty.
- [ ] **Step 6:** Commit: `test(wsimod): config_no_deposition.yml replaces config_neutral.yml (X20, X38)`.

### Task 2: The general soil temperature (`land.py:1060-1079`)

**Implementer:** standard.

**Files:**
- Modify: `src/noodl/apps/wsimod/land.py` (`PerviousSurface.calculate_soil_temperature`, `PerviousSurface.bind`, `_validate_surface`, the deposition messages at `:859`, `:870`), `tests/apps/wsimod/test_land.py`, `tests/verification/_wsimod_cases.py`, `tests/verification/test_wsimod_parity.py` (the Task 2 marks), `docs/applications/wsimod.md`.

**Interfaces:**
- Consumes: 3a's `PerviousSurface` and its drivers `wsimod.PerviousSurface.soil_temp_w_prev`, `…_w_air`, `…_w_deep`, `…soil_temp_deep`.
- Produces:
  - `_validate_surface` accepts any weights with `w_air + w_deep + w_prev != 0`;
  - `PerviousSurface.bind` raises `ValueError` naming the surface and `"total weight"` where any instance's sum is 0;
  - the deposition refusals name only `"3c"`;
  - case `soil_temperature` in `CASES`.

- [ ] **Step 1: Write the failing tests** in `test_land.py`:
  - `test_the_soil_temperature_moves_with_wsimods_default_weights`: storage temperature 12, air 18, `w_prev 0.1, w_air 0.6, w_deep 0.1, soil_temp_deep 10` (`land.py:809-812`). The new value is `(12*0.1 + 18*0.6 + 10*0.1) / (0.6 + 0.1 + 0.1)` (Python's expression, `torch.equal`), and the returned in/out VQIPs are empty.
  - `test_a_zero_previous_weight_forgets_the_storage_temperature`: `w_prev 0, w_air 1, w_deep 0`: the new value is `(12*0 + 18*1 + 10*0) / (1 + 0 + 0)`.
  - `test_a_zero_total_soil_weight_is_refused`: weights `(0, 0, 0)` in the configuration, `ValueError` matching `"total weight"`, both modes. The same through a batched driver at `bind` (one instance 0).
  - `test_deposition_refusals_name_only_3c`: a non-zero `pollutant_load`, `match="3c"`, and the message does not contain `"3b"`.
  - Update `test_porosity_not_above_field_capacity_is_refused`'s parametrisation (`:737-745`): the two `"3b"` rows become accepted configurations.
  - `test_the_soil_temperature_is_neutral_with_the_neutral_weights` stays.
  - Replay and whole model: case `soil_temperature`. It has a `Land` with two `PerviousSurface`s whose weights come through the case's `overrides` (one `w_prev = 0`, one `(0.3, 0.5, 0.2)` with `soil_temp_deep 8`) and an `ImperviousSurface`, draining to a `Node` → `Waste` and a `Groundwater`. Pollutants `temperature`, `phosphate`; no `River`. Air temperature crosses 0 and 20. `replay=("<land>",)`.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_land.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement. Remove the marks whose reason names Task 2.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS. The quickstart pollutant-mode demo tests now pass bitwise. Add `soil_temperature` to `test_the_cases_cover_every_listed_feature`'s list. Docs: the soil temperature is live, with its formula and the zero-weight refusal.
- [ ] **Step 5:** Commit: `feat(wsimod): the pervious surface's soil temperature with any weights`.

### Task 3: WTW treatment (`wtw.py:163-214`)

**Implementer:** standard.

**Files:**
- Modify: `src/noodl/apps/wsimod/vqip.py` and `land.py` (D4), `wtw.py` (`WTW.__init__`, `bind`, `treat_current_input`, `validate`), `config.py` (`_read_node`, D20), `day.py` (`_node_parameters` and `_driver_layout` for the new drivers, `_PARAMETER_UNITS`), `run.py` (`EVENT_NAMES`, D18), `tests/apps/wsimod/test_sewer_wtw.py`, `tests/apps/wsimod/test_config.py`, `tests/apps/wsimod/test_build.py`, `tests/verification/_wsimod_cases.py`, the Task 3 marks, `docs/applications/wsimod.md`, `docs/formats/wsimod.md`.

**Interfaces:**
- Consumes: 3a's `WTW`, `WWTW`, `FWTW`, `_fill_wtw_defaults`, `wtw_derived`.
- Produces:

```python
# vqip.py (moved, D4)
def pow_python(x: Tensor, y: Tensor) -> Tensor      # x ** y bitwise as Python's float.__pow__
# drivers (X10, X34), rows every node of the class in configuration order (D6 of 3a)
"wsimod.<WTW|WWTW|FWTW>.process_parameters.constant"   # (..., n, K_add), columns the additive pollutants
"wsimod.<WTW|WWTW|FWTW>.process_parameters.exponent"   # (..., n, K_add); absent in a quantity-only run
"wsimod.<WTW|WWTW|FWTW>.liquor_multiplier"             # (..., n, 1 + K_add), columns "volume", then additive; (..., n, 1) quantity-only
# WTW
def treat_current_input(self, *, active: Tensor) -> None
    # general: discharge[k] = influent[k] * constant[k] * pow_python(exponent[k], 20 - influent["temperature"])
    # (volume: factor 1, constant c_v); liquor[k] = influent[k] * liquor_multiplier[k]; solids remainder;
    # event "wtw_negative_solids" where any additive solids column < 0 (X22)
```

- [ ] **Step 1: Write the failing tests** in `test_sewer_wtw.py` (and `test_config.py`, `test_build.py`):
  - `test_treat_current_input_equals_wsimods_on_random_inputs` (`importorskip`). A WSIMOD `WTW` with pollutants `phosphate`, `ammonia` (additive) and `temperature`; per-pollutant constants random in `[0.005, 0.5]`, exponents in `[0.95, 1.1]` with one exactly 1, liquor multipliers in `[0.1, 0.9]` other than the volume's 0.03. 200 random influents with temperatures in `[-5, 35]` and 10 % zero volume. After each call, `treated`, `liquor` and `solids` are `torch.equal` to WSIMOD's.
  - `test_the_general_treatment_reproduces_the_neutral_split_bitwise`: 3a's `test_wtw_splits_by_its_volume_constant` numbers through the general code, `torch.equal` to 3a's expressions.
  - `test_cold_influent_makes_negative_solids_and_says_so`: constant 0.5, exponent 1.1, multiplier 0.7, temperature 0. Treated phosphate is `x * 0.5 * 1.1 ** (20 - 0)`, solids are `(x - treated) - liquor < 0`, `"wtw_negative_solids"` is true, and the node's lists close.
  - `test_wtw_treatment_refusals` (parametrised, `match=`):
    - `process_parameters` lacking `phosphate` in the configuration (`"phosphate"`, D20);
    - an entry for `temperature` (`"temperature"`);
    - an unknown name (`"nitratee"`);
    - a `volume` `exponent` (`"exponent"`);
    - `exponent: 0` and `exponent: -1` (`"exponent"`), also through a batched driver at `bind`;
    - `liquor_multiplier` lacking an additive pollutant.

    Kept from 3a: additive pollutants without `temperature`, and a configured volume constant other than the derived one.
  - `test_a_partial_wtw_override_merges` (`importorskip`): an override `{"process_parameters": {"phosphate": {"constant": 0.2}}}` on WSIMOD's defaults gives `{"constant": 0.2, "exponent": 1.001}`, equal to WSIMOD's merged attribute after `add_overrides`.
  - `test_a_3a_shaped_liquor_multiplier_driver_is_refused` (D19).
  - `test_quantity_only_driver_shapes` (O2, in `test_build.py`): oxford built with `pollutants=False`. `model.driver_template["wsimod.WWTW.liquor_multiplier"]` and the FWTW one have shape `(n, 1)`, and no `wsimod.*.process_parameters.*` key exists. Tasks 6, 8 and 10 extend the test to `decays.*`.
  - Replace `test_non_neutral_wtw_processes_are_refused`: WSIMOD's defaults now build (constant 0.01, exponent 1.001, multiplier 0.7, volume 0.03).
  - Replay and whole model: case `wtw_treatment`, on the topologies of 3a's `wwtw_storm_tank` and `fwtw_deficit_sludge`. An `FWTW` and a `WWTW` have per-pollutant constants and exponents other than the defaults and liquor multipliers other than the volume's. Catchment temperatures cross 20 °C, and cold days make solids negative (the event asserted on at least one day). The storm tank fills and drains. `replay=` both works.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement: move `pow_python` (D4); build the drivers; write the general treatment in WSIMOD's order (`:188-208`, columns independent, left to right); remove the neutral check; refuse per D20. Remove the marks whose reason names Task 3.
- [ ] **Step 4:** Run the wsimod subset and `-m slow` on `test_wsimod_parity.py test_wsimod_replay.py`; expected: PASS. oxford in pollutant mode now runs WWTW and FWTW treatment on its 11 additive pollutants, bitwise. `test_a_pollutant_mode_demo_replay_notices_a_difference` now discriminates: run it once with the mutation removed and confirm it then fails. No mark remains (`git grep -n "is Task [23]" tests` empty). Regenerate the catalogue. Docs: the treatment formula, its quirks (the factor rising as the temperature falls, negative solids, the event), the new driver shapes, and the refusals.
- [ ] **Step 5:** Commit: `feat(wsimod): general WTW treatment with per-pollutant processes and liquor multipliers`.

### Task 4: The contiguous mass-balance window (X8, X32)

**Implementer:** most capable (the record and the window are what every later term relies on).

**Files:**
- Modify: `src/noodl/apps/wsimod/day.py` (`DayRecord`, `_day`, `_finish_record`), `run.py` (`_balance`), `arcs.py` (`Arc.mass_balance`), `tests/apps/wsimod/test_run.py`, `docs/applications/wsimod.md` (the balance section).

**Interfaces:**
- Consumes: 3a's `DayRecord`, `WsimodDay._held`, `Ctx.term`, `Engine.end_day`.
- Produces:

```python
@dataclass
class DayRecord:                    # 3a's fields unchanged in meaning (X32), plus:
    storage_end: Tensor | None      # (..., 1 + K_add): the water held after end_timestep; None until
                                    # _finish_record (not readable at the run_days yield point, D14)
    # terms, dropped, dropped_arcs and events are re-snapshotted after end_timestep (D22)
class WsimodDay:
    def _finish_record(self, ctx: Ctx, shape: torch.Size) -> None   # D14, mutates last_record
class Arc:
    mass_balance: ClassVar[dict[str, str]] = {}   # term -> "in" | "out", as Node.mass_balance
# run._balance: storage_change = storage_end - storage_start; directions from nodes and arcs
```

- [ ] **Step 1: Write the failing tests** in `test_run.py`:
  - `test_consecutive_balance_windows_meet[<case>-<mode>]`, over every `CASES` entry and mode. The completed records satisfy `storage_start[d + 1] == storage_end[d]` (`torch.equal`) on every day.
  - `test_end_timestep_records_no_decision_and_no_maxiter[<case>-<mode>]` (D22), over every `CASES` entry and mode, recording decisions. `len(ctx.decisions)` and `ctx.maxiter` are the same immediately before and after `engine.end_day()`.
  - `test_storage_end_is_unset_at_the_yield_point`: inside `run_days`, `last_record.storage_end is None`. After the generator resumes, it is set.
  - `test_a_term_recorded_at_the_end_of_the_day_counts_in_that_day`. In case `storage_distribute`, monkeypatch one `Storage`'s `end_timestep` to extract 1 m³, record the out-term `"Test.removed"` (declared on that instance's `mass_balance`), and record an event `"test_event"`. The residual is within `1e-12 * scale` every day, `terms["Test.removed"]` is 1 on every day, and the event appears in the day's record.
  - `test_an_arc_term_needs_a_declared_direction`: an undeclared arc term raises `RuntimeError` naming it. A declared one counts.
  - `test_storage_total_keeps_its_meaning`: `storage_total` equals 3a's record-point value (`_held` before `end_day`) on `user_alt_queue_arc`.
  - 3a's `test_the_mass_balance_closes_on_every_case` and `test_the_mass_balance_closes_by_hand` re-run unchanged (spec: the 3a numbers change only by rounding).
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_run.py -v`; expected: FAIL (`storage_end` missing).
- [ ] **Step 3:** Implement D14 and D22. `_balance` uses `storage_end`.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS. Record each case's worst residual (relative to scale) before and after in the progress file. Docs: the window.
- [ ] **Step 5:** Commit: `feat(wsimod): a contiguous mass-balance window through end_timestep (X8)`.

### Task 5: The decay function and `DecayTank` (`core.py:432-521`, `tanks.py:403-457`)

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/decay.py`, `tests/apps/wsimod/test_decay.py`.
- Modify: `constants.py` (`DECAY_REFERENCE_TEMPERATURE = 20`, `constants.py:66`), `tanks.py` (`DecayTank`), `engine.py` (docstring sites).

**Interfaces:**
- Consumes: `pow_python` (Task 3), `Ctx.decide`, `Ctx.term`, `Arc.mass_balance` (Task 4).
- Produces:

```python
# decay.py
@dataclass(frozen=True)
class DecaySpec:                         # one element's configured decays (structural, X9)
    keys: tuple[str, ...]                # configuration order
    constant: dict[str, float]; exponent: dict[str, float]
def read_decays(where: str, decays: Mapping | None, pollutants: Pollutants) -> DecaySpec | None
    # None for decays None; refusals per D6, by name
def check_inert(where: str, key: str, present: Tensor, constant: Tensor, exponent: Tensor) -> None
    # refuses, naming where, key and pollutant, an absent column whose (constant, exponent) is not (0, 1) (X9, D15)
class Decay:
    def __init__(self, pollutants: Pollutants, spec: DecaySpec, *, site: str) -> None
    present: Tensor                      # (1 + K,) bool; column 0 False
    constant: Tensor; exponent: Tensor   # (..., K), inert (0, 1) at absent keys
    factor: Tensor                       # (..., 1 + K), the day's min(c·e^(T−20), 1), 0 where absent
    def bind(self, ctx: Ctx, constant: Tensor, exponent: Tensor, temperature: Tensor) -> None
        # check_inert; refuses exponent <= 0 at present keys by name; D5
    def make_decay(self, vqip: Tensor) -> tuple[Tensor, Tensor]
        # (decayed, diff): diff = t * factor at present columns, 0 elsewhere and at volume;
        # decayed = where(present, t - diff, t)  (core.py:507-519)
# tanks.py
class DecayTank(Tank):
    def __init__(self, pollutants, *, parent, decay: Decay | None, **kwargs) -> None
    decay: Decay | None; total_decayed: Tensor      # per-day store "total_decayed" (D8)
    def end_timestep(self) -> None
        # total_decayed = empty; storage, diff = decay.make_decay(storage); total_decayed = sum_vqip(total, diff);
        # ctx.term("DecayTank.decayed", diff, all-true)   (tanks.py:442-447); with decay None: 3a's identity
```

- [ ] **Step 1: Write the failing tests** in `test_decay.py`:
  - `test_generic_temperature_decay_equals_decayobj_bitwise` (`importorskip`). WSIMOD's `DecayObj` reads the temperature from a dummy object with `data_input_dict` and `t` (`core.py:464-470`, `:485-487`). The `decays` cover `phosphate` (additive) and `do` (non-additive), with `nitrate` absent. 1000 random VQIPs, 10 % at zero volume with mass. Constants in `[0.001, 0.2]`, 5 % equal to 1.5 (the cap binds where `T ≥ 20`). Exponents in `[1, 1.1]`, a tenth exactly 1. `T` in `[-5, 35]`. The decayed VQIP and the `diff` (WSIMOD's `total_decayed` after one call from empty) are `torch.equal`.
  - `test_the_cap_leaves_exactly_zero`: constant 2, exponent 1.05, `T = 25`. Phosphate is `+0.0` after decay and `diff` is the old value. The decision `<site>/make_decay/cap:phosphate` has margin `2 * 1.05 ** 5 - 1`. With constant 1 and `T = 20` (factor exactly 1), the value is `t − t·1.0`, the same as with the cap.
  - `test_decay_continues_below_zero_degrees`: constant 0.1, exponent 1.05, `T = -3`. The factor is `0.1 * 1.05 ** (-3 - 20)`, applied with no guard.
  - `test_a_zero_volume_vqip_with_mass_decays`: volume 0 with phosphate 2. Phosphate is `2 - 2 * factor`.
  - `test_an_absent_pollutant_is_untouched`: `nitrate` is bitwise unchanged with a `diff` of 0. A driver entry of `(0.3, 1)` at the absent `nitrate` is refused by `check_inert`, with `match="nitrate"`.
  - `test_read_decays_refusals` (parametrised, `match=`):
    - key `volume` (`"volume"`);
    - an unconfigured pollutant (`"nitrite"`);
    - `{constant: 1}` (`"exponent"`);
    - `{constant: 1, exponent: 1, rate: 2}` (`"rate"`);
    - `exponent: 0` and `exponent: -1` (`"exponent"`).

    `exponent ≤ 0` through a batched driver is refused at `Decay.bind`.
  - `test_decay_tank_equals_wsimods_on_random_days` (`importorskip`): a WSIMOD `DecayTank` with a dummy parent and 60 days of random pushes, pulls and `end_timestep` at that day's temperature. `storage` and `total_decayed` are `torch.equal` every day.
  - `test_gradcheck_on_the_decay_function`: `torch.autograd.gradcheck` with respect to `constant`, `exponent`, `T` and the VQIP, below the cap.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_decay.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement. Python's `min(factor, 1)` keeps the factor unless `1 < factor` (spec, Decay quirks). `where(cap, 1.0, factor)` gives exactly `t − t = +0.0`. 3a's empty-decays `DecayTank` behaviour (the parent's temperature must exist) is kept for `decay=None`.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS (no configuration builds a decaying element yet). Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): temperature-dependent decay and DecayTank`.

### Task 6: The storage family's decaying tanks

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/storage.py`: `Storage._make_tank`, `bind` (D15), `validate`, `mass_balance`, the `River` docstring;
  - `config.py`: `_read_node` calls `read_decays`; `_apply_overrides` merges `decays` per pollutant (D6); `_refuse_decays` is kept for surfaces and `QueueGroundwater` until Tasks 8 and 9;
  - `classes.py`: the `Storage` and `DecayTank` override keys gain `"decays"`;
  - `structure.forcing_readers`: decaying storage-family nodes;
  - `day.py`: `_driver_layout` gains the `decays.*` drivers;
  - `tests/verification/_wsimod_reference.py`: `capture_node_calls` wraps `end_timestep`, with the stores before and after;
  - `tests/apps/wsimod/_replay.py`: replays `end_timestep` and binds the day's `temperature` from `spec.data`;
  - tests: `tests/apps/wsimod/test_storage.py`, `test_config.py`, `test_build.py`, `tests/verification/_wsimod_cases.py`;
  - docs.

**Interfaces:**
- Consumes: `read_decays`, `check_inert`, `Decay`, `DecayTank` (Task 5); the X8 window (Task 4).
- Produces:

```python
# drivers (spec, Inputs; X34): "wsimod.<Class>.decays.constant" / ".exponent", (..., n, K), every node of the
#   storage-family class Class (decays None: inert rows), absent in a quantity-only run; unit "-"
# Storage.bind(parameters): binds self.tank.decay from parameters["decays.constant"], ["decays.exponent"],
#   ["temperature"]; on a decays-None node refuses a non-inert decays entry by name (D15)
# Storage.mass_balance gains {"DecayTank.decayed": "out"}
# forcing_readers: "temperature" read by every Storage, Reservoir, RiverReservoir and River with a decays dict
#   (Groundwater and QueueGroundwater: Tasks 7 and 9)
```

- [ ] **Step 1: Write the failing tests** in `test_storage.py` (and `test_config.py`, `test_build.py`):
  - `test_storage_with_decays_reads_its_temperature`: a decaying `Storage` without `("temperature", date)` forcing is refused by name (node, `"temperature"`, the first missing date). With `decays: None` it needs none.
  - `REFUSALS` updates:
    - `(_node(type_="Storage", decays={}), "3b")` becomes an accepted configuration;
    - the `read_decays` refusals of Task 5 are added at node level;
    - `(_node(type_="Groundwater", decays={...}), "decays")` stays refused until Task 7 (its message names the missing `filename` override).
  - `test_a_decays_override_merges_into_a_dict_and_is_refused_on_none` (`importorskip`). On a node built with `{phosphate: {constant: 0.1, exponent: 1.02}}`, the override `{ammonia: {constant: 0.2, exponent: 1.0}}` gives both keys, as WSIMOD's attribute does. On a node built with `decays: None`, the override is refused with `match="decays"`, and WSIMOD raises `ValueError` (`storage.py:96-100`). An override entry `{phosphate: {constant: 0.3}}` is refused after the merge, with `match="exponent"` (D6).
  - `test_a_decays_none_node_refuses_a_changed_inert_driver` (D15): a `Reservoir` with `decays: None`; `initial_drivers(values={"wsimod.Reservoir.decays.constant": {"r": {"phosphate": 0.1}}})` is refused at `bind`, naming the node and `phosphate`.
  - `test_quantity_only_driver_shapes` extended (O2): no `wsimod.*.decays.*` key in a quantity-only build of `decay_storage`.
  - Replay and whole model: case `decay_storage` (spec, Purpose-built cases), with `replay` set to its four nodes. It has a `Storage`, a `Reservoir`, a `RiverReservoir` and a `River` without `nitrate`, each with `decays` and node temperature forcing that crosses 20 °C. One constant is large enough that the cap binds on warm days, and one day sets factor exactly 1 (Review Focus 4). It decays `do` (non-additive), keeps one pollutant undecayed, and merges a `decays` override into one node.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_storage.py tests/apps/wsimod/test_config.py tests/verification/test_wsimod_replay.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS. The generic `test_case_matches_wsimod[decay_storage-*]`, `test_quantity_only_equals_the_pollutant_run_volumes[decay_storage]` and `test_the_mass_balance_closes_on_every_case[decay_storage-*]` pass. Add the case to the listed-features test. Docs: decay, its temperature sources (table), its kinks, the refusals, and that WSIMOD's `DecayTank` drops out of `tank_*` (X40).
- [ ] **Step 5:** Commit: `feat(wsimod): decaying tanks in the storage family`.

### Task 7: `filename` overrides and the decaying `Groundwater` (X11, X27)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/config.py`: `_apply_overrides`/`_override_node` handle `filename` per D23; a string is read with `_read_csv(Path(value))` as written, relative to the working directory, and replaces `NodeSpec.data`; `WsimodNetwork.unread`, D10;
  - `classes.py`: the `Node` override keys gain `"filename"`;
  - `storage.py`: `Groundwater.validate`;
  - `build.py`: `model.unread`;
  - tests: `tests/apps/wsimod/test_config.py`, `tests/apps/wsimod/test_storage.py`, `tests/verification/_wsimod_cases.py` (`_forcing_file`, D11), `tests/verification/test_wsimod_parity.py` (crash test);
  - docs: `docs/formats/wsimod.md`, `docs/applications/wsimod.md`.

**Interfaces:**
- Consumes: the decaying storage family (Task 6).
- Produces:

```python
@dataclass
class WsimodNetwork:   # 3a's fields, plus
    unread: list[str]  # D10
def _forcing_file(case: str, rows: Mapping[tuple[str, str], float]) -> str   # absolute path (D11)
# Groundwater.validate: a decays dict without ("temperature", date) for every date in the
#   filename override's data refused by name (X11); forcing_readers adds such a Groundwater
```

- [ ] **Step 1: Write the failing tests:**
  - `test_a_filename_override_replaces_the_forcing` (`importorskip`, `tmp_path`): a `Catchment` with inline data and an override `filename: <absolute csv>`. `NodeSpec.data` equals the file's, and WSIMOD's `data_input_dict` after `add_overrides` holds the same values (keys compared as datetimes).
  - `test_a_relative_filename_override_opens_from_the_working_directory` (`monkeypatch.chdir(tmp_path)`): a relative path is read from the working directory, not from the configuration's directory.
  - `test_filename_override_edge_cases` (D23, `importorskip` for the WSIMOD side):
    - `filename: ""` leaves the forcing unchanged, as WSIMOD's does;
    - `filename: 3` is refused with `match="filename"`, and WSIMOD raises `RuntimeError`;
    - a `filename` on a `Waste` (no forcing read) is accepted, with `"<waste>.filename"` in `network.unread`.
  - `REFUSALS` update: the 3a row `(_override(node={... "filename": "x.csv"}), "filename")` becomes a missing file, refused by its path (`"x.csv"`). The surface `data_input_dict` override stays refused (`"data_input_dict"`, 3c).
  - `test_groundwater_forcing_is_lost_and_listed_unread`: a `Groundwater` with `data_input_dict` lists `"g.data_input_dict"` in `network.unread` and `model.unread`.
  - `test_a_decaying_groundwater_needs_a_filename_override`: no override gives `match="filename"`.
  - `test_case_runs_where_wsimod_crashes_decay_groundwater` (in `test_wsimod_parity.py`): the case without its override. WSIMOD raises `TypeError` at its first decay (`core.py:485`), and noodl refuses by name.
  - Whole model and replay: case `decay_groundwater`. A `Groundwater` with `decays` takes its temperature from a `_forcing_file` override; `replay=("<groundwater>",)`.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS. Add the case to the listed-features test. Docs: the `filename` override, its working-directory path ("use absolute paths"), its edge cases, and the lost `Groundwater` data.
- [ ] **Step 5:** Commit: `feat(wsimod): filename overrides and the decaying Groundwater`.

### Task 8: Surface decay and the nitrite carry-over (`land.py:425-449`)

**Implementer:** standard.

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/config.py`: `_read_surface` calls `read_decays`; `_refuse_decays` remains only for `QueueGroundwater`;
  - `classes.py`;
  - `land.py`: `Surface` builds its `Decay`; `Land.bind` binds each surface's `Decay` with the Land's temperature and refuses a non-inert entry on a surface without decays (D15); `Surface.run` carry-over; `Land.carried_paths`/`store_paths_of`; `Land.end_timestep`; `Land.mass_balance`; `Land.validate` refuses `nitrite` without `nitrate` and `ammonia`;
  - `day.py`: `STATE_KEYS` and `_STATE_TEXT` gain `"wsimod.total_decayed"`, `_NOT_HELD` gains it, `_state_layout` per D7, the surface `decays.*` drivers;
  - `tests/verification/_wsimod_reference.py`: the capture of each surface's `total_decayed`;
  - tests: `tests/apps/wsimod/test_land.py`, `tests/apps/wsimod/test_build.py`, `tests/verification/_wsimod_cases.py`;
  - the spec's State row wording (D7); docs.

**Interfaces:**
- Consumes: `Decay`, `DecayTank` (Task 5), the X8 window (Task 4).
- Produces:

```python
"wsimod.total_decayed"    # (..., n_rows, 1 + K); rows (node, "surfaces.<name>") when "nitrite" is a pollutant (X35, D7)
# Surface.run (land.py:427-439), when "nitrite" in pollutants, before the inflows:
#   storage["nitrate"] += total_decayed["nitrite"]; running_inflow_mb["nitrate"] += total_decayed["nitrite"]
#   storage["nitrite"] += total_decayed["ammonia"]; running_inflow_mb["nitrite"] += total_decayed["ammonia"]
# drivers "wsimod.<ImperviousSurface|PerviousSurface>.decays.constant" / ".exponent", (..., n, K), label (node, surface)
# Land.mass_balance gains {"DecayTank.decayed": "out"}
```

- **Labels (X35).** `"wsimod.total_decayed"` labels a surface row `(node, "surfaces.<name>")`, the attribute path. 3a's `"wsimod.storage"` surface label `(node, "surface:<name>")` does not change: two keys, two label forms, both documented.

- [ ] **Step 1: Write the failing tests** in `test_land.py` (and `test_build.py`):
  - `test_the_carry_over_converts_yesterdays_decay`. Day d decays ammonia 1.5 and nitrite 0.4 at the end of day. Day d + 1's `run` first adds 0.4 to nitrate and 1.5 to nitrite, in that order, and adds both to `running_inflow_mb`. On day 0 it adds zero.
  - `test_without_nitrite_the_decayed_mass_is_lost`: pollutants without `nitrite`. There is no carry-over, the storage still decays, and `"DecayTank.decayed"` holds the lost mass.
  - `test_nitrite_needs_nitrate_and_ammonia`: `match="nitrite"`.
  - `test_surface_decay_reads_the_lands_air_temperature`: the factor uses the `Land`'s `temperature` row, not the soil temperature (`land.py:837-839`).
  - `test_a_decays_override_reaches_only_its_surface` (D13, `importorskip`). Two surfaces are built without a `decays` key, and an override `{phosphate: {constant: 0.1, exponent: 1.0}}` is applied to the first. noodl decays only the first. WSIMOD's second surface's `decays` is the same dict object as the first's (asserted with `is`; the defect row).
  - `test_a_surface_without_decays_refuses_a_changed_inert_driver` (D15).
  - `REFUSALS`: `(_node(type_="Land", surfaces=[{..., "decays": {"a": 1}}]), "3b")` becomes `match="'a'"` (an unconfigured pollutant).
  - `test_the_total_decayed_state_round_trips`: `initial_state(values={"wsimod.total_decayed": {("land", "surfaces.rural"): {"ammonia": 2.0}}})` reaches day 0's carry-over. `state_labels("wsimod.storage")` still holds `("land", "surface:rural")`.
  - `test_quantity_only_driver_shapes` extended: no surface `decays.*` key in quantity-only.
  - Whole model and replay:
    - `surface_decay_nitrification`: impervious and pervious surfaces with VQIP initial storages (`tanks.py:36-42`) and their own `decays` on ammonia, nitrite and phosphate (D13); WSIMOD's 12 pollutants; live soil temperature. There is no `River` (nitrate would start biochemistry): water drains to a `Node` → `Waste` and a `Groundwater` without decays. `replay=("<land>",)`.
    - `land_run_twice`: a custom orchestration with `Land.run` twice and decaying surfaces, at value parity (X16).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement. 3a's omission note in the `Surface` docstring goes away.
- [ ] **Step 4:** Run the wsimod subset and `-m slow` on the demo tests; expected: PASS. quickstart in pollutant mode now runs the carry-over with empty decays, still bitwise. Add the cases to the listed-features test. Docs: the carry-over, its doubling under a repeated `Land.run`, the shared-default row, and the two label forms.
- [ ] **Step 5:** Commit: `feat(wsimod): surface decay and the nitrite carry-over`.

### Task 9: `DecayQueueTank` and the decaying `QueueGroundwater` (`tanks.py:645-696`, `arcs.py:731-805`; X12, X13, X28, X30)

**Implementer:** most capable (the lagged extraction, the order of `total_decayed`, the swapped merge, the missing `update_queue`).

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/arcs.py`: a new `DecayArcAlt(AltQueueArc)`, used here as the internal arc;
  - `tanks.py`: a new `DecayQueueTank(QueueTank)`;
  - `storage.py`: `QueueGroundwater._make_tank`, `bind`, `store_paths`/`carried_paths`, `validate`, `mass_balance`, the `number_of_timesteps` override;
  - `config.py`: `_refuse_decays` removed;
  - `classes.py`: the `QueueGroundwater` override keys gain `"decays"` and `"number_of_timesteps"`; `validate` refuses the latter on a non-decaying node;
  - `day.py`: `_state_layout` rows `(node, "tank.internal_arc")`;
  - `engine.py`: sites;
  - `tests/verification/_wsimod_reference.py`: the capture of the tank's `total_decayed` and queue;
  - `tests/apps/wsimod/_replay.py`;
  - tests: `tests/apps/wsimod/test_tanks.py`, `tests/apps/wsimod/test_storage.py`, `tests/apps/wsimod/test_build.py`, `tests/verification/_wsimod_cases.py`, `tests/verification/test_wsimod_parity.py` (crash test);
  - docs.

**Interfaces:**
- Consumes: `AltQueueArc` and its `present` register (3a, W48), `QueueTank`, `Decay` (Task 5), `filename` overrides (Task 7), `"wsimod.total_decayed"` (Task 8).
- Produces:

```python
class DecayArcAlt(AltQueueArc):            # arcs.py:731-805
    decay: Decay; total_decayed: Tensor     # stores() gains "total_decayed"
    def send_push_request(self, vqip, *, active, tag="default", force=False, time: int = 0) -> Tensor
        # enter_arc counts the undecayed request (vqip_in, flow_in); then decay; then into the register as 3a;
        # each entry's diff added to total_decayed in call order (:752-778)
    def end_timestep(self) -> None
        # reset vqip_in, vqip_out, total_decayed, flow_in, flow_out; old = register;
        # for i in 0 .. L-1 where present[i+1]: reg[i] = decay(old[i+1]), reg[i+1] = empty, total += diff;
        # reg[0] = sum_vqip(reg[0], decay(old[0])), total += diff(slot 0)   (:780-805, X30)
class DecayQueueTank(QueueTank):           # tanks.py:645-696
    internal_arc: DecayArcAlt              # number_of_timesteps default 1 (:648, X12)
    def end_timestep(self) -> None
        # storage = extract_vqip(storage, internal_arc.total_decayed);
        # ctx.term("DecayQueueTank.decayed", that total_decayed, all-true); internal_arc.end_timestep();
        # no update_queue (:689-696, X13)
# QueueGroundwater.bind binds the internal arc's Decay (D15); carried paths gain
#   "tank.internal_arc.total_decayed"; mass_balance gains {"DecayQueueTank.decayed": "out"};
#   sites node:N/tank/internal_arc/make_decay/cap:<p>
```

- [ ] **Step 1: Write the failing tests** in `test_tanks.py` (and `test_storage.py`):
  - `test_decay_queue_tank_equals_wsimods_on_random_sequences[1|0]` (`importorskip`): a WSIMOD `DecayQueueTank` with a dummy parent, built with its default `number_of_timesteps=1` and with `0`. 60 days mixing `push_storage` at times 0–2, `pull_storage`, `pull_storage_exact` and `end_timestep`. After every call, `storage`, `active_storage`, the dense register, `present` and `total_decayed` are `torch.equal` to WSIMOD's.
  - `test_the_decay_reaches_storage_a_day_late`: one push of phosphate 5 on day 0. At the end of day d, `storage` loses exactly the `total_decayed` held before the reset: day d's entry decay plus day d − 1's end-of-day decay (X13). The queue holds 5 less every diff applied so far. Asserted for days 0–3 against the stored diffs.
  - `test_the_end_of_day_merge_is_swapped`: with `temperature` a blended pollutant, the new slot 0 is `sum_vqip(decay(old[1]), decay(old[0]))`. `sum_vqip(decay(old[0]), decay(old[1]))` differs in the last bit for the chosen values (asserted, so the test discriminates).
  - `test_water_reaching_slot_zero_waits_for_the_next_push` (Review Focus 1): no push on day d + 1. The water stays in slot 0, is decayed again at the end of day d + 1, and joins `active_storage` only at `distribute`'s `update_queue` (`storage.py:352`).
  - `test_active_storage_never_decays`: `active_storage` is unchanged by `end_timestep`.
  - `test_decaying_delays_a_queue_groundwater_by_one_day` (`importorskip`): time-area `{0: 0.5, 1: 0.5}` with `decays`. WSIMOD's queue keys are 1 and 2 after a push, and noodl's present slots are 1 and 2.
  - `test_number_of_timesteps_override` (`importorskip`): on a decaying `QueueGroundwater`, the override `number_of_timesteps: 0` sets the tank's and the internal arc's, as WSIMOD's (`tanks.py:682-685`). On a non-decaying one it is refused, with `match="number_of_timesteps"`.
  - `test_case_runs_where_wsimod_crashes_decay_queue_groundwater`: without its `filename` override, WSIMOD raises `TypeError` at the first push or end of day, and noodl refuses by name.
  - `test_quantity_only_driver_shapes` extended: no `wsimod.QueueGroundwater.decays.*` key in quantity-only, while the `DecayQueueTank` structure stays (X25).
  - Whole model and replay (D12):
    - `decay_queue_groundwater`: a `QueueGroundwater` with time-area `{0: 0.5, 2: 0.5}`, `decays` and a `_forcing_file` temperature override. Abstraction pulls go through `pull_set_active` by a `Reservoir`, and some days have water reaching slot 0 with no push before `distribute` (asserted from the captured queue). `replay=("<groundwater>",)`.
    - `decay_queue_groundwater_n0`: the same with `number_of_timesteps: 0`.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement. The order of `total_decayed` is X30's: reset; `+ diff(slot 1 … L)` ascending, where an absent key adds an exact-zero `diff`; `+ diff(slot 0)`; then the next day's entries in push order.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS (the mass balance closes: `held` counts the tank by `storage`, so the lagged extraction is the term). Add the cases to the listed-features test. Docs: the `DecayQueueTank` behaviours (X13), the one-day delay and its override (X12, X28), and `tank_*` (X40).
- [ ] **Step 5:** Commit: `feat(wsimod): DecayQueueTank and the decaying QueueGroundwater`.

### Task 10: User `DecayArc` and `DecayArcAlt` (`arcs.py:668-805`; X24, X29, X33, R1)

**Implementer:** most capable (the order of the `DecayArc` term over waiting rows, and its precondition).

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/arcs.py`: a new `DecayArc(QueueArc)`; `DecayArcAlt` as a user arc; `Arc.bind_decay`; `ARC_CLASSES`;
  - `classes.py`: `ARC_SCHEMAS` gains both, with keys `name`, `type_`, `in_port`, `out_port`, `capacity`, `preference`, `number_of_timesteps`, `decays`; `OUT_OF_SCOPE` loses them;
  - `config.py`: `ArcSpec.decays: DecaySpec | None = None`; `_read_arc` accepts `decays` only on `Decay*` arcs and refuses `parent` on every arc; the message no longer names 3b; an arc `decays` override is refused;
  - `structure.forcing_readers`: each `Decay*` arc's `in_port` reads `temperature`, each node once, a `Catchment` included (X29);
  - `day.py`: `_driver_layout` gains the arc `decays.*` drivers and the Catchment temperature row; `_bind` calls `bind_decay` after the node loop (D15);
  - `catchment.py`: `bind` accepts `temperature` without touching its concentration;
  - `run.py`: `EVENT_NAMES` gains `"decay_arc_order_unknown"` (D18, D21);
  - `engine.py`: sites;
  - tests: `tests/apps/wsimod/test_queue_arc.py`, `tests/apps/wsimod/test_tanks.py`, `tests/apps/wsimod/test_config.py`, `tests/apps/wsimod/test_build.py`, `tests/verification/_wsimod_cases.py`;
  - docs.

**Interfaces:**
- Consumes: `QueueArc` rows and `since` (3a, D1, unchanged), its tracing hook (`begin_tracing`/`end_tracing`, W59), `DecayArcAlt` (Task 9), `Decay`/`check_inert` (Task 5), `Arc.mass_balance` (Task 4).
- Produces:

```python
class Arc:
    def bind_decay(self, constant: Tensor, exponent: Tensor) -> None
        # Decay* arcs: check_inert, then decay.bind(ctx, constant, exponent, in_port.params["temperature"]) (D15)
class DecayArc(QueueArc):                       # arcs.py:668-728
    mass_balance = {"DecayArc.decayed": "out"}
    decay: Decay
    entry_rank: dict[tuple[str, int], int]      # (direction, j) -> position in the day's merged static
                                                # call sequence, fixed on the tracing day (D2 of 3a)
    # entry (push :396, pull :354): enter_arc undecayed, then decay, then enqueue; a push's backflow is
    #   the decayed water; each entry's diff -> ctx.term("DecayArc.decayed") in call order
    def end_timestep(self) -> None
        # decay every present row in place; the term in the order below; the event of D21; then since as 3a (:710-728)
class DecayArcAlt(AltQueueArc):                 # user arc: in_port temperature, mass_balance {"DecayArcAlt.decayed": "out"}
ARC_SCHEMAS["DecayArc"], ARC_SCHEMAS["DecayArcAlt"]
# drivers "wsimod.<DecayArc|DecayArcAlt>.decays.constant" / ".exponent", (..., n_arcs_of_class, K), label the arc
# sites arc:A/make_decay/cap:<p>; event "decay_arc_order_unknown" (D21)
```

- **The end-of-day term order (not determined by the signature).** WSIMOD's list order is entry age descending, then the call position on the entry day. A row `(d, i, j)` has age `i + since[d]` (per instance, so not static). Add the diffs as follows:

  ```
  for age in range(2n - 1, -1, -1):                        # static
      for (d, i, j) in rows sorted by entry_rank[(d, j)]:  # static
          ctx.term("DecayArc.decayed", diff[d, i, j], active & present[d, i, j] & (i + since[d] == age))
  ```

  Every row is added exactly once. A masked add is an exact `+0.0` (`ctx.term` sums with `sum_vqip` from empty). `since` is read before it advances.

  **Precondition (R1).** Because `since` saturates at `n`, a due row that is still waiting has an age noodl cannot reconstruct. The sum is bitwise WSIMOD's while no end of day finds due-and-waiting rows in both directions. Where one does, the event `"decay_arc_order_unknown"` is recorded (D21), and only the term's last bit may differ. Delivered values and rows are exact in every case.

- [ ] **Step 1: Write the failing tests** in `test_queue_arc.py` (and `test_tanks.py`, `test_config.py`, `test_build.py`):
  - `test_decay_arc_matches_wsimod_on_irregular_requests` (`importorskip`): 3a's W21 mapping (`test_queue_arc_delivery_matches_wsimod_on_irregular_pushes`, as corrected by W58) with the arc made a `DecayArc` (`n = 2`). Decays on `phosphate` and on the non-additive `temperature`; the `in_port` has temperature forcing. Every day's `vqip_out`, the replies, the receiver's storage and every queued row are `torch.equal` to WSIMOD's. The same with `n = 0`: each request decays once and is delivered at once.
  - `test_a_pulled_request_decays_after_it_leaves_the_sender` (`importorskip`): a `Reservoir` pulling through a `DecayArc` (`n = 1`). The sender loses the undecayed volume, and the reply on the next day is the decayed request.
  - `test_backflow_is_the_decayed_water`: a rejecting receiver. The backflow equals the decayed request, and `vqip_in` falls by it (`arcs.py:399-405`).
  - `test_the_decay_arc_term_follows_wsimods_list_order` (Review Focus 2). Waiting rows of both directions with three ages, chosen inside the precondition (no end of day has due-and-waiting rows in both directions; asserted from `since` and `present`). The term equals the per-row diffs summed in WSIMOD's list order, bitwise. Summing in ascending age differs in the last bit for the chosen values (asserted, so the test discriminates).
  - `test_the_order_unknown_event_marks_the_precondition`. Rows due and waiting in both directions at one end of day give `"decay_arc_order_unknown"` true on that day and false on every other day. The delivered values and rows are still `torch.equal` to WSIMOD's (`importorskip`).
  - `test_decay_arc_alt_matches_wsimod_on_irregular_pushes` (`importorskip`, in `test_tanks.py`): a user `DecayArcAlt` (`n = 2`) with irregular pushes and `temperature` as a blended pollutant. The register, `present`, `vqip_out` and `total_decayed` are `torch.equal` every day.
  - `test_a_decay_arc_out_of_a_catchment_reads_its_temperature_row`: the arc's factor uses the catchment's `"wsimod.temperature"` row. Changing the row leaves `"wsimod.concentration"` unchanged, and the reverse holds too (X29). In a quantity-only run the arc runs without a `temperature` pollutant.
  - `test_decay_arcs_bind_after_their_in_ports` (D15): the factor uses the `in_port`'s temperature of the same day. A non-inert entry at an absent key is refused, naming the arc.
  - `REFUSALS` updates:
    - `(_arc(type_="DecayArc"), "3b")` becomes accepted;
    - `(_arc(type_="Arc", decays={}), "decays")` stays, and its message no longer contains `"3b"`;
    - new rows `(_arc(type_="DecayArc", parent="x"), "parent")` and `(_arc(type_="Arc", parent="x"), "parent")`;
    - an arc override `{"decays": {...}}`, `match="decays"`;
    - a `Decay*` arc whose `in_port` lacks `("temperature", date)`, refused naming the node.
  - `test_quantity_only_driver_shapes` extended: no arc `decays.*` key in quantity-only.
  - Whole model:
    - `decay_arc`: a `DecayArc` with `n = 2` and a second one with `n = 0`, push and pull requests, a rejecting receiver (decayed backflow), and an `in_port` with temperature forcing. It is built inside the precondition: the two directions' due rows never wait at one end of day, and noodl's event is false on every day.
    - `decay_arc_alt`: a `DecayArcAlt` with `n = 2`, irregular pushes, and `temperature` as a blended pollutant.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS (both cases at parity in both modes, the mass balance closing with the arc terms). Add the cases to the listed-features test. Regenerate the catalogue. Docs:
  - the user `Decay*` arcs in the formats page (keys, `parent` refused);
  - their temperature source (the `in_port`; the Catchment's two drivers, X29);
  - the precondition sentence, verbatim: "exact while no end of day finds due-and-waiting rows in both directions";
  - the event.
- [ ] **Step 5:** Commit: `feat(wsimod): user DecayArc and DecayArcAlt`.

### Task 11: The arc replay harness (`capture_arc_calls`; X24, R1)

**Implementer:** standard.

**Files:**
- Modify: `tests/verification/_wsimod_reference.py` (`capture_arc_calls`), `tests/apps/wsimod/_replay.py` (`ScriptedPort`, `replay_arc`), `tests/verification/_wsimod_cases.py` (`Case.replay_arcs`, D16; set on `decay_arc` and `decay_arc_alt`), `tests/verification/test_wsimod_replay.py`.

**Interfaces:**
- Consumes: `DecayArc`, `DecayArcAlt`, `entry_rank`, the event of D21 (Task 10), 3a's `ArcCall` and `NodeCall` dataclasses.
- Produces:

```python
@dataclass
class ArcDay:            # one arc, one day
    day: int
    calls: list[ArcCall]                       # request calls, update_queue, with the out_port's replies
    entry_diffs: list[tuple[int, dict]]        # (static call index, diff) of each entry make_decay
    end_diffs: list[tuple[tuple, dict]]        # (row, diff) of each end-of-day make_decay, in list order
    queue_at_end: list[dict]                   # WSIMOD's queue before end_timestep: direction, time, vqip
    total_decayed_before_reset: dict; total_decayed_after: dict
    pre: dict[str, object]; post: dict[str, object]   # rows, since or register, present
@contextmanager
def capture_arc_calls(model, arcs: Sequence[str]) -> Iterator[dict[str, list[ArcDay]]]
class ScriptedPort(Node):   # returns the captured replies in order; asserts the requests within the class bound
def replay_arc(network: WsimodNetwork, name: str, days: list[ArcDay], *, scale: float) -> None
def precondition_holds(day: ArcDay, n: int) -> bool
    # from WSIMOD's queue: not (a due request, time 0, waits in the push direction and one in the pull direction)
```

- [ ] **Step 1: Write the tests** in `test_wsimod_replay.py` (`importorskip`):
  - `test_arc_replay_matches_wsimod[<case>-<arc>]` over every `(case, arc)` in `CASES[*].replay_arcs`. The noodl arc runs alone between `ScriptedPort`s. Every row, reply and per-request `diff` is bitwise equal (bound 0).
  - `test_the_decay_arcs_cases_hold_the_order_precondition[<case>-<arc>]` (R1): `precondition_holds` is true on every captured day of `decay_arc`, from WSIMOD's own queue, and noodl's `"decay_arc_order_unknown"` is false on every day.
  - `test_the_decay_arc_term_equals_wsimods_rewindowed_diffs[<case>-<arc>]`. For each day d, the reference is day d's `entry_diffs` in call order, then its `end_diffs` in list order, summed with `sum_vqip` from empty. noodl's `"DecayArc.decayed"`/`"DecayArcAlt.decayed"` for day d must equal it exactly. A case whose captured days break the precondition falls back to W19: the difference is measured, and a named bound is asserted with its cause written in the test. No current case does.
  - `test_wsimods_total_decayed_is_its_own_window`. On every day, WSIMOD's `total_decayed_before_reset` equals the day d − 1 `end_diffs` plus the day d `entry_diffs`, summed in that order, bitwise. This checks the harness.
  - `test_the_arc_replay_notices_a_difference`: a one-ulp change of one captured diff is reported, naming the arc, day and row.
- [ ] **Step 2:** Run them; expected: PASS. A difference is debugged to its cause before any bound is written (W19).
- [ ] **Step 3:** Run the wsimod subset; record the arc capture times in the progress file.
- [ ] **Step 4:** Commit: `test(wsimod): user Decay arcs replay WSIMOD's captured calls bitwise`.

### Task 12: River biochemistry: the process (`storage.py:723-898`; X14, X15, X17, X31)

**Implementer:** most capable (the four caps' operand orders, the registers' sums, the guard masks and the branch-dependent `din`).

**Files:**
- Modify:
  - `src/noodl/apps/wsimod/storage.py` (`River`): `parameter_defaults` gains the ten parameters; `max_temp_lag` and `max_phosphorus_lag` are structural; `update_depth`, `get_din_pool`, `biochemical_processes`, `calculate_discharge`, `end_timestep`; the registers as stores; `bind`; `validate`; `mass_balance`;
  - `classes.py`: the `River` override keys gain the 13 attributes `uptake_PNratio`, `bulk_density`, `denpar_w`, `T_wdays`, `halfsatINwater`, `hsatTP`, `limpppar`, `prodNpar`, `prodPpar`, `muptNpar`, `muptPpar`, `max_temp_lag`, `max_phosphorus_lag` (`storage.py:620-644`); `bulk_density` goes to `unread` (D10);
  - `day.py`: `STATE_KEYS` gains `"wsimod.river_temperature_lag"` and `"wsimod.river_phosphorus_lag"`, both in `_FLAT_KEYS` and `_NOT_HELD`, units `degC` and `kg`; the drivers and units per spec Inputs; D9;
  - `structure.forcing_readers`: the River reads `temperature` when `nitrate` is configured;
  - `run.py`: `EVENT_NAMES` gains `"river_nitrogen_lost"` (D18);
  - `engine.py`: sites;
  - `tests/apps/wsimod/test_storage.py`;
  - docs.

**Interfaces:**
- Consumes: `pow_python` (Task 3), `dmin`/`dmax`, `Ctx.term`/`event`, the X8 window (Task 4), `unread` (Task 7).
- Produces:

```python
class River(Storage):
    lag_sizes: tuple[int, int]                    # (max_temp_lag + 1, max_phosphorus_lag + 1), structural
    lagged_temperatures: Tensor                   # (..., max_temp_lag + 1), slot 0 the oldest, zeros initially (X17)
    lagged_total_phosphorus: Tensor               # (..., max_phosphorus_lag + 1)
    def update_depth(self) -> Tensor              # S.V / area   (:723-725)
    def get_din_pool(self) -> Tensor              # 0 + S["ammonia"] + S["nitrate"]   (:727-736)
    def biochemical_processes(self, *, active: Tensor) -> tuple[Tensor, Tensor]   # (in_, out_), :738-898
    def calculate_discharge(self, *, active: Tensor) -> None
        # with "nitrate": in_, out_ = biochemical_processes; ctx.term("River.bio_in", in_), ("River.bio_out", out_)
        # summed over the day's calls (X16)
    mass_balance = {"River.bio_in": "in", "River.bio_out": "out"}
# state "wsimod.river_temperature_lag" (..., n_slots), labels (river, i); "wsimod.river_phosphorus_lag" likewise
# drivers "wsimod.River.<p>", (..., n_rivers), p in uptake_PNratio denpar_w T_wdays halfsatINwater hsatTP
#   limpppar prodNpar prodPpar muptNpar muptPpar; units per spec Inputs; stores "lagged_temperatures",
#   "lagged_total_phosphorus" (D8)
```

- **Transcription.** Steps 1–8 of the spec's River biochemistry section, in that order and with its operation order.
  - Steps 2 and 3 run under `active`, dry days included. Steps 5–8 run under `active & ~dry & ~frozen`, where `dry` is `S.V < FA` and `frozen` is `T ≤ 0`.
  - Each register shift is `write(active, cat(reg[..., 1:], new[..., None]), reg)`.
  - Each window sum starts from `+0.0` and adds its last `min(10, size)`, `min(20, size)` or `size` slots one at a time, oldest first (X31). The phosphorus sum is divided by `max_phosphorus_lag`.
  - The caps use their operand order: `dmin(denitri_water, 0.5·din)`, and `dmin(0.5·pool, x)` at every other cap.
  - The `max(0, ·)` sites are `dmax(0.0, x)`.
  - The two powers are `pow_python(2.0, (T - 20) / 10)` and `pow_python(T_pos / 20, 0.3)`.
  - Each branch is a decision with its spec site name: `dry`, `frozen`, `cold`, `denitrification_cap`, `din:denitrification`, `totalphosfcn`, `production`, `production_cap:N`, `production_cap:P`, `din:production`, `mineralisation_cap:N`, `mineralisation_cap:P`, `din:mineralisation`, `macrophyte_temperature`, `macrophyte_factor`, `macrophyte_cap:N`, `din:macrophyte`, `macrophyte_phosphorus`, `macrophyte_cap:P`. `call` is the index of the day's `calculate_discharge` call.
  - `din` is recomputed where the spec says so.
  - The event `"river_nitrogen_lost"` holds where mineralisation runs with `din = 0` and the capped `minprodN > 0`.

- [ ] **Step 1: Write the failing tests** in `test_storage.py`:
  - `test_one_biochemistry_call_by_hand`. A river at a hand-chosen point away from every guard and cap, with every intermediate of steps 1–8 written in the test as Python expressions from the spec's defaults table: `1 / 7.2`, `0.0015`, `5`, `1.5 * 1e-3`, `0.05 * 1e-3`, `0.1 * 1e-3`, `0.001`, `0.0001`, `0.001`, `0.0001`, `20`, `365`. `S`, `in_` and `out_` are `torch.equal` to them.
  - `test_biochemical_processes_equals_wsimods_on_random_states` (`importorskip`). A WSIMOD `River` with the six pollutants and the registers pre-filled to random lengths 0–366. 500 random states and temperatures in `[-5, 30]`, including `V < FA`, `T ≤ 0`, `0 < T < 5`, `din = 0` and a low P. After one call, `tank.storage`, `bio_in`, `bio_out` and both lag lists, padded by a local `pad_register`, are `torch.equal`.
  - `test_the_registers_reproduce_pythons_sums_from_a_cold_start`. On calls 1, 10, 25 and 400, the 10- and 20-day means and the 365-day mean equal `sum(list[-10:]) / 10`, `sum(list[-20:]) / 20` and `sum(list) / 365` over WSIMOD's list, bitwise. The list holds 21 and 366 entries once full. With `max_temp_lag = 5` the register has 6 slots and both windows are the whole register (Review Focus 3).
  - `test_minus_zero_phosphorus_enters_the_register_as_python_sums_it` (X31): a slot holding `-0.0` gives Python's `sum` result.
  - `test_mineralisation_without_din_destroys_nitrogen_and_says_so`. With `din = 0` and a mineralising state, `org-nitrogen` falls by the capped `minprodN`, nothing gains it, `bio_out` lists the loss, and `"river_nitrogen_lost"` is true (X15).
  - `test_minus_zero_minprodN_moves_nothing`: `totalphosfcn = 0` and `tempfcn < 0` give `minprodN = -0.0`. The mineralisation branch is taken and nothing moves.
  - `test_river_biochemistry_refusals` (parametrised, `match=`), each also at `bind` where it is a driver:
    - each of the six pollutants missing;
    - `T_wdays: 0`;
    - `length: 0` (`area`);
    - `halfsatINwater: 0` and `-1e-3` (O4);
    - `max_temp_lag: -1` and `20.5`;
    - `max_phosphorus_lag: 0` and `365.5` (X37).

    The 3a test `test_a_river_with_nitrate_is_refused` is replaced.
  - `test_river_overrides_set_the_thirteen_attributes` (`importorskip`): each override equals WSIMOD's attribute after `add_overrides`, and `bulk_density` is listed in `unread`.
  - `test_river_biochemistry_inputs_exist_only_when_it_runs` (D9): a nitrate River built quantity-only has a `"wsimod.temperature"` row but no `wsimod.River.<p>` drivers and no register rows.
  - `test_gradcheck_on_one_biochemistry_call` away from every guard and cap.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the wsimod subset; expected: PASS. Regenerate the catalogue. Docs: the biochemistry, its quirks (X14, X15), the registers and warm starts, the kinks and the singularity (spec, Differentiability), and the River drivers and units.
- [ ] **Step 5:** Commit: `feat(wsimod): River biochemistry with WSIMOD's lagged temperature and phosphorus`.

### Task 13: River biochemistry: the cases (X16, X32, X39)

**Implementer:** most capable (a case that places every branch in its first 400 days).

**Files:**
- Modify: `tests/verification/_wsimod_reference.py` (the capture of `calculate_discharge` records both lag lists and `data_input_dict[("temperature", t)]`; `pad_register`), `tests/apps/wsimod/_replay.py` (loads the registers), `tests/verification/_wsimod_cases.py` (`Case.slow_days`, D24), `tests/verification/_wsimod_parity.py` (D17), `tests/verification/test_wsimod_parity.py`, `tests/verification/test_wsimod_replay.py`, docs (verification rows).

**Interfaces:**
- Consumes: `River.biochemical_processes` and the registers (Task 12), 3a's `assert_matches_wsimod`.
- Produces:

```python
@dataclass(frozen=True)
class Case:                                   # 3a's fields, replay_arcs (Task 11), plus
    slow_days: int | None = None              # D24
def pad_register(values: list[float], size: int) -> list[float]   # WSIMOD's list, zero-padded in front (X17)
def assert_matches_wsimod(..., expected_mass_balance_prints: Sequence[str] = ()) -> dict   # D17
```

- [ ] **Step 1: Write the tests:**
  - Case `river_nutrients` (spec, Purpose-built cases), with `days=400` and `slow_days=1100`. It has two rivers in a chain, fed by catchments that carry all six pollutants, and one river overridden to `max_temp_lag = 5` with other biochemistry overrides. `replay=` both rivers. The builder places each of these inside the first 400 days (R3):
    - frost (`T ≤ 0` after relaxation);
    - days between 0 and 5 °C;
    - a dry day (`V < FA`, from an upstream abstraction);
    - the DIN-free stretch (`"river_nitrogen_lost"`);
    - a low-P river crossing both `totalphosfcn` outcomes;
    - production and mineralisation;
    - every 50 % cap: denitrification, production N and P, mineralisation N and P, macrophyte N and P.

    The full 1100 days add seasons and turn the 366-slot register over three times.
  - `test_case_matches_wsimod[river_nutrients-*]` runs 400 days by default at value parity without decisions. `test_case_matches_wsimod_on_its_slow_days[<case>-<mode>]` (`slow`, D24) runs every case that has `slow_days`, so `river_nutrients` runs its 1100 days.
  - `test_river_nutrients_reaches_every_branch` (`slow`). With decisions recorded, it asserts on the 400-day prefix and again on the full 1100 days that each site above occurs active on at least one day: `frozen`, `cold`, `dry`, both `totalphosfcn` outcomes, `production` both ways, each `*_cap:*` and `denitrification_cap` taking the cap, and `"river_nitrogen_lost"`.
  - Case `river_double_discharge`: a custom orchestration with `River.calculate_discharge` twice. `test_case_matches_wsimod[river_double_discharge-pollutants]` passes `expected_mass_balance_prints=("<river>",)` (D17). WSIMOD prints its River mass-balance error on every day, noodl's residual is 0 on every day, and the values are at parity.
  - `test_an_expected_mass_balance_print_must_appear_every_day`: removing one day's print from the reference fails the check, naming the day.
  - Replay: `test_replay_matches_wsimod[river_nutrients-<river>]` over the sampled days (W15), with both registers loaded from the capture and compared after every call.
- [ ] **Step 2:** Run them, and `-m slow`; expected: PASS.
- [ ] **Step 3: Measure (X39).** Run `pytest -n 4 --no-cov --durations=30 <the wsimod row>`. Record `river_nutrients`' default items and their times in the progress file. If one is still among the 30 slowest at 400 days, record it and the cause there. The cut is already applied, so nothing more is cut without the lead.
- [ ] **Step 4:** Add the cases to the listed-features test. Docs: the verification rows for the River cases, the 400/1100-day composition, and the W90/W91 exception.
- [ ] **Step 5:** Commit: `test(wsimod): river cases against WSIMOD, every branch in the default horizon`.

### Task 14: Batches, stepping, state and quantity-only structure

**Implementer:** standard.

**Files:**
- Modify: `tests/apps/wsimod/test_run.py`, `tests/apps/wsimod/test_build.py`.

**Interfaces:**
- Consumes: every 3b class and driver (Tasks 2–13), `initial_state`/`initial_drivers` (3a).

- [ ] **Step 1: Write the tests:**
  - `test_a_batch_of_four_3b_parameter_sets_equals_its_single_runs[<case>]`, over `decay_storage`, `wtw_treatment`, `surface_decay_nitrification`, `decay_queue_groundwater`, `decay_arc` and `river_nutrients` (first 100 days). There are 4 instances with different decay constants and exponents, WTW constants and exponents, soil weights, and `denpar_w`/`prodNpar`, chosen so that at least two instances take different cap decisions on one day (asserted from `"decisions"`). Each instance is `torch.equal` to its single run.
  - `test_step_by_step_equals_wsimod_run[<case>]`: extend 3a's parametrisation with `decay_queue_groundwater`, `decay_arc` and `river_nutrients` (60 days). The new state keys are carried through `model.step`.
  - `test_the_new_state_round_trips`: `initial_state(values=...)` set by label on `"wsimod.total_decayed"`, `"wsimod.river_temperature_lag"` and `"wsimod.river_phosphorus_lag"`. A run warm-started from another run's last state equals the continuation of that run, `torch.equal`.
  - `test_quantity_only_volumes_never_depend_on_3b_values`: on `decay_queue_groundwater`, `wtw_treatment` and `river_nutrients`, vary every 3b driver that a quantity-only run lists as ignored. The volumes are `torch.equal`.
  - `test_quantity_only_lists_the_3b_parameters`: decay values, the WTW pollutant entries and the River biochemistry parameters appear in `model.ignored_by_quantity_only`. A decaying `QueueGroundwater` keeps `DecayQueueTank`'s end of day in quantity-only (X25): its volumes differ from the same case without `decays` (asserted), and equal the pollutant run's.
  - `test_wsimod_is_never_imported` re-runs unchanged.
- [ ] **Step 2:** Run them; expected: PASS (a failure is a defect, fixed in its task's code with a test there).
- [ ] **Step 3:** Commit: `test(wsimod): batches, stepping and state through the 3b processes`.

### Task 15: Gradients

**Implementer:** standard.

**Files:**
- Modify: `tests/verification/test_wsimod_gradients.py`.

**Interfaces:**
- Consumes: `wsimod_run(..., diagnostics={"decisions": True})`, `CASES`, the 3a `Spec` table and kink rule (W92/W93).

- [ ] **Step 1: Write the tests.** Extend `test_gradient_matches_central_differences[<parameter>]` (relative step `1e-6`, agreement `1e-6`, the ±h decisions identical, the kink rule) with the following `Spec` entries. Each is checked below the cap and away from the spec's kinks.

  | Parameter | Case | Scalar |
  |---|---|---|
  | `decays.constant` and `decays.exponent` | `decay_storage`, a storage node | its final phosphate |
  | `decays.constant` | `decay_arc` | the receiver's phosphate |
  | `process_parameters.constant` and `.exponent`, a pollutant `liquor_multiplier` | `wtw_treatment` | the WWTW out-arc's phosphate |
  | `denpar_w`, `halfsatINwater`, `prodNpar`, `muptNpar`, `T_wdays` | `river_nutrients`, 120 days | the downstream river's final nitrate and org-nitrogen |
  | the three soil-temperature weights | `soil_temperature` | the percolation arc's temperature |
  | the temperature forcing | `decay_storage`, `wtw_treatment` (catchment temperature), `river_nutrients` | through decay, treatment and the river |

  Mark slow any entry whose run exceeds 10 s, as 3a's `SLOW` set does.
- [ ] **Step 2:** Run them, and `-m slow`; expected: PASS. Record the worst relative agreement per parameter in the progress file.
- [ ] **Step 3:** Commit: `test(wsimod): gradients through decay, treatment, soil temperature and the river`.

### Task 16: Documentation, benchmark, final checks

**Implementer:** standard.

**Files:**
- Modify: `docs/applications/wsimod.md`, `docs/formats/wsimod.md`, `docs/verification.md`, `tests/data/wsimod/README.md`, `benchmarks/wsimod_run.py`, `docs/catalogue/wsimod.md` (regenerated), `AGENTS.md` (the wsimod row's measured time only).

- [ ] **Step 1: Docs.** Complete `wsimod.md` with:
  - the ported processes, by class (the spec's Scope table);
  - the decay's temperature sources (the spec's table);
  - the River's quirks (X14, X15);
  - the `DecayQueueTank` behaviours (X13, X12/X28);
  - the user `DecayArc` term's precondition (R1) and its event;
  - the new refusals, kinks and events;
  - the defect table with noodl's behaviour;
  - the demo configurations (`config_no_deposition.yml`) and what still needs neutralising for 3c (each surface's `pollutant_load`);
  - `tank_*` (X40);
  - the `filename` override, its working-directory path and edge cases (X27, D23);
  - the catchment temperature held in two drivers (X29);
  - `model.unread` (D10);
  - the two surface label forms (Task 8);
  - the verification table with the measured numbers of Tasks 2–15.

  `formats/wsimod.md`: `decays`, the `Decay*` arcs, the River overrides, the WTW entries, the soil-temperature overrides, `filename`. `docs/verification.md`: the WSIMOD row covers the 3b classes, the cases and the worst measured values. Regenerate the catalogue.
- [ ] **Step 2: Benchmark.** `benchmarks/wsimod_run.py` gains oxford on `config_no_deposition.yml` with `pollutants=True` at batch sizes 1, 10 and 100, no grad and grad (spec, Cost), with no asserts. Run it once and record the result beside Task 0's medians.
- [ ] **Step 3: Final checks.** Run each of these; expected: all pass.
  - `ruff check .`;
  - the full `python -m pytest` with coverage (≥ 90 %);
  - `python -m pytest -q -n 4 --no-cov -m slow tests/apps/wsimod tests/verification/test_allocation_wsimod_*.py tests/verification/test_wsimod_*.py`;
  - `python -m mkdocs build --strict`.

  Record the wsimod row's default wall time against the 10-minute budget, and update AGENTS.md's measured-time sentence.
- [ ] **Step 4:** Commit: `docs(wsimod): the in-water processes, verification and benchmark (3b)`.

## Defect table: each row and its test

| Row (spec, WSIMOD defects) | noodl | Test (task) |
|---|---|---|
| `Groundwater` loses its `data_input_dict` | reproduced; forcing by `filename` override; refused without | `test_case_runs_where_wsimod_crashes_decay_groundwater`, `test_case_matches_wsimod[decay_groundwater-*]` (7) |
| The same in `QueueGroundwater` | the same | `test_case_runs_where_wsimod_crashes_decay_queue_groundwater`, `[decay_queue_groundwater-*]` (9) |
| `DecayQueueTank` default `number_of_timesteps=1` | reproduced; override accepted | `test_decaying_delays_a_queue_groundwater_by_one_day`, `test_number_of_timesteps_override`, `[decay_queue_groundwater_n0-*]` (9) |
| `River.calculate_discharge` assigns `bio_in`/`bio_out` | both calls' terms added | `test_case_matches_wsimod[river_double_discharge-pollutants]` with D17 (13) |
| Mineralisation with `din = 0` | reproduced, event | `test_mineralisation_without_din_destroys_nitrogen_and_says_so` (12); `river_nutrients` DIN-free stretch (13) |
| Phosphorus mean 366 ÷ 365, cold-start windows | reproduced | `test_the_registers_reproduce_pythons_sums_from_a_cold_start` (12); `river_nutrients` (13) |
| Total phosphorus mass vs concentrations | reproduced, documented | `test_river_nutrients_reaches_every_branch`, both `totalphosfcn` outcomes in 400 days (13, slow) |
| No `exponent` check (decay, treatment) | `exponent ≤ 0` refused | `test_read_decays_refusals` (5); `test_wtw_treatment_refusals` (3) |
| WTW solids unclamped | transcribed, event | `test_cold_influent_makes_negative_solids_and_says_so`; `wtw_treatment` (3) |
| `decays` naming `volume` | refused | `test_read_decays_refusals[volume]` (5) |
| Shared `decays={}` default | per element | `test_a_decays_override_reaches_only_its_surface` (8) |
| `Surface.run` converts at every call | transcribed | `test_case_matches_wsimod[land_run_twice-*]` (8) |

## Self-review

- **Spec coverage.**
  - Scope: `DecayObj` (5); `DecayTank` (5, 6, 8); `DecayQueueTank` (9); user `DecayArc`/`DecayArcAlt` (10, 11); the storage family with `decays` (6, 7, 9); River biochemistry (12, 13); WTW (3); the carry-over (8); the soil temperature (2).
  - Every 3a "3b" refusal is decided: `Storage`/`_refuse_decays` (6, 9); surface `decays` (8); `classes.py:278`/`:246` (6); `OUT_OF_SCOPE` (10); arc `decays` (10); River `nitrate` (12); WTW (3); soil temperature (2); deposition messages (2); `tanks.py:277` (5).
  - Approach rules 1–4 (Global Constraints; D4, D5; 5).
  - State (8, 9, 12; D7, D8). Inputs (2, 3, 6, 8, 10, 12; D9, D19, O2 in 3, 6, 8, 9, 10). Mass balance and X8 (4, then each term's task; D14, D22).
  - Validation (2, 3, 5, 6, 7, 8, 10, 12). Loading and overrides (3, 6, 7, 9, 12; D6, D10, D20, D23). Vocabulary, events (3, 5, 9, 10, 12; D18, D21). Replay harness (6, 8, 9, 11, 13).
  - The demos (1, 2, 3, 8). Defect table (above). Differentiability (5, 12 gradcheck; 15; docs 16). Verification 1–4 (2–15). Documentation (each task; 16).
- **Dependency order.**
  - Task 1's fixtures come first; Tasks 2 and 3 make the demos live.
  - Task 4's window precedes every end-of-day term (6, 8, 9, 10, 12).
  - `Decay` (5) precedes every decaying element (6–10).
  - `filename` overrides (7) precede `DecayQueueTank`'s cases (9).
  - `"wsimod.total_decayed"` (8) precedes its tank rows (9).
  - `DecayArcAlt` (9) precedes the user arcs (10), which precede their harness (11).
  - The River process (12) precedes its cases (13); `Case.replay_arcs` (11) and `Case.slow_days` (13) are added where first used.
  - Each case sits in the first task in which all its classes exist: `soil_temperature` (2), `wtw_treatment` (3), `decay_storage` (6), `decay_groundwater` (7), `surface_decay_nitrification` and `land_run_twice` (8), `decay_queue_groundwater` and `_n0` (9), `decay_arc` and `decay_arc_alt` (10), `river_nutrients` and `river_double_discharge` (13).
- **Interfaces across tasks.**
  - `pow_python`'s new home (3) serves 5, 9, 10 and 12.
  - `DayRecord.storage_end` and `Arc.mass_balance` (4) serve the terms of 6, 8, 9, 10 and 12. D22's test (4) guards the re-snapshot set against a later decision recorded at `end_timestep`.
  - `check_inert` and `Decay.bind` (5) are called from `Storage.bind` (6), `Land.bind` (8), `QueueGroundwater.bind` (9) and `Arc.bind_decay` (10), per D15.
  - `DecayArcAlt` (9) is reused as a user arc in 10.
  - `expected_mass_balance_prints` (13) is used only by `river_double_discharge`.
- **Bitwise everywhere.** No task plans an admission or a non-zero `replay_bound`.
  - The `DecayArc` term is bitwise inside R1's precondition. The cases are built inside it, Task 11 proves that from WSIMOD's queue, and a violating case would fall back to W19.
  - The River's register sums are designed bitwise (X31, tested in 12).
