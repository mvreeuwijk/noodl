# WSIMOD Core and Water Quantity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `apps.wsimod` reads WSIMOD 0.8.1's own `config.yml` and runs its daily simulation natively (batched, float64, differentiable, no `wsimod` import at runtime), matching live WSIMOD node by node on replayed inputs, on the quickstart and oxford demos, and on purpose-built cases.

**Architecture:** A transcription of WSIMOD's procedure (W2). Each WSIMOD node and arc is a Python object of a transcribed class with WSIMOD's handler dicts keyed by tag; its values are batched tensors. The call structure is static (it depends on the configuration only); every data-dependent branch is a detached boolean mask applied with `torch.where`, loops run exactly `MAXITER` masked rounds, and guarded divisions use safe denominators. One integrating closure, `WsimodDay`, runs a day on a layer-less noodl `Model`; `wsimod_run` loops it over the configuration's dates and records what `Model.run` records.

**Tech Stack:** Python 3.12, PyTorch (float64), PyYAML, pytest (+ xdist), `wsimod==0.8.1` (the `reference` extra; tests and the fixture script only).

**Spec:** `docs/superpowers/specs/2026-11-11-wsimod-core-design.md` (binding, read it whole; it is the authority wherever this plan is silent; corrected with this plan's review for D1 and the six drop sites). Rulings W1–W28: `.superpowers/sdd/2026-11-11-wsimod-core/progress.md`; the spec review `spec-review.md` and the plan review `plan-review.md` beside it. WSIMOD source: `C:/projects/noodl/.venv/Lib/site-packages/wsimod/`; every `file:line` below is relative to it.

## Global Constraints

- **Start point.** Worktree `C:/Users/mvr/AppData/Local/Temp/noodl-wt/wsimod-core`, branch `wsimod-core` (from `main` 08b9a2d plus the spec and plan commits). One PR, opened by the lead after a Fable review (W7); this plan never pushes.
- **Interpreter and tests.** `C:/projects/noodl/.venv/Scripts/python.exe`, `PYTHONPATH=src`, from the worktree root. "The core subset": `tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py`. "The wsimod subset": `python -m pytest -q -n 4 --no-cov <core subset> <the AGENTS.md wsimod row>`. `-m slow` when a task changes what a slow test compares.
- **Before each commit** `ruff check .`; **at the end of each task** `python -m mkdocs build --strict`. A task that adds or changes a public name runs `python scripts/gen_catalogue.py` (`tests/test_docs_catalogue.py`).
- **Git.** `git -c user.email=58522531+mvreeuwijk@users.noreply.github.com -c user.name="Maarten van Reeuwijk" commit`. Never `git stash`. Never push. Every commit message ends with a `Co-Authored-By:` line naming the model that wrote the commit (the implementer's own model; the lead uses the attribution its session gives it).
- **Project rules.** Physical names; no aliases; code, tests and docs change together; no wall-clock assertions (times are measured, never asserted); `ruff check .` passes; 90 % coverage on the full suite.
- **Transcription.** WSIMOD's values are transcribed, not reformulated (W2). Every transcribed behaviour cites its WSIMOD `file:line` in a docstring or comment. Where the spec and the source disagree, the source wins and the spec is corrected in the same commit.
- **Numerics (spec, Approach).** `torch.float64` throughout. Masks: every state write is `where(active, new, old)`. Loops bounded by `MAXITER = 5` run exactly 5 masked rounds. Safe denominators: `where(den != 0, den, 1)` under the branch's mask (double `where`). Ties: `min(a, b)` returns `a` unless `b < a`; `max(a, b)` returns `a` unless `b > a`; comparisons detached, values in the graph. Reductions WSIMOD walks sequentially are added one term at a time in its order (no `torch.sum` over such an axis). Unbounded is `UNBOUNDED_CAPACITY = 1e15`, never `inf`. `FLOAT_ACCURACY` is per model (default `1e-11`).
- **Reference.** Live WSIMOD 0.8.1 through `pytest.importorskip("wsimod")`; `reset_wsimod_globals()` before every WSIMOD build (Task 2). WSIMOD is never imported by `src/`.
- **Cases with a `River`** declare their own pollutant lists without `nitrate` (the spec refuses `River` biochemistry, `storage.py:917-924`; WSIMOD's defaults contain `nitrate`), keeping `temperature` (the WTW rule needs it). `Model.load` honours the lists (`model.py:232-238`), so WSIMOD skips biochemistry too.
- **Bounds (spec, Verification; W19).** Per compared series, `scale` = its largest magnitude over the run; every (day, element) within `1e-12 * scale`. A larger difference is admitted only when the noodl run's `"decisions"` record shows a branch with `|margin| <= 1e-9 * scale` on that day; the test names each admission (case, element, day, site) and asserts the count. A bound is loosened only with a measured cause written in the test. Mass balance closes within `1e-12 * scale` every day. Gradients: central differences at relative step `1e-6`, agreement `1e-6` relative, the ±h runs' `"decisions"` outcomes identical.
- **Suite budget (W15, W23).** The wsimod row's default run, WSIMOD captures included, fits in 10 minutes on the developer machine, by the measure-then-cut rule of Task 16.
- **Progress file** `.superpowers/sdd/2026-11-11-wsimod-core/progress.md` (git-ignored): baselines, measured times, admissions and open causes.
- **Implementer models.** cheap = Haiku, standard = Sonnet, most capable = Opus (the user's rule: Opus only for delicate numerics). Each task names one.

## Decisions taken where the spec is silent or ambiguous

Each is binding for the implementer; the plan review confirmed them as amended here.

- **D1 `QueueArc` rows** (now in the spec, Representation). Rows `(i, j)`: `i ∈ [0, n-1]` the entry day counted back from the arc's last call in that direction that reached `update_queue`, `j` the request call's index in the arc's static daily call order (site and round merged). `"wsimod.arc_queue_since"` per (arc, direction), saturating at `n`, starts at `n`. Due iff `since + i >= n`; delivery `i = n-1 … 0`, then `j`; survivors move to `i + since`. `J = max(J_push, J_pull)` per arc, unused rows permanently empty. A row's `average_flow` is `V/(n+1)`.
- **D2 Call enumeration.** `build_model` runs one tracing day (batch shape `()`, `torch.no_grad()`, day-0 forcing, values discarded) to enumerate every `QueueArc` request call and every distribute-loop site. Every later day asserts the same sequence (an internal error otherwise).
- **D3 Masks on checks.** Every arc call, checks included, takes `active`. An inactive call is computed and discarded; `ScriptedArc` skips it and returns zeros.
- **D4 Pull requests.** noodl never rescales a pull request's additive entries (no in-scope `pull_set` reads them, spec Protocol › Request mutation). The replay compares pull requests by volume (deep-copied at entry), push requests and every reply in full. The `QueueArc` zero-volume-pull defect row then holds by construction; its test checks the value is finite.
- **D5 Overrides.** The reader merges the `overrides` section into the constructor parameters (dict-valued `process_parameters`, `liquor_multiplier`, `pollutant_load`, `surfaces` merge by `update`, per node: the shared-default-dict defect is not reproduced). Derived quantities (`Surface.capacity`, pervious depths, `River` area, WTW volume constant, `Sewer` tank fields, `Land` residence tanks) are computed once from the merged values, as construction would. This equals WSIMOD's `apply_overrides` to ≤ 1 ulp (`land.py:858-884`); a test pins it per class.
- **D6 Driver names.** Nodes: `"wsimod.<configured class>.<parameter>"` (`NodeSpec.cls`), rows the nodes of that class in configuration order. Surfaces: `"wsimod.<surface class>.<parameter>"` over every `Land`'s surfaces in order. Arcs: every arc class shares `"wsimod.Arc.capacity"` and `"wsimod.Arc.preference"` `(..., n_arcs)`. A pollutant-keyed dict parameter is `(..., n, K)` (absent keys 0); `liquor_multiplier` is its volume entry `(..., n)`; time-area dicts are `(..., n, L+1)`. Pollutant entries of `process_parameters`/`liquor_multiplier` are checked neutral and are not drivers.
- **D7 Where refusals live.** The reader refuses keys, classes and formats (no physics). `build_model` refuses everything that depends on the parameters or on `pollutants=` (neutral processes, `decays`, porosity, leakage, preferences, travel times, structure, forcing), through `check_network` and each class's `validate`.
- **D8 Purpose-built cases** are Python mappings in `tests/verification/_wsimod_cases.py` in `Model.load`'s post-read form (inline `data_input_dict` keyed `(variable, date string)`); WSIMOD builds them by `Model.load`'s own steps (`model.py:232-274`, Task 2's `wsimod_model`), noodl by `read_wsimod_config(mapping)`. Each case is added in the first task where every class it needs exists. Committed directories: the two demos and `small`.
- **D9 `small`.** `tests/data/wsimod/small/`: `inflow` (`Catchment`) → `river` (`River`) → `outlet` (`Waste`), pollutant lists empty, 10 dates from 2000-01-01, forcing in an uncompressed `inflow-inputs.csv`. It replaces the old builder's callers and runs the docs example.
- **D10 Store order.** `"wsimod.storage"` rows and `tank_names`: nodes in configuration order, each node's tanks in `dir()` (alphabetical) order, as `record_all` walks them (`model.py:1119-1133`); in the state, each `Land` node's surfaces follow its tanks as `(node, "surface:<name>")`. Arc state labels: `"wsimod.travel_queue"` `(arc, i)`, `"wsimod.arc_queue"` `(arc, direction, i, j)`, `"wsimod.arc_queue_since"` `(arc, direction)`.
- **D11 Mass balance.** The terms transcribe each class's `mass_balance_in`/`_out`/`_ds` lists (`nodes.py:76-78` and every class's appends: `catchment.py:55-56`, `demand.py:62-64`, `distribution.py:182`, `land.py:158-162`, `waste.py:38`, `wtw.py:293-295`, `:542-544`), keyed `"<Class>.<term>"`, plus `"dropped"`: the six sub-`FLOAT_ACCURACY` drops, `QueueArc` pushes (`arcs.py:378-379`), waiting `QueueArc` requests removed on update (`arcs.py:483-485`), the `Land.run` percolation push-back (`land.py:213-215`), the `WWTW` storm-tank rejection (`wtw.py:403-404`), the `WWTW.make_discharge` push-back (`:347-348`), and the `QueueTank` storage-without-queue entry. Arc terms cancel and are not listed; storage change covers every state store and queue.
- **D12 Decisions.** `"decisions"` records node and arc branch sites with their margin (`lhs - rhs`). The VQIP algebra's internal `V > 0` guards are not recorded (thousands per day); the gradient tests choose points away from zero volumes.
- **D13 Event names.** The spec's eight: `"storage_unable_to_push"` (`storage.py:127`, `:229`), `"demand_deficit"` (`demand.py:95`), `"reservoir_spill"` (`storage.py:989`), `"wwtw_couldnt_push"` (`wtw.py:349`), `"service_reservoirs_not_filled"` (`wtw.py:599`), `"sludge_nowhere"` (`wtw.py:611`), `"river_cant_push"` (`storage.py:936`), `"ponded_water_cant_reenter"` (`sewer.py:239`); WSIMOD's further prints `"groundwater_couldnt_push_all"` (`storage.py:357`), `"miscalculated_tank_storage"` (`:364`, `sewer.py:231`), `"environmental_not_able_to_push"` (`storage.py:1125`), `"demand_not_able_to_push"` (`demand.py:115`), `"leakage_not_going_to_gw"` (`distribution.py:48`), `"more_treated_than_input"` (`wtw.py:214`), `"excess_treated_water"` (`:617`), `"more_backflow_than_vqip"` (`arcs.py:402`); and noodl's `"wtw_liquor_zero_denominator"` and `"inaccurate_evaporation"` (`land.py:1012`). The parity helper checks the two prints that carry the node and the date (`"demand deficit of … at <node> on <date>"`, `"Maxiter reached in <node> at <date>"`) day by day against noodl's `"demand_deficit"` and `"maxiter"`.
- **D14 Reservoir spill.** The spill branch (`storage.py:983-989`) is reached when the pull returns more than the tank's excess, which a `Distribution` with leakage and no `Groundwater` does (`distribution.py:40-50` returns `reply + unsuccessful_leakage`). Tested three ways: a `ScriptedArc` reply (Task 7), `reservoir_over_capacity` (a tank started above capacity: nothing spills downstream, Task 7), and the whole-model `reservoir_spill` (Task 10).
- **D15 Replay sampling.** A day's "branch outcome" for a node is the tuple of `(arc, method, tag)` of its captured arc calls that day; a day whose outcome is not among the sampled days' is added to the sample. The capture covers the days the sample can reach (Task 16 rule 0).
- **D16 Run bounds.** `wsimod_run(days=)` needs `1 <= days <= len(dates)`; `model.step` needs `t` a whole multiple of 86400 s inside the dates. Both refused by name.
- **D17 Layer-less `Model`.** `Model(net, [], closures=[closure])` already steps an integrating closure; W11 needs no `Model` change, only a test in `tests/test_model.py` that pins it.
- **D18 Renames.** `benchmarks/wsimod_oxford.py` (the allocation layer's) becomes `benchmarks/allocation_oxford.py`; `_wsimod_reference.py` keeps its name (shared by both harnesses) and its pending docstring edit is committed in Task 1 with the renamed test.
- **D19 Test-table row.** The AGENTS.md/`select_tests.py` wsimod row grows as its paths come into existence (Tasks 1, 2, 3), so every row is runnable at every commit. Its triggers include its own test paths, as the other rows' do.
- **D20 Dates.** Forcing keys are normalised to `datetime` (so `"2000-01"` and `"2000-01-01"` name one day, as `to_datetime.__eq__`, `model.py:164`); `WsimodResults.dates` keeps the labels as written.
- **D21 Surface data.** A surface's own `data_input_dict` may hold only deposition keys (`land.py:375-380`); with pollutants on they are refused (3c), with `pollutants=False` they are ignored and listed in `ignored_by_quantity_only`. Any other surface key is refused by name.
- **D22 Capture re-entry.** `capture_node_calls` refuses a re-entrant call into a target node; no case re-enters.
- **D23 Reference stdout.** `wsimod_reference` runs `run(verbose=True)`: `verbose=False` replaces `sys.stdout` for the whole run (`model.py:975-976`) and so hides every print the tests read; tqdm's bar goes to stderr.

## Review Focus

1. **An arc to a node whose type the sender's `of_type` excludes** (a `Node` → `Groundwater` arc): expect WSIMOD's silent behaviour (nothing pushed there; a single such out-arc returns the VQIP unpushed), not an error and not a delivery. Test in Task 5.
2. **A batch whose instances take different branches on the same day** (one instance over-subscribed, one not; one IHACRES branch each way): expect each instance to equal its own single run. Test in Task 14.
3. **Forcing dated `YYYY-MM`, `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`**, and dates with gaps: expect WSIMOD's matching (`to_datetime` equality) and one step per listed date, not a missing-forcing refusal. Test in Task 4.
4. **Initial storage above capacity** (oxford's urban surface 5e6 with capacity 0; a reservoir above capacity): expect WSIMOD's day-1 behaviour (all of it leaves the surface; the reservoir is never clamped), not a clamp at load. Tests in Tasks 7 and 12.
5. **Stepping outside the dates** (`model.step` without `t`, at a fractional day or past the last date; `wsimod_run(days=0)` or more days than dates): expect a refusal naming the argument, not an index error or a silent wrap. Test in Task 14.

---

### Task 0: Baseline

**Files:** none changed. **Implementer:** cheap.

- [ ] **Step 1:** `git status` on `wsimod-core`: only `tests/verification/_wsimod_reference.py` and `tests/verification/test_wsimod_parity.py` modified (the two pending docstring edits, left uncommitted until Task 1). Anything else: stop and report.
- [ ] **Step 2:** Run `python -m pytest -q -n 4 --no-cov --durations=20 <core subset> tests/verification/test_wsimod_*.py`; expected: pass. Record the wall time and the slowest 20 in the progress file under "Baseline".
- [ ] **Step 3:** Run `python benchmarks/wsimod_oxford.py` three times; record the median per batch size.

### Task 1: The allocation layer's WSIMOD evidence leaves the application (W6, part 1)

**Implementer:** cheap.

**Files:**
- Rename (`git mv`): `tests/verification/test_wsimod_parity.py` → `tests/verification/test_allocation_wsimod_replay.py`; `tests/verification/test_wsimod_capacity.py` → `tests/verification/test_allocation_wsimod_capacity.py`; `tests/data/wsimod/{quickstart,quickstart_tight,oxford}_{topology.json,events.csv}` → `tests/data/allocation/`; `benchmarks/wsimod_oxford.py` → `benchmarks/allocation_oxford.py`
- Modify: the two renamed tests (fixture directory; module docstrings carry the pending edit), `tests/verification/_wsimod_reference.py` (the pending edit; the new file names in its docstring), `scripts/regenerate_wsimod_fixtures.py` (`ALLOCATION_DIR = tests/data/allocation`), `benchmarks/allocation_oxford.py`, `benchmarks/coupling_street_building.py` (docstring), `tests/apps/test_initialisation.py`, `tests/test_refs.py`, `tests/test_units.py` (fixture path only; the old builder stays until Task 13), `tests/test_docs_usage.py` (`FIXTURES`: `"allocation/*.json"`), `docs/applications/allocation.md`, `docs/applications/index.md`, `docs/verification.md` (file names), `AGENTS.md`, `.github/scripts/select_tests.py`

- [ ] **Step 1:** Make the renames and path edits. `AGENTS.md`: the water row drops `tests/verification/test_wsimod_*.py`; a new row `| wsimod | tests/verification/test_allocation_wsimod_*.py |` (it grows in Tasks 2 and 3). `select_tests.py`: `"water"` drops every `wsimod` prefix and test path; new `"wsimod"`: prefixes `src/noodl/apps/wsimod/`, `src/noodl/layers/allocation.py`, `tests/data/wsimod/`, `tests/data/allocation/`, `tests/apps/wsimod/`, `tests/verification/test_wsimod_`, `tests/verification/test_allocation_wsimod_`, `tests/verification/_wsimod_reference.py`, `tests/verification/_wsimod_cases.py`, `tests/verification/_wsimod_parity.py`, `scripts/regenerate_wsimod_fixtures.py`; tests `tests/verification/test_allocation_wsimod_*.py`.
- [ ] **Step 2:** Run `echo src/noodl/layers/allocation.py | python .github/scripts/select_tests.py`; expected: the core paths then `tests/verification/test_allocation_wsimod_*.py`, no water path. `echo src/noodl/apps/water/inp.py | python .github/scripts/select_tests.py`; expected: no `wsimod` path.
- [ ] **Step 3:** Run the core subset plus `tests/verification/test_allocation_wsimod_*.py`, and `python benchmarks/allocation_oxford.py` once; expected: pass, the same test count as Task 0; `git grep -n "test_wsimod_parity\|test_wsimod_capacity\|data/wsimod/.*topology"` empty.
- [ ] **Step 4:** Commit: `test(allocation): the layer's WSIMOD replay tests and fixtures move out of apps.wsimod`.

### Task 2: Demo configurations and the WSIMOD reference harness

**Implementer:** standard.

**Files:**
- Modify: `scripts/regenerate_wsimod_fixtures.py` (`write_demo_configs(data_folder) -> None`, beside the allocation fixture writer), `tests/verification/_wsimod_reference.py`, `AGENTS.md` and `select_tests.py` (row gains `tests/verification/test_wsimod_*.py`)
- Create: `tests/data/wsimod/{quickstart,oxford}/{config.yml,config_neutral.yml,<node>-inputs.csv.gz}`, `tests/data/wsimod/README.md`, `tests/verification/test_wsimod_fixtures.py`

**Interfaces (produced, in `_wsimod_reference.py`):**

```python
def reset_wsimod_globals() -> None
    # constants.POLLUTANTS / ADDITIVE_POLLUTANTS / NON_ADDITIVE_POLLUTANTS = fresh copies of
    # 0.8.1's defaults (constants.py:21-50); FLOAT_ACCURACY = 1e-11
def wsimod_model(source: Path | Mapping, *, config_name: str = "config.yml"):
    # reset_wsimod_globals(), then Model.load(source, config_name) for a directory, or for a
    # mapping (D8) the same steps (model.py:232-274: globals, orchestration, add_nodes,
    # add_arcs, add_overrides, dates) on a deep copy
@dataclass
class WsimodReference:
    pollutants: tuple[str, ...]
    dates: list[str]
    arc_names: list[str];  arc_flow: Tensor;  arc_quality: Tensor        # (T, n_arcs), (T, n_arcs, K)
    tank_names: list[tuple[str, str]];  tank_storage: Tensor;  tank_quality: Tensor   # (node, prop)
    surface_names: list[tuple[str, str]];  surface: dict[str, Tensor];  surface_quality: Tensor
        # fields storage evaporation precipitation percolation subsurface_r surface_r
        # tank_recharge, (T, n_surfaces), NaN where WSIMOD records nothing (impervious)
    printed: list[str]                                                 # stdout lines of the run
def wsimod_reference(source, *, config_name: str = "config.yml", days: int | None = None) -> WsimodReference
    # wsimod_model, then run(dates=model.dates[:days], verbose=True) under redirect_stdout (D23)
```

- The script: build quickstart (inline, as now) and oxford (`create_oxford_model`); `Model.save(dir, "config.yml", compress=True)`; rewrite each `PerviousSurface`'s `depth` in the YAML to the physical depth (`surface.depth / surface.total_porosity`, the `save` defect, `model.py:419-420`; W12); write `config_neutral.yml` = `config.yml` plus an `overrides` section (every `WWTW`/`FWTW`: `process_parameters[p] = {constant: c_v, exponent: 1}`, `liquor_multiplier[p] = liquor_multiplier["volume"]` for every additive `p`; every surface of every `Land`: `pollutant_load` zero for every key it has, `soil_temp_w_prev 1, soil_temp_w_air 0, soil_temp_w_deep 0`); reload both with `wsimod_model` and run: the reloaded `config.yml` run must equal the original model's run (flows, tanks, surfaces), else exit non-zero naming the first difference. README: the forcing is WSIMOD's BSD-3-Clause `timeseries_data.csv` (W14), the save fix, how to regenerate.

- [ ] **Step 1: Write the failing tests** in `test_wsimod_fixtures.py` (module-level `pytest.importorskip("wsimod")`):
  - `test_the_demo_configurations_load_in_wsimod[quickstart|oxford]`: `wsimod_model(dir)`: quickstart 5 nodes, 6 arcs; oxford 18 nodes, 21 arcs; both 1456 dates.
  - `test_the_saved_pervious_depth_is_the_physical_depth`: oxford's `rural` surface has `depth: 0.5` in `config.yml`, and the loaded surface's `depth == 0.5 * 0.4`.
  - `test_the_neutral_configuration_differs_only_by_its_overrides[quickstart|oxford]`: the two YAML mappings are equal with `"overrides"` removed.
  - `test_reset_wsimod_globals_restores_the_defaults`: after `wsimod_model(oxford)`, `constants.FLOAT_ACCURACY == 1e-8`; after `reset_wsimod_globals()`, `1e-11` and the 12 default pollutants in order.
  - `test_the_reference_records_wsimods_run`: quickstart, `days=10`: `arc_flow.shape == (10, 6)`; the impervious surface's `tank_recharge` all NaN; no line of `printed` contains `"mass balance error"`.
- [ ] **Step 2:** Run `pytest tests/verification/test_wsimod_fixtures.py -v`; expected: FAIL (no configs; `wsimod_reference` missing).
- [ ] **Step 3:** Implement the harness; run the script once to write the fixtures (check the gzipped forcing is about 1 MB). Extend the AGENTS row and `select_tests.py`.
- [ ] **Step 4:** Run the same; expected: PASS. Run the allocation tests; expected: PASS.
- [ ] **Step 5:** Commit: `test(wsimod): WSIMOD's demo configurations and the reference harness`.

### Task 3: VQIP algebra and the decision primitives

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/constants.py`, `src/noodl/apps/wsimod/vqip.py`, `tests/apps/wsimod/__init__.py`, `tests/apps/wsimod/test_vqip.py`
- Modify: `AGENTS.md`, `select_tests.py` (row gains `tests/apps/wsimod`; it is now the spec's row)

**Interfaces (produced):**

```python
# constants.py (constants.py:18-57)
MAXITER = 5;  UNBOUNDED_CAPACITY = 1e15;  DT_DAYS = 1;  FLOAT_ACCURACY = 1e-11
POLLUTANTS: tuple[str, ...]; ADDITIVE_POLLUTANTS: tuple[str, ...]; NON_ADDITIVE_POLLUTANTS: tuple[str, ...]  # 0.8.1's defaults
# vqip.py: a VQIP is a tensor (..., 1 + K): column 0 volume, then the pollutants in order
@dataclass(frozen=True)
class Pollutants:
    names: tuple[str, ...]; additive: tuple[str, ...]; non_additive: tuple[str, ...]
    K: int                                  # len(names)
    additive_mask: Tensor; non_additive_mask: Tensor        # (1 + K,) bool, column 0 False
    def column(self, name: str) -> int                      # 1-based
    def to_vqip(self, values: Mapping[str, float], shape=()) -> Tensor   # "volume" and pollutants; absent 0
    def to_dict(self, vqip: Tensor) -> dict[str, float]     # one instance
    # __post_init__ refuses, by name, a pollutant in neither or both lists, or a list naming
    # something else (spec, Representation)
QUANTITY_ONLY = Pollutants((), (), ())
def empty(p: Pollutants, shape) -> Tensor
def sum_vqip(p, t1, t2) -> Tensor        # core.py:83-116, (t2·V2 + t1·V1)/V in that order when V > 0
def extract_vqip(p, t1, t2) -> Tensor    # :150-178
def v_change_vqip(p, t, v) -> Tensor     # :251-284; V > 0: ratio = v/V, volume = V·ratio
def v_distill_vqip(t, v) -> Tensor       # :209-223
def ds_vqip(p, t1, t2) -> Tensor         # :305-328; non-additive columns 0
def tmin(a, b) -> Tensor;  def tmax(a, b) -> Tensor        # Python's tie rule, detached comparison
def safe_den(den) -> Tensor              # where(den != 0, den, 1)
def safe_sqrt(x) -> Tensor               # value sqrt(x); gradient 0 where x == 0
def write(active, new, old) -> Tensor    # where(active[..., None] for a VQIP, new, old)
```

- [ ] **Step 1: Write the failing tests** in `test_vqip.py`:
  - `test_vqip_examples_from_wsimods_docstrings` (pollutants `phosphate` additive, `temperature` non-additive): `sum_vqip` of (100, 0.25, 10) and (10, 0.25, 15) gives volume 110, phosphate 0.5, temperature `(15*10 + 10*100)/110`; `extract_vqip` gives 90, 0.0, 10; `v_change_vqip((100, 0.25, ·), 10)` gives 10, 0.025; `ds_vqip((100, 0.25, 10), (90, 0.2, 9))` gives 10, 0.25-0.2, 0.
  - `test_v_change_scales_by_the_ratio_not_to_the_target`: `V = 49, v = 1` gives volume `49 * (1/49)` (`0.9999999999999999`, not 1.0); at `V = 0` the volume is `v` and the pollutants are unchanged.
  - `test_vqip_operations_equal_wsiobj_bitwise_on_random_inputs` (`importorskip`): globals set to `phosphate, nitrate` additive and `temperature, do` non-additive; 1000 random pairs, 10 % with a zero volume and 5 % whose sum has volume 0; each of the five operations, batched over the 1000, `torch.equal` to `WSIObj`'s per pair (+ − × / are correctly rounded in both).
  - `test_quantity_only_vqips_have_one_column`: `QUANTITY_ONLY` operations on `(..., 1)`.
  - `test_inconsistent_pollutant_lists_are_refused_by_name`: `phosphate` in both lists, in neither, and `nitrate` listed only as additive: `ValueError` naming the pollutant.
  - `test_ties_follow_pythons_min_and_max`: `a = b = 2.0` requiring grad: `d tmin/da == 1`, `d tmin/db == 0`; the same for `tmax`; `tmin(3, 2) == 2` with gradient to `b`.
  - `test_guarded_division_and_root_keep_gradients_finite`: `where(d > 0, n / safe_den(d), 0)` at `d = 0` has value 0 and finite gradients; `safe_sqrt` at 0 has gradient 0, at 4 value 2 and gradient 0.25.
  - `test_vqip_operations_pass_gradcheck` (`torch.autograd.gradcheck`, positive volumes).
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_vqip.py -v`; expected: FAIL (`vqip` missing).
- [ ] **Step 3:** Implement; each function cites its `core.py` lines.
- [ ] **Step 4:** Run it; expected: PASS. Extend the row.
- [ ] **Step 5:** Commit: `feat(wsimod): VQIP algebra and decision primitives as float64 tensors`.

### Task 4: Reading WSIMOD's model description

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/classes.py`, `src/noodl/apps/wsimod/config.py`, `tests/apps/wsimod/test_config.py`, `tests/data/wsimod/small/{config.yml,inflow-inputs.csv}` (D9), `docs/formats/wsimod.md`
- Modify: `mkdocs.yml` (Formats nav)

**Interfaces (produced):**

```python
# classes.py: transcribed constructor chains (the whole MRO's keywords and 0.8.1 defaults)
@dataclass(frozen=True)
class ClassSchema:
    name: str
    parameters: Mapping[str, object]     # constructor keyword -> WSIMOD default
    override_keys: frozenset[str]        # what its apply_overrides consumes (whole chain)
    renamed_type: str                    # wsimod_type after renames (storage.py:294, :1043, demand.py:214)
NODE_SCHEMAS: dict[str, ClassSchema]     # the 19 in-scope node classes
SURFACE_SCHEMAS: dict[str, ClassSchema]  # Surface, ImperviousSurface, PerviousSurface
ARC_SCHEMAS: dict[str, ClassSchema]      # Arc, QueueArc, AltQueueArc, PullArc, PushArc, SewerArc, WeirArc
OUT_OF_SCOPE: dict[str, str]             # class -> sub-project: DecayArc, DecayArcAlt -> "3b";
    # GrowingSurface, IrrigationSurface, GardenSurface, VariableAreaSurface -> "3c";
    # EnfieldFoulSewer -> "refused (W4: undefined storm_exchange, sewer.py:288-291)"
# config.py
@dataclass
class SurfaceSpec:
    name: str; cls: str; parameters: dict[str, object]; data: dict[tuple[str, datetime], float]
@dataclass
class NodeSpec:
    name: str; cls: str; slot_key: str          # cls: node_type_override or type_; slot_key: "Foul" if "foul" in name else type_ (model.py:650-673)
    parameters: dict[str, object]                # schema defaults, then the config, then overrides (D5)
    data: dict[tuple[str, datetime], float]
    surfaces: list[SurfaceSpec]
@dataclass
class ArcSpec:
    name: str; cls: str; in_port: str; out_port: str
    capacity: float = 1e15; preference: float = 1.0; number_of_timesteps: int = 0
@dataclass
class WsimodNetwork:
    pollutants: Pollutants; float_accuracy: float
    orchestration: list[tuple[str, str]]         # (slot key, method); default model.py:191-205
    dates: list[datetime]; date_labels: list[str]
    nodes: list[NodeSpec]; arcs: list[ArcSpec]   # configuration order
def parse_date(text: str) -> datetime            # to_datetime's formats in its order (model.py:139-162)
def read_wsimod_config(source: str | PathLike | Mapping, *, config_name: str = "config.yml",
                       overrides: Mapping | None = None) -> WsimodNetwork
```

- Keys as the spec's Loading tables: top-level `nodes` (required), `arcs`, `overrides`, `orchestration`, the three pollutant lists, `float_accuracy`, `dates` read; `river_discharge_order` ignored; non-empty `extensions` and `unified_data_file` refused by name (W18); anything else refused. Node keys: `name`, `type_`, `node_type_override`, `geometry` dropped, `filename` (CSV or CSV.GZ, `read_csv`, `model.py:1252-1269`) or `data_input_dict`, `surfaces` (dict or list); every other key must be a schema parameter of the class's chain (else refused: WSIMOD `TypeError`). Arc keys: the `ArcSpec` fields; `decays` refused (3b); anything else refused. Overrides: node lookup by `(type_ = slot key, name)` (`model.py:819-825`), unknown node or arc refused, a key outside `override_keys` refused, `filename` or a surface `data_input_dict` path refused (3a). `overrides=` replaces top-level keys first (`model.py:229-230`). Out-of-scope classes refused by name with the sub-project.

- [ ] **Step 1: Write the failing tests** in `test_config.py`:
  - `test_the_class_schemas_match_wsimods_constructors` (`importorskip`): for every schema, the keywords and defaults of `inspect.signature` over the WSIMOD class's MRO equal `parameters` (mutable defaults by value).
  - `test_the_demos_read_as_wsimod_loads_them[quickstart|oxford]` (`importorskip`): node names, `cls`, `slot_key` and arc ends equal `wsimod_model(dir)`'s `nodes_type`/`arcs` in order; oxford `float_accuracy == 1e-8`; every numeric parameter equal to the loaded object's attribute; `data` equal to the object's `data_input_dict` (keys compared as datetimes).
  - `test_the_small_config_reads`: 3 nodes, 2 arcs, 10 dates, `pollutants.K == 0`, `inflow`'s `("flow", 2000-01-03)` read from the CSV.
  - `test_foul_sewers_and_type_overrides_take_their_slot_keys`: `{"name": "foul_1", "type_": "Sewer"}` has slot `"Foul"`; `{"type_": "Demand", "node_type_override": "ResidentialDemand"}` has `cls` `ResidentialDemand`, slot `"Demand"`.
  - `test_overrides_merge_into_the_parameters` (`importorskip`; a mapping with a `PerviousSurface`, `ImperviousSurface`, `River`, `WWTW`, `FWTW`, `Sewer` and `Land`, overriding `depth`, `total_porosity`, `field_capacity`, `area`, `pore_depth`, `length`, `percent_solids`, `liquor_multiplier` (one key: the others kept), `capacity` and the residence times): every overridden `NodeSpec`/`SurfaceSpec` parameter within `1e-15` relative of the WSIMOD object's attribute after `add_overrides` (the physical depth: the object's `depth / total_porosity`, WSIMOD's own round trip, `land.py:858`). The derived values are compared in Tasks 9, 11 and 12, where they are computed (D5).
  - `test_an_override_of_pollutant_load_reaches_only_its_node` (defect row "shared default dicts"): two `Demand` nodes built without `pollutant_load`; an override `{"phosphate": 2.0}` on the first: the second's `pollutant_load` is `{}`.
  - Review Focus 3, `test_monthly_daily_and_timestamp_dates_name_the_same_day`: forcing keyed `"2000-01"`, dates `["2000-01-01", "2000-01-02 00:00:00", "2000-01-05"]` (a gap): `dates` has 3 entries, `data[("flow", datetime(2000, 1, 1))]` is found; `date_labels` as written.
  - `test_refusals_name_the_key` (parametrised, `match=`): `extensions: [x.py]`, `unified_data_file`, `foo: 1` (top level); `type_: EnfieldFoulSewer` (`"EnfieldFoulSewer"`), a `GrowingSurface` (`"3c"`), an arc `type_: DecayArc` (`"3b"`), an arc with `decays` (`"decays"`), an arc with `diameter: 1` (`"diameter"`), a node keyword `capacityy` (`"capacityy"`); overrides naming an unknown node, an unknown arc, an unconsumed key, a `filename`; a date `"01/02/2000"`.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_config.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement; write `docs/formats/wsimod.md` (the keys read, ignored and refused, the date formats, the override lookup by slot key, the cwd note for `filename` overrides) and its nav entry.
- [ ] **Step 4:** Run it; expected: PASS. `mkdocs build --strict`.
- [ ] **Step 5:** Commit: `feat(wsimod): read WSIMOD's config.yml into a WsimodNetwork`.

### Task 5: The protocol: day context, arcs, `Node`, `Waste`, `Catchment`, and the replay harness

The core every node task builds on; the masked loops are where a mistake would hide in every later parity result. `Catchment` is here because it is small and every later case needs its water.

**Implementer:** most capable.

**Files:**
- Create: `src/noodl/apps/wsimod/engine.py`, `src/noodl/apps/wsimod/arcs.py`, `src/noodl/apps/wsimod/node.py`, `src/noodl/apps/wsimod/catchment.py`, `tests/apps/wsimod/_replay.py`, `tests/apps/wsimod/test_protocol.py`, `tests/verification/_wsimod_cases.py`, `tests/verification/test_wsimod_replay.py`
- Modify: `tests/verification/_wsimod_reference.py` (`capture_node_calls`)

**Interfaces:**
- Consumes: the VQIP functions and `Pollutants` (Task 3), `WsimodNetwork`, `NodeSpec`, `ArcSpec` (Task 4).
- Produces:

```python
Tag = str | tuple[str, str]
# engine.py
class Ctx:   # one day; every node and arc holds it as self.ctx (Engine.begin_day sets it)
    def __init__(self, p: Pollutants, float_accuracy: float, shape: torch.Size, *, day: int,
                 node_names: Sequence[str], record_decisions: bool = False) -> None
    p; fa: float; shape; day: int
    def empty(self) -> Tensor;  def zeros(self) -> Tensor          # shape + (1+K,), shape
    def decide(self, site: str, lhs: Tensor, op: str, rhs) -> Tensor
        # op in "<" "<=" ">" ">=" "==" "!=": detached bool of `shape`; appends (site, outcome, lhs - rhs) when recording (D12)
    def event(self, condition: str, node: str, mask: Tensor) -> None   # D13 names only
    def drop(self, vqip: Tensor, mask: Tensor) -> None                 # sub-FLOAT_ACCURACY water (D11 "dropped")
    def term(self, name: str, vqip: Tensor, mask: Tensor) -> None      # "<Class>.<term>" (D11)
    def loop_site(self, node: str, method: str) -> tuple[str, str, int]  # (node, method, call index this day)
    def loop_exit(self, site: tuple[str, str, int], hit: Tensor) -> None # ended at MAXITER
    decisions: list[tuple[str, Tensor, Tensor]]; events: dict[str, dict[str, Tensor]]
    maxiter: dict[tuple[str, str, int], Tensor]; dropped: Tensor; terms: dict[str, Tensor]
class Engine:
    def __init__(self, network: WsimodNetwork, *, pollutants: Pollutants) -> None
    nodes: dict[str, Node]; arcs: dict[str, Arc]          # configuration order
    def begin_day(self, ctx: Ctx) -> None
def make_node(spec: NodeSpec, network: WsimodNetwork, pollutants: Pollutants) -> Node   # NODE_CLASSES[spec.cls]
# arcs.py (arcs.py:18-290, :808-905)
class Arc:
    def __init__(self, spec: ArcSpec, in_port: Node, out_port: Node, pollutants: Pollutants) -> None
    capacity: Tensor; preference: Tensor                   # (...) ; bind() replaces them per day
    flow_in: Tensor; flow_out: Tensor; vqip_in: Tensor; vqip_out: Tensor
    def send_push_request(self, vqip, *, active, tag: Tag = "default", force: bool = False) -> Tensor  # not pushed
    def send_pull_request(self, vqip, *, active, tag: Tag = "default") -> Tensor                       # obtained
    def send_push_check(self, vqip=None, *, active, tag: Tag = "default") -> Tensor
    def send_pull_check(self, vqip=None, *, active, tag: Tag = "default") -> Tensor
    def get_excess(self, direction: str, vqip=None, *, active, tag: Tag = "default") -> Tensor
    def bind(self, capacity: Tensor, preference: Tensor) -> None
    def stores(self) -> dict[str, Tensor]; def load_stores(self, stores: Mapping[str, Tensor]) -> None
        # carried state only: {} for a plain arc; "register" (Task 6), "rows" and "since" (Task 8)
    def end_timestep(self) -> None
class PullArc(Arc); class PushArc(Arc); class SewerArc(Arc); class WeirArc(Arc)
ARC_CLASSES: dict[str, type[Arc]]
# node.py (nodes.py:15-646; waste.py)
class Node:
    wsimod_type: ClassVar[str] = "Node"                  # after renames: of_type and the river order use it
    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None
    name; in_arcs: dict[str, Arc]; out_arcs: dict[str, Arc]
    in_arcs_type: dict[str, dict[str, Arc]]; out_arcs_type: dict[str, dict[str, Arc]]
    pull_set_handler / push_set_handler / pull_check_handler / push_check_handler: dict[Tag, Callable]
        # set: handler(vqip, *, active) -> Tensor; check: handler(vqip=None, *, active) -> Tensor
    def connect(self, arc: Arc, end: str) -> None           # end "in" | "out", configuration order
    def pull_set(self, vqip, tag, *, active); push_set(...); pull_check(vqip, tag, *, active); push_check(...)
        # query_handler (nodes.py:319-343): a missing tag raises KeyError naming node and tag
    def get_connected(self, direction="pull", of_type=None, tag="default", *, active) -> Connected
    def pull_distributed(self, vqip, of_type=None, tag="default", *, active) -> Tensor
    def push_distributed(self, vqip, of_type=None, tag="default", *, active) -> Tensor
    def check_basic(self, direction, vqip=None, of_type=None, tag="default", *, active) -> Tensor
    def bind(self, parameters: Mapping[str, Tensor]) -> None              # this day's parameter slices
    def stores(self) -> dict[str, Tensor]; def load_stores(self, stores: Mapping[str, Tensor]) -> None
        # keys are WSIMOD attribute paths ("tank.storage", "tank.active_storage", "liquor", "surfaces.<name>.storage", ...)
    def end_timestep(self) -> None
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]
        # raises ValueError by name; returns the parameters ignored when quantity-only
    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool   # spec, Approach: recursion edges
@dataclass
class Connected:
    avail: Tensor; priority: Tensor; allocation: dict[str, Tensor]; capacity: dict[str, Tensor]
class Waste(Node)                                         # waste.py:10-49
NODE_CLASSES: dict[str, type[Node]]                       # each module registers its classes
# catchment.py (catchment.py:12-138)
class Catchment(Node)    # get_flow, get_avail over every out-arc's vqip_in (:87-101), route (:76-85),
                         # pull_check/pull_set_abstraction (:103-134), "Catchment.flow"/"Catchment.unrouted" terms
# tests/verification/_wsimod_reference.py
@dataclass
class ArcCall: arc: str; method: str; vqip: dict | None; tag: Tag; force: bool; reply: dict
    flow_in: float; vqip_in: dict             # the arc's accumulators at call entry
@dataclass
class NodeCall: day: int; method: str; vqip: dict | None; tag: Tag | None; reply: dict | None
    pre: dict[str, object]; post: dict[str, object]; arc_calls: list[ArcCall]
    # pre/post include "arcs.<name>.vqip_in" and "arcs.<name>.flow_in" of the target's out-arcs
@contextmanager
def capture_node_calls(model, nodes: Sequence[str]) -> Iterator[dict[str, list[NodeCall]]]
    # wraps each target's orchestration methods and its four handler entry points, and the
    # send_* of the arcs it calls (out-arcs push, in-arcs pull); deep-copies every argument at
    # entry (spec, Request mutation); stores by the same attribute paths as Node.stores; D22
# tests/apps/wsimod/_replay.py
class ScriptedArc:   # Arc's call interface; active calls consume the script in order and
                     # assert method, tag, force equal and vqip within 1e-12 * scale (pull: volume, D4)
    def __init__(self, spec: ArcSpec, calls: list[ArcCall], pollutants: Pollutants, *, scale: float,
                 in_type: str, out_type: str) -> None
    flow_in: Tensor; vqip_in: Tensor                      # settable; replay loads them from NodeCall.pre
    def assert_exhausted(self) -> None
class StoreNode(Node):  # test double: a VQIP store of a given capacity and settable wsimod_type
    # push_set accepts up to the spare capacity, pull_set gives up to its volume; both checks
    # with the None case (unclipped) and the request case (clipped)
def replay(network: WsimodNetwork, name: str, calls: list[NodeCall], *, scale: float) -> None
    # make_node alone, ScriptedArcs on every arc, load_stores(pre), call, compare reply and post
# tests/verification/_wsimod_cases.py
@dataclass(frozen=True)
class Case:
    """A purpose-built case (D8). A case with a River declares pollutant lists without nitrate."""
    build: Callable[[], dict]                 # the mapping
    days: int                                 # 30-120
    replay: tuple[str, ...] = ()              # nodes the replay test captures
    modes: tuple[str, ...] = ("quantity", "pollutants")
    wsimod: bool = True                       # False: noodl-only (a WSIMOD crash), Task 17
CASES: dict[str, Case]
```

- `push_distributed` and `pull_distributed` transcribe `nodes.py:345-478` with every branch masked: the single-arc shortcut is structural (`len(self.out_arcs) == 1`; of_type tested as `nodes.py:424-438`); the first-round capacity switch (`:446-450`) a per-instance `where` on `allocation`/`priority`; five rounds under `active & (not_pushed > fa) & (avail > fa)`; `loop_exit` when the round count reaches 5. `get_connected` zeroes `avail < fa`, sums in arc order. Arcs per `arcs.py:112-265`: `force` skips the clip but adds to `flow_in` (a negative pipe excess is transcribed, not clamped).

- [ ] **Step 1: Write the failing tests.** Unit (`test_protocol.py`, noodl only, receivers and sources `StoreNode`s or `Catchment`s):
  - `test_an_oversubscribed_push_shares_by_capacity_in_the_first_round`: a `Node` with out-arcs to three `Waste`s, capacities 2, 3, 1, preferences 1, 2, 3; `push_set` of volume 10: arc flows 2, 3, 1; reply volume 4.
  - `test_an_undersubscribed_push_shares_by_preference_times_avail`: the same, volume 3: arc flows `3*2/11`, `3*6/11`, `3*3/11` (Python's expressions), `|reply| <= 1e-15`.
  - `test_a_check_without_a_request_is_not_clipped`: a `Node` pulling from two `StoreNode`s holding 3 and 4: `pull_check(None)` volume 7, `pull_check(volume 5)` volume 5.
  - `test_a_forced_push_makes_the_pipe_excess_negative`: arc capacity 2 into a `StoreNode` with spare capacity 10, a forced push of 5: `get_excess("push")` volume −3; the upstream `Node`'s `check_basic("push")` volume −3; `get_connected` avail 0.
  - `test_a_missing_tag_is_refused_by_name`: `push_set(v, ("Demand", "Garden"))` on a `Node`: `KeyError` matching the node and the tag.
  - `test_pull_and_push_arcs_deny_one_direction`: a `PullArc` push returns the VQIP, a `PushArc` pull returns empty (`arcs.py:808-893`).
  - `test_an_inactive_call_changes_nothing`: batch of 2, `active = [True, False]`: instance 1's arcs and stores equal their values before the call.
  - `test_catchment_avail_is_flow_less_every_out_arcs_vqip_in`: flow 10, a `StoreNode` abstracts 3 through a pull: `route` pushes 7.
  - Review Focus 1, `test_a_node_never_pushes_to_a_type_outside_its_of_type`: `Node` → `StoreNode(wsimod_type="Groundwater")` and `Node` → `Waste`: a push of 5 reaches only the `Waste`; with the `Groundwater` arc alone, `push_set` returns all 5 and the arc's flow is 0.
  - Replay (`test_wsimod_replay.py`, `importorskip`): `test_replay_matches_wsimod[<case>-<node>]` parametrised over every `(case, node)` in `CASES[*].replay`; Task 5 adds `three_out_arcs` (two `Catchment`s → a `Node` → three `Waste`s, the over-subscription switch on some days) and `node_chain` (a `Catchment` → three pass-through `Node`s → a `Waste`, check recursion), replaying the `Catchment`s and `Node`s.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_protocol.py tests/verification/test_wsimod_replay.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS (a difference beyond the bound is debugged to its cause with `superpowers:systematic-debugging` before anything is admitted). Then the wsimod subset.
- [ ] **Step 5:** Commit: `feat(wsimod): WSIMOD's pull/push protocol, arcs, Node, Waste and Catchment, with node replay`.

### Task 6: Tanks, `QueueTank` and the `AltQueueArc` shift register

**Implementer:** most capable (the queue flush order decides the demos' sewer volumes).

**Files:**
- Create: `src/noodl/apps/wsimod/tanks.py`, `tests/apps/wsimod/test_tanks.py`
- Modify: `src/noodl/apps/wsimod/arcs.py` (`AltQueueArc`)

**Interfaces:**
- Consumes: `Ctx`, `Arc` (`stores`/`load_stores`), the VQIP functions (Tasks 3, 5).
- Produces:

```python
class Tank:                       # tanks.py:10-356
    def __init__(self, pollutants, *, capacity: Tensor, area: Tensor, datum: Tensor, initial_storage: Tensor) -> None
    storage: Tensor; ctx: Ctx
    def get_excess(self, vqip=None, *, active) -> Tensor;  def get_avail(self, vqip=None, *, active) -> Tensor
    def push_storage(self, vqip, *, active, force: bool = False) -> Tensor   # rejected
    def pull_storage(self, vqip, *, active) -> Tensor                        # obtained; empty when V == 0 exactly
    def pull_ponded(self, *, active) -> Tensor
    def evaporate(self, evap: Tensor, *, active) -> Tensor                   # evaporated volume, v_distill
class ResidenceTank(Tank):        # :359-400
    residence_time: Tensor; def pull_outflow(self, *, active) -> Tensor
class DecayTank(Tank):            # :403-457, empty decays only: end_timestep reads ("temperature", day) and is an identity
class AltQueueArc(Arc):           # arcs.py:560-660
    register: Tensor              # (..., L+1, 1+K), L from number_of_timesteps and the times the structure can enter;
                                  # stores() key "register"
    def send_push_request(self, vqip, *, active, tag="default", force=False, time: int = 0) -> Tensor
    def update_queue(self, *, active) -> Tensor                              # pushes register[0] without a tag; backflow
    def end_timestep(self) -> None # new[i] = old[i+1] (1 <= i < L), new[0] = sum_vqip(old[0], old[1])
class QueueTank(Tank):            # :460-642
    number_of_timesteps: int; active_storage: Tensor; internal_arc: AltQueueArc
    def push_storage(self, vqip, *, active, time: int = 0, force: bool = False) -> Tensor
    def push_check(self, vqip=None, *, active) -> Tensor;  def push_set(self, vqip, *, active) -> Tensor
    def pull_storage(self, vqip, *, active) -> Tensor;  def pull_storage_exact(self, vqip, *, active) -> Tensor
    def end_timestep(self) -> None # internal shift, then update_queue into active_storage (:631-635)
```

- [ ] **Step 1: Write the failing tests** in `test_tanks.py`:
  - `test_tanks_equal_wsimod_tanks_on_random_sequences` (`importorskip`): `Tank`, `ResidenceTank` (residence time 3), 200 random calls (push with and without force, pull, pull_ponded, evaporate, get_excess/get_avail with and without a request, storage exactly 0 included), one additive and one non-additive pollutant: storage and every reply within `1e-12 * scale`.
  - `test_queue_tank_equals_wsimod_on_random_pushes` (`importorskip`): `number_of_timesteps` 0 and 2, push times 0..3, pulls and exact pulls, 60 days with `end_timestep`: `storage`, `active_storage` and the register (dense) within the bound.
  - `test_water_entered_with_total_time_zero_is_active_at_once`: `number_of_timesteps = 0`, push 5 at time 0: `active_storage` volume 5 after the call.
  - `test_end_timestep_shifts_and_flushes_the_internal_arc`: push 3 at time 1 on day 0; after `end_timestep`, `register[0]` is empty and `active_storage` gained 3 (`tanks.py:631-635`; `QueueTank.push_set` always accepts, `:610-629`).
  - `test_a_tiny_push_is_kept_in_storage_but_never_queued`: push `1e-12` with `fa = 1e-11`: `storage` +1e-12, register and `active_storage` unchanged, `ctx.dropped[..., 0] == 1e-12`.
  - `test_pull_ponded_on_a_queue_tank_is_limited_to_active_storage`: capacity 5, storage 8 of which 2 active: `pull_ponded` returns 2.
  - `test_a_user_alt_queue_arc_shifts_daily_but_delivers_only_on_a_push`: `number_of_timesteps = 2` into a `Waste`: push 4 on day 0; day 2 `vqip_out` volume 0; a push of 1 on day 3 delivers 4 (`vqip_out` 4) and enters 1.
  - `test_a_non_integer_travel_time_is_refused_by_name`: an `AltQueueArc` spec with `number_of_timesteps: 1.5` and a time-area key `-1`: `ValueError` naming the arc and the key (raised by `validate`, called by Task 13's `check_network`).
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_tanks.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): tanks, QueueTank and the AltQueueArc shift register`.

### Task 7: `Storage`, `Groundwater`, `QueueGroundwater`, `Reservoir`, `RiverReservoir`

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/storage.py`, `tests/apps/wsimod/test_storage.py`
- Modify: `tests/verification/_wsimod_cases.py`, `tests/apps/wsimod/test_protocol.py` (the random-arc test)

**Interfaces:**
- Consumes: `Node`, `Catchment`, `Tank`, `QueueTank`, `ScriptedArc` (Tasks 5, 6).
- Produces: `Storage`, `Groundwater`, `QueueGroundwater` (`wsimod_type` `"Groundwater"`), `Reservoir`, `RiverReservoir` (`wsimod_type` `"Reservoir"`) in `NODE_CLASSES`, each transcribing `storage.py` per the spec's Nodes section: handlers `:70-73`; `distribute` (`:120-127`, `:222-229`, `:350-364`); `infiltrate` (`:231-245`, `safe_sqrt`; `QueueGroundwater` a no-op `:366-368`); `push_set_timearea` (`:330-348`); `pull_check_active` / `pull_set_active` pro rata over every queue slot and the active store (`:370-433`); `make_abstractions` (`:983-989`); the `RiverReservoir` push, check (`:1060-1110`) and `satisfy_environmental` (`:1112-1128`), `total_environmental_satisfied` reset in `end_timestep` (`:1130-1133`). `validate` refuses `decays` other than `None` on the family (`"decays"`, `"3b"`).

- [ ] **Step 1: Write the failing tests** in `test_storage.py` (and `test_protocol.py`):
  - `test_arcs_equal_wsimod_arcs_on_random_requests` (`importorskip`, in `test_protocol.py`): two `Storage` nodes joined by an `Arc` of random capacity, 50 days of 5 random push/pull/check calls (some forced, some below `FLOAT_ACCURACY`): every reply, `flow_in`, `vqip_out` within `1e-12 * scale` of WSIMOD's.
  - `test_infiltration_is_the_root_of_the_excess_share`: capacity 100, `infiltration_threshold` 0.5, `infiltration_pct` 0.04, storage 75, out-arc a `ScriptedArc` with `out_type="Sewer"`: 1.0 pushed (`(25*0.04)**0.5`); at storage 50, 0 pushed and `d pushed / d storage == 0`.
  - `test_a_decays_dict_on_a_storage_node_is_refused`: `decays: {}` on `Storage`, `Groundwater`, `Reservoir`: `ValueError` matching `"decays"` and `"3b"`; `decays: None` accepted.
  - `test_a_spill_is_forced_back_into_the_reservoir` (D14): a `Reservoir` with capacity 10 holding 8 whose single in-arc is a `ScriptedArc` returning 5 to the request of 2: storage 13, event `"reservoir_spill"` true, nothing pushed downstream.
  - Review Focus 4, `test_a_reservoir_above_capacity_is_never_clamped`: initial storage 12, capacity 10: after one `make_abstractions` the storage is 12 and no pull is requested.
  - `test_pull_set_active_takes_pro_rata_from_every_slot`: a `QueueGroundwater` with 2 active and 6 in a slot of time 2: a pull of 4 takes 1 from the active store and 3 from the slot (`storage.py:386-433`).
  - Replay (`CASES`, each replaying its storage nodes; sources are `Catchment`s or `Storage`s): `forced_negative_excess`, `maxiter_pull` (a `Reservoir` pulling through capacity-bound arcs from `Storage`s, so WSIMOD ends a `pull_distributed` at `MAXITER`; the case asserts `"Maxiter reached"` in `wsimod_reference(...).printed`), `pull_and_push_arcs` (a `PullArc` and a `PushArc` between `Storage`s, custom orchestration `Storage.distribute`), `user_alt_queue_arc` (a `Storage` distributing through an `AltQueueArc` into a `Storage`), `storage_distribute` (custom orchestration `Storage.distribute`), `queue_groundwater` (time-area `{0: 0.5, 2: 0.5}`, pulls through `pull_set_active` by a `Reservoir`), `reservoir_over_capacity` (D14), `river_reservoir_environmental` (custom orchestration with `RiverReservoir.satisfy_environmental`, no `River`), `catchment_abstraction` (a `Reservoir` abstracting from a `Catchment`).
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_storage.py tests/apps/wsimod/test_protocol.py tests/verification/test_wsimod_replay.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): Storage, Groundwater, QueueGroundwater, Reservoir and RiverReservoir`.

### Task 8: `QueueArc` request rows (W21, D1)

**Implementer:** most capable (D1's anchored rows and delivery order).

**Files:**
- Modify: `src/noodl/apps/wsimod/arcs.py` (`QueueArc`), `tests/verification/_wsimod_cases.py`
- Test: `tests/apps/wsimod/test_queue_arc.py` (new)

**Interfaces:**
- Consumes: `Arc`, `Ctx` (Task 5), `Storage`, `Reservoir` (Task 7).
- Produces:

```python
class QueueArc(Arc):                          # arcs.py:290-545
    number_of_timesteps: int                  # n
    calls: dict[str, list[str]]               # per direction, the static daily call labels j (D2)
    rows: Tensor                              # (..., 2, n, J, 1+K), J = max(J_push, J_pull), unused rows empty (D1)
    since: Tensor                             # (..., 2), saturating at n, starts at n; stores() keys "rows", "since"
    def send_push_request(self, vqip, *, active, tag="default", force=False) -> Tensor
        # V < fa: return empty before enqueue and delivery (arcs.py:378-379), masked; the water is ctx.drop-ped
        # else: clip (average_flow V/(n+1) against capacity, :409-430); deliver due rows i = n-1..0, j order,
        # each a push_set with its own tag, rejected parts as backflow (:448-528); rows below fa removed
        # through ctx.drop (:483-485); shift waiting rows to i + since; since = 0; enqueue at (0, j)
        # (n = 0: deliver the new request last)
    def send_pull_request(self, vqip, *, active, tag="default") -> Tensor
        # pull_set without tag (:337, :348), enqueue, return the due pull rows; always reaches
        # update_queue, so since[pull] resets on every active pull call; no additive rescale (D4)
    def end_timestep(self) -> None            # since = min(since + 1, n); accumulators reset
```

- [ ] **Step 1: Write the failing tests** in `test_queue_arc.py`:
  - W21, `test_queue_arc_delivery_matches_wsimod_on_irregular_pushes` (`importorskip`): a mapping with a `QueueArc` (`n = 2`) into a `Storage` of capacity 6 and two senders: `Node A` (the `QueueArc` its only out-arc, fed by a `Catchment`) pushes on days 0, 1, 5 (a gap longer than `n + 1`), 9 (a `1e-12` push on a due day, through the single-arc shortcut, `nodes.py:424-438`) and 10; `Node B` (the `QueueArc` plus an `Arc` to a `Waste`, fed by a `Storage`'s `distribute`) pushes on day 6, when backflow from the full `Storage` makes its `push_distributed` run a second round (two call sites and two rounds on one day); the arc's capacity clips one request to below `fa`, so the update-time drop (`arcs.py:483-485`) is exercised. Both modes with one additive and one non-additive pollutant: every day's arc `vqip_out`, the replies, the `Storage`'s storage within `1e-12 * scale`; on day 9 `vqip_out` volume 0, on day 10 the waiting requests delivered; noodl's mass balance closes (it does only if both drops are recorded).
  - `test_a_queue_arc_pull_hands_over_water_pulled_on_earlier_days` (`importorskip`): `n = 1`, a `Reservoir` (`make_abstractions`) pulling through it from a `Storage`: day 0's reply empty, day 1's reply day 0's pull; matches WSIMOD.
  - `test_a_zero_volume_pull_with_pollutants_skips_the_rescale` (defect row): a request of volume 0 and phosphate 1: reply empty, every value finite, no exception (WSIMOD: `ZeroDivisionError`, `arcs.py:341-343`).
  - `test_the_tiny_push_is_recorded_as_dropped`: `ctx.dropped` volume equals the tiny push.
  - `test_rows_and_anchors_are_carried_state`: `stores()` returns `"rows"` and `"since"`, `since == n` at construction, and `end_timestep` keeps the rows.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_queue_arc.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement. Add case `queue_arc` (the W21 mapping as a whole model; push and pull with delay ≥ 1, a rejecting receiver, days without a push) to `CASES`.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): QueueArc request rows anchored at the last delivery (W21)`.

### Task 9: `River` without biochemistry

**Implementer:** standard.

**Files:**
- Modify: `src/noodl/apps/wsimod/storage.py` (`River`), `tests/verification/_wsimod_cases.py`
- Test: `tests/apps/wsimod/test_storage.py`

**Interfaces:**
- Consumes: `Storage`, `Reservoir` (Task 7).
- Produces: `River` (`storage.py:452-955` without `:738-898`) and `river_derived(spec) -> dict[str, float]` (area, capacity): `area = length*width`, `capacity = 1e15` (`:524-533`), `warnings.warn` when `depth != 2` (`:514-518`); `riverrc(kt) = 1 - kt + kt*exp(-1/kt)` with `kt = damp*length/velocity`, 1 at `kt == 0` (`:900-915`, safe denominator); `distribute` (`:926-936`); push forced, check `1e15` (`:582-583`, `:711-721`); `pull_check_river` with the raised-volume base (`:660-686`); `pull_set_river` (`:688-709`); `calculate_discharge` a no-op. `validate`: `"nitrate"` among the pollutants refused (`"3b"`, `storage.py:917-924`); a `("RiparianBuffer", "volume")` check raises `KeyError` naming `"3c"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_the_river_routing_coefficient`: `damp 0.1, length 1000, velocity 0.2` gives `1 - 500 + 500*math.exp(-1/500)`; `damp 0` gives 1 with a finite gradient for `velocity`.
  - `test_pull_check_spreads_the_rivers_mass_over_upstream_water`: river tank volume 4 with phosphate 2, upstream avail 6, `mrf` 1, `riverrc` 0.5: reply volume `max(4 + 6 - 1/0.5, 0) = 8`, phosphate `2 * 8/10`.
  - `test_a_river_with_nitrate_is_refused`: `match="nitrate.*3b"`.
  - `test_river_derives_what_construction_derives` (`importorskip`, Task 4's mapping): `river_derived` within `1e-15` relative of the WSIMOD object after `add_overrides`.
  - Replay (`CASES`, its own pollutant lists without `nitrate`): `river_chain` (three levels with a tie at one level, `damp = 0` on one river, `mrf > 0`, upstream pulls by a `Reservoir`, a river not linked to the `Waste`), replaying every river.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): River routing, minimum residual flow and upstream pulls`.

### Task 10: Demand and Distribution

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/demand.py` (the `Demand` family, `Distribution`, `UnlimitedDistribution`), `tests/apps/wsimod/test_demand.py`
- Modify: `tests/verification/_wsimod_cases.py`

**Interfaces:**
- Consumes: `Node`, `Storage`, `Groundwater`, `Reservoir`, `ScriptedArc` (Tasks 5, 7).
- Produces: `Demand`, `ResidentialDemand` (`wsimod_type` `"Demand"`), `NonResidentialDemand` (W4: `{"house": Demand.get_demand(self)["default"]}`), `Distribution` (leakage decorators applied once when the configured `leakage > 0`, `distribution.py:13-139`; W17; an unsuccessful leak returned to the caller with the reply, `:40-50`), `UnlimitedDistribution` (`"UnlimitedDistribution.supplied"` term). `create_demand` runs one pull and up to three pushes, in dict order `garden`, `house`, `default` (`demand.py:79-119`), each its own `loop_site`. `validate`: `leakage >= 1` refused; `leakage > 0` on `UnlimitedDistribution` refused; a `ResidentialDemand` `pollutant_load` lacking an additive pollutant refused (pollutants on). `Distribution.bind` refuses a `leakage` driver batch mixing zero and non-zero on a decorated node (Task 13 calls it).

- [ ] **Step 1: Write the failing tests** in `test_demand.py`:
  - `test_non_residential_demand_is_its_constant_demand_as_house` (defect row): `constant_demand 5`, out-arc a `ScriptedArc` with `out_type="Sewer"`: 5 pushed with tag `"Demand"`.
  - `test_an_overridden_leakage_is_applied_once` (defect row): `Distribution` with leakage overridden to 0.2 and a `Groundwater` taking the leak, a request of 10: the upstream request is `10 / 0.8` (12.5; WSIMOD 15.625), 2.5 leaks, the reply volume 10.
  - `test_leakage_that_cannot_leak_returns_to_the_caller`: the same with a `ScriptedArc` (`out_type="Groundwater"`) rejecting the leak: the reply volume 12.5, more than requested (`distribution.py:40-50`).
  - `test_leakage_refusals`: `leakage: 1.0` (`match="leakage"`), `UnlimitedDistribution` with `leakage: 0.1`.
  - `test_residential_house_demand_and_mixed_temperature`: `population` 100, `per_capita` 0.15, `pollutant_load` phosphate 0.01, `constant_temp` 14, `constant_weighting` 0.2, air temperature 10 (`demand.py:303-324`): volume `100*0.15`, phosphate `0.01*100`, temperature `10*(1 - 0.2) + 14*0.2`; with `temperature` not a pollutant the write is dropped.
  - Replay (`CASES`): `distribution_leakage` (with and without a `Groundwater`), `unlimited_distribution`, `demand_constant_and_loads` (`Demand` with `constant_demand` and `pollutant_load`, pushing to a `Waste` through the `default` component), `reservoir_spill` (D14: `Storage` → `Distribution` (leakage 0.2, no `Groundwater`) → `Reservoir` whose excess is below the upstream supply; asserts the days with `"reservoir_spill"` equal the days WSIMOD printed `"Spill at reservoir"`, and the tank ends above capacity on both sides).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): demands and distribution with leakage`.

### Task 11: `Sewer`, `WTW`, `WWTW`, `FWTW`

**Implementer:** most capable (the liquor blend's double `where` and the carried WTW stocks).

**Files:**
- Create: `src/noodl/apps/wsimod/sewer.py`, `src/noodl/apps/wsimod/wtw.py`, `tests/apps/wsimod/test_sewer_wtw.py`
- Modify: `tests/verification/_wsimod_cases.py`

**Interfaces:**
- Consumes: `Node`, `QueueTank`, `Tank` (Tasks 5, 6), `Groundwater`, `Reservoir` (Task 7), `Demand` (Task 10).
- Produces: `Sewer` (`sewer.py:14-247`: `QueueTank` with `number_of_timesteps 0`; tags `"default"`/`"Sewer"` at `pipe_time`, `"Land"`/`"Demand"` over `pipe_timearea`; `make_discharge` `:201-239`; `warnings.warn` when `pipe_timearea` does not sum to 1); `WTW` (`wtw.py:14-220`), `WWTW` (`:223-453`), `FWTW` (`:456-657`) per the spec's Nodes section, with `liquor`, `previous_pulled` and `treated` as carried stores (`stores()` keys `"liquor"`, `"previous_pulled"`, `"treated"`); `wtw_derived(spec)` and `sewer_derived(spec)` (volume constant; tank fields). The liquor blend keeps the previous non-additive value where `V_liq + V_in*m_v == 0` (double `where`, event `"wtw_liquor_zero_denominator"`). `make_discharge`'s rejection at or below `fa` vanishes into `ctx.drop` (W25). `FWTW`'s deficit is created water at yesterday's `previous_pulled` quality, a `"FWTW.total_deficit"` term. `validate` (pollutants on): `process_parameters[p] == {constant: c_v, exponent: 1}` with `c_v = 1 - percent_solids - liquor_multiplier["volume"]` and `liquor_multiplier[p] == liquor_multiplier["volume"]` for every additive `p` (WSIMOD's defaults refused, `match="process_parameters.*3b"`); additive pollutants without `temperature` refused (`match="temperature"`).

- [ ] **Step 1: Write the failing tests** in `test_sewer_wtw.py`:
  - `test_wtw_and_sewer_derive_what_construction_derives` (`importorskip`, Task 4's mapping): the WTW volume constants and the `Sewer` tank capacity and fields within `1e-15` relative of the WSIMOD objects after `add_overrides`.
  - `test_wtw_splits_by_its_volume_constant`: `percent_solids 0.0002`, `liquor_multiplier["volume"] 0.03`, input 100 (neutral pollutants, phosphate 1): treated `100*(1 - 0.0002 - 0.03)`, liquor `100*0.03`, solids `100*0.0002`; phosphate split in the same proportions.
  - `test_zero_input_on_day_one_keeps_the_previous_liquor_value` (defect row): a `WWTW` and an `FWTW` with zero input on day 1, `temperature` pollutant: liquor temperature 0 (its previous value), finite gradients, event `"wtw_liquor_zero_denominator"` true; with `liquor_multiplier["volume"] = 0` on every day.
  - `test_a_tiny_discharge_rejection_vanishes_and_is_recorded_as_dropped` (W25): `make_discharge` into a `ScriptedArc` rejecting `5e-12` with `fa = 1e-11`: storm tank unchanged, `ctx.dropped` volume `5e-12`; rejecting 1.0: forced into the storm tank, event `"wwtw_couldnt_push"`.
  - `test_treated_accumulates_without_make_discharge`: a custom orchestration of `WWTW.calculate_discharge` only: `treated` volume after day 2 is the sum of both days' treated volume.
  - `test_the_fwtw_deficit_takes_yesterdays_quality`: day 1 pulls water at temperature 8; day 2 pulls nothing: day 2's created water has temperature 8.
  - `test_non_neutral_wtw_processes_are_refused` and `test_additive_pollutants_need_temperature`.
  - Replay (`CASES`): `sewer_timearea` (`pipe_time 1`, `pipe_timearea {0: 0.5, 1: 0.3, 2: 0.2}`, reached through the `"Demand"` tag from a `Demand` and the default tag from a `Catchment`-fed `Node`), `foul_sewers` (a node named `foul_main` and a custom orchestration naming `"Foul"`), `groundwater_infiltrate` (`infiltration_pct > 0` into a `Sewer`), `wwtw_storm_tank` (filling and draining past throughput), `fwtw_deficit_sludge` (a deficit, liquor and solids to a `Sewer`, a full sewer for unpushed sludge), `queue_tank_time_zero` (`pipe_time 0`: active and discharged on the same day); and noodl-only `wtw_zero_input` (`wsimod=False`).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): Sewer and the water treatment works`.

### Task 12: `Land` and its surfaces

**Implementer:** most capable (IHACRES's branches and kinks, the evident-intent ponding).

**Files:**
- Create: `src/noodl/apps/wsimod/land.py`, `tests/apps/wsimod/test_land.py`
- Modify: `tests/verification/_wsimod_cases.py`

**Interfaces:**
- Consumes: `Node`, `ResidenceTank`, `DecayTank` (Tasks 5, 6), `Groundwater` (Task 7), `ResidentialDemand` (Task 10), `Sewer` (Task 11).
- Produces: `Land` (`land.py:17-300`), `Surface`, `ImperviousSurface`, `PerviousSurface` (`:303-1079` without deposition and soil-temperature processes), and `surface_derived(spec: SurfaceSpec) -> dict[str, float]` per surface class (capacity, simulated depth, `field_capacity_m`, `wilting_point_m`, `subsurface_coefficient`). `Land.run` (`:200-244`; the percolation push-back only `> fa`, else `ctx.drop`); the evident-intent `"Sewer"` check and set (W4: accepts all when the land has an `ImperviousSurface`, forces into the first); `apply_irrigation` a no-op; the garden query answers 0 (no `GardenSurface`, W16). IHACRES per the spec (`:913-1044`), the evaporation-clamp event `"inaccurate_evaporation"`. `stores()` keys `"surface_runoff.storage"`, `"subsurface_runoff.storage"`, `"percolation.storage"`, `"surfaces.<name>.storage"` and the surfaces' per-day flows. `validate`: `total_porosity <= field_capacity` or `<= wilting_point` refused after overrides; non-zero `pollutant_load` or a non-additive entry refused (pollutants on, `"3c"`); `temperature` a pollutant needs `soil_temp_w_air = soil_temp_w_deep = 0`, `soil_temp_w_prev > 0` (`"3b"`); surface data per D21.

- [ ] **Step 1: Write the failing tests** in `test_land.py`:
  - `test_impervious_surface_ponds_and_evaporates`: area 10, `pore_depth` 0, precipitation 0.003, et0 0.001, `et0_to_e` 1: `(0.003 - 0.001)*10` pushed to the `Sewer`; with precipitation 0.0005: evaporation `min((0.001 - 0.0005)*10, storage)`.
  - Review Focus 4, `test_an_impervious_surface_starting_above_capacity_empties_on_day_one`: oxford's urban numbers (area 1e7, capacity 0, initial 5e6) into an accepting `Sewer`: storage 0 after day 1, 5e6 plus the day's net rain pushed.
  - `test_ihacres_by_hand`: one `PerviousSurface` day at a hand-computed point away from the `min(1, ·)` kinks (the parameters and every intermediate of `:913-1044` written in the test as Python expressions).
  - `test_sewer_ponding_into_land_is_forced_into_the_first_impervious_surface` (defect row): push 3 with tag `"Sewer"`: the check returns 3 and the impervious storage gains 3; a land without an impervious surface: the check returns volume 0.
  - `test_garden_demand_is_zero_without_a_garden_surface`: a `ResidentialDemand` with a `Land` of one pervious surface: the garden push is made (one `get_connected` to `"Land"` with tag `("Demand", "Garden")`) and is 0.
  - `test_porosity_not_above_field_capacity_is_refused` (defect row): `total_porosity 0.3, field_capacity 0.3`: `match="total_porosity"`; the same after an override.
  - `test_surface_decay_reads_temperature_as_an_identity`: `DecayTank.end_timestep` leaves the storage unchanged.
  - `test_surfaces_derive_what_construction_derives` (`importorskip`, Task 4's mapping): `surface_derived` and the `Land` residence tanks within `1e-15` relative of the WSIMOD objects after `add_overrides`.
  - `test_gradcheck_on_one_ihacres_day`.
  - Replay (`CASES`): `impervious_only`, `pervious_only` (both IHACRES branches over the days), `evaporation_clamp` (WSIMOD prints `"inaccurate evaporation"`), `temperature_blending` (non-additive temperature through runoff and a `Sewer`), `land_percolation_threshold`, `residential_with_land` (garden 0); noodl-only `land_sewer_ponding` (`wsimod=False`: `Land → Sewer` runoff plus a `Sewer → Land` ponding arc, the legal topological cycle).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS.
- [ ] **Step 5:** Commit: `feat(wsimod): Land with impervious and IHACRES pervious surfaces`.

### Task 13: Structure, `build_model` and the `WsimodDay` closure (W6, part 2)

**Implementer:** most capable (state layout, tracing day, recursion graph).

**Files:**
- Create: `src/noodl/apps/wsimod/structure.py`, `src/noodl/apps/wsimod/day.py`, `src/noodl/apps/wsimod/build.py`, `tests/apps/wsimod/test_build.py`, `docs/applications/wsimod.md`
- Modify: `src/noodl/apps/wsimod/__init__.py` (exports below, docstring); delete `src/noodl/apps/wsimod/network.py`; `tests/apps/test_initialisation.py`, `tests/test_refs.py`, `tests/test_units.py` (the old builder's callers move to `small`), `tests/test_docs_usage.py` (`PAGES` gains `applications/wsimod.md`; `FIXTURES` gains `wsimod/small/*`), `tests/test_model.py`, `docs/applications/allocation.md` ("From a WSIMOD topology" builds the layer directly; links to the new page), `docs/applications/index.md` (a WSIMOD row; the allocation row names `AllocatedFlowLayer` only), `docs/api.md`, `scripts/gen_catalogue.py` (`("wsimod", "WSIMOD", "../applications/wsimod.md", ["noodl.apps.wsimod"], ...)`), `docs/catalogue/wsimod.md` (regenerated), `mkdocs.yml`

**Interfaces:**
- Consumes: every node and arc class (Tasks 5–12), `Node.stores`/`Arc.stores`, `read_wsimod_config` (Task 4).
- Produces:

```python
# structure.py
def check_network(network: WsimodNetwork, *, pollutants: bool) -> list[str]
    # every build-time refusal (D7) by name; returns ignored_by_quantity_only
    # ("<node>.<parameter>", "<node>.surfaces.<surface>.<parameter>")
def recursion_cycles(network) -> list[list[str]]         # per (direction, tag), from check_recurses (spec, Approach)
def river_order(network) -> list[str]                     # add_arcs :697-735 + assign_upstream :780-807; ties by configuration order (W13)
def orchestration_plan(network) -> list[tuple[str, str, list[str]]]   # (slot key, method, node names); a key naming no node skipped
def forcing_readers(network) -> dict[str, list[str]]      # variable -> nodes reading it (spec, Forcing checks)
# day.py
class WsimodDay:
    integrates = True
    state_keys: tuple[str, ...]   # "wsimod.storage", "wsimod.active_storage", "wsimod.travel_queue",
                                  # "wsimod.arc_queue", "wsimod.arc_queue_since", "wsimod.liquor",
                                  # "wsimod.previous_pulled", "wsimod.treated" (labels per D10)
    output_units: dict[str, str]; key_labels: dict[str, tuple]; input_specs: dict
    def __init__(self, network: WsimodNetwork, *, pollutants: bool) -> None
    engine: Engine; river: list[str]; plan: list[tuple[str, str, list[str]]]
    last_record: DayRecord | None
    def __call__(self, state, drivers, ctx: StepContext | None) -> dict[str, Tensor]
        # ctx None: {}; t whole days inside the dates (D16); load nodes' and arcs' stores from state,
        # bind drivers' day-k slices, run the plan, River.distribute in river order, record,
        # end_timestep (nodes then arcs), dump nodes' and arcs' stores
@dataclass
class DayRecord:
    arc_flow: Tensor; arc_quality: Tensor; tank_storage: Tensor; tank_quality: Tensor
    surface: dict[str, Tensor]; surface_quality: Tensor
    terms: dict[str, Tensor]; dropped: Tensor; storage_total: Tensor        # (..., 1 + K_add)
    events: dict[str, Tensor]; maxiter: dict; decisions: list | None
# build.py
def build_model(network: WsimodNetwork, *, pollutants: bool = True) -> tuple[Model, State, Drivers]
    # Network: one node per WSIMOD node, one edge of kind "arc" per arc (named as in the config);
    # Model(net, [], closures=[WsimodDay(...)]); the tracing day (D2); model.wsimod_day,
    # model.ignored_by_quantity_only, model.driver_template
def initial_state(model: Model, *, values: Mapping | None = None) -> State
    # the configuration's values, then values by key and row label: a number is a volume with zero
    # pollutants, a dict a VQIP by name; QueueTank active_storage = storage, registers and rows empty,
    # since = n
def initial_drivers(model: Model, *, values: Mapping | None = None) -> Drivers
    # D6 names; forcing "wsimod.precipitation" "wsimod.et0" "wsimod.temperature" (..., T, n_readers),
    # "wsimod.flow" (..., T, n_catchments), "wsimod.concentration" (..., T, n_catchments, K)
```

- Units: volumes `m3`, flows and capacities `m3/d`, additive pollutants `kg`, non-additive by name (`temperature` `degC`), never converted (W20).

- [ ] **Step 1: Write the failing tests** in `test_build.py` (and `tests/test_model.py`):
  - `test_a_layer_less_model_steps_an_integrating_closure` (D17, `tests/test_model.py`): `Model(net, [], closures=[c])` with a closure adding 1 to its state key: `step(..., dt=86400.0, t=0.0)` returns 1.
  - `test_the_small_model_builds_and_steps`: `build_model(read_wsimod_config(small))`; `model.step(state, drivers, dt=86400, t=0)` changes `"wsimod.storage"` row `("river", "tank")`; the model's `Network` has edges `inflow_to_river`, `river_to_outlet` of kind `"arc"`; `model.check(state, drivers)` names the units.
  - `test_initial_state_and_drivers_take_values_by_label`: `values={"wsimod.storage": {("river", "tank"): 9.0}}` sets 9; `{"wsimod.Arc.capacity": {"river_to_outlet": 0.3}}` sets 0.3; an unknown label `KeyError` with a suggestion.
  - `test_structural_refusals` (parametrised, `match=`): a two-`Node` `push_check` cycle (names both nodes); `Land → Sewer` plus `Sewer → Land` accepted (topological, not recursive); two out-arcs of preference 0 and a negative preference; `arc_into_catchment` (defect row); `river_without_waste` (defect row); a slot naming a `Waste` with method `distribute`; missing forcing (node, variable, first missing date), including `temperature` for a `Land` with `K = 0`; a `leakage` batch mixing 0 and 0.1 on a decorated `Distribution`.
  - `test_quantity_only_lists_the_ignored_parameters`: quickstart with `pollutants=False` builds despite its non-neutral surface loads; `ignored_by_quantity_only` contains `"my_land.surfaces.urban.pollutant_load"` and `"my_land.surfaces.rural.pollutant_load"`; with `pollutants=True` it is refused (`"3c"`).
  - `test_slot_keys_decide_what_runs`: a node written `type_: ResidentialDemand` without the override is skipped by the default orchestration (its demand never created), as `model.py:658-660`.
  - `test_the_trace_fixes_the_call_sequence`: the small model's `WsimodDay` call sequence on day 0 and day 5 is identical.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_build.py tests/test_model.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement. Delete `network.py`; move `test_initialisation.py`'s wsimod block, `test_refs.py`'s units block and `test_units.py`'s `_wsimod` to `small` (assert `wsimod.storage` in `m3`, `wsimod.Arc.capacity` in `m3/d`); `git grep -n "apps.wsimod.network\|wsimod.build_model(.*topology"` empty. Write `docs/applications/wsimod.md`: what is ported, running (`read_wsimod_config`, `build_model`, `initial_state`, `initial_drivers`, `model.step`) with a runnable example on `small`, quantity-only versus pollutants, refusals, units; rewrite the allocation page's section; regenerate the catalogue.
- [ ] **Step 4:** Run the wsimod subset, `tests/test_docs_usage.py`, `tests/test_docs_catalogue.py`; expected: PASS. `mkdocs build --strict`.
- [ ] **Step 5:** Commit: `feat(wsimod): build_model and the WsimodDay closure replace the allocation builder`.

### Task 14: `wsimod_run`, results, diagnostics and the mass balance

**Implementer:** standard.

**Files:**
- Create: `src/noodl/apps/wsimod/run.py`, `tests/apps/wsimod/test_run.py`
- Modify: `src/noodl/apps/wsimod/__init__.py`, `docs/applications/wsimod.md` (running over the dates, results, diagnostics), `tests/verification/_wsimod_cases.py`

**Interfaces:**
- Consumes: `WsimodDay`, `DayRecord` (Task 13).
- Produces:

```python
@dataclass
class WsimodMassBalance:
    inflow: Tensor; outflow: Tensor; storage_change: Tensor; residual: Tensor   # (..., T, 1 + K_add)
    terms: dict[str, Tensor]                                                    # D11 keys
@dataclass
class WsimodResults:
    dates: list[str]; arc_names: list[str]
    arc_flow: Tensor; arc_quality: Tensor                  # (..., T, n_arcs), (..., T, n_arcs, K)
    tank_names: list[tuple[str, str]]; tank_storage: Tensor; tank_quality: Tensor   # D10, record_all's classes only
    surface_names: list[tuple[str, str]]; surface: dict[str, Tensor]; surface_quality: Tensor
        # surfaces of nodes under the slot key "Land" (model.py:1135); NaN where WSIMOD records nothing
    mass_balance: WsimodMassBalance
def wsimod_run(model: Model, state: State | None = None, drivers: Drivers | None = None, *,
               days: int | None = None, diagnostics: dict | None = None) -> WsimodResults
    # torch.no_grad() unless an input requires grad (W9); diagnostics: "maxiter", "events",
    # "decisions" (only when diagnostics["decisions"] is True on entry), "mass_balance"
```

- [ ] **Step 1: Write the failing tests** in `test_run.py`:
  - `test_the_small_model_matches_wsimod` (`importorskip`): `wsimod_run` on `small` against `wsimod_reference(small)`: arc flows and the river tank within `1e-12 * scale`.
  - `test_step_by_step_equals_wsimod_run`: `small` and case `queue_arc`: 10 `model.step`s equal `wsimod_run(days=10)` (`torch.equal`).
  - Review Focus 2, `test_a_batch_equals_its_single_runs`: case `batch_divergence` (added to `CASES` here, `wsimod=True` so Task 17 also checks it against WSIMOD: `three_out_arcs`'s split `Node` fed by `pervious_only`'s `Land` instead of the catchments) batched over 4 parameter sets chosen so instance 0 is over-subscribed on day 3 and instance 1 is not, and instances 2 and 3 take different IHACRES branches on day 5 (asserted from `"decisions"`): each instance `torch.equal` to its single run.
  - `test_the_mass_balance_closes_by_hand`: `small` over 10 days: `inflow` volume = catchment flow; `residual` within `1e-12 * scale` every day; `terms` keys are D11's.
  - `test_diagnostics_record_maxiter_events_and_decisions`: case `maxiter_pull`: a `"maxiter"` entry true on the days WSIMOD printed `"Maxiter reached in <node> at <date>"`; `"events"` keys are D13's with shape `(T, n_nodes)`; `"decisions"` present only when asked.
  - Review Focus 5, `test_stepping_outside_the_dates_is_refused`: `model.step` with `t=None`, `t=43200`, `t=10*86400` on `small`: `ValueError` matching `"t"`; `wsimod_run(days=0)` and `days=11`: `match="days"`; a model without dates: `match="dates"`.
  - `test_wsimod_is_never_imported`: a subprocess runs `import noodl.apps.wsimod`, builds `small` and `wsimod_run`s it, then asserts `"wsimod" not in sys.modules`.
  - `test_a_run_without_grad_builds_no_graph`: no input requiring grad: `arc_flow.requires_grad` is False; one initial storage requiring grad: True, values `torch.equal`.
- [ ] **Step 2:** Run `pytest tests/apps/wsimod/test_run.py -v`; expected: FAIL.
- [ ] **Step 3:** Implement; document running, results and diagnostics.
- [ ] **Step 4:** Run them, then the wsimod subset; expected: PASS. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(wsimod): wsimod_run with WSIMOD's records, diagnostics and a real mass balance`.

### Task 15: Per-node replay on the demos

**Implementer:** standard.

**Files:**
- Modify: `tests/verification/test_wsimod_replay.py`

**Interfaces:**
- Consumes: `capture_node_calls`, `replay` (Task 5), the demo configs (Task 2).

- [ ] **Step 1: Write the tests:** a module-scoped fixture per (demo, config) runs one live WSIMOD capture that wraps every replayed node at once, over the days Task 16's rule 0 fixes (quickstart all 1456, oxford the first 365; the full oxford capture under `-m slow`). `test_demo_nodes_replay_wsimod[<demo>-<class>]` indexes it: quickstart and oxford, quantity-only, every node of `Land` (impervious and pervious, both IHACRES branches among the sampled days, asserted), `Sewer`, `WWTW`, `FWTW`, `Reservoir`, `Groundwater`, `ResidentialDemand`, `Catchment`, `Node`, `Waste`; replayed over the sample: days 0–59, every 20th, plus each captured day with a new branch outcome (D15); bound `1e-12 * scale`, scale per node over its captured records. `test_demo_nodes_replay_wsimod_on_every_day[...]` (`slow`): every captured day. The neutralised config is replayed for the classes that carry pollutants.
- [ ] **Step 2:** Run them, and `-m slow`; a difference beyond the bound is debugged to its cause before any admission. Record the sample sizes, the per-worker capture time (xdist runs a module fixture once per worker) and the test times in the progress file.
- [ ] **Step 3:** Commit: `test(wsimod): every demo node replays WSIMOD's captured calls`.

### Task 16: Whole-model parity on the demos, then the suite cut (W15, W23)

**Implementer:** standard (escalate a divergence it cannot explain to most capable).

**Files:**
- Create: `tests/verification/test_wsimod_parity.py`, `tests/verification/_wsimod_parity.py` (the comparison)
- Modify: `AGENTS.md` (the chosen composition in a note under the table), `docs/applications/wsimod.md` (verification table), `tests/verification/test_wsimod_replay.py` (only if step 3 cuts it)

**Interfaces (produced, `_wsimod_parity.py`):**

```python
@dataclass(frozen=True)
class Admission: case: str; element: str; day: int; site: str
def assert_matches_wsimod(ours: WsimodResults, theirs: WsimodReference, *, diagnostics: dict,
                          case: str, quality: bool, admitted: Sequence[Admission] = ()) -> dict
    # every arc flow and pollutant, every recorded tank, every surface field, every day:
    # within 1e-12 * scale, or in `admitted` with a decision |margin| <= 1e-9 * scale that day;
    # the dated prints of D13 against "demand_deficit" and "maxiter" day by day; no
    # "mass balance error" in theirs.printed; names the first divergence;
    # returns {"worst", "scale", "admissions"}
```

- [ ] **Step 1: Write the tests:** `test_demo_matches_wsimod[quickstart-quantity|quickstart-neutral|oxford365-quantity|oxford365-neutral]` and, `slow`, `[oxford-quantity|oxford-neutral]` (full); each also asserts the mass balance within `1e-12 * scale` every day. `test_quantity_only_equals_the_neutralised_volumes[quickstart]`: every volume series `torch.equal` between the two modes (spec, Verification 3). oxford's urban surface: storage 0 after day 1 (Review Focus 4, through the whole model).
- [ ] **Step 2:** Run them; expected: PASS (debug before admitting; each admission named in the test with its measured margin).
- [ ] **Step 3: Measure, then cut (W23).** Run `python -m pytest -n 4 --no-cov --durations=0 <the wsimod row>`; write each default candidate's wall time (quickstart per mode, oxford-365 per mode, the demo captures, the sampled replays, the case set) and the row's total in the progress file. The rules, in order: (0) the default demo capture covers the days the sample can reach (quickstart all 1456, oxford the first 365), the full capture `slow` (Task 15). If the total exceeds 10 minutes, apply until it fits, re-measuring after each: (1) oxford-365 to `slow`, keeping oxford's first 60 days (and a 60-day oxford capture) in the default; (2) the replay sample to every 40th day; (3) quickstart's quantity-only mode to its first 365 days, the neutralised mode staying full. Nothing asserts a time.
- [ ] **Step 4:** Record the chosen composition in `AGENTS.md` and in the application page's verification table, with the measured numbers.
- [ ] **Step 5:** Commit: `test(wsimod): quickstart and oxford match WSIMOD 0.8.1 in both modes`.

### Task 17: Purpose-built cases, whole model

**Implementer:** standard.

**Files:**
- Modify: `tests/verification/_wsimod_cases.py`, `tests/verification/test_wsimod_parity.py`

**Interfaces:**
- Consumes: `CASES` (Tasks 5–14), `assert_matches_wsimod` (Task 16).

- [ ] **Step 1: Add the cases no node task owns:** `capacity_first_come` (binding arc capacities served first come, first served across two orchestration slots), `simple_pollutants` (`set_simple_pollutants`'s lists), `float_accuracy_override` (`float_accuracy: 1e-6`, a push between 1e-11 and 1e-6 dropped), `node_type_override` (a `ResidentialDemand` under `type_: Demand`). Check every spec case is in `CASES`, by its task: 5 `three_out_arcs`, `node_chain`; 7 `forced_negative_excess`, `maxiter_pull`, `pull_and_push_arcs`, `user_alt_queue_arc`, `storage_distribute`, `queue_groundwater`, `reservoir_over_capacity`, `river_reservoir_environmental`, `catchment_abstraction`; 8 `queue_arc`; 9 `river_chain`; 10 `distribution_leakage`, `unlimited_distribution`, `demand_constant_and_loads`, `reservoir_spill`; 11 `sewer_timearea`, `foul_sewers`, `groundwater_infiltrate`, `wwtw_storm_tank`, `fwtw_deficit_sludge`, `queue_tank_time_zero`, `wtw_zero_input`; 12 `impervious_only`, `pervious_only`, `evaporation_clamp`, `temperature_blending`, `land_percolation_threshold`, `residential_with_land`, `land_sewer_ponding`; 14 `batch_divergence`; 17 the four above.
- [ ] **Step 2: Write the tests:** `test_case_matches_wsimod[<case>-<mode>]` over every `wsimod=True` case and mode (mass balance and prints as Task 16); `test_case_runs_where_wsimod_crashes[land_sewer_ponding|wtw_zero_input]` (the hand expectations of Tasks 11 and 12 through the whole model, the mass balance closing, and WSIMOD raising its documented exception: `AttributeError`/`KeyError`, `ZeroDivisionError`); `test_quantity_only_equals_the_pollutant_run_volumes[<case>]` (`torch.equal`).
- [ ] **Step 3:** Run them; expected: PASS (debug before admitting).
- [ ] **Step 4:** Commit: `test(wsimod): purpose-built cases for every feature the demos skip`.

### Task 18: Gradients

**Implementer:** standard.

**Files:**
- Create: `tests/verification/test_wsimod_gradients.py`

**Interfaces:**
- Consumes: `wsimod_run(..., diagnostics={"decisions": True})` (Task 14), `CASES`.

- [ ] **Step 1: Write the tests:** `test_gradient_matches_central_differences[<parameter>]`: autograd of a scalar (an arc's flow summed over days, a final storage, or a pollutant mass) against `(f(p + h) - f(p - h)) / 2h`, `h = 1e-6 * |p|`, `|ad - fd| <= 1e-6 * max(|ad|, |fd|)`, the ±h runs' decision outcomes identical. Parameters (spec, Verification 4), on quickstart 60 days or the named case: arc capacity (binding, `capacity_first_come`), preference (`three_out_arcs`), `Groundwater.residence_time`, `PerviousSurface.field_capacity`, `ihacres_p`, `et0_coefficient`, `ImperviousSurface.et0_to_e`, the `Land` residence times, `River.velocity` and `mrf` (`river_chain`), `WWTW` and `FWTW` throughput (`wwtw_storm_tank`, `fwtw_deficit_sludge`), `ResidentialDemand.per_capita`, `Distribution.leakage` (`distribution_leakage`), a time-area fraction (`sewer_timearea`), the precipitation and catchment-flow forcing, an initial storage, a catchment concentration and a `Demand.pollutant_load` (neutral mode). Each point is checked off the spec's known jumps (decision margins above `1e-6 * scale` in the record).
- [ ] **Step 2:** Run them; expected: PASS (the mechanism is Tasks 3–14's; a failure is a defect fixed in its task's code, with a test there).
- [ ] **Step 3:** Commit: `test(wsimod): gradients against central differences`.

### Task 19: Documentation, benchmark, final checks

**Implementer:** standard.

**Files:**
- Create: `benchmarks/wsimod_run.py`
- Modify: `docs/applications/wsimod.md`, `docs/formats/wsimod.md`, `docs/verification.md`, `docs/applications/index.md`, `docs/catalogue/wsimod.md`

- [ ] **Step 1:** `wsimod.md` complete: what is ported (the scope table, by class), running, quantity-only versus pollutants (W10, "until 3b, real configurations need `pollutants=False`"), refusals, the defect table with noodl's behaviour, differentiability (learnable, structural, detached; the known jumps and kinks), batches and their structural limits, graph size and fitting in windows (W9), units never converted (W20), the river tie rule (W13), the verification table with Tasks 15–18's measured numbers and the suite composition, limitations (3b, 3c, IO, extensions, speed W8). `docs/verification.md`: WSIMOD rows (replay, demos per mode, cases, gradients) with measured worst values. Regenerate the catalogue.
- [ ] **Step 2:** `benchmarks/wsimod_run.py`: quickstart and oxford, batch sizes 1, 10 and 100, time per day and the autograd graph size (node count) of a 60-day run with one input requiring grad; prints, no asserts. Run it once; record in the progress file.
- [ ] **Step 3:** `ruff check .`, the full `pytest` with coverage (≥ 90 %), `-m slow` on the wsimod row, `python -m mkdocs build --strict`; expected: all pass. Record the wsimod row's default wall time against Task 16's budget.
- [ ] **Step 4:** Commit: `docs(wsimod): the WSIMOD application page, verification and benchmark`.

## Defect table: each row and its test

| Row (spec, WSIMOD defects) | noodl | Test (task) |
|---|---|---|
| `Land.push_set_sewer` + no `"Sewer"` check | evident intent | `test_sewer_ponding_into_land_is_forced_into_the_first_impervious_surface` (12); `test_case_runs_where_wsimod_crashes[land_sewer_ponding]` (17) |
| `EnfieldFoulSewer.make_discharge` | refused by name | `test_refusals_name_the_key[EnfieldFoulSewer]` (4) |
| `NonResidentialDemand.get_demand` recursion | evident intent | `test_non_residential_demand_is_its_constant_demand_as_house` (10) |
| `GrowingSurface.residue` unwired | 3c; refused as a class | `test_refusals_name_the_key[GrowingSurface]` (4) |
| System mass balance compares `sys_in` with itself | real balance | `test_the_mass_balance_closes_by_hand` (14); every parity test (16, 17) |
| `Distribution.apply_overrides` re-decorates | once | `test_an_overridden_leakage_is_applied_once` (10) |
| `UnlimitedDistribution` loses its leakage | refused | `test_leakage_refusals` (10) |
| Overrides update shared default dicts | per node | `test_an_override_of_pollutant_load_reaches_only_its_node` (4) |
| River arcs but no `Waste` | refused | `test_structural_refusals[river_without_waste]` (13) |
| An arc into a `Catchment` | refused | `test_structural_refusals[arc_into_catchment]` (13) |
| WTW liquor blend divides by zero | previous value kept, event | `test_zero_input_on_day_one_keeps_the_previous_liquor_value` (11); `test_case_runs_where_wsimod_crashes[wtw_zero_input]` (17) |
| `QueueArc` push below `FLOAT_ACCURACY` (reproduced, W24) | reproduced, dropped recorded | `test_queue_arc_delivery_matches_wsimod_on_irregular_pushes` (8); `test_the_tiny_push_is_recorded_as_dropped` (8) |
| `QueueArc` zero-volume pull with additive keys (W26) | rescale skipped | `test_a_zero_volume_pull_with_pollutants_skips_the_rescale` (8) |
| `WWTW.make_discharge` rejection ≤ `FLOAT_ACCURACY` (W25) | tolerance, dropped recorded | `test_a_tiny_discharge_rejection_vanishes_and_is_recorded_as_dropped` (11) |
| `decays: {}` on the `Storage` family | refused (3b) | `test_a_decays_dict_on_a_storage_node_is_refused` (7) |
| `total_porosity` not above `field_capacity`/`wilting_point` | refused | `test_porosity_not_above_field_capacity_is_refused` (12) |

## Self-review (done, after the plan review)

- **Spec coverage.** Scope table: VQIP (3), constants (3), `Model.run` order, recording, `end_timestep` (13, 14), `Model.load` and dates (4), `Node`/arcs/`Catchment` (5), tanks and `AltQueueArc` (6), the storage family (7), `QueueArc` (8), `River` (9), demands and distribution (10), `Sewer`/WTWs (11), `Land`/surfaces (12). Out-of-scope refusals (4, 7, 9, 11, 12). Approach rules (Global Constraints; 3, 5). Recursion graph and cycles (5's `check_recurses`, 13). Representation: pollutant lists (3), neutral processes (7–12 `validate`), state dict (13, D1, D10), shift register (6), `QueueArc` (8, D1), batches and structural inputs (10, 13, 14). Daily driver, slot keys, river order, methods table (13). Entry points (4, 13, 14). Inputs, outputs, diagnostics, mass balance with six drop sites (13, 14, D11–D13). Defect table (above). Loading (4). Differentiability (3, 12, 14, 18; docs 19). Verification 1 (5–12, 15), 2 (16, 17, W23), 3 (4, 13, 14, 16, 17), 4 (18). W6 (1, 13). Documentation (4, 13, 14, 16, 19). Benchmark (19).
- **Dependency order.** Each task consumes only earlier tasks' classes; each case sits in the first task where all its classes exist (Task 17's checklist); unit tests needing a receiver use `StoreNode` or `ScriptedArc`.
- **Interfaces across the renumbering.** `Arc.stores`/`load_stores` (5) are filled by `AltQueueArc` (6) and `QueueArc` (8) and loaded by `WsimodDay` (13); `ScriptedArc.flow_in`/`vqip_in` (5) serve the `Catchment` replay (5, 15); the derived-value functions `river_derived` (9), `wtw_derived`/`sewer_derived` (11), `surface_derived` (12) are what Task 4's merged parameters feed; `assert_matches_wsimod` (16) takes `diagnostics`, as Task 17 passes it.
- **Optional review suggestions** applied: D13's extended events with the dated-print check, the commit trailer, `J = max(J_push, J_pull)`, D17's test file, the `verbose` question (D23), `batch_divergence` stated as a WSIMOD case. None skipped.
