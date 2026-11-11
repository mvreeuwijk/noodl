# EPANET Water Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `water_run` runs EPANET 2.2's water quality (every mode, reaction, source type and tank mixing model, read from the `.inp`) and matches EPANET 2.2 on its example networks and on one small case per feature.

**Architecture:** A transcription of EPANET's Lagrangian procedure (`quality.c`, `qualroute.c`, `qualreact.c`). A `QualityRouter` keeps every segment in one pool (1-D float64 tensors `c`, `v`; EPANET's `FirstSeg`/`LastSeg`/`prev` chains as Python index lists), reacts the pool in one vectorised operation per quality step and walks the nodes in EPANET's sorted order. `water_run` calls it after each solve with that solve's flows, states and delivered demands; quality never feeds back.

**Tech Stack:** Python 3.12, PyTorch (float64), pytest, `wntr` 1.5.0 (EPANET 2.2's DLL through `wntr.epanet.toolkit.ENepanet`/`ENlib`; verification only).

**Spec:** `docs/superpowers/specs/2026-11-11-epanet-quality-design.md` (read it whole; it is the authority where this plan is silent).

## Global Constraints

- **Start point.** Worktree `C:/Users/mvr/AppData/Local/Temp/noodl-wt/epanet-quality`, branch `epanet-quality` (from `main` 595f5e7 plus the spec commits); `main` merged in after PR #35 lands (Task 0, else Task 13). One PR, opened by the lead after a Fable review; this plan never pushes.
- **Interpreter and tests.** `C:/projects/noodl/.venv/Scripts/python.exe`, `PYTHONPATH=src`, from the worktree root. "The water subset": `python -m pytest -q -n 4 --no-cov tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py tests/apps/water tests/apps/inp tests/verification/test_water_parity.py tests/verification/test_water_networks.py tests/verification/test_water_quality*.py tests/verification/test_wsimod_*.py` (the AGENTS.md core list and `water` row; the quality glob joins the row in Task 2). `-m slow` when a task changes what a slow test compares.
- **Before each commit** `ruff check .`; **at the end of each task** `python -m mkdocs build --strict`. A task that adds or changes a public name runs `python scripts/gen_catalogue.py` (`tests/test_docs_catalogue.py`).
- **Git.** Identity `58522531+mvreeuwijk@users.noreply.github.com`. Never `git stash`. Never push. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the model that wrote it).
- **Names.** Physical names; no aliases; code, tests and docs change together; no wall-clock assertions.
- **Neutrality.** `NOODL_GOLDEN_EXACT=1 pytest tests/test_golden_orchestration.py` green after Tasks 3, 8, 13. Hydraulics-only speed: `benchmarks/water_eps.py` within ~5 % of Task 0's baseline (Task 13).
- **Transcription.** Every transcribed formula cites its EPANET 2.2 file and function (tag `v2.2`) in its docstring; EPANET's quirks (below) are transcribed, not fixed.
- **Reference.** EPANET 2.2 through the toolkit on the committed file, never `EpanetSimulator`'s rewrite; EPANET's copy adds only `[OPTIONS] ACCURACY 1e-8`, `TRIALS 400` (the hydraulic parity's setting, `test_water_networks.py`) and `[REPORT] STATUS YES`; noodl reads the unedited file.
- **Bounds (spec, Verification).** Node and link quality over the full (node, time) matrix: 99.9th percentile within 1e-5 of the quality's scale, worst within `Ctol` absolute; a worst case beyond 1e-5 of scale is admitted only with the flipped merge (segment count differing by one at that link and step) written in the test. Mass balance: terms 2e-5 relative, ratio 1e-6. Loosen only with a measured cause in the test.
- **Progress file** `.superpowers/sdd/epanet-quality/progress.md` (not committed): baselines, measured numbers, open causes.

## EPANET facts the tasks rely on (from the v2.2 source)

- **Indices.** Nodes: junctions in file order, then reservoirs and tanks in file order (`input3.c`, `juncdata`, `tankdata`); links: file order across `[PIPES] [PUMPS] [VALVES]`. `buildadjlists` (`project.c`) prepends, so adjacency runs in decreasing link index. `sortnodes` pushes in-degree-0 nodes in index order and pops LIFO.
- **CV pipes.** `CVPIPE = 0`, `PIPE = 1` (`types.h`); `initsegs` and `reactpipes` test `Type == PIPE`, so a check-valve pipe has no initial segment and never reacts; `setreactflag` and `adjustdata` include it (`Type <= PIPE`).
- **Pumps and valves** have chains like pipes: empty at the start, filled by `evalnodeoutflow` and emptied by `evalnodeinflow`, counted by `findstoredmass` and `avgqual`, reversed, never reacted.
- **Reversal** (`flowdirchanged`): `q = 0` when `LinkStatus <= CLOSED`, direction 0 when `|q| < Q_STAGNANT`, segments reversed only when `newdir * olddir < 0` (positive, stagnant, negative is never reversed); any direction change re-sorts.
- **2COMP** (`tankmix2`): mixing zone `LastSeg`, stagnant zone `FirstSeg` (`initsegs`'s comments say the reverse); with `vt == 0` the stagnant volume is set to 0, so a tank with `vnet == 0` loses that zone's mass.
- **Last step.** `nextqual` transports only when `Htime <= Dur`, and EPANET's hydraulic step is not shortened to the duration (`nexthyd`): a last step overrunning the duration is not transported.
- **`EN_stepQ` is not `EN_solveQ`.** `stepqual` splits a quality step across a hydraulic event and can call `transport` with `dt = 0`; the harness uses `EN_runQ`/`EN_nextQ` (`EN_solveQ`'s own loop, `epanet.c`): quality at every hydraulic step.
- **Float32 hydraulics.** EPANET's quality reads flows and demands back as `REAL4` (`output.c`, `savehyd`/`readhyd`; closed links saved as 0): a ~6e-8 relative floor.
- **Mass-balance units** (`writemassbalance`): internal sums; CHEMICAL in the chemical's mass unit with `LperFT3 = 28.317` (5.4e-6 above an exact 1000 L/m³), AGE and TRACE in quality x ft³; `.rpt` terms `%12.5e`, ratio `%.5f`; `EN_getstatistic(EN_MASSBALANCE = 4)` gives the ratio in full.
- **`Q_STAGNANT`** `= 0.005 / 448.831` cfs (`quality.c`); in m³/s `Q_STAGNANT_CFS / rounded_cfs_per_m3s(flow_units)` (`units.py`; `1/0.3048**3` when `None`).

## Review Focus

1. **A check-valve pipe (`CV`) in a quality run:** expect EPANET 2.2's answer (no volume, no reaction), not a pipe's travel time and decay. Tests in Task 3 and Task 4.
2. **A link whose flow goes positive, stagnant or closed for a step, then negative:** expect EPANET's qualities with the segments not reversed. Test in Task 3.
3. **A directed flow cycle (a pump driving water round a loop):** expect `selectnonstacknode`'s order and EPANET's qualities, never a hang or an error. Test in Task 3.
4. **A duration no whole hydraulic step reaches (`DURATION 5:30` at 1 h):** expect EPANET's quality and mass balance, the last transport ending at 5:00. Test in Task 8.
5. **A batch (two bulk coefficients, two demand levels):** expect each instance to be its own run (own segments, own node order). Test in Task 8.

---

### Task 0: Baseline

**Files:** none changed.

- [ ] **Step 1:** `git status` clean on `epanet-quality`; `gh pr view 35 --json state`. `MERGED`: `git fetch origin; git merge origin/main` (commit the merge with the footer). `OPEN`: record "merge origin/main after #35, before the PR" in the progress file (create it) and go on.
- [ ] **Step 2:** The water subset (no quality glob yet) and the golden check; expected: pass.
- [ ] **Step 3:** `benchmarks/water_eps.py` three times; record the median.

### Task 1: Quality data and the reader

**Files:**
- Create: `src/noodl/apps/water/quality_data.py` (leaf), `src/noodl/apps/water/inp_quality.py`
- Modify: `src/noodl/apps/water/inp.py` (`[QUALITY] [SOURCES] [REACTIONS] [MIXING]` from `_IGNORED` to `_READ`; `[OPTIONS] QUALITY|DIFFUSIVITY|TOLERANCE`; `[TIMES] QUALITY TIMESTEP`; docstring), `network.py` (`WaterNetwork.quality`, `check()`), `units.py`, `__init__.py`, `docs/formats/epanet.md`, `docs/catalogue/water.md`
- Test: `tests/apps/water/test_inp_quality.py` (new), `tests/apps/water/test_inp.py`

**Interfaces (produced):**

```python
@dataclass(frozen=True)
class QualitySource:
    node: str
    kind: str                 # "CONCEN" | "MASS" | "SETPOINT" | "FLOWPACED"
    strength: float           # quality units; MASS: mass/s (the file's mass/min / 60)
    pattern: str | None = None

@dataclass(frozen=True)
class WaterQuality:
    mode: str = "NONE"                    # "NONE" | "CHEMICAL" | "AGE" | "TRACE"
    chemical: str = "Chemical"            # AGE "Age"; TRACE "Trace <node>"
    units: str = "mg/L"                   # AGE "hrs"; TRACE "%"
    trace_node: str | None = None
    quality_timestep: float | None = None # s; None: hydraulic step // 10 (Task 3)
    tolerance: float = 0.01               # Ctol, quality units (CHEMTOL = AGETOL)
    diffusivity: float = DIFFUSIVITY_M2   # m2/s; 0 switches mass transfer off
    bulk_order: float = 1.0
    wall_order: float = 1.0               # 0 or 1
    tank_order: float = 1.0
    global_bulk: float = 0.0              # (quality unit)^(1-n)/s
    global_wall: float = 0.0              # order 0 mass/m2/s, order 1 m/s
    pipe_bulk: dict[str, float] = field(default_factory=dict)
    pipe_wall: dict[str, float] = field(default_factory=dict)
    tank_bulk: dict[str, float] = field(default_factory=dict)
    limiting_potential: float = 0.0
    roughness_correlation: float = 0.0    # the file's factor x the wall conversion
    initial: dict[str, float] = field(default_factory=dict)
    sources: tuple[QualitySource, ...] = ()
    mixing: dict[str, tuple[str, float]] = field(default_factory=dict)   # tank -> (model, fraction)
    node_order: tuple[str, ...] = ()      # EPANET's node indices; () = builder's order
    link_order: tuple[str, ...] = ()      # EPANET's link indices; () = pipes, pumps, valves
```

- `units.py`: `DIFFUSIVITY_FT2 = 1.3e-8` (`DIFFUS`), `DIFFUSIVITY_M2 = DIFFUSIVITY_FT2 * M_PER_FT**2`, `Q_STAGNANT_CFS = 0.005 / 448.831`, `LITRES_PER_M3 = 1000.0`.
- `WaterNetwork.quality: WaterQuality | None = None`; `read_epanet_inp` always fills it (mode NONE when absent); `None` is a hand-built network without quality; both mean off.
- `read_quality(sections, path, *, us: bool, junctions, reservoirs, tanks, pipes, pumps, valves, patterns) -> tuple[WaterQuality, list[str]]`, the list going to `net.notes["ignored_quality"]`.
- Rules (`input3.c`; cite per rule): keywords match as EPANET's `match` (`input2.c`: case-insensitive, the token begins with the keyword); a later line overrides an earlier; `QUALITY` per `optionchoice` (`CHEMICAL` keeps `"Chemical"`/`"mg/L"` and ignores a following word; another word is CHEMICAL so named, the next token its units); range forms `id1 id2 value` in `[QUALITY]` (all nodes), `BULK`/`WALL` (all links), `TANK` (reservoirs and tanks), numeric when C `atol` of both is positive (leading digits: `atol("10x") == 10`), else `strcmp` order (`qualdata`, `reactdata`); legacy `[SOURCES]` `node value [pattern]` is CONCEN, `*` no pattern (`sourcedata`); `[MIXING]` fraction 0 means 1.
- Conversions (`input1.c`, `adjustdata`, `convertunits`), after the whole section is read: bulk and tank /86400; wall order 1 `x L /86400` (m/s), order 0 `/ L**2 /86400` (mass/m²/s), `L = M_PER_FT` in a US file else 1; the correlation factor likewise; diffusivity > 1e-4 a multiplier of `DIFFUSIVITY_M2`, else actual (US ft²/s x `M_PER_FT**2`); MASS /60; the rest in quality units.
- Refused by name: `ORDER WALL` not 0 or 1 (213); `GLOBAL TANK` (213); TRACE without a node or with an unknown one (212); a negative initial quality (209); an unknown mixing model or source type; an unknown node, link or pattern (the reader's rule; EPANET 203/205). Ignored as EPANET does, and recorded: `[MIXING]` on a junction or reservoir, `TANK` on a junction, `BULK`/`WALL` on a pump or valve.

- [ ] **Step 1: Write the failing tests** in `test_inp_quality.py` (wntr's files via `wntr.__file__`; inline files in `tmp_path`):
  - `test_net1_reads_chlorine_reactions_and_initial_quality` (committed Net1): `("CHEMICAL", "Chlorine", "mg/L")`; `global_bulk == -0.5/86400`; `global_wall == -0.3048/86400`; `initial["10"] == 0.5`, `initial["9"] == initial["2"] == 1.0`; `quality_timestep == 300`; `diffusivity == DIFFUSIVITY_M2`.
  - `test_net2_net3_ky10_modes`: Net2 `sources == (QualitySource("1", "CONCEN", 1.0, "3"),)`; Net3 TRACE `trace_node == "Lake"`, `units == "%"`; ky10 (`AGE mg/L`) AGE, `units == "hrs"`.
  - `test_the_chemical_keyword_keeps_epanets_default_units`: `QUALITY CHEMICAL ug/L` gives `("Chemical", "mg/L")`; `QUALITY chem` is CHEMICAL.
  - `test_range_forms_are_numeric_or_lexicographic`: nodes `10 11 12 110`, `10 12 0.5` sets the first three; nodes `A AB B BA`, `A B 0.3` sets `A AB B`; `10x 12 0.5` is numeric; the same for `BULK` on links and `TANK` on tanks.
  - `test_legacy_and_mass_sources`: `J1 2.5 PAT` is `("CONCEN", 2.5, "PAT")`; `J1 2.5` no pattern; `J1 MASS 6 *` is `("MASS", 0.1, None)`.
  - `test_wall_units_follow_the_order_and_the_unit_system`: GPM `GLOBAL WALL 2` order 0 gives `2/86400/0.3048**2`, order 1 `2*0.3048/86400`; LPS both `2/86400`; `ORDER WALL 0` after the coefficient gives the same.
  - `test_diffusivity_multiplier_actual_and_zero`: `1.0` gives `DIFFUSIVITY_M2`; `2e-9` (LPS) `2e-9`; `1e-5` (GPM) `1e-5*0.3048**2`; `0` gives `0.0`.
  - `test_later_lines_mixing_fractions_and_epanet_orders`: two `GLOBAL BULK` lines, the second wins; `T1 2COMP 0` gives `("2COMP", 1.0)`; with `[TANKS]` before `[RESERVOIRS]` and `[VALVES]` before `[PIPES]`, `node_order` is junctions, tank, reservoir and `link_order` starts with the valve.
  - `test_refusals_name_the_feature` (parametrised, `match=`): `ORDER WALL 2`, `GLOBAL TANK -1`, `QUALITY TRACE`, `QUALITY TRACE NOPE`, `[QUALITY] J1 -1`, `[MIXING] T1 PLUG`, `[QUALITY] NOPE 1`, `BULK NOPE -1`, `[SOURCES] J1 CONCEN 1 NOPAT`.
  - `test_lines_epanet_ignores_are_recorded`: `[MIXING] J1 FIFO`, `TANK J1 -1`, `BULK PU1 -1` change nothing and each is in `notes["ignored_quality"]`.
  - In `test_inp.py`, the two assertions that `Quality`/`Quality Timestep 0:05` are in the unrecognised notes now assert they are not.
- [ ] **Step 2:** Run `pytest tests/apps/water/test_inp_quality.py tests/apps/water/test_inp.py -v`; expected: FAIL (`quality_data` missing).
- [ ] **Step 3:** Implement; `WaterNetwork.check()` also checks a hand-built `WaterQuality`'s names, mode and wall order by name.
- [ ] **Step 4:** `docs/formats/epanet.md` (sections and options read, ignored lines, refusals); export `WaterQuality`, `QualitySource`; regenerate the catalogue. Run the water subset; expected: PASS, the load-coverage test unchanged.
- [ ] **Step 5:** Commit: `feat(water): read every EPANET quality section into WaterQuality`.

### Task 2: The EPANET quality reference harness

Test infrastructure first: every later parity test calls it.

**Files:**
- Create: `tests/verification/_epanet_quality.py`, `tests/verification/test_water_quality.py`, `tests/data/water/quality_base.inp`
- Modify: `AGENTS.md`, `.github/scripts/select_tests.py` (`tests/verification/test_water_quality*.py` in the `water` row, both)

**Interfaces (produced):**

```python
def edited(source: Path, tmp_path: Path, name: str, *, add: Mapping[str, Sequence[str]] = {},
           replace: Sequence[tuple[str, str]] = ()) -> Path
    # noodl's input: literal replacements (each must match), lines appended to named sections
    # (a missing section is created before [END])
def epanet_copy(path: Path) -> Path   # + ACCURACY 1e-8 (clamped to 1e-5; _open sets 1e-7), TRIALS 400, [REPORT] STATUS YES, before [END]
@dataclass
class QualityReference:
    times: list[int]                  # every EN_runQ time
    node_names: list[str]; link_names: list[str]
    node: Tensor; link: Tensor        # (T, n) EN_QUALITY, EN_LINKQUAL, quality units
    ratio: float                      # EN_getstatistic(EN_MASSBALANCE) at the end
    terms: dict[str, float]           # "initial" "inflow" "outflow" "reacted" "final" "ratio", .rpt
def epanet_quality(path: Path) -> QualityReference
def run_quality(path: Path, **water_run_kwargs) -> tuple[WaterResults, dict]
    # noodl on `path`: read, build, water_run(quality=True, diagnostics=d)
def assert_matches_epanet(diagnostics: dict, reference: QualityReference, *, ctol: float,
                          admitted: Sequence[tuple[int, str]] = (), instance: int = 0) -> dict
    # the bounds at the common times; returns {"p999", "worst", "scale", "first"}
def first_divergence(times, ours, theirs, names, *, limit: float) -> tuple[int, str, float] | None
```

- Toolkit (`epanet.c`, `EN_solveQ`): `ENopen`, `ENsolveH`, `ENopenQ`, `ENinitQ(0)`; repeat `t = ENrunQ()`, read every node's `EN_QUALITY` and link's `EN_LINKQUAL`, `ENnextQ()` until it returns 0; `en.ENlib.EN_getstatistic(en._project, 4, ctypes.byref(ctypes.c_double()))`; `ENcloseQ`, `ENclose`; parse the `.rpt` from "Initial Mass:" to "Mass Ratio:".
- `scale = max(|reference|, ctol)`; a point beyond `1e-5 * scale` must be in `admitted`; failures name `first_divergence`. noodl's side: `diagnostics["quality"][instance]` `"times"`, `"node_quality"`, `"link_quality"` (Task 3).
- `quality_base.inp`: LPS, H-W, `QUALITY CHEMICAL mg/L`, no reactions; DURATION 24:00, HYDRAULIC 1:00, QUALITY 0:05, PATTERN 1:00, REPORT 1:00. `R1` (reservoir, `[QUALITY] R1 1.0`); `P1` `R1`–`J1`; loop `P2` `J1`–`J2`, `P3` `J2`–`J3`, `P4` `J3`–`J1`; `P5` `J3`–`J4`; `P6` `J4`–`T1` (cylindrical, MIXED, mid-range start); dead end `P7` `J2`–`J5` (demand 0); TCV `V1` `J1`→`J6`; pump `PU1` `J4`→`J7` (one-point curve); a pattern on `J4` draining `T1` over the day's middle. Names fixed (later tasks edit by them); sizes the implementer's, to the properties tested.

- [ ] **Step 1: Write the failing tests** in `test_water_quality.py`: `test_the_step_harness_reproduces_epanets_binary_output` (Net1: at the report times `node` equals EPANET's own `ENsolveQ` binary output, read with `wntr.epanet.io.BinFile`, within 1e-6 relative); `test_the_mass_balance_is_read_in_full_and_from_the_report` (Net1: `terms["ratio"]` within 5e-6 of `ratio`; five terms; `initial > 0`); `test_the_bounds_name_the_first_divergence` (synthetic data 2e-3 apart at 7200 s, node `J3`: the `AssertionError` names both; 1e-9 apart passes); `test_the_quality_base_network_has_its_properties` (EPANET: `P6` changes sign at least twice; `T1` rises and falls inside its limits; `P7` carries 0 at every step, every other link more than `Q_STAGNANT`); `test_the_quality_base_hydraulics_match_epanet` (`water_run`: heads 1e-5 relative, flows 1e-6 m³/s, level 2e-4 m).
- [ ] **Step 2:** Run `pytest tests/verification/test_water_quality.py -v`; expected: FAIL (`_epanet_quality` missing).
- [ ] **Step 3:** Implement the harness and the base network; add the glob to `AGENTS.md` and `select_tests.py`.
- [ ] **Step 4:** Run the same; expected: PASS.
- [ ] **Step 5:** Commit: `test(water): EPANET quality reference harness and the per-feature base network`.

### Task 3: The router: segments, sort and transport

**Ordering, decided:** the router is tested through `water_run` from here on: only `water_run` produces EPANET's hydraulic step sequence (pattern, report, tank-limit and control steps), and a hand-written loop would test another procedure. `quality=None` stays OFF until Task 8, so no existing hydraulic test meets a feature not yet built; Tasks 3–7 pass `quality=True`.

**Files:**
- Create: `src/noodl/apps/water/quality.py`
- Modify: `src/noodl/apps/water/simulate.py` (`water_run`, `WaterResults`, `_Recorder`), `__init__.py`, `docs/catalogue/water.md`
- Test: `tests/apps/water/test_quality.py` (new), `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: `WaterQuality` (Task 1), the harness (Task 2).
- Produces, in `quality.py`:

```python
class SegmentPool:   # mempool/FreeSeg/addseg/reversesegs: c, v (capacity doubles), prev (-1 NULL),
                     # first/last per chain (-1 empty), owner per segment, a free list
    def __init__(self, n_chains: int, *, attached: bool) -> None
    def add(self, chain: int, v, c) -> None          # addseg, upstream of LastSeg
    def reverse(self, chain: int) -> None
    def segments(self, chain: int) -> list[int]      # FirstSeg (leading) -> LastSeg
    def counts(self) -> tuple[int, ...]
def quality_step(net, hydraulic_step: int) -> int    # adjustdata: explicit or step // 10, <= step; 0 refused by name
def sort_nodes(n_nodes: int, ends: Sequence[tuple[int, int]], direction: Sequence[int]
               ) -> tuple[list[int], int]            # sortnodes + selectnonstacknode: (order, cycle breaks)
class QualityRouter:
    def __init__(self, model, quality: WaterQuality, *, inputs: Mapping[str, Tensor],
                 quality_step: int, attached: bool, diagnostics: dict | None = None) -> None
    def node_quality(self) -> Tensor                 # (n_nodes,), model.node_names
    def link_quality(self) -> Tensor                 # (n_links,), avgqual, model.link_names
    def advance(self, t: int, step: int, *, flow: Tensor, closed: Tensor, demand: Tensor) -> None
        # runqual at t (ratecoeffs, flowdirchanged, sortnodes), then nextqual over [t, t+step]
```

- `inputs["quality.initial"]` (n_nodes, quality units) from `state`, else `WaterQuality.initial` (0 where absent). `flow` (n_links) m³/s; `closed` (n_links) `link_states_from_cut_off(...) == CLOSED` (EPANET's `LINKFLOW` zero); `demand` (n_junctions) delivered demand plus emitter flow (`hydsolver.c`: `NodeDemand`).
- Chains: link `k` of `model.link_names` is chain `k`, tank `i` of `net.tanks` chain `n_links + i`. `node_order`/`link_order` (else the builder's order) give the adjacency and the sort only.
- Transcribed: `initqual` (TRACE all 0, then the trace node 100); `initsegs` (pipes one segment of `0.785398 L D²` m³ at the quality of the link's end node `v` after `initqual`; none for pumps, valves, CV pipes; a MIXED tank one at `V0 = Vmin + A (init - min)`, `Vmin` the MinVol when positive else `A min`, or the volume curve (`inittanks`), at the file's initial quality even in TRACE); `flowdirchanged`; `sortnodes`/`selectnonstacknode`; `transport`, `evalnodeinflow`, `evalnodeoutflow` (merge at `Ctol`), `findnodequal` (junction `volin -= min(0, demand) dt`; no inflow keeps its quality while nothing reacts; reservoir keeps its previous quality; tank `tankmix1`; TRACE at the trace node: reservoir source 100, junction and tank `max(100 - c, 0)`, quality 100); `avgqual`. The quality step loop is `nextqual`'s: `dt = min(Qstep, step - qtime)`.
- Ruling Q5: decisions on `float()` of detached values. `attached` per run: grad enabled and (a quality input or the first solve's `water.q` requires grad); detached runs under `torch.no_grad()`. The node loop works on scalars: detached via one `tolist()` and one `torch.tensor` write-back per quality step; attached via a dict of 0-d tensors written back by one out-of-place `index_put` per step.
- `diagnostics["quality"]`: a list, one dict per instance: `"times"`, `"node_quality"`, `"link_quality"` (per solve, before transport); `"node_order"` (`(t, names)` per sort); `"cycle_breaks"`; `"reversals"` (`(t, link)`); `"segments"` (`counts()` per quality step); `"inflow_volume"`, `"inflow_mass"`, `"outflow_volume"` (n_nodes, per quality step).
- `simulate.py`: `water_run(model, state=None, drivers=None, *, duration=None, quality: bool | None = None, diagnostics=None, **solve_kwargs)` (here `True` runs, `None`/`False` do not); `WaterResults` gains `node_quality`, `link_quality` (`Tensor | None = None`, (..., T, n)), `quality_units: str | None = None`; `_Recorder.demands(solved, drivers) -> tuple[Tensor, Tensor]` shared by `record` and the router; after the solve at `t`: record with `router.node_quality()`, `router.advance(t, step, ...)`, `t += step`. `quality.py` imports `pattern_factor` lazily (`simulate` imports `quality`).
- Temporary refusals (`ValueError` naming the feature, "not yet modelled"), each removed by its task: AGE or a nonzero reaction coefficient (Task 4), `[SOURCES]` (Task 5), a mixing model other than MIXED (Task 6), a batched run (Task 8).

- [ ] **Step 1: Write the failing tests.** Unit (`test_quality.py`): `test_the_sort_follows_epanets_stack_and_adjacency_order` (nodes `A=1 B=2 C=3 R=4`, links `L1 R->A`, `L2 A->B`, `L3 A->C`: `[R, A, B, C]`; `L2`, `L3` swapped: `[R, A, C, B]`); `test_a_cycle_is_broken_by_selectnonstacknode` (`L1 R->A`, `L2 A->B`, `L3 B->C`, `L4 C->A`: `[R, A, B, C]`, one break); `test_initial_segments` (base: one segment per pipe at `0.785398 L D²` with its end node's quality; none in `V1`, `PU1`, nor in `P2` made CV; `T1` at `V0`; TRACE `J1`: pipes ending at `J1` start at 100); `test_segments_reverse_only_on_a_direct_sign_change` (`P6` at `+q`, `|q| < Q_STAGNANT`, `-q`: no entry in `"reversals"`; `+q`, `-q`: one); `test_a_quality_step_of_zero_seconds_is_refused` (hydraulic step 5 s: `match="QUALITY TIMESTEP"`); `test_an_attached_run_equals_a_detached_run` (base 3 h, `quality.initial` requiring grad or not: `torch.equal`); `test_features_of_later_tasks_are_refused_by_name`.
  - Parity (`test_water_quality.py`; each `run_quality` + `epanet_quality` + `assert_matches_epanet` on `edited(quality_base.inp, ...)`, and registered in a module-level `QUALITY_CASES: dict[str, Callable[[Path], Path]]` for Task 7): `test_trace_matches_epanet[junction|reservoir|tank]` (`J1`, `R1`, `T1`); `test_chemical_transport_matches_epanet`; `test_a_reversing_pipe_matches_epanet` (`P6`); `test_a_pump_and_a_valve_in_the_path_match_epanet`; `test_a_dead_end_below_q_stagnant_matches_epanet` (`J5`); `test_a_step_shorter_than_the_quality_step_at_a_tank_limit_matches_epanet` (T1's maximum lowered; `"times"` holds a time not a multiple of 300 s).
  - Review Focus 1, `test_a_check_valve_pipe_has_no_volume_like_epanet`: `P5` made `CV`: its chain is empty after every quality step (`"segments"`), and it matches EPANET. Review Focus 2, `test_a_pipe_through_stagnant_to_reversed_matches_epanet`: time controls close `P6` for the hour in which `T1` turns from filling to draining: no reversal of `P6` in `"reversals"`, and it matches EPANET. Review Focus 3, `test_a_directed_flow_cycle_matches_epanet`: a pump `J3`→`J1` strong enough that water circulates `J1 -> J2 -> J3 -> J1`: a positive `"cycle_breaks"` entry, and it matches EPANET.
- [ ] **Step 2:** Run `pytest tests/apps/water/test_quality.py tests/verification/test_water_quality.py -v`; expected: FAIL (`quality` missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them; expected: PASS (a point beyond `1e-5 * scale` is debugged to its cause, `superpowers:systematic-debugging`, before it is admitted). Then the water subset and the golden check; regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(water): EPANET's quality transport, sort and reversal in water_run`.

### Task 4: Reactions and water age

**Files:**
- Create: `src/noodl/apps/water/quality_react.py`
- Modify: `quality.py`, `network.py` (`model.input_specs` for the three keys, `required=False`)
- Test: `tests/apps/water/test_quality.py`, `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: `SegmentPool`, `QualityRouter` (Task 3).
- Produces, in `quality_react.py`:

```python
def pipe_coefficients(net, quality) -> tuple[Tensor, Tensor]   # (kb, kw) per pipe, SI per s (adjustdata)
def tank_coefficients(net, quality) -> Tensor                  # own TANK value, else global BULK
def bulk_rate(c: Tensor, kb: Tensor, order: float, limit: float, tiny: float) -> Tensor        # bulkrate
def wall_coefficients(flow: Tensor, diameter: Tensor, length: Tensor, kw: Tensor, *, order: float,
                      viscosity: float, diffusivity: float) -> Tensor                         # piperate: Rc
def wall_rate(c: Tensor, diameter: Tensor, kw: Tensor, rc: Tensor, *, order: float) -> Tensor  # wallrate
```

- `pipe_coefficients`: own value, else global; a pipe without its own WALL under ROUGHNESS CORRELATION `F`: H-W `F/C`, C-M `F n`, D-W `F/|ln(e/d)|` with `e/d` in the FILE's units (`adjustdata` precedes `convertunits`: mm/mm, or millifeet/inches, i.e. `e/d * 1000 * 0.0254 / 0.3048`, for a US `net.options.flow_units`); 0 when `C`, `n` or `d` is not positive.
- `bulk_rate`'s Michaelis–Menten floor is EPANET's `TINY = 1e-6` internal mass/ft³, `tiny = 1e-6 / 28.317` in quality units. `piperate`: `Sc = viscosity / diffusivity` (0 when diffusivity is 0); viscosity `net.options.viscosity * NU_WATER_FT2 * M_PER_FT**2`; exponents `0.88`, `0.333`, `0.667`; `Sc == 0`: order 0 `BIG` (1e10 ft/s = 3.048e9 m/s), order 1 `kw (4/d)`. Order-0 `wallrate` takes the smaller magnitude of `sgn(kw) c kf` and `kw / LITRES_PER_M3`, times `4/d`. `flow` is zero on closed links.
- Router: `reactpipes`/`reacttanks` as one vectorised operation over the live segments before `transport`, coefficients gathered by `owner` (pumps, valves, CV pipes rate 0); `max(0, .)` Euler; AGE adds `dt/3600` to every pipe and tank segment; `ratecoeffs` once per `advance` (not AGE); `Reactflag` (`setreactflag`) from the effective coefficients; `noflowqual` when `Reactflag`; `reacted` kept for Task 7.
- `inputs` gains `"quality.bulk_coefficient"`, `"quality.wall_coefficient"` (n_pipes) and `"quality.tank_coefficient"` (n_tanks) from `drivers`, else the two functions. Remove the reaction and AGE refusals.

- [ ] **Step 1: Write the failing tests.** Unit: `test_bulk_rate_by_hand` (order 1 `kb c`, 0 `kb`, 2 `kb c²`, 1.5 `kb c**1.5`; `kb=0.5, limit=2`: `0.75` at `c=0.5`, `0` at `c=3`; `kb=-0.5, limit=0.2`: `-0.4` at `c=1`, `0` at `c=0.1`; order -1 `kb=-1, limit=2, c=1`: `-1`; `|limit + sgn c| < tiny` takes `±tiny`); `test_wall_coefficients_by_hand` (Re < 1 `Sh = 2`, Re 1000 Graetz, Re 1e4 Notter–Sleicher, each against the formula; diffusivity 0: order 0 `3.048e9`, order 1 `4 kw/d`); `test_wall_rate_is_mass_transfer_limited` (order 0: large `kw` gives `sgn(kw) c kf 4/d`, small `kw/1000 * 4/d`); `test_pipe_coefficients_with_a_roughness_correlation` (H-W, D-W SI, D-W GPM with the `83.33` factor, C-M; own WALL wins; `F = 0` takes the global).
  - Parity (each in `QUALITY_CASES`): `test_bulk_reaction_matches_epanet[order0|order1|order2|michaelis_menten|limit_growth|limit_decay]`; `test_wall_reaction_matches_epanet[order0_si|order1|order1_mass_transfer|diffusivity_zero]` (base) and `[order0_us]` (committed Net1, 6 h, `ORDER WALL 0`, `GLOBAL WALL -0.05`); `test_roughness_correlation_matches_epanet[hazen_williams|darcy_weisbach_us]` (base; Net1 with `test_water_parity._DW_NET1`); `test_tank_reaction_matches_epanet` (`TANK T1 -2`, `ORDER TANK 2`); `test_water_age_matches_epanet`; `test_a_reactive_dead_end_junction_matches_epanet` (`J5`); Review Focus 1, `test_a_check_valve_pipe_does_not_react_like_epanet` (`P5` CV under bulk and wall decay).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, then the water subset; expected: PASS. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(water): EPANET's bulk, wall and tank reactions and water age`.

### Task 5: Sources

**Files:**
- Modify: `quality.py`, `network.py` (`input_specs`)
- Test: `tests/apps/water/test_quality.py`, `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: `QualityRouter` (Task 3), `QualitySource` (Task 1).
- Produces: `findsourcequal`/`sourcequal` (`quality.c`) inside `findnodequal`: CHEMICAL only; nothing when the base strength is 0 or `volout/dt <= Q_STAGNANT`; strength x `pattern_factor(net, pattern, t)` at the step's start (EPANET's `Qtime`); CONCEN at a junction only with negative demand, `-c demand dt / volout`; MASS `c dt / (volout LITRES_PER_M3)`; SETPOINT `max(c - quality, 0)`; FLOWPACED `c`; added to a junction's quality, to a tank's outflow only, replacing a reservoir's (which keeps the previous one when 0 is added). `inputs` gains `"quality.source_strength"` (n_nodes; read at source nodes). Remove the sources refusal.

- [ ] **Step 1: Write the failing tests.** Unit: `test_no_source_below_q_stagnant` (a MASS source at the dead end `J5`: `node_quality` equal, `torch.equal`, to the run without it). Parity (each in `QUALITY_CASES`): `test_source_matches_epanet[concen_reservoir|mass|setpoint|flowpaced]-[constant|pattern]` (CONCEN at `R1`, the rest at `J1`; a pattern with zeros and twos); `test_a_reservoir_source_whose_pattern_goes_to_zero_keeps_its_last_value` (`R1` during the zero hours equals the hour before); `test_setpoint_below_the_node_quality_adds_nothing` (SETPOINT 0.5 at `J1` under a 1.0 supply: `J1` equals the run without it); `test_concen_at_a_junction_needs_a_negative_demand` (`J6` at `+q` adds nothing; at `-q` it adds).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, then the water subset; expected: PASS.
- [ ] **Step 5:** Commit: `feat(water): EPANET's quality sources`.

### Task 6: Tank mixing models

**Files:**
- Create: `src/noodl/apps/water/quality_tanks.py` (`tankmix1` moves here)
- Modify: `quality.py`
- Test: `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: `SegmentPool` (Task 3).
- Produces: `mix_tank(pool, chain: int, model: str, vmz: float, ctol: float, vin, win, vnet) -> scalar` (`mixtank`, `tankmix1`..`tankmix4` with every branch: 2COMP's zones per the facts, `stagzone->v = 0` when `vt == 0`; FIFO's and LIFO's last segment giving whatever is asked, `if (seg == LastSeg) vseg = vout`, and kept when used up; LIFO's withdrawal between two `reversesegs`). 2COMP `initsegs`: one segment at `V0` set to `max(0, V0 - V1max)`, then one of the rest; `V1max = fraction x Vmax`, `Vmax = Vmin + A (max - min)` or the curve at the maximum. Remove the mixing refusal.

- [ ] **Step 1: Write the failing tests** (each in `QUALITY_CASES`): `test_mixing_model_matches_epanet[mixed|2comp|fifo|lifo]` (2COMP fraction 0.4); `test_a_2comp_tank_crossing_v1max_matches_epanet` (noodl's `T1` volume crosses `V1max` both ways); `test_fifo_and_lifo_tanks_emptying_through_several_segments_match_epanet` (`T1`'s chain holds at least three segments and empties through them, `"segments"`); `test_a_2comp_tank_without_net_flow_matches_epanet` (time controls close `P6` for 2 h while the stagnant zone holds water; EPANET's `P6` flow is 0 then).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, then the water subset; expected: PASS.
- [ ] **Step 5:** Commit: `feat(water): EPANET's tank mixing models`.

### Task 7: Mass balance

**Ordering, decided:** before the integration: the balance accumulates inside `transport` and the reactions, and Task 8's last-step rule (Review Focus 4) shows only in the final mass.

**Files:**
- Modify: `quality.py`, `simulate.py`, `docs/catalogue/water.md`
- Test: `tests/apps/water/test_quality.py`, `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: the router (Tasks 3–6); `QualityReference.ratio`, `.terms`.
- Produces: `@dataclass class QualityMassBalance: initial, inflow, outflow, reacted, final, ratio: Tensor` ((...,) each); `QualityRouter.mass_balance() -> QualityMassBalance`; `WaterResults.mass_balance: QualityMassBalance | None = None`. Transcribed: initial `findstoredmass` after `initsegs`; `updatemassbalance` (junction lost `max(0, demand) dt c`, added `source x volout`; reservoir lost `massin`, added `source x volout` when positive, else `quality x volout`; tank added `source x volout`); `reacted += (c_old - c_new) v`; `evalmassbalance` after every `advance` (ratio 1 when the inflow side is 0). Mass is `c v LITRES_PER_M3` for CHEMICAL (the chemical's mass unit), `c v` (quality x m³) for AGE and TRACE.

- [ ] **Step 1: Write the failing tests.** `test_mass_balance_units` (one pipe of 1 m³ at 1 mg/L: `initial == 1000.0`; TRACE: quality x m³). `test_mass_balance_matches_epanet[case]` over `QUALITY_CASES`: ratio within 1e-6 of `reference.ratio`; each term within 2e-5 relative of `reference.terms` after EPANET's units are converted (CHEMICAL: divided by `28.317 / 0.3048**3 / 1000`; AGE, TRACE: times `0.3048**3`); terms 0 in both equal; the 2COMP-without-flow case's ratio below 1 in both.
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, then the water subset; expected: PASS. Regenerate the catalogue.
- [ ] **Step 5:** Commit: `feat(water): EPANET's water-quality mass balance`.

### Task 8: `water_run` integration

**Files:**
- Modify: `simulate.py`; `benchmarks/water_eps.py` and `tests/verification/test_water_networks.py::test_network_matches_epanet` (pass `quality=False`: they measure hydraulics only); `docs/applications/water.md` (`quality=` in Running over time; the steady tracer named so, its verification row "steady tracer, single source")
- Test: `tests/apps/water/test_simulate.py`, `tests/verification/test_water_quality.py`

**Interfaces:**
- Consumes: Tasks 3–7.
- Produces: `quality=None` runs when `net.quality` has a mode other than NONE and the run's duration is positive (`adjustdata`); `False` never; `True` raises `ValueError` naming `[OPTIONS] QUALITY` without a mode and `DURATION 0` at a zero duration. A model with the steady layer `"quality"` (`build_model(quality=float)`) is refused by name ("steady tracer") when EPANET quality resolves on.
- Produces: the last-step rule (`nextqual`): transport over `[t, t + step]` only when `t + whole <= duration`, `whole` EPANET's unshortened step (`plan(False, ...)` when `t + clock.free_step(t, hydraulic_report) > clock.duration`, shared with the energy's; else `step`).
- Produces: batches: one router per instance of `torch.broadcast_shapes` of the flows' and the quality inputs' leading shapes, fed its slices; results stacked (..., T, n); one `diagnostics["quality"]` entry per instance. Remove the batch refusal.

- [ ] **Step 1: Write the failing tests.** `test_simulate.py`: `test_quality_runs_by_default_when_the_network_defines_it` (Net1: `node_quality` set, `quality_units == "mg/L"`; `twoloop_si.inp`: `None`; Net1 `duration=0`: `None`); `test_quality_false_turns_it_off_and_true_demands_it` (`match="QUALITY"`, `match="DURATION"`); `test_the_steady_tracer_is_refused_alongside_epanet_quality` (`twoloop_trace.inp`, `build_model(quality=0.0)`: `match="steady tracer"`; `quality=False` runs); `test_quality_leaves_the_hydraulics_unchanged` (Net1 24 h: `head`, `flow`, `tank_level`, `status`, the energy fields `torch.equal` on and off).
  - `test_water_quality.py`: Review Focus 4, `test_a_duration_past_the_last_whole_step_matches_epanet` (base, `DURATION 5:30`: `"segments"` holds exactly 60 quality steps, none after 5:00; qualities and mass balance per the bounds and Task 7's). Review Focus 5, `test_a_batch_runs_each_instance_as_its_own_run` (base with bulk decay: `quality.bulk_coefficient` batched over two values equals each single run, `torch.equal`; `water.sources` batched over 0.8 and 1.2 times the demands equals each single run within `1e-8 * scale`, the base having no tank-limit or control event; two `diagnostics` entries).
- [ ] **Step 2:** Run them; expected: FAIL.
- [ ] **Step 3:** Implement, with the docs and the two callers under Files.
- [ ] **Step 4:** The water subset, `-m slow` on `test_water_networks.py`, the golden check; expected: PASS.
- [ ] **Step 5:** Commit: `feat(water): water_run runs EPANET quality when the network defines it`.

### Task 9: EPANET's example networks

**Files:**
- Test: `tests/verification/test_water_quality_networks.py` (new, under the Task 2 glob)

- [ ] **Step 1: Write the tests** (Net2, Net3, Net6 `slow`; files as shipped): `test_net1_quality_matches_epanet` (committed Net1, 24 h, nodes, links, mass balance); `test_net2_hydraulics_match_epanet` (55 h: heads 1e-5 relative, levels 2e-4 m, flows 1e-6 m³/s; never measured, it gates the next) and `test_net2_quality_matches_epanet` (CONCEN with its pattern at junction 1, negative demand); `test_net3_quality_matches_epanet` (TRACE `Lake`, 168 h; `"reversals"` and a re-sort after t = 0 present); `test_net6_quality_is_a_smoke_case` (shipped zeros: runs, all 0, ratio within 1e-3 of 1); `test_net6_with_an_added_source_matches_epanet` (`[SOURCES] RESERVOIR-3323 CONCEN 1.0`, 96 h; the bounds, loosened only with the measured cause, Net6's hydraulic parity being levels 3e-3 m and flows 5e-4 m³/s at its switching times, `test_water_networks._NETWORKS["Net6"]`).
- [ ] **Step 2:** Run `-m slow`; a miss is debugged to its cause before any bound moves; measured numbers and cause beside each bound.
- [ ] **Step 3:** Commit: `test(water): Net1, Net2, Net3 and Net6 water quality against EPANET`.

### Task 10: ky4 and ky10 extended-period hydraulics (gate)

ky4 (TRACE) and ky10 (AGE) ship with `DURATION 0`; their quality needs a 24 h run whose hydraulics nobody has compared with EPANET.

**Files:**
- Test: `tests/verification/test_water_networks.py`

- [ ] **Step 1: Write the test** `test_ky_extended_period_hydraulics_match_epanet[ky4|ky10]` (slow): ` Duration           \t0` replaced by `24:00` (asserted to match), EPANET through the file's `_epanet` helper (`epanet_results`: the toolkit at a true ACCURACY 1e-7, Task 2b), `water_run(model, quality=False)`; heads 1e-5 relative, levels 2e-4 m, flows 1e-6 m³/s on every link (ky4's parallel-pair exemption was removed in Task 2b: 2.6e-7 at 1e-7), states equal, every report time.
- [ ] **Step 2:** Run `-m slow`. A miss is debugged; a measured cause (as Net6's closed-link leak) is written beside a loosened bound and carried into Task 11. **Stop condition:** no cause found by the debugging skill's process: record the measured values, mark the test `xfail(strict=True, reason=<values>)`, skip Task 11, tell the user; Task 13's docs list ky4/ky10 quality as unverified.
- [ ] **Step 3:** Commit: `test(water): ky4 and ky10 extended-period hydraulics against EPANET`.

### Task 11: ky4 and ky10 water quality (only after Task 10 passed or found its cause)

**Files:**
- Test: `tests/verification/test_water_quality_networks.py`

- [ ] **Step 1: Write the tests** (slow) on Task 10's edit: `test_ky10_age_matches_epanet`, `test_ky4_trace_matches_epanet` (`R-1`, QUALITY TIMESTEP = the hydraulic step); nodes, links, mass balance; the bounds or Task 10's cause.
- [ ] **Step 2:** Run `-m slow`; a miss is debugged to its cause before loosening.
- [ ] **Step 3:** Commit: `test(water): ky4 trace and ky10 age against EPANET`.

### Task 12: Gradients and structural identity

**Files:**
- Test: `tests/verification/test_water_quality.py`

- [ ] **Step 1: Write the tests.** `test_quality_gradients_against_central_differences[bulk|wall|source|initial|roughness]`: base, 6 h, bulk order 1 at `-0.5/day`, wall order 1, CONCEN at `R1` with a pattern, `water_run(..., differentiable=True)`; `d node_quality[-1, J4]` with respect to `quality.bulk_coefficient[P5]`, `quality.wall_coefficient[P5]`, `quality.source_strength[R1]`, `quality.initial[T1]` and `P3`'s roughness (`roughness.requires_grad_(True)`, as `test_water_parity.test_gradient_reaches_a_learnable_roughness`); central differences at `h = 1e-4 |p|` (`1e-4` at 0); `|ad - fd| <= 1e-5 max(|ad|, |fd|)`; the three runs' `"segments"` and `"node_order"` identical (Ruling Q5). `test_the_router_runs_detached_unless_asked` (nothing requiring grad: `node_quality.requires_grad` False; values `torch.equal` to the attached run's).
- [ ] **Step 2:** Run them; expected: PASS (the mechanism is Task 3's; a failure is a defect fixed in its task's code with a test).
- [ ] **Step 3:** Commit: `test(water): quality gradients against central differences`.

### Task 13: Documentation, benchmark, final checks

**Files:**
- Create: `benchmarks/water_quality.py`
- Modify: `docs/applications/water.md`, `docs/formats/epanet.md`, `docs/catalogue/water.md`

- [ ] **Step 1:** `water.md`: a water-quality section (modes, reactions, sources, mixing, running and `quality=`, the learnable keys with shapes and units, differentiability and its jumps, batches one instance at a time, the EPANET 2.2 behaviours transcribed as they are: CV pipes, stagnant-then-reversed links, the 2COMP zone without net flow, the last step past the duration); verification rows with the measured numbers of Tasks 3–12; limitations (the spec's out of scope; ky4/ky10 if Task 10 stopped). `epanet.md` final. Regenerate the catalogue.
- [ ] **Step 2:** `benchmarks/water_quality.py`: Net1 24 h detached, Net3 168 h detached, Net1 6 h attached (bulk coefficient requiring grad): wall time, quality steps, peak segments; no asserts.
- [ ] **Step 3:** `benchmarks/water_eps.py` three times: median within ~5 % of Task 0's, else fixed first; both in the progress file.
- [ ] **Step 4:** If #35 merged after Task 0: `git fetch origin; git merge origin/main`. Then `ruff check .`, the full `pytest` with coverage (≥ 90 %), `-m slow` water tests, the golden check, `python -m mkdocs build --strict`; expected: all pass.
- [ ] **Step 5:** Commit: `docs(water): EPANET water quality`.
