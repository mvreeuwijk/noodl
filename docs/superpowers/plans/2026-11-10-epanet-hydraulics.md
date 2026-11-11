# EPANET Hydraulics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `apps.water` reads, solves and runs over time every hydraulic feature of EPANET 2.2 and matches EPANET on Net3, Net6, ky4, ky10 and Anytown.

**Architecture:** Every device that changes state (check valve, closed link, PRV, PSV, pump at shut-off, link into a full or empty tank) is one exact equation in the single Newton solve, written as a complementarity condition with `torch.where` and explicit tie rules, as `BreakerValve` is today. Statuses and settings are drivers; controls and rules change them between solves; a new extended-period driver owns the clock, the step-length rule, patterns and reporting.

**Tech Stack:** Python 3.12, PyTorch (float64), pytest, `wntr` 1.5.0 (EPANET 2.2 reference, verification only).

**Spec:** `docs/superpowers/specs/2026-11-10-epanet-hydraulics-design.md`. The PRV/PSV design note `docs/plans/water-prv-psv-design.md` (§1–§4) is part of the spec.

## Global Constraints

- **Start point.** Branch `epanet-hydraulics` from `main` AFTER the core-unification M4–M8 PR has merged. Worktree: `C:/Users/mvr/AppData/Local/Temp/noodl-wt/epanet-hydraulics` (never a sibling of `C:\projects`). File names below are as on the integration branch `core-unification-m4-m8` (commit 2a0f1ea); line numbers are deliberately omitted.
- **Interpreter and tests.** `C:/projects/noodl/.venv/Scripts/python.exe` with `PYTHONPATH=src`, run from the worktree root. Subset per task: the AGENTS.md core list plus the `water` row, `-n 4 --no-cov`. Tests marked `slow` run with `-m slow` when a task changes what they compare.
- **Reference.** EPANET 2.2 through `wntr` 1.5.0 `EpanetSimulator` (`pytest.importorskip("wntr")`). EPANET source for transcriptions: tag `v2.2` of OpenWaterAnalytics/EPANET; cite file and function in the docstring of every transcribed formula.
- **Bounds (per device):** heads 1e-6 relative; flows 1e-6 m³/s absolute; a valve's controlled head equal to its setting to Newton tolerance and within 1e-6 of EPANET's; inside EPANET's `HTOL` band (1.524e-4 m) heads within `HTOL`.
- **Bounds (whole networks over time):** heads 1e-5 relative, tank levels 2e-4 m. Loosen only with a measured cause written in the test.
- **No loss of speed.** `benchmarks/water_eps.py` and the steady water benchmarks before (Task 0) and after (Task 13), same machine, same session; a slowdown beyond ~5 % on a network without new devices is a defect.
- **Bitwise neutrality of the core change.** `NOODL_GOLDEN_EXACT=1 pytest tests/test_golden_orchestration.py` passes after Task 1.
- **Names.** Physical names, no numerical-method jargon in public names. No aliases when renaming (`water_event_step` and the per-pump `water.status` are replaced, not kept).
- **Code, tests and docs change together.** A task that changes public names regenerates the catalogue (`python scripts/gen_catalogue.py`) and updates the pages that mention them.
- **Before each commit:** `ruff check .` clean. **At the end of each task:** `python -m mkdocs build --strict` passes.
- **Git.** Identity 58522531+mvreeuwijk@users.noreply.github.com. Never `git stash`. Never push. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No wall-clock assertions in tests.**

## Shared vocabulary (used by several tasks)

- `model.link_names`: every link in layer order (pipes, pumps, then valves by kind). Every per-link driver below is in this order.
- Driver `"water.status"`: per link, `CLOSED = 0`, `OPEN = 1`, `ACTIVE = 2` (`ACTIVE` only for PRV, PSV, PBV, FCV, TCV, GPV: the valve acts on its setting; `OPEN` on a valve means fully open, ignoring the setting). Constants live in `noodl.apps.water.controls`.
- Driver `"water.setting"`: per link; pump relative speed, PRV/PSV/PBV head setting (layer potential unit), FCV flow (m³/s), TCV loss coefficient; ignored for pipes and GPVs. A calibration passes a tensor that requires grad here.
- Driver `"water.one_way"`: per link, `0` both directions, `+1` forward flow only, `-1` reverse flow only. Set from `CV` pipes (always `+1`) and from full/empty tanks (Task 7).
- Row scale `kappa` (layer potential per m³/s) of every complementarity residual: the link's own head-loss slope at 1 ft/s, computed once at build. The solution does not depend on it.

## Review Focus

1. **A closed link that disconnects junctions with demand** (a `[STATUS] CLOSED` or a control isolating part of the network): expect a `ValueError` naming the disconnected junctions, not a singular-matrix traceback or a hang. Owned by Task 3.
2. **A device that makes the solve singular** (two PRVs in parallel with equal settings; a PRV whose controlled node is a reservoir): expect an error naming the valves, raised from the solve's singular diagnosis. Owned by Task 4.
3. **A tank already full at t = 0 with a pump filling it**: expect the pump's flow 0 from the first step and EPANET's state CLOSED. Owned by Task 7.
4. **A control and a rule acting on the same link in the same step**: expect EPANET's order (simple controls first, then rules; the rule's action stands). Owned by Task 9.
5. **A batch whose instances switch at different times**: expect each instance to equal its own single run except for the extra step boundaries, and never an instance stepping past its own event. Owned by Task 10.

---

### Task 0: Baseline

**Files:** none changed.

- [ ] **Step 1:** Create the worktree and branch per Global Constraints; confirm `main` contains the M4–M8 merge (`git log --oneline -1 main` names it).
- [ ] **Step 2:** Run the water subset; expected: all pass. Run `NOODL_GOLDEN_EXACT=1 pytest tests/test_golden_orchestration.py`; expected: pass.
- [ ] **Step 3:** Run `benchmarks/water_eps.py` and the steady water benchmarks three times; record the medians in `.superpowers/sdd/epanet-hydraulics/progress.md` (create it; not committed).

### Task 1: Implicit laws that read absolute heads

The PRV/PSV equations need the head at one end of the valve, not only the drop across it.

**Files:**
- Modify: `src/noodl/elements/base.py` (`LawContext`, `ConstitutiveLaw`)
- Modify: `src/noodl/layers/kirchhoff.py` (`_flows_and_residuals`, `_operator`, `dflows`, layer construction)
- Modify: `src/noodl/layers/_flow_unknowns.py` (the `src`/`sign` layout of the Jacobian values)
- Modify: `src/noodl/layers/_functional.py` (differentiable path)
- Test: `tests/layers/test_law_heads.py` (new)

**Interfaces:**
- Produces: `ConstitutiveLaw.reads_heads: bool = False` (class attribute; declared, not inferred). `LawContext.phi_from: Tensor | None = None`, `LawContext.phi_to: Tensor | None = None`: the potentials of each of the law's edges' start and end nodes, shaped like the law's `dp`, set only for a law with `reads_heads = True`.
- Produces: for such a law, `jacobian(dp, q, drivers, ctx)` returns `(dr_ddp, dr_dq, dr_dphi_from, dr_dphi_to)`, each the partial with the others held fixed; the default computes them by `_colour_blocks` over `(dp, q, ctx.phi_from, ctx.phi_to)`. The layer writes `dr_ddp + dr_dphi_from` in the start node's column and `-dr_ddp + dr_dphi_to` in the end node's, in the slots the pattern already has (no new sparsity). A boundary end node has no column; its partial is dropped, as for `dp` today.
- Produces: a layer refuses, at construction, a law with `reads_heads = True` that is not an `ImplicitLaw`, or that has `ports != 1` (`TypeError` naming the law).

- [ ] **Step 1: Write the failing tests** in `tests/layers/test_law_heads.py`. Define in the test file a law `HeldDownstreamHead(ImplicitLaw)` with `reads_heads = True` and `residual = ctx.phi_to - setting` (an ideal downstream-head source). Network: reservoir R (head 50) → pipe → J1 → law edge → J2 → pipe → J3 (demand 0.01 m³/s).
  - `test_the_end_head_is_held`: steady solve gives `phi[J2] == 40.0` to 1e-10 for `setting = 40.0`.
  - `test_jacobian_matches_autograd_of_the_residual`: the layer's assembled operator at a non-converged point equals `torch.autograd.functional.jacobian` of the layer's `_residual` in `x`, to 1e-12.
  - `test_gradient_of_the_held_head`: `d phi[J3] / d setting == 1.0` to 1e-8 through `solve(differentiable=True)`, and `d phi[J1] / d setting` matches central differences (step 1e-4) to 1e-6 relative.
  - `test_reads_heads_on_a_flow_law_is_refused`: a `ConstitutiveLaw` subclass with `reads_heads = True` raises `TypeError` mentioning `reads_heads`.
- [ ] **Step 2:** Run `pytest tests/layers/test_law_heads.py -v`; expected: FAIL (`LawContext` has no `phi_from`).
- [ ] **Step 3:** Implement. Build the per-law context with `phi_from`/`phi_to` only for laws that declare `reads_heads` (other laws keep receiving the cached context object), so layers without such laws run the same code path as before.
- [ ] **Step 4:** Run the new tests, then the core subset, then `NOODL_GOLDEN_EXACT=1 pytest tests/test_golden_orchestration.py`; expected: all PASS.
- [ ] **Step 5:** Commit: `feat(layers): implicit laws that read the heads at their end nodes`.

### Task 2: Network data and reader for every EPANET 2.2 hydraulic section

The reader and the data classes learn everything up front. `WaterNetwork.validate()` keeps refusing each feature by name until the task that builds it removes the refusal.

**Files:**
- Modify: `src/noodl/apps/water/network.py` (data classes, `validate`)
- Modify: `src/noodl/apps/water/inp.py`
- Create: `src/noodl/apps/water/controls.py` (data classes only in this task)
- Test: `tests/apps/water/test_inp.py`

**Interfaces (produced):**
- `WaterPipe.status` in `{"OPEN", "CLOSED", "CV"}` (read from the column, as today).
- `Pump(name, u, v, curve, power=None, speed=1.0, speed_pattern=None, efficiency_curve=None, price=None, price_pattern=None)`.
- `Tank(..., volume_curve: str | None = None, overflow: bool = False)`.
- `Valve.kind` may be `PRV`/`PSV`.
- `Emitter(node: str, coefficient: float)` (coefficient in m³/s per m^exponent, converted from the file's flow and pressure units); `WaterNetwork.emitters: tuple[Emitter, ...] = ()`; `WaterOptions.emitter_exponent: float = 0.5`.
- `WaterNetwork.initial_status: dict[str, str | float]` from `[STATUS]` (link → `"OPEN"`, `"CLOSED"`, or a numeric setting).
- In `controls.py`: `CLOSED, OPEN, ACTIVE = 0, 1, 2`; `Control(link: str, status: int | None, setting: float | None, trigger: str, node: str | None = None, above: bool | None = None, value: float = 0.0)` with `trigger` in `{"time", "clocktime", "node"}`, `value` in seconds for time triggers, metres for a node trigger (level for a tank, pressure for a junction, head for a reservoir, as EPANET); `Rule(name, premises, then_actions, else_actions, priority)`, `Premise(conjunction, object, id, attribute, relation, value)`, `Action(link, attribute, value)`.
- `WaterNetwork.controls: tuple[Control, ...]`, `rules: tuple[Rule, ...] = ()`, `energy: EnergyOptions` (`global_efficiency=75.0`, `global_price=0.0`, `global_pattern=None`, `demand_charge=0.0`).
- Times: `WaterNetwork.rule_timestep` (default `hydraulic_timestep / 10`), `pattern_start`, `report_start`, `start_clocktime` (seconds; `12 am`, `6:30 PM`, `18` all read).
- `headloss` may be `"C-M"`.
- **Curves by role.** Each `[CURVES]` entry is converted by the role that uses it: pump head and GPV head-loss (flow, length), efficiency (flow, percent), tank volume (length, volume). A curve used in two roles with different units is refused by name.

- [ ] **Step 1: Write failing tests** in `tests/apps/water/test_inp.py`, reading `wntr`'s `Net3.inp`, `Net6.inp`, `ky10.inp` and `Anytown.inp` (path from `wntr.__file__`):
  - `test_net3_reads_status_time_controls_and_clock`: `initial_status["10"] == "CLOSED"`; 14 controls with `trigger == "time"`; `start_clocktime == 0.0`; `duration == 168 * 3600`.
  - `test_net6_reads_prv_cv_and_power_pump`: two valves with `kind == "PRV"`; one pipe with `status == "CV"`; one pump with `power` set, converted to kW.
  - `test_anytown_reads_speed_patterns_and_efficiency_curves`: pumps `78`, `79`, `80` have `speed_pattern` and `efficiency_curve == "E1"`; `E1`'s y values are the file's percentages unchanged; `start_clocktime == 18 * 3600`.
  - `test_a_curve_used_as_head_and_volume_is_refused`: a small inline `.inp` raises `ValueError` naming the curve.
  - `test_rules_are_parsed`: an inline `[RULES]` block with `IF`/`AND`/`OR`/`THEN`/`ELSE`/`PRIORITY` round-trips into `Rule` objects with the expected premises and priority.
  - `test_emitters_are_converted`: an inline GPM file with `EMITTER 1.0` gives `coefficient` in m³/s per m^0.5 equal to `6.309019640343977e-05 / PSI_TO_M**0.5`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. `validate()` refuses, each by name and naming the task's feature: PRV/PSV, `CV`/`CLOSED` pipes and `[STATUS]`, constant-power pumps, multipoint curves, speed patterns, volume curves, overflow, emitters, `C-M`, rules, every control form except the existing tank-level one. (Tasks 3–9 each remove their own refusal.)
- [ ] **Step 4:** Run the water subset; expected PASS (existing behaviour unchanged for files that loaded before).
- [ ] **Step 5:** Commit: `feat(water): read every EPANET 2.2 hydraulic section`.

### Task 3: Link status, settings and check valves

**Files:**
- Modify: `src/noodl/apps/water/elements.py` (`FlowControlledPipe`, `MinorLoss`, `FixedFlow` use, `BreakerValve`, `GeneralPurposeValve`)
- Modify: `src/noodl/apps/water/network.py` (`build_model`, `initial_drivers`, `validate`)
- Test: `tests/apps/water/test_status.py` (new), `tests/verification/test_water_parity.py`

**Interfaces:**
- Consumes: `CLOSED/OPEN/ACTIVE`, `WaterPipe.status`, `initial_status` (Task 2).
- Produces: drivers `"water.status"`, `"water.setting"`, `"water.one_way"` per link (Shared vocabulary), filled at build from the network (`CV` → `one_way = +1`; `CLOSED` pipe and `[STATUS]` → status; valve settings → setting; valves start `ACTIVE`). The per-pump `"water.status"` of today is replaced.
- Produces: in `FlowControlledPipe.residual`, with `h = head_loss(q)`: closed `kappa * q`; both ways `dp - h`; forward only `max(-kappa q, dp - h)`; reverse only `min(-kappa q, dp - h)`. Ties select the open branch (`torch.where(a >= b, ...)` written so equality picks `dp - h`).
- Produces: every other law multiplies its flow by `status != CLOSED` (flow laws) or returns `kappa * q` when closed (implicit laws); a valve with status `OPEN` is its minor loss.
- Produces: `build_model` refuses `CV`/`CLOSED`/`[STATUS]` with `pipe_form="potential"` by name.
- Produces: `water_steady` raises `ValueError("water_steady: junction(s) [...] are cut off from every reservoir and tank by closed links")` before solving when a junction with nonzero demand has no open path to a fixed-head node (union-find over open links: `noodl._unionfind`).

- [ ] **Step 1: Write failing tests.** In `test_status.py`: `test_a_closed_pipe_carries_no_flow`; `test_a_check_valve_closes_against_reverse_head` (two reservoirs, CV pipe pointing from low to high: `q == 0`, heads equal the reservoirs); `test_a_check_valve_opens_forward`; `test_closing_a_link_that_cuts_off_a_junction_is_refused` (Review Focus 1: message names the junction). In `test_water_parity.py`: `test_check_valve_and_closed_pipe_match_epanet` on the two-loop fixture with edits (one pipe `CV` closing, one `CV` open, one `CLOSED` via `[STATUS]`), bounds per Global Constraints, and states compared with `results.link["status"]`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the Task 2 refusals for `CV`, `CLOSED` and `[STATUS]` (numeric `[STATUS]` settings on valves included).
- [ ] **Step 4:** Run the water subset and `-m slow` water tests; expected PASS; update any test that set the old per-pump `"water.status"`.
- [ ] **Step 5:** Commit: `feat(water): link status, settings and check valves in the solve`.

### Task 4: PRV and PSV

**Files:**
- Modify: `src/noodl/apps/water/elements.py`
- Modify: `src/noodl/apps/water/network.py`
- Test: `tests/apps/water/test_elements.py`, `tests/verification/test_water_parity.py`

**Interfaces:**
- Consumes: `reads_heads`/`ctx.phi_from`/`ctx.phi_to` (Task 1); status/setting drivers (Task 3).
- Produces: `PressureReducingValve(ImplicitLaw)` and `PressureSustainingValve(ImplicitLaw)`, both `reads_heads = True`, `__init__(self, elevation, minor, *, kappa, smoothing: float = 0.0, kind)` where `elevation` is the sensed node's (downstream for PRV, upstream for PSV) in the layer potential unit and `minor` the open-valve coefficient (`epanet_minor_coefficient`). Residual per design note §2: `r = min(kappa q, g)` with PRV `g = max(H_v - Hset, m q|q| - dp)`, PSV `g = max(Hset - H_u, m q|q| - dp)`, `Hset = elevation + setting`; written with `torch.where` and the note's tie rules; status `OPEN` → minor loss, `CLOSED` → `kappa q`. `smoothing > 0` (metres) replaces the exact `min` by Fischer–Burmeister `a + b - sqrt(a² + b² + 2 eps²)` with `eps = smoothing`.
- Produces: `build_model(..., valve_smoothing: float = 0.0)` passes `smoothing`; edge kinds `"prv"`, `"psv"`.
- Produces: `link_states(model, state, drivers) -> Tensor` in `report.py`: per link `CLOSED/OPEN/ACTIVE` read from which branch the converged residual selects (pipes and pumps: `CLOSED` or `OPEN`).

- [ ] **Step 1: Write failing tests.** Parity: one parametrised `test_prv_psv_match_epanet` over the 11 cases of design note §4 (networks and settings exactly as tabled there), comparing heads, flows, the controlled head and `link_states` vs `results.link["status"]` (wntr: 0 closed, 1 open, 2 active). `test_prv_inside_htol_band` (near-switch sweep; inside the band heads within `HTOL`). Gradients: `test_prv_gradients_against_central_differences` (setting, demands, roughness; `dH_J2/dsetting == 1` in the tree case; `d/dsetting == 0` when open), `test_psv_gradients_against_central_differences` (setting, demands, roughness on the PSV-active case), `test_prv_one_sided_gradients_jump` (±delta either side of the switch). `test_parallel_prvs_with_equal_settings_are_refused` (Review Focus 2: error names both valves). `test_smoothed_prv_converges_to_exact` (smoothing 1e-3 m: heads within 2e-3 m of exact).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the PRV/PSV refusal. Seed every PRV/PSV flow as `ACTIVE` at the pipes' 1 ft/s first iterate (`model.initial_flows`).
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): pressure reducing and sustaining valves`.

### Task 5: Pumps

**Files:**
- Modify: `src/noodl/apps/water/elements.py` (`PumpCurve`, `FlowControlledPump`, new `ConstantPowerPump`)
- Modify: `src/noodl/apps/water/network.py` (curve fitting, `_fit_three_points` replaced)
- Test: `tests/apps/water/test_elements.py`, `tests/verification/test_water_parity.py`

**Interfaces:**
- Produces: the curve choice of EPANET 2.2 (`input1.c`, `updatepumpparams`, `powercurve`): one point → power law with `n = 2` as today; three points with the first at zero flow → power law `h = A - B qᶜ` with `C` fitted; every other curve → piecewise linear between its points (`curvecoeff`), extended by its end segments.
- Produces: `PumpCurve(h0, r, n, *, ...)` takes per-pump `n` (tensor); `PiecewisePumpCurve(flows, heads, n_points, *, kind="pump")` with the same padded layout as `GeneralPurposeValve`, speed scaling `h(q; w) = w² h(q / w)`.
- Produces: `ConstantPowerPump(power_kw, *, kind="pump")`: head gain `8.814 * P_hp / q` in ft and cfs (`hydcoeffs.c`, `pumpcoeff`, constant horsepower), converted to SI, with EPANET's low-flow floor.
- Produces: every pump residual is `max(-kappa q, dp + gain(q))` (forward only; shut-off closes it) and `kappa q` when status `CLOSED` or speed 0. Speed is read from `"water.setting"`. `water_steady`'s `q_max` refusal is kept for power-law pumps and removed for the others (EPANET extrapolates their last segment).

- [ ] **Step 1: Write failing tests.** Parity on the two-loop network with a pump edit, one test per case: `test_three_point_pump_with_general_exponent` (Net3's own curve points), `test_multipoint_pump_curve` (Anytown's 5-point curve), `test_constant_power_pump` (Net6's power value), `test_pump_closes_beyond_shut_off` (reservoir head raised above shut-off: `q == 0`, state CLOSED), `test_pump_speed_setting` (speed 0.8 via `"water.setting"`). Gradient: `test_pump_speed_gradient_against_central_differences`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the refusals for power pumps and multipoint curves (speed patterns are applied by Task 10; remove that refusal there).
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): every EPANET pump curve, constant-power pumps and shut-off`.

### Task 6: Emitters and Chezy-Manning

**Files:**
- Modify: `src/noodl/apps/water/demand.py` (new `Emitter` withdrawal), `src/noodl/apps/water/elements.py` (new `ChezyManning`), `src/noodl/apps/water/network.py`
- Test: `tests/apps/water/test_elements.py`, `tests/verification/test_water_parity.py`

**Interfaces:**
- Produces: `EmitterOutflow(NodeWithdrawal)`: `__init__(self, nodes, coefficient, elevation, *, exponent=0.5, dp_transition=1e-9, learnable=False)`; flow `sign(p) C |p|^exponent`, `p = phi - elevation`, linear below `dp_transition` (finite slope at `p = 0`, as `HazenWilliams`); sign convention transcribed from `hydcoeffs.c` `emittercoeffs`/`emitflowchange`.
- Produces: `ChezyManning(length, diameter, roughness, *, minor_loss=0.0, kind="pipe", learnable=False)` with `head_loss(q)` for `FlowControlledPipe`; constant from EPANET's `input1.c`/`hydcoeffs.c` (internal feet and cfs, converted).
- Produces: `WaterResults.emitter_flow` is filled from this withdrawal (Task 10).

- [ ] **Step 1: Write failing tests.** `test_emitters_match_epanet` (two-loop, emitters at J4 and J6), `test_chezy_manning_matches_epanet` (two-loop with `HEADLOSS C-M`, roughness 0.011), `test_emitter_coefficient_gradient_against_central_differences`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove both refusals.
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): emitters and Chezy-Manning head loss`.

### Task 7: Tanks: volume curves, full and empty, overflow

**Files:**
- Modify: `src/noodl/storage.py` (optional `level_of`-style inverse on `NodeStorage`), `src/noodl/layers/kirchhoff.py` (`advance_storage`)
- Modify: `src/noodl/apps/water/tanks.py`, `src/noodl/apps/water/network.py`
- Test: `tests/apps/water/test_tanks.py`, `tests/test_storage.py` (or the existing storage test file), `tests/verification/test_water_parity.py`

**Interfaces:**
- Produces: `NodeStorage.potential_of(amount, drivers=None) -> Tensor | None` (default `None`). When a storage defines it, `advance_storage` moves the node to `potential_of(amount(start) + dt * flow)` (EPANET's `V += q dt; H = tankgrade(V)`); otherwise as today. `LinearStorage` keeps today's arithmetic bit for bit.
- Produces: `TankVolumeCurve(NodeStorage)` in `tanks.py`: `__init__(self, nodes, bottom, levels, volumes, *, head_scale=1.0)`, `amount` piecewise linear in level, `capacity` its slope, `potential_of` its inverse.
- Produces: the tank closure sets `"water.one_way"` on every link attached to a full tank (level ≥ max − `HTOL`) or empty tank (level ≤ min + `HTOL`) so it cannot fill or drain it, as `hydstatus.c` `tankstatus` does (pumps and CV pipes included); a link that both rules forbid gets status `CLOSED` for the step. The `[min, max]` refusal is removed for `tank_scheme="forward_euler"` and kept for the implicit schemes.
- Produces: `Tank.overflow = True`: when full, the tank is not closed; its level is held at max and the excess inflow is reported as `WaterResults.tank_overflow`. Refused by name with the implicit schemes.
- Produces: `TankLevels.time_to_limit(level, rate) -> Tensor` (seconds to max or min at the current rate, `inf` when moving away), used by Task 10's step rule.

- [ ] **Step 1: Write failing tests.** `test_linear_storage_advance_is_unchanged` (bitwise); `test_volume_curve_advance_conserves_volume`; `test_volume_curve_gradient_against_central_differences` (d level after 3 steps / d one curve volume, 1e-6 × scale); parity `test_volume_curve_tank_matches_epanet` (Net1 tank with a volume curve, 24 h, levels 2e-4 m); `test_tank_fills_and_closes_its_inlet_like_epanet`; `test_tank_empties_like_epanet`; `test_full_tank_at_start_closes_the_pump` (Review Focus 3); `test_tank_overflow_like_epanet`. These run through `water_run` once Task 10 exists; until then use a local loop of solve + `advance_storage` in the test, and switch them to `water_run` in Task 10.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the volume-curve and overflow refusals.
- [ ] **Step 4:** Run the core and water subsets; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): tank volume curves, full and empty tanks, overflow`.

### Task 8: Controls

**Files:**
- Modify: `src/noodl/apps/water/controls.py`, `src/noodl/apps/water/tanks.py` (controls move out of `TankLevels`)
- Test: `tests/apps/water/test_controls.py` (new), `tests/verification/test_water_parity.py`

**Interfaces:**
- Consumes: `Control` (Task 2); status/setting drivers (Task 3).
- Produces: `apply_controls(controls, *, time, clocktime, levels, pressures, heads, status, setting) -> tuple[Tensor, Tensor, Tensor]` returning new status, new setting and a per-instance `changed` flag; EPANET semantics (`hydraul.c` `controls`, `pswitch`): time controls fire when the clock reaches them; tank-level controls at the step's start levels; junction pressure controls are checked against `Grade ± HTOL` after a converged solve.
- Produces: `next_control_time(controls, *, time, clocktime, levels, rates) -> Tensor`: seconds to the next time control or tank-level crossing, per instance (replaces `TankLevels.event_step`).
- Produces: the state keys `"water.link_status"` and `"water.link_setting"` (per link) carried across steps; the closure writes `"water.status"`, `"water.setting"` and `"water.one_way"` from them.

- [ ] **Step 1: Write failing tests.** Unit: `test_time_control_fires_at_its_time`, `test_clocktime_control_respects_start_clocktime`, `test_level_control_both_directions`, `test_pressure_control_needs_a_resolve`, `test_setting_action_activates_a_valve`. Parity (through `water_run`, so finished in Task 10, written here): `test_net1_extended_period_tank_level` keeps its 8.181e-5 m result.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the control-form refusals.
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): every EPANET control form`.

### Task 9: Rules

**Files:**
- Create: `src/noodl/apps/water/rules.py`
- Test: `tests/apps/water/test_rules.py` (new), `tests/verification/test_water_parity.py`

**Interfaces:**
- Consumes: `Rule`, `Premise`, `Action` (Task 2).
- Produces: `evaluate_rules(rules, snapshot, *, time, clocktime) -> list[Action]`: transcribed from EPANET 2.2 `rules.c` (`checkrules`, `evalpremises`, `updateactionlist`, `takeactions`): premises on node and link attributes (`DEMAND`, `HEAD`, `GRADE`, `LEVEL`, `PRESSURE`, `FLOW`, `STATUS`, `SETTING`) and `SYSTEM` `TIME`, `CLOCKTIME`, `DEMAND`; `AND` binds tighter than `OR`; among actions on one link the highest `PRIORITY` wins, ties to the rule listed first. `snapshot` is a mapping of the current heads, pressures, levels, flows, statuses and settings by name.
- Produces: rules are checked at `rule_timestep` within each hydraulic step (Task 10 calls this), after simple controls.

- [ ] **Step 1: Write failing tests.** Unit: `test_and_binds_tighter_than_or`, `test_else_actions`, `test_priority_resolves_conflicts`, `test_control_and_rule_on_one_link` (Review Focus 4). Parity (finished in Task 10): `test_rules_match_epanet` on a small network with a tank, a pump and a PRV under four rules exercising `ELSE`, `PRIORITY`, `CLOCKTIME` and a `PRESSURE` premise.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove the rules refusal.
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): rule-based controls`.

### Task 10: Extended-period run

**Files:**
- Create: `src/noodl/apps/water/simulate.py`
- Modify: `src/noodl/apps/water/network.py` (remove `water_event_step`), `src/noodl/apps/water/__init__.py`, `benchmarks/water_eps.py`
- Test: `tests/apps/water/test_simulate.py` (new), `tests/verification/test_water_parity.py`

**Interfaces:**
- Consumes: Tasks 3–9.
- Produces:

```python
@dataclass
class WaterResults:
    time: Tensor            # (T,) seconds from the start, at the report times
    head: Tensor            # (..., T, n_nodes) m
    pressure: Tensor        # (..., T, n_nodes) m
    flow: Tensor            # (..., T, n_links) m3/s
    status: Tensor          # (..., T, n_links) CLOSED/OPEN/ACTIVE, from link_states
    setting: Tensor         # (..., T, n_links)
    tank_level: Tensor      # (..., T, n_tanks) m
    tank_overflow: Tensor   # (..., T, n_tanks) m3/s
    demand: Tensor          # (..., T, n_junctions) m3/s delivered
    emitter_flow: Tensor    # (..., T, n_junctions) m3/s
    energy: "PumpEnergy | None"   # Task 11
    node_names: list[str]
    link_names: list[str]

def water_run(model, state=None, drivers=None, *, duration: float | None = None,
              **solve_kwargs) -> WaterResults: ...
def pattern_factor(net, pattern: str | None, time: float) -> float: ...
```

- Step rule per spec (EPANET `hydraul.c` `timestep`, `tanktimestep`, `controltimestep`, `ruletimestep`): the step ends at the earliest of the hydraulic step, the next pattern-step boundary (with `PATTERN START`), the next report time, `next_control_time`, and `TankLevels.time_to_limit`; then rules are checked every `rule_timestep` inside it, ending the step early if a rule fires. Pattern values apply to junction demands (default pattern rule as in the reader), reservoir heads, pump speeds and energy prices. Pressure controls re-solve at the same time up to 10 passes, then raise by name.
- Batches: one shared step, the minimum over instances.

- [ ] **Step 1: Write failing tests.** `test_report_times_follow_report_start_and_step`; `test_a_pattern_boundary_inside_a_hydraulic_step_ends_the_step`; `test_batched_run_equals_single_runs` (Review Focus 5: two instances with different demand multipliers whose pump switches at different times; each instance's levels within 2e-4 m of its single run, and no step crosses either instance's event); move Net1's extended-period comparison onto `water_run` with its existing bound; finish the parity tests left open in Tasks 7–9.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement; remove `water_event_step` and the speed-pattern refusal; update `benchmarks/water_eps.py` to call `water_run`.
- [ ] **Step 4:** Run the water subset and `-m slow` water tests; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): extended-period run with EPANET's step rule`.

### Task 11: Pump energy and cost

**Files:**
- Create: `src/noodl/apps/water/energy.py`
- Modify: `src/noodl/apps/water/simulate.py`
- Test: `tests/apps/water/test_energy.py` (new), `tests/verification/test_water_parity.py`

**Interfaces:**
- Produces: `PumpEnergy` dataclass, per pump: `utilisation` (% of time on), `efficiency` (average %), `energy_per_volume` (kWh/m³), `average_power` and `peak_power` (kW), `cost` (per day), plus `demand_charge` (network); accumulated per hydraulic step as EPANET's `hydraul.c` `addenergy`/`getenergy` (efficiency from the pump's curve or the global value; price × price pattern).
- Produces: `water_run` fills `WaterResults.energy`.

- [ ] **Step 1: Write failing tests.** `test_energy_of_a_constant_pump_by_hand` (one pump, constant flow and head: kWh equals `rho g q h t / eff`); `test_energy_matches_epanet` on Net3 and Anytown against EPANET's energy section of its binary output (read with `wntr.epanet.io.BinFile`; if `wntr` does not expose it, parse the section from the binary per the EPANET 2.2 manual's output-file layout, in a test helper), bound 1e-5 relative.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the water subset; expected PASS.
- [ ] **Step 5:** Commit: `feat(water): pump energy and cost`.

### Task 12: Whole networks and load coverage

**Files:**
- Test: `tests/verification/test_water_networks.py` (new)
- Modify: `AGENTS.md` and `.github/scripts/select_tests.py` (add the new file to the `water` row)

- [ ] **Step 1: Write the tests.** One parametrised `test_network_matches_epanet` over Net3, Net6, ky4, ky10 and Anytown: full `DURATION` (ky4 and ky10 have 0, so one steady solve); heads, flows, link states and tank levels at every report time; bounds per Global Constraints; all marked `slow`. `test_every_wntr_network_builds_or_names_its_refusal`: every `.inp` under `wntr`'s `library/networks` and `tests/networks_for_testing`, except `bad_syntax.inp` and `bad_values.inp`, either builds a model or raises a `ValueError` whose message names the section or feature; the test records which, and fails on any other exception.
- [ ] **Step 2:** Run `-m slow`; where a network misses a bound, find the cause (debug, do not loosen first). A bound is loosened only with the measured cause written in the test.
- [ ] **Step 3:** Commit: `test(water): Net3, Net6, ky4, ky10 and Anytown against EPANET`.

### Task 13: Documentation, benchmarks, final checks

**Files:**
- Modify: `docs/applications/water.md`, `docs/formats/epanet.md`, `docs/plans/water-prv-psv-design.md`, `docs/catalogue/water.md` (generated)

- [ ] **Step 1:** Rewrite `water.md` (objects, physics per device, running over time, verification table with the measured numbers from Tasks 3–12, Limitations reduced to what is still refused) and `epanet.md` (read / ignored / refused lists). Mark the PRV/PSV note as decided, recording the answers listed at the end of the spec. Regenerate the catalogue.
- [ ] **Step 2:** Run the benchmarks of Task 0 three times; record the medians beside the baseline in the progress file; any slowdown beyond ~5 % on a network without new devices is fixed before continuing.
- [ ] **Step 3:** `ruff check .`; full `pytest` with coverage; `-m slow` water tests; `python -m mkdocs build --strict`; expected: all pass, coverage ≥ 90 %.
- [ ] **Step 4:** Commit: `docs(water): complete EPANET hydraulics`.
