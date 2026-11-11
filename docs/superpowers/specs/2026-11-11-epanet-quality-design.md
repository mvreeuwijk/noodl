# EPANET water quality in `apps.water`

Status: designed 2026-11-11 under the user's standing instruction ("do not wait for me and make
sensible decisions"); decisions are recorded as rulings; reviewed by a Fable agent in place of the
user (verdict: approve with changes — all incorporated in this revision). Sub-project 2 of 3:
sub-project 1 (hydraulics) is merged (PR #34); the WSIMOD port follows.

## Goal

`apps.water` runs EPANET 2.2's water-quality simulation inside `water_run` and matches EPANET 2.2:
every quality mode, reaction, source type and tank mixing model, read from the `.inp` file.
Success is measured on EPANET's example networks and on one small case per feature (Verification).

Reference: EPANET 2.2 through `wntr` 1.5.0, driven through its toolkit (`ENepanet` / `ENlib`) on
the committed file as written (Ruling Q6), not through `EpanetSimulator`'s rewrite of the `.inp`.

## Constraints

- Branch `epanet-quality` from `main`; merge `main` in after PR #35 lands and before code is
  written; one PR; a Fable review before it is merged.
- Hydraulics are unchanged: quality reads the hydraulic results of each step; it never feeds back.
- `water_run(..., quality=False)` or a network without quality pays nothing.
- Physical names; no aliases; code, tests and docs change together; no wall-clock assertions.
- Every transcribed formula cites its EPANET 2.2 file and function (tag v2.2).

## Approach: EPANET's Lagrangian procedure, transcribed (Ruling Q1)

EPANET's answer is not the advection–reaction equation's answer but a specific discrete procedure
(`quality.c`, `qualroute.c`, `qualreact.c`): explicit-Euler reactions per quality step
(`pipereact`: `cnew = max(0, c + (rbulk + rwall) dt)`), segment merging at the tolerance `Ctol`
(`evalnodeoutflow`), a topological node order with a cycle-breaking rule (`sortnodes`,
`selectnonstacknode`), zero-volume pumps and valves passing water within the step, stagnant flows
(`Q_STAGNANT`) handled apart. Only a transcription reproduces it; an Eulerian discretisation would
not, however refined (rejected; may come later as a separate option).

### State layout (Ruling Q4)

All segments of all links and tanks live in one pool of two 1-D float64 tensors, concentration
`c` and volume `v`; each pipe and tank keeps an ordered index chain in Python (EPANET's
`FirstSeg`/`LastSeg`). Reactions are one vectorised operation over the pool per quality step
(per-link rate coefficients gathered by a link-index tensor); the node loop stays sequential in
EPANET's order.

### What is differentiable (Ruling Q5)

Every volume and concentration stays in the autograd graph; only the decisions are made on
detached values: `v > 0`, `vseg >= seg.v`, `|c_seg - c| < Ctol`, `vnet > 0`, flow direction and
the node order. Segment volumes are `|q| dt` from the solve's flow tensors, so results are
piecewise differentiable with respect to reaction coefficients, source strengths, initial quality
AND hydraulic parameters (demands, roughness, pump speed) through travel times. Results jump where
a decision flips; gradient tests must show the ±h runs have identical segment counts per link and
identical node order (exposed in `diagnostics`). Autograd is enabled for quality only when a
quality input or the solve's flows require grad; otherwise the router runs detached (a 168 h run
would otherwise build ~1e6 graph nodes). `benchmarks/water_quality.py` measures speed; no asserts.

### Batches

Run one instance at a time (each instance has its own segments and node order); the hydraulic
step is shared as today. Documented.

## EPANET semantics to transcribe (each cites its function)

- **Initial segments** (`initsegs`): one segment per pipe with the DOWNSTREAM node's initial
  quality and volume `0.785398 L D^2`; none for pumps and valves; one per tank at its initial
  volume (2COMP: mixing zone `min(V0, V1max)`, the rest stagnant).
- **Initial quality** (`initqual`): `[QUALITY]` per node; for TRACE every node starts at 0 (the
  file's values are ignored) but tank segments take the file's initial quality (`initsegs`).
- **Quality step** (`adjustdata`): `QUALITY TIMESTEP` default `Hstep / 10` (integer s), capped at
  `Hstep`; `DURATION 0` forces `QUALITY NONE`. Within a hydraulic step: `dt = min(Qstep, hydstep -
  qtime)`, the remainder last (`nextqual`); quality never runs past the duration.
- **Order within a quality step**: reactions first (`reactpipes`, `reacttanks`: pipes and all tank
  segments incl. the 2COMP stagnant zone; pumps/valves do not react), then transport over the node
  order (`transport`). Wall rate coefficients are recomputed once per hydraulic step from that
  step's flow (`ratecoeffs`); the node order only when a flow direction changes (`flowdirchanged`).
- **Stagnant flow** (`Q_STAGNANT = 0.005 gpm`): a link below it has no direction, is left out of
  the sort and moved N1 -> N2 with its actual tiny flow; a node with `volout/dt <= Q_STAGNANT` gets
  no source. noodl's ideal closed links (zero flow) fall under it.
- **Node order** (`sortnodes`, `selectnonstacknode`, adjacency order of `buildadjlists`):
  transcribed with its tie rules; recorded in `diagnostics`.
- **Node mixing** (`findnodequal`, `evalnodeinflow`, `evalnodeoutflow`, `noflowqual`): inflow mass
  and volume; outflow segments merged at `Ctol`; a node with no inflow keeps its quality
  (non-reactive) or takes the mean of adjoining segment ends (reactive); a junction's demand is
  the hydraulics' delivered demand plus emitter flow (`hydsolver.c`: `NodeDemand`), `max(0, ·)` in
  outflow, `min(0, ·)` as diluting inflow.
- **Flow reversal** (`reversesegs`).
- **Reservoirs**: the reservoir's quality is overwritten by its source contribution when that is
  nonzero and otherwise KEEPS ITS PREVIOUS VALUE (`findnodequal`); `[QUALITY]` on a reservoir is
  its quality.
- **Sources** (`sourcequal`): CONCEN (acts at a junction only when its demand is negative),
  MASS (mass per minute, `/60`), SETPOINT (`max(c - quality, 0)`), FLOWPACED; each with an
  optional pattern; at the trace node, TRACE holds 100 (reservoir: source 100; junction/tank:
  `max(100 - quality, 0)`, quality 100).
- **Tanks**: quality tracks its own segment volumes (`seg.v += vnet`, `max(0, ·)` in
  `tankmix1`), not the hydraulic tank volume; the reported tank quality is the node's quality (the
  mixing-zone or withdrawn water; an external source is added to the outflow but not reported).
- **Mixing models** (`tankmix1`..`tankmix4`): MIXED, 2COMP (fraction; 0 means 1), FIFO, LIFO
  (inflow merging at `Ctol`).
- **Bulk reactions** (`bulkrate`): order n with the limiting potential `c1 = max(0, sgn(kb)(Climit
  - c))` for n >= 1 (order 0: c = 1); negative order is Michaelis–Menten (`c1 = Climit + sgn(kb) c;
  rate = kb c / c1`), transcribed; global coefficient and per-pipe `BULK`.
- **Wall reactions** (`piperate`, `wallrate`): order 0 (mass/area/day) or 1 (length/day), any
  other order refused by name (EPANET error 213); mass-transfer coefficient from the Sherwood
  number: `Re < 1: Sh = 2`; `Re >= 2300: Sh = 0.0149 Re^0.88 Sc^0.333`; else Graetz `3.65 + 0.0668
  y / (1 + 0.04 y^0.667)`, `y = (d/L) Re Sc`; `Sc = viscosity / diffusivity` from `[OPTIONS]
  VISCOSITY` and `DIFFUSIVITY` (relative to chlorine's); `DIFFUSIVITY 0` switches the limitation
  off (order 0: `kf = BIG`; order 1: `kw (4/d)`); global and per-pipe `WALL`; `ROUGHNESS
  CORRELATION` for pipes without their own wall coefficient.
- **Tank reactions** (`tankreact`): `bulkrate(c, Kb_tank, TankOrder)` with ONE global `ORDER TANK`
  (default 1); a tank's coefficient is `TANK <id> <k>`, defaulting to the global BULK coefficient
  (`adjustdata`). There is no `GLOBAL TANK` keyword.
- **AGE** adds `dt / 3600` hours per step; `Ctol` 0.01 h (`AGETOL`). **TRACE** has no reactions.
- **Units**: concentrations stay in the file's quality unit (mg/L, ug/L, hours, percent) with
  volumes in m^3 — EPANET's internal `Bucf`/`Tucf` artefacts are not transcribed; the mass balance
  is reported in the chemical's mass unit (internal c x V in mg/L x m^3 = g, x 1000 for mg).
- **Mass balance** (`updatemassbalance`, `evalmassbalance`): initial, inflow, outflow, reacted,
  final, and the ratio, with EPANET's sign and side conventions (reservoir inflow counts as
  outflow mass; a reservoir without a source adds `quality x volout` as inflow; reacted keeps its
  sign).
- **Link quality** (`avgqual`): volume-weighted segment average; end-node mean for a link without
  segments.

## Reader

`read_epanet_inp` reads `[QUALITY]`, `[SOURCES]`, `[REACTIONS]`, `[MIXING]`, `[OPTIONS] QUALITY`
(`NONE|CHEMICAL|AGE|TRACE <node>`, and EPANET's form "any other word [units]" = CHEMICAL named so,
`optionchoice` — Net1 writes "Chlorine mg/L"), `DIFFUSIVITY`, `TOLERANCE`, `[TIMES] QUALITY
TIMESTEP`; EPANET's range forms `id1 id2 value` in `[QUALITY]` and `[REACTIONS] BULK|WALL|TANK`
(numeric when both parse as positive integers, else lexicographic — `qualdata`, `reactdata`); the
legacy 3-token `[SOURCES]` line (defaults to CONCEN). Into `WaterNetwork.quality: WaterQuality |
None` (mode, chemical name and units, trace node, Qstep, Ctol, diffusivity, the three orders,
global coefficients, per-pipe bulk and wall, per-tank coefficients, limiting potential, roughness
correlation, initial qualities, sources, mixing models), in SI. Anything not modelled is refused by
name. No quality section is ignored any more.

## Interfaces (Ruling Q7)

- `WaterNetwork.quality: WaterQuality | None`, filled by the reader, settable by hand.
- `water_run(..., quality: bool | None = None)`: `None` runs it when the network defines a mode
  other than NONE and the duration is positive (EPANET's rule); `False` turns it off; `True`
  demands it and raises by name when the network has none.
- A `QualityRouter` in `src/noodl/apps/water/quality.py` (with `quality_react.py` for the rates
  if the module grows), called from `water_run`'s loop: after the solve at `t` the recorder takes
  the quality as it stands (EPANET's `saveoutput` before `transport`), then the router transports
  over the planned step with that solve's flows, statuses and delivered demands, before
  `t += step`.
- `WaterResults` gains `node_quality` (..., T, n_nodes), `link_quality` (..., T, n_links),
  `quality_units` (str), `mass_balance: QualityMassBalance | None` (initial, inflow, outflow,
  reacted, final, ratio); all `None` when quality is off. No separate tank field.
- Learnable inputs, as drivers/state keys: `quality.bulk_coefficient` (n_pipes, 1/s),
  `quality.wall_coefficient` (n_pipes), `quality.tank_coefficient` (n_tanks),
  `quality.source_strength` (n_nodes, pattern-scaled), `quality.initial` (n_nodes, state).
- Coexistence with the steady tracer (Ruling Q2): the existing `build_model(quality=float)` keeps
  its `TransportLayer` (named "steady tracer" in the docs; not renamed — no aliases, users rely on
  it; the two `quality` parameters have different types, accepted). `water_run` refuses a model
  that carries the steady quality layer when EPANET quality is on, by name. The docs' existing
  TRACE row is relabelled as the steady tracer's.
- `diagnostics`: per-step segment counts per link, node order, per-node inflow volume/mass and
  outflow volume.

## Verification (Ruling Q6)

References: EPANET 2.2 through the toolkit on the committed file (`ENopen`, `EN_solveH`,
`EN_solveQ`, `EN_report`, then wntr's `BinFile` reader); a step-level harness (`EN_openQ`,
`EN_initQ`, `EN_runQ`, `EN_stepQ`, reading `EN_QUALITY` per node and `EN_LINKQUAL` per link at
every quality step) names the first diverging step and node; the mass-balance ratio through
`EN_getstatistic(EN_MASSBALANCE)` (ctypes on `ENlib`) and the five terms from the `.rpt` with
`[REPORT] STATUS YES` (`writemassbalance`, 6 significant digits).

Bounds: node and link quality over the full (node, time) matrix — 99.9th percentile within 1e-5
of the quality's scale, worst within `Ctol` absolute; a worst case beyond 1e-5 of scale is admitted
only with the flipped merge (segment count differing by one at that link and step) written in the
test. Mass balance: terms 2e-5 relative, ratio 1e-6. Loosen only with a measured cause in the test.

Networks:
- **Net1** (chlorine, bulk -0.5, wall -1, 24 h) — the reactive case.
- **Net2** (fluoride, CONCEN source with a pattern, 55 h, one tank).
- **Net3** (TRACE of the lake, 168 h, reversing flows, three tanks) — the sort and reversal case.
- **Net6**: shipped quality is identically zero; a smoke case (runs; mass-balance ratio within
  1e-3), plus a version with an added source, bounded by its hydraulic parity (levels 2e-3 m)
  with the cause written.
- **ky10** (AGE) and **ky4** (TRACE) ship with `DURATION 0` (no quality in EPANET): run with an
  edited duration (24 h), CONDITIONAL on first measuring their extended-period hydraulics against
  EPANET at that duration (a plan dependency).

Per-feature cases: each mode; bulk orders 0, 1, 2, negative (Michaelis–Menten), with a limiting
concentration (growth and decay); wall order 0 (SI and US units) and 1, mass-transfer limited and
`DIFFUSIVITY 0`; roughness correlation; each source type with and without a pattern; a reservoir
source whose pattern goes to 0 (keeps its last value); SETPOINT below the node's quality (adds 0);
CONCEN at a junction with negative demand vs one without; each mixing model; a 2COMP tank crossing
`V1max`; FIFO/LIFO tanks emptying through several segments; a pipe whose flow reverses; a pump or
valve in the path; a dead end below `Q_STAGNANT`; a reactive dead-end junction (`noflowqual`); a
step shorter than Qstep ending at a tank limit; TRACE at a junction, a reservoir and a tank.
Gradients: d(node quality)/d(bulk coefficient, wall coefficient, source strength, initial quality,
a pipe roughness) against central differences, with structural identity of the ±h runs asserted.

## Documentation

`docs/applications/water.md`: a water-quality section (modes, reactions, sources, mixing,
running, differentiability and its jumps, verification rows, limitations); the steady tracer
relabelled; `docs/formats/epanet.md`: the quality sections read; the catalogue regenerated.

## Out of scope

EPANET-MSX; EPANET 2.3 features; an Eulerian discretisation; the per-link average reaction-rate
output and the `Wbulk/Wwall/Wtank/Wsource` report summary (report cosmetics).

## Rulings

- Q1: transcribe EPANET's Lagrangian procedure; no fixed-capacity segment buffers (AGE segments
  never merge, node order is per instance) — cost if wrong: no vectorised batches.
- Q2: keep the steady `quality=` option as the "steady tracer", refused alongside EPANET quality —
  cost if wrong: two quality options.
- Q3: Fable reviews the spec in place of the user — cost if wrong: a decision the user would change.
- Q4: flat tensor pool + Python index chains — cost if wrong: a re-layout.
- Q5: only decisions detached; volumes and concentrations attached; grad only when requested —
  cost if wrong: weaker gradient claims.
- Q6: references from EPANET's toolkit on the committed file; percentile + worst-within-Ctol
  bounds; ky4/ky10 need edited durations after an EPS hydraulic check; Net6 a smoke case — cost if
  wrong: weaker evidence on two networks.
- Q7: interfaces as above (`WaterNetwork.quality`, `water_run(quality=)`, `WaterResults` fields,
  learnable keys, `QualityRouter`) — cost if wrong: an API change before release.
