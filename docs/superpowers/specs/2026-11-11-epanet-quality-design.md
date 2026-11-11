# EPANET water quality in `apps.water`

Status: designed 2026-11-11 under the user's standing instruction ("do not wait for me and make
sensible decisions"); decisions are recorded as rulings, and the spec is reviewed by a Fable agent
in place of the user. Sub-project 2 of 3: sub-project 1 (hydraulics) is merged (PR #34); the
WSIMOD port follows.

## Goal

`apps.water` runs EPANET 2.2's water-quality simulation alongside `water_run` and matches EPANET
2.2 on it: every quality mode, reaction, source type and tank mixing model, read from the `.inp`
file. Success is measured on EPANET's own example networks (Net1 chlorine, Net2, Net3 two-source
trace, Net6 chemical, ky10 water age) and on one small case per feature.

The reference stays EPANET 2.2 through `wntr` 1.5.0 (`EpanetSimulator` and, where needed, its
toolkit `ENepanet`).

## Constraints

- Branch `epanet-quality` from `main` 595f5e7; one PR; a Fable review before it is merged.
- Hydraulics are unchanged: quality reads the hydraulic results; it never feeds back.
- No loss of speed for hydraulics-only runs: a network with `QUALITY NONE` pays nothing.
- Physical names; no aliases; code, tests and docs change together; no wall-clock assertions.
- Cite the EPANET 2.2 source file and function in the docstring of every transcribed formula.

## Approach: EPANET's Lagrangian transport, transcribed

EPANET 2.2 moves water quality with a Lagrangian time-driven method (`quality.c`, `qualroute.c`,
`qualreact.c`): each pipe is an ordered list of segments of water, each with a volume and a
concentration; at each quality step the segments advance with the pipe's flow, water leaving a
pipe mixes at its downstream node with all other inflow, and new segments enter at the upstream
end (merging with the last one when the concentrations differ by less than the quality
tolerance). Reactions act on each segment and on each tank.

noodl transcribes this method (Ruling Q1). Concentrations and segment volumes are tensors, so
results are differentiable with respect to reaction coefficients, source strengths and initial
qualities; the segment bookkeeping (when a segment is created, merged or removed) is a discrete
decision made on detached values, as link statuses are in the hydraulics. A batch is run one
instance at a time, because each instance has its own segment structure; this is documented.

The steady quality option that exists today (`build_model(quality=...)`, a `TransportLayer` with
bulk decay at steady state) stays as it is, documented as a steady-state approximation distinct
from the EPANET mode.

### Rejected alternative

An Eulerian finite-volume discretisation of each pipe inside `TransportLayer` (batched and fully
differentiable) carries numerical diffusion that only converges with refinement, so it cannot
reach EPANET's precision as the hydraulics do. It may come later as a separate option; it is not
part of this spec.

## Scope

### Quality modes (`[OPTIONS] QUALITY`)

- `NONE`; `CHEMICAL` with a name and units (mg/L or ug/L); `AGE` (hours); `TRACE <node>`
  (percent of the water from that node).

### Reactions (`[REACTIONS]`)

- Bulk reactions of any order n (`ORDER BULK`), with a limiting concentration (`LIMITING
  POTENTIAL`) for growth and decay; global and per-pipe coefficients (`GLOBAL BULK`, `BULK`).
- Wall reactions of order 0 or 1 (`ORDER WALL`), with the mass-transfer limitation EPANET applies
  (Sherwood-number correlation for laminar and turbulent flow, using `[OPTIONS] DIFFUSIVITY`
  relative to chlorine's); global and per-pipe coefficients (`GLOBAL WALL`, `WALL`); the
  roughness correlation (`ROUGHNESS CORRELATION`) for pipes without their own wall coefficient.
- Tank reactions (`ORDER TANK`, `GLOBAL`/`TANK` coefficients), using the tank's own bulk order.
- Reactions apply in CHEMICAL mode; AGE adds time; TRACE has none.

### Sources and initial quality

- `[QUALITY]` initial quality at every node (junctions, reservoirs, tanks).
- `[SOURCES]` of each type: `CONCEN`, `MASS`, `SETPOINT`, `FLOWPACED`, with an optional pattern.
- Reservoir quality follows its source (or its initial quality).

### Tank mixing (`[MIXING]`)

- `MIXED`, `2COMP` (with the mixing-zone fraction), `FIFO`, `LIFO`, as `qualroute.c`.

### Time

- `[TIMES] QUALITY TIMESTEP` (default: EPANET's rule, the hydraulic step / 10 or 5 min, check the
  source); quality steps run inside each hydraulic step, on that step's flows.
- `[OPTIONS] TOLERANCE` for segment merging.
- Reported at the report times, as the hydraulic results are.

### Results

`WaterResults` gains a quality block: node quality, link quality (EPANET's average over a link's
segments), and tank quality, at every report time; and EPANET's mass-balance terms
(`ENgetqualinfo`-style: initial, inflow, outflow, reacted, final mass) for the run.

### Out of scope

- EPANET-MSX (multi-species), EPANET 2.3 features, and an Eulerian discretisation (above).

## Reader

`read_epanet_inp` reads `[QUALITY]`, `[SOURCES]`, `[REACTIONS]`, `[MIXING]`, `[OPTIONS] QUALITY`,
`DIFFUSIVITY`, `TOLERANCE`, and `[TIMES] QUALITY TIMESTEP`, converting to SI (concentration units
as EPANET reports them: mg/L, ug/L; wall coefficients in the file's length units per day; mass
sources in mass per minute). They are no longer ignored. Anything not modelled is refused by name.

## Verification

Against EPANET 2.2 through `wntr` 1.5.0. EPANET's quality output is float32; bounds per feature:
node and link quality 1e-5 relative to the quality's scale (with an absolute floor at 1e-6 of the
scale), mass balance 1e-5 relative. A bound is loosened only with a measured cause written in the
test.

- One case per feature on small networks: each mode; bulk order 0, 1, 2 and with a limiting
  concentration; wall order 0 and 1 (mass-transfer limited and not); roughness correlation; each
  source type with and without a pattern; each mixing model; a tank emptying and refilling.
- Whole networks over their duration: Net1 (chlorine, bulk and wall reactions), Net2, Net3 (TRACE
  of the lake), Net6 (CHEMICAL), ky10 (AGE), ky4 (TRACE); slow-marked where large.
- Gradients: d(node quality)/d(bulk coefficient, wall coefficient, source strength, initial
  quality) against central differences.

## Documentation

`docs/applications/water.md` gains a water-quality section (modes, reactions, sources, mixing,
running, verification rows, limitations); `docs/formats/epanet.md` lists the quality sections as
read; the catalogue is regenerated.

## Rulings

- Q1: EPANET's Lagrangian method transcribed, not an Eulerian discretisation — the reference is
  EPANET and sub-project 1 set the parity standard at EPANET's float32 floor; cost if wrong: no
  batched quality (instances loop) and no gradient through the segment structure.
- Q2: the existing steady `quality=` option is kept as a distinct, documented steady-state
  approximation — it is tested and some users may rely on it; cost if wrong: two quality options.
- Q3: Fable reviews the spec in place of the user (standing instruction) — cost if wrong: a spec
  decision the user would have changed.
