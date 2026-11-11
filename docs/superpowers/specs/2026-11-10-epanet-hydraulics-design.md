# EPANET hydraulics: completing `apps.water`

Status: design approved in conversation on 2026-11-10; this document is for review before planning.
Sub-project 1 of 3. The other two, EPANET water quality and a noodl-native port of WSIMOD, get
their own specs.

## Goal

`apps.water` reads and solves every hydraulic feature of an EPANET 2.2 `.inp` file and matches
EPANET 2.2 on them. Success is measured on real networks: Net3, Net6, ky4, ky10 and Anytown
load, run for their full `DURATION` and match EPANET (see Verification). Today they are refused.

The reference stays EPANET 2.2, run through `wntr` 1.5.0's `EpanetSimulator`.

## Constraints

- **Timing.** Work starts from `main` after the core-unification M4–M8 PR has merged. That PR
  rewrites water tanks (`m4-water`: tanks become storage at boundary nodes, `TankStorage` and the
  datum node are deleted) and the law Jacobian contract (M5). This spec names the behaviour; the
  plan names files and classes against the merged core.
- **No loss of speed.** Networks without the new devices must not get slower (see Speed).
- **Understandable names.** Public names describe the physical device or quantity, not the
  numerical method.
- **Code, tests and docs change together** (CONTRIBUTING.md).
- **One branch, one PR.**

## Approach: each device is one equation in the one solve

Every device that changes state (PRV, PSV, check valve, a pump that cannot reach the required
head, a link closed by a full or empty tank) is written as one exact equation that holds in each
of its states, solved together with the rest of the network in a single Newton solve. There is no
outer loop that flips statuses and solves again. This is option B of
`docs/plans/water-prv-psv-design.md` (commit 46e38ef, on the core-unification branches; it
reaches `main` with the M4–M8 PR), the same mechanism `BreakerValve` (PBV) uses today, and its
§1–§3 are part of this spec.

The state each device ended in (active, open, closed) is read from the converged answer and
reported.

Consequences, accepted:

- noodl gives the ideal-device answer everywhere. EPANET differs from it only inside its own
  tolerance band (`HTOL` = 1.524e-4 m of head), where EPANET can report an active PRV with a
  small negative head drop. Tests allow `HTOL` in head inside that band and the bound targets
  outside it.
- Gradients are exact inside each state. On a switch they are the derivative of the state the
  tie rule selects (one-sided), not an average.
- An opt-in smoothed form, with a width in metres, gives a gradient that blends across a switch,
  for calibrating settings. The exact form is the default.

### Absolute heads

A PRV or PSV needs the absolute head at one end, but a law sees only the head difference and
the flow of its own edge. The design note's sense edge to a datum node no longer works once M4
deletes the datum. The approach: give a law the heads of its two end nodes, as part of what the
core passes to it, so any law can use them. The plan settles the exact interface against the
merged core, keeping the change inside the law contract and the Jacobian assembly.

## Scope: devices and physics

### Valves and link status

- **PRV and PSV.** As in the design note: one condition that holds in the active, open and closed
  states, using EPANET's targets (`Hset` = elevation of the sensed node + setting).
- **Check valves (`CV` pipes).** The reverse-flow half of the same condition: closed when the
  flow would reverse, otherwise an ordinary pipe.
- **Fixed status.** `[STATUS] OPEN|CLOSED` and a pipe's `CLOSED` column. A closed link carries no
  flow. A fixed status on a PRV/PSV/FCV/TCV/PBV/GPV overrides its setting, as EPANET does.
- **Settings and statuses as inputs.** Every valve setting and every link status is a driver, so
  `[STATUS]`, controls and rules can change it between steps. A valve setting can also be made
  learnable.
- **Unreachable settings.** A PRV/PSV whose setting cannot be reached ends open, and is reported
  in that state, matching EPANET's result. An error is raised by name only when the solve is
  singular (EPANET's `XPRESSURE` cases, for example two parallel PRVs with equal settings).

### Pumps

- **Three-point curves with any exponent.** EPANET's fit `h = A − B qᶜ` through three points
  (replacing today's refusal when the exponent is not 2).
- **Multipoint curves.** Four or more points, joined by straight segments as EPANET does.
- **Constant-power pumps.** Transcribed from EPANET 2.2's `hydcoeffs.c`, as PBV and GPV were.
- **Speed.** Relative speed is a driver, so speed patterns and controls can change it.
- **Shut-off.** A pump that cannot deliver the required head closes like a check valve, by the same
  condition as the valves. This replaces today's large-penalty leak beyond shut-off.

### Tanks

- **Volume curves.** A tank's volume as a function of level, from `[CURVES]`, in place of a
  cylinder.
- **Full and empty tanks.** A full tank closes the links that would fill it and an empty tank the
  links that would drain it, as EPANET does.
- **Overflow.** EPANET 2.2's `[TANKS]` overflow flag: a full tank spills instead of closing.

### Other physics

- **Emitters.** `[EMITTERS]` and the `EMITTER EXPONENT` option: an outflow `q = C pᵞ` at a node,
  a node source like `PressureDrivenDemand`.
- **Chezy-Manning head loss.** `HEADLOSS C-M`, with EPANET's constant taken from its source.
- **Energy and cost.** Pump energy and cost from `[ENERGY]` (global and per-pump price, pattern
  and efficiency curve, demand charge), computed after the run. It does not change the hydraulics.

### Out of scope

- EPANET 2.3 features, including `[LEAKAGE]`. They are refused by name.
- Water quality (sub-project 2). Its sections stay ignored by the reader until then.

## Running over time

A new function runs an extended-period simulation for the file's `DURATION`. Each step:

1. applies the current pattern values: junction demands, reservoir heads, pump speeds, energy
   prices;
2. applies controls and rules;
3. solves;
4. advances the tank levels.

It returns time series at the report times (`REPORT START`, `REPORT TIMESTEP`): heads, pressures,
flows, link states, valve and pump settings, tank levels, delivered demand and emitter outflow,
and pump energy.

**Step length.** A step ends at the earliest of the next hydraulic step, a pattern-step boundary,
a report time, a time-based control, the time a tank would fill or empty, and a tank-level
control crossing. Today's `event_step` becomes one case of this.

**Controls.** Every `[CONTROLS]` form: `AT TIME`, `AT CLOCKTIME`, and `IF NODE` on a tank level or
a junction pressure, with a status (`OPEN`, `CLOSED`) or a numeric setting as the action. Pressure
controls on junctions are checked after a solve converges, as in EPANET's `pswitch`.

**Rules.** `[RULES]` transcribed from EPANET 2.2's `rules.c`: `IF`/`AND`/`OR`/`THEN`/`ELSE` over
every premise EPANET supports (node and link attributes, `SYSTEM` time, clock time and demand),
`PRIORITY` between conflicting rules, checked at the rule time step inside each hydraulic step.

**Gradients.** Link states are discrete. Gradients flow through heads, flows, levels, settings,
roughness and demands, never through the decision to switch.

**Batches.** All instances in a batch take the same step, the shortest any of them needs. A
single run follows EPANET's step sequence exactly. In a batch, an instance whose step was
shortened by another instance's event differs from its own single run only by that extra step
boundary. This removes today's limitation that tank-level events are not batch-safe.

## Reader

`read_epanet_inp` reads every hydraulic section and option of EPANET 2.2. New sections:
`[STATUS]`, `[RULES]`, `[EMITTERS]`, `[ENERGY]`. Extended: `[PIPES]` status column, `[PUMPS]`
`POWER` and `PATTERN`, `[TANKS]` volume curve and overflow flag, `[VALVES]` PRV/PSV, `[CONTROLS]`
all forms, `[OPTIONS]` `HEADLOSS C-M`, `EMITTER EXPONENT`, `[TIMES]` `RULE TIMESTEP`,
`REPORT START`, `START CLOCKTIME`, `PATTERN START`. Anything still unsupported is refused by name.

## Verification

All against EPANET 2.2 through `wntr` 1.5.0, in `tests/verification/test_water_parity.py` or new
files beside it.

**Per device.** Bounds as today: heads 1e-6 relative, flows 1e-6 m³/s absolute (EPANET's closed
links leak about 2e-8 m³/s through its large-penalty term), a valve's controlled head equal to its
setting. The state noodl reports is compared with `wntr`'s `link["status"]`. Cases:

- the 11 PRV/PSV cases in §4 of the design note, including the near-switch sweep;
- check valve closed and open; a pipe closed by `[STATUS]` and by the `CLOSED` column; a fixed
  status on a PRV;
- pumps: a three-point curve with an exponent other than 2, a multipoint curve, constant power, a
  speed pattern, a pump closed beyond shut-off;
- tanks: a volume curve, a tank that fills and closes its inlet, one that empties, overflow;
- emitters; Chezy-Manning on the two-loop network;
- each control form; rules with `ELSE` and conflicting priorities.

**Whole networks over time.** Net3, Net6, ky4, ky10 and Anytown over their full `DURATION`.
Heads, flows, link states and tank levels at every report time. Targets: heads 1e-5 relative,
tank levels 2e-4 m (today's Net1 bound). A target is loosened only with a measured cause recorded
in the test (the `HTOL` band, or a control switching one step apart). The large networks are
marked `slow`.

**Load coverage.** Every `.inp` in `wntr`'s bundled networks either builds a model or is refused
with the feature named, excluding its deliberately malformed files (`bad_syntax.inp`,
`bad_values.inp`) and files that need quality (sub-project 2).

**Gradients.** Central differences through an active PRV and an active PSV (setting, demands,
roughness), pump speed, a tank volume curve and an emitter coefficient, at 1e-6 × scale. One-sided
checks either side of a switch confirm the gradient jumps between them.

**Speed.** `benchmarks/water_eps.py` and the steady water benchmarks before and after, same
machine, same session. A slowdown on a network without new devices beyond run-to-run noise (about
5 %) is a defect.

## Documentation

- `docs/applications/water.md` and `docs/formats/epanet.md` rewritten for the new scope; the
  Limitations section shrinks to what is still refused.
- The catalogue regenerated with `scripts/gen_catalogue.py`.
- `docs/plans/water-prv-psv-design.md` marked as decided, recording the answers to its open
  questions: exact by default with opt-in smoothing; the ideal answer inside `HTOL`; settings as
  drivers and learnable; unreachable settings reported as open, raised only when singular; check
  valves, closed pipes and fixed statuses in scope; absolute heads through the law's end-node
  heads.
