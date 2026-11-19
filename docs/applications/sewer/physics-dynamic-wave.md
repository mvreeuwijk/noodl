# Sewer physics: the dynamic wave

The equations, the solve and the deliberate differences from SWMM under
`routing="dynamic_wave"`. For running it see [Dynamic-wave runs](dynamic-wave.md); for the
geometry, the kinematic path, the air and the quality see [Physics](physics.md).


With `routing="dynamic_wave"` (SWMM's `DYNWAVE`) the network is a Kirchhoff layer on **head**.
Where a conduit's flow follows from continuity alone, as on a dendritic kinematic-wave
network, nothing else is needed; where backwater, a flat conduit or a loop makes the flow depend
on the heads at both ends, the heads are the unknowns and the flows follow from the head
differences. The water side then has one `KirchhoffLayer` named `"water"` (quantity `head`
in m, flow in m³/s) over the conduits, with the outfalls as its boundary, and a
`JunctionStorage` at every junction.

`JunctionStorage` is the layer's node storage (`noodl.NodeStorage`): the water stored at a
junction is $A_s H$, and the layer steps it with its own `dt` and its trapezoidal scheme. SWMM's
junction rules (the bed, the EXTRAN band and surcharge anchor, the flood wall, ponding and the
drop-end area switch, all described below) are `max`/`min` compositions over the storage line,
so `JunctionStorage` overrides the storage's `rate` and `rate_slope` and keeps each rule as SWMM
has it. Its step start is SWMM's own (the drivers `"water.head_start"` and
`"water.net_inflow_start"`), not the layer's step-start potentials or the flow into storage, so
a step that ends unconverged keeps SWMM's fixed points. After every step the model state holds
`"water.storage_flow"`, the flow into each junction's storage (m³/s); it is an output, read by
nothing.

The rest of this section follows the physics in the order it enters the solve: the
[equations](#equations) and how they are [solved](#what-is-solved) (iteration, Anderson
acceleration and the flow tolerance it needs); the three places where SWMM's own iteration has
no fixed point and noodl takes a documented tie rule (the [crown hold band](#surcharge-slot-and-extran) of EXTRAN, the
[normal-flow slope switch](#normal-flow-slope-switch) and the [drop-end area switch](#drop-end-area-switch));
the [starting state](#starting-state-and-zero-fall); [surcharge](#surcharge-slot-and-extran), [flooding](#flooding),
[storage units](dynamic-wave-structures.md#storage-units), [orifices, weirs and outlets](dynamic-wave-structures.md#orifices-weirs-and-outlets) and
the [conduit options](dynamic-wave.md#conduit-options); the [variable step and time-varying inflow](dynamic-wave.md#variable-step-and-time-varying-inflow),
[tidal and time-series outfalls](dynamic-wave.md#tidal-and-time-series-outfalls); [water quality](headspace-sulfide.md#quality-on-the-dynamic-wave) and
[headspace air](headspace-sulfide.md#headspace-air-on-the-dynamic-wave); and last the
[intentional differences from SWMM](#intentional-differences-from-swmm). What is not supported is
listed once, under [Limitations](limitations.md#what-is-not-supported).

## Equations

SWMM 5.2.4's conduit and junction updates (`dwflow.c`, `dynwave.c`) are solved at their fixed
point rather than replayed trial by trial. For one barrel of conduit length $L$ (m) between
heads $h_1$ and $h_2$ (m), after a step $\Delta t$ (s):

$$
q = \frac{q_{\text{old}} + \Delta q_3 + \Delta q_4}{1 + \Delta q_1}
  + \frac{g\,\Delta t\,A_w}{L\,(1 + \Delta q_1)}\,(h_1 - h_2)
$$

where $q_{\text{old}}$ is the flow at the step start (m³/s), $\Delta q_1$ the friction and
$\Delta q_3$, $\Delta q_4$ the convective and area-change momentum terms, and $A_w$ the
weighted flow area (m²) that SWMM forms from the end and mid-conduit areas. The coefficients
are evaluated at the converged depths and flow. SWMM's two post-conditions are applied
afterwards: the normal-flow limit (`NORMAL_FLOW_LIMITED`, `SLOPE`/`FROUDE`/`BOTH`/`NO`), and the
dry-node clamps, which zero a flow that would draw from a junction with no water. At a junction
that is not an outfall, with surface area $A_s$ (m², the sum of the surface areas of the
conduits meeting there, plus a storage unit's own area, floored at `MIN_SURFAREA`) and depth
$y$ (m):

$$
A_s\,(y - y_{\text{old}}) = \tfrac12\,\Delta t\,(N_{\text{old}} + N)
$$

with $N$ the net inflow (m³/s: lateral inflow plus the conduit flows into the junction).
Outfalls are boundary heads: `FREE` takes the lesser of normal and critical depth in the
outfall conduit, `NORMAL` its normal depth, `FIXED` the stage with SWMM's stage logic, each
from the flow in that conduit. `TIDAL` and `TIMESERIES` outfalls use the same stage logic with
a stage that varies in time (see [Tidal and time-series outfalls](dynamic-wave.md#tidal-and-time-series-outfalls)).

## What is solved

For frozen coefficients the conduit and junction equations are linear in the heads apart from
the min-clamps, so one pass is a single `KirchhoffLayer` solve. The model then refreshes the
coefficients and repeats until the heads change by less than `iterate_tol` (m) and the flows
by less than `iterate_flow_tol` (m³/s), or `iterate_max` passes are used
(`Model(coupling="iterate")`, converging on the layer's own interior head; the defaults follow
the file, see [Iteration and convergence](#iteration-and-convergence)). The result is the head
and flow field at which a SWMM trial reproduces its own input.

The passes are Anderson-accelerated by default (`noodl.apps.sewer.build_dynamic_wave_model(net,
acceleration="anderson", memory=5)`; `Model`'s `iterate_acceleration`,
see [Layers and models](../../concepts/layers.md)). Each pass's flows must then also reproduce the
flows it was fed to within `iterate_flow_tol` (m³/s): the head solve stops on its warm start once
every junction balance is within its `atol` (`solve_atol`), so near convergence the heads stand
still while the flows of a flat or nearly dry conduit still drift. Against plain relaxation judged on the heads alone the
mean passes per step fall from 34.8 to 14.1 (tree), 30.2 to 10.4 (backwater line) and 20.8 to 9.7
(loop) over the first 60 steps, and from 42.8 to 11.3 over the loop's first 120, whose worst step
drops from 327 passes to 22. The two iterations land on the same fixed point: with a tight head
solve (`atol=1e-12`) they agree to 2e-12 m; at the default `solve_atol` each lands within the head
solve's own resolution, `atol dt / (2 A_s)` (a few 1e-9 m). The mixing keeps to the fixed points
plain relaxation (and SWMM's relaxed trials) can reach: a flat, near-dry conduit's flow can have
a second, repelling fixed point, and a mixed step heading for it is replaced by the relaxed step.
`acceleration=None` is plain relaxation. The flows are judged either way, never the heads
alone, because on the normal-flow slope switch (below) the heads bound the flows' residual only
weakly, and a heads-only test was measured stopping 3e-5 m³/s short of the fixed point.

**SWMM's iteration artefacts are not modelled**: trial relaxation, the 0.001 cfs flow-reversal
rule and link bypassing. Each is a device for reaching the fixed point and none moves it.

## Iteration and convergence

`HEAD_TOLERANCE` and `MAX_TRIALS` are read into `DynamicWaveOptions` and are the defaults of
the iteration (`noodl.apps.sewer.dynwave.iterate_tolerances`); an explicit `iterate_tol`,
`iterate_flow_tol` or `iterate_max` overrides each:

| control | default | SWMM |
|---|---|---|
| `iterate_tol` (m) | `HEAD_TOLERANCE` (the file's length unit), else 0.005 ft = 0.001524 m | the same test: every node's depth change between trials (`dynwave.c::findNodeDepths`) |
| `iterate_flow_tol` (m³/s) | $2 A_{\min} \epsilon_H / \Delta t$ ($\epsilon_H$ = `iterate_tol`), at least 1e-10 | none: SWMM judges the heads alone |
| `iterate_max` (passes) | `MAX_TRIALS`, else 8 (at least 2) | the same limit |

The flow tolerance is the flow residual that matches the head tolerance: a flow error $dQ$
moves a junction's next depth by $\Delta t\, dQ / (2 A_s)$ (the trapezoid), so at the
surface-area floor $A_{\min}$ (`MIN_SURFAREA`, 12.566 ft² by default) and the routing step
$\Delta t$ it is a move of exactly `iterate_tol`, and less at every larger junction. At SWMM's
defaults on a 5 s step that is 7.1e-4 m³/s.

A step that uses up the file's trials is **accepted**, as SWMM accepts it (it counts it as "not
converging", `NonConvergeCount`): the step is taken with its last pass, its diagnostics say
`converged` False, and `simulate_dynamic_wave` returns `out["converged"]` per step, so
`(~out["converged"]).sum()` is SWMM's count. With an explicit `iterate_max` such a step is
refused by name, as before, and a step that has to be differentiated is always refused when it
does not converge (there is no fixed point to differentiate).

**The head solve within a pass.** Every law of a pass is non-decreasing in its own head
difference (conduits, regulators) or head (junction withdrawals), so the head solve's residual is
the gradient of a convex function, and the layer's Newton runs with `line_search="convex"` and
`step_floor=16` (`dynwave.HEAD_SOLVE_NEWTON`; see [Solvers](../../concepts/solvers.md#newton)). The
first cuts back a step that overshoots across a kink: the stiff bed below a dry junction's
invert met by a fast inflow, the normal-flow switch's flat-steep-flat flow, the hold band and
flood wall. A damped Newton step there could fall into a 2-cycle it never left, and the run
stopped with "failed to converge after 50 iterations" (13 files of the SWMM corpus did, among
them a junction fed down a 20 m drop and a looped siphon). The second ends a solve whose residual
sits at the round-off of its own terms above `solve_atol` (a junction flooding 17 m above its
outfall behind a 0.1 m³/s cap, or a large storage area at a high head), which failed every step
before. A pass whose head solve still stops short is one iterate of the pass iteration, not the
end of the step: it hands back its last Newton iterate, is never judged converged, and the
iteration goes on from it (`Model`, ``on_failure="return_primal"``); a differentiated solve
still raises. Both kinds are compared with SWMM (`robust_orifice_dynwave.inp` at every step,
`robust_drop_dynwave.inp` until SWMM's own first non-converging step).

Measured at SWMM's defaults (the fixtures with `HEAD_TOLERANCE` and `MAX_TRIALS` removed, 30
min), against SWMM 5.2.4's own 2.02 trials per step on the tree:

| network | passes per step | steps at the limit | worst depth against the tight fixed point (m) | SWMM at its defaults, same measure (m) |
|---|---|---|---|---|
| tree | 2.06 | 1 / 360 | 1.6e-3 | 2.3e-3 |
| backwater | 2.22 | 1 / 360 | 6.9e-3 | 1.3e-2 |
| loop | 2.01 | 0 / 360 | 7.0e-4 | 3.6e-3 |
| structures | 2.39 | 0 / 360 | 4.4e-4 | 5.9e-3 |
| storage | 2.06 | 1 / 360 | 1.0e-3 | 2.3e-3 |

So noodl at SWMM's tolerances is at least as close to the converged answer as SWMM is, in about
SWMM's number of trials. A judgement on the flows at 1e-4 m³/s instead takes 2.54 passes on the
tree for a 7.8e-5 m answer; the parity tests pass their own tight tolerances (1e-13 m, 2000
passes) and the fixtures set `HEAD_TOLERANCE` 1e-12 and `MAX_TRIALS` 500.

**Cost.** One pass costs a nearly fixed few thousand small tensor operations, whatever the
network's size, so a large network is not much slower per pass than a small one. Most of the
cost is the conduit update, SWMM's normal and critical depths, and the head solve, whose Newton Jacobian (and the implicit
adjoint's) is the elements' own analytic `dflow`: every sewer law's `dflow` is the exact
derivative of its flow (`exact_slope`, [Writing a constitutive law](../../concepts/elements.md)). A batch of B scenarios (`simulate_dynamic_wave`
with a `(B, n)` inflow) costs about the same per pass, so an ensemble amortises it.

## Bed stiffening and lagged pieces

SWMM clamps a junction depth at 0 after the update. `JunctionStorage` stands in for that with a
stiff bed term, $K_{\text{bed}} = 10^{6}$ times the storage slope applied to
$\min(H - z_{\text{bed}}, 0)$, built on `MIN_SURFAREA` rather than on the pass's $A_s$. The
slope of a junction's balance is therefore positive everywhere, so each junction is grounded
by itself and the Jacobian stays symmetric positive definite. The default Newton tolerance
`solve_atol` (m³/s) is scaled per model to the round-off floor of that term (about
2.5e-9 m³/s, or about 5e-9 m of head, on the committed fixtures; it grows with invert
elevation).

Some quantities are taken from the previous pass, as SWMM takes them from the previous trial,
and are consistent at the fixed point: the flow class and the weighting $f_{\text{asnh}}$, the
surface areas $A_s$, the normal-flow-limit flag (its Froude test uses the momentum flow at the
fed heads; its slope test is applied within the pass where it can be, see the next section),
the normal flow, the dry-node flags, the outfall heads (the surface areas' jump at a drop
end is applied within the pass where it can have no fixed point, see "Drop-end area
switch"), and the head at any
conduit end SWMM pins (the head is raised to the invert plus offset when it lies below it, and
set to critical depth in `UP_CRITICAL`/`DN_CRITICAL`). A conduit with a pinned end has no head
dependence there: its conveyance is zero and the pinned head enters the constant term.

## Normal-flow slope switch

SWMM limits a conduit's flow to the normal flow $q_{\text{norm}}$ at its upstream depth when its
slope test $y_1 < y_2$ passes (`dwflow.c:637-686`; always on for a conduit at an outfall, and
under `NORMAL_FLOW_LIMITED SLOPE` or `BOTH`). With both end depths inside their clamps that
test is $h_1 - h_2 < z_1 - z_2$ (the conduit-end inverts), a switch on the conduit's own head
difference, so `DynamicWaveConduit` applies it within the head solve instead of from the
previous pass:

$$
q = \min\!\big(q_{\text{lin}},\ \max(q_{\text{norm}},\ q_{\text{last}} + S\,(h_1 - h_2 - z_1 + z_2))\big)
$$

with $q_{\text{lin}}$ the momentum flow above, $q_{\text{last}}$ the previous pass's flow and
$S$ (m²/s) a wall slope. At a fixed point ($q = q_{\text{last}}$) this is SWMM's unlimited flow
above the switch and its limited flow below it, so every SWMM fixed point is unchanged. It has
one more kind of fixed point: on the switching surface $y_1 = y_2$, with the flow between
$q_{\text{norm}}$ and $q_{\text{lin}}$. That is the tie rule for the case where SWMM has no
fixed point, which a draining network reaches at its outfall conduit. A `FREE` or `NORMAL`
outfall's depth is set from its conduit's flow, and the limited flow is Manning's normal flow
at the upstream depth $y_1$, at which the outfall's normal depth is $y_1$ again, up to the
interpolation mismatch of SWMM's tables (about 1e-5 m). Once the junction upstream has fallen
to the outfall's depth, the limited side is consistent only where that mismatch happens to
put the outfall above the junction, and the unlimited side's larger flow raises the outfall
above it. SWMM's trials then alternate around the surface: on the tree fixture with every
inflow ramped down, SWMM 5.2.4 at `HEAD_TOLERANCE 1e-12` m and 500 trials fails to converge on
82 of the 108 steps after the junction meets the outfall depth, its last trial within about
6e-4 m of the surface on either side. A flag lagged by one pass chattered there until the pass
budget ran out. noodl converges on every step: on SWMM's fixed point where one exists (it
found the limited ones the lagged flag missed), and exactly on the surface where none does.
The cost against SWMM is confined to those steps, and it is bounded by SWMM's own oscillation,
since SWMM has no converged answer there; from SWMM's first such step on its history is that
of an unconverged trial, so later steps differ by that much too (asserted to 1e-3 m and
5e-4 m³/s in `test_ramp_down_matches_swmm_until_swmm_stops_converging`; before that step the
targets hold, 7e-10 m).

The wall slope does not move a fixed point; it shapes the passes. Where the outfall depth
follows the conduit's flow, $S = 1/(c + 1/S_{\max})$ with $c$ the outfall depth's rise per
unit flow, so the head solve sees the switching surface move with the flow as it does at the
fixed point. $S_{\max}$ is the smaller end junction's grounding (storage slope $2A_s/\Delta t$
plus its conduits' conveyance; `K_SWITCH` = 1). A steeper wall would turn the flow into an
S-curve in the head that the damped Newton head solve cycles on. The Froude test
($\mathrm{Fr} \ge 1$ on the momentum flow) and a slope test at a clamped end depth stay
lagged flags.

## Drop-end area switch

A conduit end with an offset gives its junction no surface area while that end is
`DN_CRITICAL` (downstream, end depth below $\min(y_N, y_C)$ on forward flow) or `UP_CRITICAL`
(upstream, on reverse flow), and none while the end is dry below a wet other end. Above that
lower threshold the end gives the junction a quarter of the conduit, scaled downstream by
$f_{\text{asnh}}$, which falls from 1 to 0 up to $\max(y_N, y_C)$ and jumps back to 1 there
(`dwflow.c:357-411, 472-499`). The lower threshold is $\max(\mathrm{FUDGE}, \min(y_N, y_C))$
where the flow leaves the conduit at that end, and FUDGE elsewhere. So the junction's area
jumps up with its own depth at both thresholds. While the junction **fills**, its storage
equation $A_s (H - H_{\text{start}}) = \tfrac12 \Delta t (N_{\text{start}} + N)$ can then
land above the threshold with the lower area and below it with the upper one: there is no
fixed point. SWMM 5.2.4 does not converge on such steps (its trials cycle around the
threshold), and an area lagged by one pass chattered until the pass budget ran out, or until
the head solve failed when the class also moved the other end's area. While the junction
drains the jump runs the other way, a solution always exists, and the lagged area settles as
SWMM's does.

`DynamicWaveHydraulics` finds, for each junction, the threshold of its attached drop ends
nearest its head, and where it lies above the step-start head gives `JunctionStorage` the
storage lines with the area just below ($A_{\text{lo}}$) and just above ($A_{\text{hi}}$) it.
The junction then withdraws, within the head solve,

$$
w = \max\!\big(w_{\text{lo}}(H),\ \min(w_{\text{hi}}(H),\ N_{\text{last}} + S\,(H - H^*))\big)
$$

with $N_{\text{last}}$ the previous pass's net inflow and $S$ a wall slope (`K_SWITCH` times
the junction's grounding above the threshold). At a fixed point ($w = N = N_{\text{last}}$)
this is SWMM's storage with the lower area below $H^*$ and with the upper area above it, so
every SWMM fixed point is unchanged; where neither side holds, the junction sits exactly on
the threshold, $H = H^*$, with its withdrawal inside the jump. That is the tie rule. A
junction held there (its fed net inflow inside the jump, at the fraction $\theta$ of it) has
the threshold's conduit end classified as SWMM classifies it at the threshold (SWMM's tests
are strict, so the upper side: not critical, wet, $f_{\text{asnh}} = 1$ at the upper
threshold), and, at the lower threshold, the conduit's other end gets the sliding area
$(2 - \theta)$ quarters of the conduit: half (the critical side's) at $\theta = 0$, a quarter
(the upper side's) at $\theta = 1$. Holding only the junction's own area was not enough: on
`options_dynwave.inp` with a 0.03 m outlet offset on C3, C3's class also moves J3's area,
J3's head moves C3's flow and with it the threshold, and the passes cycled.

On three variants of `options_dynwave.inp` (a 0.03 m or 0.1 m outlet offset on C3, or C6
without its flap gate) every step converges, with Anderson and with plain relaxation. In
noodl's own runs the rule holds a junction only on the steps where SWMM's run first fails
(0.03 m: 60 and 62; 0.1 m: 107; gate off: 106 and 129), nowhere else over the 359 steps.
SWMM also fails on step 167 of the 0.03 m variant; that step lies in SWMM's own history,
which differs from noodl's after step 60. Started from SWMM's state before it, noodl holds
J5 on its threshold there too (see [Verification: sewers](../../reference/verification/sewer.md#dynamic-wave-against-swmm-in-detail)).

## Starting state and zero fall

The start follows SWMM's `flowrout.c`. A cold, dry network: junction depth 0, each outfall at its
outfall depth for zero flow (the stage for `FIXED`, and its stage at t = 0 for `TIDAL` and
`TIMESERIES`), each conduit's depth the mean of its two
clamped end depths, and a previous net inflow of 0 at the first step. Initial conditions change
that exactly as SWMM does (`initNodeDepths`, `initLinkDepths`, `initNodes`):

- A conduit with an `InitFlow` (`Pipe.initial_flow`, m³/s, a total over its barrels, positive
  from the pipe's first node to its second) starts at the normal depth of one barrel's flow,
  capped at the depth of SWMM's `qMax`, and keeps that flow as its previous flow. The reader
  flips the sign of the `InitFlow` of a conduit it reverses, as SWMM does.
- A junction with an `InitDepth` (`Manhole.initial_depth`, m) starts there. One without it
  starts at the mean, over its attached conduits, of each conduit's initial depth plus that
  conduit's first-end offset (a conduit without `InitFlow` counts as depth 0 in the mean).
  An `InitDepth` above the full depth plus `SurDepth` is refused by name, as SWMM's own check.
- The outfalls start at the depth SWMM gives for their conduit's initial flow, a conduit
  without `InitFlow` at the mean of its end-node depths above its offsets, and each junction's
  previous net inflow is that of the initial flows through it (minus an outflowing `InitFlow`),
  so the first step's trapezoid sees them.

SWMM orients a conduit with exactly
zero fall by round-off under `DYNWAVE`, so noodl can legitimately differ on an exactly flat
conduit; the backwater fixture avoids an exact tie for that reason.

## Surcharge: SLOT and EXTRAN

`SURCHARGE_METHOD SLOT` is SWMM's Preissmann slot: above `0.985257 y_full` the top width is the
Sjoberg slot width `0.5423 w_max exp(-(y/y_full)^2.4)` (`0.01 w_max` above `1.78 y_full`), the
area above the crown is `a_full + (y - y_full) w_slot`, the friction radius stays `r_full`, a
full conduit takes no inertial terms, and the reported link depth stays `min(y_mid, y_full)`
while the stored volume may exceed the full volume. The junction equation is unchanged (SWMM
never takes its node-surcharge branch under SLOT; the junction surface-area floor governs a
pressurised junction).

`SURCHARGE_METHOD EXTRAN` (the default) caps each conduit's end depths at its crown while the
end heads keep driving a full conduit, which takes no inertial terms. A junction whose depth
exceeds its crown -- the highest `offset + y_full` over the conduit ends it carries -- is
surcharged: SWMM's update `y_new = y_last + corr dQ/denom` (`dynwave.c:716-741`) is converged
exactly when the junction's net inflow `dQ` is zero, so a surcharged junction stores nothing
and holds instantaneous continuity (`corr` and SWMM's `exp(-15 f)` blend only shape SWMM's
iteration). The surcharge flag is evaluated from the previous pass's depth, as SWMM evaluates
it from its previous trial, and it drops when the solution falls below the crown. Where SWMM
has no fixed point -- the storage equation would end above the crown while at the crown the
junction already loses water, so SWMM's trials alternate between its two branches and do not
converge -- noodl holds the junction at its crown (within a band below a millimetre) and
releases the water the balance needs; every step then converges. `sewer.surcharged` reports
the junctions above their crowns (SWMM's test): a junction held at its crown by that rule
reports 1 without holding zero net inflow. A surcharged junction's head is resolved to
`atol / sum(k)` with `sum(k)` its conduits' conveyance (~0.01 m2/s for a 0.3 m pipe), so the
default head-solve tolerance (`solve_atol`, a few 1e-9 m3/s) leaves it a few 1e-7 m from the
fixed point; pass `atol=1e-12` to `simulate_dynamic_wave` for tighter heads where no junction
sits on its bed.

## Flooding

Flooding follows SWMM without ponding (`ALLOW_PONDING NO`, or a junction without a ponded
area; [Ponding](#ponding) below, `dynwave.c:743-795`). A junction's
full depth is its `MaxDepth`, raised to the top of every conduit end it carries (so a
`MaxDepth` of 0 means the crown, `link.c:440-463`), and it floods above
`y_max = full depth + SurDepth` (`Manhole.surcharge_depth`). A flooded junction is held at
`y_max`, and the water that cannot enter it leaves the system: `sewer.overflow` (m³/s per
junction, `record=("sewer.overflow",)` in `simulate_dynamic_wave`) is SWMM's flooding loss,
the step's trapezoidal net inflow `0.5 (N_start + N)`, set to 0 below SWMM's 1e-4 cfs. It does
not return to the network and is not part of the net inflow carried to the next step, as in
SWMM. In the head solve every junction's withdrawal is capped by a steep "wall" at `y_max`
offset by the previous pass's excess of net inflow over what the junction would otherwise
take there; at the fixed point that offset is the solved excess, so a flooded junction sits
exactly at `y_max` (not at a penalty's offset above it), and the excess falling to zero
releases it on the next pass, without a flag to chatter.

## Ponding

With `DynamicWaveOptions.allow_ponding` (`ALLOW_PONDING YES`) a junction with a positive
`Manhole.ponded_area` (SWMM's Aponded, m²) does not flood (`dynwave.c:661, 738-748, 753, 766-795`):
the water above its full depth stays, spread over `ponded_area` plus the conduit ends' surface
areas (`node_getPondedArea`, `dynwave.c:309-312`), and `SurDepth` plays no part (`yMax` is the full
depth, `dynwave.c:748`). SWMM takes the area from the previous trial's depth, so the junction
is the drop-end area switch of the head solve with its threshold at the full head: the line with
the conduit areas below it, the line with the ponded area added above it, and, from a step start
below the full head, the junction held there where the lower line ends above it and the upper
one below it (SWMM's trials alternate about the full depth there); from a start at or above it
the line of the previous pass's side. A junction above its full depth does not surcharge
(`dynwave.c:679-680`) and has no hold band; the flood wall of the previous section acts only on a
ponding junction that is surcharged, at the full head plus SWMM's `FUDGE` depth (a newly ponded
node rises barely above its full depth in the surcharged update, `dynwave.c:738-740`, and
continues in the ponded one). A junction whose full depth is its crown has no surcharge zone.
`sewer.overflow` of a ponding junction is the step's trapezoidal net inflow while it ends above
its full depth (set to 0 below SWMM's 1e-4 cfs), which is the rate of SWMM's ponded-volume
growth (`dynwave.c:787-792`); the ponded water stays in the network.

The tie is the only place the model and SWMM can differ at the level of SWMM's own
iteration: on the step where a rising junction crosses its full depth with the lower line ending
above it and the upper one below it, SWMM's trials alternate about the full depth, stop wherever
`MAX_TRIALS` or the head tolerance leaves them (a few `FUDGE` above or below), and the next steps
inherit that depth; the model holds the junction exactly at the full depth. Ponding is refused
together with water quality or headspace air (the ponded water would have to stay in the transport
volume), and for a storage unit (SWMM deprecates its ponded area).

## Intentional differences from SWMM

Everywhere else the dynamic wave reproduces SWMM's fixed point (the parity numbers are under
[Dynamic wave against SWMM](../../reference/verification/sewer.md#dynamic-wave-against-swmm-in-detail)). These differences are deliberate, and
each is confined to the cases named.

- **Tie rules where SWMM has no fixed point.** SWMM's trials alternate between two branches
  on these steps, so SWMM does not converge (it reports it) and has no answer to compare with.
  noodl converges on every step, and its answer lies on the switch between the branches:
    - the EXTRAN crown hold band: a junction whose storage equation would end above its crown
      while it loses water at the crown is held within a millimetre of the crown
      ([Surcharge](#surcharge-slot-and-extran));
    - the normal-flow slope switch: a draining outfall conduit sits on $y_1 = y_2$
      ([Normal-flow slope switch](#normal-flow-slope-switch)). Cost: 1e-5 m of SWMM's
      table mismatch at the switch, and steps after SWMM's first unconverged one differ by
      SWMM's own oscillation (about 6e-4 m on the tree);
    - the drop-end area switch: a junction filling through a drop end's area threshold is held on
      the threshold ([Drop-end area switch](#drop-end-area-switch)), with histories then
      differing by SWMM's oscillation (3e-4 m on the options fixture).

  At a gradient the tie rule is what is differentiated: the sensitivity of the constrained
  solution, where SWMM has no fixed point.
- **A surcharged junction drained only by outlets.** SWMM's surcharged update divides by
  the attached links' `dqdh`, which an outlet leaves 0, so above 1.25 crowns SWMM freezes the
  junction's depth and loses the inflow it cannot pass. noodl keeps instantaneous continuity: the
  outlet passes what arrives. Such a junction has no SWMM reference; drain an outlet from a
  storage unit, as SWMM's own examples do.
- **Water-quality transients.** noodl's reactors are the junctions' control volumes and SWMM's
  the conduits, so in transit the tracer arrives shifted by the difference of the mean delays;
  steady concentrations agree (see [Quality on the dynamic wave](headspace-sulfide.md#quality-on-the-dynamic-wave)).
- **Iteration artefacts.** Trial relaxation, link bypassing and the 0.001 cfs reversal rule are
  not modelled; `HEAD_TOLERANCE` and `MAX_TRIALS` are the iteration's defaults, with a flow
  tolerance SWMM does not have (see [Iteration and convergence](#iteration-and-convergence)).
  SWMM at its default tolerance is off its own fixed point by centimetres on a backwater, and
  by up to metres on some corpus files (see [the SWMM corpus](../../reference/verification/sewer.md#dynamic-wave-against-swmm-in-detail)),
  which noodl, solving to the fixed point, does not reproduce.
- **Refused where SWMM accepts.** A `FUNCTIONAL` storage exponent below 0 (infinite area at zero
  depth), a negative storage `MaxDepth`, `InitDepth` or `SurDepth`, a nonzero pollutant
  `Cinit`, and the sections listed under [Limitations](limitations.md#what-is-not-supported).
- **One call is one run under a variable step.** A second call to `simulate_dynamic_wave`
  restarts the variable step as a new SWMM run does.
