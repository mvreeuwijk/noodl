# Sewers

Gravity sewer networks, with the water, the air above it and the sulfide it carries coupled on one
network. Manholes are nodes and circular pipes are edges, draining as a tree to an outfall;
wastewater flows downhill under gravity, and the question the model answers is corrosion and
odour risk: how much H₂S is generated, and how much reaches the air where it attacks the crown
of the pipe.

Three physical subsystems share the network, each conserving its own quantity at every manhole —
water volume, headspace air, and BOD and sulfide mass — and each uses a different part of the
framework:

- **Water hydraulics** — partially full circular pipes under gravity. Under kinematic-wave
  routing, Manning's equation on a dendritic network: the cycle space is empty, so continuity
  alone fixes every discharge in closed form, with no potential solve at all. Under
  dynamic-wave routing, SWMM's momentum equation as a potential layer on head (see
  [Dynamic wave](#dynamic-wave)), which admits backwater, flat conduits and loops.
- **Headspace air** — the air above the water surface is its own flow network, a full
  `PotentialFlowLayer` driven by wall friction, the drag of the moving water surface, and
  buoyancy from the sewer-to-ambient temperature difference, vented through manhole leaks and
  optional fans.
- **Water quality** — BOD decay and Pomeroy–Parkhurst sulfide generation in the wastewater, with
  two-film Henry's-law transfer of H₂S into the headspace air, as two coupled transport layers.

**Units.** Water and headspace air flow m³/s, depth and head m, pressure Pa, concentrations
kg/m³ (with helpers to mg/L and ppm), sulfide transfer kg S/s.

```python
from noodl.apps.sewer import (
    SewerNetwork, Manhole, Pipe, Outfall, tree_steady,
    build_model, sewer_steady, initial_state,
    pipe_table, to_ppm, to_mg_per_litre,
)
```

![A sewer pipe carrying water below and headspace air above, and a dendritic network draining to an outfall](../assets/app-sewer.svg)

Every type this application offers — what it does and how to call it — is listed in its [catalogue](../catalogue/sewer.md). It is set up and run like every other model; see [Using noodl](../usage.md).

## A worked example

```python
from noodl.apps.sewer import build_model, sewer_steady, pipe_table, tree_steady

model, state, drivers = build_model(tree_steady())
final = sewer_steady(model, state, drivers)
rows = pipe_table(model, final, drivers, path="pipes.csv")

# rows[0] is pipe "C1": q = 0.05 m3/s, h = 0.153007001 m
```

*(From `tests/apps/sewer/test_report.py`.)* `tree_steady()` is the committed 5-conduit, 6-node
fixture; its steady discharges are 0.05, 0.08, 0.03, 0.13 and 0.16 m³/s.

Start from a different state by naming it — here, sulfide already present in manhole
`J1` — and read the result by manhole and species:

```python
from noodl.apps.sewer import initial_state

wq = model.refs.water_quality                  # species ("bod", "sulfide"), kg/m3
start = initial_state(model, drivers=drivers, values={wq.concentration: {"J1": {"sulfide": 1e-3}}})
model.check(start, drivers).raise_for_errors(strict=True)
after = model.step(start, drivers, 60.0)
print(sorted(wq.concentration.named(after[wq.concentration])))
```

```text
['J1', 'J2', 'J3', 'J4', 'J5']
```

The sewer's `initial_state` takes the drivers as well (`drivers=`), because the storage it starts from
depends on the inflows. In the dictionaries the concentrations are
`"water_quality.x"` and `"air_quality.x"` (manhole order, species last); see
[Using noodl](../usage.md).

A model can also be read from a SWMM `.inp` file (see [File formats](../formats/swmm.md)):

```python
from noodl.apps.sewer import read_swmm_inp, build_model

net, inflows, pollutants = read_swmm_inp("tree_steady.inp")
model, state, drivers = build_model(net)
```

For a single step rather than a steady state:

```python
new = model.step(state, drivers, 60.0)
residuals = model.residuals(new, drivers)     # per-layer nodal balance
```

## Building a network

| Object | Fields |
|---|---|
| `Manhole(name, invert, ground=None, inflow=0.0, surface_area=None, surcharge_depth=0.0, initial_depth=0.0, storage=None)` | A junction. `surface_area` defaults to SWMM's `MIN_SURFAREA`, 1.167 m² — a 4 ft shaft. `surcharge_depth`, `initial_depth` and `storage` are dynamic-wave physics (see "Dynamic wave"). |
| `storage_unit(name, invert, max_depth, curve, *, surcharge_depth=0.0, initial_depth=0.0, inflow=0.0)` | A SWMM storage unit: a `Manhole` whose `storage` is the surface-area `curve` (`("functional", a, b, c)` for `A = a y^b + c`, `("tabular", depths, areas)`, or `("cylindrical", L, W)`, `("conical", L, W, Z)`, `("paraboloid", L, W, H)`, `("pyramidal", L, W, Z)`; m and m²) and whose ground is `invert + max_depth`. Dynamic wave only. |
| `Pipe(name, u, v, length, diameter, n, slope, inlet_offset=0.0, outlet_offset=0.0)` | A circular gravity conduit; offsets (m) are the invert heights above the end nodes' inverts. |
| `Outfall(name, invert, kind="free", stage=None)` | The root of a tree. `kind` is `"free"`, `"normal"` or `"fixed"` (`stage`, m, required for `"fixed"`); only dynamic-wave routing reads them. |
| `SewerNetwork(manholes, pipes, outfalls, routing="kinematic_wave", notes={}, dynamic_wave=DynamicWaveOptions())` | The whole network. `routing` is `"steady"`, `"kinematic_wave"` or `"dynamic_wave"`; SWMM's `STEADY`/`KINWAVE`/`DYNWAVE` are accepted and normalised. |

`validate()` refuses, by name: duplicate node or pipe names; a pipe naming an unknown node;
non-positive slope, diameter, length or roughness; **a manhole with anything other than exactly
one outgoing pipe**; an outfall with an outgoing pipe; and a manhole whose component never
reaches an outfall — which also catches cycles. With `routing="dynamic_wave"` these tree
rules are replaced: loops and flat pipes are allowed, every connected component needs an
outfall, and each outfall ends exactly one conduit.

That one-outgoing-pipe rule is the dendritic constraint, and it is what makes the closed-form
flow solve valid.

## `build_model`

```py
model, state, drivers = build_model(
    net,
    storage=False,          # implicit-Euler manhole storage sweep
    air=True,               # the headspace layer
    quality=True,           # the two quality layers
    species=("bod", "sulfide"),
    f_air=0.02, f_i=7.49e-4, c_s=1.0,
    leak_area=8e-4, leak_cd=0.6, fans=(),
    coupling="pingpong", scheme="implicit", dt_storage=None,
)
```

`air=False` gives the water-only model the SWMM parity rows use. `fans` names manholes carrying
a prescribed extraction (m³/s, positive out).

`build_model` dispatches on `net.routing`. A `"dynamic_wave"` network is built by
`build_dynamic_wave_model`: water only, with `iterate_tol` (m, default 1e-11) and
`iterate_max` (default 2000 passes) controlling the head iteration. For it, `air`, `quality`
and `storage` default to `False` and `True` is refused, naming the option; `iterate_tol` and
`iterate_max` are refused for the other routings. Run it with `simulate_dynamic_wave` (see
[Dynamic wave](#dynamic-wave)), not `sewer_steady`.

`dt_storage` (optional) declares the interval the manhole storage sweep integrates over. When given,
a step at any other interval is refused by name. Omit it (the default) to follow the model's own
step interval via the `StepContext` the closure receives.

`initial_state(model, drivers=None)` builds the all-zero state, dispatching on each layer's
quantity. When `drivers` is given, it also evaluates the initial storage (by querying
`model.initial_capacities`), which a step with a changing capacity requires to be in the step-start
state. Drivers are required for initial_state whenever the model has a quality layer, because the
hydraulics closure writes those layers' capacity from the flows and the initial storage must be
evaluated from the drivers; this holds with or without `storage=True`.

`sewer_steady(model, state, drivers, *, reaction=None, dt=60.0, max_iter=500, tol=1e-12)` steps
repeatedly until every transport layer stops changing. You need it rather than `Model.steady`
whenever quality is on, because `Model.steady` never applies reactions.

`model._apply_closures(state, drivers)` resolves the hydraulics without stepping any layer,
which is the quickest way to inspect `sewer.q`, `sewer.h`, `sewer.v` and the rest.

## The physics

### Exact circular geometry

With the wetted half-angle $\theta = 2\arccos(1 - 2h/D)$:

$$
A = \frac{D^2(\theta - \sin\theta)}{8},\quad
P = \frac{D\theta}{2},\quad
R = \frac{A}{P} = \frac{D}{4}\left(1 - \frac{\sin\theta}{\theta}\right),\quad
T = D\sin\frac{\theta}{2}
$$

These exact relations are the default for networks built in code (`SewerNetwork(...,
geometry="analytic")`).

### SWMM's own circular geometry

SWMM 5 does not evaluate the circle. For a `CIRCULAR` conduit it interpolates 51-point tables
(`A/Afull`, `Y/Yfull` and the section factor `S/Sfull`, from `xsect.dat`) and near the invert
switches to closed forms solved by Newton iterations with their own starting guesses and stopping
rule (`xsect.c`). The tables differ from the true circle by up to ~5e-3 in depth, and by far more
in velocity near the invert. `geometry="tabulated"` (module `noodl.apps.sewer.xsect_tables`) reproduces
all of it, operation for operation, and is what `read_swmm_inp` uses by default, because a
network read from a SWMM file is read to reproduce SWMM (`read_swmm_inp(path,
geometry="analytic")` gives the exact circle).

At a kinematic-wave steady state SWMM gives a conduit carrying $q$ an inlet area
$a_1 = A(S = q/\beta)$ (inverse section-factor lookup) and an outlet area $a_2$ solving
$\beta\,S(a_2) = q$. It reports depth $\tfrac12(Y(a_1) + Y(a_2))$, volume
$\tfrac12(a_1 + a_2)L$, and velocity $q / A(\text{depth})$, which is zero at a depth of 0.01 ft
or less. $a_1 = a_2$ on the tables. Near the invert the two Newton solves differ slightly, and
both are carried. SWMM's units are reproduced too: it computes in feet with Manning's 1.486, but
converts CMS flows and volumes with 0.02832 m³/ft³ rather than $0.3048^3$. That is an effective
1.6e-4 on the Manning coefficient and 1.2e-4 on reported volumes. Everything is differentiable:
piecewise-linear tables, unrolled Newton loops, and one implicit-function root.

Scope: `geometry="tabulated"` covers the steady kinematic-wave path only, and is refused by name with
`storage=True`. The discharge limit is SWMM's `Qfull` rather than the analytic 0.938 D capacity.
The headspace, top width, hydraulic radius and mean depth (air side, sulfide) remain the exact
circle, evaluated at SWMM's depth. SWMM has no headspace to compare them with.

Numerical care worth noting: the `arccos` argument is clamped to the *exact* domain $[-1, 1]$,
not an epsilon-shrunk one — that was tried and rejected because it shifts the boundary value by
$O(\sqrt{\varepsilon})$. Every $0/0$ site is guarded on **both** branches of its `torch.where`,
so a masked-out branch cannot back-propagate `inf * 0 = nan`.

### Manning and normal depth

$Q = \frac{1}{n} A R^{2/3} \sqrt{S_0}$, with the SI constant 1.0. $Q$ is strictly increasing in
$h$ up to $h/D = 0.938$ and decreases above it, so the ascending branch is the entire invertible
domain. `normal_depth` inverts it with `solve_monotone`, batched over pipes and instances.

`q = 0` returns exactly `h = 0` with gradient 0 — a documented modelling choice, since the true
$dh/dq$ diverges there. A discharge above `capacity_flow` is **refused by name**: surcharge and
backwater are out of scope.

### The storage sweep

With `storage=True`, each manhole's water level is advanced by implicit Euler:

$$
A_s \frac{H^{n+1} - H^{n}}{\Delta t} + Q_{\text{out}}(H^{n+1}) = \sum Q_{\text{up}} + \text{lateral}
$$

solved with `solve_monotone` bracketed on $[0, 0.938D]$, valid because both left-hand terms are
strictly increasing in $H^{n+1}$. The sweep runs level-synchronously from leaves to outfall using
topological levels computed once; the only Python loop is over tree *levels* (depth 3 on the
committed fixture), never over individual pipes. Each level is one gather and one scatter.

When `storage=True` the normal-depth inversion is skipped entirely — the sweep has already solved
each manhole's own level, which *is* its outgoing pipe's entrance depth, bit-for-bit and with no
root-find.

### Dynamic wave

With `routing="dynamic_wave"` (SWMM's `DYNWAVE`) the network is a potential layer on **head**.
Where a conduit's flow follows from continuity alone, as on a dendritic kinematic-wave
network, nothing else is needed; where backwater, a flat conduit or a loop makes the flow depend
on the heads at both ends, the heads are the unknowns and the flows follow from the head
differences. The water side then has one `PotentialFlowLayer` named `"water"` (quantity `head`
in m, flow in m³/s) over the conduits, with the outfalls as its boundary, and a
`JunctionStorage` node source at every junction.

#### Equations

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
from the flow in that conduit.

#### What is solved

For frozen coefficients the conduit and junction equations are linear in the heads apart from
the min-clamps, so one pass is a single `PotentialFlowLayer` solve. The model then refreshes the
coefficients and repeats until the heads change by less than `iterate_tol` (m, default 1e-11)
or `iterate_max` passes (default 2000) are used (`Model(coupling="iterate")`, converging on the
layer's own interior head). The result is the head and flow field at which a SWMM trial
reproduces its own input.

The passes are Anderson-accelerated by default (`build_dynamic_wave_model(...,
acceleration="anderson", memory=5, iterate_flow_tol=1e-10)`; `Model`'s `iterate_acceleration`,
see [Layers and models](../concepts/layers.md)). Each pass's flows must then also reproduce the
flows it was fed to within `iterate_flow_tol` (m³/s): the head solve stops on its warm start once
every junction balance is within its `atol` (`solve_atol`), so near convergence the heads stand
still while the flows of a flat or nearly dry conduit still drift. Against phase A's plain relaxation (heads judged alone) the
mean passes per step fall from 34.8 to 14.1 (tree), 30.2 to 10.4 (backwater line) and 20.8 to 9.7
(loop) over the first 60 steps, and from 42.8 to 11.3 over the loop's first 120, whose worst step
drops from 327 passes to 22. The two iterations land on the same fixed point: with a tight head
solve (`atol=1e-12`) they agree to 2e-12 m; at the default `solve_atol` each lands within the head
solve's own resolution, `atol dt / (2 A_s)` (a few 1e-9 m). The mixing keeps to the fixed points
plain relaxation (and SWMM's relaxed trials) can reach: a flat, near-dry conduit's flow can have
a second, repelling fixed point, and a mixed step heading for it is replaced by the relaxed step.
`acceleration=None` is plain relaxation. A flow tolerance is required either way:
`iterate_flow_tol=None` (the heads judged alone, as in phase A) is refused by name, because on
the normal-flow slope switch (below) the heads bound the flows' residual only weakly, and a
heads-only test was measured stopping 3e-5 m³/s short of the fixed point.

**SWMM's iteration artefacts are not modelled**: trial relaxation, the 0.001 cfs flow-reversal
rule, link bypassing, `MAX_TRIALS` and `HEAD_TOLERANCE`. Each is a device for reaching the fixed
point and none moves it. The reader records `HEAD_TOLERANCE` and `MAX_TRIALS` in
`DynamicWaveOptions`, but noodl does not use them. This has a visible consequence. SWMM stops a
step when two successive relaxed trials differ by less than `HEAD_TOLERANCE`, which at SWMM's
defaults (0.0015 m, 8 trials) leaves its answer short of its own fixed point (see
[Verification](#verification)). SWMM still reports 0 % of steps not converging, because its
criterion is the change between trials, not the distance to the fixed point.

#### Units

SWMM computes in feet with $g = 32.2$ ft/s² and Manning's 1.486, and the conduit coefficients
here are evaluated exactly that way (reusing `xsect_tables.LCF`, `QCF`, `PHI`). Outputs are
converted with SWMM's own factors: head in m is ft × `LCF`, flow in m³/s is cfs × `QCF`,
conveyance in m²/s is ft²/s × `QCF/LCF`, and an area in m² is ft² × `QCF/LCF`, which is the
area `kinwave_steady` reports and makes $A_s\,\Delta y = \tfrac12\Delta t\,N$ exact in m and
m³/s.

#### Bed stiffening and lagged pieces

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

#### Normal-flow slope switch

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

#### Drop-end area switch

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
J5 on its threshold there too (see "Dynamic wave against SWMM").

#### Starting state and zero fall

The start follows SWMM's `flowrout.c`. A cold, dry network: junction depth 0, each outfall at its
outfall depth for zero flow (the stage for `FIXED`), each conduit's depth the mean of its two
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

#### Scope

Circular conduits, SWMM's tabulated geometry; the fixed routing step or
SWMM's variable step (`VARIABLE_STEP > 0`, below); a cold start or SWMM's `InitDepth` and
`InitFlow` (above); constant or
driver-supplied lateral inflows; `FREE`, `NORMAL` and `FIXED` outfalls, with flap gates;
parallel barrels, `MaxFlow`, conduit flap gates and conduit lengthening (below); storage
units with `FUNCTIONAL`, `TABULAR` and the closed-shape area curves (below);
`INERTIAL_DAMPING` `NONE`/`PARTIAL`/`FULL` and every `NORMAL_FLOW_LIMITED` option.
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

Flooding follows SWMM without ponding (`ALLOW_PONDING NO`, `dynwave.c:743-795`). A junction's
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

**Storage units** (`[STORAGE]`, `storage_unit`, `Manhole.storage`) follow SWMM's storage
node (`node.c`, `dynwave.c`). A unit is a junction whose surface area adds its curve's area
at the previous pass's depth (`node_getSurfArea`) to the attached conduit ends' (SWMM does
not zero a conduit's end area at a storage node, only a regulator's), so it enters the same
trapezoidal equation. Its area curve is `FUNCTIONAL` `A = a y^b + c`, `TABULAR` (linear
between the curve's points, linear from zero below the first, the last segment's slope,
floored at 0, beyond the last; `table_lookupEx`) or one of SWMM's closed shapes (cylinder,
cone frustum, paraboloid, pyramid frustum, each a quadratic in depth). Its volume, reported as
`"sewer.storage_volume"` (m³ per junction, 0 at a plain junction; `record=` in
`simulate_dynamic_wave`), is SWMM's `storage_getVolume`: the integral of the area (the
end-area method for `TABULAR`, as `table_getStorageVolume`), held at the full volume at and
above the full depth, so `SurDepth` stores nothing. SWMM's storage rules differ from a
junction's in three places:

- its full depth is extended to the attached conduit crowns only when it has a `SurDepth`
  (`link.c:450-458`), so a unit with no `SurDepth` floods at its own `MaxDepth`, below its
  crown if that is lower;
- under EXTRAN it surcharges only with a `SurDepth`, and then above its full depth rather
  than its crown (`dynwave.c:683-687`); the surcharged form, the hold band and the flood
  wall are those of a junction with the full depth as its crown;
- it keeps its own `InitDepth`, 0 included, instead of the average of its conduits
  (`flowrout.c:375`).

Evaporation (`Fevap`) and exfiltration are refused by name, so a unit loses no water but
through its conduits and its overflow.

Conduit options, each a SWMM `[CONDUITS]`, `[XSECTIONS]`, `[LOSSES]` or `[OUTFALLS]` entry
and a `Pipe` or `Outfall` field:

- **Barrels** (`Pipe.barrels`, `[XSECTIONS]` 7th field, 1 to 127). A conduit's `length`,
  `diameter`, `n` and slope are one barrel's. Its momentum update is one barrel's, and its
  flow, conveyance, surface areas and volume are totals (`n` times a barrel's, `dwflow.c:243,
  292`; `dynwave.c:575`), so `n` barrels at equal heads carry exactly `n` times one barrel's
  flow. The reported velocity is one barrel's.
- **`MaxFlow`** (`Pipe.max_flow`, m³/s, a limit on each barrel, `dwflow.c:270-273`) clamps the
  flow after the normal-flow limit.
- **Flap gates** (`[LOSSES]` flap gate YES for a conduit, `Gated` for an outfall,
  `link_setFlapGate`): flow against a conduit gate's direction is zeroed, and so is flow
  that would enter the conduit from a gated outfall. A conduit the reader reversed (adverse)
  keeps the direction of its file's From to To, `Pipe.flap_gate_reversed`.
- **Lengthening** (`LENGTHENING_STEP`, `DynamicWaveOptions.lengthening_step`, s):
  a conduit shorter than `(sqrt(g y_full) + v_full) t` is lengthened by the factor
  `lf = (sqrt(g y_full) + v_full) t / L`, `t` the smaller of `LENGTHENING_STEP` and the file's
  routing step, to `modLength = lf L` in the momentum terms and the surface areas, with its
  roughness divided by `sqrt(lf)` (the friction coefficient by `lf`), so its normal flow is
  unchanged and its volume keeps the true length. The variable step's Courant limit is scaled
  by `modLength / L`.

These limits apply in SWMM's order (`dwflow.c:255-281`): the normal-flow limit, `MaxFlow`,
the flap gate, then the dry-node caps (at `FUDGE` times the barrels). Each is a clamp of
the element's flow against a constant, so the flow stays continuous and non-decreasing in the
head difference, as the head solve's Newton step needs. A set `InitDepth`, `InitFlow`,
`MaxFlow`, flap gate or barrel count is refused by name under STEADY or KINWAVE routing.

The reader refuses by name, with the line number, `ALLOW_PONDING YES`, loss coefficients
and seepage in `[LOSSES]`, storage evaporation and exfiltration, and TIDAL/TIMESERIES
outfalls.
`build_model` with `air=True`, `quality=True` or `storage=True` is refused by name. Ponding,
headspace air and quality on the dynamic wave are planned.

```python
from noodl.apps.sewer import read_swmm_inp, build_model, simulate_dynamic_wave

net, _, _ = read_swmm_inp("tree_dynwave.inp")
model, state, drivers = build_model(net)
state, hist = simulate_dynamic_wave(model, state, drivers, steps=24)    # 24 x 5 s
print([round(float(q), 4) for q in hist["water.q"][-1]])                # m3/s, C1..C5
```

```text
[0.0184, 0.031, 0.0032, 0.0092, 0.0004]
```

`hist` holds `"water.phi"` (head at every node, m) and `"water.q"` (flow in every conduit,
m³/s), stacked over steps, with `"t"` (s, the end time of each step) and `"dt"` (s, each
step's length). `simulate_dynamic_wave` also takes an `inflow=` override (`(n,)`
constant, `(B, n)` a batch, `(steps, B, n)` a series, or a function `inflow(t)` of the
start time `t` in s) and `record=` for other state keys and for `"sewer.overflow"` (m³/s
per junction).

#### Variable step and time-varying inflow

Give `duration=` (s) instead of `steps=` to run a stretch of simulated time. With
`VARIABLE_STEP > 0` (a `DynamicWaveOptions.variable_step`, the Courant factor, 0 to 2) every
step is chosen as SWMM does (`dynwave_getRoutingStep`): the first step is `MINIMUM_STEP`
(`minimum_step`, 0.5 s by default, capped at the routing step and floored at 1 ms), and
each later one is the smallest of

- the routing step;
- each conduit's Courant step, `V / |q| * Fr / (1 + Fr) * VARIABLE_STEP` with `V` the
  conduit's volume, `q` its flow and `Fr` its Froude number (a dry, still or full closed
  conduit, `Fr <= 0.01`, does not limit it);
- each junction's depth-change step, the time to rise a quarter of its crown depth at the
  rate it rose over the last step (a dry junction, one at or above its crown, and a quasi-steady
  one do not limit it);

floored at the minimum step and truncated to a millisecond. The reader takes an absent
`VARIABLE_STEP` as SWMM's 0.75 and lowers `ROUTING_STEP` to `WET_STEP` (default 300 s) when
larger, as SWMM does; `DynamicWaveOptions()` built in code defaults to the fixed step (0). The last step is cut to the time
that remains (at least 1 ms), so the run ends at `duration`; this clipping applies to a fixed
step too. One call is one run: under a variable step a second call restarts with the minimum step and
with `t` from 0, as a new SWMM run does (continuing needs the previous step's state, which
the call does not carry). A variable step needs a single network (no batched `inflow`), and `dt=` is refused
with it, since the step is then chosen by the options.

An inflow function `inflow(t)` returns the `(n,)` lateral inflow of the step that starts at
`t` s. SWMM reads a time series at the start of the step (its own clock is 1 ms ahead,
`swmm5.c`'s `getDateTime`: a function that stands for a SWMM `[TIMESERIES]` should add
0.001 s), and the inflow enters the step's trapezoid with the end state, as a constant
inflow does. A function (or an `(steps, B, n)` series) in `simulate_dynamic_wave` is how
time variation enters: the reader still refuses `[TIMESERIES]`. The step sequence and every
state of the variable-step tree, with and without a ramping inflow, match SWMM 5.2.4 to the
millisecond and to the targets of the other dynamic-wave comparisons
(`tests/verification/test_sewer_dynwave_parity.py`). Because the step depends on the state
through switches, a variable-step run is not differentiable through its step sizes.

### Headspace air

`Headspace` is an `Element` with $\Delta p = R(h)\,\lvert Q \rvert Q$ and

$$
R(h) = \frac{f_{\text{air}} L \rho}{2 D_h A_{\text{air}}^2}
$$

inverted in closed form with a laminar blend near zero. The air-side wetted perimeter is
$P_{\text{air}} = \pi D - P(h) + T(h)$ — the dry wall plus the water surface acting as a moving
wall — and $D_h = 4A_{\text{air}}/P_{\text{air}}$.

`Drag` is a `Drive`: $D = \tfrac{1}{2} f_i \rho U_s \lvert U_s \rvert T L / A_{\text{air}}$ with
$U_s = c_s V_w$. The full branch law is $\Delta p = R\lvert Q\rvert Q - D - B$, where $B$ is the
**same `Stack` buoyancy drive the building application uses** — reused unchanged.

The drag sign convention is documented and was verified rather than assumed: the drive returns
$+D$ so that with both ends at ambient the air moves from the upstream manhole toward the
downstream one, the direction the wastewater drags it. The opposite sign was tried and caught by
the measurement.

### Quality

| Quantity | Form |
|---|---|
| Henry's constant | $H(T) = 1/(H_{cp}(T) R T)$, $H_{cp}(T) = H_{cp}(298.15)\exp\!\big(2100(1/T - 1/298.15)\big)$ |
| Free sulfide fraction | $f = 1/(1 + 10^{\,\mathrm{pH} - \mathrm{p}K_a})$ |
| Transfer coefficient | $K_L a = a(1 + b\,\mathrm{Fr}^2)(S_0 V)^{3/8} / d_m$ per hour |
| Two-film flux | $K_L a \, V \, \big(f C_S - C_G (M_S/M_{\mathrm{H_2S}}) / H\big)$ kg S/s |
| Sulfide generation | $d[S]/dt = M' \mathrm{EBOD} / R_h - m [S] (S_0 V)^{3/8} / d_m$ |
| BOD decay | $d[\mathrm{BOD}]/dt = -k_{\mathrm{BOD}} \cdot 1.07^{\,T-20} [\mathrm{BOD}]$ |

The gas concentration is converted to sulfur-equivalent before the Henry partition is applied, so
the flux is symmetric between the two source terms in moles of S.

`LateralLoads` must be registered **before** `H2STransfer` in the closure list, so that
`H2STransfer` adds its transfer term rather than overwriting the lateral load.

The model is built with `reaction_order="before_transport"`: each step applies sulfide generation
and BOD decay to the step-start state, then transports it, and returns the post-transport
concentration, as SWMM does. `H2STransfer` reads the step-start state either way.

## Verification

Against SWMM 5.2.4 through `pyswmm`, with `geometry="tabulated"`, on three kinematic-wave networks
of circular conduits:

- `tree_kinwave.inp`, the hand-built 5-conduit tree.
- `example1_kinwave.inp`, the 13-conduit network of EPA SWMM's published *Example 1*. It is
  converted to metres, and its runoff is replaced by rational-method constant inflows, because
  runoff and CFS units are outside this reader.
- `tree_kinwave_large.inp`, a synthetic 32-conduit tree with fills from 8e-5 to 0.81 of `Qfull`.

`tests/data/sewer/make_kinwave_fixtures.py` regenerates the last two. The engine identity is
pinned: `engine_version == "5.2.4"` and `flow_routing_error == 0.0`.

| Check | Tolerance | Measured worst (5 / 13 / 32 conduits) | Exact circle, for comparison |
|---|---|---|---|
| Pipe flows | rel 1e-9 | 1.4e-14 / 3.1e-15 / 6.2e-14 | the same |
| Depths (live, float64) | rel 1e-10 | 7.5e-15 / 1.2e-15 / 1.7e-14 | 5.5e-4 / 1.8e-3 / 5.2e-3 |
| Volumes (live, float64) | rel 1e-10 | 8.6e-15 / 1.8e-15 / 2.3e-14 | 6.5e-4 / 1.7e-3 / 4.8e-3 |
| Velocities (binary output, float32) | rel 1e-7 | 3.7e-8 / 5.8e-8 / 5.3e-8 | 6.3e-4 / 2.5e-3 / 1.8e-1 |
| Tracer, tank-in-series closed form | rel 1e-9 | 4.5e-15 | 7.8e-6 |
| Tracer, the model's quality layer, steady | rel 1e-9 | 6.3e-13 | 3.5e-3 at dt = 60 s if sampled after the reaction |
| Tracer, time-varying load, every 5 s step, head conduit | rel 1e-9 | 1.1e-14 | |
| Tracer, same run, downstream conduits (SWMM's junction lag, see below) | not asserted | 1.1e-2 / 1.7e-2 of peak at 5 s | 2.2e-3 / 3.5e-3 at 1 s |

The tolerances are not fitted to these numbers. Depth and volume are read live, as float64, and
both sides run the same float64 operations. 1e-10 is three decades above a rounding bound of
about 1e-13. Velocity has no live accessor under KINWAVE. The binary output stores it as float32,
whose half-ulp is 5.96e-8.

The quality layer uses SWMM's scheme and reports where SWMM reports. Both codes split a step
into an explicit first-order decay $c \to c(1 - k\Delta t)$ followed by an implicit upwind mixing
over the conduit volume (SWMM `qualrout.c`: `getReactedQual`, then `getMixedQual`), and both
return the concentration after the mixing: `build_model` sets
`Model(reaction_order="before_transport")`. The fixed point satisfies
$c(q + kV) = \sum q_u c_u + \text{load}$ at any $\Delta t$, which is SWMM's own. Sampled after the
reaction instead (the framework's default order), the fixed point would be $(1 - k\Delta t)$
times SWMM's, 3.47e-3 at 60 s.

The transient check drives the model with the inflow concentration SWMM applied at each step
(its `J1` node quality) and compares every routing step. A conduit fed only by a lateral inflow
takes exactly SWMM's step. A conduit fed by upstream conduits does not: SWMM mixes a junction's
inflow from the upstream links' start-of-step concentrations (`findLinkMassFlow` reads
`Link.oldQual`), a one-step lag per junction, where the model solves the whole tree implicitly.
That difference is first order in $\Delta t$. The test asserts that SWMM equals exactly this
lagged recursion on the model's own volumes and flows (5e-15), so the lag is the whole
difference. Transient hydraulics are not compared: the model's water side is quasi-steady (or
manhole storage), not a kinematic wave. SWMM also mixes over the step-start volume,
$(cV_1 + c_{in} q_{in}\Delta t)/(V_1 + q_{in}\Delta t)$, where the model's amount form divides by
$V_2 + q_{out}\Delta t$; the two agree when $V_2 - V_1 = (q_{in} - q_{out})\Delta t$.

### Dynamic wave against SWMM

`tests/verification/test_sewer_dynwave_parity.py` runs live SWMM 5.2.4 through `pyswmm` with a
fixed 5 s routing step for 30 min and compares every step's node depths and link flows
(files `tests/data/sewer/{tree,backwater,loop}_dynwave.inp`):

- a 5-conduit tree, filling from dry;
- a backwater line, with a drop at J2, a flat conduit C2 and a `FIXED` outfall stage above the outfall invert;
- a loop with an adverse conduit.

SWMM is run with `HEAD_TOLERANCE` 1e-12 m and `MAX_TRIALS` 500; noodl with `iterate_tol` 1e-13 m
and `iterate_max` 2000. Each test asserts that SWMM's report says 0 % of steps not converging.
The two sides compare the same fixed point only if SWMM has reached it, and SWMM's trials are
relaxed, so a loose `HEAD_TOLERANCE` stops it short (second table). Tightening it to 1e-12 m
brought the differences below the targets without changing either code's equations.

| Check | Target | Measured worst (tree / backwater / loop) |
|---|---|---|
| Node depths, every 5 s step | abs 1e-7 m | 1.2e-9 / 4.5e-9 / 3.4e-9 m |
| Link flows, every 5 s step | rel 1e-6 + abs 1e-9 m³/s | within the target at every step |

The options are compared together on `options_dynwave.inp`: J1 starts at 0.25 m (`InitDepth`)
and C3 at 0.03 m³/s (`InitFlow`); C1 is limited to 0.012 m³/s (`MaxFlow`) and has a 0.05 m
outlet offset; C3 has two barrels; C4 has a flap gate that closes on the water backing up from
J4; C6 is adverse in the file (it is reversed, with its `InitFlow` negated, and the offsets
swap), has 0.003 m³/s of `InitFlow` and a flap gate that, being reversed, passes only the
file's J5-to-J3 direction, so the downhill flow is closed; the `FIXED` outfall (stage 9.3 m,
above J4's invert and C5's crown) is gated, so C5 carries nothing until J4 rises over the
stage; C2 is 10 m long and `LENGTHENING_STEP 5` lengthens it. The fixture is therefore not
free-surface throughout. SWMM converges on every step, and noodl meets the targets above:
worst depth 6.4e-9 m over the 359 steps. Under `VARIABLE_STEP 0.75` the same fixture's 304
steps match SWMM's step sequence to the millisecond (the Courant step reads one barrel's flow
and volume and the lengthened conduit's `modLength / length`) and its states to the same
targets.

Three variants of that fixture take a junction through a drop end's area threshold while it
fills (see "Drop-end area switch"): a 0.03 m or 0.1 m outlet offset on C3 (J4 fills behind
the closed outfall gate) and C6 without its flap gate (J5 fills). SWMM fails to converge on
steps 60, 62 and 167, on step 107, and on steps 106 and 129 respectively, found by cutting
the run after each step (SWMM floors a run's duration, so each cut run ends at 5n + 1 to
3 s). noodl converges on every step. Its own run meets the targets on every step before
SWMM's first failure; after it the two histories differ by what each did on that step
(SWMM's last trial against noodl's held solution), up to 3e-4 m stored behind the gate. Each
step is therefore also run from SWMM's own state before it: on every step SWMM converged
from a converged start the targets hold (worst 2.0e-9 m), and on SWMM's failing steps and
the steps after them noodl is within SWMM's own oscillation (at most 3.1e-4 m and
2.5e-5 m³/s; bounded at 1e-3 m and 5e-5 m³/s). The fixture itself keeps C1's offset, where
nothing fills through a threshold.

Surcharge is compared on the same tree at higher inflows. `tree_slot_dynwave.inp` (2.5x,
`SURCHARGE_METHOD SLOT`) pressurises C4 from step 95. `tree_extran_dynwave.inp` (3x,
`SURCHARGE_METHOD EXTRAN`) surcharges J2, J4 and J3 above their crowns at steps 21, 139 and
270; there noodl's head solve runs at `atol=1e-12` m3/s, because a surcharged junction has no
storage to damp a balance residual. SWMM converges on every step of both runs. Every step meets
the targets above: the worst depth difference is 8.3e-10 m (SLOT) and 1.6e-10 m (EXTRAN).
SWMM's EXTRAN iteration has no fixed point when a junction would fall back through its crown
while the storage equation keeps it above (see "Dynamic wave"). Such steps have no SWMM
reference, and none occurs in this fixture.

Flooding is compared on depths, flows and `sewer.overflow` against pyswmm's `Node.flooding`
(SWMM's overflow rate of the step), at the same targets (the overflow at the flow target):

- `flooding_dynwave.inp` (EXTRAN, 3x): J2, with `SurDepth` 1.7 m, floods at 2.0 m on steps
  21-27 and is then released; J3 floods from step 270.
- `flooding_slot_dynwave.inp` (the same junctions under SLOT): J3 floods from step 307.

SWMM converges on every step of both, and the worst differences are 1.2e-10 m and
1.4e-11 m³/s of overflow.

`flooding_outfall_dynwave.inp` floods J4, whose only outflow conduit C5 ends at the free
outfall, from step 236. There SWMM's answer is not converged for one quantity, although
SWMM reports 0 % of steps not converging:

- SWMM's stopping rule tests only non-outfall depths. J4 is clamped at `y_max`, so once J4
  is clamped on two trials in a row nothing tested depends on C5's flow.
- SWMM therefore stops while C5's relaxed flow is still 3.2e-5 m³/s from its fixed point.
  A reproduction of SWMM's trials at that step lands on SWMM's value to 1e-10 m³/s and
  converges to noodl's.
- That error decays over the next 22 steps.

The test therefore leaves out C5's flow, the outfall depth and J4's overflow on steps
236-258, where they are held only to the onset error's scale. Every other quantity on every
step meets the targets.

Storage units are compared on depths, flows, overflow and each unit's volume (against
pyswmm's `Node.volume`, at the depth target times the unit's area):

- `storage_dynwave.inp`: the tree with J5 `CONICAL`, J3 `FUNCTIONAL` and J4 `TABULAR`,
  filling from dry below every crown;
- `storage_extran_dynwave.inp` (EXTRAN, 3x): J3 (`MaxDepth` 0.4 m below its 0.45 m crown, no
  `SurDepth`) floods at 0.4 m, its full depth not extended; J4 (`MaxDepth` 0.3 m, `SurDepth`
  1.0 m) has its full depth extended to its 0.45 m crown and surcharges above it, its volume
  held at the full volume;
- the same under SLOT.

SWMM converges on every step of all three, and every step meets the targets (worst depth
7.6e-10 m). Without the storage area the first fixture misses by 1.2e-2 m within 40 steps,
and with the conduit end areas zeroed at the storage units by 0.17 m.

The same networks with SWMM at its **default** tolerance (`HEAD_TOLERANCE` 0.0015 m,
`MAX_TRIALS` 8), against noodl's converged answer:

| SWMM at default tolerance | Worst depth difference (m) | Worst flow difference (% of the network's maximum flow) |
|---|---|---|
| tree | 2.3e-3 | 0.8 |
| backwater | 1.3e-2 | 7.4 |
| loop | 3.6e-3 | 3.6 |

SWMM reports 0 % of steps not converging at those defaults too. Its stopping rule bounds the
change between relaxed trials, not the distance to the fixed point, so a result that SWMM
calls converged can sit centimetres away from the equations' solution on a backwater.

Non-SWMM checks, for the physics SWMM does not model:

| Check | Result |
|---|---|
| Pescod & Price Tests 7–9, air-to-water velocity ratio in a 300 mm UPVC pipe, 15 m, open at both ends (20–40 % band) | 24.14 %, 25.00 %, 25.15 % against measured 35 %, 25 %, 27.5 % (Test 8 calibrates `f_i`); each inside the band and equal to the closed form to round-off (asserted at rel 1e-6) |
| Pescod & Price Tyneside ventilation band (105–315 m³/h) | bracketed; open-both-ends gives 1253.78 m³/h, a 17.27% velocity ratio, inside their 5–30% envelope |
| Fan draws exactly through the leaks | balance 1.234e-14, nodal residual 6.3e-15, power residual 8.1e-13 |
| Transfer-dominated state reaches Henry equilibrium | rel 1e-10, **measured 1.4e-16** |
| Air-layer nodal residual, and Tellegen power balance | 4.518e-13 m3/s, 6.141e-12 W |
| Cross-phase sulfide conservation | rtol 1e-12 |
| Adjoint gradients vs Richardson-extrapolated central differences | rel 1e-6, holds |
| Storage sweep vs the closed-form steady state | **5.7e-15** on flows, **4.4e-15** on depths |

## Coefficient provenance

Each coefficient is labelled, and three of them are not verified. This matters if you intend to
publish numbers from this application.

| Coefficient | Default | Status |
|---|---|---|
| `f_air`, headspace wall friction | 0.02 | **Unverified.** Stated to lie in the reported 0.015–0.045 range for sewer crowns; the source is paywalled and was not independently checked. |
| `f_i`, interfacial drag | 7.49e-4 | **Calibrated** against one Pescod & Price test (Test 8), cross-checked against Tests 7 and 9 at 24.14% and 25.15% against measured 35% and 27.5% — inside a fairly wide 20–40% acceptance band. |
| Henry's constant, van't Hoff slope | 1.0e-3 mol/m³/Pa, 2100 K | **Verified** (Sander 2023). |
| p$K_a$ | 7.0 | Verified value; its temperature dependence is approximated as constant. |
| $K_L a$ constants $a$, $b$ | 0.86, 0.20 | Correlation form corroborated, **constants unverified** (paywalled). |
| Pomeroy–Parkhurst $M'$, $m$ | 0.32e-3 m/h, 0.64 | **Vendor defaults, unverified** against Pomeroy and Parkhurst 1977. |
| $k_{\mathrm{BOD}}$ | 0.2/day | Generic (Metcalf & Eddy). |

## Limitations

- **Kinematic-wave and steady routing have no surcharge or backwater.** A network that would
  surcharge or need backwater is refused by name there; use `routing="dynamic_wave"` for
  backwater.
- **Kinematic-wave and steady routing: dendritic networks only**, one outgoing pipe per
  manhole, no loops. Dynamic-wave routing allows loops and flat or adverse conduits.
- **Dynamic wave: no ponding, headspace air or quality yet** (planned); storage units
  without evaporation or exfiltration.
  Flooded water is lost, as in SWMM without ponding; with no ground (`MaxDepth` 0) and no
  `SurDepth` a junction floods as soon as it fills to its crown. `build_model(...,
  air=True)` or `quality=True` is refused. Circular conduits only. Under EXTRAN, where SWMM's own
  surcharge iteration has no fixed point, noodl holds the junction at its crown (see
  "Dynamic wave"), so such steps have no SWMM reference. Likewise where SWMM's normal-flow
  slope test has no consistent side (a draining outfall conduit), noodl puts the solution
  on its switch, y1 = y2 (see "Normal-flow slope switch"); SWMM does not converge on such
  steps, and its unconverged trials differ by up to its own oscillation (~6e-4 m on the
  tree). The Froude test of the normal-flow limit stays a lagged flag. At a step that lands
  on the switch the gradient is the tie rule's: the sensitivity of a solution constrained
  to y1 = y2, where SWMM has no fixed point to differentiate. It does not depend on the
  wall slope S, which only shapes the passes (the fixed point does not move with S; on the
  tree's drain-down the gradient is the same to 11 digits for S scaled by 0.2 and 5, and
  matches central differences to 2e-8 relative). Likewise a junction filling through a
  drop end's area threshold, where SWMM's storage equation has no fixed point, is held on
  the threshold (see "Drop-end area switch"); SWMM does not converge on such steps, and the
  histories then differ by up to SWMM's oscillation (3e-4 m on the options fixture with a
  0.03 m offset on C3). The rule picks one threshold per junction, the nearest to its
  head; two drop ends switching at one junction in the same step are not covered.
- **No gas-phase sulfide sink.** H₂S leaves the headspace by ventilation, at the outfall, and
  by re-absorption into the water (the two-film flux is signed); no wall uptake or gas-phase
  oxidation is modelled.
- **Headspace drag uses the absolute water-surface velocity.** `Drag` drives the air with
  $U_s$ rather than the relative velocity $U_s - U_{\text{air}}$, which would need the air
  layer's own solved state. The drag is overestimated where the headspace air already moves
  with the water.
- **`f_i` is not differentiable.** It is a `Drive` attribute, and the differentiable solve
  refuses `requires_grad=True` on it; you cannot fit it by gradient descent.
- **Without a ground elevation, manhole shafts add no headspace volume**, so the air-quality
  capacity is the conduit headspace alone. `model.notes` says when this applies.

## Install

Nothing beyond the base dependencies. `pyswmm` is needed only to reproduce the SWMM parity tests:

```bash
pip install "noodl-physics[dev]"
```
