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
| `Manhole(name, invert, ground=None, inflow=0.0, surface_area=None)` | A junction. `surface_area` defaults to SWMM's `MIN_SURFAREA`, 1.167 m² — a 4 ft shaft. |
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
conduits meeting there, floored at `MIN_SURFAREA`) and depth $y$ (m):

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
still while the flows of a flat or nearly dry conduit still drift. Against plain relaxation the
mean passes per step fall from 34.8 to 14.1 (tree), 30.2 to 10.4 (backwater line) and 20.8 to 9.7
(loop) over the first 60 steps, and from 42.8 to 11.3 over the loop's first 120, whose worst step
drops from 327 passes to 22. The two iterations land on the same fixed point: with a tight head
solve (`atol=1e-12`) they agree to 2e-12 m; at the default `solve_atol` each lands within the head
solve's own resolution, `atol dt / (2 A_s)` (a few 1e-9 m). The mixing keeps to the fixed points
plain relaxation (and SWMM's relaxed trials) can reach: a flat, near-dry conduit's flow can have
a second, repelling fixed point, and a mixed step heading for it is replaced by the relaxed step.
`acceleration=None, iterate_flow_tol=None` restores plain relaxation on the heads alone.

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
fed heads), the normal flow, the dry-node flags, the outfall heads, and the head at any
conduit end SWMM pins (the head is raised to the invert plus offset when it lies below it, and
set to critical depth in `UP_CRITICAL`/`DN_CRITICAL`). A conduit with a pinned end has no head
dependence there: its conveyance is zero and the pinned head enters the constant term.

#### Starting state and zero fall

The cold start follows SWMM's `flowrout.c`: junction depth 0, each outfall at its outfall depth
for zero flow (the stage for `FIXED`), each conduit's depth the mean of its two clamped end
depths, and a previous net inflow of 0 at the first step. SWMM orients a conduit with exactly
zero fall by round-off under `DYNWAVE`, so noodl can legitimately differ on an exactly flat
conduit; the backwater fixture avoids an exact tie for that reason.

#### Scope

Circular conduits, one barrel, SWMM's tabulated geometry; a fixed routing step
(`VARIABLE_STEP 0`); a cold dry start (zero `InitDepth` and `InitFlow`); constant or
driver-supplied lateral inflows; `FREE`, `NORMAL` and `FIXED` outfalls without flap gates;
`INERTIAL_DAMPING` `NONE`/`PARTIAL`/`FULL` and every `NORMAL_FLOW_LIMITED` option.
`SURCHARGE_METHOD SLOT` is SWMM's Preissmann slot: above `0.985257 y_full` the top width is the
Sjoberg slot width `0.5423 w_max exp(-(y/y_full)^2.4)` (`0.01 w_max` above `1.78 y_full`), the
area above the crown is `a_full + (y - y_full) w_slot`, the friction radius stays `r_full`, a
full conduit takes no inertial terms, and the reported link depth stays `min(y_mid, y_full)`
while the stored volume may exceed the full volume. The junction equation is unchanged (SWMM
never takes its node-surcharge branch under SLOT; the junction surface-area floor governs a
pressurised junction). Under `SURCHARGE_METHOD EXTRAN` (the default) any conduit end that
reaches the crown stops the run with an error naming the conduit and the time. The reader refuses by name,
with the line number, `VARIABLE_STEP > 0`, `LENGTHENING_STEP > 0`, `ALLOW_PONDING YES`, nonzero
`InitDepth`/`InitFlow`/`MaxFlow`, flap gates, barrels other than 1 and TIDAL/TIMESERIES outfalls.
`build_model` with `air=True`, `quality=True` or `storage=True` is refused by name. Variable step,
ponding, EXTRAN surcharge, headspace air and quality on the dynamic wave are planned.

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
m³/s), stacked over steps. `simulate_dynamic_wave` also takes an `inflow=` override (`(n,)`
constant, `(B, n)` a batch, `(steps, B, n)` a series) and `record=` for other state keys.

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
- **Dynamic wave: no EXTRAN surcharge, variable step, ponding, headspace air or quality yet**
  (planned). Under `SURCHARGE_METHOD EXTRAN` a conduit end reaching the crown stops the run
  with the conduit's name and the time; use `SLOT` for surcharged networks; `build_model(..., air=True)` or `quality=True` is
  refused. Circular conduits only, one barrel, cold dry start.
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
