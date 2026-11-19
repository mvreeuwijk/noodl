# Sewer physics

The equations the sewer application solves and the assumptions behind them: circular geometry,
Manning's equation, the water side as two different shapes, the manhole storage sweep, the
headspace air and the sulfide chemistry. The dynamic-wave momentum solve has its own page,
[Dynamic-wave physics](physics-dynamic-wave.md). How to run these models is in the
[guides](index.md#in-this-section).

## Exact circular geometry

With the wetted half-angle $\theta = 2\arccos(1 - 2h/D)$:

$$
A = \frac{D^2(\theta - \sin\theta)}{8},\quad
P = \frac{D\theta}{2},\quad
R = \frac{A}{P} = \frac{D}{4}\left(1 - \frac{\sin\theta}{\theta}\right),\quad
T = D\sin\frac{\theta}{2}
$$

These exact relations are the default for networks built in code (`SewerNetwork(...,
geometry="analytic")`).

## SWMM's own circular geometry

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

Scope: on the kinematic path `geometry="tabulated"` covers the steady case only, and is refused
by name with `storage=True`. Dynamic-wave routing always uses the tables and refuses
`geometry="analytic"` by name. The discharge limit is SWMM's `Qfull` rather than the analytic 0.938 D capacity.
The headspace, top width, hydraulic radius and mean depth (air side, sulfide) of a circular conduit remain the exact
circle, evaluated at SWMM's depth. SWMM has no headspace to compare them with. Other shapes are
dynamic-wave only (see [Cross sections](dynamic-wave-structures.md#cross-sections-losses-transects-and-culverts)); the kinematic and steady paths refuse them by name.

Numerical care worth noting: the `arccos` argument is clamped to the *exact* domain $[-1, 1]$,
not an epsilon-shrunk one, which would shift the boundary value by
$O(\sqrt{\varepsilon})$. Every $0/0$ site is guarded on **both** branches of its `torch.where`,
so a masked-out branch cannot back-propagate `inf * 0 = nan`.

## Manning and normal depth

$Q = \frac{1}{n} A R^{2/3} \sqrt{S_0}$, with the SI constant 1.0. $Q$ is strictly increasing in
$h$ up to $h/D = 0.938$ and decreases above it, so the ascending branch is the entire invertible
domain. `normal_depth` inverts it with `solve_monotone`, batched over pipes and instances.

`q = 0` returns exactly `h = 0` with gradient 0 — a documented modelling choice, since the true
$dh/dq$ diverges there. A discharge above `capacity_flow` is **refused by name**: surcharge and
backwater are out of scope.

## The storage sweep

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


## The two water sides

The sewer's kinematic-wave path is
CONTINUITY-first. A gravity sewer runs free-surface: on the dendritic tree of section 1's own
analysis, normal flow at every pipe is set by continuity alone -- the net inflow upstream of it
(`cycles.particular_flow`, closed form on a tree, no Newton solve at all) -- and depth follows
from Manning's law given that flow, not the other way round; a head difference does not drive the
water there, so a Kirchhoff layer on head is the wrong shape for it. The dynamic wave is the
case where head differences do drive the flow, and there the sewer IS a Kirchhoff layer on head.
SWMM's dynamic-wave update (the Saint-Venant momentum equation integrated over a step) is, for
frozen depth-dependent coefficients, linear in the end-head difference of each conduit,
$q = c_0 + k\,(h_1 - h_2)$ with $k \ge 0$, and a junction's storage adds a monotone term in its
own head, so one pass is an ordinary `KirchhoffLayer` solve (see [Kirchhoff layers](../../concepts/layers.md)) with the outfalls as
boundary heads. The coefficients depend on the heads, so the pass is iterated
(`coupling="iterate"`) until the heads stop changing. SWMM's own Picard trials, which relax and
average its iterates, and noodl's iterated implicit head solve share one fixed point: a state
that reproduces its own coefficients and satisfies the conduit and junction equations. noodl
converges to that fixed point and does not replay the trials (see the
[dynamic-wave physics](physics-dynamic-wave.md#what-is-solved)). Conduits may form loops and may be flat or
adverse, which the continuity-first path cannot represent.

## Surcharge as instantaneous continuity

Above its crown a junction is pressurised: SWMM's
EXTRAN update $y = y_{\text{last}} + \text{corr}\,\Delta Q/\text{denom}$ is converged exactly
when the junction's net inflow $\Delta Q$ is zero, so a surcharged junction stores nothing and
its balance is $\sum Q = 0$, an algebraic constraint in place of the storage term
$A_s\,\Delta H/\Delta t$ (the correction factor and SWMM's blend only shape its trials). The
conduits meanwhile cap their end depths at the crown and keep the heads driving a full
conduit. The storage term is therefore non-smooth at the crown, switching from
$A_s\,\Delta H/\Delta t$ to zero, but monotone in the head on each side, which is all the
head solve needs. Flooding is the same device at $y_{\max}$: a wall in the head solve holds the
junction at $y_{\max}$ and the excess leaves the network. The Preissmann slot (`SLOT`) is the
smooth alternative: above $0.985\,y_{\text{full}}$ a conduit's top width follows Sjoberg's slot
width, so the area above the crown grows as $a_{\text{full}} + (y - y_{\text{full}})\,w_{\text{slot}}$
and the storage term stays smooth and positive, which is why SLOT needs no junction branch.

## Tie rules

Three of SWMM's switches have no fixed point on a band of states, because
they test a quantity that the switched branch itself moves: SWMM's trials then alternate between
the branches and it reports non-convergence. Each is replaced by a stiff in-pass switch that
agrees with SWMM wherever SWMM has a fixed point and holds the state on the switching surface
where it has none. (1) A junction at its crown, where the depth-only surcharge flag would
chatter: a hold band keeps it at the crown while the net inflow needed lies inside the
jump. (2) The normal-flow slope limit of a conduit, $q = \min(q_{\text{lin}}, \max(q_{\text{norm}},
q_{\text{last}} + S\,(\Delta h - \Delta z)))$, which has a fixed point on the surface
$y_1 = y_2$ between the two flows when a draining outfall conduit ties its end depths. (3) A
junction filling through a drop end's surface-area threshold, where the area jumps with the
junction's own depth: its withdrawal is $\max(w_{\text{lo}}, \min(w_{\text{hi}}, N_{\text{last}}
+ S\,(H - H^*)))$, so the junction sits on the threshold $H^*$ inside the jump. In each the
wall slope $S$ shapes only the passes, and the fixed point does not depend on it. A gradient
at such a state is that of the constrained solution; see the [dynamic-wave physics](physics-dynamic-wave.md#intentional-differences-from-swmm)
for the costs against SWMM.

## The manhole-level storage idealisation, and what it is not

On the kinematic-wave path, with `storage=True`,
each manhole gets a surface area and a level advanced by implicit Euler,
$A_s\, dH/dt = \sum \text{inflow} - Q_{\text{out}}(H)$, where $Q_{\text{out}}$ is read off the SAME Manning law as the
quasi-steady case at depth $H$. This is a lumped storage-node approximation -- "the
manhole's level equals its own outgoing pipe's entrance depth" -- not a hydraulic profile
along the pipe and not SWMM's dynamic wave, which carries one head per junction and a
backwater-coupled momentum equation between them; under constant inflow the storage sweep's
fixed point is the quasi-steady solution (measured 5.7e-15 relative on flows, 4.4e-15 on
depths), which is the only property this idealisation is required to
have. The sweep advances every manhole LEVEL-SYNCHRONOUSLY, leaves to the outfall, in the
same order the tree's own closed-form flow solve uses: it precomputes, once, the leaf-to-
root level order and, per level, flat gather/scatter index tensors, so the per-call sweep is
one batched `solve_monotone` call plus one `index_add` per tree LEVEL rather than a Python
loop over manholes: the loop count is the tree depth, not the manhole count. The pipe carrying each
manhole's outflow is found by which pipe drains INTO that manhole's own outfall, never
assumed to be the last pipe read from the file, because the reader's pipe order is whatever
order the `.inp` lists conduits in, not necessarily the order the tree drains in;
and the headspace `Stack` buoyancy head's `z_path` on a `headspace` edge is the pipe's
CROWN elevation -- the mean of its two end inverts plus its own diameter -- not the mean
invert alone, which is only the pipe's invert at its midpoint. A forest of more
than one tree is supported on both the water and air sides: each component gets
its own outfall and, on the air side, its own outfall-to-ambient headspace edge, so `air.q`
and `sewer.q` conserve mass within each component independently rather than at one shared
sink.

## The `air.phi` datum

Every `leak` edge's `Stack` carries a manhole term that cancels
against the manhole's own ground level (`z_ref - z_path = 0` there) but an AMBIENT term that
does not, because `ambient` has no `ground` attribute of its own and so falls back to a
`0.0` datum rather than to the manhole's; the same `0.0` is used identically at the outfall's
open-air edge and at `ambient`'s own fixed boundary condition (`air.phi_boundary = 0`), so
every air pressure in the solve is shifted by the same constant -- `air.phi` is
DATUM-REFERENCED to this convention, not to a physical zero, and every head, refusal and
verification row reads only DIFFERENCES between manholes, which the convention leaves
unaffected.

## The headspace momentum balance, and the head-cannot-see-`phi` rule

Air in the
headspace above the flow is driven by three terms per pipe: Darcy-Weisbach wall
friction on the air itself, buoyancy from the density difference between a warm manhole
shaft and ambient (the SAME `Stack` head the building application's stack effect uses, one
full-node air-density driver, `rho_air_nodes`), and drag from the moving water surface
dragging the air along above it. That last term is written as a head term --
$D = \tfrac{1}{2} f_i\, \rho_{\text{air}} U_s \lvert U_s \rvert\, T L / A_{\text{air}}$ in the ABSOLUTE water-surface velocity $U_s$, not the
physically more complete relative form $(U_s - V_{\text{air}})$ -- because a head is a function of
the drivers alone and is never allowed to read the layer's own solved potential $\phi$
(see [Kirchhoff layers](../../concepts/layers.md)): letting the drag depend on the air's own flow would put a term into the
Newton residual that the Jacobian $A \operatorname{diag}(g') A^\top$ does not account for, silently breaking
the solver's own convergence theory. The absolute form is therefore not a simplification of
convenience but the one shape the framework's `HeadTerm` protocol admits; the relative form
would need an element-side formulation (a batched monotone root on
$R \lvert Q \rvert Q - \tfrac{1}{2} f_i\, \rho\, (U_s - Q/A)^2\, T L / A_{\text{air}} = \Delta p$) rather than a head.

## The two-film coupling, and its lagged pass

Total sulfide in the water and H₂S in the
headspace air are two separate `TransportLayer` species, coupled by one closure,
`H2STransfer`, computing a flux $J = K_L a\, V_{\text{wet}} (f C_S - C_G/H)$ per manhole from the free-
sulfide fraction $f$, Henry's constant $H(T)$ and a two-film transfer coefficient $K_L a$,
and writing $+J$ onto the air side and $-J$ onto the water side (equal in moles of S by
construction, checked node by node). The lateral inflow-concentration
drivers, `bod_in`/`sulfide_in`, enter the water-quality layer's source term the same way
every other nodal load does -- $\text{inflow} \times \text{concentration}$ (m³/s times kg/m³, kg/s per species)
-- written by a separate closure, `LateralLoads`, registered before `H2STransfer` so its
load and the two-film transfer's own source term add rather than one overwriting the other.
Both correlations read per-PIPE quantities --
hydraulic radius, slope, wetted velocity, mean depth -- at each manhole's own SINGLE
outgoing pipe (a tree guarantees exactly one), gathered once at construction into a
per-manhole index (`out_pipe`) rather than looked up per call; the same gather
underlies the Pomeroy-Parkhurst sulfide-generation reaction. The closure reads the PREVIOUS
pass's concentrations -- the framework's ordinary lagged coupling (see [Layers](../../concepts/layers.md)) -- so
`coupling="pingpong"` carries a one-pass lag on the cross-phase source terms exactly as the
building application's heat/species coupling does, and `coupling="iterate"` removes it by
Hensen's-onion successive substitution when the two quality layers' own `iterate_tol` is set.

## Headspace air

`Headspace` is a `ConstitutiveLaw` with $\Delta p = R(h)\,\lvert Q \rvert Q$ and

$$
R(h) = \frac{f_{\text{air}} L \rho}{2 D_h A_{\text{air}}^2}
$$

inverted in closed form with a laminar blend near zero. The air-side wetted perimeter is
$P_{\text{air}} = \pi D - P(h) + T(h)$ — the dry wall plus the water surface acting as a moving
wall — and $D_h = 4A_{\text{air}}/P_{\text{air}}$.

`Drag` is a head term: $D = \tfrac{1}{2} f_i \rho U_s \lvert U_s \rvert T L / A_{\text{air}}$ with
$U_s = c_s V_w$. The full branch law is $\Delta p = R\lvert Q\rvert Q - D - B$, where $B$ is the
**same `Stack` buoyancy term the building application uses** — reused unchanged. `air_heads`
adds `Drag` and the two `Stack` terms (headspace and leak) up in an `EdgeHeads` closure that
writes the layer's head driver `"air.edge_head"`, after the hydraulics closure that writes the
`sewer.*` drivers `Drag` reads.

The drag sign convention is documented and was verified rather than assumed: the term returns
$+D$ so that with both ends at ambient the air moves from the upstream manhole toward the
downstream one, the direction the wastewater drags it. The opposite sign would reverse the air
flow against the Pescod and Price measurements.

## Quality

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
