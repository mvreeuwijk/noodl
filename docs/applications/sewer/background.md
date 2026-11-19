# Sewers: theoretical background

Background for the [sewer application](index.md), moved from the framework theory page ([theory](../../concepts/theory.md)). The water-distribution part is on the [water background page](../water/background.md).

## 9. Sewers and water distribution

**Two sewer water sides, and the water-distribution side.** The sewer's kinematic-wave path is
CONTINUITY-first. A gravity sewer runs free-surface: on the dendritic tree of section 1's own
analysis, normal flow at every pipe is set by continuity alone -- the net inflow upstream of it
(`cycles.particular_flow`, closed form on a tree, no Newton solve at all) -- and depth follows
from Manning's law given that flow, not the other way round; a head difference does not drive the
water there, so a Kirchhoff layer on head is the wrong shape for it. The dynamic wave is the
case where head differences do drive the flow, and there the sewer IS a Kirchhoff layer on head.
SWMM's dynamic-wave update (the Saint-Venant momentum equation integrated over a step) is, for
frozen depth-dependent coefficients, linear in the end-head difference of each conduit,
$q = c_0 + k\,(h_1 - h_2)$ with $k \ge 0$, and a junction's storage adds a monotone term in its
own head, so one pass is an ordinary `KirchhoffLayer` solve (§1, §3) with the outfalls as
boundary heads. The coefficients depend on the heads, so the pass is iterated
(`coupling="iterate"`) until the heads stop changing. SWMM's own Picard trials, which relax and
average its iterates, and noodl's iterated implicit head solve share one fixed point: a state
that reproduces its own coefficients and satisfies the conduit and junction equations. noodl
converges to that fixed point and does not replay the trials (see the
[sewer page](index.md#dynamic-wave)). Conduits may form loops and may be flat or
adverse, which the continuity-first path cannot represent.

**Surcharge as instantaneous continuity.** Above its crown a junction is pressurised: SWMM's
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

**Tie rules.** Three of SWMM's switches have no fixed point on a band of states, because
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
at such a state is that of the constrained solution; see the [sewer page](index.md#intentional-differences-from-swmm)
for the costs against SWMM.

**The manhole-level storage idealisation, and what it is not.** On the kinematic-wave path, with `storage=True`,
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
loop over manholes -- the loop count is the tree depth, not the manhole count, satisfying
the rule that no per-step path loops in Python. The pipe carrying each
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

**The `air.phi` datum.** Every `leak` edge's `Stack` carries a manhole term that cancels
against the manhole's own ground level (`z_ref - z_path = 0` there) but an AMBIENT term that
does not, because `ambient` has no `ground` attribute of its own and so falls back to a
`0.0` datum rather than to the manhole's; the same `0.0` is used identically at the outfall's
open-air edge and at `ambient`'s own fixed boundary condition (`air.phi_boundary = 0`), so
every air pressure in the solve is shifted by the same constant -- `air.phi` is
DATUM-REFERENCED to this convention, not to a physical zero, and every head, refusal and
verification row reads only DIFFERENCES between manholes, which the convention leaves
unaffected.

**The headspace momentum balance, and the head-cannot-see-`phi` rule.** Air in the
headspace above the flow is driven by three terms per pipe: Darcy-Weisbach wall
friction on the air itself, buoyancy from the density difference between a warm manhole
shaft and ambient (the SAME `Stack` head the building application's stack effect uses, one
full-node air-density driver, `rho_air_nodes`), and drag from the moving water surface
dragging the air along above it. That last term is written as a head term --
$D = \tfrac{1}{2} f_i\, \rho_{\text{air}} U_s \lvert U_s \rvert\, T L / A_{\text{air}}$ in the ABSOLUTE water-surface velocity $U_s$, not the
physically more complete relative form $(U_s - V_{\text{air}})$ -- because a head is a function of
the drivers alone and is never allowed to read the layer's own solved potential $\phi$
(§1, `heads.py`): letting the drag depend on the air's own flow would put a term into the
Newton residual that the Jacobian $A \operatorname{diag}(g') A^\top$ does not account for, silently breaking
the solver's own convergence theory. The absolute form is therefore not a simplification of
convenience but the one shape the framework's `HeadTerm` protocol admits; the relative form
would need an element-side formulation (a batched monotone root on
$R \lvert Q \rvert Q - \tfrac{1}{2} f_i\, \rho\, (U_s - Q/A)^2\, T L / A_{\text{air}} = \Delta p$) rather than a head.

**The two-film coupling, and its lagged pass.** Total sulfide in the water and H₂S in the
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
pass's concentrations -- the framework's ordinary lagged coupling (§5, §7 above) -- so
`coupling="pingpong"` carries a one-pass lag on the cross-phase source terms exactly as the
building application's heat/species coupling does, and `coupling="iterate"` removes it by
Hensen's-onion successive substitution when the two quality layers' own `iterate_tol` is set.

**Verification.** The sewer's water flows, depths, velocities and a first-order-decay
tracer are checked against pyswmm/SWMM 5.2.4 under KINWAVE routing; the air
side against Pescod and Price's laboratory ratios and Tyneside field range and
the H₂S two-film transfer's Henry-equilibrium fixed point; the water-distribution
application's heads, flows, pump gain, tank trajectory, pressure-driven demand, loop
consistency, gradients and trace quality against EPANET 2.2 through wntr. $f_i$ and
$f_{\text{air}}$ are CALIBRATED to a single source (Pescod and Price Test 8), not
literature-pinned, so the air-side ratio and Tyneside-band checks are consistency checks
against the calibration source rather than independent validation. See
[Sewers](index.md) and [Water distribution](../water/index.md#verification).
