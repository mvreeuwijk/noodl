# Street air quality: theoretical background

Background for the [street air quality application](index.md), moved from the framework theory page ([theory](../../concepts/theory.md)).

## 8. Street canyons as a transport layer

**The street balance IS the framework's transport equation, again.** For a canyon street
$i$ of length $L_i$, width $W_i$ and height $H_i$, holding a well-mixed pollutant mass
concentration $x_i$ (kg/m³),

$$
L_i W_i H_i \frac{dx_i}{dt} = \sum_r Q_r\, x_{\text{up}(r)} - u_{d,i} (L_i W_i)\, x_i + u_{d,i} (L_i W_i)\, x_b + S_i
$$

is exactly $dx/dt = M x + N x_b + \text{sources}/\text{capacity}$ (section 6 above), with
`capacity` $= L W H$, `carrier` $= 1$ (the transported quantity IS the concentration, not a scaled energy
or momentum density as in section 7's heat layer), the routed inter-street terms
$Q_r\, x_{\text{up}(r)}$ playing the advective role, and the roof exchange $u_d (L W)(x_b - x_i)$ playing
the same role a conduction edge plays for heat. So the street network is not a third
physics either: it is a third `TransportLayer`, over a graph whose edges are `route`
(street-to-street, through an eliminated junction), `vent` (ONLY the junction's closure
share -- the `to_atmosphere`/`from_atmosphere` imbalance `node_closure` computes for that
junction end; a street's own flux rides the `route` edges) and `exchange` (roof exchange
with the atmosphere boundary node), with `carrier` $= 1$. The `exchange` pair is written as
two ADVECTIVE edges rather than as the framework's conduction kind because a conductance is
fixed at construction, whereas $u_d$ varies with the forcing at every step; the two agree
to 1.2e-16, pinned by `test_the_exchange_edge_pair_is_exactly_the_conduction_term`.

**Why the intersections are ELIMINATED rather than modelled.** A junction with its own
storage would need a potential (a pressure-like quantity) to drive flow between the streets
meeting there, but the only physical constraint at a street intersection is that the
*flows* balance -- there is no independent state for a junction to hold, and the
prescribed-flow edges (`route`, `vent`) that would carry that balance are FIXED-FLOW, not
potential-driven, edges. A Kirchhoff layer built entirely from fixed-flow edges has no
element relating a node's potential to anything, so its would-be conductance matrix is
identically zero -- singular by construction. Eliminating the
junction sidesteps that non-problem entirely: what a junction actually does is redistribute
the fixed street-end fluxes among each other and the atmosphere, and that redistribution is
computed once per solve, outside any Kirchhoff layer, as a routed flux (`routing_matrix`,
the non-crossing fill below) plus a roof closure (`node_closure`, MUNICH's roof-exchange
correction, which lands asymmetrically on the side needed to keep every one of a junction's
in/out totals matched, per `StreetNetworkTransport.cxx:2980-3014`). The result is written
straight onto the `route`/`vent`/`exchange` edges of the `TransportLayer` as this step's
flow, exactly like a head, before that layer's own solve runs at all.

**Why every `route`, `vent` and `exchange` flow is non-negative, with direction living in
the topology.** Each of these three edge kinds is added to the graph in BOTH directions
(`direction="in"`/`"out"`, or the ordered pair of distinct street ends at a junction), so a
physical flow that reverses under a wind-direction change does not need its own sign: it is
a different, non-negative amount on the OTHER edge of the pair, exactly as
`assert_forward_oriented` already requires of every fixed-flow edge in the framework. This
reproduces, for free, MUNICH's one-way dead ends: a street whose one
open end is unambiguously downwind gets zero flux on the edge that would carry flow the
other way, not a negative one, because that edge's non-negative closure output is zero
there -- no special-cased dead-end branch is needed anywhere in `StreetFlows`.

**The two canyon-wind closures and the two exchange closures.** `canyon_velocity` (Soulhac,
Perkins and Salizzoni 2008; Soulhac et al. 2011, the SIRANE form) and the exponential
in-canyon profile (Kim et al. 2018, 2022; the MUNICH form) both turn a reference wind into
an along-street canyon velocity, differing in their treatment of the boundary-layer profile
above the canyon and in their von Karman constant (0.40 against MUNICH's 0.41). `exchange_velocity` similarly has a SIRANE branch
($u_d = \sigma_w/(\sqrt{2}\,\pi)$, S11 Eq. (5), K18 Eq. (3), K22 Eq. (B10), `StreetNetworkTransport.cxx
:3273`; see the street page's [note on this constant](index.md#limitations-and-caveats))
and a Schulte mixing-length branch
(`ASPECT_RATIO_EXCHANGE_BETA` $= 2/(\sqrt{2}\,\pi)$, fixed by matching the SIRANE form at $a_r = 1$, K18 p.
613). Both pairs are selectable independently (`canyon_wind="bessel_profile"` or
`"exponential_profile"`, `roof_exchange="turbulent_velocity"` or `"aspect_ratio_scaled"`)
because the two source codebases do not always pair them the same way, and the parity tests
exercise both pairings; `preset="sirane"` and `preset="munich"` give each code's own pairing.

**The Soulhac form, written out.** With $d_i = \min(W/2, H)$ and the in-canyon roughness
$z_{0,b}$, the shape parameter $c$ is the root of

$$
\frac{1}{2} \frac{z_{0,b}}{d_i}\, c = \exp\left(\frac{\pi}{2} \frac{Y_1(c)}{J_1(c)} - \gamma_E\right)
$$

(`bessel_shape_residual`, solved by `solve_monotone` on $[10^{-4}, 3]$ rather than by MUNICH's
0.01-wide brute-force grid), and the roof-level wind that the in-canyon integral is written
on is

$$
u_h = u_* \sqrt{\frac{\pi}{\sqrt{2}\, \kappa^2 c} \left[Y_0(c) - J_0(c) \frac{Y_1(c)}{J_1(c)}\right]}
$$

The four Bessel functions are the reason `apps/street_aq/canyon.py` wraps
`torch.special.bessel_j0/j1/y0/y1` in `torch.autograd.Function`s: the torch kernels carry
no `grad_fn` at all, and `solve_monotone` differentiates its own residual.

**MUNICH's direction averaging.** The wind direction is not a single number: MUNICH spreads
it with $\sigma_\theta = \min(\sigma_v/U,\ 10^\circ)$ and takes $n_\theta = \lfloor \sigma_\theta \text{ in degrees} \rfloor$
samples, clamped to $[1, 10]$, uniformly on $\pm 2 \sigma_\theta$ with UNNORMALISED weights
(`StreetNetworkTransport.cxx:3567`, `:3568`, `:3562-3616`; `sigma_theta_turbulence_intensity`,
`n_theta_rectangle_rule`, `direction_offsets`). The canyon velocities are computed ONCE from the
MEAN direction and are NOT recomputed per sample -- only the in/out classification and the
angular ordering at each junction change from sample to sample -- which is what makes the
averaged flux matrix comparable with MUNICH's at all.

**The non-crossing routing as the north-west-corner rule.** MUNICH's `ComputeAlpha` walks a
junction's inflows counter-clockwise and its outflows clockwise, greedily filling each
inflow/outflow cell with $\min(\text{remaining}_{\text{in}}, \text{remaining}_{\text{out}})$ -- which is exactly the
transportation-problem north-west-corner rule applied to the two ORDERED marginals, and so
has the closed form $F_{pr} = \max\bigl(0, \min(A_p, B_r) - \max(A_{p-1}, B_{r-1})\bigr)$ on their
cumulative sums $A$, $B$ (`routing_matrix`). Writing it this way rather
than as MUNICH's literal loop makes the fill BATCHED (every junction, every wind direction,
every forcing step in one call) and DIFFERENTIABLE (piecewise linear in the marginals, so
`torch.autograd` sees a gradient through the fluxes), while the one genuinely
combinatorial part -- WHICH order the inflows and outflows are visited in -- is confined to
`order_slots`'s argsort and run under `no_grad`: the ordering is a discrete choice (it can
only change at a measure-zero set of exactly-tied angles), so no gradient through it is
either needed or well-defined, and keeping it out of the traced graph is what lets the
fluxes it feeds stay differentiable.

**Verification.** Thirteen exact formula pairs are checked against MUNICH's own source
(each at the precision its source publishes), and the idealised 12-street case against Kim
et al. (2022) Fig. 1; conservation, junction elimination against hand algebra, and
gradients through the whole model against central differences are checked as well. See
[Street air quality](index.md#verification).
