# Street air quality: physics

The equations the street air quality application solves and the assumptions behind them. The options that select between the variants are on [Choosing the closures](closures.md).

## The street balance

For a canyon street $i$ of length $L_i$, width $W_i$ and height $H_i$, holding a well-mixed pollutant mass concentration $x_i$ (kg/m³),

$$
L_i W_i H_i \frac{dx_i}{dt} = \sum_r Q_r\, x_{\text{up}(r)} - u_{d,i} (L_i W_i)\, x_i + u_{d,i} (L_i W_i)\, x_b + S_i
$$

This is the framework's transport equation, $dx/dt = M x + N x_b + \text{sources}/\text{capacity}$, with `capacity` $= L W H$ and `carrier` $= 1$ (the transported quantity is the concentration itself, not a scaled energy or momentum density as in the heat layer). The routed inter-street terms $Q_r\, x_{\text{up}(r)}$ play the advective role, and the roof exchange $u_d (L W)(x_b - x_i)$ plays the role a conduction edge plays for heat. A street network is therefore another `TransportLayer`, over a graph whose edges are

- `route`: street to street, through an eliminated junction;
- `vent`: only the junction's closure share, the `to_atmosphere`/`from_atmosphere` imbalance `node_closure` computes for that junction end (a street's own flux rides the `route` edges);
- `exchange`: roof exchange with the atmosphere boundary node.

The `exchange` pair is written as two advective edges rather than as a conduction edge, because a conductance is fixed at construction whereas $u_d$ varies with the forcing at every step. The two forms agree to 1.2e-16.

## Junctions are eliminated

A junction with its own storage would need a potential to drive flow between the streets meeting there. The only physical constraint at an intersection is that the flows balance; there is no independent state for a junction to hold. The prescribed-flow edges (`route`, `vent`) that carry that balance are fixed-flow edges, so a Kirchhoff layer built only from them has an identically zero conductance matrix and is singular by construction.

Eliminating the junction avoids that. What a junction does is redistribute the fixed street-end fluxes among each other and the atmosphere. This is computed once per solve, outside any Kirchhoff layer, as a routed flux (`routing_matrix`, the non-crossing fill below) plus a roof closure (`node_closure`, MUNICH's roof-exchange correction, which lands asymmetrically on the side needed to keep every one of a junction's in and out totals matched). The result is written onto the `route`, `vent` and `exchange` edges as that step's flow, like a head, before the layer's own solve runs.

## Flow signs

Every `route`, `vent` and `exchange` flow is non-negative; direction lives in the topology. Each of the three edge kinds is added in both directions (`direction="in"` and `"out"`, or the ordered pair of distinct street ends at a junction), so a flow that reverses under a wind-direction change is a different, non-negative amount on the other edge of the pair. This reproduces MUNICH's one-way dead ends: a street whose one open end is unambiguously downwind gets zero flux on the edge that would carry flow the other way, because that edge's output is zero there. No special-cased dead-end branch is needed.

## Canyon wind and roof exchange

`canyon_velocity` (Soulhac, Perkins and Salizzoni 2008; Soulhac et al. 2011, the SIRANE form) and the exponential in-canyon profile (Kim et al. 2018, 2022, the MUNICH form) both turn a reference wind into an along-street canyon velocity. They differ in their treatment of the boundary-layer profile above the canyon and in the von Karman constant (0.40 against MUNICH's 0.41).

`exchange_velocity` has a SIRANE branch, $u_d = \sigma_w/(\sqrt{2}\,\pi)$ (Soulhac et al. 2011 Eq. 5; Kim et al. 2018 Eq. 3; Kim et al. 2022 Eq. B10), and a Schulte mixing-length branch, with `ASPECT_RATIO_EXCHANGE_BETA` $= 2/(\sqrt{2}\,\pi)$ fixed by matching the SIRANE form at aspect ratio 1 (Kim et al. 2018, p. 613). The exchange coefficient is $\sigma_w/(\sqrt{2}\,\pi)$ (`EXCHANGE_SIGMA_W_RATIO` = 0.225079…), as in MUNICH's source, not $\sigma_w/\sqrt{2\pi}$. The two pairs are selectable independently (`canyon_wind="bessel_profile"` or `"exponential_profile"`, `roof_exchange="turbulent_velocity"` or `"aspect_ratio_scaled"`) because the two reference codes do not always pair them the same way.

**The Soulhac form.** With $d_i = \min(W/2, H)$ and the in-canyon roughness $z_{0,b}$, the shape parameter $c$ is the root of

$$
\frac{1}{2} \frac{z_{0,b}}{d_i}\, c = \exp\left(\frac{\pi}{2} \frac{Y_1(c)}{J_1(c)} - \gamma_E\right)
$$

(`bessel_shape_residual`), solved by `solve_monotone` on $[10^{-4}, 3]$ rather than by MUNICH's 0.01-wide brute-force grid. The roof-level wind on which the in-canyon integral is written is

$$
u_h = u_* \sqrt{\frac{\pi}{\sqrt{2}\, \kappa^2 c} \left[Y_0(c) - J_0(c) \frac{Y_1(c)}{J_1(c)}\right]}
$$

The four Bessel functions are wrapped from `torch.special.bessel_j0/j1/y0/y1` in `torch.autograd.Function`s, because the torch kernels carry no gradient, and `solve_monotone` differentiates its own residual.

**Tall streets.** The neutral $\sigma_w$ goes negative when a street is taller than $1.25\,h_{\text{abl}}$. `build_model`'s `pblh_floor=True` (the default) applies MUNICH's guard, raising the boundary-layer height to at least the tallest street. With `pblh_floor=False`, `exchange_velocity` raises an error on such a step rather than clamping it.

## Direction averaging and junction routing

The wind direction is not a single number. MUNICH spreads it with $\sigma_\theta = \min(\sigma_v/U,\ 10^\circ)$ and takes $n_\theta = \lfloor \sigma_\theta \text{ in degrees} \rfloor$ samples, clamped to $[1, 10]$, uniformly on $\pm 2 \sigma_\theta$ with unnormalised weights (`sigma_theta_turbulence_intensity`, `n_theta_rectangle_rule`, `direction_offsets`). The canyon velocities are computed once from the mean direction and are not recomputed per sample. Only the in/out classification and the angular ordering at each junction change from sample to sample, which is what makes the averaged flux matrix comparable with MUNICH's.

MUNICH's `ComputeAlpha` walks a junction's inflows counter-clockwise and its outflows clockwise, greedily filling each inflow/outflow cell with $\min(\text{remaining}_{\text{in}}, \text{remaining}_{\text{out}})$. That is the transportation-problem north-west-corner rule applied to the two ordered marginals, with the closed form $F_{pr} = \max\bigl(0, \min(A_p, B_r) - \max(A_{p-1}, B_{r-1})\bigr)$ on their cumulative sums $A$, $B$ (`routing_matrix`). Written this way the fill is batched (every junction, every wind direction, every forcing step in one call) and differentiable (piecewise linear in the marginals). The one combinatorial part, which order the inflows and outflows are visited in, is confined to an argsort (`order_slots`) run without gradient: the order can only change at exactly tied angles, so no gradient through it is needed or defined.

## Chemistry

The three species relax to the state satisfying $J\,[\mathrm{NO_2}] = k\,[\mathrm{NO}][\mathrm{O_3}]$
at the same NOx ($[\mathrm{NO}]+[\mathrm{NO_2}]$) and Ox ($[\mathrm{NO_2}]+[\mathrm{O_3}]$) as the
input, conserved in molar terms: it is an equilibrium, not a rate, and applying it twice changes
nothing. $J$ (the `J_NO2` driver, 1/s) is supplied by you; `j_no2_zenith_table(zenith_deg,
attenuation=1.0)`
(see [Build and run a street network](networks.md#add-chemistry)) gives the clear-sky rate from MUNICH's 11-point tabulation, piecewise-linear between the
tabulated zenith angles. **Solar geometry is not computed** for this form — you pass the zenith
angle you want.

The rate $k$ of NO + O3 is evaluated at the driver `temperature` (K), which is **required**:
with no constant override, a missing `temperature` driver raises by name rather than assuming a
value. The `Photostationary` reaction (`noodl.apps.street_aq`) defaults to the JPL rate
`k_no_o3_jpl2003(T) = 3.0\times10^{-12}\exp(-1500/T)` cm³ molecule⁻¹ s⁻¹ (NASA/JPL 2003, as in
MUNICH); to use a constant 298 K rate instead of evaluating it at the air temperature, pass
the constant override `k_no_o3=K_NO_O3` (`noodl.apps.street_aq.K_NO_O3`, the 298 K value in
m³ kg⁻¹ s⁻¹ for the kg/m³ state); with that override `temperature` is not read.

### The SIRANE preset

`preset="sirane"` (the default) takes SIRANE's own rate and floor, both evaluated at the driver
`temperature`:

- the rate `k_no_o3_soulhac2011(T) = 1.325\times10^{6}\exp(-1430/T)` m³ mol⁻¹ s⁻¹, at SIRANE's own
  ground-level air temperature ($T_g$: its preprocessed value, not the input temperature away
  from neutral conditions — cooler when stable, warmer when unstable);
- a floor on the photolysis-to-rate ratio, $K = \max(J/k,\ 2\ \text{ppb})$ (`PHOTOSTATIONARY_FLOOR_PPB`),
  so NO and O3 no longer titrate to whichever runs out at night. `floor_ppb` overrides the floor
  (0 disables it).

The floor and the equilibrium are solved in ppb, via the driver `molar_volume` (m³/mol; falls
back to the ideal-gas `molar_volume(T)` at 101325 Pa when not given). `molar_volume` enters
nowhere else: without a floor the equilibrium is the same whatever unit it is solved in.

`solar_elevation(latitude_deg, day_of_year, hour)` and `j_no2_elevation_cloud(elevation_deg,
cloud_octas=0.0)` give SIRANE's own photolysis rate $k_1$, a corrected form of Soulhac et al.
(2011) Eq. 32:

$$k_1 = \frac{1}{60}\max\{0,\ 0.5699 - [9.056\times10^{-3}(90 - \alpha)]^{2.546}\}
\left[1 - 0.75\left(\frac{N}{8}\right)^{3.4}\right]$$

with $\alpha$ the solar elevation (degrees) and $N$ the cloud cover (octas, 0-8). The bracket is
clipped at zero, which makes $k_1$ zero for $\alpha \le 1.458°$ and so at night; the clear-sky
overhead value is $9.50\times10^{-3}$ s⁻¹. The elevation follows Cooper's declination,
$\delta = 23.45°\sin(360°(284+n)/365)$ with $n$ the day of year, and the hour angle
$\omega = 15°(h-12)$ with $h$ the clock time taken as local solar time — no longitude, no
equation-of-time correction.

This differs from the published Eqs. 31-32 (Soulhac et al. 2011): the $k_3$ prefactor is ten
times the paper's (1.325e6, not 1.325e5), its temperature is SIRANE's ground-level value rather
than an input temperature, the split works in ppb where the paper only says "molar
concentrations", there is a 2 ppb floor on $K$, and $k_1$ is clipped at zero at night and below
1.458°; none of this is stated in the paper. The SIRANE reference data follow these corrected
forms, not the printed equations.

Applied pointwise — to a street's concentration, a receptor, or the grid — this equilibrium is
exactly `street_steady`'s converged fixed point (see its docstring): transport treats NO, NO2 and
O3 alike, so it carries NOx and Ox as passive tracers unchanged by the reaction, and the
equilibrium depends on nothing else.

## The above-roof plume kernel

The kernel is SIRANE v2.1's own street-plume mechanism, reverse-engineered from SIRANE
reference data for single-street cases (one isolated street under imposed meteorology, its
concentration grid) rather than derived from the paper's closed forms: SIRANE does not
evaluate a Gaussian for each source-receptor pair, it tabulates one plume trajectory per hour and reads every
pair off that table (Soulhac, Salizzoni, Cierco and Perkins 2011, *Atmospheric Environment*
45:7379-7395; equation numbers below are that paper's).

For each hour, `plume_table` builds the trajectory:

- Meteorology is evaluated at `H_R`, SIRANE's reflection height, with `sigma_v` and
  `sigma_w` floored at SIRANE's own defaults (0.5 and 0.3 m/s; either floor can be turned
  off) and no floor on the plume's advection speed.
- The trajectory advances in 10 s steps: the plume's height follows an empirical centre
  law (below) from the previous step's `sigma_z`, its speed is the wind at that height, and
  `sigma_y`, `sigma_z` follow Eqs. (28)-(29), each evaluated at a time offset (`tau_y`,
  `tau_z`, also fitted) behind the step's own clock. The vertical profile at `H_R` is a
  Gaussian of that height with its image in the roof and its image in the inversion.
- The trajectory is resampled onto a 10 m distance grid; every source-receptor pair is
  then a linear read of that grid rather than a fresh evaluation.

Sources: a street of coordinate length `L` is cut into `floor(L / 10) + 1` equal
sub-sources, each a flat-top crosswind profile of the sub-source's projected length
`(L/n)|sin phi|` spliced into the Gaussian tail, and a vertical profile capped at `min(10 / W, 1 / (H (1 - |sin phi|)))`
(`W`, `H` the street's width and height, `phi` the angle between the street and the wind).
A junction is one point source of the width and height of the streets meeting there,
capped at `1 / H`. A receptor's height is not used -- the above-roof field has no vertical
structure of its own.

A pair contributes exactly zero upwind of the receptor, more than four crosswind standard
deviations beyond the source's flat top, or beyond the downwind cut-off: the x-size of
SIRANE's meteorology grid cell over the cosine of the wind's angle to it (`meteo_cell_dx`),
or 700 m (`DOWNWIND_CUTOFF_M`) when no cell size is given. That is SIRANE's own rule,
valid for cells of about 700 m or larger; smaller cells switch SIRANE to a cell-based far
field that this kernel does not model.

Four elements of the mechanism are fitted to SIRANE's output rather than derived from the paper; they are listed with their defaults under [Above-roof concentration](above-roof.md#empirical-inputs). The steady relation and its solve are on the same page.


## Unsteady puffs above the roofs

The steady relation $C_{\text{ext}} = C_{\text{bg}} + K F$ moves a roof flux to every downwind street at once. That is right for an hour of traffic emissions, but not for a cloud that passes in minutes: a short release has to take the travel time to reach a street downwind, and its peak has to pass. With `puffs=PuffOptions(...)` the steady relation is replaced by Gaussian puffs above the roofs, fed by the streets and feeding back into them, after SIRANERISK (Soulhac et al. 2016). The streets stay what they were: one well-mixed box each. What changes is who writes their boundary `C_ext`. How to run it is on [Unsteady above-roof puffs](puffs.md).

### What is kept from SIRANERISK

SIRANERISK (Soulhac et al. 2016) is SIRANE's unsteady version for accidental releases. Above
the roofs it carries the material in Gaussian puffs whose spreads follow a covariance
equation with Lagrangian time scales (Eqs. 4-6 of that paper); inside the streets it follows
the material in Lagrangian boxes that move along each street (its Sect. 2.2). The puffs are
fed by the streets' roof fluxes and the junctions' vertical fluxes and feed back into the
streets through the same roof exchange as the steady model. The comparison of its
above-roof puffs with a Lagrangian model over flat ground is in the HARMO 14 paper, not
the 2016 one.

noodl keeps the structure (puffs above the roofs, two-way through the roof exchange) and
departs from SIRANERISK where the steady model has to remain its limit:

| | SIRANERISK | noodl |
|---|---|---|
| Spreads of a puff | its own covariance equation with Lagrangian time scales | SIRANE's steady plume laws, read from the steady kernel's own table at the puff's age |
| Inside a street | Lagrangian boxes moving along the street | one well-mixed box per street, as the steady model |
| Along the wind | plain Gaussian puffs | a slug of the length the puff train covers in one step, alone (`"none"`) or convolved with a Gaussian of `sigma_y` (`"sigma_y"`, the default) |
| Puff records | one per emission | one per source (street, junction, release) per step; a street's sub-sources are expanded at evaluation |
| Emission | within the step | one step late: the closure emits at the start of a step what the street step before it moved |
| Housekeeping | not described in the 2016 paper | fixed capacity per batch instance, puffs retired past the domain and below a relative faint threshold, with every retired mass counted |

The reason for the first and third choices is the consistency claim the paper makes: under
constant meteorology and emissions the puff model must converge to the steady kernel as the
step shrinks (with along-wind spreading off). That holds exactly as built; it
would not hold with SIRANERISK's covariance equation, whose spreads are not the steady
table's.

### Puffs and their mass

At every street step each source sheds one puff: each street its roof flux, each junction
its vertical flux, and each explicit above-roof release its mass. The puff's mass is
exactly what backward Euler moved over the step, so the streets and the puffs together
conserve mass:

- street `s`: $m_s = q_s\,(x_s - C_{\text{ext},s})\,\Delta t$ (kg), with $q_s$ the roof
  exchange flow (m³/s), signed: a street cleaner than the air above it emits a negative
  puff, the same linear treatment as the steady kernel;
- junction `j`: $m_j = \sum_{\text{out}} q\,(x - C_{\text{bg}})\,\Delta t$ under
  `junction_source="upward"` (the default), or that minus
  $\sum_{\text{in}} q\,(C_{\text{ext}} - C_{\text{bg}})\,\Delta t$ under `"net"`;
- release `r`: its rate (kg/s) times the overlap of its window with the step (s).

A puff of age $a$ (s) has travelled the distance $x(a)$ (m) along the wind. The puffs read
the steady kernel's own 10 m distance table (`plume_table`'s wind speed $u$, crosswind
spread $\sigma_y$ and roof-level vertical profile $p_z$, plus the scan's vertical spread
$\sigma_z$ and centre height $z_c$ on the same grid; `puff_tables`), and $x(a)$ is the
exact travel distance at that table's piecewise-linear speed $u(x)$. Along the wind a puff
is a slug spanning $[x(a - \Delta t/2), x(a + \Delta t/2)]$ from its source, so the slugs
of a continuous source tile the wind axis for any step $\Delta t$. A street's puff is its
`floor(L/10) + 1` sub-sources, all moved by the same displacement, each with the flat-top
width and roof-level cap of the steady kernel. A junction's puff is one point with the
junction's width and cap; a release is one point with neither.

A puff's concentration (kg/m³) at a receptor at along-wind offset $s$ (m) from it and
crosswind offset $y$ (m) from a sub-source of weight $w_i$ is

```
C = m * w_i * g_x * f_y(y; sigma_y, width_i) * P_z
```

with $f_y$ (1/m) the steady kernel's crosswind profile and $P_z$ (1/m) its vertical
profile, both read from the table **at the receptor's own distance** $x(a) + s$. At roof
level $P_z = \min(p_z, \text{cap}_i)$; at a free receptor of height $z$
(`concentration_at`) it is the Gaussian at $z_c$ with its images in the canopy top $H_R$
and the boundary-layer top $h$, without the cap. The along-wind factor $g_x$ (1/m) is set
by `along_wind`:

| `along_wind` | $g_x$ inside the slug | Steady limit |
|---|---|---|
| `"sigma_y"` (default) | the slug, of length $\ell$, convolved with a Gaussian of $\sigma_x = \sigma_y(x(a))$: $[\mathrm{erf}((s - s_{lo})/(\sqrt{2}\sigma_x)) - \mathrm{erf}((s - s_{hi})/(\sqrt{2}\sigma_x))]/(2\ell)$, with the table read clamped to the slug's ends $s_{lo}$, $s_{hi}$ | the steady kernel smoothed along the wind: a floor of about 4.7 % of the largest kernel entry against the plain kernel |
| `"none"` | $1/(\tau\,u(x(a) + s))$, with $\tau$ the slug's age span (s): the slug is a segment of the steady plume, not a uniform top-hat | the steady kernel itself, to rounding, for any step |

### Wind-shear stretching

With `PuffOptions(shear=True)` (Sykes and Henn, 1995) a vertical shear $S = dU/dz$ of the
tables' wind profile, at $z_s = \max(z_c, 0.65\,\sigma_z)$, grows a covariance
$\sigma_{xz}$ between the along-wind and vertical positions in a puff, and through it an
extra along-wind variance:

$$
\frac{d\sigma_{xz}}{dt} = S\,\sigma_z^2, \qquad
\frac{d\sigma_{x,\text{shear}}^2}{dt} = 2\,S\,\sigma_{xz}.
$$

Both are per-puff state (`"puff.sxz"`, `"puff.sx2_shear"`, zero at birth), integrated per
step with $S$ at the step's middle age held over the step and $\sigma_z^2$ quadratic in age
through its values at the step's start, middle and end; this is exact for a constant $S$
and $\sigma_z \propto t$ ($\sigma_{xz} = S a^2 t^3/3$,
$\sigma_{x,\text{shear}}^2 = S^2 a^2 t^4/6$ for $\sigma_z = a t$). $S$ is the derivative
of `plume.wind_speed` in closed form, $u_* \phi_m(z'/L)/(\kappa z')$ with
$z' = z - d + z_0$, so gradients reach the wind profile's $u_*$ and $L$. Under `puff_dt`
a puff is born at a positive age with zero moments, missing their growth over that age
(cubic and quartic in it, small for intervals of tens of seconds).

The puff is then a correlated $(x, z)$ Gaussian, of along-wind variance
$\sigma_x^2 = \sigma_y(x(a))^2 + \sigma_{x,\text{shear}}^2$, convolved along the wind
with the slug. It is evaluated as the vertical profile times the conditional along-wind
profile: each term of $P_z$, centred at $z_k$ ($z_c$, $2H_R - z_c$, $2h - z_c$), multiplies
the slug convolved with a Gaussian of the conditional variance
$\sigma_x^2 - \sigma_{xz}^2/\sigma_z^2$, shifted along the wind by
$\beta (z - z_k)$, with $\beta = \sigma_{xz}/\sigma_z^2$ for the plume and $-\beta$ for
its two images (a reflection in $z$ flips the sign of $\sigma_{xz}$). $\beta$ and the
conditional variance are the puff's own, at $x(a)$; the vertical terms are read at the
receptor's distance, as without shear. Every term integrates to one along the wind, so at
every height the line integral of a sheared puff along the wind is the unsheared one: the
shear moves material along the wind and neither makes nor loses any. At roof level the
plume and its canopy image have the same shift, and the factor is $\min(p_z, \text{cap}_i)$
times the terms' weighted mean of their along-wind profiles.

Pairs more than `cutoff_sigma` crosswind standard deviations beyond the flat top, or (under
`"sigma_y"`) along-wind standard deviations ($\sigma_x$, with the shear part when it is
on) beyond the slug's ends, are exactly zero. A street's own puffs do not count in its own `C_ext`, as in `street_kernel`.

### The step

`PuffClosure` runs inside the street model's own step, after the flow closure. At the
start of step $k$, from the state at that time, it

1. emits the puffs of step $k-1$: the masses above, from the street concentrations at the
   end of step $k-1$ and the flows and `C_ext` it stored then;
2. advances every puff by $\Delta t$ on the current wind direction;
3. retires the puffs that have left (`retire`, above) and those fainter than `drop_below`
   times the brightest puff of their batch instance;
4. evaluates `C_ext` at every street midpoint at roof level and writes it as
   `"<layer>.x_boundary"`: the caller's background driver plus the puffs.

The street layer then takes its step with that boundary. The puffs live in the model's
state (the `"puff.*"` keys, `PuffState.to_state()`), so `noodl.couple.union` couples a
puff street model to a building unchanged. Because the closure runs at the start of a step
it cannot see that step's end, so the emission lags the flux by one step; the lag is an
error of first order in $\Delta t$ (see [Verification](../../reference/verification/street.md#unsteady-puffs)).
