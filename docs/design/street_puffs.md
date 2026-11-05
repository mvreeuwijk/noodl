# Street puffs: unsteady above-roof transport

The design and the measurements behind `noodl.apps.street_aq.puff`, the unsteady
above-roof puff model of the street application. How to use it is in
[Street air quality: Unsteady above-roof transport](../applications/street_aq.md#unsteady-above-roof-transport-puffs);
this page says what SIRANERISK does, what noodl does differently and why, and what was
measured. The section numbers are the ones the source docstrings cite as "design Sect. N".
All numbers were taken in November 2026 on a shared Windows machine (CPU, float64); timings
on that machine varied by up to 2x between runs, so treat them as orders of magnitude.

Common setup of the accuracy measurements, unless a section says otherwise:
`twelve_street_grid()` (L, W, H = 100, 20, 20 m), u* = 0.5 m/s, wind 0.3 rad, h = 800 m,
neutral, sigma_theta = 6 degrees, H_R = 20 m, z0 = 1 m, d = 13 m, x_max = 700 m, default
empirical inputs, `junction_source="upward"`.

## 1. SIRANERISK and what noodl keeps of it

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
| Spreads of a puff | its own covariance equation with Lagrangian time scales | SIRANE's steady plume laws, read from the steady kernel's own table at the puff's age (Sect. 4.4) |
| Inside a street | Lagrangian boxes moving along the street | one well-mixed box per street, as the steady model |
| Along the wind | plain Gaussian puffs | a slug of the length the puff train covers in one step, alone (`"none"`) or convolved with a Gaussian of `sigma_y` (`"sigma_y"`, the default) (Sect. 4.3) |
| Puff records | one per emission | one per source (street, junction, release) per step; a street's sub-sources are expanded at evaluation (Sect. 4.3) |
| Emission | within the step | one step late: the closure emits at the start of a step what the street step before it moved (Sect. 4.2) |
| Housekeeping | not described in the 2016 paper | fixed capacity per batch instance, puffs retired past the domain and below a relative faint threshold, with every retired mass counted (Sect. 4.8) |

The reason for the first and third choices is the consistency claim the paper makes: under
constant meteorology and emissions the puff model must converge to the steady kernel as the
step shrinks (with along-wind spreading off). That holds exactly as built (Sect. 2); it
would not hold with SIRANERISK's covariance equation, whose spreads are not the steady
table's.

## 2. The steady limits

`twelve_street_grid`, the steady puff train of a unit continuous source at each of the 24
sources (12 streets, 12 junctions; ages dt, 2 dt, ... out to x_max, mass dt) evaluated at
the street midpoints, against `[street_kernel | junction_kernel]` from `plume_table` with
the same inputs. Error: max |K_puff - K| / max K.

**`along_wind="none"`: the kernel itself.** At every step tried (10, 8, 6, 5, 4, 3, 2.5, 2,
1.25, 1, 0.625, 0.3125, 0.15625, 0.078125 s) the error is 1.9e-16 to 9.4e-16. The test
(`test_steady_limit_without_along_wind_spread_is_the_kernel`) asserts 1e-12 at 10, 5, 2.5,
1.25 and 0.625 s, and the same bound in a weakly unstable (h/|L| = 0.165; measured 4.8e-16)
and a weakly stable (h/L = 0.5) hour (`test_steady_limit_in_a_weakly_unstable_hour_is_the_kernel`,
`test_steady_limit_in_a_weakly_stable_hour_is_the_kernel`).
The coupled run against `street_steady_with_plume` (run until `C_ext` changes by less than
1e-12 relative per step) reaches 7.9e-12 at 10 s and 1.6e-11 at 5 s, the stopping rule's
level (`test_coupled_steady_limit_without_along_wind_spread_is_the_steady_model`, slow).

What these checks cannot see: the one-step emission lag. A continuous source's newest
material sits at ages `[dt/2, 3dt/2]` when `C_ext` is evaluated, so the puff train leaves a
gap `[0, x(dt/2))` downwind of every source. No receptor-sub-source pair of this grid lies
in that gap at any of these steps, so the lag's first-order term is invisible here; a
receptor that close to a source (closer than half a step's travel, about 6.5 m at 5 s here) would
see it.

**`along_wind="sigma_y"`: a floor.** The design expected convergence to the kernel smoothed
along the wind with the puff's own profile. As built (Sect. 4.3) the error against the
plain kernel converges at first order in dt to a floor: 0.117 (10 s), 0.0699 (5 s), 0.0577
(2.5 s), 0.0457 (1.25 s), 0.0468 (0.625 s), toward about 0.047. The floor is the along-wind
diffusion the steady Gaussian plume neglects (the slender-plume approximation), a property
of the steady kernel rather than an error of the puffs. The smoothed-kernel reference was
not built. The test (`test_steady_limit_with_along_wind_spread_has_a_converged_floor`)
asserts the floor is below 0.10 for dt <= 5 s and changes by less than 10 % between 1.25 s
and 0.625 s.

## 3. Constraints

The puff mass is what one backward-Euler street step moved, so the street model must be
built with `background="per_street"`, `scheme="implicit"`, one substep, one species,
ping-pong coupling, and no removal or reaction; everything else is refused by name. Under
`coupling="iterate"` the closure would read a relaxed end-of-step state from the second
pass on; a building coupled two-way through `noodl.couple.union` is unaffected, since
`union` re-steps the whole street model from the step start. `union` itself is unchanged:
the puffs live in the street model's state as closure-carried keys.

## 4. The model

### 4.2 Sources and the emitted mass

Each step, from the street state at its start, the closure emits the previous step's
masses: street roof fluxes `q_s (x_s - C_ext,s) dt` (signed), junction vertical fluxes as
excess over the background (`"upward"`: out through the vents; `"net"`: minus what came
back down), and each release's `pulse_mass` over the previous step. The emission lags the
flux by one step, an error of first order in dt (Sect. 2). Above-roof releases are emitted
one step late too, from the same emission point. Emission uses the current step's
background, exact for a background constant in time.

`PulseSource` adds a canopy release to `"<layer>.sources"`; it does not apply
`Street.emission_scale`, because the transport layer that `build_model` builds already
scales the sources, and scaling here would apply it twice.

### 4.3 What a puff is

A puff record carries a mass, an age, a displacement from its source, its source and its
table. A street's puff is evaluated as its `floor(L/10) + 1` sub-sources
(`plume.source_points`), all displaced together, so the steady limit is `street_kernel`,
itself a sum over those sub-sources; one record per street keeps the state small.

**As built, not as first designed.** The design gave the slug the length `ell = U(a) dt`,
centred on the puff, read every profile at the puff's own distance, and promised
convergence to the smoothed kernel under `"sigma_y"`. Two measured problems changed that:

- `ell = U(a) dt` does not tile the axis exactly where the table's speed changes between
  neighbouring puffs. The slug of a puff of age `a` now spans `[x(a - dt/2), x(a + dt/2)]`
  (ages clamped at 0), which tiles for any dt.
- Reading `sigma_y` and `p_z` at the puff's distance for every receptor the slug covers is
  off by O(ell) at the slug's ends (sigma_y grows by ~0.37 m per m there), and the error
  aliases with the sub-sources' along-wind spacing (2.69 m at this wind): at 2, 6 and 8 s
  the slug length is close to a multiple of the spacing and the error spikes to 0.046,
  0.17 and 0.21 of max K, against 0.009-0.036 at 2.5, 3, 4, 5 and 10 s (inside an envelope
  of about 0.026 dt). The error was first order but not monotone in dt.

So **a receptor reads the table at its own distance** `x(a) + s` from the source (`s` its
along-wind offset from the puff):

- `"none"`: the slug is a segment of the steady plume, not a uniform top-hat. Its density
  along the wind is `m / (tau u(x(a) + s))`, `tau` the slug's age span; it integrates to
  `m` over the slug because the integral of `dx / u` across it is `tau`. The steady limit
  is then the kernel to rounding (Sect. 2).
- `"sigma_y"`: the read is clamped to the slug's ends and multiplied by the slug convolved
  with a Gaussian of `sigma_x = sigma_y(x(a))`, density `m / ell`. The Gaussian smooths
  the remaining aliasing; the steady limit is the floor of Sect. 2, not the smoothed
  kernel.

The default is `"sigma_y"` (a passing cloud spreads along the wind); the consistency claim
is stated for `"none"`.

### 4.4 Trajectory and spreads: the steady table in time

The puffs read the steady kernel's own 10 m distance table: `plume_table`'s `u`, `sigma_y`
and roof-level `p_z` bit for bit, plus the scan's `sigma_z` and `z_c` resampled onto the
same grid. A puff's age maps to its distance by the exact travel time of the table's
piecewise-linear `u(x)` (in cell k, `dx/dt = u_k + g_k (x - x_k)`, so `x(t)` is exponential
in `t`, linear where `g_k = 0`), and `advance` moves a puff by `x(a + dt) - x(a)`, exact
for any dt.

The first build read the scan's 10 s time nodes instead, as the design planned. Its steady
limit had a floor that did not fall with dt (kernel level, max |dK| / max K):

| dt (s) | 10 | 5 | 2.5 | 1 | 0.5 |
|---|---|---|---|---|---|
| error | 0.023 | 0.049 | 0.056 | 0.053 | 0.053 |

Two causes, both in the tables: the scan moves the plume at its node speed, which jumps by
up to 7 % each time the centre height passes a whole metre, while the kernel divides by the
speed interpolated on the distance grid (a 7.0 % speed mismatch, which accounts for the
whole far-field floor); and `P_z` computed from
`sigma_z(0) = 0` diverged in the first 10 s, where the kernel holds `P_z0 = P_z1`. Reading
the kernel's own grid removes both.

When the meteorology changes, the tables are rebuilt and live puffs read the new table at
their age; when the wind direction turns, the source field (sub-source widths and caps, a
projection of each street on the wind) is rebuilt too, and the puffs' further path turns.

**Stability classes.** The tables share `plume_table`'s classification, SIRANE's (ADMS's)
classes in `h/L`: the neutral class `-0.3 <= h/L <= 1` (`plume.CONVECTIVE_H_OVER_L`,
`plume.STABLE_H_OVER_L`) is tabulated with neutral turbulence (`a = 2.5 u*/h`, no
Brunt-Vaisala frequency), `L` entering only the wind profile; `h/L > 1` takes the stable
law. On SIRANE probe runs along the wind the stable law was off in that band by 1-11 % at
the median (sigma_y 21, 8 and 1.5 % off at h/L = 0.25, 0.5 and 0.9); the neutral
tabulation is about 2.5e-4. Convective hours, `h/|L| > 0.3`, are refused by name: SIRANE
switches there to another vertical mechanism, not yet reproduced. That is an open
limitation: 43451 of the Paris case's 96936 street-hours over its week (45 %) are
convective.

### 4.5 `C_ext` and the step

The closure's order is emit, advance, housekeep, evaluate. The flux of step k is emitted at
the start of step k+1 at age 0 and advanced by dt before `C_ext` is evaluated, so its slug
then covers `[x(dt/2), x(3 dt/2)]`: the ages of the material that left during step k,
measured at the middle of step k+1, the interval over which that `C_ext` is used.

The tables and the field are cached and rebuilt when a meteorology driver changes in value
or is a different tensor; a value-only cache would reuse the graph of an earlier run and
give a wrong gradient under finite differences or repeated losses. The cache is also
dropped at the start of every run and keyed on the grad mode.

**Memory under autograd.** The whole puff step (emit, advance, housekeep, evaluate) is
checkpointed, and inside it `_evaluate` checkpoints its pair chunks. Autograd-saved bytes,
summed by `saved_tensors_hooks` over 40 steps at capacity 1024, B = 1
(`test_saved_bytes_grow_linearly_with_the_live_puffs`):

| checkpoint | per step | after 40 steps |
|---|---|---|
| on (default) | 22 KB, constant | 0.86 MB |
| off | 1.6 MB (step 0) to 29 MB (step 35), growing with the live puffs | 605 MB |

These are the bytes the hooks see. Not counted: each step's input `PuffState` held by
reference (about 12 (B, K) arrays, ~0.1 MB per step at K = 1024) and the field and tables
the checkpointed step captures. Checkpointing only the evaluation left ~200 (B, K) arrays a
step (1.6 MB at K = 1024, 6.4 MB at K = 4096), which is why the whole step is checkpointed;
the saved memory then scales with the capacity K, not with the live count.

The cost: forward plus backward 1.8-1.9x the uncheckpointed time (`twelve_street_grid`, 40
steps, 627 live puffs at the end, 4 threads):

| B | checkpoint | forward, no grad | forward, grad | backward | forward + backward |
|---|---|---|---|---|---|
| 1 | on | 36.9 ms/step | 72.7 | 113.5 | 186.2 |
| 3 | on | 62.1 | 116.8 | 162.0 | 278.8 |
| 1 | off | 34.4 | 59.6 | 41.7 | 101.3 |
| 3 | off | 64.9 | 89.9 | 55.1 | 145.0 |

Part of the backward's cost is the nesting: in a differentiated step the backward
recomputes the step, whose chunk checkpoints then recompute each pair block once more in
their own backward, so the pair-block work runs three times rather than two.

Gradients against central differences (Test 4, `drop_below = 0`): relative differences
4.2e-10 (canopy pulse rate), 1.5e-9 (pulse start), 8.2e-11 (release rate), 9.7e-8 (release
start, a small gradient from cancelling terms, at the difference quotient's rounding floor),
5.5e-10 (release duration), 6.9e-10 (u*), 2.3e-9 (wind direction), 9.8e-10 (release x).
With the old default `drop_below = 1e-6` a faint-drop decision flipped within the
finite-difference step of the release start (5e-3 off); the test turns dropping off, and a
pulse ending exactly on a step boundary puts the overlap's kink on the difference point,
which the test avoids.

### 4.6 Receptors and standalone mode

`concentration_at` evaluates the puffs at points of any height, with the vertical profile
of the interpolated `sigma_z` and `z_c` and without the roof-level cap; `PuffField.points`
with only explicit releases is the flat-ground model. At roof level `C_ext` reads the
table's `p_z`, linear between grid points, while a free receptor at `H_R` evaluates the
profile of the interpolated `sigma_z` and `z_c`: on the grid points the two agree, between
them they differ slightly.

### 4.7 Wind-shear stretching

`PuffOptions(shear=True)` (Sykes and Henn 1995, "Representation of velocity gradient
effects in a Gaussian puff model", *J. Appl. Meteor.* 34, 2715-2723): a vertical shear
`S = dU/dz` of the tables' wind profile `plume.wind_speed` (row by row,
`PuffTables.shear_rate`, the derivative in closed form, below), at
`z_s = max(z_c, 0.65 sigma_z)`, grows two per-puff moments:

    d(sigma_xz)/dt = S sigma_z^2,   d(sigma_x^2)/dt|shear = 2 S sigma_xz

(`"puff.sxz"`, `"puff.sx2_shear"`, zero at birth; they move with the puffs through emit,
eviction, housekeeping, batching and checkpointing like the other fields). `advance(...,
shear=True)` integrates them per step with `S` at the step's middle age `a + dt/2` held
over the step and `sigma_z^2` the quadratic in age through its values `q0`, `qm`, `q1` at
`a`, `a + dt/2`, `a + dt`:

    sxz       += S dt (q0 + 4 qm + q1) / 6
    sx2_shear += 2 S (sxz dt + S dt^2 (q0 + 2 qm) / 6)

exact for a constant `S` and a `sigma_z^2` quadratic in age (tested against
`sigma_xz = S a^2 t^3 / 3`, `sigma_x,shear^2 = S^2 a^2 t^4 / 6` for `sigma_z = a t` to
1e-10 at `dt = 0.1 s`). On real tables the rate varies with `z_s` and `sigma_z` is
piecewise linear on the 10 m grid, whose kinks mask a clean order in `dt`: a puff released
at 30 m in a stable hour has, at age 120 s, `sxz` within 1.8e-5 and `sx2_shear` within
1.1e-4 (relative) at `dt = 10 s` of their values at `dt = 0.25 s`, and 5.2e-6 / 1.9e-5 at
2 s. `S` is the closed-form derivative of the profile, `u* phi_m(z'/L) / (kappa z')` with
`z' = z - d + z0` and `phi_m` Businger's (`1 + 5 zeta` stable and neutral,
`(1 - 16 zeta)^(-1/4)` unstable), consistent with `plume._psi_m`; it needs no autograd, so
it runs under `torch.inference_mode`. Under `puff_dt` a puff is born at a positive age
(the middle of its interval less a step) with zero moments: it misses their growth over
that age, cubic (`sxz`) and quartic (`sx2_shear`) in it, small for intervals of tens of
seconds.

Evaluation. The puff is a correlated `(x, z)` Gaussian of along-wind variance
`sigma_x^2 = sigma_y(x(a))^2 + sx2_shear` and covariance `sxz`, convolved along the wind
with the slug. It is written as the vertical profile times the conditional along-wind
profile: each term of the vertical profile, centred at `z_k` (`z_c`, `2 H_R - z_c`,
`2 h - z_c`), multiplies the slug convolved with a Gaussian of the conditional variance
`sigma_x^2 - sxz^2 / sigma_z(x(a))^2` (the shear part floored at zero), shifted along the
wind by `beta (z - z_k)`, with `beta = sxz / sigma_z(x(a))^2` for the plume and `-beta` for
both images (a reflection in `z` flips the sign of `sxz`). `beta` and the conditional
variance are the puff's own, at its distance `x(a)`, as `sigma_x` is. The floor on the
conditional variance's shear part, `max(sx2_shear - sxz beta, 0)`, can be active: a table
whose `sigma_z` stops growing (or a meteorology rebuild that shrinks it, since the moments
are kept while `sigma_z` is read afresh) makes `sxz^2 / sigma_z^2` exceed `sx2_shear`.
Mass stays exact there (the along-wind profile still integrates to one); the gradient
through the floored part is zero. The vertical terms
are read at the receptor's distance, as without shear. For a slug of vanishing length and
constant spreads this is exactly the bivariate Gaussian with its images (tested to 1e-8).

Mass. Every term's along-wind factor integrates to one whatever its shift and width, so
the line integral of a sheared puff along the wind at any height is the unsheared one
(tested to 1e-9 at five heights and at roof level): the shear moves material along the
wind and neither makes nor loses any, and the budget identity of Sect. 4.8 holds
unchanged (tested every step over 60 steps with the wind turning).

Roof level. At `z = H_R` the plume and its canopy image have the same shift
`beta (H_R - z_c)`, the inversion image `beta (2h - H_R - z_c)`; the factor is the
table's `min(p_z, clamp)` times the weighted mean of the two along-wind factors, the
inversion image's weight its share of the profile at `H_R` (a logistic of the exponents'
difference, finite as `sigma_z -> 0`). With `sxz = 0` this is the unsheared factor.

The along-wind cut-off (exact cut and coarse mask) takes `sigma_x` with its shear part.
`shear=True` needs `along_wind="sigma_y"`: the `"none"` slug is a segment of the steady
plume with no along-wind Gaussian to stretch or correlate with `z`, and is refused by
name. With `shear=False` nothing of this runs: the values are bit-identical to the
unsheared model (the evaluator and one-puff-per-step regression fixtures pass unchanged).

### 4.8 Housekeeping and the budget

Fixed capacity K per batch instance, arrays (B, K) and a live mask, no Python lists of
puffs. Each step retires the puffs past `x_max` into `mass_out` and those whose peak
estimate is below `drop_below` times the instance's largest into `mass_dropped`; when new
puffs do not fit, the faintest live ones are evicted into `mass_dropped`. The peak estimate
reads the profiles at the puff's own distance, so it is a ranking, not a bound (a receptor
can see up to about twice it).

`drop_below`: on a pulse release over `twelve_street_grid` (above-roof release 3e-3 kg/s
over [12, 29] s, 80 steps of 5 s) 1e-3 against 1e-6 changes the most-hit street's
concentration by 1.26e-4 of its peak, its `C_ext` by 1.06e-4 and its dose by -5.5e-5; the
default is 1e-3.

The budget identity of the application page holds to 2.9e-15 relative over 60 steps in all
tested cases (both junction sources, with and without eviction, with the wind turning);
leaving out the not-yet-emitted mass gives 3.6e-2, so the identity discriminates.

**Emission interval `puff_dt`.** With a number, each source's mass is accumulated over the
interval and emitted as one puff whose slug spans it, aged from the interval's middle.
`puff_dt = None` (the default) is the one-puff-per-step model bit for bit, and so is
`puff_dt` equal to the step. At 30 s over 5 s steps (along_wind "none"):

- kernel level, right after an emission the train is the kernel (7.5e-16); over the next
  five steps the pending material leaves a growing lag gap: 3.8e-16, 3.8e-16, 7.1e-2,
  0.81, 0.995 of max K;
- coupled steady run over each interval's six steps: `C_ext` error 8.4e-3, 8.4e-3, 8.4e-3,
  2.7e-2, 0.59, 0.74 of max (C_ext - C_bg), repeating; street concentrations 2.1-2.9e-2;
- the pulse above: puff_dt 30 against None changes the most-hit street's concentration
  peak by 0.376, its `C_ext` by 0.570 and its dose by 2.5e-2.

It buys no time (Sect. 4.10): the pair work follows the area the cloud covers, and a 30 s
slug covers six times the (receptor, sub-source) pairs of a 5 s one. It saves capacity
only, so the default is `None` and a number is an opt-in.

### 4.9 Batching and differentiability

A leading batch over releases and meteorology on every per-puff array and on the tables; a
batch of three releases (different streets, start times, u*) equals three runs to
rounding. Masks, drop decisions and evictions are taken on detached values; second-order
differentiation is refused by name. Under `meteo="per_street"` the tables take the first
street's entry of each (uniform) meteorology driver, so its gradient reaches that entry
only.

### 4.10 Cost at Paris scale

The design estimated 50-100 ms per step per instance. Case: the Le Perreux-sur-Marne MUNICH
case (577 streets, `benchmarks/street_puffs.py` with `NOODL_PARIS_CASE`), hour 2 held
constant, `sigma_theta` 6 degrees, a pulse of 1 kg over 30 s on the street nearest the
upwind edge, dt = 5 s, one hour (720 steps), `torch.no_grad()` for forward-only runs.

| build | puff_dt | capacity | run | ms/step | peak live | evicted |
|---|---|---|---|---|---|---|
| first evaluator | None | 4096 | forward B=1 | 1507 | 3904 | 374730 |
| first evaluator | 30 s | 4096 | forward B=1 | 1851 | 1834 | 0 |
| first evaluator | 30 s | 4096 | forward B=8 | 4411 | 1963 | 0 |
| first evaluator | 30 s | 4096 | forward + backward B=1, 120 steps | 5239 | 300 | 0 |
| first evaluator | None | 16384 | forward B=1 | 3900 | 6350 | 0 |
| current | None | 16384 | forward B=1 | 239 (median of 232-246) | 6350 | 0 |
| current | None | 8192 | forward B=1 | 228 | 6350 | 0 |
| current | None | 16384 | forward B=8 | 780 | 6738 | 0 |
| current | None | 16384 | forward + backward B=1, 120 steps | 342 (312-432) | 1134 | 0 |

At capacity 4096 and one puff per step the store saturates and evicts (that hour is not a
valid result); 8192 evicts nothing at batch 1 or 8, and is the benchmark's default.

The current evaluator gives the first one's values bit for bit (`torch.equal` on the
recorded fixture of `test_evaluator_matches_the_recorded_reference` and on the Paris step
360); its gradients differ by ~1e-16 relative, from the backward's accumulation order. What
made it 16x faster: the active puffs compacted once instead of a dense (B, K, receptor)
mask; a coarse mask in each instance's wind frame; receptors sorted along the wind once so
that each puff's along-wind band is a `searchsorted` run; pair chunks sized by the
sub-source pairs they expand to; table reads split so each mode reads only the columns it
needs.

The forward + backward rows ran over the first 120 steps only (peak 1134 live puffs),
not the hour; the hour's forward + backward was not measured.

Where a step's time goes (B = 1, steps 360-390, 4127 live, capacity 16384, profiled on the
code just before the last evaluator change, so the parts sum to more than the 239 ms
median of the final code; `_evaluate` alone on the captured step-360 state then took
40-90 ms on a loaded machine): the street
model outside the puffs about 187 ms, almost all the canyon wind's shape-parameter solve
(`routing.velocities` -> `canyon.roof_wind` -> `bessel_shape_parameter`), recomputed every
step under constant meteorology; the puff closure 163 ms, of which `_evaluate` 115 ms,
`advance` 18 ms and `housekeep` 23 ms (both over all K slots, so they grow with the
capacity) and `emit` 4 ms. Pair counts at step 360 (5167 live, 577 receptors): 341 k coarse
candidates of 2.98 M (puff, receptor), 71 k coarse pairs, 356 k sub-source pairs, 203 k
non-zero contributions.

## 5. Reproduce

- Steady limits, budget, gradients, batch, `puff_dt`, evaluator fixture:
  `pytest tests/apps/street_aq/test_puff.py tests/apps/street_aq/test_puff_coupled.py`
  (`-m slow` for the coupled steady runs; `-s` prints the `puff_dt` and `drop_below`
  numbers).
- Paris cost: `python benchmarks/street_puffs.py --batch 1 8 --steps 720` with
  `NOODL_PARIS_CASE` pointing at the case directory (`--profile` for the breakdown).
- Stability classes: `tests/apps/street_aq/test_plume.py` and the probe cases in
  `tests/verification/test_sirane.py`.
