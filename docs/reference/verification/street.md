# Verification: street air quality

Evidence for the [street air quality application](../../applications/street/index.md); the summary is on the [verification overview](index.md). *Verification* here is code-to-code comparison: it says nothing about agreement with measurements.

## Status

Two reference tools are compared, and they stand differently. Every comparison below is a test that passes at the tolerance stated (the longest coupled runs under the `slow` marker).

- **SIRANE v2.1 (rev 128).** Roof exchange velocity, in-canyon wind, the above-roof plume kernel and the NO-NO₂-O₃ chemistry (`preset="sirane"`) are reproduced against its archived output, to its printed precision or to the stated regression guards. The street-concentration comparison is not meaningful, because the archived run's emission field is not the deck's (see below).
- **MUNICH v2.2.** Its own runs of the 12-street network, written by `write_case` and recorded with their output, are reproduced street by street and hour by hour to the float32 precision of that output, under three closure sets that between them cover every MUNICH option `model_options` translates. Thirteen formula pairs from its source are exact to the precision it publishes. The published figure of the idealised 12-street case (Kim et al. 2022, Fig. 1), whose inputs were never published, is compared only in scale-invariant terms; its relative pattern is not reproduced (worst of 19 ratios off by 51 %, against a 5 % target). MUNICH's chemistry (the JPL rate, no floor) is verified through its rate constant only.

The unsteady puffs are verified against the steady kernel they must reduce to, not against SIRANERISK.

## MUNICH formulas
Thirteen exact input/output pairs transcribed from MUNICH's own source, each checked at the
precision that source publishes:

| Quantity | Tolerance | Measured |
|---|---|---|
| `EXCHANGE_SIGMA_W_RATIO` constant | < 1e-15 | 0.225079079039277, exact |
| Exchange velocity $u_d$ | < 1e-7 relative | 0.08681174 m/s to 8 significant figures |
| `bessel_shape_parameter` root $c$ | < 1e-13 | 0.6198293039179747 |
| Macdonald $d_c$, $z_{0c}$ | < 1e-12 | 4.617352498423888 m, 0.6614635677623194 m |
| Direction-quadrature weight sums, $n = 2\ldots10$ | 1e-6 | e.g. 0.974953 at $n=10$ |

That last row is deliberate: MUNICH's weights are **not** normalised to 1, and reproducing that
artefact is the point. "Fixing" it would introduce a bias relative to MUNICH rather than remove
one. The same applies to the shape parameter: MUNICH quantises the root to a 0.01 grid (0.62),
where `bessel_shape_parameter` solves it continuously (0.6198293…). The quantisation is small
in $u_M$ (4e-4) but not in $u_H$, whose canyon-mean factor is sensitive to $c$ (up to about
2 %), so `preset="munich"` takes MUNICH's grid search (`shape_constant="grid_search"`), and
`preset="sirane"` keeps the exact root.

## MUNICH runs

MUNICH v2.2 (`With_stationary_hypothesis: yes`, chemistry and deposition off) run on
`twelve_street_grid(L=100, W=20, H=20)`, inputs written by `write_case(format="munich")`
and kept with MUNICH's output in `tests/data/street/munich_twelve_street/`
(`tests/verification/test_munich_run.py`). Eight hours: winds from 0°, 45°, 90°, 135°, 200°,
225°, 270° and 315° at 2-6 m/s, neutral, stable and unstable (L from -50 m to 400 m), three
emitting streets (one changing hour by hour) and a uniform background. noodl physics reads
the case back (`read_case`, `model_options()`) and solves each hour to steady state. Measured
error: max |noodl - MUNICH| / MUNICH over 12 streets and 8 hours.

| Closure set (`[street]`) | Tolerance | Measured |
|---|---|---|
| Exponential canyon wind, Schulte exchange, Bessel roof wind, horizontal fluctuation (`write_case`'s defaults) | 2e-6 | 7.0e-7 |
| Bessel canyon wind, SIRANE exchange, Macdonald roof wind, no fluctuation | 2e-6 | 5.4e-8 |
| Bessel canyon and roof wind, SIRANE exchange | 2e-6 | 5.3e-8 |

MUNICH saves float32 (about 6e-8 relative) and ends its steady sweep once no street changes
by more than 0.01 µg/m³ (`StreetNetworkTransport.cxx:2515`); the 7.0e-7 is one hour (wind
from due north) where that stopping rule leaves the most. The inputs are also checked to be
byte for byte what `write_case` writes, so the writer is under test as well.

Three things the runs pinned. MUNICH's Bessel canyon wind is written on its single surface
roughness `z0_surface` (0.01 m by default), not on a building roughness, so `read_case` gives
every street that `z0_b`. On the Macdonald roof wind, MUNICH's Bessel street wind starts from
`u_H / f_mean(c)` rather than from u*. And MUNICH uses κ = 0.41 under every closure, so
`model_options()` sets it rather than inferring it from the closures chosen. Each moved a
closure set from 20-37 % off to the precision above.

## MUNICH idealised 12-street case
Kim et al. (2022) Fig. 1 gives concentrations on an idealised 12-street network, but not the
street length, width and height behind them, so `twelve_street_grid(L, W, H)` takes them as
arguments and the case can only be compared in terms that do not depend on them.

| Check | Target | Measured |
|---|---|---|
| Linearity in wind speed at 210° and 240°: doubling $U$ halves every concentration | < 1e-9 | holds (an exact model identity) |
| 270° canyon-wind ratio pattern | paper's value | 1.99451, against the paper's 1.99451 |
| Concentrations relative to one reference street, 19 ratios | 5 % | **not met**: worst ratio off by 51 %, 7 of 19 within 15 % |

The relative-pattern miss is the one open discrepancy against MUNICH; it is listed under
[Limitations](../../applications/street/limitations.md).

## SIRANE
A code-to-code comparison against SIRANE v2.1 rev 128 on its own South Kensington network
(46 streets, 36 junctions), using part of its archived results -- hours 00 and 01 of
7 January 2014 (`tests/verification/test_sirane.py`; the fixture's own caveats are recorded with it). That archived run's own meteorology
differs from the deck's meteo file (a wind from 315°, where the deck's file gives 135°, and
every street's `Sigma_wH` at SIRANE's default 0.30 m/s floor, where the deck's file turns
that floor off), so every comparison here drives noodl physics with the *results'* own
meteorology (`read_results(...).meteo`), never the deck's.

- **Roof exchange velocity.** `exchange_velocity(roof_exchange="turbulent_velocity")`, evaluated at SIRANE's own
  printed `Sigma_wH` with each street's own height and width, reproduces SIRANE's printed
  `u_d` on all 46 streets across both hours to within SIRANE's own printed half-step
  (0.005 m/s) -- the tightest bound two-decimal printed output allows; the worst street
  misses by 0.0025 m/s. The alternative reading `sigma_w / sqrt(2 pi)` misses by far more
  (about ten printed half-steps on every street), which is what confirms SIRANE's own
  exchange constant, $\sigma_w / (\sqrt{2}\,\pi)$, rather than $\sigma_w/\sqrt{2\pi}$.

- **In-canyon wind.** The Soulhac canyon velocity, driven per street with SIRANE's own
  friction velocity (0.14 m/s) and direction (315°), agrees with SIRANE's printed `U_moy` on
  all 46 streets to within its own printing precision: median relative difference 4.5 %, 90th
  percentile 12.8 %, and every street inside the rounding envelope that SIRANE's own two
  printed quantities allow (`printed half-step + 3.6 % |noodl physics' value|`, since u*
  itself is printed to +-3.6 %). The one-sided streets (buildings on one side only, height
  `mean(HG, HD)`) are no worse than the rest.

- **Source-receptor concentration.** With SIRANE's own above-roof concentration imposed as
  the per-street background, and the deck's unit emission (1 g/s of O3 on one street), the
  archived run's own mass balance -- SIRANE's roof export plus its dry-deposition flux,
  computed entirely from SIRANE's own printed output -- comes to more than ten times the
  deck's stated emission, and the street with the highest in-canyon concentration is not the
  emitting one. That is inconsistent with a 1 g/s source, so this fixture's archived
  concentrations come from a different emission field than the deck's, and a concentration
  comparison against them is not meaningful at this emission. noodl physics' own residual
  under the deck's stated emission (median relative difference 0.888) is recorded rather than
  checked against a tolerance. A meaningful concentration comparison needs a result whose
  emission field is known, e.g. from `write_sweep`'s own decks.

- **Above-roof plume kernel.** Fed SIRANE's own archived roof and junction fluxes directly
  (not noodl physics' own street solve), `street_kernel` and `junction_kernel` reproduce
  SIRANE's printed `Cext` on 45 of the network's 46 streets (the 46th sits at the upwind
  corner, with `Cext` exactly zero) across both hours, under the library's defaults: median
  relative difference 0.3 %, 90th percentile 1.4 %. This uses two inputs the archive does
  not print exactly -- `u*` = 0.138 m/s, read off the near-field plateau of a matching
  kernel-probe case below rather than SIRANE's own two-decimal 0.14 m/s, and `theta*` =
  0.060 K, fitted to the probe cases of the same meteorology rather than SIRANE's own printed
  0.072 K; using the printed values instead gives visibly larger errors (the 90th percentile
  roughly doubles with printed `theta*`, and the largest error roughly doubles with printed
  `u*`, 0.055 against 0.028). The check is out of sample in geometry (mixed street lengths, widths, heights and
  angles, 36 junctions), though the default time offsets and `theta*` were fitted partly on
  this hour's meteorology.

  Nineteen further runs of one isolated street under imposed meteorology check the kernel directly against SIRANE's own
  concentration grids. Ten, across, along and oblique to the wind in neutral and stable
  hours, are in sample -- the empirical parameters above are fitted on them. Nine test
  SIRANE's stability classes and were not fitted on: four weakly unstable (L = -1500 to
  -30000 m, across and oblique), one at h / |L| = 0.29 and four along the wind at
  h / L = 0.25, 0.5, 0.9 and 1.1. Over the sampled cells the per-run median relative
  difference is 0.016-0.6 % and the 90th percentile 0.1-2.9 %
  (`PROBE_TOLERANCES`). Including a street's own roof-flux
  points in its own `Cext` (`self_contribution=True`) overstates it (measured median ratio
  2.33 on the South Kensington fluxes), which is why it is excluded by default.

## SIRANE chemistry
A code-to-code comparison of `preset="sirane"` against SIRANE v2.1 output with
  its NO-NO2-O3 chemistry switched on (`tests/verification/test_sirane_chemistry.py`), by case:
  - `solar_elevation` against 768 printed hourly values (eleven meteorology sets, every season,
    clear to overcast, day, night and the low-sun threshold): worst absolute difference
    0.000497°, inside SIRANE's own three-decimal printing (±0.0005°).
  - `j_no2_elevation_cloud` ($k_1$) from that elevation and the printed cloud cover, against
    the same 768 printed $k_1$ values to three significant figures: 0 mismatches.
  - `k_no_o3_soulhac2011` ($k_3$), at SIRANE's printed ground temperature and molar volume,
    against the same 768 printed $k_3$ values (ppb⁻¹ s⁻¹) to three significant figures: 768/768 consistent,
    over the rounding envelope of the printed inputs. The negative control — the paper's 1.325e5
    prefactor — matches none of the 768 rows.
  - The photostationary split itself, on 176 street and receptor values from six single-street
    cases (day, day without background, night, dawn, stable, NO emitted as NO2-equivalent mass):
    176/176 reproduced within SIRANE's own printed precision (±0.005 µg/m³), allowing exactly the
    rounding its inputs carry — the passive NO and NO2 (±0.005 µg/m³ each), the ground temperature
    ($T_g$, ±0.05 K) and the molar volume ($V_m$, ±0.005 L/mol).

## Unsteady puffs

The checks are in `tests/apps/street_aq/test_puff.py` and `test_puff_coupled.py`. On `twelve_street_grid`, the steady puff train of a unit continuous source at each of the 24
sources (12 streets, 12 junctions; ages dt, 2 dt, ... out to x_max, mass dt) evaluated at
the street midpoints, against `[street_kernel | junction_kernel]` from `plume_table` with
the same inputs. Error: max |K_puff - K| / max K.

**`along_wind="none"`: the kernel itself.** At every step tried (10, 8, 6, 5, 4, 3, 2.5, 2,
1.25, 1, 0.625, 0.3125, 0.15625, 0.078125 s) the error is 1.9e-16 to 9.4e-16. The checks assert 1e-12 at 10, 5, 2.5,
1.25 and 0.625 s, and the same bound in a weakly unstable (h/|L| = 0.165; measured 4.8e-16)
and a weakly stable (h/L = 0.5) hour.
The coupled run against `street_steady_with_plume` (run until `C_ext` changes by less than
1e-12 relative per step) reaches 7.9e-12 at 10 s and 1.6e-11 at 5 s, the stopping rule's
level.

What these checks cannot see: the one-step emission lag. A continuous source's newest
material sits at ages `[dt/2, 3dt/2]` when `C_ext` is evaluated, so the puff train leaves a
gap `[0, x(dt/2))` downwind of every source. No receptor-sub-source pair of this grid lies
in that gap at any of these steps, so the lag's first-order term is invisible here; a
receptor that close to a source (closer than half a step's travel, about 6.5 m at 5 s here) would
see it.

**`along_wind="sigma_y"`: a floor.** The design expected convergence to the kernel smoothed
along the wind with the puff's own profile. As built (see the [design note](../../project/design/street_puffs.md)) the error against the
plain kernel converges at first order in dt to a floor: 0.117 (10 s), 0.0699 (5 s), 0.0577
(2.5 s), 0.0457 (1.25 s), 0.0468 (0.625 s), toward about 0.047. The floor is the along-wind
diffusion the steady Gaussian plume neglects (the slender-plume approximation), a property
of the steady kernel rather than an error of the puffs. The smoothed-kernel reference was
not built. The checks assert the floor is below 0.10 for dt <= 5 s and changes by less than 10 % between 1.25 s
and 0.625 s.
