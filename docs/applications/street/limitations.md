# Street air quality: limitations

What the street air quality application supports and what it does not. The overview has the short list; this page gives the reasons and the sizes. Evidence is on the [verification page](../../reference/verification/street.md).

## Agreement with the reference tools

- **The MUNICH 12-street idealised case is not reproduced.** Its street geometry was never published, so absolute concentrations cannot be compared, and the pattern of concentrations relative to one street misses its 5 % target: the worst of the nineteen ratios is off by 51 %, and seven are within 15 %. The scale-invariant checks (linearity in wind speed, the 270° canyon-wind ratio) do match.
- **Street concentrations are not compared with SIRANE.** The archived South Kensington run's emission field is not the deck's, so no concentration comparison against it is meaningful; the residual under the deck's stated emission (median relative difference 0.888) is recorded, not checked.
- **SIRANE's exchange coefficient.** The code uses $\sigma_w / (\sqrt{2}\,\pi)$, as in MUNICH's source and SIRANE's, not $\sigma_w/\sqrt{2\pi}$.
- **MUNICH's chemistry** (the JPL rate, no floor) is checked through its rate constant, not against MUNICH output.

## Physics not modelled

- **Convective hours.** The above-roof plume and the puffs refuse hours with $h/|L| > 0.3$ by name. SIRANE switches there to another vertical mechanism, close to Eq. (29)'s unstable $\sigma_z$ with a Lagrangian time scale that is not the paper's Eq. (18) as printed, and it is not reproduced.
- **Plume rise, wet-deposition depletion and SIRANE's cell-based retrotrajectory path** (`B_RUE_DECOUP = 1`) are not implemented. The plume's downwind cut-off follows SIRANE's rule for meteorology cells of about 700 m or larger; smaller cells switch SIRANE to a cell-based far field the kernel does not model.
- **Chemistry** is the photostationary NO-NO₂-O₃ equilibrium. Solar geometry is not computed for MUNICH's photolysis form: you pass the zenith angle. Puffs carry no chemistry, deposition or concentration fluctuations.
- **Meteorological preprocessing.** noodl has no SIRANE preprocessor: a SIRANE case is driven with its own $u_*$, boundary-layer height, Obukhov length and direction spread, taken from its results. The dispersion site's roughness and displacement height are recorded but not used; the canopy takes $d = 2 h_{\text{mean}}/3$ and $z_0 = h_{\text{mean}}/10$ from the network.
- **Tall streets.** A street taller than $1.25\,h_{\text{abl}}$ needs the boundary-layer guard (`pblh_floor=True`, the default); without it `exchange_velocity` raises an error rather than clamping. See the [physics page](physics.md#canyon-wind-and-roof-exchange).

## Fitted elements

Four elements of the above-roof kernel are fitted to SIRANE output rather than derived (see [Above-roof concentration](above-roof.md#empirical-inputs)): they were fitted on neutral and stable hours across a range of street orientation, width, height and length. The kernel is discontinuous in the Obukhov length at the edges of SIRANE's neutral class, as SIRANE's is, and several elements are piecewise linear; the gradient is that of the class or segment the hour sits in.

## Gradients

Street directions are geometry constants and are not differentiable in the coordinates. A street's width and height carry no gradient through the kernel. Puff masks, drop decisions and evictions are taken on detached values, and second derivatives through the puffs are refused by name.

## Case files

`read_case` refuses, by name, anything beyond what the supported SIRANE decks exercise: a network file that is not a PolyLine Shapefile, meteorology that is not one measured station, user-supplied street velocities, NO emitted as NO₂-equivalent, a per-hour meteorology override file, a non-empty source-groups or surface-emission file, and a point source with non-zero emission. MUNICH's `Wang` transfer parameterisation is refused. See [Reading and writing cases](cases.md).

## Puffs

- One well-mixed box per street: SIRANERISK follows the material along a street in moving
  boxes, here a street releases each puff from all its sub-sources at once.
- SIRANE's steady plume laws, not SIRANERISK's puff dispersion laws: a puff's spreads are
  the steady plume's at its age. When the meteorology changes, live puffs read the new
  table at their age; their spreads are not integrated through the change.
- No concentration fluctuations, deposition, chemistry or plume rise in the puffs.
  Wind-shear stretching (`shear=True`) is available under `along_wind="sigma_y"` only.
- Convective hours ($h/|L| > 0.3$) are refused by name, as in the steady kernel (see
  [Above-roof concentration](above-roof.md#differentiability)).
- At roof level the table's $p_z$ is interpolated between grid points; at a free receptor
  at $H_R$ the vertical profile of the interpolated $\sigma_z$ and $z_c$ is used. The two
  differ slightly between grid points.
- The background is one value per street per step, and the mass budget is exact only for a
  background constant in time.
