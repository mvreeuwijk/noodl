# Above-roof concentration

How to replace the background above the roofs with a modelled concentration, in the steady case. The unsteady version is on [Unsteady above-roof puffs](puffs.md); the plume mechanism is on the [physics page](physics.md#the-above-roof-plume-kernel).

By default the atmosphere above the canyon is whatever the `"<layer>.x_boundary"` driver says: an input, not a modelled quantity. Optionally, as in SIRANE, `C_ext`, the concentration just above each street's roof, is instead computed as the sum of a fixed background and the superposed plumes of every upwind street's roof flux and every upwind junction's vertical flux:

```
C_ext = C_bg + K_s F_s + K_j F_j
```

`F_s` is each street's signed roof flux, $u_d W L (C - C_{	ext{ext}})$, read off the same `exchange` edges the transport layer already carries, so it conserves mass exactly with the street solve. `F_j` is each junction's vertical flux: the excess over background carried up out of the canopy through its outgoing `vent` edges (`junction_source="upward"`, the default), or that minus what comes back down through the incoming vents (`junction_source="net"`, for comparison). `K_s` and `K_j` are kernels, s/m³, built once per hour from the network geometry and meteorology by `street_kernel` and `junction_kernel` (`noodl.apps.street_aq.plume`).

## Solve the coupled problem

`street_steady_with_plume` (`noodl.apps.street_aq.above_roof`) solves the street network and this relation together: starting from `C_ext = C_bg`, it solves the network, recomputes `C_ext` from the resulting fluxes, and repeats until the largest change (relative to the largest `|C_ext|`) falls below `tol`, typically 6 to 13 passes on a network the size of Paris (577 streets). It needs a model built with `background="per_street"`, and takes the two kernels as required keyword arguments: there is no `build_model` option for this, since the kernels depend on the wind and must be rebuilt every hour. The gradient is that of the converged fixed point, by an implicit adjoint (one GMRES solve per instance, hours never coupled). The iteration is `Model.fixed_point`, judged per hour.

| Option | Values | What changes |
|---|---|---|
| `junction_source` | `"upward"` / `"net"` | The junction plume source: the excess flux carried up out of the canopy, or that minus the excess carried back down. `"upward"` is the default; on SIRANE's archived output the two agree equally well. |
| `self_contribution` (`street_kernel`) | `False` / `True` | Whether a street's own roof-flux points contribute to its own `C_ext`. Excluded by default: including them overstates `C_ext` (measured median ratio 2.33 against SIRANE's own fluxes). |
| `meteo_cell_dx` (`street_kernel`, `junction_kernel`) | the x-size of SIRANE's meteorology grid cell, m | The downwind cut-off, `meteo_cell_dx / \|cos theta_w\|`; `None` (the default) uses the table's own extent, 700 m. |

## Empirical inputs

Four elements of the mechanism are fitted to SIRANE's output rather than derived from the
paper, each a named parameter of `plume_table` with a default:

| Parameter | Default | What it is |
|---|---|---|
| `tau_y`, `tau_z` | `TAU_Y_BY_SIGMA_V`, `TAU_Z_BY_SIGMA_W`: piecewise-linear tables of the hour's floored `sigma_v`, `sigma_w` | The time offsets behind which `sigma_y` (Eq. 28) and `sigma_z` (Eq. 29) are evaluated. |
| `centre_c`, `centre_k` | 10 m, 0.675 | The plume-centre law, `z_c = max(H_R, d + E\|N(c - d, (k sigma_z)^2)\|)`. |
| `theta_star` | `u*^2 T / (kappa g L)` | The temperature scale in the Brunt-Vaisala frequency. On the one stable hour with a real SIRANE comparison (South Kensington), SIRANE's own preprocessor prints 0.072 K, but 0.060 K reproduces its output far better (see [Verification](../../reference/verification/street.md)). |
| `meteo_cell_dx` | none (falls back to the table's `x_max`, 700 m) | The x-size of SIRANE's meteorology grid cell, for the downwind cut-off. |

All four were fitted on the SIRANE reference data described above, across neutral and
stable meteorology and a range of street orientation, width, height and length.

## Differentiability

The kernel is linear in the fluxes, and differentiable in `u*`, the boundary-layer height,
the Obukhov length, the wind direction, `sigma_theta`, the roughness `z0`, the displacement
height `d`, the reflection height `H_R`, the temperature and the four empirical parameters
above; a street's width and height (which set its sub-sources' widths and caps) carry no
gradient. Several elements are piecewise constant or linear -- exact to SIRANE's own
mechanism, not a smoothing choice: the trajectory's speed steps through integer heights,
the 10 m table and its linear reads have kinks every 10 m, the default `tau_y`/`tau_z`
tables are piecewise linear with a floor, and the crosswind and downwind cut-offs are hard
(a pair that crosses one jumps to exactly zero, and its gradient is that of whichever side
it sits on). The kernel is also discontinuous in the Obukhov length at the edges of
SIRANE's neutral class next, as SIRANE's is: at `h / L = 1` it jumps between the neutral
and the stable law, and at `h / L = -0.3` it ends (the convective side is refused). The
gradient in `L` is that of the class the hour sits in; a gradient-based search over `L`
does not see the jump.

SIRANE classifies an hour by `h / L`, as the ADMS scheme it cites: an hour in the neutral
class `-0.3 <= h / L <= 1` (`plume.CONVECTIVE_H_OVER_L`, `plume.STABLE_H_OVER_L`) is
tabulated as a neutral one -- neutral turbulence, `a = 2.5 u*/h`, no Brunt-Vaisala term --
with only the wind profile seeing `L`; the stable law applies for `h / L > 1`. SIRANE probe
cases on both sides agree with this at the neutral cases' level: on the unstable side
L = -1500 to -30000 m across and oblique to the wind, and h / |L| = 0.29 along it; on
the stable side h / L = 0.25, 0.5 and 0.9, along the wind only, where the stable law was
1-11 % off at the median, and 1.1, which follows the stable law. The Paris case's first
hour (541 unstable streets, all with `h / |L| <= 0.15`) falls in this class. The puff
tables (`puff_tables`) share this classification.

Not implemented: the convective regime (`h / |L| > 0.3`, refused by name -- SIRANE switches
there to another vertical mechanism, close to Eq. (29)'s unstable `sigma_z` with a
Lagrangian time scale that is not the paper's Eq. (18) as printed, which is not yet
reproduced), plume rise, wet-deposition depletion, and SIRANE's cell-based retrotrajectory
path (`B_RUE_DECOUP = 1`).
