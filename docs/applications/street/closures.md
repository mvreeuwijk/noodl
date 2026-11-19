# Choosing the closures

`build_model` takes one keyword per modelling choice. A preset (`"sirane"`, the default, or `"munich"`) fixes them all, and any keyword left at `None` takes the preset's value, so the choices can be mixed. This page lists every option and what it computes. How to build and drive a network is on [Build and run a street network](networks.md); the equations are on the [physics page](physics.md).

```python
from noodl.apps.street_aq import build_model, from_test_network

net = from_test_network()
model, state, drivers = build_model(
    net,
    preset="sirane",              # 'sirane' | 'munich': the full closure set
    canyon_wind=None,             # 'bessel_profile' | 'exponential_profile'
    roof_wind=None,               # 'bessel_canyon_mean' | 'canopy_log_law'
    roof_exchange=None,           # 'turbulent_velocity' | 'aspect_ratio_scaled'
    junction_routing=None,        # 'non_crossing_streamlines' | 'perfect_mixing'
    direction_averaging=None,     # 'none' | 'exact_gaussian' | 'rectangle_rule' | 'gauss_hermite'
    direction_spread=None,        # 'driver' | 'turbulence_intensity'
    stability=None,               # 'neutral' | 'monin_obukhov'
    sigma_w_height=None,          # 'canopy_height' | 'street_height'
    h_canopy=None,                # m: the canopy height (SIRANE's H_R)
    shape_constant=None,          # 'exact_root' | 'grid_search'
    species=("nox",),
    chemistry=None,
    kappa=None, canyon_wind_min=None, u_d_min=None, sigma_w_min=None, sigma_v_min=None,
    z_ref=30.0, pblh_floor=True,
    meteo="uniform",              # 'uniform' | 'per_street'
    background="uniform",         # 'uniform' | 'per_street'
    puffs=None,                   # PuffOptions(...): unsteady above-roof puffs
)
```

Every closure keyword left at `None` takes the preset's value; any other value overrides it, so `build_model(net, preset="munich", roof_wind="canopy_log_law")` is MUNICH's closure set with Macdonald's roof wind.

**The defaults are SIRANE's.** A model built with no options floors $\sigma_w$ at 0.30 m/s before the roof exchange velocity (`sigma_w_min=0.30`), which binds in light winds; pass `sigma_w_min=0.0` for no floor. It evaluates that $\sigma_w$ at one height for the whole network, the canopy height `h_canopy` (20 m, SIRANE's `H_R`), as SIRANE does; `sigma_w_height="street_height"` uses each street's own height instead. The default street chemistry, `photostationary_for_streets(species)`, is the Soulhac et al. (2011) rate with a 2 ppb floor on $J/k$. `preset="munich"` gives MUNICH's settings throughout: no turbulence floors, and for the chemistry the JPL (2003) rate with no floor.

## The options

Every option value is named for the physics it computes. The two reference models' full sets are the presets (`noodl.apps.street_aq.PRESETS`); the options are independent, so any mix is allowed.

| Option | Value | Physics | `sirane` | `munich` |
|---|---|---|---|---|
| `canyon_wind` | `"bessel_profile"` | Soulhac–Perkins–Salizzoni (2008) closed-form Bessel profile from the friction velocity. Needs `z0_b`. | ✓ | |
| | `"exponential_profile"` | K22 Eq. (B14) exponential profile from the roof-level wind, $u = u_H \cos\varphi\,(2/a_r)[1 - e^{(a_r/2)(z_{0s}/H - 1)}]$. **Not** the K18 formula of the same name — they diverge by 37% for narrow canyons. | | ✓ |
| `roof_wind` | `"bessel_canyon_mean"` | $u_H$ as the canyon mean of the Bessel profile (K22 Eq. B12). Used by `"exponential_profile"` only. | ✓ | ✓ |
| | `"canopy_log_law"` | $u_H = (u_*/\kappa)\ln((H - d_c)/z_{0c})$ with Macdonald's (1998) network-mean displacement height and roughness (K22 Eq. B13). | | |
| `roof_exchange` | `"turbulent_velocity"` | $u_d = \sigma_w / (\sqrt{2}\,\pi)$, aspect-ratio independent. | ✓ | |
| | `"aspect_ratio_scaled"` | Schulte's $u_d = \beta\,\sigma_w / (1 + H/W)$, $\beta = 2/(\sqrt{2}\,\pi)$. Equals `"turbulent_velocity"` exactly at $H = W$. | | ✓ |
| `junction_routing` | `"non_crossing_streamlines"` | Inflows fill the outflows in angular order without crossing (a closed-form north-west-corner fill). | ✓ | ✓ |
| | `"perfect_mixing"` | Every inflow shared in proportion to the outflows. The two differ only at junctions with 2+ inflows **and** 2+ outflows. | | |
| `direction_averaging` | `"none"` | The junction routing at the mean wind direction alone. | | |
| | `"exact_gaussian"` | The exact Gaussian average of the junction routing over the direction spread (Soulhac et al. 2011, Eq. 7): the routing changes only where a street switches between inflow and outflow, so the average is a sum over those intervals weighted by their Gaussian mass. Differentiable in $\sigma_\theta$; needs $0 \le \sigma_\theta < \pi/4$. | ✓ | |
| | `"rectangle_rule"` | Uniform samples on $[-2\sigma_\theta, 2\sigma_\theta]$, $n = \lfloor\sigma_\theta\text{ in degrees}\rfloor$ (at most 10), with weights $\Delta\,N(\theta_k; 0, \sigma_\theta)$ that are **not** normalised (they sum to 0.975 at $n = 10$, 0.432 at $n = 2$). | | ✓ |
| | `"gauss_hermite"` | An `n_theta`-point Gauss–Hermite quadrature with normalised weights. | | |
| `direction_spread` | `"driver"` | $\sigma_\theta$ from the `sigma_theta` driver, else the `sigma_theta` keyword, else 0 (a single direction). | ✓ | |
| | `"turbulence_intensity"` | $\sigma_\theta = \min(\sigma_v / U_{\text{ref}},\ 10°)$ at each junction; a `sigma_theta` driver is refused. | | ✓ |
| `stability` | `"neutral"` | $\sigma_w = 1.3\,u_*(1 - 0.8\,z/h_{\text{abl}})$, $\sigma_v = 1.2\,u_*$. | | |
| | `"monin_obukhov"` | Stable, neutral and unstable branches selected by the `lmo` driver; the neutral form wherever `lmo` is not given. | ✓ | ✓ |
| `sigma_w_height` | `"canopy_height"` | The exchange's $\sigma_w$ is evaluated at $z = h_{\text{canopy}}$ above ground, the same for every street: SIRANE's $u_d$ "is only defined by the external flow condition" (Soulhac et al. 2011, Sec. 4.2.2). | ✓ | |
| | `"street_height"` | The exchange's $\sigma_w$ at each street's own height $H$ (MUNICH's `ComputeSigmaW`). | | ✓ |
| `h_canopy` | m | The canopy height of `"canopy_height"`, SIRANE's deck keyword `H_R` ("Canopy height", "Hauteur de reflexion des bouffees"); `StreetCase.model_options()` reads it from a SIRANE deck. | 20 | (unused) |
| `shape_constant` | `"exact_root"` | The Bessel shape parameter $c$ (used by `"bessel_profile"` and `"bessel_canyon_mean"`) as the root of the shape equation, to machine precision. | ✓ | |
| | `"grid_search"` | $c$ as the point of the grid $0.01, 0.02, \ldots, 1.00$ with the smallest residual, as MUNICH evaluates it (`ComputeSiraneC`). $c$ is quantised to 0.01, which moves $u_H$ by up to about 2 % against the exact root. The argmin is piecewise constant, so the gradient carried is the exact root's (straight-through). | | ✓ |
| `kappa` | float | The von Karman constant. | 0.40 | 0.41 |
| `canyon_wind_min` | m/s | Floor on $\lvert u \rvert$ in the canyon, sign kept. | 0 | 0.1 |
| `u_d_min` | m/s | Floor on the exchange velocity. | 0 | 0.001 |
| `sigma_w_min` | m/s | Floor on $\sigma_w$ before the exchange velocity. | 0.30 | 0 |
| `sigma_v_min` | m/s | Floor on $\sigma_v$ before the turbulence-intensity spread; acts only with `direction_spread="turbulence_intensity"` (the above-roof plume has floors of its own, `plume_table(sigma_v_min=, sigma_w_min=)`). | 0.5 | 0 |

Two more choices shape the drivers rather than the physics:

| Option | Values | What changes |
|---|---|---|
| `meteo` | `"uniform"` / `"per_street"` | One instance value per driver, or a trailing street axis on `U_ref`, `theta_w`, `h_abl`, `u_star`, `lmo`, with junction routing from the mean of the streets meeting there (or an explicit `"<key>_junction"` driver). |
| `background` | `"uniform"` / `"per_street"` | One atmosphere boundary node shared by every street, or one atmosphere node per street with its own `"<layer>.x_boundary"` row -- needed for the coupled above-roof plume, see [Above-roof concentration](above-roof.md), and for the puffs. |
| `puffs` | `None` / `PuffOptions(...)` | No above-roof model (the background is the driver), or the unsteady puff model writing every street's `C_ext`; see [Unsteady above-roof puffs](puffs.md). |

`kappa=None` is the preset's constant. When any closure option is given explicitly it is instead
0.41 if a choice written with that constant is selected (`canyon_wind="exponential_profile"`,
`roof_exchange="aspect_ratio_scaled"` or `roof_wind="canopy_log_law"`) and 0.40 otherwise. An
explicit value always wins. The two values are the constants `closures.KAPPA_040` and
`closures.KAPPA_041`. For example, `build_model(net, preset="munich",
canyon_wind="bessel_profile", roof_exchange="turbulent_velocity")` has $\kappa = 0.40$, not
the preset's 0.41, because no choice written with 0.41 is left.


## Canyon and boundary-layer quantities

`noodl.apps.street_aq.canyon` exposes the pieces directly (signatures in the [catalogue](../../reference/catalogue/street_aq.md)). `boundary_layer(h_mean, u_ref, h_abl, ...)` returns a `BoundaryLayer` with $d = 2h/3$, $z_0 = h/10$ and $u_* = \kappa U_{\text{ref}} / \ln((z_{\text{ref}} - d)/z_0)$; its `sigma_w` and `sigma_v` methods give the velocity standard deviations that drive the roof exchange. `canyon_velocity` returns the **signed** along-canyon velocity (m/s), `exchange_velocity` the roof exchange velocity $u_d$, `roof_wind` the roof-level wind $u_H$, `canopy_displacement_roughness` the Macdonald (1998) network means $(d_c, z_{0c})$, and `bessel_shape_parameter`, `grid_shape_parameter` and `shape_parameter` the Bessel shape parameter $c$ (a differentiable root, MUNICH's 0.01 grid with the exact root's gradient, or either by the `shape_constant` option).
