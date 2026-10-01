# Street air quality

Urban air quality on a street network. Road segments flanked by buildings are "canyons", and
each canyon is a node holding a well-mixed pollutant concentration (kg/m³); the atmosphere above
roof level is the boundary node. Edges carry air between streets at junctions, and exchange it
with the atmosphere at junctions and through the canyon roof. Pollutant mass is conserved in
every canyon: it builds up from traffic emissions (kg/s), is ventilated into the atmosphere, and
is carried from street to street by the wind-driven flow along them.

A street network is built by hand, as `StreetNetwork` and `Street` objects, or read from a
street-network case on disk with `read_case` -- see [Reading and writing street-network
cases](street_aq_cases.md), which covers both MUNICH's own file format and SIRANE's decks.

This application is the framework's clearest case of **closure-computed flows**. There is no
potential variable anywhere: the along-canyon velocity is a closed-form function of the wind
aloft and the canyon geometry, so a closure computes every flow and writes it, and the transport
layer advects on what it wrote.

```python
from noodl.apps.street_aq import (
    build_model, street_steady, street_index, initial_state,
    StreetNetwork, Street, from_test_network, twelve_street_grid,
    write_network_concentration, to_ug_m3,
    photostationary_for_streets,
)
```

![A street canyon in cross-section with roof-level exchange and a canyon vortex, and routing at a junction](../assets/app-street.svg)

## A worked example

```python
import math
import torch
from noodl.apps.street_aq import build_model, from_test_network

DT = torch.float64

net = from_test_network()                     # a 4-junction, 3-street toy network
model, state, _ = build_model(net, pblh_floor=False)

street_net = model.net
sources = torch.zeros(street_net.n, dtype=DT)
for name, value in zip(("r1", "r2", "r3"), (1.0, 2.0, 3.0), strict=True):
    sources[street_net.node_index(name)] = value

drivers = {
    "street.x_boundary": torch.tensor([1.0e-4], dtype=DT),   # background, kg/m3
    "street.sources": sources,                                # emissions
    "U_ref": torch.tensor(2.0, dtype=DT),                     # wind speed, m/s
    "theta_w": torch.tensor(0.25 * math.pi, dtype=DT),        # rad CCW from east, blowing TOWARD
    "h_abl": torch.tensor(1200.0, dtype=DT),                  # boundary-layer depth, m
}

solved = model.steady(state, drivers)
print(solved["street.x"])                     # kg/m3 per street, interior order
```

*(From `tests/apps/street_aq/test_conservation.py`, which then checks that every kilogram emitted
crosses the atmosphere boundary exactly, rtol $10^{-12}$.)*

**The drivers returned by `build_model` are templates only.** They carry zero-shaped
`x_boundary` and `sources`; you must supply `U_ref`, `theta_w` and `h_abl` yourself. The
Obukhov length `lmo` is optional: with `stability="monin_obukhov"` (the default) it selects the
stable, neutral or unstable branch, and without it the turbulence is the neutral form.

`u_star`, when given, is the friction velocity itself and replaces the log law that would
otherwise derive it from `U_ref`; `U_ref` is then needed only for the turbulence-intensity
direction spread (`direction_spread="turbulence_intensity"`), $\sigma_\theta = \sigma_v / U$.

With `direction_spread="driver"` (the default) the direction spread is the `sigma_theta`
driver (radians; one value per instance, or one per junction under `meteo="per_street"`) --
useful whenever the spread varies hour by hour, as SIRANE's does, since it needs no rebuilt
model. Without that driver it is the `sigma_theta` given to `build_model`, and without that
zero: the junction routing then uses the mean direction alone, exactly as
`direction_averaging="none"`, which ignores any spread.

`meteo="per_street"` gives every one of `U_ref`, `theta_w`, `h_abl`, `u_star` and `lmo` a
trailing street axis, `(..., n_streets)`, so each street gets its own wind and boundary layer.
Junction routing then reads an explicit `"<key>_junction"` driver when given --
`theta_w_junction`, `U_ref_junction`, `u_star_junction`, `h_abl_junction`, `lmo_junction`, in
`StreetNetwork.junctions` order -- and otherwise falls back to the mean of the streets meeting
at that junction (circular for direction, through $1/L$ for the Obukhov length).

`background="per_street"` gives every street its own atmosphere node, so `"<layer>.x_boundary"`
becomes `(n_streets,)` or `(n_streets, n_species)`, in street order, instead of one
network-wide value shared by every street.

A case read with `read_case` supplies all of these automatically -- see [Reading and writing
street-network cases](street_aq_cases.md).

## Building a network

| Object | Purpose |
|---|---|
| `Street(name, u, v, length, width, height, z0_b=0.15, emission_scale=1.0)` | One canyon segment between two junctions. |
| `StreetNetwork(streets, x, y)` | The network plus junction coordinates. `azimuth` gives each street's bearing in radians CCW from east; `junctions` and `degree(node)` describe the topology. |
| `from_test_network()` | A 4-junction, 3-street test network. |
| `twelve_street_grid(L=100.0, W=20.0, H=20.0)` | The 12-street network of Kim et al. 2022 Fig. 1 (MUNICH's idealised case). `L`, `W`, `H` are arguments because the paper never published them. |

Construction validates: unique street names, `u != v`, coordinates for every named junction, and
strictly positive `length`, `width`, `height` and `z0_b`.

For your own streets, build the `StreetNetwork` directly from whatever source you have (a GIS
layer, OpenStreetMap, a hand-drawn sketch): one `Street` per canyon segment, named by its two
junctions, plus a coordinate for every junction. The coordinates only set each street's bearing
relative to the wind, so any projected system in metres will do; the lengths are yours to give.

```python
import math
from noodl.apps.street_aq import Street, StreetNetwork, build_model, street_index

x = {"a": 0.0, "b": 200.0, "c": 200.0, "d": 400.0}    # junction coordinates, m
y = {"a": 0.0, "b": 0.0, "c": 150.0, "d": 0.0}

def street(name, u, v, width, height):
    return Street(name, u, v, math.hypot(x[v] - x[u], y[v] - y[u]), width, height)

net = StreetNetwork(
    streets=[street("main_w", "a", "b", 20.0, 18.0),
             street("main_e", "b", "d", 20.0, 18.0),
             street("side", "b", "c", 12.0, 15.0)],
    x=x, y=y,
)
model, state, drivers = build_model(net)
street_index(model)                           # {'main_w': 0, 'main_e': 1, 'side': 2}
```

From there the drivers are supplied exactly as in the worked example above: emissions in
`street.sources` at each street's node, a background in `street.x_boundary`, and the wind and
boundary layer in `U_ref`, `theta_w` and `h_abl`.

## Reading and writing street-network cases

A street-network case bundles a network with the meteorology, emissions and background
concentrations that drive it hour by hour, read from a MUNICH case directory or a SIRANE
master `.dat` file with `read_case`, and driven with `drivers_at` -- see [Reading and
writing street-network cases](street_aq_cases.md) for `StreetCase`, `read_results` /
`StreetResults`, `model_options()`, `write_case` and `write_sweep`.

## `build_model`

```python
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
    shape_constant=None,          # 'exact_root' | 'grid_search'
    species=("nox",),
    chemistry=None,
    kappa=None, canyon_wind_min=None, u_d_min=None, sigma_w_min=None, sigma_v_min=None,
    z_ref=30.0, pblh_floor=True,
    meteo="uniform",              # 'uniform' | 'per_street'
    background="uniform",         # 'uniform' | 'per_street'
)
```

Every closure keyword left at `None` takes the preset's value; any other value overrides it,
so `build_model(net, preset="munich", roof_wind="canopy_log_law")` is MUNICH's closure set
with Macdonald's roof wind.

**The defaults are SIRANE's.** A model built with no options floors $\sigma_w$ at 0.30 m/s
before the roof exchange velocity (`sigma_w_min=0.30`), which binds in light winds and under
low buildings; pass `sigma_w_min=0.0` for no floor. The default street chemistry,
`photostationary_for_streets(species)`, is the Soulhac et al. (2011) rate with a 2 ppb floor
on $J/k$. `preset="munich"` gives MUNICH's settings throughout: no turbulence floors, and for
the chemistry the JPL (2003) rate with no floor.

It builds one `atmosphere` boundary node plus one node per street, then, for every junction,
directed `route` edges between all distinct street ends meeting there, two `vent` edges per
(street, end), and two `exchange` edges per street. A `TransportLayer` and a `StreetFlows`
closure are wrapped into a `Model`.

`street_index(model)` maps street name to row index of `"street.x"`.

### The closure choices

Every option value is named for the physics it computes. The two reference models' full sets
are the presets (`noodl.apps.street_aq.PRESETS`); the options are independent, so any mix is
allowed:

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
| `background` | `"uniform"` / `"per_street"` | One atmosphere boundary node shared by every street, or one atmosphere node per street with its own `"<layer>.x_boundary"` row -- needed for the coupled above-roof plume, see [Above-roof concentration](#above-roof-concentration). |

`kappa=None` is the preset's constant. When any closure option is given explicitly it is instead
0.41 if a choice written with that constant is selected (`canyon_wind="exponential_profile"`,
`roof_exchange="aspect_ratio_scaled"` or `roof_wind="canopy_log_law"`) and 0.40 otherwise. An
explicit value always wins. The two values are the constants `closures.KAPPA_040` and
`closures.KAPPA_041`. For example, `build_model(net, preset="munich",
canyon_wind="bessel_profile", roof_exchange="turbulent_velocity")` has $\kappa = 0.40$, not
the preset's 0.41, because no choice written with 0.41 is left.

#### Earlier option names

The earlier spellings are still accepted, with a `DeprecationWarning` naming the replacement
(`noodl.apps.street_aq.closures.LEGACY`):

| Earlier | Current |
|---|---|
| `exchange=`, `routing=`, `roof_wind_form=` | `roof_exchange=`, `junction_routing=`, `roof_wind=` |
| `canyon_wind="soulhac"` / `"exponential"` | `"bessel_profile"` / `"exponential_profile"` |
| `roof_wind_form="sirane"` / `"macdonald"` | `roof_wind="bessel_canyon_mean"` / `"canopy_log_law"` |
| `exchange="sirane"` / `"schulte"` | `roof_exchange="turbulent_velocity"` / `"aspect_ratio_scaled"` |
| `routing="sirane"` / `"mixing"` | `junction_routing="non_crossing_streamlines"` / `"perfect_mixing"` |
| `direction_averaging="sirane"` / `"gauss"` | `"exact_gaussian"` / `"gauss_hermite"` |
| `direction_averaging="munich"` | `"rectangle_rule"` with `direction_spread="turbulence_intensity"` |
| `stability="munich"` | `"monin_obukhov"` |
| `photostationary_for_streets(closure=...)` | `preset=...` |

The lower-level functions take the same keyword as the option they implement; their earlier
keywords are accepted with a `DeprecationWarning` too:

| Earlier | Current |
|---|---|
| `canyon_velocity(form=...)` | `canyon_velocity(canyon_wind=...)` |
| `roof_wind(form=...)` | `roof_wind(roof_wind=...)` |
| `exchange_velocity(form=...)` | `exchange_velocity(roof_exchange=...)` |
| `routing_matrix(model=...)` | `routing_matrix(junction_routing=...)` |
| `direction_offsets(scheme=...)` | `direction_offsets(direction_averaging=...)` |

The earlier, model-named functions and constants are kept as aliases of the same objects,
without a warning:

| Earlier | Current |
|---|---|
| `soulhac_shape`, `soulhac_residual` | `bessel_shape_parameter`, `bessel_shape_residual` |
| `macdonald_profile` | `canopy_displacement_roughness` |
| `sigma_theta_munich`, `n_theta_munich` | `sigma_theta_turbulence_intensity`, `n_theta_rectangle_rule` |
| `sirane_direction_samples` | `exact_gaussian_direction_samples` |
| `munich_idealised` | `twelve_street_grid` |
| `SIRANE_EXCHANGE`, `SCHULTE_BETA` | `EXCHANGE_SIGMA_W_RATIO`, `ASPECT_RATIO_EXCHANGE_BETA` |
| `KAPPA_MUNICH` | `KAPPA_041` (and `KAPPA` is `KAPPA_040`) |
| `GAMMA_E` | `EULER_GAMMA_TRUNCATED` (0.577, deliberately truncated) |
| `MAX_SIGMA_THETA_SIRANE`, `SIRANE_WINDOW` | `MAX_SIGMA_THETA_EXACT`, `EXACT_GAUSSIAN_WINDOW` |
| `SEUIL_GAUSS` | `GAUSS_CUTOFF_SIGMAS` |
| `SIRANE_K3_PREFACTOR`, `SIRANE_K3_ACTIVATION` | `K_NO_O3_SOULHAC2011_PREFACTOR`, `K_NO_O3_SOULHAC2011_ACTIVATION` |
| `SIRANE_K_FLOOR_PPB` | `PHOTOSTATIONARY_FLOOR_PPB` |
| `k_no_o3_munich`, `k_no_o3_sirane` | `k_no_o3_jpl2003`, `k_no_o3_soulhac2011` |
| `j_no2`, `j_no2_sirane` | `j_no2_zenith_table`, `j_no2_elevation_cloud` |

### The canyon physics

`noodl.apps.street_aq.canyon` exposes the pieces directly:

| Function | Returns |
|---|---|
| `boundary_layer(h_mean, u_ref, h_abl, *, z_ref=30.0, kappa=0.4, pblh_floor=None)` | A `BoundaryLayer` with $d = 2h/3$, $z_0 = h/10$, $u_* = \kappa U_{\text{ref}} / \ln((z_{\text{ref}} - d)/z_0)$. |
| `BoundaryLayer.sigma_w(z, ...)` / `.sigma_v(...)` | Velocity standard deviations driving the roof exchange. |
| `canyon_velocity(W, H, phi, ...)` | The **signed** along-canyon velocity, m/s. |
| `exchange_velocity(sigma_w, H, W, roof_exchange=..., u_d_min=0.0, sigma_w_min=0.0)` | The roof exchange velocity $u_d$. |
| `roof_wind(u_star, H, W, roof_wind=...)` | Roof-level wind $u_H$. |
| `canopy_displacement_roughness(h_mean, w_mean, ...)` | $(d_c, z_{0c})$, Macdonald (1998) network means. |
| `bessel_shape_parameter(ratio)` | The Bessel shape parameter $c$, as a differentiable root. |
| `grid_shape_parameter(ratio)` | $c$ on MUNICH's 0.01 grid, with the exact root's gradient. |
| `shape_parameter(ratio, shape_constant)` | Either of the two, by the `shape_constant` option. |

## Chemistry

`Model.steady` never applies reactions. For a coupled transport-and-chemistry steady state use
`street_steady`, which iterates the two by successive substitution:

```python
from noodl.apps.street_aq import photostationary_for_streets, street_steady

reaction = photostationary_for_streets(("no", "no2", "o3"))
model, state, drivers = build_model(net, species=("no", "no2", "o3"), chemistry=reaction)
drivers["temperature"] = torch.tensor(293.15, dtype=torch.float64)   # K
solved = street_steady(model, state, drivers, reaction=reaction, tol=1e-18, max_iter=200)
```

`photostationary_for_streets(species, *, preset="sirane", no_o3_rate=None, floor_ppb=None,
j_key="J_NO2")` wires the Leighton NO/NO₂/O₃ equilibrium to the matching columns of `species`,
case-insensitively, and raises naming any missing one. The preset sets the rate of NO + O3
and the floor on $J/k$; `no_o3_rate` (`"soulhac_2011"`, `"jpl_2003"` or any callable $k(T)$
in m³ mol⁻¹ s⁻¹) and `floor_ppb` override them:

| `preset` | Rate $k$ | Floor on $J/k$ |
|---|---|---|
| `"sirane"` (default) | `k_no_o3_soulhac2011(T)` $= 1.325\times10^{6}\exp(-1430/T)$ m³ mol⁻¹ s⁻¹ | 2 ppb |
| `"munich"` | `k_no_o3_jpl2003(T)` $= 3.0\times10^{-12}\exp(-1500/T)$ cm³ molecule⁻¹ s⁻¹ | none |

### The Leighton equilibrium

The three species relax to the state satisfying $J\,[\mathrm{NO_2}] = k\,[\mathrm{NO}][\mathrm{O_3}]$
at the same NOx ($[\mathrm{NO}]+[\mathrm{NO_2}]$) and Ox ($[\mathrm{NO_2}]+[\mathrm{O_3}]$) as the
input, conserved in molar terms: it is an equilibrium, not a rate, and applying it twice changes
nothing. $J$ (the `J_NO2` driver, 1/s) is supplied by you; `j_no2_zenith_table(zenith_deg,
attenuation=1.0)`
gives the clear-sky rate from MUNICH's 11-point tabulation, piecewise-linear between the
tabulated zenith angles. **Solar geometry is not computed** for this form — you pass the zenith
angle you want.

The rate $k$ of NO + O3 is evaluated at the driver `temperature` (K), which is now **required**:
with no constant override, a missing `temperature` driver raises by name rather than assuming a
value. The core `Photostationary` reaction (`noodl.layers.reaction`) defaults to the JPL rate
`k_no_o3_jpl2003(T) = 3.0\times10^{-12}\exp(-1500/T)` cm³ molecule⁻¹ s⁻¹ (NASA/JPL 2003, as in
MUNICH); to use a constant 298 K rate instead of evaluating it at the air temperature, pass
the constant override `k_no_o3=K_NO_O3` (`noodl.layers.reaction.K_NO_O3`, the 298 K value in
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

## Writing results

`write_network_concentration(path, ...)` writes a `network_concentration_<year>.nc` with one
record per (time, street): the time axis, each street's feature index and OSM id, the background,
the signed canyon velocity and the concentration increment. The file is classic CDF (so `scipy`
can read it back) in float64 throughout. Needs the
[`street_aq` extra](../installation.md#optional-extras).

## Above-roof concentration

By default the atmosphere above the canyon is whatever the `"<layer>.x_boundary"` driver
says -- an input, not a modelled quantity. Optionally, as in SIRANE, `C_ext`, the
concentration just above each street's roof, is instead computed as the sum of a fixed
background and the superposed plumes of every upwind street's roof flux and every upwind
junction's vertical flux:

```
C_ext = C_bg + K_s F_s + K_j F_j
```

`F_s` is each street's signed roof flux, $u_d W L (C - C_{\text{ext}})$ -- read off the same
`exchange` edges the transport layer already carries, so it conserves mass exactly with the
street solve. `F_j` is each junction's vertical flux: the excess over background carried up
out of the canopy through its outgoing `vent` edges (`junction_source="upward"`, the
default), or that minus what comes back down through the incoming vents
(`junction_source="net"`, for comparison). `K_s` and `K_j` are kernels, s/m3, built once per
hour from the network geometry and meteorology by `street_kernel` and `junction_kernel`
(`noodl.apps.street_aq.plume`).

### The kernel

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

### Empirical inputs

Four elements of the mechanism are fitted to SIRANE's output rather than derived from the
paper, each a named parameter of `plume_table` with a default:

| Parameter | Default | What it is |
|---|---|---|
| `tau_y`, `tau_z` | `TAU_Y_BY_SIGMA_V`, `TAU_Z_BY_SIGMA_W`: piecewise-linear tables of the hour's floored `sigma_v`, `sigma_w` | The time offsets behind which `sigma_y` (Eq. 28) and `sigma_z` (Eq. 29) are evaluated. |
| `centre_c`, `centre_k` | 10 m, 0.675 | The plume-centre law, `z_c = max(H_R, d + E\|N(c - d, (k sigma_z)^2)\|)`. |
| `theta_star` | `u*^2 T / (kappa g L)` | The temperature scale in the Brunt-Vaisala frequency. On the one stable hour with a real SIRANE comparison (South Kensington), SIRANE's own preprocessor prints 0.072 K, but 0.060 K reproduces its output far better (see Verification). |
| `meteo_cell_dx` | none (falls back to the table's `x_max`, 700 m) | The x-size of SIRANE's meteorology grid cell, for the downwind cut-off. |

All four were fitted on the SIRANE reference data described above, across neutral and
stable meteorology and a range of street orientation, width, height and length.

### Differentiability

The kernel is linear in the fluxes, and differentiable in `u*`, the boundary-layer height,
the Obukhov length, the wind direction, `sigma_theta`, the roughness `z0`, the displacement
height `d`, the reflection height `H_R`, the temperature and the four empirical parameters
above; a street's width and height (which set its sub-sources' widths and caps) carry no
gradient. Several elements are piecewise constant or linear -- exact to SIRANE's own
mechanism, not a smoothing choice: the trajectory's speed steps through integer heights,
the 10 m table and its linear reads have kinks every 10 m, the default `tau_y`/`tau_z`
tables are piecewise linear with a floor, and the crosswind and downwind cut-offs are hard
(a pair that crosses one jumps to exactly zero, and its gradient is that of whichever side
it sits on).

Not implemented: the unstable regime (a negative Monin-Obukhov length -- the paper's
bi-Gaussian and Lagrangian time scale for that regime cannot be transcribed as printed),
plume rise, wet-deposition depletion, and SIRANE's cell-based retrotrajectory path
(`B_RUE_DECOUP = 1`).

### Cost

Measured at Paris scale (577 streets, one hour, CPU): building the plume table takes about
0.03 s and the street kernel about 0.5 s; the whole coupled forward solve below is
0.9-2.0 s.

`street_steady_with_plume` (`noodl.apps.street_aq.above_roof`) solves the street network and
this relation together: starting from `C_ext = C_bg`, it solves the network, recomputes
`C_ext` from the resulting fluxes, and repeats until the largest change (relative to the
largest `|C_ext|`) falls below `tol` -- typically 6-13 passes on a network the size of
Paris. It needs a model built with `background="per_street"`, and takes the two kernels as
required keyword arguments: there is no `build_model` option for this, since the kernels
depend on the wind and must be rebuilt every hour. The gradient is that of the converged
fixed point, by an implicit adjoint (one GMRES solve per instance, hours never coupled).

| Option | Values | What changes |
|---|---|---|
| `junction_source` | `"upward"` / `"net"` | The junction plume source: the excess flux carried up out of the canopy, or that minus the excess carried back down. `"upward"` is the default; on SIRANE's archived output the two agree equally well. |
| `self_contribution` (`street_kernel`) | `False` / `True` | Whether a street's own roof-flux points contribute to its own `C_ext`. Excluded by default: including them overstates `C_ext` (measured median ratio 2.33 against SIRANE's own fluxes). |
| `meteo_cell_dx` (`street_kernel`, `junction_kernel`) | the x-size of SIRANE's meteorology grid cell, m | The downwind cut-off, `meteo_cell_dx / \|cos theta_w\|`; `None` (the default) uses the table's own extent, 700 m. |

## Verification

### MUNICH formulas

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
`preset="sirane"` keeps the exact root. With the grid search and a steady solve per hour (MUNICH's
`With_stationary_hypothesis: yes`), noodl physics matches MUNICH's output on the idealised
12-street case to 4.3e-8 relative at most, the float32 precision of that output.

### MUNICH idealised 12-street case

Kim et al. (2022) Fig. 1 gives concentrations on an idealised 12-street network, but not the
street length, width and height behind them, so `twelve_street_grid(L, W, H)` takes them as
arguments and the case can only be compared in terms that do not depend on them.

| Check | Target | Measured |
|---|---|---|
| Linearity in wind speed at 210° and 240°: doubling $U$ halves every concentration | < 1e-9 | holds (an exact model identity) |
| 270° canyon-wind ratio pattern | paper's value | 1.99451, against the paper's 1.99451 |
| Concentrations relative to one reference street, 19 ratios | 5 % | **not met**: worst ratio off by 51 %, 7 of 19 within 15 % |

The relative-pattern miss is the one open discrepancy against MUNICH; it is listed under
[Limitations](#limitations-and-caveats).

### SIRANE

A code-to-code comparison against SIRANE v2.1 rev 128 on its own South Kensington network
(46 streets, 36 junctions), using part of its archived results -- hours 00 and 01 of
7 January 2014 (`tests/verification/test_sirane.py`; the fixture and its own caveats are in
`tests/data/street/sirane_south_kensington/NOTICE.md`). That archived run's own meteorology
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

  Ten further runs of one isolated street under imposed meteorology (across, along and
  oblique to the wind, neutral and stable; `tests/data/street/sirane_kernel_probe`) check
  the kernel directly against SIRANE's own concentration grids. These are in sample -- the
  empirical parameters above are fitted on them -- and give a per-run median relative
  difference of about 0.02-0.6 % over the sampled cells. Including a street's own roof-flux
  points in its own `Cext` (`self_contribution=True`) overstates it (measured median ratio
  2.33 on the South Kensington fluxes), which is why it is excluded by default.

- **Chemistry (`preset="sirane"`).** A code-to-code comparison against SIRANE v2.1 output with
  its NO-NO2-O3 chemistry switched on (`tests/data/street/sirane_chemistry`;
  `tests/verification/test_sirane_chemistry.py`), by case:
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

## Limitations and caveats

- **The MUNICH 12-street idealised case is not reproduced.** Its street geometry was never
  published, so absolute concentrations cannot be compared, and the pattern of concentrations
  relative to one street misses its 5 % target: the worst of the nineteen ratios is off by
  51 %, and seven are within 15 %. The scale-invariant checks (linearity in wind speed, the
  270° canyon-wind ratio) do match; see [Verification](#munich-idealised-12-street-case).
- **Tall streets need the boundary-layer guard.** The neutral $\sigma_w$ goes negative when a
  street is taller than $1.25\,h_{\text{abl}}$. `build_model`'s `pblh_floor=True` (the default)
  applies MUNICH's guard, raising the boundary-layer height to at least the tallest street. With
  `pblh_floor=False`, `exchange_velocity` raises an error on such a step rather than clamping
  it.
- **SIRANE exchange coefficient.** The code uses $\sigma_w / (\sqrt{2}\,\pi)$
  (`EXCHANGE_SIGMA_W_RATIO` = 0.225079…), as in MUNICH's source, not $\sigma_w/\sqrt{2\pi}$.

## Install

```bash
pip install "noodl-physics[street_aq]"
```

Needed only for `write_network_concentration`. The modelling API — `build_model`,
`street_steady`, the closures — needs only the base dependencies.
