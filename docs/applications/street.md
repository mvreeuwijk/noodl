# Street canyons

Urban air quality on a street network — the SIRANE/MUNICH class of problem. Road segments
flanked by buildings are "canyons"; pollutant builds up in each from traffic emissions, is
ventilated into the atmosphere above roof level, and is transported between streets through
junctions.

This application is the framework's clearest case of **closure-computed flows**. There is no
potential variable anywhere: the along-canyon velocity is a closed-form function of the wind
aloft and the canyon geometry, so a closure computes every flow and writes it, and the transport
layer advects on what it wrote.

```python
from noodl.apps.street import (
    build_street_model, street_steady, street_index, initial_state,
    StreetNetwork, Street, from_test_network, munich_idealised,
    read_aqdt, write_network_concentration, to_ug_m3,
    photostationary_for_streets,
)
```

![A street canyon in cross-section with roof-level exchange and a canyon vortex, and routing at a junction](../assets/app-street.svg)

## A worked example

```python
import math
import torch
from noodl.apps.street import build_street_model, from_test_network

DT = torch.float64

net = from_test_network()                     # a 4-junction, 3-street toy network
model, state, _ = build_street_model(net, pblh_floor=False)

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

*(From `tests/apps/street/test_conservation.py`, which then checks that every kilogram emitted
crosses the atmosphere boundary exactly, rtol $10^{-12}$.)*

**The drivers returned by `build_street_model` are templates only.** They carry zero-shaped
`x_boundary` and `sources`; you must supply `U_ref`, `theta_w` and `h_abl` yourself (and `lmo`
if you chose `stability="munich"`).

## Building a network

| Object | Purpose |
|---|---|
| `Street(name, u, v, length, width, height, z0_b=0.15, emission_scale=1.0)` | One canyon segment between two junctions. |
| `StreetNetwork(streets, x, y)` | The network plus junction coordinates. `azimuth` gives each street's bearing in radians CCW from east; `junctions` and `degree(node)` describe the topology. |
| `from_test_network()` | IMPAQ's 4-junction, 3-road network, reproduced exactly. |
| `munich_idealised(L=100.0, W=20.0, H=20.0)` | The 12-street network of Kim et al. 2022 Fig. 1. `L`, `W`, `H` are arguments because the paper never published them. |
| `read_aqdt(...)` | A real AQ_DT domain from its GeoJSON and NetCDF products. |

Construction validates: unique street names, `u != v`, coordinates for every named junction, and
strictly positive `length`, `width`, `height` and `z0_b`.

## `build_street_model`

```python
model, state, drivers = build_street_model(
    net,
    canyon_wind="soulhac",        # 'soulhac' | 'exponential'
    exchange="sirane",            # 'sirane'  | 'schulte'
    routing="sirane",             # 'sirane'  | 'mixing'
    direction_averaging="none",   # 'none' | 'munich' | 'gauss'
    species=("nox",),
    chemistry=None,
    stability="impaq",            # 'impaq' | 'munich'
    roof_wind_form="sirane",      # 'sirane' | 'macdonald'
    kappa=None, canyon_wind_min=0.0, u_d_min=0.0,
    z_ref=30.0, pblh_floor=True,
)
```

It builds one `atmosphere` boundary node plus one node per street, then, for every junction,
directed `route` edges between all distinct street ends meeting there, two `vent` edges per
(street, end), and two `exchange` edges per street. A `TransportLayer` and a `StreetFlows`
closure are wrapped into a `Model`.

`street_index(model)` maps street name to row index of `"street.x"`.

### The closure choices

These select between the IMPAQ/SIRANE and MUNICH lineages, and they are independent:

| Option | Values | What changes |
|---|---|---|
| `canyon_wind` | `"soulhac"` | Soulhac–Perkins–Salizzoni (2008) closed-form Bessel profile from the friction velocity. Needs `z0_b`. |
| | `"exponential"` | MUNICH's K22 Eq. (B14) exponential profile from the roof-level wind. **Not** the K18 formula of the same name — they diverge by 37% for narrow canyons. |
| `roof_wind_form` | `"sirane"` / `"macdonald"` | How $u_H$ is computed when `canyon_wind="exponential"`. Macdonald uses a log law with a network-mean displacement height and roughness. |
| `exchange` | `"sirane"` | $u_d = \sigma_w / (\sqrt{2}\,\pi)$, aspect-ratio independent. |
| | `"schulte"` | $u_d = \sigma_w \beta / (1 + H/W)$, MUNICH v2's default. Equals the SIRANE form exactly at $H = W$. |
| `routing` | `"mixing"` / `"sirane"` | Perfect mixing, or SIRANE's non-crossing-streamline rule. They differ only at junctions with 2+ inflows **and** 2+ outflows. |
| `direction_averaging` | `"none"` / `"munich"` / `"gauss"` | Single direction; MUNICH's own quadrature over a turbulence-derived $\sigma_\theta$; or noodl's normalised Gauss–Hermite rule. |
| `stability` | `"impaq"` / `"munich"` | Neutral only, or MUNICH's three-branch stability dependence (needs an `lmo` driver). |

`kappa=None` resolves automatically: MUNICH's 0.41 if any MUNICH-style option is chosen, else
IMPAQ's 0.40. An explicit value always wins.

For strict IMPAQ parity:

```python
build_street_model(net, canyon_wind="soulhac", exchange="sirane",
                   direction_averaging="none", kappa=0.4, canyon_wind_min=0.0,
                   u_d_min=0.0, stability="impaq", z_ref=30.0, pblh_floor=False)
```

### The canyon physics

`noodl.apps.street.canyon` exposes the pieces directly:

| Function | Returns |
|---|---|
| `boundary_layer(h_mean, u_ref, h_abl, *, z_ref=30.0, kappa=0.4, pblh_floor=None)` | A `BoundaryLayer` with $d = 2h/3$, $z_0 = h/10$, $u_* = \kappa U_{\text{ref}} / \ln((z_{\text{ref}} - d)/z_0)$. |
| `BoundaryLayer.sigma_w(z, ...)` / `.sigma_v(...)` | Velocity standard deviations driving the roof exchange. |
| `canyon_velocity(W, H, phi, ...)` | The **signed** along-canyon velocity, m/s. |
| `exchange_velocity(sigma_w, H, W, form=...)` | The roof exchange velocity $u_d$. |
| `roof_wind(u_star, H, W, form=...)` | Roof-level wind $u_H$. |
| `macdonald_profile(h_mean, w_mean, ...)` | $(d_c, z_{0c})$, Macdonald (1998) network means. |
| `soulhac_shape(ratio)` | The Bessel shape parameter $c$, as a differentiable root. |

## Chemistry

`Model.steady` never applies reactions. For a coupled transport-and-chemistry steady state use
`street_steady`, which iterates the two by successive substitution:

```python
from noodl.apps.street import photostationary_for_streets, street_steady

reaction = photostationary_for_streets(("no", "no2", "o3"))
model, state, drivers = build_street_model(net, species=("no", "no2", "o3"), chemistry=reaction)
solved = street_steady(model, state, drivers, reaction=reaction, tol=1e-18, max_iter=200)
```

`photostationary_for_streets(species, j_key="J_NO2")` wires the Leighton NO/NO₂/O₃ cycle to the
matching columns of `species`, case-insensitively, and raises naming any missing one.
`j_no2(zenith_deg, attenuation=1.0)` gives the clear-sky photolysis rate from MUNICH's 11-point
tabulation. **Solar geometry is not computed** — you pass the zenith angle you want.

## Reading and writing AQ_DT products

```python
data = read_aqdt(stage1_dir, stage2_dir, year=2024, select="network_transport",
                 emissions="normalized", align="emission_key")
```

Reads junction and edge GeoJSON, plus forcing and emission NetCDF, into an `AqdtData` carrying a
`StreetNetwork`, a `Forcing` record and an emission array. Needs the
[`street` extra](../installation.md#optional-extras).

Its ambiguities are handled by **explicit failure rather than silent defaults**, and this is
worth knowing before you point it at your own data:

- The file's `reference_height_m` label is not trusted — the wind is really ERA5 10 m `u10`/`v10`
  with no extrapolation. Override with `trust_file_height=True` only if you know better.
- `align="emission_key"` (the default) matches emission rows to features through the
  `(osmid, u, v)` key table. `align="edge_index"` trusts the NetCDF's positional contract, and is
  *verified* against the key table when present — on a real `leiden_small` snapshot the two
  disagreed on 515 of 904 rows, and the reader raises naming them.
- `emissions="kg_per_year"` raises if that series is non-finite anywhere, which it is on the
  `leiden_small` snapshot.

`write_network_concentration(path, ...)` writes `network_concentration_<year>.nc` in AQ_DT's own
layout, with two recorded differences: classic CDF rather than NETCDF4 (so `scipy` can read it),
and float64 rather than float32.

## Validation

### MUNICH formulas

Thirteen exact input/output pairs transcribed from MUNICH's own source, each checked at the
precision that source publishes:

| Quantity | Tolerance | Measured |
|---|---|---|
| `SIRANE_EXCHANGE` constant | < 1e-15 | 0.225079079039277, exact |
| Exchange velocity $u_d$ | < 1e-7 relative | 0.08681174 m/s to 8 significant figures |
| `soulhac_shape` root $c$ | < 1e-13 | 0.6198293039179747 |
| Macdonald $d_c$, $z_{0c}$ | < 1e-12 | 4.617352498423888 m, 0.6614635677623194 m |
| Direction-quadrature weight sums, $n = 2\ldots10$ | 1e-6 | e.g. 0.974953 at $n=10$ |

That last row is deliberate: MUNICH's weights are **not** normalised to 1, and reproducing that
artefact is the point. "Fixing" it would introduce a bias relative to MUNICH rather than remove
one. The same applies to `soulhac_shape`: MUNICH quantises the root to a 0.01 grid (0.62), noodl
solves it continuously (0.6198293…), and the resulting 4e-4 relative difference in $u_M$ is
documented rather than matched.

### IMPAQ oracle

`impaq.py` is a byte-faithful numpy/scipy port of the AQ_DT prototype, kept so parity tests need
no external checkout. It reproduces the prototype to rtol $10^{-12}$.

On IMPAQ's four-node network, after fixing the prototype's two documented bugs, noodl and the
oracle agree to **machine precision** (< 1e-9 relative). With those bugs left unfixed the two
disagree by **30–45%** — which is reported, not hidden. On the real `leiden_small` domain (162
streets, 230 junctions) canyon velocities match exactly and concentrations match the fixed oracle
with median relative difference below $10^{-9}$.

Parity testing also turned up a **third, undocumented defect** in IMPAQ's `flow_route`: it sorts
by angle with `argsort` but un-sorts with `order` rather than `argsort(order)`, mis-permuting
routing at three-way junctions and breaking the oracle's own conservation — measured at 12 roads
at one step, worst factor 13.95. The port reproduces it faithfully, because it is the oracle, not
the model.

## Limitations and caveats

- **`impaq.py` is an oracle, not a model path.** It is numpy and scipy, it is not
  differentiable, and nothing else in the application imports it.
- **The MUNICH 12-street idealised case cannot be reproduced absolutely.** Its published figure
  depends on a geometry that was never published. Scale-invariant properties *are* checked and do
  pass — exact 2x linearity in wind speed (< 1e-9, a provable model identity), and the 270°
  canyon-wind ratio pattern (model 1.99451 vs paper 1.99451). But the absolute relative-pattern
  comparison **does not reach its 5% target**: the worst residual is 0.507, with seven of
  nineteen ratios inside 15%. The test asserts `worst < 0.6` as a loose regression guard and says
  so, rather than reporting a false pass.
- **IMPAQ's $\sigma_w$ is unguarded** and goes negative when a street is taller than
  $1.25\,h_{\text{abl}}$ — measured on 7 of the 474,336 (time, street) pairs of `leiden_small`.
  `exchange_velocity` then raises rather than silently clamping. `pblh_floor=True` (the default)
  applies MUNICH's guard; strict IMPAQ parity needs `pblh_floor=False` and accepts the risk.
- **A retracted claim.** Earlier documentation held that the SIRANE exchange coefficient should
  be $\sigma_w/\sqrt{2\pi}$ rather than $\sigma_w/(\sqrt{2}\,\pi)$. That was retracted after
  checking the source PDFs at glyph level; the code uses the latter and a test pins it.
- **The saved real AQ_DT product used for validation was found stale** against its own geometry
  file — 160 edges vs 162 features, 94 of 160 rows with mismatched `edge_osmid`. That test skips
  itself with the diagnosis recorded rather than reporting a false pass or fail.

## Install

```bash
pip install "noodl[street]"
```

Needed for `read_aqdt`, `write_network_concentration` and `impaq.py`. The modelling API —
`build_street_model`, `street_steady`, the closures — needs only the base dependencies.
