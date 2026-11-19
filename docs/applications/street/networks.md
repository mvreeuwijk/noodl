# Build and run a street network

How to describe a street network, supply the weather, emissions and background, run it to a steady state and, optionally, add NO-NO₂-O₃ chemistry. The closure options are on [Choosing the closures](closures.md); cases read from files are on [Reading and writing cases](cases.md).

## Describe the network

A network is a list of `Street` objects (one canyon segment between two junctions) plus a coordinate for every junction, in a `StreetNetwork`.

- `Street(name, u, v, length, width, height, z0_b=0.15, emission_scale=1.0)`: `emission_scale` multiplies the street's `street.sources` inside the model (so also for hand-supplied sources and `initial_drivers(values=...)`) and its emissions in `write_case`'s decks; it is matched by street name and applied once.
- `StreetNetwork(streets, x, y, street_direction="planar", lon=None, lat=None)`: the network plus junction coordinates. `azimuth` gives each street's direction in radians counter-clockwise from east, computed as `street_direction` says (below); `junctions` and `degree(node)` describe the topology.
- `from_test_network()` is a 4-junction, 3-street network; `twelve_street_grid(L=100.0, W=20.0, H=20.0)` is the 12-street network of Kim et al. (2022), Fig. 1 (MUNICH's idealised case). `L`, `W` and `H` are arguments because the paper never published them.

Construction validates unique street names, `u != v`, coordinates for every named junction, and strictly positive `length`, `width`, `height` and `z0_b`. The full signatures are in the [catalogue](../../reference/catalogue/street_aq.md).

For your own streets, build the `StreetNetwork` directly from whatever source you have (a GIS layer, OpenStreetMap, a hand-drawn sketch): one `Street` per canyon segment, named by its two junctions, plus a coordinate for every junction. The coordinates only set each street's bearing relative to the wind, so any projected system in metres will do; the lengths are yours to give.

How street directions come from the coordinates is `street_direction`:

| `street_direction` | Street direction |
|---|---|
| `"planar"` (default) | `atan2(dy, dx)` on the planar `x`/`y` metres. |
| `"midlatitude_bearing"` | From geographic `lon`/`lat` (degrees, one per junction): east-west distance `R cos(lat_mid) dlon` at the street's own mid-latitude, north-south `R dlat`, `R = 6371229 m`. This is MUNICH's `ComputeStreetAngle`, operation for operation, as `midlatitude_bearing(lon1, lat1, lon2, lat2)`. |

`read_case(format="munich")` uses `"midlatitude_bearing"`, since MUNICH's coordinates are lon/lat: its single projection to `x`/`y` (at the network's mean latitude) tilts each street by up to about `1e-4` rad on a city-sized network, which moves a junction's switch between inflow and outflow past a nearby wind sample. The directions are geometry constants (plain floats), not differentiable in the coordinates.

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

`build_model` creates one `atmosphere` boundary node plus one node per street, then, for every junction, directed `route` edges between all distinct street ends meeting there, two `vent` edges per (street, end), and two `exchange` edges per street. A `TransportLayer` and a `StreetFlows` closure are wrapped into a `Model`. `street_index(model)` maps street name to row index of `"street.x"`.

## Supply the drivers

From there the drivers are supplied as in the [worked example](index.md#a-worked-example): emissions in `street.sources` at each street's node, a background in `street.x_boundary`, and the wind and boundary layer in `U_ref`, `theta_w` and `h_abl`.

Emissions and the background are given by street and boundary name; the meteorology is
the street closures' own input and is given as plain tensors. In the dictionaries, the
concentration is `"street.x"` (street order), the emissions `"street.sources"` (full node
order) and the background `"street.x_boundary"` (boundary order) -- see
[Using noodl](../../guide/workflow.md).

**The drivers `build_model` returns are templates only.** They carry zero `x_boundary`
and `sources`; you must supply `U_ref`, `theta_w` and `h_abl` yourself. The
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

A case read with `read_case` supplies all of these automatically; see [Reading and writing cases](cases.md).

## Add chemistry

`Model.steady` never applies reactions. For a coupled transport-and-chemistry steady state use
`street_steady`, which iterates the two by successive substitution (`Model.fixed_point`, so
gradients are those of the converged fixed point, by the implicit adjoint, not of the unrolled
passes):

```python
import torch
from noodl.apps.street_aq import (
    build_model, initial_drivers, photostationary_for_streets, street_steady,
)

DT = torch.float64

reaction = photostationary_for_streets(("no", "no2", "o3"))
model, state, _ = build_model(net, species=("no", "no2", "o3"), chemistry=reaction)
street = model.refs.street
drivers = initial_drivers(model, values={
    street.sources: {"main_w": {"no": 2.0e-3, "no2": 4.0e-4}},       # kg/s, by species
    street.boundary_concentration: {"atmosphere": {"o3": 8.0e-8}},    # ozone aloft, kg/m3
    "U_ref": torch.tensor(3.0, dtype=DT), "theta_w": torch.tensor(0.0, dtype=DT),
    "h_abl": torch.tensor(800.0, dtype=DT),
    "J_NO2": torch.tensor(5.0e-3, dtype=DT),                          # photolysis, 1/s
    "temperature": torch.tensor(293.15, dtype=DT),                    # K
})
solved = street_steady(model, state, drivers, reaction=reaction, tol=1e-18, max_iter=200)
print(street.concentration.named(solved[street.concentration])["main_w"].shape)  # NO, NO2, O3
```

```text
torch.Size([3])
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

The equilibrium itself, the SIRANE rate and floor, and the photolysis forms are on the [physics page](physics.md#chemistry). `j_no2_zenith_table(zenith_deg, attenuation=1.0)` gives MUNICH's clear-sky photolysis rate; `solar_elevation` and `j_no2_elevation_cloud` give SIRANE's.

## Write results

`write_network_concentration(path, ...)` writes a `network_concentration_<year>.nc` with one
record per (time, street): the time axis, each street's feature index and OSM id, the background,
the signed canyon velocity and the concentration increment. The file is classic CDF (so `scipy`
can read it back) in float64 throughout. Inputs may be tensors that require grad (they are
detached to CPU first), and the file is written beside the destination and moved into place
only when complete, so a failed export never truncates an existing file. Needs the
[`street_aq` extra](../../get-started/installation.md#optional-extras).
