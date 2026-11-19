# A release in a street network, for every wind direction

**Goal.** A vehicle idles in one street of a small city block: a source of a pollutant at a
constant rate. Where does the pollutant go, and how does that depend on the wind direction?
Because the model runs with a leading batch dimension, all 24 wind directions are one call,
not 24 runs.

The network is the 12-street grid used in the literature on street-network models: four
junctions on a square, with a dead-end stub on each. Street "11" is the emitter. The steady
concentration in each street comes from a mass balance of the street, the exchange with the air
above the roofs, and the routing of the air at each junction.

```python
import math
import torch
from noodl.apps.street_aq import build_model, initial_drivers, twelve_street_grid

net, names = twelve_street_grid()
model, state, _ = build_model(net, pblh_floor=False)
street = model.refs.street

n = 24
angle = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)    # blowing toward, rad
drivers = initial_drivers(model, values={
    street.boundary_concentration: {"atmosphere": 0.0},             # clean air aloft
    street.sources: {"11": 1.0e-3},                                  # kg/s in street 11
    "U_ref": torch.full((n,), 3.0, dtype=torch.float64),             # wind speed, m/s
    "theta_w": angle,
    "h_abl": torch.full((n,), 1000.0, dtype=torch.float64),          # boundary layer, m
})
state = {key: value.expand(n, *value.shape).clone() for key, value in state.items()}

solved = model.steady(state, drivers)
ug = solved[street.concentration] * 1e9                              # ug/m3, shape (24, 12)

reached = (ug > 1.0).sum(dim=-1)
for degrees in (0, 45, 90, 180):
    i = degrees // 15
    print(f"wind toward {degrees:3d} deg: street 11 {float(ug[i, 10]):7.1f} ug/m3, "
          f"{int(reached[i])} streets above 1 ug/m3")
```

```text
wind toward   0 deg: street 11  3068.4 ug/m3, 1 streets above 1 ug/m3
wind toward  45 deg: street 11  1161.9 ug/m3, 4 streets above 1 ug/m3
wind toward  90 deg: street 11   924.1 ug/m3, 3 streets above 1 ug/m3
wind toward 180 deg: street 11  3068.4 ug/m3, 1 streets above 1 ug/m3
```

![Street map coloured by concentration for three wind directions](../assets/examples/street-release.svg#only-light)
![Street map coloured by concentration for three wind directions](../assets/examples/street-release-dark.svg#only-dark)

Street 11 runs north to south. A wind across it (0 and 180 degrees) sweeps no air along the
street, so the pollutant leaves only through the roof and the emitter street reads highest of
all, with no other street polluted. A wind along it (90 degrees) carries the pollutant out of
the street and on to its neighbours at the junction, and the emitter's own concentration
falls to under a third. In between, the junction routes the plume into more streets.

**Things to try.** Ask which direction gives the highest exposure in a given street by
looking along the first axis of `ug[:, k]`. Because the solve is differentiable, the
gradient of that exposure with respect to the wind direction or the emission rate comes from
one backward pass. To run a real city, read a MUNICH or SIRANE case with `read_case`; see
[Street air quality](../applications/street/index.md) and the case reader in
[Street cases](../applications/street/cases.md).
