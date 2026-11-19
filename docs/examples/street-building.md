# Street air drawn into a building

**Goal.** A small street canyon carries traffic pollution. Next to it stands a three-zone
building with a mechanical ventilation system, drawing its outdoor air from the street. How
quickly does the pollution show up indoors, and how much of the street level reaches each
zone? Two independent models, a street network and a CONTAM building, are coupled: the
street's concentration becomes the building's outdoor concentration, and the building's
infiltration feeds back on the street. Neither model is rebuilt; `union` only declares which
values they exchange. See [Coupling models](../guide/coupling.md) for the general recipe.

The street steps once per hour and the building takes six sub-steps in each hour. The coupled
step repeats until the interface values stop changing.

```python
import math
import torch
from noodl.apps.building_physics import project_to_model, read_prj
from noodl.apps.street_aq import Street, StreetNetwork, build_model as build_street, street_index
from noodl.couple import (
    CONCENTRATION_TO_MASS_FRACTION, MATH_RAD_TO_COMPASS_DEG, DriverAlias, ValueLink, union,
)

F64 = torch.float64

# A street junction layout: three streets meeting at n1.
x = {"n0": 0.0, "n1": 30.0, "n2": 60.0, "n3": 30.0}
y = {"n0": 40.0, "n1": 30.0, "n2": 40.0, "n3": 0.0}
spec = [("r1", "n0", "n1"), ("r2", "n1", "n2"), ("r3", "n3", "n1")]
net = StreetNetwork(
    streets=[Street(n, a, b, math.hypot(x[b] - x[a], y[b] - y[a]), 2.0, 3.0)
             for n, a, b in spec],
    x=x, y=y,
)
street_model, street_state, street_drivers = build_street(net, species=("nox",), z_ref=10.0)
graph = street_model.net
emissions = torch.zeros(graph.n, dtype=F64)
for s in net.streets:
    emissions[graph.node_index(s.name)] = 1.0e-7                 # kg/s per street
street_drivers = {
    **street_drivers,
    "street.x_boundary": torch.full((1,), 2.0e-8, dtype=F64),    # background, kg/m3
    "street.sources": emissions,
    "U_ref": torch.tensor(2.5, dtype=F64),                       # wind speed, m/s
    "theta_w": torch.tensor(0.0, dtype=F64),
    "h_abl": torch.tensor(1200.0, dtype=F64),
}
street_state = {"street.x": torch.zeros(3, dtype=F64)}

building = project_to_model(read_prj("valThreeZonesWthCtm-UseApi.prj"))

link = ValueLink(                       # street r2 -> the building's outdoor concentration
    from_model="street", from_key="street.x", from_index=street_index(street_model)["r2"],
    to_model="building", to_key="species.x_boundary", to_index=0,
    convert=CONCENTRATION_TO_MASS_FRACTION, two_way=True,
)
wind = [                                # one weather, two models
    DriverAlias(source=("street", "U_ref"), targets=(("building", "V_met", None),)),
    DriverAlias(source=("street", "theta_w"),
                targets=(("building", "theta_w", MATH_RAD_TO_COMPASS_DEG),)),
]
city, state, drivers = union(
    {"street": (street_model, street_state, street_drivers), "building": building},
    shared=[link, *wind], substeps={"building": 6}, iterate_rtol=1e-8, iterate_max=50,
)

outdoor = float(street_state["street.x"][1])
for hour in range(1, 7):
    state = city.step(state, drivers, dt=3600.0)
    street_c = float(state["street"]["street.x"][1])               # kg/m3 in street r2
    zones = state["building"]["species.x"][:, 0]                    # kg/kg in each zone
    ratio = [round(float(z) * 1.2 / street_c, 3) for z in zones]    # indoor / street
    print(f"hour {hour}: indoor/street ratio per zone {ratio}")
```

```text
hour 1: indoor/street ratio per zone [0.367, 0.049, 0.003]
hour 2: indoor/street ratio per zone [0.597, 0.146, 0.018]
hour 3: indoor/street ratio per zone [0.744, 0.26, 0.046]
hour 4: indoor/street ratio per zone [0.837, 0.373, 0.087]
hour 5: indoor/street ratio per zone [0.896, 0.478, 0.138]
hour 6: indoor/street ratio per zone [0.933, 0.57, 0.196]
```

![Indoor to street concentration ratio of the three zones over six hours](../assets/examples/street-building.svg#only-light)
![Indoor to street concentration ratio of the three zones over six hours](../assets/examples/street-building-dark.svg#only-dark)

The street concentration is nearly steady after the first hour, but the building is a slow
reservoir, and its zones differ a great deal. After six hours one zone has reached 93 % of the
street level, the next 57 % and the last 20 %, because the outdoor air reaches the rear zones
only through the zones in front of them and the flows between.

**Things to try.** Batch the pair over many wind speeds by giving every street driver and
every building state a leading dimension; the street-building benchmark in the repository does
this for up to a hundred copies. Remove `two_way=True` to couple one way and compare the street
concentration. See [Coupling models](../guide/coupling.md) for the coupling modes and their
convergence diagnostics.
