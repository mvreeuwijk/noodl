# Natural ventilation of two rooms

**Goal.** A heated room and an unheated room are joined by a doorway, and the unheated room
has a low and a high leak to the outside. The outdoor temperature follows a daily cycle. How
warm do the rooms get, and how much air does the doorway move as the stack effect rises and
falls? This is the smallest example in which airflow and temperature depend on each other: the
temperature sets the air density, the density sets the buoyancy-driven flow, and the flow
carries the heat.

The two physics (an air-pressure network and a thermal network) are stepped together with
`coupling="iterate"`, which repeats each step until the temperatures stop changing within a
tolerance. [Layers and how they are stepped together](../concepts/layers.md) explains the
other coupling mode, `"pingpong"`.

```python
import math
import torch
from noodl.apps.building_physics import (
    Zone, add_zone, add_large_opening, orifice_law_from_edges, build_model, initial_state,
)
from noodl.heads import Stack
from noodl.topology import Network

F64 = torch.float64
T_MEAN = 283.15                       # mean outdoor temperature, K

net = Network(dtype=F64)
net.add_node("ambient", z_ref=0.0)
add_zone(net, Zone("A", volume=60.0, T0=T_MEAN + 5.0))      # heated room
add_zone(net, Zone("B", volume=60.0, T0=T_MEAN + 2.0))      # unheated room
add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)   # the doorway
net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)   # low leak
net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)   # high leak

model = build_model(
    net, air_laws=[orifice_law_from_edges(net, "airpath")],
    heads=[Stack.from_network(net, "airpath")],
    coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50,
)
state = initial_state(model)
heater = torch.zeros(net.n, dtype=F64)
heater[net.node_index("A")] = 300.0                           # W in room A

dt = 600.0
for k in range(144):                                          # 24 h in 10 min steps
    t = (k + 1) * dt
    outdoor = T_MEAN + 6.0 * math.sin(2.0 * math.pi * (t - 9.0 * 3600.0) / 86400.0)
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "thermal.x_boundary": torch.tensor([outdoor], dtype=F64),
        "thermal.sources": heater,
    }
    state = model.step(state, drivers, dt)
    if (k + 1) % 36 == 0:
        a, b = (state["thermal.x"] - 273.15).tolist()
        door = abs(float(state["air.q"][0]))
        print(f"{t / 3600:4.0f} h  A {a:5.1f} C  B {b:5.1f} C  doorway {door:.2f} kg/s")
```

```text
   6 h  A  17.2 C  B  15.7 C  doorway 0.18 kg/s
  12 h  A  25.3 C  B  23.8 C  doorway 0.18 kg/s
  18 h  A  27.5 C  B  25.8 C  doorway 0.19 kg/s
  24 h  A  19.3 C  B  17.6 C  doorway 0.19 kg/s
```

![Temperatures of the two rooms and the doorway exchange over 24 hours](../assets/examples/natural-ventilation.svg#only-light)
![Temperatures of the two rooms and the doorway exchange over 24 hours](../assets/examples/natural-ventilation-dark.svg#only-dark)

The rooms follow the outdoor cycle with a lag, and room A stays warmer than room B throughout.
The doorway exchange is nearly constant because the heater keeps the stack effect between the
rooms almost the same all day.

**Things to try.** Change `heater[...]` to 0 and watch the rooms settle to the outdoor
temperature. Build the model with `coupling="pingpong"` and compare the doorway flow of the two modes at
a 1 h step; the single-pass mode is cheaper per step, and [Layers and how they are stepped
together](../concepts/layers.md) explains what it gives up. The two-room model comes from the building application; see
[Building airflow and heat](../applications/building/index.md) for the elements, and
[Time stepping and schedules](../guide/time-stepping.md) for how drivers change over time.
