# Hydrogen sulfide in a sewer under a load surge

**Goal.** A gravity sewer tree carries domestic flow. For a few hours the flow and the organic
load rise by a quarter, and then fall back. Sulfide forms in the sewage, transfers to the air
above it, and collects in the headspace of each manhole. How high does the headspace
hydrogen sulfide get, and where? The model couples four things on one network: the hydraulics,
the air moving in the pipes, the dissolved species in the water and the gas-phase species in
the air.

```python
import math
import torch
from noodl.apps.sewer import build_model, tree_steady, to_ppm

model, state, drivers = build_model(tree_steady())
base = drivers["inflow"].clone()
manholes = model.refs.air_quality.concentration.labels

dt = 300.0                                         # s
peak = torch.zeros(len(manholes), dtype=torch.float64)
history = []
for step in range(60):                             # 5 h
    hours = step * dt / 3600.0
    factor = 1.0 + 0.25 * math.sin(2.0 * math.pi * hours / 8.0)   # an 8 h surge cycle
    step_drivers = {
        **drivers,
        "inflow": base * factor,                                    # m3/s
        "bod_in": torch.where(base > 0, 0.30 * factor, 0.0 * base),  # kg/m3 organic load
    }
    state = model.step(state, step_drivers, dt)
    ppm = to_ppm(state["air_quality.x"], drivers["T_head"])
    history.append(ppm)
    peak = torch.maximum(peak, ppm)

print("manholes:", list(manholes))
print("peak headspace H2S (ppm):", [round(float(p), 2) for p in peak])
print("after 5 h (ppm):", [round(float(p), 2) for p in history[-1]])
```

```text
manholes: ['J1', 'J2', 'J5', 'J3', 'J4']
peak headspace H2S (ppm): [6.13, 4.63, 8.96, 9.17, 12.83]
after 5 h (ppm): [5.8, 4.14, 8.04, 8.19, 9.11]
```

![Headspace hydrogen sulfide at each manhole over five hours](../assets/examples/sewer-sulfide.svg#only-light)
![Headspace hydrogen sulfide at each manhole over five hours](../assets/examples/sewer-sulfide-dark.svg#only-dark)

Headspace concentrations build up from clean air as the sulfide forms and transfers, and
then follow the load. The manholes differ by a factor of two to three, because each collects the
sulfide and the air of a different part of the tree. The step of 300 s is the largest that stays stable on this
tree with the default settings; a longer step makes the explicit reaction step diverge, which
is a reason to look at a run's trajectory rather than only its end state.

**Things to try.** Give `build_model` `storage=True` to let the manholes store water and
surcharge. Vary the load in a batch of instances (see [Batches](../guide/batches.md)) to
find which pipe's ventilation matters most for the peak. See
[Sewers](../applications/sewer/index.md) for the physics and the SWMM reader.
