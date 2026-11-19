# A day of pumping on EPANET Net1

**Goal.** Run EPANET's Example Network 1 for 24 hours, with its two simple controls that
switch the pump on and off at tank levels, and see how the tank responds to a lower and a higher
demand than the file's. The run follows EPANET 2.2's rule for the hydraulic step: it is shortened
to the next pattern change, report time, or moment a control fires, so the pump switches at the
right time rather than at the next fixed step.

```python
import torch
from noodl.apps.water import read_epanet_inp, build_model, water_run

net = read_epanet_inp("Net1.inp")
model, state, drivers = build_model(net)

levels = {}
for scale in (0.8, 1.0, 1.2):
    scaled = {**drivers, "water.sources": drivers["water.sources"] * scale}
    run = water_run(model, drivers=scaled, quality=False)
    levels[scale] = run.tank_level[:, 0]            # tank 2, m, at each report time
    pump = run.link_names.index("9")
    hours_off = int((run.status[:-1, pump] == 0).sum())
    print(f"demand x{scale}: final level {float(levels[scale][-1]):.3f} m, "
          f"pump closed for {hours_off} of {len(run.time) - 1} hours")
```

```text
demand x0.8: final level 39.596 m, pump closed for 11 of 24 hours
demand x1.0: final level 35.175 m, pump closed for 10 of 24 hours
demand x1.2: final level 35.755 m, pump closed for 7 of 24 hours
```

![Tank level over 24 hours for three demand levels](../assets/examples/water-day.svg#only-light)
![Tank level over 24 hours for three demand levels](../assets/examples/water-day-dark.svg#only-dark)

With the file's demand the tank fills until the pump's closing level and then drains until the
opening level. A higher demand drains the tank faster, so the pump is needed for more hours
of the day. Each demand level is a separate run here because each switches the pump at its own
times; a batched run shares one step sequence and is the better tool when the controls do not
fire (for example for a steady-state study over many demands). See
[Batches](../guide/batches.md).

**Things to try.** Read `run.pressure` for the junction pressures through the day, or `run.energy`
for the pump's energy use and cost. Ask for water quality by leaving `quality=None`: the file
asks for chlorine decay and the run then reports `run.node_quality`. See
[Run a model from a file](../get-started/run-from-file.md) and
[Water distribution](../applications/water/index.md).
