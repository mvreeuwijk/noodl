# Run a model from a file

This tutorial reads an EPANET network, solves it, and reads the results by name. EPANET's
`Net1.inp` is a small example with a reservoir, a tank, a pump, nine junctions and twelve pipes.
The same steps work for every application that has a file reader: see
[File formats](../reference/formats.md).

## Read the file

```python
from noodl.apps.water import read_epanet_inp

net = read_epanet_inp("Net1.inp")
print(len(net.junctions), "junctions,", len(net.pipes), "pipes,",
      len(net.pumps), "pump,", len(net.tanks), "tank")
```

```text
9 junctions, 12 pipes, 1 pump, 1 tank
```

`read_epanet_inp` converts everything to SI units and refuses, by name, anything it does not
support. The result is a plain description of the network, not yet a model.

## Build the model and run it

```python
from noodl.apps.water import build_model, water_steady

model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)
```

`build_model` returns the model, an initial state and the inputs (drivers) read from the file.
`water_steady` solves the hydraulics at the file's initial conditions.

## Read the results by name

Results come back as tensors in the model's internal order. `model.refs` turns them into
dictionaries keyed by the names in your file.

```python
water = model.refs.water
heads = water.head.named(final[water.head])     # m, by node name
flows = water.flow.named(final[water.flow])     # m3/s, by link name

print("node  elevation_m  head_m  pressure_m")
for junction in net.junctions[:5]:
    h = float(heads[junction.name])
    print(f"{junction.name:>4}  {junction.elevation:10.2f}  {h:6.2f}  {h - junction.elevation:9.2f}")
print("pipe 10 flow (L/s):", round(1000 * float(flows["10"]), 2))
```

```text
node  elevation_m  head_m  pressure_m
  10      216.41  306.13      89.72
  11      216.41  300.30      83.89
  12      213.36  295.68      82.32
  13      211.84  295.31      83.48
  21      213.36  296.13      82.77
pipe 10 flow (L/s): 117.74
```

| Node | Elevation (m) | Head (m) | Pressure head (m) |
|---|---|---|---|
| 10 | 216.41 | 306.13 | 89.72 |
| 11 | 216.41 | 300.30 | 83.89 |
| 12 | 213.36 | 295.68 | 82.32 |
| 13 | 211.84 | 295.31 | 83.48 |
| 21 | 213.36 | 296.13 | 82.77 |

These match EPANET 2.2 to about 1e-7 relative; see the
[water verification](../reference/verification/index.md).

Next: [fit a parameter to measurements](calibrate.md), or read the
[water distribution application](../applications/water/index.md) for extended-period runs,
water quality and every supported element.
