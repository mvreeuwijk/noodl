# Water distribution: build and solve a network

How to describe a network, build the model, solve it at steady state, change inputs by name and read the results. Time stepping is on [Extended-period simulation](extended-period.md). The overview is [Water distribution](index.md); every class and function is listed in the [catalogue](../../reference/catalogue/water.md).

**Units.** Head and elevation m, flow and demand m³/s, pressure head m (with a helper to kPa).

## The objects

| Object | Fields |
|---|---|
| `Junction(name, elevation, demand=0.0, pattern=None)` | A demand node. |
| `Reservoir(name, head, pattern=None)` | Fixed-head boundary. |
| `Tank(name, elevation, init_level, min_level, max_level, diameter, volume_curve=None, overflow=False, min_volume=0.0)` | A cylinder of `diameter`, or the volume curve `volume_curve` (a `curves` key, level m against volume m³); `overflow` lets it spill when full. |
| `WaterPipe(name, u, v, length, diameter, roughness, minor_loss=0.0, status="OPEN")` | A full pipe. |
| `Pump(name, u, v, curve, power=None, speed=1.0)` | Edge oriented **suction → discharge**; on a head `curve` (a `curves` key) or at a constant `power` (kW), at relative `speed`. |
| `Valve(name, u, v, diameter, kind, setting, minor_loss=0.0, curve=None)` | TCV, FCV, PBV, GPV, PRV or PSV (`curve`: a GPV's head-loss curve). |
| `WaterOptions(demand_model="DDA", minimum_pressure=0.0, required_pressure=0.1, pressure_exponent=0.5, ...)` | Carried from the file's `[OPTIONS]`. |

`WaterNetwork` collects them plus `curves`, `patterns`, `controls` and the timing options.
`nodes()` returns junctions, then reservoirs, then tanks; `links()` returns pipes, pumps, TCVs,
FCVs, PBVs, GPVs, PRVs, PSVs in that order; `validate()` refuses anything out of scope by name.

## `build_model`

```py
model, state, drivers = build_model(
    net,
    headloss=None,      # 'H-W' (default) or 'D-W'; None takes the network's own
    pda=None,           # pressure-driven demand; None takes net.options
    p_min=None, p_req=None, exponent=None,
    quality=None,       # bulk decay coefficient, 1/day; adds a quality TransportLayer
    coupling="pingpong",
    dt=None,            # overrides net.hydraulic_timestep for the tank closure
    friction="composite",  # D-W friction law: 'composite' (default) or 'colebrook'
    pipe_form=None,     # 'flow' (resistance form; the default) or 'potential'
    tank_scheme="forward_euler",
    valve_smoothing=0.0,  # m: smooth the PRV/PSV switch over a band of this width
)
```

`pipe_form` is how the pipes enter the solve. In the **resistance form** (`"flow"`, the default
except with `friction="colebrook"`) each pipe is `dp = h(q)` (`FlowControlledPipe`) with its
flow an unknown, solved from EPANET's own first iterate (every pipe
at 1 ft/s). In the **potential form** (`"potential"`) each pipe's flow is the inverted law
$q = g(dp)$, with no flow unknowns. Both give the same solution to round-off; they differ in how
Newton gets there. The potential form's slope grows without bound as a pipe's flow goes to
zero, so on Net1's 24 h run the step where a control closes the pump and leaves pipe 10
dead-ended needs 136 Newton iterations; the resistance form's slope goes to EPANET's `RQtol`
floor instead, and no step needs more than 6. The resistance form is therefore the
faster of the two on a 24 h run.

`water_steady(model, state, drivers, **solve_kwargs)` wraps `model.steady` with defaults
`atol=rtol=1e-11`, `max_iter=200` (both chosen empirically, the iteration cap for the
potential form; caller overrides win), starts a cold start of the resistance form (every pipe
flow exactly zero) from `model.initial_flows`, and the pumps from their design flows (EPANET's
own first iterate), then checks
every converged power-function pump's flow against its own `q_max()` at its speed and raises
naming the pump and batch instance if it is exceeded — rather than silently accepting an
extrapolated point off the end of the curve. (A segmented curve's last segment and a
constant-power pump are extended without a limit, as EPANET extends them.) A converged pump
carrying flow backwards is refused by name too. The flows are seeded, as EPANET seeds them,
rather than left to the layer's linear guess or its fallback ladder
([When Newton fails](../../concepts/solvers.md#when-newton-fails-the-fallback-ladder)). From the
seeded flows Newton needs 5 iterations on Net1 and the two-loop network. The linear guess
(`linear_init`, and the ladder's rungs) starts from the same flows: every implicit water law
gives EPANET's first iterate as its start flow (`start_flow`): a pipe or a valve 1 ft/s
through its section, a pump its design flow times its speed (the middle point of a
three-point curve, the point of a one-point curve, the mean of a custom curve's first and
last flows, 1 cfs at constant power), a closed link or a stopped pump zero. On Net1 that
guess is within a factor of two of the converged flows; taken at zero flow instead, where
every resistance-form pipe sits on its `RQtol` slope floor, it carried the 52 m between the
reservoir and the tank as 1.9e7 m³/s.


## Change an input by name

Give an input by name over the builder's own values. A demand is a negative source (m³/s
withdrawn; `demand=` does the sign for you), and junctions not named keep their demand. A
pressure-driven model (`pda=True`) refuses both `demand=` and a `water.sources` value, since its
demands are withdrawals set at build time.

```python
from noodl.apps.water import build_model, initial_drivers, twoloop, water_steady

model, state, drivers = build_model(twoloop())
water = model.refs.water
more = initial_drivers(model, demand={"J2": 0.012})      # m3/s withdrawn at J2
final = water_steady(model, state, more)
print(round(float(water.head.named(final[water.head])["J2"]), 6))
```

```text
48.641165
```

In the dictionaries the heads are `"water.phi"` (full node order) and the flows `"water.q"`
(link order); a tank's level is its head in `"water.phi"` less its bottom (`tank_levels`),
and the status and setting that the controls carry across steps are the state keys
`"water.link_status"` and `"water.link_setting"` (per link). Both are set by tank and pump name
with `initial_state(model, values=...)`, the levels under the key `"water.tank_level"`, which
`initial_state` reads and writes into the tank heads (it is no state key). See
[The workflow](../../guide/workflow.md).

### The steady tracer

On a model built with `quality=`, the quality sink is the withdrawal of the current hydraulic
solution (a scaled or patterned `water.sources`, or the delivered demand under `pda=True`), so a
tracer entering at the reservoir stays at its concentration at every demand level.

- A junction with a positive `water.sources` (an injection of water) is no sink: the injected
  water carries no tracer of its own and dilutes what arrives through the pipes, and any tracer
  injected with it is `quality.sources` (kg/s). On R → A → B with the reservoir at 100, an
  injection of 0.005 m³/s and 0.1 kg/s at A and a withdrawal of 0.01 m³/s at B, both junctions
  read (0.005 × 100 + 0.1) / 0.01 = 60, and the tracer withdrawn equals the tracer supplied.
- A stagnant node (no inflow and no outflow, for example a dead end whose delivered demand is
  zero under `pda=True` because the pressure is below the minimum) has no steady concentration
  without decay: `water_steady` raises `UndefinedSteadyState` naming the node and the cause, as
  the transport layer does for any stagnant node (a steady solve never invents a value). With a
  bulk decay (`quality > 0`) the node reads 0; or step the quality in time from an initial
  concentration.
- Quality advects through every hydraulic edge kind: pipes, pumps and valves.

This steady tracer is a layer of the model and differs from EPANET's water quality, which
`water_run` runs; see [Water quality](quality.md).

## Read a network from a file

```python
from noodl.apps.water import read_epanet_inp, build_model, water_steady

net = read_epanet_inp("twoloop_si.inp")
model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)
```

`final["water.phi"]` is the head (m) at every node and `final["water.q"]` the flow (m³/s) in
every link. What the reader accepts and refuses is on [Importing EPANET files](epanet.md).
