# Unsteady above-roof puffs

How to model a short release, or any cloud that passes in minutes, with Gaussian puffs above the roofs. `build_model(..., puffs=PuffOptions(...))` replaces the steady above-roof relation of [Above-roof concentration](above-roof.md) with puffs fed by the streets and feeding back into them. The streets stay one well-mixed box each, stepped by `Model.step`; what changes is who writes their boundary `C_ext`. The model is described on the [physics page](physics.md#unsteady-puffs-above-the-roofs), and the cost measurements are in the [design note](../../project/design/street_puffs.md).

## A short release above the roofs

```python
import math
import torch
from noodl.apps.street_aq import (
    PuffOptions, PuffState, build_model, mass_in_streets, pending_roof_mass, puff_mass,
    twelve_street_grid,
)

DT = torch.float64

net, names = twelve_street_grid()               # 12 streets, L = 100 m, W = H = 20 m
model, state, drivers = build_model(net, background="per_street", puffs=PuffOptions(
    sigma_theta=math.radians(6.0),              # wind-direction spread, rad
    n_releases=1,                               # one explicit above-roof release
))
c_bg = torch.zeros(12, dtype=DT)                # background above each roof, kg/m3
drivers = {
    **drivers,
    "street.x_boundary": c_bg,
    "u_star": torch.tensor(0.5, dtype=DT),      # friction velocity, m/s
    "theta_w": torch.tensor(0.3, dtype=DT),     # rad CCW from east, blowing TOWARD
    "h_abl": torch.tensor(800.0, dtype=DT),     # boundary-layer depth, m
    "puff.release_xy": torch.tensor([[-150.0, 20.0]], dtype=DT),  # m
    "puff.release_z": torch.tensor([30.0], dtype=DT),             # release height, m
    "puff.release_rate": torch.tensor([3e-3], dtype=DT),          # kg/s
    "puff.release_start": torch.tensor([12.0], dtype=DT),         # s
    "puff.release_duration": torch.tensor([17.0], dtype=DT),      # s
}

dt, c_ext, x = 5.0, [], []
for k in range(80):                             # 400 s
    state = model.step(state, drivers, dt, t=dt * k)
    c_ext.append(state["puff.c_ext"])           # C_ext over step k, kg/m3
    x.append(state["street.x"])                 # street concentration at its end, kg/m3
c_ext, x = torch.stack(c_ext), torch.stack(x)

dose = x.sum(0) * dt                            # kg s/m3
s = int(dose.argmax())
print(f"most exposed street {names[s]}: dose {float(dose[s]):.3g} kg s/m3")
print(f"C_ext first above zero at t = {int((c_ext[:, s] > 0).nonzero()[0]) * dt:.0f} s, "
      f"peaks at t = {int(c_ext[:, s].argmax()) * dt:.0f} s")
print(f"street concentration peaks at t = {(int(x[:, s].argmax()) + 1) * dt:.0f} s")

# the mass budget: streets + puffs + left + dropped - drawn down + not yet emitted
puff = PuffState.from_state(state)
emitted, drawn = pending_roof_mass(model, state, c_bg)
held = (mass_in_streets(model, state, c_bg) + puff_mass(puff) + puff.mass_out
        + puff.mass_dropped - puff.mass_drawn_down + emitted - drawn)
print(f"released {3e-3 * 17.0:.6f} kg, accounted for {float(held[0]):.6f} kg")
```

```text
most exposed street 6: dose 3.93e-06 kg s/m3
C_ext first above zero at t = 35 s, peaks at t = 60 s
street concentration peaks at t = 85 s
released 0.051000 kg, accounted for 0.051000 kg
```

The release is 50 m west of the grid's western street ends and 10 m above the roofs. Its cloud
reaches the streets after the travel time and passes; `C_ext` above the most exposed street
peaks before the street itself, which then flushes at its own exchange rate. A steady run
would have spread the same release over the whole hour.

A release inside a street is a source on the street layer: any `"street.sources"` history
the caller writes, or the `PulseSource` closure, which adds a top-hat release
(`"pulse.rate"` kg/s from `"pulse.start"` for `"pulse.duration"` s, shared over the streets
by `"pulse.weights"`) whose start and duration are differentiable even inside a step. It
runs before the flow closure, and the puff closure is bound to the new model:

```py
from noodl.apps.street_aq.puff import PulseSource
from noodl.model import Model

pulsed = Model(model.net, model.layers, closures=[PulseSource(model.net, names),
                                                  *model.closures])
model.closures[-1].bind(pulsed)                 # the puff closure reads its wiring
```

## Options and drivers

`PuffOptions`, all keyword arguments:

| Option | Default | Unit | What it sets |
|---|---|---|---|
| `h_canopy`, `z0`, `d` | 20, 1, 13 | m | The tables' canopy (reflection) height $H_R$, roughness length and displacement height, as `plume_table`'s. |
| `sigma_theta` | `None` | rad | The tables' wind-direction spread; `None` reads the `"sigma_theta"` driver. |
| `junction_source` | `"upward"` | | `"upward"` or `"net"`, as for `street_steady_with_plume`; under `"upward"` the puff material that incoming vents carry down into the streets is counted in `mass_drawn_down`. |
| `along_wind` | `"sigma_y"` | | The along-wind profile (table on the [physics page](physics.md#puffs-and-their-mass)). |
| `capacity` | 4096 | puffs | Slots per batch instance. When they are full the faintest live puffs are evicted into `mass_dropped` and `PuffClosure.last["evicted"]` counts them; raise the capacity until that count is zero (a 577-street city case peaks at 6350 live puffs and needs 8192). |
| `puff_dt` | `None` | s | The emission interval. `None` gives one puff per source per street step. A number (a multiple of the street step) accumulates each source's mass over the interval into one puff whose slug spans it. It saves capacity, not time: the work of evaluating `C_ext` follows the area the cloud covers, not the number of puffs. Within an interval the newest material is pending rather than a puff, so `C_ext` near the sources lags by up to one interval and short releases are smeared over it. On `twelve_street_grid` with 30 s intervals and 5 s steps a steady `C_ext` flickers by up to 0.74 of its peak each interval (street concentrations stay within 3 %), and a 17 s release's peak street concentration is off by 0.38 of it and its dose by 2.5 %. |
| `drop_below` | 1e-3 | | The faint threshold, relative to the brightest puff of the instance by a cheap peak estimate; dropped mass goes to `mass_dropped`. Against 1e-6 it changes street concentrations and `C_ext` by about 1e-4 of their peak. |
| `cutoff_sigma` | 4 | standard deviations | The profiles' cut-off, as the steady kernel's. |
| `x_max` | 700 | m | The steady kernel's downwind cut-off. Under `retire="distance"` it is the tables' extent, and a puff retires, into `mass_out`, when its centre passes it, while the near end of its slug may still be inside, so receptors within half a slug of `x_max` see a small step in `C_ext`. Under `"domain"` the tables reach the network's diagonal plus `x_max`. |
| `retire` | `"distance"` | | How a puff leaves, into `mass_out`: `"distance"` past `x_max` from its source, as the steady kernel; `"domain"` once the whole network lies upwind of its rear edge ([the physics page](physics.md#puffs-and-their-mass)), for releases whose cloud crosses the network. |
| `n_releases` | 0 | | The number of explicit above-roof releases. |
| `shear` | `False` | | Wind-shear stretching along the wind (Sykes and Henn, 1995; [physics](physics.md#wind-shear-stretching)); needs `along_wind="sigma_y"` (with `"none"` it is refused by name). |
| `table_kwargs` | `{}` | | The empirical inputs passed on to `puff_tables` (`theta_star`, `temperature`, `tau_y`, `tau_z`, `centre_k`, the floors); a `"centre_c"` entry (m) is the centre-law constant of the street and junction sources' table. |

The drivers the closure reads, one value per batch instance. Under `meteo="per_street"`
the meteorology carries a street axis whose values must all be equal, since the tables
take one meteorology per instance; a gradient with respect to such a driver reaches its
first entry only.

| Driver | Unit | What it is |
|---|---|---|
| `"street.x_boundary"` | kg/m³, `(..., n_streets)` | The background $C_{\text{bg}}$ above each street; the closure writes `C_ext` over it. Needs `background="per_street"`. |
| `"u_star"`, `"theta_w"`, `"h_abl"` | m/s, rad, m | Friction velocity, wind direction (counter-clockwise from east, blowing toward) and boundary-layer depth. Required. |
| `"lmo"` | m | The Obukhov length; neutral when absent. |
| `"sigma_theta"` | rad | Only when `PuffOptions.sigma_theta` is `None`. |
| `"puff.release_xy"` | m, `(B, R, 2)` or `(R, 2)` | The releases' horizontal positions. |
| `"puff.release_z"` | m, `(B, R)` or `(R,)` | Their heights. Each release has its own table, whose centre law starts at that height. |
| `"puff.release_rate"`, `"puff.release_start"`, `"puff.release_duration"` | kg/s, s, s; `(B, R)` or `(R,)` | Each release's top-hat history. Needs `Model.step(t=...)`. |

The tables are rebuilt only when one of their drivers changes (in value, or as a different
tensor), so under constant meteorology a run builds them once. A wind direction that
changes between steps turns every puff's further path.

`build_model(..., puffs=...)` needs `background="per_street"`, `scheme="backward_euler"`, one
species and no chemistry. The closure also refuses, by name, a model coupled other than
ping-pong, a sub-stepped street layer and a removal on it: a puff's mass is what one
backward-Euler step without sinks moved.

## How a puff leaves

How a puff leaves is set by `retire`:

- `"distance"` (the default) retires a puff once it is `x_max` (700 m) from its source.
  That is the steady kernel's downwind cut-off, so a continuous source reproduces
  `street_kernel`.
- `"domain"` retires a puff once the whole network lies upwind of its rear edge: its most
  upwind point, less half its slug and `cutoff_sigma` along-wind spreads. Use it for a
  release, whose cloud matters wherever it is in the network. On a 1.5 km line of streets
  a release under `"distance"` vanishes 700 m from where it was emitted; under `"domain"`
  it reaches the last street at about ten times the concentration. The tables then reach
  the network's diagonal plus `x_max`, and beyond their end a puff keeps its last spread.
  In a network smaller than `x_max`, puffs leave sooner than under `"distance"`.

The decision uses the current wind, so a puff retired just before the wind turns back is
not revived. Under either rule, puffs fainter than `drop_below` times the brightest puff
are dropped, and so are the faintest when the slots run out.

## Resuming a run

A query (`model.current_flows`, `model.residuals`, ...) evaluates `C_ext` from the puffs
without advancing them. `PuffClosure.last` holds the latest step's live and evicted puffs
per instance, and whether the tables were rebuilt.

The closure caches the tables and the field across steps and rebuilds them when the
meteorology, the release heights or the wind direction change. A driver that requires
grad matches by identity and value, one that does not by value alone, so new tensors of
the same value every step cost no rebuild. The cache holds the autograd graph of the run
that built it, so it is dropped at the start of every run (a state with no pending step)
and when a run resumes from a state the closure did not produce while the cache holds a
graph: to continue a run after a backward (truncated backpropagation through time),
detach every tensor of the state (`"puff.time"` included: the closure recognises a resume
by that tensor's identity) and step on with the same driver tensors, and the gradient is that of a
fresh closure resumed from the detached state. `PuffClosure.reset()` drops the cache by
hand, to continue a run after a backward in any other way (a driver changed in place, a
state kept with its graph and stepped again).

## The mass budget

With every mass in excess of the background, after every step:

```
sum_s V_s (x_s - C_bg,s) + sum_live m + mass_out + mass_dropped - mass_drawn_down
    + (mass the next step will emit) - (mass it will draw down)
  = sum over steps of (street sources * dt + release mass)
```

`mass_in_streets` gives the first term, `puff_mass` the second, the `PuffState` counters
the next three, and `pending_roof_mass` the last step's roof and junction mass not yet
emitted (with `puff_dt`, also the mass accumulated towards the next emission). The last
step's release mass, also emitted only by the next step, is not in it: add
`pulse_mass(t, dt, start, duration, rate)` of the last step. The identity
holds to 1e-12 relative at every step: with puffs evicted for capacity, with
the wind turning between steps and under both junction sources. It is exact for a
background constant in time (each emission uses the current step's background).

## Steady limits

Under constant meteorology and emissions the puffs of a continuous source settle to a steady `C_ext`, and the model is built so that this limit is the steady kernel.

- With `along_wind="none"` the steady puff train of a unit source at every street and junction reproduces `street_kernel` and `junction_kernel` to 1e-12 at every step from 10 s down to 0.625 s in a neutral hour, and to the same bound in a weakly unstable and a weakly stable hour; the coupled run reaches `street_steady_with_plume`'s solution to the run's own stopping tolerance. These checks do not see the one-step emission lag: a receptor closer to a source than half a step's travel would.
- With `along_wind="sigma_y"` the limit is the kernel smoothed along the wind, the along-wind diffusion the steady Gaussian plume neglects. Against the plain kernel the difference converges at first order in the step to a floor of about 4.7 % of the largest kernel entry.

The measurements are on the [verification page](../../reference/verification/street.md#unsteady-puffs).

## Gradients and cost

A puff's mass is linear in the street state and the fluxes, and a release's rate, start,
duration, position and height, the meteorology (`u*`, `h_abl`, `L`, `sigma_theta`, through
the tables) and the wind direction are all differentiable. Gradients of the coupled run agree with central differences in a canopy pulse's rate and start, a release's rate, start,
duration and x position, `u*` and the wind direction; the tables' gradients in `u*`, `L`
and the release height are checked to be finite and non-zero, and the steady table's in
`h_abl` and `sigma_theta` against central differences; the release height, `h_abl` and
`sigma_theta` are not checked by central differences through a puff run. A start time inside a step is
differentiable: a release's mass in a step is the overlap of its window with the step,
piecewise linear. Masks, drop decisions and evictions are taken on detached values, so the
gradient is that of the puffs kept. The cut-offs are hard, as in the steady kernel, and a
faint-drop decision that flips under a finite-difference step is a kink too
(`drop_below=0` removes it). Second derivatives are refused by name. Under autograd each
step's puffs are checkpointed: the graph keeps each step's puff state, and the backward
recomputes the step.


A batch of releases is a leading batch dimension on every driver and on the puffs; a batch of $N$ releases (different streets, start times, `u*`) equals $N$ separate runs to rounding.

The cost of an evaluation follows the area the cloud covers, not the number of puffs, and the street model's own canyon-wind solve is recomputed every step. Measurements at the scale of a 577-street city, and where the time goes, are in the [design note](../../project/design/street_puffs.md#410-cost-at-paris-scale).
