# Kinematic-wave and steady runs

How to build a sewer network, turn it into a model and run it under `routing="steady"` or
`routing="kinematic_wave"`. Both routings solve Manning's equation on a dendritic network (see
[Physics](physics.md)); for backwater, loops and surcharge use the
[dynamic wave](dynamic-wave.md). The first example is on the [overview](index.md).

## Read a network from a SWMM file

A model can be read from a SWMM `.inp` file (see [SWMM `.inp`](swmm.md)):

```python
from noodl.apps.sewer import read_swmm_inp, build_model

net, inflows, pollutants = read_swmm_inp("tree_steady.inp")
model, state, drivers = build_model(net)
```

The file's `FLOW_UNITS` (CFS, GPM, MGD, CMS, LPS or MLD) are converted to noodl's SI as it is
read; see the Units paragraph of [SWMM `.inp`](swmm.md). A network with no links (no conduits,
orifices, weirs or outlets) is refused by `build_model` by name.

For a single step rather than a steady state:

```python
new = model.step(state, drivers, 60.0)
residuals = model.residuals(new, drivers)     # per-layer nodal balance
```

## Building a network

A network is built from `Manhole`, `Pipe` and `Outfall` objects (plus `storage_unit`, `Orifice`,
`Weir` and `Outlet` for the dynamic wave) collected in a `SewerNetwork`; their fields are in the
[catalogue](../../reference/catalogue/sewer.md). The conventions the catalogue does not spell out:

- All lengths, elevations and depths are in m, areas in m², flows in m³/s.
- A manhole's `surface_area` defaults to SWMM's `MIN_SURFAREA`, 1.167 m² (a 4 ft shaft).
  `surcharge_depth`, `initial_depth` and `storage` are dynamic-wave physics (see
  [Dynamic wave](dynamic-wave.md)).
- Pipe offsets (`inlet_offset`, `outlet_offset`, m) are the invert heights above the end nodes'
  inverts.
- `storage_unit(...)` makes a `Manhole` whose `storage` is a surface-area curve and whose ground
  is `invert + max_depth`. The curve is `("functional", a, b, c)` for $A = a y^b + c$,
  `("tabular", depths, areas)`, or `("cylindrical", L, W)`, `("conical", L, W, Z)`,
  `("paraboloid", L, W, H)`, `("pyramidal", L, W, Z)` (m and m²). Dynamic wave only.
- An `Outfall` is the root of a tree. Its `kind` is `"free"`, `"normal"`, `"fixed"` (`stage`, m,
  required), `"tidal"` (`tide`, a SWMM `TIDAL` curve as `(hour, stage m)` pairs, required) or
  `"timeseries"` (the stage comes from `stage_series`, the file's series as an
  `inflows.TimeSeries` in s and m held at both ends, or else from the `outfall_stage` driver;
  `series` is the SWMM series' name). Only dynamic-wave routing reads the stage.
- An `Orifice` runs from `u` to `v`: `orientation` is `"side"` or `"bottom"`, `shape` is
  `"circular"` (`height` its diameter, m) or `"rectangular"` (`height`, `width`, m); its crest is
  `crest_height` m above `u`'s invert and Cd is dimensionless. Dynamic wave only.
- A `Weir`'s `kind` is `"transverse"`, `"sideflow"` (opening `width` m long), `"v_notch"` (`width`
  the top width) or `"trapezoidal"` (`width` the bottom width, `side_slopes` horizontal per
  vertical); Cd is in SWMM's SI units (m^0.5/s). Dynamic wave only.
- An `Outlet`'s `rating` is `("functional", a, b)` for $q = a h^b$ or `("tabular", heads, flows)`
  (m, m³/s); $h$ is the depth above the crest, or the head across it with `head_difference`.
  Dynamic wave only.
- `SewerNetwork.routing` is `"steady"`, `"kinematic_wave"` or `"dynamic_wave"`; SWMM's
  `STEADY`, `KINWAVE` and `DYNWAVE` are accepted and normalised.

`validate()` refuses, by name: duplicate node or pipe names; a pipe naming an unknown node;
non-positive slope, diameter, length or roughness; **a manhole with anything other than exactly
one outgoing pipe**; an outfall with an outgoing pipe; and a manhole whose component never
reaches an outfall — which also catches cycles. With `routing="dynamic_wave"` these tree
rules are replaced: loops and flat pipes are allowed, the network needs at least one
outfall (SWMM's `ERR_NO_OUTLETS`, flowrout.c:325; nothing is asked of each component, so a
junction no link reaches, or a group of nodes without an outfall, is read and routed by its
own continuity, filling from its inflow and flooding at its full depth), and each outfall
ends exactly one link (a conduit or a regulator). Orifices, weirs
and outlets are refused by name for the other routings, and malformed ones (an unknown
node, a duplicate link name, a bad shape, a negative crest or coefficient, an unsorted
rating) for every routing.

That one-outgoing-pipe rule is the dendritic constraint, and it is what makes the closed-form
flow solve valid.

## `build_model`

```py
model, state, drivers = build_model(
    net,
    storage=False,          # implicit-Euler manhole storage sweep
    air=True,               # the headspace layer
    quality=True,           # the two quality layers
    species=("bod", "sulfide"),
    f_air=0.02, f_i=7.49e-4, c_s=1.0,
    leak_area=8e-4, leak_cd=0.6, fans=(),
    coupling="pingpong", scheme="backward_euler", dt_storage=None,
)
```

`species` are matched by name, never by position, so their order does not matter: `"bod"`
decays and, when `"sulfide"` is also present, sulfide is generated from it; any other name is
a passive tracer without reactions. `"sulfide"` without `"bod"` and repeated names are
refused. The air-quality layer and the water-to-air transfer exist only when `"sulfide"` is
among the species (the air-flow layer is built by `air=True` regardless); a selection
without `"bod"` has no reaction at all. Every species `s` has a lateral-inflow
concentration driver `"<s>_in"` (kg/m3, default 0), as in the dynamic-wave builder. A
single species (`species=("bod",)` or `("tracer",)`) is held with no species axis,
`"water_quality.x"` of shape `(manholes,)`, as `TransportLayer` holds any one-species
layer; BOD decays on it as it does beside other species, in both builders.

`air=False` gives the water-only model that SWMM can be compared with. `fans` names manholes carrying
a prescribed extraction (m³/s, positive out).

`build_model` dispatches on `net.routing`. A `"dynamic_wave"` network is built by
`build_dynamic_wave_model`, with `iterate_tol` (m), `iterate_flow_tol` (m³/s) and
`iterate_max` (passes) controlling the iteration; left out, they follow the file's
`HEAD_TOLERANCE` and `MAX_TRIALS` (see [Iteration and convergence](physics-dynamic-wave.md#iteration-and-convergence)).
For it, `air`, `quality`
and `storage` default to `False`; `quality=True` adds the water-quality layer of `species`
(see [Quality on the dynamic wave](headspace-sulfide.md#quality-on-the-dynamic-wave)); `air=True` builds the
headspace air on the solved conduits (see
[Headspace air on the dynamic wave](headspace-sulfide.md#headspace-air-on-the-dynamic-wave)) and takes `f_air`,
`f_i`, `c_s`, `leak_area`, `leak_cd` and `fans` as above; `storage=True`, a non-default
`coupling` or `dt_storage` (kinematic-wave options: the dynamic wave always iterates its passes
on its routing step) and a `geometry="analytic"` network (it always routes on SWMM's tabulated
circle) are refused, naming the option; `iterate_tol`, `iterate_flow_tol` and
`iterate_max` are refused for the other routings. Run it with `simulate_dynamic_wave` (see
[Dynamic wave](physics-dynamic-wave.md)), not `sewer_steady`. To tune the iteration itself
(`relaxation`, `acceleration`, `memory`), call
`build_dynamic_wave_model(net, ...)` from `noodl.apps.sewer` directly; it takes the same
dynamic-wave arguments as `build_model`.

`dt_storage` (optional) declares the interval the manhole storage sweep integrates over. When given,
a step at any other interval is refused by name. Omit it (the default) to follow the model's own
step interval via the `StepContext` the closure receives.

`initial_state(model, drivers=None)` builds the all-zero state, dispatching on each layer's
quantity. When `drivers` is given, it also evaluates the initial storage (by querying
`model.initial_capacities`), which a step with a changing capacity requires to be in the step-start
state. Drivers are required for initial_state whenever the model has a quality layer, because the
hydraulics closure writes those layers' capacity from the flows and the initial storage must be
evaluated from the drivers; this holds with or without `storage=True`.

`sewer_steady(model, state, drivers, *, reaction=None, dt=60.0, max_iter=500, tol=1e-12)` steps
repeatedly until every transport layer stops changing (`Model.fixed_point`: gradients are those
of the fixed point, by the implicit adjoint, with backward memory of one step). You need it
rather than `Model.steady` whenever quality is on, because `Model.steady` never applies
reactions.

`pipe_table(model, state, drivers)` resolves the hydraulics without stepping any layer and
reports `sewer.q`, `sewer.h`, `sewer.v` and the rest.
