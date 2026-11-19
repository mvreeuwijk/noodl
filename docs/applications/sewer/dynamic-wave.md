# Dynamic-wave runs

How to run a sewer under `routing="dynamic_wave"` (SWMM's `DYNWAVE`): where it differs from the
kinematic-wave path, what it covers, how the step is chosen, how outfalls and conduit options
are read. The equations are in [Dynamic-wave physics](physics-dynamic-wave.md). Storage units,
regulators, pumps, controls and cross sections are in
[Dynamic-wave structures and cross sections](dynamic-wave-structures.md); lateral inflows and
runoff from a file in [Lateral inflows, runoff and RDII](swmm-hydrology.md).

## Run it

Read a file (or build a `SewerNetwork` with `routing="dynamic_wave"`), build the model with
`build_model` and run it with `simulate_dynamic_wave`, not `sewer_steady`. The example below
is in the Conduit options section; `hist` holds the recorded history.

## Units

SWMM computes in feet with $g = 32.2$ ft/s² and Manning's 1.486, and the conduit coefficients
here are evaluated exactly that way (reusing `xsect_tables.LCF`, `QCF`, `PHI`). Outputs are
converted with SWMM's own factors: head in m is ft × `LCF`, flow in m³/s is cfs × `QCF`,
conveyance in m²/s is ft²/s × `QCF/LCF`, and an area in m² is ft² × `QCF/LCF`, which is the
area `kinwave_steady` reports and makes $A_s\,\Delta y = \tfrac12\Delta t\,N$ exact in m and
m³/s.

## Scope

Conduits of every SWMM cross section (the circle's tabulated geometry or any other shape, see [structures](dynamic-wave-structures.md#cross-sections-losses-transects-and-culverts))
with local losses, force mains, transects and culverts; the fixed routing step or
SWMM's variable step (`VARIABLE_STEP > 0`, see [Variable step](#variable-step-and-time-varying-inflow)); a cold start or SWMM's `InitDepth` and
`InitFlow` ([starting state](physics-dynamic-wave.md#starting-state-and-zero-fall)); constant, file-defined (time series, dry-weather flow and patterns, see
[Lateral inflows from a file](swmm-hydrology.md#lateral-inflows-from-a-swmm-file)) or driver-supplied lateral inflows; `FREE`, `NORMAL`, `FIXED`, `TIDAL` and `TIMESERIES` outfalls,
with flap gates;
parallel barrels, `MaxFlow`, conduit flap gates and conduit lengthening ([Conduit options](#conduit-options)); storage
units with `FUNCTIONAL`, `TABULAR` and the closed-shape area curves ([structures](dynamic-wave-structures.md#storage-units)); orifices
(side and bottom, circular and rectangular), transverse, sideflow, V-notch and trapezoidal
weirs, and outlets with a functional or tabular rating, with their flap gates ([structures](dynamic-wave-structures.md#orifices-weirs-and-outlets));
`INERTIAL_DAMPING` `NONE`/`PARTIAL`/`FULL` and every `NORMAL_FLOW_LIMITED` option.

## Variable step and time-varying inflow

Give `duration=` (s) instead of `steps=` to run a stretch of simulated time. With
`VARIABLE_STEP > 0` (a `DynamicWaveOptions.variable_step`, the Courant factor, 0 to 2) every
step is chosen as SWMM does (`dynwave_getRoutingStep`): the first step is `MINIMUM_STEP`
(`minimum_step`, 0.5 s by default, capped at the routing step and floored at 1 ms), and
each later one is the smallest of

- the routing step;
- each conduit's Courant step, `V / |q| * Fr / (1 + Fr) * VARIABLE_STEP` with `V` the
  conduit's volume, `q` its flow and `Fr` its Froude number (a dry, still or full closed
  conduit, `Fr <= 0.01`, does not limit it);
- each junction's depth-change step, the time to rise a quarter of its crown depth at the
  rate it rose over the last step (a dry junction, one at or above its crown, and a quasi-steady
  one do not limit it);

floored at the minimum step and truncated to a millisecond. The reader takes an absent
`VARIABLE_STEP` as SWMM's 0.75 and lowers `ROUTING_STEP` to `WET_STEP` (default 300 s) when
larger, as SWMM does; `DynamicWaveOptions()` built in code defaults to the fixed step (0). The last step is cut to the time
that remains (at least 1 ms), so the run ends at `duration`; this clipping applies to a fixed
step too. One call is one run: under a variable step a second call restarts with the minimum step and
with `t` from 0, as a new SWMM run does (continuing needs the previous step's state, which
the call does not carry). A variable step needs a single network (no batched `inflow`), and `dt=` is refused
with it, since the step is then chosen by the options.

An inflow function `inflow(t)` returns the `(n,)` lateral inflow of the step that starts at
`t` s. SWMM reads a time series at the start of the step (its own clock is 1 ms ahead,
`swmm5.c`'s `getDateTime`: a function that stands for a SWMM `[TIMESERIES]` should add
0.001 s), and the inflow enters the step's trapezoid with the end state, as a constant
inflow does. A function (or an `(steps, B, n)` series) in `simulate_dynamic_wave` is how
time variation enters; a SWMM file's own `[TIMESERIES]`, `[PATTERNS]`, `[INFLOWS]` and
`[DWF]` become such a function ([below](swmm-hydrology.md#lateral-inflows-from-a-swmm-file)). The step sequence and every
state of the variable-step tree, with and without a ramping inflow, match SWMM 5.2.4 to the
millisecond and to the targets of the other dynamic-wave comparisons. Because the step depends on the state
through switches, a variable-step run is not differentiable through its step sizes.

## Tidal and time-series outfalls

A `TIDAL` outfall (`Outfall(kind="tidal", tide=...)`, the `.inp`'s `[CURVES]` TIDAL curve of
`(hour of day, stage elevation)` pairs) and a `TIMESERIES` outfall are `FIXED` outfalls whose
stage changes in time: noodl applies the same stage rule as for `FIXED` (the stage above the
conduit's critical-depth elevation sets the outfall depth, otherwise the critical or normal
depth) to the stage of the step's end, because SWMM sets the outfall depth with its routing
clock already advanced (`routing.c:301-302`, `node.c:1445-1456`). That is later than
the step start at which a lateral inflow is read. A tidal stage is `tidal_stage(tide, t)`: the
curve is read at its first hour plus the elapsed hours of the run (the start time of day is
not added), repeats every 24 h of elapsed time, holds its first stage below and its last
stage above the curve, and interpolates linearly (`table_lookup`).

A `TIMESERIES` outfall takes its stage elevation (m, relative to datum, not to the invert)
from a driver, as time variation belongs in the drivers. A `.inp` file's TIMESERIES outfall
needs none: the reader reads its `[TIMESERIES]` series (values in the file's length unit,
converted to m) as the outfall's default stage, sampled at the step's END time and held at
its first and last values outside the series (SWMM's `table_tseriesLookup` with `extend`,
node.c:1454-1458). Give `outfall_stage={"Outfall":
stage}` to `build_model` (it sets the initial heads at t = 0 and becomes the model's default)
or to `simulate_dynamic_wave`: `stage` is a number, or a function of the elapsed time `t` in s
that is called once per step with the step's END time (no 1 ms offset, unlike the inflow
function). The two share one code path: `simulate_dynamic_wave` fills the per-outfall driver
`"outfall_stage"` (m) each step, from the tide curves and the functions, and the closure
reads it for every `TIDAL` and `TIMESERIES` outfall; a user stepping `Model.step` directly
supplies that driver. Both work under the fixed and the variable step. A `.inp` with a
`TIMESERIES` outfall is read with its series as the default stage (a network built by hand
has no series, and `build_model` refuses it without an `outfall_stage` by name). The tide fixture
`tidal_dynwave.inp` (a tide that backs the outfall conduit up to 0.38 m of its 0.45 m and
lets it fall) is compared with SWMM 5.2.4 at every step for the `TIDAL` outfall and for the same stage
as a `TIMESERIES` outfall, and the variable-step tree with the tide, with SWMM converging on
every step: the worst depth difference is 8.4e-10 m.

## Conduit options

Each of these is a SWMM `[CONDUITS]`, `[XSECTIONS]`, `[LOSSES]` or `[OUTFALLS]` entry
and a `Pipe` or `Outfall` field:

- **Barrels** (`Pipe.barrels`, `[XSECTIONS]` 7th field, 1 to 127). A conduit's `length`,
  `diameter`, `n` and slope are one barrel's. Its momentum update is one barrel's, and its
  flow, conveyance, surface areas and volume are totals (`n` times a barrel's, `dwflow.c:243,
  292`; `dynwave.c:575`), so `n` barrels at equal heads carry exactly `n` times one barrel's
  flow. The reported velocity is one barrel's.
- **`MaxFlow`** (`Pipe.max_flow`, m³/s, a limit on each barrel, `dwflow.c:270-273`) clamps the
  flow after the normal-flow limit.
- **Flap gates** (`[LOSSES]` flap gate YES for a conduit, `Gated` for an outfall,
  `link_setFlapGate`): flow against a conduit gate's direction is zeroed, and so is flow
  that would enter the conduit from a gated outfall. A conduit the reader reversed (adverse)
  keeps the direction of its file's From to To, `Pipe.flap_gate_reversed`.
- **Lengthening** (`LENGTHENING_STEP`, `DynamicWaveOptions.lengthening_step`, s):
  a conduit shorter than `(sqrt(g y_full) + v_full) t` is lengthened by the factor
  `lf = (sqrt(g y_full) + v_full) t / L`, `t` the smaller of `LENGTHENING_STEP` and the file's
  routing step, to `modLength = lf L` in the momentum terms and the surface areas, with its
  roughness divided by `sqrt(lf)` (the friction coefficient by `lf`), so its normal flow is
  unchanged and its volume keeps the true length. The variable step's Courant limit is scaled
  by `modLength / L`.

These limits apply in SWMM's order (`dwflow.c:255-281`): the normal-flow limit, `MaxFlow`,
the flap gate, then the dry-node caps (at `FUDGE` times the barrels). Each is a clamp of
the element's flow against a constant, so the flow stays continuous and non-decreasing in the
head difference, as the head solve's Newton step needs. A set `InitDepth`, `InitFlow`,
`MaxFlow`, flap gate or barrel count is refused by name under STEADY or KINWAVE routing.

The reader refuses by name, with the line number, the `DUMMY` and `STREET` sections, roadway weirs, weir coefficient curves and regulators with `LINK_OFFSETS
ELEVATION` (see [SWMM `.inp`](swmm.md)).
`build_model` with `storage=True` is refused by name; `quality=True` builds
the water-quality layer on the solved flows (see
[Quality on the dynamic wave](headspace-sulfide.md#quality-on-the-dynamic-wave)) and `air=True` the headspace air
(see [Headspace air on the dynamic wave](headspace-sulfide.md#headspace-air-on-the-dynamic-wave)). Ponding
(`ALLOW_PONDING YES` with a ponded area) is modelled, see [Ponding](physics-dynamic-wave.md#ponding), but not together
with water quality or air; see [Limitations](limitations.md#what-is-not-supported).

```python
from noodl.apps.sewer import read_swmm_inp, build_model, simulate_dynamic_wave

net, _, _ = read_swmm_inp("tree_dynwave.inp")
model, state, drivers = build_model(net)
state, hist = simulate_dynamic_wave(model, state, drivers, steps=24)    # 24 x 5 s
print([round(float(q), 4) for q in hist["water.q"][-1]])                # m3/s, C1..C5
```

```text
[0.0184, 0.031, 0.0032, 0.0092, 0.0004]
```

`hist` holds `"water.phi"` (head at every node, m) and `"water.q"` (flow in every link,
m³/s: the conduits, then any orifices, weirs and outlets, as `model.link_names`), stacked
over steps, with `"t"` (s, the end time of each step) and `"dt"` (s, each
step's length). `simulate_dynamic_wave` also takes an `inflow=` override (`(n,)`
constant, `(B, n)` a batch, `(steps, B, n)` a series, or a function `inflow(t)` of the
start time `t` in s) and `record=` for other state keys and for `"sewer.overflow"` (m³/s
per junction).
