# Lateral inflows, runoff and RDII from a SWMM file

How the lateral inflow of a SWMM file reaches the sewer model: time series, dry-weather flow and
patterns, subcatchment runoff and RDII. The sections below say what is read and how SWMM's own
rules are reproduced; the section-by-section list of the file is on [SWMM `.inp`](swmm.md). The
routed run is described in [Dynamic-wave runs](dynamic-wave.md).

## Lateral inflows from a SWMM file

`read_swmm_inp` reads SWMM's `[INFLOWS]` (`FLOW` from a time series, with `Sfactor`, a baseline
and a baseline pattern), `[DWF]` with `[PATTERNS]` (monthly, daily, hourly and weekend, one pattern
per kind, as SWMM keeps them) and `[TIMESERIES]` (inline, or `name FILE path` beside the
`.inp`), on the calendar of `START_DATE` and `START_TIME`, with SWMM 5.2.4's own rules for
interpolation (linear; zero outside the series; a series with no dates counts hours since the
start), the sampling time (step start + 1 ms), `Mfactor` (ignored for a flow) and the `FLOW_TOL`
of 1e-5 cfs ([SWMM `.inp`](swmm.md) lists them with their sources).
A file whose flows vary gets `SewerNetwork.inflows`, an `InflowSchedule`; the model keeps it as
`model.inflow_schedule`, and

```python
from noodl.apps.sewer import read_swmm_inp, build_model, simulate_dynamic_wave

net, _, _ = read_swmm_inp("inflow_dwf_dynwave.inp")
model, state, drivers = build_model(net)
state, hist = simulate_dynamic_wave(model, state, drivers, duration=60.0)     # uses the schedule
inflow = net.inflows.driver(list(model.net.nodes))                          # inflow(t) -> (n,) m3/s
```

`simulate_dynamic_wave` takes the schedule when it is given no `inflow`; the function above is
the same thing as an explicit `inflow=`, and a caller's own `inflow` overrides it. Under
`STEADY` and `KINWAVE` the model is quasi-steady, so the caller sets
`dict(drivers, inflow=inflow(t))` for the time it wants. The lateral inflow of every node is compared with SWMM's own at every step, over inline dated and
hour-based series, a `FILE` series, dry-weather flow with all four pattern kinds across a
midnight, month and weekend change, and an MGD file with `Mfactor`, `Sfactor` and a baseline
pattern, and routes three of them against SWMM at the depth (1e-7 m) and flow (1e-6 relative)
targets of the other dynamic-wave comparisons.

## Subcatchment runoff from a SWMM file

A file with `[SUBCATCHMENTS]` gets its hydrology: `read_swmm_inp` reads `[RAINGAGES]`,
`[SUBCATCHMENTS]`, `[SUBAREAS]`, `[INFILTRATION]`, `[EVAPORATION]` and `[ADJUSTMENTS]` into
`net.hydrology` (`noodl.apps.sewer.runoff.Hydrology`), and `net.inflows` becomes a
`RunoffInflows`: the file's inflow schedule plus the runoff of the subcatchments, which
`simulate_dynamic_wave` takes as its lateral inflow (`model.inflow_schedule`) with no argument from
the caller. It is SWMM 5.2.4's own computation, operation for operation (`subcatch.c`,
`runoff.c`, `infil.c`, `gage.c`, `climate.c`, `odesolve.c` and `routing.c`; the module
docstring of `runoff.py` lists each rule with its source):

- **Rain.** A `TIMESERIES` gage, `INTENSITY`, `VOLUME` or `CUMULATIVE`; the rain of an entry
  falls for the gage's recording interval from the entry's date and is zero until the next
  non-zero entry; a record that starts after (before) the run is dry before it (already running);
  gages that share a series take the rainfall of the first. A gage's interval shorter than
  `WET_STEP` lowers `WET_STEP`, and with it the routing step, as `gage_validate` does.
- **Runoff clock.** SWMM's runoff steps are not the routing steps. A runoff step is
  `WET_STEP` while it rains or a subcatchment had runoff on the previous step, `DRY_STEP`
  otherwise, shortened to the next change of rain or of monthly evaporation, and the last one is
  cut at the end of the run; routing then adds each subcatchment's runoff to its outlet node
  linearly interpolated between the runoff at the start and at the end of the runoff step that
  covers the start of the routing step. `simulate_dynamic_wave` therefore passes the length of
  each routing step to the runoff driver (an inflow callable with `step_aware = True` is called
  as `inflow(t, step)`).
- **Surface.** Each subcatchment is an impervious area without depression storage, an impervious
  area with it and a pervious area; each is a nonlinear reservoir `q = alpha (d - d_s)^(5/3)`
  integrated over the step by SWMM's Cash-Karp Runge-Kutta with its adaptive step (tolerance
  1e-4). `RouteTo` `IMPERVIOUS` or `PERVIOUS` sends `PctRouted` of one area's runoff of the
  previous step onto the other; the runoff of a subcatchment that drains to another one is that
  one's runon on the next runoff step.
- **Infiltration.** `HORTON`, `MODIFIED_HORTON`, `GREEN_AMPT`, `MODIFIED_GREEN_AMPT` and
  `CURVE_NUMBER`, as the `[OPTIONS]` default or per subcatchment on its `[INFILTRATION]` line
  (SWMM 5.2's trailing method keyword), with the soil recovery of dry weather and the monthly
  `RECOVERY` pattern. A subcatchment with no `[INFILTRATION]` line has none, as in SWMM.
- **Evaporation.** `CONSTANT`, `MONTHLY` or `TIMESERIES` (a step function), `DRY_ONLY`, and the
  `[ADJUSTMENTS]` of evaporation, rainfall and conductivity and the `N-PERV`, `DSTORE` and
  `INFIL` patterns, which SWMM applies only to a subcatchment named *before* the line (it resets
  them when it reads `[SUBCATCHMENTS]`; the reader does the same, so a file with them in the other
  order has them ignored, here as in SWMM).

Refused by name, with the line: `[LID_USAGE]`; `[GROUNDWATER]` (unless `IGNORE_GROUNDWATER YES`); a
subcatchment with a snow pack (unless `IGNORE_SNOWMELT YES`); `[LANDUSES]`, `[COVERAGES]`,
`[BUILDUP]`, `[WASHOFF]`, `[LOADINGS]` and a rain concentration of a pollutant (their load is not
modelled; `IGNORE_QUALITY YES` lets them pass); an outfall that routes onto a subcatchment;
rain `FILE` gages; `[EVAPORATION]` `TEMPERATURE` and `FILE`. The
evaporation of **open conduits** and the storage units' is that of
[Evaporation and seepage losses](dynamic-wave-structures.md#evaporation-and-seepage-losses). `[SNOWPACKS]` no
subcatchment names, `[TEMPERATURE]`, `[AQUIFERS]` and `[LID_CONTROLS]` that nothing uses are
inert, as they are in SWMM.

```python
from noodl.apps.sewer import read_swmm_inp
from noodl.apps.sewer.runoff import RunoffModel

net, _, _ = read_swmm_inp("runoff_horton_dynwave.inp")
hydrology = net.hydrology
runoff = RunoffModel(hydrology)
runoff.advance_to(540_000.0)           # the runoff steps SWMM has taken by t = 540 s
print({name: round(float(q), 4) for name, q in runoff.runoff_m3s.items()})
print(runoff.steps[:3], len(runoff.steps))
```

```text
{'S1': 0.916, 'S2': 0.2232, 'S3': 0.3022}
[(0.0, 120.0), (120.0, 60.0), (180.0, 60.0)] 8
```

(The first runoff step is the 120 s the dry step is cut to by the first rain; the runoff steps are
then the one-minute `WET_STEP`.) The routed run needs nothing more:

```py
model, state, drivers = build_model(net)
state, hist = simulate_dynamic_wave(model, state, drivers, duration=600.0)   # runoff is the inflow
```

**Differentiability.** The runoff is plain arithmetic in which tensors flow, so every
parameter of a subcatchment can carry a gradient: area, percent imperviousness, width, slope,
the Manning roughness and depression storage of both areas, `PctZero`, `PctRouted`, and every
parameter of the infiltration models (rates, decay, drying time, suction, conductivity, initial
deficit, curve number). `Hydrology.with_gradients` makes leaf tensors of them (and returns them
by name, e.g. `"S1.area"`, `"S1.infiltration.f0"`); a `RunoffInflows` built from it
(`RunoffInflows(h2, schedule, nodes)`, set as `model.inflow_schedule`) gives
`simulate_dynamic_wave` a lateral inflow with that graph, and the gradient of a routed depth or
flow reaches the runoff parameters through the head solve's implicit differentiation:

```python
h2, leaves = hydrology.with_gradients(fields=("area",), subareas=(), infiltration=False,
                                      only=("S1",))
m = RunoffModel(h2)
m.advance_to(540_000.0)
m.runoff_m3s["S1"].backward()
print(round(float(leaves["S1.area"].grad), 8))     # d runoff / d area, m3/s per m2
```

```text
2.136e-05
```

The gradient is that of the branch the values select (rain on or off, ponding above the depression
storage, saturation of the Green-Ampt or Curve Number surface, the Horton curve against the
available water, the 10..99 clamp of the Curve Number), exact in each smooth region and
one-sided at a switch, and it is that of the discrete scheme: the Runge-Kutta step sizes are chosen
on the values and held fixed (the solver's tolerance, 1e-4, bounds the difference from the
derivative of the exact solution; the tests compare with finite differences to 2e-3). Not
differentiable: the rain itself (series values, recording interval, snow catch factor), the
**runoff clock** (which step is wet or dry, where the last step is cut) and everything that
selects the routing step. `Hydrology.with_duration` re-cuts the run, as SWMM does, if the
simulation is shorter than the file's `END_DATE`.

**Notes on reproducing SWMM.** (1) SWMM's total duration is the floor of `(END - START) * 86400`
evaluated by the compiled engine as `(EndDate - StartDate) + (EndTime - StartTime)`, which is
1199 s, not 1200 s, for 12:20:00 after 12:00:00; the reader does the same arithmetic
(`inp_hydrology.swmm_total_duration`), because the last runoff step is cut there. (2) SWMM matches
keywords by prefix of the field (`IMPERV`, `PERV`, `VOLUME`...), and so does the reader.
(3) `WET_STEP` and `DRY_STEP` are read as SWMM's times (decimal hours or `H:MM:SS`).

The runoff is compared with live SWMM 5.2.4 for
every subcatchment and the lateral inflow of every node after every routing step (1e-6 relative
+ 1e-9 m3/s) and the routed depths (1e-7 m) and flows of five fixtures (Horton, modified Horton,
Curve Number, Green-Ampt and modified Green-Ampt; SI, LPS and US units; `RouteTo` `OUTLET`,
`IMPERVIOUS` and `PERVIOUS`; subcatchment-to-subcatchment routing; the three rain formats; co-gages; monthly,
constant and time-series evaporation; the adjustments and recovery patterns; a month boundary;
gages faster than `WET_STEP`) and of sixteen variants of them.

## RDII from a SWMM file

`[HYDROGRAPHS]` and `[RDII]` give rainfall-dependent infiltration and inflow: a triangular unit
hydrograph set (three responses, SHORT, MEDIUM and LONG, per month) convolved with the rain of a
gage and applied to the sewershed area of a node. `read_swmm_inp` reads them into
`net.hydrology.rdii` (`noodl.apps.sewer.rdii.Rdii`; the reader is `inp_rdii.py`) and
`RunoffInflows` adds the RDII of each node to the lateral inflow, **whether or not the file has
subcatchments** (a file with only RDII has `net.hydrology.subcatchments == ()`). It is SWMM 5.2.4's
own computation (`rdii.c`, with `rain.c`, `gage.c` and `routing.c`'s `addRdiiInflows`); SWMM writes
the RDII of the whole run to an interface file before it starts and the routing reads that file, and
noodl computes the same records directly, in order, as the routing asks for them. The module
docstring of `rdii.py` has every rule with its source:

- **Unit hydrographs.** `Name Gage`, then `Name Month Response R T K [Dmax Drecov Dinit]` (`ALL`
  or `JAN`..`DEC`; later lines replace earlier ones, so `ALL` then `JAN` overrides January) or
  the older one-line `Name Month R1 T1 K1 R2 T2 K2 R3 T3 K3 [Dmax Drecov Dinit]`. The triangle has
  area R, time to peak `T` hours and base `T (1 + K)`, both truncated to whole seconds. The
  initial abstraction (`Dmax` depth, `Drecov` depth per day, `Dinit` used at the start) is per
  response and month.
- **Rainfall.** The rain interval of a set is the `WET_STEP` (already lowered to the smallest
  gage interval, which now includes the gages of the unit hydrographs, as in SWMM) lowered to
  the shorter limb of any of its triangles; the RDII step is the `WET_STEP`. At each record the
  rain of every interval since the last is taken from the gage, reduced by the unused initial
  abstraction of the interval's month, stored, and convolved with the triangle at the middle of
  each past interval. A dry spell longer than the triangle's base begins a new event.
- **Flow.** A node's RDII is the sets' flow times its area over the unit factor of rainfall, in
  cfs, zero below 1e-4 cfs and stored in single precision. A record holds for its `WET_STEP`
  and the routing reads the record in force at the step's date plus 1 ms: the flow is **constant
  within a record, not interpolated** (unlike runoff). The area is in acres (hectares in SI units).
  RDII at an outfall is dropped, as runoff is.

SWMM quirks that are reproduced, because a parity test fails without each of them: a gage's state is
set once per record, so a set whose rain interval is shorter than the record uses the rain of the
first of its intervals for all of them; sets that share a gage advance its one state in turn; the
month of the stored rain is that of the record while the abstraction takes the month of the
interval; the dates are SWMM's double-precision days and are compared as SWMM does; and, when the
file also has subcatchments and `[ADJUSTMENTS]` `RAINFALL`, SWMM leaves the rain factor of the month
of the last RDII interval behind and the runoff gage reads its first records with it (the reader
reproduces that by running the RDII to the end of the file's duration when the factors differ).

Refused by name: a set a node uses with no `Name Gage` line (SWMM would read the first gage of the
file), an unknown gage, node or set, negative ratios or times, ratios that add to more than 1.01 in
a month (SWMM's `ERR_UNITHYD_RATIOS`), a negative area, a pollutant's RDII concentration `Crdii` (its
load is not modelled; `IGNORE_QUALITY YES` lets it pass), and `[FILES]` `USE RDII`. `IGNORE_RDII YES`
and `IGNORE_RAINFALL YES` give no RDII (the gages of the sets still count as used).

```py
net, _, _ = read_swmm_inp("rdii.inp")
rdii = net.hydrology.rdii                  # sets, inflows (m2), ignore
model = rdii.model(net.hydrology)          # SWMM's records, computed in order
print(model.flows(1200.0))                 # {node: m3/s} of the routing step that starts at 1200 s
```

**Differentiability.** The response ratios R, the initial abstraction parameters and the sewershed
areas carry exact gradients; the time to peak T and the recession ratio K do so straight-through: the
value is SWMM's truncation to whole seconds, the derivative that of the untruncated product (the true
derivative of a truncation is zero). `Rdii.with_gradients` makes leaf tensors (`"UH1.ALL.SHORT.r"`,
`"J1.area"`; months that share a response share a leaf), and a `RunoffInflows` built from the
resulting hydrology (`dataclasses.replace(hydrology, rdii=...)`) gives `simulate_dynamic_wave` a
differentiable lateral inflow; tests compare the gradient of a routed depth in the area with a
finite difference. The comparisons (rain on or off, abstraction used up, the dry period, the
triangle's limbs), the rain interval, the number of past intervals and the single-precision rounding
are decided on the values. Not differentiable: the rainfall and the dates.

RDII is compared with SWMM by comparing the lateral inflow
of every node after every routing step (1e-6 relative + 1e-9 m3/s) and the routed depths (1e-7 m)
and flows with SWMM at `HEAD_TOLERANCE` 1e-12: one set with all three responses, monthly parameters
across a month boundary, an initial abstraction with recovery, US units with an intensity gage,
RDII with subcatchment runoff on the same nodes, a short limb with the records off the rain grid,
two sets on one gage and on co-gages, the older line, a gage faster than `WET_STEP`, the rain
adjustment left behind for the runoff, RDII tails below SWMM's 1e-4 cfs, `IGNORE_RDII` and
`IGNORE_RAINFALL`.
