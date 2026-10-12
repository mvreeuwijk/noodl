# SWMM `.inp`

Reading a SWMM input file produces a model for [sewers](../applications/sewer.md). It shares its
lexical tokenizer (`noodl.apps.inpfile`) with the EPANET reader; each format's own section names,
column layouts and units are read by `apps/sewer/inp.py`.

**Read:** `[TITLE]`, `[OPTIONS]`, `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`,
`[LOSSES]` (the flap gate, and under `DYNWAVE` the entry, exit and average loss coefficients),
`[INFLOWS]`, `[POLLUTANTS]`, and under `DYNWAVE` `[STORAGE]`, `[ORIFICES]`, `[WEIRS]`,
`[OUTLETS]` and `[TRANSECTS]` with the `[CURVES]` of type `STORAGE`, `RATING`, `TIDAL` and
`SHAPE`, and the lateral inflows `[TIMESERIES]` (inline and `FILE`), `[PATTERNS]`,
`[DWF]` and `[INFLOWS]` (see Lateral inflows below).

**Ignored** because no conduit or node ever references them: `[EVAPORATION]`, `[REPORT]`,
`[TAGS]`, `[MAP]`, `[COORDINATES]`, `[VERTICES]`, `[POLYGONS]`, `[SYMBOLS]`, `[LABELS]`,
`[BACKDROP]`, `[PROFILES]`; `[FILES]` is read for its `USE` lines (below). Any other section that carries content is refused, and
so are the options and fields listed below.

**Routing.** `FLOW_ROUTING` is `STEADY`, `KINWAVE` or `DYNWAVE`, read as the physical names
`steady`, `kinematic_wave` and `dynamic_wave` (`SewerNetwork.routing`). `DYNWAVE` also reads
`ROUTING_STEP`, `WET_STEP` (default 300 s) and `MINIMUM_STEP` (seconds or `HH:MM:SS`),
`VARIABLE_STEP` (the Courant factor, 0 to 2; absent means SWMM's default 0.75), `INERTIAL_DAMPING`
(`NO` or `NONE`, `PARTIAL`, `FULL`), `NORMAL_FLOW_LIMITED` (`SLOPE`, `FROUDE`, `BOTH`, `NO`, `NONE` or
`NEITHER`), `SURCHARGE_METHOD` (`EXTRAN` or `SLOT`), `MIN_SURFAREA` (m²), `HEAD_TOLERANCE`,
`MAX_TRIALS`, `MIN_SLOPE` (percent, as in SWMM) and `LENGTHENING_STEP` (seconds or `HH:MM:SS`) into
`DynamicWaveOptions`; `HEAD_TOLERANCE` (in the file's length unit) and `MAX_TRIALS` are SWMM's
own iteration controls and the defaults of noodl's, and `FORCE_MAIN_EQUATION` (`H-W` or `D-W`: the
friction law of a full `FORCE_MAIN`) (see
[Iteration and convergence](../applications/sewer.md#iteration-and-convergence)). Any other `[OPTIONS]` key is not read; the options above are read only
under `DYNWAVE`, and an unrecognised value of a keyword option is refused with the line number. As SWMM does,
the reader lowers `ROUTING_STEP` to `WET_STEP` when it is larger (a file that omits
`VARIABLE_STEP` runs the 0.75 variable step, so give `duration=` rather than `dt=` to
`simulate_dynamic_wave`; `DynamicWaveOptions()` built in code defaults to the fixed step).

**Dynamic wave, options refused.** With `DYNWAVE` the reader refuses, with the line number:
`ALLOW_PONDING YES` (ponding is not modelled), a negative `LENGTHENING_STEP` or
`MINIMUM_STEP`, and a `VARIABLE_STEP` outside 0 to 2.
The conduit lengthening uses the file's own `ROUTING_STEP`, as SWMM validates its conduits
before it lowers the step to `WET_STEP`.

**Initial conditions and limits (dynamic wave).** The junction `InitDepth` (the fourth
`[JUNCTIONS]` field) is `Manhole.initial_depth`; the conduit `InitFlow` and `MaxFlow` (the
eighth and ninth `[CONDUITS]` fields) are `Pipe.initial_flow` (a total over the barrels) and
`Pipe.max_flow` (a limit on each barrel); a conduit flap gate (`[LOSSES]` fifth field YES) is
`Pipe.flap_gate`, and an outfall's `Gated` YES is `Outfall.flap_gate`. A conduit the reader
reverses (adverse, as SWMM does) has its `InitFlow` negated and its gate recorded as
reversed. The entry, exit and average loss coefficients of `[LOSSES]` are
`Pipe.loss_inlet`, `loss_outlet` and `loss_average` (swapped, as SWMM does, for a conduit the
reader reverses); the seepage rate must be 0 (refused with the line number otherwise: seepage
takes volume from the nodes at a rate set by the conduit's wetted width and volume, which is
not modelled). Under `STEADY` or `KINWAVE` a non-zero `InitDepth`, `InitFlow` or
`MaxFlow`, a flap gate, a gated outfall and a loss coefficient are refused by name.

**Cross sections (dynamic wave).** `[XSECTIONS] Link Shape Geom1 Geom2 Geom3 Geom4 [Barrels
[Culvert]]` is read for every SWMM shape but `DUMMY` and `STREET` (refused by name):
`CIRCULAR`, `FILLED_CIRCULAR`, `RECT_CLOSED`, `RECT_OPEN`, `TRAPEZOIDAL`, `TRIANGULAR`,
`PARABOLIC`, `POWER`, `RECT_TRIANGULAR`, `RECT_ROUND`, `MODBASKETHANDLE`, `HORIZ_ELLIPSE`,
`VERT_ELLIPSE`, `ARCH`, `EGG`, `HORSESHOE`, `GOTHIC`, `CATENARY`, `SEMIELLIPTICAL`,
`BASKETHANDLE`, `SEMICIRCULAR`, `IRREGULAR` (a transect), `CUSTOM` (`Geom1` the depth, `Geom2`
the name of a `[CURVES]` SHAPE curve of width against height) and `FORCE_MAIN` (`Geom2` the
Hazen-Williams C, or the Darcy-Weisbach roughness height in inches or mm, which the reader
converts to m). The lengths among `Geom1..Geom4` are converted (`LENGTH_FIELDS`); side slopes,
exponents, size codes and the C factor are not. For `HORIZ_ELLIPSE`, `VERT_ELLIPSE` and `ARCH`
a zero `Geom2` makes `Geom1` the standard size code, as in SWMM (`xsect.c`); with a positive
`Geom3` the standard size of that code is used whatever `Geom1` and `Geom2` say. Each conduit
becomes a `Pipe` with an `Xsect` (`xsect_spec.py`) and the full height as its `diameter`; the
section's constants are computed exactly as `xsect.c::xsect_setParams` does, from the file's
numbers (the unit factor applied as SWMM applies it). Under `STEADY` or `KINWAVE` any shape
but `CIRCULAR` is refused with the line number, as is `[TRANSECTS]`.

**Transects (dynamic wave).** `[TRANSECTS]` `NC nLeft nRight nChannel`, `X1 Name nSta
xLeftBank xRightBank 0 0 Lfactor Xfactor Yfactor` and `GR Elevation Station ...` lines are read
with SWMM's reader state (`transect.c`): a zero Manning n on an `NC` line keeps the value
of the previous transect (the main channel's, as SWMM leaves it after scaling it by
`sqrt(Lfactor)`), an overbank n of 0 takes the channel's, the station multiplier `Xfactor` and
the elevation offset `Yfactor` apply to every station (SWMM divides `Yfactor` by the unit
factor twice, `transect.c:353, 377`, and so does the reader). A conduit that uses a transect
takes its roughness from it (the file's Manning n is ignored) times `sqrt(Lfactor)`, and its length is the
file's over `Lfactor` (`Pipe.transect_length_factor`). Refused with the line number: fewer
than two stations or 1500 or more, stations that run backwards, no depth, no main-channel n,
a left bank right of the right bank, and a `GR` line before an `X1` line.
SWMM validates a transect when the next `NC` line or the next section begins; a transect that
is the last thing in the file is therefore never validated by SWMM, which then refuses the
link that uses it (its error 119) -- end a file's `[TRANSECTS]` with another section.

**Culverts.** The culvert code (the eighth `[XSECTIONS]` field, 1 to 57) is `Pipe.culvert`:
under `DYNWAVE` the conduit is limited, while it is not full, to the inlet-control flow of
FHWA HEC-5 (`culvert.py`); a code above 57 is ignored, as SWMM does; under `STEADY` or
`KINWAVE` a non-zero code is refused.

**Junctions.** `Elevation` is the invert, `MaxDepth` sets `Manhole.ground` (invert plus
depth) and `SurDepth` is `Manhole.surcharge_depth` (m, at least 0; 0 when absent): the depth
a junction may pressurise above its full depth before it floods under `DYNWAVE`. `Aponded`
is not read, since ponding is refused.

**Storage units (dynamic wave).** A `[STORAGE]` line `Name Elev MaxDepth InitDepth Shape
... [SurDepth [Fevap]]` becomes a `Manhole` with a `storage` curve (`storage_unit`): the
invert, `ground` = invert + `MaxDepth`, `initial_depth` and `surcharge_depth`. The shapes are
SWMM 5.2.4's: `FUNCTIONAL a1 a2 a0` (area `a1 y^a2 + a0`, m², read as `("functional", a1, a2,
a0)`; `a0 >= 0` and `a2 >= 0`), `TABULAR Curve` (a `[CURVES]` curve of type `STORAGE`, depth in
m against area in m², its points over any number of lines, depths strictly increasing) and the
closed shapes `CYLINDRICAL L W -`, `CONICAL L W Z`, `PARABOLIC L W H` (SWMM 5.2.4's keyword; `PARABOLOID` is accepted too) and
`PYRAMIDAL L W Z`
(m; `Z` a side slope, run over rise). A non-zero evaporation factor `Fevap` and a non-zero
exfiltration `Ksat` (the value after `Fevap`, or the second of suction head, `Ksat`,
`IMDmax`; a `Ksat` of 0 is no exfiltration, as in SWMM) are refused with the line number, as
are a `[CURVES]` curve of any other type and a curve name that is not in `[CURVES]`. A
negative `MaxDepth`, `InitDepth` or `SurDepth` on a storage line is refused too, which goes
beyond SWMM's own input checks for storage units. `[STORAGE]` under
`STEADY` or `KINWAVE` is refused by name: SWMM routes a storage unit by its volume there,
which is not modelled.

**Orifices, weirs and outlets (dynamic wave).** Each becomes an `Orifice`, `Weir` or
`Outlet` of `SewerNetwork.orifices`, `.weirs` or `.outlets`, between two nodes:

- `[ORIFICES] Name N1 N2 SIDE|BOTTOM CrestHt Cd [Gated] [CloseTime]` with an `[XSECTIONS]`
  opening `CIRCULAR` (diameter) or `RECT_CLOSED` (height, width). `CloseTime` only shapes
  control actions, which are not read, so it is accepted and has no effect.
- `[WEIRS] Name N1 N2 Type CrestHt Cd [Gated] [EndCon] [Cd2] [CanSurcharge]` (a `*` keeps
  SWMM's default: no gate, no end contractions, `Cd2` 0, `CanSurcharge` YES), `Type`
  `TRANSVERSE` or `SIDEFLOW` (opening `RECT_OPEN`: height, length), `V-NOTCH` (`TRIANGULAR`:
  height, top width) or `TRAPEZOIDAL` (`TRAPEZOIDAL`: height, bottom width, the two side
  slopes). `Cd` is SWMM's SI coefficient (m^0.5/s). The road width and surface fields, read
  by SWMM for roadway weirs only, are ignored.
- `[OUTLETS] Name N1 N2 Offset TABULAR/DEPTH|TABULAR/HEAD Curve [Gated]` or `...
  FUNCTIONAL/DEPTH|FUNCTIONAL/HEAD C1 C2 [Gated]` (`q = C1 h^C2`, m and m3/s); the curve is a
  `[CURVES]` curve of type `RATING` (head or depth in m against flow in m3/s, x values
  strictly increasing). A negative offset reads as 0, as in SWMM.

`CrestHt`/`Offset` is the height above the first node's invert (`crest_height`, m).
Refused with the line number: a `ROADWAY` weir, a weir coefficient curve (the thirteenth
field), an opening of the wrong shape (SWMM's ERR_REGULATOR_SHAPE), a regulator without an
`[XSECTIONS]` entry, an outlet with one, a negative orifice or weir crest height (SWMM
raises it to 0 at the inlet end only), a negative coefficient, unknown nodes and keywords,
`[CURVES]` of a type other than `STORAGE`, `RATING` and `TIDAL`, the regulator sections under
`STEADY` or `KINWAVE` (SWMM routes them by the inflow alone there), and regulators with
`LINK_OFFSETS ELEVATION` (SWMM then sets each end's offset from its own node).

**Outfalls.** `FREE`, `NORMAL`, `FIXED` (with its stage, m), `TIDAL` (with the name of its
`[CURVES]` TIDAL curve, read as hours and stage elevations) and `TIMESERIES` (with the name of
its series) are read. A `TIMESERIES` outfall takes its stage from the `outfall_stage` driver
of `build_model` / `simulate_dynamic_wave` (time variation belongs in the drivers): that
series is not read. An outfall may sit
at the end of a conduit or of a regulator (exactly one link, SWMM's ERR_OUTFALL).

**Barrels.** A conduit with more than one barrel is read under `DYNWAVE` (1 to 127; the
flows, volumes and surface areas of the pipe are totals) and refused with the line number under
`STEADY` and `KINWAVE`.

**Units.** `FLOW_UNITS` may be `CFS`, `GPM`, `MGD` (the US system: lengths in ft) or `CMS`,
`LPS`, `MLD` (the SI system: lengths in m); an absent `FLOW_UNITS` is SWMM's default, `CFS`. The
reader converts everything to SI (m, m3/s, s) as it reads, with SWMM's own constants
(`unit_factors`). Lengths multiply by 0.3048 (US) or 1 (SI), areas by its square, flows by
`0.02832 / Qcf`, where `Qcf` is SWMM's number of user flow units per cfs. SWMM 5.2.4 rounds
these (`units.c`: 448.831 gpm, 0.64632 mgd, 0.02832 cms, 28.317 lps, 2.4466 mld per cfs), so
1 LPS is 1.00011e-3 of noodl's m3/s and a CMS file converts by exactly 1; using the exact
$0.0283168$ would disagree with SWMM by 1e-4.

| `FLOW_UNITS` | system | length unit | m3/s per flow unit |
|---|---|---|---|
| `CFS` | US | ft = 0.3048 m | 0.02832 |
| `GPM` | US | ft | 0.02832 / 448.831 = 6.3101e-5 |
| `MGD` | US | ft | 0.02832 / 0.64632 = 0.043816 |
| `CMS` | SI | m | 1 |
| `LPS` | SI | m | 0.02832 / 28.317 = 1.00011e-3 |
| `MLD` | SI | m | 0.02832 / 2.4466 = 0.011575 |

Converted: node invert elevations, maximum, initial and surcharge depths, fixed and tidal
outfall stages, conduit lengths, offsets and diameters, regulator openings (`Geom1`, `Geom2`),
orifice, weir and outlet crest heights, `[CURVES]` (STORAGE: depth and area; RATING: depth or
head and flow; TIDAL: stage; SHAPE: none, the curve is relative to the section's height),
`FUNCTIONAL` storage (`A = a1 y^a2 + a0`: `a1` scales with
length^(2 - a2), `a0` with length^2), the `CYLINDRICAL`, `CONICAL`, `PARABOLOID` and
`PYRAMIDAL` lengths, `InitFlow`, `MaxFlow`, constant `FLOW` baselines, `MIN_SURFAREA` (an
area) and `HEAD_TOLERANCE` (a length). A `FUNCTIONAL` outlet `q = a h^b` converts as
`a * flow / length^b`. A weir's Cd (`Q = Cd L h^1.5`) in a US file, in ft^0.5/s, becomes
the SI one by SWMM's own `M3perFT3 / ft^2.5` (0.028317 / 0.3048^2.5); an SI file, whatever its
flow unit, gives it in m^0.5/s as it is. Not converted: Manning's n, slopes, side slopes,
orifice Cd, end contractions, `MIN_SLOPE` (percent), times and concentrations. The test suite
rewrites each CMS fixture into the other five systems and requires the same network back, and
runs the rewritten files through SWMM itself.

**Constraints:** `LINK_OFFSETS` must be `DEPTH` or `ELEVATION` (`DEPTH` with regulators). Under
`STEADY` and `KINWAVE` only `CIRCULAR` conduit cross-sections. A pollutant inflow must be a constant
`CONCEN` (or `CONCENTRATION`) baseline with Mfactor 1.

**Lateral inflows.** `[INFLOWS]`, `[DWF]`, `[PATTERNS]` and `[TIMESERIES]` are read with
SWMM 5.2.4's rules (`apps/sewer/inflows.py`; every statement below was checked against the
engine through pyswmm, step by step, in
`tests/verification/test_sewer_inflows_parity.py`):

- `[INFLOWS] Node FLOW Series Type Mfactor Sfactor Baseline Pattern`: the flow is
  `Sfactor * series(t) + Baseline * pattern(t)` in the file's flow units (converted with
  `unit_factors`). `Sfactor` scales the series only and `Pattern` multiplies the baseline only.
  SWMM **ignores `Mfactor` for a `FLOW` inflow** (it converts pollutant mass flows), so it is
  read and has no effect. A repeated line for a node replaces the earlier one. `Series` is `""`
  or `*` for none.
- `[DWF] Node FLOW Average Pat1..Pat4`: the average times the monthly, daily and hourly
  factors; on Saturday and Sunday the weekend pattern takes the place of the hourly one (and
  is 1 on other days). SWMM reads the first four pattern fields only and keeps one pattern per
  kind, the last of its kind; `""` skips a field; an unknown name (and `*`) is refused. A
  repeated line replaces the earlier one. Only `FLOW` is read: a `[DWF]` line for a pollutant,
  or a non-zero `Cdwf` in `[POLLUTANTS]` while a `[DWF]` flow exists, is refused.
- `[PATTERNS]`: `MONTHLY` (12 factors), `DAILY` (7, Sunday first), `HOURLY` (24), `WEEKEND`
  (24); a pattern may continue on later lines, a missing factor is 1 and an extra one is
  ignored, as in SWMM. A pattern name defined in quotes is not accepted by SWMM itself; a
  reference in quotes (`"PATN-1"`) is.
- `[TIMESERIES]`: entries `[date] time value` repeated on a line, a date applying to the later
  entries including on later lines; `time` is decimal hours or `H:M[:S]`; dates are `M/D/Y`,
  `Y-M-D` or `D-MON-Y`. A series with no dates counts hours **since the start of the
  simulation** (`START_TIME` included); a dated one uses the calendar. Times must strictly
  increase. The series is linearly interpolated and is **0 before its first and after its last
  entry**. `name FILE path` reads `[date] time value` lines (`;` comments) from a file resolved
  against the `.inp`'s folder; a missing file is refused naming it. A series that no inflow
  names (a `TIMESERIES` outfall's stage series included) is not read.
- Sampling: the inflow of the step that starts `t` s into the run is evaluated at
  `START_DATE + START_TIME + t + 0.001 s` (SWMM's clock reads 1 ms late); the month, day of
  week and hour of the patterns are those of that clock. `START_DATE` defaults to SWMM's
  01/01/2004.
- A flow smaller in magnitude than SWMM's `FLOW_TOL` of 1e-5 cfs is dropped, separately for the
  external inflow of a node (series plus baseline) and for its dry-weather flow.

A file whose flows vary in time (a series, a pattern, patterned dry-weather flow) gets
`SewerNetwork.inflows`, an `InflowSchedule`; `read_swmm_inp` still returns its three values.
`schedule.driver(list(model.net.nodes))` is the `inflow(t)` callable of
`simulate_dynamic_wave`, which uses the model's schedule when it is given no `inflow`;
`Manhole.inflow` is each node's value at the start. Under `STEADY` and `KINWAVE` the model is
quasi-steady, so a time-varying inflow reaches it as the driver
`dict(drivers, inflow=schedule.driver(names)(t))` the caller sets at the time of interest
(`model.inflow_schedule`). Time series for pollutants, `MASS` pollutant inflows and an
`Mfactor` other than 1 on a pollutant inflow are refused with the line number.

**`[FILES]`.** `SAVE` lines (hot starts, outflows, rainfall, runoff, RDII) write files and are
ignored. `USE HOTSTART`, `USE INFLOWS`, `USE RDII` and `USE RUNOFF` change the physics and are
refused by name; `USE RAINFALL` only feeds rain gauges, which are refused anyway.

**Refused by name**, because skipping them would silently change the physics:
`[STORAGE]`, `[ORIFICES]`, `[WEIRS]` and `[OUTLETS]` under `STEADY` or `KINWAVE`, `[PUMPS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[STREETS]`, `[CURVES]` of any
type but `STORAGE`, `RATING`, `TIDAL` and `SHAPE`, `[LID_USAGE]`, and any other section carrying
content.

**Geometry.** `read_swmm_inp` builds its network with `geometry="tabulated"`: SWMM 5.2's own
tabulated circular section, lookup rules and unit constants. Depths, volumes and velocities
therefore reproduce SWMM's rather than the exact circle's (see
[SWMM's own circular geometry](../applications/sewer.md#swmms-own-circular-geometry)). Pass
`geometry="analytic"` for the exact circle. `geometry="tabulated"` does not combine with
`storage=True`.

**Water quality.** The sewer model reacts first and mixes second, and returns the
concentration after mixing, which is where SWMM's `qualrout.c` samples it. A steady pollutant
therefore matches SWMM's link quality to round-off at any step (see
[the SWMM comparison](../applications/sewer.md#verification)). In a transient, conduits fed by upstream
conduits differ by SWMM's one-step junction lag, which is first order in the step.
Under `DYNWAVE`, `build_model(net, quality=True)` puts the quality reactors at the junctions
on the solved flows, where SWMM's are the conduits (see
[Quality on the dynamic wave](../applications/sewer.md#quality-on-the-dynamic-wave)): a
conservative pollutant matches SWMM's link quality at steady state (to 1e-8 relative on the
tree and the loop), and in transit the two differ by their residence times. A pollutant's
`Kdecay` is read (`pollutants[name]["decay"]`, 1/s); the model's own reactions are BOD decay
and sulfide generation, and a tracer of any other name is conservative. A nonzero `Cinit` is refused (the model starts
with clean water); the other pollutant fields feed only runoff, groundwater, RDII and
dry-weather inflow, which are refused or absent.

**Not supported**, which rules out most published SWMM examples as they stand:

- runoff (`[SUBCATCHMENTS]` and everything that feeds them) and dry-weather patterns;
- the `DUMMY` and `STREET` cross sections, conduit seepage and evaporation losses;
- pumps, dividers, roadway weirs, weir coefficient curves and control rules (regulator
  settings); orifices, weirs, outlets and storage units under `STEADY` or `KINWAVE`, and
  storage evaporation or exfiltration;
- ponding (`ALLOW_PONDING YES`). Both `SURCHARGE_METHOD`s model
  surcharge, and a junction floods (without ponding) above `MaxDepth` (extended to its
  crown) plus `SurDepth`; the water that cannot enter is lost, as in SWMM without ponding.

The complete list of what the dynamic wave does not support, and of what it implements without
a live SWMM comparison, is under [Limitations](../applications/sewer.md#limitations).

**Slope** follows SWMM's own definition $S_0 = dy/dx$ with $dx = \sqrt{L^2 - dy^2}$ — the 3-D
chord, not the naive $dy/L$. Under `STEADY` and `KINWAVE`, zero or adverse fall is refused, as
is a fall $\ge$ length. Under `DYNWAVE` the reader follows SWMM's `conduit_getSlope`: the fall is
floored at SWMM's `MIN_DELTA_Z` (0.001 ft, 0.0003048 m), `MIN_SLOPE` floors the slope, and a
conduit is reversed only when it is adverse, so a flat conduit keeps the orientation of the
file. Conduit offsets are kept (`Pipe.inlet_offset`, `Pipe.outlet_offset`, m above the end
node's invert) and swap with the ends on a reversal.

```python
from noodl.apps.sewer import read_swmm_inp, build_model

net, inflows, pollutants = read_swmm_inp("tree_steady.inp")
model, state, drivers = build_model(net)
```
