# SWMM `.inp`

Reading a SWMM input file produces a model for [sewers](../applications/sewer.md). It shares its
lexical tokenizer (`noodl.apps.inpfile`) with the EPANET reader; each format's own section names,
column layouts and units are read by `apps/sewer/inp.py`.

**Read:** `[TITLE]`, `[OPTIONS]`, `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`,
`[LOSSES]` (the flap gate, and under `DYNWAVE` the entry, exit and average loss coefficients),
`[INFLOWS]` (a flow into an outfall is checked and dropped: dynamic-wave routing never solves an
outfall's depth from its inflow), `[POLLUTANTS]`, and under `DYNWAVE` `[DIVIDERS]` (as junctions, below), `[STORAGE]`,
`[ORIFICES]`, `[WEIRS]`, `[OUTLETS]`, `[PUMPS]`, `[CONTROLS]` and `[TRANSECTS]` with the `[CURVES]` of type
`STORAGE`, `RATING`, `TIDAL`, `SHAPE`, `DIVERSION`, `PUMP1`-`PUMP5` and `CONTROL`, the lateral inflows
`[TIMESERIES]` (inline and `FILE`), `[PATTERNS]`,
`[DWF]` and `[INFLOWS]` (see Lateral inflows below) and the hydrology `[RAINGAGES]`,
`[SUBCATCHMENTS]`, `[SUBAREAS]`, `[INFILTRATION]`, `[EVAPORATION]` and `[ADJUSTMENTS]`, and the
RDII sections `[HYDROGRAPHS]` and `[RDII]` (see Hydrology below).

**Before the first section.** SWMM reads a line before the first `[SECTION]` header as `[TITLE]`
text (`input.c:174`: the second pass over the file starts in section 0, the title), so the reader
does the same. A version-control merge-conflict marker (`<<<<<<<`, `=======`, `>>>>>>>`) anywhere
is refused with its line: such a file holds both sides of a merge, and SWMM itself rejects what
that leaves (duplicate names).

**`[DIVIDERS]` (dynamic wave).** `Name Elevation DivertedLink Type [parameters] [MaxDepth
InitDepth SurDepth Aponded]`, where `Type` is `OVERFLOW` (no parameters), `CUTOFF` (the cutoff
flow), `TABULAR` (a diversion curve of `[CURVES]`, type `DIVERSION`, which is accepted and not
used) or `WEIR` (`Qmin`, `dhMax`, `Cweir`); the four optional depth fields follow the
type-specific ones (`node.c:1113-1210`). Under `DYNWAVE` SWMM routes a divider as an ordinary
junction: the diversion (`divider_getOutflow`) is reached through `link_getInflow` and
`conduit_getInflow`, which dynamic-wave routing calls only for links that are not conduits
(`dynwave.c:440`) and then through the link's own inflow rule, and the divider-degree check is
`validateTreeLayout`'s, for `STEADY` and `KINWAVE` only. The reader makes it a junction with the
same `MaxDepth`, `InitDepth`, `SurDepth` and `Aponded` rules. What SWMM's `divider_validate`
(`node.c:1216-1247`) rejects under any routing is refused with the line: a diverted link that is
missing (`*`), unknown or not attached to the divider, and a `WEIR` divider with a non-positive
`dhMax` or `Cweir` or a `Qmin` above `Cweir dhMax^1.5`. Under `STEADY` or `KINWAVE`, where SWMM
diverts flow, `[DIVIDERS]` stays refused by name.

**Ignored** because no conduit or node ever references them: `[REPORT]`,
`[TAGS]`, `[MAP]`, `[COORDINATES]`, `[VERTICES]`, `[POLYGONS]`, `[SYMBOLS]`, `[LABELS]`,
`[BACKDROP]`, `[PROFILES]`; `[FILES]` is read for its `USE` lines (below). Any other section that carries content is refused, and
so are the options and fields listed below. Two are ignored only when nothing uses them:

- `[RAINGAGES]`, when the file has no `[SUBCATCHMENTS]`, `[HYDROGRAPHS]` or `[RDII]`, and any
  gage no subcatchment or unit hydrograph names;
- `[EVAPORATION]` and `[ADJUSTMENTS]`, when the file has no `[SUBCATCHMENTS]` (which read them,
  below) and nothing uses them: no open conduit, no conduit with a seepage rate, no storage unit
  with an evaporation factor or an exfiltration. Under dynamic-wave routing those losses read them
  (`CONSTANT`, `MONTHLY`, `TIMESERIES` with `RECOVERY`, and the `EVAPORATION` and `CONDUCTIVITY`
  adjustments; a `TEMPERATURE` or `FILE` source is refused naming its line; see
  [Evaporation and seepage losses](../applications/sewer.md#evaporation-and-seepage-losses));
  under `STEADY` or `KINWAVE` a rate that can be positive over an open conduit is refused naming
  the first open conduit.

**Options that change what is routed.** `IGNORE_ROUTING YES` and `FLOW_ROUTING NONE` (SWMM does
not route at all) and `SKIP_STEADY_STATE YES` (a step whose flows are within `SYS_FLOW_TOL` and
`LAT_FLOW_TOL` of the last is skipped, `routing.c:386-392`, which stops a slow transient early)
are refused with the line number under every routing; `NO` is SWMM's default and changes
nothing. The other `IGNORE_*` options and `SLOPE_WEIGHTING` do not reach the routed flows.

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

**Ponding.** `ALLOW_PONDING` takes `YES` or `NO` (SWMM's keyword test; anything else is refused
with the line) under every routing. With `YES` under `DYNWAVE`, a junction or divider with a
positive `Aponded` (the sixth `[JUNCTIONS]` field, `Manhole.ponded_area`) ponds instead of
flooding (see [Ponding](../applications/sewer.md#ponding)); under `STEADY` or `KINWAVE` SWMM
re-releases the ponded water into the next step's flow, so `ALLOW_PONDING YES` with any positive
`Aponded` is refused by name there. An `Aponded` is otherwise read and unused.

**Dynamic wave, options refused.** With `DYNWAVE` the reader refuses, with the line number: a
negative `LENGTHENING_STEP` or `MINIMUM_STEP`, and a `VARIABLE_STEP` outside 0 to 2.
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
reader reverses); the seepage rate (the sixth field, in/hr or mm/hr, read as
`Pipe.seepage` in m/s) is a loss of water under dynamic-wave routing (see
[Evaporation and seepage losses](../applications/sewer.md#evaporation-and-seepage-losses)) and
refused with the line number under `STEADY` or `KINWAVE` when positive (a negative one does
nothing in SWMM). Under `STEADY` or `KINWAVE` a non-zero `InitDepth`, `InitFlow` or
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
(m; `Z` a side slope, run over rise). The evaporation factor `Fevap` is `Manhole.evaporation_factor`;
the exfiltration after it is `Ksat` alone (in/hr or mm/hr) or the suction head (in or mm),
`Ksat` and `IMDmax` (`Exfiltration`; a `Ksat` of 0 is no exfiltration, as in SWMM; both are
[losses of water](../applications/sewer.md#evaporation-and-seepage-losses)), and an
exfiltrating `PARABOLIC` unit is refused (SWMM 5.2.4 leaves its bottom and bank areas unset).
A `[CURVES]` curve of any other type and a curve name that is not in `[CURVES]` are refused with
the line number. A
negative `MaxDepth`, `InitDepth` or `SurDepth` on a storage line is refused too, which goes
beyond SWMM's own input checks for storage units. `[STORAGE]` under
`STEADY` or `KINWAVE` is refused by name: SWMM routes a storage unit by its volume there,
which is not modelled.

**Orifices, weirs and outlets (dynamic wave).** Each becomes an `Orifice`, `Weir` or
`Outlet` of `SewerNetwork.orifices`, `.weirs` or `.outlets`, between two nodes:

- `[ORIFICES] Name N1 N2 SIDE|BOTTOM CrestHt Cd [Gated] [CloseTime]` with an `[XSECTIONS]`
  opening `CIRCULAR` (diameter) or `RECT_CLOSED` (height, width). `CloseTime` (hours) is the
  time a control rule takes to move the orifice through its full opening (0: at once).
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
`[CURVES]` of a type other than `STORAGE`, `RATING`, `TIDAL`, `SHAPE`, `PUMP1`-`PUMP5` and
`CONTROL`, the regulator sections under
`STEADY` or `KINWAVE` (SWMM routes them by the inflow alone there), and regulators with
`LINK_OFFSETS ELEVATION` (SWMM then sets each end's offset from its own node).

**Pumps (dynamic wave).** `[PUMPS] Name N1 N2 [Curve|*] [ON|OFF] [Startup] [Shutoff]` becomes a
`Pump` of `SewerNetwork.pumps`: `*` or no curve is an ideal pump, otherwise the curve is a
`[CURVES]` entry of type `PUMP1`-`PUMP5` (SWMM's ERR_NO_CURVE otherwise), converted to SI (a
PUMP1 curve's x values are a volume, converted with SWMM's volume factor, m3 per ft3 0.02832),
the startup and shutoff depths in the file's length unit (default 0: none; SWMM's
ERR_PUMP_LIMITS when the startup depth is positive and not above the shutoff depth). Refused
with the line number: pumps under `STEADY` or `KINWAVE`, an unknown node or curve, a status
other than ON or OFF, a negative depth.

**Control rules (dynamic wave).** `[CONTROLS]` becomes `SewerNetwork.controls`, a
`ControlSet` (`RULE`, `IF`, `AND`, `OR`, `THEN`, `ELSE`, `PRIORITY`; premises on `NODE
DEPTH|HEAD|VOLUME|INFLOW`, `LINK|CONDUIT|PUMP|ORIFICE|WEIR|OUTLET FLOW|STATUS|SETTING|TIMEOPEN|
TIMECLOSED` and `SIMULATION TIME|DATE|CLOCKTIME|DAYOFYEAR|DAY|MONTH`, against a number or another
variable; actions on a pump's `STATUS` and `SETTING` and an orifice's, weir's and outlet's
`SETTING`, as a number, `CURVE` (a `[CURVES]` entry of type `CONTROL`), `TIMESERIES` or `PID`)
with the `RULE_STEP` option. The values stay in the file's units, which is how SWMM compares
them. Refused with the line number: controls under `STEADY` or `KINWAVE`, `VARIABLE` and
`EXPRESSION`, rain-gauge premises, the attributes MAXDEPTH, FULLFLOW, FULLDEPTH, LENGTH, SLOPE
and VELOCITY and a link's DEPTH, conduit actions, and everything SWMM itself rejects.

**Outfalls.** `FREE`, `NORMAL`, `FIXED` (with its stage, m), `TIDAL` (with the name of its
`[CURVES]` TIDAL curve, read as hours and stage elevations) and `TIMESERIES` (with the name of
its series) are read. A `TIMESERIES` outfall takes its stage from its `[TIMESERIES]` series
(`Outfall.stage_series`: times in seconds since the start, values converted from the file's
length unit to m), which is the default of the `outfall_stage` argument of `build_model` /
`simulate_dynamic_wave`; naming the outfall there overrides it. SWMM reads the series at the
step's END time with the series held at its first value before it and its last after it
(`table_tseriesLookup` with `extend` TRUE, node.c:1454-1458), unlike an inflow series, which
is 0 outside its range. An outfall may sit
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
refused by name; `USE RAINFALL` only feeds `FILE` rain gages, which are refused.

**Refused by name**, because skipping them would silently change the physics:
`[STORAGE]`, `[ORIFICES]`, `[WEIRS]`, `[OUTLETS]`, `[PUMPS]`, `[CONTROLS]` and `[DIVIDERS]` under
`STEADY` or `KINWAVE`, `[STREETS]`, `[CURVES]` of any type but `STORAGE`, `RATING`, `TIDAL`, `SHAPE`,
`DIVERSION`, `PUMP1`-`PUMP5` and `CONTROL`, `[LID_USAGE]`, and any other section carrying
content.

**Hydrology** (a file with `[SUBCATCHMENTS]`, `[HYDROGRAPHS]` or `[RDII]`; `apps/sewer/inp_hydrology.py`, running in
`apps/sewer/runoff.py`; [the runoff](../applications/sewer.md#subcatchment-runoff-from-a-swmm-file)
is SWMM 5.2.4's own computation). SWMM reads the file in order and so does the reader.

- `[RAINGAGES]` ``Name Format Interval SCF TIMESERIES Series``: Format `INTENSITY`, `VOLUME` or
  `CUMULATIVE`; Interval decimal hours or `H:MM`; the series' values are in inches or millimetres
  (per hour for `INTENSITY`) by the file's unit system. A `FILE` source is refused.
- `[SUBCATCHMENTS]` ``Name Gage Outlet Area %Imperv Width %Slope CurbLen [Snowpack]``: the outlet
  is a node or another subcatchment (one that is both is refused, as SWMM's `ERR_SUBCATCH_OUTLET`);
  area in acres or hectares, width in ft or m. Runoff to an outfall is dropped (an outfall takes
  no lateral inflow under `DYNWAVE`).
- `[SUBAREAS]` ``Name N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo [PctRouted]``, after the
  subcatchment's line. A subcatchment with no line sheds nothing (SWMM's zeroed subareas).
- `[INFILTRATION]` ``Name p1 p2 ... [Method]``: `HORTON` and `MODIFIED_HORTON` ``MaxRate MinRate
  Decay DryTime [MaxInfil]``, `GREEN_AMPT` and `MODIFIED_GREEN_AMPT` ``Suction Ksat IMDmax``,
  `CURVE_NUMBER` ``CN Conductivity DryTime``; the model of `[OPTIONS]` `INFILTRATION` (default
  `HORTON`) unless the line ends with a method. SWMM's range checks are refused by name.
- `[EVAPORATION]` `CONSTANT`, `MONTHLY`, `TIMESERIES`, `RECOVERY` and `DRY_ONLY` (the
  subcatchments' alone: open conduits and storage units evaporate while it rains); `TEMPERATURE`
  and `FILE` (a climate file) are refused.
- `[ADJUSTMENTS]` `EVAPORATION`, `RAINFALL`, `CONDUCTIVITY`, `N-PERV`, `DSTORE`, `INFIL`
  (the last three count only for a subcatchment defined above them); `TEMPERATURE` is ignored.
- `[HYDROGRAPHS]` ``Name Gage`` and ``Name Month Response R T K [Dmax Drecov Dinit]`` (Month `ALL`
  or `JAN`..`DEC`, Response `SHORT`, `MEDIUM` or `LONG`, T in hours, K the recession ratio; the older
  nine-number line is read too) and `[RDII]` ``Node UnitHydrograph SewerArea`` (acres, or hectares in
  SI units): the rainfall-dependent inflow of the unit hydrographs, added to the lateral inflow of
  the node (`apps/sewer/inp_rdii.py`, `rdii.py`; [RDII](../applications/sewer.md#rdii-from-a-swmm-file)).
  A file with only RDII has no subcatchments. A gage a unit hydrograph names counts as used.
- `[OPTIONS]` `WET_STEP`, `DRY_STEP`, `START_*`, `END_*` (SWMM's total duration),
  `INFILTRATION`, `IGNORE_RAINFALL`, `IGNORE_RDII`, `IGNORE_SNOWMELT`, `IGNORE_GROUNDWATER` and
  `IGNORE_QUALITY`.
  Under `DYNWAVE` the routing step is lowered to the effective `WET_STEP`, which a gage whose
  recording interval is shorter lowers first (`gage_validate`, `project_validate`).

Refused by name: `[LID_USAGE]`, `[GROUNDWATER]` (without `IGNORE_GROUNDWATER YES`), a snow
pack on a subcatchment (without `IGNORE_SNOWMELT YES`), `[LANDUSES]`, `[COVERAGES]`, `[BUILDUP]`,
`[WASHOFF]`, `[LOADINGS]` and a rain concentration (`Crain`) while the file has pollutants
(without `IGNORE_QUALITY YES`), an outfall's `Route To`, a unit hydrograph with no gage line that a
node uses, response ratios beyond SWMM's limits, a pollutant's RDII concentration (`Crdii`) while a
node has RDII (without `IGNORE_QUALITY YES`) and `[FILES]` `USE RDII`.
`[SNOWPACKS]`, `[TEMPERATURE]`, `[AQUIFERS]` and `[LID_CONTROLS]` that nothing uses change
nothing, in SWMM and here. The runoff reaches the model as `SewerNetwork.inflows`
(`runoff.RunoffInflows`) and `SewerNetwork.hydrology`.

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
with clean water); the other pollutant fields feed only runoff quality, groundwater, the quality of RDII and
dry-weather inflow, which are refused or absent.

**Not supported**, which rules out most published SWMM examples as they stand:

- groundwater, LID, snowmelt, pollutant buildup and washoff and rain files (the surface
  runoff and RDII themselves are read: Hydrology above);
- the `DUMMY` and `STREET` cross sections, and conduit seepage and evaporation under `STEADY` or
  `KINWAVE`;
- dividers under `STEADY` or `KINWAVE`, roadway weirs, weir coefficient curves, named variables
  and expressions in control rules; pumps, control rules, orifices, weirs, outlets and storage
  units under `STEADY` or `KINWAVE`, and
  water quality together with evaporation, seepage or exfiltration;
- ponding together with water quality or headspace air, and ponding under `STEADY` or `KINWAVE`.
  Both `SURCHARGE_METHOD`s model surcharge, and a junction without a ponded area floods above
  `MaxDepth` (extended to its crown) plus `SurDepth`; the water that cannot enter is lost, as in
  SWMM without ponding.

The complete list of what the dynamic wave does not support, and of what it implements without
a live SWMM comparison, is under [Limitations](../applications/sewer.md#limitations).

**Slope** follows SWMM's own definition $S_0 = dy/dx$ with $dx = \sqrt{L^2 - dy^2}$ — the 3-D
chord, not the naive $dy/L$. Under `STEADY` and `KINWAVE`, zero or adverse fall is refused, as
is a fall $\ge$ length. Under `DYNWAVE` the reader follows SWMM's `conduit_getSlope`: the fall is
floored at SWMM's `MIN_DELTA_Z` (0.001 ft, 0.0003048 m), `MIN_SLOPE` floors the slope, and a
conduit is reversed only when it is adverse, so a flat conduit keeps the orientation of the
file. Conduit offsets are kept (`Pipe.inlet_offset`, `Pipe.outlet_offset`, m above the end
node's invert) and swap with the ends on a reversal. A negative offset is 0, as in SWMM
(a warning there): `link_getOffsetHeight` (link.c:498-502) clamps an elevation below the node's
invert under `LINK_OFFSETS ELEVATION`, and `conduit_validate` (link.c:1057-1067) a negative
depth offset under `DEPTH`, for every routing; the slope comes from the clamped end
elevations (link.c:1269). Regulators already read a negative offset as 0.

```python
from noodl.apps.sewer import read_swmm_inp, build_model

net, inflows, pollutants = read_swmm_inp("tree_steady.inp")
model, state, drivers = build_model(net)
```
