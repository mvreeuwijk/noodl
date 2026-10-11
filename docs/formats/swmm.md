# SWMM `.inp`

Reading a SWMM input file produces a model for [sewers](../applications/sewer.md). It shares its
lexical tokenizer (`noodl.apps.inpfile`) with the EPANET reader; each format's own section names,
column layouts and units are read by `apps/sewer/inp.py`.

**Read:** `[TITLE]`, `[OPTIONS]`, `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`,
`[LOSSES]` (the flap gate only), `[INFLOWS]`, `[POLLUTANTS]`, and under `DYNWAVE`
`[STORAGE]`, `[ORIFICES]`, `[WEIRS]` and `[OUTLETS]` with the `[CURVES]` of type `STORAGE`
and `RATING`.

**Routing.** `FLOW_ROUTING` is `STEADY`, `KINWAVE` or `DYNWAVE`, read as the physical names
`steady`, `kinematic_wave` and `dynamic_wave` (`SewerNetwork.routing`). `DYNWAVE` also reads
`ROUTING_STEP`, `WET_STEP` (default 300 s) and `MINIMUM_STEP` (seconds or `HH:MM:SS`),
`VARIABLE_STEP` (the Courant factor, 0 to 2; absent means SWMM's default 0.75), `INERTIAL_DAMPING` (`NO`/`PARTIAL`/`FULL`),
`NORMAL_FLOW_LIMITED` (`SLOPE`/`FROUDE`/`BOTH`/`NO`), `SURCHARGE_METHOD` (`EXTRAN` or `SLOT`), `MIN_SURFAREA`,
`HEAD_TOLERANCE`, `MAX_TRIALS`, `MIN_SLOPE` and `LENGTHENING_STEP` into `DynamicWaveOptions`; `HEAD_TOLERANCE`
and `MAX_TRIALS` are SWMM's own iteration controls, recorded but not used. As SWMM does,
the reader lowers `ROUTING_STEP` to `WET_STEP` when it is larger (a file that omits
`VARIABLE_STEP` runs the 0.75 variable step, so give `duration=` rather than `dt=` to
`simulate_dynamic_wave`; `DynamicWaveOptions()` built in code defaults to the fixed step).

**Dynamic wave, what is refused for now.** With `DYNWAVE` the reader refuses, with the line
number: `ALLOW_PONDING YES`, a negative `LENGTHENING_STEP` and `TIDAL`/`TIMESERIES` outfalls.
The conduit lengthening uses the file's own `ROUTING_STEP`, as SWMM validates its conduits
before it lowers the step to `WET_STEP`.

**Initial conditions and limits (dynamic wave).** The junction `InitDepth` (the fourth
`[JUNCTIONS]` field) is `Manhole.initial_depth`; the conduit `InitFlow` and `MaxFlow` (the
eighth and ninth `[CONDUITS]` fields) are `Pipe.initial_flow` (a total over the barrels) and
`Pipe.max_flow` (a limit on each barrel); a conduit flap gate (`[LOSSES]` fifth field YES) is
`Pipe.flap_gate`, and an outfall's `Gated` YES is `Outfall.flap_gate`. A conduit the reader
reverses (adverse, as SWMM does) has its `InitFlow` negated and its gate recorded as
reversed. The loss coefficients and the seepage rate of `[LOSSES]` must be 0 (refused with
the line number otherwise). Under `STEADY` or `KINWAVE` a non-zero `InitDepth`, `InitFlow` or
`MaxFlow`, a flap gate and a gated outfall are refused by name.

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
closed shapes `CYLINDRICAL L W -`, `CONICAL L W Z`, `PARABOLOID L W H` and `PYRAMIDAL L W Z`
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
`[CURVES]` of a type other than `STORAGE` and `RATING`, the regulator sections under
`STEADY` or `KINWAVE` (SWMM routes them by the inflow alone there), and regulators with
`LINK_OFFSETS ELEVATION` (SWMM then sets each end's offset from its own node).

**Outfalls.** `FREE`, `NORMAL` and `FIXED` (with its stage, m) are read. An outfall may sit
at the end of a conduit or of a regulator (exactly one link, SWMM's ERR_OUTFALL).

**Barrels.** A conduit with more than one barrel is read under `DYNWAVE` (1 to 127; the
flows, volumes and surface areas of the pipe are totals) and refused with the line number under
`STEADY` and `KINWAVE`; a culvert code is refused for every routing. Both used to be ignored
silently.

**Constraints:** `FLOW_UNITS` must be `CMS` — the reader is SI-native and does not convert.
`LINK_OFFSETS` must be `DEPTH` or `ELEVATION` (`DEPTH` with regulators). Only `CIRCULAR`
conduit cross-sections (regulator openings above). Only constant
`FLOW` baselines and constant `CONCENTRATION` pollutant baselines; a time-series inflow is
refused, because time variation belongs in the drivers.

**Refused by name**, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]`, `[ORIFICES]`, `[WEIRS]` and `[OUTLETS]` under `STEADY` or `KINWAVE`, `[PUMPS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]` of any type but
`STORAGE` and `RATING`, `[TIMESERIES]`, `[LID_USAGE]`, and any other section carrying
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

**Not supported**, which rules out most published SWMM examples as they stand:

- flow units other than CMS;
- runoff (`[SUBCATCHMENTS]` and everything that feeds them) and dry-weather patterns;
- non-circular shapes;
- pumps, dividers, roadway weirs, weir coefficient curves and control rules (regulator
  settings); orifices, weirs, outlets and storage units under `STEADY` or `KINWAVE`, and
  storage evaporation or exfiltration;
- dynamic wave with ponding. Both `SURCHARGE_METHOD`s model
  surcharge, and a junction floods (without ponding) above `MaxDepth` (extended to its
  crown) plus `SurDepth`.

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
