# SWMM `.inp`

Reading a SWMM input file produces a model for [sewers](../applications/sewer.md). It shares its
lexical tokenizer (`noodl.apps.inpfile`) with the EPANET reader; each format's own section names,
column layouts and units are read by `apps/sewer/inp.py`.

**Read:** `[TITLE]`, `[OPTIONS]`, `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`,
`[INFLOWS]`, `[POLLUTANTS]`.

**Routing.** `FLOW_ROUTING` is `STEADY`, `KINWAVE` or `DYNWAVE`, read as the physical names
`steady`, `kinematic_wave` and `dynamic_wave` (`SewerNetwork.routing`). `DYNWAVE` also reads
`ROUTING_STEP` (seconds or `HH:MM:SS`), `INERTIAL_DAMPING` (`NO`/`PARTIAL`/`FULL`),
`NORMAL_FLOW_LIMITED` (`SLOPE`/`FROUDE`/`BOTH`/`NO`), `SURCHARGE_METHOD` (`EXTRAN` or `SLOT`), `MIN_SURFAREA`,
`HEAD_TOLERANCE`, `MAX_TRIALS` and `MIN_SLOPE` into `DynamicWaveOptions`; `HEAD_TOLERANCE`
and `MAX_TRIALS` are SWMM's own iteration controls, recorded but not used.

**Dynamic wave, what is refused for now.** With `DYNWAVE` the reader refuses, with the line
number: `VARIABLE_STEP` above 0 (set it to 0), `LENGTHENING_STEP` above 0, `ALLOW_PONDING YES`,
a junction `InitDepth` other than 0, a conduit `InitFlow` or `MaxFlow`
other than 0, flap-gated outfalls and `TIDAL`/`TIMESERIES` outfalls.

**Outfalls.** `FREE`, `NORMAL` and `FIXED` (with its stage, m) are read.

**Barrels.** A conduit with more than one barrel, or a culvert code, is refused for every
routing; it used to be ignored silently.

**Constraints:** `FLOW_UNITS` must be `CMS` — the reader is SI-native and does not convert.
`LINK_OFFSETS` must be `DEPTH` or `ELEVATION`. Only `CIRCULAR` cross-sections. Only constant
`FLOW` baselines and constant `CONCENTRATION` pollutant baselines; a time-series inflow is
refused, because time variation belongs in the drivers.

**Refused by name**, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]` with non-constant area, `[PUMPS]`, `[WEIRS]`, `[ORIFICES]`, `[OUTLETS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]`, `[TIMESERIES]`,
`[LID_USAGE]`, and any other section carrying content.

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
- storage units, pumps, weirs, orifices and outlets;
- dynamic wave with a variable time step or ponding; a junction filling past its full depth
  (MaxDepth, extended to its crown) stops the run, as flooding is not modelled yet, and
  `SurDepth` is not read. Both `SURCHARGE_METHOD`s model surcharge.

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
