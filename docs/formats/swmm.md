# SWMM `.inp`

Reading a SWMM input file produces a model for [sewers](../applications/sewer.md). It shares its
lexical tokenizer (`noodl.apps.inpfile`) with the EPANET reader; each format's own section names,
column layouts and units are read by `apps/sewer/inp.py`.

**Read:** `[TITLE]`, `[OPTIONS]`, `[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]`,
`[INFLOWS]`, `[POLLUTANTS]`.

**Constraints:** `FLOW_UNITS` must be `CMS` — the reader is SI-native and does not convert.
`LINK_OFFSETS` must be `DEPTH` or `ELEVATION`. Only `CIRCULAR` cross-sections. Only constant
`FLOW` baselines and constant `CONCENTRATION` pollutant baselines; a time-series inflow is
refused, because time variation belongs in the drivers.

**Refused by name**, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]` with non-constant area, `[PUMPS]`, `[WEIRS]`, `[ORIFICES]`, `[OUTLETS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]`, `[TIMESERIES]`,
`[LID_USAGE]`, and any other section carrying content.

**Slope** follows SWMM's own definition $S_0 = dy/dx$ with $dx = \sqrt{L^2 - dy^2}$ — the 3-D
chord, not the naive $dy/L$. Zero or adverse fall is refused, as is a fall $\ge$ length.

```python
from noodl.apps.sewer import read_swmm_inp, build_model

net, inflows, pollutants = read_swmm_inp("tree_steady.inp")
model, state, drivers = build_model(net)
```
