# EPANET `.inp`

Reading an EPANET input file produces a model for [water
distribution](../applications/water.md). It shares its lexical tokenizer
(`noodl.apps.inpfile`) with the SWMM reader; each format's own section names, column layouts and
units are read by `apps/water/inp.py`.

**Read:** `[JUNCTIONS]`, `[RESERVOIRS]`, `[TANKS]`, `[PIPES]`, `[PUMPS]`, `[VALVES]`,
`[DEMANDS]`, `[PATTERNS]`, `[CURVES]`, `[CONTROLS]`, `[OPTIONS]`, `[TIMES]`.

**Ignored** because nothing in the hydraulics references them: `[TITLE]`, `[REPORT]`,
`[COORDINATES]`, `[VERTICES]`, `[LABELS]`, `[BACKDROP]`, `[TAGS]`, `[ENERGY]`, `[END]`, and the
quality sections (a quality run is configured through `build_model(quality=...)` instead).

**Refused by name:** `[RULES]`; `[EMITTERS]`; `[STATUS]` with content; time-based controls (only
`LINK <id> OPEN|CLOSED IF NODE <tank> BELOW|ABOVE <level>` is read); PRV, PSV, PBV and GPV valves;
a constant-**power** pump; a pump with a speed pattern; a tank with a volume curve; a `HEADLOSS`
other than H-W or D-W; a closed pipe or check valve.

A constant-power pump is refused with a specific reason: the head-flow relation EPANET uses
internally for one is not stated in the manual, so implementing it would be a guess.

**Units** follow `[OPTIONS] UNITS`: `CFS GPM MGD IMGD AFD` are US, `LPS LPM MLD CMH CMD` are SI,
and everything is converted to SI on read. Unrecognised `[OPTIONS]` lines are recorded verbatim in
`notes["unrecognised_options"]` rather than dropped.

```python
from noodl.apps.water import read_epanet_inp, build_model, water_steady

net = read_epanet_inp("twoloop_si.inp")
model, state, drivers = build_model(net)
final = water_steady(model, state, drivers)

# final["water.phi"] is head (m) at every node
# final["water.q"] is flow (m3/s) in every link
```
