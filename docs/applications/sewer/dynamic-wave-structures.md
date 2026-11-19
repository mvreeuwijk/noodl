# Dynamic-wave structures and cross sections

How to model storage units, losses to evaporation and seepage, orifices, weirs, outlets, pumps,
control rules and every SWMM cross section under dynamic-wave routing. Each section says what
is modelled, how SWMM's rule is reproduced and, where it exists, how it was compared with SWMM
([Verification: sewers](../../reference/verification/sewer.md)). The run itself is described in
[Dynamic-wave runs](dynamic-wave.md); the junction and conduit equations in
[Dynamic-wave physics](physics-dynamic-wave.md).

## Storage units

A storage unit (`[STORAGE]`, `storage_unit`, `Manhole.storage`) follows SWMM's storage
node (`node.c`, `dynwave.c`). A unit is a junction whose surface area adds its curve's area
at the previous pass's depth (`node_getSurfArea`) to the attached conduit ends' (SWMM does
not zero a conduit's end area at a storage node, only a regulator's), so it enters the same
trapezoidal equation. Its area curve is `FUNCTIONAL` `A = a y^b + c`, `TABULAR` (linear
between the curve's points, linear from zero below the first, the last segment's slope,
floored at 0, beyond the last; `table_lookupEx`) or one of SWMM's closed shapes (cylinder,
cone frustum, paraboloid, pyramid frustum, each a quadratic in depth). Its volume, reported as
`"sewer.storage_volume"` (m³ per junction, 0 at a plain junction; `record=` in
`simulate_dynamic_wave`), is SWMM's `storage_getVolume`: the integral of the area (the
end-area method for `TABULAR`, as `table_getStorageVolume`), held at the full volume at and
above the full depth, so `SurDepth` stores nothing. SWMM's storage rules differ from a
junction's in three places:

- its full depth is extended to the attached conduit crowns only when it has a `SurDepth`
  (`link.c:450-458`), so a unit with no `SurDepth` floods at its own `MaxDepth`, below its
  crown if that is lower;
- under EXTRAN it surcharges only with a `SurDepth`, and then above its full depth rather
  than its crown (`dynwave.c:683-687`); the surcharged form, the hold band and the flood
  wall are those of a junction with the full depth as its crown;
- it keeps its own `InitDepth`, 0 included, instead of the average of its conduits
  (`flowrout.c:375`).

A unit evaporates (`Fevap`, `Manhole.evaporation_factor`) and exfiltrates (`Manhole.exfiltration`)
as [Evaporation and seepage losses](#evaporation-and-seepage-losses) describes.

## Evaporation and seepage losses

Water leaves the network in three places besides the outfalls and the floods, all from the
evaporation rate and the conductivity adjustment of the file's climate
(`noodl.apps.sewer.losses`). They are modelled for the dynamic wave only; under STEADY or
KINWAVE routing a positive seepage rate or an evaporation rate that can be positive over an open
conduit is refused by name.

**Conduits** (`link.c::conduit_getLossRate`, link.c:1334-1399). From the mean of the step-start
link depth and the pass's, $d = \tfrac12 (y_\text{old} + y_\text{new})$, an *open* conduit
(`xsect_isOpen`) evaporates

$$ Q_e = T(d)\, L\, E $$

with $T$ the section's own top width (`xsect_getWofY`, not the crown-cut width of the momentum
update), $L$ the true length and $E$ the evaporation rate, while $E > 0$; and any conduit with a
`[LOSSES]` Seepage rate $s$ (in/hr or mm/hr) seeps

$$ Q_s = s\, W(\min(d, y_{w\max}))\, L\, f_K $$

with $W$ the same width at the depth limited to the section's first widest point $y_{w\max}$ (a
fraction of the height for the standard shapes, `wMax` for `RECT_CLOSED`, the first maximum of
the width table of a `CUSTOM` or `IRREGULAR` section) and $f_K$ the month's `[ADJUSTMENTS]
CONDUCTIVITY`. Neither exists at a depth of `FUDGE` (1e-4 ft) or less or on the dry exit of the
conduit update, and their sum is scaled down so that it does not exceed the link's volume over
the step, $V/\Delta t$ (all barrels' volume against one barrel's rate, as SWMM compares them).
The rate enters the momentum update as $\Delta q_6 = (Q_e + Q_s)\, 2.5\, \Delta t\, v / L$
(`dwflow.c:236`, added to the numerator with $\Delta q_3$, $\Delta q_4$) and is taken from the
end nodes, times the barrels, half from each, or all from the one that is not an outfall
(`dynwave.c:555-572`). The depth, volume and velocity are those of the previous trial in SWMM and
of the previous pass here, the same at the fixed point; the step-start depth is a new carried key,
`"sewer.pipe_depth"` (SWMM's `Link.oldDepth`), present only in a network with conduit losses.

**Storage units** (`node.c::storage_getLosses`, node.c:1052-1100; `exfil.c`). Once per routing step,
from the step-start depth $y$, surface area $A(y)$ and volume $V$ (routing.c:363-365): evaporation
$F_\text{evap}\, E\, A$ (just $F_\text{evap} E$, SWMM's unit slip, when $V \le$ `FUDGE`), and the
exfiltration through the bottom and the banks. With a single Ksat, or `IMDmax` 0, the rate is Ksat
$\times f_K$ over the bottom area plus the bank area (the surface area beyond the bottom, from the
depth the banks start at). With a positive `IMDmax` it is SWMM's modified Green-Ampt model
(`runoff._GreenAmptState`, with its state kept from step to step, reset at the start of every
`simulate_dynamic_wave` call) at the depth of the water for the bottom and, for the banks,
half the depth above their foot (or, above the last growth of a tabular curve, that depth). The
bottom and bank areas are `exfil_initState`'s: for a `TABULAR` curve the area at depth 0 and the
depths at which the curve is flat and then grows; for the other shapes the constant `a0` and banks
from the floor up. SWMM reads that `a0` in the file's own length unit as ft$^2$, which for an SI
file is the number of m$^2$ taken as ft$^2$; the reader reproduces it (`Exfiltration.us_units`).
The total is capped at $V / \Delta t$, where $V$ is SWMM's `storage_getVolume`: the curve's volume
in the file's unit divided by `UCF(VOLUME)`, which is 1 for US units but 0.02832 (not the true
0.3048$^3$) for SI, so an SI file's cap is 1.1e-4 larger than the unit's true volume. While the cap
binds (a unit filling faster than it exfiltrates, the first steps of a run) that slip moved the
depth by 2e-7 m against SWMM until it was reproduced (`StorageLosses.volume_per_m3`).

**The climate** is `climate_setState` as SWMM's routing steps see it. With subcatchments, the
state at the start of the last runoff step taken before the routing step
(`runoff.RunoffModel`, shared with the lateral-inflow driver); without any, the state at the start
of the routing step (`swmm5.c:556`), where a file with no `[SUBCATCHMENTS]` still has its
`[EVAPORATION]` (`CONSTANT`, `MONTHLY`, `TIMESERIES` with `RECOVERY`) and `[ADJUSTMENTS]` read
(`SewerNetwork.climate`, a `Hydrology` with no gages or subcatchments; `read_climate`).
**`DRY_ONLY` is the subcatchments' alone** (`subcatch.c:685`): a conduit or storage unit
evaporates while it rains, and the parity tests check that against SWMM. The Green-Ampt
model of a storage unit reads SWMM's `InfilFactor`, the last subcatchment's (its pattern or the
month's conductivity adjustment), and the recovery factor.

`simulate_dynamic_wave` sets three drivers each step from this clock: `"evap_rate"` (m/s),
`"hydcon"` and `"storage_loss"` (m$^3$/s per junction, `DynamicWaveHydraulics.storage_loss`); a
network with losses built by `build_model` starts them at the climate of t = 0, and a closure
called without them raises naming the driver. Only a network that loses water pays for any of it:
a circular, loss-free network builds none of the machinery.

**Gradients.** The conduit rates are functions of the pass's depths, volumes and velocities and the
storage rates of the step-start depth and volume, all torch operations, so a gradient with
respect to an inflow, a roughness or a surface runs through the loss (the head solve differentiates
the converged pass, which contains it); the rate of a unit with a positive `IMDmax` also, through
the depth, but not through its infiltration state. The evaporation rate, the conductivity
adjustment and the Green-Ampt state are plain floats of the climate and carry no gradient.

**Water quality.** Conduit seepage takes its dissolved mass with it and evaporation does not, as
in SWMM (`qualrout.c`: a seeping conduit's water leaves at the conduit's concentration, an
evaporating one's concentrates; probed against SWMM 5.2.4, 100 mg/L through a seeping conduit
stays 100 mg/L and through an evaporating one rises to 113 and 132). The conduit's seepage
(`ConduitTerms.seep_loss_rate`, after the volume cap) joins a negative lateral
inflow in the withdrawal of a junction (inside `DynamicWaveHydraulics`), which is an
outflow of the transport step itself, the removal rate `"water_quality.removal"` (1/s: the
withdrawal over the end-of-step volume for the backward Euler scheme, over the mean volume for the
trapezoidal one), so it leaves at the local concentration; a uniform concentration stays uniform
while the water and the mass fall. Evaporation only lowers the transport volume, which raises
the concentration. Quality needs circular conduits, which are closed and so do not evaporate: the
evaporation side is covered by unit tests. The removal has no cap, so a
step whose seepage exceeds a junction's step-start contents (the first steps of a dry start)
still conserves a uniform concentration.

**Not modelled, refused by name.** Evaporation computed from temperature or read from a climate file
(`TEMPERATURE`, `FILE`); an exfiltrating `PARABOLOID` storage unit, for which SWMM 5.2.4's
`exfil_initState` has no case and leaves the bottom and bank areas unset; water quality with any
of the storage units' losses; seepage under STEADY or KINWAVE routing. A conductivity adjustment applies to the
Green-Ampt exfiltration through SWMM's `InfilFactor` (the last subcatchment's), which the clock
reads from the runoff model.

**Measured against SWMM** (every SWMM step
converged, targets 1e-7 m and 1e-6 relative + 1e-9 m$^3$/s): constant and monthly evaporation of
an open trapezoid and rectangle across a month boundary with an evaporation adjustment, `DRY_ONLY
YES` through a rain period, seepage of a circle, trapezoid, open and closed rectangle with a
conductivity adjustment, and storage evaporation, Ksat and Green-Ampt exfiltration of tabular,
functional, conical and cylindrical units. SWMM does not always converge with losses (a
free outfall conduit, or a seepage close to the inflow, leaves it oscillating between two
branches on a few steps, as without them); those steps have no reference. A *dry* junction that
is losing water (a net outflow, from a loss or a negative lateral inflow) sits a hair below its
invert on the stiff bed and moves the flow of the conduit above it by up to 4e-9 m$^3$/s while
it is dry: the same with or without a loss, and the depth is reported clamped at 0 as SWMM's is.

## Orifices, weirs and outlets

Orifices, weirs and outlets (`[ORIFICES]`, `[WEIRS]`, `[OUTLETS]`; `Orifice`, `Weir`,
`Outlet`) are flow regulators between two nodes, each an edge of its own kind (`"orifice"`,
`"weir"`, `"outlet"`) in the `"water"` layer, whose flows follow the pipes' in `"water.q"`
(`model.link_names`). Their flow is SWMM's `link_getInflow` (`link.c`, transcribed in
`noodl.apps.sewer.structures`, in feet as SWMM computes it): the orifice equation with its
weir regime below the opening's critical height and Villemonte's submergence factor there,
the four weir formulas with end contractions, Villemonte's factor and, above the opening,
the equivalent orifice (`CanSurcharge`) or a head capped at the opening, the outlet's rating
on the depth above the crest (`/DEPTH`) or the head across it (`/HEAD`), each closed (no
flow) below its crest, with no water upstream, or against a flap gate; gated orifices and
weirs take SWMM's ARMCO head loss. A regulator never reverses: flow from its second node to
its first takes the first node's crest, as in SWMM. Within a pass a regulator is the
element `StructureLink`, ``q = c0 + g (dp - dp_fed)``: ``c0`` is SWMM's flow at the previous
pass's heads, ``dp_fed`` that pass's head difference, ``g >= 0`` SWMM's `dqdh` (an outlet,
which has none, a central difference in its upstream head). At the fixed point ``dp =
dp_fed``, so the flow is SWMM's own whatever ``g``, which only makes each pass a Newton-like
step. SWMM's under-relaxation of these flows is iteration only and is not modelled.
A regulator also enters the junctions as SWMM's does:

- an orifice gives each end half its equivalent surface area (`dynwave.c:503-524`; a side
  orifice ``W(depth) L``, a bottom one its opening's area, ``L`` an equivalent length of at
  least 200 ft), none at its downstream end while DN_CRITICAL (or upstream, UP_CRITICAL) and
  none at a storage unit; weirs and outlets give none;
- its two ends raise both nodes' crowns (node 1 by the crest plus the opening height, node 2
  by the crest height as given plus the opening height; an outlet's opening is SWMM's DUMMY
  height of 1e-6 ft), and it extends its first node's full depth to that crown (a bottom
  orifice does not), as `link_validate` does; a crest below the downstream invert is raised
  to it at the first node only;
- an outfall at the end of a regulator takes normal and critical depth 0, then its `FREE`,
  `NORMAL` or `FIXED` rule with the regulator's crest height as its offset;
- a regulator starts dry and counts, with depth 0, in its nodes' initial average depth.

Roadway weirs and weir coefficient curves are refused by name. Flow dividers (`[DIVIDERS]`) are
junctions under dynamic-wave routing, as in SWMM, and are refused by name under steady and
kinematic-wave routing, where SWMM diverts flow (see [SWMM `.inp`](swmm.md)).

## Pumps

A pump (`[PUMPS]`; `Pump`) is a fourth `StructureLink` kind, `"pump"`, whose column follows the
outlets' in `"water.q"` (`model.link_names`). Its flow is SWMM's `pump_getInflow`
(`link.c:1548-1634`, `noodl.apps.sewer.pumps`): the curve at the previous pass's inlet state,
times the pump's setting, never negative. The curve kinds are PUMP1 (a step in the inlet node's
volume: the flow of the first curve point whose volume exceeds it), PUMP2 (the same in the inlet
depth), PUMP3 (continuous in the head DELIVERED, outlet minus inlet head, clamped at 0), PUMP4
(continuous in the inlet depth) and PUMP5 (PUMP3 with the head divided by the squared speed
setting), and an *ideal* pump (`*` for the curve) takes its inlet node's inflow. `dqdh` (the
slope that shapes a pass) is the negated curve slope for PUMP3 and PUMP5, none for the others;
the fixed point does not depend on it.

SWMM then limits the flow to the water at the inlet (`dynwave.c::getModPumpFlow`,
`pumps.mod_pump_flow`): at a storage unit and for a PUMP1 the outflow is capped at the node's
inflow plus its volume over the step; for PUMP2-4 at a junction, where the step would leave a
depth that is not positive the pump takes the inflow. These read the inlet node's inflow,
outflow and surface area **so far in the trial**, and SWMM visits the conduits first, then the
other links in input order, so a regulator or pump listed after the pump does not count yet.
`pumps.PumpStation` reproduces that (the links' file order is the line number of the entry).
An ideal pump also takes the node's overflow in SWMM; a run in which an ideal pump's inlet
floods is refused by name (RuntimeError), not simulated differently. A PUMP1 curve gives a plain
junction inlet its largest volume as its full volume, linear in the depth (`link.c:1521-1527`).

Status ON or OFF, the startup and shutoff depths (the pump switches OFF below the shutoff
depth and ON above the startup depth of its inlet, `link_setTargetSetting`) and the speed
`setting` are the pump's discrete state (below). Pumps need at least one conduit in the
network, like any dynamic-wave network, and an outfall at a pump's end gets normal and critical
depth 0 as at any regulator.

A stepped curve (PUMP1, PUMP2) crossed by a rising or falling inlet can have no fixed point
inside a step: the pump flow depends on the depth, the depth on the pump flow, and the two
branches never agree. SWMM then oscillates until its trials run out (it reports the step as not
converging) and noodl's iteration does not converge either. That is the equations, not the
solver, and the verification fixtures avoid it.

## Link settings and control rules

With pumps or `[CONTROLS]` the closure carries each regulator's and pump's setting as the
state keys `"sewer.setting"` (SWMM's `Link.setting`: a pump's speed, an orifice's, weir's or
outlet's fraction open; `hyd.setting_names` names the columns), `"sewer.target_setting"`,
`"sewer.time_last_set"` (the DateTime it was last opened or closed, for TIMEOPEN and
TIMECLOSED) and `"sewer.orifice_hold"`. They are discrete state between steps: the pump depth
limits, the rules and the orifice opening rate change them at the start of each step and
nothing differentiates through them. `simulate_dynamic_wave(..., record=("water.q",
"sewer.setting"))` reports each link's flow and setting.

`linkcontrol.LinkControl` is SWMM's `evaluateControlRules` (`routing.c:271-310`) on the state
at the START of a step: pump targets from the depth limits, then the rules
(`controls.ControlEngine`, `controls_evaluate`) every routing step or, with `RULE_STEP`, at
each multiple of it (the routing steps are shortened to end there), then every link whose
target differs from its setting is changed - pumps, weirs and outlets at once, an orifice at
the rate `CloseTime` gives (hours per full opening) - and a link opened or closed has its
time-last-set updated. Rules follow `controls.c`: premises are judged strictly left to right
(an AND after a false result stops the chain; an OR is tried only while false), the actions of
the branch taken are visited last-written first and, per link, the first stays unless a later
one comes from a strictly higher PRIORITY; a CURVE setting is the CONTROL curve at the value
of the last premise compared, a TIMESERIES one the series at the clock (one millisecond late,
as SWMM's clock is), a PID one SWMM's recursive PID on the last comparison's two sides.
Comparisons use the values in the file's units (as SWMM), the clock premises SWMM's DateTime.
`NODE INFLOW` reads the lateral inflow of the PREVIOUS step, as SWMM evaluates the rules
before adding the new inflows.

A setting changes the devices as in SWMM: an orifice's opening height is `setting` times the
full height (`orifice_setSetting`, its `cOrif`, `cWeir` and critical height follow), a weir's
crest rises by `(1 - setting)` of its opening, a V-notch becomes a trapezoid below 1 and the
equivalent orifice coefficient is recomputed (`weir_setSetting`), an outlet's flow is scaled,
and a setting of 0 closes the device (`link_getInflow`). A closed orifice keeps contributing
the surface area it last had to its end nodes, as in SWMM, where it is no longer evaluated.

Not modelled, refused by name when the file is read: `VARIABLE` and `EXPRESSION` in `[CONTROLS]`,
rain-gauge premises, the premise attributes MAXDEPTH, FULLFLOW, FULLDEPTH, LENGTH, SLOPE and
VELOCITY and the DEPTH of a link, and actions on conduits (a closed conduit). Control rules
need an unbatched state.

## Cross sections, losses, transects and culverts

A conduit's section is SWMM's, whatever its shape (`Pipe.xsect`, an `Xsect` of
`noodl.apps.sewer.xsect_spec`; a `Pipe` without one is the tabulated circle of
[SWMM's own circular geometry](physics.md#swmms-own-circular-geometry), whose code path is unchanged).
SWMM evaluates every section through the same few functions -- area, top width and hydraulic
radius from the depth, depth from the area, and, for the normal depth that classifies the flow
and sets a free outfall, the area from the section factor -- and the dynamic wave reads them
through the surface areas, the momentum update, the Froude number and the critical and normal
depth. `noodl.apps.sewer.xsect_shapes` reproduces each, operation for operation:

- the **tabulated shapes** of `xsect.dat` (`EGG`, `HORSESHOE`, `GOTHIC`, `CATENARY`,
  `SEMIELLIPTICAL`, `BASKETHANDLE`, `SEMICIRCULAR`, `HORIZ_ELLIPSE`, `VERT_ELLIPSE`, `ARCH`, with
  the standard elliptical and arch sizes by code), through SWMM's `lookup` / `invLookup`; which
  tables each shape reads forwards and which it inverts is SWMM's (a `GOTHIC` area is the
  inverse of its depth table, an `EGG` area a direct lookup);
- the **closed forms**: `RECT_CLOSED` (with its 0.97 sloped section-factor limit), `RECT_OPEN`,
  `TRAPEZOIDAL`, `TRIANGULAR`, `PARABOLIC`, `POWER` (its perimeter by numerical arc length),
  `RECT_TRIANGULAR`, `RECT_ROUND` and `MODBASKETHANDLE`, `FILLED_CIRCULAR` (the circle's tables
  shifted by the fill, with the fill added to both offsets) and `FORCE_MAIN` (the circle's
  tables with its own section factor);
- the **per-conduit tables**: `IRREGULAR`, from the 51-point tables SWMM builds from a
  `[TRANSECTS]` entry (`xsect_spec.transect_tables`: each depth slice's area, width and
  conveyance with the overbank and channel Manning n, the hydraulic radius from the Manning
  flow), and `CUSTOM`, from the tables of a SHAPE curve (`custom_tables`);
- the **area from the section factor**, `generic_getAofS`: a bracketed Newton iteration on the
  shape's own `S(A)` and `dS/dA` (`findroot_Newton`, 60 iterations, tolerance `1e-4 A_full`),
  not the exact root; and the **critical depth**, `xsect_getYcrit`: a closed form for the
  rectangular, triangular, parabolic and power sections, the 25-interval enumeration where the
  section's area is within a factor of two of the equivalent circle's, Ridder's method
  (`findroot_Ridder`, tolerance 0.001 ft, and `-1e20` where it does not bracket, as SWMM) where
  it is not.

SWMM's own quirks are reproduced because the engine is the reference: the hydraulic radius of the
circular bottom of `RECT_ROUND` is `0.5 r (1 - sin t) / t` (`xsect.c:2062`), which falls to zero at
the springing line; a `FILLED_CIRCULAR` reads the width of its fill at `(2 y_fill) / (D + y_fill)`
(`xsect.c:281`); the transect `Yfactor` is divided by the unit factor twice.

All of it is batched: the conduits are grouped by shape (`XsectSet`) and each query is a few
vectorised calls per shape present, not one per conduit, so a mixed network costs a few
more tensor operations per pass, and a network of circles none (the circle path is
`xsect_tables`, bit for bit). The normal and critical depths are evaluated only for the
conduits that read them: those with an offset (the flow class) and the outfall conduits. An
open shape (`xsect_isOpen`: `RECT_OPEN`, `TRAPEZOIDAL`, `TRIANGULAR`, `PARABOLIC`, `POWER`,
`IRREGULAR`) has no crown: its top width is not frozen near the top, it has no slot, and
`yFull` of its Courant length is `A_full / W(y_full)`. Reporting keys: `"sewer.R_h"`, `"sewer.d_m"`,
`"sewer.T"`, `"sewer.A_air"`, `"sewer.D_h"` and `"sewer.V_air"` come from the section for a
non-circular conduit (the headspace perimeter is `P_full - A/R + W`); the quality and air models are the
circular-pipe correlations and are refused by name for a network with a non-circular conduit.

**Local losses.** The `[LOSSES]` entry, exit and average coefficients enter the momentum
update's denominator as `dq5 = (c_in q/a_1 + c_out q/a_2 + c_avg q/a_mid)/2/L dt`
(`dwflow.c:228-239`, each term where its area exceeds `FUDGE`), with the coefficients
swapped for a conduit the reader reverses. Seepage (the sixth field) and evaporation are
[losses of water](#evaporation-and-seepage-losses).

**Force mains.** A `FORCE_MAIN` flows partly full as a circular conduit with the Manning n that
gives the same full normal flow as its Hazen-Williams C or Darcy-Weisbach roughness
(`forcmain.c::forcemain_getEquivN`, from `FORCE_MAIN_EQUATION`; the file's n is ignored), and
full by the force-main friction slope instead of Manning's: `sBot v^0.852 / R^1.1667`
(Hazen-Williams) or `f sBot v / R` (Darcy-Weisbach, the Swamee-Jain friction factor of the
Reynolds number `4 R v / 1.1e-5`).

**Culverts.** A culvert code (1 to 57, `culvert.c`, FHWA HEC-5) replaces the normal-flow limit of a
conduit that is not full by its inlet-control flow for the head `y` above its upstream invert:
unsubmerged (`y <= 0.95 y_full`) the form-1 implicit critical-depth equation (Ridder's method,
whose flow is SWMM's value at the iteration's LAST evaluation) or the form-2 power law,
submerged (`y >= y_full (16 C + Y - scf)`) an orifice equation, and linear between. Like the
normal-flow limit it is a cap on the conduit's flow; because the cap follows the upstream head
within a pass, it is applied as the line `q_inlet + (dQ/dH)(dH_1 - dH_1,fed)` (SWMM's own `dQ/dH`,
capped by the junction's grounding as the slope switch's wall is), which has the same fixed
points and converges where a lagged flag does not.

**Verification.** The shape fixtures are run through live SWMM 5.2.4 and noodl (one independent
chain per section, subcritical so that SWMM's own iteration converges on every step, asserted):

| Fixture | Chains | Worst depth (m) | Flows |
|---|---|---|---|
| `shapes_dynwave.inp` | every modelled shape, free surface | 3.0e-9 | within rel 1e-6 + 1e-9 at every step |
| `shapes_drop_dynwave.inp` | the same with offsets at both ends (flow classes DN_CRITICAL) | 1.2e-9 | idem |
| `shapes_surcharge_dynwave.inp` / `shapes_slot_dynwave.inp` | closed shapes under a FIXED stage past the crown | 7.2e-8 / 2.1e-8 | idem |
| `shapes_us_dynwave.inp` | CFS: standard ellipse and arch sizes by code, channels | 1.6e-9 | idem |
| `force_main_hw_dynwave.inp` / `force_main_dw_dynwave.inp` | full force mains | 1.8e-8 / 2.7e-8 | idem |
| `losses_dynwave.inp` | entry, exit and average losses | 5.6e-9 | idem |
| `irregular_dynwave.inp` | compound and meandering transects | 5.8e-10 | idem |
| `culvert_dynwave.inp` | six inlet types, unsubmerged to submerged | 1.1e-8 | idem |

Where a fixture sat in a regime in which SWMM itself does not converge (a dry start, a
supercritical slope) there is no answer to compare with and it was changed, not the tolerance.
The `RECT_ROUND` chain is omitted from the offset fixture: with a free fall at the outfall its end depth sits in the
circular bottom, whose radius is SWMM's `(1 - sin t)/t` above, and the lagged equations then
have two fixed points that SWMM and noodl may pick between in the same step.
