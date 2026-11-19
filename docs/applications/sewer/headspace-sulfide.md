# Headspace air and sulfide

How to run the water quality and the headspace air on the dynamic wave: the BOD and sulfide
layers on the solved flows, and the air above the water with the two-film transfer of H₂S. The
equations (Henry's constant, the transfer coefficient, sulfide generation, the air momentum
balance) are in [Physics](physics.md#headspace-air); the coefficients and which of them are
verified are in [Limitations](limitations.md#coefficient-provenance). On the kinematic-wave path
the same layers are switched on by `build_model(net, air=True, quality=True)`; see
[Kinematic-wave and steady runs](kinematic-wave.md#build_model).

## Headspace air on the dynamic wave

`build_model(net, air=True)` on a `"dynamic_wave"` network reuses the kinematic path's air layer
unchanged (`Headspace`, `Drag`, `Stack`, the leaks and fans, built by the same `air_layer` and
`air_heads`) and feeds it from the solved conduits. Each conduit's air geometry is read at its **link depth**
$y = \min(y_{\text{mid}}, y_{\text{full}})$ (SWMM's `Link.newDepth`, one barrel's) on the exact
circle (`air_geometry`), whatever `net.geometry` is: the tabulated section is the water side's,
and the air has no SWMM reference, as with the kinematic path at `geometry="tabulated"`.
With $n_b$ barrels the closure writes `sewer.A_air` (m², total, $n_b A_{\text{air}}$),
`sewer.D_h` (m, one barrel's), `sewer.T` (m, total free-surface width, so $T/A_{\text{air}}$,
which the drag reads, is one barrel's) and `sewer.V_air` (m³, total): $n_b$ parallel headspaces
have $1/n_b^2$ of one barrel's resistance. These four keys, with `sewer.R_h`, `sewer.d_m`,
`sewer.q`, `sewer.h`, `sewer.v` and `sewer.V_wet`, are the columns of `pipe_table` for every
dynamic-wave model, with or without the air.

A conduit flowing full has no headspace ($A_{\text{air}} = 0$, infinite resistance). The air
side reads the depth capped at 0.999 D, which leaves a finite area of about $6\times10^{-5}D^2$
and a resistance so large that the conduit is nearly closed, with finite derivatives on both
ends of the circle; a surcharged tree keeps solving. It is not closed: the interfacial drag does
not depend on the headspace's area, and still moves a small, nonzero flow of air through a capped
full conduit.

The closure also writes `rho_air_nodes` (from `T_head` and `T_amb`) and `air_quality.capacity`
(m³, per junction): the headspace volume of the junction's outgoing conduit plus the air of its
shaft above the water, $A_s(y_{\text{full}} - y_j)$ for a junction ($A_s$ the manhole's
`surface_area`, else 1.167 m²; $y_j$ the junction's depth) and the storage curve's volume
between the water and the full depth for a storage unit. As on the kinematic path the shaft term
needs `ground` on every manhole. With quality and sulfide the `air_quality` layer of H₂S and
`H2STransfer` are built and judged by the iteration beside the water quality (the air layer is
not built without sulfide, its only species).

The air side assumes every junction has exactly one outgoing conduit (its capacity, leak and
sulfide transfer are that conduit's). A loop, a junction with several outgoing conduits or none
(an adverse conduit, or one drained only by a regulator) and an outfall that is a conduit's
upstream end are refused by name with `air=True`.

## Quality on the dynamic wave

`build_model(net, quality=True, species=...)` on a dynamic-wave network adds the transport
layer `"water_quality"` (kg/m³, one row per junction, the outfalls its boundary) on the
solved flows. Every edge kind of the `"water"` layer (conduits, orifices, weirs, outlets) is
one of its flow kinds, so it reads `"water.q"` and is upwind by the sign of each flow: a
conduit carrying its water against its own orientation (the loop's adverse conduit) carries
the tracer from its `v` end. An orifice, weir or outlet carries water between junctions but
holds none, as in SWMM, which gives a non-conduit link its upstream node's quality
(`qualrout.c:282-288`). The head iteration judges the layer too (`iterate_tol` 1e-12 kg/m³);
its concentrations follow the flows of each pass, which costs 1-5 % more passes.

Each species `s` enters with the lateral inflow at the concentration of the driver
`"<s>_in"` (kg/m³, one per node); only a positive inflow loads. A negative lateral inflow (a
withdrawal) takes water and mass out at the junction's own concentration, whatever `"<s>_in"`
is, so a uniform tracer stays uniform. It is an outflow of the transport step itself (the
removal rate `"water_quality.removal"`, $w / V_{	ext{end}}$ for the backward Euler scheme), so the
water that enters during the step and is withdrawn again carries its mass out too, even
when one step withdraws several times what the junction held at its start. `"bod"` decays and `"sulfide"` is generated
(Pomeroy–Parkhurst, see [Quality](physics.md#quality)); a species of any other name is a conservative
tracer. Sulfide needs `"bod"`, and it needs a single outgoing conduit at every junction,
whose hydraulic radius `sewer.R_h`, speed `sewer.v` and mean depth `sewer.d_m` (exact circle
at the link depth, one barrel) it reads. A loop, an adverse conduit or a junction drained
only by a regulator therefore refuses sulfide, naming the junctions; BOD and tracers need no
conduit and build on any network. In a conduit flowing full (`d_m = 0`, no free surface)
sulfide is still generated, without the loss term, and the loss term reads the speed $|v|$.

**Control volumes.** Each junction's capacity is the water its transport control volume
holds, carried as `"sewer.quality_volume"` (m³). It starts at half the summed wetted volume
of the attached conduits (`sewer.V_wet`, totals over barrels) plus a storage unit's own
volume, floored at 1e-9 m³; half of an outfall conduit belongs to no junction. From there it
follows the junction's continuity on the flows the transport sees,

$$
V_{\text{end}} = \max\big(V_{\text{start}} + \Delta t\,(N - Q_{\text{overflow}}),\ 10^{-9}\ \text{m}^3\big),
$$

with $N$ the junction's net inflow (its lateral inflow and every link's flow) and
$Q_{\text{overflow}}$ SWMM's flooding loss. The transport step's amount form,
$V_{\text{end}} x_{\text{end}} - V_{\text{start}} x_{\text{start}} = \Delta t\,F(x_{\text{end}})$,
then balances water as exactly as it balances mass, so water that starts and enters at one
concentration keeps it while the network fills and drains. Half-volumes alone do not: a
conduit filling from its upstream end gives its surface area, and so its water, to that
junction (`dwflow.c:357-411`), while the halves split it, and on the tree's filling a uniform
inflow concentration of 1 came out anywhere between 0 and 5. A flooded junction's overflow
leaves at the junction's concentration (`OverflowLoss`, applied before the transport step,
with the same water taken out of the control volume).

**Against SWMM's scheme.** A conduit's concentration is its upwind junction's. SWMM's
reactors are the conduits: each mixes its upstream node's concentration over its own volume,
with the inflow raised by the volume change so that the conduit's water balances
(`qualrout.c:315-341`). Its junctions hold no volume and mix the upstream links'
start-of-step concentrations (`qualrout.c:210`, a lag of one step each). At steady flow
both give every conduit the flow-weighted mixture of what enters upstream. In transit the
two differ by their residence times, by design: the tracer arrives shifted by the
difference of the chains' mean delays, $T_{\text{noodl}} - T_{\text{SWMM}}$ (on the tree
noodl's is the shorter: a junction's control volume holds roughly the upstream part of its
conduits, where SWMM's reactor is the whole conduit; see
[Dynamic wave against SWMM](../../reference/verification/sewer.md#dynamic-wave-against-swmm-in-detail)).
