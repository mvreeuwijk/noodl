# Sewer limitations

What the sewer application does not support, where it is implemented but not compared with
SWMM, and which coefficients are not verified. Unsupported inputs are refused by name, with
the line number when they come from a file. The evidence for what is supported is on
[Verification: sewers](../../reference/verification/sewer.md).

## What is not supported

- **Kinematic-wave and steady routing have no surcharge or backwater.** A network that would
  surcharge or need backwater is refused by name there; use `routing="dynamic_wave"` for
  backwater.
- **Kinematic-wave and steady routing: dendritic networks only**, one outgoing pipe per
  manhole, no loops. Dynamic-wave routing allows loops and flat or adverse conduits.
- **Dynamic wave, not supported.** Each is refused by name, with the line number when it comes
  from a file (see [SWMM `.inp`](swmm.md)), unless it says otherwise:
    - flow dividers under steady and kinematic-wave routing (a divider is a junction under
      dynamic-wave routing, as in SWMM), roadway weirs and weir coefficient curves; of the control
      rules, `VARIABLE` and `EXPRESSION`, rain-gauge premises, some premise attributes and
      conduit actions (see [Link settings and control rules](dynamic-wave-structures.md#link-settings-and-control-rules)); an ideal pump whose inlet floods;
    - ponding together with water quality or headspace air, and for a storage unit; a junction
      with no ground (`MaxDepth` 0) and no `SurDepth` floods as soon as it fills to its crown;
    - groundwater, LID, snowmelt and the pollutant buildup and washoff of the runoff
      (`[GROUNDWATER]`, `[LID_USAGE]`, snow packs, `[LANDUSES]`), rain `FILE` gages, evaporation read from a climate file or computed from
      temperature, pollutant time-series, mass and
      dry-weather inflows, and the `[FILES]` `USE HOTSTART`, `USE INFLOWS`, `USE RDII` and
      `USE RUNOFF` (a `TIMESERIES` outfall's stage is a driver); surface runoff itself is
      modelled ([Subcatchment runoff](swmm-hydrology.md#subcatchment-runoff-from-a-swmm-file));
    - conduit evaporation, seepage and storage losses under STEADY or KINWAVE routing, an
      exfiltrating `PARABOLOID` storage unit and water quality with a storage unit's loss of water (see
      [Evaporation and seepage losses](dynamic-wave-structures.md#evaporation-and-seepage-losses)), and a nonzero pollutant `Cinit`;
    - the `DUMMY` section (a link that routes its node's outflow without a section) and the
      `STREET` section, and non-circular conduits under steady or kinematic-wave routing;
    - water quality and headspace air for a network with a non-circular conduit (their
      correlations are the circular pipe's);
    - regulators with `LINK_OFFSETS ELEVATION`;
    - headspace air on networks that are not dendritic: loops, junctions with several outgoing
      conduits or none, a junction drained only by a regulator (orifice, weir or outlet), a cycle
      that reaches no outfall (refused by network validation), and an outfall at a conduit's
      upstream end are refused with `air=True` (see [Headspace air on the dynamic wave](headspace-sulfide.md#headspace-air-on-the-dynamic-wave));
    - a variable step on a batched network, and `dt=` with a variable step (the step is the
      options'); a variable-step run is not differentiable through its step sizes.
- **Dynamic wave, implemented but not compared with SWMM live.** Each is transcribed from
  SWMM's source and checked against its closed forms or in unit tests, with no live reference
  run yet: a surcharged weir (above its opening, `CanSurcharge` YES or NO), bottom orifices,
  regulator flap gates and the ARMCO head loss, sideflow and trapezoidal weirs, functional and
  `/HEAD` outlets, and an orifice ending at a storage unit; the cylindrical, paraboloid and
  pyramidal storage shapes and a `TABULAR` curve beyond its last point; a storage unit falling
  back through its full depth under EXTRAN; a tide with a nonzero `START_TIME` or beyond 24 h of
  elapsed time, and batched `TIDAL` or `TIMESERIES` outfalls (a batched stage only via the
  driver); and a fully submerged weir whose flow reverses through zero, where SWMM itself
  fails to converge on a few steps.
- **Dynamic wave, shapes: where SWMM's trials limit-cycle.** A section or outfall regime in which
  SWMM's own iteration does not converge (a dry start, a supercritical slope, culverts running
  from unsubmerged to submerged in a few steps) has no converged answer to compare with, and
  noodl's lagged flags can fail to settle there too (`ValueError`: the step did not converge). The
  culvert cap is applied as a line in the head difference, not the upstream head alone, so the
  iteration of a culvert whose downstream end moves is slower (about 30 passes per step on the
  parity fixture) though its fixed point is SWMM's. A free outfall below the springing line of a
  `RECT_ROUND` conduit is such a regime (the section above).
- **Dynamic wave, where SWMM has no fixed point** (the three tie rules above): the EXTRAN
  crown hold, the normal-flow slope switch and the drop-end area switch. The Froude test of the
  normal-flow limit stays a lagged flag. The area switch picks one threshold per junction (the
  nearest to its head), so two drop ends of one junction switching in the same step are not
  covered. At a step that lands on the slope switch the gradient is the tie rule's, the
  sensitivity of a solution constrained to $y_1 = y_2$; it does not depend on the wall slope $S$,
  which only shapes the passes (on the tree's drain-down the gradient is the same to 11 digits
  for $S$ scaled by 0.2 and 5, and matches central differences to 2e-8 relative).
- **Dynamic wave, resolution.** The default head solve (`solve_atol`) leaves a surcharged
  junction up to about 2.5e-7 m from the fixed point (pass `atol=1e-12` for tight heads where no
  junction sits on its bed). Plain relaxation (`acceleration=None`) does not converge on some
  steps where the default Anderson iteration does (the flooding tree's EXTRAN ramp-down at
  t = 175 s), and `iterate_flow_tol` is required either way.
- **Dynamic-wave quality: junction reactors, not SWMM's conduit reactors.** Steady
  concentrations agree with SWMM; in transit the tracer arrives shifted by the difference of
  the residence times (on the tree's ramp up to 25 % of the plateau at the head conduit,
  9 % at the outfall conduit). The transport control volume follows the junction's
  continuity, not its geometry, so it drifts from the half-volumes it starts at (at the
  tree's steady state J1 holds 2.0 m³ against C1's 6.5 m³). A junction drained dry keeps the
  1e-9 m³ floor, where the water balance is no longer exact. Sulfide needs a single
  outgoing conduit at every junction (refused by name on loops, adverse conduits and
  junctions drained by a regulator); BOD and tracers do not. Water starts clean: a nonzero
  SWMM `Cinit` is refused by name. Headspace air and the H₂S transfer are built on dendritic
  networks only (see [Headspace air on the dynamic wave](headspace-sulfide.md#headspace-air-on-the-dynamic-wave)).
- **No gas-phase sulfide sink.** H₂S leaves the headspace by ventilation, at the outfall, and
  by re-absorption into the water (the two-film flux is signed); no wall uptake or gas-phase
  oxidation is modelled.
- **Headspace drag uses the absolute water-surface velocity.** `Drag` forces the air with
  $U_s$ rather than the relative velocity $U_s - U_{\text{air}}$, which would need the air
  layer's own solved state. The drag is overestimated where the headspace air already moves
  with the water.
- **Headspace buoyancy ignores conduit offsets.** The stack term takes a conduit's crown
  elevation as the mean of its end nodes' inverts plus its diameter, as the kinematic path does;
  a dynamic-wave conduit raised on an inlet or outlet offset has its crown that much higher than
  the stack term sees.
- **`f_i` is a calibrated constant.** It is an attribute of the `Drag` head term. `EdgeHeads` writes
  the head as the driver `"air.edge_head"`, so a gradient reaches `f_i` when it is a tensor with
  `requires_grad=True`, but the value is calibrated against Pescod and Price ([Verification: sewers](../../reference/verification/sewer.md#checks-for-physics-swmm-does-not-model)), not fitted.
- **Without a ground elevation, manhole shafts add no headspace volume**, so the air-quality
  capacity is the conduit headspace alone. `model.notes` says when this applies.

## Coefficient provenance

Each coefficient is labelled, and three of them are not verified. This matters if you intend to
publish numbers from this application.

| Coefficient | Default | Status |
|---|---|---|
| `f_air`, headspace wall friction | 0.02 | **Unverified.** Stated to lie in the reported 0.015–0.045 range for sewer crowns; the source is paywalled and was not independently checked. |
| `f_i`, interfacial drag | 7.49e-4 | **Calibrated** against one Pescod & Price test (Test 8), cross-checked against Tests 7 and 9 at 24.14% and 25.15% against measured 35% and 27.5% — inside a fairly wide 20–40% acceptance band. |
| Henry's constant, van't Hoff slope | 1.0e-3 mol/m³/Pa, 2100 K | **Verified** (Sander 2023). |
| p$K_a$ | 7.0 | Verified value; its temperature dependence is approximated as constant. |
| $K_L a$ constants $a$, $b$ | 0.86, 0.20 | Correlation form corroborated, **constants unverified** (paywalled). |
| Pomeroy–Parkhurst $M'$, $m$ | 0.32e-3 m/h, 0.64 | **Vendor defaults, unverified** against Pomeroy and Parkhurst 1977. |
| $k_{\mathrm{BOD}}$ | 0.2/day | Generic (Metcalf & Eddy). |
