# Building physics: limitations

What the building physics application does not model, and where it differs on purpose from CONTAM and the Modelica Buildings Library. The supported subset is summarised on the [overview](index.md#supported-and-not-supported). Anything outside it is refused with an error that names it, unless a bullet below says otherwise.

## Limitations

- **Airflow is quasi-steady.** Pressures and flows balance instantly within a step. On the
  CONTAM route a zone's air mass is held fixed within a step, as in CONTAM, unless the project
  asks for the zone's air-mass change from a changing temperature (`dodMdt`), which then enters
  the balance as in ContamX. The Modelica route stores air mass by default (volume mass storage,
  see [Importing Modelica Buildings](modelica.md#volume-mass-storage)); `read_modelica(..., mass_storage=False)` gives the quasi-steady airflow there too, and
  the storage-dominated group on the [verification page](../../reference/verification/building.md#against-openmodelica-modelica-buildings-library) shows
  what that costs.
- **Control nodes cover ContamX's sensors, schedules, arithmetic, logic, switches,
  hysteresis and P and PI controllers.** Integration, running-average, delay, modifier,
  phantom nodes and value files are refused by name when an actuator, or a named report node
  of a project with actuators, depends on them (unused, they are ignored), super nodes
  wherever they are, as are occupancy and exposure
  sensors, a flow or pressure-difference sensor on a doorway, a control on a plain duct
  junction, on a triggered source or on an AHS outdoor-air or exhaust path, and control
  references in a steady airflow run.
- **Duct leakage, duct balancing and super airflow elements are refused by name.** Duct leakage
  (a duct element's leakage rate) and ContamX's duct-balancing run mode are not modelled;
  `sup_afe` super elements cannot be simulated by ContamX itself. A schedule on a
  Darcy-Colebrook duct segment is refused because ContamX ignores it.
- **A learnable duct roughness is not re-checked.** `AirDuct` refuses a roughness of 3.7
  diameters or more (the Colebrook equation has no solution beyond) when it is built; with
  `learnable=True` an optimiser can later push the roughness past that bound unchecked, so
  keep it inside the bound (a parametrisation such as a sigmoid of the diameter does so).
- **Contaminant physics follows ContamX's subset.** Boundary-layer diffusion and
  deposition-with-resuspension sinks, super source elements, loading-dependent gaseous filters,
  UVGI filters in duct networks, and non-trace species (which change the air density) are
  refused by name. A
  species' decay constant is kept as data and not applied, as in ContamX, with a
  `ContamCompatibilityWarning` (filter it by category to silence it); use a kinetic reaction
  for a decay.
- **Schedule-triggered sources are refused.** A decaying, burst, power-law or peak source on a
  schedule is triggered by it in CONTAM rather than multiplied; such a project is refused by
  name. `PowerLawDecaySource` and `PeakEmissionSource` count their time from the simulation
  start.
- **A weather or ambient file must cover the whole run**; nothing is extrapolated.
- **Occupants follow ContamX's subset.** An occupant's coordinates (for 1-D zones) are
  ignored, generation from value files is refused, and exposure needs a transient
  contaminant run. Exposure is a mass-fraction integral; dose and infection risk need a
  breathing rate from the user. Occupancy and exposure sensors on control nodes are refused
  (see the control-node bullet above).
- **Wells-Riley risk, inhaled dose and zone air change rates are additions of noodl.** The
  age of air and the building air change rate are ContamX outputs and are verified against
  them, as are the occupants' exposures.
- **The default coupling is a single pass.** `build_model`'s `coupling` defaults to
  `"pingpong"`: a `.steady()` call then solves the airflow at the *initial* temperatures and does
  not re-converge. On the linear-density single-zone case it returns
  $T_z = T_o + S / (c_p F(293.15\,\mathrm{K}))$ rather than the coupled answer. Whenever the
  airflow depends on the temperatures it carries, pass `coupling="iterate"` with an
  `iterate_tol`.
- **`iterate_tol` cannot be tighter than the airflow solve itself.** Its floor is the potential
  solve's own Newton tolerance, propagated through the temperature response. A tolerance below
  that floor stalls, and `steady` raises an error naming the layer, the change reached and the
  tolerance asked for. Loosen `iterate_tol`, or tighten the airflow solve by passing Newton's
  `atol`/`rtol` as keyword arguments to `step` or `steady` (they reach the potential solve).
- **Multiple steady states: `"iterate"` may not find the one you want.** On Li and Delsante's
  opposing-wind single-zone case, which has three steady states, `coupling="iterate"` (a fixed
  0.5 relaxation) cannot hold the wind-driven, upward-flow state: it moves away from it even when
  started on it. Time stepping with the default `"pingpong"` coupling resolves all three states
  correctly, so for a case like this, step to steady state instead of calling `.steady()`.
- **A zone without airflow has no steady contaminant state.** At a `.prj` file's steady instant
  the ambient may sit at the zone temperature with no wind (the one-zone stack projects NIST ships
  do): no air moves, the zone keeps whatever enters it, and its steady mass fraction is undefined
  without a source and unbounded with one. `prj.steady` refuses it with `UndefinedSteadyState`,
  naming the zone and giving the ambient temperature, zone temperatures and wind speed at that
  instant (ContamX returns the initial value there). Simulate the run instead, or set an
  ambient that drives a flow. A group of zones that exchange air only with each other is
  refused the same way.
- **Wall conductance is not learnable** through this application's API: `WallMass` takes plain
  floats for `ua_zone` and `ua_ambient`, so gradients do not reach them.
- **Doorway laminar transition is approximated.** Doorway openings use the generic laminar
  transition of `PowerLaw` (1e-3 Pa) rather than one derived from the door's own record. The
  effect is confined to pressure differences below $10^{-3}$ Pa across the doorway.
- **`PressurizationData` is imported but not compared.** It is MBL's 62.5 m3 room between two
  `Outside_CpLowRise` walls over 30 days of San Francisco weather.
  The room flushes in about 10 s, against steps of 50 to 3000 s. Where the flow fades, in a
  calm or with the wind parallel to both walls, it can fall by orders of magnitude within one
  step, and the room freezes at the state it had when its flushing stopped. The run does not
  resolve that decay: the transport step sees the flows only at the step's two ends. It
  therefore freezes the room at the wrong state, and this error, first order in the step, stays
  for the whole calm. Over every row at 1 and 2 substeps, the room temperature is 2.9e-3 relative
  from the reference (0.82 K, at the end of the first calm, 25920 s) and its humidity 3.4e-2 (at
  93312 s). Flows are within 1e-6 beyond the reference's resolution and the pressure within 2e-11.
  Refining the grid where the boundary pressures fade reduced the temperature error to 1.6e-4 and
  the humidity error to 2.0e-3 in an experiment. That refinement, together with a transport step
  that treats the flows' change within a step implicitly, would remove this error.
  The wind boundaries themselves are verified on the `Buildings.Fluid.Sources` models
  (on the [verification page](../../reference/verification/building.md#against-openmodelica-modelica-buildings-library)).
- **`TrickleVent` is imported but not compared.** Its room, held at its set point by a P
  controller with a 5 s loop, lags the flows' change by those 5 s, which a step that sees the
  flows at its two ends only misses. Over its first 7200 s the room temperature is up to
  4.3e-5 K (1.5e-7 relative) from the reference at 1 and 2 and at 2 and 4 substeps alike, and
  the controller output, five times the room's deviation from its set point, 1.1e-2 relative;
  the flows are within 1.5e-5 and the humidity within 1.8e-4. Over every row the run at 1
  substep does not get through: at 959400-960000 s (day 11) its airflow solve stops at a
  0.142 kg/s residual (cause not identified). Its weather and wind pressures are compared (see the [verification page](../../reference/verification/building.md#against-openmodelica-modelica-buildings-library)).
- **The Modelica import accepts a stated subset.** A `LimPID` is read as a P controller on a
  zone temperature driving a heat flow. Refused with a named error: a `LimPID` with an
  integral or derivative part (`controllerType` PI, PD or PID), with feed-forward or with the
  steady-state or initial-output initialisation; a feedback signal driving anything but a
  `PrescribedHeatFlow`; a door or zonal flow inside a column chain, and a ring of one-way
  elements in series with no volume or boundary; and every class outside the supported list,
  the CDL blocks (`Buildings.Controls.OBC.CDL`, their PID and hysteresis included) among them.
  Only the weather reader's airflow outputs are read; see [the Modelica import's refused
  content](modelica.md).

## Intentional differences from CONTAM and Modelica

- **noodl follows ContamX where ContamX departs from its documentation.** A species' decay constant is kept as data and not applied (with a `ContamCompatibilityWarning`); the NRCC power-law emission is evaluated with time in hours; a cutoff source is not clamped above its cutoff; a UVGI filter crossed against its path's direction gets a fixed dose of 1 J/m²; duct-balancing data has no effect; a control value on a path is not clipped. All were measured on ContamX 3.4.1.7.
- **noodl refuses what ContamX would accept or ignore, when the meaning is ambiguous.** A cubic-spline curve that is not increasing is refused, because the inversion would be ambiguous. A schedule on a Darcy-Colebrook duct segment and a schedule or flow on an air handler's outdoor-air or exhaust path are refused, because ContamX ignores them.
- **Reference-density coefficients.** The quadratic, damper and constant-flow fan families read from a `.prj` keep reference-density coefficients, because they are not part of the parity evidence. `Duct`, the Colebrook law of TN 1887, differs from ContamX's duct friction law by about 0.1 %; the duct elements used for CONTAM duct networks (`AirDuct`) follow ContamX's.
- **Additions of noodl.** The inhaled dose, the Wells-Riley probability and the zone air change rate are not CONTAM outputs. The age of air, the building air change rate and the exposure are, and are verified against ContamX.
- **Modelica air mass.** The Modelica route stores each volume's air compressibly, as Modelica's `MixingVolume` does; the CONTAM route holds the zone air mass fixed within a step. Wind angles follow each tool's own convention: in MBL the wind pressure acts on the boundary node and angles are measured as MBL defines them, whereas in CONTAM it acts on the envelope path (see [Importing Modelica Buildings](modelica.md#weather-and-wind)).
- **Differentiability and batching** have no counterpart in either tool; they carry the limits listed above (for example, the wall conductance is not learnable).
