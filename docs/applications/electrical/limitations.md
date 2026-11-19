# Electrical: limitations

What the electrical application does not support, and what it refuses. Where the application
cannot do something it refuses by name; it does not approximate silently.

## Circuits

- **Fixed time grid.** No adaptive step, and `transient` inserts no breakpoints: a source edge
  inside a step is seen at the step end (the Modelica reader's step grid holds every source
  event).
- **No event location.** A `Switch` is a discontinuous law: it changes state only where its
  control driver is evaluated, at a step end, and the step in which it switches is not
  refined. (The `Diode` is continuously differentiable, its exponential continuing linearly
  above `Maxexp`, and needs no event.)
- **No switching inside a transient.** `transient` takes one `drivers` mapping for the whole
  run, so a `Switch` keeps the state its control driver gives it throughout. To switch at a
  time, step the model yourself (see
  [Build and analyse a circuit](circuits.md#time-stepping-and-breakpoints)).
- **No ideal diode.** The piecewise-linear ideal diode (MSL `Ideal.IdealDiode`) is not
  provided; `Diode` is the exponential junction diode with its linear continuation.
- **One frequency per phasor model.** No harmonic balance, no small-signal linearisation of a
  nonlinear operating point: `ac_sweep` takes linear devices only. The nonlinear phasor laws
  (constant-power loads, converters) are solved by Newton at one frequency.
- **No dynamic phasors.** A phasor model is quasi-static; MBL's `FixedZ_dynamic` models are
  refused.
- **Lumped, balanced.** No distributed transmission lines; balanced three-phase models are
  read as their one-phase equivalent, and unbalanced three-phase systems are not supported.
- **No batteries, photovoltaics or grid sources** from MBL (refused by name).
- **Initial points.** A capacitor loop, series inductors or an inductor in series with a
  current source make the initial point singular; start from the operating point (see
  [Build and analyse a circuit](circuits.md#step-in-time)).
- **Inputs a circuit refuses.** Unknown device types, duplicate or empty names, a port on one
  node, non-scalar parameters, nodes with no path to ground, loops of voltage sources and cut
  sets of current sources (see [Build and analyse a circuit](circuits.md#what-a-circuit-refuses)).

## Power flow

- **No reactive-power limits.** A PV bus never switches to PQ at a limit;
  `build_power_flow(..., enforce_q_limits=True)` is refused.
- **No DC lines** (`mpc.dcline` is refused by name) and **no isolated buses** (MATPOWER bus
  type 4, refused by name).
- **Balanced only.** One positive-sequence phase; no unbalanced three-phase power flow.
- **One slack bus**, no distributed slack.
- **Setpoints are law parameters.** Load and generator setpoints are not drivers: a batch of
  operating points is a batch of drivers (the slack voltage), or of separately built models.
- **Refused networks.** More or fewer than one slack bus, a slack bus without a generator,
  conflicting voltage setpoints on a bus, zero-impedance branches, islands, and for DC a
  branch with negative `x tau` (see [Power flow](power-flow.md#what-the-builders-refuse)).
- **MATPOWER files.** Only case format version 2 with whole-field assignments; no indexed
  assignments, `define_constants` or block comments (see
  [MATPOWER files](matpower.md#what-is-read-ignored-and-refused)).

## Modelica circuits

The reader converts the listed classes of MSL 4.1.0 and MBL v13.0.0 and refuses the rest,
naming every offending instance. The refused classes and settings, with reasons, are listed
in [Modelica circuits](modelica.md#what-is-refused). In short: batteries, photovoltaics and
grid sources, `IdealDiode`, unbalanced three-phase classes, dynamic phasor models
(`FixedZ_dynamic`), and any class not in the supported table.

## Coupling

Joule-heating coupling is conduction-only on the heat side (the heat layer conducts heat only
along conductors); a heat capacitor directly on a heat port needs a `ThermalConductor`
between them (see [Electro-thermal coupling](electro-thermal.md)).
