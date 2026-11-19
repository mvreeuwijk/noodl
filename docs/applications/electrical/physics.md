# Electrical: physics

The equations the electrical application solves and the conventions behind its numbers. For
how to use them see [Build and analyse a circuit](circuits.md), [AC analysis](ac.md) and
[Power flow](power-flow.md).

## One Kirchhoff layer

The whole circuit is **one** `KirchhoffLayer` whose potential is the node voltage: Kirchhoff's
current law holds at every node, and the currents of voltage sources, inductors,
transformers and controlled sources are unknowns alongside the node voltages (see
[Implicit, coupled and dynamic laws](../../concepts/layers.md#implicit-coupled-and-dynamic-laws)).
This is the modified nodal analysis (MNA) form. A circuit is analysed at its DC operating
point (capacitors open, inductors shorted), as a transient on a fixed time grid, or in the
sinusoidal steady state by phasors. A circuit with no capacitor or inductor has no storage,
and each transient step solves its operating point at the step end.

The device laws transcribe the Modelica Standard Library (`Modelica.Electrical.Analog`); the
[device table](circuits.md#devices) gives each law next to its MSL model and the
constitutive law behind it. Each of those laws can be used directly in any `KirchhoffLayer`:
[AC analysis](ac.md#impedances-constant-power-loads-and-converters) adds four laws from the
Modelica Buildings Library, and the [power flow](#power-flow-model) four more.

## Sign conventions

- A one-port device `p -> n` is one edge of the network. Its voltage is `v = v_p - v_n` and its
  current `i` is the current entering pin `p` (and leaving pin `n`): the MSL convention, under
  which every device absorbs the power `v i`.
- A source that holds node `a` at `+V` relative to ground is the device `p = a`, `n = ground`.
  A source delivering power carries a negative current, as `V1` does in the divider.
- A two-port device is two consecutive edges, port 1 `p1 -> n1` then port 2 `p2 -> n2`;
  `device_values` gives its voltage and current as pairs `(port 1, port 2)`.
- A capacitor with one pin on the ground is not an edge: it is a node storage at its other
  pin (`layer.storage`), and its current is the entry of `"electrical.storage_flow"` that
  `model.circuit_storage[name]` gives (`(entry, p, n, sign)`; `sign` is `+1` when the ground
  is pin `n`, `-1` when it is pin `p`). `device_values` and the transient result
  (`"electrical.storage_flow"`) report it. A capacitor between two other nodes is a
  `Capacitance` edge, and a phasor model keeps every capacitor as an edge. With
  `learnable=True` a grounded capacitor's `C` is `layer.storage[0].C`, not on a law. A state
  that starts a trapezoidal step away from rest (a hot start) must carry
  `"electrical.storage_flow"`, as `initial_state` and `operating_point` write it; a state
  without it starts from zero capacitor current, and the first step rings.
- `"electrical.q"` is ordered by kind (kinds in order of first appearance), then by device;
  `model.circuit_index[name]` is the device's `(kind, slice)` in that order.

## Phasor conventions

Phasors are peak-amplitude and sine-referenced: the complex value `X` stands for the time
signal `x(t) = |X| sin(omega t + arg X)`, matching the `Sine` waveform. The complex power a
device absorbs is therefore `S = (1/2) V conj(I)` (W). A source has one phase, its
waveform's: a `Sine(frequency, phase, start_time)` source is the phasor
`amplitude * exp(j (phase - omega start_time))`, the steady state of its time signal; a
`Constant` source is `amplitude` at phase 0 (the AC magnitude, as SPICE's `AC` value; its
`offset` has no phasor). Other waveforms are refused in an AC analysis, by name, as are
nonlinear devices (`Diode`).

The power laws (`ConstantPowerLoad`, `PowerConverter`) are the exception: they define
complex power as `S = dp conj(q)`, without the factor 1/2, so they assume root-mean-square
phasors. A circuit that mixes a power law with a `Sine` source must give that source its RMS
amplitude (`V_peak / sqrt(2)`): otherwise the source's phasor is its peak value, which the
power laws read as an RMS value, a factor `sqrt(2)` in voltage. Linear laws hold for either
scaling. The [Modelica reader](modelica.md) uses MBL's RMS phasors throughout, and the power
flow is in RMS p.u.

## Time integration

The trapezoidal rule is second order but not L-stable, so it rings after a discontinuity.
The circuit therefore uses backward Euler on the first step and on every step whose interval
contains a source breakpoint, and the trapezoidal rule otherwise, as SPICE does. The step
grid is fixed. A heat layer coupled to the circuit takes the same rule on every step, so
that both layers integrate a step the same way (see
[Electro-thermal coupling](electro-thermal.md#step-with-the-circuits-rule)).

## Power flow model

The power flow solves the balanced, positive-sequence sinusoidal steady state of a network
in per unit. It has an AC model on a phasor layer and a DC approximation on a real layer.
It follows MATPOWER 8.1.

### The AC model

`build_power_flow(net)` returns `(model, state, drivers)` for one phasor `KirchhoffLayer`
named `"power"`, whose sources' currents are unknowns of the solve. Its nodes are the buses
(`str(bus.id)`, in table order) and one more node, `"ground"`, the common reference at 0.
Every device is an edge **to ground**: the edge's potential difference is then the bus
voltage phasor `v` itself, and its flow is the current `i` leaving the bus through the device
into ground, so Kirchhoff's current law at a bus is the power-flow balance of that bus. A
load, a shunt or a generator connects one bus to the reference, so it is one edge bus `->`
ground. A branch is a two-port: MATPOWER's branch model, a series impedance with line
charging behind a phase-shifting transformer, is not a two-terminal element (with a phase
shift its from and to admittances differ), so it is a pair of edges, from bus `->` ground and
to bus `->` ground, and its constitutive law relates both currents to both voltages. Four
constitutive laws describe the network (`v` the bus voltage, `i` the edge current into
ground, absorbed power `v conj(i)`):

| Constitutive law (kind) | Edges | Relation |
|---|---|---|
| `PiBranch(r, x, b, ratio, shift)` (`branch`) | from bus `->` ground, to bus `->` ground | `[i_f; i_t] = [[Y_ff, Y_ft], [Y_tf, Y_tt]] [v_f; v_t]`, `Y_s = 1/(r + jx)`, `tau = ratio e^{j shift}` (ratio 0 means 1; normalised to 1 on construction, so a learnable `ratio` starts from, and has the gradient of, 1), `Y_tt = Y_s + jb/2`, `Y_ff = Y_tt / \|tau\|^2`, `Y_ft = -Y_s / conj(tau)`, `Y_tf = -Y_s / tau` |
| `ConstantPowerLoad(P, Q)` (`load`) | bus `->` ground | `v conj(i) = P + jQ`, the power absorbed |
| `Shunt(g, b)` (`shunt`) | bus `->` ground | `i = (g + jb) v`: `g` consumes `g \|v\|^2`, a positive `b` injects `b \|v\|^2` |
| `VoltageControlledGenerator(P, V)` (`generator`) | bus `->` ground | `\|v\| = V` and `Re(v conj(i)) = -P`: injects the real power `P` and whatever reactive power holds the magnitude (implicit: its current is an unknown) |

`ConstantPowerLoad` is the law of [AC analysis](ac.md#impedances-constant-power-loads-and-converters),
with `P`, `Q` in p.u.; its edge carries the bus's net fixed demand, the loads less the
generation of any generator on a bus solved as PQ. Sign convention: every law absorbs
`v conj(i)`, so a load's `P` is positive and a generator absorbs `-P`. Like the other power
laws, these assume root-mean-square phasors.

The **slack bus** is a second boundary node of the layer, held at `V_set e^{j theta}` (its
generators' voltage setpoint and the bus's angle from the bus table). Its Kirchhoff residual
is not imposed, so it takes up whatever power balances the network, and
`power_flow_solution` reports its generation as `v conj(I)` with `I` the total current its
edges carry away. The boundary voltages are the driver `"power.phi_boundary"` (ground first,
then the slack); `"power.frequency"` is required by every phasor layer and read by no
power-flow law. The state is the **flat start**: every regulated bus (the slack and each PV
bus with generation) at its voltage setpoint, every other bus at 1 p.u., all at the slack
angle; edge currents zero.

### The DC model

`dc_power_flow(net)` builds the DC approximation (all voltage magnitudes 1 p.u., no branch
resistance or charging, small angle differences) on a real `KirchhoffLayer` with no flow
unknowns (the fast path) named `"power_dc"`. The potential is the voltage angle `theta` (rad)
at the buses; there is no ground node. Each in-service branch is one edge from bus `->` to
bus with a `Conductance(1 / (x tau))` (`tau` the tap ratio, 0 meaning 1), and the phase
shifts `sigma` are the layer's head driver `"power_dc.edge_head"` (per branch, in edge order)
holding `-sigma`, so the branch flow is

`Pf = (theta_f - theta_t - sigma) / (x tau)`,

MATPOWER's `Bf theta + Pfinj`. The node sources (`"power_dc.sources"`) are each bus's net
injection `(sum Pg - sum Pd - Gs) / base_mva`, and the slack bus is the layer's one boundary
node at its table angle (`"power_dc.phi_boundary"`). The layer solves `B theta = P` with one
linear solve of the weighted graph Laplacian, through the same `solve_power_flow`. Line
charging, `Bs` and reactive demand play no part. The slack bus's generation is the power its
branches carry away plus its own demand and shunt conductance. Every other generator bus
reports its generators' setpoints from the network, not from the `"power_dc.sources"` driver:
with a batch of sources, the angles, branch flows and slack generation follow each instance,
the other buses' generation does not.

### MATPOWER semantics

The builders follow MATPOWER 8.1 (`makeYbus`, `makeBdc`, `bustypes`, `makeSbus`, `dcpf`)
where it is defined, and refuse where MATPOWER is silent or lenient (generators on one bus
with conflicting `Vg`, other than exactly one slack bus):

- a branch tap ratio of 0 means 1; the phase shift is in degrees in the network and in
  radians in the laws;
- a bus shunt's `Gs`, `Bs` are the MW consumed and MVAr injected at 1 p.u.;
- generators and branches of status `<= 0` are dropped;
- generators on one bus are summed;
- a PV bus with no in-service generator is solved as PQ;
- the slack voltage magnitude is its generators' `Vg` and its angle the bus `Va` column; a PV
  bus's magnitude is its generators' `Vg`;
- the start is flat, as described above, with the slack keeping its file angle.
