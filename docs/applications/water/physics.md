# Water distribution: physics

The equations the water distribution application solves and the assumptions behind them: head loss in pipes, pumps, pressure-driven demand and emitters. Valves, link status and check valves are on [Valves and link status](valves.md), tanks, controls and rules on [Tanks, controls and rules](tanks-controls.md), and the water-quality equations on [Water quality](quality.md). The application's overview is [Water distribution](index.md).

## Background

A pressurised water-distribution network is the opposite case to the kinematic-wave sewer, and
exactly the framework's own: every pipe runs full, and head loss is a monotone function of the
head difference at each pipe, pump and valve, so `noodl.apps.water` is an ordinary
`KirchhoffLayer` (see [Theory and related work](../../concepts/theory.md)) with Hazen-Williams or Darcy-Weisbach pipes, `PumpCurve` and
minor-loss elements, solved by the same batched, damped Newton iteration as every other
Kirchhoff layer in this codebase. SWMM's dynamic-wave engine is the heavier comparison: it solves
the sewer's full 1-D Saint-Venant equations with surcharging and reverse flow, "a genuinely
dynamic (not quasi-steady) network". SWMM's KINWAVE routing, which shares the continuity-first
assumption, is the comparison target for the kinematic-wave path and its live dynamic-wave runs
for the dynamic-wave path; see the [sewer background](../sewer/physics.md).

## Head loss in pipes

### Hazen-Williams

$$
h_L = K q^{1.852} + m q^2, \qquad
K = \frac{10.666829500036\, L}{C^{1.852} d^{4.871}}, \qquad
m = \frac{K_{\text{minor}}}{2 g A^2}
$$

The SI constant was re-derived from EPANET's own US constant 4.727 and matches `wntr`'s `hw_k` to
1.04e-9 relative. For a network read from a file, $K$ carries EPANET's own rounded flow-unit
factor, as Chezy-Manning and Darcy-Weisbach do: EPANET divides a file's flows by `LPSperCFS =
28.317`, `GPMperCFS = 448.831`, ... (`types.h`; exact 28.3168466 and 448.8311688) and evaluates
the law in those cfs, so $K$ is multiplied by $(c \cdot 0.3048^3)^{1.852}$ with $c$ EPANET's cfs
per m³/s (`HazenWilliams(cfs_per_m3s=...)`), and the minor loss is EPANET's $0.02517\,K/d^4$
(ft, cfs) with the same factor. A network built by hand has no flow unit and keeps the exact
conversion and $m = K_{\text{minor}}/(2 g A^2)$. With the exact factor an LPS file's heads are
1.0e-5 of the losses off EPANET's (3.1e-5 relative on a network with 110 m of losses); with the
rounded one the two-loop heads agree to 5.3e-8. With no minor loss the inversion is closed form; with one, it is a batched
monotone root solve bracketed by the pure Hazen-Williams flow as an upper bound.

Below `dp_transition` (default $10^{-9}$ m) the law blends to a laminar linear form to keep the
Jacobian finite. That value is empirically chosen: at 1e-9 the 24-hour tank trajectory is
converged to 8.18e-5 m against EPANET, and Newton iteration counts run 50/93/136/179 at
1e-3/1e-6/1e-9/1e-12.

### Chezy-Manning

`HEADLOSS C-M` is EPANET 2.2's resistance (`hydcoeffs.c`, `resistcoeff`; `input1.c`, `Hexp = 2`),
in feet and cfs with Manning's roughness $n$ (dimensionless):

$$
h_L = (R + m) q^2, \qquad R = \left(\frac{4 n}{1.49\,\pi d^2}\right)^{2} \left(\frac{d}{4}\right)^{-1.333} L
$$

The exponent $-1.333$ is EPANET's literal, not $-4/3$. `ChezyManning` converts $R$ to metres per
$(\mathrm{m^3/s})^2$ with the file's own flow factor (as Darcy-Weisbach does), takes EPANET's
`RQtol` line below the gradient floor, and carries the minor loss $m$ as `K / d^4` with EPANET's
rounded constant. It works in metres of head, so, like Hazen-Williams, it refuses a
non-default `VISCOSITY` by name; a `SPECIFIC GRAVITY` only converts the file's pressures to
heads, which the reader does. It enters the solve through
`FlowControlledPipe` like Hazen-Williams, so check valves and closed pipes work unchanged.

### Darcy-Weisbach

`CompositeDarcyWeisbach` (`friction="composite"`) is EPANET 2.2's own composite law:
Hagen-Poiseuille, then Dunlop's transition cubic, then Swamee-Jain. It is transcribed from its source
(`hydcoeffs.c`, `DWpipecoeff` and `frictionFactor`), in EPANET's internal feet and cfs:

$$
h_L = \begin{cases}
(16 \pi \nu d R + m \lvert q \rvert)\, q & Re \le 2000 \quad \text{(Hagen-Poiseuille)} \\
(f(Re) R + m)\, \lvert q \rvert q & Re > 2000
\end{cases}
\qquad R = \frac{L}{2 \cdot 32.2\, d A^2}, \quad m = \frac{0.02517\, K_{\text{minor}}}{d^4}
$$

with $f$ Dunlop's cubic interpolation for $2000 < Re < 4000$ and Swamee-Jain above. The
composite is $C^1$: the cubic matches $64/Re$ and Swamee-Jain in value and slope at its ends.
EPANET's constants are kept as EPANET has them, because they are part of what it computes:
$g = 32.2$ ft/s² (not 32.174), the rounded minor-loss factor 0.02517, water at
$\nu = 1.1 \times 10^{-5}$ ft²/s times `[OPTIONS] VISCOSITY`, and the flow-unit factor EPANET
divides a file's flows by (`LPSperCFS = 28.317`, `GPMperCFS = 448.831`, ...;
`WaterOptions.flow_units` carries the file's unit). EPANET has no low-flow linearisation for D-W:
the laminar branch is already linear through $q = 0$, so the element needs no transition blend.
The flow is found by a safeguarded Newton inversion of the odd law $h_L(q)$ with EPANET's own
analytic gradient, and differentiated by the implicit-function rule.

This is the default whenever the head loss is D-W, including every `.inp` read with
`HEADLOSS D-W`, because the application's reference is EPANET. `friction="colebrook"` selects
the framework's generic `Duct` instead (implicit Colebrook, a straight laminar line to its
Re = 2000 value, water at $1.002\times10^{-3}/998.2$ m²/s, standard gravity); it is a different
law and does not reproduce EPANET (3.6e-4 on the two-loop heads, up to 81 % on a pipe's head
loss near Re = 2000).


## Pumps

Each pump takes the curve EPANET 2.2 gives it (`input2.c`, `updatepumpparams`), and its head
gain at flow $q$ and relative speed $w$ follows EPANET's `pumpcoeff` (`hydcoeffs.c`), with the
affinity laws for the speed:

| Pump in the file | Curve | Head gain |
|---|---|---|
| `HEAD` curve of one point $(q_d, H)$ | `PumpCurve`, `three_point_curve`: EPANET's points $(0, 1.33334 H)$, $(q_d, H)$, $(2 q_d, 0)$ fitted as three points: $h_0 = 1.33334 H$, $n = 1.99998$, $r = 0.33334 H / q_d^n$ | $w^2 h_0 - r w^{2-n} q^n$ |
| `HEAD` curve of three points, the first at zero flow | `PumpCurve`, `power_curve` (EPANET's `powercurve`): $h_0$ the first head, $n = \ln\frac{h_0 - h_2}{h_0 - h_1} / \ln\frac{q_2}{q_1}$, $r = (h_0 - h_1)/q_1^n$ | $w^2 h_0 - r w^{2-n} q^n$ |
| any other `HEAD` curve | `PiecewisePumpCurve`: straight segments (`curvecoeff`), the end segments extended | $w^2 h(q/w)$ |
| `POWER` $P$ | `ConstantPowerPump`: $8.814\,P/q$ in feet, horsepower and cfs | $8.814\,P w^3 / q$, converted to SI |

EPANET's own rounded 1.33334, not the manual's 133 % with exact thirds (which would force
$n = 2$, as wntr's `get_head_curve_coefficients` has it): with $n = 2$ a pump at zero flow
sat $6.7 \cdot 10^{-6} H$ below EPANET's (1.05e-6 relative at 63 m) and Net1's flows
missed EPANET's by 2.9e-6 relative instead of 2.6e-7 (measured).
`q_max = w (h_0/r)^{1/n}` is a power function's flow at zero head gain.

Every pump enters the solve in its **head-gain form**, whatever the `pipe_form`, as EPANET and
wntr write it: `FlowControlledPump`, $H_{\text{discharge}} - H_{\text{suction}} =
h_{\text{gain}}(q)$ with the pump's flow an unknown. One law holds every pump of a network,
each on its own curve. The gain is strictly decreasing in $q$, so every head difference has
exactly one flow on the curve:

- where a power function's slope falls below EPANET's `RQtol` (at low flow when $n > 1$) or
  rises above its `CBIG` (at low flow when $n < 1$), and where a constant-power pump's slope
  $8.814 P w^3/q^2$ leaves `[RQtol, CBIG]`, the gain is the curve's tangent there (wntr's
  `get_pump_line_params` construction), continuous in value and slope. EPANET's own lines
  there are its iteration's linearisation, not a decreasing curve; the two agree wherever a
  running pump works;
- a segmented curve whose first point is above zero flow shuts at that point's head, as
  EPANET's `pumpstatus` does (below the point's flow the gain is that head plus a line of
  slope `RQtol`).

The pump's speed is the per-link driver `"water.setting"` (its `SPEED` in the file at the
build), so a calibration can differentiate through it. Its status is `"water.status"`. The
residual of each pump, with $r = dp + h_{\text{gain}}(q)$, is

- $\kappa q$ (no flow) when its status is `CLOSED` or its speed is 0;
- otherwise $\max(-\kappa q, r)$: the pump runs on its curve with $q \ge 0$, or it is shut,
  $q = 0$, where the head asked of it is at or above its shut-off head $h_{\text{gain}}(0)$.
  EPANET closes such a pump (`pumpstatus`, `XHEAD`) and leaves it the closed-link leak
  `excess / CBIG`; here there is no leak: its row is $\kappa q = 0$, so its flow is zero to round-off, and `link_states` reports it `CLOSED`.

The branch is selected as for a check valve (`FlowControlledPipe`): the no-flow row only where
$r < -\kappa q$ and the open row's own Newton step would reverse the flow, never at a tie (a
pump with no demand behind it runs at zero flow on its curve, `OPEN`, as EPANET reports it),
and never where shutting the pump, together with the check valves the same iterate shuts
(`ShutLinks`, which the pipes' law fills first), would cut a group of nodes off from every
reservoir and tank. $\kappa$ is 1 m per m³/s. The Jacobian is the closed form `(1, -h'_{\text{gain}})` on
the curve's row.

`PumpCurve.flow` still gives the inverted curve, $q = w((h_0 + dp/w^2)/r)^{1/n}$, with a laminar
blend near shut-off, but the solve does not use it: its slope is infinite at shut-off and
exactly zero beyond it. A Newton iterate beyond shut-off then has no Jacobian entry between the
pump's two sides, and with a pressure-driven demand downstream only the demand's own small slope
ties those heads to a boundary; the next step throws them far outside the curve and the system
goes singular. That was measured on a reservoir, a pump and a pipe feeding a PDA junction
(singular from the fifth iterate on). In the head-gain form the pump's row ties its two heads
together on every iterate, and the same network converges quadratically, in 5 Newton
iterations (19 with `pipe_form="potential"`).

## Pressure-driven demand

EPANET's Wagner function (Manual eq. 13.6), with $p = H - z$:

$$
d(p) = \begin{cases}
D & p \ge P_{\text{req}} \\
D \left(\dfrac{p - P_{\min}}{P_{\text{req}} - P_{\min}}\right)^{e} & P_{\min} < p < P_{\text{req}} \\
0 & p \le P_{\min}
\end{cases}
$$

Implemented with a guarded kink so both `where` branches stay finite and differentiable.
`PressureDrivenDemand` refuses `p_req <= p_min` — a zero-span demand curve.

## Emitters

An emitter is a flow that leaves a junction as a power of its pressure head,
$q = \operatorname{sign}(p)\, C |p|^{n}$ with $p = H - z$ and $n$ the `EMITTER EXPONENT` (default
0.5), `EmitterOutflow` (a potential-dependent `NodeWithdrawal`). EPANET models it as a fictitious
pipe to a reservoir at the junction's elevation (`hydcoeffs.c`, `emittercoeffs`,
`emitterheadloss`); that loss is odd in the flow and not clipped, so **at negative pressure an
emitter is an inflow** (verified against EPANET). Below `dp_transition` (1e-9 m) the law is its
tangent line, which keeps the slope finite at $p = 0$. The coefficient is differentiable
(`learnable`), and matches central differences to 4.8e-11 relative. Emitters add no work to a
network without them.

