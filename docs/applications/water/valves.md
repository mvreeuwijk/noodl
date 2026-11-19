# Water distribution: valves and link status

How valves, closed links and check valves enter the solve, and how to set a link's status and setting by name. Head loss and pumps are on [Physics](physics.md); the overview is [Water distribution](index.md).

## Valves

A **TCV** (`ThrottleValve`) is a minor loss whose coefficient is its setting, and an **FCV**
(`FlowControlValve`) a fixed flow. A **PBV** (pressure breaker) and a **GPV** (general purpose
valve) follow EPANET 2.2's own coefficient routines (`hydcoeffs.c`, `pbvcoeff` and `gpvcoeff`)
as `BreakerValve` and `GeneralPurposeValve`. All four are implicit laws, whose flow is an
unknown of the layer:

- **PBV:** the head drop from the valve's start node to its end node is its setting,
  whichever way the water flows. Where the open valve's minor loss $m q^2$ exceeds the setting,
  or the setting is 0, it is an open valve instead, $h = m q |q|$, with EPANET's minor-loss
  coefficient $0.02517 K / d^4$ (ft, cfs) and its rounded flow-unit factor.
- **GPV:** $h = \operatorname{sgn}(q)\,(h_0 + r |q|)$ on the segment of its head-loss curve
  that brackets $|q|$ (the first or last segment, extended, outside the curve), with
  $\operatorname{sgn}(0) = +1$.

A valve's flow is an unknown of the solve (sparse LU).

A **PRV** (`PressureReducingValve`) holds the head at its end node at no more than
$H_\text{set} = z + s$ (the node's elevation $z$ plus the setting $s$); a **PSV**
(`PressureSustainingValve`) holds the head at its start node at no less than that node's
$z + s$. Each is one exact complementarity condition on the absolute head at its controlled
node (the law reads its end nodes' heads, `ConstitutiveLaw.reads_heads`), solved in the same
Newton iteration as everything else, with no outer loop over states and no memory of the
previous one:

$$\min\bigl(\kappa q,\ \max(\text{held},\ K_m q|q| - \Delta\phi)\bigr) = 0,$$

with $\text{held} = H_v - H_\text{set}$ for a PRV and $H_\text{set} - H_u$ for a PSV, the
controlled head beyond its setting. The selected row is the valve's state: $\kappa q$
(no flow) is `CLOSED`, the held row `ACTIVE` (the controlled head at its setting), the
minor-loss row `OPEN` (the controlled head short of its setting, the valve fully open).
That is EPANET 2.2's three states (`hydstatus.c`, `prvstatus` and `psvstatus`) without
their hysteresis. Inside EPANET's `HTOL` band (1.524e-4 m below the switch) EPANET keeps an
active PRV active with the downstream head at its setting above the upstream head, a
negative drop through the valve; noodl gives the ideal valve there, `OPEN` with no drop, and
the heads differ by less than `HTOL`.

The rows are selected with `torch.where` and fixed tie rules: at a tie of the held and open
rows the held row; the no-flow row only where the flow is not forward. With the fixed
$\kappa$ alone (1 m per m³/s for a valve without a minor loss) an iterate carrying 20 L/s
forward through a PRV whose downstream head is 7 m above its setting picks the no-flow row
and cuts off the junctions downstream, with their demand (measured: a singular step). A
side of the valve with no demand and no other way to a reservoir or tank (a dead end)
carries no flow; the valve then takes a row that ties that side's head, with EPANET's
state: a dead end on a PRV's end leaves it `ACTIVE` at its setting (or `OPEN` at the
upstream head below it), on a PSV's end `OPEN` at its upstream head (or `CLOSED` where that
is below its setting); a dead end on either valve's start closes it (no water flows out of
a group without demand). A closed valve's dead end is held at the head of its other side,
the `1 / CBIG` limit. A valve with both sides dead (a part without demand that closed
links cut off) is refused by name.

The gradient is the implicit-function one of the selected row: through an `ACTIVE` PRV the
downstream head moves one for one with the setting, through an `OPEN` or `CLOSED` one not at
all. The solution is continuous across a switch and its derivative jumps there.
`build_model(valve_smoothing=eps)` (metres) replaces the `min` by the Fischer-Burmeister
function $a + b - \sqrt{a^2 + b^2 + 2\epsilon^2}$ and the `max` by its smooth counterpart, for an
optimiser that must cross a switch: the derivative is then continuous, and the answer moves
by $O(\epsilon)$ near a switch (within 2e-3 m of the exact one at $\epsilon$ = 1 mm).

`validate()` refuses, by name, a PRV or PSV whose controlled node is a reservoir or tank,
whose setting or minor loss is negative, and two valves that hold one node twice: two PRVs
sharing their end node or in series, two PSVs sharing their start node or in series, a PSV
starting at a PRV's or an FCV's end, a PRV ending at an FCV's start (EPANET's own errors 219
and 220, `valvecheck`); the error names both valves. Unlike EPANET, a PRV or PSV may start or
end at a reservoir or tank on its other side.

## Link status, settings and check valves

Every link's status, setting and allowed direction are drivers, one value per link in
`model.link_names` order (pipes, pumps, then TCVs, FCVs, PBVs, GPVs, PRVs, PSVs):

| Driver | Values |
|---|---|
| `"water.status"` | `CLOSED` 0, `OPEN` 1, `ACTIVE` 2 (`noodl.apps.water.controls`); `ACTIVE` only for a valve, which then acts on its setting |
| `"water.setting"` | a pump's relative speed, a PBV's head drop, a PRV's or PSV's pressure head (m; the layer's potential unit), an FCV's flow (m³/s), a TCV's loss coefficient; unused for pipes and GPVs |
| `"water.one_way"` | 0 both directions, +1 forward flow only, -1 reverse flow only |

`build_model` fills them from the file: a `CLOSED` pipe or a `[STATUS]` word sets the status, a
`CV` pipe is forward-only, a valve's setting (or a numeric `[STATUS]` value) is its setting,
and every valve starts `ACTIVE`. On a pump `[STATUS]` acts as EPANET's `changestatus`: a number is
its speed (the pump `OPEN` at it, `CLOSED` at 0), `OPEN` runs it at speed 1, `CLOSED` keeps its
speed. They are set by link name like any other driver:

```python
from noodl.apps.water import (
    CLOSED, build_model, initial_drivers, link_states, twoloop, water_steady,
)

model, state, drivers = build_model(twoloop())
closed = initial_drivers(model, values={"water.status": {"P7": CLOSED}})
final = water_steady(model, state, closed)
print(link_states(model, final, closed).tolist())
```

```text
[1, 1, 1, 1, 1, 1, 0, 1]
```

A `CLOSED` link carries no flow: its residual is $\kappa q$. A valve whose status is `OPEN`
is its minor loss, $h = K_m q|q|$, whatever its setting. A check valve is one exact
complementarity condition, solved in the same Newton iteration as everything else, with no
outer loop over statuses: $\max(-\kappa q,\ \Delta\phi - h(q)) = 0$, either the pipe flows
forward with $\Delta\phi = h(q)$, or it holds $q = 0$ against a reverse head. $\kappa$, a pipe's own head-loss slope at 1 ft/s,
only scales the row. Newton picks the no-flow branch only where the open equation's residual
is below both $-\kappa q$ and $-h'(q)\,q$: with $\kappa$ alone, an iterate that overshoots
forward closes an open check valve, sets every pipe in series with it to zero flow (to round-off),
and cycles (`FlowControlledPipe`). An iterate that would close every check valve feeding a
junction, which leaves the step singular, keeps open the one whose open equation predicts the
most forward flow, so the step is taken; `water_steady` then refuses a converged check valve
carrying flow the wrong way. Closed links and check valves need the pipes' resistance form
(`pipe_form="flow"`, the default); `pipe_form="potential"` refuses them.

`link_states(model, state, drivers)` gives every link's state in a solution: `CLOSED`, `OPEN`
or `ACTIVE`, the codes `wntr` reports EPANET's in (an `ACTIVE` PRV or PSV: the row its
condition selects there). `water_steady` refuses, before it solves, a
junction with a demand that closed links cut off from every reservoir and tank, and one that
no water reaches through the open links in the directions check valves and full or empty tanks
leave them (a pipe into an empty tank only fills it). A group of
junctions without demand that is cut off at the solution, by closed links or by check valves
holding no flow, has no head of its own in the exact equations: any head that keeps those
valves shut solves them. EPANET gives each closed link the conductance $1/C_\text{BIG}$ and
opens a check valve whose upstream head is the higher, and as $C_\text{BIG}$ grows its
heads tend to the solution of the convex program

$$\min_H \tfrac12 \sum_{(a,b)} (H_a - H_b)^2 \quad \text{subject to} \quad H_u \le H_v
\text{ for every check valve } u \to v,$$

the sum over every closed link and every check valve holding no flow, with the heads outside
the group at the solve's own. It is the resistor-and-diode network with one conductance per
link and an ideal diode across each check valve: the constraints' multipliers are the diodes'
flows, so a check valve is `OPEN` where its multiplier is positive (it ties its two ends) and
`CLOSED` where it is zero. `water_steady` sets the group's heads to that solution, with every
flow into it zero to round-off (`noodl.apps.water.elements.PIN_KEY`), by the dual active-set
method of Goldfarb and Idnani on the group's nodes alone (`noodl.solvers.qp.solve_qp`: the objective
rises strictly over every full step and no active set returns, so it ends). For a single
junction behind shut valves that is the plain average of the far ends; a valve the average
would drive is open and the junction takes its far end's head; and a valve that opened in an
intermediate active set is shut again where a later one leaves it with nothing to carry. The
heads are a linear map of the solved heads around them for the active set found, so a
gradient reaches them.

The TCV's coefficient is EPANET's own, $0.02517\,K/d^4$ in feet and cfs: $8/(32.2\,\pi^2)
= 0.025173$ (EPANET's 32.2 ft/s² gravity) rounded to four figures, 9.2e-4 below the exact
$8/(g\pi^2) = 0.0251933$ a `MinorLoss` uses.

