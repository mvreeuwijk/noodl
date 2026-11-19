# Constitutive laws and heads

A **constitutive law** is the relation, on an edge kind, between the potential
difference across each edge of that kind and the flow along it. Most constitutive laws give the
flow as a function of the potential difference; some are implicit laws, with the
flow an unknown (an ideal source, an inductance), and some couple several edges (a
transformer's two windings). A **head term** is a term added to that potential difference before
the constitutive law sees it — buoyancy, wind pressure, interfacial drag.

Together they are the replaceable part of the framework. The network, the conservation law and
the solve are fixed; the constitutive law is where your physics goes.

## What a law provides

A law belongs to one edge kind and, in its default potential-controlled form, provides three things: the flow given the potential difference, the derivative of that flow, and a linear surrogate for the starting point.

The derivative is supplied analytically rather than taken from autograd because it is assembled into the Jacobian $A_I \operatorname{diag}(g') A_I^\top$ at every Newton iteration, where an autograd round-trip per iteration would dominate the cost. Solvers that need a law's slopes as local blocks (a law with several ports, an implicit law, a phasor law) use the law's `jacobian`, which a law may give in closed form; otherwise it is computed by automatic differentiation. Every law of the electrical application gives it in closed form, so a circuit's Newton iterations evaluate no derivatives by automatic differentiation.

The linear surrogate matters more than it appears. A nonlinear network solved from a cold start can take many Newton iterations or fail outright; solving the linear surrogate first puts the iteration inside the basin of attraction. Every built-in law provides one.

A law may also report where it stops being smooth. `switching(dp, drivers)` returns values that change sign where the flow switches between two analytic pieces, such as the edges of a regularisation band, in units of the band's width. The default, `None`, means the law is smooth. The Modelica Buildings Library laws report their bands: the power law's `dp = ±dp_turbulent`, a door's buoyancy band and a discretised door's compartment bands. A time integrator can then put a step boundary on every switch (the Modelica runner's `simulate(scheme="midpoint")` does), because a band crossed inside a step spoils the error expansion that Richardson extrapolation relies on.

A law's differentiable quantities are registered parameters, built with `learnable=True`, so that the differentiable solve can reach them (see [Differentiability](differentiability.md)). The rules a new law must follow are in [Extending noodl](../project/extending/index.md).

## Built-in laws

| Law | Relation | Typical use |
|---|---|---|
| `PowerLaw(C, n)` | $q = C \,\mathrm{sign}(\Delta p)\,\lvert\Delta p\rvert^{n}$ | Cracks and leaks. The CONTAM power-law family; $n=0.5$ is a sharp orifice, $n \to 1$ a laminar crack. |
| `Orifice` | `PowerLaw` at $n = 1/2$ | Openings whose area and discharge coefficient are known. |
| `Quadratic(a, b)` | Inverts $\Delta p = a q + b \lvert q \rvert q$ | Quadratic drag; CONTAM's `qfr_*` family. |
| `Conductance(g)` | $q = g\,\Delta p$ | Linear conduction — thermal walls, passive exchange. |
| `FixedFlow(q0)` | $q = q_0$, independent of $\Delta p$ | Constant-flow fans. Declares `dp_independent = True`, which lets the solve take an exact-zero Jacobian column. |
| `FanCurve(coeffs, q_max)` | Cubic pressure-flow curve inverted for $q$ on $[0, q_{\max}]$ | Fans specified by a performance curve. |
| `AffinityFan(curve, ...)` | A `FanCurve` at another speed and air density by the fan laws, $q = s\,r\,q_0(\Delta p/(s^2 r))$ | Scheduled or controlled fans; CONTAM's performance-curve fan with its speed schedule. |
| `Duct(...)` | Colebrook friction, laminar below a transition Reynolds number | Ducts, and Darcy-Weisbach pipes with the water application's `friction="colebrook"`. |
| `AirDuct(length, diameter, roughness, ...)` | Darcy-Weisbach with ContamX's friction factor (laminar, Colebrook, linear between), inverted for $q$ at the entering air's density and viscosity | Duct segments of a building's duct network; CONTAM's Darcy-Colebrook duct element. |
| `DuctTerminal(area, loss, ...)` | $\Delta p = C_e q^2/(2\rho A^2)$, linear below a small flow; $C_e$ grows as the opening closes | The zero-length duct between a junction and its zone; CONTAM's duct terminal with its balancing schedule. |
| `FixedVolumeFlow(volume_flow, ...)` | $q = Q\,\rho_{\text{up}}$, independent of $\Delta p$ | Constant-volume fans: a volume flow delivered at the density of the air drawn in. |
| `PrescribedFlow(key, *, kind)` | $q$ = `drivers[key]`, independent of $\Delta p$ | Flows a closure or an input sets each step, such as an air handler's. |
| `CubicSplineFlow(x, y, ...)` | A measured curve fitted as ContamX fits it (straight ends, clamped cubic between), as $q(\Delta p)$ or inverted from $\Delta p(q)$ | Measured airflow elements; CONTAM's cubic-spline elements. |
| `Damper(...)` | Separate power-law coefficient and exponent per flow direction | Backdraught dampers, one-way devices. |
| `UpstreamDensityPowerLaw(...)` | `PowerLaw` with $C$ scaled by $(\rho_{\text{up}}/\rho_{\text{ref}})^{m}$ | The upstream-density correction on CONTAM power-law elements. |

`noodl.elements` also carries the Modelica Buildings Library (MBL) family, which expresses the
same physics — power laws, tables, doors — as MBL's own equations rather than CONTAM's:
`RegularizedPowerLaw` (built by `regularized_orifice`, `effective_leakage_area`,
`power_law_from_point`, `power_law_from_points` and `power_law_coefficient`) is the power law
family; `SplineFlowTable` a tabulated pressure-flow curve; `OpenDoor`/`OperableDoor` a lumped
door pair (built by `open_door_pair`/`operable_door_pair`) and `DoorCompartment`/
`OperableDoorCompartment`, with the `DoorCompartmentHead` head term, a height-discretised one (built
by `discretized_door`/`discretized_operable_door`, or wrapped as two port streams by
`DoorPortStream`); `AirMedium`, built by `medium(name)`, the fixed-density medium the others
read. See
[Airflow elements](../applications/building/physics.md) for the physics, the
law each one evaluates and how it differs from its CONTAM counterpart.

Applications add their own: the water application contributes `HazenWilliams`,
`CompositeDarcyWeisbach`, `PumpCurve` and `MinorLoss`; the sewer application contributes
`Headspace`; the [electrical application](../applications/electrical/index.md) contributes the circuit
laws `Resistance`, `Capacitance`, `Inductance`, `PrescribedVoltage`, `PrescribedCurrent`,
`JunctionDiode`, `SwitchedConductance`, `TransformerRatio`, `MutualInductance` and
`DependentSource`, most of them in the implicit, multi-port or dynamic forms described
[below](#implicit-coupled-and-dynamic-constitutive-laws).

### A note on `Duct`

`Duct` unrolls a fixed number of Colebrook iterations (default `n_iter=4`) rather than iterating
to convergence, so that the law is a fixed-depth differentiable expression. The consequence is
documented in its own docstring: a **known discontinuity** at the laminar-turbulent transition
$\lvert \Delta p \rvert = \Delta p_t$, of relative size $10^{-6}$ to $10^{-4}$ at the default
depth. This is a property of the flow law as implemented, not a solver artefact, and it is
stated rather than hidden because a Newton solve landing exactly on it will notice.

## A law in a few lines

The minimum is a flow, its derivative and a linear surrogate:


```python
import torch
from noodl.elements.base import ConstitutiveLaw


class Squared(ConstitutiveLaw):
    """q = k * sign(dp) * dp**2."""

    def __init__(self, k, *, kind="pipe", learnable=False):
        super().__init__(kind)
        self.k = self._param(k, learnable)

    def flow(self, dp, drivers=None):
        return self.k * torch.sign(dp) * dp**2

    def dflow(self, dp, drivers=None):
        return 2.0 * self.k * torch.abs(dp)

    def linear_init(self, drivers=None):
        # (offset, slope): the tangent at dp = 0 is flat (singular), so this is a surrogate,
        # the linear law q = k dp, whose solve gives Newton its starting point
        return torch.zeros_like(self.k), self.k
```

Two of them in series between prescribed potentials, `k = 1` then `k = 4`: equal flows need
`(6 - phi_b)^2 = 4 phi_b^2`, so `phi_b = 2` and `q = 16`.

```python
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

net = Network(dtype=torch.float64)
for node in ("a", "b", "c"):
    net.add_node(node)
net.add_edge("a", "b", kind="pipe")
net.add_edge("b", "c", kind="pipe")
layer = KirchhoffLayer(
    net, "p", [Squared(torch.tensor([1.0, 4.0], dtype=torch.float64))], boundary=["a", "c"])
phi, q = layer.solve(torch.tensor([6.0, 0.0], dtype=torch.float64))
print([round(v, 9) for v in phi.tolist()], [round(v, 9) for v in q.tolist()])
```

```text
[6.0, 2.0, 0.0] [16.0, 16.0]
```

## Implicit, coupled and dynamic constitutive laws

Write `flow`; if you cannot, write `residual`. A law that gives its flow, `q = g(dp)`, one flow
per edge as a function of that edge's own potential difference, writes `flow(dp, drivers)`. Every
built-in airflow, water and sewer law does, and a layer whose laws all do is solved by nodal
analysis. A law that cannot (a flow-controlled law, an ideal source, an inductance) subclasses
`ImplicitLaw` and writes `residual(dp, q, drivers, ctx)` instead, the relation between the
potential difference and the flow, zero when they agree; the layer then adds its flow as an
unknown. Every law has `residual`: for a law with a flow it is `q - flow(dp)`, which the solve
never calls. To read the law context (time, frequency, step-start values), name the third
parameter of `flow` `ctx`. A `KirchhoffLayer` with an implicit law or a multi-port law adds those
edges' flows as unknowns
([Implicit, coupled and dynamic laws](../reference/transport-options.md#implicit-coupled-and-dynamic-laws));
a dynamic or `ctx`-reading law that gives its flow adds none. The class-level declarations a law makes about itself (`ports`, `dynamic`, `phasor_capable`, `reads_heads`) are listed in [Extending noodl](../project/extending/index.md).

**The two forms.** A law with a flow is the nodal half of circuit theory: its flow is eliminated
in favour of the potentials. An `ImplicitLaw` covers what cannot be written that way: a
flow-controlled law `dp = h(q)`, an ideal potential source `dp = V` (no flow is a function of
`dp`), an ideal constraint inside a multi-port law, and a law whose flow and potential are
related through stored state. A subclass implements `residual`, which returns one value per
edge, with the shape of `dp`. An implicit law has no `flow`: its flow is read from the layer's
solution.

A loss given as potential difference against flow, `dp = r q |q|`, in series with a linear
conductance, between two prescribed potentials:

```python
import torch

from noodl.elements import Conductance, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64


class QuadraticLoss(ImplicitLaw):
    """A loss given as potential difference against flow: dp = r q |q|."""

    def __init__(self, r, *, kind="loss", learnable=False):
        super().__init__(kind)
        self.r = self._param(r, learnable)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.r * q * q.abs()


net = Network(dtype=F64)
for node in ("in", "mid", "out"):
    net.add_node(node)
net.add_edge("in", "mid", kind="loss")
net.add_edge("mid", "out", kind="pipe")
layer = KirchhoffLayer(
    net, "w",
    [QuadraticLoss(torch.tensor(2.0, dtype=F64)),
     Conductance(torch.tensor(1.0, dtype=F64), kind="pipe")],
    boundary=["in", "out"],
)
phi, q = layer.solve(torch.tensor([10.0, 0.0], dtype=F64))
print(layer.implicit_edges.tolist())
print(f"phi_mid = {float(phi[1]):.6f}, q = {[round(v, 9) for v in q.tolist()]}")
```

```text
[0]
phi_mid = 2.000000, q = [2.0, 2.0]
```

The flow satisfies `2 q^2 + q = 10`, so `q = 2` and `phi_mid = 2`. `implicit_edges` lists the
edges whose flow is an unknown, in layer edge order.

## Heads

A head term contributes an additive term to the potential difference, before the law evaluates:

$$
q = g\!\left(A^{\top}\phi + \text{head}\right)
$$

A layer adds one value per edge, the driver `"<layer>.edge_head"`, to every edge's potential
difference. The closure `EdgeHeads` writes it: it adds up the head terms of one layer, each on
the edges of its own kind (see [Heads](layers.md#heads-stack-wind-and-drag)). `noodl.heads`
provides the terms:

| Head term | What it adds |
|---|---|
| `DriverHead(kind, key)` | A head read from a driver: the value `drivers[key]`, one per edge of the kind. |
| `Stack(...)` | Buoyancy: the hydrostatic pressure difference from a density difference over the height between an edge's endpoints. `Stack.from_network(net, kind)` builds it from the nodes' own `z_ref` attributes. |
| `DuctStack(kind, ...)` | The air column of a duct segment between its two junctions, at the density of their mean temperature. `DuctStack.from_network(net, kind)` builds it from the junctions' `z_ref`. |
| `Wind(...)` | Wind pressure on a façade, from wind speed, direction and a pressure coefficient. `Wind.from_network(net, kind, ambient=...)`. |
| `WindProfile(...)` | A vertical wind profile, identified by CONTAM's profile number. |

Head terms are reused across applications far more than constitutive laws are. The sewer
application's headspace air layer solves with the *same* `Stack` the building application
uses for room buoyancy, plus a sewer-specific `Drag` term: the physics is different, the
structure is not.

### Why a head cannot read the potential

A head term is a function of the drivers alone: it may read anything from the `drivers` mapping, and nothing else, not `phi`, not another layer's state.

This is structural. Keeping a head independent of $\phi$ is what keeps the Newton Jacobian exactly $A_I \operatorname{diag}(g') A_I^\top$, with no extra coupling term. A head that read $\phi$ would silently invalidate the Jacobian every solver in the package assembles.

The sewer application's `Drag` is the worked consequence. Physically, the drag the moving water
surface exerts on the headspace air depends on the *relative* velocity $(U_s - U_{\text{air}})$,
and $U_{\text{air}}$ is exactly the quantity the air layer is solving for. Modelling it that way
would break the rule, so `Drag` uses the absolute surface velocity instead. Its docstring says so
and records the relative-velocity form as out of scope rather than quietly approximating it —
"structurally not differentiable by this framework's design, not an oversight."

If you genuinely need a term that depends on solved state, it belongs in a **closure**, which
runs between layer solves and writes driver values. See [Transport and model options](../reference/transport-options.md#closures).
Because the head reaches the layer as a driver, a gradient reaches whatever produced it, and
`EdgeHeads` holds no tensor the differentiable solve would miss.

## Withdrawals and node storage

Two objects act at a node rather than on an edge, each as a function of that node's own
potential, and each adds only a diagonal entry to the Jacobian:

- **`NodeStorage`** (`noodl.NodeStorage`, `noodl.LinearStorage`): an amount stored at the node,
  which the layer turns into a withdrawal over each step with its scheme's step rate. A
  capacitor to ground, a tank, a zone's air mass. Every layer takes it, with or without
  flow unknowns (not a phasor layer); at the consistent initial point a storage with an
  `initial()` potential holds its node there. Under
  `scheme="forward_euler"` a linear storage sits at a boundary node instead (an EPANET tank),
  held through the solve and moved by explicit Euler after it. See
  [Node storage](../reference/transport-options.md#node-storage).
- **`NodeWithdrawal`** (`noodl.NodeWithdrawal`): a withdrawal that is **not** storage, `w(phi)` at the
  current instant: EPANET's pressure-driven demand, a leak. A storage written as a withdrawal
  with its own step-start drivers duplicates what the layer already does; give it to
  `KirchhoffLayer(storage=...)` instead.

A dynamic edge law (`Capacitance`, `Inductance`) and a node storage both turn a change into a
rate with `noodl.elements.step_rate`: `change / dt` under backward Euler, `2 change / dt -
previous` under the trapezoidal scheme, with `step_rate_factor` (`1/dt` or `2/dt`) its
derivative.
