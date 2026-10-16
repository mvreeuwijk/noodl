# Constitutive laws and drives

A **constitutive law** is the relation, on an edge kind, between the potential
difference across each edge of that kind and the flow along it. Most constitutive laws give the
flow as a function of the potential difference; some are relations between the two, with the
flow an unknown (an ideal source, an inductance), and some couple several edges (a
transformer's two windings). A **drive** is a term added to that potential difference before
the constitutive law sees it — buoyancy, wind pressure, interfacial drag.

Together they are the replaceable part of the framework. The network, the conservation law and
the solve are fixed; the constitutive law is where your physics goes.

## The `ConstitutiveLaw` contract

`noodl.elements.base.ConstitutiveLaw` subclasses `torch.nn.Module`. A law owns one edge `kind`.
A constitutive law in the default, potential-controlled form provides three things:

```py
class ConstitutiveLaw(torch.nn.Module):
    kind: str

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """The branch law: flow given the potential difference."""

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Its derivative, d(flow)/d(dp). This is the Jacobian the Newton solve assembles."""

    def linear_init(self, drivers: Mapping | None = None) -> tuple[Tensor, Tensor]:
        """A linear surrogate (offset c, slope k), q ~ c + k dp, for the starting point."""
```

`dflow` is supplied analytically rather than taken from autograd because it is assembled into
the Jacobian $A_I \operatorname{diag}(g') A_I^\top$ at every Newton iteration, where an
autograd round-trip per iteration would dominate the cost.

`linear_init` matters more than it appears. A nonlinear network solved from a cold start can
take many Newton iterations or fail outright; solving the linear surrogate first puts the
iteration inside the basin of attraction. Every built-in law provides one.

A law may also report where it stops being smooth. `switching(dp, drivers)` returns
values that change sign where the flow switches between two analytic pieces, such as the edges
of a regularisation band, in units of the band's width (`noodl.elements.base.band_edges(x, w)` builds the pair
`(x - w)/w`, `(x + w)/w`). The default, `None`, means the law is smooth. The MBL laws report
their bands: the power law's `dp = ±dp_turbulent`, a door's buoyancy band and a discretised
door's compartment bands. A time integrator can then put a step boundary on every switch
(`simulate(scheme="midpoint")` in the Modelica runner does), because a band crossed inside a
step spoils the error expansion that Richardson extrapolation relies on.

### Parameters and learnability

A law's differentiable quantities must be **registered parameters**, constructed with
`learnable=True`:

```py
leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65), learnable=True)
```

This is a hard requirement of the differentiable solve, not a style preference. A bare tensor
held with `requires_grad=True` outside `named_parameters()` is invisible to the adjoint, and the
solve raises naming the offending law and attribute rather than silently returning an absent
gradient. See [Differentiability](differentiability.md#the-contract).

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
`OperableDoorCompartment`, with the `DoorCompartmentHead` drive, a height-discretised one (built
by `discretized_door`/`discretized_operable_door`, or wrapped as two port streams by
`DoorPortStream`); `AirMedium`, built by `medium(name)`, the fixed-density medium the others
read. See
[Airflow elements](../applications/building_physics.md#airflow-elements) for the physics, the
law each one evaluates and how it differs from its CONTAM counterpart.

Applications add their own: the water application contributes `HazenWilliams`,
`CompositeDarcyWeisbach`, `PumpCurve` and `MinorLoss`; the sewer application contributes
`Headspace`; the [electrical application](../applications/electrical.md) contributes the circuit
laws `Resistance`, `Capacitance`, `Inductance`, `PrescribedVoltage`, `PrescribedCurrent`,
`JunctionDiode`, `SwitchedConductance`, `TransformerRatio`, `MutualInductance` and
`DependentSource`, most of them in the implicit, multi-port or dynamic forms described
[below](#implicit-coupled-and-dynamic-constitutive-laws).

**Pre-rename names.** Identifiers now say what a law does physically, not which tool it
comes from. The earlier names still import, as aliases of the same objects:

| Pre-rename name | Name |
|---|---|
| `MBLPowerLaw` | `RegularizedPowerLaw` |
| `mbl_orifice`, `mbl_ela` | `regularized_orifice`, `effective_leakage_area` |
| `mbl_point`, `mbl_points`, `mbl_coefficient` | `power_law_from_point`, `power_law_from_points`, `power_law_coefficient` |
| `MBLTable` | `SplineFlowTable` |
| `MBLDoorOpen`, `MBLDoorOperable` | `OpenDoor`, `OperableDoor` |
| `mbl_door_pair`, `mbl_operable_door_pair` | `open_door_pair`, `operable_door_pair` |
| `MBLDoorCompartment`, `MBLDoorCompartmentOperable` | `DoorCompartment`, `OperableDoorCompartment` |
| `MBLDoorPortStream` | `DoorPortStream` |
| `mbl_discretized_door`, `mbl_discretized_operable_door` | `discretized_door`, `discretized_operable_door` |
| `MBLMedium` | `AirMedium` |
| `EpanetDarcyWeisbach`, `epanet_friction_factor` (water) | `CompositeDarcyWeisbach`, `composite_friction_factor` |
| `friction="epanet"` (water `build_model`) | `friction="composite"` |
| `geometry="swmm"` (sewer) | `geometry="tabulated"` |
| `soulhac_shape`, `macdonald_profile`, `munich_idealised`, `SIRANE_EXCHANGE`, `KAPPA_MUNICH`, ... (street air quality) | `bessel_shape_parameter`, `canopy_displacement_roughness`, `twelve_street_grid`, `EXCHANGE_SIGMA_W_RATIO`, `KAPPA_041`, ...: the full table is under [Earlier option names](../applications/street_aq.md#earlier-option-names) |

### A note on `Duct`

`Duct` unrolls a fixed number of Colebrook iterations (default `n_iter=4`) rather than iterating
to convergence, so that the law is a fixed-depth differentiable expression. The consequence is
documented in its own docstring: a **known discontinuity** at the laminar-turbulent transition
$\lvert \Delta p \rvert = \Delta p_t$, of relative size $10^{-6}$ to $10^{-4}$ at the default
depth. This is a property of the flow law as implemented, not a solver artefact, and it is
stated rather than hidden because a Newton solve landing exactly on it will notice.

## Writing your own

The minimum is a `flow`, a `dflow` and a `linear_init`:

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

Two rules worth stating explicitly, because both produce wrong answers rather than errors:

- **Regularise at zero.** A law whose derivative diverges or vanishes at $\Delta p = 0$ will
  stall Newton at the first iteration, since a cold start usually sits there. Every built-in
  law is regularised there, but not all the same way: `PowerLaw` blends to a laminar
  (linear) form below a `dp_transition` threshold, while the MBL power-law family blends to a
  polynomial below a (wider) `dp_turbulent` threshold — see
  [Airflow elements](../applications/building_physics.md#airflow-elements). `PowerLaw` with
  $n < 1$ has infinite slope at the origin without it.
- **Keep `dflow` consistent with `flow`.** Nothing checks them against each other at runtime. An
  inconsistent pair converges to the right answer slowly, or to a wrong one confidently. Test it
  with `torch.autograd.gradcheck` against `flow`.

## Implicit, coupled and dynamic constitutive laws

The contract above is the potential-controlled form, `q = g(dp)`: one flow per edge, a function
of that edge's own potential difference. Every built-in airflow, water and sewer law has it, and
a layer whose laws all have it is solved by nodal analysis. Five class-level declarations on
`ConstitutiveLaw` describe the other cases; a `KirchhoffLayer` with any of them is solved by
[Modified Nodal Analysis](layers.md#modified-nodal-analysis-implicit-coupled-and-dynamic-laws).

| Declaration | Default | Meaning |
|---|---|---|
| `law_form` | `"potential"` | `"potential"`: `flow(dp, drivers) -> q`. `"implicit"`: the law is a relation `residual(dp, q, drivers, ctx) = 0` and the flow of each of its edges is an unknown of the layer. |
| `ports` | `1` | The edges of the kind form consecutive groups of `ports` edges, in the order they were added; the law may couple the edges within a group (a transformer's primary and secondary), never across groups. |
| `dynamic` | `False` | The law stores (a capacitance, an inductance): it reads its step-start values from the context and implements every mode below. |
| `contextual` | `False` | `flow` takes the `LawContext` as a third argument (a time- or frequency-dependent potential-controlled law, such as a current source with a waveform). Implicit laws are always contextual. |
| `phasor_capable` | `False` | The law accepts complex `dp` and `q` (a linear law with real or complex coefficients, or a law written for phasors). Only such laws are allowed in a phasor layer. |

**The two forms.** A potential-controlled law is the nodal half of circuit theory: its flow is
eliminated in favour of the potentials. An implicit law covers what cannot be written that way:
a flow-controlled law `dp = h(q)`, an ideal potential source `dp = V` (no flow is a function of
`dp`), an ideal constraint inside a multi-port law, and a law whose flow and potential are
related through stored state. The base class `ImplicitLaw` sets `law_form = "implicit"` and
`contextual = True`; a subclass implements `residual`, which returns one value per edge, with
the shape of `dp`. An implicit law has no `flow`: its flow is read from the layer's solution.

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
print(layer.mna, layer.implicit_edges.tolist())
print(f"phi_mid = {float(phi[1]):.6f}, q = {[round(v, 9) for v in q.tolist()]}")
```

```text
True [0]
phi_mid = 2.000000, q = [2.0, 2.0]
```

The flow satisfies `2 q^2 + q = 10`, so `q = 2` and `phi_mid = 2`. `implicit_edges` lists the
edges whose flow is an unknown, in layer edge order.

**The law context.** A contextual law receives a `LawContext` (`noodl.elements.base`), a frozen
dataclass the layer builds for every solve:

| Field | Meaning |
|---|---|
| `mode` | `"dc"` (the steady operating point), `"initial"` (the consistent initial point), `"transient"` (one implicit step) or `"phasor"` (the sinusoidal steady state) |
| `dt` | The step length (transient only, else `None`) |
| `t` | The evaluation time: the END of a transient step, the given time otherwise; `None` when the caller tracks no clock |
| `integration` | `"trapezoidal"` (default) or `"backward_euler"`; read in the transient mode |
| `dp_prev`, `q_prev` | This kind's own potential differences and flows at the step start (transient only) |
| `omega` | The angular frequency `2 pi f`, one per instance (phasor only) |

A dynamic law implements each mode as a companion form. The capacitance of the electrical
application, `q = C d(dp)/dt`:

```py
def residual(self, dp, q, drivers, ctx):
    if ctx.mode == "dc":
        return q                                    # no current at rest
    if ctx.mode == "initial":
        return dp - self.v0                         # hold the initial potential difference
    if ctx.mode == "phasor":
        return q - 1j * ctx.omega * self.C * dp
    if ctx.integration == "backward_euler":
        return q - self.C * (dp - ctx.dp_prev) / ctx.dt
    return q + ctx.q_prev - 2.0 * self.C / ctx.dt * (dp - ctx.dp_prev)   # trapezoidal
```

A dynamic law is implicit in every mode, so the layer's unknowns are the same whatever the mode.
A multi-port law (`ports = 2`) receives the edges of all its groups together and pairs them
itself: `dp.unflatten(-1, (-1, 2))` gives each group's `(port 1, port 2)`. The Jacobian is
assembled within groups only, so a law that couples edges across groups gets a wrong Jacobian
and wrong gradients, not merely a slower Newton convergence.

## Drives

A drive contributes an additive term to the potential difference, before the law evaluates:

$$
q = g\!\left(A^{\top}\phi + \text{drive}\right)
$$

`noodl.drives` provides:

| Drive | What it adds |
|---|---|
| `ConstantDrive(value)` | A fixed potential difference per edge. |
| `Stack(...)` | Buoyancy: the hydrostatic pressure difference from a density difference over the height between an edge's endpoints. `Stack.from_network(net, kind)` builds it from the nodes' own `z_ref` attributes. |
| `DuctStack(kind, ...)` | The air column of a duct segment between its two junctions, at the density of their mean temperature. `DuctStack.from_network(net, kind)` builds it from the junctions' `z_ref`. |
| `Wind(...)` | Wind pressure on a façade, from wind speed, direction and a pressure coefficient. `Wind.from_network(net, kind, ambient=...)`. |
| `WindProfile(...)` | A vertical wind profile, identified by CONTAM's profile number. |

Drives are reused across applications far more than constitutive laws are. The sewer application's
headspace air layer drives its Newton solve with the *same* `Stack` the building application
uses for room buoyancy, plus a sewer-specific `Drag` term — the physics is different, the
structure is not.

### The rule a drive must obey

**A drive is a function of the drivers alone.** It may read anything from the `drivers` mapping
passed to `solve()`, and nothing else — not `phi`, not another layer's state, not an instance
attribute holding a differentiable tensor.

This is structural. Keeping a drive independent of $\phi$ is what keeps the Newton Jacobian
exactly $A_I \operatorname{diag}(g') A_I^\top$, with no extra coupling term. A drive that read
$\phi$ would silently invalidate the Jacobian every solver in the package assembles.

The sewer application's `Drag` is the worked consequence. Physically, the drag the moving water
surface exerts on the headspace air depends on the *relative* velocity $(U_s - U_{\text{air}})$,
and $U_{\text{air}}$ is exactly the quantity the air layer is solving for. Modelling it that way
would break the rule, so `Drag` uses the absolute surface velocity instead. Its docstring says so
and records the relative-velocity form as out of scope rather than quietly approximating it —
"structurally not differentiable by this framework's design, not an oversight."

If you genuinely need a term that depends on solved state, it belongs in a **closure**, which
runs between layer solves and writes driver values. See [Layers and models](layers.md#closures).
