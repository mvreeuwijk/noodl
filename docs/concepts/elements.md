# Elements and drives

An **element** is the constitutive law on an edge kind: given the potential difference across
the edge, it returns the flow along it. A **drive** is a term added to that potential
difference before the element sees it — buoyancy, wind pressure, interfacial drag.

Together they are the replaceable part of the framework. The network, the conservation law and
the solve are fixed; the element is where your physics goes.

## The `Element` contract

`noodl.elements.base.Element` subclasses `torch.nn.Module`. An element owns one edge `kind` and
must provide three things:

```python
class Element(torch.nn.Module):
    kind: str

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """The branch law: flow given the potential difference."""

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Its derivative, d(flow)/d(dp). This is the Jacobian the Newton solve assembles."""

    def linear_init(self, drivers: Mapping | None = None) -> tuple[Tensor, Tensor]:
        """A linear surrogate (slope, offset) used to produce the Newton starting point."""
```

`dflow` is supplied analytically rather than taken from autograd because it is assembled into
the Jacobian $A_I \operatorname{diag}(g') A_I^\top$ at every Newton iteration, where an
autograd round-trip per iteration would dominate the cost.

`linear_init` matters more than it appears. A nonlinear network solved from a cold start can
take many Newton iterations or fail outright; solving the linear surrogate first puts the
iteration inside the basin of attraction. Every built-in element provides one.

### Parameters and learnability

An element's differentiable quantities must be **registered parameters**, constructed with
`learnable=True`:

```python
leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65), learnable=True)
```

This is a hard requirement of the differentiable solve, not a style preference. A bare tensor
held with `requires_grad=True` outside `named_parameters()` is invisible to the adjoint, and the
solve raises naming the offending element and attribute rather than silently returning an absent
gradient. See [Differentiability](differentiability.md#the-contract).

## Built-in elements

| Element | Law | Typical use |
|---|---|---|
| `PowerLaw(C, n)` | $q = C \,\mathrm{sign}(\Delta p)\,\lvert\Delta p\rvert^{n}$ | Cracks and leaks. The CONTAM power-law family; $n=0.5$ is a sharp orifice, $n \to 1$ a laminar crack. |
| `Orifice` | `PowerLaw` at $n = 1/2$ | Openings whose area and discharge coefficient are known. |
| `Quadratic(a, b)` | Inverts $\Delta p = a q + b \lvert q \rvert q$ | Quadratic drag; CONTAM's `qfr_*` family. |
| `Conductance(g)` | $q = g\,\Delta p$ | Linear conduction — thermal walls, passive exchange. |
| `FixedFlow(q0)` | $q = q_0$, independent of $\Delta p$ | Constant-flow fans. Declares `dp_independent = True`, which lets the solve take an exact-zero Jacobian column. |
| `FanCurve(coeffs, q_max)` | Cubic pressure-flow curve inverted for $q$ on $[0, q_{\max}]$ | Fans specified by a performance curve. |
| `Duct(...)` | Colebrook friction, laminar below a transition Reynolds number | Ducts and Darcy-Weisbach pipes. |
| `Damper(...)` | Separate power-law coefficient and exponent per flow direction | Backdraught dampers, one-way devices. |
| `UpstreamDensityPowerLaw(...)` | `PowerLaw` with $C$ scaled by $(\rho_{\text{up}}/\rho_{\text{ref}})^{m}$ | The upstream-density correction on CONTAM power-law elements. |

Applications add their own: the water application contributes `HazenWilliams`, `PumpCurve` and
`MinorLoss`; the sewer application contributes `Headspace`.

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
from noodl.elements.base import Element


class Squared(Element):
    """q = k * sign(dp) * dp**2."""

    def __init__(self, k, *, kind="pipe", learnable=False):
        super().__init__(kind)
        self.k = self._param("k", k, learnable)

    def flow(self, dp, drivers=None):
        return self.k * torch.sign(dp) * dp**2

    def dflow(self, dp, drivers=None):
        return 2.0 * self.k * torch.abs(dp)

    def linear_init(self, drivers=None):
        return self.k, torch.zeros_like(self.k)
```

Two rules worth stating explicitly, because both produce wrong answers rather than errors:

- **Regularise at zero.** A law whose derivative diverges or vanishes at $\Delta p = 0$ will
  stall Newton at the first iteration, since a cold start usually sits there. Every built-in
  element blends to a laminar (linear) form below a `dp_transition` threshold. `PowerLaw` with
  $n < 1$ has infinite slope at the origin without it.
- **Keep `dflow` consistent with `flow`.** Nothing checks them against each other at runtime. An
  inconsistent pair converges to the right answer slowly, or to a wrong one confidently. Test it
  with `torch.autograd.gradcheck` against `flow`.

## Drives

A drive contributes an additive term to the potential difference, before the element evaluates:

$$
q = g\!\left(A^{\top}\phi + \text{drive}\right)
$$

`noodl.drives` provides:

| Drive | What it adds |
|---|---|
| `ConstantDrive(value)` | A fixed potential difference per edge. |
| `Stack(...)` | Buoyancy: the hydrostatic pressure difference from a density difference over the height between an edge's endpoints. `Stack.from_network(net, kind)` builds it from the nodes' own `z_ref` attributes. |
| `Wind(...)` | Wind pressure on a façade, from wind speed, direction and a pressure coefficient. `Wind.from_network(net, kind, ambient=...)`. |
| `WindProfile(...)` | A vertical wind profile, identified by CONTAM's profile number. |

Drives are reused across applications far more than elements are. The sewer application's
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
