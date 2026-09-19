# Differentiability

Every quantity in noodl is a PyTorch tensor and every solve is differentiable. This is the
difference between a simulator and a tool you can calibrate, invert and optimise with.

## What you get

For a converged solve, gradients flow back to:

- `phi_boundary` and `x_boundary` — the prescribed boundary values,
- `sources` — the nodal source terms,
- every value in the `drivers` mapping,
- every element parameter registered with `learnable=True`,
- and, across a [coupled model](../applications/coupling.md), through the join into the *other*
  model's parameters.

```python
phi, q = layer.solve(phi_boundary, drivers, sources=sources)   # differentiable=True default
loss = (phi[room] - measured) ** 2
loss.backward()
```

## The adjoint, not an unrolled tape

![Unrolling the Newton iteration stores every step; the implicit adjoint needs one linear solve at the converged solution](../assets/adjoint.svg)

The naive way to differentiate a Newton solve is to record every iteration and back-propagate
through all of them. That costs memory proportional to the iteration count and is numerically
noisy — the early iterates are far from the solution and contribute derivative noise.

noodl does not do that. It applies the implicit-function theorem at the converged solution.
For a residual $r(x, \theta) = 0$,

$$
\frac{\partial x}{\partial \theta} = -\left(\frac{\partial r}{\partial x}\right)^{-1}
\frac{\partial r}{\partial \theta}
$$

so the backward pass is **one transposed linear solve**, whatever the forward took. The Jacobian
is the one Newton already assembled, and the transposed solve reuses the same operator through
`TransposeOperator`. For a symmetric operator — which a potential-flow layer's is — `rmatvec` is
the forward action, so nothing extra is built.

The practical consequence: a model that takes 12 Newton iterations to converge costs the same to
differentiate as one that takes 3.

**The exception is iteration at the `Model` level.** `coupling="iterate"` is differentiated by
*unrolling* its passes, so memory does grow with pass count there. The convergence decision
itself is made on detached copies and never enters the graph. The same applies to
`CoupledModel`'s two-way fixed point; an implicit treatment of that outer loop is a recorded
follow-up.

## The contract

For `differentiable=True` to be *safe*, two rules must hold. Both are checked before any solve
is attempted, and a violation raises naming the offending element or drive and the attribute —
rather than silently returning a wrong or absent gradient.

**1. An element's differentiable state must be a registered parameter.**

```python
leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65), learnable=True)   # correct
```

A bare tensor held with `requires_grad=True` outside `named_parameters()` is a supported
construction, and its graph is reachable through the element's own component calls — but it is
invisible to the differentiable solve. Since `differentiable=False` returns detached tensors,
*neither* path would carry a gradient to it.

**2. A drive must read every differentiable quantity from the `drivers` mapping**, never hold
one as an instance attribute. See [Elements and drives](elements.md#the-rule-a-drive-must-obey)
for why this is structural rather than stylistic.

## What this is for

### Parameter estimation

The gradient of a mismatch against measurements, with respect to a physical coefficient, is what
gradient-based calibration needs. The coupling application's first inverse example calibrates a
building's leakage coefficient through a coupled street-building join, with the gradient running
through the coupled solve and back into the street model: relative error $1.29\times10^{-4}$
against the true value, final loss $3.46\times10^{-8}$.

### Source attribution

Which emission source is responsible for the concentration measured here? That is
$\partial c / \partial s_j$ — one adjoint pass, not one forward run per source. The coupling
application's second inverse example matches central differences to relative $10^{-4}$
(measured $3.6\times10^{-7}$ and $5.3\times10^{-7}$), and correctly returns a *structural* zero
for the source that cannot reach the receptor.

### Latent state recovery

`project_measured` and the cycle-space tools recover unmeasured flows from a few measured ones.
The third inverse example recovers all four branch flows of a network from one measured path,
exactly (rtol $10^{-10}$).

### Design optimisation

Anything the forward model computes can be an objective, and any element parameter can be a
design variable.

## Where gradients are zero, and why

Not every derivative that exists mathematically is non-zero in the model, and noodl documents
the cases rather than leaving you to discover them.

- **At a hard clip.** `CapacitatedTransferLayer` in `mode="hard"` carries *exactly zero*
  gradient across a capacity or headroom crossing — the branch choice is discrete. That is the
  entire reason `mode="smooth"` and `mode="projection"` exist. See
  [Capacitated transfer](../applications/capacitated.md).
- **At the sharing site, in smooth mode.** Smooth mode's proportional-share formula depends only
  on preference weights and total headroom, never on any individual competitor's request, so the
  cross-gradient $\partial f_i / \partial r_j$ is provably zero. `mode="projection"` routes that
  site through a real coupled QP to give a genuine cross-gradient. This was found as a live bug,
  not anticipated.
- **At zero flow in the sewer.** `normal_depth` returns exactly $h = 0$ with gradient $0$ at
  $q = 0$, a documented modelling choice, because the true $dh/dq$ diverges there.
- **Structurally, for `f_i`.** The sewer's interfacial drag coefficient is a `Drive` attribute
  and therefore not differentiable by this framework's design — stated as such, "not an
  oversight."
- **For a tank's area.** In the water application, tank area is not reached by a single
  `water_steady` call — only `bottom + level` enters the steady solve. Gradients with respect to
  it flow through `TankLevels.advance`, not through the steady state.

## Verification

Every application checks its gradients against finite differences, and the numbers are recorded:

| Application | Check | Tolerance | Measured |
|---|---|---|---|
| Water (D7) | demand, roughness, pump $h_0$, tank area | $10^{-6}\times$ scale | 3.2e-6 / 2.5e-10 / 1.0e-6 / 2.8e-9 |
| Sewer (C3) | inflow, `T_head`, leak area, `f_air`, vs Richardson-extrapolated central differences | rel $10^{-6}$ | holds |
| Capacitated (W6) | `torch.autograd.gradcheck` through smooth and projection modes | analytic | holds |
| Coupling | gradient across the join vs central differences | rel $10^{-5}$ | holds |

If you write your own element, do the same: `torch.autograd.gradcheck` in float64 against a
finite-difference reference is cheap insurance against a `dflow` that disagrees with its `flow`.

## Performance notes

- **Use float64.** Finite-difference comparisons and tight Newton tolerances both need it, and
  `gradcheck` is effectively meaningless in float32.
- **The sparse-direct backend also speeds up the backward pass**, measured 3.25x at ensemble 1
  and 3.66x at ensemble 100. Install the [`sparse` extra](../installation.md#optional-extras).
- **Batch instead of looping.** A gradient over a 100-instance ensemble is one backward pass, not
  100.
