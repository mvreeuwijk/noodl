# A new constitutive law

How to add a branch law to an existing layer: a new relation between the potential difference
across an edge and the flow along it. The network, the conservation law and the solve stay as
they are; the law is where your physics goes. Nothing in `noodl` itself changes: the law lives
in your own package. For what a law is and where it sits in a model, see
[Constitutive laws and heads](../../concepts/elements.md).

## The contract

`noodl.elements.base.ConstitutiveLaw` subclasses `torch.nn.Module`. A law owns one edge `kind`.
A law in the default, potential-controlled form provides three things:

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
autograd round-trip per iteration would dominate the cost. A law with no convenient closed
form may leave `dflow` and `linear_init` to the base class, which computes them by autograd:
slower, and exact.

`jacobian` is the same idea for the solvers that need a law's slopes as local blocks (a law with
several ports, an implicit law, a phasor law). A law may give its slopes in closed form by
overriding `jacobian`; otherwise they are computed by automatic differentiation. Every law of
the electrical application gives them in closed form, so a circuit's Newton iterations evaluate
no derivatives by automatic differentiation.

`linear_init` matters more than it appears. A nonlinear network solved from a cold start can
take many Newton iterations or fail outright; solving the linear surrogate first puts the
iteration inside the basin of attraction. Every built-in law provides one.

A law may also report where it stops being smooth. `switching(dp, drivers)` returns values
that change sign where the flow switches between two analytic pieces, such as the edges of a
regularisation band, in units of the band's width (`noodl.elements.base.band_edges(x, w)`
builds the pair `(x - w)/w`, `(x + w)/w`). The default, `None`, means the law is smooth. A time
integrator can then put a step boundary on every switch, because a band crossed inside a step
spoils the error expansion that Richardson extrapolation relies on. Where a law is not smooth
(a damper's kink at zero, a laminar-to-turbulent transition), declare the transition and keep
the slope finite through it with `torch.where` over branches that are each finite, rather than
leaving autograd to discover the kink.

### Parameters and learnability

A law's differentiable quantities must be **registered parameters**, constructed with
`learnable=True`. Wrap each coefficient with `ConstitutiveLaw._param(value, learnable)`:

```py
leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65), learnable=True)
```

This is a hard requirement of the differentiable solve, not a style preference. The solve
threads a fixed set of tensors through `Function.apply`: each law's and withdrawal's registered
parameters, the driver values, `sources` and the boundary potentials (and, on a transient step,
the step-start state). Any other tensor that requires grad and reaches the residual is invisible
to the implicit adjoint, and the solve raises a `ValueError`, naming the law and the attribute
where it can, rather than returning a wrong or absent gradient. Two checks look for one:

- **Before solving, by attribute.** Every tensor a law or withdrawal holds (plain
  attributes, registered buffers, and tensors nested in lists, tuples, dicts, sets and plain
  objects, up to eight levels deep) is scanned, and one that requires grad without being
  threaded is refused by name.
- **At every Newton iterate of a differentiated solve, by evaluation.** The forward Newton
  evaluates the residual with every threaded tensor detached; if the result still requires
  grad, some other tensor reaches it (for example one captured by a closure, which no
  attribute scan can see), and the solve is refused.

So a bare tensor held with `requires_grad=True` outside `named_parameters()` is not a way to
make a coefficient learnable: neither path would carry a gradient to it. A head term (stack,
wind, drag) is not held by the layer: `EdgeHeads` writes the total as the driver
`"<layer>.edge_head"`, so a gradient reaches whatever the head reads.

## A worked law

Here is a new one, `Sigmoid`, with a single learnable coefficient (a steepness):

```py
from noodl.elements.base import ConstitutiveLaw
import torch


class Sigmoid(ConstitutiveLaw):
    """q = tanh(k * dp): a smooth, bounded, odd branch law with a learnable steepness `k`.

    Follows `Conductance`'s contract exactly: `__init__` wraps its one coefficient with
    `ConstitutiveLaw._param` (so `learnable=True` registers it as a real `nn.Parameter`, reachable
    by `torch.func.functional_call` the same way every built-in law is), and `flow`,
    `dflow` and `linear_init` are all overridden with closed forms rather than left to the
    base class's autograd default -- exact and cheap, exactly the reason `Conductance` does
    the same.
    """

    def __init__(self, k, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        self.k = self._param(k, learnable)

    def flow(self, dp: torch.Tensor, drivers=None) -> torch.Tensor:
        return torch.tanh(self.k * dp)

    def dflow(self, dp: torch.Tensor, drivers=None) -> torch.Tensor:
        q = torch.tanh(self.k * dp)
        return self.k * (1.0 - q * q)

    def linear_init(self, drivers=None) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(self.k), self.k
```

This follows `noodl.elements.conductance.Conductance`'s contract exactly:

- `__init__` wraps its one coefficient with `ConstitutiveLaw._param(value, learnable)`, never with a
  bare `nn.Parameter(...)` or a plain tensor held on `self`. `_param` is what makes
  `learnable=True` produce a real, module-registered parameter, visible to
  `.named_parameters()`, `state_dict()`, and the `torch.func.functional_call` substitution the
  differentiable solve relies on.
- `flow`, `dflow` and `linear_init` are all given closed forms, rather than left to the
  base class's autograd-based defaults. The default exists and works, and is what a law with no
  convenient closed form should use, but a cheap analytic derivative is preferable when there
  is one, as `Conductance` and `PowerLaw` do.

No other file changes. To use the law, hand it to a layer in place of a built-in one: the next
page, [A new application](application.md), builds the smallest network that does so.

### Check it

Two cheap checks a new law should always pass before anything more elaborate is built on it:
the nodal balance is zero at the solution, and `torch.autograd.gradcheck` agrees with the
analytic `dflow` through the whole implicit-adjoint solve, not just law by law. Use float64:
`gradcheck` is meaningless in float32. With `build_two_zone` from the next page:

```py
def test_interior_flow_balance_and_gradcheck_through_the_new_law():
    model, state, drivers = build_two_zone()

    new_state = model.steady(state, drivers)
    assert new_state["air.phi"].shape == (3,)

    residual = model.residuals(new_state, drivers)["air"]
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=1e-10, rtol=0.0
    )

    layer = model.kirchhoff["air"]
    law = layer._laws[0]
    phi_b = torch.zeros(1, dtype=F64)

    def f(k_):
        phi, _ = layer.solve(phi_b, {}, None, differentiable=True, atol=1e-12, rtol=1e-12)
        return phi[1:]

    assert torch.autograd.gradcheck(f, (law.k,), eps=1e-6, atol=1e-5)
```

`model.residuals` reports the interior nodal balance at a state, zero to solver tolerance at a
steady state.

## Rules that give wrong answers, not errors

- **Regularise at zero.** A law whose derivative diverges or vanishes at $\Delta p = 0$ will
  stall Newton at the first iteration, since a cold start usually sits there. Every built-in
  law is regularised there, but not all the same way: `PowerLaw` blends to a laminar
  (linear) form below a `dp_transition` threshold, while the MBL power-law family blends to a
  polynomial below a (wider) `dp_turbulent` threshold. `PowerLaw` with $n < 1$ has infinite
  slope at the origin without it.
- **Keep `dflow` consistent with `flow`.** Nothing checks them against each other at runtime. An
  inconsistent pair converges to the right answer slowly, or to a wrong one confidently. Test it
  with `torch.autograd.gradcheck` against `flow`.
- **Declare a `dflow` that is not the derivative.** The differentiable solve uses `dflow` for
  its Newton Jacobian and for the adjoint that computes gradients, so a `dflow` that is a
  deliberate Newton aid (a slope floor, a slope split between two edges) would make the
  gradients silently wrong. Such a law sets the class attribute `exact_slope = False`; the
  differentiable solve then differentiates its `flow` by autograd instead, while the
  non-differentiable solve keeps using `dflow`. `DoorPortStream` is the built-in example; every
  other built-in law's `dflow` is exact. The same attribute exists on `NodeWithdrawal`, and on
  `NodeStorage` for its `rate_slope`.
- **A law is its module call.** A layer calls each law as a module, so an overridden
  `forward` or a forward hook is part of the law. The module call of a law with a flow is
  `law(dp, drivers, ctx)`, its flow; that of an implicit law is `law(dp, q, drivers, ctx)`,
  its residual (`ImplicitLaw.forward` returns `residual`), so a hook on an implicit law acts
  on its residual: scaling it leaves the root where it is, adding a term moves it. Wherever
  such a law's module call differs from its own method, the layer evaluates it by that call
  and takes its slope from that call by autograd, not from its `dflow` or `jacobian`: in
  every solve, with or without `differentiable`, with or without flow unknowns; in the
  starting guess (the call's value and slope at `dp = 0` replace its `linear_init`) and the
  storage-ramp fallback; and in the layer's public `flows`, `residual`, `balance`,
  `storage_flows`, `jacobian`, `dflows`, `adjoint` and `linear_init`. A `dp_independent`
  law is still checked against its own `dflow`, but a hook may make its call depend on `dp`,
  so the slope of that call is taken by autograd too, zero where it does not depend on `dp`.
  The same holds for a withdrawal and a storage. A law without either (every library law)
  keeps its own `flow`, `dflow`, `jacobian` and `linear_init`.

## Implicit, coupled and dynamic laws

Write `flow`; if you cannot, write `residual`. A law that gives its flow, `q = g(dp)`, one flow
per edge as a function of that edge's own potential difference, writes `flow(dp, drivers)`. Every
built-in airflow, water and sewer law does, and a layer whose laws all do is solved by nodal
analysis. A law that cannot (a flow-controlled law, an ideal source, an inductance) subclasses
`noodl.elements.ImplicitLaw` and writes `residual(dp, q, drivers, ctx)` instead, the relation
between the potential difference and the flow, zero when they agree; the layer then adds its
flow as an unknown. Every law has `residual`: for a law with a flow it is `q - flow(dp)`, which
the solve never calls. To read the law context (time, frequency, step-start values), name the
third parameter of `flow` `ctx`. A `KirchhoffLayer` with an implicit law or a multi-port law adds
those edges' flows as unknowns ([Layers](../../concepts/layers.md)); a dynamic or `ctx`-reading
law that gives its flow adds none. Four class-level declarations on `ConstitutiveLaw` describe
the rest.

| Declaration | Default | Meaning |
|---|---|---|
| `ports` | `1` | The edges of the kind form consecutive groups of `ports` edges, in the order they were added; the law may couple the edges within a group (a transformer's primary and secondary), never across groups. |
| `dynamic` | `False` | The law reads its step-start values (`ctx.dp_prev`, `ctx.q_prev`): the layer then needs `history` on a step. A capacitance and an inductance implement every mode below. |
| `phasor_capable` | `False` | The law accepts complex `dp` and `q` (a linear law with real or complex coefficients, or a law written for phasors). Only such laws are allowed in a phasor layer. |
| `reads_heads` | `False` | The law's residual reads the absolute potentials at its edges' start and end nodes (`ctx.phi_from`, `ctx.phi_to`), not only the difference `dp`: a valve that holds the head downstream of it. Its `jacobian` then also returns `d r / d phi_from` and `d r / d phi_to`. Only a one-port `ImplicitLaw` may declare it; a layer refuses any other law that does (`TypeError`). |

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

**The start flow.** A layer's linear guess (`linear_init`, and the rungs of its
[fallback ladder](../../concepts/solvers.md#when-newton-fails-the-fallback-ladder)) takes each
implicit law's residual and Jacobian at zero interior potentials and at the law's start flow,
`start_flow(drivers, ctx)`: one value per edge (with any batch dimensions in front, which must
broadcast against the layer's batch), or `None`, the default, for zero flow. A start flow of
another width or batch is refused, naming the law. The loss above has no slope at zero flow, so a
guess taken there puts the whole 10 across it, at `q = 10`; with a start flow of 1 the guess is
`phi_mid = q = 2.4`, and with the root, 2, it is the solution. Give one where a law's tangent at
zero flow says little about its operating point: the water application's pipes start at 1 ft/s
and its pumps at their design flow, EPANET's own first iterate. It moves only the guess: a solve
still starts from zeros, or from its warm start, and the start flow carries no gradient.

```py
class StartedLoss(QuadraticLoss):
    def start_flow(self, drivers=None, ctx=None):
        return torch.ones(1, dtype=F64)      # one value per edge of the kind
```

**One scheme vocabulary for every layer.** A dynamic law reads `ctx.scheme`, one of the names in
`SCHEMES = ("exact", "backward_euler", "trapezoidal")` (`noodl.elements.base`): the same names and
the same keyword `scheme` a `TransportLayer` takes. A `KirchhoffLayer` gives its laws
`"backward_euler"` or `"trapezoidal"`; `"exact"` is for transport layers only.

**The law context.** A law whose `flow` has a `ctx` parameter (and every `ImplicitLaw`) receives
a `LawContext` (`noodl.elements.base`), a frozen dataclass the layer builds for every solve:

| Field | Meaning |
|---|---|
| `mode` | `"dc"` (the steady operating point), `"initial"` (the consistent initial point), `"transient"` (one implicit step) or `"phasor"` (the sinusoidal steady state) |
| `dt` | The step length (transient only, else `None`) |
| `t` | The evaluation time: the END of a transient step, the given time otherwise; `None` when the caller tracks no clock |
| `scheme` | `"trapezoidal"` (default) or `"backward_euler"`; read in the transient mode |
| `dp_prev`, `q_prev` | This kind's own potential differences and flows at the step start (transient only) |
| `omega` | The angular frequency `2 pi f`, one per instance (phasor only) |
| `phi_from`, `phi_to` | The potentials at the start and end nodes of each of this kind's edges, shaped like `dp` (only for a law that declares `reads_heads`, else `None`) |

A dynamic law implements a form for each mode. The capacitance of the electrical
application, `q = C d(dp)/dt`:

```py
def residual(self, dp, q, drivers, ctx):
    if ctx.mode == "dc":
        return q                                    # no current at rest
    if ctx.mode == "initial":
        return dp - self.v0                         # hold the initial potential difference
    if ctx.mode == "phasor":
        return q - 1j * ctx.omega * self.C * dp
    if ctx.scheme == "backward_euler":
        return q - self.C * (dp - ctx.dp_prev) / ctx.dt
    return q + ctx.q_prev - 2.0 * self.C / ctx.dt * (dp - ctx.dp_prev)   # trapezoidal
```

A dynamic law is implicit in every mode, so the layer's unknowns are the same whatever the mode.
A multi-port law (`ports = 2`) receives the edges of all its groups together and pairs them
itself: `dp.unflatten(-1, (-1, 2))` gives each group's `(port 1, port 2)`. The Jacobian is
assembled within groups only, so a law that couples edges across groups gets a wrong Jacobian
and wrong gradients, not merely a slower Newton convergence.

## A new head term

A head term adds to the potential difference before the law evaluates it,
$q = g(A^{\top}\phi + \text{head})$. `noodl.heads` provides `DriverHead`, `Stack`, `DuctStack`,
`Wind` and `WindProfile`; see [Layers](../../concepts/layers.md) for how a layer adds them up.

**A head term is a function of the drivers alone.** It may read anything from the `drivers`
mapping, and nothing else: not `phi`, not another layer's state. This is structural. Keeping a
head independent of $\phi$ is what keeps the Newton Jacobian exactly
$A_I \operatorname{diag}(g') A_I^\top$, with no extra coupling term. A head that read $\phi$
would silently invalidate the Jacobian every solver in the package assembles.

The sewer application's `Drag` is the worked consequence. Physically, the drag the moving water
surface exerts on the headspace air depends on the *relative* velocity $(U_s - U_{\text{air}})$,
and $U_{\text{air}}$ is exactly the quantity the air layer is solving for. Modelling it that way
would break the rule, so `Drag` uses the absolute surface velocity instead, and says so in its
docstring. If you need a term that depends on solved state, it belongs in a **closure**, which
runs between layer solves and writes driver values; see
[The closure contract](application.md#the-closure-contract). Because the head reaches the layer
as a driver, a gradient reaches whatever produced it.

## Withdrawals and node storage

Two objects act at a node rather than on an edge, each as a function of that node's own
potential, and each adds only a diagonal entry to the Jacobian:

- **`NodeStorage`** (`noodl.NodeStorage`, `noodl.LinearStorage`): an amount stored at the node,
  which the layer turns into a withdrawal over each step with its scheme's step rate. A
  capacitor to ground, a tank, a zone's air mass. Every layer takes it, with or without
  flow unknowns (not a phasor layer); at the consistent initial point a storage with an
  `initial()` potential holds its node there. Under `scheme="forward_euler"` a linear storage
  sits at a boundary node instead (an EPANET tank), held through the solve and moved by
  explicit Euler after it.
- **`NodeWithdrawal`** (`noodl.NodeWithdrawal`): a withdrawal that is **not** storage, `w(phi)` at
  the current instant: EPANET's pressure-driven demand, a leak. A storage written as a
  withdrawal with its own step-start drivers duplicates what the layer already does; give it to
  `KirchhoffLayer(storage=...)` instead.

A dynamic edge law (`Capacitance`, `Inductance`) and a node storage both turn a change into a
rate with `noodl.elements.step_rate`: `change / dt` under backward Euler, `2 change / dt -
previous` under the trapezoidal scheme, with `step_rate_factor` (`1/dt` or `2/dt`) its
derivative.
