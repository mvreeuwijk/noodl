# Extending noodl

This page walks through the two extension levels noodl supports below "a new numerical
block/backend": **a new branch law inside an existing layer**, and **a new application
composed of existing blocks**. The exact code below is `tests/test_extension_recipe.py`, an
extension-level integration test that imports nothing from `noodl.apps` and does not modify
anything under `src/` — proof that both levels work through noodl's ordinary public surface,
with no core change required.

## Level 1: a new branch law

Every branch law is a `torch.nn.Module` subclass of `noodl.elements.base.Element`. See
[Elements and drives](../concepts/elements.md#the-element-contract) for the full contract;
in short, an element owns one edge `kind` and provides `flow`, `dflow` and `linear_init`.

Here is a new one, `Sigmoid`, with a single learnable coefficient (a "steepness") — verbatim
from `tests/test_extension_recipe.py`:

```python
from noodl.elements.base import Element
import torch


class Sigmoid(Element):
    """q = tanh(k * dp): a smooth, bounded, odd branch law with a learnable steepness `k`.

    Follows `Conductance`'s contract exactly: `__init__` wraps its one coefficient with
    `Element._param` (so `learnable=True` registers it as a real `nn.Parameter`, reachable
    by `torch.func.functional_call` the same way every built-in element is), and `flow`,
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

- `__init__` wraps its one coefficient with `Element._param(value, learnable)`, never with a
  bare `nn.Parameter(...)` or a plain tensor held on `self`. `_param` is what makes
  `learnable=True` produce a real, module-registered parameter — visible to
  `.named_parameters()`, `state_dict()`, and the `torch.func.functional_call` substitution the
  differentiable solve relies on. A bare `requires_grad=True` tensor held outside that
  registration is invisible to it (see
  [Differentiability: the contract](../concepts/differentiability.md#the-contract)).
- `flow`, `dflow` and `linear_init` are all given closed forms, rather than left to the base
  class's autograd-based `dflow`/`linear_init` defaults. That default exists and works — it is
  what a law with no convenient closed form should use — but a cheap analytic derivative is
  preferable when there is one, exactly as `Conductance` and `PowerLaw` do.

No other file changes. The law lives entirely in the test module (or, in a real project, in
your own package) and needs no change to `noodl.elements`, `noodl.layers` or `noodl.model`.

## Level 2: a new application

A "new application" is nothing but a builder function returning `(model, state, drivers)` from
existing layers and elements, plus (if the application steps forward in time) an
`initial_state` function and any closures it needs. Here is the smallest one that exercises
`Sigmoid`: an ambient boundary and two zones, joined by three `airpath` edges, one
`PotentialFlowLayer`, one `Model`. Verbatim from `tests/test_extension_recipe.py`:

```python
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network
import torch

F64 = torch.float64


def build_two_zone(*, learnable: bool = True):
    """(model, state, drivers) for the smallest network the new law can be checked on.

    ambient (boundary) -> z1 -> z2 -> ambient, three `airpath` edges, one `Sigmoid` element
    covering all three (steepness `k`, shape (3,)). One `PotentialFlowLayer`, one `Model`.
    No `TransportLayer`, no closures: nothing here needs one, and the guide's point is the
    smallest network that exercises the new law, not a full application.
    """
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")

    k = torch.tensor([1.0, 0.8, 1.2], dtype=F64)
    element = Sigmoid(k, learnable=learnable)
    layer = PotentialFlowLayer(
        net, "air", [element], boundary=["ambient"], quantity="pressure", unit="Pa",
    )
    model = Model(net, {"air": layer})

    state = {}
    drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64)}
    return model, state, drivers
```

This is deliberately the minimum: one potential layer and no transport layer, no closures. A
real application adds more of the same kind of thing, in a fixed place:

- **`src/noodl/apps/<name>/network.py`** — the module a real application lives in. It defines
  the domain objects (see `Street`/`StreetNetwork` in
  `src/noodl/apps/street_aq/network.py` for a worked example: frozen dataclasses that validate
  their own inputs), a **builder** (`build_model` there, `build_two_zone` here) that
  turns them into `(Model, State, Drivers)` by constructing a `Network`, one `Element` per
  edge kind, one layer per physical quantity, any closures, and the `Model` itself, and an
  **`initial_state(model)`** function that returns a state dict of the right shape for every
  layer the model owns (zeros, for `initial_state` in `network.py`, keyed by each transport
  layer's own state key).
- **Closures with `integrates` declared.** A closure that only reads the current state and
  drivers to produce new driver values needs no declaration. One that instead **integrates**
  some state over the step's own interval — advances a counter, a controller, a stored level —
  must declare `integrates = True` (see `noodl.model.Closure`, `StepContext`) so
  `Model` calls it as `closure(state, drivers, ctx)` with the interval to advance by, and so
  `Model.steady()` (which has no interval) refuses it by name instead of silently either
  skipping it or integrating over an undefined `dt`.
- **Drivers.** Every boundary value and every source a layer's `solve`/`step` needs is a
  driver — `"<layer>.phi_boundary"`, `"<layer>.x_boundary"`, `"<layer>.sources"` — supplied by
  the builder as a template and overwritten by the caller (or by a closure) at each step. See
  the [Layers and models](../concepts/layers.md) page for the full key vocabulary.

**Numerical blocks are not extension points (yet).** The tested extension path is laws and
applications, as above — a new `Element` and a new builder composing existing layers. A new
solver or layer class is a framework change, not an extension: it needs the same design,
testing and review as any other change under `src/noodl/layers` or `src/noodl/solvers`. `ConstitutiveLayer` is the worked example of a
standalone numerical block that was added this way rather than adapted into an existing
extension point — see [Layers and models](../concepts/layers.md#constitutivelayer).

## Checking it

The test itself does two things with `build_two_zone`, both cheap sanity checks a new law and
a new network should always pass before anything more elaborate is built on them. Verbatim
from `tests/test_extension_recipe.py`:

```python
def test_interior_flow_balance_and_gradcheck_through_the_new_law():
    model, state, drivers = build_two_zone()

    new_state = model.steady(state, drivers)
    assert new_state["air.phi"].shape == (3,)

    residual = model.residuals(new_state, drivers)["air"]
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=1e-10, rtol=0.0
    )

    layer = model.potential["air"]
    element = layer._elements[0]
    phi_b = torch.zeros(1, dtype=F64)

    def f(k_):
        phi, _ = layer.solve(phi_b, {}, None, differentiable=True, atol=1e-12, rtol=1e-12)
        return phi[1:]

    assert torch.autograd.gradcheck(f, (element.k,), eps=1e-6, atol=1e-5)
```

`model.residuals` reports the interior nodal balance at a state — zero, to solver tolerance, at
a steady state — and `gradcheck` confirms `Sigmoid`'s analytic `dflow` agrees with its `flow`
through the whole implicit-adjoint solve, not just element-by-element.

## What a real application must ship

A network and a builder are not enough to trust an application's numbers. Every application in
this repository ships four kinds of test beyond ordinary unit coverage of its own code:

1. **An independent reference.** A case with a known answer this application did not itself
   produce — a closed form, a published value, `scipy`, or a legacy tool's own output —
   checked to a stated tolerance. Testing the code against itself is not this.
2. **Conservation.** The nodal or interior balance this application's `steady()` converges to
   is zero (to solver tolerance) at that fixed point, and stays a fixed balance under whatever
   coupling or sub-stepping the application uses.
3. **Directional derivatives.** `torch.autograd.gradcheck` (or, at application scale, a
   central-difference comparison) through the application's own solve, for every parameter and
   driver a calibration or optimisation loop would actually differentiate against — not just
   the new law in isolation.
4. **A scaling fixture.** A case parametrised over ensemble size (and, where relevant, network
   size), so a regression in batching or in solver selection is caught as a shape or cost
   change rather than discovered later as a silent slowdown.

`tests/test_extension_recipe.py` ships (1)-(3) in miniature for `Sigmoid`/`build_two_zone`; a
real application under `src/noodl/apps/<name>/` ships all four, sized to its own domain, the
way `tests/apps/street_aq/`, `tests/apps/sewer/` and `tests/apps/water/` do for theirs.
