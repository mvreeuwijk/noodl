"""Extension-level integration test for docs/development/extending.md.

This module is the worked example the guide walks through: a new branch law defined
entirely outside `src/` (import nothing from `noodl.apps`, touch nothing under `src/`) and a
small two-zone "application" built from it. It exists to prove the extension recipe works
through noodl's own public surface -- `noodl.elements.base.Element`, `noodl.topology`,
`noodl.layers.potential`, `noodl.model` -- not to add a feature.
"""

from __future__ import annotations

import torch

from noodl.elements.base import Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


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
