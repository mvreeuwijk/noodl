"""Fixed-flow branch element: q = q0 for any potential difference.

Used for prescribed exhaust/supply flows and, structurally, to make the nodal Jacobian
singular on any subnetwork whose only connection to the rest of the graph is through
fixed-flow branches (dflow = 0 contributes nothing to J = A diag(dflow) A^T); Task 6's
`floating_nodes` detects and reports exactly this case.
"""

from __future__ import annotations

import torch

from tellegen.elements.base import Element

Tensor = torch.Tensor


class FixedFlow(Element):
    """q = q0 regardless of dp; dflow = 0; linear_init = (q0, 0).

    Declares ``dp_independent = True`` (see ``Element``): the differentiable solve path
    trusts this declaration to take the exact-zero Jacobian column shortcut for this
    element's edges, rather than inferring dp-independence from whatever autograd graph
    ``flow`` happens to produce.
    """

    dp_independent = True

    def __init__(self, q0, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        self.q0 = self._param(q0, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self.q0 + torch.zeros_like(dp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.zeros_like(self.flow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        return self.q0, torch.zeros_like(self.q0)
