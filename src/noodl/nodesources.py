"""NodeSource: a nodal withdrawal that depends on its OWN node's potential.

A `Drive` never sees `phi` (see `drives.py`): a drive is an additive term on a BRANCH's
potential difference, and letting it read the state would put a term into the residual that
the Jacobian `A_I diag(g') A_I^T` does not account for. A NodeSource is the other shape --
a flow leaving the network AT a node, as a function of that node's own potential -- and it
is exactly as safe, because its contribution to the Jacobian is a DIAGONAL entry at that
node:

    r_I(phi) = A_I g(A^T phi + drive) - s_I + w(phi_I)
    J(phi)   = A_I diag(g') A_I^T + diag(w')

EPANET's own pressure-driven demand is formulated this way internally (EPANET 2.2 Manual
section 13.1, p.110: "a virtual pipe from the junction to a fictitious reservoir"), which is
what this class exists for.

Sign convention: `flow` returns the WITHDRAWAL, positive OUT of the network, in the same
units as any other branch flow. A source (injection) is a negative withdrawal.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

Tensor = torch.Tensor


class NodeSource(torch.nn.Module):
    """Base class: a potential-dependent withdrawal at a fixed set of nodes.

    Subclasses set `self.nodes` (a 1-D `long` tensor of positions in the network's FULL node
    index space, registered as a buffer by `__init__`) and implement
    `flow(phi_nodes, drivers) -> Tensor` with trailing shape `(len(nodes),)`.

    `dflow` defaults to autograd on `flow`, exactly as `Element.dflow` does; override it
    with the analytic derivative where one is available and cheaper.
    """

    def __init__(self, nodes) -> None:
        super().__init__()
        idx = torch.as_tensor(nodes, dtype=torch.long)
        if idx.dim() != 1 or idx.numel() == 0:
            raise ValueError(
                f"{type(self).__name__}: nodes must be a non-empty 1-D index tensor, got "
                f"shape {tuple(idx.shape)}"
            )
        self.register_buffer("nodes", idx)

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        raise NotImplementedError

    def forward(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        # `torch.func.functional_call` always invokes `forward`, so the differentiable path
        # in `PotentialFlowLayer.solve` reaches `flow` through here (the same arrangement
        # `Element` uses).
        return self.flow(phi_nodes, drivers)

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """d(withdrawal)/d(phi) at these nodes; autograd by default."""
        leaf = phi_nodes.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            value = self.flow(leaf, drivers)
            if not value.requires_grad:
                # A withdrawal that does not depend on phi is a plain nodal source; it
                # belongs in `sources`, where it costs no Jacobian entry at all. Saying so
                # is better than returning a silently exact-zero diagonal.
                raise RuntimeError(
                    f"{type(self).__name__}: flow() does not depend on phi_nodes, so it is "
                    f"not a potential-dependent node source; pass a constant withdrawal "
                    f"through `sources` instead"
                )
            (grad,) = torch.autograd.grad(value.sum(), leaf)
        return grad
