"""NodeStorage: an amount stored at nodes, integrated over a step by the layer.

A node stores an amount; the layer integrates it. A `NodeStorage` says how much is stored at
each of its nodes as a function of that node's own potential (`amount`): the charge of a
capacitor to ground `C phi`, the water in a cylindrical tank `A H`, the air in a zone. Over a
time step of length `dt` the `KirchhoffLayer` turns the change of that amount into a
withdrawal, positive INTO storage, with the step rate of the layer's scheme
(`noodl.elements.step_rate`):

    backward Euler:  w = (S(phi) - S(phi_start)) / dt
    trapezoidal:     w = 2 (S(phi) - S(phi_start)) / dt - w_start

`phi_start` is the potential at the step start and `w_start` the storage flow at the step
start (the state key `"<layer>.storage_flow"`). Like a `NodeSource`, a storage adds only a
diagonal entry to the Jacobian, `step_rate_factor * capacity` with `capacity = dS/dphi`, so a
network of conductances with storage keeps its symmetric positive definite system. Without a
step (a steady solve) storage is absent: nothing flows into it at rest.

Which method to write:

- `amount` alone, for an amount that depends on the node's potential only. The default
  `change` is `amount(phi) - amount(phi_start)`, both at the CURRENT drivers (the lagged
  coefficient form SWMM and EPANET use).
- `change`, when the amount at the step start is not `amount(phi_start)` at the current
  drivers: an amount that depends on a driver that changes over the step (a zone's air mass
  depends on its temperature), or a difference that must be formed without round-off.
- `rate` and `rate_slope`, for an application whose withdrawal has rules beyond storage.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.elements.base import ConstitutiveLaw, LawContext, step_rate, step_rate_factor

Tensor = torch.Tensor


class NodeStorage(torch.nn.Module):
    """An amount stored at nodes as a function of each node's own potential; the layer
    turns its change over a step into a withdrawal (positive into storage).

    `nodes` is a 1-D long tensor of positions in the network's FULL node index space
    (registered as a buffer); repeats are allowed, and several storages may share a node:
    their withdrawals add up. One `(node, storage)` pair is an ENTRY; the layer reports the
    flow into every entry (`KirchhoffLayer.storage_flows`) in entry order: storages in the
    layer's list order, then each one's nodes.

    Subclasses implement `amount(phi_nodes, drivers)`, with trailing shape `(len(nodes),)`.
    The other methods have working defaults (see the module docstring for which to
    override): `change` the difference of `amount`, `capacity` its derivative by autograd,
    `rate` the step rate of the change, `rate_slope` its derivative, and `initial` `None`
    (no potential held at the consistent initial point).

    Learnable coefficients must be registered `nn.Parameter`s, as on a `NodeSource`: the
    differentiable solve threads exactly those.
    """

    #: True when `amount` is affine in the node potentials once the drivers are fixed: the
    #: layer is then linear (`KirchhoffLayer.linear`) if its laws are.
    linear: bool = False

    def __init__(self, nodes) -> None:
        super().__init__()
        idx = torch.as_tensor(nodes, dtype=torch.long)
        if idx.dim() != 1 or idx.numel() == 0:
            raise ValueError(
                f"{type(self).__name__}: nodes must be a non-empty 1-D index tensor, got "
                f"shape {tuple(idx.shape)}"
            )
        self.register_buffer("nodes", idx)

    def amount(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """The amount stored at each node, `S(phi)`."""
        raise NotImplementedError(f"{type(self).__name__} does not define amount()")

    def change(self, phi_nodes: Tensor, phi_start: Tensor,
               drivers: Mapping | None = None) -> Tensor:
        """The change of the stored amount over the step: by default
        `amount(phi) - amount(phi_start)`, both at the current drivers."""
        return self.amount(phi_nodes, drivers) - self.amount(phi_start, drivers)

    def capacity(self, phi_nodes: Tensor, phi_start: Tensor,
                 drivers: Mapping | None = None) -> Tensor:
        """d change / d phi at each node (autograd on `change` by default)."""
        leaf = phi_nodes.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            value = self.change(leaf, phi_start, drivers)
            if not value.requires_grad:
                raise RuntimeError(
                    f"{type(self).__name__}: change() does not depend on phi_nodes, so it "
                    f"stores nothing that the potential moves; check amount() or change() "
                    f"for a stray detach"
                )
            (grad,) = torch.autograd.grad(value.sum(), leaf)
        return grad

    def initial(self, drivers: Mapping | None = None) -> Tensor | None:
        """The potential each node is held at in `mode="initial"`, or `None` (the default)
        when this storage holds none."""
        return None

    def rate(self, phi_nodes: Tensor, phi_start: Tensor, previous: Tensor | None,
             ctx: LawContext, drivers: Mapping | None = None) -> Tensor:
        """The withdrawal into storage over the step that `ctx` describes:
        `step_rate(change(...), ctx, previous)`, with `previous` the flow into these entries
        at the step start (needed by the trapezoidal scheme only)."""
        return step_rate(self.change(phi_nodes, phi_start, drivers), ctx, previous)

    def rate_slope(self, phi_nodes: Tensor, phi_start: Tensor, previous: Tensor | None,
                   ctx: LawContext, drivers: Mapping | None = None) -> Tensor:
        """d rate / d phi at each node: `step_rate_factor(ctx) * capacity(...)`."""
        return step_rate_factor(ctx) * self.capacity(phi_nodes, phi_start, drivers)

    def forward(self, phi_nodes: Tensor, phi_start: Tensor, previous: Tensor | None,
                ctx: LawContext, drivers: Mapping | None = None) -> Tensor:
        # `torch.func.functional_call` always invokes `forward`, so the differentiable
        # solve reaches `rate` through here (as `NodeSource.forward` reaches `flow`).
        return self.rate(phi_nodes, phi_start, previous, ctx, drivers)


class LinearStorage(NodeStorage):
    """S = C phi with a constant capacity C per entry (a capacitor to ground, a
    cylindrical tank of area A).

    `capacity` is a scalar or one value per node (trailing shape `(len(nodes),)`; a leading
    batch shape is allowed). With `learnable=True` it is a trainable `nn.Parameter`.
    `initial` (default `None`) is the potential each node holds in `mode="initial"`, a
    scalar or one value per node.
    """

    linear = True

    def __init__(self, nodes, capacity, *, initial=None, learnable: bool = False) -> None:
        super().__init__(nodes)
        self.C = ConstitutiveLaw._param(capacity, learnable)
        if initial is None:
            self._initial = None
        else:
            self.register_buffer(
                "_initial", torch.as_tensor(initial, dtype=self.C.dtype).clone())

    def amount(self, phi_nodes, drivers=None):
        return self.C * phi_nodes

    def change(self, phi_nodes, phi_start, drivers=None):
        # The difference is formed before the multiply: C (phi - phi_start), never
        # C phi - C phi_start, which loses digits when the step changes phi little.
        return self.C * (phi_nodes - phi_start)

    def capacity(self, phi_nodes, phi_start, drivers=None):
        return self.C * torch.ones_like(phi_nodes)

    def initial(self, drivers=None):
        return self._initial

    def extra_repr(self) -> str:
        return f"nodes={self.nodes.tolist()}"
