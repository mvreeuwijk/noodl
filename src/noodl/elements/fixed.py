"""Fixed-flow branch element: q = q0 for any potential difference.

Used for prescribed exhaust/supply flows and, structurally, to make the nodal Jacobian
singular on any subnetwork whose only connection to the rest of the graph is through
fixed-flow branches (dflow = 0 contributes nothing to J = A diag(dflow) A^T);
``solvers.grounding.spd_certificate`` detects and reports exactly this case.
"""

from __future__ import annotations

import torch

from noodl.elements.base import ConstitutiveLaw

Tensor = torch.Tensor


class FixedFlow(ConstitutiveLaw):
    """q = q0 regardless of dp (`q0` in the flow unit); dflow = 0; linear_init = (q0, 0).

    Declares ``dp_independent = True`` (see ``ConstitutiveLaw``): the differentiable solve path
    trusts this declaration to take the exact-zero Jacobian column shortcut for this
    element's edges, rather than inferring dp-independence from whatever autograd graph
    ``flow`` happens to produce.
    """

    dp_independent = True
    phasor_capable = True

    def __init__(self, q0, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        self.q0 = self._param(q0, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self.q0 + torch.zeros_like(dp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.zeros_like(self.flow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        return self.q0, torch.zeros_like(self.q0)


class FixedVolumeFlow(ConstitutiveLaw):
    """F = Q rho_up whatever dp: a constant volume flow `volume_flow` Q (m3/s, one per edge)
    delivered as mass flow at the density of the node it draws from (`src` when Q >= 0,
    `tgt` otherwise), read from `drivers[rho_key]` (full node order). ContamX's
    constant-volume fan in a duct (measured)."""

    dp_independent = True

    def __init__(self, volume_flow, *, src, tgt, rho_key: str = "rho", kind: str = "fan",
                 learnable: bool = False) -> None:
        super().__init__(kind)
        src_t = torch.as_tensor(src).to(torch.long).reshape(-1)
        tgt_t = torch.as_tensor(tgt).to(torch.long).reshape(-1)
        if src_t.shape != tgt_t.shape:
            raise ValueError(f"FixedVolumeFlow (kind {kind!r}): src and tgt must give one "
                             f"node position per edge, got {tuple(src_t.shape)} and "
                             f"{tuple(tgt_t.shape)}")
        if not isinstance(volume_flow, torch.Tensor):     # never the float32 default
            volume_flow = torch.as_tensor(volume_flow, dtype=torch.float64)
        self.volume_flow = self._param(volume_flow, learnable)
        self.register_buffer("src", src_t)
        self.register_buffer("tgt", tgt_t)
        self.rho_key = str(rho_key)

    def _mass_flow(self, drivers) -> Tensor:
        if drivers is None or self.rho_key not in drivers:
            raise KeyError(f"FixedVolumeFlow (kind {self.kind!r}): driver {self.rho_key!r} "
                           f"not found; it needs the full-node air density")
        rho = torch.as_tensor(drivers[self.rho_key])
        Q = self.volume_flow.to(rho.dtype)
        return Q * torch.where(Q >= 0, rho[..., self.src], rho[..., self.tgt])

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self._mass_flow(drivers) + torch.zeros_like(dp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.zeros_like(self.flow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        q = self._mass_flow(drivers)
        return q, torch.zeros_like(q)


class PrescribedFlow(ConstitutiveLaw):
    """F = drivers[key] whatever dp: a flow (kg/s, one value per edge of the kind) that a
    closure or an input prescribes every step, e.g. an air handler's supply flows.
    `dtype` is the layer's: `linear_init`, which has no dp to take it from, returns it."""

    dp_independent = True

    def __init__(self, key: str, *, kind: str, dtype: torch.dtype = torch.float64) -> None:
        super().__init__(kind)
        self.key = str(key)
        self.register_buffer("_like", torch.zeros((), dtype=dtype))

    def _q(self, drivers) -> Tensor:
        if drivers is None or self.key not in drivers:
            raise KeyError(f"PrescribedFlow (kind {self.kind!r}): driver {self.key!r} not "
                           f"found; the closure that prescribes these flows must run first")
        return torch.as_tensor(drivers[self.key])

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self._q(drivers).to(dp.dtype) + torch.zeros_like(dp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.zeros_like(self.flow(dp, drivers))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        q = self._q(drivers).to(self._like.dtype)
        return q, torch.zeros_like(q)
