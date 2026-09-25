"""Pressure-driven demand as a potential-dependent nodal source (EPANET 2.2 PDA).

EPANET's Wagner-type demand function (Manual eq. 13.6, p.110, and the `[OPTIONS]`
`DEMAND MODEL PDA` description):

    d(p) = D                                       if p >= P_req
    d(p) = D ((p - P_min) / (P_req - P_min))^e      if P_min < p < P_req
    d(p) = 0                                       if p <= P_min

with `p = H - z` the pressure head, `P_min` the pressure below which nothing is delivered
(`MINIMUM PRESSURE`, default 0.0), `P_req` the pressure at which the full demand is
delivered (`REQUIRED PRESSURE`, default 0.0 -- so PDA is a NO-OP unless both are set), and
`e` the exponent (`PRESSURE EXPONENT`, default 0.5, "to mimic flow through an orifice").
The limits are global, one set for the whole network, exactly as EPANET applies them.

Measured against EPANET's own PDA on the committed two-loop fixture at `P_req = 60 m`:
heads agree to 3.521e-7 relative and delivered demands to 2.184e-7 (verification row D5,
tolerance 1e-5).
"""

from __future__ import annotations

import torch

from noodl.nodesources import NodeSource

Tensor = torch.Tensor
F64 = torch.float64


class PressureDrivenDemand(NodeSource):
    """The Wagner demand function as a `NodeSource` (withdrawal, positive OUT).

    ``flow`` uses a GUARDED kink, not a plain floor-and-power: the piecewise
    law is EXACTLY zero at and below `P_min` (not the `q_required * 1e-12**exponent` leak a
    naive `clamp(min=1e-12)` would deliver there), and it is EXACTLY `q_required` at and
    above `P_req`. `active = fraction > 0` selects the branch; the base fed to `** exponent`
    on the INACTIVE side is the constant `1.0`, never `fraction` itself (which can be zero or
    negative there) -- the same both-branches-safe pattern `HazenWilliams` and `PumpCurve`
    use, so `where`'s backward never multiplies an infinite derivative (a fractional power's
    derivative at a zero or negative base) by a zero mask and produces `nan`.
    """

    def __init__(
        self,
        nodes,
        q_required,
        elevation,
        *,
        p_min: float = 0.0,
        p_req: float = 0.0,
        exponent: float = 0.5,
        learnable: bool = False,
    ) -> None:
        super().__init__(nodes)
        if not p_req > p_min:
            raise ValueError(
                f"PressureDrivenDemand: REQUIRED PRESSURE ({p_req}) must exceed MINIMUM "
                f"PRESSURE ({p_min}); EPANET's PDA is a no-op when both are left at their "
                f"0.0 defaults, and a zero span has no demand curve at all"
            )
        value = torch.as_tensor(q_required, dtype=F64)
        self.q_required = (
            torch.nn.Parameter(value) if learnable else torch.nn.Parameter(value, False)
        )
        self.register_buffer("elevation", torch.as_tensor(elevation, dtype=F64))
        self.p_min = float(p_min)
        self.p_req = float(p_req)
        self.exponent = float(exponent)

    def flow(self, phi_nodes: Tensor, drivers=None) -> Tensor:
        pressure = phi_nodes - self.elevation
        fraction = (pressure - self.p_min) / (self.p_req - self.p_min)
        active = fraction > 0
        # On the inactive branch (fraction <= 0, i.e. p <= P_min) the base substituted into
        # `** exponent` is the constant 1.0, never `fraction` itself: `fraction` can be zero
        # or negative there, and a fractional power of a non-positive base is either a
        # singular derivative (at 0) or not real-valued at all (negative). Both branches of
        # `torch.where` are evaluated in the forward pass, so the substitution must be safe
        # regardless of which branch is finally selected.
        safe = torch.where(active, fraction.clamp(max=1.0), torch.ones_like(fraction))
        delivered = self.q_required * safe**self.exponent
        return torch.where(active, delivered, torch.zeros_like(delivered))
