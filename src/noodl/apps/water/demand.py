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
heads agree to 3.521e-7 relative and delivered demands to 2.184e-7 (checked in
`tests/verification/test_water_parity.py::test_pressure_driven_demand` against a 1e-5
tolerance).
"""

from __future__ import annotations

import torch

from noodl.apps.water.elements import DEMAND_FACTOR_KEY
from noodl.withdrawals import NodeWithdrawal

Tensor = torch.Tensor
F64 = torch.float64


class PressureDrivenDemand(NodeWithdrawal):
    """The Wagner demand function as a `NodeWithdrawal` (withdrawal, positive OUT).

    ``flow`` uses a GUARDED kink, not a plain floor-and-power: the piecewise
    law is EXACTLY zero at and below `P_min` (not the `q_required * 1e-12**exponent` leak a
    naive `clamp(min=1e-12)` would deliver there), and it is EXACTLY `q_required` at and
    above `P_req`. `active = fraction > 0` selects the branch; the base fed to `** exponent`
    on the INACTIVE side is the constant `1.0`, never `fraction` itself (which can be zero or
    negative there) -- the same both-branches-safe pattern `HazenWilliams` and `PumpCurve`
    use, so `where`'s backward never multiplies an infinite derivative (a fractional power's
    derivative at a zero or negative base) by a zero mask and produces `nan`.

    The driver `"water.demand_factor"` (`DEMAND_FACTOR_KEY`, one multiplier per node of this
    withdrawal), when given, scales `q_required`: the demand pattern's multiplier, which
    `water_run` sets every step (EPANET's `demands` scales the required demand,
    `hydraul.c`).
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
        required = self.q_required
        factor = None if drivers is None else drivers.get(DEMAND_FACTOR_KEY)
        if factor is not None:
            required = required * factor
        delivered = required * safe**self.exponent
        return torch.where(active, delivered, torch.zeros_like(delivered))


class EmitterOutflow(NodeWithdrawal):
    """EPANET's emitters as a `NodeWithdrawal` (withdrawal, positive OUT): at each node the
    outflow is ``q = sign(p) C |p|^n`` with ``p = phi - elevation`` the pressure head (m), ``C``
    the discharge coefficient (m3/s per m^n) and ``n`` the emitter exponent.

    Transcribed from EPANET 2.2 (tag v2.2): `hydcoeffs.c`, `emittercoeffs` and
    `emitterheadloss` model the emitter as a fictitious pipe to a reservoir at the junction's
    elevation whose head loss is ``Ke q^(1/n)`` (``Ke = ucf / C^(1/n)``, `input1.c`,
    `convertunits`; ``Qexp = 1/n``, `input3.c`, `EMITTER EXPONENT`), with the loss
    ``hgrad q / Qexp`` carrying the SIGN of ``q``: the emitter is odd in the pressure, so at
    negative pressure it is an INFLOW of ``C |p|^n`` (EPANET does not clip it; `hydsolver.c`,
    `emitflowchange`, updates the same signed `EmitterFlow`). Inverting that law gives the
    flow above, which is what a potential-dependent withdrawal needs.

    EPANET linearises the fictitious pipe where its gradient falls below ``RQtol``; this
    law instead is linear below ``dp_transition`` in the pressure, tangent to the power
    law there, which keeps the slope at ``p = 0`` finite as `HazenWilliams` does.
    ``dp_transition`` is in the units of ``phi`` (the layer's potential); `build_model` gives
    it as 1e-9 m of head whatever the layer's unit (times ``rho g`` on a pressure layer,
    whose ``elevation`` and ``coefficient`` it also scales), so the transition is the same
    physical head difference on a Hazen-Williams, Chezy-Manning or Darcy-Weisbach network. Both
    branches of the ``torch.where`` are finite (the base fed to the power on the linear side
    is the constant ``dp_transition``), as `PressureDrivenDemand` does.
    """

    def __init__(
        self,
        nodes,
        coefficient,
        elevation,
        *,
        exponent: float = 0.5,
        dp_transition: float = 1e-9,
        learnable: bool = False,
    ) -> None:
        super().__init__(nodes)
        value = torch.as_tensor(coefficient, dtype=F64)
        self.coefficient = (
            torch.nn.Parameter(value) if learnable else torch.nn.Parameter(value, False)
        )
        self.register_buffer("elevation", torch.as_tensor(elevation, dtype=F64))
        self.exponent = float(exponent)
        self.dp_transition = float(dp_transition)

    def flow(self, phi_nodes: Tensor, drivers=None) -> Tensor:
        pressure = phi_nodes - self.elevation
        dpt = self.dp_transition
        linear = pressure.abs() < dpt
        safe = torch.where(linear, torch.full_like(pressure, dpt), pressure.abs())
        slope = self.coefficient * dpt ** (self.exponent - 1.0)
        return torch.where(
            linear, slope * pressure, torch.sign(pressure) * self.coefficient * safe**self.exponent
        )
