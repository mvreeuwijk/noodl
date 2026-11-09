"""Constitutive laws of a balanced AC power-flow network, in per unit.

Every law lives on edges to the common `"ground"` boundary node of a phasor layer: the edge's
potential difference is the bus voltage phasor `v` (p.u., root-mean-square), and its flow is
the current `i` (p.u.) leaving the bus through the device into ground. A branch is a two-port
on the edge pair (from bus, to bus), exactly MATPOWER's branch model (`makeYbus`): a series
impedance `r + jx` with total line charging `b` split between the ends, behind an ideal
transformer of ratio `ratio` (0 meaning 1) and phase shift `shift` (radians) at the from end.
A constant-power load is `noodl.apps.electrical.ConstantPowerLoad` with `P`, `Q` in p.u.
"""

from __future__ import annotations

import torch

from noodl._broadcast import broadcast_shapes
from noodl.elements import ConstitutiveLaw, ImplicitLaw
from noodl.elements._slopes import group_shape, one_port_blocks, slope_blocks, stack_entries

Tensor = torch.Tensor


class PiBranch(ConstitutiveLaw):
    """MATPOWER's branch two-port in per unit,
    `[I_f; I_t] = [[Y_ff, Y_ft], [Y_tf, Y_tt]] [V_f; V_t]` with `Y_s = 1/(r + jx)`,
    `tau = ratio e^{j shift}`, `Y_tt = Y_s + jb/2`, `Y_ff = Y_tt/(tau conj(tau))`,
    `Y_ft = -Y_s/conj(tau)`, `Y_tf = -Y_s/tau`.

    The kind's edges are consecutive pairs (from bus -> ground, to bus -> ground), one pair
    per branch; `dp` holds the bus voltages `V_f, V_t` and the flow the currents `I_f, I_t`
    leaving each bus into the branch (p.u.). Every coefficient has one entry per branch (a
    scalar broadcasts): series resistance `r` and reactance `x` and total charging
    susceptance `b` in p.u., off-nominal tap `ratio` (0 means 1) and phase shift `shift` in
    radians. `learnable` applies to every coefficient.
    """

    ports = 2
    phasor_capable = True
    linear = True

    def __init__(self, r, x, b, ratio, shift, *, kind: str = "branch",
                 learnable: bool = False) -> None:
        super().__init__(kind)
        self.r = self._param(r, learnable)
        self.x = self._param(x, learnable)
        self.b = self._param(b, learnable)
        # the file format's 0 means unity: normalise before registering the parameter, so
        # equivalent encodings (0 and 1) have the same values and gradients
        if not isinstance(ratio, torch.Tensor):
            ratio = torch.as_tensor(ratio, dtype=torch.get_default_dtype())   # as `_param`
        self.ratio = self._param(torch.where(ratio == 0, torch.ones_like(ratio), ratio),
                                 learnable)
        self.shift = self._param(shift, learnable)

    def admittances(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`(Y_ff, Y_ft, Y_tf, Y_tt)` in p.u., one entry per branch."""
        ys = 1.0 / torch.complex(self.r, self.x)
        ratio = self.ratio
        tau = torch.polar(ratio, self.shift)
        ytt = ys + torch.complex(torch.zeros_like(self.b), self.b / 2)
        return ytt / (ratio * ratio), -ys / tau.conj(), -ys / tau, ytt

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        yff, yft, ytf, ytt = self.admittances()
        v = dp.unflatten(-1, (-1, 2))
        vf, vt = v[..., 0], v[..., 1]
        return torch.stack([yff * vf + yft * vt, ytf * vf + ytt * vt], dim=-1).flatten(-2)

    def jacobian(self, dp, q, drivers, ctx):
        yff, yft, ytf, ytt = self.admittances()
        return (slope_blocks(stack_entries([[yff, yft], [ytf, ytt]], group_shape(2, dp), dp),
                             dp),)


class Shunt(ConstitutiveLaw):
    """A constant-admittance shunt, `i = (g + jb) v` (p.u.), per edge bus -> ground: `g`
    consumes the real power `g |v|^2` and a positive `b` injects the reactive power
    `b |v|^2` (a capacitor). MATPOWER's bus `Gs`, `Bs` divided by the base power.
    `learnable` applies to `g` and `b`.
    """

    phasor_capable = True
    linear = True

    def __init__(self, g, b, *, kind: str = "shunt", learnable: bool = False) -> None:
        super().__init__(kind)
        self.g = self._param(g, learnable)
        self.b = self._param(b, learnable)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return torch.complex(self.g, self.b) * dp

    def dflow(self, dp, drivers=None):
        return torch.complex(self.g, self.b) * torch.ones_like(dp)

    def jacobian(self, dp, q, drivers, ctx):
        return (one_port_blocks(torch.complex(self.g, self.b), dp),)


class VoltageControlledGenerator(ImplicitLaw):
    """A generator injecting the real power `P` (p.u.) at the voltage magnitude `V` (p.u.),
    per edge bus -> ground: `residual(v, i) = (|v| - V) + j (Re(v conj(i)) + P)`. The edge
    current `i` leaves the bus into ground, so the injected complex power is `-v conj(i)`; its
    real part is `P` and its reactive part is whatever holds `|v| = V` (a PV bus, without
    reactive limits). Implicit: the current is an unknown of the layer. Power is
    `v conj(i)`, so the phasors are root-mean-square. `learnable` applies to `P` and `V`.
    """

    phasor_capable = True

    def __init__(self, P, V, *, kind: str = "generator", learnable: bool = False) -> None:
        super().__init__(kind)
        self.P = self._param(P, learnable)
        self.V = self._param(V, learnable)

    def residual(self, dp, q, drivers, ctx):
        magnitude = torch.sqrt(dp.real ** 2 + dp.imag ** 2)   # |v| in forward-mode-safe form
        injected = (dp * q.conj()).real + self.P
        return torch.complex(magnitude - self.V, injected)

    def jacobian(self, dp, q, drivers, ctx):
        """The real 2x2 blocks of the (non-holomorphic) residual: the magnitude's gradient
        `v / |v|` and the real power's `(Re i, Im i)` and `(Re v, Im v)`."""
        vx, vy, ix, iy = dp.real, dp.imag, q.real, q.imag
        magnitude = torch.sqrt(vx ** 2 + vy ** 2)
        d_dp = [[[[vx / magnitude, vy / magnitude], [ix, iy]]]]
        d_q = [[[[0.0, 0.0], [vx, vy]]]]
        shape = tuple(broadcast_shapes(dp.shape, q.shape))
        real = torch.zeros((), dtype=vx.dtype, device=vx.device)
        return stack_entries(d_dp, shape, real), stack_entries(d_q, shape, real)


__all__ = ["PiBranch", "Shunt", "VoltageControlledGenerator"]
