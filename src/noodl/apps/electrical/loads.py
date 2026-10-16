"""Loads of the Modelica Buildings Library (`Buildings.Electrical`): a series impedance and
a constant-power load (nonlinear or linearised, direct or alternating current).

Sign convention as for every electrical law: a load on the edge `p -> n` sees `dp = v_p - v_n`
and carries the current `q` entering pin `p`. Every parameter is per edge (a scalar
broadcasts). Phasor conventions: `Impedance` is linear, so it holds for any phasor scaling;
`ConstantPowerLoad` defines complex power as `dp conj(q)`, so it assumes root-mean-square
phasors (the Modelica Buildings Library's, `S = V conj(I)`): in a circuit with a power law,
give source amplitudes as RMS values (a `Sine` source's phasor is otherwise its peak).

Transcribed from MBL `Electrical/Interfaces/Impedance.mo` and
`Electrical/AC/OnePhase/Loads/Impedance.mo` (series `R` with `X = omega L` inductive or
`X = -1/(omega C)` capacitive), `Electrical/DC/Loads/Conductor.mo` (constant power and its
linearised banded form), `Electrical/AC/OnePhase/Loads/Inductive.mo` and `Resistive.mo`
(constant `P + jQ`, the homotopy `simplified` start forms and the linearised form).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from noodl.apps.electrical.laws import _driver
from noodl.elements import ConstitutiveLaw, ImplicitLaw

Tensor = torch.Tensor

#: Below this fraction of the nominal voltage a constant-power load returns MBL's homotopy
#: `simplified` form, which is finite at zero voltage.
_START_FRACTION = 1e-3


class Impedance(ImplicitLaw):
    """A series impedance `dp = (R + jX) q` with the reactance `X = omega L - 1/(omega C)`
    (`R` in ohm, `L` in H, `C` in F; `C = inf`, the default, is no capacitance) at the phasor
    layer's angular frequency `omega`. On a real layer (modes dc, initial, transient) it is
    the resistance `dp = R q`, and an edge with a reactance (`L != 0` or a finite `C`) is
    refused; at zero frequency an inductance is a short and a finite capacitance (an open
    circuit) is refused. `R = 0` is a short. Implicit (the flow is an unknown), so a zero resistance
    needs no special case. MBL `AC.OnePhase.Loads.Impedance`, `DC.Loads.Resistor` and the
    MBL short-circuit loads. `learnable` applies to `R`. Linear: valid for any phasor
    scaling.
    """

    phasor_capable = True
    linear = True

    def __init__(self, R, *, L=0.0, C=math.inf, kind: str = "impedance",
                 learnable: bool = False) -> None:
        super().__init__(kind)
        self.R = self._param(R, learnable)
        self.L = self._param(L, False)
        self.C = self._param(C, False)

    def residual(self, dp, q, drivers, ctx):
        if ctx is not None and ctx.mode == "phasor":
            capacitive = torch.isfinite(self.C)
            at_rest = ctx.omega == 0
            if bool((capacitive & at_rest).any()):
                raise ValueError(
                    f"impedance kind {self.kind!r}: a capacitance at zero frequency is an "
                    f"open circuit, which a series impedance cannot carry; give a nonzero "
                    f"frequency or C = inf")
            # 1/(omega C) only where C is finite and omega nonzero, from a safe denominator
            # so neither branch (nor its derivative) holds an inf or a NaN
            live = capacitive & ~at_rest
            Cs = torch.where(live, self.C, torch.ones_like(self.C))
            wC = torch.where(live, ctx.omega * Cs, torch.ones_like(Cs))
            x = ctx.omega * self.L - torch.where(live, 1.0 / wC, torch.zeros_like(wC))
            return dp - (self.R + 1j * x) * q
        reactive = (self.L != 0) | torch.isfinite(self.C)
        if bool(reactive.any()):
            raise ValueError(f"impedance kind {self.kind!r}: a reactance needs a phasor layer")
        return dp - self.R * q


class ConstantPowerLoad(ConstitutiveLaw):
    """A load absorbing the complex power `S = P + jQ` (W, var; positive = consumption, the
    power-flow convention), per edge: `q = conj(S / dp)`, on a real layer `q = P / dp`.
    `scale` names a driver whose per-edge values multiply `S` (a load profile).

    `linearized=True` gives MBL's linearised loads (`V_nominal` in V required): alternating
    current `q = conj(S) dp / V_nominal^2`; direct current (`direct_current=True`)
    `q = S (2/Vb - dp/Vb^2)` with `Vb = 0.8 V_nominal` where `|dp| <= (8/9) V_nominal`,
    `Vb = 1.2 V_nominal` where `|dp| >= (12/11) V_nominal`, else `Vb = V_nominal`.

    With `V_nominal` given, the nonlinear load is finite at and near zero voltage: where
    `|dp| <= 1e-3 V_nominal` it returns MBL's homotopy `simplified` form instead, direct
    current `S (2/V_nominal - dp/V_nominal^2)`, alternating current
    `conj(S) dp / V_nominal^2`, so a solve may start from zero potentials; at a solution
    above that voltage the actual law holds. Without `V_nominal` the load is singular at
    `dp = 0`: start the solve from a nonzero guess (`phi0`).

    Phasors: power is `S = dp conj(q)`, so the law assumes root-mean-square phasors (MBL's);
    in a circuit with a power law give source amplitudes as RMS values. A nonzero `Q` on a
    direct-current load is refused at construction, on a real layer when evaluated. MBL
    `DC.Loads.Conductor`,
    `AC.OnePhase.Loads.Inductive`, `Capacitive`, `Resistive` with `P_mbl = -P`,
    `Q_mbl = -Q` (MBL counts consumption negative). `learnable` applies to `P` and `Q`.
    """

    phasor_capable = True

    def __init__(self, P, Q=0.0, *, V_nominal=None, linearized: bool = False,
                 direct_current: bool = False, scale: str | None = None,
                 kind: str = "load", learnable: bool = False) -> None:
        super().__init__(kind)
        if direct_current and bool((torch.as_tensor(Q) != 0).any()):
            raise ValueError(f"constant-power load kind {kind!r}: reactive power Q on a "
                             f"direct-current load; give Q = 0")
        if linearized and V_nominal is None:
            raise ValueError(f"constant-power load kind {kind!r}: linearized=True needs "
                             f"V_nominal (the nominal voltage, V)")
        self.P = self._param(P, learnable)
        self.Q = self._param(Q, learnable)
        self.V_nominal = None if V_nominal is None else self._param(V_nominal, False)
        self.linearized = linearized
        self.direct_current = direct_current
        self.scale = scale

    def power(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """`S` per edge (complex for a complex `dp`, else the real `P`), times the scale."""
        if not dp.is_complex() and bool((self.Q != 0).any()):
            raise ValueError(f"constant-power load kind {self.kind!r}: reactive power Q on "
                             f"a real layer; give Q = 0")
        S = self.P + 1j * self.Q if dp.is_complex() else self.P
        if self.scale is not None:
            S = S * _driver(drivers, self.scale, self.kind)
        return S

    def flow(self, dp, drivers=None):
        S = self.power(dp, drivers)
        Vn = self.V_nominal
        if self.linearized:
            if not self.direct_current:
                return S.conj() * dp / Vn ** 2
            mag = dp.abs()
            Vb = torch.where(mag <= (8.0 / 9.0) * Vn, 0.8 * Vn,
                             torch.where(mag >= (12.0 / 11.0) * Vn, 1.2 * Vn, Vn))
            return S * (2.0 / Vb - dp / Vb ** 2)
        if Vn is None:
            return (S / dp).conj()
        small = dp.abs() <= _START_FRACTION * Vn
        safe = torch.where(small, Vn.to(dp.dtype), dp)
        actual = (S / safe).conj()
        if self.direct_current:
            simplified = S * (2.0 / Vn - dp / Vn ** 2)
        else:
            simplified = S.conj() * dp / Vn ** 2
        return torch.where(small, simplified, actual)


__all__ = ["ConstantPowerLoad", "Impedance"]
