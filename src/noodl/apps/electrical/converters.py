"""Converters of the Modelica Buildings Library (`Buildings.Electrical`): an AC/AC
transformer with a series primary impedance and an AC/DC power converter, both phasor-layer
two-port laws, with the smoothing functions their equations use.

Edges per device, in kind order: `[port 1, port 2]` (see each law). Transcribed from MBL
`Electrical/AC/OnePhase/Conversion/ACACTransformer.mo`, `ACDCConverter.mo`,
`Electrical/PhaseSystems/OnePhase.mo` (`systemVoltage = regRoot(v*v, 1e-5)`),
`Utilities/Math/Functions/spliceFunction.mo` and MSL `Fluid/Utilities.mo` (`regRoot`).
"""

from __future__ import annotations

import math

import torch

from noodl.apps.electrical.laws import _pairs, _ports
from noodl.elements import ImplicitLaw

Tensor = torch.Tensor

#: `spliceFunction`'s bound on the scaled argument (keeps `tan` away from its poles).
_SPLICE_LIMIT = 0.9999999999


def reg_root(x: Tensor, delta: float) -> Tensor:
    """`x / (x^2 + delta^2)^(1/4)`: `sqrt(|x|)` signed, smoothed around zero within
    `delta`. MSL `Modelica.Fluid.Utilities.regRoot`."""
    return x / (x * x + delta * delta) ** 0.25


def splice(pos: Tensor, neg: Tensor, x: Tensor, deltax: float) -> Tensor:
    """`neg` for `x <= -deltax`, `pos` for `x >= deltax` (to within `1e-10 deltax`), and
    between them the once continuously differentiable blend `pos y + (1 - y) neg` with
    `y = (tanh(tan(x/deltax pi/2)) + 1)/2`. MBL `Utilities.Math.Functions.spliceFunction`."""
    scaled = x / deltax
    inner = torch.clamp(scaled, -_SPLICE_LIMIT, _SPLICE_LIMIT)  # inactive inside the band
    y = (torch.tanh(torch.tan(inner * math.asin(1.0))) + 1.0) / 2.0
    blend = pos * y + (1.0 - y) * neg
    return torch.where(scaled <= -_SPLICE_LIMIT, neg,
                       torch.where(scaled >= _SPLICE_LIMIT, pos, blend))


def _phasor_only(law, ctx) -> None:
    if ctx is None or ctx.mode != "phasor":
        mode = None if ctx is None else ctx.mode
        raise ValueError(f"{type(law).__name__} kind {law.kind!r} needs a phasor layer "
                         f"(mode {mode!r})")


class ImpedanceTransformer(ImplicitLaw):
    """An ideal transformer behind a series primary impedance: `dp1 = Z q1 + n dp2`,
    `q2 = -n q1` (winding ratio `n`, primary to secondary voltage; complex primary impedance
    `Z` in ohm, per device). Edges `[primary node -> reference, secondary node -> reference]`
    per device. Phasor layers only. Linear, so valid for any phasor scaling. MBL
    `AC.OnePhase.Conversion.ACACTransformer` (`n = VHigh/VLow`,
    `Z = |Zp| (cos(atan(XoverR)) + j sin(atan(XoverR)))`, `|Zp| = VHigh^2 Zperc / VABase`).
    """

    ports = 2
    phasor_capable = True

    def __init__(self, n, impedance, *, kind: str = "impedance_transformer") -> None:
        super().__init__(kind)
        self.n = self._param(n, False)
        if not isinstance(impedance, Tensor):
            impedance = torch.as_tensor(impedance, dtype=torch.complex128)
        self.impedance = self._param(impedance, False)

    def residual(self, dp, q, drivers, ctx):
        _phasor_only(self, ctx)
        v1, v2 = _ports(dp)
        i1, i2 = _ports(q)
        return _pairs(v1 - self.impedance * i1 - self.n * v2, i2 + self.n * i1)


class PowerConverter(ImplicitLaw):
    """An AC/DC converter with a constant efficiency, in a phasor layer whose direct-current
    side is a complex node with zero imaginary part. Edges
    `[ac node -> reference, dc conductor 1 -> dc conductor 2]` per device, conductor 2 at the
    reference potential. Equations, with `S_ac = dp_ac conj(q_ac)` and
    `P_dc = Re(dp_dc conj(q_dc))` the powers into the two ports:
    `dp_dc = k |dp_ac|` (with MBL's smoothed modulus `reg_root(|dp_ac|^2, 1e-5)`; the
    imaginary part holds the DC side's at zero) and `Re S_ac + P_dc = P_loss`,
    `Im S_ac = 0` (unity power factor), with the loss
    `P_loss = (1 - eta) splice(P_dc, Re S_ac, Re q_dc, 0.1)`: a fraction of the power of
    whichever side feeds the converter. `k` (`voltage_ratio`) is the DC voltage over the AC
    root-mean-square voltage and the power balance takes root-mean-square phasors (MBL's,
    `S = V conj(I)`); `eta` (`efficiency`) is between 0 and 1. Phasor layers only. MBL
    `AC.OnePhase.Conversion.ACDCConverter` (`conversionFactor`, `eta`, `ground_DC = true`).
    """

    ports = 2
    phasor_capable = True

    def __init__(self, voltage_ratio, efficiency, *, kind: str = "power_converter") -> None:
        super().__init__(kind)
        self.voltage_ratio = self._param(voltage_ratio, False)
        self.efficiency = self._param(efficiency, False)

    def residual(self, dp, q, drivers, ctx):
        _phasor_only(self, ctx)
        v_ac, v_dc = _ports(dp)
        i_ac, i_dc = _ports(q)
        s_ac = v_ac * i_ac.conj()
        p_dc = (v_dc * i_dc.conj()).real
        loss = (1.0 - self.efficiency) * splice(p_dc, s_ac.real, i_dc.real, 0.1)
        modulus = reg_root(v_ac.real ** 2 + v_ac.imag ** 2, 1e-5)
        r1 = self.voltage_ratio * modulus - v_dc
        r2 = (s_ac.real + p_dc - loss) + 1j * s_ac.imag
        return _pairs(r1, r2)


__all__ = ["ImpedanceTransformer", "PowerConverter", "reg_root", "splice"]
