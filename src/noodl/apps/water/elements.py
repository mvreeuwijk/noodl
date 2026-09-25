"""Branch elements for a pressurised water distribution network (EPANET 2.2 forms).

Potential is hydraulic HEAD in metres; flow is m^3/s, signed along the edge's own
orientation. All constants are SI.
"""

from __future__ import annotations

import torch

from noodl.elements.base import Element
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor
F64 = torch.float64

#: SI Hazen-Williams resistance constant, ``h = K C^-1.852 d^-4.871 L q^1.852``.
#: Re-derived from the EPANET 2.2 Manual Table 3.1 US value 4.727 (feet, cfs) and equal to
#: wntr's own ``hw_k`` to 1.04e-9 relative. VERIFIED.
HW_SI = 10.666829500036

#: SI Darcy-Weisbach resistance constant ``h = K f d^-5 L q^2`` = ``8 / (g pi^2)``.
#: Confirmed against wntr's ``dw_k``; NOT re-derived from the manual's US form.
DW_SI = 0.0826

#: Standard gravity (m/s^2), used by the minor-loss and pump forms.
G = 9.80665


class HazenWilliams(Element):
    """``h_L = K q^1.852 + m q^2`` with ``K = 10.666829500036 C^-1.852 d^-4.871 L``.

    ``m = K_minor / (2 g A^2)`` is the optional minor-loss term of the same pipe (EPANET
    2.2 Manual p.19, ``h_m = K v^2 / 2g``); with ``m == 0`` (the usual case, and the only
    one in both committed fixtures) the inversion is the closed form
    ``q = sign(dp) (|dp| / K)^(1/1.852)``, laminar-blended below ``dp_transition`` exactly
    as ``PowerLaw`` does. With any nonzero ``m`` the law is inverted by a batched monotone
    root on ``[0, (|dp|/K)^(1/1.852)]``; that bracket is justified: ``h_L(q)`` is strictly
    increasing for ``q >= 0`` and the minor-loss term is non-negative, so the pure
    Hazen-Williams flow is an upper bound on the true one.

    ``dp_transition`` defaults to ``1e-9`` m. MEASURED, and NOT for the reason one would
    guess: on the committed two-loop fixture the choice is irrelevant -- the smallest head
    loss there is 5.8188e-3 m (pipe P7, the near-stagnant loop-closer), above every
    candidate, and the whole D1 parity is bit-identical at 1e-3, 1e-6, 1e-9 and 1e-12.
    Where it bites is a pipe carrying EXACTLY zero flow, which Net1 produces the moment a
    control closes its pump and leaves pipe 10 dead-ended: there ``dp`` is exactly 0, the
    blend is the only thing keeping the Jacobian finite, and its tangent slope
    ``(dp_transition/K)^(1/1.852) / dp_transition`` grows as the transition shrinks. Over
    Net1's 24 h duty cycle the worst Newton iteration count runs 50, 93, 136, 179 at
    1e-3, 1e-6, 1e-9, 1e-12, while the D3 tank trajectory CONVERGES with respect to the
    transition at 1e-6 and below (8.1813e-5 m against EPANET at 1e-6, 1e-9 and 1e-12;
    6.4922e-5 m at 1e-3, which is a coincidence of EPANET's own unquantified low-flow
    linearisation, not a better answer). 1e-9 is therefore the converged, faithful choice,
    and it is what fixes ``water_steady``'s ``max_iter`` at 200.
    """

    def __init__(
        self,
        length,
        diameter,
        roughness,
        *,
        minor_loss=0.0,
        dp_transition: float = 1e-9,
        kind: str = "pipe",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        self.diameter = self._param(diameter, False)
        self.roughness = self._param(roughness, learnable)
        self.minor_loss = self._param(minor_loss, False)
        self.dp_transition = float(dp_transition)
        self._has_minor = bool(torch.any(torch.as_tensor(minor_loss) != 0))

    def resistance(self) -> Tensor:
        """``K`` in ``h_L = K q^1.852`` (s^1.852 m^-4.556)."""
        return (
            HW_SI
            * self.roughness ** (-1.852)
            * self.diameter ** (-4.871)
            * self.length
        )

    def _minor(self) -> Tensor:
        """``m`` in ``h_m = m q^2``: ``K_minor / (2 g A^2)``."""
        area = torch.pi * self.diameter**2 / 4.0
        return self.minor_loss / (2.0 * G * area**2)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        k = self.resistance()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        # The un-taken branch must never see |dp| == 0: |dp|^(1/1.852 - 1) is infinite
        # there, and where()'s backward would turn inf * 0 into nan. Substituting the
        # constant transition value on the masked entries keeps both branches finite.
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        if not self._has_minor:
            sharp = (dp_safe / k) ** (1.0 / 1.852)
            slope = (dpt / k) ** (1.0 / 1.852) / dpt
            return torch.where(mask, slope * dp, torch.sign(dp) * sharp)
        m = self._minor()
        hi = (dp_safe / k) ** (1.0 / 1.852)

        def residual(q, h_t, k_t, m_t):
            return k_t * q**1.852 + m_t * q**2 - h_t

        sharp = solve_monotone(
            residual,
            torch.zeros_like(hi),
            hi,
            dp_safe,
            k * torch.ones_like(hi),
            m * torch.ones_like(hi),
            tol=1e-15,
            max_iter=100,
        )
        hi_t = (torch.full_like(dp, dpt) / k) ** (1.0 / 1.852)
        q_t = solve_monotone(
            residual,
            torch.zeros_like(hi_t),
            hi_t,
            torch.full_like(dp, dpt),
            k * torch.ones_like(hi_t),
            m * torch.ones_like(hi_t),
            tol=1e-15,
            max_iter=100,
        )
        return torch.where(mask, q_t * dp / dpt, torch.sign(dp) * sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        k = self.resistance()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        q = self.flow(dp, drivers).abs()
        q_safe = torch.where(mask, torch.full_like(q, 1e-30), q)
        slope_sharp = 1.0 / (1.852 * k * q_safe**0.852 + 2.0 * self._minor() * q_safe)
        if not self._has_minor:
            slope_lam = (dpt / k) ** (1.0 / 1.852) / dpt
        else:
            slope_lam = self.flow(torch.full_like(dp, dpt), drivers) / dpt
        return torch.where(mask, slope_lam * torch.ones_like(dp), slope_sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        k = self.resistance()
        dpt = self.dp_transition
        slope = (dpt / k) ** (1.0 / 1.852) / dpt
        zero = torch.zeros_like(slope)
        return zero, slope + zero


def three_point_curve(q_design: Tensor, h_design: Tensor) -> tuple[Tensor, Tensor, float]:
    """EPANET's single-point pump-curve fit: ``(h0, r, n)`` for ``h = h0 - r q^n``.

    EPANET 2.2 Manual p.19: "EPANET adds two more points to the curve by assuming a shutoff
    head at zero flow equal to 133% of the design head and a maximum flow at zero head equal
    to twice the design flow", then fits ``h = A - B q^C`` through the three points. Those
    three points force the exponent EXACTLY: ``B q_d^C = H/3`` and ``B (2 q_d)^C = 4H/3``
    give ``2^C = 4``, so ``C = 2`` and ``h0 = 4 H / 3``. (Confirmed against wntr's own
    `get_head_curve_coefficients` on Net1: (101.6, 2836.1385287628063, 2) for the file's
    1500 GPM / 250 ft point.)
    """
    h0 = 4.0 * h_design / 3.0
    r = (h_design / 3.0) / q_design**2
    return h0, r, 2.0


class PumpCurve(Element):
    """Constant-speed pump on its own edge kind: ``h_gain = w^2 (h0 - r (q/w)^n)``.

    The edge is oriented SUCTION -> DISCHARGE, so the layer's own
    ``dp = phi_suction - phi_discharge`` is the NEGATIVE of the head gain and the inversion
    is closed form:

        q = w ((h0 + dp / w^2) / r)^(1/n)      for  h0 + dp / w^2 > 0
        q = 0                                  otherwise (shut off)

    ``w`` is the relative speed setting (EPANET's affinity laws, Manual p.108); ``status``
    is read per edge from the driver ``status_key`` when given (0 closed, 1 open), which is
    how a ``[CONTROLS]`` line reaches the element. The fitted curve is a strict power law,
    so no root solve is needed; a monotone root solve would only be needed
    for a MULTI-point (piecewise-linear) curve, which is out of scope.

    Flow is monotone increasing in ``dp`` (``dq/ddp > 0``), so the Newton Jacobian stays
    SPD-certifiable. At shut-off the slope of the sharp branch diverges, so the law is
    laminar-blended over ``h0 + dp/w^2 < dp_transition`` with a safe input substituted on
    BOTH branches of the ``where``.

    ``q_max`` is the flow at zero head gain,
    ``w (h0/r)^(1/n)`` -- EPANET's own curve is only DEFINED up to this point (its own
    "two more points" construction stops there), and beyond it a positive ``dp`` (the
    pump run backwards, discharge below suction) still has a well-defined closed-form
    ``flow``, silently EXTRAPOLATING the power law with no physical curve behind it. This
    element keeps that extrapolation -- refusing it here would abort Newton on a merely
    transient iterate, before the solve has had a chance to walk back inside the curve's
    domain -- and leaves the refusal to ``water_steady``, which checks only the CONVERGED
    flow against ``q_max`` and raises by name (no clamp, no warning, matching EPANET's own
    silent extrapolation being exactly what this application declines to reproduce
    uncontrolled).
    """

    def __init__(
        self,
        h0,
        r,
        n=2.0,
        *,
        speed=1.0,
        dp_transition: float = 1e-6,
        status_key: str | None = None,
        kind: str = "pump",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.h0 = self._param(h0, learnable)
        self.r = self._param(r, learnable)
        self.n = self._param(n, False)
        self.speed = self._param(speed, False)
        self.dp_transition = float(dp_transition)
        self.status_key = status_key

    def _status(self, dp: Tensor, drivers) -> Tensor:
        if self.status_key is None:
            return torch.ones_like(dp)
        if drivers is None or self.status_key not in drivers:
            raise KeyError(
                f"PumpCurve (kind {self.kind!r}): driver {self.status_key!r} is required "
                f"and was not given; it carries each pump's open/closed status (1/0)"
            )
        return drivers[self.status_key].to(dp.dtype)

    def q_max(self) -> Tensor:
        """The flow at zero head gain, ``w (h0/r)^(1/n)``; see the class docstring."""
        return self.speed * (self.h0 / self.r) ** (1.0 / self.n)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        w = self.speed
        head = self.h0 + dp / w**2
        dpt = self.dp_transition
        shut = head < dpt
        head_safe = torch.where(shut, torch.full_like(head, dpt), head)
        sharp = w * (head_safe / self.r) ** (1.0 / self.n)
        # Below the transition the law is the straight line through (0, 0) and
        # (dp_transition, q(dp_transition)): continuous in value, finite in slope, and it
        # reaches exactly zero at the shut-off head instead of the infinite slope the sharp
        # branch has there.
        q_t = w * (torch.full_like(head, dpt) / self.r) ** (1.0 / self.n)
        laminar = q_t * torch.clamp(head, min=0.0) / dpt
        return self._status(dp, drivers) * torch.where(shut, laminar, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        w = self.speed
        head = self.h0 + dp / w**2
        dpt = self.dp_transition
        shut = head < dpt
        head_safe = torch.where(shut, torch.full_like(head, dpt), head)
        sharp = (w / (self.n * self.r * w**2)) * (head_safe / self.r) ** (1.0 / self.n - 1.0)
        q_t = w * (torch.full_like(head, dpt) / self.r) ** (1.0 / self.n)
        laminar = torch.where(head > 0.0, q_t / (dpt * w**2), torch.zeros_like(head))
        return self._status(dp, drivers) * torch.where(shut, laminar, sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        # Tangent at dp = 0 (the pump running on its own curve at zero head difference).
        zero = torch.zeros_like(self.h0 + self.r)
        return self.flow(zero, drivers), self.dflow(zero, drivers)


class MinorLoss(Element):
    """``h = m q |q|`` with ``m = K / (2 g A^2)``: a throttle control valve (TCV) or any
    pure minor-loss link. Inverted in closed form, ``q = sign(h) sqrt(|h| / m)``, with the
    same laminar blend and both-branch safe substitution as ``HazenWilliams``."""

    def __init__(
        self,
        setting,
        diameter,
        *,
        dp_transition: float = 1e-9,
        kind: str = "valve",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.setting = self._param(setting, learnable)
        self.diameter = self._param(diameter, False)
        self.dp_transition = float(dp_transition)

    def coefficient(self) -> Tensor:
        area = torch.pi * self.diameter**2 / 4.0
        return self.setting / (2.0 * G * area**2)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        m = self.coefficient()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = torch.sign(dp) * torch.sqrt(dp_safe / m)
        # `torch.as_tensor(dpt)` on a bare Python float casts with
        # `torch.get_default_dtype()` (float32 in this repo, `elements/duct.py`'s
        # documented gotcha), which would silently downcast a float64 element; `dtype=
        # dp.dtype` keeps it in the caller's own precision, as `Headspace` does.
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / m) / dpt
        return torch.where(mask, slope * dp, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        m = self.coefficient()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = 0.5 / torch.sqrt(m * dp_safe)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / m) / dpt
        return torch.where(mask, slope * torch.ones_like(dp), sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        m = self.coefficient()
        dpt = self.dp_transition
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=m.dtype) / m) / dpt
        zero = torch.zeros_like(slope)
        return zero, slope + zero
