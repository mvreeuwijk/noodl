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
    candidate, and the whole `test_two_loop_heads_and_flows` comparison is bit-identical
    at 1e-3, 1e-6, 1e-9 and 1e-12.
    Where it bites is a pipe carrying EXACTLY zero flow, which Net1 produces the moment a
    control closes its pump and leaves pipe 10 dead-ended: there ``dp`` is exactly 0, the
    blend is the only thing keeping the Jacobian finite, and its tangent slope
    ``(dp_transition/K)^(1/1.852) / dp_transition`` grows as the transition shrinks. Over
    Net1's 24 h duty cycle the worst Newton iteration count runs 50, 93, 136, 179 at
    1e-3, 1e-6, 1e-9, 1e-12, while the Net1 tank trajectory
    (`test_net1_extended_period_tank_level`) CONVERGES with respect to the
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


# ------------------------------------------------------------- composite (EPANET) D-W
#: EPANET 2.2's own constants for the Darcy-Weisbach friction factor, copied digit for
#: digit from `src/hydcoeffs.c` (v2.2), where they are ``A1``, ``A2``, ``A8``, ``A9``,
#: ``AB`` and ``AC``. ``w = q / (nu d) = Re pi / 4``, so ``W_RE_4000`` (``A1``) is
#: Re = 4000 and ``W_RE_2000`` (``A2``) is Re = 2000.
W_RE_4000 = 3.14159265358979323850e03  # 1000 pi
W_RE_2000 = 1.57079632679489661930e03  # 500 pi
SWAMEE_JAIN_W_COEFF = 4.61841319859066668690e00  # 5.74 (pi/4)^0.9
SWAMEE_JAIN_LOG_COEFF = -8.68588963806503655300e-01  # -2 / ln(10)
DUNLOP_AB = 3.28895476345399058690e-03  # 5.74 / 4000^0.9
DUNLOP_AC = -5.14214965799093883760e-03  # -2 * 0.9 * 2 / ln(10) * AB

#: EPANET computes in feet and cfs. ``MperFT`` (`types.h`) is exact; the gravity in the
#: D-W resistance ``R = L / (2 * 32.2 * d * A^2)`` (`hydcoeffs.c`, `resistcoeff`) is
#: 32.2 ft/s^2, NOT standard gravity's 32.174, and the minor-loss conversion
#: ``0.02517 K / d^4`` (`input1.c`, `convertunits`) is ``8 / (32.2 pi^2)`` ROUNDED to
#: four figures. Both are part of what EPANET computes, so both are reproduced here.
M_PER_FT = 0.3048
G_FT_S2 = 32.2
MINOR_LOSS_K_FT = 0.02517
#: EPANET's default kinematic viscosity of water, ``VISCOS = 1.1e-5`` ft^2/s (`types.h`),
#: = 1.0219e-6 m^2/s; ``[OPTIONS] VISCOSITY`` multiplies it.
NU_WATER_FT2 = 1.1e-5
NU_WATER_REF = NU_WATER_FT2 * M_PER_FT**2
#: EPANET's own flow-unit factors, file units per cfs (`types.h`, ``GPMperCFS`` ...).
#: Several are rounded (``LPSperCFS = 28.317`` against the exact 28.3168466), and EPANET
#: divides a file's flows by exactly these numbers on the way in.
EPANET_QCF = {
    "CFS": 1.0,
    "GPM": 448.831,
    "MGD": 0.64632,
    "IMGD": 0.5382,
    "AFD": 1.9837,
    "LPS": 28.317,
    "LPM": 1699.0,
    "MLD": 2.4466,
    "CMH": 101.94,
    "CMD": 2446.6,
}


def _dunlop(w: Tensor, e: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Dunlop's cubic coefficients ``x1..x4`` in ``r = w / A2`` (`frictionFactor`)."""
    y2 = e / 3.7 + DUNLOP_AB
    y3 = SWAMEE_JAIN_LOG_COEFF * torch.log(y2)
    fa = 1.0 / (y3 * y3)
    fb = (2.0 + DUNLOP_AC / (y2 * y3)) * fa
    x1 = 7.0 * fa - fb
    x2 = 0.128 - 17.0 * fa + 2.5 * fb
    x3 = -0.128 + 13.0 * fa - (fb + fb)
    x4 = 0.032 - 3.0 * fa + 0.5 * fb
    return x1, x2, x3, x4


def composite_friction_factor(q: Tensor, e: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """Composite friction factor for ``Re > 2000``: ``(f, df/dq)``.

    This is EPANET 2.2's ``frictionFactor`` (`hydcoeffs.c`).

    ``q`` is ``|flow|``, ``e`` the relative roughness ``eps / d`` and ``s = nu d`` (any
    consistent units: ``w = q / s = Re pi / 4``). Swamee-Jain for Re >= 4000, Dunlop's
    cubic in ``r = Re / 2000`` between 2000 and 4000. The laminar ``f = 64 / Re`` is not
    here because EPANET does not compute it as a friction factor either: `DWpipecoeff`
    applies Hagen-Poiseuille to the head loss directly.

    Both branches are evaluated everywhere and selected with ``where``; the Swamee-Jain
    branch sees ``w`` clamped to at least ``A1`` so its un-taken values stay finite.
    """
    w = q / s
    turbulent = w >= W_RE_4000
    w_sj = torch.clamp(w, min=W_RE_4000)
    q_sj = w_sj * s
    y1 = SWAMEE_JAIN_W_COEFF / w_sj**0.9
    y2 = e / 3.7 + y1
    y3 = SWAMEE_JAIN_LOG_COEFF * torch.log(y2)
    f_sj = 1.0 / (y3 * y3)
    dfdq_sj = 1.8 * f_sj * y1 * SWAMEE_JAIN_LOG_COEFF / y2 / y3 / q_sj
    x1, x2, x3, x4 = _dunlop(w, e)
    r = w / W_RE_2000
    f_tr = x1 + r * (x2 + r * (x3 + r * x4))
    dfdq_tr = (x2 + r * (2.0 * x3 + r * 3.0 * x4)) / s / W_RE_2000
    return torch.where(turbulent, f_sj, f_tr), torch.where(turbulent, dfdq_sj, dfdq_tr)


class CompositeDarcyWeisbach(Element):
    """Composite Darcy-Weisbach pipe: Hagen-Poiseuille, Dunlop's cubic, then Swamee-Jain.

    This is EPANET 2.2's Darcy-Weisbach pipe, reproduced operation for operation
    (`DWpipecoeff`). The pre-rename name ``EpanetDarcyWeisbach`` is an alias.

    Head loss in EPANET's units (feet, cfs; ``q`` signed)::

        h = (16 pi nu d R + m |q|) q        Re <= 2000   (Hagen-Poiseuille)
        h = (f(Re) R + m) |q| q             Re >  2000   (Dunlop cubic, then Swamee-Jain)

    with ``R = L / (2 * 32.2 * d * A^2)`` and ``m = 0.02517 K / d^4``. The composite is
    C^1 in exact arithmetic: Dunlop's cubic matches ``64 / Re`` in value and slope at
    Re = 2000 and Swamee-Jain in value and slope at Re = 4000. EPANET has NO low-flow
    linearisation for D-W (the ``RQtol`` floor in `pipecoeff` is on the H-W / C-M branch
    only): the laminar branch is already linear through ``q = 0``, which is also why this
    element needs no transition blend and has a finite slope at ``dp = 0``.

    Inputs are SI (m, m^3/s, m^2/s) and are converted to feet internally. ``cfs_per_m3s``
    is the flow conversion EPANET itself applies: it reads flows in the file's units and
    divides by its own ROUNDED factor (``LPSperCFS = 28.317``, ``GPMperCFS = 448.831``, ...
    in `types.h`), so the internal cfs of an LPS file are ``1000 / 28.317`` per m^3/s,
    not ``1 / 0.3048^3``. ``None`` selects the exact conversion.

    The layer potential may be head times ``scale`` (the water application's D-W path
    works in pressure, ``scale = rho g``): ``dp = scale * h`` with ``h`` in metres.

    ``flow(dp)`` inverts the odd, strictly increasing ``h(q)`` by Newton's method with
    EPANET's own analytic gradient, safeguarded by bisection on ``[0, q_hi]`` (``q_hi``
    doubled until it brackets the root) and run without autograd; ONE final Newton step is
    then taken on the graph, whose value is the converged flow and whose first derivative
    in ``dp`` and in every parameter is the implicit-function one, ``-(dh/dtheta) / h'(q)``.
    It is taken on the SIGNED flow, so the derivative at ``q = 0`` is the finite laminar
    ``1 / h'(0)`` rather than the zero a ``sign(dp)`` factor would give. (A generic
    ``solve_monotone`` inversion was measured 5x slower, since it needs an autograd
    derivative per iteration and a collapsed bracket to stop.)
    """

    def __init__(
        self,
        length,
        diameter,
        roughness,
        *,
        minor_loss=0.0,
        nu: float = NU_WATER_REF,
        cfs_per_m3s: float | None = None,
        scale: float = 1.0,
        rtol: float = 1e-15,
        max_iter: int = 100,
        kind: str = "pipe",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        self.diameter = self._param(diameter, False)
        self.roughness = self._param(roughness, learnable)
        self.minor_loss = self._param(minor_loss, False)
        self.nu = float(nu)
        self.cfs_per_m3s = (
            1.0 / M_PER_FT**3 if cfs_per_m3s is None else float(cfs_per_m3s)
        )
        self.scale = float(scale)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)

    def _coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(R, m, e, s)`` in feet/cfs: resistance, minor-loss coefficient, relative
        roughness and ``nu d``, broadcast to a common shape."""
        d = self.diameter / M_PER_FT
        length = self.length / M_PER_FT
        area = torch.pi * d**2 / 4.0
        r = length / 2.0 / G_FT_S2 / d / area**2
        m = MINOR_LOSS_K_FT * self.minor_loss / d**2 / d**2
        e = (self.roughness / M_PER_FT) / d
        s = (self.nu / M_PER_FT**2) * d
        return torch.broadcast_tensors(r, m, e, s)

    @staticmethod
    def _head_ft(q: Tensor, r: Tensor, m: Tensor, e: Tensor, s: Tensor) -> Tensor:
        """Signed head loss (ft) at signed flow ``q`` (cfs)."""
        a = q.abs()
        laminar = a <= W_RE_2000 * s
        f, _ = composite_friction_factor(torch.where(laminar, W_RE_2000 * s, a), e, s)
        h_lam = (16.0 * torch.pi * s * r + m * a) * q
        h_turb = (f * r + m) * a * q
        return torch.where(laminar, h_lam, h_turb)

    def head_loss(self, q: Tensor) -> Tensor:
        """Signed head loss in METRES at signed flow ``q`` in m^3/s (EPANET's ``hloss``)."""
        r, m, e, s = self._coefficients()
        return M_PER_FT * self._head_ft(q * self.cfs_per_m3s, r, m, e, s)

    @staticmethod
    def _grad_ft(a: Tensor, r: Tensor, m: Tensor, e: Tensor, s: Tensor) -> Tensor:
        """EPANET's ``hgrad`` (ft per cfs) at ``|q| = a`` (cfs)."""
        laminar = a <= W_RE_2000 * s
        a_t = torch.where(laminar, W_RE_2000 * s, a)
        f, dfdq = composite_friction_factor(a_t, e, s)
        g_lam = 16.0 * torch.pi * s * r + 2.0 * m * a
        g_turb = 2.0 * (f * r + m) * a + dfdq * r * a * a
        return torch.where(laminar, g_lam, g_turb)

    def head_gradient(self, q: Tensor) -> Tensor:
        """``dh/dq`` in m per (m^3/s), from EPANET's own analytic ``hgrad``."""
        r, m, e, s = self._coefficients()
        a = (q * self.cfs_per_m3s).abs()
        return M_PER_FT * self.cfs_per_m3s * self._grad_ft(a, r, m, e, s)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        r, m, e, s = self._coefficients()
        c = self.cfs_per_m3s
        h_ft, r, m, e, s = torch.broadcast_tensors(
            dp / self.scale / M_PER_FT, r, m, e, s
        )
        with torch.no_grad():
            r0, m0, e0, s0 = r.detach(), m.detach(), e.detach(), s.detach()
            target = h_ft.detach().abs()
            # Upper bound on |q|: the larger of the laminar flow and the flow at an
            # optimistic f = 0.005, doubled until the head loss there reaches the target.
            hi = torch.maximum(
                torch.sqrt(target / (0.005 * r0 + m0)),
                target / (16.0 * torch.pi * s0 * r0),
            )
            for _ in range(200):
                short = self._head_ft(hi, r0, m0, e0, s0) < target
                if not bool(short.any()):
                    break
                hi = torch.where(short, 2.0 * hi, hi)
            # Safeguarded Newton with EPANET's analytic gradient, from the upper bound.
            lo = torch.zeros_like(hi)
            a = hi.clone()
            for _ in range(self.max_iter):
                res = self._head_ft(a, r0, m0, e0, s0) - target
                hi = torch.where(res > 0, a, hi)
                lo = torch.where(res < 0, a, lo)
                step = res / self._grad_ft(a, r0, m0, e0, s0)
                nxt = a - step
                inside = (nxt >= lo) & (nxt <= hi) & torch.isfinite(nxt)
                nxt = torch.where(inside, nxt, 0.5 * (lo + hi))
                converged = (res == 0) | ((nxt - a).abs() <= self.rtol * a)
                a = torch.where(res == 0, a, nxt)
                if bool(converged.all()):
                    break
            else:
                raise RuntimeError(
                    f"CompositeDarcyWeisbach (kind {self.kind!r}): the head-loss inversion "
                    f"did not converge in {self.max_iter} iterations"
                )
            q_star = torch.sign(h_ft.detach()) * a
            g_star = self._grad_ft(a, r0, m0, e0, s0)
        # One Newton step taken WITH the autograd graph: its value is q_star (the residual
        # is zero to rounding) and its first derivative in every input is the implicit-
        # function one, -(dh/dtheta - dh_t/dtheta) / h'(q), finite at q = 0.
        q_cfs = q_star - (self._head_ft(q_star, r, m, e, s) - h_ft) / g_star
        return q_cfs / c

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        q = self.flow(dp, drivers)
        return 1.0 / (self.scale * self.head_gradient(q))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        # The laminar tangent at q = 0, as `Duct.linear_init` uses its laminar line.
        slope = self.dflow(torch.zeros_like(self.length * self.diameter), drivers)
        return torch.zeros_like(slope), slope


# Pre-rename names, kept as aliases so existing code keeps working.
EpanetDarcyWeisbach = CompositeDarcyWeisbach  # alias, the pre-rename name
epanet_friction_factor = composite_friction_factor  # alias, the pre-rename name
EPANET_A1 = W_RE_4000  # alias, the pre-rename name
EPANET_A2 = W_RE_2000  # alias, the pre-rename name
EPANET_A8 = SWAMEE_JAIN_W_COEFF  # alias, the pre-rename name
EPANET_A9 = SWAMEE_JAIN_LOG_COEFF  # alias, the pre-rename name
EPANET_AB = DUNLOP_AB  # alias, the pre-rename name
EPANET_AC = DUNLOP_AC  # alias, the pre-rename name
EPANET_FOOT = M_PER_FT  # alias, the pre-rename name
EPANET_G_FT = G_FT_S2  # alias, the pre-rename name
EPANET_KM = MINOR_LOSS_K_FT  # alias, the pre-rename name
EPANET_VISCOS_FT2 = NU_WATER_FT2  # alias, the pre-rename name
EPANET_NU = NU_WATER_REF  # alias, the pre-rename name
