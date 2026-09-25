"""MBL's power-law flow-element family, transcribed from MBL v13.0.0
(commit 55abf579598ca81cae0a82f337350375958e6722).

All six of MBL's `Buildings.Airflow.Multizone` power-law elements (`Orifice`,
`EffectiveAirLeakageArea`, `Point_m_flow`, `Points_m_flow`, `Coefficient_V_flow`,
`Coefficient_m_flow`) ultimately evaluate one regularised law,

    q = C sign(dp) |dp|^m,           |dp| >= dp_turbulent
    q = C dp_turbulent^m * pi * (a + pi^2*(b + pi^2*(c + pi^2*d))),   pi = dp/dp_turbulent,
                                      |dp| <  dp_turbulent

(``BaseClasses/powerLaw.mo:17-29``), a 7th-order odd polynomial in ``pi`` chosen to be twice
continuously differentiable in ``dp`` at the switch, with

    a = gamma,                                    gamma = 1.5
    b = 1/8 m^2 - 3 gamma - 3/2 m + 35/8
    c = -1/4 m^2 + 3 gamma + 5/2 m - 21/4
    d = 1/8 m^2 - gamma - m + 15/8

(``BaseClasses/PowerLawResistanceParameters.mo:6-15``). MBL calls this ``powerLawFixedM`` (or
its ``m=0.5`` specialisation ``powerLaw05``) from ``Coefficient_V_flow``/``Coefficient_m_flow``
because, in every element in this family, ``m`` is a *parameter* fixed at construction, never a
time-varying variable (``Coefficient_V_flow.mo:5-24``, ``Coefficient_m_flow.mo:4-24``: both
switch on ``Modelica.Math.isEqual(m, 0.5, 1E-10)`` between ``powerLaw05`` and
``powerLawFixedM``, never the runtime-variable-``m`` ``powerLaw`` function). Because
``powerLawFixedM``'s ``a``, ``b``, ``c``, ``d`` are exactly ``PowerLawResistanceParameters``'
own formula evaluated at that same fixed ``m``, evaluating the polynomial directly from ``m``
(as :meth:`MBLPowerLaw._regularised` below does) is mathematically identical to precomputing
those coefficients once and calling ``powerLawFixedM`` -- ``test_powerlaw.py`` checks this
equivalence independently in NumPy.

Two forms share this one law (the "volume"/"mass" split):

* ``form="volume"`` (``Coefficient_V_flow``, and hence ``Orifice``/``EffectiveAirLeakageArea``,
  which extend it): the law computes a VOLUME flow rate, ``m_flow = rho_default * V_flow(dp)``
  (``Coefficient_V_flow.mo:4``: ``m_flow = V_flow*rho``, with ``rho = rho_default`` under the
  ``useDefaultProperties=true`` scope implemented here,
  ``BaseClasses/PartialOneWayFlowElement.mo:54-57``).
* ``form="mass"`` (``Coefficient_m_flow``, and hence ``Point_m_flow``/``Points_m_flow``, which
  extend it): the parameter is a MASS-flow coefficient ``k`` and the law computes ``m_flow``
  directly, with no separate ``rho`` multiplication needed here -- ``Coefficient_m_flow.mo:31``
  builds ``C = k/rho_default`` purely so the *same* volumetric ``powerLaw05``/``powerLawFixedM``
  call can be reused, then ``Coefficient_m_flow.mo:4-24`` multiplies the result by
  ``rho = rho_default`` again, so the ``rho_default`` MBL introduces and removes cancels
  exactly: ``m_flow = rho_default * (k/rho_default) * dp^m = k * dp^m``. This module's
  ``form="mass"`` therefore evaluates the polynomial directly on the caller's ``C`` (MBL's
  ``k``) with no ``rho_default`` factor, which is exact under ``useDefaultProperties=true``
  (the only implemented path) and is what the docstrings above call "the same
  regularisation" on a plain mass-flow coefficient.
"""

from __future__ import annotations

import math

import torch

from noodl.elements.base import Element

Tensor = torch.Tensor

_GAMMA = 1.5  # BaseClasses/PowerLawResistanceParameters.mo:6 (also powerLaw.mo:15)


def _f64(value) -> Tensor:
    """A caller's tensor keeps its own dtype; a bare Python number becomes float64 -- this
    package is explicit-float64 throughout, unlike
    ``noodl.elements.powerlaw``'s default-dtype convenience, which this new package has no
    need to mirror.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


class MBLPowerLaw(Element):
    """MBL's regularised power law, ``q = C sign(dp) |dp|^m``, volume or mass form.

    ``flow(dp)`` always returns MASS flow in kg/s: ``form="volume"`` multiplies the volumetric
    law by ``rho_default``; ``form="mass"`` returns the law's own output directly (see the
    module docstring for why no further density factor belongs there).
    """

    def __init__(
        self,
        C,
        m,
        *,
        dp_turbulent: float = 0.1,
        form: str,
        rho_default: float,
        kind: str = "airpath",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        if form not in ("volume", "mass"):
            raise ValueError(
                f"MBLPowerLaw (kind {kind!r}): form must be 'volume' or 'mass', got {form!r}"
            )
        self.C = self._param(_f64(C), learnable)
        self.m = self._param(_f64(m), learnable)
        self.dp_turbulent = float(dp_turbulent)
        self.form = form
        self.rho_default = float(rho_default)

    def _regularised(self, dp: Tensor) -> Tensor:
        """``BaseClasses/powerLaw.mo:17-29`` (equivalently ``powerLawFixedM.mo:21-30`` with
        ``a``, ``b``, ``c``, ``d`` precomputed from ``m`` by
        ``PowerLawResistanceParameters.mo:9-15`` -- see the module docstring for why
        evaluating them inline here is mathematically identical): the sharp law
        ``C sign(dp) |dp|^m`` for ``|dp| >= dp_turbulent``, replaced by a 7th-order odd
        polynomial in ``pi = dp/dp_turbulent`` inside the band.

        Follows ``noodl.elements.powerlaw.PowerLaw``'s ``dp_safe`` technique: the unselected
        "sharp" branch never evaluates ``|dp|^m`` at ``dp == 0`` itself (``|dp|^m`` is singular
        there for ``m < 1``), substituting the constant ``dp_turbulent`` under the mask so
        ``torch.where``'s backward multiplies a finite value by the zero mask instead of
        ``inf * 0 = nan``.
        """
        C, m = self.C, self.m
        dpt = self.dp_turbulent
        a = _GAMMA
        b = 1 / 8 * m**2 - 3 * _GAMMA - 3 / 2 * m + 35.0 / 8
        c = -1 / 4 * m**2 + 3 * _GAMMA + 5 / 2 * m - 21.0 / 4
        d = 1 / 8 * m**2 - _GAMMA - m + 15.0 / 8

        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = C * torch.sign(dp) * dp_safe**m

        pi = dp / dpt
        pi2 = pi * pi
        inner = C * dpt**m * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))

        return torch.where(mask, inner, sharp)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        q = self._regularised(dp)
        if self.form == "volume":
            return self.rho_default * q
        return q


def mbl_orifice(
    A,
    CD: float = 0.65,
    m: float = 0.5,
    dp_turbulent: float = 0.1,
    *,
    rho_default: float,
    kind: str = "airpath",
    learnable: bool = False,
) -> MBLPowerLaw:
    """``Orifice.mo:3-5,9``: ``C = CD * A * sqrt(2 / rho_default)``, ``m = 0.5`` by default
    (overridable, per ``Orifice.mo:3``'s non-``final`` ``m=0.5`` modifier); ``C`` itself is
    ``final`` in MBL (does not depend on ``m``). ``CD`` defaults to 0.65
    (``Orifice.mo:9``, the sharp-edged-orifice discharge coefficient).
    """
    A_t, CD_t, m_t = _f64(A), _f64(CD), _f64(m)
    C = CD_t * A_t * math.sqrt(2.0 / rho_default)
    return MBLPowerLaw(
        C=C,
        m=m_t,
        dp_turbulent=dp_turbulent,
        form="volume",
        rho_default=rho_default,
        kind=kind,
        learnable=learnable,
    )


def mbl_ela(
    L,
    dpRat: float = 4.0,
    CDRat: float = 1.0,
    m: float = 0.65,
    dp_turbulent: float = 0.1,
    *,
    rho_default: float,
    kind: str = "airpath",
    learnable: bool = False,
) -> MBLPowerLaw:
    """``EffectiveAirLeakageArea.mo:3-5,7-14,16``:
    ``C = L * CDRat * sqrt(2 / rho_default) * dpRat ** (0.5 - m)``, ``m = 0.65`` by default
    (``:4``), ``dpRat = 4`` Pa (``:7-9``), ``CDRat = 1`` (``:11-14``).
    """
    L_t, dpRat_t, CDRat_t, m_t = _f64(L), _f64(dpRat), _f64(CDRat), _f64(m)
    C = L_t * CDRat_t * math.sqrt(2.0 / rho_default) * dpRat_t ** (0.5 - m_t)
    return MBLPowerLaw(
        C=C,
        m=m_t,
        dp_turbulent=dp_turbulent,
        form="volume",
        rho_default=rho_default,
        kind=kind,
        learnable=learnable,
    )


def mbl_point(
    dpMea,
    mMea_flow,
    m: float = 0.5,
    dp_turbulent: float = 0.1,
    *,
    rho_default: float,
    kind: str = "airpath",
    learnable: bool = False,
) -> MBLPowerLaw:
    """``Point_m_flow.mo:4-5``: ``k = mMea_flow_nominal / dpMea_nominal ** m``, fit from one
    measured (dp, m_flow) pair; ``m`` stays a free parameter (default 0.5, inherited from
    ``Coefficient_m_flow``'s ``PowerLawResistanceParameters(m=0.5)``,
    ``Coefficient_m_flow.mo:27-28``).
    """
    dpMea_t, mMea_t, m_t = _f64(dpMea), _f64(mMea_flow), _f64(m)
    k = mMea_t / dpMea_t**m_t
    return MBLPowerLaw(
        C=k,
        m=m_t,
        dp_turbulent=dp_turbulent,
        form="mass",
        rho_default=rho_default,
        kind=kind,
        learnable=learnable,
    )


def mbl_points(
    dpMea,
    mMea_flow,
    dp_turbulent: float = 0.1,
    *,
    rho_default: float,
    kind: str = "airpath",
    learnable: bool = False,
) -> MBLPowerLaw:
    """``Points_m_flow.mo:4-6,15-16``: the flow exponent itself is derived from two measured
    (dp, m_flow) pairs, ``m = (ln(mMea[0]) - ln(mMea[1])) / (ln(dpMea[0]) - ln(dpMea[1]))``,
    then ``k = mMea[0] / dpMea[0] ** m``. ``dpMea``/``mMea_flow`` are each a 2-element sequence.
    """
    dpMea_t = _f64(dpMea)
    mMea_t = _f64(mMea_flow)
    if dpMea_t.shape[-1] != 2 or mMea_t.shape[-1] != 2:
        raise ValueError(
            "mbl_points: dpMea and mMea_flow must each carry exactly 2 measured points "
            f"(the two-point fit of Points_m_flow.mo), got shapes "
            f"{tuple(dpMea_t.shape)} and {tuple(mMea_t.shape)}"
        )
    dp1, dp2 = dpMea_t[..., 0], dpMea_t[..., 1]
    mf1, mf2 = mMea_t[..., 0], mMea_t[..., 1]
    m = (torch.log(mf1) - torch.log(mf2)) / (torch.log(dp1) - torch.log(dp2))
    k = mf1 / dp1**m
    return MBLPowerLaw(
        C=k,
        m=m,
        dp_turbulent=dp_turbulent,
        form="mass",
        rho_default=rho_default,
        kind=kind,
        learnable=learnable,
    )


def mbl_coefficient(
    C,
    m,
    form: str,
    dp_turbulent: float = 0.1,
    *,
    rho_default: float,
    kind: str = "airpath",
    learnable: bool = False,
) -> MBLPowerLaw:
    """``Coefficient_V_flow.mo``/``Coefficient_m_flow.mo``: ``C``/``k`` and ``m`` given
    directly, with no further formula -- a direct pass-through to :class:`MBLPowerLaw`.
    """
    return MBLPowerLaw(
        C=_f64(C),
        m=_f64(m),
        dp_turbulent=dp_turbulent,
        form=form,
        rho_default=rho_default,
        kind=kind,
        learnable=learnable,
    )
