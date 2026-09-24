"""MBL/MSL media constants and densities.

Transcribed from Modelica Buildings Library (MBL) v13.0.0
(commit 55abf579598ca81cae0a82f337350375958e6722) and the Modelica Standard Library (MSL)
v4.1.0, for the three media MBL's `Buildings.Airflow.Multizone` validation/example models use:
`Buildings.Media.Air`, `Buildings.Media.Specialized.Air.PerfectGas` and
`Modelica.Media.Air.SimpleAir`.

Two densities matter, and MBL keeps them structurally separate (design section 2, "Fidelity";
research inventory section C):

* ``rho_default``: the medium's density at its own *fixed* default state
  (``p_default``/``T_default``/``X_default``), used to build the flow coefficient ``C``/``k``
  of every power-law element in :mod:`noodl.elements.mbl.powerlaw`
  (``BaseClasses/PartialOneWayFlowElement.mo:23-28``: ``sta_default =
  Medium.setState_pTX(T=Medium.T_default, p=Medium.p_default, X=Medium.X_default)``,
  ``rho_default = Medium.density(sta_default)``). It is a plain Python ``float``, computed once
  and never revisited at simulation time.
* ``buoyancy_density(T, X_w)``: the buoyancy-relevant density MBL's ``MediumColumn``,
  ``TwoWayFlowElement`` and ``DoorDiscretized`` call instead of ``Medium.density`` -- always
  ``Buildings.Utilities.Psychrometrics.Functions.density_pTX`` (or, for a medium other than
  ``Buildings.Media.Air``, that medium's own ideal-gas density function), evaluated at the
  medium's fixed ``p_default`` but the *actual* local temperature and water content. This
  method is differentiable in ``T`` (and ``X_w`` where applicable): the design's door/column
  primitives (a later task) read live zone temperatures through it.

For ``Buildings.Media.Air`` these two are NOT the same function: ``Medium.density`` is
pressure-only (``Air.mo``'s own banner: "decouples pressure and temperature"), so
``rho_default`` is a fixed ~1.2 kg/m3 regardless of temperature, while ``buoyancy_density``
genuinely depends on ``T``. Only the ``useDefaultProperties=true`` path (MBL's default, and the
only one this task implements) is covered; a class whose default is ``false`` is out of scope
for this task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

Tensor = torch.Tensor

# ---------------------------------------------------------------------------------------
# Shared MSL constants
# ---------------------------------------------------------------------------------------

# Modelica/Constants.mo:47 (k, Boltzmann constant), :53 (N_A, Avogadro constant), :49
# (R = k * N_A, the CODATA universal gas constant Modelica.Media.Air.SimpleAir uses).
_BOLTZMANN_K = 1.380649e-23
_AVOGADRO_N_A = 6.02214076e23
_MODELICA_CONSTANTS_R = _BOLTZMANN_K * _AVOGADRO_N_A

# Modelica/Media/IdealGases/Common/SingleGasesData.mo:5 (R_NASA_2002, the NASA-2002 universal
# gas constant every ideal-gas species record uses to build its own specific gas constant
# R_s = R_NASA_2002 / MM), :49 (Air.MM), :59 (Air.R_s), :9187 (H2O.MM), :9197 (H2O.R_s). These
# feed both `Buildings.Utilities.Psychrometrics.Functions.density_pTX` and
# `Buildings.Media.Specialized.Air.PerfectGas`'s `gasConstant` (PerfectGas.mo:134-136,
# :562-572), which build the identical mixture gas constant from the identical two species.
_R_NASA_2002 = 8.314510
_MM_AIR = 0.0289651159
_MM_H2O = 0.01801528
_R_AIR = _R_NASA_2002 / _MM_AIR
_R_H2O = _R_NASA_2002 / _MM_H2O

# Modelica/Media/package.mo:3969 (p_default=101325 Pa), :3971 (T_default=from_degC(20)=293.15
# K): PartialMedium's generic defaults. None of the three media below override them
# (Buildings/Media/Air.mo:1-15, Buildings/Media/Specialized/Air/PerfectGas.mo:1-14,
# Modelica/Media/Air/SimpleAir.mo:1-15), so all three share the same p_default/T_default.
_P_DEFAULT = 101325.0
_T_DEFAULT = 293.15


def _moist_air_gas_constant(X_w) -> Tensor:
    """R(X_w) = R_air*(1 - X_w) + R_h2o*X_w.

    Buildings/Utilities/Psychrometrics/Functions/density_pTX.mo:11-14 (also
    Buildings/Media/Specialized/Air/PerfectGas.mo:134-136, function ``gasConstant``).
    Differentiable in ``X_w``.
    """
    X_w = torch.as_tensor(X_w)
    return _R_AIR * (1 - X_w) + _R_H2O * X_w


@dataclass(frozen=True)
class MBLMedium:
    """One MBL/MSL medium's default state and the two densities its flow elements need.

    ``buoyancy_density`` is stored as a private callable rather than dispatched by ``name``,
    so each medium's own ideal-gas law (or, for ``Buildings.Media.Air``, ``density_pTX``) is
    fixed at construction time in :func:`medium` below and never re-derived from ``name`` at
    call time.
    """

    name: str
    p_default: float
    T_default: float
    X_default: tuple[float, ...]
    rho_default: float
    has_moisture: bool
    _buoyancy: Callable[[Tensor, Tensor | float], Tensor] = field(repr=False, compare=False)

    def buoyancy_density(self, T: Tensor, X_w: Tensor | float) -> Tensor:
        """The buoyancy-relevant density at this medium's fixed ``p_default`` and actual T/X_w.

        For ``Buildings.Media.Air`` this is ``density_pTX(p_default, T, X_w)``
        (``Utilities/Psychrometrics/Functions/density_pTX.mo``); for the other two media it is
        that medium's own ideal-gas density function evaluated at ``p_default`` instead of the
        actual port pressure. ``X_w`` is accepted but unused for a medium with no moisture
        (``has_moisture`` is ``False``).
        """
        return self._buoyancy(T, X_w)


def _air_medium() -> MBLMedium:
    """Buildings.Media.Air (Buildings/Media/Air.mo).

    ``Air.mo:43-45``: ``pStp = reference_p`` (=101325 Pa, MSL default), ``dStp = 1.2`` kg/m3.
    ``Air.mo:210-215`` (function ``density``): ``d := state.p*dStp/pStp`` -- pressure-only,
    exactly as the package banner ("decouples pressure and temperature") states.
    ``BaseClasses/PartialOneWayFlowElement.mo:23-28``: ``sta_default`` is built at
    ``Medium.p_default`` (=101325 Pa, unchanged from the MSL default since Air.mo does not
    override it), so ``rho_default = p_default*dStp/pStp`` collapses to exactly ``dStp``
    because ``p_default == pStp == reference_p`` here.
    """
    dStp = 1.2
    pStp = _P_DEFAULT  # reference_p, Air.mo does not override PartialMedium's p_default
    rho_default = _P_DEFAULT * dStp / pStp

    def buoyancy(T, X_w):
        # Utilities/Psychrometrics/Functions/density_pTX.mo:5-16: d := p/(R*T), evaluated at
        # this medium's fixed p_default (never the actual port pressure).
        T = torch.as_tensor(T)
        R = _moist_air_gas_constant(X_w).to(T.dtype)
        return _P_DEFAULT / (R * T)

    return MBLMedium(
        name="Buildings.Media.Air",
        p_default=_P_DEFAULT,
        T_default=_T_DEFAULT,
        X_default=(0.01, 0.99),  # Air.mo:9 (reference_X={0.01,0.99}); X_default=reference_X
        rho_default=rho_default,
        has_moisture=True,
        _buoyancy=buoyancy,
    )


def _perfectgas_medium() -> MBLMedium:
    """Buildings.Media.Specialized.Air.PerfectGas (Buildings/Media/Specialized/Air/PerfectGas.mo).

    A true ideal gas: ``PerfectGas.mo:229-231`` (function ``density``):
    ``d := state.p/(gasConstant(state)*state.T)``; ``:134-136`` (function ``gasConstant``):
    ``R_s := dryair.R*(1-X[Water]) + steam.R*X[Water]``; ``:562-572``: ``dryair.R``/``steam.R``
    are built from ``SingleGasesData.Air/H2O.R_s`` -- the identical formula
    ``_moist_air_gas_constant`` above already transcribes for ``density_pTX``. ``X_default``
    is ``reference_X = {0.01, 0.99}`` (``PerfectGas.mo:8``), unused-parameter-defaulted the
    same way as ``Buildings.Media.Air``.
    """
    r_at_x_default = _R_AIR * (1 - 0.01) + _R_H2O * 0.01
    rho_default = _P_DEFAULT / (r_at_x_default * _T_DEFAULT)

    def buoyancy(T, X_w):
        # "The medium's own ideal-gas law at p_default" (design resolution): identical
        # functional form to Buildings.Media.Air's density_pTX, since PerfectGas's gasConstant
        # uses the same two species records.
        T = torch.as_tensor(T)
        R = _moist_air_gas_constant(X_w).to(T.dtype)
        return _P_DEFAULT / (R * T)

    return MBLMedium(
        name="Buildings.Media.Specialized.Air.PerfectGas",
        p_default=_P_DEFAULT,
        T_default=_T_DEFAULT,
        X_default=(0.01, 0.99),
        rho_default=rho_default,
        has_moisture=True,
        _buoyancy=buoyancy,
    )


def _simpleair_medium() -> MBLMedium:
    """Modelica.Media.Air.SimpleAir (MSL Modelica/Media/Air/SimpleAir.mo).

    No moisture (single substance, ``nXi=0``). ``SimpleAir.mo:6-8``:
    ``MM_const=0.0289651159``, ``R_gas=Constants.R/MM_const`` (the CODATA ``Modelica.Constants.R``,
    not the NASA-2002 constant the two moist-air media above use).
    ``Modelica/Media/package.mo:6321-6323`` (``PartialSimpleIdealGasMedium``, function
    ``density``): ``d := state.p/(R_gas*state.T)``.
    """
    r_gas = _MODELICA_CONSTANTS_R / _MM_AIR
    rho_default = _P_DEFAULT / (r_gas * _T_DEFAULT)

    def buoyancy(T, X_w):
        # No moisture: the density is p_default/(R_gas*T) regardless of X_w.
        T = torch.as_tensor(T)
        return _P_DEFAULT / (r_gas * T)

    return MBLMedium(
        name="Modelica.Media.Air.SimpleAir",
        p_default=_P_DEFAULT,
        T_default=_T_DEFAULT,
        X_default=(),
        rho_default=rho_default,
        has_moisture=False,
        _buoyancy=buoyancy,
    )


_MEDIA: dict[str, Callable[[], MBLMedium]] = {
    "Buildings.Media.Air": _air_medium,
    "Buildings.Media.Specialized.Air.PerfectGas": _perfectgas_medium,
    "Modelica.Media.Air.SimpleAir": _simpleair_medium,
}


def medium(name: str) -> MBLMedium:
    """Look up one of the three MBL/MSL media this importer supports.

    Raises ``KeyError`` naming the three supported media if ``name`` is anything else.
    """
    try:
        factory = _MEDIA[name]
    except KeyError:
        supported = ", ".join(sorted(_MEDIA))
        raise KeyError(
            f"unsupported MBL medium {name!r}; supported media are: {supported}"
        ) from None
    return factory()
