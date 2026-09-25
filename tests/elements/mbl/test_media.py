"""Tests for `noodl.elements.mbl.media`: MBL/MSL media constants and densities.

Every reference value below is transcribed directly from the Modelica Buildings Library (MBL)
v13.0.0 (commit 55abf579598ca81cae0a82f337350375958e6722) and the Modelica Standard Library
(MSL) v4.1.0 source, never by importing `noodl.elements.mbl.media` itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from noodl.elements.mbl import media

# ---------------------------------------------------------------------------------------
# NumPy/Python transcription of the MBL/MSL source (independent of noodl.elements.mbl.media)
# ---------------------------------------------------------------------------------------

# Modelica Standard Library, Modelica/Constants.mo:47 (k, Boltzmann), :53 (N_A, Avogadro),
# :49 (R = k * N_A, the CODATA universal gas constant used by Modelica.Media.Air.SimpleAir).
_BOLTZMANN_K = 1.380649e-23
_AVOGADRO_N_A = 6.02214076e23
_MODELICA_CONSTANTS_R = _BOLTZMANN_K * _AVOGADRO_N_A

# MSL Modelica/Media/IdealGases/Common/SingleGasesData.mo:5 (R_NASA_2002, the NASA-2002
# universal gas constant used to build every species record's R_s = R_NASA_2002 / MM), :49
# (Air.MM), :9187 (H2O.MM); each species record sets R_s=R_NASA_2002/<species>.MM (:59, :9197).
_R_NASA_2002 = 8.314510
_MM_AIR = 0.0289651159
_MM_H2O = 0.01801528
_R_AIR = _R_NASA_2002 / _MM_AIR
_R_H2O = _R_NASA_2002 / _MM_H2O

# MSL Modelica/Media/package.mo:3969 (p_default=101325), :3971 (T_default = from_degC(20)):
# PartialMedium's generic defaults, inherited unchanged by Buildings.Media.Air (Air.mo:1-15),
# Buildings.Media.Specialized.Air.PerfectGas (PerfectGas.mo:1-14) and Modelica.Media.Air.SimpleAir
# (SimpleAir.mo:1-15, none of which override p_default/T_default).
_P_DEFAULT = 101325.0
_T_DEFAULT = 293.15


def _moist_air_gas_constant_np(X_w):
    """R(X_w) = R_air*(1 - X_w) + R_h2o*X_w.

    Buildings/Utilities/Psychrometrics/Functions/density_pTX.mo:11-14 (also
    Buildings/Media/Specialized/Air/PerfectGas.mo:134-136, function `gasConstant`).
    """
    return _R_AIR * (1 - X_w) + _R_H2O * X_w


# ---------------------------------------------------------------------------------------
# rho_default of each medium
# ---------------------------------------------------------------------------------------


def test_air_rho_default_and_state_defaults():
    """Buildings/Media/Air.mo:43-45 (pStp=reference_p=101325, dStp=1.2 kg/m3), :210-215
    (function density: `d := state.p*dStp/pStp`); BaseClasses/PartialOneWayFlowElement.mo:23-28
    (`sta_default` built at `Medium.p_default`/`Medium.T_default`/`Medium.X_default`,
    `rho_default = Medium.density(sta_default)`). Since Buildings.Media.Air's p_default equals
    pStp (both 101325 Pa, MSL Media/package.mo:3969), the ratio is exactly 1 and rho_default
    equals dStp exactly.
    """
    dStp = 1.2
    pStp = 101325.0
    expected_rho = _P_DEFAULT * dStp / pStp
    m = media.medium("Buildings.Media.Air")
    assert m.rho_default == pytest.approx(expected_rho, rel=1e-12)
    assert m.p_default == pytest.approx(_P_DEFAULT)
    assert m.T_default == pytest.approx(_T_DEFAULT)
    assert m.has_moisture is True
    assert m.X_default == pytest.approx((0.01, 0.99))


def test_perfectgas_rho_default():
    """Buildings/Media/Specialized/Air/PerfectGas.mo:8 (reference_X={0.01,0.99}, so
    X_default=reference_X per MSL Media/package.mo's PartialMedium default), :229-231
    (`redeclare function extends density`: `d := state.p/(gasConstant(state)*state.T)`),
    :134-136 (`gasConstant`: `R_s := dryair.R*(1-X[Water]) + steam.R*X[Water]`), :562-572
    (dryair.R/steam.R built from `SingleGasesData.Air/H2O.R_s`, identical formula to
    `density_pTX`'s R).
    """
    x_water_default = 0.01
    r = _moist_air_gas_constant_np(x_water_default)
    expected_rho = _P_DEFAULT / (r * _T_DEFAULT)
    m = media.medium("Buildings.Media.Specialized.Air.PerfectGas")
    assert m.rho_default == pytest.approx(expected_rho, rel=1e-12)
    assert m.has_moisture is True
    assert m.X_default == pytest.approx((0.01, 0.99))


def test_simpleair_rho_default():
    """MSL Modelica/Media/Air/SimpleAir.mo:6-8 (MM_const=0.0289651159,
    `R_gas=Constants.R/MM_const`); Modelica/Media/package.mo:6321-6323
    (`redeclare function extends density`: `d := state.p/(R_gas*state.T)`), :6192-6202
    (`PartialSimpleIdealGasMedium` -- no moisture, `nXi=0`, single substance).
    """
    r_gas = _MODELICA_CONSTANTS_R / _MM_AIR
    expected_rho = _P_DEFAULT / (r_gas * _T_DEFAULT)
    m = media.medium("Modelica.Media.Air.SimpleAir")
    assert m.rho_default == pytest.approx(expected_rho, rel=1e-12)
    assert m.has_moisture is False


def test_unknown_medium_raises_key_error_naming_the_three_supported():
    with pytest.raises(KeyError) as excinfo:
        media.medium("Not.A.Real.Medium")
    message = str(excinfo.value)
    for name in (
        "Buildings.Media.Air",
        "Buildings.Media.Specialized.Air.PerfectGas",
        "Modelica.Media.Air.SimpleAir",
    ):
        assert name in message


# ---------------------------------------------------------------------------------------
# buoyancy_density: Buildings.Media.Air against density_pTX.mo, on a T/X_w grid
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_kelvin", np.linspace(263.15, 323.15, 5).tolist())
@pytest.mark.parametrize("x_w", [0.0, 0.01, 0.02])
def test_air_buoyancy_density_matches_density_pTX(t_kelvin, x_w):
    """Buildings/Utilities/Psychrometrics/Functions/density_pTX.mo:5-16: `d := p/(R*T)`,
    evaluated at Buildings.Media.Air's *fixed* `p_default` (never the actual port pressure)
    with the *actual* local temperature and water mass fraction -- the buoyancy-relevant
    density, structurally different from `Medium.density` (pressure-only, tested above).
    """
    r = _moist_air_gas_constant_np(x_w)
    expected = _P_DEFAULT / (r * t_kelvin)
    m = media.medium("Buildings.Media.Air")
    got = m.buoyancy_density(
        torch.tensor(t_kelvin, dtype=torch.float64), torch.tensor(x_w, dtype=torch.float64)
    )
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("t_kelvin", [263.15, 293.15, 323.15])
@pytest.mark.parametrize("x_w", [0.0, 0.01, 0.02])
def test_perfectgas_buoyancy_density_matches_its_own_ideal_gas_law(t_kelvin, x_w):
    """PerfectGas.mo:229-231/:134-136: the medium's own ideal-gas density function,
    `d = p/(gasConstant(X)*T)`, evaluated at `p_default` per the design's resolution
    ("the medium's own ideal-gas law at p_default" for media other than Buildings.Media.Air).
    """
    r = _moist_air_gas_constant_np(x_w)
    expected = _P_DEFAULT / (r * t_kelvin)
    m = media.medium("Buildings.Media.Specialized.Air.PerfectGas")
    got = m.buoyancy_density(
        torch.tensor(t_kelvin, dtype=torch.float64), torch.tensor(x_w, dtype=torch.float64)
    )
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("t_kelvin", [263.15, 293.15, 323.15])
def test_simpleair_buoyancy_density_matches_its_own_ideal_gas_law(t_kelvin):
    """Modelica/Media/package.mo:6321-6323: `d = p/(R_gas*T)`, evaluated at `p_default`;
    SimpleAir has no moisture so `X_w` plays no role (has_moisture is False).
    """
    r_gas = _MODELICA_CONSTANTS_R / _MM_AIR
    expected = _P_DEFAULT / (r_gas * t_kelvin)
    m = media.medium("Modelica.Media.Air.SimpleAir")
    got = m.buoyancy_density(torch.tensor(t_kelvin, dtype=torch.float64), 0.0)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0.0
    )


# ---------------------------------------------------------------------------------------
# Medium.density and specificHeatCapacityCp (zone mass and heat capacity, zonal
# flow density). Transcribed from the three media's own functions.
# ---------------------------------------------------------------------------------------


def test_air_density_is_pressure_only():
    # Buildings/Media/Air.mo:210-215: d := state.p*dStp/pStp (dStp = 1.2, pStp = 101325).
    m = media.medium("Buildings.Media.Air")
    p = torch.tensor([101325.0, 101330.0], dtype=torch.float64)
    got = m.density(p, torch.tensor([280.0, 310.0], dtype=torch.float64), 0.02)
    np.testing.assert_allclose(got.numpy(), p.numpy() * 1.2 / 101325.0, rtol=1e-15)


def test_perfectgas_and_simpleair_density_are_ideal_gas():
    # PerfectGas.mo:229-231, :134-136; SimpleAir.mo:6-8 with package.mo:6321-6323.
    pg = media.medium("Buildings.Media.Specialized.Air.PerfectGas")
    got = pg.density(torch.tensor(101000.0, dtype=torch.float64), 300.0, 0.012)
    expected = 101000.0 / ((_R_AIR * (1 - 0.012) + _R_H2O * 0.012) * 300.0)
    assert float(got) == pytest.approx(expected, rel=1e-15)
    sa = media.medium("Modelica.Media.Air.SimpleAir")
    got = sa.density(torch.tensor(101000.0, dtype=torch.float64), 300.0, 0.5)
    assert float(got) == pytest.approx(101000.0 / (_MODELICA_CONSTANTS_R / _MM_AIR * 300.0),
                                       rel=1e-15)


def test_specific_heat_capacity_cp():
    # Air.mo:567-575 and PerfectGas.mo:382-387: cp = dryair.cp*(1 - X_w) + steam.cp*X_w with
    # Buildings/Utilities/Psychrometrics/Constants.mo:6,8 (cpAir = 1006, cpSte = 1860);
    # SimpleAir.mo:6 (cp_const = 1005.45).
    for name in ("Buildings.Media.Air", "Buildings.Media.Specialized.Air.PerfectGas"):
        assert media.medium(name).specific_heat_cp(0.01) == pytest.approx(
            1006 * 0.99 + 1860 * 0.01, rel=1e-15
        )
    assert media.medium("Modelica.Media.Air.SimpleAir").specific_heat_cp(0.3) == 1005.45
