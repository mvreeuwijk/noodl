"""`noodl.apps.building_physics.modelica.outside`: the outside air of MBL's `Outside`
boundaries. Reference values are written out from the MBL source formulas (cited)."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.building_physics.modelica import outside
from noodl.elements import medium

F64 = torch.float64
PI = math.pi
AIR = medium("Buildings.Media.Air")
K_WATER_AIR = 0.621964713077499  # X_pSatpphi.mo:13


def _t(*v):
    return torch.tensor(v if len(v) > 1 else v[0], dtype=F64)


# --------------------------------------------------------------------------- regStep
def test_reg_step_joins_its_two_values_with_a_flat_cubic():
    y = outside.reg_step(_t(-2.0, -1.0, 0.0, 1.0, 2.0), 3.0, 1.0, 1.0)
    assert y.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
    # regStep.mo:12-14 between: (x/xs)((x/xs)^2 - 3)(y2 - y1)/4 + (y1 + y2)/2
    assert float(outside.reg_step(_t(0.5), 3.0, 1.0, 1.0)) == 0.5 * (0.25 - 3) * (1 - 3) / 4 + 2


# ---------------------------------------------------------------- low-rise coefficient
def _swami_chandra(a, G):  # windPressureLowRise.mo:65-66
    s2, c2 = math.sin(a / 2), math.cos(a / 2)
    return math.log(1.248 - 0.703 * s2 - 1.175 * math.sin(a) ** 2
                    + 0.131 * math.sin(2 * a * G) ** 3 + 0.769 * c2
                    + 0.071 * G ** 2 * s2 ** 2 + 0.717 * c2 ** 2)


def test_low_rise_coefficient_between_the_transitions_is_swami_and_chandra():
    G = math.log(5.0)
    for deg in (10.0, 45.0, 90.0, 135.0, 170.0):
        a = math.radians(deg)
        got = float(outside.low_rise_pressure_coefficient(_t(a), 0.6, 5.0))
        assert got == pytest.approx(0.6 * _swami_chandra(a, G), rel=1e-14, abs=0.0)


def test_low_rise_coefficient_is_cp0_normal_to_the_wall_and_a180_behind_it():
    G = math.log(0.2)
    a180 = math.log(1.248 - 0.703 + 0.131 * math.sin(2 * PI * G) ** 3 + 0.071 * G ** 2)
    assert float(outside.low_rise_pressure_coefficient(_t(0.0), 0.6, 0.2)) == pytest.approx(
        0.6, rel=1e-15, abs=0.0)
    assert float(outside.low_rise_pressure_coefficient(_t(PI), 0.6, 0.2)) == pytest.approx(
        0.6 * a180, rel=1e-14, abs=0.0)


def test_low_rise_coefficient_is_even_and_periodic_in_the_incidence():
    a = _t(0.3, 1.0, 2.5, 3.0, 6.0)
    cp = outside.low_rise_pressure_coefficient(a, 0.6, 5.0)
    for other in (-a, a + 4 * PI, 2 * PI - a):
        torch.testing.assert_close(outside.low_rise_pressure_coefficient(other, 0.6, 5.0), cp,
                                   rtol=1e-12, atol=1e-12)


def test_low_rise_coefficient_is_continuous_at_its_transitions():
    for edge in (5.0, 175.0):
        a = math.radians(edge)
        lo, hi = (float(outside.low_rise_pressure_coefficient(_t(a + s), 0.6, 5.0))
                  for s in (-1e-9, 1e-9))
        assert abs(hi - lo) < 1e-8


def test_low_rise_side_ratio_must_be_positive():
    with pytest.raises(ValueError, match="side ratio"):
        outside.low_rise_pressure_coefficient(_t(0.0), 0.6, 0.0)


def test_incidence_angle_follows_mbl_conventions():
    """A west wind (winDir = 3 pi/2: it comes FROM the west) hits a west-facing wall (azi =
    pi/2, Buildings.Types.Azimuth.W) normally: alpha = winDir - (azi - pi) = 2 pi, i.e. 0."""
    assert float(outside.incidence_angle(_t(1.5 * PI), PI / 2)) == pytest.approx(2 * PI)


# ------------------------------------------------------------------ Cp profile
INC = [math.radians(d) for d in (0, 45, 90, 135, 180, 225, 270, 315)]
CP = [0.4, 0.1, -0.3, -0.35, -0.2, -0.6, -0.9, -0.1]  # Outside_CpData_Angles.mo, CpAsy


def test_profile_passes_through_its_table():
    prof = outside.PressureCoefficientProfile(INC, CP)
    torch.testing.assert_close(prof(torch.tensor(INC, dtype=F64)),
                               torch.tensor(CP, dtype=F64), rtol=0, atol=1e-15)


def test_profile_is_periodic():
    prof = outside.PressureCoefficientProfile(INC, CP)
    a = _t(0.1, 2.0, 5.0)
    torch.testing.assert_close(prof(a - 2 * PI), prof(a), rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(prof(a + 2 * PI), prof(a), rtol=1e-12, atol=1e-15)


def test_profile_slope_at_zero_uses_the_second_to_last_point_as_mbl_does():
    """Outside_CpData.mo:44-56 extends the table with (inc[n-1] - 2 pi, Cp[n-1]) (1-based: the
    SECOND-TO-LAST point) before 0; splineDerivatives(ensureMonotonicity=false) averages the two
    secants at 0 (splineDerivatives.mo:40-51)."""
    prof = outside.PressureCoefficientProfile(INC, CP)
    left = (CP[0] - CP[-2]) / (0.0 - (INC[-2] - 2 * PI))
    right = (CP[1] - CP[0]) / INC[1]
    eps = 1e-7
    slope = (float(prof(_t(eps))) - CP[0]) / eps
    assert slope == pytest.approx((left + right) / 2, rel=1e-5)


@pytest.mark.parametrize(("inc", "cp", "match"), [
    ([0.0, 1.0], [0.1], "length"),
    ([0.0], [0.1], "two"),
    ([0.1, 1.0], [0.1, 0.2], "first"),
    ([0.0, 2 * PI - 1e-5], [0.1, 0.2], "below 2 pi"),
    ([0.0, 2.0, 1.0], [0.1, 0.2, 0.3], "increasing"),
])
def test_profile_refuses_what_mbl_asserts(inc, cp, match):
    with pytest.raises(ValueError, match=match):
        outside.PressureCoefficientProfile(inc, cp)


# ---------------------------------------------------------------- psychrometrics
def _liquid(T):  # saturationPressureLiquid.mo:10
    return 611.657 * math.exp(17.2799 - 4102.99 / (T - 35.719))


def _ice(T):  # sublimationPressureIce.mo:12-16
    r = T / 273.16
    return math.exp(-13.9281690 - (-13.9281690) * r ** -1.5
                    + 34.7078238 - 34.7078238 * r ** -1.25) * 611.657


def test_saturation_pressure_is_over_liquid_above_and_ice_below_the_triple_point():
    assert float(outside.saturation_pressure(_t(293.15))) == pytest.approx(_liquid(293.15),
                                                                           rel=1e-14, abs=0.0)
    assert float(outside.saturation_pressure(_t(250.0))) == pytest.approx(_ice(250.0),
                                                                          rel=1e-14, abs=0.0)
    mid = (_liquid(273.16) + _ice(273.16)) / 2  # regStep at x = 0
    assert float(outside.saturation_pressure(_t(273.16))) == pytest.approx(mid, rel=1e-14, abs=0.0)


def test_water_mass_fraction_from_relative_humidity():
    p, T, phi = 101325.0, 293.15, 0.5
    ps = _liquid(T)
    expected = phi * K_WATER_AIR / (K_WATER_AIR * phi + p / ps - phi)  # X_pSatpphi.mo:15
    assert float(outside.water_mass_fraction(_t(p), _t(T), _t(phi))) == pytest.approx(
        expected, rel=1e-14, abs=0.0)


# ------------------------------------------------------------------- outside air
def test_still_outside_air_sits_at_the_atmospheric_pressure():
    p, T, phi = _t(101000.0), _t(283.15), _t(0.8)
    air = outside.outside_air(p, T, phi, AIR)
    assert air.wind_pressure is None and float(air.p) == 101000.0
    assert float(air.X_w) == float(outside.water_mass_fraction(p, T, phi))


def test_dry_air_needs_no_humidity():
    air = outside.outside_air(_t(101000.0), _t(283.15), None,
                              medium("Modelica.Media.Air.SimpleAir"))
    assert float(air.X_w) == 0.0


def test_moist_air_without_humidity_is_an_error():
    with pytest.raises(ValueError, match="relative humidity"):
        outside.outside_air(_t(101000.0), _t(283.15), None, AIR)


def test_wind_pressure_at_the_default_composition_and_atmospheric_pressure():
    """Outside_Cp.mo:32-37: pWin = 0.5 Cp v^2 d(pWea, T, X_default); Buildings.Media.Air's
    d = p dStp/pStp (Air.mo:215)."""
    p, T, phi, v = _t(101000.0), _t(283.15), _t(0.5), _t(6.0)
    air = outside.wind_exposed_air(p, T, phi, AIR, pressure_coefficient=_t(0.6), wind_speed=v,
                                   density_at_boundary=False)
    expected = 0.5 * 0.6 * 6.0 * 6.0 * (101000.0 * 1.2 / 101325.0)
    assert float(air.wind_pressure) == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert float(air.p) == pytest.approx(101000.0 + expected, rel=1e-15, abs=0.0)
    assert float(air.X_w) == pytest.approx(float(outside.water_mass_fraction(air.p, T, phi)),
                                           rel=1e-15, abs=0.0)


def test_wind_pressure_at_the_boundary_pressure_solves_its_implicit_density():
    """Outside_CpLowRise.mo:21-45 / Outside_CpData.mo:21, 68-69: d at the boundary's own
    pressure p = pAtm + Cs 0.5 Cp d(p) v^2; with d = p dStp/pStp,
    p = pAtm / (1 - Cs 0.5 Cp v^2 dStp/pStp)."""
    p, T, phi, v, cp, cs = 101000.0, 283.15, 0.5, 12.0, -0.45, 0.8
    air = outside.wind_exposed_air(_t(p), _t(T), _t(phi), AIR, pressure_coefficient=_t(cp),
                                   wind_speed=_t(v), density_at_boundary=True,
                                   speed_modifier=cs)
    closed = p / (1 - cs * 0.5 * cp * v * v * 1.2 / 101325.0)
    assert float(air.p) == pytest.approx(closed, rel=1e-14, abs=0.0)
    assert float(air.wind_pressure) == pytest.approx(closed - p, rel=1e-9)


def test_wind_pressure_iteration_for_a_temperature_dependent_density():
    gas = medium("Buildings.Media.Specialized.Air.PerfectGas")
    p, T, phi, v, cp = 101000.0, 283.15, 0.5, 15.0, 0.6
    air = outside.wind_exposed_air(_t(p), _t(T), _t(phi), gas, pressure_coefficient=_t(cp),
                                   wind_speed=_t(v), density_at_boundary=True)
    d = float(gas.density(air.p, T, air.X_w))
    assert float(air.p) == pytest.approx(p + 0.5 * cp * d * v * v, rel=1e-14, abs=0.0)


def test_calm_air_has_no_wind_pressure():
    air = outside.wind_exposed_air(_t(101000.0), _t(283.15), _t(0.5), AIR,
                                   pressure_coefficient=_t(0.6), wind_speed=_t(0.0),
                                   density_at_boundary=True)
    assert float(air.wind_pressure) == 0.0 and float(air.p) == 101000.0
