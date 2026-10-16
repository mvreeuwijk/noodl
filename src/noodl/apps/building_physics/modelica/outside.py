"""The outside air at an MBL `Outside` boundary: pressure, temperature and humidity from the
weather bus, and the wind pressure added to the pressure.

Transcribed from MBL v13.0.0 (commit 55abf579) source, never from its documentation:

- `Fluid/Sources/BaseClasses/Outside.mo`: the port is at `p_in_internal`, temperature
  `weaBus.TDryBul` (:46), composition from `X_pTphi(p_in_internal, T, weaBus.relHum)`
  (:49-53), or `X = 1` for a single-substance medium (:54-56).
- `Outside.mo:7`: `p_in_internal = weaBus.pAtm` (`outside_air`).
- `Outside_Cp.mo:32-37`: `pWin = 0.5 Cp v^2 Medium.density(setState_pTX(pWea, T, X_default))`,
  `p = pWea + pWin`: the density at the ATMOSPHERIC pressure and the DEFAULT composition
  (`wind_exposed_air(..., density_at_boundary=False)`).
- `Outside_CpLowRise.mo:13-23, 40-44`: `alpha = winDir - (azi - pi)`, `Cp =
  windPressureLowRise(Cp0, alpha, log(s))`, `pWin = 0.5 Cp d v^2` with `d =
  Medium.density(setState_pTX(p, T, X))` at the boundary's OWN pressure `p = pWea + pWin` and
  composition: an implicit equation, solved by fixed-point iteration
  (`wind_exposed_air(..., density_at_boundary=True)`).
- `Outside_CpData.mo:21, 31, 36, 68-70`: the same with `Cp = windPressureProfile(alpha, ...)` and
  `pWin = Cs 0.5 Cp d v^2`.

Angles: `winDir` is the direction the wind comes FROM, N = 0, E = pi/2 (the TMY3 file,
`ReaderTMY3`); a surface's azimuth `azi` is the direction its outside faces, S = 0, W = pi/2
(`Buildings.Types.Azimuth`); `alpha = 0` is wind normal onto the surface. These are not
CONTAM's, and `noodl.drives.Wind`/`WindProfile` are not reused: there the wind pressure is a
term of an envelope PATH's pressure difference and Cp a piecewise-linear profile, while MBL adds
it to the boundary NODE's pressure and interpolates its profile with a cubic Hermite spline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from noodl.elements import AirMedium

Tensor = torch.Tensor
F64 = torch.float64
PI = math.pi
_K_WATER_AIR = 0.621964713077499  # Utilities/Psychrometrics/Functions/X_pSatpphi.mo:13
# windPressureLowRise.mo:12-17: transitions over 5 degrees at either end of 0..pi.
_A_DEL = 5 * PI / 180
_A_DEL2 = _A_DEL / 2
_A_MAX = 175 * PI / 180
# The implicit density of `wind_exposed_air`: d pWin/dp = pWin/p is ~1e-4 for wind speeds
# below 40 m/s, so each pass gains about four digits; 4 eps of the pressure is round-off.
FIXED_POINT_RTOL = 4 * torch.finfo(F64).eps
FIXED_POINT_MAX = 30


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=F64)


def reg_step(x, y1, y2, x_small: float) -> Tensor:
    """`Utilities/Math/Functions/regStep.mo:12-14`: `y1` for `x > x_small`, `y2` for
    `x < -x_small`, and between them the cubic `s (s^2 - 3) (y2 - y1)/4 + (y1 + y2)/2`,
    `s = x/x_small`, whose slope vanishes at both ends."""
    x, y1, y2 = _f64(x), _f64(y1), _f64(y2)
    if x_small > 0:
        s = x / x_small
        mid = s * (s ** 2 - 3) * (y2 - y1) / 4 + (y1 + y2) / 2
    else:
        mid = (y1 + y2) / 2 + torch.zeros_like(x)
    return torch.where(x > x_small, y1, torch.where(x < -x_small, y2, mid))


def incidence_angle(wind_direction, surface_azimuth: float) -> Tensor:
    """Wind incidence angle on a surface (rad, 0 = normal onto it): `alpha = winDir - surOut`,
    `surOut = azi - pi` (`Outside_CpLowRise.mo:13, 40`; `Outside_CpData.mo:31, 36`).
    `wind_direction` (rad) is where the wind comes from, N = 0, E = pi/2; `surface_azimuth`
    (rad) where the surface faces, S = 0, W = pi/2."""
    return _f64(wind_direction) - (float(surface_azimuth) - PI)


def _swami_chandra(aR: Tensor, G: float) -> Tensor:
    sin_half, cos_half = torch.sin(aR / 2), torch.cos(aR / 2)
    return torch.log(1.248 - 0.703 * sin_half - 1.175 * torch.sin(aR) ** 2
                     + 0.131 * torch.sin(2 * aR * G) ** 3 + 0.769 * cos_half
                     + 0.071 * G ** 2 * sin_half ** 2 + 0.717 * cos_half ** 2)


def low_rise_pressure_coefficient(incidence, cp_normal: float, side_ratio: float) -> Tensor:
    """Wind pressure coefficient (1) on a wall of a low-rise rectangular building, Swami and
    Chandra (1987) as MBL implements it (`Airflow/Multizone/BaseClasses/windPressureLowRise.mo:
    19-67`). `incidence` (rad) is the wind incidence angle (`incidence_angle`), `cp_normal` the
    coefficient for wind normal to the wall (MBL's `Cp0`), `side_ratio` the length of this wall
    over the adjacent one (MBL's `s`; `G = log s`). Even and 2 pi periodic in the incidence; the
    correlation is blended by `reg_step` into `cp_normal` below 5 degrees and into its value at
    180 degrees above 175, so that it is once continuously differentiable."""
    if not side_ratio > 0:
        raise ValueError(f"side ratio s must be positive, got {side_ratio!r}")
    a = _f64(incidence)
    G = math.log(side_ratio)
    pi2 = 2 * PI
    aR = torch.where(a < 0, -a, a)                                       # :32
    aR = torch.where(aR > pi2, aR - torch.floor(aR / pi2) * pi2, aR)    # :34-36
    aR = torch.where(aR > PI, pi2 - aR, aR)                              # :38-40
    a180 = math.log(1.248 - 0.703 + 0.131 * math.sin(2 * PI * G) ** 3
                    + 0.071 * G ** 2)                                    # :19-21
    body = _swami_chandra(aR, G)                                         # :65-66
    low = reg_step(aR - _A_DEL2, body, 1.0, _A_DEL2)                     # :48-55
    high = reg_step(aR + _A_DEL2 - PI, a180, body, _A_DEL2)              # :56-64
    return cp_normal * torch.where(aR < _A_DEL, low, torch.where(aR > _A_MAX, high, body))


def _secant_mean_derivatives(x: Tensor, y: Tensor) -> Tensor:
    """`splineDerivatives(x, y, ensureMonotonicity=false)` for three or more points
    (`Utilities/Math/Functions/splineDerivatives.mo:40-51`): one-sided at the ends, the mean
    of the two adjacent secants inside."""
    delta = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d = torch.empty_like(y)
    d[0], d[-1] = delta[0], delta[-1]
    d[1:-1] = (delta[:-1] + delta[1:]) / 2
    return d


def _hermite_linear(x, x1, x2, y1, y2, y1d, y2d) -> Tensor:
    """`cubicHermiteLinearExtrapolation.mo` with MSL `Fluid.Utilities.cubicHermite`
    (`Modelica/Fluid/Utilities.mo:807-823`)."""
    h = x2 - x1
    t = (x - x1) / h
    aux3, aux2 = t ** 3, t ** 2
    h00 = 2 * aux3 - 3 * aux2 + 1
    h10 = aux3 - 2 * aux2 + t
    h01 = -2 * aux3 + 3 * aux2
    h11 = aux3 - aux2
    inside = y1 * h00 + h * y1d * h10 + y2 * h01 + h * y2d * h11
    return torch.where((x > x1) & (x < x2), inside,
                       torch.where(x <= x1, y1 + (x - x1) * y1d, y2 + (x - x2) * y2d))


class PressureCoefficientProfile:
    """Wind pressure coefficient (1) against the wind incidence angle (rad), periodic: MBL's
    `Outside_CpData` table (`Outside_CpData.mo:39-55`) evaluated by `windPressureProfile.mo:
    17-38`.

    `incidence` (rad) starts at 0 and stays below 2 pi; `cp` holds the coefficients there. The
    table is extended by one point before it and two after it, `x = {inc[n-1] - 2 pi, inc,
    2 pi + inc[1], 2 pi + inc[2]}`, `y = {Cp[n-1], Cp, Cp[1], Cp[2]}` (1-based, as MBL writes
    it: the point before 0 is the SECOND-TO-LAST one, not the last, so the slope at 0 and at
    2 pi may differ, exactly as in MBL), with `splineDerivatives(ensureMonotonicity=false)`
    slopes; the coefficient at `alpha` is the cubic Hermite (linear outside, never reached) on
    the interval holding `mod(alpha, 2 pi)`. Raises `ValueError` for a table MBL's own asserts
    reject (`Outside_CpData.mo:73-82`, `splineDerivatives.mo:19-24`)."""

    def __init__(self, incidence: Sequence[float], cp: Sequence[float]) -> None:
        inc = [float(a) for a in incidence]
        c = [float(v) for v in cp]
        n = len(inc)
        if n != len(c):
            raise ValueError(f"incidence angles ({n}) and Cp values ({len(c)}) differ in "
                             f"length")
        if n < 2:
            raise ValueError("at least two incidence angles are needed (the table is "
                             "extended from its second-to-last point)")
        if not abs(inc[0]) < 1e-4:
            raise ValueError(f"the first incidence angle must be 0, got {inc[0]!r} rad")
        if not 2 * PI - inc[-1] > 1e-4:
            raise ValueError(f"the last incidence angle must be below 2 pi by more than "
                             f"1e-4 rad, got {inc[-1]!r} rad")
        x = [inc[n - 2] - 2 * PI, *inc, 2 * PI + inc[0], 2 * PI + inc[1]]
        y = [c[n - 2], *c, c[0], c[1]]
        if any(b <= a for a, b in zip(x, x[1:], strict=False)):
            raise ValueError("incidence angles must be strictly increasing")
        self.x = torch.tensor(x, dtype=F64)
        self.y = torch.tensor(y, dtype=F64)
        self.d = _secant_mean_derivatives(self.x, self.y)

    def __call__(self, incidence) -> Tensor:
        aR = torch.remainder(_f64(incidence), 2 * PI)                         # :19
        m = self.x.numel()
        # :21-26: the last j in 1..m-1 with aR > x[j] (1 if none); 0-based here.
        i = ((aR.unsqueeze(-1) > self.x[:m - 1]).sum(-1) - 1).clamp(min=0)
        return _hermite_linear(aR, self.x[i], self.x[i + 1], self.y[i], self.y[i + 1],
                               self.d[i], self.d[i + 1])


def saturation_pressure(T) -> Tensor:
    """Saturation pressure of water (Pa) at temperature `T` (K): over liquid water above the
    triple point and over ice below it, blended over +-1 K (`Utilities/Psychrometrics/
    Functions/saturationPressure.mo:11-15` of `saturationPressureLiquid.mo:10` and
    `sublimationPressureIce.mo:12-16`; `Buildings.Media.Air.saturationPressure`, `Air.mo:
    406-414`, returns it)."""
    T = _f64(T)
    liquid = 611.657 * torch.exp(17.2799 - 4102.99 / (T - 35.719))
    a1, a2, n1, n2 = -13.9281690, 34.7078238, -1.5, -1.25
    r1 = T / 273.16
    ice = torch.exp(a1 - a1 * r1 ** n1 + a2 - a2 * r1 ** n2) * 611.657
    return reg_step(T - 273.16, liquid, ice, 1.0)


def water_mass_fraction(p, T, relative_humidity) -> Tensor:
    """Water vapour mass fraction (kg/kg of moist air) at pressure `p` (Pa), temperature `T`
    (K) and relative humidity `relative_humidity` (1): `X = phi k / (k phi + p/pSat - phi)`,
    `k = 0.621964713077499` (`Utilities/Psychrometrics/X_pTphi.mo:36-42`,
    `Functions/X_pSatpphi.mo:13-15`)."""
    phi = _f64(relative_humidity)
    return phi * _K_WATER_AIR / (_K_WATER_AIR * phi + _f64(p) / saturation_pressure(T) - phi)


@dataclass(frozen=True)
class OutsideAir:
    """The state an `Outside` boundary imposes: absolute pressure `p` (Pa), temperature `T`
    (K), water mass fraction `X_w` (kg/kg; 0 for a medium without water) and the wind pressure
    `wind_pressure` (Pa, MBL's `pWin`; `None` for a boundary without wind)."""

    p: Tensor
    T: Tensor
    X_w: Tensor
    wind_pressure: Tensor | None


def _water(p: Tensor, T: Tensor, relative_humidity, med: AirMedium) -> Tensor:
    if not med.has_moisture:
        return torch.zeros_like(p * T)  # single substance: X = 1, no water
    if relative_humidity is None:
        raise ValueError("a medium with water needs the relative humidity (weaBus.relHum)")
    return water_mass_fraction(p, T, relative_humidity)


def outside_air(p_atm, T, relative_humidity, med: AirMedium) -> OutsideAir:
    """`Outside`: the atmospheric pressure `p_atm` (Pa), the dry-bulb temperature `T` (K), the
    water mass fraction of `relative_humidity` (1; `None` for a medium without water)."""
    p_atm, T = _f64(p_atm), _f64(T)
    return OutsideAir(p=p_atm, T=T, X_w=_water(p_atm, T, relative_humidity, med),
                      wind_pressure=None)


def wind_exposed_air(p_atm, T, relative_humidity, med: AirMedium, *, pressure_coefficient,
                     wind_speed, density_at_boundary: bool,
                     speed_modifier: float = 1.0) -> OutsideAir:
    """An `Outside` boundary with wind (module docstring): pressure `p_atm + pWin` (Pa) with
    `pWin = speed_modifier 0.5 Cp d v^2`, `Cp` = `pressure_coefficient` (1), `v` =
    `wind_speed` (m/s). `density_at_boundary=False` takes `d` at `p_atm` and the medium's
    default composition (`Outside_Cp`); `True` at the boundary's own pressure and water mass
    fraction (`Outside_CpLowRise`, `Outside_CpData`), solved by fixed-point iteration.
    `speed_modifier` is `Outside_CpData`'s `Cs` (1), which multiplies the pressure."""
    p_atm, T = _f64(p_atm), _f64(T)
    cp, v = _f64(pressure_coefficient), _f64(wind_speed)
    if not density_at_boundary:
        X0 = med.X_default[0] if med.has_moisture else 0.0
        p_win = 0.5 * cp * v * v * med.density(p_atm, T, X0)  # Outside_Cp.mo:32-35
        p = p_atm + p_win
        return OutsideAir(p=p, T=T, X_w=_water(p, T, relative_humidity, med),
                          wind_pressure=p_win)
    p = p_atm
    for _ in range(FIXED_POINT_MAX):
        d = med.density(p, T, _water(p, T, relative_humidity, med))
        p_win = speed_modifier * 0.5 * cp * d * v * v  # Outside_CpData.mo:21
        p_next = p_atm + p_win
        done = bool(((p_next - p).abs() <= FIXED_POINT_RTOL * p_atm.abs()).all())
        p = p_next
        if done:
            break
    else:
        raise ValueError(f"the wind pressure did not converge in {FIXED_POINT_MAX} passes")
    return OutsideAir(p=p, T=T, X_w=_water(p, T, relative_humidity, med), wind_pressure=p_win)
