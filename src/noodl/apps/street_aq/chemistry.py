"""Photostationary chemistry for a street model, and the steady state that includes it."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from noodl.apps.street_aq.closures import PRESETS, preset_options, warn_deprecated
from noodl.layers.reaction import (
    Photostationary,
    Reaction,
    k_no_o3_jpl2003,
    k_no_o3_munich,
    molar_volume,
)
from noodl.model import Drivers, Model, State

__all__ = [
    "CLOSURES",
    "J_NO2_CLEAR_SKY",
    "J_NO2_ZENITH_DEG",
    "NO_O3_RATES",
    "SIRANE_K3_ACTIVATION",
    "SIRANE_K3_PREFACTOR",
    "SIRANE_K_FLOOR_PPB",
    "j_no2",
    "j_no2_elevation_cloud",
    "j_no2_sirane",
    "j_no2_zenith_table",
    "k_no_o3_jpl2003",
    "k_no_o3_munich",
    "k_no_o3_sirane",
    "k_no_o3_soulhac2011",
    "molar_volume",
    "photostationary_for_streets",
    "solar_elevation",
    "street_steady",
]


CLOSURES = tuple(PRESETS)
"""The presets `photostationary_for_streets` offers (`"sirane"`, `"munich"`)."""


def photostationary_for_streets(
    species: Sequence[str],
    *,
    preset: str | None = None,
    no_o3_rate: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    floor_ppb: float | None = None,
    j_key: str = "J_NO2",
    temperature_key: str = "temperature",
    molar_volume_key: str = "molar_volume",
    closure: str | None = None,
) -> Photostationary:
    """The Leighton reaction wired to the `"no"`, `"no2"` and `"o3"` columns of `species`.

    Names are matched case-insensitively and exactly; a missing one is named rather than
    guessed at, because a silently mis-wired species column produces a plausible-looking
    answer that is simply wrong.

    `preset` (default `"sirane"`) sets the rate of NO + O3 and the floor on `J/k`
    (`closures.PRESETS[preset]["chemistry"]`), both evaluated at the driver
    `temperature_key` (K); `no_o3_rate` and `floor_ppb` override them:

    - `no_o3_rate="soulhac_2011"` (preset `"sirane"`): `k_no_o3_soulhac2011`,
      1.325e6 exp(-1430/T) m3 mol^-1 s^-1, with `K = max(J/k, 2 ppb)`
      (`SIRANE_K_FLOOR_PPB`). The ppb conversion uses the driver `molar_volume_key`
      (m3/mol) when given, else `molar_volume(T)` at 101325 Pa. SIRANE's `k1` is
      `j_no2_elevation_cloud`, passed as the `j_key` driver. Reproducing SIRANE exactly
      needs the molar-volume driver (SIRANE's ground-level V_m): the fallback uses T and
      101325 Pa.
    - `no_o3_rate="jpl_2003"` (preset `"munich"`): `k_no_o3_jpl2003`,
      3.0e-12 exp(-1500/T) cm3 molecule^-1 s^-1; no floor.
    - `no_o3_rate=<callable>`: any rate `k(T)` in m3 mol^-1 s^-1.

    `floor_ppb` overrides the preset's floor (0 means none). With any rate the
    equilibrium conserves molar NOx and Ox, so background NO, NO2 and O3 enter through
    those two totals and the NO2 share of the transported NOx plays the role of SIRANE's
    emitted NO2/NOx ratio. `closure=` is the deprecated spelling of `preset`.
    """
    if closure is not None:
        if preset is not None:
            raise TypeError(
                "photostationary_for_streets: give preset or its deprecated spelling "
                "closure, not both"
            )
        warn_deprecated(f"photostationary_for_streets: the keyword 'closure' is deprecated; "
                        f"use preset={closure!r}")
        preset = closure
    preset = "sirane" if preset is None else preset
    if preset not in PRESETS:
        raise ValueError(
            f"photostationary_for_streets: preset must be one of {CLOSURES}, got "
            f"{preset!r}"
        )
    settings = preset_options(preset)["chemistry"]
    chosen = settings["no_o3_rate"] if no_o3_rate is None else no_o3_rate
    if callable(chosen):
        rate = chosen
    elif chosen in NO_O3_RATES:
        rate = NO_O3_RATES[chosen]
    else:
        raise ValueError(
            f"photostationary_for_streets: no_o3_rate must be one of {tuple(NO_O3_RATES)} "
            f"or a callable k(T), got {chosen!r}"
        )
    default_floor = settings["floor_ppb"]
    lowered = [str(name).lower() for name in species]
    columns = []
    for wanted in ("no", "no2", "o3"):
        if wanted not in lowered:
            raise ValueError(
                f"photostationary_for_streets: species {wanted!r} is not among "
                f"{tuple(species)}; the Leighton state needs all of 'no', 'no2' and 'o3'"
            )
        columns.append(lowered.index(wanted))
    return Photostationary(
        columns[0], columns[1], columns[2], j_key=j_key, rate=rate,
        temperature_key=temperature_key, molar_volume_key=molar_volume_key,
        floor_ppb=default_floor if floor_ppb is None else floor_ppb,
    )


def street_steady(
    model: Model,
    state: State,
    drivers: Drivers,
    *,
    reaction: Reaction | None = None,
    tol: float = 1e-14,
    max_iter: int = 50,
    layer_name: str = "street",
    **solve_kwargs,
) -> State:
    """The steady state of transport AND chemistry, by an explicit fixed point.

    `Model.steady` does NOT apply reactions -- `Model._pass` applies them only on its
    stepping branch, and `Model.steady`'s own docstring records that as deliberate. This
    function is the steady state WITH chemistry. With `reaction=None` this is exactly
    `model.steady(...)`, returned unchanged.

    The iteration is plain successive substitution -- solve transport at the current
    composition, relax the composition to its photostationary state, repeat -- and it is
    differentiable by unrolling: every pass stays on the autograd graph, so memory grows
    with the pass count here (unlike `Model`'s own `coupling="iterate"`, which differentiates
    its converged fixed point implicitly -- see `noodl.solvers.fixed_point`; this loop is a
    separate mechanism and has not been given the same treatment). `tol` is an ABSOLUTE
    tolerance on the state in its own units (kg/m3), tested on the largest change over all
    streets and species; a budget it cannot meet raises, naming the pass count and the
    change that was left.

    For an instantaneous equilibrium that conserves molar NOx and Ox (`Photostationary`),
    and transport that treats NO, NO2 and O3 alike (the same flows, no species-dependent
    loss), the fixed point is the equilibrium applied street by street to the
    transport-only steady state: the reaction leaves the two totals unchanged, transport
    carries them as passive tracers, and the equilibrium depends on nothing else. So
    `reaction.apply(model.steady(...)["street.x"], None, drivers)` gives the same answer in
    one solve; it is also how the equilibrium is applied pointwise to any other
    passive-plus-background concentration (receptors, grids), as SIRANE does.
    """
    if reaction is None:
        return model.steady(state, drivers, **solve_kwargs)
    key = f"{layer_name}.x"
    current = dict(state)
    change = float("inf")
    # `_passes` rather than `passes`: the count is not read inside the body (ruff B007),
    # only the budget matters, and the failure message names `max_iter` itself.
    for _passes in range(int(max_iter)):
        solved = model.steady(current, drivers, **solve_kwargs)
        relaxed = dict(solved)
        relaxed[key] = reaction.apply(solved[key], None, drivers)
        with torch.no_grad():
            if key in current:
                change = float((relaxed[key] - current[key]).abs().max())
        current = relaxed
        if change <= tol:
            return current
    raise RuntimeError(
        f"street_steady: the transport-and-chemistry fixed point did not converge within "
        f"{max_iter} passes; the largest change on the last pass was {change} against an "
        f"absolute tolerance of {tol} (kg/m3)"
    )


J_NO2_ZENITH_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 78.0, 86.0, 90.0)
"""The solar zenith angles MUNICH's Leighton mechanism tabulates `J_NO2` at.
`include/modules/chemistry/Leighton/reactions`, `SET TABULATION 11 DEGREES ...`."""

J_NO2_CLEAR_SKY = (
    9.31026e-3, 9.21901e-3, 8.90995e-3, 8.37928e-3, 7.60031e-3, 6.52988e-3,
    5.10803e-3, 3.29332e-3, 1.74121e-3, 5.11393e-4, 1.63208e-4,
)
"""Clear-sky `J_NO2` [1/s] at those angles, from the same file's `KINETIC PHOTOLYSIS`
line (a RACM tabulation, per the comment there)."""


def j_no2_zenith_table(zenith_deg, attenuation=1.0) -> torch.Tensor:
    """Clear-sky `J_NO2` [1/s] at `zenith_deg`, times `attenuation`.

    Piecewise-linear in the zenith angle between MUNICH's eleven tabulated points, and
    held at the end values outside them -- which is the interpolation MUNICH's own
    `Photolysis_tabulation_option: 2` performs on the same table. `attenuation` is
    MUNICH's per-street `Attenuation` field, the only modulation its chemistry applies:
    `include/modules/chemistry/Common/chem.f:232-234` computes
    `J = Zatt * J_tabulated` and nothing else -- there is no separate canyon shading
    factor. The SOLAR GEOMETRY is deliberately not computed here: a zenith angle wants a
    date, a latitude and a longitude, none of which the street application carries, and
    chemistry is exercised on synthetic cases only. Pass the angle you want.
    """
    zenith = torch.as_tensor(zenith_deg, dtype=torch.float64)
    angles = torch.tensor(J_NO2_ZENITH_DEG, dtype=torch.float64)
    values = torch.tensor(J_NO2_CLEAR_SKY, dtype=torch.float64)
    clamped = torch.clamp(zenith, min=float(angles[0]), max=float(angles[-1]))
    upper = torch.clamp(torch.searchsorted(angles, clamped, right=True), 1,
                        len(angles) - 1)
    lower = upper - 1
    span = angles[upper] - angles[lower]
    weight = (clamped - angles[lower]) / span
    interpolated = values[lower] + weight * (values[upper] - values[lower])
    return interpolated * torch.as_tensor(attenuation, dtype=torch.float64)


j_no2 = j_no2_zenith_table
"""Deprecated name of `j_no2_zenith_table`."""


# ------------------------------------------------------------- SIRANE's chemistry closure
#
# Soulhac et al. (2011), Atmospheric Environment 45, 7379-7395, Eqs. 31-32, with the
# corrections that reproduce SIRANE v2.1 output: the k3 prefactor is 1.325e6 (the paper
# prints 1.325e5), k1 is clipped at zero, and the split works in ppb with a 2 ppb floor on
# k1/k3.

SIRANE_K3_PREFACTOR = 1.325e6
"""m3 mol^-1 s^-1: the prefactor of SIRANE's k(NO + O3), 2.2e-12 cm3 molecule^-1 s^-1 x N_A."""

SIRANE_K3_ACTIVATION = 1430.0
"""K: the activation temperature of SIRANE's k(NO + O3)."""

SIRANE_K_FLOOR_PPB = 2.0
"""ppb: SIRANE's lower bound on `k1/k3` in its photostationary split."""


def solar_elevation(latitude_deg, day_of_year, hour) -> torch.Tensor:
    """The solar elevation (degrees) at `latitude_deg`, day `day_of_year` (1 January = 1)
    and clock time `hour` (fractional hours), in SIRANE's convention.

    `sin a = sin(phi) sin(delta) + cos(phi) cos(delta) cos(omega)` with Cooper's
    declination `delta = 23.45 deg sin(360 deg (284 + n) / 365)` and the hour angle
    `omega = 15 deg (hour - 12)`: the clock time is taken as local solar time, with no
    longitude and no equation-of-time correction. All three arguments broadcast.
    """
    lat = torch.deg2rad(torch.as_tensor(latitude_deg, dtype=torch.float64))
    n = torch.as_tensor(day_of_year, dtype=torch.float64)
    h = torch.as_tensor(hour, dtype=torch.float64)
    declination = torch.deg2rad(23.45 * torch.sin(torch.deg2rad(360.0 * (284.0 + n) / 365.0)))
    omega = torch.deg2rad(15.0 * (h - 12.0))
    sin_a = (torch.sin(lat) * torch.sin(declination)
             + torch.cos(lat) * torch.cos(declination) * torch.cos(omega))
    return torch.rad2deg(torch.asin(torch.clamp(sin_a, -1.0, 1.0)))


def j_no2_elevation_cloud(elevation_deg, cloud_octas=0.0) -> torch.Tensor:
    """SIRANE's NO2 photolysis rate `k1` (1/s) at solar elevation `elevation_deg` under
    `cloud_octas` of cloud (0 to 8).

    `k1 = (1/60) max{0, 0.5699 - [9.056e-3 (90 - a)]^2.546} [1 - 0.75 (N/8)^3.4]`: Soulhac
    et al. (2011) Eq. 32 with the bracket clipped at zero, which makes `k1` zero for
    `a <= 1.458 deg` and so at night. The clear-sky overhead value is 9.50e-3 1/s. The
    clip is the only non-smooth point.
    """
    elevation = torch.as_tensor(elevation_deg, dtype=torch.float64)
    cloud = torch.as_tensor(cloud_octas, dtype=torch.float64)
    with torch.no_grad():
        if not bool(torch.isfinite(cloud).all()) or bool(((cloud < 0) | (cloud > 8)).any()):
            raise ValueError(
                f"j_no2_elevation_cloud: cloud_octas must lie in [0, 8]; got "
                f"{cloud.detach().flatten()[:4].tolist()}"
            )
    zenith = torch.clamp(90.0 - elevation, min=0.0)
    clear = torch.clamp(0.5699 - (9.056e-3 * zenith) ** 2.546, min=0.0) / 60.0
    return clear * (1.0 - 0.75 * (cloud / 8.0) ** 3.4)


j_no2_sirane = j_no2_elevation_cloud
"""Deprecated name of `j_no2_elevation_cloud`."""


def k_no_o3_soulhac2011(temperature) -> torch.Tensor:
    """k(NO + O3) = 1.325e6 exp(-1430/T) m3 mol^-1 s^-1 at `temperature` (K), SIRANE's
    rate (Soulhac et al. 2011, with the prefactor that reproduces SIRANE v2.1 output).

    In SIRANE the temperature is its ground-level air temperature. Divided by the molar
    volume in litres and times 1e-6 it is the rate in ppb^-1 s^-1 that SIRANE prints.
    """
    t = torch.as_tensor(temperature, dtype=torch.float64)
    return SIRANE_K3_PREFACTOR * torch.exp(-SIRANE_K3_ACTIVATION / t)


k_no_o3_sirane = k_no_o3_soulhac2011
"""Deprecated name of `k_no_o3_soulhac2011`."""

NO_O3_RATES = {"soulhac_2011": k_no_o3_soulhac2011, "jpl_2003": k_no_o3_jpl2003}
"""The named NO + O3 rates `photostationary_for_streets(no_o3_rate=...)` takes."""
