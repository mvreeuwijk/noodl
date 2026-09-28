"""The closure options of the street air-quality application, named for their physics.

Every option value says what it computes (`bessel_profile`, `turbulent_velocity`,
`non_crossing_streamlines`, ...), not which model first used it. The two reference models'
full closure sets are the presets `"sirane"` (the default everywhere) and `"munich"`.

- `OPTIONS`: option name -> its allowed values.
- `PRESETS`: preset name -> every option, the von Karman constant, the floors and the
  chemistry settings `photostationary_for_streets` takes.
- `LEGACY`: the earlier option names and values, which are still accepted with a
  `DeprecationWarning` naming the replacement.

`resolve` merges a preset with explicit choices and is what `build_model` and
`StreetFlows` call; `normalise` checks one value and is what the lower-level functions
(`canyon_velocity`, `routing_matrix`, `direction_offsets`, ...) call.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping

__all__ = [
    "CLOSURE_OPTIONS",
    "LEGACY",
    "LEGACY_NAMES",
    "LEGACY_VALUES",
    "OPTIONS",
    "PRESETS",
    "SIGMA_V_MIN",
    "SIGMA_W_MIN",
    "infer_kappa",
    "normalise",
    "preset_options",
    "resolve",
    "warn_deprecated",
]

OPTIONS: dict[str, tuple[str, ...]] = {
    "canyon_wind": ("bessel_profile", "exponential_profile"),
    "roof_wind": ("bessel_canyon_mean", "canopy_log_law"),
    "roof_exchange": ("turbulent_velocity", "aspect_ratio_scaled"),
    "junction_routing": ("non_crossing_streamlines", "perfect_mixing"),
    "direction_averaging": ("none", "exact_gaussian", "rectangle_rule", "gauss_hermite"),
    "direction_spread": ("driver", "turbulence_intensity"),
    "stability": ("neutral", "monin_obukhov"),
}
"""Every string-valued closure option and its allowed values.

- `canyon_wind`: the along-canyon velocity. `bessel_profile` is the Soulhac, Perkins and
  Salizzoni (2008) closed form on the friction velocity; `exponential_profile` is the
  single-regime exponential profile on the roof-level wind (Kim et al. 2022, Eq. B14).
- `roof_wind`: the roof-level wind `u_H` the exponential profile needs.
  `bessel_canyon_mean` is the canyon mean of the Bessel profile (Kim et al. 2022, Eq. B12);
  `canopy_log_law` is a log law with Macdonald's (1998) displacement height and roughness.
- `roof_exchange`: the roof exchange velocity. `turbulent_velocity` is
  `u_d = sigma_w / (sqrt(2) pi)`; `aspect_ratio_scaled` is Schulte's
  `u_d = beta sigma_w / (1 + H/W)`.
- `junction_routing`: how a junction's inflows are shared among its outflows.
  `non_crossing_streamlines` fills the outflows in angular order without crossing;
  `perfect_mixing` shares every inflow in proportion to the outflows.
- `direction_averaging`: the average of the junction routing over the wind-direction
  spread. `none` routes at the mean direction; `exact_gaussian` is the exact Gaussian
  average of the piecewise-constant routing; `rectangle_rule` is uniform sampling on
  `[-2 sigma, 2 sigma]` with unnormalised weights; `gauss_hermite` is an `n_theta`-point
  Gauss-Hermite quadrature.
- `direction_spread`: where the spread `sigma_theta` comes from. `driver` takes the
  `sigma_theta` driver, else the constructor value, else 0; `turbulence_intensity` is
  `sigma_theta = min(sigma_v / U_ref, 10 deg)`.
- `stability`: the turbulence profiles. `neutral` is `sigma_w = 1.3 u* (1 - 0.8 z/h)`;
  `monin_obukhov` switches between unstable, neutral and stable branches on the Obukhov
  length, and is the neutral form wherever no `lmo` driver is given.
"""

CLOSURE_OPTIONS = tuple(OPTIONS)
"""The string-valued option names, in `OPTIONS` order."""

SIGMA_W_MIN = 0.3
"""SIRANE keyword `SIGMA_W_MIN` ("Minimum sigma_w", default 0.3 m/s): the sirane preset's
`sigma_w_min`, and the above-roof plume's default floor."""

SIGMA_V_MIN = 0.5
"""SIRANE keyword `SIGMA_V_MIN` ("Minimum sigma_v", default 0.5 m/s): the sirane preset's
`sigma_v_min`, and the above-roof plume's default floor."""

_NUMERIC = ("kappa", "canyon_wind_min", "u_d_min", "sigma_w_min", "sigma_v_min")
"""The numeric settings a preset fixes besides the string options."""

PRESETS: dict[str, dict] = {
    "sirane": {
        "canyon_wind": "bessel_profile",
        "roof_wind": "bessel_canyon_mean",
        "roof_exchange": "turbulent_velocity",
        "junction_routing": "non_crossing_streamlines",
        "direction_averaging": "exact_gaussian",
        "direction_spread": "driver",
        "stability": "monin_obukhov",
        "kappa": 0.40,
        "canyon_wind_min": 0.0,
        "u_d_min": 0.0,
        "sigma_w_min": SIGMA_W_MIN,
        "sigma_v_min": SIGMA_V_MIN,
        "chemistry": {"no_o3_rate": "soulhac_2011", "floor_ppb": 2.0},
    },
    "munich": {
        "canyon_wind": "exponential_profile",
        "roof_wind": "bessel_canyon_mean",
        "roof_exchange": "aspect_ratio_scaled",
        "junction_routing": "non_crossing_streamlines",
        "direction_averaging": "rectangle_rule",
        "direction_spread": "turbulence_intensity",
        "stability": "monin_obukhov",
        "kappa": 0.41,
        "canyon_wind_min": 0.1,
        "u_d_min": 0.001,
        "sigma_w_min": 0.0,
        "sigma_v_min": 0.0,
        "chemistry": {"no_o3_rate": "jpl_2003", "floor_ppb": 0.0},
    },
}
"""SIRANE's and MUNICH's closure sets.

`sirane`: the Bessel canyon wind, `u_d = sigma_w / (sqrt(2) pi)`, non-crossing junction
routing, the exact Gaussian direction average over the driven spread, Monin-Obukhov
turbulence, `kappa = 0.40`, SIRANE's default turbulence floors `sigma_w >= 0.30 m/s` and
`sigma_v >= 0.5 m/s`, and the Soulhac et al. (2011) NO + O3 rate with a 2 ppb floor on `J/k`.
`sigma_w_min` floors the `sigma_w` of the roof exchange velocity; `sigma_v_min` acts only
with `direction_spread="turbulence_intensity"` (the `sirane` preset's driven spread never
reads it; the above-roof plume, `plume.plume_table`, has floors of its own).

`munich`: the exponential canyon wind on the Bessel roof wind, Schulte's exchange,
non-crossing routing, the rectangle-rule direction average over the turbulence-intensity
spread, Monin-Obukhov turbulence, `kappa = 0.41`, a 0.1 m/s floor on the canyon wind and a
0.001 m/s floor on the exchange velocity, no turbulence floors, and the JPL (2003) NO + O3
rate with no floor.
"""

LEGACY_NAMES: dict[str, str] = {
    "roof_wind_form": "roof_wind",
    "exchange": "roof_exchange",
    "routing": "junction_routing",
}
"""Deprecated keyword -> its current name."""

LEGACY_VALUES: dict[str, dict[str, str]] = {
    "canyon_wind": {"soulhac": "bessel_profile", "exponential": "exponential_profile"},
    "roof_wind": {"sirane": "bessel_canyon_mean", "macdonald": "canopy_log_law"},
    "roof_exchange": {"sirane": "turbulent_velocity", "schulte": "aspect_ratio_scaled"},
    "junction_routing": {"sirane": "non_crossing_streamlines", "mixing": "perfect_mixing"},
    "direction_averaging": {"sirane": "exact_gaussian", "munich": "rectangle_rule",
                            "gauss": "gauss_hermite"},
    "stability": {"munich": "monin_obukhov"},
}
"""Per option, deprecated value -> its current value. `direction_averaging="munich"` also
implies `direction_spread="turbulence_intensity"` (see `resolve`)."""

LEGACY = {"names": LEGACY_NAMES, "values": LEGACY_VALUES}
"""Both deprecation maps together."""

_MUNICH_STYLE = {
    "canyon_wind": "exponential_profile",
    "roof_exchange": "aspect_ratio_scaled",
    "roof_wind": "canopy_log_law",
}
"""The choices whose formulas are written with `kappa = 0.41` (see `infer_kappa`)."""


_PACKAGE = __name__.rpartition(".")[0]


def warn_deprecated(message: str) -> None:
    """A `DeprecationWarning` attributed to the line that used the deprecated spelling:
    the caller of the street application's function that called into this module (or the
    direct caller of this module's own functions)."""
    frame = sys._getframe(1)
    level = 2
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    if frame is not None and str(frame.f_globals.get("__name__", "")).startswith(_PACKAGE):
        level += 1
    warnings.warn(message, DeprecationWarning, stacklevel=level)


def preset_options(name: str) -> dict:
    """A copy of preset `name`'s full option set (`PRESETS`)."""
    if name not in PRESETS:
        raise ValueError(f"preset must be one of {tuple(PRESETS)}, got {name!r}")
    options = dict(PRESETS[name])
    options["chemistry"] = dict(options["chemistry"])
    return options


def normalise(option: str, value: str, where: str) -> str:
    """`value` as a current value of `option`.

    A current value is returned unchanged; a deprecated one (`LEGACY_VALUES`) is mapped to
    its replacement with a `DeprecationWarning`; anything else raises `ValueError` naming
    `where`, the option and its allowed values."""
    allowed = OPTIONS[option]
    if value in allowed:
        return value
    replacement = LEGACY_VALUES.get(option, {}).get(value)
    if replacement is not None:
        warn_deprecated(
            f"{where}: {option}={value!r} is deprecated; use {option}={replacement!r}"
        )
        return replacement
    raise ValueError(f"{where}: {option} must be one of {allowed}, got {value!r}")


def infer_kappa(options: Mapping[str, str]) -> float:
    """0.41 when any choice written with that constant is selected (`exponential_profile`,
    `aspect_ratio_scaled` or `canopy_log_law`), else 0.40."""
    munich_form = any(options.get(key) == value for key, value in _MUNICH_STYLE.items())
    return 0.41 if munich_form else 0.40


def resolve(preset: str, options: Mapping[str, object], where: str) -> dict:
    """The full option set: preset `preset` with every non-`None` entry of `options` on top.

    `options` may use deprecated names and values (`LEGACY`); each is translated with a
    `DeprecationWarning`, and a deprecated name given together with its current one raises
    `TypeError`. `direction_averaging="munich"` (deprecated) also sets
    `direction_spread="turbulence_intensity"` unless `direction_spread` is given.

    The von Karman constant: an explicit `kappa` wins; otherwise, when any string option is
    given explicitly, it is `infer_kappa` of the merged set; otherwise the preset's.
    """
    merged = preset_options(preset)
    explicit: dict[str, object] = {}
    given_as: dict[str, str] = {}
    implied_spread = False
    for key, value in options.items():
        if value is None:
            continue
        name = key
        if key in LEGACY_NAMES:
            name = LEGACY_NAMES[key]
            warn_deprecated(f"{where}: the keyword {key!r} is deprecated; use {name!r}")
        if name in explicit:
            raise TypeError(
                f"{where}: {given_as[name]!r} and {key!r} both set {name!r}; give only one"
            )
        given_as[name] = key
        if name in OPTIONS:
            if name == "direction_averaging" and value == "munich":
                implied_spread = True
            value = normalise(name, value, where)
        elif name not in _NUMERIC:
            raise TypeError(f"{where}: unexpected keyword argument {key!r}")
        explicit[name] = value
    if implied_spread and "direction_spread" not in explicit:
        explicit["direction_spread"] = "turbulence_intensity"
    merged.update(explicit)
    if "kappa" not in explicit and any(key in OPTIONS for key in explicit):
        merged["kappa"] = infer_kappa(merged)
    for key in _NUMERIC:
        merged[key] = float(merged[key])
    return merged
