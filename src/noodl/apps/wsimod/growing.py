"""``GrowingSurface`` (``wsimod/nodes/land.py:1082-2312``): the crop calendar, the crop cover, the
nutrient pool and the soil processes.

A growing surface is a :class:`~noodl.apps.wsimod.land.PerviousSurface` whose physical depth is its
``rooting_depth`` (``:1192``, ``:1251``), with a sowing and harvest calendar, a crop factor, a crop
and ground cover and a water-stress coefficient. ``calc_crop_cover`` (``:1514-1572``) runs first in
the surface's ``inflows`` (``:1275``) and writes the day's ``et0_coefficient``
(``crop_factor * Ks``), which ``ihacres`` reads through ``get_climate``
(:meth:`~noodl.apps.wsimod.land.PerviousSurface.crop_et0_coefficient`).

**The nutrient pool** (``nitrate`` among the pollutants, ``has_nutrients``; ``land.py:1273``,
``:1276``). The surface owns a :class:`~noodl.apps.wsimod.nutrient_pool.NutrientPool`
(``nutrient_pool``, ``None`` without ``nitrate``), started from the tank's storage and
``initial_soil_storage`` (``:1293-1324``). Its inputs are the deposition overrides
(``:2267-2311``), ``fertiliser`` and ``manure`` (``:1649-1710``), each scaled to the tank by
``adjust_vqip_to_liquid`` (``:1574-1603``); its outputs so far are ``pull_storage``
(``:1433-1493``), whose nutrient columns leave in the pool's proportions, and
``effective_precipitation_flushing`` (``:1605-1647``), which extracts the same outflows from the
dissolved pools a second time (L8, reproduced: Y16, Y45). A tank with no inorganic N gives the
three N species 0 in ``pull_storage``, where WSIMOD raises ``ZeroDivisionError`` (``:1460``; L5,
Y18; the decision ``S/pull_storage/no_din`` and the event ``"pull_storage_no_din"``).

**The soil processes** run after ``calculate_soil_temperature``, in WSIMOD's list
(``:1283-1291``, D20): the temperature and soil-moisture
factors of the day (``:1813-1856``; the temperature factor jumps at 5 °C, L12, Y19), the pool's
transformations scaled back into the tank (``:1744-1811``), crop uptake (``:1858-1926``), erosion
(``:1928-2094``; its sediment and the adsorbed and humus P go straight into the day's outflows,
bypassing the tank, and a short pool records ``"erosion_pool_short"``), denitrification
(``:2096-2152``) and the Freundlich adsorption of phosphorus (``:2154-2265``; its Newton loop is
unrolled to ``adsorption_nr_maxiter`` passes and its convergence test lags one pass, both
transcribed; no adsorbed P takes WSIMOD's shortcut, ``"adsorption_shortcut"``, and a request
above its pool records ``"freundlich_flow_adjusted"``, L11). Uptake and denitrification take all
their N as nitrate (L9).
Each pulls from the tank through :meth:`GrowingSurface._pull_tracked`, whose clamp records
``"nutrient_pool_not_tracking"`` (L10). Python's ``**``, ``exp``, ``log``, ``log10`` and ``sin`` are
evaluated per instance (``pow_python``, ``exp_python``, ``log_python``, ``log10_python``,
``sin_python``). Two
zero denominators are guarded: ``nitrate + ammonia = 0`` in the transformation (proportion 0,
``"transformation_no_din"``, L6) and ``V = 0`` in the uptake (no available uptake,
``"crop_uptake_dry_soil"``, L7), where WSIMOD raises ``ZeroDivisionError``.

**The calendar is built at build time** (spec, Approach rule 2, D7).
:meth:`GrowingSurface.set_dates` evaluates, for every date of the network and in WSIMOD's own
arithmetic, the raw and the adjusted day of year, the sowing and harvest tests, the ``bisect_left``
indices of both interpolations and the sine of erosion (:class:`CalendarDay`). Only the
interpolated values are tensors (:func:`quick_interp`), so the result is Python's, and the
gradients follow the piecewise-linear segments. A date on which WSIMOD's ``quick_interp`` would
raise (``IndexError`` past the last stage date, or a zero-width segment) is refused by name with
the first such date.

Quirks transcribed (spec, ``calc_crop_cover``):

- **Every year shifts the day of year after 28 February.** ``if t.is_leap_year:`` (``:1526``) tests
  the bound *method* of the individual-files loader's ``to_datetime``, which is truthy, so the
  shift (``:1528-1529``) happens in every year, not only in leap years: 1 March and 28 February
  both interpolate at day 59 and 31 December at 364. WSIMOD's unified-data loader (parquet) has
  pandas timestamps, whose ``is_leap_year`` is a property, so there it happens in leap years only.
  This port reads the individual files and reproduces that loader (L1).
- **Raw and adjusted days.** The sowing and harvest tests use the raw ``t.dayofyear``; the
  interpolations the adjusted one.
- **The sowing day has no crop.** ``days_after_sow = 0`` is false at ``:1550``: the covers start to
  change the day after sowing.
- **The calendar is unsorted** when ``sowing_day > harvest_day`` or ``harvest_day = 365``
  (``[0, sowing, harvest, harvest + 1, 365]``). ``bisect_left`` then returns indices whose segments
  all have ``y = 0``, so an autumn-sown crop has no cover all year (L2); reproduced.
- **Index wrap.** ``x <= xp[0]`` reads ``xp[-1]`` and ``yp[-1]`` (Python's negative index): for
  ``xp = [1, 365]``, ``yp = [0.3, 1]`` at ``x = 1`` the value is ``0.30000000000000004`` (L23).
- **The stress coefficient** divides by ``(1 - ET_depletion_factor) * TAW`` (``:1564-1568``): with
  the default ``ET_depletion_factor = 1`` that is 0 once the soil is at or below its wilting point,
  and WSIMOD raises ``ZeroDivisionError`` (L3). noodl gives ``Ks = 0`` (the limit; FAO-56's value
  at the wilting point) and records ``"crop_water_stress_zero_denominator"`` (Y17).
  ``wilting_point >= field_capacity`` (``TAW <= 0``) is refused (L4, Y44).
- **The harvest branch's ``crop_factor = crop_factor_stages[0]``** (``:1539``) is overwritten at
  ``:1547`` and is not ported (L26).
- **A growing surface never reads its ``et0_coefficient``** (``:1265``, ``:1570`` overwrite it): it
  is listed in ``model.unread`` (L22, Y39).

State: ``days_after_sow`` is ``-1`` for WSIMOD's ``None`` (and ``0`` on the sowing day), the covers
persist outside the season until the harvest resets them (``(...)`` tensors, written under the
caller's ``active`` mask). The decisions are ``S/calc_crop_cover/{unsown,crop,depletion,unstressed,
stress_floor,stress_zero_denominator}``, ``S`` being ``node:N/surfaces.<s>``; with the pool,
``S/pull_storage/{empty,clip,no_din}`` and ``S/adjust_vqip_to_liquid/{N,P}`` (``call`` counts the
dry, wet, fertiliser and manure calls in that order), the pool's own
``S.nutrient_pool.<pool>/extract/clip:<N|P>``, and the soil processes'
``S/calc_temperature_dependence_factor/{warm,thawed}``,
``S/calc_soil_moisture_dependence_factor/{saturated,wilted,min:wp,min:fc}``,
``S/soil_pool_transformation/{no_din,inorganic_N,organic_N,inorganic_P,organic_P}``,
``S/calc_crop_uptake/{crop,temperature_cap,temperature_floor,shape,available_water,dry_soil,cap:P,
cap:N}``, ``S/erosion/{rain,runoff,transport,enrichment_cap,flow,phosphorus}``,
``S/denitrification/{saturated,moist}`` and ``S/adsorption/{empty,shortcut,
newton:<j>/{residual,step,positive},change,adsorb}``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.constants import (
    G_M2_TO_KG_M2,
    KG_M2_TO_KG_KM2,
    KG_M3_TO_KG_KM3,
    KG_M3_TO_MG_L,
    KG_TO_MG,
    M2_TO_KM2,
    M_TO_KM,
    M_TO_MM,
    SURFACE_INPUTS,
)
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.land import SURFACE_CLASSES, PerviousSurface, month_start, runs_nutrients
from noodl.apps.wsimod.nutrient_pool import (
    ADS,
    DI,
    DO,
    FAST,
    NUTRIENT_POOL_PAIRS,
    NUTRIENT_POOL_SCALARS,
    NutrientPool,
)
from noodl.apps.wsimod.storage import exp_python
from noodl.apps.wsimod.vqip import (
    Pollutants,
    extract_vqip,
    log10_python,
    log_python,
    pow_python,
    safe_den,
    sin_python,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import SurfaceSpec, WsimodNetwork
    from noodl.apps.wsimod.land import Land

__all__ = [
    "GROWING_DEFAULTS",
    "NUTRIENT_POLLUTANTS",
    "STRUCTURE_DEFAULTS",
    "CalendarDay",
    "GrowingSurface",
    "quick_interp",
]

# With ``nitrate`` a pollutant a growing surface reads these (``land.py:1294-1307``, ``:1465``,
# ``:2088``, ``:1822``; WSIMOD's docstring, ``:1119-1121``, omits ``solids``: L24)
NUTRIENT_POLLUTANTS = (
    "ammonia",
    "nitrite",
    "nitrate",
    "phosphate",
    "org-nitrogen",
    "org-phosphorus",
    "solids",
    "temperature",
)
# land.py:1308-1324: the keys of initial_soil_storage, in the order WSIMOD reads them
_SOIL_STORAGE_KEYS = (
    "phosphate",
    "ammonia",
    "nitrate",
    "nitrite",
    "org-nitrogen",
    "org-phosphorus",
)
# The parameter domains that, with ``nitrate``, stand for a WSIMOD crash (spec, Validation):
# (name, the refused comparison with 0, why)
_DOMAIN = (
    ("rooting_depth", "<=", "the soil's physical depth (land.py:1849, :2186 divide by it)"),
    ("bulk_density", "<=", "adsorption divides by it (land.py:2186)"),
    ("kfr", "<=", "adsorption takes log(kfr) (land.py:2199)"),
    ("nfr", "<=", "adsorption divides by it (land.py:2199)"),
    ("cohesion", "==", "erosion divides by 0.5 * cohesion (land.py:1965)"),
    ("thetaupp", "==", "the soil-moisture factor divides by it (land.py:1849)"),
    ("thetalow", "==", "the soil-moisture factor divides by it (land.py:1852)"),
    ("hsatINs", "<=", "din_conc + hsatINs can be 0 in denitrification (land.py:2125)"),
    ("limpar", "<", "a dry soil then divides DI.N by 0 in denitrification (land.py:2109-2125)"),
    ("field_capacity", "<=", "denitrification divides by field_capacity_m (land.py:2109)"),
)
_REFUSED = {"<=": "<= 0", "<": "< 0", "==": "0"}

# land.py:1186-1250, WSIMOD's attributes as written (Python evaluates the expressions), set only
# in ``__init__`` and reachable through the overrides (``:1376-1410``); and the two constructor
# keywords that are batched parameters (``:1085-1094``)
GROWING_DEFAULTS: dict[str, float] = {
    "ET_depletion_factor": 1,
    "rooting_depth": 1,
    "crop_cover_max": 0.9,
    "ground_cover_max": 0.3,
    "satact": 0.6,
    "thetaupp": 0.12,
    "thetalow": 0.08,
    "thetapow": 1,
    "uptake1": 15,
    "uptake2": 1,
    "uptake3": 0.02,
    "uptake_PNratio": 1 / 7.2,
    "erodibility": 0.0025,
    "sreroexp": 1.2,
    "cohesion": 1,
    "slope": 5,
    "srfilt": 0.7,
    "macrofilt": 0.01,
    "limpar": 0.7,
    "exppar": 2.5,
    "hsatINs": 1,
    "denpar": 0.015,
    "adosorption_nr_limit": 0.00001,  # WSIMOD's spelling
    "kfr": 153.7,
    "nfr": 1 / 2.6,
    "kadsdes": 0.03,
    "bulk_density": 1300,
}
# the structural parameters: Python values read from the spec at construction, never drivers
# (``:1085-1094``, ``:1330-1349``, ``:2205``)
STRUCTURE_DEFAULTS: dict[str, object] = {
    "crop_factor_stages": [1, 1],
    "crop_factor_stage_dates": [0, 365],
    "sowing_day": 1,
    "harvest_day": 365,
    "adsorption_nr_maxiter": 20,
}
_PI = math.pi  # constants.PI


@dataclass(frozen=True)
class CalendarDay:
    """One date's structural crop arithmetic, in WSIMOD's own (``:1524-1558``, ``:1948``).

    ``raw`` is ``t.dayofyear`` and ``adjusted`` is ``raw`` less 1 after day 59, every year (L1);
    ``sowing`` and ``harvest`` test ``raw`` against ``sowing_day`` and ``harvest_day``;
    ``factor_index`` and ``cover_index`` are ``bisect_left`` of ``adjusted`` in
    ``crop_factor_stage_dates`` and in ``harvest_sow_calendar``; ``erosion_sine`` is
    ``math.sin(2 * PI * ((raw - 70) / 365))``.
    """

    raw: int
    adjusted: int
    sowing: bool
    harvest: bool
    factor_index: int
    cover_index: int
    erosion_sine: float


def quick_interp(x: int, xp: Sequence[float], yp: Tensor, i: int) -> Tensor:
    """``land.py:1495-1512``: ``yp[i-1] + (yp[i] - yp[i-1]) * (x - xp[i-1]) / (xp[i] - xp[i-1])``
    with ``i = bisect_left(xp, x)``, which the calendar holds. ``x``, ``xp`` and ``i`` are
    Python numbers; ``yp`` is ``(..., S)`` and the last axis is read at ``i - 1`` and ``i``
    (Python's negative index at ``i = 0``: the wrap, L23). WSIMOD's order: ``((Δy * dif) / Δx)``,
    then added."""
    x_left, x_right = xp[i - 1], xp[i]
    dif = x - x_left
    y_left, y_right = yp[..., i - 1], yp[..., i]
    return y_left + (y_right - y_left) * dif / (x_right - x_left)


def _integral(values: Mapping, name: str, where: str) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {name} {value!r} is not a number")
    if not float(value).is_integer():
        raise ValueError(
            f"{where}: {name} {value!r} is not a whole day of the year (the sowing and harvest "
            "tests compare it with t.dayofyear, land.py:1532, :1536)"
        )
    return int(value)


def _check_structure(values: Mapping, where: str) -> None:
    """The refusals from the parameters alone: the stage lists of one length, whole sowing and
    harvest days, a non-negative integer ``adsorption_nr_maxiter`` (a loop length, ``:2205``) and
    ``wilting_point < field_capacity`` (``TAW > 0``: ``TAW = 0`` divides by zero on day 0, and
    ``TAW < 0`` only prints in WSIMOD, ``:1360-1366``; L4, Y44)."""
    stages, dates = values["crop_factor_stages"], values["crop_factor_stage_dates"]
    if len(stages) != len(dates):
        raise ValueError(
            f"{where}: crop_factor_stages has {len(stages)} entries but crop_factor_stage_dates "
            f"has {len(dates)}: quick_interp pairs them (land.py:1547-1549)"
        )
    if len(dates) < 2:
        raise ValueError(f"{where}: crop_factor_stage_dates needs at least two dates")
    _integral(values, "sowing_day", where)
    _integral(values, "harvest_day", where)
    maxiter = values["adsorption_nr_maxiter"]
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, float)):
        raise ValueError(f"{where}: adsorption_nr_maxiter {maxiter!r} is not a number")
    if not float(maxiter).is_integer() or maxiter < 0:
        raise ValueError(
            f"{where}: adsorption_nr_maxiter {maxiter!r} is not a non-negative integer "
            "(the Newton-Raphson loop's length, land.py:2205)"
        )
    if not values["wilting_point"] < values["field_capacity"]:
        raise ValueError(
            f"{where}: wilting_point {values['wilting_point']} is not below field_capacity "
            f"{values['field_capacity']}: TAW <= 0 (land.py:1360-1366; TAW = 0 divides by zero on "
            "day 0, L4, Y44)"
        )


def _check_domain(values: Mapping, where: str) -> None:
    """With ``nitrate``, the parameter domains of :data:`_DOMAIN` (spec, Validation): each
    refused value is a WSIMOD crash. ``values`` are numbers or batched tensors; any instance
    outside the domain is refused."""
    for name, op, why in _DOMAIN:
        value = _f64(values[name])
        bad = {"<=": value <= 0, "<": value < 0, "==": value == 0}[op]
        if bool(bad.any()):
            raise ValueError(f"{where}: {name} {value.tolist()} is {_REFUSED[op]}: {why}")


def _check_negative_base(values: Mapping, where: str) -> None:
    """With ``nitrate``: ``thetaupp < 0`` or ``thetalow < 0`` where ``thetapow`` is not integral,
    per instance. The soil-moisture factor's base ``fc_diff / (thetaupp * rooting_depth)`` (or its
    ``thetalow`` twin) is then negative, Python's ``**`` returns a complex number and WSIMOD's
    ``min`` raises ``TypeError`` (``land.py:1849-1853``). With an integral ``thetapow`` WSIMOD
    runs, and so does noodl."""
    power = _f64(values["thetapow"])
    fractional = power != torch.floor(power)
    for name in ("thetaupp", "thetalow"):
        value = _f64(values[name])
        bad = (value < 0) & fractional
        if bool(bad.any()):
            raise ValueError(
                f"{where}: {name} {value.tolist()} < 0 with a fractional thetapow "
                f"{power.tolist()}: a negative base to a fractional power, WSIMOD's min raises "
                "TypeError (land.py:1849-1853)"
            )


# the largest x with a finite math.exp(x): math.log(sys.float_info.max); exp(-kadsdes) overflows
# beyond it
_EXP_LIMIT = 709.782712893384


def _check_kadsdes(values: Mapping, where: str) -> None:
    """With ``nitrate``, per instance: ``kadsdes < -709.782712893384`` (``-kadsdes`` above
    ``math.log(sys.float_info.max)``), where WSIMOD's ``math.exp(-self.kadsdes)`` raises
    ``OverflowError`` on the first day the adsorbed concentration moves (``land.py:2230-2232``).
    A negative ``kadsdes`` within that bound is accepted, as WSIMOD runs it (non-physical: the
    concentration moves away from the equilibrium)."""
    value = _f64(values["kadsdes"])
    bad = -value > _EXP_LIMIT
    if bool(bad.any()):
        raise ValueError(
            f"{where}: kadsdes {value.tolist()} is below -{_EXP_LIMIT}: math.exp(-kadsdes) "
            "overflows, and WSIMOD's adsorption raises OverflowError (land.py:2230-2232)"
        )


def _soil_storage(value: object, where: str) -> Mapping | None:
    """``initial_soil_storage`` (``land.py:1308-1324``, read only with ``nitrate``): falsy (no
    solid pools) or a mapping with :data:`_SOIL_STORAGE_KEYS`, each a number. A float, which
    WSIMOD's docstring allows (``:1142``), raises ``TypeError`` at ``:1312`` and is refused (L16,
    Y37), and so is a missing key (``KeyError``)."""
    if not value:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{where}: initial_soil_storage {value!r} is not a mapping of "
            f"{', '.join(_SOIL_STORAGE_KEYS)} (WSIMOD subscripts it: TypeError at land.py:1312; "
            "L16, Y37)"
        )
    for key in _SOIL_STORAGE_KEYS:
        if key not in value:
            raise ValueError(
                f"{where}: initial_soil_storage lacks {key!r} (land.py:1311-1324 read "
                f"{', '.join(_SOIL_STORAGE_KEYS)}: KeyError)"
            )
        entry = value[key]
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(f"{where}: initial_soil_storage[{key!r}] {entry!r} is not a number")
    return value


def _stages_tensor(top: Tensor) -> Tensor:
    """``[0, 0, top, 0, 0]`` (``land.py:1342-1343``) with ``top`` ``(...)``: ``(..., 5)``."""
    zero = torch.zeros_like(top)
    return torch.stack([zero, zero, top, zero, zero], dim=-1)


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


class GrowingSurface(PerviousSurface):
    """``GrowingSurface`` (``land.py:1082-2312``): construction, overrides, the crop calendar,
    ``calc_crop_cover``, the nutrient pool's inputs and outputs and the soil processes, erosion
    and adsorption among them (see the module text).

    Parameters (``parameter_defaults``): :class:`~noodl.apps.wsimod.land.PerviousSurface`'s without
    ``depth`` (its physical depth is ``rooting_depth``) and ``et0_coefficient`` (overwritten every
    day, never read), plus :data:`GROWING_DEFAULTS`. The structural ones
    (:data:`STRUCTURE_DEFAULTS`) are Python values read from the spec; ``crop_factor_stages`` is
    also accepted by :meth:`bind` as a ``(..., S)`` tensor. ``TAW`` and ``RAW`` (``:1360-1363``),
    the covers' stage tensors (``:1342-1343``) and the derived depths are computed again by every
    :meth:`bind`, so they equal construction's after an override (``:1422-1431``).
    """

    surface_class: ClassVar[str] = "GrowingSurface"
    is_growing: ClassVar[bool] = True
    # D4, D5: the crop registers, carried in "wsimod.<field>" (one row per surface); the crop
    # factor and the day's et0_coefficient are per-day stores, recomputed before every reader
    carried_fields: ClassVar[tuple[str, ...]] = ("days_after_sow", "crop_cover", "ground_cover")
    crop_fields: ClassVar[tuple[str, ...]] = (*carried_fields, "crop_factor", "et0_coefficient")
    parameter_defaults: ClassVar[dict[str, float]] = {
        **{
            k: v
            for k, v in PerviousSurface.parameter_defaults.items()
            if k not in ("depth", "et0_coefficient")
        },
        **GROWING_DEFAULTS,
    }

    def __init__(self, spec: SurfaceSpec, pollutants: Pollutants, *, parent: Land) -> None:
        where = f"node {parent.name!r}, surface {spec.name!r}"
        structure = {**STRUCTURE_DEFAULTS, **spec.parameters}
        _check_structure({**self.parameter_defaults, **structure}, where)
        super().__init__(spec, pollutants, parent=parent)
        # land.py:1326-1351: the calendar, from the structural parameters
        self.crop_factor_stage_dates: list = list(structure["crop_factor_stage_dates"])
        self._stages_default = _f64(list(structure["crop_factor_stages"]))
        self.crop_factor_stages = self._stages_default
        # the driver "wsimod.<Class>.crop_factor_stages" (..., n, S), padded with 0 to the
        # class's longest list (D16, Y33)
        self.defaults["crop_factor_stages"] = self._stages_default
        self.sowing_day = _integral(structure, "sowing_day", where)
        self.harvest_day = _integral(structure, "harvest_day", where)
        self.adsorption_nr_maxiter = int(structure["adsorption_nr_maxiter"])
        self.harvest_sow_calendar = [
            0,
            self.sowing_day,
            self.harvest_day,
            self.harvest_day + 1,
            365,
        ]
        self.autumn_sow = self.sowing_day > 181  # :1346-1349
        # :1261-1270 (``days_after_sow = None`` is -1)
        self.days_after_sow = _f64(-1.0)
        self.crop_cover = _f64(0.0)
        self.ground_cover = _f64(0.0)
        self.crop_factor = _f64(0.0)
        self.et0_coefficient = _f64(1.0)
        self.has_nutrients = runs_nutrients(pollutants.names)  # :1276, as Land.store_paths_of
        self.calendar: list[CalendarDay] = []
        self.inflows.insert(0, self.calc_crop_cover)  # :1275
        # D14: the pool, only with nitrate (an inert pool has no observable effect, spec); its
        # parameters are drivers "wsimod.<Class>.nutrient_pool.<p>" with WSIMOD's defaults
        # (np.py:16-23; L20, Y24, D16)
        self.nutrient_pool: NutrientPool | None = None
        if self.has_nutrients:
            self.nutrient_pool = NutrientPool(site=f"{self.site}.nutrient_pool")
            for name, pair in NUTRIENT_POOL_PAIRS.items():
                self.defaults[f"nutrient_pool.{name}"] = _f64(pair)
            for name, value in NUTRIENT_POOL_SCALARS.items():
                self.defaults[f"nutrient_pool.{name}"] = _f64(value)
            self._fill_pool(_soil_storage(structure.get("initial_soil_storage"), where))
            # :1278-1281 (residue stays unwired, :1281)
            self.inflows += [self.effective_precipitation_flushing, self.fertiliser, self.manure]
            # :1283-1291, after calculate_soil_temperature (D20)
            self.processes += [
                self.calc_temperature_dependence_factor,
                self.calc_soil_moisture_dependence_factor,
                self.soil_pool_transformation,
                self.calc_crop_uptake,
                self.erosion,
                self.denitrification,
                self.adsorption,
            ]

    def _fill_pool(self, soil: Mapping | None) -> None:
        """``land.py:1293-1324``: the dissolved pools from the tank's storage, then the solid
        ones from ``initial_soil_storage`` when it is truthy; every other entry stays 0."""
        storage, column = self.storage, self.p.column
        pool = torch.zeros(5, 2, dtype=torch.float64)
        pool[DI, 1] = storage[..., column("phosphate")]
        pool[DI, 0] = (
            storage[..., column("nitrate")]
            + storage[..., column("ammonia")]
            + storage[..., column("nitrite")]
        )
        pool[DO, 1] = storage[..., column("org-phosphorus")]
        pool[DO, 0] = storage[..., column("org-nitrogen")]
        if soil:
            pool[ADS, 1] = _f64(soil["phosphate"])
            pool[ADS, 0] = _f64(soil["ammonia"]) + _f64(soil["nitrate"]) + _f64(soil["nitrite"])
            pool[FAST, 0] = _f64(soil["org-nitrogen"])
            pool[FAST, 1] = _f64(soil["org-phosphorus"])
        self.nutrient_pool.storage = pool

    # --- the day's context (D14) ----------------------------------------------------------------
    @property
    def ctx(self) -> Ctx:
        return PerviousSurface.ctx.fget(self)

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        PerviousSurface.ctx.fset(self, ctx)
        pool = getattr(self, "nutrient_pool", None)
        if pool is not None:
            pool.ctx = ctx

    # --- parameters -----------------------------------------------------------------------------
    def _set_derived(self, derived: Mapping[str, Tensor]) -> None:
        super()._set_derived(derived)
        # :1353-1365, :1342-1343
        self.TAW = self.field_capacity_m - self.wilting_point_m
        self.RAW = self.TAW * self._param("ET_depletion_factor")
        self.ground_cover_stages = _stages_tensor(self._param("ground_cover_max"))
        self.crop_cover_stages = _stages_tensor(self._param("crop_cover_max"))
        # erosion's sin(slope / 100) (:1966), Python's per instance, constant between binds
        self.slope_sine = sin_python(self._param("slope") / 100)

    @classmethod
    def carried_fields_of(cls, *, nutrients: bool) -> tuple[str, ...]:
        """D5: the crop registers, and with ``nutrients`` the pool."""
        return (*cls.carried_fields, "nutrient_pool") if nutrients else cls.carried_fields

    @classmethod
    def state_fields(cls, *, nutrients: bool) -> tuple[str, ...]:
        """D5: the crop registers and the day's crop factor and ``et0_coefficient``; with
        ``nutrients`` the pool's ``(5, 2)`` storage too."""
        return (*cls.crop_fields, "nutrient_pool") if nutrients else cls.crop_fields

    def stores(self) -> dict[str, Tensor]:
        """:meth:`Surface.stores`, with the pool's storage tensor at ``nutrient_pool``."""
        out = super().stores()
        if self.nutrient_pool is not None:
            out["nutrient_pool"] = self.nutrient_pool.storage
        return out

    def load_store(self, field: str, value: Tensor) -> None:
        """``nutrient_pool`` is the pool's storage; anything else is an attribute.

        Loading the pool (once a day, from the state) also resets its two factors to a scalar
        0: each day sets them before any reader (``land.py:1813-1856``, before ``:1744`` and
        ``:2096``), so their value never carries over, but a factor written under a batched
        mask would carry yesterday's batch shape into a day of another shape."""
        if field == "nutrient_pool" and self.nutrient_pool is not None:
            self.nutrient_pool.storage = value
            self.nutrient_pool.temperature_dependence_factor = _f64(0.0)
            self.nutrient_pool.soil_moisture_dependence_factor = _f64(0.0)
            return
        super().load_store(field, value)

    def driver_columns(self) -> dict[str, tuple | str]:
        """D16: ``crop_factor_stages`` is padded to the class's longest list (``"stages"``); the
        six pair parameters of the pool have the columns ``("N", "P")``."""
        columns: dict[str, tuple | str] = {"crop_factor_stages": "stages"}
        if self.nutrient_pool is not None:
            columns.update({f"nutrient_pool.{name}": ("N", "P") for name in NUTRIENT_POOL_PAIRS})
        return columns

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters, ``crop_factor_stages`` ``(..., S)`` among them, ``S`` at least
        the length ``L`` of ``crop_factor_stage_dates``: the driver of the class is padded to
        its longest list (D16), and an entry beyond ``L`` must be 0 (inert, Y33): another value
        is refused, naming the surface and the index. ``wilting_point`` must stay below
        ``field_capacity`` (``TAW > 0``, L4). With the pool, every instance must be inside the
        parameter domains (:data:`_DOMAIN`), and the ``nutrient_pool.<p>`` parameters bind the
        pool (the pairs brought to the batch shape of all seven)."""
        parameters = dict(parameters)
        if self.nutrient_pool is not None:
            values = {**self.defaults, **parameters}
            _check_domain({name: values[name] for name, _, _ in _DOMAIN}, self.site)
            _check_negative_base(values, self.site)
            _check_kadsdes(values, self.site)
        stages = parameters.pop("crop_factor_stages", None)
        stages = self._stages_default if stages is None else _f64(stages)
        length = len(self.crop_factor_stage_dates)
        if stages.ndim == 0 or stages.shape[-1] < length:
            count = stages.shape[-1] if stages.ndim else 0
            raise ValueError(
                f"{self.site}: crop_factor_stages has {count} entries but "
                f"crop_factor_stage_dates has {length} (land.py:1547)"
            )
        if stages.shape[-1] > length:
            padding = stages[..., length:].reshape(-1, stages.shape[-1] - length)
            nonzero = (padding != 0).any(dim=0).nonzero().flatten().tolist()
            if nonzero:
                index = length + nonzero[0]
                raise ValueError(
                    f"{self.site}: crop_factor_stages[{index}] is "
                    f"{padding[:, nonzero[0]].tolist()}, beyond the surface's {length} stages: "
                    "the padding of the driver is inert and must stay 0 (D16, Y33)"
                )
            stages = stages[..., :length]
        super().bind(parameters)
        self.crop_factor_stages = stages
        if bool((self.TAW <= 0).any()):
            raise ValueError(
                f"{self.site}: wilting_point is not below field_capacity: TAW <= 0 "
                "(land.py:1360-1366, L4)"
            )
        if self.nutrient_pool is not None:
            pool = {n: self._param(f"nutrient_pool.{n}") for n in NUTRIENT_POOL_PAIRS}
            scalars = {n: self._param(f"nutrient_pool.{n}") for n in NUTRIENT_POOL_SCALARS}
            batch = torch.broadcast_shapes(
                *(v.shape[:-1] for v in pool.values()), *(v.shape for v in scalars.values())
            )
            pool = {n: v.expand((*batch, 2)) for n, v in pool.items()}
            self.nutrient_pool.bind({**pool, **scalars})

    @classmethod
    def validate_spec(
        cls, surface: SurfaceSpec, network: WsimodNetwork, where: str, *, pollutants: bool
    ) -> list[str]:
        """Refused by name: the structural parameters (:func:`_check_structure`) and, whenever
        ``nitrate`` is a pollutant of the network (in both modes, D18, Y49): a missing one of
        :data:`NUTRIENT_POLLUTANTS`, a parameter outside its domain (:data:`_DOMAIN`), an
        ``initial_soil_storage`` that is truthy but no mapping of its six keys (L16, Y37), a
        missing fertiliser or manure variable (``land.py:1649-1710`` reads all six) or a missing
        first-of-the-month row on a run date. A quantity-only run lists the data as ignored (the
        caller does)."""
        values = {**cls.parameter_defaults, **STRUCTURE_DEFAULTS, **surface.parameters}
        names = network.pollutants.names
        if "nitrate" in names:
            for name in NUTRIENT_POLLUTANTS:
                if name not in names:
                    raise ValueError(
                        f"{where}: with 'nitrate' a pollutant a growing surface needs "
                        f"{', '.join(NUTRIENT_POLLUTANTS)}, and {name!r} is not one "
                        "(land.py:1294-1307, :1465, :2088, :1822: KeyError; L24)"
                    )
            _check_domain(values, where)
            _check_negative_base(values, where)
            _check_kadsdes(values, where)
            _soil_storage(values.get("initial_soil_storage"), where)
        _check_structure(values, where)
        if "nitrate" not in names:
            return []
        variables = {variable for variable, _ in surface.data}
        for variable in SURFACE_INPUTS:
            if variable not in variables:
                raise ValueError(
                    f"{where}: surface data lacks {variable!r}: with 'nitrate' a pollutant a "
                    "growing surface reads the fertiliser and manure data every day "
                    "(land.py:1649-1710, KeyError; Y49)"
                )
        for date in network.dates:
            month = month_start(date)
            for variable in SURFACE_INPUTS:
                if (variable, month) not in surface.data:
                    raise ValueError(
                        f"{where}: no {variable!r} data on {month:%Y-%m-%d}, the first of the "
                        f"month of {date:%Y-%m-%d} (land.py:1649-1710, via :471)"
                    )
        return []

    # --- the calendar ---------------------------------------------------------------------------
    def set_dates(self, dates: Sequence[datetime], labels: Sequence[str]) -> None:
        """Build one :class:`CalendarDay` per date (D7), in WSIMOD's arithmetic: ``bisect_left`` on
        the same Python lists, Python indexing (the wrap included) and ``math.sin``. ``labels``
        name the dates in a refusal.

        A date on which WSIMOD's ``quick_interp`` over ``crop_factor_stage_dates`` would raise is
        refused, naming the surface, the list and the first such date: ``IndexError`` past the
        last date (``:1505-1509``), or a division by zero from a zero-width segment
        (``[365, 365]``). A duplicate inside the list is no refusal in itself
        (``[0, 100, 100, 365]`` runs: ``bisect_left`` returns the first duplicate). The calendar
        of the covers is checked the same way."""
        where = f"{self.site}"
        calendar: list[CalendarDay] = []
        for date, label in zip(dates, labels, strict=True):
            raw = date.timetuple().tm_yday  # t.dayofyear, :1524
            adjusted = raw - 1 if raw > 59 else raw  # :1526-1529: every year (L1)
            indices = []
            for what, xp in (
                ("crop_factor_stage_dates", self.crop_factor_stage_dates),
                ("the harvest/sow calendar", self.harvest_sow_calendar),
            ):
                i = bisect_left(xp, adjusted)
                if i >= len(xp):
                    raise ValueError(
                        f"{where}: {what} {xp} ends before day {adjusted} of the year, on "
                        f"{label}: quick_interp raises IndexError (land.py:1505-1509)"
                    )
                if xp[i] - xp[i - 1] == 0:
                    raise ValueError(
                        f"{where}: {what} {xp}: quick_interp divides by zero at day {adjusted}, on "
                        f"{label} (a zero-width segment, land.py:1512)"
                    )
                indices.append(i)
            calendar.append(
                CalendarDay(
                    raw=raw,
                    adjusted=adjusted,
                    sowing=raw == self.sowing_day,
                    harvest=raw == self.harvest_day,
                    factor_index=indices[0],
                    cover_index=indices[1],
                    erosion_sine=math.sin(2 * _PI * ((raw - 70) / 365)),
                )
            )
        self.calendar = calendar

    def _day(self) -> CalendarDay:
        day = self.ctx.day
        if not 0 <= day < len(self.calendar):
            raise RuntimeError(
                f"{self.site}: no calendar for day {day}; set_dates builds one per network date"
            )
        return self.calendar[day]

    # --- WSIMOD's methods -----------------------------------------------------------------------
    def crop_et0_coefficient(self) -> Tensor:
        """The day's ``et0_coefficient`` (``land.py:1265``, ``:1570``): the configured one is
        never read (Y39)."""
        return self.et0_coefficient

    def calc_crop_cover(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1514-1572``: sow and harvest, the crop factor and covers, and the day's
        ``et0_coefficient = Ks * crop_factor``.

        ``days_after_sow`` is ``-1`` for ``None`` (the decision ``unsown``, ``:1531``): an unsown
        surface sows on ``sowing_day`` (``0``), a sown one is reset on ``harvest_day`` (``-1``,
        both covers ``0``) or counts a day more (``:1544``). The crop factor is interpolated every
        day; the covers only where ``days_after_sow > 0`` (the decision ``crop``, ``:1550``: the
        sowing day itself has no crop). ``root_zone_depletion = max(fc_m - smc, 0)``
        (``depletion``) and ``Ks`` is 1 below ``RAW`` (``unstressed``) and else
        ``max(0, (TAW - depletion) / ((1 - ET_depletion_factor) * TAW))`` (``stress_floor``). A
        zero denominator (``ET_depletion_factor = 1``) gives ``Ks = 0``
        (``stress_zero_denominator``, Y17), where WSIMOD raises ``ZeroDivisionError``.

        The harvest branch's ``crop_factor = crop_factor_stages[0]`` (``:1539``) is a dead store,
        overwritten at ``:1547``, and is not ported (L26). The sowing, harvest and leap tests are
        structural (:class:`CalendarDay`), and nothing is added to the mass balance.
        """
        ctx = self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        day = self._day()
        call = self._call("calc_crop_cover")
        site = f"{self.site}/calc_crop_cover"
        # :1531-1544
        unsown = ctx.decide(
            f"{site}/unsown", self.days_after_sow, "<", 0.0, active=active, call=call
        )
        sow = active & unsown & torch.as_tensor(day.sowing)
        sown = active & ~unsown
        harvest = sown & torch.as_tensor(day.harvest)
        increment = sown & ~torch.as_tensor(day.harvest)
        days = write(sow, _f64(0.0), self.days_after_sow)
        days = write(harvest, _f64(-1.0), days)
        days = write(increment, days + 1, days)
        self.days_after_sow = days
        self.crop_cover = write(harvest, _f64(0.0), self.crop_cover)
        self.ground_cover = write(harvest, _f64(0.0), self.ground_cover)
        # :1547-1549
        factor = quick_interp(
            day.adjusted, self.crop_factor_stage_dates, self.crop_factor_stages, day.factor_index
        )
        self.crop_factor = write(active, factor, self.crop_factor)
        # :1550-1558
        crop = active & ctx.decide(f"{site}/crop", days, ">", 0.0, active=active, call=call)
        cover = quick_interp(
            day.adjusted, self.harvest_sow_calendar, self.crop_cover_stages, day.cover_index
        )
        ground = quick_interp(
            day.adjusted, self.harvest_sow_calendar, self.ground_cover_stages, day.cover_index
        )
        self.crop_cover = write(crop, cover, self.crop_cover)
        self.ground_cover = write(crop, ground, self.ground_cover)
        # :1560-1568, get_smc (:895-903)
        moisture = self.storage[..., 0] / self.area
        depletion = dmax(
            ctx,
            f"{site}/depletion",
            self.field_capacity_m - moisture,
            0.0,
            active=active,
            call=call,
        )
        unstressed = ctx.decide(
            f"{site}/unstressed", depletion, "<", self.RAW, active=active, call=call
        )
        stressed = active & ~unstressed
        denominator = (1 - self._param("ET_depletion_factor")) * self.TAW
        zero = ctx.decide(
            f"{site}/stress_zero_denominator", denominator, "==", 0.0, active=stressed, call=call
        )
        ctx.event("crop_water_stress_zero_denominator", self.parent.name, stressed & zero)
        stress = dmax(
            ctx,
            f"{site}/stress_floor",
            0.0,
            (self.TAW - depletion) / safe_den(denominator),
            active=stressed & ~zero,
            call=call,
        )
        stress = torch.where(unstressed, _f64(1.0), torch.where(zero, _f64(0.0), stress))
        # :1570
        self.et0_coefficient = write(active, stress * self.crop_factor, self.et0_coefficient)
        return ctx.empty(), ctx.empty()

    # --- the pool's inputs and outputs (land.py:1433-1710, :2267-2311) -----------------------
    def _with_columns(self, vqip: Tensor, values: Mapping[str, Tensor]) -> Tensor:
        """``vqip`` with ``vqip[name] = value`` for each entry (WSIMOD's dict assignments)."""
        shape = torch.broadcast_shapes(vqip.shape[:-1], *(_f64(v).shape for v in values.values()))
        vqip = vqip.expand(shape + vqip.shape[-1:])
        for name, value in values.items():
            hot = torch.zeros(1 + self.p.K, dtype=torch.bool)
            hot[self.p.column(name)] = True
            vqip = torch.where(hot, _f64(value).expand(shape)[..., None], vqip)
        return vqip

    def _nutrients(self, vqip: Tensor) -> Tensor:
        """``deposition = {N: nitrate + ammonia, P: phosphate}`` (``land.py:1671-1672``,
        ``:2280-2281``, ``:2303-2304``), ``(..., 2)``."""
        column = self.p.column
        n = vqip[..., column("nitrate")] + vqip[..., column("ammonia")]
        return torch.stack([n, vqip[..., column("phosphate")]], dim=-1)

    def pull_storage(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``land.py:1433-1493``. Without the pool, :meth:`Tank.pull_storage`. With it, empty when
        ``V == 0`` (``empty``); otherwise ``reply = v_change_vqip(storage, min(request, V))``
        (``clip``), ``prop = reply.volume / V``, the pool's ``extract_dissolved(prop)``, and the
        reply's nutrient columns from what the pool gave: ``nitrate``, ``ammonia`` and
        ``nitrite`` each ``inorganic.N * storage[x] / din`` with ``din = nitrate + ammonia +
        nitrite`` of the storage, ``phosphate`` the inorganic P and ``org-phosphorus``,
        ``org-nitrogen`` the organic P and N; then the reply leaves the storage. The volumes never
        depend on the pool, but the nutrient columns leave in the pool's proportions.

        ``din = 0`` (``no_din``): WSIMOD raises ``ZeroDivisionError`` (``:1460``); noodl gives
        the three species 0, the tank having none to give, and records
        ``"pull_storage_no_din"`` (L5, Y18). The pool has lost its share all the same."""
        pool = self.nutrient_pool
        if pool is None:
            return super().pull_storage(vqip, active=active)
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("pull_storage")
        site = f"{self.site}/pull_storage"
        storage = self.storage
        # :1446
        dry = ctx.decide(f"{site}/empty", storage[..., 0], "==", 0.0, active=active, call=call)
        run = active & ~dry
        # :1450-1453
        volume = dmin(ctx, f"{site}/clip", vqip[..., 0], storage[..., 0], active=run, call=call)
        reply = v_change_vqip(p, storage, volume)
        # :1457-1458 (V > 0 where it is used)
        proportion = reply[..., 0] / safe_den(storage[..., 0])
        inorganic, organic = pool.extract_dissolved(proportion, active=run)
        # :1459-1485
        column = p.column
        din = storage[..., column("nitrate")] + storage[..., column("ammonia")]
        din = din + storage[..., column("nitrite")]
        no_din = ctx.decide(f"{site}/no_din", din, "==", 0.0, active=run, call=call)
        ctx.event("pull_storage_no_din", self.parent.name, run & no_din)
        den = torch.where(no_din, torch.ones_like(din), din)
        zero = torch.zeros_like(din)
        species = {
            name: torch.where(no_din, zero, inorganic[..., 0] * storage[..., column(name)] / den)
            for name in ("nitrate", "ammonia", "nitrite")
        }
        reply = self._with_columns(
            reply,
            {
                **species,
                "phosphate": inorganic[..., 1],
                "org-phosphorus": organic[..., 1],
                "org-nitrogen": organic[..., 0],
            },
        )
        # :1491
        self.storage = write(run, extract_vqip(p, storage, reply), storage)
        return torch.where(dry[..., None], ctx.empty(), reply)

    def adjust_vqip_to_liquid(
        self, vqip: Tensor, dep: Tensor, in_: Tensor, *, active: Tensor
    ) -> Tensor:
        """``land.py:1574-1603``: with ``dep.N > 0`` (decision ``N``) ``nitrate``, ``ammonia`` and
        ``org-nitrogen`` become ``x * (in_.N / dep.N)``; with ``dep.P > 0`` (``P``)
        ``phosphate`` and ``org-phosphorus`` become ``x * (in_.P / dep.P)``. The ratio is
        evaluated even where it is exactly 1 (fertiliser, wet deposition). A non-positive
        ``dep`` (negative data, Review Focus 3) leaves ``vqip`` as it is. Without the pool
        nothing changes (``:1594``)."""
        if self.nutrient_pool is None:
            return vqip
        ctx = self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("adjust_vqip_to_liquid")
        site = f"{self.site}/adjust_vqip_to_liquid"
        for j, nutrient, names in (
            (0, "N", ("nitrate", "ammonia", "org-nitrogen")),
            (1, "P", ("phosphate", "org-phosphorus")),
        ):
            positive = ctx.decide(
                f"{site}/{nutrient}", dep[..., j], ">", 0.0, active=active, call=call
            )
            ratio = in_[..., j] / safe_den(dep[..., j])  # dep > 0 where it is used
            scaled = self._with_columns(
                vqip, {name: vqip[..., self.p.column(name)] * ratio for name in names}
            )
            vqip = torch.where((active & positive)[..., None], scaled, vqip)
        return vqip

    def dry_deposition_to_tank(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``land.py:2267-2288``: ``dep = {N: nitrate + ammonia, P: phosphate}`` to the pool's
        ``allocate_dry_deposition`` (the N split between the fast and dissolved inorganic pools,
        all the P to the adsorbed pool), the tank's share scaled by
        :meth:`adjust_vqip_to_liquid` (its phosphate ``x * (0 / P) = 0``) and forced in. Without
        the pool, :class:`~noodl.apps.wsimod.land.Surface`'s: the whole deposition enters."""
        if self.nutrient_pool is None:
            return super().dry_deposition_to_tank(vqip, active=active)
        dep = self._nutrients(vqip)
        in_ = self.nutrient_pool.allocate_dry_deposition(dep, active=active)
        vqip = self.adjust_vqip_to_liquid(vqip, dep, in_, active=active)
        self.push_storage(vqip, force=True, active=active)
        return vqip

    def wet_deposition_to_tank(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``land.py:2290-2311``: as the dry one, with ``allocate_wet_deposition`` (all into the
        dissolved inorganic pool, so the ratios are 1)."""
        if self.nutrient_pool is None:
            return super().wet_deposition_to_tank(vqip, active=active)
        dep = self._nutrients(vqip)
        in_ = self.nutrient_pool.allocate_wet_deposition(dep, active=active)
        vqip = self.adjust_vqip_to_liquid(vqip, dep, in_, active=active)
        self.push_storage(vqip, force=True, active=active)
        return vqip

    def effective_precipitation_flushing(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1605-1647``: the inorganic N and P of the day's subsurface flow,
        percolation and infiltration excess (in WSIMOD's order of terms) are extracted from the
        dissolved inorganic pool, then their organic N and P from the dissolved organic pool.

        ``pull_storage`` has already extracted the subsurface flow's and the percolation's from
        the same pools (``:1458``), so the pools lose them twice (L8): reproduced, with no switch
        (Y16, Y45). Nothing enters or leaves the tank."""
        pool, column = self.nutrient_pool, self.p.column
        ss, perc, ie = self.subsurface_flow, self.percolation, self.infiltration_excess

        def terms(names: tuple[str, ...]) -> Tensor:
            total = None
            for flow in (ss, perc, ie):
                for name in names:
                    value = flow[..., column(name)]
                    total = value if total is None else total + value
            return total

        # :1615-1631
        inorganic = torch.stack(
            torch.broadcast_tensors(
                terms(("ammonia", "nitrite", "nitrate")), terms(("phosphate",))
            ),
            dim=-1,
        )
        pool.extract(DI, inorganic, active=active)
        # :1635-1645
        organic = torch.stack(
            torch.broadcast_tensors(terms(("org-nitrogen",)), terms(("org-phosphorus",))), dim=-1
        )
        pool.extract(DO, organic, active=active)
        return self.ctx.empty(), self.ctx.empty()

    def _applied(self, kind: str, allocate, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``fertiliser`` and ``manure`` (``land.py:1649-1710``): the day's month's ``nhx``,
        ``noy`` and ``srp`` data times the area as ``ammonia``, ``nitrate`` and ``phosphate``,
        ``dep = {N: nitrate + ammonia, P: phosphate}`` into the pool by ``allocate``, the tank's
        share scaled by :meth:`adjust_vqip_to_liquid` and forced in."""
        vqip = self._deposition(kind)
        dep = self._nutrients(vqip)
        in_ = allocate(dep, active=active)
        vqip = self.adjust_vqip_to_liquid(vqip, dep, in_, active=active)
        self.push_storage(vqip, force=True, active=active)
        return vqip, self.ctx.empty()

    def fertiliser(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1649-1679``: all into the dissolved inorganic pool
        (``allocate_fertiliser``, ``np.py:314-328``), so the ratios are 1."""
        return self._applied("fertiliser", self.nutrient_pool.allocate_fertiliser, active=active)

    def manure(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1681-1710``: ``fraction_manure_to_dissolved_inorganic`` of each nutrient
        into the dissolved inorganic pool and the rest into the fast pool (``allocate_manure``,
        ``np.py:267-290``); the tank gets the dissolved share."""
        return self._applied("manure", self.nutrient_pool.allocate_manure, active=active)

    # --- the soil processes (land.py:1744-1926, :2096-2152) -----------------------------------
    def _pull_tracked(self, out_: Tensor, *, active: Tensor) -> Tensor:
        """D19: :meth:`Tank.pull_pollutants` of ``out_`` (``:1806``, ``:1920``, ``:2148``,
        ``:2250``), returning the clamped VQIP, as WSIMOD's mutated argument is. WSIMOD's check
        ``compare_vqip(out_, out2_)`` compares that object with itself and never prints (L10);
        its evident intent is the event ``"nutrient_pool_not_tracking"``, recorded where a clamp
        removed more than ``FLOAT_ACCURACY`` from any column (Y16)."""
        active = torch.as_tensor(active, dtype=torch.bool)
        clamped = self.pull_pollutants(out_, active=active)
        lost = ((out_ - clamped) > self.ctx.fa).any(dim=-1)
        self.ctx.event("nutrient_pool_not_tracking", self.parent.name, active & lost)
        return clamped

    def _smc(self) -> Tensor:
        """``get_smc`` (``land.py:895-903``): ``storage["volume"] / area``."""
        return self.storage[..., 0] / self.area

    def calc_temperature_dependence_factor(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1813-1831``: the pool's ``temperature_dependence_factor`` from the soil
        storage's temperature ``T`` (after ``calculate_soil_temperature``): ``2 ** ((T - 20) /
        10)`` where ``T > 5`` (``warm``), ``T / 5`` where ``0 < T <= 5`` (``thawed``), else 0.

        The factor jumps at 5 °C, from 1 to ``2 ** -1.5``, so it is larger at 4.9 °C than at
        15 °C: HYPE, and WSIMOD's own River, use the product ``2 ** ((T - 20) / 10) * T / 5``
        below 5 °C (L12). Reproduced (Y19). Nothing enters or leaves the tank."""
        ctx, pool = self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("calc_temperature_dependence_factor")
        site = f"{self.site}/calc_temperature_dependence_factor"
        temperature = self.storage[..., self.p.column("temperature")]
        warm = ctx.decide(f"{site}/warm", temperature, ">", 5.0, active=active, call=call)
        thawed = ctx.decide(
            f"{site}/thawed", temperature, ">", 0.0, active=active & ~warm, call=call
        )
        # 2 ** x per instance (Python's pow); a cold instance raises 2 to 0 (unused)
        exponent = torch.where(warm, (temperature - 20) / 10, torch.zeros_like(temperature))
        factor = torch.where(
            warm,
            pow_python(2.0, exponent),
            torch.where(thawed, temperature / 5, torch.zeros_like(temperature)),
        )
        pool.temperature_dependence_factor = write(
            active, factor, pool.temperature_dependence_factor
        )
        return ctx.empty(), ctx.empty()

    def calc_soil_moisture_dependence_factor(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1833-1856``: the pool's ``soil_moisture_dependence_factor`` from ``smc``:
        ``satact`` where ``smc >= field_capacity_m`` (``saturated``), 0 where ``smc <=
        wilting_point_m`` (``wilted``), else ``min(1, wp_comp, fc_comp)`` with ``fc_comp = (1 -
        satact) * ((fc_m - smc) / (thetaupp * rooting_depth)) ** thetapow + satact`` and
        ``wp_comp = ((smc - wp_m) / (thetalow * rooting_depth)) ** thetapow``. Python's ``min``
        takes the first of the smallest: ``min:wp`` is ``wp_comp < 1`` and ``min:fc`` is
        ``fc_comp`` below that. Nothing enters or leaves the tank."""
        ctx, pool = self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("calc_soil_moisture_dependence_factor")
        site = f"{self.site}/calc_soil_moisture_dependence_factor"
        smc = self._smc()
        fc_m, wp_m = self.field_capacity_m, self.wilting_point_m
        saturated = ctx.decide(f"{site}/saturated", smc, ">=", fc_m, active=active, call=call)
        wilted = ctx.decide(
            f"{site}/wilted", smc, "<=", wp_m, active=active & ~saturated, call=call
        )
        between = active & ~saturated & ~wilted
        rooting_depth, power = self._param("rooting_depth"), self._param("thetapow")
        satact = self._param("satact")
        one = torch.ones_like(smc)
        # the bases are positive where they are used; elsewhere 1 (Python's ** of a negative
        # base would be complex)
        fc_diff = fc_m - smc
        fc_comp = pow_python(
            torch.where(between, fc_diff / (self._param("thetaupp") * rooting_depth), one), power
        )
        fc_comp = (1 - satact) * fc_comp + satact
        wp_diff = smc - wp_m
        wp_comp = pow_python(
            torch.where(between, wp_diff / (self._param("thetalow") * rooting_depth), one), power
        )
        smallest = dmin(ctx, f"{site}/min:wp", 1.0, wp_comp, active=between, call=call)
        smallest = dmin(ctx, f"{site}/min:fc", smallest, fc_comp, active=between, call=call)
        factor = torch.where(
            saturated, satact + torch.zeros_like(smc), torch.where(wilted, 0 * one, smallest)
        )
        pool.soil_moisture_dependence_factor = write(
            active, factor, pool.soil_moisture_dependence_factor
        )
        return ctx.empty(), ctx.empty()

    def soil_pool_transformation(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1744-1811``: ``nitrate_proportion = nitrate / (nitrate + ammonia)`` of the
        tank (nitrite ignored), the pool's five transformations (``np.py:396-450``), then each
        increase into the tank: an increase in dissolved inorganic N above 0 (``inorganic_N``)
        enters as ``in_.nitrate = inc * proportion`` and ``in_.ammonia = inc * (1 -
        proportion)``, otherwise ``-inc`` times the same leaves in ``out_``; likewise the
        organic N (``organic_N``), the phosphate (``inorganic_P``) and the organic P
        (``organic_P``), each whole. An increase of exactly 0 takes the ``else`` branch (``out_``
        gets ``-0.0``, rule 6). ``in_`` is forced in and ``out_`` pulled
        (:meth:`_pull_tracked`, clamped); returns both.

        ``nitrate + ammonia = 0`` (``no_din``): WSIMOD raises ``ZeroDivisionError``
        (``:1758``); noodl uses the proportion 0, which sends the change to ammonia, the product
        of mineralisation, and records ``"transformation_no_din"`` (L6, Y18)."""
        p, ctx, pool = self.p, self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("soil_pool_transformation")
        site = f"{self.site}/soil_pool_transformation"
        storage = self.storage
        # :1758-1760
        nitrate = storage[..., p.column("nitrate")]
        din = nitrate + storage[..., p.column("ammonia")]
        no_din = ctx.decide(f"{site}/no_din", din, "==", 0.0, active=active, call=call)
        ctx.event("transformation_no_din", self.parent.name, active & no_din)
        proportion = torch.where(no_din, torch.zeros_like(din), nitrate / safe_den(din))
        # np.py:396-450
        inorganic, organic = pool.soil_pool_transformation(active=active)
        # :1770-1802
        ins, outs = {}, {}
        for branch, increase, names in (
            ("inorganic_N", inorganic[..., 0], ("nitrate", "ammonia")),
            ("organic_N", organic[..., 0], ("org-nitrogen",)),
            ("inorganic_P", inorganic[..., 1], ("phosphate",)),
            ("organic_P", organic[..., 1], ("org-phosphorus",)),
        ):
            up = ctx.decide(f"{site}/{branch}", increase, ">", 0.0, active=active, call=call)
            zero = torch.zeros_like(increase)
            if branch == "inorganic_N":
                parts = {"nitrate": proportion, "ammonia": 1 - proportion}
                into = {n: increase * share for n, share in parts.items()}
                out = {n: -increase * share for n, share in parts.items()}
            else:
                into, out = {names[0]: increase}, {names[0]: -increase}
            for name in names:
                ins[name] = torch.where(up, into[name], zero)
                outs[name] = torch.where(up, zero, out[name])
        in_ = self._with_columns(ctx.empty(), ins)
        out_ = self._with_columns(ctx.empty(), outs)
        # :1805-1806
        self.push_storage(in_, force=True, active=active)
        out_ = self._pull_tracked(out_, active=active)
        return in_, out_

    def calc_crop_uptake(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1858-1926``. With a crop (``days_after_sow > 0``, WSIMOD's truthy
        ``days_after_sow``; ``crop``):

        - ``temp_func`` is 1, or for an autumn-sown crop (structural) ``max(0, min(1, (T - 5) /
          20))`` of the soil storage's temperature (``temperature_cap``, ``temperature_floor``)
          with ``d = days_after_sow - 25``;
        - ``uptake_par = (uptake1 - uptake2) * exp(-uptake3 * d)``, and where ``uptake_par +
          uptake2 > 0`` (``shape``) ``N = uptake1 * uptake2 * uptake3 * uptake_par / (uptake2 +
          uptake_par) ** 2``, else 0; then ``N *= temp_func * G_M2_TO_KG_M2 * area`` and ``P =
          N * uptake_PNratio``;
        - the available uptake ``max(0, V - wilting_point_m * area) / V`` (``available_water``)
          times the dissolved inorganic pool's N and P;
        - ``{"P": min(P, P_max), "N": min(N, N_max)}`` (``cap:P``, ``cap:N``) is extracted from
          the dissolved inorganic pool in that order, P first, and leaves the tank as nitrate
          and phosphate (:meth:`_pull_tracked`; all the N as nitrate, L9).

        ``V = 0`` with a crop (``dry_soil``): WSIMOD raises ``ZeroDivisionError`` (``:1899``,
        reachable with ``ET_depletion_factor < 1``); noodl makes the available uptake 0, there
        being no water, and records ``"crop_uptake_dry_soil"`` (L7, Y18). Without a crop both
        VQIPs are empty."""
        p, ctx, pool = self.p, self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("calc_crop_uptake")
        site = f"{self.site}/calc_crop_uptake"
        days = self.days_after_sow
        crop = active & ctx.decide(f"{site}/crop", days, ">", 0.0, active=active, call=call)
        temperature = self.storage[..., p.column("temperature")]
        if self.autumn_sow:  # :1877-1881
            capped = dmin(
                ctx, f"{site}/temperature_cap", 1.0, (temperature - 5) / 20, active=crop, call=call
            )
            temp_func = dmax(ctx, f"{site}/temperature_floor", 0.0, capped, active=crop, call=call)
            days = days - 25
        else:
            temp_func = _f64(1.0)
        # :1884-1896
        uptake1, uptake2 = self._param("uptake1"), self._param("uptake2")
        uptake3 = self._param("uptake3")
        # exp only where a crop grows: WSIMOD never evaluates it elsewhere, so a no-crop day or a
        # masked instance cannot overflow math.exp
        exponent = torch.where(crop, -uptake3 * days, torch.zeros_like(uptake3 * days))
        uptake_par = (uptake1 - uptake2) * exp_python(exponent)
        shape = ctx.decide(f"{site}/shape", uptake_par + uptake2, ">", 0.0, active=crop, call=call)
        total = torch.where(shape, uptake2 + uptake_par, torch.ones_like(uptake_par))
        common_n = uptake1 * uptake2 * uptake3 * uptake_par / pow_python(total, 2.0)
        common_n = torch.where(shape, common_n, torch.zeros_like(common_n))
        common_n = common_n * (temp_func * G_M2_TO_KG_M2 * self.area)
        common_p = common_n * self._param("uptake_PNratio")
        # :1898-1907
        volume = self.storage[..., 0]
        available = dmax(
            ctx,
            f"{site}/available_water",
            0.0,
            volume - self.wilting_point_m * self.area,
            active=crop,
            call=call,
        )
        dry = ctx.decide(f"{site}/dry_soil", volume, "==", 0.0, active=crop, call=call)
        ctx.event("crop_uptake_dry_soil", self.parent.name, crop & dry)
        share = torch.where(dry, torch.zeros_like(volume), available / safe_den(volume))
        stored = pool.storage[..., DI, :]
        max_n, max_p = share * stored[..., 0], share * stored[..., 1]
        # :1909-1913
        uptake_p = dmin(ctx, f"{site}/cap:P", common_p, max_p, active=crop, call=call)
        uptake_n = dmin(ctx, f"{site}/cap:N", common_n, max_n, active=crop, call=call)
        request = torch.stack(torch.broadcast_tensors(uptake_n, uptake_p), dim=-1)
        taken = pool.extract(DI, request, active=crop, order=("P", "N"))
        # :1915-1920
        out_ = self._with_columns(
            ctx.empty(), {"nitrate": taken[..., 0], "phosphate": taken[..., 1]}
        )
        out_ = self._pull_tracked(out_, active=crop)
        return ctx.empty(), torch.where(crop[..., None], out_, ctx.empty())

    def erosion(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1928-2094`` (HYPE's erosion), in WSIMOD's operand order:

        - the rain depth ``pd = precipitation * M_TO_MM`` is the Land's forcing (``:1940``);
          above 5 mm (``rain``) ``mobilised_rain = (8.95 + 8.44 * log10(pd * (0.257 +
          sin(2 * PI * ((raw - 70) / 365)) * 0.09) * 2)) * pd * (1 - crop_cover) *
          erodibility``, the sine of the raw day of the year (:attr:`CalendarDay.erosion_sine`),
          else 0;
        - with infiltration excess (``runoff``) ``mobilised_flow = (ie / area * M_TO_MM * 365)
          ** sreroexp * ((1 - ground_cover) * (1 / (0.5 * cohesion)) * sin(slope / 100) /
          365)``, else 0;
        - ``total_flows = ie + ss + perc`` (the three volumes; surface runoff is not one),
          ``erodingflow = total_flows / area * M_TO_MM``, the transport factor ``min(1,
          (erodingflow / 4) ** 1.3)`` (``transport``) and ``erodedsed = 1000 * (mobilised_flow
          + mobilised_rain) * transportfactor``;
        - the enrichment is 1.5 above 4 mm (``enrichment_cap``), ``4 - (4 - 1.5) * erodingflow /
          4`` above 0 (``flow``); otherwise both VQIPs are empty and nothing changes;
        - ``erodedP = erodedsed * (erodable / (rooting_depth * M_TO_KM * bulk_density *
          KG_M3_TO_KG_KM3)) * enrichment`` with the pool's ``ads.P + humus.P`` per km²; both in
          kg (``* (area * M2_TO_KM2)``), split over the three flows by ``srfilt * ie /
          total_flows`` and ``macrofilt * ss`` (or ``perc``) ``/ total_flows``;
        - where ``eff_erodedP = perc_P + surface_P + sub_P > 0`` (``phosphorus``) the pool's
          ``erode_P`` removes it from the adsorbed and humus pools, and each flow gains its share
          of the removed organic P as ``org-phosphorus`` and of the adsorbed P as
          ``phosphate``. A pool holding less than ``eff_erodedP`` (``abs(inorg + org - eff) >
          FLOAT_ACCURACY``) records ``"erosion_pool_short"``, where WSIMOD prints "weird
          nutrients" (``:2051-2054``);
        - each flow gains its sediment as ``solids`` (``:2088-2090``; refused without ``solids``,
          L24).

        The eroded masses go straight into the day's outflow VQIPs and are returned as the
        inflow ``in_`` (``phosphate``, ``org-phosphorus``, ``solids``): they never pass through
        the tank (transcribed, ``:2079-2081``). Every power, ``log10`` and ``sin`` is Python's,
        per instance, on a base or argument replaced by 1 where its branch is not taken."""
        p, ctx, pool = self.p, self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("erosion")
        site = f"{self.site}/erosion"
        day = self._day()
        area = self.area
        ie, ss, perc = self.infiltration_excess, self.subsurface_flow, self.percolation
        ie_v, ss_v, perc_v = ie[..., 0], ss[..., 0], perc[..., 0]
        # :1940-1956
        depth = self.parent.params["precipitation"] * M_TO_MM
        rain = active & ctx.decide(f"{site}/rain", depth, ">", 5.0, active=active, call=call)
        seasonal = 0.257 + day.erosion_sine * 0.09  # a Python float of the date (rule 2)
        argument = torch.where(rain, depth * seasonal * 2, torch.ones_like(depth))
        energy = 8.95 + 8.44 * log10_python(argument)
        energy = energy * depth
        mobilised_rain = energy * (1 - self.crop_cover) * self._param("erodibility")
        mobilised_rain = torch.where(rain, mobilised_rain, torch.zeros_like(mobilised_rain))
        # :1959-1970
        runoff = active & ctx.decide(f"{site}/runoff", ie_v, ">", 0.0, active=active, call=call)
        base = torch.where(runoff, ie_v / area * M_TO_MM * 365, torch.ones_like(ie_v))
        mobilised_flow = pow_python(base, self._param("sreroexp"))
        cover = (1 - self.ground_cover) * (1 / (0.5 * self._param("cohesion")))
        mobilised_flow = mobilised_flow * (cover * self.slope_sine / 365)
        mobilised_flow = torch.where(runoff, mobilised_flow, torch.zeros_like(mobilised_flow))
        # :1974-1987
        total_flows = ie_v + ss_v + perc_v
        eroding = total_flows / area * M_TO_MM
        # the volumes are non-negative, so the base is; an inactive one is replaced
        quarter = torch.where(active, eroding / 4, torch.zeros_like(eroding))
        transport = dmin(
            ctx, f"{site}/transport", 1.0, pow_python(quarter, 1.3), active=active, call=call
        )
        sediment = 1000 * (mobilised_flow + mobilised_rain) * transport
        # :1992-1997
        cap = ctx.decide(f"{site}/enrichment_cap", eroding, ">", 4.0, active=active, call=call)
        flowing = ctx.decide(f"{site}/flow", eroding, ">", 0.0, active=active & ~cap, call=call)
        run = active & (cap | flowing)
        enrichment = torch.where(cap, _f64(1.5), 4 - (4 - 1.5) * eroding / 4)
        # :2000-2019
        erodable = pool.get_erodable_P() / area * KG_M2_TO_KG_KM2
        soil = self._param("rooting_depth") * M_TO_KM * self._param("bulk_density")
        soil = soil * KG_M3_TO_KG_KM3
        eroded_p = sediment * (erodable / soil) * enrichment
        eroded_p = eroded_p * (area * M2_TO_KM2)
        sediment = sediment * (area * M2_TO_KM2)
        # :2022-2041 (total_flows > 0 where it is used)
        total = torch.where(run, total_flows, torch.ones_like(total_flows))
        srfilt, macrofilt = self._param("srfilt"), self._param("macrofilt")
        surface_p = srfilt * ie_v / total * eroded_p
        surface_sed = srfilt * ie_v / total * sediment
        sub_p = macrofilt * ss_v / total * eroded_p
        sub_sed = macrofilt * ss_v / total * sediment
        perc_p = macrofilt * perc_v / total * eroded_p
        perc_sed = macrofilt * perc_v / total * sediment
        # :2047-2054
        eff = perc_p + surface_p + sub_p
        phosphorus = run & ctx.decide(f"{site}/phosphorus", eff, ">", 0.0, active=run, call=call)
        organic, inorganic = pool.erode_P(eff, active=phosphorus)
        removed = inorganic + organic
        short = (removed - eff).abs() > ctx.fa
        ctx.event("erosion_pool_short", self.parent.name, phosphorus & short)
        # :2057-2077 (eff > 0 where it is used)
        eff = torch.where(phosphorus, eff, torch.ones_like(eff))
        column = p.column
        flows = []
        for flow, share_p, share_sed in (
            (ie, surface_p, surface_sed),
            (ss, sub_p, sub_sed),
            (perc, perc_p, perc_sed),
        ):
            gained = self._with_columns(
                flow,
                {
                    "org-phosphorus": flow[..., column("org-phosphorus")] + share_p * organic / eff,
                    "phosphate": flow[..., column("phosphate")] + share_p * inorganic / eff,
                },
            )
            flow = write(phosphorus, gained, flow)
            # :2088-2090
            gained = self._with_columns(flow, {"solids": flow[..., column("solids")] + share_sed})
            flows.append(write(run, gained, flow))
        self.infiltration_excess, self.subsurface_flow, self.percolation = flows
        # :2079-2092
        zero = torch.zeros_like(removed)
        in_ = self._with_columns(
            ctx.empty(),
            {
                "phosphate": torch.where(phosphorus, inorganic, zero),
                "org-phosphorus": torch.where(phosphorus, organic, zero),
                "solids": surface_sed + sub_sed + perc_sed,
            },
        )
        return torch.where(run[..., None], in_, ctx.empty()), ctx.empty()

    def denitrification(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:2096-2152``: the moisture dependence ``f`` is 1 where ``smc >
        field_capacity_m`` (``saturated``), ``((smc / fc_m - limpar) / (1 - limpar)) ** exppar``
        where ``smc / fc_m > limpar`` (``moist``; at ``smc == fc_m`` this is ``1.0``), and
        otherwise both VQIPs are empty. ``din_conc = DI.N / V * KG_M3_TO_MG_L``, ``h = din_conc
        / (din_conc + hsatINs)`` and ``DI.N * h * f * temperature_dependence_factor * denpar``
        is extracted from the dissolved inorganic pool (``{N, P: 0}``) and leaves the tank as
        nitrate (:meth:`_pull_tracked`; all the N as nitrate, L9)."""
        ctx, pool = self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("denitrification")
        site = f"{self.site}/denitrification"
        smc, fc_m = self._smc(), self.field_capacity_m
        saturated = ctx.decide(f"{site}/saturated", smc, ">", fc_m, active=active, call=call)
        ratio = smc / fc_m
        limpar = self._param("limpar")
        moist = ctx.decide(
            f"{site}/moist", ratio, ">", limpar, active=active & ~saturated, call=call
        )
        run = active & (saturated | moist)
        base = torch.where(
            ~saturated & moist, (ratio - limpar) / safe_den(1 - limpar), torch.ones_like(ratio)
        )
        dependence = torch.where(
            saturated, torch.ones_like(ratio), pow_python(base, self._param("exppar"))
        )
        # :2119-2135
        volume = self.storage[..., 0]
        din = pool.storage[..., DI, 0]
        concentration = din / safe_den(volume) * KG_M3_TO_MG_L
        half_saturation = concentration / safe_den(concentration + self._param("hsatINs"))
        denitrified = din * half_saturation * dependence * pool.temperature_dependence_factor
        denitrified = denitrified * self._param("denpar")
        # :2136-2140, the request {N: dN, P: 0}
        request = torch.stack([denitrified, torch.zeros_like(denitrified)], dim=-1)
        taken = pool.extract(DI, request, active=run)
        # :2144-2152
        out_ = self._with_columns(ctx.empty(), {"nitrate": taken[..., 0]})
        out_ = self._pull_tracked(out_, active=run)
        return ctx.empty(), torch.where(run[..., None], out_, ctx.empty())

    def adsorption(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:2154-2265``: the Freundlich adsorption and desorption of phosphorus (HYPE's),
        in WSIMOD's operand order.

        - ``pool_ = (ads.P + DI.P) / (area * M2_TO_KM2)``; where it is 0 (``empty``, ``:2176``)
          both VQIPs are empty and nothing changes;
        - ``smc_mm = smc * M_TO_MM``, ``conc_sol = ads.P * KG_TO_MG / (bulk_density *
          rooting_depth * area)`` and ``coeff = kfr * bulk_density * rooting_depth``;
        - where ``conc_sol <= 0`` (``shortcut``, ``:2191``) the equilibrium is ``kfr * pool_ /
          (smc_mm + coeff)``, and ``"adsorption_shortcut"`` is recorded where WSIMOD prints
          "Warning: soil partP <=0...";
        - otherwise Newton's method from ``x0 = exp((log(conc_sol) - log(kfr)) / nfr)``, unrolled
          to ``adsorption_nr_maxiter`` passes under a ``running`` mask (Y25). Pass ``j`` runs
          where ``abs(fxn) > limit`` (``newton:<j>/residual``, ``:2206``); it computes ``fxn =
          xn * smc_mm + coeff * xn ** nfr - pool_``, ``fprimxn = smc_mm + nfr * coeff * xn **
          (nfr - 1)`` and ``dx = fxn / fprimxn``, stops where ``abs(dx) < 0.000001 * xn``
          (``newton:<j>/step``, the ``break``), and otherwise sets ``xn = xn_1 = xn - dx``, or
          ``1e-10`` where that is not positive (``newton:<j>/positive``, ``:2217``; true is
          ``xn - dx > 0``). A stopped instance keeps its ``xn``, ``xn_1`` and ``fxn``. **The
          residual test lags one pass** (transcribed): it reads the ``fxn`` of the previous
          pass, the residual of the iterate before its last update, so a loop the residual ends
          takes one step more than the residual needs: the first two tests read the same
          ``f(x0)``, so it runs two passes at least, unless ``f(x0)`` is already within the
          limit, when none runs and the equilibrium is the first guess's.
          WSIMOD's last test, after the ``maxiter``-th pass, decides nothing and is not
          recorded. The equilibrium is ``kfr * xn_1 ** nfr``;
        - where ``abs(equi - conc_sol) > 1e-6`` (``change``, ``:2225``) the request ``P =
          (equi - conc_sol) * (1 - exp(-kadsdes)) * bulk_density * rooting_depth * (area *
          M2_TO_KM2)``. Above 0 (``adsorb``) it is extracted from the dissolved inorganic pool
          into the adsorbed one and leaves the tank as phosphate (``out_``,
          :meth:`_pull_tracked`); otherwise ``-P`` is extracted from the adsorbed pool into the
          dissolved inorganic one and forced into the tank (``in_``). The request's ``N = 0``
          moves too (``:2226``).

        WSIMOD's "freundlich flow adjusted" warning tests ``extracted - request > FA``
        (``:2242``, ``:2257``), which ``min`` makes ``<= 0``, so it never prints (L11); noodl
        records the evident ``request - extracted > FA`` as ``"freundlich_flow_adjusted"``
        (Y16). Every ``log``, ``exp`` and power is Python's, per instance, on an argument
        replaced by a safe one (1, or 0 for ``exp``) where its branch is not taken, so a masked,
        empty, shortcut or stopped instance never overflows, divides by zero or raises a
        negative number to a fractional power. ``adosorption_nr_limit`` (WSIMOD's spelling)
        enters only comparisons, so it has no gradient."""
        ctx, pool = self.ctx, self.nutrient_pool
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("adsorption")
        site = f"{self.site}/adsorption"
        area = self.area
        kfr, nfr = self._param("kfr"), self._param("nfr")
        bulk_density, rooting_depth = self._param("bulk_density"), self._param("rooting_depth")
        limit = self._param("adosorption_nr_limit")
        # :2171-2177
        ads_p, di_p = pool.storage[..., ADS, 1], pool.storage[..., DI, 1]
        pool_ = ads_p + di_p
        pool_ = pool_ / (area * M2_TO_KM2)
        empty = ctx.decide(f"{site}/empty", pool_, "==", 0.0, active=active, call=call)
        run = active & ~empty
        # :2180-2188
        smc_mm = self._smc() * M_TO_MM
        conc_sol = ads_p * KG_TO_MG / (bulk_density * rooting_depth * area)
        coeff = kfr * bulk_density * rooting_depth
        # :2191-2195 (smc_mm + coeff > 0: kfr, bulk_density and rooting_depth are refused <= 0)
        shortcut = run & ctx.decide(f"{site}/shortcut", conc_sol, "<=", 0.0, active=run, call=call)
        ctx.event("adsorption_shortcut", self.parent.name, shortcut)
        equi_shortcut = kfr * (pool_ / (smc_mm + coeff))
        # :2197-2221, unrolled (Y25); conc_sol > 0 where the log is taken
        newton = run & ~shortcut
        one = torch.ones_like(conc_sol)
        exponent = (log_python(torch.where(newton, conc_sol, one)) - log_python(kfr)) / nfr
        x0 = exp_python(torch.where(newton, exponent, torch.zeros_like(exponent)))
        xn = torch.where(newton, x0, one)
        fxn = xn * smc_mm + coeff * pow_python(xn, nfr) - pool_
        xn_1 = xn
        running = newton
        for j in range(self.adsorption_nr_maxiter):
            running = running & ctx.decide(
                f"{site}/newton:{j}/residual", fxn.abs(), ">", limit, active=running, call=call
            )
            # xn > 0 on every instance: x0 = exp(...), the update or 1e-10, and 1 elsewhere
            value = xn * smc_mm + coeff * pow_python(xn, nfr) - pool_
            slope = smc_mm + nfr * coeff * pow_python(xn, nfr - 1)
            dx = value / slope
            fxn = write(running, value, fxn)
            step = ctx.decide(
                f"{site}/newton:{j}/step", dx.abs(), "<", 0.000001 * xn, active=running, call=call
            )
            running = running & ~step
            update = xn - dx
            positive = ctx.decide(
                f"{site}/newton:{j}/positive", update, ">", 0.0, active=running, call=call
            )
            update = torch.where(positive, update, torch.full_like(update, 1e-10))
            xn_1 = write(running, update, xn_1)
            xn = write(running, update, xn)
        equi = torch.where(shortcut, equi_shortcut, kfr * pow_python(xn_1, nfr))
        # :2225-2238
        difference = equi - conc_sol
        change = run & ctx.decide(
            f"{site}/change", difference.abs(), ">", 1e-6, active=run, call=call
        )
        rate = 1 - exp_python(torch.where(change, -self._param("kadsdes"), torch.zeros_like(one)))
        adsdes = difference * rate
        request = adsdes * bulk_density * rooting_depth * (area * M2_TO_KM2)
        adsorb = change & ctx.decide(f"{site}/adsorb", request, ">", 0.0, active=change, call=call)
        desorb = change & ~adsorb
        zero = torch.zeros_like(request)
        # :2239-2251, adsorption (the request {N: 0, P})
        wanted = torch.stack(torch.broadcast_tensors(zero, request), dim=-1)
        adsorbed = pool.extract(DI, wanted, active=adsorb)
        pool.receive(ADS, adsorbed, active=adsorb)
        short = adsorb & ((request - adsorbed[..., 1]) > ctx.fa)
        out_ = self._with_columns(ctx.empty(), {"phosphate": adsorbed[..., 1]})
        out_ = self._pull_tracked(out_, active=adsorb)
        # :2252-2263, desorption
        wanted = torch.stack(torch.broadcast_tensors(zero, -request), dim=-1)
        desorbed = pool.extract(ADS, wanted, active=desorb)
        pool.receive(DI, desorbed, active=desorb)
        short = short | (desorb & ((-request - desorbed[..., 1]) > ctx.fa))
        ctx.event("freundlich_flow_adjusted", self.parent.name, short)
        in_ = self._with_columns(ctx.empty(), {"phosphate": desorbed[..., 1]})
        in_ = torch.where(desorb[..., None], in_, ctx.empty())
        self.push_storage(in_, force=True, active=desorb)
        return in_, torch.where(adsorb[..., None], out_, ctx.empty())


SURFACE_CLASSES["GrowingSurface"] = GrowingSurface
