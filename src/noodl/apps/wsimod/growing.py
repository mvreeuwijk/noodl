"""``GrowingSurface`` (``wsimod/nodes/land.py:1082-1572``): the crop calendar and the crop cover.

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
dissolved pools a second time (L8, reproduced: Y16, Y45). The soil processes after
``calculate_soil_temperature`` are later tasks (D20). A tank with no inorganic N gives the three
N species 0 in ``pull_storage``, where WSIMOD raises ``ZeroDivisionError`` (``:1460``; L5, Y18;
the decision ``S/pull_storage/no_din`` and the event ``"pull_storage_no_din"``).

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
dry, wet, fertiliser and manure calls in that order), and the pool's own
``S.nutrient_pool.<pool>/extract/clip:<N|P>``.
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

from noodl.apps.wsimod.constants import SURFACE_INPUTS
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
from noodl.apps.wsimod.vqip import Pollutants, extract_vqip, safe_den, v_change_vqip, write

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
    """``GrowingSurface`` (``land.py:1082-2312``): construction, overrides, the crop calendar and
    ``calc_crop_cover`` (see the module text).

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
            # D5: the pool is carried with the crop registers
            self.carried_fields = (*type(self).carried_fields, "nutrient_pool")

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
        """``nutrient_pool`` is the pool's storage; anything else is an attribute."""
        if field == "nutrient_pool" and self.nutrient_pool is not None:
            self.nutrient_pool.storage = value
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


SURFACE_CLASSES["GrowingSurface"] = GrowingSurface
