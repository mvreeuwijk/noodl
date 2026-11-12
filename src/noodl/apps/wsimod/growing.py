"""``GrowingSurface`` (``wsimod/nodes/land.py:1082-1572``): the crop calendar and the crop cover.

A growing surface is a :class:`~noodl.apps.wsimod.land.PerviousSurface` whose physical depth is its
``rooting_depth`` (``:1192``, ``:1251``), with a sowing and harvest calendar, a crop factor, a crop
and ground cover and a water-stress coefficient. ``calc_crop_cover`` (``:1514-1572``) runs first in
the surface's ``inflows`` (``:1275``) and writes the day's ``et0_coefficient``
(``crop_factor * Ks``), which ``ihacres`` reads through ``get_climate``
(:meth:`~noodl.apps.wsimod.land.PerviousSurface.crop_et0_coefficient`). The nutrient processes
(``nitrate`` among the pollutants) are later tasks; ``has_nutrients`` records the switch.

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
stress_floor,stress_zero_denominator}``, ``S`` being ``node:N/surfaces.<s>``.
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
from noodl.apps.wsimod.engine import dmax
from noodl.apps.wsimod.land import SURFACE_CLASSES, PerviousSurface, month_start
from noodl.apps.wsimod.vqip import Pollutants, safe_den, write

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import SurfaceSpec, WsimodNetwork
    from noodl.apps.wsimod.land import Land

__all__ = [
    "GROWING_DEFAULTS",
    "STRUCTURE_DEFAULTS",
    "CalendarDay",
    "GrowingSurface",
    "quick_interp",
]

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
        self.has_nutrients = "nitrate" in pollutants.names  # :1276
        self.calendar: list[CalendarDay] = []
        self.inflows.insert(0, self.calc_crop_cover)  # :1275

    # --- parameters -----------------------------------------------------------------------------
    def _set_derived(self, derived: Mapping[str, Tensor]) -> None:
        super()._set_derived(derived)
        # :1353-1365, :1342-1343
        self.TAW = self.field_capacity_m - self.wilting_point_m
        self.RAW = self.TAW * self._param("ET_depletion_factor")
        self.ground_cover_stages = _stages_tensor(self._param("ground_cover_max"))
        self.crop_cover_stages = _stages_tensor(self._param("crop_cover_max"))

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters, ``crop_factor_stages`` ``(..., S)`` among them (``S`` the length
        of ``crop_factor_stage_dates``). ``wilting_point`` must stay below ``field_capacity``
        (``TAW > 0``, L4)."""
        parameters = dict(parameters)
        stages = parameters.pop("crop_factor_stages", None)
        stages = self._stages_default if stages is None else _f64(stages)
        if stages.shape[-1:] != (len(self.crop_factor_stage_dates),):
            raise ValueError(
                f"{self.site}: crop_factor_stages has {stages.shape[-1]} entries but "
                f"crop_factor_stage_dates has {len(self.crop_factor_stage_dates)} (land.py:1547)"
            )
        super().bind(parameters)
        self.crop_factor_stages = stages
        if bool((self.TAW <= 0).any()):
            raise ValueError(
                f"{self.site}: wilting_point is not below field_capacity: TAW <= 0 "
                "(land.py:1360-1366, L4)"
            )

    @classmethod
    def validate_spec(
        cls, surface: SurfaceSpec, network: WsimodNetwork, where: str, *, pollutants: bool
    ) -> list[str]:
        """Refused by name: the structural parameters (:func:`_check_structure`) and, whenever
        ``nitrate`` is a pollutant of the network (in both modes, Y49), a missing fertiliser or
        manure variable (``land.py:1649-1710`` reads all six) or a missing first-of-the-month row
        on a run date. A quantity-only run lists the data as ignored (the caller does)."""
        values = {**cls.parameter_defaults, **STRUCTURE_DEFAULTS, **surface.parameters}
        _check_structure(values, where)
        if "nitrate" not in network.pollutants.names:
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


SURFACE_CLASSES["GrowingSurface"] = GrowingSurface
