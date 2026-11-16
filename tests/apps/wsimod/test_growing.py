"""``GrowingSurface``: construction, overrides, the crop calendar and ``calc_crop_cover`` (3c).

Unit tests against WSIMOD's own object, with no cases. Each WSIMOD-side test builds
``wsimod.nodes.land.GrowingSurface`` on a stub parent whose ``t`` is WSIMOD's ``to_datetime``
(the individual-files loader's date, which gives the day-of-year shift of L1).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import torch

import noodl.apps.wsimod.growing  # noqa: F401  (registers the class)
from noodl.apps.wsimod.classes import SURFACE_SCHEMAS
from noodl.apps.wsimod.config import NodeSpec, SurfaceSpec, read_wsimod_config
from noodl.apps.wsimod.constants import SURFACE_DEPOSITION, SURFACE_INPUTS
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.growing import (
    GROWING_DEFAULTS,
    NUTRIENT_POLLUTANTS,
    STRUCTURE_DEFAULTS,
    GrowingSurface,
    quick_interp,
)
from noodl.apps.wsimod.land import SURFACE_CLASSES, Land, month_start, surface_derived
from noodl.apps.wsimod.nutrient_pool import ADS, DI, DO, FAST, POOLS
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants

P = QUANTITY_ONLY
FA = 1e-11
TRUE = torch.tensor(True)
F64 = torch.float64
EVENT = "crop_water_stress_zero_denominator"


def _dates(start: str, days: int) -> list[datetime]:
    first = datetime.strptime(start, "%Y-%m-%d")
    return [first + timedelta(days=k) for k in range(days)]


def _labels(dates) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in dates]


def make_land(**parameters) -> Land:
    spec = SurfaceSpec("g", "GrowingSurface", parameters)
    return make_node(NodeSpec("land", "Land", "Land", {}, {}, [spec]), None, P)


def growing(*, dates=None, **parameters) -> GrowingSurface:
    surface = make_land(**parameters).surfaces[0]
    dates = dates if dates is not None else _dates("2003-01-01", 1096)
    surface.set_dates(dates, _labels(dates))
    return surface


def at(land_or_surface, day: int) -> Ctx:
    surface = getattr(land_or_surface, "surfaces", [land_or_surface])[0]
    ctx = Ctx(P, FA, torch.Size(()), day=day, node_names=["land"], record_decisions=True)
    surface.parent.ctx = ctx
    return ctx


def volume(v: float) -> torch.Tensor:
    return P.to_vqip({"volume": v})


def _wsimod_growing(storage: float = 0.0, **kwargs):
    pytest.importorskip("wsimod")
    from wsimod.core import constants
    from wsimod.nodes.land import GrowingSurface as WsimodGrowing

    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()
    constants.POLLUTANTS = []
    constants.ADDITIVE_POLLUTANTS = []
    constants.NON_ADDITIVE_POLLUTANTS = []
    parent = SimpleNamespace()
    return WsimodGrowing(
        parent=parent,
        surface="g",
        pollutant_load={},
        initial_storage={"volume": storage},
        **kwargs,
    )


def _wsimod_day(surface, date: datetime) -> None:
    from wsimod.orchestration.model import to_datetime

    surface.parent.t = to_datetime(date.strftime("%Y-%m-%d"))


# --- quick_interp -------------------------------------------------------------------------------
def test_quick_interp_equals_wsimods():
    """``land.py:1495-1512``, bitwise, on random sorted lists, the default calendar
    ``[0, 1, 365, 366, 365]`` and the wrap ``xp = [1, 365]`` at ``x = 1`` (L23)."""
    from bisect import bisect_left

    w = _wsimod_growing()
    rng = random.Random(7)
    cases = []
    for _ in range(30):
        xp = sorted(rng.uniform(0, 365) for _ in range(6))
        cases.append((xp, [rng.uniform(0, 2) for _ in range(6)]))
    cases.append(([0, 1, 365, 366, 365], [0, 0, 0.9, 0, 0]))
    cases.append(([1, 365], [0.3, 1]))
    for xp, yp in cases:
        for x in (1, 2, 59, 100, 250, 364, 365):
            i = bisect_left(xp, x)
            if i >= len(xp):
                continue
            theirs = w.quick_interp(x, xp, yp)
            ours = quick_interp(x, xp, torch.tensor(yp, dtype=F64), i)
            assert float(ours) == theirs, (xp, yp, x)
    wrap = quick_interp(1, [1, 365], torch.tensor([0.3, 1.0], dtype=F64), 0)
    assert float(wrap) == 0.30000000000000004 == w.quick_interp(1, [1, 365], [0.3, 1])


def test_gradcheck_on_quick_interp():
    """Spec, Verification 4: the gradient with respect to ``yp`` (piecewise linear)."""
    xp = [0, 80, 160, 240, 365]
    for x, i in ((50, 1), (200, 4), (365, 4), (0, 0)):
        yp = torch.tensor([0.4, 0.7, 1.15, 0.8, 0.4], dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda y, x=x, i=i: quick_interp(x, xp, y, i), (yp,))


# --- calc_crop_cover against WSIMOD, every day --------------------------------------------------
_STAGES5 = {
    "crop_factor_stage_dates": [0, 80, 160, 240, 365],
    "crop_factor_stages": [0.4, 0.7, 1.15, 0.8, 0.4],
}
_CALENDARS = {
    "spring": {"sowing_day": 90, "harvest_day": 250, **_STAGES5},
    "autumn": {"sowing_day": 280, "harvest_day": 200, **_STAGES5},
    "default": {"sowing_day": 1, "harvest_day": 365, **_STAGES5},
    "wrap": {
        "sowing_day": 90,
        "harvest_day": 250,
        "crop_factor_stage_dates": [1, 365],
        "crop_factor_stages": [0.3, 1.0],
    },
}


@pytest.mark.parametrize("name", list(_CALENDARS))
def test_calc_crop_cover_equals_wsimods_every_day(name):
    """``land.py:1514-1572`` on every day of 2003-2005 with ``ET_depletion_factor = 0.5`` and a
    soil moisture drawn either side of ``RAW``: ``days_after_sow`` (-1 for ``None``), both covers,
    ``crop_factor`` and ``et0_coefficient`` are equal every day."""
    parameters = {**_CALENDARS[name], "ET_depletion_factor": 0.5}
    w = _wsimod_growing(**parameters)
    dates = _dates("2003-01-01", 1096)
    ours = growing(dates=dates, **parameters)
    land = ours.parent
    rng = random.Random(11)
    for k, date in enumerate(dates):
        smc = rng.uniform(0.0, 0.4)
        _wsimod_day(w, date)
        w.storage["volume"] = smc
        w.calc_crop_cover()
        at(land, k)
        ours.storage = volume(smc)
        ours.calc_crop_cover(active=TRUE)
        assert float(ours.days_after_sow) == (-1 if w.days_after_sow is None else w.days_after_sow)
        for attribute in ("crop_cover", "ground_cover", "crop_factor", "et0_coefficient"):
            assert float(getattr(ours, attribute)) == float(getattr(w, attribute)), (k, attribute)


def test_a_wsimod_object_and_noodls_start_alike():
    w = _wsimod_growing()
    ours = growing()
    assert w.days_after_sow is None and float(ours.days_after_sow) == -1
    assert (w.crop_cover, w.ground_cover, w.crop_factor, w.et0_coefficient) == (0, 0, 0, 1)
    assert [float(getattr(ours, a)) for a in ("crop_cover", "ground_cover", "crop_factor")] == [
        0
    ] * 3
    assert float(ours.et0_coefficient) == 1
    assert ours.inflows[0] == ours.calc_crop_cover  # :1275
    assert ours.harvest_sow_calendar == w.harvest_sow_calendar == [0, 1, 365, 366, 365]
    assert ours.autumn_sow is w.autumn_sow is False


def test_every_year_shifts_the_day_after_february():
    """L1: ``if t.is_leap_year:`` tests a bound method, so every year shifts after day 59."""
    dates = [
        datetime(2003, 2, 28),
        datetime(2003, 3, 1),
        datetime(2003, 12, 31),
        datetime(2004, 3, 1),
    ]
    surface = growing(dates=dates)
    assert [d.adjusted for d in surface.calendar] == [59, 59, 364, 60]
    assert [d.raw for d in surface.calendar] == [59, 60, 365, 61]
    w = _wsimod_growing()
    for date, day in zip(dates, surface.calendar, strict=True):
        _wsimod_day(w, date)
        assert bool(w.parent.t.is_leap_year)  # a bound method: truthy (L1)
        assert day.erosion_sine == math.sin(2 * w_pi() * ((w.parent.t.dayofyear - 70) / 365))


def w_pi() -> float:
    from wsimod.core import constants

    return constants.PI


def test_the_sowing_day_has_no_crop():
    """``days_after_sow = 0`` is false at ``:1550``: the covers are unchanged on the sowing day,
    and interpolated from the next."""
    dates = _dates("2003-03-30", 5)  # raw 89, 90 (sowing), 91, 92, 93
    surface = growing(dates=dates, sowing_day=90, harvest_day=250)
    surface.crop_cover = torch.tensor(0.5, dtype=F64)
    surface.ground_cover = torch.tensor(0.25, dtype=F64)
    seen = []
    for k in range(5):
        at(surface.parent, k)
        surface.storage = volume(0.3)
        surface.calc_crop_cover(active=TRUE)
        seen.append(
            (float(surface.days_after_sow), float(surface.crop_cover), float(surface.ground_cover))
        )
    assert seen[0] == (-1, 0.5, 0.25) and seen[1] == (0, 0.5, 0.25)
    # raw 91 is adjusted 90: the first segment [0, 90] has y = 0; raw 92 is on the way up
    assert seen[2] == (1, 0.0, 0.0)
    assert seen[3][0] == 2 and seen[3][1] == pytest.approx(0.9 / 160, rel=1e-14)
    assert seen[4][1] > seen[3][1] > 0


def test_an_autumn_calendar_has_no_cover_all_year():
    """L2, Y15: ``[0, 280, 200, 201, 365]`` is unsorted, ``bisect_left`` finds only zero
    segments: the cover profile over days 1-365 peaks at 0.0 for (280, 200) and at 0.9 for
    (90, 250). A run reaches the profile's peak only before the harvest, so the state's maximum
    is the last interpolated day (raw 249): WSIMOD's own, checked on both."""
    from bisect import bisect_left

    dates = _dates("2003-01-01", 365)
    for sowing, harvest, profile_peak in ((280, 200, 0.0), (90, 250, 0.9)):
        surface = growing(dates=dates, sowing_day=sowing, harvest_day=harvest)
        stages = surface.crop_cover_stages
        profile = [
            float(
                quick_interp(
                    x,
                    surface.harvest_sow_calendar,
                    stages,
                    bisect_left(surface.harvest_sow_calendar, x),
                )
            )
            for x in range(1, 366)
        ]
        assert max(profile) == pytest.approx(profile_peak, abs=1e-15)
        w = _wsimod_growing(sowing_day=sowing, harvest_day=harvest)
        peak = theirs = 0.0
        for k, date in enumerate(dates):
            at(surface.parent, k)
            surface.storage = volume(0.3)
            surface.calc_crop_cover(active=TRUE)
            _wsimod_day(w, date)
            w.storage["volume"] = 0.3
            w.calc_crop_cover()
            peak, theirs = max(peak, float(surface.crop_cover)), max(theirs, w.crop_cover)
        assert peak == theirs
        assert (peak == 0.0) == (sowing == 280)


def test_the_harvest_resets_the_crop_and_the_next_sowing_starts_again():
    dates = _dates("2003-01-01", 365)
    surface = growing(dates=dates, sowing_day=90, harvest_day=250)
    states = {}
    for k in range(365):
        at(surface.parent, k)
        surface.storage = volume(0.3)
        surface.calc_crop_cover(active=TRUE)
        states[dates[k].timetuple().tm_yday] = (
            float(surface.days_after_sow),
            float(surface.crop_cover),
            float(surface.ground_cover),
        )
    assert states[249][0] == 159 and states[249][1] > 0
    assert states[250] == (-1.0, 0.0, 0.0)  # harvest: None, both covers 0
    assert states[260] == (-1.0, 0.0, 0.0)  # and nothing until the next sowing


# --- the stress coefficient ---------------------------------------------------------------------
def test_the_stress_zero_denominator_gives_zero():
    """L3, Y17: ``ET_depletion_factor = 1`` and the soil at its wilting point. WSIMOD raises
    ``ZeroDivisionError`` at ``land.py:1566``; noodl's ``Ks = 0``, with the event."""
    w = _wsimod_growing(storage=0.1)
    _wsimod_day(w, datetime(2003, 6, 1))
    with pytest.raises(ZeroDivisionError):
        w.calc_crop_cover()
    surface = growing(dates=[datetime(2003, 6, 1)])
    ctx = at(surface.parent, 0)
    surface.storage = volume(0.1)  # wilting point 0.12: rooting depth 1
    surface.calc_crop_cover(active=TRUE)
    assert float(surface.et0_coefficient) == 0.0
    assert float(surface.crop_factor) == 1.0  # the factor itself is not zero
    assert bool(ctx.events[EVENT]["land"])
    sites = {d.site.rsplit("/", 1)[1]: bool(d.outcome) for d in ctx.decisions}
    assert sites["stress_zero_denominator"] and not sites["unstressed"]
    # above RAW = TAW nothing is stressed and the event is silent
    ctx = at(surface.parent, 0)
    surface.storage = volume(0.2)
    surface.calc_crop_cover(active=TRUE)
    assert float(surface.et0_coefficient) == 1.0
    assert not bool(ctx.events.get(EVENT, {}).get("land", torch.tensor(False)))


def test_stress_either_side_of_raw():
    """``:1560-1568`` with exact binary values: ``fc_m = 0.5``, ``wp_m = 0.25``, ``p = 0.5``,
    ``RAW = 0.125``. Just below ``RAW`` the coefficient is 1; at ``RAW`` it is
    ``max(0, (TAW - RAW) / ((1 - p) * TAW))``, which is 1 too; deeper it falls to 0."""
    surface = growing(
        dates=[datetime(2003, 6, 1)],
        total_porosity=0.9,
        field_capacity=0.5,
        wilting_point=0.25,
        ET_depletion_factor=0.5,
    )
    assert float(surface.TAW) == 0.25 and float(surface.RAW) == 0.125
    w = _wsimod_growing(
        total_porosity=0.9, field_capacity=0.5, wilting_point=0.25, ET_depletion_factor=0.5
    )
    _wsimod_day(w, datetime(2003, 6, 1))
    results = []
    for smc in (0.375 + 2**-20, 0.375, 0.3125, 0.25):
        ctx = at(surface.parent, 0)
        surface.storage = volume(smc)
        surface.calc_crop_cover(active=TRUE)
        w.storage["volume"] = smc
        w.calc_crop_cover()
        assert float(surface.et0_coefficient) == w.et0_coefficient
        results.append(float(surface.et0_coefficient))
        assert bool(next(d for d in ctx.decisions if d.site.endswith("/unstressed")).outcome) == (
            smc > 0.375
        )
    assert results == [1.0, 1.0, 0.5, 0.0]


# --- refusals -----------------------------------------------------------------------------------
def _first_date_raising(upto: int, stage_dates, start="2003-10-01") -> str:
    from bisect import bisect_left

    for date in _dates(start, upto):
        raw = date.timetuple().tm_yday
        adjusted = raw - 1 if raw > 59 else raw
        if bisect_left(stage_dates, adjusted) >= len(stage_dates):
            return date.strftime("%Y-%m-%d")
    raise AssertionError("no date raises")


_RAISES_ON = _first_date_raising(90, [0, 300])
_CALENDAR_REFUSALS = [
    ({"crop_factor_stage_dates": [0, 365], "crop_factor_stages": [1, 1, 1]}, "crop_factor_stages"),
    (
        {"crop_factor_stage_dates": [0, 300], "crop_factor_stages": [1, 1]},
        f"crop_factor_stage_dates.*{_RAISES_ON}",
    ),
    (
        {"crop_factor_stage_dates": [365, 365], "crop_factor_stages": [1, 1]},
        "divides by zero.*2003-10-01",
    ),
    ({"sowing_day": 90.5}, "sowing_day"),
    ({"harvest_day": 250.5}, "harvest_day"),
    ({"adsorption_nr_maxiter": -1}, "adsorption_nr_maxiter"),
    ({"adsorption_nr_maxiter": 2.5}, "adsorption_nr_maxiter"),
]


@pytest.mark.parametrize(("parameters", "match"), _CALENDAR_REFUSALS)
def test_calendar_refusals(parameters, match):
    """D7: the refusals name the parameter, and for the dates the first date that raises."""
    dates = _dates("2003-10-01", 90)
    with pytest.raises(ValueError, match=match):
        make_land(**parameters).surfaces[0].set_dates(dates, _labels(dates))
    assert _RAISES_ON == "2003-10-29"


def test_integral_floats_and_duplicate_stage_dates_are_accepted():
    dates = _dates("2003-01-01", 1096)
    growing(dates=dates, sowing_day=90.0, harvest_day=250.0, adsorption_nr_maxiter=20.0)
    stages = [1.0, 2.0, 3.0, 4.0]
    surface = growing(
        dates=dates, crop_factor_stage_dates=[0, 100, 100, 365], crop_factor_stages=stages
    )
    w = _wsimod_growing(crop_factor_stage_dates=[0, 100, 100, 365], crop_factor_stages=stages)
    for day in surface.calendar:
        theirs = w.quick_interp(day.adjusted, [0, 100, 100, 365], stages)
        ours = quick_interp(
            day.adjusted, [0, 100, 100, 365], torch.tensor(stages, dtype=F64), day.factor_index
        )
        assert float(ours) == theirs


_DATES = ["2000-01-01", "2000-01-02"]


def _mapping(surfaces, *, overrides=None, data=None, lists=None) -> dict:
    if data is None:
        data = {(v, d): 0.001 for v in ("precipitation", "et0", "temperature") for d in _DATES}
    mapping = {
        **(lists or {}),
        "dates": list(_DATES),
        "nodes": [{"name": "land", "type_": "Land", "surfaces": surfaces, "data_input_dict": data}],
    }
    if overrides is not None:
        mapping["overrides"] = {
            "nodes": {"land": {"type_": "Land", "name": "land", "surfaces": overrides}}
        }
    return mapping


_LISTS = {
    "pollutants": ["phosphate", "temperature"],
    "additive_pollutants": ["phosphate"],
    "non_additive_pollutants": ["temperature"],
}


def _validate(mapping, *, pollutants=False):
    network = read_wsimod_config(mapping)
    return Land.validate(network.nodes[0], network, pollutants=pollutants)


def _ignored(mapping, *, pollutants):
    """What a quantity-only run ignores, less the soil-temperature weights (3b's)."""
    return [i for i in _validate(mapping, pollutants=pollutants) if "soil_temp" not in i]


def _growing(**values):
    return {"type_": "GrowingSurface", "surface": "g", **values}


_GROWING_REFUSALS = [
    (_mapping([_growing(depth=1.0)], lists=_LISTS), "depth"),  # L17
    (_mapping([_growing()], overrides={"g": {"depth": 1.0}}, lists=_LISTS), "depth"),
    (_mapping([_growing(wilting_point=0.35)], lists=_LISTS), "wilting_point"),  # L4, Y44
    (_mapping([_growing(wilting_point=0.3)], lists=_LISTS), "wilting_point"),
    (
        _mapping([_growing()], overrides={"g": {"wilting_point": 0.31}}, lists=_LISTS),
        "wilting_point",
    ),
    (_mapping([_growing(area=0)], lists=_LISTS), "area"),
    (_mapping([_growing(total_porosity=0.3)], lists=_LISTS), "total_porosity"),  # inherited, D18
    (_mapping([_growing(rooting_depth=0.0)], lists=_LISTS), "rooting_depth|simulated depth"),
    (_mapping([_growing(sowing_day=90.5)], lists=_LISTS), "sowing_day"),
    (_mapping([_growing(crop_factor_stages=[1, 2, 3])], lists=_LISTS), "crop_factor_stages"),
    (
        _mapping([_growing()], overrides={"g": {"adsorption_nr_maxiter": -1}}, lists=_LISTS),
        "adsorption_nr_maxiter",
    ),
]


@pytest.mark.parametrize(("mapping", "match"), _GROWING_REFUSALS)
def test_growing_refusals(mapping, match):
    with pytest.raises(ValueError, match=match):
        _validate(mapping)


def test_a_growing_surface_is_accepted():
    assert _ignored(_mapping([_growing(rooting_depth=0.5)], lists=_LISTS), pollutants=False) == []
    assert _validate(_mapping([_growing()], lists=_LISTS), pollutants=True) == []


def test_rooting_depth_is_the_physical_depth():
    """``:1192``, ``:1251``: ``field_capacity_m = field_capacity * rooting_depth`` and the tank's
    depth is ``rooting_depth * total_porosity``."""
    spec = read_wsimod_config(_mapping([_growing(rooting_depth=0.6)])).nodes[0].surfaces[0]
    derived = surface_derived(spec)
    assert derived["field_capacity_m"] == 0.3 * 0.6 and derived["wilting_point_m"] == 0.12 * 0.6
    assert derived["depth"] == 0.6 * 0.4 and derived["capacity"] == 1 * (0.6 * 0.4)
    surface = make_land(rooting_depth=0.6).surfaces[0]
    assert float(surface.depth) == 0.6 * 0.4 and float(surface.field_capacity_m) == 0.3 * 0.6


def _data(variables, *, dates=("2000-01-01",)):
    return {(v, d): 0.001 for v in variables for d in dates}


def _forcing_data():
    return {(v, d): 0.001 for v in ("precipitation", "et0", "temperature") for d in _DATES}


# the eight pollutants a growing surface needs with nitrate (NUTRIENT_POLLUTANTS), and one more
_NITRATE = {
    "pollutants": [*NUTRIENT_POLLUTANTS, "cod"],
    "additive_pollutants": [n for n in (*NUTRIENT_POLLUTANTS, "cod") if n != "temperature"],
    "non_additive_pollutants": ["temperature"],
}


@pytest.mark.parametrize("pollutants", [True, False])
def test_fertiliser_and_manure_data_are_needed_with_nitrate(pollutants):
    """Y49: WSIMOD reads the six variables whenever ``nitrate`` is in its ``POLLUTANTS``, in either
    of noodl's modes."""
    with pytest.raises(ValueError, match="nhx-fertiliser"):
        _validate(_mapping([_growing()], lists=_NITRATE), pollutants=pollutants)
    partial = _data(v for v in SURFACE_INPUTS if v != "srp-manure")
    with pytest.raises(ValueError, match="srp-manure"):
        _validate(
            _mapping([_growing(data_input_dict=partial)], lists=_NITRATE), pollutants=pollutants
        )
    complete = _mapping([_growing(data_input_dict=_data(SURFACE_INPUTS))], lists=_NITRATE)
    ignored = _ignored(complete, pollutants=pollutants)
    # with them, a quantity-only run lists the data as ignored
    assert ignored == ([] if pollutants else ["surfaces.g.data_input_dict"])


def test_a_missing_month_of_fertiliser_data_is_refused():
    other_month = _data(SURFACE_INPUTS, dates=("2000-02-01",))
    with pytest.raises(ValueError, match=r"nhx-fertiliser.*2000-01-01"):
        _validate(_mapping([_growing(data_input_dict=other_month)], lists=_NITRATE))


def test_no_fertiliser_data_are_needed_without_nitrate():
    assert _validate(_mapping([_growing()], lists=_LISTS), pollutants=True) == []
    assert _ignored(_mapping([_growing()], lists=_LISTS), pollutants=False) == []
    # and a surface that is not a growing one still refuses them
    pervious = {
        "type_": "PerviousSurface",
        "surface": "g",
        "data_input_dict": _data(SURFACE_INPUTS),
    }
    with pytest.raises(ValueError, match="growing surface"):
        _validate(_mapping([pervious], lists=_LISTS))


def test_a_growing_et0_coefficient_is_unread():
    """L22, Y39: the configured or overridden ``et0_coefficient`` is never read (``:1265``,
    ``:1570``). It is listed in ``model.unread``, and a run is blind to it."""
    network = read_wsimod_config(
        _mapping([_growing(et0_coefficient=0.9)], overrides={"g": {"et0_coefficient": 0.1}})
    )
    assert "land.surfaces.g.et0_coefficient" in network.unread
    plain = read_wsimod_config(_mapping([{"type_": "PerviousSurface", "surface": "p"}]))
    assert not [u for u in plain.unread if "et0_coefficient" in u]
    results = []
    for et0_coefficient in (0.9, 0.1):
        surface = make_land(et0_coefficient=et0_coefficient).surfaces[0]
        land = surface.parent
        surface.set_dates([datetime(2003, 6, 1)], ["2003-06-01"])
        at(land, 0)
        surface.storage = volume(0.2)
        land.bind(
            {
                "precipitation": torch.tensor(0.0, dtype=F64),
                "et0": torch.tensor(0.004, dtype=F64),
                "temperature": torch.tensor(10.0, dtype=F64),
            }
        )
        land.run(active=TRUE)
        results.append((float(surface.evaporation[0]), float(surface.storage[0])))
        assert float(surface.crop_et0_coefficient()) == 1.0  # the day's: Ks 1 * crop_factor 1
    assert results[0] == results[1] and results[0][0] > 0


def test_ihacres_reads_the_crop_coefficient_through_get_climate():
    """``land.py:905-911``: ``et0 * et0_coefficient``, the attribute ``calc_crop_cover`` wrote."""
    surface = make_land(crop_factor_stages=[0.5, 0.5]).surfaces[0]
    land = surface.parent
    surface.set_dates([datetime(2003, 6, 1)], ["2003-06-01"])
    at(land, 0)
    surface.storage = volume(0.25)
    land.bind(
        {
            "precipitation": torch.tensor(0.001, dtype=F64),
            "et0": torch.tensor(0.004, dtype=F64),
            "temperature": torch.tensor(10.0, dtype=F64),
        }
    )
    surface.calc_crop_cover(active=TRUE)
    precipitation, evaporation = surface.get_climate()
    assert float(precipitation) == 0.001
    assert float(evaporation) == 0.004 * float(surface.et0_coefficient) == 0.004 * 0.5


# --- overrides ----------------------------------------------------------------------------------
_OVERRIDE_VALUES = {
    "ET_depletion_factor": 0.6,
    "crop_cover_max": 0.8,
    "ground_cover_max": 0.2,
    "crop_factor_stages": [0.5, 1.2],
    "crop_factor_stage_dates": [0, 300],
    "sowing_day": 100,
    "harvest_day": 200,
    "satact": 0.5,
    "thetaupp": 0.13,
    "thetalow": 0.07,
    "thetapow": 2,
    "uptake1": 14,
    "uptake2": 2,
    "uptake3": 0.03,
    "uptake_PNratio": 0.2,
    "erodibility": 0.003,
    "sreroexp": 1.3,
    "cohesion": 2,
    "slope": 6,
    "srfilt": 0.6,
    "macrofilt": 0.02,
    "limpar": 0.6,
    "exppar": 2.4,
    "hsatINs": 2,
    "denpar": 0.016,
    "adosorption_nr_limit": 0.0001,
    "adsorption_nr_maxiter": 10,
    "kfr": 150.5,
    "nfr": 0.4,
    "kadsdes": 0.04,
    "bulk_density": 1200,
    "rooting_depth": 0.7,
}


def test_the_override_names_are_wsimods_31_plus_rooting_depth():
    own = set(_OVERRIDE_VALUES)
    assert len(own) == 32
    pervious = SURFACE_SCHEMAS["PerviousSurface"].override_keys
    assert SURFACE_SCHEMAS["GrowingSurface"].override_keys == pervious - {"depth"} | own
    # the 25 batched attributes, ET_depletion_factor and rooting_depth are the drivers (D17);
    # the five structural parameters are Python values
    assert len(GROWING_DEFAULTS) == 27
    assert set(GROWING_DEFAULTS) | set(STRUCTURE_DEFAULTS) == own


@pytest.mark.parametrize("name", list(_OVERRIDE_VALUES))
def test_growing_overrides_equal_wsimods(name):
    """Each override applied to WSIMOD's object by ``apply_overrides`` and to noodl's spec by the
    reader: the attributes, ``capacity``, ``field_capacity_m``, ``wilting_point_m``, the calendar,
    ``TAW`` and ``RAW`` are equal, and the derived values equal construction's bitwise."""
    value = _OVERRIDE_VALUES[name]
    w = _wsimod_growing(area=3.0)
    w.apply_overrides({name: value})
    spec = (
        read_wsimod_config(
            _mapping([_growing(area=3.0)], overrides={"g": {name: value}}, lists=_LISTS)
        )
        .nodes[0]
        .surfaces[0]
    )
    ours = make_node(NodeSpec("land", "Land", "Land", {}, {}, [spec]), None, P).surfaces[0]
    for attribute in {**GROWING_DEFAULTS, **STRUCTURE_DEFAULTS}:
        if attribute in ("crop_factor_stages",):
            assert ours.crop_factor_stages.tolist() == [float(x) for x in w.crop_factor_stages]
        elif attribute == "crop_factor_stage_dates":
            assert ours.crop_factor_stage_dates == w.crop_factor_stage_dates
        elif attribute in ("sowing_day", "harvest_day", "adsorption_nr_maxiter"):
            assert getattr(ours, attribute) == getattr(w, attribute)
        else:
            assert float(ours._param(attribute)) == getattr(w, attribute), attribute
    assert float(ours.capacity) == w.capacity
    assert float(ours.depth) == w.depth
    assert float(ours.field_capacity_m) == w.field_capacity_m
    assert float(ours.wilting_point_m) == w.wilting_point_m
    assert float(ours.TAW) == w.total_available_water
    assert float(ours.RAW) == w.readily_available_water
    assert ours.harvest_sow_calendar == w.harvest_sow_calendar
    assert ours.autumn_sow == w.autumn_sow
    assert ours.ground_cover_stages.tolist() == [float(x) for x in w.ground_cover_stages]
    assert ours.crop_cover_stages.tolist() == [float(x) for x in w.crop_cover_stages]
    # the derived values equal construction's: build both with the value as a constructor keyword
    if name == "rooting_depth":
        built = _wsimod_growing(area=3.0, rooting_depth=value)
        assert (w.capacity, w.depth, w.field_capacity_m, w.wilting_point_m) == (
            built.capacity,
            built.depth,
            built.field_capacity_m,
            built.wilting_point_m,
        )
        spec = read_wsimod_config(_mapping([_growing(area=3.0, rooting_depth=value)])).nodes[0]
        built_ours = make_node(spec, None, P).surfaces[0]
        for attribute in ("capacity", "depth", "field_capacity_m", "wilting_point_m", "TAW", "RAW"):
            assert torch.equal(getattr(ours, attribute), getattr(built_ours, attribute))


def test_the_class_is_registered_and_is_a_growing_surface():
    assert SURFACE_CLASSES["GrowingSurface"] is GrowingSurface
    assert GrowingSurface.is_growing
    assert "rooting_depth" in GrowingSurface.parameter_defaults
    assert "depth" not in GrowingSurface.parameter_defaults
    assert "et0_coefficient" not in GrowingSurface.parameter_defaults


def test_a_batch_of_instances_runs_calc_crop_cover_under_a_mask():
    """Each instance has its own state; an inactive one is untouched."""
    surface = growing(dates=[datetime(2003, 4, 1)], sowing_day=91)
    ctx = Ctx(P, FA, torch.Size((3,)), day=0, node_names=["land"], record_decisions=True)
    surface.parent.ctx = ctx
    surface.storage = P.to_vqip({"volume": 0.3}, shape=(3,))
    surface.calc_crop_cover(active=torch.tensor([True, False, True]))
    assert surface.days_after_sow.tolist() == [0.0, -1.0, 0.0]
    assert surface.crop_factor.tolist() == [1.0, 0.0, 1.0]


def test_every_surface_function_has_a_part_or_none():
    """``Surface.run`` raises ``KeyError`` for a function in neither table (D11)."""
    from noodl.apps.wsimod.land import NO_PART, PART_NAMES

    land = make_land()
    for surface in make_land().surfaces + land.surfaces + _nutrient_land().surfaces:
        for f in surface.inflows + surface.processes + surface.outflows:
            assert f.__name__ in PART_NAMES or f.__name__ in NO_PART, f.__name__
    assert not NO_PART & set(PART_NAMES)


def test_validate_registers_the_growing_class_in_a_fresh_interpreter():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    code = (
        "from noodl.apps.wsimod.config import read_wsimod_config\n"
        "from noodl.apps.wsimod.land import Land\n"
        "dates = ['2000-01-01']\n"
        "data = {(v, d): 0.001 for v in ('precipitation', 'et0', 'temperature') for d in dates}\n"
        "n = read_wsimod_config({'pollutants': ['phosphate'], 'additive_pollutants': ['phosphate'],"
        " 'non_additive_pollutants': [], 'dates': dates, 'nodes': [{'name': 'land',"
        " 'type_': 'Land',"
        " 'data_input_dict': data, 'surfaces': [{'type_': 'GrowingSurface', 'surface': 'g'}]}]})\n"
        "Land.validate(n.nodes[0], n, pollutants=False)\n"
    )
    path = os.pathsep.join(p for p in (str(root / "src"), os.environ.get("PYTHONPATH")) if p)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=300,
        env={**os.environ, "PYTHONPATH": path},
    )
    assert done.returncode == 0, done.stderr


# --- the crop state, the drivers and the results (Task 5b) ---------------------------------------
_FIVE = {
    "crop_factor_stages": [0.3, 0.3, 1.2, 1.2, 0.3],
    "crop_factor_stage_dates": [0, 90, 160, 250, 365],
}


def _two_growing(**lists) -> dict:
    """Two growing surfaces, of 2 and 5 crop-factor stages, a pervious and an impervious one."""
    return _mapping(
        [
            {"type_": "GrowingSurface", "surface": "short", "initial_storage": 0.25},
            {
                "type_": "GrowingSurface",
                "surface": "long",
                "ET_depletion_factor": 0.5,
                "initial_storage": 0.25,
                **_FIVE,
            },
            {"type_": "PerviousSurface", "surface": "lawn", "et0_coefficient": 0.7},
            {"type_": "ImperviousSurface", "surface": "roof"},
        ],
        lists=lists or _LISTS,
    )


def test_crop_factor_stages_padding_is_inert():
    """Y33, D16: the driver is ``(n, S)``, ``S`` the class's longest list, the shorter row
    padded with 0; a non-zero padding entry is refused at bind, naming the surface and the
    index."""
    from noodl.apps.wsimod import build_model, initial_drivers

    model, state, drivers = build_model(read_wsimod_config(_two_growing()))
    key = "wsimod.GrowingSurface.crop_factor_stages"
    layout = model.wsimod_day.drivers[key]
    assert layout.labels == [("land", "short"), ("land", "long")]
    assert layout.columns == (0, 1, 2, 3, 4)
    assert drivers[key].tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0], [0.3, 0.3, 1.2, 1.2, 0.3]]
    assert layout.unit == "-"
    model.step(state, drivers, dt=86400.0, t=0.0)  # the padding is inert
    bad = drivers[key].clone()
    bad[0, 2] = 0.1
    padded = initial_drivers(model, values={key: bad})
    with pytest.raises(ValueError, match=r"surfaces\.short.*crop_factor_stages\[2\]"):
        model.step(state, padded, dt=86400.0, t=0.0)


@pytest.mark.parametrize("pollutants", [True, False])
def test_the_crop_state_exists_in_quantity_only(pollutants):
    """D4: three flat keys, one row per growing surface, in both modes (the calendar drives the
    volumes); ``days_after_sow`` starts at -1 (``None``), the covers at 0. They are not water."""
    from noodl.apps.wsimod import build_model, wsimod_run

    model, state, drivers = build_model(read_wsimod_config(_two_growing()), pollutants=pollutants)
    day = model.wsimod_day
    rows = [("land", "surfaces.short"), ("land", "surfaces.long")]
    for key, start, unit in (
        ("wsimod.days_after_sow", -1.0, "d"),
        ("wsimod.crop_cover", 0.0, "-"),
        ("wsimod.ground_cover", 0.0, "-"),
    ):
        assert day.state_labels(key) == rows
        assert state[key].tolist() == [start, start]
        assert day.key_labels[key].unit == unit
    results = wsimod_run(model, state, drivers, days=2)
    assert float(results.mass_balance.residual.abs().max()) <= 1e-12


def test_the_crop_state_round_trips_through_the_state():
    """Y5: the end of the sowing day holds ``days_after_sow = 0`` and the next day's ``1`` and a
    cover; a fresh model warm-started from the first day's state gives the second day's state
    bitwise."""
    from noodl.apps.wsimod import build_model

    mapping = _two_growing()
    mapping["nodes"][0]["surfaces"][1]["sowing_day"] = 200
    model, state, drivers = build_model(read_wsimod_config(mapping))
    one = model.step(state, drivers, dt=86400.0, t=0.0)
    assert one["wsimod.days_after_sow"].tolist() == [0.0, -1.0]  # 2000-01-01 is day 1
    two = model.step(one, drivers, dt=86400.0, t=86400.0)
    assert two["wsimod.days_after_sow"].tolist() == [1.0, -1.0]
    assert float(two["wsimod.crop_cover"][0]) > 0.0 and float(two["wsimod.crop_cover"][1]) == 0.0
    fresh, _, _ = build_model(read_wsimod_config(mapping))
    again = fresh.step(one, drivers, dt=86400.0, t=86400.0)
    assert set(again) == set(two)
    assert all(torch.equal(again[k], two[k]) for k in two)


def test_step_cannot_leave_the_network_dates():
    """D7, Y51: ``model.step`` runs only on the configuration's dates (3a's refusal), so the
    calendar, built for every one of them, cannot be read past them."""
    from noodl.apps.wsimod import build_model

    network = read_wsimod_config(_two_growing())
    model, state, drivers = build_model(network)
    with pytest.raises(ValueError, match="is not a whole number of days inside the dates"):
        model.step(state, drivers, dt=86400.0, t=len(network.dates) * 86400.0)


def test_et0_coef_is_recorded_for_every_non_impervious_surface():
    """D23 (``model.py:1150``): ``et0_coef`` is a ``PerviousSurface``'s parameter and a growing
    surface's coefficient of the day (``Ks * crop_factor``); NaN on an impervious one.
    ``crop_factor`` (Y32) is NaN except on the growing surfaces."""
    from noodl.apps.wsimod import build_model, wsimod_run

    network = read_wsimod_config(_two_growing())
    model, state, drivers = build_model(network)
    results = wsimod_run(model, state, drivers, days=2)
    column = {name: i for i, (_, name) in enumerate(results.surface_names)}
    et0 = results.surface["et0_coef"]
    crop = results.surface["crop_factor"]
    assert et0[:, column["lawn"]].tolist() == [0.7, 0.7]
    assert torch.isnan(et0[:, column["roof"]]).all()
    assert torch.isnan(crop[:, column["lawn"]]).all() and torch.isnan(crop[:, column["roof"]]).all()
    # the soil is above RAW on both: Ks = 1 and et0_coef is the day's crop factor, not the
    # PerviousSurface default of 0.5
    stages = torch.tensor(_FIVE["crop_factor_stages"], dtype=F64)
    assert crop[:, column["short"]].tolist() == [1.0, 1.0]
    assert crop[:, column["long"]].tolist() == [
        float(quick_interp(d, _FIVE["crop_factor_stage_dates"], stages, 1)) for d in (1, 2)
    ]
    for name in ("short", "long"):
        assert torch.equal(et0[:, column[name]], crop[:, column[name]])


# the unit of every growing parameter (WSIMOD's comments, land.py:1170-1250)
_GROWING_UNITS = {
    "ET_depletion_factor": "-", "rooting_depth": "m", "crop_factor_stages": "-",
    "crop_cover_max": "-", "ground_cover_max": "-", "satact": "-", "thetaupp": "-",
    "thetalow": "-", "thetapow": "-", "uptake1": "g/m2/y", "uptake2": "-", "uptake3": "1/d",
    "uptake_PNratio": "-", "erodibility": "g*d/(J*mm)", "sreroexp": "-", "cohesion": "kPa",
    "slope": "-", "srfilt": "-", "macrofilt": "-", "limpar": "-", "exppar": "-",
    "hsatINs": "mg/l", "denpar": "-", "adosorption_nr_limit": "-", "kfr": "l/kg", "nfr": "-",
    "kadsdes": "1/d", "bulk_density": "kg/m3",
}  # fmt: skip


def test_every_growing_parameter_has_its_unit():
    """D6, W20: each of the 27 growing parameters and ``crop_factor_stages`` has WSIMOD's
    unit on its driver ``"wsimod.GrowingSurface.<p>"``."""
    from noodl.apps.wsimod import build_model

    assert set(_GROWING_UNITS) == {*GROWING_DEFAULTS, "crop_factor_stages"}
    model, _, _ = build_model(read_wsimod_config(_two_growing()))
    drivers = model.wsimod_day.drivers
    for name, unit in _GROWING_UNITS.items():
        assert drivers[f"wsimod.GrowingSurface.{name}"].unit == unit, name


def test_store_paths_of_registers_the_growing_class_in_a_fresh_interpreter():
    """``Land.store_paths_of`` reads ``SURFACE_CLASSES`` after ``_register_all``."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    code = (
        "from noodl.apps.wsimod.land import Land\n"
        "paths = Land.store_paths_of([('GrowingSurface', 'g')])\n"
        "assert 'surfaces.g.days_after_sow' in paths, paths\n"
    )
    path = os.pathsep.join(p for p in (str(root / "src"), os.environ.get("PYTHONPATH")) if p)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=300,
        env={**os.environ, "PYTHONPATH": path},
    )
    assert done.returncode == 0, done.stderr


# --- the nutrient pool on the surface (Task 6) ---------------------------------------------------
NUT = Pollutants(
    tuple(_NITRATE["pollutants"]),
    tuple(_NITRATE["additive_pollutants"]),
    tuple(_NITRATE["non_additive_pollutants"]),
)
_KEYS = ("volume", *NUT.names)
# random values, so that a sum taken in another order (association) is another float
_SOIL_KEYS = ("phosphate", "ammonia", "nitrate", "nitrite", "org-nitrogen", "org-phosphorus")
_SOIL = {key: random.Random(3).uniform(0.0, 3.0) * (i + 1) / 7 for i, key in enumerate(_SOIL_KEYS)}


def _random_tank(rng, *, volume=None) -> dict:
    """A tank of ``NUT`` with random, order-sensitive masses (inorganic N > 0)."""
    tank = {n: rng.uniform(1e-3, 0.5) for n in NUT.additive}
    tank.update(volume=rng.uniform(0.5, 3.0) if volume is None else volume, temperature=9.5)
    return tank


def _random_soil(rng) -> dict:
    return {key: rng.uniform(0.0, 3.0) for key in _SOIL_KEYS}


def _nutrient_land(*, data=None, **parameters) -> Land:
    data = dict(data or {})
    spec = SurfaceSpec(
        "g",
        "GrowingSurface",
        parameters,
        data,
        constructed_variables=frozenset(v for v, _ in data),
        constructed_load=(),
    )
    return make_node(NodeSpec("land", "Land", "Land", {}, {}, [spec]), None, NUT)


def _nut_ctx(land, *, day: int = 0) -> Ctx:
    ctx = Ctx(NUT, FA, torch.Size(()), day=day, node_names=["land"], record_decisions=True)
    land.ctx = ctx
    return ctx


def _wsimod_nutrient(storage, *, data=None, area=1.0, **kwargs):
    """WSIMOD's ``GrowingSurface`` with ``nitrate`` among its pollutants (``NUT``)."""
    pytest.importorskip("wsimod")
    from wsimod.core import constants
    from wsimod.nodes.land import GrowingSurface as WsimodGrowing
    from wsimod.orchestration.model import to_datetime

    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()
    constants.POLLUTANTS = list(NUT.names)
    constants.ADDITIVE_POLLUTANTS = list(NUT.additive)
    constants.NON_ADDITIVE_POLLUTANTS = list(NUT.non_additive)
    keyed = {(v, to_datetime(d.strftime("%Y-%m-%d"))): x for (v, d), x in (data or {}).items()}
    return WsimodGrowing(
        parent=SimpleNamespace(),
        surface="g",
        area=area,
        pollutant_load={},
        data_input_dict=keyed,
        initial_storage={**dict.fromkeys(_KEYS, 0.0), **storage},
        **kwargs,
    )


def _vector(vqip) -> torch.Tensor:
    return torch.tensor([float(vqip[k]) for k in _KEYS], dtype=F64)


def _their_pools(theirs) -> torch.Tensor:
    return torch.tensor(
        [[float(pool.storage[n]) for n in "NP"] for pool in theirs.nutrient_pool.pools], dtype=F64
    )


def _set_their_pools(theirs, pools) -> None:
    for pool, (n, p) in zip(theirs.nutrient_pool.pools, pools, strict=True):
        pool.storage = {"N": n, "P": p}


def _n(vqip) -> float:
    return float(vqip[NUT.column("ammonia")] + vqip[NUT.column("nitrite")]) + float(
        vqip[NUT.column("nitrate")]
    )


def test_pull_storage_equals_wsimods():
    """``land.py:1433-1493``, bitwise, on 300 random tanks and pools: an empty tank (an empty
    reply), requests above ``V`` (capped at ``V``) and pools larger than the tank's columns
    (whose columns then go negative). The reply, the storage and the pools are WSIMOD's."""
    theirs = _wsimod_nutrient({"volume": 1.0})
    land = _nutrient_land()
    ours = land.surfaces[0]
    rng = random.Random(11)
    empty = capped = larger = 0
    for k in range(300):
        volume = 0.0 if k % 25 == 0 else rng.uniform(0.01, 2.0)
        storage = {n: rng.uniform(0.0, 0.5) for n in NUT.additive}
        storage.update(volume=volume, nitrate=rng.uniform(1e-3, 0.5), temperature=12.5)
        pools = [[rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)] for _ in POOLS]
        request = volume * rng.uniform(0.0, 1.5) if volume else rng.uniform(0.0, 1.0)
        empty += volume == 0
        capped += request > volume > 0
        larger += pools[DI][0] > storage["nitrate"] + storage["ammonia"] + storage["nitrite"]
        theirs.storage = dict(storage)
        _set_their_pools(theirs, pools)
        ours.storage = NUT.to_vqip(storage)
        ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
        _nut_ctx(land)
        reply = ours.pull_storage(NUT.to_vqip({"volume": request}), active=TRUE)
        their = theirs.pull_storage({"volume": request})
        assert torch.equal(reply, _vector(their)), k
        assert torch.equal(ours.storage, _vector(theirs.storage)), k
        assert torch.equal(ours.nutrient_pool.storage, _their_pools(theirs)), k
    assert empty and capped and larger


def test_pull_storage_without_din_gives_no_species():
    """L5, Y18: a tank with no inorganic N and ``V > 0``. WSIMOD raises ``ZeroDivisionError``
    at ``land.py:1460``; noodl gives the three species 0, the pool has still lost
    ``DI.N * prop``, and ``"pull_storage_no_din"`` is true."""
    import traceback

    storage = {"volume": 1.0, "phosphate": 0.2, "org-nitrogen": 0.1, "org-phosphorus": 0.05}
    storage.update(solids=0.3, temperature=10.0)
    pools = [[0.0, 0.0], [0.0, 0.0], [0.4, 0.2], [0.1, 0.05], [0.0, 0.0]]
    theirs = _wsimod_nutrient(storage)
    _set_their_pools(theirs, pools)
    with pytest.raises(ZeroDivisionError) as raised:
        theirs.pull_storage({"volume": 0.25})
    frame = traceback.extract_tb(raised.value.__traceback__)[-1]
    assert frame.filename.endswith("land.py") and frame.lineno == 1460
    land = _nutrient_land(initial_storage=storage)
    ours = land.surfaces[0]
    ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
    ctx = _nut_ctx(land)
    reply = ours.pull_storage(NUT.to_vqip({"volume": 0.25}), active=TRUE)
    proportion = 0.25 / 1.0
    for name in ("nitrate", "ammonia", "nitrite"):
        assert float(reply[NUT.column(name)]) == 0.0
    assert float(reply[NUT.column("phosphate")]) == 0.2 * proportion
    assert float(ours.nutrient_pool.storage[DI, 0]) == 0.4 - 0.4 * proportion
    assert bool(ctx.events["pull_storage_no_din"]["land"])
    (decision,) = [d for d in ctx.decisions if d.site.endswith("/pull_storage/no_din")]
    assert decision.site == "node:land/surfaces.g/pull_storage/no_din" and bool(decision.outcome)
    # with inorganic N in the tank the guard is not taken
    ours.storage = NUT.to_vqip({**storage, "nitrate": 0.01})
    ctx = _nut_ctx(land)
    ours.pull_storage(NUT.to_vqip({"volume": 0.25}), active=TRUE)
    assert not bool(ctx.events["pull_storage_no_din"]["land"])


def _month(k: int) -> datetime:
    return datetime(2003 + (9 + k) // 12, (9 + k) % 12 + 1, 1)


def _land_data(*, months=15, negative=None, zero=()) -> dict:
    """``{(variable, first of month): value}`` of the 12 surface variables from 2003-10,
    distinct per variable and month; the month ``negative`` holds negative values (Review
    Focus 3) and the variables in ``zero`` are 0."""
    data = {}
    for i, variable in enumerate((*SURFACE_DEPOSITION, *SURFACE_INPUTS)):
        for k in range(months):
            value = (i + 1) * 1e-6 + k * 1.7e-7
            if k == negative:
                value = -value
            data[(variable, _month(k))] = 0.0 if variable in zero else value
    return data


def _bind_month(land, data, date) -> None:
    month = month_start(date)
    land.bind(
        {
            "precipitation": torch.tensor(0.0, dtype=F64),
            "et0": torch.tensor(0.0, dtype=F64),
            "temperature": torch.tensor(10.0, dtype=F64),
            **{
                f"surfaces.g.{v}": torch.tensor(x, dtype=F64)
                for (v, m), x in data.items()
                if m == month
            },
        }
    )


_INPUTS = ("atmospheric_deposition", "precipitation_deposition", "fertiliser", "manure")
_START = {"volume": 3.0, "ammonia": 0.01, "nitrate": 0.02, "phosphate": 0.1, "temperature": 11.0}


def test_fertiliser_manure_and_deposition_equal_wsimods():
    """``land.py:1574-1710``, ``:2267-2311`` and ``np.py:226-328`` against live WSIMOD over 400
    days from 2003-10-01, with 15 months of the 12 variables, one month negative (Review
    Focus 3): the storage, the pools and each ``(in_, out_)`` are bitwise WSIMOD's and finite.
    The manure goes ``fmdi`` to the dissolved inorganic pool and ``1 - fmdi`` to the fast one;
    the dry phosphate goes to the adsorbed pool and the tank gets ``x * (0 / P) = 0``."""
    pytest.importorskip("wsimod")
    from wsimod.orchestration.model import to_datetime

    data = _land_data(negative=5)
    theirs = _wsimod_nutrient(_START, data=data, area=3.0)
    land = _nutrient_land(data=data, area=3.0, initial_storage=_START)
    ours = land.surfaces[0]
    pool = ours.nutrient_pool
    assert [f.__name__ for f in theirs.inflows] == [f.__name__ for f in ours.inflows]
    assert torch.equal(pool.storage, _their_pools(theirs))
    negative = 0
    for k in range(400):
        date = datetime(2003, 10, 1) + timedelta(days=k)
        _nut_ctx(land)
        _bind_month(land, data, date)
        theirs.parent.monthyear = to_datetime(f"{date.year}-{date.month:02d}")
        for name in _INPUTS:
            before = pool.storage.clone()
            their_in, their_out = getattr(theirs, name)()
            in_, out_ = getattr(ours, name)(active=TRUE)
            assert torch.equal(in_, _vector(their_in)), (k, name)
            assert torch.equal(out_, _vector(their_out)), (k, name)
            assert torch.equal(ours.storage, _vector(theirs.storage)), (k, name)
            assert torch.equal(pool.storage, _their_pools(theirs)), (k, name)
            assert not (torch.isnan(in_).any() or torch.isnan(pool.storage).any())
            negative += bool((in_ < 0).any())
            if name == "manure":
                nhx, noy, srp = (ours.forcing[f"{v}-manure"] * 3.0 for v in ("nhx", "noy", "srp"))
                dep = torch.stack([noy + nhx, srp])
                fmdi = pool.fraction_manure_to_dissolved_inorganic
                assert torch.equal(pool.storage[DI], before[DI] + dep * fmdi)
                assert torch.equal(pool.storage[FAST], before[FAST] + dep * (1 - fmdi))
            if name == "atmospheric_deposition" and float(ours.forcing["srp-dry"]) > 0:
                srp_dry = ours.forcing["srp-dry"] * 3.0
                assert float(in_[NUT.column("phosphate")]) == 0.0
                assert torch.equal(pool.storage[ADS, 1], before[ADS, 1] + srp_dry)
    assert negative  # the negative month enters, unscaled


def test_the_liquid_ratio_is_evaluated():
    """``land.py:1595-1602``: with fertiliser, the tank's nitrate is ``noy * area * (in_.N /
    dep.N)``, the ratio exactly 1.0; with ``dep.N = 0`` the line is skipped. The decisions
    ``adjust_vqip_to_liquid/N|P`` carry ``call`` 0 to 3 in the order dry, wet, fertiliser,
    manure."""
    date = datetime(2003, 10, 1)
    for zero in ((), ("nhx-fertiliser", "noy-fertiliser")):
        data = _land_data(months=1, zero=zero)
        land = _nutrient_land(data=data, area=3.0, initial_storage=_START)
        ours = land.surfaces[0]
        ctx = _nut_ctx(land)
        _bind_month(land, data, date)
        replies = {name: getattr(ours, name)(active=TRUE)[0] for name in _INPUTS}
        noy = ours.forcing["noy-fertiliser"] * 3.0
        dep_n = noy + ours.forcing["nhx-fertiliser"] * 3.0
        for nutrient in ("N", "P"):
            site = f"node:land/surfaces.g/adjust_vqip_to_liquid/{nutrient}"
            decisions = [d for d in ctx.decisions if d.site == site]
            assert [d.call for d in decisions] == [0, 1, 2, 3]
        fertiliser_n = [d for d in ctx.decisions if d.site.endswith("liquid/N") and d.call == 2][0]
        assert bool(fertiliser_n.outcome) == (not zero)
        nitrate = replies["fertiliser"][NUT.column("nitrate")]
        if not zero:
            assert float(dep_n / dep_n) == 1.0
            assert torch.equal(nitrate, noy * (dep_n / dep_n))
        else:
            assert float(nitrate) == 0.0


def test_flushing_extracts_the_outflows_again():
    """L8 (``land.py:1605-1647``, reproduced: Y16, Y45), on 50 random tanks and requests: after
    two ``pull_storage`` calls (the subsurface flow and the percolation) and
    ``effective_precipitation_flushing``, the pools and the tank are WSIMOD's, bitwise. Every
    other case has nutrients in the infiltration excess (a unit test may: in a day erosion adds
    them only later), which pins its terms and their order. Without them, the dissolved
    inorganic N is smaller than ``DI0 - (ss + perc)`` by the second extraction."""
    rng = random.Random(23)
    theirs = _wsimod_nutrient({"volume": 1.0})
    land = _nutrient_land()
    ours = land.surfaces[0]
    for k in range(50):
        storage = _random_tank(rng)
        pools = [[rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)] for _ in POOLS]
        requests = [storage["volume"] * rng.uniform(0.05, 0.4) for _ in range(2)]
        excess = {"volume": 0.0, **dict.fromkeys(NUT.names, 0.0)}
        if k % 2:
            excess.update({n: rng.uniform(0.0, 0.05) for n in NUT.additive}, volume=0.1)
        theirs.storage = dict(storage)
        _set_their_pools(theirs, pools)
        ours.storage = NUT.to_vqip(storage)
        ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
        _nut_ctx(land)
        start = ours.nutrient_pool.storage.clone()
        their_ss, their_perc = (theirs.pull_storage({"volume": v}) for v in requests)
        theirs.subsurface_flow, theirs.percolation = their_ss, their_perc
        theirs.infiltration_excess = dict(excess)
        ss, perc = (ours.pull_storage(NUT.to_vqip({"volume": v}), active=TRUE) for v in requests)
        ours.subsurface_flow, ours.percolation = ss, perc
        ours.infiltration_excess = NUT.to_vqip(excess)
        once = ours.nutrient_pool.storage.clone()
        assert theirs.effective_precipitation_flushing() == (theirs.empty_vqip(),) * 2
        in_, out_ = ours.effective_precipitation_flushing(active=TRUE)
        assert not in_.any() and not out_.any()
        assert torch.equal(ours.nutrient_pool.storage, _their_pools(theirs)), k
        assert torch.equal(ours.storage, _vector(theirs.storage)), k
        if k % 2 == 0 and float(once[DI, 0]) > _n(ss) + _n(perc):  # nothing clipped
            flowed = _n(ss) + _n(perc)
            di_n = float(ours.nutrient_pool.storage[DI, 0])
            assert di_n < float(start[DI, 0]) - flowed
            assert di_n == pytest.approx(float(once[DI, 0]) - flowed, rel=1e-9, abs=1e-15)


def test_the_initial_pool_comes_from_the_storage_and_initial_soil_storage():
    """``land.py:1293-1324``: ``DI = (nitrate + ammonia + nitrite, phosphate)`` and ``DO =
    (org-nitrogen, org-phosphorus)`` of the tank; ``ads = (ammonia + nitrate + nitrite,
    phosphate)`` and ``fast = (org-nitrogen, org-phosphorus)`` of ``initial_soil_storage``,
    in WSIMOD's order of terms (random values, so another association is another float); the
    humus pool stays 0."""
    rng = random.Random(29)
    for k in range(40):
        storage = _random_tank(rng)
        soil = (_random_soil(rng), None, 0, {})[k % 4]
        theirs = _wsimod_nutrient(storage, initial_soil_storage=soil)
        pool = _nutrient_land(initial_storage=storage, initial_soil_storage=soil).surfaces[0]
        pool = pool.nutrient_pool.storage
        assert torch.equal(pool, _their_pools(theirs)), k
        di_n = (storage["nitrate"] + storage["ammonia"]) + storage["nitrite"]
        assert pool[DI].tolist() == [di_n, storage["phosphate"]]
        assert pool[DO].tolist() == [storage["org-nitrogen"], storage["org-phosphorus"]]
        if soil:
            ads_n = (soil["ammonia"] + soil["nitrate"]) + soil["nitrite"]
            assert pool[ADS].tolist() == [ads_n, soil["phosphate"]]
            assert pool[FAST].tolist() == [soil["org-nitrogen"], soil["org-phosphorus"]]
        else:
            assert not pool[ADS].any() and not pool[FAST].any()
        assert not pool[POOLS.index("humus_pool")].any()


def test_adjust_vqip_to_liquid_equals_wsimods():
    """``land.py:1574-1603`` on 200 random VQIPs with every scaled column non-zero
    (``org-nitrogen`` and ``org-phosphorus`` included), ``dep`` positive, zero or negative and
    ``in_`` any fraction of it: bitwise WSIMOD's."""
    rng = random.Random(31)
    theirs = _wsimod_nutrient({"volume": 1.0})
    land = _nutrient_land()
    ours = land.surfaces[0]
    for k in range(200):
        vqip = {"volume": 0.0, **{n: rng.uniform(-0.1, 1.0) for n in NUT.additive}}
        vqip["temperature"] = 0.0
        dep = {n: rng.choice((rng.uniform(0.01, 2.0), 0.0, -rng.uniform(0.01, 2.0))) for n in "NP"}
        in_ = {n: dep[n] * rng.uniform(0.0, 1.0) for n in "NP"}
        _nut_ctx(land)
        got = ours.adjust_vqip_to_liquid(
            NUT.to_vqip(vqip),
            torch.tensor([dep["N"], dep["P"]], dtype=F64),
            torch.tensor([in_["N"], in_["P"]], dtype=F64),
            active=TRUE,
        )
        want = theirs.adjust_vqip_to_liquid(dict(vqip), dict(dep), dict(in_))
        assert torch.equal(got, _vector(want)), k


def _complete(**values):
    return _growing(data_input_dict=_data(SURFACE_INPUTS), **values)


def test_initial_soil_storage_refusals():
    """L16, Y37: a float, which WSIMOD's docstring allows, raises ``TypeError`` at
    ``land.py:1312`` and is refused; so is a mapping lacking a key. ``0`` and ``None`` are
    falsy and accepted. Without ``nitrate`` it is never read."""
    with pytest.raises(TypeError):
        _wsimod_nutrient({"volume": 1.0}, initial_soil_storage=0.5)
    with pytest.raises(ValueError, match="initial_soil_storage"):
        _validate(_mapping([_complete(initial_soil_storage=0.5)], lists=_NITRATE))
    lacking = {k: v for k, v in _SOIL.items() if k != "org-phosphorus"}
    with pytest.raises(ValueError, match="org-phosphorus"):
        _validate(_mapping([_complete(initial_soil_storage=lacking)], lists=_NITRATE))
    with pytest.raises(ValueError, match="org-phosphorus"):
        _nutrient_land(initial_soil_storage=lacking)
    for falsy in (0, None):
        mapping = _mapping([_complete(initial_soil_storage=falsy)], lists=_NITRATE)
        assert _validate(mapping, pollutants=True) == []
    quiet = _mapping([_growing(initial_soil_storage=0.5)], lists=_LISTS)
    assert _validate(quiet, pollutants=True) == []


@pytest.mark.parametrize("name", [n for n in NUTRIENT_POLLUTANTS if n != "nitrate"])
def test_nitrate_needs_the_eight_pollutants(name):
    """With ``nitrate``, the seven other names of :data:`NUTRIENT_POLLUTANTS` are needed in either
    mode (``land.py:1294-1307``, ``:1465``, ``:2088``, ``:1822``; ``solids`` is L24)."""
    lists = {key: [n for n in names if n != name] for key, names in _NITRATE.items()}
    mapping = _mapping([_complete()], lists=lists)
    for pollutants in (True, False):
        with pytest.raises(ValueError, match=name):
            _validate(mapping, pollutants=pollutants)


_DOMAINS = [
    ("rooting_depth", 0.0),
    ("bulk_density", 0.0),
    ("kfr", -1.0),
    ("nfr", 0.0),
    ("cohesion", 0.0),
    ("thetaupp", 0.0),
    ("thetalow", 0.0),
    ("hsatINs", 0.0),
    ("limpar", -0.1),
    ("field_capacity", 0.0),
]


@pytest.mark.parametrize(("name", "value"), _DOMAINS)
def test_the_domain_refusals(name, value):
    """Spec, Validation: with ``nitrate`` each of these is a WSIMOD crash, refused at build (the
    merged parameters, in either mode) and at bind (any batched instance, with pollutants on,
    where the pool exists). Without ``nitrate`` the build
    accepts what it reads nothing from."""
    mapping = _mapping([_complete()], overrides={"g": {name: value}}, lists=_NITRATE)
    for pollutants in (True, False):  # at build in either mode (D18); at bind with the pool
        with pytest.raises(ValueError, match=name):
            _validate(mapping, pollutants=pollutants)
    surface = _nutrient_land(initial_storage=_START).surfaces[0]
    good = float(surface.defaults[name])
    with pytest.raises(ValueError, match=name):
        surface.bind({name: torch.tensor([good, value], dtype=F64)})
    surface.bind({name: torch.tensor([good, good], dtype=F64)})
    if name not in ("rooting_depth", "field_capacity"):
        quiet = _mapping([_growing()], overrides={"g": {name: value}}, lists=_LISTS)
        assert _validate(quiet, pollutants=True) == []


def _nitrate_mapping(**surface) -> dict:
    data = {(v, d): 0.001 for v in ("precipitation", "et0", "temperature") for d in _DATES}
    start = {**dict.fromkeys(_KEYS, 0.0), "volume": 0.2, "nitrate": 0.01, "ammonia": 0.002}
    start.update(phosphate=0.001, temperature=10.0)  # every key: WSIMOD's tank reads them all
    growing = _complete(initial_storage=start, initial_soil_storage=_SOIL, **surface)
    return _mapping([growing], data=data, lists=_NITRATE)


def test_nutrient_pool_parameters_are_drivers_with_wsimods_defaults():
    """L20, Y24: the pool's parameters are drivers ``"wsimod.<Class>.nutrient_pool.<p>"`` with
    WSIMOD's defaults (``np.py:16-23``), pairs ``(n, 2)`` with the columns ``("N", "P")``; a
    driver reaches the pool. A surface key or an override naming one is refused, naming
    ``NutrientPool``."""
    from noodl.apps.wsimod import build_model, initial_drivers
    from noodl.apps.wsimod.nutrient_pool import NUTRIENT_POOL_PAIRS, NUTRIENT_POOL_SCALARS

    model, state, _ = build_model(read_wsimod_config(_nitrate_mapping()))
    layout = model.wsimod_day.drivers
    key = "wsimod.GrowingSurface.nutrient_pool.degrhpar"
    assert layout[key].template.tolist() == [[7 * 1e-5, 7 * 1e-6]]
    assert layout[key].columns == ("N", "P")
    assert layout[key].labels == [("land", "g")]
    for name, pair in NUTRIENT_POOL_PAIRS.items():
        d = layout[f"wsimod.GrowingSurface.nutrient_pool.{name}"]
        assert d.template.tolist() == [list(pair)] and d.unit == "-"
    for name, value in NUTRIENT_POOL_SCALARS.items():
        d = layout[f"wsimod.GrowingSurface.nutrient_pool.{name}"]
        assert d.template.tolist() == [value] and d.columns is None
    fmdi = "wsimod.GrowingSurface.nutrient_pool.fraction_manure_to_dissolved_inorganic"
    drivers = initial_drivers(model, values={fmdi: torch.tensor([[0.25, 0.75]], dtype=F64)})
    _, _, engine = next(model.wsimod_day.run_days(state, drivers, days=1))
    pool = engine.nodes["land"].surfaces[0].nutrient_pool
    assert pool.fraction_manure_to_dissolved_inorganic.tolist() == [0.25, 0.75]
    assert pool.fraction_manure_to_fast.tolist() == [0.75, 0.25]
    with pytest.raises(ValueError, match="degrhpar.*NutrientPool"):
        read_wsimod_config(_nitrate_mapping(degrhpar={"N": 1e-4, "P": 1e-5}))
    mapping = _nitrate_mapping()
    mapping["overrides"] = {
        "nodes": {"land": {"type_": "Land", "name": "land", "surfaces": {"g": {"degrhpar": {}}}}}
    }
    with pytest.raises(ValueError, match="degrhpar.*NutrientPool"):
        read_wsimod_config(mapping)


def test_without_nitrate_there_is_no_pool():
    """D14, D16: without ``nitrate`` (or in a quantity-only run) there are no
    ``"wsimod.nutrient_pool"`` rows and no pool drivers, and the dry deposition enters the tank
    whole (``Surface``'s ``dry_deposition_to_tank``)."""
    from noodl.apps.wsimod import build_model

    for mapping, pollutants in (
        (_mapping([_growing()], lists=_LISTS), True),
        (_nitrate_mapping(), False),
    ):
        model, state, _ = build_model(read_wsimod_config(mapping), pollutants=pollutants)
        day = model.wsimod_day
        assert day.state_labels("wsimod.nutrient_pool") == []
        assert state["wsimod.nutrient_pool"].shape == (0,)
        assert not [k for k in day.drivers if "nutrient_pool" in k]
        assert day.engine.nodes["land"].surfaces[0].nutrient_pool is None
    dep = Pollutants(
        ("ammonia", "phosphate", "temperature"), ("ammonia", "phosphate"), ("temperature",)
    )
    data = _land_data(months=1)
    spec = SurfaceSpec(
        "g",
        "GrowingSurface",
        {"area": 2.0},
        data,
        constructed_variables=frozenset(v for v, _ in data),
    )
    land = make_node(NodeSpec("land", "Land", "Land", {}, {}, [spec]), None, dep)
    ours = land.surfaces[0]
    assert ours.nutrient_pool is None
    land.ctx = Ctx(dep, FA, torch.Size(()), day=0, node_names=["land"])
    month = _month(0)
    land.bind(
        {
            "precipitation": torch.tensor(0.0, dtype=F64),
            "et0": torch.tensor(0.0, dtype=F64),
            "temperature": torch.tensor(10.0, dtype=F64),
            **{
                f"surfaces.g.{v}": torch.tensor(x, dtype=F64)
                for (v, m), x in data.items()
                if m == month
            },
        }
    )
    before = ours.storage.clone()
    in_, _ = ours.atmospheric_deposition(active=TRUE)
    assert float(in_[dep.column("ammonia")]) == data[("nhx-dry", month)] * 2.0
    assert float(in_[dep.column("phosphate")]) == data[("srp-dry", month)] * 2.0
    assert torch.equal(ours.storage[1:3], before[1:3] + in_[1:3])


def test_the_pool_state_has_ten_labelled_rows():
    """D4: ``"wsimod.nutrient_pool"`` is flat, ten rows ``(land, "surfaces.<s>.nutrient_pool",
    <pool>, <N|P>)`` per growing surface, kg, not water held; it starts at the constructor's
    pools, and a load then dump reproduces the engine's ``(5, 2)`` tensor exactly. One day runs
    and the balance closes."""
    from noodl.apps.wsimod import build_model, initial_state, wsimod_run

    model, state, drivers = build_model(read_wsimod_config(_nitrate_mapping()))
    day = model.wsimod_day
    labels = day.state_labels("wsimod.nutrient_pool")
    assert len(labels) == 10
    assert labels[0] == ("land", "surfaces.g.nutrient_pool", "fast_pool", "N")
    assert labels[9] == ("land", "surfaces.g.nutrient_pool", "adsorbed_inorganic_pool", "P")
    assert day.key_labels["wsimod.nutrient_pool"].unit == "kg"
    land = day.engine.nodes["land"]
    assert "surfaces.g.nutrient_pool" in land.carried_paths
    assert land.store_paths == Land.store_paths_of([("GrowingSurface", "g")], nutrients=True)
    assert "surfaces.g.nutrient_pool" not in Land.store_paths_of([("GrowingSurface", "g")])
    expected = torch.tensor(
        [
            [_SOIL["org-nitrogen"], _SOIL["org-phosphorus"]],
            [0.0, 0.0],
            [0.01 + 0.002 + 0.0, 0.001],
            [0.0, 0.0],
            [(_SOIL["ammonia"] + _SOIL["nitrate"]) + _SOIL["nitrite"], _SOIL["phosphate"]],
        ],
        dtype=F64,
    )
    assert torch.equal(state["wsimod.nutrient_pool"].reshape(5, 2), expected)
    values = {label: 0.1 * (i + 1) for i, label in enumerate(labels)}
    given = initial_state(model, values={"wsimod.nutrient_pool": values})
    day._load(given, torch.Size(()))
    pool = land.surfaces[0].nutrient_pool.storage
    assert pool.shape == (5, 2)
    assert torch.equal(pool.reshape(10), given["wsimod.nutrient_pool"])
    assert torch.equal(
        day._dump(torch.Size(()))["wsimod.nutrient_pool"], given["wsimod.nutrient_pool"]
    )
    results = wsimod_run(model, given, drivers, days=2)
    assert float(results.mass_balance.residual.abs().max()) <= 1e-12


def test_the_harness_captures_and_replays_the_pool():
    """The replay harness's pool support, before any whole-model nitrate case (D20):
    ``_wsimod_store`` of a live WSIMOD Land's ``surfaces.g.nutrient_pool`` is its five stores'
    ``(N, P)``; ``load_value`` makes the ``(5, 2)`` tensor that ``load_stores`` takes;
    ``compare_store`` passes on it and raises ``Mismatch`` on one perturbed entry; and
    ``call_forcing`` captures the six fertiliser and manure values with nitrate, none
    without."""
    pytest.importorskip("wsimod")
    from wsimod.orchestration.model import to_datetime

    from tests.apps.wsimod._replay import Mismatch, _Comparer, compare_store, load_value
    from tests.verification._wsimod_reference import _wsimod_store, call_forcing, wsimod_model

    path = "surfaces.g.nutrient_pool"
    mapping = _nitrate_mapping()
    theirs = wsimod_model(mapping).nodes["land"]
    pools = theirs.get_surface("g").nutrient_pool.pools
    rng = random.Random(37)
    for pool in pools:
        pool.storage = {"N": rng.uniform(0.0, 2.0), "P": rng.uniform(0.0, 2.0)}
    value = _wsimod_store(theirs, path)
    assert value == [[pool.storage[n] for n in "NP"] for pool in pools]
    network = read_wsimod_config(mapping)
    ours = make_node(network.nodes[0], network, network.pollutants)
    assert path in ours.store_paths
    tensor = load_value(network.pollutants, ours, path, value)
    assert tensor.shape == (5, 2) and tensor.tolist() == value
    ours.load_stores({path: tensor})
    assert torch.equal(ours.surfaces[0].nutrient_pool.storage, tensor)
    compare = _Comparer(network.pollutants, 1.0, 0.0)
    compare_store(compare, "pool", path, ours.stores()[path], value)
    perturbed = [list(row) for row in value]
    perturbed[3][1] = math.nextafter(perturbed[3][1], math.inf)
    with pytest.raises(Mismatch, match=r"pool\[3, 1\]"):
        compare_store(compare, "pool", path, ours.stores()[path], perturbed)
    theirs.monthyear = to_datetime("2000-01")
    captured = {k: v for k, v in call_forcing(theirs).items() if k.startswith("forcing.surfaces.")}
    assert captured == {f"forcing.surfaces.g.{v}": 0.001 for v in SURFACE_INPUTS}
    plain = wsimod_model(_mapping([_complete()], lists=_LISTS)).nodes["land"]
    plain.monthyear = to_datetime("2000-01")
    assert not [k for k in call_forcing(plain) if k.startswith("forcing.surfaces.")]


# --- the soil factors, the transformation, crop uptake and denitrification (Task 7) -------------
# land.py:1744-1926, :2096-2152. Every fixture is random, so that an operation taken in another
# order (an association, a factor moved) gives another float, and is compared bitwise against a
# live WSIMOD GrowingSurface.
_SOIL_SCALARS = (
    "satact",
    "thetaupp",
    "thetalow",
    "thetapow",
    "uptake1",
    "uptake2",
    "uptake3",
    "uptake_PNratio",
    "limpar",
    "exppar",
    "hsatINs",
    "denpar",
)
_SOIL_BUILT = ("area", "rooting_depth", "field_capacity", "wilting_point")
_SOIL_PROCESSES = (
    "calc_temperature_dependence_factor",
    "calc_soil_moisture_dependence_factor",
    "soil_pool_transformation",
    "calc_crop_uptake",
    "denitrification",
)
_RATES = ("degrhpar", "dishpar", "minfpar", "disfpar", "immobdpar")
S7 = "node:land/surfaces.g"


def _soil_parameters(rng) -> dict:
    return {
        "area": rng.uniform(5.0, 50.0),
        "rooting_depth": rng.uniform(0.4, 1.6),
        "field_capacity": rng.uniform(0.25, 0.39),
        "wilting_point": rng.uniform(0.05, 0.2),
        "satact": rng.uniform(0.3, 0.9),
        "thetaupp": rng.uniform(0.05, 0.2),
        "thetalow": rng.uniform(0.04, 0.15),
        "thetapow": rng.uniform(0.6, 2.4),
        "uptake1": rng.uniform(5.0, 25.0),
        "uptake2": rng.uniform(0.5, 3.0),
        "uptake3": rng.uniform(0.005, 0.05),
        "uptake_PNratio": rng.uniform(0.05, 0.3),
        "limpar": rng.uniform(0.4, 0.8),
        "exppar": rng.uniform(1.2, 3.1),
        "hsatINs": rng.uniform(0.3, 3.0),
        "denpar": rng.uniform(0.005, 0.05),
    }


def _soil_pair(parameters: dict, **structure):
    """WSIMOD's surface (its built parameters through the constructor, the rest set as
    attributes, as ``apply_overrides`` does) and noodl's, from the same values."""
    built = {k: parameters[k] for k in _SOIL_BUILT if k != "area" and k in parameters}
    theirs = _wsimod_nutrient(
        {"volume": 1.0}, area=parameters.get("area", 1.0), **built, **structure
    )
    for name in _SOIL_SCALARS:
        if name in parameters:
            setattr(theirs, name, parameters[name])
    land = _nutrient_land(**parameters, **structure)
    return theirs, land, land.surfaces[0]


def _set_state(theirs, ours, tank, pools, *, tdf=None, smdf=None, days=None) -> None:
    theirs.storage = dict(tank)
    ours.storage = NUT.to_vqip(tank)
    _set_their_pools(theirs, pools)
    ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
    if tdf is not None:
        theirs.nutrient_pool.temperature_dependence_factor = tdf
        ours.nutrient_pool.temperature_dependence_factor = torch.tensor(tdf, dtype=F64)
    if smdf is not None:
        theirs.nutrient_pool.soil_moisture_dependence_factor = smdf
        ours.nutrient_pool.soil_moisture_dependence_factor = torch.tensor(smdf, dtype=F64)
    if days is not None:
        theirs.days_after_sow = None if days < 0 else days
        ours.days_after_sow = torch.tensor(float(days), dtype=F64)


def _assert_same(theirs, ours, their_pair, our_pair, label) -> None:
    for their, our in zip(their_pair, our_pair, strict=True):
        assert torch.equal(our, _vector(their)), label
    assert torch.equal(ours.storage, _vector(theirs.storage)), label
    assert torch.equal(ours.nutrient_pool.storage, _their_pools(theirs)), label


def _seen(ctx, seen: dict) -> None:
    for d in ctx.decisions:
        if bool(d.active):
            seen.setdefault(d.site, set()).add(bool(d.outcome))


def test_the_soil_processes_follow_wsimods_list():
    """D20: ``processes`` is WSIMOD's (``land.py:1283-1291``) without the later tasks' erosion
    and adsorption, so each process is at its place; the factors have no part, the other three
    have theirs (Y57)."""
    from noodl.apps.wsimod.land import NO_PART, PART_NAMES

    theirs, _, ours = _soil_pair({})
    names = [f.__name__ for f in theirs.processes if f.__name__ not in ("erosion", "adsorption")]
    assert [f.__name__ for f in ours.processes] == names
    assert names[1:] == list(_SOIL_PROCESSES)
    assert {"calc_temperature_dependence_factor", "calc_soil_moisture_dependence_factor"} <= NO_PART
    assert PART_NAMES["soil_pool_transformation"] == "GrowingSurface.soil_pool_transformation"
    assert PART_NAMES["calc_crop_uptake"] == "GrowingSurface.calc_crop_uptake"
    assert PART_NAMES["denitrification"] == "GrowingSurface.denitrification"


def test_the_soil_factors_equal_wsimods():
    """``land.py:1813-1856`` (L12, Y19), bitwise, on random parameters, every temperature of
    ``{-1, 0, 2.5, 4.9, 5, 5.000001, 15, 25}`` and soil moisture at, between and beyond the
    wilting point and field capacity. The temperature factor jumps at 5 °C and is larger at
    4.9 °C than at 15 °C; ``min(1, wp_comp, fc_comp)`` is Python's, its two decisions in
    Python's order, each ``b < a``."""
    rng = random.Random(31)
    temperatures = (-1.0, 0.0, 2.5, 4.9, 5.0, 5.000001, 15.0, 25.0)
    seen: dict = {}
    for _ in range(4):
        parameters = _soil_parameters(rng)
        theirs, land, ours = _soil_pair(parameters)
        fc_m, wp_m, area = theirs.field_capacity_m, theirs.wilting_point_m, parameters["area"]
        assert float(ours.field_capacity_m) == fc_m and float(ours.wilting_point_m) == wp_m
        moistures = [wp_m * 0.5, wp_m, fc_m, fc_m * 1.2]
        moistures += [rng.uniform(wp_m, fc_m) for _ in range(25)]
        for temperature in (*temperatures, rng.uniform(-3.0, 30.0)):
            for smc in moistures:
                tank = {**_random_tank(rng, volume=smc * area), "temperature": temperature}
                _set_state(theirs, ours, tank, [[0.0, 0.0]] * 5)
                ctx = _nut_ctx(land)
                empty = theirs.empty_vqip()
                assert theirs.calc_temperature_dependence_factor() == (empty, empty)
                assert theirs.calc_soil_moisture_dependence_factor() == (empty, empty)
                for name in _SOIL_PROCESSES[:2]:
                    in_, out_ = getattr(ours, name)(active=TRUE)
                    assert not in_.any() and not out_.any()
                pool, their_pool = ours.nutrient_pool, theirs.nutrient_pool
                label = (temperature, smc)
                tdf = float(pool.temperature_dependence_factor)
                assert tdf == their_pool.temperature_dependence_factor, label
                smdf = float(pool.soil_moisture_dependence_factor)
                assert smdf == their_pool.soil_moisture_dependence_factor, label
                for d in ctx.decisions:  # the three-way min's decisions are each b < a
                    if d.site.endswith("/min:wp") and bool(d.active):
                        wp = theirs.storage["volume"] / area - wp_m
                        wp_comp = wp / (parameters["thetalow"] * parameters["rooting_depth"])
                        wp_comp = wp_comp ** parameters["thetapow"]
                        assert bool(d.outcome) == (wp_comp < 1)
                _seen(ctx, seen)
        for _ in range(100):  # random temperatures: (T - 20) / 10 and Python's 2 ** x
            temperature = rng.uniform(-3.0, 40.0)
            _set_state(
                theirs, ours, {**_random_tank(rng), "temperature": temperature}, [[0, 0]] * 5
            )
            ctx = _nut_ctx(land)
            theirs.calc_temperature_dependence_factor()
            ours.calc_temperature_dependence_factor(active=TRUE)
            their_factor = theirs.nutrient_pool.temperature_dependence_factor
            assert float(ours.nutrient_pool.temperature_dependence_factor) == their_factor
    for branch in ("saturated", "wilted", "min:wp", "min:fc"):
        assert seen[f"{S7}/calc_soil_moisture_dependence_factor/{branch}"] == {True, False}
    for branch in ("warm", "thawed"):
        assert seen[f"{S7}/calc_temperature_dependence_factor/{branch}"] == {True, False}
    # L12: the values either side of 5 °C, and the ties of the moisture factor (area 2: exact)
    theirs, land, ours = _soil_pair({"area": 2.0, "rooting_depth": 0.9})
    factors = {}
    for temperature in temperatures:
        _set_state(theirs, ours, {**_random_tank(rng), "temperature": temperature}, [[0, 0]] * 5)
        _nut_ctx(land)
        ours.calc_temperature_dependence_factor(active=TRUE)
        factors[temperature] = float(ours.nutrient_pool.temperature_dependence_factor)
    assert factors[-1.0] == factors[0.0] == 0.0 and factors[2.5] == 0.5
    assert factors[4.9] == 4.9 / 5 and factors[5.0] == 1.0
    assert factors[5.000001] == 2 ** ((5.000001 - 20) / 10)
    assert factors[15.0] == 2 ** (-0.5) and factors[15.0] < factors[4.9]
    assert factors[25.0] == 2**0.5
    for smc, expected in ((ours.field_capacity_m, 0.6), (ours.wilting_point_m, 0.0)):
        volume = float(smc) * 2.0
        assert volume / 2.0 == float(smc)
        _set_state(theirs, ours, _random_tank(rng, volume=volume), [[0, 0]] * 5)
        ctx = _nut_ctx(land)
        ours.calc_soil_moisture_dependence_factor(active=TRUE)
        theirs.calc_soil_moisture_dependence_factor()
        assert float(ours.nutrient_pool.soil_moisture_dependence_factor) == expected
        assert theirs.nutrient_pool.soil_moisture_dependence_factor == expected


def _random_pools(rng, *, scale=1.0) -> list[list[float]]:
    return [[rng.uniform(0.0, scale), rng.uniform(0.0, scale)] for _ in POOLS]


def _random_rates(rng, theirs, ours) -> None:
    rates = {name: (rng.uniform(0.0, 0.6), rng.uniform(0.0, 0.6)) for name in _RATES}
    for name, (n, p) in rates.items():
        setattr(theirs.nutrient_pool, name, {"N": n, "P": p})
    pool = ours.nutrient_pool
    pool.bind(
        {
            "fraction_manure_to_dissolved_inorganic": pool.fraction_manure_to_dissolved_inorganic,
            "fraction_dry_n_to_dissolved_inorganic": pool.fraction_dry_n_to_dissolved_inorganic,
            **{name: torch.tensor(pair, dtype=F64) for name, pair in rates.items()},
        }
    )


def test_soil_pool_transformation_equals_wsimods():
    """``land.py:1744-1811`` with ``np.py:396-481``, bitwise, on 300 random tanks, pools, rates
    and factors: each of the four signs both ways, an increase of exactly 0 (a zero factor)
    taking the ``else`` branches (rule 6), and tanks smaller than the ``out_`` (clamped)."""
    rng = random.Random(37)
    theirs, land, ours = _soil_pair(_soil_parameters(rng))
    seen: dict = {}
    zero = clamped = 0
    for k in range(300):
        tank = _random_tank(rng)
        if k % 5 == 0:
            tank.update(ammonia=rng.uniform(0.0, 1e-3), phosphate=rng.uniform(0.0, 1e-3))
        tdf = 0.0 if k % 17 == 0 else rng.uniform(0.0, 1.0)
        _set_state(theirs, ours, tank, _random_pools(rng), tdf=tdf, smdf=rng.uniform(0.0, 1.0))
        _random_rates(rng, theirs, ours)
        ctx = _nut_ctx(land)
        their = theirs.soil_pool_transformation()
        ours_ = ours.soil_pool_transformation(active=TRUE)
        _assert_same(theirs, ours, their, ours_, k)
        zero += tdf == 0.0
        clamped += bool(ctx.events["nutrient_pool_not_tracking"]["land"])
        if tdf == 0.0:
            for branch in ("inorganic_N", "organic_N", "inorganic_P", "organic_P"):
                (d,) = [d for d in ctx.decisions if d.site.endswith(f"transformation/{branch}")]
                assert not bool(d.outcome) and float(d.margin) == 0.0
            assert not ours_[0].any()
        _seen(ctx, seen)
        assert not bool(ctx.events["transformation_no_din"]["land"])
    for branch in ("inorganic_N", "organic_N", "inorganic_P", "organic_P"):
        assert seen[f"{S7}/soil_pool_transformation/{branch}"] == {True, False}
    assert zero and clamped


def test_the_transformation_without_din_sends_the_change_to_ammonia():
    """L6, Y18: ``nitrate + ammonia = 0``. WSIMOD raises ``ZeroDivisionError`` at
    ``land.py:1758``; noodl uses the proportion 0, so the whole change in dissolved inorganic N
    is ammonia, and records ``"transformation_no_din"``."""
    import traceback

    rng = random.Random(43)
    theirs, land, ours = _soil_pair(_soil_parameters(rng))
    for sign, rates in ((1, (0.5, 0.0)), (-1, (0.0, 0.5))):  # mineralisation, immobilisation
        tank = {**_random_tank(rng), "nitrate": 0.0, "ammonia": 0.0}
        _set_state(theirs, ours, tank, _random_pools(rng), tdf=0.7, smdf=0.8)
        _random_rates(rng, theirs, ours)
        theirs.nutrient_pool.minfpar = {"N": rates[0], "P": 0.1}
        theirs.nutrient_pool.immobdpar = {"N": rates[1], "P": 0.1}
        values = {
            name: getattr(ours.nutrient_pool, name)
            for name in (*_RATES, "fraction_manure_to_dissolved_inorganic")
        }
        values.update(
            minfpar=torch.tensor([rates[0], 0.1], dtype=F64),
            immobdpar=torch.tensor([rates[1], 0.1], dtype=F64),
            fraction_dry_n_to_dissolved_inorganic=torch.tensor(0.9, dtype=F64),
        )
        ours.nutrient_pool.bind(values)
        with pytest.raises(ZeroDivisionError) as raised:
            theirs.soil_pool_transformation()
        frame = traceback.extract_tb(raised.value.__traceback__)[-1]
        assert frame.filename.endswith("land.py") and frame.lineno == 1758
        ctx = _nut_ctx(land)
        before = ours.nutrient_pool.storage.clone()
        in_, out_ = ours.soil_pool_transformation(active=TRUE)
        change = ours.nutrient_pool.storage[DI, 0] - before[DI, 0]
        assert bool(ctx.events["transformation_no_din"]["land"])
        tracking = bool(ctx.events["nutrient_pool_not_tracking"]["land"])
        if sign > 0:  # the increase enters, all of it as ammonia
            assert float(in_[NUT.column("nitrate")]) == 0.0
            assert float(in_[NUT.column("ammonia")]) == float(change) > 0.0
            assert not tracking
        else:  # the decrease is requested as ammonia, which the tank does not have: clamped
            assert float(change) < 0.0
            assert float(out_[NUT.column("nitrate")]) == float(out_[NUT.column("ammonia")]) == 0
            assert tracking
        assert torch.isfinite(in_).all() and torch.isfinite(out_).all()


def test_a_clamped_pull_records_the_event():
    """L10 (D19): an ``out_`` larger than the tank's column. ``pull_pollutants`` clamps, the
    returned ``out_`` is the clamped one (WSIMOD's too: it mutates its argument, so its
    ``compare_vqip`` is dead), and ``"nutrient_pool_not_tracking"`` is true. With the tank
    above the request it is false."""
    rng = random.Random(47)
    theirs, land, ours = _soil_pair(_soil_parameters(rng))
    for small in (True, False):
        tank = _random_tank(rng)
        tank.update(nitrate=1e-7, ammonia=1e-7) if small else tank.update(nitrate=5.0, ammonia=5.0)
        pools = _random_pools(rng)
        pools[DI] = [2.0, 0.01]
        _set_state(theirs, ours, tank, pools, tdf=0.9, smdf=0.9)
        _random_rates(rng, theirs, ours)
        theirs.nutrient_pool.immobdpar["N"] = 0.5
        ours.nutrient_pool.immobdpar = torch.tensor(
            [0.5, float(ours.nutrient_pool.immobdpar[1])], dtype=F64
        )
        ctx = _nut_ctx(land)
        their = theirs.soil_pool_transformation()
        in_, out_ = ours.soil_pool_transformation(active=TRUE)
        _assert_same(theirs, ours, their, (in_, out_), small)
        ammonia = float(out_[NUT.column("ammonia")])
        assert bool(ctx.events["nutrient_pool_not_tracking"]["land"]) is small
        if small:
            for name in ("nitrate", "ammonia"):
                assert float(out_[NUT.column(name)]) == 1e-7
                assert float(ours.storage[NUT.column(name)]) == 0.0
        else:
            assert ammonia > 1e-3


def _uptake_scale(parameters, days, autumn) -> float:
    """The size of the day's common uptake (only to scale the random pools)."""
    u1, u2, u3 = (parameters[f"uptake{i}"] for i in (1, 2, 3))
    d = days - 25 if autumn else days
    par = (u1 - u2) * math.exp(-u3 * d)
    total = u2 + par
    shape = u1 * u2 * u3 * par / total**2 if total > 0 else 0.0
    return abs(shape) * 1e-3 * parameters["area"] + 1e-6


@pytest.mark.parametrize("season", ["spring", "autumn"])
def test_calc_crop_uptake_equals_wsimods(season):
    """``land.py:1858-1926``, bitwise, on random parameters, days, temperatures, tanks and
    pools: every cap binds on some draw and not on others, autumn's ``temp_func = max(0,
    min(1, (T - 5) / 20))`` and ``days_after_sow - 25``, ``uptake_par + uptake2 <= 0``
    skipping the shape (a negative ``uptake1``), no crop (``None`` and the sowing day 0), the
    available water ``max`` both ways. All the uptake leaves as nitrate, and the dissolved
    inorganic pool is extracted P first, then N."""
    autumn = season == "autumn"
    rng = random.Random(53 if autumn else 59)
    seen: dict = {}
    order_checked = False
    for j in range(6):
        parameters = _soil_parameters(rng)
        if j == 5:
            parameters["uptake1"] = -rng.uniform(2.0, 6.0)
        theirs, land, ours = _soil_pair(parameters, sowing_day=250 if autumn else 90)
        assert theirs.autumn_sow is autumn and ours.autumn_sow is autumn
        wp_volume = theirs.wilting_point_m * parameters["area"]
        for k in range(60):
            days = (-1, 0)[k] if k < 2 else rng.randint(1, 40 if k % 3 else 300)
            temperature = rng.uniform(-5.0, 32.0)
            tank = {**_random_tank(rng, volume=wp_volume * rng.uniform(0.6, 1.6))}
            tank["temperature"] = temperature
            scale = _uptake_scale(parameters, max(days, 1), autumn)
            pools = _random_pools(rng)
            pools[DI] = [scale * rng.uniform(0.0, 4.0), scale * rng.uniform(0.0, 1.0)]
            if k % 4 == 0:
                tank["nitrate"] = scale * rng.uniform(0.0, 0.2)  # the tank clamps
            _set_state(theirs, ours, tank, pools, days=days)
            ctx = _nut_ctx(land)
            their = theirs.calc_crop_uptake()
            in_, out_ = ours.calc_crop_uptake(active=TRUE)
            _assert_same(theirs, ours, their, (in_, out_), (j, k))
            assert not in_.any()
            others = [n for n in NUT.names if n not in ("nitrate", "phosphate")]
            assert not any(float(out_[NUT.column(n)]) for n in others)
            sites = [d.site for d in ctx.decisions if "dissolved_inorganic_pool/extract" in d.site]
            if sites:
                di = f"{S7}.nutrient_pool.dissolved_inorganic_pool/extract"
                assert sites == [f"{di}/clip:P", f"{di}/clip:N"]
                order_checked = True
            _seen(ctx, seen)
            assert not bool(ctx.events["crop_uptake_dry_soil"]["land"])
    branches = ["crop", "shape", "available_water", "cap:P", "cap:N", "dry_soil"]
    if autumn:
        branches += ["temperature_cap", "temperature_floor"]
    else:
        assert not [s for s in seen if "temperature_" in s]
    for branch in branches:
        expected = {False} if branch == "dry_soil" else {True, False}
        assert seen[f"{S7}/calc_crop_uptake/{branch}"] == expected, branch
    assert seen[f"{S7}/pull_pollutants/clip:nitrate"] == {True, False}
    assert order_checked
    if autumn:  # the tie uptake_par + uptake2 == 0 (uptake1 = 0, d = 25 - 25) skips the shape
        theirs, land, ours = _soil_pair({**_soil_parameters(rng), "uptake1": 0.0}, sowing_day=250)
        tank = {**_random_tank(rng), "temperature": 15.0}
        _set_state(theirs, ours, tank, _random_pools(rng, scale=1e-3), days=25)
        ctx = _nut_ctx(land)
        their = theirs.calc_crop_uptake()
        in_, out_ = ours.calc_crop_uptake(active=TRUE)
        _assert_same(theirs, ours, their, (in_, out_), "tie")
        (d,) = [d for d in ctx.decisions if d.site == f"{S7}/calc_crop_uptake/shape"]
        assert not bool(d.outcome) and float(d.margin) == 0.0 and not out_.any()


def test_crop_uptake_from_an_empty_soil_is_zero():
    """L7, Y18: ``V = 0`` with a crop (``ET_depletion_factor = 0.5``, ``days_after_sow = 1``).
    WSIMOD raises ``ZeroDivisionError`` at ``land.py:1899``; noodl gives ``N_max = P_max = 0``,
    so nothing is taken up, and ``"crop_uptake_dry_soil"`` is true."""
    import traceback

    rng = random.Random(61)
    parameters = {"ET_depletion_factor": 0.5, **_soil_parameters(rng)}
    theirs, land, ours = _soil_pair(parameters)
    theirs.ET_depletion_factor = 0.5
    tank = {**_random_tank(rng, volume=0.0)}
    pools = _random_pools(rng)
    _set_state(theirs, ours, tank, pools, days=1)
    with pytest.raises(ZeroDivisionError) as raised:
        theirs.calc_crop_uptake()
    frame = traceback.extract_tb(raised.value.__traceback__)[-1]
    assert frame.filename.endswith("land.py") and frame.lineno == 1899
    ctx = _nut_ctx(land)
    in_, out_ = ours.calc_crop_uptake(active=TRUE)
    assert not in_.any() and not out_.any()
    assert torch.equal(ours.nutrient_pool.storage, torch.tensor(pools, dtype=F64))
    assert bool(ctx.events["crop_uptake_dry_soil"]["land"])
    (d,) = [d for d in ctx.decisions if d.site == f"{S7}/calc_crop_uptake/dry_soil"]
    assert bool(d.outcome)


def test_denitrification_equals_wsimods():
    """``land.py:2096-2152``, bitwise, on random parameters, soils from below ``limpar`` to above
    field capacity, tanks and pools: each branch, the clamped nitrate, and nothing at all where
    ``smc / fc_m <= limpar``. At ``smc == fc_m`` (area 2: exact) the ratio branch is taken and
    ``f == 1.0``."""
    rng = random.Random(67)
    seen: dict = {}
    dry = 0
    for _ in range(5):
        parameters = _soil_parameters(rng)
        theirs, land, ours = _soil_pair(parameters)
        fc_volume = theirs.field_capacity_m * parameters["area"]
        for k in range(60):
            tank = _random_tank(rng, volume=fc_volume * rng.uniform(0.3, 1.3))
            if k % 6 == 0:
                tank["nitrate"] = rng.uniform(0.0, 1e-4)
            pools = _random_pools(rng)
            _set_state(theirs, ours, tank, pools, tdf=rng.uniform(0.0, 1.5))
            ctx = _nut_ctx(land)
            their = theirs.denitrification()
            in_, out_ = ours.denitrification(active=TRUE)
            _assert_same(theirs, ours, their, (in_, out_), k)
            if tank["volume"] / parameters["area"] / theirs.field_capacity_m <= theirs.limpar:
                dry += 1
                assert not out_.any()
                assert torch.equal(ours.nutrient_pool.storage, torch.tensor(pools, dtype=F64))
            _seen(ctx, seen)
    assert seen[f"{S7}/denitrification/saturated"] == {True, False}
    assert seen[f"{S7}/denitrification/moist"] == {True, False}
    assert seen[f"{S7}/pull_pollutants/clip:nitrate"] == {True, False}
    assert dry
    # the tie: smc == fc_m takes the ratio branch, ((1 - limpar) / (1 - limpar)) ** exppar = 1.0
    parameters = {**_soil_parameters(rng), "area": 2.0}
    theirs, land, ours = _soil_pair(parameters)
    volume = theirs.field_capacity_m * 2.0
    assert volume / 2.0 == theirs.field_capacity_m
    pools = _random_pools(rng)
    tank = {**_random_tank(rng, volume=volume), "nitrate": 5.0}  # no clamp
    _set_state(theirs, ours, tank, pools, tdf=0.8)
    ctx = _nut_ctx(land)
    their = theirs.denitrification()
    in_, out_ = ours.denitrification(active=TRUE)
    _assert_same(theirs, ours, their, (in_, out_), "tie")
    decisions = {d.site.rsplit("/", 1)[1]: bool(d.outcome) for d in ctx.decisions}
    assert decisions["saturated"] is False and decisions["moist"] is True
    limpar, exppar = parameters["limpar"], parameters["exppar"]
    assert ((1.0 - limpar) / (1 - limpar)) ** exppar == 1.0
    din = pools[DI][0]
    conc = din / volume * 1e3
    expected = din * (conc / (conc + parameters["hsatINs"])) * 1.0 * 0.8 * parameters["denpar"]
    assert float(out_[NUT.column("nitrate")]) == expected


def test_a_batch_of_soil_processes_equals_wsimods_instances():
    """64 instances with their own parameters, tanks, pools and crop days, one of them masked:
    the five processes in WSIMOD's order on the batch equal, instance by instance, a live
    WSIMOD surface running the same five (bitwise), and the masked instance is unchanged.
    Python's ``**`` and ``exp`` are evaluated per instance (torch's batched ``pow`` and ``exp``
    can differ from them in the last bit)."""
    rng = random.Random(71)
    n = 64
    draws = [_soil_parameters(rng) for _ in range(n)]
    tanks, pools, days = [], [], []
    for parameters in draws:
        fc_volume = parameters["field_capacity"] * parameters["rooting_depth"] * parameters["area"]
        tanks.append({**_random_tank(rng, volume=fc_volume * rng.uniform(0.5, 1.2))})
        tanks[-1]["temperature"] = rng.uniform(1.0, 25.0)
        pools.append(_random_pools(rng, scale=0.01))
        days.append(rng.randint(1, 200))
    active = torch.ones(n, dtype=torch.bool)
    active[2] = False
    land = _nutrient_land()
    ours = land.surfaces[0]
    ours.bind({k: torch.tensor([p[k] for p in draws], dtype=F64) for k in draws[0]})
    ours.storage = torch.stack([NUT.to_vqip(t) for t in tanks])
    ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
    ours.days_after_sow = torch.tensor(days, dtype=F64)
    land.ctx = Ctx(NUT, FA, torch.Size((n,)), day=0, node_names=["land"])
    results = [getattr(ours, name)(active=active) for name in _SOIL_PROCESSES]
    for i in range(n):
        if not bool(active[i]):
            assert torch.equal(ours.storage[i], NUT.to_vqip(tanks[i]))
            assert torch.equal(ours.nutrient_pool.storage[i], torch.tensor(pools[i], dtype=F64))
            continue
        theirs, _, _ = _soil_pair(draws[i])
        theirs.storage = dict(tanks[i])
        _set_their_pools(theirs, pools[i])
        theirs.days_after_sow = days[i]
        for name, (in_, out_) in zip(_SOIL_PROCESSES, results, strict=True):
            their_in, their_out = getattr(theirs, name)()
            assert torch.equal(in_[i], _vector(their_in)), (i, name)
            assert torch.equal(out_[i], _vector(their_out)), (i, name)
        assert torch.equal(ours.storage[i], _vector(theirs.storage)), i
        assert torch.equal(ours.nutrient_pool.storage[i], _their_pools(theirs)), i


def test_the_batched_factors_are_pythons_per_instance():
    """The two factors and denitrification on a batch of 2000 soils equal WSIMOD's, instance by
    instance: torch's batched ``pow`` differs from Python's ``2 ** x``, ``x ** thetapow`` and
    ``x ** exppar`` in the last bit on about 1.6 % of values, so ``pow_python`` evaluates each
    instance (Approach rule 1)."""
    rng = random.Random(79)
    parameters = _soil_parameters(rng)
    theirs, land, ours = _soil_pair(parameters)
    n, area = 2000, parameters["area"]
    fc_m, wp_m = theirs.field_capacity_m, theirs.wilting_point_m
    temperatures = [rng.uniform(-2.0, 40.0) for _ in range(n)]
    volumes = [rng.uniform(wp_m * 0.9, fc_m * 1.1) * area for _ in range(n)]
    pools = [_random_pools(rng) for _ in range(n)]
    tank = {**_random_tank(rng), "nitrate": 5.0}
    ours.storage = torch.stack(
        [
            NUT.to_vqip({**tank, "temperature": t, "volume": v})
            for t, v in zip(temperatures, volumes, strict=True)
        ]
    )
    land.ctx = Ctx(NUT, FA, torch.Size((n,)), day=0, node_names=["land"])
    active = torch.ones(n, dtype=torch.bool)
    ours.calc_temperature_dependence_factor(active=active)
    ours.calc_soil_moisture_dependence_factor(active=active)
    ours.nutrient_pool.storage = torch.tensor(pools, dtype=F64)
    _, out_ = ours.denitrification(active=active)
    tdf = ours.nutrient_pool.temperature_dependence_factor.tolist()
    smdf = ours.nutrient_pool.soil_moisture_dependence_factor.tolist()
    for i, (temperature, volume) in enumerate(zip(temperatures, volumes, strict=True)):
        theirs.storage = {**tank, "temperature": temperature, "volume": volume}
        theirs.calc_temperature_dependence_factor()
        theirs.calc_soil_moisture_dependence_factor()
        assert tdf[i] == theirs.nutrient_pool.temperature_dependence_factor, i
        assert smdf[i] == theirs.nutrient_pool.soil_moisture_dependence_factor, i
        _set_their_pools(theirs, pools[i])
        assert torch.equal(out_[i], _vector(theirs.denitrification()[1])), i
        assert torch.equal(ours.nutrient_pool.storage[i], _their_pools(theirs)), i


def test_gradcheck_on_the_factors_and_uptake():
    """Spec, Verification 4: the gradients of the two factors (through ``2 ** x``, ``T / 5`` and
    ``** thetapow``) and of the uptake (through ``exp`` and ``** 2``), away from 0 °C, 5 °C and
    the caps."""
    land = _nutrient_land(area=20.0, sowing_day=90)
    ours = land.surfaces[0]
    base = {**_random_tank(random.Random(73), volume=6.0)}

    def factors(temperature, volume, thetapow):
        ours.bind({"thetapow": thetapow})
        tank = NUT.to_vqip(base)
        hot = torch.zeros(1 + NUT.K, dtype=torch.bool)
        hot[NUT.column("temperature")] = True
        hot0 = torch.zeros(1 + NUT.K, dtype=torch.bool)
        hot0[0] = True
        tank = torch.where(hot, temperature, torch.where(hot0, volume, tank))
        ours.storage = tank
        pool = ours.nutrient_pool
        pool.temperature_dependence_factor = torch.tensor(0.0, dtype=F64)
        pool.soil_moisture_dependence_factor = torch.tensor(0.0, dtype=F64)
        _nut_ctx(land)
        ours.calc_temperature_dependence_factor(active=TRUE)
        ours.calc_soil_moisture_dependence_factor(active=TRUE)
        return pool.temperature_dependence_factor, pool.soil_moisture_dependence_factor

    wp_volume = float(ours.wilting_point_m) * 20.0
    fc_volume = float(ours.field_capacity_m) * 20.0
    for temperature in (2.3, 12.7):
        inputs = (
            torch.tensor(temperature, dtype=F64, requires_grad=True),
            torch.tensor((wp_volume + fc_volume) / 2, dtype=F64, requires_grad=True),
            torch.tensor(1.7, dtype=F64, requires_grad=True),
        )
        assert torch.autograd.gradcheck(factors, inputs)

    def uptake(u1, u2, u3, din, dip):
        ours.bind({"uptake1": u1, "uptake2": u2, "uptake3": u3})
        ours.storage = NUT.to_vqip({**base, "nitrate": 1.0, "phosphate": 1.0})
        ours.nutrient_pool.storage = torch.stack(
            [torch.stack([din, dip]) if i == DI else torch.zeros(2, dtype=F64) for i in range(5)]
        )
        ours.days_after_sow = torch.tensor(30.0, dtype=F64)
        _nut_ctx(land)
        _, out_ = ours.calc_crop_uptake(active=TRUE)
        return out_[NUT.column("nitrate")], out_[NUT.column("phosphate")]

    inputs = tuple(
        torch.tensor(v, dtype=F64, requires_grad=True) for v in (15.0, 1.3, 0.02, 1.0, 1.0)
    )
    assert torch.autograd.gradcheck(uptake, inputs)
    # the caps: a small pool binds both, and the gradient reaches the pool
    inputs = tuple(
        torch.tensor(v, dtype=F64, requires_grad=True) for v in (15.0, 1.3, 0.02, 1e-4, 1e-5)
    )
    assert torch.autograd.gradcheck(uptake, inputs)
