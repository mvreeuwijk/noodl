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
from noodl.apps.wsimod.constants import SURFACE_INPUTS
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.growing import (
    GROWING_DEFAULTS,
    STRUCTURE_DEFAULTS,
    GrowingSurface,
    quick_interp,
)
from noodl.apps.wsimod.land import SURFACE_CLASSES, Land, surface_derived
from noodl.apps.wsimod.vqip import QUANTITY_ONLY

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


_NITRATE = {
    "pollutants": ["nitrate", "temperature"],
    "additive_pollutants": ["nitrate"],
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
