"""``Land``, ``Surface``, ``ImperviousSurface`` and ``PerviousSurface`` (noodl only, except where a
test says it drives live WSIMOD).

The values asserted are WSIMOD's, worked out from ``land.py`` by hand; the replay
(``tests/verification/test_wsimod_replay.py``) checks the same code against live WSIMOD on the
``impervious_only``, ``pervious_only``, ``evaporation_clamp``, ``temperature_blending``,
``land_percolation_threshold`` and ``residential_with_land`` cases.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, SurfaceSpec, read_wsimod_config
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.engine import Ctx, Engine, make_node
from noodl.apps.wsimod.land import (
    SOIL_TEMPERATURE_DEFAULTS,
    SURFACE_CLASSES,
    ImperviousSurface,
    Land,
    PerviousSurface,
    Surface,
    pow_python,
    surface_derived,
)
from noodl.apps.wsimod.node import NODE_CLASSES
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from tests.apps.wsimod._replay import ScriptedArc
from tests.verification._wsimod_reference import ArcCall

P = QUANTITY_ONLY
PP = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)

_NEW_ARCS: list = []  # the arcs built since the last day(); day() gives them the context


def surface(cls: str, name: str, **parameters) -> SurfaceSpec:
    return SurfaceSpec(name, cls, parameters)


def make_land(surfaces, name: str = "land", pollutants: Pollutants = P, **parameters) -> Land:
    return make_node(
        NodeSpec(name, "Land", "Land", parameters, {}, list(surfaces)), None, pollutants
    )


def make(cls: str, name: str, pollutants: Pollutants = P, **parameters):
    return make_node(NodeSpec(name, cls, cls, parameters), None, pollutants)


def day(*objects, shape=(), pollutants: Pollutants = P):
    ctx = Ctx(pollutants, FA, torch.Size(shape), day=0, node_names=[], record_decisions=True)
    for obj in (*objects, *_NEW_ARCS):
        obj.ctx = ctx
    _NEW_ARCS.clear()
    return ctx


def arc(name, a, b, *, capacity=UNBOUNDED_CAPACITY, pollutants: Pollutants = P):
    spec = ArcSpec(name, "Arc", a.name, b.name, capacity=capacity)
    _NEW_ARCS.append(ARC_CLASSES["Arc"](spec, a, b, pollutants))
    return _NEW_ARCS[-1]


def forcing(precipitation, et0, temperature=10.0) -> dict:
    return {
        "precipitation": torch.as_tensor(precipitation, dtype=torch.float64),
        "et0": torch.as_tensor(et0, dtype=torch.float64),
        "temperature": torch.as_tensor(temperature, dtype=torch.float64),
    }


def call(name, method, request, reply, *, tag="default") -> ArcCall:
    request = {"volume": request} if not isinstance(request, dict) else request
    reply = {"volume": reply} if not isinstance(reply, dict) else reply
    return ArcCall(
        name, method, request, tag, False, reply, 0.0, {"volume": 0.0}, 0.0, {"volume": 0.0}
    )


def scripted(name, calls, *, out_type):
    return ScriptedArc(
        ArcSpec(name, "Arc", "land", "other"),
        calls,
        P,
        scale=1.0,
        in_type="Land",
        out_type=out_type,
    )


def volume(x) -> float:
    return float(x[..., 0])


def test_the_classes_are_registered_under_the_names_wsimod_gives_them():
    assert NODE_CLASSES["Land"] is Land and Land.wsimod_type == "Land"
    assert Land.step_methods == ("run", "apply_irrigation")
    assert SURFACE_CLASSES == {
        "Surface": Surface,
        "ImperviousSurface": ImperviousSurface,
        "PerviousSurface": PerviousSurface,
    }
    assert Land.replay_bound == 0.0


# --- ImperviousSurface --------------------------------------------------------------------------
def test_impervious_surface_ponds_and_evaporates():
    """``land.py:645-721`` with area 10, ``pore_depth`` 0 (capacity 0) and ``et0_to_e`` 1. Wet:
    ``(P - E) * area`` is forced in and all of it ponds to the sewer. Dry: the pores evaporate
    ``min((E - P) * area, storage)`` and the evaporation is that plus ``P * area``."""
    land = make_land([surface("ImperviousSurface", "roads", area=10.0, pore_depth=0.0)])
    roads = land.surfaces[0]
    sewer = make("Sewer", "drain", capacity=1e9)
    to_drain = arc("to_drain", land, sewer)
    day(land, sewer)
    land.bind(forcing(0.003, 0.001))
    land.run(active=TRUE)
    assert float(to_drain.flow_in) == (0.003 - 0.001) * 10
    assert volume(roads.storage) == 0.0
    assert volume(roads.evaporation) == 0.001 * 10
    assert volume(roads.precipitation) == 0.003 * 10
    assert volume(land.running_inflow_mb) == 0.003 * 10
    assert volume(land.running_outflow_mb) == 0.001 * 10

    for stored in (0.003, 0.01):  # the store limits the evaporation, then the demand does
        land.end_timestep()
        to_drain.end_timestep()
        sewer.end_timestep()
        ctx = day(land, sewer, to_drain)
        roads.storage = P.to_vqip({"volume": stored})
        land.bind(forcing(0.0005, 0.001))
        land.run(active=TRUE)
        pores = min((0.001 - 0.0005) * 10, stored)
        assert volume(roads.evaporation) == pores + 0.0005 * 10
        assert float(to_drain.flow_in) == stored - pores  # what is left ponds (capacity 0)
        assert volume(roads.storage) == 0.0
        (dry,) = [d for d in ctx.decisions if d.site.endswith("precipitation_evaporation/dry")]
        assert bool(dry.outcome) and float(dry.margin) == 0.0005 - 0.001

    # the tie P == E * et0_to_e is not dry (land.py:660 is a strict <): a net rain of 0 at the
    # air temperature is forced in, and the blend (0 x 10 + 0.1 x 3) / 3 moves the stored
    # temperature by an ulp, which the dry branch would not
    land = make_land(
        [surface("ImperviousSurface", "roads", area=10.0, pore_depth=1.0)], pollutants=PP
    )
    roads = land.surfaces[0]
    ctx = day(land, pollutants=PP)
    roads.storage = PP.to_vqip({"volume": 3.0, "temperature": 0.1})
    land.bind(forcing(0.001, 0.001, 10.0))
    land.run(active=TRUE)
    (dry,) = [d for d in ctx.decisions if d.site.endswith("precipitation_evaporation/dry")]
    assert not bool(dry.outcome) and float(dry.margin) == 0.0
    assert volume(roads.evaporation) == 0.001 * 10
    assert float(roads.storage[2]) == (10.0 * 0.0 + 0.1 * 3.0) / 3.0 != 0.1


def test_an_impervious_surface_starting_above_capacity_empties_on_day_one():
    """Review Focus 4: oxford's urban surface (area 1e7, capacity 0, initial storage 5e6) into
    an accepting sewer: the 5e6 and the day's net rain leave on day 1 (``land.py:699-721``)."""
    land = make_land(
        [surface("ImperviousSurface", "urban", area=1e7, pore_depth=0.0, initial_storage=5e6)]
    )
    urban = land.surfaces[0]
    assert float(urban.capacity) == 0.0
    sewer = make("Sewer", "drain", capacity=1e9)
    to_drain = arc("to_drain", land, sewer)
    day(land, sewer)
    land.bind(forcing(0.004, 0.001))
    land.run(active=TRUE)
    net = (0.004 - 0.001) * 1e7
    assert volume(urban.storage) == 0.0
    assert float(to_drain.flow_in) == 5e6 + net
    assert volume(sewer.sewer_tank.storage) == 5e6 + net


def test_a_rejected_runoff_is_forced_back_into_the_surface():
    """``land.py:716``: what the sewer refuses stays on the surface, above its capacity."""
    land = make_land([surface("ImperviousSurface", "roads", area=100.0, pore_depth=0.001)])
    roads = land.surfaces[0]
    sewer = make("Sewer", "drain", capacity=0.25)
    to_drain = arc("to_drain", land, sewer)
    day(land, sewer)
    land.bind(forcing(0.01, 0.0))
    land.run(active=TRUE)
    assert float(to_drain.flow_in) == 0.25
    assert volume(roads.storage) == 0.01 * 100 - 0.25  # capacity 0.1 is not a limit on force


# --- PerviousSurface: IHACRES -------------------------------------------------------------------
_RURAL = {
    "area": 100.0,
    "depth": 0.5,
    "total_porosity": 0.4,
    "field_capacity": 0.3,
    "wilting_point": 0.12,
    "infiltration_capacity": 0.003,
    "surface_coefficient": 0.05,
    "percolation_coefficient": 0.75,
    "et0_coefficient": 0.5,
    "ihacres_p": 3,
}


def _ihacres_by_hand(storage, precipitation, et0, k=_RURAL):
    """``land.py:913-1044`` in Python floats, every intermediate as WSIMOD computes it."""
    depth = k["depth"] * k["total_porosity"]
    fc_m, wp_m = k["field_capacity"] * k["depth"], k["wilting_point"] * k["depth"]
    capacity = k["area"] * depth
    evaporation_depth = et0 * k["et0_coefficient"]
    infiltrated = min(precipitation, k["infiltration_capacity"])
    excess = max(precipitation - infiltrated, 0)
    vol = max(capacity - storage, 0)
    deficit = storage * (vol / storage) / k["area"]  # get_excess's v_change_vqip
    evaporation = evaporation_depth * min(1, math.exp(2 * (1 - deficit / (depth - wp_m))))
    outflow = infiltrated * (1 - min(1, (deficit / (depth - fc_m)) ** k["ihacres_p"]))
    evaporation = min(evaporation, precipitation + storage / k["area"])
    surface = outflow * k["surface_coefficient"] * k["area"]
    percolation = (
        outflow * (1 - k["surface_coefficient"]) * k["percolation_coefficient"] * k["area"]
    )
    subsurface = (
        outflow * (1 - k["surface_coefficient"]) * (1 - k["percolation_coefficient"]) * k["area"]
    )
    recharge = (infiltrated - evaporation - outflow) * k["area"]
    excess = excess * k["area"] + surface
    total = recharge + subsurface + percolation
    return {
        "deficit": deficit,
        "outflow": outflow,
        "evaporation": evaporation * k["area"],
        "precipitation": precipitation * k["area"],
        "excess": excess,
        "percolation": percolation,
        "subsurface": subsurface,
        "recharge": recharge,
        "total": total,
    }


def _one_pervious_day(storage, precipitation, et0, *, pollutants=P, **parameters):
    spec = surface("PerviousSurface", "rural", **{**_RURAL, **parameters})
    land = make_land([spec], pollutants=pollutants)
    rural = land.surfaces[0]
    ctx = day(land, pollutants=pollutants)
    rural.storage = pollutants.to_vqip({"volume": storage})
    land.bind(forcing(precipitation, et0))
    land.run(active=TRUE)
    return land, rural, ctx


def test_ihacres_by_hand():
    """Two points away from the ``min(1, .)`` kinks. Wet (deficit 0.02 below ``D - fc_m =
    0.05``): ``(0.4) ** 3`` cuts the outflow and the soil gains water, which is forced in and
    pulled out as percolation and subsurface flow. Dry (deficit 0.15 above ``D - wp_m = 0.14``):
    the evaporation is cut by ``exp(2 (1 - 0.15/0.14)) < 1``, nothing flows out, and the store
    evaporates."""
    # wet: 18 of capacity 20
    hand = _ihacres_by_hand(18.0, 0.004, 0.002)
    land, rural, ctx = _one_pervious_day(18.0, 0.004, 0.002)
    assert hand["deficit"] == (20.0 - 18.0) / 100 and hand["total"] > 0
    assert 0 < hand["outflow"] < 0.003
    assert volume(rural.precipitation) == hand["precipitation"]
    assert volume(rural.evaporation) == hand["evaporation"] == 0.002 * 0.5 * 100
    assert float(rural.tank_recharge) == hand["recharge"]
    assert volume(rural.percolation) == hand["percolation"]
    assert volume(rural.subsurface_flow) == hand["subsurface"]
    assert volume(rural.infiltration_excess) == hand["excess"]
    stored = 18.0 + hand["total"]
    stored = stored - hand["subsurface"]
    stored = stored - hand["percolation"]
    assert volume(rural.storage) == stored
    # route (land.py:1046-1058): the flows reached the residence tanks, and run sent part on
    # with no groundwater arc the day's outflow V / 50 is rejected and forced back (land.py:213)
    outflow = hand["percolation"] / 50
    assert volume(land.percolation.storage) == hand["percolation"] - outflow + outflow
    through = [d for d in ctx.decisions if d.site.endswith("ihacres/through_soil")]
    assert bool(through[0].outcome)

    # dry: 5 of capacity 20
    hand = _ihacres_by_hand(5.0, 0.0, 0.004)
    land, rural, ctx = _one_pervious_day(5.0, 0.0, 0.004)
    assert hand["deficit"] == 0.15 and hand["outflow"] == 0 and hand["total"] < 0
    assert volume(rural.evaporation) == hand["evaporation"] < 0.004 * 0.5 * 100
    assert float(rural.tank_recharge) == hand["recharge"]
    assert volume(rural.storage) == 5.0 - (-hand["total"])
    assert volume(rural.percolation) == volume(rural.subsurface_flow) == 0.0
    assert volume(rural.infiltration_excess) == 0.0
    assert not ctx.events.get("inaccurate_evaporation", {}).get("land", torch.tensor(False))
    kinks = {d.site.rsplit("/", 1)[1]: bool(d.outcome) for d in ctx.decisions if "kink" in d.site}
    assert kinks == {"evaporation_kink": True, "outflow_kink": False}


def test_an_infiltration_excess_in_the_evaporating_branch_is_the_inaccurate_evaporation():
    """``land.py:1003-1021``: in the evaporating branch the check ``|evap + P_inf * area -
    evaporation - excess|`` is the excess itself, so rain above the infiltration capacity on a
    dry soil prints; WSIMOD keeps the IHACRES evaporation as the outflow term (the event)."""
    land, rural, ctx = _one_pervious_day(5.0, 0.004, 0.02)
    assert float(rural.tank_recharge) < 0
    assert volume(rural.infiltration_excess) == pytest.approx((0.004 - 0.003) * 100)
    assert bool(ctx.events["inaccurate_evaporation"]["land"])
    # residence time 1: all of it runs off, and with no River or Node it is forced back
    assert volume(land.surface_runoff.storage) == pytest.approx(
        volume(rural.infiltration_excess), rel=1e-15
    )


def test_the_soil_temperature_is_neutral_with_the_neutral_weights():
    """``land.py:1060-1079`` with ``w_air = w_deep = 0``, ``w_prev = 1``: the storage keeps its
    temperature, rain enters at the air temperature and the blend is WSIMOD's ``sum_vqip``."""
    neutral = {"soil_temp_w_prev": 1, "soil_temp_w_air": 0, "soil_temp_w_deep": 0}
    land, rural, _ = _one_pervious_day(18.0, 0.004, 0.002, pollutants=PP, **neutral)
    hand = _ihacres_by_hand(18.0, 0.004, 0.002)
    # storage 18 at temperature 0 blended with total at 10, before the pulls keep it
    blended = (10.0 * hand["total"] + 0.0 * 18.0) / (18.0 + hand["total"])
    assert float(rural.storage[1 + PP.names.index("temperature")]) == blended
    assert float(rural.infiltration_excess[2]) == 10.0


def _soil_temperature(weights, storage_temperature=12.0, air=18.0, *, shape=()):
    """One call of ``calculate_soil_temperature`` with the storage at that temperature."""
    land = make_land([surface("PerviousSurface", "rural", **_RURAL)], pollutants=PP)
    rural = land.surfaces[0]
    day(land, pollutants=PP, shape=shape)
    rural.storage = PP.to_vqip({"volume": 20.0, "temperature": storage_temperature})
    parameters = {
        f"surfaces.rural.{k}": torch.as_tensor(v, dtype=torch.float64) for k, v in weights.items()
    }
    land.bind({**forcing(0.0, 0.0, air), **parameters})
    in_, out_ = rural.calculate_soil_temperature(active=TRUE)
    return rural, in_, out_


def test_the_soil_temperature_moves_with_wsimods_default_weights():
    """``land.py:1073-1079`` with the defaults of ``:809-812``; Python's expression, bitwise."""
    weights = {
        "soil_temp_w_prev": 0.1,
        "soil_temp_w_air": 0.6,
        "soil_temp_w_deep": 0.1,
        "soil_temp_deep": 10,
    }
    rural, in_, out_ = _soil_temperature(weights)
    expected = (12.0 * 0.1 + 18.0 * 0.6 + 10 * 0.1) / (0.6 + 0.1 + 0.1)
    assert torch.equal(
        rural.storage[1 + PP.names.index("temperature")],
        torch.tensor(expected, dtype=torch.float64),
    )
    assert float(in_.abs().sum()) == 0 and float(out_.abs().sum()) == 0


def test_a_zero_previous_weight_forgets_the_storage_temperature():
    weights = {"soil_temp_w_prev": 0, "soil_temp_w_air": 1, "soil_temp_w_deep": 0}
    rural, _, _ = _soil_temperature(weights)
    expected = (12.0 * 0 + 18.0 * 1 + 10 * 0) / (1 + 0 + 0)
    assert float(rural.storage[1 + PP.names.index("temperature")]) == expected == 18.0


def test_a_zero_total_soil_weight_is_refused():
    """``land.py:1073`` divides by the total weight: refused in the configuration and at bind."""
    zero = {"soil_temp_w_prev": 0, "soil_temp_w_air": 0, "soil_temp_w_deep": 0}
    mapping = _mapping([_pervious()], overrides={"rural": zero}, lists=_LISTS)
    with pytest.raises(ValueError, match="total weight"):
        _validate(mapping, pollutants=True)
    # a quantity-only run never divides
    _validate(mapping, pollutants=False)
    land = make_land([surface("PerviousSurface", "rural", **_RURAL)], pollutants=PP)
    day(land, shape=(2,), pollutants=PP)
    batch = torch.tensor([0.0, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"rural.*total weight"):
        land.bind(
            {
                "surfaces.rural.soil_temp_w_prev": batch,
                "surfaces.rural.soil_temp_w_air": torch.zeros(2, dtype=torch.float64),
                "surfaces.rural.soil_temp_w_deep": torch.zeros(2, dtype=torch.float64),
            }
        )


def test_deposition_refusals_name_only_3c():
    mapping = _mapping(
        [_pervious(pollutant_load={"phosphate": 1e-6})],
        overrides={"rural": _NEUTRAL},
        lists=_LISTS,
    )
    with pytest.raises(ValueError, match="3c") as error:
        _validate(mapping, pollutants=True)
    assert "3b" not in str(error.value)
    data = _mapping(
        [
            {
                "type_": "ImperviousSurface",
                "surface": "r",
                "data_input_dict": {("nhx-dry", "2000-01-01"): 1e-6},
            }
        ],
        lists=_LISTS,
    )
    with pytest.raises(ValueError, match="3c") as error:
        _validate(data, pollutants=True)
    assert "3b" not in str(error.value)


def test_any_soil_weights_with_a_non_zero_total_are_accepted():
    assert _validate(_mapping([_pervious()], lists=_LISTS), pollutants=True) == []
    zero_prev = {**_NEUTRAL, "soil_temp_w_prev": 0, "soil_temp_w_air": 1}
    mapping = _mapping([_pervious()], overrides={"rural": zero_prev}, lists=_LISTS)
    assert _validate(mapping, pollutants=True) == []


# --- Land.run -----------------------------------------------------------------------------------
def test_a_percolation_rejection_above_float_accuracy_is_forced_back_and_below_it_dropped():
    """``land.py:210-215``: the percolation tank's outflow to the groundwater arc; a rejection
    above ``FLOAT_ACCURACY`` is forced back, one at or below it vanishes (dropped water)."""
    for rejected, kept in ((5e-11, True), (5e-12, False)):
        land = make_land([], percolation_residence_time=2)
        land.percolation.storage = P.to_vqip({"volume": 1.0})
        gw = scripted("to_gw", [call("to_gw", "send_push_request", 0.5, rejected)], out_type="x")
        gw.out_type = "Groundwater"
        land.connect(gw, "out")
        ctx = day(land)
        land.bind(forcing(0.0, 0.0))
        land.run(active=TRUE)
        gw.assert_exhausted()
        assert volume(land.percolation.storage) == 0.5 + (rejected if kept else 0.0)
        assert float(ctx.dropped[0]) == (0.0 if kept else rejected)


def test_a_runoff_rejection_is_split_pro_rata_and_forced_back():
    """``land.py:217-244``: surface (residence 1) and subsurface (residence 5) runoff are summed
    and pushed to River/Node; a rejection is split by their volumes."""
    land = make_land([])
    land.surface_runoff.storage = P.to_vqip({"volume": 6.0})
    land.subsurface_runoff.storage = P.to_vqip({"volume": 20.0})
    river = scripted("to_river", [call("to_river", "send_push_request", 10.0, 5.0)], out_type="x")
    river.out_type = "River"
    land.connect(river, "out")
    day(land)
    land.bind(forcing(0.0, 0.0))
    land.run(active=TRUE)
    river.assert_exhausted()
    assert volume(land.surface_runoff.storage) == 0.0 + 5.0 * 6.0 / 10.0
    assert volume(land.subsurface_runoff.storage) == 16.0 + 5.0 * 4.0 / 10.0


def test_a_land_without_runoff_pushes_none():
    land = make_land([])
    river = scripted("to_river", [], out_type="x")
    river.out_type = "River"
    land.connect(river, "out")
    ctx = day(land)
    land.bind(forcing(0.0, 0.0))
    land.run(active=TRUE)
    (runoff,) = [d for d in ctx.decisions if d.site == "node:land/run/runoff"]
    assert not bool(runoff.outcome)


# --- handlers (W4, W16) ------------------------------------------------------------------------
def test_sewer_ponding_into_land_is_forced_into_the_first_impervious_surface():
    """W4, the evident intent of ``push_set_sewer`` (``land.py:246-264``): the ``"Sewer"`` check
    accepts everything when there is an ``ImperviousSurface`` and the push goes into the first
    one; without one the check is empty and the push is refused."""
    land = make_land(
        [
            surface("PerviousSurface", "fields", area=10.0),
            surface("ImperviousSurface", "roads", area=10.0),
            surface("ImperviousSurface", "roofs", area=10.0),
        ]
    )
    day(land)
    three = P.to_vqip({"volume": 3.0})
    assert volume(land.push_check(three, "Sewer", active=TRUE)) == 3.0
    assert volume(land.push_check(None, "Sewer", active=TRUE)) == UNBOUNDED_CAPACITY
    assert volume(land.push_set(three, "Sewer", active=TRUE)) == 0.0
    assert [volume(s.storage) for s in land.surfaces] == [0.0, 3.0, 0.0]
    dry = make_land([surface("PerviousSurface", "fields", area=10.0)])
    day(dry)
    assert volume(dry.push_check(three, "Sewer", active=TRUE)) == 0.0
    assert volume(dry.push_set(three, "Sewer", active=TRUE)) == 3.0
    # default pushes are denied (land.py:120-121)
    assert volume(land.push_check(three, active=TRUE)) == 0.0
    assert volume(land.push_set(three, active=TRUE)) == 3.0


def test_garden_demand_is_zero_without_a_garden_surface():
    """W16: a ``ResidentialDemand`` asks its ``Land`` for the garden capacity once
    (``get_connected`` with ``("Demand", "Garden")``, ``demand.py:250-301``); the land answers 0
    (``land.py:93-94``), and the garden push of 0 is still made."""
    homes = make("ResidentialDemand", "homes")
    land = make_land([surface("PerviousSurface", "lawn", area=100.0)])
    to_land = arc("to_land", homes, land)
    ctx = day(homes, land)
    homes.bind({"temperature": torch.tensor(10.0, dtype=torch.float64)})
    homes.create_demand(active=TRUE)
    asked = [d for d in ctx.decisions if d.site == "node:homes/get_connected/push:to_land/below_fa"]
    assert len(asked) == 1 and float(asked[0].margin) == -FA  # an avail of 0
    assert ctx.call("arc:to_land", "send_push_request") == 1  # one push was made
    assert float(to_land.flow_in) == 0.0
    assert volume(homes.total_demand) == 1 * 0.12  # the house demand alone


def test_apply_irrigation_does_nothing():
    land = make_land([surface("PerviousSurface", "lawn", area=100.0)])
    day(land)
    before = {k: v.clone() for k, v in land.stores().items()}
    land.apply_irrigation(active=TRUE)
    assert all(torch.equal(before[k], v) for k, v in land.stores().items())


# --- the decay, state and mass balance ---------------------------------------------------------
def test_surface_decay_reads_temperature_as_an_identity():
    """``tanks.py:442-447`` with empty decays reads the parent's ``("temperature", t)``
    (``core.py:484-487``) and leaves the storage unchanged."""
    land = make_land([surface("PerviousSurface", "lawn", area=100.0)], pollutants=PP)
    lawn = land.surfaces[0]
    day(land, pollutants=PP)
    lawn.storage = PP.to_vqip({"volume": 7.0, "phosphate": 0.3, "temperature": 12.0})
    before = lawn.storage.clone()
    land.bind(forcing(0.0, 0.0, 9.0))
    land.end_timestep()
    assert torch.equal(lawn.storage, before)
    land.bind({})
    with pytest.raises(KeyError, match="temperature"):
        lawn.end_timestep()


def test_the_stores_are_the_tanks_the_surfaces_and_the_running_balance():
    """D10: the residence tanks in ``dir()`` order, then each surface's storage and per-day
    flows, then the running mass balance; ``load_stores`` writes them back."""
    land = make_land(
        [
            surface("ImperviousSurface", "roads", area=10.0),
            surface("PerviousSurface", "lawn", area=10.0),
        ]
    )
    assert land.store_paths == (
        "percolation.storage",
        "subsurface_runoff.storage",
        "surface_runoff.storage",
        "surfaces.roads.storage",
        "surfaces.roads.evaporation",
        "surfaces.roads.precipitation",
        "surfaces.roads.total_decayed",
        "surfaces.lawn.storage",
        "surfaces.lawn.evaporation",
        "surfaces.lawn.infiltration_excess",
        "surfaces.lawn.percolation",
        "surfaces.lawn.precipitation",
        "surfaces.lawn.subsurface_flow",
        "surfaces.lawn.tank_recharge",
        "surfaces.lawn.total_decayed",
        "running_inflow_mb",
        "running_outflow_mb",
    )
    assert tuple(land.stores()) == land.store_paths
    assert land.store_paths == Land.store_paths_of(
        [("ImperviousSurface", "roads"), ("PerviousSurface", "lawn")]
    )
    values = {path: P.to_vqip({"volume": float(i)}) for i, path in enumerate(land.store_paths)}
    values["surfaces.lawn.tank_recharge"] = torch.tensor(2.5, dtype=torch.float64)
    assert land.carried_paths == (
        "percolation.storage",
        "subsurface_runoff.storage",
        "surface_runoff.storage",
        "surfaces.roads.storage",
        "surfaces.lawn.storage",
    )
    land.load_stores(values)
    assert all(torch.equal(land.stores()[k], v) for k, v in values.items())
    with pytest.raises(KeyError, match="nowhere"):
        land.load_stores({"surfaces.nowhere.storage": values["surfaces.roads.storage"]})


def test_the_mass_balance_terms_are_the_running_balance():
    """``land.py:158-162``: ``running_inflow_mb`` in, ``running_outflow_mb`` out (D11), the
    surfaces' precipitation and evaporation; reset at ``end_timestep`` (``:266-269``)."""
    assert Land.mass_balance == {
        "Land.running_inflow_mb": "in",
        "Land.running_outflow_mb": "out",
        "DecayTank.decayed": "out",
    }
    land = make_land(
        [
            surface("ImperviousSurface", "roads", area=10.0),
            surface("PerviousSurface", "lawn", area=20.0),
        ]
    )
    ctx = day(land)
    land.bind(forcing(0.002, 0.001))
    land.run(active=TRUE)
    land.record_terms()
    roads, lawn = land.surfaces
    inflow = volume(roads.precipitation) + volume(lawn.precipitation)
    outflow = volume(roads.evaporation) + volume(lawn.evaporation)
    assert volume(ctx.terms["Land.running_inflow_mb"]) == inflow == 0.002 * 10 + 0.002 * 20
    assert volume(ctx.terms["Land.running_outflow_mb"]) == outflow
    land.end_timestep()
    assert volume(land.running_inflow_mb) == volume(land.running_outflow_mb) == 0.0


def test_a_simple_deposition_of_zero_loads_is_pushed_at_volume_zero():
    """``land.py:381-382``, ``:510-532``: a configured ``pollutant_load`` (zero, the only form
    accepted with pollutants on) adds a volume-0 push of zero loads to the surface first."""
    spec = surface("ImperviousSurface", "roads", area=10.0, pollutant_load={"phosphate": 0.0})
    land = make_land([spec], pollutants=PP)
    roads = land.surfaces[0]
    assert roads.inflows == [roads.simple_deposition, roads.precipitation_evaporation]
    day(land, pollutants=PP)
    land.bind(forcing(0.0, 0.001))
    land.run(active=TRUE)
    assert volume(land.running_inflow_mb) == 0.0
    assert make_land([surface("ImperviousSurface", "r")]).surfaces[0].inflows[0].__name__ == (
        "precipitation_evaporation"
    )


# --- batches and gradients ----------------------------------------------------------------------
def test_a_batch_equals_its_single_runs_and_a_masked_instance_is_unchanged():
    """Three instances: wet, dry and masked (the wet forcing, inactive). Each active one equals
    its own single run bitwise; the masked one keeps its state."""
    precipitation = torch.tensor([0.004, 0.0, 0.004], dtype=torch.float64)
    et0 = torch.tensor([0.002, 0.004, 0.002], dtype=torch.float64)
    storage = torch.tensor([18.0, 5.0, 18.0], dtype=torch.float64)
    specs = [
        surface("ImperviousSurface", "roads", area=10.0, pore_depth=0.001),
        surface("PerviousSurface", "rural", **_RURAL),
    ]
    active = torch.tensor([True, True, False])
    batch = make_land(specs)
    day(batch, shape=(3,))
    batch.surfaces[1].storage = storage[..., None]
    batch.bind(forcing(precipitation, et0))
    batch.run(active=active)
    for i in range(2):
        one = make_land(specs)
        day(one)
        one.surfaces[1].storage = P.to_vqip({"volume": float(storage[i])})
        one.bind(forcing(precipitation[i], et0[i]))
        one.run(active=TRUE)
        for path, value in one.stores().items():
            if path.endswith(".total_decayed"):  # WsimodDay batches it; nothing here writes it
                continue
            assert torch.equal(batch.stores()[path][i], value.expand_as(batch.stores()[path][i]))
    fresh = make_land(specs)
    for path in fresh.stores():
        if path.endswith(".total_decayed"):
            continue
        expected = 18.0 if path == "surfaces.rural.storage" else 0.0
        assert float(batch.stores()[path][2].reshape(-1)[0]) == expected, path


def test_gradcheck_on_one_ihacres_day():
    """``torch.autograd.gradcheck`` through one IHACRES day at two points away from every kink.
    Wet: ``field_capacity``, ``ihacres_p``, ``et0_coefficient`` and the precipitation (the
    ``**`` and the soil-store pulls). Dry: ``wilting_point``, ``depth``, ``et0_coefficient`` and
    the precipitation (``exp`` and the evaporating branch)."""
    spec = surface("PerviousSurface", "rural", **_RURAL)

    def f(field_capacity, ihacres_p, et0_coefficient, precipitation):
        land = make_land([spec])
        day(land)
        land.surfaces[0].storage = P.to_vqip({"volume": 18.0})
        land.bind(
            {
                **forcing(precipitation, 0.002),
                "surfaces.rural.field_capacity": field_capacity,
                "surfaces.rural.ihacres_p": ihacres_p,
                "surfaces.rural.et0_coefficient": et0_coefficient,
            }
        )
        land.run(active=TRUE)
        rural = land.surfaces[0]
        return torch.stack(
            [
                rural.storage[0],
                rural.percolation[0],
                rural.subsurface_flow[0],
                land.surface_runoff.storage[0],
                rural.evaporation[0],
            ]
        )

    inputs = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.3, 3.0, 0.5, 0.004)
    )
    assert torch.autograd.gradcheck(f, inputs, eps=1e-7, atol=1e-8, rtol=1e-6)

    def g(wilting_point, depth, et0_coefficient, precipitation):
        land = make_land([spec])
        ctx = day(land)
        land.surfaces[0].storage = P.to_vqip({"volume": 5.0})
        land.bind(
            {
                **forcing(precipitation, 0.004),
                "surfaces.rural.wilting_point": wilting_point,
                "surfaces.rural.depth": depth,
                "surfaces.rural.et0_coefficient": et0_coefficient,
            }
        )
        land.run(active=TRUE)
        outcomes = {d.site.rsplit("/", 1)[1]: bool(d.outcome) for d in ctx.decisions}
        assert not outcomes["through_soil"] and outcomes["evaporation_kink"]
        rural = land.surfaces[0]
        return torch.stack([rural.storage[0], rural.evaporation[0], rural.tank_recharge])

    inputs = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.12, 0.5, 0.5, 0.0005)
    )
    assert torch.autograd.gradcheck(g, inputs, eps=1e-7, atol=1e-8, rtol=1e-6)


def test_pow_python_is_pythons_pow_bitwise_with_a_zero_slope_at_zero():
    torch.manual_seed(0)
    x = torch.rand(2000, dtype=torch.float64) * 1.5
    y = torch.rand(2000, dtype=torch.float64) * 12
    ours = pow_python(x, y)
    assert ours.tolist() == [a**b for a, b in zip(x.tolist(), y.tolist(), strict=True)]
    assert float(pow_python(torch.tensor(0.4, dtype=torch.float64), 10)) == 0.4**10
    x = torch.tensor([0.0, 0.5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    pow_python(x, y).sum().backward()
    assert x.grad.tolist() == [0.0, 3 * 0.5**2]
    assert float(y.grad) == pytest.approx(0.5**3 * math.log(0.5), rel=1e-15)


# --- validation --------------------------------------------------------------------------------
_DATES = ["2000-01-01", "2000-01-02"]


def _mapping(surfaces, *, overrides=None, data=None, lists=None, **land) -> dict:
    if data is None:
        data = {(v, d): 0.001 for v in ("precipitation", "et0", "temperature") for d in _DATES}
    mapping = {
        **(lists or {}),
        "dates": list(_DATES),
        "nodes": [
            {"name": "land", "type_": "Land", "surfaces": surfaces, "data_input_dict": data, **land}
        ],
    }
    if overrides is not None:
        mapping["overrides"] = {
            "nodes": {"land": {"type_": "Land", "name": "land", "surfaces": overrides}}
        }
    return mapping


def _validate(mapping, *, pollutants=False):
    network = read_wsimod_config(mapping)
    return Land.validate(network.nodes[0], network, pollutants=pollutants)


def _pervious(**values):
    return {"type_": "PerviousSurface", "surface": "rural", **values}


_NEUTRAL = {"soil_temp_w_prev": 1, "soil_temp_w_air": 0, "soil_temp_w_deep": 0}
_LISTS = {
    "pollutants": ["phosphate", "temperature"],
    "additive_pollutants": ["phosphate"],
    "non_additive_pollutants": ["temperature"],
}


def test_porosity_not_above_field_capacity_is_refused():
    """Defect row: ``land.py:942``, ``:950`` divide by ``depth * (total_porosity - fc)`` and
    raise to ``ihacres_p``; refused by name, after the overrides are applied."""
    with pytest.raises(ValueError, match="total_porosity"):
        _validate(_mapping([_pervious(total_porosity=0.3, field_capacity=0.3)]))
    with pytest.raises(ValueError, match="total_porosity.*wilting_point"):
        _validate(_mapping([_pervious(total_porosity=0.3, field_capacity=0.2, wilting_point=0.3)]))
    with pytest.raises(ValueError, match="total_porosity"):
        _validate(_mapping([_pervious()], overrides={"rural": {"total_porosity": 0.3}}))
    accepted = _mapping([_pervious(total_porosity=0.3, field_capacity=0.2)], lists=_LISTS)
    assert _validate(accepted) == [f"surfaces.rural.{k}" for k in SOIL_TEMPERATURE_DEFAULTS]


@pytest.mark.parametrize(
    ("mapping", "pollutants", "match"),
    [
        (
            _mapping(
                [
                    _pervious(pollutant_load={"phosphate": 1e-6}),
                ],
                lists=_LISTS,
            ),
            True,
            "3c",
        ),
        (
            _mapping(
                [
                    {
                        "type_": "ImperviousSurface",
                        "surface": "r",
                        "pollutant_load": {"temperature": 0},
                    }
                ],
                lists=_LISTS,
            ),
            True,
            "3c",
        ),
        (
            _mapping(
                [
                    {
                        "type_": "ImperviousSurface",
                        "surface": "r",
                        "data_input_dict": {("nhx-dry", "2000-01-01"): 1e-6},
                    }
                ],
                lists=_LISTS,
            ),
            True,
            "3c",
        ),
        (
            _mapping(
                [
                    {
                        "type_": "ImperviousSurface",
                        "surface": "r",
                        "data_input_dict": {("precipitation", "2000-01-01"): 1e-3},
                    }
                ]
            ),
            False,
            "'precipitation' is not read",
        ),
        (
            _mapping(
                [_pervious()],
                overrides={"rural": _NEUTRAL},
                lists={
                    "pollutants": ["phosphate"],
                    "additive_pollutants": ["phosphate"],
                    "non_additive_pollutants": [],
                },
            ),
            True,
            "'temperature' pollutant",
        ),
        (_mapping([_pervious(ihacres_p=-1)]), False, "ihacres_p"),
        (_mapping([_pervious(area=0)]), False, "area is 0"),
        (_mapping([_pervious(), _pervious()]), False, "two surfaces are named 'rural'"),
        (_mapping([{"type_": "Surface", "surface": "a.b"}]), False, "contains '.'"),
        (_mapping([], percolation_residence_time=0), False, "percolation_residence_time is 0"),
        (
            _mapping(
                [],
                data={
                    **{(v, d): 0.001 for v in ("precipitation", "et0") for d in _DATES},
                    ("temperature", "2000-01-01"): 5.0,
                },
            ),
            False,
            "no 'temperature' forcing on 2000-01-02",
        ),
    ],
)
def test_what_land_cannot_run_is_refused_by_name(mapping, pollutants, match):
    with pytest.raises(ValueError, match=match):
        _validate(mapping, pollutants=pollutants)


def test_a_quantity_only_run_ignores_loads_deposition_and_the_soil_temperature():
    """D21 and spec W10: with ``pollutants=False`` the loads, the deposition data and a
    non-neutral soil temperature are ignored and listed."""
    mapping = _mapping(
        [
            _pervious(pollutant_load={"phosphate": 1e-6}),
            {
                "type_": "ImperviousSurface",
                "surface": "roads",
                "data_input_dict": {("nhx-dry", "2000-01-01"): 1e-6},
            },
        ],
        lists=_LISTS,
    )
    assert _validate(mapping) == [
        "surfaces.rural.pollutant_load",
        "surfaces.rural.soil_temp_w_prev",
        "surfaces.rural.soil_temp_w_air",
        "surfaces.rural.soil_temp_w_deep",
        "surfaces.rural.soil_temp_deep",
        "surfaces.roads.data_input_dict",
    ]
    neutral = _mapping(
        [_pervious(pollutant_load={"phosphate": 0.0})], overrides={"rural": _NEUTRAL}, lists=_LISTS
    )
    assert _validate(neutral, pollutants=True) == []


def test_a_bound_porosity_at_field_capacity_is_refused():
    land = make_land([surface("PerviousSurface", "rural", **_RURAL)])
    with pytest.raises(ValueError, match="total_porosity is not above field_capacity"):
        land.bind({"surfaces.rural.total_porosity": torch.tensor([0.4, 0.3], dtype=torch.float64)})
    with pytest.raises(KeyError, match="no surface 'nowhere'"):
        land.bind({"surfaces.nowhere.area": torch.tensor(1.0)})
    with pytest.raises(KeyError, match="no parameter 'colour'"):
        land.bind({"surfaces.rural.colour": torch.tensor(1.0)})


# --- derived values (D5) -----------------------------------------------------------------------
def test_surfaces_derive_what_construction_derives():
    """D5: ``surface_derived`` and the ``Land`` residence tanks equal the live WSIMOD objects'
    after ``add_overrides`` (Task 4's mapping), to ``1e-15`` relative."""
    pytest.importorskip("wsimod")
    from tests.apps.wsimod.test_config import _everything
    from tests.verification._wsimod_reference import wsimod_model

    mapping = _everything()
    model = wsimod_model(mapping)
    network = read_wsimod_config(mapping)
    close = {"rel": 1e-15, "abs": 0}
    spec = next(n for n in network.nodes if n.cls == "Land")
    theirs = model.nodes[spec.name]
    classes = set()
    for surface_spec in spec.surfaces:
        wsimod_surface = theirs.get_surface(surface_spec.name)
        derived = surface_derived(surface_spec)
        for key, value in derived.items():
            assert value == pytest.approx(getattr(wsimod_surface, key), **close), key
        classes.add(surface_spec.cls)
    assert classes == {"ImperviousSurface", "PerviousSurface"}
    ours = make_node(spec, network, QUANTITY_ONLY)
    for name in ("surface_runoff", "subsurface_runoff", "percolation"):
        assert float(getattr(ours, name).residence_time) == getattr(theirs, name).residence_time
        assert float(getattr(ours, name).capacity) == getattr(theirs, name).capacity
    for ours_surface in ours.surfaces:
        wsimod_surface = theirs.get_surface(ours_surface.name)
        assert float(ours_surface.capacity) == pytest.approx(wsimod_surface.capacity, **close)


# --- the noodl-only ponding cycle ---------------------------------------------------------------
def test_land_sewer_ponding_runs_the_cycle_and_ponds_into_the_streets():
    """``land_sewer_ponding``: the ``Land -> Sewer`` and ``Sewer -> Land`` arcs are a legal cycle.
    No WSIMOD push makes a sewer exceed its capacity, so with fixed parameters it never ponds;
    a capacity bound lower between ``Land.run`` and ``make_discharge`` on the first full day (a
    driver) leaves the stored water above it, which ponds onto the land with the tag
    ``"Sewer"`` and enters the streets (W4's evident intent), where WSIMOD raises."""
    from tests.verification._wsimod_cases import CASES

    spec = CASES["land_sewer_ponding"]
    network = read_wsimod_config(spec.build())
    engine = Engine(network, pollutants=network.pollutants)
    city, combined = engine.nodes["city"], engine.nodes["combined"]
    active = TRUE
    ponded_days = []
    for number in range(spec.days):
        ctx = Ctx(
            network.pollutants,
            network.float_accuracy,
            torch.Size(()),
            day=number,
            node_names=list(engine.nodes),
            record_decisions=True,
        )
        engine.begin_day(ctx)
        for node in engine.nodes.values():
            node.bind(node.configured_parameters(network.dates[number]))
        combined.bind({"capacity": torch.tensor(10.0, dtype=torch.float64)})
        city.run(active=active)
        if not ponded_days and volume(combined.sewer_tank.storage) > 7.0:
            combined.bind({"capacity": torch.tensor(2.0, dtype=torch.float64)})
        before = volume(city.surfaces[0].storage)
        combined.make_discharge(active=active)
        ponds = [d for d in ctx.decisions if d.site == "node:combined/make_discharge/ponded"]
        if bool(ponds[0].outcome):
            gained = volume(city.surfaces[0].storage) - before
            assert gained > 0
            assert float(engine.arcs["combined_to_city"].flow_in) == pytest.approx(gained)
            ponded_days.append(number)
        engine.end_day()
    assert len(ponded_days) == 1


def test_an_override_in_one_wsimod_build_does_not_reach_the_next():
    """WSIMOD's override of ``pollutant_load`` updates the shared default ``{}`` in place
    (``land.py:397``, ``:415``; defect table), so in one process it reaches every later surface
    built with the default. ``reset_wsimod_globals`` restores the constructor defaults, so each
    reference build starts as a fresh process would."""
    pytest.importorskip("wsimod")
    from tests.apps.wsimod.test_config import _everything
    from tests.verification._wsimod_cases import CASES
    from tests.verification._wsimod_reference import wsimod_model

    model = wsimod_model(_everything())  # overrides rural's pollutant_load (phosphate 1e-7)
    assert model.nodes["land"].get_surface("rural").pollutant_load == {"phosphate": 1e-7}
    later = wsimod_model(CASES["pervious_only"].build())
    assert [s.pollutant_load for s in later.nodes["farm"].surfaces] == [{}, {}]


# --- surface decay and the nitrite carry-over (land.py:425-449) ---------------------------------
NP = Pollutants(
    ("ammonia", "nitrite", "nitrate", "phosphate", "temperature"),
    ("ammonia", "nitrite", "nitrate", "phosphate"),
    ("temperature",),
)
_DECAYS = {
    "ammonia": {"constant": 0.5, "exponent": 1.0},
    "nitrite": {"constant": 0.25, "exponent": 1.0},
}


def _decaying_land(pollutants=NP, decays=None):
    land = make_land(
        [
            surface(
                "ImperviousSurface",
                "roads",
                area=10.0,
                pore_depth=0.01,
                decays=decays or _DECAYS,
                initial_storage={"volume": 1.0, "nitrate": 1.0, "nitrite": 2.0},
            )
        ],
        pollutants=pollutants,
    )
    return land, land.surfaces[0], day(land, pollutants=pollutants)


def test_the_carry_over_converts_yesterdays_decay():
    """``land.py:427-439``: the decayed nitrite joins the nitrate, then the decayed ammonia joins
    the nitrite, and both are inflows of the land. On day 0 ``total_decayed`` is empty."""
    land, roads, ctx = _decaying_land()
    land.bind(forcing(0.0, 0.0))
    land.run(active=TRUE)
    assert float(roads.storage[NP.column("nitrate")]) == 1.0
    assert float(land.running_inflow_mb[NP.column("nitrite")]) == 0.0
    roads.storage = NP.to_vqip({"volume": 1.0, "nitrate": 1.0, "nitrite": 2.0})
    land.running_inflow_mb = NP.to_vqip({"nitrate": 0.5})
    roads.total_decayed = NP.to_vqip({"ammonia": 1.5, "nitrite": 0.4})
    land.run(active=TRUE)
    assert float(roads.storage[NP.column("nitrate")]) == 1.0 + 0.4
    assert float(roads.storage[NP.column("nitrite")]) == 2.0 + 1.5
    assert float(roads.storage[NP.column("ammonia")]) == 0.0
    assert float(land.running_inflow_mb[NP.column("nitrate")]) == 0.5 + 0.4
    assert float(land.running_inflow_mb[NP.column("nitrite")]) == 1.5
    # a masked instance keeps its state (D3)
    before = roads.storage.clone()
    land.run(active=torch.tensor(False))
    assert torch.equal(roads.storage, before)
    # a second run on the same day converts the same mass again (X16)
    land.run(active=TRUE)
    assert float(roads.storage[NP.column("nitrate")]) == (1.0 + 0.4) + 0.4


def test_the_surface_decays_at_the_end_of_the_day_and_the_mass_is_a_balance_term():
    land, roads, ctx = _decaying_land()
    roads.storage = NP.to_vqip({"volume": 1.0, "ammonia": 8.0, "nitrite": 4.0, "nitrate": 2.0})
    land.bind(forcing(0.0, 0.0, 20.0))
    land.end_timestep()
    assert float(roads.storage[NP.column("ammonia")]) == 4.0
    assert float(roads.storage[NP.column("nitrite")]) == 3.0
    assert float(roads.storage[NP.column("nitrate")]) == 2.0  # not listed: stays
    assert float(roads.total_decayed[NP.column("ammonia")]) == 4.0
    assert float(roads.total_decayed[NP.column("nitrite")]) == 1.0
    assert torch.equal(ctx.terms["DecayTank.decayed"], roads.total_decayed)


def test_without_nitrite_the_decayed_mass_is_lost():
    """Without ``nitrite`` the block is skipped (``:427``): no carry-over, the storage still
    decays, and the mass is the ``"DecayTank.decayed"`` out-term."""
    land, roads, ctx = _decaying_land(PP, {"phosphate": {"constant": 0.1, "exponent": 1.0}})
    assert not any(path.endswith(".total_decayed") for path in land.carried_paths)
    roads.storage = PP.to_vqip({"volume": 1.0, "phosphate": 1.0, "temperature": 5.0})
    roads.total_decayed = PP.to_vqip({"phosphate": 9.0})  # never converted
    land.bind(forcing(0.0, 0.0, 20.0))
    land.run(active=TRUE)
    assert float(roads.storage[PP.column("phosphate")]) == 1.0
    assert float(land.running_inflow_mb[PP.column("phosphate")]) == 0.0
    land.end_timestep()
    assert float(roads.storage[PP.column("phosphate")]) == 1.0 - 1.0 * 0.1
    assert float(ctx.terms["DecayTank.decayed"][PP.column("phosphate")]) == 1.0 * 0.1


def _lists_mapping(names, additive, non_additive):
    return _mapping(
        [{"type_": "ImperviousSurface", "surface": "roads"}],
        lists={
            "pollutants": names,
            "additive_pollutants": additive,
            "non_additive_pollutants": non_additive,
        },
    )


def test_nitrite_needs_nitrate_and_ammonia():
    """``land.py:427-439`` reads all three (``KeyError``): refused by name."""
    for names in (["nitrite", "ammonia"], ["nitrite", "nitrate"], ["nitrite"]):
        with pytest.raises(ValueError, match="nitrite"):
            _validate(_lists_mapping(names, names, []), pollutants=True)
    names = ["nitrite", "nitrate", "ammonia"]
    _validate(_lists_mapping(names, names, []), pollutants=True)
    # a quantity-only run carries no nitrite
    _validate(_lists_mapping(["nitrite"], ["nitrite"], []), pollutants=False)


def test_surface_decay_reads_the_lands_air_temperature():
    """``DecayObj`` reads the parent's ``("temperature", t)`` (``core.py:462-468``), not the
    soil temperature the pervious surface works out (``land.py:837-839``, ``:1060-1079``)."""
    spec = surface(
        "PerviousSurface",
        "lawn",
        area=100.0,
        decays={"phosphate": {"constant": 0.1, "exponent": 2.0}},
    )
    land = make_land([spec], pollutants=PP)
    lawn = land.surfaces[0]
    day(land, pollutants=PP)
    lawn.storage = PP.to_vqip({"volume": 7.0, "phosphate": 1.0, "temperature": 30.0})
    land.bind(forcing(0.0, 0.0, 22.0))
    assert float(lawn.decay.factor[PP.column("phosphate")]) == 0.1 * 2.0**2
    land.end_timestep()
    assert float(lawn.storage[PP.column("phosphate")]) == 1.0 - 1.0 * (0.1 * 2.0**2)


def test_a_surface_without_decays_refuses_a_changed_inert_driver():
    """D15, X9: the key set is structure; an entry at a pollutant the surface does not list must
    stay (0, 1)."""
    land = make_land([surface("ImperviousSurface", "roads", area=1.0)], pollutants=PP)
    day(land, pollutants=PP)
    inert = {
        "surfaces.roads.decays.constant": torch.zeros(2, dtype=torch.float64),
        "surfaces.roads.decays.exponent": torch.ones(2, dtype=torch.float64),
    }
    land.bind({**forcing(0.0, 0.0), **inert})
    changed = {**inert, "surfaces.roads.decays.constant": torch.tensor([0.1, 0.0])}
    with pytest.raises(ValueError, match="phosphate"):
        land.bind({**forcing(0.0, 0.0), **changed})
    # a listed pollutant takes a driver value; an unlisted one is still refused
    land, roads, _ = _decaying_land()
    constant = torch.tensor([0.5, 0.25, 0.3, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="nitrate"):
        land.bind({**forcing(0.0, 0.0), "surfaces.roads.decays.constant": constant})


def test_a_decays_override_reaches_only_its_surface():
    """D13: noodl merges ``decays`` per surface. WSIMOD's default ``{}`` is one dict shared by
    every surface built without a ``decays`` key (the defect row), so its override reaches all
    of them."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    override = {"phosphate": {"constant": 0.1, "exponent": 1.0}}
    mapping = _mapping(
        [
            {"type_": "ImperviousSurface", "surface": "roads"},
            {"type_": "ImperviousSurface", "surface": "roofs"},
        ],
        overrides={"roads": {"decays": override}},
        lists=_LISTS,
    )
    network = read_wsimod_config(mapping)
    roads, roofs = make_node(network.nodes[0], network, network.pollutants).surfaces
    assert roads.decay is not None and roofs.decay is None
    assert network.nodes[0].surfaces[1].parameters["decays"] == {}
    theirs = wsimod_model(mapping).nodes["land"].surfaces
    assert theirs[0].decays is theirs[1].decays
    assert theirs[1].decays == override


def test_the_surfaces_decay_drivers_are_dense_and_absent_in_a_quantity_only_run():
    spec = surface("ImperviousSurface", "roads", area=1.0, decays=_DECAYS)
    land = make_land([spec], pollutants=NP)
    roads = land.surfaces[0]
    assert roads.defaults["decays.constant"].tolist() == [0.5, 0.25, 0.0, 0.0, 0.0]
    assert roads.defaults["decays.exponent"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    quantity = make_land([spec])  # P: unlisted names are dropped
    assert quantity.surfaces[0].decay is None
    assert quantity.surfaces[0].defaults["decays.constant"].shape == (0,)
