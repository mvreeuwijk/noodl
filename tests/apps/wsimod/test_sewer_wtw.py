"""``Sewer``, ``WTW``, ``WWTW`` and ``FWTW`` (noodl only, except where a test says it drives live
WSIMOD).

The values asserted are WSIMOD's, worked out from ``sewer.py`` and ``wtw.py`` by hand; the replay
(``tests/verification/test_wsimod_replay.py``) checks the same code against live WSIMOD on the
``sewer_timearea``, ``foul_sewers``, ``groundwater_infiltrate``, ``wwtw_storm_tank``,
``fwtw_deficit_sludge`` and ``queue_tank_time_zero`` cases.
"""

from __future__ import annotations

import pytest
import torch

from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.node import NODE_CLASSES
from noodl.apps.wsimod.sewer import Sewer, sewer_derived
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from noodl.apps.wsimod.wtw import FWTW, WTW, WWTW, wtw_derived
from tests.apps.wsimod._replay import ScriptedArc
from tests.verification._wsimod_reference import ArcCall

P = QUANTITY_ONLY
PP = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)
C_V = 1 - 0.0002 - 0.03  # wtw.py:110 with WSIMOD's defaults


def neutral(liquor: float = 0.03, percent_solids: float = 0.0002, **values) -> dict:
    c_v = 1 - percent_solids - liquor
    return {
        "percent_solids": percent_solids,
        "process_parameters": {"phosphate": {"constant": c_v, "exponent": 1}},
        "liquor_multiplier": {"phosphate": liquor, "volume": liquor},
        **values,
    }


def make(cls: str, name: str = "n", pollutants: Pollutants = P, **parameters):
    return make_node(NodeSpec(name, cls, cls, parameters), None, pollutants)


def day(*objects, shape=(), pollutants: Pollutants = P, number=0):
    ctx = Ctx(pollutants, FA, torch.Size(shape), day=number, node_names=[], record_decisions=True)
    for obj in objects:
        obj.ctx = ctx
    return ctx


def call(name, method, request, reply, *, tag="default") -> ArcCall:
    """A scripted arc call of WSIMOD's; ``request`` and ``reply`` are volumes or VQIP dicts."""
    request = {"volume": request} if not isinstance(request, dict) else request
    reply = {"volume": reply} if not isinstance(reply, dict) else reply
    return ArcCall(
        name, method, request, tag, False, reply, 0.0, {"volume": 0.0}, 0.0, {"volume": 0.0}
    )


def scripted(name, calls, *, out_type="Node", pollutants: Pollutants = P):
    return ScriptedArc(
        ArcSpec(name, "Arc", "n", "other"),
        calls,
        pollutants,
        scale=1.0,
        in_type="Node",
        out_type=out_type,
    )


def test_the_classes_are_registered_under_the_names_wsimod_gives_them():
    for name, cls in (("Sewer", Sewer), ("WTW", WTW), ("WWTW", WWTW), ("FWTW", FWTW)):
        assert NODE_CLASSES[name] is cls and cls.wsimod_type == name
    assert WWTW.step_methods == ("calculate_discharge", "make_discharge")
    assert FWTW.step_methods == ("treat_water",) and Sewer.step_methods == ("make_discharge",)


# --- derived values -----------------------------------------------------------------------------
def test_wtw_and_sewer_derive_what_construction_derives():
    """D5: the volume constant and the sewer tank's fields from the merged parameters equal
    the live WSIMOD objects' after ``add_overrides``."""
    pytest.importorskip("wsimod")
    from tests.apps.wsimod.test_config import _everything
    from tests.verification._wsimod_reference import wsimod_model

    mapping = _everything()
    model = wsimod_model(mapping)
    network = read_wsimod_config(mapping)
    close = {"rel": 1e-15, "abs": 0}
    checked = set()
    for spec in network.nodes:
        theirs = model.nodes[spec.name]
        if spec.cls in ("WWTW", "FWTW"):
            constant = theirs.process_parameters["volume"]["constant"]
            assert wtw_derived(spec)["volume_constant"] == pytest.approx(constant, **close)
            ours = make_node(spec, network, QUANTITY_ONLY)
            ours.bind({})
            assert float(ours.volume_constant()) == constant
            checked.add(spec.cls)
        if spec.cls == "Sewer":
            derived = sewer_derived(spec)
            tank = theirs.sewer_tank
            for key, value in (
                ("capacity", tank.capacity),
                ("area", tank.area),
                ("datum", tank.datum),
            ):
                assert derived[key] == pytest.approx(value, **close), key
            ours = make_node(spec, network, QUANTITY_ONLY)
            assert float(ours.sewer_tank.capacity) == tank.capacity
            assert float(ours.sewer_tank.area) == tank.area
            assert float(ours.sewer_tank.datum) == tank.datum
            assert ours.pipe_time == theirs.pipe_time
            checked.add(spec.cls)
    assert checked == {"WWTW", "FWTW", "Sewer"}
    # the override changed the WWTW's percent_solids and volume multiplier
    wwtw = next(s for s in network.nodes if s.name == "wwtw")
    assert wtw_derived(wwtw)["volume_constant"] == 1 - 0.001 - 0.05


# --- treatment --------------------------------------------------------------------------------
def test_wtw_splits_by_its_volume_constant():
    """``wtw.py:163-214`` with the neutral process: 100 of input at phosphate 1."""
    works = make("WWTW", pollutants=PP, **neutral())
    day(works, pollutants=PP)
    works.current_input = PP.to_vqip({"volume": 100.0, "phosphate": 1.0, "temperature": 10.0})
    works.treat_current_input(active=TRUE)
    treated, liquor, solids = (PP.to_dict(x) for x in (works.treated, works.liquor, works.solids))
    assert treated["volume"] == 100 * C_V and treated["phosphate"] == 1.0 * C_V
    assert liquor["volume"] == 100 * 0.03 and liquor["phosphate"] == 1.0 * 0.03
    assert solids["volume"] == 100 * 0.0002
    assert solids["phosphate"] == 1.0 - 1.0 * C_V - 1.0 * 0.03
    assert solids["phosphate"] == pytest.approx(0.0002, rel=1e-12)
    # non-additive: the treated water keeps the influent's, the liquor blends with the old
    # (empty) liquor, (0 x 0 + 10 x 100 x 0.03) / (0 + 100 x 0.03); the solids' is never set
    assert treated["temperature"] == 10.0 and liquor["temperature"] == 10.0
    assert solids["temperature"] == 0.0


def test_the_liquor_blends_with_the_previous_liquor():
    """``wtw.py:174-182``: the old liquor's temperature weighted by its volume, the influent's by
    ``V_in x m_v``."""
    works = make("WWTW", pollutants=PP, **neutral())
    day(works, pollutants=PP)
    works.liquor = PP.to_vqip({"volume": 2.0, "phosphate": 0.1, "temperature": 20.0})
    works.current_input = PP.to_vqip({"volume": 100.0, "phosphate": 1.0, "temperature": 10.0})
    works.treat_current_input(active=TRUE)
    liquor = PP.to_dict(works.liquor)
    assert liquor["temperature"] == (20.0 * 2.0 + 10.0 * 100.0 * 0.03) / (2.0 + 100.0 * 0.03)
    assert liquor["volume"] == 100.0 * 0.03  # overwritten, not added (wtw.py:201)


def test_zero_input_on_day_one_keeps_the_previous_liquor_value():
    """Defect table: WSIMOD divides by ``V_liq + V_in x m_v = 0`` and raises
    ``ZeroDivisionError``; noodl keeps the previous liquor value and records the event, with a
    finite gradient through the untaken branch."""
    m_v = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    works = make("WWTW", pollutants=PP, **neutral())
    plant = make("FWTW", pollutants=PP, **neutral(service_reservoir_initial_storage=10.0))
    for node in (works, plant):
        ctx = day(node, pollutants=PP)
        node.bind({"liquor_multiplier": m_v})
        if node is works:
            node.calculate_discharge(active=TRUE)
        else:
            node.treat_water(active=TRUE)  # the reservoir is full: nothing to treat
        assert PP.to_dict(node.liquor.detach())["temperature"] == 0.0
        assert bool(ctx.events["wtw_liquor_zero_denominator"][node.name])
        (grad,) = torch.autograd.grad(node.liquor.sum() + node.treated.sum(), m_v)
        assert torch.isfinite(grad)
    # liquor_multiplier["volume"] = 0: the denominator is 0 on every day, whatever comes in
    works = make("WWTW", pollutants=PP, **neutral(liquor=0.0))
    for number in range(3):
        ctx = day(works, pollutants=PP, number=number)
        works.push_set(
            PP.to_vqip({"volume": 10.0, "phosphate": 0.1, "temperature": 12.0}), active=TRUE
        )
        works.calculate_discharge(active=TRUE)
        assert bool(ctx.events["wtw_liquor_zero_denominator"]["n"])
        assert PP.to_dict(works.liquor) == {"volume": 0.0, "phosphate": 0.0, "temperature": 0.0}
        works.end_timestep()


def test_a_zero_denominator_keeps_the_liquor_value_of_the_day_before():
    """Defect table, the kept branch: a ``WWTW`` treats 10 at temperature 10 on day 1. On day 2
    nothing comes in and ``liquor_multiplier["volume"]`` is 0: only the recycled liquor (0.3 at
    10) is treated, so the blend divides by 0.3 and the liquor volume becomes 0. On day 3 the
    denominator is ``0 + 0 x 0 = 0``: the liquor keeps day 2's temperature, 10, not 0."""
    works = make("WWTW", pollutants=PP, **neutral())
    m_v = torch.tensor([0.03, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    inflows = (10.0, 0.0, 0.0)
    fired = []
    for number, inflow in enumerate(inflows):
        ctx = day(works, pollutants=PP, number=number)
        works.bind({"liquor_multiplier": m_v[number]})
        if inflow:
            vqip = PP.to_vqip({"volume": inflow, "phosphate": 0.1, "temperature": 10.0})
            works.push_set(vqip, active=TRUE)
        works.calculate_discharge(active=TRUE)
        fired.append(bool(ctx.events["wtw_liquor_zero_denominator"]["n"]))
        liquor = PP.to_dict(works.liquor.detach())
        assert liquor["temperature"] == 10.0, number
        works.end_timestep()
    assert liquor["volume"] == 0.0
    assert fired == [False, False, True]
    (grad,) = torch.autograd.grad(works.liquor.sum() + works.treated.sum(), m_v)
    assert torch.isfinite(grad).all()


def test_without_non_additive_pollutants_there_is_no_liquor_blend_to_divide():
    """``wtw.py:174`` loops over the non-additive pollutants only: a quantity-only run (or one
    with additive pollutants only) never divides, so no decision or event is recorded."""
    only_additive = Pollutants(("phosphate",), ("phosphate",), ())
    for pollutants in (P, only_additive):
        works = make("WWTW", pollutants=pollutants)
        ctx = day(works, pollutants=pollutants)
        works.calculate_discharge(active=TRUE)  # zero input, no liquor: the denominator is 0
        assert "wtw_liquor_zero_denominator" not in ctx.events
        assert not any("liquor_zero_denominator" in d.site for d in ctx.decisions)


def test_the_zero_input_case_runs_and_records_the_event_where_wsimod_would_raise():
    from tests.apps.wsimod._stepping import step_days
    from tests.verification._wsimod_cases import CASES

    spec = CASES["wtw_zero_input"]
    assert not spec.wsimod and not spec.replay
    network = read_wsimod_config(spec.build())
    events = {"plant": [], "works": []}
    for _, ctx, engine in step_days(network, network.pollutants, 3):
        table = ctx.events["wtw_liquor_zero_denominator"]
        for name in events:
            events[name].append(bool(table.get(name, torch.tensor(False))))
        for name in events:
            assert torch.isfinite(engine.nodes[name].liquor).all()
    # plant: full reservoir on day 1 only; works: liquor_multiplier 0 every day
    assert events == {"plant": [True, False, False], "works": [True, True, True]}


def test_more_treated_than_input_is_an_event():
    """``wtw.py:213-214``: ``treated`` carried over from an earlier day exceeds the input."""
    works = make("WWTW")
    ctx = day(works)
    works.treated = P.to_vqip({"volume": 5.0})
    works.current_input = P.to_vqip({"volume": 1.0})
    works.treat_current_input(active=TRUE)
    assert bool(ctx.events["more_treated_than_input"]["n"])


# --- WWTW ---------------------------------------------------------------------------------------
def test_a_push_fills_the_throughput_then_the_stormwater_tank():
    """``wtw.py:374-407``: throughput 10, tank 5. 8 goes straight in (exact equality: nothing
    rejected); 12 more fills the throughput by 2 and the tank by 5 and 5 is rejected."""
    works = make("WWTW", treatment_throughput_capacity=10.0, stormwater_storage_capacity=5.0)
    ctx = day(works)
    assert float(works.push_check(active=TRUE)[0]) == 15.0
    assert float(works.push_set(P.to_vqip({"volume": 8.0}), active=TRUE)[0]) == 0.0
    assert float(works.push_check(P.to_vqip({"volume": 30.0}), active=TRUE)[0]) == 7.0
    reply = works.push_set(P.to_vqip({"volume": 12.0}), active=TRUE)
    assert float(reply[0]) == 5.0
    assert float(works.current_input[0]) == 10.0
    assert float(works.stormwater_tank.storage[0]) == 5.0
    assert float(ctx.dropped[0]) == 0.0


def test_a_tiny_stormwater_rejection_vanishes_and_is_dropped():
    """``wtw.py:403-404``: the tank's rejection below ``FLOAT_ACCURACY`` is not returned."""
    works = make("WWTW", treatment_throughput_capacity=1.0, stormwater_storage_capacity=2.0)
    ctx = day(works)
    reply = works.push_set(P.to_vqip({"volume": 3.0 + 4e-12}), active=TRUE)
    assert float(reply[0]) == 0.0
    assert float(ctx.dropped[0]) == pytest.approx(4e-12, rel=1e-3)


def test_calculate_discharge_drains_the_tank_into_the_spare_throughput_and_adds_liquor():
    """``wtw.py:323-341``: throughput 10, input 4, tank 9: 6 drained; then yesterday's liquor."""
    works = make("WWTW", treatment_throughput_capacity=10.0)
    day(works)
    works.current_input = P.to_vqip({"volume": 4.0})
    works.stormwater_tank.storage = P.to_vqip({"volume": 9.0})
    works.liquor = P.to_vqip({"volume": 0.5})
    works.calculate_discharge(active=TRUE)
    assert float(works.stormwater_tank.storage[0]) == 3.0
    assert float(works.treated[0]) == 10.5 * C_V
    assert float(works.liquor[0]) == 10.5 * 0.03


def test_a_tiny_discharge_rejection_vanishes_and_is_recorded_as_dropped():
    """W25 (``wtw.py:343-349``): a rejection at or below ``FLOAT_ACCURACY`` is not forced into
    the stormwater tank and vanishes as dropped water; a larger one is forced in."""
    for rejected, stored, couldnt in ((5e-12, 0.0, False), (1.0, 1.0, True)):
        works = make("WWTW")
        out = scripted("out", [call("out", "send_push_request", 10.0, rejected)])
        works.connect(out, "out")
        ctx = day(works)
        works.treated = P.to_vqip({"volume": 10.0})
        works.make_discharge(active=TRUE)
        out.assert_exhausted()
        assert float(works.treated[0]) == 0.0
        assert float(works.stormwater_tank.storage[0]) == stored
        assert float(ctx.dropped[0]) == (rejected if not couldnt else 0.0)
        assert bool(ctx.events.get("wwtw_couldnt_push", {}).get("n", False)) is couldnt


def test_treated_accumulates_without_make_discharge():
    """``wtw.py:447-453``: ``end_timestep_`` does not empty ``treated``, so an orchestration of
    ``calculate_discharge`` alone sums the days' treated water (live WSIMOD on the same calls)."""
    pytest.importorskip("wsimod")
    from wsimod.nodes.wtw import WWTW as TheirWWTW

    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()
    from wsimod.core import constants

    constants.POLLUTANTS, constants.ADDITIVE_POLLUTANTS = [], []
    constants.NON_ADDITIVE_POLLUTANTS = []
    theirs = TheirWWTW(name="w", liquor_multiplier={"volume": 0.03})
    ours = make("WWTW")
    try:
        for number, inflow in enumerate((10.0, 8.0)):
            day(ours, number=number)
            ours.push_set(P.to_vqip({"volume": inflow}), active=TRUE)
            ours.calculate_discharge(active=TRUE)
            theirs.push_set({"volume": inflow})
            theirs.calculate_discharge()
            ours.end_timestep()
            theirs.end_timestep()
            assert float(ours.treated[0]) == theirs.treated["volume"]
    finally:
        reset_wsimod_globals()
    day_one = 10.0 * C_V
    day_two = (8.0 + 10.0 * 0.03) * C_V  # day 2 retreats day 1's liquor
    assert float(ours.treated[0]) == day_one + day_two


def test_reuse_pulls_come_from_the_treated_water():
    """``wtw.py:409-445``."""
    works = make("WWTW", pollutants=PP)
    day(works, pollutants=PP)
    works.treated = PP.to_vqip({"volume": 10.0, "phosphate": 0.5, "temperature": 9.0})
    assert PP.to_dict(works.pull_check(active=TRUE))["volume"] == 10.0
    reply = PP.to_dict(works.pull_set(PP.to_vqip({"volume": 4.0}), active=TRUE))
    assert reply == {"volume": 4.0, "phosphate": 0.5 * (4.0 / 10.0), "temperature": 9.0}
    assert float(works.treated[0]) == 6.0
    reply = works.pull_set(PP.to_vqip({"volume": 7.0}), active=TRUE)
    assert float(reply[0]) == 6.0 and float(works.treated[0]) == 0.0


def test_a_masked_instance_is_left_alone_and_the_rest_match_single_runs():
    """D3: a batch of throughputs with one instance inactive."""
    caps = torch.tensor([5.0, 8.0, 12.0], dtype=torch.float64)
    active = torch.tensor([True, False, True])

    def run(cap, mask, shape):
        works = make("WWTW", stormwater_storage_capacity=3.0)
        day(works, shape=shape)
        works.bind({"treatment_throughput_capacity": cap})
        reply = works.push_set(P.to_vqip({"volume": 10.0}), active=mask)
        works.calculate_discharge(active=mask)
        return reply[..., 0], works.treated[..., 0], works.stormwater_tank.storage[..., 0]

    batch = run(caps, active, (3,))
    assert [float(x[1]) for x in batch] == [0.0, 0.0, 0.0]
    for i in (0, 2):
        single = run(caps[i], TRUE, ())
        assert [float(x[i]) for x in batch] == [float(x) for x in single]


def test_the_treated_volume_has_the_gradient_of_its_volume_constant():
    """d treated / d percent_solids = -V_in (central differences agree)."""
    works = make("WWTW")
    ps = torch.tensor(0.0002, dtype=torch.float64, requires_grad=True)

    def treated(value):
        day(works)
        works.bind({"percent_solids": value})
        works.current_input = P.to_vqip({"volume": 50.0})
        works.treated = P.to_vqip({"volume": 0.0})
        works.liquor = P.to_vqip({"volume": 0.0})
        works.treat_current_input(active=TRUE)
        return works.treated[0]

    (grad,) = torch.autograd.grad(treated(ps), ps)
    h = 1e-6 * 0.0002
    with torch.no_grad():
        numeric = (treated(ps + h) - treated(ps - h)) / (2 * h)
    assert float(grad) == pytest.approx(-50.0, rel=1e-12)
    assert float(numeric) == pytest.approx(float(grad), rel=1e-6)


def test_the_wwtw_records_its_solids_as_an_outflow():
    works = make("WWTW")
    ctx = day(works)
    works.current_input = P.to_vqip({"volume": 100.0})
    works.treat_current_input(active=TRUE)
    works.record_terms()
    assert set(ctx.terms) == set(WWTW.mass_balance) == {"WWTW.solids"}
    assert float(ctx.terms["WWTW.solids"][0]) == 100.0 * 0.0002
    works.end_timestep()
    assert float(works.solids[0]) == 0.0 and float(works.current_input[0]) == 0.0
    assert float(works.liquor[0]) == 3.0  # carried


# --- FWTW ---------------------------------------------------------------------------------------
def test_the_fwtw_deficit_takes_yesterdays_quality():
    """``wtw.py:588-596``, ``:650``: day 1 pulls water at temperature 8 from the reservoir;
    day 2 pulls nothing from upstream, so the shortfall is created at temperature 8 and at
    day 1's pulled phosphate concentration."""
    initial = {"volume": 20.0, "phosphate": 0.4, "temperature": 8.0}
    plant = make(
        "FWTW",
        pollutants=PP,
        **neutral(
            treatment_throughput_capacity=10.0,
            service_reservoir_storage_capacity=30.0,
            service_reservoir_initial_storage=initial,
        ),
    )
    day(plant, pollutants=PP)
    pulled = PP.to_dict(plant.pull_set(PP.to_vqip({"volume": 5.0}), active=TRUE))
    assert pulled["temperature"] == 8.0
    plant.end_timestep()
    assert PP.to_dict(plant.previous_pulled) == pulled
    ctx = day(plant, pollutants=PP, number=1)
    plant.treat_water(active=TRUE)  # no in-arcs: nothing comes from upstream
    deficit = PP.to_dict(plant.total_deficit)
    assert deficit["volume"] == 10.0  # min(30 - 15, 10)
    assert deficit["temperature"] == 8.0
    assert deficit["phosphate"] == pulled["phosphate"] * (10.0 / pulled["volume"])
    assert PP.to_dict(plant.current_input) == deficit
    assert bool(ctx.events["service_reservoirs_not_filled"]["n"])
    plant.record_terms()
    assert set(ctx.terms) == set(FWTW.mass_balance)
    assert PP.to_dict(ctx.terms["FWTW.total_deficit"]) == deficit


def test_the_fwtw_sends_liquor_and_solids_to_sewers_and_keeps_what_they_reject():
    """``wtw.py:606-617``: the target is ``min(12 - 5, 10) = 7``, all pulled; liquor 0.21 +
    solids 0.0014 go to the sewer, which takes 0.1; the treated water goes into the
    reservoir."""
    plant = make(
        "FWTW",
        treatment_throughput_capacity=10.0,
        service_reservoir_storage_capacity=12.0,
        service_reservoir_initial_storage=5.0,
    )
    upstream = scripted("up", [call("up", "send_pull_request", 7.0, 7.0)])
    sludge = 7.0 * 0.03 + 7.0 * 0.0002
    sewer = scripted(
        "sw", [call("sw", "send_push_request", sludge, sludge - 0.1)], out_type="Sewer"
    )
    plant.connect(upstream, "in")
    plant.connect(sewer, "out")
    ctx = day(plant)
    plant.treat_water(active=TRUE)
    upstream.assert_exhausted()
    sewer.assert_exhausted()
    assert float(plant.total_deficit[0]) == 0.0
    assert float(plant.unpushed_sludge[0]) == sludge - 0.1
    assert bool(ctx.events["sludge_nowhere"]["n"])
    assert float(plant.service_reservoir_tank.storage[0]) == 5.0 + 7.0 * (1 - 0.0002 - 0.03)
    assert "excess_treated_water" not in ctx.events or not bool(
        ctx.events["excess_treated_water"]["n"]
    )


# --- Sewer --------------------------------------------------------------------------------------
def test_sewer_tags_enter_at_the_pipe_time_or_over_the_time_area_diagram():
    """``sewer.py:96-104``, ``:163-199``: default and Sewer at ``pipe_time``; Land and Demand
    over ``pipe_timearea``; the share with time 0 is active at once."""
    sewer = make("Sewer", capacity=100.0, pipe_time=1, pipe_timearea={0: 0.5, 1: 0.3, 2: 0.2})
    day(sewer)
    tank = sewer.sewer_tank
    assert tank.internal_arc.length == 2
    sewer.push_set(P.to_vqip({"volume": 10.0}), "default", active=TRUE)
    assert float(tank.active_storage[0]) == 0.0 and float(tank.storage[0]) == 10.0
    sewer.push_set(P.to_vqip({"volume": 10.0}), "Demand", active=TRUE)
    assert float(tank.active_storage[0]) == 5.0
    register = tank.internal_arc.register[..., 0]
    assert [float(x) for x in register] == [0.0, 10.0 + 3.0, 2.0]
    assert float(sewer.push_check(P.to_vqip({"volume": 500.0}), "Land", active=TRUE)[0]) == 80.0


def test_a_sewer_with_pipe_time_zero_discharges_the_same_day():
    """Spec, Flushing a ``QueueTank`` on push: time 0 is active at once and leaves with
    ``make_discharge``."""
    sewer = make("Sewer", capacity=100.0)
    out = scripted("out", [call("out", "send_push_request", 7.0, 0.0)])
    sewer.connect(out, "out")
    day(sewer)
    sewer.push_set(P.to_vqip({"volume": 7.0}), active=TRUE)
    assert float(sewer.sewer_tank.active_storage[0]) == 7.0
    sewer.make_discharge(active=TRUE)
    out.assert_exhausted()
    assert float(sewer.sewer_tank.storage[0]) == 0.0


def test_ponded_water_goes_to_land_and_the_rest_is_forced_back():
    """``sewer.py:233-239``: a tank 2 above capacity, its downstream full: the ponded 2 go to
    the ``Land`` arc with the tag ``"Sewer"``, which takes 1.5; 0.5 is forced back."""
    sewer = make("Sewer", capacity=10.0)
    land = scripted(
        "land",
        [
            call("land", "send_push_request", 12.0, 12.0),
            call("land", "send_push_request", 2.0, 0.5, tag="Sewer"),
        ],
        out_type="Land",
    )
    sewer.connect(land, "out")
    ctx = day(sewer)
    sewer.sewer_tank.storage = P.to_vqip({"volume": 12.0})
    sewer.sewer_tank.active_storage = P.to_vqip({"volume": 12.0})
    sewer.make_discharge(active=TRUE)
    land.assert_exhausted()
    assert float(sewer.sewer_tank.storage[0]) == 10.5
    assert float(sewer.sewer_tank.active_storage[0]) == 10.5
    ponded = [d for d in ctx.decisions if d.site == "node:n/make_discharge/ponded"]
    assert len(ponded) == 1 and bool(ponded[0].outcome)
    assert not bool(ctx.events.get("ponded_water_cant_reenter", {}).get("n", False))


def test_ponded_water_at_or_below_float_accuracy_vanishes_and_is_dropped():
    """``sewer.py:234-235``: ``pull_ponded`` takes the excess out of the tank; at or below
    ``FLOAT_ACCURACY`` nothing pushes it anywhere, so it is recorded as dropped water."""
    sewer = make("Sewer", capacity=10.0)
    ctx = day(sewer)
    over = 10.0 + 1e-12
    sewer.sewer_tank.storage = P.to_vqip({"volume": over})
    sewer.sewer_tank.active_storage = P.to_vqip({"volume": over})
    sewer.make_discharge(active=TRUE)  # no out-arcs: nothing is pushed downstream
    excess = over - 10.0
    assert 0 < excess < FA
    assert float(ctx.dropped[0]) == excess
    assert float(sewer.sewer_tank.storage[0]) == over - excess == 10.0


def test_sewer_travel_times_are_whole_days_and_the_diagram_sums_to_one_or_warns():
    network = read_wsimod_config(
        {"nodes": [{"name": "s", "type_": "Sewer", "pipe_timearea": {0: 0.5, 1: 0.4}}]}
    )
    with pytest.warns(UserWarning, match="pipe_timearea"):
        Sewer.validate(network.nodes[0], network, pollutants=False)
    for bad in ({"pipe_time": 1.5}, {"pipe_timearea": {-1: 1.0}}, {"pipe_timearea": {}}):
        network = read_wsimod_config({"nodes": [{"name": "s", "type_": "Sewer", **bad}]})
        with pytest.raises(ValueError, match="node 's'.*pipe_time"):
            Sewer.validate(network.nodes[0], network, pollutants=False)
    with pytest.raises(ValueError, match="pipe_time"):
        make("Sewer", pipe_time=-1)


# --- refusals -----------------------------------------------------------------------------------
_LISTS = {
    "pollutants": ["phosphate", "temperature"],
    "additive_pollutants": ["phosphate"],
    "non_additive_pollutants": ["temperature"],
}


def _works(**node) -> tuple[NodeSpec, object]:
    network = read_wsimod_config({**_LISTS, "nodes": [{"name": "w", "type_": "WWTW", **node}]})
    return network.nodes[0], network


def test_non_neutral_wtw_processes_are_refused():
    """Spec, Pollutants in 3a: WSIMOD's defaults (constant 0.01, exponent 1.001, multiplier 0.7)
    and any other non-neutral form are 3b; a quantity-only run ignores them."""
    spec, network = _works()  # the reader fills WSIMOD's defaults (wtw.py:73-85)
    with pytest.raises(ValueError, match="process_parameters.*3b"):
        WWTW.validate(spec, network, pollutants=True)
    assert WWTW.validate(spec, network, pollutants=False) == [
        "process_parameters",
        "liquor_multiplier",
    ]
    for node in (
        neutral() | {"liquor_multiplier": {"phosphate": 0.7, "volume": 0.03}},
        neutral() | {"process_parameters": {"phosphate": {"constant": C_V, "exponent": 1.001}}},
        neutral() | {"process_parameters": {"phosphate": {"constant": 0.9, "exponent": 1}}},
    ):
        spec, network = _works(**node)
        with pytest.raises(ValueError, match="process_parameters.*3b"):
            WWTW.validate(spec, network, pollutants=True)
    spec, network = _works(**neutral())
    assert WWTW.validate(spec, network, pollutants=True) == []
    # a configured volume constant that is not the derived one is ambiguous (wtw.py:92, :146)
    stale = neutral()
    stale["process_parameters"]["volume"] = {"constant": 0.5}
    spec, network = _works(**stale)
    with pytest.raises(ValueError, match="volume"):
        WWTW.validate(spec, network, pollutants=False)
    spec, network = _works(liquor_multiplier={"phosphate": 0.03})
    with pytest.raises(ValueError, match="liquor_multiplier has no 'volume'"):
        WWTW.validate(spec, network, pollutants=False)


def test_additive_pollutants_need_temperature():
    """``wtw.py:191-193`` reads the influent's temperature for every additive pollutant."""
    network = read_wsimod_config(
        {
            "pollutants": ["phosphate"],
            "additive_pollutants": ["phosphate"],
            "non_additive_pollutants": [],
            "nodes": [{"name": "f", "type_": "FWTW", **neutral()}],
        }
    )
    with pytest.raises(ValueError, match="temperature"):
        FWTW.validate(network.nodes[0], network, pollutants=True)
    # a quantity-only run ignores the pollutant entries
    assert FWTW.validate(network.nodes[0], network, pollutants=False) == [
        "process_parameters",
        "liquor_multiplier",
    ]
