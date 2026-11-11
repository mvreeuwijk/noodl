"""``Storage``, ``Groundwater``, ``QueueGroundwater``, ``Reservoir`` and ``RiverReservoir`` (noodl
only, except where a test says it drives live WSIMOD).

The values asserted are WSIMOD's, worked out from ``storage.py`` by hand; the replay
(``tests/verification/test_wsimod_replay.py``) checks the same code against live WSIMOD.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.node import NODE_CLASSES, Waste
from noodl.apps.wsimod.storage import (
    Groundwater,
    QueueGroundwater,
    Reservoir,
    River,
    RiverReservoir,
    Storage,
    exp_python,
    pow_half,
    river_derived,
    riverrc,
)
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from tests.apps.wsimod._replay import ScriptedArc, StoreNode
from tests.verification._wsimod_reference import ArcCall

P = QUANTITY_ONLY
FA = 1e-11
TRUE = torch.tensor(True)


def make(cls: str, name: str = "n", **parameters):
    return make_node(NodeSpec(name, cls, cls, parameters), None, P)


_NEW_ARCS: list = []  # the arcs built since the last day(); day() gives them the context


def day(*objects, shape=(), record=False):
    ctx = Ctx(P, FA, torch.Size(shape), day=0, node_names=[], record_decisions=record)
    for obj in (*objects, *_NEW_ARCS):
        obj.ctx = ctx
    _NEW_ARCS.clear()
    return ctx


def vol(v):
    return P.to_vqip({"volume": v})


def arc(name, a, b, *, capacity=1e15, cls="Arc"):
    spec = ArcSpec(name, cls, a.name, b.name, capacity=capacity)
    _NEW_ARCS.append(ARC_CLASSES[cls](spec, a, b, P))
    return _NEW_ARCS[-1]


def call(arc_name, method, request, reply, *, before=0.0, after=None) -> ArcCall:
    """A scripted arc call of WSIMOD's: volumes only; ``after`` is the arc's flow afterwards."""
    after = before + (reply if method == "send_pull_request" else request - reply)
    return ArcCall(
        arc_name,
        method,
        {"volume": request},
        "default",
        False,
        {"volume": reply},
        before,
        {"volume": before},
        after,
        {"volume": after},
    )


def scripted(name, calls, *, in_type="Node", out_type="Node"):
    return ScriptedArc(
        ArcSpec(name, "Arc", "n", "other"), calls, P, scale=1.0, in_type=in_type, out_type=out_type
    )


def test_the_classes_are_registered_under_the_names_wsimod_gives_them():
    for name, cls in (
        ("Storage", Storage),
        ("Groundwater", Groundwater),
        ("QueueGroundwater", QueueGroundwater),
        ("Reservoir", Reservoir),
        ("RiverReservoir", RiverReservoir),
        ("River", River),
    ):
        assert NODE_CLASSES[name] is cls
    # storage.py:294, :1043: the renames other nodes' of_type filters see
    assert QueueGroundwater.wsimod_type == "Groundwater"
    assert RiverReservoir.wsimod_type == "Reservoir"
    assert River.wsimod_type == "River"


# --- Groundwater.infiltrate ---------------------------------------------------------------
def test_infiltration_is_the_root_of_the_excess_share():
    # storage.py:231-245: capacity 100 x threshold 0.5 = 50; (75 - 50) x 0.04 = 1; root 1
    parameters = {
        "capacity": 100.0,
        "infiltration_threshold": 0.5,
        "infiltration_pct": 0.04,
    }
    for storage, pushed in ((75.0, 1.0), (50.0, 0.0)):
        gw = make("Groundwater", initial_storage=storage, **parameters)
        sewer = scripted(
            "to_sewer", [call("to_sewer", "send_push_request", pushed, 0.0)], out_type="Sewer"
        )
        gw.connect(sewer, "out")
        day(gw)
        gw.infiltrate(active=TRUE)
        sewer.assert_exhausted()  # the push, of exactly the root, was made
        assert float(gw.tank.storage[0]) == storage - pushed


def test_the_infiltration_gradient_is_zero_at_the_threshold():
    parameters = {"capacity": 100.0, "infiltration_threshold": 0.5, "infiltration_pct": 0.04}
    for storage, pushed, slope in ((75.0, 1.0, 0.5 / 1.0 * 0.04), (50.0, 0.0, 0.0)):
        gw = make("Groundwater", **parameters)
        sewer = StoreNode("sewer", P, wsimod_type="Sewer")
        arc("to_sewer", gw, sewer)
        day(gw, sewer)
        level = torch.tensor([storage], dtype=torch.float64, requires_grad=True)
        gw.load_stores({"tank.storage": level})
        gw.infiltrate(active=TRUE)
        assert float(sewer.storage[0].detach()) == pushed
        sewer.storage[0].backward()
        assert level.grad.tolist() == [pytest.approx(slope, rel=1e-14, abs=0.0)]
        assert (level.grad.tolist() == [0.0]) == (storage == 50.0)


def test_pow_half_is_pythons_pow_bitwise_with_a_zero_slope_at_zero():
    rng = random.Random(4)
    values = [0.0, 1.0, 2.0, 1e-300] + [
        rng.uniform(0, 1e4) ** rng.choice([1, 2, 0.5]) for _ in range(20000)
    ]
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    y = pow_half(x)
    assert y.tolist() == [v**0.5 for v in values]
    y.sum().backward()
    assert x.grad[0] == 0.0 and torch.isfinite(x.grad).all()
    assert x.grad[2].item() == pytest.approx(0.5 / 2.0**0.5, rel=1e-15)
    # torch's own root differs from Python's in the last bit for some arguments, which is why
    # pow_half exists
    assert torch.sqrt(x.detach()).tolist() != y.tolist()


# --- decays ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls", ["Storage", "Groundwater", "Reservoir", "QueueGroundwater", "RiverReservoir"]
)
def test_a_decays_dict_on_a_storage_node_is_refused(cls):
    network = read_wsimod_config(
        {
            "pollutants": [],
            "additive_pollutants": [],
            "non_additive_pollutants": [],
            "dates": ["2000-01-01"],
            "nodes": [{"name": "n", "type_": cls, "decays": None}],
            "arcs": [],
        }
    )
    spec = network.nodes[0]
    assert NODE_CLASSES[cls].validate(spec, network, pollutants=True) == []  # None is valid
    for decays in ({}, {"phosphate": {"constant": 0.1, "exponent": 1.1}}):
        spec.parameters["decays"] = decays
        with pytest.raises(ValueError, match=r"decays.*3b"):
            NODE_CLASSES[cls].validate(spec, network, pollutants=True)


def test_the_reader_refuses_decays_on_a_storage_node_too():
    with pytest.raises(ValueError, match=r"decays.*3b"):
        read_wsimod_config(
            {
                "pollutants": [],
                "additive_pollutants": [],
                "non_additive_pollutants": [],
                "dates": ["2000-01-01"],
                "nodes": [{"name": "n", "type_": "Storage", "decays": {}}],
                "arcs": [],
            }
        )


def _validate(cls, **parameters):
    network = read_wsimod_config(
        {
            "pollutants": [],
            "additive_pollutants": [],
            "non_additive_pollutants": [],
            "dates": ["2000-01-01"],
            "nodes": [{"name": "n", "type_": cls}],
            "arcs": [],
        }
    )
    spec = network.nodes[0]
    spec.parameters.update(parameters)
    return NODE_CLASSES[cls].validate(spec, network, pollutants=True)


@pytest.mark.parametrize(
    ("cls", "parameters", "match"),
    [
        ("Groundwater", {"residence_time": 0}, "residence_time is 0"),
        ("Groundwater", {"infiltration_pct": -0.1}, "infiltration_pct is negative"),
        ("QueueGroundwater", {"timearea": {-1: 1.0}}, r"timearea key -1.*whole number"),
        ("QueueGroundwater", {"timearea": {1.5: 1.0}}, r"timearea key 1\.5"),
        ("QueueGroundwater", {"timearea": {"a": 1.0}}, "timearea key 'a'"),
        ("QueueGroundwater", {"timearea": {}}, "non-empty mapping"),
    ],
)
def test_parameters_that_wsimod_cannot_run_are_refused_by_name(cls, parameters, match):
    with pytest.raises(ValueError, match=match):
        _validate(cls, **parameters)


def test_valid_parameters_are_accepted():
    assert _validate("Groundwater", residence_time=3, infiltration_pct=0.1) == []
    assert _validate("QueueGroundwater", timearea={0: 0.5, 3: 0.5}) == []
    assert _validate("Storage", initial_storage={"volume": 3.0}) == []


def test_a_quantity_only_run_ignores_the_pollutants_of_a_vqip_initial_storage():
    network = read_wsimod_config(
        {
            "pollutants": [],
            "additive_pollutants": [],
            "non_additive_pollutants": [],
            "dates": ["2000-01-01"],
            "nodes": [
                {
                    "name": "n",
                    "type_": "Storage",
                    "initial_storage": {"volume": 2.0, "phosphate": 1.0},
                }
            ],
            "arcs": [],
        }
    )
    spec = network.nodes[0]
    assert Storage.validate(spec, network, pollutants=False) == ["initial_storage"]
    assert (
        float(make("Storage", initial_storage=spec.parameters["initial_storage"]).tank.storage[0])
        == 2.0
    )


# --- Reservoir ------------------------------------------------------------------------------
def test_a_spill_is_forced_back_into_the_reservoir():
    """D14: the pull returns 5 to a request of 2 (a ``Distribution`` with leakage does that):
    2 fit, the other 3 spill, and are forced in as well (``storage.py:983-989``)."""
    res = make("Reservoir", capacity=10.0, initial_storage=8.0)
    supply = scripted("from_supply", [call("from_supply", "send_pull_request", 2.0, 5.0)])
    down = scripted("to_down", [])  # a call would raise: nothing is pushed downstream
    res.connect(supply, "in")
    res.connect(down, "out")
    ctx = day(res)
    res.make_abstractions(active=TRUE)
    supply.assert_exhausted()
    assert float(res.tank.storage[0]) == 13.0
    assert bool(ctx.events["reservoir_spill"]["n"])


def test_a_reservoir_above_capacity_is_never_clamped():
    """Review Focus 4: a tank started above its capacity has no excess, so WSIMOD still issues
    one pull request, of volume 0 (``storage.py:985``), and the storage stays."""
    res = make("Reservoir", capacity=10.0, initial_storage=12.0)
    supply = scripted("from_supply", [call("from_supply", "send_pull_request", 0.0, 0.0)])
    res.connect(supply, "in")
    ctx = day(res)
    res.make_abstractions(active=TRUE)
    supply.assert_exhausted()  # the one request asked for volume 0
    assert float(res.tank.storage[0]) == 12.0
    assert "reservoir_spill" not in ctx.events or not bool(ctx.events["reservoir_spill"]["n"])
    # and a real arc moves nothing
    res = make("Reservoir", capacity=10.0, initial_storage=12.0)
    source = StoreNode("source", P, storage=100.0)
    pipe = arc("pipe", source, res)
    day(res, source, pipe)
    res.make_abstractions(active=TRUE)
    assert float(res.tank.storage[0]) == 12.0 and float(source.storage[0]) == 100.0
    assert float(pipe.flow_in) == 0.0


def test_a_reservoir_fills_to_capacity_and_no_further():
    res = make("Reservoir", capacity=10.0, initial_storage=4.0)
    source = StoreNode("source", P, storage=100.0)
    pipe = arc("pipe", source, res, capacity=5.0)
    day(res, source, pipe)
    res.make_abstractions(active=TRUE)
    assert float(res.tank.storage[0]) == 9.0  # the pipe's capacity
    pipe.end_timestep()
    res.make_abstractions(active=TRUE)
    assert float(res.tank.storage[0]) == 10.0  # the excess
    assert float(source.storage[0]) == 100.0 - 6.0


def test_storage_distribute_pushes_what_it_can_and_forces_the_rest_back():
    store = make("Storage", capacity=10.0, initial_storage=5.0)
    sea = make_node(NodeSpec("sea", "Waste", "Waste", {}), None, P)
    arc("out", store, sea, capacity=1.0)
    ctx = day(store, sea)
    store.distribute(active=TRUE)
    # storage.py:120-127: all 5 pulled, 1 pushed, 4 forced back; unable to push
    assert float(store.tank.storage[0]) == 4.0
    assert bool(ctx.events["storage_unable_to_push"]["n"])


def test_groundwater_distribute_sends_the_residence_time_share():
    gw = make("Groundwater", capacity=20.0, initial_storage=8.0, residence_time=4.0)
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    arc("out", gw, sea, capacity=1.0)
    ctx = day(gw, sea)
    gw.distribute(active=TRUE)
    # storage.py:222-229: 8 / 4 = 2 pulled, 1 pushed (capacity), 1 forced back
    assert float(gw.tank.storage[0]) == 7.0
    assert bool(ctx.events["storage_unable_to_push"]["n"])
    # an out-arc to a Storage is outside of_type [Node, River, Waste]: all of it comes back
    gw = make("Groundwater", capacity=20.0, initial_storage=8.0, residence_time=4.0)
    other = make("Storage", name="other", capacity=5.0)
    pipe = arc("out", gw, other)
    day(gw, other, pipe)
    gw.distribute(active=TRUE)
    assert float(gw.tank.storage[0]) == 8.0 and float(pipe.flow_in) == 0.0


def test_an_inactive_call_changes_nothing_and_a_batch_equals_its_instances():
    def run(capacity, active, shape):
        res = make("Reservoir", capacity=10.0, initial_storage=6.0)
        source = StoreNode("source", P, storage=100.0)
        pipe = arc("pipe", source, res, capacity=7.0)
        day(res, source, pipe, shape=shape)
        res.bind({"capacity": capacity})
        res.make_abstractions(active=active)
        return res.tank.storage[..., 0], source.storage[..., 0]

    capacities = torch.tensor([8.0, 10.0, 12.0], dtype=torch.float64)
    active = torch.tensor([True, False, True])
    tank, source = run(capacities, active, (3,))
    assert tank.tolist() == [8.0, 6.0, 12.0]
    assert source.tolist() == [98.0, 100.0, 94.0]
    for i in (0, 2):
        single_tank, single_source = run(capacities[i], TRUE, ())
        assert float(single_tank) == float(tank[i]) and float(single_source) == float(source[i])


def test_handlers_and_recursion_follow_the_class():
    spec = NodeSpec("n", "Storage", "Storage", {})
    for cls in (Storage, Groundwater, QueueGroundwater, Reservoir):
        assert not cls.check_recurses(spec, "push", "default")
        assert not cls.check_recurses(spec, "pull", "default")
    assert RiverReservoir.check_recurses(spec, "push", "default")
    assert not RiverReservoir.check_recurses(spec, "pull", "default")
    assert not RiverReservoir.check_recurses(spec, "push", "Sewer")
    assert Groundwater.step_methods == ("infiltrate", "distribute")
    assert RiverReservoir.step_methods[-1] == "satisfy_environmental"


# --- QueueGroundwater -----------------------------------------------------------------------
def queue_groundwater(**parameters):
    return make("QueueGroundwater", capacity=100.0, **parameters)


def test_pull_set_active_takes_pro_rata_from_every_slot():
    """``storage.py:386-433``: 2 active and 6 in the slot of time 2, a pull of 4 of the 8 in
    store takes 4 x 2/8 = 1 from the active store and 4 x 6/8 = 3 from the slot."""
    gw = queue_groundwater(initial_storage=2.0, timearea={2: 1.0})
    day(gw)
    assert float(gw.push_set(vol(6.0), active=TRUE)[0]) == 0.0
    assert float(gw.tank.storage[0]) == 8.0 and float(gw.tank.active_storage[0]) == 2.0
    # pull_check_active reports the active store only (:370-384): inconsistent with the set
    assert float(gw.pull_check(vol(4.0), active=TRUE)[0]) == 2.0
    pulled = gw.pull_set(vol(4.0), active=TRUE)
    assert float(pulled[0]) == 4.0
    assert float(gw.tank.active_storage[0]) == 1.0
    assert float(gw.tank.internal_arc.register[2, 0]) == 3.0
    assert float(gw.tank.storage[0]) == 4.0


def test_a_pull_below_float_accuracy_is_empty():
    gw = queue_groundwater(initial_storage=2.0, timearea={2: 1.0})
    day(gw)
    assert float(gw.pull_set(vol(FA / 2), active=TRUE)[0]) == 0.0
    assert float(gw.tank.storage[0]) == 2.0
    empty = queue_groundwater()
    day(empty)
    assert float(empty.pull_set(vol(3.0), active=TRUE)[0]) == 0.0


def test_the_time_area_splits_a_push_over_the_queue():
    gw = queue_groundwater(timearea={0: 0.25, 3: 0.75})
    day(gw)
    gw.push_set(vol(8.0), active=TRUE)
    # time 0 is flushed into the active store at once (spec, Flushing a QueueTank on push)
    assert float(gw.tank.active_storage[0]) == 2.0
    assert float(gw.tank.internal_arc.register[3, 0]) == 6.0
    assert float(gw.tank.storage[0]) == 8.0
    gw.end_timestep()
    gw.end_timestep()
    gw.end_timestep()
    assert float(gw.tank.active_storage[0]) == 8.0  # slot 3 reached 0 on the third day


def test_the_push_check_is_the_frozen_first_tank():
    """``Storage.__init__`` binds the handler to the plain tank it builds, and
    ``QueueGroundwater`` then replaces ``self.tank`` (``storage.py:53-59``, ``:71``, ``:301``):
    the check reports ``capacity - initial storage`` for ever."""
    gw = queue_groundwater(initial_storage=4.0, timearea={0: 1.0})
    day(gw)
    assert float(gw.push_check(None, active=TRUE)[0]) == 96.0
    gw.push_set(vol(30.0), active=TRUE)
    assert float(gw.tank.storage[0]) == 34.0
    assert float(gw.push_check(None, active=TRUE)[0]) == 96.0
    assert float(gw.push_check(vol(5.0), active=TRUE)[0]) == 5.0


def test_the_frozen_push_check_equals_live_wsimod():
    pytest.importorskip("wsimod")
    from wsimod.core import constants
    from wsimod.nodes import storage as wsimod_storage

    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()

    def full(volume):  # a WSIMOD VQIP names every pollutant
        return {"volume": volume, **dict.fromkeys(constants.POLLUTANTS, 0.0)}

    theirs = wsimod_storage.QueueGroundwater(
        name="n", capacity=100, initial_storage=4, timearea={0: 1}
    )
    ours = queue_groundwater(initial_storage=4.0, timearea={0: 1.0})
    day(ours)
    ours.push_set(vol(30.0), active=TRUE)
    theirs.push_set(full(30.0))
    assert theirs.tank.storage["volume"] == 34.0
    for request in (None, 5.0, 500.0):
        got = ours.push_check(None if request is None else vol(request), active=TRUE)
        ref = theirs.push_check(None if request is None else full(request))
        assert float(got[0]) == ref["volume"]


def test_distribute_pushes_the_active_storage_and_pulls_what_was_sent():
    gw = queue_groundwater(initial_storage=5.0, timearea={0: 1.0})
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    arc("out", gw, sea, capacity=3.0)
    ctx = day(gw, sea)
    gw.distribute(active=TRUE)  # storage.py:350-364
    assert float(gw.tank.active_storage[0]) == 2.0 and float(gw.tank.storage[0]) == 2.0
    assert bool(ctx.events["groundwater_couldnt_push_all"]["n"])
    assert "miscalculated_tank_storage" not in ctx.events or not bool(
        ctx.events["miscalculated_tank_storage"]["n"]
    )
    gw.infiltrate(active=TRUE)  # a no-op (:366-368)
    assert float(gw.tank.storage[0]) == 2.0


# --- RiverReservoir ---------------------------------------------------------------------------
def test_a_river_reservoir_spills_downstream_and_counts_it_as_environmental_flow():
    rr = make("RiverReservoir", capacity=10.0, initial_storage=8.0, environmental_flow=5.0)
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    pipe = arc("out", rr, sea, capacity=4.0)
    ctx = day(rr, sea)
    # push (storage.py:1060-1084): 8 + 5 = 13 forced in, 3 ponded and pushed downstream
    assert float(rr.push_set(vol(5.0), active=TRUE)[0]) == 0.0
    assert float(rr.tank.storage[0]) == 10.0
    assert float(rr.total_environmental_satisfied) == 3.0
    # check (:1086-1110): no excess left, plus the 1 the pipe can still take
    assert float(rr.push_check(None, active=TRUE)[0]) == 1.0
    assert float(rr.push_check(vol(0.5), active=TRUE)[0]) == 0.5
    # satisfy (:1112-1128): 5 - 3 = 2 owed and pulled, 1 fits the pipe, 1 forced back
    rr.satisfy_environmental(active=TRUE)
    assert float(rr.tank.storage[0]) == 9.0
    assert float(pipe.flow_in) == 4.0
    assert float(rr.total_environmental_satisfied) == 5.0
    assert bool(ctx.events["environmental_not_able_to_push"]["n"])
    rr.end_timestep()  # :1130-1133
    assert float(rr.total_environmental_satisfied) == 0.0
    assert set(rr.stores()) == {"tank.storage", "total_environmental_satisfied"}


def test_the_river_reservoir_gradient_reaches_the_environmental_flow():
    rr = make("RiverReservoir", capacity=10.0, initial_storage=8.0)
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    arc("out", rr, sea)
    day(rr, sea)
    flow = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    rr.bind({"environmental_flow": flow})
    rr.satisfy_environmental(active=TRUE)
    # 5 owed and pulled out of 8, all pushed: the storage is 8 - flow
    assert float(rr.tank.storage[0].detach()) == 3.0
    rr.tank.storage[0].backward()
    assert float(flow.grad) == -1.0


def test_a_negative_bound_infiltration_pct_is_refused_by_name():
    gw = make("Groundwater")
    gw.bind({"infiltration_pct": torch.tensor([0.1, 0.0])})
    with pytest.raises(ValueError, match=r"node 'n'.*infiltration_pct is negative"):
        gw.bind({"infiltration_pct": torch.tensor([0.1, -0.2])})


# --- River ----------------------------------------------------------------------------------
PP = Pollutants(("phosphate",), ("phosphate",), ())  # one additive pollutant


def river(**parameters):
    return make("River", **parameters)


def test_the_river_routing_coefficient():
    # storage.py:900-915: kt = damp x (length / velocity) = 0.1 x 5000 = 500
    node = river(damp=0.1, length=1000.0, velocity=0.2)
    day(node)
    rc = node.get_riverrc()
    assert float(rc) == pytest.approx(1 - 500 + 500 * math.exp(-1 / 500), rel=1e-15, abs=0)
    # damp 0: kt == 0 gives 1, and the gradient with respect to the velocity is finite
    still = river(damp=0.0, length=1000.0)
    day(still)
    velocity = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    still.bind({"velocity": velocity})
    rc = still.get_riverrc()
    assert float(rc.detach()) == 1.0
    rc.backward()
    assert float(velocity.grad) == 0.0
    # the gradient with respect to damp where kt > 0: d/dkt (1 - kt + kt exp(-1/kt)) x dkt/ddamp
    damp = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    node.bind({"damp": damp})
    node.get_riverrc().backward()
    kt = 0.1 * (1000.0 / 0.2)
    slope = -1 + math.exp(-1 / kt) + math.exp(-1 / kt) / kt
    assert float(damp.grad) == pytest.approx(slope * (1000.0 / 0.2), rel=1e-12)


def test_exp_python_is_pythons_exp_bitwise():
    rng = random.Random(7)
    values = [-1 / rng.uniform(1e-3, 50) for _ in range(20000)] + [0.0, -1.0, -700.0]
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    y = exp_python(x)
    assert y.tolist() == [math.exp(v) for v in values]
    y.sum().backward()
    assert x.grad.tolist() == pytest.approx(y.tolist(), rel=1e-15, abs=0)
    # riverrc is a batch of instances, each its own value
    kt = torch.tensor([0.0, 0.5, 500.0], dtype=torch.float64)
    assert riverrc(kt).tolist() == [
        1.0,
        1 - 0.5 + 0.5 * math.exp(-1 / 0.5),
        1 - 500.0 + 500.0 * math.exp(-1 / 500.0),
    ]


def upstream_check(volume):
    """The WSIMOD call a river's pull check makes on its upstream arc (no request)."""
    return ArcCall(
        "up",
        "send_pull_check",
        None,
        "default",
        False,
        {"volume": volume, "phosphate": 0.0},
        0.0,
        {"volume": 0.0, "phosphate": 0.0},
        0.0,
        {"volume": 0.0, "phosphate": 0.0},
    )


def test_pull_check_spreads_the_rivers_mass_over_upstream_water():
    """storage.py:660-686: tank volume 4 with phosphate 2, upstream 6, mrf 1, riverrc 0.5. The
    reply volume is max(4 + 6 - 1 / 0.5, 0) = 8 and the phosphate 2 x 8 / 10: the rescale's base
    is the raised volume 10, not the tank's 4."""
    spec = NodeSpec(
        "n", "River", "River", {"initial_storage": {"volume": 4.0, "phosphate": 2.0}, "mrf": 1.0}
    )
    node = make_node(spec, None, PP)
    node.get_riverrc = lambda: torch.tensor(0.5, dtype=torch.float64)  # riverrc fixed at 0.5
    script: list = []
    upstream = ScriptedArc(
        ArcSpec("up", "Arc", "other", "n"),
        script,
        PP,
        scale=1.0,
        in_type="River",
        out_type="River",
    )
    node.connect(upstream, "in")
    node.ctx = Ctx(PP, FA, torch.Size(()), day=0, node_names=["n"])
    script[:] = [upstream_check(6.0)]
    reply = node.pull_check(None, active=TRUE)
    assert float(reply[0]) == 8.0
    assert float(reply[1]) == 2.0 * (8.0 / 10.0)
    upstream.assert_exhausted()
    # a request of 3 clips the volume; the base is still 10
    script[:] = [upstream_check(6.0)]
    reply = node.pull_check(PP.to_vqip({"volume": 3.0}), active=TRUE)
    assert float(reply[0]) == 3.0 and float(reply[1]) == 2.0 * (3.0 / 10.0)
    # an mrf above everything leaves nothing (the max against 0)
    node.bind({"mrf": torch.tensor(100.0, dtype=torch.float64)})
    script[:] = [upstream_check(6.0)]
    assert float(node.pull_check(None, active=TRUE)[0]) == 0.0


def test_pull_set_takes_the_tank_first_then_the_upstream_arcs():
    # storage.py:688-709: avail = min(max(3 + 5 - 0, 0), 6) = 6; the tank gives 3 and the single
    # upstream River arc is asked for the other 3
    node = river(initial_storage=3.0, mrf=0.0)
    script: list = [
        upstream_check(5.0),
        call("up", "send_pull_request", 3.0, 3.0),
    ]
    upstream = ScriptedArc(
        ArcSpec("up", "Arc", "other", "n"),
        script,
        P,
        scale=1.0,
        in_type="River",
        out_type="River",
    )
    script[0] = ArcCall(
        "up",
        "send_pull_check",
        None,
        "default",
        False,
        {"volume": 5.0},
        0.0,
        {"volume": 0.0},
        0.0,
        {"volume": 0.0},
    )
    node.connect(upstream, "in")
    day(node)
    reply = node.pull_set(vol(6.0), active=TRUE)
    upstream.assert_exhausted()
    assert float(reply[0]) == 6.0 and float(node.tank.storage[0]) == 0.0


def test_push_is_forced_and_the_check_returns_the_unbounded_capacity():
    node = river(initial_storage=1e16)  # far above the capacity: still accepted
    day(node)
    assert float(node.push_set(vol(5.0), active=TRUE)[0]) == 0.0
    assert float(node.tank.storage[0]) == 1e16 + 5.0
    assert float(node.push_check(None, active=TRUE)[0]) == 1e15
    assert float(node.push_check(vol(7.0), active=TRUE)[0]) == 7.0


def test_distribute_sends_the_routing_share_and_forces_the_rest_back():
    node = river(initial_storage=100.0, length=17280.0, velocity=17280.0, damp=0.5)
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    pipe = arc("out", node, sea, capacity=10.0)
    ctx = day(node, sea)
    rc = float(node.get_riverrc())  # kt = 0.5
    assert rc == 1 - 0.5 + 0.5 * math.exp(-1 / 0.5)
    node.distribute(active=TRUE)
    # 100 x rc is asked of the pipe, which takes 10; the rest is forced back
    assert float(pipe.flow_in) == 10.0
    assert float(node.tank.storage[0]) == 90.0
    assert bool(ctx.events["river_cant_push"]["n"])
    node.calculate_discharge(active=TRUE)  # a no-op without biochemistry
    assert float(node.tank.storage[0]) == 90.0


def test_the_tank_is_never_full_and_area_is_length_times_width():
    node = river(length=50.0, width=4.0, capacity=3.0, area=1.0, initial_storage=2.0)
    day(node)
    # the merged capacity and area are ignored (storage.py:524-540, :643-658)
    assert float(node.tank.area) == 200.0 and float(node.tank.capacity) == 1e15
    node.bind({"length": torch.tensor([10.0, 20.0], dtype=torch.float64)})
    assert node.tank.area.tolist() == [40.0, 80.0]
    assert node.tank.capacity.tolist() == [1e15, 1e15]


def test_river_derived_warns_when_the_depth_is_not_the_default():
    spec = NodeSpec("r", "River", "River", {"length": 10, "width": 3, "depth": 5})
    with pytest.warns(UserWarning, match="depth parameter is unused"):
        assert river_derived(spec) == {"area": 30.0, "capacity": 1e15}
    quiet = NodeSpec("r", "River", "River", {"length": 10, "width": 3, "depth": 2})
    assert river_derived(quiet) == {"area": 30.0, "capacity": 1e15}


def test_a_river_with_nitrate_is_refused():
    network = read_wsimod_config(
        {
            "pollutants": ["nitrate", "temperature"],
            "additive_pollutants": ["nitrate"],
            "non_additive_pollutants": ["temperature"],
            "dates": ["2000-01-01"],
            "nodes": [{"name": "r", "type_": "River"}],
            "arcs": [],
        }
    )
    with pytest.raises(ValueError, match="nitrate.*3b"):
        River.validate(network.nodes[0], network, pollutants=True)
    # a quantity-only run reads no pollutant (W10)
    assert River.validate(network.nodes[0], network, pollutants=False) == []


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"velocity": 0.0}, "velocity 0.0 is not positive"),
        ({"damp": -0.1}, "damp -0.1 is negative"),
    ],
)
def test_parameters_that_wsimod_cannot_run_a_river_on_are_refused(parameters, match):
    with pytest.raises(ValueError, match=match):
        _validate("River", **parameters)
    assert _validate("River", damp=0.0, mrf=2.0) == []


def test_the_riparian_buffer_check_is_refused_by_name_with_3c():
    node = river()
    day(node)
    with pytest.raises(KeyError, match="3c"):
        node.pull_check(None, ("RiparianBuffer", "volume"), active=TRUE)


def test_river_handlers_and_recursion():
    spec = NodeSpec("n", "River", "River", {})
    assert River.check_recurses(spec, "pull", "default")
    assert not River.check_recurses(spec, "push", "default")
    assert River.step_methods == ("distribute", "calculate_discharge")
    assert River.distributes == ("out", "in")


def test_a_batch_of_rivers_equals_its_instances():
    def run(damp, active, shape):
        node = river(initial_storage=40.0, length=17280.0, velocity=17280.0)
        sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
        pipe = arc("out", node, sea, capacity=15.0)
        day(node, sea, pipe, shape=shape)
        node.bind({"damp": damp})
        node.distribute(active=active)
        return node.tank.storage[..., 0], pipe.flow_in

    damps = torch.tensor([0.2, 0.0, 1.5], dtype=torch.float64)
    active = torch.tensor([True, False, True])
    tank, flow = run(damps, active, (3,))
    assert float(tank[1]) == 40.0 and float(flow[1]) == 0.0
    for i in (0, 2):
        single_tank, single_flow = run(damps[i], TRUE, ())
        assert float(single_tank) == float(tank[i]) and float(single_flow) == float(flow[i])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_river_derives_what_construction_derives():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    mapping = {
        "pollutants": ["phosphate", "temperature"],
        "additive_pollutants": ["phosphate"],
        "non_additive_pollutants": ["temperature"],
        "dates": ["2000-01-01"],
        "nodes": [
            {"name": "r", "type_": "River", "length": 300.0, "width": 7.0},
            {"name": "plain", "type_": "River"},
        ],
        "arcs": [],
        "overrides": {
            "nodes": {
                "r": {
                    "type_": "River",
                    "name": "r",
                    "length": 450.0,
                    "width": 9.0,
                    "area": 5.0,  # ignored by WSIMOD (a warning), and by noodl
                    "capacity": 3.0,
                }
            }
        },
    }
    model = wsimod_model(mapping)
    network = read_wsimod_config(mapping)
    for spec in network.nodes:
        theirs = model.nodes[spec.name]
        derived = river_derived(spec)
        assert derived["area"] == pytest.approx(theirs.tank.area, rel=1e-15, abs=0)
        assert derived["capacity"] == pytest.approx(theirs.tank.capacity, rel=1e-15, abs=0)
        ours = make_node(spec, network, QUANTITY_ONLY)
        assert float(ours.tank.area) == theirs.tank.area
        assert float(ours.tank.capacity) == theirs.tank.capacity
        assert float(ours.get_riverrc()) == theirs.get_riverrc()
    assert river_derived(network.nodes[0])["area"] == 450.0 * 9.0


# --- the river order and a stepped chain --------------------------------------------------------
def test_the_river_order_is_most_upstream_first_with_ties_in_configuration_order():
    from tests.apps.wsimod._stepping import river_order
    from tests.verification._wsimod_cases import CASES

    network = read_wsimod_config(CASES["river_chain"].build())
    # r_lost links to no Waste and is left out (model.py:730-735, :780-807)
    assert river_order(network) == ["r_head", "r_mid", "r_side", "r_low", "r_end"]
    # no river arc at all: no order
    no_river_arcs = read_wsimod_config(
        {**CASES["river_chain"].build(), "arcs": [], "orchestration": []}
    )
    assert river_order(no_river_arcs) == []
    mapping = CASES["river_chain"].build()
    mapping["nodes"] = [n for n in mapping["nodes"] if n["name"] != "sea"]
    mapping["arcs"] = [a for a in mapping["arcs"] if "sea" not in (a["in_port"], a["out_port"])]
    with pytest.raises(ValueError, match="no Waste"):
        river_order(read_wsimod_config(mapping))


@pytest.mark.parametrize("quantity_only", [True, False])
def test_the_stepped_river_chain_equals_wsimod_day_by_day(quantity_only):
    pytest.importorskip("wsimod")
    from tests.apps.wsimod._stepping import step_days
    from tests.verification._wsimod_cases import CASES
    from tests.verification._wsimod_reference import wsimod_model, wsimod_reference

    case = CASES["river_chain"]
    days = case.days
    model = wsimod_model(case.build())
    # WSIMOD's own river order, tied rivers in hash order
    assert model.river_discharge_order[0] == "r_head"
    assert set(model.river_discharge_order[1:3]) == {"r_mid", "r_side"}
    assert set(model.river_discharge_order[3:]) == {"r_low", "r_end"}
    reference = wsimod_reference(case.build(), days=days)
    network = read_wsimod_config(case.build())
    p = QUANTITY_ONLY if quantity_only else network.pollutants
    columns = [reference.pollutants.index(n) for n in p.names]
    rivers = ["r_head", "r_mid", "r_low", "r_side", "r_end", "r_lost", "res"]
    for day_, _, engine in step_days(network, p, days):
        for name in rivers:
            row = reference.tank_names.index((name, "tank"))
            ours = engine.nodes[name].tank.storage
            assert float(ours[0]) == float(reference.tank_storage[day_, row]), (day_, name)
            for k, column in enumerate(columns):
                assert float(ours[1 + k]) == float(reference.tank_quality[day_, row, column])
