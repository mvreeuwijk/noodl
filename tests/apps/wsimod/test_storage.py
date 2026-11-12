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

from noodl.apps.wsimod import build_model, initial_drivers, initial_state
from noodl.apps.wsimod.arcs import ARC_CLASSES, DecayArcAlt
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.node import NODE_CLASSES, Waste
from noodl.apps.wsimod.run import wsimod_run
from noodl.apps.wsimod.storage import (
    COMPENSATED_SUM,
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
    window_sum,
)
from noodl.apps.wsimod.structure import forcing_readers
from noodl.apps.wsimod.tanks import DecayQueueTank
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from tests.apps.wsimod._replay import ScriptedArc, StoreNode
from tests.verification._wsimod_reference import ArcCall, pad_register

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
DECAYS = {"phosphate": {"constant": 0.1, "exponent": 1.02}}
POLLUTANT_LISTS = {
    "pollutants": ["phosphate", "ammonia"],
    "additive_pollutants": ["phosphate", "ammonia"],
    "non_additive_pollutants": [],
}


def _decaying(type_="Storage", *, decays=DECAYS, days=3, missing=(), overrides=None, **values):
    """A node ``r`` of ``type_`` with ``decays`` (``None`` included) and a temperature forcing
    on every date but those at the indices ``missing``."""
    dates = [f"2000-01-{k + 1:02d}" for k in range(days)]
    data = {("temperature", d): 15.0 for k, d in enumerate(dates) if k not in missing}
    node = {"name": "r", "type_": type_, "decays": decays, "data_input_dict": data, **values}
    mapping = {
        **POLLUTANT_LISTS,
        "dates": dates,
        "nodes": [node, {"name": "sink", "type_": "Waste"}],
        "arcs": [],
        "orchestration": [],
    }
    if overrides is not None:
        mapping["overrides"] = {"nodes": {"r": overrides}}
    return mapping


def _override(path) -> dict:
    return {"type_": "Groundwater", "name": "r", "filename": str(path)}


def test_a_decaying_groundwater_needs_a_filename_override(tmp_path):
    """The configured forcing is lost (storage.py:202-203), so the temperature comes from an
    override's file (X11); a file that misses a date is refused by name too."""
    network = read_wsimod_config(_decaying("Groundwater", decays=None))
    spec = network.nodes[0]
    assert NODE_CLASSES["Groundwater"].validate(spec, network, pollutants=True) == []
    with pytest.raises(ValueError, match=r"Groundwater.*filename"):
        build_model(read_wsimod_config(_decaying("Groundwater")))
    with pytest.raises(ValueError, match=r"Groundwater.*filename"):
        build_model(read_wsimod_config(_decaying("Groundwater", decays={})))
    rows = "".join(f"temperature,2000-01-0{k},15.0\n" for k in (1, 2))
    short = tmp_path / "short.csv"
    short.write_text("variable,time,value\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError, match=r"2000-01-03.*filename"):
        build_model(read_wsimod_config(_decaying("Groundwater", overrides=_override(short))))
    full = tmp_path / "full.csv"
    full.write_text(short.read_text() + "temperature,2000-01-03,15.0\n", encoding="utf-8")
    network = read_wsimod_config(_decaying("Groundwater", overrides=_override(full)))
    model, _, drivers = build_model(network)
    assert model.unread == ["r.data_input_dict"]  # the decoy
    assert drivers["wsimod.temperature"].shape == (3, 1)


def _queue_override(path, **values) -> dict:
    return {"type_": "QueueGroundwater", "name": "r", "filename": str(path), **values}


def _temperature_file(tmp_path, days=3):
    path = tmp_path / "temperature.csv"
    rows = "".join(f"temperature,2000-01-{k + 1:02d},15.0\n" for k in range(days))
    path.write_text("variable,time,value\n" + rows, encoding="utf-8")
    return path


def test_a_decaying_queue_groundwater_needs_a_filename_override(tmp_path):
    """As a ``Groundwater`` (X11): the configured forcing is lost (``storage.py:291-292``), so a
    ``decays`` dict, empty or not, needs an override's temperature; with one it builds a
    ``DecayQueueTank`` (``storage.py:307-315``)."""
    network = read_wsimod_config(_decaying("QueueGroundwater", decays=None))
    spec = network.nodes[0]
    assert NODE_CLASSES["QueueGroundwater"].validate(spec, network, pollutants=True) == []
    for decays in (DECAYS, {}):
        with pytest.raises(ValueError, match=r"QueueGroundwater.*filename"):
            build_model(read_wsimod_config(_decaying("QueueGroundwater", decays=decays)))
    override = _queue_override(_temperature_file(tmp_path))
    network = read_wsimod_config(_decaying("QueueGroundwater", overrides=override))
    model, _, drivers = build_model(network)
    assert model.unread == ["r.data_input_dict"]  # the decoy
    assert drivers["wsimod.temperature"].shape == (3, 1)
    tank = model.wsimod_day.engine.nodes["r"].tank
    assert isinstance(tank, DecayQueueTank) and isinstance(tank.internal_arc, DecayArcAlt)
    assert model.wsimod_day.state_labels("wsimod.total_decayed") == [("r", "tank.internal_arc")]


def _bound_queue_groundwater(network):
    spec = network.nodes[0]
    node = make_node(spec, network, network.pollutants)
    node.ctx = Ctx(network.pollutants, FA, torch.Size(()), day=0, node_names=["r"])
    node.bind(node.configured_parameters(network.dates[0]))
    return node


def test_decaying_delays_a_queue_groundwater_by_one_day(tmp_path):
    """X12: ``DecayQueueTank``'s default ``number_of_timesteps`` is 1 (``tanks.py:648``) and
    ``QueueGroundwater`` passes none (``storage.py:308-315``): with time-area ``{0: 0.5, 1:
    0.5}``, WSIMOD's queue holds the push under keys 1 and 2, and so does noodl's register;
    without ``decays`` the half under time 0 is active at once."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    push = {"volume": 4.0, "phosphate": 0.4, "ammonia": 0.2}
    for decays, keys in ((DECAYS, [1, 2]), (None, [1])):
        override = _queue_override(_temperature_file(tmp_path)) if decays else None
        mapping = _decaying(
            "QueueGroundwater",
            decays=decays,
            overrides=override,
            timearea={0: 0.5, 1: 0.5},
            capacity=100.0,
        )
        model = wsimod_model(mapping)
        theirs = model.nodes["r"]
        theirs.t = model.dates[0]
        theirs.push_set(dict(push))
        queue = theirs.tank.internal_arc.queue
        assert sorted(k for k, v in queue.items() if v["volume"] > 0) == keys
        ours = _bound_queue_groundwater(read_wsimod_config(mapping))
        ours.push_set(ours.p.to_vqip(push), "default", active=torch.tensor(True))
        register = ours.tank.internal_arc.register
        assert [t for t in range(register.shape[-2]) if register[t, 0] > 0] == keys
        for t in keys:
            assert ours.tank.internal_arc.present[t]
            assert register[t, 0].item() == queue[t]["volume"]
            assert register[t, 1].item() == queue[t]["phosphate"]
        active = 0.0 if decays else 2.0  # the share under time 0
        assert ours.tank.active_storage[0].item() == theirs.tank.active_storage["volume"] == active


def test_number_of_timesteps_override(tmp_path):
    """X28: on a decaying ``QueueGroundwater`` the override ``number_of_timesteps`` sets the
    tank's and its internal arc's, as ``DecayQueueTank.apply_overrides`` does
    (``tanks.py:682-685``); on a non-decaying one it is refused by name, as in 3a."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    override = _queue_override(_temperature_file(tmp_path), number_of_timesteps=0)
    mapping = _decaying("QueueGroundwater", overrides=override, timearea={0: 0.5, 2: 0.5})
    tank = wsimod_model(mapping).nodes["r"].tank
    assert (tank.number_of_timesteps, tank.internal_arc.number_of_timesteps) == (0, 0)
    ours = _bound_queue_groundwater(read_wsimod_config(mapping)).tank
    assert (ours.number_of_timesteps, ours.internal_arc.number_of_timesteps) == (0, 0)
    assert ours.internal_arc.length == 2  # 0 + the largest time-area key
    plain = _decaying("QueueGroundwater", decays=None, overrides=dict(override))
    with pytest.raises(ValueError, match="number_of_timesteps"):
        build_model(read_wsimod_config(plain))
    bad = _queue_override(_temperature_file(tmp_path), number_of_timesteps=1.5)
    with pytest.raises(ValueError, match="number_of_timesteps"):
        build_model(read_wsimod_config(_decaying("QueueGroundwater", overrides=bad)))


@pytest.mark.parametrize("cls", ["Storage", "Reservoir", "RiverReservoir", "River"])
def test_storage_with_decays_reads_its_temperature(cls):
    """The DecayTank reads ``("temperature", t)`` at the end of every day (``core.py:484-487``),
    also with an empty ``decays``; a node with ``decays: None`` needs none."""
    for decays in (DECAYS, {}):
        missing = r"node 'r' has no 'temperature' forcing on 2000-01-02"
        with pytest.raises(ValueError, match=missing):
            build_model(read_wsimod_config(_decaying(cls, decays=decays, missing=(1, 2))))
        build_model(read_wsimod_config(_decaying(cls, decays=decays)))
    build_model(read_wsimod_config(_decaying(cls, decays=None, missing=(0, 1, 2))))


def test_a_decaying_storage_family_node_is_a_temperature_reader():
    network = read_wsimod_config(_decaying("Reservoir", decays={}))
    assert forcing_readers(network)["temperature"] == ["r"]
    network = read_wsimod_config(_decaying("Reservoir", decays=None))
    assert "temperature" not in forcing_readers(network)


def test_the_decays_drivers_are_dense_with_inert_rows_for_a_node_without_decays():
    """X9, X34: ``(n, K)`` rows, the node's own decays where listed, ``(0, 1)`` elsewhere."""
    mapping = _decaying("Storage")
    mapping["nodes"].insert(1, {"name": "plain", "type_": "Storage", "decays": None})
    model, _, drivers = build_model(read_wsimod_config(mapping))
    constant = drivers["wsimod.Storage.decays.constant"]
    exponent = drivers["wsimod.Storage.decays.exponent"]
    assert model.wsimod_day.drivers["wsimod.Storage.decays.constant"].labels == ["r", "plain"]
    assert constant.tolist() == [[0.1, 0.0], [0.0, 0.0]]
    assert exponent.tolist() == [[1.02, 1.0], [1.0, 1.0]]


def test_a_decays_override_merges_into_a_dict_and_is_refused_on_none():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    merged = {"ammonia": {"constant": 0.2, "exponent": 1.0}}
    override = {"type_": "Storage", "name": "r", "decays": merged}
    mapping = _decaying("Storage", overrides=override)
    ours = read_wsimod_config(mapping).nodes[0].parameters["decays"]
    theirs = wsimod_model(mapping).nodes["r"].decays
    assert ours == theirs == {**DECAYS, **merged}
    # a decays-None node refuses it, as WSIMOD does (storage.py:96-100)
    with pytest.raises(ValueError, match="decays"):
        read_wsimod_config(_decaying("Storage", decays=None, overrides=override))
    with pytest.raises(ValueError, match="decays"):
        wsimod_model(_decaying("Storage", decays=None, overrides=override))
    # an entry replaces the pollutant's whole dict (dict.update), so a lone constant is refused
    # by build_model after the merge, as WSIMOD's make_decay would crash on the missing exponent
    partial = {"type_": "Storage", "name": "r", "decays": {"phosphate": {"constant": 0.3}}}
    network = read_wsimod_config(_decaying("Storage", overrides=partial))
    assert network.nodes[0].parameters["decays"] == {"phosphate": {"constant": 0.3}}
    with pytest.raises(ValueError, match="exponent"):
        build_model(network)


def test_a_decays_none_node_refuses_a_changed_inert_driver():
    """D15: ``Reservoir.bind`` refuses a ``decays.*`` entry other than ``(0, 1)`` on a node
    without ``decays`` (and on a pollutant its ``decays`` does not list), naming node and
    pollutant."""
    model, _, _ = build_model(read_wsimod_config(_decaying("Reservoir", decays=None)))
    for key, pollutant, value in (
        ("wsimod.Reservoir.decays.constant", "phosphate", 0.1),
        ("wsimod.Reservoir.decays.exponent", "ammonia", 1.1),
    ):
        drivers = initial_drivers(model, values={key: {"r": {pollutant: value}}})
        with pytest.raises(ValueError, match=rf"node 'r'.*{key.rsplit('.', 1)[1]}.*{pollutant}"):
            model.step(initial_state(model), drivers, dt=86400.0, t=0.0)
    # a listed pollutant takes any value; an unlisted one stays inert
    model, _, _ = build_model(read_wsimod_config(_decaying("Reservoir")))
    key = "wsimod.Reservoir.decays.constant"
    drivers = initial_drivers(model, values={key: {"r": {"phosphate": 0.3}}})
    model.step(initial_state(model), drivers, dt=86400.0, t=0.0)
    drivers = initial_drivers(model, values={key: {"r": {"ammonia": 0.3}}})
    with pytest.raises(ValueError, match="ammonia"):
        model.step(initial_state(model), drivers, dt=86400.0, t=0.0)


def test_a_decaying_storage_decays_at_the_end_of_the_day_and_closes_the_balance():
    """``DecayTank.end_timestep_decay``: ``phosphate`` falls by ``constant * exponent ** (T -
    20)`` of its storage; the decayed mass is the ``"DecayTank.decayed"`` term and leaves the
    system, so the balance closes; ``total_decayed`` is a per-day store, not carried."""
    initial = {"volume": 10.0, "phosphate": 4.0, "ammonia": 1.0}
    network = read_wsimod_config(_decaying("Storage", initial_storage=initial))
    model, state, drivers = build_model(network)
    result = wsimod_run(model, state, drivers)
    factor = 0.1 * 1.02 ** (15.0 - 20.0)
    decayed = result.mass_balance.terms["DecayTank.decayed"]
    assert decayed[0].tolist() == [0.0, 4.0 * factor, 0.0]
    assert float(result.mass_balance.residual.abs().max()) < 1e-12
    node = model.wsimod_day.engine.nodes["r"]
    assert "tank.total_decayed" in node.store_paths
    assert "tank.total_decayed" not in node.carried_paths
    # WSIMOD's record_all lists Tank, QueueTank and ResidenceTank only: the DecayTank drops out
    # of the tank_* results while its storage stays in the state (X40)
    assert model.wsimod_day.tank_names == []
    assert model.wsimod_day.state_labels("wsimod.storage") == [("r", "tank")]


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


# --- a stepped river chain (the river order: test_build.py) -------------------------------------
@pytest.mark.parametrize("quantity_only", [True, False])
def test_the_stepped_river_chain_equals_wsimod_day_by_day(quantity_only):
    pytest.importorskip("wsimod")
    from noodl.apps.wsimod.day import WsimodDay
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
    for day_, _, engine in WsimodDay(network, pollutants=not quantity_only).run_days(days=days):
        for name in rivers:
            row = reference.tank_names.index((name, "tank"))
            ours = engine.nodes[name].tank.storage
            assert float(ours[0]) == float(reference.tank_storage[day_, row]), (day_, name)
            for k, column in enumerate(columns):
                assert float(ours[1 + k]) == float(reference.tank_quality[day_, row, column])


def test_the_river_pull_gradient_is_finite_clipped_and_masked():
    """Three rivers pulled through an upstream arc: 0 is in the differentiable range of
    ``max(V + upstream - mrf / riverrc, 0)``, 1 sits on the clip at 0 (constant), 2 is inactive."""
    node = river(initial_storage=3.0, length=17280.0, velocity=17280.0, damp=0.5)
    upstream = StoreNode("up", P, storage=5.0, wsimod_type="River")
    pipe = arc("up_pipe", upstream, node)
    day(node, upstream, pipe, shape=(3,))
    mrf = torch.tensor([2.0, 100.0, 2.0], dtype=torch.float64, requires_grad=True)
    damp = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64, requires_grad=True)
    node.bind({"mrf": mrf, "damp": damp})
    active = torch.tensor([True, True, False])
    reply = node.pull_set(vol(6.0), active=active)
    held = node.tank.storage[..., 0] + upstream.storage[..., 0]
    assert held[2].item() == 8.0  # the inactive instance is unchanged
    assert node.tank.storage[2, 0].item() == 3.0 and upstream.storage[2, 0].item() == 5.0
    assert reply[1, 0].item() == 0.0  # mrf / riverrc exceeds everything: the clip at 0
    held.sum().backward()
    for grad in (mrf.grad, damp.grad):
        assert torch.isfinite(grad).all()
    assert mrf.grad[0].item() != 0.0 and damp.grad[0].item() != 0.0
    # the clipped and the inactive instances take no gradient
    assert mrf.grad[1].item() == 0.0 and mrf.grad[2].item() == 0.0
    assert damp.grad[1].item() == 0.0 and damp.grad[2].item() == 0.0


def test_the_river_distribute_gradient_of_the_arc_flow():
    """The flow alone: tank plus flow is conserved, so its gradient would be 0."""
    node = river(initial_storage=40.0, length=17280.0, velocity=17280.0)
    sea = Waste(NodeSpec("sea", "Waste", "Waste", {}), P)
    pipe = arc("out", node, sea)
    day(node, sea, pipe, shape=(3,))
    damp = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
    node.bind({"damp": damp})
    node.distribute(active=torch.tensor([True, True, False]))
    assert pipe.flow_in[2].item() == 0.0
    pipe.flow_in.sum().backward()
    assert torch.isfinite(damp.grad).all()
    for i, d in enumerate((0.2, 0.5)):  # flow = 40 x riverrc(damp), d riverrc / d kt at kt = d
        e = math.exp(-1 / d)
        assert damp.grad[i].item() == pytest.approx(40.0 * (-1 + e + e / d), rel=1e-12)
    assert damp.grad[2].item() == 0.0


# --- River biochemistry (storage.py:723-898) ----------------------------------------------------
BIO_POLLUTANTS = (
    "temperature",
    "phosphate",
    "org-phosphorus",
    "ammonia",
    "nitrate",
    "org-nitrogen",
)
BIO = Pollutants(BIO_POLLUTANTS, BIO_POLLUTANTS[1:], ("temperature",))
BIO_LISTS = {
    "pollutants": list(BIO.names),
    "additive_pollutants": list(BIO.additive),
    "non_additive_pollutants": list(BIO.non_additive),
}
BIO_PARAMETERS = (
    "uptake_PNratio",
    "denpar_w",
    "T_wdays",
    "halfsatINwater",
    "hsatTP",
    "limpppar",
    "prodNpar",
    "prodPpar",
    "muptNpar",
    "muptPpar",
)
SITE = "node:r/biochemical_processes"


def bio_river(state, t_air, temperatures, phosphorus, *, p=BIO, record=False, **parameters):
    """A River with biochemistry at one state: ``state`` a VQIP mapping, the registers padded
    from WSIMOD-style lists."""
    node = make_node(NodeSpec("r", "River", "River", parameters), None, p)
    ctx = Ctx(p, FA, torch.Size(()), day=0, node_names=["r"], record_decisions=record)
    node.ctx = ctx
    node.tank.storage = p.to_vqip(state)
    node.lagged_temperatures = pad_register(temperatures, node.lag_sizes[0])
    node.lagged_total_phosphorus = pad_register(phosphorus, node.lag_sizes[1])
    node.bind({"temperature": torch.tensor(t_air, dtype=torch.float64)})
    return node, ctx


# a point away from every guard and cap: warm (T > 5), producing, nothing capped
HAND_STATE = {
    "volume": 1000.0,
    "temperature": 12.0,
    "phosphate": 1.0,
    "org-phosphorus": 0.3,
    "ammonia": 3.0,
    "nitrate": 10.0,
    "org-nitrogen": 2.0,
}
HAND_TEMPERATURES = [5.0 + 0.5 * i for i in range(21)]  # full, and rising
HAND_PHOSPHORUS = [0.002] * 366  # full


def test_one_biochemistry_call_by_hand():
    """Steps 1-8 of the spec at one point, every intermediate a Python expression from the
    defaults table (``storage.py:539-564``, ``:723-898``)."""
    node, ctx = bio_river(HAND_STATE, 17.0, HAND_TEMPERATURES, HAND_PHOSPHORUS, record=True)
    in_, out_ = node.biochemical_processes(active=TRUE)
    uptake_PNratio, denpar_w, T_wdays = 1 / 7.2, 0.0015, 5
    halfsatINwater, hsatTP, limpppar = 1.5 * 1e-3, 0.05 * 1e-3, 0.1 * 1e-3
    prodNpar, prodPpar, muptNpar, muptPpar = 0.001, 0.0001, 0.001, 0.0001
    max_temp_lag, max_phosphorus_lag = 20, 365
    s = dict(HAND_STATE)
    area = 200 * 20
    # 1. depth
    current_depth = s["volume"] / area
    # 2. temperature relaxation
    s["temperature"] = (1 - 1 / T_wdays) * s["temperature"] + (1 / T_wdays) * 17.0
    T = s["temperature"]
    # 3. lags (full lists: the oldest goes)
    temperatures = HAND_TEMPERATURES[1:] + [T]
    phosphorus = HAND_PHOSPHORUS[1:] + [s["phosphate"] + s["org-phosphorus"]]
    assert len(temperatures) == max_temp_lag + 1 and len(phosphorus) == max_phosphorus_lag + 1
    # 4. no guard: wet and warm
    assert s["volume"] >= FA and T > 0
    # 5. denitrification
    tempfcn = 2 ** ((T - 20) / 10)
    assert not T < 5
    din = sum([s["ammonia"], s["nitrate"]])
    din_concentration = din / s["volume"]
    confcn = din_concentration / (din_concentration + halfsatINwater)
    denitri_water = denpar_w * area * tempfcn * confcn
    assert denitri_water < 0.5 * din
    din_before = din
    din_ = din - denitri_water
    gain = {n: 0.0 for n in BIO_POLLUTANTS}
    loss = {n: 0.0 for n in BIO_POLLUTANTS}
    for pol in ("ammonia", "nitrate"):
        lost = (din - din_) / din * s[pol]
        loss[pol] += lost
        s[pol] -= lost
    din = sum([s["ammonia"], s["nitrate"]])
    # 6. means and coefficients
    temp_10_day = sum(temperatures[-10:]) / 10
    temp_20_day = sum(temperatures[-20:]) / 20
    total_phos_365_day = sum(phosphorus) / max_phosphorus_lag
    tempfcn = T / 20 * (temp_10_day - temp_20_day) / 5
    assert total_phos_365_day - limpppar + hsatTP > 0
    totalphosfcn = (total_phos_365_day - limpppar) / (total_phos_365_day - limpppar + hsatTP)
    # 7. production
    minprodN = prodNpar * totalphosfcn * tempfcn * area * current_depth
    minprodP = prodPpar * totalphosfcn * tempfcn * area * current_depth * uptake_PNratio
    assert minprodN > 0 and minprodN < 0.5 * din and minprodP < 0.5 * s["phosphate"]
    s_phosphate, din_production = s["phosphate"], din
    loss["phosphate"] = minprodP
    s["phosphate"] -= minprodP
    gain["org-phosphorus"] = minprodP
    s["org-phosphorus"] += minprodP
    for pol in ("ammonia", "nitrate"):
        lost = minprodN * s[pol] / din
        loss[pol] += lost
        s[pol] -= lost
    gain["org-nitrogen"] = minprodN
    s["org-nitrogen"] += minprodN
    din = sum([s["ammonia"], s["nitrate"]])
    # 8. macrophytes
    tempfcn1 = (max(0, T) / 20) ** 0.3
    tempfcn2 = (T - temp_20_day) / 5
    tempfcn = max(0, tempfcn1 * tempfcn2)
    macrouptN = muptNpar * tempfcn * area
    assert 0 < macrouptN < 0.5 * din
    for pol in ("ammonia", "nitrate"):
        lost = macrouptN * s[pol] / din
        loss[pol] += lost
        s[pol] -= lost
    macrouptP = muptPpar * tempfcn * max(0, totalphosfcn) * area
    phosphate_before_macro = s["phosphate"]
    assert 0 < macrouptP < 0.5 * s["phosphate"]
    loss["phosphate"] += macrouptP
    s["phosphate"] -= macrouptP
    assert torch.equal(node.tank.storage, BIO.to_vqip(s))
    assert torch.equal(in_, BIO.to_vqip(gain))
    assert torch.equal(out_, BIO.to_vqip(loss))
    assert torch.equal(node.lagged_temperatures, torch.tensor(temperatures, dtype=torch.float64))
    assert torch.equal(node.lagged_total_phosphorus, torch.tensor(phosphorus, dtype=torch.float64))
    sites = [d.site.rsplit("/", 1)[-1] for d in ctx.decisions]
    margins = {d.site.rsplit("/", 1)[-1]: float(d.margin) for d in ctx.decisions}
    # the caps' operand orders (Python's min(a, b) is b when b < a): margin = b - a
    assert margins["denitrification_cap"] == 0.5 * din_before - denitri_water
    assert margins["production_cap:N"] == minprodN - 0.5 * din_production
    assert margins["production_cap:P"] == minprodP - 0.5 * s_phosphate
    assert margins["macrophyte_cap:N"] == macrouptN - 0.5 * din
    assert margins["macrophyte_cap:P"] == macrouptP - 0.5 * phosphate_before_macro
    assert sites == [
        "dry",
        "frozen",
        "cold",
        "denitrification_cap",
        "din:denitrification",
        "totalphosfcn",
        "production",
        "production_cap:N",
        "production_cap:P",
        "din:production",
        "mineralisation_cap:N",
        "mineralisation_cap:P",
        "din:mineralisation",
        "macrophyte_temperature",
        "macrophyte_factor",
        "macrophyte_cap:N",
        "din:macrophyte",
        "macrophyte_phosphorus",
        "macrophyte_cap:P",
    ]
    assert all(d.site.startswith(SITE) and d.call == 0 for d in ctx.decisions)
    # the mineralisation branch is evaluated under its own (here empty) mask
    masks = {d.site.rsplit("/", 1)[-1]: bool(d.active) for d in ctx.decisions}
    assert masks["production_cap:N"] and not masks["mineralisation_cap:N"]


def _random_bio_state(rng):
    """One random River state with lists of random lengths, across the guards and caps."""
    kind = rng.random()
    volume = 0.0 if kind < 0.03 else 5e-12 if kind < 0.06 else 10 ** rng.uniform(-3, 5)
    din_free = rng.random() < 0.1
    low_p = rng.random() < 0.3

    def mass(lo, hi):
        return 0.0 if rng.random() < 0.05 else 10 ** rng.uniform(lo, hi)

    state = {
        "volume": volume,
        "temperature": rng.uniform(-5, 30),
        "phosphate": mass(-6, -3) if low_p else mass(-5, 1),
        "org-phosphorus": mass(-6, -3) if low_p else mass(-5, 1),
        "ammonia": 0.0 if din_free else mass(-6, 2),
        "nitrate": 0.0 if din_free else mass(-6, 2),
        "org-nitrogen": mass(-6, 2),
    }
    temperatures = [rng.uniform(-5, 30) for _ in range(rng.randint(0, 21))]
    p_scale = (-6, -3.3) if low_p else (-5, 1)
    phosphorus = [10 ** rng.uniform(*p_scale) for _ in range(rng.randint(0, 366))]
    return state, rng.uniform(-5, 30), temperatures, phosphorus


def test_biochemical_processes_equals_wsimods_on_random_states():
    """500 random states in one batch against live WSIMOD's ``calculate_discharge``, one call
    each: the storage, ``bio_in``, ``bio_out`` and both lag lists, bitwise."""
    pytest.importorskip("wsimod")
    from datetime import datetime

    from wsimod.nodes.storage import River as WsimodRiver

    from noodl.apps.wsimod import constants
    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()  # WSIMOD's 12 pollutants, the six among them
    p = Pollutants(
        constants.POLLUTANTS, constants.ADDITIVE_POLLUTANTS, constants.NON_ADDITIVE_POLLUTANTS
    )
    rng = random.Random(20261112)
    cases = [_random_bio_state(rng) for _ in range(500)]
    n = len(cases)
    node = make_node(NodeSpec("r", "River", "River", {}), None, p)
    ctx = Ctx(p, FA, torch.Size((n,)), day=0, node_names=["r"], record_decisions=True)
    node.ctx = ctx
    node.tank.storage = torch.stack([p.to_vqip(c[0]) for c in cases])
    node.lagged_temperatures = torch.stack([pad_register(c[2], 21) for c in cases])
    node.lagged_total_phosphorus = torch.stack([pad_register(c[3], 366) for c in cases])
    node.bind({"temperature": torch.tensor([c[1] for c in cases], dtype=torch.float64)})
    node.calculate_discharge(active=torch.ones(n, dtype=torch.bool))
    t = datetime(2000, 1, 1)
    theirs = WsimodRiver(name="r", data_input_dict={})
    theirs.t = t
    for i, (state, t_air, temperatures, phosphorus) in enumerate(cases):
        theirs.tank.storage = {"volume": 0.0, **dict.fromkeys(p.names, 0.0), **state}
        theirs.lagged_temperatures = list(temperatures)
        theirs.lagged_total_phosphorus = list(phosphorus)
        theirs.data_input_dict = {("temperature", t): t_air}
        theirs.calculate_discharge()
        for ours, ref in (
            (node.tank.storage[i], theirs.tank.storage),
            (node.bio_in[i], theirs.bio_in),
            (node.bio_out[i], theirs.bio_out),
        ):
            assert torch.equal(ours, p.to_vqip(ref)), (i, p.to_dict(ours), ref)
        assert torch.equal(
            node.lagged_temperatures[i], pad_register(theirs.lagged_temperatures, 21)
        )
        assert torch.equal(
            node.lagged_total_phosphorus[i], pad_register(theirs.lagged_total_phosphorus, 366)
        )
    # the sample crosses every guard, branch and cap both ways (the macrophyte temperature is
    # T > 0, which the frozen guard already holds)
    outcomes: dict[str, set[bool]] = {}
    for d in ctx.decisions:
        branch = d.site.rsplit("/", 1)[-1]
        outcomes.setdefault(branch, set()).update(d.outcome[d.active].tolist())
    for branch, seen in outcomes.items():
        expected = {True} if branch == "macrophyte_temperature" else {True, False}
        assert seen == expected, branch
    assert len(outcomes) == 19


def test_the_registers_reproduce_pythons_sums_from_a_cold_start():
    """Over 400 calls from empty registers, the windows and the phosphorus mean equal Python's
    sums over WSIMOD's lists, bitwise; the lists then hold 21 and 366 entries. With
    ``max_temp_lag = 5`` both temperature windows are the whole 6-slot register."""
    rng = random.Random(5)
    for max_temp_lag, sizes in ((20, (21, 366)), (5, (6, 366))):
        node, _ = bio_river({"volume": 0.0}, 0.0, [], [], max_temp_lag=max_temp_lag)
        assert node.lag_sizes == sizes
        temperatures: list[float] = []
        phosphorus: list[float] = []
        for k in range(1, 401):
            state = {
                "volume": 0.0,  # dry: only steps 1-3 run
                "temperature": rng.uniform(-5, 30),
                "phosphate": 10 ** rng.uniform(-5, 0),
                "org-phosphorus": 10 ** rng.uniform(-5, 0),
            }
            node.tank.storage = BIO.to_vqip(state)
            node.bind({"temperature": torch.tensor(rng.uniform(-5, 30), dtype=torch.float64)})
            node.biochemical_processes(active=TRUE)
            for values, value, lag in (
                (temperatures, float(node.tank.storage[1]), max_temp_lag),
                (phosphorus, state["phosphate"] + state["org-phosphorus"], 365),
            ):
                if len(values) > lag:  # storage.py:753-763
                    del values[0]
                values.append(value)
            if k in (1, 10, 25, 400):
                t10, t20, tp = node.lagged_means()
                assert float(t10) == sum(temperatures[-10:]) / 10
                assert float(t20) == sum(temperatures[-20:]) / 20
                assert float(tp) == sum(phosphorus) / 365
                assert torch.equal(node.lagged_temperatures, pad_register(temperatures, sizes[0]))
        assert len(temperatures) == max_temp_lag + 1 and len(phosphorus) == 366


def test_compensated_sum_follows_the_interpreter():
    assert COMPENSATED_SUM == (sum([1e16, 1.0, -1e16]) == 1.0)


def test_minus_zero_phosphorus_enters_the_register_as_python_sums_it():
    """X31: a slot holding ``-0.0`` (``phosphate + org-phosphorus`` of two ``-0.0``) sums as
    CPython's ``sum``, which starts from the integer 0: ``+0.0``, whatever the padding."""
    state = {"volume": 0.0, "temperature": 10.0, "phosphate": -0.0, "org-phosphorus": -0.0}
    for before in ([], [-0.0] * 365):
        node, _ = bio_river(state, 10.0, [], before)
        node.biochemical_processes(active=TRUE)
        listed = [*before, -0.0 + -0.0]
        assert math.copysign(1.0, float(node.lagged_total_phosphorus[-1])) == -1.0
        python = sum(listed) / 365
        _, _, tp = node.lagged_means()
        assert math.copysign(1.0, float(tp)) == math.copysign(1.0, python) == 1.0
    assert float(window_sum(torch.tensor([-0.0, -0.0], dtype=torch.float64), 2)) == sum([-0.0])
    assert math.copysign(1.0, float(window_sum(torch.tensor([-0.0]), 1))) == 1.0


FALLING_TEMPERATURES = [20.0 - 0.5 * i for i in range(21)]  # T10 < T20: mineralising


def test_mineralisation_without_din_destroys_nitrogen_and_says_so():
    """X15 (``storage.py:856-873``): with ``din == 0`` the mineralised nitrogen leaves
    ``org-nitrogen``, nothing gains it, ``bio_out`` lists it and ``"river_nitrogen_lost"`` holds."""
    state = {
        "volume": 1000.0,
        "temperature": 10.0,
        "phosphate": 0.5,
        "org-phosphorus": 0.2,
        "org-nitrogen": 1.0,
    }
    node, ctx = bio_river(state, 10.0, FALLING_TEMPERATURES, [0.002] * 366, record=True)
    node.calculate_discharge(active=TRUE)
    margins = {d.site.rsplit("/", 1)[-1]: float(d.margin) for d in ctx.decisions}
    # min(0.5 pool, -minprodX): margin = -minprodX - 0.5 pool
    T = (1 - 1 / 5) * 10.0 + (1 / 5) * 10.0
    temperatures = FALLING_TEMPERATURES[1:] + [T]
    phosphorus = [0.002] * 365 + [0.5 + 0.2]
    t10, t20 = sum(temperatures[-10:]) / 10, sum(temperatures[-20:]) / 20
    tp = sum(phosphorus) / 365
    tempfcn = T / 20 * (t10 - t20) / 5
    totalphosfcn = (tp - 0.1 * 1e-3) / (tp - 0.1 * 1e-3 + 0.05 * 1e-3)
    area, depth = 200 * 20, 1000.0 / (200 * 20)
    minprodN = 0.001 * totalphosfcn * tempfcn * area * depth
    minprodP = 0.0001 * totalphosfcn * tempfcn * area * depth * (1 / 7.2)
    assert minprodN < 0
    assert margins["mineralisation_cap:N"] == -minprodN - 0.5 * 1.0
    assert margins["mineralisation_cap:P"] == -minprodP - 0.5 * 0.2
    lost = float(node.bio_out[BIO.column("org-nitrogen")])
    assert 0 < lost <= 0.5 * 1.0
    assert float(node.tank.storage[BIO.column("org-nitrogen")]) == 1.0 - lost
    for pol in ("ammonia", "nitrate"):
        assert float(node.tank.storage[BIO.column(pol)]) == 0.0
        assert float(node.bio_in[BIO.column(pol)]) == 0.0
    assert float(node.bio_in[BIO.column("org-nitrogen")]) == 0.0
    assert bool(ctx.events["river_nitrogen_lost"]["r"])
    # the lists close: the storage change is in minus out, column by column (additive)
    change = node.tank.storage - BIO.to_vqip({**state, "temperature": 10.0})
    assert torch.allclose(
        change[BIO.additive_mask], (node.bio_in - node.bio_out)[BIO.additive_mask], atol=1e-15
    )
    assert ctx.terms["River.bio_out"][BIO.column("org-nitrogen")] == lost


def test_minus_zero_minprodN_moves_nothing():
    """``totalphosfcn == 0`` (a cold-start phosphorus register below ``limpppar - hsatTP``) and
    ``tempfcn < 0`` give ``minprodN = 0.0 * (negative) = -0.0``: not ``> 0``, so the
    mineralisation branch runs, with ``min(0.5 org-N, +0.0) = +0.0``, and moves nothing."""
    state = {
        "volume": 1000.0,
        "temperature": 10.0,
        "phosphate": 0.001,
        "org-phosphorus": 0.001,
        "ammonia": 1.0,
        "nitrate": 1.0,
        "org-nitrogen": 1.0,
    }
    node, ctx = bio_river(state, 10.0, FALLING_TEMPERATURES, [], record=True)
    in_, out_ = node.biochemical_processes(active=TRUE)
    decisions = {d.site.rsplit("/", 1)[-1]: d for d in ctx.decisions}
    assert not bool(decisions["totalphosfcn"].outcome)
    production = decisions["production"]
    assert not bool(production.outcome) and math.copysign(1.0, float(production.margin)) == -1.0
    assert bool(decisions["mineralisation_cap:N"].active)
    for name in ("phosphate", "org-phosphorus", "org-nitrogen"):
        c = BIO.column(name)
        assert float(node.tank.storage[c]) == state[name]
        assert float(out_[c]) == 0.0
    assert torch.equal(in_, torch.zeros_like(in_))
    assert not bool(ctx.events["river_nitrogen_lost"]["r"])


def _bio_config(*, lists=None, overrides=None, days=2, missing=()):
    dates = [f"2000-01-{k + 1:02d}" for k in range(days)]
    data = {("temperature", d): 12.0 for k, d in enumerate(dates) if k not in missing}
    initial = {"volume": 500.0, "phosphate": 0.4, "ammonia": 2.0, "nitrate": 3.0}
    mapping = {
        **(BIO_LISTS if lists is None else lists),
        "dates": dates,
        "nodes": [
            {"name": "r", "type_": "River", "data_input_dict": data, "initial_storage": initial},
            {"name": "sink", "type_": "Waste"},
        ],
        "arcs": [{"name": "out", "type_": "Arc", "in_port": "r", "out_port": "sink"}],
    }
    if overrides:
        mapping["overrides"] = {"nodes": {"r": {"type_": "River", "name": "r", **overrides}}}
    return mapping


def _without(pollutant):
    names = [n for n in BIO.names if n != pollutant]
    return {
        "pollutants": names,
        "additive_pollutants": [n for n in BIO.additive if n != pollutant],
        "non_additive_pollutants": [n for n in BIO.non_additive if n != pollutant],
    }


@pytest.mark.parametrize(
    ("config", "driver", "match"),
    [
        *[
            (dict(lists=_without(n)), None, f"biochemistry.*reads '{n}'")
            for n in BIO_POLLUTANTS
            if n != "nitrate"
        ],
        (dict(missing=(1,)), None, "'temperature' forcing.*2000-01-02"),
        (dict(overrides={"T_wdays": 0}), {"T_wdays": 0.0}, "T_wdays is 0"),
        (dict(overrides={"length": 0}), {"length": 0.0}, r"area = length \* width is 0"),
        (
            dict(overrides={"halfsatINwater": 0}),
            {"halfsatINwater": 0.0},
            "halfsatINwater.*not positive",
        ),
        (
            dict(overrides={"halfsatINwater": -1e-3}),
            {"halfsatINwater": -1e-3},
            "halfsatINwater.*not positive",
        ),
        (
            dict(overrides={"max_temp_lag": -1}),
            None,
            "max_temp_lag -1 is not a non-negative integer",
        ),
        (dict(overrides={"max_temp_lag": 20.5}), None, "max_temp_lag 20.5 is not a non-negative"),
        (dict(overrides={"max_phosphorus_lag": 0}), None, "max_phosphorus_lag 0 is not a positive"),
        (dict(overrides={"max_phosphorus_lag": 365.5}), None, "max_phosphorus_lag 365.5 is not"),
    ],
)
def test_river_biochemistry_refusals(config, driver, match):
    """X19: what WSIMOD's biochemistry cannot run, refused by name at build (in either mode,
    since WSIMOD runs it whatever noodl's mode) and, for a driver, at ``bind``."""
    network = read_wsimod_config(_bio_config(**config))
    for pollutants in (True, False):
        with pytest.raises(ValueError, match=match):
            build_model(network, pollutants=pollutants)
    if driver is not None:
        node = make_node(read_wsimod_config(_bio_config()).nodes[0], None, BIO)
        with pytest.raises(ValueError, match=match):
            node.bind({k: torch.tensor([1.0, v], dtype=torch.float64) for k, v in driver.items()})


def test_a_river_without_nitrate_runs_no_biochemistry():
    """``storage.py:919``: no ``nitrate``, no biochemistry, no drivers or registers; the
    biochemistry refusals do not apply (WSIMOD never divides)."""
    network = read_wsimod_config(
        _bio_config(lists=_without("nitrate"), overrides={"T_wdays": 0}, missing=(0, 1))
    )
    model, state, _ = build_model(network)
    day_ = model.wsimod_day
    assert not day_.engine.nodes["r"].biochemistry
    assert "temperature" not in forcing_readers(network)
    assert not any(
        k.startswith("wsimod.River.") and k.split(".")[2] in BIO_PARAMETERS for k in day_.drivers
    )
    assert state["wsimod.river_temperature_lag"].shape == (0,)


BIO_OVERRIDES = {
    "uptake_PNratio": 0.125,
    "bulk_density": 1250,
    "denpar_w": 0.002,
    "T_wdays": 4,
    "halfsatINwater": 2e-3,
    "hsatTP": 6e-5,
    "limpppar": 1.2e-4,
    "prodNpar": 0.002,
    "prodPpar": 0.0002,
    "muptNpar": 0.0015,
    "muptPpar": 0.00015,
    "max_temp_lag": 15,
    "max_phosphorus_lag": 300,
}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_river_overrides_set_the_thirteen_attributes():
    """``River.apply_overrides`` (``storage.py:620-644``) sets 13 attributes; each equals
    WSIMOD's after ``add_overrides``. ``bulk_density`` is never read (``model.unread``, D10)."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_model

    mapping = _bio_config(overrides=BIO_OVERRIDES)
    theirs = wsimod_model(mapping).nodes["r"]
    model, _, drivers = build_model(read_wsimod_config(mapping))
    ours = model.wsimod_day.engine.nodes["r"]
    for name in BIO_PARAMETERS:
        assert float(ours.defaults[name]) == getattr(theirs, name), name
        assert float(drivers[f"wsimod.River.{name}"][0]) == getattr(theirs, name), name
    assert ours.max_temp_lag == theirs.max_temp_lag == 15
    assert ours.max_phosphorus_lag == theirs.max_phosphorus_lag == 300
    assert ours.lag_sizes == (16, 301)
    assert theirs.bulk_density == 1250
    assert model.unread == ["r.bulk_density"]
    # and the defaults are WSIMOD's own attributes (storage.py:539-564)
    plain = wsimod_model(_bio_config()).nodes["r"]
    fresh = build_model(read_wsimod_config(_bio_config()))[0].wsimod_day.engine.nodes["r"]
    for name in BIO_PARAMETERS:
        assert float(fresh.defaults[name]) == getattr(plain, name), name
    assert (fresh.max_temp_lag, fresh.max_phosphorus_lag) == (plain.max_temp_lag, 365)


def test_river_biochemistry_inputs_exist_only_when_it_runs():
    """D9: the ``"wsimod.River.<p>"`` drivers and both registers exist with biochemistry; a
    nitrate River built quantity-only keeps its ``"wsimod.temperature"`` row (WSIMOD reads it
    whatever the mode) and lists its configured parameters as ignored."""
    network = read_wsimod_config(_bio_config(overrides={"denpar_w": 0.002}))
    for pollutants in (False, True):
        model, state, drivers = build_model(network, pollutants=pollutants)
        day_ = model.wsimod_day
        assert day_.drivers["wsimod.temperature"].labels == ["r"]
        assert "wsimod.temperature" in drivers
        params = {f"wsimod.River.{p}" for p in BIO_PARAMETERS}
        temperature = day_.state_labels("wsimod.river_temperature_lag")
        phosphorus = day_.state_labels("wsimod.river_phosphorus_lag")
        if not pollutants:
            assert not params & set(day_.drivers)
            assert temperature == [] and phosphorus == []
            assert state["wsimod.river_temperature_lag"].shape == (0,)
            assert "r.denpar_w" in model.ignored_by_quantity_only
        else:
            assert params <= set(day_.drivers)
            assert temperature == [("r", i) for i in range(21)]
            assert phosphorus == [("r", i) for i in range(366)]
            assert state["wsimod.river_phosphorus_lag"].shape == (366,)
            assert day_.input_specs["wsimod.River.denpar_w"].unit == "kg/m2/d"
            assert day_.input_specs["wsimod.River.halfsatINwater"].unit == "kg/m3"
            assert day_.key_labels["wsimod.river_temperature_lag"].unit == "degC"
            assert day_.key_labels["wsimod.river_phosphorus_lag"].unit == "kg"
            assert float(drivers["wsimod.River.denpar_w"][0]) == 0.002


def test_a_biochemistry_river_carries_its_registers_and_closes_its_balance():
    """Three days of a River with biochemistry: each day appends to the registers carried in
    the state, ``"River.bio_in"``/``"River.bio_out"`` close the balance, and a warm start from
    the dumped registers continues them."""
    network = read_wsimod_config(_bio_config(days=3))
    model, state, drivers = build_model(network)
    result = wsimod_run(model, state, drivers)
    assert float(result.mass_balance.residual.abs().max()) < 1e-12
    assert "River.bio_out" in result.mass_balance.terms
    final = model.wsimod_day.last_state
    temperatures = final["wsimod.river_temperature_lag"]
    assert torch.equal(temperatures[:-3], torch.zeros(18, dtype=torch.float64))
    assert bool((temperatures[-3:] != 0).all())
    phosphorus = final["wsimod.river_phosphorus_lag"]
    assert phosphorus[-3] == 0.4  # the first call's phosphate, before any process
    assert torch.equal(phosphorus[:-3], torch.zeros(363, dtype=torch.float64))


def test_gradcheck_on_one_biochemistry_call():
    """``torch.autograd.gradcheck`` through one call at the hand-worked point, away from every
    guard and cap: the storage's pollutants, the air temperature and five parameters."""

    def f(pollutants, t_air, denpar_w, halfsat, prod_n, mupt_n, mupt_p):
        state = {k: v for k, v in HAND_STATE.items() if k == "volume"}
        node, _ = bio_river(state, 17.0, HAND_TEMPERATURES, HAND_PHOSPHORUS)
        node.tank.storage = torch.cat([torch.tensor([1000.0], dtype=torch.float64), pollutants])
        node.bind(
            {
                "temperature": t_air,
                "denpar_w": denpar_w,
                "halfsatINwater": halfsat,
                "prodNpar": prod_n,
                "muptNpar": mupt_n,
                "muptPpar": mupt_p,
            }
        )
        in_, out_ = node.biochemical_processes(active=TRUE)
        return torch.cat([node.tank.storage, in_, out_])

    def leaf(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    pollutants = leaf([HAND_STATE[n] for n in BIO_POLLUTANTS])
    inputs = (
        pollutants,
        leaf(17.0),
        leaf(0.0015),
        leaf(1.5e-3),
        leaf(0.001),
        leaf(0.001),
        leaf(1e-4),
    )
    assert torch.autograd.gradcheck(f, inputs, eps=1e-7, atol=1e-7, rtol=1e-6)
