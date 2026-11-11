"""``Demand``, ``NonResidentialDemand``, ``ResidentialDemand``, ``Distribution`` and
``UnlimitedDistribution`` (noodl only).

The values asserted are WSIMOD's, worked out from ``demand.py`` and ``distribution.py`` by hand;
the replay (``tests/verification/test_wsimod_replay.py``) checks the same code against live
WSIMOD on the ``distribution_leakage``, ``unlimited_distribution``, ``demand_constant_and_loads``,
``residential_demand`` and ``reservoir_spill`` cases.
"""

from __future__ import annotations

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.demand import (
    Demand,
    Distribution,
    NonResidentialDemand,
    ResidentialDemand,
    UnlimitedDistribution,
)
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.node import NODE_CLASSES
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from tests.apps.wsimod._replay import ScriptedArc, StoreNode
from tests.verification._wsimod_reference import ArcCall

P = QUANTITY_ONLY
PP = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)


def make(cls: str, name: str = "n", pollutants: Pollutants = P, **parameters):
    return make_node(NodeSpec(name, cls, cls, parameters), None, pollutants)


_NEW_ARCS: list = []  # the arcs built since the last day(); day() gives them the context


def day(*objects, shape=(), pollutants: Pollutants = P):
    ctx = Ctx(pollutants, FA, torch.Size(shape), day=0, node_names=[], record_decisions=True)
    for obj in (*objects, *_NEW_ARCS):
        obj.ctx = ctx
    _NEW_ARCS.clear()
    return ctx


def vol(v):
    return P.to_vqip({"volume": v})


def arc(name, a, b, *, capacity=UNBOUNDED_CAPACITY):
    spec = ArcSpec(name, "Arc", a.name, b.name, capacity=capacity)
    _NEW_ARCS.append(ARC_CLASSES["Arc"](spec, a, b, P))
    return _NEW_ARCS[-1]


def call(name, method, request, reply, *, tag="default", pollutants=None) -> ArcCall:
    """A scripted arc call of WSIMOD's; ``request`` and ``reply`` are volumes or VQIP dicts."""
    request = {"volume": request} if not isinstance(request, dict) else request
    reply = {"volume": reply} if not isinstance(reply, dict) else reply
    return ArcCall(
        name, method, request, tag, False, reply, 0.0, {"volume": 0.0}, 0.0, {"volume": 0.0}
    )


def scripted(name, calls, *, in_type="Node", out_type="Node", pollutants: Pollutants = P):
    return ScriptedArc(
        ArcSpec(name, "Arc", "n", "other"),
        calls,
        pollutants,
        scale=1.0,
        in_type=in_type,
        out_type=out_type,
    )


def test_the_classes_are_registered_under_the_names_wsimod_gives_them():
    for name, cls in (
        ("Demand", Demand),
        ("NonResidentialDemand", NonResidentialDemand),
        ("ResidentialDemand", ResidentialDemand),
        ("Distribution", Distribution),
        ("UnlimitedDistribution", UnlimitedDistribution),
    ):
        assert NODE_CLASSES[name] is cls
    # demand.py:214: a ResidentialDemand renames its class Demand, which other nodes' of_type
    # filters see; NonResidentialDemand keeps its name
    assert Demand.wsimod_type == ResidentialDemand.wsimod_type == "Demand"
    assert NonResidentialDemand.wsimod_type == "NonResidentialDemand"


# --- Demand -----------------------------------------------------------------------------------
def test_non_residential_demand_is_its_constant_demand_as_house():
    """W4: WSIMOD's ``get_demand`` recurses for ever; the evident intent is the constant demand
    as ``house``, pushed to a Sewer with the tag ``"Demand"``."""
    node = make("NonResidentialDemand", constant_demand=5.0)
    sewer = scripted(
        "to_sewer",
        [call("to_sewer", "send_push_request", 5.0, 0.0, tag="Demand")],
        out_type="Sewer",
    )
    node.connect(sewer, "out")
    ctx = day(node)
    node.create_demand(active=TRUE)
    sewer.assert_exhausted()
    assert float(node.total_demand[0]) == 5.0 and float(node.total_backup[0]) == 0.0
    # the demand was pulled from nowhere: a shortfall of 5 (demand.py:90)
    assert bool(ctx.events["demand_deficit"]["n"])
    assert "demand_not_able_to_push" not in ctx.events or not bool(
        ctx.events["demand_not_able_to_push"]["n"]
    )


def test_a_demand_pulls_its_total_then_pushes_the_default_component_and_keeps_the_rest():
    """``demand.py:79-119``: the pull is of the demand's volume only; what the arc refuses is
    ``total_backup`` and an event."""
    node = make("Demand", constant_demand=4.0)
    source = scripted("from_well", [call("from_well", "send_pull_request", 4.0, 4.0)])
    sink = scripted("to_sea", [call("to_sea", "send_push_request", 4.0, 1.5)], out_type="Waste")
    node.connect(source, "in")
    node.connect(sink, "out")
    ctx = day(node)
    node.create_demand(active=TRUE)
    source.assert_exhausted()
    sink.assert_exhausted()
    assert float(node.total_received[0]) == 4.0
    assert float(node.total_backup[0]) == 1.5
    assert float(node.total_demand[0]) == 4.0
    assert not bool(ctx.events["demand_deficit"]["n"])
    assert bool(ctx.events["demand_not_able_to_push"]["n"])
    # the totals are the mass balance's (D11) and the day's: end_timestep clears them
    ctx.terms.clear()
    node.record_terms()
    assert float(ctx.terms["Demand.total_backup"][0]) == 1.5
    assert set(Demand.mass_balance) == set(ctx.terms)
    node.end_timestep()
    assert float(node.total_demand[0]) == 0.0 and float(node.total_backup[0]) == 0.0


def test_a_shortfall_above_float_accuracy_is_a_deficit_event():
    for received, expected in ((4.0, False), (4.0 - 1e-12, False), (3.0, True)):
        node = make("Demand", constant_demand=4.0)
        source = scripted("from_well", [call("from_well", "send_pull_request", 4.0, received)])
        node.connect(source, "in")
        ctx = day(node)
        node.create_demand(active=TRUE)
        assert bool(ctx.events["demand_deficit"]["n"]) == expected


def test_a_demand_pushes_the_load_as_given_not_scaled_by_its_volume():
    """``demand.py:121-131``: ``pollutant_load`` entries are set as given."""
    node = make(
        "Demand",
        pollutants=PP,
        constant_demand=4.0,
        pollutant_load={"phosphate": 0.3, "temperature": 15.0},
    )
    sea = scripted(
        "to_sea",
        [
            call(
                "to_sea",
                "send_push_request",
                {"volume": 4.0, "phosphate": 0.3, "temperature": 15.0},
                {"volume": 0.0, "phosphate": 0.0, "temperature": 0.0},
            )
        ],
        out_type="Waste",
        pollutants=PP,
    )
    node.connect(sea, "out")
    day(node, pollutants=PP)
    node.create_demand(active=TRUE)
    sea.assert_exhausted()


def test_an_inactive_demand_changes_nothing():
    node = make("Demand", constant_demand=4.0)
    sea = scripted("to_sea", [])  # no call is made
    node.connect(sea, "out")
    day(node)
    node.create_demand(active=torch.tensor(False))
    assert float(node.total_demand[0]) == 0.0 and float(node.total_received[0]) == 0.0


# --- ResidentialDemand --------------------------------------------------------------------------
def test_residential_house_demand_and_mixed_temperature():
    """``demand.py:303-324``: ``population x per_capita`` of water, the additive load times the
    population, and the temperature ``T_air (1 - w) + T_const w``."""
    node = make(
        "ResidentialDemand",
        pollutants=PP,
        population=100.0,
        per_capita=0.15,
        pollutant_load={"phosphate": 0.01},
        constant_temp=14.0,
        constant_weighting=0.2,
    )
    node.bind({"temperature": torch.tensor(10.0)})
    house = node.get_house_demand()
    assert float(house[0]) == 100 * 0.15
    assert float(house[1]) == 0.01 * 100
    assert float(house[2]) == 10 * (1 - 0.2) + 14 * 0.2  # 10.8: bitwise, not approximately


def test_residential_house_demand_without_temperature_as_a_pollutant_drops_the_write():
    p = Pollutants(("phosphate",), ("phosphate",), ())
    node = make(
        "ResidentialDemand",
        pollutants=p,
        population=100.0,
        per_capita=0.15,
        pollutant_load={"phosphate": 0.01},
    )
    node.bind({"temperature": torch.tensor(10.0)})
    house = node.get_house_demand()
    assert house.tolist() == [100 * 0.15, 0.01 * 100]


def test_a_residential_demand_pushes_foul_water_to_a_sewer_and_garden_demand_is_zero():
    node = make(
        "ResidentialDemand",
        pollutants=PP,
        population=100.0,
        per_capita=0.15,
        pollutant_load={"phosphate": 0.01},
        constant_temp=14.0,
        constant_weighting=0.2,
    )
    node.bind({"temperature": torch.tensor(10.0)})
    mixed = 10 * (1 - 0.2) + 14 * 0.2
    foul = {"volume": 100 * 0.15, "phosphate": 0.01 * 100, "temperature": mixed}
    sewer = scripted(
        "to_sewer",
        [
            call(
                "to_sewer",
                "send_push_request",
                foul,
                {"volume": 0.0, "phosphate": 0.0, "temperature": 0.0},
                tag="Demand",
            )
        ],
        out_type="Sewer",
        pollutants=PP,
    )
    node.connect(sewer, "out")
    day(node, pollutants=PP)
    node.create_demand(active=TRUE)
    sewer.assert_exhausted()  # the garden component is pushed to Land only: no such arc
    assert node.total_demand.tolist() == [100 * 0.15, 0.01 * 100, mixed]  # garden adds 0
    assert node.total_backup[0] == 0.0


def test_garden_demand_is_the_land_capacity_times_the_gardening_efficiency():
    """``demand.py:250-301``: the push check with the tag ``("Demand", "Garden")`` over the
    ``Land`` arcs, times ``gardening_efficiency`` (the default 0.6 x 0.7)."""
    node = make("ResidentialDemand", gardening_efficiency=0.5)
    land = StoreNode("land", P, wsimod_type="Land")
    land.push_check_handler[("Demand", "Garden")] = lambda vqip=None, *, active: vol(8.0)
    arc("to_land", node, land)
    day(node, land)
    node.bind({"temperature": torch.tensor(10.0)})
    assert float(node.get_garden_demand(active=TRUE)[0]) == 8.0 * 0.5
    assert float(make("ResidentialDemand").defaults["constant_demand"]) == 0.0
    assert ResidentialDemand.parameter_defaults["gardening_efficiency"] == 0.6 * 0.7


def test_a_residential_demand_is_batched_over_its_parameters():
    """Two instances, two populations: each equals its own single run."""
    population = torch.tensor([100.0, 40.0], dtype=torch.float64)
    temperature = torch.tensor([10.0, 3.0], dtype=torch.float64)
    node = make("ResidentialDemand", pollutants=PP, pollutant_load={"phosphate": 0.01})
    node.bind({"population": population, "temperature": temperature})
    batch = node.get_house_demand()
    for i in range(2):
        one = make("ResidentialDemand", pollutants=PP, pollutant_load={"phosphate": 0.01})
        one.bind({"population": population[i], "temperature": temperature[i]})
        assert batch[i].tolist() == one.get_house_demand().tolist()


# --- Distribution -----------------------------------------------------------------------------
def _leaking_distribution(leakage, leak_call=None, *, gw=True):
    """A ``Distribution`` pulled through an upstream scripted arc (the request must be the
    inflated 12.5), leaking into a real ``Groundwater`` (or a scripted arc)."""
    dist = make("Distribution", "dist", leakage=leakage)
    up = scripted("up", [call("up", "send_pull_request", 12.5, 12.5)])
    dist.connect(up, "in")
    groundwater = None
    if gw:
        groundwater = make("Groundwater", "gw", capacity=100.0)
        arc("leak", dist, groundwater)
    else:
        leak = scripted("leak", [leak_call], out_type="Groundwater")
        dist.connect(leak, "out")
    return dist, up, groundwater


def test_an_overridden_leakage_is_applied_once():
    """W17: WSIMOD decorates again after ``apply_overrides`` (``distribution.py:152``), so a
    request of 10 at leakage 0.2 asks upstream for 15.625 = 10 / 0.8^2. noodl asks for 12.5,
    leaks 2.5 and answers 10."""
    dist, up, gw = _leaking_distribution(0.2)
    day(dist, gw)
    reply = dist.pull_set(vol(10.0), active=TRUE)
    up.assert_exhausted()  # the scripted request was exactly 10 / 0.8
    assert float(reply[0]) == 10.0
    assert float(gw.tank.storage[0]) == 2.5


def test_the_pull_check_is_the_check_of_the_inflated_request_less_the_leak():
    dist = make("Distribution", "dist", leakage=0.2)
    up = scripted("up", [call("up", "send_pull_check", 0.0, 50.0)])
    up.calls[0].vqip = None  # the arcs are polled without a request (nodes.py:480-508)
    dist.connect(up, "in")
    day(dist)
    reply = dist.pull_check(vol(10.0), active=TRUE)
    up.assert_exhausted()
    assert float(reply[0]) == 10.0  # min(50, 10 / 0.8) = 12.5, less the leak 2.5
    # with no request the unclipped check is reduced by the leak (distribution.py:90-98)
    dist = make("Distribution", "dist", leakage=0.2)
    up = scripted("up", [call("up", "send_pull_check", 0.0, 50.0)])
    up.calls[0].vqip = None
    dist.connect(up, "in")
    day(dist)
    assert float(dist.pull_check(None, active=TRUE)[0]) == 50.0 - 50.0 * 0.2


def test_leakage_that_cannot_leak_returns_to_the_caller():
    """``distribution.py:40-50``: the leak no groundwater takes is added to the reply, which is
    then more than was requested (12.5 for 10)."""
    leak = call("leak", "send_push_request", 2.5, 2.5)
    dist, up, _ = _leaking_distribution(0.2, leak, gw=False)
    ctx = day(dist)
    reply = dist.pull_set(vol(10.0), active=TRUE)
    up.assert_exhausted()
    assert float(reply[0]) == 12.5
    assert bool(ctx.events["leakage_not_going_to_gw"]["dist"])


def test_leakage_with_no_groundwater_arc_returns_the_whole_leak():
    dist = make("Distribution", "dist", leakage=0.2)
    up = scripted("up", [call("up", "send_pull_request", 12.5, 12.5)])
    dist.connect(up, "in")
    ctx = day(dist)
    assert float(dist.pull_set(vol(10.0), active=TRUE)[0]) == 12.5
    assert bool(ctx.events["leakage_not_going_to_gw"]["dist"])


def test_an_undecorated_distribution_conveys_and_denies_pushes():
    dist = make("Distribution", "dist")
    up = scripted("up", [call("up", "send_pull_request", 10.0, 7.0)])
    dist.connect(up, "in")
    day(dist)
    assert not dist.decorated
    assert float(dist.pull_set(vol(10.0), active=TRUE)[0]) == 7.0
    assert dist.push_set(vol(3.0), active=TRUE).tolist() == [3.0]
    assert dist.push_check(vol(3.0), active=TRUE).tolist() == [0.0]


def test_the_leak_gradient_flows_to_leakage():
    """The groundwater gets ``R / (1 - l) x l``; its slope in ``l`` is ``R / (1 - l)^2``."""
    dist = make("Distribution", "dist", leakage=0.2)
    upstream = StoreNode("well", P, storage=100.0)
    groundwater = make("Groundwater", "gw", capacity=100.0)
    arc("up", upstream, dist)
    arc("leak", dist, groundwater)
    day(upstream, dist, groundwater)
    leakage = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    dist.bind({"leakage": leakage})
    dist.pull_set(vol(10.0), active=TRUE)
    groundwater.tank.storage[0].backward()
    assert leakage.grad.item() == pytest.approx(10.0 / 0.8**2, rel=1e-12)


def test_a_leakage_batch_equals_its_single_runs():
    values = [0.1, 0.2, 0.35]

    def run(leakage, shape):
        dist = make("Distribution", "dist", leakage=0.2)
        upstream = StoreNode("well", P, storage=100.0)
        groundwater = make("Groundwater", "gw", capacity=100.0)
        arc("up", upstream, dist)
        arc("leak", dist, groundwater)
        day(upstream, dist, groundwater, shape=shape)
        dist.bind({"leakage": torch.tensor(leakage, dtype=torch.float64)})
        reply = dist.pull_set(vol(10.0), active=torch.ones(shape, dtype=torch.bool))
        return reply[..., 0], groundwater.tank.storage[..., 0]

    reply, stored = run(values, (3,))
    for i, leakage in enumerate(values):
        one_reply, one_stored = run(leakage, ())
        assert reply[i] == one_reply and stored[i] == one_stored


@pytest.mark.parametrize(
    ("configured", "batch", "match"),
    [
        (0.2, [0.1, 0.0], "mixes zero and non-zero"),  # decorated: zero is a structure change
        (0.2, [0.0, 0.0], "mixes zero and non-zero"),
        (0.0, [0.1, 0.0], "non-zero on a node configured without leakage"),
        (0.0, 0.2, "non-zero on a node configured without leakage"),
        (0.2, [0.1, 1.0], r"\[0, 1\)"),
        (0.2, -0.1, r"\[0, 1\)"),
    ],
)
def test_a_leakage_batch_that_changes_the_structure_is_refused(configured, batch, match):
    dist = make("Distribution", "dist", leakage=configured)
    with pytest.raises(ValueError, match=rf"node 'dist': leakage.*{match}|{match}") as info:
        dist.bind({"leakage": torch.tensor(batch, dtype=torch.float64)})
    assert "dist" in str(info.value) and "leakage" in str(info.value)


def test_a_valid_leakage_batch_and_the_configured_value_bind():
    dist = make("Distribution", "dist", leakage=0.2)
    dist.bind({"leakage": torch.tensor([0.1, 0.5], dtype=torch.float64)})
    dist.bind({})
    assert float(dist._leakage()) == 0.2
    make("Distribution", "plain").bind({"leakage": torch.zeros(3, dtype=torch.float64)})


# --- UnlimitedDistribution --------------------------------------------------------------------
def test_an_unlimited_distribution_supplies_clean_water_of_the_requested_volume():
    """``distribution.py:184-196``: ``supplied`` accumulates; the check is ``1e15`` whatever the
    request (``:174-176``); both reset at ``end_timestep``."""
    node = make("UnlimitedDistribution", pollutants=PP)
    ctx = day(node, pollutants=PP)
    request = PP.to_vqip({"volume": 3.0, "phosphate": 9.0, "temperature": 12.0})
    reply = node.pull_set(request, active=TRUE)
    assert reply.tolist() == [3.0, 0.0, 0.0]
    node.pull_set(request, active=torch.tensor(False))  # not counted
    node.pull_set(request, active=TRUE)
    assert node.supplied.tolist() == [6.0, 0.0, 0.0]
    assert node.pull_check(request, active=TRUE).tolist() == [UNBOUNDED_CAPACITY, 0.0, 0.0]
    assert node.pull_check(None, active=TRUE).tolist() == [UNBOUNDED_CAPACITY, 0.0, 0.0]
    assert node.push_set(request, active=TRUE).tolist() == request.tolist()
    node.record_terms()
    assert ctx.terms["UnlimitedDistribution.supplied"].tolist() == [6.0, 0.0, 0.0]
    node.end_timestep()
    assert node.supplied.tolist() == [0.0, 0.0, 0.0]


# --- validation -------------------------------------------------------------------------------
def _network(nodes, *, pollutants=("phosphate", "temperature")):
    return read_wsimod_config(
        {
            "pollutants": list(pollutants),
            "additive_pollutants": [x for x in pollutants if x != "temperature"],
            "non_additive_pollutants": [x for x in pollutants if x == "temperature"],
            "dates": ["2000-01-01", "2000-01-02"],
            "nodes": nodes,
            "arcs": [],
        }
    )


def _validate(cls, *, pollutants=True, **parameters):
    nodes = [{"name": "n", "type_": cls}]
    network = _network(nodes)
    spec = network.nodes[0]
    spec.parameters.update(parameters)
    return NODE_CLASSES[cls].validate(spec, network, pollutants=pollutants)


def test_leakage_refusals():
    with pytest.raises(ValueError, match="leakage 1.0 is not in"):
        _validate("Distribution", leakage=1.0)
    with pytest.raises(ValueError, match="leakage -0.1 is not in"):
        _validate("Distribution", leakage=-0.1)
    with pytest.raises(ValueError, match=r"node 'n': leakage > 0 on an UnlimitedDistribution"):
        _validate("UnlimitedDistribution", leakage=0.1)
    assert _validate("Distribution", leakage=0.99) == []
    assert _validate("UnlimitedDistribution", leakage=0) == []


def test_a_leaking_distribution_with_two_zero_preference_groundwater_arcs_is_refused():
    nodes = [
        {"name": "d", "type_": "Distribution", "leakage": 0.1},
        {"name": "a", "type_": "Groundwater"},
        {"name": "b", "type_": "Groundwater"},
    ]
    arcs = [
        {"name": "da", "type_": "Arc", "in_port": "d", "out_port": "a", "preference": 0},
        {"name": "db", "type_": "Arc", "in_port": "d", "out_port": "b", "preference": 0},
    ]
    config = {
        "pollutants": [],
        "additive_pollutants": [],
        "non_additive_pollutants": [],
        "dates": ["2000-01-01"],
        "nodes": nodes,
        "arcs": arcs,
    }
    network = read_wsimod_config(config)
    with pytest.raises(ValueError, match="preference 0"):
        Distribution.validate(network.nodes[0], network, pollutants=True)
    nodes[0]["leakage"] = 0.0  # undecorated: the arcs are never used
    network = read_wsimod_config(config)
    assert Distribution.validate(network.nodes[0], network, pollutants=True) == []


def test_a_residential_pollutant_load_lacking_a_pollutant_is_refused():
    base = {"type_": "ResidentialDemand", "name": "n"}
    forcing = {("temperature", "2000-01-01"): 5.0, ("temperature", "2000-01-02"): 6.0}
    network = _network(
        [{**base, "pollutant_load": {"phosphate": 0.01}, "data_input_dict": forcing}]
    )
    spec = network.nodes[0]
    assert ResidentialDemand.validate(spec, network, pollutants=True) == []
    spec.parameters["pollutant_load"] = {}
    with pytest.raises(ValueError, match=r"pollutant_load has no 'phosphate'"):
        ResidentialDemand.validate(spec, network, pollutants=True)
    # a quantity-only run does not read the load: it is reported as ignored, not refused
    assert ResidentialDemand.validate(spec, network, pollutants=False) == []
    spec.parameters["pollutant_load"] = {"ammonia": 1.0}
    with pytest.raises(ValueError, match=r"pollutant_load has no 'phosphate'"):
        ResidentialDemand.validate(spec, network, pollutants=True)
    assert ResidentialDemand.validate(spec, network, pollutants=False) == ["pollutant_load"]


def test_a_residential_demand_needs_its_temperature_forcing_on_every_date():
    network = _network(
        [
            {
                "name": "n",
                "type_": "ResidentialDemand",
                "pollutant_load": {"phosphate": 0.01},
                "data_input_dict": {("temperature", "2000-01-01"): 5.0},
            }
        ]
    )
    with pytest.raises(ValueError, match=r"node 'n' has no 'temperature' forcing on 2000-01-02"):
        ResidentialDemand.validate(network.nodes[0], network, pollutants=True)


def test_demand_validate_reports_the_load_a_quantity_only_run_ignores():
    assert _validate("Demand", pollutant_load={"phosphate": 1.0}, pollutants=False) == [
        "pollutant_load"
    ]
    assert _validate("Demand", pollutant_load={"phosphate": 1.0}) == []
    assert _validate("NonResidentialDemand", constant_demand=3.0) == []


def test_no_demand_or_distribution_handler_recurses_except_the_distribution_pull_check():
    spec = NodeSpec("n", "Demand", "Demand", {})
    for cls in (Demand, NonResidentialDemand, ResidentialDemand, UnlimitedDistribution):
        assert not cls.check_recurses(spec, "pull", "default")
    assert Distribution.check_recurses(spec, "pull", "default")
    assert not Distribution.check_recurses(spec, "push", "default")
