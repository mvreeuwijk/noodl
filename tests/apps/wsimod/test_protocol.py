"""WSIMOD's pull/push protocol in noodl: arcs, Node, Waste and Catchment (noodl only).

The values asserted are WSIMOD's, worked out from ``nodes.py``/``arcs.py`` by hand; the replay
(``tests/verification/test_wsimod_replay.py``) checks the same code against live WSIMOD.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES
from noodl.apps.wsimod.catchment import Catchment
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, WsimodNetwork, read_wsimod_config
from noodl.apps.wsimod.engine import Ctx, Engine, make_node
from noodl.apps.wsimod.node import Node, Waste
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from tests.apps.wsimod._replay import StoreNode

P = QUANTITY_ONLY
PQ = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)


def node(name: str, cls=Node, p=P):
    return cls(NodeSpec(name, cls.__name__, cls.__name__, {}), p)


def arc(name, a, b, *, capacity=1e15, preference=1.0, cls="Arc", p=P):
    spec = ArcSpec(name, cls, a.name, b.name, capacity=capacity, preference=preference)
    return ARC_CLASSES[cls](spec, a, b, p)


def day(*objects, shape=(), p=P, record=False):
    ctx = Ctx(p, FA, torch.Size(shape), day=0, node_names=[], record_decisions=record)
    for obj in objects:
        obj.ctx = ctx
    return ctx


def vol(v, p=P):
    return p.to_vqip({"volume": v})


def three_wastes(volumes_capacities=(2.0, 3.0, 1.0), preferences=(1.0, 2.0, 3.0)):
    junction = node("junction")
    wastes = [node(f"w{i}", Waste) for i in range(3)]
    arcs = [
        arc(f"to_w{i}", junction, w, capacity=c, preference=pref)
        for i, (w, c, pref) in enumerate(zip(wastes, volumes_capacities, preferences, strict=True))
    ]
    return junction, wastes, arcs


def test_an_oversubscribed_push_shares_by_capacity_in_the_first_round():
    junction, wastes, arcs = three_wastes()
    day(junction, *wastes, *arcs)
    reply = junction.push_set(vol(10.0), active=TRUE)
    assert [float(a.flow_in) for a in arcs] == [2.0, 3.0, 1.0]
    assert float(reply[0]) == 4.0


def test_an_undersubscribed_push_shares_by_preference_times_avail():
    junction, wastes, arcs = three_wastes()
    day(junction, *wastes, *arcs)
    reply = junction.push_set(vol(3.0), active=TRUE)
    # nodes.py:459-467 in Python floats: priority 2 + 6 + 3, amount min(6, 3), each share
    # v_change'd from what is still unpushed (core.py:251-284)
    expected, not_pushed = [], 3.0
    for allocation in (2.0, 6.0, 3.0):
        to_send = 3.0 * allocation / 11.0
        sent = not_pushed * (to_send / not_pushed)
        expected.append(sent)
        not_pushed = not_pushed - sent
    assert [float(a.flow_in) for a in arcs] == expected
    assert expected == pytest.approx([3 * 2 / 11, 3 * 6 / 11, 3 * 3 / 11], rel=1e-15)
    assert abs(float(reply[0])) <= 1e-15


def test_a_check_without_a_request_is_not_clipped():
    puller = node("puller")
    stores = [StoreNode("s3", P, storage=3.0), StoreNode("s4", P, storage=4.0)]
    arcs = [arc(f"from_{s.name}", s, puller) for s in stores]
    day(puller, *stores, *arcs)
    assert float(puller.pull_check(None, active=TRUE)[0]) == 7.0
    assert float(puller.pull_check(vol(5.0), active=TRUE)[0]) == 5.0


def test_a_pull_is_shared_by_preference_times_avail():
    puller = node("puller")
    stores = [StoreNode("s4", P, storage=4.0), StoreNode("s5", P, storage=5.0)]
    arcs = [arc(f"from_{s.name}", s, puller, capacity=3.0) for s in stores]
    ctx = day(puller, *stores, *arcs)
    pulled = puller.pull_set(vol(7.0), active=TRUE)
    # round 1: avail 3 + 3, each asked 7 * 3 / 6 = 3.5, clipped to 3; round 2 finds no avail
    assert float(pulled[0]) == 6.0
    assert [float(s.storage[0]) for s in stores] == [1.0, 2.0]
    assert not bool(ctx.maxiter[("puller", "pull_distributed", 0)])


def test_a_forced_push_makes_the_pipe_excess_negative():
    upstream = node("upstream")
    store = StoreNode("store", P, capacity=10.0)
    pipe = arc("pipe", upstream, store, capacity=2.0)
    day(upstream, store, pipe)
    reply = pipe.send_push_request(vol(5.0), active=TRUE, force=True)
    assert float(reply[0]) == 0.0
    assert float(store.storage[0]) == 5.0
    assert float(pipe.get_excess("push", active=TRUE)[0]) == -3.0
    assert float(upstream.check_basic("push", active=TRUE)[0]) == -3.0
    assert float(upstream.get_connected("push", active=TRUE).avail) == 0.0


def test_a_missing_tag_is_refused_by_name():
    junction = node("junction")
    day(junction)
    with pytest.raises(KeyError, match=r"junction.*push_set.*Demand.*Garden"):
        junction.push_set(vol(1.0), ("Demand", "Garden"), active=TRUE)


def test_pull_and_push_arcs_deny_one_direction():
    a, b = StoreNode("a", P, storage=5.0), StoreNode("b", P, storage=5.0)
    pull_arc = arc("pull_only", a, b, cls="PullArc")
    push_arc = arc("push_only", a, b, cls="PushArc")
    day(a, b, pull_arc, push_arc)
    request = vol(2.0)
    assert torch.equal(pull_arc.send_push_request(request, active=TRUE), request)
    assert float(pull_arc.send_push_check(active=TRUE)[0]) == 0.0
    assert float(push_arc.send_pull_request(request, active=TRUE)[0]) == 0.0
    assert float(push_arc.send_pull_check(active=TRUE)[0]) == 0.0
    assert float(pull_arc.flow_in) == 0.0 and float(push_arc.flow_in) == 0.0
    assert float(a.storage[0]) == 5.0 and float(b.storage[0]) == 5.0
    # the allowed directions work
    assert float(pull_arc.send_pull_request(request, active=TRUE)[0]) == 2.0
    assert float(push_arc.send_push_request(request, active=TRUE)[0]) == 0.0


def _two_way_network(p=P):
    """A junction pulling from two stores and pushing into two others."""
    junction = node("junction", p=p)
    sources = [StoreNode(f"src{i}", p, storage=4.0 + i) for i in range(2)]
    sinks = [StoreNode(f"sink{i}", p, capacity=3.0 + i) for i in range(2)]
    arcs = [arc(f"in{i}", s, junction, capacity=3.0, p=p) for i, s in enumerate(sources)]
    arcs += [
        arc(f"out{i}", junction, s, capacity=2.5, preference=1.0 + i, p=p)
        for i, s in enumerate(sinks)
    ]
    return junction, sources + sinks, arcs


def test_an_inactive_call_changes_nothing():
    junction, stores, arcs = _two_way_network(PQ)
    day(junction, *stores, *arcs, shape=(2,), p=PQ)
    before = [(a.flow_in.expand(2).clone(), a.vqip_in.expand(2, 3).clone()) for a in arcs]
    stored = [s.storage.expand(2, 3).clone() for s in stores]
    active = torch.tensor([True, False])
    push = PQ.to_vqip({"volume": 6.0, "phosphate": 0.3, "temperature": 12.0}, (2,))
    junction.push_set(push, active=active)
    junction.pull_set(PQ.to_vqip({"volume": 5.0}, (2,)), active=active)
    for a, (flow, vqip) in zip(arcs, before, strict=True):
        assert float(a.flow_in.expand(2)[1]) == float(flow[1])
        assert torch.equal(a.vqip_in.expand(2, 3)[1], vqip[1])
        assert float(a.flow_in.expand(2)[0]) > 0.0
    for s, old in zip(stores, stored, strict=True):
        assert torch.equal(s.storage.expand(2, 3)[1], old[1])
        assert not torch.equal(s.storage.expand(2, 3)[0], old[0])


def test_a_batch_equals_its_instances_run_one_at_a_time():
    volumes = [1.0, 4.5, 9.0]
    singles = []
    for v in volumes:
        junction, stores, arcs = _two_way_network()
        day(junction, *stores, *arcs)
        reply = junction.push_set(vol(v), active=TRUE)
        singles.append((reply, [a.flow_in for a in arcs], [s.storage for s in stores]))
    junction, stores, arcs = _two_way_network()
    day(junction, *stores, *arcs, shape=(3,))
    batch = torch.tensor(volumes, dtype=torch.float64)[:, None]
    reply = junction.push_set(batch, active=torch.ones(3, dtype=torch.bool))
    for i, (r, flows, storages) in enumerate(singles):
        assert torch.equal(reply[i], r)
        for a, f in zip(arcs, flows, strict=True):
            assert torch.equal(a.flow_in.expand(3)[i], f)
        for s, st in zip(stores, storages, strict=True):
            assert torch.equal(s.storage.expand(3, 1)[i], st)


def test_catchment_avail_is_flow_less_every_out_arcs_vqip_in():
    catchment = node("catchment", Catchment)
    abstractor = StoreNode("abstractor", P, wsimod_type="Demand")
    outlet = node("outlet", Waste)
    abstraction = arc("abstraction", catchment, abstractor)
    river = arc("river", catchment, outlet)
    ctx = day(catchment, abstractor, outlet, abstraction, river)
    catchment.bind({"flow": torch.tensor(10.0), "concentration": torch.zeros(0)})
    assert float(abstraction.send_pull_request(vol(3.0), active=TRUE)[0]) == 3.0
    assert float(catchment.pull_check(None, active=TRUE)[0]) == 7.0
    catchment.route(active=TRUE)
    assert float(river.flow_in) == 7.0
    assert float(catchment.unrouted_water[0]) == 0.0
    catchment.record_terms()
    assert float(ctx.terms["Catchment.flow"][0]) == 10.0
    assert float(ctx.terms["Catchment.unrouted"][0]) == 0.0


def test_catchment_flow_carries_additive_pollutants_as_amounts():
    catchment = node("catchment", Catchment, p=PQ)
    day(catchment, p=PQ)
    catchment.bind({"flow": torch.tensor(4.0), "concentration": torch.tensor([0.5, 11.0])})
    assert catchment.get_flow().tolist() == [4.0, 2.0, 11.0]


def test_a_node_never_pushes_to_a_type_outside_its_of_type():
    junction = node("junction")
    aquifer = StoreNode("aquifer", P, wsimod_type="Groundwater")
    outlet = node("outlet", Waste)
    to_aquifer = arc("to_aquifer", junction, aquifer)
    to_outlet = arc("to_outlet", junction, outlet)
    day(junction, aquifer, outlet, to_aquifer, to_outlet)
    assert float(junction.push_set(vol(5.0), active=TRUE)[0]) == 0.0
    assert float(to_outlet.flow_in) == 5.0
    assert float(to_aquifer.flow_in) == 0.0

    junction = node("junction")
    aquifer = StoreNode("aquifer", P, wsimod_type="Groundwater")
    alone = arc("to_aquifer", junction, aquifer)
    day(junction, aquifer, alone)
    assert float(junction.push_set(vol(5.0), active=TRUE)[0]) == 5.0
    assert float(alone.flow_in) == 0.0


class _Stingy(StoreNode):
    """Says it can take anything, takes one unit per push: the loop never converges."""

    def __init__(self, name):
        super().__init__(name, P)
        self.push_check_handler["default"] = lambda vqip=None, *, active: self._volume_vqip(
            torch.tensor(1e15 if vqip is None else float(vqip[0]), dtype=torch.float64)
        )
        self.push_set_handler["default"] = self._take_one

    def _take_one(self, vqip, *, active):
        taken = torch.minimum(vqip[..., 0], torch.tensor(1.0, dtype=torch.float64))
        self.storage = self.storage + self._volume_vqip(taken)
        return self._volume_vqip(vqip[..., 0] - taken)


def test_the_distribute_loop_stops_at_maxiter_and_says_so():
    junction = node("junction")
    sinks = [_Stingy(f"sink{i}") for i in range(2)]
    arcs = [arc(f"to_{s.name}", junction, s) for s in sinks]
    ctx = day(junction, *sinks, *arcs, record=True)
    reply = junction.push_distributed(vol(50.0), active=TRUE)
    assert float(reply[0]) == 40.0  # five rounds of one unit per sink
    assert bool(ctx.maxiter[("junction", "push_distributed", 0)])
    sites = [site for site, _, _ in ctx.decisions]
    assert sites.count("junction.push_distributed.0.not_pushed") == 5
    junction.push_distributed(vol(1.0), active=TRUE)
    assert not bool(ctx.maxiter[("junction", "push_distributed", 1)])


def test_decisions_record_the_margin():
    junction, wastes, arcs = three_wastes()
    ctx = day(junction, *wastes, *arcs, record=True)
    junction.push_set(vol(10.0), active=TRUE)
    site, outcome, margin = next(
        d for d in ctx.decisions if d[0] == "junction.push_distributed.0.oversubscribed"
    )
    assert bool(outcome) and float(margin) == 4.0


def _network(nodes, arcs, dates=("2000-01-01",)) -> WsimodNetwork:
    return read_wsimod_config(
        {
            "pollutants": [],
            "additive_pollutants": [],
            "non_additive_pollutants": [],
            "dates": list(dates),
            "nodes": nodes,
            "arcs": arcs,
        }
    )


def _flow(name, value=1.0):
    return {"name": name, "type_": "Catchment", "data_input_dict": {("flow", "2000-01-01"): value}}


def test_an_arc_into_a_catchment_is_refused_by_name():
    net = _network(
        [_flow("c"), {"name": "n", "type_": "Node"}],
        [{"name": "back", "type_": "Arc", "in_port": "n", "out_port": "c"}],
    )
    with pytest.raises(ValueError, match="'back'.*catchment 'c'"):
        Catchment.validate(net.nodes[0], net, pollutants=True)


def test_missing_catchment_forcing_is_refused_by_name():
    net = _network([_flow("c")], [], dates=("2000-01-01", "2000-01-02"))
    with pytest.raises(ValueError, match="'c' has no 'flow' forcing on 2000-01-02"):
        Catchment.validate(net.nodes[0], net, pollutants=True)


@pytest.mark.parametrize(
    ("preferences", "match"), [((0.0, 0.0), "preference 0"), ((1.0, -1.0), "negative")]
)
def test_zero_and_negative_preferences_are_refused(preferences, match):
    net = _network(
        [_flow("c"), {"name": "a", "type_": "Waste"}, {"name": "b", "type_": "Waste"}],
        [
            {"name": f"to_{t}", "type_": "Arc", "in_port": "c", "out_port": t, "preference": pr}
            for t, pr in zip("ab", preferences, strict=True)
        ],
    )
    with pytest.raises(ValueError, match=match):
        Catchment.validate(net.nodes[0], net, pollutants=True)


def test_check_recursion_is_the_basic_checks_only():
    spec = NodeSpec("n", "Node", "Node", {})
    assert Node.check_recurses(spec, "push", "default")
    assert Node.check_recurses(spec, "pull", "default")
    assert not Node.check_recurses(spec, "push", "Sewer")
    assert not Waste.check_recurses(spec, "push", "default")
    assert not Catchment.check_recurses(spec, "pull", "default")


def test_the_engine_connects_arcs_in_configuration_order():
    net = _network(
        [_flow("c"), {"name": "j", "type_": "Node"}, {"name": "w", "type_": "Waste"}],
        [
            {"name": name, "type_": "Arc", "in_port": a, "out_port": b}
            for name, a, b in (("z_jw", "j", "w"), ("a_cj", "c", "j"), ("m_jw", "j", "w"))
        ],
    )
    engine = Engine(net, pollutants=QUANTITY_ONLY)
    assert list(engine.arcs) == ["z_jw", "a_cj", "m_jw"]
    assert list(engine.nodes["j"].out_arcs) == ["z_jw", "m_jw"]
    ctx = Ctx(QUANTITY_ONLY, FA, torch.Size(()), day=0, node_names=list(engine.nodes))
    engine.begin_day(ctx)
    c = engine.nodes["c"]
    c.bind(c.configured_parameters(datetime(2000, 1, 1)))
    c.route(active=TRUE)
    assert float(engine.arcs["a_cj"].flow_in) == 1.0
    engine.nodes["w"].record_terms()
    assert float(ctx.terms["Waste.total_in"][0]) == 1.0
    engine.end_day()
    assert float(engine.arcs["a_cj"].flow_in) == 0.0


def test_an_unported_class_is_refused_by_name():
    with pytest.raises(ValueError, match="'Sewer' is not ported"):
        make_node(NodeSpec("s", "Sewer", "Sewer", {}), None, QUANTITY_ONLY)


def test_gradients_flow_through_the_masked_rounds():
    junction, wastes, arcs = three_wastes()
    day(junction, *wastes, *arcs)
    capacities = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64, requires_grad=True)
    for a, c in zip(arcs, capacities, strict=True):
        a.bind(c, a.preference)
    request = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    reply = junction.push_set(request, active=TRUE)
    reply[0].backward()
    # everything above the total capacity comes back: d reply / d capacity = -1, d / d V = 1
    assert capacities.grad.tolist() == [-1.0, -1.0, -1.0]
    assert request.grad.tolist() == [1.0]
