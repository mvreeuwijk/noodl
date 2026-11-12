"""``QueueArc`` request rows (W21, D1, D2): ``arcs.py:273-552``.

The whole-mapping tests step noodl's day driver (``WsimodDay.run_days``) day by day and
live WSIMOD 0.8.1 through ``Model.run`` on the same mapping, and compare every day's arc
``vqip_out``, every request call's reply and every tank's storage bitwise: ``QueueArc`` is
``+ - * /`` and ``min``/``max`` only, in WSIMOD's order (rulings W41, W47).
"""

from __future__ import annotations

import copy
import random

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES, DecayArc, QueueArc
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.day import WsimodDay
from noodl.apps.wsimod.engine import Ctx
from noodl.apps.wsimod.node import Node, Waste
from noodl.apps.wsimod.storage import Reservoir, Storage
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants, empty, extract_vqip, sum_vqip
from tests.verification._wsimod_cases import CASES

PQ = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)
KEYS = ("volume", "phosphate", "temperature")


# --- the two runs ----------------------------------------------------------------------------
def _queue_arcs(arcs):
    return [arc for arc in arcs.values() if type(arc).__name__ in ("QueueArc", "DecayArc")]


def run_wsimod(mapping: dict, days: int):
    """WSIMOD's own run, recorded (``record_all``), every request call into a QueueArc or a
    DecayArc: ``(day, arc, method, request volume, reply)``, and each such arc's queue at the
    record point, ``{(day, arc): queue}`` (copied as ``end_timestep`` starts: nothing runs
    between ``record_all`` and the arcs' ``end_timestep``, ``model.py:1063-1176``)."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import record_wsimod_run, wsimod_model

    model = wsimod_model(mapping)
    index = {date: k for k, date in enumerate(model.dates)}
    calls = []
    for arc in _queue_arcs(model.arcs):
        for method in ("send_push_request", "send_pull_request"):
            original = getattr(arc, method)

            def wrapped(vqip, *args, _original=original, _arc=arc, _method=method, **kwargs):
                volume = vqip["volume"]
                reply = _original(vqip, *args, **kwargs)
                day = index[_arc.in_port.t]
                calls.append((day, _arc.name, _method, volume, copy.deepcopy(reply)))
                return reply

            setattr(arc, method, wrapped)
        original_end = arc.end_timestep

        def end(_original=original_end, _arc=arc):
            queues[(index[_arc.in_port.t], _arc.name)] = copy.deepcopy(_arc.queue)
            return _original()

        arc.end_timestep = end
    queues: dict = {}
    return record_wsimod_run(model, days=days), calls, queues


class Run:
    """noodl's run of a mapping: per day the arcs' ``vqip_out``, the tanks' storage, the
    dropped water, the mass-balance residual and the QueueArcs' stores; and every active
    request call ``(day, arc, method, request volume, reply, frames)``."""

    def __init__(self, mapping: dict, pollutants: Pollutants, days: int) -> None:
        network = read_wsimod_config(mapping)
        day = WsimodDay(network, pollutants=pollutants is not QUANTITY_ONLY)
        assert day.p == pollutants
        engine = day.engine
        self.engine, self.p = engine, pollutants
        self.calls: list[tuple] = []
        for arc in _queue_arcs(engine.arcs):
            for method in ("send_push_request", "send_pull_request"):
                self._wrap(arc, method)
        self.vqip_out: list[dict[str, torch.Tensor]] = []
        self.terms: list[dict[str, torch.Tensor]] = []  # at the record point
        self.storage: list[dict[str, torch.Tensor]] = []
        self.dropped: list[torch.Tensor] = []
        self.residual: list[torch.Tensor] = []
        self.unbalanced: list[torch.Tensor] = []  # the residual without the dropped water
        self.stores: list[dict[str, dict[str, torch.Tensor]]] = []
        day.load_state(day.initial_state())  # the tracing day (D2) left its values behind
        held = self._held()
        for _, ctx, _ in day.run_days(days=days):
            self.vqip_out.append({n: a.vqip_out.clone() for n, a in engine.arcs.items()})
            self.terms.append({k: v.clone() for k, v in ctx.terms.items()})
            self.storage.append(
                {n: x.tank.storage.clone() for n, x in engine.nodes.items() if hasattr(x, "tank")}
            )
            self.stores.append(
                {a.name: {k: v.clone() for k, v in a.stores().items()} for a in self._qs()}
            )
            now = self._held()  # the driver has recorded the day's terms (D11)
            inflow = ctx.terms.get("Catchment.flow", ctx.empty())
            outflow = sum_vqip(
                pollutants,
                ctx.terms.get("Catchment.unrouted", ctx.empty()),
                ctx.terms.get("Waste.total_in", ctx.empty()),
            )
            mask = pollutants.volume_or_additive_mask
            gap = (inflow - outflow - (now - held)) * mask
            self.unbalanced.append(gap)
            self.residual.append(gap - ctx.dropped * mask)
            self.dropped.append(ctx.dropped.clone())
            held = now

    def _qs(self) -> list[QueueArc]:
        return _queue_arcs(self.engine.arcs)

    def _held(self) -> torch.Tensor:
        """Every tank's storage and every QueueArc row (volume and additive columns)."""
        total = empty(self.p, ())
        for node in self.engine.nodes.values():
            if hasattr(node, "tank"):
                total = total + node.tank.storage
        for arc in self._qs():
            total = total + arc.rows.reshape(-1, 1 + self.p.K).sum(0)
        return total

    def _wrap(self, arc: QueueArc, method: str) -> None:
        original = getattr(arc, method)

        def wrapped(vqip, *, active, **kwargs):
            frames = list(arc.ctx.frames)
            reply = original(vqip, active=active, **kwargs)
            if bool(active):
                volume = float(vqip[0])
                self.calls.append((arc.ctx.day, arc.name, method, volume, reply.clone(), frames))
            return reply

        setattr(arc, method, wrapped)


class Bitwise:
    """Every comparison is exact; the worst difference is kept for the report."""

    def __init__(self, keys) -> None:
        self.keys, self.worst, self.count = keys, 0.0, 0

    def vqip(self, label, ours: torch.Tensor, theirs) -> None:
        for i, key in enumerate(self.keys):
            value = theirs[key] if isinstance(theirs, dict) else theirs[i]
            diff = abs(float(ours[i]) - float(value))
            self.worst, self.count = max(self.worst, diff), self.count + 1
            assert float(ours[i]) == float(value), (
                f"{label}[{key}]: noodl {float(ours[i])!r}, WSIMOD {float(value)!r}"
            )


def compare(mapping: dict, days: int, pollutants: Pollutants):
    """Both runs on ``mapping``: arcs' ``vqip_out``, replies, tanks and every queued request
    bitwise. Returns them."""
    ref, theirs, queues = run_wsimod(mapping, days)
    ours = Run(mapping, pollutants, days)
    keys = ("volume", *pollutants.names)
    check = Bitwise(keys)
    columns = [ref.pollutants.index(name) for name in pollutants.names]
    for day in range(days):
        for a, name in enumerate(ref.arc_names):
            theirs_out = [float(ref.arc_flow[day, a])] + [
                float(ref.arc_quality[day, a, c]) for c in columns
            ]
            check.vqip(f"day {day} {name} vqip_out", ours.vqip_out[day][name], theirs_out)
        for t, (node, _) in enumerate(ref.tank_names):
            theirs_tank = [float(ref.tank_storage[day, t])] + [
                float(ref.tank_quality[day, t, c]) for c in columns
            ]
            check.vqip(f"day {day} {node} storage", ours.storage[day][node], theirs_tank)
    assert [c[:3] for c in ours.calls] == [c[:3] for c in theirs]
    for mine, ref_call in zip(ours.calls, theirs, strict=True):
        assert mine[3] == ref_call[3], (mine[:4], ref_call[:4])
        check.vqip(f"day {mine[0]} {mine[1]} {mine[2]} reply", mine[4], ref_call[4])
    for (day, name), queue in queues.items():
        if day < days:
            stores = ours.stores[day][name]
            arc = ours.engine.arcs[name]
            _check_rows(check, f"day {day} {name}", stores["rows"], stores["since"], queue, arc)
    assert check.worst == 0.0
    return ours, theirs, ref, check


def _check_rows(check, label, rows, since, queue, arc) -> None:
    """noodl's rows ``(2, n, J, 1+K)`` against WSIMOD's list: per direction, the present rows by
    ``i`` descending then ``j`` (D1) are the list's requests of that direction in list order,
    bitwise, with WSIMOD's countdown ``time = max(n - (i + since), 0)``."""
    n = arc.number_of_timesteps
    for d, direction in enumerate(("push", "pull")):
        theirs = [r for r in queue if r["direction"] == direction]
        mine = [
            (i, j)
            for i in reversed(range(n))
            for j in range(rows.shape[-2])
            if float(rows[d, i, j, 0]) != 0.0
        ]
        assert len(mine) == len(theirs), (label, direction, mine, len(theirs))
        for (i, j), request in zip(mine, theirs, strict=True):
            check.vqip(f"{label} {direction} row ({i}, {j})", rows[d, i, j], request["vqip"])
            age = i + float(since[d])
            assert request["time"] == max(n - age, 0), (label, direction, i, j, request["time"])


MODES = [pytest.param(PQ, id="pollutants"), pytest.param(QUANTITY_ONLY, id="quantity")]


# --- W21 ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("pollutants", MODES)
def test_queue_arc_delivery_matches_wsimod_on_irregular_pushes(pollutants):
    """D1's rows against WSIMOD's list (spec, ``QueueArc`` slots; case ``queue_arc``).

    ``a`` pushes on days 0, 1, 5, 9 (1e-12) and 10 through the single-arc shortcut; ``b``'s
    ``push_distributed`` reaches ``q`` (through ``a``) in its rounds on days 1, 7 and 8. Day 0
    clips a request below ``FLOAT_ACCURACY``, so update_queue removes it (``arcs.py:483-485``);
    days 1 to 5 are a gap longer than ``n + 1``; on day 7 the due day-5 request meets a full
    ``res`` and its backflow sends ``b`` into a second round; on day 9 the tiny push returns
    before ``update_queue`` although day 7's request is due, and day 10 delivers it.
    """
    spec = CASES["queue_arc"]
    ours, theirs, ref, check = compare(spec.build(), spec.days, pollutants)
    q = [c for c in theirs if c[1] == "q"]
    by_day = {d: [c for c in q if c[0] == d] for d in range(spec.days)}
    # day 0: the arc src -> a let 1.00044e-11 through, q's clip left 9.998e-12 (arcs.py:382-390),
    # which update_queue removed: the water is recorded as dropped
    assert FA <= by_day[0][0][3] and 0.0 < float(ours.dropped[0][0]) < FA
    assert float(ours.vqip_out[0]["q"][0]) == 0.0
    # day 1: one call from a's own site and b's rounds (two sites, several rounds)
    frames = [tuple(c[5]) for c in ours.calls if c[0] == 1 and c[1] == "q"]
    assert frames[0] == () and {f[-1].split("/")[-1] for f in frames[1:]} >= {"round 1", "round 2"}
    # day 7: the due request's backflow comes back to b, whose loop runs a second round
    day7 = [c for c in ours.calls if c[0] == 7 and c[1] == "q"]
    assert by_day[7][1][4]["volume"] > 0.0  # round 1's reply carries the backflow
    assert [f[-1][-7:] for _, _, _, _, _, f in day7[1:]] == ["round 1", "round 2"]
    # day 9: a push of 1e-12 on a due day hands nothing over; day 10 delivers
    assert by_day[9][0][3] == 1e-12
    assert float(ours.vqip_out[9]["q"][0]) == 0.0
    due = ours.stores[9]["q"]["rows"][0, :, :, 0]
    assert float(due.sum()) > 0.0 and float(ours.stores[9]["q"]["since"][0]) == 2.0
    assert float(ours.vqip_out[10]["q"][0]) > 0.0
    assert float(ours.dropped[9][0]) == 1e-12
    # mass balance closes, and only because both drops are recorded
    scale = spec.scale
    for day, residual in enumerate(ours.residual):
        assert float(residual.abs().max()) <= 1e-12 * scale, (day, residual)
    worst = max(float(r.abs().max()) for r in ours.residual)
    assert worst < 1e-13 < float(ours.unbalanced[9][0]) and float(ours.unbalanced[0][0]) > 1e-13
    # the q_pull part of the case: each pull hands over the previous day's
    pulls = [c for c in theirs if c[1] == "q_pull"]
    assert pulls[0][4]["volume"] == 0.0 and all(c[4]["volume"] > 0.0 for c in pulls[1:])


def _pull_mapping(days: int) -> dict:
    case = CASES["queue_arc"].build()
    keep = {"well", "basin", "sink"}
    return {
        **{k: case[k] for k in ("pollutants", "additive_pollutants", "non_additive_pollutants")},
        "dates": case["dates"][:days],
        "nodes": [
            {**n, "type_": "Reservoir"} if n["name"] == "basin" else n
            for n in case["nodes"]
            if n["name"] in keep
        ],
        "arcs": [a for a in case["arcs"] if a["in_port"] in keep and a["out_port"] in keep],
        "orchestration": [{"Reservoir": "make_abstractions"}, {"Reservoir": "distribute"}],
    }


@pytest.mark.parametrize("pollutants", MODES)
def test_a_queue_arc_pull_hands_over_water_pulled_on_earlier_days(pollutants):
    """``arcs.py:318-358`` with ``n = 1``: the reservoir's pull through the arc takes water
    from the storage now and receives the request that is due, so day 0's reply is empty and
    day 1's is what day 0 pulled."""
    days = 6
    ours, theirs, _, _ = compare(_pull_mapping(days), days, pollutants)
    replies = [c[4] for c in ours.calls]
    assert len(replies) == days
    assert float(replies[0][0]) == 0.0
    rows = ours.stores[0]["q_pull"]["rows"]
    pulled_day_0 = rows[1, 0].sum(0)  # the pull row entered on day 0 (i = 0)
    assert float(pulled_day_0[0]) > 0.0
    assert torch.equal(replies[1][0], pulled_day_0[0])


# --- defect rows and drops (noodl only) --------------------------------------------------------
def _pair(n=1):
    """A storage holding 50 and a QueueArc of ``n`` days out of it into a Waste."""
    stocked = {"volume": 50.0, "phosphate": 0.1, "temperature": 11.0}
    spec = NodeSpec("well", "Storage", "Storage", {"capacity": 100.0, "initial_storage": stocked})
    source = Storage(spec, PQ)
    sink = Waste(NodeSpec("sink", "Waste", "Waste", {}), PQ)
    arc = ArcSpec("q", "QueueArc", "well", "sink", number_of_timesteps=n)
    return source, sink, QueueArc(arc, source, sink, PQ)


def _day(*objects, p=PQ, shape=(), record=False):
    ctx = Ctx(p, FA, torch.Size(shape), day=0, node_names=[], record_decisions=record)
    for obj in objects:
        obj.ctx = ctx
    return ctx


@pytest.mark.parametrize("n", [0, 1])
def test_a_zero_volume_pull_with_pollutants_skips_the_rescale(n):
    """Defect row: WSIMOD rescales the request's additive entries by ``volume / 0`` and raises
    ``ZeroDivisionError`` (``arcs.py:341-343``); noodl skips the rescale (D4), as ``Arc`` does."""
    source, sink, q = _pair(n)
    _day(source, sink, q)
    request = PQ.to_vqip({"volume": 0.0, "phosphate": 1.0})
    reply = q.send_pull_request(request, active=TRUE)
    assert reply.tolist() == [0.0, 0.0, 0.0]
    assert all(torch.isfinite(t).all() for t in (reply, q.rows, q.vqip_in, q.flow_in))
    assert float(source.tank.storage[0]) == 50.0


def test_the_tiny_push_is_recorded_as_dropped():
    """``arcs.py:378-379``: a push below ``FLOAT_ACCURACY`` returns empty before the queue, so
    the water is lost; noodl records it as dropped, and nothing is enqueued or handed over."""
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    sink = Waste(NodeSpec("sink", "Waste", "Waste", {}), PQ)
    q = QueueArc(ArcSpec("q", "QueueArc", "src", "sink", number_of_timesteps=0), src, sink, PQ)
    ctx = _day(src, sink, q)
    tiny = PQ.to_vqip({"volume": 1e-12, "phosphate": 3e-13, "temperature": 9.0})
    assert q.send_push_request(tiny, active=TRUE).tolist() == [0.0, 0.0, 0.0]
    assert float(ctx.dropped[0]) == 1e-12 and float(ctx.dropped[1]) == 3e-13
    assert float(q.flow_in) == 0.0 and float(q.vqip_out[0]) == 0.0
    # an inactive tiny push drops nothing
    ctx = _day(src, sink, q)
    q.send_push_request(tiny, active=torch.tensor(False))
    assert float(ctx.dropped[0]) == 0.0


def test_rows_and_anchors_are_carried_state():
    source, sink, q = _pair(n=2)
    assert set(q.stores()) == {"rows", "since"}
    assert q.since.tolist() == [2.0, 2.0]  # n at construction
    assert tuple(q.rows.shape) == (2, 2, 0, 1 + PQ.K)
    _day(source, sink, q)
    q.send_push_request(PQ.to_vqip({"volume": 4.0, "phosphate": 0.2}), active=TRUE)
    assert q.since.tolist() == [0.0, 2.0]
    rows = q.rows.clone()
    assert float(rows[0, 0, 0, 0]) == 4.0
    q.end_timestep()
    assert torch.equal(q.rows, rows)  # end_timestep only ages the anchor (arcs.py:544-545)
    assert q.since.tolist() == [1.0, 2.0]
    q.end_timestep()
    q.end_timestep()
    assert q.since.tolist() == [2.0, 2.0]  # saturates at n
    # load_stores round-trips, and refuses what it does not know
    other = _pair(n=2)[2]
    other.load_stores(q.stores())
    assert torch.equal(other.rows, q.rows) and torch.equal(other.since, q.since)
    with pytest.raises(KeyError, match="register"):
        other.load_stores({"register": q.rows})
    with pytest.raises(ValueError, match="rows of shape"):
        other.load_stores({"rows": torch.zeros(2, 3, 1, 1 + PQ.K)})
    with pytest.raises(ValueError, match=r"arc 'q': since of shape \(3,\)"):
        other.load_stores({"since": torch.zeros(3)})
    for bad in ([3.0, 0.0], [-1.0, 0.0], [0.5, 1.0]):
        with pytest.raises(ValueError, match=r"arc 'q': since must hold whole days in \[0, 2\]"):
            other.load_stores({"since": torch.tensor(bad)})
    other.load_stores({"since": torch.tensor([[0.0, 2.0], [1.0, 1.0]])})  # batched


def test_a_fully_clipped_push_leaves_its_row_empty():
    """A push into a full reservoir enters with volume 0; the row stays absent (all zero), not
    a zero volume carrying the request's temperature."""
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    full = {"capacity": 5.0, "initial_storage": 5.0}
    res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", full), PQ)
    q = QueueArc(ArcSpec("q", "QueueArc", "src", "res", number_of_timesteps=1), src, res, PQ)
    _day(src, res, q)
    push = PQ.to_vqip({"volume": 3.0, "phosphate": 0.03, "temperature": 12.0})
    reply = q.send_push_request(push, active=TRUE)
    assert reply.tolist() == push.tolist()  # all of it refused at the clip
    assert q.rows.shape[-2] == 1 and float(q.rows.abs().sum()) == 0.0


def test_a_request_waits_n_days_and_goes_out_on_the_next_push():
    """``n = 2``: day 0's push is due on day 2, but only a push hands it over (``:399``). The
    due rows go out oldest first (``i = n-1 ... 0``, WSIMOD's list order): on day 6 the reservoir
    takes all of day 3's request and part of day 4's, so the backflow is day 4's rejected part,
    at day 4's temperature."""
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", {"capacity": 6.0}), PQ)
    q = QueueArc(ArcSpec("q", "QueueArc", "src", "res", number_of_timesteps=2), src, res, PQ)
    pushes = {0: (3.0, 10.0), 3: (5.0, 20.0), 4: (2.5, 25.0), 6: (1.0, 30.0)}
    out, replies = [], []
    for day in range(7):
        _day(src, res, q)
        volume, temperature = pushes.get(day, (0.0, 0.0))
        push = PQ.to_vqip(
            {"volume": volume, "phosphate": 0.01 * volume, "temperature": temperature}
        )
        replies.append(q.send_push_request(push, active=TRUE))
        out.append(q.vqip_out.clone())
        q.end_timestep()
        if day == 3:
            res.tank.storage = PQ.to_vqip({})  # emptied, so day 4 enters in full
    # day 3 hands over day 0's 3 (days 1 and 2 pushed nothing, so nothing reached update_queue);
    # day 4's push finds day 3's request one day old; day 6 hands over days 3 and 4
    assert [float(v[0]) for v in out] == [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 6.0]
    backflow = replies[6]
    assert float(backflow[0]) == 1.5  # day 4's 2.5, of which 1 fitted after day 3's 5
    assert float(backflow[2]) == 25.0  # at day 4's temperature, not day 3's 20
    assert float(backflow[1]) == pytest.approx(0.025 * 1.5 / 2.5, rel=1e-15)
    assert float(res.tank.storage[0]) == 6.0


def _random_schedule_run(seed: int, n: int, batch: int = 4, days: int = 25) -> int:
    """One random schedule of push and pull calls on a QueueArc from a storage into a
    reservoir: noodl runs the ``batch`` instances at once (independent schedules, a masked call
    where an instance makes none), WSIMOD one model per instance. Every reply, ``vqip_out``,
    both tanks and the queued volume are compared bitwise; returns the number of values."""
    from wsimod.arcs.arcs import QueueArc as WQueueArc
    from wsimod.nodes.storage import Reservoir as WReservoir
    from wsimod.nodes.storage import Storage as WStorage

    rng = random.Random(seed)
    cap_arc = rng.choice([1e15, 4.0, 2.5])
    cap_res = rng.choice([3.0, 6.0, 20.0])
    slots = [rng.choice(["push", "push", "pull"]) for _ in range(rng.randint(1, 5))]

    def volume():
        r = rng.random()
        if r < 0.1:
            return 0.0
        if r < 0.2:
            return 1e-12
        if r < 0.25:
            return FA * rng.uniform(1.0, 1.5)
        return rng.uniform(0.0, 4.0)

    schedule = [
        [
            [volume() if rng.random() < rng.choice([0.3, 0.6, 0.9]) else None for _ in slots]
            for _ in range(days)
        ]
        for _ in range(batch)
    ]
    empties = [[rng.random() < 0.25 for _ in range(days)] for _ in range(batch)]
    stocked = {"volume": 60.0, "phosphate": 0.3, "temperature": 0.0}
    theirs = []
    for _ in range(batch):
        w_well = WStorage(name="well", capacity=1e4, initial_storage=dict(stocked))
        w_res = WReservoir(name="res", capacity=cap_res)
        w_arc = WQueueArc(
            name="q", capacity=cap_arc, number_of_timesteps=n, in_port=w_well, out_port=w_res
        )
        theirs.append((w_well, w_res, w_arc))
    well = Storage(
        NodeSpec("well", "Storage", "Storage", {"capacity": 1e4, "initial_storage": stocked}), PQ
    )
    res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", {"capacity": cap_res}), PQ)
    spec = ArcSpec("q", "QueueArc", "well", "res", capacity=cap_arc, number_of_timesteps=n)
    q = QueueArc(spec, well, res, PQ)
    check = Bitwise(KEYS)

    def per_instance(t):
        return torch.broadcast_to(t, (batch, *t.shape[-1:]))

    for day in range(days):
        ctx = _day(well, res, q, shape=(batch,))
        empty = torch.tensor([empties[b][day] for b in range(batch)])
        res.tank.storage = torch.where(
            empty[:, None], torch.zeros(1 + PQ.K, dtype=torch.float64), res.tank.storage
        )
        for b in range(batch):
            if empties[b][day]:
                theirs[b][1].tank.storage = theirs[b][1].empty_vqip()
        for j, direction in enumerate(slots):
            given = [schedule[b][day][j] for b in range(batch)]
            active = torch.tensor([v is not None for v in given])
            volumes = torch.tensor([0.0 if v is None else v for v in given], dtype=torch.float64)
            temperature = torch.full_like(volumes, 10.0 + j + 0.37 * day)
            with ctx.frame(f"slot {j}"):
                if direction == "push":
                    request = torch.stack([volumes, volumes * 0.013, temperature], -1)
                    reply = q.send_push_request(request, active=active)
                else:
                    request = torch.stack([volumes, volumes * 0, volumes * 0], -1)
                    reply = q.send_pull_request(request, active=active)
            for b in range(batch):
                if given[b] is None:
                    continue
                w_arc = theirs[b][2]
                if direction == "push":
                    row = {k: float(request[b, i]) for i, k in enumerate(KEYS)}
                    ref = w_arc.send_push_request(row)
                else:
                    ref = w_arc.send_pull_request({"volume": given[b]})
                check.vqip(f"seed {seed} n {n} day {day} {direction} {j} [{b}]", reply[b], ref)
        out = per_instance(q.vqip_out)
        res_storage = per_instance(res.tank.storage)
        well_storage = per_instance(well.tank.storage)
        rows = torch.broadcast_to(q.rows, (batch, *q.rows.shape[-4:]))
        for b, (w_well, w_res, w_arc) in enumerate(theirs):
            label = f"seed {seed} n {n} day {day} [{b}]"
            check.vqip(f"{label} vqip_out", out[b], w_arc.vqip_out)
            check.vqip(f"{label} res", res_storage[b], w_res.tank.storage)
            check.vqip(f"{label} well", well_storage[b], w_well.tank.storage)
            queued = 0.0
            for request in w_arc.queue:
                queued += request["vqip"]["volume"]
            # the row sum is torch's, not WSIMOD's order: equal to rounding only
            assert float(rows[b, ..., 0].sum()) == pytest.approx(queued, rel=1e-12, abs=1e-12)
        q.end_timestep()
        well.end_timestep()
        res.end_timestep()
        for objects in theirs:
            for obj in objects:
                obj.end_timestep()
    assert check.worst == 0.0
    return check.count


@pytest.fixture
def simple_wsimod(monkeypatch):
    pytest.importorskip("wsimod")
    from wsimod.core import constants

    monkeypatch.setattr(constants, "POLLUTANTS", ["phosphate", "temperature"])
    monkeypatch.setattr(constants, "ADDITIVE_POLLUTANTS", ["phosphate"])
    monkeypatch.setattr(constants, "NON_ADDITIVE_POLLUTANTS", ["temperature"])
    monkeypatch.setattr(constants, "FLOAT_ACCURACY", FA)
    return constants


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_queue_arc_equals_wsimod_on_random_schedules(simple_wsimod, seed, n):
    """Random schedules of up to five push and pull call sites a day, tiny and zero requests,
    a clipping arc, a receiver emptied at random: noodl's rows (D1) hand over in WSIMOD's list
    order, bitwise, for every instance of the batch."""
    assert _random_schedule_run(seed, n) > 0


def test_the_call_sequence_is_recorded_once_and_then_enforced():
    """D2: the first day records the labels; a later day with another sequence is an internal
    error, and ``begin_tracing``/``end_tracing`` record a new one."""
    src = Node(NodeSpec("src", "Node", "Node", {}), QUANTITY_ONLY)
    sink = Waste(NodeSpec("sink", "Waste", "Waste", {}), QUANTITY_ONLY)
    q = QueueArc(
        ArcSpec("q", "QueueArc", "src", "sink", number_of_timesteps=1), src, sink, QUANTITY_ONLY
    )
    ctx = _day(src, sink, q, p=QUANTITY_ONLY)
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    with ctx.frame("node:x/push_distributed#0/round 1"):
        q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    q.end_timestep()
    assert q.calls["push"] == [
        "- | tag 'default'",
        "node:x/push_distributed#0/round 1 | tag 'default'",
    ]
    assert q.rows.shape[-2] == 2  # J
    _day(src, sink, q, p=QUANTITY_ONLY)
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    with pytest.raises(RuntimeError, match="D2"):  # the second call outside the loop frame
        q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    _day(src, sink, q, p=QUANTITY_ONLY)
    q.begin_tracing()
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    assert q.end_tracing() == {"push": ["- | tag 'default'"], "pull": []}
    q.end_timestep()
    _day(src, sink, q, p=QUANTITY_ONLY)
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    q.end_timestep()  # the same one call: accepted
    _day(src, sink, q, p=QUANTITY_ONLY)
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    with pytest.raises(RuntimeError, match="D2"):  # one call more than traced
        q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    q.begin_tracing()  # two calls traced, then a day with one: refused at the end of the day
    _day(src, sink, q, p=QUANTITY_ONLY)
    for _ in range(2):
        q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    q.end_tracing()
    q.end_timestep()
    _day(src, sink, q, p=QUANTITY_ONLY)
    q.send_push_request(QUANTITY_ONLY.to_vqip({"volume": 1.0}), active=TRUE)
    with pytest.raises(RuntimeError, match="1 push request calls today, 2 on the traced day"):
        q.end_timestep()


def test_a_batch_equals_its_instances_and_a_masked_call_changes_nothing():
    """Three instances push on different days, so their ``since`` differ; each equals its own
    single run, and the instance that never pushes is untouched."""
    pushes = torch.tensor(
        [
            [True, False, True],
            [False, False, True],
            [True, False, False],
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ]
    )  # (day, instance)

    def run(volumes, schedule, shape):
        src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
        res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", {"capacity": 5.0}), PQ)
        q = QueueArc(ArcSpec("q", "QueueArc", "src", "res", number_of_timesteps=2), src, res, PQ)
        replies, since = [], []
        for day in range(schedule.shape[0]):
            _day(src, res, q, shape=shape)
            push = torch.stack(
                [volumes * (day + 1), volumes * 0.01, torch.full_like(volumes, 10.0 + day)], -1
            )
            replies.append(q.send_push_request(push, active=schedule[day]))
            q.end_timestep()
            since.append(torch.broadcast_to(q.since, (*shape, 2)).clone())
        return torch.stack(replies), res.tank.storage, q.rows, torch.stack(since)

    volumes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    batch = run(volumes, pushes, (3,))
    assert not torch.equal(batch[3][:, 0], batch[3][:, 2])  # the anchors differ
    for b in (0, 2):
        single = run(volumes[b], pushes[:, b], ())
        for day in range(pushes.shape[0]):
            if pushes[day, b]:
                assert torch.equal(batch[0][day, b], single[0][day])
        assert torch.equal(batch[1][b], single[1])
        assert torch.equal(batch[2][b], single[2]) and torch.equal(batch[3][:, b], single[3])
    assert float(batch[1][1, 0]) == 0.0 and float(batch[2][1].abs().sum()) == 0.0
    assert batch[3][-1, 1].tolist() == [2.0, 2.0]


def test_the_queue_arc_is_registered_and_validates_its_delay():
    assert ARC_CLASSES["QueueArc"] is QueueArc
    from noodl.apps.wsimod.config import WsimodNetwork

    network = WsimodNetwork(PQ, FA, [], [], [], [], [])
    QueueArc.validate(ArcSpec("q", "QueueArc", "a", "b", number_of_timesteps=3), network)
    with pytest.raises(ValueError, match=r"arc 'q'.*number_of_timesteps"):
        QueueArc.validate(ArcSpec("q", "QueueArc", "a", "b", number_of_timesteps=-1), network)


def test_gradients_through_the_rows_are_finite():
    """A tiny push, a masked instance and a delivery with backflow keep finite gradients: every
    division is over a safe denominator and every branch a detached ``where``."""
    x = torch.tensor([2.0, 1e-12, 3.0], dtype=torch.float64, requires_grad=True)
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", {"capacity": 2.5}), PQ)
    q = QueueArc(ArcSpec("q", "QueueArc", "src", "res", number_of_timesteps=1), src, res, PQ)
    active = torch.tensor([True, True, False])
    replies = torch.zeros((), dtype=torch.float64)
    for day in range(3):
        _day(src, res, q, shape=(3,))
        push = torch.stack([x * (day + 1), x * 0.01, x * 0 + 12.0], -1)
        replies = replies + q.send_push_request(push, active=active).sum()
        q.end_timestep()
    stores = res.tank.storage.sum() + q.rows.sum()
    (grad,) = torch.autograd.grad(stores + replies, x, retain_graph=True)
    assert torch.isfinite(grad).all()
    (grad,) = torch.autograd.grad(stores, x)
    assert torch.isfinite(grad).all()
    assert grad[2] == 0.0  # the masked instance's stores do not depend on its inputs
    assert grad[0] != 0.0


# --- DecayArc (arcs.py:668-728; X24, X33, D21) -----------------------------------------------
ARC_DECAYS = {
    "phosphate": {"constant": 0.08, "exponent": 1.06},
    "temperature": {"constant": 0.01, "exponent": 1.02},  # a non-additive column decays too
}
STOCKED = {"volume": 60.0, "phosphate": 0.3, "temperature": 9.0}
TERM = "DecayArc.decayed"
EVENT = "decay_arc_order_unknown"


def _decay_mapping(n: int) -> dict:
    """W21's mapping (case ``queue_arc``) with ``q`` a ``DecayArc`` of ``n`` days decaying
    ``phosphate`` and ``temperature``, its in-port ``a`` given a temperature forcing, and
    ``q_pull`` a ``DecayArc`` of one day out of ``well``, which gets one too."""
    mapping = CASES["queue_arc"].build()
    temperatures = {}
    for k, date in enumerate(mapping["dates"]):
        temperatures[("temperature", date)] = 12.0 + 1.37 * k - 0.05 * k * k
    for node in mapping["nodes"]:
        if node["name"] in ("a", "well"):
            node["data_input_dict"] = dict(temperatures)
    for arc in mapping["arcs"]:
        if arc["name"] == "q":
            arc.update(type_="DecayArc", number_of_timesteps=n, decays=ARC_DECAYS)
        if arc["name"] == "q_pull":
            arc.update(type_="DecayArc", decays={"phosphate": ARC_DECAYS["phosphate"]})
    return mapping


@pytest.mark.parametrize("n", [2, 0])
@pytest.mark.parametrize("pollutants", MODES)
def test_decay_arc_matches_wsimod_on_irregular_requests(pollutants, n):
    """3a's W21 mapping (as corrected by W58) with ``q`` a ``DecayArc``: every day's
    ``vqip_out``, every reply, every tank and every queued request equal WSIMOD's (D1's rows in
    list order), in both modes. With ``n = 0`` each request decays once, on entry, and is
    delivered at once: nothing waits. The mass balance closes with the arcs' terms."""
    from noodl.apps.wsimod.build import build_model
    from noodl.apps.wsimod.run import wsimod_run

    mapping = _decay_mapping(n)
    days = CASES["queue_arc"].days
    ours, _, _, check = compare(mapping, days, pollutants)
    assert check.count > 0
    q = ours.engine.arcs["q"]
    assert isinstance(q, DecayArc) and q.number_of_timesteps == n
    if n == 0:
        assert all(float(s["q"]["rows"].abs().sum()) == 0.0 for s in ours.stores)
    if pollutants is PQ:
        delivered = [d for d in range(days) if float(ours.vqip_out[d]["q"][0]) > 0.0]
        assert delivered
        # every delivery day decays something on entry or at the end of a day
        assert all(float(ours.terms[d][TERM][1]) > 0.0 for d in delivered if n == 0)
    model, state, drivers = build_model(
        read_wsimod_config(mapping), pollutants=pollutants is not QUANTITY_ONLY
    )
    balance = wsimod_run(model, state, drivers).mass_balance
    assert float(balance.residual.abs().max()) <= 1e-12 * CASES["queue_arc"].scale
    if pollutants is PQ:
        assert float(balance.terms[TERM][:, 1].sum()) > 0.0


def _decay_pair(n, *, cap_res=6.0, cap_arc=1e15, decays=ARC_DECAYS):
    """A noodl ``DecayArc`` of ``n`` days out of a storage (its in-port, whose temperature it
    reads) into a reservoir of capacity ``cap_res``."""
    from noodl.apps.wsimod.decay import read_decays

    spec = {"capacity": 1e4, "initial_storage": STOCKED}
    well = Storage(NodeSpec("well", "Storage", "Storage", spec), PQ)
    res = Reservoir(NodeSpec("res", "Reservoir", "Reservoir", {"capacity": cap_res}), PQ)
    arc = ArcSpec(
        "q",
        "DecayArc",
        "well",
        "res",
        capacity=cap_arc,
        number_of_timesteps=n,
        decays=read_decays("arc 'q'", decays, PQ),
    )
    return well, res, DecayArc(arc, well, res, PQ)


def _start(well, res, q, temperature, *, shape=(), d=0):
    """Day ``d``'s context, the in-port's temperature, then the arc's decay (D15)."""
    ctx = Ctx(PQ, FA, torch.Size(shape), day=d, node_names=[], record_decisions=False)
    for obj in (well, res, q):
        obj.ctx = ctx
    well.bind({"temperature": torch.as_tensor(temperature, dtype=torch.float64)})
    q.bind_decay(q.defaults["decays.constant"], q.defaults["decays.exponent"])
    return ctx


def _request(direction, volumes, j, day):
    temperature = torch.full_like(volumes, 10.0 + j + 0.37 * day)
    if direction == "push":
        return torch.stack([volumes, volumes * 0.013, temperature], -1)
    return torch.stack([volumes, volumes * 0, volumes * 0], -1)


def _decay_schedule_run(schedule, slots, temperatures, *, n, cap_res=6.0, cap_arc=1e15, emptied=()):
    """``schedule[b][day][j]`` is slot ``j``'s request volume for instance ``b`` (``None``: no
    call). noodl runs the instances as one batch with masked calls, WSIMOD one ``DecayArc`` per
    instance. Every reply, ``vqip_out``, both tanks and every queued request (after each end of
    day) are compared bitwise, and each day's ``"DecayArc.decayed"`` term against WSIMOD's own
    ``diff``s (the day's entries, then its end of day, summed from empty) on every instance and
    day without the event of D21. The reservoir is emptied on the days in ``emptied``.
    Returns ``(check, events (days, batch), terms compared)``."""
    pytest.importorskip("wsimod")
    from wsimod.arcs.arcs import DecayArc as WDecayArc
    from wsimod.nodes.storage import Reservoir as WReservoir
    from wsimod.nodes.storage import Storage as WStorage

    batch, days = len(schedule), len(schedule[0])
    theirs, diffs = [], [[] for _ in range(batch)]
    for b in range(batch):
        w_well = WStorage(
            name="well", capacity=1e4, initial_storage=dict(STOCKED), data_input_dict={}
        )
        w_res = WReservoir(name="res", capacity=cap_res)
        w_arc = WDecayArc(
            name="q",
            capacity=cap_arc,
            number_of_timesteps=n,
            in_port=w_well,
            out_port=w_res,
            decays=copy.deepcopy(ARC_DECAYS),
        )
        original = w_arc.generic_temperature_decay

        def capture(t, d, temperature, _original=original, _b=b):
            out, diff = _original(t, d, temperature)
            diffs[_b].append(diff)
            return out, diff

        w_arc.generic_temperature_decay = capture
        theirs.append((w_well, w_res, w_arc))
    well, res, q = _decay_pair(n, cap_res=cap_res, cap_arc=cap_arc)
    check = Bitwise(KEYS)
    events, compared = [], 0

    def per_instance(t):
        return torch.broadcast_to(t, (batch, *t.shape[-1:]))

    for day in range(days):
        temps = [temperatures[b][day] for b in range(batch)]
        ctx = _start(well, res, q, temps, shape=(batch,), d=day)
        for b, (w_well, _, _) in enumerate(theirs):
            w_well.t = day
            w_well.data_input_dict[("temperature", day)] = temps[b]
            diffs[b].clear()
        if day in emptied:
            res.tank.storage = torch.zeros(1 + PQ.K, dtype=torch.float64)
            for _, w_res, _ in theirs:
                w_res.tank.storage = w_res.empty_vqip()
        for j, direction in enumerate(slots):
            given = [schedule[b][day][j] for b in range(batch)]
            active = torch.tensor([v is not None for v in given])
            volumes = torch.tensor([0.0 if v is None else v for v in given], dtype=torch.float64)
            request = _request(direction, volumes, j, day)
            with ctx.frame(f"slot {j}"):
                if direction == "push":
                    reply = q.send_push_request(request, active=active)
                else:
                    reply = q.send_pull_request(request, active=active)
            for b in range(batch):
                if given[b] is None:
                    continue
                w_arc = theirs[b][2]
                if direction == "push":
                    row = {k: float(request[b, i]) for i, k in enumerate(KEYS)}
                    ref = w_arc.send_push_request(row)
                else:
                    ref = w_arc.send_pull_request({"volume": given[b]})
                check.vqip(f"n {n} day {day} {direction} {j} [{b}]", reply[b], ref)
        out = per_instance(q.vqip_out)
        res_storage, well_storage = per_instance(res.tank.storage), per_instance(well.tank.storage)
        for b, (w_well, w_res, w_arc) in enumerate(theirs):
            label = f"n {n} day {day} [{b}]"
            check.vqip(f"{label} vqip_out", out[b], w_arc.vqip_out)
            check.vqip(f"{label} res", res_storage[b], w_res.tank.storage)
            check.vqip(f"{label} well", well_storage[b], w_well.tank.storage)
        q.end_timestep()
        well.end_timestep()
        res.end_timestep()
        for objects in theirs:
            for obj in objects:
                obj.end_timestep()
        rows = torch.broadcast_to(q.rows, (batch, *q.rows.shape[-4:]))
        since = torch.broadcast_to(q.since, (batch, 2))
        event = ctx.events.get(EVENT, {}).get("q", torch.tensor(False))
        events.append(torch.broadcast_to(event, (batch,)).clone())
        term = per_instance(ctx.terms.get(TERM, ctx.empty()))
        for b, (_, _, w_arc) in enumerate(theirs):
            label = f"n {n} end of day {day} [{b}]"
            for d, direction in enumerate(("push", "pull")):
                mine = [
                    (i, j)
                    for i in reversed(range(n))
                    for j in range(rows.shape[-2])
                    if float(rows[b, d, i, j, 0]) != 0.0
                ]
                listed = [r for r in w_arc.queue if r["direction"] == direction]
                assert len(mine) == len(listed), (label, direction)
                for (i, j), request in zip(mine, listed, strict=True):
                    check.vqip(f"{label} {direction} ({i}, {j})", rows[b, d, i, j], request["vqip"])
                    # since has advanced, and WSIMOD's countdown has been decremented
                    assert request["time"] == max(n - (i + float(since[b, d])), 0), label
            if not bool(events[-1][b]):
                total = 0.0
                for diff in diffs[b]:
                    total = total + diff["phosphate"]
                assert float(term[b, 1]) == total, (label, float(term[b, 1]), total)
                assert float(term[b, 0]) == 0.0
                compared += 1
    assert check.worst == 0.0
    return check, torch.stack(events), compared


def _random_decay_schedule(seed: int, batch: int = 4, days: int = 25):
    rng = random.Random(seed)
    slots = [rng.choice(["push", "push", "pull"]) for _ in range(rng.randint(2, 4))]

    def volume():
        r = rng.random()
        if r < 0.1:
            return 0.0
        if r < 0.2:
            return 1e-12
        return rng.uniform(0.0, 4.0)

    schedule = [
        [
            [volume() if rng.random() < rng.choice([0.3, 0.6, 0.9]) else None for _ in slots]
            for _ in range(days)
        ]
        for _ in range(batch)
    ]
    temperatures = [[rng.uniform(-5.0, 45.0) for _ in range(days)] for _ in range(batch)]
    emptied = {d for d in range(days) if rng.random() < 0.25}
    return schedule, slots, temperatures, emptied


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decay_arc_equals_wsimod_on_random_schedules(simple_wsimod, seed, n):
    """Random push and pull schedules through a ``DecayArc`` into a reservoir emptied at random,
    temperatures from -5 to 45 degC (the phosphate cap binds on the hot days): replies, rows,
    tanks and, inside the precondition, the term are WSIMOD's bitwise for every instance."""
    schedule, slots, temperatures, emptied = _random_decay_schedule(seed)
    check, _, compared = _decay_schedule_run(
        schedule, slots, temperatures, n=n, cap_res=6.0, emptied=emptied
    )
    assert check.count > 0 and compared > 0


def test_a_pulled_request_decays_after_it_leaves_the_sender(simple_wsimod):
    """``arcs.py:318-358`` with ``n = 1``: the sender loses the undecayed request
    (``pull_set``, ``:348``), the request decays on entry and at the end of the day, and the
    next day's reply is the decayed request, WSIMOD's bitwise."""
    schedule = [[[2.0], [1.0], [None]]]
    temperatures = [[18.0, 25.0, 21.0]]
    _decay_schedule_run(schedule, ["pull"], temperatures, n=1)
    well, res, q = _decay_pair(1)
    before = well.tank.storage.clone()
    _start(well, res, q, 18.0)
    first = q.send_pull_request(PQ.to_vqip({"volume": 2.0}), active=TRUE)
    lost = extract_vqip(PQ, before, well.tank.storage)
    assert float(first[0]) == 0.0
    assert float(lost[0]) == 2.0 and float(lost[1]) == pytest.approx(0.01, rel=1e-12)
    entered = q.rows[1, 0, 0].clone()
    assert float(entered[0]) == 2.0 and float(entered[1]) < float(lost[1])
    # decayed after it left the sender (the extracted difference is the pulled water to rounding)
    assert entered.tolist() == pytest.approx(q.decay.make_decay(lost)[0].tolist(), rel=1e-12)
    q.end_timestep()
    waiting = q.rows[1, 0, 0].clone()
    assert torch.equal(waiting, q.decay.make_decay(entered)[0])  # and again at the end of day 0
    _start(well, res, q, 25.0, d=1)
    second = q.send_pull_request(PQ.to_vqip({"volume": 1.0}), active=TRUE)
    assert torch.equal(second, waiting)


def test_backflow_is_the_decayed_water():
    """``arcs.py:396-405``: a due request that meets a full receiver comes back as backflow. It
    is the water as it waited, decayed on entry and at the end of the day, and ``vqip_in`` falls
    by it; the sender gets the decayed water back."""
    well, res, q = _decay_pair(1, cap_res=10.0)
    _start(well, res, q, 22.0)
    push = PQ.to_vqip({"volume": 3.0, "phosphate": 0.06, "temperature": 14.0})
    q.send_push_request(push, active=TRUE)
    q.end_timestep()
    waiting = q.rows[0, 0, 0].clone()
    assert float(waiting[1]) < 0.06
    res.tank.storage = PQ.to_vqip({"volume": 10.0, "phosphate": 0.1, "temperature": 9.0})
    _start(well, res, q, 24.0, d=1)
    second = PQ.to_vqip({"volume": 1.0, "phosphate": 0.02, "temperature": 16.0})
    reply = q.send_push_request(second, active=TRUE, force=True)
    entered = q.rows[0, 0, 0]  # day 1's request, decayed on entry, waits (n = 1)
    assert float(entered[0]) == 1.0
    # forced, so nothing was clipped: the reply is the backflow, the waiting water (the reply's
    # temperature is blended with the empty not-pushed part, core.py:112-114)
    assert torch.equal(reply[:2], waiting[:2])
    assert float(reply[2]) == pytest.approx(float(waiting[2]), rel=1e-15)
    assert torch.equal(q.vqip_in, extract_vqip(PQ, second, waiting))
    assert float(q.vqip_out.abs().sum()) == 0.0


def _ordered_case():
    """``n = 3`` and three call sites a day (push, pull, push). Day 0 pushes, day 1 pulls, day 2
    pulls and pushes at both sites: at the end of day 2 the list holds the day-0 push (age 2),
    the day-1 pull (age 1) and day 2's pull and two pushes (age 0), none due."""
    slots = ["push", "pull", "push"]
    schedule = [
        [[0.7, None, None], [None, 1.9, None], [2.3, 0.37, 3.1]],
    ]
    return slots, schedule, [[13.0, 27.0, 31.0]]


def test_the_decay_arc_term_follows_wsimods_list_order():
    """Review Focus 2 (X24): the end-of-day term is the rows' ``diff``s summed in WSIMOD's list
    order, entry age descending, then the merged call order of the entry day, bitwise. The
    rows are inside the precondition (no due row waits in either direction), asserted from
    ``since`` and the rows. Summing in ascending age gives another last bit, so the test
    discriminates."""
    slots, schedule, temperatures = _ordered_case()
    well, res, q = _decay_pair(3, cap_res=1e4)
    for day in range(3):
        ctx = _start(well, res, q, temperatures[0][day], d=day)
        for j, direction in enumerate(slots):
            volume = schedule[0][day][j]
            active = torch.tensor(volume is not None)
            request = _request(direction, torch.tensor(volume or 0.0, dtype=torch.float64), j, day)
            with ctx.frame(f"slot {j}"):
                if direction == "push":
                    q.send_push_request(request, active=active)
                else:
                    q.send_pull_request(request, active=active)
        if day < 2:
            q.end_timestep()
    assert q.entry_rank == {("push", 0): 0, ("pull", 0): 1, ("push", 1): 2}
    rows, since = q.rows.clone(), q.since.clone()
    present = rows[..., 0] != 0
    due = since[:, None] + torch.arange(3) >= 3
    assert not bool((present & due[..., None]).any())  # inside the precondition
    # WSIMOD's list: (direction, i, j) in entry order; age = i + since
    listed = [(0, 2, 0), (1, 1, 0), (0, 0, 0), (1, 0, 0), (0, 0, 1)]
    assert sorted(listed) == sorted(tuple(x) for x in present.nonzero().tolist())
    assert [i + int(since[d]) for d, i, _ in listed] == [2, 1, 0, 0, 0]
    diffs = {key: q.make_decay(rows[key])[1] for key in listed}

    def summed(order):
        total = empty(PQ, ())
        for key in order:
            total = sum_vqip(PQ, total, diffs[key])
        return total

    end = Ctx(PQ, FA, torch.Size(()), day=2, node_names=[], record_decisions=False)
    q.ctx = end  # the end of day alone, without the day's entries
    q.end_timestep()
    term = end.terms[TERM]
    assert torch.equal(term, summed(listed))
    ascending = summed(sorted(listed, key=lambda k: (k[1] + int(since[k[0]]), listed.index(k))))
    assert not torch.equal(term[1], ascending[1])
    assert float((term - ascending).abs().max()) < 1e-15
    assert EVENT not in end.events or not bool(end.events[EVENT]["q"])


def test_the_order_unknown_event_marks_the_precondition(simple_wsimod):
    """D21: with ``n = 1``, day 0 enters a push and a pull and day 1 calls nothing, so at the end
    of day 1 both directions hold a due request still waiting: the event is true on that day
    only. Day 2's calls hand both over. The delivered values and the rows equal WSIMOD's on
    every day; instance 1 never pulls, so its event stays false."""
    schedule = [
        [[1.5, 2.0], [None, None], [0.5, 0.8], [0.4, None]],
        [[1.5, None], [None, None], [0.5, None], [0.4, None]],
    ]
    temperatures = [[15.0, 22.0, 19.0, 17.0], [15.0, 22.0, 19.0, 17.0]]
    _, events, compared = _decay_schedule_run(schedule, ["push", "pull"], temperatures, n=1)
    assert events[:, 0].tolist() == [False, True, False, False]
    assert not bool(events[:, 1].any())
    assert compared == 7  # every instance and day but the flagged one


def test_the_decay_arc_is_registered_and_validated():
    """``ARC_CLASSES`` holds both user Decay arcs; ``validate`` refuses a missing in-port
    temperature, naming the arc, the node and the date."""
    from noodl.apps.wsimod.arcs import DecayArcAlt
    from noodl.apps.wsimod.config import WsimodNetwork, parse_date

    assert ARC_CLASSES["DecayArc"] is DecayArc and ARC_CLASSES["DecayArcAlt"] is DecayArcAlt
    date = parse_date("2000-01-01")
    port = NodeSpec("a", "Node", "Node", {}, data={("temperature", date): 9.0})
    network = WsimodNetwork(PQ, FA, [], [date], ["2000-01-01"], [port], [])
    for cls in (DecayArc, DecayArcAlt):
        cls.validate(ArcSpec("q", cls.__name__, "a", "b", number_of_timesteps=1), network)
    port.data = {}
    for cls in (DecayArc, DecayArcAlt):
        with pytest.raises(ValueError, match=r"arc 'q'.*node 'a'.*temperature.*2000-01-01"):
            cls.validate(ArcSpec("q", cls.__name__, "a", "b"), network)
