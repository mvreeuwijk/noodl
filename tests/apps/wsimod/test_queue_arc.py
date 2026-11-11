"""``QueueArc`` request rows (W21, D1, D2): ``arcs.py:273-552``.

The whole-mapping tests step noodl's :class:`Engine` day by day (``_stepping.step_days``) and
live WSIMOD 0.8.1 through ``Model.run`` on the same mapping, and compare every day's arc
``vqip_out``, every request call's reply and every tank's storage bitwise: ``QueueArc`` is
``+ - * /`` and ``min``/``max`` only, in WSIMOD's order (rulings W41, W47).
"""

from __future__ import annotations

import copy
import random

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES, QueueArc
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, read_wsimod_config
from noodl.apps.wsimod.engine import Ctx, Engine
from noodl.apps.wsimod.node import Node, Waste
from noodl.apps.wsimod.storage import Reservoir, Storage
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants, empty, sum_vqip
from tests.apps.wsimod._stepping import step_days
from tests.verification._wsimod_cases import CASES

PQ = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)
KEYS = ("volume", "phosphate", "temperature")


# --- the two runs ----------------------------------------------------------------------------
def _queue_arcs(arcs):
    return [arc for arc in arcs.values() if type(arc).__name__ == "QueueArc"]


def run_wsimod(mapping: dict, days: int):
    """WSIMOD's own run, recorded (``record_all``), and every request call into a QueueArc:
    ``(day, arc, method, request volume, reply)``."""
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
    return record_wsimod_run(model, days=days), calls


class Run:
    """noodl's run of a mapping: per day the arcs' ``vqip_out``, the tanks' storage, the
    dropped water, the mass-balance residual and the QueueArcs' stores; and every active
    request call ``(day, arc, method, request volume, reply, frames)``."""

    def __init__(self, mapping: dict, pollutants: Pollutants, days: int) -> None:
        network = read_wsimod_config(mapping)
        engine = Engine(network, pollutants=pollutants)
        self.engine, self.p = engine, pollutants
        self.calls: list[tuple] = []
        for arc in _queue_arcs(engine.arcs):
            for method in ("send_push_request", "send_pull_request"):
                self._wrap(arc, method)
        self.vqip_out: list[dict[str, torch.Tensor]] = []
        self.storage: list[dict[str, torch.Tensor]] = []
        self.dropped: list[torch.Tensor] = []
        self.residual: list[torch.Tensor] = []
        self.unbalanced: list[torch.Tensor] = []  # the residual without the dropped water
        self.stores: list[dict[str, dict[str, torch.Tensor]]] = []
        held = self._held()
        for _, ctx, _ in step_days(network, pollutants, days, engine=engine):
            self.vqip_out.append({n: a.vqip_out.clone() for n, a in engine.arcs.items()})
            self.storage.append(
                {n: x.tank.storage.clone() for n, x in engine.nodes.items() if hasattr(x, "tank")}
            )
            self.stores.append(
                {a.name: {k: v.clone() for k, v in a.stores().items()} for a in self._qs()}
            )
            for node in engine.nodes.values():
                node.record_terms()
            now = self._held()
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
    """Both runs on ``mapping``: arcs' ``vqip_out``, replies and tanks bitwise. Returns them."""
    ref, theirs = run_wsimod(mapping, days)
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
    assert check.worst == 0.0
    return ours, theirs, ref, check


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
