"""Tanks, ``QueueTank`` and the ``AltQueueArc`` shift register (``tanks.py``, ``arcs.py:555-665``).

The random-sequence tests drive noodl's tanks and live WSIMOD tanks with the same calls and
compare every reply and store bitwise: tank arithmetic is ``+ - * /`` and ``min``/``max`` only,
in WSIMOD's order, so no rounding bound is needed. The noodl side runs a batch of independent
sequences at once with a random ``active`` mask per call; a WSIMOD instance makes a call only
where it is active (D3).
"""

from __future__ import annotations

import pytest
import torch

from noodl.apps.wsimod.arcs import ARC_CLASSES, AltQueueArc
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, WsimodNetwork
from noodl.apps.wsimod.engine import Ctx
from noodl.apps.wsimod.node import Node, Waste
from noodl.apps.wsimod.tanks import DecayQueueTank, DecayTank, QueueTank, ResidenceTank, Tank
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants

PQ = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))
FA = 1e-11
TRUE = torch.tensor(True)
B = 4  # instances per noodl batch


def day(*objects, p=QUANTITY_ONLY, shape=(), d=0, record=False):
    ctx = Ctx(p, FA, torch.Size(shape), day=d, node_names=[], record_decisions=record)
    for obj in objects:
        obj.ctx = ctx
    return ctx


def vol(v, p=QUANTITY_ONLY):
    return p.to_vqip({"volume": v})


def queue_tank(n=0, times=(0,), capacity=10.0, initial=0.0, p=QUANTITY_ONLY):
    return QueueTank(
        p,
        number_of_timesteps=n,
        push_times=times,
        capacity=capacity,
        area=1.0,
        datum=0.0,
        initial_storage=initial,
        owner="N",
        path="sewer_tank",
    )


# --- live WSIMOD -----------------------------------------------------------------------------
@pytest.fixture
def wsimod_tanks(monkeypatch):
    pytest.importorskip("wsimod")
    from wsimod.core import constants
    from wsimod.nodes import tanks

    monkeypatch.setattr(constants, "POLLUTANTS", ["phosphate", "temperature"])
    monkeypatch.setattr(constants, "ADDITIVE_POLLUTANTS", ["phosphate"])
    monkeypatch.setattr(constants, "NON_ADDITIVE_POLLUTANTS", ["temperature"])
    monkeypatch.setattr(constants, "FLOAT_ACCURACY", FA)
    return tanks


class Worst:
    """Every comparison is bitwise; the worst difference is kept for the report."""

    def __init__(self) -> None:
        self.worst = 0.0

    def vqip(self, label, ours, theirs: dict) -> None:
        for i, key in enumerate(("volume", *PQ.names)):
            self.value(f"{label}[{key}]", ours[i], theirs[key])

    def value(self, label, ours, theirs) -> None:
        diff = abs(float(ours) - float(theirs))
        self.worst = max(self.worst, diff)
        assert float(ours) == float(theirs), f"{label}: noodl {float(ours)!r}, WSIMOD {theirs!r}"


def random_vqip(g, scale=5.0):
    v = float(torch.rand((), generator=g, dtype=torch.float64)) * scale
    ph = float(torch.rand((), generator=g, dtype=torch.float64))
    t = float(torch.rand((), generator=g, dtype=torch.float64)) * 20
    return {"volume": v, "phosphate": ph, "temperature": t}


def stack(dicts):
    return torch.stack([PQ.to_vqip(x) for x in dicts])


def test_tanks_equal_wsimod_tanks_on_random_sequences(wsimod_tanks):
    g = torch.Generator().manual_seed(6)
    for kind in ("Tank", "ResidenceTank"):
        theirs, caps, inits = [], [], []
        for _ in range(B):
            cap = float(torch.rand((), generator=g, dtype=torch.float64)) * 10 + 1
            init = random_vqip(g, 6.0)
            kw = {"capacity": cap, "area": 1.0, "datum": 0.0, "initial_storage": dict(init)}
            if kind == "ResidenceTank":
                theirs.append(wsimod_tanks.ResidenceTank(residence_time=3, **kw))
            else:
                theirs.append(wsimod_tanks.Tank(**kw))
            caps.append(cap)
            inits.append(init)
        kw = {
            "capacity": torch.tensor(caps, dtype=torch.float64),
            "area": 1.0,
            "datum": 0.0,
            "initial_storage": stack(inits),
            "owner": "N",
        }
        if kind == "ResidenceTank":
            ours = ResidenceTank(PQ, residence_time=3.0, **kw)
        else:
            ours = Tank(PQ, **kw)
        day(ours, p=PQ, shape=(B,))
        ops = ["push", "push_force", "pull", "pull_all", "pull_ponded", "evaporate"]
        ops += ["excess", "excess_req", "avail", "avail_req"]
        if kind == "ResidenceTank":
            ops.append("pull_outflow")
        w = Worst()
        hits_zero = 0
        for step in range(200):
            op = ops[int(torch.randint(len(ops), (), generator=g))]
            active = torch.rand(B, generator=g) < 0.8
            requests = [random_vqip(g) for _ in range(B)]
            for r in requests[: B // 2]:
                if torch.rand((), generator=g) < 0.2:
                    r["volume"] = 0.0
            req = stack(requests)
            evap = torch.tensor([r["volume"] for r in requests], dtype=torch.float64)
            if op == "push":
                got = ours.push_storage(req, active=active)
            elif op == "push_force":
                got = ours.push_storage(req, active=active, force=True)
            elif op == "pull":
                got = ours.pull_storage(req, active=active)
            elif op == "pull_all":
                got = ours.pull_storage(vol(1e6, PQ), active=active)
            elif op == "pull_ponded":
                got = ours.pull_ponded(active=active)
            elif op == "evaporate":
                got = ours.evaporate(evap, active=active)
            elif op == "excess":
                got = ours.get_excess(active=active)
            elif op == "excess_req":
                got = ours.get_excess(req, active=active)
            elif op == "avail":
                got = ours.get_avail(active=active)
            elif op == "avail_req":
                got = ours.get_avail(req, active=active)
            else:
                got = ours.pull_outflow(active=active)
            got = torch.broadcast_to(got, (B,) + got.shape[-1:] if got.ndim > 1 else (B,))
            for b, tank in enumerate(theirs):
                if not active[b]:
                    continue
                r = dict(requests[b])
                if op == "push":
                    ref = tank.push_storage(r)
                elif op == "push_force":
                    ref = tank.push_storage(r, force=True)
                elif op == "pull":
                    ref = tank.pull_storage(r)
                elif op == "pull_all":
                    ref = tank.pull_storage({"volume": 1e6})
                elif op == "pull_ponded":
                    ref = tank.pull_ponded()
                elif op == "evaporate":
                    ref = tank.evaporate(r["volume"])
                elif op == "excess":
                    ref = tank.get_excess()
                elif op == "excess_req":
                    ref = tank.get_excess(r)
                elif op == "avail":
                    ref = tank.get_avail()
                elif op == "avail_req":
                    ref = tank.get_avail(r)
                else:
                    ref = tank.pull_outflow()
                label = f"{kind} step {step} {op} [{b}]"
                if isinstance(ref, dict):
                    w.vqip(label, got[b], ref)
                else:
                    w.value(label, got[b], ref)
            storage = torch.broadcast_to(ours.storage, (B, 1 + PQ.K))
            for b, tank in enumerate(theirs):
                w.vqip(f"{kind} step {step} storage [{b}]", storage[b], tank.storage)
                hits_zero += tank.storage["volume"] == 0
        assert hits_zero > 0  # the exact V == 0 branch of pull_storage was reached
        assert w.worst == 0.0


def wsimod_register(queue: dict, length: int):
    rows = [
        queue.get(i, {"volume": 0.0, "phosphate": 0.0, "temperature": 0.0})
        for i in range(length + 1)
    ]
    return rows, [i in queue for i in range(length + 1)]


@pytest.mark.parametrize("n", [0, 2])
def test_queue_tank_equals_wsimod_on_random_pushes(wsimod_tanks, n):
    g = torch.Generator().manual_seed(60 + n)
    times = range(4)
    theirs, caps = [], []
    for _ in range(B):
        cap = float(torch.rand((), generator=g, dtype=torch.float64)) * 20 + 5
        theirs.append(
            wsimod_tanks.QueueTank(
                number_of_timesteps=n, capacity=cap, area=1.0, datum=0.0, initial_storage=0
            )
        )
        caps.append(cap)
    ours = QueueTank(
        PQ,
        number_of_timesteps=n,
        push_times=times,
        capacity=torch.tensor(caps, dtype=torch.float64),
        area=1.0,
        datum=0.0,
        initial_storage=0.0,
        owner="N",
        path="sewer_tank",
    )
    length = ours.internal_arc.length
    assert length == n + 3
    ops = ["push", "push", "push", "push_tiny", "push_force", "pull", "pull_exact"]
    ops += ["pull_ponded", "excess", "check", "evaporate"]
    w = Worst()
    dropped = 0.0
    tiny_pushes = 0
    for d in range(60):
        ctx = day(ours, p=PQ, shape=(B,), d=d)
        for k in range(int(torch.randint(1, 6, (), generator=g))):
            op = ops[int(torch.randint(len(ops), (), generator=g))]
            time = int(torch.randint(4, (), generator=g))
            active = torch.rand(B, generator=g) < 0.85
            requests = [random_vqip(g, 6.0) for _ in range(B)]
            if op == "push_tiny":
                for r in requests:
                    r["volume"] = 1e-12
                tiny_pushes += int(active.sum())  # below FLOAT_ACCURACY (1e-11) and active
            req = stack(requests)
            if op in ("push", "push_tiny"):
                got = ours.push_storage(req, active=active, time=time)
            elif op == "push_force":
                got = ours.push_storage(req, active=active, force=True)
            elif op == "pull":
                got = ours.pull_storage(req, active=active)
            elif op == "pull_exact":
                got = ours.pull_storage_exact(req, active=active)
            elif op == "pull_ponded":
                got = ours.pull_ponded(active=active)
            elif op == "excess":
                got = ours.get_excess(active=active)
            elif op == "check":
                got = ours.push_check(req, active=active)
            else:
                got = ours.evaporate(req[..., 0], active=active)
            got = torch.broadcast_to(got, (B,) + got.shape[1:] if got.ndim > 1 else (B,))
            for b, tank in enumerate(theirs):
                if not active[b]:
                    continue
                r = dict(requests[b])
                if op in ("push", "push_tiny"):
                    ref = tank.push_storage(r, time=time)
                elif op == "push_force":
                    ref = tank.push_storage(r, force=True)
                elif op == "pull":
                    ref = tank.pull_storage(r)
                elif op == "pull_exact":
                    ref = tank.pull_storage_exact(r)
                elif op == "pull_ponded":
                    ref = tank.pull_ponded()
                elif op == "excess":
                    ref = tank.get_excess()
                elif op == "check":
                    ref = tank.push_check(r)
                else:
                    ref = tank.evaporate(r["volume"])
                label = f"n={n} day {d} call {k} {op} [{b}]"
                if isinstance(ref, dict):
                    w.vqip(label, got[b], ref)
                else:
                    w.value(label, got[b], ref)
        dropped += float(ctx.dropped[..., 0].sum())  # W49: a tiny push is storage, no drop
        ours.end_timestep()
        for tank in theirs:
            tank.end_timestep()
        storage = torch.broadcast_to(ours.storage, (B, 1 + PQ.K))
        active_storage = torch.broadcast_to(ours.active_storage, (B, 1 + PQ.K))
        register = torch.broadcast_to(ours.internal_arc.register, (B, length + 1, 1 + PQ.K))
        present = torch.broadcast_to(ours.internal_arc.present, (B, length + 1))
        for b, tank in enumerate(theirs):
            label = f"n={n} end of day {d} [{b}]"
            w.vqip(f"{label} storage", storage[b], tank.storage)
            w.vqip(f"{label} active_storage", active_storage[b], tank.active_storage)
            assert max(tank.internal_arc.queue) <= length
            rows, keys = wsimod_register(tank.internal_arc.queue, length)
            for i, row in enumerate(rows):
                w.vqip(f"{label} register[{i}]", register[b, i], row)
            assert present[b].tolist() == keys, label
            w.value(
                f"{label} flow_out",
                ours.internal_arc.flow_out.expand(B)[b],
                tank.internal_arc.flow_out,
            )
    assert dropped == 0.0
    assert tiny_pushes > 0  # the tiny pushes were reached, and none is a drop (W49)
    assert w.worst == 0.0


# --- noodl only ------------------------------------------------------------------------------
def test_water_entered_with_total_time_zero_is_active_at_once():
    tank = queue_tank(n=0)
    day(tank)
    reply = tank.push_storage(vol(5.0), active=TRUE, time=0)
    assert reply.tolist() == [0.0]
    assert tank.active_storage.tolist() == [5.0]
    assert tank.storage.tolist() == [5.0]
    assert tank.internal_arc.register.abs().sum() == 0


def test_end_timestep_shifts_and_flushes_the_internal_arc():
    tank = queue_tank(n=0, times=(0, 1))
    day(tank)
    tank.push_storage(vol(3.0), active=TRUE, time=1)
    assert tank.internal_arc.register[1].tolist() == [3.0]
    assert tank.active_storage.tolist() == [0.0]
    tank.end_timestep()  # shift: register[0] = 0 + 3, then update_queue pushes it (push_set)
    assert tank.internal_arc.register[0].tolist() == [0.0]
    assert tank.active_storage.tolist() == [3.0]
    assert tank.storage.tolist() == [3.0]


def test_a_tiny_push_is_kept_in_storage_but_never_queued():
    tank = queue_tank(n=0, times=(0, 1))
    ctx = day(tank)
    tank.push_storage(vol(1e-12), active=TRUE, time=1)
    assert tank.storage.tolist() == [1e-12]
    assert tank.internal_arc.register.abs().sum() == 0
    assert tank.internal_arc.present.tolist() == [True, True]
    assert tank.active_storage.tolist() == [0.0]
    assert ctx.dropped[..., 0] == 0.0  # W49: held in storage, so not also a drop


def test_a_tiny_push_does_not_flush_what_is_due():
    tank = queue_tank(n=0, times=(0, 1))
    day(tank)
    tank.push_storage(vol(2.0), active=TRUE, time=1)
    tank.internal_arc.end_timestep()  # slot 0 now holds 2, not yet delivered
    day(tank, d=1)
    tank.push_storage(vol(1e-12), active=TRUE)
    assert tank.active_storage.tolist() == [0.0]
    tank.push_storage(vol(1.0), active=TRUE)
    assert tank.active_storage.tolist() == [3.0]


def test_pull_ponded_on_a_queue_tank_is_limited_to_active_storage():
    tank = queue_tank(n=0, times=(0, 1), capacity=5.0)
    day(tank)
    tank.push_storage(vol(2.0), active=TRUE, force=True)  # active and stored
    tank.push_storage(vol(3.0), active=TRUE, time=1)  # stored, queued
    tank.storage = vol(8.0)  # 8 stored, of which 2 active
    assert tank.pull_ponded(active=TRUE).tolist() == [2.0]
    assert tank.storage.tolist() == [6.0]
    assert tank.active_storage.tolist() == [0.0]


def test_evaporation_from_a_queue_tank_is_limited_to_active_storage():
    tank = queue_tank(n=0, times=(0, 1))
    day(tank)
    tank.push_storage(vol(1.0), active=TRUE)
    tank.push_storage(vol(4.0), active=TRUE, time=1)
    assert tank.evaporate(torch.tensor(3.0, dtype=torch.float64), active=TRUE) == 1.0
    assert tank.storage.tolist() == [4.0]  # tanks.py:313-316: storage, not active_storage


def test_a_user_alt_queue_arc_shifts_daily_but_delivers_only_on_a_push():
    spec = NodeSpec("src", "Node", "Node", {})
    src, out = (
        Node(spec, QUANTITY_ONLY),
        Waste(NodeSpec("out", "Waste", "Waste", {}), QUANTITY_ONLY),
    )
    arc = ARC_CLASSES["AltQueueArc"](
        ArcSpec("q", "AltQueueArc", "src", "out", number_of_timesteps=2), src, out, QUANTITY_ONLY
    )
    assert isinstance(arc, AltQueueArc) and arc.length == 2
    day(src, out, arc)
    assert arc.send_push_request(vol(4.0), active=TRUE).tolist() == [0.0]
    assert arc.vqip_out.tolist() == [0.0]
    for d in (1, 2):
        arc.end_timestep()
        day(src, out, arc, d=d)
        assert arc.vqip_out.tolist() == [0.0]
    assert arc.register[0].tolist() == [4.0]  # due, waiting for a push
    arc.end_timestep()
    day(src, out, arc, d=3)
    arc.send_push_request(vol(1.0), active=TRUE)
    assert arc.vqip_out.tolist() == [4.0]
    assert arc.vqip_in.tolist() == [1.0]
    assert arc.register[2].tolist() == [1.0]
    assert arc.flow_in.item() == 1.0 / 3  # enter_arc's average flow, V / (time + 1)
    assert set(arc.stores()) == {"register", "present"}


def test_a_masked_push_changes_nothing():
    tank = queue_tank(n=0, times=(0, 1))
    day(tank, shape=(2,))
    active = torch.tensor([True, False])
    tank.push_storage(torch.tensor([[2.0], [5.0]], dtype=torch.float64), active=active, time=1)
    assert tank.storage.tolist() == [[2.0], [0.0]]
    assert tank.internal_arc.register[..., 1, 0].tolist() == [2.0, 0.0]
    assert tank.internal_arc.flow_in.tolist() == [2.0 / 2, 0.0]


def test_tank_sites_name_the_owner_and_the_attribute():
    tank = queue_tank(n=0, times=(0, 1))
    ctx = day(tank, record=True)
    tank.push_storage(vol(1.0), active=TRUE, time=1)
    tank.pull_storage(vol(1.0), active=TRUE)
    sites = [r.site for r in ctx.decisions]
    assert "node:N/sewer_tank/internal_arc/send_push_request/below_fa" in sites
    assert "node:N/sewer_tank/get_excess/headroom" in sites
    assert "node:N/sewer_tank/push_check/clip" in sites
    assert "node:N/sewer_tank/internal_arc/get_excess/push/min" in sites
    assert "node:N/sewer_tank/pull_storage/clip" in sites


def test_a_decay_tank_with_no_decays_needs_the_parents_temperature():
    parent = Node(NodeSpec("N", "Node", "Node", {}), QUANTITY_ONLY)
    tank = DecayTank(
        QUANTITY_ONLY,
        parent=parent,
        capacity=1.0,
        area=1.0,
        datum=0.0,
        initial_storage=0.5,
        owner="N",
    )
    with pytest.raises(KeyError, match="temperature"):
        tank.end_timestep()
    parent.bind({"temperature": torch.tensor(10.0)})
    tank.end_timestep()
    assert tank.storage.tolist() == [0.5]


def test_tank_stores_load_by_attribute_path():
    tank = queue_tank(n=2, times=(0,))
    stores = tank.stores()
    assert set(stores) == {
        "storage",
        "active_storage",
        "internal_arc.register",
        "internal_arc.present",
    }
    tank.load_stores({"internal_arc.present": torch.ones(3, dtype=torch.float64)})
    assert tank.internal_arc.present.dtype == torch.bool
    with pytest.raises(KeyError, match="nothing"):
        tank.load_stores({"nothing": torch.zeros(1)})


def _network(arc: ArcSpec) -> WsimodNetwork:
    return WsimodNetwork(QUANTITY_ONLY, FA, [], [], [], [], [arc])


def test_a_non_integer_travel_time_is_refused_by_name():
    bad = ArcSpec("slow", "AltQueueArc", "a", "b", number_of_timesteps=1.5)
    with pytest.raises(ValueError, match=r"arc 'slow'.*number_of_timesteps"):
        AltQueueArc.validate(bad, _network(bad))
    negative = ArcSpec("back", "AltQueueArc", "a", "b", number_of_timesteps=-1)
    with pytest.raises(ValueError, match=r"arc 'back'.*number_of_timesteps -1"):
        AltQueueArc.validate(negative, _network(negative))
    # a time-area key -1 entering a queue tank is refused naming the owner and the key
    with pytest.raises(ValueError, match=r"node:N/sewer_tank.*push time -1"):
        queue_tank(n=0, times=(0, -1))
    ok = ArcSpec("fine", "AltQueueArc", "a", "b", number_of_timesteps=2)
    AltQueueArc.validate(ok, _network(ok))


def test_queue_misuse_is_refused_by_name():
    plain = Tank(QUANTITY_ONLY, capacity=1.0, area=1.0, datum=0.0, initial_storage=0.0, owner="N")
    with pytest.raises(RuntimeError, match="node:N/tank"):
        plain.push_storage(vol(1.0), active=TRUE)
    tank = queue_tank(n=1, times=(0,))
    day(tank)
    with pytest.raises(ValueError, match="travel time 3 exceeds the register length 1"):
        tank.push_storage(vol(1.0), active=TRUE, time=2)
    with pytest.raises(TypeError, match="no request"):
        tank.get_avail(vol(1.0), active=TRUE)
    arc = tank.internal_arc
    with pytest.raises(KeyError, match="queue"):
        arc.load_stores({"queue": torch.zeros(1)})
    arc.load_stores({"register": torch.ones(2, 1), "present": torch.tensor([1.0, 0.0])})
    assert arc.present.tolist() == [True, False]
    spec = ArcSpec("q", "AltQueueArc", "N", "N", number_of_timesteps=3)
    with pytest.raises(ValueError, match="shorter than number_of_timesteps 3"):
        AltQueueArc(spec, tank, tank, QUANTITY_ONLY, length=2)


def test_a_forced_push_into_a_user_alt_queue_arc_skips_the_clip():
    src = Node(NodeSpec("src", "Node", "Node", {}), QUANTITY_ONLY)
    out = Waste(NodeSpec("out", "Waste", "Waste", {}), QUANTITY_ONLY)
    spec = ArcSpec("q", "AltQueueArc", "src", "out", capacity=1.0, number_of_timesteps=0)
    arc = AltQueueArc(spec, src, out, QUANTITY_ONLY)
    day(src, out, arc)
    assert arc.send_push_request(vol(3.0), active=TRUE, force=True).tolist() == [0.0]
    assert arc.vqip_out.tolist() == [3.0]
    # the pipe excess is now 1 - 3 = -2, transcribed unclamped (arcs.py:246-257, :384-390):
    # not_pushed = max(3 - (-2), 0) = 5, so -2 enters and 5 comes back
    assert arc.send_push_request(vol(3.0), active=TRUE).tolist() == [5.0]
    assert arc.vqip_in.tolist() == [1.0]


@pytest.mark.parametrize("n", [0, 2])
def test_user_alt_queue_arc_equals_wsimod_on_random_pushes(wsimod_tanks, n):
    """A user arc's slot 0 waits for a push, so the shift's ``new[0] = sum_vqip(old[0],
    old[1])`` meets a non-empty ``old[0]`` on the days without a push (arcs.py:660)."""
    from wsimod.arcs.arcs import AltQueueArc as WAltQueueArc
    from wsimod.nodes.nodes import Node as WNode
    from wsimod.nodes.waste import Waste as WWaste

    g = torch.Generator().manual_seed(70 + n)
    theirs = []
    for b in range(B):
        src, out = WNode(name=f"src{b}"), WWaste(name=f"out{b}")
        theirs.append(WAltQueueArc(name="q", in_port=src, out_port=out, number_of_timesteps=n))
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    out = Waste(NodeSpec("out", "Waste", "Waste", {}), PQ)
    spec = ArcSpec("q", "AltQueueArc", "src", "out", number_of_timesteps=n)
    ours = AltQueueArc(spec, src, out, PQ, length=n + 2)
    length = ours.length
    w = Worst()
    old0_full = 0
    for d in range(60):
        day(src, out, ours, p=PQ, shape=(B,), d=d)
        for k in range(int(torch.randint(0, 4, (), generator=g))):
            kind = ["push", "push", "force", "tiny"][int(torch.randint(4, (), generator=g))]
            time = int(torch.randint(3, (), generator=g))
            active = torch.rand(B, generator=g) < 0.85
            requests = [random_vqip(g, 6.0) for _ in range(B)]
            if kind == "tiny":
                for r in requests:
                    r["volume"] = 1e-12
            got = ours.send_push_request(
                stack(requests), active=active, force=kind == "force", time=time
            )
            for b, arc in enumerate(theirs):
                if active[b]:
                    full = {name: 0.0 for name in wsimod_tanks.constants.POLLUTANTS}
                    full.update(requests[b])
                    ref = arc.send_push_request(full, force=kind == "force", time=time)
                    w.vqip(f"n={n} day {d} call {k} {kind} [{b}]", got[b], ref)
        flow_in = ours.flow_in.expand(B)
        flow_out = ours.flow_out.expand(B)
        vqip_in = torch.broadcast_to(ours.vqip_in, (B, 1 + PQ.K))
        vqip_out = torch.broadcast_to(ours.vqip_out, (B, 1 + PQ.K))
        for b, arc in enumerate(theirs):
            label = f"n={n} day {d} [{b}]"
            w.value(f"{label} flow_in", flow_in[b], arc.flow_in)
            w.value(f"{label} flow_out", flow_out[b], arc.flow_out)
            w.vqip(f"{label} vqip_in", vqip_in[b], arc.vqip_in)
            w.vqip(f"{label} vqip_out", vqip_out[b], arc.vqip_out)
            old0_full += arc.queue[0]["volume"] > 0 and arc.queue[1]["volume"] > 0
        ours.end_timestep()
        for arc in theirs:
            arc.end_timestep()
        register = torch.broadcast_to(ours.register, (B, length + 1, 1 + PQ.K))
        present = torch.broadcast_to(ours.present, (B, length + 1))
        for b, arc in enumerate(theirs):
            label = f"n={n} end of day {d} [{b}]"
            assert max(arc.queue) <= length
            rows, keys = wsimod_register(arc.queue, length)
            for i, row in enumerate(rows):
                w.vqip(f"{label} register[{i}]", register[b, i], row)
            assert present[b].tolist() == keys, label
    assert old0_full > 0  # the shift summed two non-empty slots
    assert w.worst == 0.0


def test_gradients_through_the_tanks_and_the_register_are_finite():
    """The V == 0 branch of pull_storage, a tiny queue push and a masked instance stay finite:
    every division goes through a safe denominator and every branch is a detached ``where``."""
    x = torch.tensor([2.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)
    tank = Tank(PQ, capacity=5.0, area=1.0, datum=0.0, initial_storage=0.0, owner="N")
    qtank = QueueTank(
        PQ,
        number_of_timesteps=1,
        push_times=(0, 1),
        capacity=10.0,
        area=1.0,
        datum=0.0,
        initial_storage=0.0,
        owner="N",
        path="sewer_tank",
    )
    day(tank, qtank, p=PQ, shape=(3,))
    active = torch.tensor([True, True, False])
    push = torch.stack([x, 0.5 * x, 10.0 + x], dim=-1)  # instance 1 has volume 0
    tank.push_storage(push, active=active)
    pulled = tank.pull_storage(push * 0.5, active=active)  # instance 1: V == 0, empty
    tiny = torch.stack([x * 0 + 1e-12, x, x], dim=-1)  # below FLOAT_ACCURACY everywhere
    qtank.push_storage(tiny, active=active, time=1)
    qtank.push_storage(push, active=active, time=0)
    qtank.end_timestep()
    out = qtank.pull_storage(push, active=active)
    stores = (
        tank.storage.sum()
        + qtank.storage.sum()
        + qtank.active_storage.sum()
        + qtank.internal_arc.register.sum()
    )
    (grad,) = torch.autograd.grad(stores + pulled.sum() + out.sum(), x, retain_graph=True)
    assert torch.isfinite(grad).all()
    (grad,) = torch.autograd.grad(stores, x)
    assert torch.isfinite(grad).all()
    assert grad[2] == 0.0  # the masked instance's stores do not depend on its inputs
    assert grad[0] != 0.0


# --- DecayQueueTank (tanks.py:645-696) and its DecayArcAlt (arcs.py:731-805) -----------------
QUEUE_DECAYS = {
    "phosphate": {"constant": 0.3, "exponent": 1.07},  # capped above about 37.8 degC
    "temperature": {"constant": 0.01, "exponent": 1.02},  # a non-additive column decays too
}


class Parent:
    """What WSIMOD's ``DecayObj`` reads the temperature from (``core.py:464-470``, ``:485``)."""

    def __init__(self) -> None:
        self.t = 0
        self.data_input_dict = {}


def decay_queue_tank(decays=QUEUE_DECAYS, *, n=1, times=(0,), capacity=50.0, shape=()):
    """A noodl ``DecayQueueTank`` owned by node ``N``, its ``Decay`` and that node (the parent
    whose bound ``temperature`` an empty ``decays`` still needs)."""
    from noodl.apps.wsimod.decay import Decay, read_decays

    parent = Node(NodeSpec("N", "Node", "Node", {}), PQ)
    spec = read_decays("node 'N'", decays, PQ)
    decay = Decay(PQ, spec, site="node:N/tank/internal_arc") if spec.keys else None
    tank = DecayQueueTank(
        PQ,
        parent=parent,
        decay=decay,
        number_of_timesteps=n,
        push_times=times,
        capacity=capacity,
        area=1.0,
        datum=0.0,
        initial_storage=0.0,
        owner="N",
        path="tank",
    )
    return tank, decay, parent


def start_day(tank, decay, temperature, *, shape=(), d=0):
    """The day's context and the decay bound at ``temperature`` (D5: once per day)."""
    ctx = day(tank, p=PQ, shape=shape, d=d, record=True)
    if decay is not None:
        decay.bind(ctx, decay.constant, decay.exponent, torch.as_tensor(temperature, dtype=F64))
    return ctx


F64 = torch.float64
PHOSPHATE = PQ.column("phosphate")
TEMPERATURE = PQ.column("temperature")


@pytest.mark.parametrize("n", [1, 0])
def test_decay_queue_tank_equals_wsimods_on_random_sequences(wsimod_tanks, n):
    """A WSIMOD ``DecayQueueTank`` (its default ``number_of_timesteps=1``, ``tanks.py:648``, and
    0) with a dummy parent, against noodl's over 60 days of pushes at times 0-2, pulls, exact
    pulls and ends of day: after every call ``storage``, ``active_storage``, the dense register,
    ``present`` and ``total_decayed`` are bitwise WSIMOD's (X13, X30). Pushes below
    ``FLOAT_ACCURACY`` and forced pushes are among the calls, and one day in three pushes
    nothing, so slot 0 still holds water at its end of day beside other queued slots: the
    position of ``diff(slot 0)`` in ``total_decayed``'s sum is compared with WSIMOD's."""
    g = torch.Generator().manual_seed(90 + n)
    u = lambda: float(torch.rand((), generator=g, dtype=F64))  # noqa: E731
    parents = [Parent() for _ in range(B)]
    caps = [u() * 20 + 5 for _ in range(B)]
    theirs = [
        wsimod_tanks.DecayQueueTank(
            decays=QUEUE_DECAYS,
            parent=parent,
            number_of_timesteps=n,
            capacity=cap,
            area=1.0,
            datum=0.0,
            initial_storage=0,
        )
        for parent, cap in zip(parents, caps, strict=True)
    ]
    ours, decay, _ = decay_queue_tank(n=n, times=(0, 1, 2), capacity=torch.tensor(caps, dtype=F64))
    length = ours.internal_arc.length
    assert length == max(1, n + 2)
    w = Worst()
    ops = ["push", "push", "push", "push_tiny", "push_force", "pull", "pull_exact"]
    capped = 0
    full_ends = 0  # ends of day with water in slot 0 and in other slots

    def compare(label):
        arc = ours.internal_arc
        stores = {
            "storage": torch.broadcast_to(ours.storage, (B, 1 + PQ.K)),
            "active_storage": torch.broadcast_to(ours.active_storage, (B, 1 + PQ.K)),
            "total_decayed": torch.broadcast_to(arc.total_decayed, (B, 1 + PQ.K)),
        }
        register = torch.broadcast_to(arc.register, (B, length + 1, 1 + PQ.K))
        present = torch.broadcast_to(arc.present, (B, length + 1))
        for b, tank in enumerate(theirs):
            w.vqip(f"{label} [{b}] storage", stores["storage"][b], tank.storage)
            w.vqip(
                f"{label} [{b}] active_storage", stores["active_storage"][b], tank.active_storage
            )
            w.vqip(
                f"{label} [{b}] total_decayed",
                stores["total_decayed"][b],
                tank.internal_arc.total_decayed,
            )
            assert max(tank.internal_arc.queue) <= length
            rows, keys = wsimod_register(tank.internal_arc.queue, length)
            for i, row in enumerate(rows):
                w.vqip(f"{label} [{b}] register[{i}]", register[b, i], row)
            assert present[b].tolist() == keys, label

    for d in range(60):
        temperatures = [-5 + 50 * u() for _ in range(B)]
        for parent, temperature in zip(parents, temperatures, strict=True):
            parent.t = d
            parent.data_input_dict[("temperature", d)] = temperature
        ctx = start_day(ours, decay, temperatures, shape=(B,), d=d)
        capped += int(ctx.decisions[0].outcome.sum())  # cap:phosphate, the first decision
        quiet = d % 3 == 2  # no push: slot 0 keeps what the last end of day moved there
        for k in range(int(torch.randint(1, 6, (), generator=g))):
            op = ops[int(torch.randint(len(ops), (), generator=g))]
            if quiet:
                op = ("pull", "pull_exact")[k % 2]
            time = int(torch.randint(3, (), generator=g))
            active = torch.rand(B, generator=g) < 0.85
            requests = [random_vqip(g, 6.0) for _ in range(B)]
            if op == "push_tiny":
                for r in requests:
                    r["volume"] = 1e-12
            req = stack(requests)
            if op in ("push", "push_tiny"):
                got = ours.push_storage(req, active=active, time=time)
            elif op == "push_force":
                got = ours.push_storage(req, active=active, force=True)
            elif op == "pull":
                got = ours.pull_storage(req, active=active)
            else:
                got = ours.pull_storage_exact(req, active=active)
            got = torch.broadcast_to(got, (B, 1 + PQ.K))
            for b, tank in enumerate(theirs):
                if not active[b]:
                    continue
                r = dict(requests[b])
                if op in ("push", "push_tiny"):
                    ref = tank.push_storage(r, time=time)
                elif op == "push_force":
                    ref = tank.push_storage(r, force=True)
                elif op == "pull":
                    ref = tank.pull_storage(r)
                else:
                    ref = tank.pull_storage_exact(r)
                w.vqip(f"n={n} day {d} call {k} {op} [{b}] reply", got[b], ref)
            compare(f"n={n} day {d} after call {k} {op}")
        held = torch.broadcast_to(ours.internal_arc.register[..., 0], (B, length + 1)) > 0
        # n = 0 leaves nothing beyond slot 1 on a day without pushes; two terms add the same in
        # either order, so the slot-0 position is discriminated by n = 1 (and the direct test)
        full_ends += int((held[:, 0] & (held[:, 1:].sum(-1) >= (2 if n else 1))).sum())
        ours.end_timestep()
        for tank in theirs:
            tank.end_timestep()
        compare(f"n={n} end of day {d}")
    assert capped > 0  # the cap bound on some hot days
    assert full_ends > 0
    assert w.worst == 0.0


def test_the_end_of_day_decay_order_puts_slot_zero_last():
    """X30 (``arcs.py:798-805``): ``total_decayed`` is reset, then ``diff(slot 1) ... diff(slot
    L)`` and ``diff(slot 0)`` last. With 2.8, 2.2 and 0.7 kg of phosphate in slots 0-2 and a
    factor of exactly 0.1, ``(0 + 0.22 + 0.07) + 0.28`` is 0.5700000000000001 while the
    slot-0-first order gives 0.57, so the test discriminates."""
    tank, decay, _ = decay_queue_tank(
        {"phosphate": {"constant": 0.1, "exponent": 1.0}}, n=0, times=(0, 2)
    )
    start_day(tank, decay, 15.0)
    arc = tank.internal_arc
    rows = [PQ.to_vqip({"volume": 1.0, "phosphate": x}) for x in (2.8, 2.2, 0.7)]
    arc.register, arc.present = torch.stack(rows), torch.tensor([True, True, True])
    tank.end_timestep()
    got = arc.total_decayed[PHOSPHATE].item()
    diffs = [x * 0.1 for x in (2.8, 2.2, 0.7)]
    assert got == (0.0 + diffs[1] + diffs[2]) + diffs[0] == 0.5700000000000001
    assert (0.0 + diffs[0] + diffs[1]) + diffs[2] == 0.57  # slot 0 first


def test_two_empty_slots_with_different_temperatures_merge_as_wsimods(wsimod_tanks):
    """X46 reached through WSIMOD's own calls: a push into a full ``DecayQueueTank`` enters,
    under key 3, a zero-volume row that keeps the request's temperature (13); after slot 0 is
    flushed (``update_queue``, as ``distribute`` does), each end of day merges two zero-volume
    slots with different temperatures, where ``sum_vqip``'s argument order decides the result
    (``arcs.py:805``). Every store is bitwise WSIMOD's."""
    decays = {"phosphate": {"constant": 0.1, "exponent": 1.0}}
    parent = Parent()
    theirs = wsimod_tanks.DecayQueueTank(
        decays=decays,
        parent=parent,
        number_of_timesteps=1,
        capacity=1.0,
        area=1.0,
        datum=0.0,
        initial_storage=0,
    )
    ours, decay, _ = decay_queue_tank(decays, n=1, times=(0, 1, 2), capacity=1.0)
    arc, length = ours.internal_arc, ours.internal_arc.length
    w = Worst()

    def compare(label):
        w.vqip(f"{label} storage", ours.storage, theirs.storage)
        w.vqip(f"{label} total_decayed", arc.total_decayed, theirs.internal_arc.total_decayed)
        rows, keys = wsimod_register(theirs.internal_arc.queue, length)
        for i, row in enumerate(rows):
            w.vqip(f"{label} register[{i}]", arc.register[i], row)
        assert arc.present.tolist() == keys, label

    for d in range(4):
        parent.t = d
        parent.data_input_dict[("temperature", d)] = 15.0
        start_day(ours, decay, 15.0, d=d)
        if d == 0:
            for push, time in (
                ({"volume": 1.0, "phosphate": 0.5, "temperature": 10.0}, 0),  # fills the tank
                ({"volume": 1.0, "phosphate": 0.5, "temperature": 13.0}, 2),  # rejected: V 0
            ):
                w.vqip(
                    f"day 0 push at {time}",
                    ours.push_storage(PQ.to_vqip(push), active=TRUE, time=time),
                    theirs.push_storage(dict(push), time=time),
                )
            assert arc.register[3].tolist() == [0.0, 0.0, 13.0]
        arc.update_queue(active=TRUE)  # distribute's (storage.py:352)
        theirs.internal_arc.update_queue(direction="push")
        compare(f"day {d} flushed")
        ours.end_timestep()
        theirs.end_timestep()
        compare(f"end of day {d}")
        if d == 2:  # sum_vqip(decay(old[1]) = (0, 0, 13), decay(old[0]) = (0, 0, 0)) keeps 13
            assert arc.register[0].tolist() == [0.0, 0.0, 13.0]
    assert w.worst == 0.0


def test_the_decay_reaches_storage_a_day_late():
    """X13: at the end of day d ``storage`` loses the ``total_decayed`` held before the reset,
    day d's entry decay plus day d - 1's end-of-day decay; the queue holds 5 less every diff
    applied so far. A factor of exactly 0.1 (``1.0 ** (T - 20)`` is 1)."""
    tank, decay, _ = decay_queue_tank({"phosphate": {"constant": 0.1, "exponent": 1.0}})
    arc = tank.internal_arc
    start_day(tank, decay, 15.0)
    tank.push_storage(PQ.to_vqip({"volume": 10.0, "phosphate": 5.0}), active=TRUE, time=0)
    assert tank.storage[PHOSPHATE].item() == 5.0  # the undecayed push (tanks.py:536-538)
    entry = arc.total_decayed[PHOSPHATE].item()
    assert entry == 0.5  # decayed on entry (arcs.py:769)
    assert arc.present.tolist() == [True, True]  # under key 1: number_of_timesteps is 1
    end_of_day = [0.0]  # the end-of-day diffs, day -1 has none
    applied = entry
    for d in range(4):
        if d:
            start_day(tank, decay, 15.0, d=d)
        held = arc.total_decayed[PHOSPHATE].item()
        assert held == (entry if d == 0 else 0.0) + end_of_day[-1]
        before = tank.storage[PHOSPHATE].item()
        tank.end_timestep()
        assert tank.storage[PHOSPHATE].item() == before - held  # tanks.py:694
        end_of_day.append(arc.total_decayed[PHOSPHATE].item())
        applied += end_of_day[-1]
        queued = arc.register[..., PHOSPHATE].sum().item()
        assert queued == pytest.approx(5.0 - applied, rel=1e-15, abs=0.0)
        # the lag: storage is the queue of the day before
        assert tank.storage[PHOSPHATE].item() == pytest.approx(queued + end_of_day[-1], rel=1e-15)
    assert end_of_day[1:] == pytest.approx([0.45, 0.405, 0.3645, 0.32805], rel=1e-15)
    assert tank.active_storage.tolist() == [0.0, 0.0, 0.0]  # no update_queue at the end of day


def test_the_end_of_day_merge_is_swapped():
    """``arcs.py:805``: the new slot 0 is ``sum_vqip(decay(old[1]), decay(old[0]))``, the other
    order from ``AltQueueArc``'s ``sum_vqip(old[0], old[1])`` (``:660``). The blend
    ``(t2 V2 + t1 V1) / V`` is symmetric in IEEE arithmetic, so the order shows only where the
    merged volume is 0 and ``sum_vqip`` keeps its first argument's ``temperature``
    (``core.py:109-114``): two drained slots (``pull_set_active`` leaves a row's
    non-additive values behind) with different temperatures."""
    from noodl.apps.wsimod.vqip import sum_vqip

    tank, decay, _ = decay_queue_tank({"phosphate": {"constant": 0.1, "exponent": 1.07}})
    start_day(tank, decay, 23.0)
    arc = tank.internal_arc
    for old0, old1 in (
        (
            {"volume": 0.0, "phosphate": 0.0, "temperature": 7.0},
            {"volume": 0.0, "phosphate": 0.0, "temperature": 13.0},
        ),
        (
            {"volume": 2.0, "phosphate": 0.3, "temperature": 7.1},
            {"volume": 3.0, "phosphate": 0.7, "temperature": 13.3},
        ),
    ):
        old = torch.stack([PQ.to_vqip(old0), PQ.to_vqip(old1)])
        arc.register, arc.present = old.clone(), torch.tensor([True, True])
        tank.end_timestep()
        d0, d1 = decay.make_decay(old[0])[0], decay.make_decay(old[1])[0]
        assert torch.equal(arc.register[0], sum_vqip(PQ, d1, d0))
        assert torch.equal(arc.register[1], torch.zeros(1 + PQ.K, dtype=F64))
        if old0["volume"] == 0:  # the test discriminates: the other order keeps 7.0
            assert arc.register[0, TEMPERATURE].item() == 13.0
            assert sum_vqip(PQ, d0, d1)[TEMPERATURE].item() == 7.0
        else:  # a positive volume blends the same in either order
            assert torch.equal(sum_vqip(PQ, d0, d1), sum_vqip(PQ, d1, d0))


def test_water_reaching_slot_zero_waits_for_the_next_push():
    """Plan Review Focus 1, X13: water that the end of day d moves to slot 0 is not flushed
    (``DecayQueueTank._end_timestep`` has no ``update_queue``). With no push on day d + 1 it
    stays there, decays again at the end of day d + 1, and joins ``active_storage`` only at the
    owner's ``update_queue`` (``QueueGroundwater.distribute``, ``storage.py:352``)."""
    tank, decay, _ = decay_queue_tank({"phosphate": {"constant": 0.1, "exponent": 1.0}})
    arc = tank.internal_arc
    start_day(tank, decay, 15.0)
    tank.push_storage(PQ.to_vqip({"volume": 4.0, "phosphate": 2.0}), active=TRUE, time=0)
    tank.end_timestep()  # day 0: key 1 moves to slot 0, decayed
    assert arc.register[0].tolist() == pytest.approx([4.0, 1.62, 0.0], rel=1e-15)
    start_day(tank, decay, 15.0, d=1)  # day 1: no push
    tank.end_timestep()
    slot0 = arc.register[0, PHOSPHATE].item()
    assert slot0 == pytest.approx(1.62 * 0.9, rel=1e-15)  # decayed again in slot 0
    assert tank.active_storage.tolist() == [0.0, 0.0, 0.0]
    start_day(tank, decay, 15.0, d=2)
    arc.update_queue(active=TRUE)  # distribute's (storage.py:352)
    assert tank.active_storage.tolist() == [4.0, slot0, 0.0]
    assert arc.register[0].tolist() == [0.0, 0.0, 0.0]


def test_active_storage_never_decays():
    """The TODO at ``tanks.py:692-693``: ``end_timestep`` decays the queue only. Water that
    entered with total time 0 (``number_of_timesteps`` 0) decayed once, on entry."""
    tank, decay, _ = decay_queue_tank({"phosphate": {"constant": 0.1, "exponent": 1.0}}, n=0)
    start_day(tank, decay, 15.0)
    tank.push_storage(PQ.to_vqip({"volume": 4.0, "phosphate": 2.0}), active=TRUE, time=0)
    active = tank.active_storage.clone()
    assert active.tolist() == pytest.approx([4.0, 1.8, 0.0], rel=1e-15)
    for d in range(3):
        start_day(tank, decay, 15.0, d=d)
        tank.end_timestep()
        assert torch.equal(tank.active_storage, active)
    assert torch.equal(tank.storage, active)  # the entry decay, extracted on day 0


def test_a_decay_queue_tank_with_no_decays_needs_the_parents_temperature():
    """``make_decay`` reads the temperature even over an empty ``decays`` (``core.py:484-487``)."""
    tank, decay, parent = decay_queue_tank({})
    assert decay is None
    day(tank, p=PQ)
    with pytest.raises(KeyError, match="temperature"):
        tank.push_storage(PQ.to_vqip({"volume": 1.0}), active=TRUE, time=0)
    parent.bind({"temperature": torch.tensor(10.0, dtype=F64)})
    tank.push_storage(PQ.to_vqip({"volume": 1.0, "phosphate": 0.5}), active=TRUE, time=0)
    tank.end_timestep()
    assert tank.internal_arc.register[0].tolist() == [1.0, 0.5, 0.0]
    assert tank.storage.tolist() == [1.0, 0.5, 0.0]


# --- the user DecayArcAlt (arcs.py:731-805) ---------------------------------------------------
def test_decay_arc_alt_matches_wsimod_on_irregular_pushes(wsimod_tanks):
    """A user ``DecayArcAlt`` (``n = 2``) out of a node whose temperature it reads
    (``core.py:464-470``): irregular pushes at travel times 0-2, tiny and forced ones among them,
    ``temperature`` a blended pollutant that decays too. Every reply, the register, ``present``,
    ``vqip_out`` and ``total_decayed`` are WSIMOD's bitwise every day, and each day's
    ``"DecayArcAlt.decayed"`` term is WSIMOD's ``diff``s of the day (its pushes, then its end of
    day) summed from empty."""
    from wsimod.arcs.arcs import DecayArcAlt as WDecayArcAlt
    from wsimod.nodes.nodes import Node as WNode
    from wsimod.nodes.waste import Waste as WWaste

    from noodl.apps.wsimod.arcs import DecayArcAlt
    from noodl.apps.wsimod.decay import read_decays

    n = 2
    g = torch.Generator().manual_seed(171)
    u = lambda: float(torch.rand((), generator=g, dtype=F64))  # noqa: E731
    theirs, diffs = [], [[] for _ in range(B)]
    for b in range(B):
        src = WNode(name=f"src{b}", data_input_dict={})
        arc = WDecayArcAlt(
            name="q",
            in_port=src,
            out_port=WWaste(name=f"out{b}"),
            number_of_timesteps=n,
            decays={k: dict(v) for k, v in QUEUE_DECAYS.items()},
        )
        original = arc.generic_temperature_decay

        def capture(t, d, temperature, _original=original, _b=b):
            out, diff = _original(t, d, temperature)
            diffs[_b].append(diff)
            return out, diff

        arc.generic_temperature_decay = capture
        theirs.append((src, arc))
    src = Node(NodeSpec("src", "Node", "Node", {}), PQ)
    out = Waste(NodeSpec("out", "Waste", "Waste", {}), PQ)
    spec = ArcSpec(
        "q",
        "DecayArcAlt",
        "src",
        "out",
        number_of_timesteps=n,
        decays=read_decays("arc 'q'", QUEUE_DECAYS, PQ),
    )
    ours = DecayArcAlt(spec, src, out, PQ, length=n + 2)
    assert ours.user and ours.decay is not None and ours.site == "arc:q"
    length = ours.length
    w = Worst()
    blended = 0  # pushes into a key that already held water: the temperature blends
    for d in range(50):
        temperatures = [-5 + 50 * u() for _ in range(B)]
        ctx = day(src, out, ours, p=PQ, shape=(B,), d=d)
        src.bind({"temperature": torch.tensor(temperatures, dtype=F64)})
        ours.bind_decay(ours.defaults["decays.constant"], ours.defaults["decays.exponent"])
        for b, (w_src, _) in enumerate(theirs):
            w_src.t = d
            w_src.data_input_dict[("temperature", d)] = temperatures[b]
            diffs[b].clear()
        for k in range(int(torch.randint(0, 4, (), generator=g))):
            kind = ["push", "push", "force", "tiny"][int(torch.randint(4, (), generator=g))]
            time = int(torch.randint(3, (), generator=g))
            active = torch.rand(B, generator=g) < 0.85
            requests = [random_vqip(g, 6.0) for _ in range(B)]
            if kind == "tiny":
                for r in requests:
                    r["volume"] = 1e-12
            present = torch.broadcast_to(ours.present, (B, length + 1))[:, time + n]
            blended += int((present & active).sum())
            got = ours.send_push_request(
                stack(requests), active=active, force=kind == "force", time=time
            )
            for b, (_, arc) in enumerate(theirs):
                if active[b]:
                    ref = arc.send_push_request(dict(requests[b]), force=kind == "force", time=time)
                    w.vqip(f"day {d} call {k} {kind} [{b}]", got[b], ref)
        vqip_out = torch.broadcast_to(ours.vqip_out, (B, 1 + PQ.K))
        for b, (_, arc) in enumerate(theirs):
            w.vqip(f"day {d} [{b}] vqip_out", vqip_out[b], arc.vqip_out)
        ours.end_timestep()
        for _, arc in theirs:
            arc.end_timestep()
        register = torch.broadcast_to(ours.register, (B, length + 1, 1 + PQ.K))
        present = torch.broadcast_to(ours.present, (B, length + 1))
        total = torch.broadcast_to(ours.total_decayed, (B, 1 + PQ.K))
        term = torch.broadcast_to(ctx.terms["DecayArcAlt.decayed"], (B, 1 + PQ.K))
        for b, (_, arc) in enumerate(theirs):
            label = f"end of day {d} [{b}]"
            rows, keys = wsimod_register(arc.queue, length)
            for i, row in enumerate(rows):
                w.vqip(f"{label} register[{i}]", register[b, i], row)
            assert present[b].tolist() == keys, label
            w.vqip(f"{label} total_decayed", total[b], arc.total_decayed)
            reference = 0.0
            for diff in diffs[b]:
                reference = reference + diff["phosphate"]
            w.value(f"{label} term", term[b, PHOSPHATE], reference)
    assert blended > 0
    assert w.worst == 0.0
