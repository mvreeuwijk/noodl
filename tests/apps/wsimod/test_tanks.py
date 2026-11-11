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
from noodl.apps.wsimod.tanks import DecayTank, QueueTank, ResidenceTank, Tank
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
