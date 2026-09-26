"""WSIMOD parity where arcs genuinely BIND: capacity, receiver headroom, and several
requests competing for one arc's remaining capacity within a timestep.

`test_wsimod_parity.py` replays WSIMOD's two shipped demos, where no arc's own capacity
ever binds. This module closes that gap two ways, both with WSIMOD 0.8.1's own `Arc` /
`Node` / `Storage` / `Waste` classes as the reference (nothing of WSIMOD is reimplemented;
the harness `_wsimod_reference.capture_events` records what WSIMOD actually did):

1. **Scripted cases, run live against WSIMOD** (fast, seconds). Small networks built from
   WSIMOD's own node classes, driven by WSIMOD's own request primitives
   (`Node.push_distributed`, `Node.pull_distributed`, and a `Storage.distribute`-style
   pull-push-return) for a handful of timesteps. Each case asserts from WSIMOD's OWN
   output that the bound it targets actually cut a request (non-vacuousness), then
   replays the captured per-arc requests through `CapacitatedTransferLayer` at
   `dt = 86400 s` (WSIMOD's daily volumes turned into m3/s rates, so the layer's
   volume/rate conversion is exercised too) and compares realised volumes, and storage
   wherever a node's storage changes only through the captured arcs.
2. **`quickstart_tight` fixture** -- WSIMOD's own `quickstart_demo` with three arc
   capacities lowered (`runoff` 0.03, `baseflow` 0.04, `percolation` 0.08), captured by
   `scripts/regenerate_wsimod_fixtures.py`. Over 1456 days `baseflow` is asked for more
   than its capacity at 1085 timesteps and WSIMOD's own arc clip cuts each to the
   capacity; `runoff` and `percolation` run AT capacity at 866 and 552 timesteps, but
   there WSIMOD's `Land` node already sizes its requests from `send_push_check` (which
   includes `capacity - flow_in`), so request == capacity and the clip sits exactly on
   its kink. All six arcs are compared, none excluded.

**WSIMOD arc semantics covered.** `Arc.get_excess` clips each request against
`capacity - flow_in`, where `flow_in` ACCUMULATES every push AND pull realised on the arc
since the last `end_timestep`. Requests within a timestep are therefore served
first-come-first-served: the per-EVENT split is order dependent, but the per-timestep
AGGREGATE is `min(sum_k r_k, capacity)` whatever the order (induction on
`F_k = F_{k-1} + min(r_k, C - F_{k-1})`), and that aggregate is exactly what one
`CapacitatedTransferLayer.step` computes from the summed request. Likewise a single
in-arc filling a `Storage` gives `min(sum_k r_k, capacity, headroom)`. Covered: a push
clipped by arc capacity (single and several requests per timestep, including an
order-permutation check); push and pull sharing ONE arc's capacity in the same timestep
(both orders); receiver-headroom clipping with storage evolving over several timesteps;
and WSIMOD's own preference-weighted `push_distributed` / `pull_distributed`, whose
unequal preferences make a round's request exceed an arc's capacity so the arc clips
inside WSIMOD's `MAXITER` loop.

**What is NOT the same, pinned rather than hidden.**
- Several arcs pushing into ONE node with too little headroom: WSIMOD serves them
  first-come-first-served in call order; `CapacitatedTransferLayer` shares the headroom
  preference-proportionally and order-free. Totals and storage agree; the per-arc split
  does not (`test_competing_pushes_into_one_node_...` pins both). WSIMOD's own
  proportional sharing lives in the SENDER's or PULLER's request sizing
  (`push_distributed`/`pull_distributed`, driven by checks), i.e. in the captured
  requests themselves, not at the receiver.
- The layer uses start-of-step headroom: an outflow in the same step does not free
  headroom for an inflow. WSIMOD does free it if the outflow is called first; the
  scripted cases call inflows first.
- A bottleneck DOWNSTREAM of a pass-through WSIMOD `Node` propagates back to the arcs
  feeding it within the timestep (the node forwards each push immediately and returns
  what it cannot forward); the layer is receiver-local and does not. Measured on
  `quickstart_demo` with `catchment_outflow` lowered to 0.1: that arc itself replays
  exactly, but `baseflow` and `storm_outflow` then differ by up to 0.24 and 0.04. The
  `quickstart_tight` overrides are chosen on arcs whose receivers accept everything.
- Source availability on a pull (`in_port.pull_check`) is WSIMOD node science the layer
  does not model; the scripted pulls stay within the source's storage.
- `QueueArc`/`DecayArc` travel time and decay, `force=True` pushes, and pollutant
  vectors are not compared: the layer claims none of them (the capacitated docs page
  lists species transport on this layer as out of scope).

**Tolerance.** Both sides are float64 and do the same arithmetic up to association
order (WSIMOD: `requested - (requested - (capacity - flow_in))`; the layer: `min` of the
summed request, divided by and multiplied back by `dt`). Each realised value is a few
roundings of quantities no larger than the case's scale (largest request, capacity or
storage ceiling), so errors are a small multiple of `eps * scale` (~1e-15 relative).
`RTOL = 1e-12` of scale sits ~4500 eps above that and ~1e11 below the O(1)-relative error
of a clip taken at the wrong bound or in the wrong order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

pytest.importorskip("wsimod")

from wsimod.arcs.arcs import Arc  # noqa: E402
from wsimod.core import constants  # noqa: E402
from wsimod.nodes.nodes import Node  # noqa: E402
from wsimod.nodes.storage import Storage  # noqa: E402
from wsimod.nodes.waste import Waste  # noqa: E402
from wsimod.orchestration.model import Model  # noqa: E402

from noodl.layers.capacitated import CapacitatedTransferLayer  # noqa: E402
from noodl.topology import Network  # noqa: E402
from tests.verification._wsimod_reference import capture_events  # noqa: E402

F64 = torch.float64
DT = 86400.0  # WSIMOD's daily timestep, s
RTOL = 1e-12
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "wsimod"


# ---------------------------------------------------------------------------
# Scripted WSIMOD runs
# ---------------------------------------------------------------------------


class _Run:
    """WSIMOD nodes/arcs registered on a bare `Model` so `capture_events` can wrap them,
    driven for `len(schedule)` timesteps by `schedule[t]()`."""

    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs
        self.model = Model()
        for n in nodes:
            self.model.nodes[n.name] = n
        for a in arcs:
            self.model.arcs[a.name] = a

    def run(self, schedule):
        self.model.dates = list(range(len(schedule)))
        storage = []
        with capture_events(self.model) as events:
            for t, actions in enumerate(schedule):
                for n in self.nodes:
                    n.t = t
                actions()
                storage.append(
                    {n.name: n.tank.storage["volume"] for n in self.nodes if isinstance(n, Storage)}
                )
                for a in self.arcs:
                    a.end_timestep()
                for n in self.nodes:
                    n.end_timestep()
        self.events = pd.DataFrame(events)
        self.storage = storage
        return self


def _vqip(node, volume):
    v = node.empty_vqip()
    v["volume"] = volume
    return v


def _push(node, volume):
    node.push_distributed(_vqip(node, volume))


def _pull(node, volume):
    node.pull_distributed(_vqip(node, volume))


def _distribute(node: Storage, volume):
    """`Storage.distribute` for a given volume rather than the whole tank: take it from
    the tank, `push_distributed` it, return what was not pushed with `force=True` --
    WSIMOD's own pattern, so the source tank loses exactly what its arcs realised."""
    water = node.tank.pull_storage(_vqip(node, volume))
    retained = node.push_distributed(water)
    node.tank.push_storage(retained, force=True)


def _aggregate(run: _Run):
    """Per (arc, t): summed requested and realised volumes, as [T, E] tensors in the
    run's arc order."""
    names = [a.name for a in run.arcs]
    g = run.events.groupby(["t", "arc"])[["requested", "realised"]].sum()
    n_t = len(run.storage)
    req = torch.zeros(n_t, len(names), dtype=F64)
    real = torch.zeros(n_t, len(names), dtype=F64)
    for (t, arc), row in g.iterrows():
        req[t, names.index(arc)] = row["requested"]
        real[t, names.index(arc)] = row["realised"]
    return req, real


def _replay(run: _Run, req: torch.Tensor):
    """Replays the summed per-(arc, t) request volumes through the hard-clip layer as
    rates at `DT`. Returns realised volumes [T, E] and storage [T, N]."""
    net = Network()
    for n in run.nodes:
        net.add_node(n.name)
    for a in run.arcs:
        net.add_edge(a.in_port.name, a.out_port.name, kind="link", name=a.name)
    inf = float("inf")
    s_max = torch.tensor(
        [n.tank.capacity if isinstance(n, Storage) else inf for n in run.nodes], dtype=F64
    )
    s = torch.tensor(
        [n.initial_storage if isinstance(n, Storage) else 0.0 for n in run.nodes], dtype=F64
    )
    c_arc = torch.tensor([a.capacity for a in run.arcs], dtype=F64) / DT
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    flows, states = [], []
    for t in range(req.shape[0]):
        s, f = layer.step(s, {"cap.requests": req[t] / DT}, dt=DT)
        flows.append(f * DT)
        states.append(s.clone())
    return torch.stack(flows), torch.stack(states)


def _scale(run: _Run, req):
    """Largest magnitude in the case: requests, BOUNDED arc capacities and storage
    ceilings. WSIMOD's `UNBOUNDED_CAPACITY` (1e15) is a sentinel, not a magnitude any
    arithmetic here touches -- letting it in would inflate the tolerance to ~1e3."""
    bounded = [a.capacity for a in run.arcs if a.capacity < constants.UNBOUNDED_CAPACITY]
    bounded += [n.tank.capacity for n in run.nodes if isinstance(n, Storage)]
    return max([req.abs().max().item(), *bounded])


def _assert_parity(run: _Run, *, compare_storage=()):
    req, real = _aggregate(run)
    f, s = _replay(run, req)
    atol = RTOL * _scale(run, req)
    err = (f - real).abs().max().item()
    print(f"realised volume max error {err:.3e} (atol {atol:.3e})")
    assert err <= atol, f"realised volume max error {err:.3e} > {atol:.3e}"
    names = [n.name for n in run.nodes]
    for name in compare_storage:
        ws = torch.tensor([st[name] for st in run.storage], dtype=F64)
        s_err = (s[:, names.index(name)] - ws).abs().max().item()
        print(f"{name} storage max error {s_err:.3e}")
        assert s_err <= atol, f"{name} storage max error {s_err:.3e} > {atol:.3e}"
    return req, real, f


def _events_at(run, arc, t):
    e = run.events
    return e[(e.arc == arc) & (e.t == t)]


# ---------------------------------------------------------------------------
# Arc capacity
# ---------------------------------------------------------------------------


def _push_capacity_run(t2_order):
    src, out = Node(name="src"), Waste(name="out")
    arc = Arc(name="a", in_port=src, out_port=out, capacity=10.3)
    schedule = [
        lambda: _push(src, 15.7),
        lambda: _push(src, 4.2),
        lambda: [_push(src, v) for v in t2_order],
        lambda: _push(src, 10.3),
        lambda: [_push(src, v) for v in (0.0, 12.9)],
        lambda: [_push(src, v) for v in (2.0, 2.0, 2.0)],
    ]
    return _Run([src, out], [arc]).run(schedule)


def test_push_clipped_by_arc_capacity_matches_wsimod():
    run = _push_capacity_run((6.1, 5.9, 3.3))
    req, real, f = _assert_parity(run)
    # Non-vacuous: WSIMOD itself cut the request down to the capacity at t0, t2, t4.
    for t in (0, 2, 4):
        assert req[t, 0] > 10.3 and real[t, 0] == pytest.approx(10.3, rel=RTOL)
    # ... and at t2 the capacity was shared first-come-first-served within WSIMOD.
    assert _events_at(run, "a", 2).realised.tolist() == pytest.approx([6.1, 4.2, 0.0])


def test_push_capacity_order_changes_wsimod_split_not_the_aggregate():
    forward = _push_capacity_run((6.1, 5.9, 3.3))
    backward = _push_capacity_run((3.3, 5.9, 6.1))
    assert _events_at(forward, "a", 2).realised.tolist() == pytest.approx([6.1, 4.2, 0.0])
    assert _events_at(backward, "a", 2).realised.tolist() == pytest.approx([3.3, 5.9, 1.1])
    # The aggregate -- the only thing a single layer step sees -- is order-free, and
    # the layer matches it for both orders.
    _assert_parity(backward)


def test_push_and_pull_share_one_arcs_capacity_within_a_timestep():
    """A push (store -> out) and a pull (out pulls from store) on the SAME arc in one
    timestep accumulate into one `flow_in` and so compete for one capacity."""
    store = Storage(name="store", capacity=1200.0, initial_storage=1000.0)
    out = Waste(name="out")
    arc = Arc(name="a", in_port=store, out_port=out, capacity=8.0)
    schedule = [
        lambda: (_distribute(store, 5.0), _pull(out, 6.0)),
        lambda: (_pull(out, 6.0), _distribute(store, 5.0)),
        lambda: (_distribute(store, 9.0), _pull(out, 1.0)),
        lambda: (_pull(out, 2.0), _distribute(store, 2.0)),
    ]
    run = _Run([store, out], [arc]).run(schedule)
    _, real, _ = _assert_parity(run, compare_storage=("store",))
    e = run.events
    # Non-vacuous: the second event of t0..t2 was cut by what the first already used.
    assert e[e.t == 0].realised.tolist() == pytest.approx([5.0, 3.0])  # push, pull
    assert e[e.t == 1].realised.tolist() == pytest.approx([6.0, 2.0])  # pull, push
    assert e[e.t == 2].realised.tolist() == pytest.approx([8.0, 0.0])  # push, pull
    assert real[:3, 0].tolist() == pytest.approx([8.0, 8.0, 8.0])


# ---------------------------------------------------------------------------
# Receiver headroom, with storage evolving over time
# ---------------------------------------------------------------------------


def test_headroom_and_capacity_binding_with_evolving_storage_matches_wsimod():
    """src pushes into a finite tank (capacity 100, initially 20) over an arc of
    capacity 35; `out` drains the tank by pulling AFTER the push each step, so the
    headroom the push sees is the start-of-step headroom, as in the layer."""
    src, out = Node(name="src"), Waste(name="out")
    tank = Storage(name="tank", capacity=100.0, initial_storage=20.0)
    fill = Arc(name="fill", in_port=src, out_port=tank, capacity=35.0)
    drain = Arc(name="drain", in_port=tank, out_port=out)
    plan = [(30.0, 0.0), (50.0, 5.0), (40.0, 10.0), (25.0, 0.0), (12.0, 60.0)]  # (push, pull)
    schedule = [lambda p=p: (_push(src, p[0]), _pull(out, p[1])) for p in plan]
    # t5: two requests on the same arc, 20 + 20 against capacity 35.
    schedule.append(lambda: (_push(src, 20.0), _push(src, 20.0)))
    run = _Run([src, tank, out], [fill, drain]).run(schedule)
    _, real, _ = _assert_parity(run, compare_storage=("tank",))
    # Non-vacuous, from WSIMOD's own realised volumes: capacity binds at t1 and t5,
    # headroom (80, 90 then 100 full) binds at t2, t3 and t4.
    assert real[:, 0].tolist() == pytest.approx([30.0, 35.0, 20.0, 10.0, 0.0, 35.0])
    assert [st["tank"] for st in run.storage] == pytest.approx(
        [50.0, 80.0, 90.0, 100.0, 40.0, 75.0]
    )


# ---------------------------------------------------------------------------
# WSIMOD's own preference-weighted distribution, clipped inside its MAXITER loop
# ---------------------------------------------------------------------------


def test_push_distributed_with_unequal_preferences_clips_and_matches():
    """`push_distributed` sizes each out-arc's request as `amount * avail_i * pref_i /
    sum(avail * pref)`; with pref 5 on a capacity-4 arc the first round asks it for
    9 * 20/26 ~ 6.92 and WSIMOD's own arc clip cuts it to 4, the remainder going to
    the other arc in later rounds (several events per arc per timestep)."""
    src = Node(name="src")
    big = Storage(name="big", capacity=100.0)
    small = Storage(name="small", capacity=6.0)
    a1 = Arc(name="a1", in_port=src, out_port=big, capacity=4.0, preference=5.0)
    a2 = Arc(name="a2", in_port=src, out_port=small, capacity=10.0, preference=1.0)
    run = _Run([src, big, small], [a1, a2]).run(
        [lambda: _push(src, 9.0), lambda: _push(src, 20.0)]
    )
    req, real, _ = _assert_parity(run, compare_storage=("big", "small"))
    assert req[0, 0] == pytest.approx(9.0 * 20.0 / 26.0) and real[0, 0] == pytest.approx(4.0)
    assert real[0, 1] == pytest.approx(5.0)
    assert len(_events_at(run, "a2", 0)) == 2  # a second round went to a2
    # t1: 20 offered, only 4 (a1 capacity) + 1 (small's headroom) can go.
    assert real[1].tolist() == pytest.approx([4.0, 1.0])


def test_pull_distributed_with_unequal_preferences_clips_and_matches():
    """`pull_distributed` with pref 4 on a capacity-3 arc: round one asks it for
    10 * 12/32 = 3.75, WSIMOD's own arc clip cuts it to 3, and the deficit is pulled
    through the other arc in round two. Sources hold ample water, so the bound that
    binds is the arc's capacity, not source availability (which the layer does not
    model)."""
    sa = Storage(name="sa", capacity=60.0, initial_storage=50.0)
    sb = Storage(name="sb", capacity=60.0, initial_storage=50.0)
    puller = Node(name="puller")
    a = Arc(name="a", in_port=sa, out_port=puller, capacity=3.0, preference=4.0)
    b = Arc(name="b", in_port=sb, out_port=puller, capacity=20.0, preference=1.0)
    run = _Run([sa, sb, puller], [a, b]).run([lambda: _pull(puller, 10.0)] * 3)
    req, real, _ = _assert_parity(run, compare_storage=("sa", "sb"))
    assert req[0].tolist() == pytest.approx([3.75, 7.0])
    assert real[0].tolist() == pytest.approx([3.0, 7.0])
    assert [st["sa"] for st in run.storage] == pytest.approx([47.0, 44.0, 41.0])


# ---------------------------------------------------------------------------
# Pinned divergence: several arcs competing for one receiver's headroom
# ---------------------------------------------------------------------------


def _competing_run(first):
    s1, s2 = Node(name="s1"), Node(name="s2")
    tank = Storage(name="tank", capacity=100.0, initial_storage=88.0)
    a1 = Arc(name="a1", in_port=s1, out_port=tank)
    a2 = Arc(name="a2", in_port=s2, out_port=tank)
    order = (s1, s2) if first == "s1" else (s2, s1)
    return _Run([s1, s2, tank], [a1, a2]).run([lambda: [_push(n, 10.0) for n in order]])


@pytest.mark.parametrize("first", ["s1", "s2"])
def test_competing_pushes_into_one_node_total_matches_but_split_differs(first):
    """Two 10-unit pushes into a tank with 12 of headroom. WSIMOD: first come, first
    served, so the split follows call order. The layer: preference-proportional and
    order-free, [6, 6]. The total and the tank's storage agree exactly; the per-arc
    split is a documented semantic difference, not a tolerance question."""
    run = _competing_run(first)
    req, real = _aggregate(run)
    f, s = _replay(run, req)
    wsimod_split = [10.0, 2.0] if first == "s1" else [2.0, 10.0]
    assert real[0].tolist() == pytest.approx(wsimod_split)
    assert f[0].tolist() == pytest.approx([6.0, 6.0], rel=RTOL)
    atol = RTOL * _scale(run, req)
    assert abs(f[0].sum() - real[0].sum()).item() <= atol
    assert s[0, 2].item() == pytest.approx(run.storage[0]["tank"], rel=RTOL, abs=atol)


# ---------------------------------------------------------------------------
# quickstart_demo with three capacities lowered (committed fixture)
# ---------------------------------------------------------------------------


def test_quickstart_tight_capacities_match_wsimod_realised_flows():
    topology = json.loads((FIXTURE_DIR / "quickstart_tight_topology.json").read_text())
    events = pd.read_csv(FIXTURE_DIR / "quickstart_tight_events.csv")
    names = [a["name"] for a in topology["arcs"]]
    g = events.groupby(["t", "arc"])[["requested", "realised"]].sum().reset_index()
    n_t = int(g.t.max()) + 1
    req = torch.zeros(n_t, len(names), dtype=F64)
    real = torch.zeros(n_t, len(names), dtype=F64)
    rows = torch.tensor(g.t.tolist())
    col = torch.tensor(g.arc.map(names.index).tolist())
    req[rows, col] = torch.tensor(g.requested.tolist(), dtype=F64)
    real[rows, col] = torch.tensor(g.realised.tolist(), dtype=F64)

    cap = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    scale = req.abs().max().item()
    atol = RTOL * scale
    tight = {"runoff": 0.03, "baseflow": 0.04, "percolation": 0.08}
    for name, c in tight.items():
        assert cap[names.index(name)].item() == c
    # Non-vacuous, from WSIMOD's own numbers. `baseflow` (Groundwater pushes straight
    # down its single out-arc) is asked for MORE than its capacity at 1085 timesteps and
    # WSIMOD's own arc clip cuts every one of them to exactly the capacity. `runoff` and
    # `percolation` reach their capacity at 866 and 552 timesteps, but via `Land`'s
    # check-sized requests (`push_distributed` asks each arc for what `send_push_check`
    # says it can take), so there request == capacity and the clip sits on its kink.
    over = req > cap + atol
    at_cap = (real - cap).abs() <= atol
    n_over = dict(zip(names, over.sum(0).tolist(), strict=True))
    n_at = dict(zip(names, at_cap.sum(0).tolist(), strict=True))
    assert {k: v for k, v in n_over.items() if v} == {"baseflow": 1085}
    assert (n_at["baseflow"], n_at["runoff"], n_at["percolation"]) == (1085, 866, 552)
    assert bool(at_cap[over].all())

    net = Network()
    for node in topology["nodes"]:
        net.add_node(node["name"])
    for arc in topology["arcs"]:
        net.add_edge(arc["source"], arc["target"], kind="link", name=arc["name"])
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=cap / DT)
    s = torch.zeros(net.n, dtype=F64)
    f = torch.empty_like(req)
    for t in range(n_t):
        s, f_t = layer.step(s, {"cap.requests": req[t] / DT}, dt=DT)
        f[t] = f_t * DT
    err = (f - real).abs().max().item()
    print(f"quickstart_tight: max abs error {err:.3e} (scale {scale:.3e})")
    assert err <= atol
