"""EPANET 2.2's quality routing (`noodl.apps.water.quality`): the segment pool, the node sort,
the initial segments, the reversal rule, the quality step, an attached run and the refusals.
Parity with EPANET itself is `tests/verification/test_water_quality.py`."""

import dataclasses
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, initial_state
from noodl.apps.water.quality import (
    LINKVOL_FACTOR,
    QualityRouter,
    SegmentPool,
    quality_step,
    sort_nodes,
)
from noodl.apps.water.quality_data import QualitySource
from noodl.apps.water.simulate import water_run

F64 = torch.float64
BASE = Path(__file__).resolve().parents[2] / "data" / "water" / "quality_base.inp"


def _base(**changes):
    net = read_epanet_inp(BASE)
    if changes:
        net = dataclasses.replace(net, quality=dataclasses.replace(net.quality, **changes))
    return net


def _router(net, diagnostics=None):
    model, _, _ = build_model(net)
    router = QualityRouter(model, net.quality, inputs={}, quality_step=300, attached=False,
                           diagnostics=diagnostics)
    return model, router


def test_the_segment_pool_adds_upstream_reverses_and_recycles():
    """`addseg` puts a segment upstream of the chain's last; `reversesegs` turns the chain
    round; a used-up leading segment goes to the free list and is taken again first."""
    pool = SegmentPool(2, attached=False)
    pool.add(0, 1.0, 0.1)
    pool.add(0, 2.0, 0.2)
    pool.add(1, 3.0, 0.3)
    pool.add(0, 4.0, 0.4)
    assert pool.segments(0) == [0, 1, 3] and pool.segments(1) == [2]
    assert pool.counts() == (3, 1)
    pool.reverse(0)
    assert pool.segments(0) == [3, 1, 0]
    assert (pool.first[0], pool.last[0]) == (3, 0)
    pool.open()
    pool.remove_first(0)
    pool.close()
    assert pool.segments(0) == [1, 0] and pool.free == 3
    for k in range(20):              # past the first capacity of 16
        pool.add(1, float(k), 1.0)
    assert pool.segments(1)[1] == 3   # the recycled one first
    assert pool.capacity >= pool.size == 23
    assert pool.v[pool.segments(1)[-1]] == 19.0


def test_the_sort_follows_epanets_stack_and_adjacency_order():
    """`sortnodes`: nodes without inflow pushed in index order, popped last in first out; a
    sorted node frees its downstream nodes in its adjacency order (`buildadjlists`
    prepends: decreasing link index), so the later link's node is pushed first and popped
    last. Nodes A=0, B=1, C=2, R=3."""
    a, b, c, r = 0, 1, 2, 3
    order, breaks = sort_nodes(4, [(r, a), (a, b), (a, c)], [1, 1, 1])
    assert (order, breaks) == ([r, a, b, c], 0)
    order, _ = sort_nodes(4, [(r, a), (a, c), (a, b)], [1, 1, 1])
    assert order == [r, a, c, b]
    # a negative direction runs from N2 to N1; a stagnant link is left out
    order, _ = sort_nodes(4, [(a, r), (a, b), (a, c)], [-1, 1, 0])
    assert order == [r, a, b, c]


def test_a_cycle_is_broken_by_selectnonstacknode():
    """`L1 R->A, L2 A->B, L3 B->C, L4 C->A`: after R the stack is empty with A, B, C still
    waiting; `selectnonstacknode` takes the first node with inflow left adjacent to the
    last sorted node, A."""
    a, b, c, r = 0, 1, 2, 3
    order, breaks = sort_nodes(4, [(r, a), (a, b), (b, c), (c, a)], [1, 1, 1, 1])
    assert (order, breaks) == ([r, a, b, c], 1)


def test_initial_segments():
    """`initsegs`: one segment per pipe of `0.785398 L D^2` m3 at its end node's quality
    after `initqual`, none in the valve `V1`, the pump `PU1` or a check-valve pipe; the
    tank one at `V0 = Vmin + A (H0 - Hmin)`. TRACE at `J1`: the pipes that end at `J1`
    start at 100, every other at 0."""
    net = read_epanet_inp(BASE)
    pipes = tuple(dataclasses.replace(p, status="CV") if p.name == "P2" else p
                  for p in net.pipes)
    model, router = _router(dataclasses.replace(net, pipes=pipes))
    pool = router.pool
    names = model.link_names
    for pipe in net.pipes:
        segments = pool.segments(names.index(pipe.name))
        if pipe.name == "P2":
            assert segments == []
            continue
        assert len(segments) == 1
        assert float(pool.v[segments[0]]) == LINKVOL_FACTOR * pipe.length * pipe.diameter**2
        assert float(pool.c[segments[0]]) == 0.0      # every end node starts at 0
    assert pool.segments(names.index("V1")) == pool.segments(names.index("PU1")) == []
    tank = net.tanks[0]
    area = torch.pi * tank.diameter**2 / 4
    (seg,) = pool.segments(len(names))
    assert float(pool.v[seg]) == pytest.approx(area * tank.init_level, rel=1e-15)
    model, router = _router(_base(mode="TRACE", trace_node="J1", units="%"))
    for pipe in net.pipes:
        (seg,) = router.pool.segments(model.link_names.index(pipe.name))
        assert float(router.pool.c[seg]) == (100.0 if pipe.v == "J1" else 0.0)
    assert router.node_quality()[model.node_names.index("J1")] == 100.0
    assert float(router.node_quality().sum()) == 100.0


def _advance(router, model, p6: float, t: int) -> None:
    flow = torch.full((len(model.link_names),), 0.01, dtype=F64)
    flow[model.link_names.index("P6")] = p6
    router.advance(t, 300, flow=flow, closed=torch.zeros(flow.shape, dtype=torch.bool),
                   demand=torch.zeros(len(model.water_network.junctions), dtype=F64))


def test_segments_reverse_only_on_a_direct_sign_change():
    """`flowdirchanged`: `P6` at `+q`, then below `Q_STAGNANT` (3.15e-7 m3/s), then `-q`:
    never reversed (the direction passed through 0); `+q` then `-q`: reversed once."""
    diagnostics = {}
    model, router = _router(_base(), diagnostics)
    for t, q in enumerate((0.01, 1e-7, -0.01)):
        _advance(router, model, q, 300 * t)
    record = diagnostics["quality"][0]
    assert record["reversals"] == []
    assert len(record["node_order"]) == 3             # every direction change re-sorts
    diagnostics = {}
    model, router = _router(_base(), diagnostics)
    for t, q in enumerate((0.01, -0.01)):
        _advance(router, model, q, 300 * t)
    assert diagnostics["quality"][0]["reversals"] == [(300, "P6")]


def test_a_quality_step_of_zero_seconds_is_refused():
    """`adjustdata`: a tenth of a 5 s hydraulic step is 0 s in whole seconds; `nextqual`
    would never advance."""
    net = _base(quality_timestep=None)
    assert quality_step(net, 3600) == 360
    assert quality_step(_base(), 3600) == 300
    assert quality_step(_base(quality_timestep=7200.0), 3600) == 3600
    with pytest.raises(ValueError, match="QUALITY TIMESTEP"):
        quality_step(net, 5)
    # and through the run: a 5 s hydraulic step and no QUALITY TIMESTEP
    model, _, _ = build_model(dataclasses.replace(net, hydraulic_timestep=5.0))
    with pytest.raises(ValueError, match="QUALITY TIMESTEP"):
        water_run(model, duration=60, quality=True)


def test_an_attached_run_equals_a_detached_run():
    """Three hours of the base with `quality.initial` requiring a gradient or not: the
    same qualities, bit for bit, and the attached one carries a gradient back to the
    initial qualities."""
    net = read_epanet_inp(BASE)
    model, _, _ = build_model(net)
    initial = torch.zeros(len(model.node_names), dtype=F64)
    initial[model.node_names.index("R1")] = 1.0
    initial[model.node_names.index("T1")] = 0.3
    runs = []
    for grad in (False, True):
        state = initial_state(model)
        state["quality.initial"] = initial.clone().requires_grad_(grad)
        runs.append((water_run(model, state, duration=3 * 3600, quality=True),
                     state["quality.initial"]))
    (detached, _), (attached, leaf) = runs
    assert not detached.node_quality.requires_grad
    assert attached.node_quality.requires_grad
    assert torch.equal(detached.node_quality, attached.node_quality.detach())
    assert torch.equal(detached.link_quality, attached.link_quality.detach())
    attached.node_quality[-1, model.node_names.index("J4")].backward()
    assert leaf.grad is not None and float(leaf.grad.abs().sum()) > 0


@pytest.mark.parametrize("change, match", [
    (dict(mode="AGE", units="hrs"), "AGE"),
    (dict(global_bulk=-1e-5), "REACTIONS"),
    (dict(pipe_wall={"P1": -1e-6}), "REACTIONS"),
    (dict(sources=(QualitySource("J1", "MASS", 1.0),)), "SOURCES"),
    (dict(mixing={"T1": ("FIFO", 1.0)}), "MIXING"),
])
def test_features_of_later_tasks_are_refused_by_name(change, match):
    model, _, _ = build_model(_base(**change))
    with pytest.raises(ValueError, match=match):
        water_run(model, duration=3600, quality=True)


def test_a_batched_quality_run_and_a_network_without_quality_are_refused():
    model, _, _ = build_model(read_epanet_inp(BASE))
    state = initial_state(model)
    state["quality.initial"] = torch.zeros(2, len(model.node_names), dtype=F64)
    with pytest.raises(ValueError, match="batched"):
        water_run(model, state, duration=3600, quality=True)
    model, _, _ = build_model(dataclasses.replace(read_epanet_inp(BASE), quality=None))
    with pytest.raises(ValueError, match="QUALITY"):
        water_run(model, duration=3600, quality=True)


def test_quality_is_off_unless_asked_for():
    """Until the integration (`quality=None` follows the network later), `None` and `False`
    run no quality: the results carry none, and the hydraulics are the same."""
    model, _, _ = build_model(read_epanet_inp(BASE))
    off = water_run(model, duration=7200)
    on = water_run(model, duration=7200, quality=True)
    assert off.node_quality is None and off.link_quality is None and off.quality_units is None
    assert on.node_quality.shape == (3, len(model.node_names))
    assert on.link_quality.shape == (3, len(model.link_names))
    assert torch.equal(off.head, on.head) and torch.equal(off.flow, on.flow)


@pytest.mark.parametrize("quality", [None, False])
def test_a_quality_input_without_quality_is_refused(quality):
    """`quality.initial` in the state of a run without quality would be dropped: refused
    by name."""
    model, _, _ = build_model(read_epanet_inp(BASE))
    state = initial_state(model)
    state["quality.initial"] = torch.zeros(len(model.node_names), dtype=F64)
    with pytest.raises(ValueError, match="quality.initial"):
        water_run(model, state, duration=3600, quality=quality)
