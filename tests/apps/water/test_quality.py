"""EPANET 2.2's quality routing (`noodl.apps.water.quality`): the segment pool, the node sort,
the initial segments, the reversal rule, the quality step, an attached run and the refusals.
Parity with EPANET itself is `tests/verification/test_water_quality.py`."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, initial_drivers, initial_state
from noodl.apps.water.quality import (
    LINKVOL_FACTOR,
    QualityRouter,
    SegmentPool,
    quality_step,
    sort_nodes,
)
from noodl.apps.water.quality_data import QualitySource
from noodl.apps.water.quality_react import (
    BULK_TINY,
    EPANET_LITRES_PER_M3,
    WALL_BIG,
    bulk_rate,
    pipe_coefficients,
    tank_coefficients,
    wall_coefficients,
    wall_rate,
)
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
    (dict(mixing={"T1": ("FIFO", 1.0)}), "MIXING"),
])
def test_features_of_later_tasks_are_refused_by_name(change, match):
    model, _, _ = build_model(_base(**change))
    with pytest.raises(ValueError, match=match):
        water_run(model, duration=3600, quality=True)


# ---- sources (`findsourcequal`, `sourcequal`; `quality.c`) ------------------------------------

def _run(net, *, drivers=None, hours=3):
    model, _, _ = build_model(net)
    return model, water_run(model, drivers=drivers, duration=hours * 3600, quality=True)


def test_no_source_below_q_stagnant():
    """A MASS source at the dead end `J5` (no demand, so no outflow above `Q_STAGNANT`: `P7`
    carries a residual flow into it, and `volout / dt <= Q_STAGNANT` there): nothing is
    added, and the qualities equal the run without the source, bit for bit."""
    _, plain = _run(_base(), hours=3)
    _, sourced = _run(_base(sources=(QualitySource("J5", "MASS", 100.0),)), hours=3)
    assert torch.equal(sourced.node_quality, plain.node_quality)
    assert torch.equal(sourced.link_quality, plain.link_quality)


def test_sources_act_in_a_chemical_run_only():
    """`findsourcequal` returns 0 unless `Qualflag == CHEM`: the same source in an AGE run
    changes nothing."""
    age = dict(mode="AGE")
    _, plain = _run(_base(**age))
    _, sourced = _run(_base(sources=(QualitySource("J1", "FLOWPACED", 5.0),), **age))
    assert torch.equal(sourced.node_quality, plain.node_quality)


def test_each_source_type_adds_its_concentration_by_hand():
    """`findsourcequal` for a node whose outflow is 19 L/s over a 300 s step: MASS `c dt /
    (volout LperFT3/ft3)` (c in mg/s), SETPOINT `max(c - quality, 0)`, FLOWPACED `c`, CONCEN
    at a junction `-c demand dt / volout` with a negative demand and 0 otherwise; at the
    reservoir CONCEN is `c`. Nothing for a zero base strength or at `Q_STAGNANT`."""
    net = _base(sources=tuple(QualitySource(n, k, 2.0) for n, k in (
        ("J1", "MASS"), ("J2", "SETPOINT"), ("J3", "FLOWPACED"), ("J4", "CONCEN"),
        ("R1", "CONCEN"), ("J7", "MASS"))) + (QualitySource("J6", "MASS", 0.0),))
    model, router = _router(net)
    index = {n: i for i, n in enumerate(model.node_names)}
    router.source_c = {m: 2.0 * 3.0 for m in router.source_base}      # a factor of 3
    dt, volout = 300, 0.019 * 300
    router.demand = [0.0] * len(model.node_names)
    router.node[index["J2"]] = 1.0
    assert router._source_quality(index["J1"], volout, dt) == pytest.approx(
        6.0 * dt / (volout * EPANET_LITRES_PER_M3), rel=1e-15)
    assert router._source_quality(index["J2"], volout, dt) == 5.0       # 6 - 1
    router.node[index["J2"]] = 7.0
    assert router._source_quality(index["J2"], volout, dt) == 0.0       # max(6 - 7, 0)
    assert router._source_quality(index["J3"], volout, dt) == 6.0
    assert router._source_quality(index["J4"], volout, dt) == 0.0       # a positive demand
    router.demand[index["J4"]] = -0.003
    assert router._source_quality(index["J4"], volout, dt) == pytest.approx(
        6.0 * 0.003 * dt / volout, rel=1e-15)
    assert router._source_quality(index["R1"], volout, dt) == 6.0
    assert router._source_quality(index["J6"], volout, dt) == 0.0       # base strength 0
    assert router._source_quality(index["J7"], router.q_stagnant * dt, dt) == 0.0


def test_the_source_strength_input_replaces_the_files_and_keeps_its_gradient():
    """`drivers["quality.source_strength"]` (n_nodes, read at the source nodes) replaces the
    file's strength: the file's value gives the same qualities, bit for bit; a different
    one other qualities, and a run with it requiring a gradient equals the detached one and
    carries the gradient back."""
    net = _base(sources=(QualitySource("J1", "FLOWPACED", 0.5),))
    model, _, _ = build_model(net)
    n = len(model.node_names)
    j1 = model.node_names.index("J1")
    file = torch.zeros(n, dtype=F64)
    file[j1] = 0.5
    plain = water_run(model, duration=3 * 3600, quality=True)
    same = water_run(model, drivers={**initial_drivers(model), "quality.source_strength": file},
                     duration=3 * 3600, quality=True)
    assert torch.equal(same.node_quality, plain.node_quality)
    leaf = file.clone().requires_grad_(True)
    attached = water_run(model, drivers={**initial_drivers(model),
                                         "quality.source_strength": leaf},
                         duration=3 * 3600, quality=True)
    assert torch.equal(attached.node_quality.detach(), plain.node_quality)
    attached.node_quality[-1, model.node_names.index("J2")].backward()
    assert leaf.grad is not None and float(leaf.grad[j1]) > 0.0
    assert float(leaf.grad.abs().sum()) == pytest.approx(float(leaf.grad[j1]), rel=1e-12)
    bigger = water_run(model, drivers={**initial_drivers(model),
                                       "quality.source_strength": file * 2.0},
                       duration=3 * 3600, quality=True)
    assert float((bigger.node_quality - plain.node_quality).abs().max()) > 0.1


def test_a_source_strength_of_the_wrong_shape_is_refused():
    net = _base(sources=(QualitySource("J1", "FLOWPACED", 0.5),))
    model, _, _ = build_model(net)
    with pytest.raises(ValueError, match="quality.source_strength"):
        water_run(model, drivers={**initial_drivers(model),
                                  "quality.source_strength": torch.zeros(3, dtype=F64)},
                  duration=3600, quality=True)


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


# ---- reactions and water age (`quality_react`; `qualreact.c`) -------------------------------

def _t(*values):
    return torch.tensor(values, dtype=F64)


def test_bulk_rate_by_hand():
    """`bulkrate` (`qualreact.c`): order n `kb c^n` (order 0 `kb`); with a limiting potential
    `c1 = max(0, sgn(kb) (Climit - c))` times `c^(n-1)`; Michaelis-Menten (order < 0) `kb c /
    (Climit + sgn(kb) c)`, the denominator floored at `+-TINY` (EPANET's 1e-6 mass/ft3 in
    quality units); a negative potential is 0."""
    c = _t(0.0, 0.5, 2.0)
    kb = torch.tensor(-0.3, dtype=F64)
    assert torch.equal(bulk_rate(c, kb, 1.0, 0.0, BULK_TINY), kb * c)
    assert torch.equal(bulk_rate(c, kb, 0.0, 0.0, BULK_TINY), kb.expand(3))
    assert torch.equal(bulk_rate(c, kb, 2.0, 0.0, BULK_TINY), kb * c * c)
    assert torch.allclose(bulk_rate(c, kb, 1.5, 0.0, BULK_TINY), kb * c**1.5, rtol=1e-15,
                          atol=0.0)
    growth = bulk_rate(_t(0.5, 3.0), torch.tensor(0.5, dtype=F64), 1.0, 2.0, BULK_TINY)
    assert growth.tolist() == [0.75, 0.0]
    decay = bulk_rate(_t(1.0, 0.1), torch.tensor(-0.5, dtype=F64), 1.0, 0.2, BULK_TINY)
    assert decay.tolist() == pytest.approx([-0.4, 0.0], abs=1e-15)
    mm = bulk_rate(_t(1.0), torch.tensor(-1.0, dtype=F64), -1.0, 2.0, BULK_TINY)
    assert mm.tolist() == [-1.0]
    # |Climit + sgn(kb) c| < TINY: +TINY at 0 (SGN(0) = 1), -TINY below (then clamped at 0)
    assert BULK_TINY == 1e-6 / 28.317
    edge = bulk_rate(_t(1.0, 1.0 + BULK_TINY / 2), torch.tensor(-1.0, dtype=F64), -1.0, 1.0,
                     BULK_TINY)
    assert edge.tolist() == [-1.0 / BULK_TINY, 0.0]


_NU = 1.1e-5 * 0.3048**2          # VISCOS, m2/s
_DIFF = 1.3e-8 * 0.3048**2        # DIFFUS, m2/s


def _reynolds_flow(re, d):
    return torch.tensor(re * torch.pi * d * _NU / 4.0, dtype=F64)


def test_wall_coefficients_by_hand():
    """`piperate` (`qualreact.c`): Sherwood 2 below Re 1, Graetz's `3.65 + 0.0668 y / (1 +
    0.04 y^0.667)`, `y = (d/L) Re Sc`, below 2300, Notter-Sleicher's `0.0149 Re^0.88
    Sc^0.333` above; `kf = Sh D / d`; order 0 gives `kf`, order 1 `(4/d) kw kf / (kf +
    |kw|)`; DIFFUSIVITY 0 (`Sc = 0`): order 0 `BIG` (1e10 ft/s), order 1 `4 kw / d`; a pipe
    without a wall coefficient 0 (`ratecoeffs`)."""
    d, length, kw = 0.15, 400.0, -2e-6
    sc = _NU / _DIFF
    diameter = _t(d, d, d, d)
    lengths = _t(length, length, length, length)
    walls = _t(kw, kw, kw, 0.0)
    flow = torch.stack([-_reynolds_flow(0.5, d), _reynolds_flow(1000.0, d),
                        _reynolds_flow(1e4, d), _reynolds_flow(1e4, d)])
    y = d / length * 1000.0 * sc
    sh = [2.0, 3.65 + 0.0668 * y / (1 + 0.04 * y**0.667), 0.0149 * 1e4**0.88 * sc**0.333]
    kf = [s * _DIFF / d for s in sh]
    order0 = wall_coefficients(flow, diameter, lengths, walls, order=0.0, viscosity=_NU,
                               diffusivity=_DIFF)
    assert order0.tolist() == pytest.approx([*kf, 0.0], rel=1e-12)
    order1 = wall_coefficients(flow, diameter, lengths, walls, order=1.0, viscosity=_NU,
                               diffusivity=_DIFF)
    assert order1.tolist() == pytest.approx(
        [4.0 / d * kw * k / (k + abs(kw)) for k in kf] + [0.0], rel=1e-12)
    off0 = wall_coefficients(flow, diameter, lengths, walls, order=0.0, viscosity=_NU,
                             diffusivity=0.0)
    assert off0.tolist() == [WALL_BIG] * 3 + [0.0]
    assert WALL_BIG == pytest.approx(3.048e9, rel=1e-15)
    off1 = wall_coefficients(flow, diameter, lengths, walls, order=1.0, viscosity=_NU,
                             diffusivity=0.0)
    assert off1.tolist() == pytest.approx([4.0 * kw / d] * 3 + [0.0], rel=1e-15)


def test_wall_rate_is_mass_transfer_limited():
    """`wallrate` (`qualreact.c`), order 0: the smaller in magnitude of the mass transfer
    `sgn(kw) c kf` and the reaction `kw` (mass per m2 s, over EPANET's litres per m3: its
    `LperFT3 = 28.317` over the exact ft3), times `4/d`; order 1 `c Rc`; 0 without a wall
    coefficient."""
    assert EPANET_LITRES_PER_M3 == 28.317 / 0.3048**3
    d = _t(0.1, 0.1, 0.1)
    c = _t(1.0, 1.0, 1.0)
    kf = _t(1e-6, 1e-6, 1e-6)
    kw = _t(-1.0, -1e-6, 0.0)
    rate = wall_rate(c, d, kw, kf, order=0.0)
    assert rate.tolist() == pytest.approx([-1e-6 * 40.0, -1e-6 / EPANET_LITRES_PER_M3 * 40.0,
                                           0.0], rel=1e-15)
    rc = _t(-3e-5, -3e-5, 0.0)
    assert wall_rate(_t(0.5, 2.0, 1.0), d, kw, rc, order=1.0).tolist() == [-1.5e-5, -6e-5,
                                                                          0.0]


def test_pipe_coefficients_with_a_roughness_correlation():
    """`adjustdata` (`input1.c`): a pipe's own BULK and WALL, else the global; a pipe without
    its own WALL under ROUGHNESS CORRELATION `F` takes H-W `F / C`, D-W `F / |ln(e/d)|`
    (`e/d` in the FILE's units: mm over mm, or millifeet over inches, 83.33 e/d, in a US
    file) and C-M `F n`; `F = 0` takes the global. A tank its own TANK, else the global
    BULK."""
    net = read_epanet_inp(BASE)
    names = [p.name for p in net.pipes]
    p2 = net.pipes[names.index("P2")]
    quality = dataclasses.replace(net.quality, global_bulk=-1e-5, global_wall=-3e-6,
                                  pipe_bulk={"P3": -2e-5}, pipe_wall={"P1": -4e-6},
                                  roughness_correlation=-1.2e-4)
    kb, kw = pipe_coefficients(net, quality)
    assert kb.tolist() == [-2e-5 if n == "P3" else -1e-5 for n in names]
    assert kw[names.index("P1")] == -4e-6
    assert float(kw[names.index("P2")]) == -1.2e-4 / 120.0
    _, kw = pipe_coefficients(net, dataclasses.replace(quality, roughness_correlation=0.0))
    assert kw.tolist() == [-4e-6 if n == "P1" else -3e-6 for n in names]
    # Darcy-Weisbach, e = 0.15 mm: SI e/d, US e/d * 1000 * 0.0254 / 0.3048
    pipes = tuple(dataclasses.replace(p, roughness=0.15e-3) for p in net.pipes)
    dw = dataclasses.replace(net, headloss="D-W", pipes=pipes)
    _, kw = pipe_coefficients(dw, quality)
    ratio = 0.15e-3 / p2.diameter
    assert float(kw[names.index("P2")]) == pytest.approx(
        -1.2e-4 / abs(math.log(ratio)), rel=1e-14)
    us = dataclasses.replace(dw, options=dataclasses.replace(dw.options, flow_units="GPM"))
    _, kw = pipe_coefficients(us, quality)
    assert float(kw[names.index("P2")]) == pytest.approx(
        -1.2e-4 / abs(math.log(ratio * 1000 * 0.0254 / 0.3048)), rel=1e-14)
    assert 1000 * 0.0254 / 0.3048 == pytest.approx(83.333, rel=1e-5)
    pipes = tuple(dataclasses.replace(p, roughness=0.012) for p in net.pipes)
    cm = dataclasses.replace(net, headloss="C-M", pipes=pipes)
    _, kw = pipe_coefficients(cm, quality)
    assert float(kw[names.index("P2")]) == -1.2e-4 * 0.012
    assert kw[names.index("P1")] == -4e-6
    tanks = tank_coefficients(net, quality)
    assert tanks.tolist() == [-1e-5]
    tanks = tank_coefficients(net, dataclasses.replace(quality, tank_bulk={"T1": -7e-6}))
    assert tanks.tolist() == [-7e-6]


def test_a_check_valve_pipe_and_the_pumps_and_valves_never_react():
    """`reactpipes` reacts only `Type == PIPE` (`types.h`: `CVPIPE = 0`, `PIPE = 1`): a
    segment put into the check-valve pipe `P5`, the pump `PU1` or the valve `V1` keeps its
    quality while `P1`'s and the tank's decay (`reacttanks`); the mass reacted is kept."""
    net = read_epanet_inp(BASE)
    pipes = tuple(dataclasses.replace(p, status="CV") if p.name == "P5" else p
                  for p in net.pipes)
    net = dataclasses.replace(net, pipes=pipes, quality=dataclasses.replace(
        net.quality, global_bulk=-1.0 / 86400, initial={"J1": 1.0, "T1": 1.0}))
    model, router = _router(net)
    names = model.link_names
    for name in ("P5", "PU1", "V1"):
        router.pool.add(names.index(name), 1.0, 1.0)
    before = router.pool.c.clone()
    router.react(300)
    pool = router.pool
    for name in ("P5", "PU1", "V1"):
        (seg,) = pool.segments(names.index(name))
        assert float(pool.c[seg]) == 1.0
    (p1,) = pool.segments(names.index("P1"))        # at J1's quality, 1
    (tank,) = pool.segments(len(names))
    assert float(pool.c[p1]) == pytest.approx(1.0 - 300 / 86400, rel=1e-15)
    assert float(pool.c[tank]) == pytest.approx(1.0 - 300 / 86400, rel=1e-15)
    assert float(router.reacted) == pytest.approx(
        float(((before - pool.c) * pool.v).sum()), rel=1e-12)
    assert float(router.reacted) > 300 / 86400 * float(pool.v[p1] + pool.v[tank])


def test_water_age_adds_the_step_to_every_pipe_and_tank_segment():
    """`pipereact`/`tankreact` under AGE: `c + dt / 3600` hours, in pipes and tanks only."""
    model, router = _router(_base(mode="AGE", units="hrs", initial={}))
    names = model.link_names
    router.pool.add(names.index("PU1"), 1.0, 0.0)
    router.react(300)
    pool = router.pool
    for k, name in enumerate(names):
        for seg in pool.segments(k):
            assert float(pool.c[seg]) == (0.0 if name == "PU1" else 300 / 3600)
    (tank,) = pool.segments(len(names))
    assert float(pool.c[tank]) == 300 / 3600


def _reactive_base(**changes):
    return _base(global_bulk=-1.0 / 86400, global_wall=-0.3 / 86400,
                 initial={"R1": 1.0, "T1": 0.6}, **changes)


@pytest.mark.parametrize("key, count", [("quality.bulk_coefficient", "pipes"),
                                        ("quality.wall_coefficient", "pipes"),
                                        ("quality.tank_coefficient", "tanks")])
def test_the_quality_has_the_gradient_of_each_reaction_coefficient(key, count):
    """d(J4's and T1's quality at 4 h)/d(the coefficients given as drivers) by autograd
    against a central difference (h = 1e-3 of each): the reaction coefficients are
    differentiable inputs (`quality.bulk_coefficient`, `quality.wall_coefficient`,
    `quality.tank_coefficient`, SI per second)."""
    net = _reactive_base()
    model, _, _ = build_model(net)
    kb, kw = pipe_coefficients(net, net.quality)
    base = {"quality.bulk_coefficient": kb, "quality.wall_coefficient": kw,
            "quality.tank_coefficient": tank_coefficients(net, net.quality)}[key]
    assert base.shape == (len(getattr(net, count)),)
    nodes = [model.node_names.index(n) for n in ("J4", "T1")]
    weights = torch.tensor([1.0, 0.5], dtype=F64)

    def objective(value):
        drivers = initial_drivers(model)
        drivers[key] = value
        run = water_run(model, drivers=drivers, duration=4 * 3600, quality=True)
        return (run.node_quality[-1, nodes] * weights).sum()

    value = base.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(objective(value), value)
    assert float(grad.abs().max()) > 0
    with torch.no_grad():
        for i in range(base.numel()):
            h = 1e-3 * abs(float(base[i]))
            up, down = base.clone(), base.clone()
            up[i] += h
            down[i] -= h
            central = (objective(up) - objective(down)) / (2 * h)
            assert float(grad[i]) == pytest.approx(float(central), rel=1e-5, abs=1e-9)


def test_a_reaction_coefficient_of_the_wrong_shape_or_without_quality_is_refused():
    model, _, _ = build_model(_reactive_base())
    drivers = initial_drivers(model)
    drivers["quality.bulk_coefficient"] = torch.zeros(3, dtype=F64)
    with pytest.raises(ValueError, match="quality.bulk_coefficient"):
        water_run(model, drivers=drivers, duration=3600, quality=True)
    drivers["quality.bulk_coefficient"] = torch.zeros(len(model.water_network.pipes),
                                                      dtype=F64)
    with pytest.raises(ValueError, match="quality.bulk_coefficient"):
        water_run(model, drivers=drivers, duration=3600)


def test_the_reaction_coefficients_are_declared_inputs():
    model, _, _ = build_model(_reactive_base())
    net = model.water_network
    for key, over in (("quality.bulk_coefficient", [p.name for p in net.pipes]),
                      ("quality.wall_coefficient", [p.name for p in net.pipes]),
                      ("quality.tank_coefficient", [t.name for t in net.tanks])):
        spec = model.input_specs[key]
        assert spec["over"] == over and spec["required"] is False


@pytest.mark.parametrize("key", ["quality.bulk_coefficient", "quality.wall_coefficient",
                                 "quality.tank_coefficient"])
def test_the_gradient_at_zero_coefficients_is_the_reactions(key):
    """The base network does not react (EPANET's `Reactflag` 0), but coefficients of 0 that
    require a gradient react (adding exactly 0: the same qualities) and carry the
    derivative: the gradient along all of them against a central difference (h = 1e-12,
    which makes EPANET react; small against the smallest mass transfer, `P7`'s 2.4e-8 m/s,
    whose `kw kf / (kf + |kw|)` curves the wall's difference by `h / kf`: MEASURED 5e-5
    relative at h = 1e-9)."""
    net = _base(initial={"R1": 1.0, "T1": 0.6})
    model, _, _ = build_model(net)
    count = len(net.tanks) if key == "quality.tank_coefficient" else len(net.pipes)
    nodes = [model.node_names.index(n) for n in ("J4", "T1")]

    def objective(value):
        drivers = initial_drivers(model)
        drivers[key] = value
        run = water_run(model, drivers=drivers, duration=4 * 3600, quality=True)
        return run.node_quality[-1, nodes].sum(), run

    value = torch.zeros(count, dtype=F64, requires_grad=True)
    total, attached = objective(value)
    (grad,) = torch.autograd.grad(total, value)
    plain = water_run(model, duration=4 * 3600, quality=True)
    assert torch.equal(attached.node_quality.detach(), plain.node_quality)
    h = 1e-12
    with torch.no_grad():
        up, _ = objective(torch.full((count,), h, dtype=F64))
        down, _ = objective(torch.full((count,), -h, dtype=F64))
    central = (up - down) / (2 * h)
    assert float(grad.sum()) != 0.0
    assert float(grad.sum()) == pytest.approx(float(central), rel=1e-5)


@pytest.mark.parametrize("change, match", [(dict(bulk_order=0.5), "ORDER BULK 0.5"),
                                           (dict(tank_order=0.5), "ORDER TANK 0.5")])
def test_a_reaction_order_between_0_and_1_is_refused(change, match):
    """Ruling Q20: EPANET 2.2's `bulkrate` takes `pow(max(0, c), n - 1) = inf` at zero
    concentration for `0 < n < 1` and returns NaN (MEASURED by the review, ORDER BULK 0.5 on
    this network: the report's mass balance `-1.#IND`, 70 of 225 node-times NaN), so there is no
    finite answer to reproduce: refused by name."""
    model, _, _ = build_model(_reactive_base(**change))
    with pytest.raises(ValueError, match=match):
        water_run(model, duration=3600, quality=True)


def test_a_zero_bulk_coefficient_rate_is_exactly_zero():
    """`bulk_rate` is exactly 0 where `kb` is 0, even where the potential is not finite (an
    order between 0 and 1 at `c = 0`); the derivative in `kb` is the potential."""
    kb = torch.zeros(2, dtype=F64, requires_grad=True)
    rate = bulk_rate(_t(0.0, 4.0), kb, 0.5, 0.0, BULK_TINY)
    assert rate.tolist() == [0.0, 0.0]
    (grad,) = torch.autograd.grad(rate.sum(), kb)
    assert grad.tolist() == [0.0, 2.0]


def test_zero_coefficients_requiring_a_gradient_leave_a_reactive_run_unchanged():
    """Ruling Q19: on a reactive network (wall decay, bulk order 2), bulk coefficients of 0
    that require a gradient give the plain run's qualities bit for bit."""
    model, _, _ = build_model(_base(global_wall=-0.3 / 86400, bulk_order=2.0,
                                    initial={"R1": 1.0, "T1": 0.6}))
    drivers = initial_drivers(model)
    drivers["quality.bulk_coefficient"] = torch.zeros(len(model.water_network.pipes),
                                                      dtype=F64, requires_grad=True)
    attached = water_run(model, drivers=drivers, duration=4 * 3600, quality=True)
    plain = water_run(model, duration=4 * 3600, quality=True)
    assert attached.node_quality.requires_grad
    assert torch.equal(attached.node_quality.detach(), plain.node_quality)
    assert torch.equal(attached.link_quality.detach(), plain.link_quality)


def test_a_darcy_weisbach_roughness_not_below_the_diameter_is_refused_with_a_correlation():
    """`adjustdata` divides by `|log(e/d)|`, 0 at `e = d` (in the file's units)."""
    net = read_epanet_inp(BASE)
    pipes = tuple(dataclasses.replace(p, roughness=p.diameter) if p.name == "P3" else
                  dataclasses.replace(p, roughness=0.15e-3) for p in net.pipes)
    dw = dataclasses.replace(net, headloss="D-W", pipes=pipes)
    with pytest.raises(ValueError, match="'P3'"):
        pipe_coefficients(dw, dataclasses.replace(net.quality, roughness_correlation=-1e-5))
    _, kw = pipe_coefficients(dw, net.quality)          # no correlation: no refusal
    assert float(kw.abs().max()) == 0.0
