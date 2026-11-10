"""`TankLevels.event_step`, per tank; tanks held at boundary nodes and stepped by forward
Euler (`tank_scheme="forward_euler"`, the default); and tanks stepped inside the solve as
interior storage (`tank_scheme="backward_euler"`)."""

import math
from pathlib import Path

import pytest
import torch

from noodl import Model
from noodl.apps.water.controls import CLOSED, OPEN, Control
from noodl.apps.water.elements import ONE_WAY_KEY, STATUS_KEY
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    Junction,
    Pump,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterPipe,
    build_model,
    initial_state,
    tank_inflow,
    tank_levels,
    water_event_step,
    water_step,
)
from noodl.apps.water.tanks import TankLevels, TankVolumeCurve
from noodl.elements.base import LawContext
from noodl.storage import LinearStorage

F64 = torch.float64


def _two_tank_closure(controls):
    tanks = [
        Tank("TA", elevation=0.0, init_level=9.99, min_level=0.0, max_level=20.0,
             diameter=1.0),
        Tank("TB", elevation=0.0, init_level=5.0, min_level=0.0, max_level=20.0,
             diameter=1.0),
    ]
    return TankLevels(
        None, tanks, controls, ["P1"],
        tank_nodes=[0, 1], dt=3600.0, reservoir_heads=torch.zeros(0, dtype=F64),
        link_names=["P1"], link_status=torch.ones(1, dtype=F64),
    )


def _control(link, status, node, above, value):
    """A tank-level control: `LINK <link> <status> IF NODE <node> ABOVE|BELOW <value>`."""
    return Control(link, status, None, "node", node, above, value)


def test_event_step_tests_each_control_against_its_own_tank():
    """`TankLevels.event_step` must not test EVERY control against tank 0's level and
    rate, whichever tank the control actually names. Here the control names "TB",
    which is far from its trigger and barely moving (a crossing at 5000 s, past the 100 s
    nominal step): the step must stay nominal. Tank "TA" (index 0) is deliberately placed
    RIGHT next to a level of 10 with a fast rate, so that the bug -- reading index
    0 regardless of the control's own node -- would have crossed almost immediately and
    wrongly shortened the step to about 0.01 s.
    """
    closure = _two_tank_closure(
        (_control("P1", CLOSED, "TB", True, 10.0),)
    )
    level = torch.tensor([9.99, 5.0], dtype=F64)
    rate = torch.tensor([1.0, 1e-3], dtype=F64)
    assert closure.event_step(level, rate, 100.0) == 100.0


def test_event_step_shortens_to_the_named_tanks_own_crossing():
    """The mirror case: the control's own tank DOES cross within the nominal step, while
    the OTHER tank (which the bug would have used) does not."""
    closure = _two_tank_closure(
        (_control("P1", CLOSED, "TB", True, 10.0),)
    )
    level = torch.tensor([1.0, 9.99], dtype=F64)
    rate = torch.tensor([1e-6, 1.0], dtype=F64)
    step = closure.event_step(level, rate, 100.0)
    assert step < 1.0
    expected = (10.0 - 9.99) / 1.0 + 1e-9
    assert step == expected


def test_event_step_with_controls_on_both_tanks_takes_the_earliest_crossing():
    closure = _two_tank_closure(
        (
            _control("P1", CLOSED, "TA", True, 10.0),
            _control("P1", OPEN, "TB", True, 10.0),
        )
    )
    level = torch.tensor([9.0, 9.9], dtype=F64)
    rate = torch.tensor([1.0, 1.0], dtype=F64)
    # TA crosses at 1.0 s, TB at 0.1 s -- the earlier one governs.
    step = closure.event_step(level, rate, 100.0)
    assert step == (10.0 - 9.9) / 1.0 + 1e-9


# ------------------------------------------------------------------ tank storage
NET1 = Path(__file__).resolve().parents[2] / "data" / "water" / "Net1.inp"


def test_tank_storage_rate_in_every_scheme():
    """A tank is `LinearStorage(A / scale, initial=head0 * scale)`: the withdrawal into it
    is `A dH / dt` (backward Euler) or `2 A dH / dt - w_start` (trapezoidal); `initial`
    is the held head."""
    store = LinearStorage([0], torch.tensor([2.0 / 3.0], dtype=F64),
                          initial=torch.tensor([30.0], dtype=F64))
    phi = torch.tensor([33.0], dtype=F64)
    start = torch.tensor([30.0], dtype=F64)
    prev = torch.tensor([0.25], dtype=F64)
    be = LawContext(mode="transient", dt=4.0, scheme="backward_euler")
    tr = LawContext(mode="transient", dt=4.0, scheme="trapezoidal")
    # capacity A / scale = 2 / 3 (area 2, scale 3): head rise 1 m is 2 m3 over 4 s
    assert float(store.rate(phi, start, None, be)) == pytest.approx(0.5)
    assert float(store.rate(phi, start, prev, tr)) == pytest.approx(1.0 - 0.25)
    assert float(store.initial()) == 30.0


def test_forward_euler_holds_the_tanks_at_boundary_nodes():
    model, state, drivers = build_model(read_epanet_inp(NET1))
    layer = model.kirchhoff["water"]
    assert layer.scheme == "forward_euler" and model.tank_scheme == "forward_euler"
    bound = [model.node_names[i] for i in layer.bound.tolist()]
    assert bound == ["9", "2"] and not layer.dynamic
    (storage,) = layer.storage
    assert [model.node_names[i] for i in storage.nodes.tolist()] == ["2"]
    assert "water.tank_level" not in state and state["water.storage_flow"].shape == (1,)
    torch.testing.assert_close(tank_levels(model, state), torch.tensor([120.0 * 0.3048],
                                                                       dtype=F64))
    fresh = initial_state(model)
    assert torch.equal(fresh["water.phi"], state["water.phi"])


def test_event_step_solves_once_and_moves_the_tank_by_dt_inflow_over_area():
    """One solve with the tank held at its step-start head, then `dt * inflow / A`; the
    flow into the tank is the inflow the solve delivered (`tank_inflow`)."""
    model, state, drivers = build_model(read_epanet_inp(NET1))
    layer = model.kirchhoff["water"]
    tank = model.node_names.index("2")
    calls = []
    solve = layer.solve
    layer.solve = lambda *a, **k: calls.append(1) or solve(*a, **k)
    try:
        stepped, dt = water_event_step(model, state, drivers, 1800.0)
    finally:
        del layer.solve
    assert calls == [1] and dt == 1800.0
    inflow = tank_inflow(model, stepped)
    torch.testing.assert_close(
        inflow, -layer.net_outflow(stepped["water.q"])[..., [tank]], rtol=0, atol=0)
    area = torch.pi * model.water_network.tanks[0].diameter ** 2 / 4.0
    expected = state["water.phi"][tank] + 1800.0 * inflow[0] / area
    assert float(stepped["water.phi"][tank]) == pytest.approx(float(expected), rel=1e-15)
    assert float(inflow[0]) > 0.0


def test_storage_mode_makes_the_tanks_interior_storage_nodes():
    model, state, drivers = build_model(read_epanet_inp(NET1), tank_scheme="backward_euler")
    layer = model.kirchhoff["water"]
    bound = [model.node_names[i] for i in layer.bound.tolist()]
    assert bound == ["9"] and "tank" not in layer.kinds
    assert "_datum" not in model.node_names and state["water.phi"].shape == (len(
        model.node_names),)
    (storage,) = layer.storage
    assert [model.node_names[i] for i in storage.nodes.tolist()] == ["2"]
    assert state["water.storage_flow"].shape == (1,)
    torch.testing.assert_close(tank_levels(model, state), torch.tensor([120.0 * 0.3048],
                                                                       dtype=F64))
    fresh = initial_state(model)
    assert torch.equal(fresh["water.phi"], state["water.phi"])
    assert torch.equal(fresh["water.storage_flow"], state["water.storage_flow"])


def test_storage_mode_refuses_quality_and_each_scheme_refuses_the_others_step():
    with pytest.raises(ValueError, match="tank_scheme='backward_euler' with quality"):
        build_model(read_epanet_inp(NET1), tank_scheme="backward_euler", quality=0.5)
    with pytest.raises(ValueError, match="tank_scheme must be"):
        build_model(read_epanet_inp(NET1), tank_scheme="explicit")
    model, state, drivers = build_model(read_epanet_inp(NET1))
    with pytest.raises(ValueError, match="water_event_step"):
        water_step(model, state, drivers, 60.0)
    model, state, drivers = build_model(read_epanet_inp(NET1), tank_scheme="backward_euler")
    with pytest.raises(ValueError, match="with water_step"):
        water_event_step(model, state, drivers, 60.0)


def _first_hours(tank_scheme, dt, hours=3.0):
    """Net1 at its first demand multiplier for `hours`, fixed steps of `dt` s."""
    net = read_epanet_inp(NET1)
    model, state, drivers = build_model(net, tank_scheme=tank_scheme)
    for _ in range(round(hours * 3600 / dt)):
        if tank_scheme == "backward_euler":
            state = water_step(model, state, drivers, dt)
        else:
            state, taken = water_event_step(model, state, drivers, dt)
            assert taken == dt   # no control is crossed in the first hours
    return float(tank_levels(model, state)[0])


def test_the_storage_step_converges_at_first_order_to_the_explicit_one():
    """Backward Euler and EPANET's explicit Euler share the exact trajectory as dt -> 0:
    halving the storage step halves its distance from a fine explicit run (first order).
    MEASURED: 3 h of Net1 at 900 s and 450 s steps."""
    reference = _first_hours("forward_euler", 30.0)
    coarse = abs(_first_hours("backward_euler", 900.0) - reference)
    fine = abs(_first_hours("backward_euler", 450.0) - reference)
    assert fine < 0.6 * coarse and coarse < 0.05, (coarse, fine)


def test_gradient_through_storage_steps_against_central_differences():
    """d(tank level after 3 steps) / d(demand at junction 22), through the implicit
    adjoint of every step, against central differences."""
    net = read_epanet_inp(NET1)
    model, state, drivers = build_model(net, tank_scheme="backward_euler")
    k = model.node_names.index("22")

    def level(sources):
        s = state
        for _ in range(3):
            s = water_step(model, s, {**drivers, "water.sources": sources}, 600.0)
        return tank_levels(model, s)[0]

    sources = drivers["water.sources"].clone().requires_grad_(True)
    analytic, = torch.autograd.grad(level(sources), sources)
    h = 1e-6
    e = torch.zeros_like(sources)
    e[k] = h
    with torch.no_grad():
        fd = (level(sources + e) - level(sources - e)) / (2 * h)
    assert math.isclose(float(analytic[k]), float(fd), rel_tol=1e-6), (analytic[k], fd)


def test_controls_read_the_step_start_level_in_every_pass_of_a_step(monkeypatch):
    """Under coupling="iterate" a step takes several passes. The tank is held at its
    step-start head through all of them, and the level controls read that head too, as
    EPANET (and the closure-carried level before) did, not the level a previous pass of the
    same step has advanced it to. Here the step carries the tank across the 140 ft ABOVE
    trigger, so an advanced level would close the pump inside the step."""
    pingpong, _, drivers = build_model(read_epanet_inp(NET1))
    model = Model(pingpong.net, {"water": pingpong.kirchhoff["water"]},
                  closures=[pingpong.tank_closure], coupling="iterate",
                  iterate_tol={"water": 1e-9})
    state = initial_state(pingpong, values={"water.tank_level": {"2": 42.5}})
    read = []
    levels = TankLevels.levels
    monkeypatch.setattr(TankLevels, "levels",
                        lambda self, s: read.append(float(levels(self, s)[0]))
                        or levels(self, s))
    kwargs = dict(atol=1e-11, rtol=1e-11, max_iter=200)
    diagnostics: dict = {}
    stepped = model.step(state, drivers, 1800.0, diagnostics=diagnostics, **kwargs)
    assert diagnostics["passes"] >= 2 and len(read) >= 2
    assert read == pytest.approx([42.5] * len(read), abs=1e-12)
    assert float(tank_levels(pingpong, stepped)[0]) > 140 * 0.3048   # crossed in this step
    assert stepped["water.link_status"].tolist() == [1.0]             # ... the pump stays open
    once = pingpong.step(state, drivers, 1800.0, **kwargs)
    torch.testing.assert_close(stepped["water.phi"], once["water.phi"], rtol=0, atol=1e-9)


# ------------------------------------------------------------------ volume curves
def test_volume_curve_amount_and_inverse_follow_epanets_interp():
    """`TankVolumeCurve` is EPANET's `tankvolume`/`tankgrade` (`hydraul.c`), both through
    `interp` (`project.c`): linear between the points, the left segment AT a point, and
    the end values beyond the ends (no extrapolation)."""
    curve = TankVolumeCurve([0, 1], torch.tensor([10.0, 0.0], dtype=F64),
                            [[0.0, 1.0, 3.0], [0.0, 2.0]], [[0.0, 2.0, 10.0], [0.0, 8.0]],
                            head_scale=2.0)
    level = torch.tensor([[0.5, 1.0], [1.0, 2.0], [2.0, -1.0], [-1.0, 0.5], [4.0, 1.5]],
                         dtype=F64)
    phi = (torch.tensor([10.0, 0.0], dtype=F64) + level) * 2.0
    volume = curve.amount(phi)
    expected = torch.tensor([[1.0, 4.0], [2.0, 8.0], [6.0, 0.0], [0.0, 2.0], [10.0, 6.0]],
                            dtype=F64)
    torch.testing.assert_close(volume, expected, rtol=0, atol=1e-14)
    # capacity: d volume / d phi of the segment (the left one at a point), 0 beyond the ends
    capacity = curve.capacity(phi, phi)
    expected_slope = torch.tensor([[2.0, 4.0], [2.0, 4.0], [4.0, 0.0], [0.0, 4.0],
                                   [0.0, 4.0]], dtype=F64) / 2.0
    torch.testing.assert_close(capacity, expected_slope, rtol=0, atol=1e-14)
    # potential_of inverts amount inside the curve
    inside = torch.tensor([[1.0, 4.0], [2.0, 6.0], [6.0, 2.0]], dtype=F64)
    back = curve.amount(curve.potential_of(inside))
    torch.testing.assert_close(back, inside, rtol=0, atol=1e-13)
    assert not curve.linear


def _net1_with_volume_curve(tmp_path, points=((0, 0), (60, 90000), (110, 200000),
                                              (130, 250000), (160, 360000))):
    """Net1 with tank 2 on a volume curve (level ft, volume ft3) instead of its cylinder."""
    text = NET1.read_text()
    row = (" 2               \t850         \t120         \t100         \t150         "
           "\t50.5        \t0           \t                \t;")
    assert row in text
    text = text.replace(row, row.replace("\t                \t;", "\tVC              \t;"))
    text = text.replace("[CURVES]", "[CURVES]\n" + "\n".join(f" VC {x} {y}" for x, y in points))
    path = tmp_path / "Net1_vc.inp"
    path.write_text(text)
    return path


def test_a_volume_curve_tank_is_one_tank_volume_curve_storage(tmp_path):
    model, state, drivers = build_model(read_epanet_inp(_net1_with_volume_curve(tmp_path)))
    (storage,) = model.kirchhoff["water"].storage
    assert isinstance(storage, TankVolumeCurve)
    torch.testing.assert_close(tank_levels(model, state), torch.tensor([120.0 * 0.3048],
                                                                       dtype=F64))


def test_volume_curve_advance_conserves_volume(tmp_path):
    """One EPANET step of a tank on a volume curve changes its VOLUME by exactly
    dt * inflow (`hydraul.c`, `tanklevels`: V += q dt, H = tankgrade(V))."""
    model, state, drivers = build_model(read_epanet_inp(_net1_with_volume_curve(tmp_path)))
    (storage,) = model.kirchhoff["water"].storage
    tank = storage.nodes
    stepped, dt = water_event_step(model, state, drivers, 1800.0)
    assert dt == 1800.0
    inflow = tank_inflow(model, stepped)
    before = storage.amount(state["water.phi"][tank])
    after = storage.amount(stepped["water.phi"][tank])
    torch.testing.assert_close(after - before, dt * inflow, rtol=1e-12, atol=0)
    assert float(inflow[0]) > 0.0


def test_volume_curve_gradient_against_central_differences(tmp_path):
    """d(tank level after 3 steps) / d(one volume of the curve), through the inverse of
    every step, against central differences."""
    model, state, drivers = build_model(read_epanet_inp(_net1_with_volume_curve(tmp_path)))
    (storage,) = model.kirchhoff["water"].storage
    storage.volumes.requires_grad_(True)
    j = 2   # the point at 110 ft, whose segments the level is on

    def level():
        s = state
        for _ in range(3):
            s, dt = water_event_step(model, s, drivers, 1200.0)
            assert dt == 1200.0
        return tank_levels(model, s)[0]

    (analytic,) = torch.autograd.grad(level(), storage.volumes)
    # MEASURED: 1.4e-9 relative at this step; 1e-3 of the volume leaves 1.6e-5 of truncation
    h = 1e-5 * float(storage.volumes[0, j].detach())
    with torch.no_grad():
        base = storage.volumes.detach().clone()
        storage.volumes[0, j] = base[0, j] + h
        up = float(level())
        storage.volumes[0, j] = base[0, j] - h
        down = float(level())
        storage.volumes.copy_(base)
    fd = (up - down) / (2 * h)
    assert fd != 0.0
    assert math.isclose(float(analytic[0, j]), fd, rel_tol=1e-6), (analytic[0, j], fd)


# ------------------------------------------------------------------ full and empty tanks
def _tank_rules_net(*, overflow=False, valve=False):
    """A reservoir, two junctions and two tanks, joined by every kind of link whose
    `tankstatus` (`hydstatus.c`) rule differs: pipes with the tank at either end, a check
    valve into the tank, a pipe between the tanks, and pumps into, out of and from a
    reservoir to the tank."""
    curve = {"C1": ((0.01, 30.0),)}
    tanks = (Tank("T1", 50.0, 5.0, 1.0, 10.0, 5.0, overflow=overflow),
             Tank("T2", 50.0, 5.0, 1.0, 10.0, 5.0))
    pipes = (WaterPipe("P0", "R", "J0", 100.0, 0.3, 100.0),
             WaterPipe("A", "T1", "J1", 100.0, 0.3, 100.0),
             WaterPipe("B", "J1", "T1", 100.0, 0.3, 100.0),
             WaterPipe("CV", "J1", "T1", 100.0, 0.3, 100.0, status="CV"),
             WaterPipe("T", "T1", "T2", 100.0, 0.3, 100.0),
             WaterPipe("B2", "J1", "T2", 100.0, 0.3, 100.0))
    pumps = (Pump("PU", "J0", "T1", "C1"), Pump("PR", "R", "T1", "C1"),
             Pump("PS", "T1", "J1", "C1"))
    valves = (Valve("V", "J1", "T1", 0.3, "TCV", 1.0),) if valve else ()
    return WaterNetwork(
        junctions=(Junction("J0", 0.0), Junction("J1", 0.0, 0.001)),
        reservoirs=(Reservoir("R", 100.0),), tanks=tanks, pipes=pipes, pumps=pumps,
        valves=valves, curves=curve)


def _rules(levels, **kwargs):
    model, _, drivers = build_model(_tank_rules_net(**kwargs))
    state = initial_state(model, values={"water.tank_level": levels})
    out = model.tank_closure(state, drivers)
    names = model.link_names
    one_way = out.get(ONE_WAY_KEY, drivers[ONE_WAY_KEY])
    return ({n: int(v) for n, v in zip(names, out[STATUS_KEY].tolist(), strict=True)},
            {n: int(v) for n, v in zip(names, one_way.tolist(), strict=True)})


def test_a_full_tank_lets_its_links_only_drain_it():
    """`tankstatus`: a full tank (head >= Hmax - Htol) closes a link that would fill it: a
    pipe takes the one direction out of the tank, a pump discharging into it closes, and a
    link both rules forbid (the check valve into it) is CLOSED. Only the link's START node
    is examined when it is a tank or reservoir (`n1` first), so the pump from the reservoir
    and the pipe from T1 to T2 never see the end node."""
    status, one_way = _rules({"T1": 10.0 - 1e-4})
    assert one_way == {"P0": 0, "A": 1, "B": -1, "CV": 1, "T": 1, "B2": 0, "PU": 0, "PR": 0,
                       "PS": 0}
    assert status == {"P0": OPEN, "A": OPEN, "B": OPEN, "CV": CLOSED, "T": OPEN, "B2": OPEN,
                      "PU": CLOSED, "PR": OPEN, "PS": OPEN}
    status, one_way = _rules({"T2": 10.0})
    assert one_way["T"] == 0 and one_way["B2"] == -1
    # just below the band, nothing changes
    status, one_way = _rules({"T1": 10.0 - 2e-4})
    assert one_way["A"] == 0 and one_way["B"] == 0 and one_way["CV"] == 1
    assert status["CV"] == OPEN and status["PU"] == OPEN


def test_an_empty_tank_lets_its_links_only_fill_it():
    status, one_way = _rules({"T1": 1.0 + 1e-4})
    assert one_way == {"P0": 0, "A": -1, "B": 1, "CV": 1, "T": -1, "B2": 0, "PU": 0, "PR": 0,
                       "PS": 0}
    assert status == {"P0": OPEN, "A": OPEN, "B": OPEN, "CV": OPEN, "T": OPEN, "B2": OPEN,
                      "PU": OPEN, "PR": OPEN, "PS": CLOSED}


def test_a_full_tank_that_may_overflow_closes_nothing():
    status, one_way = _rules({"T1": 10.0}, overflow=True)
    assert one_way["A"] == 0 and one_way["B"] == 0 and status["PU"] == OPEN
    assert status["CV"] == OPEN


def test_a_valve_on_a_full_tank_is_refused_by_name():
    model, _, drivers = build_model(_tank_rules_net(valve=True))
    state = initial_state(model, values={"water.tank_level": {"T1": 10.0}})
    with pytest.raises(ValueError, match=r"valve\(s\) \['V'\].*full or empty tank"):
        model.tank_closure(state, drivers)
    model.tank_closure(initial_state(model), drivers)   # not at a limit: no refusal


def test_time_to_limit():
    closure = _two_tank_closure(())
    level = torch.tensor([19.0, 4.0], dtype=F64)
    # TA rises to 20 in 1 / 0.01 s; TB falls to 0 in 4 / 0.5 s
    torch.testing.assert_close(
        closure.time_to_limit(level, torch.tensor([0.01, -0.5], dtype=F64)),
        torch.tensor([100.0, 8.0], dtype=F64))
    # already at the limit and still moving towards it, or standing still: never
    out = closure.time_to_limit(torch.tensor([20.0, 4.0], dtype=F64),
                                torch.tensor([0.01, 0.0], dtype=F64))
    assert torch.isinf(out).all()
    # moving away from the nearer limit: the time to the other one
    out = closure.time_to_limit(torch.tensor([19.0], dtype=F64).expand(2),
                                torch.tensor([-1.0, 1.0], dtype=F64))
    torch.testing.assert_close(out, torch.tensor([19.0, 1.0], dtype=F64))


def test_the_implicit_schemes_refuse_volume_curves_and_overflow(tmp_path):
    net = read_epanet_inp(_net1_with_volume_curve(tmp_path))
    with pytest.raises(ValueError, match=r"tank\(s\) \['2'\].*volume curve"):
        build_model(net, tank_scheme="backward_euler")
    with pytest.raises(ValueError, match=r"tank\(s\) \['T1'\].*overflow"):
        build_model(_tank_rules_net(overflow=True), tank_scheme="trapezoidal")


def test_only_the_implicit_schemes_refuse_a_level_outside_the_limits():
    net = read_epanet_inp(NET1)
    model, _, drivers = build_model(net)
    state = initial_state(model, values={"water.tank_level": {"2": 100.0}})
    model.tank_closure(state, drivers)   # forward Euler: a full tank, no refusal
    model, _, drivers = build_model(net, tank_scheme="backward_euler")
    state = initial_state(model, values={"water.tank_level": {"2": 100.0}})
    with pytest.raises(ValueError, match=r"tank\(s\) \['2'\] are outside"):
        model.tank_closure(state, drivers)


def test_a_volume_curve_of_one_point_or_decreasing_volumes_is_refused(tmp_path):
    """A curve needs two points and volumes that do not decrease: one point leaves no
    segment, and a volume that falls as the level rises is not physical (EPANET accepts
    it). Flat volumes are accepted, as EPANET accepts them."""
    with pytest.raises(ValueError, match="curve 0 has 1 point"):
        TankVolumeCurve([0], torch.zeros(1, dtype=F64), [[1.0]], [[10.0]])
    with pytest.raises(ValueError, match="curve 0 must have strictly increasing levels and "
                                         "volumes that do not decrease"):
        TankVolumeCurve([0], torch.zeros(1, dtype=F64), [[0.0, 1.0, 2.0, 3.0]],
                        [[0.0, 10.0, 9.0, 20.0]])
    TankVolumeCurve([0], torch.zeros(1, dtype=F64), [[0.0, 1.0, 2.0, 3.0]],
                    [[0.0, 10.0, 10.0, 20.0]])
    for points in (((0, 0),), ((0, 0), (60, 90000), (110, 80000), (160, 360000))):
        net = read_epanet_inp(_net1_with_volume_curve(tmp_path, points=points))
        with pytest.raises(ValueError, match=r"tank '2' has the volume curve 'VC' with"):
            net.validate()
    read_epanet_inp(_net1_with_volume_curve(
        tmp_path, points=((0, 0), (110, 200000), (160, 200000)))).validate()


def test_a_flat_volume_segment_is_a_jump_in_level():
    """Levels 0, 1, 2, 3 m, volumes 0, 10, 10, 20 m3 (EPANET's `tankgrade` through
    `interp`): the flat volume 10 is at the segment's LOWER level, 1 m; a volume just above
    it is just above 2 m; and the volume is flat (10) at every level between. The area
    toward the flat segment is the neighbour's, and 1 inside it (a level only a tank's
    initial level can have): `water_event_step` reads only its sign, never a time or a
    level."""
    curve = TankVolumeCurve([0], torch.zeros(1, dtype=F64), [[0.0, 1.0, 2.0, 3.0]],
                            [[0.0, 10.0, 10.0, 20.0]])
    level = curve.potential_of(torch.tensor([[10.0], [10.0 + 1e-3], [5.0]], dtype=F64))
    torch.testing.assert_close(level, torch.tensor([[1.0], [2.0001], [0.5]], dtype=F64))
    volume = curve.amount(torch.tensor([[1.0], [1.5], [2.0]], dtype=F64))
    torch.testing.assert_close(volume, torch.full((3, 1), 10.0, dtype=F64))
    def area(level, flow):
        return float(curve.area_toward(torch.tensor([level], dtype=F64),
                                       torch.tensor([flow], dtype=F64)))

    assert (area(1.0, 1.0), area(2.0, -1.0), area(1.5, 1.0), area(1.5, -1.0)) == (
        10.0, 10.0, 1.0, 1.0)


def test_a_full_tank_on_a_flat_tail_is_put_at_its_lower_end(tmp_path):
    """Net1's tank on (0, 0), (60, 90000), (110, 200000), (160, 200000) ft/ft3, its maximum
    level 150 ft on the flat tail: `Vmax` is 200000 ft3, and a tank filled to it is put at
    `tankgrade(Vmax)`, 110 ft (`hydraul.c`, `tanklevels`), not at 150 ft. It starts at
    120 ft, on the tail, so already full by volume; the pump's control to close above
    140 ft holds by volume there (`controls`), so the first step drains it."""
    path = _net1_with_volume_curve(tmp_path, points=((0, 0), (60, 90000), (110, 200000),
                                                     (160, 200000)))
    model, state, drivers = build_model(read_epanet_inp(path))
    stepped, dt = water_event_step(model, state, drivers, 3600.0)
    assert float(tank_inflow(model, stepped)[0]) < 0.0
    assert float(tank_levels(model, stepped)[0]) < 110.0 * 0.3048
    # the same tank with the pump held open fills it: put at 110 ft
    model, state, drivers = build_model(read_epanet_inp(path))
    model.tank_closure.controls = []
    stepped, dt = water_event_step(model, state, drivers, 3600.0)
    assert float(tank_inflow(model, stepped)[0]) > 0.0
    torch.testing.assert_close(tank_levels(model, stepped),
                               torch.tensor([110.0 * 0.3048], dtype=F64))


def test_time_to_limit_skips_a_tank_with_no_net_inflow():
    """`tanktimestep` skips a tank whose net inflow is at most QZERO (1e-6 cfs)."""
    closure = _two_tank_closure(())
    level = torch.tensor([19.0, 19.0], dtype=F64)
    rate = torch.tensor([1e-9, 1e-9], dtype=F64)
    inflow = torch.tensor([2e-8, 3e-8], dtype=F64)   # QZERO = 2.83e-8 m3/s
    out = closure.time_to_limit(level, rate, inflow)
    assert math.isinf(float(out[0])) and float(out[1]) == pytest.approx(1e9)
