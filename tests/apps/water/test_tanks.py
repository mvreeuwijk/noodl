"""`TankLevels.event_step`, per tank; tanks held at boundary nodes and stepped by forward
Euler (`tank_scheme="forward_euler"`, the default); and tanks stepped inside the solve as
interior storage (`tank_scheme="backward_euler"`)."""

import math
from pathlib import Path

import pytest
import torch

from noodl import Model
from noodl.apps.water.controls import CLOSED, OPEN, Control
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    Tank,
    build_model,
    initial_state,
    tank_inflow,
    tank_levels,
    water_event_step,
    water_step,
)
from noodl.apps.water.tanks import TankLevels
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
