"""`TankLevels.event_step`, per tank; and tanks held as storage (`tank_mode="storage"`)."""

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.elements import TankStorage
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    DATUM,
    Tank,
    build_model,
    initial_state,
    tank_inflow,
    tank_levels,
    water_steady,
    water_step,
)
from noodl.apps.water.tanks import Control, TankLevels
from noodl.elements.base import LawContext

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
        dt=3600.0, reservoir_heads=torch.zeros(0, dtype=F64), tank_first_index=0,
    )


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
        (Control("P1", "CLOSED", "TB", "ABOVE", 10.0),)
    )
    level = torch.tensor([9.99, 5.0], dtype=F64)
    rate = torch.tensor([1.0, 1e-3], dtype=F64)
    assert closure.event_step(level, rate, 100.0) == 100.0


def test_event_step_shortens_to_the_named_tanks_own_crossing():
    """The mirror case: the control's own tank DOES cross within the nominal step, while
    the OTHER tank (which the bug would have used) does not."""
    closure = _two_tank_closure(
        (Control("P1", "CLOSED", "TB", "ABOVE", 10.0),)
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
            Control("P1", "CLOSED", "TA", "ABOVE", 10.0),
            Control("P1", "OPEN", "TB", "ABOVE", 10.0),
        )
    )
    level = torch.tensor([9.0, 9.9], dtype=F64)
    rate = torch.tensor([1.0, 1.0], dtype=F64)
    # TA crosses at 1.0 s, TB at 0.1 s -- the earlier one governs.
    step = closure.event_step(level, rate, 100.0)
    assert step == (10.0 - 9.9) / 1.0 + 1e-9


# ------------------------------------------------------------------ tank_mode="storage"
NET1 = Path(__file__).resolve().parents[2] / "data" / "water" / "Net1.inp"


def test_tank_storage_residual_in_every_mode():
    law = TankStorage(torch.tensor([2.0], dtype=F64), torch.tensor([10.0], dtype=F64), scale=3.0)
    dp, q = torch.tensor([33.0], dtype=F64), torch.tensor([0.5], dtype=F64)
    prev = dict(dp_prev=torch.tensor([30.0], dtype=F64), q_prev=torch.tensor([0.25], dtype=F64))
    be = LawContext(mode="transient", dt=4.0, integration="backward_euler", **prev)
    tr = LawContext(mode="transient", dt=4.0, integration="trapezoidal", **prev)
    # head rise 1 m (dp / scale), area 2: volume 2 m3 over 4 s
    assert float(law.residual(dp, q, {}, be)) == pytest.approx(0.5 - 0.5)
    assert float(law.residual(dp, q, {}, tr)) == pytest.approx(0.5 + 0.25 - 1.0)
    assert float(law.residual(dp, q, {}, LawContext(mode="initial"))) == 33.0 - 30.0
    assert float(law.residual(dp, q, {}, LawContext(mode="dc"))) == 0.5


def test_storage_mode_makes_the_tanks_interior_against_a_datum():
    model, state, drivers = build_model(read_epanet_inp(NET1), tank_mode="storage")
    layer = model.kirchhoff["water"]
    bound = [model.node_names[i] for i in layer.bound.tolist()]
    assert bound == ["9", DATUM] and model.node_names[-1] == DATUM
    assert layer.mna and "tank" in layer.kinds
    torch.testing.assert_close(tank_levels(model, state), torch.tensor([120.0 * 0.3048],
                                                                       dtype=F64))
    fresh = initial_state(model)
    assert torch.equal(fresh["water.phi"], state["water.phi"])


def test_storage_mode_refuses_quality_and_an_explicit_model_refuses_water_step():
    with pytest.raises(ValueError, match="tank_mode='storage' with quality"):
        build_model(read_epanet_inp(NET1), tank_mode="storage", quality=0.5)
    model, state, drivers = build_model(read_epanet_inp(NET1))
    with pytest.raises(ValueError, match="not storage"):
        water_step(model, state, drivers, 60.0)


def _first_hours(tank_mode, dt, hours=3.0):
    """Net1 at its first demand multiplier for `hours`, fixed steps of `dt` s."""
    net = read_epanet_inp(NET1)
    model, state, drivers = build_model(net, tank_mode=tank_mode)
    closure = model.tank_closure
    for _ in range(round(hours * 3600 / dt)):
        if tank_mode == "storage":
            state = water_step(model, state, drivers, dt)
        else:
            solved = water_steady(model, state, drivers)
            level = closure.advance(state["water.tank_level"], tank_inflow(model, solved), dt)
            state = {**solved, "water.tank_level": level}
    return float(tank_levels(model, state)[0])


def test_the_storage_step_converges_at_first_order_to_the_explicit_one():
    """Backward Euler and EPANET's explicit Euler share the exact trajectory as dt -> 0:
    halving the storage step halves its distance from a fine explicit run (first order).
    MEASURED: 3 h of Net1 at 900 s and 450 s steps."""
    reference = _first_hours("explicit", 30.0)
    coarse = abs(_first_hours("storage", 900.0) - reference)
    fine = abs(_first_hours("storage", 450.0) - reference)
    assert fine < 0.6 * coarse and coarse < 0.05, (coarse, fine)


def test_gradient_through_storage_steps_against_central_differences():
    """d(tank level after 3 steps) / d(demand at junction 22), through the implicit
    adjoint of every step, against central differences."""
    net = read_epanet_inp(NET1)
    model, state, drivers = build_model(net, tank_mode="storage")
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
