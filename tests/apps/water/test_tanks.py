"""`TankLevels.event_step`, per tank (N13)."""

import torch

from tellegen.apps.water.network import Tank
from tellegen.apps.water.tanks import Control, TankLevels

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
    """N13: `TankLevels.event_step` used to test EVERY control against tank 0's level and
    rate, no matter which tank the control actually named. Here the control names "TB",
    which is far from its trigger and barely moving (a crossing at 5000 s, past the 100 s
    nominal step): the step must stay nominal. Tank "TA" (index 0) is deliberately placed
    RIGHT next to a level of 10 with a fast rate, so that the pre-fix bug -- reading index
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
