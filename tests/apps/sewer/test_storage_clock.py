"""R6: the storage sweep must integrate over the step the model was asked for."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.sewer.network import build_sewer_model, tree_steady

F64 = torch.float64

# One 60 s implicit-Euler step from a dry network under tree_steady()'s constant inflows,
# manhole order J1, J2, J5, J3, J4. Recorded from the code as it stands, where the sweep does
# use 60 s: after R6 is fixed a 60 s step must still give exactly this.
H_AFTER_60_S = torch.tensor(
    [0.1477580995372192, 0.20056456811542192, 0.1102345361550266,
     0.2479186549813016, 0.27709022477286743],
    dtype=F64,
)


def _model():
    return build_sewer_model(
        tree_steady(), air=False, quality=False, storage=True, dt_storage=60.0,
    )


def test_a_60_s_step_reproduces_the_recorded_levels():
    model, state, drivers = _model()
    torch.testing.assert_close(
        model.step(state, drivers, 60.0)["sewer.H"], H_AFTER_60_S, rtol=1e-12, atol=1e-14
    )


@pytest.mark.xfail(
    strict=True,
    reason="R6: SewerHydraulics integrates with its constructor dt, not the step's",
)
def test_a_1_s_step_fills_the_manholes_less_than_a_60_s_step():
    """Filling from dry under constant inflow is monotone in time: every level after 1 s is
    strictly positive and strictly below the level after 60 s."""
    model, state, drivers = _model()
    h_1 = model.step(state, drivers, 1.0)["sewer.H"]
    h_60 = model.step(state, drivers, 60.0)["sewer.H"]
    assert bool((h_1 > 0).all())
    assert bool((h_1 < h_60).all())
