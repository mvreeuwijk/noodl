"""R6: the storage sweep must integrate over the step the model was asked for."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.sewer import geometry as geom
from noodl.apps.sewer.network import build_model, tree_steady

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
    return build_model(
        tree_steady(), air=False, quality=False, storage=True, dt_storage=60.0,
    )


def test_a_60_s_step_reproduces_the_recorded_levels():
    model, state, drivers = _model()
    torch.testing.assert_close(
        model.step(state, drivers, 60.0)["sewer.H"], H_AFTER_60_S, rtol=1e-12, atol=1e-14
    )


def test_a_1_s_step_fills_the_manholes_less_than_a_60_s_step():
    """Filling from dry under constant inflow is monotone in time: every level after 1 s is
    strictly positive and strictly below the level after 60 s. Built WITHOUT a declared
    `dt_storage` (R6): with one declared, a 1 s step is now the declared-mismatch case that
    refuses by name -- see
    `test_a_constructor_dt_that_disagrees_with_the_step_is_refused_by_name`."""
    model, state, drivers = build_model(
        tree_steady(), air=False, quality=False, storage=True,
    )
    h_1 = model.step(state, drivers, 1.0)["sewer.H"]
    h_60 = model.step(state, drivers, 60.0)["sewer.H"]
    assert bool((h_1 > 0).all())
    assert bool((h_1 < h_60).all())


def test_a_constructor_dt_that_disagrees_with_the_step_is_refused_by_name():
    model, state, drivers = _model()          # dt_storage=60.0 declared
    with pytest.raises(ValueError, match="SewerHydraulics.*60.*1"):
        model.step(state, drivers, 1.0)


def test_without_a_declared_dt_the_step_interval_is_used():
    model, state, drivers = build_model(
        tree_steady(), air=False, quality=False, storage=True,
    )
    torch.testing.assert_close(
        model.step(state, drivers, 60.0)["sewer.H"], H_AFTER_60_S, rtol=1e-12, atol=1e-14
    )
    h_1 = model.step(state, drivers, 1.0)["sewer.H"]
    assert bool((h_1 > 0).all()) and bool((h_1 < H_AFTER_60_S).all())


def test_each_leaf_manholes_60_s_level_satisfies_its_own_implicit_euler_residual():
    """T5-1 (controller ruling): the recorded 60 s levels are more than a pinned regression
    value -- each LEAF manhole's own implicit-Euler residual

        A_s * H / dt + Q_manning(H) = lateral inflow

    (dry start, so `H_old = 0`) is satisfied by the level `model.step` returns, checked
    independently of how the levels were actually computed. Manhole order is J1, J2, J5, J3,
    J4 (positions 0, 1, 2, 3, 4); the three LEAVES J1, J2, J5 each drain through their own
    0.30 m diameter, n=0.013, slope=0.010 outgoing pipe (C1, C2, C4) straight to inflow-only
    lateral demand -- no upstream contribution, unlike J3 and J4."""
    model, state, drivers = build_model(
        tree_steady(), air=False, quality=False, storage=True,
    )
    h = model.step(state, drivers, 60.0)["sewer.H"]
    a_s = 1.167
    dt = 60.0
    diameter = torch.tensor(0.30, dtype=F64)
    roughness = torch.tensor(0.013, dtype=F64)
    slope = torch.tensor(0.010, dtype=F64)
    # positions 0, 1, 2 are J1, J2, J5; their own lateral inflows (m3/s), tree_steady().
    leaf_inflow = {0: 0.05, 1: 0.08, 2: 0.03}
    for position, inflow in leaf_inflow.items():
        q_out = geom.manning_flow(h[position], diameter, roughness, slope)
        residual = a_s * float(h[position]) / dt + float(q_out) - inflow
        assert abs(residual) < 1e-10
