"""The plume trajectory nodes and the shared profiles (`noodl.apps.street_aq.plume`)."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from noodl.apps.street_aq.plume import (
    _trajectory_nodes,
    crosswind_profile,
    plume_table,
    vertical_profile,
)

DT = torch.float64
REGRESSION = Path(__file__).parents[2] / "data" / "street" / "plume_table_regression.pt"
_SITE = {"z0": 1.0, "d": 13.0}
_NEUTRAL = {"u_star": 0.45, "h_abl": 800.0, "lmo": math.inf, "h_canopy": 20.0,
            "sigma_theta": math.radians(6.0), **_SITE}
_STABLE = {"u_star": 0.2, "h_abl": 250.0, "lmo": 80.0, "h_canopy": 25.0,
           "sigma_theta": math.radians(8.0), "theta_star": 0.05, "temperature": 277.15,
           **_SITE}
# Tables written from the unmodified `_trajectory` (before it was re-expressed on the
# nodes); the steady table must stay bit-identical.
CASES = [
    {"name": "neutral", **_NEUTRAL, "x_max": 1500.0},
    {"name": "stable", **_STABLE, "x_max": 1500.0},
    {"name": "explicit_tau", **_NEUTRAL, "tau_y": 20.0, "tau_z": 6.0, "x_max": 1000.0},
    {"name": "low_wind", **{**_STABLE, "u_star": 0.05}, "x_max": 600.0},
    {"name": "batched",
     **{**_NEUTRAL, "u_star": torch.tensor([0.45, 0.15, 0.7], dtype=DT),
        "lmo": torch.tensor([math.inf, 60.0, math.inf], dtype=DT),
        "h_abl": torch.tensor([800.0, 200.0, 1000.0], dtype=DT)},
     "x_max": 1200.0},
]


def test_resampled_nodes_equal_the_existing_table_bit_for_bit():
    ref_all = torch.load(REGRESSION, weights_only=True)
    for kwargs in CASES:
        kwargs = dict(kwargs)
        name = kwargs.pop("name")
        t = plume_table(**kwargs)
        ref = ref_all[name]
        assert torch.equal(t.sigma_y, ref["sigma_y"]), name
        assert torch.equal(t.p_z, ref["p_z"]), name
        assert torch.equal(t.u, ref["u"]), name


def _nodes(met=_NEUTRAL, x_end=1000.0):
    from noodl.apps.street_aq import plume
    from noodl.apps.street_aq.canyon import KAPPA
    # build the parameter dict exactly as plume_table does, via its private pieces
    captured = {}
    real = plume._trajectory

    def spy(p, n_grid, kappa):
        captured["p"] = p
        return real(p, n_grid, kappa)

    plume._trajectory = spy
    try:
        plume_table(**met, x_max=x_end)
    finally:
        plume._trajectory = real
    return captured["p"], _trajectory_nodes(captured["p"], None, KAPPA, x_end=x_end)


def test_nodes_are_time_ordered_and_start_at_the_source():
    _, nodes = _nodes()
    assert set(nodes) == {"t", "x", "u", "sigma_y", "sigma_z", "z_c", "p_z"}
    assert nodes["x"][0, 0] == 0.0 and nodes["t"][0, 0] == 0.0
    assert bool((nodes["x"][:, 1:] >= nodes["x"][:, :-1]).all())
    assert float(nodes["x"][0, -1]) > 1000.0
    assert nodes["p_z"][0, 0] == nodes["p_z"][0, 1]


def test_nodes_pad_finished_rows_by_repeating_the_last_node():
    met = {**_NEUTRAL, "u_star": torch.tensor([0.2, 0.9], dtype=DT)}
    _, n = _nodes(met, 800.0)
    # row 1 is the faster one: it leaves earlier and repeats its last node
    assert torch.equal(n["x"][1, -1], n["x"][1, -2])
    assert torch.equal(n["t"][1, -1], n["t"][1, -2])
    assert float(n["x"][0, -1]) > 800.0


def test_n_nodes_max_truncates_the_scan():
    p, _ = _nodes()
    from noodl.apps.street_aq.canyon import KAPPA
    n = _trajectory_nodes(p, 5, KAPPA, x_end=1000.0)
    assert n["x"].shape == (1, 5)


def test_vertical_profile_at_roof_level_is_the_table_p_z():
    p, nodes = _nodes()
    h_r = p["h_r"][:, None]
    h_abl = p["h_abl"][:, None]
    pz = vertical_profile(h_r, nodes["z_c"], nodes["sigma_z"], h_r, h_abl)
    torch.testing.assert_close(pz[:, 1:], nodes["p_z"][:, 1:], rtol=1e-14, atol=0)


def test_crosswind_profile_integrates_to_one():
    y = torch.linspace(-200.0, 200.0, 400001, dtype=DT)
    f = crosswind_profile(y, torch.tensor(5.0, dtype=DT), torch.tensor(30.0, dtype=DT))
    assert abs(float(torch.trapezoid(f, y)) - 1.0) < 1e-10
