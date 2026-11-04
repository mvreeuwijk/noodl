"""The plume trajectory nodes and the shared profiles (`noodl.apps.street_aq.plume`)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.street_aq.plume import (
    _table_inputs,
    _trajectory_nodes,
    crosswind_profile,
    plume_table,
    vertical_profile,
)
from noodl.apps.street_aq.puff import puff_tables

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
    from noodl.apps.street_aq.canyon import KAPPA
    p, _, _ = _table_inputs("plume_table", **met, x_max=x_end)
    return p, _trajectory_nodes(p, None, KAPPA, x_end=x_end)


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


_PUFF_MET = {**_NEUTRAL, "x_max": 1000.0}


def test_table_at_node_ages_reproduces_the_scan():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    _, nodes = _nodes()
    zero = torch.zeros(1, dtype=torch.long)
    x, _, sy, sz, zc = tabs.at_age(nodes["t"][0], zero, zero)
    assert torch.equal(x, nodes["x"][0]) and torch.equal(sy, nodes["sigma_y"][0])
    assert torch.equal(sz, nodes["sigma_z"][0]) and torch.equal(zc, nodes["z_c"][0])
    assert tabs.t_max == float(nodes["t"][0, -1])
    assert tabs.x.shape == (1, 1, nodes["x"].shape[1])


def test_at_age_interpolates_linearly_and_holds_the_speed_per_segment():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    zero = torch.zeros((), dtype=torch.long)
    age = torch.tensor([15.0, 10.0, 0.0], dtype=DT)
    x, u, _, _, _ = tabs.at_age(age, zero, zero)
    assert float(x[0]) == 0.5 * float(tabs.x[0, 0, 1] + tabs.x[0, 0, 2])
    assert float(u[0]) == float(tabs.u[0, 0, 2])            # segment (10, 20]
    assert float(u[1]) == float(tabs.u[0, 0, 2])            # node 1 opens segment 2
    assert float(u[2]) == float(tabs.u[0, 0, 1])
    # the displacement is advanced on the segment speed: x_j = x_{j-1} + u_j dt
    torch.testing.assert_close(tabs.x[0, 0, 1:] - tabs.x[0, 0, :-1],
                               tabs.u[0, 0, 1:] * 10.0, rtol=1e-13, atol=0)


def test_age_beyond_the_last_node_returns_the_last_node():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    zero = torch.zeros((), dtype=torch.long)
    x, u, sy, sz, zc = tabs.at_age(torch.tensor(tabs.t_max + 500.0, dtype=DT), zero, zero)
    assert float(x) == float(tabs.x[0, 0, -1]) and float(u) == float(tabs.u[0, 0, -1])
    assert float(sy) == float(tabs.sigma_y[0, 0, -1]) and float(zc) == float(tabs.z_c[0, 0, -1])


def test_distance_resampled_puff_table_is_the_plume_table():
    ref = plume_table(**_PUFF_MET)
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    x, u, sy = tabs.x[0, 0], tabs.u[0, 0], tabs.sigma_y[0, 0]
    pz = vertical_profile(tabs.h_canopy[0], tabs.z_c[0, 0], tabs.sigma_z[0, 0],
                          tabs.h_canopy[0], tabs.h_abl[0])
    pz = torch.cat([pz[1:2], pz[1:]])                        # node 0 copies node 1
    grid = ref.distance
    j = torch.clamp(torch.searchsorted(x, grid, right=True), 1, x.numel() - 1)
    w = (grid - x[j - 1]) / (x[j] - x[j - 1])
    for table, node in ((ref.sigma_y[0], sy), (ref.p_z[0], pz), (ref.u[0], u)):
        got = node[j - 1] + (node[j] - node[j - 1]) * w
        torch.testing.assert_close(got, table, rtol=1e-13, atol=0)


def test_second_table_uses_its_own_centre():
    tabs = puff_tables(centres=torch.tensor([10.0, 40.0], dtype=DT), **_PUFF_MET)
    assert tabs.z_c.shape[:2] == (2, 1)
    # later nodes differ little (the spread dominates the centre) and a row that leaves
    # early repeats its last node, so compare the first nodes
    assert (tabs.z_c[1, :, 1:6] > tabs.z_c[0, :, 1:6]).all()


def test_per_batch_centres_and_meteorology_broadcast():
    met = {**_PUFF_MET, "u_star": torch.tensor([0.45, 0.2], dtype=DT)}
    tabs = puff_tables(centres=torch.tensor([[10.0, 10.0], [40.0, 10.0]], dtype=DT), **met)
    assert tabs.x.shape[:2] == (2, 2) and tabs.h_canopy.shape == (2,)
    one = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    n = one.x.shape[-1]
    assert torch.equal(tabs.x[0, 0, :n], one.x[0, 0])       # rows pad by repeating the end
    assert torch.equal(tabs.x[0, 0, n:], one.x[0, 0, -1].expand(tabs.x.shape[-1] - n))


def test_calm_meteorology_is_refused_by_name():
    with pytest.raises(ValueError, match="MAX_TABLE_STEPS"):
        puff_tables(centres=torch.tensor([10.0], dtype=DT), **{**_PUFF_MET, "u_star": 1e-4})


def test_puff_tables_refuses_the_unstable_regime_by_name():
    with pytest.raises(NotImplementedError, match="puff_tables"):
        puff_tables(centres=torch.tensor([10.0], dtype=DT), **{**_PUFF_MET, "lmo": -50.0})


def test_gradients_reach_meteorology_and_centres_and_second_order_raises():
    u_star = torch.tensor([0.45], dtype=DT, requires_grad=True)
    lmo = torch.tensor([200.0], dtype=DT, requires_grad=True)
    centres = torch.tensor([40.0], dtype=DT, requires_grad=True)
    met = {**_PUFF_MET, "u_star": u_star, "lmo": lmo}
    tabs = puff_tables(centres=centres, **met)
    zero = torch.zeros((), dtype=torch.long)
    x, _, _, sz, zc = tabs.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
    loss = x + sz + zc
    grads = torch.autograd.grad(loss, (u_star, lmo, centres), create_graph=False,
                                allow_unused=False)
    assert all(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads)
    # a finite-difference check on u_star
    h = 1e-6

    def value(us):
        t = puff_tables(centres=centres.detach(), **{**met, "u_star": us, "lmo": lmo.detach()})
        xx, _, _, ss, cc = t.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
        return float(xx + ss + cc)

    fd = (value(u_star.detach() + h) - value(u_star.detach() - h)) / (2 * h)
    assert abs(float(grads[0]) - fd) < 1e-5 * abs(fd)
    tabs = puff_tables(centres=centres, **met)       # the first graph is spent
    x, _, _, sz, zc = tabs.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
    loss = x + sz + zc
    with pytest.raises(RuntimeError, match="second-order"):
        g1 = torch.autograd.grad(loss, u_star, create_graph=True)[0]
        torch.autograd.grad(g1, u_star)
