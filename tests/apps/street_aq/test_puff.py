"""The plume trajectory nodes and the shared profiles (`noodl.apps.street_aq.plume`)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from noodl.apps.street_aq.network import twelve_street_grid
from noodl.apps.street_aq.plume import (
    _table_inputs,
    _trajectory_nodes,
    crosswind_profile,
    plume_table,
    street_midpoints,
    vertical_profile,
)
from noodl.apps.street_aq.puff import (
    PuffField,
    PuffState,
    PuffTables,
    advance,
    concentration_at,
    emit,
    housekeep,
    puff_mass,
    puff_tables,
    pulse_mass,
    roof_concentration,
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
_ZERO = torch.zeros((), dtype=torch.long)


def _interp(x, grid, col):
    """`col` on `grid` linearly interpolated at `x`, independent of `at_age`."""
    j = torch.clamp(torch.searchsorted(grid, x, right=True), 1, grid.numel() - 1)
    w = (x - grid[j - 1]) / (grid[j] - grid[j - 1])
    return col[j - 1] + (col[j] - col[j - 1]) * w


def test_puff_table_is_the_plume_table_on_its_own_distance_grid():
    ref = plume_table(**_PUFF_MET)
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    assert tabs.u.shape == (1, 1, ref.u.shape[-1]) and tabs.step == ref.step
    assert torch.equal(tabs.distance, ref.distance)
    for name in ("sigma_y", "p_z", "u"):
        assert torch.equal(getattr(tabs, name)[0], getattr(ref, name)), name
    # sigma_z and z_c: the scan nodes resampled the same way; the source node's sigma_z
    # (0) is node 1's, as P_z0 = P_z1
    _, nodes = _nodes(_NEUTRAL, float(ref.distance[-1]))
    s_z = torch.cat([nodes["sigma_z"][0, 1:2], nodes["sigma_z"][0, 1:]])
    for col, node in ((tabs.sigma_z, s_z), (tabs.z_c, nodes["z_c"][0])):
        got = _interp(ref.distance, nodes["x"][0], node)
        torch.testing.assert_close(col[0, 0], got, rtol=1e-13, atol=0)
    assert float(tabs.sigma_z[0, 0, 0]) == float(nodes["sigma_z"][0, 1]) > 0.0


def _quad_travel_time(u, step):
    """Cumulative travel time to each grid point at the piecewise-linear speed `u`, by
    Simpson quadrature of `1/u(x)` (independent of the closed form)."""
    s = torch.linspace(0.0, 1.0, 2001, dtype=DT)
    w = torch.ones_like(s)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    inv = 1.0 / (u[:-1, None] + (u[1:] - u[:-1])[:, None] * s)
    cell = (inv * w).sum(1) * step / (3.0 * (s.numel() - 1))
    return torch.cat([torch.zeros(1, dtype=DT), torch.cumsum(cell, 0)])


def test_age_maps_to_distance_by_the_exact_travel_time_of_the_table_speed():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    u, grid = tabs.u[0, 0], tabs.distance
    t = _quad_travel_time(u, tabs.step)
    torch.testing.assert_close(tabs.t[0, 0], t, rtol=1e-12, atol=0)
    x, u_x, *_ = tabs.at_age(t, _ZERO, _ZERO)
    torch.testing.assert_close(x, grid, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(u_x, u, rtol=1e-12, atol=0)
    # inside the cells, dx/da = u(x): the age is the exact travel time of u(x)
    a = 0.6 * t[:-1] + 0.4 * t[1:]
    h = 1e-3
    xa, ua, *_ = tabs.at_age(a, _ZERO, _ZERO)
    slope = (tabs.at_age(a + h, _ZERO, _ZERO)[0] - tabs.at_age(a - h, _ZERO, _ZERO)[0]) / (2 * h)
    torch.testing.assert_close(slope, ua, rtol=1e-8, atol=0)
    torch.testing.assert_close(ua, _interp(xa, grid, u), rtol=1e-13, atol=0)


def test_at_age_interpolates_every_column_linearly_in_distance():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    age = torch.tensor([0.0, 3.0, 17.5, 41.2, 100.0, 250.0], dtype=DT)
    x, u, s_y, s_z, z_c, p_z = tabs.at_age(age, _ZERO, _ZERO)
    assert float(x[0]) == 0.0 and bool((x[1:] > x[:-1]).all())
    for col, got in ((tabs.u, u), (tabs.sigma_y, s_y), (tabs.sigma_z, s_z),
                     (tabs.z_c, z_c), (tabs.p_z, p_z)):
        torch.testing.assert_close(got, _interp(x, tabs.distance, col[0, 0]), rtol=1e-13,
                                   atol=0)
    # at a grid point's travel time a column takes the grid value
    at_grid = tabs.at_age(tabs.t[0, 0], _ZERO, _ZERO)
    for col, got in ((tabs.sigma_y, at_grid[2]), (tabs.p_z, at_grid[5])):
        torch.testing.assert_close(got, col[0, 0], rtol=1e-12, atol=0)


def test_age_beyond_the_grid_moves_at_the_last_speed_and_holds_the_rest():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    t_end = tabs.t[0, 0, -1]
    x, u, s_y, s_z, z_c, p_z = tabs.at_age(t_end + 50.0, _ZERO, _ZERO)
    torch.testing.assert_close(x, tabs.distance[-1] + 50.0 * tabs.u[0, 0, -1], rtol=1e-13,
                               atol=0)
    for col, got in ((tabs.u, u), (tabs.sigma_y, s_y), (tabs.sigma_z, s_z), (tabs.z_c, z_c),
                     (tabs.p_z, p_z)):
        assert float(got) == float(col[0, 0, -1])
    # a negative age is the source's
    x, _, s_y, *_ = tabs.at_age(torch.tensor(-5.0, dtype=DT), _ZERO, _ZERO)
    assert float(x) == 0.0 and float(s_y) == float(tabs.sigma_y[0, 0, 0])


def test_second_table_uses_its_own_centre():
    tabs = puff_tables(centres=torch.tensor([10.0, 40.0], dtype=DT), **_PUFF_MET)
    assert tabs.z_c.shape[:2] == (2, 1)
    # later the spread dominates the centre law, so compare the first grid points
    assert (tabs.z_c[1, :, :6] > tabs.z_c[0, :, :6]).all()


def test_per_batch_centres_and_meteorology_broadcast():
    met = {**_PUFF_MET, "u_star": torch.tensor([0.45, 0.2], dtype=DT)}
    tabs = puff_tables(centres=torch.tensor([[10.0, 10.0], [40.0, 10.0]], dtype=DT), **met)
    assert tabs.u.shape[:2] == (2, 2) and tabs.h_canopy.shape == (2,)
    assert tabs.t.shape == tabs.u.shape
    one = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    for name in ("u", "sigma_y", "sigma_z", "z_c", "p_z", "t"):
        assert torch.equal(getattr(tabs, name)[0, 0], getattr(one, name)[0, 0]), name
    assert torch.equal(tabs.z_c[1, 1], tabs.z_c[0, 1])       # same centre, same row


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
    x, _, _, sz, zc, pz = tabs.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
    loss = x + sz + zc + pz
    grads = torch.autograd.grad(loss, (u_star, lmo, centres), create_graph=False,
                                allow_unused=False)
    assert all(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads)
    # a finite-difference check on u_star
    h = 1e-6

    def value(us):
        t = puff_tables(centres=centres.detach(), **{**met, "u_star": us, "lmo": lmo.detach()})
        xx, _, _, ss, cc, pp = t.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
        return float(xx + ss + cc + pp)

    fd = (value(u_star.detach() + h) - value(u_star.detach() - h)) / (2 * h)
    assert abs(float(grads[0]) - fd) < 1e-5 * abs(fd)
    tabs = puff_tables(centres=centres, **met)       # the first graph is spent
    x, _, _, sz, zc, pz = tabs.at_age(torch.tensor(60.0, dtype=DT), zero, zero)
    loss = x + sz + zc + pz
    with pytest.raises(RuntimeError, match="second-order"):
        g1 = torch.autograd.grad(loss, u_star, create_graph=True)[0]
        torch.autograd.grad(g1, u_star)


# --- the puff field: emit, advance, evaluate ---------------------------------------------

_LONG = torch.long


def _spread(x):
    """The hand tables' spreads at distance `x`: `0.5 + 0.2 t` at 3 m/s."""
    return 0.5 + 0.2 * x / 3.0


def _hand_tables(speeds, *, n_grid=200, z_c=15.0, h_canopy=0.0, h_abl=1e6):
    """A hand-made `PuffTables` on the 10 m grid (out to `x_max`, its end) with speeds
    `speeds(k)` at grid point `k` (one row per entry of `speeds`), spreads `_spread(x)`
    and the centre `z_c`."""
    x = torch.arange(n_grid, dtype=DT) * 10.0
    n_b = len(speeds)
    u = torch.stack([torch.tensor([speed(k) for k in range(n_grid)], dtype=DT)
                     for speed in speeds])[None]
    sigma = _spread(x).expand(1, n_b, n_grid).clone()
    z = torch.full((1, n_b, n_grid), z_c, dtype=DT)
    h_c = torch.full((n_b,), h_canopy, dtype=DT)
    h_t = torch.full((n_b,), h_abl, dtype=DT)
    p_z = vertical_profile(h_c[:, None], z, sigma, h_c[:, None], h_t[:, None])
    return PuffTables(u=u, sigma_y=sigma, sigma_z=sigma.clone(), z_c=z, p_z=p_z,
                      h_canopy=h_c, h_abl=h_t, x_max=float(x[-1]))


def _x_linear_speed(u0, g, t):
    """The distance at age `t` for a speed `u0 + g x`: `(u0 / g)(exp(g t) - 1)`."""
    return u0 * t if g == 0 else u0 / g * math.expm1(g * t)


def _gauss(s, sigma):
    return torch.exp(-0.5 * (s / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def _one_release(field, tables, *, mass, origin, dt, steps, theta):
    state = PuffState.empty((1,), 8)
    state, dropped = emit(field, state, torch.tensor([[mass]], dtype=DT),
                          torch.zeros(1, dtype=_LONG),
                          torch.tensor([[origin]], dtype=DT))
    assert int(dropped.sum()) == 0
    for _ in range(steps):
        state = advance(state, tables, dt, torch.tensor([theta], dtype=DT))
    return state


def _wind_points(origin, theta, along, cross, z):
    """`(R, 3)` points at `along`/`cross` wind offsets from `origin` and heights `z`, with
    the offsets and heights flattened alongside."""
    a, c, zz = (torch.tensor(v, dtype=DT) for v in (along, cross, z))
    a, c, zz = (v.reshape(-1) for v in torch.meshgrid(a, c, zz, indexing="ij"))
    e = torch.tensor([math.cos(theta), math.sin(theta)], dtype=DT)
    n = torch.tensor([-math.sin(theta), math.cos(theta)], dtype=DT)
    xy = torch.tensor(origin, dtype=DT) + a[:, None] * e + c[:, None] * n
    return torch.cat([xy, zz[:, None]], dim=1), a, c, zz


def test_single_point_puff_matches_the_closed_form():
    # constant u = 3, spreads 0.5 + 0.2 x / 3, z_c = 15, H_R = 0 (the canopy image is the
    # ground image), h = 1e6; one puff of 2 kg advanced to age 60 s (x = 180 m) in 5 s
    # steps. Along the wind: the slug convolved with sigma_x = sigma_y(180 m); crosswind
    # and vertical: the spreads at the receptor's distance clamped to the slug's ends
    tabs = _hand_tables([lambda k: 3.0])
    field = PuffField.points(1, along_wind="sigma_y")
    theta, origin = 0.4, (30.0, -20.0)
    state = _one_release(field, tabs, mass=2.0, origin=origin, dt=5.0, steps=12, theta=theta)
    assert float(state.age[0, 0]) == 60.0
    points, a, c, z = _wind_points(origin, theta, [168.0, 177.0, 180.0, 186.0, 200.0],
                                   [0.0, 4.0, -9.0], [1.0, 15.0, 25.0])
    sigma_x = _spread(torch.tensor(180.0, dtype=DT))
    # the slug spans x(57.5) - x(60) = -7.5 m to x(62.5) - x(60) = +7.5 m: ell = 15 m
    lo, hi, ell = -7.5, 7.5, 15.0
    u = a - 180.0
    root2 = math.sqrt(2.0) * sigma_x
    g_x = (torch.erf((u - lo) / root2) - torch.erf((u - hi) / root2)) / (2.0 * ell)
    sigma = _spread(180.0 + torch.clamp(u, lo, hi))
    vertical = _gauss(z - 15.0, sigma) + _gauss(z + 15.0, sigma)
    expected = 2.0 * g_x * _gauss(c, sigma) * vertical
    th = torch.tensor([theta], dtype=DT)
    got = concentration_at(field, state, tabs, points, th, 5.0)
    assert got.shape == (1, points.shape[0])
    torch.testing.assert_close(got[0], expected, rtol=1e-13, atol=0)
    # ell -> 0: the plain Gaussian puff
    tiny = concentration_at(field, state, tabs, points, th, 1e-4)
    near = _spread(180.0 + torch.clamp(u, -1.5e-4, 1.5e-4))  # slug ends at dt = 1e-4 s
    plain = 2.0 * _gauss(u, sigma_x) * _gauss(c, near) * (
        _gauss(z - 15.0, near) + _gauss(z + 15.0, near))
    torch.testing.assert_close(tiny[0], plain, rtol=1e-8, atol=0)


def test_slug_without_along_wind_spread_is_a_segment_of_the_steady_plume():
    # the slug [172.5, 187.5) m from the origin carries density m / (dt u) with the
    # spreads read at each receptor's own distance: the steady plume of rate m / dt
    tabs = _hand_tables([lambda k: 3.0])
    field = PuffField.points(1, along_wind="none")
    theta, origin = 1.1, (0.0, 0.0)
    state = _one_release(field, tabs, mass=2.0, origin=origin, dt=5.0, steps=12, theta=theta)
    along = [172.6, 177.0, 180.0, 187.4]
    inside, _, _, _ = _wind_points(origin, theta, along, [2.0], [15.0])
    outside, _, _, _ = _wind_points(origin, theta, [172.4, 187.6, 210.0, 100.0], [2.0], [15.0])
    th = torch.tensor([theta], dtype=DT)
    sigma = _spread(torch.tensor(along, dtype=DT))
    level = 2.0 / (5.0 * 3.0) * _gauss(torch.tensor(2.0, dtype=DT), sigma) * (
        _gauss(torch.zeros_like(sigma), sigma) + _gauss(torch.full_like(sigma, 30.0), sigma))
    torch.testing.assert_close(concentration_at(field, state, tabs, inside, th, 5.0)[0],
                               level, rtol=1e-13, atol=0)
    assert torch.equal(concentration_at(field, state, tabs, outside, th, 5.0),
                       torch.zeros(1, 4, dtype=DT))
    # beyond cutoff_sigma crosswind standard deviations (of the receptor's own sigma_y)
    # the puff is exactly zero
    far, _, _, _ = _wind_points(origin, theta, [180.0], [4.0 * 12.5 + 0.01], [15.0])
    assert float(concentration_at(field, state, tabs, far, th, 5.0)) == 0.0
    near, _, _, _ = _wind_points(origin, theta, [180.0], [4.0 * 12.5 - 0.01], [15.0])
    assert float(concentration_at(field, state, tabs, near, th, 5.0)) > 0.0


def test_roof_level_reads_the_table_p_z_and_free_receptors_the_profile():
    # a point puff of age 5 s, inside the scan's first 10 s, where its sigma_z starts at 0:
    # at roof level the table's p_z at the puff's distance (finite, P_z0 = P_z1); at a free
    # receptor the vertical profile of sigma_z and z_c read at that distance
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    field = PuffField.points(1, along_wind="none")
    th = torch.tensor([0.0], dtype=DT)
    state = _one_release(field, tabs, mass=2.0, origin=(0.0, 0.0), dt=5.0, steps=1, theta=0.0)
    x, u, s_y, s_z, z_c, p_z = tabs.at_age(state.age[0, 0], _ZERO, _ZERO)
    # the slug covers the ages [2.5, 7.5] s: density m / (5 s * u(x))
    level = 2.0 / (5.0 * u) * crosswind_profile(torch.zeros((), dtype=DT), s_y,
                                                torch.zeros((), dtype=DT))
    xy = torch.stack([x, torch.zeros((), dtype=DT)])[None]
    roof = roof_concentration(field, state, tabs, xy, th, 5.0)[0, 0]
    torch.testing.assert_close(roof, level * p_z, rtol=1e-13, atol=0)
    h_r = tabs.h_canopy[0]
    free = concentration_at(field, state, tabs, torch.cat([xy, h_r.reshape(1, 1)], dim=1),
                            th, 5.0)[0, 0]
    expected = level * vertical_profile(h_r, z_c, s_z, h_r, tabs.h_abl[0])
    torch.testing.assert_close(free, expected, rtol=1e-13, atol=0)
    assert bool(torch.isfinite(free)) and float(roof) > 0.0


@pytest.mark.parametrize("speed, u0, g", [(lambda k: 3.0, 3.0, 0.0),
                                          (lambda k: 2.0 + 0.01 * k, 2.0, 0.001)],
                         ids=["constant", "growing"])
@pytest.mark.parametrize("dt", [2.0, 5.0, 10.0])
def test_continuous_source_is_the_steady_plume(dt, speed, u0, g):
    tabs = _hand_tables([speed])
    field = PuffField.points(1, along_wind="none")
    theta = 0.7
    th = torch.tensor([theta], dtype=DT)
    state = PuffState.empty((1,), 80)
    for _ in range(60):  # unit rate: mass dt per step
        state, _ = emit(field, state, torch.tensor([[dt]], dtype=DT),
                        torch.zeros(1, dtype=_LONG), torch.zeros(1, 1, 2, dtype=DT))
        state = advance(state, tabs, dt, th)
    # receptors at ages (k + f) dt along the axis, x(t) in closed form for u = u0 + g x:
    # each lies in exactly one slug, and the train is the steady plume, profile / u(x)
    ks = list(range(2, 58))
    fs = [-0.4, -0.1, 0.2, 0.45]
    x_r = torch.tensor([_x_linear_speed(u0, g, (k + f) * dt) for k in ks for f in fs],
                       dtype=DT)
    e = torch.tensor([math.cos(theta), math.sin(theta)], dtype=DT)
    z = 3.0
    points = torch.cat([x_r[:, None] * e, torch.full((x_r.numel(), 1), z, dtype=DT)], dim=1)
    got = concentration_at(field, state, tabs, points, th, dt)[0]
    sigma = _spread(x_r)
    zz = torch.full_like(sigma, z)
    profile = _gauss(torch.zeros_like(sigma), sigma) * (
        _gauss(zz - 15.0, sigma) + _gauss(zz + 15.0, sigma))
    torch.testing.assert_close(got, profile / (u0 + g * x_r), rtol=1e-11, atol=0)


def test_receptor_owner_excludes_own_puffs():
    tabs = _hand_tables([lambda k: 3.0], h_canopy=2.0)
    field = PuffField.points(2)
    state = PuffState.empty((1,), 4)
    state, _ = emit(field, state, torch.tensor([[1.0, 3.0]], dtype=DT),
                    torch.zeros(2, dtype=_LONG),
                    torch.tensor([[[0.0, 0.0], [10.0, 0.0]]], dtype=DT))
    th = torch.tensor([0.0], dtype=DT)
    for _ in range(3):
        state = advance(state, tabs, 10.0, th)
    # the receptor is 100 m and 90 m from the two releases, grid distances, where the
    # table's p_z is the vertical profile exactly
    xy = torch.tensor([[100.0, 1.0]] * 3, dtype=DT)
    owner = torch.tensor([0, 1, -1], dtype=_LONG)
    rc = roof_concentration(field, state, tabs, xy, th, 10.0, receptor_owner=owner)[0]
    free = concentration_at(field, state, tabs, torch.cat(
        [xy, torch.full((3, 1), 2.0, dtype=DT)], dim=1), th, 10.0)[0]
    assert float(rc[0]) > 0 and float(rc[1]) > 0
    torch.testing.assert_close(rc[0] + rc[1], rc[2], rtol=1e-14, atol=0)
    torch.testing.assert_close(rc[2], free[2], rtol=1e-14, atol=0)  # no clamp on points
    assert torch.equal(roof_concentration(field, state, tabs, xy, th, 10.0,
                                          receptor_owner=None)[0], rc[2].expand(3))


def test_street_puffs_skip_their_own_street_and_reach_downwind_ones():
    net, names = twelve_street_grid()
    theta = math.pi / 2 - 0.2  # toward north, slightly east
    th = torch.tensor([theta], dtype=DT)
    field = PuffField.for_network(net, theta_w=th)
    n_s, n_j = len(net.streets), len(net.junctions)
    assert field.n_sources == n_s + n_j
    assert field.source_xy0.shape == (n_s + n_j, 2)
    assert torch.equal(field.source_xy0[:n_s], street_midpoints(net))
    assert field.width.shape == (1, field.xy.shape[0])
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    emitter = names.index("11")              # E11 (0,-100) -> C (0,0)
    masses = torch.zeros(1, n_s + n_j, dtype=DT)
    masses[0, emitter] = 1.0
    state, _ = emit(field, PuffState.empty((1,), 16), masses,
                    torch.zeros(n_s + n_j, dtype=_LONG))
    for _ in range(3):
        state = advance(state, tabs, 5.0, th)
    mids = street_midpoints(net)
    own = torch.arange(n_s, dtype=_LONG)
    with_own = roof_concentration(field, state, tabs, mids, th, 5.0, receptor_owner=None)[0]
    without = roof_concentration(field, state, tabs, mids, th, 5.0, receptor_owner=own)[0]
    assert float(with_own[emitter]) > 0 and float(without[emitter]) == 0.0
    others = torch.ones(n_s, dtype=torch.bool)
    others[emitter] = False
    assert torch.equal(with_own[others], without[others])
    assert float(without[names.index("6")]) > 0           # C -> A, downwind


def test_advance_follows_the_wind_direction():
    tabs = _hand_tables([lambda k: 3.0, lambda k: 1.0 + 0.01 * k])
    field = PuffField.points(1)
    theta = torch.tensor([0.3, 2.0], dtype=DT)
    state = PuffState.empty((2,), 3)
    state, _ = emit(field, state, torch.tensor([[1.0], [2.0]], dtype=DT),
                    torch.zeros(1, dtype=_LONG), torch.zeros(2, 1, 2, dtype=DT))
    for _ in range(7):  # 3 s steps cross the 10 m grid points
        state = advance(state, tabs, 3.0, theta)
    assert torch.equal(state.age[:, 0], torch.full((2,), 21.0, dtype=DT))
    x = torch.tensor([_x_linear_speed(3.0, 0.0, 21.0), _x_linear_speed(1.0, 1e-3, 21.0)],
                     dtype=DT)
    expected = x[:, None] * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    torch.testing.assert_close(state.disp[:, 0], expected, rtol=1e-13, atol=1e-13)
    # free slots do not move
    assert torch.equal(state.disp[:, 1:], torch.zeros(2, 2, 2, dtype=DT))
    assert torch.equal(state.age[:, 1:], torch.zeros(2, 2, dtype=DT))


def test_state_round_trips_through_state_keys():
    state = PuffState.empty((2,), 5)
    keys = state.to_state()
    assert "puff.mass" in keys and keys["puff.disp"].shape == (2, 5, 2)
    assert keys["puff.live"].dtype == torch.bool and keys["puff.source"].dtype == _LONG
    assert keys["puff.mass_out"].shape == (2,) and keys["puff.mass"].dtype == DT
    back = PuffState.from_state(keys)
    assert all(torch.equal(getattr(back, k.split(".", 1)[1]), v) for k, v in keys.items())
    with pytest.raises(KeyError, match="puff.age"):
        PuffState.from_state({k: v for k, v in keys.items() if k != "puff.age"})
    assert set(state.to_state("cloud")) == {k.replace("puff.", "cloud.") for k in keys}


def test_emit_writes_free_slots_counts_mass_and_evicts_on_overflow():
    field = PuffField.points(3)
    state = PuffState.empty((2,), 3)
    masses = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=DT)
    xy = torch.tensor([[[1.0, 2.0]] * 3, [[3.0, 4.0]] * 3], dtype=DT)
    state, dropped = emit(field, state, masses, torch.tensor([0, 1, 2], dtype=_LONG), xy)
    assert torch.equal(dropped, torch.zeros(2, dtype=_LONG))
    assert state.live.sum(1).tolist() == [2, 1]
    assert state.source[0, :2].tolist() == [0, 2] and state.table[0, :2].tolist() == [0, 2]
    assert torch.equal(state.disp[1, 0], torch.tensor([3.0, 4.0], dtype=DT))
    assert torch.equal(state.mass_emitted, torch.tensor([3.0, 4.0], dtype=DT))
    state, _ = emit(field, state, torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DT),
                    torch.zeros(3, dtype=_LONG), xy)
    assert state.live.sum(1).tolist() == [3, 2]
    with pytest.raises(ValueError, match="capacity"):
        emit(field, PuffState.empty((1,), 2), torch.ones(1, 3, dtype=DT),
             torch.zeros(3, dtype=_LONG), xy[:1])
    with pytest.raises(ValueError, match="release_xy"):
        emit(field, PuffState.empty((2,), 3), masses, torch.zeros(3, dtype=_LONG), None)


def test_emit_refuses_a_bad_age_by_name():
    field = PuffField.points(1)
    args = (PuffState.empty((1,), 2), torch.ones(1, 1, dtype=DT), torch.zeros(1, dtype=_LONG),
            torch.zeros(1, 1, 2, dtype=DT))
    tabs = _hand_tables([lambda k: 3.0])
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="age must be >= 0"):
            emit(field, *args, tables=tabs, age=bad, theta_w=0.0)
    with pytest.raises(ValueError, match="tables= and theta_w="):
        emit(field, *args, age=12.5, theta_w=0.0)
    with pytest.raises(ValueError, match="tables= and theta_w="):
        emit(field, *args, tables=tabs, age=12.5)
    # a positive age starts the puff x(age) downwind: 3 m/s for 12.5 s along 0
    state, _ = emit(field, *args, tables=tabs, age=12.5, theta_w=0.0)
    assert float(state.age[0, 0]) == 12.5
    torch.testing.assert_close(state.disp[0, 0], torch.tensor([37.5, 0.0], dtype=DT),
                               rtol=1e-14, atol=1e-12)


def test_along_wind_option_is_checked_by_name():
    with pytest.raises(ValueError, match="along_wind"):
        PuffField.points(1, along_wind="gaussian")


def test_gradient_reaches_mass_and_release_position():
    tabs = _hand_tables([lambda k: 3.0])
    field = PuffField.points(1)
    mass = torch.tensor([[2.0]], dtype=DT, requires_grad=True)
    origin = torch.tensor([[[30.0, -20.0]]], dtype=DT, requires_grad=True)
    state, _ = emit(field, PuffState.empty((1,), 2), mass, torch.zeros(1, dtype=_LONG), origin)
    th = torch.tensor([0.4], dtype=DT)
    for _ in range(4):
        state = advance(state, tabs, 5.0, th)
    points = torch.tensor([[85.0, 5.0, 10.0]], dtype=DT)
    c = concentration_at(field, state, tabs, points, th, 5.0).sum()
    g_m, g_xy = torch.autograd.grad(c, (mass, origin))
    torch.testing.assert_close(g_m, (c / 2.0).detach().reshape(1, 1), rtol=1e-14, atol=0)
    assert bool(torch.isfinite(g_xy).all()) and float(g_xy.abs().sum()) > 0


# --- housekeeping and the mass account -----------------------------------------------------


def _identity(state):
    """emitted == live + out + dropped, exactly (the masses in these tests are dyadic)."""
    total = puff_mass(state) + state.mass_out + state.mass_dropped
    assert torch.equal(state.mass_emitted, total), (state.mass_emitted, total)


def _released(masses, capacity=8, tabs=None):
    field = PuffField.points(len(masses))
    tabs = tabs or _hand_tables([lambda k: 3.0])
    state, n = emit(field, PuffState.empty((1,), capacity),
                    torch.tensor([masses], dtype=DT),
                    torch.zeros(len(masses), dtype=_LONG),
                    torch.zeros(1, len(masses), 2, dtype=DT))
    assert int(n.sum()) == 0
    _identity(state)
    return field, tabs, state


def test_puffs_leaving_the_domain_go_to_mass_out():
    field, tabs, state = _released([4.0, 2.0])
    th = torch.zeros(1, dtype=DT)
    for _ in range(3):
        state = advance(state, tabs, 5.0, th)
    kept = housekeep(field, state, tabs)
    assert torch.equal(kept.live, state.live) and float(kept.mass_out) == 0.0
    # beyond the table's end (x(age) > x_max, the grid's last point)
    t_end = float(tabs.t[0, 0, -1])
    old = replace(state, age=torch.tensor([[t_end + 1.0, 5.0] + [0.0] * 6], dtype=DT))
    out = housekeep(field, old, tabs)
    assert out.live[0].tolist() == [False, True] + [False] * 6
    assert float(out.mass_out) == 4.0 and float(out.mass[0, 0]) == 0.0
    _identity(out)
    # beyond a shorter extent x_max, inside the grid
    short = replace(tabs, x_max=50.0)
    far = replace(state, age=torch.tensor([[30.0, 5.0] + [0.0] * 6], dtype=DT))
    out = housekeep(field, far, short)
    assert out.live[0].tolist() == [False, True] + [False] * 6
    assert float(out.mass_out) == 4.0 and float(out.mass_dropped) == 0.0
    _identity(out)


def test_faint_puffs_go_to_mass_dropped_and_identity_holds():
    field, tabs, state = _released([1.0, 2.0**-10, 2.0**-30, -1.0])
    th = torch.zeros(1, dtype=DT)
    state = advance(state, tabs, 5.0, th)
    _identity(state)
    kept = housekeep(field, state, tabs)
    assert kept.live[0, :4].tolist() == [True, True, False, True]
    assert float(kept.mass_dropped) == 2.0**-30 and float(kept.mass_out) == 0.0
    assert float(kept.mass[0, 2]) == 0.0
    _identity(kept)
    # a looser threshold drops the second, and a signed puff counts by its magnitude
    loose = housekeep(field, state, tabs, drop_below=1e-2)
    assert loose.live[0, :4].tolist() == [True, False, False, True]
    _identity(loose)
    again = housekeep(field, loose, tabs, drop_below=1e-2)
    assert torch.equal(again.live, loose.live)
    with pytest.raises(ValueError, match="drop_below"):
        housekeep(field, state, tabs, drop_below=1.5)


def test_faint_test_uses_the_peak_not_the_mass():
    # a wide source dilutes its peak: same mass, lower peak, so it is the faint one
    field = PuffField.points(2)
    field.width = torch.tensor([[0.0, 5.0e7]], dtype=DT)
    tabs = _hand_tables([lambda k: 3.0])
    state, _ = emit(field, PuffState.empty((1,), 4), torch.ones(1, 2, dtype=DT),
                    torch.zeros(2, dtype=_LONG), torch.zeros(1, 2, 2, dtype=DT))
    kept = housekeep(field, advance(state, tabs, 5.0, torch.zeros(1, dtype=DT)), tabs)
    assert kept.live[0, :2].tolist() == [True, False]
    _identity(kept)


def test_capacity_eviction_reports_count_and_keeps_identity():
    for with_tables in (False, True):
        field = PuffField.points(3)
        tabs = _hand_tables([lambda k: 3.0])
        kw = {"tables": tabs} if with_tables else {}
        zero = torch.zeros(1, 3, 2, dtype=DT)
        state, n = emit(field, PuffState.empty((1,), 4),
                        torch.tensor([[8.0, 4.0, 2.0]], dtype=DT),
                        torch.zeros(3, dtype=_LONG), zero, **kw)
        assert n.tolist() == [0]
        state, n = emit(field, state, torch.tensor([[1.0, 1.0, 0.0]], dtype=DT),
                        torch.zeros(3, dtype=_LONG), zero, **kw)
        assert n.tolist() == [1] and n.dtype == _LONG
        assert float(state.mass_dropped) == 2.0 and float(state.mass_emitted) == 16.0
        assert sorted(state.mass[0].tolist()) == [1.0, 1.0, 4.0, 8.0]
        _identity(state)


def test_zero_mass_sources_cost_no_slot():
    field = PuffField.points(3)
    state, n = emit(field, PuffState.empty((1,), 1), torch.tensor([[0.0, 3.0, 0.0]], dtype=DT),
                    torch.zeros(3, dtype=_LONG), torch.zeros(1, 3, 2, dtype=DT))
    assert n.tolist() == [0] and state.live.sum().item() == 1


def test_empty_instance_housekeeps_without_nan():
    field = PuffField.points(1)
    tabs = _hand_tables([lambda k: 3.0])
    mass = torch.tensor([[0.0], [2.0]], dtype=DT, requires_grad=True)
    state, _ = emit(field, PuffState.empty((2,), 3), mass, torch.zeros(1, dtype=_LONG),
                    torch.zeros(2, 1, 2, dtype=DT))
    state = advance(state, tabs, 5.0, torch.zeros(2, dtype=DT))
    kept = housekeep(field, state, tabs)
    assert kept.live.sum(1).tolist() == [0, 1]
    for name in ("mass", "mass_out", "mass_dropped", "mass_emitted"):
        assert bool(torch.isfinite(getattr(kept, name)).all()), name
    assert puff_mass(kept).tolist() == [0.0, 2.0]
    puff_mass(kept)[1].backward(retain_graph=True)
    assert mass.grad.tolist() == [[0.0], [1.0]]
    # the counters keep the graph
    old = replace(state, age=torch.full((2, 3), float(tabs.t[0, 0, -1]) + 1.0, dtype=DT))
    out = housekeep(field, old, tabs)
    g, = torch.autograd.grad(out.mass_out[1], mass)
    assert g.tolist() == [[0.0], [1.0]]


def test_puff_width_batch_must_match_by_name():
    field = PuffField.points(1)
    field.width = torch.ones(3, 1, dtype=DT)
    state = PuffState.empty((2,), 2)
    with pytest.raises(ValueError, match="width"):
        roof_concentration(field, state, _hand_tables([lambda k: 3.0]),
                           torch.zeros(1, 2, dtype=DT), 0.0, 5.0)


def test_age_zero_puffs_have_finite_bounds_with_real_tables():
    tabs = puff_tables(centres=torch.tensor([20.0], dtype=DT), **_NEUTRAL, x_max=700.0)
    field = PuffField.points(3)
    zero = torch.zeros(1, 3, 2, dtype=DT)
    index = torch.zeros(3, dtype=_LONG)
    state, _ = emit(field, PuffState.empty((1,), 3), torch.tensor([[4.0, 1.0, 2.0]], dtype=DT),
                    index, zero, tables=tabs)
    kept = housekeep(field, state, tabs)
    assert kept.live[0].tolist() == [True] * 3 and float(kept.mass_dropped) == 0.0
    # eviction ranks fresh puffs by strength: the 1.0 puff is the faintest
    state, n = emit(field, state, torch.tensor([[0.0, 0.0, 8.0]], dtype=DT), index, zero,
                    tables=tabs)
    assert n.tolist() == [1] and float(state.mass_dropped) == 1.0
    assert sorted(state.mass[0].tolist()) == [2.0, 4.0, 8.0]
    # and an age-0 puff beside a faint one still drops the faint one
    state, _ = emit(field, PuffState.empty((1,), 3),
                    torch.tensor([[1.0, 2.0**-30, 0.0]], dtype=DT), index, zero, tables=tabs)
    kept = housekeep(field, state, tabs)
    assert kept.live[0].tolist() == [True, False, False]
    assert float(kept.mass_dropped) == 2.0**-30


# ---------------------------------------------------------------- pulse releases
def _t(*v):
    return torch.tensor(v, dtype=DT)


def test_pulse_mass_sums_to_rate_times_duration_over_steps():
    start, duration, rate = _t(23.0), _t(95.0), _t(2.0)
    total = sum(pulse_mass(k * 10.0, 10.0, start, duration, rate) for k in range(30))
    assert torch.allclose(total, rate * duration, rtol=1e-12)
    m = pulse_mass(20.0, 10.0, _t(0.0, 25.0, 100.0), _t(50.0), _t(1.0))
    assert m.tolist() == [10.0, 5.0, 0.0]


def test_pulse_start_inside_a_step_has_the_central_difference_gradient():
    t, dt, h = 100.0, 10.0, 1e-4
    start = (t + 0.3 * dt) * torch.ones(1, dtype=DT)
    start.requires_grad_(True)
    duration, rate = _t(50.0), _t(3.0)
    m = pulse_mass(t, dt, start, duration, rate)
    (g,) = torch.autograd.grad(m.sum(), start)
    fd = (pulse_mass(t, dt, start.detach() + h, duration, rate)
          - pulse_mass(t, dt, start.detach() - h, duration, rate)) / (2 * h)
    assert torch.allclose(g, fd, rtol=1e-8)
    assert g.item() == pytest.approx(-3.0)
    # the whole window inside one step: the total is independent of start
    s2 = _t(103.0).requires_grad_(True)
    total = pulse_mass(100.0, 10.0, s2, _t(5.0), rate)
    (g2,) = torch.autograd.grad(total.sum(), s2)
    assert g2.item() == 0.0


def test_pulse_outside_window_is_zero_with_finite_grad():
    for t in (0.0, 900.0):
        start = _t(500.0).requires_grad_(True)
        duration = _t(20.0).requires_grad_(True)
        rate = _t(1.0).requires_grad_(True)
        m = pulse_mass(t, 10.0, start, duration, rate)
        assert m.item() == 0.0
        grads = torch.autograd.grad(m.sum(), (start, duration, rate))
        assert all(torch.isfinite(g).all() and g.item() == 0.0 for g in grads)


def test_zero_duration_pulse_is_zero_with_finite_grad():
    for t in (0.0, 10.0, 20.0):
        start = _t(15.0).requires_grad_(True)
        duration = _t(0.0).requires_grad_(True)
        rate = _t(2.0).requires_grad_(True)
        m = pulse_mass(t, 10.0, start, duration, rate)
        assert m.item() == 0.0
        grads = torch.autograd.grad(m.sum(), (start, duration, rate))
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        assert grads[0].item() == 0.0 and grads[2].item() == 0.0


def test_pulse_source_adds_to_existing_sources_and_refuses_missing_t():
    from noodl.apps.street_aq.network import build_model
    from noodl.apps.street_aq.puff import PulseSource
    from noodl.contracts import StepContext, closure_spec

    net, _ = twelve_street_grid()
    model, state, drivers = build_model(net, background="per_street")
    names = [s.name for s in net.streets]
    src = PulseSource(model.net, names)
    spec = closure_spec(src, index=0)
    assert spec.integrates and spec.state_keys == ()
    i = model.net.node_index(names[10])
    w = torch.zeros((1, len(names)), dtype=DT)
    w[0, 10] = 1.0
    drv = dict(drivers)
    drv.update({"pulse.weights": w, "pulse.rate": _t(4.0), "pulse.start": _t(5.0),
                "pulse.duration": _t(10.0)})
    pre = torch.zeros(model.net.n, dtype=DT)
    pre[i] = 1.0
    drv["street.sources"] = pre
    ctx = StepContext(dt=10.0, t=0.0)
    out = src(state, drv, ctx)["street.sources"]
    assert out.shape == (1, model.net.n)
    assert out[0, i].item() == pytest.approx(1.0 + 4.0 * 5.0 / 10.0)
    others = torch.ones(model.net.n, dtype=torch.bool)
    others[i] = False
    assert torch.count_nonzero(out[0][others]) == 0
    drv["street.sources"] = pre.unsqueeze(-1)
    assert src(state, drv, ctx)["street.sources"].shape == (1, model.net.n, 1)
    del drv["street.sources"]
    assert src(state, drv, ctx)["street.sources"][0, i].item() == pytest.approx(2.0)
    assert src(state, drv, None) == {}
    with pytest.raises(ValueError, match=r"Model\.step\(t=\.\.\.\)"):
        src(state, drv, StepContext(dt=10.0, t=None))


def test_pulse_source_on_a_real_model_step_delivers_the_pulse_mass():
    from noodl.apps.street_aq.network import build_model
    from noodl.apps.street_aq.puff import PulseSource
    from noodl.model import Model

    net, _ = twelve_street_grid()
    model, state, drivers = build_model(net, background="per_street")
    names = [s.name for s in net.streets]
    src = PulseSource(model.net, names)
    pulsed = Model(model.net, model.layers, closures=[*model.closures, src])
    w = torch.zeros((1, len(names)), dtype=DT)
    w[0, 3] = 1.0
    drv = dict(drivers)
    drv.update({"pulse.weights": w, "pulse.rate": _t(2.0), "pulse.start": _t(0.0),
                "pulse.duration": _t(30.0)})
    drv.update({"U_ref": _t(3.0).squeeze(), "theta_w": _t(0.0).squeeze(),
                "h_abl": _t(800.0).squeeze()})
    base = {k: v for k, v in drv.items() if not k.startswith("pulse.")}
    dt = 10.0
    x, x_ref = dict(state), dict(state)
    for k in range(5):
        x = pulsed.step(x, drv, dt, t=k * dt)
        x_ref = model.step(x_ref, base, dt, t=k * dt)
    extra = (x["street.x"] - x_ref["street.x"]).detach().reshape(-1)
    assert extra[3] > 0 and torch.isfinite(extra).all()
    assert extra[3] == extra.max()


# --- review fixes: flat-cell gradients, slug mass ---------------------------------------


def _flat_leaf_tables(u, n=20, spread=2.0):
    s = torch.full((1, 1, n), spread, dtype=DT)
    return PuffTables(u=u, sigma_y=s, sigma_z=s.clone(), z_c=torch.full_like(s, 15.0),
                      p_z=s.clone(), h_canopy=torch.zeros(1, dtype=DT),
                      h_abl=torch.full((1,), 1e6, dtype=DT), x_max=10.0 * (n - 1))


@pytest.mark.parametrize("bump", [0.0, 1e-7, 0.2], ids=["flat", "nearly_flat", "sloped"])
def test_distance_gradient_to_the_speed_table_matches_finite_differences(bump):
    # the travel-time ratios switch to their series near a flat cell: value AND derivative
    n = 20
    u0 = torch.full((1, 1, n), 3.0, dtype=DT)
    u0[0, 0, 7:] += bump
    age = torch.tensor([17.0, 26.0], dtype=DT)       # inside cells 5 and 7 at u = 3
    u = u0.clone().requires_grad_(True)
    x = _flat_leaf_tables(u).at_age(age, _ZERO, _ZERO)[0].sum()
    g, = torch.autograd.grad(x, u)
    h = 1e-6
    fd = torch.zeros(n, dtype=DT)
    for k in range(n):
        up, dn = u0.clone(), u0.clone()
        up[0, 0, k] += h
        dn[0, 0, k] -= h
        fd[k] = (_flat_leaf_tables(up).at_age(age, _ZERO, _ZERO)[0].sum()
                 - _flat_leaf_tables(dn).at_age(age, _ZERO, _ZERO)[0].sum()) / (2 * h)
    assert float(fd.abs().max()) > 1.0
    torch.testing.assert_close(g[0, 0], fd, rtol=0, atol=1e-6)


@pytest.mark.parametrize("along_wind", ["none", "sigma_y"])
@pytest.mark.parametrize("age", [3.0, 10.0, 40.0])
def test_one_slug_carries_its_mass_along_the_wind(age, along_wind):
    # speed 2 + 0.005 x, constant spreads: the line integral of C along the axis, divided
    # by the (constant) crosswind and vertical factors, is the mass -- including a puff
    # younger than dt / 2, whose slug starts at the source (ages clamped at 0)
    n, dt, sigma, z = 120, 10.0, 5.0, 15.0
    u = (2.0 + 0.05 * torch.arange(n, dtype=DT)).reshape(1, 1, n)
    s = torch.full((1, 1, n), sigma, dtype=DT)
    tabs = PuffTables(u=u, sigma_y=s, sigma_z=s.clone(), z_c=torch.full_like(s, z),
                      p_z=s.clone(), h_canopy=torch.zeros(1, dtype=DT),
                      h_abl=torch.full((1,), 1e6, dtype=DT), x_max=10.0 * (n - 1))
    field = PuffField.points(1, along_wind=along_wind, cutoff_sigma=10.0)
    state, _ = emit(field, PuffState.empty((1,), 2), torch.tensor([[2.0]], dtype=DT),
                    torch.zeros(1, dtype=_LONG), torch.zeros(1, 1, 2, dtype=DT))
    state = replace(state, age=torch.tensor([[age, 0.0]], dtype=DT),
                    disp=state.disp.index_put((torch.tensor([0]), torch.tensor([0])),
                                              tabs.at_age(torch.tensor([age], dtype=DT),
                                                          _ZERO, _ZERO)[0][:, None]
                                              * torch.tensor([[1.0, 0.0]], dtype=DT)))
    lo = float(tabs.at_age(torch.tensor(max(age - dt / 2, 0.0), dtype=DT), _ZERO, _ZERO)[0])
    hi = float(tabs.at_age(torch.tensor(age + dt / 2, dtype=DT), _ZERO, _ZERO)[0])
    pad = 10.0 * sigma if along_wind == "sigma_y" else 0.0
    m = 200_000
    edges = torch.linspace(lo - pad, hi + pad, m + 1, dtype=DT)
    xs = 0.5 * (edges[1:] + edges[:-1])                     # midpoint rule
    points = torch.stack([xs, torch.zeros_like(xs), torch.full_like(xs, z)], dim=1)
    c = concentration_at(field, state, tabs, points, 0.0, dt)[0]
    sig = torch.tensor(sigma, dtype=DT)
    factors = _gauss(torch.zeros((), dtype=DT), sig) * (
        _gauss(torch.zeros((), dtype=DT), sig) + _gauss(torch.tensor(2.0 * z, dtype=DT), sig))
    mass = float(c.sum() * (edges[1] - edges[0]) / factors)
    # midpoint-rule error at 200 000 points is below 1e-9 relative (measured <= 4.2e-11)
    assert mass == pytest.approx(2.0, rel=1e-9)
