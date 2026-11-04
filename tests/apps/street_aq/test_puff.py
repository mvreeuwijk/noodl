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
    age = torch.tensor([15.0, 10.0, 0.0, 10.0 + 1e-9], dtype=DT)
    x, u, _, _, _ = tabs.at_age(age, zero, zero)
    assert float(x[0]) == 0.5 * float(tabs.x[0, 0, 1] + tabs.x[0, 0, 2])
    assert float(u[0]) == float(tabs.u[0, 0, 2])            # segment (10, 20]
    assert float(u[1]) == float(tabs.u[0, 0, 1])            # segments are right-closed
    assert float(u[2]) == float(tabs.u[0, 0, 1])
    assert float(u[3]) == float(tabs.u[0, 0, 2])
    # the displacement is advanced on the segment speed: x_j = x_{j-1} + u_j dt
    torch.testing.assert_close(tabs.x[0, 0, 1:] - tabs.x[0, 0, :-1],
                               tabs.u[0, 0, 1:] * 10.0, rtol=1e-13, atol=0)


def test_stepping_age_and_summing_the_speed_reproduces_the_table_distance():
    tabs = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_PUFF_MET)
    zero = torch.zeros((), dtype=torch.long)
    disp = torch.zeros((), dtype=DT)
    age = torch.zeros((), dtype=DT)
    for j in range(1, tabs.x.shape[-1]):
        age = age + 10.0
        disp = disp + tabs.at_age(age, zero, zero)[1] * 10.0
        torch.testing.assert_close(disp, tabs.x[0, 0, j], rtol=1e-13, atol=0)


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


# --- the puff field: emit, advance, evaluate ---------------------------------------------

_LONG = torch.long


def _hand_tables(speeds, *, n_nodes=90, z_c=15.0, h_canopy=0.0, h_abl=1e6):
    """A hand-made `PuffTables` with spreads `0.5 + 0.2 t` and node speeds `speeds(j)`
    (one row per entry of `speeds`; `x_j = x_{j-1} + 10 u_j`, node 0 holds `u_1`)."""
    t = torch.arange(n_nodes, dtype=DT) * 10.0
    rows_x, rows_u = [], []
    for speed in speeds:
        u = torch.tensor([speed(max(j, 1)) for j in range(n_nodes)], dtype=DT)
        x = torch.cat([torch.zeros(1, dtype=DT), torch.cumsum(u[1:] * 10.0, 0)])
        rows_x.append(x)
        rows_u.append(u)
    n_b = len(speeds)
    sigma = (0.5 + 0.2 * t).expand(1, n_b, n_nodes).clone()
    return PuffTables(
        x=torch.stack(rows_x)[None], u=torch.stack(rows_u)[None], sigma_y=sigma,
        sigma_z=sigma.clone(), z_c=torch.full((1, n_b, n_nodes), z_c, dtype=DT),
        h_canopy=torch.full((n_b,), h_canopy, dtype=DT),
        h_abl=torch.full((n_b,), h_abl, dtype=DT), x_max=1e9)


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
    # constant u = 3, sigma_y = sigma_z = 0.5 + 0.2 t, z_c = 15, H_R = 0 (the canopy image
    # is the ground image), h = 1e6; one puff of 2 kg advanced to age 60 s in steps of 5 s
    tabs = _hand_tables([lambda j: 3.0])
    field = PuffField.points(1, along_wind="sigma_y")
    theta, origin = 0.4, (30.0, -20.0)
    state = _one_release(field, tabs, mass=2.0, origin=origin, dt=5.0, steps=12, theta=theta)
    assert float(state.age[0, 0]) == 60.0
    points, a, c, z = _wind_points(origin, theta, [168.0, 177.0, 180.0, 186.0, 200.0],
                                   [0.0, 4.0, -9.0], [1.0, 15.0, 25.0])
    sigma = 0.5 + 0.2 * 60.0
    # the slug spans x(57.5) - x(60) = -7.5 m to x(62.5) - x(60) = +7.5 m: ell = 15 m
    lo, hi, ell = -7.5, 7.5, 15.0
    u = a - 180.0
    root2 = math.sqrt(2.0) * sigma
    g_x = (torch.erf((u - lo) / root2) - torch.erf((u - hi) / root2)) / (2.0 * ell)
    vertical = _gauss(z - 15.0, sigma) + _gauss(z + 15.0, sigma)
    expected = 2.0 * g_x * _gauss(c, sigma) * vertical
    th = torch.tensor([theta], dtype=DT)
    got = concentration_at(field, state, tabs, points, th, 5.0)
    assert got.shape == (1, points.shape[0])
    torch.testing.assert_close(got[0], expected, rtol=1e-13, atol=0)
    # ell -> 0: the plain Gaussian puff
    tiny = concentration_at(field, state, tabs, points, th, 1e-4)
    plain = 2.0 * _gauss(u, sigma) * _gauss(c, sigma) * vertical
    torch.testing.assert_close(tiny[0], plain, rtol=1e-8, atol=0)


def test_slug_without_along_wind_spread_is_a_top_hat():
    tabs = _hand_tables([lambda j: 3.0])
    field = PuffField.points(1, along_wind="none")
    theta, origin = 1.1, (0.0, 0.0)
    state = _one_release(field, tabs, mass=2.0, origin=origin, dt=5.0, steps=12, theta=theta)
    # slug [172.5, 187.5) m from the origin
    inside, _, _, _ = _wind_points(origin, theta, [172.6, 177.0, 180.0, 187.4], [2.0], [15.0])
    outside, _, _, _ = _wind_points(origin, theta, [172.4, 187.6, 210.0, 100.0], [2.0], [15.0])
    th = torch.tensor([theta], dtype=DT)
    sigma = torch.tensor(12.5, dtype=DT)
    level = 2.0 / 15.0 * _gauss(torch.tensor(2.0, dtype=DT), sigma) * (
        _gauss(torch.tensor(0.0, dtype=DT), sigma) + _gauss(torch.tensor(30.0, dtype=DT), sigma))
    torch.testing.assert_close(concentration_at(field, state, tabs, inside, th, 5.0)[0],
                               level.expand(4), rtol=1e-13, atol=0)
    assert torch.equal(concentration_at(field, state, tabs, outside, th, 5.0),
                       torch.zeros(1, 4, dtype=DT))
    # beyond cutoff_sigma crosswind standard deviations the puff is exactly zero
    far, _, _, _ = _wind_points(origin, theta, [180.0], [4.0 * 12.5 + 0.01], [15.0])
    assert float(concentration_at(field, state, tabs, far, th, 5.0)) == 0.0


def _x_at(x, t):
    """The distance `x` (nodes every 10 s) at ages `t`, independent of `at_age`."""
    pos = t / 10.0
    j = torch.clamp(torch.floor(pos).long(), 0, x.numel() - 2)
    return x[j] + (x[j + 1] - x[j]) * (pos - j)


@pytest.mark.parametrize("speed", [lambda j: 3.0, lambda j: 2.0 + 0.05 * j],
                         ids=["constant", "growing"])
@pytest.mark.parametrize("dt", [2.0, 5.0, 10.0])
def test_continuous_source_tiles_to_one_over_u(dt, speed):
    tabs = _hand_tables([speed])
    field = PuffField.points(1, along_wind="none")
    theta = 0.7
    th = torch.tensor([theta], dtype=DT)
    state = PuffState.empty((1,), 80)
    for _ in range(60):  # unit rate: mass dt per step
        state, _ = emit(field, state, torch.tensor([[dt]], dtype=DT),
                        torch.zeros(1, dtype=_LONG), torch.zeros(1, 1, 2, dtype=DT))
        state = advance(state, tabs, dt, th)
    # receptors at ages (k + f) dt along the axis: the slug of the puff of age k dt covers
    # them (a slug spans [x(a - dt/2), x(a + dt/2)])
    ks = torch.arange(2, 58, dtype=DT)
    fs = torch.tensor([-0.4, -0.1, 0.2, 0.45], dtype=DT)
    age_c = (ks[:, None] * dt).expand(-1, 4).reshape(-1)
    x_tab = tabs.x[0, 0]
    x_r = _x_at(x_tab, ((ks[:, None] + fs[None]) * dt).reshape(-1))
    e = torch.tensor([math.cos(theta), math.sin(theta)], dtype=DT)
    z = 3.0
    points = torch.cat([x_r[:, None] * e, torch.full((x_r.numel(), 1), z, dtype=DT)], dim=1)
    got = concentration_at(field, state, tabs, points, th, dt)[0]
    sigma = 0.5 + 0.2 * age_c
    zz = torch.full_like(sigma, z)
    profile = _gauss(torch.zeros_like(sigma), sigma) * (
        _gauss(zz - 15.0, sigma) + _gauss(zz + 15.0, sigma))
    ell = _x_at(x_tab, age_c + dt / 2) - _x_at(x_tab, age_c - dt / 2)
    torch.testing.assert_close(got, profile * dt / ell, rtol=1e-12, atol=0)
    # where the covering slug lies within one table segment, dt / ell is exactly 1/u there
    seg_lo = torch.floor((age_c - dt / 2) / 10.0)
    seg_hi = torch.ceil((age_c + dt / 2) / 10.0)
    within = seg_hi - seg_lo == 1
    u_seg = tabs.u[0, 0][seg_hi.long()]
    if dt < 10.0:
        assert bool(within.any())
        torch.testing.assert_close(got[within], (profile / u_seg)[within], rtol=1e-12, atol=0)
    if speed(1) == speed(7):
        torch.testing.assert_close(got, profile / 3.0, rtol=1e-12, atol=0)


def test_receptor_owner_excludes_own_puffs():
    tabs = _hand_tables([lambda j: 3.0], h_canopy=2.0)
    field = PuffField.points(2)
    state = PuffState.empty((1,), 4)
    state, _ = emit(field, state, torch.tensor([[1.0, 3.0]], dtype=DT),
                    torch.zeros(2, dtype=_LONG),
                    torch.tensor([[[0.0, 0.0], [5.0, 0.0]]], dtype=DT))
    th = torch.tensor([0.0], dtype=DT)
    for _ in range(6):
        state = advance(state, tabs, 5.0, th)
    xy = torch.tensor([[95.0, 1.0]] * 3, dtype=DT)
    owner = torch.tensor([0, 1, -1], dtype=_LONG)
    rc = roof_concentration(field, state, tabs, xy, th, 5.0, receptor_owner=owner)[0]
    free = concentration_at(field, state, tabs, torch.cat(
        [xy, torch.full((3, 1), 2.0, dtype=DT)], dim=1), th, 5.0)[0]
    assert float(rc[0]) > 0 and float(rc[1]) > 0
    torch.testing.assert_close(rc[0] + rc[1], rc[2], rtol=1e-14, atol=0)
    torch.testing.assert_close(rc[2], free[2], rtol=1e-14, atol=0)  # no clamp on points
    assert torch.equal(roof_concentration(field, state, tabs, xy, th, 5.0,
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
    tabs = _hand_tables([lambda j: 3.0, lambda j: 1.0 + 0.1 * j])
    field = PuffField.points(1)
    theta = torch.tensor([0.3, 2.0], dtype=DT)
    state = PuffState.empty((2,), 3)
    state, _ = emit(field, state, torch.tensor([[1.0], [2.0]], dtype=DT),
                    torch.zeros(1, dtype=_LONG), torch.zeros(2, 1, 2, dtype=DT))
    for _ in range(7):  # 3 s steps cross the 10 s nodes
        state = advance(state, tabs, 3.0, theta)
    assert torch.equal(state.age[:, 0], torch.full((2,), 21.0, dtype=DT))
    x = torch.stack([_x_at(tabs.x[0, b], torch.tensor(21.0, dtype=DT)) for b in range(2)])
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


def test_along_wind_option_is_checked_by_name():
    with pytest.raises(ValueError, match="along_wind"):
        PuffField.points(1, along_wind="gaussian")


def test_gradient_reaches_mass_and_release_position():
    tabs = _hand_tables([lambda j: 3.0])
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
    tabs = tabs or _hand_tables([lambda j: 3.0])
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
    # beyond the table's last node (age > t_max)
    old = replace(state, age=torch.tensor([[tabs.t_max + 1.0, 5.0] + [0.0] * 6], dtype=DT))
    out = housekeep(field, old, tabs)
    assert out.live[0].tolist() == [False, True] + [False] * 6
    assert float(out.mass_out) == 4.0 and float(out.mass[0, 0]) == 0.0
    _identity(out)
    # beyond the extent x_max, before the last node
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
    tabs = _hand_tables([lambda j: 3.0])
    state, _ = emit(field, PuffState.empty((1,), 4), torch.ones(1, 2, dtype=DT),
                    torch.zeros(2, dtype=_LONG), torch.zeros(1, 2, 2, dtype=DT))
    kept = housekeep(field, advance(state, tabs, 5.0, torch.zeros(1, dtype=DT)), tabs)
    assert kept.live[0, :2].tolist() == [True, False]
    _identity(kept)


def test_capacity_eviction_reports_count_and_keeps_identity():
    for with_tables in (False, True):
        field = PuffField.points(3)
        tabs = _hand_tables([lambda j: 3.0])
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
    tabs = _hand_tables([lambda j: 3.0])
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
    old = replace(state, age=torch.full((2, 3), tabs.t_max + 1.0, dtype=DT))
    out = housekeep(field, old, tabs)
    g, = torch.autograd.grad(out.mass_out[1], mass)
    assert g.tolist() == [[0.0], [1.0]]


def test_puff_width_batch_must_match_by_name():
    field = PuffField.points(1)
    field.width = torch.ones(3, 1, dtype=DT)
    state = PuffState.empty((2,), 2)
    with pytest.raises(ValueError, match="width"):
        roof_concentration(field, state, _hand_tables([lambda j: 3.0]),
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
