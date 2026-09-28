"""SIRANE's above-roof street-plume kernel: the trajectory table, the sources and the
evaluation (`noodl.apps.street_aq.plume`).

Hand values are written out with `math` from the mechanism in the module docstring. The
parity values are from an independent numpy implementation of the same mechanism (a
Python loop over the 10 s nodes and `numpy.interp` for both resamplings), pinned here.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from noodl.apps.street_aq import plume
from noodl.apps.street_aq.canyon import KAPPA
from noodl.apps.street_aq.network import Street, StreetNetwork, from_test_network
from noodl.apps.street_aq.plume import (
    KERNEL_BYTES_CAP,
    SEUIL_GAUSS,
    default_tau,
    junction_kernel,
    junction_sources,
    plume_kernel,
    plume_table,
    rows_per_chunk,
    source_points,
    street_kernel,
    street_midpoints,
    turbulence,
    wind_speed,
)

DT = torch.float64
SITE = {"z0": 1.0, "d": 13.0}
NEUTRAL = {"u_star": 0.45, "h_abl": 800.0, "lmo": math.inf, "h_canopy": 20.0,
           "sigma_theta": math.radians(6.0), **SITE}
STABLE = {"u_star": 0.2, "h_abl": 250.0, "lmo": 80.0, "h_canopy": 25.0,
          "sigma_theta": math.radians(8.0), "theta_star": 0.05, "temperature": 277.15,
          **SITE}


def _one_street(x0=-30.0, y0=-80.0, x1=40.0, y1=90.0, width=20.0, height=15.0):
    return StreetNetwork([Street("s", "a", "b", math.hypot(x1 - x0, y1 - y0), width,
                                 height)], {"a": x0, "b": x1}, {"a": y0, "b": y1})


def _downwind(theta: float, x: float, ys: torch.Tensor) -> torch.Tensor:
    """Receptors at downwind distance `x` and crosswind offsets `ys` from the origin."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.stack([x * c - ys * s, x * s + ys * c], dim=-1)


def _u(z, met):
    return float(wind_speed(z, u_star=met["u_star"], lmo=met["lmo"], z0=met["z0"],
                            d=met["d"]))


# ------------------------------------------------------------------------ meteorology


def test_neutral_and_stable_turbulence_by_hand():
    z, h, us = 20.0, 800.0, 0.45
    sv, sw, n = turbulence(z, u_star=us, h_abl=h, lmo=math.inf)
    assert float(sv) == pytest.approx(2 * us * (1 - 0.8 * z / h), rel=1e-12)
    assert float(sw) == pytest.approx(1.3 * us * (1 - 0.8 * z / h), rel=1e-12)
    assert float(n) == 0.0
    z, h, us, L = 25.0, 250.0, 0.2, 80.0
    sv, sw, n = turbulence(z, u_star=us, h_abl=h, lmo=L)
    assert float(sv) == pytest.approx(2 * us * (1 - 0.5 * z / h) ** 0.75, rel=1e-12)
    assert float(sw) == pytest.approx(1.3 * us * (1 - 0.5 * z / h) ** 0.75, rel=1e-12)
    # default theta* = u*^2 T / (kappa g L): N is independent of T
    assert float(n) == pytest.approx(math.sqrt(us**2 / (KAPPA**2 * L) * (1 / z + 5 / L)),
                                     rel=1e-12)
    theta_star = us**2 * 280.0 / (KAPPA * plume.GRAVITY * L)
    _, _, n_t = turbulence(z, u_star=us, h_abl=h, lmo=L, theta_star=theta_star,
                           temperature=280.0)
    assert float(n_t) == pytest.approx(float(n), rel=1e-12)
    _, _, n_o = turbulence(z, u_star=us, h_abl=h, lmo=L, theta_star=0.06, temperature=277.15)
    assert float(n_o) == pytest.approx(
        math.sqrt(plume.GRAVITY / 277.15 * 0.06 / KAPPA * (1 / z + 5 / L)), rel=1e-12)


def test_neutral_band_and_bad_lengths():
    sv, _, n = turbulence(30.0, u_star=0.5, h_abl=800.0, lmo=5e4, neutral_abs_lmo=1e4)
    assert float(n) == 0.0
    assert float(sv) == pytest.approx(2 * 0.5 * (1 - 0.8 * 30 / 800), rel=1e-12)
    for bad in (0.0, math.nan):
        with pytest.raises(ValueError, match="non-zero and not NaN"):
            turbulence(30.0, u_star=0.5, h_abl=800.0, lmo=bad)


def test_wind_speed_profile():
    u = wind_speed(30.0, u_star=0.5, lmo=math.inf, z0=1.0, d=13.0)
    assert float(u) == pytest.approx(0.5 / KAPPA * math.log(18.0), rel=1e-12)
    u_s = wind_speed(30.0, u_star=0.5, lmo=100.0, z0=1.0, d=13.0)
    assert float(u_s) == pytest.approx(
        0.5 / KAPPA * (math.log(18.0) + 5 * 18.0 / 100.0 - 5 * 1.0 / 100.0), rel=1e-12)
    L = -100.0
    u_u = wind_speed(30.0, u_star=0.5, lmo=L, z0=1.0, d=13.0)

    def psi(zeta):
        xx = (1 - 16 * zeta) ** 0.25
        return (2 * math.log((1 + xx) / 2) + math.log((1 + xx * xx) / 2)
                - 2 * math.atan(xx) + math.pi / 2)

    assert float(u_u) == pytest.approx(
        0.5 / KAPPA * (math.log(18.0) - psi(18.0 / L) + psi(1.0 / L)), rel=1e-12)


def test_default_time_offsets_follow_the_tables():
    ty, tz = default_tau(torch.tensor([0.1, 0.26, 0.5, 0.885, 5.0], dtype=DT),
                         torch.tensor([0.1, 0.169, 0.30, 0.574, 5.0], dtype=DT))
    # constant beyond the end knots, exact at the knots, linear between
    assert ty.tolist() == pytest.approx([1.03, 1.03, 1.86, (2.03 + 2.72) / 2, 3.26],
                                        abs=1e-12)
    assert tz.tolist() == pytest.approx([-0.89, -0.89, 0.56, (0.64 + 1.53) / 2, 2.20],
                                        abs=1e-12)


# ------------------------------------------------------------------------ the table


def test_first_nodes_and_grid_by_hand():
    """Nodes 0-2 of a neutral trajectory and the 10 m grid between them, from the
    mechanism written out with `math`."""
    met, tau_y, tau_z = NEUTRAL, 2.5, 1.2
    t = plume_table(**met, sigma_v_min=0.0, sigma_w_min=0.0, tau_y=tau_y, tau_z=tau_z)
    hr, d, h, us, sth = met["h_canopy"], met["d"], met["h_abl"], met["u_star"], \
        met["sigma_theta"]
    sv, sw = 2 * us * (1 - 0.8 * hr / h), 1.3 * us * (1 - 0.8 * hr / h)
    a = 2.5 * us / h
    u0 = _u(hr, met)
    # node 1: the previous sigma_z is zero, so z_c = max(H_R, d + |c - d|) = max(20, 16)
    u1 = _u(hr, met)
    x1 = 10 * u1
    sz1 = 0.4 * sw * (10 - tau_z)
    ty1 = 10 - tau_y
    sy1 = math.sqrt((sv * ty1) ** 2 / (1 + a * ty1) + (sth * (x1 - u0 * tau_y)) ** 2)
    pz1 = (2 + math.exp(-0.5 * ((2 * h - 2 * hr) / sz1) ** 2)) / (math.sqrt(2 * math.pi) * sz1)
    # node 2: z_c from node 1's sigma_z
    m, s = plume.PLUME_CENTRE_C_M - d, plume.PLUME_CENTRE_K * sz1
    zc2 = max(hr, d + s * math.sqrt(2 / math.pi) * math.exp(-0.5 * (m / s) ** 2)
              + m * math.erf(m / (s * math.sqrt(2))))
    u2 = _u(max(math.floor(zc2), hr), met)
    x2 = x1 + 10 * u2
    sz2 = 0.4 * sw * (20 - tau_z)
    ty2 = 20 - tau_y
    sy2 = math.sqrt((sv * ty2) ** 2 / (1 + a * ty2) + (sth * (x2 - u0 * tau_y)) ** 2)
    g = math.exp(-0.5 * ((zc2 - hr) / sz2) ** 2)
    pz2 = (2 * g + math.exp(-0.5 * ((2 * h - zc2 - hr) / sz2) ** 2)) / (
        math.sqrt(2 * math.pi) * sz2)
    # grid points 10 and 20 m lie between node 0 (x = 0, sigma_y = 0.1, P_z = P_z1) and
    # node 1 (x1 = 23.4 m); 30 and 40 m between nodes 1 and 2
    assert 20.0 < x1 < 30.0 and 40.0 < x2
    for k, xg in ((1, 10.0), (2, 20.0)):
        f = xg / x1
        assert float(t.sigma_y[0, k]) == pytest.approx(0.1 + (sy1 - 0.1) * f, rel=1e-12)
        assert float(t.p_z[0, k]) == pytest.approx(pz1, rel=1e-12)
        assert float(t.u[0, k]) == pytest.approx(u0, rel=1e-12)
    for k, xg in ((3, 30.0), (4, 40.0)):
        f = (xg - x1) / (x2 - x1)
        assert float(t.sigma_y[0, k]) == pytest.approx(sy1 + (sy2 - sy1) * f, rel=1e-12)
        assert float(t.p_z[0, k]) == pytest.approx(pz1 + (pz2 - pz1) * f, rel=1e-12)
        assert float(t.u[0, k]) == pytest.approx(u1 + (u2 - u1) * f, rel=1e-12)
    assert float(t.sigma_y[0, 0]) == plume.SIGMA_Y0_M


def test_table_speeds_step_through_integer_heights_and_cover_the_cutoff():
    t = plume_table(**NEUTRAL)
    grid = t.distance
    assert float(grid[-1]) == plume.DOWNWIND_CUTOFF_M and t.step == plume.TABLE_STEP_M
    # between two nodes the speed interpolates two values U(n), U(n'), n, n' integers;
    # at grid points that coincide with a speed plateau it is exactly U(integer)
    speeds = {round(_u(z, NEUTRAL), 12) for z in range(20, 200)}
    on_plateau = [round(float(v), 12) in speeds for v in t.u[0]]
    assert sum(on_plateau) > 10
    assert float(t.u[0].min()) == pytest.approx(_u(20.0, NEUTRAL), rel=1e-12)
    assert bool((t.u[0, 1:] >= t.u[0, :-1] - 1e-12).all())      # never slows down
    # a non-multiple cut-off adds one grid point past it
    t2 = plume_table(**NEUTRAL, x_max=655.0)
    assert float(t2.distance[-1]) == 660.0


def test_table_floors_and_refusals():
    base = plume_table(**NEUTRAL, sigma_v_min=0.0, sigma_w_min=0.0)
    floored = plume_table(**NEUTRAL, sigma_v_min=5.0, sigma_w_min=5.0)
    assert not torch.equal(base.sigma_y, floored.sigma_y)
    with pytest.raises(ValueError, match="u_star must be strictly positive"):
        plume_table(**{**NEUTRAL, "u_star": 0.0})
    with pytest.raises(ValueError, match="must exceed the reflection height"):
        plume_table(**{**NEUTRAL, "h_abl": 20.0})
    with pytest.raises(ValueError, match="must exceed the displacement"):
        plume_table(**{**NEUTRAL, "d": 20.0})
    with pytest.raises(ValueError, match="tau_z must be below"):
        plume_table(**NEUTRAL, tau_z=10.0)
    with pytest.raises(NotImplementedError, match="unstable"):
        plume_table(**{**NEUTRAL, "lmo": -50.0})
    with pytest.raises(ValueError, match="x_max must be positive"):
        plume_table(**NEUTRAL, x_max=0.0)


def test_batched_table_matches_single_hours():
    us = torch.tensor([0.3, 0.45, 0.8], dtype=DT)
    lmo = torch.tensor([math.inf, 150.0, 60.0], dtype=DT)
    t = plume_table(**{**NEUTRAL, "u_star": us, "lmo": lmo})
    assert t.batch == (3,) and t.u.shape[0] == 3
    for b in range(3):
        one = plume_table(**{**NEUTRAL, "u_star": float(us[b]), "lmo": float(lmo[b])})
        for key in ("sigma_y", "p_z", "u"):
            assert torch.allclose(getattr(t, key)[b], getattr(one, key)[0], rtol=1e-14,
                                  atol=0)


def test_a_calm_hour_in_a_long_batch_is_chunked_and_matches_single_hours(monkeypatch):
    """A batch with one calm hour (hundreds of 10 s steps) among fast ones: each step
    advances only the hours still short of the grid's end, the batch is built in chunks
    under the cap (here made small), and every hour equals its own single-hour table, with
    and without a recorded gradient."""
    us = torch.tensor([0.5, 0.02, 0.8, 0.3, 0.45, 0.6], dtype=DT)
    steps = plume.table_chunks([10.0] * 6)
    assert steps == [(0, 6)]
    monkeypatch.setattr(plume, "KERNEL_BYTES_CAP", 60 * plume._TABLE_BYTES_PER_STEP)
    assert plume.table_chunks([30.0, 450.0, 20.0, 40.0, 25.0, 25.0]) == [
        (0, 1), (1, 2), (2, 4), (4, 6)]
    batched = plume_table(**{**NEUTRAL, "u_star": us})
    for b in range(6):
        one = plume_table(**{**NEUTRAL, "u_star": float(us[b])})
        for key in ("sigma_y", "p_z", "u"):
            assert torch.allclose(getattr(batched, key)[b], getattr(one, key)[0],
                                  rtol=1e-14, atol=0)
    u = us.clone().requires_grad_(True)
    t = plume_table(**{**NEUTRAL, "u_star": u})
    (g,) = torch.autograd.grad((t.u + t.p_z + t.sigma_y).sum(), u)
    for b in (1, 4):
        ub = torch.tensor(float(us[b]), dtype=DT, requires_grad=True)
        tb = plume_table(**{**NEUTRAL, "u_star": ub})
        (gb,) = torch.autograd.grad((tb.u + tb.p_z + tb.sigma_y).sum(), ub)
        assert float(g[b]) == pytest.approx(float(gb), rel=1e-12)


def test_downwind_cutoff_follows_the_meteo_cell():
    """`meteo_cell_dx / |cos theta|`: 700 m cells give 700 m along x and ~990 m at 45
    degrees; 900 m cells 900 m; the South Kensington cell (750 m) at 315 degrees 1061 m."""
    assert float(plume.downwind_cutoff(0.0, 700.0)) == pytest.approx(700.0, rel=1e-12)
    assert float(plume.downwind_cutoff(math.pi, 900.0)) == pytest.approx(900.0, rel=1e-12)
    assert float(plume.downwind_cutoff(math.radians(135.0), 750.0)) == pytest.approx(
        1060.66, abs=0.01)
    with pytest.raises(ValueError, match="meteo_cell_dx must be positive"):
        plume.downwind_cutoff(0.0, 0.0)
    theta = math.radians(135.0)
    t = plume_table(**NEUTRAL, x_max=float(plume.downwind_cutoff(theta, 750.0)))
    src = torch.zeros(1, 2, dtype=DT)
    rec = _downwind(theta, 1000.0, torch.tensor([0.0], dtype=DT))
    inside = plume_kernel(rec, src, t, theta_w=theta, meteo_cell_dx=750.0)
    assert float(inside) > 0
    assert float(plume_kernel(rec, src, t, theta_w=theta, meteo_cell_dx=700.0)) == 0.0
    beyond = _downwind(theta, 1065.0, torch.tensor([0.0], dtype=DT))
    assert float(plume_kernel(beyond, src, t, theta_w=theta, meteo_cell_dx=750.0)) == 0.0
    # without the cell, the table's extent is the cut-off
    assert float(plume_kernel(rec, src, t, theta_w=theta)) == float(inside)
    with pytest.raises(ValueError, match="beyond the table's x_max"):
        plume_kernel(rec, src, plume_table(**NEUTRAL), theta_w=theta, meteo_cell_dx=750.0)
    with pytest.raises(ValueError, match="beyond the table's x_max"):
        street_kernel(from_test_network(), plume_table(**NEUTRAL), theta_w=theta,
                      meteo_cell_dx=750.0)


# ------------------------------------------------------------------------ sources


def test_street_subsources_count_positions_widths_and_clamps():
    for length, n in ((101.0, 11), (150.0, 16), (199.0, 20), (200.0, 21), (5.0, 1)):
        src = source_points(_one_street(0.0, 0.0, 0.0, length), theta_w=0.0)
        assert src.xy.shape[0] == n
        assert float(src.weight.sum()) == pytest.approx(1.0, abs=1e-14)
        # at the centres of n equal segments
        centres = (torch.arange(n, dtype=DT) + 0.5) * length / n
        assert torch.allclose(src.xy[:, 1], centres, rtol=0, atol=1e-12)
        # perpendicular to the wind: width L/n, clamp 10/W
        assert torch.allclose(src.width, torch.full((n,), length / n, dtype=DT))
        assert torch.allclose(src.clamp, torch.full((n,), 10.0 / 20.0, dtype=DT))


@pytest.mark.parametrize("angle_deg, measured", [(10, 0.0605), (20, 0.0758), (30, 0.1000),
                                                  (45, 0.1707)])
def test_oblique_clamp_is_the_measured_cap(angle_deg, measured):
    """The cap `1/(H (1 - |sin phi|))` against the plateaus SIRANE shows for a street at
    `phi` to the wind (H = 20 m, W = 20 m, read off the probe grids to 4 decimals)."""
    phi = math.radians(angle_deg)
    street = _one_street(0.0, 0.0, 100.0 * math.cos(phi), 100.0 * math.sin(phi),
                         width=20.0, height=20.0)
    src = source_points(street, theta_w=0.0)
    n = src.xy.shape[0]
    assert float(src.clamp[0]) == pytest.approx(measured, abs=6e-4)
    assert float(src.width[0]) == pytest.approx(100.0 / n * math.sin(phi), rel=1e-12)
    # from 75 degrees the 10/W cap takes over
    steep = source_points(_one_street(0.0, 0.0, 26.0, 97.0, width=20.0, height=20.0),
                          theta_w=0.0)
    assert float(steep.clamp[0]) == pytest.approx(0.5, rel=1e-12)


def test_a_street_along_the_wind_has_zero_width_and_the_height_cap():
    src = source_points(_one_street(0.0, 0.0, 100.0, 0.0), theta_w=0.0)
    assert torch.equal(src.width, torch.zeros(11, dtype=DT))
    assert float(src.clamp[0]) == pytest.approx(1.0 / 15.0, rel=1e-12)   # 1/(H (1 - 0))


def test_batched_directions_give_batched_widths():
    theta = torch.tensor([0.0, 0.3, 1.0], dtype=DT)
    src = source_points(_one_street(), theta_w=theta)
    assert src.width.shape == (3, src.xy.shape[0])
    for b in range(3):
        one = source_points(_one_street(), theta_w=float(theta[b]))
        assert torch.equal(src.width[b], one.width) and torch.equal(src.clamp[b], one.clamp)


def test_junction_sources_take_the_mean_width_and_height():
    net = from_test_network()
    xy, width, clamp = junction_sources(net)
    for k, j in enumerate(net.junctions):
        meeting = [s for s in net.streets if j in (s.u, s.v)]
        assert float(width[k]) == pytest.approx(
            sum(s.width for s in meeting) / len(meeting), rel=1e-12)
        assert float(clamp[k]) == pytest.approx(
            len(meeting) / sum(s.height for s in meeting), rel=1e-12)
        assert xy[k].tolist() == [net.x[j], net.y[j]]


# ------------------------------------------------------------------------ evaluation


def test_crosswind_profile_is_a_normalised_flat_top():
    """At fixed `x`, `U * integral(C dy) = min(P_z, clamp)` for a flat-topped source
    (without the crosswind cut-off), and the flat top holds `1/w` when it is wider than
    `sqrt(2 pi) sigma_y`."""
    t = plume_table(**NEUTRAL)
    theta, x, w, cap = 0.7, 30.0, 60.0, 0.02
    sy, pz, u = (float(v) for v in t.lookup(torch.tensor(x, dtype=DT)))
    assert w > math.sqrt(2 * math.pi) * sy
    half = w / 2 + 12 * sy
    ys = torch.linspace(-half, half, 20001, dtype=DT)
    k = plume_kernel(_downwind(theta, x, ys), torch.zeros(1, 2, dtype=DT), t,
                     theta_w=theta, width=w, clamp=cap, cutoff_sigma=math.inf)[:, 0]
    integral = float(torch.trapezoid(k, ys)) * u
    assert integral == pytest.approx(min(pz, cap), rel=1e-7)
    assert float(k[10000]) == pytest.approx(min(pz, cap) / (w * u), rel=1e-12)
    # a point source (w = 0) is the plain Gaussian
    k0 = plume_kernel(_downwind(theta, x, torch.tensor([0.0, sy], dtype=DT)),
                      torch.zeros(1, 2, dtype=DT), t, theta_w=theta)[:, 0]
    peak = pz / (math.sqrt(2 * math.pi) * sy * u)
    assert float(k0[0]) == pytest.approx(peak, rel=1e-12)
    assert float(k0[1]) == pytest.approx(peak * math.exp(-0.5), rel=1e-12)


def test_cutoffs_upwind_downwind_and_crosswind():
    t = plume_table(**NEUTRAL)
    theta, w = 1.1, 30.0
    src = torch.zeros(1, 2, dtype=DT)
    rec = torch.cat([
        _downwind(theta, -50.0, torch.tensor([0.0], dtype=DT)),
        _downwind(theta, 0.0, torch.tensor([0.0], dtype=DT)),
        _downwind(theta, 699.9, torch.tensor([0.0], dtype=DT)),
        _downwind(theta, 700.1, torch.tensor([0.0], dtype=DT)),
    ])
    k = plume_kernel(rec, src, t, theta_w=theta, width=w)[:, 0]
    assert k[0] == 0 and k[1] == 0 and k[2] > 0 and k[3] == 0
    # crosswind: zero beyond 4 sigma_y past the flat top's edge
    x = 150.0
    sy = float(t.lookup(torch.tensor(x, dtype=DT))[0])
    y_c = max(0.0, (w - math.sqrt(2 * math.pi) * sy) / 2)
    offs = torch.tensor([3.999, 4.001, -3.999, -4.001], dtype=DT)
    ys = torch.sign(offs) * (y_c + offs.abs() * sy)
    kc = plume_kernel(_downwind(theta, x, ys), src, t, theta_w=theta, width=w)[:, 0]
    assert kc[0] > 0 and kc[2] > 0 and kc[1] == 0 and kc[3] == 0
    uncut = plume_kernel(_downwind(theta, x, ys), src, t, theta_w=theta, width=w,
                         cutoff_sigma=math.inf)[:, 0]
    assert kc[0] == uncut[0] and uncut[1] > 0


# (receptor, street kernel, junction kernel) values of the independent implementation, per
# unit rate, s/m3, for the street (-30, -80) -> (40, 90) (W = 20 m, H = 15 m) and a junction
# of width 17.5 m and clamp 1/15 at (40, 90), wind 0.4 rad, table grid at 10, 50, 300 m.
# That implementation was run with: the plume-centre law z_c = max(H_R, d + E|N(c - d,
# (k sigma_z,prev)^2)|) with c = 10 m, k = 0.675 from the previous node's sigma_z; the node
# speed U(max(floor(z_c), H_R)); both images (canopy top and inversion); sigma_y0 = 0.1 m;
# 10 s steps; the 10 m grid; the 700 m cut-off; z0 = 1 m, d = 13 m; the explicit time
# offsets below. In "stable_low_reflection" (H_R = 16 m) z_c leaves H_R from the fourth
# node and the node speed steps through U(16), U(17), U(18), ... inside the first 100 m.
_RECEPTORS = [[60.0, 30.0], [120.0, -20.0], [250.0, 40.0], [400.0, -60.0], [650.0, 100.0],
              [35.0, 95.0], [80.0, -70.0], [500.0, 10.0]]
_PARITY = {
    "neutral": (
        dict(NEUTRAL, sigma_v_min=0.0, sigma_w_min=0.0, tau_y=2.5, tau_z=1.2),
        [0.0004568223783393172, 8.017133563827156e-05, 3.873506876480515e-05,
         4.97053227584809e-06, 5.842591478196127e-06, 3.212749311004568e-06,
         3.7249258213419256e-05, 6.558724747541794e-06],
        [0.0, 0.0, 1.5918187362178943e-05, 5.211943576183716e-07, 4.267201831000672e-06,
         0.0, 0.0, 2.8165942940534293e-06],
        [2.9777306547869413, 17.06117621597564, 100.34927045096472, 0.39538069113568247,
         0.17624710131771776, 0.028611921399801305, 2.339371734389815, 2.339371734389815,
         2.963459739580403],
    ),
    "stable": (
        dict(STABLE, sigma_v_min=0.5, sigma_w_min=0.3, tau_y=1.5, tau_z=-0.5),
        [0.0008137488201996576, 0.00015566761822461686, 9.308003735607946e-05,
         1.2648837839572936e-05, 1.5705610594916397e-05, 1.9694827612160404e-06,
         5.993415736267853e-05, 1.791218060608127e-05],
        [0.0, 0.0, 2.0402961186663867e-05, 6.978320307217343e-07, 1.1779226691038332e-05,
         0.0, 0.0, 7.43666545666181e-06],
        [2.8588515811866597, 15.676231587494547, 94.60633969656867, 0.6349895283063827,
         0.220288993074184, 0.04135568252094032, 1.6574746787307684, 1.6574746787307684,
         1.6574746787307684],
    ),
}


_PARITY["stable_low_reflection"] = (
    dict(STABLE, h_canopy=16.0, sigma_v_min=0.5, sigma_w_min=0.3, tau_y=1.5, tau_z=-0.5),
    [0.0006766480155193977, 0.0001295945629366471, 6.336310262072519e-05,
     1.2537416334065058e-05, 1.235460524449125e-05, 0.00018694792397632983,
     9.396037607493923e-05, 1.4433084751952639e-05],
    [0.0, 0.0, 4.253552130071551e-05, 3.2583879607849776e-06, 9.996626128587544e-06, 0.0,
     0.0, 8.291610481472426e-06],
    [5.715223362801256, 30.1987107513937, 120.79745181845264, 0.5519206163791166,
     0.11088444919680776, 0.029736771759873097, 0.7868971805599453, 0.9297189562170501,
     1.6935186587362945],
)


@pytest.mark.parametrize("case", sorted(_PARITY))
def test_parity_with_the_independent_implementation(case):
    kwargs, street_values, junction_values, grid_values = _PARITY[case]
    t = plume_table(**kwargs)
    ours = [float(getattr(t, key)[0, i]) for key in ("sigma_y", "p_z", "u")
            for i in (1, 5, 30)]
    assert ours == pytest.approx(grid_values, rel=1e-12)
    rec = torch.tensor(_RECEPTORS, dtype=DT)
    src = source_points(_one_street(), theta_w=0.4)
    street = plume_kernel(rec, src.xy, t, theta_w=0.4, width=src.width,
                          clamp=src.clamp) @ src.weight
    assert street.tolist() == pytest.approx(street_values, rel=1e-12, abs=0)
    junction = plume_kernel(rec, torch.tensor([[40.0, 90.0]], dtype=DT), t, theta_w=0.4,
                            width=17.5, clamp=1 / 15.0)[:, 0]
    assert junction.tolist() == pytest.approx(junction_values, rel=1e-12, abs=0)


def test_batched_directions_match_single_hours():
    t = plume_table(**NEUTRAL)
    rec = torch.tensor(_RECEPTORS, dtype=DT)
    theta = torch.tensor([0.1, 0.4, 2.0], dtype=DT)
    src = source_points(_one_street(), theta_w=theta)
    k = plume_kernel(rec, src.xy, t, theta_w=theta, width=src.width, clamp=src.clamp)
    assert k.shape == (3, 8, src.xy.shape[0])
    for b in range(3):
        one = source_points(_one_street(), theta_w=float(theta[b]))
        kb = plume_kernel(rec, one.xy, t, theta_w=float(theta[b]), width=one.width,
                          clamp=one.clamp)
        assert torch.equal(k[b], kb)


def test_plume_kernel_refuses_above_cap(monkeypatch):
    t = plume_table(**NEUTRAL)
    monkeypatch.setattr(plume, "KERNEL_BYTES_CAP", 8 * 5)
    with pytest.raises(ValueError, match="use street_kernel"):
        plume_kernel(torch.zeros(3, 2, dtype=DT), torch.zeros(2, 2, dtype=DT), t,
                     theta_w=0.0)


# ------------------------------------------------------------------------ networks


def test_street_kernel_is_the_weighted_subsource_sum():
    net = from_test_network()
    t = plume_table(**NEUTRAL)
    theta = torch.tensor([0.4, -1.3], dtype=DT)
    with_self = street_kernel(net, t, theta_w=theta, self_contribution=True)
    src = source_points(net, theta_w=theta)
    point = plume_kernel(street_midpoints(net), src.xy, t, theta_w=theta, width=src.width,
                         clamp=src.clamp)
    manual = torch.zeros(2, 3, 3, dtype=DT)
    for j in range(3):
        mine = src.owner == j
        manual[..., j] = (point[..., mine] * src.weight[mine]).sum(-1)
    assert torch.allclose(with_self, manual, rtol=1e-14, atol=0)
    assert float(with_self.max()) > 0
    k = street_kernel(net, t, theta_w=theta)
    assert torch.equal(torch.diagonal(k, dim1=-2, dim2=-1), torch.zeros(2, 3, dtype=DT))
    off = ~torch.eye(3, dtype=torch.bool)
    assert torch.equal(k[:, off], with_self[:, off])
    with pytest.raises(TypeError, match="unexpected keyword"):
        street_kernel(net, t, theta_w=0.0, bogus=1)


def test_junction_kernel_is_the_point_kernel_at_the_junctions():
    net = from_test_network()
    t = plume_table(**STABLE)
    kj = junction_kernel(net, t, theta_w=0.9)
    xy, width, clamp = junction_sources(net)
    assert torch.equal(kj, plume_kernel(street_midpoints(net), xy, t, theta_w=0.9,
                                        width=width, clamp=clamp))
    assert kj.shape == (3, len(net.junctions))


def test_street_kernel_chunks_agree_with_one_block(monkeypatch):
    net = from_test_network()
    t = plume_table(**STABLE)
    whole = street_kernel(net, t, theta_w=0.9)
    monkeypatch.setattr(plume, "KERNEL_BYTES_CAP", 1)  # one receptor row per chunk
    assert rows_per_chunk(10_000, 1) == 1
    assert torch.equal(street_kernel(net, t, theta_w=0.9), whole)


# ------------------------------------------------------------------------ gradients


def _grid_net() -> StreetNetwork:
    """Four short streets, small enough for gradcheck."""
    x = {"a": 0.0, "b": 60.0, "c": 130.0, "d": 60.0, "e": 190.0}
    y = {"a": 0.0, "b": 10.0, "c": -5.0, "d": 90.0, "e": 40.0}
    streets = [Street("ab", "a", "b", 60.8, 20.0, 15.0), Street("bc", "b", "c", 70.2, 18.0,
                                                                 20.0),
               Street("bd", "b", "d", 80.0, 22.0, 18.0), Street("ce", "c", "e", 75.0, 20.0,
                                                                 16.0)]
    return StreetNetwork(streets, x, y)


def test_gradcheck_fluxes_friction_velocity_and_direction():
    """`torch.autograd.gradcheck` on `C_ext = K_s F_s + K_j F_j` in the fluxes, `u*` and the
    wind direction, without the crosswind cut-off (its jump is a stated non-smoothness).
    `u*` and the direction are chosen away from a `floor(z_c)` step and a 10 m grid
    node; there the table and the reads are smooth."""
    net = _grid_net()

    def c_ext(f_s, f_j, u_star, theta):
        t = plume_table(**{**NEUTRAL, "u_star": u_star})
        k_s = street_kernel(net, t, theta_w=theta, cutoff_sigma=math.inf)
        k_j = junction_kernel(net, t, theta_w=theta, cutoff_sigma=math.inf)
        return k_s @ f_s + k_j @ f_j

    inputs = (torch.tensor([1.0, 0.5, 2.0, 0.7], dtype=DT, requires_grad=True),
              torch.tensor([0.3, 0.2, 0.1, 0.4, 0.25], dtype=DT, requires_grad=True),
              torch.tensor(0.45, dtype=DT, requires_grad=True),
              torch.tensor(0.35, dtype=DT, requires_grad=True))
    assert torch.autograd.gradcheck(c_ext, inputs, eps=1e-7, atol=1e-10, rtol=1e-5)


def test_gradients_through_the_stable_table_match_finite_differences():
    """Every tensor input of the stable table against central differences: `u*`, `L`,
    `h`, `sigma_theta`, `theta*`, the temperature, `z0`, `d`, the reflection height, the
    plume-centre law and explicit time offsets (the reflection height 24.7 m keeps
    `max(floor(z_c), H_R)` off its kink)."""
    net = _grid_net()
    base = {"u_star": 0.2, "lmo": 80.0, "h_abl": 250.0, "sigma_theta": math.radians(8.0),
            "theta_star": 0.05, "temperature": 277.15, "z0": 1.0, "d": 13.0,
            "h_canopy": 24.7, "centre_c": 10.0, "centre_k": 0.675, "tau_y": 1.5,
            "tau_z": -0.5}

    def total(**vals):
        t = plume_table(**{**STABLE, **vals})
        return street_kernel(net, t, theta_w=0.35, cutoff_sigma=math.inf).sum()

    params = {k: torch.tensor(v, dtype=DT, requires_grad=True) for k, v in base.items()}
    grads = torch.autograd.grad(total(**params), list(params.values()))
    for (key, value), g in zip(base.items(), grads, strict=True):
        eps = 1e-6 * max(0.1, abs(value))
        fd = (float(total(**{**base, key: value + eps}))
              - float(total(**{**base, key: value - eps}))) / (2 * eps)
        assert bool(torch.isfinite(g))
        assert float(g) == pytest.approx(fd, rel=1e-5, abs=1e-12), key


@pytest.mark.parametrize("name", ["u_star", "d"])
def test_second_order_differentiation_of_the_table_raises_by_name(name):
    """The table's backward builds no graph over its own gradient, so a second-order
    request would silently lose that term (measured: 0.0 for u*, whose true second
    derivative is ~9e7); `create_graph=True` is refused by name instead (the convention of
    `noodl.solvers.implicit`)."""
    value = torch.tensor(NEUTRAL[name], dtype=DT, requires_grad=True)
    t = plume_table(**{**NEUTRAL, name: value})
    with pytest.raises(RuntimeError, match="plume_table: second-order"):
        torch.autograd.grad((t.u + t.p_z + t.sigma_y).sum(), value, create_graph=True)


def test_gradient_with_the_hard_cutoffs_is_finite():
    net = _grid_net()
    th = torch.tensor(0.35, dtype=DT, requires_grad=True)
    us = torch.tensor(0.45, dtype=DT, requires_grad=True)
    k = street_kernel(net, plume_table(**{**NEUTRAL, "u_star": us}), theta_w=th)
    g = torch.autograd.grad(k.sum(), (th, us))
    assert all(bool(torch.isfinite(v)) for v in g)


# ------------------------------------------------------------------------ Paris scale


def _paris_size_network(n_target: int = 577, block: float = 90.0) -> StreetNetwork:
    """A 577-street lattice with Haussmann-scale blocks, standing in for the Paris case."""
    side = 17
    x, y, streets = {}, {}, []
    for i in range(side + 1):
        for j in range(side + 1):
            x[f"{i},{j}"] = i * block + 7.0 * math.sin(j)
            y[f"{i},{j}"] = j * block + 7.0 * math.cos(i)
    for i in range(side + 1):
        for j in range(side + 1):
            for di, dj in ((1, 0), (0, 1)):
                a, b = f"{i},{j}", f"{i + di},{j + dj}"
                if b in x and len(streets) < n_target:
                    length = math.hypot(x[b] - x[a], y[b] - y[a])
                    streets.append(Street(f"s{len(streets)}", a, b, length, 15.0, 20.0))
    return StreetNetwork(streets=streets, x=x, y=y)


def test_paris_size_kernel_builds_in_chunks_under_the_cap():
    net = _paris_size_network()
    assert len(net.streets) == 577
    n_src = source_points(net, theta_w=0.6).xy.shape[0]
    rows = rows_per_chunk(n_src, 1)
    assert rows * n_src * 8 * plume._LIVE_TEMPORARIES <= KERNEL_BYTES_CAP
    assert rows < 577  # really chunked
    assert rows_per_chunk(n_src, 1, backward=True) < rows
    start = time.perf_counter()
    t = plume_table(**NEUTRAL)
    k = street_kernel(net, t, theta_w=0.6)
    kj = junction_kernel(net, t, theta_w=0.6)
    elapsed = time.perf_counter() - start
    print(f"\nParis-size kernels: 577 streets, {n_src} sub-sources, "
          f"{len(net.junctions)} junctions, {rows} rows/chunk, {elapsed:.3f} s")
    assert k.shape == (577, 577) and kj.shape == (577, len(net.junctions))
    assert bool(torch.isfinite(k).all()) and bool((k >= 0).all())
    assert int((k > 0).sum()) > 577
    assert SEUIL_GAUSS == 4.0
