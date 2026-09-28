"""The optional `u_star` driver, per-street meteorology and per-street background.

All three are opt-in. With uniform values they must reproduce the uniform model exactly
(to rounding); with genuinely per-street values they must do what MUNICH does: each street's
canyon wind and exchange from its own meteorology, each junction's routing from its own
direction and spread, and each street's inflow from the atmosphere at its own background.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.canyon import boundary_layer
from noodl.apps.street_aq.network import (
    build_model,
    from_test_network,
    munich_idealised,
    street_index,
)

DT = torch.float64
MUNICH = dict(canyon_wind="exponential", exchange="schulte", stability="munich",
              direction_averaging="munich", roof_wind_form="sirane")


def _uniform_drivers(model, *, u_ref=5.0, theta=0.3, h_abl=800.0, lmo=1e6, bg=2e-8):
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    for i, name in enumerate(street_index(model)):
        sources[net.node_index(name)] = 1e-6 * (i + 1)
    return {
        "street.x_boundary": torch.tensor([bg], dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(u_ref, dtype=DT),
        "theta_w": torch.tensor(theta, dtype=DT),
        "h_abl": torch.tensor(h_abl, dtype=DT),
        "lmo": torch.tensor(lmo, dtype=DT),
    }


def _q(model, drivers):
    return model.closures[0](None, drivers)["street.q"]


# ------------------------------------------------------------------------------ u_star

def test_u_star_driver_replaces_the_log_law():
    net, _ = munich_idealised()
    model, state, _ = build_model(net, **MUNICH)
    d = _uniform_drivers(model, u_ref=5.0)
    flows = model.closures[0]
    u_star = float(boundary_layer(flows.h_mean, torch.tensor(5.0, dtype=DT),
                                  torch.tensor(800.0, dtype=DT), z_ref=flows.z_ref,
                                  kappa=flows.kappa).u_star)
    with_u_star = dict(d, u_star=torch.tensor(u_star, dtype=DT))
    torch.testing.assert_close(_q(model, with_u_star), _q(model, d), rtol=1e-13, atol=0)
    # And it wins over U_ref for everything but sigma_theta: doubling u* at fixed U_ref
    # doubles the exchange flows exactly (sigma_theta is capped at 10 degrees either way).
    doubled = dict(d, u_star=torch.tensor(2.0 * u_star, dtype=DT))
    n_ex = len(net.streets)
    torch.testing.assert_close(_q(model, doubled)[-2 * n_ex:],
                               2.0 * _q(model, d)[-2 * n_ex:], rtol=1e-12, atol=0)


def test_u_star_without_u_ref_is_refused_only_where_u_ref_is_needed():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, **MUNICH)
    d = _uniform_drivers(model)
    d.pop("U_ref")
    d["u_star"] = torch.tensor(0.5, dtype=DT)
    with pytest.raises(KeyError, match="U_ref"):
        _q(model, d)          # MUNICH's sigma_theta = sigma_v / U needs a wind speed
    plain, _, _ = build_model(net, canyon_wind="exponential", exchange="schulte")
    _q(plain, d)              # no direction averaging: u* alone is enough


# -------------------------------------------------------------------- per-street meteo

def _expand(d, n_streets, n_junctions=None):
    out = dict(d)
    for key in ("U_ref", "theta_w", "h_abl", "lmo", "u_star"):
        if key in out:
            out[key] = out[key].expand(n_streets).clone()
    return out


def test_per_street_meteo_with_uniform_values_is_the_uniform_model():
    net, _ = munich_idealised()
    uniform, _, _ = build_model(net, **MUNICH)
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    d = _uniform_drivers(uniform)
    torch.testing.assert_close(_q(per, _expand(d, len(net.streets))), _q(uniform, d),
                               rtol=1e-13, atol=1e-18)


def test_per_street_meteo_batches():
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    d = _expand(_uniform_drivers(per), len(net.streets))
    batched = dict(d)
    for key in ("U_ref", "theta_w", "h_abl", "lmo"):
        batched[key] = torch.stack([d[key], d[key] * (1.1 if key != "theta_w" else 1.0)])
    q = _q(per, batched)
    assert q.shape[0] == 2
    torch.testing.assert_close(q[0], _q(per, d), rtol=1e-13, atol=1e-18)


def test_each_street_takes_its_own_wind_and_each_junction_its_own_direction():
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    uniform, _, _ = build_model(net, **MUNICH)
    n = len(net.streets)
    d = _expand(_uniform_drivers(per, u_ref=5.0), n)
    # Street k's canyon wind depends only on street k's own U_ref and direction.
    d2 = dict(d)
    d2["U_ref"] = d["U_ref"].clone()
    d2["U_ref"][3] = 8.0
    u_a = per.closures[0](None, d)["street.u_canyon"]
    u_b = per.closures[0](None, d2)["street.u_canyon"]
    changed = (u_a != u_b).nonzero().flatten().tolist()
    assert changed == [3]
    ref = uniform.closures[0](None, dict(_uniform_drivers(uniform, u_ref=8.0)))["street.u_canyon"]
    torch.testing.assert_close(u_b[3], ref[3], rtol=1e-13, atol=0)


def test_junction_drivers_override_the_street_mean():
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    n, n_j = len(net.streets), len(net.junctions)
    d = _expand(_uniform_drivers(per, theta=0.3), n)
    explicit = dict(d, theta_w_junction=torch.full((n_j,), 0.3, dtype=DT))
    torch.testing.assert_close(_q(per, explicit), _q(per, d), rtol=1e-13, atol=1e-18)
    # A junction direction differing from its streets' changes that junction's routing.
    turned = dict(explicit, theta_w_junction=explicit["theta_w_junction"].clone())
    turned["theta_w_junction"][net.junctions.index("A")] = 0.3 + math.pi / 2
    assert not torch.allclose(_q(per, turned), _q(per, d))


def test_default_junction_direction_is_a_circular_mean():
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    n = len(net.streets)
    d = _expand(_uniform_drivers(per, theta=0.0), n)
    # Streets alternating just either side of theta = 0 (i.e. of 2 pi): the circular mean
    # at every junction is ~0, where an arithmetic mean of angles in [0, 2 pi) is ~pi.
    wobble = torch.tensor([0.01 if i % 2 else 2 * math.pi - 0.01 for i in range(n)],
                          dtype=DT)
    d_w = dict(d, theta_w=wobble)
    flows = per.closures[0]
    theta_j = flows.junction_values(d_w)["theta_w"]
    assert float(torch.remainder(theta_j + math.pi, 2 * math.pi).sub(math.pi).abs().max()) < 0.02


def test_uniform_mode_never_reads_a_batch_as_a_street_axis():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, **MUNICH)
    n = len(net.streets)
    d = _uniform_drivers(model)
    batched = dict(d)
    for key in ("U_ref", "theta_w", "h_abl", "lmo"):
        batched[key] = d[key].expand(n).clone()          # a batch of n identical instances
    q = _q(model, batched)
    assert q.shape[0] == n
    torch.testing.assert_close(q[5], _q(model, d), rtol=1e-13, atol=1e-18)


def test_default_junction_lmo_keeps_the_unstable_sign():
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    n = len(net.streets)
    d = _expand(_uniform_drivers(per), n)
    lmo = torch.full((n,), 1e6, dtype=DT)
    lmo[0] = -10.0                                # street "1" (N1 -> A) is one of the four
                                                   # streets meeting at junction "A"
    j = per.closures[0].junction_values(dict(d, lmo=lmo))["lmo"]
    k = net.junctions.index("A")
    # The reciprocal (1/L) mean of {-10, 1e6, 1e6, 1e6} is ~ -40.0, dominated by the one
    # unstable street; an ARITHMETIC mean of the same four values would be +749997.5 --
    # firmly stable, and firmly the wrong sign. Asserting the value, not just its sign,
    # is what tells the two apart.
    torch.testing.assert_close(float(j[k]), -40.00120003600108, rtol=1e-9, atol=0)


# --------------------------------------------------------------- per-street background

def test_per_street_background_with_one_value_is_the_uniform_model():
    net, _ = munich_idealised()
    uniform, _, _ = build_model(net, **MUNICH)
    per, _, _ = build_model(net, background="per_street", **MUNICH)
    du = _uniform_drivers(uniform, bg=2e-8)
    dp = dict(du, **{"street.x_boundary": torch.full((len(net.streets),), 2e-8, dtype=DT),
                     "street.sources": torch.zeros(per.net.n, dtype=DT)})
    for name in street_index(per):
        dp["street.sources"][per.net.node_index(name)] = \
            du["street.sources"][uniform.net.node_index(name)]
    _, su, _ = build_model(net, **MUNICH)
    xu = uniform.steady(su, du)["street.x"]
    _, sp, _ = build_model(net, background="per_street", **MUNICH)
    xp = per.steady(sp, dp)["street.x"]
    torch.testing.assert_close(xp, xu, rtol=1e-12, atol=0)


def test_per_street_background_enters_only_through_that_streets_own_inflow():
    """MUNICH's rule (`ComputeInflowRateExtended`): the atmosphere's contribution to street
    j is street j's background times the flows from the atmosphere INTO street j. With no
    emission, a background on one street alone must reach the others only through that
    street's own outflow."""
    net, _ = munich_idealised()
    per, state, _ = build_model(net, background="per_street", **MUNICH)
    n = len(net.streets)
    d = _uniform_drivers(per, bg=0.0)
    d["street.sources"] = torch.zeros(per.net.n, dtype=DT)
    bg = torch.zeros(n, dtype=DT)
    k = street_index(per)["11"]
    bg[k] = 1e-8
    d["street.x_boundary"] = bg
    x = per.steady(state, d)["street.x"]
    # Street 11 holds at most its background; everything else holds only what 11 exports.
    assert 0 < float(x[k]) <= 1e-8 * (1 + 1e-12)
    assert float(x.sum()) > float(x[k])
    # Linearity in the per-street backgrounds: the sum of single-street responses is the
    # all-streets response.
    total = torch.zeros(n, dtype=DT)
    for j in range(n):
        bj = torch.zeros(n, dtype=DT)
        bj[j] = 1e-8
        total = total + per.steady(state, dict(d, **{"street.x_boundary": bj}))["street.x"]
    all_on = per.steady(state, dict(d, **{"street.x_boundary": torch.full((n,), 1e-8,
                                                                          dtype=DT)}))
    torch.testing.assert_close(total, all_on["street.x"], rtol=1e-10, atol=0)
    torch.testing.assert_close(all_on["street.x"], torch.full((n,), 1e-8, dtype=DT),
                               rtol=1e-10, atol=0)


def test_per_street_background_with_species_and_chemistry():
    from noodl.apps.street_aq.chemistry import photostationary_for_streets, street_steady
    net, _ = munich_idealised()
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    per, state, drivers = build_model(net, species=("no", "no2", "o3"), chemistry=reaction,
                                      background="per_street", **MUNICH)
    assert drivers["street.x_boundary"].shape == (len(net.streets), 3)
    d = dict(_uniform_drivers(per), J_NO2=torch.tensor(5e-3, dtype=DT),
             temperature=torch.tensor(293.15, dtype=DT))
    d["street.sources"] = torch.zeros(per.net.n, 3, dtype=DT)
    bg = torch.zeros(len(net.streets), 3, dtype=DT)
    bg[:, 2] = 8e-8
    d["street.x_boundary"] = bg
    out = street_steady(per, state, d, reaction=reaction, tol=1e-18, max_iter=200)
    torch.testing.assert_close(out["street.x"][:, 2], torch.full((len(net.streets),), 8e-8,
                               dtype=DT), rtol=1e-10, atol=0)


def test_bad_options_are_named():
    net, _ = munich_idealised()
    with pytest.raises(ValueError, match="meteo"):
        build_model(net, meteo="per_junction")
    with pytest.raises(ValueError, match="background"):
        build_model(net, background="gridded")


def test_junction_means_stay_finite_when_street_zero_is_degenerate():
    """Padded junction slots index street 0; a degenerate street-0 value (lmo exactly 0,
    or a pair of exactly opposing directions) must neither poison the junctions that do
    not touch street 0 nor give a NaN gradient."""
    net, _ = munich_idealised()
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    flows = per.closures[0]
    n = len(net.streets)
    d = _expand(_uniform_drivers(per), n)
    lmo = torch.full((n,), 1e6, dtype=DT)
    lmo[0] = 0.0
    lmo.requires_grad_(True)
    j = flows.junction_values(dict(d, lmo=lmo))["lmo"]
    assert torch.isfinite(j).all()
    untouched = [k for k in range(len(net.junctions))
                 if 0 not in flows.slot_street[k][flows.slot_active[k]].tolist()]
    assert untouched
    torch.testing.assert_close(j[untouched], torch.full((len(untouched),), 1e6, dtype=DT),
                               rtol=1e-12, atol=0)
    j.sum().backward()
    assert torch.isfinite(lmo.grad).all()
    # Streets whose reciprocals cancel exactly at a junction: 1/L sums to zero.
    lmo2 = torch.full((n,), 1e6, dtype=DT)
    k_a = net.junctions.index("A")
    a_streets = flows.slot_street[k_a][flows.slot_active[k_a]].tolist()
    assert len(a_streets) == 4                    # the four-way hub of munich_idealised
    lmo2[a_streets] = torch.tensor([50.0, -50.0, 200.0, -200.0], dtype=DT)
    lmo2.requires_grad_(True)
    j2 = flows.junction_values(dict(d, lmo=lmo2))["lmo"]
    assert torch.isfinite(j2).all() and float(j2[k_a].detach()) > 1e100     # neutral, not NaN
    j2.sum().backward()
    assert torch.isfinite(lmo2.grad).all()
    # Exactly opposing directions: sin and cos sums both vanish (atan2 at the origin).
    theta = torch.zeros(n, dtype=DT)
    theta[a_streets] = torch.tensor([0.0, math.pi, math.pi / 2, 3 * math.pi / 2], dtype=DT)
    theta.requires_grad_(True)
    t = flows.junction_values(dict(d, theta_w=theta))["theta_w"]
    assert torch.isfinite(t).all()
    t.sum().backward()
    assert torch.isfinite(theta.grad).all()


def test_safe_atan2_and_reciprocal_have_finite_gradients_at_their_singular_points():
    from noodl.apps.street_aq.routing import _safe_atan2, _safe_reciprocal
    s = torch.zeros(2, dtype=DT, requires_grad=True)
    c = torch.tensor([0.0, 2.0], dtype=DT, requires_grad=True)
    out = _safe_atan2(s, c)
    assert out.detach().tolist() == [0.0, 0.0]   # torch.atan2 itself gives a 0 gradient here;
                                                 # the guard keeps it so on every backend
    out.sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(c.grad).all()
    x = torch.tensor([0.0, -0.0, 4.0, -1e-320], dtype=DT, requires_grad=True)
    r = _safe_reciprocal(x)
    r_ = r.detach()
    assert torch.isfinite(r_).all() and float(r_[2]) == 0.25 and float(r_[3]) < 0
    r.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_gradients_flow_through_every_per_street_driver():
    """d(total street concentration)/d(each per-street driver), through `model.steady`:
    finite everywhere, and non-zero for every driver (a driver the forward pass silently
    dropped, or a masked NaN, would show here)."""
    net, _ = munich_idealised()
    n = len(net.streets)
    per, state, _ = build_model(net, meteo="per_street", background="per_street", **MUNICH)
    d = _expand(_uniform_drivers(per), n)
    k = torch.arange(n, dtype=DT)
    leaves = {
        "U_ref": 5.0 + 0.1 * k,
        "theta_w": 0.3 + 0.05 * k,
        "h_abl": 800.0 + 10.0 * k,
        "lmo": torch.where(k % 3 == 0, -50.0 - k, 200.0 + 10.0 * k),
        "u_star": 0.3 + 0.01 * k,
        "street.x_boundary": 2e-8 * (1.0 + 0.1 * k),
    }
    for value in leaves.values():
        value.requires_grad_(True)
    x = per.steady(state, dict(d, **leaves))["street.x"]
    grads = torch.autograd.grad(x.sum(), list(leaves.values()))
    for (key, _), g in zip(leaves.items(), grads, strict=True):
        assert torch.isfinite(g).all(), key
        assert float(g.abs().max()) > 0.0, key


def test_u_star_driver_under_per_street_meteo():
    net, _ = munich_idealised()
    n = len(net.streets)
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    uniform, _, _ = build_model(net, **MUNICH)
    d = _uniform_drivers(uniform)
    du = dict(d, u_star=torch.tensor(0.4, dtype=DT))
    dp = _expand(du, n)
    torch.testing.assert_close(_q(per, dp), _q(uniform, du), rtol=1e-13, atol=1e-18)
    # Each street's exchange follows its own u_star: doubling one street's changes only
    # that street's two exchange flows (the last 2 n entries: street k owns 2k, 2k + 1).
    doubled = dict(dp, u_star=dp["u_star"].clone())
    doubled["u_star"][4] = 0.8
    ex_a, ex_b = _q(per, dp)[-2 * n:], _q(per, doubled)[-2 * n:]
    changed = sorted({i // 2 for i in (ex_a != ex_b).nonzero().flatten().tolist()})
    assert changed == [4]
    # Where it is given, the junctions route with the mean of the streets' u_star.
    j = per.closures[0].junction_values(doubled)["u_star"]
    assert torch.isfinite(j).all() and float(j.max()) > 0.4


def test_per_street_and_junction_shape_errors_are_named():
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    per, _, _ = build_model(net, meteo="per_street", **MUNICH)
    d = _expand(_uniform_drivers(per), n)
    with pytest.raises(ValueError, match=rf"'U_ref' with a trailing street axis of length {n}"):
        _q(per, dict(d, U_ref=torch.full((n + 1,), 5.0, dtype=DT)))
    with pytest.raises(ValueError, match=r"'lmo' with a trailing street axis.*shape \(\)"):
        _q(per, dict(d, lmo=torch.tensor(1e6, dtype=DT)))
    with pytest.raises(ValueError, match=rf"'theta_w_junction' needs a trailing junction "
                                         rf"axis of length {n_j}"):
        _q(per, dict(d, theta_w_junction=torch.zeros(n_j - 1, dtype=DT)))


# ------------------------------------------------------------------------- sigma_theta

GAUSS = dict(canyon_wind="soulhac", exchange="sirane", stability="munich",
             direction_averaging="gauss", n_theta=8)


def test_a_sigma_theta_driver_equal_to_the_constructor_value_is_the_same_model():
    net, _ = munich_idealised()
    fixed, _, _ = build_model(net, sigma_theta=0.12, **GAUSS)
    free, _, _ = build_model(net, **GAUSS)
    d = _uniform_drivers(fixed)
    reference = _q(fixed, d)
    # Driven on a model built with the same constant, or with none at all: identical.
    driven = dict(d, sigma_theta=torch.tensor(0.12, dtype=DT))
    torch.testing.assert_close(_q(fixed, driven), reference, rtol=0, atol=0)
    torch.testing.assert_close(_q(free, driven), reference, rtol=0, atol=0)


def test_a_different_sigma_theta_driver_changes_the_routing_and_overrides_the_constructor():
    net, _ = munich_idealised()
    fixed, _, _ = build_model(net, sigma_theta=0.12, **GAUSS)
    wider, _, _ = build_model(net, sigma_theta=0.3, **GAUSS)
    d = _uniform_drivers(fixed)
    n_ex = 2 * len(net.streets)
    q_fixed = _q(fixed, d)
    q_driven = _q(fixed, dict(d, sigma_theta=torch.tensor(0.3, dtype=DT)))
    assert not torch.equal(q_driven[:-n_ex], q_fixed[:-n_ex])       # route and vent move
    torch.testing.assert_close(q_driven[-n_ex:], q_fixed[-n_ex:], rtol=0, atol=0)
    torch.testing.assert_close(q_driven, _q(wider, d), rtol=0, atol=0)


def test_a_batched_sigma_theta_driver_is_one_spread_per_instance():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, **GAUSS)
    d = _uniform_drivers(model)
    batched = {k: torch.stack([v, v]) for k, v in d.items() if k not in
               ("street.x_boundary", "street.sources")}
    batched["sigma_theta"] = torch.tensor([0.05, 0.25], dtype=DT)
    q = _q(model, batched)
    for i, sigma in enumerate((0.05, 0.25)):
        torch.testing.assert_close(
            q[i], _q(model, dict(d, sigma_theta=torch.tensor(sigma, dtype=DT))),
            rtol=1e-14, atol=0)


def test_a_per_junction_sigma_theta_driver_under_per_street_meteo():
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    uniform, _, _ = build_model(net, **GAUSS)
    per, _, _ = build_model(net, meteo="per_street", **GAUSS)
    d = _uniform_drivers(uniform)
    flat = _q(uniform, dict(d, sigma_theta=torch.tensor(0.2, dtype=DT)))
    d_per = _expand(d, n)
    torch.testing.assert_close(
        _q(per, dict(d_per, sigma_theta=torch.full((n_j,), 0.2, dtype=DT))), flat,
        rtol=1e-13, atol=1e-18)
    # One junction's own spread moves only the flows routed at that junction.
    spread = torch.full((n_j,), 0.2, dtype=DT)
    spread[0] = 0.02
    assert not torch.equal(_q(per, dict(d_per, sigma_theta=spread)), flat)
    with pytest.raises(ValueError, match=rf"'sigma_theta' with a trailing junction "
                                         rf"axis of length {n_j}"):
        _q(per, dict(d_per, sigma_theta=torch.tensor(0.2, dtype=DT)))


def test_a_sigma_theta_driver_is_refused_where_munich_computes_its_own():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, **MUNICH)
    d = _uniform_drivers(model)
    with pytest.raises(ValueError, match="sigma_theta.*direction_averaging='munich'"):
        _q(model, dict(d, sigma_theta=torch.tensor(0.1, dtype=DT)))


def test_uniform_sigma_theta_driver_shapes_are_named():
    net, _ = munich_idealised()
    n_j = len(net.junctions)
    model, _, _ = build_model(net, **GAUSS)
    d = _uniform_drivers(model)
    with pytest.raises(ValueError, match=r"meteo='uniform' needs 'sigma_theta' with one "
                                         r"value per instance, shape \(\).*meteo='per_street'"):
        _q(model, dict(d, sigma_theta=torch.full((n_j,), 0.1, dtype=DT)))
    batched = {k: torch.stack([v, v]) for k, v in d.items() if k not in
               ("street.x_boundary", "street.sources")}
    with pytest.raises(ValueError, match=r"shape \(2,\).*got shape \(3,\)"):
        _q(model, dict(batched, sigma_theta=torch.full((3,), 0.1, dtype=DT)))
    # A scalar spread applies to every instance of a batch.
    torch.testing.assert_close(
        _q(model, dict(batched, sigma_theta=torch.tensor(0.1, dtype=DT))),
        _q(model, dict(batched, sigma_theta=torch.tensor([0.1, 0.1], dtype=DT))),
        rtol=0, atol=0)


# --------------------------------------------------------------- z_ref inside the canopy

def test_z_ref_inside_the_canopy_runs_with_u_star_and_is_refused_with_u_ref_only():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, z_ref=1.0, **GAUSS)      # far below d + z0
    reference, _, _ = build_model(net, **GAUSS)             # z_ref = 30 m, same u*
    d = dict(_uniform_drivers(model), u_star=torch.tensor(0.4, dtype=DT))
    torch.testing.assert_close(_q(model, d), _q(reference, d), rtol=0, atol=0)
    with pytest.raises(ValueError, match="z_ref must clear the displacement height"):
        _q(model, _uniform_drivers(model))


# ------------------------------------------------- direction_averaging="sirane" (exact)

SIRANE = dict(canyon_wind="soulhac", exchange="sirane", stability="munich",
              routing="sirane")


def _ndtr(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _routed_at(per_model, d, phi, n_streets):
    """`q` of a `direction_averaging="none"`, `meteo="per_street"` model whose streets
    all see the mean direction of `d` but whose junctions all route at `phi` (a tensor of
    shape `(N,)` for a batch of `N` junction directions): the routing at one fixed sample
    direction with the canyon velocities left at the mean, which is exactly what a
    direction average holds fixed."""
    phi = torch.as_tensor(phi, dtype=DT)
    n_j = per_model.closures[0].n_junctions
    batch = tuple(phi.shape)
    out = {k: v for k, v in d.items()
           if k not in ("street.x_boundary", "street.sources", "sigma_theta")}
    for key in ("U_ref", "theta_w", "h_abl", "lmo", "u_star"):
        if key in out:
            out[key] = out[key].expand(*batch, n_streets).clone()
    out["theta_w_junction"] = phi.unsqueeze(-1).expand(*batch, n_j).clone()
    return _q(per_model, out)


def _r1_switch_angle(sn):
    """The direction at which street `r1` of `from_test_network` switches between inflow
    and outflow at BOTH its ends: its azimuth at `n1` minus a right angle (about 71.57 deg).
    No other street of the network switches within 0.64 rad of it."""
    return math.atan2(sn.y["n0"] - sn.y["n1"], sn.x["n0"] - sn.x["n1"]) - math.pi / 2.0


def test_sirane_average_is_exact_across_a_single_switch():
    sn = from_test_network()
    n = len(sn.streets)
    sigma, s = 0.05, _r1_switch_angle(sn)
    phi0 = s + 0.03
    exact, _, _ = build_model(sn, direction_averaging="sirane", **SIRANE)
    fixed, _, _ = build_model(sn, direction_averaging="none", meteo="per_street", **SIRANE)
    d = dict(_uniform_drivers(exact, theta=phi0), sigma_theta=torch.tensor(sigma, dtype=DT))
    below = _routed_at(fixed, d, s - 0.2, n)
    above = _routed_at(fixed, d, s + 0.2, n)
    assert not torch.allclose(below, above)            # the switch does move the routing
    w = _ndtr((s - phi0) / sigma)
    expected = w * below + (1.0 - w) * above
    torch.testing.assert_close(_q(exact, d), expected, rtol=1e-12, atol=1e-12)


def test_sirane_average_tends_to_no_average_as_the_spread_vanishes():
    net, _ = munich_idealised()
    exact, _, _ = build_model(net, direction_averaging="sirane", **SIRANE)
    none, _, _ = build_model(net, direction_averaging="none", **SIRANE)
    for theta in (0.3, 1.2, 4.0):                      # none of them on a switch angle
        d = _uniform_drivers(exact, theta=theta)
        torch.testing.assert_close(
            _q(exact, dict(d, sigma_theta=torch.tensor(1e-6, dtype=DT))), _q(none, d),
            rtol=1e-12, atol=1e-12)
        # With no spread at all (neither driver nor constructor value), also "none".
        torch.testing.assert_close(_q(exact, d), _q(none, d), rtol=1e-12, atol=1e-12)


def test_sirane_average_agrees_with_a_brute_force_gaussian_average():
    net, _ = munich_idealised()
    n = len(net.streets)
    exact, _, _ = build_model(net, direction_averaging="sirane", **SIRANE)
    fixed, _, _ = build_model(net, direction_averaging="none", meteo="per_street", **SIRANE)
    # A rectangle rule on a piecewise-constant integrand errs by up to one cell's Gaussian
    # mass per switch, O(1/N): the gap is bounded by one cell's mass per switch (measured
    # up to 8.6e-6 of the flow scale at 20 001 points, 3.3e-7 at 200 001). That gap is the
    # rectangle rule's, which the last check shows by its convergence; the exact average
    # has no such error.

    def gap(theta, sigma, n_points):
        d = dict(_uniform_drivers(exact, theta=theta),
                 sigma_theta=torch.tensor(sigma, dtype=DT))
        x = torch.linspace(-8.0 * sigma, 8.0 * sigma, n_points, dtype=DT)
        pdf = torch.exp(-0.5 * (x / sigma) ** 2)
        brute = ((pdf / pdf.sum()).unsqueeze(-1) * _routed_at(fixed, d, theta + x, n)).sum(0)
        q = _q(exact, d)
        return float((q - brute).abs().max()) / float(q.abs().max())

    for theta, sigma in ((0.3, 0.1), (1.5, 0.05), (3.3, 0.17), (0.02, 0.12)):
        assert gap(theta, sigma, 20001) <= 1e-5
    assert gap(0.3, 0.1, 2001) > 5.0 * gap(0.3, 0.1, 20001)


def test_sirane_average_is_differentiable_in_the_spread_and_the_mean_direction():
    sn = from_test_network()
    s = _r1_switch_angle(sn)
    model, _, _ = build_model(sn, direction_averaging="sirane", **SIRANE)
    base = _uniform_drivers(model)
    closure = model.closures[0]

    def q_of(sigma, theta):
        return closure(None, dict(base, sigma_theta=sigma, theta_w=theta))["street.q"]

    sigma = torch.tensor(0.05, dtype=DT, requires_grad=True)
    theta = torch.tensor(s + 0.03, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(q_of, (sigma, theta), eps=1e-7, atol=1e-6,
                                    rtol=1e-5)
    jac_sigma, jac_theta = torch.autograd.functional.jacobian(q_of, (sigma, theta))
    assert bool(torch.isfinite(jac_sigma).all()) and bool(torch.isfinite(jac_theta).all())
    assert float(jac_sigma.abs().max()) > 0.0          # the switch is inside the window
    # The routed flows move with the mean direction through the weights too, not only
    # through the canyon velocities: "none" at the same direction has no weights, and its
    # direction derivative differs.
    none, _, _ = build_model(sn, direction_averaging="none", **SIRANE)
    jac_none = torch.autograd.functional.jacobian(
        lambda t: none.closures[0](None, dict(base, theta_w=t))["street.q"], theta)
    assert not torch.allclose(jac_theta, jac_none)


def test_sirane_average_under_per_street_meteo_and_in_a_batch():
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    uniform, _, _ = build_model(net, direction_averaging="sirane", **SIRANE)
    per, _, _ = build_model(net, direction_averaging="sirane", meteo="per_street",
                            **SIRANE)
    d = _uniform_drivers(uniform)
    flat = _q(uniform, dict(d, sigma_theta=torch.tensor(0.15, dtype=DT)))
    torch.testing.assert_close(
        _q(per, dict(_expand(d, n), sigma_theta=torch.full((n_j,), 0.15, dtype=DT))),
        flat, rtol=1e-13, atol=1e-18)
    batched = {k: torch.stack([v, v]) for k, v in d.items() if k not in
               ("street.x_boundary", "street.sources")}
    batched["sigma_theta"] = torch.tensor([0.0, 0.15], dtype=DT)
    q = _q(uniform, batched)
    torch.testing.assert_close(q[1], flat, rtol=1e-14, atol=0)
    torch.testing.assert_close(q[0], _q(uniform, d), rtol=1e-14, atol=0)


def test_sirane_average_with_a_different_spread_at_each_junction():
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    uniform, _, _ = build_model(net, direction_averaging="sirane", **SIRANE)
    per, _, _ = build_model(net, direction_averaging="sirane", meteo="per_street",
                            **SIRANE)
    closure = per.closures[0]
    d = _uniform_drivers(uniform, theta=0.3)
    spreads = (0.05, 0.2)
    sigma = torch.tensor([spreads[j % 2] for j in range(n_j)], dtype=DT)
    q = _q(per, dict(_expand(d, n), sigma_theta=sigma))
    n_route, n_vent = len(closure.route_flat), len(closure.vent_flat)
    d2 = closure.d_max ** 2
    junction_of = torch.cat([closure.route_flat // d2, closure.vent_flat // closure.d_max])
    for k, spread in enumerate(spreads):
        reference = _q(uniform, dict(d, sigma_theta=torch.tensor(spread, dtype=DT)))
        mine = (junction_of % 2) == k                   # flows routed at those junctions
        torch.testing.assert_close(q[: n_route + n_vent][mine],
                                   reference[: n_route + n_vent][mine],
                                   rtol=1e-13, atol=1e-18)
    # The exchange flows do not depend on the routing at all.
    torch.testing.assert_close(q[n_route + n_vent:], reference[n_route + n_vent:],
                               rtol=1e-13, atol=0)


def test_sirane_average_refuses_a_spread_of_a_quarter_turn_or_more():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, direction_averaging="sirane", **SIRANE)
    d = _uniform_drivers(model)
    for bad in (math.pi / 4.0, 1.0, -0.01, float("nan")):
        with pytest.raises(ValueError, match=r"direction_averaging='sirane'.*sigma_theta"):
            _q(model, dict(d, sigma_theta=torch.tensor(bad, dtype=DT)))
    with pytest.raises(ValueError, match=r"direction_averaging='sirane'.*sigma_theta"):
        build_model(net, direction_averaging="sirane", sigma_theta=1.0, **SIRANE)
