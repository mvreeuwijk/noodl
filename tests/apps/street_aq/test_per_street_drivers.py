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
from noodl.apps.street_aq.network import build_model, munich_idealised, street_index

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
    lmo[0] = -10.0                                    # street "1" alone touches junction "N1"
    j = per.closures[0].junction_values(dict(d, lmo=lmo))["lmo"]
    k = net.junctions.index("N1")
    assert float(j[k]) < 0


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


def test_bad_options_are_named():
    net, _ = munich_idealised()
    with pytest.raises(ValueError, match="meteo"):
        build_model(net, meteo="per_junction")
    with pytest.raises(ValueError, match="background"):
        build_model(net, background="gridded")
