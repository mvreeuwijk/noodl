"""The two presets, the fallbacks of the default closure set, the turbulence floors and the
deprecated spellings, on whole models."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.apps.street_aq.canyon import SIRANE_EXCHANGE
from noodl.apps.street_aq.case import StreetCase, drivers_at
from noodl.apps.street_aq.network import build_model, from_test_network, munich_idealised
from noodl.apps.street_aq.routing import StreetFlows

DT = torch.float64


def _flows(model) -> StreetFlows:
    return next(c for c in model.closures if isinstance(c, StreetFlows))


def _q(model, drivers):
    return _flows(model)({}, drivers)["street.q"]


def _drivers(model, *, u_ref=6.0, theta=0.3, h_abl=800.0, **extra):
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    sources[net.node_index(net.nodes[-1])] = 1.0e-3
    return dict({
        "street.x_boundary": torch.zeros(1, dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(u_ref, dtype=DT),
        "theta_w": torch.tensor(theta, dtype=DT),
        "h_abl": torch.tensor(h_abl, dtype=DT),
    }, **extra)


def test_the_default_is_the_neutral_single_direction_model_without_lmo_or_spread():
    """With no `lmo` and no `sigma_theta` the sirane preset's Monin-Obukhov turbulence is
    the neutral form and its exact Gaussian average is the single mean direction, bit for
    bit; where the turbulence clears SIRANE's floors, nothing else differs."""
    for net in (from_test_network(), munich_idealised()[0]):
        default, state, _ = build_model(net)
        plain, _, _ = build_model(net, direction_averaging="none", stability="neutral",
                                  sigma_w_min=0.0, sigma_v_min=0.0)
        d = _drivers(default)
        flows = _flows(default)
        _bl, _u, u_d = flows.velocities(d)
        assert bool((u_d > 0.30 * SIRANE_EXCHANGE).all())      # the floor does not bind
        assert torch.equal(_q(default, d), _q(plain, d))
        assert torch.equal(default.steady(state, d)["street.x"],
                           plain.steady(state, d)["street.x"])


def test_the_legacy_spellings_build_the_same_model():
    net = from_test_network()
    new, _, _ = build_model(net, canyon_wind="bessel_profile",
                            roof_exchange="turbulent_velocity",
                            junction_routing="non_crossing_streamlines",
                            direction_averaging="none", stability="neutral")
    with pytest.warns(DeprecationWarning) as record:
        old, _, _ = build_model(net, canyon_wind="soulhac", exchange="sirane",
                                routing="sirane", direction_averaging="none",
                                stability="neutral")
    messages = {str(w.message) for w in record}
    assert "build_model: the keyword 'exchange' is deprecated; use 'roof_exchange'" in messages
    assert ("build_model: canyon_wind='soulhac' is deprecated; use "
            "canyon_wind='bessel_profile'") in messages
    assert all(w.filename == __file__ for w in record)
    d = _drivers(new)
    assert torch.equal(_q(old, d), _q(new, d))
    assert _flows(old).options == _flows(new).options


def test_the_munich_preset_is_the_earlier_munich_option_set():
    """`preset="munich"` against the same closures spelt out one by one (the earlier
    strings), with a stable and an unstable Obukhov length: bit for bit."""
    net, _ = munich_idealised()
    preset, _, _ = build_model(net, preset="munich")
    with pytest.warns(DeprecationWarning):
        spelt, _, _ = build_model(
            net, canyon_wind="exponential", exchange="schulte", routing="sirane",
            direction_averaging="munich", roof_wind_form="sirane", stability="munich",
            kappa=0.41, canyon_wind_min=0.1, u_d_min=0.001, sigma_w_min=0.0,
            sigma_v_min=0.0, sigma_w_height="street_height",
            shape_constant="grid_search",
        )
    assert _flows(preset).options == _flows(spelt).options
    for lmo in (150.0, -40.0):
        d = _drivers(preset, lmo=torch.tensor(lmo, dtype=DT))
        assert torch.equal(_q(preset, d), _q(spelt, d))


def test_the_sigma_w_floor_is_the_exchange_velocity_floor_it_replaces_bit_for_bit():
    """`sigma_w_min` floors `sigma_w` before `u_d = sigma_w / (sqrt(2) pi)`; since that is
    linear, it equals the exchange-velocity floor `u_d_min = sigma_w_min / (sqrt(2) pi)`
    exactly, including where it binds (a light wind)."""
    net = from_test_network()
    floored, _, _ = build_model(net, direction_averaging="none", sigma_w_min=0.30)
    emulated, _, _ = build_model(net, direction_averaging="none", sigma_w_min=0.0,
                                 u_d_min=0.30 * SIRANE_EXCHANGE)
    unfloored, _, _ = build_model(net, direction_averaging="none", sigma_w_min=0.0)
    for u_ref in (0.3, 1.0, 6.0):
        d = _drivers(floored, u_ref=u_ref, lmo=torch.tensor(80.0, dtype=DT))
        assert torch.equal(_q(floored, d), _q(emulated, d))
    light = _drivers(floored, u_ref=0.3)
    assert not torch.equal(_q(floored, light), _q(unfloored, light))


def test_the_sigma_v_floor_widens_the_turbulence_intensity_spread():
    """`sigma_v_min` floors the `sigma_v` of `sigma_theta = min(sigma_v / U_ref, 10 deg)`.
    Under a weak friction velocity (u* = 0.1 m/s, so sigma_v = 0.12 m/s) and an 8 m/s wind
    the spread is 0.86 degrees (one sample); a 0.5 m/s floor makes it 3.6 degrees (three
    rectangle-rule samples) and a 5 m/s floor reaches the 10-degree cap (ten)."""
    net, _ = munich_idealised()
    counts = []
    for floor in (0.0, 0.5, 5.0):
        model, _, _ = build_model(net, preset="munich", sigma_v_min=floor)
        d = _drivers(model, u_ref=8.0, u_star=torch.tensor(0.1, dtype=DT))
        _theta, _offsets, weights = _flows(model)._samples(d)
        counts.append(weights.shape[-1])
    assert counts == [1, 3, 10]


def test_a_driven_spread_is_refused_by_the_turbulence_intensity_spread():
    net, _ = munich_idealised()
    model, _, _ = build_model(net, preset="munich")
    with pytest.raises(ValueError, match=r"'sigma_theta' is not used with "
                                         r"direction_spread='turbulence_intensity'"):
        _q(model, _drivers(model, sigma_theta=torch.tensor(0.1, dtype=DT)))


def test_the_spread_driver_falls_back_to_the_constructor_value_then_zero():
    net, _ = munich_idealised()
    fixed, _, _ = build_model(net, sigma_theta=0.1, sigma_w_min=0.0)
    free, _, _ = build_model(net, sigma_w_min=0.0)
    none, _, _ = build_model(net, direction_averaging="none", sigma_w_min=0.0)
    d = _drivers(free)
    assert torch.equal(_q(fixed, d), _q(free, dict(d, sigma_theta=torch.tensor(0.1, dtype=DT))))
    assert torch.equal(_q(free, d), _q(none, d))
    assert not torch.equal(_q(fixed, d), _q(none, d))


def _case(net, **meteo):
    return StreetCase.synthetic(
        net, species=["NO2"], times=[0.0, 3600.0],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0, h_abl=800.0, u_star=0.4,
                   lmo=1e5, **meteo),
        emissions=1e-4, background=0.0,
    )


def test_drivers_at_passes_the_direction_spread_through():
    net, _ = munich_idealised()
    n_s, n_j = len(net.streets), len(net.junctions)
    spread = np.full((2, n_s), math.radians(8.0))
    spread[1] = math.radians(4.0)
    case = _case(net, sigma_theta=spread)
    uniform, _, _ = build_model(net, species=("NO2",))
    d = drivers_at(case, uniform, 1)
    assert float(d["sigma_theta"]) == pytest.approx(math.radians(4.0), rel=1e-15)
    per, _, _ = build_model(net, species=("NO2",), meteo="per_street")
    d = drivers_at(case, per, 0)
    assert d["sigma_theta"].shape == (n_j,)
    torch.testing.assert_close(d["sigma_theta"],
                               torch.full((n_j,), math.radians(8.0), dtype=DT),
                               rtol=1e-15, atol=0)
    junction = np.tile(np.linspace(0.05, 0.1, n_j), (2, 1))
    case_j = StreetCase.synthetic(
        net, species=["NO2"], times=[0.0, 3600.0],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0, h_abl=800.0, u_star=0.4,
                   sigma_theta=spread),
        meteo_junction=dict(sigma_theta=junction), emissions=1e-4, background=0.0,
    )
    torch.testing.assert_close(drivers_at(case_j, per, 1)["sigma_theta"],
                               torch.as_tensor(junction[1], dtype=DT), rtol=0, atol=0)
    # The junction spread alone is enough.
    only_j = StreetCase.synthetic(
        net, species=["NO2"], times=[0.0, 3600.0],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0, h_abl=800.0, u_star=0.4),
        meteo_junction=dict(sigma_theta=junction), emissions=1e-4, background=0.0,
    )
    torch.testing.assert_close(drivers_at(only_j, per, 0)["sigma_theta"],
                               torch.as_tensor(junction[0], dtype=DT), rtol=0, atol=0)
    assert float(drivers_at(only_j, uniform, 0)["sigma_theta"]) == pytest.approx(
        float(np.mean(junction[0])), rel=1e-15)
    # A model computing its own spread is not handed one.
    munich, _, _ = build_model(net, species=("NO2",), preset="munich")
    assert "sigma_theta" not in drivers_at(case, munich, 0)
    _q(munich, drivers_at(case, munich, 0))


def _u_d(model, *, u_star, h_abl, lmo=None):
    d = {"u_star": torch.tensor(u_star, dtype=DT), "theta_w": torch.tensor(0.3, dtype=DT),
         "h_abl": torch.tensor(h_abl, dtype=DT)}
    if lmo is not None:
        d["lmo"] = torch.tensor(lmo, dtype=DT)
    return _flows(model).velocities(d)[2]


def test_the_sirane_preset_evaluates_the_exchange_sigma_w_at_the_canopy_height():
    """SIRANE's `u_d = sigma_w / (sqrt(2) pi)` depends on the external flow only: `sigma_w`
    is evaluated at the canopy height `h_canopy` (SIRANE's `H_R`, 20 m by default) for
    every street, whatever the street's own height. `sigma_w_height="street_height"`
    restores the per-street height, which is the `munich` preset's choice."""
    net = from_test_network()
    heights = torch.tensor([s.height for s in net.streets], dtype=DT)
    assert heights.unique().numel() > 1
    u_star, h_abl = 0.9, 400.0

    def closed_form(z):
        return 1.3 * u_star * (1.0 - 0.8 * z / h_abl) * SIRANE_EXCHANGE

    default, _, _ = build_model(net, sigma_w_min=0.0)
    assert _flows(default).sigma_w_height == "canopy_height"
    torch.testing.assert_close(_u_d(default, u_star=u_star, h_abl=h_abl),
                               torch.full_like(heights, closed_form(20.0)),
                               rtol=1e-14, atol=0.0)
    taller, _, _ = build_model(net, sigma_w_min=0.0, h_canopy=30.0)
    torch.testing.assert_close(_u_d(taller, u_star=u_star, h_abl=h_abl),
                               torch.full_like(heights, closed_form(30.0)),
                               rtol=1e-14, atol=0.0)
    per_street, _, _ = build_model(net, sigma_w_min=0.0, sigma_w_height="street_height")
    torch.testing.assert_close(_u_d(per_street, u_star=u_star, h_abl=h_abl),
                               closed_form(heights), rtol=1e-14, atol=0.0)
    munich, _, _ = build_model(net, preset="munich")
    assert _flows(munich).sigma_w_height == "street_height"
    with pytest.raises(ValueError, match="h_canopy must be finite and > 0"):
        build_model(net, h_canopy=0.0)
    with pytest.raises(ValueError, match="sigma_w_height must be one of"):
        build_model(net, sigma_w_height="roof")


@pytest.mark.parametrize(("h_canopy", "sirane_sigma_w"),
                         [(10.0, 1.92331), (20.0, 1.91383), (30.0, 1.90425)])
def test_the_canopy_height_sigma_w_matches_sirane_hand_entered_values(h_canopy,
                                                                      sirane_sigma_w):
    """Hand-entered from three SIRANE v2.1 rev 128 runs of its South Kensington deck, hour
    01 (wind 9 m/s from 315 degrees, turbulence floors zeroed), which differ only in the
    canopy height `H_R` (10, 20, 30 m). SIRANE's preprocessor gave u* = 1.487 m/s,
    h = 1615.5 m and L = 2152.72 m (the neutral branch, L > h). Each value is the median
    over the 46 streets of `sqrt(2) pi u_d`, with `u_d` backed out of SIRANE's printed roof
    flux, `F / (W L (C_int - C_ext))`; the 46 values spread by only 0.1 % although the
    street heights do not. The tolerance is set by SIRANE's printed digits: u* to 3
    decimals (3.4e-4 relative) and the flux and concentrations to 4 significant figures.
    Moving `H_R` by 10 m moves `sigma_w` by 0.5 %, so the height is pinned to about 1 m."""
    net = from_test_network()
    model, _, _ = build_model(net, sigma_w_min=0.0, h_canopy=h_canopy)
    u_d = _u_d(model, u_star=1.487, h_abl=1615.5, lmo=2152.72)
    sigma_w = u_d / SIRANE_EXCHANGE
    assert torch.allclose(sigma_w, torch.full_like(sigma_w, sirane_sigma_w), rtol=4e-4,
                          atol=0.0)


def test_the_canopy_height_sigma_w_ratio_matches_sirane_without_u_star():
    """The same SIRANE runs at `H_R` = 10 and 30 m: the ratio of the two `sigma_w`
    (1.92331 / 1.90425) needs no u*, and the neutral closed form gives it to 5e-5."""
    h = 1615.5
    assert abs((1 - 0.8 * 10 / h) / (1 - 0.8 * 30 / h) - 1.92331 / 1.90425) < 1e-4
