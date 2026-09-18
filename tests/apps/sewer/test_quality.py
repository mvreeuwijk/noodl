"""Thermodynamics, kinetics and two-film transfer (rows H1, H2, S1)."""

import pytest
import torch

from tellegen.apps.sewer import quality as q

F64 = torch.float64


def test_h1_henry_constant_against_sander_2023():
    """Row H1. Measured: H(293.15) = 0.363854, H(298.15) = 0.403418."""
    assert float(q.henry_h2s(torch.tensor(293.15, dtype=F64))) == pytest.approx(
        0.36, abs=0.01
    )
    assert float(q.henry_h2s(torch.tensor(293.15, dtype=F64))) == pytest.approx(
        0.363854, abs=1e-6
    )
    assert float(q.henry_h2s(torch.tensor(298.15, dtype=F64))) == pytest.approx(
        0.403418, abs=1e-6
    )


def test_henry_increases_with_temperature():
    temps = torch.tensor([283.15, 293.15, 303.15], dtype=F64)
    values = q.henry_h2s(temps)
    assert bool((values[1:] > values[:-1]).all())


def test_free_fraction_is_one_half_at_the_pka():
    assert float(q.free_fraction(torch.tensor(7.0, dtype=F64))) == pytest.approx(0.5)
    assert float(q.free_fraction(torch.tensor(6.0, dtype=F64))) == pytest.approx(
        1.0 / 1.1, rel=1e-12
    )
    assert float(q.free_fraction(torch.tensor(8.0, dtype=F64))) == pytest.approx(
        1.0 / 11.0, rel=1e-12
    )


def test_kla_matches_the_published_form():
    slope = torch.tensor([0.01], dtype=F64)
    velocity = torch.tensor([1.3795], dtype=F64)
    depth = torch.tensor([0.1085], dtype=F64)
    froude2 = 1.3795**2 / (9.80665 * 0.1085)
    expected = 0.86 * (1.0 + 0.20 * froude2) * (0.01 * 1.3795) ** 0.375 / 0.1085 / 3600.0
    assert float(q.kla_h2s(slope, velocity, depth)) == pytest.approx(expected, rel=1e-14)
    # measured: 5.997451395e-04 1/s, i.e. 2.159083 1/h -- inside the 0.1-20 1/h plausibility
    # band of the research note's section 6.4
    assert 0.1 < float(q.kla_h2s(slope, velocity, depth)) * 3600.0 < 20.0


def test_kla_is_zero_and_finite_on_a_dry_pipe():
    zero = torch.zeros(1, dtype=F64, requires_grad=True)
    value = q.kla_h2s(torch.tensor([0.01], dtype=F64), zero, torch.zeros(1, dtype=F64))
    # ruling M4-R10: value tracks grad through `zero`, so float() gets .detach() first.
    assert float(value.detach()) == 0.0
    value.sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_s1_sulfide_rate_against_the_closed_form():
    """Row S1. Measured: exact agreement (relative difference 0.0)."""
    theta = 1.07 ** (18.0 - 20.0)
    expected = (
        0.32e-3 * 0.3 * theta / 0.0764
        - 0.64 * 2.0e-3 * theta * (0.01 * 1.3795) ** 0.375 / 0.1085
    ) / 3600.0
    value = q.sulfide_rate(
        torch.tensor([0.3], dtype=F64), torch.tensor([2.0e-3], dtype=F64),
        torch.tensor([18.0], dtype=F64), torch.tensor([0.0764], dtype=F64),
        torch.tensor([0.01], dtype=F64), torch.tensor([1.3795], dtype=F64),
        torch.tensor([0.1085], dtype=F64),
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_sulfide_rate_is_zero_on_a_dry_pipe():
    value = q.sulfide_rate(
        torch.tensor([0.3], dtype=F64), torch.tensor([2.0e-3], dtype=F64),
        torch.tensor([18.0], dtype=F64), torch.zeros(1, dtype=F64),
        torch.tensor([0.01], dtype=F64), torch.zeros(1, dtype=F64),
        torch.zeros(1, dtype=F64),
    )
    assert float(value) == 0.0


def test_bod_decay_is_first_order_with_the_temperature_factor():
    theta = 1.07 ** (18.0 - 20.0)
    expected = -0.2 * theta * 0.3 / 86400.0
    value = q.bod_rate(torch.tensor([0.3], dtype=F64), torch.tensor([18.0], dtype=F64))
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_h2_two_film_flux_against_the_closed_form():
    """Row H2. Measured: exact agreement (relative difference 0.0)."""
    sulfide = torch.tensor([2.0e-3], dtype=F64)
    gas = torch.tensor([5.0e-5], dtype=F64)
    volume = torch.tensor([7.25], dtype=F64)
    kla = q.kla_h2s(torch.tensor([0.01], dtype=F64), torch.tensor([1.3795], dtype=F64),
                    torch.tensor([0.1085], dtype=F64))
    free = q.free_fraction(torch.tensor([7.0], dtype=F64))
    henry = q.henry_h2s(torch.tensor([293.15], dtype=F64))
    flux = q.two_film_flux(sulfide, gas, volume, kla, free, henry)
    expected = float(kla) * 7.25 * (
        0.5 * 2.0e-3 - 5.0e-5 * (32.06 / 34.08) / float(henry)
    )
    assert float(flux) == pytest.approx(expected, rel=1e-12)
    assert float(flux) > 0.0


def test_the_flux_vanishes_at_henry_equilibrium():
    """Row H3's algebraic core: at C_G = H f C_S M_H2S / M_S there is no net transfer."""
    sulfide = torch.tensor([2.0e-3], dtype=F64)
    volume = torch.tensor([7.25], dtype=F64)
    kla = torch.tensor([1e-3], dtype=F64)
    free = q.free_fraction(torch.tensor([7.0], dtype=F64))
    henry = q.henry_h2s(torch.tensor([293.15], dtype=F64))
    gas = henry * free * sulfide * (q.M_H2S / q.M_S)
    assert float(q.two_film_flux(sulfide, gas, volume, kla, free, henry)) == (
        pytest.approx(0.0, abs=1e-18)
    )


def test_ppm_conversion_uses_the_ideal_gas_law():
    gas = torch.tensor([3.867789040e-4], dtype=F64)
    ppm = q.ppm_from_concentration(gas, torch.tensor([293.15], dtype=F64))
    assert float(ppm) == pytest.approx(272.990, abs=1e-3)


def test_sulfide_generation_reaction_advances_both_species():
    from tellegen.apps.sewer.quality import SulfideGeneration

    reaction = SulfideGeneration()
    x = torch.tensor([[0.3, 2.0e-3]], dtype=F64)
    drivers = {
        "sewer.R_h": torch.tensor([0.0764], dtype=F64),
        "sewer.q_slope": torch.tensor([0.01], dtype=F64),
        "sewer.v": torch.tensor([1.3795], dtype=F64),
        "sewer.d_m": torch.tensor([0.1085], dtype=F64),
        "T_water": torch.tensor([18.0], dtype=F64),
    }
    out = reaction.apply(x, 60.0, drivers)
    expected_s = float(x[0, 1]) + 60.0 * float(
        q.sulfide_rate(x[:, 0], x[:, 1], drivers["T_water"], drivers["sewer.R_h"],
                       drivers["sewer.q_slope"], drivers["sewer.v"], drivers["sewer.d_m"])
    )
    expected_b = float(x[0, 0]) + 60.0 * float(q.bod_rate(x[:, 0], drivers["T_water"]))
    assert float(out[0, 1]) == pytest.approx(expected_s, rel=1e-12)
    assert float(out[0, 0]) == pytest.approx(expected_b, rel=1e-12)


def test_sulfide_generation_refuses_a_missing_driver():
    from tellegen.apps.sewer.quality import SulfideGeneration

    with pytest.raises(KeyError, match="'sewer.R_h'"):
        SulfideGeneration().apply(torch.zeros(1, 2, dtype=F64), 60.0, {})


def test_c2_transfer_sources_are_exactly_opposite_in_moles_of_sulfur():
    """Row C2's algebraic core, node by node."""
    from tellegen.apps.sewer.quality import H2STransfer

    closure = H2STransfer(4, torch.tensor([0, 1, 2]))
    state = {
        "water_quality.x": torch.tensor([[0.3, 2.0e-3]] * 3, dtype=F64),
        "air_quality.x": torch.tensor([5.0e-5] * 3, dtype=F64),
    }
    drivers = {
        "sewer.V_wet": torch.tensor([7.25, 10.5, 5.0], dtype=F64),
        "sewer.q_slope": torch.tensor([0.01, 0.01, 0.01], dtype=F64),
        "sewer.v": torch.tensor([1.38, 1.53, 1.21], dtype=F64),
        "sewer.d_m": torch.tensor([0.1085, 0.14, 0.09], dtype=F64),
        "T_water": torch.tensor(18.0, dtype=F64),
        "T_head": torch.tensor(293.15, dtype=F64),
        "pH": torch.tensor(7.0, dtype=F64),
    }
    out = closure(state, drivers)
    water = out["water_quality.sources"][..., 1]
    air = out["air_quality.sources"]
    moles_out = -water / q.M_S
    moles_in = air / q.M_H2S
    assert torch.allclose(moles_out, moles_in, atol=1e-30, rtol=1e-14)


def test_out_pipe_gathers_per_pipe_drivers_into_manhole_order():
    """M4-R4 amendment: a non-identity out_pipe permutes the per-pipe drivers before use,
    while sewer.q_slope (already per manhole) is used as given."""
    from tellegen.apps.sewer.quality import H2STransfer, SulfideGeneration

    out_pipe = torch.tensor([0, 1, 3, 2, 4])
    per_pipe_v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=F64)
    per_pipe_r_h = torch.tensor([0.05, 0.06, 0.07, 0.08, 0.09], dtype=F64)
    per_pipe_d_m = torch.tensor([0.10, 0.11, 0.12, 0.13, 0.14], dtype=F64)
    per_pipe_v_wet = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=F64)
    q_slope = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01], dtype=F64)

    x = torch.tensor([[0.3, 2.0e-3]] * 5, dtype=F64)
    t_water = torch.full((5,), 18.0, dtype=F64)

    reaction = SulfideGeneration(out_pipe=out_pipe)
    drivers = {
        "sewer.R_h": per_pipe_r_h,
        "sewer.q_slope": q_slope,
        "sewer.v": per_pipe_v,
        "sewer.d_m": per_pipe_d_m,
        "T_water": t_water,
    }
    out = reaction.apply(x, 60.0, drivers)

    # Manhole 2's outgoing pipe is 3; manhole 3's outgoing pipe is 2 (out_pipe = [0,1,3,2,4]).
    expected_rate_m2 = q.sulfide_rate(
        x[2:3, 0], x[2:3, 1], t_water[2:3], per_pipe_r_h[3:4],
        q_slope[2:3], per_pipe_v[3:4], per_pipe_d_m[3:4],
    )
    expected_rate_m3 = q.sulfide_rate(
        x[3:4, 0], x[3:4, 1], t_water[3:4], per_pipe_r_h[2:3],
        q_slope[3:4], per_pipe_v[2:3], per_pipe_d_m[2:3],
    )
    assert float(out[2, 1]) == pytest.approx(
        float(x[2, 1]) + 60.0 * float(expected_rate_m2), rel=1e-12
    )
    assert float(out[3, 1]) == pytest.approx(
        float(x[3, 1]) + 60.0 * float(expected_rate_m3), rel=1e-12
    )

    closure = H2STransfer(5, torch.tensor([0, 1, 2, 3, 4]), out_pipe=out_pipe)
    state = {
        "water_quality.x": x,
        "air_quality.x": torch.tensor([5.0e-5] * 5, dtype=F64),
    }
    transfer_drivers = {
        "sewer.V_wet": per_pipe_v_wet,
        "sewer.q_slope": q_slope,
        "sewer.v": per_pipe_v,
        "sewer.d_m": per_pipe_d_m,
        "T_water": torch.tensor(18.0, dtype=F64),
        "T_head": torch.tensor(293.15, dtype=F64),
        "pH": torch.tensor(7.0, dtype=F64),
    }
    out_h2s = closure(state, transfer_drivers)

    kla_m2 = q.kla_h2s(q_slope[2:3], per_pipe_v[3:4], per_pipe_d_m[3:4])
    kla_m3 = q.kla_h2s(q_slope[3:4], per_pipe_v[2:3], per_pipe_d_m[2:3])
    free = q.free_fraction(transfer_drivers["pH"])
    henry = q.henry_h2s(transfer_drivers["T_head"])
    expected_flux_m2 = q.two_film_flux(
        x[2:3, 1], state["air_quality.x"][2:3], per_pipe_v_wet[3:4], kla_m2, free, henry
    )
    expected_flux_m3 = q.two_film_flux(
        x[3:4, 1], state["air_quality.x"][3:4], per_pipe_v_wet[2:3], kla_m3, free, henry
    )
    assert float(out_h2s["water_quality.sources"][2, 1]) == pytest.approx(
        float(-expected_flux_m2), rel=1e-12
    )
    assert float(out_h2s["water_quality.sources"][3, 1]) == pytest.approx(
        float(-expected_flux_m3), rel=1e-12
    )
