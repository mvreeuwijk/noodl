"""MUNICH as the independent reference — spec section 7, rows 10 and 11.

Part one: the thirteen exact input/output pairs of `.superpowers/munich-formulas.md`
section 8, each with its equation, page and `file:line`. Part two: the published 12-street
idealised case of Kim et al. 2022 Fig. 1, whose INPUTS WERE NEVER PUBLISHED, so what is
checked is every scale-invariant property of it and nothing else.

Tolerances. Where the research record carries a number at full double precision the check
is at 1e-12; where it quotes six or seven significant figures the check is at the precision
those figures support, and the comment says so. Nothing here is loosened to make a test
pass: a miss is a transcription error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.street_aq.canyon import (
    KAPPA_MUNICH,
    SCHULTE_BETA,
    SIRANE_EXCHANGE,
    BoundaryLayer,
    canyon_velocity,
    exchange_velocity,
    macdonald_profile,
    roof_wind,
    soulhac_shape,
)
from noodl.apps.street_aq.chemistry import photostationary_for_streets, street_steady
from noodl.apps.street_aq.network import build_model, munich_idealised, street_index
from noodl.apps.street_aq.routing import (
    direction_offsets,
    n_theta_munich,
    node_closure,
    routing_matrix,
    sigma_theta_munich,
)
from noodl.layers.reaction import K_NO_O3, K_NO_O3_298, MOLAR_MASS

DT = torch.float64
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "street" / "munich_idealised.json"
# MUNICH's own default Paris geometry (K18 p. 617), which every pinned pair below uses.
H, W, Z0_S, U_STAR, PBLH = 6.9, 7.5, 0.01, 0.3, 500.0
SIGMA_W = 0.38569440


def _t(value):
    return torch.tensor([value], dtype=DT)


def test_t1_sirane_exchange_velocity(record_property):
    """S11 Eq. (5) p. 7386; K18 Eq. (3) p. 613; K22 Eq. (B10) p. 7387;
    `StreetNetworkTransport.cxx:3273`. The regression guard against `1/sqrt(2 pi)`."""
    assert abs(SIRANE_EXCHANGE - 0.225079079039277) < 1e-15
    u_d = exchange_velocity(_t(SIGMA_W), _t(H), _t(W), form="sirane")
    # The record quotes 0.08681174 to eight significant figures.
    assert abs(float(u_d) / 0.08681174 - 1.0) < 1e-7
    assert abs(1.0 / math.sqrt(2.0 * math.pi) - 0.398942280401433) < 1e-15
    record_property("check", "Exchange velocity u_d")
    record_property("tolerance", "rel 1e-7")
    record_property("measured_rel", abs(float(u_d) / 0.08681174 - 1.0))
    record_property("check2", "SIRANE_EXCHANGE constant")
    record_property("tolerance2", "< 1e-15")
    record_property("measured2_diff", abs(SIRANE_EXCHANGE - 0.225079079039277))


def test_t2_schulte_mixing_length_and_exchange_velocity():
    """K18 Eqs. (4)-(8) p. 613, K22 Eq. (B11); ATM `MeteorologyStreet.cxx:547-554`."""
    assert abs(SCHULTE_BETA - 0.450158158078553) < 1e-15
    lm = SCHULTE_BETA / (1.0 + H / W)
    assert abs(lm - 0.234457373999246) < 1e-15
    u_d = exchange_velocity(_t(SIGMA_W), _t(H), _t(W), form="schulte")
    assert abs(float(u_d) / 0.09042890 - 1.0) < 1e-7
    # At a_r = 1 the Schulte form collapses onto the SIRANE one exactly.
    same = exchange_velocity(_t(SIGMA_W), _t(10.0), _t(10.0), form="schulte")
    torch.testing.assert_close(
        same, exchange_velocity(_t(SIGMA_W), _t(10.0), _t(10.0), form="sirane"),
        rtol=1e-15, atol=0,
    )


def test_t3_sigma_w_in_all_three_stability_branches():
    """`ComputeSigmaW`, `StreetNetworkTransport.cxx:3221-3260`; neither paper gives it."""
    layer = BoundaryLayer(u_star=_t(U_STAR), h_abl=_t(PBLH), z_ref=_t(30.0),
                          d=_t(4.6), z0=_t(0.69), kappa=KAPPA_MUNICH)
    neutral = layer.sigma_w(_t(H), lmo=_t(1.0e6), stability="munich")
    assert abs(float(neutral) - 0.3856944) < 1e-13
    stable = layer.sigma_w(_t(H), lmo=_t(100.0), stability="munich")
    assert abs(float(stable) - 0.38798000423523393) < 1e-13
    unstable = layer.sigma_w(_t(H), lmo=_t(-50.0), stability="munich")
    # The record quotes 0.473173 to six significant figures.
    assert abs(float(unstable) / 0.473173 - 1.0) < 1e-6
    # And the neutral branch IS IMPAQ's formula, evaluated at z = H.
    torch.testing.assert_close(neutral, layer.sigma_w(_t(H)), rtol=1e-15, atol=0)


def test_t4_exponential_canyon_wind_is_kim_2022_b14_and_not_kim_2018_9_to_11():
    """K22 Eq. (B14) p. 7388; ATM `ComputeExpUstreet`, `MeteorologyStreet.cxx:257-263`.

    Single regime, `2/a_r` prefactor, integrated from `z0_s`. K18 Eqs. (9)-(11) are three
    aspect-ratio regimes with a `2/pi` prefactor integrated from 0 -- a different formula
    under the same name, and 37 % away for a narrow canyon.
    """
    u_street = canyon_velocity(_t(W), _t(H), _t(0.0), u_h=_t(5.0), form="exponential",
                               z0_s=Z0_S)
    assert abs(float(u_street) - 4.003210417532266) < 1e-12
    assert abs(float(u_street) / 5.0 - 0.8006420835064532) < 1e-13
    for angle, expected in ((30.0, 3.466882), (60.0, 2.001605)):
        got = canyon_velocity(_t(W), _t(H), _t(math.radians(angle)), u_h=_t(5.0),
                              form="exponential", z0_s=Z0_S)
        assert abs(float(got) / expected - 1.0) < 1e-6      # quoted to seven figures


def test_t5_macdonald_displacement_roughness_and_roof_wind(record_property):
    """K22 Eqs. (1)-(3) p. 7373 and (B13); `ComputeMacdonaldProfile` `:3302-3353`."""
    d_c, z0c = macdonald_profile(_t(H), _t(W))
    assert abs(float(d_c) - 4.617352498423888) < 1e-12
    assert abs(float(z0c) - 0.6614635677623194) < 1e-12
    u_h = roof_wind(_t(U_STAR), _t(H), _t(W), form="macdonald", h_mean=_t(H),
                    w_mean=_t(W))
    assert abs(float(u_h) - 0.9063192631810709) < 1e-12
    u_star_from_ref = 5.0 * KAPPA_MUNICH / math.log((30.0 - float(d_c)) / float(z0c))
    assert abs(u_star_from_ref - 0.5620494130195227) < 1e-12
    worst = max(abs(float(d_c) - 4.617352498423888), abs(float(z0c) - 0.6614635677623194))
    record_property("check", "Macdonald d_c, z_0c")
    record_property("tolerance", "< 1e-12")
    record_property("measured_diff", worst)


def test_t6_soulhac_shape_parameter_and_bessel_roof_wind(record_property):
    """K22 Eqs. (B12) and (B15) p. 7387-7388; ATM `MeteorologyStreet.cxx:114-226`."""
    delta_i = min(H, W / 2.0)
    assert delta_i == 3.75
    ratio = Z0_S / delta_i
    assert abs(ratio - 0.0026666666666666666) < 1e-18
    c = soulhac_shape(_t(ratio))
    # MUNICH searches a 0.01 grid and returns 0.62; this is the continuous root, and the
    # quantisation costs 4e-4 relative in u_M (`.superpowers/munich-formulas.md` T6).
    assert abs(float(c) - 0.6198293039179747) < 1e-13
    u_h = roof_wind(_t(U_STAR), _t(H), _t(W), form="sirane", z0_s=Z0_S,
                    kappa=KAPPA_MUNICH)
    # u_H/u* = (u_M/u*) * f_mean = 8.611791 * 0.880654, both quoted to seven figures.
    assert abs(float(u_h) / U_STAR / (8.611791 * 0.880654) - 1.0) < 2e-6
    record_property("check", "soulhac_shape root c")
    record_property("tolerance", "< 1e-13")
    record_property("measured_diff", abs(float(c) - 0.6198293039179747))


@pytest.mark.parametrize(
    "n, total",
    [(2, 0.431928), (3, 1.013848), (4, 0.995837), (5, 0.990866), (6, 0.986315),
     (7, 0.982560), (8, 0.979509), (9, 0.977016), (10, 0.974953)],
)
def test_t7_the_nine_unnormalised_quadrature_weight_sums(n, total, record_property):
    """K22 Eq. (B16) p. 7388; `ComputeWindDirectionFluctuation` `:3562-3616`."""
    sigma = _t((n + 0.5) * math.pi / 180.0)
    _offsets, weights = direction_offsets("munich", sigma)
    assert weights.shape[-1] == n
    measured = float(weights.sum())
    assert abs(measured - total) < 1e-6      # quoted to six decimal places
    # `measured` is recorded as the ABSOLUTE DIFFERENCE from the quoted K22 total, not the
    # raw sum -- the raw sum (e.g. 0.432 for n=2) reads, out of context in a results
    # table, as a failed check against some unstated target (final whole-branch review,
    # finding/minor 10).
    record_property("check", f"Direction-quadrature weight sum, n={n}, vs the K22 quoted {total}")
    record_property("tolerance", "1e-6")
    record_property("measured_abs_diff", abs(measured - total))


def test_t7_sigma_v_sigma_theta_and_the_sample_count():
    """`ComputeSigmaV` `:3194-3216`; the neutral branch collapses to exactly 1.2 u*."""
    layer = BoundaryLayer(u_star=_t(U_STAR), h_abl=_t(PBLH), z_ref=_t(30.0),
                          d=_t(4.6), z0=_t(0.69), kappa=KAPPA_MUNICH)
    sigma_v = layer.sigma_v()
    assert abs(float(sigma_v) - 1.2 * U_STAR) < 1e-15
    assert abs(float(sigma_v) - 0.36) < 1e-15
    five = sigma_theta_munich(sigma_v, _t(5.0))
    ten = sigma_theta_munich(sigma_v, _t(10.0))
    assert abs(float(five) - 0.072) < 1e-15
    assert abs(float(ten) - 0.036) < 1e-15
    assert abs(float(five) * 180.0 / math.pi - 4.1253) < 1e-3
    assert abs(float(ten) * 180.0 / math.pi - 2.0626) < 1e-3
    assert int(n_theta_munich(five)) == 4
    assert int(n_theta_munich(ten)) == 2


def test_t8_the_non_crossing_routing_matrix_distinguishes_sirane_from_mixing():
    """`ComputeAlpha`, `StreetNetworkTransport.cxx:3620-3648`."""
    flux_in = torch.tensor([10.0, 4.0], dtype=DT)
    flux_out = torch.tensor([6.0, 8.0], dtype=DT)
    torch.testing.assert_close(
        routing_matrix(flux_in, flux_out, model="sirane"),
        torch.tensor([[6.0, 4.0], [0.0, 4.0]], dtype=DT), rtol=0, atol=1e-14,
    )
    torch.testing.assert_close(
        routing_matrix(flux_in, flux_out, model="mixing"),
        torch.tensor([[30.0 / 7.0, 40.0 / 7.0], [12.0 / 7.0, 16.0 / 7.0]], dtype=DT),
        rtol=1e-14, atol=0,
    )


def test_t9_the_node_closure_both_ways():
    """`ComputeIntersectionFlux` `:2980-3014`."""
    _p_in, _p_out, to_atm, _from_atm = node_closure(
        torch.tensor([10.0, 4.0, -6.0, -5.0], dtype=DT)
    )
    assert abs(float(to_atm[0]) - 2.142857142857143) < 1e-13
    assert abs(float(to_atm[1]) - 0.8571428571428571) < 1e-13
    _p_in, _p_out, _to_atm, from_atm = node_closure(
        torch.tensor([4.0, 2.0, -6.0, -5.0], dtype=DT)
    )
    assert abs(float(from_atm[2]) - 2.727272727272727) < 1e-13
    assert abs(float(from_atm[3]) - 2.2727272727272725) < 1e-13


def test_t10_the_steady_single_street():
    """`.superpowers/munich-formulas.md` T10: the stationary solve of
    `StreetNetworkTransport.cxx:2573-2575` on one street with no inflow."""
    length = 100.0
    u_d = float(exchange_velocity(_t(SIGMA_W), _t(H), _t(W), form="schulte"))
    roof = u_d * W * length
    outflow = H * W * 4.0
    assert abs(roof - 67.82167214) < 1e-7          # quoted to ten figures
    assert abs(outflow - 207.0) < 1e-12
    assert abs(1.0 / (roof + outflow) / 3.63872e-3 - 1.0) < 1e-5   # six figures


def test_t11_the_leighton_rate_constant_and_its_conversion():
    """`include/modules/chemistry/Leighton/reactions`; SPACK `ARR2 A B` = A exp(-B/T)."""
    assert abs(K_NO_O3_298 - 1.9546779094727322e-14) < 1e-29
    assert abs(K_NO_O3_298 / 1.954678e-14 - 1.0) < 1e-6           # seven figures
    assert abs(3.0e-12 * math.exp(-1500.0 / 298.15) / 1.959634e-14 - 1.0) < 1e-6
    assert abs(3.0e-12 * math.exp(-1500.0 / 300.0) / 2.021384e-14 - 1.0) < 1e-6
    conversion = 6.02214076e23 / (1e6 * 48.0e-3)
    assert abs(conversion - 1.2546126583333333e19) < 1e4
    assert abs(K_NO_O3 / 2.45236e5 - 1.0) < 2e-6                  # six figures


def test_t13_the_two_paper_versus_paper_divergences_are_pinned():
    """The two places a reader of one paper alone would implement the wrong thing."""
    # (i) the exchange coefficient: 1/(sqrt2 pi), not 1/sqrt(2 pi) -- a factor sqrt(pi).
    assert abs((1.0 / math.sqrt(2.0 * math.pi)) / SIRANE_EXCHANGE
               - math.sqrt(math.pi)) < 1e-12
    # (ii) the exponential profile: K22 B14's 2/a_r prefactor, not K18 Eq. (9)'s 2/pi.
    a_r = H / W
    b14 = (2.0 / a_r) * (1.0 - math.exp(0.5 * a_r * (Z0_S / H - 1.0)))
    k18 = (2.0 / math.pi) * (2.0 / a_r) * (1.0 - math.exp(0.5 * a_r * (-1.0)))
    assert abs(b14 - 0.8006420835064532) < 1e-13
    assert abs(k18 / b14 - 1.0) > 0.35


# ------------------------------------------------------- the 12-street idealised case

def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _run(fixture):
    """The 12-street case at the fixture's stated geometry and option set."""
    geometry = fixture["fitted_geometry"]
    options = dict(fixture["options"])
    options.pop("comment")
    net, names = munich_idealised(L=geometry["L_m"], W=geometry["W_m"],
                                  H=geometry["H_m"])
    model, state, _ = build_model(net, stability="impaq", pblh_floor=True,
                                         **options)
    graph = model.net
    sources = torch.zeros(graph.n, dtype=DT)
    sources[graph.node_index(fixture["emitting_street"])] = 1.0
    out = {}
    for scenario in fixture["scenarios"]:
        solved = model.steady(state, {
            "street.x_boundary": torch.zeros(1, dtype=DT),
            "street.sources": sources,
            "U_ref": torch.tensor(scenario["speed_mps"], dtype=DT),
            "theta_w": torch.tensor(scenario["theta_w_rad"], dtype=DT),
            "h_abl": torch.tensor(geometry["h_abl_m"], dtype=DT),
        })["street.x"]
        out[scenario["panel"]] = {n: float(solved[i]) for i, n in enumerate(names)}
    return out, names


def test_the_fixture_matches_the_network_the_application_builds():
    fixture = _fixture()
    net, names = munich_idealised()
    assert names == [s["id"] for s in fixture["streets"]]
    for street, record in zip(net.streets, fixture["streets"], strict=True):
        assert (street.name, street.u, street.v) == (record["id"], record["u"],
                                                     record["v"])
    for scenario in fixture["scenarios"]:
        expected = math.radians((-90.0 - scenario["wind_from_deg"]) % 360.0)
        assert abs(scenario["theta_w_rad"] - expected) < 1e-15


def test_the_transcribed_table_is_linear_in_the_wind_speed_where_the_paper_says_it_is():
    """The paper's own numbers: doubling the wind halves every concentration at 210 and
    240 degrees, which is what proves no chemistry, no deposition and a zero background.

    The test is on the DIFFERENCE, not the ratio: the figure prints two decimal places, so
    the slow value carries +-0.005 and twice the fast value +-0.010, and a pair like
    street 1's 0.05 against 0.03 has a printed RATIO of 1.67 while still being consistent
    with exactly 2. 0.015 is that rounding envelope and nothing more.
    """
    table = _fixture()["concentrations_ug_m3"]
    for slow, fast in (("a", "d"), ("b", "e")):
        for street, value in table[slow].items():
            assert abs(value - 2.0 * table[fast][street]) <= 0.015
    # And at 270 degrees it is NOT 2, which is the `ustreet_min` floor's fingerprint.
    assert abs(table["c"]["11"] / table["f"]["11"] - 1.99451) < 1e-5
    assert abs(table["c"]["9"] / table["f"]["9"] - 4.0) < 1e-12


def test_the_model_is_exactly_linear_in_the_wind_speed_at_210_and_240_degrees():
    """A property of THIS model, provable and therefore assertable at solver precision:
    every flow scales with `u*`, `u*` scales with `U_ref`, `sigma_theta = min(1.2 u*/U, 10
    deg)` is independent of it, the background is zero and there is no chemistry, so every
    concentration is exactly proportional to `1/U`."""
    out, names = _run(_fixture())
    for slow, fast in (("a", "d"), ("b", "e")):
        for street in names:
            if out[fast][street] <= 0.0:
                assert out[slow][street] == 0.0
                continue
            assert abs(out[slow][street] / out[fast][street] - 2.0) < 1e-9


def test_the_270_degree_fingerprints_of_the_canyon_wind_floor():
    """At 270 degrees street 11 is exactly perpendicular to the wind, so its along-canyon
    speed is the 0.1 m/s floor -- a U-INDEPENDENT quantity. Two consequences follow, and
    both are visible in the paper."""
    fixture = _fixture()
    out, _names = _run(fixture)
    eleven = out["c"]["11"] / out["f"]["11"]
    nine = out["c"]["9"] / out["f"]["9"]
    # (i) street 11's ratio is strictly between 1 and 2, because its outflow is
    # `u_d W L (proportional to U) + a constant`.
    assert 1.0 < eleven < 2.0
    # (ii) street 9 is fed a U-independent flux and diluted by its own U-proportional
    # outflow, so its ratio is exactly twice street 11's. This is an identity of the
    # model, not a fit, and it holds to solver precision.
    assert abs(nine / (2.0 * eleven) - 1.0) < 1e-9
    # Against the paper: 1.99451 and 4.00. The agreement depends on the geometry, which is
    # unpublished; at the fixture's fitted geometry it is close enough to be evidence.
    table = fixture["concentrations_ug_m3"]
    assert abs(eleven / (table["c"]["11"] / table["f"]["11"]) - 1.0) < 5e-3
    assert abs(nine / 4.0 - 1.0) < 1e-2


def test_the_labelled_and_unlabelled_pattern_is_reproduced():
    """With the emission scaled so that street 11 matches the paper panel by panel, every
    street the paper labels must come out at or above the 0.005 ug/m3 label threshold and
    every blank one below it -- except where the model lands within a factor two of the
    threshold itself, where the figure's rounding decides and the test cannot."""
    fixture = _fixture()
    out, names = _run(fixture)
    table = fixture["concentrations_ug_m3"]
    threshold = fixture["blank_threshold_ug_m3"]
    marginal = []
    for panel, printed in table.items():
        scale = printed["11"] / out[panel]["11"]
        for street in names:
            value = out[panel][street] * scale
            ours = value >= threshold
            theirs = street in printed
            if ours != theirs:
                if 0.5 * threshold <= value <= 2.0 * threshold:
                    marginal.append((panel, street, value))
                    continue
                raise AssertionError(
                    f"panel {panel}, street {street}: the model gives {value:.4g} "
                    f"ug/m3, which is {'above' if ours else 'below'} the {threshold} "
                    f"label threshold, and the paper "
                    f"{'labels' if theirs else 'leaves blank'} it"
                )
    # Measured while the plan was written: exactly two marginal cells, street 2 of panel
    # (a) at 0.0051 and street 10 of panel (f) at 0.0050, both within a hair of 0.005.
    assert len(marginal) <= 2


def test_the_relative_pattern_is_compared_and_its_residual_recorded(capsys):
    """Spec section 7's last row for this case, honestly.

    The paper's inputs are unpublished, so the comparison is made under the fixture's
    STATED uniform geometry -- the best of a 150-point grid. Under that assumption seven of
    the nineteen printed ratios land inside 15 % and the worst is 51 %; the spec's 5 % is NOT
    attained, and this test records the residual rather than absorbing it. A change in any
    of these numbers is a change in the model.
    """
    fixture = _fixture()
    out, _names = _run(fixture)
    table = fixture["concentrations_ug_m3"]
    rows = []
    for panel, printed in table.items():
        for street, value in printed.items():
            if street == "11":
                continue
            ours = out[panel][street] / out[panel]["11"]
            theirs = value / printed["11"]
            rows.append((panel, street, ours, theirs, ours / theirs - 1.0))
    with capsys.disabled():
        print("\nMUNICH Fig. 1, ratio to street 11 under the fitted uniform geometry:")
        for panel, street, ours, theirs, residual in rows:
            print(f"  panel {panel} street {street:>2}: model {ours:.4e}  "
                  f"paper {theirs:.4e}  residual {residual:+.1%}")
    worst = max(abs(residual) for *_rest, residual in rows)
    inside_15 = sum(1 for *_rest, residual in rows if abs(residual) < 0.15)
    with capsys.disabled():
        print(f"  worst residual {worst:.1%}; {inside_15} of {len(rows)} ratios "
              f"inside 15 %")
    # Measured 0.507 on 18 September 2026 (seven of nineteen ratios inside 15 %); a
    # regression past 0.6 is a real change in the relative pattern, not noise. The bounds
    # are wide enough to be a regression guard and tight enough that a real change trips
    # them.
    assert worst < 0.6
    assert inside_15 >= 6


def test_photostationary_chemistry_on_the_twelve_street_network():
    """Ruling M3-R8 (spec section 4.6 as amended at 2280a7e).

    MUNICH's idealised case itself has NO chemistry -- its 5-vs-10 m/s ratios are exactly 2
    -- and its inputs are unpublished, so the photostationary reaction is checked here on the
    same 12-street network run through `street_steady`: the Leighton state is reached on
    every street that holds any NOx, and molar NOx and Ox are conserved street by street
    against the transport-only steady state (the reaction moves mass between NO, NO2 and O3
    inside a street; it never creates or destroys nitrogen or odd oxygen).
    """
    fixture = _fixture()
    geometry = fixture["fitted_geometry"]
    options = dict(fixture["options"])
    options.pop("comment")
    net, names = munich_idealised(L=geometry["L_m"], W=geometry["W_m"],
                                  H=geometry["H_m"])
    reaction = photostationary_for_streets(("no", "no2", "o3"))
    model, state, _ = build_model(
        net, species=("no", "no2", "o3"), chemistry=reaction, stability="impaq",
        pblh_floor=True, **options,
    )
    graph = model.net
    sources = torch.zeros(graph.n, 3, dtype=DT)
    emitter = graph.node_index(fixture["emitting_street"])
    sources[emitter, 0] = 8.0e-4                  # NO,  kg/s
    sources[emitter, 1] = 2.0e-4                  # NO2, kg/s
    background = torch.zeros(1, 3, dtype=DT)
    background[0, 2] = 8.0e-8                     # O3 aloft, kg/m3
    j = 5.0e-3                                    # 1/s, a mid-morning J_NO2
    scenario = fixture["scenarios"][0]
    drivers = {
        "street.x_boundary": background,
        "street.sources": sources,
        "U_ref": torch.tensor(scenario["speed_mps"], dtype=DT),
        "theta_w": torch.tensor(scenario["theta_w_rad"], dtype=DT),
        "h_abl": torch.tensor(geometry["h_abl_m"], dtype=DT),
        "J_NO2": torch.tensor(j, dtype=DT),
    }
    transport_only = model.steady(state, drivers)["street.x"]
    out = street_steady(model, state, drivers, reaction=reaction, tol=1e-18, max_iter=200)
    x = out["street.x"]
    assert x.shape == (len(names), 3)
    assert bool((x >= 0).all())
    masses = torch.tensor([MOLAR_MASS["no"], MOLAR_MASS["no2"], MOLAR_MASS["o3"]],
                          dtype=DT)
    c, c0 = x / masses, transport_only / masses
    # Molar NOx and Ox, street by street, against the transport-only state.
    torch.testing.assert_close(c[:, 0] + c[:, 1], c0[:, 0] + c0[:, 1], rtol=1e-12, atol=0)
    torch.testing.assert_close(c[:, 1] + c[:, 2], c0[:, 1] + c0[:, 2], rtol=1e-12, atol=0)
    # k [NO][O3] = J [NO2] wherever there is any NOx at all. Streets the plume never reaches
    # hold NO = NO2 = 0 exactly (the background carries no NOx), where the balance is 0 = 0
    # and a relative residual is undefined; they are excluded by the mask, and the mask is
    # asserted non-trivial so the check cannot pass vacuously.
    k_mol = K_NO_O3 * MOLAR_MASS["o3"]
    scale = j * c[:, 1] + k_mol * c[:, 0] * c[:, 2]
    reached = scale > 0
    assert bool(reached[street_index(model)[fixture["emitting_street"]]])
    assert int(reached.sum()) >= 3
    residual = j * c[:, 1] - k_mol * c[:, 0] * c[:, 2]
    assert float((residual[reached] / scale[reached]).abs().max()) < 1e-10


def test_the_idealised_case_matches_its_golden():
    from tests.golden import load_golden

    out, names = _run(_fixture())
    golden = load_golden("munich_idealised")
    for panel, values in golden.items():
        for street in names:
            torch.testing.assert_close(
                torch.tensor(out[panel][street], dtype=DT),
                torch.tensor(values[street], dtype=DT), rtol=1e-8, atol=1e-30,
            )
