"""SIRANE's NO-NO2-O3 chemistry closure: solar elevation, k1, k3 and the floored split."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.chemistry import (
    SIRANE_K_FLOOR_PPB,
    j_no2_elevation_cloud,
    k_no_o3_soulhac2011,
    molar_volume,
    photostationary_for_streets,
    solar_elevation,
    street_steady,
)
from noodl.apps.street_aq.network import Street, StreetNetwork, build_model
from noodl.layers.reaction import MOLAR_MASS, k_no_o3_jpl2003

DT = torch.float64
MASSES = torch.tensor([MOLAR_MASS["no"], MOLAR_MASS["no2"], MOLAR_MASS["o3"]], dtype=DT)


def _elevation_by_hand(lat, n, hour):
    dec = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))
    w = math.radians(15.0 * (hour - 12.0))
    p = math.radians(lat)
    return math.degrees(math.asin(math.sin(p) * math.sin(dec)
                                  + math.cos(p) * math.cos(dec) * math.cos(w)))


# ------------------------------------------------------------------------ solar position

def test_solar_elevation_follows_cooper_and_clock_time():
    for lat, n, hour in ((51.49415, 171, 12.0), (51.49415, 80, 7.5), (0.0, 355, 18.0),
                         (-33.9, 20, 9.25)):
        assert abs(float(solar_elevation(lat, n, hour)) - _elevation_by_hand(lat, n, hour)) \
            < 1e-12
    # Solar noon on the day Cooper's declination is zero ((284 + n) / 365 = 1, n = 81):
    # the elevation is 90 - latitude, and it is symmetric about noon (no equation of time).
    assert abs(float(solar_elevation(51.5, 81, 12.0)) - 38.5) < 1e-10
    torch.testing.assert_close(solar_elevation(51.5, 100, 9.0),
                               solar_elevation(51.5, 100, 15.0), rtol=1e-14, atol=0)


def test_solar_elevation_broadcasts_and_is_differentiable():
    hours = torch.linspace(0.0, 23.0, 24, dtype=DT)
    out = solar_elevation(51.49415, 171, hours)
    assert out.shape == (24,)
    assert float(out.max()) == pytest.approx(_elevation_by_hand(51.49415, 171, 12.0))
    h = torch.tensor([7.3, 12.6], dtype=DT, requires_grad=True)
    torch.autograd.gradcheck(lambda x: solar_elevation(51.49415, 200, x), (h,))


# ------------------------------------------------------------------------------------ k1

def test_k1_overhead_threshold_night_and_cloud():
    # Overhead, clear: 0.5699 / 60 = 9.50e-3 1/s.
    assert abs(float(j_no2_elevation_cloud(90.0)) - 0.5699 / 60.0) < 1e-18
    assert f"{float(j_no2_elevation_cloud(90.0)):.2e}" == "9.50e-03"
    # The bracket 0.5699 - [9.056e-3 (90 - a)]^2.546 reaches zero at a = 1.458 degrees.
    threshold = 90.0 - 0.5699 ** (1.0 / 2.546) / 9.056e-3
    assert abs(threshold - 1.458) < 5e-4
    assert float(j_no2_elevation_cloud(threshold - 1e-6)) == 0.0
    assert float(j_no2_elevation_cloud(threshold + 1e-3)) > 0.0
    assert float(j_no2_elevation_cloud(0.0)) == 0.0 and float(j_no2_elevation_cloud(-20.0)) == 0.0
    # A hand value at 30 degrees under 4 octas.
    a, n = 30.0, 4.0
    by_hand = (0.5699 - (9.056e-3 * (90 - a)) ** 2.546) / 60 * (1 - 0.75 * (n / 8) ** 3.4)
    assert abs(float(j_no2_elevation_cloud(a, n)) / by_hand - 1.0) < 1e-14
    # Full cloud keeps a quarter.
    full = float(j_no2_elevation_cloud(45.0, 8.0))
    assert abs(full / float(j_no2_elevation_cloud(45.0)) - 0.25) < 1e-15


def test_k1_names_cloud_outside_zero_to_eight():
    with pytest.raises(ValueError, match=r"j_no2_elevation_cloud: cloud_octas"):
        j_no2_elevation_cloud(30.0, 9.0)
    with pytest.raises(ValueError, match=r"j_no2_elevation_cloud: cloud_octas"):
        j_no2_elevation_cloud(30.0, -0.5)


def test_k1_is_differentiable_away_from_the_clip():
    a = torch.tensor([10.0, 45.0, 80.0], dtype=DT, requires_grad=True)
    n = torch.tensor([0.5, 4.0, 7.0], dtype=DT, requires_grad=True)
    torch.autograd.gradcheck(j_no2_elevation_cloud, (a, n))
    below = torch.tensor([0.5, -10.0], dtype=DT, requires_grad=True)
    (g,) = torch.autograd.grad(j_no2_elevation_cloud(below).sum(), below)
    assert torch.equal(g, torch.zeros(2, dtype=DT))


# ------------------------------------------------------------------------------------ k3

def test_k3_at_given_temperatures_and_in_ppb():
    for t in (263.15, 293.15, 308.15):
        assert abs(float(k_no_o3_soulhac2011(t)) / (1.325e6 * math.exp(-1430.0 / t)) - 1.0) < 1e-15
    # 1.325e6 m3 mol^-1 s^-1 is 2.2e-12 cm3 molecule^-1 s^-1 x N_A (to four figures).
    assert abs(1.325e6 / (2.2e-12 * 6.02214076e23 * 1e-6) - 1.0) < 5e-4
    # In ppb^-1 s^-1: k3 * 1e-9 / V_m. SIRANE prints 3.01e-04 at a ground temperature of
    # -3.0 C and a molar volume of 22.15 L/mol.
    k3_ppb = float(k_no_o3_soulhac2011(273.15 - 3.0)) * 1e-9 / 22.15e-3
    assert f"{k3_ppb:.2e}" == "3.01e-04"
    t = torch.tensor([280.0, 300.0], dtype=DT, requires_grad=True)
    torch.autograd.gradcheck(k_no_o3_soulhac2011, (t,))


# ---------------------------------------------------------------- the photostationary split

def _eq31_ppb(no2b, nob, o3b, nox_d, zeta, k_ppb):
    """Soulhac et al. (2011) Eq. 31 in ppb, written out as SIRANE applies it."""
    b = k_ppb + o3b + nob + 2 * no2b + (1 + zeta) * nox_d
    c = (o3b + no2b + zeta * nox_d) * (nob + no2b + nox_d)
    no2 = (b - math.sqrt(b * b - 4 * c)) / 2
    return nob + no2b + nox_d - no2, no2, o3b + no2b + zeta * nox_d - no2


def _sirane_drivers(k1, t, v_m):
    return {"J_NO2": torch.tensor(k1, dtype=DT), "temperature": torch.tensor(t, dtype=DT),
            "molar_volume": torch.tensor(v_m, dtype=DT)}


@pytest.mark.parametrize("k1", [6.0e-3, 1.0e-6, 0.0])
def test_the_sirane_closure_is_eq31_in_ppb_with_its_floor(k1):
    """Background plus passive NO/NO2 through the SIRANE closure equals Eq. 31 in ppb with
    K = max(k1/k3, 2 ppb); the three k1 values put K well above, just under and at the
    floor."""
    t, v_l = 291.0, 23.95                                     # K, L/mol
    reaction = photostationary_for_streets(("no", "no2", "o3"), preset="sirane")
    assert reaction.floor_ppb == SIRANE_K_FLOOR_PPB == 2.0
    bg = {"no2": 40.0, "no": 15.0, "o3": 50.0}                # ug/m3
    passive = {"no2": 12.0, "no": 55.0}                       # ug/m3, emissions only
    ug = torch.tensor([[bg["no"] + passive["no"], bg["no2"] + passive["no2"], bg["o3"]]],
                      dtype=DT)
    out = reaction.apply(ug * 1e-9, None, _sirane_drivers(k1, t, v_l * 1e-3)) * 1e9

    def ppb(value, species):
        return value * v_l / (MOLAR_MASS[species] * 1e3)

    no2_d, no_d = ppb(passive["no2"], "no2"), ppb(passive["no"], "no")
    k_ppb = max(k1 / (float(k_no_o3_soulhac2011(t)) * 1e-6 / v_l), 2.0)
    want = _eq31_ppb(ppb(bg["no2"], "no2"), ppb(bg["no"], "no"), ppb(bg["o3"], "o3"),
                     no2_d + no_d, no2_d / (no2_d + no_d), k_ppb)
    back = [want[0] * 30.0 / v_l, want[1] * 46.0 / v_l, want[2] * 48.0 / v_l]
    torch.testing.assert_close(out[0], torch.tensor(back, dtype=DT), rtol=1e-12, atol=0)


def test_the_molar_volume_only_matters_through_the_floor():
    reaction = photostationary_for_streets(("no", "no2", "o3"), preset="sirane")
    x = torch.tensor([[70e-9, 52e-9, 50e-9]], dtype=DT)
    day_a = reaction.apply(x, None, _sirane_drivers(6e-3, 291.0, 0.0240))
    day_b = reaction.apply(x, None, _sirane_drivers(6e-3, 291.0, 0.0220))
    torch.testing.assert_close(day_a, day_b, rtol=0, atol=0)
    night_a = reaction.apply(x, None, _sirane_drivers(0.0, 291.0, 0.0240))
    night_b = reaction.apply(x, None, _sirane_drivers(0.0, 291.0, 0.0220))
    assert float((night_a - night_b).abs().max()) > 0.0
    # Without the driver, the ideal-gas value at the temperature and 101325 Pa.
    no_vm = reaction.apply(x, None, {"J_NO2": 0.0, "temperature": 291.0})
    torch.testing.assert_close(
        no_vm, reaction.apply(x, None, _sirane_drivers(0.0, 291.0, float(molar_volume(291.0)))),
        rtol=0, atol=0)


def test_the_sirane_closure_conserves_nox_and_ox_and_is_differentiable():
    reaction = photostationary_for_streets(("no", "no2", "o3"), preset="sirane")
    ug = torch.tensor([[70.0, 52.0, 50.0], [5.0, 30.0, 80.0]], dtype=DT, requires_grad=True)
    k1 = torch.tensor([6e-3, 2e-3], dtype=DT, requires_grad=True)
    t = torch.tensor([291.0, 280.0], dtype=DT, requires_grad=True)

    def split(xx, kk, tt):
        """In and out in ug/m3, so that gradcheck's finite steps are resolvable."""
        drivers = {"J_NO2": kk, "temperature": tt,
                   "molar_volume": torch.tensor(0.024, dtype=DT)}
        return reaction.apply(xx * 1e-9, None, drivers) * 1e9

    y = split(ug, k1, t)
    c0, c1 = ug.detach() / MASSES, y.detach() / MASSES
    torch.testing.assert_close(c1[:, 0] + c1[:, 1], c0[:, 0] + c0[:, 1], rtol=1e-13, atol=0)
    torch.testing.assert_close(c1[:, 1] + c1[:, 2], c0[:, 1] + c0[:, 2], rtol=1e-13, atol=0)
    torch.autograd.gradcheck(split, (ug, k1, t))


def test_the_preset_is_named_and_sets_the_rate_and_the_floor():
    default = photostationary_for_streets(("no", "no2", "o3"))
    assert default.floor_ppb == SIRANE_K_FLOOR_PPB and default.rate is k_no_o3_soulhac2011
    munich = photostationary_for_streets(("no", "no2", "o3"), preset="munich")
    assert munich.floor_ppb == 0.0 and munich.rate is k_no_o3_jpl2003
    assert photostationary_for_streets(("no", "no2", "o3"), preset="sirane",
                                       floor_ppb=0.0).floor_ppb == 0.0
    mixed = photostationary_for_streets(("no", "no2", "o3"), no_o3_rate="jpl_2003")
    assert mixed.rate is k_no_o3_jpl2003 and mixed.floor_ppb == SIRANE_K_FLOOR_PPB
    custom = photostationary_for_streets(("no", "no2", "o3"), no_o3_rate=k_no_o3_soulhac2011,
                                         preset="munich")
    assert custom.rate is k_no_o3_soulhac2011 and custom.floor_ppb == 0.0
    with pytest.raises(ValueError, match=r"photostationary_for_streets: preset must be one "
                                         r"of \('sirane', 'munich'\), got 'chapman'"):
        photostationary_for_streets(("no", "no2", "o3"), preset="chapman")
    with pytest.raises(ValueError, match=r"no_o3_rate must be one of \('soulhac_2011', "
                                         r"'jpl_2003'\) or a callable"):
        photostationary_for_streets(("no", "no2", "o3"), no_o3_rate="arrhenius")


def test_the_closure_keyword_is_a_deprecated_spelling_of_preset():
    with pytest.warns(DeprecationWarning, match=r"'closure' is deprecated; use "
                                                r"preset='munich'"):
        old = photostationary_for_streets(("no", "no2", "o3"), closure="munich")
    assert old.rate is k_no_o3_jpl2003 and old.floor_ppb == 0.0
    with pytest.raises(TypeError, match=r"preset or its deprecated spelling closure"):
        photostationary_for_streets(("no", "no2", "o3"), closure="munich", preset="sirane")


def test_the_old_rate_names_are_the_same_functions():
    from noodl.apps.street_aq import chemistry
    from noodl.layers import reaction

    assert chemistry.k_no_o3_sirane is chemistry.k_no_o3_soulhac2011
    assert chemistry.j_no2_sirane is chemistry.j_no2_elevation_cloud
    assert chemistry.j_no2 is chemistry.j_no2_zenith_table
    assert reaction.k_no_o3_munich is reaction.k_no_o3_jpl2003
    assert chemistry.NO_O3_RATES == {"soulhac_2011": k_no_o3_soulhac2011,
                                     "jpl_2003": k_no_o3_jpl2003}


# ------------------------------------------ the coupled steady state and post-processing

def test_street_steady_equals_the_equilibrium_of_the_passive_steady_state():
    """Instantaneous chemistry conserving molar NOx and Ox, over transport that treats the
    three species alike: the coupled fixed point is the equilibrium applied street by
    street to the transport-only (passive plus background) steady state."""
    sn = StreetNetwork(
        streets=[Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 Street("s2", "b", "c", 100.0, 20.0, 15.0),
                 Street("s3", "b", "d", 80.0, 12.0, 18.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0, "d": 100.0},
        y={"a": 0.0, "b": 0.0, "c": 0.0, "d": 80.0},
    )
    reaction = photostationary_for_streets(("no", "no2", "o3"), preset="sirane")
    model, state, _ = build_model(sn, species=("no", "no2", "o3"), chemistry=reaction,
                                  pblh_floor=False)
    net = model.net
    sources = torch.zeros(net.n, 3, dtype=DT)
    sources[net.node_index("s1"), 0] = 2.0e-3
    sources[net.node_index("s1"), 1] = 4.0e-4
    sources[net.node_index("s3"), 0] = 5.0e-4
    background = torch.tensor([[15e-9, 40e-9, 50e-9]], dtype=DT)
    for k1 in (6.0e-3, 0.0):
        drivers = {
            "street.x_boundary": background, "street.sources": sources,
            "U_ref": torch.tensor(3.0, dtype=DT), "theta_w": torch.tensor(0.4, dtype=DT),
            "h_abl": torch.tensor(800.0, dtype=DT),
            **_sirane_drivers(k1, 291.0, 0.0239),
        }
        coupled = street_steady(model, state, drivers, reaction=reaction, tol=1e-20,
                                max_iter=200)["street.x"]
        post = reaction.apply(model.steady(state, drivers)["street.x"], None, drivers)
        torch.testing.assert_close(coupled, post, rtol=1e-11, atol=1e-22)
