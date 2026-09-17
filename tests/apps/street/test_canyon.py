"""Canyon boundary layer, wind and exchange velocity — milestone 3 spec section 4.2, 4.3.

The Soulhac reference here is written out in scipy INSIDE this module, from IMPAQ's
`canyon_velocity` formula, so the test needs neither the AQ_DT repository nor the port of
Task 7. `brentq` replaces IMPAQ's `fsolve`: `fsolve` from x0 = 1.0 silently returns 1.0
unconverged for roughness ratios above about 0.5 (measured), which is outside this test's
range but is not a property a reference should have.
"""

from __future__ import annotations

import math

import pytest
import torch

from tellegen.apps.street.canyon import (
    C_BRACKET_HI,
    C_BRACKET_LO,
    GAMMA_E,
    KAPPA_IMPAQ,
    KAPPA_MUNICH,
    SCHULTE_BETA,
    SIRANE_EXCHANGE,
    Z0_B_DEFAULT,
    BoundaryLayer,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    boundary_layer,
    canyon_velocity,
    exchange_velocity,
    macdonald_profile,
    roof_wind,
    soulhac_residual,
    soulhac_shape,
)

DT = torch.float64


def _scipy_shape(ratio: float) -> float:
    from scipy.optimize import brentq
    from scipy.special import jv, yv

    def f(c):
        u = math.pi / 2.0 * yv(1, c) / jv(1, c) - GAMMA_E
        return 0.5 * ratio * c - (math.exp(u) if u < 700.0 else float("inf"))

    return brentq(f, C_BRACKET_LO, C_BRACKET_HI, xtol=1e-16, rtol=8.9e-16)


def _scipy_canyon_velocity(width, height, roughness, phi, u_star):
    """IMPAQ's `canyon_velocity`, scalar, with brentq for the shape parameter."""
    from scipy.special import jv, yv

    di = min(width / 2.0, height)
    c = _scipy_shape(roughness / di)
    alpha = math.log(di / roughness)
    beta = math.exp(c / math.sqrt(2.0) * (1.0 - height / di))
    u_h = u_star * math.sqrt(
        math.pi / (math.sqrt(2.0) * KAPPA_IMPAQ**2 * c)
        * (yv(0, c) - jv(0, c) * yv(1, c) / jv(1, c))
    )
    return (
        u_h * math.cos(phi) * di**2 / width / height
        * (
            2.0 * math.sqrt(2.0) / c * (1.0 - beta) * (1.0 - c**2 / 3.0 + c**4 / 45.0)
            + beta * (2.0 * alpha - 3.0) / alpha
            + (width / di - 2.0) * (alpha - 1.0) / alpha
        )
    )


def test_bessel_wrappers_match_scipy_over_the_solve_range():
    from scipy.special import jv, yv

    x = torch.linspace(0.05, 3.5, 200, dtype=DT)
    xs = x.numpy()
    for wrapped, reference in (
        (bessel_j0, jv(0, xs)), (bessel_j1, jv(1, xs)),
        (bessel_y0, yv(0, xs)), (bessel_y1, yv(1, xs)),
    ):
        # All four cross zero inside this range, where a relative comparison says nothing;
        # the absolute tolerance carries those points. The measured worst absolute
        # disagreement between the torch kernels and scipy over [0.05, 3.5] is 9e-11.
        torch.testing.assert_close(
            wrapped(x), torch.as_tensor(reference, dtype=DT), rtol=1e-9, atol=1e-9
        )


def test_bessel_wrappers_are_differentiable_and_use_the_standard_identities():
    x = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=DT, requires_grad=True)
    for wrapped, analytic in (
        (bessel_j0, lambda v: -bessel_j1(v)),
        (bessel_y0, lambda v: -bessel_y1(v)),
        (bessel_j1, lambda v: bessel_j0(v) - bessel_j1(v) / v),
        (bessel_y1, lambda v: bessel_y0(v) - bessel_y1(v) / v),
    ):
        (grad,) = torch.autograd.grad(wrapped(x).sum(), x)
        torch.testing.assert_close(grad, analytic(x.detach()), rtol=1e-12, atol=0)
        step = 1e-6
        central = (wrapped(x.detach() + step) - wrapped(x.detach() - step)) / (2 * step)
        torch.testing.assert_close(grad, central, rtol=2e-6, atol=1e-9)


def test_soulhac_shape_matches_brentq():
    ratios = [0.0026666666666666666, 0.0075, 0.015, 0.03, 0.1, 0.5]
    got = soulhac_shape(torch.tensor(ratios, dtype=DT))
    want = torch.tensor([_scipy_shape(r) for r in ratios], dtype=DT)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        soulhac_residual(got, torch.tensor(ratios, dtype=DT)),
        torch.zeros(len(ratios), dtype=DT), rtol=0, atol=1e-14,
    )


def test_soulhac_shape_is_differentiable_in_the_roughness_ratio():
    ratio = torch.tensor([0.015, 0.03], dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(soulhac_shape(ratio).sum(), ratio)
    step = 1e-8
    for i in range(2):
        plus = ratio.detach().clone()
        minus = ratio.detach().clone()
        plus[i] += step
        minus[i] -= step
        central = (soulhac_shape(plus)[i] - soulhac_shape(minus)[i]) / (2 * step)
        assert abs(float(grad[i]) / float(central) - 1.0) < 1e-6


def test_soulhac_shape_refuses_a_roughness_comparable_to_the_canyon():
    with pytest.raises(ValueError, match=r"soulhac_shape.*1\.6.*index"):
        soulhac_shape(torch.tensor([0.01, 2.0], dtype=DT))
    with pytest.raises(ValueError, match=r"soulhac_shape.*strictly positive"):
        soulhac_shape(torch.tensor([0.0], dtype=DT))


@pytest.mark.parametrize("w_over_h", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 90.0])
def test_soulhac_canyon_velocity_matches_the_scipy_reference(w_over_h, angle_deg):
    height, u_star = 20.0, 0.4388
    width = w_over_h * height
    phi = math.radians(angle_deg)
    got = canyon_velocity(
        torch.tensor([width], dtype=DT), torch.tensor([height], dtype=DT),
        torch.tensor([phi], dtype=DT), u_star=torch.tensor([u_star], dtype=DT),
        form="soulhac", z0_b=Z0_B_DEFAULT, kappa=KAPPA_IMPAQ,
    )
    want = _scipy_canyon_velocity(width, height, Z0_B_DEFAULT, phi, u_star)
    # atol covers the 90-degree column, where cos(phi) is 1e-17 and a relative
    # comparison is meaningless; every other column is decided by rtol.
    torch.testing.assert_close(got, torch.tensor([want], dtype=DT), rtol=1e-9, atol=1e-15)


def test_canyon_velocity_is_batched_over_streets_and_forcing_steps():
    width = torch.tensor([10.0, 20.0, 40.0], dtype=DT)
    height = torch.tensor([20.0, 20.0, 30.0], dtype=DT)
    phi = torch.tensor([[0.0, 0.5, 1.0], [2.0, 2.5, 3.0]], dtype=DT)
    u_star = torch.tensor([[0.3], [0.6]], dtype=DT)
    out = canyon_velocity(width, height, phi, u_star=u_star, form="soulhac")
    assert out.shape == (2, 3)
    # u_star enters linearly, so doubling it doubles the velocity exactly.
    torch.testing.assert_close(out[1], 2.0 * canyon_velocity(
        width, height, phi[1], u_star=torch.tensor([0.3], dtype=DT), form="soulhac"
    ), rtol=1e-12, atol=0)


def test_canyon_velocity_sign_follows_the_axis_and_the_floor_keeps_it():
    width = torch.tensor([20.0], dtype=DT)
    height = torch.tensor([20.0], dtype=DT)
    u_star = torch.tensor([0.4], dtype=DT)
    forward = canyon_velocity(width, height, torch.tensor([0.0], dtype=DT),
                              u_star=u_star, form="soulhac")
    backward = canyon_velocity(width, height, torch.tensor([math.pi], dtype=DT),
                               u_star=u_star, form="soulhac")
    assert float(forward) > 0.0 and float(backward) < 0.0
    torch.testing.assert_close(forward, -backward, rtol=1e-12, atol=0)
    # Exactly perpendicular: cos(phi) is +0.0 at pi/2 in float64 only to 6e-17, so the
    # floored speed keeps the POSITIVE sign -- MUNICH's `>` test at dangle == pi/2.
    floored = canyon_velocity(width, height, torch.tensor([0.5 * math.pi], dtype=DT),
                              u_star=u_star, form="soulhac", canyon_wind_min=0.1)
    torch.testing.assert_close(floored, torch.tensor([0.1], dtype=DT), rtol=1e-12, atol=0)
    unfloored = canyon_velocity(width, height, torch.tensor([0.5 * math.pi], dtype=DT),
                                u_star=u_star, form="soulhac")
    assert abs(float(unfloored)) < 1e-15


def test_exponential_canyon_velocity_is_kim_2022_equation_b14():
    # K22 Eq. (B14), p. 7388; ATM `ComputeExpUstreet`, MeteorologyStreet.cxx:257-263.
    u_h = torch.tensor([5.0], dtype=DT)
    got = canyon_velocity(
        torch.tensor([7.5], dtype=DT), torch.tensor([6.9], dtype=DT),
        torch.tensor([0.0], dtype=DT), u_h=u_h, form="exponential", z0_s=0.01,
    )
    torch.testing.assert_close(got, torch.tensor([4.003210417532266], dtype=DT),
                               rtol=1e-12, atol=0)


def test_exchange_velocity_sirane_and_schulte_agree_at_unit_aspect_ratio():
    sigma_w = torch.tensor([0.38569440], dtype=DT)
    one = torch.tensor([10.0], dtype=DT)
    sirane = exchange_velocity(sigma_w, one, one, form="sirane")
    schulte = exchange_velocity(sigma_w, one, one, form="schulte")
    torch.testing.assert_close(sirane, schulte, rtol=1e-14, atol=0)
    assert abs(SIRANE_EXCHANGE - 1.0 / (math.sqrt(2.0) * math.pi)) < 1e-16
    assert abs(SCHULTE_BETA - 2.0 * SIRANE_EXCHANGE) < 1e-16


def test_exchange_velocity_refuses_a_negative_sigma_w():
    with pytest.raises(ValueError, match=r"exchange_velocity.*negative.*pblh_floor"):
        exchange_velocity(torch.tensor([-0.01], dtype=DT), torch.tensor([20.0], dtype=DT),
                          torch.tensor([10.0], dtype=DT), form="sirane")


def test_boundary_layer_reproduces_impaq_and_floors_the_abl_when_asked():
    bl = boundary_layer(torch.tensor(23.3333333333333333, dtype=DT),
                        torch.tensor([2.0], dtype=DT), torch.tensor([1200.0], dtype=DT),
                        z_ref=30.0, kappa=KAPPA_IMPAQ)
    d = 2.0 * 23.3333333333333333 / 3.0
    z0 = 23.3333333333333333 / 10.0
    want = 0.4 * 2.0 / math.log((30.0 - d) / z0)
    torch.testing.assert_close(bl.u_star, torch.tensor([want], dtype=DT), rtol=1e-13, atol=0)
    floored = boundary_layer(torch.tensor(8.0, dtype=DT), torch.tensor([2.0], dtype=DT),
                             torch.tensor([12.4939704112137], dtype=DT), z_ref=30.0,
                             pblh_floor=22.5)
    torch.testing.assert_close(floored.h_abl, torch.tensor([22.5], dtype=DT),
                               rtol=1e-14, atol=0)


def test_boundary_layer_refuses_a_reference_height_inside_the_canopy():
    with pytest.raises(ValueError, match=r"boundary_layer.*z_ref.*displacement"):
        boundary_layer(torch.tensor(15.0, dtype=DT), torch.tensor([2.0], dtype=DT),
                       torch.tensor([1200.0], dtype=DT), z_ref=10.0)


def test_sigma_w_impaq_form_and_the_three_munich_branches():
    bl = BoundaryLayer(u_star=torch.tensor([0.3], dtype=DT),
                       h_abl=torch.tensor([500.0], dtype=DT),
                       z_ref=torch.tensor([30.0], dtype=DT),
                       d=torch.tensor([4.6], dtype=DT), z0=torch.tensor([0.69], dtype=DT),
                       kappa=KAPPA_MUNICH)
    z = torch.tensor([6.9], dtype=DT)
    torch.testing.assert_close(bl.sigma_w(z), torch.tensor([0.3856944], dtype=DT),
                               rtol=1e-13, atol=0)
    neutral = bl.sigma_w(z, lmo=torch.tensor([1e6], dtype=DT), stability="munich")
    torch.testing.assert_close(neutral, torch.tensor([0.3856944], dtype=DT),
                               rtol=1e-13, atol=0)
    stable = bl.sigma_w(z, lmo=torch.tensor([100.0], dtype=DT), stability="munich")
    torch.testing.assert_close(stable, torch.tensor([0.38798000423523393], dtype=DT),
                               rtol=1e-13, atol=0)
    unstable = bl.sigma_w(z, lmo=torch.tensor([-50.0], dtype=DT), stability="munich")
    torch.testing.assert_close(unstable, torch.tensor([0.4731731377067678], dtype=DT),
                               rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match=r"sigma_w.*stability='munich'.*lmo"):
        bl.sigma_w(z, stability="munich")


def test_sigma_v_neutral_is_exactly_1_point_2_u_star():
    bl = BoundaryLayer(u_star=torch.tensor([0.3], dtype=DT),
                       h_abl=torch.tensor([500.0], dtype=DT),
                       z_ref=torch.tensor([30.0], dtype=DT),
                       d=torch.tensor([4.6], dtype=DT), z0=torch.tensor([0.69], dtype=DT),
                       kappa=KAPPA_MUNICH)
    torch.testing.assert_close(bl.sigma_v(), torch.tensor([0.36], dtype=DT),
                               rtol=1e-13, atol=0)


def test_macdonald_and_bessel_roof_wind_reproduce_the_published_geometry():
    h = torch.tensor([6.9], dtype=DT)
    w = torch.tensor([7.5], dtype=DT)
    d_c, z0c = macdonald_profile(h, w)
    torch.testing.assert_close(d_c, torch.tensor([4.617352498423888], dtype=DT),
                               rtol=1e-12, atol=0)
    torch.testing.assert_close(z0c, torch.tensor([0.6614635677623194], dtype=DT),
                               rtol=1e-12, atol=0)
    u_star = torch.tensor([0.3], dtype=DT)
    macdonald = roof_wind(u_star, h, w, form="macdonald", h_mean=h, w_mean=w)
    torch.testing.assert_close(macdonald, torch.tensor([0.9063192631810709], dtype=DT),
                               rtol=1e-12, atol=0)
    sirane = roof_wind(u_star, h, w, form="sirane", z0_s=0.01, kappa=KAPPA_MUNICH)
    # u_H / u* = (u_M/u*) * f_mean = 8.611791 * 0.880654 at the continuous root.
    torch.testing.assert_close(sirane / u_star, torch.tensor([8.611791 * 0.880654],
                                                             dtype=DT), rtol=2e-6, atol=0)


def test_unknown_form_names_the_offender():
    args = (torch.tensor([10.0], dtype=DT), torch.tensor([20.0], dtype=DT))
    with pytest.raises(ValueError, match=r"canyon_velocity.*'soulhac'.*'lemonsu'"):
        canyon_velocity(*args, torch.tensor([0.0], dtype=DT),
                        u_star=torch.tensor([0.4], dtype=DT), form="lemonsu")
    with pytest.raises(ValueError, match=r"exchange_velocity.*'sirane'.*'wang'"):
        exchange_velocity(torch.tensor([0.3], dtype=DT), *args, form="wang")
    with pytest.raises(ValueError, match=r"roof_wind.*'sirane'.*'wang'"):
        roof_wind(torch.tensor([0.3], dtype=DT), *args, form="wang")
    with pytest.raises(ValueError, match=r"canyon_velocity.*form='soulhac'.*u_star"):
        canyon_velocity(*args, torch.tensor([0.0], dtype=DT), form="soulhac")
    with pytest.raises(ValueError, match=r"canyon_velocity.*form='exponential'.*u_h"):
        canyon_velocity(*args, torch.tensor([0.0], dtype=DT), form="exponential")
