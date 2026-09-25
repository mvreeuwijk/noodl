"""Tests for `noodl.elements.mbl.powerlaw`: MBL's power-law flow-element family.

The NumPy reference functions below are transcribed directly from the Modelica Buildings
Library (MBL) v13.0.0 (commit 55abf579598ca81cae0a82f337350375958e6722) source, never by
importing `noodl.elements.mbl.powerlaw` itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.elements.mbl.powerlaw import (
    MBLPowerLaw,
    mbl_coefficient,
    mbl_ela,
    mbl_orifice,
    mbl_point,
    mbl_points,
)

# ---------------------------------------------------------------------------------------
# NumPy transcription of BaseClasses/powerLaw.mo and powerLawFixedM.mo
# ---------------------------------------------------------------------------------------


def _mbl_powerlaw_np(C, dp, m, dp_t):
    """Buildings/Airflow/Multizone/BaseClasses/powerLaw.mo:17-29, transcribed for this test."""
    gamma = 1.5
    dp = np.asarray(dp, dtype=np.float64)
    pi = dp / dp_t
    pi2 = pi * pi
    a = gamma
    b = 1 / 8 * m**2 - 3 * gamma - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * gamma + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - gamma - m + 15.0 / 8
    inner = C * dp_t**m * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))
    outer = np.sign(dp) * C * np.abs(dp) ** m
    return np.where(np.abs(dp) >= dp_t, outer, inner)


def _resistance_coefficients_np(m):
    """Buildings/Airflow/Multizone/BaseClasses/PowerLawResistanceParameters.mo:6-15."""
    gamma = 1.5
    a = gamma
    b = 1 / 8 * m**2 - 3 * gamma - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * gamma + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - gamma - m + 15.0 / 8
    return a, b, c, d


def _mbl_powerlaw_fixedm_np(C, dp, m, dp_t, a, b, c, d):
    """Buildings/Airflow/Multizone/BaseClasses/powerLawFixedM.mo:21-30, transcribed for this
    test; takes the polynomial coefficients as precomputed inputs, as MBL's Coefficient_V_flow
    and Coefficient_m_flow do (via PowerLawResistanceParameters) whenever `m` is a fixed
    parameter rather than a time-varying variable.
    """
    dp = np.asarray(dp, dtype=np.float64)
    pi = dp / dp_t
    pi2 = pi * pi
    inner = C * dp_t**m * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))
    outer = np.sign(dp) * C * np.abs(dp) ** m
    return np.where(np.abs(dp) >= dp_t, outer, inner)


_M_VALUES = (0.5, 0.65, 1.0)
_DP_T_VALUES = (0.1, 0.01)


def _dp_grid(dp_t):
    return np.concatenate(
        [np.linspace(-2 * dp_t, 2 * dp_t, 401), np.array([-50.0, -4.0, -1.0, 1.0, 4.0, 50.0])]
    )


# ---------------------------------------------------------------------------------------
# 0. Sanity: powerLaw (general m) and powerLawFixedM (precomputed a,b,c,d) agree exactly
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("dp_t", _DP_T_VALUES)
def test_powerlaw_and_powerlawfixedm_agree_for_the_same_m(m, dp_t):
    a, b, c, d = _resistance_coefficients_np(m)
    dp = _dp_grid(dp_t)
    lhs = _mbl_powerlaw_np(1.3, dp, m, dp_t)
    rhs = _mbl_powerlaw_fixedm_np(1.3, dp, m, dp_t, a, b, c, d)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-14)


# ---------------------------------------------------------------------------------------
# 1-2. MBLPowerLaw flow matches the MBL regularised power law, volume and mass forms
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("dp_t", _DP_T_VALUES)
def test_volume_form_flow_is_rho_default_times_mbl_powerlaw(m, dp_t):
    C = 1.3
    rho_default = 1.2
    el = MBLPowerLaw(
        C=torch.tensor(C, dtype=torch.float64),
        m=torch.tensor(m, dtype=torch.float64),
        dp_turbulent=dp_t,
        form="volume",
        rho_default=rho_default,
    )
    dp_np = _dp_grid(dp_t)
    dp = torch.tensor(dp_np, dtype=torch.float64)
    expected = rho_default * _mbl_powerlaw_np(C, dp_np, m, dp_t)
    got = el.flow(dp)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("dp_t", _DP_T_VALUES)
def test_mass_form_flow_is_mbl_powerlaw_without_rho_default(m, dp_t):
    C = 1.3
    el = MBLPowerLaw(
        C=torch.tensor(C, dtype=torch.float64),
        m=torch.tensor(m, dtype=torch.float64),
        dp_turbulent=dp_t,
        form="mass",
        rho_default=1.2,
    )
    dp_np = _dp_grid(dp_t)
    dp = torch.tensor(dp_np, dtype=torch.float64)
    expected = _mbl_powerlaw_np(C, dp_np, m, dp_t)
    got = el.flow(dp)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


# ---------------------------------------------------------------------------------------
# 3. Continuity of value and first two derivatives at dp = +-dp_turbulent
# ---------------------------------------------------------------------------------------


def _inner_branch_torch(C, m, dp, dp_t):
    """The regularised polynomial branch alone (``powerLaw.mo:24-29``), evaluated with no
    ``torch.where``/mask -- used only to probe continuity AT the exact switch point, where
    ``MBLPowerLaw._regularised`` itself always selects the "outer" branch (its mask is a
    strict ``<``).
    """
    gamma = 1.5
    a = gamma
    b = 1 / 8 * m**2 - 3 * gamma - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * gamma + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - gamma - m + 15.0 / 8
    pi = dp / dp_t
    pi2 = pi * pi
    return C * dp_t**m * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))


def _outer_branch_torch(C, m, dp):
    """The sharp branch alone (``powerLaw.mo:20-23``), evaluated with no mask."""
    return C * torch.sign(dp) * dp.abs() ** m


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("dp_t", _DP_T_VALUES)
def test_continuity_of_value_and_first_two_derivatives_at_transition(m, dp_t):
    """The polynomial is chosen (powerLaw.mo's documentation) to be twice continuously
    differentiable in dp at the switch: evaluate the inner (regularised) and outer (sharp)
    branches directly AT dp = +-dp_turbulent (not offset by a finite epsilon, which would
    conflate a genuine discontinuity with the ordinary Taylor difference of a smooth,
    nonzero-slope function) and compare value, 1st and 2nd derivative from each formula.
    """
    C_t = torch.tensor(1.3, dtype=torch.float64)
    m_t = torch.tensor(m, dtype=torch.float64)
    for boundary in (dp_t, -dp_t):
        inside = torch.tensor(boundary, dtype=torch.float64, requires_grad=True)
        outside = torch.tensor(boundary, dtype=torch.float64, requires_grad=True)

        v_in = _inner_branch_torch(C_t, m_t, inside, dp_t)
        v_out = _outer_branch_torch(C_t, m_t, outside)
        (d1_in,) = torch.autograd.grad(v_in, inside, create_graph=True)
        (d1_out,) = torch.autograd.grad(v_out, outside, create_graph=True)
        (d2_in,) = torch.autograd.grad(d1_in, inside)
        (d2_out,) = torch.autograd.grad(d1_out, outside)

        torch.testing.assert_close(v_in, v_out, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(d1_in, d1_out, rtol=1e-12, atol=1e-15)
        # d2 is the analytic SECOND derivative of a 7th-order polynomial vs. of |dp|^m,
        # computed through two nested torch.autograd.grad calls; for m == 1.0 the true value
        # on both sides is exactly 0, so this is a near-zero comparison dominated by float64
        # roundoff accumulated over the extra differentiation (observed ~1e-13), not by a
        # genuine discontinuity -- atol is loosened accordingly, unlike the exact
        # flow-vs-reference comparisons elsewhere in this file (rtol=1e-12, atol=1e-15).
        torch.testing.assert_close(d2_in, d2_out, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------------------
# 4. gradcheck, including dp = 0
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("dp_t", _DP_T_VALUES)
@pytest.mark.parametrize("form", ["volume", "mass"])
def test_gradcheck_inside_and_outside_the_band_including_zero(m, dp_t, form):
    el = MBLPowerLaw(
        C=torch.tensor(1.3, dtype=torch.float64),
        m=torch.tensor(m, dtype=torch.float64),
        dp_turbulent=dp_t,
        form=form,
        rho_default=1.2,
    )
    dp = torch.tensor(
        [-2 * dp_t, -0.5 * dp_t, 0.0, 0.5 * dp_t, 2 * dp_t],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp,), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------------------
# 5. Constructors' C matches a NumPy transcription of the MBL class's coefficient formula
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "A,CD,rho_default",
    [(0.01, 0.65, 1.2), (0.05, 0.7, 1.0)],
)
def test_mbl_orifice_coefficient(A, CD, rho_default):
    """Orifice.mo:3-5,9: `C = CD * A * sqrt(2 / rho_default)`, `m = 0.5` fixed default."""
    el = mbl_orifice(A=A, CD=CD, rho_default=rho_default)
    expected_C = CD * A * math.sqrt(2.0 / rho_default)
    torch.testing.assert_close(
        el.C, torch.tensor(expected_C, dtype=torch.float64), rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(el.m, torch.tensor(0.5, dtype=torch.float64))
    assert el.form == "volume"


@pytest.mark.parametrize(
    "L,dpRat,CDRat,m,rho_default",
    [(20e-4, 4.0, 1.0, 0.65, 1.2), (10e-4, 10.0, 0.6, 0.6, 1.0)],
)
def test_mbl_ela_coefficient(L, dpRat, CDRat, m, rho_default):
    """EffectiveAirLeakageArea.mo:3-5,7-14,16:
    `C = L * CDRat * sqrt(2 / rho_default) * dpRat ** (0.5 - m)`.
    """
    el = mbl_ela(L=L, dpRat=dpRat, CDRat=CDRat, m=m, rho_default=rho_default)
    expected_C = L * CDRat * math.sqrt(2.0 / rho_default) * dpRat ** (0.5 - m)
    torch.testing.assert_close(
        el.C, torch.tensor(expected_C, dtype=torch.float64), rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(el.m, torch.tensor(m, dtype=torch.float64))
    assert el.form == "volume"


@pytest.mark.parametrize(
    "dpMea,mMea,m",
    [(4.0, 0.019, 0.5), (50.0, 1.2 / 3600.0, 0.66)],
)
def test_mbl_point_coefficient(dpMea, mMea, m):
    """Point_m_flow.mo:4-5: `k = mMea_flow_nominal / dpMea_nominal ** m` (the `C` of the
    Coefficient_m_flow mass form)."""
    el = mbl_point(dpMea=dpMea, mMea_flow=mMea, m=m, rho_default=1.2)
    expected_k = mMea / dpMea**m
    torch.testing.assert_close(
        el.C, torch.tensor(expected_k, dtype=torch.float64), rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(el.m, torch.tensor(m, dtype=torch.float64))
    assert el.form == "mass"


@pytest.mark.parametrize(
    "dpMea,mMea",
    [((4.0, 50.0), (0.019, 0.15)), ((10.0, 100.0), (0.05, 0.25))],
)
def test_mbl_points_coefficient(dpMea, mMea):
    """Points_m_flow.mo:4-6,15-16: `m = (ln(mMea1)-ln(mMea2))/(ln(dpMea1)-ln(dpMea2))`,
    `k = mMea1 / dpMea1 ** m`."""
    el = mbl_points(dpMea=dpMea, mMea_flow=mMea, rho_default=1.2)
    expected_m = (math.log(mMea[0]) - math.log(mMea[1])) / (
        math.log(dpMea[0]) - math.log(dpMea[1])
    )
    expected_k = mMea[0] / dpMea[0] ** expected_m
    torch.testing.assert_close(
        el.m, torch.tensor(expected_m, dtype=torch.float64), rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(
        el.C, torch.tensor(expected_k, dtype=torch.float64), rtol=1e-12, atol=0.0
    )
    assert el.form == "mass"


@pytest.mark.parametrize(
    "C,m,rho_default",
    [(0.01, 0.59, 1.2), (3.33e-5 / 1.2, 0.59, 1.2)],
)
def test_mbl_coefficient_is_a_direct_pass_through(C, m, rho_default):
    """Coefficient_V_flow.mo/Coefficient_m_flow.mo: `C`/`k` and `m` are given directly."""
    el_v = mbl_coefficient(C=C, m=m, form="volume", rho_default=rho_default)
    torch.testing.assert_close(el_v.C, torch.tensor(C, dtype=torch.float64))
    torch.testing.assert_close(el_v.m, torch.tensor(m, dtype=torch.float64))
    assert el_v.form == "volume"

    el_m = mbl_coefficient(C=C, m=m, form="mass", rho_default=rho_default)
    torch.testing.assert_close(el_m.C, torch.tensor(C, dtype=torch.float64))
    assert el_m.form == "mass"


# ---------------------------------------------------------------------------------------
# 6. mbl_points reproduces both measured points exactly (well outside the turbulent band)
# ---------------------------------------------------------------------------------------


def test_mbl_points_fit_reproduces_both_measured_points():
    dpMea = (4.0, 50.0)
    mMea = (0.019, 0.15)
    el = mbl_points(dpMea=dpMea, mMea_flow=mMea, dp_turbulent=0.1, rho_default=1.2)
    dp = torch.tensor(dpMea, dtype=torch.float64)
    got = el.flow(dp)
    torch.testing.assert_close(got, torch.tensor(mMea, dtype=torch.float64), rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------------------
# 7. Broadcasting over leading batch dimensions (global constraint, per PowerLaw)
# ---------------------------------------------------------------------------------------


def test_mblpowerlaw_broadcasts_leading_batch_dims_against_per_edge_parameters():
    """Elements broadcast over leading batch dims, like every other noodl element
    (``noodl.elements.powerlaw.PowerLaw``'s own
    ``test_broadcasts_batched_parameters_against_batched_dp``) -- ``MBLPowerLaw`` must too,
    since ``_regularised`` uses only ordinary elementwise torch broadcasting with no
    special-casing of ``dp``'s shape. Per-edge ``C``/``m`` of shape ``(b,)`` against ``dp`` of
    shape ``(B, b)`` and ``(B1, B2, b)``.
    """
    dp_t = 0.1
    C = torch.tensor([1.0, 1.3, 0.7], dtype=torch.float64)  # per-edge, shape (b=3,)
    m = torch.tensor([0.5, 0.65, 1.0], dtype=torch.float64)  # per-edge, shape (b=3,)
    b = C.shape[0]
    el = MBLPowerLaw(C=C, m=m, dp_turbulent=dp_t, form="mass", rho_default=1.2)

    dp_np = _dp_grid(dp_t)  # shape (B,)
    B = dp_np.shape[0]
    dp2 = torch.tensor(dp_np, dtype=torch.float64).unsqueeze(-1).expand(B, b).contiguous()
    q2 = el.flow(dp2)
    assert q2.shape == (B, b)
    for e in range(b):
        expected = _mbl_powerlaw_np(C[e].item(), dp_np, m[e].item(), dp_t)
        torch.testing.assert_close(
            q2[:, e], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
        )

    B1, B2 = 4, 5
    dp3_np = np.linspace(-2 * dp_t, 2 * dp_t, B1 * B2).reshape(B1, B2)
    dp3 = torch.tensor(dp3_np, dtype=torch.float64).unsqueeze(-1).expand(B1, B2, b).contiguous()
    q3 = el.flow(dp3)
    assert q3.shape == (B1, B2, b)
    for e in range(b):
        expected = _mbl_powerlaw_np(C[e].item(), dp3_np, m[e].item(), dp_t)
        torch.testing.assert_close(
            q3[..., e], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
        )

    dp2_grad = dp2.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp2_grad,), eps=1e-6, atol=1e-8)


def test_mbl_orifice_constructor_broadcasts_leading_batch_dims():
    """Same broadcasting requirement for a constructor's output: ``mbl_orifice`` with a
    per-edge area ``A`` of shape ``(b,)`` against ``dp`` of shape ``(B, b)`` and
    ``(B1, B2, b)``.
    """
    dp_t = 0.1
    rho_default = 1.2
    A = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)  # per-edge, shape (b=3,)
    b = A.shape[0]
    el = mbl_orifice(A=A, CD=0.65, dp_turbulent=dp_t, rho_default=rho_default)
    expected_C = 0.65 * A * math.sqrt(2.0 / rho_default)
    torch.testing.assert_close(el.C, expected_C, rtol=1e-12, atol=0.0)

    dp_np = _dp_grid(dp_t)
    B = dp_np.shape[0]
    dp2 = torch.tensor(dp_np, dtype=torch.float64).unsqueeze(-1).expand(B, b).contiguous()
    q2 = el.flow(dp2)
    assert q2.shape == (B, b)
    for e in range(b):
        expected = rho_default * _mbl_powerlaw_np(expected_C[e].item(), dp_np, 0.5, dp_t)
        torch.testing.assert_close(
            q2[:, e], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
        )

    B1, B2 = 4, 5
    dp3_np = np.linspace(-2 * dp_t, 2 * dp_t, B1 * B2).reshape(B1, B2)
    dp3 = torch.tensor(dp3_np, dtype=torch.float64).unsqueeze(-1).expand(B1, B2, b).contiguous()
    q3 = el.flow(dp3)
    assert q3.shape == (B1, B2, b)
    for e in range(b):
        expected = rho_default * _mbl_powerlaw_np(expected_C[e].item(), dp3_np, 0.5, dp_t)
        torch.testing.assert_close(
            q3[..., e], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
        )

    dp2_grad = dp2.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp2_grad,), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------------------


def test_invalid_form_is_rejected():
    with pytest.raises(ValueError):
        MBLPowerLaw(
            C=torch.tensor(1.0, dtype=torch.float64),
            m=torch.tensor(0.5, dtype=torch.float64),
            form="bogus",
            rho_default=1.2,
        )
