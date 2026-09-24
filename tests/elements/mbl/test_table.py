"""Tests for `noodl.elements.mbl.table`: MBL's tabulated flow law (`Table_m_flow`/
`Table_V_flow`).

The NumPy reference functions below are transcribed directly from the Modelica Buildings
Library (MBL) v13.0.0 (commit 55abf579598ca81cae0a82f337350375958e6722) source and the
Modelica Standard Library (MSL) v4.1.0's `Modelica.Fluid.Utilities.cubicHermite`, never by
importing `noodl.elements.mbl.table` itself. Each Modelica `for`/`if` loop is transcribed as
its own Python loop (not vectorised with NumPy tricks like `searchsorted`), so this reference
cannot share a bug with the `torch.searchsorted`-based production implementation.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

from noodl.elements.mbl.table import MBLTable

# ---------------------------------------------------------------------------------------
# NumPy transcription of Modelica.Fluid.Utilities.cubicHermite (MSL Fluid/Utilities.mo:787-832)
# ---------------------------------------------------------------------------------------


def _cubic_hermite_np(x, x1, x2, y1, y2, y1d, y2d):
    """Fluid/Utilities.mo:807-824 (`cubicHermite`)."""
    h = x2 - x1
    if abs(h) > 0:
        t = (x - x1) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return y1 * h00 + h * y1d * h10 + y2 * h01 + h * y2d * h11
    # Degenerate case, x1 == x2 (Fluid/Utilities.mo:821-823); unreachable for a strictly
    # increasing MBLTable knot sequence but transcribed for completeness.
    return (y1 + y2) / 2


# ---------------------------------------------------------------------------------------
# NumPy transcription of Buildings/Utilities/Math/Functions/cubicHermiteLinearExtrapolation.mo
# ---------------------------------------------------------------------------------------


def _cubic_hermite_linear_extrapolation_np(x, x1, x2, y1, y2, y1d, y2d):
    """cubicHermiteLinearExtrapolation.mo:13-27."""
    if x > x1 and x < x2:
        return _cubic_hermite_np(x, x1, x2, y1, y2, y1d, y2d)
    elif x <= x1:
        return y1 + (x - x1) * y1d
    else:
        return y2 + (x - x2) * y2d


# ---------------------------------------------------------------------------------------
# NumPy transcription of Buildings/Utilities/Math/Functions/interpolate.mo
# ---------------------------------------------------------------------------------------


def _interpolate_scalar_np(u, xd, yd, d):
    """interpolate.mo:16-31, one scalar `u` at a time, Modelica's own 1-indexed `for` loop
    transcribed directly (kept 1-indexed in comments; `xd`/`yd`/`d` are 0-indexed Python
    sequences holding the same values in the same order as Modelica's 1-indexed arrays, so
    Modelica's `xd[j]` is `xd[j - 1]` here).
    """
    n = len(xd)
    i = 1  # interpolate.mo:17
    for j in range(1, n):  # interpolate.mo:18, "for j in 1:size(xd, 1) - 1"
        if u > xd[j - 1]:  # interpolate.mo:19, "if u > xd[j]"
            i = j  # interpolate.mo:20
    x1, x2 = xd[i - 1], xd[i]  # interpolate.mo:26-27, "xd[i]", "xd[i + 1]"
    y1, y2 = yd[i - 1], yd[i]
    y1d, y2d = d[i - 1], d[i]
    return _cubic_hermite_linear_extrapolation_np(u, x1, x2, y1, y2, y1d, y2d)


def _interpolate_np(u_array, xd, yd, d):
    u_array = np.asarray(u_array, dtype=np.float64)
    flat = [_interpolate_scalar_np(float(u), xd, yd, d) for u in u_array.reshape(-1)]
    return np.asarray(flat, dtype=np.float64).reshape(u_array.shape)


# ---------------------------------------------------------------------------------------
# NumPy transcription of Buildings/Utilities/Math/Functions/splineDerivatives.mo
# ---------------------------------------------------------------------------------------

# Modelica/Constants.mo:20 ("small = Minimum normalized positive floating-point number",
# ModelicaServices.Machine.small): for IEEE754 double precision this is DBL_MIN,
# 2.2250738585072014e-308, identical to Python's sys.float_info.min.
_MODELICA_SMALL = sys.float_info.min


def _spline_derivatives_np(x, y):
    """splineDerivatives.mo:31-67 with `ensureMonotonicity=true` (the only value
    `Table_m_flow.mo:19-23` ever passes). `x`, `y` are 0-indexed Python sequences of equal
    length holding the same values as Modelica's 1-indexed `x`/`y`.
    """
    x = list(x)
    y = list(y)
    n = len(x)
    if n == 1:
        return [0.0]  # splineDerivatives.mo:34
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])  # splineDerivatives.mo:37-38
        return [slope, slope]

    delta = [(y[i + 1] - y[i]) / (x[i + 1] - x[i]) for i in range(n - 1)]  # :42
    d = [0.0] * n
    d[0] = delta[0]  # :46, "d[1] := delta[1]"
    d[-1] = delta[-1]  # :47, "d[n] := delta[n - 1]"
    for i in range(1, n - 1):  # :49, "for i in 2:n - 1"
        d[i] = (delta[i - 1] + delta[i]) / 2  # :50

    # Ensure monotonicity (splineDerivatives.mo:55-68), n > 2 guaranteed here. Genuinely
    # sequential: interval i's update reads d[i], which interval i - 1 may have just
    # overwritten as its own d[i + 1].
    for i in range(n - 1):  # :56, "for i in 1:n - 1"
        if abs(delta[i]) < _MODELICA_SMALL:  # :57
            d[i] = 0.0  # :58
            d[i + 1] = 0.0  # :59
        else:
            alpha = d[i] / delta[i]  # :61
            beta = d[i + 1] / delta[i]  # :62
            if alpha**2 + beta**2 > 9:  # :64
                tau = 3 / math.sqrt(alpha**2 + beta**2)  # :65
                d[i] = delta[i] * alpha * tau  # :66
                d[i + 1] = delta[i] * beta * tau  # :67
    return d


# ---------------------------------------------------------------------------------------
# 0. Sanity: the two transcriptions agree with each other on a small hand-built table
# ---------------------------------------------------------------------------------------


def test_spline_derivatives_np_two_point_table_is_the_secant_slope():
    d = _spline_derivatives_np([0.0, 2.0], [0.0, 4.0])
    assert d == pytest.approx([2.0, 2.0])


# ---------------------------------------------------------------------------------------
# 1. MBLTable(form="mass") matches the NumPy reference on MBL's OneWayFlow tabDat_m_flow
# ---------------------------------------------------------------------------------------

# Buildings/Airflow/Multizone/Validation/OneWayFlow.mo:99-103 ("tabDat_m_flow").
_ONEWAYFLOW_DP = [-50, -25, -10, -5, -3, -2, -1, 0, 1, 2, 3, 4.5, 50]
_ONEWAYFLOW_M_FLOW = [
    -0.08709,
    -0.06158,
    -0.03895,
    -0.02754,
    -0.02133,
    -0.01742,
    -0.01232,
    0,
    0.01232,
    0.01742,
    0.02133,
    0.02613,
    0.02614,
]
# Buildings/Airflow/Multizone/Validation/OneWayFlow.mo:104-108 ("tabDat_V_flow"); same
# dpMea_nominal, VMea_flow_nominal is numerically identical to _ONEWAYFLOW_M_FLOW here (the
# validation model's two tables share the same numbers; Table_V_flow.mo:5 is what actually
# turns VMea_flow_nominal into a mass-flow table via `rho_default`).
_ONEWAYFLOW_V_FLOW = _ONEWAYFLOW_M_FLOW


def _dp_probe_grid(dp_points):
    """Below range, at every knot, midway inside every interval, and above range."""
    dp_points = list(dp_points)
    probes = [dp_points[0] - 17.0, dp_points[-1] + 17.0]
    probes += list(dp_points)
    for a, b in zip(dp_points[:-1], dp_points[1:], strict=True):
        probes.append((a + b) / 2.0)
    return np.array(probes, dtype=np.float64)


def test_mass_form_matches_reference_on_onewayflow_tabdat_m_flow():
    el = MBLTable(_ONEWAYFLOW_DP, _ONEWAYFLOW_M_FLOW, form="mass")
    d_ref = _spline_derivatives_np(_ONEWAYFLOW_DP, _ONEWAYFLOW_M_FLOW)
    dp_np = _dp_probe_grid(_ONEWAYFLOW_DP)
    expected = _interpolate_np(dp_np, _ONEWAYFLOW_DP, _ONEWAYFLOW_M_FLOW, d_ref)

    dp = torch.tensor(dp_np, dtype=torch.float64)
    got = el.flow(dp)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


def test_volume_form_matches_reference_scaled_by_rho_default_before_the_spline():
    """Table_V_flow.mo:4-5: `mMea_flow_nominal = VMea_flow_nominal*rho_default` -- the
    volume-rate table is scaled to a mass-flow table BEFORE `splineDerivatives`/`interpolate`
    run (not scaled after, the way `MBLPowerLaw`'s volume form multiplies `rho_default` in
    at evaluation time), since the Fritsch-Carlson monotonicity correction only commutes with
    positive scaling because `alpha`/`beta` are themselves scale-invariant ratios -- this test
    exercises the literal MBL order, not that equivalence.
    """
    rho_default = 1.2
    mass_points_ref = [rho_default * v for v in _ONEWAYFLOW_V_FLOW]
    d_ref = _spline_derivatives_np(_ONEWAYFLOW_DP, mass_points_ref)
    dp_np = _dp_probe_grid(_ONEWAYFLOW_DP)
    expected = _interpolate_np(dp_np, _ONEWAYFLOW_DP, mass_points_ref, d_ref)

    el = MBLTable(_ONEWAYFLOW_DP, _ONEWAYFLOW_V_FLOW, form="volume", rho_default=rho_default)
    dp = torch.tensor(dp_np, dtype=torch.float64)
    got = el.flow(dp)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


# ---------------------------------------------------------------------------------------
# 2. A random (non-uniformly spaced) monotone table
# ---------------------------------------------------------------------------------------


def _random_monotone_table(rng, n=9):
    dp = np.sort(rng.uniform(-100.0, 100.0, size=n))
    # Force strict increase (uniform draws could tie in principle).
    dp = dp + np.arange(n) * 1e-6
    steps = rng.uniform(0.5, 2.0, size=n)
    flow = np.cumsum(steps) - np.sum(steps) / 2.0  # strictly increasing
    return dp, flow


def test_mass_form_matches_reference_on_a_random_monotone_table():
    rng = np.random.default_rng(20260924)
    dp_np, flow_np = _random_monotone_table(rng)
    el = MBLTable(dp_np.tolist(), flow_np.tolist(), form="mass")
    d_ref = _spline_derivatives_np(dp_np.tolist(), flow_np.tolist())
    probes = _dp_probe_grid(dp_np.tolist())
    expected = _interpolate_np(probes, dp_np.tolist(), flow_np.tolist(), d_ref)

    dp = torch.tensor(probes, dtype=torch.float64)
    got = el.flow(dp)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


def test_monotonicity_preserved_on_a_monotone_table_sampled_densely():
    rng = np.random.default_rng(999)
    dp_np, flow_np = _random_monotone_table(rng, n=7)
    el = MBLTable(dp_np.tolist(), flow_np.tolist(), form="mass")

    probes = np.linspace(dp_np[0] - 10.0, dp_np[-1] + 10.0, 10_000)
    got = el.flow(torch.tensor(probes, dtype=torch.float64)).numpy()
    assert np.all(np.diff(got) >= -1e-12), "monotone table must yield a monotone interpolant"


# ---------------------------------------------------------------------------------------
# 3. gradcheck: inside intervals, at knots, and in the extrapolation range
# ---------------------------------------------------------------------------------------


def test_gradcheck_inside_at_knots_and_extrapolating():
    rng = np.random.default_rng(7)
    dp_np, flow_np = _random_monotone_table(rng, n=6)
    el = MBLTable(dp_np.tolist(), flow_np.tolist(), form="mass")

    probes = _dp_probe_grid(dp_np.tolist())
    dp = torch.tensor(probes, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp,), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------------------
# 4. Broadcasting over leading batch dimensions
# ---------------------------------------------------------------------------------------


def test_flow_broadcasts_leading_batch_dims():
    rng = np.random.default_rng(3)
    dp_np, flow_np = _random_monotone_table(rng, n=8)
    el = MBLTable(dp_np.tolist(), flow_np.tolist(), form="mass")
    d_ref = _spline_derivatives_np(dp_np.tolist(), flow_np.tolist())

    probes = _dp_probe_grid(dp_np.tolist())
    expected = _interpolate_np(probes, dp_np.tolist(), flow_np.tolist(), d_ref)
    B = probes.shape[0]

    dp2 = torch.tensor(probes, dtype=torch.float64).unsqueeze(0).expand(4, B).contiguous()
    q2 = el.flow(dp2)
    assert q2.shape == (4, B)
    for row in range(4):
        torch.testing.assert_close(
            q2[row], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
        )

    dp3 = dp2.unsqueeze(0).expand(3, 4, B).contiguous()
    q3 = el.flow(dp3)
    assert q3.shape == (3, 4, B)
    for a in range(3):
        for b in range(4):
            torch.testing.assert_close(
                q3[a, b], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-15
            )


# ---------------------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------------------


def test_invalid_form_is_rejected():
    with pytest.raises(ValueError):
        MBLTable([0.0, 1.0], [0.0, 1.0], form="bogus")


def test_volume_form_without_rho_default_is_rejected():
    with pytest.raises(ValueError):
        MBLTable([0.0, 1.0], [0.0, 1.0], form="volume")


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        MBLTable([0.0, 1.0, 2.0], [0.0, 1.0], form="mass")


def test_non_strictly_increasing_knots_are_rejected_naming_the_index():
    with pytest.raises(ValueError, match="2"):
        MBLTable([0.0, 1.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0], form="mass")


def test_decreasing_knots_are_rejected():
    with pytest.raises(ValueError):
        MBLTable([0.0, 2.0, 1.0], [0.0, 1.0, 2.0], form="mass")


def test_too_few_knots_are_rejected():
    with pytest.raises(ValueError):
        MBLTable([0.0], [0.0], form="mass")
