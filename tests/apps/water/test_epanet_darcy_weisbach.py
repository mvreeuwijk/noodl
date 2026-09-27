"""`EpanetDarcyWeisbach` against a line-by-line scalar transcription of EPANET 2.2's C.

The reference below is `DWpipecoeff` and `frictionFactor` from EPANET 2.2
`src/hydcoeffs.c`, with `resistcoeff`'s D-W resistance and `input1.c`'s minor-loss
conversion, written in plain `math` on floats in EPANET's own feet and cfs. It is NOT
built from the element's code, so agreement to rounding (1e-13 relative) checks the tensor
port operation for operation.
"""

import math

import pytest
import torch

from noodl.apps.water.elements import (
    EPANET_NU,
    EpanetDarcyWeisbach,
    epanet_friction_factor,
)

F64 = torch.float64
FT = 0.3048

# hydcoeffs.c, v2.2
A1 = 3.14159265358979323850e03
A2 = 1.57079632679489661930e03
A8 = 4.61841319859066668690e00
A9 = -8.68588963806503655300e-01
AB = 3.28895476345399058690e-03
AC = -5.14214965799093883760e-03


def _c_friction_factor(q, e, s):
    """`frictionFactor(q, e, s, &dfdq)` -> (f, dfdq)."""
    w = q / s
    if w >= A1:
        y1 = A8 / w**0.9
        y2 = e / 3.7 + y1
        y3 = A9 * math.log(y2)
        f = 1.0 / (y3 * y3)
        return f, 1.8 * f * y1 * A9 / y2 / y3 / q
    y2 = e / 3.7 + AB
    y3 = A9 * math.log(y2)
    fa = 1.0 / (y3 * y3)
    fb = (2.0 + AC / (y2 * y3)) * fa
    r = w / A2
    x1 = 7.0 * fa - fb
    x2 = 0.128 - 17.0 * fa + 2.5 * fb
    x3 = -0.128 + 13.0 * fa - (fb + fb)
    x4 = 0.032 - 3.0 * fa + 0.5 * fb
    return x1 + r * (x2 + r * (x3 + r * x4)), (x2 + r * (2.0 * x3 + r * 3.0 * x4)) / s / A2


def _c_dw(q_cfs, length_ft, d_ft, eps_ft, km, viscos=1.1e-5):
    """`DWpipecoeff` -> (hloss ft, hgrad ft/cfs) for SIGNED flow q (cfs)."""
    r = length_ft / 2.0 / 32.2 / d_ft / (math.pi * d_ft**2 / 4.0) ** 2
    ml = 0.02517 * km / d_ft**2 / d_ft**2
    e = eps_ft / d_ft
    s = viscos * d_ft
    q = abs(q_cfs)
    if q <= A2 * s:
        r = 16.0 * math.pi * s * r
        return q_cfs * (r + ml * q), r + 2.0 * ml * q
    f, dfdq = _c_friction_factor(q, e, s)
    r1 = f * r + ml
    return r1 * q * q_cfs, 2.0 * r1 * q + dfdq * r * q * q


LENGTH, DIAMETER, EPS = 300.0, 0.05, 0.05e-3  # m


def _reynolds_flow(re, diameter=DIAMETER):
    """The flow (m3/s) at Reynolds number `re` in EPANET's own water."""
    return re * math.pi * diameter * EPANET_NU / 4.0


def _pipe(minor=0.0, **kw):
    return EpanetDarcyWeisbach(
        torch.tensor([LENGTH], dtype=F64),
        torch.tensor([DIAMETER], dtype=F64),
        torch.tensor([EPS], dtype=F64),
        minor_loss=torch.tensor([minor], dtype=F64),
        **kw,
    )


@pytest.mark.parametrize("minor", [0.0, 2.0])
@pytest.mark.parametrize("re", [50.0, 1000.0, 1999.0, 2500.0, 3000.0, 3999.0, 4001.0, 1e5, 1e7])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_head_loss_and_gradient_match_the_c_source(re, minor, sign):
    """Every regime (laminar, Dunlop, Swamee-Jain) and both flow directions."""
    q = sign * _reynolds_flow(re)
    pipe = _pipe(minor)
    mine = float(pipe.head_loss(torch.tensor([q], dtype=F64))[0])
    grad = float(pipe.head_gradient(torch.tensor([q], dtype=F64))[0])
    c = 1.0 / FT**3
    h_ft, g_ft = _c_dw(q * c, LENGTH / FT, DIAMETER / FT, EPS / FT, minor)
    assert mine == pytest.approx(FT * h_ft, rel=1e-13)
    assert grad == pytest.approx(FT * c * g_ft, rel=1e-13)


def test_the_flow_unit_factor_scales_the_internal_cfs():
    """An LPS file's flows reach EPANET as q_lps / 28.317 cfs."""
    q = _reynolds_flow(3.0e4)
    cfs = 1000.0 / 28.317
    mine = float(_pipe(cfs_per_m3s=cfs).head_loss(torch.tensor([q], dtype=F64))[0])
    h_ft, _ = _c_dw(q * cfs, LENGTH / FT, DIAMETER / FT, EPS / FT, 0.0)
    assert mine == pytest.approx(FT * h_ft, rel=1e-13)


def test_the_friction_factor_is_continuous_with_its_slope_at_2000_and_4000():
    """Dunlop's cubic meets 64/Re in value and slope at Re = 2000 and Swamee-Jain in value
    and slope at Re = 4000 -- in exact arithmetic; the C constants AB/AC are rounded to
    ~17 digits, so the joins are checked to 1e-12 (value) and 1e-9 (slope)."""
    d = DIAMETER / FT
    s = torch.tensor([1.1e-5 * d], dtype=F64)
    e = torch.tensor([EPS / DIAMETER], dtype=F64)
    below, above = 1.0 - 1e-12, 1.0 + 1e-12
    # Re = 2000: laminar f = 64/Re = 16 pi s / q in these variables
    q2 = A2 * float(s[0])
    f_tr, dfdq_tr = epanet_friction_factor(torch.tensor([q2 * above], dtype=F64), e, s)
    assert float(f_tr[0]) == pytest.approx(64.0 / 2000.0, rel=1e-10)
    assert float(dfdq_tr[0]) == pytest.approx(-16.0 * math.pi * float(s[0]) / q2**2, rel=1e-9)
    # Re = 4000
    q4 = A1 * float(s[0])
    f_lo, g_lo = epanet_friction_factor(torch.tensor([q4 * below], dtype=F64), e, s)
    f_hi, g_hi = epanet_friction_factor(torch.tensor([q4], dtype=F64), e, s)
    assert float(f_lo[0]) == pytest.approx(float(f_hi[0]), rel=1e-11)
    assert float(g_lo[0]) == pytest.approx(float(g_hi[0]), rel=1e-9)


def test_flow_inverts_the_head_loss_in_every_regime():
    res = torch.tensor([0.0, 10.0, 1500.0, 2000.0, 2600.0, 3900.0, 4100.0, 5e4, 2e6], dtype=F64)
    q = torch.cat([res, -res[1:]]) * math.pi * DIAMETER * EPANET_NU / 4.0
    n = q.numel()
    pipe = EpanetDarcyWeisbach(
        torch.full((n,), LENGTH, dtype=F64), torch.full((n,), DIAMETER, dtype=F64),
        torch.full((n,), EPS, dtype=F64), minor_loss=torch.full((n,), 1.5, dtype=F64),
        scale=998.2 * 9.80665,
    )
    dp = pipe.head_loss(q) * pipe.scale
    back = pipe.flow(dp)
    assert torch.allclose(back, q, rtol=1e-13, atol=1e-20)
    # odd in dp
    assert torch.equal(pipe.flow(-dp), -back)


def test_dflow_is_the_reciprocal_head_gradient_and_finite_at_zero_flow():
    pipe = _pipe(2.0)
    zero = torch.zeros(1, dtype=F64)
    lam = 16.0 * math.pi * 1.1e-5 * (DIAMETER / FT) * (
        LENGTH / FT / 2.0 / 32.2 / (DIAMETER / FT) / (math.pi * (DIAMETER / FT) ** 2 / 4.0) ** 2
    )
    expected = 1.0 / (FT * lam / FT**3)
    assert float(pipe.dflow(zero)[0]) == pytest.approx(expected, rel=1e-13)
    c, k = pipe.linear_init()
    assert float(c[0]) == 0.0
    assert float(k[0]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("re", [0.0, 800.0, 3000.0, 4.0e4])
def test_flow_gradcheck_in_dp_and_roughness(re):
    """Implicit-function derivatives of the inversion, in dp and in the roughness.

    Both inputs are scaled to O(1) so that gradcheck's single step (1e-7) is a small
    relative perturbation of each and stays inside one regime."""
    q = _reynolds_flow(re)
    h = float(_pipe(1.0).head_loss(torch.tensor([q], dtype=F64))[0])
    h_ref = max(h, 1e-3)

    def fn(x, y):
        pipe = EpanetDarcyWeisbach(
            torch.tensor([LENGTH], dtype=F64), torch.tensor([DIAMETER], dtype=F64),
            EPS * y, minor_loss=torch.tensor([1.0], dtype=F64),
        )
        return pipe.flow(h_ref * x)

    x = torch.tensor([h / h_ref], dtype=F64, requires_grad=True)
    y = torch.ones(1, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x, y), eps=1e-7, atol=1e-10, rtol=1e-6)


def test_autograd_through_flow_agrees_with_dflow():
    q = torch.tensor([_reynolds_flow(r) for r in (0.0, 500.0, 2500.0, 3e4)], dtype=F64)
    n = q.numel()
    pipe = EpanetDarcyWeisbach(
        torch.full((n,), LENGTH, dtype=F64), torch.full((n,), DIAMETER, dtype=F64),
        torch.full((n,), EPS, dtype=F64), scale=9789.0,
    )
    dp = (pipe.head_loss(q) * pipe.scale).detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(pipe.flow(dp).sum(), dp)
    assert torch.allclose(grad, pipe.dflow(dp.detach()), rtol=1e-12)
