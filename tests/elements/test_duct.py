"""Duct: CONTAM's Colebrook duct (TN 1887r1 eq. 50-52) against a scipy reference.

PARAMS' geometry values are constructed as float64 tensors (rather than the plain Python
floats a reader might expect) because ``Element._param`` casts a non-tensor value with
``torch.get_default_dtype()``, which is float32 in this repo. The scipy oracle below is
evaluated in float64, and several assertions compare against it at ``rel=1e-8``/``1e-9`` --
tolerances a float32 ``Duct`` could not meet (relative error from float32 alone is about
1e-7). Passing float64 tensors makes ``Duct``'s own parameters float64 (via
``Element._dtype()``), so the comparisons are real dtype-matched checks, not accidents of
rounding.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.elements import Duct

F64 = torch.float64
PARAMS = dict(
    L=torch.tensor(10.0, dtype=F64),
    D=torch.tensor(0.3, dtype=F64),
    eps=torch.tensor(1.5e-4, dtype=F64),
    sum_C=torch.tensor(2.0, dtype=F64),
)
A = math.pi * 0.3**2 / 4
RHO, MU = 1.2041, 1.81625e-5


def _reference_F(dp: float) -> float:
    """F = sqrt(2 rho A^2 dp / (f L/D + sum_C)) with f from Colebrook, by nested brentq."""
    from scipy.optimize import brentq

    L = PARAMS["L"].item()
    D = PARAMS["D"].item()
    eps = PARAMS["eps"].item()
    sum_C = PARAMS["sum_C"].item()
    rel = eps / D

    def f_of_Re(Re: float) -> float:
        def colebrook(f: float) -> float:
            return 1 / math.sqrt(f) - (
                1.14 - 2 * math.log10(rel) - 2 * math.log10(1 + 9.3 / (Re * rel * math.sqrt(f)))
            )

        # Widened from (1e-4, 1.0): as Re -> 0 (the outer brentq's own lower endpoint probes
        # this), Colebrook's f leaves [1e-4, 1] -- colebrook(f) is positive at both original
        # ends and the inner solve can't bracket a root at all. (1e-6, 1e6) brackets the same
        # root everywhere the narrower interval already worked (agrees to ~1e-16 there) and
        # additionally lets the outer brentq evaluate its endpoint instead of raising.
        return brentq(colebrook, 1e-6, 1e6)

    def residual(F: float) -> float:
        f = f_of_Re(F * D / (MU * A))
        return F - math.sqrt(2 * RHO * A**2 * dp / (f * L / D + sum_C))

    return brentq(residual, 1e-6, 1e3)


@pytest.mark.parametrize("dp", [5.0, 50.0, 500.0])
def test_turbulent_flow_matches_the_scipy_reference(dp):
    el = Duct(**PARAMS, n_iter=8)
    F = el.flow(torch.tensor([dp], dtype=F64))
    assert F.item() == pytest.approx(_reference_F(dp), rel=1e-8)


def test_default_four_iterations_are_within_1e5_of_converged():
    dp = torch.tensor([5.0, 50.0, 500.0], dtype=F64)
    torch.testing.assert_close(
        Duct(**PARAMS).flow(dp), Duct(**PARAMS, n_iter=12).flow(dp), rtol=1e-5, atol=0.0
    )


def test_flow_is_odd_and_monotone():
    el = Duct(**PARAMS)
    dp = torch.linspace(-100.0, 100.0, 41, dtype=F64)
    q = el.flow(dp)
    torch.testing.assert_close(q, -el.flow(-dp))
    assert bool((q[1:] > q[:-1]).all())


def test_laminar_regime_is_linear_and_meets_the_turbulent_curve_at_Re_t():
    el = Duct(**PARAMS)
    F_t, dp_t = el._transition()
    assert F_t.item() * PARAMS["D"].item() / (MU * A) == pytest.approx(2000.0, rel=1e-9)
    half = el.flow(0.5 * dp_t)
    assert half.item() == pytest.approx(0.5 * F_t.item(), rel=1e-12)
    just_above = el.flow(dp_t * (1 + 1e-9))
    assert just_above.item() == pytest.approx(F_t.item(), rel=1e-5)


def test_the_transition_step_is_bounded_and_shrinks_with_n_iter():
    """`flow()` is DISCONTINUOUS at |dp| = dp_t, and this bounds the jump.

    `_transition()` iterates the Colebrook fixed point at the fixed `Re = Re_t`, while
    `_turbulent()` re-evaluates Re from the current F; at a finite `n_iter` the two stop at
    different iterates, so the laminar branch (which meets F_t exactly) and the turbulent
    branch do not meet. The step is the fixed point's own truncation error, so it must shrink
    with `n_iter` -- which is what makes it a truncation artefact rather than a modelling
    disagreement. See the `Duct` class docstring.
    """
    step = {}
    for n_iter in (1, 2, 4, 8):
        el = Duct(**PARAMS, n_iter=n_iter)
        F_t, dp_t = el._transition()
        step[n_iter] = abs(el._turbulent(dp_t).item() - F_t.item()) / F_t.item()
    assert step[4] < 1e-5                                    # the default: measured 1.6e-6
    assert step[8] < 1e-9                                    # measured 4.3e-11
    assert step[1] > step[2] > step[4] > step[8]
    # The jump is REAL: evaluated either side of dp_t, `flow` differs by that much and no
    # less, so the bound above is on the flow law and not on an artefact of _turbulent.
    el = Duct(**PARAMS)
    F_t, dp_t = el._transition()
    below = el.flow(dp_t * (1 - 1e-12)).item()
    above = el.flow(dp_t * (1 + 1e-12)).item()
    assert abs(above - below) / F_t.item() == pytest.approx(step[4], rel=1e-6)


def test_gradcheck_flow_wrt_dp_at_zero_inside_and_outside_the_transition():
    el = Duct(**PARAMS)
    _, dp_t = el._transition()
    t = dp_t.item()
    dp = torch.tensor([-30.0, -0.5 * t, 0.0, 0.5 * t, 30.0], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp,), eps=1e-7, atol=1e-6)


def test_dflow_default_matches_central_finite_differences():
    el = Duct(**PARAMS)
    dp = torch.tensor([-40.0, 3.0, 75.0], dtype=F64)
    h = 1e-5
    fd = (el.flow(dp + h) - el.flow(dp - h)) / (2 * h)
    torch.testing.assert_close(el.dflow(dp), fd, rtol=1e-6, atol=1e-9)


def test_learnable_duct_registers_parameters_and_eps_receives_a_gradient():
    el = Duct(**PARAMS, learnable=True)
    names = {n for n, _ in el.named_parameters()}
    assert {"L", "D", "eps", "sum_C"} <= names
    el.flow(torch.tensor([50.0], dtype=F64)).sum().backward()
    assert el.eps.grad is not None
    assert torch.isfinite(el.eps.grad).all() and el.eps.grad.abs().sum() > 0


def test_invalid_geometry_is_refused_naming_the_argument():
    with pytest.raises(ValueError, match=r"Duct.*eps"):
        Duct(L=10.0, D=0.3, eps=0.0)
    with pytest.raises(ValueError, match=r"Duct.*n_iter"):
        Duct(L=10.0, D=0.3, eps=1e-4, n_iter=0)


def test_batched_dp_broadcasts_against_scalar_parameters():
    el = Duct(**PARAMS)
    dp = torch.rand(4, 3, dtype=F64) * 100.0 + 1.0
    q = el.flow(dp)
    assert q.shape == (4, 3)
    torch.testing.assert_close(q[2, 1:2], el.flow(dp[2, 1:2]))
