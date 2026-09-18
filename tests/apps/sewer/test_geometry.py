"""Circular geometry and the Manning inversion (spec rows W5, W6)."""

import pytest
import torch

from tellegen.apps.sewer import geometry as g

F64 = torch.float64


def test_half_full_pipe_matches_the_closed_form():
    d = torch.tensor([0.30], dtype=F64)
    h = 0.5 * d
    assert float(g.flow_area(h, d)) == pytest.approx(torch.pi * 0.09 / 8.0, rel=1e-14)
    assert float(g.wetted_perimeter(h, d)) == pytest.approx(torch.pi * 0.15, rel=1e-14)
    assert float(g.top_width(h, d)) == pytest.approx(0.30, rel=1e-14)
    assert float(g.hydraulic_radius(h, d)) == pytest.approx(0.075, rel=1e-14)


def test_full_pipe_area_and_radius():
    d = torch.tensor([0.45], dtype=F64)
    h = d.clone()
    assert float(g.flow_area(h, d)) == pytest.approx(torch.pi * 0.45**2 / 4.0, rel=1e-9)
    assert float(g.hydraulic_radius(h, d)) == pytest.approx(0.45 / 4.0, rel=1e-9)


def test_geometry_is_finite_and_zero_at_zero_depth():
    d = torch.tensor([0.30], dtype=F64)
    h = torch.zeros(1, dtype=F64)
    for fn in (g.flow_area, g.wetted_perimeter, g.top_width, g.hydraulic_radius,
               g.hydraulic_mean_depth):
        value = fn(h, d)
        assert torch.isfinite(value).all()
        assert float(value) == pytest.approx(0.0, abs=1e-9)


def test_gradients_are_finite_at_both_ends_of_the_branch():
    d = torch.tensor([0.30, 0.30], dtype=F64)
    h = torch.tensor([0.0, g.H_MAX_RATIO * 0.30], dtype=F64, requires_grad=True)
    total = (
        g.flow_area(h, d) + g.hydraulic_radius(h, d) + g.top_width(h, d)
        + g.hydraulic_mean_depth(h, d)
    ).sum()
    total.backward()
    assert torch.isfinite(h.grad).all()


def test_manning_uses_the_si_constant_one():
    """A half-full 0.30 m pipe, n = 0.013, S0 = 0.01: Q = (1/n) A R^(2/3) sqrt(S0)."""
    d = torch.tensor([0.30], dtype=F64)
    h = 0.5 * d
    q = g.manning_flow(h, d, torch.tensor([0.013], dtype=F64),
                       torch.tensor([0.01], dtype=F64))
    area = torch.pi * 0.09 / 8.0
    expected = area * 0.075 ** (2.0 / 3.0) * 0.01**0.5 / 0.013
    assert float(q) == pytest.approx(expected, rel=1e-14)


def test_manning_flow_is_zero_and_finite_gradient_at_a_dry_pipe():
    """M4-R15: R^(2/3) has an infinite local derivative at R = 0 (h = 0), and diameter
    reaches hydraulic_radius through a plain multiplicative factor that bypasses _theta's
    clamp gate, so d(Q)/d(diameter) was NaN at exactly h = 0 before the R-floor guard."""
    h = torch.zeros(1, dtype=F64, requires_grad=True)
    d = torch.tensor([0.30], dtype=F64, requires_grad=True)
    n = torch.tensor([0.013], dtype=F64, requires_grad=True)
    s = torch.tensor([0.01], dtype=F64, requires_grad=True)
    q = g.manning_flow(h, d, n, s)
    assert float(q.detach()) == pytest.approx(0.0, abs=1e-9)
    grads = torch.autograd.grad(q.sum(), (h, d, n, s))
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert float(grad) == 0.0


def test_discharge_increases_strictly_on_the_ascending_branch():
    n = 2000
    d = torch.full((n,), 0.30, dtype=F64)
    h = torch.linspace(1e-9, g.H_MAX_RATIO * 0.30, n, dtype=F64)
    q = g.manning_flow(h, d, torch.full((n,), 0.013, dtype=F64),
                       torch.full((n,), 0.01, dtype=F64))
    assert bool((q[1:] > q[:-1]).all())


def test_capacity_is_the_discharge_at_the_peak_depth():
    d = torch.tensor([0.30], dtype=F64)
    n = torch.tensor([0.013], dtype=F64)
    s = torch.tensor([0.01], dtype=F64)
    assert float(g.capacity_flow(d, n, s)) == pytest.approx(
        float(g.manning_flow(g.H_MAX_RATIO * d, d, n, s)), rel=0.0
    )
    # Measured by the plan writer: 0.10402157331502969 m3/s.
    assert float(g.capacity_flow(d, n, s)) == pytest.approx(0.10402157331502969, rel=1e-12)


@pytest.mark.parametrize("diameter", [0.15, 0.30, 0.45, 1.65])
def test_w5_manning_inversion_round_trip(diameter):
    """Row W5. Measured worst 7.5e-14 relative over h/D in [0.01, 0.938]; BELOW h/D = 0.01
    the solver's absolute bracket tolerance dominates and it degrades to 1.1e-10, which is
    why the sweep starts at 0.01."""
    frac = torch.linspace(0.01, g.H_MAX_RATIO, 401, dtype=F64)
    d = torch.full_like(frac, diameter)
    n = torch.full_like(frac, 0.013)
    s = torch.full_like(frac, 0.005)
    h = frac * d
    q = g.manning_flow(h, d, n, s)
    back = g.normal_depth(q, d, n, s)
    again = g.manning_flow(back, d, n, s)
    assert float(((again - q).abs() / q).max()) < 1e-12
    assert float(((back - h).abs() / h).max()) < 1e-12


def test_normal_depth_reproduces_the_committed_tree():
    """The five conduits of tests/data/sewer/tree_steady.inp, measured against the research
    note's independent bisection solver (`.superpowers/swmm-research/manning_design.py`)."""
    d = torch.tensor([0.30, 0.30, 0.45, 0.30, 0.45], dtype=F64)
    n = torch.full((5,), 0.013, dtype=F64)
    s = torch.tensor([0.010, 0.010, 0.005, 0.010, 0.005], dtype=F64)
    q = torch.tensor([0.05, 0.08, 0.13, 0.03, 0.16], dtype=F64)
    h = g.normal_depth(q, d, n, s)
    expected = [0.153007001, 0.208101322, 0.262927897, 0.114723186, 0.302698602]
    assert h.tolist() == pytest.approx(expected, abs=1e-9)


def test_zero_flow_gives_zero_depth_and_finite_gradients():
    q = torch.tensor([0.05, 0.0, 0.13], dtype=F64, requires_grad=True)
    d = torch.tensor([0.30, 0.30, 0.45], dtype=F64)
    n = torch.tensor([0.013, 0.013, 0.013], dtype=F64, requires_grad=True)
    s = torch.tensor([0.010, 0.010, 0.005], dtype=F64)
    h = g.normal_depth(q, d, n, s)
    assert float(h[1].detach()) == 0.0
    h.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(n.grad).all()
    # Spec amendment A7: the zero-flow pipe's reported sensitivity is exactly zero.
    assert float(q.grad[1]) == 0.0
    assert float(n.grad[1]) == 0.0
    assert float(q.grad[0]) > 0.0


def test_w6_surcharge_is_refused_naming_the_pipe():
    """Row W6."""
    with pytest.raises(ValueError, match=r"surcharge at pipe\(s\) \['C1'\]"):
        g.normal_depth(
            torch.tensor([0.2], dtype=F64), torch.tensor([0.30], dtype=F64),
            torch.tensor([0.013], dtype=F64), torch.tensor([0.01], dtype=F64),
            names=["C1"],
        )


def test_negative_discharge_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        g.normal_depth(
            torch.tensor([-0.01], dtype=F64), torch.tensor([0.30], dtype=F64),
            torch.tensor([0.013], dtype=F64), torch.tensor([0.01], dtype=F64),
        )


def test_air_geometry_complements_the_water():
    d = torch.tensor([0.30], dtype=F64)
    h = 0.6 * d
    a_air, p_air, d_h = g.air_geometry(h, d)
    assert float(a_air) == pytest.approx(
        torch.pi * 0.09 / 4.0 - float(g.flow_area(h, d)), rel=1e-14
    )
    assert float(p_air) == pytest.approx(
        torch.pi * 0.30 - float(g.wetted_perimeter(h, d)) + float(g.top_width(h, d)),
        rel=1e-14,
    )
    assert float(d_h) == pytest.approx(4.0 * float(a_air) / float(p_air), rel=1e-14)
    # Measured at h/D = 0.6, D = 0.30 m: A_air 0.026403, T 0.293939, D_h 0.149855.
    assert float(a_air) == pytest.approx(0.026403, abs=1e-6)
    assert float(d_h) == pytest.approx(0.149855, abs=1e-6)


def test_air_geometry_is_zero_and_finite_at_a_full_pipe():
    d = torch.tensor([0.30], dtype=F64)
    a_air, p_air, d_h = g.air_geometry(d, d)
    assert float(a_air) == pytest.approx(0.0, abs=1e-9)
    assert torch.isfinite(d_h).all()
