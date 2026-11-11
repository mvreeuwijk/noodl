"""`solve_qp`, the dual active-set method for a small strictly convex quadratic program."""

import pytest
import torch

from noodl.solvers.qp import InfeasibleConstraints, solve_qp

F64 = torch.float64


def test_solve_qp_on_small_diode_networks():
    """`solve_qp` on the resistor-and-diode programs of the water application's limit heads
    (`noodl.apps.water.cut_off._limit`), against the answers by hand (unit conductances; the
    multipliers are the open valves' flows).

    One junction J between reservoirs at 100, 50 and 90 m, with check valves J -> R1,
    R2 -> J and R3 -> J: `1/2 [(x - 100)^2 + (x - 50)^2 + (x - 90)^2]` with `x <= 100`,
    `x >= 50`, `x >= 90`. The average 80 violates `x >= 90`, so J = 90 with a multiplier
    3 * 90 - 240 = 30 (P6 open). With R3 at 60 instead nothing is active and J = 70.

    The re-shut case: J1 and J2, Ra (90) -> J1, J1 -> J2 and Rb (60) -> J2 check valves,
    J1 - Rc (65) closed and ten closed links from J2 to reservoirs at 0:
    `G = [[3, -1], [-1, 12]]`, `a = [-155, -60]`. The unconstrained minimum (54.86, 9.57)
    violates `x2 >= 60` the most (by 50.4, against 35.1 and 45.3), so the method's first
    full step opens Pb; the answer is x1 = x2 = 90 with Pa and P7 active (multipliers 955
    and 930) and Pb shut again, dropped when P7's normal could not be added with both
    heads pinned."""
    g = torch.tensor([[3.0]], dtype=F64)
    a = torch.tensor([-240.0], dtype=F64)
    c = torch.tensor([[-1.0], [1.0], [1.0]], dtype=F64)
    b = torch.tensor([-100.0, 50.0, 90.0], dtype=F64)
    x, active, u = solve_qp(g, a, c, b, 1e-9)
    assert x.tolist() == pytest.approx([90.0], abs=1e-12)
    assert active == [2] and u == pytest.approx([30.0], abs=1e-12)
    x, active, u = solve_qp(g, torch.tensor([-210.0], dtype=F64), c,
                                    torch.tensor([-100.0, 50.0, 60.0], dtype=F64), 1e-9)
    assert x.tolist() == pytest.approx([70.0], abs=1e-12) and active == [] and u == []

    g = torch.tensor([[3.0, -1.0], [-1.0, 12.0]], dtype=F64)
    a = torch.tensor([-155.0, -60.0], dtype=F64)
    c = torch.tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]], dtype=F64)   # Pa, P7, Pb
    b = torch.tensor([90.0, 0.0, 60.0], dtype=F64)
    x0 = -torch.linalg.solve(g, a)
    assert int(torch.argmin(c @ x0 - b)) == 2   # Pb is the first constraint it reaches
    x, active, u = solve_qp(g, a, c, b, 1e-9)
    assert x.tolist() == pytest.approx([90.0, 90.0], abs=1e-12)
    assert sorted(active) == [0, 1] and 2 not in active
    assert dict(zip(active, u, strict=True)) == pytest.approx({0: 955.0, 1: 930.0}, abs=1e-9)


def test_an_infeasible_program_is_refused():
    g = torch.tensor([[1.0]], dtype=F64)
    c = torch.tensor([[1.0], [-1.0]], dtype=F64)    # x >= 2 and x <= 1
    with pytest.raises(InfeasibleConstraints):
        solve_qp(g, torch.zeros(1, dtype=F64), c,
                         torch.tensor([2.0, -1.0], dtype=F64), 1e-9)
