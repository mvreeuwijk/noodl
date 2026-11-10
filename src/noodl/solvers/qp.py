"""A small, dense, strictly convex quadratic program with linear inequality constraints.

`solve_qp` is the dual active-set method of Goldfarb and Idnani (Mathematical Programming
27, 1983). It returns the active set and the multipliers as well as the minimiser, which is
what a caller that reads a network's state from them needs (the water application's limit
heads behind closed links and check valves holding no flow, `noodl.apps.water.cut_off`).
"""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


class InfeasibleConstraints(ValueError):
    """`solve_qp`: no point satisfies the constraints `rows` (indices into `c`)."""

    def __init__(self, rows: list[int]) -> None:
        super().__init__(f"infeasible constraints {rows}")
        self.rows = rows


def solve_qp(g: Tensor, a: Tensor, c: Tensor, b: Tensor, tol: float,
             ) -> tuple[Tensor, list[int], list[float]]:
    """`(x, active, multipliers)` of the strictly convex quadratic program
    ``minimise 1/2 x' g x + a' x subject to c x >= b`` (`g` symmetric positive definite,
    `c` one constraint per row): the dual active-set method of Goldfarb and Idnani
    (Mathematical Programming 27, 1983), in dense arithmetic, for a small program. A
    constraint is met when ``c x - b >= -tol``; `active` lists the rows held as equalities
    at `x`, in the order they joined, and `multipliers` their multipliers.

    It starts at the unconstrained minimum with no constraint active, and holds throughout:
    `x` minimises the objective with the active constraints as equalities, their
    multipliers are non-negative and their normals independent. Each outer step picks a
    violated constraint `p` and moves `x` towards it along the direction `z` that keeps the
    active equalities (the dual step `r` moves their multipliers); a multiplier that would
    turn negative drops its constraint first (a partial step), and `p` joins the active set
    when it is reached (a full step).

    It ends. With `p` violated and `z` nonzero, a full step has a positive length, so the
    objective, which the method never lets fall, strictly rises over every full step; after
    a full step `x` is the unique minimiser over the active set, whose objective depends on
    that set alone, so no active set returns, and there are finitely many. Between two
    full steps only drops occur, at most one per active constraint. When `z` is zero (the
    normal of `p` lies in the span of the active ones) and no multiplier can fall, no point
    satisfies the constraints (`InfeasibleConstraints`, naming `p` and the active set)."""
    chol = torch.linalg.cholesky(g)
    g_inv = torch.cholesky_inverse(chol)
    x = -(g_inv @ a)
    active: list[int] = []
    u: list[float] = []
    for _ in range(100 * (c.shape[0] + 1) * (g.shape[0] + 1)):
        s = c @ x - b
        p = int(torch.argmin(s))
        if float(s[p]) >= -tol:
            return x, active, u
        n_p = c[p]
        u_p = 0.0
        while True:
            if active:
                big_n = c[active].T                           # (n, k)
                m = big_n.T @ g_inv @ big_n                    # (k, k), positive definite
                n_star = torch.linalg.solve(m, big_n.T @ g_inv)
                h = g_inv - g_inv @ big_n @ n_star
                r = (n_star @ n_p).tolist()
            else:
                h = g_inv
                r = []
            z = h @ n_p
            z_n = float(z @ n_p)
            t1, drop = math.inf, None
            for j, (u_j, r_j) in enumerate(zip(u, r, strict=True)):
                if r_j > 1e-12 and u_j / r_j < t1:
                    t1, drop = u_j / r_j, j
            s_p = float(n_p @ x - b[p])
            t2 = math.inf if z_n <= 1e-12 else -s_p / z_n
            t = min(t1, t2)
            if t == math.inf:
                raise InfeasibleConstraints([p, *active])
            u = [u_j - t * r_j for u_j, r_j in zip(u, r, strict=True)]
            u_p += t
            if t2 != math.inf:
                x = x + t * z
            if t == t2:
                active.append(p)
                u.append(u_p)
                break
            del active[drop], u[drop]
    raise RuntimeError("solve_qp: the dual active-set method did not end")
