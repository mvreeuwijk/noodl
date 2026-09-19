"""Batched scalar root finding for monotone branch laws (e.g. inverting a fan curve).

Signature corrected from the spine's one-argument ``solve_monotone(f, lo, hi, *, tol,
max_iter)`` with ``f(x)``: an ``autograd.Function``'s ``backward`` only ever sees what
``forward`` saved on ``ctx``, never a closure's captured cells, so any tensor ``f`` depends
on besides ``x`` must be an explicit ``*params`` argument, ``f(x, *params)``, threaded
through ``Function.apply``. ``FanCurve`` (this task) is written against this signature.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Tensor = torch.Tensor


def solve_monotone(
    f: Callable[..., Tensor],
    lo: Tensor,
    hi: Tensor,
    *params: Tensor,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tensor:
    """Batched root of f(x, *params), monotone in x, bracketed by [lo, hi].

    Safeguarded Newton: each iteration takes a Newton step from the current iterate using
    the autograd derivative of f with respect to x; the step is accepted only if it lands
    strictly inside the current bracket, otherwise the iteration bisects. The bracket is
    updated every iteration from the sign of f at the current iterate, so it always contains
    the root. Differentiable in `params` by the implicit-function rule.
    """
    return _SolveMonotone.apply(f, lo, hi, tol, max_iter, *params)


class _SolveMonotone(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, lo, hi, tol, max_iter, *params):
        with torch.no_grad():
            lo_c = lo.clone()
            hi_c = hi.clone()
            f_lo = f(lo_c, *params)
            f_hi = f(hi_c, *params)
            bad_sign = (torch.sign(f_lo) == torch.sign(f_hi)) & (f_lo != 0) & (f_hi != 0)
            if torch.any(bad_sign):
                bad = torch.nonzero(bad_sign).flatten().tolist()
                raise RuntimeError(
                    f"solve_monotone: f(lo) and f(hi) have the same sign at batch "
                    f"indices {bad}; no sign change is bracketed"
                )
            increasing = f_hi >= f_lo

            x = 0.5 * (lo_c + hi_c)
            # at_floor tracks, per batch element, whether the bracket has hit its dtype's
            # representable-value floor: no float strictly between lo_c and hi_c exists, so
            # bisecting further recomputes one of the two endpoints exactly and can never
            # narrow the bracket again. This is unavoidable well above tol=1e-12 in float32
            # (whose relative precision is ~1.2e-7) for any bracket of order-1 magnitude or
            # larger, which is the normal regime for e.g. FanCurve's flow/pressure values; it
            # is what "as converged as this dtype allows" means, and is treated as success
            # below rather than as non-convergence. Recomputed fresh each iteration below (not
            # just once) because it must reflect the *final* lo_c/hi_c when the loop exits,
            # whether by `break` or by exhausting max_iter.
            at_floor = torch.zeros_like(x, dtype=torch.bool)
            for _ in range(max_iter):
                fx = f(x, *params)
                above = torch.where(increasing, fx > 0, fx < 0)
                hi_c = torch.where(above, x, hi_c)
                lo_c = torch.where(above, lo_c, x)

                # Bracket width alone (once combined with at_floor below) is the sound
                # convergence test: the bracket always contains the root (by construction from
                # the sign test above), so a narrow-enough bracket guarantees x is within tol
                # of the root regardless of how f behaves near it. |f(x)| < tol is NOT a safe
                # alternative on its own: at a root of multiplicity > 1 (f'(root) == 0, e.g.
                # f(x) = x**3 - c at c = 0), f flattens out near the root, so |f(x)| can already
                # be tiny while x is still far from the root in absolute terms -- this was
                # verified to produce a spurious early exit (x ~ 1e-4 reported as converged at
                # tol 1e-12) before this was pinned to the bracket width instead.
                done = (hi_c - lo_c).abs() < tol

                mid = 0.5 * (lo_c + hi_c)
                at_floor = (mid == lo_c) | (mid == hi_c)
                if torch.all(done | at_floor):
                    break

                x_leaf = x.detach().clone()
                x_leaf.requires_grad_(True)
                with torch.enable_grad():
                    fx_g = f(x_leaf, *params)
                    (dfx,) = torch.autograd.grad(fx_g.sum(), x_leaf)
                newton_x = torch.where(dfx != 0, x - fx / dfx, mid)
                inside = (newton_x > lo_c) & (newton_x < hi_c) & torch.isfinite(newton_x)
                x = torch.where(inside, newton_x, mid)

            # The loop above only ever narrows the bracket (or hits the dtype's floor), so if
            # it exhausts max_iter with some element neither within tol nor at_floor, that
            # element genuinely had budget left to make and did not use it: report it instead
            # of silently returning an under-converged x. Stays inside the enclosing
            # `no_grad` block (and this staticmethod runs with autograd tracking off
            # regardless, since it is a Function.forward), so it never touches the autograd
            # tape.
            final_width = (hi_c - lo_c).abs()
            unconverged = (final_width >= tol) & ~at_floor
            if torch.any(unconverged):
                bad = torch.nonzero(unconverged).flatten().tolist()
                widths = final_width[unconverged].tolist()
                raise RuntimeError(
                    f"solve_monotone: failed to converge within max_iter={max_iter} "
                    f"iterations at batch indices {bad} (bracket width(s) {widths}, "
                    f"tol={tol})"
                )

        ctx.save_for_backward(x, *params)
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, *params = ctx.saved_tensors
        f = ctx.f

        x_req = x.detach().clone()
        x_req.requires_grad_(True)
        params_req = tuple(p.detach().clone().requires_grad_(True) for p in params)
        with torch.enable_grad():
            fx = f(x_req, *params_req)
            (f_x,) = torch.autograd.grad(fx.sum(), x_req, create_graph=True, retain_graph=True)
            weight = -grad_output / f_x
            param_grads = torch.autograd.grad(
                fx, params_req, grad_outputs=weight, allow_unused=True
            )
        return (None, None, None, None, None, *param_grads)
