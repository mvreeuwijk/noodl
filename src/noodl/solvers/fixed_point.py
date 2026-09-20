"""Implicit differentiation of a converged fixed point (P1-2 of the 20 Sep 2026 review).

A fixed-point iteration z_{k+1} = G(z_k, theta) that is differentiated by UNROLLING returns
the derivative of the truncated iteration, whose error is O(rho^passes) counted from the
START state and is tied to nothing the caller controls: a primal that starts at (or near) the
fixed point stops after one (or two) passes and returns one pass's derivative. This module
differentiates the converged interface equations instead. With `new = S(z*, theta)` the pass
at the fixed point and `G = F o S` (F reads the next iterate from `new`),

    d new / d theta = S_theta + S_z (I - G_z)^{-1} G_theta,

so for a cotangent g on `new` the parameter gradient is S_theta^T g + G_theta^T v with
(I - G_z^T) v = S_z^T g. `differentiate_fixed_point` runs ONE differentiable pass at z*,
wraps its outputs in an identity `autograd.Function` whose backward solves that small system
by GMRES (`solvers.iterative.gmres`, on the flattened interface, with its own residual
check) and hands autograd the cotangent g on the outputs AND the cotangent v on `z_next`,
which the ordinary backward pass through the retained single-pass graph turns into exactly
those two terms. Memory is one pass, independent of the pass count.

The derivation in full, for the record. Let z*(theta) solve z = G(z, theta) and write the
pass's whole output as new = S(z*(theta), theta), with z_next = G(z, theta) whatever part of
the pass produced it. Then

    d new/d theta = S_theta + S_z dz*/d theta,   dz*/d theta = (I - G_z)^{-1} G_theta,

so for a cotangent g,

    g^T d new/d theta = g^T S_theta + [(I - G_z^T)^{-1} S_z^T g]^T G_theta
                      = g^T S_theta + v^T G_theta,   (I - G_z^T) v = S_z^T g.

`backward` therefore needs three VJPs of the retained pass graph: `S_z^T g` (the outputs
against the interface leaves) for the right-hand side, `G_z^T w` (z_next against the leaves)
for the GMRES operator, and then `g^T S_theta + v^T G_theta`, which it does NOT compute
itself -- it returns g as the cotangent on the `outputs` inputs and v as the cotangent on the
`z_next` inputs of the Function, and autograd's own backward pass accumulates both through
the one graph to every parameter. The graph's only other terminals, the interface leaves z,
are detached from theta by construction, so nothing else escapes.

That last step is why `z_next` is an input of the Function rather than something `backward`
projects onto the outputs. It is tempting to write the second term as (F^T v)^T S_theta with
F the read of the next iterate out of `new`, and to evaluate F^T v as one more VJP of z_next
against outputs. That is WRONG in general, and wrong in precisely the case this module
exists for: when one output was computed FROM another (a Gauss-Seidel sweep, where the
second subsystem is solved with the first's new value), `torch.autograd.grad(z_next, outputs,
v)` propagates v along that inter-output edge as well, so the read picks up a term that
autograd then applies S_theta^T to a second time. Measured on x = (y + a)/2 followed by
y = (x + b)/2 at its fixed point, that construction returns dx*/da = 5/6 against a true 2/3.
Returning v on `z_next` has no such failure mode: where z_next IS an outputs entry, autograd
sums the two cotangents at the shared node (a selection, which is what F really is), and
where z_next was read out of the outputs by some other computation, that read's own VJP is
part of the same backward pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from noodl.solvers.iterative import gmres

__all__ = ["differentiate_fixed_point"]


def _flatten(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: Tensor, like: Sequence[Tensor]) -> list[Tensor]:
    out, offset = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[offset : offset + n].reshape(t.shape))
        offset += n
    return out


def _or_zeros(grads: Sequence[Tensor | None], like: Sequence[Tensor]) -> list[Tensor]:
    return [torch.zeros_like(t) if g is None else g for g, t in zip(grads, like, strict=True)]


def _vjp(
    outputs: Sequence[Tensor], cotangents: Sequence[Tensor], inputs: Sequence[Tensor]
) -> list[Tensor]:
    """`(d outputs / d inputs)^T @ cotangents`, as a list of dense tensors the length and
    shapes of `inputs`, zero where nothing reached an input.

    Entries on either side that carry no graph are dropped before the call: a pass is free to
    hand on a tensor that nothing differentiable reached (a constant boundary value, say), and
    `torch.autograd.grad` refuses such a tensor outright -- "does not require grad and does
    not have a grad_fn" for an output, "One of the differentiated Tensors does not require
    grad" for an input -- rather than returning `None` for it. Their contribution is exactly
    zero, so dropping them is not an approximation.
    """
    zeros = [torch.zeros_like(t) for t in inputs]
    live_out = [(o, c) for o, c in zip(outputs, cotangents, strict=True) if o.requires_grad]
    live_in = [(i, t) for i, t in enumerate(inputs) if t.requires_grad]
    if not live_out or not live_in:
        return zeros
    grads = torch.autograd.grad(
        [o for o, _ in live_out],
        [t for _, t in live_in],
        grad_outputs=[c for _, c in live_out],
        retain_graph=True,
        allow_unused=True,
    )
    for (i, _), g in zip(live_in, grads, strict=True):
        if g is not None:
            zeros[i] = g
    return zeros


class _AdjointOperator:
    """(I - J^T) on the flattened interface; J^T w is one VJP of the read z_next = G(z)."""

    def __init__(self, z: Sequence[Tensor], z_next: Sequence[Tensor]) -> None:
        self.z, self.z_next = list(z), list(z_next)
        m = sum(t.numel() for t in self.z)
        self.shape = (m, m)
        self.dtype, self.device = self.z[0].dtype, self.z[0].device

    def matvec(self, v: Tensor) -> Tensor:
        w = _unflatten(v, self.z)
        return v - _flatten(_vjp(self.z_next, w, self.z))


class _FixedPointAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n_z, rtol, atol, max_iter, where, *tensors):
        ctx.n_z, ctx.rtol, ctx.atol, ctx.max_iter, ctx.where = n_z, rtol, atol, max_iter, where
        ctx.z = tensors[:n_z]
        ctx.z_next = tensors[n_z : 2 * n_z]
        ctx.outputs = tensors[2 * n_z :]
        # Identity on the outputs: a view of each, so autograd routes their cotangents
        # through `backward` below and then on into the pass's own graph.
        return tuple(o.view_as(o) for o in ctx.outputs)

    @staticmethod
    def backward(ctx, *g_out):
        # Grad mode is ON inside a custom backward only under `create_graph=True`, i.e. when
        # a second differentiation is being prepared; the adjoint solve below builds no
        # graph, so that second derivative would silently miss a term. Refuse it here, as
        # `solvers/implicit.py` does.
        if torch.is_grad_enabled():
            raise RuntimeError(
                f"{ctx.where}: second-order differentiation (create_graph=True) through the "
                f"fixed-point adjoint is not supported; only first-order gradients via the "
                f"adjoint method are implemented, because its backward solves the adjoint "
                f"system outside the graph, exactly as the implicit solves do (see "
                f"solvers/implicit.py). Detach the first-order gradient before using it in a "
                f"further differentiable loss (e.g. a gradient-penalty term)."
            )
        z, z_next, outputs = ctx.z, ctx.z_next, ctx.outputs
        # An output nobody's loss touched arrives as a materialised zero under PyTorch's
        # default `materialize_grads`; `_or_zeros` keeps the `None` of the opposite setting
        # from reaching the VJPs below either way.
        g = _or_zeros(g_out, outputs)
        rhs = _flatten(_vjp(outputs, g, z))
        m = rhs.numel()
        op = _AdjointOperator(z, z_next)
        result = gmres(
            op,
            rhs,
            rtol=ctx.rtol,
            atol=ctx.atol,
            max_iter=ctx.max_iter,
            restart=min(m, 100),
        )
        if not bool(result.converged.all()):
            raise RuntimeError(
                f"{ctx.where}: the fixed-point adjoint solve did not converge (residual "
                f"{float(result.residual.max()):.3e} after {int(result.iterations.max())} "
                f"GMRES iterations, rtol={ctx.rtol}, atol={ctx.atol}); the interface "
                f"Jacobian is not a contraction at this point, or max_iter is too small"
            )
        # g on the `outputs` inputs and v on the `z_next` inputs: autograd's own backward
        # pass then accumulates g^T S_theta + v^T G_theta through the one retained graph.
        # See the module docstring for why v is NOT projected onto the outputs here.
        v = _unflatten(result.x, z)
        return (None, None, None, None, None, *([None] * ctx.n_z), *v, *g)


def differentiate_fixed_point(
    z_star: Sequence[Tensor],
    pass_fn: Callable[[list[Tensor]], tuple[list[Tensor], list[Tensor]]],
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    where: str = "fixed point",
) -> list[Tensor]:
    """One differentiable pass at the converged interface `z_star`, with the implicit adjoint
    attached to its outputs. `pass_fn(z)` returns `(outputs, z_next)`; see the module
    docstring. Under `no_grad`, or when nothing in the pass requires grad, the plain outputs
    come back. Raises by name from `backward` when the adjoint GMRES does not converge."""
    z = [t.detach().requires_grad_(torch.is_grad_enabled()) for t in z_star]
    outputs, z_next = pass_fn(z)
    outputs, z_next = list(outputs), list(z_next)
    if len(z_next) != len(z):
        raise ValueError(
            f"{where}: pass_fn returned {len(z_next)} next-iterate tensors for "
            f"{len(z)} interface tensors"
        )
    if not torch.is_grad_enabled() or not any(o.requires_grad for o in outputs):
        return [o.detach() if o.requires_grad else o for o in outputs]
    wrapped = _FixedPointAdjoint.apply(len(z), rtol, atol, max_iter, where, *z, *z_next, *outputs)
    return list(wrapped)
