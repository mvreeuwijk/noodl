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

The `pass_fn` contract, in full, because two of its rules are traps:

* `pass_fn(z)` returns `(outputs, z_next)`. `outputs` is EVERY tensor the caller will hand on
  that depends on `z`, flattened in a fixed order; `z_next` is the next iterate as read from
  those outputs, the same length and the same SHAPES as `z`. An entry of `z_next` may be an
  entry of `outputs` itself (the identity read), a slice or reshape of one, or any other
  function of them.
* **Every entry of `z_next` must be RECOMPUTED by the pass.** Handing back the leaf `z[j]`
  unchanged -- the obvious way to express "this interface entry is prescribed, the pass never
  updates it" -- puts a unit row in `G_z`, which makes `I - G_z` singular and the adjoint
  solve fail with the non-convergence error below. A prescribed entry is not an unknown of
  the interface equations at all: `z_j = z_j` determines nothing. Keep it OUT of `z_star`, or
  hand it back detached from `z` (the original graph-free tensor), which leaves a zero row
  where the identity would have been and is exactly right.
* The returned outputs are VIEWS of the pass's own tensors (an identity `autograd.Function`).
  A caller reassembling state dicts from them must not write into them in place: that would
  mutate the pass's graph under autograd and trip its version counter at backward time, or
  silently alias a tensor the caller still owns. Copy first if a buffer must be written.

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
    def forward(ctx, n_z, rtol, atol, max_iter, restart, where, *tensors):
        ctx.n_z, ctx.rtol, ctx.atol = n_z, rtol, atol
        ctx.max_iter, ctx.restart, ctx.where = max_iter, restart, where
        ctx.z = tensors[:n_z]
        ctx.z_next = tensors[n_z : 2 * n_z]
        ctx.outputs = tensors[2 * n_z :]
        # Identity on the outputs: a view of each, so autograd routes their cotangents
        # through `backward` below and then on into the pass's own graph.
        wrapped = tuple(o.view_as(o) for o in ctx.outputs)
        # A pass may hand on a tensor nothing differentiable reached. Without this, the
        # Function's own output would come back `requires_grad=True` with a `grad_fn`,
        # because ANY input requiring grad makes EVERY output of a Function require it --
        # and this repo branches on `requires_grad`: `solvers/select.py` drops the SuperLU
        # fast path for an input that requires grad and turns an explicit
        # `method="sparse_direct"` into a RuntimeError. A boundary constant carried through a
        # Model pass would silently cost the fast path, or crash, for no real dependence.
        dead = [w for w, o in zip(wrapped, ctx.outputs, strict=True) if not o.requires_grad]
        if dead:
            ctx.mark_non_differentiable(*dead)
        return wrapped

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
            restart=min(m, 100) if ctx.restart is None else ctx.restart,
        )
        if not bool(result.converged.all()):
            raise RuntimeError(
                f"{ctx.where}: the fixed-point adjoint solve did not converge (residual "
                f"{float(result.residual.max()):.3e} after {int(result.iterations.max())} "
                f"GMRES iterations, rtol={ctx.rtol}, atol={ctx.atol}); I - G_z is singular or "
                f"badly conditioned at this point -- an interface entry the pass hands back "
                f"unchanged puts a unit row in G_z and does exactly that -- or max_iter is "
                f"too small. Note that a contraction is NOT required: implicit "
                f"differentiation needs only a nonsingular I - G_z"
            )
        # g on the `outputs` inputs and v on the `z_next` inputs: autograd's own backward
        # pass then accumulates g^T S_theta + v^T G_theta through the one retained graph.
        # See the module docstring for why v is NOT projected onto the outputs here.
        v = _unflatten(result.x, z)
        return (None, None, None, None, None, None, *([None] * ctx.n_z), *v, *g)


def differentiate_fixed_point(
    z_star: Sequence[Tensor],
    pass_fn: Callable[[list[Tensor]], tuple[list[Tensor], list[Tensor]]],
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    restart: int | None = None,
    where: str = "fixed point",
) -> list[Tensor]:
    """One differentiable pass at the converged interface `z_star`, with the implicit adjoint
    attached to its outputs. `pass_fn(z)` returns `(outputs, z_next)`; the module docstring
    has the contract in full, including the two rules that bite: every entry of `z_next` must
    be RECOMPUTED (handing the leaf `z[j]` straight back for a prescribed entry makes
    `I - G_z` singular), and the returned outputs are VIEWS that must not be written into in
    place.

    `rtol`, `atol`, `max_iter` and `restart` go to `solvers.iterative.gmres` for the adjoint
    solve; `restart` defaults to `min(m, 100)` on the flattened interface of size `m`. Each
    GMRES matvec is one full VJP through the pass graph, so a large interface may want a
    smaller `restart` (less basis memory, more matvecs) or a larger `max_iter`.

    Under `no_grad`, when nothing in the pass requires grad, or when the interface is empty,
    the plain outputs come back. Raises `ValueError` here for a `z_next` that does not match
    `z` in length or shape and for an out-of-range `max_iter`/`restart`; raises `RuntimeError`
    by name from `backward` when the adjoint GMRES does not converge.
    """
    if max_iter is not None and max_iter < 1:
        raise ValueError(f"{where}: max_iter must be >= 1 when given, got {max_iter!r}")
    if restart is not None and restart < 1:
        raise ValueError(f"{where}: restart must be >= 1 when given, got {restart!r}")
    z = [t.detach().requires_grad_(torch.is_grad_enabled()) for t in z_star]
    outputs, z_next = pass_fn(z)
    outputs, z_next = list(outputs), list(z_next)
    if len(z_next) != len(z):
        raise ValueError(
            f"{where}: pass_fn returned {len(z_next)} next-iterate tensors for "
            f"{len(z)} interface tensors"
        )
    for i, (t, n) in enumerate(zip(z, z_next, strict=True)):
        if n.shape != t.shape:
            # Checked here, where `where` and the entry index are in hand: a same-numel
            # mismatch would otherwise surface from inside `backward` as a bare autograd
            # shape error with nothing to locate it by.
            raise ValueError(
                f"{where}: pass_fn returned next-iterate tensor {i} of shape "
                f"{tuple(n.shape)} for an interface tensor of shape {tuple(t.shape)}"
            )
    # An empty interface has no implicit term at all -- d new/d theta is just S_theta, which
    # the pass graph already carries. Short-circuit, or `backward` would die in `torch.cat`
    # on an empty list with no `where` to locate it by.
    if not z:
        return list(outputs)
    if not torch.is_grad_enabled() or not any(o.requires_grad for o in outputs):
        return [o.detach() if o.requires_grad else o for o in outputs]
    wrapped = _FixedPointAdjoint.apply(
        len(z), rtol, atol, max_iter, restart, where, *z, *z_next, *outputs
    )
    return list(wrapped)
