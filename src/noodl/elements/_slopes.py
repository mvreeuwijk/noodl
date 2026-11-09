"""Local derivative blocks of a constitutive law, by reverse-mode colouring.

A law of `ports` m whose edges form G groups has, per group, an m x m block of derivatives
of its outputs with respect to each input. `_colour_blocks` obtains all G blocks from ONE
evaluation of the law and m reverse-mode derivatives (`torch.autograd.grad`), cotangent k
seeding slot k of every group at once (the groups are independent, so each gradient holds
row k of every group's block). A phasor law (complex potentials) is differentiated through
the real views of its input and output, so each entry becomes the real 2x2 block of the
(possibly non-holomorphic) law. This is the default of `ConstitutiveLaw.jacobian`.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _complex(z: Tensor, cdtype: torch.dtype = torch.complex128) -> Tensor:
    """`z` as a `cdtype` tensor (the layer's complex dtype: complex128 on a float64
    network, complex64 on a float32 one) that `view_as_real` accepts: a phasor-capable law
    may return a real tensor (a constant current) or a lazily conjugated one (a law written
    with `torch.conj`). Both conversions are differentiable."""
    return z.to(cdtype).resolve_conj()


def _colour_blocks(fn, zs: tuple[Tensor, ...], m: int, w: int,
                   cdtype: torch.dtype = torch.complex128) -> list[Tensor]:
    """`(..., G, m, m, w, w)` per input: `d fn(*zs)[g, k, ck] / d zs[i][g, l, cl]` for a
    groupwise `fn`, from one evaluation of `fn` and `m * w` reverse-mode gradients.

    Each input is a fresh leaf (in its real view when `w == 2`), so the gradients are with
    respect to it alone; the values carry no graph (the operator is only ever used under
    `no_grad`). An output that does not depend on an input gives a zero block. An output
    whose batch shape is wider than an input's (a batched driver) is evaluated again with
    the inputs expanded to it, so the gradient is per instance, not summed over the batch.
    """
    with torch.enable_grad():
        leaves, out = _evaluate(fn, zs, w, cdtype)
        batch = _batch(out, w)
        if any(_batch(leaf, w) != batch for leaf in leaves):
            zs = tuple(z.expand(batch + z.shape[-1:]) for z in zs)
            leaves, out = _evaluate(fn, zs, w, cdtype)
        live = list(range(len(leaves))) if out.requires_grad else []
        G = (out.shape[-2] if w == 2 else out.shape[-1]) // m
        rows: list[list[Tensor]] = [[] for _ in leaves]
        n_seeds = m * w
        for seed in range(n_seeds):
            k, ck = divmod(seed, w)
            cot = torch.zeros(batch + (G, m, w), dtype=out.dtype, device=out.device)
            cot[..., k, ck] = 1.0
            cot = cot.reshape(out.shape)
            grads = (torch.autograd.grad(out, [leaves[i] for i in live], cot,
                                         retain_graph=seed < n_seeds - 1, allow_unused=True)
                     if live else ())
            by_input = dict(zip(live, grads, strict=True))
            for i, leaf in enumerate(leaves):
                g = by_input.get(i)
                g = torch.zeros_like(leaf) if g is None else g
                rows[i].append(g.reshape(batch + (G, m, w)))   # [g, l, cl] for this (k, ck)
    # (..., G, k*w + ck, l, cl) -> (..., G, k, l, ck, cl)
    return [torch.stack(r, dim=-3).unflatten(-3, (m, w)).transpose(-3, -2).detach()
            for r in rows]


def _batch(x: Tensor, w: int) -> tuple[int, ...]:
    """The batch shape of a value per edge (in its real view when `w == 2`)."""
    return tuple(x.shape[:-2] if w == 2 else x.shape[:-1])


def _evaluate(fn, zs: tuple[Tensor, ...], w: int,
              cdtype: torch.dtype = torch.complex128) -> tuple[list[Tensor], Tensor]:
    """`fn` at fresh leaves for `zs` (real views when `w == 2`), and its output (real view
    when `w == 2`)."""
    if w == 2:
        leaves = [torch.view_as_real(_complex(z, cdtype).detach()).clone().requires_grad_(True)
                  for z in zs]
        out = fn(*(torch.view_as_complex(leaf) for leaf in leaves))
        return leaves, torch.view_as_real(_complex(out, cdtype))
    leaves = [z.detach().clone().requires_grad_(True) for z in zs]
    return leaves, fn(*leaves)


def complex_slope_block(c: Tensor) -> Tensor:
    """The real 2x2 block `[[Re c, -Im c], [Im c, Re c]]` of a holomorphic slope `c`, shape
    `c.shape + (2, 2)`: how a phasor law that knows its complex derivative `c` writes it as a
    `jacobian` block entry (the real view of multiplication by `c`)."""
    re, im = c.real, c.imag
    return torch.stack([torch.stack([re, -im], dim=-1), torch.stack([im, re], dim=-1)], dim=-2)

