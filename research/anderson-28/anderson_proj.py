"""Anderson type-II mixing for the iterated coupling of `noodl.model.Model`.

`_anderson_mix` is the per-instance least-squares mixing step; `_AndersonHistory` is the pass
history `Model._iterate` keeps for one step (``iterate_acceleration="anderson"``). Moved here
from `noodl.model`, which re-imports both names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from noodl._broadcast import broadcast_shapes

if TYPE_CHECKING:
    from noodl.model import Model, State

Tensor = torch.Tensor

#: Largest ratio between the leading diagonal entries of the Anderson least-squares `R`
#: (newest column first) that `_anderson_mix` keeps; a column below it is treated as
#: linearly dependent on the newer ones and dropped together with every older one.
_ANDERSON_COND_MAX = 1.0e8

#: Smallest ratio between a column's orthogonalised norm and its rounding-noise estimate
#: that `_anderson_mix` keeps (when given `ns`); a column below it carries no information
#: above rounding and is dropped together with every older one.
_ANDERSON_NOISE_MIN = 10.0


def _anderson_mix(xs: Sequence[Tensor], gs: Sequence[Tensor], rs: Sequence[Tensor],
                  ns: Sequence[Tensor] | None = None) -> tuple[Tensor, Tensor]:
    """Anderson type-II mixing (Walker & Ni 2011, undamped), per batch instance.

    `xs` are the fed states and `gs` the pass outputs, `(*batch, N)` each, oldest first and
    of equal length; `rs` are the residuals the least-squares problem minimises, `(*batch,
    M)` each -- `g - x` on whatever entries are judged, scaled to be dimensionless (the
    caller divides by the tolerance). With the newest output `g`, residual `r` and the
    differences `dG`, `dR` of consecutive history entries, ``gamma = argmin |r - dR gamma|``
    and the mixed state is ``g - dG gamma``.

    The least-squares problem is solved by modified Gram-Schmidt on the columns of `dR`,
    NEWEST first. Per instance it keeps the longest run of leading columns whose
    orthogonalised norm stays above `1 / _ANDERSON_COND_MAX` of the largest one so far (and
    is finite and non-zero), so an ill-conditioned history loses its oldest columns rather
    than producing a wild step. With `ns` -- per history entry, `(*batch, M)`, a bound on the
    rounding error of each entry of `rs`, in the same scaled units -- a column is kept only
    while its orthogonalised norm is also at least `_ANDERSON_NOISE_MIN` times its own noise,
    the norm of ``ns[k] + ns[k - 1]`` of the two entries it differences. The relative rule
    alone does not see the absolute rounding floor: near convergence the leading column
    shrinks with the residual, and a column of nothing but rounding (say in directions the
    residual no longer has any signal in) passes it, gets a coefficient of order one and
    drags its large output difference into the step. Near convergence the noise rule drops
    such columns, and the mixing degrades to fewer columns, and in the end to none (the
    caller's relaxed step), instead of following the last ulp (``docs/design/
    orchestration.md``, "Anderson mixing", has the case and the choice of the multiple).

    At most `M` columns are used. Returns the mixed state and a batch-shaped bool, True
    where at least one column was kept; where none was, the mixed state is `g` itself and
    the caller decides what to feed instead.
    """
    g, r = gs[-1], rs[-1]
    batch = broadcast_shapes(*(t.shape[:-1] for t in (*xs, *gs, *rs)))
    m = min(len(xs) - 1, r.shape[-1])
    keep = torch.full(batch, m > 0, dtype=torch.bool, device=r.device)
    # Gram-Schmidt and back substitution are written out over the (at most `M`, in practice
    # a handful of) history columns: elementwise torch only. A LAPACK QR or triangular
    # solve here costs far more than its arithmetic: after one, the OpenMP worker threads
    # it wakes slowed every later small elementwise op of the pass about threefold
    # (measured on the sewer dynamic wave).
    q_cols: list[Tensor] = []
    r_cols: list[list[Tensor]] = []        # r_cols[j][i] = R[i, j], i < j
    diags: list[Tensor] = []
    keeps: list[Tensor] = []
    running_max = torch.zeros(batch, dtype=r.dtype, device=r.device)
    for j in range(m):
        v = rs[-1 - j] - rs[-2 - j]
        col: list[Tensor] = []
        for q_i in q_cols:
            r_ij = (q_i * v).sum(-1)
            col.append(r_ij)
            v = v - r_ij[..., None] * q_i
        r_cols.append(col)
        d = torch.sqrt((v * v).sum(-1))
        running_max = torch.maximum(running_max, d)
        keep = keep & torch.isfinite(d) & (d > 0) & (d * _ANDERSON_COND_MAX > running_max)
        if ns is not None:
            # the rounding of `v` along its own direction: at most sum |v_i| n_i / d
            noise = (v.abs() * (ns[-1 - j] + ns[-2 - j])).sum(-1)
            keep = keep & (d * d > _ANDERSON_NOISE_MIN * noise)
        safe = torch.where(keep, d, torch.ones_like(d))
        q_cols.append(torch.where(keep[..., None], v / safe[..., None], torch.zeros_like(v)))
        diags.append(safe)
        keeps.append(keep)
    gamma: list[Tensor] = [torch.zeros(batch, dtype=r.dtype, device=r.device)] * m
    for j in reversed(range(m)):
        acc = (q_cols[j] * r).sum(-1)
        for k in range(j + 1, m):
            acc = acc - r_cols[k][j] * gamma[k]
        gamma[j] = torch.where(keeps[j], acc / diags[j], torch.zeros_like(acc))
    mixed = g.expand(batch + g.shape[-1:])
    for j in range(m):
        mixed = mixed - gamma[j][..., None] * (gs[-1 - j] - gs[-2 - j])
    return mixed, (keeps[0] if m else keep)


class _AndersonHistory:
    """The pass history `Model._iterate` mixes with `_anderson_mix` (``iterate_acceleration=
    "anderson"``), for one step.

    The mixed state is ``"<layer>.phi"`` and ``"<layer>.q"`` of every Kirchhoff layer named
    in `iterate_tol`, stacked per instance. The least-squares residual is each pass's output
    minus its fed state: `phi` over the nodes convergence is judged on (interior and
    boundary) divided by the layer's `iterate_tol`, and `q` divided by its
    `iterate_flow_tol`, so that one dimensionless problem weighs every entry by its own
    tolerance. Both are needed: a head solve whose warm start already meets its residual
    tolerance hands the fed `phi` back unchanged, and the iteration then moves only through
    `q`. Each pass also records that residual's rounding noise, one ulp of the fed state and
    of the output per entry (``eps (|x| + |g|)``) scaled the same way, for `_anderson_mix`'s
    noise bound: near convergence the history columns of rounding alone are dropped, and the
    mixing falls back to fewer columns and in the end to the relaxed step.

    The next fed state is, per instance:

    * after the first pass, that pass's output (as the relaxed iteration's second pass);
    * the Anderson-mixed state, when it keeps at least one history column and moves no
      entry whose residual exceeds its tolerance AGAINST that residual;
    * otherwise the relaxed step ``x + w (g - x)``, `w` the model's `iterate_relaxation`.

    The second rule is the safeguard that keeps the fixed point the relaxed iteration's. A
    closure's equations may have several fixed points (the sewer's flat, near-dry conduit:
    one of its flows has two), and the relaxed iteration only reaches those it is
    attracted to. Near a repelling one the relaxed step points away from it and a secant
    step towards it, so an Anderson step that reverses an entry's own residual is the sign
    of exactly that; the relaxed step is taken instead. An instance already converged is
    held at the fed state of its certified pass, so the passes the rest of the batch still
    needs leave its result alone.
    """

    def __init__(self, model: Model, memory: int) -> None:
        self.layers = [
            (name, torch.cat((model.kirchhoff[name].interior, model.kirchhoff[name].bound)),
             float(tol), model.iterate_flow_tol[name])
            for name, tol in model.iterate_tol.items() if name in model.kirchhoff
        ]
        self.keys = [f"{name}.{s}" for name, *_ in self.layers for s in ("phi", "q")]
        self.memory = memory
        self.xs: list[Tensor] = []
        self.gs: list[Tensor] = []
        self.rs: list[Tensor] = []
        self.ns: list[Tensor] = []

    def mix(self, fed: Mapping[str, Tensor], out: Mapping[str, Tensor], w: float,
            converged: Tensor | None) -> State:
        """Record the pass that ran on `fed` and returned `out`; the next fed values of the
        mixed keys. `w` is the relaxation of the fallback step; `converged` (batch-shaped,
        or None before any verdict) marks the instances to hold at `fed`."""
        fed = {k: fed.get(k, out[k]) for k in self.keys}
        batch = broadcast_shapes(
            *(t.shape[:-1] for k in self.keys for t in (fed[k], out[k])))
        sizes = [out[k].shape[-1] for k in self.keys]

        def full(v: Tensor) -> Tensor:
            return v.expand(batch + v.shape[-1:])

        def judged(parts: Sequence[Tensor], scale: bool) -> Tensor:
            # (phi, q) per layer -> phi on the judged nodes and q, divided by the
            # tolerances when `scale`
            pieces = []
            for i, (_, nodes, tol, flow_tol) in enumerate(self.layers):
                pieces.append(full(parts[2 * i])[..., nodes] / (tol if scale else 1.0))
                pieces.append(full(parts[2 * i + 1]) / (flow_tol if scale else 1.0))
            return torch.cat(pieces, -1)

        x = torch.cat([full(fed[k]) for k in self.keys], -1)
        g = torch.cat([full(out[k]) for k in self.keys], -1)
        r = judged([out[k] - fed[k] for k in self.keys], scale=True)
        # the rounding error of `r`: one ulp of the fed state and of the output, per entry
        n = torch.finfo(r.dtype).eps * judged(
            [out[k].abs() + fed[k].abs() for k in self.keys], scale=True)
        for hist, v in ((self.xs, x), (self.gs, g), (self.rs, r), (self.ns, n)):
            hist.append(v)
            del hist[:-(self.memory + 1)]
        if len(self.xs) == 1:
            mixed = g
        else:
            accelerated, kept = _anderson_mix(self.xs, self.gs, self.rs, self.ns)
            step = judged(torch.split(accelerated - x, sizes, dim=-1), scale=False)
            reverses = ((step * r < 0) & (r.abs() > 1.0)).any(-1)
            mixed = torch.where((kept & ~reverses)[..., None], accelerated, x + w * (g - x))
        if converged is not None:
            mixed = torch.where(converged[..., None], x, mixed)
        return dict(zip(self.keys, torch.split(mixed, sizes, dim=-1), strict=True))
