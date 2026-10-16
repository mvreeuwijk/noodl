"""Batch axes, physical axes and species layout: the rules, written once.

Tensors in noodl carry leading batch axes `(B..., )` in front of their physical axes. A
per-node field is `(..., n)` or `(..., n, K)` for K species; a per-instance value is just
`(B...)`. `TransportLayer` accepts a single-species field either way, and these helpers
implement that one rule for everything outside the layer:

- STACKED `(..., n, K)` is tested first;
- for `K == 1` the REDUCED `(..., n)` is also accepted.

Each helper takes the tolerance an adopter's old private rule had (`min_dim`, `where`), so
adopting one never changes what is accepted.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from noodl._broadcast import broadcast_shapes

__all__ = [
    "batch_shape",
    "broadcast_batch",
    "per_instance",
    "is_stacked",
    "as_stacked",
    "as_layout",
    "reduced_single",
    "write_at",
    "species_columns",
]


def batch_shape(x: Tensor, n_trailing: int) -> torch.Size:
    """The batch shape of `x`: its shape without its last `n_trailing` physical dimensions.
    `ValueError` if `x` has fewer dimensions than that."""
    if x.dim() < n_trailing:
        raise ValueError(
            f"a tensor with {n_trailing} trailing physical dimension(s) needs at least that "
            f"many dimensions, got shape {tuple(x.shape)}"
        )
    return x.shape[: x.dim() - n_trailing]


def broadcast_batch(*items: tuple[Tensor, int]) -> tuple[torch.Size, list[Tensor]]:
    """Broadcast the batch shapes of `items`, each `(tensor, number of trailing physical
    dims)`. Returns the common batch shape and every tensor `.expand()`-ed to that batch plus
    its own trailing shape (views, no copies). The batch is aligned on the LEFT of the
    physical axes, never against them, so a batch of size `n` cannot be mistaken for a node
    axis. `RuntimeError` naming the shapes if the batches do not broadcast."""
    batches = [batch_shape(x, n) for x, n in items]
    try:
        batch = broadcast_shapes(*batches)
    except RuntimeError:
        raise RuntimeError(
            f"batch shapes do not broadcast: {[tuple(x.shape) for x, _ in items]} with "
            f"trailing physical dims {[n for _, n in items]} (batch shapes "
            f"{[tuple(b) for b in batches]})"
        ) from None
    out = [x.expand(*batch, *x.shape[x.dim() - n:]) for x, n in items]
    return batch, out


def per_instance(value, like: Tensor, n_trailing: int) -> Tensor:
    """A float, or a tensor with one value per instance `(B...)`, as `(B..., 1, ...)` with
    `n_trailing` trailing singleton axes, in `like`'s dtype and device. It then broadcasts
    against `like` `(B..., physical...)` over the batch only, never against a physical axis
    (a batch of size `n` is not a node axis). `RuntimeError` if its batch does not broadcast
    with `like`'s."""
    v = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    broadcast_shapes(v.shape, batch_shape(like, n_trailing))
    return v.reshape(*v.shape, *([1] * n_trailing))


def is_stacked(x: Tensor, n: int, K: int = 1, *, min_dim: int = 2) -> bool:
    """True when `x` is in STACKED layout `(..., n, K)`: at least `min_dim` dimensions
    (never fewer than 2) and trailing shape `(n, K)`. With `K == 1` this is the `(..., n, 1)`
    column, as opposed to the reduced `(..., n)`."""
    return x.dim() >= max(min_dim, 2) and x.shape[-1] == K and x.shape[-2] == n


def as_stacked(
    x: Tensor, n: int, K: int, *, name: str, where: str, min_dim: int = 2,
) -> tuple[Tensor, bool]:
    """`(x as (..., n, K), reduced?)` by the one rule: stacked `(..., n, K)` first; for
    `K == 1` also reduced `(..., n)`, returned with a trailing axis and `reduced=True`.
    `min_dim` is the fewest dimensions of the stacked form (the reduced form needs one fewer):
    3 where the leading axis is time. `ValueError("{where}: {name!r} must be ...")`
    otherwise."""
    if is_stacked(x, n, K, min_dim=min_dim):
        return x, False
    if K == 1 and x.dim() >= max(min_dim - 1, 1) and x.shape[-1] == n:
        return x.unsqueeze(-1), True
    also = f" or, for one species, (..., {n})" if K == 1 else ""
    raise ValueError(
        f"{where}: {name!r} must have trailing shape ({n}, {K}){also}, got {tuple(x.shape)}"
    )


def as_layout(x_stacked: Tensor, reduced: bool) -> Tensor:
    """`x_stacked` `(..., n, 1)` back in the caller's layout: `(..., n)` if `reduced`,
    unchanged otherwise (the inverse of `as_stacked` for one species)."""
    return x_stacked.squeeze(-1) if reduced else x_stacked


def reduced_single(x: Tensor, n: int, *, where: str) -> Tensor:
    """A single-species tensor in REDUCED layout `(..., n)`, whether it came stacked
    `(..., n, 1)` or reduced. A batched `(1, 1)` is both and holds one number, so the
    ambiguity at `n == 1` is harmless. `ValueError("{where}: ...")` if its last axis is not
    `n` entries."""
    if is_stacked(x, n):
        return x.squeeze(-1)
    if x.shape[-1] != n:
        raise ValueError(
            f"{where}: expected a tensor with {n} entries on its node axis, got shape "
            f"{tuple(x.shape)}"
        )
    return x


def write_at(
    target: Tensor, n: int, position: int, value: Tensor, *, add: bool = False,
    where: str = "layout.write_at",
) -> Tensor:
    """`target` with `value` written (`add=True`: ADDED) at node `position`, in the caller's
    own layout (reduced `(..., n)` or stacked `(..., n, 1)`) but not necessarily its own
    shape: a batched `value` written into an unbatched `target` broadcasts the target up to
    the value's batch. Never writes into `target` itself. `where` prefixes the layout error."""
    reduced = reduced_single(target, n, where=where)
    batch = broadcast_shapes(reduced.shape[:-1], value.shape)
    reduced = reduced.expand(*batch, n).clone()
    reduced[..., position] = reduced[..., position] + value if add else value
    if is_stacked(target, n):
        return reduced.reshape(*batch, n, 1)
    return reduced.reshape(*batch, n)


def species_columns(
    declared: Sequence[str] | None,
    wanted: Sequence[str],
    *,
    where: str,
    n_species: int | None = None,
    order: str = "any",
) -> list[int]:
    """The column of each `wanted` species name in the `declared` order (a layer's
    `species_names`). `order="any"` allows a subset in any order; `order="exact"` requires
    `wanted == declared`. Unknown and repeated names raise `ValueError`, naming them, with
    `where` as prefix. With `declared=None` (the layer names no species) the names cannot be
    resolved: the identity `[0, 1, ...]` is returned, only when `len(wanted) == n_species`."""
    if order not in ("any", "exact"):
        raise ValueError(f"{where}: order must be 'any' or 'exact', got {order!r}")
    wanted = list(wanted)
    repeated = sorted({w for w in wanted if wanted.count(w) > 1})
    if repeated:
        raise ValueError(f"{where}: species {repeated} are named more than once in {wanted}")
    if declared is None:
        if n_species is None or len(wanted) != n_species:
            raise ValueError(
                f"{where}: the layer declares no species_names, so {wanted} cannot be "
                f"matched to columns; they are accepted in order only for all "
                f"n_species={n_species} species"
            )
        return list(range(len(wanted)))
    declared = list(declared)
    if len(set(declared)) != len(declared):
        raise ValueError(f"{where}: declared species_names {declared} repeat a name")
    if order == "exact" and wanted != declared:
        raise ValueError(f"{where}: species must be exactly {declared} in that order, got {wanted}")
    unknown = [w for w in wanted if w not in declared]
    if unknown:
        raise ValueError(f"{where}: unknown species {unknown}; the layer declares {declared}")
    return [declared.index(w) for w in wanted]
