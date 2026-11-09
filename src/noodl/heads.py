"""Heads: the additive potential terms (stack, wind, drag) written once per solve.

A head is a term added to an edge's potential difference: the buoyancy of a warm column of
air, the wind pressure on an opening, the drag a flow exerts on the air around it. It is a
function of the drivers only, never of the potential, so the Newton Jacobian stays exact and
symmetric (see `noodl.drives`). `EdgeHeads` is a `Model` closure that adds up the head terms
of one potential layer and writes the total as the driver `"<layer>.edge_head"`, one value per
edge in the layer's edge order. The layer adds it to every edge's potential difference once,
so a buoyancy head is a driver, like wind speed, and it can carry a gradient to whatever
parameter its closure reads.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

Tensor = torch.Tensor


@runtime_checkable
class HeadTerm(Protocol):
    """One additive head on the edges of one kind, computed from the drivers only.

    `kind` names the edge kind (one law of the layer); calling the term with the drivers
    returns one value per edge of that kind, `(..., b_kind)`, in the potential's unit. An
    optional `inputs` tuple lists the driver keys it reads. A term that holds tensors needs its
    own `to(device=None, dtype=None)` method for `Model.to` to move them (the terms defined in
    `noodl.drives` are moved without one).
    """

    kind: str

    def __call__(self, drivers: Mapping[str, Tensor]) -> Tensor: ...


def _check_term_signature(term, *, where: str) -> None:
    """Raise `TypeError` unless `term` is callable as `term(drivers)`.

    A loud guard against the older `(phi, drivers)` form, which would otherwise fail deep
    inside the closure with an unattributed `TypeError`.
    """
    try:
        sig = inspect.signature(term.__call__)
    except (TypeError, ValueError):  # builtins without a signature: nothing to check
        return
    positional = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        raise TypeError(
            f"{where}: head term {type(term).__name__} (kind {getattr(term, 'kind', '?')!r}) "
            f"must be callable as term(drivers) -- a head is a function of the drivers "
            f"mapping only; got parameters {positional}. A term that used to take "
            f"(phi, drivers) must drop phi."
        )


def _declared_inputs(term) -> tuple[str, ...] | None:
    """The driver keys `term` declares in its `inputs`, or None when it declares none."""
    try:
        return tuple(term.inputs)
    except AttributeError:
        return None


class EdgeHeads:
    """Adds up head terms (stack, wind, drag, a head read from a driver) on the edges of one
    potential layer and writes the total, in the layer's edge order, as the driver
    `"<layer>.edge_head"`; the layer adds it to every edge's potential difference.

    Terms are added in the order given, each into the columns of its own kind; an edge with
    no term gets exactly zero. A term is refused at construction if its kind is not one of
    the layer's kinds or if the law of that kind is dynamic (a capacitance or an inductance
    carries no head). The closure keeps the layer's name, kind slices and edge count, not the
    layer itself.
    """

    def __init__(self, layer, terms: Sequence[HeadTerm]) -> None:
        self.layer_name = layer.name
        self.slices = dict(layer._kind_slices)
        self.width = len(layer.cols)
        dynamic = {el.kind for el in layer._laws if el.dynamic}
        self.terms = list(terms)
        if not self.terms:
            raise ValueError(
                f"EdgeHeads for layer {self.layer_name!r}: no head terms were given"
            )
        for term in self.terms:
            _check_term_signature(term, where=f"EdgeHeads for layer {self.layer_name!r}")
            if term.kind not in self.slices:
                raise ValueError(
                    f"head term kind {term.kind!r} is not one of this layer's law kinds "
                    f"{sorted(self.slices)} (layer {self.layer_name!r})"
                )
            if any(d.kind == term.kind for d in layer._drives):
                raise ValueError(
                    f"EdgeHeads for layer {self.layer_name!r}: kind {term.kind!r} already has "
                    f"a drive in the layer's drives=; give each edge kind its head through "
                    f"drives= or through EdgeHeads, not both (the two would add up)"
                )
            if term.kind in dynamic:
                raise ValueError(
                    f"EdgeHeads for layer {self.layer_name!r}: kind {term.kind!r} has a "
                    f"dynamic law, which may not carry a head (its step-start values are "
                    f"the potential differences without one)"
                )
        self.outputs: tuple[str, ...] = (f"{self.layer_name}.edge_head",)
        keys = [_declared_inputs(t) for t in self.terms]
        self.inputs: tuple[str, ...] | None = (
            None if any(k is None for k in keys)
            else tuple(sorted({key for k in keys for key in k}))
        )

    def head(self, drivers: Mapping[str, Tensor]) -> Tensor:
        """The total head on every edge of the layer, `(..., b_layer)`: zeros, plus each
        term in its kind's columns. Refuses a term that returns the wrong width."""
        total = None
        for term in self.terms:
            start, end = self.slices[term.kind]
            value = term(drivers)
            if value.ndim > 0 and value.shape[-1] != end - start:
                raise ValueError(
                    f"head term kind {term.kind!r} returned width {value.shape[-1]} but "
                    f"the layer's {term.kind!r} block has {end - start} edges "
                    f"(layer {self.layer_name!r})"
                )
            if total is None:
                batch = value.shape[:-1] if value.ndim > 0 else ()
                total = value.new_zeros(batch + (self.width,))
            else:
                batch = torch.broadcast_shapes(total.shape[:-1], value.shape[:-1])
                if batch != total.shape[:-1]:
                    total = total.expand(batch + (self.width,))
            block = total[..., start:end] + value
            total = torch.cat([total[..., :start], block, total[..., end:]], dim=-1)
        return total

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        return {self.outputs[0]: self.head(drivers)}
