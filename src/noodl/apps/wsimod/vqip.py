"""VQIP algebra and decision primitives as float64 tensors.

A VQIP is a tensor ``(..., 1 + K)``: column 0 is the volume, columns ``1..K`` the pollutants
in the order of :class:`Pollutants`. Each operation transcribes WSIMOD 0.8.1's ``WSIObj``
(``wsimod/core/core.py``) in the same operation order, so results are bitwise equal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class Pollutants:
    """The pollutant columns of a VQIP and which of them blend or add."""

    names: tuple[str, ...]
    additive: tuple[str, ...]
    non_additive: tuple[str, ...]
    K: int = field(init=False)
    additive_mask: Tensor = field(init=False, compare=False, repr=False)
    non_additive_mask: Tensor = field(init=False, compare=False, repr=False)
    volume_or_additive_mask: Tensor = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "additive", tuple(self.additive))
        object.__setattr__(self, "non_additive", tuple(self.non_additive))
        if len(set(names)) != len(names):
            dup = next(n for n in names if names.count(n) > 1)
            raise ValueError(f"pollutant {dup!r} is listed twice in names")
        for name in self.additive + self.non_additive:
            if name not in names:
                raise ValueError(f"pollutant {name!r} is in an additive list but not in names")
        for name in names:
            a, n = name in self.additive, name in self.non_additive
            if a and n:
                raise ValueError(f"pollutant {name!r} is both additive and non-additive")
            if not a and not n:
                raise ValueError(f"pollutant {name!r} is neither additive nor non-additive")
        object.__setattr__(self, "K", len(names))
        add = torch.tensor([False] + [n in self.additive for n in names], dtype=torch.bool)
        non = torch.tensor([False] + [n in self.non_additive for n in names], dtype=torch.bool)
        object.__setattr__(self, "additive_mask", add)
        object.__setattr__(self, "non_additive_mask", non)
        object.__setattr__(
            self, "volume_or_additive_mask", add | torch.tensor([True] + [False] * len(names))
        )

    def column(self, name: str) -> int:
        """Column of pollutant ``name`` (1-based; column 0 is the volume)."""
        if name not in self.names:
            raise KeyError(f"unknown pollutant {name!r}")
        return self.names.index(name) + 1

    def to_vqip(self, values: Mapping[str, float], shape=()) -> Tensor:
        """A VQIP from ``{"volume": ..., name: ...}``; absent entries are 0."""
        unknown = set(values) - set(self.names) - {"volume"}
        if unknown:
            raise KeyError(f"unknown pollutant {sorted(unknown)[0]!r}")
        row = torch.zeros(1 + self.K, dtype=torch.float64)
        row[0] = values.get("volume", 0.0)
        for i, n in enumerate(self.names):
            row[1 + i] = values.get(n, 0.0)
        return row.expand(*tuple(shape), 1 + self.K).clone()

    def to_dict(self, vqip: Tensor) -> dict[str, float]:
        """One instance as ``{"volume": ..., name: ...}``."""
        out = {"volume": float(vqip[0])}
        for i, n in enumerate(self.names):
            out[n] = float(vqip[1 + i])
        return out


QUANTITY_ONLY = Pollutants((), (), ())


def empty(p: Pollutants, shape) -> Tensor:
    """All-zero VQIP of batch shape ``shape`` (core.py:34-45)."""
    return torch.zeros(*tuple(shape), 1 + p.K, dtype=torch.float64)


def safe_den(den: Tensor) -> Tensor:
    """``where(den != 0, den, 1)``: a denominator that never divides by zero."""
    return torch.where(den != 0, den, torch.ones_like(den))


def tmin(a: Tensor, b: Tensor) -> Tensor:
    """Python's ``min(a, b)``: ``a`` unless ``b < a``; detached comparison, gradient to the pick."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return torch.where((b < a).detach(), b, a)


def tmax(a: Tensor, b: Tensor) -> Tensor:
    """Python's ``max(a, b)``: ``a`` unless ``b > a``; detached comparison, gradient to the pick."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return torch.where((b > a).detach(), b, a)


def write(active: Tensor, new: Tensor, old: Tensor) -> Tensor:
    """``where(active, new, old)``; ``active`` ``(...)`` broadcasts over a VQIP's column axis.

    The column axis is added when either value has more dimensions than the mask, so an
    unbatched ``new`` written into a batched VQIP (or the reverse) is aligned by column, never
    by batch (a quantity-only VQIP has one column, which would otherwise broadcast against a
    batch of the same length)."""
    new, old = torch.as_tensor(new), torch.as_tensor(old)
    if max(new.ndim, old.ndim) > active.ndim:
        active = active[..., None]
    return torch.where(active, new, old)


def sum_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:83-116. Volume and additive columns add; non-additive columns blend
    ``(t2·V2 + t1·V1)/V`` when ``V > 0``, otherwise keep ``t1``."""
    vol = t1[..., :1] + t2[..., :1]
    summed = t1 + t2
    blended = (t2 * t2[..., :1] + t1 * t1[..., :1]) / safe_den(vol)
    mixed = torch.where(p.non_additive_mask & (vol > 0), blended, t1)
    return torch.where(p.volume_or_additive_mask, summed, mixed)


def extract_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:150-178. Volume and additive columns subtract; non-additive keep ``t1``."""
    return torch.where(p.volume_or_additive_mask, t1 - t2, t1)


def _broadcast(t: Tensor, v) -> tuple[Tensor, Tensor]:
    """A VQIP ``(..., 1 + K)`` and a volume ``(...)`` brought to one batch shape."""
    v = torch.as_tensor(v, dtype=torch.float64)
    shape = torch.broadcast_shapes(t.shape[:-1], v.shape)
    return t.expand(shape + t.shape[-1:]), v.expand(shape)


def v_distill_vqip(t: Tensor, v: Tensor) -> Tensor:
    """core.py:209-223. Remove volume ``v``; pollutants unchanged."""
    t, v = _broadcast(t, v)
    return torch.cat([t[..., :1] - v[..., None], t[..., 1:]], dim=-1)


def v_change_vqip(p: Pollutants, t: Tensor, v: Tensor) -> Tensor:
    """core.py:251-284. ``V > 0``: ``ratio = v/V``, volume ``V·ratio``, additive columns
    times ``ratio``; otherwise volume ``v`` and pollutants unchanged."""
    t, v = _broadcast(t, v)
    v = v[..., None]
    pos = t[..., :1] > 0
    ratio = v / safe_den(t[..., :1])
    scaled = torch.where(p.volume_or_additive_mask, t * ratio, t)
    return torch.where(pos, scaled, torch.cat([v, t[..., 1:]], dim=-1))


def ds_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:305-328. Volume and additive differences; non-additive columns 0."""
    return torch.where(p.volume_or_additive_mask, t1 - t2, torch.zeros_like(t1))


class _PowPython(torch.autograd.Function):
    """``x ** y`` as Python evaluates it for floats (the C library's ``pow``), elementwise.

    IHACRES raises the moisture ratio to ``ihacres_p`` (``land.py:950``) and a treatment works
    raises its ``exponent`` to ``20 - T`` (``wtw.py:191-193``); ``torch.pow`` can
    differ from ``float.__pow__`` in the last bit, so the forward pass calls Python's, once per
    instance (the :func:`~noodl.apps.wsimod.storage.pow_half` pattern). An ``int`` exponent in
    WSIMOD (the default ``10``) is converted to a double by ``float.__pow__``, so the float
    exponent here is the same call. The gradients are ``y x^(y-1)`` and ``x^y ln x`` where
    ``x > 0``, and 0 where ``x == 0`` (a saturated soil; the slope there is documented, not
    propagated).
    """

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        xb, yb = torch.broadcast_tensors(x.detach(), y.detach())
        pairs = zip(xb.reshape(-1).tolist(), yb.reshape(-1).tolist(), strict=True)
        out = torch.tensor([a**b for a, b in pairs], dtype=torch.float64).reshape(xb.shape)
        ctx.save_for_backward(x, y, out)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: Tensor) -> tuple[Tensor, Tensor]:
        x, y, out = ctx.saved_tensors
        positive = x > 0
        safe = torch.where(positive, x, torch.ones_like(x))
        zero = torch.zeros_like(grad)
        gx = torch.where(positive, grad * y * out / safe, zero)
        gy = torch.where(positive, grad * out * torch.log(safe), zero)
        return gx.sum_to_size(x.shape), gy.sum_to_size(y.shape)


def pow_python(x: Tensor, y: Tensor) -> Tensor:
    """``x ** y`` bitwise as Python computes it for floats; gradient 0 where ``x == 0``."""
    return _PowPython.apply(
        torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    )
