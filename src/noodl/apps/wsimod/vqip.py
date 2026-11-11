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


@dataclass(frozen=True)
class Pollutants:
    """The pollutant columns of a VQIP and which of them blend or add."""

    names: tuple[str, ...]
    additive: tuple[str, ...]
    non_additive: tuple[str, ...]
    K: int = field(init=False)
    additive_mask: Tensor = field(init=False, compare=False, repr=False)
    non_additive_mask: Tensor = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(self.names)
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


def _vol_or_add(p: Pollutants) -> Tensor:
    m = p.additive_mask.clone()
    m[0] = True
    return m


def safe_den(den: Tensor) -> Tensor:
    """``where(den != 0, den, 1)``: a denominator that never divides by zero."""
    return torch.where(den != 0, den, torch.ones_like(den))


def safe_sqrt(x: Tensor) -> Tensor:
    """sqrt(x) with gradient 0 (not infinity) where ``x == 0``."""
    zero = x == 0
    root = torch.sqrt(torch.where(zero, torch.ones_like(x), x))
    return torch.where(zero, torch.zeros_like(x), root)


def tmin(a: Tensor, b: Tensor) -> Tensor:
    """Python's ``min(a, b)``: ``a`` unless ``b < a``; detached comparison, gradient to the pick."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return torch.where((b < a).detach(), b, a)


def tmax(a: Tensor, b: Tensor) -> Tensor:
    """Python's ``max(a, b)``: ``a`` unless ``b > a``; detached comparison, gradient to the pick."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return torch.where((b > a).detach(), b, a)


def write(active: Tensor, new: Tensor, old: Tensor) -> Tensor:
    """``where(active, new, old)``; ``active`` ``(...)`` broadcasts over a VQIP's column axis."""
    if new.ndim > active.ndim:
        active = active[..., None]
    return torch.where(active, new, old)


def sum_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:83-116. Volume and additive columns add; non-additive columns blend
    ``(t2·V2 + t1·V1)/V`` when ``V > 0``, otherwise keep ``t1``."""
    vol = t1[..., :1] + t2[..., :1]
    summed = t1 + t2
    blended = (t2 * t2[..., :1] + t1 * t1[..., :1]) / safe_den(vol)
    mixed = torch.where(p.non_additive_mask & (vol > 0), blended, t1)
    return torch.where(_vol_or_add(p), summed, mixed)


def extract_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:150-178. Volume and additive columns subtract; non-additive keep ``t1``."""
    return torch.where(_vol_or_add(p), t1 - t2, t1)


def v_distill_vqip(t: Tensor, v: Tensor) -> Tensor:
    """core.py:209-223. Remove volume ``v``; pollutants unchanged."""
    v = torch.as_tensor(v, dtype=torch.float64)
    return torch.cat([t[..., :1] - v[..., None], t[..., 1:]], dim=-1)


def v_change_vqip(p: Pollutants, t: Tensor, v: Tensor) -> Tensor:
    """core.py:251-284. ``V > 0``: ``ratio = v/V``, volume ``V·ratio``, additive columns
    times ``ratio``; otherwise volume ``v`` and pollutants unchanged."""
    v = torch.as_tensor(v, dtype=torch.float64)[..., None]
    pos = t[..., :1] > 0
    ratio = v / safe_den(t[..., :1])
    scaled = torch.where(_vol_or_add(p), t * ratio, t)
    return torch.where(pos, scaled, torch.cat([v, t[..., 1:]], dim=-1))


def ds_vqip(p: Pollutants, t1: Tensor, t2: Tensor) -> Tensor:
    """core.py:305-328. Volume and additive differences; non-additive columns 0."""
    return torch.where(_vol_or_add(p), t1 - t2, torch.zeros_like(t1))
