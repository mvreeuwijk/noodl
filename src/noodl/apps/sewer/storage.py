"""SWMM 5.2.4 storage units: the surface area and stored volume of a `[STORAGE]` node.

A storage unit's curve (`Manhole.storage`) is one of, in SI user units (depth y in m, area
in m2, lengths in m):

- ``("functional", a, b, c)``: ``A = a y^b + c`` (SWMM's ``FUNCTIONAL a1 a2 a0``, so
  ``a = a1``, ``b = a2``, ``c = a0``; `node.c:1001-1003`), with ``b >= 0`` and ``c >= 0``;
- ``("tabular", depths, areas)``: SWMM's STORAGE curve, depths strictly increasing
  (`table.c:285-329`); the area is `table_lookupEx` (linear from 0 below the first depth,
  linear between depths, the last segment's slope, floored at 0, beyond the last), the
  volume the end-area method of `table_getStorageVolume` (`table.c:587-647`);
- ``("cylindrical", L, W)``, ``("conical", L, W, Z)``, ``("paraboloid", L, W, H)`` and
  ``("pyramidal", L, W, Z)``: the elliptic cylinder of axes L x W, the elliptic cone
  frustum on an L x W base with side slope Z (run over rise), the paraboloid with an L x W
  top at height H and the rectangular frustum on an L x W base with side slope Z, which
  SWMM turns into the quadratic ``A = a0 + a1 y + a2 y^2`` (`node.c:750-783`).

The volume is SWMM's `storage_getVolume` (`node.c:930-975`): 0 at zero depth, the integral
of the area up to the depth, and the full volume (the volume at the full depth) at or above
the full depth, so the surcharge depth above it stores nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

Tensor = torch.Tensor
F64 = torch.float64

#: Storage curve shapes (SWMM's `RelationWords`, keywords.c:108, in lower case).
STORAGE_SHAPES = ("functional", "tabular", "cylindrical", "conical", "paraboloid", "pyramidal")
_NONE, _FUNCTIONAL, _QUADRATIC, _TABULAR = 0, 1, 2, 3
#: SWMM's thresholds in table.c (user units, m): a first depth below `_X_TINY` has no
#: triangle under it (table.c:608), a last segment of `_X_TINY` or less is not extrapolated
#: (table.c:634).
_X_TINY = 1.0e-6


def validate_storage_curve(owner: str, curve) -> None:
    """Refuse a malformed storage curve of junction `owner`, BY NAME (shapes in the module
    docstring; SWMM's own checks, node.c:724-738 and table.c:285-329)."""
    if not isinstance(curve, tuple) or not curve or not isinstance(curve[0], str):
        raise ValueError(
            f"storage unit {owner!r}: the curve must be a tuple (shape, ...), got {curve!r}")
    shape, args = curve[0], curve[1:]
    arity = {"functional": 3, "tabular": 2, "cylindrical": 2, "conical": 3,
             "paraboloid": 3, "pyramidal": 3}
    if shape not in arity:
        raise ValueError(
            f"storage unit {owner!r}: the curve shape must be one of {STORAGE_SHAPES}, got "
            f"{shape!r}")
    if len(args) != arity[shape]:
        raise ValueError(
            f"storage unit {owner!r}: a {shape} curve takes {arity[shape]} values, got "
            f"{len(args)}")
    if shape == "tabular":
        depths, areas = (tuple(float(v) for v in a) for a in args)
        if not depths:
            raise ValueError(f"storage unit {owner!r}: a tabular curve needs at least one "
                             f"(depth, area) point")
        if len(depths) != len(areas):
            raise ValueError(f"storage unit {owner!r}: a tabular curve needs as many areas "
                             f"as depths, got {len(depths)} and {len(areas)}")
        if not all(math.isfinite(v) for v in depths + areas):
            raise ValueError(f"storage unit {owner!r}: a tabular curve must be finite")
        if any(b <= a for a, b in zip(depths, depths[1:], strict=False)):
            raise ValueError(f"storage unit {owner!r}: tabular curve depths must be strictly "
                             f"increasing (SWMM's ERR_CURVE_SEQUENCE), got {depths}")
        return
    values = [float(v) for v in args]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"storage unit {owner!r}: {shape} curve values must be finite, got "
                         f"{values}")
    if shape == "functional":
        if values[2] < 0.0:                                                   # node.c:724
            raise ValueError(f"storage unit {owner!r}: functional c (area at zero depth, "
                             f"m2) must be >= 0, got {values[2]}")
        if values[1] < 0.0:
            raise ValueError(f"storage unit {owner!r}: functional exponent b must be >= 0 "
                             f"(a negative one gives an infinite area at zero depth), got "
                             f"{values[1]}")
        return
    if values[0] <= 0.0 or values[1] <= 0.0:                                  # node.c:732-733
        raise ValueError(f"storage unit {owner!r}: {shape} length and width must be > 0 m, "
                         f"got {values[0]}, {values[1]}")
    if shape in ("conical", "pyramidal") and values[2] < 0.0:                 # node.c:734
        raise ValueError(f"storage unit {owner!r}: {shape} side slope must be >= 0, got "
                         f"{values[2]}")
    if shape == "paraboloid" and values[2] <= 0.0:                            # node.c:738
        raise ValueError(f"storage unit {owner!r}: paraboloid height must be > 0 m, got "
                         f"{values[2]}")


def _coefficients(curve) -> tuple[int, float, float, float]:
    """(kind, a0, a1, a2) of a closed-form curve: SWMM's `Storage.a0/a1/a2` (node.c:742-783),
    FUNCTIONAL A = a0 + a1 y^a2, the others A = a0 + a1 y + a2 y^2."""
    shape, v = curve[0], [float(x) for x in curve[1:]]
    if shape == "functional":
        a, b, c = v
        return _FUNCTIONAL, c, a, b                                           # node.c:745-747
    if shape == "cylindrical":
        semi_l, semi_w = v[0] / 2.0, v[1] / 2.0
        return _QUADRATIC, math.pi * semi_l * semi_w, 0.0, 0.0                # node.c:750-756
    if shape == "conical":
        semi_l, semi_w, z = v[0] / 2.0, v[1] / 2.0, v[2]
        return (_QUADRATIC, math.pi * semi_l * semi_w, 2.0 * math.pi * semi_w * z,
                math.pi * semi_w / semi_l * z * z)                            # node.c:758-765
    if shape == "paraboloid":
        semi_l, semi_w, z = v[0] / 2.0, v[1] / 2.0, v[2]
        return _QUADRATIC, 0.0, math.pi * semi_l * semi_w / z, 0.0            # node.c:767-774
    length, width, z = v                                                      # pyramidal
    return _QUADRATIC, length * width, 2.0 * (length + width) * z, 4.0 * z * z  # :776-782


class StorageCurves:
    """The storage area (m2) and volume (m3) of every junction as functions of its depth
    (m), vectorised over the junctions (no per-junction loop on a pass); a junction with no
    curve (``None``) has neither. `full_depth` (m, per junction) caps the volume
    (node.c:943-945); a curve whose full volume is negative is refused by name with
    `names` (node.c:219-222)."""

    def __init__(self, curves: Sequence[tuple | None], full_depth: Tensor,
                 names: Sequence[str] | None = None) -> None:
        names = list(names) if names is not None else [f"#{i}" for i in range(len(curves))]
        for name, curve in zip(names, curves, strict=True):
            if curve is not None:
                validate_storage_curve(name, curve)
        n = len(curves)
        kind, a0, a1, a2 = [_NONE] * n, [0.0] * n, [0.0] * n, [0.0] * n
        tables = {}
        for i, curve in enumerate(curves):
            if curve is None:
                continue
            if curve[0] == "tabular":
                kind[i] = _TABULAR
                tables[i] = (tuple(float(v) for v in curve[1]),
                             tuple(float(v) for v in curve[2]))
            else:
                kind[i], a0[i], a1[i], a2[i] = _coefficients(curve)
        self.kind = torch.tensor(kind, dtype=torch.long)
        self.a0, self.a1, self.a2 = (torch.tensor(c, dtype=F64) for c in (a0, a1, a2))
        #: the FUNCTIONAL exponent, 0 on every other row: `area` and `_raw_volume` evaluate
        #: the power law on all rows, and a closed shape's a2 (a quadratic coefficient, e.g.
        #: 4 Z^2) as an exponent would overflow d^a2 there, whose discarded branch still
        #: gives 0 * inf = NaN in the backward pass
        self.exponent = torch.where(self.kind == _FUNCTIONAL, self.a2, torch.zeros_like(self.a2))
        self._tables(n, tables)
        self.full_depth = torch.as_tensor(full_depth, dtype=F64).detach()
        #: per-junction full volume (m3): the uncapped volume at the full depth (node.c:251)
        self.full_volume = self._raw_volume(self.full_depth)
        bad = (self.kind != _NONE) & (self.full_volume < 0.0)
        if bool(bad.any()):
            which = [names[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(
                f"storage unit(s) {which}: the curve gives a negative volume at the full depth "
                f"(SWMM's ERR_STORAGE_VOLUME)")

    def _tables(self, n: int, tables: dict) -> None:
        """Pad the tabular curves into (n, M) tensors: depths padded with +inf (never below
        a depth), areas with the last area; the cumulative end-area volume at each depth;
        and each table's last-point extrapolation slopes."""
        m = max([2] + [len(x) for x, _ in tables.values()])    # >= 2: a segment to gather
        xs = torch.full((n, m), torch.inf, dtype=F64)
        ys = torch.zeros((n, m), dtype=F64)
        xs[:, 0] = 1.0                       # a harmless first depth for non-tabular rows
        count = torch.ones(n, dtype=torch.long)
        for i, (x, y) in tables.items():
            xs[i, :len(x)] = torch.tensor(x, dtype=F64)
            ys[i, :len(y)] = torch.tensor(y, dtype=F64)
            ys[i, len(y):] = y[-1]
            count[i] = len(x)
        self.tx, self.ty, self.count = xs, ys, count
        dx = torch.where(torch.isfinite(xs[:, 1:]), xs[:, 1:] - xs[:, :-1],
                         torch.zeros_like(xs[:, 1:]))
        seg = (ys[:, :-1] + ys[:, 1:]) / 2.0 * dx                             # table.c:627
        self.tv = torch.cat([torch.zeros((n, 1), dtype=F64), seg.cumsum(-1)], -1)
        last = count - 1
        prev = torch.clamp(last - 1, min=0)
        self.x_last = xs.gather(-1, last[:, None])[:, 0]
        self.y_last = ys.gather(-1, last[:, None])[:, 0]
        self.v_last = self.tv.gather(-1, last[:, None])[:, 0]
        dx_last = torch.where(count > 1, self.x_last - xs.gather(-1, prev[:, None])[:, 0],
                              torch.zeros_like(self.x_last))
        dy_last = self.y_last - ys.gather(-1, prev[:, None])[:, 0]
        has = dx_last > 0.0
        slope = torch.where(has, dy_last / torch.where(has, dx_last, torch.ones_like(dx_last)),
                            torch.zeros_like(dx_last))
        #: the area's slope beyond the last depth, floored at 0 (table.c:497)
        self.area_slope = torch.clamp(slope, min=0.0)
        #: the volume's: the last segment's slope where that segment exceeds _X_TINY
        #: (table.c:634), else no extrapolation
        self.volume_slope = slope
        self.volume_extends = dx_last > _X_TINY

    # ------------------------------------------------------------------ tabular pieces
    def _bracket(self, d: Tensor) -> tuple[Tensor, ...]:
        """For depths `d` (..., n): the region (below the first depth, inside, beyond the
        last) and the bracketing segment's end points ``x1 < d <= x2``."""
        below = d <= self.tx[:, 0]
        beyond = d > self.x_last
        k = (self.tx < d[..., None]).sum(-1)               # first depth >= d is at index k
        hi = torch.clamp(k, min=1, max=self.tx.shape[-1] - 1)
        hi = torch.minimum(hi, torch.clamp(self.count - 1, min=1))
        lo = hi - 1
        shape = d.shape[:-1] + self.tx.shape

        def at(table: Tensor, idx: Tensor) -> Tensor:
            return table.expand(shape).gather(-1, idx[..., None])[..., 0]

        x1, x2, y1, y2 = at(self.tx, lo), at(self.tx, hi), at(self.ty, lo), at(self.ty, hi)
        v1 = at(self.tv, lo)
        x2 = torch.where(torch.isfinite(x2), x2, x1 + 1.0)    # one-point tables: unused
        return below, beyond, x1, x2, y1, y2, v1

    def _table_area(self, d: Tensor) -> Tensor:
        """`table_lookupEx` (table.c:464-499) at depths `d` (m)."""
        below, beyond, x1, x2, y1, y2, _ = self._bracket(d)
        first_x, first_y = self.tx[:, 0], self.ty[:, 0]
        positive = first_x > 0.0
        under = torch.where(positive, d / torch.where(positive, first_x, torch.ones_like(d))
                            * first_y, first_y)                               # table.c:484-485
        inside = y1 + (d - x1) * (y2 - y1) / (x2 - x1)                        # table.c:62
        past = self.y_last + self.area_slope * (d - self.x_last)              # table.c:497-498
        return torch.where(below, under, torch.where(beyond, past, inside))

    def _table_volume(self, d: Tensor) -> Tensor:
        """`table_getStorageVolume` (table.c:587-647) at depths `d` (m)."""
        below, beyond, x1, x2, y1, y2, v1 = self._bracket(d)
        first_x, first_y = self.tx[:, 0], self.ty[:, 0]
        wide = first_x >= _X_TINY
        under = torch.where(wide, first_y / torch.where(wide, first_x, torch.ones_like(d))
                            * d * d / 2.0, torch.zeros_like(d))               # table.c:608-609
        a = y1 + (d - x1) * (y2 - y1) / (x2 - x1)                             # table.c:619
        inside = v1 + (y1 + a) / 2.0 * (d - x1)                               # table.c:620
        run = d - self.x_last
        a_past = self.y_last + self.volume_slope * run                        # table.c:637
        falling = self.volume_slope < 0.0
        safe_s = torch.where(falling, self.volume_slope, -torch.ones_like(self.volume_slope))
        past = torch.where(
            a_past < 0.0, self.v_last - self.y_last * self.y_last / safe_s / 2.0,  # table.c:640-641
            self.v_last + (self.y_last + a_past) / 2.0 * run)                 # table.c:644
        past = torch.where(self.volume_extends, past, self.v_last.expand_as(past))
        return torch.where(below, under, torch.where(beyond, past, inside))

    # ------------------------------------------------------------------ public
    def area(self, depth: Tensor) -> Tensor:
        """Storage surface area (m2) at `depth` (m, >= 0, per junction): SWMM's
        `storage_getSurfArea` (node.c:979-1018); 0 at a junction with no curve."""
        d = depth
        pos = d > 0.0
        d_safe = torch.where(pos, d, torch.ones_like(d))
        b = self.exponent
        power = torch.where(pos, d_safe ** b, (b == 0.0).to(F64))            # C pow(0, b)
        functional = self.a0 + self.a1 * power                               # node.c:1001-1003
        quadratic = self.a0 + d * (self.a1 + d * self.a2)                    # node.c:1012
        tabular = self._table_area(d)                                        # node.c:997
        return torch.where(self.kind == _FUNCTIONAL, functional,
                           torch.where(self.kind == _QUADRATIC, quadratic,
                                       torch.where(self.kind == _TABULAR, tabular,
                                                   torch.zeros_like(d))))

    def _raw_volume(self, d: Tensor) -> Tensor:
        """The curve's volume (m3) at `d` (m) without the full-volume cap."""
        pos = d > 0.0
        d_safe = torch.where(pos, d, torch.ones_like(d))
        n = self.exponent + 1.0
        functional = torch.where(pos, self.a0 * d_safe + self.a1 / n * d_safe ** n,
                                 torch.zeros_like(d))                         # node.c:959-962
        quadratic = d * (self.a0 + d * (self.a1 / 2.0 + d * self.a2 / 3.0))  # node.c:970
        tabular = self._table_volume(d)                                       # node.c:953
        return torch.where(self.kind == _FUNCTIONAL, functional,
                           torch.where(self.kind == _QUADRATIC, quadratic,
                                       torch.where(self.kind == _TABULAR, tabular,
                                                   torch.zeros_like(d))))

    def volume(self, depth: Tensor) -> Tensor:
        """Stored volume (m3) at `depth` (m, >= 0, per junction): SWMM's `storage_getVolume`
        (node.c:930-975), 0 at zero depth and the full volume at or above the full depth
        (where that is positive); 0 at a junction with no curve."""
        raw = self._raw_volume(depth)
        capped = (depth >= self.full_depth) & (self.full_volume > 0.0)        # node.c:944-945
        out = torch.where(capped, self.full_volume.expand_as(raw), raw)
        return torch.where(depth == 0.0, torch.zeros_like(out), out)          # node.c:943
