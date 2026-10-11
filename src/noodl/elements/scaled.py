"""An element whose flow is another element's times a per-edge multiplier read from the drivers.

    F = s F_inner(dp),   dF/ddp = s dF_inner/ddp,   s = drivers[key]

This is how a schedule or a control signal acts on an airflow path. CONTAM's path schedule
multiplies the path's flow (TN 1887 section 3.17). The multiplier is a driver, so a run can
change it every step without rebuilding the element, and gradients reach it like any other
driver. `s` is one value per edge of the kind, or one for all of them, with leading batch
dimensions like every driver.

A zero multiplier closes the edge. A node whose every edge is closed has no path to a boundary
potential, and the airflow solve's grounding check raises naming it instead of returning a
`nan`. Scaling does not move the inner law's switching points, so `switching` is the inner
element's.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.elements.base import Element

Tensor = torch.Tensor


class Scaled(Element):
    """F = s F_element: `element`'s flow times the dimensionless multiplier `drivers[key]`.

    `drivers[key]` holds one value per edge of the element's kind, or a scalar for all of
    them, with leading batch dimensions like any driver. The kind is `element`'s own. Use it
    for a scheduled or controlled path, fan or damper: the schedule writes `drivers[key]`.
    """

    def __init__(self, element: Element, key: str) -> None:
        super().__init__(element.kind)
        self.inner = element
        self.key = str(key)
        self.dp_independent = bool(element.dp_independent)

    def _s(self, width: int | None, drivers: Mapping[str, Tensor] | None) -> Tensor:
        """The multiplier; `width` is the number of edges of the kind, when known."""
        if drivers is None or self.key not in drivers:
            raise KeyError(
                f"Scaled (kind {self.kind!r}): driver {self.key!r} not found; it carries "
                f"this kind's per-edge flow multiplier"
            )
        s = torch.as_tensor(drivers[self.key])
        if s.ndim and width is not None and s.shape[-1] not in (1, width):
            raise ValueError(
                f"Scaled (kind {self.kind!r}): driver {self.key!r} has {s.shape[-1]} values "
                f"on its last axis but the kind has {width} edges"
            )
        return s

    @staticmethod
    def _width(dp: Tensor) -> int | None:
        return dp.shape[-1] if dp.ndim else None

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        return self._s(self._width(dp), drivers) * self.inner.flow(dp, drivers)

    def dflow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        return self._s(self._width(dp), drivers) * self.inner.dflow(dp, drivers)

    def linear_init(self, drivers: Mapping[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        c, k = self.inner.linear_init(drivers)
        s = self._s(c.shape[-1] if c.ndim else None, drivers)
        return s * c, s * k

    def switching(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None):
        return self.inner.switching(dp, drivers)
