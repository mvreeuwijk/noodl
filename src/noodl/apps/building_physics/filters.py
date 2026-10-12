"""Contaminant filters on airflow paths.

A filter removes the fraction `efficiency` of each species from the air passing through
its edge, in whichever direction the air flows (measured against ContamX 3.4.1.7). The
species layer holds it as its `transmission`, the pass fraction 1 - efficiency per edge and
species; `filter_transmission` builds that from filter objects on edges. The three filters
are CONTAM's constant-efficiency, simple particle and super filters (TN 1887 Appendix A
section 8a); filters whose efficiency depends on their loading, and UVGI filters, are not
modelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from noodl.apps.building_physics.species import Species
from noodl.topology import EdgeKey, Network

Tensor = torch.Tensor
F64 = torch.float64


class ConstantEfficiencyFilter:
    """Removes a fixed fraction of each species it lists, nothing of the others.

    `efficiencies` maps species names to fractions in [0, 1] (floats or tensors)."""

    def __init__(self, efficiencies: Mapping[str, float | Tensor], *, name: str = "") -> None:
        self.efficiencies = {s: torch.as_tensor(e, dtype=F64) for s, e in efficiencies.items()}
        bad = {s: e.item() for s, e in self.efficiencies.items()
               if not 0.0 <= e.detach().item() <= 1.0}
        if bad:
            raise ValueError(f"ConstantEfficiencyFilter {name!r}: efficiencies must be "
                             f"between 0 and 1, got {bad}")
        self.name = name

    def efficiency(self, species: Sequence[Species]) -> Tensor:
        """`(K,)`: each species' removed fraction, in `species` order."""
        names = [s.name for s in species]
        unknown = sorted(set(self.efficiencies) - set(names))
        if unknown:
            raise KeyError(f"ConstantEfficiencyFilter {self.name!r}: species {unknown} are not "
                           f"among the model's species {names}")
        zero = torch.zeros((), dtype=F64)
        return torch.stack([self.efficiencies.get(n, zero) for n in names])


def _clamped_slopes(x: Tensor, y: Tensor) -> Tensor:
    """Node slopes of the C2 cubic spline through (x, y) with zero slope at both ends."""
    n = x.numel()
    if n <= 2:
        return torch.zeros(n, dtype=F64)
    inv = 1.0 / (x[1:] - x[:-1])
    secant = (y[1:] - y[:-1]) * inv
    A = (torch.diag(2.0 * (inv[:-1] + inv[1:])) + torch.diag(inv[1:-1], 1)
         + torch.diag(inv[1:-1], -1))
    b = 3.0 * (secant[:-1] * inv[:-1] + secant[1:] * inv[1:])
    zero = torch.zeros(1, dtype=F64)
    return torch.cat([zero, torch.linalg.solve(A, b), zero])


class ParticleFilter:
    """Efficiency as a function of particle diameter (CONTAM's simple particle filter, e.g.
    a MERV curve); zero for a gas.

    `diameters` (m, increasing) and `efficiencies` (0..1) are the curve's points. As in
    ContamX 3.4.1.7 (measured): the curve is a cubic spline in log10(diameter) through the
    points from the second to the second-last, with zero slope at both of them, joined to
    the first and the last point by flat segments, and constant beyond the curve. ContamX
    refuses a curve whose first two or last two efficiencies differ, and so does this
    class. Efficiencies may be tensors that carry gradients.
    """

    def __init__(self, diameters, efficiencies, *, name: str = "") -> None:
        d = torch.as_tensor(diameters, dtype=F64).reshape(-1)
        e = torch.as_tensor(efficiencies, dtype=F64).reshape(-1)
        label = f"ParticleFilter {name!r}"
        if d.shape != e.shape or d.numel() < 2:
            raise ValueError(f"{label}: diameters and efficiencies must be two lists of "
                             f"equal length, at least 2 points")
        dd, ee = d.detach(), e.detach()
        if bool((dd <= 0).any()) or bool((dd[1:] <= dd[:-1]).any()):
            raise ValueError(f"{label}: diameters must be positive and increasing, "
                             f"got {dd.tolist()}")
        if bool((ee < 0).any()) or bool((ee > 1).any()):
            raise ValueError(f"{label}: efficiencies must be between 0 and 1, "
                             f"got {ee.tolist()}")
        if ee[0] != ee[1] or ee[-2] != ee[-1]:
            raise ValueError(f"{label}: the first two and the last two efficiencies must be "
                             f"equal (the curve is flat at both ends; ContamX refuses this "
                             f"curve), got {ee.tolist()}")
        self.diameters, self.efficiencies, self.name = d, e, name
        self._x, self._y = torch.log10(d[1:-1]), e[1:-1]
        self._slopes = _clamped_slopes(self._x, self._y)

    def at(self, diameter: float | Tensor) -> Tensor:
        """The efficiency at a particle diameter (m)."""
        d = torch.as_tensor(diameter, dtype=F64)
        xs, ys, ms = self._x, self._y, self._slopes
        if xs.numel() == 0:
            return self.efficiencies[0] * torch.ones_like(d)
        if xs.numel() == 1:
            return ys[0] * torch.ones_like(d)
        x = torch.log10(d).clamp(xs[0].item(), xs[-1].item())
        j = torch.searchsorted(xs.detach(), x.detach().reshape(-1), right=True)
        j = (j.reshape(x.shape) - 1).clamp(0, xs.numel() - 2)
        h = xs[j + 1] - xs[j]
        t = (x - xs[j]) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * ys[j] + (t3 - 2 * t2 + t) * h * ms[j]
                + (3 * t2 - 2 * t3) * ys[j + 1] + (t3 - t2) * h * ms[j + 1])

    def efficiency(self, species: Sequence[Species]) -> Tensor:
        """`(K,)`: the curve at each particle species' mean diameter, 0 for a gas."""
        zero = torch.zeros((), dtype=F64)
        return torch.stack([self.at(s.mean_diameter) if s.is_particle else zero
                            for s in species])


class FilterSeries:
    """Filters in series (CONTAM's super filter): the air passes each stage in turn, so the
    pass fraction is the product of the stages' pass fractions."""

    def __init__(self, stages: Sequence, *, name: str = "") -> None:
        if not stages:
            raise ValueError(f"FilterSeries {name!r}: needs at least one stage")
        self.stages, self.name = list(stages), name

    def efficiency(self, species: Sequence[Species]) -> Tensor:
        """`(K,)`: 1 minus the product of the stages' pass fractions."""
        passed = torch.ones(len(species), dtype=F64)
        for stage in self.stages:
            passed = passed * (1.0 - stage.efficiency(species))
        return 1.0 - passed


def filter_transmission(net: Network, flow_kinds: Sequence[str], species: Sequence[Species],
                        filters: Mapping[EdgeKey, object]) -> Tensor:
    """The species layer's `transmission`, `(b_flow, K)`: 1 - efficiency on each filtered
    edge, 1 elsewhere. Rows follow the layer's flow order: the edges of each kind in
    `flow_kinds`, kind by kind, each kind in network order; pass the layer's own
    `flow_kinds`. An edge is named by the `(source, target, key)` `add_edge` returned."""
    order: list[int] = []
    for kind in flow_kinds:
        order += net.edge_index(kind).tolist()
    position = {col: i for i, col in enumerate(order)}
    edges, kinds = net.edges, net.edge_kinds()
    rows = [torch.ones(len(species), dtype=F64) for _ in order]
    for edge, flt in filters.items():
        try:
            col = edges.index(tuple(edge))
        except ValueError as exc:
            raise KeyError(f"filter_transmission: edge {edge!r} is not in the network") from exc
        if col not in position:
            raise KeyError(f"filter_transmission: edge {edge!r} is of kind {kinds[col]!r}, "
                           f"not one of the species layer's flow kinds {tuple(flow_kinds)}")
        rows[position[col]] = 1.0 - flt.efficiency(species)
    return torch.stack(rows) if rows else torch.ones(0, len(species), dtype=F64)
