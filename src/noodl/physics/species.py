"""Compatibility wrapper: SpeciesTransport as a thin layer over TransportLayer.

Kept so that downstream code (which imports ``SpeciesTransport`` and reads its
``interior``, ``boundary`` and ``volumes`` attributes directly) runs
unchanged while the species balance is
now implemented once, generally, in
:class:`noodl.layers.transport.TransportLayer`.
"""

from __future__ import annotations

from collections.abc import Hashable

import torch

from noodl.layers.transport import TransportLayer, active_interior
from noodl.topology import Network


class SpeciesTransport:
    """Species balance  V dc/dt = -div(q * U c) + s  on the interior nodes of a network.

    Boundary nodes hold prescribed concentrations. Flows must be non-negative, so the
    upwind operator is fixed by the edge directions (use a forward-oriented graph).
    Units are the caller's: with c in ppm, V in m3 and q in m3/s, sources are ppm m3/s.

    `interior` (and so `volumes`, and the `sources`/`c` vectors below) covers the nodes an
    edge of `kind` touches, not every non-boundary node: see `layers.transport.active_interior`.
    On a network of one edge kind, which is how callers use this class, the two are the
    same set.

    This wrapper's own `sources` stays INTERIOR-only (its callers use it that way),
    unlike `TransportLayer.step`'s `sources`, which milestone 2 changed to FULL node
    order (spec 4.2): this class's `step` pads `sources` to FULL node order internally
    before delegating to `TransportLayer.step`.
    """

    def __init__(
        self,
        net: Network,
        volumes: dict[Hashable, float],
        boundary: list[Hashable],
        kind: str = "airpath",
    ) -> None:
        self.net = net
        self.kind = kind
        self.boundary = list(boundary)
        # The interior is the non-boundary nodes an edge of `kind` TOUCHES, in node order --
        # not every non-boundary node (spec 14, 4.5). On a network that also carries edges of
        # other kinds (a wall-conduction node, a street node in the composed model) the two
        # differ, and it is the active one that `TransportLayer` below sizes its rows by: the
        # old "all non-boundary nodes" spelling handed it a `capacity` vector one entry per
        # untouched node too long and was refused by its capacity check. A volume is required
        # for the active nodes only, since those are the only ones this layer has a state for.
        interior_idx, _inactive_idx = active_interior(net, (kind,), self.boundary)
        self.interior = [net.nodes[i] for i in interior_idx.tolist()]
        missing = [n for n in self.interior if n not in volumes]
        if missing:
            raise KeyError(f"no volume for interior nodes {missing}")
        self.volumes = torch.tensor([float(volumes[n]) for n in self.interior])
        self._layer = TransportLayer(
            net,
            "species",
            capacity=self.volumes,
            flow_kind=kind,
            boundary=self.boundary,
            n_species=1,
            scheme="exact",
        )

    def step(
        self,
        c: torch.Tensor,
        q: torch.Tensor,
        sources: torch.Tensor,
        c_boundary: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance interior concentrations by ``dt`` with flows and sources held fixed."""
        if torch.any(q < 0):
            raise ValueError("branch flows must be non-negative on a forward-oriented graph")
        out_dtype = c.dtype
        dtype = torch.float64
        c, q, sources, c_boundary = (v.to(dtype) for v in (c, q, sources, c_boundary))
        full = torch.zeros(*sources.shape[:-1], self.net.n, dtype=dtype)
        full[..., self._layer.interior_idx] = sources
        result = self._layer.step(c, q, full, c_boundary, dt)
        return result.to(out_dtype)
