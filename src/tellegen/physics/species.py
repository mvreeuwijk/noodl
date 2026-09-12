"""Compatibility wrapper: SpeciesTransport as a thin layer over TransportLayer.

Kept so that downstream code (which imports ``SpeciesTransport`` and reads its
``interior``, ``boundary`` and ``volumes`` attributes directly) runs
unchanged while the species balance is
now implemented once, generally, in
:class:`tellegen.layers.transport.TransportLayer`.
"""

from __future__ import annotations

from collections.abc import Hashable

import torch

from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


class SpeciesTransport:
    """Species balance  V dc/dt = -div(q * U c) + s  on the interior nodes of a network.

    Boundary nodes hold prescribed concentrations. Flows must be non-negative, so the
    upwind operator is fixed by the edge directions (use a forward-oriented graph).
    Units are the caller's: with c in ppm, V in m3 and q in m3/s, sources are ppm m3/s.
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
        self.interior = [n for n in net.nodes if n not in self.boundary]
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
        result = self._layer.step(c, q, sources, c_boundary, dt)
        return result.to(out_dtype)
