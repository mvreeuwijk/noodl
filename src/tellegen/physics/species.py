"""Exact linear step of a species balance on a graph with prescribed branch flows."""

from __future__ import annotations

from collections.abc import Hashable

import torch

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

        index = {n: i for i, n in enumerate(net.nodes)}
        self._i_idx = torch.tensor([index[n] for n in self.interior], dtype=torch.long)
        self._b_idx = torch.tensor([index[n] for n in self.boundary], dtype=torch.long)
        cols = net.edge_index(kind)
        self._div = net.incidence(kind)  # (n, b_kind)
        # upwind for forward flows: select the source node of every edge of this kind
        q_ref = torch.zeros(net.b)
        q_ref[cols] = 1.0
        self._U = net.upwind(q_ref)[cols]  # (b_kind, n)

    def step(
        self,
        c: torch.Tensor,
        q: torch.Tensor,
        sources: torch.Tensor,
        c_boundary: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance interior concentrations by ``dt`` with flows and sources held fixed.

        Uses the augmented matrix exponential, so zero flows (singular ``A``) are fine.
        """
        if torch.any(q < 0):
            raise ValueError("branch flows must be non-negative on a forward-oriented graph")
        out_dtype = c.dtype
        dtype = torch.float64  # matrix exponential in double; the sizes are small
        c, q, sources, c_boundary = (v.to(dtype) for v in (c, q, sources, c_boundary))
        div = self._div.to(dtype)
        U = self._U.to(dtype)
        V = self.volumes.to(dtype)
        # M = div diag(q) U maps all-node concentrations to the net species outflow per node
        M = div @ (q[..., :, None] * U)  # (..., n, n)
        Mii = M[..., self._i_idx[:, None], self._i_idx[None, :]]
        Mib = M[..., self._i_idx[:, None], self._b_idx[None, :]]
        A = -Mii / V[:, None]
        inflow = -(Mib @ c_boundary[..., None])[..., 0]
        b0 = (inflow + sources) / V
        n = len(self.interior)
        batch = A.shape[:-2]
        Z = torch.zeros(*batch, 2 * n, 2 * n, dtype=dtype)
        Z[..., :n, :n] = A * dt
        Z[..., :n, n:] = torch.eye(n, dtype=dtype) * dt
        E = torch.linalg.matrix_exp(Z)
        Ed = E[..., :n, :n]
        Phi = E[..., :n, n:]
        result = (Ed @ c[..., None])[..., 0] + (Phi @ b0[..., None])[..., 0]
        return result.to(out_dtype)
