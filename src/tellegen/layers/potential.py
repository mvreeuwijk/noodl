"""PotentialFlowLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + drive; theta) - s_I from a set of Element laws
on typed edges (one Element per kind) and Drive terms (additive potential differences).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tellegen.drives import Drive
from tellegen.elements.base import Element
from tellegen.solvers.linear import floating_nodes, solve
from tellegen.solvers.newton import newton
from tellegen.topology import Network


class PotentialFlowLayer:
    def __init__(
        self,
        net: Network,
        name: str,
        elements: Sequence[Element],
        drives: Sequence[Drive] = (),
        boundary: Sequence = (),
    ) -> None:
        seen_kinds: set[str] = set()
        for el in elements:
            if el.kind in seen_kinds:
                raise ValueError(f"duplicate element kind {el.kind!r} in layer {name!r}")
            seen_kinds.add(el.kind)

        self.net = net
        self.name = name
        self._elements = list(elements)
        self._drives = list(drives)
        self.kinds = [el.kind for el in elements]

        cols_list = []
        self._kind_slices: dict[str, tuple[int, int]] = {}
        self._elem_slices: list[tuple[int, int]] = []
        offset = 0
        for el in elements:
            # net.edge_index raises KeyError when el.kind carries no edge at all (it never
            # returns an empty tensor for a non-None kind: a kind either has edges, in which
            # case idx is non-empty, or it appears nowhere in the network, in which case
            # edge_index itself raises). Re-raise as ValueError so a missing element kind and
            # a genuinely unknown node/kind lookup elsewhere stay distinguishable to callers.
            try:
                idx = net.edge_index(el.kind)
            except KeyError as exc:
                raise ValueError(
                    f"element kind {el.kind!r} has no edges in the network"
                ) from exc
            cols_list.append(idx)
            n_e = idx.numel()
            self._kind_slices[el.kind] = (offset, offset + n_e)
            self._elem_slices.append((offset, offset + n_e))
            offset += n_e
        self.cols = torch.cat(cols_list)
        self.A = net.incidence()[:, self.cols]

        node_index = {node: i for i, node in enumerate(net.nodes)}
        for b in boundary:
            if b not in node_index:
                raise KeyError(f"unknown boundary node {b!r} in layer {name!r}")

        self.interior = net.interior_index(boundary)
        self.bound = net.boundary_index(boundary)
        self._interior_names = [net.nodes[i] for i in self.interior.tolist()]

    # ------------------------------------------------------------------ assembly
    def dp(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        drivers = drivers or {}
        d = torch.einsum("ie,...i->...e", self.A, phi)
        parts = []
        for kind, (start, end) in self._kind_slices.items():
            block = d[..., start:end]
            for drv in self._drives:
                if drv.kind == kind:
                    block = block + drv(phi, drivers)
            parts.append((start, block))
        parts.sort(key=lambda p: p[0])
        return torch.cat([p[1] for p in parts], dim=-1)

    def flows(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        parts = [
            el.flow(d[..., s:e], drivers)
            for el, (s, e) in zip(self._elements, self._elem_slices, strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def dflows(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        parts = [
            el.dflow(d[..., s:e], drivers)
            for el, (s, e) in zip(self._elements, self._elem_slices, strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def assemble(self, phi_interior: torch.Tensor, phi_boundary: torch.Tensor) -> torch.Tensor:
        batch_shape = torch.broadcast_shapes(
            phi_interior.shape[:-1], phi_boundary.shape[:-1]
        )
        n = self.A.shape[0]
        phi = torch.zeros(
            batch_shape + (n,), dtype=phi_interior.dtype, device=phi_interior.device
        )
        phi[..., self.interior] = phi_interior.expand(batch_shape + (len(self.interior),))
        phi[..., self.bound] = phi_boundary.expand(batch_shape + (len(self.bound),))
        return phi

    # ------------------------------------------------------------------ Newton residual
    def _source_interior(self, sources: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if sources is None:
            return torch.zeros(
                ref.shape[:-1] + (len(self.interior),), dtype=ref.dtype, device=ref.device
            )
        return sources[..., self.interior]

    def residual(self, phi_interior, phi_boundary, drivers, sources):
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        q = self.flows(phi, drivers)
        A_I = self.A[self.interior]
        lhs = torch.einsum("ie,...e->...i", A_I, q)
        s_I = self._source_interior(sources, phi_interior)
        return lhs - s_I

    def jacobian(self, phi_interior, phi_boundary, drivers):
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self.dflows(phi, drivers)
        A_I = self.A[self.interior]
        return torch.einsum("ie,...e,je->...ij", A_I, dq, A_I)
