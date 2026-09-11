"""PotentialFlowLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + drive; theta) - s_I from a set of Element laws
on typed edges (one Element per kind) and Drive terms (additive potential differences).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tellegen.drives import Drive
from tellegen.elements.base import Element
from tellegen.solvers.linear import solve
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

        kind_set = set(self.kinds)
        for drv in self._drives:
            if drv.kind not in kind_set:
                raise ValueError(
                    f"drive kind {drv.kind!r} is not one of this layer's element kinds "
                    f"{sorted(kind_set)} (layer {name!r})"
                )

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
            width = end - start
            for drv in self._drives:
                if drv.kind == kind:
                    value = drv(phi, drivers)
                    # A width mismatch here is silent corruption, not a crash: torch.cat below
                    # would happily accept a wrong-width block, shifting every later kind's
                    # slice out from under `_elem_slices` so a DIFFERENT element ends up being
                    # fed this kind's drive value with no exception anywhere in the call chain.
                    if value.ndim > 0 and value.shape[-1] != width:
                        raise ValueError(
                            f"drive kind {drv.kind!r} returned width {value.shape[-1]} but "
                            f"the layer's {drv.kind!r} block has {width} edges"
                        )
                    block = block + value
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

    def _linear_ck(self, drivers):
        # Each element's own (c_e, k_e) covers only its own n_e = e - s edges; only the
        # leading BATCH dims (everything but the trailing edge-count dim) are meant to be
        # unified across elements before concatenating along the edge axis. Using
        # torch.broadcast_tensors directly on the raw (c_e, k_e) list (as an earlier version
        # of this method did) broadcasts the trailing edge dim too, so two elements with
        # different edge counts silently expand to a shared (wrong) edge count instead of
        # concatenating -- e.g. a 2-edge FixedFlow and a 1-edge PowerLaw would both become
        # 2-edge before torch.cat, yielding a 4-wide result instead of the correct 3.
        cs, ks, n_es = [], [], []
        for el, (s, e) in zip(self._elements, self._elem_slices, strict=True):
            c_e, k_e = el.linear_init(drivers)
            cs.append(c_e)
            ks.append(k_e)
            n_es.append(e - s)
        batch_shape = torch.broadcast_shapes(
            *(c.shape[:-1] if c.ndim > 0 else () for c in cs),
            *(k.shape[:-1] if k.ndim > 0 else () for k in ks),
        )
        dtype = cs[0].dtype
        device = cs[0].device
        c_parts = []
        k_parts = []
        for c_e, k_e, n_e in zip(cs, ks, n_es, strict=True):
            zero = torch.zeros(batch_shape + (n_e,), dtype=dtype, device=device)
            c_parts.append(c_e + zero)
            k_parts.append(k_e + zero)
        return torch.cat(c_parts, dim=-1), torch.cat(k_parts, dim=-1)

    def _floating_group_nodes(self, k: torch.Tensor) -> list:
        """Interior node names in any connected GROUP with no path to a boundary node.

        Two nodes are "connected" here only through an EDGE whose own linear slope k is
        nonzero somewhere in the batch: an edge whose element contributes no slope at all
        at that edge (e.g. any FixedFlow edge, whose dflow is identically 0, or a closed
        damper of otherwise-slope-bearing kind sitting at g = 0) cannot carry a potential
        difference to a boundary node and so cannot rescue a floating group. This subsumes
        the isolated-single-node case (a node with zero diagonal in J0 is exactly a
        singleton component here, since J0's diagonal at node i is the sum of k over i's
        own incident edges) as well as a floating GROUP of two or more mutually-connected
        nodes that, as a whole, has no path to any boundary node -- which a per-node
        diagonal check alone cannot see, because each member's own diagonal is nonzero from
        its internal edges.

        Filtering must happen at EDGE granularity, not kind granularity: a kind can mix a
        zero-slope edge (a closed damper, g = 0) with a nonzero-slope edge of the very same
        kind (an open one, g = 1), and gating on "does this kind have any nonzero edge
        anywhere" would let the zero-slope edge itself connect a group it cannot actually
        support. `net.component_labels(kind)`/`net.n_components_of(kind)` cannot express
        this (they know edges by kind, not by slope), so this method does not use them: it
        unions the source and target of each of the LAYER's OWN columns (`self.cols`, whose
        order matches `k`'s) individually, filtered by that edge's own slope. Iterating the
        layer's own columns is already kind-restricted by construction (Task 2's
        whole-graph-vs-kind-restricted trap does not apply here, since no whole-graph
        connectivity operator is used at all).
        """
        n = self.net.n
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        k_flat = k.reshape(-1, k.shape[-1])
        nonzero_anywhere = (k_flat != 0).any(dim=0)
        node_index = {node: i for i, node in enumerate(self.net.nodes)}
        edges = self.net.edges
        for col_pos, col in enumerate(self.cols.tolist()):
            if not bool(nonzero_anywhere[col_pos]):
                continue
            u, v, _ = edges[col]
            union(node_index[u], node_index[v])

        boundary_roots = {find(i) for i in self.bound.tolist()}
        return [
            self.net.nodes[i] for i in self.interior.tolist() if find(i) not in boundary_roots
        ]

    def linear_init(self, phi_boundary, drivers, sources):
        drivers = drivers or {}
        batch_shape = phi_boundary.shape[:-1]
        phi_i0 = torch.zeros(
            batch_shape + (len(self.interior),),
            dtype=phi_boundary.dtype,
            device=phi_boundary.device,
        )
        phi0 = self.assemble(phi_i0, phi_boundary)
        dp0 = self.dp(phi0, drivers)
        c, k = self._linear_ck(drivers)
        A_I = self.A[self.interior]
        rhs = self._source_interior(sources, phi0) - torch.einsum(
            "ie,...e->...i", A_I, c + k * dp0
        )
        J0 = torch.einsum("ie,...e,je->...ij", A_I, k, A_I)
        bad = self._floating_group_nodes(k)
        if bad:
            raise RuntimeError(
                f"floating nodes with no path to a boundary potential: {bad}"
            )
        # J0 picks up a batch dimension only if some element's linear_init(drivers) actually
        # depends on drivers (e.g. a driver-conditioned slope); with purely constant slopes
        # (every test element here) J0 stays unbatched (n_I, n_I) even when `rhs` is batched
        # via phi_boundary/sources. torch.linalg.solve does not broadcast an unbatched A
        # against a batched b the way one might expect: with A.ndim == b.ndim it instead
        # treats b as a single (n_I, k) right-hand-side matrix, so a (2, 2) A against a
        # (50, 2) b raises (misreported as a singular system) rather than solving 50
        # independent 2x2 systems. Broadcasting both to their common batch shape first
        # makes A.ndim == b.ndim + 1 in the batched case, which torch.linalg.solve does
        # broadcast correctly as one system per batch element.
        solve_batch = torch.broadcast_shapes(J0.shape[:-2], rhs.shape[:-1])
        J0 = J0.expand(solve_batch + J0.shape[-2:])
        rhs = rhs.expand(solve_batch + rhs.shape[-1:])
        return solve(J0, rhs)

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        phi_boundary,
        drivers=None,
        sources=None,
        phi0=None,
        *,
        differentiable=True,
        **newton_kwargs,
    ):
        drivers = drivers or {}
        if phi0 is None:
            phi0 = self.linear_init(phi_boundary, drivers, sources)
        if differentiable:
            raise NotImplementedError(
                "differentiable=True is implemented in Task 8 (solvers/implicit.py)"
            )

        def residual_fn(x):
            return self.residual(x, phi_boundary, drivers, sources)

        def jacobian_fn(x):
            return self.jacobian(x, phi_boundary, drivers)

        result = newton(residual_fn, jacobian_fn, phi0, **newton_kwargs)
        phi = self.assemble(result.x, phi_boundary)
        q = self.flows(phi, drivers)
        return phi, q

    def power_residual(self, phi, q, drivers, sources=None):
        """Tellegen's power identity, zero at a converged solution.

        (dp*q).sum() - (drive*q).sum() - (phi_bound * (A_bound @ q)).sum()
        - (phi_interior * sources_interior).sum()

        Derivation: dp = A^T phi + drive, so dp^T q = phi^T (A q) + drive^T q. Splitting
        phi^T (A q) into interior and boundary parts and using A_I q = sources_I at a
        converged solution (`residual` is exactly this equation) gives
        dp^T q - drive^T q - phi_bound.(A_bound q) - phi_interior.sources_I == 0.
        `sources=None` (the default) means zero interior sources, matching `residual` and
        `linear_init`.
        """
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        drive_only = d - torch.einsum("ie,...i->...e", self.A, phi)
        A_bound = self.A[self.bound]
        boundary_flow = torch.einsum("be,...e->...b", A_bound, q)
        phi_b = phi[..., self.bound]
        phi_i = phi[..., self.interior]
        s_I = self._source_interior(sources, phi_i)
        return (
            (d * q).sum(-1)
            - (drive_only * q).sum(-1)
            - (phi_b * boundary_flow).sum(-1)
            - (phi_i * s_I).sum(-1)
        )
