"""PotentialFlowLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + drive; theta) - s_I from a set of Element laws
on typed edges (one Element per kind) and Drive terms (additive potential differences).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

import torch

from tellegen.drives import Drive, check_drive_signature
from tellegen.elements.base import Element
from tellegen.nodesources import NodeSource
from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.grounding import spd_certificate, spd_diagnosis
from tellegen.solvers.implicit import adjoint as _adjoint_solve
from tellegen.solvers.implicit import implicit_solve
from tellegen.solvers.newton import inner_solve_rtol, newton
from tellegen.solvers.select import solve as select_solve
from tellegen.topology import Network

# Inner linear solvers a layer may be configured with; forwarded verbatim as
# `solvers.select.solve`'s `method`. "direct" is the retained milestone-1 reference (the
# operator's explicit A_I diag(g) A_I^T, LU-factorised); "auto" is the migrated default;
# "sparse_direct" is the spec's section 6.2 SciPy SuperLU reference, which factorises the
# operator's O(E) COO form per instance instead. All four reach BOTH passes: `linear_init`
# and every Newton inner solve on the forward, and the implicit adjoint on the backward.
#
# What "auto" resolves to on THIS layer's operator (a certified-SPD GraphLaplacianOperator
# that declares `assemble_sparse`) is sparse-direct up to an ensemble of
# `select._SPARSE_DIRECT_MAX_BATCH` = 32 instances and Jacobi-preconditioned CG above it,
# because SuperLU is driven by a per-instance Python loop whose cost is linear in the
# ensemble while PCG's is sub-linear; the measured crossover sweep is at that constant. An
# explicit `linear_solver=` is honoured verbatim at every ensemble size. `diagnostics`
# reports which backend actually ran (see `solve`).
_LINEAR_SOLVERS = ("auto", "cg", "gmres", "direct", "sparse_direct")


class _DiagonalShifted:
    """`A_I diag(g) A_I^T + diag(d)`: the Jacobian of a layer carrying potential-dependent
    NODAL sources (spec 13.4), satisfying the LinearOperator duck type
    `solvers.select.solve` reads.

    The shift is symmetric by construction, so the whole operator stays symmetric and the
    adjoint's `rmatvec` is still the forward action. The SPD certificate is delegated to the
    Laplacian and kept ONLY when every shift entry is non-negative: a non-negative diagonal
    added to an SPD matrix is SPD, while a negative one could destroy definiteness, so the
    certificate is dropped there rather than asserted (`solvers.select.solve` then routes to
    GMRES, which needs none).
    """

    symmetric = True

    def __init__(self, base: GraphLaplacianOperator, diag: torch.Tensor) -> None:
        self.base = base
        self.diag_shift = diag
        self.shape = torch.broadcast_shapes(base.shape[:-2], diag.shape[:-1]) + base.shape[-2:]
        self.dtype = base.dtype
        self.device = base.device

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.matvec(x) + self.diag_shift * x

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.rmatvec(x) + self.diag_shift * x

    def diagonal(self) -> torch.Tensor:
        return self.base.diagonal() + self.diag_shift

    def assemble(self) -> torch.Tensor:
        return self.base.assemble() + torch.diag_embed(
            self.diag_shift.expand(self.shape[:-1])
        )

    def assemble_sparse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col, values = self.base.assemble_sparse()
        n_i = self.base.n_interior
        idx = torch.arange(n_i, dtype=torch.int64, device=row.device)
        shift = self.diag_shift.expand(values.shape[:-1] + (n_i,))
        return (
            torch.cat([row, idx]),
            torch.cat([col, idx]),
            torch.cat([values, shift], dim=-1),
        )

    def spd_certificate(self):
        if bool((self.diag_shift < 0).any()):
            return None
        return self.base.spd_certificate()


class PotentialFlowLayer:
    def __init__(
        self,
        net: Network,
        name: str,
        elements: Sequence[Element],
        drives: Sequence[Drive] = (),
        boundary: Sequence = (),
        linear_solver: str = "auto",
        *,
        node_sources: Sequence[NodeSource] = (),
        quantity: str = "potential",
        unit: str = "",
    ) -> None:
        if linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"unknown linear_solver {linear_solver!r} in layer {name!r}; expected one "
                f"of {_LINEAR_SOLVERS}"
            )
        self.linear_solver = linear_solver
        # Metadata only: what this layer's potential IS and what it is measured in (spec 4.4).
        # Nothing in the numerics reads them; `Model` reports them, and a caller composing
        # several layers over one network uses them to tell a pressure layer from a
        # temperature one without matching on layer names.
        self.quantity = quantity
        self.unit = unit

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
        # The node count, cached as a plain int: `assemble` (once per Newton iteration) and
        # solve's zero-source default used to read it off self.A.shape[0], which is exactly
        # the kind of incidental dense-matrix access that keeps an (n, b) tensor alive.
        self._n_nodes = net.n

        # This layer's own edge endpoints (restricted to self.cols, in the same order as
        # the columns of the (lazy) self.A / rows of self._diff), and a node -> interior-
        # position map, both needed to construct a GraphLaplacianOperator (and to run the
        # per-instance SPD certificate) without a per-solve Python loop. net.endpoints()
        # (kind=None) returns whole-graph (src, tgt) arrays in network edge order; indexing
        # by self.cols restricts them to this layer's own edges, exactly as the lazy
        # self.A == net.incidence()[:, self.cols] does for the incidence matrix. Since
        # Task 15 these ARE the layer's representation of its own topology: every hot-path
        # site gathers or scatter-adds with them instead of contracting against A/_diff.
        src_all, tgt_all = net.endpoints()
        self._src = src_all[self.cols]
        self._tgt = tgt_all[self.cols]

        kind_set = set(self.kinds)
        for drv in self._drives:
            check_drive_signature(drv, where=f"PotentialFlowLayer {name!r}")
            if drv.kind not in kind_set:
                raise ValueError(
                    f"drive kind {drv.kind!r} is not one of this layer's element kinds "
                    f"{sorted(kind_set)} (layer {name!r})"
                )

        node_index = {node: i for i, node in enumerate(net.nodes)}
        for b in boundary:
            if b not in node_index:
                raise KeyError(f"unknown boundary node {b!r} in layer {name!r}")

        touched = torch.zeros(net.n, dtype=torch.bool)
        touched[self._src] = True
        touched[self._tgt] = True
        all_interior = net.interior_index(boundary)
        # Spec 14, 4.5: a node no edge of this layer's kinds touches is INACTIVE for this
        # layer -- not an unknown, and not a singular row. A wall-mass node in a building
        # network is the motivating case: it has conduction edges and no airpath edges.
        # `touched` is built from this layer's OWN edge endpoints (`self._src`/`self._tgt`,
        # already restricted to `self.cols`), which is exactly "an edge of one of
        # `self.kinds`" -- the same rule `layers.transport.active_interior` applies for a
        # transport layer. An inactive node is not an unknown, so it is not reported as
        # ungrounded either: `solvers.grounding` gates on `_interior_of_node >= 0`, which
        # stays -1 there.
        self.interior = all_interior[touched[all_interior]]
        self.inactive = all_interior[~touched[all_interior]]
        self.bound = net.boundary_index(boundary)
        self._interior_names = [net.nodes[i] for i in self.interior.tolist()]
        self._inactive_names = [net.nodes[i] for i in self.inactive.tolist()]

        # interior_of_node: -1 at a boundary node's position AND at an inactive one's (an
        # inactive node is an unknown of neither kind, see the split above), else that node's
        # 0-based position within self.interior. boundary_mask: True at a boundary node's
        # position only -- an inactive node is not a prescribed value. Both are
        # (n,) and consumed by GraphLaplacianOperator's constructor and by
        # solvers.grounding; computing them once here, at construction time, avoids
        # rebuilding them on every solve() call.
        self._interior_of_node = torch.full((net.n,), -1, dtype=torch.long)
        self._interior_of_node[self.interior] = torch.arange(
            len(self.interior), dtype=torch.long
        )
        self._boundary_mask = torch.zeros(net.n, dtype=torch.bool)
        self._boundary_mask[self.bound] = True

        # POTENTIAL-DEPENDENT NODAL SOURCES (spec 13.4). Empty by default, so every
        # existing layer is unchanged. Each source's nodes must be INTERIOR unknowns of
        # this layer: a withdrawal at a prescribed node is absorbed by the boundary and
        # changes nothing, and one at an inactive node has no row to enter -- both are
        # wiring mistakes, named here rather than silently dropped. `_source_positions`
        # holds each source's nodes in this layer's COMPACT interior indexing, which is
        # what the residual's scatter-add and the Jacobian's diagonal both need.
        self._node_sources = list(node_sources)
        self._source_positions: list[torch.Tensor] = []
        for ns in self._node_sources:
            if not isinstance(ns, NodeSource):
                raise TypeError(
                    f"PotentialFlowLayer {name!r}: node_sources entry "
                    f"{type(ns).__name__} is not a NodeSource"
                )
            pos = self._interior_of_node[ns.nodes]
            if bool((pos < 0).any()):
                bad = [
                    net.nodes[int(i)]
                    for i, p in zip(ns.nodes.tolist(), pos.tolist(), strict=True)
                    if p < 0
                ]
                raise ValueError(
                    f"PotentialFlowLayer {name!r}: node source {ns!r} acts at {bad}, "
                    f"which are not interior unknowns of this layer (a boundary node's "
                    f"potential is prescribed and an inactive node has no row)"
                )
            self._source_positions.append(pos)

    # ------------------------------------------------------- dense oracles (lazy)
    @functools.cached_property
    def A(self) -> torch.Tensor:
        """This layer's (n, b_layer) incidence matrix, built on FIRST ACCESS only.

        Held as a `cached_property` rather than an `__init__` attribute since Task 15: at
        the composed model's reference size this matrix is 18.1 MB per layer and grows 4x
        per node doubling, which on its own broke the milestone's memory shape gate
        (measured 3.16x against a 2.5x budget in Task 14). Nothing inside this class reads
        it except `jacobian()`, the retained dense oracle -- every hot-path site gathers or
        scatter-adds with `_src`/`_tgt` instead (`dp`, `residual`, `linear_init`,
        `power_residual`). It stays public, with exactly its old value
        (`net.incidence()[:, self.cols]`, i.e. columns in LAYER edge order, not network
        order), because callers outside the layer legitimately want the matrix: the
        composed-model conservation tests form `A @ q` with it.
        """
        return self.net.incidence()[:, self.cols]

    @functools.cached_property
    def _diff(self) -> torch.Tensor:
        """`self.A.T`: the (b_layer, n) difference matrix, built on first access only.

        `net.difference()` (== `net.incidence().T`, the "source minus target" convention
        this whole solve path uses -- see topology.py's module docstring) restricted to this
        layer's own edge columns. Kept for callers and for symmetry with `A`; `dp()` gathers
        `phi[..., _src] - phi[..., _tgt]` instead of contracting against it.

        Like `A`, cached on first access and never invalidated: both assume `net` is not
        mutated after this layer is constructed. Nothing in this codebase does that -- a
        `Network` is built, then layers are constructed over it, then it is solved -- but a
        caller who mutated `net` afterwards (adding a node or edge) would get a layer whose
        `A`/`_diff` (if already accessed) or `_src`/`_tgt`/`cols` (fixed at `__init__`) no
        longer agree with the network's current topology, with no check anywhere that
        catches it.
        """
        return self.net.difference()[self.cols]

    # ------------------------------------------------------- sparse topology primitives
    def _difference(self, phi: torch.Tensor) -> torch.Tensor:
        """`self._diff @ phi` without the matrix: `phi[..., src] - phi[..., tgt]`.

        `Network.difference_ep` restricted to this layer's own columns. Exact, not merely
        close: the matrix form sums n terms of which all but two are exactly 0.0, and adding
        0.0 is exact in IEEE arithmetic, so the gather reproduces it bit for bit.
        """
        return phi[..., self._src] - phi[..., self._tgt]

    def _accumulate(self, w: torch.Tensor) -> torch.Tensor:
        """`self.A @ w` without the matrix: `+w` scattered at `src`, `-w` at `tgt`.

        `Network.accumulate` restricted to this layer's own columns (the network-level
        method works in network edge order and cannot express a layer's column subset).
        `index_add` is out of place, on a zero tensor this call itself allocates, so it is
        exactly as differentiable w.r.t. `w` as the matrix product is and cannot corrupt a
        tensor some earlier op still needs for its own backward pass.
        """
        out = torch.zeros(
            w.shape[:-1] + (self._n_nodes,), dtype=w.dtype, device=w.device
        )
        return out.index_add(-1, self._src, w).index_add(-1, self._tgt, -w)

    def _accumulate_interior(self, w: torch.Tensor) -> torch.Tensor:
        """`self._accumulate(w)` restricted to interior rows -- the repeated `A_I w` site."""
        return self._accumulate(w)[..., self.interior]

    def _accumulate_bound(self, w: torch.Tensor) -> torch.Tensor:
        """`self._accumulate(w)` restricted to boundary rows -- the repeated `A_bound w` site."""
        return self._accumulate(w)[..., self.bound]

    def _node_source_withdrawal(
        self, phi: torch.Tensor, drivers: Mapping
    ) -> torch.Tensor | None:
        """Total potential-dependent withdrawal per INTERIOR row, or `None` when this layer
        has no node sources (spec 13.4). `phi` is full-node."""
        if not self._node_sources:
            return None
        out = None
        for ns, pos in zip(self._node_sources, self._source_positions, strict=True):
            w = ns.flow(phi[..., ns.nodes], drivers)
            zeros = torch.zeros(
                w.shape[:-1] + (len(self.interior),), dtype=w.dtype, device=w.device
            )
            term = zeros.index_add(-1, pos, w)
            out = term if out is None else out + term
        return out

    def _node_source_slopes(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor | None:
        """`diag(w')` per INTERIOR row, or `None` when this layer has no node sources."""
        if not self._node_sources:
            return None
        out = None
        for ns, pos in zip(self._node_sources, self._source_positions, strict=True):
            d = ns.dflow(phi[..., ns.nodes], drivers)
            zeros = torch.zeros(
                d.shape[:-1] + (len(self.interior),), dtype=d.dtype, device=d.device
            )
            term = zeros.index_add(-1, pos, d)
            out = term if out is None else out + term
        return out

    def _operator_at(
        self, slopes: torch.Tensor, node_slopes: torch.Tensor | None = None
    ) -> GraphLaplacianOperator | _DiagonalShifted:
        """The matvec-free `A_I diag(slopes) A_I^T` operator at the given per-edge slope, in
        this layer's own endpoint/interior-index representation -- the construction repeated
        by `linear_init`, `solve`'s `operator_fn` (both the non-differentiable and
        differentiable branches) and `adjoint`. `slopes` is `dflows`'s actual Jacobian
        diagonal at the current iterate for all but `linear_init`, which instead passes its
        own tangent-at-zero `k` -- the same operator shape, at a different slope.

        `node_slopes` (spec 13.4) is the `diag(w')` a potential-dependent nodal source adds to
        that Jacobian; `None` (the usual case) returns the bare Laplacian, so nothing about an
        existing layer's operator changes.
        """
        op = GraphLaplacianOperator(
            self._src,
            self._tgt,
            slopes,
            len(self.interior),
            self._interior_of_node,
            boundary_mask=self._boundary_mask,
        )
        if node_slopes is None:
            return op
        return _DiagonalShifted(op, node_slopes)

    # ------------------------------------------------------------------ assembly
    def dp(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        drivers = drivers or {}
        d = self._difference(phi)
        parts = []
        for kind, (start, end) in self._kind_slices.items():
            block = d[..., start:end]
            width = end - start
            for drv in self._drives:
                if drv.kind == kind:
                    value = drv(drivers)
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

    def kind_slice(self, kind: str) -> slice:
        """Columns of `q` (as returned by `solve`) that belong to `kind`."""
        try:
            start, end = self._kind_slices[kind]
        except KeyError as exc:
            raise KeyError(
                f"PotentialFlowLayer {self.name!r} has no element of kind {kind!r}; its kinds "
                f"are {self.kinds}"
            ) from exc
        return slice(start, end)

    def flows_of_kind(self, q: torch.Tensor, kinds) -> torch.Tensor:
        """`q` restricted to `kinds` (a name or a sequence), concatenated in the given order.

        This is what a `TransportLayer` with `flow_kinds == kinds` expects as its `q`: the
        transport layer reads its flow edges kind by kind, in ITS OWN `flow_kinds` order,
        which need not be this layer's element order. `Model` couples the two layers with
        exactly this call.
        """
        if isinstance(kinds, str):
            kinds = (kinds,)
        return torch.cat([q[..., self.kind_slice(k)] for k in kinds], dim=-1)

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
        n = self._n_nodes
        phi = torch.zeros(
            batch_shape + (n,), dtype=phi_interior.dtype, device=phi_interior.device
        )
        phi[..., self.interior] = phi_interior.expand(batch_shape + (len(self.interior),))
        phi[..., self.bound] = phi_boundary.expand(batch_shape + (len(self.bound),))
        return phi

    # ------------------------------------------------------------------ Newton residual
    def _source_interior(self, sources: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        """Full-node `sources` -> this layer's interior rows; refuse a source on an INACTIVE
        node by name.

        An inactive node has no unknown in this layer, so a source there cannot be balanced
        by anything: it is a modelling error (a source put on the wall-mass node of an
        airflow layer), not a value to drop silently. Costs nothing on the usual layer, whose
        kinds touch every node: `self.inactive` is then empty and the check short-circuits.
        """
        if sources is None:
            return torch.zeros(
                ref.shape[:-1] + (len(self.interior),), dtype=ref.dtype, device=ref.device
            )
        if self.inactive.numel():
            nonzero = sources[..., self.inactive] != 0
            if bool(nonzero.any()):
                flat = nonzero.reshape(-1, nonzero.shape[-1]).any(0)
                names = [
                    self._inactive_names[i] for i in flat.nonzero().flatten().tolist()
                ]
                raise ValueError(
                    f"PotentialFlowLayer {self.name!r}: sources must be zero on inactive "
                    f"nodes (no edge of kinds {self.kinds} touches them); nonzero at {names}"
                )
        return sources[..., self.interior]

    def residual(self, phi_interior, phi_boundary, drivers, sources):
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        q = self.flows(phi, drivers)
        # (A_I q), by scatter-add over this layer's edges then a select of the interior
        # rows, rather than einsum against the (n_I, b) slice of the dense incidence: this
        # is once per Newton residual evaluation, and at ensemble 100 the einsum form alone
        # cost 14.2 ms of a 41.8 ms residual (Task 14's profile).
        lhs = self._accumulate_interior(q)
        s_I = self._source_interior(sources, phi_interior)
        w = self._node_source_withdrawal(phi, drivers)
        return lhs - s_I if w is None else lhs - s_I + w

    def jacobian(self, phi_interior, phi_boundary, drivers):
        """The dense (n_I, n_I) Jacobian A_I diag(dq) A_I^T -- the retained ORACLE.

        This is the one method that reads `self.A` (and so materialises it, once, on first
        access); no solve path calls it. Tests compare `GraphLaplacianOperator`'s matvec and
        the adjoint against this, so its einsum form is deliberately unchanged.
        """
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self.dflows(phi, drivers)
        A_I = self.A[self.interior]
        dense = torch.einsum("ie,...e,je->...ij", A_I, dq, A_I)
        node_slopes = self._node_source_slopes(phi, drivers)
        if node_slopes is None:
            return dense
        return dense + torch.diag_embed(node_slopes.expand(dense.shape[:-1]))

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

    def _grounding_check(self, slopes: torch.Tensor, *, where: str) -> None:
        """Raise unless every instance in `slopes` certifies SPD.

        The certificate (Task 3's `solvers.grounding.spd_certificate`) tests both of spec
        section 3.1's testable conditions -- every branch slope non-negative, and every
        interior node grounded through strictly positive slopes -- so the batched message
        below ("do not certify a grounded, positive-slope system") states exactly what was
        checked. It is run on the slopes
        GIVEN -- the caller decides whether those are `linear_init`'s tangent-at-zero slopes
        or the actual `dflows` at a solve point -- and it is per instance. That is the whole
        point: the pre-Task-11 check ORed "is this edge's slope nonzero" across the WHOLE
        batch before testing connectivity, so an edge closed in one instance but open in
        another counted as present for both, and a genuinely ungrounded instance sailed
        through to a dense factorisation that could only report "singular", if it reported
        anything at all.

        The message is built from `solvers.grounding.spd_diagnosis` (amendment A2), which
        names, per failing instance, either the negative-slope EDGES or the ungrounded
        interior NODES. Node indices are rendered as node NAMES here, because this layer --
        unlike the raw operator -- knows them, and because the error text every existing
        (unbatched) test in test_potential.py asserts on is exactly those names. A batched
        failure additionally leads with the failing BATCH INDICES, since node-level detail
        alone is not attributable across unrelated per-instance failures. Every message
        names the offending LAYER first: a model composes several layers over one network,
        and "solve: floating nodes ... ['z']" alone does not say which of them failed.
        """
        certified = spd_certificate(
            self._src, self._tgt, slopes, self._interior_of_node, self._boundary_mask, atol=0.0
        )
        if bool(torch.all(certified)):
            return
        records = spd_diagnosis(
            self._src, self._tgt, slopes, self._interior_of_node, self._boundary_mask, atol=0.0
        )
        if certified.ndim == 0:
            # Unbatched: one instance, so a bare batch index would say nothing. Name the
            # nodes (or edges) directly, as the pre-Task-11 message did.
            raise RuntimeError(
                f"PotentialFlowLayer {self.name!r}: {where}: "
                f"{self._describe_grounding(records[0])}"
            )
        bad_idx = torch.nonzero(~certified.reshape(-1), as_tuple=False).flatten().tolist()
        details = "; ".join(
            f"instance {rec['instance']}: {self._describe_grounding(rec)}" for rec in records
        )
        raise RuntimeError(
            f"PotentialFlowLayer {self.name!r}: {where}: batch indices {bad_idx} do not "
            f"certify a grounded, positive-slope system; {details}"
        )

    def _describe_grounding(self, record: dict) -> str:
        """One failing instance's `spd_diagnosis` record, with node INDICES resolved to the
        network's own node names (which the operator-level diagnosis cannot know)."""
        if record["reason"] == "negative_slope":
            return f"negative slope on edges {record['edges']}"
        names = [self.net.nodes[i] for i in record["nodes"]]
        return (
            f"floating nodes with no path to a boundary potential: "
            f"ungrounded interior nodes {names}"
        )

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
        rhs = self._source_interior(sources, phi0) - self._accumulate_interior(c + k * dp0)
        # A node source contributes its tangent at phi = 0 to the same linearisation:
        # w(phi) ~ w(0) + w'(0) phi, so w(0) moves to the right-hand side and w'(0) joins
        # the operator's diagonal (spec 13.4).
        node_slopes = None
        if self._node_sources:
            w0 = self._node_source_withdrawal(phi0, drivers)
            node_slopes = self._node_source_slopes(phi0, drivers)
            rhs = rhs - w0
        self._grounding_check(k, where="linear_init")
        # k picks up a batch dimension only if some element's linear_init(drivers) actually
        # depends on drivers (e.g. a driver-conditioned slope); with purely constant slopes
        # k stays unbatched (b,) even when `rhs` is batched via phi_boundary/sources.
        # Broadcasting both to their common batch shape first keeps the operator's own batch
        # shape and the right-hand side's in agreement, so the solve is one system per batch
        # element rather than one system with several right-hand sides.
        solve_batch = torch.broadcast_shapes(k.shape[:-1], rhs.shape[:-1])
        k = k.expand(solve_batch + k.shape[-1:])
        rhs = rhs.expand(solve_batch + rhs.shape[-1:])
        op = self._operator_at(k, node_slopes)
        result = select_solve(
            op,
            rhs,
            method=self.linear_solver,
            where=f"PotentialFlowLayer {self.name!r} linear_init",
            rtol=inner_solve_rtol(rhs.dtype),
        )
        return result.x

    def _check_no_unreachable_differentiable_tensors(self) -> None:
        """Guard against a gradient that would be silently wrong or absent (review finding
        2/3): the differentiable solve threads only two kinds of tensor into
        `Function.apply` -- each Element's own registered `nn.Parameter`s (found via
        `named_parameters()`, substituted in via `functional_call`) and whatever is reachable
        through the `drivers`/`sources`/`phi_boundary` arguments to `solve()`. Any OTHER
        tensor an Element or Drive happens to hold with `requires_grad=True` is captured only
        by closure (the element/drive object itself, not its tensor payload) and is
        invisible to `_Implicit.backward`: for an Element, this happens when it was
        constructed with `learnable=False` on a tensor that already had `requires_grad=True`
        (`Element._param`'s documented pass-through exception, which is still a correct
        construction -- its graph is reachable through this layer's own component calls
        (`flows`, `dflows`, `residual`, `jacobian`, `linear_init`), though NOT through
        `solve(differentiable=False)`, which since the Task C fix returns detached tensors);
        for a Drive, this happens whenever a Drive
        implementation owns a learnable coefficient directly instead of reading it from the
        `drivers` mapping passed to `solve()`. Raise now, before dispatching, rather than
        return a gradient that is silently wrong (if some other path happens to also touch
        the same value) or silently `None` (if it doesn't).
        """
        for el in (*self._elements, *self._node_sources):
            for name, value in vars(el).items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raise ValueError(
                        f"element {el!r} (kind {getattr(el, 'kind', 'node source')!r}) has a "
                        f"tensor attribute {name!r} with requires_grad=True that is not a "
                        f"registered nn.Parameter, so the differentiable solve cannot reach "
                        f"it through Function.apply and its gradient would be silently wrong "
                        f"or absent. Construct this element with learnable=True (so {name!r} "
                        f"is registered and reachable via named_parameters()), or use "
                        f"differentiable=False."
                    )
        for drv in self._drives:
            for name, value in vars(drv).items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raise ValueError(
                        f"drive {drv!r} (kind {drv.kind!r}) has a tensor attribute {name!r} "
                        f"with requires_grad=True; a Drive is captured by closure inside the "
                        f"differentiable solve, not threaded through Function.apply, so its "
                        f"gradient would be silently absent. A Drive must read every "
                        f"differentiable quantity from the `drivers` mapping passed to "
                        f"solve() rather than owning it directly, or use "
                        f"differentiable=False."
                    )

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        phi_boundary,
        drivers=None,
        sources=None,
        phi0=None,
        *,
        differentiable=True,
        diagnostics: dict | None = None,
        **newton_kwargs,
    ):
        """Solve for interior potentials and branch flows.

        With `differentiable=True` (the default), gradients flow back to `phi_boundary`,
        `sources`, every value in `drivers`, and every Element's registered `nn.Parameter`s
        (i.e. constructed with `learnable=True`) via the implicit-function adjoint
        (`tellegen.solvers.implicit`). Only tensors reachable one of those ways are threaded
        through `Function.apply`. Contract each Element and Drive must satisfy for
        `differentiable=True` to be safe:

        - An Element's own differentiable state must be a registered parameter
          (`learnable=True`), never a bare tensor held with `requires_grad=True` outside
          `named_parameters()` (the latter is still a supported construction, per
          `Element._param`, and its graph is reachable through this layer's component calls
          -- but it is invisible to the differentiable solve, and `differentiable=False` now
          returns detached tensors, so neither `solve` path carries a gradient to it).
        - A Drive must read every differentiable quantity from the `drivers` mapping passed
          to `solve()` (see `tellegen.drives.Drive`), never hold one of its own as an
          instance attribute.

        Violating either raises `ValueError` naming the offending element/drive and
        attribute before any solve is attempted, rather than silently returning a wrong or
        absent gradient.

        `diagnostics`, when a dict is passed, is filled with this solve's own
        `{"newton_iterations", "linear_iterations", "method", "backend", "converged",
        "residual_norm"}` -- the Newton step count, the per-instance maximum inner-solver
        iteration count (`None` if no linear solve happened), the inner method REQUESTED,
        the inner backend that actually RAN, the per-instance convergence flag and the
        per-instance final residual norm. Passing `None` (the default) changes nothing; the
        dict is an out-parameter rather than an extra return value so that `solve`'s
        `(phi, q)` contract, which every existing caller unpacks, is untouched.

        `"method"` and `"backend"` are deliberately separate. `"method"` is what this call
        asked for -- usually `"auto"`, the layer's default. `"backend"` is one of
        `"sparse_direct"`, `"pcg"`, `"gmres"`, `"direct"`: what `solvers.select.solve`
        resolved that request to, which under `"auto"` depends on runtime predicates the
        caller has no other way to observe (the ensemble size against
        `select._SPARSE_DIRECT_MAX_BATCH`, and whether SciPy is importable at all -- it is
        an optional extra, `pip install tellegen[sparse]`). Without it, an installation
        missing that extra takes the ~4.6x-slower PCG path with nothing saying so; the
        indirect signal is `"linear_iterations"` (1 for a factorisation, ~170 for PCG).
        It is `None` only when no linear solve happened at all.

        `on_failure` (forwarded to `newton` among `newton_kwargs`) is `"raise"` by default:
        a batch that fails to converge within `max_iter` raises, naming the failing
        instances. `"return"` is the explicit, narrow escape hatch of design section 3.2 --
        a calibration loop that would rather inspect or down-weight a failed instance than
        abort -- and it is accepted here under two conditions, because this method returns
        `(phi, q)` tensors with no room for a status:

        - `diagnostics=` must be supplied, so `converged`/`residual_norm` have somewhere to
          go. Without it the status would be silently dropped and a non-converged `phi`
          would be indistinguishable from a converged one; that is refused with a
          `ValueError`.
        - `differentiable=False` is required. On the differentiable path the escape hatch is
          refused outright by `solvers.implicit.implicit_solve`: the adjoint linearises at
          the returned point, and at a non-converged point the gradient is silently wrong.

        The inner linear solver for NEWTON's own iteration is this layer's `linear_solver`
        (set at construction), unless the caller overrides it with an explicit `method=`
        among `newton_kwargs`. That override does NOT reach the initial guess: when `phi0`
        is `None` (the default) it is computed by `linear_init`, which always solves with
        `self.linear_solver` regardless of any `method=` passed to this call -- pass an
        explicit `phi0` instead if the override must apply there too.

        `phi0` is a starting guess and nothing else: it is DETACHED on entry (and the guess
        this method computes for itself when `phi0 is None` is computed under `no_grad`),
        because the implicit adjoint linearises at the converged point and never returns a
        gradient w.r.t. the starting guess -- `solvers.implicit._Implicit.backward` returns
        `None` for it. See the comment at the top of the body for what tracing it cost.

        With `differentiable=False`, the returned `(phi, q)` are ALWAYS DETACHED: the whole
        branch -- the initial guess, the grounding check, the Newton iteration, every inner
        linear solve and the final assemble/flows -- runs under `torch.no_grad()`, whatever
        grad mode the caller is in.

        This branch used to run Newton's closures under ORDINARY autograd, so with a
        `learnable=True` Element (or a grad-requiring `phi_boundary`/`drivers`/`sources`) the
        result carried an UNROLLED graph through the converged iterate. Those gradients were
        real but were never the implicit-function ones `differentiable=True` computes, and
        nothing asked for them. Worse, they made the INNER SOLVER'S CHOICE depend on the
        caller's ambient grad mode: `solvers.select.solve`'s `"auto"` will not hand a
        grad-requiring solve to the non-differentiable sparse-direct backend, so the same
        call factorised through SuperLU from a plain call site and fell back to PCG -- 4.6x
        slower -- from inside `torch.enable_grad()` with a grad-requiring `sources`. A
        backend must not be a function of who is calling. A caller who wants gradients calls
        `differentiable=True`, which is unchanged (`solvers.implicit._Implicit.forward`
        already solved under `no_grad` and takes its gradients from the adjoint).
        """
        drivers = drivers or {}
        newton_kwargs.setdefault("method", self.linear_solver)
        # Name this layer on every error the Newton solve or its INNER linear solves raise.
        # Grounding is certified at phi0, but slopes change between Newton iterates, so an
        # instance can lose it mid-iteration; without this the refusal read "newton:
        # method='auto' refuses to split the batch ..." with no way to tell which layer of a
        # composed model over one network produced it.
        newton_kwargs.setdefault("where", f"PotentialFlowLayer {self.name!r} solve")
        if newton_kwargs.get("on_failure") == "return" and diagnostics is None:
            # `solve` returns (phi, q) tensors; without a diagnostics dict there is nowhere
            # for `converged`/`residual_norm` to go, and a non-converged phi would be
            # indistinguishable from a converged one. Design section 3.2: the escape hatch
            # "is never silent: the result carries the status".
            raise ValueError(
                f"PotentialFlowLayer {self.name!r}: on_failure='return' requires "
                f"diagnostics= so the status is not silently dropped; pass a dict and read "
                f"its 'converged' and 'residual_norm' entries."
            )

        # The initial guess is NOT a differentiable quantity, and neither is the grounding
        # check below. Both are computed under no_grad (and a caller-supplied phi0 is
        # detached) so nothing that produced them is traced into the autograd graph.
        #
        # This is a memory fix, not a numerical one: the converged point is where the
        # implicit-function adjoint linearises, and that point is independent of the guess
        # the iteration started from, so tracing the guess buys no gradient at all. What it
        # costs is everything `linear_init` does -- a whole preconditioned-CG loop, its
        # int64 gather indices and every iterate -- retained until backward(). Measured on
        # the composed model at ensemble 100 (Task 14 review): 2567 MB of the 2571 MB saved
        # per differentiable step came from here; with a detached guess the same step saves
        # 74.7 MB. `linear_init` itself is untouched and stays differentiable for callers
        # who want it directly.
        with torch.no_grad():
            if phi0 is None:
                phi0 = self.linear_init(phi_boundary, drivers, sources)
            else:
                phi0 = phi0.detach()

            # Grounding is certified on the ACTUAL slopes at the point the Newton iteration
            # is about to start from, whatever its source. linear_init's own check sees only
            # its tangent-at-zero slopes, and is not run at all when a caller supplies phi0
            # -- so a supplied phi0 used to bypass grounding entirely, and an element whose
            # slope is dp-dependent (a fan past its shutoff point, whose dflow is exactly
            # zero) could leave the operator singular with nothing to say about it but a
            # Newton non-convergence. Placed BEFORE the differentiable branch so both paths
            # run it unconditionally and identically. It is a CHECK: it raises or it does
            # not, and no tensor it computes reaches the result, so it runs under no_grad
            # too (`dflows` here is each Element's own analytic `dflow`, not the autograd
            # path the differentiable branch builds below).
            phi0_full = self.assemble(phi0, phi_boundary)
            dq0 = self.dflows(phi0_full, drivers)
            self._grounding_check(dq0, where="solve")

        if not differentiable:
            # The WHOLE non-differentiable solve runs under no_grad -- the Newton iteration,
            # every inner linear solve inside it, and the final assemble/flows. Two reasons,
            # the second of which is the one that made this mandatory rather than tidy:
            #
            # 1. There is no legitimate graph to build here. This branch does not use the
            #    implicit adjoint, so any graph it leaves behind is an UNROLLED trace through
            #    the converged Newton iterate -- real gradients, but not the ones
            #    `differentiable=True` computes, retained for a caller who asked for the
            #    non-differentiable path. Task 11 recorded that leak as a pre-existing wart;
            #    this closes it.
            # 2. It made the inner solver's choice depend on the CALLER's ambient grad mode.
            #    `solvers.select.solve`'s "auto" refuses the non-differentiable sparse-direct
            #    backend when grad mode is on and an input requires grad (correctly: it would
            #    silently detach). With the iteration running under whatever mode the caller
            #    happened to be in, the same `solve(differentiable=False)` factorised through
            #    SuperLU from a plain call and fell back to PCG -- 4.6x slower, and a
            #    different code path -- from inside `torch.enable_grad()` with a
            #    grad-requiring `sources`. A solver choice must not be a function of the
            #    caller's context. The guard in `select.solve` stays as the last line of
            #    defence; this removes the condition that was tripping it.
            #
            # `diagnostics` is filled exactly as before (no_grad does not touch it), and the
            # returned `(phi, q)` are now unconditionally detached -- see the docstring.
            with torch.no_grad():

                def residual_fn(x):
                    return self.residual(x, phi_boundary, drivers, sources)

                def operator_fn(x):
                    # A matvec-free A_I diag(dq) A_I^T at the current iterate, instead of the
                    # dense (n_interior, n_interior) einsum layer.jacobian() assembles.
                    # Rebuilt each iteration because dq is what changes; the endpoint/index
                    # tensors it closes over are cached on the layer at construction.
                    phi = self.assemble(x, phi_boundary)
                    dq = self.dflows(phi, drivers)
                    return self._operator_at(dq, self._node_source_slopes(phi, drivers))

                result = newton(residual_fn, operator_fn, phi0, **newton_kwargs)
                if diagnostics is not None:
                    diagnostics["newton_iterations"] = result.iterations
                    diagnostics["linear_iterations"] = result.linear_iterations
                    diagnostics["method"] = newton_kwargs["method"]
                    # `method` is what was REQUESTED, `backend` what RAN. Under "auto" the
                    # two differ: the batch threshold and SciPy's presence decide which
                    # side of `select`'s eligibility table this solve landed on, and
                    # nothing else reports it (final review I5).
                    diagnostics["backend"] = result.backend
                    # The per-instance STATUS, not only the cost. Without these two,
                    # `on_failure="return"` returned a non-converged phi with nothing
                    # anywhere reporting it (final review C2).
                    diagnostics["converged"] = result.converged
                    diagnostics["residual_norm"] = result.residual_norm
                phi = self.assemble(result.x, phi_boundary)
                q = self.flows(phi, drivers)
            return phi, q

        self._check_no_unreachable_differentiable_tensors()

        # Node sources are threaded through 'Function.apply' exactly as elements are: their
        # own nn.Parameters are substituted back in with functional_call, so a learnable
        # demand exponent or required pressure gets a correct implicit-function gradient.
        owners = [*self._elements, *self._node_sources]
        n_el = len(self._elements)
        param_dicts = [dict(owner.named_parameters()) for owner in owners]
        param_names = [list(d.keys()) for d in param_dicts]
        param_tensors = [
            d[name] for d, names in zip(param_dicts, param_names, strict=True) for name in names
        ]
        driver_keys = sorted(drivers.keys())
        driver_tensors = [drivers[k] for k in driver_keys]
        sources_tensor = (
            sources
            if sources is not None
            else torch.zeros(
                phi_boundary.shape[:-1] + (self._n_nodes,),
                dtype=phi_boundary.dtype,
                device=phi_boundary.device,
            )
        )
        all_params = (*param_tensors, *driver_tensors, sources_tensor, phi_boundary)

        def _rebuild(params):
            offset = 0
            rebuilt = []
            for names in param_names:
                d = {name: params[offset + j] for j, name in enumerate(names)}
                rebuilt.append(d)
                offset += len(names)
            drv = dict(zip(driver_keys, params[offset : offset + len(driver_keys)], strict=True))
            offset += len(driver_keys)
            src = params[offset]
            pb = params[offset + 1]
            return rebuilt, drv, src, pb

        def _dp_functional(phi, drv):
            d = self._difference(phi)
            parts = []
            for kind, (start, end) in self._kind_slices.items():
                block = d[..., start:end]
                for driven in self._drives:
                    if driven.kind == kind:
                        block = block + driven(drv)
                parts.append((start, block))
            parts.sort(key=lambda p: p[0])
            return torch.cat([p[1] for p in parts], dim=-1)

        def _flows_functional(phi, drv, rebuilt):
            dp_full = _dp_functional(phi, drv)
            parts = []
            for el, d, (s, e) in zip(
                self._elements, rebuilt[:n_el], self._elem_slices, strict=True
            ):
                parts.append(torch.func.functional_call(el, d, (dp_full[..., s:e], drv)))
            return torch.cat(parts, dim=-1)

        def _node_withdrawal_functional(phi, drv, rebuilt):
            if not self._node_sources:
                return None
            out = None
            for ns, d, pos in zip(
                self._node_sources, rebuilt[n_el:], self._source_positions, strict=True
            ):
                w = torch.func.functional_call(ns, d, (phi[..., ns.nodes], drv))
                zeros = torch.zeros(
                    w.shape[:-1] + (len(self.interior),), dtype=w.dtype, device=w.device
                )
                term = zeros.index_add(-1, pos, w)
                out = term if out is None else out + term
            return out

        def _node_slopes_functional(phi, drv, rebuilt):
            if not self._node_sources:
                return None
            out = None
            for ns, d, pos in zip(
                self._node_sources, rebuilt[n_el:], self._source_positions, strict=True
            ):
                leaf = phi[..., ns.nodes].detach().requires_grad_(True)
                with torch.enable_grad():
                    w = torch.func.functional_call(ns, d, (leaf, drv))
                    (grad,) = torch.autograd.grad(w.sum(), leaf)
                zeros = torch.zeros(
                    grad.shape[:-1] + (len(self.interior),),
                    dtype=grad.dtype,
                    device=grad.device,
                )
                term = zeros.index_add(-1, pos, grad)
                out = term if out is None else out + term
            return out

        def _dp_dependence_error(el, i: int) -> RuntimeError:
            return RuntimeError(
                f"element {i} of kind {el.kind!r} (type {type(el).__name__}) produced a "
                f"flow that does not depend on dp; if that is intended, set "
                f"dp_independent = True on this element's class, otherwise its flow() is "
                f"dropping the autograd graph (e.g. a stray dp.detach())."
            )

        def _dflows_functional(phi, drv, rebuilt):
            dp_full = _dp_functional(phi, drv)
            parts = []
            for i, (el, d, (s, e)) in enumerate(
                zip(self._elements, rebuilt[:n_el], self._elem_slices, strict=True)
            ):
                dp_slice = dp_full[..., s:e].detach().requires_grad_(True)
                # The REDUCTION `flow.sum()` must itself execute inside the enable_grad
                # block, not just the `functional_call` that produces `flow`: this whole
                # method runs under the outer no_grad of Newton's forward solve (Task 8's
                # memory-saving guarantee), and `.sum()` is an ordinary tensor op like any
                # other -- performed under the ambient grad mode at the point it actually
                # runs, regardless of whether its input (`flow`) already carries a grad_fn
                # from an earlier, enable_grad-wrapped computation. Writing
                # `torch.autograd.grad(flow.sum(), dp_slice)` with the `with
                # torch.enable_grad():` block closed before that line (as an earlier,
                # incorrect version of this code did) evaluates `flow.sum()` under the outer
                # no_grad, so the tensor actually handed to `autograd.grad` as `outputs` has
                # no grad_fn of its own -- even though `flow` printed `requires_grad=True`
                # right before the call. `torch.autograd.grad` itself does not consult the
                # ambient grad mode at its own call site (confirmed: calling it under no_grad
                # against an output already fully built under enable_grad, or with an explicit
                # `grad_outputs=` and no further reduction, both work); the reduction is what
                # must be inside the block.
                with torch.enable_grad():
                    flow = torch.func.functional_call(el, d, (dp_slice, drv))
                    # Whether a missing/zero Jacobian contribution here is legitimate is
                    # decided ONLY by el.dp_independent -- a class-level DECLARATION (see
                    # Element) -- never inferred from whatever autograd graph `flow` happens
                    # to carry. A previous version of this code inferred it from
                    # `flow.requires_grad` (with allow_unused=True as a catch-all), which
                    # meant ANY element whose flow() accidentally lost the autograd graph
                    # (e.g. a stray dp.detach() in a third-party subclass) silently got an
                    # exact-zero Jacobian column instead of an error: the forward solve still
                    # looked fine (Newton converges on the exact residual regardless), but
                    # the adjoint gradient computed from that Jacobian was silently wrong.
                    if el.dp_independent:
                        # Cross-check the declaration against the element's own analytic
                        # dflow(), which for a genuinely dp-independent law (FixedFlow's
                        # spec) must be identically zero. This is called on `el` directly
                        # (not through functional_call/`d`) rather than via a second
                        # functional_call: torch.func.functional_call always invokes the
                        # module's forward()/flow(), with no way to redirect it to dflow(),
                        # and `d`'s tensors are the very same objects as el's own current
                        # parameters (both trace back to el.named_parameters() at the start
                        # of solve()), so reading el's own attributes here gives the same
                        # values a substituted call would. This catches a wrongly-declared
                        # dp_independent = True (or a dflow() inconsistent with it) instead
                        # of silently trusting a possibly-wrong flag.
                        analytic = el.dflow(dp_slice.detach(), drv)
                        if not torch.equal(analytic, torch.zeros_like(analytic)):
                            raise RuntimeError(
                                f"element {i} of kind {el.kind!r} (type "
                                f"{type(el).__name__}) declares dp_independent = True but "
                                f"its own dflow() is not identically zero; either the flag "
                                f"is wrong or dflow() is inconsistent with a dp-independent "
                                f"flow law."
                            )
                        grad = torch.zeros_like(dp_slice)
                    elif not flow.requires_grad:
                        raise _dp_dependence_error(el, i)
                    else:
                        try:
                            (grad,) = torch.autograd.grad(
                                flow.sum(), dp_slice, create_graph=False
                            )
                        except RuntimeError as exc:
                            raise _dp_dependence_error(el, i) from exc
                parts.append(grad)
            return torch.cat(parts, dim=-1)

        def residual_fn(x, *params):
            rebuilt, drv, src, pb = _rebuild(params)
            phi = self.assemble(x, pb)
            q = _flows_functional(phi, drv, rebuilt)
            lhs = self._accumulate_interior(q)
            s_I = src[..., self.interior]
            w = _node_withdrawal_functional(phi, drv, rebuilt)
            return lhs - s_I if w is None else lhs - s_I + w

        def operator_fn(x, *params):
            rebuilt, drv, src, pb = _rebuild(params)
            phi = self.assemble(x, pb)
            dq = _dflows_functional(phi, drv, rebuilt)
            return self._operator_at(dq, _node_slopes_functional(phi, drv, rebuilt))

        x = implicit_solve(
            residual_fn,
            operator_fn,
            phi0,
            all_params,
            diagnostics=diagnostics,
            **newton_kwargs,
        )
        if diagnostics is not None:
            diagnostics["method"] = newton_kwargs["method"]
        phi = self.assemble(x, phi_boundary)
        q = self.flows(phi, drivers)
        return phi, q

    def adjoint(self, phi_interior, phi_boundary, drivers, grad_phi_interior):
        """Solve J(phi)^T lambda = grad_phi_interior at the given point.

        The operator is the SAME GraphLaplacianOperator `solve`'s Newton iteration builds
        (same endpoints, same `dflows` slopes), never the dense einsum `jacobian()` -- so
        the adjoint costs one matvec-free transposed solve rather than an (n_I, n_I)
        materialisation, and cannot drift from the forward path's own operator. The
        transposed action comes from the operator's `rmatvec` via
        `solvers.implicit.TransposeOperator`; for this symmetric Laplacian that equals its
        `matvec`, but nothing here assumes it. `method=self.linear_solver` carries the
        layer's configured inner solver onto the backward pass too (amendment A3.3), so
        `linear_solver="direct"` is the retained milestone-1 numerics on BOTH passes.

        CALLED DIRECTLY UNDER GRAD MODE with grad-requiring `drivers` (or a grad-requiring
        `phi_interior`/`phi_boundary`), `linear_solver="auto"` falls back to PCG here rather
        than factorising: `solvers.select.solve` will not hand a grad-requiring solve to the
        non-differentiable sparse-direct backend, and an explicit
        `linear_solver="sparse_direct"` raises outright in that situation. Every in-repo
        caller reaches this method under `no_grad` -- `solvers.implicit._Implicit.backward`
        is the only one on the hot path -- so this is a direct-caller's concern, stated here
        for the same reason `linear_init`'s equivalent is ("stays differentiable for callers
        who want it directly"). Wrap the call in `torch.no_grad()` to get the factorisation.
        """
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self.dflows(phi, drivers)
        op = self._operator_at(dq, self._node_source_slopes(phi, drivers))
        return _adjoint_solve(
            op,
            grad_phi_interior,
            where=f"PotentialFlowLayer {self.name!r} adjoint",
            method=self.linear_solver,
        )

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
        drive_only = d - self._difference(phi)
        boundary_flow = self._accumulate_bound(q)
        phi_b = phi[..., self.bound]
        phi_i = phi[..., self.interior]
        s_I = self._source_interior(sources, phi_i)
        return (
            (d * q).sum(-1)
            - (drive_only * q).sum(-1)
            - (phi_b * boundary_flow).sum(-1)
            - (phi_i * s_I).sum(-1)
        )
