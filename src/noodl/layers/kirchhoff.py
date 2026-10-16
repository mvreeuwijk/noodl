"""KirchhoffLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + drive; theta) - s_I from a set of constitutive laws
on typed edges (one ConstitutiveLaw per kind) and Drive terms (additive potential differences).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

import torch

from noodl._broadcast import broadcast_shapes
from noodl._tracking import bulk_values
from noodl.drives import Drive, check_drive_signature
from noodl.elements.base import ConstitutiveLaw
from noodl.layers._functional import FunctionalLaws
from noodl.layers._mna import MnaSystem
from noodl.layers._solve_policy import resolve_classic, resolve_mna
from noodl.layers._topology import TopologyGuard
from noodl.nodesources import NodeSource
from noodl.operators.graph import GraphLaplacianOperator
from noodl.solvers.grounding import spd_certificate, spd_diagnosis
from noodl.solvers.implicit import adjoint as _adjoint_solve
from noodl.solvers.implicit import implicit_solve
from noodl.solvers.newton import inner_solve_rtol, newton
from noodl.solvers.select import solve as select_solve
from noodl.topology import Network

# Inner linear solvers a layer may be configured with; forwarded verbatim as
# `solvers.select.solve`'s `method`. "direct" is the retained dense reference (the
# operator's explicit A_I diag(g) A_I^T, LU-factorised); "auto" is the default;
# "sparse_direct" is the SciPy SuperLU reference, which factorises the
# operator's O(E) COO form per instance instead. All four reach BOTH passes: `linear_init`
# and every Newton inner solve on the forward, and the implicit adjoint on the backward.
#
# What "auto" resolves to on THIS layer's operator (a certified-SPD GraphLaplacianOperator
# that declares `assemble_sparse`) is sparse-direct up to an ensemble of
# `select._SPARSE_DIRECT_MAX_BATCH` = 32 instances and Jacobi-preconditioned CG above it,
# because SuperLU is driven by a per-instance Python loop whose cost is linear in the
# ensemble while PCG's is sub-linear (crossover sweep: docs/design/solvers.md). An
# explicit `linear_solver=` is honoured verbatim at every ensemble size. `diagnostics`
# reports which backend actually ran (see `solve`).
_LINEAR_SOLVERS = ("auto", "cg", "gmres", "direct", "sparse_direct")


class _DiagonalShifted:
    """`A_I diag(g) A_I^T + diag(d)`: the Jacobian of a layer carrying potential-dependent
    NODAL sources, satisfying the LinearOperator duck type
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
        self.shape = broadcast_shapes(base.shape[:-2], diag.shape[:-1]) + base.shape[-2:]
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
        # A strictly positive shift grounds its node (`KirchhoffLayer._extra_grounded`,
        # the rule the layer's own grounding check applies): a network with no boundary path
        # held only by its node sources (a closed building of compressible volumes) is SPD.
        b = self.base
        idx = b.interior_of_node
        interior = idx >= 0
        positive = self.diag_shift > 0
        extra = torch.zeros(positive.shape[:-1] + idx.shape, dtype=torch.bool,
                            device=positive.device)
        extra[..., interior] = positive[..., idx[interior]]
        return spd_certificate(b.src, b.tgt, b.slopes, idx, b.boundary_mask,
                               extra_grounded=extra)


def _held_tensors(obj):
    """Yield `(name, tensor)` for every tensor `obj` holds: its attributes, its registered
    buffers (when a module) and tensors nested at any depth in lists, tuples, dicts and sets.
    A module's registered parameters live in `_parameters` and are not yielded; a parameter
    held in a container or a plain attribute is, since the solve cannot reach it."""
    def walk(label, value, depth):
        if isinstance(value, torch.Tensor):
            yield label, value
        elif depth < 8 and isinstance(value, dict):
            for k, v in value.items():
                yield from walk(f"{label}[{k!r}]", v, depth + 1)
        elif depth < 8 and isinstance(value, (list, tuple, set, frozenset)):
            for i, v in enumerate(value):
                yield from walk(f"{label}[{i}]", v, depth + 1)

    for name, value in vars(obj).items():
        if name in ("_parameters", "_buffers", "_modules", "_memo_cache"):
            continue
        yield from walk(name, value, 0)
    if isinstance(obj, torch.nn.Module):
        for name, value in obj._buffers.items():
            if value is not None:
                yield from walk(name, value, 0)


class KirchhoffLayer(TopologyGuard):
    """Flows driven by a potential (pressure, head): solves nodal conservation
    `A_I g(A^T phi) = s_I` for the interior potentials by Newton's method, with the constitutive
    laws on its edge kinds, drives added to the potential differences, and prescribed
    potentials at its boundary nodes; differentiable through the converged solution.
    `flow_unit` is reported by `model.refs`, never used numerically.

    A layer whose constitutive laws include an implicit law (`ImplicitLaw`: an ideal source,
    a flow-controlled law), a coupled multi-port law, a dynamic law or a law reading the
    `LawContext` is solved by Modified Nodal Analysis (`mna` is True): the flows of its
    implicit-law edges (`implicit_edges`) become unknowns alongside the interior potentials,
    and every connected part of its edges must contain a boundary node (a ground).

    With `phasor=True` the layer solves the sinusoidal steady state: potentials and flows are
    complex amplitudes (complex128), the frequency in Hz comes from the driver
    `"<name>.frequency"` (one per instance) and reaches the laws as `LawContext.omega`
    (`2 pi f`). Each complex unknown is solved as its (real, imaginary) pair, so a law that is
    not holomorphic in `dp` (a constant-power load, `q = conj(S / dp)`) is handled exactly.
    Every law must declare `phasor_capable`, node sources are refused (a load is an edge to
    ground), and `power_residual` is the complex power balance.

    `analytic_primal` (default ``False``): a ``differentiable=True`` solve that nothing
    differentiable reaches (grad mode off, or no input requiring grad) runs the
    ``differentiable=False`` branch, whose Newton Jacobian is the laws' and node
    sources' own analytic ``dflow`` rather than autograd through ``functional_call``. The
    result is the same root to the Newton tolerance (not bitwise: the two Jacobians may
    differ at a kink), returned detached as it would be anyway. It is opt-in because it
    trusts every ``dflow`` to be the derivative of its ``flow`` (the sewer dynamic wave sets
    it: its laws' ``dflow`` are exact). It has no effect on a layer solved by Modified Nodal
    Analysis, whose solve already assembles the analytic Jacobian.

    `newton_defaults` (default none) are Newton keywords (`solvers.newton.newton`) every
    `solve` of this layer uses unless the call gives its own, for laws that need them: the
    sewer dynamic wave sets ``line_search="convex"`` and ``step_floor`` (its kinked laws are
    each non-decreasing, so the residual is the gradient of a convex function). A layer
    solved by Modified Nodal Analysis refuses ``line_search="convex"``, by keyword or here: its
    residual is not the gradient of a convex function."""

    def __init__(
        self,
        net: Network,
        name: str,
        laws: Sequence[ConstitutiveLaw],
        drives: Sequence[Drive] = (),
        boundary: Sequence = (),
        linear_solver: str = "auto",
        *,
        node_sources: Sequence[NodeSource] = (),
        quantity: str = "potential",
        unit: str = "",
        flow_unit: str = "",
        phasor: bool = False,
        analytic_primal: bool = False,
        newton_defaults: Mapping | None = None,
    ) -> None:
        self.analytic_primal = bool(analytic_primal)
        self.newton_defaults = dict(newton_defaults or {})
        if linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"unknown linear_solver {linear_solver!r} in layer {name!r}; expected one "
                f"of {_LINEAR_SOLVERS}"
            )
        self.linear_solver = linear_solver
        # Metadata only: what this layer's potential IS and what it is measured in.
        # Nothing in the numerics reads them; `Model` reports them, and a caller composing
        # several layers over one network uses them to tell a pressure layer from a
        # temperature one without matching on layer names.
        self.quantity = quantity
        self.unit = unit
        # Metadata only, like `quantity`/`unit`: the unit of the branch flows and nodal
        # sources this layer solves (kg/s for air, m3/s for water). `noodl.refs` reports it.
        self.flow_unit = flow_unit

        seen_kinds: set[str] = set()
        for el in laws:
            if el.kind in seen_kinds:
                raise ValueError(f"duplicate law kind {el.kind!r} in layer {name!r}")
            seen_kinds.add(el.kind)
            if el.law_form not in ("potential", "implicit"):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: kind {el.kind!r} has law "
                    f"{type(el).__name__} with law_form={el.law_form!r}; expected "
                    f"'potential' or 'implicit'"
                )

        self.net = net
        self._record_topology(net)
        self.name = name
        self._laws = list(laws)
        self._drives = list(drives)
        self.kinds = [el.kind for el in laws]
        # Modified Nodal Analysis: a law whose flow is an unknown (implicit), that couples
        # several edges (ports > 1), that stores (dynamic) or that reads the law context
        # cannot be solved by the classic nodal path; such a layer is solved by `MnaSystem`.
        self.phasor = phasor
        self.dynamic = any(el.dynamic for el in self._laws)
        self.mna = self.phasor or any(
            el.law_form == "implicit" or el.ports > 1 or el.dynamic or el.contextual
            for el in self._laws
        )
        # Node name -> a likely cause, appended when that node is reported floating (an
        # application's builder fills it, e.g. for a node only prescribed flows touch).
        self.node_hints: dict = {}

        cols_list = []
        self._kind_slices: dict[str, tuple[int, int]] = {}
        self._law_slices: list[tuple[int, int]] = []
        offset = 0
        for el in laws:
            # net.edge_index raises KeyError when el.kind carries no edge at all (it never
            # returns an empty tensor for a non-None kind: a kind either has edges, in which
            # case idx is non-empty, or it appears nowhere in the network, in which case
            # edge_index itself raises). Re-raise as ValueError so a missing law kind and
            # a genuinely unknown node/kind lookup elsewhere stay distinguishable to callers.
            try:
                idx = net.edge_index(el.kind)
            except KeyError as exc:
                raise ValueError(
                    f"law kind {el.kind!r} has no edges in the network"
                ) from exc
            cols_list.append(idx)
            n_e = idx.numel()
            self._kind_slices[el.kind] = (offset, offset + n_e)
            self._law_slices.append((offset, offset + n_e))
            offset += n_e
        self.cols = torch.cat(cols_list)
        # The node count, cached as a plain int: reading it off self.A.shape[0] would be an
        # incidental dense-matrix access that keeps an (n, b) tensor alive.
        self._n_nodes = net.n

        # This layer's own edge endpoints (restricted to self.cols, in the same order as
        # the columns of the (lazy) self.A / rows of self._diff), and a node -> interior-
        # position map, both needed to construct a GraphLaplacianOperator (and to run the
        # per-instance SPD certificate) without a per-solve Python loop. net.endpoints()
        # (kind=None) returns whole-graph (src, tgt) arrays in network edge order; indexing
        # by self.cols restricts them to this layer's own edges, exactly as the lazy
        # self.A == net.incidence()[:, self.cols] does for the incidence matrix. These
        # ARE the layer's representation of its own topology: every hot-path
        # site gathers or scatter-adds with them instead of contracting against A/_diff.
        src_all, tgt_all = net.endpoints()
        self._src = src_all[self.cols]
        self._tgt = tgt_all[self.cols]

        kind_set = set(self.kinds)
        for drv in self._drives:
            check_drive_signature(drv, where=f"KirchhoffLayer {name!r}")
            if drv.kind not in kind_set:
                raise ValueError(
                    f"drive kind {drv.kind!r} is not one of this layer's law kinds "
                    f"{sorted(kind_set)} (layer {name!r})"
                )

        node_index = {node: i for i, node in enumerate(net.nodes)}
        for b in boundary:
            if net._resolve(b) not in node_index:
                raise KeyError(f"unknown boundary node {b!r} in layer {name!r}")
        boundary = net.canonical(boundary)

        touched = torch.zeros(net.n, dtype=torch.bool)
        touched[self._src] = True
        touched[self._tgt] = True
        # A node a node source acts at is active too: its row is the source alone (a
        # junction no conduit reaches still has its storage, `sewer.dynwave.JunctionStorage`).
        for ns in node_sources:
            if isinstance(ns, NodeSource):
                touched[ns.nodes] = True
        all_interior = net.interior_index(boundary)
        # A node no edge of this layer's kinds touches is INACTIVE for this
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

        # POTENTIAL-DEPENDENT NODAL SOURCES. Empty by default, so every
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
                    f"KirchhoffLayer {name!r}: node_sources entry "
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
                    f"KirchhoffLayer {name!r}: node source {ns!r} acts at {bad}, "
                    f"which are not interior unknowns of this layer (a boundary node's "
                    f"potential is prescribed and an inactive node has no row)"
                )
            self._source_positions.append(pos)

        self._mna: MnaSystem | None = None
        if self.mna:
            self._check_mna_structure()
            self._mna = MnaSystem(self)

    def _check_mna_structure(self) -> None:
        """Construction checks of a Modified Nodal Analysis layer: port groups, solver,
        drives on dynamic laws, and a boundary node in every connected part."""
        name = self.name
        for el, (s, e) in zip(self._laws, self._law_slices, strict=True):
            if (e - s) % el.ports:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: kind {el.kind!r} has {e - s} edges, not a "
                    f"multiple of its law's ports={el.ports}"
                )
        if self.linear_solver == "cg":
            raise ValueError(
                f"KirchhoffLayer {name!r}: linear_solver='cg' needs a symmetric positive "
                f"definite Jacobian, which a Modified Nodal Analysis system is not; use "
                f"'auto', 'sparse_direct', 'direct' or 'gmres'"
            )
        if self.phasor:
            for el in self._laws:
                if not el.phasor_capable:
                    raise ValueError(
                        f"KirchhoffLayer {name!r}: kind {el.kind!r} has law "
                        f"{type(el).__name__}, which is not phasor-capable (a phasor layer "
                        f"takes linear laws with real or complex coefficients and laws that "
                        f"declare phasor_capable)"
                    )
            if self._node_sources:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a phasor layer takes no node_sources; "
                    f"loads are edges to ground in a phasor layer"
                )
        dynamic_kinds = {el.kind for el in self._laws if el.dynamic}
        for drv in self._drives:
            if drv.kind in dynamic_kinds:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: kind {drv.kind!r} has a dynamic law, which "
                    f"may not carry a drive (its step-start values are the potential "
                    f"differences without one)"
                )
        # Every connected part of this layer's edges must hold a boundary node: the
        # potential of a part with none is defined only up to a constant.
        parent = list(range(self._n_nodes))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for u, v in zip(self._src.tolist(), self._tgt.tolist(), strict=True):
            parent[find(v)] = find(u)
        grounded = {find(i) for i in self.bound.tolist()}
        floating = [
            self.net.nodes[i] for i in self.interior.tolist() if find(i) not in grounded
        ]
        if floating:
            raise ValueError(
                f"KirchhoffLayer {name!r}: nodes {floating} have no path through this "
                f"layer's edges to a boundary node, so their potentials are undetermined; "
                f"add a node of each such part to boundary (a ground)"
            )

    @property
    def implicit_edges(self) -> torch.Tensor:
        """Positions, in layer edge order (the columns of `solve`'s `q`), of the edges whose
        flow is an unknown of the layer (implicit constitutive laws); empty for a classic
        layer."""
        if self._mna is None:
            return torch.zeros(0, dtype=torch.long)
        return self._mna.f_edges

    @property
    def frequency_key(self) -> str | None:
        """The driver key a phasor layer reads its frequency from (`"<name>.frequency"`, in
        Hz, one value per instance); `None` for a layer that is not a phasor layer."""
        return self._mna.frequency_key if self.phasor else None

    def _refuse_on_mna(self, method: str) -> None:
        """Classic-only methods assume every flow is a function of its own edge's potential
        difference; a Modified Nodal Analysis layer refuses them by name."""
        if self.mna:
            raise TypeError(
                f"KirchhoffLayer {self.name!r} is solved by Modified Nodal Analysis "
                f"(its laws include implicit, multi-port, dynamic or context-reading ones), "
                f"so {method}() is not defined on it; read the flows from solve()'s q"
            )

    @property
    def node_sources(self) -> tuple:
        """The `NodeSource` objects this layer evaluates at its interior nodes, in the order
        given at construction (read-only)."""
        return tuple(self._node_sources)

    # ------------------------------------------------------- dense references (lazy)
    @functools.cached_property
    def A(self) -> torch.Tensor:
        """This layer's (n, b_layer) incidence matrix, built on FIRST ACCESS only.

        Held as a `cached_property` rather than an `__init__` attribute: it grows 4x per node
        doubling and would break the composed model's memory-scaling budget (numbers:
        docs/design/orchestration.md#dense-incidence-matrix). Nothing inside this class reads
        it except `jacobian()`, the retained dense reference -- every hot-path site gathers or
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

        Like `A`, cached on first access: both assume `net` is not
        mutated after this layer is constructed. Nothing in this codebase does that -- a
        `Network` is built, then layers are constructed over it, then it is solved -- but a
        caller who mutated `net` afterwards (adding a node or edge) would get a layer whose
        `A`/`_diff` (if already accessed) or `_src`/`_tgt`/`cols` (fixed at `__init__`) no
        longer agree with the network's current topology; `solve`, `dp`, `assemble` and
        `residual` raise (`_require_current_topology`) once the network has changed.
        """
        return self.net.difference()[self.cols]

    @property
    def linear(self) -> bool:
        """True when every law and node source is affine (`ConstitutiveLaw.linear`): an MNA
        solve then takes the full Newton step from the start and converges in one
        iteration."""
        return (all(el.linear for el in self._laws)
                and all(ns.linear for ns in self._node_sources))

    def net_outflow(self, q: torch.Tensor) -> torch.Tensor:
        """The net flow OUT of the network at every node, `(..., n_nodes)`, from the edge
        flows `q` (`(..., b)`, in this layer's edge order): each edge's flow leaves its
        source node and enters its target node. Interior rows are the KCL imbalance (zero
        at a converged solve without sources); boundary rows are what the boundary
        supplies, negated (the inflow from the network is `-net_outflow(q)[..., bound]`).
        Differentiable in `q`."""
        return self._accumulate(q)

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

    def _scatter_node_sources(self, value) -> torch.Tensor | None:
        """The sum over node sources of `value(j, ns)` (one value per node of source `j`),
        scattered to the INTERIOR rows; `None` when this layer has no node sources. Every
        node-source withdrawal and slope (classic, threaded and Modified Nodal Analysis)
        is assembled here."""
        if not self._node_sources:
            return None
        out = None
        for j, (ns, pos) in enumerate(
                zip(self._node_sources, self._source_positions, strict=True)):
            v = value(j, ns)
            zeros = torch.zeros(
                v.shape[:-1] + (len(self.interior),), dtype=v.dtype, device=v.device
            )
            term = zeros.index_add(-1, pos, v)
            out = term if out is None else out + term
        return out

    def _node_source_withdrawal(
        self, phi: torch.Tensor, drivers: Mapping
    ) -> torch.Tensor | None:
        """Total potential-dependent withdrawal per INTERIOR row, or `None` when this layer
        has no node sources. `phi` is full-node."""
        return self._scatter_node_sources(lambda j, ns: ns.flow(phi[..., ns.nodes], drivers))

    def _node_source_slopes(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor | None:
        """`diag(w')` per INTERIOR row, or `None` when this layer has no node sources."""
        return self._scatter_node_sources(lambda j, ns: ns.dflow(phi[..., ns.nodes], drivers))

    def _operator_at(
        self, slopes: torch.Tensor, node_slopes: torch.Tensor | None = None
    ) -> GraphLaplacianOperator | _DiagonalShifted:
        """The matvec-free `A_I diag(slopes) A_I^T` operator at the given per-edge slope, in
        this layer's own endpoint/interior-index representation -- the construction repeated
        by `linear_init`, `solve`'s `operator_fn` (both the non-differentiable and
        differentiable branches) and `adjoint`. `slopes` is `dflows`'s actual Jacobian
        diagonal at the current iterate for all but `linear_init`, which instead passes its
        own tangent-at-zero `k` -- the same operator shape, at a different slope.

        `node_slopes` is the `diag(w')` a potential-dependent nodal source adds to
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
        self._require_current_topology()
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
                    # slice out from under `_law_slices` so a DIFFERENT law ends up being
                    # fed this kind's drive value with no exception anywhere in the call chain.
                    if value.ndim > 0 and value.shape[-1] != width:
                        raise ValueError(
                            f"drive kind {drv.kind!r} returned width {value.shape[-1]} but "
                            f"the layer's {drv.kind!r} block has {width} edges"
                        )
                    block = block + value
            parts.append((start, block))
        parts.sort(key=lambda p: p[0])
        # A batched drive on one kind batches that kind's block only; every block is
        # broadcast to the common batch shape before they are joined.
        batch = broadcast_shapes(*(b.shape[:-1] for _, b in parts))
        return torch.cat([b.expand(batch + b.shape[-1:]) for _, b in parts], dim=-1)

    def flows(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        self._refuse_on_mna("flows")
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        parts = [
            el.flow(d[..., s:e], drivers)
            for el, (s, e) in zip(self._laws, self._law_slices, strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def kind_slice(self, kind: str) -> slice:
        """Columns of `q` (as returned by `solve`) that belong to `kind`."""
        try:
            start, end = self._kind_slices[kind]
        except KeyError as exc:
            raise KeyError(
                f"KirchhoffLayer {self.name!r} has no law of kind {kind!r}; its kinds "
                f"are {self.kinds}"
            ) from exc
        return slice(start, end)

    def law_for(self, kind: str) -> tuple[ConstitutiveLaw, slice]:
        """This layer's own `ConstitutiveLaw` instance for `kind`, and the slice of `q`'s (and
        `dp`'s) columns it occupies: the PUBLIC accessor a caller outside this layer
        uses instead of reaching into its private `_laws`/`_law_slices`. Raises
        `KeyError` naming the kind and this layer's known kinds when absent, exactly as
        `kind_slice` does (each law kind is unique within a layer, so the two agree by
        construction).
        """
        sl = self.kind_slice(kind)
        return self._laws[self.kinds.index(kind)], sl

    def flows_of_kind(self, q: torch.Tensor, kinds) -> torch.Tensor:
        """`q` restricted to `kinds` (a name or a sequence), concatenated in the given order.

        This is what a `TransportLayer` with `flow_kinds == kinds` expects as its `q`: the
        transport layer reads its flow edges kind by kind, in ITS OWN `flow_kinds` order,
        which need not be this layer's law order. `Model` couples the two layers with
        exactly this call.
        """
        if isinstance(kinds, str):
            kinds = (kinds,)
        return torch.cat([q[..., self.kind_slice(k)] for k in kinds], dim=-1)

    def dflows(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        self._refuse_on_mna("dflows")
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        parts = [
            el.dflow(d[..., s:e], drivers)
            for el, (s, e) in zip(self._laws, self._law_slices, strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def assemble(self, phi_interior: torch.Tensor, phi_boundary: torch.Tensor) -> torch.Tensor:
        self._require_current_topology()
        batch_shape = broadcast_shapes(
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
                    f"KirchhoffLayer {self.name!r}: sources must be zero on inactive "
                    f"nodes (no edge of kinds {self.kinds} touches them); nonzero at {names}"
                )
        return sources[..., self.interior]

    def residual(self, phi_interior, phi_boundary, drivers, sources):
        self._refuse_on_mna("residual")
        self._require_current_topology()
        return self._residual_parts(phi_interior, phi_boundary, drivers or {}, sources)[0]

    def balance(self, phi, q, sources=None, drivers=None):
        """The interior flow balance `A_I q - s_I (+ w(phi))` of a state's own flows.

        `phi` is full-node, `q` every edge's flow in layer edge order (what `solve`
        returns), `sources` full-node (`None` means zero) and `drivers` what the node
        sources read. Zero (to solver tolerance) at a solved state. Unlike `residual`, which
        recomputes the flows from `phi`, this takes them as given, so it is defined on every
        layer, including one solved by Modified Nodal Analysis, whose implicit-law flows are
        not functions of `phi`.
        """
        phi_i = phi[..., self.interior]
        lhs = self._accumulate_interior(q) - self._source_interior(sources, phi_i)
        w = self._node_source_withdrawal(phi, drivers or {})
        return lhs if w is None else lhs + w

    def _residual_parts(self, phi_interior, phi_boundary, drivers, sources):
        """`(residual, phi, q)`: the residual and the full-node potentials and edge flows it
        was evaluated at (`solve` reuses the last two instead of recomputing them)."""
        phi = self.assemble(phi_interior, phi_boundary)
        q = self.flows(phi, drivers)
        # (A_I q), by scatter-add over this layer's edges then a select of the interior
        # rows, rather than einsum against the (n_I, b) slice of the dense incidence: this
        # is once per Newton residual evaluation, where the einsum form dominates the cost
        # (numbers: docs/design/orchestration.md#residual-incidence-product).
        lhs = self._accumulate_interior(q)
        s_I = self._source_interior(sources, phi_interior)
        w = self._node_source_withdrawal(phi, drivers)
        return (lhs - s_I if w is None else lhs - s_I + w), phi, q

    def jacobian(self, phi_interior, phi_boundary, drivers):
        """The dense (n_I, n_I) Jacobian A_I diag(dq) A_I^T -- the retained dense reference.

        This is the one method that reads `self.A` (and so materialises it, once, on first
        access); no solve path calls it. Tests compare `GraphLaplacianOperator`'s matvec and
        the adjoint against this, so its einsum form is deliberately unchanged.
        """
        self._refuse_on_mna("jacobian")
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
        # Each law's own (c_e, k_e) covers only its own n_e = e - s edges; only the
        # leading BATCH dims (everything but the trailing edge-count dim) are meant to be
        # unified across laws before concatenating along the edge axis. Using
        # torch.broadcast_tensors directly on the raw (c_e, k_e) list would broadcast the
        # trailing edge dim too, so two laws with different edge counts silently expand to
        # a shared (wrong) edge count instead of concatenating -- e.g. a 2-edge FixedFlow and
        # a 1-edge PowerLaw would both become
        # 2-edge before torch.cat, yielding a 4-wide result instead of the correct 3.
        cs, ks, n_es = [], [], []
        for el, (s, e) in zip(self._laws, self._law_slices, strict=True):
            c_e, k_e = el.linear_init(drivers)
            cs.append(c_e)
            ks.append(k_e)
            n_es.append(e - s)
        batch_shape = broadcast_shapes(
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

    def _extra_grounded(self, node_slopes: torch.Tensor | None) -> torch.Tensor | None:
        """Full-node-order bool from `node_slopes` (per COMPACT interior row, as
        `_node_source_slopes` returns): True at an interior node whose own node-source slope
        is strictly positive there. Such a node is grounded independent of any edge
        path -- `_DiagonalShifted`'s positive diagonal shift there is exactly the SPD
        contribution a boundary connection would otherwise have to supply -- so
        `_grounding_check` must treat it exactly like a boundary node when deciding whether
        the rest of the graph reaches ground. `None` when this layer has no node sources,
        which leaves `_grounding_check` unchanged for every existing layer.
        """
        if node_slopes is None:
            return None
        positive = node_slopes > 0
        out = torch.zeros(
            positive.shape[:-1] + (self._n_nodes,), dtype=torch.bool, device=positive.device
        )
        return out.index_copy(-1, self.interior, positive)

    def _grounding_check(
        self, slopes: torch.Tensor, *, where: str, node_slopes: torch.Tensor | None = None
    ) -> None:
        """Raise unless every instance in `slopes` certifies SPD.

        It is run on the slopes GIVEN -- the caller decides whether those are
        `linear_init`'s tangent-at-zero slopes or the actual `dflows` at a solve point -- and
        it is per instance, so an edge closed in one instance but open in another cannot be
        conflated with it.

        `node_slopes`, when this layer has node sources, is `diag(w')`
        per compact interior row -- the same tensor `_node_source_slopes` returns. A node
        with a strictly positive entry there is grounded independent of any edge path (see
        `_extra_grounded`), so passing it here is what keeps a node grounded ONLY through a
        node source from being reported ungrounded.
        """
        extra_grounded = self._extra_grounded(node_slopes)
        certified = spd_certificate(
            self._src, self._tgt, slopes, self._interior_of_node, self._boundary_mask,
            atol=0.0, extra_grounded=extra_grounded,
        )
        if bool(torch.all(certified)):
            return
        records = spd_diagnosis(
            self._src, self._tgt, slopes, self._interior_of_node, self._boundary_mask,
            atol=0.0, extra_grounded=extra_grounded,
        )
        if certified.ndim == 0:
            # Unbatched: one instance, so a bare batch index would say nothing. Name the
            # nodes (or edges) directly.
            raise RuntimeError(
                f"KirchhoffLayer {self.name!r}: {where}: "
                f"{self._describe_grounding(records[0])}"
            )
        bad_idx = torch.nonzero(~certified.reshape(-1), as_tuple=False).flatten().tolist()
        details = "; ".join(
            f"instance {rec['instance']}: {self._describe_grounding(rec)}" for rec in records
        )
        raise RuntimeError(
            f"KirchhoffLayer {self.name!r}: {where}: batch indices {bad_idx} do not "
            f"certify a grounded, positive-slope system; {details}"
        )

    def _describe_grounding(self, record: dict) -> str:
        """One failing instance's `spd_diagnosis` record, with node INDICES resolved to the
        network's own node names (which the operator-level diagnosis cannot know)."""
        if record["reason"] == "negative_slope":
            return f"negative slope on edges {record['edges']}"
        names = [self.net.nodes[i] for i in record["nodes"]]
        hints = sorted({self.node_hints[n] for n in names if n in self.node_hints})
        return (
            f"floating nodes with no path to a boundary potential: "
            f"ungrounded interior nodes {names}"
            + "".join(f" ({h})" for h in hints)
        )

    def linear_init(self, phi_boundary, drivers, sources):
        self._refuse_on_mna("linear_init")
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
        # the operator's diagonal.
        node_slopes = None
        if self._node_sources:
            w0 = self._node_source_withdrawal(phi0, drivers)
            node_slopes = self._node_source_slopes(phi0, drivers)
            rhs = rhs - w0
        self._grounding_check(k, where="linear_init", node_slopes=node_slopes)
        # k picks up a batch dimension only if some law's linear_init(drivers) actually
        # depends on drivers (e.g. a driver-conditioned slope); with purely constant slopes
        # k stays unbatched (b,) even when `rhs` is batched via phi_boundary/sources.
        # Broadcasting both to their common batch shape first keeps the operator's own batch
        # shape and the right-hand side's in agreement, so the solve is one system per batch
        # instance rather than one system with several right-hand sides.
        solve_batch = broadcast_shapes(k.shape[:-1], rhs.shape[:-1])
        k = k.expand(solve_batch + k.shape[-1:])
        rhs = rhs.expand(solve_batch + rhs.shape[-1:])
        op = self._operator_at(k, node_slopes)
        result = select_solve(
            op,
            rhs,
            method=self.linear_solver,
            where=f"KirchhoffLayer {self.name!r} linear_init",
            rtol=inner_solve_rtol(rhs.dtype),
        )
        return result.x

    def _check_no_unreachable_differentiable_tensors(self) -> None:
        """Guard against a gradient that would be silently wrong or absent: the differentiable
        solve threads only two kinds of tensor into
        `Function.apply` -- each ConstitutiveLaw's own registered `nn.Parameter`s (found via
        `named_parameters()`, substituted in via `functional_call`) and whatever is reachable
        through the `drivers`/`sources`/`phi_boundary` arguments to `solve()`. Any OTHER
        tensor a ConstitutiveLaw or Drive happens to hold with `requires_grad=True` is captured only
        by closure (the law/drive object itself, not its tensor payload) and is
        invisible to `_Implicit.backward`: for a ConstitutiveLaw, this happens when it was
        constructed with `learnable=False` on a tensor that already had `requires_grad=True`
        (`ConstitutiveLaw._param`'s documented pass-through exception, which is still a correct
        construction -- its graph is reachable through this layer's own component calls
        (`flows`, `dflows`, `residual`, `jacobian`, `linear_init`), though NOT through
        `solve(differentiable=False)`, which returns detached tensors);
        for a Drive, this happens whenever a Drive
        implementation owns a learnable coefficient directly instead of reading it from the
        `drivers` mapping passed to `solve()`. Raise now, before dispatching, rather than
        return a gradient that is silently wrong (if some other path happens to also touch
        the same value) or silently `None` (if it doesn't).
        """
        for el in (*self._laws, *self._node_sources):
            # A law may wrap another (`Scaled`): scan every submodule's own attributes,
            # its registered buffers, and tensors nested in lists, tuples, dicts and sets.
            holders = el.modules() if isinstance(el, torch.nn.Module) else (el,)
            for name, value in (kv for h in holders for kv in _held_tensors(h)):
                if value.requires_grad:
                    if isinstance(el, NodeSource):
                        # A NodeSource has no `learnable=` constructor kwarg (unlike
                        # ConstitutiveLaw) -- its fix is to register the tensor as an nn.Parameter
                        # directly, so it must be named and fixed as a node source, not
                        # rendered (and told to construct itself) as if it were a ConstitutiveLaw.
                        raise ValueError(
                            f"node source {el!r} has a tensor attribute {name!r} with "
                            f"requires_grad=True that is not a registered nn.Parameter, so "
                            f"the differentiable solve cannot reach it through "
                            f"Function.apply and its gradient would be silently wrong or "
                            f"absent. Register it as an nn.Parameter "
                            f"(self.<name> = torch.nn.Parameter(...)) in __init__ so it is "
                            f"reachable via named_parameters() (buffers and tensors inside "
                            f"containers are not), or use differentiable=False."
                        )
                    raise ValueError(
                        f"law {el!r} (kind {el.kind!r}) has a "
                        f"tensor attribute {name!r} with requires_grad=True that is not a "
                        f"registered nn.Parameter (a buffer or a tensor inside a container "
                        f"is not one either), so the differentiable solve cannot reach "
                        f"it through Function.apply and its gradient would be silently wrong "
                        f"or absent. Construct this law with learnable=True (so the tensor "
                        f"is registered and reachable via named_parameters()), or use "
                        f"differentiable=False."
                    )
        for drv in self._drives:
            for name, value in _held_tensors(drv):
                if value.requires_grad:
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
    def _reaches_grad(self, drivers: Mapping, sources, phi_boundary) -> bool:
        """Whether anything a differentiable solve threads through `Function.apply` requires
        grad under the current grad mode: a driver, `sources`, `phi_boundary`, or a law
        or node source's parameter."""
        if not torch.is_grad_enabled():
            return False
        values = bulk_values(drivers, list(drivers.keys()))
        values += [sources, phi_boundary]
        for owner in (*self._laws, *self._node_sources):
            values += list(owner.parameters())
        return any(isinstance(v, torch.Tensor) and v.requires_grad for v in values)

    def solve(
        self,
        phi_boundary,
        drivers=None,
        sources=None,
        phi0=None,
        *,
        q0=None,
        history=None,
        dt=None,
        t=None,
        integration="trapezoidal",
        mode=None,
        differentiable=True,
        diagnostics: dict | None = None,
        **newton_kwargs,
    ):
        """Solve for interior potentials and branch flows.

        With `differentiable=True` (the default), gradients flow back to `phi_boundary`,
        `sources`, every value in `drivers`, and every ConstitutiveLaw's registered `nn.Parameter`s
        (i.e. constructed with `learnable=True`) via the implicit-function adjoint
        (`noodl.solvers.implicit`). Only tensors reachable one of those ways are threaded
        through `Function.apply`. Contract each ConstitutiveLaw and Drive must satisfy for
        `differentiable=True` to be safe:

        - A ConstitutiveLaw's own differentiable state must be a registered parameter
          (`learnable=True`), never a bare tensor held with `requires_grad=True` outside
          `named_parameters()` (the latter is still a supported construction, per
          `ConstitutiveLaw._param`, and its graph is reachable through this layer's component calls
          -- but it is invisible to the differentiable solve, and `differentiable=False` now
          returns detached tensors, so neither `solve` path carries a gradient to it).
        - A Drive must read every differentiable quantity from the `drivers` mapping passed
          to `solve()` (see `noodl.drives.Drive`), never hold one of its own as an
          instance attribute.

        Violating either raises `ValueError` naming the offending law/drive and
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
        an optional extra, `pip install noodl-physics[sparse]`). Without it, an installation
        missing that extra takes the much slower PCG path with nothing saying so; the
        indirect signal is `"linear_iterations"` (1 for a factorisation, ~170 for PCG).
        It is `None` only when no linear solve happened at all.

        `on_failure` (forwarded to `newton` among `newton_kwargs`) is `"raise"` by default:
        a batch that fails to converge within `max_iter` raises, naming the failing
        instances. `"return"` is the explicit, narrow escape hatch --
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

        `on_failure="return_primal"` is `"return"` where the solve takes the primal
        (non-differentiable) path -- `differentiable=False`, or `analytic_primal` with nothing
        differentiable reaching it -- and `"raise"` where it is differentiated; it needs
        `diagnostics=` too. It is for a caller that iterates the solve inside an outer
        fixed point (`Model(coupling="iterate")`), where one unconverged pass is only an
        iterate and the outer iteration judges it, but where a differentiated solve must
        still converge.

        `newton_defaults` (constructor) supplies Newton keywords (e.g. ``line_search``,
        ``step_floor``, ``max_iter``) this layer's own laws call for; a keyword given here
        overrides it.

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
        grad mode the caller is in. A backend choice must not be a function of who is
        calling: a caller who wants gradients calls `differentiable=True`, which is unchanged
        (`solvers.implicit._Implicit.forward` already solved under `no_grad` and takes its
        gradients from the adjoint).

        A layer solved by Modified Nodal Analysis (`self.mna`) solves for the interior
        potentials and the flows of its implicit-law edges together (`noodl.layers._mna`),
        from zeros or the warm start `phi0` (interior potentials) and `q0` (the flows of
        `implicit_edges`); the returned `q` covers every edge. `history`, `dt`, `t`,
        `integration` and `mode` set the context its laws are evaluated in (`LawContext`):
        on a phasor layer always `"phasor"` (no `dt` or `history`; the frequency driver
        `"<name>.frequency"` is required, and `phi_boundary`, `sources`, `phi0`, `q0` may be
        real or complex and are taken as complex128); otherwise
        with `dt` given, one implicit step of length `dt` ending at time `t`, with the
        `integration` rule (`"trapezoidal"` or `"backward_euler"`) from the step-start state
        `history=(phi, q)` (full-node potentials and every edge's flow, as this method
        returns them; required when a law is dynamic); without `dt`, the steady operating
        point (`mode="dc"`, the default), or with `mode="initial"` the consistent initial
        point (each storage law at its initial value). On the differentiable path gradients
        flow to `history` too, so a chain of steps back-propagates through time. There
        `diagnostics["method"]` is likewise the method requested and `"backend"` the one that
        ran; `method="cg"` is refused (the system is not symmetric positive definite). A
        classic layer, whose laws are static, accepts and ignores `q0`, `t`, `integration`
        and the modes `"dc"` and `"initial"` (both its one steady solution), and refuses
        `mode="phasor"`, `dt` and `history`: it takes no step and solves no phasors.
        """
        self._require_current_topology()
        for key, value in self.newton_defaults.items():
            newton_kwargs.setdefault(key, value)
        if self.mna:
            return self._solve_mna(
                phi_boundary, drivers, sources, phi0, q0, history, dt, t, integration, mode,
                differentiable, diagnostics, newton_kwargs,
            )
        self._check_classic_arguments(mode, dt, history)
        drivers = drivers or {}
        policy = resolve_classic(
            self, newton_kwargs, diagnostics, differentiable, drivers, sources, phi_boundary
        )
        phi0, dq0, node_slopes0 = self._starting_point(phi0, phi_boundary, drivers, sources)
        if not policy.differentiable:
            return self._solve_primal(
                phi0, dq0, node_slopes0, phi_boundary, drivers, sources, diagnostics,
                newton_kwargs,
            )
        self._check_no_unreachable_differentiable_tensors()
        fn = FunctionalLaws(self, drivers, sources, phi_boundary)
        return self._solve_implicit(
            fn, phi0, phi_boundary, drivers, policy, diagnostics, newton_kwargs
        )

    def _solve_mna(
        self, phi_boundary, drivers, sources, phi0, q0, history, dt, t, integration, mode,
        differentiable, diagnostics, newton_kwargs,
    ):
        """`solve` of a layer solved by Modified Nodal Analysis (`noodl.layers._mna`)."""
        differentiable = resolve_mna(
            self, newton_kwargs, diagnostics, differentiable, drivers, sources, phi_boundary,
            history,
        )
        return self._mna.solve(
            phi_boundary, drivers, sources, phi0, q0=q0, history=history, dt=dt, t=t,
            integration=integration, mode=mode, differentiable=differentiable,
            diagnostics=diagnostics, **newton_kwargs,
        )

    def _check_classic_arguments(self, mode, dt, history) -> None:
        """Refuse the keywords a classic (static-law) layer cannot honour."""
        if mode == "phasor":
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: mode='phasor' needs a layer built with "
                f"phasor=True"
            )
        if mode is not None and mode not in ("dc", "initial"):
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: mode={mode!r}; expected 'dc', 'initial' or "
                f"None (transient when dt is given, dc otherwise)"
            )
        stepping = [k for k, v in (("dt", dt), ("history", history)) if v is not None]
        if stepping:
            raise ValueError(
                f"KirchhoffLayer {self.name!r} is solved by nodal analysis: its laws are "
                f"static, so it takes no step; drop {stepping}"
            )

    def _starting_point(self, phi0, phi_boundary, drivers, sources):
        """The Newton starting guess, certified grounded: `(phi0, dq0, node_slopes0)`.

        `dq0` and `node_slopes0` are the slopes at that point; the first operator reuses them.
        """
        # The initial guess is NOT a differentiable quantity, and neither is the grounding
        # check below. Both are computed under no_grad (and a caller-supplied phi0 is
        # detached) so nothing that produced them is traced into the autograd graph.
        #
        # This is a memory fix, not a numerical one: the converged point is where the
        # implicit-function adjoint linearises, and that point is independent of the guess
        # the iteration started from, so tracing the guess buys no gradient at all.
        # `linear_init` itself is untouched and stays differentiable for callers who want it
        # directly.
        with torch.no_grad():
            if phi0 is None:
                phi0 = self.linear_init(phi_boundary, drivers, sources)
            else:
                phi0 = phi0.detach()

            # Grounding is certified on the ACTUAL slopes at the point the Newton iteration
            # is about to start from, whatever its source. linear_init's own check sees only
            # its tangent-at-zero slopes, and is not run at all when a caller supplies phi0
            # -- so a supplied phi0 must not bypass grounding, and a law whose
            # slope is dp-dependent (a fan past its shutoff point, whose dflow is exactly
            # zero) could leave the operator singular with nothing to say about it but a
            # Newton non-convergence. Placed BEFORE the differentiable branch so both paths
            # run it unconditionally and identically. It is a CHECK: it raises or it does
            # not, and no tensor it computes reaches the result, so it runs under no_grad
            # too (`dflows` here is each ConstitutiveLaw's own analytic `dflow`, not the autograd
            # path the differentiable branch builds below).
            phi0_full = self.assemble(phi0, phi_boundary)
            dq0 = self.dflows(phi0_full, drivers)
            node_slopes0 = self._node_source_slopes(phi0_full, drivers)
            self._grounding_check(dq0, where="solve", node_slopes=node_slopes0)
        return phi0, dq0, node_slopes0

    def _solve_primal(
        self, phi0, dq0, node_slopes0, phi_boundary, drivers, sources, diagnostics,
        newton_kwargs,
    ):
        """The non-differentiable classic solve: Newton alone, detached `(phi, q)`."""
        # The WHOLE non-differentiable solve runs under no_grad -- the Newton iteration,
        # every inner linear solve inside it, and the final assemble/flows. Two reasons,
        # the second of which is the one that made this mandatory rather than tidy:
        #
        # 1. There is no legitimate graph to build here. This branch does not use the
        #    implicit adjoint, so any graph it leaves behind is an UNROLLED trace through
        #    the converged Newton iterate -- real gradients, but not the ones
        #    `differentiable=True` computes, retained for a caller who asked for the
        #    non-differentiable path.
        # 2. It made the inner solver's choice depend on the CALLER's ambient grad mode.
        #    `solvers.select.solve`'s "auto" refuses the non-differentiable sparse-direct
        #    backend when grad mode is on and an input requires grad (correctly: it would
        #    silently detach). With the iteration running under whatever mode the caller
        #    happened to be in, the same `solve(differentiable=False)` factorised through
        #    SuperLU from a plain call and fell back to PCG -- a much slower, different
        #    code path (docs/design/solvers.md) -- from inside `torch.enable_grad()` with a
        #    grad-requiring `sources`. A solver choice must not be a function of the
        #    caller's context. The guard in `select.solve` stays as the last line of
        #    defence; this removes the condition that was tripping it.
        #
        # `diagnostics` is unaffected by no_grad; the returned `(phi, q)` are
        # unconditionally detached -- see the docstring.
        with torch.no_grad():
            # The last residual's point, potentials and flows: Newton returns the iterate
            # it last evaluated the residual at, whose flows are then the result's.
            last: dict = {}

            def residual_fn(x):
                r, phi, q = self._residual_parts(x, phi_boundary, drivers, sources)
                last.update(x=x, phi=phi, q=q)
                return r

            def operator_fn(x):
                # A matvec-free A_I diag(dq) A_I^T at the current iterate, instead of the
                # dense (n_interior, n_interior) einsum layer.jacobian() assembles.
                # Rebuilt each iteration because dq is what changes; the endpoint/index
                # tensors it closes over are cached on the layer at construction. At the
                # starting point the grounding check's slopes above are those slopes.
                if x is phi0:
                    return self._operator_at(dq0, node_slopes0)
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
                # nothing else reports it.
                diagnostics["backend"] = result.backend
                # The per-instance STATUS, not only the cost. Without these two,
                # `on_failure="return"` returned a non-converged phi with nothing
                # anywhere reporting it.
                diagnostics["converged"] = result.converged
                diagnostics["residual_norm"] = result.residual_norm
            if last.get("x") is result.x:
                phi, q = last["phi"], last["q"]
            else:
                phi = self.assemble(result.x, phi_boundary)
                q = self.flows(phi, drivers)
        return phi, q

    def _solve_implicit(
        self, fn, phi0, phi_boundary, drivers, policy, diagnostics, newton_kwargs
    ):
        """The differentiable classic solve through the implicit-function adjoint."""
        all_params = fn.all_params
        residual_fn, operator_fn = fn.residual_fn, fn.operator_fn
        if policy.forward_only:
            # `implicit_solve`'s own forward (the same Newton on the same autograd Jacobian),
            # which may return unconverged: nothing reaches a gradient, so no adjoint is
            # ever asked of this point.
            with torch.no_grad():
                result = newton(
                    lambda x: residual_fn(x, *all_params),
                    lambda x: operator_fn(x, *all_params),
                    phi0,
                    **newton_kwargs,
                )
            x = result.x
            if diagnostics is not None:
                diagnostics.update(
                    newton_iterations=result.iterations,
                    linear_iterations=result.linear_iterations,
                    backend=result.backend,
                    converged=result.converged,
                    residual_norm=result.residual_norm,
                )
        else:
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
        layer's configured inner solver onto the backward pass too, so
        `linear_solver="direct"` is the dense reference numerics on BOTH passes.

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
        self._refuse_on_mna("adjoint")
        drivers = drivers or {}
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self.dflows(phi, drivers)
        op = self._operator_at(dq, self._node_source_slopes(phi, drivers))
        return _adjoint_solve(
            op,
            grad_phi_interior,
            where=f"KirchhoffLayer {self.name!r} adjoint",
            method=self.linear_solver,
        )

    def power_residual(self, phi, q, drivers, sources=None):
        """Tellegen's power identity, zero at a converged solution.

        (dp*conj(q)).sum() - (drive*conj(q)).sum() - (phi_bound * conj(A_bound @ q)).sum()
        - (phi_interior * conj(sources_interior)).sum()

        The conjugates change nothing on a real layer; on a phasor layer the identity is the
        complex power balance (real and imaginary parts both zero).

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
            (d * q.conj()).sum(-1)
            - (drive_only * q.conj()).sum(-1)
            - (phi_b * boundary_flow.conj()).sum(-1)
            - (phi_i * s_I.conj()).sum(-1)
        )
