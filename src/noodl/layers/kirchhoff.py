"""KirchhoffLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + head; theta) - s_I from a set of constitutive laws
on typed edges (one ConstitutiveLaw per kind) and the per-edge head driver
`"<layer>.edge_head"` (an additive potential difference, see `noodl.heads`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from noodl._broadcast import broadcast_shapes
from noodl._tracking import bulk_values
from noodl.elements._slopes import _colour_blocks
from noodl.elements.base import SCHEMES, ConstitutiveLaw, ImplicitLaw, LawContext
from noodl.layers._flow_unknowns import (
    FlowUnknowns,
    _cat,
    _Held,
    angular_frequency,
    cdtype,
    check_frequency,
    first_failing,
    flow_unknowns,
    linear_method,
    singular,
    singular_error,
    split,
    to_complex,
    to_real,
)
from noodl.layers._functional import FunctionalLaws, layer_slopes, step_start
from noodl.layers._solve_policy import resolve
from noodl.layers._topology import TopologyGuard
from noodl.operators.graph import GraphLaplacianOperator
from noodl.operators.shifted import ShiftedOperator
from noodl.operators.sparse import SparseOperator
from noodl.solvers.continuation import storage_ramp
from noodl.solvers.errors import ConvergenceError
from noodl.solvers.grounding import spd_certificate, spd_diagnosis
from noodl.solvers.implicit import adjoint as _adjoint_solve
from noodl.solvers.implicit import implicit_solve
from noodl.solvers.newton import inner_solve_rtol, newton
from noodl.solvers.select import solve as select_solve
from noodl.storage import NodeStorage
from noodl.topology import Network
from noodl.withdrawals import NodeWithdrawal

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

#: The steady context: what a law that reads the context sees outside a solve (`flows`,
#: `dflows`, `linear_init` called directly) and in a solve without a step.
_DC = LawContext(mode="dc")


#: The rungs of the fallback ladder (`KirchhoffLayer(fallback=)`), in the order of the docs.
FALLBACK_RUNGS = ("linear_guess", "storage_ramp")


def check_fallback(layer_name: str, fallback) -> tuple[str, ...]:
    """`fallback` as a tuple of rung names, each one of `FALLBACK_RUNGS` and given once;
    anything else is refused by name."""
    if isinstance(fallback, str):
        raise TypeError(
            f"KirchhoffLayer {layer_name!r}: fallback takes a sequence of rung names, e.g. "
            f"({fallback!r},), not the string {fallback!r}"
        )
    rungs = tuple(fallback)
    unknown = [r for r in rungs if r not in FALLBACK_RUNGS]
    if unknown:
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: unknown fallback rung(s) {unknown}; the rungs "
            f"are {list(FALLBACK_RUNGS)}"
        )
    if len(set(rungs)) != len(rungs):
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: fallback {list(rungs)} names a rung twice"
        )
    return rungs


def _converged(diagnostics: dict) -> bool:
    """Whether every instance of the solve that wrote `diagnostics` converged."""
    return bool(torch.as_tensor(diagnostics["converged"]).all())


def check_scheme(layer_name: str, scheme) -> None:
    """Refuse a time-integration scheme a `KirchhoffLayer` cannot take, by name.

    A circuit or network layer integrates with `"backward_euler"` or `"trapezoidal"`, or
    with `"forward_euler"` when its storage sits at boundary nodes (the layer checks that);
    `"exact"` (the exponential of a linear transport step) belongs to a `TransportLayer`,
    and anything else is not a scheme name. The message lists `SCHEMES`.
    """
    if scheme == "exact":
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: scheme='exact' is a TransportLayer scheme; a "
            f"KirchhoffLayer takes 'backward_euler' or 'trapezoidal' ('forward_euler' for "
            f"storage at boundary nodes; schemes: {SCHEMES})"
        )
    if scheme not in SCHEMES:
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: scheme={scheme!r}; expected one of {SCHEMES} "
            f"('exact' is for a TransportLayer only)"
        )


#: Containers longer than this are not entered by `_held_tensors`: a long list or dict on a
#: law is a numeric table, not a parameter holder, and walking it item by item would cost
#: more than the solve (`KirchhoffLayer._require_threaded`, which evaluates the residual,
#: needs no walk).
_SCAN_MAX_ITEMS = 1024
_ATOMS = (str, bytes, int, float, complex, bool, type(None))


@dataclass(frozen=True)
class _StorageStep:
    """The storage of one step: the step-start potentials `phi_prev` (full-node) and the
    flows into every storage entry at the step start, `previous` (`None` when not given; the
    trapezoidal scheme needs it)."""

    phi_prev: torch.Tensor
    previous: torch.Tensor | None
    #: storage `j`'s `(phi_start, previous)` columns by `j`: constants of the step, taken
    #: once per solve rather than at every residual (`KirchhoffLayer._storage_args`)
    start_args: dict = dataclasses.field(default_factory=dict, init=False, compare=False,
                                         repr=False)


@dataclass(frozen=True)
class _Step:
    """What one solve evaluates its laws and storage in: the law context (mode, dt, t,
    scheme, the step-start dp_prev/q_prev of the laws that read them, omega on a phasor
    layer) and what the storage does (`None`; a `_StorageStep` on a step; the held initial
    rows, `_Held`, at the initial point). Built once per solve by `KirchhoffLayer._step`.

    `layout` is the flow-unknown layout this one solve uses when it is not the layer's own:
    the initial point of a layer without flow unknowns whose storage holds an initial
    potential is solved with the held rows on the sparse operator, in a layout with no
    unknown flows (`KirchhoffLayer._held_layout`). `None` everywhere else."""

    ctx: LawContext
    storage: _StorageStep | _Held | None
    layout: FlowUnknowns | None = None
_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
           types.MethodType)


def _detached(params) -> list:
    """`params` with every tensor detached."""
    return [p.detach() if isinstance(p, torch.Tensor) else p for p in params]


def _same_objects(a, b) -> bool:
    """Whether the sequences `a` and `b` hold the same objects (`is`), position by position."""
    return a is not None and len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


def _shallow(obj):
    """What one object of a scan holds, by identity: a container's items (its length alone
    past `_SCAN_MAX_ITEMS`, where the scan does not enter it), an object's attributes. Equal
    shallow signatures of every object a scan visited mean the scan would find the same
    tensors (`KirchhoffLayer._held_scan`)."""
    if isinstance(obj, dict):
        if len(obj) > _SCAN_MAX_ITEMS:
            return len(obj)
        return tuple(obj), tuple(map(id, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)):
        if len(obj) > _SCAN_MAX_ITEMS:
            return len(obj)
        return tuple(map(id, obj))
    held = vars(obj)
    return tuple(held), tuple(map(id, held.values()))


def _held_tensors(obj, *, registered_threaded: bool = True, visited: list | None = None):
    """Yield `(name, tensor)` for every tensor `obj` holds: its attributes, its registered
    buffers (when a module), and tensors nested at any depth (up to 8) in lists, tuples,
    dicts, sets and the attributes of plain objects and modules held there.

    With `registered_threaded`, `obj`'s own registered parameters (`_parameters`) and
    submodules (`_modules`, scanned as holders of their own) are not yielded: the
    differentiable solve threads them. A module reached through a container or a plain
    attribute is not registered, so its parameters ARE yielded. Each object is visited once
    (cycles end), and a container longer than `_SCAN_MAX_ITEMS` is not entered. `visited`,
    when given, collects every object whose contents the scan read (`_shallow`)."""
    seen: set[int] = set()
    record = visited.append if visited is not None else (lambda _o: None)

    def walk(label, value, depth):
        if isinstance(value, torch.Tensor):
            yield label, value
            return
        if depth >= 8 or isinstance(value, _ATOMS + _OPAQUE) or id(value) in seen:
            return
        seen.add(id(value))
        record(value)
        if isinstance(value, dict):
            if len(value) <= _SCAN_MAX_ITEMS:
                for k, v in value.items():
                    yield from walk(f"{label}[{k!r}]", v, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if len(value) <= _SCAN_MAX_ITEMS:
                for i, v in enumerate(value):
                    yield from walk(f"{label}[{i}]", v, depth + 1)
        elif hasattr(value, "__dict__"):
            for k, v in vars(value).items():
                if k != "_memo_cache":
                    yield from walk(f"{label}.{k}", v, depth + 1)

    skip = {"_memo_cache"}
    record(obj)
    if registered_threaded:
        skip |= {"_parameters", "_buffers", "_modules"}
        if isinstance(obj, torch.nn.Module):
            # what is registered decides what is skipped: a change there changes the scan
            for registry in (obj._parameters, obj._buffers, obj._modules):
                record(registry)
    for name, value in vars(obj).items():
        if name not in skip:
            yield from walk(name, value, 0)
    if registered_threaded and isinstance(obj, torch.nn.Module):
        for name, value in obj._buffers.items():
            if value is not None:
                yield name, value


class KirchhoffLayer(TopologyGuard):
    """Flows driven by a potential (pressure, head): solves the flow balance at the interior
    nodes for the interior potentials by Newton's method, with the constitutive laws on its
    edge kinds, the head driver added to the potential differences, and prescribed potentials
    at its boundary nodes; differentiable through the converged solution. `flow_unit` is
    reported by `model.refs`, never used numerically.

    Flows from potentials (the fast path). A law that gives its flow from the potential
    difference, `q = g(A^T phi + head)`, adds no unknown: when every law does, the balance is
    `A_I g(A^T phi) = s_I`, the unknowns are the interior potentials alone and the Jacobian
    is the symmetric network matrix. A law that reads the `LawContext` (the time, its
    step-start values: a time-dependent current source, a storing law in flow form) is such a
    law too. The grounding of this system is certified at the start of every solve.

    A law that gives a relation adds its flow as an unknown. A law that cannot give the flow
    from the potential difference (`ImplicitLaw`: an ideal source, a flow-controlled law, a
    capacitance or inductance) gives a relation `residual(dp, q) = 0` instead, and a coupled
    multi-port law gives its edges' flows together; the flows of those edges
    (`implicit_edges`) become unknowns alongside the interior potentials, the relations add
    their rows, and the Jacobian is a sparse, nonsymmetric matrix. Every connected part of
    such a layer's edges must contain a boundary node (a ground). Every method (`solve`,
    `flows`, `dflows`, `residual`, `jacobian`, `adjoint`, `linear_init`) is defined on every
    layer; those that need the flows of the relations take the solved `q`.

    Phasor. With `phasor=True` the layer solves the sinusoidal steady state, always with the
    sparse operator of the previous paragraph: potentials and flows are complex amplitudes
    (complex128), the frequency in Hz comes from the driver `"<name>.frequency"` (one per
    instance) and reaches the laws as `LawContext.omega` (`2 pi f`). Each complex unknown is
    solved as its (real, imaginary) pair, so a law that is not holomorphic in `dp` (a
    constant-power load, `q = conj(S / dp)`) is handled exactly.
    Every law must declare `phasor_capable`, withdrawals are refused (a load is an edge to
    ground), and `power_residual` is the complex power balance.

    `storage` (default none) are `NodeStorage` objects (`noodl.storage`): an amount stored at
    interior nodes, which a step of `dt` turns into a withdrawal into storage with the step
    rate of the layer's scheme; without a step (a steady solve) storage is absent. A layer
    with storage is `dynamic` and takes `dt`, `history` and `storage_flow` in `solve`.
    Storage adds no flow unknowns: without flow unknowns its Jacobian gains only a
    non-negative diagonal, so a network of conductances with storage keeps its symmetric
    positive definite system, and a storage node counts as grounded during a step. A layer
    with flow unknowns takes storage too (its diagonal slot of each node row).
    `mode="initial"` holds each node with a storage entry whose `initial()` is not `None` at
    that potential, on every layer: without flow unknowns that one solve uses the
    sparse operator with the held rows. A phasor layer takes no storage (a capacitance is an
    edge law there).

    With `scheme="forward_euler"` the storage sits at BOUNDARY nodes instead, and is stepped
    by forward Euler (EPANET's tank): it is never an unknown. The solve holds each storage
    node at its potential at the step start (`hold_storage`; every `Model` path writes it
    into `"<name>.phi_boundary"`, whatever the driver gives there), and the step then moves
    it by `dt * inflow / C` (`advance_storage`), `inflow` being the flow the solve delivers
    to the node (`storage_flows`) and `C` the storage's capacity. Only a linear storage (a constant
    capacity, `LinearStorage`) is accepted there, and only one entry per node. Such a
    layer's solve takes no step and does not depend on `dt`.

    `scheme` (default `"trapezoidal"`; `"backward_euler"` the other) is the time-integration
    scheme of a transient step of a layer with dynamic laws or storage, one of `SCHEMES`
    (`noodl.elements.base`); a `solve` call may give its own. `"exact"` is a `TransportLayer`
    scheme and is refused here by name. `"forward_euler"` is for storage at boundary nodes
    only (above): a layer with storage at interior nodes, with dynamic laws or with no storage
    refuses it, and a layer with boundary storage refuses every other scheme.

    `newton_defaults` (default none) are Newton keywords (`solvers.newton.newton`) every
    `solve` of this layer uses unless the call gives its own, for laws that need them: the
    sewer dynamic wave sets ``line_search="convex"`` and ``step_floor`` (its kinked laws are
    each non-decreasing, so the residual is the gradient of a convex function). A layer
    with flow unknowns refuses ``line_search="convex"``, by keyword or here: its residual is
    not the gradient of a convex function.

    `fallback` (default none) is the ladder `solve` climbs when Newton does not converge,
    rungs from `FALLBACK_RUNGS` tried in order: `"linear_guess"` solves again from the
    linear initial guess, `"storage_ramp"` lets the network settle from it through a
    fictitious storage removed in steps (`noodl.solvers.continuation.storage_ramp`) and
    solves again from there. A `solve` call may give its own; see `solve`."""

    def __init__(
        self,
        net: Network,
        name: str,
        laws: Sequence[ConstitutiveLaw],
        boundary: Sequence = (),
        linear_solver: str = "auto",
        *,
        withdrawals: Sequence[NodeWithdrawal] = (),
        quantity: str = "potential",
        unit: str = "",
        flow_unit: str = "",
        phasor: bool = False,
        newton_defaults: Mapping | None = None,
        scheme: str = "trapezoidal",
        storage: Sequence[NodeStorage] = (),
        fallback: Sequence[str] = (),
    ) -> None:
        check_scheme(name, scheme)
        self.scheme = scheme
        self.newton_defaults = dict(newton_defaults or {})
        self.fallback = check_fallback(name, fallback)
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

        self.net = net
        self._record_topology(net)
        self.name = name
        self.head_key = f"{name}.edge_head"
        self._laws = list(laws)
        # A relation (`ImplicitLaw`) has its flow as an unknown of the layer, read once here.
        self._relation = [isinstance(el, ImplicitLaw) for el in laws]
        self.kinds = [el.kind for el in laws]
        self._storage = list(storage)
        for st in self._storage:
            if not isinstance(st, NodeStorage):
                raise TypeError(
                    f"KirchhoffLayer {name!r}: storage entry {type(st).__name__} is not a "
                    f"NodeStorage"
                )
        # Under `scheme="forward_euler"` every storage sits at boundary nodes, held through
        # the solve and stepped after it (`advance_storage`): none of the interior storage
        # machinery below sees it, so it is kept apart.
        self._boundary_storage: list[NodeStorage] = []
        if scheme == "forward_euler":
            self._boundary_storage, self._storage = self._storage, []
        # Flow unknowns: a law whose flow is an unknown (a relation) or that couples several
        # edges (ports > 1), or a phasor layer, cannot be solved by the fast path; such a
        # layer adds flow unknowns (`_flow_unknowns`). A law that reads the law context (the
        # time, its step-start values) still gives its flow from the potential difference,
        # and node storage adds a diagonal to the nodal Jacobian: both stay on the fast path,
        # and a storing law or a storage makes the layer dynamic (it steps from its state).
        self.phasor = phasor
        self.dynamic = any(el.dynamic for el in self._laws) or bool(self._storage)
        # every flow follows from the potentials: no law is a relation, none couples edges, and
        # the layer is real, so the Jacobian is the symmetric network matrix (plus the storage
        # and withdrawal diagonal)
        self._eliminated = not self.phasor and not any(self._relation) and all(
            el.ports == 1 for el in self._laws)
        # The laws that read the law context, by position: the fast path passes them the
        # step's context (restricted to their edges, `_law_ctx`); every other law keeps the
        # two-argument call. Empty on most layers, which then never build a restriction.
        self._ctx_laws = tuple(i for i, el in enumerate(self._laws) if el._takes_ctx)
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

        node_index = {node: i for i, node in enumerate(net.nodes)}
        for b in boundary:
            if net._resolve(b) not in node_index:
                raise KeyError(f"unknown boundary node {b!r} in layer {name!r}")
        boundary = net.canonical(boundary)

        touched = torch.zeros(net.n, dtype=torch.bool)
        touched[self._src] = True
        touched[self._tgt] = True
        # A node a withdrawal acts at is active too: its row is the source alone (a
        # junction no conduit reaches still has its storage, `sewer.dynwave.JunctionStorage`).
        for ns in withdrawals:
            if isinstance(ns, NodeWithdrawal):
                touched[ns.nodes] = True
        # A storage node is active too, for the same reason.
        for st in self._storage:
            touched[st.nodes] = True
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
        # wiring mistakes, named here rather than silently dropped. `_withdrawal_positions`
        # holds each source's nodes in this layer's COMPACT interior indexing, which is
        # what the residual's scatter-add and the Jacobian's diagonal both need.
        self._withdrawals = list(withdrawals)
        self._withdrawal_positions: list[torch.Tensor] = []
        for ns in self._withdrawals:
            if not isinstance(ns, NodeWithdrawal):
                raise TypeError(
                    f"KirchhoffLayer {name!r}: withdrawals entry "
                    f"{type(ns).__name__} is not a NodeWithdrawal"
                )
            pos = self._interior_of_node[ns.nodes]
            if bool((pos < 0).any()):
                bad = [
                    net.nodes[int(i)]
                    for i, p in zip(ns.nodes.tolist(), pos.tolist(), strict=True)
                    if p < 0
                ]
                raise ValueError(
                    f"KirchhoffLayer {name!r}: withdrawal {ns!r} acts at {bad}, "
                    f"which are not interior unknowns of this layer (a boundary node's "
                    f"potential is prescribed and an inactive node has no row)"
                )
            self._withdrawal_positions.append(pos)

        # NODE STORAGE. Each storage's nodes must be interior unknowns, as for node
        # sources. `_storage_positions` are their compact interior rows; `_storage_slices`
        # the columns of each storage's entries in `storage_flows` (and `storage_flow`).
        self._storage_positions: list[torch.Tensor] = []
        self._storage_slices: list[tuple[int, int]] = []
        offset = 0
        for st in self._storage:
            pos = self._interior_of_node[st.nodes]
            if bool((pos < 0).any()):
                bad = [
                    net.nodes[int(i)]
                    for i, p in zip(st.nodes.tolist(), pos.tolist(), strict=True)
                    if p < 0
                ]
                raise ValueError(
                    f"KirchhoffLayer {name!r}: storage {st!r} is at {bad}, which are not "
                    f"interior unknowns of this layer (a boundary node's potential is "
                    f"prescribed, unless the layer steps it with scheme='forward_euler', "
                    f"and an inactive node has no row)"
                )
            self._storage_positions.append(pos)
            self._storage_slices.append((offset, offset + pos.numel()))
            offset += pos.numel()
        self._boundary_storage_cols = self._check_boundary_storage(net, name, scheme)
        offset += sum(int(cols.numel()) for cols in self._boundary_storage_cols)
        self._n_entries = offset

        # The edges of the laws that store (dynamic), and whether a step has anything to
        # take: a dynamic law, a storage or a law that reads the step's context.
        dyn = [torch.arange(s, e) for el, (s, e) in
               zip(self._laws, self._law_slices, strict=True) if el.dynamic]
        self._dynamic_edges = torch.cat(dyn) if dyn else torch.zeros(0, dtype=torch.long)
        self._takes_step = bool(dyn) or bool(self._storage) or any(
            el._takes_ctx for el in self._laws)
        # whether a law reads the step-start potential differences and flows of its edges
        self._reads_history = bool(dyn) or any(self._relation) or any(
            el._takes_ctx for el in self._laws)

        # The layout of the flow unknowns; `None` on the fast path. The one attribute every
        # path decision of `_residual`, `_operator` and `solve` reads.
        self._unknowns: FlowUnknowns | None = None
        # The step-start restrictions of the law context, per law (`_law_ctx`).
        self._law_ctxs: tuple[LawContext | None, dict] = (None, {})
        # The layout of the initial point with held storage on a layer without flow
        # unknowns, built on the first such solve (`_held_layout`); `None` until then.
        self._held_unknowns: FlowUnknowns | None = None
        if not self._eliminated:
            self._check_flow_unknowns_structure()
            self._unknowns = flow_unknowns(self)

    def _check_boundary_storage(self, net, name, scheme) -> list[torch.Tensor]:
        """Construction checks of storage under `scheme="forward_euler"`, and each boundary
        storage's columns in `"<name>.phi_boundary"` (the positions of its nodes in
        `bound`)."""
        if scheme != "forward_euler":
            return []
        if not self._boundary_storage:
            raise ValueError(
                f"KirchhoffLayer {name!r}: scheme='forward_euler' steps storage at boundary "
                f"nodes, and this layer has none; give it storage=[LinearStorage at boundary "
                f"nodes], or use 'backward_euler' or 'trapezoidal'"
            )
        dynamic = [el.kind for el in self._laws if el.dynamic]
        if dynamic:
            raise ValueError(
                f"KirchhoffLayer {name!r}: scheme='forward_euler' steps storage at boundary "
                f"nodes only, and kinds {dynamic} have dynamic laws; use 'backward_euler' or "
                f"'trapezoidal'"
            )
        column = {int(n): j for j, n in enumerate(self.bound.tolist())}
        cols, seen = [], set()
        for st in self._boundary_storage:
            nodes = st.nodes.tolist()
            interior = [net.nodes[i] for i in nodes if i not in column]
            if interior:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: storage {st!r} is at {interior}, which are not "
                    f"boundary nodes; scheme='forward_euler' steps storage at boundary nodes "
                    f"only (storage at interior nodes steps with 'backward_euler' or "
                    f"'trapezoidal')"
                )
            if not st.linear:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: storage {type(st).__name__} at boundary nodes "
                    f"is not linear; scheme='forward_euler' steps a node by dt * inflow / C "
                    f"with a constant capacity C (a LinearStorage)"
                )
            again = [net.nodes[i] for i in nodes if i in seen]
            if again:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: boundary storage is given twice at {again}; "
                    f"scheme='forward_euler' takes one storage entry per node"
                )
            seen.update(nodes)
            cols.append(torch.tensor([column[i] for i in nodes], dtype=torch.long))
        return cols

    def _check_flow_unknowns_structure(self) -> None:
        """Construction checks of a layer with flow unknowns: port groups, solver, and a
        boundary node in every connected part."""
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
                f"definite Jacobian, which a layer with flow unknowns does not have; use "
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
            if self._withdrawals:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a phasor layer takes no withdrawals; "
                    f"loads are edges to ground in a phasor layer"
                )
            if self._storage:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a phasor layer takes no storage; a "
                    f"capacitance is an edge law (to ground) in a phasor layer"
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
        flow is an unknown of the layer (the edges of relations and multi-port laws); empty
        on a layer without flow unknowns."""
        if self._unknowns is None:
            return torch.zeros(0, dtype=torch.long)
        return self._unknowns.edges

    @property
    def _steps_with_flows(self) -> bool:
        """Whether a step reads the step-start flows (`history`'s `q`): a layer with flow
        unknowns does, and so does one with a dynamic law (its edges' step-start flows); a
        layer whose only storage is node storage reads only the potentials."""
        return self._unknowns is not None or bool(len(self._dynamic_edges))

    @property
    def frequency_key(self) -> str | None:
        """The driver key a phasor layer reads its frequency from (`"<name>.frequency"`, in
        Hz, one value per instance); `None` for a layer that is not a phasor layer."""
        return f"{self.name}.frequency" if self.phasor else None

    def _relation_kinds(self) -> list[str]:
        """The kinds of this layer that give a relation, whose flows are its unknowns."""
        return [el.kind for el, rel in zip(self._laws, self._relation, strict=True) if rel]

    def _flow_unknown_columns(self, q, what: str, like: torch.Tensor) -> torch.Tensor:
        """`q`'s columns of the flow unknowns, for a layer with flow unknowns. `q` is every
        edge's flow, as `solve` returns it. Without it a layer with relations cannot give
        `what` (the unknown flows are not functions of the potentials) and refuses by name;
        a layer with no relation needs none (`like` sets the empty columns' type)."""
        edges = self._unknowns.edges
        if q is None:
            if len(edges):
                raise ValueError(
                    f"KirchhoffLayer {self.name!r}: kinds {self._relation_kinds()} give a "
                    f"relation, so their flows are unknowns of the layer; pass q (the flows "
                    f"solve returned) to {what}")
            return like.new_zeros(like.shape[:-1] + (0,))
        n_edges = len(self.cols)
        if q.shape[-1] != n_edges:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: q must hold every edge's flow "
                f"({n_edges}) as solve() returns it; got last dimension {q.shape[-1]}")
        return q[..., edges]

    def _unknown_vector(self, phi_interior, q, what: str) -> torch.Tensor:
        """The unknowns `[phi_I, q_F]` in the real view, from the interior potentials and
        every edge's flow `q` (`_flow_unknown_columns`)."""
        q_F = self._flow_unknown_columns(q, what, phi_interior)
        return _cat([to_real(self, phi_interior), to_real(self, q_F)], phi_interior)

    @property
    def withdrawals(self) -> tuple:
        """The `NodeWithdrawal` objects this layer evaluates at its interior nodes, in the order
        given at construction (read-only)."""
        return tuple(self._withdrawals)

    @property
    def storage(self) -> tuple:
        """The `NodeStorage` objects of this layer (at its interior nodes, or at its boundary
        nodes under `scheme="forward_euler"`), in the order given at construction
        (read-only); their entries are the columns of `storage_flows`."""
        return tuple(self._storage) + tuple(self._boundary_storage)

    @property
    def holds_storage(self) -> bool:
        """True when this layer's storage sits at boundary nodes (`scheme="forward_euler"`):
        every solve holds those nodes at their step-start potentials (`hold_storage`), and a
        step moves them afterwards (`advance_storage`)."""
        return bool(self._boundary_storage)

    # ------------------------------------------------------- dense references (lazy)
    @property
    def A(self) -> torch.Tensor:
        """This layer's (n, b_layer) incidence matrix, built on FIRST ACCESS only.

        Refused (`_require_current_topology`) once the network has changed, even when it
        was already built: the cached matrix describes the old graph.

        Held in a `cached_property` rather than an `__init__` attribute: it grows 4x per node
        doubling and would break the composed model's memory-scaling budget (numbers:
        docs/design/orchestration.md#dense-incidence-matrix). Nothing inside this class reads
        it except `jacobian()`, the retained dense reference -- every hot-path site gathers or
        scatter-adds with `_src`/`_tgt` instead (`dp`, `residual`, `linear_init`,
        `power_residual`). It stays public, with exactly its old value
        (`net.incidence()[:, self.cols]`, i.e. columns in LAYER edge order, not network
        order), because callers outside the layer legitimately want the matrix: the
        composed-model conservation tests form `A @ q` with it.
        """
        self._require_current_topology()
        return self._dense_incidence

    @functools.cached_property
    def _dense_incidence(self) -> torch.Tensor:
        """The cache behind `A`."""
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
        `residual`, `A`, `balance` and `net_outflow` raise (`_require_current_topology`)
        once the network has changed.
        """
        return self.net.difference()[self.cols]

    @property
    def linear(self) -> bool:
        """True when every law, withdrawal and storage is affine (`ConstitutiveLaw.linear`):
        a solve with flow unknowns (or the held initial point) then takes the full Newton
        step from the start and converges in one iteration."""
        return (all(el.linear for el in self._laws)
                and all(ns.linear for ns in self._withdrawals)
                and all(st.linear for st in self._storage))

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

        Refused once the network has changed (`_require_current_topology`, one integer
        comparison): the endpoints and node count are construction-time ones. Every
        flow-balance site goes through here (`net_outflow`, `balance`, `residual`).
        """
        self._require_current_topology()
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

    def _scatter_entries(self, owners, positions, value) -> torch.Tensor | None:
        """The sum over `owners` of `value(j, owner)` (one value per node of owner `j`, at
        the compact interior rows `positions[j]`), scattered to the INTERIOR rows; `None`
        when there are no owners."""
        if not owners:
            return None
        out = None
        for j, (owner, pos) in enumerate(zip(owners, positions, strict=True)):
            v = value(j, owner)
            zeros = torch.zeros(
                v.shape[:-1] + (len(self.interior),), dtype=v.dtype, device=v.device
            )
            term = zeros.index_add(-1, pos, v)
            out = term if out is None else out + term
        return out

    def _scatter_withdrawals(self, value) -> torch.Tensor | None:
        """The sum over withdrawals of `value(j, ns)` (one value per node of source `j`),
        scattered to the INTERIOR rows; `None` when this layer has no withdrawals. Every
        withdrawal and its slope (with or without flow unknowns, primal or threaded) is
        assembled here."""
        return self._scatter_entries(self._withdrawals, self._withdrawal_positions, value)

    def _scatter_storage(self, value) -> torch.Tensor | None:
        """`_scatter_withdrawals` over the storages: `value(j, storage)` per storage."""
        return self._scatter_entries(self._storage, self._storage_positions, value)

    def _storage_args(self, j: int, st: NodeStorage, phi: torch.Tensor, step: _StorageStep):
        """`(phi_nodes, phi_start, previous)` of storage `j` at the full-node `phi`."""
        start = step.start_args.get(j)
        if start is None:
            previous = step.previous
            if previous is not None:
                s, e = self._storage_slices[j]
                previous = previous[..., s:e]
            start = (step.phi_prev[..., st.nodes], previous)
            # kept only when the step's own tensors carry no graph (the indexed column is a
            # copy, graph-free under no_grad whatever `phi_prev` is): no two residuals share
            # one, and a read under grad mode never gets a column without its graph
            if not (step.phi_prev.requires_grad
                    or (step.previous is not None and step.previous.requires_grad)):
                step.start_args[j] = start
        return (phi[..., st.nodes], *start)

    def _owner_call(self, i: int, args: tuple, kwargs: dict | None = None):
        """Owner `i` (a law, then past the laws a withdrawal, then a storage) called with its
        own parameters."""
        owners = (*self._laws, *self._withdrawals, *self._storage)
        return owners[i](*args, **(kwargs or {}))

    @staticmethod
    def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
        """`a + b`, where `None` is an absent term."""
        if b is None:
            return a
        return b if a is None else a + b

    def _withdrawal(
        self, phi: torch.Tensor, drivers: Mapping, step: _Step | None = None
    ) -> torch.Tensor | None:
        """Total potential-dependent withdrawal per INTERIOR row: the withdrawals', plus
        the storages' step rates on a step (`step`); `None` when there is none. `phi` is
        full-node."""
        w = self._scatter_withdrawals(lambda j, ns: ns.flow(phi[..., ns.nodes], drivers))
        if step is None or step.storage is None:
            return w
        return self._add(w, self._scatter_storage(
            lambda j, st: st.rate(*self._storage_args(j, st, phi, step.storage), step.ctx,
                                  drivers)))

    def _withdrawal_slopes(
        self, phi: torch.Tensor, drivers: Mapping, step: _Step | None = None
    ) -> torch.Tensor | None:
        """`diag(w')` per INTERIOR row: the withdrawals' slopes, plus the storages'
        `rate_slope` on a step; `None` when there is none."""
        d = self._scatter_withdrawals(lambda j, ns: ns.dflow(phi[..., ns.nodes], drivers))
        if step is None or step.storage is None:
            return d
        return self._add(d, self._scatter_storage(
            lambda j, st: st.rate_slope(
                *self._storage_args(j, st, phi, step.storage), step.ctx, drivers)))

    def _operator_at(
        self, slopes: torch.Tensor, node_slopes: torch.Tensor | None = None
    ) -> GraphLaplacianOperator | ShiftedOperator:
        """The matvec-free `A_I diag(slopes) A_I^T` operator at the given per-edge slope, in
        this layer's own endpoint/interior-index representation -- the construction repeated
        by `linear_init`, `solve`'s `operator_fn` (both the non-differentiable and
        differentiable branches) and `adjoint`. `slopes` is `dflows`'s actual Jacobian
        diagonal at the current iterate for all but `linear_init`, which instead passes its
        own tangent-at-zero `k` -- the same operator shape, at a different slope.

        `node_slopes` is the `diag(w')` a potential-dependent withdrawal adds to
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
        return ShiftedOperator(op, node_slopes)

    # ------------------------------------------------------------------ assembly
    def dp(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        """Every edge's potential difference `phi[src] - phi[tgt]` plus its head: the driver
        `"<layer>.edge_head"` (`noodl.heads.EdgeHeads`; absent means zero)."""
        self._require_current_topology()
        drivers = drivers or {}
        d = self._difference(phi)
        head = drivers.get(self.head_key)
        if head is not None:
            self._check_head_width(head)
            d = d + head
        return d

    def _check_head_width(self, head: torch.Tensor) -> None:
        """Refuse a head driver that is not one value per edge of this layer."""
        if head.ndim == 0 or head.shape[-1] != len(self.cols):
            given = "a scalar" if head.ndim == 0 else head.shape[-1]
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: driver {self.head_key!r} must have one value "
                f"per edge of the layer ({len(self.cols)}, in edge order of kinds "
                f"{tuple(self.kinds)}) in its last dimension; got width {given}"
            )

    def flows(
        self, phi: torch.Tensor, drivers: Mapping, q: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Every edge's flow at the full-node potentials `phi`, in layer edge order.

        On a layer without flow unknowns every flow is a function of its own edge's potential
        difference, so `phi` and `drivers` determine it; `q` is ignored. A law that reads the
        law context is evaluated in the steady context (`mode="dc"`, no time).

        A layer with flow unknowns (`implicit_edges`) has edges whose flow is NOT a
        function of `phi`: an implicit law (an ideal pressure or voltage source, a fan with
        a prescribed rise) is a relation `residual(dp, q) = 0` that may not involve `q` at
        all, and the flow is then fixed by conservation at the nodes, which depends on what
        else is attached (a storing node's rate, the other edges). The flows of those edges
        are therefore an INPUT here: pass `q`, the flows returned by `solve` (or by a
        previous state), with every edge's column, shape `(..., n_edges)`. Their columns
        (`implicit_edges`) are returned as given, and the potential-controlled edges'
        columns are evaluated from `phi` and `drivers` by their laws, in the steady context
        (`mode="dc"`). The result is differentiable in `phi`, `drivers` and the laws'
        parameters on the potential edges, and carries `q`'s graph on the implicit edges.
        At `solve`'s own `(phi, q)` it returns `q`. A phasor layer has no such accessor.
        Without `q` such a layer raises `ValueError`, naming the kinds that give a relation.
        """
        if self._unknowns is not None:
            return self._flows_given(phi, drivers or {}, q)
        return self._flows(phi, drivers or {}, _DC)

    def _flows(self, phi, drivers: Mapping, ctx: LawContext) -> torch.Tensor:
        """Every edge's flow on a layer without flow unknowns, at the full-node `phi`; a law
        that reads the context is given `ctx`, restricted to its edges (`_law_ctx`)."""
        d = self.dp(phi, drivers)
        if not self._ctx_laws:
            parts = [
                el.flow(d[..., s:e], drivers)
                for el, (s, e) in zip(self._laws, self._law_slices, strict=True)
            ]
        else:
            parts = [
                el.flow(d[..., s:e], drivers, self._law_ctx(ctx, s, e)) if el._takes_ctx
                else el.flow(d[..., s:e], drivers)
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

    def dflows(self, phi: torch.Tensor, drivers: Mapping,
               q: torch.Tensor | None = None) -> torch.Tensor:
        """Every edge's slope `d q / d dp` at the full-node `phi`, in layer edge order (a law
        that reads the law context in the steady context, as `flows`).

        On a layer without flow unknowns that is each law's own slope. On a layer with flow
        unknowns the edges of a flow-form law give theirs the same way, and an edge of a
        relation `r(dp, q) = 0` gives the slope the relation implies, `-(d r / d dp) / (d r /
        d q)`, taken at `q`'s flow on that edge: `q` is every edge's flow, as `solve` returns
        it, and is required when the layer has a relation (`ValueError` without it, as for
        `flows`). The implied slope is `nan` where `d r / d q` is zero, as on an ideal
        potential source, whose flow does not follow from its potential difference. Every
        edge of a law with several ports is `nan` too: its slope is a matrix, not a number
        per edge. A phasor layer has no such accessor (`TypeError`: its slopes are
        complex)."""
        if self._unknowns is None:
            return self._dflows(phi, drivers or {}, _DC)
        if self.phasor:
            raise TypeError(
                f"KirchhoffLayer {self.name!r} is a phasor layer: dflows() is not defined on "
                f"it (its slopes are complex); its Jacobian is jacobian()'s")
        drivers = drivers or {}
        q_F = self._flow_unknown_columns(q, "dflows", phi)
        index = self._unknowns.index
        ctx = LawContext(mode="dc")
        d = self.dp(phi, drivers)
        parts = []
        for i, (el, rel, (s, e)) in enumerate(zip(self._laws, self._relation,
                                                  self._law_slices, strict=True)):
            dp_s = d[..., s:e]
            if el.ports != 1:
                parts.append(torch.full_like(dp_s, math.nan))
            elif rel:
                f0 = int(index[s])
                dr_dp, dr_dq = el.jacobian(dp_s, q_F[..., f0:f0 + e - s], drivers,
                                           self._law_ctx(ctx, s, e))
                dr_dp, dr_dq = dr_dp[..., 0, 0, 0, 0], dr_dq[..., 0, 0, 0, 0]
                parts.append(torch.where(dr_dq != 0, -dr_dp / torch.where(
                    dr_dq != 0, dr_dq, torch.ones_like(dr_dq)),
                    torch.full_like(dr_dq, math.nan)))
            else:
                parts.append(self._law_slopes(i, el, dp_s, drivers, ctx))
        return _cat(parts, d)

    def _dflows(self, phi, drivers: Mapping, ctx: LawContext) -> torch.Tensor:
        """`dflows` in the context `ctx` (`_law_slopes` per law)."""
        d = self.dp(phi, drivers)
        if not self._ctx_laws:
            parts = [
                el.dflow(d[..., s:e], drivers)
                for el, (s, e) in zip(self._laws, self._law_slices, strict=True)
            ]
        else:
            parts = [
                self._law_slopes(i, el, d[..., s:e], drivers, ctx)
                for i, (el, (s, e)) in enumerate(
                    zip(self._laws, self._law_slices, strict=True))
            ]
        return torch.cat(parts, dim=-1)

    def _law_slopes(self, i: int, el, dp, drivers, ctx, adjoint: bool = False):
        """Law `i`'s slopes `d q / d dp` at its edges' `dp`, in the context `ctx` (the
        step's; restricted here to the law's edges). A law that does not read the context
        gives its `dflow`. One that does has no `dflow` with a context: its slope is the
        one-port block of its `jacobian` (its analytic one where it overrides it, else
        autograd of its `forward` with `ctx`); with `adjoint` (the operator also builds the
        adjoint) a law that declares `exact_slope = False` gives autograd of its `forward`,
        as on the general path (`_operator`)."""
        if not el._takes_ctx:
            return el.dflow(dp, drivers)
        lctx = self._law_ctx(ctx, *self._law_slices[i])
        if adjoint and not el.exact_slope:
            block = el._coloured_jacobian(dp, None, drivers, lctx)[0]
        else:
            block = el.jacobian(dp, None, drivers, lctx)[0]
        return block[..., 0, 0, 0, 0]

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

    def residual(self, phi_interior, phi_boundary, drivers, sources, *, q=None, dt=None,
                 scheme=None, history=None, storage_flow=None):
        """The Newton residual at the given point.

        On a layer without flow unknowns it is the interior balance `A_I q(phi) - s_I (+
        w(phi))`, `n_I` rows; `q` is ignored. On a layer with flow unknowns the unknowns are
        the interior potentials and the flows of the relations' edges, `[phi_I, q_F]` in that
        order, and the residual is the balance rows, then the relations' own residuals:
        `q` is every edge's flow, as `solve` returns it (its flow-unknown columns are read),
        and is required (`ValueError` without it). A phasor layer's rows are complex.

        The step keywords are `solve`'s. On a layer with storage, with `dt` the storage
        withdrawal of that step is included (the residual `solve` drives to zero); without
        `dt` it is the steady residual, where storage is absent. A layer without storage
        refuses them."""
        self._require_current_topology()
        drivers = drivers or {}
        step = self._public_step(dt, scheme, history, storage_flow, drivers)
        if self._unknowns is None:
            return self._residual(phi_interior, phi_boundary, drivers, sources, step)[0]
        phi_boundary, sources, phi_interior, q = self._complex_inputs(
            phi_boundary, drivers, sources, phi_interior, q)
        x = self._unknown_vector(phi_interior, q, "residual")
        r = self._residual(x, phi_boundary, drivers, sources, step)[0]
        return to_complex(r) if self.phasor else r

    def _public_step(self, dt, scheme, history, storage_flow, drivers) -> _Step:
        """The `_Step` of `residual`, `jacobian` and `adjoint`: their step keywords,
        checked as `solve` checks them (refused on a layer that takes no step)."""
        if scheme is None:
            scheme = self.scheme
        self._check_forward_euler(scheme)
        return self._step(None, dt, None, scheme, history, storage_flow, drivers)

    def balance(self, phi, q, sources=None, drivers=None, storage_flow=None):
        """The interior flow balance `A_I q - s_I (+ w(phi)) (+ storage)` of a state's own
        flows.

        `phi` is full-node, `q` every edge's flow in layer edge order (what `solve`
        returns), `sources` full-node (`None` means zero) and `drivers` what the node
        sources read. `storage_flow`, on a layer with storage, is the flow into every storage
        entry (`storage_flows`, the state key `"<layer>.storage_flow"`), taken as given like
        `q`; `None` leaves storage out (a steady state). Zero (to solver tolerance) at a
        solved state. Unlike `residual`, which recomputes the flows from `phi`, this takes
        them as given, so it is defined on every layer, including one with flow unknowns,
        whose relations' flows are not functions of `phi`.
        """
        self._require_current_topology()
        phi_i = phi[..., self.interior]
        lhs = self._accumulate_interior(q) - self._source_interior(sources, phi_i)
        w = self._withdrawal(phi, drivers or {})
        if storage_flow is not None and self._storage:
            w = self._add(w, self._scatter_storage(
                lambda j, st: storage_flow[..., slice(*self._storage_slices[j])]))
        return lhs if w is None else lhs + w

    def _residual(self, x, phi_boundary, drivers, sources, step=None, call=None,
                  tangents=None):
        """`(residual, phi, q)`: the residual at the unknowns `x` and the full-node potentials
        and every edge's flow it was evaluated at (`solve` reuses the last two instead of
        recomputing them). `step` is the solve's `_Step` (`None` is the steady point without
        storage).

        On the fast path `x` is the interior potentials, and the residual is the balance
        `A_I q(phi) - s_I (+ w(phi))`, with the storage withdrawal on a step; `call` is not
        used (the laws are the layer's own). With flow unknowns `x` is `[phi_I, q_F]` in the
        real view, the residual is the balance rows, then the relations' own residuals, and
        `call(i, args)` evaluates owner `i` (`_owner_call`, the default, or the threaded
        parameters of the differentiable solve); a step adds the storage withdrawal, and the
        initial point replaces the held rows (`_Held`). The layout is the step's when it
        has one (`_Step.layout`). `tangents` (`_tangents`, the linear guess only) puts each
        listed law's tangent at zero in place of its flow."""
        u = self._unknowns if step is None or step.layout is None else step.layout
        if u is None:
            phi = self.assemble(x, phi_boundary)
            q = self._flows(phi, drivers, _DC if step is None else step.ctx)
            # (A_I q), by scatter-add over this layer's edges then a select of the interior
            # rows, rather than einsum against the (n_I, b) slice of the dense incidence:
            # this is once per Newton residual evaluation, where the einsum form dominates
            # the cost (numbers: docs/design/orchestration.md#residual-incidence-product).
            lhs = self._accumulate_interior(q)
            s_I = self._source_interior(sources, x)
            # `(lhs - s) + (w_withdrawal + w_storage)`: `_withdrawal` adds the two first
            w = self._withdrawal(phi, drivers, step)
            return (lhs - s_I if w is None else lhs - s_I + w), phi, q
        if call is None:
            call = self._owner_call
        ctx, store = (LawContext(mode="dc"), None) if step is None else (step.ctx, step.storage)
        phi_I, q_F = split(u, x)
        phi = self.assemble(phi_I, phi_boundary)
        q, r_F = self._flows_and_residuals(phi, q_F, drivers, ctx, call, u, tangents)
        kcl = self._accumulate_interior(q) - self._source_interior(sources, phi_I)
        # `((lhs - s) + w_withdrawal) + w_storage`, not the fast path's association
        w = self._withdrawal_called(phi, drivers, call)
        if w is not None:
            kcl = kcl + w
        if isinstance(store, _Held):
            _, value = self._held_values(drivers, call)
            kcl = torch.where(store.rows, phi_I - value.to(phi_I.dtype), kcl)
        elif store is not None:
            kcl = kcl + self._storage_withdrawal(phi, drivers, store, ctx, call)
        return to_real(self, _cat([kcl, r_F], kcl)), phi, q

    def _operator(self, x, phi_boundary, drivers, step=None, slopes=None,
                  adjoint: bool = False, tangents=None):
        """The Jacobian of `_residual` at the unknowns `x`.

        On the fast path the matvec-free `A_I diag(dflows) A_I^T` (`_operator_at`), shifted
        by the withdrawals' and, on a step, the storages' slopes; `slopes` and `adjoint` are
        not used. With flow unknowns the COO triplets on the fixed pattern (a
        `SparseOperator`), with the rows the initial point holds replaced (`_hold`).
        `slopes(phi, adjoint)` gives the diagonal of the withdrawals and storages
        (`_slopes_fn`; by default the layer's own, with `step`). `adjoint`: the operator also
        builds the adjoint (the differentiable solve). A law that declares
        `exact_slope = False` then gives its blocks by autograd of its own `forward`
        (`_coloured_jacobian`), not its `jacobian`, whose `dflow` shortcut would hand the
        adjoint a Newton aid; the same flag reaches `slopes`, where
        an inexact withdrawal or storage gives autograd too. The primal solve
        (`adjoint=False`) keeps the analytic slopes. `tangents` is `_residual`'s: a listed
        law's block is the slope of its tangent at zero."""
        u = self._unknowns if step is None or step.layout is None else step.layout
        if u is None:
            phi = self.assemble(x, phi_boundary)
            dq = self._dflows(phi, drivers, _DC if step is None else step.ctx)
            return self._operator_at(dq, self._withdrawal_slopes(phi, drivers, step))
        ctx = LawContext(mode="dc") if step is None else step.ctx
        phi_I, q_F = split(u, x)
        phi = self.assemble(phi_I, phi_boundary)
        dp = self.dp(phi, drivers)
        blocks = []
        for i, (el, rel, (s, e)) in enumerate(zip(self._laws, self._relation,
                                                  self._law_slices, strict=True)):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            # Called on the law module itself, also on the differentiable path: the values
            # equal those of a `functional_call` with the threaded parameters, because the
            # threaded tensors ARE the module's own parameters (the same argument as
            # `dp_independent` in `_functional.py`), and the blocks carry no graph.
            if rel:
                f0 = int(u.index[s])
                q_s = q_F[..., f0:f0 + e - s]
            else:
                q_s = None
            if tangents is not None and i in tangents:
                k = tangents[i][1] + torch.zeros_like(dp_s)
                blocks.append(k[..., None, None, None, None])
            elif adjoint and not el.exact_slope:
                blocks += el._coloured_jacobian(dp_s, q_s, drivers, lctx)
            else:
                blocks += el.jacobian(dp_s, q_s, drivers, lctx)
        parts = [blk.flatten(-5) for blk in blocks]
        parts.append(torch.ones(1, dtype=x.dtype, device=x.device))
        if u.diag:
            if slopes is None:
                slopes = self._slopes_fn(drivers, self._owner_call, step)
            d = slopes(phi, adjoint)
            parts.append(d if d is not None else
                         torch.zeros(u.n_I, dtype=x.dtype, device=x.device))
        flat = _cat(parts, x)
        values = flat[..., u.src] * u.sign.to(dtype=flat.dtype, device=flat.device)
        op = SparseOperator(u.row, u.col, values, u.n)
        return op if step is None else self._hold(op, step.storage)

    # ------------------------------------------------------------------ flow unknowns
    def _complex_inputs(self, phi_boundary, drivers, sources, phi0, q0):
        """`(phi_boundary, sources, phi0, q0)` as a solve with flow unknowns takes them: on a
        phasor layer in the complex dtype of its network's precision (complex128 on a
        float64 network), real or complex as given, with the frequency driver checked
        against their batch."""
        if not self.phasor:
            return phi_boundary, sources, phi0, q0
        dtype = cdtype(self)
        phi_boundary, sources, phi0, q0 = (
            None if v is None else torch.as_tensor(v).to(dtype)
            for v in (phi_boundary, sources, phi0, q0))
        check_frequency(self, drivers, [
            v.shape[:-1] for v in (phi_boundary, sources, phi0, q0) if v is not None])
        return phi_boundary, sources, phi0, q0

    def _refuse_newton_options(self, newton_kwargs,
                               what: str = "a layer with flow unknowns") -> None:
        """Refuse the Newton keywords a layer with flow unknowns (or `what` else solves on
        the sparse operator) cannot take, by name: the convex line search and conjugate
        gradients."""
        if newton_kwargs.get("line_search") == "convex":
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: line_search='convex' needs a residual "
                f"that is the gradient of a convex function, which the residual of {what} "
                f"is not; use the default line search"
            )
        if newton_kwargs.get("method", self.linear_solver) == "cg":
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: method='cg' needs a symmetric positive "
                f"definite Jacobian, which {what} does not have; use "
                f"'auto', 'sparse_direct', 'direct' or 'gmres'"
            )

    def _law_ctx(self, ctx: LawContext, s: int, e: int) -> LawContext:
        """`ctx` with the step-start values restricted to one law's edges `[s, e)`. The
        restrictions of the last context seen are kept: a Newton solve calls this with the
        same context on every iteration."""
        if ctx.dp_prev is None and ctx.q_prev is None:
            return ctx
        cached_ctx, by_slice = self._law_ctxs
        if cached_ctx is not ctx:
            by_slice = {}
            self._law_ctxs = (ctx, by_slice)
        lctx = by_slice.get((s, e))
        if lctx is None:
            changes = {}
            if ctx.dp_prev is not None:
                changes["dp_prev"] = ctx.dp_prev[..., s:e]
            if ctx.q_prev is not None:
                changes["q_prev"] = ctx.q_prev[..., s:e]
            lctx = by_slice[(s, e)] = dataclasses.replace(ctx, **changes)
        return lctx

    def _check_width(self, el, out, n_edges: int, what: str) -> None:
        """A law returns one value per edge of its kind: refuse any other trailing width."""
        if out.dim() == 0 or out.shape[-1] != n_edges:
            width = "a scalar" if out.dim() == 0 else f"width {out.shape[-1]}"
            raise ValueError(
                f"layer {self.name!r}: the {what} of the {el.kind!r} law is {width} "
                f"but the kind has {n_edges} edges")

    def _flows_and_residuals(self, phi, q_F, drivers, ctx, call, u=None, tangents=None):
        """`(q, r_F)`: every edge's flow (the law flow on potential-controlled edges, the
        unknown on relation edges) and the relations' residuals, in layer edge order; `u` is
        the layout (the layer's own by default). A law listed in `tangents` (`_tangents`)
        gives its tangent at zero, `c + k dp`, in place of its flow."""
        index = (self._unknowns if u is None else u).index
        dp = self.dp(phi, drivers)
        q_parts, r_parts = [], []
        for i, (el, (s, e)) in enumerate(
                zip(self._laws, self._law_slices, strict=True)):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            if self._relation[i]:
                f0 = int(index[s])
                q_s = q_F[..., f0:f0 + e - s]
                r_s = call(i, (dp_s, q_s, drivers, lctx))
                self._check_width(el, r_s, e - s, "residual")
                r_parts.append(r_s)
            elif tangents is not None and i in tangents:
                c, k = tangents[i]
                q_s = c + k * dp_s
                self._check_width(el, q_s, e - s, "flow")
            else:
                q_s = call(i, (dp_s, drivers, lctx))
                self._check_width(el, q_s, e - s, "flow")
            q_parts.append(q_s)
        return _cat(q_parts, dp), _cat(r_parts, dp)

    def _flows_given(self, phi, drivers, q) -> torch.Tensor:
        """`flows` of a layer with flow unknowns: `q`'s flow-unknown columns as given, the
        other edges' flows from their laws at `phi` (steady context)."""
        if self.phasor:
            raise TypeError(
                f"KirchhoffLayer {self.name!r} is a phasor layer: flows() is not defined on "
                f"it; read the flows from solve()'s q")
        q_F = self._flow_unknown_columns(q, "flows", phi)
        out, _ = self._flows_and_residuals(phi, q_F, drivers,
                                           LawContext(mode="dc"), self._owner_call)
        return out

    def _withdrawal_called(self, phi, drivers, call):
        """The withdrawals' withdrawal per interior row, each called through `call`; `None`
        without withdrawals."""
        n_el = len(self._laws)
        return self._scatter_withdrawals(
            lambda j, ns: call(n_el + j, (phi[..., ns.nodes], drivers)))

    def _node_slopes(self, phi, drivers, adjoint: bool = False):
        """`d w / d phi` per interior row: a withdrawal reads its own node's potential only,
        so its own `dflow` is the whole diagonal; except, when the operator also builds the
        adjoint (`adjoint`, the differentiable solve), for a withdrawal that declares
        `exact_slope = False`, whose slope is then the autograd derivative of its flow."""
        def slope(j, ns):
            phi_n = phi[..., ns.nodes]
            if adjoint and not ns.exact_slope:
                leaf = phi_n.detach().requires_grad_(True)
                with torch.enable_grad():
                    (grad,) = torch.autograd.grad(ns(leaf, drivers).sum(), leaf)
                return grad
            return ns.dflow(phi_n, drivers)

        return self._scatter_withdrawals(slope)

    def _held_values(self, drivers, call, *, check: bool = False):
        """`(rows, value)`: the interior rows held at the initial point ((n_I,) bool) and
        the potential each is held at ((..., n_I), zero elsewhere); `None` when no storage
        holds an initial potential. `initial` is evaluated through `call`, so on the
        differentiable path with the threaded parameters. `check` refuses entries at one
        node with different initial values, by name."""
        n_I = len(self.interior)
        base = len(self._laws) + len(self._withdrawals)
        held = []
        for j, _st in enumerate(self._storage):
            v = call(base + j, (drivers,), {"part": "initial"})
            if v is not None:
                pos = self._storage_positions[j]
                v = torch.as_tensor(v, device=pos.device)
                held.append((j, pos, v.expand(v.shape[:-1] + pos.shape) if v.dim()
                             else v.expand(pos.shape)))
        if not held:
            return None
        batch = broadcast_shapes(*(v.shape[:-1] for _, _, v in held))
        dtype = held[0][2].dtype
        value = torch.zeros(batch + (n_I,), dtype=dtype, device=held[0][2].device)
        rows = torch.zeros(n_I, dtype=torch.bool, device=value.device)
        for _, pos, v in held:
            value = value.index_copy(-1, pos, v.to(dtype).expand(batch + pos.shape))
            rows[pos] = True
        if check:
            for _, pos, v in held:
                differ = (v.to(dtype) != value[..., pos]).reshape(-1, pos.numel()).any(0)
                if bool(differ.any()):
                    row = pos[differ.nonzero()[0, 0]]
                    names = [f"{type(self._storage[k]).__name__} (storage {k})"
                             for k, p, _ in held if bool((p == row).any())]
                    node = self.net.nodes[int(self.interior[row])]
                    raise ValueError(
                        f"KirchhoffLayer {self.name!r}: the storage entries at node "
                        f"{node!r} hold different initial potentials ({', '.join(names)}); "
                        f"mode='initial' holds a node at one value")
        return rows, value

    def _held(self, found, u: FlowUnknowns) -> _Held:
        """The held rows of `mode="initial"` (`found`, from `_held_values`) and their masks
        on the values of the Jacobian of layout `u`."""
        rows = found[0].to(u.row.device)
        full = torch.zeros(u.n, dtype=torch.bool, device=rows.device)
        full[: u.n_I] = rows
        diag = torch.arange(len(u.row) - u.n_I, len(u.row), device=rows.device)
        return _Held(rows=rows, clear=full[u.row], ones=diag[rows])

    def _held_layout(self) -> FlowUnknowns:
        """The layout of the initial point with held storage on a layer without flow
        unknowns: no unknown flows, the balance rows and their diagonal slots (`flow_unknowns`
        of this layer). Built on the first such solve and kept; no other solve reads it."""
        if self._held_unknowns is None:
            self._held_unknowns = flow_unknowns(self)
        return self._held_unknowns

    def _storage_withdrawal(self, phi, drivers, store, ctx, call):
        """The storages' withdrawal `NodeStorage.rate` per interior row on a transient
        step (`store`, a `_StorageStep`), in the context `ctx`."""
        base = len(self._laws) + len(self._withdrawals)
        return self._scatter_storage(lambda j, st: call(
            base + j, (*self._storage_args(j, st, phi, store), ctx, drivers)))

    def _storage_slopes(self, phi, drivers, store, ctx, call, adjoint: bool):
        """`d w / d phi` of the storages per interior row on a transient step (`store`, a
        `_StorageStep`; a storage reads its own node's potential only, so this is the whole
        diagonal): their analytic `rate_slope`, except, when the operator also builds the
        adjoint (`adjoint`, the differentiable solve), for a storage that declares
        `exact_slope = False`, whose slope is then the derivative of its `rate` by autograd.
        The values carry no graph, like every Jacobian entry."""
        base = len(self._laws) + len(self._withdrawals)

        def slope(j, st):
            phi_n, phi_start, previous = self._storage_args(j, st, phi, store)
            if adjoint and not st.exact_slope:
                block, = _colour_blocks(
                    lambda v: call(base + j, (v, phi_start, previous, ctx, drivers)),
                    (phi_n,), 1, 1)
                return block[..., 0, 0, 0, 0]
            return st.rate_slope(phi_n, phi_start, previous, ctx, drivers)

        d = self._scatter_storage(slope)
        return None if d is None else d.detach()

    def _slopes_fn(self, drivers, call, step):
        """The diagonal of the withdrawals and, on a step, of the storages, as a function
        `slopes(phi, adjoint)` of the full-node potentials (`None` when there is neither).
        `adjoint` is `_operator`'s own flag, passed on to `_node_slopes` and
        `_storage_slopes`: the operator also builds the adjoint."""
        ctx, store = (LawContext(mode="dc"), None) if step is None else (step.ctx, step.storage)
        stepping = store is not None and not isinstance(store, _Held)

        def slopes(phi, adjoint: bool = False):
            d = self._node_slopes(phi, drivers, adjoint)
            if not stepping:
                return d
            s = self._storage_slopes(phi, drivers, store, ctx, call, adjoint)
            return s if d is None else d + s

        return slopes

    def _hold(self, op: SparseOperator, store) -> SparseOperator:
        """`op` with each row `mode="initial"` holds replaced by the row of
        `phi_i - initial_i`: its values zeroed, 1 on its diagonal slot."""
        if not isinstance(store, _Held):
            return op
        values = op.values.masked_fill(store.clear, 0.0)
        values[..., store.ones] = 1.0
        return SparseOperator(op.row, op.col, values, op.shape[-1])

    def jacobian(self, phi_interior, phi_boundary, drivers, *, q=None, dt=None, scheme=None,
                 history=None, storage_flow=None):
        """The dense Jacobian of `residual` at the given point.

        On a layer without flow unknowns it is the `(n_I, n_I)` matrix A_I diag(dq) A_I^T --
        the retained dense reference. That is the one method that reads `self.A` (and so
        materialises it, once, on first access); no solve path calls it. Tests compare
        `GraphLaplacianOperator`'s matvec and the adjoint against this, so its einsum form is
        deliberately unchanged; `q` is ignored. On a layer with flow unknowns it is the
        matrix of the operator `solve`'s Newton iteration builds, `(n_I + n_F)` square over
        the unknowns `[phi_I, q_F]` (`residual`), with `q` required; a phasor layer's matrix
        is in the real view (each complex unknown a (Re, Im) pair, twice the width).

        The step keywords are `residual`'s: with `dt`, a layer's storage adds its `rate_slope`
        to the diagonal; without, it is the steady Jacobian.
        """
        self._require_current_topology()
        drivers = drivers or {}
        step = self._public_step(dt, scheme, history, storage_flow, drivers)
        if self._unknowns is not None:
            phi_boundary, _, phi_interior, q = self._complex_inputs(
                phi_boundary, drivers, None, phi_interior, q)
            x = self._unknown_vector(phi_interior, q, "jacobian")
            return self._operator(x, phi_boundary, drivers, step).assemble()
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self._dflows(phi, drivers, step.ctx)
        A_I = self.A[self.interior]
        dense = torch.einsum("ie,...e,je->...ij", A_I, dq, A_I)
        node_slopes = self._withdrawal_slopes(phi, drivers, step)
        if node_slopes is None:
            return dense
        return dense + torch.diag_embed(node_slopes.expand(dense.shape[:-1]))

    def _linear_ck(self, drivers, ctx: LawContext = _DC, device=None):
        # Each law's own (c_e, k_e) covers only its own n_e = e - s edges; only the
        # leading BATCH dims (everything but the trailing edge-count dim) are meant to be
        # unified across laws before concatenating along the edge axis. Using
        # torch.broadcast_tensors directly on the raw (c_e, k_e) list would broadcast the
        # trailing edge dim too, so two laws with different edge counts silently expand to
        # a shared (wrong) edge count instead of concatenating -- e.g. a 2-edge FixedFlow and
        # a 1-edge PowerLaw would both become
        # 2-edge before torch.cat, yielding a 4-wide result instead of the correct 3.
        # A law that reads the context has no `linear_init` with one: its tangent at dp = 0
        # is its flow and slope there, in the solve's context (`_law_slopes`).
        cs, ks, n_es = [], [], []
        for i, (el, (s, e)) in enumerate(zip(self._laws, self._law_slices, strict=True)):
            c_e, k_e = self._law_tangent(i, el, drivers, ctx, device)
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

    def _law_tangent(self, i: int, el, drivers, ctx: LawContext, device=None):
        """`(c, k)`: law `i`'s tangent at `dp = 0`, its flow `c + k dp` near zero. Its own
        `linear_init`; a law that reads the context has none with one, and gives its flow and
        slope at zero in the solve's context `ctx` (`_law_slopes`)."""
        if el._takes_ctx:
            s, e = self._law_slices[i]
            zero = torch.zeros(e - s, dtype=el._dtype(), device=device)
            return (el.flow(zero, drivers, self._law_ctx(ctx, s, e)),
                    self._law_slopes(i, el, zero, drivers, ctx))
        return el.linear_init(drivers)

    def _tangents(self, drivers, ctx: LawContext, device=None) -> dict:
        """`{i: (c, k)}`: the tangent at zero (`_law_tangent`) of every law `i` of a layer with
        flow unknowns that has one, a real one-port law with a flow. A relation has none (its
        flow is an unknown), and neither has a law of several ports or a phasor law, whose
        tangent couples edges or is complex: the linear guess evaluates those as Newton does,
        at the potentials it starts from."""
        if self.phasor:
            return {}
        return {i: self._law_tangent(i, el, drivers, ctx, device)
                for i, (el, rel) in enumerate(zip(self._laws, self._relation, strict=True))
                if not rel and el.ports == 1}

    def _linear_guess(self, phi_boundary, drivers, sources, step: _Step) -> torch.Tensor:
        """The linear guess of a layer with flow unknowns, `[phi_I, q_F]` in the real view:
        one Newton step from zero (interior potentials and flow unknowns zero) on the system
        whose laws with a flow are replaced by their tangents at zero (`_tangents`). The
        relations give their residual and Jacobian at that point, the withdrawals and the
        storages their value and slope there, and the initial point holds its held rows.
        `step` is the solve's `_Step`. Runs without gradients."""
        u = self._unknowns
        real_dtype = (phi_boundary.real.dtype if phi_boundary.is_complex()
                      else phi_boundary.dtype)
        device = phi_boundary.device
        call = self._owner_call
        # outside no_grad: a law that reads the context may take its slope by autograd
        # (`_law_slopes`), which needs grad mode
        tangents = self._tangents(drivers, step.ctx, device)
        with torch.no_grad():
            x = torch.zeros(u.n, dtype=real_dtype, device=device)
            r = self._residual(x, phi_boundary, drivers, sources, step, call, tangents)[0]
            op = self._operator(x, phi_boundary, drivers, step,
                                self._slopes_fn(drivers, call, step), tangents=tangents)
            batch = broadcast_shapes(op.values.shape[:-1], r.shape[:-1])
            values = op.values.expand(batch + op.values.shape[-1:])
            op = SparseOperator(op.row, op.col, values, u.n)
            r = r.expand(batch + r.shape[-1:])
            where = f"KirchhoffLayer {self.name!r} linear_init"
            result = select_solve(op, r, method=linear_method(self), on_failure="return",
                                  where=where, rtol=inner_solve_rtol(r.dtype))
            if not bool(torch.all(result.converged)):
                failing = int(torch.nonzero(~result.converged.reshape(-1))[0, 0])
                if singular(op, u.n, failing):
                    raise self._singular_error(step.ctx.mode)
                result.raise_on_failure(where)
            return -result.x

    def _singular_error(self, mode: str) -> RuntimeError:
        """`singular_error` of this layer, naming its relation kinds."""
        kinds = [el.kind for el, rel in zip(self._laws, self._relation, strict=True) if rel]
        return singular_error(self, kinds, mode)

    def _extra_grounded(self, node_slopes: torch.Tensor | None) -> torch.Tensor | None:
        """Full-node-order bool from `node_slopes` (per COMPACT interior row, as
        `_withdrawal_slopes` returns): True at an interior node whose own withdrawal slope
        is strictly positive there. Such a node is grounded independent of any edge
        path -- `ShiftedOperator`'s positive diagonal shift there is exactly the SPD
        contribution a boundary connection would otherwise have to supply -- so
        `_grounding_check` must treat it exactly like a boundary node when deciding whether
        the rest of the graph reaches ground. `None` when this layer has no withdrawals,
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

        `node_slopes`, when this layer has withdrawals, is `diag(w')`
        per compact interior row -- the same tensor `_withdrawal_slopes` returns. A node
        with a strictly positive entry there is grounded independent of any edge path (see
        `_extra_grounded`), so passing it here is what keeps a node grounded ONLY through a
        withdrawal from being reported ungrounded.
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

    def linear_init(self, phi_boundary, drivers, sources, *, q=False, _step=None):
        """The solution of the network linearised at zero interior potentials
        (`ConstitutiveLaw.linear_init`): the Newton starting guess, the interior potentials.
        `_step` (internal) is the solve's step: a layer with storage linearises its
        withdrawal the same way, and a law that reads the context is linearised in the
        step's context (the steady one without; on a phasor layer the phasor one).

        On a layer with flow unknowns (`implicit_edges`) it is the linear guess of the
        potentials and the flow unknowns together: each law with a flow is replaced by its
        tangent at zero (a law with several ports, and every law on a phasor layer, is taken
        at its flow and Jacobian at the starting potentials instead: the same for a linear
        law), each relation by its residual and Jacobian at zero interior potentials and
        zero flows, and the system is solved once. On a layer with flow unknowns the result
        is detached: no gradient flows to `phi_boundary`, `sources`, the drivers or the law
        parameters, unlike on a layer without flow unknowns. A phasor layer returns complex
        values. `q=True` returns
        `(phi_interior, q_implicit)`, the second the flows of `implicit_edges` (empty on a
        layer without flow unknowns)."""
        drivers = drivers or {}
        if self._unknowns is not None:
            phi_boundary, sources, _, _ = self._complex_inputs(
                phi_boundary, drivers, sources, None, None)
            if _step is None:
                _step = self._step(None, None, None, self.scheme, None, None, drivers)
            phi_I, q_F = split(self._unknowns,
                               self._linear_guess(phi_boundary, drivers, sources, _step))
            return (phi_I, q_F) if q else phi_I
        phi_I = self._nodal_linear_init(phi_boundary, drivers, sources, _step)
        return (phi_I, phi_I.new_zeros(phi_I.shape[:-1] + (0,))) if q else phi_I

    def _nodal_linear_init(self, phi_boundary, drivers, sources, _step):
        """`linear_init` of a layer without flow unknowns."""
        batch_shape = phi_boundary.shape[:-1]
        phi_i0 = torch.zeros(
            batch_shape + (len(self.interior),),
            dtype=phi_boundary.dtype,
            device=phi_boundary.device,
        )
        phi0 = self.assemble(phi_i0, phi_boundary)
        dp0 = self.dp(phi0, drivers)
        c, k = self._linear_ck(drivers, _DC if _step is None else _step.ctx,
                               phi_boundary.device)
        rhs = self._source_interior(sources, phi0) - self._accumulate_interior(c + k * dp0)
        # A withdrawal contributes its tangent at phi = 0 to the same linearisation:
        # w(phi) ~ w(0) + w'(0) phi, so w(0) moves to the right-hand side and w'(0) joins
        # the operator's diagonal.
        node_slopes = None
        if self._withdrawals or (_step is not None and _step.storage is not None):
            w0 = self._withdrawal(phi0, drivers, _step)
            node_slopes = self._withdrawal_slopes(phi0, drivers, _step)
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
        tensor a ConstitutiveLaw happens to hold with `requires_grad=True` is captured only
        by closure (the law object itself, not its tensor payload) and is
        invisible to `_Implicit.backward`: for a ConstitutiveLaw, this happens when it was
        constructed with `learnable=False` on a tensor that already had `requires_grad=True`
        (`ConstitutiveLaw._param`'s documented pass-through exception, which is still a correct
        construction -- its graph is reachable through this layer's own component calls
        (`flows`, `dflows`, `residual`, `jacobian`, `linear_init`), though NOT through
        `solve(differentiable=False)`, which returns detached tensors).
        This raises now, before dispatching, rather than returning a gradient that is silently
        wrong (if some other path happens to also touch the same value) or silently `None` (if
        it doesn't).
        """
        for role, el, name, value in self._held_scan():
            if not value.requires_grad:
                continue
            if role == "storage":
                raise ValueError(
                    f"storage {el!r} has a tensor attribute {name!r} with "
                    f"requires_grad=True that is not a registered nn.Parameter, so "
                    f"the differentiable solve cannot reach it through "
                    f"Function.apply and its gradient would be silently wrong or "
                    f"absent. Construct it with learnable=True, or register it as an "
                    f"nn.Parameter in __init__, or use differentiable=False."
                )
            if role == "source":
                # A NodeWithdrawal has no `learnable=` constructor kwarg (unlike
                # ConstitutiveLaw) -- its fix is to register the tensor as an nn.Parameter
                # directly, so it must be named and fixed as a withdrawal, not
                # rendered (and told to construct itself) as if it were a ConstitutiveLaw.
                raise ValueError(
                    f"withdrawal {el!r} has a tensor attribute {name!r} with "
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

    def _held_scan(self) -> list:
        """`(role, owner, name, tensor)` for every tensor the laws, withdrawals, storages
        (every submodule: a law may wrap another, `Scaled`) hold where the
        differentiable solve does not thread it (`_held_tensors`). The scan is cached: it is
        repeated only when an object it read holds something else (`_shallow`), and the tensors'
        `requires_grad` is read live."""
        cache = getattr(self, "_held_cache", None)
        if cache is not None and all(_shallow(o) == sig for o, sig in cache[1]):
            return cache[0]
        visited: list = [self._laws, self._withdrawals, self._storage]
        found = []
        for role, owners in (("law", self._laws), ("source", self._withdrawals),
                             ("storage", self._storage)):
            for el in owners:
                holders = el.modules() if isinstance(el, torch.nn.Module) else (el,)
                for h in holders:
                    found += [(role, el, n, v) for n, v in _held_tensors(h, visited=visited)]
        # the visited objects are kept alive by the cache, so their ids are not reused
        self._held_cache = (found, [(o, _shallow(o)) for o in visited])
        return found

    def _require_threaded(self, residual, params) -> None:
        """Refuse a differentiable solve whose residual depends on a tensor that is not threaded.

        `residual(*params)` is the residual at one point (the starting guess before the
        solve, the converged root after it) as a function of exactly the tensors handed to
        `Function.apply`. It is evaluated once, under `enable_grad`, with every one of them
        detached: if the result still requires grad, some tensor that requires grad reaches
        the residual by another route (held by a law, a withdrawal in a plain
        object, a module in a plain list, or captured by a closure), and the implicit
        adjoint would return a gradient for it that is silently wrong or absent.

        What this establishes is structural and holds AT THAT POINT only: a law that reads
        an un-threaded tensor on a branch inactive there is not seen. The differentiable
        solve therefore applies it to every residual its forward Newton evaluates
        (`_newton_differentiable`, `_threaded_residual`, at no extra evaluation), from the
        starting guess to the root, which is the point the adjoint linearises at (its
        residual VJP and Jacobian operator are both evaluated there); the Jacobian operator
        is not checked separately: it is the same laws' slopes, their exact `dflow` or the
        autograd derivative of their flows (`FunctionalLaws.dflows`), a constant of the
        adjoint.
        Tensors held as attributes, buffers or in containers are refused before solving
        by `_check_no_unreachable_differentiable_tensors`, which also names them; this
        check catches the rest (a closure capture) and describes the layer.
        """
        self._threaded_residual(residual, params)

    def _threaded_residual(self, residual, params, *, detach: bool = True):
        """`residual(*params)` evaluated with every one of `params` detached (already, with
        `detach=False`), under grad mode, and refused (`_require_threaded`) if it still
        requires grad; returned otherwise (it then carries no graph)."""
        detached = _detached(params) if detach else params
        with torch.enable_grad():
            r = residual(*detached)
        if not (isinstance(r, torch.Tensor) and r.requires_grad):
            return r
        raise ValueError(
            f"KirchhoffLayer {self.name!r}: the residual depends on a tensor with "
            f"requires_grad=True that is not a registered nn.Parameter of a law or node "
            f"source, a driver, the sources or the boundary potentials (it is captured by "
            f"closure or held where no attribute scan reaches), so the differentiable solve "
            f"cannot reach it through Function.apply and its gradient would be silently "
            f"wrong or absent. Register it as an nn.Parameter of the law, pass it through "
            f"the `drivers` mapping, or use differentiable=False."
        )

    # ------------------------------------------------------------------ solve
    def _reaches_grad(self, drivers: Mapping, sources, phi_boundary, extra=()) -> bool:
        """Whether anything a differentiable solve threads through `Function.apply` requires
        grad under the current grad mode: a driver, `sources`, `phi_boundary`, a law's, node
        source's or storage's parameter, or one of `extra` (the step-start tensors of a step
        of a layer with storage)."""
        return torch.is_grad_enabled() and self._threads_grad(drivers, sources, phi_boundary,
                                                              extra)

    def _plain_calls(self) -> bool:
        """Whether calling every owner as a module (what the threaded solve does) is calling
        its own method (what the primal solve does): no law, withdrawal or storage overrides
        `forward` (its alias for `flow` or `rate`) or carries a forward hook."""
        owners = [*((el, ConstitutiveLaw) for el in self._laws),
                  *((ns, NodeWithdrawal) for ns in self._withdrawals),
                  *((st, NodeStorage) for st in self._storage)]
        return not any(type(o).forward is not base.forward or o._forward_hooks
                       or o._forward_pre_hooks for o, base in owners)

    def _threads_grad(self, drivers: Mapping, sources, phi_boundary, extra=()) -> bool:
        """`_reaches_grad` whatever the grad mode: whether any of those tensors requires
        grad."""
        values = bulk_values(drivers, list(drivers.keys()))
        values += [sources, phi_boundary, *extra]
        for owner in (*self._laws, *self._withdrawals, *self._storage):
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
        scheme=None,
        mode=None,
        storage_flow=None,
        differentiable=True,
        diagnostics: dict | None = None,
        fallback=None,
        **newton_kwargs,
    ):
        """Solve for interior potentials and branch flows.

        With `differentiable=True` (the default), gradients flow back to `phi_boundary`,
        `sources`, every value in `drivers`, and every ConstitutiveLaw's registered `nn.Parameter`s
        (i.e. constructed with `learnable=True`) via the implicit-function adjoint
        (`noodl.solvers.implicit`). Only tensors reachable one of those ways are threaded
        through `Function.apply`. Contract each ConstitutiveLaw must satisfy for
        `differentiable=True` to be safe:

        - A ConstitutiveLaw's own differentiable state must be a registered parameter
          (`learnable=True`), never a bare tensor held with `requires_grad=True` outside
          `named_parameters()` (the latter is still a supported construction, per
          `ConstitutiveLaw._param`, and its graph is reachable through this layer's component calls
          -- but it is invisible to the differentiable solve, and `differentiable=False` now
          returns detached tensors, so neither `solve` path carries a gradient to it).

        A head the potential layer reads as the driver `"<layer>.edge_head"` (see
        `noodl.heads.EdgeHeads`) is part of `drivers` only after the model's closures have run:
        a direct `solve` on raw drivers has no head, so apply the closures first (`Model.step`,
        `steady` and `simulate` do).

        Violating it raises `ValueError` naming the offending law and
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
        (non-differentiable) path -- `differentiable=False` -- and `"raise"` where it is
        differentiated (a differentiable solve that nothing differentiable reaches is its
        forward Newton alone, and returns as `"return"` does); it needs
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
        `None` for it. See `_starting_point` for what tracing it cost.

        With `differentiable=False`, the returned `(phi, q)` are ALWAYS DETACHED: the whole
        branch -- the initial guess, the grounding check, the Newton iteration, every inner
        linear solve and the final assemble/flows -- runs under `torch.no_grad()`, whatever
        grad mode the caller is in. A backend choice must not be a function of who is
        calling: a caller who wants gradients calls `differentiable=True`, which is unchanged
        (`solvers.implicit._Implicit.forward` already solved under `no_grad` and takes its
        gradients from the adjoint).

        What a solve takes (`_step` checks it, by name, on both paths): `mode=None` is a step
        when `dt` is given and the steady operating point (`"dc"`) otherwise; `"initial"` is
        the consistent initial point, where a storage that holds an `initial()` potential
        holds its node there (on a layer without flow unknowns that one solve uses the
        sparse operator with the held rows, `_held_layout`); `"dc"` and `"initial"`
        take no `dt`. A step is one implicit step of length `dt > 0` ending at time `t` with
        the `scheme` (`None` is the layer's own, `"trapezoidal"` by default), from the
        step-start state `history=(phi, q)` (full-node potentials and every edge's flow, as
        this method returns them): required by a layer with storage or a dynamic law, and `q`
        only by a dynamic law. Under the trapezoidal scheme a layer with storage also needs
        `storage_flow`, the flow into every storage entry at the step start, in
        `storage_flows`' order. A layer with no storage, no dynamic law and no law that reads
        the context takes no step and refuses `dt`, `history` and `storage_flow`; it accepts
        and ignores `t`, `q0` and a valid `scheme`. Without `dt` storage is absent.

        A layer with flow unknowns (`implicit_edges`) solves for the interior potentials and the
        flows of its implicit-law edges together (`noodl.layers._flow_unknowns`), from zeros
        or the warm start `phi0` (interior potentials) and `q0` (the flows of
        `implicit_edges`); the returned `q` covers every edge. A phasor layer is always in
        `"phasor"` mode (no `dt` or `history`; the frequency driver `"<name>.frequency"` is
        required, and `phi_boundary`, `sources`, `phi0`, `q0` may be real or complex and are
        taken as complex128). On the differentiable path gradients flow to `history` and
        `storage_flow` too, so a chain of steps back-propagates through time. There
        `diagnostics["method"]` is likewise the method requested and `"backend"` the one that
        ran; `method="cg"` is refused (the system is not symmetric positive definite).

        `fallback` is the ladder climbed when Newton does not converge (`None`, the default,
        is the layer's own `fallback`; `()` none). It runs only then, in every `on_failure`
        mode, and tries its rungs in order until one converges; the whole batch is solved
        again on each rung:

        - `"linear_guess"`: solve again from the linear initial guess (`linear_init`). It is
          skipped when the failed solve already started there (a layer without flow
          unknowns and no `phi0`); the initial point with held storage of a layer without
          flow unknowns, which has no linear guess, starts again from zeros instead.
        - `"storage_ramp"`: from that same start (or, when the linear guess cannot be
          formed, from where the failed solve started), let the network settle through a
          fictitious storage: `g (x - x_prev)` added to every row, `g` a multiple of the
          row's own Jacobian diagonal, removed over four stages, each with a storage ten
          times smaller (`noodl.solvers.continuation.storage_ramp`), and solve again from
          where it settled. A potential (balance) row takes the diagonal's size, a capacity
          at the node; the row of a flow unknown takes it with its sign, an inertia of the
          flow (a relation may be written either way round, and a shift of the other sign
          would cancel its diagonal). A row with a zero diagonal (an ideal source; a node
          whose every edge is a flow unknown) gets none. The stages are a starting point and
          carry no gradient; the solve that follows is this one's own, so the gradient is at
          the exact root.

        `diagnostics`, when given, then also receives `"fallback"`, the rung that converged
        (`None` if the first solve did, or no rung did), and `"fallback_iterations"`, the
        storage stages' Newton iterations (0 for `"linear_guess"`); the other entries are
        the converged solve's. When no rung converges, the first failure stands: its error
        is raised (naming the rungs tried and the last one's error), or, under
        `on_failure="return"`/`"return_primal"`, its result and diagnostics are returned.
        Only non-convergence (`ConvergenceError`) of the first solve climbs the ladder; every
        other error of it passes through. A rung that raises any error has failed, and the
        next is tried. On a layer with flow unknowns the rank test that names a singular
        system diagnoses the first solve's failure only once every rung has failed too (it
        is raised then, in place of the first error), in every `on_failure` mode. With no
        fallback (the default) none of this runs, the rank test is made at once, and
        `diagnostics` gains no entry.
        """
        try:
            rungs = self.fallback if fallback is None else check_fallback(self.name, fallback)
            if not rungs:
                return self._solve(phi_boundary, drivers, sources, phi0, q0=q0,
                                   history=history, dt=dt, t=t, scheme=scheme, mode=mode,
                                   storage_flow=storage_flow, differentiable=differentiable,
                                   diagnostics=diagnostics, **newton_kwargs)
            return self._solve_with_fallback(
                rungs, phi_boundary, drivers, sources, phi0, q0=q0, history=history, dt=dt,
                t=t, scheme=scheme, mode=mode, storage_flow=storage_flow,
                differentiable=differentiable, diagnostics=diagnostics, **newton_kwargs)
        finally:
            # The restrictions of this solve's context (`_law_ctx`) hold its step-start
            # tensors, and their graph in a differentiable chain: none outlives the solve
            # (a later solve builds a new context, so nothing is reused across solves).
            self._law_ctxs = (None, {})

    def _solve_with_fallback(self, rungs, phi_boundary, drivers, sources, phi0, *, q0,
                             diagnostics, **kwargs):
        """`_solve`, then, if Newton did not converge, the rungs of `rungs` in order until
        one does (see `solve`'s `fallback`)."""
        report: dict = {}
        first_error = None
        try:
            out = self._solve(phi_boundary, drivers, sources, phi0, q0=q0,
                              diagnostics=diagnostics, _report=report, **kwargs)
        except ConvergenceError as exc:
            first_error = exc
        else:
            if diagnostics is None or _converged(diagnostics):
                if diagnostics is not None:
                    diagnostics.update(fallback=None, fallback_iterations=0)
                return out
            first = (out, dict(diagnostics))
        # The rungs start from the linear guess, or from zeros at the held initial point of
        # a layer without flow unknowns (which has none).
        rung_start = "zeros" if report.get("held") else "linear"
        last_error = None
        for rung in rungs:
            if rung == "linear_guess" and report.get("start") == rung_start:
                continue
            ramp: dict = {}
            ramped = rung == "storage_ramp"
            try:
                # the storage ramp falls back to the failed solve's own start when the
                # linear guess cannot be formed (`_solve`)
                out = self._solve(phi_boundary, drivers, sources, phi0 if ramped else None,
                                  q0=q0 if ramped else None, diagnostics=diagnostics,
                                  _start="ramp" if ramped else "linear",
                                  _report=ramp, **kwargs)
            except RuntimeError as exc:
                # any error of a rung (a singular linear guess, a grounding refusal at it,
                # non-convergence) is that rung failing; the first failure still stands
                last_error = exc
                continue
            if diagnostics is None or _converged(diagnostics):
                if diagnostics is not None:
                    diagnostics.update(fallback=rung,
                                       fallback_iterations=ramp.get("ramp_iterations", 0))
                return out
        if first_error is not None:
            if "singular" in report:
                # the rank test of the first solve, diagnosed once every rung has failed
                raise report["singular"] from first_error
            if last_error is None:
                raise first_error
            raise ConvergenceError(
                f"{first_error}; the fallback {list(rungs)} did not converge either: "
                f"{last_error}"
            ) from last_error
        out, diag = first
        diagnostics.clear()
        diagnostics.update(diag, fallback=None, fallback_iterations=0)
        return out

    def _solve(self, phi_boundary, drivers, sources, phi0, *, q0, history, dt, t, scheme,
               mode, storage_flow, differentiable, diagnostics, _start=None, _report=None,
               **newton_kwargs):
        """The body of `solve`, which see. `_start` and `_report` are the fallback ladder's
        (`_solve_with_fallback`): `_start="linear"` starts from the linear initial guess
        (zeros at the held initial point of a layer without flow unknowns), `"ramp"` from
        where `storage_ramp` settles from there; `_report` receives where the solve
        started (`"start"`: `"linear"`, `"zeros"` or `"warm"`), whether it is that held
        initial point (`"held"`) and the storage stages' Newton iterations
        (`"ramp_iterations"`)."""
        self._require_current_topology()
        if scheme is None:
            scheme = self.scheme
        self._check_forward_euler(scheme)
        for key, value in self.newton_defaults.items():
            newton_kwargs.setdefault(key, value)
        drivers = drivers or {}
        # The order of the refusals is part of the contract: (phasor) inputs, the step's
        # keywords, the Newton options a layer with flow unknowns refuses, the policy, the
        # starting point (with its grounding check), the unreachable-tensor check, then the
        # solve.
        u = self._unknowns
        if u is not None:
            phi_boundary, sources, phi0, q0 = self._complex_inputs(
                phi_boundary, drivers, sources, phi0, q0)
        step = self._step(mode, dt, t, scheme, history, storage_flow, drivers)
        if step.layout is not None:
            # the initial point with held storage of a layer without flow unknowns
            u = step.layout
            self._refuse_newton_options(newton_kwargs, "the initial point with held storage")
        elif u is not None:
            self._refuse_newton_options(newton_kwargs)
        starts = step_start(step, history)
        policy = resolve(
            self, newton_kwargs, diagnostics, differentiable, drivers, sources, phi_boundary,
            starts,
        )
        if _start is None:
            x0, start = self._starting_point(phi0, phi_boundary, drivers, sources, step,
                                             q0=q0)
        else:
            try:
                x0, start = self._starting_point(None, phi_boundary, drivers, sources, step,
                                                 linear=True)
            except RuntimeError:
                if _start != "ramp":
                    raise
                # no linear guess to settle from (singular, or not grounded there): the
                # storage ramp starts where the failed solve did
                x0, start = self._starting_point(phi0, phi_boundary, drivers, sources, step,
                                                 q0=q0)
        if _report is not None:
            _report["held"] = step.layout is not None
            _report["start"] = ("warm" if phi0 is not None or q0 is not None
                                else "zeros" if u is not None and _start is None
                                else "linear")
        # A tensor held where the solve does not thread it is refused before solving, by
        # name; one reached only through a closure, at every residual the forward Newton
        # evaluates, the starting guess first and the root last (`_newton_differentiable`).
        # A differentiable solve that nothing reaches is the primal one: with flow unknowns
        # as it is; on the fast path with that closure check at every residual
        # (`primal_checked`) where nothing it threads requires grad (under `no_grad` too:
        # the threaded residual detaches what does), the threaded slopes are the primal ones
        # (`layer_slopes`), no law is `dp_independent` (a declaration the threaded operator
        # cross-checks) and every owner's module call is its own method (`_plain_calls`).
        # Its result is the threaded forward Newton's, bit for bit, without the threading's
        # cost.
        fn = None
        primal_checked = False
        if policy.differentiable:
            self._check_no_unreachable_differentiable_tensors()
            primal_checked = (
                policy.forward_only and u is None and layer_slopes(self)
                and not any(el.dp_independent for el in self._laws) and self._plain_calls()
                and (torch.is_grad_enabled()
                     or not self._threads_grad(drivers, sources, phi_boundary, starts)))
            if not (primal_checked or (policy.forward_only and u is not None)):
                fn = FunctionalLaws(self, drivers, sources, phi_boundary, step, history)

        requested = newton_kwargs["method"]
        on_fail = last = None
        if u is not None:
            newton_kwargs["method"] = linear_method(self, requested)
            if self.linear:
                # The residual is affine: the full Newton step lands on the solution, so the
                # damped first step (`newton`'s omega=0.75) would only cost a second
                # iteration.
                newton_kwargs.setdefault("omega", 1.0)
            # The last Jacobian and the first and last residuals, kept to diagnose a failure
            # by a rank test, which may raise a clearer error than Newton's.
            last = {}

            def remember(r):
                last.setdefault("r0", r.detach())
                last["r"] = r.detach()

            def on_fail(exc):
                if singular(last.get("op"), u.n, first_failing(last, newton_kwargs)):
                    error = self._singular_error(step.ctx.mode)
                    if _report is not None:
                        # under the fallback ladder the rank test is a diagnosis made only
                        # once every rung has failed too (`_solve_with_fallback`)
                        _report["singular"] = error
                        return
                    raise error from exc

        if _start == "ramp":
            # the storage ramp's starting point; the solve below starts from it
            x0 = self._ramp(x0, phi_boundary, drivers, sources, step, newton_kwargs, _report)
            start = None

        if fn is None:
            # The primal solve (with flow unknowns also a differentiable one that nothing
            # reaches). The WHOLE of it runs under no_grad -- the Newton iteration, every
            # inner linear solve inside it, and the final assemble/flows -- whatever grad
            # mode the caller is in: there is no legitimate graph to build (any graph would
            # be an unrolled trace through the converged iterate, not the gradient
            # `differentiable=True` computes), and `solvers.select.solve`'s "auto" refuses
            # sparse-direct when grad mode is on and an input requires grad, which made the
            # backend a function of the caller's context. The returned `(phi, q)` are
            # detached; `diagnostics` is unaffected.
            plain = self._owner_call
            slopes = None if u is None else self._slopes_fn(drivers, plain, step)
            # The last residual's point, potentials and flows: Newton returns the iterate it
            # last evaluated the residual at, whose flows are then the result's.
            point: dict = {}

            def evaluate(x):
                r, phi, q = self._residual(x, phi_boundary, drivers, sources, step, plain)
                point.update(x=x, phi=phi, q=q)
                if last is not None:
                    remember(r)
                return r

            if primal_checked:
                # nothing it threads requires grad, so the layer's own residual under grad
                # mode is `_threaded_residual`'s check
                def residual_fn(x):
                    return self._threaded_residual(evaluate, (x,), detach=False)
            else:
                residual_fn = evaluate

            def operator_fn(x):
                # At the fast path's starting point the grounding check's slopes are the
                # first operator's.
                if start is not None and x is x0:
                    return self._operator_at(*start)
                op = self._operator(x, phi_boundary, drivers, step, slopes)
                if last is not None:
                    last["op"] = op
                return op

            with torch.no_grad():
                result = self._newton_primal(residual_fn, operator_fn, x0, newton_kwargs,
                                             diagnostics, on_fail, requested=requested)
                if primal_checked and point.get("x") is not result.x:
                    residual_fn(result.x)   # the root is checked too
                if point.get("x") is result.x:
                    return point["phi"], point["q"]
                return self._output(result.x, phi_boundary, drivers, step)

        if u is None:
            # the threaded residual and operator of the fast path
            residual_fn = fn.residual_fn
            reuse_start = fn.layer_slopes and start is not None

            def operator_fn(x, *params):
                # At the starting point the grounding check's slopes are the first
                # operator's, as on the primal path, where they are the same slopes
                # (`layer_slopes`).
                if x is x0 and reuse_start and not torch.is_grad_enabled():
                    return self._operator_at(*start)
                return fn.operator_fn(x, *params)
        else:
            def residual_fn(x, *params):
                rebuilt, drv, src, pb, st = fn.rebuild(params)
                r, phi, q = self._residual(x, pb, drv, src, st,
                                           functools.partial(fn.call, rebuilt))
                if not q.requires_grad:
                    fn.last = (x, phi, q)
                remember(r)
                return r

            def operator_fn(x, *params):
                rebuilt, drv, src, pb, st = fn.rebuild(params)
                call = functools.partial(fn.call, rebuilt)
                # the one `adjoint` flag: the operator also builds the adjoint, so every
                # law, withdrawal and storage that declares `exact_slope = False` gives
                # autograd
                last["op"] = self._operator(x, pb, drv, st, self._slopes_fn(drv, call, st),
                                            adjoint=True)
                return last["op"]

        x = self._newton_differentiable(fn, residual_fn, operator_fn, x0, policy,
                                        newton_kwargs, diagnostics, on_fail,
                                        requested=requested)
        parts = fn.root_parts(x)
        if parts is not None:
            return parts
        return self._output(x, phi_boundary, drivers, step)

    def _check_forward_euler(self, scheme) -> None:
        """Refuse `"forward_euler"` on a layer that steps anything inside its solve (storage
        at interior nodes, dynamic laws), and any other scheme on a layer whose storage sits
        at boundary nodes, by name."""
        if scheme == "forward_euler" and self.dynamic:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: scheme='forward_euler' steps storage at "
                f"boundary nodes only; this layer steps storage at interior nodes or dynamic "
                f"laws inside its solve, which take 'backward_euler' or 'trapezoidal'"
            )
        if self._boundary_storage and scheme != "forward_euler":
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: its storage sits at boundary nodes, which "
                f"step by scheme='forward_euler' only; got scheme={scheme!r}"
            )

    def _step(self, mode, dt, t, scheme, history, storage_flow, drivers, *,
              check_dt: bool = True, omega: bool = True) -> _Step:
        """The `_Step` of one solve (and of `residual`, `jacobian`, `adjoint` and
        `storage_flows`): the context its laws are evaluated in and what its storage does.

        `mode=None` is a step when `dt` is given and the steady operating point (`"dc"`)
        otherwise; `"dc"` and `"initial"` (the consistent initial point) take no `dt`, and
        `"phasor"`, on a phasor layer only (which is always in it), no `dt` or `history`. A
        step needs `dt > 0` and, from the step start, `history=(phi, q)`: required on a layer
        with a storage or a dynamic law, with `q` required only by a dynamic law and `None`
        otherwise. A layer with storage under the trapezoidal scheme needs `storage_flow`,
        the flow into every storage entry at the step start; `history=(phi, None)` is the
        step of a layer whose storage is all that steps. A layer with no storage, no dynamic
        law and no law that reads the context takes no step: it refuses `dt` and `history`.
        In practice that is a fast-path layer only: a layer with an ideal or relation law
        reads the context, so it steps. A scheme outside `SCHEMES` is refused on every
        layer but a phasor one, which ignores it. Every keyword is checked by name;
        `check_dt=False` skips the positivity of `dt` (a host sync for a tensor `dt`) when
        the solve of the same step has checked it, and `omega=False` leaves a phasor layer's
        angular frequency out (`storage_flows` reads no law)."""
        name = self.name
        if self.phasor:
            if mode not in (None, "phasor"):
                raise ValueError(
                    f"KirchhoffLayer {name!r} is a phasor layer: mode={mode!r}; expected "
                    f"'phasor' or None"
                )
            if dt is not None or history is not None:
                raise ValueError(
                    f"KirchhoffLayer {name!r} is a phasor layer, which solves the "
                    f"sinusoidal steady state and takes no step; drop dt and history"
                )
        else:
            # a phasor layer ignores the scheme
            check_scheme(name, scheme)
            if mode == "phasor":
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode='phasor' needs a layer built with "
                    f"phasor=True"
                )
            if mode is not None and mode not in ("dc", "initial"):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode={mode!r}; expected 'dc', 'initial' or "
                    f"None (transient when dt is given, dc otherwise)"
                )
            if mode is not None and dt is not None:
                what = ("the consistent initial point" if mode == "initial"
                        else "the steady operating point")
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode={mode!r} is {what}, which takes no step; "
                    f"drop dt"
                )
        if storage_flow is not None and not self._storage:
            raise ValueError(f"KirchhoffLayer {name!r} has no storage; drop storage_flow")
        if not self._takes_step:
            stepping = [k for k, v in (("dt", dt), ("history", history)) if v is not None]
            if stepping:
                raise ValueError(
                    f"KirchhoffLayer {name!r} has no storage, no dynamic law and no law that "
                    f"reads the step's context, so it takes no step; drop {stepping}"
                )
        if dt is None:
            extra = [k for k, v in (("history", history), ("storage_flow", storage_flow))
                     if v is not None]
            if extra:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: {extra} given without dt; a steady solve "
                    f"takes no step"
                )
            if self.phasor:
                ctx = LawContext(mode="phasor", t=t,
                                 omega=angular_frequency(self, drivers) if omega else None)
                return _Step(ctx, None)
            ctx = LawContext(mode="dc" if mode is None else mode, t=t)
            found = (self._held_values(drivers, self._owner_call, check=True)
                     if mode == "initial" and self._storage else None)
            if found is None:
                return _Step(ctx, None)
            if self._unknowns is not None:
                return _Step(ctx, self._held(found, self._unknowns))
            # a layer without flow unknowns: this one solve holds the rows on the sparse
            # operator, in a layout with no unknown flows
            u = self._held_layout()
            return _Step(ctx, self._held(found, u), u)
        if not check_dt:
            # the solve of this step has checked it
            ...
        elif isinstance(dt, (int, float)):
            if not dt > 0:
                raise ValueError(f"KirchhoffLayer {name!r}: dt must be positive, got {dt!r}")
        elif not bool((torch.as_tensor(dt) > 0).all()):
            raise ValueError(f"KirchhoffLayer {name!r}: dt must be positive, got {dt!r}")
        if len(self._dynamic_edges):
            self._check_head_off_dynamic_edges(drivers)
        ctx = LawContext(mode="transient", dt=dt, t=t, scheme=scheme)
        if history is None:
            if len(self._dynamic_edges):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a transient step needs history=(phi, q) "
                    f"at the step start"
                )
            if self._storage:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a step of a layer with storage needs "
                    f"history=(phi, q) at the step start"
                )
            return _Step(ctx, None)
        if not isinstance(history, (tuple, list)) or len(history) != 2:
            raise ValueError(
                f"KirchhoffLayer {name!r}: history must be the pair (phi, q) at the step "
                f"start; got {type(history).__name__}"
            )
        phi_prev, q_prev = history
        n_edges = len(self.cols)
        if q_prev is None:
            if len(self._dynamic_edges):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a layer with a dynamic law needs "
                    f"history=(phi, q) with q, every edge's flow at the step start"
                )
            if phi_prev.shape[-1] != self._n_nodes:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: history's phi must hold every node's "
                    f"potential ({self._n_nodes}); got last dimension {phi_prev.shape[-1]}"
                )
        elif phi_prev.shape[-1] != self._n_nodes or q_prev.shape[-1] != n_edges:
            raise ValueError(
                f"KirchhoffLayer {name!r}: history=(phi, q) must hold every node's "
                f"potential ({self._n_nodes}) and every edge's flow ({n_edges}); got last "
                f"dimensions {phi_prev.shape[-1]} and {q_prev.shape[-1]}"
            )
        if self._reads_history:
            ctx = self._with_history(ctx, phi_prev, q_prev)
        if not self._storage:
            return _Step(ctx, None)
        if storage_flow is None:
            if scheme == "trapezoidal":
                raise ValueError(
                    f"KirchhoffLayer {name!r}: the trapezoidal scheme needs storage_flow, "
                    f"the flow into every storage entry at the step start "
                    f"(storage_flows; the state key {name + '.storage_flow'!r})"
                )
        elif storage_flow.shape[-1] != self._n_entries:
            raise ValueError(
                f"KirchhoffLayer {name!r}: storage_flow must hold one flow per storage "
                f"entry ({self._n_entries}); got last dimension {storage_flow.shape[-1]}"
            )
        return _Step(ctx, _StorageStep(phi_prev, storage_flow))

    def _with_history(self, ctx: LawContext, phi_prev, q_prev) -> LawContext:
        """`ctx` with the step-start potential differences and flows taken from
        `(phi_prev, q_prev)`."""
        return dataclasses.replace(ctx, dp_prev=self._difference(phi_prev), q_prev=q_prev)

    def _check_head_off_dynamic_edges(self, drivers) -> None:
        """Refuse a head driver that is not zero on an edge with a dynamic law: the step-start
        potential difference of such an edge is taken without a head.

        `EdgeHeads` guarantees this by construction (it refuses a term on a dynamic kind), so
        the value check is only for a head driver given directly, and is made on a CPU tensor
        only: reading values off an accelerator is a host sync per solve, which this path does
        not pay. The width is always checked (no sync). Free when there is no head or no
        dynamic law.
        """
        head = drivers.get(self.head_key)
        if head is None:
            return
        self._check_head_width(head)
        if head.device.type != "cpu":
            return
        if bool((head[..., self._dynamic_edges] != 0).any()):
            kinds = sorted({el.kind for el in self._laws if el.dynamic})
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: driver {self.head_key!r} is not "
                f"zero on an edge of kind {kinds}, whose law is dynamic (its step-start "
                f"potential difference is taken without a head)"
            )

    def storage_flows(self, phi, drivers, *, dt, scheme=None, history=None,
                      storage_flow=None, mode=None, q=None, sources=None,
                      _checked: bool = False) -> torch.Tensor:
        """The flow into every storage entry at the end of a step, `(..., n_entries)`, in
        entry order (storages in list order, then each one's nodes): what each
        `NodeStorage.rate` withdraws at the full-node potentials `phi`. The keywords are
        `solve`'s; without `dt` (a steady state) storage is absent and the flows are zeros.
        Differentiable in `phi`, `history`, `storage_flow`, the drivers and the storages'
        parameters. Empty on a layer without storage. (`_checked`, internal: the `solve` of
        this step has already checked `dt`.)

        With `mode="initial"`, a node held at
        its storage's initial potential takes whatever flow balances it: the flow into
        storage there is the node's flow imbalance `s_I - A_I q - w(phi)` (its sources, less
        the net outflow of the edge flows `q` and the withdrawals' withdrawal), split among
        the node's storage entries in proportion to their `capacity` (equally where the
        node's capacities sum to zero). It needs `q` (every edge's flow, as `solve` returns
        it) and `sources` (full-node; `None` means zero). An entry at a node that is not held
        takes zero.

        Under `scheme="forward_euler"` (storage at boundary nodes) it is the flow the edge
        flows `q` deliver to every storage node, whether or not the call is a step: the
        storage is held through the solve and takes whatever reaches it. It needs `q`, and
        reads nothing else."""
        if self._boundary_storage:
            if q is None:
                raise ValueError(
                    f"KirchhoffLayer {self.name!r}: storage_flows of storage at boundary "
                    f"nodes needs q, every edge's flow (as solve returns it)"
                )
            inflow = -self._accumulate_bound(q)
            return torch.cat([inflow[..., cols] for cols in self._boundary_storage_cols],
                             dim=-1)
        drivers = drivers or {}
        if scheme is None:
            scheme = self.scheme
        zeros = torch.zeros(phi.shape[:-1] + (self._n_entries,), dtype=phi.dtype,
                            device=phi.device)
        if not self._storage:
            return zeros
        if mode == "initial":
            return self._initial_storage_flows(phi, q, sources, drivers, zeros)
        step = self._step(mode, dt, None, scheme, history, storage_flow, drivers,
                          check_dt=not _checked, omega=False)
        if not isinstance(step.storage, _StorageStep):
            return zeros
        parts = [st.rate(*self._storage_args(j, st, phi, step.storage), step.ctx, drivers)
                 for j, st in enumerate(self._storage)]
        batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
        return torch.cat([p.expand(batch + p.shape[-1:]) for p in parts], dim=-1)

    def _held_nodes(self) -> torch.Tensor:
        """The full-node positions of every boundary storage node, in entry order."""
        return torch.cat([st.nodes for st in self._boundary_storage])

    def hold_potentials(self, phi, phi_start) -> torch.Tensor:
        """`phi` (full-node) with every boundary storage node (`scheme="forward_euler"`) at
        its potential in `phi_start`: what a closure reading the state sees of such a node
        during a step, the value the solve holds it at. Unchanged on a layer without
        boundary storage."""
        if not self._boundary_storage:
            return phi
        idx = self._held_nodes().to(phi.device)
        held = phi_start[..., idx]
        batch = broadcast_shapes(phi.shape[:-1], held.shape[:-1])
        out = phi.expand(batch + phi.shape[-1:]).clone()
        out[..., idx] = held.to(out.dtype)
        return out

    def hold_storage(self, phi_boundary, phi_start) -> torch.Tensor:
        """`phi_boundary` with the column of every storage node at a boundary node
        (`scheme="forward_euler"`) replaced by its potential in `phi_start` (full-node, the
        step start): the value the solve holds it at, whatever the driver gave there.
        Unchanged on a layer without boundary storage."""
        if not self._boundary_storage:
            return phi_boundary
        cols = torch.cat(self._boundary_storage_cols).to(phi_boundary.device)
        held = phi_start[..., self.bound[cols].to(phi_start.device)]
        batch = broadcast_shapes(phi_boundary.shape[:-1], held.shape[:-1])
        out = phi_boundary.expand(batch + phi_boundary.shape[-1:]).clone()
        out[..., cols] = held.to(out.dtype)
        return out

    def advance_storage(self, phi, storage_flow, dt, drivers=None) -> torch.Tensor:
        """The potentials `phi` (full-node) after one forward-Euler step of `dt` of the
        storage at boundary nodes (`scheme="forward_euler"`): each storage node moves by
        `dt * storage_flow / C`, with `storage_flow` the flow into every entry over the step
        (`storage_flows`, `"<name>.storage_flow"`) and `C` the storage's capacity. Every
        other potential is unchanged. Differentiable in `phi`, `storage_flow`, `dt` and the
        capacities."""
        if not self._boundary_storage:
            raise ValueError(
                f"KirchhoffLayer {self.name!r} has no storage at boundary nodes "
                f"(scheme='forward_euler'); advance_storage steps nothing"
            )
        if isinstance(dt, (int, float)):
            if not dt > 0:
                raise ValueError(
                    f"KirchhoffLayer {self.name!r}: dt must be positive, got {dt!r}")
        elif not bool((torch.as_tensor(dt) > 0).all()):
            raise ValueError(f"KirchhoffLayer {self.name!r}: dt must be positive, got {dt!r}")
        if storage_flow.shape[-1] != self._n_entries:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: storage_flow must hold one flow per storage "
                f"entry ({self._n_entries}); got last dimension {storage_flow.shape[-1]}"
            )
        nodes, moves, offset = [], [], 0
        for st in self._boundary_storage:
            n = st.nodes.numel()
            start = phi[..., st.nodes]
            moves.append(dt * storage_flow[..., offset:offset + n]
                         / st.capacity(start, start, drivers))
            nodes.append(st.nodes)
            offset += n
        move = torch.cat(moves, dim=-1)
        idx = torch.cat(nodes)
        batch = broadcast_shapes(phi.shape[:-1], move.shape[:-1])
        out = phi.expand(batch + phi.shape[-1:]).clone()
        out[..., idx] = out[..., idx] + move
        return out

    def _initial_storage_flows(self, phi, q, sources, drivers, zeros) -> torch.Tensor:
        """`storage_flows` at the initial point (`mode="initial"`): each held node's flow
        imbalance, split among its entries by capacity."""
        held = self._held_values(drivers, self._owner_call)
        if held is None:
            return zeros
        if q is None:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: storage_flows(mode='initial') needs q, every "
                f"edge's flow at the initial point (as solve returns it)"
            )
        rows, _ = held
        phi_i = phi[..., self.interior]
        imbalance = self._source_interior(sources, phi_i) - self._accumulate_interior(q)
        w = self._withdrawal(phi, drivers)
        if w is not None:
            imbalance = imbalance - w
        imbalance = torch.where(rows, imbalance, torch.zeros_like(imbalance))
        caps = [st.capacity(phi[..., st.nodes], phi[..., st.nodes], drivers)
                for st in self._storage]
        total = self._scatter_storage(lambda j, st: caps[j])
        count = self._scatter_storage(lambda j, st: torch.ones_like(caps[j]))
        parts = []
        for cap, pos in zip(caps, self._storage_positions, strict=True):
            tot = total[..., pos]
            share = torch.where(tot != 0, cap / torch.where(tot != 0, tot, 1.0),
                                1.0 / count[..., pos])
            parts.append(share * imbalance[..., pos])
        batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
        return torch.cat([p.expand(batch + p.shape[-1:]) for p in parts], dim=-1)

    def _starting_point(self, phi0, phi_boundary, drivers, sources, step=None, *, q0=None,
                        linear: bool = False):
        """The Newton starting guess `x0` and, on the fast path, the slopes there `start`.

        The fast path starts from `linear_init` (or `phi0`), certified grounded; `start` is
        `(dq0, node_slopes0)`, the slopes at that point, which the first operator reuses.
        A layer with flow unknowns starts from zeros, or the warm start `phi0` (interior
        potentials) and `q0` (the flow unknowns), in the real view; `start` is `None`.
        `step` is the solve's `_Step`. (Its linear guess, `linear_init`, is not used here:
        starting the general path from it moved the pinned roots by up to Newton's
        tolerance, so a cold solve keeps the zero start.) `linear=True` (the fallback
        ladder's) starts a layer with flow unknowns from it instead.
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
        u = self._unknowns if step is None or step.layout is None else step.layout
        if u is not None:
            real_dtype = (phi_boundary.real.dtype if phi_boundary.is_complex()
                          else phi_boundary.dtype)
            with torch.no_grad():
                probe = self._residual(
                    torch.zeros(u.n, dtype=real_dtype, device=phi_boundary.device),
                    phi_boundary, drivers, sources, step, self._owner_call)[0]
                shapes = [probe.shape[:-1]]
                if phi0 is not None:
                    shapes.append(phi0.shape[:-1])
                if q0 is not None:
                    shapes.append(q0.shape[:-1])
                B = broadcast_shapes(*shapes)
                x0 = torch.zeros(B + (u.n,), dtype=real_dtype, device=phi_boundary.device)
                if linear and self._unknowns is not None:
                    # outside no_grad: the guess may take a law's slope by autograd
                    with torch.enable_grad():
                        guess = self._linear_guess(phi_boundary, drivers, sources, step)
                    x0 = x0 + guess
                n_phi = u.w * u.n_I
                if phi0 is not None:
                    x0[..., :n_phi] = to_real(self, phi0.detach())
                if q0 is not None:
                    x0[..., n_phi:] = to_real(self, q0.detach())
                if self._unknowns is None:
                    # the held initial point of a layer without flow unknowns: certified
                    # grounded like every other solve of it, a held row counting as a ground
                    self._held_grounding_check(x0, phi_boundary, drivers, step)
            return x0, None
        with torch.no_grad():
            if phi0 is None:
                phi0 = self.linear_init(phi_boundary, drivers, sources, _step=step)
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
            dq0 = self._dflows(phi0_full, drivers, _DC if step is None else step.ctx)
            node_slopes0 = self._withdrawal_slopes(phi0_full, drivers, step)
            self._grounding_check(dq0, where="solve", node_slopes=node_slopes0)
        return phi0, (dq0, node_slopes0)

    def _ramp(self, x0, phi_boundary, drivers, sources, step, newton_kwargs, report):
        """`storage_ramp` from `x0` on this solve's own residual and Jacobian (the laws
        called on their modules, without gradients): the starting point of the fallback
        ladder's `"storage_ramp"` rung. The rows of the flow unknowns take a storage with
        the sign of their own diagonal (an inertia), the potential rows its size."""
        call = self._owner_call
        u = self._unknowns if step.layout is None else step.layout

        def residual(x):
            return self._residual(x, phi_boundary, drivers, sources, step, call)[0]

        def operator(x):
            return self._operator(x, phi_boundary, drivers, step)

        signed = None
        if u is not None:
            signed = torch.arange(u.n, device=x0.device) >= u.w * u.n_I
        stages: dict = {}
        x = storage_ramp(residual, operator, x0, signed=signed, diagnostics=stages,
                         **{k: v for k, v in newton_kwargs.items() if k != "on_failure"})
        if report is not None:
            report["ramp_iterations"] = stages["newton_iterations"]
        return x

    def _held_grounding_check(self, x0, phi_boundary, drivers, step) -> None:
        """The grounding certificate of the held initial point of a layer without flow
        unknowns, at its starting point `x0`: the law slopes there, with every held row
        grounded (a positive slope on its diagonal), so a node that is neither held nor
        joined to a boundary or a held node is refused by name."""
        phi = self.assemble(x0, phi_boundary)
        dq = self._dflows(phi, drivers, step.ctx)
        rows = step.storage.rows
        ns = self._withdrawal_slopes(phi, drivers)
        held = rows.to(dtype=dq.dtype)
        ns = held if ns is None else torch.where(rows, held, ns)
        self._grounding_check(dq, where="solve", node_slopes=ns)

    def _output(self, x, phi_boundary, drivers, step):
        """`(phi, q)` at the solution `x`: full-node potentials and every edge's flow. The
        laws are called on the layer's own modules, so on the differentiable path `q` keeps
        the parameters' graph."""
        u = self._unknowns if step.layout is None else step.layout
        if u is None:
            phi = self.assemble(x, phi_boundary)
            return phi, self._flows(phi, drivers, step.ctx)
        phi_I, q_F = split(u, x)
        phi = self.assemble(phi_I, phi_boundary)
        q, _ = self._flows_and_residuals(phi, q_F, drivers, step.ctx, self._owner_call, u)
        return phi, q

    def _newton_primal(self, residual_fn, operator_fn, x0, newton_kwargs, diagnostics,
                       on_fail=None, *, requested=None):
        """Newton alone, under `no_grad`, on either path; returns its `NewtonResult`.

        `on_fail(exc)` is called with a Newton error before it propagates (the rank test of
        a layer with flow unknowns, which may raise a clearer error). `diagnostics` gets the
        result, with `"method"` the inner solver `requested` (`newton_kwargs["method"]`
        unless given): under `"auto"` the batch threshold and SciPy's presence decide which
        backend ran, and nothing else reports it."""
        if requested is None:
            requested = newton_kwargs["method"]
        with torch.no_grad():
            try:
                result = newton(residual_fn, operator_fn, x0, **newton_kwargs)
            except RuntimeError as exc:
                if on_fail is not None:
                    on_fail(exc)
                raise
        if diagnostics is not None:
            result.into(diagnostics, method=requested)
        return result

    def _newton_differentiable(self, fn, residual_fn, operator_fn, x0, policy, newton_kwargs,
                               diagnostics, on_fail=None, *, requested=None):
        """The differentiable solve of `residual_fn(x, *fn.all_params) = 0`, on either path;
        returns the root, carrying the implicit-function adjoint.

        The forward Newton evaluates each residual with the threaded tensors detached and
        refuses one that still requires grad (`_threaded_residual`: `_require_threaded`'s
        check at no extra evaluation); the adjoint's VJP, under grad mode, takes the
        residual's graph to `params`. `policy.forward_only` (on the fast path, where `_solve`
        does not take the primal Newton instead) runs the forward Newton alone: nothing
        reaches a gradient, so no adjoint is ever asked of a point it may return
        unconverged. The root is checked again only when Newton's last
        residual (`fn.last`) was not at it, which leaves `fn.last` at the root either way.
        `on_fail` and `requested` are `_newton_primal`'s."""
        if requested is None:
            requested = newton_kwargs["method"]
        all_params = fn.all_params
        detached: dict = {}

        def residual(x, *params):
            if torch.is_grad_enabled():
                return residual_fn(x, *params)
            if not _same_objects(detached.get("of"), params):   # detached once per solve
                detached.update(of=params, to=_detached(params))
            return self._threaded_residual(lambda *p: residual_fn(x, *p), detached["to"],
                                           detach=False)

        try:
            if policy.forward_only:
                # `implicit_solve`'s own forward (the same Newton on the same Jacobian)
                x = self._newton_primal(
                    lambda x: residual(x, *all_params),
                    lambda x: operator_fn(x, *all_params),
                    x0, newton_kwargs, diagnostics, requested=requested,
                ).x
            else:
                x = implicit_solve(residual, operator_fn, x0, all_params,
                                   diagnostics=diagnostics, **newton_kwargs)
                if diagnostics is not None:
                    # `implicit_solve` writes the outcome; the method requested goes here
                    diagnostics["method"] = requested
        except RuntimeError as exc:
            if on_fail is not None:
                on_fail(exc)
            raise
        # The adjoint linearises at the root, so the threading check must hold there: a law
        # may read an un-threaded tensor only on a branch the starting guess did not reach.
        # Newton's last residual is normally at the root it returns, and was checked.
        root = x.detach()
        last = fn.last
        if not (last is not None and last[0].shape == root.shape
                and torch.equal(last[0], root)):
            self._require_threaded(lambda *params: residual_fn(root, *params), all_params)
        return x

    def adjoint(self, phi_interior, phi_boundary, drivers, grad_phi_interior, *, q=None,
                dt=None, scheme=None, history=None, storage_flow=None):
        """Solve J^T lambda = grad_phi_interior at the given point, `J` the Jacobian of
        `residual` (`jacobian`).

        On a layer with flow unknowns `J` is over the unknowns `[phi_I, q_F]`: the gradient
        has their width, the interior potentials then the flow unknowns (`q` is every edge's
        flow, as `solve` returns it, and is required), and so does the result; on a phasor
        layer both are in the real view. The system is the sparse operator `solve`'s Newton
        iteration builds, solved with the layer's inner solver.

        The step keywords are `residual`'s: with `dt`, a layer's storage slope is in `J`;
        without, `J` is the steady Jacobian.

        Without flow unknowns the operator is the SAME GraphLaplacianOperator `solve`'s Newton
        iteration builds (same endpoints, same `dflows` slopes), never the dense einsum
        `jacobian()` -- so the adjoint costs one matvec-free transposed solve rather than an
        (n_I, n_I) materialisation, and cannot drift from the forward path's own operator. The
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

        On a layer with flow unknowns the operator is built without gradients, so the result
        is differentiable at most in `grad_phi_interior`; under grad mode with a grad-requiring
        `grad_phi_interior`, `linear_solver="auto"` (sparse direct there) refuses: wrap the
        call in `torch.no_grad()`, or use `linear_solver="direct"`. A gradient of the wrong
        width, or a complex one, is refused by name.
        """
        self._require_current_topology()
        drivers = drivers or {}
        step = self._public_step(dt, scheme, history, storage_flow, drivers)
        if self._unknowns is not None:
            phi_boundary, _, phi_interior, q = self._complex_inputs(
                phi_boundary, drivers, None, phi_interior, q)
            x = self._unknown_vector(phi_interior, q, "adjoint")
            self._check_adjoint_grad(grad_phi_interior)
            # the operator's values carry no graph, like every Jacobian entry of a solve
            with torch.no_grad():
                op = self._operator(x, phi_boundary, drivers, step)
            return _adjoint_solve(
                op,
                grad_phi_interior,
                where=f"KirchhoffLayer {self.name!r} adjoint",
                method=linear_method(self),
            )
        phi = self.assemble(phi_interior, phi_boundary)
        dq = self._dflows(phi, drivers, step.ctx)
        op = self._operator_at(dq, self._withdrawal_slopes(phi, drivers, step))
        return _adjoint_solve(
            op,
            grad_phi_interior,
            where=f"KirchhoffLayer {self.name!r} adjoint",
            method=self.linear_solver,
        )

    def _check_adjoint_grad(self, grad: torch.Tensor) -> None:
        """Refuse an `adjoint` gradient on a layer with flow unknowns unless it is real and
        has the unknowns' width (a fast-path habit of passing `n_I` columns, or a complex
        gradient on a phasor layer, would otherwise fail deep in the linear solver)."""
        u = self._unknowns
        if grad.shape[-1] != u.n or grad.is_complex():
            got = "a complex gradient" if grad.is_complex() else str(grad.shape[-1])
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: grad must have the width of the unknowns "
                f"[phi_I, q_F], {u.n} ({u.n_I} interior potentials, then {u.n_F} flow "
                f"unknown(s){', each a (Re, Im) pair: the real view' if u.w == 2 else ''}); "
                f"got {got}")

    def power_residual(self, phi, q, drivers, sources=None):
        """Tellegen's power identity, zero at a converged solution.

        (dp*conj(q)).sum() - (head*conj(q)).sum() - (phi_bound * conj(A_bound @ q)).sum()
        - (phi_interior * conj(sources_interior)).sum()

        The conjugates change nothing on a real layer; on a phasor layer the identity is the
        complex power balance (real and imaginary parts both zero).

        Derivation: dp = A^T phi + head, so dp^T q = phi^T (A q) + head^T q. Splitting
        phi^T (A q) into interior and boundary parts and using A_I q = sources_I at a
        converged solution (`residual` is exactly this equation) gives
        dp^T q - head^T q - phi_bound.(A_bound q) - phi_interior.sources_I == 0.
        `sources=None` (the default) means zero interior sources, matching `residual` and
        `linear_init`.
        """
        drivers = drivers or {}
        d = self.dp(phi, drivers)
        head_only = d - self._difference(phi)
        boundary_flow = self._accumulate_bound(q)
        phi_b = phi[..., self.bound]
        phi_i = phi[..., self.interior]
        s_I = self._source_interior(sources, phi_i)
        return (
            (d * q.conj()).sum(-1)
            - (head_only * q.conj()).sum(-1)
            - (phi_b * boundary_flow.conj()).sum(-1)
            - (phi_i * s_I.conj()).sum(-1)
        )
