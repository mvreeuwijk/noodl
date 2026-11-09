"""The layout of a `KirchhoffLayer` whose flows are not all eliminated.

A law that cannot give the flow from the potential difference gives a relation instead,
and the layer adds its flow as an unknown; a layer whose laws couple several edges, store,
or read the law context is laid out the same way. Unknowns are `[phi_I, q_F]`: the interior
potentials and the flows of the relation edges, in layer edge order. Residual rows are the
flow balance at the interior nodes (`A_I q - s_I + w(phi)`, with `q` the law flow on
potential-controlled edges and the unknown on relation edges), then the relations' own
residuals. The Jacobian is assembled as COO (`SparseOperator`) on a fixed pattern from
per-law local blocks: for a law of `ports` m whose edges form G groups, the block
`d out[g, k] / d in[g, l]` is obtained from ONE evaluation of the law and m reverse-mode
derivatives (`torch.autograd.grad`), cotangent k seeding slot k of every group at once (the
groups are independent, so each gradient holds row k of every group's block). A relation's
blocks with respect to `dp` and to `q` come from the same evaluation. Reverse mode, not
forward mode: PyTorch's forward-mode rules for elementwise ops run through its Python
reference implementations (`torch._refs`), about 80 times the cost of the op itself, which
made the Jacobian most of a step.

This module holds that layout (`FlowUnknowns`, built by `flow_unknowns`), the phasor views
and the diagnosis of a singular system, and `MnaSystem`, the residual, Jacobian and solve on
that layout.

Phasor layers (complex potentials and flows) are solved in the real view: each complex
unknown is a (Re, Im) pair, `w = 2`, and each local block entry becomes the real 2x2 block of
the (possibly non-holomorphic) law, differentiated through the real views of its input and
output.

Node storage (`KirchhoffLayer(storage=)`, real layers only) enters the balance rows as on the
fast path: on a transient step its withdrawal `NodeStorage.rate` is added beside the node
sources', and its analytic `rate_slope` to the diagonal slot of each interior row. At the
consistent initial point (`mode="initial"`) a node with a storage entry whose `initial()` is
not `None` is held at that potential: its balance row is replaced by `phi_i - initial_i = 0`
(`_Held`). In the steady operating point (`mode="dc"`) storage is absent.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from noodl._broadcast import broadcast_shapes
from noodl.elements._slopes import _colour_blocks, _complex
from noodl.elements.base import LawContext
from noodl.layers._device import complex_of
from noodl.layers._solve_policy import SolvePolicy
from noodl.operators.sparse import SparseOperator
from noodl.solvers.newton import newton_tolerances
from noodl.solvers.select import scipy_importable

Tensor = torch.Tensor

#: The policy of the differentiable solve `MnaSystem.solve` runs when it is given `fn`.
_DIFFERENTIATED = SolvePolicy(differentiable=True)

#: The largest system whose failed solve is diagnosed by a dense rank test; above it the
#: original solver error is re-raised unchanged (a dense SVD of a big system costs more
#: than the diagnosis is worth).
_RANK_TEST_MAX_UNKNOWNS = 2000


def _cat(parts: list[Tensor], like: Tensor) -> Tensor:
    """Concatenate on the last dimension after broadcasting the leading (batch) dims."""
    if not parts:
        return torch.zeros(like.shape[:-1] + (0,), dtype=like.dtype, device=like.device)
    batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
    return torch.cat([p if p.shape[:-1] == batch else p.expand(batch + p.shape[-1:])
                      for p in parts], dim=-1)


@dataclasses.dataclass(frozen=True)
class _Held:
    """The rows `mode="initial"` holds at a storage's initial potential: `rows` (n_I,) bool
    marks the held interior rows; on the Jacobian's COO values, `clear` (nnz,) bool marks
    every entry of a held row and `ones` indexes the diagonal-slot entries of the held rows,
    which become 1."""

    rows: Tensor
    clear: Tensor
    ones: Tensor


@dataclasses.dataclass(frozen=True)
class FlowUnknowns:
    """The layout of a layer whose flows are not all eliminated: which edges' flows are
    unknowns (`edges`, layer edge order), where each sits in the unknown vector
    `[phi_I, q_F]` (`index`), the fixed sparse pattern of the Jacobian (`row`, `col`,
    `src`, `sign`) and the real width `w` (2 on a phasor layer).

    `src` is, for each entry of the pattern, the slot of the flat block vector it takes its
    value from, with `sign`; `one_slot` is the slot of the constant 1 of the balance rows'
    flow-unknown columns, `node_slot` the first of the diagonal slots of withdrawals and
    storage, present when `diag` (the last `n_I` entries of the pattern)."""

    w: int
    n_I: int
    edges: Tensor
    index: Tensor
    row: Tensor
    col: Tensor
    src: Tensor
    sign: Tensor
    one_slot: int
    node_slot: int
    diag: bool

    @property
    def n_F(self) -> int:
        """The number of flow unknowns."""
        return len(self.edges)

    @property
    def n(self) -> int:
        """The real width of the unknown vector, `w * (n_I + n_F)`."""
        return self.w * (self.n_I + self.n_F)


def flow_unknowns(layer) -> FlowUnknowns:
    """The `FlowUnknowns` of `layer`: its flow unknowns (the edges of its relations) and the
    fixed COO pattern of its Jacobian."""
    w = 2 if layer.phasor else 1
    b = len(layer.cols)
    f_parts = [torch.arange(s, e) for rel, (s, e) in
               zip(layer._relation, layer._law_slices, strict=True) if rel]
    edges = torch.cat(f_parts) if f_parts else torch.zeros(0, dtype=torch.long)
    n_I = len(layer.interior)
    n_F = len(edges)
    f_index = torch.full((b,), -1, dtype=torch.long)
    f_index[edges] = torch.arange(n_F)
    node = torch.stack([layer._interior_of_node[layer._src],
                        layer._interior_of_node[layer._tgt]], dim=-1)   # (b, 2), -1 bound

    sgn = torch.tensor([1.0, -1.0], dtype=torch.float64)
    rows, cols, srcs, signs = [], [], [], []
    offset = 0
    ar = torch.arange
    for el, rel, (s, e) in zip(layer._laws, layer._relation, layer._law_slices, strict=True):
        m = el.ports
        G = (e - s) // m
        g, k, l_, ck, cl, a, bb = torch.meshgrid(
            ar(G), ar(m), ar(m), ar(w), ar(w), ar(2), ar(2), indexing="ij")
        ek = s + g * m + k
        el_ = s + g * m + l_
        flat = (((g * m + k) * m + l_) * w + ck) * w + cl
        size = G * m * m * w * w
        if not rel:
            # balance x phi: +-q_k at edge k's endpoints, d dp_l / d phi = +-1 at edge l's
            rn, cn = node[ek, a], node[el_, bb]
            keep = (rn >= 0) & (cn >= 0)
            rows.append((rn * w + ck)[keep])
            cols.append((cn * w + cl)[keep])
            srcs.append((offset + flat)[keep])
            signs.append((sgn[a] * sgn[bb])[keep])
            offset += size
        else:
            rrow = (n_I + f_index[ek]) * w + ck
            # d r / d dp: columns at the potentials of edge l's endpoints
            cn = node[el_, bb]
            keep = (cn >= 0) & (a == 0)
            rows.append(rrow[keep])
            cols.append((cn * w + cl)[keep])
            srcs.append((offset + flat)[keep])
            signs.append(sgn[bb][keep])
            offset += size
            # d r / d q: columns at the flow unknowns of edge l
            keep = (a == 0) & (bb == 0)
            rows.append(rrow[keep])
            cols.append(((n_I + f_index[el_]) * w + cl)[keep])
            srcs.append((offset + flat)[keep])
            signs.append(torch.ones(int(keep.sum()), dtype=torch.float64))
            offset += size
    # balance x q_F: the incidence of relation edges, a constant +-1 (slot `one`)
    one_slot = offset
    offset += 1
    if n_F:
        f, a, c = torch.meshgrid(ar(n_F), ar(2), ar(w), indexing="ij")
        rn = node[edges[f], a]
        keep = rn >= 0
        rows.append((rn * w + c)[keep])
        cols.append(((n_I + f) * w + c)[keep])
        srcs.append(torch.full((int(keep.sum()),), one_slot))
        signs.append(sgn[a][keep])
    # withdrawals and storage (real layers only; a phasor layer refuses both): the
    # diagonal slope per interior row, the last n_I entries of the pattern
    node_slot = offset
    diag = bool(layer._withdrawals or layer._storage) and w == 1
    if diag:
        idx = ar(n_I)
        rows.append(idx)
        cols.append(idx)
        srcs.append(offset + idx)
        signs.append(torch.ones(n_I, dtype=torch.float64))
    return FlowUnknowns(
        w=w, n_I=n_I, edges=edges, index=f_index, row=torch.cat(rows), col=torch.cat(cols),
        src=torch.cat(srcs), sign=torch.cat(signs), one_slot=one_slot, node_slot=node_slot,
        diag=diag)


# ----------------------------------------------------------------- phasor views
def cdtype(layer) -> torch.dtype:
    """The complex dtype of a phasor layer: that of its network's precision (read at each
    use, so `Model.to` changes it)."""
    return complex_of(layer.net.dtype)


def to_real(layer, z: Tensor) -> Tensor:
    """`z` in the real view of `layer`'s unknowns: each complex value a (Re, Im) pair on a
    phasor layer, `z` itself otherwise."""
    return torch.view_as_real(_complex(z, cdtype(layer))).flatten(-2) if layer.phasor else z


def to_complex(x: Tensor) -> Tensor:
    """The complex values of the real view `x` (consecutive (Re, Im) pairs)."""
    return torch.view_as_complex(x.unflatten(-1, (-1, 2)).contiguous())


def split(u: FlowUnknowns, x: Tensor) -> tuple[Tensor, Tensor]:
    """`(phi_I, q_F)` of the unknown vector `x`, in the complex view on a phasor layer."""
    z = to_complex(x) if u.w == 2 else x
    return z[..., : u.n_I], z[..., u.n_I:]


def angular_frequency(layer, drivers) -> Tensor:
    """`2 pi f` from a phasor layer's frequency driver, with a trailing axis for the law's
    edges."""
    key = layer.frequency_key
    if key not in drivers:
        raise KeyError(
            f"KirchhoffLayer {layer.name!r} is a phasor layer and needs the "
            f"driver {key!r}: the frequency in Hz, one per instance"
        )
    f = drivers[key]
    if not isinstance(f, Tensor):
        f = torch.tensor(f, dtype=torch.float64)
    return 2.0 * math.pi * f.unsqueeze(-1)


def check_frequency(layer, drivers, batches) -> None:
    """Refuse a frequency driver that is not one value per instance of the solve.

    The frequency must broadcast with the inputs' batch shape (`batches`, from the boundary
    potentials, sources and warm starts) without adding dimensions: a frequency of shape
    `(..., 1)` against a batch `(...)` would otherwise broadcast into an outer product of
    instances and frequencies. With unbatched inputs the frequency sets the batch itself."""
    key = layer.frequency_key
    f = drivers.get(key)
    if not isinstance(f, Tensor):
        return
    batch = broadcast_shapes(*batches)
    if not batch:
        return
    try:
        broadcast_shapes(tuple(f.shape), batch)
        fits = f.dim() <= len(batch)
    except RuntimeError:
        fits = False
    if not fits:
        raise ValueError(
            f"KirchhoffLayer {layer.name!r}: driver {key!r} has shape "
            f"{tuple(f.shape)}, which is not one frequency per instance of the batch "
            f"{tuple(batch)}; give it the batch shape (...), not (..., 1)"
        )


# ----------------------------------------------------------------- linear solver
def linear_method(layer, requested: str | None = None) -> str:
    """The inner linear solver for a request (default: the layer's `linear_solver`): the
    request itself unless `"auto"`, which factorises: SciPy's sparse LU per instance
    (`"sparse_direct"`) whenever SciPy is importable, at any batch size, and batched dense
    LU (`"direct"`) only when it is not. The system is indefinite, so never CG.

    Sparse LU at every batch size: a dense batched factorisation costs `batch * n^2` memory
    and `batch * n^3` work, so a frequency sweep of a mid-sized circuit (64 frequencies, 1604
    unknowns) took 6.1 s and 1.3 GB dense against 0.48 s per instance with SuperLU. A tiny
    circuit at a huge batch may be faster dense; ask for `method="direct"` there."""
    requested = layer.linear_solver if requested is None else requested
    if requested != "auto":
        return requested
    return "sparse_direct" if scipy_importable() else "direct"


# ----------------------------------------------------------------- diagnosis
def singular(op: SparseOperator | None, n: int, failing: int) -> bool:
    """True when the last Jacobian `op` (of `n` unknowns) is (numerically) rank deficient in
    the flat batch instance `failing`. Only a system of at most `_RANK_TEST_MAX_UNKNOWNS`
    unknowns is tested (a dense rank test of one instance); a larger one is reported as not
    singular, so the caller re-raises the solver's own error."""
    if op is None or n == 0 or n > _RANK_TEST_MAX_UNKNOWNS:
        return False
    values = op.values.reshape(-1, op.values.shape[-1])
    A = SparseOperator(op.row, op.col, values[min(failing, len(values) - 1)], n)
    A = A.assemble()
    if not bool(torch.isfinite(A).all()):
        return False
    return bool(torch.linalg.matrix_rank(A) < n)


def first_failing(last: dict, newton_kwargs: dict) -> int:
    """The first flat batch instance whose last residual is above Newton's tolerance
    (relative to its first residual), or 0 when none is (or no residual was seen)."""
    if "r" not in last:
        return 0
    r0 = last["r0"].reshape(-1, last["r0"].shape[-1])
    r = last["r"].reshape(-1, last["r"].shape[-1])
    if r.shape != r0.shape or r.shape[-1] == 0:
        return 0
    atol, rtol = newton_tolerances(
        r.dtype, newton_kwargs.get("atol"), newton_kwargs.get("rtol"))
    norm0 = r0.abs().amax(dim=-1)
    norm = r.abs().amax(dim=-1)
    bad = torch.nonzero(~(norm < atol + rtol * norm0)).flatten()
    return int(bad[0]) if len(bad) else 0


def singular_error(layer, kinds, mode: str) -> RuntimeError:
    """The error for a singular system of potentials and flow unknowns, naming the relation
    `kinds`; at the initial point it also says that storage held at its initial value is
    itself a source, and where else to start."""
    initial = (
        "; at the initial point (mode='initial') every dynamic law is held at its initial "
        "value, so a capacitor acts as a voltage source at its v0 and an inductor as a "
        "current source at its i0: a capacitor across a voltage source, or an inductor "
        "in series with another or with a current source, makes it singular. Start the "
        "transient from the operating point instead (mode='dc'; operating_point in the "
        "electrical application)"
    ) if mode == "initial" else ""
    return RuntimeError(
        f"KirchhoffLayer {layer.name!r}: the system of potentials and flow unknowns is "
        f"singular; check for a loop of ideal potential sources or a cut set of flow "
        f"sources among the relation kinds {kinds}, or a part connected to a boundary "
        f"only through flow-prescribed edges{initial}"
    )


class MnaSystem:
    """The residual, Jacobian and solve of one `KirchhoffLayer` with flow unknowns, on its
    layout `unknowns` (`FlowUnknowns`)."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self._law_ctxs: tuple[LawContext | None, dict] = (None, {})
        u = self.unknowns = flow_unknowns(layer)
        self.w, self.n_I, self.n_F, self.n = u.w, u.n_I, u.n_F, u.n
        self.f_edges, self._f_index = u.edges, u.index
        self.row, self.col, self.src, self.sign = u.row, u.col, u.src, u.sign
        self._one_slot, self._node_slot, self._diag = u.one_slot, u.node_slot, u.diag
        self.implicit_kinds = [el.kind for el, rel in
                               zip(layer._laws, layer._relation, strict=True) if rel]

    # ------------------------------------------------------------- views
    def _to_real(self, z: Tensor) -> Tensor:
        return to_real(self.layer, z)

    def _unknowns(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return split(self.unknowns, x)

    # ------------------------------------------------------------- laws
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
                f"layer {self.layer.name!r}: the {what} of the {el.kind!r} law is {width} "
                f"but the kind has {n_edges} edges")

    def _flows_and_residuals(self, phi, q_F, drivers, ctx, call) -> tuple[Tensor, Tensor]:
        """`(q, r_F)`: every edge's flow (the law flow on potential-controlled edges, the
        unknown on implicit ones) and the implicit laws' residuals, in layer edge order."""
        layer = self.layer
        dp = layer.dp(phi, drivers)
        q_parts, r_parts = [], []
        for i, (el, (s, e)) in enumerate(
                zip(layer._laws, layer._law_slices, strict=True)):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            if layer._relation[i]:
                f0 = int(self._f_index[s])
                q_s = q_F[..., f0:f0 + e - s]
                r_s = call(i, (dp_s, q_s, drivers, lctx))
                self._check_width(el, r_s, e - s, "residual")
                r_parts.append(r_s)
            else:
                q_s = call(i, (dp_s, drivers, lctx))
                self._check_width(el, q_s, e - s, "flow")
            q_parts.append(q_s)
        return _cat(q_parts, dp), _cat(r_parts, dp)

    def flows(self, phi, drivers, q) -> Tensor:
        """`KirchhoffLayer.flows` of an MNA layer: `q`'s implicit-edge columns as given, the
        other edges' flows from their laws at `phi` (steady context)."""
        layer = self.layer
        if self.w == 2:
            raise TypeError(
                f"KirchhoffLayer {layer.name!r} is a phasor layer: flows() is not defined on "
                f"it; read the flows from solve()'s q")
        n_edges = len(layer.cols)
        if q.shape[-1] != n_edges:
            raise ValueError(
                f"KirchhoffLayer {layer.name!r}: q must hold every edge's flow "
                f"({n_edges}) as solve() returns it; got last dimension {q.shape[-1]}")
        out, _ = self._flows_and_residuals(phi, q[..., self.f_edges], drivers,
                                           LawContext(mode="dc"), layer._owner_call)
        return out

    def _withdrawal(self, phi, drivers, call) -> Tensor | None:
        """The withdrawals' withdrawal per interior row, or `None` without withdrawals."""
        n_el = len(self.layer._laws)
        return self.layer._scatter_withdrawals(
            lambda j, ns: call(n_el + j, (phi[..., ns.nodes], drivers)))

    def _node_slopes(self, phi, drivers, adjoint: bool = False) -> Tensor | None:
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

        return self.layer._scatter_withdrawals(slope)

    # ------------------------------------------------------------- storage
    def _held_values(self, drivers, call, *, check: bool = False):
        """`(rows, value)`: the interior rows held at the initial point ((n_I,) bool) and
        the potential each is held at ((..., n_I), zero elsewhere); `None` when no storage
        holds an initial potential. `initial` is evaluated through `call`, so on the
        differentiable path with the threaded parameters. `check` refuses entries at one
        node with different initial values, by name."""
        layer = self.layer
        base = len(layer._laws) + len(layer._withdrawals)
        held = []
        for j, _st in enumerate(layer._storage):
            v = call(base + j, (drivers,), {"part": "initial"})
            if v is not None:
                pos = layer._storage_positions[j]
                v = torch.as_tensor(v, device=pos.device)
                held.append((j, pos, v.expand(v.shape[:-1] + pos.shape) if v.dim()
                             else v.expand(pos.shape)))
        if not held:
            return None
        batch = broadcast_shapes(*(v.shape[:-1] for _, _, v in held))
        dtype = held[0][2].dtype
        value = torch.zeros(batch + (self.n_I,), dtype=dtype, device=held[0][2].device)
        rows = torch.zeros(self.n_I, dtype=torch.bool, device=value.device)
        for _, pos, v in held:
            value = value.index_copy(-1, pos, v.to(dtype).expand(batch + pos.shape))
            rows[pos] = True
        if check:
            for _, pos, v in held:
                differ = (v.to(dtype) != value[..., pos]).reshape(-1, pos.numel()).any(0)
                if bool(differ.any()):
                    row = pos[differ.nonzero()[0, 0]]
                    names = [f"{type(layer._storage[k]).__name__} (storage {k})"
                             for k, p, _ in held if bool((p == row).any())]
                    node = layer.net.nodes[int(layer.interior[row])]
                    raise ValueError(
                        f"KirchhoffLayer {layer.name!r}: the storage entries at node "
                        f"{node!r} hold different initial potentials ({', '.join(names)}); "
                        f"mode='initial' holds a node at one value")
        return rows, value

    def _held(self, drivers, call) -> _Held | None:
        """The held rows of `mode="initial"` and their masks on the Jacobian's values
        (refusing entries at one node that disagree)."""
        found = self._held_values(drivers, call, check=True)
        if found is None:
            return None
        rows = found[0].to(self.row.device)
        full = torch.zeros(self.n, dtype=torch.bool, device=rows.device)
        full[: self.n_I] = rows
        diag = torch.arange(len(self.row) - self.n_I, len(self.row), device=rows.device)
        return _Held(rows=rows, clear=full[self.row], ones=diag[rows])

    def _storage_withdrawal(self, phi, drivers, step, ctx, call) -> Tensor | None:
        """The storages' withdrawal `NodeStorage.rate` per interior row on a transient
        `step` (a `_StorageStep`), in the context `ctx`."""
        layer = self.layer
        base = len(layer._laws) + len(layer._withdrawals)
        return layer._scatter_storage(lambda j, st: call(
            base + j, (*layer._storage_args(j, st, phi, step), ctx, drivers)))

    def _storage_slopes(self, phi, drivers, step, ctx, call, adjoint: bool) -> Tensor | None:
        """`d w / d phi` of the storages per interior row on a transient `step` (a storage
        reads its own node's potential only, so this is the whole diagonal): their analytic
        `rate_slope`, except, when the operator also builds the adjoint (`adjoint`, the
        differentiable solve), for a storage that declares `exact_slope = False`, whose
        slope is then the derivative of its `rate` by autograd. The values carry no graph,
        like every Jacobian entry."""
        layer = self.layer
        base = len(layer._laws) + len(layer._withdrawals)

        def slope(j, st):
            phi_n, phi_start, previous = layer._storage_args(j, st, phi, step)
            if adjoint and not st.exact_slope:
                block, = _colour_blocks(
                    lambda v: call(base + j, (v, phi_start, previous, ctx, drivers)),
                    (phi_n,), 1, 1)
                return block[..., 0, 0, 0, 0]
            return st.rate_slope(phi_n, phi_start, previous, ctx, drivers)

        d = layer._scatter_storage(slope)
        return None if d is None else d.detach()

    def _slopes_fn(self, drivers, call, ctx, store):
        """The diagonal of the withdrawals and, on a step, of the storages, as a function
        `slopes(phi, adjoint)` of the full-node potentials (`None` when there is neither).
        `adjoint` is `operator`'s own flag, passed on to `_node_slopes` and
        `_storage_slopes`: the operator also builds the adjoint."""
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
        return SparseOperator(op.row, op.col, values, self.n)

    # ------------------------------------------------------------- residual and Jacobian
    def residual(self, x, phi_boundary, drivers, sources, ctx, call, store=None) -> Tensor:
        """KCL at the interior nodes, then the implicit laws' residuals (real view).
        `store` (the step's `storage`) adds the storage withdrawal on a step, or replaces the
        held rows at the initial point."""
        return self._residual_parts(x, phi_boundary, drivers, sources, ctx, call, store)[0]

    def _residual_parts(self, x, phi_boundary, drivers, sources, ctx, call, store=None):
        """`(residual, phi, q)`: `residual` and the full-node potentials and every edge's flow
        it was evaluated at (the solve returns the last two at the root)."""
        layer = self.layer
        phi_I, q_F = self._unknowns(x)
        phi = layer.assemble(phi_I, phi_boundary)
        q, r_F = self._flows_and_residuals(phi, q_F, drivers, ctx, call)
        kcl = layer._accumulate_interior(q) - layer._source_interior(sources, phi_I)
        w = self._withdrawal(phi, drivers, call)
        if w is not None:
            kcl = kcl + w
        if isinstance(store, _Held):
            _, value = self._held_values(drivers, call)
            kcl = torch.where(store.rows, phi_I - value.to(phi_I.dtype), kcl)
        elif store is not None:
            kcl = kcl + self._storage_withdrawal(phi, drivers, store, ctx, call)
        return self._to_real(_cat([kcl, r_F], kcl)), phi, q

    def operator(self, x, phi_boundary, drivers, ctx, node_slope_fn,
                 adjoint: bool = False) -> SparseOperator:
        """The Jacobian of `residual` at `x`, as COO triplets on the fixed pattern.

        `node_slope_fn(phi, adjoint)` gives the diagonal of the withdrawals and storages
        (`_slopes_fn`). `adjoint`: the operator also builds the adjoint (the differentiable
        solve). A law that declares `exact_slope = False` then gives its blocks by autograd
        of its own `forward` (`_coloured_jacobian`), not its `jacobian`, whose `dflow`
        shortcut would hand the adjoint a Newton aid; the same flag reaches
        `node_slope_fn`, where an inexact withdrawal or storage gives autograd too. The
        primal solve (`adjoint=False`) keeps the analytic slopes."""
        layer = self.layer
        phi_I, q_F = self._unknowns(x)
        phi = layer.assemble(phi_I, phi_boundary)
        dp = layer.dp(phi, drivers)
        blocks = []
        for el, rel, (s, e) in zip(layer._laws, layer._relation, layer._law_slices,
                                   strict=True):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            # Called on the law module itself, also on the differentiable path: the values
            # equal those of a `functional_call` with the threaded parameters, because the
            # threaded tensors ARE the module's own parameters (the same argument as
            # `dp_independent` in `_functional.py`), and the blocks carry no graph.
            if rel:
                f0 = int(self._f_index[s])
                q_s = q_F[..., f0:f0 + e - s]
            else:
                q_s = None
            if adjoint and not el.exact_slope:
                blocks += el._coloured_jacobian(dp_s, q_s, drivers, lctx)
            else:
                blocks += el.jacobian(dp_s, q_s, drivers, lctx)
        parts = [blk.flatten(-5) for blk in blocks]
        parts.append(torch.ones(1, dtype=x.dtype, device=x.device))
        if self._diag:
            d = node_slope_fn(phi, adjoint)
            parts.append(d if d is not None else
                         torch.zeros(self.n_I, dtype=x.dtype, device=x.device))
        flat = _cat(parts, x)
        values = flat[..., self.src] * self.sign.to(dtype=flat.dtype, device=flat.device)
        return SparseOperator(self.row, self.col, values, self.n)

    # ------------------------------------------------------------- solve
    def inputs(self, phi_boundary, drivers, sources, phi0, q0):
        """`(phi_boundary, sources, phi0, q0)` as the solve takes them: on a phasor layer in
        the complex dtype of its network's precision (complex128 on a float64 network), real
        or complex as given, with the frequency driver checked against their batch."""
        if self.w != 2:
            return phi_boundary, sources, phi0, q0
        dtype = cdtype(self.layer)
        phi_boundary, sources, phi0, q0 = (
            None if v is None else torch.as_tensor(v).to(dtype)
            for v in (phi_boundary, sources, phi0, q0))
        check_frequency(self.layer, drivers, [
            v.shape[:-1] for v in (phi_boundary, sources, phi0, q0) if v is not None])
        return phi_boundary, sources, phi0, q0

    def refuse(self, newton_kwargs) -> None:
        """Refuse the Newton keywords this system cannot take, by name: the convex line
        search and conjugate gradients."""
        layer = self.layer
        if newton_kwargs.get("line_search") == "convex":
            raise ValueError(
                f"KirchhoffLayer {layer.name!r}: line_search='convex' needs a residual "
                f"that is the gradient of a convex function, which the residual of a layer "
                f"with flow unknowns is not; use the default line search"
            )
        if newton_kwargs.get("method", layer.linear_solver) == "cg":
            raise ValueError(
                f"KirchhoffLayer {layer.name!r}: method='cg' needs a symmetric positive "
                f"definite Jacobian, which a layer with flow unknowns does not have; use "
                f"'auto', 'sparse_direct', 'direct' or 'gmres'"
            )

    def solve(self, phi_boundary, drivers, sources, phi0, q0, step, fn, diagnostics,
              newton_kwargs):
        """The solve of `KirchhoffLayer.solve` on this system, after its checks and its
        policy: the implicit-function solve through the threaded parameters `fn` (a
        `FunctionalLaws`) where the solve is differentiated, else (`fn` is `None`) the primal
        Newton, which is also what a differentiable solve that nothing reaches runs."""
        layer = self.layer
        ctx, store = step.ctx, step.storage
        plain = layer._owner_call
        real_dtype = phi_boundary.real.dtype if phi_boundary.is_complex() else phi_boundary.dtype

        # The initial guess: zeros, or the caller's warm start. It is not differentiable
        # (the adjoint linearises at the converged point), so it is built under no_grad.
        with torch.no_grad():
            probe = self.residual(
                torch.zeros(self.n, dtype=real_dtype, device=phi_boundary.device),
                phi_boundary, drivers, sources, ctx, plain, store)
            shapes = [probe.shape[:-1]]
            if phi0 is not None:
                shapes.append(phi0.shape[:-1])
            if q0 is not None:
                shapes.append(q0.shape[:-1])
            B = broadcast_shapes(*shapes)
            x0 = torch.zeros(B + (self.n,), dtype=real_dtype, device=phi_boundary.device)
            n_phi = self.w * self.n_I
            if phi0 is not None:
                x0[..., :n_phi] = self._to_real(phi0.detach())
            if q0 is not None:
                x0[..., n_phi:] = self._to_real(q0.detach())

        requested = newton_kwargs["method"]
        newton_kwargs["method"] = linear_method(layer, requested)
        if layer.linear:
            # The residual is affine: the full Newton step lands on the solution, so the
            # damped first step (`newton`'s omega=0.75) would only cost a second iteration.
            newton_kwargs.setdefault("omega", 1.0)
        # The last Jacobian and the first and last residuals, kept to diagnose a failure.
        last: dict = {}

        def remember(r):
            last.setdefault("r0", r.detach())
            last["r"] = r.detach()
            return r

        def diagnose(exc):
            if singular(last.get("op"), self.n, first_failing(last, newton_kwargs)):
                raise singular_error(layer, self.implicit_kinds, ctx.mode) from exc

        if fn is None:
            # the primal solve: also a differentiable one that nothing reaches
            point: dict = {}

            def residual_fn(x):
                r, phi, q = self._residual_parts(
                    x, phi_boundary, drivers, sources, ctx, plain, store)
                point.update(x=x, phi=phi, q=q)
                return remember(r)

            slopes_fn = self._slopes_fn(drivers, plain, ctx, store)

            def operator_fn(x):
                last["op"] = self._hold(
                    self.operator(x, phi_boundary, drivers, ctx, slopes_fn), store)
                return last["op"]

            with torch.no_grad():
                result = layer._newton_primal(residual_fn, operator_fn, x0, newton_kwargs,
                                              diagnostics, diagnose, requested=requested)
                if point.get("x") is result.x:
                    # Newton's last residual was at the point it returns
                    return point["phi"], point["q"]
                return self._output(result.x, phi_boundary, drivers, ctx)

        def residual_fn(x, *params):
            rebuilt, drv, src, pb, st = fn.rebuild(params)
            r, phi, q = self._residual_parts(x, pb, drv, src, st.ctx,
                                             functools.partial(fn.call, rebuilt), st.storage)
            if not q.requires_grad:
                fn.last = (x, phi, q)
            return remember(r)

        def operator_fn(x, *params):
            rebuilt, drv, src, pb, st = fn.rebuild(params)
            call = functools.partial(fn.call, rebuilt)
            # the one `adjoint` flag: the operator also builds the adjoint, so every law,
            # withdrawal and storage that declares `exact_slope = False` gives autograd
            last["op"] = self._hold(
                self.operator(x, pb, drv, st.ctx, self._slopes_fn(drv, call, st.ctx, st.storage),
                              adjoint=True), st.storage)
            return last["op"]

        x = layer._newton_differentiable(fn, residual_fn, operator_fn, x0, _DIFFERENTIATED,
                                         newton_kwargs, diagnostics, diagnose,
                                         requested=requested)
        parts = fn.root_parts(x)
        if parts is not None:
            return parts
        return self._output(x, phi_boundary, drivers, ctx)

    def _output(self, x, phi_boundary, drivers, ctx) -> tuple[Tensor, Tensor]:
        """`(phi, q)` at the solution `x`: full-node potentials and every edge's flow. The
        laws are called on the layer's own modules, so on the differentiable path `q` keeps
        the parameters' graph as the classic path's final `flows` does."""
        phi_I, q_F = self._unknowns(x)
        phi = self.layer.assemble(phi_I, phi_boundary)
        q, _ = self._flows_and_residuals(phi, q_F, drivers, ctx, self.layer._owner_call)
        return phi, q
