"""The layout of a `KirchhoffLayer` whose flows are not all eliminated.

A law that cannot give the flow from the potential difference is an implicit law (a relation
`r(dp, q) = 0`), and the layer adds its flow as an unknown; a layer whose laws couple
several edges, store, or read the law context is laid out the same way. Unknowns are
`[phi_I, q_F]`: the interior potentials and the flows of the implicit edges, in layer edge
order. Residual rows are the
flow balance at the interior nodes (`A_I q - s_I + w(phi)`, with `q` the law flow on
potential-controlled edges and the unknown on implicit edges), then the implicit laws' own
residuals. The Jacobian is assembled as COO (`SparseOperator`) on a fixed pattern from
per-law local blocks: for a law of `ports` m whose edges form G groups, the block
`d out[g, k] / d in[g, l]` is obtained from ONE evaluation of the law and m reverse-mode
derivatives (`torch.autograd.grad`), cotangent k seeding slot k of every group at once (the
groups are independent, so each gradient holds row k of every group's block). An implicit law's
blocks with respect to `dp` and to `q` come from the same evaluation. Reverse mode, not
forward mode: PyTorch's forward-mode rules for elementwise ops run through its Python
reference implementations (`torch._refs`), about 80 times the cost of the op itself, which
made the Jacobian most of a step.

This module holds that layout (`FlowUnknowns`, built by `flow_unknowns`), the phasor views
and the diagnosis of a singular system. The residual, the Jacobian and the solve are
`KirchhoffLayer`'s (`_residual`, `_operator`, `solve`).

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
import math

import torch

from noodl._broadcast import broadcast_shapes
from noodl.elements._slopes import _complex
from noodl.layers._device import complex_of
from noodl.operators.sparse import SparseOperator
from noodl.solvers.errors import SingularSystemError
from noodl.solvers.newton import newton_tolerances
from noodl.solvers.select import scipy_importable

Tensor = torch.Tensor

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
    storage, present when `diag` (the last `n_I` entries of the pattern).

    An implicit law takes two blocks, `d r / d dp` (used with sign +1 in the start node's column
    and -1 in the end node's) and `d r / d q`; one that `reads_heads` takes three, the
    derivative in the start node's column, `d r / d q` and the derivative in the end node's
    column, each with sign +1, on the same pattern."""

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
    """The `FlowUnknowns` of `layer`: its flow unknowns (the edges of its implicit laws) and the
    fixed COO pattern of its Jacobian."""
    w = 2 if layer.phasor else 1
    b = len(layer.cols)
    f_parts = [torch.arange(s, e) for implicit, (s, e) in
               zip(layer._implicit, layer._law_slices, strict=True) if implicit]
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
    for el, implicit, (s, e) in zip(layer._laws, layer._implicit, layer._law_slices, strict=True):
        m = el.ports
        G = (e - s) // m
        g, k, l_, ck, cl, a, bb = torch.meshgrid(
            ar(G), ar(m), ar(m), ar(w), ar(w), ar(2), ar(2), indexing="ij")
        ek = s + g * m + k
        el_ = s + g * m + l_
        flat = (((g * m + k) * m + l_) * w + ck) * w + cl
        size = G * m * m * w * w
        if not implicit:
            # balance x phi: +-q_k at edge k's endpoints, d dp_l / d phi = +-1 at edge l's
            rn, cn = node[ek, a], node[el_, bb]
            keep = (rn >= 0) & (cn >= 0)
            rows.append((rn * w + ck)[keep])
            cols.append((cn * w + cl)[keep])
            srcs.append((offset + flat)[keep])
            signs.append((sgn[a] * sgn[bb])[keep])
            offset += size
        elif not el.reads_heads:
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
        else:
            # An implicit law that reads the heads at its end nodes (one port): its blocks are
            # `[d r / d phi_from, d r / d q, d r / d phi_to]`, the first and last each the
            # whole derivative in that node's column (`dp`'s share included), so the start
            # node's column takes the first block and the end node's the third, both +1.
            rrow = (n_I + f_index[ek]) * w + ck
            cn = node[el_, bb]
            keep = (cn >= 0) & (a == 0)
            rows.append(rrow[keep])
            cols.append((cn * w + cl)[keep])
            srcs.append((offset + 2 * size * bb + flat)[keep])
            signs.append(torch.ones(int(keep.sum()), dtype=torch.float64))
            keep = (a == 0) & (bb == 0)
            rows.append(rrow[keep])
            cols.append(((n_I + f_index[el_]) * w + cl)[keep])
            srcs.append((offset + size + flat)[keep])
            signs.append(torch.ones(int(keep.sum()), dtype=torch.float64))
            offset += 3 * size
    # balance x q_F: the incidence of implicit edges, a constant +-1 (slot `one`)
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


def singular_error(layer, kinds, mode: str) -> SingularSystemError:
    """The error for a singular system of potentials and flow unknowns, naming the implicit
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
    return SingularSystemError(
        f"KirchhoffLayer {layer.name!r}: the system of potentials and flow unknowns is "
        f"singular; check for a loop of ideal potential sources or a cut set of flow "
        f"sources among the implicit kinds {kinds}, or a part connected to a boundary "
        f"only through flow-prescribed edges{initial}"
    )
