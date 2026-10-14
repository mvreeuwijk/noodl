"""Modified Nodal Analysis for `PotentialFlowLayer`.

A layer whose constitutive laws include an implicit law (flow is an unknown), a coupled
multi-port law, a dynamic law or a context-reading law is solved here instead of by the
classic nodal path. Unknowns are `[phi_I, q_F]`: the interior potentials and the flows of
the implicit-law edges, in layer edge order. Residual rows are Kirchhoff's current law at
the interior nodes (`A_I q - s_I + w(phi)`, with `q` the law flow on potential-controlled
edges and the unknown on implicit ones), then the implicit laws' own residuals. The Jacobian
is assembled as COO (`MnaOperator`) from per-law local blocks: for a law of `ports` m whose
edges form G groups, the block `d out[g, k] / d in[g, l]` is obtained from m forward-mode
derivatives (`torch.func.jvp`), tangent l setting slot l of every group at once.

Phasor layers (complex potentials and flows) are solved in the real view: each complex
unknown is a (Re, Im) pair, `w = 2`, and each local block entry becomes the real 2x2 block of
the (possibly non-holomorphic) law, from the two tangents 1 and 1j.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import warnings

import torch
from torch.func import functional_call, jvp

from noodl._broadcast import broadcast_shapes
from noodl._tracking import bulk_values, same_tracking
from noodl.elements.base import INTEGRATIONS, LawContext
from noodl.operators.mna import MnaOperator
from noodl.solvers.implicit import implicit_solve
from noodl.solvers.newton import newton, newton_tolerances

Tensor = torch.Tensor

#: The largest system whose failed solve is diagnosed by a dense rank test; above it the
#: original solver error is re-raised unchanged (a dense SVD of a big system costs more
#: than the diagnosis is worth).
_RANK_TEST_MAX_UNKNOWNS = 2000


def _scipy_importable() -> bool:
    """True when SciPy (the optional `noodl-physics[sparse]` extra) can be imported."""
    return importlib.util.find_spec("scipy") is not None


_forward_mode_ready = False


def _prepare_forward_mode() -> None:
    """Load PyTorch's forward-mode derivative rules once, quietly.

    The first `torch.func.jvp` of a process compiles them with `torch.jit.script`, which
    PyTorch 2.x reports as a deprecation (`FutureWarning`) that no caller can act on. Doing
    that one load here, under a filter for exactly that message, keeps it from surfacing in
    the first circuit solve of every program.
    """
    global _forward_mode_ready
    if _forward_mode_ready:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".torch\.jit\.script. is deprecated", category=FutureWarning)
        jvp(torch.sin, (torch.zeros(1),), (torch.ones(1),))
    _forward_mode_ready = True


def _complex(z: Tensor) -> Tensor:
    """`z` as a complex128 tensor that `view_as_real` accepts: a phasor-capable law may
    return a real tensor (a constant current) or a lazily conjugated one (a law written with
    `torch.conj`). Both conversions are differentiable."""
    return z.to(torch.complex128).resolve_conj()


def _colour_jacobian(fn, z: Tensor, m: int, w: int) -> Tensor:
    """`(..., G, m, m, w, w)`: d fn(z)[g, k, ck] / d z[g, l, cl] for a groupwise `fn`."""
    shape = z.shape
    G = shape[-1] // m
    cols = []
    for l_ in range(m):
        for cl in range(w):
            t = torch.zeros(shape[:-1] + (G, m), dtype=z.dtype, device=z.device)
            t[..., l_] = 1.0 if cl == 0 else 1j
            _, d = jvp(fn, (z,), (t.reshape(shape),))
            d = d.reshape(d.shape[:-1] + (G, m))
            cols.append(torch.view_as_real(_complex(d)) if w == 2 else d.unsqueeze(-1))
    J = torch.stack(cols, dim=-1).unflatten(-1, (m, w))  # (..., G, k, ck, l, cl)
    return J.transpose(-3, -2)                            # (..., G, k, l, ck, cl)


def _cat(parts: list[Tensor], like: Tensor) -> Tensor:
    """Concatenate on the last dimension after broadcasting the leading (batch) dims."""
    if not parts:
        return torch.zeros(like.shape[:-1] + (0,), dtype=like.dtype, device=like.device)
    batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
    return torch.cat([p.expand(batch + p.shape[-1:]) for p in parts], dim=-1)


class MnaSystem:
    """The Modified Nodal Analysis system of one `PotentialFlowLayer`: its unknown layout,
    the fixed COO pattern of its Jacobian, and the Newton (or implicit-function) solve."""

    def __init__(self, layer) -> None:
        self.layer = layer
        _prepare_forward_mode()
        self.w = 2 if layer.phasor else 1
        b = len(layer.cols)
        f_parts = [torch.arange(s, e) for el, (s, e) in
                   zip(layer._elements, layer._elem_slices, strict=True)
                   if el.law_form == "implicit"]
        self.f_edges = torch.cat(f_parts) if f_parts else torch.zeros(0, dtype=torch.long)
        self.n_I = len(layer.interior)
        self.n_F = len(self.f_edges)
        self.n = self.w * (self.n_I + self.n_F)
        f_index = torch.full((b,), -1, dtype=torch.long)
        f_index[self.f_edges] = torch.arange(self.n_F)
        self._f_index = f_index
        self.implicit_kinds = [el.kind for el in layer._elements if el.law_form == "implicit"]
        node = torch.stack([layer._interior_of_node[layer._src],
                            layer._interior_of_node[layer._tgt]], dim=-1)   # (b, 2), -1 bound
        self._build_structure(node, f_index)

    # ------------------------------------------------------------- structure
    def _build_structure(self, node: Tensor, f_index: Tensor) -> None:
        """Fixed COO pattern: `row`, `col`, and for each entry the slot `src` of the flat
        block vector it takes its value from, with `sign`."""
        w, n_I = self.w, self.n_I
        sgn = torch.tensor([1.0, -1.0], dtype=torch.float64)
        rows, cols, srcs, signs = [], [], [], []
        offset = 0
        ar = torch.arange
        for el, (s, e) in zip(self.layer._elements, self.layer._elem_slices, strict=True):
            m = el.ports
            G = (e - s) // m
            g, k, l_, ck, cl, a, bb = torch.meshgrid(
                ar(G), ar(m), ar(m), ar(w), ar(w), ar(2), ar(2), indexing="ij")
            ek = s + g * m + k
            el_ = s + g * m + l_
            flat = (((g * m + k) * m + l_) * w + ck) * w + cl
            size = G * m * m * w * w
            if el.law_form == "potential":
                # KCL x phi: +-q_k at edge k's endpoints, d dp_l / d phi = +-1 at edge l's
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
        # KCL x q_F: the incidence of implicit edges, a constant +-1 (slot `one`)
        self._one_slot = offset
        offset += 1
        if self.n_F:
            f, a, c = torch.meshgrid(ar(self.n_F), ar(2), ar(w), indexing="ij")
            rn = node[self.f_edges[f], a]
            keep = rn >= 0
            rows.append((rn * w + c)[keep])
            cols.append(((n_I + f) * w + c)[keep])
            srcs.append(torch.full((int(keep.sum()),), self._one_slot))
            signs.append(sgn[a][keep])
        # node sources (real layers only; a phasor layer refuses them): the diagonal slope
        # per interior row
        self._node_slot = offset
        if self.layer._node_sources and w == 1:
            idx = ar(n_I)
            rows.append(idx)
            cols.append(idx)
            srcs.append(offset + idx)
            signs.append(torch.ones(n_I, dtype=torch.float64))
        self.row = torch.cat(rows)
        self.col = torch.cat(cols)
        self.src = torch.cat(srcs)
        self.sign = torch.cat(signs)

    # ------------------------------------------------------------- views
    def _to_real(self, z: Tensor) -> Tensor:
        return torch.view_as_real(_complex(z)).flatten(-2) if self.w == 2 else z

    def _to_complex(self, x: Tensor) -> Tensor:
        return torch.view_as_complex(x.unflatten(-1, (-1, 2)).contiguous())

    def _split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """`(phi_I, q_F)` of the unknown vector, in the complex view on a phasor layer."""
        return z[..., : self.n_I], z[..., self.n_I:]

    def _unknowns(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._split(self._to_complex(x) if self.w == 2 else x)

    # ------------------------------------------------------------- laws
    @property
    def frequency_key(self) -> str:
        """The driver key of a phasor layer's frequency (Hz, one per instance)."""
        return f"{self.layer.name}.frequency"

    def _omega(self, drivers) -> Tensor:
        """`2 pi f` from the frequency driver, with a trailing axis for the law's edges."""
        key = self.frequency_key
        if key not in drivers:
            raise KeyError(
                f"PotentialFlowLayer {self.layer.name!r} is a phasor layer and needs the "
                f"driver {key!r}: the frequency in Hz, one per instance"
            )
        f = drivers[key]
        if not isinstance(f, Tensor):
            f = torch.tensor(f, dtype=torch.float64)
        return 2.0 * math.pi * f.unsqueeze(-1)

    def _check_frequency(self, drivers, batches) -> None:
        """Refuse a frequency driver that is not one value per instance of the solve.

        The frequency must broadcast with the inputs' batch shape (`batches`, from the
        boundary potentials, sources and warm starts) without adding dimensions: a frequency
        of shape `(..., 1)` against a batch `(...)` would otherwise broadcast into an outer
        product of instances and frequencies. With unbatched inputs the frequency sets the
        batch itself."""
        key = self.frequency_key
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
                f"PotentialFlowLayer {self.layer.name!r}: driver {key!r} has shape "
                f"{tuple(f.shape)}, which is not one frequency per instance of the batch "
                f"{tuple(batch)}; give it the batch shape (...), not (..., 1)"
            )

    def _phasor_context(self, mode, dt, t, history, drivers) -> LawContext:
        """The context of a phasor layer: the sinusoidal steady state at the frequency
        driver's angular frequency, which takes no step."""
        name = self.layer.name
        if mode not in (None, "phasor"):
            raise ValueError(
                f"PotentialFlowLayer {name!r} is a phasor layer: mode={mode!r}; expected "
                f"'phasor' or None"
            )
        if dt is not None or history is not None:
            raise ValueError(
                f"PotentialFlowLayer {name!r} is a phasor layer, which solves the "
                f"sinusoidal steady state and takes no step; drop dt and history"
            )
        return LawContext(mode="phasor", t=t, omega=self._omega(drivers))

    def _context(self, mode, dt, t, integration, history, drivers) -> LawContext:
        """The `LawContext` the laws are evaluated in.

        A phasor layer is always in `"phasor"` mode (see `_phasor_context`). Otherwise:

        `mode=None` is `"transient"` when a step `dt` is given and `"dc"` (the steady
        operating point) otherwise; a caller may ask for `"dc"` or `"initial"` (the
        consistent initial point, with no step) explicitly. A transient step reads the
        step-start state `history=(phi, q)` (full-node potentials, every edge's flow in layer
        edge order), required on a layer with a dynamic law; its potential differences are
        taken without drives, which dynamic kinds may not carry.
        """
        if self.w == 2:
            return self._phasor_context(mode, dt, t, history, drivers)
        name = self.layer.name
        if mode == "phasor":
            raise ValueError(
                f"PotentialFlowLayer {name!r}: mode='phasor' needs a layer built with "
                f"phasor=True"
            )
        if mode is not None and mode not in ("dc", "initial"):
            raise ValueError(
                f"PotentialFlowLayer {name!r}: mode={mode!r}; expected 'dc', 'initial' or "
                f"None (transient when dt is given, dc otherwise)"
            )
        if mode in ("dc", "initial") and dt is not None:
            what = ("the consistent initial point" if mode == "initial"
                    else "the steady operating point")
            raise ValueError(
                f"PotentialFlowLayer {name!r}: mode={mode!r} is {what}, which takes no step; "
                f"drop dt"
            )
        if mode is None:
            mode = "transient" if dt is not None else "dc"
        if mode != "transient":
            return LawContext(mode=mode, t=t)
        if not bool((torch.as_tensor(dt) > 0).all()):
            raise ValueError(f"PotentialFlowLayer {name!r}: dt must be positive, got {dt!r}")
        if integration not in INTEGRATIONS:
            raise ValueError(
                f"PotentialFlowLayer {name!r}: integration={integration!r}; expected one of "
                f"{INTEGRATIONS}"
            )
        ctx = LawContext(mode="transient", dt=dt, t=t, integration=integration)
        if history is None:
            if self.layer.dynamic:
                raise ValueError(
                    f"PotentialFlowLayer {name!r}: a transient step needs history=(phi, q) "
                    f"at the step start"
                )
            return ctx
        if not isinstance(history, (tuple, list)) or len(history) != 2:
            raise ValueError(
                f"PotentialFlowLayer {name!r}: history must be the pair (phi, q) at the step "
                f"start; got {type(history).__name__}"
            )
        phi_prev, q_prev = history
        n_nodes, n_edges = self.layer._n_nodes, len(self.layer.cols)
        if phi_prev.shape[-1] != n_nodes or q_prev.shape[-1] != n_edges:
            raise ValueError(
                f"PotentialFlowLayer {name!r}: history=(phi, q) must hold every node's "
                f"potential ({n_nodes}) and every edge's flow ({n_edges}); got last "
                f"dimensions {phi_prev.shape[-1]} and {q_prev.shape[-1]}"
            )
        return self._with_history(ctx, phi_prev, q_prev)

    def _with_history(self, ctx: LawContext, phi_prev: Tensor, q_prev: Tensor) -> LawContext:
        """`ctx` with the step-start values taken from `(phi_prev, q_prev)`."""
        return dataclasses.replace(
            ctx, dp_prev=self.layer._difference(phi_prev), q_prev=q_prev)

    @staticmethod
    def _law_ctx(ctx: LawContext, s: int, e: int) -> LawContext:
        """`ctx` with the step-start values restricted to one law's edges `[s, e)`."""
        changes = {}
        if ctx.dp_prev is not None:
            changes["dp_prev"] = ctx.dp_prev[..., s:e]
        if ctx.q_prev is not None:
            changes["q_prev"] = ctx.q_prev[..., s:e]
        return dataclasses.replace(ctx, **changes) if changes else ctx

    def _owners(self) -> list:
        return [*self.layer._elements, *self.layer._node_sources]

    def _plain_call(self, i: int, args: tuple):
        """Law (or, past the laws, node source) `i` with its own parameters."""
        return self._owners()[i](*args)

    def _functional_call(self, rebuilt: list[dict]):
        """A `call` substituting the threaded parameters `rebuilt[i]` into owner `i`."""
        owners = self._owners()

        def call(i: int, args: tuple):
            return functional_call(owners[i], rebuilt[i], args)

        return call

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
                zip(layer._elements, layer._elem_slices, strict=True)):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            if el.law_form == "implicit":
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

    def _withdrawal(self, phi, drivers, call) -> Tensor | None:
        """The node sources' withdrawal per interior row, or `None` without node sources."""
        n_el = len(self.layer._elements)
        return self.layer._scatter_node_sources(
            lambda j, ns: call(n_el + j, (phi[..., ns.nodes], drivers)))

    def _node_slopes(self, phi, drivers, call) -> Tensor | None:
        """`d w / d phi` per interior row (a node source reads its own node's potential
        only, so one forward-mode derivative gives the whole diagonal)."""
        n_el = len(self.layer._elements)

        def slope(j, ns):
            phi_n = phi[..., ns.nodes]
            _, d = jvp(lambda v: call(n_el + j, (v, drivers)), (phi_n,),
                       (torch.ones_like(phi_n),))
            return d

        return self.layer._scatter_node_sources(slope)

    # ------------------------------------------------------------- residual and Jacobian
    def residual(self, x, phi_boundary, drivers, sources, ctx, call) -> Tensor:
        """KCL at the interior nodes, then the implicit laws' residuals (real view)."""
        layer = self.layer
        phi_I, q_F = self._unknowns(x)
        phi = layer.assemble(phi_I, phi_boundary)
        q, r_F = self._flows_and_residuals(phi, q_F, drivers, ctx, call)
        kcl = layer._accumulate_interior(q) - layer._source_interior(sources, phi_I)
        w = self._withdrawal(phi, drivers, call)
        if w is not None:
            kcl = kcl + w
        return self._to_real(_cat([kcl, r_F], kcl))

    @staticmethod
    def _potential_block(call, i, drivers, lctx, dp_s, m, w):
        return _colour_jacobian(lambda d: call(i, (d, drivers, lctx)), dp_s, m, w)

    @staticmethod
    def _implicit_blocks(call, i, drivers, lctx, dp_s, q_s, m, w):
        return (
            _colour_jacobian(lambda d: call(i, (d, q_s, drivers, lctx)), dp_s, m, w),
            _colour_jacobian(lambda qq: call(i, (dp_s, qq, drivers, lctx)), q_s, m, w),
        )

    def operator(self, x, phi_boundary, drivers, ctx, call, node_slope_fn) -> MnaOperator:
        """The Jacobian of `residual` at `x`, as COO triplets on the fixed pattern."""
        layer = self.layer
        w = self.w
        phi_I, q_F = self._unknowns(x)
        phi = layer.assemble(phi_I, phi_boundary)
        dp = layer.dp(phi, drivers)
        blocks = []
        for i, (el, (s, e)) in enumerate(
                zip(layer._elements, layer._elem_slices, strict=True)):
            lctx = self._law_ctx(ctx, s, e)
            dp_s = dp[..., s:e]
            if el.law_form == "implicit":
                f0 = int(self._f_index[s])
                q_s = q_F[..., f0:f0 + e - s]
                blocks += self._implicit_blocks(call, i, drivers, lctx, dp_s, q_s, el.ports, w)
            else:
                blocks.append(self._potential_block(call, i, drivers, lctx, dp_s, el.ports, w))
        parts = [blk.flatten(-5) for blk in blocks]
        parts.append(torch.ones(1, dtype=x.dtype, device=x.device))
        if w == 1 and layer._node_sources:
            parts.append(node_slope_fn(phi))
        flat = _cat(parts, x)
        values = flat[..., self.src] * self.sign.to(dtype=flat.dtype, device=flat.device)
        return MnaOperator(self.row, self.col, values, self.n)

    # ------------------------------------------------------------- solve
    def method(self, requested: str | None = None) -> str:
        """The inner linear solver for a request (default: the layer's `linear_solver`):
        the request itself unless `"auto"`, which factorises: SciPy's sparse LU per instance
        (`"sparse_direct"`) whenever SciPy is importable, at any batch size, and batched
        dense LU (`"direct"`) only when it is not. The system is indefinite, so never CG.

        Sparse LU at every batch size: a dense batched factorisation costs `batch * n^2`
        memory and `batch * n^3` work, so a frequency sweep of a mid-sized circuit (64
        frequencies, 1604 unknowns) took 6.1 s and 1.3 GB dense against 0.48 s per
        instance with SuperLU. A tiny circuit at a huge batch may be faster dense; ask for
        `method="direct"` there."""
        requested = self.layer.linear_solver if requested is None else requested
        if requested != "auto":
            return requested
        return "sparse_direct" if _scipy_importable() else "direct"

    def _singular(self, op: MnaOperator | None, failing: int) -> bool:
        """True when the last Jacobian is (numerically) rank deficient in the flat batch
        instance `failing`. Only a system of at most `_RANK_TEST_MAX_UNKNOWNS` unknowns is
        tested (a dense rank test of one instance); a larger one is reported as not
        singular, so the caller re-raises the solver's own error."""
        if op is None or self.n == 0 or self.n > _RANK_TEST_MAX_UNKNOWNS:
            return False
        values = op.values.reshape(-1, op.values.shape[-1])
        A = MnaOperator(op.row, op.col, values[min(failing, len(values) - 1)], self.n)
        A = A.assemble()
        if not bool(torch.isfinite(A).all()):
            return False
        return bool(torch.linalg.matrix_rank(A) < self.n)

    @staticmethod
    def _first_failing(last: dict, newton_kwargs: dict) -> int:
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

    def _singular_error(self, mode: str) -> RuntimeError:
        """The error for a singular system; at the initial point it also says that storage
        held at its initial value is itself a source, and where else to start."""
        initial = (
            "; at the initial point (mode='initial') every dynamic law is held at its initial "
            "value, so a capacitor acts as a voltage source at its v0 and an inductor as a "
            "current source at its i0: a capacitor across a voltage source, or an inductor "
            "in series with another or with a current source, makes it singular. Start the "
            "transient from the operating point instead (mode='dc'; operating_point in the "
            "electrical application)"
        ) if mode == "initial" else ""
        return RuntimeError(
            f"PotentialFlowLayer {self.layer.name!r}: the Modified Nodal Analysis system is "
            f"singular; check for a loop of ideal potential sources or a cut set of flow "
            f"sources among kinds {self.implicit_kinds}, or a part connected to a boundary "
            f"only through flow-prescribed edges{initial}"
        )

    def solve(self, phi_boundary, drivers, sources, phi0, *, q0, history, dt, t, integration,
              mode, differentiable, diagnostics, **newton_kwargs):
        layer = self.layer
        drivers = drivers or {}
        if self.w == 2:
            # Real or complex inputs; the phasor layer works in complex128.
            phi_boundary, sources, phi0, q0 = (
                None if v is None else torch.as_tensor(v).to(torch.complex128)
                for v in (phi_boundary, sources, phi0, q0))
            self._check_frequency(drivers, [
                v.shape[:-1] for v in (phi_boundary, sources, phi0, q0) if v is not None])
        ctx = self._context(mode, dt, t, integration, history, drivers)
        requested = newton_kwargs.pop("method", layer.linear_solver)
        if requested == "cg":
            raise ValueError(
                f"PotentialFlowLayer {layer.name!r}: method='cg' needs a symmetric positive "
                f"definite Jacobian, which a Modified Nodal Analysis system is not; use "
                f"'auto', 'sparse_direct', 'direct' or 'gmres'"
            )
        if newton_kwargs.get("on_failure") == "return" and diagnostics is None:
            raise ValueError(
                f"PotentialFlowLayer {layer.name!r}: on_failure='return' requires "
                f"diagnostics= so the status is not silently dropped; pass a dict and read "
                f"its 'converged' and 'residual_norm' entries."
            )
        plain = self._plain_call
        real_dtype = phi_boundary.real.dtype if phi_boundary.is_complex() else phi_boundary.dtype

        # The initial guess: zeros, or the caller's warm start. It is not differentiable
        # (the adjoint linearises at the converged point), so it is built under no_grad.
        with torch.no_grad():
            probe = self.residual(
                torch.zeros(self.n, dtype=real_dtype, device=phi_boundary.device),
                phi_boundary, drivers, sources, ctx, plain)
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

        newton_kwargs["method"] = self.method(requested)
        newton_kwargs.setdefault("where", f"PotentialFlowLayer {layer.name!r} solve")
        # The last Jacobian and the first and last residuals, kept to diagnose a failure.
        last: dict = {}

        def remember(r):
            last.setdefault("r0", r.detach())
            last["r"] = r.detach()
            return r

        def diagnose(exc):
            if self._singular(last.get("op"), self._first_failing(last, newton_kwargs)):
                raise self._singular_error(ctx.mode) from exc

        if not differentiable:
            def residual_fn(x):
                return remember(self.residual(x, phi_boundary, drivers, sources, ctx, plain))

            def slopes_fn(phi):
                return self._node_slopes(phi, drivers, plain)

            def operator_fn(x):
                last["op"] = self.operator(x, phi_boundary, drivers, ctx, plain, slopes_fn)
                return last["op"]

            with torch.no_grad():
                try:
                    result = newton(residual_fn, operator_fn, x0, **newton_kwargs)
                except RuntimeError as exc:
                    diagnose(exc)
                    raise
                if diagnostics is not None:
                    diagnostics["newton_iterations"] = result.iterations
                    diagnostics["linear_iterations"] = result.linear_iterations
                    diagnostics["method"] = requested
                    diagnostics["backend"] = result.backend
                    diagnostics["converged"] = result.converged
                    diagnostics["residual_norm"] = result.residual_norm
                return self._output(result.x, phi_boundary, drivers, ctx)

        layer._check_no_unreachable_differentiable_tensors()
        # Threaded through `Function.apply` exactly as on the classic path: every law's and
        # node source's own parameters (substituted back in with `functional_call`), every
        # driver, the sources (zero-filled when absent) and the boundary potentials; on a
        # transient step also the step-start state, so gradients flow back through time.
        owners = self._owners()
        param_dicts = [dict(owner.named_parameters()) for owner in owners]
        param_names = [list(d.keys()) for d in param_dicts]
        param_tensors = [
            d[name] for d, names in zip(param_dicts, param_names, strict=True) for name in names
        ]
        driver_keys = sorted(drivers.keys())
        driver_tensors = bulk_values(drivers, driver_keys)
        sources_tensor = (
            sources
            if sources is not None
            else torch.zeros(
                phi_boundary.shape[:-1] + (layer._n_nodes,),
                dtype=phi_boundary.dtype,
                device=phi_boundary.device,
            )
        )
        history_tensors = tuple(history) if ctx.dp_prev is not None else ()
        all_params = (*param_tensors, *driver_tensors, sources_tensor, phi_boundary,
                      *history_tensors)

        def _rebuild(params):
            offset = 0
            rebuilt = []
            for names in param_names:
                rebuilt.append({name: params[offset + j] for j, name in enumerate(names)})
                offset += len(names)
            drv = same_tracking(
                drivers,
                dict(zip(driver_keys, params[offset: offset + len(driver_keys)], strict=True)),
            )
            offset += len(driver_keys)
            step = (self._with_history(ctx, params[offset + 2], params[offset + 3])
                    if history_tensors else ctx)
            if self.w == 2:
                # the angular frequency from the threaded frequency driver
                step = dataclasses.replace(step, omega=self._omega(drv))
            return (self._functional_call(rebuilt), drv, params[offset], params[offset + 1],
                    step)

        def residual_fn(x, *params):
            call, drv, src, pb, step = _rebuild(params)
            return remember(self.residual(x, pb, drv, src, step, call))

        def operator_fn(x, *params):
            call, drv, src, pb, step = _rebuild(params)
            last["op"] = self.operator(
                x, pb, drv, step, call, lambda phi: self._node_slopes(phi, drv, call))
            return last["op"]

        try:
            x = implicit_solve(residual_fn, operator_fn, x0, all_params,
                               diagnostics=diagnostics, **newton_kwargs)
        except RuntimeError as exc:
            diagnose(exc)
            raise
        if diagnostics is not None:
            diagnostics["method"] = requested
        return self._output(x, phi_boundary, drivers, ctx)

    def _output(self, x, phi_boundary, drivers, ctx) -> tuple[Tensor, Tensor]:
        """`(phi, q)` at the solution `x`: full-node potentials and every edge's flow. The
        laws are called on the layer's own modules, so on the differentiable path `q` keeps
        the parameters' graph as the classic path's final `flows` does."""
        phi_I, q_F = self._unknowns(x)
        phi = self.layer.assemble(phi_I, phi_boundary)
        q, _ = self._flows_and_residuals(phi, q_F, drivers, ctx, self._plain_call)
        return phi, q
