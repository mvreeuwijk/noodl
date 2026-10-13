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
from noodl.elements.base import LawContext
from noodl.operators.mna import MnaOperator
from noodl.solvers.implicit import implicit_solve
from noodl.solvers.newton import newton

Tensor = torch.Tensor

#: The flat batch above which `"auto"` factorises densely instead of per instance with
#: SuperLU (`solvers.select._SPARSE_DIRECT_MAX_BATCH`, measured for the nodal layer).
_SPARSE_DIRECT_MAX_BATCH = 32


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
            cols.append(torch.view_as_real(d) if w == 2 else d.unsqueeze(-1))
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
        self._block_sizes = []
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
                self._block_sizes.append(size)
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
                self._block_sizes += [size, size]
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
        # node sources (real layers only): the diagonal slope per interior row
        self._node_slot = offset
        if self.layer._node_sources:
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
        return torch.view_as_real(z).flatten(-2) if self.w == 2 else z

    def _to_complex(self, x: Tensor) -> Tensor:
        return torch.view_as_complex(x.unflatten(-1, (-1, 2)).contiguous())

    def _split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """`(phi_I, q_F)` of the unknown vector, in the complex view on a phasor layer."""
        return z[..., : self.n_I], z[..., self.n_I:]

    def _unknowns(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._split(self._to_complex(x) if self.w == 2 else x)

    # ------------------------------------------------------------- laws
    def _context(self, mode, dt, t, integration, history, drivers) -> LawContext:
        """The `LawContext` the laws are evaluated in: the steady operating point."""
        return LawContext(mode="dc", t=t)

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
                r_parts.append(call(i, (dp_s, q_s, drivers, lctx)))
            else:
                q_s = call(i, (dp_s, drivers, lctx))
            q_parts.append(q_s)
        return _cat(q_parts, dp), _cat(r_parts, dp)

    def _withdrawal(self, phi, drivers, call) -> Tensor | None:
        """The node sources' withdrawal per interior row, or `None` without node sources."""
        layer = self.layer
        n_el = len(layer._elements)
        out = None
        for j, (ns, pos) in enumerate(
                zip(layer._node_sources, layer._source_positions, strict=True)):
            w = call(n_el + j, (phi[..., ns.nodes], drivers))
            zeros = torch.zeros(w.shape[:-1] + (self.n_I,), dtype=w.dtype, device=w.device)
            term = zeros.index_add(-1, pos, w)
            out = term if out is None else out + term
        return out

    def _node_slopes(self, phi, drivers, call) -> Tensor | None:
        """`d w / d phi` per interior row (a node source reads its own node's potential
        only, so one forward-mode derivative gives the whole diagonal)."""
        layer = self.layer
        n_el = len(layer._elements)
        out = None
        for j, (ns, pos) in enumerate(
                zip(layer._node_sources, layer._source_positions, strict=True)):
            phi_n = phi[..., ns.nodes]
            _, d = jvp(self._node_source_fn(call, n_el + j, drivers), (phi_n,),
                       (torch.ones_like(phi_n),))
            zeros = torch.zeros(d.shape[:-1] + (self.n_I,), dtype=d.dtype, device=d.device)
            term = zeros.index_add(-1, pos, d)
            out = term if out is None else out + term
        return out

    @staticmethod
    def _node_source_fn(call, i, drivers):
        return lambda v: call(i, (v, drivers))

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
    def method(self, batch_numel: int) -> str:
        """The inner linear solver: the layer's own unless `"auto"`, which factorises
        (SuperLU per instance up to `_SPARSE_DIRECT_MAX_BATCH` instances when SciPy is
        importable, batched dense LU otherwise; the system is indefinite, so never CG)."""
        if self.layer.linear_solver != "auto":
            return self.layer.linear_solver
        if (importlib.util.find_spec("scipy") is not None
                and batch_numel <= _SPARSE_DIRECT_MAX_BATCH):
            return "sparse_direct"
        return "direct"

    def _singular(self, op: MnaOperator | None) -> bool:
        """True when the last Jacobian is (numerically) rank deficient in some instance."""
        if op is None or self.n == 0:
            return False
        A = op.assemble()
        if not bool(torch.isfinite(A).all()):
            return False
        return bool((torch.linalg.matrix_rank(A) < self.n).any())

    def _singular_error(self) -> RuntimeError:
        return RuntimeError(
            f"PotentialFlowLayer {self.layer.name!r}: the Modified Nodal Analysis system is "
            f"singular; check for a loop of ideal potential sources or a cut set of flow "
            f"sources among kinds {self.implicit_kinds}"
        )

    def solve(self, phi_boundary, drivers, sources, phi0, *, q0, history, dt, t, integration,
              mode, differentiable, diagnostics, **newton_kwargs):
        layer = self.layer
        drivers = drivers or {}
        ctx = self._context(mode, dt, t, integration, history, drivers)
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

        newton_kwargs.setdefault("method", self.method(math.prod(B)))
        newton_kwargs.setdefault("where", f"PotentialFlowLayer {layer.name!r} solve")
        last: dict = {}

        if not differentiable:
            def residual_fn(x):
                return self.residual(x, phi_boundary, drivers, sources, ctx, plain)

            def slopes_fn(phi):
                return self._node_slopes(phi, drivers, plain)

            def operator_fn(x):
                last["op"] = self.operator(x, phi_boundary, drivers, ctx, plain, slopes_fn)
                return last["op"]

            with torch.no_grad():
                try:
                    result = newton(residual_fn, operator_fn, x0, **newton_kwargs)
                except RuntimeError as exc:
                    if self._singular(last.get("op")):
                        raise self._singular_error() from exc
                    raise
                if diagnostics is not None:
                    diagnostics["newton_iterations"] = result.iterations
                    diagnostics["linear_iterations"] = result.linear_iterations
                    diagnostics["method"] = newton_kwargs["method"]
                    diagnostics["backend"] = result.backend
                    diagnostics["converged"] = result.converged
                    diagnostics["residual_norm"] = result.residual_norm
                return self._output(result.x, phi_boundary, drivers, ctx)

        layer._check_no_unreachable_differentiable_tensors()
        # Threaded through `Function.apply` exactly as on the classic path: every law's and
        # node source's own parameters (substituted back in with `functional_call`), every
        # driver, the sources (zero-filled when absent) and the boundary potentials.
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
        all_params = (*param_tensors, *driver_tensors, sources_tensor, phi_boundary)

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
            return self._functional_call(rebuilt), drv, params[offset], params[offset + 1]

        def residual_fn(x, *params):
            call, drv, src, pb = _rebuild(params)
            return self.residual(x, pb, drv, src, ctx, call)

        def operator_fn(x, *params):
            call, drv, src, pb = _rebuild(params)
            last["op"] = self.operator(
                x, pb, drv, ctx, call, lambda phi: self._node_slopes(phi, drv, call))
            return last["op"]

        try:
            x = implicit_solve(residual_fn, operator_fn, x0, all_params,
                               diagnostics=diagnostics, **newton_kwargs)
        except RuntimeError as exc:
            if self._singular(last.get("op")):
                raise self._singular_error() from exc
            raise
        if diagnostics is not None:
            diagnostics["method"] = newton_kwargs["method"]
        return self._output(x, phi_boundary, drivers, ctx)

    def _output(self, x, phi_boundary, drivers, ctx) -> tuple[Tensor, Tensor]:
        """`(phi, q)` at the solution `x`: full-node potentials and every edge's flow. The
        laws are called on the layer's own modules, so on the differentiable path `q` keeps
        the parameters' graph as the classic path's final `flows` does."""
        phi_I, q_F = self._unknowns(x)
        phi = self.layer.assemble(phi_I, phi_boundary)
        q, _ = self._flows_and_residuals(phi, q_F, drivers, ctx, self._plain_call)
        return phi, q
