"""PotentialFlowLayer: nodal conservation residual, Jacobian and Newton solve.

Assembles r(phi_I) = A_I g(A^T phi + drive; theta) - s_I from a set of Element laws
on typed edges (one Element per kind) and Drive terms (additive potential differences).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tellegen.drives import Drive
from tellegen.elements.base import Element
from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.grounding import spd_certificate, spd_diagnosis
from tellegen.solvers.implicit import adjoint as _adjoint_solve
from tellegen.solvers.implicit import implicit_solve
from tellegen.solvers.newton import newton
from tellegen.solvers.select import solve as select_solve
from tellegen.topology import Network

# The relative residual this layer asks its inner linear solves for, matching
# `solvers.select.solve`'s own default, and the ULP multiple below which no dtype can
# deliver it. See `_linear_rtol`.
_LINEAR_RTOL = 1e-10
_LINEAR_RTOL_ULPS = 32


def _linear_rtol(dtype: torch.dtype) -> float:
    """Relative residual to ask an inner linear solve for, floored by the working dtype.

    A Krylov solver's achievable relative residual is bounded below by the rounding error it
    accumulates, a small multiple of `finfo(dtype).eps`; asking for less does not make the
    answer better, it just spends every remaining iteration and then reports MAX_ITER on a
    solve that is in fact as converged as the dtype allows. In float64 the pinned 1e-10 is
    comfortably above that floor and is used unchanged; in float32 (this project's declared
    default dtype) eps is 1.2e-7, so 1e-10 is unreachable by several orders of magnitude --
    measured: the float32 CONTAM series case in tests/verification floors at 3.2e-8 and was
    reported as a linear_init failure until this floor was applied. This is the same
    "the dtype cannot be asked for precision it does not have" argument `newton`'s own
    dtype-derived atol/rtol default makes, applied to the linear solve instead of to the
    Newton convergence test.
    """
    return max(_LINEAR_RTOL, _LINEAR_RTOL_ULPS * float(torch.finfo(dtype).eps))


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
        # net.difference() (== net.incidence().T, the "source minus target" convention this
        # whole solve path uses -- see topology.py's module docstring) restricted to this
        # layer's own edge columns; equal to self.A.T, computed via the named operator rather
        # than repeating the einsum/transpose inline in dp().
        self._diff = net.difference()[self.cols]

        # This layer's own edge endpoints (restricted to self.cols, in the same order as
        # self._diff / self.A's columns), and a node -> interior-position map, both needed to
        # construct a GraphLaplacianOperator (and to run the per-instance SPD certificate)
        # without a per-solve Python loop. net.endpoints() (kind=None) returns whole-graph
        # (src, tgt) arrays in network edge order; indexing by self.cols restricts them to
        # this layer's own edges, exactly as self.A = net.incidence()[:, self.cols] already
        # does for the incidence matrix.
        src_all, tgt_all = net.endpoints()
        self._src = src_all[self.cols]
        self._tgt = tgt_all[self.cols]

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

        # interior_of_node: -1 at a boundary node's position, else its 0-based position
        # within self.interior. boundary_mask: True at a boundary node's position. Both are
        # (n,) and consumed by GraphLaplacianOperator's constructor and by
        # solvers.grounding; computing them once here, at construction time, avoids
        # rebuilding them on every solve() call.
        self._interior_of_node = torch.full((net.n,), -1, dtype=torch.long)
        self._interior_of_node[self.interior] = torch.arange(
            len(self.interior), dtype=torch.long
        )
        self._boundary_mask = torch.zeros(net.n, dtype=torch.bool)
        self._boundary_mask[self.bound] = True

    # ------------------------------------------------------------------ assembly
    def dp(self, phi: torch.Tensor, drivers: Mapping) -> torch.Tensor:
        drivers = drivers or {}
        d = torch.einsum("en,...n->...e", self._diff, phi)
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

    def _grounding_check(self, slopes: torch.Tensor, *, where: str) -> None:
        """Raise unless every instance in `slopes` certifies SPD grounding.

        The certificate (Task 3's `solvers.grounding.spd_certificate`) is run on the slopes
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
        alone is not attributable across unrelated per-instance failures.
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
            raise RuntimeError(f"{where}: {self._describe_grounding(records[0])}")
        bad_idx = torch.nonzero(~certified.reshape(-1), as_tuple=False).flatten().tolist()
        details = "; ".join(
            f"instance {rec['instance']}: {self._describe_grounding(rec)}" for rec in records
        )
        raise RuntimeError(
            f"{where}: batch indices {bad_idx} do not certify a grounded, positive-slope "
            f"system; {details}"
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
        A_I = self.A[self.interior]
        rhs = self._source_interior(sources, phi0) - torch.einsum(
            "ie,...e->...i", A_I, c + k * dp0
        )
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
        op = GraphLaplacianOperator(
            self._src,
            self._tgt,
            k,
            len(self.interior),
            self._interior_of_node,
            boundary_mask=self._boundary_mask,
        )
        result = select_solve(
            op, rhs, method="auto", where="linear_init", rtol=_linear_rtol(rhs.dtype)
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
        (`Element._param`'s documented pass-through exception, which is correct and used
        deliberately for `differentiable=False`); for a Drive, this happens whenever a Drive
        implementation owns a learnable coefficient directly instead of reading it from the
        `drivers` mapping passed to `solve()`. Raise now, before dispatching, rather than
        return a gradient that is silently wrong (if some other path happens to also touch
        the same value) or silently `None` (if it doesn't).
        """
        for el in self._elements:
            for name, value in vars(el).items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raise ValueError(
                        f"element {el!r} (kind {el.kind!r}) has a tensor attribute "
                        f"{name!r} with requires_grad=True that is not a registered "
                        f"nn.Parameter, so the differentiable solve cannot reach it through "
                        f"Function.apply and its gradient would be silently wrong or "
                        f"absent. Construct this element with learnable=True (so {name!r} "
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
          `named_parameters()` (the latter is a supported, correct construction for
          `differentiable=False`, per `Element._param`, but is invisible to the
          differentiable solve).
        - A Drive must read every differentiable quantity from the `drivers` mapping passed
          to `solve()` (see `tellegen.drives.Drive`), never hold one of its own as an
          instance attribute.

        Violating either raises `ValueError` naming the offending element/drive and
        attribute before any solve is attempted, rather than silently returning a wrong or
        absent gradient.
        """
        drivers = drivers or {}
        if phi0 is None:
            phi0 = self.linear_init(phi_boundary, drivers, sources)

        if not differentiable:

            def residual_fn(x):
                return self.residual(x, phi_boundary, drivers, sources)

            def jacobian_fn(x):
                return self.jacobian(x, phi_boundary, drivers)

            result = newton(residual_fn, jacobian_fn, phi0, **newton_kwargs)
            phi = self.assemble(result.x, phi_boundary)
            q = self.flows(phi, drivers)
            return phi, q

        self._check_no_unreachable_differentiable_tensors()

        param_dicts = [dict(el.named_parameters()) for el in self._elements]
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
                phi_boundary.shape[:-1] + (self.A.shape[0],),
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
            d = torch.einsum("en,...n->...e", self._diff, phi)
            parts = []
            for kind, (start, end) in self._kind_slices.items():
                block = d[..., start:end]
                for driven in self._drives:
                    if driven.kind == kind:
                        block = block + driven(phi, drv)
                parts.append((start, block))
            parts.sort(key=lambda p: p[0])
            return torch.cat([p[1] for p in parts], dim=-1)

        def _flows_functional(phi, drv, rebuilt):
            dp_full = _dp_functional(phi, drv)
            parts = []
            for el, d, (s, e) in zip(self._elements, rebuilt, self._elem_slices, strict=True):
                parts.append(torch.func.functional_call(el, d, (dp_full[..., s:e], drv)))
            return torch.cat(parts, dim=-1)

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
                zip(self._elements, rebuilt, self._elem_slices, strict=True)
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
            A_I = self.A[self.interior]
            lhs = torch.einsum("ie,...e->...i", A_I, q)
            s_I = src[..., self.interior]
            return lhs - s_I

        def jacobian_fn(x, *params):
            rebuilt, drv, src, pb = _rebuild(params)
            phi = self.assemble(x, pb)
            dq = _dflows_functional(phi, drv, rebuilt)
            A_I = self.A[self.interior]
            return torch.einsum("ie,...e,je->...ij", A_I, dq, A_I)

        x = implicit_solve(residual_fn, jacobian_fn, phi0, all_params, **newton_kwargs)
        phi = self.assemble(x, phi_boundary)
        q = self.flows(phi, drivers)
        return phi, q

    def adjoint(self, phi_interior, phi_boundary, drivers, grad_phi_interior):
        drivers = drivers or {}
        J = self.jacobian(phi_interior, phi_boundary, drivers)
        return _adjoint_solve(J, grad_phi_interior)

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
        drive_only = d - torch.einsum("en,...n->...e", self._diff, phi)
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
