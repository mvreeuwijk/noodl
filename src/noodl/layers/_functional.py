"""The functional (parameters-as-arguments) view of a `KirchhoffLayer`'s laws.

`implicit_solve` differentiates a solve with respect to every tensor passed to
`Function.apply`. `FunctionalLaws` therefore presents the layer's residual and Jacobian
operator as functions of the iterate and a flat tuple of tensors: the laws' and node sources'
own `nn.Parameter`s (substituted back in with `torch.func.functional_call`), the driver
values, `sources` and `phi_boundary`. Node sources are threaded through exactly as laws are,
so a learnable demand exponent or required pressure gets a correct implicit-function gradient.

The Jacobian operator (the forward Newton's and the adjoint's) takes each law's and node
source's own analytic `dflow` where it declares `exact_slope` (the default), and autograd
through its flow where it does not (`FunctionalLaws._law_slope`, `_source_slope`). The
parameter gradients come from one autograd pass through the residual at the root
(`solvers.implicit`), never from the slopes.
"""

from __future__ import annotations

import torch

from noodl._tracking import bulk_values, same_tracking
from noodl.elements.base import ConstitutiveLaw
from noodl.nodesources import NodeSource


def _call(module, params: dict, args: tuple):
    """`module(*args)` with its parameters replaced by `params` (`functional_call`); a module
    without parameters is called directly, which is the same call without the cost."""
    if not params:
        return module(*args)
    return torch.func.functional_call(module, params, args)


def _analytic(owner, base) -> bool:
    """Whether `owner` overrides `base.dflow` (else its `dflow` is autograd anyway, and the
    threaded autograd here gives the same slope with this module's own error messages)."""
    return type(owner).dflow is not base.dflow


class FunctionalLaws:
    """`residual_fn(x, *params)` and `operator_fn(x, *params)` of one layer at one call.

    `all_params` is the tuple to hand to `implicit_solve`; `params` in the two functions is
    that tuple.
    """

    def __init__(self, layer, drivers, sources, phi_boundary) -> None:
        self.layer = layer
        self.drivers = drivers
        owners = [*layer._laws, *layer._node_sources]
        self.n_el = len(layer._laws)
        param_dicts = [dict(owner.named_parameters()) for owner in owners]
        self.param_names = [list(d.keys()) for d in param_dicts]
        param_tensors = [
            d[name]
            for d, names in zip(param_dicts, self.param_names, strict=True)
            for name in names
        ]
        self.driver_keys = sorted(drivers.keys())
        driver_tensors = bulk_values(drivers, self.driver_keys)
        sources_tensor = (
            sources
            if sources is not None
            else torch.zeros(
                phi_boundary.shape[:-1] + (layer._n_nodes,),
                dtype=phi_boundary.dtype,
                device=phi_boundary.device,
            )
        )
        self.all_params = (*param_tensors, *driver_tensors, sources_tensor, phi_boundary)
        self.last = None
        self._rebuilt = None

    def rebuild(self, params):
        """`(law and node-source parameter dicts, drivers, sources, phi_boundary)` from the
        flat `params`; the last result is reused while `params` holds the same objects (the
        forward Newton calls with one tuple throughout)."""
        memo = self._rebuilt
        if memo is not None and len(memo[0]) == len(params) and all(
                a is b for a, b in zip(memo[0], params, strict=True)):
            return memo[1]
        out = self._rebuild(params)
        self._rebuilt = (tuple(params), out)
        return out

    def _rebuild(self, params):
        offset = 0
        rebuilt = []
        for names in self.param_names:
            d = {name: params[offset + j] for j, name in enumerate(names)}
            rebuilt.append(d)
            offset += len(names)
        drv = same_tracking(
            self.drivers,
            dict(
                zip(
                    self.driver_keys,
                    params[offset : offset + len(self.driver_keys)],
                    strict=True,
                )
            ),
        )
        offset += len(self.driver_keys)
        src = params[offset]
        pb = params[offset + 1]
        return rebuilt, drv, src, pb

    def dp(self, phi, drv):
        layer = self.layer
        d = layer._difference(phi)
        parts = []
        for kind, (start, end) in layer._kind_slices.items():
            block = d[..., start:end]
            for driven in layer._drives:
                if driven.kind == kind:
                    block = block + driven(drv)
            parts.append((start, block))
        parts.sort(key=lambda p: p[0])
        return torch.cat([p[1] for p in parts], dim=-1)

    def flows(self, phi, drv, rebuilt):
        layer = self.layer
        dp_full = self.dp(phi, drv)
        parts = []
        for el, d, (s, e) in zip(
            layer._laws, rebuilt[: self.n_el], layer._law_slices, strict=True
        ):
            parts.append(_call(el, d, (dp_full[..., s:e], drv)))
        return torch.cat(parts, dim=-1)

    def node_withdrawal(self, phi, drv, rebuilt):
        layer = self.layer
        return layer._scatter_node_sources(lambda j, ns: _call(
            ns, rebuilt[self.n_el + j], (phi[..., ns.nodes], drv)))

    def node_slopes(self, phi, drv, rebuilt):
        """The node sources' slopes `diag(w')`, scattered into the interior rows (or None)."""
        layer = self.layer
        return layer._scatter_node_sources(
            lambda j, ns: self._source_slope(ns, rebuilt[self.n_el + j], phi[..., ns.nodes],
                                             drv))

    @staticmethod
    def _source_slope(ns, d, phi_nodes, drv):
        """One node source's `dw/dphi` at its nodes: its own `dflow` where that is exact
        (`exact_slope`), else autograd through its withdrawal."""
        if ns.exact_slope and _analytic(ns, NodeSource):
            # `d`'s tensors are `ns`'s own parameters (see `_law_slope`).
            with torch.no_grad():
                return ns.dflow(phi_nodes, drv).expand(phi_nodes.shape)
        leaf = phi_nodes.detach().requires_grad_(True)
        with torch.enable_grad():
            w = _call(ns, d, (leaf, drv))
            (grad,) = torch.autograd.grad(w.sum(), leaf)
        return grad

    @staticmethod
    def _dp_dependence_error(el, i: int) -> RuntimeError:
        return RuntimeError(
            f"law {i} of kind {el.kind!r} (type {type(el).__name__}) produced a "
            f"flow that does not depend on dp; if that is intended, set "
            f"dp_independent = True on this law's class, otherwise its flow() is "
            f"dropping the autograd graph (e.g. a stray dp.detach())."
        )

    def dflows(self, phi, drv, rebuilt):
        """Every edge's slope `dq/d(dp)`, law by law (`_law_slope`), in edge order."""
        layer = self.layer
        dp_full = self.dp(phi, drv)
        return torch.cat([
            self._law_slope(i, el, d, dp_full[..., s:e], drv)
            for i, (el, d, (s, e)) in enumerate(
                zip(layer._laws, rebuilt[: self.n_el], layer._law_slices, strict=True))
        ], dim=-1)

    def _law_slope(self, i, el, d, dp, drv):
        """Law `i`'s slopes at its edges' `dp`: zero for a `dp_independent` law (checked
        against its `dflow`), its own `dflow` where that is the exact derivative of its flow
        (`exact_slope`), else autograd through its flow with the threaded parameters `d`.

        The adjoint linearises with these slopes, so only an exact `dflow` may stand in for
        autograd; a law whose `dflow` is a Newton aid declares `exact_slope = False`.
        `dflow` is called on `el` itself, not through `functional_call`: `d`'s tensors are
        the very same objects as `el`'s own parameters (both come from
        `el.named_parameters()` at the start of the solve), so the values are equal. Under
        `no_grad`: the slopes are constants of the operator, never differentiated."""
        if el.dp_independent:
            # Cross-check the declaration against the law's own analytic dflow(), which for
            # a genuinely dp-independent law (FixedFlow's, say) must be identically zero.
            # This catches a wrongly-declared dp_independent = True (or a dflow()
            # inconsistent with it) instead of silently trusting a possibly-wrong flag.
            with torch.no_grad():
                analytic = el.dflow(dp.detach(), drv)
            if not torch.equal(analytic, torch.zeros_like(analytic)):
                raise RuntimeError(
                    f"law {i} of kind {el.kind!r} (type "
                    f"{type(el).__name__}) declares dp_independent = True but "
                    f"its own dflow() is not identically zero; either the flag "
                    f"is wrong or dflow() is inconsistent with a dp-independent "
                    f"flow law."
                )
            return torch.zeros_like(dp)
        if el.exact_slope and _analytic(el, ConstitutiveLaw):
            with torch.no_grad():
                return el.dflow(dp.detach(), drv).expand(dp.shape)
        dp_slice = dp.detach().requires_grad_(True)
        # The REDUCTION `flow.sum()` must itself execute inside the enable_grad block, not
        # just the `functional_call` that produces `flow`: this method runs under the outer
        # no_grad of Newton's forward solve, and `.sum()` performed under no_grad hands
        # `autograd.grad` an output without a grad_fn even though `flow` carries one.
        with torch.enable_grad():
            flow = _call(el, d, (dp_slice, drv))
            # Whether a missing Jacobian contribution is legitimate is decided ONLY by
            # el.dp_independent, a class-level DECLARATION, never inferred from the graph
            # `flow` happens to carry: a flow() that accidentally lost the autograd graph
            # (a stray dp.detach()) would otherwise get an exact-zero Jacobian column and a
            # silently wrong adjoint gradient.
            if not flow.requires_grad:
                raise self._dp_dependence_error(el, i)
            try:
                (grad,) = torch.autograd.grad(flow.sum(), dp_slice, create_graph=False)
            except RuntimeError as exc:
                raise self._dp_dependence_error(el, i) from exc
        return grad

    def residual_fn(self, x, *params):
        layer = self.layer
        rebuilt, drv, src, pb = self.rebuild(params)
        phi = layer.assemble(x, pb)
        q = self.flows(phi, drv, rebuilt)
        if not q.requires_grad:
            # the point, potentials and flows of the last graph-free residual
            # (`KirchhoffLayer._solve_implicit` returns them when no gradient is wanted)
            self.last = (x, phi, q)
        lhs = layer._accumulate_interior(q)
        s_I = src[..., layer.interior]
        w = self.node_withdrawal(phi, drv, rebuilt)
        return lhs - s_I if w is None else lhs - s_I + w

    @property
    def layer_slopes(self) -> bool:
        """Whether `dflows`/`node_slopes` equal the layer's own `dflows` and node-source
        slopes: true unless a law or node source overrides `dflow` with a slope it does not
        declare exact (`exact_slope = False`), which only the layer's slopes use."""
        layer = self.layer
        return not any(
            not owner.exact_slope and _analytic(owner, base)
            for owners, base in ((layer._laws, ConstitutiveLaw),
                                 (layer._node_sources, NodeSource))
            for owner in owners)

    def operator_fn(self, x, *params):
        layer = self.layer
        rebuilt, drv, src, pb = self.rebuild(params)
        phi = layer.assemble(x, pb)
        dq = self.dflows(phi, drv, rebuilt)
        return layer._operator_at(dq, self.node_slopes(phi, drv, rebuilt))
