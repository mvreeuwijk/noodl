"""The functional (parameters-as-arguments) view of a `KirchhoffLayer`'s laws.

`implicit_solve` differentiates a solve with respect to every tensor passed to
`Function.apply`. `FunctionalLaws` therefore presents the layer's residual and Jacobian
operator as functions of the iterate and a flat tuple of tensors: the laws' and node sources'
own `nn.Parameter`s (substituted back in with `torch.func.functional_call`), the driver
values, `sources` and `phi_boundary`. Node sources are threaded through exactly as laws are,
so a learnable demand exponent or required pressure gets a correct implicit-function gradient.
"""

from __future__ import annotations

import torch

from noodl._tracking import bulk_values, same_tracking


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

    def rebuild(self, params):
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
            parts.append(torch.func.functional_call(el, d, (dp_full[..., s:e], drv)))
        return torch.cat(parts, dim=-1)

    def node_withdrawal(self, phi, drv, rebuilt):
        layer = self.layer
        return layer._scatter_node_sources(lambda j, ns: torch.func.functional_call(
            ns, rebuilt[self.n_el + j], (phi[..., ns.nodes], drv)))

    def node_slopes(self, phi, drv, rebuilt):
        layer = self.layer
        def slope(j, ns):
            leaf = phi[..., ns.nodes].detach().requires_grad_(True)
            with torch.enable_grad():
                w = torch.func.functional_call(ns, rebuilt[self.n_el + j], (leaf, drv))
                (grad,) = torch.autograd.grad(w.sum(), leaf)
            return grad

        return layer._scatter_node_sources(slope)

    @staticmethod
    def _dp_dependence_error(el, i: int) -> RuntimeError:
        return RuntimeError(
            f"law {i} of kind {el.kind!r} (type {type(el).__name__}) produced a "
            f"flow that does not depend on dp; if that is intended, set "
            f"dp_independent = True on this law's class, otherwise its flow() is "
            f"dropping the autograd graph (e.g. a stray dp.detach())."
        )

    def dflows(self, phi, drv, rebuilt):
        layer = self.layer
        dp_full = self.dp(phi, drv)
        parts = []
        for i, (el, d, (s, e)) in enumerate(
            zip(layer._laws, rebuilt[: self.n_el], layer._law_slices, strict=True)
        ):
            dp_slice = dp_full[..., s:e].detach().requires_grad_(True)
            # The REDUCTION `flow.sum()` must itself execute inside the enable_grad
            # block, not just the `functional_call` that produces `flow`: this whole
            # method runs under the outer no_grad of Newton's forward solve (which is
            # what keeps the forward's memory flat), and `.sum()` is an ordinary tensor op like
            # any other -- performed under the ambient grad mode at the point it actually
            # runs, regardless of whether its input (`flow`) already carries a grad_fn
            # from an earlier, enable_grad-wrapped computation. Writing
            # `torch.autograd.grad(flow.sum(), dp_slice)` with the `with
            # torch.enable_grad():` block closed before that line evaluates `flow.sum()`
            # under the outer no_grad, so the tensor actually handed to `autograd.grad` as
            # `outputs` has no grad_fn of its own -- even though `flow` printed `requires_grad=True`
            # right before the call. `torch.autograd.grad` itself does not consult the
            # ambient grad mode at its own call site (confirmed: calling it under no_grad
            # against an output already fully built under enable_grad, or with an explicit
            # `grad_outputs=` and no further reduction, both work); the reduction is what
            # must be inside the block.
            with torch.enable_grad():
                flow = torch.func.functional_call(el, d, (dp_slice, drv))
                # Whether a missing/zero Jacobian contribution here is legitimate is
                # decided ONLY by el.dp_independent -- a class-level DECLARATION (see
                # ConstitutiveLaw) -- never inferred from whatever autograd graph `flow` happens
                # to carry (e.g. `flow.requires_grad` with allow_unused=True): that would give
                # ANY law whose flow() accidentally lost the autograd graph (e.g. a stray
                # dp.detach() in a third-party subclass) an exact-zero Jacobian column
                # instead of an error. The forward solve would still look fine (Newton
                # converges on the exact residual regardless), but the adjoint gradient
                # computed from that Jacobian would be silently wrong.
                if el.dp_independent:
                    # Cross-check the declaration against the law's own analytic
                    # dflow(), which for a genuinely dp-independent law (FixedFlow's,
                    # say) must be identically zero. This is called on `el` directly
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
                            f"law {i} of kind {el.kind!r} (type "
                            f"{type(el).__name__}) declares dp_independent = True but "
                            f"its own dflow() is not identically zero; either the flag "
                            f"is wrong or dflow() is inconsistent with a dp-independent "
                            f"flow law."
                        )
                    grad = torch.zeros_like(dp_slice)
                elif not flow.requires_grad:
                    raise self._dp_dependence_error(el, i)
                else:
                    try:
                        (grad,) = torch.autograd.grad(
                            flow.sum(), dp_slice, create_graph=False
                        )
                    except RuntimeError as exc:
                        raise self._dp_dependence_error(el, i) from exc
            parts.append(grad)
        return torch.cat(parts, dim=-1)

    def residual_fn(self, x, *params):
        layer = self.layer
        rebuilt, drv, src, pb = self.rebuild(params)
        phi = layer.assemble(x, pb)
        q = self.flows(phi, drv, rebuilt)
        lhs = layer._accumulate_interior(q)
        s_I = src[..., layer.interior]
        w = self.node_withdrawal(phi, drv, rebuilt)
        return lhs - s_I if w is None else lhs - s_I + w

    def operator_fn(self, x, *params):
        layer = self.layer
        rebuilt, drv, src, pb = self.rebuild(params)
        phi = layer.assemble(x, pb)
        dq = self.dflows(phi, drv, rebuilt)
        return layer._operator_at(dq, self.node_slopes(phi, drv, rebuilt))
