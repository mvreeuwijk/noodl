"""The functional (parameters-as-arguments) view of a `KirchhoffLayer`'s laws.

`implicit_solve` differentiates a solve with respect to every tensor passed to
`Function.apply`. `FunctionalLaws` therefore presents the layer's residual and Jacobian
operator as functions of the iterate and a flat tuple of tensors: the laws' and withdrawals'
own `nn.Parameter`s (substituted back in with `torch.func.functional_call`), the driver
values, `sources` and `phi_boundary`. Withdrawals and node storages are threaded through
exactly as laws are, so a learnable demand exponent, required pressure or capacity gets a
correct implicit-function gradient. On a step the step-start state is threaded too: the
potentials (read by the storage and by the laws that read the step-start potential
difference), the flows (where a law reads them) and the flows into storage, so a chain of
steps back-propagates through time. It serves both paths: the fast path's residual and
operator are its own `residual_fn` and `operator_fn`; a layer with flow unknowns builds its
own from `rebuild` and `call`.

The Jacobian operator (the forward Newton's and the adjoint's) takes each law's and node
source's own analytic `dflow`, and each storage's `rate_slope`, where it declares
`exact_slope` (the default), and autograd through its flow (or rate) where it does not
(`FunctionalLaws._law_slope`, `_withdrawal_slope`, `_storage_slope`). The parameter gradients
come from one autograd pass through the residual at the root (`solvers.implicit`), never
from the slopes.
"""

from __future__ import annotations

import dataclasses

import torch

from noodl._tracking import bulk_values, same_tracking
from noodl.elements.base import ConstitutiveLaw
from noodl.layers._flow_unknowns import _Held, angular_frequency
from noodl.withdrawals import NodeWithdrawal


def _call(module, params: dict, args: tuple, kwargs: dict | None = None):
    """`module(*args, **kwargs)` with its parameters replaced by `params` (`functional_call`);
    a module without parameters is called directly, which is the same call without the
    cost."""
    if not params:
        return module(*args) if kwargs is None else module(*args, **kwargs)
    return torch.func.functional_call(module, params, args, kwargs)


def _stepping(step) -> bool:
    """Whether `step` is a step of a layer with storage (its `storage` a `_StorageStep`)."""
    return step is not None and step.storage is not None and not isinstance(step.storage, _Held)


def step_start(step, history=None) -> tuple:
    """The step-start tensors a differentiable solve threads, in order: the potentials
    `phi_prev` (on a step of a layer with storage, or where a law reads the step-start
    potential difference), the flows `q_prev` (where a law reads them) and the flows into
    storage `previous` (where given). Empty off a step. `history` is the solve's
    `(phi, q)`; the storage step holds the same `phi`."""
    if step is None:
        return ()
    ctx = step.ctx
    stepping = _stepping(step)
    if not stepping and ctx.dp_prev is None:
        return ()
    out = (step.storage.phi_prev if stepping else history[0],)
    if ctx.q_prev is not None:
        out += (ctx.q_prev,)
    if stepping and step.storage.previous is not None:
        out += (step.storage.previous,)
    return out


def _analytic(owner, base) -> bool:
    """Whether `owner` overrides `base.dflow` (else its `dflow` is autograd anyway, and the
    threaded autograd here gives the same slope with this module's own error messages)."""
    return type(owner).dflow is not base.dflow


class FunctionalLaws:
    """`residual_fn(x, *params)` and `operator_fn(x, *params)` of one layer at one call.

    `all_params` is the tuple to hand to `implicit_solve`, in the order `(parameters,
    drivers, sources, phi_boundary, *step_start(step, history))`; `params` in the two
    functions is that tuple. `last` is `(x, phi, q)` of the last residual evaluated without
    a graph (`root_parts`); a residual a layer with flow unknowns builds sets it too.
    """

    def __init__(self, layer, drivers, sources, phi_boundary, step=None, history=None) -> None:
        self.layer = layer
        self.drivers = drivers
        self.step = step
        self._owners = owners = (*layer._laws, *layer._withdrawals, *layer._storage)
        self.n_el = len(layer._laws)
        self.n_owned = self.n_el + len(layer._withdrawals)
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
        start = step_start(step, history)
        self._starts = len(start)
        self._phasor = layer.phasor
        self.all_params = (*param_tensors, *driver_tensors, sources_tensor, phi_boundary,
                           *start)
        self.last = None
        self._rebuilt: list = []

    def rebuild(self, params):
        """`(law, withdrawal and storage parameter dicts, drivers, sources, phi_boundary,
        step)` from the flat `params`; `step` is the layer's step rebuilt from `params`: its
        storage step's `phi_prev` and `previous`, its context's `dp_prev` and `q_prev` where
        the laws read them, and on a phasor layer its `omega` from the frequency driver
        (`None` off a step). The last two results
        are reused while `params` holds the same objects: the forward Newton alternates two
        tuples (the residual's detached one and the operator's). Everything that depends on
        `params` is in the returned value, never in a side effect, so a memo hit cannot hand
        back another call's step tensors."""
        for key, out in self._rebuilt:
            if len(key) == len(params) and all(
                    a is b for a, b in zip(key, params, strict=True)):
                return out
        out = self._rebuild(params)
        self._rebuilt = [(tuple(params), out), *self._rebuilt[:1]]
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
        step = self.step
        if self._starts:
            # the step-start tensors, threaded, in place of the originals
            ctx, store = step.ctx, step.storage
            k = offset + 2
            phi_prev = params[k]
            q_prev = None
            if ctx.q_prev is not None:
                k += 1
                q_prev = params[k]
            if ctx.dp_prev is not None:
                ctx = self.layer._with_history(ctx, phi_prev, q_prev)
            if _stepping(step):
                store = dataclasses.replace(
                    store, phi_prev=phi_prev,
                    previous=None if store.previous is None else params[k + 1])
            step = dataclasses.replace(step, ctx=ctx, storage=store)
        if self._phasor and step is not None:
            # the angular frequency from the threaded frequency driver
            step = dataclasses.replace(step, ctx=dataclasses.replace(
                step.ctx, omega=angular_frequency(self.layer, drv)))
        return rebuilt, drv, src, pb, step

    def call(self, rebuilt, i: int, args: tuple, kwargs: dict | None = None):
        """Owner `i` (a law, then past the laws a withdrawal, then a storage) called with its
        threaded parameters `rebuilt[i]` (from `rebuild`); one without parameters is called
        directly."""
        return _call(self._owners[i], rebuilt[i], args, kwargs)

    def root_parts(self, x):
        """`(phi, q)` at the solve's root `x` as Newton's last residual computed them, when
        no gradient is wanted (the same values `assemble` and the flows give); `None` when
        one is, and the caller computes them with their graph.

        After `KirchhoffLayer._newton_differentiable` the last residual is at the root: it
        is Newton's last, or the threading check's at the root."""
        wanted = torch.is_grad_enabled() and (x.requires_grad or any(
            isinstance(t, torch.Tensor) and t.requires_grad for t in self.all_params))
        if wanted:
            return None
        return self.last[1], self.last[2]

    def dp(self, phi, drv):
        layer = self.layer
        d = layer._difference(phi)
        head = drv.get(layer.head_key)
        if head is not None:
            layer._check_head_width(head)
            d = d + head
        return d

    def flows(self, phi, drv, rebuilt):
        layer = self.layer
        dp_full = self.dp(phi, drv)
        parts = []
        for el, d, (s, e) in zip(
            layer._laws, rebuilt[: self.n_el], layer._law_slices, strict=True
        ):
            parts.append(_call(el, d, (dp_full[..., s:e], drv)))
        return torch.cat(parts, dim=-1)

    def node_withdrawal(self, phi, drv, rebuilt, step=None):
        """The withdrawals' withdrawal, plus the storages' rates on a step (`step`, the
        threaded one from `rebuild`), per interior row (or None)."""
        layer = self.layer
        w = layer._scatter_withdrawals(lambda j, ns: _call(
            ns, rebuilt[self.n_el + j], (phi[..., ns.nodes], drv)))
        if step is None or step.storage is None:
            return w
        return layer._add(w, layer._scatter_storage(lambda j, st: _call(
            st, rebuilt[self.n_owned + j],
            (*layer._storage_args(j, st, phi, step.storage), step.ctx, drv))))

    def node_slopes(self, phi, drv, rebuilt, step=None):
        """The withdrawals' slopes `diag(w')`, plus the storages' on a step (`step`, the
        threaded one from `rebuild`), scattered into the interior rows (or None)."""
        layer = self.layer
        d = layer._scatter_withdrawals(
            lambda j, ns: self._withdrawal_slope(ns, rebuilt[self.n_el + j], phi[..., ns.nodes],
                                             drv))
        if step is None or step.storage is None:
            return d
        return layer._add(d, layer._scatter_storage(
            lambda j, st: self._storage_slope(st, rebuilt[self.n_owned + j],
                                              layer._storage_args(j, st, phi, step.storage),
                                              step.ctx, drv)))

    @staticmethod
    def _withdrawal_slope(ns, d, phi_nodes, drv):
        """One withdrawal's `dw/dphi` at its nodes: its own `dflow` where that is exact
        (`exact_slope`), else autograd through its withdrawal."""
        if ns.exact_slope and _analytic(ns, NodeWithdrawal):
            # `d`'s tensors are `ns`'s own parameters (see `_law_slope`).
            with torch.no_grad():
                return ns.dflow(phi_nodes, drv).expand(phi_nodes.shape)
        leaf = phi_nodes.detach().requires_grad_(True)
        with torch.enable_grad():
            w = _call(ns, d, (leaf, drv))
            (grad,) = torch.autograd.grad(w.sum(), leaf)
        return grad

    @staticmethod
    def _storage_slope(st, d, args, ctx, drv):
        """One storage's `d rate/d phi` at its nodes, `args = (phi_nodes, phi_start,
        previous)`: its `rate_slope` where that is exact (`exact_slope`, the default), else
        autograd through its rate. `rate_slope` is taken for every exact storage, overridden
        or not: it is the slope the primal Newton uses (`KirchhoffLayer._withdrawal_slopes`),
        so a differentiable solve that nothing reaches stays the primal solve bit for bit."""
        phi_nodes, phi_start, previous = args
        if st.exact_slope:
            # `d`'s tensors are `st`'s own parameters (see `_law_slope`).
            with torch.no_grad():
                return st.rate_slope(phi_nodes, phi_start, previous, ctx, drv).expand(
                    phi_nodes.shape)
        leaf = phi_nodes.detach().requires_grad_(True)
        with torch.enable_grad():
            w = _call(st, d, (leaf, phi_start, previous, ctx, drv))
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
        rebuilt, drv, src, pb, step = self.rebuild(params)
        phi = layer.assemble(x, pb)
        q = self.flows(phi, drv, rebuilt)
        if not q.requires_grad:
            # the point, potentials and flows of the last graph-free residual
            # (`root_parts` returns them when no gradient is wanted)
            self.last = (x, phi, q)
        lhs = layer._accumulate_interior(q)
        s_I = src[..., layer.interior]
        w = self.node_withdrawal(phi, drv, rebuilt, step)
        return lhs - s_I if w is None else lhs - s_I + w

    @property
    def layer_slopes(self) -> bool:
        """Whether `dflows`/`node_slopes` equal the layer's own `dflows` and withdrawal and
        storage slopes (`KirchhoffLayer._withdrawal_slopes`): true unless a law or node
        source overrides `dflow` with a slope it does not declare exact
        (`exact_slope = False`), or a storage declares its `rate_slope` inexact; only the
        layer's slopes use those. Cached on the layer against its owners' `exact_slope`
        flags (which may be set per instance)."""
        layer = self.layer
        n_el, n_owned = self.n_el, self.n_owned
        owners = (*layer._laws, *layer._withdrawals, *layer._storage)
        flags = tuple(o.exact_slope for o in owners)
        cache = getattr(layer, "_layer_slopes", None)
        if cache is None or cache[0] != flags:
            value = not any(
                not o.exact_slope and (
                    i >= n_owned
                    or _analytic(o, ConstitutiveLaw if i < n_el else NodeWithdrawal))
                for i, o in enumerate(owners))
            cache = layer._layer_slopes = (flags, value)
        return cache[1]

    def operator_fn(self, x, *params):
        layer = self.layer
        rebuilt, drv, src, pb, step = self.rebuild(params)
        phi = layer.assemble(x, pb)
        dq = self.dflows(phi, drv, rebuilt)
        return layer._operator_at(dq, self.node_slopes(phi, drv, rebuilt, step))
