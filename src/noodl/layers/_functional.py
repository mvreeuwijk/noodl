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
steps back-propagates through time. It serves both paths: `residual_fn` and `operator_fn`
are the layer's own `_residual` and `_operator`, each owner called through `call` with its
threaded parameters.

The Jacobian operator (the forward Newton's and the adjoint's) takes each law's and node
source's own analytic `dflow`, and each storage's `rate_slope`, where it declares
`exact_slope` (the default), and autograd through its flow (or rate) where it does not
(`KirchhoffLayer._law_slopes` and `_node_slopes` with `adjoint=True`). The parameter
gradients come from one autograd pass through the residual at the root (`solvers.implicit`),
never from the slopes.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

from noodl._tracking import bulk_values, same_tracking
from noodl.elements.base import ConstitutiveLaw, LawContext
from noodl.layers._flow_unknowns import _Held, angular_frequency
from noodl.withdrawals import NodeWithdrawal

#: The steady context, of a solve without a step (`KirchhoffLayer._DC`).
_DC = LawContext(mode="dc")


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
                store = self.layer._storage_step(
                    phi_prev, None if store.previous is None else params[k + 1])
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

    def residual_fn(self, x, *params):
        """The layer's residual (`KirchhoffLayer._residual`) with the threaded `params`."""
        rebuilt, drv, src, pb, step = self.rebuild(params)
        r, phi, q = self.layer._residual(x, pb, drv, src, step,
                                         functools.partial(self.call, rebuilt))
        if not q.requires_grad:
            # the point, potentials and flows of the last graph-free residual
            # (`root_parts` returns them when no gradient is wanted)
            self.last = (x, phi, q)
        return r

    def operator_fn(self, x, *params):
        """The layer's Jacobian operator (`KirchhoffLayer._operator`) with the threaded
        `params`. It also builds the adjoint (`adjoint=True`): every law, withdrawal and
        storage whose slope is not its own exact one gives autograd."""
        rebuilt, drv, _, pb, step = self.rebuild(params)
        return self.layer._operator(x, pb, drv, step, functools.partial(self.call, rebuilt),
                                    adjoint=True)

    @property
    def layer_slopes(self) -> bool:
        """Whether the threaded operator's slopes equal the layer's own (`layer_slopes`)."""
        return layer_slopes(self.layer)


def layer_slopes(layer) -> bool:
    """Whether the slopes of `layer`'s threaded operator (`adjoint=True`) equal its primal
    ones (`KirchhoffLayer._dflows`, `_node_slopes`): true unless a law or
    node source overrides `dflow` with a slope it does not declare exact (`exact_slope =
    False`), a law that reads the context declares its slope inexact, or a storage declares
    its `rate_slope` inexact; only the layer's slopes use those. Cached on the layer against
    its owners' `exact_slope` flags (which may be set per instance)."""
    n_el = len(layer._laws)
    n_owned = n_el + len(layer._withdrawals)
    owners = (*layer._laws, *layer._withdrawals, *layer._storage)
    flags = tuple(o.exact_slope for o in owners)
    cache = getattr(layer, "_layer_slopes", None)
    if cache is None or cache[0] != flags:
        value = not any(
            not o.exact_slope and (
                i >= n_owned
                or (i < n_el and o._takes_ctx)
                or _analytic(o, ConstitutiveLaw if i < n_el else NodeWithdrawal))
            for i, o in enumerate(owners))
        cache = layer._layer_slopes = (flags, value)
    return cache[1]
