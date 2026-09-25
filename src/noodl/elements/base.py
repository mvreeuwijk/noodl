"""Branch constitutive laws: q = flow(dp, drivers) mapping potential difference to flow.

Every :class:`Element` is a ``torch.nn.Module`` so its numeric parameters register as
``nn.Parameter`` and participate in gradient-based calibration whenever ``learnable=True``.

Elementwise assumption: ``flow`` maps each entry of ``dp`` independently of every other
entry (branch e's flow depends only on branch e's own potential difference). This is what
makes the ``dflow`` default below correct: it sums ``flow(dp)`` before differentiating, and
because there is no cross-branch coupling, the gradient of the sum with respect to ``dp``
has, at each position, exactly that position's own partial derivative.

``forward`` is defined as a method that delegates to ``self.flow(...)`` rather than as a
bare ``forward = flow`` class-body assignment. The latter would bind the *base* class's
``flow`` function object at class-definition time, so subclasses overriding ``flow`` would
not be reachable through ``__call__``/``forward`` (and hence not through
``torch.func.functional_call``, which the differentiable solve relies on). Delegating
through ``self.flow`` re-resolves to the most-derived override on every call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

Tensor = torch.Tensor


class Element(torch.nn.Module):
    """Abstract branch law: flow as a function of potential difference.

    Subclasses implement :meth:`flow`. :meth:`dflow` and :meth:`linear_init` have working
    defaults built on :meth:`flow` and autograd; subclasses may override either with an
    analytic form for speed and exactness where autograd is ill-conditioned.

    ``dp_independent`` (default ``False``) is a class-level DECLARATION, not something
    inferred from whether ``flow`` happens to build an autograd graph back to ``dp``. Set it
    ``True`` only when ``flow`` is mathematically independent of ``dp`` (so ``dflow`` is
    identically zero everywhere) -- ``FixedFlow`` is the built-in example. The differentiable
    solve path (``PotentialFlowLayer.solve(differentiable=True)``, in
    ``noodl.layers.potential._dflows_functional``) uses this flag, and only this flag, to
    decide whether a missing gradient of ``flow`` with respect to ``dp`` is legitimate (this
    element declares it does not depend on ``dp``) or a bug (some other element's ``flow``
    silently dropped the autograd graph, e.g. via a stray ``dp.detach()``). Declaring instead
    of inferring keeps that distinction loud: a third-party ``Element`` subclass that
    accidentally detaches ``dp`` now raises a clear error instead of silently receiving a
    zero Jacobian column (which leaves the forward solve looking fine -- Newton still
    converges on the exact residual -- while the adjoint gradient computed from that Jacobian
    is silently wrong).
    """

    kind: str
    dp_independent: bool = False

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """Branch flow q(dp); elementwise in dp. Abstract in the base class."""
        raise NotImplementedError(f"{type(self).__name__} does not implement flow()")

    def forward(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """Alias for :meth:`flow` so ``Module.__call__``/``torch.func.functional_call`` work.

        Delegates to ``self.flow`` (rather than being assigned ``forward = flow`` in the
        class body) so that subclass overrides of ``flow`` remain reachable.
        """
        return self.flow(dp, drivers)

    def dflow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """d flow / d dp, elementwise, by autograd on ``flow(dp).sum()``.

        ``dp`` is detached and re-wrapped as a fresh leaf so this never perturbs the
        caller's own graph. ``create_graph`` follows the caller's ``torch.is_grad_enabled()``
        (captured before ``enable_grad`` is forced locally): no graph is built under
        ``torch.no_grad()`` (the batched Newton solve); a graph is built under
        ordinary tracking, including inside ``gradcheck``, which differentiates this twice.

        This default assumes ``flow`` actually depends on ``dp`` (true of every well-behaved
        subclass); a genuinely ``dp_independent`` element (``FixedFlow``) overrides both
        ``dflow`` and ``linear_init`` with its own analytic zero and never reaches this
        method. If ``flow`` does not depend on ``dp`` here -- either because it produces an
        output that does not require grad at all, or because ``dp`` is silently unused inside
        it (e.g. a stray ``dp.detach()``) -- that is the same gradient hazard the
        differentiable solve path guards against in
        ``noodl.layers.potential._dflows_functional``, reached here via
        ``PotentialFlowLayer.linear_init`` instead. Raise a clear error naming the element
        and its kind rather than letting a raw, unattributed autograd error escape.
        """
        grad_enabled = torch.is_grad_enabled()
        x = dp.detach().clone()
        x.requires_grad_(True)
        with torch.enable_grad():
            q = self.flow(x, drivers)
            if not q.requires_grad:
                raise RuntimeError(
                    f"{type(self).__name__} (kind {self.kind!r}) produced a flow that does "
                    f"not depend on dp (flow.requires_grad is False), so the default "
                    f"autograd-based dflow() cannot differentiate it. If this element's flow "
                    f"is genuinely independent of dp, override dflow() (and linear_init()) "
                    f"with the analytic zero, as FixedFlow does, and set "
                    f"dp_independent = True. Otherwise flow() is dropping the autograd "
                    f"graph (e.g. a stray dp.detach())."
                )
            try:
                (grad,) = torch.autograd.grad(q.sum(), x, create_graph=grad_enabled)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{type(self).__name__} (kind {self.kind!r}) produced a flow that does "
                    f"not use its dp input anywhere in the computation, so the default "
                    f"autograd-based dflow() cannot differentiate it. If this element's flow "
                    f"is genuinely independent of dp, override dflow() (and linear_init()) "
                    f"with the analytic zero, as FixedFlow does, and set "
                    f"dp_independent = True. Otherwise flow() is dropping the autograd "
                    f"graph (e.g. a stray dp.detach())."
                ) from exc
        return grad

    def linear_init(
        self, drivers: Mapping[str, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """(c, k): tangent flow ~ c + k * dp near dp = 0, default from flow/dflow at 0."""
        zero = torch.zeros((), dtype=self._dtype())
        return self.flow(zero, drivers), self.dflow(zero, drivers)

    @staticmethod
    def _param(value, learnable: bool) -> torch.nn.Parameter | Tensor:
        """Wrap ``value`` as a leaf ``nn.Parameter`` with ``requires_grad=learnable``.

        Exception: if ``learnable`` is False and ``value`` is already a tensor with
        ``requires_grad=True``, it is returned unchanged rather than wrapped.
        ``nn.Parameter`` always constructs a fresh, detached leaf (even when
        ``requires_grad=True`` is requested), so wrapping here would silently sever any
        existing autograd connection -- e.g. when a caller reconstructs an element inside a
        closure that ``torch.autograd.gradcheck``/an outer optimizer differentiates with
        respect to.

        This exception is gated on ``not learnable``: ``learnable=True`` always yields a
        real, module-registered ``nn.Parameter`` (so ``.parameters()``, ``state_dict()``,
        and ``torch.func.functional_call`` name-based substitution all see it), even if that
        means detaching from whatever graph the incoming tensor happened to carry -- a
        learnable element's whole point is to be *this* module's own optimizable leaf, not a
        transparent view onto an external computation.
        """
        if isinstance(value, torch.Tensor):
            if value.requires_grad and not learnable:
                return value
            tensor = value.clone()
        else:
            tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
        return torch.nn.Parameter(tensor, requires_grad=learnable)

    def _dtype(self) -> torch.dtype:
        for p in self.parameters():
            return p.dtype
        return torch.get_default_dtype()
