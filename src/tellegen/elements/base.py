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
``torch.func.functional_call``, which Task 8's differentiable solve relies on). Delegating
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
    """

    kind: str

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
        ``torch.no_grad()`` (the batched Newton solve of Tasks 6/7); a graph is built under
        ordinary tracking, including inside ``gradcheck``, which differentiates this twice.
        """
        grad_enabled = torch.is_grad_enabled()
        x = dp.detach().clone()
        x.requires_grad_(True)
        with torch.enable_grad():
            q = self.flow(x, drivers)
            (grad,) = torch.autograd.grad(q.sum(), x, create_graph=grad_enabled)
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
