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
from dataclasses import dataclass

import torch

from noodl._broadcast import broadcast_shapes

Tensor = torch.Tensor

#: The modes a dynamic or implicit constitutive law is evaluated in (see `LawContext`).
MODES = ("dc", "initial", "transient", "phasor")
#: The implicit time integrations a dynamic law implements in the transient mode.
INTEGRATIONS = ("trapezoidal", "backward_euler")


@dataclass(frozen=True)
class LawContext:
    """What a dynamic or implicit constitutive law needs besides its own edge values.

    `mode` is one of `MODES`: `"dc"` the steady operating point (storage elements at rest:
    a capacitance carries no current, an inductance no voltage); `"initial"` the consistent
    initial point (a capacitance holds its initial voltage, an inductance its initial
    current); `"transient"` one implicit step of length `dt` ending at time `t`, from the
    step-start values `dp_prev` and `q_prev` of the law's own edges, with `integration` one
    of `INTEGRATIONS`; `"phasor"` the quasi-static sinusoidal state at angular frequency
    `omega` (a tensor broadcasting against the law's edges). `t` is the evaluation time
    (the END of a transient step), `None` when the caller tracks no clock.
    """

    mode: str
    dt: float | None = None
    t: float | None = None
    integration: str = "trapezoidal"
    dp_prev: Tensor | None = None
    q_prev: Tensor | None = None
    omega: Tensor | None = None


def memo(owner, name: str, deps: tuple, fn):
    """`fn()`, remembered on `owner` under `name` for as long as `deps` are the same objects
    (identity, and for tensors the same `_version`, so an in-place change is seen).

    For the parts of a law that depend only on its drivers and parameters (a door's inflow
    densities, a power law's coefficients): a Newton solve evaluates the law many times at
    fixed drivers, and small tensor operations cost more in dispatch than in arithmetic. The
    value is the one `fn()` returns, so the law's results are unchanged bit for bit. A value
    that carries an autograd graph (a learnable parameter, a driver that requires grad) is
    never remembered: it is recomputed on every call, as without the memo, and neither is a
    value that depends on a tensor that requires grad or does not count its versions (an
    inference-mode tensor)."""
    try:
        key = tuple((d, (d._version, d.requires_grad)) if isinstance(d, Tensor) else (d, None)
                    for d in deps)
    except RuntimeError:  # an inference tensor has no version counter
        return fn()
    if any(isinstance(d, Tensor) and d.requires_grad for d in deps):
        return fn()
    cache = owner.__dict__.setdefault("_memo_cache", {})
    hit = cache.get(name)
    if hit is not None and len(hit[0]) == len(key) and all(
            a[0] is b[0] and a[1] == b[1] for a, b in zip(hit[0], key, strict=True)):
        return hit[1]
    value = fn()
    parts = value if isinstance(value, tuple) else (value,)
    if not any(isinstance(v, Tensor) and v.requires_grad for v in parts):
        cache[name] = (key, value)
    else:
        cache.pop(name, None)
    return value


def band_edges(x: Tensor, width) -> Tensor:
    """``((x - w)/w, (x + w)/w)`` side by side on the last dimension: the two edges
    ``x = -w`` and ``x = +w`` of a regularisation band ``|x| < w`` as switching values
    (:meth:`Element.switching`), in units of the band's width."""
    w = torch.as_tensor(width, dtype=x.dtype)
    shape = broadcast_shapes(x.shape, w.shape)
    return torch.cat([torch.broadcast_to((x - w) / w, shape),
                      torch.broadcast_to((x + w) / w, shape)], dim=-1)


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
    #: "potential": flow is a function of the potential difference, `q = flow(dp)`.
    #: "implicit": the law is a relation `residual(dp, q) = 0` and the flow is an unknown
    #: of the layer (`ImplicitLaw`); this covers flow-controlled laws and ideal sources.
    law_form: str = "potential"
    #: Edges of this law's kind form consecutive groups of `ports` edges (kind order); the
    #: law may couple the edges within a group (a transformer's two windings), never across.
    ports: int = 1
    #: True for a law with storage (a capacitance, an inductance): it implements every mode
    #: of `LawContext` and reads its step-start values from the context.
    dynamic: bool = False
    #: True when `flow` takes a third argument, the `LawContext` (a dynamic or
    #: frequency-dependent law). False keeps the two-argument `flow(dp, drivers)`.
    contextual: bool = False
    #: True when the law accepts complex `dp` (phasor layers): a linear law with a real or
    #: complex coefficient. Nonlinear laws stay False and are refused by a phasor layer.
    phasor_capable: bool = False

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """Branch flow q(dp); elementwise in dp. Abstract in the base class."""
        raise NotImplementedError(f"{type(self).__name__} does not implement flow()")

    def forward(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None,
                ctx: LawContext | None = None) -> Tensor:
        """Alias for :meth:`flow` so ``Module.__call__``/``torch.func.functional_call`` work.

        Delegates to ``self.flow`` (rather than being assigned ``forward = flow`` in the
        class body) so that subclass overrides of ``flow`` remain reachable. A contextual
        law receives the `LawContext` as a third argument; any other law keeps the
        two-argument call whatever `ctx` is.
        """
        if self.contextual:
            return self.flow(dp, drivers, ctx)
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

    def switching(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None
                  ) -> Tensor | None:
        """Where the law's smoothness breaks: values (last dimension of any length), each of
        which changes sign where ``flow`` switches between two analytic pieces (the edges of
        a regularisation band, :func:`band_edges`), in units of the piece's own scale. ``None``
        (the default): the law is smooth in ``dp`` and in its drivers. A time integrator can
        put a step boundary on every switch (``noodl.apps.building_physics.modelica.run``), so
        that the flows are smooth within every step."""
        return None

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


class ImplicitLaw(Element):
    """A constitutive law written as a relation `residual(dp, q, drivers, ctx) = 0`.

    The flow of an implicit law's edges is an unknown of the layer, alongside the nodal
    potentials (Modified Nodal Analysis). This is the form of a flow-controlled law
    (`dp = h(q)`), of an ideal potential source (`dp = V`), and of a law that couples flow and
    potential through stored state (an inductance). `residual` returns one value per edge,
    with the shape of `dp`.
    """

    law_form = "implicit"
    contextual = True

    def residual(self, dp: Tensor, q: Tensor, drivers: Mapping[str, Tensor] | None,
                 ctx: LawContext | None) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement residual()")

    def forward(self, dp: Tensor, q: Tensor, drivers: Mapping[str, Tensor] | None = None,
                ctx: LawContext | None = None) -> Tensor:
        """Return `residual(dp, q, drivers, ctx)`.

        An implicit law is called as `law(dp, q, drivers, ctx)`, with the flow `q` as the
        second argument, unlike a potential-controlled law, which is called as
        `law(dp, drivers, ctx)` and computes its flow from `dp`.
        """
        return self.residual(dp, q, drivers, ctx)

    def _no_flow(self) -> TypeError:
        return TypeError(
            f"{type(self).__name__} (kind {self.kind!r}) is an implicit law: its flow is an "
            f"unknown of the layer, not a function flow(dp); read it from the layer's solved q"
        )

    def flow(self, dp, drivers=None, ctx=None):
        raise self._no_flow()

    def dflow(self, dp, drivers=None):
        raise self._no_flow()

    def linear_init(self, drivers=None):
        raise self._no_flow()
