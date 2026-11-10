"""Branch constitutive laws: q = flow(dp, drivers) mapping potential difference to flow.

Every :class:`ConstitutiveLaw` is a ``torch.nn.Module`` so its numeric parameters register as
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

import inspect
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from noodl._broadcast import broadcast_shapes
from noodl.elements._slopes import _colour_blocks

Tensor = torch.Tensor

#: The modes a dynamic or implicit constitutive law is evaluated in (see `LawContext`).
MODES = ("dc", "initial", "transient", "phasor")
#: The time-integration schemes, one vocabulary for every layer kind. `"exact"` (the exact
#: exponential of a linear step) is valid for a `TransportLayer` only; `"backward_euler"`
#: (first order, flows at the step's end) and `"trapezoidal"` (second order) are valid for both
#: it and a `KirchhoffLayer`, which refuses `"exact"` by name. `"forward_euler"` is valid for
#: a `KirchhoffLayer` whose storage sits at BOUNDARY nodes only (an EPANET tank): the solve
#: holds each such node at its step-start potential, and the step then moves it by
#: `dt * inflow / capacity`, the inflow being the one the solve gave. A layer with storage at
#: interior nodes, or with dynamic laws, refuses it, and so does a `TransportLayer`.
SCHEMES = ("exact", "backward_euler", "trapezoidal", "forward_euler")


@dataclass(frozen=True)
class LawContext:
    """What a dynamic or implicit constitutive law needs besides its own edge values.

    `mode` is one of `MODES`: `"dc"` the steady operating point (storage elements at rest:
    a capacitance carries no current, an inductance no voltage); `"initial"` the consistent
    initial point (a capacitance holds its initial voltage, an inductance its initial
    current); `"transient"` one implicit step of length `dt` ending at time `t`, from the
    step-start values `dp_prev` and `q_prev` of the law's own edges, with `scheme` one
    of `SCHEMES` (`"backward_euler"` or `"trapezoidal"`); `"phasor"` the quasi-static
    sinusoidal state at angular frequency `omega` (a tensor broadcasting against the law's
    edges). `t` is the evaluation time (the END of a transient step), `None` when the caller
    tracks no clock.
    """

    mode: str
    dt: float | None = None
    t: float | None = None
    scheme: str = "trapezoidal"
    dp_prev: Tensor | None = None
    q_prev: Tensor | None = None
    omega: Tensor | None = None


def step_rate(change: Tensor, ctx: LawContext, previous: Tensor | None = None) -> Tensor:
    """The rate at which a stored quantity changes over one implicit step: `change / dt`
    (backward Euler) or `2 change / dt - previous` (trapezoidal, `previous` = the rate at
    the step start).

    `change` is the stored amount's change over the step (a capacitor's `C (dp - dp_prev)`, a
    tank's `A (H - H_prev)`), and `ctx` a transient `LawContext` that gives `dt` and `scheme`.
    Every storage in noodl turns its change into a rate here: the capacitance and inductance
    laws, and the `NodeStorage` withdrawal of a layer (`noodl.storage`).
    """
    if ctx.scheme == "backward_euler":
        return change / ctx.dt
    if ctx.scheme != "trapezoidal":
        raise ValueError(
            f"step_rate: scheme={ctx.scheme!r}; a storage steps with 'backward_euler' or "
            f"'trapezoidal' (schemes: {SCHEMES})"
        )
    if previous is None:
        raise ValueError(
            "step_rate: the trapezoidal scheme needs `previous`, the rate at the step start"
        )
    return 2.0 * change / ctx.dt - previous


def step_rate_factor(ctx: LawContext) -> Tensor | float:
    """d step_rate / d change: `1/dt` (backward Euler) or `2/dt` (trapezoidal).

    The slope a storage adds to the Jacobian is this factor times its capacity
    (`NodeStorage.rate_slope`)."""
    if ctx.scheme == "backward_euler":
        return 1.0 / ctx.dt
    if ctx.scheme != "trapezoidal":
        raise ValueError(
            f"step_rate_factor: scheme={ctx.scheme!r}; a storage steps with "
            f"'backward_euler' or 'trapezoidal' (schemes: {SCHEMES})"
        )
    return 2.0 / ctx.dt


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
    (:meth:`ConstitutiveLaw.switching`), in units of the band's width."""
    w = torch.as_tensor(width, dtype=x.dtype)
    shape = broadcast_shapes(x.shape, w.shape)
    return torch.cat([torch.broadcast_to((x - w) / w, shape),
                      torch.broadcast_to((x + w) / w, shape)], dim=-1)


class ConstitutiveLaw(torch.nn.Module):
    """Abstract branch law: flow as a function of potential difference.

    Subclasses implement :meth:`flow`. :meth:`dflow` and :meth:`linear_init` have working
    defaults built on :meth:`flow` and autograd; subclasses may override either with an
    analytic form for speed and exactness where autograd is ill-conditioned.

    ``dp_independent`` (default ``False``) is a class-level DECLARATION, not something
    inferred from whether ``flow`` happens to build an autograd graph back to ``dp``. Set it
    ``True`` only when ``flow`` is mathematically independent of ``dp`` (so ``dflow`` is
    identically zero everywhere) -- ``FixedFlow`` is the built-in example. The differentiable
    solve path (``KirchhoffLayer.solve(differentiable=True)``, in
    ``noodl.layers._functional.FunctionalLaws.dflows``) uses this flag, and only this flag, to
    decide whether a missing gradient of ``flow`` with respect to ``dp`` is legitimate (this
    element declares it does not depend on ``dp``) or a bug (some other element's ``flow``
    silently dropped the autograd graph, e.g. via a stray ``dp.detach()``). Declaring instead
    of inferring keeps that distinction loud: a third-party ``ConstitutiveLaw`` subclass that
    accidentally detaches ``dp`` now raises a clear error instead of silently receiving a
    zero Jacobian column (which leaves the forward solve looking fine -- Newton still
    converges on the exact residual -- while the adjoint gradient computed from that Jacobian
    is silently wrong).
    """

    kind: str
    dp_independent: bool = False
    #: True when `dflow` is the exact derivative of `flow` (to rounding) wherever `flow` is
    #: smooth, so the differentiable solve may use it for its Newton Jacobian and its adjoint.
    #: False for a law whose `dflow` is a deliberate Newton aid (a slope floor, a slope where
    #: the flow is flat): the differentiable solve then differentiates `flow` by autograd.
    #: Declared, not inferred; audited by `tests/elements/test_dflow_exact.py`.
    exact_slope: bool = True
    #: Edges of this law's kind form consecutive groups of `ports` edges (kind order); the
    #: law may couple the edges within a group (a transformer's two windings), never across.
    ports: int = 1
    #: True for a law that reads its step-start values (`ctx.dp_prev`, `ctx.q_prev`): the
    #: layer then needs `history` on a step.
    dynamic: bool = False
    #: Set per class from the signature of `flow`: True when it has a parameter named `ctx`,
    #: and the law is then given the `LawContext`.
    _takes_ctx: bool = False
    #: True when the law accepts complex `dp` (phasor layers): a linear law with a real or
    #: complex coefficient. Nonlinear laws stay False and are refused by a phasor layer.
    phasor_capable: bool = False
    #: True when the law's flow (or residual) is affine in `dp` (and `q`) once its drivers
    #: and `LawContext` are fixed: its Jacobian is then constant within one solve. A layer
    #: whose laws and withdrawals are all linear takes the full Newton step from the start
    #: (`KirchhoffLayer.linear`), so it converges in one iteration. Declared, not inferred.
    linear: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for gone, use in (
                ("law_form", "a law that gives a relation subclasses ImplicitLaw"),
                ("contextual", "declare a `ctx` parameter in `flow` to receive the context")):
            if gone in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} defines `{gone}`, which no longer exists: {use}")
        cls._takes_ctx = "ctx" in inspect.signature(cls.flow).parameters

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        """Branch flow q(dp); elementwise in dp. Abstract in the base class.

        A law that needs the `LawContext` (time, frequency, step-start values) names a third
        parameter `ctx`; the layer then passes it."""
        raise NotImplementedError(f"{type(self).__name__} does not implement flow()")

    def residual(self, dp: Tensor, q: Tensor, drivers: Mapping[str, Tensor] | None = None,
                 ctx: LawContext | None = None) -> Tensor:
        """The law as a relation between the potential difference and the flow, zero when
        they agree: `q - flow(dp)` for a law that gives its flow. A law that cannot give the
        flow from the potential difference (an `ImplicitLaw`) writes this instead, and the
        layer then solves for its flow as an unknown."""
        return q - self(dp, drivers, ctx)

    def forward(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None,
                ctx: LawContext | None = None) -> Tensor:
        """Alias for :meth:`flow` so ``Module.__call__``/``torch.func.functional_call`` work.

        Delegates to ``self.flow`` (rather than being assigned ``forward = flow`` in the
        class body) so that subclass overrides of ``flow`` remain reachable. A law whose
        ``flow`` has a ``ctx`` parameter receives the `LawContext`; any other law keeps the
        two-argument call whatever `ctx` is.
        """
        if self._takes_ctx:
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
        ``noodl.layers._functional.FunctionalLaws.dflows``, reached here via
        ``KirchhoffLayer.linear_init`` instead. Raise a clear error naming the element
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
                    f"autograd-based dflow() cannot differentiate it. If this law's flow "
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
                    f"autograd-based dflow() cannot differentiate it. If this law's flow "
                    f"is genuinely independent of dp, override dflow() (and linear_init()) "
                    f"with the analytic zero, as FixedFlow does, and set "
                    f"dp_independent = True. Otherwise flow() is dropping the autograd "
                    f"graph (e.g. a stray dp.detach())."
                ) from exc
        return grad

    def jacobian(self, dp: Tensor, q: Tensor | None, drivers: Mapping[str, Tensor] | None,
                 ctx: LawContext | None) -> tuple[Tensor, ...]:
        """The law's local derivative blocks, shape `(..., G, m, m, w, w)` per input: G groups
        of `ports` = m edges, `w = 2` for a complex `dp` (its real view), else 1.

        Returns `(d flow / d dp,)` for a law with a flow and `(d r / d dp, d r / d q)` for an
        implicit law; `q` is the edges' flow, used by an implicit law only. A law that knows
        its derivatives overrides this (a phasor law writes its entries as real 2x2 blocks).

        The default is `dflow` for a one-port, law without `ctx` on a real `dp`, and
        otherwise the law's own `forward` differentiated by reverse-mode colouring.
        """
        if (self.ports == 1 and not self._takes_ctx and not dp.is_complex()
                and self._drivers_fit(dp, drivers)):
            return (self.dflow(dp, drivers)[..., None, None, None, None],)
        return self._coloured_jacobian(dp, q, drivers, ctx)

    @staticmethod
    def _drivers_fit(dp: Tensor, drivers: Mapping[str, Tensor] | None) -> bool:
        """Whether every driver's batch shape broadcasts into `dp`'s: only then is the
        autograd `dflow` per instance (a wider driver batch would be summed over)."""
        batch = tuple(dp.shape[:-1])
        for v in (drivers or {}).values():
            if isinstance(v, Tensor) and v.dim() > 0:
                try:
                    if tuple(broadcast_shapes(batch, v.shape[:-1])) != batch:
                        return False
                except (RuntimeError, ValueError):
                    return False
        return True

    def _coloured_jacobian(self, dp, q, drivers, ctx) -> tuple[Tensor, ...]:
        """`_colour_blocks` of `forward` with respect to `dp` (and `q`, for an implicit law)."""
        from noodl.layers._device import complex_of
        w = 2 if dp.is_complex() else 1
        cdtype = complex_of(dp.dtype)
        if q is None:
            return tuple(_colour_blocks(lambda d: self(d, drivers, ctx), (dp,), self.ports, w,
                                        cdtype))
        return tuple(_colour_blocks(lambda d, qq: self(d, qq, drivers, ctx), (dp, q),
                                    self.ports, w, cdtype))

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


class ImplicitLaw(ConstitutiveLaw):
    """A law that cannot give the flow from the potential difference (an ideal source, a
    flow-controlled pipe, an inductance) gives a relation `residual(dp, q, drivers, ctx) = 0`
    instead, and the layer adds its flow as an unknown (Modified Nodal Analysis).

    Subclasses implement `residual`, which returns one value per edge, with the shape of
    `dp`. `flow` and `dflow` refuse: the flow is read from the layer's solved `q`.
    """

    def residual(self, dp: Tensor, q: Tensor, drivers: Mapping[str, Tensor] | None = None,
                 ctx: LawContext | None = None) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement residual()")

    def forward(self, dp: Tensor, q: Tensor, drivers: Mapping[str, Tensor] | None = None,
                ctx: LawContext | None = None) -> Tensor:
        """Return `residual(dp, q, drivers, ctx)`.

        An implicit law is called as `law(dp, q, drivers, ctx)`, with the flow `q` as the
        second argument, unlike a law with a flow, which is called as `law(dp, drivers, ctx)`
        and computes its flow from `dp`.
        """
        return self.residual(dp, q, drivers, ctx)

    def _no_flow(self) -> TypeError:
        return TypeError(
            f"{type(self).__name__} (kind {self.kind!r}) is a relation: its flow is an "
            f"unknown of the layer; read it from the solved q"
        )

    def flow(self, dp, drivers=None, ctx=None):
        raise self._no_flow()

    def dflow(self, dp, drivers=None):
        raise self._no_flow()

    def linear_init(self, drivers=None):
        """Refused: a relation has no tangent `(c, k)`. A layer's linear guess
        (`KirchhoffLayer.linear_init`) takes its `residual` and `jacobian` at zero flow
        instead."""
        raise self._no_flow()
