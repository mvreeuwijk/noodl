"""Model: several physics layers on one typed graph, stepped together.

Per step (framework spec section 3, "time stepping of a model"): closures update the drivers
from the current state; every potential layer is solved quasi-steadily; every capacitated
layer takes its one explicit clip/allocate step (spec 4.2b); every transport layer is
advanced (sub-stepped if asked) on the flows of its kinds; reactions are applied.
`coupling="pingpong"` does that once per step with the state at the START of the step
(Hensen 1995); `coupling="iterate"` ("onion") repeats it until the named transport states stop
changing.

Keys. State: "<layer>.phi" (full-node order), "<layer>.q" (the layer's kind order),
"<layer>.x" (interior order, (n_i,) or (n_i, K)), "<layer>.s" (a capacitated layer's
per-node storage, full-node order; that layer also writes its realised flows to
"<layer>.q"). Drivers: "<layer>.phi_boundary", "<layer>.x_boundary", optional
"<layer>.sources" (FULL-node order, zeros on boundary and inactive nodes), optional
"<layer>.capacity" (a transport layer's per-step capacity override, spec 4.6b; absent, the
layer's construction-time capacity stands), "<layer>.requests" (a capacitated layer's
per-edge requested flow, required every step). Closures return driver updates; they may
not write state keys. A transport layer whose kinds no
potential layer provides reads its branch flows from the driver "<layer>.q", in the layer's
own flow_kinds order; a layer with both a potential owner and that driver raises. A closure
may also declare `state_keys`, keys it carries across steps itself (spec 4.6a); those are
copied from its return into the returned state, evaluated from the step-start state in
every pass of a coupling that takes more than one (N1) -- never fed forward from an earlier
pass's own output, which would integrate a stateful closure once per pass instead of once
per step. A `coupling="iterate"` step started without such a key already in the step-start
state warns by name (`RuntimeWarning`, A3): N1's pinning has nothing to pin on that step, so
the closure integrates once per pass instead of once per step until the key is seeded.

`**solve_kwargs` of `step`/`steady` reach the POTENTIAL solves only (`differentiable`,
`on_failure`, `method`, Newton kwargs). Transport steps always raise on failure.

Differentiability: ping-pong is one pass of differentiable operations; iterate carries no
graph on its primal passes and attaches the IMPLICIT adjoint of its converged interface to
one extra pass instead of unrolling them, so the gradient is the landed fixed point's rather
than the truncated iteration's (P1-2; the convergence decision is made on detached copies, as
Newton's mask is). Failure follows the layers: raise by default, naming the offender.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from noodl.layers.capacitated import CapacitatedTransferLayer
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.reaction import Reaction
from noodl.layers.transport import TransportLayer
from noodl.solvers.fixed_point import differentiate_fixed_point
from noodl.topology import Network

Tensor = torch.Tensor
State = dict[str, Tensor]
Drivers = dict[str, Tensor]
_STATE_SUFFIXES = ("phi", "q", "x", "s")


def _detached(diag):
    """The same (nested) diagnostics structure with every tensor detached.

    Diagnostics are a REPORT. A tensor in them that is still attached to a pass's graph
    offers a second, silent route around the fixed-point adjoint: a loss touching it would
    be differentiated through that single pass, which is exactly the truncated derivative
    P1-2 removes. `_iterate` therefore detaches the diagnostics of both the pass it runs
    only to discover whether anything differentiable reaches the state and the one the
    adjoint is attached to.
    """
    if isinstance(diag, Tensor):
        return diag.detach()
    if isinstance(diag, dict):
        return {k: _detached(v) for k, v in diag.items()}
    return diag


@runtime_checkable
class Closure(Protocol):
    """state, drivers -> driver updates (a mapping of NEW driver values, merged in order).

    A closure that INTEGRATES some state over the step (R6) declares a class or instance
    attribute `integrates = True` and is instead called as `closure(state, drivers, ctx)`,
    receiving the `StepContext` it must advance by; see `StepContext` and
    `Model._apply_closures`.
    """

    def __call__(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class StepContext:
    """What a state transition needs from the model: the interval it must integrate over
    (`None` under `steady()`, which an integrating closure refuses) and, when the caller
    tracks it, the time at the start of the step."""

    dt: float | None
    t: float | None = None


@dataclass(frozen=True)
class Ports:
    """What a coupled or external model may prescribe and read (framework spec 4.1)."""

    boundary_nodes: dict[str, list]
    prescribed_keys: dict[str, str]
    boundary_flows: dict[str, Tensor]
    # Transport layer -> the driver key its branch flows are read from. Present for EVERY
    # transport layer, whether or not a potential layer owns its kinds: a coupled model that
    # wants to drive a layer's flows needs the key, and a model that wants to know whether
    # it MAY do so reads `Model.flow_layer_of[name] is None`. Defaulted so that any existing
    # construction of `Ports` keeps working.
    flow_keys: dict[str, str] = field(default_factory=dict)


class Model:
    """Layers on one `Network`, stepped together. See the module docstring for the keys.

    Two couplings (Hensen 1995), chosen with `coupling`:

    `"pingpong"` (the default) takes exactly ONE pass per step -- closures, potential solves,
    capacitated steps, transport steps, reactions -- with the state at the START of the step
    (the capacitated step sits between the potential solves and the transport steps so that a
    transport layer reading `"<layer>.q"` sees a freshly written flow whichever kind of layer
    wrote it; see `_pass`). It is cheap and it is what a weakly coupled model wants; its
    splitting error is first order in `dt`.

    `"iterate"` (the "onion") repeats that pass within the one step until the transport
    states named in `iterate_tol` stop changing, at most `iterate_max` times. Successive
    substitution with 0.5 relaxation, which takes a pass to start: pass 1 sees the state at
    the start of the step, pass 2 sees pass 1's transport states UNRELAXED (there is no
    earlier pass to average them with), and pass 3 is the first to see a mean -- from there
    on, pass k sees the mean of passes k-2 and k-1. Every pass re-advances the transport
    layers from the state at the start of the step. `iterate_tol` is REQUIRED in this mode
    and is an ABSOLUTE tolerance per layer, on that layer's own units; a transport layer left
    out of it is stepped every pass but not tested, which is what a species layer whose mass
    fractions are ~1e-3 wants when a thermal layer in kelvin sets the pace. Convergence is
    decided per batch instance, and a batch that does not converge within `iterate_max`
    raises, naming the instances (`on_failure="return"` with `diagnostics` returns instead --
    but only when nothing differentiable reached the state on this call, e.g. under
    `torch.no_grad()` or with `differentiable=False`; a non-converged iteration has no fixed
    point to differentiate, so `on_failure="return"` still raises, naming that reason, when a
    gradient was wanted (A2)).

    The onion's fixed point is differentiated IMPLICITLY, not by unrolling the passes: the
    primal passes carry no graph, and after convergence the certified pass runs once more on
    the graph with `solvers.fixed_point.differentiate_fixed_point` attaching the adjoint of
    the interface equations (`adjoint_rtol`, the GMRES tolerance of that one small solve).
    The returned gradient is the derivative of the fixed point the iteration LANDED on, so
    what remains of the pass count and of `iterate_tol` in it is only that the landing point
    is a fixed point to within `iterate_tol` -- an O(primal residual) error, against the
    unrolled O(contraction^passes) truncation it replaces, and exactly zero where the
    interface is linear. Backward memory is one pass rather than all of them (P1-2).
    `diagnostics["adjoint"]` says whether that pass ran: `"implicit"` when it did, `None`
    when nothing differentiable reached the state (or the iteration did not converge) and no
    extra pass was needed. `diagnostics["adjoint_batched"]` says whether that solve ran as one
    independent GMRES system per batch instance (every state key the pass recomputes carrying
    the batch as its leading dims) or, when some key does not (a value shared across
    instances), as today's single system flattened over the whole interface; `False` when
    `"adjoint"` is `None` too.
    """

    def __init__(
        self,
        net: Network,
        layers: Mapping[str, PotentialFlowLayer | TransportLayer | CapacitatedTransferLayer],
        closures: Sequence[Closure] = (),
        reactions: Sequence[tuple[str, Reaction]] = (),
        coupling: str = "pingpong",
        iterate_tol: Mapping[str, float] | None = None,
        iterate_max: int = 20,
        substeps: Mapping[str, int] | None = None,
        adjoint_rtol: float = 1e-10,
    ) -> None:
        if coupling not in ("pingpong", "iterate"):
            raise ValueError(
                f"Model: coupling must be 'pingpong' or 'iterate', got {coupling!r}"
            )
        self.net = net
        self.layers = dict(layers)
        self.potential: dict[str, PotentialFlowLayer] = {}
        self.transport: dict[str, TransportLayer] = {}
        self.capacitated: dict[str, CapacitatedTransferLayer] = {}
        for name, layer in self.layers.items():
            if isinstance(layer, PotentialFlowLayer):
                self.potential[name] = layer
            elif isinstance(layer, TransportLayer):
                self.transport[name] = layer
            elif isinstance(layer, CapacitatedTransferLayer):
                self.capacitated[name] = layer
            else:
                raise TypeError(
                    f"Model: layer {name!r} is a {type(layer).__name__}, not a "
                    f"PotentialFlowLayer, TransportLayer or CapacitatedTransferLayer"
                )
            if layer.net is not net:
                raise ValueError(
                    f"Model: layer {name!r} is built on a different Network than the model"
                )
        # Moved here (was computed at the very end of __init__) so that FR-2's
        # closure_state_keys check below can consult `self.flow_layer_of` -- both only need
        # `self.potential`/`self.transport`, already built above, and nothing between here
        # and the old position read them first.
        self.flow_layer_of: dict[str, str | None] = {}
        self.flow_driver_of: dict[str, str] = {}
        for tname, tl in self.transport.items():
            owners = [
                pn for pn, pl in self.potential.items()
                if all(k in pl.kinds for k in tl.flow_kinds)
            ] + [
                cn for cn, cl in self.capacitated.items()
                if all(k in cl.kinds for k in tl.flow_kinds)
            ]
            # Spec section 5, "Model: driver-prescribed flows". Two potential layers both
            # providing a transport layer's kinds is still ambiguous and still refused here.
            # ZERO owners is no longer an error: the flows are then read from the driver
            # "<layer>.q", which a closure writes -- the street application's whole
            # architecture (prescribed canyon fluxes, routed at intersections) depends on
            # it. The driver's PRESENCE cannot be checked here, because drivers arrive per
            # call; `_kind_flows` does that, and refuses a layer that has both sources.
            if len(owners) > 1:
                raise ValueError(
                    f"Model: transport layer {tname!r} advects on kinds {tl.flow_kinds}, "
                    f"which {len(owners)} potential layers of this model provide "
                    f"({owners}); at most one may"
                )
            self.flow_layer_of[tname] = owners[0] if owners else None
            self.flow_driver_of[tname] = f"{tname}.q"
        self.closures = list(closures)
        # Both callability checks are made HERE, not at the first `step`: a non-callable
        # closure or a reaction without `apply` would otherwise surface a bare
        # "'X' object is not callable" from inside `_pass`, naming neither the model nor
        # which of the closures it was.
        for i, closure in enumerate(self.closures):
            if not isinstance(closure, Closure):
                raise TypeError(
                    f"Model: closure {i} is a {type(closure).__name__}, which is not "
                    f"callable; a closure is called as closure(state, drivers) and returns "
                    f"driver updates"
                )
        # R6: a closure that INTEGRATES some state over the step (rather than merely
        # carrying it, unchanged, across steps) declares `integrates = True` and is called
        # with a third argument, the `StepContext` it must advance by -- see
        # `_apply_closures`. A closure declaring `state_keys` without declaring `integrates`
        # is refused HERE, by name, rather than left to silently integrate with whatever
        # interval its own constructor happened to be given (R6's own bug).
        self._integrating: set[int] = set()
        for closure in self.closures:
            declared = getattr(closure, "integrates", None)
            carries = tuple(getattr(closure, "state_keys", ()))
            if carries and declared is None:
                raise ValueError(
                    f"Model: closure {closure!r} declares state_keys {carries} but not "
                    f"`integrates`; set `integrates = True` if it advances that state over "
                    f"the step (it then receives a StepContext), or `integrates = False` if "
                    f"something else advances it"
                )
            if declared:
                params = inspect.signature(closure).parameters
                positional = [
                    p for p in params.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                if len(positional) < 3 and not any(
                    p.kind is p.VAR_POSITIONAL for p in params.values()
                ):
                    raise ValueError(
                        f"Model: closure {closure!r} declares integrates=True but its call "
                        f"signature takes {len(positional)} positional arguments; an "
                        f"integrating closure is called as closure(state, drivers, ctx)"
                    )
                self._integrating.add(id(closure))
        # CLOSURE-CARRIED STATE (spec 4.6a). A closure may declare
        # `state_keys: tuple[str, ...]`: keys it both READS from the state and WRITES back
        # every call, which `_pass` copies from its return into the returned state. They are
        # neither drivers (they persist across steps) nor layer state (no layer owns them);
        # a sewer manhole level under `storage=True`, or a tank level and the controlled
        # links' status in the water application, are the motivating cases. Two closures
        # claiming one key is refused HERE, naming both, rather than resolved by whichever
        # ran last.
        self.closure_state_keys: dict[str, Closure] = {}
        for closure in self.closures:
            for key in getattr(closure, "state_keys", ()):
                head, _, tail = str(key).rpartition(".")
                if head in self.layers and tail in _STATE_SUFFIXES:
                    # FR-2: mirror `_apply_closures`' own ownerless-"<layer>.q" carve-out --
                    # for a transport layer no potential layer owns, "<layer>.q" is the
                    # DRIVER a flow closure writes, not that layer's state, so a closure may
                    # declare it as state_keys (a flow closure that also remembers its own
                    # last-written flow across steps is exactly this shape). Every other
                    # "<layer>.phi/q/x" stays refused, including "<layer>.q" for a layer a
                    # potential layer DOES own.
                    prescribed = (
                        tail == "q"
                        and head in self.transport
                        and self.flow_layer_of.get(head) is None
                    )
                    if not prescribed:
                        raise ValueError(
                            f"Model: closure {closure!r} declares state_keys entry {key!r}, "
                            f"which is layer {head!r}'s own state key; a closure may carry "
                            f"its OWN state, never a layer's"
                        )
                if key in self.closure_state_keys:
                    raise ValueError(
                        f"Model: closures {self.closure_state_keys[key]!r} and "
                        f"{closure!r} both declare the state key {key!r}; exactly one "
                        f"closure may own a state key"
                    )
                self.closure_state_keys[key] = closure
        self.reactions = list(reactions)
        for lname, reaction in self.reactions:
            if lname not in self.transport:
                raise KeyError(
                    f"Model: reaction targets {lname!r}, not a transport layer of this "
                    f"model ({sorted(self.transport)})"
                )
            if not callable(getattr(reaction, "apply", None)):
                raise TypeError(
                    f"Model: reaction for layer {lname!r} is a {type(reaction).__name__}, "
                    f"not a Reaction; it must provide apply(x, dt, drivers)"
                )
        self.coupling = coupling
        self.iterate_tol = dict(iterate_tol or {})
        self.iterate_max = int(iterate_max)
        # The tolerance of the ONE implicit-adjoint GMRES solve at the converged interface
        # (P1-2), deliberately independent of the primal `iterate_tol`: that independence is
        # the whole point of differentiating the fixed point instead of the iteration.
        self.adjoint_rtol = float(adjoint_rtol)
        if not self.adjoint_rtol > 0.0:
            raise ValueError(
                f"Model: adjoint_rtol must be positive, got {adjoint_rtol!r}; it is the "
                f"relative tolerance of the fixed-point adjoint's GMRES solve"
            )
        if coupling == "iterate":
            if not self.iterate_tol:
                raise ValueError(
                    "Model: coupling='iterate' requires iterate_tol={transport layer name: "
                    "absolute tolerance on that layer's state}"
                )
            unknown = sorted(set(self.iterate_tol) - set(self.transport))
            if unknown:
                raise KeyError(
                    f"Model: iterate_tol names {unknown}, not transport layers of this model "
                    f"({sorted(self.transport)})"
                )
            # Refused HERE rather than left to fail at the first `step`: one pass has no
            # second pass to compare against, so it can never report convergence -- a budget
            # below two is a configuration that always raises, and a `iterate_max=0` would
            # return the input state untouched before doing so. Enforcing it also keeps the
            # diagnostics one shape: `converged` and `max_change` are always the ones a real
            # comparison produced, never a degenerate 0-dim placeholder and an empty dict.
            if self.iterate_max < 2:
                raise ValueError(
                    f"Model: coupling='iterate' requires iterate_max >= 2, got "
                    f"{iterate_max!r}; a single pass has no predecessor to compare against, "
                    f"so it can never be reported as converged"
                )
        self.substeps = {name: 1 for name in self.transport}
        for name, k in (substeps or {}).items():
            if name not in self.transport:
                raise KeyError(
                    f"Model: substeps names {name!r}, not a transport layer of this model "
                    f"({sorted(self.transport)})"
                )
            try:
                k_int = int(k)
            except (TypeError, ValueError) as exc:
                # Without this, a substeps value that is not a number at all fails as a bare
                # "invalid literal for int()" naming neither the model nor the layer.
                raise TypeError(
                    f"Model: substeps[{name!r}] must be an integer, got {k!r} "
                    f"({type(k).__name__})"
                ) from exc
            if k_int < 1:
                raise ValueError(
                    f"Model: substeps[{name!r}] must be >= 1, got {k!r}"
                )
            self.substeps[name] = k_int

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _require(drivers: Mapping[str, Tensor], key: str) -> Tensor:
        try:
            return drivers[key]
        except KeyError as exc:
            raise KeyError(f"Model: driver {key!r} is required and was not given") from exc

    def _apply_closures(
        self,
        state: Mapping[str, Tensor],
        drivers: Mapping[str, Tensor],
        written: set[str] | None = None,
        ctx: StepContext | None = None,
    ) -> Drivers:
        """Run every closure, in order, over `drivers`; return the resulting drivers.

        `written` (keyword-out-parameter, like `solve`'s `diagnostics`; `None` by default so
        every existing caller -- including several apps' tests that call this directly -- is
        unaffected) is filled with whichever `closure_state_keys` some closure's OWN return
        actually wrote THIS call: N14's fix. A same-named entry already sitting in `drivers`
        (the caller's own, unrelated to any closure) must not be mistaken for a closure
        having written its declared state key -- `key in drv` alone cannot tell the two
        apart, since `drv` starts as a copy of `drivers`.

        `ctx` (R6) is threaded to an INTEGRATING closure only (`closure.integrates = True`):
        `StepContext(dt)` from `_pass` inside `step`, `StepContext(dt=None)` inside `steady`
        (refused here by name -- an integrating closure has no interval to integrate over),
        and `None` from a query (`residuals`, `current_flows`, `ports`, the sewer report),
        where an integrating closure evaluates its algebraic outputs at the given state
        without advancing. A non-integrating closure never sees `ctx` at all, so its call
        signature and behaviour are exactly as before R6.
        """
        drv: Drivers = dict(drivers)
        for closure in self.closures:
            if id(closure) in self._integrating:
                if ctx is not None and ctx.dt is None:
                    raise ValueError(
                        f"Model: closure {closure!r} declares integrates=True and cannot be "
                        f"evaluated by steady(), which has no interval to integrate over"
                    )
                result = closure(state, drv, ctx)
            else:
                result = closure(state, drv)
            for key, value in result.items():
                head, _, tail = key.rpartition(".")
                if head in self.layers and tail in _STATE_SUFFIXES:
                    # ONE exception to "closures write drivers only": "<layer>.q" for a
                    # transport layer no potential layer owns is a DRIVER, not a state --
                    # it is the key `_kind_flows` reads that layer's branch flows from, and
                    # writing it is exactly what a flow closure is for. Every other
                    # "<layer>.phi/q/x" stays refused, including "<layer>.q" for a layer a
                    # potential layer DOES own, where it would silently contradict the
                    # solve.
                    prescribed = (
                        tail == "q"
                        and head in self.transport
                        and self.flow_layer_of.get(head) is None
                    )
                    if not prescribed:
                        raise ValueError(
                            f"Model: closure {closure!r} returned {key!r}, which is a "
                            f"state key; closures write drivers only (the one exception "
                            f"is '<layer>.q' for a transport layer whose flows no "
                            f"potential layer provides)"
                        )
                drv[key] = value
                if written is not None and key in self.closure_state_keys:
                    written.add(key)
        return drv

    def _zero_sources(self, layer: TransportLayer, like: Tensor, n_like: int) -> Tensor:
        """Full-node zeros in `like`'s LAYOUT (reduced `(n,)` or stacked `(n, K)`).

        `like` decides the layout, so it must be a tensor whose own layout is the one the
        caller wants back: the state `x` where there is one (`step`, `residuals`, and
        `steady` whenever the state carries `"<layer>.x"`). `TransportLayer.steady` reads its
        `reduced` flag off the SOURCES rather than off `x`, so a `(n_b, K)`-shaped
        `x_boundary` used as `like` would flip the returned state to `(..., n_i, K)` for a
        single-species layer; only fall back to it when there is no `x` at all.
        """
        _, reduced = layer._to_stacked(like, n_like, "state")
        batch = like.shape[:-1] if reduced else like.shape[:-2]
        tail = (self.net.n,) if reduced else (self.net.n, layer.n_species)
        return torch.zeros(batch + tail, dtype=like.dtype, device=like.device)

    def _kind_flows(self, name: str, state: Mapping[str, Tensor],
                    drivers: Mapping[str, Tensor]) -> Tensor:
        """This transport layer's branch flows, in its own `flow_kinds` order.

        From the owning potential layer's solved `q` when there is one, otherwise from the
        driver `"<layer>.q"`. A layer with BOTH is a contradiction -- two different answers
        for the same flows -- and is refused by name rather than resolved by a precedence
        rule nobody would remember.

        `__init__`'s ownership scan also lets a `CapacitatedTransferLayer` become a
        transport layer's flow owner (it writes `"<name>.q"` in the same key convention).
        Reading those flows would need `CapacitatedTransferLayer.flows_of_kind`, which is
        the species/quality-transport follow-up (design spec amendment A5) and is NOT built:
        refused here by name rather than left to raise a bare `KeyError` off
        `self.potential[owner]`.
        """
        layer = self.transport[name]
        owner = self.flow_layer_of[name]
        key = self.flow_driver_of[name]
        if owner is not None:
            # Checked BEFORE the both-sources refusal below, whose message says "potential
            # layer" and would be factually wrong about a capacitated owner.
            if owner in self.capacitated:
                raise NotImplementedError(
                    f"Model: transport layer {name!r} advects on kinds {layer.flow_kinds}, "
                    f"which capacitated layer {owner!r} provides; reading a capacitated "
                    f"layer's flows into a transport layer is not implemented (design spec "
                    f"amendment A5, species/quality transport). Drive {name!r} from the "
                    f"driver {key!r} instead, or give its kinds to a potential layer"
                )
            if key in drivers:
                raise ValueError(
                    f"Model: transport layer {name!r} takes its flows from potential layer "
                    f"{owner!r}, and the driver {key!r} was also given; a layer may have "
                    f"one source of flows, not two -- drop {key!r} or remove {owner!r}"
                )
            return self.potential[owner].flows_of_kind(state[f"{owner}.q"], layer.flow_kinds)
        q = self._require(drivers, key)
        b = int(layer._flow_src.shape[0])
        if q.dim() == 0 or q.shape[-1] != b:
            got = q.shape[-1] if q.dim() >= 1 else 0
            raise ValueError(
                f"Model: driver {key!r} must have trailing shape ({b},) -- transport layer "
                f"{name!r} advects on {b} flow edges of kinds {layer.flow_kinds}, in that "
                f"order -- got {got} (shape {tuple(q.shape)})"
            )
        return q

    # -------------------------------------------------------------------- pass
    def _pass(
        self, state, drivers, dt, solve_kwargs, *, step_from: State | None = None,
        t: float | None = None, boundary_transfers: bool | Collection[str] = False,
        produced: list[str] | None = None,
    ) -> tuple[State, dict, Drivers]:
        """One closures -> potential -> capacitated -> transport -> reactions pass;
        `dt=None` means steady (and is refused outright by a model owning a capacitated
        layer, which is inherently discrete-time).

        `t` is the caller's own start-of-step time, if it tracks one; both `dt` and `t` are
        carried to every closure as a `StepContext` (R6), so an INTEGRATING closure reads the
        model's own clock rather than an interval fixed at its own construction.

        `state` is what the closures read and what the potential solves warm-start from.
        `step_from` is the state a transport STEP starts at, which is a different thing
        inside an iterated coupling: pass k there re-advances the SAME time step from the
        state at the start of it, with only the closures' view of the new state updated.
        It defaults to `state`, which is what a single ping-pong pass wants.

        `boundary_transfers`: `True` runs `step_with_transfer` on every transport layer; a
        collection of layer names runs it on ONLY those (every other transport layer takes
        the plain, cheaper `step`); `False` (or an empty collection) runs it on none. Only
        a layer actually asked for gets `diag[name]["boundary_transfer"]` -- a caller that
        names one linked layer out of several must not pay `step_with_transfer`'s extra cost
        (no diagonal shift on the `exact` scheme's Taylor accumulator, milestone 5 R2 review
        finding) on layers whose transfer nothing reads.

        `diag[name]["linear"]` (Task 5, B2), when this transport layer's `scheme` runs a
        linear solve (`"implicit"`, `"trapezoidal"`; not `"exact"`), is the LAST substep's
        `{"backend", "iterations", "residual"}` from `TransportLayer.step`'s own
        `diagnostics=` out-parameter -- see `TransportLayer._resolve_solver` and
        `layers.transport._LinearSolve` for what each entry means and how `linear_solver`
        resolves to it. Absent for `scheme="exact"`, which has no linear solve to report.

        `produced` (a keyword out-parameter, like `_apply_closures`'s `written` and `solve`'s
        `diagnostics`; `None` by default, so every other caller is unaffected) is REPLACED
        with the state keys this pass actually RECOMPUTED, in write order. The returned state
        starts as a copy of the fed one, so a key no layer here writes is handed back
        UNCHANGED, and `_iterate` must know which is which: an unchanged entry in the
        implicit adjoint's interface is a unit row in the interface Jacobian and makes
        `I - G_z` exactly singular (`solvers.fixed_point`). Asking the pass itself is the
        only answer that cannot drift from what the pass does.
        """
        base: State = state if step_from is None else step_from
        # N1: closure-carried state (spec 4.6a) must be evaluated from the STEP-START state
        # on every pass, never from the previous pass's own output -- a closure that
        # integrates (adds dt each call) would otherwise integrate once per PASS instead of
        # once per STEP, because `_iterate` feeds pass k-1's output into pass k. Every other
        # key a closure reads still sees `state` (pass k-1's relaxed transport `.x`, which is
        # exactly what the relaxed coupling wants); only the keys this model's closures
        # themselves carry are pinned to `base`.
        closure_state: State = dict(state)
        for key in self.closure_state_keys:
            if key in base:
                closure_state[key] = base[key]
        written: set[str] = set()
        ctx = StepContext(dt=dt, t=t)
        drv = self._apply_closures(closure_state, drivers, written, ctx)
        new: State = dict(state)
        made: list[str] = [] if produced is None else produced
        made.clear()
        diag: dict = {}
        for name, layer in self.potential.items():
            pb = self._require(drv, f"{name}.phi_boundary")
            sources = drv.get(f"{name}.sources")
            phi_prev = state.get(f"{name}.phi")
            phi0 = None if phi_prev is None else phi_prev[..., layer.interior]
            d: dict = {}
            phi, q = layer.solve(pb, drv, sources, phi0=phi0, diagnostics=d, **solve_kwargs)
            new[f"{name}.phi"], new[f"{name}.q"] = phi, q
            made += [f"{name}.phi", f"{name}.q"]
            diag[name] = d
        if self.capacitated and dt is None:
            raise ValueError(
                "Model: a steady (dt=None) pass has no defined meaning for a "
                "CapacitatedTransferLayer, which is inherently discrete-time"
            )
        for name, layer in self.capacitated.items():
            # `base`, not `state` -- the same N1 rule the comment above states for
            # closure-carried state and the transport loop below follows for `"<layer>.x"`:
            # a layer that INTEGRATES its own state must advance from the STEP-START state
            # on every pass, never from the previous pass's output, or `coupling="iterate"`
            # integrates it once per PASS instead of once per STEP.
            s_prev = base.get(f"{name}.s")
            if s_prev is None:
                raise KeyError(
                    f"Model: state {name + '.s'!r} is required to step capacitated "
                    f"layer {name!r}"
                )
            cd: dict = {}
            s_new, f = layer.step(s_prev, drv, dt, diagnostics=cd)
            new[f"{name}.s"], new[f"{name}.q"] = s_new, f
            made += [f"{name}.s", f"{name}.q"]
            diag[name] = cd
        if isinstance(boundary_transfers, str):
            # Defensive: `step()` already refuses a bare string before reaching here, but
            # `_pass` is not otherwise unreachable except through it -- see that check for
            # why a bare `str` (itself a `Collection[str]`) must never reach the ternary
            # below, which would silently iterate its characters instead.
            raise TypeError(
                f"Model.step: boundary_transfers must be a bool or a collection of layer "
                f"names, not a bare string; pass {boundary_transfers!r} in a set"
            )
        transfer_layers: set[str] = (
            set(self.transport) if boundary_transfers is True
            else set(boundary_transfers) if boundary_transfers
            else set()
        )
        if dt is None and transfer_layers:
            raise ValueError(
                "Model: boundary_transfers has no meaning for a steady solve, which "
                "integrates nothing"
            )
        for name, layer in self.transport.items():
            want_transfer = name in transfer_layers
            q_kind = self._kind_flows(name, new, drv)
            xb = self._require(drv, f"{name}.x_boundary")
            sources = drv.get(f"{name}.sources")
            # Spec 4.6b: a per-step capacity, written by a closure that owns the geometry
            # (a sewer conduit's wetted volume, a headspace volume). Absent, the layer's
            # construction-time capacity stands, so nothing changes for a fixed-storage
            # layer. Shape and positivity are checked by the layer, naming the nodes.
            cap = drv.get(f"{name}.capacity")
            transfer_total = None
            # Task 5 (B2): a fresh dict per layer, threaded into `steady`/`step`/
            # `step_with_transfer` as an out-parameter and read back below into
            # `diag[name]["linear"]`. Only the LAST substep's entry survives (each substep
            # overwrites it), matching how `diag[name]["substeps"]` already reports a count
            # rather than a per-substep history. A scheme with no linear solve (`"exact"`)
            # leaves this dict empty, so no `"linear"` key is added.
            layer_diag: dict = {}
            if dt is None:
                if sources is None:
                    # The state's own `x` is the layout authority when it is there; `x_b`
                    # is the fallback for a steady solve started without one.
                    like = base.get(f"{name}.x")
                    sources = (
                        self._zero_sources(layer, xb, layer.n_b)
                        if like is None
                        else self._zero_sources(layer, like, layer.n_i)
                    )
                x = layer.steady(q_kind, sources, xb, capacity=cap, diagnostics=layer_diag)
            else:
                x = base.get(f"{name}.x")
                if x is None:
                    raise KeyError(
                        f"Model: state {name + '.x'!r} is required to step transport layer "
                        f"{name!r}"
                    )
                if sources is None:
                    sources = self._zero_sources(layer, x, layer.n_i)
                cap_prev = None
                if cap is not None:
                    cap_prev = base.get(f"{name}.capacity")
                    if cap_prev is None:
                        raise KeyError(
                            f"Model: a closure writes the driver {name + '.capacity'!r}, so "
                            f"the step-start state must carry {name + '.capacity'!r} (the "
                            f"storage at the state's own time); build it with "
                            f"Model.initial_capacities(state, drivers)"
                        )
                k = self.substeps[name]
                for j in range(1, k + 1):
                    if cap is None:
                        cap_j = cap_prev_j = None
                    else:
                        cap_prev_j = cap_prev + (j - 1) / k * (cap - cap_prev)
                        cap_j = cap_prev + j / k * (cap - cap_prev)
                    if want_transfer:
                        stepped = layer.step_with_transfer(
                            x, q_kind, sources, xb, dt / k,
                            capacity=cap_j, capacity_prev=cap_prev_j,
                            diagnostics=layer_diag,
                        )
                        x = stepped.x
                        transfer_total = (
                            stepped.boundary_transfer if transfer_total is None
                            else transfer_total + stepped.boundary_transfer
                        )
                    else:
                        x = layer.step(x, q_kind, sources, xb, dt / k, capacity=cap_j,
                                       capacity_prev=cap_prev_j, diagnostics=layer_diag)
                if cap is not None:
                    new[f"{name}.capacity"] = cap
                    made.append(f"{name}.capacity")
                for lname, reaction in self.reactions:
                    if lname == name:
                        x = reaction.apply(x, dt, drv)
            new[f"{name}.x"] = x
            made.append(f"{name}.x")
            diag[name] = {"substeps": self.substeps[name]}
            if "linear" in layer_diag:
                diag[name]["linear"] = layer_diag["linear"]
            if want_transfer:
                diag[name]["boundary_transfer"] = transfer_total
        # Closure-carried state (spec 4.6a): a declared key is copied OUT of the closure's
        # return into the state, so the next step's closures read it back. A closure that
        # declares a key and does not write it every call would freeze that state silently,
        # so the omission is refused by name instead. N14: the check is on `written` (which
        # closure ACTUALLY returned it this call), not on `key in drv` -- a same-named
        # DRIVER the caller passed in would otherwise already be sitting in `drv` and defeat
        # this check, freezing the closure's state at the caller's value with no error.
        for key, closure in self.closure_state_keys.items():
            if key not in written:
                raise KeyError(
                    f"Model: closure {closure!r} declares the state key {key!r} but did "
                    f"not return it; a closure must write every key it declares on every "
                    f"call"
                )
            new[key] = drv[key]
            made.append(key)
        return new, diag, drv

    # ------------------------------------------------------------------ public
    def initial_capacities(self, state: State, drivers: Drivers) -> dict[str, Tensor]:
        """`{"<layer>.capacity": tensor}` for every transport layer whose closures write that
        driver, evaluated as a QUERY at `state` (no integration): the storage at the state's
        own time, which is what the step-start state must carry before the first step."""
        drv = self._apply_closures(state, drivers)
        return {
            f"{name}.capacity": drv[f"{name}.capacity"]
            for name in self.transport if f"{name}.capacity" in drv
        }

    def step(
        self, state, drivers, dt: float, *, t: float | None = None,
        diagnostics: dict | None = None,
        boundary_transfers: bool | Collection[str] = False, **solve_kwargs,
    ):
        """Advance one step (spec's ping-pong or iterated coupling, per `self.coupling`).

        `boundary_transfers` (keyword-only, default False) reports the time-integrated
        amount that crossed a transport layer's boundary during the step, summed over that
        layer's substeps, at `diagnostics["layers"][name]["boundary_transfer"]`: `True` for
        EVERY transport layer, or a collection of layer names for only those (every other
        transport layer takes the plain, cheaper `step` and gets no `"boundary_transfer"`
        entry); `False` (the default) or an empty collection for none. A name in the
        collection that is not one of this model's transport layers is refused, naming it.
        `diagnostics` is created internally when the caller passes none, so the state
        returned is unchanged either way -- only a caller who wants the transfers passes a
        dict. Sign and unit convention: see `TransportLayer.step_with_transfer`. REACTIONS
        ARE APPLIED AFTER TRANSPORT and are not part of the reported transfer.

        Naming only the layer(s) actually linked to a coupling matters for cost: `True`
        forces `step_with_transfer` (no diagonal shift on the `exact` scheme's Taylor
        accumulator) on every transport layer of the model, including ones nothing reads a
        transfer from -- a coupled recipient with an unlinked `exact`-scheme thermal layer
        paid that cost on every one of its substeps for no benefit (task 18b).
        """
        if not dt > 0:
            raise ValueError(f"Model: dt must be positive, got {dt!r}")
        if isinstance(boundary_transfers, str):
            # A bare `str` IS a `Collection[str]` -- `set("species")` iterates its
            # CHARACTERS, not the one name meant, either raising a confusing "layer 's' does
            # not exist" or, on a single-letter layer name (this repo's own tests use "a",
            # "b", "c"), silently selecting the wrong layer. Refuse it outright rather than
            # let either happen.
            raise TypeError(
                f"Model.step: boundary_transfers must be a bool or a collection of layer "
                f"names, not a bare string; pass {boundary_transfers!r} in a set"
            )
        if boundary_transfers is not True and boundary_transfers is not False:
            bad = sorted(set(boundary_transfers) - set(self.transport))
            if bad:
                raise ValueError(
                    f"Model: boundary_transfers names {bad}, not transport layer(s) of this "
                    f"model (has {sorted(self.transport)})"
                )
        return self._advance(
            state, drivers, float(dt), diagnostics, solve_kwargs, t=t,
            boundary_transfers=boundary_transfers,
        )

    def steady(self, state, drivers, *, diagnostics: dict | None = None, **solve_kwargs):
        """The quasi-steady state of every layer at `drivers` (transport layers solved to
        `rate == 0` rather than advanced).

        REACTIONS ARE NOT APPLIED. They are an operator splitting applied AFTER a transport
        step, so they belong to `step` alone: a model carrying a reaction has a `steady` that
        is the fixed point of transport only, not of transport-plus-reaction (spec section 7;
        `residuals` reports the same balance). Deliberate, and pinned by a test.

        `dt=None` here means the `StepContext` an INTEGRATING closure (R6) receives carries
        `dt=None` too, and `_apply_closures` refuses that by the closure's own name: a
        closure that integrates some state over an interval has no interval to integrate over
        at a quasi-steady state.
        """
        return self._advance(state, drivers, None, diagnostics, solve_kwargs)

    def _advance(
        self, state, drivers, dt, diagnostics, solve_kwargs, *, t=None,
        boundary_transfers: bool | Collection[str] = False,
    ) -> State:
        # The coupling seam: ping-pong is exactly ONE pass, taken with the state at the start
        # of the step; `coupling="iterate"` repeats `_pass` until the named transport states
        # stop changing, and reports the pass count it took.
        if self.coupling == "pingpong":
            new, diag, _ = self._pass(
                state, drivers, dt, solve_kwargs, t=t, boundary_transfers=boundary_transfers,
            )
            if diagnostics is not None:
                diagnostics.update({"passes": 1, "layers": diag})
            return new
        return self._iterate(
            state, drivers, dt, diagnostics, solve_kwargs, t=t,
            boundary_transfers=boundary_transfers,
        )

    def _iterate(
        self, state, drivers, dt, diagnostics, solve_kwargs, *, t=None,
        boundary_transfers: bool | Collection[str] = False,
    ) -> State:
        """Hensen's onion: repeat the pass until the named transport states stop changing.

        The relaxation takes a pass to start: `prev` is None after pass 1, so pass 2 is fed
        pass 1's transport states UNRELAXED, and pass 3 is the first fed a mean. From there
        the "<layer>.x" state fed to the closures on pass k is the mean of passes k-2 and
        k-1 (Hensen 1995, successive substitution with 0.5 relaxation). CLOSURE-CARRIED
        state (spec 4.6a, `closure_state_keys`) is the one exception to that relaxation: it
        is evaluated from the STEP-START state on every pass, never fed forward from the
        previous pass's own output (N1) -- a closure that integrates (a sewer manhole's
        storage sweep, a tank level) would otherwise advance once per PASS instead of once
        per STEP. The returned state is the LAST pass's own output, never a relaxed one.
        Convergence is judged per instance on detached copies.

        Differentiation: the primal passes carry no graph. After convergence the certified
        pass runs once more on the graph at the SAME fed state and
        `solvers.fixed_point.differentiate_fixed_point` attaches the implicit adjoint of the
        interface equations, so the returned gradient is the FIXED POINT's own and not the
        truncated iteration's (P1-2). What remains of the pass count and of `iterate_tol` in
        it is only that the landing point is a fixed point to within `iterate_tol`: the error
        is O(primal residual) rather than the unrolled O(contraction^passes), and it is
        exactly zero where the interface is linear. Backward memory is one pass. Pass 1 runs
        on the graph only to learn whether anything differentiable reaches the state -- a
        structural question no inspection of `state` and `drivers` can answer, since a
        differentiable parameter may be captured inside an element, a drive or a closure and
        never appear in either -- and its graph is dropped at once; if nothing does, no extra
        pass runs and a forward-only simulation costs exactly the passes it always did. A run
        that does NOT converge attaches no adjoint either: there is no fixed point to
        differentiate, and handing back the last pass's graph would be exactly the truncated
        derivative this replaced.

        The INTERFACE is every state key the pass RECOMPUTES (`_pass`'s `produced`), minus
        the closure-carried ones. Both exclusions are deliberate:

        * A key no layer writes is copied through `_pass`'s `new = dict(state)` UNCHANGED, so
          it would hand its own input straight back: a unit row in the interface Jacobian,
          `I - G_z` exactly singular, and the adjoint solve failing by name. Such a key is
          not an unknown of the interface at all -- the pass never updates it -- so it is
          simply not in `produced`. Its own graph is preserved another way, below.
        * Closure-carried state is pinned to the STEP-START state on every pass by N1, so a
          pass does not read it from the fed state and it carries no feedback at all; and it
          is the one key a closure may legitimately return unchanged, which would be that
          same unit row. Its gradient path survives regardless, because the differentiable
          pass reads it from `state` itself (`step_from=state`), graph and all -- as do the
          transport and capacitated steps, which advance from `state` for the same reason.
          (A closure-carried key MISSING from the step-start state is the one case N1 does
          not pin; there the pass reads the previous pass's value, and holding it fixed here
          drops a path that only exists because that key was not seeded in the first place.)

        Everything else the pass writes is IN, for the reason the coupler's one-way links
        are: the whole of `new` is fed back to the next pass, and holding any of it fixed
        against the parameters would silently drop the gradient path running through it. An
        entry the pass turns out not to depend on costs a zero row and a zero column, which
        is harmless -- only its share of the GMRES dimension.

        The interface is taken at `fed`, the state the CERTIFIED pass (the one whose output
        was just judged converged) was actually run with -- relaxed `"<layer>.x"` and all --
        so re-running `_pass` on it reproduces that pass, and the state it returns is the
        certified one to solver accuracy. Not necessarily BITWISE: `solvers/select.py` drops
        the SuperLU fast path for an input that requires grad, so this pass may take a
        different linear-solver route than the grad-free primal ones did. The next iterate
        handed to the adjoint is the pass's own UNRELAXED
        output: relaxation does not move the fixed point, but differentiating the relaxed map
        would scale `(I - G_z)^-1` by 1/relaxation and return a gradient wrong by that
        factor.
        """
        missing = [k for k in self.closure_state_keys if k not in state]
        if missing:
            warnings.warn(
                f"Model: coupling='iterate' started a step without the closure-carried state "
                f"{missing} in the step-start state. On this step the key is not pinned to the "
                f"step start (rule N1 pins only keys the state carries), so its closure "
                f"integrates once per pass instead of once per step and the adjoint omits its "
                f"compounding path. Seed it before the first step (the application's "
                f"initial_state helper, or Model.initial_capacities for capacities).",
                RuntimeWarning,
                stacklevel=4,
            )
        fed: State = dict(state)
        # The fed state of the pass being run RIGHT NOW. Equal to `fed` on the converged
        # exit, which breaks before `fed` is rebuilt, but named separately so that a later
        # edit to the loop's tail cannot silently hand the adjoint the wrong linearisation
        # point.
        certified_fed: State = fed
        prev: State | None = None
        change: dict[str, Tensor] = {}
        # A placeholder the second pass always replaces: `iterate_max >= 2` and a non-empty
        # `iterate_tol` are both refused at construction, so the loop below cannot end
        # without a real per-instance verdict of the right shape, dtype and device.
        converged: Tensor = torch.zeros((), dtype=torch.bool)
        passes = 0
        diag: dict = {}
        new: State = dict(state)
        produced: list[str] = []
        needs_adjoint = False
        # A `while` rather than `for passes in range(...)`: the pass count is wanted AFTER
        # the loop (it goes into the diagnostics and into the failure message), which a loop
        # control variable unused inside the body is not (ruff B007).
        while passes < self.iterate_max:
            passes += 1
            certified_fed = fed
            # Pass 1 runs with grad ENABLED only to learn whether anything differentiable
            # reaches the state -- its graph is dropped three lines later -- and every other
            # primal pass runs with no graph at all. The returned derivative comes from the
            # ONE differentiable pass after the loop, never from these.
            with torch.set_grad_enabled(passes == 1 and torch.is_grad_enabled()):
                new, diag, _ = self._pass(
                    fed, drivers, dt, solve_kwargs, step_from=state, t=t,
                    boundary_transfers=boundary_transfers, produced=produced,
                )
                if passes == 1:
                    needs_adjoint = any(new[k].requires_grad for k in produced)
                    new = {k: v.detach() for k, v in new.items()}
                    diag = _detached(diag)
            if prev is not None:
                with torch.no_grad():
                    ok: Tensor | None = None
                    for name, tol in self.iterate_tol.items():
                        layer = self.transport[name]
                        d = (new[f"{name}.x"] - prev[f"{name}.x"]).abs()
                        d_s, _ = layer._to_stacked(d, layer.n_i, "x")
                        change[name] = d_s.amax(-1)
                        this = change[name] <= tol
                        ok = this if ok is None else (ok & this)
                    converged = ok
                if bool(converged.all()):
                    break
            fed = dict(new)
            if prev is not None:
                for name in self.transport:
                    key = f"{name}.x"
                    fed[key] = 0.5 * (prev[key] + new[key])
            prev = new
        if not bool(converged.all()):
            failing = (
                (~converged).nonzero().flatten().tolist() if converged.dim() else "all"
            )
            worst = {name: float(c.max()) for name, c in change.items()}
            message = (
                f"Model: coupling='iterate' did not converge within {self.iterate_max} "
                f"passes for instances {failing}; largest change per layer {worst}, "
                f"tolerances {self.iterate_tol}"
            )
            if needs_adjoint:
                raise RuntimeError(
                    message + "; a gradient was requested and a non-converged iteration has "
                    "no fixed point to differentiate, so on_failure='return' cannot return a "
                    "state here (it returns the primal state only when nothing requires a "
                    "gradient, e.g. under torch.no_grad() or with differentiable=False)"
                )
            if not (solve_kwargs.get("on_failure") == "return" and diagnostics is not None):
                raise RuntimeError(message)
        # A key `_pass` does not write is carried through it verbatim, pass after pass, so
        # its value in the returned state is the caller's own -- and so is its graph, which
        # the detached primal passes above replaced with a graph-free copy of the same
        # numbers. Put the ORIGINAL tensors back. They are not unknowns of the interface
        # (the docstring says why), so this restores a path the adjoint never covers and
        # changes no number.
        written_here = set(produced)
        carried = {k: v for k, v in state.items() if k not in written_here}
        adjoint: str | None = None
        adjoint_batched = False
        if needs_adjoint and bool(converged.all()):
            # `k in certified_fed` only guards the impossible: every produced key is in the
            # previous pass's output and so in the fed state of any pass after the first,
            # and convergence is only ever declared from pass 2 on.
            keys = [
                k for k in dict.fromkeys(produced)
                if k not in self.closure_state_keys and k in certified_fed
            ]
            out_keys: list[str] = []
            adjoint_diag: dict = {}

            def pass_fn(z: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
                f = dict(certified_fed)
                f.update(carried)
                f.update(zip(keys, z, strict=True))
                out, d, _ = self._pass(
                    f, drivers, dt, solve_kwargs, step_from=state, t=t,
                    boundary_transfers=boundary_transfers,
                )
                adjoint_diag.clear()
                adjoint_diag.update(d)
                # `out` is only READ here and below: `differentiate_fixed_point` hands its
                # outputs back as VIEWS of these tensors, and writing into one would mutate
                # the retained pass graph under autograd.
                out_keys[:] = list(out)
                z_next = [out[k] for k in keys]
                # Every entry recomputed by the pass, never a leaf of `z` handed back: the
                # keys were chosen for exactly that (see the docstring), and the one way it
                # could still happen -- a closure returning a driver it read straight out of
                # the fed state -- is named here rather than surfacing as a singular
                # `I - G_z` from inside the adjoint solve.
                # "<pot>.phi" and "<layer>.capacity" may legitimately sit in `keys` even
                # though the potential solve detaches its warm start and the capacity is
                # closure-derived: both are genuinely RECOMPUTED by this pass (a fresh
                # `phi0`-detached solve, a fresh closure call), so they contribute a zero
                # row/column here, not a unit one -- see the class docstring and
                # .superpowers/sdd/2026-09-20-framework-hardening-part-3/task-8-report.md for
                # why that is harmless and this guard is what would catch it if it stopped
                # being true.
                echoed = [k for k, n, t in zip(keys, z_next, z, strict=True) if n is t]
                if echoed:
                    raise RuntimeError(
                        f"Model: coupling='iterate' cannot differentiate its fixed point "
                        f"because the pass hands the state key(s) {echoed} back unchanged "
                        f"instead of recomputing them, which makes the interface Jacobian "
                        f"singular; a closure writing such a key must compute it rather "
                        f"than echo the state it was given"
                    )
                return [out[k] for k in out_keys], z_next

            # `converged`'s shape IS the batch shape (it is judged per instance over every
            # `iterate_tol` layer, never reduced further), so it names the leading dims a
            # genuinely batched state key carries; a key some instances share unbatched
            # fails that check and `differentiate_fixed_point` falls back to the flattened
            # solve on its own.
            adjoint_report: dict = {}
            flat = differentiate_fixed_point(
                [certified_fed[k] for k in keys], pass_fn, rtol=self.adjoint_rtol,
                where="Model coupling='iterate'",
                batch_shape=tuple(converged.shape), report=adjoint_report,
            )
            new = dict(zip(out_keys, flat, strict=True))
            # The differentiated pass's own diagnostics, DETACHED: see `_detached`.
            diag = _detached(adjoint_diag)
            adjoint = "implicit"
            adjoint_batched = bool(adjoint_report["batched"])
        else:
            new = {**new, **carried}
        if diagnostics is not None:
            diagnostics.update(
                {"passes": passes, "converged": converged, "max_change": change,
                 "layers": diag, "adjoint": adjoint, "adjoint_batched": adjoint_batched}
            )
        return new

    def residuals(self, state, drivers) -> dict[str, Tensor]:
        """Nodal balances at `state`: interior residual per potential layer, dx/dt per
        transport layer (spec 14). Zero (to solver tolerance) at a steady state.

        The transport balance is TRANSPORT ONLY: reactions are an operator splitting applied
        by `step` after the transport step, so they are outside the balance reported here,
        exactly as they are outside `steady`. A model with a reaction is therefore at zero
        residual at `steady`'s fixed point, not at the reaction's.

        A model owning a `CapacitatedTransferLayer` is REFUSED by name, for the same reason
        `_pass` refuses a steady (`dt=None`) pass: this method reports the balance whose zero
        `steady` converges to, and a clip/allocate layer is inherently discrete-time -- it has
        no steady meaning to report. Returning the other layers' residuals and silently
        omitting the capacitated one would be a balance over PART of the model presented as
        the model's, which is worse than no answer.
        """
        if self.capacitated:
            raise ValueError(
                f"Model: residuals() has no defined meaning for a model owning the "
                f"CapacitatedTransferLayer(s) {sorted(self.capacitated)}, which are "
                f"inherently discrete-time (same refusal as a steady, dt=None, pass); a "
                f"residual reported over the other layers alone would be a balance over "
                f"part of the model presented as the whole"
            )
        drv = self._apply_closures(state, drivers)
        out: dict[str, Tensor] = {}
        for name, layer in self.potential.items():
            phi = state[f"{name}.phi"]
            out[name] = layer.residual(
                phi[..., layer.interior], self._require(drv, f"{name}.phi_boundary"), drv,
                drv.get(f"{name}.sources"),
            )
        for name, layer in self.transport.items():
            xb = self._require(drv, f"{name}.x_boundary")
            x = state[f"{name}.x"]
            sources = drv.get(f"{name}.sources")
            if sources is None:
                sources = self._zero_sources(layer, x, layer.n_i)
            out[name] = layer.rate(
                x, self._kind_flows(name, state, drv), sources, xb,
                capacity=drv.get(f"{name}.capacity"),
            )
        return out

    def current_flows(self, name: str, state: State, drivers: Drivers) -> Tensor:
        """Transport layer `name`'s branch flows for THIS `state`/`drivers` -- what `step`
        would use just before advancing `name`, exposed for a caller (a coupling
        orchestrator) that needs them WITHOUT stepping the model. Closures run first, so a
        driver-prescribed layer's flow is available. A potential-owned layer's flows are the
        owner's solved `q`: read from `state["<owner>.q"]` when an earlier pass or step left
        one there, otherwise SOLVED here for these drivers, exactly as `_pass` would (the
        first pass of a step starts from a state that carries no `q` yet -- design spec A6).

        Two things to know about that branch. A `"<owner>.q"` already in `state` is used
        AS-IS, even when these `drivers` would solve to a different one: that is a previous
        outer step's output threaded back in, and for a coupling orchestrator it is the
        pass-1 initial guess, corrected from pass 2 onward once a step of this model has run
        at the current drivers. And the solve here takes the layer's DEFAULTS: `step`'s
        `**solve_kwargs` (tolerances, backend, `on_failure`) are not forwarded, because this
        method has no such argument -- a caller needing a specific solver setting should step
        the model and read `"<owner>.q"` out of the result.
        """
        drv = self._apply_closures(state, drivers)
        owner = self.flow_layer_of[name]
        if owner in self.potential and f"{owner}.q" not in state:
            key = self.flow_driver_of[name]
            if key in drv:
                # The same refusal `_kind_flows` makes on the read path, made here too: the
                # solve branch would otherwise silently prefer the potential layer and never
                # notice the contradicting driver, so whether a two-sourced layer is caught
                # would depend on whether the state happened to carry a `q` yet.
                raise ValueError(
                    f"Model: transport layer {name!r} takes its flows from potential layer "
                    f"{owner!r}, and the driver {key!r} was also given; a layer may have "
                    f"one source of flows, not two -- drop {key!r} or remove {owner!r}"
                )
            layer = self.potential[owner]
            phi_prev = state.get(f"{owner}.phi")
            phi0 = None if phi_prev is None else phi_prev[..., layer.interior]
            _phi, q = layer.solve(
                self._require(drv, f"{owner}.phi_boundary"), drv,
                drv.get(f"{owner}.sources"), phi0=phi0,
            )
            return layer.flows_of_kind(q, self.transport[name].flow_kinds)
        return self._kind_flows(name, state, drv)

    def ports(self, state) -> Ports:
        """Boundary nodes, the driver keys that prescribe them, and (potential layers) the net
        flow INTO each boundary node at `state`.

        Capacitated layers contribute NOTHING here, and that is deliberate rather than the
        same omission `residuals` refuses. `Ports` answers "which NODES may a coupled model
        prescribe, and through which key" -- a boundary-node partition with a prescribed
        potential or boundary composition. A `CapacitatedTransferLayer` has no such partition:
        every node is interior, its unbounded nodes (`s_max = inf`) are a storage property and
        not a prescribable port, and its one driver (`"<name>.requests"`) is per EDGE, not per
        node, so it has no well-defined entry in any of `Ports`' four node-keyed dicts. A
        coupled model driving such a layer writes that per-edge driver directly; exposing it
        here would need a fifth, edge-keyed field, which is a follow-up and not this
        milestone's (design spec section 4.3 changes `_pass` only). Unlike `residuals`,
        nothing here is silently wrong: the dicts are complete for every layer type `Ports`
        is about.
        """
        nodes: dict[str, list] = {}
        keys: dict[str, str] = {}
        flows: dict[str, Tensor] = {}
        for name, layer in self.potential.items():
            nodes[name] = [layer.net.nodes[i] for i in layer.bound.tolist()]
            keys[name] = f"{name}.phi_boundary"
            # _accumulate(q) is A q: net OUTflow at every node; negate for the inflow.
            flows[name] = -layer._accumulate(state[f"{name}.q"])[..., layer.bound]
        flow_keys: dict[str, str] = {}
        for name, layer in self.transport.items():
            nodes[name] = list(layer.boundary)
            keys[name] = f"{name}.x_boundary"
            flow_keys[name] = self.flow_driver_of[name]
        return Ports(boundary_nodes=nodes, prescribed_keys=keys, boundary_flows=flows,
                     flow_keys=flow_keys)
