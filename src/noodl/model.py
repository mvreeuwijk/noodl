"""Model: several physics layers on one typed graph, stepped together.

Per step: closures update the drivers
from the current state; every Kirchhoff layer is solved quasi-steadily; every allocation
layer takes its one explicit clip/allocate step; every transport layer is
advanced (sub-stepped if asked) on the flows of its kinds; reactions are applied, after
the transport step by default or before it with `reaction_order="before_transport"`.
`coupling="pingpong"` does that once per step with the state at the START of the step
(Hensen 1995); `coupling="iterate"` ("onion") repeats it until the named transport states stop
changing.

Keys. State: "<layer>.phi" (full-node order), "<layer>.q" (the layer's kind order),
"<layer>.x" (interior order, (n_i,) or (n_i, K)), "<layer>.s" (an allocation layer's
per-node storage, full-node order; that layer also writes its realised flows to
"<layer>.q"). Drivers: "<layer>.phi_boundary", "<layer>.x_boundary", optional
"<layer>.sources" (FULL-node order, zeros on boundary and inactive nodes), optional
"<layer>.capacity" (a transport layer's per-step capacity override; absent, the
layer's construction-time capacity stands; when given, a step or a steady solve records it
in the state it returns), optional "<layer>.removal" (a transport layer's
per-step first-order removal rate, 1/s, (..., n_i, K); absent, the layer's
construction-time removal stands), optional "<layer>.transmission" (a transport layer's
per-step transmission, (..., b_flow) or (..., b_flow, K) in its flow order; absent, the
layer's construction-time transmission stands), "<layer>.requests" (an allocation layer's
per-edge requested flow, required every step). Closures return driver updates; they may
not write state keys. A transport layer whose kinds no
Kirchhoff layer provides reads its branch flows from the driver "<layer>.q", in the layer's
own flow_kinds order; a layer with both a Kirchhoff-layer owner and that driver raises. A closure
may also declare `state_keys`, keys it carries across steps itself; those are
copied from its return into the returned state, evaluated from the step-start state in
every pass of a coupling that takes more than one -- never fed forward from an earlier
pass's own output, which would integrate a stateful closure once per pass instead of once
per step. A `coupling="iterate"` step started without such a key already in the step-start
state warns by name (`RuntimeWarning`): the pinning has nothing to pin on that step, so
the closure integrates once per pass instead of once per step until the key is seeded.

`**solve_kwargs` of `step`/`steady` reach the POTENTIAL solves only (`differentiable`,
`on_failure`, `method`, Newton kwargs). The keywords the Model sets itself (`phi0`, `q0`,
`history`, `dt`, `t`) are refused there by name. `layer_options` gives one Kirchhoff layer
its own keywords on top of them (`{"thermal": {"atol": 1e-11}}`: a heat balance whose
round-off floor is above the electrical layer's tolerance). Transport steps always raise on failure.

Differentiability: ping-pong is one pass of differentiable operations; iterate carries no
graph on its primal passes and attaches the IMPLICIT adjoint of its converged interface to
one extra pass instead of unrolling them, so the gradient is the landed fixed point's rather
than the truncated iteration's (the convergence decision is made on detached copies, as
Newton's mask is). Failure follows the layers: raise by default, naming the offender.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from noodl._closures import ClosureSchedule
from noodl._fixed_point_loop import (
    Verdict,
    confirm_gradient_dependence,
    detached,
    run_passes,
    unsolved_mask,
)
from noodl._model_setup import (
    CouplingOptions,
    flow_owners,
    register_layers,
    validate_modes,
    validate_reactions,
    validate_substeps,
)
from noodl._pass import (
    LAYER_OPTIONS,
    circuit_kwargs,
    kind_flows,
    require,
    run_pass,
    zero_sources,
)
from noodl.contracts import AlgebraicClosure, ClosureSpec, IntegratingClosure, StepContext
from noodl.layers.allocation import AllocatedFlowLayer
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import Reaction
from noodl.layers.transport import TransportLayer
from noodl.solvers.anderson import _AndersonHistory
from noodl.solvers.errors import ConvergenceError
from noodl.solvers.fixed_point import differentiate_fixed_point
from noodl.topology import Network

Tensor = torch.Tensor
State = dict[str, Tensor]
Drivers = dict[str, Tensor]
_STATE_SUFFIXES = ("phi", "q", "x", "s")
#: Potential-solve keywords the Model sets itself from the state, the step and `t`; refused
#: among `**solve_kwargs` of `step`/`steady`.
_MODEL_SOLVE_KEYS = frozenset({"phi0", "q0", "history", "dt", "t"})
# The key `layer_options` travels under inside `solve_kwargs`, to `_pass`'s layer solves.
_LAYER_OPTIONS = LAYER_OPTIONS


#: Kept for import compatibility: the closure Protocols live in `noodl.contracts`.
Closure = AlgebraicClosure


@dataclass(frozen=True)
class Ports:
    """What a coupled or external model may prescribe and read."""

    boundary_nodes: dict[str, list]
    prescribed_keys: dict[str, str]
    boundary_flows: dict[str, Tensor]
    # Transport layer -> the driver key its branch flows are read from. Present for EVERY
    # transport layer, whether or not a Kirchhoff layer owns its kinds: a coupled model that
    # wants to drive a layer's flows needs the key, and a model that wants to know whether
    # it MAY do so reads `Model.flow_layer_of[name] is None`. Defaulted so that any existing
    # construction of `Ports` keeps working.
    flow_keys: dict[str, str] = field(default_factory=dict)



def _with_layer_options(solve_kwargs: dict, layer_options) -> dict:
    """`solve_kwargs` carrying `layer_options` (a plain dict per layer) to `_pass`."""
    if not layer_options:
        return solve_kwargs
    return {**solve_kwargs, _LAYER_OPTIONS: {k: dict(v) for k, v in layer_options.items()}}


class Model:
    """Layers on one `Network`, stepped together. See the module docstring for the keys.

    Two couplings (Hensen 1995), chosen with `coupling`:

    `"pingpong"` (the default) takes exactly ONE pass per step -- closures, potential solves,
    allocation steps, transport steps, reactions -- with the state at the START of the step
    (the allocation step sits between the potential solves and the transport steps so that a
    transport layer reading `"<layer>.q"` sees a freshly written flow whichever kind of layer
    wrote it; see `_pass`). It is cheap and it is what a weakly coupled model wants; its
    splitting error is first order in `dt`.

    `reaction_order` places the reactions in that split: `"after_transport"` (the default,
    T then R) returns the reacted state; `"before_transport"` (R then T) reacts the
    step-start state and returns the transported one, which is where SWMM samples its
    water quality (`apps.sewer` builds with it). The step sequence is the same either way;
    only the sampling point differs.

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
    gradient was wanted).

    A primal pass is one iterate: a potential solve that does not converge within a pass
    (Newton's `max_iter`) does not end the step. It returns its last Newton iterate, the pass
    is judged not converged, and the iteration goes on from it; only the iteration's own
    verdict raises (or accepts, below). When the last pass's potential solve also stopped
    short, the error is `noodl.solvers.ConvergenceError` (a `RuntimeError`), so a caller can
    retry the step from another starting point (`warm_start=False`) and let every other
    failure through. A step that is going to raise stops early, before `iterate_max`, once
    every unconverged instance's potential solve has failed again from a fed state within
    the tolerances of the previous pass's: the next pass would only repeat it. A potential
    solve that is differentiated still raises (`KirchhoffLayer.solve`'s
    ``on_failure="return_primal"``), and a caller's own `on_failure` is passed through
    unchanged.

    `iterate_on_limit="accept"` (default ``"raise"``) makes a batch that does not converge
    within `iterate_max` return its last pass instead of raising, as SWMM accepts a step
    that used up its MAX_TRIALS (`dynwave.c`, counted as non-converging), with
    ``diagnostics["converged"]`` False for those instances. A gradient is still refused
    there: there is no fixed point to differentiate.

    `iterate_flow_tol` (optional; per Kirchhoff layer named in `iterate_tol`, absolute, in
    flow units) also judges that layer's flows: a pass converges only when its returned
    `.q` is within it of the `.q` it was fed. `iterate_acceleration="anderson"` (opt-in)
    replaces the relaxation of those layers' `.phi` and `.q` by Anderson mixing over the last
    `iterate_memory` passes and requires `iterate_flow_tol` for each of them; it changes the
    sequence of fed states, never the fixed point or its adjoint (`_iterate`,
    `_AndersonHistory`).

    The onion's fixed point is differentiated IMPLICITLY, not by unrolling the passes: the
    primal passes carry no graph, and after convergence the certified pass runs once more on
    the graph with `solvers.fixed_point.differentiate_fixed_point` attaching the adjoint of
    the interface equations (`adjoint_rtol`, the GMRES tolerance of that one small solve).
    The returned gradient is the derivative of the fixed point the iteration LANDED on, so
    what remains of the pass count and of `iterate_tol` in it is only that the landing point
    is a fixed point to within `iterate_tol` -- an O(primal residual) error, against the
    unrolled O(contraction^passes) truncation it replaces, and exactly zero where the
    interface is linear. Backward memory is one pass rather than all of them.
    `diagnostics["adjoint"]` says whether that pass ran: `"implicit"` when it did, `None`
    when nothing differentiable reached the state (or the iteration did not converge) and no
    extra pass was needed. `diagnostics["adjoint_batched"]` says whether that solve ran as one
    independent GMRES system per batch instance (every state key the pass recomputes carrying
    the batch as its leading dims) or, when some key does not (a value shared across
    instances), as today's single system flattened over the whole interface; `False` when
    `"adjoint"` is `None` too.

    In grad mode, a step whose first pass finds nothing differentiable pays one more,
    uncounted pass at the certified point to confirm it (a closure may read a parameter only
    on the branch the fixed point lies on, and no inspection can rule that out). A
    forward-only run should therefore use `torch.no_grad()`, which skips that pass.
    """

    def __init__(
        self,
        net: Network,
        layers: Mapping[str, KirchhoffLayer | TransportLayer | AllocatedFlowLayer]
        | Sequence[KirchhoffLayer | TransportLayer | AllocatedFlowLayer],
        closures: Sequence[AlgebraicClosure | IntegratingClosure] = (),
        reactions: Sequence[tuple[str, Reaction]] = (),
        coupling: str = "pingpong",
        iterate_tol: Mapping[str, float] | None = None,
        iterate_max: int = 20,
        substeps: Mapping[str, int] | None = None,
        adjoint_rtol: float = 1e-10,
        iterate_relaxation: float = 0.5,
        reaction_order: str = "after_transport",
        iterate_acceleration: str | None = None,
        iterate_memory: int = 5,
        iterate_flow_tol: Mapping[str, float] | None = None,
        iterate_on_limit: str = "raise",
    ) -> None:
        validate_modes(coupling, reaction_order)
        self.reaction_order = reaction_order
        self.net = net
        registered = register_layers(net, layers)
        self.layers = registered.layers
        # Set only while `noodl.validation.check_setup` probes a step (read tracking).
        self._probe = None
        self._refs = None
        self.kirchhoff: dict[str, KirchhoffLayer] = registered.kirchhoff
        self.transport: dict[str, TransportLayer] = registered.transport
        self.allocation: dict[str, AllocatedFlowLayer] = registered.allocation
        # Computed here, before the closure_state_keys check below, which consults
        # `self.flow_layer_of`.
        self.flow_layer_of: dict[str, str | None]
        self.flow_driver_of: dict[str, str]
        self.flow_layer_of, self.flow_driver_of = flow_owners(
            self.kirchhoff, self.transport, self.allocation)
        self.closures = list(closures)
        # The closures' specs, their construction checks and `closure_state_keys`
        # (`noodl._closures.ClosureSchedule`); `_apply_closures` runs them.
        self._schedule = ClosureSchedule(
            self.closures, layers=self.layers, transport=self.transport,
            flow_layer_of=self.flow_layer_of,
        )
        self.closure_state_keys: dict[str, Callable] = self._schedule.closure_state_keys
        self.reactions = validate_reactions(self.transport, reactions)
        self._schedule.check_species_columns(self.transport, self.reactions)
        options = CouplingOptions.validate(
            coupling=coupling, iterate_tol=iterate_tol, iterate_max=iterate_max,
            iterate_relaxation=iterate_relaxation, iterate_acceleration=iterate_acceleration,
            iterate_memory=iterate_memory, iterate_on_limit=iterate_on_limit,
            adjoint_rtol=adjoint_rtol, iterate_flow_tol=iterate_flow_tol,
            kirchhoff=self.kirchhoff, transport=self.transport,
        )
        self.coupling = options.coupling
        self.iterate_tol = options.iterate_tol
        self.iterate_max = options.iterate_max
        self.iterate_relaxation = options.iterate_relaxation
        self.iterate_acceleration = options.iterate_acceleration
        self.iterate_memory = options.iterate_memory
        self.iterate_on_limit = options.iterate_on_limit
        self.adjoint_rtol = options.adjoint_rtol
        self.iterate_flow_tol = options.iterate_flow_tol
        self._schedule.check_coupling(coupling, options.iterate_flow_tol)
        options.check_acceleration(self.kirchhoff)
        self.substeps = validate_substeps(self.transport, substeps)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _require(drivers: Mapping[str, Tensor], key: str) -> Tensor:
        """`drivers[key]`, refused by name when it is missing (`noodl._pass.require`)."""
        return require(drivers, key)

    def _is_layer_state_key(self, key: str) -> bool:
        """True for a layer's own state key `"<layer>.phi/q/x/s"` a closure may not write:
        all of them but `"<layer>.q"` of a transport layer no Kirchhoff layer owns, which is
        the flow DRIVER a flow closure writes."""
        return self._schedule.is_layer_state_key(key)

    def _closure_spec(self, closure) -> ClosureSpec:
        """The `ClosureSpec` of one of `self.closures`, found by identity. A closure added to
        `closures` after construction is read (and cached) on first use."""
        return self._schedule.spec(closure, self.closures)

    @property
    def _integrating(self) -> set[int]:
        """`id()`s of the integrating closures (read-only; for tests and diagnostics)."""
        return self._schedule.integrating(self.closures)

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
        actually wrote THIS call. A same-named entry already sitting in `drivers`
        (the caller's own, unrelated to any closure) must not be mistaken for a closure
        having written its declared state key -- `key in drv` alone cannot tell the two
        apart, since `drv` starts as a copy of `drivers`.

        `ctx` is threaded to an INTEGRATING closure only (`closure.integrates = True`):
        `StepContext(dt)` from `_pass` inside `step`, `StepContext(dt=None)` inside `steady`
        (refused here by name -- an integrating closure has no interval to integrate over),
        and `None` from a query (`residuals`, `current_flows`, `ports`, the sewer report),
        where an integrating closure evaluates its algebraic outputs at the given state
        without advancing. A non-integrating closure never sees `ctx` at all, so its call
        signature and behaviour are unaffected.
        """
        return self._schedule.apply(self.closures, state, drivers, written=written, ctx=ctx,
                                    probe=self._probe)

    def _zero_sources(self, layer: TransportLayer, like: Tensor, n_like: int) -> Tensor:
        """Full-node zeros in `like`'s LAYOUT (reduced `(n,)` or stacked `(n, K)`).

        `like` decides the layout, so it must be a tensor whose own layout is the one the
        caller wants back: the state `x` where there is one (`step`, `residuals`, and
        `steady` whenever the state carries `"<layer>.x"`). `TransportLayer.steady` reads its
        `reduced` flag off the SOURCES rather than off `x`, so a `(n_b, K)`-shaped
        `x_boundary` used as `like` would flip the returned state to `(..., n_i, K)` for a
        single-species layer; only fall back to it when there is no `x` at all.
        """
        return zero_sources(layer, like, n_like, self.net.n)

    def _kind_flows(self, name: str, state: Mapping[str, Tensor],
                    drivers: Mapping[str, Tensor]) -> Tensor:
        """This transport layer's branch flows, in its own `flow_kinds` order.

        From the owning Kirchhoff layer's solved `q` when there is one, otherwise from the
        driver `"<layer>.q"`. A layer with BOTH is a contradiction -- two different answers
        for the same flows -- and is refused by name rather than resolved by a precedence
        rule nobody would remember.

        `__init__`'s ownership scan also lets an `AllocatedFlowLayer` become a
        transport layer's flow owner (it writes `"<name>.q"` in the same key convention).
        Reading those flows would need `AllocatedFlowLayer.flows_of_kind`, which is
        not built (species/quality transport on allocated flows):
        refused here by name rather than left to raise a bare `KeyError` off
        `self.kirchhoff[owner]`.
        """
        return kind_flows(self, name, state, drivers)

    @staticmethod
    def _circuit_kwargs(name: str, layer: KirchhoffLayer, state: Mapping[str, Tensor],
                        base: Mapping[str, Tensor], dt: float | None,
                        t: float | None) -> dict:
        """The solve keywords of a layer solved by Modified Nodal Analysis.

        `q0`: the warm start of its implicit-law flows, from the fed state's `"<name>.q"`.
        `t`: the time its laws are evaluated at -- the END of a step (`t + dt`, the time
        the solved state belongs to), the given time of a steady solve, `None` without one.
        On a step of a layer with dynamic laws (not a phasor layer, which takes no step),
        `history` is the STEP-START state's `(phi, q)` -- `base`, never the fed state, so an
        iterated coupling re-steps every pass from the same start -- and `dt` the step.
        """
        return circuit_kwargs(name, layer, state, base, dt, t)

    # -------------------------------------------------------------------- pass
    def _pass(
        self, state, drivers, dt, solve_kwargs, *, step_from: State | None = None,
        t: float | None = None, boundary_transfers: bool | Collection[str] = False,
        produced: list[str] | None = None, warm_start: bool = True,
    ) -> tuple[State, dict, Drivers]:
        """One closures -> potential -> allocation -> transport -> reactions pass (reactions
        before transport under `reaction_order="before_transport"`);
        `dt=None` means steady (and is refused outright by a model owning an allocation
        layer, which is inherently discrete-time).

        `t` is the caller's own start-of-step time, if it tracks one; both `dt` and `t` are
        carried to every closure as a `StepContext`, so an INTEGRATING closure reads the
        model's own clock rather than an interval fixed at its own construction.

        `state` is what the closures read and what the potential solves warm-start from
        (`warm_start=False`: they start from their layer's linear initial guess instead,
        while the closures still read `state`'s potentials).
        `step_from` is the state a transport STEP starts at, which is a different thing
        inside an iterated coupling: pass k there re-advances the SAME time step from the
        state at the start of it, with only the closures' view of the new state updated.
        It defaults to `state`, which is what a single ping-pong pass wants.

        `boundary_transfers`: `True` runs `step_with_transfer` on every transport layer; a
        collection of layer names runs it on ONLY those (every other transport layer takes
        the plain, cheaper `step`); `False` (or an empty collection) runs it on none. Only
        a layer actually asked for gets `diag[name]["boundary_transfer"]` -- a caller that
        names one linked layer out of several must not pay `step_with_transfer`'s extra cost
        (no diagonal shift on the `exact` scheme's Taylor accumulator) on layers whose
        transfer nothing reads.

        `diag[name]["linear"]`, when this transport layer's `scheme` runs a
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
        return run_pass(
            self, state, drivers, dt, solve_kwargs, step_from=step_from, t=t,
            boundary_transfers=boundary_transfers, produced=produced, warm_start=warm_start,
        )

    # ------------------------------------------------------------------- setup
    @property
    def capacitated(self) -> dict[str, AllocatedFlowLayer]:
        """Pre-rename name of `allocation`, kept as an alias (the same dict)."""
        return self.allocation

    @property
    def refs(self):
        """`noodl.refs.LayerRefs`: every registered layer as a `LayerRef` whose attributes
        are its state and driver keys, each knowing its node order, shape and unit
        (`model.refs.thermal.sources`, `model.refs["air"].phi_boundary`). Built on first
        use; `model.refs.describe()` lists everything a run needs."""
        if self._refs is None:
            from noodl.refs import LayerRefs

            self._refs = LayerRefs(self)
        return self._refs

    def drivers_from(self, values: Mapping, *, base: Mapping | None = None) -> Drivers:
        """A drivers dictionary from `{key: {node: value}}` / `{key: tensor}`; see
        `noodl.refs.drivers_from`."""
        from noodl.refs import drivers_from

        return drivers_from(self, values, base=base)

    def state_from(self, values: Mapping, *, base: Mapping | None = None) -> State:
        """A state dictionary from `{key or layer: {node: value}}`; see
        `noodl.refs.state_from`."""
        from noodl.refs import state_from

        return state_from(self, values, base=base)

    def check(self, state: Mapping | None = None, drivers: Mapping | None = None, *,
              dt: float | None = None, probe: bool = False, steady: bool = False,
              names=None):
        """A `noodl.validation.SetupReport` on this model and, when given, its state and
        drivers: required inputs, orders and shapes, units, unused edge kinds and suspicious
        keys. `probe=True` runs one step of `dt` (or a steady solve) to learn which custom
        keys are read. `names=` (from `Component.flatten`) also lists unconnected ports.
        Nothing is raised; call `.raise_for_errors()` on the report."""
        from noodl.validation import check_setup

        return check_setup(self, state, drivers, dt=dt, probe=probe, steady=steady,
                           names=names)

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
        boundary_transfers: bool | Collection[str] = False, warm_start: bool = True,
        layer_options: Mapping[str, Mapping] | None = None,
        **solve_kwargs,
    ):
        """Advance one step (ping-pong or iterated coupling, per `self.coupling`).

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
        ARE NOT PART OF THE REPORTED TRANSFER: after transport (the default) they act on the
        transported state; under `reaction_order="before_transport"` the transfer is that
        of transporting the already-reacted step-start state.

        Naming only the layer(s) actually linked to a coupling matters for cost: `True`
        forces `step_with_transfer` (no diagonal shift on the `exact` scheme's Taylor
        accumulator) on every transport layer of the model, including ones nothing reads a
        transfer from -- a coupled recipient with an unlinked `exact`-scheme thermal layer
        pays that cost on every one of its substeps for no benefit.

        `warm_start` (default True) starts each Kirchhoff layer's nonlinear solve from the
        state's `"<layer>.phi"` (and, on a layer solved by Modified Nodal Analysis, its
        implicit-law flows from `"<layer>.q"`). `False` starts it from the layer's linear
        initial guess instead, while closures and node sources still read `"<layer>.phi"`
        as an input (a retry after a warm start failed to converge, without losing that
        input).

        `layer_options` maps a Kirchhoff layer's name to keywords for that layer's solve
        only, applied over `**solve_kwargs` (and so over the layer's own `newton_defaults`):
        `{"thermal": {"atol": 1e-11}}`. A name that is not a Kirchhoff layer of this model,
        or a keyword the Model sets itself, is refused by name.
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
            state, drivers, float(dt), diagnostics,
            _with_layer_options(solve_kwargs, layer_options), t=t,
            boundary_transfers=boundary_transfers, warm_start=warm_start,
        )

    def steady(self, state, drivers, *, t: float | None = None,
               diagnostics: dict | None = None, warm_start: bool = True,
               layer_options: Mapping[str, Mapping] | None = None, **solve_kwargs):
        """The quasi-steady state of every layer at `drivers` (transport layers solved to
        `rate == 0` rather than advanced).

        `t` (optional) is the time of that state: closures receive it in their
        `StepContext`, and the constitutive laws of a layer solved by Modified Nodal
        Analysis are evaluated at it (a time-dependent source). A layer with dynamic laws is
        solved at its operating point: a capacitance open, an inductance shorted.

        REACTIONS ARE NOT APPLIED. They are an operator splitting applied around a transport
        step (after it, or before it under `reaction_order="before_transport"`), so they
        belong to `step` alone: a model carrying a reaction has a `steady` that
        is the fixed point of transport only, not of transport-plus-reaction (`residuals`
        reports the same balance). Deliberate, and pinned by a test.

        `dt=None` here means the `StepContext` an INTEGRATING closure receives carries
        `dt=None` too, and `_apply_closures` refuses that by the closure's own name: a
        closure that integrates some state over an interval has no interval to integrate over
        at a quasi-steady state. `warm_start` and `layer_options` are `step`'s.
        """
        return self._advance(state, drivers, None, diagnostics,
                             _with_layer_options(solve_kwargs, layer_options), t=t,
                             warm_start=warm_start)

    def _advance(
        self, state, drivers, dt, diagnostics, solve_kwargs, *, t=None,
        boundary_transfers: bool | Collection[str] = False, warm_start: bool = True,
    ) -> State:
        options = solve_kwargs.get(_LAYER_OPTIONS, {})
        unknown = sorted(set(options) - set(self.kirchhoff))
        if unknown:
            raise ValueError(
                f"Model: layer_options names {unknown}, not Kirchhoff layer(s) of this model "
                f"(has {sorted(self.kirchhoff)})"
            )
        for keys in (solve_kwargs, *options.values()):
            owned = sorted(_MODEL_SOLVE_KEYS.intersection(keys))
            if owned:
                raise ValueError(
                    f"Model: solve keywords {owned} are set by the Model itself, not passed "
                    f"through: phi0 and q0 (the warm starts) from the state, history from the "
                    f"step-start state, dt from step's dt and t from the t= argument"
                )
        # The coupling seam: ping-pong is exactly ONE pass, taken with the state at the start
        # of the step; `coupling="iterate"` repeats `_pass` until the named transport states
        # stop changing, and reports the pass count it took.
        if self.coupling == "pingpong":
            new, diag, _ = self._pass(
                state, drivers, dt, solve_kwargs, t=t, boundary_transfers=boundary_transfers,
                warm_start=warm_start,
            )
            if diagnostics is not None:
                diagnostics.update({"passes": 1, "layers": diag})
            return new
        return self._iterate(
            state, drivers, dt, diagnostics, solve_kwargs, t=t,
            boundary_transfers=boundary_transfers, warm_start=warm_start,
        )

    def _iterate(
        self, state, drivers, dt, diagnostics, solve_kwargs, *, t=None,
        boundary_transfers: bool | Collection[str] = False, warm_start: bool = True,
    ) -> State:
        """Hensen's onion: repeat the pass until the named transport states stop changing.

        The relaxation takes a pass to start: `prev` is None after pass 1, so pass 2 is fed
        pass 1's transport states UNRELAXED, and pass 3 is the first fed a mean. From there
        the "<layer>.x" state fed to the closures on pass k is `x_{k-2} + w (x_{k-1} -
        x_{k-2})`, `w = iterate_relaxation`: by default 0.5, the mean of passes k-2 and k-1
        (Hensen 1995, successive substitution with 0.5 relaxation). CLOSURE-CARRIED
        state (`closure_state_keys`) is the one exception to that relaxation: it
        is evaluated from the STEP-START state on every pass, never fed forward from the
        previous pass's own output -- a closure that integrates (a sewer manhole's
        storage sweep, a tank level) would otherwise advance once per PASS instead of once
        per STEP. The returned state is the LAST pass's own output, never a relaxed one.
        Convergence is judged per instance on detached copies, and a pass whose potential
        solve stopped short of its tolerance (``on_failure="return_primal"``, the default of
        the primal passes) is not converged whatever its change.

        A Kirchhoff layer named in `iterate_tol` is iterated on its own potential: it
        converges when the largest change of `phi` over its interior and boundary nodes is
        within the tolerance (in `phi` units). This is for closures that compute a layer's
        element coefficients, or its boundary potentials, from that layer's previous
        solution (the sewer dynamic wave: conduit coefficients and outfall heads). Its
        `.phi` and `.q` are relaxed like a transport `.x`, and relaxation does not move
        the fixed point. With `iterate_flow_tol` its flows are judged too, on the certified
        pass's returned `.q` against its fed `.q`.

        With ``iterate_acceleration="anderson"`` those `.phi` and `.q` are Anderson-mixed
        instead (`_AndersonHistory`: the history, the safeguards and the per-instance rules),
        and the convergence test also asks the fixed-point residual of `.phi` (returned
        against fed) to be within the tolerance: two passes fed nearly the same mixed state
        can agree with each other while neither is a fixed point. Mixing chooses only the fed
        states; the adjoint below is unchanged.

        Differentiation: the primal passes carry no graph. After convergence the certified
        pass runs once more on the graph at the SAME fed state and
        `solvers.fixed_point.differentiate_fixed_point` attaches the implicit adjoint of the
        interface equations, so the returned gradient is the FIXED POINT's own and not the
        truncated iteration's. What remains of the pass count and of `iterate_tol` in
        it is only that the landing point is a fixed point to within `iterate_tol`: the error
        is O(primal residual) rather than the unrolled O(contraction^passes), and it is
        exactly zero where the interface is linear. Backward memory is one pass. Whether
        anything differentiable reaches the state is a question no inspection of `state` and
        `drivers` can answer, since a differentiable parameter may be captured inside an
        element, a drive or a closure and never appear in either; it is asked by running a
        pass on the graph and dropping that graph at once. Pass 1 asks first; when it finds
        nothing, the question is asked again at the certified fed state, because a closure
        may take a parameter-dependent branch only once the state has moved away from the
        step start, and the answer that matters is the one at the point whose derivative
        is returned. That extra pass (not counted in `passes`) runs whenever grad mode is
        on, since no inspection can prove that nothing a pass reads requires grad: a
        forward-only simulation should run under `torch.no_grad()`, which skips it. A run
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
        * Closure-carried state is pinned to the STEP-START state on every pass, so a
          pass does not read it from the fed state and it carries no feedback at all; and it
          is the one key a closure may legitimately return unchanged, which would be that
          same unit row. Its gradient path survives regardless, because the differentiable
          pass reads it from `state` itself (`step_from=state`), graph and all -- as do the
          transport and allocation steps, which advance from `state` for the same reason.
          (A closure-carried key MISSING from the step-start state is the one case the
          pinning does not cover; there the pass reads the previous pass's value, and holding it
          fixed here drops a path that only exists because that key was not seeded in the first
          place.)

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
                f"step start (only keys the state carries are pinned), so its closure "
                f"integrates once per pass instead of once per step and the adjoint omits its "
                f"compounding path. Seed it before the first step (the application's "
                f"initial_state helper, or Model.initial_capacities for capacities).",
                RuntimeWarning,
                stacklevel=4,
            )
        # A primal pass is one iterate of the outer fixed point: a potential solve that does
        # not converge within it returns its last Newton iterate (on the primal path only,
        # "return_primal") and the pass is judged not converged, rather than ending the
        # step. A caller's own `on_failure` is kept as given.
        pass_kwargs = dict(solve_kwargs)
        pass_kwargs.setdefault("on_failure", "return_primal")
        raises_on_limit = self.iterate_on_limit != "accept" and not (
            solve_kwargs.get("on_failure") == "return" and diagnostics is not None
        )
        problem = _ModelPasses(
            self, state, drivers, dt, pass_kwargs, t=t, boundary_transfers=boundary_transfers,
            warm_start=warm_start, raises_on_limit=raises_on_limit,
        )
        # The shared primal loop (`noodl._fixed_point_loop.run_passes`). Pass 1 runs with
        # grad ENABLED to learn early whether anything differentiable reaches the state --
        # its graph is dropped right after -- and every other primal pass runs with no graph
        # at all. The returned derivative comes from the ONE differentiable pass below,
        # never from these.
        outcome = run_passes(problem, dict(state), max_passes=self.iterate_max)
        passes, stopped_early = outcome.passes, outcome.stopped_early
        # The fed state of the last pass run: the certified one on the converged exit.
        certified_fed: State = outcome.certified
        new, diag = outcome.out
        produced = problem.produced
        change = problem.change
        inner_failed = problem.inner_failed
        # `iterate_max >= 2` and a non-empty `iterate_tol` are both refused at construction,
        # so the loop cannot end without a real per-instance verdict of the right shape,
        # dtype and device; the 0-dim placeholder is never what is returned.
        converged: Tensor = (
            outcome.verdict.converged if outcome.verdict is not None
            else torch.zeros((), dtype=torch.bool)
        )
        needs_adjoint = outcome.needs_adjoint
        if bool(converged.all()) or not raises_on_limit:
            # Pass 1 saw nothing differentiable, but it ran at the START state: a closure
            # that takes a parameter-dependent branch only once the state has moved would
            # have the returned state silently detached. Ask again, once, under grad at the
            # state the certified (or, without convergence, the last) pass was fed -- the
            # point whose derivative is returned -- and drop that pass's output and graph.
            # Skipped by a run about to raise for non-convergence anyway, and under
            # `torch.no_grad()`.
            needs_adjoint = confirm_gradient_dependence(problem, outcome)
        if not bool(converged.all()):
            failing = (
                (~converged).nonzero().flatten().tolist() if converged.dim() else "all"
            )
            worst = {name: float(c.max()) for name, c in change.items()}
            unsolved = inner_failed is not None and bool(inner_failed.any())
            message = (
                "Model: coupling='iterate' did not converge within "
                + (f"{passes} passes (of at most {self.iterate_max}; stopped there, as the "
                   f"potential solve failed again from a state within the tolerances of the "
                   f"previous pass's)" if stopped_early else f"{self.iterate_max} passes")
                + f" for instances {failing}; largest change per key {worst}, "
                f"tolerances {self.iterate_tol}"
                + (f", flow tolerances {self.iterate_flow_tol}" if self.iterate_flow_tol else "")
                + ("; the last pass's potential solve did not converge either"
                   if unsolved else "")
            )
            # A potential solve that stopped at its iteration limit is non-convergence, which
            # a caller may retry from another starting point (`simulate` retries the airflow
            # solve from the linear initial guess); `ConvergenceError` is a `RuntimeError`,
            # so a caller catching that is unaffected.
            error = ConvergenceError if unsolved else RuntimeError
            if needs_adjoint:
                raise error(
                    message + "; a gradient was requested and a non-converged iteration has "
                    "no fixed point to differentiate, so on_failure='return' cannot return a "
                    "state here (it returns the primal state only when nothing requires a "
                    "gradient, e.g. under torch.no_grad() or with differentiable=False)"
                )
            if raises_on_limit:
                raise error(message)
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
            # A phasor layer's complex `.phi`/`.q` enter the adjoint in their real view
            # (real and imaginary parts as a trailing pair): its GMRES solve is real-valued.
            complex_keys = {k for k in keys if certified_fed[k].is_complex()}

            def pass_fn(z: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
                f = dict(certified_fed)
                f.update(carried)
                z = [torch.view_as_complex(v.contiguous()) if k in complex_keys else v
                     for k, v in zip(keys, z, strict=True)]
                f.update(zip(keys, z, strict=True))
                out, d, _ = self._pass(
                    f, drivers, dt, solve_kwargs, step_from=state, t=t,
                    boundary_transfers=boundary_transfers, warm_start=warm_start,
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
                # row/column here, not a unit one -- see the class docstring for why that
                # is harmless and this guard is what would catch it if it stopped being
                # true.
                echoed = [k for k, n, t in zip(keys, z_next, z, strict=True) if n is t]
                if echoed:
                    raise RuntimeError(
                        f"Model: coupling='iterate' cannot differentiate its fixed point "
                        f"because the pass hands the state key(s) {echoed} back unchanged "
                        f"instead of recomputing them, which makes the interface Jacobian "
                        f"singular; a closure writing such a key must compute it rather "
                        f"than echo the state it was given"
                    )
                z_next = [torch.view_as_real(n.resolve_conj()) if k in complex_keys else n
                          for k, n in zip(keys, z_next, strict=True)]
                return [out[k] for k in out_keys], z_next

            # `converged`'s shape IS the batch shape (it is judged per instance over every
            # `iterate_tol` layer, never reduced further), so it names the leading dims a
            # genuinely batched state key carries; a key some instances share unbatched
            # fails that check and `differentiate_fixed_point` falls back to the flattened
            # solve on its own. `batch_shape` also asserts (see that function's docstring)
            # that instances do not couple through `pass_fn` -- true here because `_pass`
            # advances every state key and layer per instance and `converged` is itself
            # judged per instance, so nothing in the pass ever mixes one instance's state
            # into another's.
            adjoint_report: dict = {}
            flat = differentiate_fixed_point(
                [torch.view_as_real(certified_fed[k].resolve_conj()) if k in complex_keys
                 else certified_fed[k] for k in keys], pass_fn, rtol=self.adjoint_rtol,
                where="Model coupling='iterate'",
                batch_shape=tuple(converged.shape), report=adjoint_report,
            )
            new = dict(zip(out_keys, flat, strict=True))
            # The differentiated pass's own diagnostics, DETACHED: see `detached`.
            diag = detached(adjoint_diag)
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

    def _unsolved(self, diag: Mapping) -> Tensor | None:
        """Per instance, whether any potential solve of a pass (its `diag`) stopped short of
        its own tolerance (`on_failure="return_primal"`); ``None`` when every one converged."""
        return unsolved_mask(diag, self.kirchhoff)

    def residuals(self, state, drivers) -> dict[str, Tensor]:
        """Nodal balances at `state`: interior residual per Kirchhoff layer, dx/dt per
        transport layer. Zero (to solver tolerance) at a steady state.

        The transport balance is TRANSPORT ONLY: reactions are an operator splitting applied
        by `step` around the transport step, so they are outside the balance reported here,
        exactly as they are outside `steady`. A model with a reaction is therefore at zero
        residual at `steady`'s fixed point, not at the reaction's.

        A model owning an `AllocatedFlowLayer` is REFUSED by name, for the same reason
        `_pass` refuses a steady (`dt=None`) pass: this method reports the balance whose zero
        `steady` converges to, and a clip/allocate layer is inherently discrete-time -- it has
        no steady meaning to report. Returning the other layers' residuals and silently
        omitting the allocation one would be a balance over PART of the model presented as
        the model's, which is worse than no answer.
        """
        if self.allocation:
            raise ValueError(
                f"Model: residuals() has no defined meaning for a model owning the "
                f"AllocatedFlowLayer(s) {sorted(self.allocation)}, which are "
                f"inherently discrete-time (same refusal as a steady, dt=None, pass); a "
                f"residual reported over the other layers alone would be a balance over "
                f"part of the model presented as the whole"
            )
        drv = self._apply_closures(state, drivers)
        out: dict[str, Tensor] = {}
        for name, layer in self.kirchhoff.items():
            phi = state[f"{name}.phi"]
            if layer.mna:
                # Implicit-law flows are unknowns, not functions of `phi`: balance the
                # state's own flows.
                out[name] = layer.balance(phi, state[f"{name}.q"], drv.get(f"{name}.sources"),
                                          drv)
                continue
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
                removal=drv.get(f"{name}.removal"),
                transmission=drv.get(f"{name}.transmission"),
            )
        return out

    def current_flows(self, name: str, state: State, drivers: Drivers) -> Tensor:
        """Transport layer `name`'s branch flows for THIS `state`/`drivers` -- what `step`
        would use just before advancing `name`, exposed for a caller (a coupling
        orchestrator) that needs them WITHOUT stepping the model. Closures run first, so a
        driver-prescribed layer's flow is available. A Kirchhoff-owned layer's flows are the
        owner's solved `q`: read from `state["<owner>.q"]` when an earlier pass or step left
        one there, otherwise SOLVED here for these drivers, exactly as `_pass` would (the
        first pass of a step starts from a state that carries no `q` yet).

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
        if owner in self.kirchhoff and self.kirchhoff[owner].mna and f"{owner}.q" not in state:
            raise KeyError(
                f"Model: transport layer {name!r} takes its flows from Kirchhoff layer "
                f"{owner!r}, which is solved by Modified Nodal Analysis and cannot be solved "
                f"here without its step history; give the state {owner + '.q'!r} (step the "
                f"model first)"
            )
        if owner in self.kirchhoff and f"{owner}.q" not in state:
            key = self.flow_driver_of[name]
            if key in drv:
                # The same refusal `_kind_flows` makes on the read path, made here too: the
                # solve branch would otherwise silently prefer the Kirchhoff layer and never
                # notice the contradicting driver, so whether a two-sourced layer is caught
                # would depend on whether the state happened to carry a `q` yet.
                raise ValueError(
                    f"Model: transport layer {name!r} takes its flows from Kirchhoff layer "
                    f"{owner!r}, and the driver {key!r} was also given; a layer may have "
                    f"one source of flows, not two -- drop {key!r} or remove {owner!r}"
                )
            layer = self.kirchhoff[owner]
            phi_prev = state.get(f"{owner}.phi")
            phi0 = None if phi_prev is None else phi_prev[..., layer.interior]
            _phi, q = layer.solve(
                self._require(drv, f"{owner}.phi_boundary"), drv,
                drv.get(f"{owner}.sources"), phi0=phi0,
            )
            return layer.flows_of_kind(q, self.transport[name].flow_kinds)
        return self._kind_flows(name, state, drv)

    def ports(self, state) -> Ports:
        """Boundary nodes, the driver keys that prescribe them, and (Kirchhoff layers) the net
        flow INTO each boundary node at `state`.

        Allocation layers contribute NOTHING here, and that is deliberate rather than the
        same omission `residuals` refuses. `Ports` answers "which NODES may a coupled model
        prescribe, and through which key" -- a boundary-node partition with a prescribed
        potential or boundary composition. An `AllocatedFlowLayer` has no such partition:
        every node is interior, its unbounded nodes (`s_max = inf`) are a storage property and
        not a prescribable port, and its one driver (`"<name>.requests"`) is per EDGE, not per
        node, so it has no well-defined entry in any of `Ports`' four node-keyed dicts. A
        coupled model driving such a layer writes that per-edge driver directly; exposing it
        here would need a fifth, edge-keyed field, which is not implemented. Unlike
        `residuals`,
        nothing here is silently wrong: the dicts are complete for every layer type `Ports`
        is about.
        """
        nodes: dict[str, list] = {}
        keys: dict[str, str] = {}
        flows: dict[str, Tensor] = {}
        for name, layer in self.kirchhoff.items():
            nodes[name] = [layer.net.nodes[i] for i in layer.bound.tolist()]
            keys[name] = f"{name}.phi_boundary"
            flows[name] = -layer.net_outflow(state[f"{name}.q"])[..., layer.bound]
        flow_keys: dict[str, str] = {}
        for name, layer in self.transport.items():
            nodes[name] = list(layer.boundary)
            keys[name] = f"{name}.x_boundary"
            flow_keys[name] = self.flow_driver_of[name]
        return Ports(boundary_nodes=nodes, prescribed_keys=keys, boundary_flows=flows,
                     flow_keys=flow_keys)


class _ModelPasses:
    """`Model._iterate`'s side of `noodl._fixed_point_loop.PassProblem`.

    `z` is the fed state; a pass's output is `(new, diag)`. `judge` gives no verdict on
    pass 1 (it has no predecessor) and from pass 2 on applies the change, Anderson-residual,
    flow-tolerance and stall rules; `advance` relaxes (or Anderson-mixes) toward the pass's
    output. `produced`, `change` and `inner_failed` are read by `_iterate` after the loop.
    """

    def __init__(self, model: Model, state: State, drivers: Drivers, dt, pass_kwargs: dict,
                 *, t, boundary_transfers, warm_start: bool, raises_on_limit: bool) -> None:
        self.model = model
        self.state = state
        self.drivers = drivers
        self.dt = dt
        self.pass_kwargs = pass_kwargs
        self.t = t
        self.boundary_transfers = boundary_transfers
        self.warm_start = warm_start
        self.raises_on_limit = raises_on_limit
        self.produced: list[str] = []
        self.change: dict[str, Tensor] = {}
        self.prev: State | None = None
        self.inner_failed: Tensor | None = None
        self.anderson = (
            _AndersonHistory(model, model.iterate_memory)
            if model.iterate_acceleration == "anderson" else None
        )

    def run(self, z: State) -> tuple[State, dict]:
        new, diag, _ = self.model._pass(
            z, self.drivers, self.dt, self.pass_kwargs, step_from=self.state, t=self.t,
            boundary_transfers=self.boundary_transfers, produced=self.produced,
            warm_start=self.warm_start,
        )
        return new, diag

    def depends_on_grad(self, out: tuple[State, dict]) -> bool:
        new, _ = out
        return any(new[k].requires_grad for k in self.produced)

    def detach(self, z: State, out: tuple[State, dict]) -> tuple[State, tuple[State, dict]]:
        new, diag = out
        return z, ({k: v.detach() for k, v in new.items()}, detached(diag))

    def judge(self, z: State, out: tuple[State, dict], *,
              needs_adjoint: bool) -> Verdict | None:
        model = self.model
        new, diag = out
        self.inner_failed = inner_failed = model._unsolved(diag)
        prev = self.prev
        if prev is None:
            return None
        change = self.change
        ok: Tensor | None = None
        for name, tol in model.iterate_tol.items():
            if name in model.transport:
                layer = model.transport[name]
                d = (new[f"{name}.x"] - prev[f"{name}.x"]).abs()
                d_s, _ = layer._to_stacked(d, layer.n_i, "x")
                change[name] = d_s.amax(-1)
            else:
                layer = model.kirchhoff[name]
                d = (new[f"{name}.phi"] - prev[f"{name}.phi"]).abs()
                # Boundary entries too: a closure may write `phi_boundary`
                # from the previous pass (the sewer's outfall heads).
                nodes = torch.cat((layer.interior, layer.bound))
                change[name] = d[..., nodes].amax(-1)
                if self.anderson is not None:
                    # Mixing can stall: two passes fed nearly the same mixed
                    # state agree with each other while neither is a fixed
                    # point. The fixed-point residual (this pass's output
                    # against its own fed potential) must be within `tol` too.
                    key = f"{name}.phi"
                    res = (new[key] - z.get(key, prev[key])).abs()
                    change[name] = torch.maximum(
                        change[name], res[..., nodes].amax(-1))
                if name in model.iterate_flow_tol:
                    # The flows' own fixed-point residual: a head solve whose
                    # warm start already meets its residual tolerance returns
                    # the fed potential unchanged, so `phi` alone can stop
                    # while the flows the closures read are still moving.
                    key = f"{name}.q"
                    res = (new[key] - z.get(key, prev[key])).abs()
                    change[key] = res.amax(-1)
                    this_q = change[key] <= model.iterate_flow_tol[name]
                    ok = this_q if ok is None else (ok & this_q)
            this = change[name] <= tol
            ok = this if ok is None else (ok & this)
        stalled = None
        if inner_failed is not None:
            # a pass whose potential solve stopped short is not a fixed point
            stalled = ok & inner_failed
            ok = ok & ~inner_failed
        converged = ok
        # Every instance not converged was fed a state within the tolerances of the
        # previous pass's and its potential solve failed again: the next pass would repeat
        # it. The step is going to raise anyway, so it raises now rather than after
        # `iterate_max` passes of `max_iter` iterations.
        stop_early = bool(
            stalled is not None and bool(stalled.any())
            and bool((converged | stalled).all())
            and (needs_adjoint or self.raises_on_limit)
        )
        return Verdict(converged=converged, change=change, stop_early=stop_early)

    def advance(self, z: State, out: tuple[State, dict], verdict: Verdict | None) -> State:
        model = self.model
        new, _ = out
        prev = self.prev
        fed = dict(new)
        w = model.iterate_relaxation
        if prev is not None:
            for name in model.transport:
                key = f"{name}.x"
                fed[key] = prev[key] + w * (new[key] - prev[key])
            if self.anderson is None:
                for name in model.kirchhoff:
                    if name in model.iterate_tol:
                        for key in (f"{name}.phi", f"{name}.q"):
                            fed[key] = prev[key] + w * (new[key] - prev[key])
        if self.anderson is not None:
            # The named Kirchhoff layers' `phi` and `q` are Anderson-mixed instead
            # (`_AndersonHistory`), from pass 1 on; transport states stay relaxed.
            fed.update(self.anderson.mix(
                z, new, w, verdict.converged if verdict is not None else None))
        self.prev = new
        return fed
