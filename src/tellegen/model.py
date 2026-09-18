"""Model: several physics layers on one typed graph, stepped together.

Per step (framework spec section 3, "time stepping of a model"): closures update the drivers
from the current state; every potential layer is solved quasi-steadily; every transport layer
is advanced (sub-stepped if asked) on the flows of its kinds; reactions are applied.
`coupling="pingpong"` does that once per step with the state at the START of the step
(Hensen 1995); `coupling="iterate"` ("onion") repeats it until the named transport states stop
changing.

Keys. State: "<layer>.phi" (full-node order), "<layer>.q" (the layer's kind order),
"<layer>.x" (interior order, (n_i,) or (n_i, K)). Drivers: "<layer>.phi_boundary",
"<layer>.x_boundary", optional "<layer>.sources" (FULL-node order, zeros on boundary and
inactive nodes), optional "<layer>.capacity" (a transport layer's per-step capacity
override, spec 4.6b; absent, the layer's construction-time capacity stands). Closures
return driver updates; they may not write state keys. A transport layer whose kinds no
potential layer provides reads its branch flows from the driver "<layer>.q", in the layer's
own flow_kinds order; a layer with both a potential owner and that driver raises. A closure
may also declare `state_keys`, keys it carries across steps itself (spec 4.6a); those are
copied from its return into the returned state, evaluated from the step-start state in
every pass of a coupling that takes more than one (N1) -- never fed forward from an earlier
pass's own output, which would integrate a stateful closure once per pass instead of once
per step.

`**solve_kwargs` of `step`/`steady` reach the POTENTIAL solves only (`differentiable`,
`on_failure`, `method`, Newton kwargs). Transport steps always raise on failure.

Differentiability: ping-pong is one pass of differentiable operations; iterate is
differentiable by unrolling its passes (the convergence decision is made on detached copies,
as Newton's mask is). Failure follows the layers: raise by default, naming the offender.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from tellegen.layers.potential import PotentialFlowLayer
from tellegen.layers.reaction import Reaction
from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network

Tensor = torch.Tensor
State = dict[str, Tensor]
Drivers = dict[str, Tensor]
_STATE_SUFFIXES = ("phi", "q", "x")


@runtime_checkable
class Closure(Protocol):
    """state, drivers -> driver updates (a mapping of NEW driver values, merged in order)."""

    def __call__(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]: ...


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
    transport steps, reactions -- with the state at the START of the step. It is cheap and
    it is what a weakly coupled model wants; its splitting error is first order in `dt`.

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
    raises, naming the instances (`on_failure="return"` with `diagnostics` returns instead).
    """

    def __init__(
        self,
        net: Network,
        layers: Mapping[str, PotentialFlowLayer | TransportLayer],
        closures: Sequence[Closure] = (),
        reactions: Sequence[tuple[str, Reaction]] = (),
        coupling: str = "pingpong",
        iterate_tol: Mapping[str, float] | None = None,
        iterate_max: int = 20,
        substeps: Mapping[str, int] | None = None,
    ) -> None:
        if coupling not in ("pingpong", "iterate"):
            raise ValueError(
                f"Model: coupling must be 'pingpong' or 'iterate', got {coupling!r}"
            )
        self.net = net
        self.layers = dict(layers)
        self.potential: dict[str, PotentialFlowLayer] = {}
        self.transport: dict[str, TransportLayer] = {}
        for name, layer in self.layers.items():
            if isinstance(layer, PotentialFlowLayer):
                self.potential[name] = layer
            elif isinstance(layer, TransportLayer):
                self.transport[name] = layer
            else:
                raise TypeError(
                    f"Model: layer {name!r} is a {type(layer).__name__}, not a "
                    f"PotentialFlowLayer or TransportLayer"
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
    ) -> Drivers:
        """Run every closure, in order, over `drivers`; return the resulting drivers.

        `written` (keyword-out-parameter, like `solve`'s `diagnostics`; `None` by default so
        every existing caller -- including several apps' tests that call this directly -- is
        unaffected) is filled with whichever `closure_state_keys` some closure's OWN return
        actually wrote THIS call: N14's fix. A same-named entry already sitting in `drivers`
        (the caller's own, unrelated to any closure) must not be mistaken for a closure
        having written its declared state key -- `key in drv` alone cannot tell the two
        apart, since `drv` starts as a copy of `drivers`.
        """
        drv: Drivers = dict(drivers)
        for closure in self.closures:
            for key, value in closure(state, drv).items():
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
        """
        layer = self.transport[name]
        owner = self.flow_layer_of[name]
        key = self.flow_driver_of[name]
        if owner is not None:
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
        self, state, drivers, dt, solve_kwargs, *, step_from: State | None = None
    ) -> tuple[State, dict, Drivers]:
        """One closures -> potential -> transport -> reactions pass; `dt=None` means steady.

        `state` is what the closures read and what the potential solves warm-start from.
        `step_from` is the state a transport STEP starts at, which is a different thing
        inside an iterated coupling: pass k there re-advances the SAME time step from the
        state at the start of it, with only the closures' view of the new state updated.
        It defaults to `state`, which is what a single ping-pong pass wants.
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
        drv = self._apply_closures(closure_state, drivers, written)
        new: State = dict(state)
        diag: dict = {}
        for name, layer in self.potential.items():
            pb = self._require(drv, f"{name}.phi_boundary")
            sources = drv.get(f"{name}.sources")
            phi_prev = state.get(f"{name}.phi")
            phi0 = None if phi_prev is None else phi_prev[..., layer.interior]
            d: dict = {}
            phi, q = layer.solve(pb, drv, sources, phi0=phi0, diagnostics=d, **solve_kwargs)
            new[f"{name}.phi"], new[f"{name}.q"] = phi, q
            diag[name] = d
        for name, layer in self.transport.items():
            q_kind = self._kind_flows(name, new, drv)
            xb = self._require(drv, f"{name}.x_boundary")
            sources = drv.get(f"{name}.sources")
            # Spec 4.6b: a per-step capacity, written by a closure that owns the geometry
            # (a sewer conduit's wetted volume, a headspace volume). Absent, the layer's
            # construction-time capacity stands, so nothing changes for a fixed-storage
            # layer. Shape and positivity are checked by the layer, naming the nodes.
            cap = drv.get(f"{name}.capacity")
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
                x = layer.steady(q_kind, sources, xb, capacity=cap)
            else:
                x = base.get(f"{name}.x")
                if x is None:
                    raise KeyError(
                        f"Model: state {name + '.x'!r} is required to step transport layer "
                        f"{name!r}"
                    )
                if sources is None:
                    sources = self._zero_sources(layer, x, layer.n_i)
                k = self.substeps[name]
                for _ in range(k):
                    x = layer.step(x, q_kind, sources, xb, dt / k, capacity=cap)
                for lname, reaction in self.reactions:
                    if lname == name:
                        x = reaction.apply(x, dt, drv)
            new[f"{name}.x"] = x
            diag[name] = {"substeps": self.substeps[name]}
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
        return new, diag, drv

    # ------------------------------------------------------------------ public
    def step(self, state, drivers, dt: float, *, diagnostics: dict | None = None, **solve_kwargs):
        if not dt > 0:
            raise ValueError(f"Model: dt must be positive, got {dt!r}")
        return self._advance(state, drivers, float(dt), diagnostics, solve_kwargs)

    def steady(self, state, drivers, *, diagnostics: dict | None = None, **solve_kwargs):
        """The quasi-steady state of every layer at `drivers` (transport layers solved to
        `rate == 0` rather than advanced).

        REACTIONS ARE NOT APPLIED. They are an operator splitting applied AFTER a transport
        step, so they belong to `step` alone: a model carrying a reaction has a `steady` that
        is the fixed point of transport only, not of transport-plus-reaction (spec section 7;
        `residuals` reports the same balance). Deliberate, and pinned by a test.
        """
        return self._advance(state, drivers, None, diagnostics, solve_kwargs)

    def _advance(self, state, drivers, dt, diagnostics, solve_kwargs) -> State:
        # The coupling seam: ping-pong is exactly ONE pass, taken with the state at the start
        # of the step; `coupling="iterate"` repeats `_pass` until the named transport states
        # stop changing, and reports the pass count it took.
        if self.coupling == "pingpong":
            new, diag, _ = self._pass(state, drivers, dt, solve_kwargs)
            if diagnostics is not None:
                diagnostics.update({"passes": 1, "layers": diag})
            return new
        return self._iterate(state, drivers, dt, diagnostics, solve_kwargs)

    def _iterate(self, state, drivers, dt, diagnostics, solve_kwargs) -> State:
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
        Convergence is judged per instance on detached copies; the passes themselves stay on
        the autograd graph (unrolled).

        The fixed point is therefore differentiated by unrolling, which carries the whole
        chain of passes on the graph. An implicit-function treatment of the fixed point (one
        adjoint solve at the converged state, memory independent of the pass count) is a
        follow-up, not this milestone.
        """
        fed: State = dict(state)
        prev: State | None = None
        change: dict[str, Tensor] = {}
        # A placeholder the second pass always replaces: `iterate_max >= 2` and a non-empty
        # `iterate_tol` are both refused at construction, so the loop below cannot end
        # without a real per-instance verdict of the right shape, dtype and device.
        converged: Tensor = torch.zeros((), dtype=torch.bool)
        passes = 0
        diag: dict = {}
        new: State = dict(state)
        # A `while` rather than `for passes in range(...)`: the pass count is wanted AFTER
        # the loop (it goes into the diagnostics and into the failure message), which a loop
        # control variable unused inside the body is not (ruff B007).
        while passes < self.iterate_max:
            passes += 1
            new, diag, _ = self._pass(fed, drivers, dt, solve_kwargs, step_from=state)
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
            if not (solve_kwargs.get("on_failure") == "return" and diagnostics is not None):
                raise RuntimeError(message)
        if diagnostics is not None:
            diagnostics.update(
                {"passes": passes, "converged": converged, "max_change": change, "layers": diag}
            )
        return new

    def residuals(self, state, drivers) -> dict[str, Tensor]:
        """Nodal balances at `state`: interior residual per potential layer, dx/dt per
        transport layer (spec 14). Zero (to solver tolerance) at a steady state.

        The transport balance is TRANSPORT ONLY: reactions are an operator splitting applied
        by `step` after the transport step, so they are outside the balance reported here,
        exactly as they are outside `steady`. A model with a reaction is therefore at zero
        residual at `steady`'s fixed point, not at the reaction's.
        """
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

    def ports(self, state) -> Ports:
        """Boundary nodes, the driver keys that prescribe them, and (potential layers) the net
        flow INTO each boundary node at `state`."""
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
