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
inactive nodes). Closures return driver updates; they may not write state keys.

`**solve_kwargs` of `step`/`steady` reach the POTENTIAL solves only (`differentiable`,
`on_failure`, `method`, Newton kwargs). Transport steps always raise on failure.

Differentiability: ping-pong is one pass of differentiable operations; iterate is
differentiable by unrolling its passes (the convergence decision is made on detached copies,
as Newton's mask is). Failure follows the layers: raise by default, naming the offender.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


class Model:
    """Layers on one `Network`, stepped together. See the module docstring for the keys."""

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
        if coupling not in ("pingpong",):
            raise ValueError(f"Model: coupling must be 'pingpong', got {coupling!r}")
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
        self.flow_layer_of: dict[str, str] = {}
        for tname, tl in self.transport.items():
            owners = [
                pn for pn, pl in self.potential.items()
                if all(k in pl.kinds for k in tl.flow_kinds)
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"Model: transport layer {tname!r} advects on kinds {tl.flow_kinds}, "
                    f"which {len(owners)} potential layers of this model provide "
                    f"({owners}); exactly one must"
                )
            self.flow_layer_of[tname] = owners[0]

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _require(drivers: Mapping[str, Tensor], key: str) -> Tensor:
        try:
            return drivers[key]
        except KeyError as exc:
            raise KeyError(f"Model: driver {key!r} is required and was not given") from exc

    def _apply_closures(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
    ) -> Drivers:
        drv: Drivers = dict(drivers)
        for closure in self.closures:
            for key, value in closure(state, drv).items():
                head, _, tail = key.rpartition(".")
                if head in self.layers and tail in _STATE_SUFFIXES:
                    raise ValueError(
                        f"Model: closure {closure!r} returned {key!r}, which is a state key; "
                        f"closures write drivers only"
                    )
                drv[key] = value
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

    def _kind_flows(self, name: str, state: Mapping[str, Tensor]) -> Tensor:
        owner = self.flow_layer_of[name]
        return self.potential[owner].flows_of_kind(
            state[f"{owner}.q"], self.transport[name].flow_kinds
        )

    # -------------------------------------------------------------------- pass
    def _pass(self, state, drivers, dt, solve_kwargs) -> tuple[State, dict, Drivers]:
        """One closures -> potential -> transport -> reactions pass; `dt=None` means steady."""
        drv = self._apply_closures(state, drivers)
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
            q_kind = self._kind_flows(name, new)
            xb = self._require(drv, f"{name}.x_boundary")
            sources = drv.get(f"{name}.sources")
            if dt is None:
                if sources is None:
                    # The state's own `x` is the layout authority when it is there; `x_b`
                    # is the fallback for a steady solve started without one.
                    like = state.get(f"{name}.x")
                    sources = (
                        self._zero_sources(layer, xb, layer.n_b)
                        if like is None
                        else self._zero_sources(layer, like, layer.n_i)
                    )
                x = layer.steady(q_kind, sources, xb)
            else:
                x = state.get(f"{name}.x")
                if x is None:
                    raise KeyError(
                        f"Model: state {name + '.x'!r} is required to step transport layer "
                        f"{name!r}"
                    )
                if sources is None:
                    sources = self._zero_sources(layer, x, layer.n_i)
                k = self.substeps[name]
                for _ in range(k):
                    x = layer.step(x, q_kind, sources, xb, dt / k)
                for lname, reaction in self.reactions:
                    if lname == name:
                        x = reaction.apply(x, dt, drv)
            new[f"{name}.x"] = x
            diag[name] = {"substeps": self.substeps[name]}
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
        # of the step. Task 9's `coupling="iterate"` repeats `_pass` here until the named
        # transport states stop changing, and reports the pass count it took.
        new, diag, _ = self._pass(state, drivers, dt, solve_kwargs)
        if diagnostics is not None:
            diagnostics.update({"passes": 1, "layers": diag})
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
            out[name] = layer.rate(x, self._kind_flows(name, state), sources, xb)
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
        for name, layer in self.transport.items():
            nodes[name] = list(layer.boundary)
            keys[name] = f"{name}.x_boundary"
        return Ports(boundary_nodes=nodes, prescribed_keys=keys, boundary_flows=flows)
