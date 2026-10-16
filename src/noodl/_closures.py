"""The closures of a `noodl.model.Model`: their specs, their checks and how a pass runs them.

`ClosureSchedule` reads every closure's declarations once (`noodl.contracts.closure_spec`),
makes the construction checks that involve the model's layers, owns `closure_state_keys`,
and runs the closures in order (`apply`, the body of `Model._apply_closures`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from noodl._tracking import TrackedDict
from noodl.contracts import (
    ClosureSpec,
    StepContext,
    check_species_columns,
    closure_spec,
    reaction_species_columns,
)

Tensor = torch.Tensor
_STATE_SUFFIXES = ("phi", "q", "x", "s")


class ClosureSchedule:
    """The closures' specs and the rules a model applies to them.

    `layers`, `transport` and `flow_layer_of` are the model's own (the same dicts), read to
    tell a layer's state key from a driver. The specs hold the closure OBJECTS, so a deepcopy
    or a pickle carries each closure with its spec; the identity lookup keyed by `id()` is
    dropped from a copy and rebuilt on first use.
    """

    def __init__(
        self, closures: Sequence[Callable], *, layers: Mapping, transport: Mapping,
        flow_layer_of: Mapping[str, str | None],
    ) -> None:
        self.layers = layers
        self.transport = transport
        self.flow_layer_of = flow_layer_of
        # Every closure's declarations are read ONCE, here, by `noodl.contracts.closure_spec`
        # (callability, `integrates`, `state_keys`, the declared inputs/outputs, ...): a
        # malformed one is refused by name now, not as a bare "'X' object is not callable"
        # from inside `_pass`. The specs hold the closure OBJECTS, so a deepcopy or a pickle
        # carries each closure with its spec; nothing is keyed by `id()` across a copy.
        self.specs: list[ClosureSpec] = [
            closure_spec(closure, index=i) for i, closure in enumerate(closures)
        ]
        self._ids: dict[int, ClosureSpec] | None = None
        for spec in self.specs:
            for key in sorted(spec.writes or ()):
                if self.is_layer_state_key(key):
                    raise ValueError(
                        f"Model: closure {spec.label} declares output {key!r}, which is a "
                        f"state key; closures write drivers only (the one exception is "
                        f"'<layer>.q' for a transport layer whose flows no Kirchhoff layer "
                        f"provides)"
                    )
        # CLOSURE-CARRIED STATE. A closure may declare
        # `state_keys: tuple[str, ...]`: keys it both READS from the state and WRITES back
        # every call, which `_pass` copies from its return into the returned state. They are
        # neither drivers (they persist across steps) nor layer state (no layer owns them);
        # a sewer manhole level under `storage=True`, or a tank level and the controlled
        # links' status in the water application, are the motivating cases. Two closures
        # claiming one key is refused HERE, naming both, rather than resolved by whichever
        # ran last.
        self.closure_state_keys: dict[str, Callable] = {}
        for spec in self.specs:
            closure = spec.closure
            for key in spec.state_keys:
                head, _, tail = str(key).rpartition(".")
                if head in self.layers and tail in _STATE_SUFFIXES:
                    # Mirror `apply`'s own ownerless-"<layer>.q" carve-out --
                    # for a transport layer no Kirchhoff layer owns, "<layer>.q" is the
                    # DRIVER a flow closure writes, not that layer's state, so a closure may
                    # declare it as state_keys (a flow closure that also remembers its own
                    # last-written flow across steps is exactly this shape). Every other
                    # "<layer>.phi/q/x" stays refused, including "<layer>.q" for a layer a
                    # Kirchhoff layer DOES own.
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

    def __getstate__(self) -> dict:
        # The identity lookup is keyed by `id()`, which a copy or a pickle does not keep;
        # `specs` holds the closures themselves and the lookup is rebuilt on first use.
        state = self.__dict__.copy()
        state["_ids"] = None
        return state

    # ------------------------------------------------------------------ checks
    def check_species_columns(self, transport: Mapping, reactions: Sequence) -> None:
        """The closures' and then the reactions' `species_columns` against the layers'
        species (`noodl.contracts.check_species_columns`)."""
        for spec in self.specs:
            check_species_columns(f"Model: closure {spec.label}", spec.species_columns,
                                  transport)
        for lname, reaction in reactions:
            check_species_columns(f"Model: reaction {reaction!r} for layer {lname!r}",
                                  reaction_species_columns(reaction), transport)

    def check_coupling(self, coupling: str, iterate_flow_tol: Mapping[str, float]) -> None:
        """Refuse a closure that declares `requires_coupling = "iterate"` on a model that
        does not iterate its `reads_flows_of` layer's flows."""
        # A closure that reads a Kirchhoff layer's flows to write something the same pass
        # uses (a flow-dependent filter's transmission) is right only at the coupling's fixed
        # point, where the flows it read are the flows the pass produced: it declares
        # `requires_coupling = "iterate"` and `reads_flows_of = "<Kirchhoff layer>"`.
        for spec in self.specs:
            if spec.requires_coupling != "iterate":
                continue
            closure = spec.closure
            flows_of = spec.reads_flows_of
            if coupling != "iterate" or flows_of not in iterate_flow_tol:
                raise ValueError(
                    f"Model: closure {type(closure).__name__} reads the flows of layer "
                    f"{flows_of!r} before the pass solves them, so it needs "
                    f"coupling='iterate' with {flows_of!r} in iterate_tol and "
                    f"iterate_flow_tol (each pass is then fed the previous pass's flows and a "
                    f"step converges on its own); got coupling={coupling!r}, "
                    f"iterate_flow_tol={iterate_flow_tol}"
                )

    def is_layer_state_key(self, key: str) -> bool:
        """True for a layer's own state key `"<layer>.phi/q/x/s"` a closure may not write:
        all of them but `"<layer>.q"` of a transport layer no Kirchhoff layer owns, which is
        the flow DRIVER a flow closure writes."""
        head, _, tail = key.rpartition(".")
        if head in self.layers and tail in _STATE_SUFFIXES:
            return not (
                tail == "q" and head in self.transport and self.flow_layer_of.get(head) is None
            )
        return False

    # ------------------------------------------------------------------ lookup
    def spec(self, closure: Callable, closures: Sequence[Callable]) -> ClosureSpec:
        """The `ClosureSpec` of `closure`, one of `closures` (the model's live list), found
        by identity. A closure added to the list after construction is read on first use
        and cached in the identity lookup only; `specs` keeps the construction-time ones.
        On every such miss the lookup drops the closures no longer in the live list, so it
        stays bounded by the live list however often closures are swapped in and out."""
        lookup = self._ids
        if lookup is None:
            lookup = self._ids = {id(s.closure): s for s in self.specs}
        spec = lookup.get(id(closure))
        if spec is None or spec.closure is not closure:
            live = {id(c) for c in closures}
            for key in [k for k in lookup if k not in live]:
                del lookup[key]
            spec = closure_spec(closure, index=closures.index(closure))
            lookup[id(closure)] = spec
        return spec

    def integrating(self, closures: Sequence[Callable]) -> set[int]:
        """`id()`s of the integrating closures among `closures`."""
        return {id(c) for c in closures if self.spec(c, closures).integrates}

    # ------------------------------------------------------------------- apply
    def apply(
        self, closures: Sequence[Callable], state: Mapping[str, Tensor],
        drivers: Mapping[str, Tensor], *, written: set[str] | None = None,
        ctx: StepContext | None = None, probe=None,
    ) -> dict:
        """Run every closure of `closures`, in order, over `drivers`; return the resulting
        drivers (see `Model._apply_closures`). `probe` is `Model._probe`: when set, reads
        are tracked and every closure's return is recorded in it."""
        if probe is None:
            drv: dict = dict(drivers)
        else:
            # `Model.check(..., probe=True)`: record which keys are read (see
            # `noodl.validation`). The returned drivers stay tracked, so the layers',
            # elements' and drives' reads are recorded too.
            drv = TrackedDict(drivers, probe.driver_reads)
            state = TrackedDict(state, probe.state_reads)
        for closure in closures:
            spec = self.spec(closure, closures)
            result = spec(state, drv, ctx)
            if probe is not None:
                probe.closure_outputs.update(result)
                probe.closure_values.update(result)
                probe.closure_writers.update(dict.fromkeys(result, type(closure).__name__))
                # Keyed by IDENTITY: two closures may share a `repr` (the label is for
                # display only).
                probe.closure_returned.setdefault(id(closure), set()).update(result)
            for key, value in result.items():
                head, _, tail = key.rpartition(".")
                if head in self.layers and tail in _STATE_SUFFIXES:
                    # ONE exception to "closures write drivers only": "<layer>.q" for a
                    # transport layer no Kirchhoff layer owns is a DRIVER, not a state --
                    # it is the key `_kind_flows` reads that layer's branch flows from, and
                    # writing it is exactly what a flow closure is for. Every other
                    # "<layer>.phi/q/x" stays refused, including "<layer>.q" for a layer a
                    # Kirchhoff layer DOES own, where it would silently contradict the
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
                            f"Kirchhoff layer provides)"
                        )
                drv[key] = value
                if written is not None and key in self.closure_state_keys:
                    written.add(key)
        return drv
