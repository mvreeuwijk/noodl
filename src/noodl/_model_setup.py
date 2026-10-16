"""Construction of a `noodl.model.Model`: layer registration, flow ownership and options.

`Model.__init__` calls these in its own order and assigns the attributes they return; every
refusal is raised here with the message the model has always given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from noodl.layers.allocation import AllocatedFlowLayer
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.topology import Network


def validate_modes(coupling: str, reaction_order: str) -> None:
    """Refuse an unknown `coupling` or `reaction_order` (checked first, before the layers)."""
    if coupling not in ("pingpong", "iterate"):
        raise ValueError(
            f"Model: coupling must be 'pingpong' or 'iterate', got {coupling!r}"
        )
    # WHERE in the operator split the reactions sit, and so where the returned state is
    # SAMPLED. Both orders take the same `... R T R T ...` sequence; they differ only in
    # which half-step the returned `"<layer>.x"` follows. `"after_transport"` (T then R,
    # the default) returns the reacted state; `"before_transport"` (R then T) reacts the
    # step-start state and returns the transported one -- the order of SWMM's
    # `qualrout.c::findLinkQual` (`getReactedQual`, then `getMixedQual`), which reports
    # the concentration after mixing. The sewer application builds with it.
    if reaction_order not in ("after_transport", "before_transport"):
        raise ValueError(
            f"Model: reaction_order must be 'after_transport' or 'before_transport', "
            f"got {reaction_order!r}"
        )


@dataclass(frozen=True)
class RegisteredLayers:
    """A model's layers by name, all of them and split by kind (each in registration
    order)."""

    layers: dict
    kirchhoff: dict[str, KirchhoffLayer]
    transport: dict[str, TransportLayer]
    allocation: dict[str, AllocatedFlowLayer]


def register_layers(
    net: Network,
    layers: Mapping[str, KirchhoffLayer | TransportLayer | AllocatedFlowLayer]
    | Sequence[KirchhoffLayer | TransportLayer | AllocatedFlowLayer],
) -> RegisteredLayers:
    """The layers by registered name, checked: unique names, each equal to the layer's own
    `.name`, a known layer kind, and built on `net`."""
    # ONE layer identity. Every key a layer reads or writes is "<name>.<suffix>", and
    # two different parties build those names: this model, from the name a layer is
    # REGISTERED under, and the layer and its helpers, from the layer's OWN `.name` (a
    # allocation layer reads "<name>.requests" itself; the building density closure
    # reads "<thermal.name>.x"). A layer registered under any other name would split
    # its keys between the two, so the registered name must BE the layer's name.
    # Layers may therefore also be given as a plain sequence, keyed by their names.
    if isinstance(layers, Mapping):
        registered = dict(layers)
    else:
        registered = {}
        for layer in layers:
            lname = getattr(layer, "name", None)
            if lname in registered:
                raise ValueError(
                    f"Model: two layers named {lname!r}; layer names are the prefix of "
                    f"every state and driver key and must be unique"
                )
            registered[lname] = layer
    for name, layer in registered.items():
        own = getattr(layer, "name", name)
        if own != name:
            raise ValueError(
                f"Model: a layer is registered as {name!r} but is named {own!r}; the "
                f"model and the layer would then build its keys under different names "
                f"({name!r}.x vs {own!r}.x). Register it under its own name, or build "
                f"it with name={name!r}"
            )
    kirchhoff: dict[str, KirchhoffLayer] = {}
    transport: dict[str, TransportLayer] = {}
    allocation: dict[str, AllocatedFlowLayer] = {}
    for name, layer in registered.items():
        if isinstance(layer, KirchhoffLayer):
            kirchhoff[name] = layer
        elif isinstance(layer, TransportLayer):
            transport[name] = layer
        elif isinstance(layer, AllocatedFlowLayer):
            allocation[name] = layer
        else:
            raise TypeError(
                f"Model: layer {name!r} is a {type(layer).__name__}, not a "
                f"KirchhoffLayer, TransportLayer or AllocatedFlowLayer"
            )
        if layer.net is not net:
            raise ValueError(
                f"Model: layer {name!r} is built on a different Network than the model"
            )
    return RegisteredLayers(registered, kirchhoff, transport, allocation)


def flow_owners(
    kirchhoff: Mapping[str, KirchhoffLayer],
    transport: Mapping[str, TransportLayer],
    allocation: Mapping[str, AllocatedFlowLayer],
) -> tuple[dict[str, str | None], dict[str, str]]:
    """`(flow_layer_of, flow_driver_of)`: per transport layer, the one layer whose flows it
    advects on (`None`: read from the driver) and the driver key `"<layer>.q"`."""
    flow_layer_of: dict[str, str | None] = {}
    flow_driver_of: dict[str, str] = {}
    for tname, tl in transport.items():
        owners = [
            pn for pn, pl in kirchhoff.items()
            if all(k in pl.kinds for k in tl.flow_kinds)
        ] + [
            cn for cn, cl in allocation.items()
            if all(k in cl.kinds for k in tl.flow_kinds)
        ]
        # Driver-prescribed flows. Two Kirchhoff layers both
        # providing a transport layer's kinds is ambiguous and refused here.
        # ZERO owners is not an error: the flows are then read from the driver
        # "<layer>.q", which a closure writes -- the street application's whole
        # architecture (prescribed canyon fluxes, routed at intersections) depends on
        # it. The driver's PRESENCE cannot be checked here, because drivers arrive per
        # call; `_kind_flows` does that, and refuses a layer that has both sources.
        if len(owners) > 1:
            raise ValueError(
                f"Model: transport layer {tname!r} advects on kinds {tl.flow_kinds}, "
                f"which {len(owners)} Kirchhoff layers of this model provide "
                f"({owners}); at most one may"
            )
        flow_layer_of[tname] = owners[0] if owners else None
        flow_driver_of[tname] = f"{tname}.q"
    return flow_layer_of, flow_driver_of


def validate_reactions(transport: Mapping[str, TransportLayer], reactions) -> list:
    """The reactions as a list, each checked to target a transport layer and to provide
    `apply(x, dt, drivers)`."""
    reactions = list(reactions)
    for lname, reaction in reactions:
        if lname not in transport:
            raise KeyError(
                f"Model: reaction targets {lname!r}, not a transport layer of this "
                f"model ({sorted(transport)})"
            )
        if not callable(getattr(reaction, "apply", None)):
            raise TypeError(
                f"Model: reaction for layer {lname!r} is a {type(reaction).__name__}, "
                f"not a Reaction; it must provide apply(x, dt, drivers)"
            )
    return reactions


@dataclass(frozen=True)
class CouplingOptions:
    """The coupling options of a `Model`, validated (`CouplingOptions.validate`); the model
    assigns each to the attribute of the same name."""

    coupling: str
    iterate_tol: dict[str, float]
    iterate_max: int
    iterate_relaxation: float
    iterate_acceleration: str | None
    iterate_memory: int
    iterate_on_limit: str
    adjoint_rtol: float
    iterate_flow_tol: dict[str, float]

    @classmethod
    def validate(
        cls, *, coupling: str, iterate_tol: Mapping[str, float] | None, iterate_max: int,
        iterate_relaxation: float, iterate_acceleration: str | None, iterate_memory: int,
        iterate_on_limit: str, adjoint_rtol: float,
        iterate_flow_tol: Mapping[str, float] | None,
        kirchhoff: Mapping[str, KirchhoffLayer], transport: Mapping[str, TransportLayer],
    ) -> CouplingOptions:
        """The options, checked in the model's order. The checks that combine
        `iterate_acceleration` with the layers come after the closures' coupling check and
        are `check_acceleration`'s."""
        tol = dict(iterate_tol or {})
        max_passes = int(iterate_max)
        # The weight of the newest pass in the transport state fed to the next one
        # (`_iterate`): 0.5 is Hensen's (1995) mean of the last two passes, 1 plain
        # successive substitution. It moves no fixed point, only how fast (or whether) the
        # passes reach it.
        relaxation = float(iterate_relaxation)
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(
                f"Model: iterate_relaxation must lie in (0, 1], got {iterate_relaxation!r}"
            )
        # How the fed state of the Kirchhoff layers named in `iterate_tol` is formed from
        # the passes so far (`_iterate`): None is the relaxation above, "anderson" Anderson
        # type-II mixing over the last `iterate_memory` passes, falling back to it.
        if iterate_acceleration not in (None, "anderson"):
            raise ValueError(
                f"Model: iterate_acceleration must be None or 'anderson', got "
                f"{iterate_acceleration!r}"
            )
        if (isinstance(iterate_memory, bool) or not isinstance(iterate_memory, int)
                or iterate_memory < 1):
            raise ValueError(
                f"Model: iterate_memory must be an integer >= 1 (the number of previous "
                f"passes Anderson mixing combines), got {iterate_memory!r}"
            )
        if iterate_on_limit not in ("raise", "accept"):
            raise ValueError(
                f"Model: iterate_on_limit must be 'raise' or 'accept', got {iterate_on_limit!r}"
            )
        # The tolerance of the ONE implicit-adjoint GMRES solve at the converged interface
        # deliberately independent of the primal `iterate_tol`: that independence is
        # the whole point of differentiating the fixed point instead of the iteration.
        rtol = float(adjoint_rtol)
        if not rtol > 0.0:
            raise ValueError(
                f"Model: adjoint_rtol must be positive, got {adjoint_rtol!r}; it is the "
                f"relative tolerance of the fixed-point adjoint's GMRES solve"
            )
        if coupling == "iterate":
            if not tol:
                raise ValueError(
                    "Model: coupling='iterate' requires iterate_tol={transport or potential "
                    "layer name: absolute tolerance on that layer's state (`.x`) or "
                    "potential (`.phi`)}"
                )
            unknown = sorted(set(tol) - set(transport) - set(kirchhoff))
            if unknown:
                raise KeyError(
                    f"Model: iterate_tol names {unknown}, neither transport layers "
                    f"({sorted(transport)}) nor Kirchhoff layers "
                    f"({sorted(kirchhoff)}) of this model"
                )
            # Refused HERE rather than left to fail at the first `step`: one pass has no
            # second pass to compare against, so it can never report convergence -- a budget
            # below two is a configuration that always raises, and a `iterate_max=0` would
            # return the input state untouched before doing so. Enforcing it also keeps the
            # diagnostics one shape: `converged` and `max_change` are always the ones a real
            # comparison produced, never a degenerate 0-dim placeholder and an empty dict.
            if max_passes < 2:
                raise ValueError(
                    f"Model: coupling='iterate' requires iterate_max >= 2, got "
                    f"{iterate_max!r}; a single pass has no predecessor to compare against, "
                    f"so it can never be reported as converged"
                )
        # The flows' fixed-point tolerance per Kirchhoff layer named in `iterate_tol`
        # (absolute, in that layer's flow units): |q returned - q fed| of the certified pass.
        flow_tol = {k: float(v) for k, v in (iterate_flow_tol or {}).items()}
        stray = sorted(k for k in flow_tol if k not in kirchhoff or k not in tol)
        if stray:
            raise KeyError(
                f"Model: iterate_flow_tol names {stray}, which are not Kirchhoff layers named "
                f"in iterate_tol ({sorted(k for k in tol if k in kirchhoff)})"
            )
        bad = {k: v for k, v in flow_tol.items() if not v > 0.0}
        if bad:
            raise ValueError(f"Model: iterate_flow_tol must be positive, got {bad}")
        return cls(
            coupling=coupling, iterate_tol=tol, iterate_max=max_passes,
            iterate_relaxation=relaxation, iterate_acceleration=iterate_acceleration,
            iterate_memory=iterate_memory, iterate_on_limit=iterate_on_limit,
            adjoint_rtol=rtol, iterate_flow_tol=flow_tol,
        )

    def check_acceleration(self, kirchhoff: Mapping[str, KirchhoffLayer]) -> None:
        """Refuse an `iterate_acceleration` the coupling or the named layers cannot take."""
        iterate_acceleration, coupling = self.iterate_acceleration, self.coupling
        if iterate_acceleration is None:
            return
        if coupling != "iterate":
            raise ValueError(
                f"Model: iterate_acceleration={iterate_acceleration!r} needs "
                f"coupling='iterate', got coupling={coupling!r}"
            )
        if not any(name in kirchhoff for name in self.iterate_tol):
            raise ValueError(
                f"Model: iterate_acceleration={iterate_acceleration!r} mixes the potential "
                f"of a Kirchhoff layer named in iterate_tol, and iterate_tol names none "
                f"({sorted(self.iterate_tol)}; Kirchhoff layers: {sorted(kirchhoff)})"
            )
        phasor = sorted(k for k in self.iterate_tol
                        if k in kirchhoff and kirchhoff[k].phasor)
        if phasor:
            raise ValueError(
                f"Model: iterate_acceleration={iterate_acceleration!r} cannot mix the "
                f"complex potentials of the phasor layer(s) {phasor}; Anderson mixing is "
                f"real-valued -- use the relaxation (iterate_acceleration=None)"
            )
        unset = sorted(k for k in self.iterate_tol
                       if k in kirchhoff and k not in self.iterate_flow_tol)
        if unset:
            raise ValueError(
                f"Model: iterate_acceleration={iterate_acceleration!r} needs "
                f"iterate_flow_tol for the Kirchhoff layer(s) {unset}: the tolerance on "
                f"their flows' fixed-point residual, which the mixing minimises and the "
                f"convergence test judges alongside the potential"
            )


def validate_substeps(
    transport: Mapping[str, TransportLayer], substeps: Mapping[str, int] | None,
) -> dict[str, int]:
    """The substep count of every transport layer: 1 unless `substeps` names it."""
    out = {name: 1 for name in transport}
    for name, k in (substeps or {}).items():
        if name not in transport:
            raise KeyError(
                f"Model: substeps names {name!r}, not a transport layer of this model "
                f"({sorted(transport)})"
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
        out[name] = k_int
    return out
