"""One pass of a `noodl.model.Model`: closures, potential solves, allocation steps,
transport steps and reactions (`run_pass`, behind `Model._pass`), and the per-layer helpers
the pass and the model's queries share (`kind_flows`, `zero_sources`, `circuit_kwargs`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

import torch

from noodl import layout
from noodl.contracts import StepContext
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer

if TYPE_CHECKING:
    from noodl.model import Drivers, Model, State

Tensor = torch.Tensor
# The key `layer_options` travels under inside `solve_kwargs`, to the pass's layer solves.
LAYER_OPTIONS = "layer_options"


def layer_kwargs(solve_kwargs: dict, name: str) -> dict:
    """The keywords of Kirchhoff layer `name`'s solve: `solve_kwargs` with that layer's
    `layer_options` applied over them."""
    if LAYER_OPTIONS not in solve_kwargs:
        return solve_kwargs
    shared = {k: v for k, v in solve_kwargs.items() if k != LAYER_OPTIONS}
    return {**shared, **solve_kwargs[LAYER_OPTIONS].get(name, {})}


def require(drivers: Mapping[str, Tensor], key: str) -> Tensor:
    """`drivers[key]`, refused by name when it is missing."""
    try:
        return drivers[key]
    except KeyError as exc:
        raise KeyError(f"Model: driver {key!r} is required and was not given") from exc


def zero_sources(layer: TransportLayer, like: Tensor, n_like: int, n_nodes: int) -> Tensor:
    """Full-node zeros (`n_nodes` nodes) in `like`'s LAYOUT (reduced `(n,)` or stacked
    `(n, K)`); see `Model._zero_sources`."""
    _, reduced = layout.as_stacked(
        like, n_like, layer.n_species, name="state", where=f"TransportLayer '{layer.name}'",
    )
    batch = like.shape[:-1] if reduced else like.shape[:-2]
    tail = (n_nodes,) if reduced else (n_nodes, layer.n_species)
    return torch.zeros(batch + tail, dtype=like.dtype, device=like.device)


def kind_flows(model: Model, name: str, state: Mapping[str, Tensor],
               drivers: Mapping[str, Tensor]) -> Tensor:
    """Transport layer `name`'s branch flows, in its own `flow_kinds` order; see
    `Model._kind_flows`."""
    layer = model.transport[name]
    owner = model.flow_layer_of[name]
    key = model.flow_driver_of[name]
    if owner is not None:
        # Checked BEFORE the both-sources refusal below, whose message says "Kirchhoff
        # layer" and would be factually wrong about an allocation owner.
        if owner in model.allocation:
            raise NotImplementedError(
                f"Model: transport layer {name!r} advects on kinds {layer.flow_kinds}, "
                f"which allocation layer {owner!r} provides; reading an allocation "
                f"layer's flows into a transport layer is not implemented. "
                f"Drive {name!r} from the "
                f"driver {key!r} instead, or give its kinds to a Kirchhoff layer"
            )
        if key in drivers:
            raise ValueError(
                f"Model: transport layer {name!r} takes its flows from Kirchhoff layer "
                f"{owner!r}, and the driver {key!r} was also given; a layer may have "
                f"one source of flows, not two -- drop {key!r} or remove {owner!r}"
            )
        return model.kirchhoff[owner].flows_of_kind(state[f"{owner}.q"], layer.flow_kinds)
    q = require(drivers, key)
    b = int(layer._flow_src.shape[0])
    if q.dim() == 0 or q.shape[-1] != b:
        got = q.shape[-1] if q.dim() >= 1 else 0
        raise ValueError(
            f"Model: driver {key!r} must have trailing shape ({b},) -- transport layer "
            f"{name!r} advects on {b} flow edges of kinds {layer.flow_kinds}, in that "
            f"order -- got {got} (shape {tuple(q.shape)})"
        )
    return q


def circuit_kwargs(name: str, layer: KirchhoffLayer, state: Mapping[str, Tensor],
                   base: Mapping[str, Tensor], dt: float | None, t: float | None) -> dict:
    """The solve keywords of a layer solved by Modified Nodal Analysis; see
    `Model._circuit_kwargs`."""
    out: dict = {"t": t if dt is None or t is None else t + dt}
    q_prev = state.get(f"{name}.q")
    if q_prev is not None:
        out["q0"] = q_prev[..., layer.implicit_edges]
    if layer.dynamic and not layer.phasor and dt is not None:
        missing = [k for k in (f"{name}.phi", f"{name}.q") if k not in base]
        if missing:
            raise KeyError(
                f"Model: state {missing} is required to step Kirchhoff layer {name!r}, "
                f"whose dynamic laws start each step from the potentials and flows at "
                f"the step start (an initial point: layer.solve(..., mode='initial'))"
            )
        out["history"] = (base[f"{name}.phi"], base[f"{name}.q"])
        out["dt"] = dt
    return out


def run_pass(
    model: Model, state, drivers, dt, solve_kwargs, *, step_from: State | None = None,
    t: float | None = None, boundary_transfers: bool | Collection[str] = False,
    produced: list[str] | None = None, warm_start: bool = True,
) -> tuple[State, dict, Drivers]:
    """One closures -> potential -> allocation -> transport -> reactions pass of `model`;
    the arguments and the returned `(state, diagnostics, drivers)` are `Model._pass`'s."""
    base: State = state if step_from is None else step_from
    # Closure-carried state must be evaluated from the STEP-START state
    # on every pass, never from the previous pass's own output -- a closure that
    # integrates (adds dt each call) would otherwise integrate once per PASS instead of
    # once per STEP, because `_iterate` feeds pass k-1's output into pass k. Every other
    # key a closure reads still sees `state` (pass k-1's relaxed transport `.x`, which is
    # exactly what the relaxed coupling wants); only the keys this model's closures
    # themselves carry are pinned to `base`.
    closure_state: State = dict(state)
    for key in model.closure_state_keys:
        if key in base:
            closure_state[key] = base[key]
    written: set[str] = set()
    ctx = StepContext(dt=dt, t=t)
    drv = model._apply_closures(closure_state, drivers, written, ctx)
    new: State = dict(state)
    made: list[str] = [] if produced is None else produced
    made.clear()
    diag: dict = {}
    _solve_potentials(model, state, base, drv, dt, t, solve_kwargs, warm_start,
                      new, made, diag)
    _step_allocations(model, base, drv, dt, new, made, diag)
    transfer_layers = _transfer_layers(model, boundary_transfers, dt)
    _advance_transport(model, base, drv, dt, transfer_layers, new, made, diag)
    _collect_closure_state(model, drv, written, new, made)
    return new, diag, drv


def _solve_potentials(model: Model, state, base, drv, dt, t, solve_kwargs, warm_start,
                      new: dict, made: list, diag: dict) -> None:
    """Solve every Kirchhoff layer at the closures' drivers, into `new`/`made`/`diag`."""
    for name, layer in model.kirchhoff.items():
        pb = require(drv, f"{name}.phi_boundary")
        sources = drv.get(f"{name}.sources")
        phi_prev = state.get(f"{name}.phi") if warm_start else None
        phi0 = None if phi_prev is None else phi_prev[..., layer.interior]
        d: dict = {}
        circuit = circuit_kwargs(name, layer, state, base, dt, t) if layer.mna else {}
        if not warm_start:
            # The implicit-law flows' warm start goes with the potentials' (`phi0`).
            circuit.pop("q0", None)
        phi, q = layer.solve(pb, drv, sources, phi0=phi0, diagnostics=d, **circuit,
                             **layer_kwargs(solve_kwargs, name))
        new[f"{name}.phi"], new[f"{name}.q"] = phi, q
        made += [f"{name}.phi", f"{name}.q"]
        diag[name] = d


def _step_allocations(model: Model, base, drv, dt, new: dict, made: list,
                      diag: dict) -> None:
    """Take every allocation layer's one explicit step from the step-start state."""
    if model.allocation and dt is None:
        raise ValueError(
            "Model: a steady (dt=None) pass has no defined meaning for a "
            "AllocatedFlowLayer, which is inherently discrete-time"
        )
    for name, layer in model.allocation.items():
        # `base`, not `state` -- the same step-start rule the comment above states for
        # closure-carried state and the transport loop below follows for `"<layer>.x"`:
        # a layer that INTEGRATES its own state must advance from the STEP-START state
        # on every pass, never from the previous pass's output, or `coupling="iterate"`
        # integrates it once per PASS instead of once per STEP.
        s_prev = base.get(f"{name}.s")
        if s_prev is None:
            raise KeyError(
                f"Model: state {name + '.s'!r} is required to step allocation "
                f"layer {name!r}"
            )
        cd: dict = {}
        s_new, f = layer.step(s_prev, drv, dt, diagnostics=cd)
        new[f"{name}.s"], new[f"{name}.q"] = s_new, f
        made += [f"{name}.s", f"{name}.q"]
        diag[name] = cd


def _transfer_layers(model: Model, boundary_transfers, dt) -> set[str]:
    """The transport layers to run `step_with_transfer` on."""
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
        set(model.transport) if boundary_transfers is True
        else set(boundary_transfers) if boundary_transfers
        else set()
    )
    if dt is None and transfer_layers:
        raise ValueError(
            "Model: boundary_transfers has no meaning for a steady solve, which "
            "integrates nothing"
        )
    return transfer_layers


def _advance_transport(model: Model, base, drv, dt, transfer_layers: set[str], new: dict,
                       made: list, diag: dict) -> None:
    """Advance (or, with `dt=None`, solve to steady) every transport layer on the flows of
    its kinds, applying the reactions on the side of the step `reaction_order` names."""
    for name, layer in model.transport.items():
        want_transfer = name in transfer_layers
        q_kind = model._kind_flows(name, new, drv)
        xb = require(drv, f"{name}.x_boundary")
        sources = drv.get(f"{name}.sources")
        # A per-step capacity, written by a closure that owns the geometry
        # (a sewer conduit's wetted volume, a headspace volume). Absent, the layer's
        # construction-time capacity stands, so nothing changes for a fixed-storage
        # layer. Shape and positivity are checked by the layer, naming the nodes.
        cap = drv.get(f"{name}.capacity")
        # A per-step first-order removal rate (1/s), written by a closure that owns the
        # sinks (deposition, a scheduled removal). Absent, the layer's own stands.
        removal = drv.get(f"{name}.removal")
        # A per-step transmission, written by a closure whose filters depend on the
        # step's flows or conditions (a UVGI filter). Absent, the layer's own stands.
        transmission = drv.get(f"{name}.transmission")
        transfer_total = None
        # A fresh dict per layer, threaded into `steady`/`step`/
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
                    model._zero_sources(layer, xb, layer.n_b)
                    if like is None
                    else model._zero_sources(layer, like, layer.n_i)
                )
            x = layer.steady(
                q_kind, sources, xb, capacity=cap, removal=removal,
                transmission=transmission, diagnostics=layer_diag
            )
            if cap is not None:
                # The capacity this state was solved with, as a step records its own.
                new[f"{name}.capacity"] = cap
                made.append(f"{name}.capacity")
        else:
            x = base.get(f"{name}.x")
            if x is None:
                raise KeyError(
                    f"Model: state {name + '.x'!r} is required to step transport layer "
                    f"{name!r}"
                )
            if sources is None:
                sources = model._zero_sources(layer, x, layer.n_i)
            if model.reaction_order == "before_transport":
                # R then T: react the step-start state over the whole step, then
                # transport it (over the step-start storage `cap_prev` in the amount
                # form below, exactly like SWMM's reaction over the old volume).
                for lname, reaction in model.reactions:
                    if lname == name:
                        x = reaction.apply(x, dt, drv)
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
            k = model.substeps[name]
            for j in range(1, k + 1):
                if cap is None:
                    cap_j = cap_prev_j = None
                else:
                    cap_prev_j = cap_prev + (j - 1) / k * (cap - cap_prev)
                    cap_j = cap_prev + j / k * (cap - cap_prev)
                if want_transfer:
                    stepped = layer.step_with_transfer(
                        x, q_kind, sources, xb, dt / k,
                        capacity=cap_j, capacity_prev=cap_prev_j, removal=removal,
                        transmission=transmission, diagnostics=layer_diag,
                    )
                    x = stepped.x
                    transfer_total = (
                        stepped.boundary_transfer if transfer_total is None
                        else transfer_total + stepped.boundary_transfer
                    )
                else:
                    x = layer.step(x, q_kind, sources, xb, dt / k, capacity=cap_j,
                                   capacity_prev=cap_prev_j, removal=removal,
                                   transmission=transmission, diagnostics=layer_diag)
            if cap is not None:
                new[f"{name}.capacity"] = cap
                made.append(f"{name}.capacity")
            if model.reaction_order == "after_transport":
                for lname, reaction in model.reactions:
                    if lname == name:
                        x = reaction.apply(x, dt, drv)
        new[f"{name}.x"] = x
        made.append(f"{name}.x")
        diag[name] = {"substeps": model.substeps[name]}
        if "linear" in layer_diag:
            diag[name]["linear"] = layer_diag["linear"]
        if "held_sources_dropped" in layer_diag:
            diag[name]["held_sources_dropped"] = layer_diag["held_sources_dropped"]
        if want_transfer:
            diag[name]["boundary_transfer"] = transfer_total


def _collect_closure_state(model: Model, drv, written: set[str], new: dict,
                           made: list) -> None:
    """Copy every closure-carried state key out of the closures' return into `new`."""
    # Closure-carried state: a declared key is copied OUT of the closure's
    # return into the state, so the next step's closures read it back. A closure that
    # declares a key and does not write it every call would freeze that state silently,
    # so the omission is refused by name instead. The check is on `written` (which
    # closure ACTUALLY returned it this call), not on `key in drv` -- a same-named
    # DRIVER the caller passed in would otherwise already be sitting in `drv` and defeat
    # this check, freezing the closure's state at the caller's value with no error.
    for key, closure in model.closure_state_keys.items():
        if key not in written:
            raise KeyError(
                f"Model: closure {closure!r} declares the state key {key!r} but did "
                f"not return it; a closure must write every key it declares on every "
                f"call"
            )
        new[key] = drv[key]
        made.append(key)
