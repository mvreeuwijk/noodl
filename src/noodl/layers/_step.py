"""What one solve evaluates its laws and storage in.

Answers: for this call (mode, dt, t, scheme, history, storage flows), what is the step
context (`_Step`, `_StorageStep`) the laws and storages are evaluated in, and is the call
well formed (`check_scheme`, the forward-Euler and head-off-dynamic-edge checks)? A mixin
of `KirchhoffLayer`: its methods use the layer's laws, storage and unknown layout.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from noodl.elements.base import SCHEMES, LawContext
from noodl.layers._flow_unknowns import (
    FlowUnknowns,
    _Held,
    angular_frequency,
)


@dataclass(frozen=True)
class _StorageStep:
    """The storage of one step: the step-start potentials `phi_prev` (full-node), the flows
    into every storage entry at the step start, `previous` (`None` when not given; the
    trapezoidal scheme needs it), and `columns[j]`, storage `j`'s own columns of the two
    (`(phi_start, previous)`, the second `None` without `previous`). The columns are
    constants of the step, taken once when it is built (`KirchhoffLayer._storage_step`),
    not at every residual."""

    phi_prev: torch.Tensor
    previous: torch.Tensor | None
    columns: tuple


@dataclass(frozen=True)
class _Step:
    """What one solve evaluates its laws and storage in: the law context (mode, dt, t,
    scheme, the step-start dp_prev/q_prev of the laws that read them, omega on a phasor
    layer) and what the storage does (`None`; a `_StorageStep` on a step; the held initial
    rows, `_Held`, at the initial point). Built once per solve by `KirchhoffLayer._step`.

    `layout` is the flow-unknown layout this one solve uses when it is not the layer's own:
    the initial point of a layer without flow unknowns whose storage holds an initial
    potential is solved with the held rows on the sparse operator, in a layout with no
    unknown flows (`KirchhoffLayer._held_layout`). `None` everywhere else."""

    ctx: LawContext
    storage: _StorageStep | _Held | None
    layout: FlowUnknowns | None = None


def check_scheme(layer_name: str, scheme) -> None:
    """Refuse a time-integration scheme a `KirchhoffLayer` cannot take, by name.

    A circuit or network layer integrates with `"backward_euler"` or `"trapezoidal"`, or
    with `"forward_euler"` when its storage sits at boundary nodes (the layer checks that);
    `"exact"` (the exponential of a linear transport step) belongs to a `TransportLayer`,
    and anything else is not a scheme name. The message lists `SCHEMES`.
    """
    if scheme == "exact":
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: scheme='exact' is a TransportLayer scheme; a "
            f"KirchhoffLayer takes 'backward_euler' or 'trapezoidal' ('forward_euler' for "
            f"storage at boundary nodes; schemes: {SCHEMES})"
        )
    if scheme not in SCHEMES:
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: scheme={scheme!r}; expected one of {SCHEMES} "
            f"('exact' is for a TransportLayer only)"
        )


class _StepMixin:
    """The `_step` methods of `KirchhoffLayer`; see the module docstring."""

    def _public_step(self, phi_boundary, dt, t, scheme, history, storage_flow,
                     drivers) -> tuple[_Step, torch.Tensor]:
        """The `_Step` of `residual`, `jacobian` and `adjoint`, and every boundary node's
        potential (`_hold_boundary`): their step keywords, checked as `solve` checks them
        (refused on a layer that takes no step)."""
        if scheme is None:
            scheme = self.scheme
        self._check_forward_euler(scheme)
        phi_boundary, history = self._hold_boundary(phi_boundary, history, dt)
        return self._step(None, dt, t, scheme, history, storage_flow, drivers), phi_boundary

    def _check_forward_euler(self, scheme) -> None:
        """Refuse `"forward_euler"` on a layer that steps anything inside its solve (storage
        at interior nodes, dynamic laws), and any other scheme on a layer whose storage sits
        at boundary nodes, by name."""
        if scheme == "forward_euler" and self.dynamic:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: scheme='forward_euler' steps storage at "
                f"boundary nodes only; this layer steps storage at interior nodes or dynamic "
                f"laws inside its solve, which take 'backward_euler' or 'trapezoidal'"
            )
        if self._boundary_storage and scheme != "forward_euler":
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: its storage sits at boundary nodes, which "
                f"step by scheme='forward_euler' only; got scheme={scheme!r}"
            )

    def _step(self, mode, dt, t, scheme, history, storage_flow, drivers, *,
              check_dt: bool = True, omega: bool = True) -> _Step:
        """The `_Step` of one solve (and of `residual`, `jacobian`, `adjoint` and
        `storage_flows`): the context its laws are evaluated in and what its storage does.

        `mode=None` is a step when `dt` is given and the steady operating point (`"dc"`)
        otherwise; `"dc"` and `"initial"` (the consistent initial point) take no `dt`, and
        `"phasor"`, on a phasor layer only (which is always in it), no `dt` or `history`. A
        step needs `dt > 0` and, from the step start, `history=(phi, q)`: required on a layer
        with a storage or a dynamic law, with `q` required only by a dynamic law and `None`
        otherwise. A layer with storage under the trapezoidal scheme needs `storage_flow`,
        the flow into every storage entry at the step start; `history=(phi, None)` is the
        step of a layer whose storage is all that steps. A layer with no storage, no dynamic
        law and no law that reads the context takes no step: it refuses `dt` and `history`.
        In practice that is a fast-path layer only: a layer with an ideal or implicit law
        reads the context, so it steps. A scheme outside `SCHEMES` is refused on every
        layer but a phasor one, which ignores it. Every keyword is checked by name;
        `check_dt=False` skips the positivity of `dt` (a host sync for a tensor `dt`) when
        the solve of the same step has checked it, and `omega=False` leaves a phasor layer's
        angular frequency out (`storage_flows` reads no law)."""
        name = self.name
        if self.phasor:
            if mode not in (None, "phasor"):
                raise ValueError(
                    f"KirchhoffLayer {name!r} is a phasor layer: mode={mode!r}; expected "
                    f"'phasor' or None"
                )
            if dt is not None or history is not None:
                raise ValueError(
                    f"KirchhoffLayer {name!r} is a phasor layer, which solves the "
                    f"sinusoidal steady state and takes no step; drop dt and history"
                )
        else:
            # a phasor layer ignores the scheme
            check_scheme(name, scheme)
            if mode == "phasor":
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode='phasor' needs a layer built with "
                    f"phasor=True"
                )
            if mode is not None and mode not in ("dc", "initial"):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode={mode!r}; expected 'dc', 'initial' or "
                    f"None (transient when dt is given, dc otherwise)"
                )
            if mode is not None and dt is not None:
                what = ("the consistent initial point" if mode == "initial"
                        else "the steady operating point")
                raise ValueError(
                    f"KirchhoffLayer {name!r}: mode={mode!r} is {what}, which takes no step; "
                    f"drop dt"
                )
        if storage_flow is not None and not self._storage:
            raise ValueError(f"KirchhoffLayer {name!r} has no storage; drop storage_flow")
        if not self._takes_step:
            stepping = [k for k, v in (("dt", dt), ("history", history)) if v is not None]
            if stepping:
                raise ValueError(
                    f"KirchhoffLayer {name!r} has no storage, no dynamic law and no law that "
                    f"reads the step's context, so it takes no step; drop {stepping}"
                )
        if dt is None:
            extra = [k for k, v in (("history", history), ("storage_flow", storage_flow))
                     if v is not None]
            if extra:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: {extra} given without dt; a steady solve "
                    f"takes no step"
                )
            if self.phasor:
                ctx = LawContext(mode="phasor", t=t,
                                 omega=angular_frequency(self, drivers) if omega else None)
                return _Step(ctx, None)
            ctx = LawContext(mode="dc" if mode is None else mode, t=t)
            found = (self._held_values(drivers, self._owner_call, check=True)
                     if mode == "initial" and self._storage else None)
            if found is None:
                return _Step(ctx, None)
            if self._unknowns is not None:
                return _Step(ctx, self._held(found, self._unknowns))
            # a layer without flow unknowns: this one solve holds the rows on the sparse
            # operator, in a layout with no unknown flows
            u = self._held_layout()
            return _Step(ctx, self._held(found, u), u)
        if not check_dt:
            # the solve of this step has checked it
            ...
        elif isinstance(dt, (int, float)):
            if not dt > 0:
                raise ValueError(f"KirchhoffLayer {name!r}: dt must be positive, got {dt!r}")
        elif not bool((torch.as_tensor(dt) > 0).all()):
            raise ValueError(f"KirchhoffLayer {name!r}: dt must be positive, got {dt!r}")
        if len(self._dynamic_edges):
            self._check_head_off_dynamic_edges(drivers)
        ctx = LawContext(mode="transient", dt=dt, t=t, scheme=scheme)
        if history is None:
            if len(self._dynamic_edges):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a transient step needs history=(phi, q) "
                    f"at the step start"
                )
            if self._storage:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a step of a layer with storage needs "
                    f"history=(phi, q) at the step start"
                )
            return _Step(ctx, None)
        if not isinstance(history, (tuple, list)) or len(history) != 2:
            raise ValueError(
                f"KirchhoffLayer {name!r}: history must be the pair (phi, q) at the step "
                f"start; got {type(history).__name__}"
            )
        phi_prev, q_prev = history
        n_edges = len(self.cols)
        if q_prev is None:
            if len(self._dynamic_edges):
                raise ValueError(
                    f"KirchhoffLayer {name!r}: a layer with a dynamic law needs "
                    f"history=(phi, q) with q, every edge's flow at the step start"
                )
            if phi_prev.shape[-1] != self._n_nodes:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: history's phi must hold every node's "
                    f"potential ({self._n_nodes}); got last dimension {phi_prev.shape[-1]}"
                )
        elif phi_prev.shape[-1] != self._n_nodes or q_prev.shape[-1] != n_edges:
            raise ValueError(
                f"KirchhoffLayer {name!r}: history=(phi, q) must hold every node's "
                f"potential ({self._n_nodes}) and every edge's flow ({n_edges}); got last "
                f"dimensions {phi_prev.shape[-1]} and {q_prev.shape[-1]}"
            )
        if self._reads_history:
            ctx = self._with_history(ctx, phi_prev, q_prev)
        if not self._storage:
            return _Step(ctx, None)
        if storage_flow is None:
            if scheme == "trapezoidal":
                raise ValueError(
                    f"KirchhoffLayer {name!r}: the trapezoidal scheme needs storage_flow, "
                    f"the flow into every storage entry at the step start "
                    f"(storage_flows; the state key {name + '.storage_flow'!r})"
                )
        elif storage_flow.shape[-1] != self._n_entries:
            raise ValueError(
                f"KirchhoffLayer {name!r}: storage_flow must hold one flow per storage "
                f"entry ({self._n_entries}); got last dimension {storage_flow.shape[-1]}"
            )
        return _Step(ctx, self._storage_step(phi_prev, storage_flow))

    def _storage_step(self, phi_prev, previous) -> _StorageStep:
        """The `_StorageStep` from the step-start potentials `phi_prev` and storage flows
        `previous`, with every storage's columns of the two."""
        columns = tuple(
            (phi_prev[..., st.nodes], None if previous is None else previous[..., s:e])
            for st, (s, e) in zip(self._storage, self._storage_slices, strict=True))
        return _StorageStep(phi_prev, previous, columns)

    def _with_history(self, ctx: LawContext, phi_prev, q_prev) -> LawContext:
        """`ctx` with the step-start potential differences and flows taken from
        `(phi_prev, q_prev)`."""
        return dataclasses.replace(ctx, dp_prev=self._difference(phi_prev), q_prev=q_prev)

    def _check_head_off_dynamic_edges(self, drivers) -> None:
        """Refuse a head driver that is not zero on an edge with a dynamic law: the step-start
        potential difference of such an edge is taken without a head.

        `EdgeHeads` guarantees this by construction (it refuses a term on a dynamic kind), so
        the value check is only for a head driver given directly, and is made on a CPU tensor
        only: reading values off an accelerator is a host sync per solve, which this path does
        not pay. The width is always checked (no sync). Free when there is no head or no
        dynamic law.
        """
        head = drivers.get(self.head_key)
        if head is None:
            return
        self._check_head_width(head)
        if head.device.type != "cpu":
            return
        if bool((head[..., self._dynamic_edges] != 0).any()):
            kinds = sorted({el.kind for el in self._laws if el.dynamic})
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: driver {self.head_key!r} is not "
                f"zero on an edge of kind {kinds}, whose law is dynamic (its step-start "
                f"potential difference is taken without a head)"
            )
