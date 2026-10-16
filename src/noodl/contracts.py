"""The closure contract: what a closure is called with, what it may declare, and how the
model reads those declarations (once, at construction).

A closure is any callable `closure(state, drivers) -> {key: tensor}` that adds or replaces
drivers (`AlgebraicClosure`). One that advances state of its own over the step declares
`integrates = True` and is called `closure(state, drivers, ctx)` with a `StepContext`
(`IntegratingClosure`). Everything else a closure may say about itself is an optional
attribute; `CLOSURE_ATTRIBUTES` is the table of them, and `closure_spec` is the ONLY code
that reads those attributes: it turns them into one normalised `ClosureSpec`, and refuses a
malformed declaration by name. `ClosureSpec.__call__` is the one place that decides between
the two-argument and the three-argument call.

The dict state/driver interface is unchanged: this module adds no new type to what a closure
receives or returns.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

__all__ = [
    "StepContext",
    "AlgebraicClosure",
    "IntegratingClosure",
    "ClosureAttribute",
    "CLOSURE_ATTRIBUTES",
    "ClosureSpec",
    "closure_spec",
    "call_closure",
    "reaction_species_columns",
    "check_species_columns",
]


@dataclass(frozen=True)
class StepContext:
    """What a state transition needs from the model: the interval it must integrate over
    (`None` under `steady()`, which an integrating closure refuses) and, when the caller
    tracks it, the time at the start of the step."""

    dt: float | None
    t: float | None = None


@runtime_checkable
class AlgebraicClosure(Protocol):
    """A closure that only computes drivers from the state and the drivers: called as
    `closure(state, drivers)`, it returns the driver entries it adds or replaces.

    As a runtime check, `isinstance(obj, AlgebraicClosure)` is true of ANY callable (a
    runtime-checkable Protocol checks that members exist, not their signatures);
    `closure_spec` is what checks the arity against the declaration."""

    def __call__(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]: ...


@runtime_checkable
class IntegratingClosure(Protocol):
    """A closure that advances state of its own over the step (`integrates = True`).

    It is called `closure(state, drivers, ctx)`. `ctx` is the `StepContext` of the step, or
    `None` for a query (`residuals`, `current_flows`, ...), where it evaluates its algebraic
    outputs at the given state without advancing. Under `steady()` (`ctx.dt is None`) it is
    refused by name: there is no interval to integrate over.

    As a runtime check, `isinstance(obj, IntegratingClosure)` is true of a callable that HAS
    an `integrates` attribute, whatever its value: it tells a closure that declares
    `integrates` (True or False) from a plain `AlgebraicClosure`, which does not. Whether it
    is called with a `StepContext` is decided by the value (`ClosureSpec.integrates`).
    """

    integrates: bool

    def __call__(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor],
        ctx: StepContext | None,
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class ClosureAttribute:
    """One optional attribute of a closure: its name, the type it must have, what it
    defaults to when absent, what it means and which code consumes it."""

    name: str
    type: str
    default: object
    meaning: str
    consumer: str


CLOSURE_ATTRIBUTES: tuple[ClosureAttribute, ...] = (
    ClosureAttribute(
        "integrates", "bool", None,
        "True: the closure advances state over the step and is called with a StepContext "
        "as a third argument; False: it only carries state that something else advances. "
        "Absent: an algebraic closure.",
        "closure_spec; ClosureSpec.__call__"),
    ClosureAttribute(
        "state_keys", "tuple[str, ...]", (),
        "State keys the closure both reads and writes back on every call, copied from its "
        "return into the returned state. Needs `integrates` declared (True or False). "
        "One closure per key; never a layer's own state key.",
        "Model (closure-carried state)"),
    ClosureAttribute(
        "key_labels", "Mapping[str, tuple]", {},
        "Labels, in order, of the entries of an input or state key the closure declares.",
        "noodl.refs (LayerRefs)"),
    ClosureAttribute(
        "inputs", "tuple[str, ...]", None,
        "The state and driver keys the closure reads. Absent: undeclared.",
        "noodl.validation (Model.check)"),
    ClosureAttribute(
        "outputs", "tuple[str, ...]", None,
        "The driver keys the closure writes. Absent: undeclared.",
        "noodl.validation (Model.check); noodl.couple"),
    ClosureAttribute(
        "input_specs", "Mapping[str, Field]", {},
        "Field specifications of the keys the closure reads, each with its `unit`. Read "
        "lazily: a closure may compute it in a property.",
        "noodl.refs (LayerRefs); noodl.validation (Model.check: unit-mismatch)"),
    ClosureAttribute(
        "output_units", "Mapping[str, str]", {},
        "{key: unit string} of the driver keys the closure writes (`outputs`) and of the state "
        "it carries (`state_keys`; the unit of the Field `key_labels` makes). Compared, "
        "after trivial respelling (`m3`, `m^3`, `m³`; `1`, `-`), with the unit of the "
        "layer field or declared input the key refers to; nothing is converted.",
        "noodl.validation (Model.check: unit-mismatch)"),
    ClosureAttribute(
        "edge_kinds", "tuple[str, ...]", (),
        "Edge kinds of the network the closure works on itself, so that no layer uses them.",
        "noodl.validation (Model.check)"),
    ClosureAttribute(
        "requires_coupling", "str | None", None,
        'Set to "iterate" when the closure is only correct inside the iterated coupling.',
        "Model"),
    ClosureAttribute(
        "reads_flows_of", "str | None", None,
        "The Kirchhoff layer whose flows the closure reads (with requires_coupling).",
        "Model"),
    ClosureAttribute(
        "species_columns", "Mapping[str, Mapping[str, int]]", {},
        "{transport layer: {species name: column}} the closure addresses by index; checked "
        "against the layer's species_names at construction (check_species_columns).",
        "Model; check_species_columns"),
)

_SEQUENCE_ATTRIBUTES = ("inputs", "outputs", "state_keys", "edge_kinds")


@dataclass(frozen=True, eq=False)
class ClosureSpec:
    """A closure and its declarations, read once by `closure_spec`.

    `closure` is the user's object, unchanged; `label` is its `repr`, as messages name it.
    `reads` and `writes` are `None` when the closure does not declare `inputs` / `outputs`
    (so a checker can tell "declares nothing" from "declares none"). `input_specs` is read
    from the closure each time, because some closures compute it in a property.
    """

    closure: Callable
    label: str
    integrates: bool
    declares_integrates: bool
    state_keys: tuple[str, ...]
    key_labels: Mapping[str, tuple]
    reads: frozenset[str] | None
    writes: frozenset[str] | None
    edge_kinds: tuple[str, ...]
    requires_coupling: str | None
    reads_flows_of: str | None
    species_columns: Mapping[str, Mapping[str, int]]
    output_units: Mapping[str, str] = field(default_factory=dict)
    owner: str = "Model"

    @property
    def input_specs(self) -> Mapping[str, Any]:
        """The closure's `input_specs`, `{}` when it declares none (evaluated on each read)."""
        return getattr(self.closure, "input_specs", None) or {}

    def __call__(
        self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor],
        ctx: StepContext | None = None,
    ) -> Mapping[str, Tensor]:
        """Call the closure: with `ctx` as a third argument if it integrates, with two
        arguments otherwise. An integrating closure under `ctx.dt is None` (`steady()`)
        raises `ValueError`."""
        if self.integrates:
            if ctx is not None and ctx.dt is None:
                raise ValueError(
                    f"{self.owner}: closure {self.label} declares integrates=True and cannot "
                    f"be evaluated by steady(), which has no interval to integrate over"
                )
            return self.closure(state, drivers, ctx)
        return self.closure(state, drivers)


def _names(owner: str, label: str, attr: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{owner}: closure {label} declares {attr} as the string {value!r}; declare "
            f"a tuple of names, e.g. {attr} = ({value!r},)"
        )
    try:
        return tuple(str(k) for k in value)
    except TypeError:
        raise TypeError(
            f"{owner}: closure {label} declares {attr} as a {type(value).__name__}, not a "
            f"tuple of names"
        ) from None


def _species_columns(
    owner: str, label: str, value: Any, kind: str = "closure",
) -> dict[str, dict[str, int]]:
    ok = isinstance(value, Mapping) and all(
        isinstance(layer, str) and isinstance(cols, Mapping) and all(
            isinstance(name, str) and isinstance(c, int) and not isinstance(c, bool)
            for name, c in cols.items())
        for layer, cols in value.items()
    )
    if not ok:
        raise TypeError(
            f"{owner}: {kind} {label} declares species_columns that is not "
            f"{{layer: {{species: column}}}} with str names and int columns; got {value!r}"
        )
    return {layer: dict(cols) for layer, cols in value.items()}


def _output_units(owner: str, label: str, value: Any) -> dict[str, str]:
    ok = isinstance(value, Mapping) and all(
        isinstance(k, str) and isinstance(u, str) for k, u in value.items())
    if not ok:
        raise TypeError(
            f"{owner}: closure {label} declares output_units that is not {{key: unit}} with "
            f"str keys and str units; got {value!r}"
        )
    return dict(value)


def _positional_arity(closure: Callable) -> tuple[int, int, bool] | None:
    """`(positional, required positional, has *args)` of `closure`'s call signature, or
    `None` when it cannot be inspected (some builtins and extension callables), in which
    case the closure is accepted as declared."""
    try:
        params = inspect.signature(closure).parameters.values()
    except (ValueError, TypeError):
        return None
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    var_positional = any(p.kind is p.VAR_POSITIONAL for p in params)
    return len(positional), len(required), var_positional


def closure_spec(closure: Callable, *, index: int, owner: str = "Model") -> ClosureSpec:
    """Read `closure`'s declarations into a `ClosureSpec`: the only reader of the closure
    attributes (`CLOSURE_ATTRIBUTES`). `index` is the closure's position, for the message
    about a non-callable; `owner` prefixes every message.

    Raises `TypeError` for a closure that is not callable, a `str` given for `inputs`,
    `outputs`, `state_keys` or `edge_kinds`, a non-bool `integrates`, or a malformed
    `species_columns` or `output_units`, or a closure that does not declare
    `integrates = True` but whose call signature REQUIRES three or more positional
    arguments (it would be called with two); `ValueError` for `state_keys` without
    `integrates`, and for `integrates = True` on a call signature with fewer than three
    positional parameters (and no `*args`). A callable whose signature cannot be inspected
    is accepted as declared.
    """
    if not callable(closure):
        raise TypeError(
            f"{owner}: closure {index} is a {type(closure).__name__}, which is not "
            f"callable; a closure is called as closure(state, drivers) and returns "
            f"driver updates"
        )
    label = repr(closure)
    declared = getattr(closure, "integrates", None)
    if declared is not None and not isinstance(declared, bool):
        raise TypeError(
            f"{owner}: closure {label} declares integrates={declared!r}; it must be True or "
            f"False"
        )
    seq = {}
    for attr in _SEQUENCE_ATTRIBUTES:
        value = getattr(closure, attr, None)
        seq[attr] = None if value is None else _names(owner, label, attr, value)
    carries = seq["state_keys"] or ()
    if carries and declared is None:
        raise ValueError(
            f"{owner}: closure {label} declares state_keys {carries} but not "
            f"`integrates`; set `integrates = True` if it advances that state over "
            f"the step (it then receives a StepContext), or `integrates = False` if "
            f"something else advances it"
        )
    arity = _positional_arity(closure)
    if arity is not None:
        n_positional, n_required, var_positional = arity
        if declared and n_positional < 3 and not var_positional:
            raise ValueError(
                f"{owner}: closure {label} declares integrates=True but its call "
                f"signature takes {n_positional} positional arguments; an "
                f"integrating closure is called as closure(state, drivers, ctx)"
            )
        if not declared and n_required >= 3:
            raise TypeError(
                f"{owner}: closure {label} requires {n_required} positional arguments but "
                f"does not declare integrates=True, so it would be called as "
                f"closure(state, drivers); declare `integrates = True` if it advances state "
                f"over the step (it is then called closure(state, drivers, ctx)), or give "
                f"its extra parameters defaults"
            )
    key_labels = getattr(closure, "key_labels", None) or {}
    species = getattr(closure, "species_columns", None) or {}
    return ClosureSpec(
        closure=closure,
        label=label,
        integrates=bool(declared),
        declares_integrates=declared is not None,
        state_keys=carries,
        key_labels=dict(key_labels),
        reads=None if seq["inputs"] is None else frozenset(seq["inputs"]),
        writes=None if seq["outputs"] is None else frozenset(seq["outputs"]),
        edge_kinds=seq["edge_kinds"] or (),
        requires_coupling=getattr(closure, "requires_coupling", None),
        reads_flows_of=getattr(closure, "reads_flows_of", None),
        species_columns=_species_columns(owner, label, species),
        output_units=_output_units(owner, label, getattr(closure, "output_units", None) or {}),
        owner=owner,
    )


def call_closure(
    closure_or_spec: Callable | ClosureSpec,
    state: Mapping[str, Tensor],
    drivers: Mapping[str, Tensor],
    ctx: StepContext | None = None,
) -> Mapping[str, Tensor]:
    """Call a closure (or its `ClosureSpec`) by the one dispatch rule: three arguments for
    an integrating closure, two otherwise. For callers outside `Model` that run a closure
    themselves; a bare closure is read with `closure_spec` first."""
    spec = (
        closure_or_spec if isinstance(closure_or_spec, ClosureSpec)
        else closure_spec(closure_or_spec, index=0)
    )
    return spec(state, drivers, ctx)


def reaction_species_columns(reaction: Any) -> Mapping[str, Mapping[str, int]]:
    """A `Reaction`'s optional `species_columns` (the same member as a closure's), `{}` when
    it declares none. `TypeError` if malformed."""
    value = getattr(reaction, "species_columns", None) or {}
    return _species_columns("Model", repr(reaction), value, kind="reaction")


def check_species_columns(
    label: str,
    declared: Mapping[str, Mapping[str, int]],
    transport: Mapping[str, Any],
) -> None:
    """Raise `ValueError` unless every `{layer: {name: column}}` in `declared` names a
    transport layer of `transport`, has `column < n_species`, and, when the layer has
    `species_names`, `species_names[column] == name`. A layer without `species_names` cannot
    verify names, so it accepts only declarations that give distinct columns. `label` names
    the closure or reaction in the message."""
    for lname, cols in declared.items():
        if lname not in transport:
            raise ValueError(
                f"{label}: species_columns names {lname!r}, not a transport layer of this "
                f"model ({sorted(transport)})"
            )
        layer = transport[lname]
        k = layer.n_species
        names = getattr(layer, "species_names", None)
        for name, col in cols.items():
            if not 0 <= col < k:
                raise ValueError(
                    f"{label}: species_columns[{lname!r}][{name!r}] = {col} is outside the "
                    f"layer's {k} species column(s)"
                )
            if names is not None and names[col] != name:
                raise ValueError(
                    f"{label}: species_columns[{lname!r}][{name!r}] = {col}, but layer "
                    f"{lname!r} has species_names[{col}] = {names[col]!r} "
                    f"(species_names = {tuple(names)})"
                )
        if names is None and len(set(cols.values())) != len(cols):
            raise ValueError(
                f"{label}: species_columns[{lname!r}] gives one column to several species "
                f"({dict(cols)}) and layer {lname!r} has no species_names to tell them apart"
            )
