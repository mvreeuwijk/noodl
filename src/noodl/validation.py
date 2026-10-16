"""Setup validation: check a model's configuration, initial state and inputs before a run.

`Model.check(state, drivers)` (or `check_setup`) returns a `SetupReport` listing, for every
key the model knows the layout of, whether it is required, in which order and unit it is
expected, and what was given; plus the problems found, as errors and warnings:

* configuration -- edge kinds present in the network that no layer uses, transport layers
  whose flows must be prescribed by a driver;
* inputs -- missing required keys, tensors whose trailing shape does not match the key's
  order, batch shapes that do not broadcast, nonzero sources on boundary or inactive nodes
  (which a Kirchhoff layer would silently drop), a state key put in the drivers;
* suspicious keys -- a key with a layer's prefix but no such input (`"thermal.source"`),
  a key whose prefix is a near-miss of a layer name (`"therml.sources"`);
* value range -- a temperature below 150 K (`value-range`: a Celsius value given as kelvin).

Keys the model does not know the layout of -- a custom closure's inputs such as a density
or a weather variable -- are NOT rejected. With `probe=True` the check runs one step (under
`torch.no_grad()`, on the given dictionaries, which it does not modify) while recording
which driver and state keys the closures, layers, laws and drives actually read; a key
read by nothing is then reported as unused. A key read only through a copy a closure made
of its drivers is invisible to the probe, so "unused" is a warning, never an error.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from noodl._broadcast import broadcast_shapes
from noodl.refs import (
    Field,
    input_unit_declarations,
    normalize_unit,
    required_state_keys,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class Issue:
    """One finding: `level` is `"error"`, `"warning"` or `"info"`; `key` the key concerned,
    if any."""

    level: str
    code: str
    message: str
    key: str | None = None

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}: {self.message}"


@dataclass(frozen=True)
class InputRow:
    """One known key: what is expected and what was given."""

    key: str
    role: str
    layer: str | None
    required: str
    ordering: str
    expected: tuple[int, ...]
    unit: str
    given: tuple[int, ...] | None
    status: str


@dataclass
class SetupReport:
    """The result of `check_setup`. `ok` is true when there are no errors."""

    issues: list[Issue] = field(default_factory=list)
    rows: list[InputRow] = field(default_factory=list)
    probed: bool = False
    driver_reads: frozenset[str] = frozenset()
    state_reads: frozenset[str] = frozenset()
    closure_outputs: frozenset[str] = frozenset()

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self, *, strict: bool = False) -> SetupReport:
        """Raise `ValueError` listing every error (and, with `strict`, every warning)."""
        bad = self.errors + self.warnings if strict else self.errors
        if bad:
            raise ValueError(
                "model setup check failed:\n" + "\n".join(f"  {i}" for i in bad)
            )
        return self

    def __str__(self) -> str:
        lines = []
        if self.issues:
            lines += [str(i) for i in self.issues]
        else:
            lines.append("no problems found")
        if self.rows:
            head = ("key", "role", "required", "order", "expected", "unit", "given", "status")
            table = [head] + [
                (r.key, r.role, r.required, r.ordering, str(r.expected), r.unit or "-",
                 "-" if r.given is None else str(r.given), r.status)
                for r in self.rows
            ]
            widths = [max(len(row[c]) for row in table) for c in range(len(head))]
            lines.append("")
            for row in table:
                lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
        if not self.probed:
            lines.append("(custom keys not probed; check(..., probe=True) runs one step)")
        return "\n".join(lines)


class _Probe:
    """What `Model._apply_closures` records while `Model._probe` is set."""

    def __init__(self) -> None:
        self.driver_reads: set[str] = set()
        self.state_reads: set[str] = set()
        self.closure_outputs: set[str] = set()
        # The value each closure output last took, and the closure that wrote it.
        self.closure_values: dict[str, object] = {}
        self.closure_writers: dict[str, str] = {}
        # The keys each closure returned over the whole probe, keyed by the closure's
        # `id()` (not its label: two closures may share a `repr`).
        self.closure_returned: dict[int, set[str]] = {}


def _check_value(f: Field, key: str, value, origin: str, add, batches) -> str:
    """Check one value against its field: a tensor, of the field's trailing shape, zero on
    labels the layer does not solve for. Adds issues; returns the row status."""
    if not isinstance(value, Tensor):
        add(Issue("error", "type",
                  f"{key!r} {origin} as a {type(value).__name__}; the layers take a "
                  f"torch.Tensor of trailing shape {f.trailing} ({f.ordering})", key))
        return "BAD TYPE"
    try:
        k = f.trailing_dims(value)
    except ValueError as exc:
        add(Issue("error", "shape", f"{exc} ({origin})", key))
        return "BAD SHAPE"
    batches.append((key, tuple(value.shape[: value.dim() - k])))
    if f.unit == "K" and bool((value.detach() < 150.0).any()):
        add(Issue("warning", "value-range",
                  f"{key!r} ({origin}) has a value below 150 K; the layer takes absolute "
                  f"temperatures in kelvin, so a Celsius value here is a mistake", key))
    if f.settable is not None and len(f.settable) < len(f.labels):
        bad = [
            lab for lab, col in f.named(value.detach()).items()
            if lab not in f.settable and bool((col != 0).any())
        ]
        if bad:
            add(Issue("error", "value-on-fixed-node",
                      f"{key!r} ({origin}) is nonzero at {bad}, which layer {f.layer!r} "
                      f"does not solve for (boundary or inactive); the value would be "
                      f"refused or silently dropped", key))
            return "BAD VALUES"
    return "ok"


def _both_roles(key: str) -> bool:
    return key.endswith(".capacity")


def _specs(model) -> list:
    return [model._closure_spec(c) for c in model.closures]


def _declared(model, attr: str) -> set[str]:
    """What the closures declare: `attr` is `"outputs"`, `"inputs"` or `"edge_kinds"`."""
    out: set[str] = set()
    for spec in _specs(model):
        out.update({"outputs": spec.writes, "inputs": spec.reads,
                    "edge_kinds": spec.edge_kinds}[attr] or ())
    return out


def _check_units(model, specs, who, add) -> None:
    """`unit-mismatch` (warning): a closure declares a unit, for an input in `input_specs` or
    an output in `output_units`, that differs (after `normalize_unit`) from the unit of the
    layer field or earlier declared input the key refers to. `unit-undeclared` (info): a
    field the model knows has no unit. Units are compared as strings; nothing is converted."""
    refs = model.refs
    fields = refs.fields()
    reference: dict[str, tuple[str, str]] = {}      # key -> (unit, where it was declared)
    for key, f in fields.items():
        if isinstance(f, Field) and f.unit and f.layer is not None:
            reference[key] = (f.unit, f"layer {f.layer!r}'s field")
    for key, f in refs.closure_state.items():
        if isinstance(f, Field) and f.unit:
            reference.setdefault(key, (f.unit, "the closure-carried state"))

    def compare(who_, key: str, unit: str, kind: str) -> None:
        if not unit:
            return
        if key not in reference and kind == "input":
            reference[key] = (unit, who_)
        ref = reference.get(key)
        if ref is None or normalize_unit(ref[0]) == normalize_unit(unit):
            return
        add(Issue("warning", "unit-mismatch",
                  f"{who_} declares {kind} {key!r} in [{unit}], but {ref[1]} {key!r} has "
                  f"unit [{ref[0]}]", key))

    for origin, key, unit in input_unit_declarations(model):
        compare(origin, key, unit, "input")
    for i, spec in enumerate(specs):
        for key, unit in sorted(spec.output_units.items()):
            compare(f"closure {who(i)}", key, unit, "output")
    for key, f in fields.items():
        if isinstance(f, Field) and not f.unit:
            add(Issue("info", "unit-undeclared",
                      f"{f.role} field {key!r} has no unit", key))


def _near_layer(model, prefix: str) -> list[str]:
    return difflib.get_close_matches(prefix, list(model.layers), 1, 0.75)


def check_setup(
    model, state: Mapping | None = None, drivers: Mapping | None = None, *,
    dt: float | None = None, probe: bool = False, steady: bool = False,
    names=None,
) -> SetupReport:
    """Check `model` and, when given, `state` and `drivers`; return a `SetupReport`.

    `steady=True` checks for `Model.steady` (no transport state required); otherwise the
    state a time step needs is required. `probe=True` additionally runs one `step` of `dt`
    (or `steady` when `dt` is None) to learn which keys are read; it needs both
    dictionaries, does not modify them, and costs one model step.
    `names` (a `noodl.components.NameMap`) adds one information line per port no component
    connected.
    """
    report = SetupReport()
    add = report.issues.append
    if names is not None:
        if names.net is not model.net:
            add(Issue("warning", "names-mismatch",
                      "names= was flattened into a different Network than this model's"))
        for port in names.unconnected_ports:
            add(Issue("info", "unconnected-port",
                      f"port {port!r} is connected to nothing; it is closed "
                      f"(no net flow through it)"))
    refs = model.refs
    known = refs.fields()

    # ---------------------------------------------------------------- configuration
    used: set[str] = set()
    for layer in model.kirchhoff.values():
        used.update(layer.kinds)
    for layer in model.transport.values():
        used.update(layer.flow_kinds)
        if getattr(layer, "conduction_kind", None):
            used.add(layer.conduction_kind)
    for layer in model.allocation.values():
        used.add(layer.kind)
    layer_kinds = set(used)
    used |= _declared(model, "edge_kinds")   # kinds a closure works on itself
    net_kinds: dict[str, int] = {}
    for k in model.net.edge_kinds():
        net_kinds[k] = net_kinds.get(k, 0) + 1
    for k, count in net_kinds.items():
        if k not in used:
            # A near-miss of a kind a layer uses is most likely a typo; otherwise the
            # kind may be one a closure works on, so it is only noted.
            hint = difflib.get_close_matches(str(k), sorted(layer_kinds), 1, 0.75)
            add(Issue(
                "warning" if hint else "info", "unused-edge-kind",
                f"edge kind {k!r} has {count} edge(s) but no layer of this model uses it"
                + (f"; did a layer mean {hint[0]!r}?" if hint else ""),
            ))

    # --------------------------------------------------------------- closure ownership
    specs = _specs(model)
    # A closure's label names it in messages only; two closures may share a `repr`, so a
    # repeated label is told apart by the closure's position in `model.closures`.
    label_counts = Counter(spec.label for spec in specs)

    def who(i: int) -> str:
        label = specs[i].label
        return label if label_counts[label] == 1 else f"{label} (closure #{i})"

    last_writer: dict[str, int] = {}
    for i, spec in enumerate(specs):
        if spec.writes is None and spec.reads is None:
            add(Issue("info", "undeclared-closure",
                      f"closure {spec.label} declares neither `inputs` nor `outputs`, so "
                      f"the checks that need them (missing inputs, overwritten outputs) "
                      f"cannot see it"))
        for key in sorted(spec.writes or ()):
            if key in last_writer and key not in (spec.reads or ()):
                add(Issue("warning", "overwritten-output",
                          f"closure {who(i)} declares output {key!r} and replaces what "
                          f"closure {who(last_writer[key])} wrote, without declaring it "
                          f"among its `inputs`", key))
            last_writer[key] = i
            head, dot, tail = key.rpartition(".")
            if dot and head in model.layers:
                ref = model.refs[head]
                suffixes = {str(ref.fields[o]).rpartition(".")[2] for o in ref.inputs}
                if tail not in suffixes:
                    hint = difflib.get_close_matches(tail, sorted(suffixes), 1, 0.6)
                    add(Issue("warning", "unknown-layer-key",
                              f"closure {spec.label} declares output {key!r}, which is not a "
                              f"driver key of layer {head!r} (its driver keys: "
                              f"{sorted(str(ref.fields[o]) for o in ref.inputs)})"
                              + (f"; did you mean {head + '.' + hint[0]!r}?" if hint else ""),
                              key))

    # ------------------------------------------------------------------------ units
    _check_units(model, specs, who, add)

    # ------------------------------------------------------------------- probe step
    probe_obj = None
    closure_values: dict = {}
    closure_writers: dict = {}
    if probe:
        if state is None or drivers is None:
            add(Issue("error", "probe-needs-inputs",
                      "probe=True needs both state and drivers"))
        else:
            probe_obj = _Probe()
            model._probe = probe_obj
            try:
                with torch.no_grad():
                    if dt is None:
                        model.steady(dict(state), dict(drivers))
                    else:
                        model.step(dict(state), dict(drivers), dt)
            except Exception as exc:  # reported, not raised: the report is the product
                hint = (" (probe=True without dt runs a steady solve; pass dt= to probe "
                        "a time step)") if dt is None else ""
                add(Issue("error", "probe-failed",
                          f"one {'step' if dt else 'steady solve'} raised "
                          f"{type(exc).__name__}: {exc}{hint}"))
            finally:
                model._probe = None
            report.probed = True
            report.driver_reads = frozenset(map(str, probe_obj.driver_reads))
            report.state_reads = frozenset(map(str, probe_obj.state_reads))
            report.closure_outputs = frozenset(map(str, probe_obj.closure_outputs))
            closure_values, closure_writers = probe_obj.closure_values, probe_obj.closure_writers
            for i, spec in enumerate(specs):
                undeclared = sorted(
                    probe_obj.closure_returned.get(id(spec.closure), set())
                    - set(spec.writes or ()) - set(spec.state_keys)
                ) if spec.writes is not None else []
                if undeclared:
                    add(Issue("warning", "declared-vs-probed",
                              f"closure {who(i)} returned {undeclared} during the probe "
                              f"step, which it does not declare among its `outputs`"))
    # Without a probe, learn what the closures write by evaluating them once as a QUERY --
    # the no-integration call `Model.initial_capacities` makes: no solve, no time step, and
    # an integrating closure evaluates its algebraic outputs without advancing.
    queried = report.probed
    if not report.probed and model.closures and state is not None and drivers is not None:
        query = _Probe()
        model._probe = query
        try:
            with torch.no_grad():
                model._apply_closures(dict(state), dict(drivers))
            queried = True
            report.closure_outputs = frozenset(map(str, query.closure_outputs))
            closure_values, closure_writers = query.closure_values, query.closure_writers
        except Exception as exc:
            add(Issue("warning", "closure-query-failed",
                      f"evaluating the closures at the given state raised "
                      f"{type(exc).__name__}: {exc}"))
        finally:
            model._probe = None
    produced = _declared(model, "outputs") | set(report.closure_outputs)
    # When the closures' outputs are known -- evaluated, or every closure declares them --
    # a required driver none of them writes is missing for certain.
    all_declared = queried or all(spec.writes is not None for spec in specs)
    consumed = _declared(model, "inputs")

    # --------------------------------------------------------------------- inputs
    batches: list[tuple[str, tuple[int, ...]]] = []
    given_any = {"driver": drivers, "state": state}
    for key, f in known.items():
        if not isinstance(f, Field):
            continue
        data = given_any[f.role]
        value = None if data is None else data.get(key)
        status = "ok"
        if value is None:
            status = "not given"
            need = f.required == "always" or (f.required == "step" and not steady)
            if data is not None and need:
                if key in produced:
                    status = "from closure"
                elif (f.role == "driver" and model.closures and not report.probed
                      and not all_declared and key not in refs.inputs):
                    # (A declared input is read by the very closure, reaction, element or
                    # drive that declares it: nothing else will write it, so its absence is
                    # an error without a probe.)
                    status = "missing?"
                    add(Issue("warning", "missing-input",
                              f"{key!r} ({f.description}, {f.ordering}) is required and "
                              f"was not given; fine only if a closure writes it "
                              f"(check(..., probe=True) finds out)", key))
                else:
                    status = "MISSING"
                    add(Issue("error", "missing-input",
                              f"{key!r} ({f.description}, {f.ordering}, trailing shape "
                              f"{f.trailing}) is required and was not given", key))
        else:
            status = _check_value(f, key, value, "was given", add, batches)
        # A closure's output is what the layers actually receive: check it the same way,
        # whether it fills a key that was not given or overwrites one that was.
        if f.role == "driver" and key in closure_values:
            writer = closure_writers.get(key, "a closure")
            status = _check_value(
                f, key, closure_values[key], f"was written by {writer}", add, batches,
            ) if status in ("ok", "not given", "from closure") else status
            if status == "ok" and value is None:
                status = "from closure"
            value = closure_values[key] if value is None else value
        report.rows.append(InputRow(
            key, f.role, f.layer, f.required, f.ordering, f.trailing, f.unit,
            tuple(value.shape) if isinstance(value, Tensor) else None, status,
        ))
    if len(batches) > 1:
        try:
            broadcast_shapes(*(b for _, b in batches))
        except (RuntimeError, ValueError):
            add(Issue("error", "batch-shape",
                      "batch shapes do not broadcast: "
                      + ", ".join(f"{k}: {b}" for k, b in batches)))

    if state is not None and not steady:
        for key in required_state_keys(model):
            if key not in state and not isinstance(known.get(key), Field):
                add(Issue("error", "missing-state",
                          f"closure-carried state {key!r} is required and was not given", key))
        # Only for a transport layer this model HAS: an application closure may write
        # the capacity of a layer the model was built without.
        for key in sorted(k for k in produced if k.endswith(".capacity")
                          and k.rpartition(".")[0] in model.transport):
            if key not in state:
                add(Issue("error", "missing-state",
                          f"a closure writes the driver {key!r}, so the step-start state "
                          f"must carry it; build it with Model.initial_capacities", key))

    for role, data, reads in (("driver", drivers, report.driver_reads),
                              ("state", state, report.state_reads)):
        if data is None:
            continue
        for key in data:
            key = str(key)
            f = known.get(key)
            if f is not None:
                # "<layer>.capacity" is both: the driver a closure writes and the storage
                # the step-start state carries (`Model.initial_capacities`).
                if isinstance(f, Field) and f.role != role and not _both_roles(key):
                    add(Issue("error", "wrong-dictionary",
                              f"{key!r} is a {f.role} key but was given in the {role}s", key))
                continue
            head, dot, tail = key.rpartition(".")
            if role == "driver" and head in model.transport and tail == "q":
                add(Issue("error", "two-flow-sources",
                          f"{key!r} given, but layer {head!r} takes its flows from potential "
                          f"layer {model.flow_layer_of[head]!r}", key))
                continue
            if report.probed and key in reads:
                continue
            if key in consumed or (role == "state" and key in model.closure_state_keys):
                continue
            if role == "state" and key.endswith(".capacity") and head in model.transport:
                continue
            if dot and head in model.layers:
                options = sorted(refs[head].inputs if role == "driver" else refs[head].state)
                suffixes = [str(refs[head].fields[o]).rpartition(".")[2] for o in options]
                hint = difflib.get_close_matches(tail, suffixes, 1, 0.6)
                add(Issue("warning", "unknown-layer-key",
                          f"{key!r} is not a {role} key of layer {head!r} (its {role} keys: "
                          f"{sorted(str(refs[head].fields[o]) for o in options)})"
                          + (f"; did you mean {head + '.' + hint[0]!r}?" if hint else "")
                          + ("" if not report.probed else "; nothing read it"), key))
                continue
            near = _near_layer(model, head) if dot else []
            if near:
                add(Issue("warning", "unknown-layer-prefix",
                          f"{key!r}: no layer {head!r}; did you mean {near[0] + '.' + tail!r}?",
                          key))
                continue
            if report.probed:
                add(Issue("warning", "unused-key",
                          f"{key!r} was given in the {role}s but nothing read it during the "
                          f"probe step (a typo, or read only through a copy the probe cannot "
                          f"see)", key))
    return report
