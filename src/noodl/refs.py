"""Typed references to a model's layers, keys that know their layout, and named inputs.

A `Model` is driven through two string-keyed dictionaries, `state` and `drivers`, whose
tensors are laid out in several different node orders: FULL node order for sources and
potentials, a layer's BOUNDARY order for prescribed values, its active INTERIOR order for
transport state, and EDGE order for flows. A tensor of the right size in the wrong order is
accepted without complaint. This module makes those layouts explicit without replacing the
dictionaries:

* `model.refs` maps each registered layer name to a `LayerRef`, whose attributes are the
  layer's state and input keys (`ref.x`, `ref.sources`, `ref.x_boundary`, ...). Arbitrary
  layer labels are kept: `model.refs["my layer"]` works for any name, attribute access for
  names that are Python identifiers.
* Each such attribute is a `Field`: a `str` subclass equal to (and hashing like) the plain
  key, so `drivers[ref.sources]` and `drivers["thermal.sources"]` are the same entry, which
  also carries the layout -- the labels of its indexed axis, in order, the species, the
  quantity and unit, and which labels may carry a value at all.
* `Field.build({"A": 1000.0})` assembles a tensor in the field's own order from a mapping of
  node (or edge) labels to values, refusing labels the network does not have and labels the
  field may not set (a source on a boundary node), and keeping gradients, batch dimensions,
  dtype and device of the values given. `Field.named(tensor)` reads one back by label.
* `state_from` / `drivers_from` assemble whole dictionaries from such mappings, passing
  plain tensors and keys this module knows nothing about (a custom closure's inputs)
  through unchanged.

Nothing here changes how a model computes: a tensor built by a `Field` is exactly the
tensor a caller would have built by hand in the same order.
"""

from __future__ import annotations

import difflib
import functools
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from noodl._broadcast import broadcast_shapes

if TYPE_CHECKING:
    from noodl.model import Model

Tensor = torch.Tensor
_REQUIRED = object()  # sentinel: every label must be given (no default fill)
_UNSET = object()


def _closest(label: Hashable, candidates: Sequence[Hashable]) -> list[str]:
    by_text = {str(c): c for c in candidates}
    return [repr(by_text[m]) for m in difflib.get_close_matches(str(label), list(by_text), 3, 0.6)]


class Field(str):
    """A state or driver key that also knows the layout of the tensor stored under it.

    Equal to, and hashing like, its plain key string, so it can index the ordinary
    `state`/`drivers` dictionaries directly. It pickles as that plain string.

    Attributes:
        role: `"state"` or `"driver"`.
        layer: The registered name of the layer the key belongs to (`None` for a
            closure-carried key).
        axis: `"node"` or `"edge"`: what the indexed (last, or second-to-last for a
            multi-species field) dimension runs over.
        labels: The node names (or edge labels) along that axis, in tensor order.
        ordering: A short description of that order (`"full node order"`, `"boundary
            order"`, `"active interior order"`, `"edge order of kinds (...)"`).
        settable: The labels that may carry a nonzero value; `None` means all of them. A
            source may not sit on a boundary or inactive node, for instance.
        n_species / species: The trailing species dimension (`n_species > 1` only) and its
            names when the layer knows them.
        quantity / unit: The layer's metadata, reported, never used numerically.
        required: `"always"`, `"step"` (needed to advance in time), or `"optional"`.
        default: The fill value for unlisted labels in `build`, or `None` when every label
            must be given explicitly.
        description: One line saying what the key holds.
    """

    def __new__(
        cls, key: str, *, role: str, layer: str | None, axis: str,
        labels: Sequence[Hashable], ordering: str, settable: Sequence[Hashable] | None = None,
        n_species: int = 1, species: Sequence[str] | None = None, quantity: str = "",
        unit: str = "", required: str = "optional", default: float | None = 0.0,
        description: str = "", dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu", aliases: Mapping[Hashable, int] | None = None,
        column: bool = False, scalar: bool = False,
    ) -> Field:
        self = super().__new__(cls, key)
        # One value per instance (a wind speed, a water temperature): no indexed axis, so
        # the whole tensor is batch shape. Built as a plain tensor, never by label.
        self.scalar = bool(scalar)
        # Whether the layer also accepts a single-species tensor as an `(n, 1)` column
        # (`TransportLayer._to_stacked`); `build` still returns the `(n,)` layout.
        self.column = bool(column) and int(n_species) == 1
        self.role = role
        self.layer = layer
        self.axis = axis
        self.labels = tuple(labels)
        self.ordering = ordering
        self.settable = None if settable is None else frozenset(settable)
        self.n_species = int(n_species)
        self.species = None if species is None else tuple(species)
        if self.species is not None and len(set(self.species)) != len(self.species):
            raise ValueError(f"Field {key!r}: species names {list(self.species)} repeat")
        self.quantity = quantity
        self.unit = unit
        self.required = required
        self.default = default
        self.description = description
        self.dtype = dtype
        self.device = torch.device(device)
        index: dict[Hashable, int] = {}
        for i, label in enumerate(self.labels):
            index.setdefault(label, i)
        for alias, i in (aliases or {}).items():
            index.setdefault(alias, i)
        self._index = index
        # The physical attribute name `LayerRef` exposes this key under, when it has one
        # (`"temperature"` for `"thermal.x"`); set by `LayerRef`.
        self.attribute: str | None = None
        return self

    # A Field is a key first: pickle and copy it as the plain string it is equal to, so
    # that a dictionary written with Field keys saves and loads without noodl.
    def __reduce__(self):
        return (str, (str(self),))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def key(self) -> str:
        """The plain key string."""
        return str(self)

    @property
    def trailing(self) -> tuple[int, ...]:
        """The trailing shape after any batch dimensions: `(n,)`, `(n, K)`, or `()` for a
        per-instance value."""
        if self.scalar:
            return ()
        n = len(self.labels)
        return (n,) if self.n_species == 1 else (n, self.n_species)

    def __repr__(self) -> str:
        return (
            f"Field({str(self)!r}, {self.role}, {self.ordering}, trailing shape "
            f"{self.trailing}{', ' + self.unit if self.unit else ''})"
        )

    def describe(self) -> str:
        """Several lines: the key, its order and labels, units and whether it is required."""
        shown = ", ".join(map(repr, self.labels[:8])) + (", ..." if len(self.labels) > 8 else "")
        name = f"{self.layer}.{self.attribute} = " if self.attribute and self.layer else ""
        lines = [
            f"{name}{str(self)!r} ({self.role}, {self.required}): {self.description}",
            f"  {self.ordering}" if self.scalar else
            f"  {self.ordering}, trailing shape {self.trailing}: [{shown}]",
        ]
        if self.species:
            lines.append(f"  species: {list(self.species)}")
        if self.quantity or self.unit:
            lines.append(f"  quantity: {self.quantity or '?'} [{self.unit or '?'}]")
        if self.settable is not None and len(self.settable) < len(self.labels):
            lines.append(
                f"  values may be set only on {len(self.settable)} of {len(self.labels)} "
                f"labels"
            )
        return "\n".join(lines)

    # ----------------------------------------------------------------- lookup
    def index(self, label: Hashable) -> int:
        """Position of `label` along this field's axis; `KeyError` with suggestions if unknown."""
        try:
            return self._index[label]
        except (KeyError, TypeError):
            pass
        hint = _closest(label, self.labels)
        raise KeyError(
            f"{str(self)!r}: unknown {self.axis} {label!r}"
            + (f"; did you mean {', '.join(hint)}?" if hint else "")
            + f" ({self.ordering} has {len(self.labels)} labels)"
        )

    def _species_index(self, name) -> int:
        if isinstance(name, int) and 0 <= name < self.n_species:
            return name
        if self.species is not None and name in self.species:
            return self.species.index(name)
        known = list(self.species) if self.species else list(range(self.n_species))
        raise KeyError(f"{str(self)!r}: unknown species {name!r}; this layer has {known}")

    # ------------------------------------------------------------------ build
    def build(
        self, values: Mapping[Hashable, Any], *, default: Any = _UNSET,
        base: Tensor | None = None, dtype: torch.dtype | None = None,
        device: torch.device | str | None = None, batch_shape: Sequence[int] = (),
    ) -> Tensor:
        """A tensor in this field's own order from `{label: value}`.

        Each value is a number, or a tensor whose shape is the batch shape (followed by the
        species dimension for a multi-species field), or -- for a multi-species field -- a
        mapping `{species name or index: value}`. Batch shapes broadcast against each other,
        against `batch_shape` and against `base`.

        Labels (and species) not given keep their value in `base`, a tensor already in this
        field's layout, when one is passed; otherwise they take `default` (the field's own
        default when omitted: zero for sources, flows and requests; none for boundary values
        and state, which must then be complete). The result is assembled with `torch.stack`,
        so a value that requires grad keeps its gradient path. Its dtype and device are the
        values' (and `base`'s) own unless `dtype`/`device` say otherwise, falling back to
        the network's.

        Raises `KeyError` for a label the field does not know and for missing labels when
        there is neither a base nor a default, and `ValueError` for a label that may not
        carry a value here.
        """
        if self.scalar:
            raise TypeError(
                f"{str(self)!r} is one value per instance ({self.description}); give it as a "
                f"number or a batch-shaped tensor, not a mapping of labels"
            )
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{str(self)!r}: build() takes a mapping {{label: value}}, got "
                f"{type(values).__name__}; pass a ready tensor straight into the dictionary"
            )
        if base is not None:
            self.check(base)
        fill = self.default if default is _UNSET else default
        positions: dict[int, Any] = {}
        for label, value in values.items():
            i = self.index(label)
            canonical = self.labels[i]
            if self.settable is not None and canonical not in self.settable:
                raise ValueError(
                    f"{str(self)!r}: {self.axis} {canonical!r} may not carry a value in this "
                    f"key ({self.description}); it is not one of the labels this layer "
                    f"solves for (a boundary or inactive {self.axis} of layer {self.layer!r})"
                )
            if i in positions:
                raise ValueError(f"{str(self)!r}: {self.axis} {canonical!r} given twice")
            positions[i] = value
        missing = [self.labels[i] for i in range(len(self.labels)) if i not in positions]
        if missing and fill is None and base is None:
            needed = [m for m in missing if self.settable is None or m in self.settable]
            if needed:
                raise KeyError(
                    f"{str(self)!r}: no value for {len(needed)} {self.axis}(s) "
                    f"{needed[:10]}{' ...' if len(needed) > 10 else ''}; give every "
                    f"label of the {self.ordering}, or pass default= or base="
                )
            fill = 0.0
        tensors = [v for v in positions.values() if isinstance(v, Tensor)]
        tensors += [
            v for m in positions.values() if isinstance(m, Mapping)
            for v in m.values() if isinstance(v, Tensor)
        ]
        if base is not None:
            tensors.append(base)
        floating = [t.dtype for t in tensors if t.is_floating_point()]
        if dtype is not None:
            out_dtype = dtype
        elif floating:
            out_dtype = functools.reduce(torch.promote_types, floating)
        else:
            out_dtype = self.dtype
        out_device = torch.device(device) if device is not None else (
            tensors[0].device if tensors else self.device
        )

        def as_tensor(v) -> Tensor:
            if isinstance(v, Tensor):
                return v.to(dtype=out_dtype, device=out_device)
            return torch.as_tensor(v, dtype=out_dtype, device=out_device)

        def from_base(i: int) -> Tensor:
            return as_tensor(self._column(base, i))

        entries: dict[int, Tensor] = {}
        for i, value in positions.items():
            label = self.labels[i]
            if self.n_species == 1:
                if isinstance(value, Mapping):
                    # `{species name: value}` names this layer's one species.
                    if len(value) != 1:
                        raise ValueError(
                            f"{str(self)!r}: {label!r} got {len(value)} species, but this "
                            f"layer carries one"
                            + (f" ({self.species[0]!r})" if self.species else "")
                        )
                    ((name, value),) = value.items()
                    self._species_index(name)
                entries[i] = as_tensor(value)
                continue
            if isinstance(value, Mapping):
                per: dict[int, Tensor] = {}
                for s, v in value.items():
                    k = self._species_index(s)
                    if k in per:
                        raise ValueError(
                            f"{str(self)!r}: {label!r} gives species {s!r} twice (by name "
                            f"and by index)"
                        )
                    per[k] = as_tensor(v)
                absent = [k for k in range(self.n_species) if k not in per]
                if absent and base is None and fill is None:
                    names = [self.species[k] if self.species else k for k in absent]
                    raise KeyError(f"{str(self)!r}: {label!r} has no value for species {names}")
                if absent and base is not None:
                    row = from_base(i)
                    per.update({k: row[..., k] for k in absent})
                shape = broadcast_shapes(*(t.shape for t in per.values()))
                zero = torch.full(shape, float(fill or 0.0), dtype=out_dtype, device=out_device)
                entries[i] = torch.stack(
                    [per[k].expand(shape) if k in per else zero for k in range(self.n_species)],
                    dim=-1,
                )
            else:
                t = as_tensor(value)
                if t.dim() == 0 or t.shape[-1] != self.n_species:
                    raise ValueError(
                        f"{str(self)!r}: {label!r} must be a mapping by species or a tensor "
                        f"whose last dimension is the {self.n_species} species "
                        f"{list(self.species) if self.species else ''}; got shape "
                        f"{tuple(t.shape)}"
                    )
                entries[i] = t
        if base is not None:
            for i in range(len(self.labels)):
                if i not in entries:
                    entries[i] = from_base(i)
        tail = () if self.n_species == 1 else (self.n_species,)
        batch = tuple(broadcast_shapes(
            tuple(batch_shape),
            *(t.shape if self.n_species == 1 else t.shape[:-1] for t in entries.values()),
        ))
        fillers: dict[float, Tensor] = {}
        cols = []
        for i in range(len(self.labels)):
            if i in entries:
                cols.append(entries[i].expand(batch + tail))
                continue
            # A label that may not carry a value (a boundary or inactive node) is zero
            # whatever `default` says: the layer refuses anything else there.
            value = 0.0 if (
                self.settable is not None and self.labels[i] not in self.settable
            ) else float(fill)
            if value not in fillers:
                fillers[value] = torch.full(batch + tail, value, dtype=out_dtype,
                                            device=out_device)
            cols.append(fillers[value])
        if not cols:
            return torch.zeros(batch + self.trailing, dtype=out_dtype, device=out_device)
        return torch.stack(cols, dim=-1 if self.n_species == 1 else -2)

    def trailing_dims(self, tensor: Tensor) -> int:
        """How many trailing dimensions of `tensor` are this field's layout; the rest are
        batch. 1 for `(n,)`, 2 for `(n, K)`, and 2 for a single-species `(n, 1)` column
        where the layer accepts one -- tested first, in the order
        `TransportLayer._to_stacked` tests them; 0 for a per-instance value, whose whole
        shape is batch. Raises `ValueError` otherwise."""
        if self.scalar and isinstance(tensor, Tensor):
            return 0
        n = len(self.labels)
        if isinstance(tensor, Tensor):
            shape = tuple(tensor.shape)
            if (self.n_species > 1 or self.column) and shape[-2:] == (n, self.n_species):
                return 2
            if self.n_species == 1 and len(shape) >= 1 and shape[-1] == n:
                return 1
        got = tuple(tensor.shape) if isinstance(tensor, Tensor) else type(tensor).__name__
        extra = " or (n, 1)" if self.column else ""
        raise ValueError(
            f"{str(self)!r} must have trailing shape {self.trailing}{extra} "
            f"({self.ordering}), got {got}"
        )

    def check(self, tensor: Tensor) -> None:
        """Raise `ValueError` unless `tensor`'s trailing shape is this field's."""
        self.trailing_dims(tensor)

    def _column(self, tensor: Tensor, i: int) -> Tensor:
        """Label `i`'s slice of `tensor`: batch shape, plus the species for `K > 1`."""
        if self.trailing_dims(tensor) == 2:
            return tensor[..., i, :] if self.n_species > 1 else tensor[..., i, 0]
        return tensor[..., i]

    def named(self, tensor: Tensor) -> dict[Hashable, Tensor]:
        """`{label: tensor[..., i]}` (or `[..., i, :]` for several species), in field order."""
        if self.scalar:
            raise TypeError(f"{str(self)!r} is one value per instance; it has no labels")
        self.check(tensor)
        return {label: self._column(tensor, i) for i, label in enumerate(self.labels)}


def input_field(
    model: Model, key: str, *, description: str, unit: str = "", over=None,
    required: bool = True, ordering: str | None = None,
) -> Field:
    """A `Field` for an input a closure, reaction, element or drive reads itself (a wind
    speed, an inflow), as a closure's `input_specs` or a builder's `model.input_specs`
    declares it.

    `over` is the layout: `None` for one value per instance; `"nodes"` for one per node
    in full node order; `("edges", kind)` for one per edge of `kind`; or a sequence of
    labels, in tensor order, for anything else (streets, junctions, manholes). `required`
    says whether a run needs it.
    """
    net = model.net
    common = dict(role="driver", layer=None, description=description, unit=unit,
                  required="always" if required else "optional", default=None,
                  dtype=net.dtype, device=net.device)
    if over is None:
        return Field(key, axis="instance", labels=(), scalar=True,
                     ordering=ordering or "one value per instance", **common)
    if over == "nodes":
        return Field(key, axis="node", labels=net.nodes,
                     ordering=ordering or "full node order", **common)
    if isinstance(over, tuple) and len(over) == 2 and over[0] == "edges":
        labels, aliases = _edge_labels(net, net.edge_index(over[1]).tolist())
        return Field(key, axis="edge", labels=labels, aliases=aliases,
                     ordering=ordering or f"edge order of kind {over[1]!r}", **common)
    labels = list(over)
    return Field(key, axis="label", labels=labels,
                 ordering=ordering or f"order of its {len(labels)} labels", **common)


def _declared_inputs(model: Model) -> dict[str, Field]:
    """The inputs `model.input_specs` and its closures' and reactions' `input_specs`
    declare, as Fields. A spec is `{key: {"description", "unit", "over", "required",
    "ordering"}}` (all but `description` optional), or `{key: Field}` already built."""
    sources = [getattr(model, "input_specs", None) or {}]
    sources += [getattr(c, "input_specs", None) or {} for c in model.closures]
    sources += [getattr(r, "input_specs", None) or {} for _, r in model.reactions]
    out: dict[str, Field] = {}
    for specs in sources:
        for key, spec in dict(specs).items():
            out[str(key)] = spec if isinstance(spec, Field) else input_field(
                model, str(key), **dict(spec))
    return out


def _edge_labels(net, cols: Sequence[int]) -> tuple[list[Hashable], dict[Hashable, int]]:
    """Labels for the given edge columns: the edge's `name` attribute when it has one,
    otherwise `(source, target)`, or `(source, target, key)` where that is ambiguous. Every
    edge is ALSO reachable through `(source, target, key)` and, when unambiguous, through
    `(source, target)`."""
    edges = net.edges
    triples = [edges[c] for c in cols]
    pairs: dict[tuple, int] = {}
    for u, v, _ in triples:
        pairs[(u, v)] = pairs.get((u, v), 0) + 1
    names = [net.graph.edges[e].get("name") for e in triples]
    name_count: dict[Hashable, int] = {}
    for n in names:
        if n is not None:
            name_count[n] = name_count.get(n, 0) + 1
    labels: list[Hashable] = []
    aliases: dict[Hashable, int] = {}
    for i, ((u, v, k), name) in enumerate(zip(triples, names, strict=True)):
        if name is not None and name_count[name] == 1:
            labels.append(name)
        elif pairs[(u, v)] == 1:
            labels.append((u, v))
        else:
            labels.append((u, v, k))
        aliases[(u, v, k)] = i
        if pairs[(u, v)] == 1:
            aliases[(u, v)] = i
    return labels, aliases


class LayerRef:
    """One registered layer of a model, with its keys as discoverable `Field` attributes.

    `kind` is `"potential"`, `"transport"` or `"allocation"`. `state` and `inputs` map each
    key's short name (`"x"`, `"sources"`, ...) to its `Field`; the same fields are attributes
    (`ref.x`, `ref.sources`). A transport layer whose flows no potential layer provides has
    the driver `ref.flows` (key `"<layer>.q"`), in the order of its `flow_kinds`.
    """

    def __init__(self, name: str, layer, kind: str, *, flow_owner: str | None = None) -> None:
        self.name = name
        self.layer = layer
        self.kind = kind
        self.quantity = getattr(layer, "quantity", "")
        self.unit = getattr(layer, "unit", "")
        net = layer.net
        self.net = net
        nodes = net.nodes
        common = dict(dtype=net.dtype, device=net.device, layer=name)
        state: dict[str, Field] = {}
        inputs: dict[str, Field] = {}
        if kind == "potential":
            self.interior_nodes = [nodes[i] for i in layer.interior.tolist()]
            self.boundary_nodes = [nodes[i] for i in layer.bound.tolist()]
            self.inactive_nodes = [nodes[i] for i in layer.inactive.tolist()]
            q_labels, q_alias = _edge_labels(net, layer.cols.tolist())
            meta = dict(quantity=self.quantity, unit=self.unit)
            state["phi"] = Field(
                f"{name}.phi", role="state", axis="node", labels=nodes,
                ordering="full node order", required="optional", default=None,
                description="solved potential (a warm start when given)", **meta, **common,
            )
            state["q"] = Field(
                f"{name}.q", role="state", axis="edge", labels=q_labels, aliases=q_alias,
                ordering=f"edge order of kinds {tuple(layer.kinds)}", required="optional",
                default=0.0, description="solved branch flows", **common,
            )
            inputs["phi_boundary"] = Field(
                f"{name}.phi_boundary", role="driver", axis="node", labels=self.boundary_nodes,
                ordering="boundary order", required="always", default=None,
                description="prescribed potential at the boundary nodes", **meta, **common,
            )
            inputs["sources"] = Field(
                f"{name}.sources", role="driver", axis="node", labels=nodes,
                settable=self.interior_nodes, ordering="full node order",
                required="optional", default=0.0,
                description="nodal injection (zero on boundary and inactive nodes)", **common,
            )
        elif kind == "transport":
            K = int(layer.n_species)
            species = getattr(layer, "species_names", None)
            self.interior_nodes = [nodes[i] for i in layer.interior_idx.tolist()]
            self.boundary_nodes = list(layer.boundary)
            self.inactive_nodes = [nodes[i] for i in layer.inactive_idx.tolist()]
            meta = dict(quantity=self.quantity, unit=self.unit, n_species=K, species=species)
            state["x"] = Field(
                f"{name}.x", role="state", axis="node", labels=self.interior_nodes,
                ordering="active interior order", required="step", default=None,
                description="transported state at the active interior nodes", column=True,
                **meta, **common,
            )
            inputs["x_boundary"] = Field(
                f"{name}.x_boundary", role="driver", axis="node", labels=self.boundary_nodes,
                ordering="boundary order", required="always", default=None,
                description="prescribed value at the boundary nodes", column=True,
                **meta, **common,
            )
            inputs["sources"] = Field(
                f"{name}.sources", role="driver", axis="node", labels=nodes,
                settable=self.interior_nodes, ordering="full node order",
                required="optional", default=0.0, n_species=K, species=species,
                description="nodal source (zero on boundary and inactive nodes)", column=True,
                **common,
            )
            inputs["capacity"] = Field(
                f"{name}.capacity", role="driver", axis="node", labels=self.interior_nodes,
                ordering="active interior order", required="optional", default=None,
                description="per-step storage capacity overriding the construction-time one",
                **common,
            )
            if flow_owner is None:
                cols: list[int] = []
                for k in layer.flow_kinds:
                    cols += net.edge_index(k).tolist()
                f_labels, f_alias = _edge_labels(net, cols)
                inputs["flows"] = Field(
                    f"{name}.q", role="driver", axis="edge", labels=f_labels, aliases=f_alias,
                    ordering=f"edge order of kinds {tuple(layer.flow_kinds)}",
                    required="always", default=0.0,
                    description="prescribed branch flows (no potential layer provides them)",
                    **common,
                )
        elif kind == "allocation":
            self.interior_nodes = list(nodes)
            self.boundary_nodes = []
            self.inactive_nodes = []
            e_labels, e_alias = _edge_labels(net, net.edge_index(layer.kind).tolist())
            state["s"] = Field(
                f"{name}.s", role="state", axis="node", labels=nodes,
                ordering="full node order", required="step", default=None,
                description="storage at every node", **common,
            )
            state["q"] = Field(
                f"{name}.q", role="state", axis="edge", labels=e_labels, aliases=e_alias,
                ordering=f"edge order of kind {layer.kind!r}", required="optional",
                default=0.0, description="realised transfers (written by the step)", **common,
            )
            inputs["requests"] = Field(
                f"{name}.requests", role="driver", axis="edge", labels=e_labels,
                aliases=e_alias, ordering=f"edge order of kind {layer.kind!r}",
                required="step", default=0.0, description="requested transfer per edge",
                **common,
            )
        else:
            raise ValueError(f"LayerRef: unknown layer kind {kind!r}")
        self.state = state
        self.inputs = inputs
        self.flow_owner = flow_owner
        # PHYSICAL names. The key suffixes ("x", "phi", "s", "q") are the solver's; the
        # attributes a user writes should name the physics: `thermal.temperature`,
        # `species.mass_fraction`, `water.head`, `wsimod.storage`. They come from the
        # layer's own `quantity` tag, so every application gets them without code of its
        # own; the short suffix names stay as aliases.
        q = self.quantity if self.quantity not in ("", "potential", "scalar") else None
        physical: dict[str, str] = {}
        if kind == "potential":
            if q:
                physical.update({q: "phi", f"boundary_{q}": "phi_boundary"})
            physical["flow"] = "q"
        elif kind == "transport":
            if q:
                physical.update({q: "x", f"boundary_{q}": "x_boundary"})
            if "flows" in inputs:
                physical["flow"] = "flows"
        else:
            physical.update({"storage": "s", "flow": "q"})
        fields = {**state, **inputs}
        self.physical = {p: s for p, s in physical.items() if p not in fields}
        for p, s in self.physical.items():
            fields[s].attribute = p

    @property
    def fields(self) -> dict[str, Field]:
        """Every key of this layer, state then inputs, by short (suffix) name."""
        return {**self.state, **self.inputs}

    def _names(self) -> dict[str, Field]:
        fields = {**self.__dict__.get("state", {}), **self.__dict__.get("inputs", {})}
        named = {p: fields[s] for p, s in self.__dict__.get("physical", {}).items()}
        return {**named, **fields}

    def __getattr__(self, attr: str) -> Field:
        if attr.startswith("_") or attr in ("state", "inputs", "physical"):
            raise AttributeError(attr)
        names = self._names()
        if attr in names:
            return names[attr]
        hint = difflib.get_close_matches(attr, list(names), 1, 0.6)
        raise AttributeError(
            f"{self.kind} layer {self.__dict__.get('name')!r} has no key {attr!r}"
            + (f"; did you mean {hint[0]!r}?" if hint else "")
            + f" (its keys: {sorted(names)})"
        )

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._names()))

    def __repr__(self) -> str:
        unit = f" [{self.unit}]" if self.unit else ""
        return (
            f"LayerRef({self.name!r}, {self.kind}, {self.quantity or '?'}{unit}; "
            f"keys {sorted(self._names())})"
        )

    def describe(self) -> str:
        """The layer and every one of its keys, one block per key."""
        head = repr(self)
        return "\n".join([head] + [f.describe() for f in self.fields.values()])


class LayerRefs(Mapping[str, LayerRef]):
    """Registered layer name -> `LayerRef`, plus the model's closure-carried keys.

    Item access works for every name; attribute access for names that are identifiers.
    `closure_state` maps each closure-carried state key to a `Field` when its closure
    declares the labels (a `key_labels` attribute, `{key: labels}`), else to the plain key.
    `inputs` maps each input a closure, reaction, element or drive reads itself -- a wind
    speed, an inflow, a photolysis rate -- to its `Field`, as the closure's or reaction's
    `input_specs`, or the builder's `model.input_specs`, declares it (`input_field`).
    """

    def __init__(self, model: Model) -> None:
        refs: dict[str, LayerRef] = {}
        for name, layer in model.potential.items():
            refs[name] = LayerRef(name, layer, "potential")
        for name, layer in model.transport.items():
            refs[name] = LayerRef(name, layer, "transport", flow_owner=model.flow_layer_of[name])
        for name, layer in model.allocation.items():
            refs[name] = LayerRef(name, layer, "allocation")
        # Keep the model's own registration order.
        self._refs = {name: refs[name] for name in model.layers}
        net = model.net
        closure_state: dict[str, str] = {}
        for key, closure in model.closure_state_keys.items():
            labels = dict(getattr(closure, "key_labels", {}) or {}).get(key)
            if labels is None:
                closure_state[key] = key
            else:
                closure_state[key] = Field(
                    key, role="state", layer=None, axis="label", labels=labels,
                    ordering=f"order of {type(closure).__name__}", required="step",
                    default=None, description=f"state carried by closure {type(closure).__name__}",
                    dtype=net.dtype, device=net.device,
                )
        self.closure_state = closure_state
        self.inputs = _declared_inputs(model)

    def __getitem__(self, name: str) -> LayerRef:
        try:
            return self._refs[name]
        except KeyError:
            hint = _closest(name, list(self._refs))
            raise KeyError(
                f"no layer {name!r}"
                + (f"; did you mean {', '.join(hint)}?" if hint else "")
                + f" (layers: {list(self._refs)})"
            ) from None

    def __getattr__(self, name: str) -> LayerRef:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | {n for n in self._refs if n.isidentifier()})

    def __iter__(self) -> Iterator[str]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __repr__(self) -> str:
        return "LayerRefs(" + ", ".join(repr(r) for r in self._refs.values()) + ")"

    def fields(self) -> dict[str, Field | str]:
        """Every key this model knows the layout of: layer keys, closure-carried keys and
        the declared inputs of its closures, reactions, elements and drives."""
        out: dict[str, Field | str] = {}
        for ref in self._refs.values():
            for f in ref.fields.values():
                out[str(f)] = f
        out.update(self.closure_state)
        for key, f in self.inputs.items():
            out.setdefault(key, f)
        return out

    def field(self, key: str) -> Field | None:
        """The `Field` for `key`, or `None` when the model does not know its layout."""
        f = self.fields().get(str(key))
        return f if isinstance(f, Field) else None

    def describe(self) -> str:
        """Every layer and key: what to give, in which order, in which unit."""
        parts = [r.describe() for r in self._refs.values()]
        for key, f in self.closure_state.items():
            parts.append(f.describe() if isinstance(f, Field) else f"{key!r} (closure state)")
        if self.inputs:
            parts.append("Inputs read by the closures, reactions, elements and drives:\n"
                         + "\n".join(f.describe() for f in self.inputs.values()))
        return "\n\n".join(parts)


def _assemble(model: Model, values: Mapping, base: Mapping | None, role: str) -> dict[str, Tensor]:
    refs = model.refs
    out: dict[str, Tensor] = dict(base or {})
    for key, value in values.items():
        k = str(key)
        f = refs.field(k)
        if f is not None and f.role != role and not k.endswith(".capacity"):
            raise KeyError(
                f"{k!r} is a {f.role} key of layer {f.layer!r}, not a {role} key; put it in "
                f"the {f.role} dictionary"
            )
        if isinstance(value, Mapping):
            if f is None:
                raise KeyError(
                    f"{k!r}: a mapping of labels was given, but the model does not know this "
                    f"key's layout; pass a tensor (a custom closure input), or check the "
                    f"spelling against model.refs.describe()"
                )
            out[k] = f.build(value, base=out.get(k))
        else:
            t = value if isinstance(value, Tensor) else torch.as_tensor(
                value, dtype=model.net.dtype, device=model.net.device
            )
            if f is not None:
                f.check(t)
            out[k] = t
    return out


def drivers_from(
    model: Model, values: Mapping, *, base: Mapping | None = None
) -> dict[str, Tensor]:
    """A drivers dictionary from `{key: {label: value}}` and/or `{key: tensor}`.

    Keys may be `Field`s or plain strings. A mapping value is assembled in the key's own
    order (`Field.build`); a tensor is shape-checked against the key's layout when the
    model knows it and otherwise passed through unchanged (a custom closure's input). The
    result starts from `base` and has plain string keys.
    """
    return _assemble(model, values, base, "driver")


def state_from(
    model: Model, values: Mapping, *, base: Mapping | None = None
) -> dict[str, Tensor]:
    """A state dictionary, as `drivers_from` builds drivers.

    A layer name may stand for that layer's own state key: `{"thermal": {"A": 293.15}}`
    means `{"thermal.x": ...}` for a transport layer and `{"<name>.s": ...}` for a
    allocation one. Raises `KeyError` if, after assembly, a key a time step needs (every
    transport `"<layer>.x"`, allocation `"<layer>.s"` and closure-carried state key) is
    still missing, naming it and its order.
    """
    expanded: dict = {}
    for key, value in values.items():
        k = str(key)
        if k in model.transport:
            k = f"{k}.x"
        elif k in model.allocation:
            k = f"{k}.s"
        expanded[k] = value
    state = _assemble(model, expanded, base, "state")
    missing = [k for k in required_state_keys(model) if k not in state]
    if missing:
        fields = model.refs.fields()
        detail = "; ".join(
            f"{k!r} ({fields[k].ordering if isinstance(fields.get(k), Field) else 'closure state'})"
            for k in missing
        )
        raise KeyError(f"state_from: no initial value for {detail}")
    return state


def initial_drivers(model: Model, *, values: Mapping | None = None) -> dict[str, Tensor]:
    """A copy of the model's driver template (`model.driver_template`, which an
    application's `build_model` records), then `values` (see `drivers_from`)."""
    template = getattr(model, "driver_template", None) or {}
    base = {k: v.clone() if isinstance(v, Tensor) else v for k, v in template.items()}
    return drivers_from(model, values or {}, base=base)


def required_state_keys(model: Model) -> list[str]:
    """The state keys `Model.step` reads from the step-start state."""
    keys = [f"{n}.x" for n in model.transport] + [f"{n}.s" for n in model.allocation]
    return keys + list(model.closure_state_keys)
