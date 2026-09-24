"""Component graph over a `ModelicaDoc` (spec section 6, conversion rules).

`build(doc)` resolves every `connect()` pair the JSON carries (`doc.connections`) to a port on
one of `doc.components`, groups ports into NODES with union-find, fuses hydrostatic-column
chains into single edges, checks two-way wiring, and returns a `ComponentGraph`. It never
evaluates a Modelica expression: every parameter it reads was already evaluated by
OpenModelica, and every check here is about topology and class support, not physics. Task 6
turns the result into a noodl `Model`.

Every phase below runs to completion and only ADDS to a shared error list; nothing raises
until `build` has run every phase, so one `ModelicaImportError` names everything wrong with
the document, not just the first thing found (fix round 1, review item 2). A phase that
depends on a node a previous phase could not resolve (the "conflict" node kind below) skips
just that one piece of work rather than crashing.

Node identity
-------------
A `MixingVolume`/`DelayFirstOrder` ("zone") or `Boundary_pT`/`Outside` ("boundary") component
owns exactly one node, named after the component: every one of its own fluid ports (all of
its declared `ports[i]`) is the SAME node, because a lumped volume or boundary is a single
pressure state that happens to expose several connection points. A `TraceSubstancesFlowSource`
or `MassFlowSource_T` ("source") likewise shares whichever node its own port(s) resolve to --
it is not a node in its own right, it decorates one (spec section 6, "becomes a node source").

Every other `connect()` pair between two ports that belong to NEITHER a zone/boundary/source
component is a plain wire: it creates a JUNCTION node, named `"_j<k>"`. Junctions are numbered
in a deterministic order: sorted by the lexicographically smallest port reference string
(`"<instance>.<port>"`) in the group (Task 5 resolution). A junction with more than two
connections -- i.e. a group of more than two port references, meaning three or more distinct
components meet at that wire rather than two -- is refused; so is a group holding ports from
two DIFFERENT zone/boundary components (nothing here wires two zones directly with no flow
element between them). Either case marks the node "conflict" rather than dropping it, so
later phases that touch it can skip cleanly instead of raising a `KeyError`; a conflict node
never appears in the `ComponentGraph.nodes` a caller sees.

Hydrostatic column chains and path orientation
-----------------------------------------------
A one-way flow element (`Orifice` and friends) together with any number of `MediumColumn`s
and degree-two junctions between it and each of its two end zones/boundaries fuses into one
`FlowPath` (spec section 6). The path is ORIENTED from the element's `port_a` side to its
`port_b` side: `FlowPath.src` (the element's "tail") is the zone/boundary reached by walking
from the element's `port_a`, `FlowPath.tgt` (its "head") the one reached from `port_b`.

Each column on the path gets a sign: **+1** if, walking the path from its `src` end to its
`tgt` end, the column's OWN `port_a` (its top -- confirmed against
`Buildings/Airflow/Multizone/MediumColumn.mo`: `port_a` is the "positive design flow
direction" inlet, drawn at the top of the icon, `port_b` at the bottom) is reached before its
`port_b` (its bottom); **-1** otherwise. `FlowPath.columns` lists them `(component, sign)` in
that `src`-to-`tgt` order, so Task 6 computes the flow element's own driving pressure
difference as

    dp_element = (phi_src - phi_tgt) + sum(sign * h * rho * g)

`rho` chosen per each column's own `densitySelection` parameter (fix round 1, review item 1 --
an earlier draft had this negated). Derivation, checked against the ThreeRoomsContam west
stack this module's tests fix as `stack_chain.json` (`MediumColumn.mo`'s own relation is
`port_a.p - port_b.p = -h*rho*g_n`, i.e. the bottom port is at the higher pressure):
`oriWesTop.port_a` is wired straight through `colWesTop` to `volTop` (`src`), so
`p(colWesTop.port_a) = p(volTop)` and hence `p(oriWesTop.port_a) = p(volTop) + h*rho*g`;
symmetrically `oriWesTop.port_b` is wired through `colWesBot` to `volWes` (`tgt`), giving
`p(oriWesTop.port_b) = p(volWes) - h*rho*g`. Both columns get sign +1 (`src`'s side reaches
each column's own `port_a` first), and

    dp_oriWesTop = p(volTop) - p(volWes) + 2*h*rho*g = (phi_src - phi_tgt) + sum(sign*h*rho*g)

matching the formula above; at zero flow (`dp_oriWesTop = 0`) this is the ordinary hydrostatic
balance `p(volWes) - p(volTop) = 2*h*rho*g` (`test_graph.py`'s `test_stack_chain_...` checks
this exact identity, and a second fixture/test checks a column wired the other way along its
path, which gets sign -1). A chain with zero flow elements (a run of columns alone between two
zones/boundaries) or more than one is refused, naming every element found in it.

Two-way elements
----------------
A door (`DoorOpen`, `DoorOperable`, `DoorDiscretizedOpen`, `DoorDiscretizedOperable`) and a
zonal-flow element (`ZonalFlow_ACS`, `ZonalFlow_m_flow` -- confirmed to share the door's
four-port `Fluid.Interfaces.PartialFourPortInterface` shape against
`Buildings/Airflow/Multizone/BaseClasses/ZonalFlow.mo` and `ZonalFlow_ACS.mo`, NOT the
two-port `port_a`/`port_b` shape) each wire directly between two zone/boundary nodes with no
junction or column in between: side A holds `port_a1` and `port_b2`, side B holds `port_b1`
and `port_a2` (`Validation/ThreeRoomsContam.mo:149-179` wires `dooOpeClo` this way). Any other
pairing -- `port_a1` and `port_b2` resolving to different nodes, or the two sides coinciding --
is refused, naming the instance.

Heat ports
----------
Fluid ports and heat ports never share a node: a `FixedTemperature.port` -> a
`ThermalConductor.port_a`/`port_b` -> a zone's `heatPort` is a separate two-hop chain, recorded
as a `Pin` for Task 6 (which reads the conductor's `G` and decides whether it is large enough
to pin the zone's temperature -- spec section 6's threshold is not applied here). Any other
heat-port wiring -- a bare `FixedTemperature` with no conductor, a conductor whose other end is
not a zone, and so on -- is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from noodl.apps.building_physics.modelica import schema
from noodl.apps.building_physics.modelica.schema import (
    Component,
    ModelicaDoc,
    ModelicaImportError,
)

_FIXED_TEMPERATURE = "Buildings.HeatTransfer.Sources.FixedTemperature"
_THERMAL_CONDUCTOR = "Modelica.Thermal.HeatTransfer.Components.ThermalConductor"

_ARRAY_PORT = re.compile(r"^ports\[\d+\]$")
_TWO_PORT = ("port_a", "port_b")
_FOUR_PORT = ("port_a1", "port_b1", "port_a2", "port_b2")


@dataclass(frozen=True)
class FlowPath:
    """One fused hydrostatic-column chain: exactly one flow element, oriented `src` (its
    `port_a` side, the element's "tail") to `tgt` (its `port_b` side, its "head"), with the
    columns encountered along the way. See the module docstring for the head formula."""

    element: Component
    src: str
    tgt: str
    columns: tuple[tuple[Component, int], ...]


@dataclass(frozen=True)
class TwoWayEdge:
    """A door or zonal-flow element wired directly between two zone/boundary nodes."""

    component: Component
    side_a: str
    side_b: str


@dataclass(frozen=True)
class Pin:
    """A `FixedTemperature` pinning `zone`'s temperature through `conductor`."""

    source: Component
    conductor: Component
    zone: str


@dataclass(frozen=True)
class ComponentGraph:
    nodes: dict[str, str]  # node name -> "zone" | "boundary" | "junction"
    zones: tuple[str, ...]
    boundaries: tuple[str, ...]
    paths: tuple[FlowPath, ...]
    doors: tuple[TwoWayEdge, ...]
    zonal: tuple[TwoWayEdge, ...]
    pins: tuple[Pin, ...]
    sources: tuple[tuple[Component, str], ...]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def __contains__(self, x: str) -> bool:
        return x in self._parent

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), set()).add(x)
        return out


class _ChainRefused(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _split(ref: str) -> tuple[str, str]:
    instance, sep, port = ref.partition(".")
    if not sep:
        raise ModelicaImportError(f"modelica: {ref!r} is not an '<instance>.<port>' reference")
    return instance, port


def _kind_of(cls: str, role: str | None) -> str:
    if role == "observer":
        return "observer"
    if cls in schema.ZONES:
        return "zone"
    if cls in schema.BOUNDARIES:
        return "boundary"
    if cls in schema.ONE_WAY:
        return "one_way"
    if cls in schema.TWO_WAY:
        return "door"
    if cls in schema.COLUMNS:
        return "column"
    if cls in schema.ZONAL:
        return "zonal"
    if cls in schema.THERMAL_PIN:
        return "pin"
    if cls in schema.SOURCES:
        return "source"
    if cls in schema.SIGNALS:
        return "signal"
    if cls in schema.OBSERVERS:
        return "observer"
    return "unknown"


def _is_fluid_port(kind: str, port: str) -> bool:
    if kind in ("zone", "boundary", "source"):
        return bool(_ARRAY_PORT.match(port))
    if kind in ("one_way", "column"):
        return port in _TWO_PORT
    if kind in ("door", "zonal"):
        return port in _FOUR_PORT
    return False


def _is_heat_port(kind: str, cls: str, port: str) -> bool:
    if kind == "zone":
        return port == "heatPort"
    if kind == "pin":
        if cls == _FIXED_TEMPERATURE:
            return port == "port"
        return port in _TWO_PORT  # ThermalConductor
    return False


def _raise(errors: list[str]) -> None:
    if not errors:
        return
    unique = sorted(set(errors))
    count = "1 item" if len(unique) == 1 else f"{len(unique)} items"
    raise ModelicaImportError(
        f"modelica: refused {count}:\n" + "\n".join(f"  - {e}" for e in unique)
    )


@dataclass
class _Ctx:
    """Shared, mutable state threaded through every phase of `build`.

    `errors` accumulates across every phase (fix round 1, review item 2): nothing raises until
    `build` has run all of them, so one `ModelicaImportError` names everything wrong with the
    document. `root_to_kind` can hold `"conflict"` for a node a phase could not resolve (two
    zone/boundary components wired directly together); later phases check for that and skip
    just the dependent piece of work instead of raising a `KeyError`.
    """

    by_name: dict[str, Component]
    kind_of: dict[str, str]
    errors: list[str]
    uf: _UnionFind = field(default_factory=_UnionFind)
    fluid_refs_by_instance: dict[str, list[str]] = field(default_factory=dict)
    members_of_root: dict[str, set[str]] = field(default_factory=dict)
    root_to_name: dict[str, str] = field(default_factory=dict)
    root_to_kind: dict[str, str] = field(default_factory=dict)

    def node_of(self, ref: str) -> str:
        return self.root_to_name[self.uf.find(ref)]

    def kind_of_ref(self, ref: str) -> str:
        return self.root_to_kind[self.uf.find(ref)]


def _classify_components(doc: ModelicaDoc) -> _Ctx:
    by_name: dict[str, Component] = {}
    kind_of: dict[str, str] = {}
    errors: list[str] = []
    for c in doc.components:
        by_name[c.name] = c
        kind = _kind_of(c.cls, c.role)
        kind_of[c.name] = kind
        if kind == "unknown":
            reason = schema.refusal_reason(c.cls) or "component class is not supported"
            errors.append(f"{c.name} ({c.cls}): {reason}")

    for s in doc.signals:
        if s.cls not in schema.SIGNALS:
            reason = schema.refusal_reason(s.cls) or "signal source is not supported"
            errors.append(f"{s.name} ({s.cls}): {reason}")

    return _Ctx(by_name=by_name, kind_of=kind_of, errors=errors)


def _process_fluid_connections(doc: ModelicaDoc, ctx: _Ctx) -> None:
    """Union every fluid `connect()` pair; note a signal-style connection's refusal, if any."""
    for a, b in doc.connections:
        ia, pa = _split(a)
        ib, pb = _split(b)
        if ia not in ctx.by_name:
            ctx.errors.append(f"{a}: connection references unknown instance {ia!r}")
            continue
        if ib not in ctx.by_name:
            ctx.errors.append(f"{b}: connection references unknown instance {ib!r}")
            continue
        ka, kb = ctx.kind_of[ia], ctx.kind_of[ib]
        a_fluid, b_fluid = _is_fluid_port(ka, pa), _is_fluid_port(kb, pb)
        a_heat = _is_heat_port(ka, ctx.by_name[ia].cls, pa)
        b_heat = _is_heat_port(kb, ctx.by_name[ib].cls, pb)
        if a_fluid and b_fluid:
            ctx.uf.union(a, b)
            ctx.fluid_refs_by_instance.setdefault(ia, []).append(a)
            ctx.fluid_refs_by_instance.setdefault(ib, []).append(b)
        elif a_heat and b_heat:
            pass  # resolved separately by `_resolve_pins`, re-reading `doc.connections`
        else:
            # Neither a fluid nor a heat connection: a signal-style wire between blocks
            # (spec section 6 / Task 5 resolution). Only refuse it if it feeds a component
            # this reader does not otherwise support; a legitimate signal link is not
            # recorded anywhere in `ComponentGraph` -- Task 6 reads `doc.signals` and
            # `doc.connections` directly for that.
            for inst in (ia, ib):
                reason = schema.refusal_reason(ctx.by_name[inst].cls)
                if reason is not None:
                    ctx.errors.append(f"{inst} ({ctx.by_name[inst].cls}): {reason}")

    # Self-union: every port of the SAME zone/boundary/source instance is the same node.
    for inst, refs in ctx.fluid_refs_by_instance.items():
        if ctx.kind_of[inst] in ("zone", "boundary", "source"):
            first = refs[0]
            for r in refs[1:]:
                ctx.uf.union(first, r)


def _assign_nodes(ctx: _Ctx) -> None:
    """Name every union-find group: a zone/boundary component's own name, or `"_j<k>"`.

    A group holding ports from two DIFFERENT zone/boundary components is marked `"conflict"`
    rather than skipped, so later phases that touch it can decline that one piece of work
    instead of hitting a missing dict entry.
    """
    ctx.members_of_root = ctx.uf.groups()
    junction_groups: list[tuple[str, set[str]]] = []
    for root, members in ctx.members_of_root.items():
        zb_names = sorted({
            ctx.by_name[_split(m)[0]].name for m in members
            if ctx.kind_of[_split(m)[0]] in ("zone", "boundary")
        })
        if len(zb_names) > 1:
            ctx.errors.append(
                f"{' and '.join(zb_names)}: connected directly to each other with no flow "
                f"element between them"
            )
            ctx.root_to_name[root] = f"_conflict({','.join(zb_names)})"
            ctx.root_to_kind[root] = "conflict"
        elif zb_names:
            ctx.root_to_name[root] = zb_names[0]
            ctx.root_to_kind[root] = ctx.kind_of[zb_names[0]]
        else:
            junction_groups.append((root, members))

    junction_groups.sort(key=lambda rm: min(rm[1]))
    for i, (root, _members) in enumerate(junction_groups):
        ctx.root_to_name[root] = f"_j{i}"
        ctx.root_to_kind[root] = "junction"


def _two_way_edges(ctx: _Ctx, want_kind: str) -> list[TwoWayEdge]:
    edges: list[TwoWayEdge] = []
    for name, comp in ctx.by_name.items():
        if ctx.kind_of[name] != want_kind:
            continue
        refs = [f"{name}.{p}" for p in _FOUR_PORT]
        missing = [r for r in refs if r not in ctx.uf]
        if missing:
            ctx.errors.append(
                f"{name} ({comp.cls}): port(s) {', '.join(missing)} are not connected"
            )
            continue
        if any(ctx.kind_of_ref(r) == "conflict" for r in refs):
            continue  # `_assign_nodes` already reported the underlying conflict
        node_a1, node_b2 = ctx.node_of(f"{name}.port_a1"), ctx.node_of(f"{name}.port_b2")
        node_b1, node_a2 = ctx.node_of(f"{name}.port_b1"), ctx.node_of(f"{name}.port_a2")
        if node_a1 != node_b2 or node_b1 != node_a2 or node_a1 == node_b1:
            ctx.errors.append(
                f"{name} ({comp.cls}): port_a1/port_b2 must share one node and "
                f"port_b1/port_a2 the other (found port_a1={node_a1!r}, "
                f"port_b2={node_b2!r}, port_b1={node_b1!r}, port_a2={node_a2!r})"
            )
            continue
        edges.append(TwoWayEdge(component=comp, side_a=node_a1, side_b=node_b1))
    return edges


def _resolve_sources(ctx: _Ctx) -> list[tuple[Component, str]]:
    """Sources decorate whichever node their own port(s) resolve to."""
    sources: list[tuple[Component, str]] = []
    for name, comp in ctx.by_name.items():
        if ctx.kind_of[name] != "source":
            continue
        refs = ctx.fluid_refs_by_instance.get(name, [])
        if not refs:
            ctx.errors.append(f"{name} ({comp.cls}): not connected to any node")
            continue
        if ctx.kind_of_ref(refs[0]) == "conflict":
            continue  # `_assign_nodes` already reported the underlying conflict
        node_names = {ctx.node_of(r) for r in refs}
        if len(node_names) != 1:
            ctx.errors.append(
                f"{name} ({comp.cls}): connects to more than one node ({sorted(node_names)})"
            )
            continue
        sources.append((comp, next(iter(node_names))))
    return sources


def _fuse_paths(ctx: _Ctx) -> list[FlowPath]:
    """One walk per one-way flow element; see the module docstring for sign and orientation."""
    visited_columns: set[str] = set()
    visited_junction_roots: set[str] = set()

    def walk(own_ref: str, reversed_walk: bool) -> tuple[str, list[tuple[Component, int]]]:
        """Walk outward from `own_ref` (a flow element's own `port_a` or `port_b`) to the
        zone/boundary it terminates at, collecting `(column, sign)` in WALK order.

        `reversed_walk` is true for the `port_a` side: that walk proceeds from the element
        toward the path's `src` end, i.e. backwards along the `src`-to-`tgt` convention the
        module docstring defines signs against, so both the per-column sign and the final
        list order (handled by the caller) are flipped relative to a plain forward walk.
        """
        if own_ref not in ctx.uf:
            raise _ChainRefused(f"{own_ref}: not connected")
        ref = own_ref
        columns: list[tuple[Component, int]] = []
        while True:
            root = ctx.uf.find(ref)
            kind = ctx.root_to_kind[root]
            if kind in ("zone", "boundary"):
                return ctx.root_to_name[root], columns
            if kind == "conflict":
                raise _ChainRefused("touches a node that was already refused")
            visited_junction_roots.add(root)
            members = ctx.members_of_root[root]
            if len(members) != 2:
                raise _ChainRefused(
                    f"junction at {sorted(members)} has {len(members)} connections, not 2"
                )
            other_ref = next(m for m in members if m != ref)
            other_inst, other_port = _split(other_ref)
            other_kind = ctx.kind_of[other_inst]
            if other_kind == "column":
                visited_columns.add(other_inst)
                raw_sign = 1 if other_port == "port_a" else -1
                sign = -raw_sign if reversed_walk else raw_sign
                columns.append((ctx.by_name[other_inst], sign))
                exit_port = "port_b" if other_port == "port_a" else "port_a"
                ref = f"{other_inst}.{exit_port}"
                continue
            if other_kind in ("one_way", "door", "zonal"):
                raise _ChainRefused(
                    f"chain has more than one flow element: also found {other_inst}"
                )
            raise _ChainRefused(f"{other_inst}: unsupported in a hydrostatic-column chain")

    paths: list[FlowPath] = []
    for name, comp in ctx.by_name.items():
        if ctx.kind_of[name] != "one_way":
            continue
        try:
            src, cols_a = walk(f"{name}.port_a", reversed_walk=True)
            tgt, cols_b = walk(f"{name}.port_b", reversed_walk=False)
        except _ChainRefused as exc:
            ctx.errors.append(f"{name} ({comp.cls}): {exc.message}")
            continue
        columns = tuple(reversed(cols_a)) + tuple(cols_b)
        paths.append(FlowPath(element=comp, src=src, tgt=tgt, columns=columns))

    all_columns = {name for name, k in ctx.kind_of.items() if k == "column"}
    for leftover in sorted(all_columns - visited_columns):
        ctx.errors.append(
            f"{leftover} ({ctx.by_name[leftover].cls}): not part of any flow-element chain "
            f"(a hydrostatic-column chain needs exactly one flow element)"
        )
    all_junction_roots = {root for root, kind in ctx.root_to_kind.items() if kind == "junction"}
    for root in sorted(all_junction_roots - visited_junction_roots,
                        key=lambda r: min(ctx.members_of_root[r])):
        ctx.errors.append(
            f"{sorted(ctx.members_of_root[root])}: connects components with no flow element"
        )

    return paths


def _resolve_pins(doc: ModelicaDoc, ctx: _Ctx) -> list[Pin]:
    """`FixedTemperature.port` -> `ThermalConductor.port_a`/`port_b` -> zone `heatPort`."""
    heat_partner: dict[tuple[str, str], tuple[str, str]] = {}
    for a, b in doc.connections:
        ia, pa = _split(a)
        ib, pb = _split(b)
        if ia not in ctx.by_name or ib not in ctx.by_name:
            continue
        ka, kb = ctx.kind_of[ia], ctx.kind_of[ib]
        if _is_heat_port(ka, ctx.by_name[ia].cls, pa) and _is_heat_port(
            kb, ctx.by_name[ib].cls, pb
        ):
            heat_partner[(ia, pa)] = (ib, pb)
            heat_partner[(ib, pb)] = (ia, pa)

    pins: list[Pin] = []
    for name, comp in ctx.by_name.items():
        if ctx.kind_of[name] != "pin" or comp.cls != _THERMAL_CONDUCTOR:
            continue
        ends = [heat_partner.get((name, "port_a")), heat_partner.get((name, "port_b"))]
        if any(e is None for e in ends):
            ctx.errors.append(f"{name} ({comp.cls}): both ports must be connected")
            continue
        fixed = [
            e for e in ends
            if ctx.kind_of[e[0]] == "pin" and ctx.by_name[e[0]].cls == _FIXED_TEMPERATURE
        ]
        zone_ends = [e for e in ends if ctx.kind_of[e[0]] == "zone"]
        if len(fixed) != 1 or len(zone_ends) != 1:
            ctx.errors.append(
                f"{name} ({comp.cls}): unsupported heat-port wiring (expected a fixed "
                f"temperature on one side and a zone's heatPort on the other)"
            )
            continue
        pins.append(Pin(source=ctx.by_name[fixed[0][0]], conductor=comp, zone=zone_ends[0][0]))

    for name, comp in ctx.by_name.items():
        if ctx.kind_of[name] != "pin" or comp.cls != _FIXED_TEMPERATURE:
            continue
        partner = heat_partner.get((name, "port"))
        if partner is None:
            ctx.errors.append(f"{name} ({comp.cls}): not connected")
        elif not (
            ctx.kind_of[partner[0]] == "pin" and ctx.by_name[partner[0]].cls == _THERMAL_CONDUCTOR
        ):
            ctx.errors.append(f"{name} ({comp.cls}): unsupported heat-port wiring")

    return pins


def build(doc: ModelicaDoc) -> ComponentGraph:
    ctx = _classify_components(doc)
    _process_fluid_connections(doc, ctx)
    _assign_nodes(ctx)

    nodes = {
        ctx.root_to_name[root]: kind
        for root, kind in ctx.root_to_kind.items()
        if kind != "conflict"
    }
    zones = tuple(c.name for c in doc.components if ctx.kind_of[c.name] == "zone")
    boundaries = tuple(c.name for c in doc.components if ctx.kind_of[c.name] == "boundary")

    doors = tuple(_two_way_edges(ctx, "door"))
    zonal = tuple(_two_way_edges(ctx, "zonal"))
    sources = tuple(_resolve_sources(ctx))
    paths = tuple(_fuse_paths(ctx))
    pins = tuple(_resolve_pins(doc, ctx))

    _raise(ctx.errors)

    return ComponentGraph(
        nodes=nodes, zones=zones, boundaries=boundaries, paths=paths, doors=doors,
        zonal=zonal, pins=pins, sources=sources,
    )
