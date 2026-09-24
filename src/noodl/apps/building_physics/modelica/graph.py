"""Component graph over a `ModelicaDoc` (spec section 6, conversion rules).

`build(doc)` resolves every `connect()` pair the JSON carries (`doc.connections`) to a port on
one of `doc.components`, groups ports into NODES with union-find, fuses hydrostatic-column
chains into single edges, checks two-way wiring, and returns a `ComponentGraph`. It never
evaluates a Modelica expression: every parameter it reads was already evaluated by
OpenModelica, and every check here is about topology and class support, not physics. Task 6
turns the result into a noodl `Model`.

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
element between them).

Hydrostatic column chains and path orientation
-----------------------------------------------
A one-way flow element (`Orifice` and friends) together with any number of `MediumColumn`s
and degree-two junctions between it and each of its two end zones/boundaries fuses into one
`FlowPath` (spec section 6). The path is ORIENTED from the element's `port_a` side to its
`port_b` side: `FlowPath.src` is the zone/boundary reached by walking from the element's
`port_a`, `FlowPath.tgt` the one reached from `port_b`.

Each column on the path gets a sign: **+1** if, walking the path from its `src` end to its
`tgt` end, the column's OWN `port_a` (its top -- confirmed against
`Buildings/Airflow/Multizone/MediumColumn.mo`: `port_a` is the "positive design flow
direction" inlet, drawn at the top of the icon, `port_b` at the bottom) is reached before its
`port_b` (its bottom); **-1** otherwise. `FlowPath.columns` lists them `(component, sign)` in
that `src`-to-`tgt` order, so Task 6 computes the path's hydrostatic head as
`sum(sign * (-h * rho * g))` over the list, `rho` chosen per each column's own
`densitySelection` parameter. A chain with zero flow elements (a run of columns alone between
two zones/boundaries) or more than one is refused, naming every element found in it.

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
from dataclasses import dataclass

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
    `port_a` side) to `tgt` (its `port_b` side), with the columns encountered along the way."""

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


def build(doc: ModelicaDoc) -> ComponentGraph:  # noqa: C901, PLR0912, PLR0915
    errors: list[str] = []

    by_name: dict[str, Component] = {}
    kind_of: dict[str, str] = {}
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

    _raise(errors)

    # --- Fluid connections -> union-find, plus signal-link refusals. -------------------
    uf = _UnionFind()
    fluid_refs_by_instance: dict[str, list[str]] = {}
    for a, b in doc.connections:
        ia, pa = _split(a)
        ib, pb = _split(b)
        if ia not in by_name:
            errors.append(f"{a}: connection references unknown instance {ia!r}")
            continue
        if ib not in by_name:
            errors.append(f"{b}: connection references unknown instance {ib!r}")
            continue
        ka, kb = kind_of[ia], kind_of[ib]
        a_fluid, b_fluid = _is_fluid_port(ka, pa), _is_fluid_port(kb, pb)
        a_heat = _is_heat_port(ka, by_name[ia].cls, pa)
        b_heat = _is_heat_port(kb, by_name[ib].cls, pb)
        if a_fluid and b_fluid:
            uf.union(a, b)
            fluid_refs_by_instance.setdefault(ia, []).append(a)
            fluid_refs_by_instance.setdefault(ib, []).append(b)
        elif a_heat and b_heat:
            pass  # resolved separately below, re-reading `doc.connections` directly
        else:
            # Neither a fluid nor a heat connection: a signal-style wire between blocks
            # (spec section 6 / Task 5 resolution). Only refuse it if it feeds a component
            # this reader does not otherwise support; a legitimate signal link is not
            # recorded anywhere in `ComponentGraph` -- Task 6 reads `doc.signals` and
            # `doc.connections` directly for that.
            for inst in (ia, ib):
                reason = schema.refusal_reason(by_name[inst].cls)
                if reason is not None:
                    errors.append(f"{inst} ({by_name[inst].cls}): {reason}")

    _raise(errors)

    # Self-union: every port of the SAME zone/boundary/source instance is the same node.
    for inst, refs in fluid_refs_by_instance.items():
        if kind_of[inst] in ("zone", "boundary", "source"):
            first = refs[0]
            for r in refs[1:]:
                uf.union(first, r)

    # --- Node identity. ------------------------------------------------------------
    members_of_root = uf.groups()
    root_to_name: dict[str, str] = {}
    root_to_kind: dict[str, str] = {}
    junction_groups: list[tuple[str, set[str]]] = []
    for root, members in members_of_root.items():
        zb_names = sorted({
            by_name[_split(m)[0]].name for m in members
            if kind_of[_split(m)[0]] in ("zone", "boundary")
        })
        if len(zb_names) > 1:
            errors.append(
                f"{' and '.join(zb_names)}: connected directly to each other with no flow "
                f"element between them"
            )
            continue
        if zb_names:
            root_to_name[root] = zb_names[0]
            root_to_kind[root] = kind_of[zb_names[0]]
        else:
            junction_groups.append((root, members))

    _raise(errors)

    junction_groups.sort(key=lambda rm: min(rm[1]))
    for i, (root, _members) in enumerate(junction_groups):
        root_to_name[root] = f"_j{i}"
        root_to_kind[root] = "junction"

    def node_of(ref: str) -> str:
        return root_to_name[uf.find(ref)]

    nodes = {root_to_name[root]: kind for root, kind in root_to_kind.items()}
    zones = tuple(c.name for c in doc.components if kind_of[c.name] == "zone")
    boundaries = tuple(c.name for c in doc.components if kind_of[c.name] == "boundary")

    # --- Two-way elements (doors and zonal-flow elements). --------------------------
    def two_way_edges(want_kind: str) -> list[TwoWayEdge]:
        edges: list[TwoWayEdge] = []
        for name, comp in by_name.items():
            if kind_of[name] != want_kind:
                continue
            refs = [f"{name}.{p}" for p in _FOUR_PORT]
            missing = [r for r in refs if r not in uf]
            if missing:
                errors.append(
                    f"{name} ({comp.cls}): port(s) {', '.join(missing)} are not connected"
                )
                continue
            node_a1, node_b2 = node_of(f"{name}.port_a1"), node_of(f"{name}.port_b2")
            node_b1, node_a2 = node_of(f"{name}.port_b1"), node_of(f"{name}.port_a2")
            if node_a1 != node_b2 or node_b1 != node_a2 or node_a1 == node_b1:
                errors.append(
                    f"{name} ({comp.cls}): port_a1/port_b2 must share one node and "
                    f"port_b1/port_a2 the other (found port_a1={node_a1!r}, "
                    f"port_b2={node_b2!r}, port_b1={node_b1!r}, port_a2={node_a2!r})"
                )
                continue
            edges.append(TwoWayEdge(component=comp, side_a=node_a1, side_b=node_b1))
        return edges

    doors = tuple(two_way_edges("door"))
    zonal = tuple(two_way_edges("zonal"))

    # --- Sources decorate whichever node their own port(s) resolve to. --------------
    sources: list[tuple[Component, str]] = []
    for name, comp in by_name.items():
        if kind_of[name] != "source":
            continue
        refs = fluid_refs_by_instance.get(name, [])
        if not refs:
            errors.append(f"{name} ({comp.cls}): not connected to any node")
            continue
        node_names = {node_of(r) for r in refs}
        if len(node_names) != 1:
            errors.append(
                f"{name} ({comp.cls}): connects to more than one node ({sorted(node_names)})"
            )
            continue
        sources.append((comp, next(iter(node_names))))

    # --- Hydrostatic-column chains, one walk per one-way flow element. --------------
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
        if own_ref not in uf:
            raise _ChainRefused(f"{own_ref}: not connected")
        ref = own_ref
        columns: list[tuple[Component, int]] = []
        while True:
            root = uf.find(ref)
            kind = root_to_kind[root]
            if kind in ("zone", "boundary"):
                return root_to_name[root], columns
            visited_junction_roots.add(root)
            members = members_of_root[root]
            if len(members) != 2:
                raise _ChainRefused(
                    f"junction at {sorted(members)} has {len(members)} connections, not 2"
                )
            other_ref = next(m for m in members if m != ref)
            other_inst, other_port = _split(other_ref)
            other_kind = kind_of[other_inst]
            if other_kind == "column":
                visited_columns.add(other_inst)
                raw_sign = 1 if other_port == "port_a" else -1
                sign = -raw_sign if reversed_walk else raw_sign
                columns.append((by_name[other_inst], sign))
                exit_port = "port_b" if other_port == "port_a" else "port_a"
                ref = f"{other_inst}.{exit_port}"
                continue
            if other_kind in ("one_way", "door", "zonal"):
                raise _ChainRefused(
                    f"chain has more than one flow element: also found {other_inst}"
                )
            raise _ChainRefused(f"{other_inst}: unsupported in a hydrostatic-column chain")

    paths: list[FlowPath] = []
    for name, comp in by_name.items():
        if kind_of[name] != "one_way":
            continue
        try:
            src, cols_a = walk(f"{name}.port_a", reversed_walk=True)
            tgt, cols_b = walk(f"{name}.port_b", reversed_walk=False)
        except _ChainRefused as exc:
            errors.append(f"{name} ({comp.cls}): {exc.message}")
            continue
        columns = tuple(reversed(cols_a)) + tuple(cols_b)
        paths.append(FlowPath(element=comp, src=src, tgt=tgt, columns=columns))

    all_columns = {name for name, k in kind_of.items() if k == "column"}
    for leftover in sorted(all_columns - visited_columns):
        errors.append(
            f"{leftover} ({by_name[leftover].cls}): not part of any flow-element chain "
            f"(a hydrostatic-column chain needs exactly one flow element)"
        )
    all_junction_roots = {root for root, kind in root_to_kind.items() if kind == "junction"}
    for root in sorted(all_junction_roots - visited_junction_roots,
                        key=lambda r: min(members_of_root[r])):
        errors.append(
            f"{sorted(members_of_root[root])}: connects components with no flow element"
        )

    # --- Heat ports: FixedTemperature -> ThermalConductor -> zone.heatPort. ---------
    heat_partner: dict[tuple[str, str], tuple[str, str]] = {}
    for a, b in doc.connections:
        ia, pa = _split(a)
        ib, pb = _split(b)
        if ia not in by_name or ib not in by_name:
            continue
        ka, kb = kind_of[ia], kind_of[ib]
        if _is_heat_port(ka, by_name[ia].cls, pa) and _is_heat_port(kb, by_name[ib].cls, pb):
            heat_partner[(ia, pa)] = (ib, pb)
            heat_partner[(ib, pb)] = (ia, pa)

    pins: list[Pin] = []
    for name, comp in by_name.items():
        if kind_of[name] != "pin" or comp.cls != _THERMAL_CONDUCTOR:
            continue
        ends = [heat_partner.get((name, "port_a")), heat_partner.get((name, "port_b"))]
        if any(e is None for e in ends):
            errors.append(f"{name} ({comp.cls}): both ports must be connected")
            continue
        fixed = [
            e for e in ends if kind_of[e[0]] == "pin" and by_name[e[0]].cls == _FIXED_TEMPERATURE
        ]
        zone_ends = [e for e in ends if kind_of[e[0]] == "zone"]
        if len(fixed) != 1 or len(zone_ends) != 1:
            errors.append(
                f"{name} ({comp.cls}): unsupported heat-port wiring (expected a fixed "
                f"temperature on one side and a zone's heatPort on the other)"
            )
            continue
        pins.append(Pin(source=by_name[fixed[0][0]], conductor=comp, zone=zone_ends[0][0]))

    for name, comp in by_name.items():
        if kind_of[name] != "pin" or comp.cls != _FIXED_TEMPERATURE:
            continue
        partner = heat_partner.get((name, "port"))
        if partner is None:
            errors.append(f"{name} ({comp.cls}): not connected")
        elif not (kind_of[partner[0]] == "pin" and by_name[partner[0]].cls == _THERMAL_CONDUCTOR):
            errors.append(f"{name} ({comp.cls}): unsupported heat-port wiring")

    _raise(errors)

    return ComponentGraph(
        nodes=nodes, zones=zones, boundaries=boundaries, paths=tuple(paths), doors=doors,
        zonal=zonal, pins=tuple(pins), sources=tuple(sources),
    )
