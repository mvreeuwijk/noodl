"""The build-time view of a WSIMOD network: refusals, check recursion, river order, the plan.

Everything here depends only on the configuration (``WsimodNetwork``) and on ``pollutants=``,
never on a day's values, so ``build_model`` runs it once (D7): the reader refuses keys,
classes and formats; :func:`check_network` refuses the rest by name, through each class's
``validate`` and the structural checks below.

- :func:`recursion_cycles`: WSIMOD's checks recurse depth-first, and only some handlers
  recurse (``check_recurses`` per class; spec, Approach). A cycle of recursing handlers that a
  check can reach makes WSIMOD's ``query_handler`` retry at every level and hang
  (``nodes.py:335-343``); it is refused, naming the cycle.
- :func:`river_order`: the rivers that ``distribute`` after the orchestration
  (``model.py:697-735``, ``:780-807``).
- :func:`orchestration_plan`: each slot of the orchestration with the nodes it runs.
- :func:`forcing_readers`: which nodes read which forcing variable (spec, Forcing checks).
"""

from __future__ import annotations

from noodl.apps.wsimod.arcs import ARC_CLASSES
from noodl.apps.wsimod.classes import NODE_SCHEMAS
from noodl.apps.wsimod.config import WsimodNetwork
from noodl.apps.wsimod.engine import _register_all

__all__ = [
    "check_network",
    "forcing_readers",
    "orchestration_plan",
    "recursion_cycles",
    "river_order",
]

# the storage-family classes whose ``decays`` dict builds a DecayTank (a QueueGroundwater's a
# DecayQueueTank) that reads the node's temperature (a Groundwater's and a QueueGroundwater's
# is a filename override's, X11)
DECAYING_STORAGE = (
    "Storage",
    "Groundwater",
    "QueueGroundwater",
    "Reservoir",
    "RiverReservoir",
    "River",
)
# model.py:697-735: the WSIMOD types (after the renames) whose arcs give the river order
_RIVER_TYPES = ("River", "Node", "Waste", "Reservoir")
# The tags a check can be asked under in 3a: the default, a Sewer's and a WWTW's "Sewer",
# the demand tags (demand.py:100-104) and a Sewer's "Land" (sewer.py:96-104).
_TAGS = ("default", "Sewer", "Demand", "Land", ("Demand", "Garden"))
# The step methods that pull from upstream, so that pull checks run at all: create_demand
# (demand.py:86), treat_water (wtw.py:578) and make_abstractions (storage.py:983).
_PULLING_METHODS = ("create_demand", "treat_water", "make_abstractions")


def _classes():
    from noodl.apps.wsimod.node import NODE_CLASSES

    _register_all()
    return NODE_CLASSES


def check_network(network: WsimodNetwork, *, pollutants: bool) -> list[str]:
    """Every build-time refusal (D7), by name; returns what a quantity-only run ignores.

    In order: each arc's and each node's ``validate`` (travel times, preferences, neutral
    processes, ``decays``, porosity, leakage, forcing on every date, ...); a cycle in the check
    recursion (:func:`recursion_cycles`); river arcs without a ``Waste``
    (:func:`river_order`); an orchestration slot that names nodes lacking the method
    (:func:`orchestration_plan`). A configuration without dates is refused too: there is no
    day to run.

    The ignored parameters are ``"<node>.<parameter>"`` and
    ``"<node>.surfaces.<surface>.<parameter>"`` (a pollutant series of a catchment is
    ``"<node>.<pollutant>"``); empty when ``pollutants`` is true.
    """
    classes = _classes()
    if not network.dates:
        raise ValueError("the configuration lists no dates: there is no day to run")
    for arc in network.arcs:
        ARC_CLASSES[arc.cls].validate(arc, network)
    ignored: list[str] = []
    for spec in network.nodes:
        if spec.cls not in classes:
            raise ValueError(f"node {spec.name!r}: class {spec.cls!r} is not ported yet")
        for item in classes[spec.cls].validate(spec, network, pollutants=pollutants):
            ignored.append(f"{spec.name}.{item}")
    for cycle in recursion_cycles(network):
        raise ValueError(
            f"the checks recurse in a cycle: {' -> '.join(cycle)}; WSIMOD's query_handler "
            "would retry the call at every level and never finish (nodes.py:335-343; spec, "
            "Approach: check recursion)"
        )
    river_order(network)
    orchestration_plan(network)
    return ignored


def _sccs(nodes: list[str], edges: dict[str, list[str]]) -> list[list[str]]:
    """The strongly connected components of ``edges`` over ``nodes`` (iterative Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(edges.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(edges.get(w, ()))))
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                out.append(component)
    return out


def _cycle_path(component: set[str], edges: dict[str, list[str]], order: list[str]) -> list[str]:
    """One closed walk through ``component``, from its first node in configuration order."""
    start = next(n for n in order if n in component)
    parent: dict[str, str] = {}
    frontier = [start]
    seen = {start}
    while frontier:
        nxt = []
        for v in frontier:
            for w in edges.get(v, ()):
                if w not in component:
                    continue
                if w == start:
                    path = [v]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return [start, *reversed(path[:-1]), start]
                if w not in seen:
                    seen.add(w)
                    parent[w] = v
                    nxt.append(w)
        frontier = nxt
    return [start, start]  # pragma: no cover - a component always closes


def recursion_cycles(network: WsimodNetwork) -> list[list[str]]:
    """The cycles of recursing check handlers a check can reach, per direction and tag
    (spec, Approach), each as a closed walk ``[a, b, ..., a]`` of node names.

    A push check on a node calls its out-arcs' ``send_push_check`` when its handler recurses
    (``check_recurses``: ``Node.push_check_basic``, ``RiverReservoir``'s), so an arc ``u -> v``
    is a push recursion edge when ``v``'s push check for the tag recurses. A pull check walks
    the in-arcs, so ``u -> v`` gives the edge ``v -> u`` when ``u``'s pull check recurses.

    Push checks run on every push request (``arcs.py:131-134``), so every push cycle is
    reachable. Pull checks run only below a node that pulls: the step methods
    ``create_demand``, ``treat_water`` and ``make_abstractions`` start them, and they go up
    the in-arcs through recursing handlers only. So WSIMOD's own ``Land -> Sewer`` runoff with
    ``Sewer -> Land`` ponding, whose pull checks would recurse into each other, is a cycle no
    check enters: it is accepted (spec, Approach).
    """
    classes = _classes()
    order = [n.name for n in network.nodes]
    specs = {n.name: n for n in network.nodes}
    pulling = {
        name
        for slot, method in network.orchestration
        if method in _PULLING_METHODS
        for name in order
        if specs[name].slot_key == slot
    }
    cycles: list[list[str]] = []
    for direction in ("push", "pull"):
        for tag in _TAGS:

            def recurses(name: str, tag=tag, direction=direction) -> bool:
                spec = specs[name]
                return classes[spec.cls].check_recurses(spec, direction, tag)

            edges: dict[str, list[str]] = {}
            for arc in network.arcs:
                if direction == "push" and recurses(arc.out_port):
                    edges.setdefault(arc.in_port, []).append(arc.out_port)
                if direction == "pull" and recurses(arc.in_port):
                    edges.setdefault(arc.out_port, []).append(arc.in_port)
            if direction == "push":
                reachable = set(order)
            else:  # the nodes a pull check reaches: upstream of a pulling node
                reachable = set(pulling)
                frontier = list(pulling)
                while frontier:
                    v = frontier.pop()
                    for w in edges.get(v, ()):
                        if w not in reachable:
                            reachable.add(w)
                            frontier.append(w)
            sub = {
                v: [w for w in ws if w in reachable] for v, ws in edges.items() if v in reachable
            }
            for component in _sccs([n for n in order if n in reachable], sub):
                members = set(component)
                if len(component) > 1 or component[0] in sub.get(component[0], ()):
                    cycles.append(_cycle_path(members, sub, order))
    return cycles


def river_order(network: WsimodNetwork) -> list[str]:
    """The rivers that ``distribute`` after the orchestration, most upstream first
    (``model.py:697-735``, ``:780-807``).

    The arcs with both ends of WSIMOD type River, Node, Waste or Reservoir give every ``Waste``
    node upstreamness 0 and each layer upstream of it one more; a node keeps the first (lowest)
    value it gets. Only nodes under the type key ``"River"`` enter, and a river no river path
    links to a ``Waste`` is left out. WSIMOD breaks ties within a level by the set-hash order of
    the node names (``model.py:780-807``), which varies with ``PYTHONHASHSEED``, so there is no
    fixed order to copy; noodl breaks them by configuration order (spec, W13, W64).
    River arcs without any ``Waste`` are refused by name (WSIMOD raises ``ValueError`` from
    ``max([])``, ``model.py:799``; defect table).
    """
    types = {n.name: NODE_SCHEMAS[n.cls].renamed_type for n in network.nodes}
    arcs = [
        a
        for a in network.arcs
        if types[a.in_port] in _RIVER_TYPES and types[a.out_port] in _RIVER_TYPES
    ]
    if not arcs:
        return []
    level = {n.name: 0 for n in network.nodes if n.slot_key == "Waste"}
    if not level:
        raise ValueError(
            f"river arcs ({', '.join(a.name for a in arcs)}) but no Waste node: WSIMOD's river "
            "order fails on max([]) (model.py:799); add a Waste node downstream"
        )
    while True:
        upstream = {a.in_port for a in arcs if a.out_port in level} - level.keys()
        if not upstream:
            break
        layer = max(level.values()) + 1
        for n in network.nodes:
            if n.name in upstream:
                level[n.name] = layer
    rivers = [n.name for n in network.nodes if n.slot_key == "River" and n.name in level]
    return sorted(rivers, key=lambda name: -level[name])


def orchestration_plan(network: WsimodNetwork) -> list[tuple[str, str, list[str]]]:
    """The orchestration as ``(slot key, method, node names)``, the nodes in configuration
    order (``model.py:999-1003``).

    A slot key is the node's ``type_`` as written, ``"Foul"`` for a name containing
    ``"foul"`` (``model.py:650-673``), so a ``node_type_override`` changes the class but not the
    slot. A key naming no node is skipped, as ``nodes_type.get`` does; a key whose nodes lack
    the method is refused by name (WSIMOD raises ``AttributeError`` mid-run).
    """
    classes = _classes()
    plan: list[tuple[str, str, list[str]]] = []
    for slot, method in network.orchestration:
        specs = [n for n in network.nodes if n.slot_key == slot]
        if not specs:
            continue
        for spec in specs:
            if method not in classes[spec.cls].step_methods:
                raise ValueError(
                    f"orchestration {{{slot}: {method}}}: node {spec.name!r} ({spec.cls}) has "
                    f"no step method {method!r} (WSIMOD raises AttributeError mid-run)"
                )
        plan.append((slot, method, [n.name for n in specs]))
    return plan


def forcing_readers(network: WsimodNetwork) -> dict[str, list[str]]:
    """``{variable: nodes reading it}`` in configuration order (spec, Forcing checks).

    ``Land`` reads ``precipitation``, ``et0`` and ``temperature`` (``land.py:451-460``; the
    temperature also for the surfaces' end-of-day decay, ``core.py:485-487``), a decaying
    ``Storage``, ``Reservoir``, ``RiverReservoir`` or ``River`` (``temperature``, the same
    end-of-day read) or ``Groundwater`` or ``QueueGroundwater`` (a ``filename`` override's, X11), a
    ``ResidentialDemand`` ``temperature`` (``demand.py:321``) and a ``Catchment`` ``flow`` and
    every pollutant of the configuration (``catchment.py:59-74``). A variable no node reads is
    absent.
    """
    readers: dict[str, list[str]] = {}
    for spec in network.nodes:
        if spec.cls == "Land":
            variables: tuple[str, ...] = ("precipitation", "et0", "temperature")
        elif spec.cls == "ResidentialDemand":
            variables = ("temperature",)
        elif spec.cls == "Catchment":
            variables = ("flow", *network.pollutants.names)
        elif spec.cls in DECAYING_STORAGE and spec.parameters.get("decays") is not None:
            variables = ("temperature",)  # the DecayTank's end of day (core.py:484-487)
        else:
            continue
        for variable in variables:
            readers.setdefault(variable, []).append(spec.name)
    return readers
