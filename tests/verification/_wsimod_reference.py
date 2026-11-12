"""Instruments a live WSIMOD run to capture per-arc requests and realised flows.

Test-only. This module runs WSIMOD's OWN engine on WSIMOD's OWN demo models and records what
it actually did, so `AllocatedFlowLayer` can be validated by replay.

This module is a plain helper, not a test file, so it does not itself guard the
`wsimod` import -- every test module that imports it must call
`pytest.importorskip("wsimod")` before doing so, matching the pyswmm/wntr pattern.

Hook point: `Arc.send_push_request`/`send_pull_request` is
the one place both the per-arc REQUEST (pre-clip, the `vqip` argument) and the REALISED
transfer (`requested - reply`, or `reply` itself for a pull -- see each wrapper below)
are visible together, at exactly the per-edge granularity `AllocatedFlowLayer`'s
incidence structure needs. `Arc.get_excess` clips against an ACCUMULATING per-timestep
`flow_in`, reset only by `end_timestep`, so if an arc sees MORE THAN ONE push/pull event
within one WSIMOD timestep, each event is captured here as a SEPARATE row -- this
function does not sum them, deliberately: aggregation is the caller's decision, made
where the replay semantics are known, not silently here.

**Arcs DO see more than one event per timestep, in a demo this repo actually ships.**
`oxford_demo` has four such arcs (`abstraction_to_farmoor`, `evenlode_to_thames`,
`thames_to_thames`, `thames_to_farmoor`), each with both a push and a pull event in the
same timestep -- 5824 of its 28056 distinct `(arc, timestep)` pairs carry two rows.
`quickstart_demo` has no push+pull pairs, but BOTH demos have several SAME-direction
events per (arc, timestep): a pass-through `Node` forwards each push it receives as its
own push (quickstart `catchment_outflow`: two or three every timestep, 10184 raw events
in 7464 (arc, direction, timestep) rows; oxford: 48995 raw events in 33880 rows, over
seven arcs). Earlier versions of this docstring claimed one event per row for both.
Anything replaying these events through one
`AllocatedFlowLayer.step` per timestep must therefore ACCUMULATE a `(arc, t)`
group's rows rather than assign them one at a time -- `tests/verification/
test_allocation_wsimod_replay.py` does exactly that, and the fixture-writing script
`scripts/regenerate_wsimod_fixtures.py` sums over (arc, DIRECTION, timestep), leaving
push and pull as separate rows for the replay to sum. Both sums are exact for a
per-timestep replay: `get_excess` clips event k at `capacity - flow_in` with `flow_in`
the sum of events 1..k-1, so the timestep total is `min(sum of requests, capacity)`
whatever the order (`tests/verification/test_allocation_wsimod_capacity.py`).

Timestep index: `wsimod.orchestration.model.Model` has no `.t` or other current-
timestep attribute of its own (confirmed by inspecting a constructed `Model` directly).
`Model.run` instead sets `node.t = date` on every
node in `model.nodelist` at the START of each timestep, before that timestep's push/pull
calls are dispatched -- so `arc.in_port.t` holds the date active during any call this
harness intercepts. `t` in each captured event is that date's 0-based position in
`model.dates` (looked up once per event against a `{date: index}` table built when
`capture_events` is entered), which requires `model.dates` to already be set -- true for
every model built by `create_oxford_model` or the quickstart inline build before `.run()`
is ever called.
"""

from __future__ import annotations

import copy
import inspect
import io
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "ArcCall",
    "ArcDay",
    "NodeCall",
    "WsimodReference",
    "capture_arc_calls",
    "capture_events",
    "capture_node_calls",
    "extract_topology",
    "record_wsimod_run",
    "reset_wsimod_globals",
    "wsimod_model",
    "wsimod_reference",
]

_SURFACE_FIELDS = (
    ("storage", "storage"),
    ("evaporation", "evaporation"),
    ("precipitation", "precipitation"),
    ("percolation", "percolation"),
    ("subsurface_r", "subsurface_r"),
    ("surface_r", "surface_r"),
    ("tank_recharge", "tank_recharge"),
)


@contextmanager
def capture_events(model):
    """Yields a list that fills with one dict per push/pull event during `model.run()`.

    Each dict: `{"arc": str, "direction": "push" | "pull", "t": int, "requested":
    float, "realised": float}`. `requested`/`realised` are WSIMOD's raw per-timestep
    volumes (WSIMOD's native unit, not m^3/s -- callers convert at the boundary). See the
    module docstring for what `t` means and for why a caller
    must aggregate per `(arc, timestep)` itself -- an arc CAN see several events in one
    timestep, and this function never sums them.

    `model.arcs` is a plain `dict[str, Arc]` on `wsimod.orchestration.model.Model`
    (confirmed directly, not guarded with `hasattr`: a permanent fallback on a fact this
    cheap to check is exactly what this project's conventions refuse). Every original
    `send_push_request`/`send_pull_request` is restored on exit, including on an
    exception raised inside the `with` block.
    """
    if not model.arcs:
        raise ValueError(f"capture_events: model {model!r} has no arcs to instrument")
    dates = getattr(model, "dates", None)
    if dates is None:
        # `Model.__init__` never sets `self.dates` at all (confirmed directly, not
        # assumed) -- it is only assigned after construction, by `create_oxford_model`
        # or the quickstart inline build. `getattr` here is
        # a genuine "has this been set yet" check, not a guess at an alternative
        # attribute name.
        raise ValueError(
            "capture_events: model.dates is not set -- build the model (which sets "
            "`.dates`, e.g. create_oxford_model) before capturing, not after"
        )

    events: list[dict] = []
    date_index = {date: i for i, date in enumerate(dates)}

    def _t(arc) -> int:
        date = arc.in_port.t
        try:
            return date_index[date]
        except KeyError as exc:
            raise ValueError(
                f"capture_events: arc {arc.name!r}'s in_port {arc.in_port.name!r} has "
                f"t={date!r}, not one of model.dates -- was the model run with a "
                "different `dates` than the one `capture_events` was entered with?"
            ) from exc

    originals = []
    for arc in model.arcs.values():
        orig_push = arc.send_push_request
        orig_pull = arc.send_pull_request

        def make_push(arc=arc, orig=orig_push):
            def wrapped(vqip, tag="default", force=False):
                requested = vqip["volume"]
                reply = orig(vqip, tag=tag, force=force)
                events.append(
                    {
                        "arc": arc.name,
                        "direction": "push",
                        "t": _t(arc),
                        "requested": requested,
                        "realised": requested - reply["volume"],
                    }
                )
                return reply

            return wrapped

        def make_pull(arc=arc, orig=orig_pull):
            def wrapped(vqip, tag="default"):
                requested = vqip["volume"]
                reply = orig(vqip, tag=tag)
                events.append(
                    {
                        "arc": arc.name,
                        "direction": "pull",
                        "t": _t(arc),
                        "requested": requested,
                        "realised": reply["volume"],
                    }
                )
                return reply

            return wrapped

        arc.send_push_request = make_push()
        arc.send_pull_request = make_pull()
        originals.append((arc, orig_push, orig_pull))
    try:
        yield events
    finally:
        for arc, orig_push, orig_pull in originals:
            arc.send_push_request = orig_push
            arc.send_pull_request = orig_pull


def extract_topology(model) -> dict:
    """Node names/types and arc `(name, source, target, capacity)`, in the model's own
    iteration order.

    Reads `model.nodes`/`model.arcs` (plain `dict`s, see `capture_events`) and each
    arc's own `.in_port`/`.out_port`/`.capacity` attributes directly off the constructed
    WSIMOD `Model` -- never hand-transcribed, closing off the class of bug a hand-port
    invites.
    """
    if not model.arcs:
        raise ValueError(f"extract_topology: model {model!r} has no arcs")
    return {
        "nodes": [{"name": n.name, "type": type(n).__name__} for n in model.nodes.values()],
        "arcs": [
            {
                "name": a.name,
                "source": a.in_port.name,
                "target": a.out_port.name,
                "capacity": float(a.capacity),
            }
            for a in model.arcs.values()
        ],
    }


def reset_wsimod_globals() -> None:
    """Restores `wsimod.core.constants`' mutable globals to WSIMOD 0.8.1's defaults.

    `Model.load` and `create_oxford_model` overwrite `POLLUTANTS`, `ADDITIVE_POLLUTANTS`,
    `NON_ADDITIVE_POLLUTANTS` and `FLOAT_ACCURACY` for the whole process (`model.py:232-241`,
    `create_oxford.py:171-180`); every WSIMOD build starts from the defaults listed in
    `constants.py:21-50`, with `FLOAT_ACCURACY = 1e-11` (`constants.py:3`).

    The mutable defaults of WSIMOD's constructors are restored too. An override of a dict
    parameter updates the dict in place (`land.py:397`, `demand.py:76`, `tanks.py:439`), so an
    override applied to an instance built with the shared default `{}` changes that default for
    every later instance in the process (the spec's defect table). Within one model WSIMOD's
    behaviour is untouched; across builds each one starts as a fresh process would.
    """
    from wsimod.core import constants

    _restore_constructor_defaults()

    constants.POLLUTANTS = [
        "do", "org-phosphorus", "phosphate", "ammonia", "solids", "bod", "cod", "ph",
        "temperature", "nitrate", "nitrite", "org-nitrogen",
    ]  # fmt: skip
    constants.NON_ADDITIVE_POLLUTANTS = ["do", "temperature", "ph"]
    constants.ADDITIVE_POLLUTANTS = [
        "org-phosphorus", "phosphate", "ammonia", "solids", "bod", "cod", "nitrate",
        "nitrite", "org-nitrogen",
    ]  # fmt: skip
    constants.FLOAT_ACCURACY = 1e-11


def _snapshot_constructor_defaults() -> dict | None:
    """Every ``__init__`` default of the classes in ``wsimod.nodes`` and ``wsimod.arcs``, taken
    when this module is imported, before any build of this process can mutate one; ``None``
    without WSIMOD (the importing test module skips then, see the module text)."""
    try:
        import wsimod.arcs
        import wsimod.nodes
    except ImportError:
        return None
    import importlib
    import pkgutil

    functions = {}
    for package in (wsimod.nodes, wsimod.arcs):
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            for value in vars(module).values():
                init = vars(value).get("__init__") if isinstance(value, type) else None
                if inspect.isfunction(init) and init.__module__.startswith("wsimod"):
                    functions[init] = (init.__defaults__, init.__kwdefaults__)
    return copy.deepcopy(functions)


_CONSTRUCTOR_DEFAULTS = _snapshot_constructor_defaults()


def _restore_constructor_defaults() -> None:
    """Restore the import-time snapshot of WSIMOD's constructor defaults."""
    for function, (defaults, kwdefaults) in (_CONSTRUCTOR_DEFAULTS or {}).items():
        function.__defaults__ = copy.deepcopy(defaults)
        function.__kwdefaults__ = copy.deepcopy(kwdefaults)


def wsimod_model(source: Path | Mapping, *, config_name: str = "config.yml"):
    """A WSIMOD `Model` built from a configuration directory or a mapping, from default globals.

    A directory goes through `Model.load(source, config_name)`. A mapping (in `Model.load`'s
    post-read form: inline `data_input_dict` keyed `(variable, date string)`, surfaces as a
    list or dict) goes through the same steps on a deep copy (`model.py:232-274`): the
    globals, the orchestration, `add_nodes`, `add_arcs`, `add_overrides`, the dates.
    """
    from wsimod.extensions import apply_patches
    from wsimod.orchestration.model import Model, to_datetime

    reset_wsimod_globals()
    model = Model()
    if not isinstance(source, Mapping):
        model.load(str(source), config_name)
        return model
    from wsimod.core import constants

    data = copy.deepcopy(dict(source))
    constants.POLLUTANTS = data.get("pollutants", constants.POLLUTANTS)
    constants.ADDITIVE_POLLUTANTS = data.get("additive_pollutants", constants.ADDITIVE_POLLUTANTS)
    constants.NON_ADDITIVE_POLLUTANTS = data.get(
        "non_additive_pollutants", constants.NON_ADDITIVE_POLLUTANTS
    )
    constants.FLOAT_ACCURACY = float(data.get("float_accuracy", constants.FLOAT_ACCURACY))
    model.extensions = data.get("extensions", [])
    if "orchestration" in data:
        model.orchestration = data["orchestration"]
    nodes = data["nodes"]
    nodes = list(nodes.values()) if isinstance(nodes, Mapping) else list(nodes)
    arcs = data.get("arcs", {})
    arcs = list(arcs.values()) if isinstance(arcs, Mapping) else list(arcs)

    def keyed(table):
        return {(variable, to_datetime(date)): value for (variable, date), value in table.items()}

    for node in nodes:
        if "data_input_dict" in node:
            node["data_input_dict"] = keyed(node["data_input_dict"])
        surfaces = node.get("surfaces")
        if isinstance(surfaces, Mapping):
            surfaces = node["surfaces"] = list(surfaces.values())
        for surface in surfaces or []:
            if "data_input_dict" in surface:
                surface["data_input_dict"] = keyed(surface["data_input_dict"])
    if "dates" in data:
        model.dates = [to_datetime(x) for x in data["dates"]]
    model.add_nodes(nodes)
    model.add_arcs(arcs)
    model.add_overrides(data.get("overrides", {}))
    apply_patches(model)
    return model


@dataclass
class WsimodReference:
    """What WSIMOD's own `run` recorded, as tensors (`record_all=True`, `model.py:1075-1133`)."""

    pollutants: tuple[str, ...]
    dates: list[str]
    arc_names: list[str]
    arc_flow: torch.Tensor  # (T, n_arcs)
    arc_quality: torch.Tensor  # (T, n_arcs, K)
    tank_names: list[tuple[str, str]]  # (node, prop)
    tank_storage: torch.Tensor  # (T, n_tanks)
    tank_quality: torch.Tensor  # (T, n_tanks, K)
    surface_names: list[tuple[str, str]]  # (node, surface)
    surface: dict[str, torch.Tensor]  # field -> (T, n_surfaces), NaN where not recorded
    surface_quality: torch.Tensor  # (T, n_surfaces, K)
    printed: list[str] = field(default_factory=list)


def _tensor(rows, key, n_days, width):
    values = [float("nan") if key not in row else row[key] for row in rows]
    return torch.tensor(values, dtype=torch.float64).reshape(n_days, width)


def record_wsimod_run(model, *, days: int | None = None) -> WsimodReference:
    """Runs `model` for its first `days` dates under `redirect_stdout` and records everything."""
    from wsimod.core import constants

    dates = list(model.dates)[:days]
    pollutants = tuple(constants.POLLUTANTS)
    out = io.StringIO()
    with redirect_stdout(out):
        flows, tanks, _, surfaces = model.run(dates=dates, verbose=True, record_all=True)
    n_days = len(dates)

    def quality(rows, width):
        values = [[row[p] for p in pollutants] for row in rows]
        return torch.tensor(values, dtype=torch.float64).reshape(n_days, width, len(pollutants))

    n_arcs = len(flows) // n_days
    n_tanks = len(tanks) // n_days
    n_surfaces = len(surfaces) // n_days
    return WsimodReference(
        pollutants=pollutants,
        dates=[str(d) for d in dates],
        arc_names=[row["arc"] for row in flows[:n_arcs]],
        arc_flow=_tensor(flows, "flow", n_days, n_arcs),
        arc_quality=quality(flows, n_arcs),
        tank_names=[(row["node"], row["prop"]) for row in tanks[:n_tanks]],
        tank_storage=_tensor(tanks, "storage", n_days, n_tanks),
        tank_quality=quality(tanks, n_tanks),
        surface_names=[(row["node"], row["surface"]) for row in surfaces[:n_surfaces]],
        surface={name: _tensor(surfaces, key, n_days, n_surfaces) for name, key in _SURFACE_FIELDS},
        surface_quality=quality(surfaces, n_surfaces),
        printed=out.getvalue().splitlines(),
    )


def wsimod_reference(
    source, *, config_name: str = "config.yml", days: int | None = None
) -> WsimodReference:
    """`wsimod_model(source)` run for `days` days (all dates by default), recorded."""
    return record_wsimod_run(wsimod_model(source, config_name=config_name), days=days)


# --- per-node call capture (spec, Verification 1) --------------------------------------------
@dataclass
class ArcCall:
    """One arc call a captured node made: the arguments as they were at entry (deep copies,
    before WSIMOD can mutate them), the reply, and the arc's accumulators at entry and at
    return."""

    arc: str
    method: str  # send_push_request, send_pull_request, send_push_check, send_pull_check
    vqip: dict | None
    tag: object
    force: bool
    reply: dict
    flow_in: float
    vqip_in: dict
    # the accumulators at return
    flow_in_after: float = 0.0
    vqip_in_after: dict = field(default_factory=dict)
    flow_out_after: float = 0.0
    vqip_out_after: dict = field(default_factory=dict)


@dataclass
class NodeCall:
    """One call into a captured node: an orchestration method (``tag`` and ``vqip`` None) or
    one of its four handler entry points, with its stores before and after and the arc calls
    it made itself, in order. ``pre``/``post`` hold the stores by ``Node.stores``' attribute
    paths and ``"arcs.<name>.vqip_in"``/``"arcs.<name>.flow_in"`` of the node's out-arcs."""

    day: int
    method: str
    vqip: dict | None
    tag: object
    reply: dict | None
    pre: dict[str, object]
    post: dict[str, object]
    arc_calls: list[ArcCall]


_HANDLERS = ("pull_set", "push_set", "pull_check", "push_check")
_ARC_METHODS = {  # the end of the arc that makes the call (arcs.py:112-228)
    "send_push_request": "in_port",
    "send_push_check": "in_port",
    "send_pull_request": "out_port",
    "send_pull_check": "out_port",
}


def _store_paths(node) -> tuple[str, ...]:
    """``Node.store_paths`` of the noodl class of ``node``. ``__qualname__`` survives the
    renames that set ``__class__.__name__`` (``storage.py:294``, ``:1043``, ``demand.py:214``)."""
    from noodl.apps.wsimod.engine import _register_all
    from noodl.apps.wsimod.node import NODE_CLASSES

    _register_all()
    cls = NODE_CLASSES[type(node).__qualname__]
    if hasattr(cls, "store_paths_of"):  # Land: the paths depend on its surfaces
        return cls.store_paths_of([(type(s).__name__, s.surface) for s in node.surfaces])
    tank = type(getattr(node, "tank", None)).__name__
    if tank == "DecayTank":  # D8: a per-day store
        return (*cls.store_paths, "tank.total_decayed")
    if tank == "DecayQueueTank":  # D8: carried, extracted at the next end of day
        return (*cls.store_paths, "tank.internal_arc.total_decayed")
    return cls.store_paths


def _wsimod_store(node, path: str):
    """The WSIMOD value of a noodl store path: an attribute, except that noodl's dense queue
    (``<tank>.internal_arc.register``/``.present``) is WSIMOD's ``queue`` dict (the rows, by
    time) and its key set (``tanks.py:481-486``, ``arcs.py:567``)."""
    for suffix, make in (
        (".internal_arc.register", lambda queue: queue),
        (".internal_arc.present", lambda queue: sorted(queue)),
    ):
        if path.endswith(suffix):
            arc = _attribute(node, path[: -len(suffix)] + ".internal_arc")
            return copy.deepcopy(make(arc.queue))
    return copy.deepcopy(_attribute(node, path))


def _attribute(obj, path: str):
    """``obj``'s attribute at the dotted ``path``; ``surfaces.<name>`` is a ``Land``'s surface of
    that name (``get_surface``, ``land.py:279-292``)."""
    parts = path.split(".")
    while parts:
        part = parts.pop(0)
        obj = obj.get_surface(parts.pop(0)) if part == "surfaces" else getattr(obj, part)
    return obj


@contextmanager
def capture_node_calls(model, nodes: Sequence[str]) -> Iterator[dict[str, list[NodeCall]]]:
    """Record every call into the named nodes while ``model`` runs (spec, Verification 1).

    Wraps each target's orchestration methods (the slots that name it, and ``distribute`` when
    it is in the river order), its ``end_timestep`` (stores before and after: a ``DecayTank``
    decays there), its four handler entry points, and the ``send_*`` methods of the
    arcs it calls (out-arcs push, in-arcs pull). Every argument is deep-copied at entry, before
    WSIMOD can mutate it (spec, Protocol, Request mutation). A call into a target that is
    already in a captured call is refused (D22); the error is raised again at exit, because
    WSIMOD's ``query_handler`` swallows exceptions and retries (``nodes.py:335-343``).
    Everything is restored on exit.
    """
    targets = {name: model.nodes[name] for name in nodes}
    calls: dict[str, list[NodeCall]] = {name: [] for name in targets}
    current: dict[str, NodeCall | None] = dict.fromkeys(targets)
    errors: list[str] = []
    restore: list[tuple[object, str, bool, object]] = []
    date_index = {date: i for i, date in enumerate(model.dates)}

    def snapshot(node) -> dict[str, object]:
        out: dict[str, object] = {path: _wsimod_store(node, path) for path in _store_paths(node)}
        for arc in node.out_arcs.values():
            out[f"arcs.{arc.name}.vqip_in"] = copy.deepcopy(arc.vqip_in)
            out[f"arcs.{arc.name}.flow_in"] = arc.flow_in
        return out

    def install(obj, attr: str, make) -> None:
        own = attr in vars(obj)
        restore.append((obj, attr, own, vars(obj).get(attr)))
        setattr(obj, attr, make(getattr(obj, attr)))

    def node_wrapper(node, method: str, handler: bool) -> None:
        def make(original):
            signature = inspect.signature(original)

            def wrapped(*args, **kwargs):
                if current[node.name] is not None:
                    message = (
                        f"capture_node_calls: {method} re-enters {node.name!r} during "
                        f"{current[node.name].method} (D22)"
                    )
                    errors.append(message)
                    raise RuntimeError(message)
                vqip = tag = None
                if handler:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    vqip, tag = bound.arguments["vqip"], bound.arguments["tag"]
                call = NodeCall(
                    day=date_index[node.t],
                    method=method,
                    vqip=copy.deepcopy(vqip),
                    tag=copy.deepcopy(tag),
                    reply=None,
                    pre=snapshot(node),
                    post={},
                    arc_calls=[],
                )
                calls[node.name].append(call)
                current[node.name] = call
                try:
                    reply = original(*args, **kwargs)
                finally:
                    current[node.name] = None
                call.reply = copy.deepcopy(reply)
                call.post = snapshot(node)
                return reply

            return wrapped

        install(node, method, make)

    def arc_wrapper(arc, method: str) -> None:
        owner = getattr(arc, _ARC_METHODS[method]).name

        def make(original):
            signature = inspect.signature(original)

            def wrapped(*args, **kwargs):
                call = current.get(owner)
                if call is None:
                    return original(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                record = ArcCall(
                    arc=arc.name,
                    method=method,
                    # QueueArc.send_push_request names the request ``vqip_`` (arcs.py:360)
                    vqip=copy.deepcopy(bound.arguments.get("vqip", bound.arguments.get("vqip_"))),
                    tag=copy.deepcopy(bound.arguments.get("tag", "default")),
                    force=bool(bound.arguments.get("force", False)),
                    reply={},
                    flow_in=arc.flow_in,
                    vqip_in=copy.deepcopy(arc.vqip_in),
                )
                call.arc_calls.append(record)
                reply = original(*args, **kwargs)
                record.reply = copy.deepcopy(reply)
                record.flow_in_after = arc.flow_in
                record.vqip_in_after = copy.deepcopy(arc.vqip_in)
                record.flow_out_after = arc.flow_out
                record.vqip_out_after = copy.deepcopy(arc.vqip_out)
                return reply

            return wrapped

        install(arc, method, make)

    for name, node in targets.items():
        steps = {
            method
            for slot in model.orchestration
            for key, method in slot.items()
            if any(other is node for other in model.nodes_type.get(key, {}).values())
        }
        if name in getattr(model, "river_discharge_order", []):
            steps.add("distribute")
        steps.add("end_timestep")  # model.py:1172-1174
        for method in sorted(steps):
            node_wrapper(node, method, handler=False)
        for method in _HANDLERS:
            node_wrapper(node, method, handler=True)
        for arc in node.out_arcs.values():
            arc_wrapper(arc, "send_push_request")
            arc_wrapper(arc, "send_push_check")
        for arc in node.in_arcs.values():
            arc_wrapper(arc, "send_pull_request")
            arc_wrapper(arc, "send_pull_check")
    try:
        yield calls
    finally:
        for obj, attr, own, value in reversed(restore):
            if own:
                setattr(obj, attr, value)
            else:
                delattr(obj, attr)
    if errors:
        raise RuntimeError(errors[0])


# --- per-arc call capture (spec, Verification 1, the replay harness) -------------------------
@dataclass
class ArcDay:
    """One user ``Decay*`` arc on one day, as WSIMOD's own run did it.

    ``calls`` holds, in the order WSIMOD made them, each request call into the arc
    (``send_push_request``/``send_pull_request``, ``reply`` what the arc returned), the
    ``update_queue`` it ran, and the calls the arc made into its ports: ``push_check`` and
    ``push_set`` into the ``out_port``, ``pull_check`` and ``pull_set`` into the ``in_port``,
    with the port's reply. (Port calls carry the arc's name in ``arc``.) A request call's
    *static call index* is its place among the day's request calls, both directions.

    ``entry_diffs`` is ``(static call index, diff)`` of each entry ``make_decay``, and
    ``end_diffs`` ``(row, diff)`` of each end-of-day one, in WSIMOD's order. A row is
    ``(direction, entry day, index among that direction's calls of its entry day)`` for a
    ``DecayArc``, and ``("slot", key)`` for a ``DecayArcAlt``. ``queue_at_end`` is the queue
    before ``end_timestep``, in list order: ``direction``, ``time`` (days left), ``vqip`` and
    the ``row`` (``slot`` and ``vqip`` for a ``DecayArcAlt``). ``pre`` and ``post`` hold WSIMOD's
    ``queue`` before the day's first call and after ``end_timestep`` (a list of requests, or the
    ``{key: vqip}`` dict). ``total_decayed_before_reset`` is WSIMOD's ``total_decayed`` as
    ``end_timestep`` was entered (the end of the day before, then this day's entries) and
    ``total_decayed_after`` as it returned (this day's end-of-day decays).
    """

    day: int
    calls: list[ArcCall] = field(default_factory=list)
    entry_diffs: list[tuple[int, dict]] = field(default_factory=list)
    end_diffs: list[tuple[tuple, dict]] = field(default_factory=list)
    queue_at_end: list[dict] = field(default_factory=list)
    total_decayed_before_reset: dict = field(default_factory=dict)
    total_decayed_after: dict = field(default_factory=dict)
    pre: dict[str, object] = field(default_factory=dict)
    post: dict[str, object] = field(default_factory=dict)


class _PortProxy:
    """The arc's view of one port: records the calls the arc makes (``arcs.py:112-258``) while
    one of its request calls runs, and forwards everything else untouched."""

    def __init__(self, real, recorder: _ArcRecorder) -> None:
        self._real, self._recorder = real, recorder

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _forward(self, method: str, vqip, tag):
        recorder = self._recorder
        record = recorder.record_call(method, vqip, tag)
        reply = getattr(self._real, method)(copy.deepcopy(vqip), tag)
        if record is not None:
            recorder.finish_call(record, reply)
        return reply

    def push_check(self, vqip=None, tag="default"):
        return self._forward("push_check", vqip, tag)

    def push_set(self, vqip, tag="default"):
        return self._forward("push_set", vqip, tag)

    def pull_check(self, vqip=None, tag="default"):
        return self._forward("pull_check", vqip, tag)

    def pull_set(self, vqip, tag="default"):
        return self._forward("pull_set", vqip, tag)


class _ArcRecorder:
    """The state of :func:`capture_arc_calls` for one arc."""

    def __init__(self, arc, date_index: dict, errors: list[str]) -> None:
        self.arc, self.date_index, self.errors = arc, date_index, errors
        self.alt = hasattr(arc, "max_travel")
        self.days: dict[int, ArcDay] = {}
        self.current: tuple[ArcCall, int] | None = None  # the request call under way, its index
        self.today: int | None = None
        self.index = 0  # request calls so far today
        self.per_direction = {"push": 0, "pull": 0}
        self.phase = "entry"
        self.raw: list[dict] = []  # the end-of-day diffs so far
        # a queued request -> its row; the request is kept, so that its id cannot be reused
        self.rows: dict[int, tuple[dict, tuple]] = {}

    def fail(self, message: str) -> None:
        self.errors.append(message)
        raise RuntimeError(message)

    def day(self) -> ArcDay:
        number = self.date_index[self.arc.in_port.t]
        if number != self.today:
            self.today, self.index = number, 0
            self.per_direction = {"push": 0, "pull": 0}
        if number not in self.days:
            self.days[number] = ArcDay(day=number, pre=self.snapshot())
        return self.days[number]

    def snapshot(self) -> dict[str, object]:
        queue = self.arc.queue
        if self.alt:
            return {"queue": copy.deepcopy(dict(queue))}
        return {"queue": [self.queue_row(request) for request in queue]}

    def queue_row(self, request: dict) -> dict:
        return {
            "direction": request["direction"],
            "time": request["time"],
            "vqip": copy.deepcopy(request["vqip"]),
            "row": self.rows[id(request)][1],
        }

    def record_call(self, method: str, vqip, tag) -> ArcCall | None:
        """A port call of the request under way (a check a node makes is not recorded)."""
        if self.current is None:
            return None
        arc = self.arc
        record = ArcCall(
            arc=arc.name,
            method=method,
            vqip=copy.deepcopy(vqip),
            tag=copy.deepcopy(tag),
            force=False,
            reply={},
            flow_in=arc.flow_in,
            vqip_in=copy.deepcopy(arc.vqip_in),
        )
        self.day().calls.append(record)
        return record

    def finish_call(self, record: ArcCall, reply) -> None:
        arc = self.arc
        record.reply = copy.deepcopy(reply)
        record.flow_in_after = arc.flow_in
        record.vqip_in_after = copy.deepcopy(arc.vqip_in)
        record.flow_out_after = arc.flow_out
        record.vqip_out_after = copy.deepcopy(arc.vqip_out)

    def request(self, method: str, original):
        signature = inspect.signature(original)
        direction = "push" if method == "send_push_request" else "pull"

        def wrapped(*args, **kwargs):
            if self.current is not None:
                self.fail(f"capture_arc_calls: {method} re-enters {self.arc.name!r}")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments.get("time", 0):
                self.fail(f"capture_arc_calls: {self.arc.name!r} got a request with a time")
            day = self.day()
            j = self.per_direction[direction]
            self.per_direction[direction] += 1
            index, self.index = self.index, self.index + 1
            record = ArcCall(
                arc=self.arc.name,
                method=method,
                # QueueArc.send_push_request names the request ``vqip_`` (arcs.py:360)
                vqip=copy.deepcopy(bound.arguments.get("vqip", bound.arguments.get("vqip_"))),
                tag=copy.deepcopy(bound.arguments.get("tag", "default")),
                force=bool(bound.arguments.get("force", False)),
                reply={},
                flow_in=self.arc.flow_in,
                vqip_in=copy.deepcopy(self.arc.vqip_in),
            )
            day.calls.append(record)
            self.current = (record, index)
            try:
                reply = original(*args, **kwargs)
            finally:
                self.current = None
            self.finish_call(record, reply)
            if not self.alt:
                for request in self.arc.queue:
                    self.rows.setdefault(id(request), (request, (direction, day.day, j)))
            return reply

        return wrapped

    def update_queue(self, original):
        def wrapped(*args, **kwargs):
            record = self.record_call("update_queue", None, None)
            reply = original(*args, **kwargs)
            if record is not None:
                self.finish_call(record, reply)
            return reply

        return wrapped

    def decay(self, original):
        """``generic_temperature_decay`` (``core.py:495-523``) returns the ``diff`` that
        ``make_decay`` adds to ``total_decayed``."""

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            diff = copy.deepcopy(result[1])
            if self.phase == "end":
                self.raw.append(diff)
            elif self.current is None:
                self.fail(f"capture_arc_calls: {self.arc.name!r} decayed outside a request call")
            else:
                self.day().entry_diffs.append((self.current[1], diff))
            return result

        return wrapped

    def end_timestep(self, original):
        arc = self.arc

        def wrapped():
            day = self.day()
            if self.alt:  # arcs.py:798-805: the keys above 0 in order, then key 0
                slots = [i + 1 for i in range(arc.max_travel) if (i + 1) in arc.queue] + [0]
                day.queue_at_end = [
                    {"slot": k, "vqip": copy.deepcopy(arc.queue[k])} for k in sorted(arc.queue)
                ]
                rows = [("slot", k) for k in slots]
            else:
                day.queue_at_end = [self.queue_row(request) for request in arc.queue]
                rows = [row["row"] for row in day.queue_at_end]
            day.total_decayed_before_reset = copy.deepcopy(arc.total_decayed)
            self.phase, self.raw = "end", []
            try:
                original()
            finally:
                self.phase = "entry"
            if len(self.raw) != len(rows):
                self.fail(
                    f"capture_arc_calls: {arc.name!r} decayed {len(self.raw)} requests at the "
                    f"end of the day, expected {len(rows)}"
                )
            day.end_diffs = list(zip(rows, self.raw, strict=True))
            day.total_decayed_after = copy.deepcopy(arc.total_decayed)
            day.post = self.snapshot()

        return wrapped


@contextmanager
def capture_arc_calls(model, arcs: Sequence[str]) -> Iterator[dict[str, list[ArcDay]]]:
    """Record the named user ``DecayArc``/``DecayArcAlt`` arcs while ``model`` runs (spec,
    Verification 1, the replay harness): one :class:`ArcDay` per arc and day.

    Wraps each arc's request calls, ``update_queue`` and ``end_timestep``, its
    ``generic_temperature_decay`` (every per-request ``diff``, tagged with its call or its
    row) and its two ports (the calls the arc makes into them, with their replies). Every
    argument is deep-copied at entry, before WSIMOD can mutate it; no result is changed, and
    everything is restored on exit. The lists fill while the model runs and are ordered by day
    when the block exits.
    """
    errors: list[str] = []
    date_index = {date: i for i, date in enumerate(model.dates)}
    recorders = {name: _ArcRecorder(model.arcs[name], date_index, errors) for name in arcs}
    restore: list[tuple[object, str, bool, object]] = []

    def install(obj, attr: str, make) -> None:
        own = attr in vars(obj)
        restore.append((obj, attr, own, vars(obj).get(attr)))
        setattr(obj, attr, make(getattr(obj, attr)))

    for recorder in recorders.values():
        arc = recorder.arc
        install(arc, "send_push_request", lambda o, r=recorder: r.request("send_push_request", o))
        install(arc, "send_pull_request", lambda o, r=recorder: r.request("send_pull_request", o))
        install(arc, "update_queue", recorder.update_queue)
        install(arc, "generic_temperature_decay", recorder.decay)
        install(arc, "end_timestep", recorder.end_timestep)
        for port in ("in_port", "out_port"):
            install(arc, port, lambda real, r=recorder: _PortProxy(real, r))
    result: dict[str, list[ArcDay]] = {name: [] for name in arcs}
    try:
        yield result
    finally:
        for obj, attr, own, value in reversed(restore):
            if own:
                setattr(obj, attr, value)
            else:
                delattr(obj, attr)
        for name, recorder in recorders.items():
            result[name].extend(recorder.days[d] for d in sorted(recorder.days))
    if errors:
        raise RuntimeError(errors[0])
