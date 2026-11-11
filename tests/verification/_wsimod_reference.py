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
    "NodeCall",
    "WsimodReference",
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
    """
    from wsimod.core import constants

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
        surface={
            name: _tensor(surfaces, key, n_days, n_surfaces) for name, key in _SURFACE_FIELDS
        },
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
    return NODE_CLASSES[type(node).__qualname__].store_paths


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
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@contextmanager
def capture_node_calls(model, nodes: Sequence[str]) -> Iterator[dict[str, list[NodeCall]]]:
    """Record every call into the named nodes while ``model`` runs (spec, Verification 1).

    Wraps each target's orchestration methods (the slots that name it, and ``distribute`` when
    it is in the river order), its four handler entry points, and the ``send_*`` methods of the
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
                    vqip=copy.deepcopy(
                        bound.arguments.get("vqip", bound.arguments.get("vqip_"))
                    ),
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
