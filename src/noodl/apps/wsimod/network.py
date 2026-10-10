"""WSIMOD-style rule-based allocation: a topology of stores and capacity-limited arcs as a `Model`.

A thin builder over `AllocatedFlowLayer`, in the `(model, state, drivers)` form the
sewer, water and street applications return. The topology is the plain structure WSIMOD
networks reduce to -- the one `tests/data/wsimod/*_topology.json` holds:

    {"nodes": [{"name": ...}, ...],
     "arcs":  [{"name": ..., "source": ..., "target": ..., "capacity": ...}, ...]}

Node storage ceilings default to unbounded (`inf`), as for WSIMOD's own non-storage nodes;
pass `storage_max={node: m3}` for the stores that have one. Requests (m3/s, per arc, in
arc order) are the driver `"<layer>.requests"`; storage (m3, full node order) is the state
`"<layer>.s"`. Build them by name through `model.refs`:

    model, state, drivers = build_model(topology)
    ref = model.refs.wsimod
    drivers = model.drivers_from({ref.requests: {"baseflow": 0.3}}, base=drivers)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch

from noodl.layers.allocation import AllocatedFlowLayer
from noodl.model import Drivers, Model, State
from noodl.refs import initial_drivers as refs_initial_drivers
from noodl.refs import state_from
from noodl.topology import Network

F64 = torch.float64


def build_model(
    topology: Mapping | str | PathLike, *, storage_max: Mapping[str, float] | None = None,
    kind: str = "link", layer_name: str = "wsimod", mode: str = "hard",
    tau: float | None = None, preference: Mapping[str, float] | None = None,
    initial_storage: Mapping[str, float] | None = None,
) -> tuple[Model, State, Drivers]:
    """`(model, state, drivers)` for a WSIMOD topology (a mapping, or a JSON file path).

    `storage_max`, `preference` and `initial_storage` are keyed by node or arc NAME and
    default to unbounded, equal and empty respectively; an unknown name is refused. The
    state is the storage `"<layer>.s"`; the drivers hold zero requests `"<layer>.requests"`.
    """
    if not isinstance(topology, Mapping):
        topology = json.loads(Path(topology).read_text())
    nodes = [n["name"] for n in topology["nodes"]]
    arcs = list(topology["arcs"])
    net = Network(dtype=F64)
    for name in nodes:
        net.add_node(name)
    for arc in arcs:
        net.add_edge(arc["source"], arc["target"], kind=kind, name=arc["name"])
    arc_names = [a["name"] for a in arcs]

    def by_name(values, names, default, what):
        values = dict(values or {})
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise KeyError(f"apps.wsimod.build_model: unknown {what} {unknown}")
        return torch.tensor([float(values.get(n, default)) for n in names], dtype=F64)

    s_max = by_name(storage_max, nodes, float("inf"), "node(s) in storage_max")
    c_arc = torch.tensor(
        [float(a.get("capacity", float("inf"))) for a in arcs], dtype=F64
    )
    pref = None if preference is None else by_name(preference, arc_names, 1.0, "arc(s)")
    layer = AllocatedFlowLayer(
        net, layer_name, kind, s_max=s_max, c_arc=c_arc, preference=pref, mode=mode, tau=tau,
        quantity="storage", unit="m3", flow_unit="m3/s",
    )
    model = Model(net, [layer])
    state: State = {
        f"{layer_name}.s": by_name(initial_storage, nodes, 0.0, "node(s) in initial_storage")
    }
    drivers: Drivers = {f"{layer_name}.requests": torch.zeros(len(arcs), dtype=F64)}
    model.driver_template = dict(drivers)
    return model, state, drivers


def initial_state(model: Model, *, values: Mapping | None = None) -> State:
    """Empty stores, `"<layer>.s" = 0` for every allocation layer, then `values`.

    `values` names what is not zero, `{layer or key: {node: m3}}` (see `noodl.refs.state_from`):
    `initial_state(model, values={"wsimod": {"my_groundwater": 5.0}})`.
    """
    state: State = {
        f"{name}.s": torch.zeros(model.net.n, dtype=model.net.dtype)
        for name in model.allocation
    }
    if values:
        state = state_from(model, values, base=state)
    return state


def initial_drivers(model: Model, *, values: Mapping | None = None) -> Drivers:
    """Zero requests (`build_model`'s driver template), then `values` by arc name:
    `initial_drivers(model, values={"wsimod.requests": {"baseflow": 0.3}})`."""
    return refs_initial_drivers(model, values=values)
