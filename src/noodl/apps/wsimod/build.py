"""A WSIMOD configuration as a noodl ``Model`` (spec, Entry points): :func:`build_model`,
:func:`initial_state` and :func:`initial_drivers`.

The model is a :class:`~noodl.topology.Network` with one node per WSIMOD node and one edge of
kind ``"arc"`` per arc, named as in the configuration, and no layers: the day is one
integrating closure, :class:`~noodl.apps.wsimod.day.WsimodDay`, which owns every
``"wsimod.*"`` state key (D17). ``model.step(state, drivers, dt=86400, t=k * 86400)`` runs day
``k``, so the model couples with other noodl models::

    network = read_wsimod_config("tests/data/wsimod/small")
    model, state, drivers = build_model(network)
    for k in range(len(network.dates)):
        state = model.step(state, drivers, dt=86400.0, t=k * 86400.0)
"""

from __future__ import annotations

import difflib
from collections.abc import Hashable, Mapping

import torch
from torch import Tensor

from noodl.apps.wsimod.config import WsimodNetwork
from noodl.apps.wsimod.day import _FLAT_KEYS, STATE_KEYS, WsimodDay
from noodl.model import Drivers, Model, State
from noodl.topology import Network

__all__ = ["build_model", "initial_drivers", "initial_state"]

F64 = torch.float64


def build_model(network: WsimodNetwork, *, pollutants: bool = True) -> tuple[Model, State, Drivers]:
    """``(model, state, drivers)`` for a WSIMOD configuration (``read_wsimod_config``).

    Every build-time refusal (D7) is raised here, by name (``check_network``): what depends on
    the parameters or on ``pollutants`` (neutral processes, ``decays``, porosity, leakage,
    preferences, travel times, check-recursion cycles, the river order, the orchestration,
    missing forcing). ``pollutants=False`` runs on the volumes alone; what that run ignores is
    listed in ``model.ignored_by_quantity_only``; what WSIMOD accepts and never reads (a
    ``Groundwater``'s forcing, a ``filename`` on a class that reads none) is ``model.unread``.
    The closure runs one tracing day (D2) that fixes the call structure. ``model.wsimod_day`` is
    the closure and ``model.driver_template`` the configured drivers; ``state`` and ``drivers``
    are :func:`initial_state` and
    :func:`initial_drivers` without values.
    """
    day = WsimodDay(network, pollutants=pollutants)
    net = Network(dtype=F64)
    for spec in network.nodes:
        net.add_node(spec.name)
    for arc in network.arcs:
        net.add_edge(arc.in_port, arc.out_port, kind="arc", name=arc.name)
    model = Model(net, [], closures=[day])
    model.wsimod_day = day
    model.ignored_by_quantity_only = list(day.ignored_by_quantity_only)
    model.unread = list(network.unread)  # accepted and never read, D10
    model.driver_template = day.driver_template()
    return model, initial_state(model), initial_drivers(model)


def _day(model: Model) -> WsimodDay:
    day = getattr(model, "wsimod_day", None)
    if not isinstance(day, WsimodDay):
        raise TypeError("not a WSIMOD model: build it with noodl.apps.wsimod.build_model")
    return day


def _unknown(what: str, value: Hashable, known) -> KeyError:
    known = list(known)
    names = [str(k) for k in known]
    hint = difflib.get_close_matches(str(value), names, n=3, cutoff=0.6)
    hint = [repr(known[names.index(h)]) for h in hint]
    return KeyError(
        f"{what} {value!r} is not known"
        + (f"; did you mean {' or '.join(hint)}?" if hint else "")
        + f" (known: {known[:12]}{' ...' if len(known) > 12 else ''})"
    )


def _row(labels: list, label: Hashable, key: str) -> int:
    if isinstance(label, list):
        label = tuple(label)
    if label in labels:
        return labels.index(label)
    raise _unknown(f"{key!r}: label", label, labels)


def _tensor(value) -> Tensor:
    return torch.as_tensor(value, dtype=F64)


def _vqip(day: WsimodDay, value, key: str, label) -> Tensor:
    """A number (or batch tensor) is a volume with zero pollutants; a mapping a VQIP by name
    (``tanks.py:36-50``)."""
    p = day.p
    if isinstance(value, Mapping):
        unknown = [k for k in value if k != "volume" and k not in p.names]
        if unknown:
            raise _unknown(f"{key!r} row {label!r}: pollutant", unknown[0], ["volume", *p.names])
        columns = [_tensor(value.get(name, 0.0)) for name in ("volume", *p.names)]
    else:
        volume = _tensor(value)
        columns = [volume, *[torch.zeros_like(volume)] * p.K]
    return torch.stack(torch.broadcast_tensors(*columns), dim=-1)


def _assemble(base: Tensor, rows: dict[int, Tensor], flat: bool) -> Tensor:
    """``base`` with the given rows replaced, assembled with ``stack`` so a value that requires
    grad keeps its path."""
    if not rows:
        return base
    axis = -1 if flat else -2
    parts = list(base.unbind(axis))
    for i, value in rows.items():
        parts[i] = value
    return torch.stack(torch.broadcast_tensors(*parts), dim=axis)


def initial_state(model: Model, *, values: Mapping | None = None) -> State:
    """The configuration's state (``WsimodDay.initial_state``), then ``values`` by key and row
    label (D10): ``{"wsimod.storage": {("river", "tank"): 9.0}}``.

    A VQIP key takes a number (or batch tensor) as a volume with zero pollutants and a mapping
    as a VQIP by name, ``{("gw", "tank"): {"volume": 5.0, "phosphate": 0.1}}``. Setting a
    ``QueueTank``'s ``"wsimod.storage"`` row sets its ``"wsimod.active_storage"`` row too, as
    WSIMOD starts it (``tanks.py:478``), unless that row is given as well. A whole key may be
    given as a tensor. An unknown key or label is a ``KeyError`` with a suggestion.
    """
    day = _day(model)
    state = day.initial_state()
    values = dict(values or {})
    for key in values:
        if key not in STATE_KEYS:
            raise _unknown("state key", key, STATE_KEYS)
    given: dict[str, dict[int, Tensor]] = {}
    for key, value in values.items():
        labels = day.state_labels(key)
        if not isinstance(value, Mapping):
            tensor = _tensor(value)
            state[key] = tensor
            continue
        rows: dict[int, Tensor] = {}
        for label, v in value.items():
            i = _row(labels, label, key)
            rows[i] = _tensor(v) if key in _FLAT_KEYS else _vqip(day, v, key, label)
        given[key] = rows
    active_labels = day.state_labels("wsimod.active_storage")
    storage_labels = day.state_labels("wsimod.storage")
    follow = given.setdefault("wsimod.active_storage", {})
    for i, row in list(given.get("wsimod.storage", {}).items()):
        label = storage_labels[i]
        if label in active_labels and active_labels.index(label) not in follow:
            follow[active_labels.index(label)] = row
    for key, rows in given.items():
        state[key] = _assemble(state[key], rows, key in _FLAT_KEYS)
    return state


def initial_drivers(model: Model, *, values: Mapping | None = None) -> Drivers:
    """The configured drivers (``model.driver_template``, D6), then ``values`` by key and row
    label.

    Row labels are node names (``"wsimod.<Class>.<parameter>"``), ``(node, surface)`` pairs
    (``"wsimod.<SurfaceClass>.<parameter>"``), arc names (``"wsimod.Arc.capacity"``,
    ``"wsimod.Arc.preference"``) and the reading nodes for the forcing series
    (``"wsimod.precipitation"``, ``"wsimod.et0"``, ``"wsimod.temperature"``,
    ``"wsimod.flow"``, ``"wsimod.concentration"``). A value is a number or batch tensor; a
    pollutant-keyed parameter takes ``{pollutant: value}`` (absent ones keep theirs) or a
    tensor ``(..., K)``, a time-area one ``{time: fraction}`` or ``(..., L+1)``; a forcing
    series a number for every date or a tensor ``(..., T)`` (``(..., T, K)`` for the
    concentrations). A whole key may be given as a tensor. An unknown key or label is a
    ``KeyError`` with a suggestion:
    ``initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlet": 0.3}})``.
    """
    day = _day(model)
    template = getattr(model, "driver_template", None) or day.driver_template()
    drivers = {k: v.clone() for k, v in template.items()}
    for key, value in dict(values or {}).items():
        layout = day.drivers.get(key)
        if layout is None:
            raise _unknown("driver key", key, list(day.drivers))
        if not isinstance(value, Mapping):
            drivers[key] = _tensor(value)
            continue
        base = drivers[key]
        # the axis the labels run along: rows, after the dates for a forcing series
        columns = layout.columns
        for label, v in value.items():
            i = _row(layout.labels, label, key)
            if layout.forcing:
                old = base[..., :, i] if columns is None else base[..., :, i, :]
                if isinstance(v, Mapping):
                    new = _columns(old, v, columns, key, label, axis=-1)
                else:
                    new = _tensor(v)
                    new = new[..., None] if columns is None and new.ndim == 0 else new
                    new = torch.broadcast_to(new, torch.broadcast_shapes(new.shape, old.shape))
                parts = list(base.unbind(-1 if columns is None else -2))
                parts[i] = new
                base = torch.stack(
                    torch.broadcast_tensors(*parts), dim=-1 if columns is None else -2
                )
            else:
                old = base[..., i] if columns is None else base[..., i, :]
                if isinstance(v, Mapping):
                    new = _columns(old, v, columns, key, label, axis=-1)
                else:
                    new = _tensor(v)
                    if columns is not None and new.ndim == 0:
                        raise ValueError(
                            f"{key!r} row {label!r} has a column per {list(columns)}; give a "
                            "mapping or a tensor (..., C)"
                        )
                base = _assemble(base, {i: new}, columns is None)
        drivers[key] = base
    return drivers


def _columns(old: Tensor, value: Mapping, columns, key: str, label, *, axis: int) -> Tensor:
    """``old`` (..., C) with the named columns replaced."""
    if columns is None:
        raise ValueError(f"{key!r} row {label!r} takes one value, not a mapping")
    parts = list(old.unbind(axis))
    for name, v in value.items():
        if name not in columns:
            raise _unknown(f"{key!r} row {label!r}: column", name, columns)
        parts[list(columns).index(name)] = _tensor(v)
    return torch.stack(torch.broadcast_tensors(*parts), dim=axis)
