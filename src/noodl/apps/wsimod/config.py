"""Reads WSIMOD's own model description (``config.yml``, ``Model.load``) into dataclasses.

The format is the one ``Model.load`` reads (``wsimod/orchestration/model.py:216-274``) with the
per-node CSV and CSV.GZ forcing files (``_load_individual_files``, ``:324-348``; ``read_csv``,
``:1252-1269``). A parsed mapping in the same structure is accepted too, with
``data_input_dict`` given inline as ``{(variable, date): value}``. Nothing of WSIMOD is
imported. The reader checks keys, classes and formats; it computes no physics (derived
quantities belong to the node classes), and it refuses what WSIMOD would silently ignore or
fail on later, by name.
"""

from __future__ import annotations

import copy
import csv
import gzip
import numbers
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from noodl.apps.wsimod.classes import (
    ARC_SCHEMAS,
    NODE_SCHEMAS,
    OUT_OF_SCOPE,
    SURFACE_SCHEMAS,
    ClassSchema,
)
from noodl.apps.wsimod.constants import (
    ADDITIVE_POLLUTANTS,
    FLOAT_ACCURACY,
    NON_ADDITIVE_POLLUTANTS,
    POLLUTANTS,
    UNBOUNDED_CAPACITY,
)
from noodl.apps.wsimod.decay import DecaySpec, read_decays
from noodl.apps.wsimod.vqip import Pollutants

__all__ = [
    "ArcSpec",
    "DEFAULT_ORCHESTRATION",
    "NodeSpec",
    "SurfaceSpec",
    "WsimodNetwork",
    "parse_date",
    "read_wsimod_config",
]

# model.py:191-205, ``Model.__init__``
DEFAULT_ORCHESTRATION: tuple[tuple[str, str], ...] = (
    ("FWTW", "treat_water"),
    ("Demand", "create_demand"),
    ("Land", "run"),
    ("Groundwater", "infiltrate"),
    ("Sewer", "make_discharge"),
    ("Foul", "make_discharge"),
    ("WWTW", "calculate_discharge"),
    ("Groundwater", "distribute"),
    ("River", "calculate_discharge"),
    ("Reservoir", "make_abstractions"),
    ("Land", "apply_irrigation"),
    ("WWTW", "make_discharge"),
    ("Catchment", "route"),
)

_TOP_LEVEL = {
    "nodes",
    "arcs",
    "overrides",
    "orchestration",
    "pollutants",
    "additive_pollutants",
    "non_additive_pollutants",
    "float_accuracy",
    "dates",
    "river_discharge_order",  # ignored: ``save`` writes it, ``load`` recomputes it
    "extensions",  # refused when non-empty
    "unified_data_file",  # refused when set
}
# Constructor keywords the reader handles itself: they are not NodeSpec/SurfaceSpec parameters.
_NOT_PARAMETERS = {"name", "data_input_dict", "surfaces", "parent", "surface"}

Data = dict[tuple[str, datetime], float]


@dataclass
class SurfaceSpec:
    """One surface of a ``Land`` node: its class, name, constructor parameters and forcing."""

    name: str
    cls: str
    parameters: dict[str, object]
    data: Data = field(default_factory=dict)


@dataclass
class NodeSpec:
    """One node. ``cls`` is ``node_type_override`` or ``type_``; ``slot_key`` is the key of
    ``Model.nodes_type`` the node is filed under (``model.py:650-673``)."""

    name: str
    cls: str
    slot_key: str
    parameters: dict[str, object]  # schema defaults, then the config, then the overrides
    data: Data = field(default_factory=dict)
    surfaces: list[SurfaceSpec] = field(default_factory=list)


@dataclass
class ArcSpec:
    name: str
    cls: str
    in_port: str
    out_port: str
    capacity: float = UNBOUNDED_CAPACITY
    preference: float = 1.0
    number_of_timesteps: int = 0
    # a DecayArc's or DecayArcAlt's ``decays`` (``arcs.py:671``, ``:734``; default ``{}``),
    # ``None`` on every other class
    decays: DecaySpec | None = None


# the arc classes that take ``decays`` (arcs.py:668-805)
_DECAY_ARCS = ("DecayArc", "DecayArcAlt")


@dataclass
class WsimodNetwork:
    """A WSIMOD model description: pollutants, orchestration, dates, nodes and arcs, in the
    configuration's order (a node's in- and out-arc order is the arcs' order)."""

    pollutants: Pollutants
    float_accuracy: float
    orchestration: list[tuple[str, str]]
    dates: list[datetime]
    date_labels: list[str]
    nodes: list[NodeSpec]
    arcs: list[ArcSpec]
    # what WSIMOD accepts and never reads (D10), as "<node>.<key>"
    unread: list[str] = field(default_factory=list)


def parse_date(text: str) -> datetime:
    """A date as ``to_datetime`` parses it (``model.py:139-162``): the formats
    ``%Y-%m-%d %H:%M:%S``, ``%Y-%m-%d`` and ``%Y-%m`` (the first of the month), in that order."""
    if not isinstance(text, str):
        raise ValueError(f"date {text!r} is not a string; write it in quotes, e.g. '2000-01-31'")
    for form in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(text, form)
        except ValueError:
            pass
    raise ValueError(
        f"Unable to parse date string {text!r}. Expected formats: 'YYYY-MM-DD', 'YYYY-MM' or "
        "'YYYY-MM-DD HH:MM:SS'"
    )


# --- forcing ---------------------------------------------------------------------------------
def _read_csv(path: Path) -> Data:
    """``read_csv`` (``model.py:1252-1269``): rows ``variable, time, value``; ``.gz`` is gzipped."""
    opener = gzip.open if str(path).endswith(".gz") else open
    data: Data = {}
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data[(row["variable"], parse_date(row["time"]))] = float(row["value"])
    return data


def _inline_data(table: object, where: str) -> Data:
    """An inline ``{(variable, date): value}``; dates are normalised (``"2000-01"`` is the
    first of January 2000, as ``to_datetime.__eq__``, ``model.py:164``, takes them)."""
    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise ValueError(f"{where}: data_input_dict must be a mapping of (variable, date) keys")
    data: Data = {}
    for key, value in table.items():
        if not (isinstance(key, (tuple, list)) and len(key) == 2 and isinstance(key[0], str)):
            raise ValueError(f"{where}: data_input_dict key {key!r} is not (variable, date)")
        when = key[1] if isinstance(key[1], datetime) else parse_date(key[1])
        data[(key[0], when)] = float(value)
    return data


def _forcing(entry: dict, base: Path | None, where: str) -> Data:
    """The forcing of a node or surface: ``filename`` (relative to the configuration's
    directory, ``model.py:326-346``) or an inline ``data_input_dict``; ``filename`` wins."""
    if "filename" in entry:
        name = entry.pop("filename")
        if not isinstance(name, str):
            raise ValueError(f"{where}: filename must be a string")
        path = Path(name) if base is None else base / name
        if not path.is_file():
            raise ValueError(f"{where}: forcing file {str(path)!r} does not exist")
        entry.pop("data_input_dict", None)
        return _read_csv(path)
    return _inline_data(entry.pop("data_input_dict", None), where)


# --- helpers ---------------------------------------------------------------------------------
def _items(section: object, what: str) -> list[dict]:
    """A ``nodes``/``arcs``/``overrides`` section as a list of copied dicts: WSIMOD takes a
    mapping and uses its values (``model.py:257-258``); a list is accepted as well."""
    if section is None:
        return []
    values = list(section.values()) if isinstance(section, Mapping) else section
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{what} must be a mapping or a list of mappings")
    out = []
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError(f"every entry of {what} must be a mapping, not {item!r}")
        out.append(copy.deepcopy(dict(item)))
    return out


def _need(entry: dict, key: str, where: str) -> object:
    if key not in entry:
        raise ValueError(f"{where} has no {key!r}")
    return entry.pop(key)


def _refuse_class(cls: str, kind: str, known: Mapping[str, object]) -> None:
    if cls in known:
        return
    if cls in OUT_OF_SCOPE:
        reason = OUT_OF_SCOPE[cls]
        why = reason if reason.startswith("refused") else f"belongs to sub-project {reason}"
        raise ValueError(f"{kind} class {cls!r} is not supported: {why}")
    raise ValueError(f"{kind} type {cls!r} not recognised")


def _checked_parameters(
    entry: dict, schema: ClassSchema, where: str, *, drop: frozenset[str] = frozenset()
) -> dict[str, object]:
    """The constructor parameters: the schema's defaults (copied), then the entry's keys. A key
    the constructor chain does not accept is refused (WSIMOD: ``TypeError``)."""
    accepted = {k for k in schema.parameters if k not in schema.derived and k not in drop}
    for key in entry:
        if key not in accepted:
            raise ValueError(
                f"{where}: {key!r} is not a constructor keyword of {schema.name}"
                + (" (the constructor computes it itself)" if key in schema.derived else "")
            )
    parameters = {
        key: copy.deepcopy(default)
        for key, default in schema.parameters.items()
        if key not in _NOT_PARAMETERS and key not in schema.derived
    }
    parameters.update(copy.deepcopy(entry))
    return parameters


def _need_real(value: object, what: str) -> None:
    """A WTW constant, exponent or liquor multiplier is multiplied (``wtw.py:191-201``): refuse
    a string, ``None`` or a boolean by name, as ``read_decays`` does."""
    if isinstance(value, (str, bool)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} {value!r} is not a number (WSIMOD multiplies it, wtw.py:191-201)")


def _refuse_partial_wtw(where: str, parameters: dict, pollutants: Pollutants) -> None:
    """A configured ``process_parameters`` or ``liquor_multiplier`` must name every additive
    pollutant, ``process_parameters`` with its ``constant`` and ``exponent``: WSIMOD reads them
    all (``wtw.py:188-201``, ``KeyError`` otherwise). An empty one is filled by
    :func:`_fill_wtw_defaults`. Overrides may be partial (they merge, X36)."""
    process = parameters.get("process_parameters") or {}
    liquor = parameters.get("liquor_multiplier") or {}
    for name in pollutants.additive if process else ():
        entry = process.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{where}: process_parameters has no entry for the additive pollutant {name!r} "
                "(WTW treatment reads one per additive pollutant, wtw.py:188-199)"
            )
        for key in ("constant", "exponent"):
            if key not in entry:
                raise ValueError(
                    f"{where}: process_parameters[{name!r}] has no {key!r} (wtw.py:191, :197)"
                )
            _need_real(entry[key], f"{where}: process_parameters[{name!r}] {key}")
    for name, value in liquor.items():
        _need_real(value, f"{where}: liquor_multiplier[{name!r}]")
    for name in pollutants.additive if liquor else ():
        if name not in liquor:
            raise ValueError(
                f"{where}: liquor_multiplier has no entry for the additive pollutant {name!r} "
                "(WTW treatment reads one per additive pollutant, wtw.py:201)"
            )


def _fill_wtw_defaults(parameters: dict, pollutants: Pollutants) -> None:
    """``WTW.__init__`` fills an empty ``process_parameters`` and ``liquor_multiplier`` from the
    model's additive pollutants (``wtw.py:73-85``), a fresh dict per node."""
    if not parameters.get("process_parameters"):
        parameters["process_parameters"] = {
            x: {"constant": 0.01, "exponent": 1.001} for x in pollutants.additive
        }
    if not parameters.get("liquor_multiplier"):
        liquor = {x: 0.7 for x in pollutants.additive}
        liquor["volume"] = 0.03
        parameters["liquor_multiplier"] = liquor


# the classes whose forcing WSIMOD reads (land.py:451-460, demand.py:321, catchment.py:59-74)
_FORCING_CLASSES = ("Land", "ResidentialDemand", "Catchment")
# the storage classes whose DecayTank reads a temperature (core.py:484-487)
_TEMPERATURE_STORAGE = (
    "Storage",
    "Groundwater",
    "QueueGroundwater",
    "Reservoir",
    "RiverReservoir",
    "River",
)


# --- overrides -------------------------------------------------------------------------------
def _apply_overrides(
    parameters: dict[str, object],
    schema: ClassSchema,
    overrides: dict,
    where: str,
    *,
    surface: bool = False,
) -> None:
    """Merge one override entry into ``parameters`` (D5): dict-valued ``pollutant_load``,
    ``liquor_multiplier`` and ``process_parameters`` update the node's own copy, everything
    else is replaced. A key outside the class's chain is refused (WSIMOD only prints)."""
    for key in overrides:
        # a node's ``filename`` is read by _override_node (X27); a surface's forcing is deposition
        # data (3c), and no override of ``data_input_dict`` is ported
        if key == "data_input_dict" or (key == "filename" and surface):
            raise ValueError(
                f"{where}: an override of {key!r} is not supported (surface and "
                "data_input_dict forcing overrides are sub-project 3c, land.py:413-421)"
            )
        if key not in schema.override_keys:
            raise ValueError(f"{where}: {key!r} is not an override of {schema.name}")
    for key, value in overrides.items():
        # The PerviousSurface soil-temperature keys (soil_temp_w_prev, _w_air, _w_deep, _deep) are
        # state attributes in WSIMOD, not constructor keywords (land.py:753-756); they are
        # stored in the surface's parameters here for the Land task to read from there.
        if key in ("surfaces", "surface", "filename"):  # filename: _override_node
            continue
        if key in ("pollutant_load", "liquor_multiplier"):
            if not isinstance(value, Mapping):
                raise ValueError(f"{where}: {key} must be a mapping")
            parameters[key] = {**parameters[key], **copy.deepcopy(dict(value))}
        elif key == "decays":  # dict.update per pollutant (storage.py:101, tanks.py:439)
            if not isinstance(value, Mapping):
                raise ValueError(f"{where}: decays must be a mapping")
            if parameters.get("decays") is None:  # storage.py:96-100
                raise ValueError(
                    f"{where}: cannot override decays on a node initialised without decays"
                )
            parameters[key] = {**parameters[key], **copy.deepcopy(dict(value))}
        elif key == "process_parameters":
            if not isinstance(value, Mapping):
                raise ValueError(f"{where}: process_parameters must be a mapping")
            merged = {k: dict(v) for k, v in parameters[key].items()}
            for pollutant, entry in value.items():
                if pollutant not in merged and pollutant != "volume":
                    raise ValueError(f"{where}: process_parameters has no {pollutant!r}")
                merged.setdefault(pollutant, {}).update(copy.deepcopy(dict(entry)))
            parameters[key] = merged
        else:
            parameters[key] = copy.deepcopy(value)


def _override_filename(
    node: NodeSpec, value: object, where: str, unread: list[str], pollutants: Pollutants
) -> None:
    """``Node.apply_overrides`` (``nodes.py:93-99``, D23): a string is read with ``read_csv`` as
    written, relative to the working directory (not the configuration's), and replaces the
    node's forcing; a falsy non-string (``None``, ``0``, ``[]``) is ignored; anything else is
    refused. The empty string is a string, so WSIMOD opens ``""`` and raises
    ``FileNotFoundError`` (``nodes.py:95-96`` test ``isinstance`` first): refused here too."""
    if isinstance(value, str):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{where}: forcing file {value!r} does not exist")
        node.data = _read_csv(path)
        # a class that reads no forcing still has its file read (D10); a Groundwater's file is
        # read by its decay only
        reads = (
            node.cls in _FORCING_CLASSES
            or (node.cls in _TEMPERATURE_STORAGE and node.parameters.get("decays") is not None)
            or (node.cls == "River" and "nitrate" in pollutants.names)
        )  # storage.py:750
        if not reads and f"{node.name}.filename" not in unread:
            unread.append(f"{node.name}.filename")
    elif not value:
        pass  # nodes.py:97-98: falsy non-string content is ignored
    else:
        raise ValueError(
            f"{where}: filename {value!r} is not a string (WSIMOD raises RuntimeError: "
            "Not recognised format for data_input_dict)"
        )


def _override_node(
    node: NodeSpec, entry: dict, where: str, unread: list[str], pollutants: Pollutants
) -> None:
    _apply_overrides(node.parameters, NODE_SCHEMAS[node.cls], entry, where)
    if "process_parameters" in node.parameters:  # the merged entries, checked as at reading
        _refuse_partial_wtw(where, node.parameters, pollutants)
    # storage.py:540, :620-644: a River's bulk_density is set and never read (D10)
    if node.cls == "River" and "bulk_density" in entry:
        if f"{node.name}.bulk_density" not in unread:
            unread.append(f"{node.name}.bulk_density")
    if "filename" in entry:
        _override_filename(node, entry["filename"], where, unread, pollutants)
    surfaces = entry.get("surfaces", {})
    if not isinstance(surfaces, Mapping):
        raise ValueError(f"{where}: surfaces must be a mapping of surface name to overrides")
    for surface_name, override in surfaces.items():  # land.py:187-188
        match = next((s for s in node.surfaces if s.name == surface_name), None)
        if match is None:
            raise ValueError(f"{where}: surface {surface_name!r} not recognised")
        override = copy.deepcopy(dict(override))
        _apply_overrides(
            match.parameters,
            SURFACE_SCHEMAS[match.cls],
            override,
            f"{where}, surface {surface_name!r}",
            surface=True,
        )
        match.name = override.get("surface", match.name)  # land.py:396


# --- the reader ------------------------------------------------------------------------------
def _orchestration(section: object) -> list[tuple[str, str]]:
    if section is None:
        return list(DEFAULT_ORCHESTRATION)
    if not isinstance(section, (list, tuple)):
        raise ValueError("orchestration must be a list of {slot: method} entries")
    out: list[tuple[str, str]] = []
    for entry in section:
        if isinstance(entry, Mapping) and len(entry) == 1:
            ((slot, method),) = entry.items()
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            slot, method = entry
        else:
            raise ValueError(f"orchestration entry {entry!r} is not a single {{slot: method}}")
        if not isinstance(slot, str) or not isinstance(method, str):
            raise ValueError(f"orchestration entry {entry!r} must name a slot and a method")
        out.append((slot, method))
    return out


def _lists(data: dict) -> tuple[Pollutants, float]:
    """The pollutant lists and ``float_accuracy`` (``model.py:232-241``), WSIMOD's defaults when
    absent. Inconsistent lists are refused by name (``Pollutants``)."""
    names = tuple(data.get("pollutants", POLLUTANTS))
    additive = tuple(data.get("additive_pollutants", ADDITIVE_POLLUTANTS))
    non_additive = tuple(data.get("non_additive_pollutants", NON_ADDITIVE_POLLUTANTS))
    return Pollutants(names, additive, non_additive), float(
        data.get("float_accuracy", FLOAT_ACCURACY)
    )


def _read_node(
    entry: dict, base: Path | None, pollutants: Pollutants, unread: list[str]
) -> NodeSpec:
    name = _need(entry, "name", "a node")
    where = f"node {name!r}"
    if not isinstance(name, str):
        raise ValueError(f"{where}: name must be a string")
    type_ = _need(entry, "type_", where)
    cls = entry.pop("node_type_override", type_)  # model.py:653-657
    _refuse_class(cls, "node", NODE_SCHEMAS)
    schema = NODE_SCHEMAS[cls]
    slot_key = "Foul" if "foul" in name else type_  # model.py:658-660
    entry.pop("geometry", None)  # model.py:661-662
    if ("filename" in entry or "data_input_dict" in entry) and (
        "data_input_dict" not in schema.parameters
    ):
        raise ValueError(f"{where}: {cls} does not take forcing data (data_input_dict)")
    lost = [k for k in ("data_input_dict", "filename") if k in entry]
    data = _forcing(entry, base, where)
    if cls in ("Groundwater", "QueueGroundwater"):
        # storage.py:202-203, :291-292: the configured forcing is stored before Node.__init__
        # resets it (nodes.py:26, :58), so it is lost (X11, D10)
        data = {}
        unread.extend(f"{name}.{k}" for k in lost)
    surfaces: list[SurfaceSpec] = []
    if "surfaces" in entry:
        if "surfaces" not in schema.parameters:
            raise ValueError(f"{where}: {cls} has no surfaces")
        surfaces = [
            _read_surface(s, base, where, pollutants)
            for s in _items(entry.pop("surfaces"), f"{where} surfaces")
        ]
    parameters = _checked_parameters(entry, schema, where)
    decays = parameters.get("decays")
    if decays is not None:  # storage.py:53, :300 build a DecayTank or a DecayQueueTank
        read_decays(where, decays, pollutants)
    if "process_parameters" in parameters:
        _refuse_partial_wtw(where, parameters, pollutants)
        _fill_wtw_defaults(parameters, pollutants)
    return NodeSpec(name, cls, slot_key, parameters, data, surfaces)


def _read_surface(
    entry: dict, base: Path | None, owner: str, pollutants: Pollutants
) -> SurfaceSpec:
    type_ = _need(entry, "type_", f"a surface of {owner}")
    _refuse_class(type_, "surface", SURFACE_SCHEMAS)
    schema = SURFACE_SCHEMAS[type_]
    name = entry.pop("surface", schema.parameters["surface"])
    where = f"{owner}, surface {name!r}"
    data = _forcing(entry, base, where)
    parameters = _checked_parameters(entry, schema, where)
    read_decays(where, parameters["decays"], pollutants)  # land.py:303-581: a DecayTank
    return SurfaceSpec(name, type_, parameters, data)


def _read_arc(entry: dict, node_names: set[str], pollutants: Pollutants) -> ArcSpec:
    """One arc. ``decays`` is read on a ``DecayArc`` or ``DecayArcAlt`` only (default ``{}``,
    ``arcs.py:671``, ``:734``); every other class never reads the key (``Arc.__init__`` stores
    it unread, ``arcs.py:52``), so it is refused. ``parent`` is refused on every arc: a
    ``DecayObj`` reads its temperature from a ``parent`` attribute when there is one
    (``core.py:465-468``), and a configured string would be read as a node and crash."""
    name = _need(entry, "name", "an arc")
    where = f"arc {name!r}"
    type_ = _need(entry, "type_", where)
    _refuse_class(type_, "arc", ARC_SCHEMAS)
    if "parent" in entry:
        raise ValueError(
            f"{where}: 'parent' is refused: WSIMOD would read the decay temperature from it "
            "(core.py:465-468) and fail on a configured value"
        )
    if "decays" in entry and type_ not in _DECAY_ARCS:
        raise ValueError(
            f"{where}: 'decays' is read only by DecayArc and DecayArcAlt, not {type_} "
            "(Arc.__init__ would store it unread, arcs.py:52)"
        )
    in_port, out_port = _need(entry, "in_port", where), _need(entry, "out_port", where)
    for port in (in_port, out_port):
        if port not in node_names:
            raise ValueError(f"{where}: node {port!r} not recognised")
    accepted = set(ARC_SCHEMAS[type_].parameters) - {"name", "in_port", "out_port"}
    for key in entry:
        if key not in accepted:
            raise ValueError(f"{where}: {key!r} is not a keyword of {type_}")
    spec = ArcSpec(name, type_, in_port, out_port)
    if type_ in _DECAY_ARCS:
        decays = entry.get("decays", {})
        if decays is None:  # generic_temperature_decay iterates it (core.py:510)
            raise ValueError(f"{where}: decays must be a mapping, got None")
        spec.decays = read_decays(where, decays, pollutants)
    for key, value in entry.items():
        if key != "decays":
            _set_arc_value(spec, key, value, where)
    return spec


def _set_arc_value(spec: ArcSpec, key: str, value: object, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a number, not {value!r}")
    if key == "number_of_timesteps":
        if int(value) != value or value < 0:
            raise ValueError(f"{where}: number_of_timesteps must be a whole number >= 0")
        value = int(value)
    else:
        value = float(value)
    setattr(spec, key, value)


def read_wsimod_config(
    source: str | os.PathLike | Mapping,
    *,
    config_name: str = "config.yml",
    overrides: Mapping | None = None,
) -> WsimodNetwork:
    """Read a WSIMOD model description.

    ``source`` is a directory holding ``config_name`` (``Model.load(address, config_name)``;
    the forcing files named in it are read relative to that directory), or the parsed mapping
    in the same structure. ``overrides`` replaces top-level keys of the description first,
    as ``Model.load``'s own argument does (``model.py:229-230``). The ``overrides`` section is
    merged into the nodes' and arcs' parameters (spec, Loading); see ``docs/formats/wsimod.md``
    for the keys read, ignored and refused. Every refusal is a ``ValueError`` naming the key.
    """
    if isinstance(source, Mapping):
        data, base = copy.deepcopy(dict(source)), None
    else:
        base = Path(source)
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - the `wsimod` extra supplies it
            raise ImportError(
                "reading a config.yml needs PyYAML (pip install 'noodl-physics[wsimod]')"
            ) from exc
        with open(base / config_name, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{config_name} is not a mapping of WSIMOD configuration keys")
    for key, value in (overrides or {}).items():
        data[key] = copy.deepcopy(value)

    for key in data:
        if key not in _TOP_LEVEL:
            raise ValueError(f"unknown top-level key {key!r} (WSIMOD would ignore it)")
    if data.get("extensions"):
        raise ValueError(
            "'extensions' is not supported: WSIMOD would import and run the named files"
        )
    if data.get("unified_data_file"):
        raise ValueError(
            "'unified_data_file' is not supported: the parquet file needs pandas and pyarrow; "
            "use per-node CSV or CSV.GZ files"
        )
    if "nodes" not in data:
        raise ValueError("No nodes found in the config ('nodes' is required)")

    pollutants, float_accuracy = _lists(data)
    orchestration = _orchestration(data.get("orchestration"))
    labels = data.get("dates", [])
    if not isinstance(labels, (list, tuple)):
        raise ValueError("dates must be a list of date strings")
    dates = [parse_date(x) for x in labels]

    unread: list[str] = []
    nodes = [_read_node(e, base, pollutants, unread) for e in _items(data["nodes"], "nodes")]
    seen: set[str] = set()
    for node in nodes:
        if node.name in seen:
            raise ValueError(f"node {node.name!r} is listed twice")
        seen.add(node.name)
    names = {n.name for n in nodes}
    arcs = [_read_arc(e, names, pollutants) for e in _items(data.get("arcs"), "arcs")]
    seen = set()
    for arc in arcs:
        if arc.name in seen:
            raise ValueError(f"arc {arc.name!r} is listed twice")
        seen.add(arc.name)

    section = data.get("overrides") or {}
    if not isinstance(section, Mapping):
        raise ValueError("overrides must be a mapping with 'nodes' and 'arcs'")
    for key in section:
        if key not in ("nodes", "arcs"):
            raise ValueError(f"unknown key {key!r} in overrides (WSIMOD would ignore it)")
    for entry in _items(section.get("nodes"), "overrides nodes"):
        type_, name = (
            _need(entry, "type_", "a node override"),
            _need(entry, "name", "a node override"),
        )
        slots = {n.slot_key for n in nodes}
        if type_ not in slots:  # model.py:819-820
            raise ValueError(f"override: node type {type_!r} not recognised (use the slot key)")
        node = next((n for n in nodes if n.slot_key == type_ and n.name == name), None)
        if node is None:  # model.py:822-823
            raise ValueError(f"override: node {name!r} not recognised")
        _override_node(node, entry, f"override of node {name!r}", unread, pollutants)
    for entry in _items(section.get("arcs"), "overrides arcs"):
        name = _need(entry, "name", "an arc override")
        entry.pop("type_", None)  # model.py:828-829
        arc = next((a for a in arcs if a.name == name), None)
        if arc is None:  # model.py:831-832
            raise ValueError(f"override: arc {name!r} not recognised")
        for key in entry:
            if key == "decays":
                raise ValueError(
                    f"override of arc {name!r}: decays cannot be overridden (WSIMOD's "
                    "Arc.apply_overrides consumes only capacity and preference, arcs.py:87-96, "
                    "and would store it unread)"
                )
            if key not in ARC_SCHEMAS[arc.cls].override_keys:
                raise ValueError(
                    f"override of arc {name!r}: {key!r} is not an override of {arc.cls}"
                )
        for key, value in entry.items():
            _set_arc_value(arc, key, value, f"override of arc {name!r}")
    # a Decay* arc reads its in_port's forcing (core.py:464-470), so that file is read
    ports = {f"{a.in_port}.filename" for a in arcs if a.cls in _DECAY_ARCS}
    unread[:] = [item for item in unread if item not in ports]

    return WsimodNetwork(
        pollutants=pollutants,
        float_accuracy=float_accuracy,
        orchestration=orchestration,
        dates=dates,
        date_labels=[str(x) for x in labels],
        nodes=nodes,
        arcs=arcs,
        unread=unread,
    )
