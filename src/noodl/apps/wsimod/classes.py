"""WSIMOD 0.8.1's classes as the reader sees them: constructor keywords, defaults, overrides.

Each schema transcribes one class's constructor chain from ``wsimod/nodes/*.py`` and
``wsimod/arcs/arcs.py``: the keywords of its own ``__init__`` and, while an ``__init__`` takes
``**kwargs`` (and so forwards them), those of the class it forwards to. A subclass's default
wins over its base's. Every default is WSIMOD's own and is held by value: a schema is never
mutated and a reader copies what it takes (WSIMOD's mutable default arguments are shared; this
port does not reproduce that, spec, defect table).

``override_keys`` is what ``apply_overrides`` consumes along the whole chain (``nodes.py:80-102``
and every subclass). ``renamed_type`` is the class name WSIMOD gives an instance after the
renames in ``storage.py:294`` (``QueueGroundwater`` -> ``Groundwater``), ``storage.py:1043``
(``RiverReservoir`` -> ``Reservoir``) and ``demand.py:214`` (``ResidentialDemand`` ->
``Demand``); the node's slot key in ``Model.nodes_type`` stays the configured ``type_``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY


class _Required:
    """The default of a constructor keyword that has none (``name``)."""

    def __repr__(self) -> str:
        return "REQUIRED"

    def __copy__(self) -> _Required:
        return self

    def __deepcopy__(self, memo) -> _Required:
        return self


REQUIRED = _Required()


@dataclass(frozen=True)
class ClassSchema:
    """One WSIMOD class: its constructor keywords and defaults, and what its overrides consume."""

    name: str
    parameters: Mapping[str, object]  # constructor keyword -> WSIMOD default (REQUIRED: none)
    override_keys: frozenset[str]  # what apply_overrides consumes, the whole chain
    renamed_type: str  # wsimod_type after the renames
    # Keywords the constructor computes itself and passes up, so a configuration naming one
    # makes WSIMOD raise ``TypeError`` (a duplicate keyword): ``Surface``'s ``capacity``
    # (``land.py:371``), ``ImperviousSurface``'s ``depth`` (``:612``), the ``parent`` Land sets
    # (``land.py:103``).
    derived: frozenset[str] = field(default_factory=frozenset)


# --- constructor chains ----------------------------------------------------------------------
# class -> (its own keywords and defaults, the class its ``**kwargs`` reach or None)
_R = REQUIRED
_CONSTRUCTORS: dict[str, tuple[dict[str, object], str | None]] = {
    # nodes.py
    "Node": ({"name": _R, "data_input_dict": None}, None),
    # catchment.py, waste.py
    "Catchment": ({"name": _R, "data_input_dict": {}}, None),
    "Waste": ({"name": _R}, None),
    # demand.py
    "Demand": (
        {"name": _R, "constant_demand": 0, "pollutant_load": {}, "data_input_dict": {}},
        None,
    ),
    "NonResidentialDemand": ({}, "Demand"),  # no __init__ of its own
    "ResidentialDemand": (
        {
            "population": 1,
            "pollutant_load": {},
            "per_capita": 0.12,
            "gardening_efficiency": 0.42,
            "data_input_dict": {},
            "constant_temp": 30,
            "constant_weighting": 0.2,
        },
        "Demand",
    ),
    # distribution.py
    "Distribution": ({"leakage": 0}, "Node"),
    "UnlimitedDistribution": ({}, "Distribution"),
    # land.py
    "Land": (
        {
            "name": _R,
            "subsurface_residence_time": 5,
            "percolation_residence_time": 50,
            "surface_residence_time": 1,
            "surfaces": [],
            "data_input_dict": {},
        },
        None,
    ),
    # sewer.py
    "Sewer": (
        {
            "name": _R,
            "capacity": 0,
            "pipe_time": 0,
            "pipe_timearea": {0: 1},
            "chamber_area": 1,
            "chamber_floor": 10,
            "data_input_dict": {},
        },
        None,
    ),
    # storage.py
    "Storage": (
        {
            "name": _R,
            "capacity": 0,
            "area": 0,
            "datum": 0,
            "decays": None,
            "initial_storage": 0,
        },
        "Node",
    ),
    "Groundwater": (
        {
            "residence_time": 200,
            "infiltration_threshold": 1,
            "infiltration_pct": 0,
            "data_input_dict": {},
        },
        "Storage",
    ),
    "QueueGroundwater": ({"timearea": {0: 1}, "data_input_dict": {}}, "Storage"),
    "River": (
        {
            "depth": 2,
            "length": 200,
            "width": 20,
            "velocity": 17280.0,  # 0.2 * constants.M_S_TO_M_DT
            "damp": 0.1,
            "mrf": 0,
        },
        "Storage",
    ),
    "Reservoir": ({}, "Storage"),
    "RiverReservoir": ({"environmental_flow": 0}, "Reservoir"),
    # wtw.py
    "WTW": (
        {
            "name": _R,
            "treatment_throughput_capacity": 10,
            "process_parameters": {},
            "liquor_multiplier": {},
            "percent_solids": 0.0002,
        },
        None,
    ),
    "WWTW": (
        {
            "stormwater_storage_capacity": 10,
            "stormwater_storage_area": 1,
            "stormwater_storage_elevation": 10,
        },
        "WTW",
    ),
    "FWTW": (
        {
            "service_reservoir_storage_capacity": 10,
            "service_reservoir_storage_area": 1,
            "service_reservoir_storage_elevation": 10,
            "service_reservoir_initial_storage": 0,
            "data_input_dict": {},
        },
        "WTW",
    ),
    # tanks.py (a Surface is a DecayTank, a Tank)
    "Tank": ({"capacity": 0, "area": 1, "datum": 10, "initial_storage": 0}, None),
    "DecayTank": ({"decays": {}, "parent": None}, "Tank"),
    # land.py
    "Surface": (
        {
            "surface": "",
            "area": 1,
            "depth": 1,
            "data_input_dict": {},
            "pollutant_load": {},
        },
        "DecayTank",
    ),
    "ImperviousSurface": ({"pore_depth": 0, "et0_to_e": 1}, "Surface"),
    "PerviousSurface": (
        {
            "depth": 0.75,
            "total_porosity": 0.4,
            "field_capacity": 0.3,
            "wilting_point": 0.12,
            "infiltration_capacity": 0.5,
            "surface_coefficient": 0.05,
            "percolation_coefficient": 0.75,
            "et0_coefficient": 0.5,
            "ihacres_p": 10,
        },
        "Surface",
    ),
    # arcs.py
    "Arc": (
        {
            "name": "",
            "capacity": UNBOUNDED_CAPACITY,
            "preference": 1,
            "in_port": None,
            "out_port": None,
        },
        None,
    ),
    "QueueArc": ({"number_of_timesteps": 0}, "Arc"),
    "AltQueueArc": ({}, "QueueArc"),
    "PullArc": ({}, "Arc"),
    "PushArc": ({}, "Arc"),
    "SewerArc": ({}, "Arc"),  # no __init__ of its own
    "WeirArc": ({}, "SewerArc"),  # no __init__ of its own
}

# --- override chains -------------------------------------------------------------------------
# class -> the keys its own ``apply_overrides`` pops (the parent chain is the one above)
_OVERRIDES: dict[str, frozenset[str]] = {
    "Node": frozenset(),  # nodes.py:80-102; a ``filename`` is refused by the reader (3a)
    "Catchment": frozenset(),
    "Waste": frozenset(),
    "Demand": frozenset({"constant_demand", "pollutant_load"}),  # demand.py:66-77
    "ResidentialDemand": frozenset(  # demand.py:216-234
        {"gardening_efficiency", "population", "per_capita", "constant_weighting", "constant_temp"}
    ),
    "Distribution": frozenset({"leakage"}),  # distribution.py:142-153
    "Land": frozenset(  # land.py:164-190
        {
            "surface_residence_time",
            "subsurface_residence_time",
            "percolation_residence_time",
            "surfaces",
        }
    ),
    "Sewer": frozenset(  # sewer.py:118-143
        {"capacity", "chamber_area", "chamber_floor", "pipe_time", "pipe_timearea"}
    ),
    # storage.py:78-104: ``decays`` is read there too, but a decaying store is refused (3b)
    "Storage": frozenset({"capacity", "area", "datum"}),
    "Groundwater": frozenset(  # storage.py:205-220
        {"residence_time", "infiltration_threshold", "infiltration_pct"}
    ),
    "QueueGroundwater": frozenset({"timearea"}),  # storage.py:317-328
    # storage.py:610-658; the biochemistry parameters it also sets belong to the refused
    # nitrate River (3b)
    "River": frozenset({"length", "width", "velocity", "damp", "mrf"}),
    "RiverReservoir": frozenset({"environmental_flow"}),  # storage.py:1045-1058
    "WTW": frozenset(  # wtw.py:130-153
        {
            "percent_solids",
            "liquor_multiplier",
            "process_parameters",
            "treatment_throughput_capacity",
        }
    ),
    "WWTW": frozenset(  # wtw.py:299-321
        {"stormwater_storage_capacity", "stormwater_storage_area", "stormwater_storage_elevation"}
    ),
    "FWTW": frozenset(  # wtw.py:546-571
        {
            "service_reservoir_storage_capacity",
            "service_reservoir_storage_area",
            "service_reservoir_storage_elevation",
        }
    ),
    # Surface.apply_overrides (land.py:384-423) pops ``capacity`` and ignores it (a printed
    # warning), so only ``area`` and ``depth`` reach the tank; ``Tank.apply_overrides``
    # (tanks.py:55-68) then takes ``datum``.
    "Tank": frozenset({"datum"}),
    "DecayTank": frozenset(),  # ``decays`` (tanks.py:430-440): refused (3b)
    "Surface": frozenset({"surface", "pollutant_load", "area", "depth"}),
    # land.py:623-643; ``depth`` is popped and ignored (a printed error)
    "ImperviousSurface": frozenset({"et0_to_e", "pore_depth"}),
    "PerviousSurface": frozenset(  # land.py:843-879
        {
            "field_capacity",
            "wilting_point",
            "total_porosity",
            "infiltration_capacity",
            "surface_coefficient",
            "percolation_coefficient",
            "et0_coefficient",
            "ihacres_p",
            "soil_temp_w_prev",
            "soil_temp_w_air",
            "soil_temp_w_deep",
            "soil_temp_deep",
        }
    ),
    "Arc": frozenset({"capacity", "preference"}),  # arcs.py:87-96
}

# The classes whose ``__init__`` is one WSIMOD's own: a renamed instance (see the module text)
_RENAMED = {
    "QueueGroundwater": "Groundwater",
    "RiverReservoir": "Reservoir",
    "ResidentialDemand": "Demand",
}
# land.py:627-632: ``ImperviousSurface.apply_overrides`` pops ``depth`` and ignores it (a printed
# error), though ``Surface`` consumes ``depth``; the override is refused.
_NOT_OVERRIDES = {"ImperviousSurface": frozenset({"depth"})}
_DERIVED = {
    "Surface": frozenset({"capacity", "parent"}),
    "ImperviousSurface": frozenset({"capacity", "parent", "depth"}),
    "PerviousSurface": frozenset({"capacity", "parent"}),
}


def _chain(name: str) -> list[str]:
    classes = [name]
    while _CONSTRUCTORS[classes[-1]][1] is not None:
        classes.append(_CONSTRUCTORS[classes[-1]][1])
    return classes


def _schema(name: str) -> ClassSchema:
    parameters: dict[str, object] = {}
    override_keys: frozenset[str] = frozenset()
    for cls in _chain(name):
        for key, value in _CONSTRUCTORS[cls][0].items():
            parameters.setdefault(key, copy.deepcopy(value))
        override_keys |= _OVERRIDES.get(cls, frozenset())
    override_keys -= _NOT_OVERRIDES.get(name, frozenset())
    return ClassSchema(
        name=name,
        parameters=parameters,
        override_keys=override_keys,
        renamed_type=_RENAMED.get(name, name),
        derived=_DERIVED.get(name, frozenset()),
    )


NODE_SCHEMAS: dict[str, ClassSchema] = {
    name: _schema(name)
    for name in (
        "Node",
        "Catchment",
        "Demand",
        "NonResidentialDemand",
        "ResidentialDemand",
        "Distribution",
        "UnlimitedDistribution",
        "Land",
        "Sewer",
        "Storage",
        "Groundwater",
        "QueueGroundwater",
        "River",
        "Reservoir",
        "RiverReservoir",
        "Waste",
        "WTW",
        "WWTW",
        "FWTW",
    )
}
SURFACE_SCHEMAS: dict[str, ClassSchema] = {
    name: _schema(name) for name in ("Surface", "ImperviousSurface", "PerviousSurface")
}
ARC_SCHEMAS: dict[str, ClassSchema] = {
    name: _schema(name)
    for name in (
        "Arc",
        "QueueArc",
        "AltQueueArc",
        "PullArc",
        "PushArc",
        "SewerArc",
        "WeirArc",
    )
}

# Classes WSIMOD has and this port does not, with where they go (spec, Scope).
OUT_OF_SCOPE: dict[str, str] = {
    "DecayArc": "3b",
    "DecayArcAlt": "3b",
    "GrowingSurface": "3c",
    "IrrigationSurface": "3c",
    "GardenSurface": "3c",
    "VariableAreaSurface": "3c",
    "EnfieldFoulSewer": "refused (W4: undefined storm_exchange, sewer.py:288-291)",
}
