"""JSON schema for the `noodl-modelica/1` intermediate format.

One JSON object per Modelica Buildings Library (MBL) model, written by
`scripts/modelica_export.py` from OpenModelica's own evaluated component tree. Every
parameter in the file has already been evaluated by OpenModelica -- `load` never parses
Modelica source and never evaluates a Modelica expression itself. It only checks the file's
SHAPE: the required top-level keys, the `format` marker, and every component's, connection's
and signal's required fields. It raises `ModelicaImportError` naming the first malformed field.

`load` does not judge whether a component's or signal's CLASS is one this importer supports --
that is `graph.build`'s job (`noodl.apps.building_physics.modelica.graph`), once every
connection has been resolved to a port and a node. This module only supplies the vocabulary
`graph.build` checks classes against: `SUPPORTED` groups the fully qualified MBL/MSL class
names the reader recognises, and `REFUSED` maps a class this importer explicitly declines to
convert to the short reason `graph.build` reports against every offending instance;
`REFUSED_PREFIXES` refuses every class of the weather-data package but `ReaderTMY3`.
`DYNAMIC_COLUMNS` are composite classes no graph sees: `expand.expand` rewrites each into
supported parts first. `HEAT_SENSORS` read a volume's temperature, which may feed a signal
block's input; `CONTROLLERS` are the feedback blocks such a sensor feeds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "noodl-modelica/1"


class ModelicaImportError(ValueError):
    """Raised for malformed `noodl-modelica/1` JSON, or for unsupported model content.

    The message names every offending field, instance or class (unsupported content is
    never silently approximated or dropped). `graph.build` gathers every
    refusal it finds and raises exactly one of these, rather than stopping at the first one.
    """


# --------------------------------------------------------------------------------------
# Supported classes, grouped by role. Every set
# holds fully qualified Modelica class names exactly as OpenModelica's scripting API and this
# reader's hand-written fixtures spell them (verified against MBL v13.0.0 source, commit
# 55abf579598ca81cae0a82f337350375958e6722, and MSL v4.1.0 for the signal blocks).
# --------------------------------------------------------------------------------------

ZONES = frozenset({
    "Buildings.Fluid.MixingVolumes.MixingVolume",
    "Buildings.Fluid.Delays.DelayFirstOrder",
})
# Weather and wind (`weather.py`, `outside.py`). The outside boundaries that read a weather bus
# (`Fluid/Sources/BaseClasses/Outside.mo`), the three of them that add a wind pressure, and the
# one weather reader. `WEATHER_BUS_OUTPUTS` are the weather-bus variables those boundaries read
# (`Outside.mo:7-8`, `Outside_Cp.mo:39-42`, `Outside_CpLowRise.mo:47-51`,
# `Outside_CpData.mo:85-89`, `BaseClasses/Outside.mo:51`): the only reader outputs evaluated.
WEATHER_READERS = frozenset({"Buildings.BoundaryConditions.WeatherData.ReaderTMY3"})
WEATHER_BUS_OUTPUTS = ("TDryBul", "pAtm", "relHum", "winSpe", "winDir")
OUTSIDE_WIND = frozenset({
    "Buildings.Fluid.Sources.Outside_Cp",
    "Buildings.Fluid.Sources.Outside_CpLowRise",
    "Buildings.Fluid.Sources.Outside_CpData",
})
OUTSIDE_FAMILY = frozenset({"Buildings.Fluid.Sources.Outside"}) | OUTSIDE_WIND

BOUNDARIES = frozenset({"Buildings.Fluid.Sources.Boundary_pT"}) | OUTSIDE_FAMILY
ONE_WAY = frozenset({
    "Buildings.Airflow.Multizone.Orifice",
    "Buildings.Airflow.Multizone.EffectiveAirLeakageArea",
    "Buildings.Airflow.Multizone.Point_m_flow",
    "Buildings.Airflow.Multizone.Points_m_flow",
    "Buildings.Airflow.Multizone.Coefficient_V_flow",
    "Buildings.Airflow.Multizone.Coefficient_m_flow",
    "Buildings.Airflow.Multizone.Table_V_flow",
    "Buildings.Airflow.Multizone.Table_m_flow",
})
TWO_WAY = frozenset({
    "Buildings.Airflow.Multizone.DoorOpen",
    "Buildings.Airflow.Multizone.DoorOperable",
    "Buildings.Airflow.Multizone.DoorDiscretizedOpen",
    "Buildings.Airflow.Multizone.DoorDiscretizedOperable",
})
COLUMNS = frozenset({"Buildings.Airflow.Multizone.MediumColumn"})
# `ZonalFlow_ACS`/`ZonalFlow_m_flow` extend `BaseClasses.ZonalFlow`, which extends
# `Fluid.Interfaces.PartialFourPortInterface` exactly like a door -- FOUR ports
# (`port_a1`/`port_b1`/`port_a2`/`port_b2`), not the two-port `port_a`/`port_b` interface one
# might assume (verified directly against
# `Buildings/Airflow/Multizone/BaseClasses/ZonalFlow.mo` and `ZonalFlow_ACS.mo`, both of which
# write `port_a1.h_outflow`/`port_a2.m_flow` equations). The reader wires a `ZonalFlow_*`
# exactly like a door: side A holds `port_a1` and `port_b2`, side B holds `port_b1` and
# `port_a2` (see `graph.py`).
ZONAL = frozenset({
    "Buildings.Airflow.Multizone.ZonalFlow_ACS",
    "Buildings.Airflow.Multizone.ZonalFlow_m_flow",
})
THERMAL_PIN = frozenset({
    "Buildings.HeatTransfer.Sources.FixedTemperature",
    "Modelica.Thermal.HeatTransfer.Components.ThermalConductor",
})
# A prescribed heat flow into a volume's `heatPort` (MSL `Thermal/HeatTransfer/Sources/
# PrescribedHeatFlow.mo:15`): a heat source of the zone's energy balance (`assemble`).
HEAT_SOURCES = frozenset({"Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow"})
# A sensor on a volume's heat port whose output `T` is the volume's temperature (MSL 4.1.0
# `Thermal/HeatTransfer/Sensors/TemperatureSensor.mo`: `T = port.T`, `port.Q_flow = 0`);
# `graph.build` records which zone it reads and which signal inputs its `T` is wired into.
HEAT_SENSORS = frozenset({"Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor"})
# Feedback controllers: MSL 4.1.0's `LimPID` (`Blocks/Continuous.mo`). They sit in the JSON's
# `signals`, like the Math blocks; one also listed in `REFUSED` is refused with that reason.
CONTROLLERS = frozenset({"Modelica.Blocks.Continuous.LimPID"})
# Composite classes `expand.expand` rewrites into supported ones before `graph.build`
# (`MediumColumnDynamic`: a mixing volume between two half columns).
DYNAMIC_COLUMNS = frozenset({"Buildings.Airflow.Multizone.MediumColumnDynamic"})
SOURCES = frozenset({
    "Buildings.Fluid.Sources.TraceSubstancesFlowSource",
    "Buildings.Fluid.Sources.MassFlowSource_T",
})
SIGNALS = frozenset({
    "Modelica.Blocks.Sources.Ramp",
    "Modelica.Blocks.Sources.Step",
    "Modelica.Blocks.Sources.Constant",
    "Modelica.Blocks.Sources.Pulse",
    "Modelica.Blocks.Sources.Sine",
    "Modelica.Blocks.Sources.TimeTable",
    "Modelica.Blocks.Sources.CombiTimeTable",
})
# Math blocks that combine other signals (`signals.combine`, MSL v4.1.0 `Blocks/Math.mo`). In
# the JSON's `signals` a Math block's inputs are the `"<block>.<input>"` names other signals
# `drives`, so a chain of blocks is resolved from its sources outwards.
MATH = frozenset({
    "Modelica.Blocks.Math." + name for name in (
        "Gain", "Add", "Add3", "Sum", "MultiSum", "Product", "Feedback", "Division",
    )
})
# In-line (two-port, flow-through) sensors: every `Buildings.Fluid.Sensors` class that extends
# `Sensors/BaseClasses/PartialFlowSensor.mo` (directly or through `PartialDynamicFlowSensor`),
# whose equations (`:13-24`) are `port_b.m_flow = -port_a.m_flow`, `port_a.p = port_b.p` and
# isenthalpic, species-preserving pass-through: no pressure drop, no storage. `graph.build`
# joins such a sensor's two ports into one node (a transparent wire) and records it so a
# caller can map it to the flow it carries. `RelativePressure` is NOT one of these: its two
# ports sit on different nodes and carry no flow (`RelativePressure.mo`), so it stays a plain
# observer, as do the one-port sensors (`TraceSubstances`, `Temperature`, ...).
INLINE_SENSORS = frozenset({
    "Buildings.Fluid.Sensors." + name for name in (
        "DensityTwoPort", "EnthalpyFlowRate", "EntropyFlowRate", "HeatMeter",
        "LatentEnthalpyFlowRate", "MassFlowRate", "MassFractionTwoPort", "PPMTwoPort",
        "RelativeHumidityTwoPort", "SensibleEnthalpyFlowRate", "SpecificEnthalpyTwoPort",
        "SpecificEntropyTwoPort", "TemperatureTwoPort", "TemperatureWetBulbTwoPort",
        "TraceSubstancesTwoPort", "Velocity", "VolumeFlowRate",
    )
})
# Sensors, adders and other blocks used only to compare a supported model against a reference
# (listed with role "observer" and ignored by the reader). A component is
# ALSO treated as an observer whenever its own `"role": "observer"` field says so, whatever its
# class (`graph.build`); this list only covers the classes the MBL inventory names, so a
# genuine network component cannot be smuggled past the reader as an "observer" by a class
# typo unless the exporter also marked it that way.
OBSERVERS = frozenset({
    "Buildings.Fluid.Sensors.MassFlowRate",
    "Buildings.Fluid.Sensors.RelativePressure",
    "Buildings.Fluid.Sensors.TemperatureTwoPort",
    "Buildings.Fluid.Sensors.TraceSubstances",
    "Modelica.Blocks.Math.Add",
    "Modelica.Blocks.Math.Add3",
    "Modelica.Blocks.Math.Gain",
    "Modelica.Blocks.Tables.CombiTable1Ds",
})

SUPPORTED = (
    ZONES | BOUNDARIES | ONE_WAY | TWO_WAY | COLUMNS | ZONAL | THERMAL_PIN | HEAT_SOURCES
    | HEAT_SENSORS | SOURCES | SIGNALS | MATH | INLINE_SENSORS | OBSERVERS | WEATHER_READERS
)

# Refused classes, each with the short reason `graph.build` reports
# against every instance of it.
REFUSED: dict[str, str] = {
    "Modelica.Blocks.Continuous.LimPID": "feedback controllers are not supported",
}
# `BoundaryConditions.WeatherData.*`: every class under this package but the supported
# `ReaderTMY3` (`WEATHER_READERS`).
REFUSED_PREFIXES: dict[str, str] = {
    "Buildings.BoundaryConditions.WeatherData.": (
        "weather data other than a ReaderTMY3 is not supported"
    ),
}


def refusal_reason(cls: str) -> str | None:
    """The reason `cls` is refused, or `None` if this module does not (yet) know it is. A
    supported class is never refused, also when it matches a refused prefix."""
    if cls in SUPPORTED:
        return None
    if cls in REFUSED:
        return REFUSED[cls]
    for prefix, reason in REFUSED_PREFIXES.items():
        if cls.startswith(prefix):
            return reason
    return None


@dataclass(frozen=True)
class Component:
    """One instance from the JSON's `components` array. `files` names the files that travel
    with the export (`scripts/modelica_export.py`): parameter -> `{"path": <bare file name
    next to the JSON>, "sha256": <hex digest>}` (a weather reader's `filNam`)."""

    name: str
    cls: str
    parameters: dict
    role: str | None = None
    files: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    """One instance from the JSON's `signals` array: a driver plus the input(s) it feeds.

    `drives` is written in the JSON as one `"<instance>.<input>"` string or a list of them
    (one block output may feed several inputs: `Examples/ZonalFlow.mo` connects one
    `Constant` to both `floExc.mAB_flow` and `floExc.mBA_flow`); it is always a tuple here.
    """

    name: str
    cls: str
    parameters: dict
    drives: tuple[str, ...]


@dataclass(frozen=True)
class ModelicaDoc:
    """The parsed, structurally validated contents of one `noodl-modelica/1` JSON file;
    `source` is the file it was read from (files named in `Component.files` sit next to it)."""

    model: str
    mbl_commit: str
    openmodelica: str
    experiment: dict
    medium: dict
    components: tuple[Component, ...]
    connections: tuple[tuple[str, str], ...]
    signals: tuple[Signal, ...]
    source: Path | None = None


def _require_dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ModelicaImportError(f"modelica: {where} must be an object, found {value!r}")
    return value


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ModelicaImportError(f"modelica: {where} must be a string, found {value!r}")
    return value


def _parameters_of(obj: dict, where: str) -> dict:
    parameters = obj.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ModelicaImportError(f"modelica: {where}.parameters must be an object")
    return dict(parameters)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _files_of(obj: dict, where: str) -> dict:
    raw = obj.get("files", {})
    if not isinstance(raw, dict):
        raise ModelicaImportError(f"modelica: {where}.files must be an object")
    out = {}
    for key, entry in raw.items():
        e = _require_dict(entry, f"{where}.files.{key}")
        path = _require_str(e.get("path"), f"{where}.files.{key}.path")
        digest = _require_str(e.get("sha256"), f"{where}.files.{key}.sha256")
        if not _SHA256.fullmatch(digest):
            raise ModelicaImportError(
                f"modelica: {where}.files.{key}.sha256 must be 64 lowercase hex digits"
            )
        out[key] = {"path": path, "sha256": digest}
    return out


def _component_from(raw: Any, index: int) -> Component:
    where = f"components[{index}]"
    obj = _require_dict(raw, where)
    name = _require_str(obj.get("name"), f"{where}.name")
    cls = _require_str(obj.get("class"), f"{where} ({name}).class")
    parameters = _parameters_of(obj, f"{where} ({name})")
    role = obj.get("role")
    if role is not None and not isinstance(role, str):
        raise ModelicaImportError(f"modelica: {where} ({name}).role must be a string")
    files = _files_of(obj, f"{where} ({name})")
    return Component(name=name, cls=cls, parameters=parameters, role=role, files=files)


def _connection_from(raw: Any, index: int) -> tuple[str, str]:
    where = f"connections[{index}]"
    if not isinstance(raw, list) or len(raw) != 2:
        raise ModelicaImportError(f"modelica: {where} must be a 2-element array, found {raw!r}")
    a = _require_str(raw[0], f"{where}[0]")
    b = _require_str(raw[1], f"{where}[1]")
    return (a, b)


def _signal_from(raw: Any, index: int) -> Signal:
    where = f"signals[{index}]"
    obj = _require_dict(raw, where)
    name = _require_str(obj.get("name"), f"{where}.name")
    cls = _require_str(obj.get("class"), f"{where} ({name}).class")
    parameters = _parameters_of(obj, f"{where} ({name})")
    raw_drives = obj.get("drives")
    if isinstance(raw_drives, str):
        drives = (raw_drives,)
    elif (isinstance(raw_drives, list) and raw_drives
          and all(isinstance(d, str) for d in raw_drives)):
        drives = tuple(raw_drives)
    else:
        raise ModelicaImportError(
            f"modelica: {where} ({name}).drives must be a string or a non-empty list of "
            f"strings, found {raw_drives!r}"
        )
    return Signal(name=name, cls=cls, parameters=parameters, drives=drives)


def load(path: str | Path) -> ModelicaDoc:
    """Read and structurally validate one `noodl-modelica/1` JSON file.

    Checks the required top-level keys, the `format` marker, and every component's,
    connection's and signal's required fields; raises `ModelicaImportError` naming the first
    malformed field. Does not judge class support -- see `graph.build`.
    """
    text = Path(path).read_text()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelicaImportError(f"modelica: {path}: not valid JSON ({exc})") from exc
    obj = _require_dict(raw, str(path))

    fmt = obj.get("format")
    if fmt != FORMAT:
        raise ModelicaImportError(f"modelica: {path}: format is {fmt!r}, expected {FORMAT!r}")

    model = _require_str(obj.get("model"), "model")
    mbl_commit = _require_str(obj.get("mbl_commit"), "mbl_commit")
    openmodelica = _require_str(obj.get("openmodelica"), "openmodelica")
    experiment = _require_dict(obj.get("experiment", {}), "experiment")
    medium = _require_dict(obj.get("medium", {}), "medium")

    components_raw = obj.get("components")
    if not isinstance(components_raw, list):
        raise ModelicaImportError("modelica: 'components' must be an array")
    components = tuple(_component_from(c, i) for i, c in enumerate(components_raw))

    connections_raw = obj.get("connections", [])
    if not isinstance(connections_raw, list):
        raise ModelicaImportError("modelica: 'connections' must be an array")
    connections = tuple(_connection_from(c, i) for i, c in enumerate(connections_raw))

    signals_raw = obj.get("signals", [])
    if not isinstance(signals_raw, list):
        raise ModelicaImportError("modelica: 'signals' must be an array")
    signals = tuple(_signal_from(s, i) for i, s in enumerate(signals_raw))

    names = [c.name for c in components]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ModelicaImportError(f"modelica: duplicate component name(s): {', '.join(dupes)}")

    return ModelicaDoc(
        model=model, mbl_commit=mbl_commit, openmodelica=openmodelica, experiment=experiment,
        medium=medium, components=components, connections=connections, signals=signals,
        source=Path(path),
    )
