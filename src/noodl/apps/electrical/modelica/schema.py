"""JSON schema and class vocabulary of the `noodl-modelica-electrical/1` format.

One JSON object per Modelica model, written by `scripts/modelica_export_electrical.py` from
OpenModelica's evaluated component tree: an example of the Modelica Standard Library's
`Modelica.Electrical.Analog` (MSL 4.1.0) or of the Modelica Buildings Library's
`Buildings.Electrical` (MBL). Every parameter has already been evaluated by OpenModelica;
`load` only checks the file's SHAPE (required keys, the `format` marker, every component's,
connection's and signal's fields) and raises `ModelicaImportError` naming the first malformed
field. Whether a class is supported is `netlist.build`'s question; this module supplies the
vocabulary it asks it with: `ROLES` (class -> role), the exporter's class groups `GROUNDS`,
`WIRES`, `STATES`, `LINES` (equal sets, checked by the export contract test), and the
refusals `REFUSED` / `REFUSED_PREFIXES` (class -> reason), and `FLAG_DEFAULTS`, the
Modelica defaults of the Boolean parameters the reader reads that are not `false`.

The signal record, the structural checks and `ModelicaImportError` are shared with the
building reader (`noodl.modelica`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noodl.modelica.errors import ModelicaImportError
from noodl.modelica.schema import Signal, fail, require_dict, require_list, require_str, signal_from

__all__ = [
    "FORMAT", "GROUNDS", "LINES", "REFUSED", "REFUSED_PREFIXES", "ROLES", "STATES", "WIRES",
    "CircuitComponent", "CircuitDoc", "FLAG_DEFAULTS", "ModelicaImportError", "Signal",
    "domain", "load", "refusal_reason",
]

FORMAT = "noodl-modelica-electrical/1"


# The Boolean parameters the reader reads whose Modelica default is `true`; every other one
# it reads defaults to `false` (MBL commit 55abf579: `AC/OnePhase/Conversion/ACDCConverter.mo`,
# `Interfaces/Impedance.mo`).
FLAG_DEFAULTS = {"ground_DC": True, "inductive": True}

_MSL = "Modelica.Electrical.Analog."
_HEAT = "Modelica.Thermal.HeatTransfer."
_DC = "Buildings.Electrical.DC."
_AC1 = "Buildings.Electrical.AC.OnePhase."
_AC3 = "Buildings.Electrical.AC.ThreePhasesBalanced."

# The exporter's class groups (`scripts/modelica_export_electrical.py`).
GROUNDS = frozenset({_MSL + "Basic.Ground"})
WIRES = frozenset({
    _MSL + "Sensors.CurrentSensor",
    _DC + "Sensors.GeneralizedSensor",
    _AC1 + "Sensors.GeneralizedSensor",
})
STATES = {
    _MSL + "Basic.Capacitor": ("v",),
    _MSL + "Basic.Inductor": ("i",),
    _MSL + "Basic.Transformer": ("i1", "i2"),
    _HEAT + "Components.HeatCapacitor": ("T",),
    _DC + "Lines.TwoPortRCLine": ("Vc",),
}
LINES = frozenset({_AC1 + "Lines.Line", _AC3 + "Lines.Line"})


def _roles() -> dict[str, str]:
    groups = {
        "ground": GROUNDS,
        "wire": WIRES,
        "observer": {_MSL + "Sensors.VoltageSensor", _MSL + "Sensors.PotentialSensor"},
        "resistor": {_MSL + "Basic.Resistor", _MSL + "Basic.VariableResistor"},
        "capacitor": {_MSL + "Basic.Capacitor"},
        "inductor": {_MSL + "Basic.Inductor"},
        "mutual_inductance": {_MSL + "Basic.Transformer"},
        "ideal_transformer": {_MSL + "Ideal.IdealTransformer"},
        "diode": {_MSL + "Semiconductors.Diode"},
        "voltage_source": {_MSL + "Sources." + n for n in (
            "ConstantVoltage", "StepVoltage", "SineVoltage", "RampVoltage", "PulseVoltage",
            "SignalVoltage", "SineVoltageVariableFrequencyAndAmplitude",
            "CosineVoltageVariableFrequencyAndAmplitude")},
        "thermal_conductor": {_HEAT + "Components.ThermalConductor"},
        "heat_capacitor": {_HEAT + "Components.HeatCapacitor"},
        "fixed_temperature": {_HEAT + "Sources.FixedTemperature",
                              _HEAT + "Celsius.FixedTemperature"},
        "dc_source": {_DC + "Sources.ConstantVoltage"},
        "dc_resistor": {_DC + "Loads.Resistor"},
        "dc_load": {_DC + "Loads.Conductor"},
        "dc_line": {_DC + "Lines.TwoPortResistance"},
        "dc_rc_line": {_DC + "Lines.TwoPortRCLine"},
        "ac_source": {_AC1 + "Sources.FixedVoltage", _AC3 + "Sources.FixedVoltage"},
        "ac_impedance": {_AC1 + "Loads.Impedance", _AC3 + "Loads.Impedance"},
        "ac_load": {_AC1 + "Loads.Inductive", _AC1 + "Loads.Resistive"},
        "ac_line": {_AC1 + "Lines." + n for n in (
            "TwoPortResistance", "TwoPortInductance", "TwoPortRL", "TwoPortRLC", "Line")}
        | {_AC3 + "Lines.Line"},
        "ac_transformer": {_AC1 + "Conversion.ACACTransformer"},
        "ac_dc_converter": {_AC1 + "Conversion.ACDCConverter"},
    }
    return {cls: role for role, classes in groups.items() for cls in sorted(classes)}


# Class -> role. Ports per role and the nodes they give are documented in `netlist`.
ROLES: dict[str, str] = _roles()

_DOMAINS = {
    "analog": ("ground", "resistor", "capacitor", "inductor", "mutual_inductance",
               "ideal_transformer", "diode", "voltage_source"),
    "thermal": ("thermal_conductor", "heat_capacitor", "fixed_temperature"),
    "dc": ("dc_source", "dc_resistor", "dc_load", "dc_line", "dc_rc_line"),
    "ac": ("ac_source", "ac_impedance", "ac_load", "ac_line", "ac_transformer",
           "ac_dc_converter"),
}
_DOMAIN_OF = {role: dom for dom, roles in _DOMAINS.items() for role in roles}


def domain(role: str, cls: str | None = None) -> str:
    """The domain of a role: `"analog"` (MSL pins), `"thermal"`, `"dc"` or `"ac"` (MBL).

    The roles `"wire"` and `"observer"` span domains; for them the class decides (`cls`
    required). `ValueError` for an unknown role.
    """
    if role in _DOMAIN_OF:
        return _DOMAIN_OF[role]
    if role in ("wire", "observer"):
        if cls is None:
            raise ValueError(f"role {role!r} spans domains: pass the class")
        if cls.startswith(_DC):
            return "dc"
        if cls.startswith("Buildings.Electrical.AC."):
            return "ac"
        if cls.startswith(_HEAT):
            return "thermal"
        return "analog"
    raise ValueError(f"unknown role {role!r}")


# Refused classes, with the reason reported against every instance.
REFUSED: dict[str, str] = {
    _DC + "Storage.Battery": "batteries are not supported",
    _AC1 + "Storage.Battery": "batteries are not supported",
    _DC + "Sources.PVSimple": "photovoltaic sources are not supported",
    _AC1 + "Sources.PVSimple": "photovoltaic sources are not supported",
    _AC1 + "Sources.Grid": "grid sources are not supported",
    _AC3 + "Sources.Grid": "grid sources are not supported",
    _MSL + "Ideal.IdealDiode": "switching with event location is not supported",
}
REFUSED_PREFIXES: dict[str, str] = {
    "Buildings.Electrical.AC.ThreePhasesUnbalanced.":
        "unbalanced three-phase systems are not supported",
}


def refusal_reason(cls: str) -> str | None:
    """The reason the class `cls` is refused as such, or `None`."""
    if cls in REFUSED:
        return REFUSED[cls]
    for prefix, reason in REFUSED_PREFIXES.items():
        if cls.startswith(prefix):
            return reason
    return None


@dataclass(frozen=True)
class CircuitComponent:
    """One instance from the JSON's `components` array.

    `initial` holds the reference's start-time values of the class's state variables
    (`STATES`), `derived` the simulated values of an MBL `Line`'s inner line (`LINES`).
    """

    name: str
    cls: str
    parameters: dict
    role: str | None = None
    initial: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitDoc:
    """The parsed, structurally validated contents of one `noodl-modelica-electrical/1` file."""

    model: str
    mbl_commit: str
    openmodelica: str
    msl: str
    experiment: dict
    components: tuple[CircuitComponent, ...]
    connections: tuple[tuple[str, str], ...]
    signals: tuple[Signal, ...]
    compared: tuple[str, ...]


def _numbers(value: Any, where: str) -> dict:
    obj = require_dict(value, where)
    for key, v in obj.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise fail(f"{where}.{key} must be a number, found {v!r}")
    return {k: float(v) for k, v in obj.items()}


def _component_from(raw: Any, index: int) -> CircuitComponent:
    where = f"components[{index}]"
    obj = require_dict(raw, where)
    name = require_str(obj.get("name"), f"{where}.name")
    where = f"{where} ({name})"
    cls = require_str(obj.get("class"), f"{where}.class")
    parameters = require_dict(obj.get("parameters", {}), f"{where}.parameters")
    role = obj.get("role")
    if role is not None:
        require_str(role, f"{where}.role")
    initial = _numbers(obj.get("initial", {}), f"{where}.initial")
    derived = _numbers(obj.get("derived", {}), f"{where}.derived")
    return CircuitComponent(name=name, cls=cls, parameters=parameters, role=role,
                            initial=initial, derived=derived)


def _connection_from(raw: Any, index: int) -> tuple[str, str]:
    where = f"connections[{index}]"
    if not isinstance(raw, list) or len(raw) != 2:
        raise fail(f"{where} must be a 2-element array, found {raw!r}")
    return (require_str(raw[0], f"{where}[0]"), require_str(raw[1], f"{where}[1]"))


def load(path: str | Path) -> CircuitDoc:
    """Read and structurally validate one `noodl-modelica-electrical/1` JSON file.

    Required: `format`, `model`, `mbl_commit`, `openmodelica`, `msl`, `experiment`,
    `components`, `compared`; `connections` and `signals` default to empty. Duplicate
    component names are refused. Class support is not judged here (`netlist.build`).
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise fail(f"{path}: not valid JSON ({exc})") from exc
    obj = require_dict(raw, str(path))
    fmt = obj.get("format")
    if fmt != FORMAT:
        raise fail(f"{path}: format is {fmt!r}, expected {FORMAT!r}")
    strings = {key: require_str(obj.get(key), key)
               for key in ("model", "mbl_commit", "openmodelica", "msl")}
    experiment = require_dict(obj.get("experiment"), "experiment")
    components = tuple(_component_from(c, i) for i, c in
                       enumerate(require_list(obj.get("components"), "components")))
    connections = tuple(_connection_from(c, i) for i, c in
                        enumerate(require_list(obj.get("connections", []), "connections")))
    signals = tuple(signal_from(s, i) for i, s in
                    enumerate(require_list(obj.get("signals", []), "signals")))
    compared = require_list(obj.get("compared"), "compared")
    for i, column in enumerate(compared):
        require_str(column, f"compared[{i}]")
    names = [c.name for c in components]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise fail(f"duplicate component name(s): {', '.join(dupes)}")
    return CircuitDoc(experiment=experiment, components=components, connections=connections,
                      signals=signals, compared=tuple(compared), **strings)
