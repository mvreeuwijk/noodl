"""Resolve a `noodl-modelica-electrical/1` document into a netlist: devices on numbered nodes.

`build(doc)` gives every supported component its role (`schema.ROLES`), joins connector ends
into nodes by union-find, checks domains and connections, and returns a `Netlist`. It never
stops at the first problem: every refusal (an unsupported class or mode, a connection between
different kinds of connector, an enabled heat port left open, a DC source floating, ...) is
gathered and raised as ONE `ModelicaImportError` naming each instance and connector.

Connectors and nodes
--------------------
An MSL `Pin` (`p`, `n`, `p1`, ...) is one real node. An MBL DC terminal (`TwoConductor`) is
TWO nodes, one per conductor, written `<inst>.<terminal>[1]` and `[2]`; a `connect` joins
conductor k with conductor k. An MBL AC terminal (one-phase, also used by the balanced
three-phase package) is ONE complex node; every one-terminal AC device is an edge from it to
the reference. MSL `Ground`, MBL's implicit AC neutral and the absolute DC reference are all
the electrical node 0, named `"0"`. Heat ports are nodes of a separate thermal network whose
node 0, `"thermal.reference"`, is the 0 K reference.

Union-find elements are tuples: `("pin", "R1.p")`, `("dc", "res.terminal", 1)`,
`("ac", "Z1.terminal")`, `("heat", "R1.heatPort")`, `("internal", "RC.mid")`, and the fixed
`REFERENCE` and `THERMAL_REFERENCE`. Ideal current sensors (`schema.WIRES`) join their two
ends; observers (voltage and potential sensors, and any component the exporter marked
`"role": "observer"`) are ignored with their connections; grounds join their pin to node 0.
Nodes are numbered in the order the devices' ports first reach them (devices in document
order, ports in the order below), after the reference; a node is named after the first
element that reached it.

`Device.nodes` keys per role (MBL `terminal_n` is the side a two-port's current enters):

- `resistor`, `capacitor`, `inductor`, `diode`, `voltage_source`: `p`, `n`;
  `mutual_inductance`, `ideal_transformer`: `p1`, `n1`, `p2`, `n2`; `heat_node` is the
  heat port's thermal node when the heat port is enabled (`useHeatPort`; MSL's `Diode` sets it
  to `useTemperatureDependency`).
- `thermal_conductor`: `port_a`, `port_b`; `heat_capacitor`, `fixed_temperature`: `port`
  (thermal node numbers).
- `dc_source`: `c1`, `c2` (`terminal[1]`, `[2]`); its pin `n` is joined to `c2` and must
  reach node 0 (a `Ground`): MBL's `v[1] = V` is absolute.
- `dc_resistor`, `dc_load`: `c1`, `c2` (`terminal`).
- `dc_line`: `p1` (`terminal_n[1]`), `n1` (`terminal_p[1]`); conductors 2 joined.
- `dc_rc_line`: `p1`, `n1`, `c2` (conductors 2 joined), `mid` (internal node `<inst>.mid`).
- `ac_source`, `ac_impedance`, `ac_load`: `a` (`terminal`).
- `ac_line`: `p` (`terminal_n`), `n` (`terminal_p`), plus `mid` (`<inst>.mid`) for
  `TwoPortRLC` and the balanced three-phase `Line`.
- `ac_transformer`: `n` (`terminal_n`), `p` (`terminal_p`).
- `ac_dc_converter`: `ac` (`terminal_n`), `p1`, `p2` (`terminal_p[1]`, `[2]`); `p2` is
  node 0 (MBL `ground_DC = true`, else refused).

`Netlist.node_of` maps the text of every electrical element that reaches a device's node
(device ports, sensor ends, ground pins, `"res.terminal_n[2]"`, ...) to its node, so a
compared terminal variable can be located; `thermal_of` does the same for heat ports.

Real or phasor
--------------
A document with any `Buildings.Electrical.AC.*` component is a phasor model at the AC
sources' common frequency `f` (Hz, `Netlist.frequency`); every other document is real. In a
phasor model MSL analog components other than `Ground` are refused, as is a DC
`TwoPortRCLine` with `use_C` (a phasor layer would give its capacitor the admittance
`j omega C`); AC sources at different frequencies are refused.

Parameters are passed through as exported (`Device.parameters`); fixed temperatures are
collected in kelvin in `Netlist.fixed_temperatures` (`Celsius.FixedTemperature` adds 273.15).
MBL line classes are checked against the simulated values of their inner line (`derived`):
the one-phase `Line` uses `R/3` and `L/3`, the balanced three-phase `Line` `R`, `L`, `C`.
The parameters that decide a refusal or a node must be exported: `use_T` of the MBL lines and
`considerMagnetization` of `IdealTransformer` as booleans, the temperature `T` of a fixed
temperature and the frequency `f` of an AC source as finite numbers. An MBL resistive part
without heat port must sit at its reference temperature (`T`, or a `Line`'s `TCable`, equal
to `T_ref`), and a `TwoPortRCLine` with `use_C` must have its conductor 2 on node 0 (its
capacitor is the edge `mid -> c2`). A `Diode` with `useHeatPort` but without
`useTemperatureDependency` is refused (its law would ignore the heat-port temperature), as is
a thermal network part (thermal nodes joined by conductors) that no fixed temperature and no
heat capacitor holds: its temperature is undetermined; the refusal names those nodes.

Connections follow Modelica's rules for what is not a physical connector: a `connect` to a
DISABLED conditional heat port is removed (legal Modelica: the connection disappears with the
connector), and a connection with a signal-block end is skipped whole (signal wiring is
carried by the signals' `drives`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass

from noodl._unionfind import UnionFind
from noodl.apps.electrical.modelica import schema
from noodl.apps.electrical.modelica.schema import (
    FLAG_DEFAULTS,
    CircuitComponent,
    CircuitDoc,
    ModelicaImportError,
    Signal,
)
from noodl.modelica.schema import flag

REFERENCE = ("reference", "0")
THERMAL_REFERENCE = ("thermal_reference", "thermal.reference")

_AC1 = "Buildings.Electrical.AC.OnePhase."
_AC3 = "Buildings.Electrical.AC.ThreePhasesBalanced."
_CELSIUS = "Modelica.Thermal.HeatTransfer.Celsius.FixedTemperature"
_ONE_PHASE_LINE = _AC1 + "Lines.Line"
_RLC_LINES = frozenset({_AC1 + "Lines.TwoPortRLC", _AC3 + "Lines.Line"})
_THERMAL_ROLES = frozenset({"thermal_conductor", "heat_capacitor", "fixed_temperature"})
_KIND_TEXT = {"pin": "pin", "dc": "DC terminal", "ac": "AC terminal", "heat": "heat port"}
_LINE_RTOL = 1e-12

_TWO_PIN = ("p", "n")
_FOUR_PIN = ("p1", "n1", "p2", "n2")
_ANALOG_PINS = {
    "ground": ("p",), "resistor": _TWO_PIN, "capacitor": _TWO_PIN, "inductor": _TWO_PIN,
    "diode": _TWO_PIN, "voltage_source": _TWO_PIN, "mutual_inductance": _FOUR_PIN,
    "ideal_transformer": _FOUR_PIN,
}
# The Boolean parameters the netlist reads; a value other than true or false refuses the
# component (`_declared_problems`), a missing one takes its Modelica default.
_FLAGS = ("useHeatPort", "useTemperatureDependency", "use_T", "use_C", "considerMagnetization",
          "ground_DC")
_THERMAL_PORTS = {"thermal_conductor": ("port_a", "port_b"), "heat_capacitor": ("port",),
                  "fixed_temperature": ("port",)}
_TWO_TERMINALS = ("terminal_n", "terminal_p")
# Connector name -> kind, for the MBL roles.
_MBL_CONNECTORS = {
    "dc_source": {"terminal": "dc", "n": "pin"},
    "dc_resistor": {"terminal": "dc", "heatPort": "heat"},
    "dc_load": {"terminal": "dc"},
    "dc_line": {"terminal_n": "dc", "terminal_p": "dc", "heatPort": "heat"},
    "dc_rc_line": {"terminal_n": "dc", "terminal_p": "dc", "heatPort": "heat"},
    "ac_source": {"terminal": "ac"},
    "ac_impedance": {"terminal": "ac"},
    "ac_load": {"terminal": "ac"},
    "ac_line": {"terminal_n": "ac", "terminal_p": "ac", "heatPort": "heat"},
    "ac_transformer": {"terminal_n": "ac", "terminal_p": "ac"},
    "ac_dc_converter": {"terminal_n": "ac", "terminal_p": "dc"},
}


@dataclass(frozen=True)
class Device:
    """One supported component on its nodes.

    `nodes` maps the role's port labels (module docstring) to electrical node numbers, or to
    thermal node numbers for the thermal roles; `heat_node` is the thermal node of an
    enabled heat port, else `None`. `parameters`, `initial` and `derived` are the exported
    values, unchanged.
    """

    name: str
    cls: str
    role: str
    parameters: dict
    nodes: dict[str, int]
    heat_node: int | None
    initial: dict
    derived: dict


@dataclass(frozen=True)
class Netlist:
    """Devices on numbered electrical and thermal nodes (node 0 is each network's reference).

    `phasor` is true for a model with MBL AC components, `frequency` their sources' common
    frequency in Hz (`None` for a real model). `signals` are the document's signal blocks
    still driving a component after `omit`. `fixed_temperatures` maps a thermal node to its
    temperature in K. `omitted` names the components dropped by `omit`.
    """

    node_names: tuple[str, ...]
    node_of: dict[str, int]
    thermal_names: tuple[str, ...]
    thermal_of: dict[str, int]
    devices: tuple[Device, ...]
    phasor: bool
    frequency: float | None
    signals: tuple[Signal, ...]
    fixed_temperatures: dict[int, float]
    omitted: frozenset[str] = frozenset()


def _text(element: tuple) -> str:
    if element[0] == "dc":
        return f"{element[1]}[{element[2]}]"
    return element[1]


def _dc(end: str) -> tuple[tuple, tuple]:
    return ("dc", end, 1), ("dc", end, 2)


# ------------------------------------------------------------------------------ per class
def _flag(parameters: dict, name: str) -> bool:
    """A Boolean parameter, its Modelica default when not exported; `_declared_problems`
    has refused a component whose value is anything but true or false."""
    return flag(parameters, name, default=FLAG_DEFAULTS.get(name, False))


def _heat_enabled(comp: CircuitComponent, role: str) -> str | None:
    """The flag that enables `comp`'s heat port, or `None` when it has none or it is off."""
    if role == "diode" and _flag(comp.parameters, "useTemperatureDependency"):
        return "useTemperatureDependency"
    if _flag(comp.parameters, "useHeatPort"):
        return "useHeatPort"
    return None


def _connectors(comp: CircuitComponent, role: str) -> dict[str, str]:
    """Connector name -> kind (`"pin"`, `"dc"`, `"ac"`, `"heat"`)."""
    if role in _ANALOG_PINS:
        kinds = {pin: "pin" for pin in _ANALOG_PINS[role]}
        if role in ("resistor", "diode"):
            kinds["heatPort"] = "heat"
        return kinds
    if role in _THERMAL_PORTS:
        return {port: "heat" for port in _THERMAL_PORTS[role]}
    if role == "wire":
        dom = schema.domain(role, comp.cls)
        if dom == "analog":
            return {"p": "pin", "n": "pin"}
        return {t: dom for t in _TWO_TERMINALS}
    return dict(_MBL_CONNECTORS[role])


def _elements(comp: CircuitComponent, role: str) -> tuple[dict[str, tuple], list[tuple]]:
    """`(port label -> element, [(a, b) joined inside the component])` for a device."""
    n = comp.name
    if role in _ANALOG_PINS:
        return {pin: ("pin", f"{n}.{pin}") for pin in _ANALOG_PINS[role]}, []
    if role in _THERMAL_PORTS:
        return {port: ("heat", f"{n}.{port}") for port in _THERMAL_PORTS[role]}, []
    if role in ("dc_source", "dc_resistor", "dc_load"):
        c1, c2 = _dc(f"{n}.terminal")
        joins = [(("pin", f"{n}.n"), c2)] if role == "dc_source" else []
        return {"c1": c1, "c2": c2}, joins
    if role in ("dc_line", "dc_rc_line"):
        (p1, p2), (n1, n2) = _dc(f"{n}.terminal_n"), _dc(f"{n}.terminal_p")
        nodes = {"p1": p1, "n1": n1}
        if role == "dc_rc_line":
            nodes.update(c2=p2, mid=("internal", f"{n}.mid"))
        return nodes, [(p2, n2)]
    if role in ("ac_source", "ac_impedance", "ac_load"):
        return {"a": ("ac", f"{n}.terminal")}, []
    if role == "ac_line":
        nodes = {"p": ("ac", f"{n}.terminal_n"), "n": ("ac", f"{n}.terminal_p")}
        if comp.cls in _RLC_LINES:
            nodes["mid"] = ("internal", f"{n}.mid")
        return nodes, []
    if role == "ac_transformer":
        return {"n": ("ac", f"{n}.terminal_n"), "p": ("ac", f"{n}.terminal_p")}, []
    if role == "ac_dc_converter":
        p1, p2 = _dc(f"{n}.terminal_p")
        return {"ac": ("ac", f"{n}.terminal_n"), "p1": p1, "p2": p2}, [(p2, REFERENCE)]
    raise AssertionError(f"no elements for role {role!r}")  # pragma: no cover


def _line_problems(comp: CircuitComponent) -> list[str]:
    """The MBL `Line` rule against the simulated inner-line values (`derived`)."""
    p, where = comp.parameters, f"{comp.name} ({comp.cls})"
    one_phase = comp.cls == _ONE_PHASE_LINE
    keys = ("R", "L") if one_phase else ("R", "L", "C")
    missing = [k for k in keys if not isinstance(p.get(k), (int, float))]
    if missing:
        return [f"{where}: parameter(s) {', '.join(missing)} missing"]
    if not comp.derived:
        return [f"{where}: the simulated inner-line values ('derived') are missing, so the "
                f"line rule cannot be checked"]
    problems = []
    for k in keys:
        expected = p[k] / 3 if one_phase else p[k]
        rule = f"{k}/3" if one_phase else k
        got = comp.derived.get(f"line.{k}")
        if got is None:
            problems.append(f"{where}: simulated line.{k} is missing")
        elif abs(got - expected) > _LINE_RTOL * max(abs(got), abs(expected)):
            problems.append(f"{where}: simulated line.{k} = {got!r} differs from the class "
                            f"rule {rule} = {expected!r}")
    return problems


def _finite(value) -> float | None:
    """`value` as a finite float, or `None` (a bool, a non-number, or a number that is not
    finite or does not fit a float, such as a 400-digit integer)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _declared_problems(comp: CircuitComponent, role: str) -> list[str]:
    """Refusal-deciding flags (booleans) and numbers that must have been exported."""
    p, where, problems = comp.parameters, f"{comp.name} ({comp.cls})", []
    for key in _FLAGS:
        flag(p, key, problems=problems, where=where)
    key = {"fixed_temperature": "T", "ac_source": "f"}.get(role)
    if key is not None and _finite(p.get(key)) is None:
        problems.append(f"{where}: {key} must be a finite number, found {p.get(key)!r}")
    return problems


def _temperature_problems(comp: CircuitComponent) -> list[str]:
    """An MBL resistive part without heat port must be at its reference temperature."""
    p = comp.parameters
    if "T_ref" not in p:
        return []
    key = "T" if "T" in p else "TCable" if "TCable" in p else None
    if key is None or p[key] == p["T_ref"]:
        return []
    return [f"{comp.name} ({comp.cls}): {key} = {p[key]!r} differs from T_ref = "
            f"{p['T_ref']!r}; a resistance off its reference temperature is not supported"]


def _component_problems(comp: CircuitComponent, role: str | None) -> list[str]:
    """Refusals that depend on one component alone."""
    where = f"{comp.name} ({comp.cls})"
    reason = schema.refusal_reason(comp.cls)
    if reason is not None:
        return [f"{where}: {reason}"]
    if role is None:
        return [f"{where}: class is not supported"]
    problems = _declared_problems(comp, role)
    if problems:
        return problems
    p = comp.parameters
    if comp.cls.startswith("Buildings.Electrical."):
        for key in ("mode", "modelMode"):
            if str(p.get(key, "")).endswith("FixedZ_dynamic"):
                problems.append(f"{where}: {key} = {p[key]}: dynamic phasor models are not "
                                f"supported")
        if _flag(p, "use_T"):
            problems.append(f"{where}: temperature-dependent cables (use_T) are not supported")
        if _flag(p, "useHeatPort"):
            problems.append(f"{where}: heat ports of MBL components are not supported "
                            f"(useHeatPort = true)")
        else:
            problems += _temperature_problems(comp)
    if (role == "diode" and _flag(p, "useHeatPort")
            and not _flag(p, "useTemperatureDependency")):
        problems.append(f"{where}: a heat port on a diode without temperature dependency "
                        f"(useHeatPort = true, useTemperatureDependency = false) is not "
                        f"supported")
    if role == "ideal_transformer" and _flag(p, "considerMagnetization"):
        problems.append(f"{where}: considerMagnetization = true is not supported")
    if role == "ac_dc_converter" and not _flag(p, "ground_DC"):
        problems.append(f"{where}: the DC side's conductor 2 must be grounded "
                        f"(ground_DC = true), found ground_DC = {p.get('ground_DC')!r}")
    if not problems and comp.cls in schema.LINES:
        problems += _line_problems(comp)
    return problems


def _floating_thermal_nodes(devices: list[Device], fixed: dict, thermal_names: list[str]
                            ) -> list[str]:
    """The thermal nodes whose part of the thermal network (joined by thermal conductors)
    holds neither a fixed temperature nor a heat capacitor: their temperature is not
    determined (in a steady solve a heat capacitor alone would not fix it either, but a
    circuit with one is always stepped)."""
    parent = list(range(len(thermal_names)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for d in devices:
        if d.role == "thermal_conductor":
            parent[find(d.nodes["port_a"])] = find(d.nodes["port_b"])
    held = {find(node) for node in fixed}
    held |= {find(d.nodes["port"]) for d in devices if d.role == "heat_capacitor"}
    held.add(find(0))       # the reference, and whatever a conductor joins to it
    floating = [thermal_names[k] for k in range(1, len(thermal_names))
                if find(k) not in held]
    if not floating:
        return []
    return [f"the thermal network is floating at {', '.join(floating)}: no fixed temperature "
            f"and no heat capacitor holds the temperature there; add a FixedTemperature"]


def raise_problems(problems: Iterable[str]) -> None:
    """Raise ONE `ModelicaImportError` listing every problem (deduplicated, in order);
    nothing when there is none."""
    unique = list(dict.fromkeys(problems))
    if not unique:
        return
    count = "1 problem" if len(unique) == 1 else f"{len(unique)} problems"
    raise ModelicaImportError(f"modelica: {count}:\n" + "\n".join(f"  - {e}" for e in unique))


# ------------------------------------------------------------------------------ build
def _kept_signals(signals: tuple[Signal, ...], omit: set[str]) -> tuple[Signal, ...]:
    kept = []
    for sig in signals:
        drives = tuple(d for d in sig.drives if d.partition(".")[0] not in omit)
        if drives:
            kept.append(dataclasses.replace(sig, drives=drives))
    return tuple(kept)


def build(doc: CircuitDoc, *, omit: Iterable[str] = ()) -> Netlist:
    """Resolve `doc` into a `Netlist`, dropping the components named in `omit` first.

    Raises ONE `ModelicaImportError` listing every problem found (module docstring).
    """
    omit = set(omit)
    problems: list[str] = []
    names = {c.name for c in doc.components}
    problems += [f"omit: no component named {o!r}" for o in sorted(omit - names)]
    components = [c for c in doc.components if c.name not in omit]
    connections = [(a, b) for a, b in doc.connections
                   if a.partition(".")[0] not in omit and b.partition(".")[0] not in omit]
    signal_names = {s.name for s in doc.signals}

    # roles and per-component refusals
    roles: dict[str, str] = {}
    by_name = {c.name: c for c in components}
    for comp in components:
        role = "observer" if comp.role == "observer" else schema.ROLES.get(comp.cls)
        found = _component_problems(comp, role)
        problems += found
        if not found:
            roles[comp.name] = role
    active = [c for c in components if c.name in roles and roles[c.name] != "observer"]

    # real or phasor
    phasor = any(c.cls.startswith("Buildings.Electrical.AC.") for c in components)
    frequency = None
    if phasor:
        frequencies: dict[str, float] = {}
        for comp in active:
            role, where = roles[comp.name], f"{comp.name} ({comp.cls})"
            if schema.domain(role, comp.cls) == "analog" and role != "ground":
                problems.append(f"{where}: Modelica.Electrical.Analog components cannot be "
                                f"part of a phasor (AC) model")
            if role == "dc_rc_line" and _flag(comp.parameters, "use_C"):
                problems.append(f"{where}: a DC line with capacitance (use_C = true) cannot "
                                f"be part of a phasor (AC) model")
            if role == "ac_source":
                frequencies[comp.name] = float(comp.parameters["f"])
        distinct = set(frequencies.values())
        refused_sources = any(c.name not in roles and schema.ROLES.get(c.cls) == "ac_source"
                              for c in components)
        if not frequencies:
            if not refused_sources:     # a refused source is reported already
                problems.append("phasor (AC) model: no AC source sets the frequency")
        elif len(distinct) > 1:
            listed = ", ".join(f"{n} at {f!r} Hz" for n, f in frequencies.items())
            problems.append(f"phasor (AC) model: AC sources at different frequencies "
                            f"({listed})")
        else:
            frequency = distinct.pop()

    # union-find over every active component's elements
    # the fixed references stay their own class's root
    uf = UnionFind(keep_root=lambda e: e in (REFERENCE, THERMAL_REFERENCE))
    uf.add(REFERENCE)
    uf.add(THERMAL_REFERENCE)
    device_elements: dict[str, dict[str, tuple]] = {}
    heat_elements: dict[str, tuple] = {}
    for comp in active:
        role = roles[comp.name]
        if role == "ground":
            uf.union(REFERENCE, ("pin", f"{comp.name}.p"))
        elif role == "wire":
            kinds = _connectors(comp, role)
            a, b = (f"{comp.name}.{k}" for k in kinds)
            kind = next(iter(kinds.values()))
            if kind == "dc":
                for ea, eb in zip(_dc(a), _dc(b), strict=True):
                    uf.union(ea, eb)
            else:
                uf.union((kind, a), (kind, b))
        else:
            nodes, joins = _elements(comp, role)
            for element in nodes.values():
                uf.add(element)
            for ea, eb in joins:
                uf.union(ea, eb)
            device_elements[comp.name] = nodes
            if _heat_enabled(comp, role) and role in ("resistor", "diode"):
                heat_elements[comp.name] = ("heat", f"{comp.name}.heatPort")

    connected_heat: set[str] = set()
    for a, b in connections:
        if any(e.partition(".")[0] in signal_names and e.partition(".")[0] not in by_name
               for e in (a, b)):
            continue    # signal wiring: carried by the signals' drives
        ends = []
        for end in (a, b):
            inst, _, connector = end.partition(".")
            comp = by_name.get(inst)
            if comp is None:
                if inst not in signal_names:
                    problems.append(f"connection {a} -- {b}: no component named {inst!r}")
                ends.append(None)
                continue
            role = roles.get(inst)
            if role is None or role == "observer":
                ends.append(None)       # refused (reported) or ignored
                continue
            kind = _connectors(comp, role).get(connector)
            if kind is None:
                problems.append(f"connection {a} -- {b}: {end} is not a connector of "
                                f"{comp.cls}")
            elif connector == "heatPort" and not _heat_enabled(comp, role):
                kind = None     # a connect to a disabled conditional connector is removed
            ends.append(None if kind is None else (end, kind))
        if None in ends:
            continue
        (ea, ka), (eb, kb) = ends
        if ka != kb:
            problems.append(f"connection {a} -- {b}: {_KIND_TEXT[ka]} {ea} cannot connect to "
                            f"{_KIND_TEXT[kb]} {eb}")
            continue
        if ka == "dc":
            for xa, xb in zip(_dc(ea), _dc(eb), strict=True):
                uf.union(xa, xb)
        else:
            uf.union((ka, ea), (ka, eb))
        connected_heat.update(e for e, k in ((ea, ka), (eb, kb)) if k == "heat")

    # per-component connection checks
    for comp in active:
        role = roles[comp.name]
        flag = _heat_enabled(comp, role)
        if role in ("resistor", "diode") and flag and f"{comp.name}.heatPort" not in connected_heat:
            problems.append(f"{comp.name}.heatPort: the heat port is enabled ({flag} = true) "
                            f"but not connected")
        if role == "dc_source" and uf.find(("pin", f"{comp.name}.n")) != uf.find(REFERENCE):
            problems.append(f"{comp.name}.n: the DC source's pin n must be connected to a "
                            f"Ground (its terminal voltage is absolute)")

    # node numbering, in the order the devices' ports first reach a node
    electrical: dict[tuple, int] = {uf.find(REFERENCE): 0}
    electrical_names = ["0"]
    thermal: dict[tuple, int] = {uf.find(THERMAL_REFERENCE): 0}
    thermal_names = ["thermal.reference"]

    def number(element: tuple, table: dict, labels: list) -> int:
        root = uf.find(element)
        if root not in table:
            table[root] = len(labels)
            labels.append(_text(element))
        return table[root]

    devices: list[Device] = []
    fixed: dict[int, tuple[str, float]] = {}
    for comp in active:
        role = roles[comp.name]
        if role in ("ground", "wire"):
            continue
        thermal_role = role in _THERMAL_ROLES
        table, labels = (thermal, thermal_names) if thermal_role else (electrical,
                                                                         electrical_names)
        nodes = {k: number(e, table, labels) for k, e in device_elements[comp.name].items()}
        heat = heat_elements.get(comp.name)
        heat_node = None if heat is None else number(heat, thermal, thermal_names)
        devices.append(Device(name=comp.name, cls=comp.cls, role=role,
                              parameters=dict(comp.parameters), nodes=nodes,
                              heat_node=heat_node, initial=dict(comp.initial),
                              derived=dict(comp.derived)))
        if role == "dc_rc_line" and _flag(comp.parameters, "use_C") and nodes["c2"] != 0:
            problems.append(f"{comp.name} ({comp.cls}): with use_C = true the line's "
                            f"conductor 2 must be connected to the reference (node 0), found "
                            f"{labels[nodes['c2']]}")
        if role == "fixed_temperature":
            t = float(comp.parameters["T"])
            t = t + 273.15 if comp.cls == _CELSIUS else t
            node = nodes["port"]
            if node in fixed:
                problems.append(f"{fixed[node][0]} and {comp.name}: two fixed temperatures on "
                                f"one thermal node ({thermal_names[node]})")
            else:
                fixed[node] = (comp.name, t)

    problems += _floating_thermal_nodes(devices, fixed, thermal_names)
    raise_problems(problems)

    def lookup(table: dict) -> dict[str, int]:
        return {_text(e): table[uf.find(e)] for e in uf.parent
                if e not in (REFERENCE, THERMAL_REFERENCE) and uf.find(e) in table}

    return Netlist(
        node_names=tuple(electrical_names), node_of=lookup(electrical),
        thermal_names=tuple(thermal_names), thermal_of=lookup(thermal),
        devices=tuple(devices), phasor=phasor, frequency=frequency,
        signals=_kept_signals(doc.signals, omit),
        fixed_temperatures={node: t for node, (_, t) in fixed.items()},
        omitted=frozenset(omit),
    )
