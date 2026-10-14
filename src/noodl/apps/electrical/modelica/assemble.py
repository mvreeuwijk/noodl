"""Assemble a `Netlist` into a noodl circuit model: laws, drivers, step grid, start state, names.

`build(netlist, doc)` returns `(model, state, drivers, names)`: a `Model` with ONE
`PotentialFlowLayer` named `"electrical"` (real, or phasor for a model with MBL AC parts), the
start state, the drivers on the step grid and a `CircuitNames` that maps the document's
compared columns to noodl quantities.

Network and laws
----------------
The `Network` holds the electrical nodes (`Netlist.node_names`, node 0 `"0"` the only
boundary) and then the thermal nodes (`Netlist.thermal_names`; no edge of the electrical layer
touches them), so a node's position in `phi` is its netlist number (electrical) or the number
of electrical nodes plus its thermal number. Every device kind is one law instance carrying
per-edge parameters in device order (a two-port device adds its two edges consecutively); the
kinds come in the order `voltage_source`, `resistor`, `diode`, `capacitor`, `inductor`,
`transformer`, `ideal_transformer`, `impedance`, `load`, `impedance_transformer`,
`power_converter` (empty ones omitted), which is also the order of the layer's flows `q`.

- MSL `Basic.Resistor` / `VariableResistor`: `Resistance(R, alpha, T_ref)` reading the
  per-edge temperature `"electrical.resistor.T"` (the device's `T` without a heat port) and,
  when a variable resistor exists, the scale `"electrical.resistor.scale"` (its input `R`;
  1 for a fixed resistor, whose resistance is `R`).
- MSL `Semiconductors.Diode`: `JunctionDiode(Ids, Vt, Maxexp, R, N, TNOM, XTI, EG)`, reading
  `"electrical.diode.T"` with `useTemperatureDependency`.
- MSL `Capacitor`, `Inductor`, `Transformer`, `IdealTransformer`: `Capacitance`,
  `Inductance`, `MutualInductance`, `TransformerRatio`, from the device parameters.
- MSL voltage sources: `PrescribedVoltage(1.0)` with one `Signal` waveform per source reading
  the series `"electrical.voltage_source.value"` (below); MBL `DC.Sources.ConstantVoltage`
  (edge `c1 -> 0`, value `V`) joins the same kind.
- MBL `DC.Loads.Resistor`, `DC.Lines.TwoPortResistance`: `Impedance(R)`; `TwoPortRCLine`:
  `Impedance(R/2)` on `p1 -> mid` and `n1 -> mid`, and with `use_C` a `Capacitance(C)` on
  `mid -> c2` starting at the recorded `Vc`.
- MBL AC impedances, `Impedance(R, L, C)` at the layer's frequency: `Loads.Impedance` (an
  inductive one `(R, L)`, a capacitive one `(R, C)`, `C > 0`; a balanced three-phase one in
  triangle, `star = false`, `R/3`, `L/3`, `3 C`), `Lines.TwoPortResistance` `(R)`,
  `TwoPortInductance` `(0, L)`, `TwoPortRL` `(R, L)`, the one-phase `Line` `(R/3, L/3)`, and
  `TwoPortRLC` / the balanced three-phase `Line` as a T: `(R/2, L/2)` on `p -> mid` and on
  `n -> mid`, and with `C > 0` the shunt `(0, C)` on `mid -> 0`.
- MBL loads, `ConstantPowerLoad(P, Q, V_nominal, linearized)` with the ABSORBED power
  `P = -P_mbl`, `Q = -P_mbl tan(acos(pf))` (`Inductive`; 0 for `Resistive` and the DC
  `Conductor`, `direct_current=True`). Load modes (`Interfaces/Load.mo`):
  `FixedZ_steady_state` `P_mbl = P_nominal`; `VariableZ_y_input` `P_nominal` scaled by
  `1e-10 + (1 - 1e-10) y`; `VariableZ_P_input` `P_mbl = 1` scaled by the input `Pow`; the
  scales are the series `"electrical.load.scale"`. The power factor enters as MBL's
  `smoothMin(pf, 0.99999, 2.5e-6)`: `pf` itself below the band, 0.99999 above it, refused
  inside it.
- MBL `ACACTransformer`: `ImpedanceTransformer(VHigh/VLow, Zp e^{j atan(XoverR)})`,
  `Zp = VHigh^2 Zperc / VABase`, edges `n -> 0`, `p -> 0`; `ACDCConverter`:
  `PowerConverter(conversionFactor, eta)`, edges `ac -> 0`, `p1 -> p2`.

Sources, signal inputs and the step grid
----------------------------------------
Every MSL voltage source becomes the equivalent `Modelica.Blocks.Sources` block
(`ConstantVoltage` -> `Constant(k=V)`, `StepVoltage` -> `Step(height=V)`, `SineVoltage` ->
`Sine(amplitude=V, f, phase)`, `RampVoltage` -> `Ramp(height=V, duration)`, `PulseVoltage`
-> `Pulse(amplitude=V, width, period)`, each with its `offset` and `startTime`), evaluated by
the building reader's `signals.evaluate` on the step grid (the left limit at an event). The
variable-frequency sources integrate their frequency exactly, `phi = 2 pi sum(mean f dt)`
over the step grid (`signals.interval_means`, `phi = 0` at the start) and give
`offset + V sin(phi)` (`cos` for the cosine class). `SignalVoltage` follows its input `v`.
Signal inputs (`VariableResistor.R`, the loads' `y` and `Pow`, the sources' `f`, `V`, `v`)
come from the document's signals, `Modelica.Blocks.Math` chains included; an enabled input
nothing drives, and a signal driving an input the reader does not read, are refused.

The step grid is the output grid plus every jump and kink of those signals and of the source
blocks (building `assemble.driver_grid`), each interval split into `substeps` steps, so the
grid at `2 r` substeps is the grid at `r` halved. A step starting at a jump (and the first
step) is backward Euler: `CircuitNames.restarts`.

Phasors
-------
A phasor model holds the MBL convention: a terminal's `v = {v[1], v[2]}` is the complex node
potential `v[1] + j v[2]` (and the same for `i`), a root-mean-square phasor in the frame
rotating with the source angle `2 pi f t` (`PhaseSystems.OnePhase.phaseVoltages`:
`v = {V cos(phi), V sin(phi)}`). The reader writes each AC source as `PrescribedVoltage` with
the complex amplitude `V e^{j phiSou}` and a `Constant` waveform, so the layer's potentials
ARE MBL's phasors (no `Sine` waveform, whose phasor is noodl's peak-amplitude convention, is
used), and the power laws, which assume root-mean-square phasors (`S = dp conj(q)`), hold.

Start state
-----------
A model with storage starts from the reference's recorded start values: `"electrical.phi"`
from the capacitor voltages, propagated breadth-first over the capacitor edges from node 0
(`phi[v] = phi[u] - v0` along `u -> v`; a group of capacitors not reaching node 0 is rooted
at 0 V on its first node; a capacitor closing a loop is not read), every other node 0; and
`"electrical.q"` zero except on inductor and transformer edges (their recorded currents). The
first step is backward Euler, so it reads only the capacitor voltages and the inductor
currents. A model without storage starts its Newton solve from a flat guess: every
electrical node but node 0 at the largest source magnitude at the start time (complex in a
phasor model), all flows zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from noodl.apps.building_physics.modelica import signals
from noodl.apps.building_physics.modelica.assemble import (
    driver_grid,
    experiment_times,
    grid_events,
)
from noodl.apps.electrical.converters import ImpedanceTransformer, PowerConverter
from noodl.apps.electrical.laws import (
    Capacitance,
    Inductance,
    JunctionDiode,
    MutualInductance,
    PrescribedVoltage,
    Resistance,
    TransformerRatio,
)
from noodl.apps.electrical.loads import ConstantPowerLoad, Impedance
from noodl.apps.electrical.modelica.netlist import Device, Netlist, raise_problems
from noodl.apps.electrical.modelica.schema import CircuitDoc, ModelicaImportError, Signal
from noodl.apps.electrical.waveforms import Constant
from noodl.apps.electrical.waveforms import Signal as SignalWaveform
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
C128 = torch.complex128
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

__all__ = ["CircuitNames", "Column", "build"]

LAYER = "electrical"
SOURCE_KEY = "electrical.voltage_source.value"
RESISTOR_T = "electrical.resistor.T"
RESISTOR_SCALE = "electrical.resistor.scale"
DIODE_T = "electrical.diode.T"
LOAD_SCALE = "electrical.load.scale"
_SERIES = "series:"
_KINDS = ("voltage_source", "resistor", "diode", "capacitor", "inductor", "transformer",
          "ideal_transformer", "impedance", "load", "impedance_transformer",
          "power_converter")
_BLOCKS = "Modelica.Blocks.Sources."
_MSL_SOURCES = "Modelica.Electrical.Analog.Sources."
_VARIABLE_FREQUENCY = ("SineVoltageVariableFrequencyAndAmplitude",
                       "CosineVoltageVariableFrequencyAndAmplitude")
# MSL voltage source -> (equivalent Blocks.Sources block, {block parameter: source parameter}).
_EQUIVALENT = {
    "ConstantVoltage": ("Constant", {"k": "V"}),
    "StepVoltage": ("Step", {"height": "V", "offset": "offset", "startTime": "startTime"}),
    "SineVoltage": ("Sine", {"amplitude": "V", "f": "f", "phase": "phase",
                             "offset": "offset", "startTime": "startTime"}),
    "RampVoltage": ("Ramp", {"height": "V", "duration": "duration", "offset": "offset",
                             "startTime": "startTime"}),
    "PulseVoltage": ("Pulse", {"amplitude": "V", "width": "width", "period": "period",
                               "nperiod": "nperiod", "offset": "offset",
                               "startTime": "startTime"}),
}
_PF_LIMIT, _PF_BAND = 0.99999, 2.5e-6       # MBL InductiveLoad: smoothMin(pf, 0.99999, 2.5e-6)
_LOAD_EPS = 1e-10                           # MBL Interfaces/Load.mo: load = eps + (1 - eps) y
_MODE = "Buildings.Electrical.Types.Load."
_TERMINAL = re.compile(r"^(?P<inst>[^.]+)\.(?P<term>terminal(?:_n|_p)?)\."
                       r"(?P<var>[vi])\[(?P<k>[12])\]$")


# ------------------------------------------------------------------------------ names
@dataclass(frozen=True)
class Column:
    """A compared value: `part(sum(c * history[key][..., i] for i, c in terms))`, with `part`
    `"real"` or `"imag"` (of a phasor; the real part of a real value is itself)."""

    key: str
    terms: tuple[tuple[int, float], ...]
    part: str = "real"


@dataclass(frozen=True, eq=False)
class CircuitNames:
    """What the reader built, by name.

    `times` is the output grid (s), `grid` the step grid; `restarts` holds the indices `k`
    whose step `grid[k] -> grid[k+1]` is backward Euler (0 and every signal jump). `dynamic`
    is true when a law stores state (a capacitor, an inductor), `phasor` for a phasor model.
    `nodes` / `thermal_nodes` map a node name to its position in the layers' `phi` (the
    network's node order); `devices` maps a device to `(kind, positions of its edges in the
    layer's q)` (the kind of its first edge: a `TwoPortRCLine` with `use_C` lists its two
    `impedance` edges, then its `capacitor` edge); `sources` maps a voltage source to its
    position among the sources, the column of `"series:electrical.voltage_source.value"` in
    a real model; `columns` maps every compared column to its `Column`.
    """

    model: str
    times: Tensor
    grid: Tensor
    restarts: frozenset[int]
    dynamic: bool
    phasor: bool
    nodes: dict[str, int]
    thermal_nodes: dict[str, int]
    devices: dict[str, tuple[str, tuple[int, ...]]]
    sources: dict[str, int]
    columns: dict[str, Column]

    def value(self, history: Mapping[str, Tensor], column: str) -> Tensor:
        """The column's value from `history` (entries `(T, ..., n)`, stacked over time):
        `(T, ...)`, float64."""
        if column not in self.columns:
            raise KeyError(f"no compared column {column!r}")
        col = self.columns[column]
        x = history[col.key]
        total = sum((c * x[..., i] for i, c in col.terms), torch.zeros_like(x[..., 0]))
        if total.is_complex():
            return total.imag if col.part == "imag" else total.real
        return torch.zeros_like(total) if col.part == "imag" else total


# ------------------------------------------------------------------------------ inputs
# What evaluating a signal block can raise: the evaluator's own refusals, and the errors a
# malformed parameter (a table column out of range, a non-numeric value) produces in it.
_EVALUATION_ERRORS = (ModelicaImportError, ValueError, TypeError, IndexError, KeyError,
                      ArithmeticError)


def _evaluation_problem(sig: Signal, exc: Exception) -> str:
    """The problem text for a signal that could not be evaluated, naming the signal."""
    if isinstance(exc, ModelicaImportError):
        return str(exc).removeprefix("modelica: ")
    return f"{sig.name} ({sig.cls}): cannot be evaluated ({type(exc).__name__}: {exc})"


class _Inputs:
    """The devices' signal inputs (`"<device>.<input>"`) from the document's signals.

    A `Modelica.Blocks.Math` block takes each input's value from the signal that drives
    `"<block>.<input>"`, recursively, so a chain is evaluated from its sources outwards; a
    block input nothing drives and a loop of blocks are refused by name (the building
    reader's rules).
    """

    def __init__(self, sigs: tuple[Signal, ...], grid: Tensor, problems: list[str]) -> None:
        self.grid = grid
        self.problems = problems
        self.by_target: dict[str, Signal] = {}
        self.used: set[str] = set()
        self.names = {s.name for s in sigs}
        self._values: dict[str, Tensor | None] = {}
        for s in sigs:
            for target in s.drives:
                if target in self.by_target:
                    problems.append(f"{s.name} ({s.cls}): drives {target}, which "
                                    f"{self.by_target[target].name} already drives")
                    continue
                self.by_target[target] = s

    def series(self, device: Device, port: str) -> Tensor | None:
        """The input's value at every grid time `(n_grid,)`; `None` (with a problem) when
        nothing drives it or it cannot be evaluated."""
        target = f"{device.name}.{port}"
        s = self.by_target.get(target)
        if s is None:
            self.problems.append(f"{device.name} ({device.cls}): input {port} is enabled but "
                                 f"no supported signal drives it")
            return None
        self.used.add(target)
        return self._series(s, ())

    def _series(self, s: Signal, chain: tuple[str, ...]) -> Tensor | None:
        if s.name in self._values:
            return self._values[s.name]
        if s.name in chain:
            loop = " -> ".join((*chain[chain.index(s.name):], s.name))
            self.problems.append(f"{s.name} ({s.cls}): signal loop {loop} (feedback is not "
                                 f"supported)")
            return None
        y: Tensor | None = None
        try:
            if signals.is_math(s):
                values: dict[str, Tensor] = {}
                for port in signals.math_inputs(s):
                    src = self.by_target.get(f"{s.name}.{port}")
                    if src is None:
                        self.problems.append(f"{s.name} ({s.cls}): input {port} is not driven "
                                             f"by any supported signal")
                        continue
                    self.used.add(f"{s.name}.{port}")
                    value = self._series(src, (*chain, s.name))
                    if value is not None:
                        values[port] = value
                if len(values) == len(signals.math_inputs(s)):
                    y = signals.combine(s, values).expand(self.grid.shape).clone()
            else:
                y = signals.evaluate(s, self.grid, t_start=float(self.grid[0]))
        except _EVALUATION_ERRORS as exc:
            self.problems.append(_evaluation_problem(s, exc))
            y = None
        self._values[s.name] = y
        return y

    def values_at(self, target: str, t: Tensor) -> Tensor:
        """The input's value at arbitrary times `t` (after `series` evaluated it)."""
        s = self.by_target[target]
        if signals.is_math(s):
            values = {port: self.values_at(f"{s.name}.{port}", t)
                      for port in signals.math_inputs(s)}
            return signals.combine(s, values).expand(t.shape)
        return signals.evaluate(s, t)

    def breaks(self, target: str) -> set[float]:
        """The breakpoints of the input's value inside the grid (`signals.breakpoints`)."""
        s = self.by_target[target]
        if signals.is_math(s):
            out: set[float] = set()
            for port in signals.math_inputs(s):
                out |= self.breaks(f"{s.name}.{port}")
            return out
        return set(signals.breakpoints(s, float(self.grid[0]), float(self.grid[-1])))

    def check_unused(self) -> None:
        for target, s in self.by_target.items():
            if target in self.used or target.split(".", 1)[0] in self.names:
                continue    # read, or feeding a Math block (reported for that block)
            self.problems.append(f"{s.name} ({s.cls}): drives {target}, which this reader "
                                 f"does not read (the input is disabled or not supported)")


# ------------------------------------------------------------------------------ helpers
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


def _shown(value) -> str:
    """`repr(value)`, shortened for the message (a huge integer has a very long one)."""
    try:
        text = repr(value)
    except ValueError:          # an integer beyond Python's digit limit for str()
        return "a very large integer"
    return text if len(text) <= 40 else f"{text[:20]}...({len(text)} characters)"


def _number(device: Device, key: str, problems: list[str], default: float | None = None
            ) -> float:
    """A finite numeric parameter of `device` (`default` when given and absent); a missing,
    non-numeric or non-finite one is a problem (0.0 stands in for it)."""
    value = device.parameters.get(key, default)
    number = _finite(value)
    if number is None:
        problems.append(f"{device.name} ({device.cls}): parameter {key} must be a finite "
                        f"number, found {_shown(value)}")
        return 0.0
    return number


def _positive(device: Device, key: str, problems: list[str]) -> float | None:
    """A numeric parameter that must be finite and > 0; `None` (with a problem)
    otherwise."""
    value = device.parameters.get(key)
    number = _finite(value)
    if number is None or not number > 0:
        problems.append(f"{device.name} ({device.cls}): parameter {key} must be a finite "
                        f"number > 0, found {_shown(value)}")
        return None
    return number


def _flag(device: Device, key: str, problems: list[str]) -> bool:
    value = device.parameters.get(key)
    if not isinstance(value, bool):
        problems.append(f"{device.name} ({device.cls}): parameter {key} must be true or "
                        f"false, found {value!r}")
        return False
    return value


def _initial(device: Device, key: str, problems: list[str]) -> float:
    if key not in device.initial:
        problems.append(f"{device.name} ({device.cls}): the start value of {key} was not "
                        f"exported")
        return 0.0
    return float(device.initial[key])


def _short(device: Device) -> str:
    return device.cls.rsplit(".", 1)[-1]


@dataclass
class _Edge:
    device: str
    u: int
    v: int
    values: dict


# ------------------------------------------------------------------------------ builder
class _Builder:
    def __init__(self, netlist: Netlist, doc: CircuitDoc, substeps: int) -> None:
        self.nl = netlist
        self.doc = doc
        self.problems: list[str] = []
        self.edges: dict[str, list[_Edge]] = {k: [] for k in _KINDS}
        self.added: list[tuple[str, _Edge]] = []     # (kind, edge) in the order added
        self.source_blocks: list[Signal] = []

        # the step grid: output times plus every event of the signals and source blocks
        for device in netlist.devices:
            if device.role == "voltage_source" and _short(device) in _EQUIVALENT:
                block = self._equivalent(device)
                if block is not None:
                    self.source_blocks.append(block)
        timing = SimpleNamespace(experiment=doc.experiment,
                                 signals=(*netlist.signals, *self.source_blocks))
        self.times = experiment_times(doc)
        self.grid = driver_grid(timing, substeps)
        jumps, _ = grid_events(timing, self.grid)
        self.restarts = frozenset({0} | {int(k) for k in
                                         torch.nonzero(torch.isin(self.grid, jumps)).flatten()})
        self.inputs = _Inputs(netlist.signals, self.grid, self.problems)
        self.n_grid = self.grid.numel()

    # ---------------------------------------------------------------- sources
    def _equivalent(self, device: Device) -> Signal | None:
        block, mapping = _EQUIVALENT[_short(device)]
        parameters = {}
        for key, src in mapping.items():
            if src in device.parameters:
                parameters[key] = device.parameters[src]
        if "V" not in device.parameters:
            self.problems.append(f"{device.name} ({device.cls}): parameter V must be a "
                                 f"number, found None")
            return None
        return Signal(name=device.name, cls=_BLOCKS + block, parameters=parameters,
                      drives=(f"{device.name}.v",))

    def _source_series(self, device: Device) -> Tensor | None:
        """An MSL source's value at every grid time."""
        short = _short(device)
        if short in _EQUIVALENT:
            block = next((b for b in self.source_blocks if b.name == device.name), None)
            if block is None:
                return None
            try:
                return signals.evaluate(block, self.grid, t_start=float(self.grid[0]))
            except _EVALUATION_ERRORS as exc:
                self.problems.append(_evaluation_problem(block, exc))
                return None
        if short == "SignalVoltage":
            return self.inputs.series(device, "v")
        if short in _VARIABLE_FREQUENCY:
            return self._variable_frequency(device, cosine=short.startswith("Cosine"))
        self.problems.append(f"{device.name} ({device.cls}): voltage source is not "
                             f"supported")  # pragma: no cover - every role class is handled
        return None

    def _variable_frequency(self, device: Device, *, cosine: bool) -> Tensor | None:
        """`offset + V sin(phi)` (`cos`), `phi = 2 pi` times the exact integral of `f`."""
        p = self.problems
        offset = _number(device, "offset", p)
        dt = self.grid[1:] - self.grid[:-1]
        if _flag(device, "useConstantFrequency", p):
            means = torch.full_like(dt, _number(device, "constantFrequency", p))
        else:
            if self.inputs.series(device, "f") is None:
                return None
            target = f"{device.name}.f"
            means = signals.interval_means(lambda t: self.inputs.values_at(target, t),
                                           self.grid, self.inputs.breaks(target))[1:]
        phi = 2.0 * math.pi * torch.cat([torch.zeros(1, dtype=F64), torch.cumsum(means * dt, 0)])
        if _flag(device, "useConstantAmplitude", p):
            amplitude = _number(device, "constantAmplitude", p)
        else:
            amplitude = self.inputs.series(device, "V")
            if amplitude is None:
                return None
        return offset + amplitude * (torch.cos(phi) if cosine else torch.sin(phi))

    # ---------------------------------------------------------------- devices
    def add(self, kind: str, device: Device, u: int, v: int, **values) -> None:
        edge = _Edge(device.name, u, v, values)
        self.edges[kind].append(edge)
        self.added.append((kind, edge))

    def device(self, d: Device) -> None:
        p, n, role = self.problems, d.nodes, d.role
        if role == "voltage_source":
            series = self._source_series(d) if not self.nl.phasor else None
            self.add("voltage_source", d, n["p"], n["n"], series=series, amplitude=0.0)
        elif role == "dc_source":
            V = _number(d, "V", p)
            self.add("voltage_source", d, n["c1"], 0,
                     series=torch.full((self.n_grid,), V, dtype=F64), amplitude=complex(V))
        elif role == "ac_source":
            V, phase = _number(d, "V", p), _number(d, "phiSou", p, 0.0)
            self.add("voltage_source", d, n["a"], 0, series=None,
                     amplitude=V * complex(math.cos(phase), math.sin(phase)))
        elif role == "resistor":
            self._resistor(d)
        elif role == "diode":
            values = {k: _number(d, k, p) for k in
                      ("Ids", "Vt", "Maxexp", "R", "N", "TNOM", "XTI", "EG", "T")}
            values["dependent"] = _flag(d, "useTemperatureDependency", p)
            self.add("diode", d, n["p"], n["n"], **values)
        elif role == "capacitor":
            self.add("capacitor", d, n["p"], n["n"], C=_number(d, "C", p),
                     v0=_initial(d, "v", p))
        elif role == "inductor":
            self.add("inductor", d, n["p"], n["n"], L=_number(d, "L", p),
                     i0=_initial(d, "i", p))
        elif role == "mutual_inductance":
            values = {k: _number(d, k, p) for k in ("L1", "L2", "M")}
            i1, i2 = _initial(d, "i1", p), _initial(d, "i2", p)
            self.add("transformer", d, n["p1"], n["n1"], i0=i1, **values)
            self.add("transformer", d, n["p2"], n["n2"], i0=i2, **values)
        elif role == "ideal_transformer":
            ratio = _number(d, "n", p)
            self.add("ideal_transformer", d, n["p1"], n["n1"], n=ratio)
            self.add("ideal_transformer", d, n["p2"], n["n2"], n=ratio)
        elif role in ("dc_resistor", "dc_line"):
            u, v = (n["c1"], n["c2"]) if role == "dc_resistor" else (n["p1"], n["n1"])
            self.add("impedance", d, u, v, R=_number(d, "R", p), L=0.0, C=math.inf)
        elif role == "dc_rc_line":
            half = _number(d, "R", p) / 2.0
            self.add("impedance", d, n["p1"], n["mid"], R=half, L=0.0, C=math.inf)
            self.add("impedance", d, n["n1"], n["mid"], R=half, L=0.0, C=math.inf)
            if _flag(d, "use_C", p):
                self.add("capacitor", d, n["mid"], n["c2"], C=_number(d, "C", p),
                         v0=_initial(d, "Vc", p))
        elif role == "ac_impedance":
            self._ac_impedance(d)
        elif role == "ac_line":
            self._ac_line(d)
        elif role in ("dc_load", "ac_load"):
            self._load(d)
        elif role == "ac_transformer":
            v_high, v_low, base = (_positive(d, k, p) for k in ("VHigh", "VLow", "VABase"))
            zperc, x_over_r = _number(d, "Zperc", p), _number(d, "XoverR", p)
            if None in (v_high, v_low, base):
                return      # reported: no arithmetic on a missing or non-positive rating
            zp = v_high ** 2 * zperc / base
            angle = math.atan(x_over_r)
            z = zp * complex(math.cos(angle), math.sin(angle))
            ratio = v_high / v_low
            self.add("impedance_transformer", d, n["n"], 0, n=ratio, Z=z)
            self.add("impedance_transformer", d, n["p"], 0, n=ratio, Z=z)
        elif role == "ac_dc_converter":
            k, eta = _number(d, "conversionFactor", p), _number(d, "eta", p)
            self.add("power_converter", d, n["ac"], 0, k=k, eta=eta)
            self.add("power_converter", d, n["p1"], n["p2"], k=k, eta=eta)
        # thermal roles: the thermal network is assembled separately

    def _resistor(self, d: Device) -> None:
        p = self.problems
        values = {k: _number(d, k, p) for k in ("alpha", "T_ref")}
        values["T"] = _number(d, "T", p)
        if _short(d) == "VariableResistor":
            values.update(R=1.0, scale=self.inputs.series(d, "R"))
        else:
            values.update(R=_number(d, "R", p), scale=None)
        self.add("resistor", d, d.nodes["p"], d.nodes["n"], **values)

    def _ac_impedance(self, d: Device) -> None:
        p = self.problems
        for key in ("use_R_in", "use_L_in", "use_C_in"):
            if d.parameters.get(key) is True:
                p.append(f"{d.name} ({d.cls}): {key} = true: impedances following a signal "
                         f"are not supported")
        R = _number(d, "R", p)
        if _flag(d, "inductive", p):
            L, C = _number(d, "L", p), math.inf
        else:
            L, C = 0.0, _number(d, "C", p)
            if not C > 0.0:
                p.append(f"{d.name} ({d.cls}): a capacitive impedance needs C > 0, found "
                         f"C = {C!r}")
                C = math.inf
        if d.parameters.get("star", True) is False:
            R, L, C = R / 3.0, L / 3.0, C * 3.0
        self.add("impedance", d, d.nodes["a"], 0, R=R, L=L, C=C)

    def _ac_line(self, d: Device) -> None:
        p, n, short = self.problems, d.nodes, _short(d)
        if short == "TwoPortInductance":
            R, L = 0.0, _number(d, "L", p)
        elif short == "TwoPortResistance":
            R, L = _number(d, "R", p), 0.0
        else:
            R, L = _number(d, "R", p), _number(d, "L", p)
        if "mid" not in n:
            if short == "Line":           # one-phase Line: TwoPortRL(R/3, L/3)
                R, L = R / 3.0, L / 3.0
            self.add("impedance", d, n["p"], n["n"], R=R, L=L, C=math.inf)
            return
        C = _number(d, "C", p)
        self.add("impedance", d, n["p"], n["mid"], R=R / 2.0, L=L / 2.0, C=math.inf)
        self.add("impedance", d, n["n"], n["mid"], R=R / 2.0, L=L / 2.0, C=math.inf)
        if C > 0.0:
            self.add("impedance", d, n["mid"], 0, R=0.0, L=0.0, C=C)

    def _load(self, d: Device) -> None:
        p = self.problems
        mode = str(d.parameters.get("mode", ""))
        P_nominal = _number(d, "P_nominal", p)
        if mode == _MODE + "FixedZ_steady_state":
            P, scale = P_nominal, torch.ones(self.n_grid, dtype=F64)
        elif mode == _MODE + "VariableZ_y_input":
            y = self.inputs.series(d, "y")
            P, scale = P_nominal, None if y is None else _LOAD_EPS + (1.0 - _LOAD_EPS) * y
        elif mode == _MODE + "VariableZ_P_input":
            P, scale = 1.0, self.inputs.series(d, "Pow")
        else:
            p.append(f"{d.name} ({d.cls}): load mode {mode!r} is not supported")
            return
        tangent = 0.0
        if _short(d) == "Inductive":
            if d.parameters.get("use_pf_in") is not False:
                p.append(f"{d.name} ({d.cls}): use_pf_in = {d.parameters.get('use_pf_in')!r}: "
                         f"a power factor input is not supported")
            pf = _number(d, "pf", p)
            if not 0.0 < pf <= 1.0:
                p.append(f"{d.name} ({d.cls}): pf = {pf!r}: a power factor must lie in "
                         f"(0, 1]")
            elif abs(pf - _PF_LIMIT) < _PF_BAND:
                p.append(f"{d.name} ({d.cls}): pf = {pf!r} lies in the band of MBL's "
                         f"smoothMin(pf, {_PF_LIMIT}, {_PF_BAND}), where its blend is not "
                         f"transcribed")
            else:
                tangent = math.tan(math.acos(min(pf, _PF_LIMIT)))
        a = d.nodes["a"] if d.role == "ac_load" else d.nodes["c1"]
        b = 0 if d.role == "ac_load" else d.nodes["c2"]
        self.add("load", d, a, b, P=-P, Q=-P * tangent, V_nominal=_number(d, "V_nominal", p),
                 linearized=_flag(d, "linearized", p), direct_current=d.role == "dc_load",
                 scale=scale)

    # ---------------------------------------------------------------- laws
    def _same(self, kind: str, key: str, what: str):
        values = {e.values[key] for e in self.edges[kind]}
        if len(values) > 1:
            names = ", ".join(sorted({e.device for e in self.edges[kind]}))
            self.problems.append(f"{names}: {what} cannot be mixed in one model")
        return next(iter(values))

    def _series_driver(self, drivers: Drivers, key: str, columns: list[Tensor | None]
                       ) -> None:
        drivers[_SERIES + key] = torch.stack(
            [torch.ones(self.n_grid, dtype=F64) if c is None else c for c in columns], dim=-1)

    def laws(self, drivers: Drivers) -> list:
        laws = []
        for kind in _KINDS:
            edges = self.edges[kind]
            if not edges:
                continue

            def col(key, edges=edges):
                return torch.tensor([e.values[key] for e in edges], dtype=F64)

            if kind == "voltage_source":
                if self.nl.phasor:
                    amplitude = torch.tensor([e.values["amplitude"] for e in edges], dtype=C128)
                    laws.append(PrescribedVoltage(amplitude, waveforms=Constant()))
                else:
                    self._series_driver(drivers, SOURCE_KEY,
                                        [e.values["series"] for e in edges])
                    laws.append(PrescribedVoltage(1.0, waveforms=[
                        SignalWaveform(SOURCE_KEY, i) for i in range(len(edges))]))
            elif kind == "resistor":
                variable = any(e.values["scale"] is not None for e in edges)
                if variable:
                    self._series_driver(drivers, RESISTOR_SCALE,
                                        [e.values["scale"] for e in edges])
                drivers[RESISTOR_T] = col("T")
                laws.append(Resistance(col("R"), alpha=col("alpha"), T_ref=col("T_ref"),
                                       temperature=RESISTOR_T,
                                       scale=RESISTOR_SCALE if variable else None))
            elif kind == "diode":
                dependent = self._same(kind, "dependent",
                                       "diodes with and without useTemperatureDependency")
                if dependent:
                    drivers[DIODE_T] = col("T")
                laws.append(JunctionDiode(col("Ids"), col("Vt"), col("Maxexp"), col("R"),
                                          temperature=DIODE_T if dependent else None,
                                          N=col("N"), TNOM=col("TNOM"), XTI=col("XTI"),
                                          EG=col("EG")))
            elif kind == "capacitor":
                laws.append(Capacitance(col("C"), v0=col("v0")))
            elif kind == "inductor":
                laws.append(Inductance(col("L"), i0=col("i0")))
            elif kind == "transformer":
                laws.append(MutualInductance(col("L1")[::2], col("L2")[::2], col("M")[::2],
                                             i0=col("i0").reshape(-1, 2)))
            elif kind == "ideal_transformer":
                laws.append(TransformerRatio(col("n")[::2]))
            elif kind == "impedance":
                laws.append(Impedance(col("R"), L=col("L"), C=col("C")))
            elif kind == "load":
                linearized = self._same(kind, "linearized", "linearized and nonlinear loads")
                direct = self._same(kind, "direct_current", "direct- and alternating-current "
                                    "loads")
                variable = any(e.values["scale"] is None or
                               bool((e.values["scale"] != 1.0).any()) for e in edges)
                if variable:
                    self._series_driver(drivers, LOAD_SCALE, [e.values["scale"] for e in edges])
                laws.append(ConstantPowerLoad(col("P"), col("Q") if not direct else 0.0,
                                              V_nominal=col("V_nominal"),
                                              linearized=linearized, direct_current=direct,
                                              scale=LOAD_SCALE if variable else None))
            elif kind == "impedance_transformer":
                z = torch.tensor([e.values["Z"] for e in edges[::2]], dtype=C128)
                laws.append(ImpedanceTransformer(col("n")[::2], z))
            else:
                laws.append(PowerConverter(col("k")[::2], col("eta")[::2]))
        return laws

    # ---------------------------------------------------------------- start state
    def start(self, laws: list, n_nodes: int, drivers: Drivers) -> tuple[State, bool]:
        dynamic = any(getattr(law, "dynamic", False) for law in laws)
        n_e = len(self.nl.node_names)
        flows = [e for kind in _KINDS for e in self.edges[kind]]
        dtype = C128 if self.nl.phasor else F64
        q = torch.zeros(len(flows), dtype=dtype)
        phi = torch.zeros(n_nodes, dtype=dtype)
        if dynamic:
            k = 0
            for kind in _KINDS:
                for e in self.edges[kind]:
                    if kind in ("inductor", "transformer"):
                        q[k] = e.values["i0"]
                    k += 1
            phi[:n_e] = self._capacitor_potentials(n_e)
        else:
            if self.nl.phasor:
                ref = max((abs(e.values["amplitude"]) for e in self.edges["voltage_source"]),
                          default=0.0)
            else:
                values = drivers.get(_SERIES + SOURCE_KEY)
                ref = 0.0 if values is None else float(values[0].abs().max())
            phi[1:n_e] = ref
        return {f"{LAYER}.phi": phi, f"{LAYER}.q": q}, dynamic

    def _capacitor_potentials(self, n_e: int) -> Tensor:
        adjacent: dict[int, list[tuple[int, float]]] = {}
        for e in self.edges["capacitor"]:
            adjacent.setdefault(e.u, []).append((e.v, -e.values["v0"]))
            adjacent.setdefault(e.v, []).append((e.u, e.values["v0"]))
        phi = torch.zeros(n_e, dtype=F64)
        seen: set[int] = set()
        for root in [0, *sorted(adjacent)]:
            if root in seen:
                continue
            seen.add(root)
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v, step in adjacent.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        phi[v] = phi[u] + step
                        queue.append(v)
        return phi

    # ---------------------------------------------------------------- columns
    def positions(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Device -> (kind of its first edge, positions of its edges in `q`), the edges in
        the order the device added them."""
        index = {id(e): k for k, e in enumerate(e for kind in _KINDS for e in self.edges[kind])}
        out: dict[str, tuple[str, list[int]]] = {}
        for kind, e in self.added:
            out.setdefault(e.device, (kind, []))[1].append(index[id(e)])
        return {name: (kind, tuple(edges)) for name, (kind, edges) in out.items()}

    def column(self, name: str, devices: dict[str, Device], positions, thermal_offset: int
               ) -> Column | None:
        nl = self.nl
        q, phi, thermal = f"{LAYER}.q", f"{LAYER}.phi", "thermal.phi"
        match = _TERMINAL.match(name)
        if match:
            inst, term, var, k = match["inst"], match["term"], match["var"], int(match["k"])
            d = devices.get(inst)
            if d is None:
                return None
            if var == "v":
                if f"{inst}.{term}[{k}]" in nl.node_of:          # DC: one node per conductor
                    return Column(phi, ((nl.node_of[f"{inst}.{term}[{k}]"], 1.0),))
                if f"{inst}.{term}" in nl.node_of:               # AC: one complex node
                    return Column(phi, ((nl.node_of[f"{inst}.{term}"], 1.0),),
                                  "real" if k == 1 else "imag")
                return None
            ac = f"{inst}.{term}" in nl.node_of
            if k == 2 and not ac:
                return None
            found = _terminal_current(d, term)
            if found is None or d.name not in positions:
                return None
            edge, sign = found
            return Column(q, ((positions[d.name][1][edge], sign),),
                          "imag" if k == 2 else "real")
        inst, _, var = name.partition(".")
        d = devices.get(inst)
        if d is None:
            return None
        edges = positions.get(inst, ("", ()))[1]
        two_pin = d.role in ("resistor", "capacitor", "inductor", "diode", "voltage_source")
        two_port = d.role in ("mutual_inductance", "ideal_transformer")
        if var == "i" and two_pin:
            return Column(q, ((edges[0], 1.0),))
        if var in ("i1", "i2") and two_port:
            return Column(q, ((edges[int(var[1]) - 1], 1.0),))
        if var == "v" and two_pin:
            return Column(phi, ((d.nodes["p"], 1.0), (d.nodes["n"], -1.0)))
        if var == "Vc" and d.role == "dc_rc_line":
            return Column(phi, ((d.nodes["mid"], 1.0), (d.nodes["c2"], -1.0)))
        if var == "T_heatPort" and d.heat_node is not None:
            return Column(thermal, ((thermal_offset + d.heat_node, 1.0),))
        if var == "T" and d.role == "heat_capacitor":
            return Column(thermal, ((thermal_offset + d.nodes["port"], 1.0),))
        return None


# Terminal -> (edge of the device, sign of its flow) for MBL terminal currents: a terminal's
# `i` is the current INTO the device there, so `+q` on an edge leaving that terminal's node,
# `-q` on an edge entering it.
_TERMINAL_CURRENTS = {
    ("dc_source", "terminal"): (0, 1.0), ("dc_resistor", "terminal"): (0, 1.0),
    ("dc_load", "terminal"): (0, 1.0),
    ("dc_line", "terminal_n"): (0, 1.0), ("dc_line", "terminal_p"): (0, -1.0),
    ("dc_rc_line", "terminal_n"): (0, 1.0), ("dc_rc_line", "terminal_p"): (1, 1.0),
    ("ac_source", "terminal"): (0, 1.0), ("ac_impedance", "terminal"): (0, 1.0),
    ("ac_load", "terminal"): (0, 1.0),
    ("ac_transformer", "terminal_n"): (0, 1.0), ("ac_transformer", "terminal_p"): (1, 1.0),
    ("ac_dc_converter", "terminal_n"): (0, 1.0), ("ac_dc_converter", "terminal_p"): (1, 1.0),
}


def _terminal_current(d: Device, term: str) -> tuple[int, float] | None:
    if d.role == "ac_line":
        if term == "terminal_n":
            return 0, 1.0
        if term == "terminal_p":
            return (1, 1.0) if "mid" in d.nodes else (0, -1.0)
        return None
    return _TERMINAL_CURRENTS.get((d.role, term))


def build(netlist: Netlist, doc: CircuitDoc, *, substeps: int = 1
          ) -> tuple[Model, State, Drivers, CircuitNames]:
    """Assemble `netlist` (from `doc`) into `(model, state, drivers, names)` (module
    docstring). Every problem found (a missing parameter, an undriven or unread input, an
    unsupported setting, a compared column that maps to nothing) is raised as ONE
    `ModelicaImportError` naming each."""
    b = _Builder(netlist, doc, substeps)
    for device in netlist.devices:
        b.device(device)
    b.inputs.check_unused()
    n_e = len(netlist.node_names)
    positions = b.positions()
    by_name = {d.name: d for d in netlist.devices}
    columns: dict[str, Column] = {}
    for column in doc.compared:
        if column.partition(".")[0] in netlist.omitted:
            continue        # an omitted component's column
        found = b.column(column, by_name, positions, n_e)
        if found is None:
            b.problems.append(f"compared column {column!r} maps to no noodl quantity")
        else:
            columns[column] = found
    raise_problems(b.problems)

    net = Network(dtype=F64)
    names = [*netlist.node_names, *netlist.thermal_names]
    for name in names:
        net.add_node(name)
    for kind in _KINDS:
        for e in b.edges[kind]:
            net.add_edge(names[e.u], names[e.v], kind=kind)

    dtype = C128 if netlist.phasor else F64
    drivers: Drivers = {f"{LAYER}.phi_boundary": torch.zeros(1, dtype=dtype)}
    laws = b.laws(drivers)
    raise_problems(b.problems)
    layer = PotentialFlowLayer(net, LAYER, laws, boundary=["0"], phasor=netlist.phasor,
                               quantity="voltage", unit="V", flow_unit="A")
    if netlist.phasor:
        drivers[layer.frequency_key] = torch.tensor(netlist.frequency, dtype=F64)
    model = Model(net, [layer])
    state, dynamic = b.start(laws, len(names), drivers)
    sources = {e.device: i for i, e in enumerate(b.edges["voltage_source"])}
    circuit_names = CircuitNames(
        model=doc.model, times=b.times, grid=b.grid, restarts=b.restarts, dynamic=dynamic,
        phasor=netlist.phasor,
        nodes={name: k for k, name in enumerate(netlist.node_names)},
        thermal_nodes={name: n_e + k for k, name in enumerate(netlist.thermal_names)},
        devices=positions, sources=sources, columns=columns)
    return model, state, drivers, circuit_names

