"""The circuit netlist: device dataclasses, `Circuit`, and the (Model, State, Drivers) builder.

A device connects named nodes through its pins. A one-port device `p -> n` is one edge of
the network, with voltage `v = v_p - v_n` (V) and current `i` entering pin `p` (A); a
two-port device (transformers, controlled sources) is two consecutive edges, port 1
`p1 -> n1` then port 2 `p2 -> n2`. The ground node (default `"0"`) is the layer's only
boundary node, held at 0 V.

The builder groups the devices by kind, one constitutive law per kind carrying all of its
devices with per-device parameters, in a single `KirchhoffLayer` named `"electrical"`
solved by Modified Nodal Analysis. Device parameters are scalars (a float or a 0-d tensor);
gradients reach the laws' parameters (`build_model(..., learnable=True)`), and an ensemble
of circuits is a batch of drivers (a `Signal` source, a switch control, a resistor
temperature), not of device parameters.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from noodl.apps.electrical.laws import (
    Capacitance,
    DependentSource,
    Inductance,
    JunctionDiode,
    MutualInductance,
    PrescribedCurrent,
    PrescribedVoltage,
    Resistance,
    SwitchedConductance,
    TransformerRatio,
)
from noodl.apps.electrical.waveforms import Constant, _Waveform
from noodl.elements import Conductance, ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: The name of the circuit layer: its state keys are `"electrical.phi"` (node voltages, V)
#: and `"electrical.q"` (device currents, A); its drivers `"electrical.phi_boundary"` (the
#: ground potential) and, for a phasor model, `"electrical.frequency"` (Hz).
LAYER = "electrical"


# ----------------------------------------------------------------------------- devices
@dataclass(frozen=True)
class Resistor:
    """A linear resistor `v = R_actual i`, `R_actual = R (1 + alpha (T - T_ref))` (`R` in
    ohm at `T_ref` in K, `alpha` in 1/K); `T` comes from the builder's `temperature` driver,
    else `T = T_ref`. MSL `Basic.Resistor`."""

    name: str
    p: str
    n: str
    R: float
    alpha: float = 0.0
    T_ref: float = 300.15


@dataclass(frozen=True)
class Conductor:
    """A linear conductance `i = G v` (`G` in S). MSL `Basic.Conductor`."""

    name: str
    p: str
    n: str
    G: float


@dataclass(frozen=True)
class Capacitor:
    """A capacitor `i = C dv/dt` (`C` in F), with initial voltage `v0` (V). MSL
    `Basic.Capacitor`."""

    name: str
    p: str
    n: str
    C: float
    v0: float = 0.0


@dataclass(frozen=True)
class Inductor:
    """An inductor `v = L di/dt` (`L` in H), with initial current `i0` (A). MSL
    `Basic.Inductor`."""

    name: str
    p: str
    n: str
    L: float
    i0: float = 0.0


@dataclass(frozen=True)
class VoltageSource:
    """An ideal voltage source `v = offset + V unit(t)` (V), `unit` the waveform. In an AC
    analysis a `Sine` source is its steady-state phasor (amplitude `V`, the waveform's phase)
    at the analysis frequency, and a `Constant` source is the magnitude `V` at phase 0; other
    waveforms are refused there. MSL `Sources.*Voltage`."""

    name: str
    p: str
    n: str
    V: float
    waveform: _Waveform = Constant()
    offset: float = 0.0


@dataclass(frozen=True)
class CurrentSource:
    """An ideal current source `i = offset + I unit(t)` (A) entering pin `p` (flowing
    through the source from `p` to `n`); waveform and AC phasor as for `VoltageSource`. MSL
    `Sources.*Current`."""

    name: str
    p: str
    n: str
    I: float  # noqa: E741 - the MSL and textbook name of a source current
    waveform: _Waveform = Constant()
    offset: float = 0.0


@dataclass(frozen=True)
class Diode:
    """A junction diode `i = Ids (exlin(v / Vt) - 1) + v / R` (`Ids` in A, `Vt` in V, `R`
    the parallel leak in ohm; exponential up to `Maxexp`, linear above). MSL
    `Semiconductors.Diode`. Not usable in an AC analysis."""

    name: str
    p: str
    n: str
    Ids: float = 1e-6
    Vt: float = 0.04
    Maxexp: float = 15.0
    R: float = 1e8


@dataclass(frozen=True)
class Switch:
    """An ideal closing switch: closed `v = Ron i`, open `i = Goff v` (`Ron` in ohm, `Goff`
    in S); closed where its control value exceeds 0.5. `control` is a driver key (read at
    index 0 of its last dimension) or a `(key, index)` pair. MSL
    `Ideal.IdealClosingSwitch`."""

    name: str
    p: str
    n: str
    control: str | tuple[str, int]
    Ron: float = 1e-5
    Goff: float = 1e-5


@dataclass(frozen=True)
class IdealTransformer:
    """An ideal transformer `v1 = n v2`, `i2 = -n i1` (turns ratio `n`). MSL
    `Ideal.IdealTransformer` without magnetisation."""

    name: str
    p1: str
    n1: str
    p2: str
    n2: str
    n: float


@dataclass(frozen=True)
class Transformer:
    """Two coupled windings `v1 = L1 di1/dt + M di2/dt`, `v2 = M di1/dt + L2 di2/dt` (H),
    with initial currents `i0 = (i1, i2)` (A). MSL `Basic.Transformer`."""

    name: str
    p1: str
    n1: str
    p2: str
    n2: str
    L1: float
    L2: float
    M: float
    i0: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ControlledSource:
    """A linear controlled source: port 1 senses (`control="voltage"`: an open circuit;
    `"current"`: a short) and port 2 imposes (`output="voltage"` or `"current"`) `gain`
    times the sensed quantity. MSL `Basic.VCV`, `VCC`, `CCV`, `CCC`."""

    name: str
    p1: str
    n1: str
    p2: str
    n2: str
    gain: float
    control: str
    output: str


_ONE_PORT = (Resistor, Conductor, Capacitor, Inductor, VoltageSource, CurrentSource, Diode,
             Switch)
_TWO_PORT = (IdealTransformer, Transformer, ControlledSource)
_DEVICES = _ONE_PORT + _TWO_PORT

#: The scalar parameters of each device type (checked at `Circuit` construction).
_SCALARS = {
    Resistor: ("R", "alpha", "T_ref"), Conductor: ("G",), Capacitor: ("C", "v0"),
    Inductor: ("L", "i0"), VoltageSource: ("V", "offset"), CurrentSource: ("I", "offset"),
    Diode: ("Ids", "Vt", "Maxexp", "R"), Switch: ("Ron", "Goff"), IdealTransformer: ("n",),
    Transformer: ("L1", "L2", "M"), ControlledSource: ("gain",),
}

#: The law kind of each device type: the default kind of its constitutive law.
_KINDS = {
    Resistor: "resistor", Conductor: "conduction", Capacitor: "capacitor",
    Inductor: "inductor", VoltageSource: "voltage_source", CurrentSource: "current_source",
    Diode: "diode", Switch: "switch", IdealTransformer: "ideal_transformer",
    Transformer: "transformer",
}

_SENSED = ("voltage", "current")


def _kind(device) -> str:
    """The law kind a device belongs to (`"vcvs"`, `"cccs"`, ... for controlled sources)."""
    if isinstance(device, ControlledSource):
        return f"{device.control[0]}c{device.output[0]}s"
    return _KINDS[type(device)]


def _ports(device) -> list[tuple[str, str]]:
    """The `(from, to)` node pairs of a device's edges, in edge order."""
    if isinstance(device, _TWO_PORT):
        return [(device.p1, device.n1), (device.p2, device.n2)]
    return [(device.p, device.n)]


def _pin_names(device) -> list[str]:
    return ["p1", "n1", "p2", "n2"] if isinstance(device, _TWO_PORT) else ["p", "n"]


def _is_scalar(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return True
    return (isinstance(value, Tensor) and value.dim() == 0 and not value.is_complex()
            and not value.requires_grad)


# ----------------------------------------------------------------------------- circuit
@dataclass(frozen=True)
class Circuit:
    """A netlist: `devices` (the dataclasses of this module) between named nodes, and the
    `ground` node (0 V). Construction checks the device types, unique non-empty names, node
    names on every pin (distinct within a port), scalar parameters, waveforms, switch
    controls and controlled-source modes, that `ground` is a pin, and that every node is
    joined to `ground` through the devices' port edges (a connectivity check only: a loop of
    voltage sources or a cut set of current sources passes it and is refused at the solve)."""

    devices: Sequence
    ground: str = "0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        seen: set[str] = set()
        for d in self.devices:
            if not isinstance(d, _DEVICES):
                raise TypeError(
                    f"Circuit: {d!r} is not a circuit device (one of "
                    f"{[t.__name__ for t in _DEVICES]})"
                )
            if not isinstance(d.name, str) or not d.name:
                raise TypeError(f"Circuit: {d!r} needs a device name (a non-empty str)")
            if d.name in seen:
                raise ValueError(f"Circuit: duplicate device name {d.name!r}")
            seen.add(d.name)
            _check_device(d)
        nodes = self.nodes
        if not any(self.ground in pair for d in self.devices for pair in _ports(d)):
            raise ValueError(f"Circuit: ground {self.ground!r} is not a pin of any device")
        self._check_grounded(nodes)

    @property
    def nodes(self) -> list[str]:
        """The node names: `ground` first, then the others in order of first appearance."""
        out = {self.ground: None}
        for d in self.devices:
            for pair in _ports(d):
                out.update(dict.fromkeys(pair))
        return list(out)

    def _check_grounded(self, nodes: list[str]) -> None:
        parent = {v: v for v in nodes}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for d in self.devices:
            for u, v in _ports(d):
                parent[find(v)] = find(u)
        floating = [v for v in nodes if find(v) != find(self.ground)]
        if floating:
            names = [d.name for d in self.devices
                     if any(v in floating for pair in _ports(d) for v in pair)]
            raise ValueError(
                f"Circuit: nodes {floating} (devices {names}) have no path to ground "
                f"{self.ground!r}, so their voltages are undetermined; connect them to ground "
                f"(through a resistor, for an isolated transformer winding)"
            )


def _check_device(d) -> None:
    """Per-device checks, each error naming the device."""
    where = f"Circuit: device {d.name!r}"
    for pin in _pin_names(d):
        node = getattr(d, pin)
        if not isinstance(node, str) or not node:
            raise TypeError(f"{where}: pin {pin} must be a node name (a non-empty str), got "
                            f"{node!r}")
    pins = _pin_names(d)
    for k, (u, v) in enumerate(_ports(d)):
        if u == v:
            raise ValueError(f"{where}: pins {pins[2 * k]} and {pins[2 * k + 1]} are both on "
                             f"node {u!r}")
    for name in _SCALARS[type(d)]:
        if not _is_scalar(getattr(d, name)):
            raise ValueError(
                f"{where}: {name} must be a scalar (a float or a 0-d tensor without "
                f"gradient), got {getattr(d, name)!r}; batch a circuit through its drivers "
                f"(a Signal source, a switch control, a resistor temperature) and take "
                f"gradients through build_model(..., learnable=True)"
            )
    if isinstance(d, (VoltageSource, CurrentSource)) and not isinstance(d.waveform, _Waveform):
        raise TypeError(
            f"{where}: waveform {d.waveform!r} is not a waveform (Constant, Step, Ramp, Sine, "
            f"Pulse or Signal from noodl.apps.electrical)"
        )
    if isinstance(d, Switch) and not (
        isinstance(d.control, str)
        or (isinstance(d.control, tuple) and len(d.control) == 2
            and isinstance(d.control[0], str) and isinstance(d.control[1], int))
    ):
        raise TypeError(f"{where}: control {d.control!r} must be a driver key or a "
                        f"(key, index) pair")
    if isinstance(d, ControlledSource):
        for mode in ("control", "output"):
            if getattr(d, mode) not in _SENSED:
                raise ValueError(f"{where}: {mode}={getattr(d, mode)!r}; expected one of "
                                 f"{_SENSED}")
    if isinstance(d, Transformer) and not (
        isinstance(d.i0, (tuple, list)) and len(d.i0) == 2 and all(map(_is_scalar, d.i0))
    ):
        raise ValueError(f"{where}: i0 must be a pair of scalar initial currents (i1, i2), "
                         f"got {d.i0!r}")


# ----------------------------------------------------------------------------- builder
def _values(devices, name: str) -> Tensor:
    """One float64 value per device."""
    return torch.stack([torch.as_tensor(getattr(d, name), dtype=F64) for d in devices])


def _law(kind: str, devices: list, temperature: str | None, learnable: bool) -> ConstitutiveLaw:
    """The constitutive law carrying every device of one kind, parameters in device order."""
    first = devices[0]
    if isinstance(first, Resistor):
        return Resistance(_values(devices, "R"), alpha=_values(devices, "alpha"),
                          T_ref=_values(devices, "T_ref"), temperature=temperature,
                          learnable=learnable)
    if isinstance(first, Conductor):
        return Conductance(_values(devices, "G"), learnable=learnable)
    if isinstance(first, Capacitor):
        return Capacitance(_values(devices, "C"), v0=_values(devices, "v0"),
                           learnable=learnable)
    if isinstance(first, Inductor):
        return Inductance(_values(devices, "L"), i0=_values(devices, "i0"),
                          learnable=learnable)
    if isinstance(first, VoltageSource):
        return PrescribedVoltage(_values(devices, "V"), offset=_values(devices, "offset"),
                                 waveforms=[d.waveform for d in devices], learnable=learnable)
    if isinstance(first, CurrentSource):
        return PrescribedCurrent(_values(devices, "I"), offset=_values(devices, "offset"),
                                 waveforms=[d.waveform for d in devices], learnable=learnable)
    if isinstance(first, Diode):
        return JunctionDiode(_values(devices, "Ids"), _values(devices, "Vt"),
                             _values(devices, "Maxexp"), _values(devices, "R"))
    if isinstance(first, Switch):
        return SwitchedConductance([d.control for d in devices], Ron=_values(devices, "Ron"),
                                   Goff=_values(devices, "Goff"))
    if isinstance(first, IdealTransformer):
        return TransformerRatio(_values(devices, "n"), learnable=learnable)
    if isinstance(first, Transformer):
        i0 = torch.stack([torch.stack([torch.as_tensor(v, dtype=F64) for v in d.i0])
                          for d in devices])
        return MutualInductance(_values(devices, "L1"), _values(devices, "L2"),
                                _values(devices, "M"), i0=i0, learnable=learnable)
    return DependentSource(_values(devices, "gain"), control=first.control,
                           output=first.output, kind=kind, learnable=learnable)


def build_model(
    circuit: Circuit, *, phasor: bool = False, frequency=None, temperature: str | None = None,
    learnable: bool = False,
) -> tuple[Model, State, Drivers]:
    """Assemble the circuit model and return `(model, state, drivers)`.

    ONE `KirchhoffLayer` named `LAYER` on a float64 `Network` whose nodes are
    `circuit.nodes` (ground first) and whose edges are added device by device (a two-port
    device's two ports consecutively), with one constitutive law per device kind (kinds in
    order of first appearance) and `circuit.ground` as the only boundary node. The layer's
    flows `"electrical.q"` are ordered by kind, then by device; `model.circuit_index` maps
    each device name to `(kind, slice)`, the slice of its edges in that order.

    `phasor=True` builds the AC (sinusoidal steady-state) model at `frequency` (Hz, a
    scalar or one value per instance), returned as the driver `"electrical.frequency"`; the
    layer's frequency governs, a `Sine` source's own frequency is not used there. A
    frequency without `phasor=True` is refused. `temperature` names a driver holding the
    resistor temperatures (K, the last dimension one per resistor in device order) for the
    resistors' `alpha`; without it every resistor is at its `T_ref`. `learnable=True`
    registers the laws' principal parameters (resistance, conductance, capacitance,
    inductance, source amplitudes, turns ratio, `L1`/`L2`/`M`, gain; one entry per device)
    as `nn.Parameter`s, reached as `model.kirchhoff["electrical"].law_for(kind)[0]`.

    Drivers: `"electrical.phi_boundary"` (the ground at 0 V; complex for a phasor model) and
    `"electrical.frequency"` for a phasor model. State: `{}`; the analyses
    (`operating_point`, `initial_state`, `transient`) produce it.
    """
    if not isinstance(circuit, Circuit):
        raise TypeError(f"build_model: expected a Circuit, got {type(circuit).__name__}")
    if phasor and frequency is None:
        raise ValueError("build_model: a phasor model needs frequency= (Hz)")
    if not phasor and frequency is not None:
        raise ValueError("build_model: frequency= is only used by a phasor model; pass "
                         "phasor=True")
    net = Network(dtype=F64)
    for node in circuit.nodes:
        net.add_node(node)
    groups: dict[str, list] = {}
    for d in circuit.devices:
        kind = _kind(d)
        groups.setdefault(kind, []).append(d)
        for u, v in _ports(d):
            net.add_edge(u, v, kind=kind)
    laws = [_law(kind, devices, temperature, learnable) for kind, devices in groups.items()]
    layer = KirchhoffLayer(net, LAYER, laws, boundary=[circuit.ground], phasor=phasor,
                           quantity="voltage", unit="V", flow_unit="A")
    model = Model(net, [layer])
    index: dict[str, tuple[str, slice]] = {}
    for kind, devices in groups.items():
        start = layer.kind_slice(kind).start
        for d in devices:
            width = len(_ports(d))
            index[d.name] = (kind, slice(start, start + width))
            start += width
    model.circuit_index = index
    dtype = torch.complex128 if phasor else F64
    drivers: Drivers = {f"{LAYER}.phi_boundary": torch.zeros(1, dtype=dtype)}
    if phasor:
        drivers[layer.frequency_key] = torch.as_tensor(frequency, dtype=F64)
    return model, {}, drivers


__all__ = [
    "LAYER", "Capacitor", "Circuit", "Conductor", "ControlledSource", "CurrentSource",
    "Diode", "IdealTransformer", "Inductor", "Resistor", "Switch", "Transformer",
    "VoltageSource", "build_model",
]
