"""Electrical circuits: resistors, capacitors, inductors, sources, diodes, switches,
transformers, controlled sources, impedances, constant-power loads and power converters,
solved by Modified Nodal Analysis in a `PotentialFlowLayer` (node voltages in V, branch
currents in A), in DC, transient and phasor (AC) modes. A netlist is a `Circuit` of devices
(`Resistor`, `Capacitor`, `VoltageSource`, ...); `build_model` turns it into a model,
analysed by `operating_point`, `initial_state`, `transient`, `ac_sweep` and
`device_values`. Device laws follow the Modelica Standard
Library (`Modelica.Electrical.Analog`); impedances, constant-power loads and converters
follow the Modelica Buildings Library (`Buildings.Electrical`). `JouleHeating` couples a
circuit to a thermal network (Joule heat out, heat-port temperatures back).
"""

from noodl.apps.electrical.analysis import (
    ac_sweep,
    device_values,
    initial_state,
    operating_point,
    transient,
)
from noodl.apps.electrical.circuit import (
    LAYER,
    Capacitor,
    Circuit,
    Conductor,
    ControlledSource,
    CurrentSource,
    Diode,
    IdealTransformer,
    Inductor,
    Resistor,
    Switch,
    Transformer,
    VoltageSource,
    build_model,
)
from noodl.apps.electrical.converters import ImpedanceTransformer, PowerConverter
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
from noodl.apps.electrical.loads import ConstantPowerLoad, Impedance
from noodl.apps.electrical.thermal import HeatedEdge, JouleHeating
from noodl.apps.electrical.waveforms import Constant, Pulse, Ramp, Signal, Sine, Step

__all__ = [
    "LAYER", "Capacitor", "Circuit", "Conductor", "ControlledSource", "CurrentSource",
    "Diode", "IdealTransformer", "Inductor", "Resistor", "Switch", "Transformer",
    "VoltageSource", "ac_sweep", "build_model", "device_values", "initial_state",
    "operating_point", "transient",
    "Capacitance", "Constant", "ConstantPowerLoad", "DependentSource", "HeatedEdge",
    "Impedance", "ImpedanceTransformer", "Inductance", "JouleHeating", "JunctionDiode",
    "MutualInductance",
    "PowerConverter", "PrescribedCurrent", "PrescribedVoltage", "Pulse", "Ramp",
    "Resistance", "Signal", "Sine", "Step", "SwitchedConductance", "TransformerRatio",
]
