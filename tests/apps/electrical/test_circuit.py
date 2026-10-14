"""The electrical netlist and its model builder: device validation, the circuit index, DC
operating points of resistive circuits, and batching through drivers."""

import numpy as np
import pytest
import torch

from noodl.apps.electrical import (
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
    Signal,
    Step,
    Switch,
    Transformer,
    VoltageSource,
    build_model,
    device_values,
    operating_point,
)

F64 = torch.float64
PHI, Q = f"{LAYER}.phi", f"{LAYER}.q"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _v(model, state, node):
    return state[PHI][..., model.net.nodes.index(node)]


# ------------------------------------------------------------------ DC operating points
def test_divider():
    circuit = Circuit([VoltageSource("V1", "in", "0", 10.0), Resistor("R1", "in", "mid", 3.0),
                       Resistor("R2", "mid", "0", 7.0)])
    model, state, drivers = build_model(circuit)
    assert state == {}
    op = operating_point(model, drivers)
    torch.testing.assert_close(_v(model, op, "mid"), torch.tensor(7.0), rtol=0, atol=1e-12)
    v, i = device_values(model, op)["V1"]
    torch.testing.assert_close(v, torch.tensor(10.0), rtol=0, atol=1e-12)
    # a source delivering power carries a negative current (current entering pin p)
    torch.testing.assert_close(i, torch.tensor(-1.0), rtol=0, atol=1e-12)


def _bridge(R1, R2, R3, R4, R5, V=12.0):
    return Circuit([
        VoltageSource("V", "top", "0", V),
        Resistor("R1", "top", "a", R1), Resistor("R2", "a", "0", R2),
        Resistor("R3", "top", "b", R3), Resistor("R4", "b", "0", R4),
        Resistor("R5", "a", "b", R5),
    ])


def test_wheatstone_bridge_balanced_carries_no_bridge_current():
    model, _, drivers = build_model(_bridge(100.0, 200.0, 50.0, 100.0, 33.0))
    op = operating_point(model, drivers)
    _, i5 = device_values(model, op)["R5"]
    torch.testing.assert_close(i5, torch.tensor(0.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(_v(model, op, "a"), torch.tensor(8.0), rtol=0, atol=1e-12)


def test_wheatstone_bridge_unbalanced_matches_hand_nodal_analysis():
    R1, R2, R3, R4, R5, V = 100.0, 150.0, 50.0, 220.0, 33.0, 12.0
    model, _, drivers = build_model(_bridge(R1, R2, R3, R4, R5, V))
    op = operating_point(model, drivers)
    # KCL at a and b with top held at V
    G = torch.tensor([[1 / R1 + 1 / R2 + 1 / R5, -1 / R5],
                      [-1 / R5, 1 / R3 + 1 / R4 + 1 / R5]])
    va, vb = torch.linalg.solve(G, torch.tensor([V / R1, V / R3]))
    torch.testing.assert_close(_v(model, op, "a"), va, rtol=0, atol=1e-12)
    torch.testing.assert_close(_v(model, op, "b"), vb, rtol=0, atol=1e-12)
    _, i5 = device_values(model, op)["R5"]
    torch.testing.assert_close(i5, (va - vb) / R5, rtol=0, atol=1e-12)


def test_conductor_current_source_and_floating_voltage_source():
    # 2 A into node a through a 0.5 S conductor; a 3 V source between a and b (no ground)
    circuit = Circuit([CurrentSource("I1", "0", "a", 2.0), Conductor("G1", "a", "0", 0.5),
                       VoltageSource("V1", "b", "a", 3.0), Resistor("R1", "b", "0", 1e9)])
    model, _, drivers = build_model(circuit)
    op = operating_point(model, drivers)
    va = _v(model, op, "a")
    torch.testing.assert_close(_v(model, op, "b") - va, torch.tensor(3.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(0.5 * va + (va + 3.0) / 1e9, torch.tensor(2.0), rtol=0,
                               atol=1e-12)


# ---------------------------------------------------------------------- model structure
def test_builder_layer_keys_kinds_and_circuit_index():
    circuit = Circuit([
        VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
        Capacitor("C1", "b", "0", 1e-3), Resistor("R2", "b", "0", 2.0),
        Inductor("L1", "a", "c", 1e-3), Resistor("R3", "c", "0", 1.0),
        IdealTransformer("T1", "a", "0", "s", "0", 2.0), Resistor("RL", "s", "0", 4.0),
        ControlledSource("E1", "b", "0", "e", "0", 2.0, "voltage", "voltage"),
        Resistor("RE", "e", "0", 1.0),
    ])
    model, state, drivers = build_model(circuit)
    layer = model.potential[LAYER]
    assert layer.kinds == ["voltage_source", "resistor", "capacitor", "inductor",
                           "ideal_transformer", "vcvs"]
    assert layer.quantity == "voltage" and layer.unit == "V" and layer.flow_unit == "A"
    assert model.net.nodes[0] == "0"
    assert set(drivers) == {f"{LAYER}.phi_boundary"}
    index = model.circuit_index
    assert index["R2"] == ("resistor", slice(2, 3))
    assert index["R3"] == ("resistor", slice(3, 4))
    assert index["T1"] == ("ideal_transformer", slice(8, 10))
    assert index["E1"] == ("vcvs", slice(10, 12))
    op = operating_point(model, drivers)
    values = device_values(model, op)
    v, i = values["T1"]
    assert v.shape == (2,) and i.shape == (2,)
    torch.testing.assert_close(v, torch.tensor([1.0, 0.5]), rtol=0, atol=1e-12)
    # RL draws 0.5 V / 4 ohm = 0.125 A out of s, so i2 = -0.125 and i1 = -i2 / n
    torch.testing.assert_close(i, torch.tensor([0.0625, -0.125]), rtol=0, atol=1e-12)
    torch.testing.assert_close(values["E1"][0][1], 2.0 * values["C1"][0], rtol=0, atol=1e-12)


def test_controlled_sources_and_coupled_windings_at_dc():
    circuit = Circuit([
        VoltageSource("V1", "a", "0", 4.0),
        # F1: current-controlled current source; its control port b-0 is a short, fed by R1
        Resistor("R1", "a", "b", 1.0),
        ControlledSource("F1", "b", "0", "c", "0", 2.0, "current", "current"),
        Resistor("R2", "c", "0", 1.0),
        # H1: current-controlled voltage source, control fed through R3, output loaded by R4
        Resistor("R3", "a", "d", 2.0),
        ControlledSource("H1", "d", "0", "e", "0", 1.5, "current", "voltage"),
        Resistor("R4", "e", "0", 3.0),
        # G1: voltage-controlled current source sensing a (open), output loaded by R5
        ControlledSource("G1", "a", "0", "f", "0", 0.25, "voltage", "current"),
        Resistor("R5", "f", "0", 2.0),
        # K1: coupled windings, shorts at DC, fed through R6 and R7
        Resistor("R6", "a", "g", 4.0), Resistor("R7", "a", "h", 2.0),
        Transformer("K1", "g", "0", "h", "0", 1e-3, 2e-3, 1e-3),
    ])
    model, _, drivers = build_model(circuit)
    index = model.circuit_index
    assert index["V1"] == ("voltage_source", slice(0, 1))
    assert index["R1"] == ("resistor", slice(1, 2))
    assert index["R7"] == ("resistor", slice(7, 8))
    assert index["F1"] == ("cccs", slice(8, 10))
    assert index["H1"] == ("ccvs", slice(10, 12))
    assert index["G1"] == ("vccs", slice(12, 14))
    assert index["K1"] == ("transformer", slice(14, 16))
    values = device_values(model, operating_point(model, drivers))

    def close(name, v, i):
        got_v, got_i = values[name]
        torch.testing.assert_close(got_v, torch.tensor(v), rtol=0, atol=1e-12)
        torch.testing.assert_close(got_i, torch.tensor(i), rtol=0, atol=1e-12)

    # F1: control current 4 V / 1 ohm = 4 A, output 2 * 4 = 8 A from c to 0, so c = -8 V
    close("F1", [0.0, -8.0], [4.0, 8.0])
    # H1: control current 4 / 2 = 2 A, output 1.5 * 2 = 3 V, R4 draws 1 A out of e
    close("H1", [0.0, 3.0], [2.0, -1.0])
    # G1: senses 4 V with no current, output 0.25 * 4 = 1 A from f to 0, so f = -2 V
    close("G1", [4.0, -2.0], [0.0, 1.0])
    # K1: both windings short at DC: 4 / 4 = 1 A and 4 / 2 = 2 A
    close("K1", [0.0, 0.0], [1.0, 2.0])
    # the source delivers R1 + R3 + R6 + R7 = 4 + 2 + 1 + 2 = 9 A
    close("V1", 4.0, -9.0)
    close("R2", -8.0, -8.0)


def test_numpy_scalars_are_device_parameters():
    circuit = Circuit([VoltageSource("V1", "a", "0", np.float32(2.0)),
                       Resistor("R1", "a", "0", np.float64(4.0))])
    model, _, drivers = build_model(circuit)
    _, i = device_values(model, operating_point(model, drivers))["R1"]
    torch.testing.assert_close(i, torch.tensor(0.5), rtol=0, atol=1e-15)


def test_phasor_model_needs_a_frequency_and_has_complex_drivers():
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "0", 1.0)])
    with pytest.raises(ValueError, match="frequency"):
        build_model(circuit, phasor=True)
    with pytest.raises(ValueError, match="frequency"):
        build_model(circuit, frequency=50.0)
    model, _, drivers = build_model(circuit, phasor=True, frequency=50.0)
    assert drivers[f"{LAYER}.phi_boundary"].dtype == torch.complex128
    assert model.potential[LAYER].frequency_key == f"{LAYER}.frequency"
    torch.testing.assert_close(drivers[f"{LAYER}.frequency"], torch.tensor(50.0))


def test_resistor_temperature_from_a_driver():
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0),
                       Resistor("R1", "a", "0", 2.0, alpha=0.01, T_ref=300.0)])
    model, _, drivers = build_model(circuit, temperature="T_wire")
    op = operating_point(model, {**drivers, "T_wire": torch.tensor([350.0])})
    _, i = device_values(model, op)["R1"]
    torch.testing.assert_close(i, torch.tensor(1.0 / 3.0), rtol=0, atol=1e-12)


def test_switch_reads_its_control_driver():
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
                       Switch("S1", "b", "0", "s"), Switch("S2", "b", "0", ("s", 1))])
    model, _, drivers = build_model(circuit)
    op = operating_point(model, {**drivers, "s": torch.tensor([[0.0, 0.0], [1.0, 0.0]])})
    i1 = device_values(model, op)["R1"][1]
    open_ = 1.0 / (1.0 + 1.0 / 2e-5)          # two open switches in parallel: 2 Goff
    closed = 1.0 / (1.0 + 1.0 / (1e5 + 1e-5))
    # conductances 1 S and 2e-5 S: the condition number (5e4) times eps bounds the agreement
    torch.testing.assert_close(i1, torch.tensor([open_, closed]), rtol=1e-11, atol=0)


# ------------------------------------------------------------------------- validation
def test_duplicate_device_name_is_refused():
    with pytest.raises(ValueError, match="duplicate device name 'R1'"):
        Circuit([Resistor("R1", "a", "0", 1.0), Resistor("R1", "a", "0", 2.0)])


def test_unknown_waveform_type_names_the_device():
    with pytest.raises(TypeError, match="device 'V1'.*'sine' is not a waveform"):
        Circuit([VoltageSource("V1", "a", "0", 1.0, waveform="sine"),
                 Resistor("R1", "a", "0", 1.0)])


def test_ungrounded_transformer_secondary_names_nodes_and_devices():
    with pytest.raises(ValueError, match=r"nodes \['s1', 's2'\].*devices \['T1', 'RL'\].*"
                                         r"no path to ground '0'"):
        Circuit([VoltageSource("V1", "a", "0", 1.0),
                 IdealTransformer("T1", "a", "0", "s1", "s2", 2.0),
                 Resistor("RL", "s1", "s2", 4.0)])


def test_other_netlist_errors_name_the_device():
    with pytest.raises(TypeError, match="not a circuit device"):
        Circuit([("R1", "a", "0", 1.0)])
    with pytest.raises(ValueError, match="device 'R1': pins p and n are both on node 'a'"):
        Circuit([Resistor("R1", "a", "a", 1.0)])
    with pytest.raises(TypeError, match="device 'R1': pin n must be a node name"):
        Circuit([Resistor("R1", "a", 0, 1.0)])
    with pytest.raises(ValueError, match="ground 'gnd' is not a pin of any device"):
        Circuit([Resistor("R1", "a", "0", 1.0)], ground="gnd")
    with pytest.raises(ValueError, match="device 'E1'.*control='power'"):
        Circuit([ControlledSource("E1", "a", "0", "b", "0", 1.0, "power", "voltage")])
    with pytest.raises(TypeError, match="device 'S1'.*control"):
        Circuit([Switch("S1", "a", "0", 3)])
    with pytest.raises(ValueError, match="device 'L1'.*i0"):
        Circuit([Transformer("L1", "a", "0", "b", "0", 1.0, 1.0, 0.5, i0=(0.0,))])
    with pytest.raises(TypeError, match="device name"):
        Circuit([Resistor("", "a", "0", 1.0)])


def test_device_parameters_are_scalars_and_batching_goes_through_drivers():
    with pytest.raises(ValueError, match="device 'R1': R must be a scalar.*drivers"):
        Circuit([VoltageSource("V1", "a", "0", 1.0),
                 Resistor("R1", "a", "0", torch.tensor([1.0, 2.0, 3.0, 4.0]))])
    # an ensemble of four source voltages through a Signal driver of shape (4, 1)
    circuit = Circuit([VoltageSource("V1", "in", "0", 1.0, waveform=Signal("v")),
                       Resistor("R1", "in", "mid", 1.0), Resistor("R2", "mid", "0", 3.0)])
    model, _, drivers = build_model(circuit)
    vs = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    op = operating_point(model, {**drivers, "v": vs})
    n = model.net.n
    assert op[PHI].shape == (4, n)
    torch.testing.assert_close(_v(model, op, "mid"), 0.75 * vs[:, 0], rtol=0, atol=1e-12)


def test_gradients_reach_learnable_law_parameters():
    with pytest.raises(ValueError, match="device 'R2'.*learnable=True"):
        Circuit([VoltageSource("V1", "in", "0", 1.0),
                 Resistor("R2", "in", "0", torch.tensor(3.0, requires_grad=True))])
    circuit = Circuit([VoltageSource("V1", "in", "0", 1.0), Resistor("R1", "in", "mid", 1.0),
                       Resistor("R2", "mid", "0", 3.0)])
    model, _, drivers = build_model(circuit, learnable=True)
    law, _ = model.potential[LAYER].element_for("resistor")
    v = _v(model, operating_point(model, drivers), "mid")
    (g,) = torch.autograd.grad(v, law.R)
    # v = R2 / (R1 + R2): dv/dR1 = -R2 / 16, dv/dR2 = R1 / 16
    torch.testing.assert_close(g, torch.tensor([-3.0 / 16.0, 1.0 / 16.0]), rtol=1e-10, atol=0)


def test_diode_circuit_and_time_dependent_source_at_the_given_time():
    circuit = Circuit([VoltageSource("V1", "a", "0", 2.0, waveform=Step(start_time=1.0)),
                       Resistor("R1", "a", "d", 100.0), Diode("D1", "d", "0")])
    model, _, drivers = build_model(circuit)
    before = operating_point(model, drivers, t=0.5)
    torch.testing.assert_close(_v(model, before, "d"), torch.tensor(0.0), rtol=0, atol=1e-14)
    after = operating_point(model, drivers, t=1.0, atol=1e-15, rtol=0.0)
    v, i = device_values(model, after)["D1"]
    torch.testing.assert_close(i, (2.0 - v) / 100.0, rtol=1e-10, atol=0)
    torch.testing.assert_close(i, 1e-6 * (torch.exp(v / 0.04) - 1.0) + v / 1e8, rtol=1e-9,
                               atol=0)
