"""A capacitor to ground is a node storage, a floating one an edge law; the transient and the
reported currents agree with the capacitor as an edge `Capacitance` to 1e-12."""

import pytest
import torch

from noodl.apps.electrical import (
    LAYER,
    Capacitance,
    Capacitor,
    Circuit,
    Resistor,
    Step,
    VoltageSource,
    build_model,
    device_values,
    initial_state,
    operating_point,
    transient,
)
from noodl.apps.electrical.laws import PrescribedVoltage, Resistance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
PHI, Q, SF = f"{LAYER}.phi", f"{LAYER}.q", f"{LAYER}.storage_flow"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def test_a_grounded_capacitor_is_a_storage_entry_and_a_floating_one_an_edge():
    circuit = Circuit([
        VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
        Capacitor("Cg", "b", "0", 2e-3, v0=0.4), Capacitor("Cn", "0", "a", 3e-3, v0=0.1),
        Capacitor("Cf", "a", "b", 5e-3, v0=0.2),
    ])
    model, _, _ = build_model(circuit)
    layer = model.kirchhoff[LAYER]
    assert layer.kinds == ["voltage_source", "resistor", "capacitor"]
    assert layer.kind_slice("capacitor").stop - layer.kind_slice("capacitor").start == 1
    assert model.circuit_index["Cf"][0] == "capacitor"
    assert "Cg" not in model.circuit_index and "Cn" not in model.circuit_index
    (storage,) = layer.storage
    nodes = model.net.nodes
    assert storage.nodes.tolist() == [nodes.index("b"), nodes.index("a")]
    torch.testing.assert_close(storage.C, torch.tensor([2e-3, 3e-3]), rtol=0, atol=0)
    # v0 when the ground is pin n, -v0 when it is pin p
    torch.testing.assert_close(storage.initial(), torch.tensor([0.4, -0.1]), rtol=0, atol=0)
    assert model.circuit_storage == {"Cg": (0, nodes.index("b"), 0, 1.0),
                                     "Cn": (1, 0, nodes.index("a"), -1.0)}
    assert layer.dynamic


def test_a_phasor_model_keeps_every_capacitor_as_an_edge():
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
                       Capacitor("C1", "b", "0", 1e-3)])
    model, _, _ = build_model(circuit, phasor=True, frequency=50.0)
    layer = model.kirchhoff[LAYER]
    assert not layer.storage and layer.kinds[-1] == "capacitor"
    assert "C1" in model.circuit_index and not model.circuit_storage


def _edge_reference(circuit_caps):
    """The same circuit with every capacitor an edge `Capacitance` (built by hand)."""
    net = Network(dtype=F64)
    for node in ("0", "a", "b", "c"):
        net.add_node(node)
    net.add_edge("a", "0", kind="voltage_source")
    net.add_edge("a", "b", kind="resistor")
    net.add_edge("b", "c", kind="resistor")
    for u, v, _, _ in circuit_caps:
        net.add_edge(u, v, kind="capacitor")
    laws = [PrescribedVoltage(torch.tensor([1.0]), waveforms=Step(start_time=0.0125)),
            Resistance(torch.tensor([2.0, 3.0])),
            Capacitance(torch.tensor([c for *_, c, _ in circuit_caps]),
                        v0=torch.tensor([v0 for *_, v0 in circuit_caps]))]
    layer = KirchhoffLayer(net, LAYER, laws, boundary=["0"], quantity="voltage", unit="V",
                           flow_unit="A")
    model = Model(net, [layer])
    drivers = {f"{LAYER}.phi_boundary": torch.zeros(1, dtype=F64)}
    return model, layer, drivers


def test_an_rc_transient_matches_the_capacitors_as_edge_laws():
    caps = [("b", "0", 1e-3, 0.4), ("0", "c", 3e-3, -0.1)]    # ground at pin n and at pin p
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0, waveform=Step(start_time=0.0125)),
                       Resistor("R1", "a", "b", 2.0), Resistor("R2", "b", "c", 3.0)]
                      + [Capacitor(f"C{k}", u, v, c, v0=v0)
                         for k, (u, v, c, v0) in enumerate(caps)])
    model, _, drivers = build_model(circuit)
    state = initial_state(model, drivers)
    times = [0.005 * k for k in range(21)]
    out = transient(model, state, drivers, times)

    ref, layer, ref_drivers = _edge_reference(caps)
    ref_drivers[f"{LAYER}.phi_boundary"] = drivers[f"{LAYER}.phi_boundary"]
    phi, q = layer.solve(ref_drivers[f"{LAYER}.phi_boundary"], ref_drivers, None,
                         mode="initial", t=0.0)
    cap = layer.kind_slice("capacitor")
    row = {PHI: phi, Q: q}
    torch.testing.assert_close(out[PHI][0], phi, rtol=0, atol=1e-12)
    for k, rule in enumerate(out["scheme"]):
        row = ref.step(row, ref_drivers, dt=times[k + 1] - times[k], t=times[k], scheme=rule)
        torch.testing.assert_close(out[PHI][k + 1], row[PHI], rtol=0, atol=1e-12)
        # the capacitor currents (entering pin p) are the storage flows, with their signs
        want = row[Q][..., cap]
        got = out[SF][k + 1] * torch.tensor([1.0, -1.0])
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    # device_values reads them the same way
    last = {PHI: out[PHI][-1], Q: out[Q][-1], SF: out[SF][-1]}
    values = device_values(model, last)
    torch.testing.assert_close(values["C1"][1], row[Q][..., cap][1], rtol=0, atol=1e-12)
    torch.testing.assert_close(values["C0"][0], row[PHI][2] - row[PHI][0], rtol=0, atol=1e-12)
    torch.testing.assert_close(values["C1"][0], row[PHI][0] - row[PHI][3], rtol=0, atol=1e-12)


def test_operating_point_has_no_capacitor_current():
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 2.0),
                       Capacitor("C1", "b", "0", 1e-3)])
    model, _, drivers = build_model(circuit)
    op = operating_point(model, drivers)
    assert float(device_values(model, op)["C1"][1]) == 0.0
    state = initial_state(model, drivers)
    assert state[SF].shape == (1,)
