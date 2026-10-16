"""Circuit analyses: consistent initial state, transient with backward-Euler restarts at the
first step and at source breakpoints, and the AC sweep."""

import math

import pytest
import torch

from noodl.apps.electrical import (
    LAYER,
    Capacitor,
    Circuit,
    ControlledSource,
    Diode,
    Inductor,
    Pulse,
    Resistor,
    Signal,
    Sine,
    Step,
    Transformer,
    VoltageSource,
    ac_sweep,
    build_model,
    device_values,
    initial_state,
    operating_point,
    transient,
)

F64 = torch.float64
PHI, Q = f"{LAYER}.phi", f"{LAYER}.q"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _rc(waveform=None, V=1.0, R=1.0, C=1.0, v0=0.0):
    source = (VoltageSource("V1", "a", "0", V) if waveform is None
              else VoltageSource("V1", "a", "0", V, waveform=waveform))
    return Circuit([source, Resistor("R1", "a", "b", R), Capacitor("C1", "b", "0", C, v0=v0)])


def _grid(dt, end=1.0):
    return [k * dt for k in range(round(end / dt) + 1)]


def _rc_error(dt):
    model, _, drivers = build_model(_rc())
    state = initial_state(model, drivers)
    out = transient(model, state, drivers, _grid(dt))
    b = model.net.nodes.index("b")
    return abs(out[PHI][-1, b].item() - (1.0 - math.exp(-1.0))), out


def test_rc_charge_is_second_order_after_a_backward_euler_start():
    e1, out = _rc_error(0.01)
    e2, _ = _rc_error(0.02)
    assert out["integration"] == ["backward_euler"] + ["trapezoidal"] * 99
    assert e1 < 1e-4
    assert math.log2(e2 / e1) >= 1.9


def test_transient_result_layout_and_start_row():
    model, _, drivers = build_model(_rc(v0=0.25))
    state = initial_state(model, drivers)
    out = transient(model, state, drivers, torch.tensor([0.0, 0.1, 0.3]))
    n, b = model.net.n, out[Q].shape[-1]
    assert out[PHI].shape == (3, n) and out[Q].shape == (3, b)
    torch.testing.assert_close(out["time"], torch.tensor([0.0, 0.1, 0.3]))
    torch.testing.assert_close(out[PHI][0], state[PHI], rtol=0, atol=0)
    assert len(out["integration"]) == 2


def test_backward_euler_everywhere_is_the_exact_recurrence():
    model, _, drivers = build_model(_rc())
    state = initial_state(model, drivers)
    dt = 0.1
    out = transient(model, state, drivers, _grid(dt), integration="backward_euler")
    assert set(out["integration"]) == {"backward_euler"}
    b = model.net.nodes.index("b")
    v = 0.0
    for k in range(1, 11):
        v = (v + dt) / (1.0 + dt)
        torch.testing.assert_close(out[PHI][k, b], torch.tensor(v), rtol=0, atol=1e-13)


def test_pulse_breakpoints_restart_with_backward_euler():
    # edges at 0.25, 0.5, 0.75, 1.0 on a grid of 0.125 (all exact binary fractions)
    pulse = Pulse(width=50.0, period=0.5, start_time=0.25)
    model, _, drivers = build_model(_rc(waveform=pulse))
    state = initial_state(model, drivers)
    out = transient(model, state, drivers, _grid(0.125))
    be, tr = "backward_euler", "trapezoidal"
    assert out["integration"] == [be, be, tr, be, tr, be, tr, be]
    # an edge inside a step restarts that step (the grid is not refined)
    pulse = Pulse(width=50.0, period=0.4, start_time=0.05)
    model, _, drivers = build_model(_rc(waveform=pulse))
    out = transient(model, initial_state(model, drivers), drivers, _grid(0.125))
    # edges 0.05, 0.25, 0.45, 0.65, 0.85
    assert out["integration"] == [be, be, tr, be, tr, be, be, tr]


def test_transient_of_a_batched_circuit():
    circuit = _rc(waveform=Signal("v"))
    model, _, drivers = build_model(circuit)
    drivers = {**drivers, "v": torch.tensor([[1.0], [2.0], [3.0], [4.0]])}
    state = initial_state(model, drivers)
    dt = 0.25
    out = transient(model, state, drivers, _grid(dt), integration="backward_euler")
    assert out[PHI].shape == (5, 4, model.net.n)
    b = model.net.nodes.index("b")
    V = torch.arange(1.0, 5.0)
    v = torch.zeros(4)
    for k in range(1, 5):                     # the backward-Euler recurrence, R C = 1
        v = (v + dt * V) / (1.0 + dt)
        torch.testing.assert_close(out[PHI][k, :, b], v, rtol=0, atol=1e-13)


def test_initial_state_holds_storage_at_its_initial_values():
    circuit = Circuit([VoltageSource("V1", "a", "0", 2.0), Resistor("R1", "a", "b", 1.0),
                       Capacitor("C1", "b", "0", 1.0, v0=0.5), Inductor("L1", "a", "c", 1.0,
                                                                         i0=0.3),
                       Resistor("R2", "c", "0", 1.0)])
    model, _, drivers = build_model(circuit)
    values = device_values(model, initial_state(model, drivers))
    torch.testing.assert_close(values["C1"][0], torch.tensor(0.5), rtol=0, atol=1e-14)
    torch.testing.assert_close(values["C1"][1], torch.tensor(1.5), rtol=0, atol=1e-14)
    torch.testing.assert_close(values["L1"][1], torch.tensor(0.3), rtol=0, atol=1e-14)
    torch.testing.assert_close(values["L1"][0], torch.tensor(1.7), rtol=0, atol=1e-14)


def test_transient_refusals():
    model, _, drivers = build_model(_rc())
    state = initial_state(model, drivers)
    with pytest.raises(ValueError, match="strictly increasing"):
        transient(model, state, drivers, [0.0, 0.1, 0.1])
    with pytest.raises(ValueError, match="integration='gear'"):
        transient(model, state, drivers, [0.0, 0.1], integration="gear")
    pmodel, _, pdrivers = build_model(_rc(), phasor=True, frequency=1.0)
    with pytest.raises(ValueError, match="phasor"):
        transient(pmodel, state, pdrivers, [0.0, 0.1])
    with pytest.raises(ValueError, match="phasor"):
        initial_state(pmodel, pdrivers)


# --------------------------------------------------------------------------------- AC
def test_series_rlc_resonance():
    R, L, C = 10.0, 1e-3, 1e-6
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", R),
                       Inductor("L1", "b", "c", L), Capacitor("C1", "c", "0", C)])
    freqs = torch.tensor([0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0]) * f0
    out = ac_sweep(circuit, freqs)
    assert out[PHI].dtype == torch.complex128 and out[PHI].shape == (7, 4)
    torch.testing.assert_close(out["frequency"], freqs)
    _, edges = build_model(circuit)[0].circuit_index["R1"]
    i = out[Q][:, edges][:, 0].abs()
    assert int(i.argmax()) == 3
    torch.testing.assert_close(i[3], torch.tensor(1.0 / R), rtol=1e-12, atol=0)
    w = 2.0 * math.pi * freqs
    expected = 1.0 / torch.sqrt(R ** 2 + (w * L - 1.0 / (w * C)) ** 2)
    torch.testing.assert_close(i, expected, rtol=1e-12, atol=0)


def test_ac_sweep_takes_the_sine_phase_and_the_sweep_frequency():
    R, C = 1.0, 1e-3
    source = VoltageSource("V1", "a", "0", 2.0, waveform=Sine(50.0, phase=0.3))
    circuit = Circuit([source, Resistor("R1", "a", "b", R), Capacitor("C1", "b", "0", C)])
    freqs = torch.tensor([10.0, 100.0, 1000.0])
    out = ac_sweep(circuit, freqs)
    b = build_model(circuit)[0].net.nodes.index("b")
    w = 2.0 * math.pi * freqs
    expected = 2.0 * torch.exp(torch.tensor(0.3j)) / (1.0 + 1j * w * R * C)
    torch.testing.assert_close(out[PHI][:, b], expected, rtol=1e-12, atol=0)


def test_ac_sweep_refuses_a_pulse_source_by_name():
    with pytest.raises(ValueError, match="'voltage_source'.*Pulse waveform has no phasor"):
        ac_sweep(_rc(waveform=Pulse(width=50.0, period=1.0)), torch.tensor([1.0]))


def test_device_values_in_a_phasor_model_are_complex_phasors():
    model, _, drivers = build_model(_rc(R=2.0, C=0.5), phasor=True,
                                    frequency=torch.tensor([0.1, 1.0]))
    values = device_values(model, operating_point(model, drivers))
    v, i = values["R1"]
    assert v.dtype == torch.complex128 and v.shape == (2,)
    w = 2.0 * math.pi * torch.tensor([0.1, 1.0])
    current = 1.0 / (2.0 + 1.0 / (1j * w * 0.5))   # 1 V across R = 2 in series with C = 0.5
    torch.testing.assert_close(i, current, rtol=1e-13, atol=0)
    torch.testing.assert_close(v, 2.0 * current, rtol=1e-13, atol=0)
    torch.testing.assert_close(values["C1"][0], current / (1j * w * 0.5), rtol=1e-13, atol=0)


def test_ac_sweep_of_a_resistive_circuit_keeps_the_frequency_dimension():
    circuit = Circuit([VoltageSource("V1", "a", "0", 3.0), Resistor("R1", "a", "b", 1.0),
                       Resistor("R2", "b", "0", 2.0)])
    out = ac_sweep(circuit, torch.tensor([1.0, 10.0, 100.0]))
    model = build_model(circuit)[0]
    assert out[PHI].shape == (3, model.net.n) and out[Q].shape == (3, 3)
    b = model.net.nodes.index("b")
    torch.testing.assert_close(out[PHI][:, b], torch.full((3,), 2.0 + 0j), rtol=0, atol=1e-12)
    _, edges = model.circuit_index["R2"]
    torch.testing.assert_close(out[Q][:, edges][:, 0], torch.full((3,), 1.0 + 0j), rtol=0,
                               atol=1e-12)


def test_diode_transient_with_tight_newton_tolerances():
    # a 2 V step through 100 ohm into a diode in parallel with 1 uF
    circuit = Circuit([VoltageSource("V1", "a", "0", 2.0, waveform=Step(start_time=2e-5)),
                       Resistor("R1", "a", "d", 100.0), Diode("D1", "d", "0"),
                       Capacitor("C1", "d", "0", 1e-6)])
    model, _, drivers = build_model(circuit)
    tight = {"atol": 1e-15, "rtol": 0.0}
    state = initial_state(model, drivers, **tight)
    out = transient(model, state, drivers, _grid(1e-5, end=1e-4), **tight)
    assert out["integration"][:3] == ["backward_euler", "backward_euler", "trapezoidal"]
    values = device_values(model, {PHI: out[PHI][-1], Q: out[Q][-1]})
    v = values["D1"][0]
    # KCL at d to round-off, and the diode current on its own law
    torch.testing.assert_close(values["R1"][1], values["D1"][1] + values["C1"][1], rtol=0,
                               atol=1e-15)
    torch.testing.assert_close(values["D1"][1], 1e-6 * (torch.exp(v / 0.04) - 1.0) + v / 1e8,
                               rtol=1e-12, atol=0)


def test_ac_sweep_of_a_resistive_circuit_returns_independent_rows():
    circuit = Circuit([VoltageSource("V1", "a", "0", 3.0), Resistor("R1", "a", "b", 1.0),
                       Resistor("R2", "b", "0", 2.0)])
    out = ac_sweep(circuit, torch.tensor([1.0, 10.0]))
    assert out[PHI].is_contiguous() and out[Q].is_contiguous()
    before = out[PHI][1].clone()
    out[PHI][0] += 1.0
    torch.testing.assert_close(out[PHI][1], before, rtol=0, atol=0)


def test_ac_sweep_forwards_solve_keywords():
    diagnostics = {}
    out = ac_sweep(_rc(), torch.tensor([1.0, 10.0]), method="direct", diagnostics=diagnostics)
    assert diagnostics["layers"][LAYER]["backend"] == "direct"
    reference = ac_sweep(_rc(), torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(out[PHI], reference[PHI], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("circuit", [
    Circuit([VoltageSource("V1", "a", "0", 1.0), Capacitor("C1", "a", "0", 1e-6),
             Resistor("R1", "a", "0", 1.0)]),
    Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
             Inductor("L1", "b", "c", 1e-3), Inductor("L2", "c", "0", 1e-3)]),
], ids=["capacitor-across-source", "inductors-in-series"])
def test_singular_initial_state_says_why_and_how_to_start(circuit):
    model, _, drivers = build_model(circuit)
    with pytest.raises(RuntimeError, match=r"(?s)singular.*capacitor acts as a voltage source "
                                           r"at its v0.*inductor as a current source at its "
                                           r"i0.*operating_point"):
        initial_state(model, drivers)
    # the operating point is a valid start
    out = transient(model, operating_point(model, drivers), drivers, [0.0, 1e-6, 2e-6])
    assert bool(torch.isfinite(out[PHI]).all())


def _coupled_run(values):
    """A source, an inductor, coupled windings and a VCCS stepped three times; the loss is a
    sum over the final voltages and currents, as a function of L, M, the gain and every R."""
    circuit = Circuit([
        VoltageSource("V1", "a", "0", 1.0, waveform=Sine(200.0)),
        Resistor("R1", "a", "b", 2.0), Inductor("L1", "b", "c", 1e-2, i0=0.1),
        Resistor("R2", "c", "0", 1.5), Transformer("T1", "c", "0", "d", "0", 1e-2, 2e-2, 5e-3),
        Resistor("R3", "d", "0", 3.0),
        ControlledSource("G1", "d", "0", "e", "0", 0.5, "voltage", "current"),
        Resistor("R4", "e", "0", 4.0),
    ])
    model, _, drivers = build_model(circuit, learnable=True)
    layer = model.kirchhoff[LAYER]
    params = [layer.law_for("inductor")[0].L, layer.law_for("transformer")[0].M,
              layer.law_for("vccs")[0].gain, layer.law_for("resistor")[0].R]
    with torch.no_grad():
        for p, v in zip(params, values, strict=True):
            p.copy_(v)
    out = transient(model, initial_state(model, drivers), drivers, [0.0, 1e-4, 2e-4, 3e-4])
    assert len(out["integration"]) == 3
    return out[Q][-1].sum() + out[PHI][-1].sum(), params


def test_transient_gradients_of_inductance_coupling_gain_and_resistance_match_differences():
    base = [torch.tensor([1e-2]), torch.tensor([5e-3]), torch.tensor([0.5]),
            torch.tensor([2.0, 1.5, 3.0, 4.0])]
    loss, params = _coupled_run(base)
    loss.backward()
    for i, p in enumerate(params):
        for j in range(p.numel()):
            h = 1e-5 * abs(float(base[i][j]))
            up = [b.clone() for b in base]
            down = [b.clone() for b in base]
            up[i][j] += h
            down[i][j] -= h
            with torch.no_grad():
                fd = (_coupled_run(up)[0] - _coupled_run(down)[0]) / (2 * h)
            torch.testing.assert_close(p.grad[j], fd, rtol=1e-8, atol=0)
