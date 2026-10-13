"""Electrical constitutive laws, each solved exactly in a small Modified Nodal Analysis layer."""

import math

import pytest
import torch
from torch.func import jvp

from noodl.apps.electrical import (
    Capacitance,
    Constant,
    DependentSource,
    Inductance,
    JunctionDiode,
    MutualInductance,
    PrescribedCurrent,
    PrescribedVoltage,
    Pulse,
    Ramp,
    Resistance,
    Signal,
    Sine,
    Step,
    SwitchedConductance,
    TransformerRatio,
)
from noodl.elements import LawContext
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network
from tests._circuit_laws import Cap, Ind

F64 = torch.float64
GROUND = torch.zeros(1, dtype=F64)
T = torch.tensor


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _layer(edges, laws, nodes=None, **kwargs):
    nodes = nodes or sorted({n for u, v, _ in edges for n in (u, v)})
    net = _net(nodes, edges)
    return net, PotentialFlowLayer(net, "e", laws, boundary=["gnd"], **kwargs)


def _at(net, phi, name):
    return phi[..., net.nodes.index(name)]


def _close(actual, expected, tol=1e-12, rtol=0.0):
    actual = torch.as_tensor(actual)
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype),
                               rtol=rtol, atol=tol)


# ----------------------------------------------------------------------------- sources
def test_prescribed_voltage_drives_a_resistor():
    # source a -> gnd holds a at +2 V (dp = v_a - v_gnd); resistor a -> gnd, R = 4
    net, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                        [PrescribedVoltage(T([2.0])), Resistance(T([4.0]))])
    phi, q = layer.solve(GROUND, differentiable=False)
    _close(_at(net, phi, "a"), 2.0)
    # 0.5 A flows a -> gnd through the resistor and gnd -> a through the source
    _close(q, [-0.5, 0.5])


def test_prescribed_voltage_offset_and_waveforms_per_edge():
    # two sources: 1 + 2 * Step(1) and 0.5 + 3 * sin(2 pi t), each across its own resistor
    edges = [("a", "gnd", "voltage_source"), ("b", "gnd", "voltage_source"),
             ("a", "gnd", "resistor"), ("b", "gnd", "resistor")]
    source = PrescribedVoltage(T([2.0, 3.0]), offset=T([1.0, 0.5]),
                               waveforms=[Step(start_time=1.0), Sine(1.0)])
    net, layer = _layer(edges, [source, Resistance(T([1.0, 1.0]))])
    for t in (0.5, 1.0, 1.125):
        phi, _ = layer.solve(GROUND, t=t, differentiable=False)
        _close(_at(net, phi, "a"), 1.0 + 2.0 * (t >= 1.0))
        _close(_at(net, phi, "b"), 0.5 + 3.0 * math.sin(2 * math.pi * t))


def test_signal_waveform_reads_a_batched_driver():
    source = PrescribedVoltage(T([2.0]), offset=T([1.0]), waveforms=[Signal("u", index=1)])
    net, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                        [source, Resistance(T([1.0]))])
    u = T([[9.0, 0.0], [9.0, 1.0], [9.0, 2.5]])
    phi, _ = layer.solve(GROUND, {"u": u}, differentiable=False)
    _close(_at(net, phi, "a"), [1.0, 3.0, 6.0])


def test_time_dependent_source_without_time_is_refused():
    _, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                      [PrescribedVoltage(T([1.0]), waveforms=Step()), Resistance(T([1.0]))])
    with pytest.raises(ValueError, match="source kind 'voltage_source' has time-dependent "
                                         r"waveforms; pass t \(the evaluation time\)"):
        layer.solve(GROUND, differentiable=False)


def test_waveform_count_must_match_the_edges():
    _, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                      [PrescribedVoltage(T([1.0]), waveforms=[Step(), Step()]),
                       Resistance(T([1.0]))])
    with pytest.raises(ValueError, match="'voltage_source'.*2 waveforms for 1 edges"):
        layer.solve(GROUND, t=0.0, differentiable=False)


def test_a_waveform_that_is_not_one_is_refused():
    with pytest.raises(TypeError, match="'vs'.*waveform"):
        PrescribedVoltage(T([1.0]), waveforms=["sine"], kind="vs")


def test_source_breakpoints_collect_every_waveform():
    waveforms = [Step(start_time=0.5), Constant(), Sine(1.0, start_time=0.2)]
    source = PrescribedCurrent(T([1.0, 1.0, 1.0]), waveforms=waveforms)
    assert source.breakpoints(0.0, 1.0) == [0.2, 0.5]
    assert source.breakpoints(0.3, 1.0) == [0.5]
    assert PrescribedVoltage(1.0).breakpoints(0.0, 1.0) == []
    assert PrescribedVoltage(1.0, waveforms=Step(0.5)).breakpoints(0.0, 1.0) == [0.5]


def test_learnable_amplitude_gradient():
    source = PrescribedVoltage(T([2.0]), learnable=True)
    _, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                      [source, Resistance(T([4.0]))])
    _, q = layer.solve(GROUND)
    (grad,) = torch.autograd.grad(q[1], source.amplitude)
    _close(grad, [0.25])


def test_prescribed_current_into_a_resistor():
    # current source a -> gnd draws 1 A out of a (MSL: i enters pin p = a); R = 2 => v_a = -2
    net, layer = _layer([("a", "gnd", "current_source"), ("a", "gnd", "resistor")],
                        [PrescribedCurrent(T([1.0])), Resistance(T([2.0]))])
    assert layer.mna
    phi, q = layer.solve(GROUND, differentiable=False)
    _close(_at(net, phi, "a"), -2.0)
    _close(q, [1.0, -1.0])


def test_sources_in_phasor_mode_are_their_waveform_phasors():
    # V: 2 sin(w (t - 0.002) + pi/3) + 5 across R = 4 -> 2 exp(j (pi/3 - w 0.002));
    # I: a Constant source of amplitude 1 -> the phasor 1 (phase 0), into b across R = 1
    w = 2 * math.pi * 50.0
    edges = [("a", "gnd", "voltage_source"), ("a", "gnd", "resistor"),
             ("b", "gnd", "resistor"), ("b", "gnd", "current_source")]
    laws = [PrescribedVoltage(T([2.0]), offset=T([5.0]),
                              waveforms=[Sine(50.0, phase=math.pi / 3, start_time=0.002)]),
            Resistance(T([4.0, 1.0])),
            PrescribedCurrent(T([1.0]))]
    net, layer = _layer(edges, laws, phasor=True)
    phi, _ = layer.solve(GROUND.to(torch.complex128), {"e.frequency": T([50.0])},
                         differentiable=False)
    angle = math.pi / 3 - w * 0.002
    _close(_at(net, phi, "a"), [2.0 * complex(math.cos(angle), math.sin(angle))])
    _close(_at(net, phi, "b"), [complex(-1.0, 0.0)])


@pytest.mark.parametrize("waveform", [Step(), Ramp(1.0), Pulse(50.0, 1.0), Signal("u")],
                         ids=lambda w: type(w).__name__)
def test_phasor_mode_refuses_waveforms_without_a_phasor(waveform):
    _, layer = _layer([("a", "gnd", "voltage_source"), ("a", "gnd", "resistor")],
                      [PrescribedVoltage(T([1.0]), waveforms=waveform), Resistance(T([1.0]))],
                      phasor=True)
    name = type(waveform).__name__
    with pytest.raises(ValueError, match=f"'voltage_source': a {name} waveform has no phasor"):
        layer.solve(GROUND.to(torch.complex128), {"e.frequency": T([1.0]), "u": T([1.0])},
                    differentiable=False)


def test_sine_source_transient_steady_state_matches_its_phasor():
    # 1 Hz sine (phase 0.4, starting at 0.3 s) through R = 1 into C = 0.02 (tau = 0.02 s):
    # after 2 periods the transient has decayed by exp(-85); the last period of the
    # trapezoidal march (100 steps per period, relative error ~ (w dt)^2 / 12 = 3e-4) must
    # follow the phasor solution, whose phase includes -w start_time
    f, phase, t_start, n = 1.0, 0.4, 0.3, 100
    edges = [("a", "gnd", "voltage_source"), ("a", "b", "resistor"), ("b", "gnd", "capacitor")]
    laws = [PrescribedVoltage(T([1.5]), waveforms=Sine(f, phase=phase, start_time=t_start)),
            Resistance(T([1.0])), Capacitance(T([0.02]))]
    net, layer = _layer(edges, laws)
    _, phasor_layer = _layer(edges, laws, phasor=True)
    phi_ac, _ = phasor_layer.solve(GROUND.to(torch.complex128), {"e.frequency": T([f])},
                                   differentiable=False)
    vb = complex(_at(net, phi_ac, "b"))
    b = net.nodes.index("b")
    dt = 1.0 / (f * n)
    phi, q = layer.solve(GROUND, mode="initial", t=0.0, differentiable=False)
    samples = []
    for k in range(1, 3 * n + 1):
        phi, q = layer.solve(GROUND, history=(phi, q), dt=dt, t=k * dt,
                             integration="trapezoidal", differentiable=False)
        if k > 2 * n:
            samples.append((k * dt, phi[b].item()))
    w = 2 * math.pi * f
    expected = [abs(vb) * math.sin(w * t + math.atan2(vb.imag, vb.real)) for t, _ in samples]
    errors = [abs(v - e) for (_, v), e in zip(samples, expected, strict=True)]
    assert max(errors) < 2e-4 * abs(vb)
    # peak and zero crossing of the last period (the sampled peak lies up to
    # 1 - cos(pi / n) = 4.9e-4 below the true one)
    assert abs(max(v for _, v in samples) - abs(vb)) < 1e-3 * abs(vb)
    rising = [t for (t, v), (_, v_next) in zip(samples[:-1], samples[1:], strict=True)
              if v <= 0.0 < v_next]
    assert len(rising) == 1
    t_zero = (-math.atan2(vb.imag, vb.real) / w) % (1.0 / f) + 2.0 / f
    assert abs(rising[0] - t_zero) <= dt


def test_waveform_units_are_computed_in_the_layer_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        source = PrescribedVoltage(1.0, waveforms=Sine(50.0))
    finally:
        torch.set_default_dtype(previous)
    assert source.amplitude.dtype == torch.float32
    t = 1.3e-3
    r = source(torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64), {},
               LawContext(mode="dc", t=t))
    assert r.dtype == F64
    assert abs(-r.item() - math.sin(2 * math.pi * 50.0 * t)) <= 1e-15


def test_laws_name_a_missing_driver():
    with pytest.raises(KeyError, match="'resistor' reads the driver 'T_wire'"):
        Resistance(T([1.0]), temperature="T_wire").flow(T([1.0]), None)
    with pytest.raises(KeyError, match="'switch' reads the driver 's'"):
        SwitchedConductance("s").flow(T([1.0]), None)
    source = PrescribedCurrent(T([1.0]), waveforms=Signal("u"))
    with pytest.raises(KeyError, match="'current_source' reads the driver 'u'"):
        source(T([1.0]), {}, LawContext(mode="dc"))


# --------------------------------------------------------------------------- resistance
def test_resistance_temperature_coefficient_reads_a_driver():
    # R_actual = R (1 + alpha (T - T_ref)) per edge, MSL Basic.Resistor
    edges = [("a", "gnd", "voltage_source"), ("a", "gnd", "resistor"), ("a", "gnd", "resistor")]
    res = Resistance(T([2.0, 4.0]), alpha=T([0.004, 0.0]), temperature="T_wire")
    net, layer = _layer(edges, [PrescribedVoltage(T([10.0])), res])
    phi, q = layer.solve(GROUND, {"T_wire": T([350.15, 0.0])}, differentiable=False)
    _close(q[1:], [10.0 / (2.0 * (1 + 0.004 * 50.0)), 10.0 / 4.0])
    # without a temperature key the resistance is R at T_ref
    flow = Resistance(T([2.0]), alpha=T([0.004])).flow(T([10.0]), {})
    _close(flow, [5.0])


def test_resistance_dflow_and_linear_init_are_the_conductance():
    res = Resistance(T([2.0, 4.0]))
    _close(res.dflow(T([1.0, -3.0])), [0.5, 0.25])
    c, k = res.linear_init()
    _close(c, [0.0, 0.0])
    _close(k, [0.5, 0.25])


# ------------------------------------------------------------- capacitance, inductance
_CONTEXTS = [
    LawContext(mode="dc"),
    LawContext(mode="initial"),
    LawContext(mode="phasor", omega=T([314.0, 1000.0])),
    LawContext(mode="transient", dt=0.1, integration="backward_euler",
               dp_prev=T([0.3, -0.2]), q_prev=T([0.05, 0.4])),
    LawContext(mode="transient", dt=0.1, integration="trapezoidal",
               dp_prev=T([0.3, -0.2]), q_prev=T([0.05, 0.4])),
]


@pytest.mark.parametrize("ctx", _CONTEXTS, ids=lambda c: f"{c.mode}-{c.integration}")
def test_capacitance_and_inductance_are_the_companion_forms(ctx):
    dp, q = T([1.5, -0.7]), T([0.2, 0.9])
    if ctx.mode == "phasor":
        dp, q = dp * (1 + 0.5j), q * (0.3 - 1j)
    pairs = [(Capacitance(T([2e-3, 1e-2]), v0=T([0.4, 0.1])),
              Cap(T([2e-3, 1e-2]), v0=T([0.4, 0.1]))),
             (Inductance(T([1e-3, 5e-2]), i0=T([0.3, -0.2])),
              Ind(T([1e-3, 5e-2]), i0=T([0.3, -0.2])))]
    for law, reference in pairs:
        assert law.dynamic and law.phasor_capable and law.law_form == "implicit"
        _close(law(dp, q, {}, ctx), reference(dp, q, {}, ctx), tol=0)


def test_rc_backward_euler_step_is_exact():
    # source a -> gnd 1 V, resistor a -> b R = 1, capacitor b -> gnd C = 1
    edges = [("a", "gnd", "voltage_source"), ("a", "b", "resistor"), ("b", "gnd", "capacitor")]
    net, layer = _layer(edges, [PrescribedVoltage(T([1.0])), Resistance(T([1.0])),
                                Capacitance(T([1.0]))])
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    v = 0.0
    for _ in range(3):
        phi, q = layer.solve(GROUND, history=(phi, q), dt=0.1, t=0.0,
                             integration="backward_euler", differentiable=False)
        v = (v + 0.1) / 1.1
        _close(_at(net, phi, "b"), v, tol=1e-13)


# ------------------------------------------------------------------------------- diode
def _diode_current(v, Ids=1e-6, Vt=0.04, Maxexp=15.0, R=1e8):
    x = v / Vt
    z = math.exp(Maxexp) * (1 + x - Maxexp) if x > Maxexp else math.exp(x)
    return Ids * (z - 1) + v / R


def test_diode_in_series_with_a_resistor():
    # 1 V source a -> gnd, R = 100 a -> b, diode b -> gnd (forward); KCL at b
    edges = [("a", "gnd", "voltage_source"), ("a", "b", "resistor"), ("b", "gnd", "diode")]
    net, layer = _layer(edges, [PrescribedVoltage(T([1.0])), Resistance(T([100.0])),
                                JunctionDiode()])
    phi, q = layer.solve(GROUND, differentiable=False, atol=1e-15, rtol=0.0)
    vb = _at(net, phi, "b").item()
    assert 0.3 < vb < 0.7
    _close(q[2], _diode_current(vb), tol=1e-13)
    _close(q[1], (1.0 - vb) / 100.0, tol=1e-13)
    _close(q[1], q[2], tol=1e-12)


def test_diode_matches_msl_on_both_branches():
    d = JunctionDiode(Ids=T([2e-6]), Vt=T([0.05]), Maxexp=T([10.0]), R=T([1e6]))
    for v in (-1.0, 0.0, 0.3, 0.5, 0.75):     # x = 20 v: below, at and above Maxexp
        _close(d.flow(T([v])), [_diode_current(v, 2e-6, 0.05, 10.0, 1e6)], tol=1e-15)
    assert not d.phasor_capable


def test_diode_no_overflow_at_large_forward_voltage():
    d = JunctionDiode()
    dp = T([100.0])
    i = d.flow(dp)
    _, di = jvp(lambda x: d.flow(x), (dp,), (torch.ones_like(dp),))
    assert torch.isfinite(i).all() and torch.isfinite(di).all()
    _close(i, [_diode_current(100.0)], tol=1e-6)
    _close(di, [1e-6 * math.exp(15.0) / 0.04 + 1e-8], tol=1e-9)


# ------------------------------------------------------------------------------ switch
def test_switch_state_from_driver():
    # 1 V across two switches in parallel to ground; switch 0 reads s[..., 0], switch 1 s[..., 1]
    edges = [("a", "gnd", "voltage_source"), ("a", "gnd", "switch"), ("a", "gnd", "switch")]
    sw = SwitchedConductance("s", Ron=T([1e-2, 1e-2]), Goff=T([1e-3, 1e-3]))
    _, layer = _layer(edges, [PrescribedVoltage(T([1.0])), sw])
    states = T([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    _, q = layer.solve(GROUND, {"s": states}, differentiable=False)
    _close(q[..., 1:], [[100.0, 1e-3], [1e-3, 100.0], [1e-3, 1e-3]])


def test_switch_control_per_edge_entries():
    sw = SwitchedConductance(["a", ("b", 2)], Ron=1e-5, Goff=1e-5)
    flow = sw.flow(T([1.0, 1.0]), {"a": T([1.0]), "b": T([0.0, 0.0, 0.2])})
    _close(flow, [1e5, 1e-5], tol=0, rtol=1e-15)
    flow = sw.flow(T([1.0, 1.0]), {"a": T([False]), "b": T([0.0, 0.0, 0.9])})
    _close(flow, [1e-5, 1e5], tol=0, rtol=1e-15)
    assert sw.phasor_capable


def test_switch_control_must_match_the_edges():
    sw = SwitchedConductance("s")
    with pytest.raises(ValueError, match="'switch'.*3 values per instance for 2 switch edges"):
        sw.flow(T([1.0, 1.0]), {"s": T([1.0, 0.0, 1.0])})
    # one value for all edges broadcasts
    _close(sw.flow(T([1.0, 1.0]), {"s": T([1.0])}), [1e5, 1e5], tol=0, rtol=1e-15)


def test_switch_control_as_a_lone_key_index_pair():
    sw = SwitchedConductance(("s", 1), Ron=T([0.5, 0.25]), Goff=0.0)
    _close(sw.flow(T([1.0, 1.0]), {"s": T([0.0, 1.0])}), [2.0, 4.0])
    _close(sw.flow(T([1.0, 1.0]), {"s": T([1.0, 0.0])}), [0.0, 0.0])


# ------------------------------------------------------------------------ two-ports
def test_ideal_transformer_reflects_load():
    # source p -> gnd 10 V; transformer [p -> gnd, s -> gnd], n = 2; load s -> gnd R = 4
    edges = [("p", "gnd", "voltage_source"), ("p", "gnd", "ideal_transformer"),
             ("s", "gnd", "ideal_transformer"), ("s", "gnd", "resistor")]
    net, layer = _layer(edges, [PrescribedVoltage(T([10.0])), TransformerRatio(T([2.0])),
                                Resistance(T([4.0]))])
    phi, q = layer.solve(GROUND, differentiable=False)
    _close(_at(net, phi, "s"), 5.0)
    # i2 = -n i1 (MSL IdealTransformer without magnetisation)
    _close(q, [-0.625, 0.625, -1.25, 1.25])


def _controlled(law):
    # control port c -> gnd; output port o -> gnd with a 4-ohm load.
    # voltage control: a 2 V source holds c; current control: 2 V at a, R = 2 a -> c.
    if law.control == "voltage":
        edges = [("c", "gnd", "voltage_source"), ("c", "gnd", law.kind),
                 ("o", "gnd", law.kind), ("o", "gnd", "load")]
    else:
        edges = [("a", "gnd", "voltage_source"), ("a", "c", "resistor"), ("c", "gnd", law.kind),
                 ("o", "gnd", law.kind), ("o", "gnd", "load")]
    laws = [PrescribedVoltage(T([2.0])), law, Resistance(T([4.0]), kind="load")]
    if law.control == "current":
        laws.insert(1, Resistance(T([2.0])))
    return _layer(edges, laws)


@pytest.mark.parametrize("control, output, gain, v_out, q_ports", [
    ("voltage", "voltage", 3.0, 6.0, [0.0, -1.5]),     # v2 = g v1, i1 = 0
    ("voltage", "current", 0.5, -4.0, [0.0, 1.0]),     # i2 = g v1, i1 = 0
    ("current", "voltage", 5.0, 5.0, [1.0, -1.25]),    # v2 = g i1, v1 = 0
    ("current", "current", 2.0, -8.0, [1.0, 2.0]),     # i2 = g i1, v1 = 0
])
def test_dependent_sources_match_hand_mna(control, output, gain, v_out, q_ports):
    law = DependentSource(T([gain]), control=control, output=output)
    assert law.kind == f"{control[0]}c{output[0]}s"
    net, layer = _controlled(law)
    phi, q = layer.solve(GROUND, differentiable=False)
    _close(_at(net, phi, "o"), v_out)
    _close(q[..., layer.kind_slice(law.kind)], q_ports)
    if control == "current":
        _close(_at(net, phi, "c"), 0.0)
    else:
        _close(_at(net, phi, "c"), 2.0)


def test_dependent_source_refuses_unknown_kinds():
    with pytest.raises(ValueError, match="control"):
        DependentSource(1.0, control="power", output="voltage")
    with pytest.raises(ValueError, match="output"):
        DependentSource(1.0, control="voltage", output="charge")


def _two_inductor_layer(laws, kinds):
    # 1 V source a -> gnd and 1 V source b -> gnd, each through R = 1 into an inductor
    # (x -> gnd, y -> gnd); the inductors are the two windings
    edges = [("a", "gnd", "voltage_source"), ("b", "gnd", "voltage_source"),
             ("a", "x", "resistor"), ("b", "y", "resistor"),
             ("x", "gnd", kinds[0]), ("y", "gnd", kinds[1])]
    return _layer(edges, [PrescribedVoltage(T([1.0, 2.0])), Resistance(T([1.0, 2.0])), *laws])


def _march(layer, steps, integration, dt=0.05):
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    for _ in range(steps):
        phi, q = layer.solve(GROUND, history=(phi, q), dt=dt, t=0.0, integration=integration,
                             differentiable=False)
    return phi, q


def test_mutual_inductance_reduces_to_two_inductors_when_M_zero():
    _, coupled = _two_inductor_layer(
        [MutualInductance(T([0.2]), T([0.5]), T([0.0]), i0=T([0.1, -0.3]))],
        ["transformer", "transformer"])
    _, single = _two_inductor_layer([Inductance(T([0.2, 0.5]), i0=T([0.1, -0.3]))],
                                    ["inductor", "inductor"])
    for integration in ("trapezoidal", "backward_euler"):
        phi_c, q_c = _march(coupled, 4, integration)
        phi_s, q_s = _march(single, 4, integration)
        _close(phi_c, phi_s, tol=1e-13)
        _close(q_c, q_s, tol=1e-13)


def test_mutual_inductance_companion_matches_hand_step():
    # one backward-Euler step of v = Lmat di/dt from i_prev, both windings driven directly
    L1, L2, M, dt = 0.2, 0.5, 0.25, 0.01
    edges = [("a", "gnd", "voltage_source"), ("b", "gnd", "voltage_source"),
             ("a", "gnd", "transformer"), ("b", "gnd", "transformer")]
    _, layer = _layer(edges, [PrescribedVoltage(T([1.0, -0.5])),
                              MutualInductance(L1, L2, M, i0=T([0.1, 0.2]))])
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    _close(q[2:], [0.1, 0.2])
    phi, q = layer.solve(GROUND, history=(phi, q), dt=dt, t=0.0,
                         integration="backward_euler", differentiable=False)
    Lmat = torch.tensor([[L1, M], [M, L2]])
    di = torch.linalg.solve(Lmat, T([1.0, -0.5])) * dt
    _close(q[2:], T([0.1, 0.2]) + di, tol=1e-12)


def test_mutual_inductance_phasor_and_dc():
    L1, L2, M = 0.2, 0.5, 0.25
    law = MutualInductance(L1, L2, M)
    dp, q = T([1.0 + 0.5j, -0.2j]), T([0.3 + 0.1j, 0.4 - 0.2j])
    w = T([100.0])
    r = law(dp, q, {}, LawContext(mode="phasor", omega=w))
    expected = torch.stack([dp[0] - 100j * (L1 * q[0] + M * q[1]),
                            dp[1] - 100j * (M * q[0] + L2 * q[1])])
    _close(r, expected, tol=1e-13)
    _close(law(dp.real, q.real, {}, LawContext(mode="dc")), dp.real, tol=0)


def test_laws_carry_many_devices_and_declare_their_forms():
    assert PrescribedVoltage(1.0).law_form == "implicit"
    assert PrescribedVoltage(1.0).phasor_capable
    cur = PrescribedCurrent(T([1.0, 2.0]), waveforms=Constant())
    assert cur.law_form == "potential" and cur.dp_independent and cur.contextual
    _close(cur(T([[5.0, 6.0]]), {}, LawContext(mode="dc")), [[1.0, 2.0]])
    assert TransformerRatio(1.0).ports == 2 and MutualInductance(1, 1, 0).ports == 2
    assert MutualInductance(1, 1, 0).dynamic
    assert DependentSource(1.0, control="current", output="current").ports == 2
    assert not Resistance(1.0).contextual and Resistance(1.0).phasor_capable
