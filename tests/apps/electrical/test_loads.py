"""Modelica Buildings Library load laws: series impedance and constant-power loads."""

import math

import pytest
import torch

from noodl.apps.electrical import ConstantPowerLoad, Impedance, PrescribedVoltage
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
C128 = torch.complex128


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net(edges):
    net = Network(dtype=F64)
    for n in dict.fromkeys(n for u, v, _ in edges for n in (u, v)):
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _solve(edges, laws, *, phasor=False, f=60.0, drivers=None, **kwargs):
    net = _net(edges)
    layer = PotentialFlowLayer(net, "electrical", laws, boundary=["0"], phasor=phasor)
    d = dict(drivers or {})
    if phasor:
        d["electrical.frequency"] = torch.tensor(f)
    pb = torch.zeros(1, dtype=C128 if phasor else F64)
    phi, q = layer.solve(pb, d, differentiable=False, **kwargs)
    return phi, q, {n: k for k, n in enumerate(net.nodes)}


def test_impedances_at_60_hz_match_mbl_test_impedance():
    # MBL AC.OnePhase.Loads.Examples.TestImpedance: 120 V on five loads, X = 1 ohm at 60 Hz.
    L = C = 1.0 / (2.0 * math.pi * 60.0)
    edges = [("s", "0", "voltage_source")] + [("s", "0", "impedance")] * 5
    laws = [PrescribedVoltage(120.0),
            Impedance(torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
                      L=torch.tensor([L, 0.0, 0.0, L, 0.0]),
                      C=torch.tensor([math.inf, C, math.inf, math.inf, C]))]
    _, q, _ = _solve(edges, laws, phasor=True)
    expect = torch.tensor([-120j, 120j, 120.0, 60 - 60j, 60 + 60j], dtype=C128)
    torch.testing.assert_close(q[1:], expect, rtol=1e-13, atol=1e-12)


def test_zero_resistance_is_a_short_in_a_real_layer():
    edges = [("s", "0", "voltage_source"), ("s", "a", "impedance"), ("a", "0", "impedance")]
    _, q, _ = _solve(edges, [PrescribedVoltage(12.0), Impedance(torch.tensor([2.0, 0.0]))])
    torch.testing.assert_close(q[1:], torch.tensor([6.0, 6.0]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("reactance", [{"L": 1e-3}, {"C": 1e-3}])
def test_reactance_in_a_real_layer_is_refused(reactance):
    edges = [("s", "0", "voltage_source"), ("s", "0", "impedance")]
    with pytest.raises(ValueError, match="impedance kind 'impedance': a reactance needs a "
                                         "phasor layer"):
        _solve(edges, [PrescribedVoltage(1.0), Impedance(1.0, **reactance)])


def test_dc_constant_power_load_through_a_line():
    # 12 V, 0.1 ohm line, 50 W absorbed: v^2 - 12 v + 5 = 0, v = 6 + sqrt(31).
    edges = [("s", "0", "voltage_source"), ("s", "a", "impedance"), ("a", "0", "load")]
    laws = [PrescribedVoltage(12.0), Impedance(0.1),
            ConstantPowerLoad(50.0, V_nominal=12.0, direct_current=True)]
    phi, q, node = _solve(edges, laws)
    assert abs(float(phi[node["a"]]) - 11.567764362830022) <= 1e-12
    assert abs(float(q[2]) - 4.32235637169978) <= 1e-12


def test_ac_constant_power_load_matches_the_fixed_point():
    S, Z = 1000 + 750j, 0.01 + 0.05j
    v = 120 + 0j
    for _ in range(200):
        v = 120 - Z * (S / v).conjugate()
    edges = [("s", "0", "voltage_source"), ("s", "a", "impedance"), ("a", "0", "load")]
    laws = [PrescribedVoltage(120.0), Impedance(0.01, L=0.05 / (2 * math.pi * 60.0)),
            ConstantPowerLoad(1000.0, 750.0, V_nominal=120.0)]
    phi, q, node = _solve(edges, laws, phasor=True)
    assert abs(complex(phi[node["a"]]) - v) <= 1e-11
    assert abs(complex(q[2]) - (S / v).conjugate()) <= 1e-11


@pytest.mark.parametrize(("dp", "expect"), [(10.0, 4.991319444444445),
                                            (12.0, 4.166666666666666),
                                            (14.0, 3.5686728395061733)])
def test_linearised_dc_load_uses_mbl_voltage_bands(dp, expect):
    law = ConstantPowerLoad(50.0, V_nominal=12.0, linearized=True, direct_current=True)
    assert abs(float(law(torch.tensor([dp]), {})) - expect) <= 1e-14


def test_linearised_ac_load_is_linear_in_the_voltage():
    law = ConstantPowerLoad(torch.tensor([1000.0]), torch.tensor([750.0]), V_nominal=120.0,
                            linearized=True)
    dp = torch.tensor([110.0 + 5.0j], dtype=C128)
    expect = (1000 - 750j) * (110 + 5j) / 120.0 ** 2  # MBL Inductive.mo, linearized
    assert abs(complex(law(dp, {})) - expect) <= 1e-13


def test_constant_power_load_is_finite_at_zero_voltage():
    law = ConstantPowerLoad(torch.tensor([50.0]), V_nominal=12.0, direct_current=True)
    dp = torch.zeros(1, requires_grad=True)
    i = law(dp, {})
    assert float(i.detach()) == pytest.approx(50.0 * 2.0 / 12.0)  # MBL homotopy simplified form
    (g,) = torch.autograd.grad(i.sum(), dp)
    assert torch.isfinite(g).all()
    ac = ConstantPowerLoad(torch.tensor([1000.0]), torch.tensor([750.0]), V_nominal=120.0)
    assert complex(ac(torch.zeros(1, dtype=C128), {})) == 0j


def test_constant_power_load_forward_derivative_is_finite_at_zero_voltage():
    from torch.func import jvp

    ac = ConstantPowerLoad(torch.tensor([1000.0]), torch.tensor([750.0]), V_nominal=120.0)
    for tangent in (1.0 + 0j, 1j):
        _, d = jvp(lambda v: ac(v, {}), (torch.zeros(1, dtype=C128),),
                   (torch.full((1,), tangent, dtype=C128),))
        expect = (1000 - 750j) * tangent / 120.0 ** 2  # the simplified form's slope
        assert abs(complex(d) - expect) <= 1e-15


def test_actual_and_simplified_forms_switch_at_a_thousandth_of_nominal():
    law = ConstantPowerLoad(torch.tensor([50.0, 50.0]), V_nominal=12.0, direct_current=True)
    lo, hi = 0.99 * 12e-3, 1.01 * 12e-3
    i = law(torch.tensor([lo, hi]), {})
    torch.testing.assert_close(i, torch.tensor([50.0 * (2 / 12.0 - lo / 144.0), 50.0 / hi]),
                               rtol=1e-15, atol=0.0)


def test_without_nominal_voltage_the_load_is_the_actual_law():
    law = ConstantPowerLoad(torch.tensor([50.0]))
    torch.testing.assert_close(law(torch.tensor([1e-3]), {}), torch.tensor([5e4]))


def test_reactive_power_on_a_real_layer_is_refused():
    law = ConstantPowerLoad(1000.0, 750.0, V_nominal=120.0, kind="motor")
    with pytest.raises(ValueError, match="'motor'.*reactive power"):
        law(torch.tensor([120.0]), {})


def test_linearised_load_needs_a_nominal_voltage():
    with pytest.raises(ValueError, match="'load'.*V_nominal"):
        ConstantPowerLoad(50.0, linearized=True)


def test_scale_driver_multiplies_the_power():
    law = ConstantPowerLoad(torch.tensor([50.0, 50.0]), V_nominal=12.0, direct_current=True,
                            scale="electrical.load.scale")
    i = law(torch.tensor([12.0, 12.0]), {"electrical.load.scale": torch.tensor([1.0, 0.5])})
    torch.testing.assert_close(i, torch.tensor([50.0 / 12.0, 25.0 / 12.0]))
    with pytest.raises(KeyError, match="electrical.load.scale"):
        law(torch.tensor([12.0, 12.0]), {})


def test_laws_declare_their_forms():
    assert Impedance(1.0).law_form == "implicit" and Impedance(1.0).phasor_capable
    assert not Impedance(1.0).dynamic
    load = ConstantPowerLoad(1.0)
    assert load.law_form == "potential" and load.phasor_capable and not load.contextual


def _load_voltage(P, R=0.1, *, differentiable=False):
    edges = [("s", "0", "voltage_source"), ("s", "a", "impedance"), ("a", "0", "load")]
    net = _net(edges)
    load = ConstantPowerLoad(P, V_nominal=12.0, direct_current=True, learnable=differentiable)
    imp = Impedance(R, learnable=differentiable)
    layer = PotentialFlowLayer(net, "electrical", [PrescribedVoltage(12.0), imp, load],
                               boundary=["0"])
    phi, _ = layer.solve(torch.zeros(1), {}, differentiable=differentiable)
    return phi[net.nodes.index("a")], imp, load


def test_gradients_of_impedance_and_load_match_finite_differences():
    v, imp, load = _load_voltage(50.0, differentiable=True)
    dR, dP = torch.autograd.grad(v, (imp.R, load.P))
    h = 1e-6
    fd_R = (float(_load_voltage(50.0, 0.1 + h)[0])
            - float(_load_voltage(50.0, 0.1 - h)[0])) / (2 * h)
    fd_P = (float(_load_voltage(50.0 + h)[0]) - float(_load_voltage(50.0 - h)[0])) / (2 * h)
    assert float(dR) == pytest.approx(fd_R, rel=1e-6)
    assert float(dP) == pytest.approx(fd_P, rel=1e-6)
