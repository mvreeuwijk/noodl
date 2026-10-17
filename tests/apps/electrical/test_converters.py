"""Modelica Buildings Library AC/AC transformer and AC/DC converter laws."""

import math

import pytest
import torch

from noodl.apps.electrical import (
    Impedance,
    ImpedanceTransformer,
    PowerConverter,
    PrescribedVoltage,
)
from noodl.apps.electrical.converters import reg_root, splice
from noodl.elements import LawContext
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
C128 = torch.complex128
FREQUENCY = {"electrical.frequency": torch.tensor(60.0, dtype=F64)}


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _layer(edges, laws):
    net = Network(dtype=F64)
    for n in dict.fromkeys(n for u, v, _ in edges for n in (u, v)):
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net, KirchhoffLayer(net, "electrical", laws, boundary=["0"], phasor=True)


def test_transformer_with_primary_impedance_feeds_a_resistor():
    # MBL ACACTransformer: VHigh 120, VLow 60, VABase 4000, Zperc 0.03, XoverR 8.
    zp = 120.0 ** 2 * 0.03 / 4000.0
    z1 = complex(zp * math.cos(math.atan(8.0)), zp * math.sin(math.atan(8.0)))
    net, layer = _layer([("s", "0", "voltage_source"), ("s", "0", "impedance_transformer"),
                         ("b", "0", "impedance_transformer"), ("b", "0", "impedance")],
                        [PrescribedVoltage(120.0),
                         ImpedanceTransformer(2.0, torch.tensor([z1], dtype=C128)),
                         Impedance(1.0)])
    phi, q = layer.solve(torch.zeros(1, dtype=C128), FREQUENCY, differentiable=False)
    vp = 60.0 / (1.0 + z1 / 4.0)  # = 59.757127513724186 - 1.59563954377388j
    node = {n: k for k, n in enumerate(net.nodes)}
    assert abs(complex(phi[node["b"]]) - vp) <= 1e-12
    assert abs(complex(q[1]) - vp / 2.0) <= 1e-12   # primary current, entering
    assert abs(complex(q[2]) + vp) <= 1e-12         # secondary current, entering


def test_transformer_takes_a_python_complex_impedance():
    law = ImpedanceTransformer(2.0, 0.1 + 0.2j)
    assert law.impedance.dtype == C128


def test_power_converter_from_flat_start():
    # MBL ACDCConverter: 120 V AC, ratio 0.5, eta 0.9; 1 ohm DC load absorbs 3600 W at 60 V.
    net, layer = _layer([("s", "0", "voltage_source"), ("s", "0", "power_converter"),
                         ("d", "0", "power_converter"), ("d", "0", "impedance")],
                        [PrescribedVoltage(120.0), PowerConverter(0.5, 0.9), Impedance(1.0)])
    node = {n: k for k, n in enumerate(net.nodes)}
    phi0 = torch.full((len(net.nodes),), 120.0 + 0j, dtype=C128)[layer.interior]
    phi, q = layer.solve(torch.zeros(1, dtype=C128), FREQUENCY, phi0=phi0,
                         differentiable=False)
    assert abs(complex(phi[node["d"]]) - 60.0) <= 1e-12          # imaginary part zero too
    assert abs(complex(q[2]) + 60.0) <= 1e-12                     # DC current leaves the converter
    assert abs(complex(q[1]) - 100.0 / 3.0) <= 1e-11              # 4000 W / 120 V, unity pf


def test_power_converter_losses_follow_the_dc_side_when_it_feeds_the_ac_side():
    # i_p > 0 (current into the DC terminal): the loss is (1 - eta) of the DC power.
    law = PowerConverter(0.5, 0.9)
    dp = torch.tensor([120.0 + 0j, 60.0 + 0j], dtype=C128)
    q = torch.tensor([-30.0 + 0j, 60.0 + 0j], dtype=C128)
    r = law(dp, q, {}, LawContext(mode="phasor", omega=torch.tensor([2 * math.pi * 60])))
    # Re S_ac + P_dc - 0.1 P_dc = -3600 + 3600 - 360
    assert abs(complex(r[1]) - (-360.0)) <= 1e-9


@pytest.mark.parametrize("law, mode", [(ImpedanceTransformer(2.0, 0.1j), "dc"),
                                       (PowerConverter(0.5, 0.9), "transient")])
def test_converters_need_a_phasor_layer(law, mode):
    ctx = LawContext(mode=mode)
    with pytest.raises(ValueError, match=f"{law.kind!r}.*phasor layer"):
        law(torch.zeros(2), torch.zeros(2), {}, ctx)


def test_splice_and_reg_root_transcribe_mbl():
    x = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(splice(torch.tensor(2.0), torch.tensor(4.0), x, 0.1),
                               torch.tensor([4.0, 3.0, 2.0]))
    expect = 14400.0 / (14400.0 ** 2 + 1e-10) ** 0.25
    assert float(reg_root(torch.tensor(14400.0), 1e-5)) == pytest.approx(expect, rel=1e-15)


def test_splice_inside_the_band_matches_the_tanh_tan_form():
    x = torch.tensor([-0.05, 0.02])
    y = (torch.tanh(torch.tan(x / 0.1 * math.asin(1.0))) + 1.0) / 2.0
    torch.testing.assert_close(splice(torch.tensor(2.0), torch.tensor(4.0), x, 0.1),
                               2.0 * y + (1.0 - y) * 4.0, rtol=1e-15, atol=0.0)
    xg = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    (g,) = torch.autograd.grad(splice(torch.tensor(2.0), torch.tensor(4.0), xg, 0.1).sum(), xg)
    assert torch.isfinite(g).all()
