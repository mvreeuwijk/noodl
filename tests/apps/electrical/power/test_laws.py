"""Power-flow constitutive laws: the MATPOWER branch two-port, loads, shunts, generators."""

import cmath
import math

import pytest
import torch

from noodl.apps.electrical import ConstantPowerLoad
from noodl.apps.electrical.power import PiBranch, Shunt, VoltageControlledGenerator
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

C128 = torch.complex128


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_pi_branch_matches_makeybus_with_tap_and_shift():
    # r, x, b, ratio, shift = 0.01, 0.1, 0.02, 0.95, 10 deg; V_f = 1, V_t = 0.98 at -5 deg.
    # Reference currents from MATPOWER's makeYbus as ported by PYPOWER 5.1.21 (agreeing with
    # the closed form to 4e-15).
    law = PiBranch(0.01, 0.1, 0.02, 0.95, math.radians(10.0))
    v = torch.tensor([1.0 + 0j, 0.98 * cmath.exp(-1j * math.radians(5.0))], dtype=C128)
    i = law(v, {})
    expected = torch.tensor([-0.810594584793739 - 0.8737767557268548j,
                             0.9051910961588779 + 0.7038857638586933j], dtype=C128)
    torch.testing.assert_close(i, expected, rtol=0, atol=1e-12)
    assert law.ports == 2 and law.law_form == "potential" and law.phasor_capable


def test_pi_branch_ratio_zero_means_one_and_batches_branches():
    two = PiBranch(torch.tensor([0.01, 0.01]), torch.tensor([0.1, 0.1]),
                   torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]))
    v = torch.tensor([1.0, 0.9, 1.0, 0.9], dtype=C128)
    i = two(v, {})
    torch.testing.assert_close(i[:2], i[2:], rtol=0, atol=0)
    ys = 1 / complex(0.01, 0.1)
    torch.testing.assert_close(i[:2], torch.tensor([ys * 0.1, -ys * 0.1], dtype=C128),
                               rtol=0, atol=1e-14)


def test_load_and_shunt_currents():
    v = torch.tensor([0.95 * cmath.exp(-0.1j)], dtype=C128)
    load = ConstantPowerLoad(0.9, 0.3)
    torch.testing.assert_close(load(v, {}), (complex(0.9, 0.3) / v).conj(), rtol=0, atol=1e-15)
    sh = Shunt(0.05, 0.19)
    torch.testing.assert_close(sh(v, {}), complex(0.05, 0.19) * v, rtol=0, atol=1e-15)
    assert sh.law_form == "potential" and sh.phasor_capable


def _two_bus(device_law, kind):
    # slack 1 at 0 rad -- lossless line x = 0.1 -- bus "2" -- device to ground
    net = Network(dtype=torch.float64)
    for node in ("1", "2", "ground"):
        net.add_node(node)
    net.add_edge("1", "ground", kind="branch")
    net.add_edge("2", "ground", kind="branch")
    net.add_edge("2", "ground", kind=kind)
    layer = PotentialFlowLayer(net, "power", [PiBranch(0.0, 0.1, 0.0, 0.0, 0.0), device_law],
                               boundary=["ground", "1"], phasor=True)
    bound_names = [net.nodes[k] for k in layer.bound.tolist()]
    pb = torch.tensor([{"ground": 0.0, "1": 1.0}[n] for n in bound_names], dtype=C128)
    drivers = {"power.frequency": torch.tensor(50.0)}
    phi0 = torch.ones(len(layer.interior), dtype=C128)
    phi, q = layer.solve(pb, drivers, None, phi0, differentiable=False, atol=1e-13, rtol=0.0)
    return net, layer, phi, q


def test_constant_power_load_two_bus_closed_form():
    # Lossless line, load P = 0.5 at bus 2: |V2|^2 = (1 + sqrt(1 - 4 x^2 P^2)) / 2.
    net, layer, phi, q = _two_bus(ConstantPowerLoad(0.5, 0.0), "load")
    v2 = phi[net.nodes.index("2")]
    assert abs(abs(v2) ** 2 - (1 + math.sqrt(0.99)) / 2) < 1e-12
    s = v2 * q[layer.kind_slice("load")][0].conj()
    assert abs(s - 0.5) < 1e-12


def test_voltage_controlled_generator_two_bus_closed_form():
    # Generator P = 0.5 at |V2| = 1 behind x = 0.1: sin(theta2) = P x, Q = (1 - cos theta2)/x.
    net, layer, phi, q = _two_bus(VoltageControlledGenerator(0.5, 1.0), "generator")
    v2 = phi[net.nodes.index("2")]
    theta = math.asin(0.05)
    assert abs(abs(v2) - 1.0) < 1e-12
    assert abs(cmath.phase(v2) - theta) < 1e-12
    s_gen = -v2 * q[layer.kind_slice("generator")][0].conj()
    assert abs(s_gen.real - 0.5) < 1e-12
    assert abs(s_gen.imag - (1 - math.cos(theta)) / 0.1) < 1e-12
    assert abs(layer.power_residual(phi, q, {})) < 1e-12   # complex Tellegen


def test_generator_is_an_implicit_phasor_law():
    gen = VoltageControlledGenerator(0.5, 1.0)
    assert gen.law_form == "implicit" and gen.phasor_capable
    with pytest.raises(TypeError, match="implicit"):
        gen.flow(torch.ones(1, dtype=C128))


def test_learnable_coefficients_carry_gradients():
    branch = PiBranch(0.01, 0.1, 0.02, 0.95, 0.1, learnable=True)
    gen = VoltageControlledGenerator(0.5, 1.0, learnable=True)
    shunt = Shunt(0.05, 0.19, learnable=True)
    names = {n for law in (branch, gen, shunt) for n, _ in law.named_parameters()}
    assert names == {"r", "x", "b", "ratio", "shift", "P", "V", "g"}
    v = torch.tensor([1.0 + 0.1j, 0.97 - 0.05j], dtype=C128)
    loss = branch(v, {}).abs().sum() + shunt(v, {}).abs().sum()
    loss = loss + gen(v, torch.tensor([0.4 - 0.1j, 0.2j], dtype=C128), {}, None).abs().sum()
    loss.backward()
    for law in (branch, gen, shunt):
        assert all(p.grad is not None for p in law.parameters())
