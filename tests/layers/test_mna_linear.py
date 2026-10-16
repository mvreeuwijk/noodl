"""A layer of linear laws takes the full Newton step from the start: one iteration."""

import pytest
import torch

from noodl.apps.electrical.laws import Capacitance, JunctionDiode, PrescribedVoltage, Resistance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
GROUND = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _layer(second):
    # source a -> g at 2 V, resistor a -> b, then `second` b -> g
    net = Network(dtype=F64)
    for n in ("g", "a", "b"):
        net.add_node(n)
    net.add_edge("a", "g", kind="voltage_source")
    net.add_edge("a", "b", kind="resistor")
    net.add_edge("b", "g", kind=second.kind)
    return KirchhoffLayer(net, "e", [PrescribedVoltage(2.0), Resistance(1.0), second],
                          boundary=["g"])


def test_a_resistor_divider_converges_in_one_newton_iteration():
    layer = _layer(Resistance(3.0, kind="load"))
    assert layer.mna and layer.linear
    d = {}
    phi, _ = layer.solve(GROUND, differentiable=False, diagnostics=d)
    assert d["newton_iterations"] == 1 and d["converged"]
    torch.testing.assert_close(phi[..., 2], torch.tensor(1.5), rtol=0, atol=1e-12)


def test_an_rc_step_converges_in_one_newton_iteration():
    layer = _layer(Capacitance(0.5))
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    d = {}
    layer.solve(GROUND, history=(phi, q), dt=0.1, t=0.0, integration="backward_euler",
                differentiable=False, diagnostics=d)
    assert d["newton_iterations"] == 1


def test_a_diode_keeps_the_damped_newton():
    layer = _layer(JunctionDiode(1e-6, 0.04, kind="diode"))
    assert not layer.linear
    d = {}
    layer.solve(GROUND, differentiable=False, diagnostics=d)
    assert d["newton_iterations"] > 1 and d["converged"]
