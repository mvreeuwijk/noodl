"""Electro-thermal coupling: Joule heat into a thermal Kirchhoff layer, temperature back."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.electrical import (
    Capacitance,
    HeatedEdge,
    JouleHeating,
    PrescribedCurrent,
    PrescribedVoltage,
    Resistance,
    Signal,
)
from noodl.elements import Conductance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.topology import Network

V, R0, ALPHA, T_REF, G, T_AMB, C_TH = 220.0, 100.0, 1e-3, 293.15, 50.0, 293.15, 100.0


def _f(x):
    """A float64 parameter (a law stores a Python float in the default dtype, float32)."""
    return torch.tensor(x, dtype=torch.float64)


def _resistor_model(dynamic):
    net = Network(dtype=torch.float64)
    for name in ("0", "s", "thermal.reference", "h", "amb"):
        net.add_node(name)
    net.add_edge("s", "0", kind="voltage_source")
    net.add_edge("s", "0", kind="resistor")
    net.add_edge("h", "amb", kind="conductor")
    net.add_edge("thermal.reference", "h", kind="heat_source")
    if dynamic:
        net.add_edge("h", "thermal.reference", kind="heat_capacitor")
    node = {n: k for k, n in enumerate(net.nodes)}
    NODE.update(node)
    electrical = KirchhoffLayer(
        net, "electrical",
        [PrescribedVoltage(_f(V)), Resistance(_f(R0), alpha=_f(ALPHA), T_ref=_f(T_REF),
                                              temperature="electrical.resistor.T")],
        boundary=["0"])
    # the laws carry the thermal edge kinds (their defaults are the electrical ones)
    laws = [Conductance(_f(G), kind="conductor"),
            PrescribedCurrent(_f(1.0), waveforms=[Signal("thermal.heat", 0)],
                              kind="heat_source")]
    if dynamic:
        laws.append(Capacitance(_f(C_TH), v0=_f(T_AMB), kind="heat_capacitor"))
    thermal = KirchhoffLayer(net, "thermal", laws, boundary=["thermal.reference", "amb"])
    heated = [HeatedEdge(edge=1, p=node["s"], n=node["0"], heat_node=node["h"],
                         temperature_key="electrical.resistor.T", slot=0)]
    closure = JouleHeating(heated, base_temperatures={"electrical.resistor.T":
                                                      torch.tensor([T_REF], dtype=torch.float64)})
    model = Model(net, [electrical, thermal], closures=[closure], coupling="iterate",
                  iterate_tol={"electrical": 1e-10, "thermal": 1e-10},
                  iterate_flow_tol={"electrical": 1e-10, "thermal": 1e-8},
                  iterate_acceleration="anderson", iterate_max=100)
    n, f64 = len(net.nodes), torch.float64
    phi_t = torch.full((n,), T_AMB, dtype=f64)
    phi_t[node["thermal.reference"]] = 0.0
    phi_e = torch.zeros(n, dtype=f64)
    phi_e[node["s"]] = V
    # thermal flows in the layer's edge order: conductor, heat source (, heat capacitor)
    state = {"electrical.phi": phi_e,
             "electrical.q": torch.tensor([V / R0, V / R0], dtype=f64),
             "thermal.phi": phi_t, "thermal.q": torch.zeros(3 if dynamic else 2, dtype=f64)}
    drivers = {"electrical.phi_boundary": torch.zeros(1, dtype=f64),
               "thermal.phi_boundary": torch.tensor([0.0, T_AMB], dtype=f64),
               "thermal.heat": torch.zeros(1, dtype=f64),
               "electrical.resistor.T": torch.tensor([T_REF], dtype=f64)}
    return model, state, drivers


NODE: dict[str, int] = {}


def _power(T):
    return V * V / (R0 * (1.0 + ALPHA * (T - T_REF)))


def _scalar_recurrence(steps, h):
    T0, out = T_AMB, []
    for k in range(steps):
        def f(T, k=k, T0=T0):
            rate = (_power(T) - G * (T - T_AMB))
            if k == 0:
                return C_TH * (T - T0) / h - rate
            return C_TH * (T - T0) / h - 0.5 * (rate + _power(T0) - G * (T0 - T_AMB))
        T = T0
        for _ in range(50):  # Newton with a numerical slope, converged to round-off
            d = (f(T + 1e-6) - f(T - 1e-6)) / 2e-6
            T -= f(T) / d
        out.append(T)
        T0 = T
    return out


def test_resistor_heated_to_its_fixed_point():
    # 220 V on R(T) = 100 (1 + 1e-3 (T - 293.15)); G = 50 W/K to 293.15 K.
    # alpha x^2 + x - V^2/(G R0) = 0  ->  x = 9.588068934114679 K, i = 2.179106575935151 A.
    model, state, drivers = _resistor_model(dynamic=False)
    out = model.steady(state, drivers)
    T = float(out["thermal.phi"][..., NODE["h"]])
    assert abs(T - 302.73806893411466) <= 1e-9
    assert abs(float(out["electrical.q"][..., 1]) - 2.179106575935151) <= 1e-11


def test_heated_resistor_with_heat_capacity_follows_the_coupled_trapezoidal_recurrence():
    # C_th = 100 J/K on h, G = 50 W/K to 293.15 K, constant 220 V; dt = 0.5 s, 6 steps
    # (first backward Euler, then trapezoidal), against a scalar Newton solve per step of
    #   BE:   C (T1 - T0)/h          = P(T1) - G (T1 - Ta)
    #   trap: C (T1 - T0)/h = 1/2 [P(T1) - G (T1 - Ta) + P(T0) - G (T0 - Ta)]
    # with P(T) = V^2 / (R0 (1 + alpha (T - Tref))).
    model, state, drivers = _resistor_model(dynamic=True)
    ref = _scalar_recurrence(steps=6, h=0.5)
    for k in range(6):
        scheme = "backward_euler" if k == 0 else "trapezoidal"
        state = model.step(state, drivers, dt=0.5, t=0.5 * k, integration=scheme)
        assert abs(float(state["thermal.phi"][..., NODE["h"]]) - ref[k]) <= 1e-9


def test_joule_heating_closure_is_batched_and_pure():
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=1, temperature_key="electrical.resistor.T",
                         slot=0)]
    close = JouleHeating(heated, base_temperatures={"electrical.resistor.T": torch.tensor([300.0])})
    state = {"electrical.phi": torch.tensor([[0.0, 10.0], [0.0, 20.0]]),
             "electrical.q": torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
             "thermal.phi": torch.tensor([[0.0, 310.0], [0.0, 320.0]])}
    d = close(state, {})
    torch.testing.assert_close(d["thermal.heat"], torch.tensor([[20.0], [20.0]]))
    torch.testing.assert_close(d["electrical.resistor.T"], torch.tensor([[310.0], [320.0]]))


def test_joule_heating_leaves_its_inputs_alone_and_carries_gradients():
    phi = torch.tensor([0.0, 10.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor([0.0, 310.0, 305.0], dtype=torch.float64, requires_grad=True)
    base = torch.tensor([300.0, 290.0], dtype=torch.float64)
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=2, temperature_key="T", slot=1)]
    close = JouleHeating(heated, heat_key="h", base_temperatures={"T": base})
    d = close({"electrical.phi": phi, "electrical.q": q, "thermal.phi": temperature}, {})
    assert d["T"].tolist() == [300.0, 305.0] and base.tolist() == [300.0, 290.0]
    (d["h"].sum() + d["T"].sum()).backward()
    assert q.grad.tolist() == [0.0, 10.0] and phi.grad.tolist() == [-2.0, 2.0]
    assert temperature.grad.tolist() == [0.0, 0.0, 1.0]


def test_joule_heating_refuses_an_inconsistent_description():
    edge = HeatedEdge(edge=0, p=1, n=0, heat_node=1, temperature_key="T", slot=2)
    with pytest.raises(ValueError, match="no heated device"):
        JouleHeating([], base_temperatures={})
    with pytest.raises(ValueError, match="no base temperatures for"):
        JouleHeating([edge], base_temperatures={"other": torch.zeros(3)})
    with pytest.raises(ValueError, match="slot 2 lies outside"):
        JouleHeating([edge], base_temperatures={"T": torch.zeros(2)})


def test_joule_heating_keeps_the_callers_temperature_of_an_unheated_device():
    # slot 0 has no heat port: the caller's 400 K driver wins over the 300 K base
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=1, temperature_key="T", slot=1)]
    close = JouleHeating(heated, base_temperatures={"T": torch.tensor([300.0, 300.0])})
    state = {"electrical.phi": torch.tensor([0.0, 10.0]),
             "electrical.q": torch.tensor([0.0, 2.0]),
             "thermal.phi": torch.tensor([0.0, 310.0])}
    d = close(state, {"T": torch.tensor([400.0, 350.0])})
    assert d["T"].tolist() == [400.0, 310.0]
    assert close(state, {})["T"].tolist() == [300.0, 310.0]
