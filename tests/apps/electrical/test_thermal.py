"""Electro-thermal coupling: Joule heat into a heat layer (a `TransportLayer`), temperature
back."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.electrical import HeatedEdge, JouleHeating, PrescribedVoltage, Resistance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

V, R0, ALPHA, T_REF, G, T_AMB, C_TH = 220.0, 100.0, 1e-3, 293.15, 50.0, 293.15, 100.0
F64 = torch.float64
COUPLING = dict(coupling="iterate", iterate_tol={"electrical": 1e-10, "thermal": 1e-10},
                iterate_flow_tol={"electrical": 1e-10}, iterate_acceleration="anderson",
                iterate_max=100)


def _f(x):
    """A float64 parameter (a law stores a Python float in the default dtype, float32)."""
    return torch.tensor(x, dtype=F64)


def _heat_layer(net, *, capacity):
    """The heat layer on h (heat port) -- amb (fixed): conduction G, capacity at h (0: the
    heat port holds no heat and is folded into its neighbour)."""
    return TransportLayer(net, "thermal", flow_kind=(), boundary=["amb"],
                          capacity=_f([capacity]), conduction_kind="conductor",
                          conductance=_f([G]), storage_free=[] if capacity else ["h"],
                          scheme="trapezoidal", quantity="temperature", unit="K",
                          source_unit="W", capacity_unit="J/K")


def _resistor_model(dynamic, *, carry_heat=True):
    net = Network(dtype=F64)
    for name in ("0", "s", "h", "amb"):
        net.add_node(name)
    net.add_edge("s", "0", kind="voltage_source")
    net.add_edge("s", "0", kind="resistor")
    net.add_edge("h", "amb", kind="conductor")
    node = {n: k for k, n in enumerate(net.nodes)}
    electrical = KirchhoffLayer(
        net, "electrical",
        [PrescribedVoltage(_f(V)), Resistance(_f(R0), alpha=_f(ALPHA), T_ref=_f(T_REF),
                                              temperature="electrical.resistor.T")],
        boundary=["0"])
    thermal = _heat_layer(net, capacity=C_TH if dynamic else 0.0)
    heated = [HeatedEdge(edge=1, p=node["s"], n=node["0"], heat_node=node["h"],
                         temperature_key="electrical.resistor.T", slot=0)]
    closure = JouleHeating(heated, thermal, base_temperatures={
        "electrical.resistor.T": torch.tensor([T_REF], dtype=F64)})
    if not carry_heat:
        closure.state_keys = ()
    model = Model(net, [electrical, thermal], closures=[closure], **COUPLING)
    n = len(net.nodes)
    phi_e = torch.zeros(n, dtype=F64)
    phi_e[node["s"]] = V
    state = {"electrical.phi": phi_e,
             "electrical.q": torch.tensor([V / R0, V / R0], dtype=F64),
             "thermal.x": _f([T_AMB]), "thermal.sources": torch.zeros(n, dtype=F64)}
    drivers = {"electrical.phi_boundary": torch.zeros(1, dtype=F64),
               "thermal.x_boundary": _f([T_AMB]),
               "electrical.resistor.T": torch.tensor([T_REF], dtype=F64)}
    return model, state, drivers


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
    assert abs(float(out["thermal.x"][0]) - 302.73806893411466) <= 1e-9
    assert abs(float(out["electrical.q"][..., 1]) - 2.179106575935151) <= 1e-11


def test_heated_resistor_with_heat_capacity_follows_the_coupled_trapezoidal_recurrence():
    # C_th = 100 J/K on h, G = 50 W/K to 293.15 K, constant 220 V; dt = 0.5 s, 6 steps
    # (first backward Euler, then trapezoidal), against a scalar Newton solve per step of
    #   BE:   C (T1 - T0)/h          = P(T1) - G (T1 - Ta)
    #   trap: C (T1 - T0)/h = 1/2 [P(T1) - G (T1 - Ta) + P(T0) - G (T0 - Ta)]
    # with P(T) = V^2 / (R0 (1 + alpha (T - Tref))): the heat layer steps with the circuit's
    # rule, and its trapezoidal step integrates the heat at both ends (the state carries
    # the heat at its own time).
    model, state, drivers = _resistor_model(dynamic=True)
    ref = _scalar_recurrence(steps=6, h=0.5)
    h = model.net.nodes.index("h")
    for k in range(6):
        scheme = "backward_euler" if k == 0 else "trapezoidal"
        heat = "implicit" if k == 0 else "trapezoidal"
        state = model.step(state, drivers, dt=0.5, t=0.5 * k, integration=scheme,
                           layer_options={"thermal": {"scheme": heat}})
        assert abs(float(state["thermal.x"][0]) - ref[k]) <= 1e-9
        assert abs(float(state["thermal.sources"][h]) - _power(ref[k])) <= 1e-6


def test_holding_the_heat_over_a_trapezoidal_step_misses_the_recurrence():
    # Without the heat at the step start (a closure that does not carry it) the trapezoidal
    # step holds the end-of-step heat over the step: first order in the heat, off the
    # recurrence by far more than its tolerance.
    model, state, drivers = _resistor_model(dynamic=True, carry_heat=False)
    ref = _scalar_recurrence(steps=2, h=0.5)
    state = model.step(state, drivers, dt=0.5, t=0.0, integration="backward_euler",
                       layer_options={"thermal": {"scheme": "implicit"}})
    assert abs(float(state["thermal.x"][0]) - ref[0]) <= 1e-9
    state = model.step(state, drivers, dt=0.5, t=0.5, integration="trapezoidal")
    assert abs(float(state["thermal.x"][0]) - ref[1]) > 1e-4


def _closure(heated, n_nodes=2, interior=(1,), boundary=(), **kw):
    """`JouleHeating` on a heat layer over `n_nodes` network nodes: conductors from each
    interior node to the first boundary node (or, without one, to node 0), and the given
    boundary."""
    net = Network(dtype=F64)
    for k in range(n_nodes):
        net.add_node(f"n{k}")
    to = boundary[0] if boundary else 0
    for k in interior:
        net.add_edge(f"n{k}", f"n{to}", kind="conductor")
    layer = TransportLayer(net, "thermal", flow_kind=(), boundary=[f"n{k}" for k in boundary],
                           capacity=torch.ones(len(interior) + (0 if boundary else 1),
                                               dtype=F64),
                           conduction_kind="conductor",
                           conductance=torch.ones(len(interior), dtype=F64))
    return JouleHeating(heated, layer, **kw)


def test_joule_heating_closure_is_batched_and_pure():
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=1, temperature_key="electrical.resistor.T",
                         slot=0)]
    close = _closure(heated, base_temperatures={"electrical.resistor.T": torch.tensor([300.0])})
    state = {"electrical.phi": torch.tensor([[0.0, 10.0], [0.0, 20.0]]),
             "electrical.q": torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
             "thermal.x": torch.tensor([[290.0, 310.0], [290.0, 320.0]])}
    d = close(state, {})
    # full node order: the heat at node 1, nothing at node 0
    torch.testing.assert_close(d["thermal.sources"], torch.tensor([[0.0, 20.0], [0.0, 20.0]]))
    torch.testing.assert_close(d["electrical.resistor.T"], torch.tensor([[310.0], [320.0]]))
    assert close.state_keys == ("thermal.sources",) and close.integrates is False


def test_joule_heating_leaves_its_inputs_alone_and_carries_gradients():
    phi = torch.tensor([0.0, 10.0], dtype=F64, requires_grad=True)
    q = torch.tensor([0.0, 2.0], dtype=F64, requires_grad=True)
    temperature = torch.tensor([310.0, 305.0], dtype=F64, requires_grad=True)
    base = torch.tensor([300.0, 290.0], dtype=F64)
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=2, temperature_key="T", slot=1)]
    close = _closure(heated, n_nodes=3, interior=(1, 2), boundary=(0,),
                     base_temperatures={"T": base})
    d = close({"electrical.phi": phi, "electrical.q": q, "thermal.x": temperature}, {})
    assert d["T"].tolist() == [300.0, 305.0] and base.tolist() == [300.0, 290.0]
    (d["thermal.sources"].sum() + d["T"].sum()).backward()
    assert q.grad.tolist() == [0.0, 10.0] and phi.grad.tolist() == [-2.0, 2.0]
    assert temperature.grad.tolist() == [0.0, 1.0]


def test_two_devices_on_one_heat_port_add_their_heat():
    heated = [HeatedEdge(edge=0, p=1, n=0, heat_node=2, temperature_key="T", slot=0),
              HeatedEdge(edge=1, p=1, n=0, heat_node=2, temperature_key="T", slot=1)]
    close = _closure(heated, n_nodes=3, interior=(2,), boundary=(0,),
                     base_temperatures={"T": torch.tensor([300.0, 300.0])})
    d = close({"electrical.phi": torch.tensor([0.0, 10.0, 0.0]),
               "electrical.q": torch.tensor([1.0, 2.0]), "thermal.x": torch.tensor([305.0])},
              {})
    assert d["thermal.sources"].tolist() == [0.0, 0.0, 30.0]
    assert d["T"].tolist() == [305.0, 305.0]


def test_a_heat_port_at_a_fixed_temperature_reads_the_boundary_and_has_no_source():
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=2, temperature_key="T", slot=0)]
    close = _closure(heated, n_nodes=3, interior=(1,), boundary=(2,),
                     base_temperatures={"T": torch.tensor([300.0])})
    d = close({"electrical.phi": torch.tensor([0.0, 10.0, 0.0]),
               "electrical.q": torch.tensor([0.0, 2.0]), "thermal.x": torch.tensor([400.0])},
              {"thermal.x_boundary": torch.tensor([310.0])})
    assert d["thermal.sources"].tolist() == [0.0, 0.0, 0.0]
    assert d["T"].tolist() == [310.0]


def test_joule_heating_refuses_an_inconsistent_description():
    edge = HeatedEdge(edge=0, p=1, n=0, heat_node=1, temperature_key="T", slot=2)
    with pytest.raises(ValueError, match="no heated device"):
        _closure([], base_temperatures={})
    with pytest.raises(ValueError, match="no base temperatures for"):
        _closure([edge], base_temperatures={"other": torch.zeros(3)})
    with pytest.raises(ValueError, match="slot 2 lies outside"):
        _closure([edge], base_temperatures={"T": torch.zeros(2)})
    outside = HeatedEdge(edge=0, p=1, n=0, heat_node=2, temperature_key="T", slot=0)
    with pytest.raises(ValueError, match=r"heat ports \['n2'\] are not nodes"):
        _closure([outside], n_nodes=3, base_temperatures={"T": torch.zeros(1)})


def test_joule_heating_keeps_the_callers_temperature_of_an_unheated_device():
    # slot 0 has no heat port: the caller's 400 K driver wins over the 300 K base
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=1, temperature_key="T", slot=1)]
    close = _closure(heated, base_temperatures={"T": torch.tensor([300.0, 300.0])})
    state = {"electrical.phi": torch.tensor([0.0, 10.0]),
             "electrical.q": torch.tensor([0.0, 2.0]),
             "thermal.x": torch.tensor([290.0, 310.0])}
    d = close(state, {"T": torch.tensor([400.0, 350.0])})
    assert d["T"].tolist() == [400.0, 310.0]
    assert close(state, {})["T"].tolist() == [300.0, 310.0]


def test_batched_temperature_driver_with_unbatched_state_matches_independent_runs():
    heated = [HeatedEdge(edge=1, p=1, n=0, heat_node=1, temperature_key="T", slot=1)]
    close = _closure(heated, base_temperatures={"T": torch.tensor([300.0, 300.0, 300.0])})
    state = {"electrical.phi": torch.tensor([0.0, 10.0]), "electrical.q": torch.tensor([0.0, 2.0]),
             "thermal.x": torch.tensor([290.0, 310.0])}
    drivers = torch.tensor([[293.15, 1.0, 5.0], [300.0, 2.0, 6.0]])
    batched = close(state, {"T": drivers})
    assert batched["T"].shape == (2, 3)
    for k in range(2):
        single = close(state, {"T": drivers[k]})
        torch.testing.assert_close(batched["T"][k], single["T"])
    # unheated devices keep their temperatures, the heated slot takes the heat port
    torch.testing.assert_close(batched["T"][:, 1], torch.full((2,), 310.0))
    torch.testing.assert_close(batched["T"][:, [0, 2]], drivers[:, [0, 2]])


def test_the_initial_point_holds_the_heat_capacity_at_its_start_temperature():
    # h (heat port, no capacity) --G1-- c (C_TH) --G-- amb. With mode="initial" the stored
    # node c is held at its start 400 K, as the old Capacitance(C, v0=T0) held it, and the
    # folded port satisfies its balance G1 (T_h - 400) = P(T_h); the DC operating point
    # instead solves c too. P(T) = V^2 / (R0 (1 + alpha (T - Tref))).
    G1 = 20.0
    net = Network(dtype=F64)
    for name in ("0", "s", "h", "c", "amb"):
        net.add_node(name)
    net.add_edge("s", "0", kind="voltage_source")
    net.add_edge("s", "0", kind="resistor")
    net.add_edge("h", "c", kind="conductor")
    net.add_edge("c", "amb", kind="conductor")
    electrical = KirchhoffLayer(
        net, "electrical",
        [PrescribedVoltage(_f(V)), Resistance(_f(R0), alpha=_f(ALPHA), T_ref=_f(T_REF),
                                              temperature="electrical.resistor.T")],
        boundary=["0"])
    thermal = TransportLayer(net, "thermal", flow_kind=(), boundary=["amb"],
                             capacity=_f([0.0, C_TH]), conduction_kind="conductor",
                             conductance=_f([G1, G]), storage_free=["h"])
    closure = JouleHeating([HeatedEdge(edge=1, p=1, n=0, heat_node=2,
                                       temperature_key="electrical.resistor.T", slot=0)],
                           thermal, base_temperatures={"electrical.resistor.T": _f([T_REF])})
    model = Model(net, [electrical, thermal], closures=[closure], **COUPLING)
    state = {"electrical.phi": _f([0.0, V, 0.0, 0.0, 0.0]),
             "electrical.q": _f([V / R0, V / R0]), "thermal.x": _f([400.0, 400.0]),
             "thermal.sources": torch.zeros(5, dtype=F64)}
    drivers = {"electrical.phi_boundary": _f([0.0]), "thermal.x_boundary": _f([T_AMB]),
               "electrical.resistor.T": _f([T_REF])}
    T_h = 400.0
    for _ in range(100):      # the hand fixed point of the port's balance
        T_h = 400.0 + _power(T_h) / G1
    out = model.steady(state, drivers, mode="initial")
    assert float(out["thermal.x"][1]) == 400.0
    assert abs(float(out["thermal.x"][0]) - T_h) <= 1e-9
    assert abs(float(out["electrical.q"][1]) - _power(T_h) / V) <= 1e-11
    dc = model.steady(state, drivers)
    assert abs(float(dc["thermal.x"][1]) - 400.0) > 50.0      # the DC point is elsewhere
