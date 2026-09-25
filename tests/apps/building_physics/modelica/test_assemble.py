"""`read_modelica` / `assemble.build` / `run.simulate` on hand-written `noodl-modelica/1` fixtures.

Reference values are computed here from the MBL source formulas (cited per test), never by
calling the elements under test. OpenModelica parity lives in Tasks 8-9.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import ModelicaImportError
from noodl.apps.building_physics.modelica.run import simulate, step_drivers

FIX = Path(__file__).parent / "fixtures"
F64 = torch.float64

# MSL SingleGasesData.mo:5,49,9187 (R_NASA_2002, Air.MM, H2O.MM).
R_AIR = 8.314510 / 0.0289651159
R_H2O = 8.314510 / 0.01801528
P_DEFAULT = 101325.0


def _load(name: str):
    return read_modelica(FIX / name, return_names=True)


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(doc))
    return path


def _doc(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# ---------------------------------------------------------------------------- (a) orifice
def test_orifice_between_two_boundaries_matches_the_mbl_power_law():
    model, state, drivers, names = _load("two_zones_orifice.json")
    assert not model.transport  # no volumes: algebraic in the boundary values
    hist = simulate(model, state, drivers, names.times[:3])
    # PerfectGas rho_default (PerfectGas.mo:229-231 at p_default, T_default, X_default).
    rho = P_DEFAULT / ((R_AIR * 0.99 + R_H2O * 0.01) * 293.15)
    C = 0.6 * 0.01 * math.sqrt(2.0 / rho)  # Orifice.mo:3-5
    expected = rho * C * 5.0**0.5  # Coefficient_V_flow.mo:4, |dp| > dp_turbulent
    ((col, sign),) = names.edges["ori"]
    got = sign * hist["air.q"][:, col]
    assert torch.allclose(got, torch.full_like(got, expected), rtol=1e-12, atol=0.0)


# ------------------------------------------------------------------- (b) two-volume door
def test_door_between_two_volumes_exchanges_air_with_zero_net_flow_and_mixes():
    model, state, drivers, names = _load("door_two_volumes.json")
    hist = simulate(model, state, drivers, names.times)  # 0..60 s
    (c_ab, s_ab), (c_ba, s_ba) = names.edges["doo"]
    q_ab, q_ba = s_ab * hist["air.q"][:, c_ab], s_ba * hist["air.q"][:, c_ba]
    iA, iB = names.nodes["volA"], names.nodes["volB"]
    # Equal pressures (one zone of the closed pair is the pressure reference, the other is
    # solved): the pressure term vanishes and the two directions are +-mABt.
    assert torch.allclose(hist["p"][:, iA], hist["p"][:, iB], rtol=0.0, atol=1e-9)
    assert float(q_ab[0]) > 1e-3  # warm A to B at the top
    assert torch.allclose(q_ab, -q_ba, rtol=1e-12, atol=1e-15)
    assert float((q_ab + q_ba).abs().max()) < 1e-12
    T_A, T_B = hist["T"][:, iA], hist["T"][:, iB]
    assert float(T_A[0]) == pytest.approx(295.15)
    assert float(T_A[-1]) < float(T_A[0]) - 0.1
    assert float(T_B[-1]) > float(T_B[0]) + 0.1
    # Equal volumes and capacities: the mean temperature is conserved by the exchange.
    assert torch.allclose(T_A + T_B, torch.full_like(T_A, 295.15 + 293.15), atol=1e-9)
    # The door port flows are exposed separately: port_a2.m_flow = -(the "ba" edge flow).
    ((c2, s2),) = names.edges["doo.port_a2"]
    assert c2 == c_ba and s2 == -1


# --------------------------------------------------------------------- (c) Ramp driver
def test_ramp_on_a_boundary_pressure_becomes_the_driver_series():
    model, state, drivers, names = _load("ramp_boundary.json")
    t = names.times
    series = drivers["series:air.phi_boundary"]
    air = model.potential["air"]
    j = air.bound.tolist().index(names.nodes["bouA"])
    # Sources.mo:244-252 with height 10, duration 30, offset 101325, startTime 10, as gauge
    # pressure relative to p_default.
    expected = torch.where(t < 10.0, torch.zeros_like(t),
                           torch.where(t < 40.0, (t - 10.0) * 10.0 / 30.0,
                                       torch.full_like(t, 10.0)))
    assert torch.allclose(series[:, j], expected, rtol=0.0, atol=1e-10)
    k = air.bound.tolist().index(names.nodes["bouB"])
    assert torch.all(series[:, k] == 0.0)
    hist = simulate(model, state, drivers, t)
    ((col, sign),) = names.edges["ori"]
    rho = 1.2  # Buildings.Media.Air rho_default (Air.mo:43-45, 210-215)
    C = 0.65 * 0.01 * math.sqrt(2.0 / rho)
    assert float(sign * hist["air.q"][-1, col]) == pytest.approx(rho * C * math.sqrt(10.0),
                                                                rel=1e-12)
    assert float(hist["air.q"][0, col]) == 0.0


def test_step_drivers_slices_the_series_and_keeps_constants():
    model, state, drivers, names = _load("ramp_boundary.json")
    d = step_drivers(drivers, names.times, 25.0)
    air = model.potential["air"]
    j = air.bound.tolist().index(names.nodes["bouA"])
    assert float(d["air.phi_boundary"][j]) == pytest.approx(5.0)
    assert not any(key.startswith("series:") for key in d)
    with pytest.raises(ValueError, match="not on the experiment grid"):
        step_drivers(drivers, names.times, 25.5)


# ------------------------------------------------------------ (d) pinned temperature
def test_fixed_temperature_through_a_stiff_conductor_pins_the_zone():
    model, state, drivers, names = _load("thermal_and_source.json")
    i = names.nodes["volA"]
    assert "thermal" not in model.transport  # the only zone is pinned: no thermal unknown
    assert "species" in model.transport  # CO2
    hist = simulate(model, state, drivers, names.times[:11])
    assert torch.all(hist["T"][:, i] == 298.15)


def test_a_finite_conductor_is_refused_by_name():
    with pytest.raises(ModelicaImportError, match="conA") as exc:
        read_modelica(FIX / "conductor_too_small.json")
    assert "ThermalConductor" in str(exc.value)
    assert "1e+06" in str(exc.value) or "1000000" in str(exc.value)


# ---------------------------------------------------------------- (e) DelayFirstOrder
def test_delay_first_order_zone_gets_the_mbl_volume():
    model, _state, _drivers, names = _load("delay_zone.json")
    # DelayFirstOrder.mo:5-6,11-12: V = V_nominal = m_flow_nominal*tau/rho_default.
    V = model.net.node_attr("volume")[names.nodes["del"]]
    assert float(V) == pytest.approx(0.1 * 60.0 / 1.2, rel=1e-15)
    assert "thermal" in model.transport


# ------------------------------------------------------------------------- zonal flows
def test_zonal_flows_are_prescribed_directional_edges_and_carry_moisture():
    model, state, drivers, names = _load("zonal_flow.json")
    assert "species" in model.transport  # X_start differs between the rooms: X_w carried
    hist = simulate(model, state, drivers, names.times)  # 0..600 s, every 10 s
    (za, sa), (zb, sb) = names.edges["zonFlo"]
    # ZonalFlow_ACS.mo:38-42: m_flow = V*ACS*(density(sta_a1) + density(sta_a2))/2, and
    # Buildings.Media.Air's density is p*dStp/pStp (Air.mo:210-215) at the port pressure,
    # here p_start = p_default in both rooms.
    m = 1.0 * 5.0 / 3600.0 * 1.2
    n = names.times.numel()
    assert torch.allclose(sa * hist["air.q"][:, za], torch.full((n,), m, dtype=F64),
                          rtol=1e-12)
    assert torch.allclose(sb * hist["air.q"][:, zb], torch.full((n,), -m, dtype=F64),
                          rtol=1e-12)
    (fa, s1), (fb, s2) = names.edges["floExc"]
    assert torch.allclose(s1 * hist["air.q"][:, fa], torch.full((n,), 0.02, dtype=F64))
    assert torch.allclose(s2 * hist["air.q"][:, fb], torch.full((n,), -0.02, dtype=F64))
    assert names.air_references == ("rooA", "rooB")  # no boundary, no pressure path
    iA, iB = names.nodes["rooA"], names.nodes["rooB"]
    X = hist["X_w"]
    assert float(X[0, iA]) == pytest.approx(0.015) and float(X[0, iB]) == pytest.approx(0.01)
    # The small room (1.2 kg of air, 0.0217 kg/s in) takes on the large room's moisture and
    # temperature within a few 55 s time constants.
    assert float(X[-1, iB]) > 0.0149
    assert float(hist["T"][-1, iB]) > 302.9


# -------------------------------------------------- mixed: doors, stack, sources, balance
def test_mixed_model_conserves_mass_and_accumulates_the_trace_substance():
    model, state, drivers, names = _load("mixed_rooms.json")
    assert len(names.edges["dooDis"]) == 4  # nCom = 4 compartment edges
    hist = simulate(model, state, drivers, names.times)
    air = model.potential["air"]
    q = hist["air.q"]
    s = drivers["air.sources"]
    net_out = torch.stack([air._accumulate(q[k]) for k in range(q.shape[0])])
    for zone in ("volA", "volB", "volTop"):
        i = names.nodes[zone]
        # Quasi-steady mass balance at every zone: outflow = injected mass (the CO2 source).
        assert torch.allclose(net_out[:, i], s[i].expand(q.shape[0]), rtol=0.0, atol=1e-10)
    assert float(s[names.nodes["volA"]]) == pytest.approx(1e-5)
    C = hist["C"][:, names.nodes["volA"], 0]
    assert float(C[0]) == 0.0 and float(C[-1]) > float(C[1]) > 0.0
    # The step on the door opening changes the door flow magnitude.
    cols = [c for c, _ in names.edges["dooDis"]]
    before = q[4, cols].abs().sum()
    after = q[6, cols].abs().sum()
    assert float(after) > 5.0 * float(before)


# ------------------------------------------------------------------------ gradients
def test_gradient_of_a_zone_temperature_wrt_an_orifice_coefficient():
    model, state, drivers, names = _load("zone_two_orifices.json")
    kind = names.kinds["oriA"][0]
    el, _ = model.potential["air"].element_for(kind)
    d = step_drivers(drivers, names.times, 1.0)
    el.C.requires_grad_(True)
    new = model.step(state, d, 1.0)
    T = new["thermal.x"][0]
    (g,) = torch.autograd.grad(T, el.C)
    h = 1e-4 * float(el.C.detach().abs())
    with torch.no_grad():
        c0 = el.C.detach().clone()
        el.C.copy_(c0 + h)
        tp = float(model.step(state, d, 1.0)["thermal.x"][0])
        el.C.copy_(c0 - h)
        tm = float(model.step(state, d, 1.0)["thermal.x"][0])
        el.C.copy_(c0)
    assert float(g) == pytest.approx((tp - tm) / (2 * h), rel=1e-6)
    assert float(g) > 0.0  # more hot inflow warms the zone faster


# ------------------------------------------------------------------------ refusals
def test_steady_state_energy_balance_is_refused(tmp_path):
    doc = _doc("zone_two_orifices.json")
    doc["components"][2]["parameters"]["energyDynamics"] = "SteadyState"
    with pytest.raises(ModelicaImportError, match="vol .*energyDynamics"):
        read_modelica(_write(tmp_path, doc))


def test_actual_column_density_is_refused(tmp_path):
    doc = _doc("mixed_rooms.json")
    for c in doc["components"]:
        if c["name"] == "colTop":
            c["parameters"]["densitySelection"] = "actual"
    with pytest.raises(ModelicaImportError, match="colTop"):
        read_modelica(_write(tmp_path, doc))


def test_missing_and_unused_signals_are_refused_together(tmp_path):
    doc = _doc("ramp_boundary.json")
    doc["signals"][0]["drives"] = "bouB.T_in"  # bouA.p_in now undriven, bouB.T_in unused
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica(_write(tmp_path, doc))
    msg = str(exc.value)
    assert "bouA" in msg and "p_in" in msg
    assert "ramp" in msg and "bouB.T_in" in msg


def test_use_default_properties_false_is_refused(tmp_path):
    doc = _doc("two_zones_orifice.json")
    doc["components"][2]["parameters"]["useDefaultProperties"] = False
    with pytest.raises(ModelicaImportError, match="ori"):
        read_modelica(_write(tmp_path, doc))


def test_unknown_medium_is_refused(tmp_path):
    doc = copy.deepcopy(_doc("two_zones_orifice.json"))
    doc["medium"]["class"] = "Modelica.Media.Water.StandardWater"
    with pytest.raises(ModelicaImportError, match="StandardWater"):
        read_modelica(_write(tmp_path, doc))


def test_read_modelica_returns_the_contam_route_triple():
    out = read_modelica(FIX / "door_two_volumes.json")
    assert len(out) == 3
    model, state, drivers = out
    assert "thermal.x" in state and "air.phi" in state
    assert drivers["air.phi_boundary"].dtype == F64


# ----------------------------------------------------------- hydrostatic column chain
def test_fused_column_chain_holds_the_hydrostatic_balance_at_zero_flow():
    # stack_chain.json: volWes - colWesBot(fromBottom) - oriWesTop - colWesTop(fromTop) -
    # volTop, and nothing else, so the orifice carries no flow and (graph.py docstring,
    # MediumColumn.mo `port_a.p - port_b.p = -h rho g_n`) p(volWes) - p(volTop) =
    # h g (rho(volTop) + rho(volWes)) with rho = density_pTX(p_default, T, X_default).
    model, state, drivers, names = _load("stack_chain.json")
    hist = simulate(model, state, drivers, names.times[:2])
    ((col, _),) = names.edges["oriWesTop"]
    assert float(hist["air.q"][0, col].abs()) < 1e-12

    def rho(T):
        return P_DEFAULT / ((R_AIR * 0.99 + R_H2O * 0.01) * T)

    expected = 1.5 * 9.80665 * (rho(293.15) + rho(298.15))
    phi = hist["air.phi"][0]
    got = phi[names.nodes["volWes"]] - phi[names.nodes["volTop"]]
    assert float(got) == pytest.approx(expected, rel=1e-9)
    assert names.air_references == ("volWes",)


# --------------------------------------------------------------------- mass sources
def _with_supply(m_flow: float) -> dict:
    doc = _doc("zone_two_orifices.json")
    doc["components"][2]["parameters"]["nPorts"] = 3
    doc["components"].append({
        "name": "sup", "class": "Buildings.Fluid.Sources.MassFlowSource_T",
        "parameters": {"m_flow": m_flow, "T": 310.0, "nPorts": 1},
    })
    doc["connections"].append(["sup.ports[1]", "vol.ports[3]"])
    return doc


def test_mass_flow_source_injects_mass_and_enthalpy(tmp_path):
    model, state, drivers, names = read_modelica(_write(tmp_path, _with_supply(0.01)),
                                                 return_names=True)
    i = names.nodes["vol"]
    cp = 1006.0 * 0.99 + 1860.0 * 0.01  # Air.mo:567-575 at X_default
    assert float(drivers["air.sources"][i]) == pytest.approx(0.01)
    assert float(drivers["thermal.sources"][i]) == pytest.approx(cp * 0.01 * 310.0)
    hist = simulate(model, state, drivers, names.times[:3])
    air = model.potential["air"]
    assert float(air._accumulate(hist["air.q"][-1])[i]) == pytest.approx(0.01, abs=1e-10)


def test_extracting_mass_flow_source_is_refused(tmp_path):
    with pytest.raises(ModelicaImportError, match="sup .*negative"):
        read_modelica(_write(tmp_path, _with_supply(-0.01)))


def test_outside_without_weather_signals_is_refused(tmp_path):
    doc = _doc("two_zones_orifice.json")
    doc["components"][1]["class"] = "Buildings.Fluid.Sources.Outside"
    with pytest.raises(ModelicaImportError, match="bouB .*weather"):
        read_modelica(_write(tmp_path, doc))


# ---------------------------------------------- every one-way class and both open doors
def _two_boundaries(element: dict, signals=(), dp: float = 5.0) -> dict:
    doc = _doc("ramp_boundary.json")
    doc["components"] = [
        {"name": "bouA", "class": "Buildings.Fluid.Sources.Boundary_pT",
         "parameters": {"p": P_DEFAULT + dp, "T": 293.15, "nPorts": 2}},
        {"name": "bouB", "class": "Buildings.Fluid.Sources.Boundary_pT",
         "parameters": {"p": P_DEFAULT, "T": 293.15, "nPorts": 2}},
        element,
    ]
    ports = (("port_a1", "port_b2"), ("port_b1", "port_a2")) if "Door" in element["class"] \
        else (("port_a",), ("port_b",))
    doc["connections"] = (
        [[f"bouA.ports[{i + 1}]", f"el.{p}"] for i, p in enumerate(ports[0])]
        + [[f"el.{p}", f"bouB.ports[{i + 1}]"] for i, p in enumerate(ports[1])]
    )
    doc["signals"] = list(signals)
    doc["experiment"]["StopTime"] = 2.0
    return doc


_M = "Buildings.Airflow.Multizone."
_CVAL_DOOR = 0.65 * 0.9 * 2.1 * math.sqrt(2.0 / 1.2)  # DoorOpen.mo:27, Door.mo:41


@pytest.mark.parametrize(
    ("cls", "params", "expected"),
    [
        # Point_m_flow.mo:4-5: k = mMea/dpMea^m.
        ("Point_m_flow", {"dpMea_nominal": 10.0, "mMea_flow_nominal": 0.02, "m": 0.6},
         0.02 * 0.5**0.6),
        # Points_m_flow.mo: the fit passes through its first point (5 Pa, 0.01 kg/s).
        ("Points_m_flow", {"dpMea_nominal": [5.0, 20.0], "mMea_flow_nominal": [0.01, 0.03]},
         0.01),
        # Coefficient_V_flow.mo:4: m_flow = rho_default*C*dp^m.
        ("Coefficient_V_flow", {"C": 0.01, "m": 0.6}, 1.2 * 0.01 * 5.0**0.6),
        # A collinear table: the monotone Hermite spline is the line itself.
        ("Table_m_flow", {"dpMea_nominal": [-10.0, 0.0, 10.0],
                          "mMea_flow_nominal": [-0.02, 0.0, 0.02]}, 0.01),
        ("Table_V_flow", {"dpMea_nominal": [-10.0, 0.0, 10.0],
                          "VMea_flow_nominal": [-0.02, 0.0, 0.02]}, 1.2 * 0.01),
        # Equal temperatures: no buoyancy term, the two edges share the orifice flow
        # (Door.mo:64-65 with mABt = 0).
        ("DoorOpen", {}, 1.2 * _CVAL_DOOR * math.sqrt(5.0)),
    ],
)
def test_one_way_classes_and_open_door_between_two_boundaries(tmp_path, cls, params,
                                                               expected):
    doc = _two_boundaries({"name": "el", "class": _M + cls, "parameters": params})
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    net = sum(s * hist["air.q"][:, c] for c, s in names.edges["el"])
    assert torch.allclose(net, torch.full_like(net, expected), rtol=1e-10)


def test_operable_and_discretised_doors_build_and_follow_their_signal(tmp_path):
    y = [{"name": "y", "class": "Modelica.Blocks.Sources.Step",
          "parameters": {"startTime": 1.0}, "drives": "el.y"}]
    doc = _two_boundaries({"name": "el", "class": _M + "DoorOperable",
                           "parameters": {"LClo": 0.001}}, y)
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    hist = simulate(model, state, drivers, names.times)
    net = sum(s * hist["air.q"][:, c] for c, s in names.edges["el"])
    assert float(net[2]) == pytest.approx(1.2 * _CVAL_DOOR * math.sqrt(5.0), rel=1e-10)
    assert 0.0 < float(net[0]) < 0.1 * float(net[2])  # closed crack before the step
    doc = _two_boundaries({"name": "el", "class": _M + "DoorDiscretizedOpen",
                           "parameters": {"nCom": 3}})
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    assert len(names.edges["el"]) == 3
    hist = simulate(model, state, drivers, names.times[:1])
    assert float(sum(s * hist["air.q"][0, c] for c, s in names.edges["el"])) > 0.0


def test_boundary_temperature_and_concentration_inputs_reach_the_zone(tmp_path):
    # zone_two_orifices: bouA -> oriA -> vol -> oriB -> bouB, flow independent of the zone
    # state. bouA's T_in and C_in[1] come from Constant signals, so the zone relaxes as
    # x(t) = x_in + (x0 - x_in) exp(-F t/(rho V)) with the (constant) through-flow F; the
    # exact transport scheme integrates exactly that on frozen flows.
    doc = _doc("zone_two_orifices.json")
    doc["medium"]["extraPropertiesNames"] = ["CO2"]
    doc["components"][0]["parameters"].update({"use_T_in": True, "use_C_in": True})
    doc["signals"] = [
        {"name": "TIn", "class": "Modelica.Blocks.Sources.Constant",
         "parameters": {"k": 313.15}, "drives": "bouA.T_in"},
        {"name": "CIn", "class": "Modelica.Blocks.Sources.Constant",
         "parameters": {"k": 4e-4}, "drives": "bouA.C_in[1]"},
    ]
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    t = names.times[:31]
    hist = simulate(model, state, drivers, t)
    ((col, sign),) = names.edges["oriA"]
    F = float(sign * hist["air.q"][0, col])
    decay = torch.exp(-F * t / (1.2 * 10.0))
    i = names.nodes["vol"]
    assert torch.allclose(hist["T"][:, i], 313.15 + (293.15 - 313.15) * decay, rtol=0.0,
                          atol=1e-7)
    assert torch.allclose(hist["C"][:, i, 0], 4e-4 * (1.0 - decay), rtol=0.0, atol=1e-13)


# ------------------------------------------------ transport time course (review item 2)
def test_heat_and_moisture_relax_on_the_same_exact_time_course():
    # Two rooms exchanging F = V ACS rho + m_flow each way (balanced), no boundary. Both the
    # temperature and the water mass fraction obey dD/dt = -F (1/M_A + 1/M_B) D for the
    # normalised room difference D, with M = rho_start V (the heat capacity's cp cancels
    # the carrier's). Frozen flows make the exact scheme exact, so D matches the
    # exponential to the iteration and solve tolerances (1e-8 K on a 10 K difference,
    # 1e-12 on 0.005).
    model, state, drivers, names = _load("zonal_flow.json")
    hist = simulate(model, state, drivers, names.times)
    iA, iB = names.nodes["rooA"], names.nodes["rooB"]
    F = 5.0 / 3600.0 * 1.2 * 1.0 + 0.02
    rate = F * (1.0 / (1.2 * 100.0) + 1.0 / (1.2 * 1.0))
    expected = torch.exp(-rate * names.times)
    DT = (hist["T"][:, iA] - hist["T"][:, iB]) / (303.15 - 293.15)
    DX = (hist["X_w"][:, iA] - hist["X_w"][:, iB]) / (0.015 - 0.01)
    for k in (1, 2, 3, 6, 12):
        assert float(DT[k]) == pytest.approx(float(expected[k]), rel=1e-7, abs=1e-9)
        assert float(DX[k]) == pytest.approx(float(expected[k]), rel=1e-7, abs=1e-9)
        assert abs(float(DT[k] - DX[k])) < 1e-8


# ------------------------------------------ unbalanced closed groups (review item 1)
def test_unbalanced_closed_zone_groups_are_refused_by_name(tmp_path):
    doc = _doc("zonal_flow.json")
    doc["signals"][1]["drives"] = "floExc.mAB_flow"
    doc["signals"].append({"name": "m_flow2", "class": "Modelica.Blocks.Sources.Constant",
                           "parameters": {"k": 0.01}, "drives": "floExc.mBA_flow"})
    doc["components"][0]["parameters"]["nPorts"] = 5
    doc["components"].append({
        "name": "sou", "class": "Buildings.Fluid.Sources.TraceSubstancesFlowSource",
        "parameters": {"nPorts": 1, "m_flow": 1e-6, "substanceName": "CO2"}})
    doc["medium"]["extraPropertiesNames"] = ["CO2"]
    doc["connections"].append(["sou.ports[1]", "rooA.ports[5]"])
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica(_write(tmp_path, doc))
    msg = str(exc.value)
    assert "floExc" in msg and "ZonalFlow_m_flow" in msg and "not balanced" in msg
    assert "sou" in msg and "TraceSubstancesFlowSource" in msg and "no boundary" in msg


def test_equal_constants_on_both_zonal_directions_count_as_balanced(tmp_path):
    doc = _doc("zonal_flow.json")
    doc["signals"][1]["drives"] = "floExc.mAB_flow"
    doc["signals"].append({"name": "m_flow2", "class": "Modelica.Blocks.Sources.Constant",
                           "parameters": {"k": 0.02}, "drives": "floExc.mBA_flow"})
    _model, _state, _drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    assert names.air_references == ("rooA", "rooB")


# ------------------------------------------------------ trace sources (review item 5)
def test_trace_substance_name_is_matched_case_insensitively(tmp_path):
    doc = _doc("mixed_rooms.json")
    doc["components"][-1]["parameters"]["substanceName"] = "co2"
    _model, _state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    assert float(drivers["species.sources"][names.nodes["volA"], 0]) == pytest.approx(1e-5)
    doc["components"][-1]["parameters"]["substanceName"] = "NO2"
    with pytest.raises(ModelicaImportError, match="sou .*NO2"):
        read_modelica(_write(tmp_path, doc))


def test_mass_flow_source_composition_inputs_are_refused(tmp_path):
    doc = _with_supply(0.01)
    doc["components"][-1]["parameters"]["use_X_in"] = True
    with pytest.raises(ModelicaImportError, match="sup .*use_X_in"):
        read_modelica(_write(tmp_path, doc))


def test_iteration_converges_to_the_tight_tolerances():
    for name in ("door_two_volumes.json", "mixed_rooms.json", "zonal_flow.json"):
        model, state, drivers, names = _load(name)
        assert model.iterate_tol.get("thermal") == pytest.approx(1e-8)
        if "species" in model.transport:
            assert model.iterate_tol["species"] == pytest.approx(1e-12)
        d = step_drivers(drivers, names.times, float(names.times[1]))
        diag: dict = {}
        model.step(state, d, float(names.times[1] - names.times[0]), diagnostics=diag,
                   atol=1e-13, rtol=1e-12)
        assert bool(diag["converged"].all())
        assert diag["passes"] < model.iterate_max


# ------------------------------------------------------------------ in-line flow sensors
def test_inline_sensors_map_to_the_flow_of_the_element_in_series():
    """`names.edges[<sensor>]` is the sensor's own `port_a.m_flow`
    (`PartialFlowSensor.mo:13`): the flow of the element it sits in series with, signed by
    which way round the sensor is wired."""
    model, state, drivers, names = _load("inline_sensors.json")
    hist = simulate(model, state, drivers, names.times[:3])
    q = hist["air.q"]

    def flow(key):
        return sum(s * q[:, c] for c, s in names.edges[key])

    # ori.port_a -> senOri -> bouB: the sensor carries ori's own port_a.m_flow.
    assert names.edges["senOri"] == names.edges["ori"]
    # doo.port_a2 -> senDoo.port_a: the sensor carries -port_a2.m_flow (flow B -> A in the
    # door's second direction leaves port_a2 into the sensor).
    assert torch.equal(flow("senDoo"), -flow("doo.port_a2"))
    # senC1 is wired backwards (its port_b faces oriCol): it reads minus oriCol's flow, and
    # senC2 (port_a facing the column) likewise.
    assert torch.equal(flow("senC1"), -flow("oriCol"))
    assert torch.equal(flow("senC2"), -flow("oriCol"))
    # The network is the one without sensors: bouA 5 Pa above bouB drives ori forwards.
    assert bool((flow("ori") > 0).all())


# ------------------------------------------------------------------ Math signal chains
def test_math_signal_chain_matches_the_closed_form():
    """Boundary inputs fed by chains of `Modelica.Blocks.Math` blocks (MSL `Math.mo`: Add
    :880, Sum :791, Gain :552, Product :976) over sources: `bouA.p_in = 2 ramp + pAmb`,
    `bouB.p_in = pAmb - 3`, `bouB.T_in = 290 * (1 * step)`."""
    model, state, drivers, names = _load("math_chain.json")
    t = names.times
    phi = drivers["series:air.phi_boundary"]
    ramp = 10.0 * torch.clamp(t / 10.0, max=1.0)
    assert torch.allclose(phi[:, 0], 2.0 * ramp + 101325.0 - P_DEFAULT, rtol=0, atol=1e-9)
    assert torch.equal(phi[:, 1], torch.full_like(t, -3.0))
    T = drivers["series:thermal.x_boundary"]
    # t <= 5: at its startTime 5 the Step takes the left limit (signals module docstring).
    expected_T = torch.where(t <= 5.0, torch.full_like(t, 290.0), torch.full_like(t, 290.0 * 1.1))
    assert torch.allclose(T[:, 1], expected_T, rtol=1e-15, atol=0.0)
    hist = simulate(model, state, drivers, t[:4])
    ((col, sign),) = names.edges["ori"]
    assert bool((sign * hist["air.q"][1:, col] > 0).all())  # bouA above bouB once ramping


def _chain_doc(**edits):
    doc = _doc("math_chain.json")
    sig = {s["name"]: s for s in doc["signals"]}
    for name, fields in edits.items():
        if fields is None:
            doc["signals"].remove(sig[name])
        else:
            sig[name].update(fields)
    return doc


def test_math_block_loop_is_refused(tmp_path):
    doc = _chain_doc(gai={"drives": ["prod.u2", "add.u2"]}, pAmb={"drives": "sum.u[1]"},
                     add={"drives": ["bouA.p_in", "gai.u"]}, ste=None)
    with pytest.raises(ModelicaImportError, match=r"add \(Modelica.Blocks.Math.Add\).*loop"):
        read_modelica(_write(tmp_path, doc))


def test_math_block_with_an_undriven_input_is_refused(tmp_path):
    doc = _chain_doc(off=None)
    with pytest.raises(ModelicaImportError, match=r"sum \(Modelica.Blocks.Math.Sum\).*u\[2\]"):
        read_modelica(_write(tmp_path, doc))


def test_unsupported_math_block_is_refused_by_name(tmp_path):
    doc = _chain_doc(gai={"class": "Modelica.Blocks.Math.Abs"})
    with pytest.raises(ModelicaImportError, match=r"gai \(Modelica.Blocks.Math.Abs\)"):
        read_modelica(_write(tmp_path, doc))


# ----------------------------------------------------------------- prescribed heat flow
def test_prescribed_heat_flow_heats_the_zone_at_the_closed_form_rate():
    """`PrescribedHeatFlow.mo:15` `port.Q_flow = -Q_flow (1 + alpha (T - T_ref))` with
    `alpha = 0` puts `Q_flow` into the volume's energy balance (`PartialMixingVolume.mo:185-
    189`, `ConservationEquation.mo:303` `der(U) = Hb_flow + Q_flow`). No air moves (one
    orifice to a boundary at the zone's pressure), so `T = T_start + Q t / (V rho cp)`."""
    model, state, drivers, names = _load("heat_flow.json")
    t = names.times
    hist = simulate(model, state, drivers, t)
    rho = 1.2 * P_DEFAULT / 101325.0  # Air.mo:210-215, d = p dStp/pStp
    cp = 1006.0 * 0.99 + 1860.0 * 0.01  # Air.mo:567-575 at X_default
    expected = 293.15 + 100.0 * t / (10.0 * rho * cp)
    T = hist["T"][:, names.nodes["vol"]]
    assert torch.allclose(T, expected, rtol=1e-12, atol=0.0)


def test_a_heat_pulse_inside_one_step_delivers_its_energy(tmp_path):
    """Source drivers are step means (module docstring, "Sources"): a 3 s pulse of 100 W
    between the 10 s output times of `heat_flow.json` is zero at every grid time, yet the
    zone must gain its 300 J, `T = T_start + 300 / (V rho cp)` from that step on."""
    doc = _doc("heat_flow.json")
    doc["signals"][0] = {"name": "one", "class": "Modelica.Blocks.Sources.Pulse",
                         "parameters": {"amplitude": 1.0, "width": 3.0, "period": 100.0,
                                        "startTime": 23.0, "nperiod": 1},
                         "drives": "gai.u"}
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    t = names.times
    i = names.nodes["vol"]
    Q = drivers["series:thermal.sources"][:, i]
    expected_Q = torch.zeros_like(t)
    expected_Q[3] = 100.0 * 3.0 / 10.0  # the step (20, 30) holds the pulse [23, 26)
    assert torch.allclose(Q, expected_Q, rtol=1e-13, atol=1e-12)
    hist = simulate(model, state, drivers, t)
    rho = 1.2 * P_DEFAULT / 101325.0  # Air.mo:210-215
    cp = 1006.0 * 0.99 + 1860.0 * 0.01  # Air.mo:567-575 at X_default
    rise = torch.tensor(300.0 / (10.0 * rho * cp), dtype=F64)
    expected = 293.15 + torch.where(t >= 30.0, rise, torch.zeros_like(t))
    assert torch.allclose(hist["T"][:, i], expected, rtol=1e-12, atol=0.0)


def test_a_trace_pulse_inside_one_step_injects_its_mass():
    """`Examples/CO2TransportStep`'s source: 8.18e-6 kg/s for 3.6 s at 3600 s, between the
    3456 s and 3628.8 s output times. The air, species and (moisture off) sources of that
    step are the pulse's mean, every other step's zero."""
    path = Path(__file__).parents[3] / "data" / "modelica" / "CO2TransportStep.json"
    _model, _state, drivers, names = read_modelica(path, return_names=True)
    i = names.nodes["volWes"]
    dt = names.times[1:] - names.times[:-1]
    mass = drivers["series:air.sources"][1:, i] * dt
    co2 = drivers["series:species.sources"][1:, i, 0] * dt
    assert float(mass.sum()) == pytest.approx(8.18e-6 * 3.6, rel=1e-12)
    assert torch.equal(mass, co2)
    assert int((mass != 0).sum()) == 1


def test_simulate_refuses_a_grid_that_skips_source_intervals():
    """Source drivers are step means over the driver grid's intervals (module docstring,
    "Sources"), so a `times` grid that skips intervals, or leaves the grid, would drop
    injected amounts: refused by name. A run of consecutive grid times is accepted."""
    path = Path(__file__).parents[3] / "data" / "modelica" / "CO2TransportStep.json"
    model, state, drivers, names = read_modelica(path, return_names=True)
    with pytest.raises(ValueError, match=r"consecutive grid times; times\[1\] = 345.6"):
        simulate(model, state, drivers, names.times[:5:2])
    with pytest.raises(ValueError, match="is not a grid time"):
        simulate(model, state, drivers, torch.tensor([0.0, 100.0], dtype=F64))
    hist = simulate(model, state, drivers, names.times[2:4])
    assert hist["time"].tolist() == names.times[2:4].tolist()


def test_temperature_dependent_heat_flow_is_refused(tmp_path):
    doc = _doc("heat_flow.json")
    doc["components"][3]["parameters"]["alpha"] = 0.01
    with pytest.raises(ModelicaImportError, match=r"preHea .*alpha"):
        read_modelica(_write(tmp_path, doc))


# ------------------------------------------------------- boundary wired straight to a volume
def test_boundary_wired_straight_to_a_volume_sets_its_pressure():
    """The attached boundary makes its volume the air-layer pressure reference of the volume's
    group at the boundary's own pressure. The group has no other boundary, so the boundary
    exchanges no air (the door's net flow into the closed volB is zero) and its temperature
    never enters: the two rooms mix exactly as without it."""
    model, state, drivers, names = _load("attached_boundary.json")
    assert names.air_references == ("volA",)
    assert names.attached == {"bou": "volA"}
    hist = simulate(model, state, drivers, names.times)
    iA, iB = names.nodes["volA"], names.nodes["volB"]
    assert torch.allclose(hist["p"][:, iA], torch.full_like(hist["p"][:, iA], 101327.0),
                          rtol=0.0, atol=1e-9)
    (c_ab, s_ab), (c_ba, s_ba) = names.edges["doo"]
    net = s_ab * hist["air.q"][:, c_ab] + s_ba * hist["air.q"][:, c_ba]
    assert float(net.abs().max()) < 1e-12
    T_A, T_B = hist["T"][:, iA], hist["T"][:, iB]
    assert float(T_A[-1]) < float(T_A[0]) - 0.1
    assert torch.allclose(T_A + T_B, torch.full_like(T_A, 295.15 + 293.15), atol=1e-9)


def test_attached_boundary_in_a_group_with_another_boundary_is_refused(tmp_path):
    doc = _doc("attached_boundary.json")
    doc["components"][2]["parameters"]["nPorts"] = 3
    doc["components"] += [
        {"name": "bouX", "class": "Buildings.Fluid.Sources.Boundary_pT",
         "parameters": {"nPorts": 1}},
        {"name": "oriX", "class": "Buildings.Airflow.Multizone.Orifice",
         "parameters": {"A": 0.01}},
    ]
    doc["connections"] += [["volB.ports[3]", "oriX.port_a"], ["oriX.port_b", "bouX.ports[1]"]]
    with pytest.raises(ModelicaImportError,
                       match=r"bou \(Buildings.Fluid.Sources.Boundary_pT\): wired straight to "
                             r"volA"):
        read_modelica(_write(tmp_path, doc))


# --------------------------------------------------------------- airflow solve robustness
def test_gauge_reference_is_the_first_boundary_pressure(tmp_path):
    """With the boundary at 1e5 Pa (`Examples/ReverseBuoyancy.mo` sets `volOut.p = 100000`)
    a `p_default` gauge would carry 1325 Pa in every potential, whose round-off (3e-13 Pa per
    `dp`) the door's stiff laminar branch turns into a residual floor above `AIR_ATOL`; the
    reader gauges against the boundary instead (assemble docstring, "Gauge reference")."""
    doc = _doc("mixed_rooms.json")
    for c in doc["components"]:
        if c["name"] == "bouOut":
            c["parameters"]["p"] = 100000.0
    doc["signals"][0] = {"name": "yDoo", "class": "Modelica.Blocks.Sources.Constant",
                         "parameters": {"k": 1.0}, "drives": "dooDis.y"}
    model, state, drivers, names = read_modelica(_write(tmp_path, doc), return_names=True)
    assert names.p_ref == 100000.0
    assert torch.equal(drivers["air.phi_boundary"], torch.zeros(1, dtype=F64))
    hist = simulate(model, state, drivers, names.times[:3])  # converges to AIR_ATOL
    assert float(hist["p"][0, names.nodes["bouOut"]]) == 100000.0
    # The layer's own linear initial guess converges at the same tolerances too.
    layer = model.potential["air"]
    d = step_drivers(drivers, names.times, 0.0)
    for c in model.closures:
        d.update(c(state, d))
    layer.solve(d["air.phi_boundary"], d, d.get("air.sources"), atol=1e-13, rtol=1e-12)


def test_initial_airflow_solve_starts_from_the_linear_guess():
    """`three_rooms_discretized_door.json` is the topology of
    `Validation/ThreeRoomsContamDiscretizedDoor.mo` (three pinned rooms, a stack, an open
    ten-compartment door). Newton seeded with every zone at `p_start` cycles (residual 0.084
    kg/s after 50 iterations); from the layer's linear initial guess it converges in three.
    The quasi-steady initial flows are therefore solved from that guess."""
    model, state, drivers, names = _load("three_rooms_discretized_door.json")
    hist = simulate(model, state, drivers, names.times)
    q = hist["air.q"]

    def flow(key):
        return sum(s * q[:, c] for c, s in names.edges[key])

    # volWes (the door's side A) exchanges air only through the door and oriWesTop (volTop
    # -> volWes): its mass balance closes.
    assert float((flow("oriWesTop") - flow("dooOpeClo")).abs().max()) < 1e-12
    assert float(flow("oriWesTop").abs().min()) > 1e-3


def test_modelica_names_requires_its_gauge_reference():
    """Review fix round 1: `p_ref` has no default, so no `ModelicaNames` can silently claim a
    `p_default` gauge."""
    from noodl.apps.building_physics.modelica import ModelicaNames

    with pytest.raises(TypeError, match="p_ref"):
        ModelicaNames(edges={}, nodes={}, kinds={}, times=torch.zeros(1, dtype=F64),
                      air_references=())


# One-way elements with a parameter MBL declares without a default: the export must carry
# it, and its absence is refused naming the instance and the parameter.
_ONE_WAY_REQUIRED = [
    ("Orifice", {"CD": 0.6}, "A"),
    ("EffectiveAirLeakageArea", {}, "L"),
    ("Point_m_flow", {"mMea_flow_nominal": 0.01}, "dpMea_nominal"),
    ("Points_m_flow", {"dpMea_nominal": [1.0, 10.0]}, "mMea_flow_nominal"),
    ("Coefficient_V_flow", {"C": 0.01}, "m"),
    ("Coefficient_m_flow", {}, "k"),
    ("Table_V_flow", {"dpMea_nominal": [0.0, 10.0]}, "VMea_flow_nominal"),
    ("Table_m_flow", {"mMea_flow_nominal": [0.0, 0.1]}, "dpMea_nominal"),
]


@pytest.mark.parametrize(("cls", "params", "missing"), _ONE_WAY_REQUIRED,
                         ids=[c for c, _, _ in _ONE_WAY_REQUIRED])
def test_a_one_way_element_without_a_required_parameter_is_refused(tmp_path, cls, params,
                                                                   missing):
    doc = _doc("two_zones_orifice.json")
    doc["components"][2]["class"] = f"Buildings.Airflow.Multizone.{cls}"
    doc["components"][2]["parameters"] = dict(params)
    with pytest.raises(ModelicaImportError,
                       match=rf"ori \(Buildings\.Airflow\.Multizone\.{cls}\): parameter "
                             rf"'{missing}' is required"):
        read_modelica(_write(tmp_path, doc))


def test_a_non_positive_output_interval_is_refused(tmp_path):
    doc = _doc("two_zones_orifice.json")
    doc["experiment"]["Interval"] = 0.0
    with pytest.raises(ModelicaImportError, match="Interval > 0"):
        read_modelica(_write(tmp_path, doc))


def test_two_signals_driving_one_input_are_refused(tmp_path):
    doc = _doc("ramp_boundary.json")
    doc["signals"].append(dict(copy.deepcopy(doc["signals"][0]), name="ramp2"))
    with pytest.raises(ModelicaImportError, match="ramp2 .*already drives"):
        read_modelica(_write(tmp_path, doc))


@pytest.mark.parametrize(
    ("fixture", "name", "change", "match"),
    [
        ("zone_two_orifices.json", "vol", {"V": None}, r"vol .*parameter 'V' is required"),
        ("zone_two_orifices.json", "vol", {"V": 0.0}, r"vol .*volume must be positive"),
        ("delay_zone.json", "del", {"m_flow_nominal": None},
         r"del .*parameter 'm_flow_nominal' is required"),
    ],
    ids=["no-V", "zero-V", "delay-no-m_flow_nominal"],
)
def test_a_zone_without_a_positive_volume_is_refused(tmp_path, fixture, name, change, match):
    doc = _doc(fixture)
    (comp,) = [c for c in doc["components"] if c["name"] == name]
    for key, value in change.items():
        if value is None:
            comp["parameters"].pop(key)
        else:
            comp["parameters"][key] = value
    with pytest.raises(ModelicaImportError, match=match):
        read_modelica(_write(tmp_path, doc))
