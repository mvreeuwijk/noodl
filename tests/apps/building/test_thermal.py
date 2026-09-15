"""The building application on top of the core: heat balance, walls, densities, gradients."""

from __future__ import annotations

import math

import pytest
import torch

from tellegen.apps.building.elements import orifice_elements_from_edges
from tellegen.apps.building.thermal import (
    CP_AIR,
    P_REF,
    R_AIR,
    RHO_0,
    IdealGasDensity,
    LinearDensity,
    WallMass,
    Zone,
    add_zone,
    build_model,
    initial_state,
    thermal_layer,
)
from tellegen.drives import Stack
from tellegen.layers.transport import TransportLayer
from tellegen.model import Model
from tellegen.topology import Network

F64 = torch.float64
T_OUT = 283.15


def _single_zone(*, wall: bool = False, source: float = 500.0, coupling="iterate"):
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    w = WallMass("z_wall", capacity=2e6, ua_zone=100.0, ua_ambient=20.0) if wall else None
    add_zone(net, Zone("z", volume=50.0, T0=293.15, wall=w))
    net.add_edge("ambient", "z", kind="airpath", z_path=0.0, Cd=0.6, area=0.01)
    net.add_edge("ambient", "z", kind="airpath", z_path=2.0, Cd=0.6, area=0.01)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(
        net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
        coupling=coupling, iterate_tol={"thermal": 1e-9}, iterate_max=200,
    )
    sources = torch.zeros(net.n, dtype=F64)
    sources[net.node_index("z")] = source
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "thermal.x_boundary": torch.tensor([T_OUT], dtype=F64),
        "thermal.sources": sources,
    }
    return net, model, initial_state(model), drivers, el


def test_build_model_names_layers_and_tags_them():
    net, model, state, _, _ = _single_zone()
    assert set(model.layers) == {"air", "thermal"}
    assert (model.layers["air"].quantity, model.layers["air"].unit) == ("pressure", "Pa")
    assert (model.layers["thermal"].quantity, model.layers["thermal"].unit) == ("temperature", "K")
    assert model.flow_layer_of == {"thermal": "air"}
    assert state["thermal.x"].tolist() == [293.15]
    th = model.layers["thermal"]
    assert th.capacity.item() == pytest.approx(RHO_0 * CP_AIR * 50.0)
    assert th.carrier.item() == pytest.approx(CP_AIR)


def test_energy_balance_closes_at_steady_state():
    net, model, state, drivers, _ = _single_zone()
    # Controller ruling R21: the assertions below are at the solve's own accuracy, so the
    # tolerance is tightened rather than the assertion loosened. At Newton's dtype-derived
    # default (1.5e-8 on a ~6e-3 kg/s mass balance) the iterate coupling stalls with a
    # per-pass thermal change of ~1e-4 K, well above its own 1e-9 K tolerance.
    ss = model.steady(state, drivers, atol=1e-14, rtol=1e-14)
    T_z = ss["thermal.x"][0].item()
    q = model.potential["air"].flows_of_kind(ss["air.q"], "airpath")
    assert q[0] > 0 and q[1] < 0                              # in low, out high
    assert CP_AIR * q[0].item() * (T_z - T_OUT) == pytest.approx(500.0, rel=1e-7)
    res = model.residuals(ss, drivers)
    assert res["thermal"].abs().max().item() < 1e-9
    assert res["air"].abs().max().item() < 1e-10


def test_ventilated_zone_with_a_wall_matches_the_algebraic_balance():
    net, model, state, drivers, _ = _single_zone(wall=True)
    ss = model.steady(state, drivers, atol=1e-14, rtol=1e-14)  # R21, as above
    th = model.layers["thermal"]
    names = [net.nodes[i] for i in th.interior_idx.tolist()]
    T = dict(zip(names, ss["thermal.x"].tolist(), strict=True))
    q = model.potential["air"].flows_of_kind(ss["air.q"], "airpath")
    F_in = q[0].item()
    # wall: 100 (T_z - T_w) = 20 (T_w - T_o); zone: E = c_p F (T_z - T_o) + 100 (T_z - T_w)
    assert 100.0 * (T["z"] - T["z_wall"]) == pytest.approx(20.0 * (T["z_wall"] - T_OUT), rel=1e-8)
    assert CP_AIR * F_in * (T["z"] - T_OUT) + 100.0 * (T["z"] - T["z_wall"]) == pytest.approx(
        500.0, rel=1e-7
    )


def test_wall_mass_relaxes_with_its_rc_time_constant():
    """Zone temperature held (fixed_temperature), no airflow: the wall node is a first-order
    RC element with tau = C / (UA_zone + UA_ambient)."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("z", volume=50.0, T0=300.0,
                       wall=WallMass("z_wall", capacity=2e6, ua_zone=100.0, ua_ambient=20.0)))
    net.add_edge("ambient", "z", kind="airpath", z_path=0.0, Cd=0.6, area=0.01)
    th = thermal_layer(net, fixed_temperature=("z",))
    assert th.n_i == 1                                           # the wall only
    q = torch.zeros(1, dtype=F64)
    xb = torch.tensor([T_OUT, 300.0], dtype=F64)                 # boundary order: ambient, z
    T_eq = (100.0 * 300.0 + 20.0 * T_OUT) / 120.0
    tau = 2e6 / 120.0
    x = torch.tensor([290.0], dtype=F64)
    sources = torch.zeros(net.n, dtype=F64)
    x1 = th.step(x, q, sources, xb, dt=tau)
    assert x1.item() == pytest.approx(T_eq + (290.0 - T_eq) * math.exp(-1.0), rel=1e-10)


def test_density_closures_write_rho_and_rho_amb_in_node_order():
    net, model, state, drivers, _ = _single_zone()
    th = model.layers["thermal"]
    amb = net.node_index("ambient")
    ideal = IdealGasDensity(th, ambient_index=amb)
    out = ideal(state, drivers)
    assert out["rho"].shape == (net.n,)
    assert out["rho"][net.node_index("z")].item() == pytest.approx(P_REF / (R_AIR * 293.15))
    assert out["rho_amb"].item() == pytest.approx(P_REF / (R_AIR * T_OUT))
    linear = LinearDensity(th, ambient_index=amb, rho_0=1.2, T_0=T_OUT)
    out2 = linear(state, drivers)
    assert out2["rho_amb"].item() == pytest.approx(1.2)
    assert out2["rho"][net.node_index("z")].item() == pytest.approx(
        1.2 * (1.0 - (293.15 - T_OUT) / T_OUT)
    )
    # batched thermal state
    batched = dict(state, **{"thermal.x": torch.tensor([[293.15], [303.15]], dtype=F64)})
    assert ideal(batched, drivers)["rho"].shape == (2, net.n)


def test_gradients_flow_through_the_application_built_model():
    """Loss on the zone temperature after one ping-pong step; gradients to the heat source
    driver, to the boundary temperature driver and to the leakage coefficient, against
    central differences."""
    net, _, _, drivers, _ = _single_zone()
    el_learn = orifice_elements_from_edges(net, "airpath", learnable=True)
    model = build_model(net, air_elements=[el_learn], drives=[Stack.from_network(net, "airpath")])
    state = initial_state(model)
    src = drivers["thermal.sources"].clone().requires_grad_(True)
    drv = dict(drivers, **{"thermal.sources": src})

    def loss(d):
        # R21: the C gradient is ~4e1, so a 1e-7 step moves the loss by ~8e-6 K -- below the
        # noise of a solve stopped at Newton's default 1.5e-8 residual. Tighten the solve.
        return model.step(state, d, 600.0, atol=1e-14, rtol=1e-14)["thermal.x"].sum()

    value = loss(drv)
    value.backward()
    iz = net.node_index("z")
    h = 1e-3

    def _sources(value: float):
        s = drivers["thermal.sources"].clone()
        return dict(drivers, **{"thermal.sources": s.index_fill_(0, torch.tensor([iz]), value)})

    up = loss(_sources(500.0 + h)).item()
    down = loss(_sources(500.0 - h)).item()
    assert src.grad[iz].item() == pytest.approx((up - down) / (2 * h), rel=1e-5)
    gC = el_learn.C.grad[0].item()
    hC = 1e-7
    with torch.no_grad():
        el_learn.C[0] += hC
        upC = loss(drivers).item()
        el_learn.C[0] -= 2 * hC
        downC = loss(drivers).item()
        el_learn.C[0] += hC
    assert gC == pytest.approx((upC - downC) / (2 * hC), rel=1e-4)
    # ... and to the BOUNDARY VALUE driver, through the density closure and the transport
    # layer's boundary block (this backward accumulates into el_learn.C.grad again, so it
    # comes after gC has been read).
    Tb = drivers["thermal.x_boundary"].clone().requires_grad_(True)
    loss(dict(drivers, **{"thermal.x_boundary": Tb})).backward()
    hT = 1e-5
    upT = loss(dict(drivers, **{"thermal.x_boundary": drivers["thermal.x_boundary"] + hT}))
    downT = loss(dict(drivers, **{"thermal.x_boundary": drivers["thermal.x_boundary"] - hT}))
    assert Tb.grad[0].item() == pytest.approx(
        (upT.item() - downT.item()) / (2 * hT), rel=1e-6
    )


def test_thermal_layer_refuses_a_node_with_no_heat_capacity():
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("z", volume=0.0))
    net.add_edge("ambient", "z", kind="airpath", z_path=0.0, Cd=0.6, area=0.01)
    with pytest.raises(ValueError, match=r"thermal.*'z'.*capacity"):
        thermal_layer(net)


def test_build_model_refuses_a_flow_kind_no_air_element_provides():
    """Narrowing `flow_kinds` to a kind the air layer does not solve would silently stop heat
    (and species) advecting on it; it is named, not dropped."""
    net, _, _, _, el = _single_zone()
    with pytest.raises(KeyError, match=r"flow_kinds.*'duct'.*airpath"):
        build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                    flow_kinds=("airpath", "duct"))


def test_initial_state_keys_transport_layers_by_their_own_name():
    """A renamed thermal layer still gets its state; an unknown quantity is refused by name
    rather than answered with a silently empty state."""
    net, model, _, _, _ = _single_zone()
    heat = thermal_layer(net, name="heat")
    st = initial_state(Model(net, {"air": model.layers["air"], "heat": heat}))
    assert list(st) == ["heat.x"] and st["heat.x"].tolist() == [293.15]
    odd = TransportLayer(net, "odd", capacity=torch.ones(1, dtype=F64), flow_kind="airpath",
                         boundary=["ambient"])
    with pytest.raises(ValueError, match=r"initial_state.*'odd'.*quantity"):
        initial_state(Model(net, {"air": model.layers["air"], "odd": odd}))


def test_species_layer_uses_zone_air_mass_as_capacity():
    net, _, _, _, el = _single_zone()
    model = build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                        species=1)
    sp = model.layers["species"]
    assert (sp.quantity, sp.unit) == ("mass_fraction", "kg/kg")
    assert sp.capacity.item() == pytest.approx(RHO_0 * 50.0)
    st = initial_state(model)
    assert st["species.x"].shape == (1,) and st["species.x"].item() == 0.0
