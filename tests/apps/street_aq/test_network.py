"""StreetNetwork, build_model and the two named networks."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.network import (
    Street,
    StreetNetwork,
    build_model,
    from_test_network,
    initial_state,
    munich_idealised,
    street_geometry,
    street_index,
)

DT = torch.float64


def _line_network():
    """Two streets end to end along +x, with one real junction and two dead ends."""
    return StreetNetwork(
        streets=[Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 Street("s2", "b", "c", 100.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0},
        y={"a": 0.0, "b": 0.0, "c": 0.0},
    )


def _wind(model, u_ref, theta_w, h_abl=1000.0, emission=None):
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    for name, value in (emission or {}).items():
        sources[net.node_index(name)] = value
    return {
        "street.x_boundary": torch.zeros(1, dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(u_ref, dtype=DT),
        "theta_w": torch.tensor(theta_w, dtype=DT),
        "h_abl": torch.tensor(h_abl, dtype=DT),
    }


def test_from_test_network_reproduces_impaq_s_four_node_geometry():
    net = from_test_network()
    assert [s.name for s in net.streets] == ["r1", "r2", "r3"]
    lengths = torch.tensor([s.length for s in net.streets], dtype=DT)
    torch.testing.assert_close(
        lengths, torch.tensor([316.227766, 316.227766, 300.0], dtype=DT),
        rtol=1e-6, atol=0,
    )
    assert [s.width for s in net.streets] == [20.0, 10.0, 20.0]
    assert [s.height for s in net.streets] == [20.0, 20.0, 30.0]
    assert [s.z0_b for s in net.streets] == [0.15, 0.15, 0.15]
    torch.testing.assert_close(
        torch.tensor(net.azimuth, dtype=DT),
        torch.tensor([math.atan2(-100.0, 300.0), math.atan2(100.0, 300.0), 0.5 * math.pi],
                     dtype=DT),
        rtol=1e-13, atol=0,
    )


def test_munich_idealised_topology():
    net, names = munich_idealised(L=100.0, W=20.0, H=20.0)
    assert names == [str(i) for i in range(1, 13)]
    assert len(net.streets) == 12
    # Four real junctions, each a full four-way (two stubs or crossings per side), and
    # eight dead ends of degree one.
    degrees = {j: net.degree(j) for j in net.junctions}
    assert sorted(v for v in degrees.values() if v > 1) == [4, 4, 4, 4]
    assert sum(1 for v in degrees.values() if v == 1) == 8
    assert len(net.junctions) == 12
    assert {s.length for s in net.streets} == {100.0}
    # Street 11 runs from the south dead end into C, so it points due north.
    eleven = next(s for s in net.streets if s.name == "11")
    assert (eleven.u, eleven.v) == ("E11", "C")
    azimuth = dict(zip([s.name for s in net.streets], net.azimuth, strict=True))
    assert abs(azimuth["11"] - 0.5 * math.pi) < 1e-12
    assert abs(azimuth["9"] - 0.0) < 1e-12


def test_build_model_wires_the_layer_and_the_three_edge_kinds():
    sn = _line_network()
    model, state, drivers = build_model(sn, pblh_floor=False)
    net = model.net
    assert net.n == 3                      # two streets and the atmosphere
    assert model.flow_layer_of["street"] is None
    layer = model.transport["street"]
    assert layer.flow_kinds == ("route", "vent", "exchange")
    assert layer.quantity == "concentration" and layer.unit == "kg/m3"
    assert [net.nodes[i] for i in layer.interior_idx.tolist()] == ["s1", "s2"]
    # One ordered pair at the middle junction; two vent edges per (street, end); two
    # exchange edges per street.
    assert len(net.edge_index("route")) == 2
    assert len(net.edge_index("vent")) == 8
    assert len(net.edge_index("exchange")) == 4
    torch.testing.assert_close(
        layer.capacity, torch.tensor([100.0 * 20.0 * 20.0] * 2, dtype=DT),
        rtol=0, atol=0,
    )
    assert state["street.x"].shape == (2,) and state["street.x"].dtype is torch.float64
    assert street_index(model) == {"s1": 0, "s2": 1}
    assert set(drivers) == {"street.x_boundary", "street.sources"}


def test_a_dead_end_is_a_one_way_exchange_with_the_background_not_a_wall():
    """A dead end, pinned on the two-street hand case."""
    sn = _line_network()
    model, state, _ = build_model(sn, pblh_floor=False)
    downwind = model.steady(state, _wind(model, 3.0, 0.0, emission={"s1": 1.0}))
    upwind = model.steady(state, _wind(model, 3.0, math.pi, emission={"s1": 1.0}))
    c_down = downwind["street.x"]
    c_up = upwind["street.x"]
    # Wind towards +x: s1 emits and feeds s2 through the junction they share.
    assert float(c_down[0]) > 0.0 and float(c_down[1]) > 0.0
    # Wind towards -x: s2 is upwind of s1, so it receives nothing -- but s1 still loses
    # its whole flux through its own dead end, so its concentration is unchanged. A wall
    # at the dead end would have trapped the emission and raised it instead.
    assert float(c_up[1]) == 0.0
    torch.testing.assert_close(c_up[0], c_down[0], rtol=1e-12, atol=0)


def test_build_model_batches_over_forcing_steps_and_keeps_float64():
    sn = from_test_network()
    model, state, _ = build_model(sn, kappa=0.4, pblh_floor=False)
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    for name in ("r1", "r2", "r3"):
        sources[net.node_index(name)] = 1.0
    drivers = {
        "street.x_boundary": torch.zeros(3, 1, dtype=DT),
        "street.sources": sources.unsqueeze(0).expand(3, net.n),
        "U_ref": torch.tensor([2.0, 4.0, 1.0], dtype=DT),
        "theta_w": torch.tensor([0.25 * math.pi, 1.4, 3.0], dtype=DT),
        "h_abl": torch.tensor([1200.0, 800.0, 400.0], dtype=DT),
    }
    out = model.steady({}, drivers)
    assert out["street.x"].shape == (3, 3)
    assert out["street.x"].dtype is torch.float64
    assert bool((out["street.x"] > 0).all())
    # Instance 1 has twice the wind of instance 0, so every concentration is lower.
    assert bool((out["street.x"][1] < out["street.x"][0]).all())


def test_build_model_accepts_every_documented_option_combination():
    sn = from_test_network()
    for canyon_wind, exchange, routing, averaging in (
        ("soulhac", "sirane", "mixing", "none"),
        ("soulhac", "schulte", "sirane", "gauss"),
        ("exponential", "schulte", "sirane", "munich"),
    ):
        model, state, _ = build_model(
            sn, canyon_wind=canyon_wind, exchange=exchange, routing=routing,
            direction_averaging=averaging, n_theta=3, sigma_theta=0.05,
            kappa=0.41, canyon_wind_min=0.1, roof_wind_form="macdonald",
        )
        out = model.steady(state, _wind(model, 5.0, 0.3, 500.0, {"r1": 1.0}))
        assert torch.isfinite(out["street.x"]).all()


def test_build_model_names_every_bad_input():
    good = _line_network()
    with pytest.raises(ValueError, match=r"StreetNetwork.*duplicate street name.*'s1'"):
        StreetNetwork(streets=[good.streets[0], good.streets[0]], x=good.x, y=good.y)
    with pytest.raises(ValueError, match=r"StreetNetwork.*'s1'.*u and v are the same"):
        StreetNetwork(streets=[Street("s1", "a", "a", 1.0, 1.0, 1.0)], x={"a": 0.0},
                      y={"a": 0.0})
    with pytest.raises(KeyError, match=r"StreetNetwork.*'z'"):
        StreetNetwork(streets=[Street("s1", "a", "z", 1.0, 1.0, 1.0)],
                      x={"a": 0.0}, y={"a": 0.0})
    with pytest.raises(ValueError, match=r"StreetNetwork.*'s1'.*width.*-1"):
        StreetNetwork(streets=[Street("s1", "a", "b", 1.0, -1.0, 1.0)],
                      x={"a": 0.0, "b": 1.0}, y={"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match=r"build_model.*'atmosphere'"):
        build_model(StreetNetwork(
            streets=[Street("atmosphere", "a", "b", 1.0, 1.0, 1.0)],
            x={"a": 0.0, "b": 1.0}, y={"a": 0.0, "b": 0.0}))


def test_street_geometry_carries_the_azimuth_and_the_per_street_roughness():
    sn = from_test_network()
    geometry = street_geometry(sn)
    assert geometry.names == ["r1", "r2", "r3"]
    assert geometry.u == ["n0", "n1", "n3"] and geometry.v == ["n1", "n2", "n1"]
    for tensor in (geometry.length, geometry.width, geometry.height, geometry.z0_b,
                   geometry.azimuth):
        assert tensor.dtype is torch.float64 and tensor.shape == (3,)


def test_initial_state_is_zero_and_shaped_by_the_species_count():
    sn = _line_network()
    _, state, _ = build_model(sn, species=("nox",))
    assert state["street.x"].shape == (2,)
    model, state3, drivers3 = build_model(sn, species=("no", "no2", "o3"))
    assert state3["street.x"].shape == (2, 3)
    assert drivers3["street.sources"].shape == (3, 3)
    assert drivers3["street.x_boundary"].shape == (1, 3)
    torch.testing.assert_close(initial_state(model)["street.x"], state3["street.x"],
                               rtol=0, atol=0)
