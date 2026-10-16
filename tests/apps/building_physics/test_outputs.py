"""Post-processing: exposure, inhaled dose, infection risk, air change rates and the local
mean age of air."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.inputs import Input
from noodl.apps.building_physics.occupants import (
    Occupancy,
    OccupancyDay,
    Occupant,
    OccupantGeneration,
    occupant_inputs,
)
from noodl.apps.building_physics.outputs import (
    building_air_change_rate,
    cumulative_exposure,
    infection_probability,
    inhaled_dose,
    local_mean_age,
    occupant_exposure,
    wells_riley,
    zone_air_change_rate,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj, steady
from noodl.apps.building_physics.schedules import DAY, Calendar, TimeSeries
from noodl.apps.building_physics.simulation import simulate, simulate_project
from noodl.apps.building_physics.thermal import R_AIR, species_layer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
AHS = DATA / "ahs_three_zone.prj"
DUCTS = DATA / "duct_three_zone.prj"
F64 = torch.float64
CAL = Calendar()


def _three():
    model, _, _ = project_to_model(read_prj(THREE))
    interior = model.transport["species"].interior_idx.tolist()
    pos = {z: interior.index(model.net.node_index(z)) for z in ("one", "two", "three")}
    return model, pos


def _history(model, pos, times, columns):
    """`columns`: {zone: values over time} -> a history with species.x (N, n_i, 1)."""
    x = torch.zeros(len(times), model.transport["species"].n_i, 1, dtype=F64)
    for zone, values in columns.items():
        x[:, pos[zone], 0] = torch.tensor(values, dtype=F64)
    return {"time": torch.tensor(times, dtype=F64), "species.x": x}


def _walker():
    """one until 450 s, two until 750 s, out after."""
    day = OccupancyDay([0.0, 450.0, 750.0, DAY], ["one", "two", None, None])
    return Occupant(Occupancy([day]), name="walker")


def test_exposure_splits_a_step_at_a_mid_step_move():
    """ContamX's exposure file, as its rule was measured: the step mean of the linearly
    interpolated mass fraction of the occupied zone, split at the move. On
    `occupants_three_zone.prj` noodl physics matches ContamX's exposure file to 4.0e-7, its
    print precision (`test_occupant_exposure_age_of_air_and_air_change_rate_match_contamx`)."""
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0, 600.0, 900.0],
                 {"one": [1e-4, 3e-4, 5e-4, 7e-4], "two": [2e-4, 4e-4, 6e-4, 8e-4]})
    e = occupant_exposure(h, _walker(), model, calendar=CAL)
    assert e.shape == (3, 1)
    # (0, 300]: one, mean 2e-4. (300, 450] one: 3.5e-4; (450, 600] two: 5.5e-4 -> 4.5e-4.
    # (600, 750] two: 6.5e-4; (750, 900] out -> 3.25e-4.
    torch.testing.assert_close(e[:, 0], torch.tensor([2e-4, 4.5e-4, 3.25e-4], dtype=F64),
                               rtol=1e-14, atol=0)


def test_the_calendar_places_the_moves_and_is_required():
    """Starting the run at 00:05 moves the walker 300 s earlier in simulation time."""
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0], {"one": [1.0, 1.0], "two": [3.0, 3.0]})
    late = Calendar(start_time=300.0)
    e = occupant_exposure(h, _walker(), model, calendar=late)
    torch.testing.assert_close(e[:, 0], torch.tensor([0.5 * 1.0 + 0.5 * 3.0], dtype=F64))
    with pytest.raises(TypeError, match="calendar"):
        occupant_exposure(h, _walker(), model)


def test_an_absent_occupant_breathes_nothing():
    model, pos = _three()
    away = Occupant(Occupancy([OccupancyDay([0.0, DAY], [None, None])]), name="away")
    h = _history(model, pos, [0.0, 300.0], {"one": [1.0, 1.0]})
    assert occupant_exposure(h, away, model, calendar=CAL).abs().sum().item() == 0.0


def test_cumulative_exposure_uses_the_actual_step_lengths():
    model, pos = _three()
    split = _history(model, pos, [0.0, 300.0, 450.0, 600.0],
                     {"one": [1e-4, 3e-4, 4e-4, 5e-4], "two": [2e-4, 4e-4, 5e-4, 6e-4]})
    e = occupant_exposure(split, _walker(), model, calendar=CAL)
    torch.testing.assert_close(e[:, 0], torch.tensor([2e-4, 3.5e-4, 5.5e-4], dtype=F64),
                               rtol=1e-14, atol=0)
    c = cumulative_exposure(e, split["time"])
    assert c.shape == (4, 1) and c[0].item() == 0.0
    assert c[-1].item() == pytest.approx(2e-4 * 300 + 3.5e-4 * 150 + 5.5e-4 * 150, rel=1e-14)


def test_exposure_of_a_batched_history():
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0, 600.0], {"one": [1e-4, 3e-4, 5e-4]})
    both = {"time": h["time"], "species.x": torch.stack([h["species.x"],
                                                         2 * h["species.x"]], dim=1)}
    e = occupant_exposure(both, _walker(), model, calendar=CAL)
    assert e.shape == (2, 2, 1)
    torch.testing.assert_close(e[:, 1], 2 * occupant_exposure(h, _walker(), model,
                                                               calendar=CAL))


def test_exposure_is_differentiable_in_the_history():
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0, 600.0], {"one": [1e-4, 3e-4, 5e-4]})
    x = h["species.x"].clone().requires_grad_(True)
    e = occupant_exposure({"time": h["time"], "species.x": x}, _walker(), model, calendar=CAL)
    cumulative_exposure(e, h["time"])[-1].sum().backward()
    # zone one: (0, 300] trapezoid 150 + 150; (300, 450] in one, weights 112.5 and 37.5
    torch.testing.assert_close(x.grad[:, pos["one"], 0],
                               torch.tensor([150.0, 262.5, 37.5], dtype=F64))


def test_a_history_of_the_wrong_row_count_is_refused():
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0], {"one": [1.0, 1.0]})
    with pytest.raises(ValueError, match=r"'species.x'.*\(N, \.\.\., 3, 1\)"):
        occupant_exposure({"time": h["time"], "species.x": h["species.x"][:, :2, 0]},
                          _walker(), model, calendar=CAL)


def _plenum_model():
    """Zones one and two, a storage-free plenum between them, and a loft no path reaches."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("one", volume=300.0)
    net.add_node("two", volume=600.0)
    net.add_node("plenum", volume=0.0)
    net.add_node("loft", volume=50.0)
    net.add_edge("ambient", "one", kind="airpath")
    net.add_edge("one", "plenum", kind="airpath")
    net.add_edge("plenum", "two", kind="airpath")
    net.add_edge("two", "ambient", kind="airpath")
    layer = TransportLayer(net, "species", capacity=torch.tensor([360.0, 720.0, 1.0], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"], storage_free=["plenum"],
                           scheme="implicit", species_names=("sarin",))
    return Model(net, {"species": layer})


@pytest.mark.parametrize(("zone", "error", "match"), [
    ("ambient", ValueError, r"'smoker'.*boundary node 'ambient'.*no zone air to breathe"),
    ("loft", ValueError, r"'smoker'.*node 'loft'.*no flow edge"),
    ("plenum", ValueError, r"'smoker'.*storage-free node 'plenum'.*no air to breathe"),
    ("kitchen", KeyError, r"'smoker'.*zone 'kitchen' is not a node"),
])
def test_an_occupant_where_the_layer_holds_no_air_is_refused_by_name(zone, error, match):
    """The same refusal as for generation: the layer holds no mass fraction to breathe
    there. Every zone of any of the occupant's day schedules is checked, also one of a day
    type the run does not reach."""
    model = _plenum_model()
    lay = model.transport["species"]
    h = {"time": torch.tensor([0.0, 300.0], dtype=F64),
         "species.x": torch.zeros(2, lay.n_i, 1, dtype=F64)}
    days = [OccupancyDay([0.0, DAY], ["one", "one"]), OccupancyDay([0.0, DAY], [zone, zone])]
    smoker = Occupant(Occupancy(days, day_types=[1] * 11 + [2]), name="smoker")
    with pytest.raises(error, match=match):
        occupant_exposure(h, smoker, model, calendar=CAL)


def test_dose_and_infection_probability():
    E = torch.tensor([0.0, 10.0], dtype=F64)
    torch.testing.assert_close(inhaled_dose(E, 1e-4, density=1.2),
                               torch.tensor([0.0, 1.2e-3], dtype=F64))
    assert infection_probability(torch.tensor(1e-20, dtype=F64)).item() == 1e-20
    p = wells_riley(2, 10 / 3600, 0.5 / 3600, 0.1, 8 * 3600.0)
    assert p.item() == pytest.approx(1 - math.exp(-2 * 10 * 0.5 * 8 / 360), rel=1e-14)


def test_wells_riley_in_quanta_concentration_form_matches_the_classic_steady_form():
    """One well-mixed room with an infector at steady state: the susceptible occupant's
    inhaled quanta p rho int x dt give P = 1 - exp(-I q p t / Q) exactly."""
    rho, V, Q = 1.2041, 100.0, 0.1                     # kg/m3, m3, m3/s
    q = rho * Q                                        # kg/s
    quanta, p, hours = 10 / 3600, 0.5 / 3600, 8
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("room", volume=V)
    net.add_edge("ambient", "room", kind="airpath")
    net.add_edge("room", "ambient", kind="airpath")
    layer = species_layer(net, n_species=1, rho=rho, species_names=["quanta"])
    always = Occupancy([OccupancyDay([0.0, DAY], ["room", "room"])])
    infector = Occupant(always, generation={"quanta": quanta}, name="infector")
    susceptible = Occupant(always, name="susceptible")

    def through(state, drivers):
        return {"species.q": torch.tensor([q, q], dtype=F64)}

    gen = OccupantGeneration([infector], layer=layer)
    model = Model(net, {"species": layer}, closures=[through, gen])
    state = {"species.x": torch.full((1, 1), quanta / q, dtype=F64)}
    drivers = {"species.x_boundary": torch.zeros(1, 1, dtype=F64)}
    times = torch.arange(hours * 6 + 1, dtype=F64) * 600.0
    out = simulate(model, state, drivers, times, inputs=occupant_inputs([infector], net),
                   calendar=CAL)
    E = cumulative_exposure(occupant_exposure(out, susceptible, model, calendar=CAL),
                            out["time"])[-1, 0]
    P = infection_probability(inhaled_dose(E, p, density=rho))
    classic = wells_riley(1, quanta, p, Q, hours * 3600.0)
    assert P.item() == pytest.approx(classic.item(), rel=1e-12)


def test_cumulative_exposure_of_a_filling_zone_against_the_analytic_solution():
    """One zone, clean at t = 0, a constant generation G and a through-flow q: x(t) = x_s (1 -
    exp(-t / tau)), x_s = G / q, tau = M / q, so int_0^T x dt = x_s (T - tau (1 - exp(-T /
    tau))). The run steps by implicit Euler, x_k = x_s (1 - r^k) with r = 1 / (1 + h / tau),
    and the exposure is the trapezoid of those rows: summing the geometric series,
    h sum (x_(k-1) + x_k) / 2 = x_s (T - tau - h / 2) + O(r^N). The first-order stepping
    error is therefore -x_s h / 2 (the implicit-Euler lag; the trapezoid adds nothing more at
    this order), and the terms dropped are below x_s tau r^N < 1e-11 x_s tau here: the
    residual must be -x_s h / 2 to 1e-9 of the integral."""
    rho, V, h, hours = 1.2041, 100.0, 300.0, 8
    q = 0.12041                                        # kg/s: tau = rho V / q = 1000 s
    G = 1e-6                                           # kg/s
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("room", volume=V)
    net.add_edge("ambient", "room", kind="airpath")
    net.add_edge("room", "ambient", kind="airpath")
    layer = species_layer(net, n_species=1, rho=rho, species_names=["tracer"])
    always = Occupancy([OccupancyDay([0.0, DAY], ["room", "room"])])
    emitter = Occupant(always, generation={"tracer": G}, name="emitter")
    resident = Occupant(always, name="resident")

    def through(state, drivers):
        return {"species.q": torch.tensor([q, q], dtype=F64)}

    model = Model(net, {"species": layer},
                  closures=[through, OccupantGeneration([emitter], layer=layer)])
    state = {"species.x": torch.zeros(1, 1, dtype=F64)}
    drivers = {"species.x_boundary": torch.zeros(1, 1, dtype=F64)}
    times = torch.arange(hours * 12 + 1, dtype=F64) * h
    out = simulate(model, state, drivers, times, inputs=occupant_inputs([emitter], net),
                   calendar=CAL)
    E = cumulative_exposure(occupant_exposure(out, resident, model, calendar=CAL),
                            out["time"])[-1, 0].item()
    x_s, tau, T = G / q, rho * V / q, hours * 3600.0
    exact = x_s * (T - tau * (1 - math.exp(-T / tau)))
    assert abs(E - exact) <= x_s * h / 2 * (1 + 1e-9)
    assert E - exact == pytest.approx(-x_s * h / 2, abs=1e-9 * exact)


# ---------------------------------------------------------------- air change rates and age


def _flow_model(volumes, edges, rho=1.2):
    """Zones of `volumes` and `edges` [(src, tgt)] of one kind; the flows are given."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    for name, v in volumes.items():
        net.add_node(name, volume=v)
    for s, t in edges:
        net.add_edge(s, t, kind="airpath")
    return Model(net, {"species": species_layer(net, n_species=1, rho=rho)})


def _q(*v):
    """A history of the species layer's flows (a transport-only model's "species.q")."""
    return {"species.q": torch.tensor(v, dtype=F64)}


def _f(*v):
    return torch.tensor(v, dtype=F64)


def _edges(model, layer="species"):
    """The layer's flow edges (source, target node indices), in its flow order."""
    lay = model.transport[layer]
    ends = [model.net.endpoints(k) for k in lay.flow_kinds]
    return (torch.cat([s for s, _ in ends]).tolist(), torch.cat([t for _, t in ends]).tolist())


def _layer_q(model, q, layer="species"):
    lay = model.transport[layer]
    return model.kirchhoff[model.flow_layer_of[layer]].flows_of_kind(q, lay.flow_kinds)


def _dense_age(model, f, M, layer="species"):
    """An independent age: sum_in q tau_up - sum_out q tau + M = 0 at every interior row,
    tau = 0 outside, assembled edge by edge from the flows `f` and masses `M` and solved
    densely (a storage-free row has M = 0: its age is the flow-weighted mix of its inflows)."""
    lay = model.transport[layer]
    pos = {node: k for k, node in enumerate(lay.interior_idx.tolist())}
    M = torch.where(lay.storage_free_mask, torch.zeros_like(M), M)
    A = torch.zeros(lay.n_i, lay.n_i, dtype=F64)
    for s, t, q in zip(*_edges(model, layer), f.tolist(), strict=True):
        up, down = (s, t) if q > 0 else (t, s)
        if up in pos:
            A[pos[up], pos[up]] -= abs(q)
            if down in pos:
                A[pos[down], pos[up]] += abs(q)
    return torch.linalg.solve(A, -M)


def _inflow_from(model, f, node):
    """The flow from `node` into the rest, summed over the layer's edges (inflow direction)."""
    i = model.net.node_index(node)
    src, tgt = _edges(model)
    return sum(torch.relu(f[..., j]) if src[j] == i else torch.relu(-f[..., j])
               for j in range(len(src)) if i in (src[j], tgt[j]))


def test_one_zone_age_is_its_air_mass_over_the_flow():
    model = _flow_model({"room": 50.0}, [("ambient", "room"), ("room", "ambient")])
    h = _q(0.12, 0.12)
    assert local_mean_age(h, model).item() == pytest.approx(1.2 * 50 / 0.12, rel=1e-12)
    rate, zones = zone_air_change_rate(h, model)
    assert zones == ["room"]
    assert rate.item() == pytest.approx(0.12 / 60 * 3600, rel=1e-14)
    assert building_air_change_rate(h, model).item() == pytest.approx(
        0.12 / (1.2041 * 50) * 3600, rel=1e-14)
    assert building_air_change_rate(h, model, density=1.2).item() == pytest.approx(7.2)


def test_ages_add_up_along_zones_in_series():
    model = _flow_model({"a": 100.0, "b": 200.0, "c": 300.0},
                        [("ambient", "a"), ("a", "b"), ("b", "c"), ("c", "ambient")])
    torch.testing.assert_close(local_mean_age(_q(0.1, 0.1, 0.1, 0.1), model),
                               _f(1200.0, 3600.0, 7200.0), rtol=1e-12, atol=0)


def test_age_with_recirculation_and_reversed_edges():
    """Closed form: tau_a = (M_a + r M_b / (q + r)) / q, tau_b = tau_a + M_b / (q + r)."""
    q, r = 0.1, 0.05
    model = _flow_model({"a": 100.0, "b": 200.0},
                        [("ambient", "a"), ("a", "b"), ("b", "a"), ("b", "ambient")])
    expected = _f(2000.0, 3600.0)
    torch.testing.assert_close(local_mean_age(_q(q, q + r, r, q), model), expected,
                               rtol=1e-12, atol=0)
    flipped = _flow_model({"a": 100.0, "b": 200.0},
                          [("a", "ambient"), ("a", "b"), ("b", "a"), ("ambient", "b")])
    torch.testing.assert_close(local_mean_age(_q(-q, q + r, r, -q), flipped), expected,
                               rtol=1e-12, atol=0)


def test_zone_air_change_rate_counts_air_from_other_zones_too():
    """Each zone's inflow from any node over its air mass: all of zone b's comes from a."""
    q, r = 0.1, 0.05
    model = _flow_model({"a": 100.0, "b": 200.0},
                        [("ambient", "a"), ("a", "b"), ("b", "a"), ("b", "ambient")])
    rate, zones = zone_air_change_rate(_q(q, q + r, r, q), model)
    assert zones == ["a", "b"]
    torch.testing.assert_close(rate, _f((q + r) / 120 * 3600, (q + r) / 240 * 3600),
                               rtol=1e-14, atol=0)


def test_building_air_change_rate_counts_what_crosses_into_the_building():
    """ContamX: every flow into the conditioned space from outside it -- the ambient or a
    zone left out of it -- over 1.2041 kg/m3 times its volume."""
    model = _flow_model({"a": 100.0, "b": 200.0, "c": 300.0},
                        [("ambient", "a"), ("a", "b"), ("b", "c"), ("c", "ambient")])
    h = _q(0.1, 0.1, 0.1, 0.1)
    everything = building_air_change_rate(h, model)
    assert everything.item() == pytest.approx(0.1 / (1.2041 * 600) * 3600, rel=1e-14)
    without_b = building_air_change_rate(h, model, conditioned=["a", "c"])
    assert without_b.item() == pytest.approx(0.2 / (1.2041 * 400) * 3600, rel=1e-14)
    with pytest.raises(KeyError, match="'ambient'"):
        building_air_change_rate(h, model, conditioned=["a", "ambient"])
    with pytest.raises(KeyError, match="'kitchen'"):
        building_air_change_rate(h, model, conditioned=["a", "kitchen"])
    with pytest.raises(ValueError, match="no nodes"):
        building_air_change_rate(h, model, conditioned=[])


def test_outputs_of_batched_flows_and_their_gradient():
    model = _flow_model({"room": 50.0}, [("ambient", "room"), ("room", "ambient")])
    q = torch.tensor([[0.12, 0.12], [0.06, 0.06]], dtype=F64, requires_grad=True)
    age = local_mean_age({"species.q": q}, model)
    assert age.shape == (2, 1)
    torch.testing.assert_close(age[:, 0], _f(500.0, 1000.0), rtol=1e-12, atol=0)
    assert zone_air_change_rate({"species.q": q}, model)[0].shape == (2, 1)
    assert building_air_change_rate({"species.q": q}, model).shape == (2,)
    (g,) = torch.autograd.grad(age[0, 0], q)
    # The balance is sum_in q tau_up - sum_out q tau + M = 0 with tau_up = 0 outside, so
    # tau = M / q_out: d tau / d q_out = -M / q^2 = -60 / 0.0144, and the inflow edge: 0.
    assert g[0, 1].item() == pytest.approx(-60 / 0.0144, rel=1e-9)
    assert g[0, 0].item() == 0.0 and g[1].abs().sum().item() == 0.0


def test_the_three_zone_projects_outputs_from_its_steady_flows():
    project = read_prj(THREE)
    model, state, drivers = project_to_model(project)
    st = steady(model, state, drivers, atol=1e-14)
    # The project's zone air masses follow its temperatures and pressure: a history without
    # them is refused; a steady state carries the ones it was solved with.
    with pytest.raises(ValueError, match=r"'species.capacity'.*PrescribedTemperatureDensity"):
        local_mean_age({"air.q": st["air.q"]}, model)
    with pytest.raises(ValueError, match=r"'species.capacity'"):
        zone_air_change_rate({"air.q": st["air.q"]}, model)
    cap = st["species.capacity"]
    torch.testing.assert_close(cap, model.initial_capacities(st, drivers)["species.capacity"],
                               rtol=0, atol=0)
    f = project.path_flows(st["air.q"])        # ambient -> one -> two -> three -> ambient
    lay = model.transport["species"]
    interior = lay.interior_idx.tolist()
    M = {z: cap[interior.index(model.net.node_index(z))].item()
         for z in ("one", "two", "three")}
    age = local_mean_age(st, model, capacity=cap)
    tau1 = M["one"] / f[0].item()
    tau2 = tau1 + M["two"] / f[1].item()
    tau3 = tau2 + M["three"] / -f[2].item()
    for z, tau in (("one", tau1), ("two", tau2), ("three", tau3)):
        assert age[interior.index(model.net.node_index(z))].item() == pytest.approx(tau,
                                                                                     rel=1e-10)
    rate, zones = zone_air_change_rate(st, model, capacity=cap)
    assert rate[zones.index("one")].item() == pytest.approx(f[0].item() / M["one"] * 3600,
                                                            rel=1e-12)
    assert building_air_change_rate(st, model).item() == pytest.approx(
        f[0].item() / (1.2041 * 1800) * 3600, rel=1e-12)


def test_age_and_zone_rate_use_each_rows_air_mass_when_the_temperature_varies():
    """Zone one warms from 293.15 K to 313.15 K over the hour: its air mass P V / (R T) drops
    by 6 %. `simulate` records the layer's capacity, and the age and the zone rate of every
    row use that row's; the construction-time mass would be off by far more than round-off."""
    project = read_prj(THREE)
    model, state, drivers = project_to_model(project)
    warming = TimeSeries([0.0, 3600.0], [[293.15], [313.15]], name="zone one")
    times = torch.arange(7, dtype=F64) * 600.0
    out = simulate(model, state, drivers, times, inputs=[Input(warming, "T", at=["one"])])
    lay = model.transport["species"]
    one = lay.interior_idx.tolist().index(model.net.node_index("one"))
    T_one = 293.15 + 20.0 * times / 3600.0
    mass_one = project.ambient_conditions["Pb"] * 300.0 / (R_AIR * T_one)
    torch.testing.assert_close(out["species.capacity"][:, one], mass_one, rtol=1e-14, atol=0)
    f = _layer_q(model, out["air.q"])
    age = local_mean_age(out, model)
    rate, zones = zone_air_change_rate(out, model)
    for k in range(len(times)):
        torch.testing.assert_close(age[k], _dense_age(model, f[k], out["species.capacity"][k]),
                                   rtol=1e-10, atol=0)
        fixed = _dense_age(model, f[k], lay.capacity)
        if k > 0:
            assert (age[k, one] - fixed[one]).abs().item() > 1e-3 * fixed[one].item()
    assert rate[-1, zones.index("one")].item() == pytest.approx(
        _inflow_into(model, f[-1], "one").item() / mass_one[-1].item() * 3600, rel=1e-12)
    torch.testing.assert_close(local_mean_age(out, model, capacity=lay.capacity)[-1],
                               _dense_age(model, f[-1], lay.capacity), rtol=1e-10, atol=0)


def _inflow_into(model, f, node):
    """The flow into `node` from any node, summed over the layer's edges."""
    i = model.net.node_index(node)
    src, tgt = _edges(model)
    return sum(torch.relu(f[..., j]) if tgt[j] == i else torch.relu(-f[..., j])
               for j in range(len(src)) if i in (src[j], tgt[j]))


def test_age_on_an_air_handler_with_a_storage_free_plenum():
    """`ahs_three_zone.prj`: the return plenum holds no air. Its row is algebraic, so its age
    is that of what flows in -- zone three's, its only inflow -- whatever capacity is given
    there (the layer's construction capacity holds a 1 kg stand-in at that row)."""
    project = read_prj(AHS)
    out = simulate_project(project)
    model, _, _ = project_to_model(project)
    lay = model.transport["species"]
    interior = lay.interior_idx.tolist()
    row = {z: interior.index(model.net.node_index(z)) for z in project.zones}
    assert lay.storage_free_mask[row["ahs(Ret)"]].item()
    assert out["species.capacity"][0, row["ahs(Ret)"]].item() == 0.0
    f = _layer_q(model, out["air.q"])
    age = local_mean_age(out, model)
    for k in range(out["time"].numel()):
        torch.testing.assert_close(age[k], _dense_age(model, f[k], out["species.capacity"][k]),
                                   rtol=1e-10, atol=0)
    torch.testing.assert_close(age[:, row["ahs(Ret)"]], age[:, row["three"]], rtol=1e-12,
                               atol=0)
    stand_in = local_mean_age(out, model, capacity=lay.capacity)
    torch.testing.assert_close(stand_in[:, row["ahs(Ret)"]], stand_in[:, row["three"]],
                               rtol=1e-12, atol=0)
    rate, zones = zone_air_change_rate(out, model)
    assert "ahs(Ret)" not in zones and "ahs(Sup)" in zones and rate.shape[-1] == len(zones)
    assert bool(torch.isfinite(rate).all())
    sup = zones.index("ahs(Sup)")
    torch.testing.assert_close(
        rate[:, sup], _inflow_into(model, f, "ahs(Sup)") / out["species.capacity"][:, row[
            "ahs(Sup)"]] * 3600, rtol=1e-14, atol=0)


def test_building_air_change_rate_counts_the_air_handler_and_its_outdoor_air():
    """ContamX (measured on this project to its print precision): the conditioned volume
    is the zones' plus the plenums' (1802 m3), and air enters it through the envelope (its
    `path` column) and the air handler's outdoor-air intake (its `duct` column)."""
    project = read_prj(AHS)
    out = simulate_project(project)
    model, _, _ = project_to_model(project)
    f = _layer_q(model, out["air.q"])
    torch.testing.assert_close(building_air_change_rate(out, model),
                               _inflow_from(model, f, "ambient") / (1.2041 * 1802.0) * 3600,
                               rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match=r"storage-free node.*'ahs\(Ret\)'"):
        building_air_change_rate(out, model, conditioned=["one", "two", "three", "ahs(Sup)"])


def test_building_air_change_rate_counts_the_duct_system_inside_the_building():
    """ContamX (measured on this project to its print precision): the conditioned volume
    includes every duct segment's (1800.61 m3), and the air a supply terminal delivers to
    a zone is not counted; what enters is the outdoor air drawn in at the ambient terminal
    (its `duct` column), here all of it."""
    project = read_prj(DUCTS)
    out = simulate_project(project)
    model, _, _ = project_to_model(project)
    net = model.net
    ducts = 0.0314159 * (4 + 1 + 0.7 + 5 + 1 + 6) + 0.0176715 * 3
    vol = net.node_attr("volume", default=0.0)
    assert sum(vol[net.node_index(j)].item() for j in project.junctions) == pytest.approx(
        ducts, rel=1e-12)
    f = _layer_q(model, out["air.q"])
    torch.testing.assert_close(building_air_change_rate(out, model),
                               _inflow_from(model, f, "ambient") / (1.2041 * (1800.0 + ducts))
                               * 3600, rtol=1e-12, atol=0)


def _zone_two_outside(base, tmp_path):
    """`base` with zone two left out of the building volume (zone flags 3 -> 19)."""
    text = base.read_bytes().decode()
    old = "   2  3   0   0   0   1   0.000   600 293.15 0 two"
    assert text.count(old) == 1
    variant = tmp_path / f"{base.stem}_two_outside.prj"
    variant.write_bytes(text.replace(old, "   2 19   0   0   0   1   0.000   600 293.15 0 two")
                        .encode())
    return read_prj(variant)


@pytest.mark.parametrize(("base", "volume"), [
    pytest.param(AHS, 1202.0, id="air-handler"),
    pytest.param(DUCTS, 1200.0 + 0.0314159 * 17.7 + 0.0176715 * 3, id="ducts"),
])
def test_a_zone_outside_the_building_volume_is_left_out_by_default(base, volume, tmp_path):
    """ContamX (measured on both variants to its print precision, 8e-8): a zone whose flags
    carry 16 is left out of the conditioned volume (1202 and 1200.61 m3), and air leaving it
    into the space enters the building. The reader marks the zone `in_building_volume` 0,
    and the default space is `project.building_zones`."""
    project = _zone_two_outside(base, tmp_path)
    assert "two" not in project.building_zones
    assert project.building_zones == [z for z in project.zones if z != "two"] + list(
        project.junctions)
    out = simulate_project(project)
    model, _, _ = project_to_model(project)
    net = model.net
    assert net.node_attr("in_building_volume", default=1.0)[net.node_index("two")] == 0.0
    vol = net.node_attr("volume", default=0.0)
    assert sum(vol[net.node_index(z)].item() for z in project.building_zones) == pytest.approx(
        volume, rel=1e-12)
    rate = building_air_change_rate(out, model)
    torch.testing.assert_close(
        rate, building_air_change_rate(out, model, conditioned=project.building_zones),
        rtol=0, atol=0)
    f = _layer_q(model, out["air.q"])
    into = _inflow_from(model, f, "ambient") + _inflow_from(model, f, "two")
    torch.testing.assert_close(rate, into / (1.2041 * volume) * 3600, rtol=1e-12, atol=0)


def test_without_the_flag_every_zone_is_in_the_building():
    project = read_prj(AHS)
    assert project.building_zones == project.zones
    model, _, _ = project_to_model(project)
    assert bool((model.net.node_attr("in_building_volume", default=1.0) == 1.0).all())


def test_a_capacity_of_the_wrong_shape_is_refused_by_name():
    model = _flow_model({"a": 100.0, "b": 200.0},
                        [("ambient", "a"), ("a", "b"), ("b", "ambient")])
    h = _q(0.1, 0.1, 0.1)
    for fn in (local_mean_age, zone_air_change_rate):
        with pytest.raises(ValueError, match=r"capacity has shape \(3,\).*2 interior rows"):
            fn(h, model, capacity=_f(1.0, 2.0, 3.0))


def _two_rooms_one_species(layout):
    """Two rooms in series with the ambient, one species, state in `layout`: "reduced"
    `(n_i,)` as `initial_state` builds it, or "stacked" `(n_i, 1)`."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("a", volume=30.0)
    net.add_node("b", volume=40.0)
    for s, t in (("ambient", "a"), ("a", "b"), ("b", "ambient")):
        net.add_edge(s, t, kind="airpath")
    layer = species_layer(net, n_species=1, rho=1.2)
    q = torch.full((3,), 0.01, dtype=F64)
    model = Model(net, {"species": layer},
                  closures=[lambda state, drivers: {"species.q": q}])
    x0 = torch.tensor([1e-4, 0.0], dtype=F64)
    state = {"species.x": x0 if layout == "reduced" else x0.unsqueeze(-1)}
    boundary = torch.zeros(1, dtype=F64)
    drivers = {"species.x_boundary": boundary if layout == "reduced" else boundary[:, None]}
    times = torch.tensor([0.0, 300.0, 600.0, 900.0], dtype=F64)
    out = simulate(model, state, drivers, times, calendar=CAL)
    return model, out


def _walker_ab():
    day = OccupancyDay([0.0, 450.0, DAY], ["a", "b", "b"])
    return Occupant(Occupancy([day]), name="walker")


def test_exposure_accepts_the_reduced_single_species_layout_of_the_standard_builder():
    occ = _walker_ab()
    model_r, out_r = _two_rooms_one_species("reduced")
    model_s, out_s = _two_rooms_one_species("stacked")
    assert out_r["species.x"].shape == (4, 2) and out_s["species.x"].shape == (4, 2, 1)
    e_r = occupant_exposure(out_r, occ, model_r, calendar=CAL)
    e_s = occupant_exposure(out_s, occ, model_s, calendar=CAL)
    assert e_s.shape == (3, 1)
    assert e_r.shape == (3,)                      # the layer's convention: no species axis
    torch.testing.assert_close(e_r, e_s[:, 0], rtol=1e-13, atol=0.0)
    torch.testing.assert_close(cumulative_exposure(e_r, out_r["time"])[-1],
                               cumulative_exposure(e_s, out_s["time"])[-1, 0])


def test_exposure_of_a_batched_reduced_history_matches_row_by_row():
    occ = _walker_ab()
    model, out = _two_rooms_one_species("reduced")
    x = out["species.x"]                                       # (N, 2)
    batched = {"time": out["time"], "species.x": torch.stack([x, 3 * x, 5 * x], dim=1)}
    e = occupant_exposure(batched, occ, model, calendar=CAL)   # (N, B=3, n_i=2) reduced
    assert e.shape == (3, 3)
    single = occupant_exposure(out, occ, model, calendar=CAL)
    for j, f in enumerate((1.0, 3.0, 5.0)):
        torch.testing.assert_close(e[:, j], f * single)


def test_exposure_refuses_a_layout_that_is_neither_reduced_nor_stacked():
    model, out = _two_rooms_one_species("reduced")
    with pytest.raises(ValueError, match=r"'species.x'.*\(N, \.\.\., 2, 1\).*\(N, \.\.\., 2\)"):
        occupant_exposure({"time": out["time"], "species.x": out["species.x"][:, :1]},
                          _walker_ab(), model, calendar=CAL)
