"""Post-processing: exposure, inhaled dose and infection risk."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.occupants import (
    Occupancy,
    OccupancyDay,
    Occupant,
    OccupantGeneration,
    occupant_inputs,
)
from noodl.apps.building_physics.outputs import (
    cumulative_exposure,
    infection_probability,
    inhaled_dose,
    occupant_exposure,
    wells_riley,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.schedules import DAY, Calendar
from noodl.apps.building_physics.simulation import simulate
from noodl.apps.building_physics.thermal import species_layer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
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
    """ContamX's exposure file: the step mean of the linearly interpolated mass fraction of
    the occupied zone, split at the move (measured to 4e-7, its print precision)."""
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


def test_a_history_without_the_species_axis_is_refused():
    model, pos = _three()
    h = _history(model, pos, [0.0, 300.0], {"one": [1.0, 1.0]})
    with pytest.raises(ValueError, match=r"'species.x'.*\(N, \.\.\., 3, 1\)"):
        occupant_exposure({"time": h["time"], "species.x": h["species.x"][..., 0]},
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
    ("ambient", ValueError, r"'smoker'.*boundary node 'ambient'"),
    ("loft", ValueError, r"'smoker'.*node 'loft'.*no flow edge"),
    ("plenum", ValueError, r"'smoker'.*storage-free node 'plenum'"),
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
