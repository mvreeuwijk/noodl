from __future__ import annotations

import math
import re

import pytest
import torch

from tellegen.couple import (
    CONCENTRATION_TO_MASS_FRACTION,
    MASS_FRACTION_TO_CONCENTRATION,
    apply_conversion,
)
from tellegen.topology import Network

F64 = torch.float64


def test_apply_conversion_none_is_identity():
    value = torch.tensor([2.0, 3.0], dtype=F64)
    out = apply_conversion(None, value, {})
    assert torch.equal(out, value)


def test_concentration_to_mass_fraction_divides_by_rho_amb():
    value = torch.tensor(2.4, dtype=F64)  # kg/m3
    drivers = {"rho_amb": torch.tensor(1.2, dtype=F64)}
    out = apply_conversion(CONCENTRATION_TO_MASS_FRACTION, value, drivers)
    assert out.item() == pytest.approx(2.0)  # kg/kg


def test_mass_fraction_to_concentration_multiplies_by_rho_amb():
    value = torch.tensor(2.0, dtype=F64)  # kg/kg
    drivers = {"rho_amb": torch.tensor(1.2, dtype=F64)}
    out = apply_conversion(MASS_FRACTION_TO_CONCENTRATION, value, drivers)
    assert out.item() == pytest.approx(2.4)  # kg/m3


def test_apply_conversion_unknown_name_raises_naming_it():
    with pytest.raises(KeyError, match="bogus"):
        apply_conversion("bogus", torch.tensor(1.0, dtype=F64), {})


def _line_network() -> object:
    """boundary --edge0--> mid --edge1--> boundary2, kind='link'. `mid` is interior."""
    from tellegen.topology import Network

    net = Network(dtype=F64)
    net.add_node("boundary")
    net.add_node("mid")
    net.add_node("boundary2")
    net.add_edge("boundary", "mid", kind="link")
    net.add_edge("mid", "boundary2", kind="link")
    return net


def test_transport_boundary_inflow_hand_computed():
    from tellegen.couple import transport_boundary_inflow

    net = _line_network()
    interior_idx = net.interior_index(["boundary", "boundary2"])  # just "mid"
    boundary_idx = net.boundary_index(["boundary", "boundary2"])
    # q > 0 means source -> target: edge0 carries 3.0 INTO "mid", edge1 carries 3.0 OUT of "mid".
    q = torch.tensor([3.0, 3.0], dtype=F64)
    x_interior = torch.tensor([5.0], dtype=F64)   # concentration at "mid"
    x_boundary = torch.tensor([2.0, 5.0], dtype=F64)  # "boundary"=2.0, "boundary2" sees mid's 5.0
    # Net inflow at "boundary" (position 0 in boundary_idx): edge0 flows OUT of "boundary" at
    # rate 3.0, carrying "boundary"'s own concentration 2.0 (boundary is the edge's SOURCE, so
    # it is upwind of itself) -- net inflow at "boundary" is therefore -3.0 * 2.0 = -6.0.
    inflow = transport_boundary_inflow(
        net, q, ["link"], x_interior, x_boundary, interior_idx, boundary_idx, node_position=0,
    )
    assert inflow.item() == pytest.approx(-6.0)
    # Net inflow at "boundary2" (position 1): edge1 flows INTO "boundary2" at rate 3.0,
    # carrying "mid"'s concentration 5.0 (mid is upwind) -- net inflow is +3.0 * 5.0 = 15.0.
    inflow2 = transport_boundary_inflow(
        net, q, ["link"], x_interior, x_boundary, interior_idx, boundary_idx, node_position=1,
    )
    assert inflow2.item() == pytest.approx(15.0)


def _tiny_street_model():
    """Two segments 'seg0'->'atm', 'seg1'->'atm', kind='vent'. Concentration state only."""
    from tellegen.layers.transport import TransportLayer
    from tellegen.model import Model

    net = Network(dtype=F64)
    net.add_node("atm")
    net.add_node("seg0")
    net.add_node("seg1")
    net.add_edge("seg0", "atm", kind="vent")
    net.add_edge("seg1", "atm", kind="vent")
    layer = TransportLayer(
        net, "street", capacity=torch.tensor([10.0, 10.0], dtype=F64), flow_kind="vent",
        boundary=["atm"], quantity="concentration", unit="kg/m3",
    )

    def closure(state, drivers):
        return {"street.q": torch.tensor([1.0, 1.0], dtype=F64)}  # both segments vent OUT at 1.0

    model = Model(net, {"street": layer}, closures=[closure])
    state = {"street.x": torch.tensor([3.0, 4.0], dtype=F64)}
    drivers = {"street.x_boundary": torch.zeros(1, dtype=F64),
               "street.sources": torch.zeros(3, dtype=F64)}
    return model, state, drivers


def _tiny_building_model():
    """One zone 'z0' with an airpath to 'ambient'. Species state only."""
    from tellegen.layers.transport import TransportLayer
    from tellegen.model import Model

    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z0")
    net.add_edge("z0", "ambient", kind="airpath")
    layer = TransportLayer(
        net, "species", capacity=torch.tensor([5.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], quantity="mass_fraction", unit="kg/kg",
    )

    def closure(state, drivers):
        # NEGATIVE: edge is z0(source)->ambient(target), and Network.upwind picks the
        # TARGET as upwind when q < 0 -- so this is infiltration FROM ambient INTO z0.
        # This sign is required, not cosmetic: with q > 0 (z0 exhausting outward), "ambient"
        # is never the upwind node for this edge, so `species.x_boundary`'s value never
        # enters the state update at all (`AdvectionOperator.boundary_forcing` evaluates to
        # zero regardless of x_boundary) and a test asserting on the resulting `species.x`
        # cannot distinguish a working coupling from a no-op one (found in Task 2's review).
        return {"species.q": torch.tensor([-0.5], dtype=F64)}  # ambient infiltrates INTO z0

    model = Model(net, {"species": layer}, closures=[closure])
    state = {"species.x": torch.tensor([0.1], dtype=F64)}
    drivers = {
        "species.x_boundary": torch.zeros(1, dtype=F64),
        "rho_amb": torch.tensor(1.2, dtype=F64),
    }
    return model, state, drivers


def test_one_way_union_sets_building_boundary_from_street_segment():
    from tellegen.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    link = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert="concentration_to_mass_fraction",
    )
    city, state, drivers = union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[link],
    )
    new_state = city.step(state, drivers, dt=1.0)
    # street.x[0] = 3.0 kg/m3 -> mass fraction 3.0 / 1.2 = 2.5 kg/kg must have been used as
    # building's species.x_boundary for this step -- checked indirectly via the building's
    # own resulting state matching a standalone run given that exact boundary value.
    expected_building_model, _, expected_building_drivers = _tiny_building_model()
    expected_building_drivers["species.x_boundary"] = torch.tensor([2.5], dtype=F64)
    expected = expected_building_model.step(building_state, expected_building_drivers, dt=1.0)
    assert torch.allclose(new_state["building"]["species.x"], expected["species.x"])


def _two_way_link():  # -> ValueLink (imported in-body, like every other test here)
    from tellegen.couple import ValueLink

    return ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert="concentration_to_mass_fraction", two_way=True,
    )


def _city(**kwargs):
    from tellegen.couple import union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    return union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[_two_way_link()], **kwargs,
    ), (street_model, building_model)


def test_two_way_step_is_a_fixed_point_of_one_step_from_the_start_state():
    """The converged output, fed back as glue values, must reproduce itself from ONE step
    of each model from the START state -- design spec A1. A time-compounding iteration
    (stepping from the previous pass's output) fails this: its output is not one dt away."""
    from tellegen.couple import transport_boundary_inflow

    (city, state, drivers), (street_model, building_model) = _city(
        iterate_rtol=1e-12, iterate_max=100)
    diag: dict = {}
    new = city.step(state, drivers, dt=1.0, diagnostics=diag)
    assert diag["converged"] and 2 <= diag["passes"] <= 100

    # Glue values from the OUTPUT, by hand:
    forward = new["street"]["street.x"][0] / drivers["building"]["rho_amb"]
    building_drivers = dict(drivers["building"])
    building_drivers["species.x_boundary"] = torch.tensor([forward.item()], dtype=F64)
    species = building_model.transport["species"]
    q = building_model.current_flows("species", new["building"], building_drivers)
    inflow = transport_boundary_inflow(
        building_model.net, q, species.flow_kinds, new["building"]["species.x"],
        building_drivers["species.x_boundary"], species.interior_idx, species.boundary_idx, 0,
    )
    street_drivers = dict(drivers["street"])
    sources = street_drivers["street.sources"].clone()
    street = street_model.transport["street"]
    sources[int(street.interior_idx[0])] += inflow
    street_drivers["street.sources"] = sources

    # ONE step of each model from the START state with those glue values:
    expect_building = building_model.step(state["building"], building_drivers, dt=1.0)
    expect_street = street_model.step(state["street"], street_drivers, dt=1.0)
    assert torch.allclose(
        new["building"]["species.x"], expect_building["species.x"], rtol=1e-9, atol=1e-14)
    assert torch.allclose(
        new["street"]["street.x"], expect_street["street.x"], rtol=1e-9, atol=1e-14)


def test_two_way_step_differs_from_a_one_way_pass_and_is_sensitive_to_the_glue():
    from tellegen.couple import ValueLink, union

    (city, state, drivers), _ = _city(iterate_max=100)
    two_way = city.step(state, drivers, dt=1.0)
    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    one_way_link = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert="concentration_to_mass_fraction",
    )
    loose, s1, d1 = union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[one_way_link],
    )
    one_way = loose.step(s1, d1, dt=1.0)
    # The street must have felt the building (a sink: infiltration draws segment air):
    assert not torch.allclose(two_way["street"]["street.x"], one_way["street"]["street.x"])
    # and the building must have seen the END-of-step street value, not the start value:
    assert not torch.allclose(two_way["building"]["species.x"], one_way["building"]["species.x"])


def test_two_way_union_raises_with_the_real_largest_change_when_it_does_not_converge():
    (city, state, drivers), _ = _city(iterate_rtol=0.0, iterate_atol=0.0, iterate_max=2)
    with pytest.raises(RuntimeError, match="did not converge") as excinfo:
        city.step(state, drivers, dt=1.0)
    largest = float(re.search(r"largest change[^0-9]*([0-9.eE+-]+)", str(excinfo.value)).group(1))
    assert largest > 0.0  # never the stale 0.0 the aliased post-loop computation produced


def test_two_way_link_with_iterate_max_below_two_is_refused_at_construction():
    with pytest.raises(ValueError, match="iterate_max"):
        _city(iterate_max=1)


def test_driver_alias_writes_every_target_through_its_own_conversion():
    """The source is authoritative and each target is written from it through that target's
    registered conversion (design spec A3): the street's theta_w, radians counter-clockwise
    from east, reaches the building as CONTAM's Wd, degrees clockwise from north."""
    from tellegen.couple import STREET_RAD_TO_CONTAM_DEG, DriverAlias, ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    street_drivers["theta_w"] = torch.tensor(0.5 * math.pi, dtype=F64)  # blowing north
    building_drivers["theta_w"] = torch.tensor(-1.0, dtype=F64)  # stale; must be overwritten
    seen: dict = {}
    original_step = building_model.step

    def spy(state, drivers, dt, **kwargs):
        seen.update(drivers)
        return original_step(state, drivers, dt, **kwargs)

    building_model.step = spy
    city, state, drivers = union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[
            ValueLink(from_model="street", from_key="street.x", from_index=0,
                      to_model="building", to_key="species.x_boundary", to_index=0,
                      convert="concentration_to_mass_fraction"),
            DriverAlias(source=("street", "theta_w"),
                        targets=(("building", "theta_w", STREET_RAD_TO_CONTAM_DEG),)),
        ],
    )
    city.step(state, drivers, dt=1.0)
    assert seen["theta_w"].item() == pytest.approx(180.0)  # (270 - 90) mod 360
    # The source is untouched, and the caller's own driver dicts are never modified:
    assert drivers["street"]["theta_w"].item() == pytest.approx(0.5 * math.pi)
    assert building_drivers["theta_w"].item() == -1.0


def test_unregistered_conversion_is_refused_at_union_construction():
    from tellegen.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0, convert="furlongs",
    )
    with pytest.raises(KeyError, match="furlongs"):
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])


def test_reduced_accepts_reduced_and_stacked_single_species_layouts():
    from tellegen.couple import _reduced

    reduced = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    stacked = reduced.reshape(3, 1)
    assert torch.equal(_reduced(reduced, 3), reduced)
    assert torch.equal(_reduced(stacked, 3), reduced)
    batched_stacked = torch.arange(6, dtype=F64).reshape(2, 3, 1)
    assert _reduced(batched_stacked, 3).shape == (2, 3)
    with pytest.raises(ValueError, match="3 entries"):
        _reduced(torch.zeros(4, dtype=F64), 3)


def test_write_at_restores_the_stacked_layout():
    from tellegen.couple import _write_at

    target = torch.zeros(1, 1, dtype=F64)  # CONTAM's x_boundary layout, n_b = 1, K = 1
    out = _write_at(target, 1, 0, torch.tensor(2.5, dtype=F64))
    assert out.shape == (1, 1) and out.item() == pytest.approx(2.5)
    assert target.item() == 0.0  # the input was cloned, not written in place
    batched = torch.zeros(4, 2, 1, dtype=F64)
    out = _write_at(batched, 2, 1, torch.arange(4, dtype=F64))
    assert out.shape == (4, 2, 1)
    assert torch.equal(out[:, 1, 0], torch.arange(4, dtype=F64))
    assert torch.all(out[:, 0, 0] == 0)


def test_write_at_broadcasts_a_batched_value_into_an_unbatched_target():
    """The ensemble case: the street model runs a batch of B forcings while the CONTAM
    reader's `species.x_boundary` stays `(1, 1)`. The target must broadcast up to the
    value's batch, keeping its own LAYOUT (stacked stays stacked)."""
    from tellegen.couple import _write_at

    value = torch.arange(4, dtype=F64)
    out = _write_at(torch.zeros(1, 1, dtype=F64), 1, 0, value)  # stacked, n_b = 1
    assert out.shape == (4, 1, 1)
    assert torch.equal(out[:, 0, 0], value)
    reduced_target = torch.tensor([7.0, 8.0], dtype=F64)  # reduced, n_b = 2
    out = _write_at(reduced_target, 2, 1, torch.arange(3, dtype=F64))
    assert out.shape == (3, 2)
    assert torch.equal(out[:, 1], torch.arange(3, dtype=F64))
    assert torch.all(out[:, 0] == 7.0)  # the untouched entry is broadcast, not zeroed
    assert torch.equal(reduced_target, torch.tensor([7.0, 8.0], dtype=F64))  # not in place


@pytest.mark.parametrize("wd, theta", [(270.0, 0.0), (0.0, 1.5 * math.pi),
                                       (90.0, math.pi), (180.0, 0.5 * math.pi)])
def test_wind_direction_conversions_on_the_cardinal_points(wd, theta):
    from tellegen.couple import CONTAM_DEG_TO_STREET_RAD, STREET_RAD_TO_CONTAM_DEG

    wd_t = torch.tensor(wd, dtype=F64)
    theta_t = torch.tensor(theta, dtype=F64)
    assert apply_conversion(CONTAM_DEG_TO_STREET_RAD, wd_t, {}).item() == pytest.approx(
        theta, abs=1e-12)
    assert apply_conversion(STREET_RAD_TO_CONTAM_DEG, theta_t, {}).item() == pytest.approx(
        wd, abs=1e-9)
