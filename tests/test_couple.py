from __future__ import annotations

import math
import re

import pytest
import torch

from noodl.couple import (
    CONCENTRATION_TO_MASS_FRACTION,
    MASS_FRACTION_TO_CONCENTRATION,
    apply_conversion,
)
from noodl.topology import Network

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
    from noodl.topology import Network

    net = Network(dtype=F64)
    net.add_node("boundary")
    net.add_node("mid")
    net.add_node("boundary2")
    net.add_edge("boundary", "mid", kind="link")
    net.add_edge("mid", "boundary2", kind="link")
    return net


def test_transport_boundary_inflow_hand_computed():
    from noodl.couple import transport_boundary_inflow

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


def _dense_boundary_inflow(
    net, q, kind, x_interior, x_boundary, interior_idx, boundary_idx, node_position
):
    """The milestone 5 dense form, kept here as the oracle for the sparse helper."""
    batch_shape = torch.broadcast_shapes(
        x_interior.shape[:-1], x_boundary.shape[:-1], q.shape[:-1]
    )
    full = torch.zeros(*batch_shape, net.n, dtype=F64)
    full[..., interior_idx] = x_interior.expand(*batch_shape, interior_idx.numel())
    full[..., boundary_idx] = x_boundary.expand(*batch_shape, boundary_idx.numel())
    selector = net.upwind(q, kind)
    mass_flux = q * torch.einsum("...bn,...n->...b", selector, full)
    return -net.accumulate(mass_flux, kind)[..., int(boundary_idx[node_position])]


@pytest.mark.parametrize("seed", range(4))
def test_transport_boundary_inflow_matches_the_dense_form_on_signed_batched_multi_boundary_cases(  # noqa: E501
    seed,
):
    from noodl.couple import transport_boundary_inflow

    g = torch.Generator().manual_seed(seed)
    net = Network(dtype=F64)
    for n in ("b0", "b1", "i0", "i1", "i2"):
        net.add_node(n)
    edges = [
        ("b0", "i0"), ("i0", "i1"), ("i1", "b1"),
        ("i2", "i0"), ("b1", "i2"), ("i1", "i2"),
    ]
    for u, v in edges:
        net.add_edge(u, v, kind="flow")
    interior_idx = net.interior_index(["b0", "b1"])
    boundary_idx = net.boundary_index(["b0", "b1"])
    q = torch.randn(3, 6, generator=g, dtype=F64)
    xi = torch.rand(3, 3, generator=g, dtype=F64)
    xb = torch.rand(3, 2, generator=g, dtype=F64)
    for pos in (0, 1):
        got = transport_boundary_inflow(net, q, ["flow"], xi, xb, interior_idx, boundary_idx, pos)
        want = _dense_boundary_inflow(net, q, "flow", xi, xb, interior_idx, boundary_idx, pos)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def _tiny_street_model():
    """Two segments 'seg0'->'atm', 'seg1'->'atm', kind='vent'. Concentration state only."""
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model

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
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model

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
    from noodl.couple import ValueLink, union

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
    from noodl.couple import ValueLink

    return ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert="concentration_to_mass_fraction", two_way=True,
    )


def _city(**kwargs):
    from noodl.couple import union

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
    (stepping from the previous pass's output) fails this: its output is not one dt away.

    Task 17 (R1, R2): the coupler now steps the RECIPIENT (building) first and feeds the
    DONOR (street) the recipient's own integrated boundary transfer as a source rate, rather
    than an endpoint flux recomputed from a stale state; at convergence, the returned donor
    forward value equals the boundary value the recipient was stepped with (the convergence
    criterion itself), so this hand reconstruction -- using the OUTPUT state's endpoint flux
    as the constant source rate -- still reproduces the same one-step-from-start fixed point
    to within the assertion's tolerance.
    """
    from noodl.couple import transport_boundary_inflow

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

    # The recipient's own integrated transfer, reported in diagnostics, is exactly the
    # source RATE the donor (street) was stepped with, times dt (R1: an integrated amount,
    # not an endpoint flux held fixed) -- `inflow` above IS that rate, by construction of
    # the hand reconstruction, so at this converged fixed point they must agree.
    key = "street:street.x[0]->building:species.x_boundary"
    assert diag["transfers"][key].item() == pytest.approx(inflow.item() * 1.0, rel=1e-9)


def test_two_way_step_differs_from_a_one_way_pass_and_is_sensitive_to_the_glue():
    from noodl.couple import ValueLink, union

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


def test_two_way_union_raises_naming_the_link_the_instances_and_the_real_largest_change():
    (city, state, drivers), _ = _city(iterate_rtol=0.0, iterate_atol=0.0, iterate_max=2)
    with pytest.raises(RuntimeError, match="did not converge") as excinfo:
        city.step(state, drivers, dt=1.0)
    message = str(excinfo.value)
    key = "street:street.x[0]->building:species.x_boundary"
    assert key in message  # which LINK failed, not just "the shared value(s)"
    assert "for instances all" in message  # unbatched run: every instance failed
    largest = float(re.search(rf"{re.escape(key)}'?: ?([0-9.eE+-]+)", message).group(1))
    assert largest > 0.0  # never the stale 0.0 the aliased post-loop computation produced


def test_two_way_convergence_and_non_convergence_are_judged_per_batch_instance():
    """A batch of street forcings: `converged` is a mask over the batch, `max_change` is per
    link and per instance, and the failure message names ONLY the instances that failed --
    as `Model._iterate` does (`model.py:595-604`).

    Task 17 (R1, R2): the coupler now judges convergence on the donor's RETURNED forward
    value against the value the recipient was stepped with THIS pass, from the first pass
    on -- re-measured below for this fixture (unchanged by the scheme change here: instance
    0 still converges by pass 4, instance 1 needs 7 total).
    """
    from noodl.couple import union

    def batched_city(**kwargs):
        street_model, street_state, street_drivers = _tiny_street_model()
        building_model, building_state, building_drivers = _tiny_building_model()
        # Two instances, differing tenfold at the coupled segment, so instance 1's residual
        # is ~10x instance 0's at every pass and one absolute tolerance separates them:
        street_state["street.x"] = torch.tensor([[3.0, 4.0], [30.0, 4.0]], dtype=F64)
        return union(
            {"street": (street_model, street_state, street_drivers),
             "building": (building_model, building_state, building_drivers)},
            shared=[_two_way_link()], **kwargs,
        )

    city, state, drivers = batched_city(iterate_rtol=1e-12, iterate_max=100)
    diag: dict = {}
    new = city.step(state, drivers, dt=1.0, diagnostics=diag)
    key = "street:street.x[0]->building:species.x_boundary"
    assert diag["converged"].shape == (2,) and bool(diag["converged"].all())
    assert diag["max_change"][key].shape == (2,)
    assert new["building"]["species.x"].shape == (2, 1)
    # Instance 0 is the unbatched fixture, and must reproduce its unbatched answer:
    (solo, s0, d0), _ = _city(iterate_rtol=1e-12, iterate_max=100)
    one = solo.step(s0, d0, dt=1.0)
    assert torch.allclose(
        new["building"]["species.x"][0], one["building"]["species.x"], rtol=1e-9, atol=1e-14)
    assert torch.allclose(
        new["street"]["street.x"][0], one["street"]["street.x"], rtol=1e-9, atol=1e-14)

    # A tolerance instance 0 meets within 6 passes and instance 1 does not (re-measured for
    # the recipient-first schedule: instance 0 is done by pass 4, instance 1 needs 7 total):
    city, state, drivers = batched_city(iterate_rtol=0.0, iterate_atol=0.05, iterate_max=6)
    with pytest.raises(RuntimeError, match=r"for instances \[1\]"):
        city.step(state, drivers, dt=1.0)


def test_two_way_feedback_adds_to_the_callers_own_sources_and_never_overwrites_them():
    """The caller's own source terms at the coupled node must survive: the feedback flux is
    ADDED to them. Every other fixture supplies zeros, where add and overwrite agree.

    Task 17 (R1, R2): the added quantity is now `transfer / dt`, the recipient's own
    integrated boundary transfer (reported in `diagnostics["transfers"]`) divided by dt, not
    `transport_boundary_inflow` recomputed from the previous pass's state.
    """
    from noodl.couple import union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    # atm, seg0, seg1 in full node order: seg0 (the coupled node) already emits 0.7, seg1 0.2.
    own_sources = torch.tensor([0.0, 0.7, 0.2], dtype=F64)
    street_drivers["street.sources"] = own_sources
    city, state, drivers = union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[_two_way_link()], iterate_rtol=1e-12, iterate_max=100,
    )
    diag: dict = {}
    dt = 1.0
    new = city.step(state, drivers, dt=dt, diagnostics=diag)

    key = "street:street.x[0]->building:species.x_boundary"
    transfer = diag["transfers"][key]
    assert transfer.item() != pytest.approx(0.0)  # there IS a flux to add
    rate = transfer / dt
    street = street_model.transport["street"]
    node = int(street.interior_idx[0])
    expected_sources = own_sources.clone()
    expected_sources[node] = expected_sources[node] + rate
    assert expected_sources[node].item() == pytest.approx(0.7 + rate.item())  # ADDED to 0.7
    assert expected_sources[2].item() == 0.2  # the caller's other entries are untouched
    hand_street = dict(drivers["street"])
    hand_street["street.sources"] = expected_sources
    expect_street = street_model.step(state["street"], hand_street, dt=dt)
    assert torch.allclose(
        new["street"]["street.x"], expect_street["street.x"], rtol=1e-9, atol=1e-14)
    # and the caller's own tensor was never written into:
    assert torch.equal(own_sources, torch.tensor([0.0, 0.7, 0.2], dtype=F64))


def test_a_to_key_that_is_not_a_boundary_driver_is_refused_at_construction():
    from noodl.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.sources", to_index=0,
    )
    with pytest.raises(ValueError, match="x_boundary"):
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])


def test_a_sources_key_that_is_not_a_source_term_is_refused_at_construction():
    from noodl.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        two_way=True, sources_key="street.capacity",
    )
    with pytest.raises(ValueError, match="sources"):
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])


def test_two_way_link_with_iterate_max_below_two_is_refused_at_construction():
    with pytest.raises(ValueError, match="iterate_max"):
        _city(iterate_max=1)


def test_driver_alias_writes_every_target_through_its_own_conversion():
    """The source is authoritative and each target is written from it through that target's
    registered conversion (design spec A3): the street's theta_w, radians counter-clockwise
    from east, reaches the building as CONTAM's Wd, degrees clockwise from north."""
    from noodl.couple import STREET_RAD_TO_CONTAM_DEG, DriverAlias, ValueLink, union

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
    from noodl.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0, convert="furlongs",
    )
    with pytest.raises(KeyError, match="furlongs"):
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])


def test_an_unconverted_link_between_unequal_units_is_refused_at_construction():
    """`convert=None` across kg/m3 -> kg/kg is silently wrong by a factor of `rho_amb`, and
    both models keep running happily -- so it must be refused at construction."""
    from noodl.couple import ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,  # convert=None
    )
    with pytest.raises(ValueError) as excinfo:
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])
    message = str(excinfo.value)
    assert "street:street.x[0]->building:species.x_boundary" in message  # WHICH link
    assert "'kg/m3'" in message and "'kg/kg'" in message                 # and both units
    assert "concentration" in message and "mass_fraction" in message     # and both quantities


def test_a_conversion_whose_units_do_not_match_the_layers_is_refused_at_construction():
    from noodl.couple import STREET_RAD_TO_CONTAM_DEG, ValueLink, union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    bad = ValueLink(  # an ANGLE conversion on a species link: rad -> deg, not kg/m3 -> kg/kg
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert=STREET_RAD_TO_CONTAM_DEG,
    )
    with pytest.raises(ValueError) as excinfo:
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)}, shared=[bad])
    message = str(excinfo.value)
    assert "street:street.x[0]->building:species.x_boundary" in message
    assert "'rad'" in message and "'deg'" in message      # the conversion's own units
    assert "'kg/m3'" in message and "'kg/kg'" in message  # against the layers'


def _multi_kind_building_model():
    """`_tiny_building_model` with TWO advecting edge kinds on its species layer."""
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model

    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z0")
    net.add_edge("z0", "ambient", kind="a")
    net.add_edge("z0", "ambient", kind="b")
    layer = TransportLayer(
        net, "species", capacity=torch.tensor([5.0], dtype=F64), flow_kind=("a", "b"),
        boundary=["ambient"], quantity="mass_fraction", unit="kg/kg",
    )

    def closure(state, drivers):
        return {"species.q": torch.tensor([-0.5, -0.5], dtype=F64)}

    model = Model(net, {"species": layer}, closures=[closure])
    state = {"species.x": torch.tensor([0.1], dtype=F64)}
    drivers = {"species.x_boundary": torch.zeros(1, dtype=F64),
               "rho_amb": torch.tensor(1.2, dtype=F64)}
    return model, state, drivers


def _two_species_building_model():
    """`_tiny_building_model` with TWO species on its species layer."""
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model

    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z0")
    net.add_edge("z0", "ambient", kind="airpath")
    layer = TransportLayer(
        net, "species", capacity=torch.tensor([5.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], n_species=2, quantity="mass_fraction", unit="kg/kg",
    )

    def closure(state, drivers):
        return {"species.q": torch.tensor([-0.5], dtype=F64)}

    model = Model(net, {"species": layer}, closures=[closure])
    state = {"species.x": torch.zeros(1, 2, dtype=F64)}
    drivers = {"species.x_boundary": torch.zeros(1, 2, dtype=F64),
               "rho_amb": torch.tensor(1.2, dtype=F64)}
    return model, state, drivers


def test_a_two_way_link_into_a_multi_flow_kind_layer_is_refused_at_construction():
    """`transport_boundary_inflow` is single-flow-kind; it used to discover that at STEP
    time, after a whole first pass had run."""
    from noodl.couple import union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _multi_kind_building_model()
    with pytest.raises(ValueError, match="single flow kind") as excinfo:
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)},
              shared=[_two_way_link()])
    message = str(excinfo.value)
    assert "street:street.x[0]->building:species.x_boundary" in message
    assert "'a'" in message and "'b'" in message  # the kinds it actually has


def test_a_two_way_link_on_a_multi_species_layer_is_refused_at_construction():
    """A stacked `(n_i, 1)` and a reduced `(n_i, K)` with `K == n_i` are the same shape, so
    the glue's layout rule cannot read a multi-species state unambiguously."""
    from noodl.couple import union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _two_species_building_model()
    with pytest.raises(ValueError, match="n_species == 1") as excinfo:
        union({"street": (street_model, street_state, street_drivers),
               "building": (building_model, building_state, building_drivers)},
              shared=[_two_way_link()])
    message = str(excinfo.value)
    assert "street:street.x[0]->building:species.x_boundary" in message
    assert "building:species has n_species=2" in message  # WHICH layer is out of scope


def test_reduced_accepts_reduced_and_stacked_single_species_layouts():
    from noodl.couple import _reduced

    reduced = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    stacked = reduced.reshape(3, 1)
    assert torch.equal(_reduced(reduced, 3), reduced)
    assert torch.equal(_reduced(stacked, 3), reduced)
    batched_stacked = torch.arange(6, dtype=F64).reshape(2, 3, 1)
    assert _reduced(batched_stacked, 3).shape == (2, 3)
    with pytest.raises(ValueError, match="3 entries"):
        _reduced(torch.zeros(4, dtype=F64), 3)


def test_write_at_restores_the_stacked_layout():
    from noodl.couple import _write_at

    target = torch.zeros(1, 1, dtype=F64)  # CONTAM's x_boundary layout, n_b = 1, K = 1
    out = _write_at(target, 1, 0, torch.tensor(2.5, dtype=F64))
    assert out.shape == (1, 1) and out.item() == pytest.approx(2.5)
    assert target.item() == 0.0  # the input was cloned, not written in place
    batched = torch.zeros(4, 2, 1, dtype=F64)
    out = _write_at(batched, 2, 1, torch.arange(4, dtype=F64))
    assert out.shape == (4, 2, 1)
    assert torch.equal(out[:, 1, 0], torch.arange(4, dtype=F64))
    assert torch.all(out[:, 0, 0] == 0)


def test_write_at_add_accumulates_and_reduces_a_stacked_sources_tensor():
    """`add=True` is the two-way feedback's write into "<layer>.sources". A STACKED
    single-species sources tensor must be reduced first: indexing it directly would address
    the species axis, silently writing the flux at the wrong place."""
    from noodl.couple import _write_at

    stacked = torch.tensor([[0.0], [0.7], [0.2]], dtype=F64)  # (n, 1), full node order
    out = _write_at(stacked, 3, 1, torch.tensor(0.5, dtype=F64), add=True)
    assert out.shape == (3, 1)
    assert out[:, 0].tolist() == pytest.approx([0.0, 1.2, 0.2])  # ADDED to 0.7, not 0.5
    reduced = torch.tensor([0.0, 0.7, 0.2], dtype=F64)
    out = _write_at(reduced, 3, 1, torch.tensor(0.5, dtype=F64), add=True)
    assert out.tolist() == pytest.approx([0.0, 1.2, 0.2])
    assert torch.equal(stacked, torch.tensor([[0.0], [0.7], [0.2]], dtype=F64))  # not in place


def test_write_at_broadcasts_a_batched_value_into_an_unbatched_target():
    """The ensemble case: the street model runs a batch of B forcings while the CONTAM
    reader's `species.x_boundary` stays `(1, 1)`. The target must broadcast up to the
    value's batch, keeping its own LAYOUT (stacked stays stacked)."""
    from noodl.couple import _write_at

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
    from noodl.couple import CONTAM_DEG_TO_STREET_RAD, STREET_RAD_TO_CONTAM_DEG

    wd_t = torch.tensor(wd, dtype=F64)
    theta_t = torch.tensor(theta, dtype=F64)
    assert apply_conversion(CONTAM_DEG_TO_STREET_RAD, wd_t, {}).item() == pytest.approx(
        theta, abs=1e-12)
    assert apply_conversion(STREET_RAD_TO_CONTAM_DEG, theta_t, {}).item() == pytest.approx(
        wd, abs=1e-9)


def test_original_models_still_run_standalone_unchanged_after_union():
    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    baseline_street = street_model.step(dict(street_state), dict(street_drivers), dt=1.0)
    baseline_building = building_model.step(dict(building_state), dict(building_drivers), dt=1.0)
    street_snapshot = {k: v.clone() for k, v in street_drivers.items()}
    building_snapshot = {k: v.clone() for k, v in building_drivers.items()}

    from noodl.couple import union
    city, state, drivers = union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[_two_way_link()], iterate_max=100,
    )
    city.step(state, drivers, dt=1.0)

    after_street = street_model.step(dict(street_state), dict(street_drivers), dt=1.0)
    after_building = building_model.step(dict(building_state), dict(building_drivers), dt=1.0)
    assert torch.equal(baseline_street["street.x"], after_street["street.x"])
    assert torch.equal(baseline_building["species.x"], after_building["species.x"])
    for k, v in street_snapshot.items():
        assert torch.equal(street_drivers[k], v), k   # the caller's driver tensors untouched
    for k, v in building_snapshot.items():
        assert torch.equal(building_drivers[k], v), k


def test_substeps_calls_the_fast_model_k_times_with_the_glue_held_constant():
    from noodl.couple import union

    street_model, street_state, street_drivers = _tiny_street_model()
    building_model, building_state, building_drivers = _tiny_building_model()
    calls: list[tuple[float, torch.Tensor]] = []
    real_step = building_model.step

    def spy(state, drivers, dt, **kw):
        # The boundary VALUE is kept as a tensor and compared exactly below: rounding it to
        # a float first (`round(v, 15)`) makes the constancy check vacuous at the ~1e-8
        # magnitudes a real mass fraction has, where every candidate value rounds together.
        calls.append((float(dt), drivers["species.x_boundary"].detach().clone()))
        return real_step(state, drivers, dt, **kw)

    building_model.step = spy  # a per-instance spy; the class is untouched
    try:
        city, state, drivers = union(
            {"street": (street_model, street_state, street_drivers),
             "building": (building_model, building_state, building_drivers)},
            shared=[_two_way_link()], substeps={"building": 6}, iterate_max=100,
        )
        diag: dict = {}
        city.step(state, drivers, dt=60.0, diagnostics=diag)
    finally:
        building_model.step = real_step
    passes = diag["passes"]
    assert len(calls) == 6 * passes
    for p in range(passes):
        chunk = calls[6 * p:6 * (p + 1)]
        # 60 s in six 10 s sub-steps, boundary held EXACTLY constant across them:
        assert all(dt == pytest.approx(10.0) for dt, _ in chunk)
        first = chunk[0][1]
        assert all(torch.equal(v, first) for _, v in chunk)


def test_gradient_flows_across_the_join_and_matches_central_differences():
    """d(building species.x) / d(street segment-0 initial concentration), through the
    coupled two-way step -- design spec section 7, 'gradients flow across the join'."""
    from noodl.couple import union

    def indoor(x0: torch.Tensor) -> torch.Tensor:
        street_model, street_state, street_drivers = _tiny_street_model()
        building_model, building_state, building_drivers = _tiny_building_model()
        street_state = dict(street_state)
        street_state["street.x"] = torch.stack([x0, street_state["street.x"][1]])
        city, state, drivers = union(
            {"street": (street_model, street_state, street_drivers),
             "building": (building_model, building_state, building_drivers)},
            shared=[_two_way_link()], iterate_rtol=1e-12, iterate_max=200,
        )
        return city.step(state, drivers, dt=1.0)["building"]["species.x"].sum()

    x0 = torch.tensor(3.0, dtype=F64, requires_grad=True)
    grad, = torch.autograd.grad(indoor(x0), x0)
    h = 1e-5
    fd = (
        indoor(torch.tensor(3.0 + h, dtype=F64)) - indoor(torch.tensor(3.0 - h, dtype=F64))
    ) / (2 * h)
    assert grad.item() == pytest.approx(fd.item(), rel=1e-5)
    assert abs(grad.item()) > 0.0
