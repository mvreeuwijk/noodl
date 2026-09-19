from __future__ import annotations

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
