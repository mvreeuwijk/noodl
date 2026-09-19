from __future__ import annotations

import pytest
import torch

from tellegen.couple import (
    CONCENTRATION_TO_MASS_FRACTION,
    MASS_FRACTION_TO_CONCENTRATION,
    apply_conversion,
)

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
