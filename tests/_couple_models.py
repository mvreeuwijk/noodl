"""Test-only models shared by the coupling tests (`tests/test_couple.py`) and the puff
street's coupling to a building (`tests/apps/street_aq/test_puff_coupled.py`)."""

from __future__ import annotations

import torch

from noodl.topology import Network

F64 = torch.float64


def tiny_building_model():
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
        # cannot distinguish a working coupling from a no-op one.
        return {"species.q": torch.tensor([-0.5], dtype=F64)}  # ambient infiltrates INTO z0

    model = Model(net, {"species": layer}, closures=[closure])
    state = {"species.x": torch.tensor([0.1], dtype=F64)}
    drivers = {
        "species.x_boundary": torch.zeros(1, dtype=F64),
        "rho_amb": torch.tensor(1.2, dtype=F64),
    }
    return model, state, drivers
