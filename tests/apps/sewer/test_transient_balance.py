"""R5 in the application: with storage=True the water-quality capacity changes every step;
the total amount in the network plus what left through the outfall must be constant when
no load enters and no reaction runs."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.sewer import geometry as geom
from noodl.apps.sewer.network import build_model, tree_steady

F64 = torch.float64


def test_transport_through_a_filling_and_draining_sewer_conserves_the_amount():
    model, state, drivers = build_model(
        tree_steady(), air=False, quality=True, storage=True,
    )
    model.reactions = []                       # transport alone: no sulfide generation or BOD decay
    drivers = dict(drivers)
    dt = 60.0
    base_inflow = drivers["inflow"].clone()
    # Spin up from dry under the base inflows so the levels are physical, then seed.
    for _ in range(60):
        state = model.step(state, drivers, dt)
    state = dict(state)
    state["water_quality.x"] = torch.ones_like(state["water_quality.x"])
    layer = model.transport["water_quality"]
    j4 = layer.interior_idx.tolist().index(model.net.node_index("J4"))
    out_pipe = tree_steady().pipes[4]           # C5: J4 -> Outfall, 0.45 m, n 0.013, S 0.005
    assert out_pipe.name == "C5"

    def amount(s):
        return float((s["water_quality.capacity"].unsqueeze(-1) * s["water_quality.x"]).sum())

    total_out = 0.0
    m0 = amount(state)
    factors = [0.7, 0.9, 1.15, 1.2, 1.0, 0.8, 0.6, 0.75, 1.1, 1.15]
    for f in factors:
        drivers["inflow"] = base_inflow * f
        new = model.step(state, drivers, dt)
        q_out = geom.manning_flow(
            new["sewer.H"][..., j4], torch.tensor(out_pipe.diameter, dtype=F64),
            torch.tensor(out_pipe.n, dtype=F64), torch.tensor(out_pipe.slope, dtype=F64),
        )
        total_out += float(dt * q_out * new["water_quality.x"][j4].sum())
        state = new
    assert amount(state) + total_out == pytest.approx(m0, rel=1e-9)
