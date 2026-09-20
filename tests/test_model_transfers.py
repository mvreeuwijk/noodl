"""Model.step exposes each transport layer's boundary transfer, summed over substeps."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _t(v):
    return torch.tensor(v, dtype=F64)


def _model(scheme, substeps):
    net = Network(dtype=F64)
    for n in ("amb", "z"):
        net.add_node(n)
    net.add_edge("amb", "z", kind="flow")
    net.add_edge("z", "amb", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([2.0]), flow_kind="flow", boundary=["amb"],
                            scheme=scheme)
    model = Model(net, {"c": layer}, closures=[lambda s, d: {"c.q": _t([0.5, 0.5])}],
                  substeps={"c": substeps})
    return model


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal", "exact"])
@pytest.mark.parametrize("substeps", [1, 3])
def test_transfer_is_summed_over_substeps_and_closes_the_balance(scheme, substeps):
    model = _model(scheme, substeps)
    state = {"c.x": _t([1.0])}
    drivers = {"c.x_boundary": _t([4.0]), "c.sources": _t([0.0, 0.3])}
    diag: dict = {}
    new = model.step(state, drivers, 1.2, diagnostics=diag, boundary_transfers=True)
    transfer = diag["layers"]["c"]["boundary_transfer"]
    assert transfer.shape == (1,)
    gained = 2.0 * (new["c.x"] - state["c.x"]).item()
    tol = 1e-9 if scheme == "exact" else 1e-12
    assert abs(gained - 1.2 * 0.3 + transfer.item()) < tol


def test_without_the_flag_no_transfer_is_reported_and_the_state_is_identical():
    model = _model("exact", 1)
    state = {"c.x": _t([1.0])}
    drivers = {"c.x_boundary": _t([0.0]), "c.sources": _t([0.0, 0.0])}
    d1: dict = {}
    d2: dict = {}
    a = model.step(state, drivers, 1.2, diagnostics=d1)
    b = model.step(state, drivers, 1.2, diagnostics=d2, boundary_transfers=True)
    assert "boundary_transfer" not in d1["layers"]["c"]
    torch.testing.assert_close(a["c.x"], b["c.x"], rtol=1e-12, atol=1e-14)
