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


def test_a_bare_string_boundary_transfers_is_refused_rather_than_iterated_as_characters():
    """`set("species")` iterates CHARACTERS, not the one name meant: on this fixture's own
    single-letter layer name "c", a bare string would otherwise be silently accepted and
    happen to select the right layer by coincidence. Refused outright instead, naming the
    string."""
    model = _model("exact", 1)
    state = {"c.x": _t([1.0])}
    drivers = {"c.x_boundary": _t([0.0]), "c.sources": _t([0.0, 0.0])}
    with pytest.raises(TypeError, match="bare string"):
        model.step(state, drivers, 1.2, boundary_transfers="c")


def _two_layer_model():
    """`thermal` (scheme='exact') and `species` (scheme='implicit') on the same net, each
    with its own flow kind -- so a caller who names only one in `boundary_transfers` can be
    checked against the other (`step_with_transfer`'s extra cost, no diagonal
    shift on the `exact` scheme's Taylor accumulator, must not be paid on a layer nothing
    reads a transfer from)."""
    net = Network(dtype=F64)
    for n in ("amb", "z"):
        net.add_node(n)
    net.add_edge("amb", "z", kind="heat")
    net.add_edge("z", "amb", kind="heat")
    net.add_edge("amb", "z", kind="conc")
    net.add_edge("z", "amb", kind="conc")
    thermal = TransportLayer(net, "thermal", capacity=_t([2.0]), flow_kind="heat",
                              boundary=["amb"], scheme="exact")
    species = TransportLayer(net, "species", capacity=_t([3.0]), flow_kind="conc",
                              boundary=["amb"], scheme="implicit")

    def closure(state, drivers):
        return {"thermal.q": _t([0.5, 0.5]), "species.q": _t([0.7, 0.7])}

    return Model(net, {"thermal": thermal, "species": species}, closures=[closure])


def _two_layer_fixture():
    model = _two_layer_model()
    state = {"thermal.x": _t([1.0]), "species.x": _t([2.0])}
    drivers = {
        "thermal.x_boundary": _t([4.0]), "species.x_boundary": _t([5.0]),
        "thermal.sources": _t([0.0, 0.0]), "species.sources": _t([0.0, 0.0]),
    }
    return model, state, drivers


def test_boundary_transfers_as_a_collection_runs_step_with_transfer_only_on_named_layers(
    monkeypatch,
):
    model, state, drivers = _two_layer_fixture()
    calls: list[str] = []
    original = TransportLayer.step_with_transfer

    def recording(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TransportLayer, "step_with_transfer", recording)
    diag: dict = {}
    model.step(state, drivers, 1.0, diagnostics=diag, boundary_transfers={"species"})
    assert "boundary_transfer" in diag["layers"]["species"]
    assert "boundary_transfer" not in diag["layers"]["thermal"]
    assert calls == ["species"]  # thermal took the plain, cheaper `step`


def test_boundary_transfers_naming_an_unknown_layer_is_refused_by_name():
    model, state, drivers = _two_layer_fixture()
    with pytest.raises(ValueError, match="bogus"):
        model.step(state, drivers, 1.0, boundary_transfers={"bogus"})
