"""`noodl._pass.run_pass`: one pass of a `Model`, called directly."""

import pytest
import torch

from noodl._pass import layer_kwargs, run_pass
from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _build(**model_kw):
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    air = KirchhoffLayer(net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64),
                                               0.65)],
                         drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"])
    layers = [air]
    for name in ("species", "tracer"):
        layers.append(TransportLayer(
            net, name, capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
            boundary=["ambient"], scheme="implicit"))
    model = Model(net, layers, **model_kw)
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "species.x_boundary": torch.tensor([1e-3], dtype=F64),
        "species.sources": torch.tensor([0.0, 2e-6, 0.0], dtype=F64),
        "tracer.x_boundary": torch.tensor([2e-3], dtype=F64),
    }
    state = {"species.x": torch.tensor([1e-4, 2e-4], dtype=F64),
             "tracer.x": torch.zeros(2, dtype=F64)}
    return model, state, drivers


class _Clock:
    integrates = True
    state_keys = ("clock.t",)

    def __call__(self, state, drivers, ctx):
        return {"clock.t": state.get("clock.t", torch.zeros((), dtype=F64)) + ctx.dt}


def test_run_pass_is_model_pass():
    model, state, drivers = _build()
    a, da, _ = run_pass(model, state, drivers, 10.0, {})
    b, db, _ = model._pass(state, drivers, 10.0, {})
    assert list(a) == list(b)
    for k in a:
        assert torch.equal(a[k], b[k])
    assert db.keys() == da.keys()


def test_produced_lists_the_recomputed_keys_in_write_order():
    model, state, drivers = _build(closures=[_Clock()])
    produced = ["stale"]
    new, _, _ = run_pass(model, {**state, "clock.t": torch.tensor(1.0, dtype=F64)},
                         drivers, 2.0, {}, produced=produced)
    assert produced == ["air.phi", "air.q", "species.x", "tracer.x", "clock.t"]
    assert new["clock.t"].item() == 3.0
    assert list(new)[:2] == ["species.x", "tracer.x"]      # the fed keys first, in order


def test_closure_state_is_read_from_the_step_start():
    model, state, drivers = _build(closures=[_Clock()])
    fed = {**state, "clock.t": torch.tensor(7.0, dtype=F64)}
    start = {**state, "clock.t": torch.tensor(1.0, dtype=F64)}
    new, _, _ = run_pass(model, fed, drivers, 2.0, {}, step_from=start)
    assert new["clock.t"].item() == 3.0


@pytest.mark.parametrize("order", ["after_transport", "before_transport"])
def test_reactions_sit_on_the_side_of_the_step_reaction_order_names(order):
    decay = FirstOrderDecay(1e-2)
    model, state, drivers = _build(reactions=[("species", decay)], reaction_order=order)
    plain, _, _ = _build()[0]._pass(state, drivers, 10.0, {})
    new, _, drv = run_pass(model, state, drivers, 10.0, {})
    layer = model.transport["species"]
    q = model._kind_flows("species", new, drv)
    x0 = state["species.x"]
    sources = drivers["species.sources"]
    xb = drivers["species.x_boundary"]
    if order == "after_transport":
        want = decay.apply(layer.step(x0, q, sources, xb, 10.0), 10.0, drv)
    else:
        want = layer.step(decay.apply(x0, 10.0, drv), q, sources, xb, 10.0)
    assert torch.equal(new["species.x"], want)
    assert torch.equal(new["tracer.x"], plain["tracer.x"])      # no reaction there


def test_a_capacity_driver_is_swept_linearly_across_the_substeps():
    cap_new = torch.tensor([60.0, 100.0], dtype=F64)

    def capacity(state, drivers):
        return {"species.capacity": cap_new}

    model, state, drivers = _build(closures=[capacity], substeps={"species": 2})
    cap_old = torch.tensor([50.0, 80.0], dtype=F64)
    start = {**state, "species.capacity": cap_old}
    produced: list = []
    new, diag, drv = run_pass(model, start, drivers, 10.0, {}, produced=produced)
    assert diag["species"]["substeps"] == 2
    assert torch.equal(new["species.capacity"], cap_new)
    assert produced.index("species.capacity") < produced.index("species.x")
    layer = model.transport["species"]
    q = model._kind_flows("species", new, drv)
    args = (drivers["species.sources"], drivers["species.x_boundary"], 5.0)
    x = layer.step(state["species.x"], q, *args, capacity=cap_old + 0.5 * (cap_new - cap_old),
                   capacity_prev=cap_old)
    x = layer.step(x, q, *args, capacity=cap_new,
                   capacity_prev=cap_old + 0.5 * (cap_new - cap_old))
    assert torch.equal(new["species.x"], x)
    with pytest.raises(KeyError, match=r"step-start state must carry 'species\.capacity'"):
        run_pass(model, state, drivers, 10.0, {})


def test_boundary_transfers_only_on_the_named_layers():
    model, state, drivers = _build()
    _, diag, _ = run_pass(model, state, drivers, 10.0, {}, boundary_transfers={"tracer"})
    assert "boundary_transfer" in diag["tracer"]
    assert "boundary_transfer" not in diag["species"]
    _, diag, _ = run_pass(model, state, drivers, 10.0, {}, boundary_transfers=True)
    assert all("boundary_transfer" in diag[n] for n in ("species", "tracer"))
    _, diag, _ = run_pass(model, state, drivers, 10.0, {})
    assert not any("boundary_transfer" in diag[n] for n in ("species", "tracer"))
    with pytest.raises(TypeError, match="not a bare string"):
        run_pass(model, state, drivers, 10.0, {}, boundary_transfers="tracer")
    with pytest.raises(ValueError, match="no meaning for a steady solve"):
        run_pass(model, state, drivers, None, {}, boundary_transfers=True)


def test_layer_kwargs_applies_one_layers_options_over_the_shared_ones():
    shared = {"atol": 1e-9, "method": "newton"}
    assert layer_kwargs(shared, "air") is shared
    kw = {**shared, "layer_options": {"air": {"atol": 1e-12}}}
    assert layer_kwargs(kw, "air") == {"atol": 1e-12, "method": "newton"}
    assert layer_kwargs(kw, "other") == shared
