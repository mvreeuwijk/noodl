"""Building physics: ensemble equivalence, species permutation, constant tracer, warm start
and gradients, on committed CONTAM projects."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.species import KineticReaction, reaction_kinetics
from noodl.apps.building_physics.thermal import species_layer
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
TIGHT = {"atol": 1e-13, "rtol": 1e-13}   # the airflow solve's residual bounds the tracer drift


def _project(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return project_to_model(read_prj(DATA / f"{name}.prj"))


def _ambient_temperatures(drivers, b):
    """`b` ambient temperatures, the other nodes at their project value: shape (b, nodes)."""
    t = drivers["T"].expand(b, -1).clone()
    t[:, 0] = 283.15 + 4.0 * torch.arange(b, dtype=F64)
    return t


# doorway_damper_fan: 2 nodes, 8 airpaths, 3 species; the batch sizes hit both counts.
@pytest.mark.parametrize("b", [1, 2, 3, 8])
def test_batched_ambient_temperature_equals_stack_of_single_steps(b):
    model, state, drivers = _project("doorway_damper_fan")
    d = dict(drivers, T=_ambient_temperatures(drivers, b))
    out = model.step(state, d, 300.0, **TIGHT)
    assert out["air.q"].shape == (b, 8) and out["species.x"].shape == (b, 1, 3)
    for i in range(b):
        single = model.step(state, dict(d, T=d["T"][i]), 300.0, **TIGHT)
        for key in ("air.q", "air.phi", "species.x"):
            torch.testing.assert_close(out[key][i], single[key], rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("name", ["doorway_damper_fan", "valThreeZonesWthCtm-UseApi"])
@pytest.mark.parametrize("b", [None, 3])
def test_a_uniform_tracer_is_preserved_by_the_airflow_transport(name, b):
    """Initial and ambient concentration c per species, no sources: x stays c, for one and
    for three species, unbatched and over a batch of ambient temperatures."""
    model, state, drivers = _project(name)
    k = state["species.x"].shape[-1]
    c = torch.tensor([[1e-3, 2e-3, 3e-3][:k]], dtype=F64)
    d = dict(drivers, **{"species.x_boundary": c.clone()})
    if "species.source_multiplier" in d:
        d["species.source_multiplier"] = torch.zeros_like(d["species.source_multiplier"])
    if b:
        d["T"] = _ambient_temperatures(drivers, b)
    state = dict(state, **{"species.x": c.expand(state["species.x"].shape).clone()})
    for _ in range(3):
        state = model.step(state, d, 300.0, **TIGHT)
    expect = c.expand(state["species.x"].shape)
    torch.testing.assert_close(state["species.x"], expect, rtol=1e-9, atol=1e-15)


def _network():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("one", volume=300.0)
    net.add_node("two", volume=600.0)
    net.add_edge("ambient", "one", kind="airpath")
    net.add_edge("one", "two", kind="airpath")
    net.add_edge("two", "ambient", kind="airpath")
    return net


@pytest.mark.parametrize("order", [("a", "b", "c"), ("c", "a", "b"), ("b", "c", "a")])
def test_species_order_permutes_kinetic_transport(order):
    """Kinetics are keyed by species NAME: reordering the species reorders the columns of
    the result and nothing else; the passive species 'c' is never reacted."""
    reaction = KineticReaction({("a", "a"): -1e-3, ("b", "a"): 1e-3, ("b", "b"): -5e-4},
                               name="chain")
    start = {"a": 1e-3, "b": 2e-4, "c": 5e-4}
    boundary = {"a": 4e-4, "b": 0.0, "c": 7e-4}
    q = torch.tensor([0.4, 0.4, 0.4], dtype=F64)

    def run(names):
        net = _network()
        kin = reaction_kinetics(net, species=list(names), reactions={"one": reaction,
                                                                     "two": reaction})
        layer = species_layer(net, n_species=3, species_names=list(names), kinetics=kin)
        x = torch.tensor([[start[n] for n in names]] * 2, dtype=F64)
        xb = torch.tensor([[boundary[n] for n in names]], dtype=F64)
        for _ in range(5):
            x = layer.step(x, q, torch.zeros(3, 3, dtype=F64), xb, 300.0)
        return x

    ref = run(("a", "b", "c"))
    cols = [("a", "b", "c").index(n) for n in order]
    torch.testing.assert_close(run(order), ref[:, cols], rtol=1e-12, atol=1e-18)


def test_the_airflow_solution_is_independent_of_the_warm_start_and_has_a_fd_gradient():
    """Value from a cold, a converged and a distant starting potential; and the autograd
    derivative of a flow/concentration loss with respect to the ambient temperature
    equals the central difference."""
    model, state, drivers = _project("doorway_damper_fan")

    def run(t_amb, st):
        t = torch.stack([t_amb, drivers["T"][1]])
        return model.step(st, dict(drivers, T=t), 300.0, **TIGHT)

    def loss(t_amb, st=state):
        out = run(t_amb, st)
        return (out["air.q"] ** 2).sum() + 1e3 * out["species.x"].sum()

    ref = run(drivers["T"][0], state)
    for phi in (ref["air.phi"].detach(), torch.tensor([0.0, 50.0], dtype=F64),
                torch.tensor([-30.0, 0.0], dtype=F64)):
        out = run(drivers["T"][0], dict(state, **{"air.phi": phi}))
        torch.testing.assert_close(out["air.q"], ref["air.q"], rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(out["species.x"], ref["species.x"], rtol=1e-9, atol=1e-15)

    t = drivers["T"][0].clone().requires_grad_(True)
    loss(t).backward()
    h = 1e-4
    with torch.no_grad():
        fd = (loss(t + h) - loss(t - h)) / (2 * h)
    assert t.grad.item() == pytest.approx(fd.item(), rel=1e-5)
