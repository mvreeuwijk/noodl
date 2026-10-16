"""Sewer: ensemble equivalence, species permutation and constant-tracer preservation, for
kinematic and dynamic-wave routing.

Dynamic-wave withdrawals, flooding and batching have their own tests in `test_dynwave.py`;
kinematic routing refuses withdrawals, so its tracer invariant has none.
"""

from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, tree_steady
from noodl.apps.sewer.dynwave import simulate_dynamic_wave

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
LOADS = {"bod": 0.3, "sulfide": 0.01, "tracer": 0.1}
ORDERS = [("bod", "sulfide", "tracer"), ("tracer", "sulfide", "bod"), ("sulfide", "bod", "tracer"),
          ("bod", "tracer", "sulfide")]


def _dynamic_tree(tmp_path):
    text = (DATA / "tree_kinwave.inp").read_text()
    text = text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
    text = text.replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
    # converged tightly: a uniform tracer is preserved to the iteration's flow tolerance
    text = text.replace("MAX_TRIALS           8", "MAX_TRIALS           2000")
    text = text.replace("HEAD_TOLERANCE       0.0015", "HEAD_TOLERANCE       1e-12")
    path = tmp_path / "tree_dyn.inp"
    path.write_text(text)
    return read_swmm_inp(path)[0]


def _kinematic(species, loads=LOADS, steps=4):
    model, state, drivers = build_model(tree_steady(), air=False, quality=True, species=species)
    n = model.net.n
    drivers = dict(drivers, **{f"{k}_in": torch.full((n,), loads[k], dtype=F64) for k in species})
    x0 = torch.tensor([[loads[k] for k in species]] * 5, dtype=F64)
    state = dict(state, **{"water_quality.x": x0.squeeze(-1) if len(species) == 1 else x0})
    state.update(model.initial_capacities(state, drivers))
    for _ in range(steps):
        state = model.step(state, drivers, 60.0)
    return model, state, drivers


def _dynamic(net, species, loads=LOADS, steps=6):
    model, state, drivers = build_model(net, quality=True, species=species)
    n = model.net.n
    drivers = dict(drivers, **{f"{k}_in": torch.full((n,), loads[k], dtype=F64) for k in species})
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=steps)
    return model, state


@pytest.mark.parametrize("species", ORDERS[1:])
def test_kinematic_species_order_permutes_the_result(species):
    """The reactions are keyed by species NAME, so reordering the species reorders the
    columns and changes nothing else (sulfide generation, BOD decay, passive tracer)."""
    ref = _kinematic(ORDERS[0])[1]["water_quality.x"]
    out = _kinematic(species)[1]["water_quality.x"]
    cols = [ORDERS[0].index(k) for k in species]
    torch.testing.assert_close(out, ref[..., cols], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("species", ORDERS[1:3])
def test_dynamic_wave_species_order_permutes_the_result(tmp_path, species):
    net = _dynamic_tree(tmp_path)
    ref = _dynamic(net, ORDERS[0])[1]["water_quality.x"]
    out = _dynamic(net, species)[1]["water_quality.x"]
    cols = [ORDERS[0].index(k) for k in species]
    torch.testing.assert_close(out, ref[..., cols], rtol=1e-10, atol=1e-14)


def test_passive_tracer_is_unreacted_next_to_reacting_species():
    """The tracer column of a bod+sulfide+tracer run equals a tracer-only run."""
    alone = _kinematic(("tracer",))[1]["water_quality.x"]
    mixed = _kinematic(("bod", "sulfide", "tracer"))[1]["water_quality.x"][..., 2]
    torch.testing.assert_close(mixed, alone, rtol=1e-12, atol=1e-15)


def test_dynamic_wave_uniform_conservative_tracer_stays_uniform(tmp_path):
    """Initial, lateral and boundary concentration all c: the tracer stays c everywhere,
    next to a reacting species."""
    c = 0.37
    model, state, drivers = build_model(_dynamic_tree(tmp_path), quality=True,
                                        species=("bod", "tracer"))
    n = model.net.n
    drivers = dict(drivers, tracer_in=torch.full((n,), c, dtype=F64),
                   bod_in=torch.full((n,), 0.3, dtype=F64),
                   **{"water_quality.x_boundary": torch.tensor([[0.3, c]], dtype=F64)})
    x0 = state["water_quality.x"].clone()
    x0[:, 1] = c
    state = dict(state, **{"water_quality.x": x0})
    for _ in range(8):
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
    torch.testing.assert_close(state["water_quality.x"][:, 1],
                               torch.full_like(state["water_quality.x"][:, 1], c),
                               rtol=1e-8, atol=0.0)


def test_kinematic_uniform_tracer_fed_laterally_stays_uniform():
    c = 0.37
    model, _, drivers = _kinematic(("tracer",), {"tracer": c}, steps=0)
    drivers = dict(drivers, **{"water_quality.x_boundary": torch.tensor([[c]], dtype=F64)})
    state = _kinematic(("tracer",), {"tracer": c}, steps=0)[1]
    for _ in range(6):
        state = model.step(state, drivers, 60.0)
    torch.testing.assert_close(state["water_quality.x"], torch.full((5,), c, dtype=F64),
                               rtol=1e-8, atol=0.0)


@pytest.mark.parametrize("b", [1, 5, 6])  # 5 manholes, 6 nodes
def test_kinematic_batched_inflow_equals_stack_of_single_runs(b):
    model, state, drivers = build_model(tree_steady(), air=False, quality=True,
                                        species=("bod", "tracer"))
    n = model.net.n
    scale = torch.linspace(0.4, 1.0, b, dtype=F64)
    d = dict(drivers, inflow=drivers["inflow"] * scale[:, None],
             tracer_in=scale[:, None] * torch.ones(n, dtype=F64))
    state.update(model.initial_capacities(state, d))
    out = model.step(state, d, 60.0)["water_quality.x"]
    assert out.shape == (b, 5, 2)
    for i in range(b):
        di = dict(d, inflow=d["inflow"][i], tracer_in=d["tracer_in"][i])
        s_i = dict(state)
        s_i.update(model.initial_capacities(state, di))
        torch.testing.assert_close(out[i], model.step(s_i, di, 60.0)["water_quality.x"],
                                   rtol=1e-9, atol=1e-14)
