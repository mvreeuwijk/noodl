import pytest
import torch

from noodl.apps.street.exposure import (
    Q_INHALATION,
    exposure_reduction_adjoint,
    exposure_reduction_forward,
    street_population,
    total_exposure,
)
from noodl.apps.street.network import build_street_model, from_test_network

F64 = torch.float64


def _street():
    net = from_test_network()
    # z_ref left at the model's own default (30.0 m): from_test_network()'s streets are up
    # to 30 m tall, and boundary_layer requires z_ref to clear H's displacement height plus
    # roughness -- 10 m (test_coupling_demo._street()'s value, for its own 3 m network)
    # does not, and raises. z_ref is unrelated to this module's own physics.
    model, state, drivers = build_street_model(net, species=("nox",))
    graph = model.net
    sources = torch.zeros(graph.n, dtype=F64)
    for k, s in enumerate(net.streets):
        sources[graph.node_index(s.name)] = 1.0e-7 * (1 + k)
    drivers = dict(drivers)
    drivers.update({
        "street.x_boundary": torch.tensor([2.0e-8], dtype=F64),
        "street.sources": sources,
        "U_ref": torch.tensor(3.0, dtype=F64),
        "theta_w": torch.tensor(0.7, dtype=F64),
        "h_abl": torch.tensor(1000.0, dtype=F64),
    })
    return net, model, state, drivers


def test_population_is_proportional_to_height_times_length_and_sums_to_total():
    net = from_test_network()
    p = street_population(net, n_total=1000.0)
    hl = torch.tensor([s.height * s.length for s in net.streets], dtype=F64)
    assert torch.allclose(p, 1000.0 * hl / hl.sum())
    assert abs(p.sum().item() - 1000.0) < 1e-9


def test_total_exposure_is_q_p_c():
    c = torch.tensor([1.0, 2.0], dtype=F64)
    p = torch.tensor([10.0, 20.0], dtype=F64)
    assert abs(total_exposure(c, p).item() - Q_INHALATION * 50.0) < 1e-15


def test_adjoint_reduction_matches_forward_reduction_and_costs_one_solve():
    net, model, state, drivers = _street()
    names = [s.name for s in net.streets]
    p = street_population(net, n_total=500.0)
    r_adj, cost_adj = exposure_reduction_adjoint(model, state, drivers, population=p, streets=names)
    r_fwd, cost_fwd = exposure_reduction_forward(model, state, drivers, population=p, streets=names)
    assert r_adj.shape == (len(names),)
    assert torch.allclose(r_adj, r_fwd, rtol=1e-6, atol=0.0)   # passive scalar: linear in Q
    assert cost_adj == {"solves": 1, "backward": 1}
    assert cost_fwd == {"solves": len(names) + 1}
    assert bool((r_adj > 0).all())   # reducing any street's emission reduces exposure


def test_batched_sources_are_rejected_loudly():
    net, model, state, drivers = _street()
    names = [s.name for s in net.streets]
    p = street_population(net, n_total=500.0)
    drivers = dict(drivers)
    drivers["street.sources"] = torch.stack([drivers["street.sources"]] * 2)
    with pytest.raises(ValueError, match="batch"):
        exposure_reduction_adjoint(model, state, drivers, population=p, streets=names)
    with pytest.raises(ValueError, match="batch"):
        exposure_reduction_forward(model, state, drivers, population=p, streets=names)
