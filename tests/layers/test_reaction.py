"""Tests for local reactions applied after a transport step by operator splitting."""

import math

import pytest
import torch

from tellegen.layers.reaction import (
    K_NO_O3,
    K_NO_O3_298,
    MOLAR_MASS,
    FirstOrderDecay,
    Photostationary,
)
from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network


def sealed_zone() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def test_zero_flow_decay_matches_analytic_exponential_exactly():
    reaction = FirstOrderDecay(0.05)
    x = torch.tensor([200.0], dtype=torch.float64)
    dt = 30.0
    for _ in range(10):
        x = reaction.apply(x, dt)
    expected = 200.0 * math.exp(-0.05 * 10 * dt)
    torch.testing.assert_close(
        x, torch.tensor([expected], dtype=torch.float64), rtol=1e-10, atol=1e-12
    )


def test_operator_splitting_matches_removal_matrix_route_to_first_order():
    net = sealed_zone()
    rate = 0.01
    cap = torch.tensor([1000.0], dtype=torch.float64)
    q = torch.tensor([0.2, 0.2], dtype=torch.float64)
    src = torch.zeros(net.n, dtype=torch.float64)  # FULL node order (spec 4.2); all zero
    x_b = torch.tensor([420.0], dtype=torch.float64)
    dt = 5.0  # small step so the splitting error stays within first-order tolerance

    layer_removal = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([rate], dtype=torch.float64),
    )
    layer_plain = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                  boundary=["ambient"])
    reaction = FirstOrderDecay(rate)

    x_removal = torch.tensor([300.0], dtype=torch.float64)
    x_split = torch.tensor([300.0], dtype=torch.float64)
    for _ in range(20):
        x_removal = layer_removal.step(x_removal, q, src, x_b, dt)
        x_split = layer_plain.step(x_split, q, src, x_b, dt)
        x_split = reaction.apply(x_split, dt)

    torch.testing.assert_close(x_removal, x_split, rtol=2e-3, atol=1e-3)


DT = torch.float64


def _state(no, no2, o3, n=1):
    x = torch.zeros(n, 3, dtype=DT)
    x[:, 0], x[:, 1], x[:, 2] = no, no2, o3
    return x


def _molar(y):
    m = torch.tensor([MOLAR_MASS["no"], MOLAR_MASS["no2"], MOLAR_MASS["o3"]], dtype=DT)
    return y / m


def test_rate_constant_matches_the_arrhenius_form_and_the_kg_conversion():
    k3 = 3.0e-12 * math.exp(-1500.0 / 298.0)
    assert abs(K_NO_O3_298 / k3 - 1.0) < 1e-15
    conversion = 6.02214076e23 / (1e6 * 48.0e-3)
    assert abs(K_NO_O3 / (K_NO_O3_298 * conversion) - 1.0) < 1e-15
    # The spec's section 7 row prints k' to six significant figures; that rounding, and
    # nothing else, is why this check is 2e-6 and not 1e-9.
    assert abs(K_NO_O3 / 2.45236e5 - 1.0) < 2e-6
    assert MOLAR_MASS == {"no": 30.0e-3, "no2": 46.0e-3, "o3": 48.0e-3}


def test_photostationary_satisfies_the_state_and_conserves_nox_and_ox():
    x = _state(2.0e-8, 4.0e-8, 6.0e-8)
    y = Photostationary(0, 1, 2).apply(x, None, {"J_NO2": torch.tensor([5.0e-3], dtype=DT)})
    c0, c1 = _molar(x), _molar(y)
    torch.testing.assert_close(c1[:, 0] + c1[:, 1], c0[:, 0] + c0[:, 1], rtol=1e-13, atol=0)
    torch.testing.assert_close(c1[:, 1] + c1[:, 2], c0[:, 1] + c0[:, 2], rtol=1e-13, atol=0)
    k_mol = K_NO_O3 * MOLAR_MASS["o3"]
    residual = 5.0e-3 * c1[:, 1] - k_mol * c1[:, 0] * c1[:, 2]
    assert float((residual / (5.0e-3 * c1[:, 1])).abs().max()) < 1e-12
    assert bool((y >= 0).all())


def test_zero_photolysis_consumes_whichever_of_no_and_o3_runs_out_first():
    y = Photostationary(0, 1, 2).apply(
        _state(2.0e-8, 4.0e-8, 6.0e-8), None, {"J_NO2": torch.zeros(1, dtype=DT)}
    )
    c = _molar(y)
    p = 2.0e-8 / MOLAR_MASS["no"] + 4.0e-8 / MOLAR_MASS["no2"]
    q = 4.0e-8 / MOLAR_MASS["no2"] + 6.0e-8 / MOLAR_MASS["o3"]
    torch.testing.assert_close(c[:, 1], torch.tensor([min(p, q)], dtype=DT),
                               rtol=1e-13, atol=0)
    assert float(c[:, 0].min()) >= 0.0 and float(c[:, 2].min()) >= 0.0


def test_huge_photolysis_drives_no2_towards_zero():
    x = _state(2.0e-8, 4.0e-8, 6.0e-8)
    y = Photostationary(0, 1, 2).apply(
        x, None, {"J_NO2": torch.tensor([1.0e6], dtype=DT)}
    )
    # `z -> 2 k P Q / J` as `J -> inf`, so NO2 does not reach exactly zero in float64; what
    # matters is that essentially all of the NOx has become NO.
    assert float(y[0, 1]) >= 0.0
    assert float(y[0, 1]) / float(x[0, 1]) < 1e-6


def test_photostationary_is_idempotent():
    reaction = Photostationary(0, 1, 2)
    drivers = {"J_NO2": torch.tensor([5.0e-3], dtype=DT)}
    y = reaction.apply(_state(2.0e-8, 4.0e-8, 6.0e-8), None, drivers)
    torch.testing.assert_close(reaction.apply(y, None, drivers), y, rtol=1e-12, atol=0)


def test_photostationary_is_differentiable_in_the_state_and_in_j():
    x = _state(2.0e-8, 4.0e-8, 6.0e-8).requires_grad_(True)
    j = torch.tensor([5.0e-3], dtype=DT, requires_grad=True)
    y = Photostationary(0, 1, 2).apply(x, None, {"J_NO2": j})
    gx, gj = torch.autograd.grad(y[:, 1].sum(), (x, j))
    assert torch.isfinite(gx).all() and torch.isfinite(gj).all()
    assert float(gj) < 0.0          # more photolysis, less NO2


def test_photostationary_batches_over_instances_and_nodes():
    x = torch.stack([_state(2.0e-8, 4.0e-8, 6.0e-8, n=4),
                     _state(1.0e-8, 2.0e-8, 9.0e-8, n=4)])
    j = torch.tensor([[5.0e-3], [1.0e-3]], dtype=DT)
    y = Photostationary(0, 1, 2).apply(x, None, {"J_NO2": j})
    assert y.shape == x.shape
    assert bool((y[0, 0] != y[1, 0]).any())


def test_photostationary_names_a_missing_driver_and_a_bad_column():
    with pytest.raises(KeyError, match="J_NO2"):
        Photostationary(0, 1, 2).apply(_state(1e-8, 1e-8, 1e-8), None, {})
    with pytest.raises(ValueError, match=r"Photostationary: column 7"):
        Photostationary(0, 1, 7).apply(_state(1e-8, 1e-8, 1e-8), None,
                                       {"J_NO2": torch.ones(1, dtype=DT)})
