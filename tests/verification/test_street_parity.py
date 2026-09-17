"""IMPAQ parity — spec section 7, rows 8 and 9.

Part one, here: the four-node network, where the tellegen model and the ported oracle can
be compared to machine precision. Part two, Task 10: `leiden_small` on the real data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tellegen.apps.street.impaq import (
    canyon_velocity,
    compute_boundary_layer,
    network_from_street_network,
    solve_steady_state,
)
from tellegen.apps.street.network import build_street_model, from_test_network

DT = torch.float64
U_REF, THETA_W, H_ABL, BACKGROUND = 2.0, 0.25 * math.pi, 1200.0, 1.0e-4
EMISSION = np.array([1.0, 1.0, 1.0])


def _oracle(street_net):
    network = network_from_street_network(street_net, EMISSION)
    layer = compute_boundary_layer(network, BACKGROUND, U_REF, THETA_W,
                                   reference_height_m=30.0, abl_height_m=H_ABL)
    network.roads.canyon_velocity_mps = canyon_velocity(
        network.roads, layer.friction_velocity_mps, layer.wind_angle_rad
    )
    return network, layer


def _tellegen(street_net, **options):
    model, state, _ = build_street_model(
        street_net, canyon_wind="soulhac", exchange="sirane",
        direction_averaging="none", kappa=0.4, canyon_wind_min=0.0, u_d_min=0.0,
        stability="impaq", z_ref=30.0, pblh_floor=False, **options,
    )
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    for street, value in zip(street_net.streets, EMISSION, strict=True):
        sources[net.node_index(street.name)] = float(value)
    drivers = {
        "street.x_boundary": torch.tensor([BACKGROUND], dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(U_REF, dtype=DT),
        "theta_w": torch.tensor(THETA_W, dtype=DT),
        "h_abl": torch.tensor(H_ABL, dtype=DT),
    }
    return model.steady(state, drivers)["street.x"].detach().numpy()


def test_the_canyon_velocities_agree_with_the_oracle_s_scipy_ones():
    street_net = from_test_network()
    network, _layer = _oracle(street_net)
    model, state, _ = build_street_model(street_net, kappa=0.4, pblh_floor=False)
    resolved = model._apply_closures(state, {
        "U_ref": torch.tensor(U_REF, dtype=DT),
        "theta_w": torch.tensor(THETA_W, dtype=DT),
        "h_abl": torch.tensor(H_ABL, dtype=DT),
    })
    np.testing.assert_allclose(
        resolved["street.u_canyon"].detach().numpy(),
        network.roads.canyon_velocity_mps, rtol=1e-9, atol=1e-15,
    )


@pytest.mark.parametrize("routing", ["mixing", "sirane"])
def test_tellegen_matches_the_fixed_impaq_oracle_on_the_four_node_network(routing):
    """Both routing models give the same answer here: no junction of this network has two
    inflows AND two outflows, so there is nothing for a routing model to decide. The test
    is about the ELIMINATION and the closure, not about routing -- Task 11's 2-in/2-out
    node is what distinguishes the routing models."""
    street_net = from_test_network()
    network, layer = _oracle(street_net)
    ours = _tellegen(street_net, routing=routing)
    theirs = solve_steady_state(network, layer, fix_a=True, fix_b=True)[:len(ours)]
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=0)


def test_the_effect_of_each_impaq_fix_is_reported_not_absorbed(capsys):
    street_net = from_test_network()
    network, layer = _oracle(street_net)
    ours = _tellegen(street_net, routing="mixing")
    rows = []
    for fix_a, fix_b in ((False, False), (True, False), (True, True)):
        theirs = solve_steady_state(network, layer, fix_a=fix_a,
                                    fix_b=fix_b)[:len(ours)]
        rows.append((fix_a, fix_b,
                     float(np.max(np.abs(ours - theirs) / np.abs(theirs)))))
    with capsys.disabled():
        print("\nIMPAQ parity 1, relative difference from the tellegen model:")
        for fix_a, fix_b, relative in rows:
            print(f"  fix_a={fix_a!s:5s} fix_b={fix_b!s:5s} -> {relative:.3e}")
    effects = {(a, b): r for a, b, r in rows}
    # Issue A is worth 37 % on this network; with it fixed the two models agree to
    # machine precision, and issue B changes nothing once A is fixed.
    assert 0.30 < effects[(False, False)] < 0.45
    assert effects[(True, True)] < 1e-9
    assert effects[(True, False)] < 1e-9
    with pytest.raises(ValueError, match=r"fix_b without fix_a is only defined"):
        solve_steady_state(network, layer, fix_b=True)


def test_the_exchange_coefficient_is_the_retracted_issue_c_form_on_both_sides():
    """`u_d = sigma_w/(sqrt(2) pi)`, not `sigma_w/sqrt(2 pi)`. The two differ by
    `sqrt(pi) = 1.7725`, so a model using the other one cannot agree with this oracle at
    any tolerance -- which makes the parity above evidence for the retraction."""
    from tellegen.apps.street.canyon import SIRANE_EXCHANGE

    assert abs(SIRANE_EXCHANGE - 1.0 / (math.sqrt(2.0) * math.pi)) < 1e-16
    wrong = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(wrong / SIRANE_EXCHANGE - math.sqrt(math.pi)) < 1e-12
