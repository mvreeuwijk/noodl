"""IMPAQ parity — spec section 7, rows 8 and 9.

Part one, here: the four-node network, where the noodl model and the ported oracle can
be compared to machine precision. Part two, Task 10: `leiden_small` on the real data.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.street.impaq import (
    canyon_velocity,
    compute_boundary_layer,
    compute_intersection_routing,
    network_from_street_network,
    solve_steady_state,
)
from noodl.apps.street.loader import read_aqdt
from noodl.apps.street.network import build_street_model, from_test_network

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


def _noodl(street_net, **options):
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
def test_noodl_matches_the_fixed_impaq_oracle_on_the_four_node_network(routing):
    """Both routing models give the same answer here: no junction of this network has two
    inflows AND two outflows, so there is nothing for a routing model to decide. The test
    is about the ELIMINATION and the closure, not about routing -- Task 11's 2-in/2-out
    node is what distinguishes the routing models."""
    street_net = from_test_network()
    network, layer = _oracle(street_net)
    ours = _noodl(street_net, routing=routing)
    theirs = solve_steady_state(network, layer, fix_a=True, fix_b=True)[:len(ours)]
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=0)


def test_the_effect_of_each_impaq_fix_is_reported_not_absorbed(capsys):
    street_net = from_test_network()
    network, layer = _oracle(street_net)
    ours = _noodl(street_net, routing="mixing")
    rows = []
    for fix_a, fix_b in ((False, False), (True, False), (True, True)):
        theirs = solve_steady_state(network, layer, fix_a=fix_a,
                                    fix_b=fix_b)[:len(ours)]
        rows.append((fix_a, fix_b,
                     float(np.max(np.abs(ours - theirs) / np.abs(theirs)))))
    with capsys.disabled():
        print("\nIMPAQ parity 1, relative difference from the noodl model:")
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
    from noodl.apps.street.canyon import SIRANE_EXCHANGE

    assert abs(SIRANE_EXCHANGE - 1.0 / (math.sqrt(2.0) * math.pi)) < 1e-16
    wrong = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(wrong / SIRANE_EXCHANGE - math.sqrt(math.pi)) < 1e-12


# --------------------------------------------------------------- IMPAQ parity 2

AQDT_DATA = Path(os.environ.get(
    "NOODL_AQDT_DATA", r"<workspace>\tmp\2026_AQ_DT\data"
))
DOMAIN = "leiden_small"
YEAR = 2024
STEPS = [0, 1000, 2000]

needs_aqdt = pytest.mark.skipif(
    not (AQDT_DATA / "stage1_geometry" / DOMAIN / "repaired_edges_canyon.geojson").exists(),
    reason=f"the AQ_DT products are not at {AQDT_DATA}; set NOODL_AQDT_DATA",
)


def _leiden_small():
    return read_aqdt(
        AQDT_DATA / "stage1_geometry" / DOMAIN, AQDT_DATA / "stage2_inputs" / DOMAIN,
        year=YEAR, wind_height_m=30.0, trust_file_height=True, times=STEPS,
    )


def _noodl_leiden(data, **options):
    """The noodl model in IMPAQ's own formulation, batched over the sampled steps."""
    model, _state, _ = build_street_model(
        data.net, canyon_wind="soulhac", exchange="sirane", routing="sirane",
        direction_averaging="none", kappa=0.4, canyon_wind_min=0.0, u_d_min=0.0,
        stability="impaq", z_ref=30.0, pblh_floor=False, **options,
    )
    net = model.net
    sources = torch.zeros(len(STEPS), net.n, dtype=DT)
    for column, street in enumerate(data.net.streets):
        sources[:, net.node_index(street.name)] = data.emission[:, column]
    drivers = {
        "street.x_boundary": torch.zeros(len(STEPS), 1, dtype=DT),
        "street.sources": sources,
        "U_ref": data.forcing.u_ref,
        "theta_w": data.forcing.theta_w,
        "h_abl": data.forcing.h_abl,
    }
    return model, drivers, model.steady({}, drivers)


def _oracle_leiden(data, step: int):
    network = network_from_street_network(data.net, data.emission[step].numpy())
    layer = compute_boundary_layer(
        network, 0.0, float(data.forcing.u_ref[step]), float(data.forcing.theta_w[step]),
        reference_height_m=30.0, abl_height_m=float(data.forcing.h_abl[step]),
    )
    network.roads.canyon_velocity_mps = canyon_velocity(
        network.roads, layer.friction_velocity_mps, layer.wind_angle_rad
    )
    return network, layer


@needs_aqdt
def test_the_loaded_leiden_small_domain_is_the_one_the_plan_measured():
    data = _leiden_small()
    assert len(data.net.streets) == 162
    assert len(data.net.junctions) == 230
    assert data.emission.shape == (len(STEPS), 162)


@needs_aqdt
def test_the_canyon_velocities_agree_with_the_oracle_on_every_leiden_small_street():
    """Everything upstream of the junction algebra is identical to the prototype's."""
    data = _leiden_small()
    model, drivers, _solved = _noodl_leiden(data)
    resolved = model._apply_closures({}, drivers)
    ours = resolved["street.u_canyon"].detach().numpy()
    for step in range(len(STEPS)):
        network, _layer = _oracle_leiden(data, step)
        np.testing.assert_allclose(ours[step], network.roads.canyon_velocity_mps,
                                   rtol=1e-9, atol=1e-12)


@needs_aqdt
def test_the_oracle_s_routing_matrix_does_not_conserve_and_noodl_s_flows_do(capsys):
    """The diagnosis. IMPAQ's `flow_route` mis-permutes at three-way junctions, so its
    routing matrix's row sums are not the streets' own fluxes; the noodl model's
    prescribed flows close the mass balance exactly. This is why the comparison below
    cannot be exact, and it is recorded rather than absorbed."""
    data = _leiden_small()
    model, drivers, solved = _noodl_leiden(data)
    resolved = model._apply_closures(solved, drivers)
    q = model._kind_flows("street", solved, resolved)
    assert bool((q >= 0).all())
    report = []
    for step in range(len(STEPS)):
        network, _layer = _oracle_leiden(data, step)
        _topology, routing = compute_intersection_routing(network)
        n_roads = len(network.roads.s)
        flux = np.abs(network.roads.canyon_velocity_mps * network.roads.width_m
                      * network.roads.height_m)
        rows = routing[:n_roads, :].sum(axis=1)
        relative = np.abs(rows - flux) / np.maximum(flux, 1e-300)
        report.append((STEPS[step], int((relative > 1e-9).sum()), float(relative.max())))
    with capsys.disabled():
        print("\nIMPAQ routing-matrix row sums against each street's own flux:")
        for step, count, worst in report:
            print(f"  step {step:5d}: {count:3d} of 162 roads do not conserve, "
                  f"worst by a factor {1.0 + worst:.2f}")
    # Measured on 17 September 2026: 12 roads at step 0, worst factor 13.95.
    for _step, count, worst in report:
        assert count > 0 and worst > 1.0


@needs_aqdt
def test_noodl_matches_the_fixed_oracle_on_the_typical_leiden_small_street(capsys):
    """The parity that IS attainable: the median street agrees to machine precision, and
    the count of streets that do not is consistent with the mis-permuted junctions
    diagnosed above (the set cross-reference is a recorded follow-up). The counts are
    printed and asserted against the range measured while this plan was written."""
    data = _leiden_small()
    _model, _drivers, solved = _noodl_leiden(data)
    ours = solved["street.x"].detach().numpy()
    rows = []
    for step in range(len(STEPS)):
        network, layer = _oracle_leiden(data, step)
        theirs = solve_steady_state(network, layer, fix_a=True,
                                    fix_b=True)[:ours.shape[1]]
        relative = np.abs(ours[step] - theirs) / np.maximum(np.abs(theirs), 1e-300)
        rows.append((STEPS[step], float(np.median(relative)),
                     int((relative > 1e-9).sum()), float(relative.max())))
    with capsys.disabled():
        print("\nIMPAQ parity 2 on leiden_small (fix_a and fix_b on):")
        for step, median, count, worst in rows:
            print(f"  step {step:5d}: median {median:.3e}, {count:3d} of 162 streets "
                  f"over 1e-9, worst {worst:.3e}")
    for _step, median, count, _worst in rows:
        assert median < 1e-9
        assert count <= 30      # measured: 15, 21 and 12


@needs_aqdt
def test_issue_a_is_much_larger_than_the_routing_defect_on_leiden_small(capsys):
    data = _leiden_small()
    _model, _drivers, solved = _noodl_leiden(data)
    ours = solved["street.x"].detach().numpy()
    fixed, unfixed = [], []
    for step in range(len(STEPS)):
        network, layer = _oracle_leiden(data, step)
        for fix_a, sink in ((True, fixed), (False, unfixed)):
            theirs = solve_steady_state(network, layer, fix_a=fix_a,
                                        fix_b=fix_a)[:ours.shape[1]]
            relative = np.abs(ours[step] - theirs) / np.maximum(np.abs(theirs), 1e-300)
            sink.append(float(np.median(relative)))
    with capsys.disabled():
        print("\nmedian relative difference, fix_a on vs off:")
        for step, on, off in zip(STEPS, fixed, unfixed, strict=True):
            print(f"  step {step:5d}: fix_a=True {on:.3e}   fix_a=False {off:.3e}")
    assert max(fixed) < 1e-9
    assert min(unfixed) > 1e-3


@needs_aqdt
def test_the_saved_network_concentration_product_is_checked_before_it_is_believed():
    """Spec section 7's `network_concentration_2024.nc` row.

    The product on disk was written BEFORE the geometry file it names, and describes a
    network that is no longer there: 160 edges against 162 `network_transport` features,
    and 94 of its 160 rows carry an `edge_osmid` that is not the osmid of the feature its
    own `edge_feature_index` points at. The numeric comparison is therefore skipped, with
    that measurement as the reason -- and it runs unchanged, at 1e-6 relative on
    `concentration_increment`, the moment the product is regenerated.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    product = AQDT_DATA / "stage3_products" / DOMAIN / f"network_concentration_{YEAR}.nc"
    if not product.exists():
        pytest.skip(f"{product} is not there")
    dataset = netCDF4.Dataset(str(product))
    try:
        feature_index = [int(v) for v in dataset.variables["edge_feature_index"][:]]
        osmid = [int(v) for v in dataset.variables["edge_osmid"][:]]
        increment = dataset.variables["concentration_increment"]
        assert increment.dimensions == ("time", "edge")
    finally:
        dataset.close()
    data = _leiden_small()
    features = json.loads(
        (AQDT_DATA / "stage1_geometry" / DOMAIN / "repaired_edges_canyon.geojson")
        .read_text(encoding="utf-8")
    )["features"]
    mismatched = sum(
        1 for index, value in zip(feature_index, osmid, strict=True)
        if features[index]["properties"].get("osmid") != value
    )
    if mismatched or len(feature_index) != len(data.net.streets):
        pytest.skip(
            f"the saved product describes a different network: {len(feature_index)} edges "
            f"against {len(data.net.streets)} network_transport features, and "
            f"{mismatched} of its {len(feature_index)} rows name a different osmid than "
            f"the feature its own edge_feature_index points at. Measured on 17 September "
            f"2026: 160, 162 and 94. Regenerate the product against the current geometry "
            f"to make this comparison meaningful."
        )
    _model, _drivers, solved = _noodl_leiden(data)
    ours = solved["street.x"].detach().numpy()
    column = {index: k for k, index in enumerate(data.feature_index)}
    dataset = netCDF4.Dataset(str(product))
    try:
        for step, row in enumerate(STEPS):
            theirs = np.array([
                float(dataset.variables["concentration_increment"][row, k])
                for k, index in enumerate(feature_index)
            ])
            mine = np.array([ours[step, column[index]] for index in feature_index])
            np.testing.assert_allclose(mine, theirs, rtol=1e-6, atol=0)
    finally:
        dataset.close()
