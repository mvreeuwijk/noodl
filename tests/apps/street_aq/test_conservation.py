"""Conservation, junction elimination and gradients — spec section 7, rows 4 to 7 and 11.

These are the physics gate: they use no reference implementation at all, only the model's own
balance and algebra done by hand in the test.
"""

from __future__ import annotations

import math

import torch

from noodl.apps.street_aq.canyon import (
    KAPPA_IMPAQ,
    boundary_layer,
    canyon_velocity,
    exchange_velocity,
)
from noodl.apps.street_aq.network import (
    Street,
    StreetNetwork,
    build_model,
    from_test_network,
)
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

DT = torch.float64
U_REF, THETA_W, H_ABL, BACKGROUND = 2.0, 0.25 * math.pi, 1200.0, 1.0e-4


def _drivers(model, emission=(1.0, 2.0, 3.0), *, u_ref=U_REF, theta_w=THETA_W,
             h_abl=H_ABL, background=BACKGROUND):
    net = model.net
    sources = torch.zeros(net.n, dtype=DT)
    for name, value in zip(("r1", "r2", "r3"), emission, strict=True):
        sources[net.node_index(name)] = value
    return {
        "street.x_boundary": torch.tensor([background], dtype=DT),
        "street.sources": sources,
        "U_ref": torch.tensor(u_ref, dtype=DT),
        "theta_w": torch.tensor(theta_w, dtype=DT),
        "h_abl": torch.tensor(h_abl, dtype=DT),
    }


def _atmosphere_balance(model, state, drivers):
    """Net mass per second leaving the network through the atmosphere node.

    Every `vent` and `exchange` edge carries the concentration of its UPSTREAM node, which
    is the street on an outgoing edge and the atmosphere on an incoming one; `route` edges
    join two streets and cancel out of the total.
    """
    net = model.net
    layer = model.transport["street"]
    resolved = model._apply_closures(state, drivers)
    q = model._kind_flows("street", state, resolved)
    x = torch.zeros(net.n, dtype=DT)
    x[layer.interior_idx] = state["street.x"]
    x[layer.boundary_idx] = drivers["street.x_boundary"]
    edges = net.edges
    total = torch.zeros((), dtype=DT)
    offset = 0
    for kind in layer.flow_kinds:
        columns = net.edge_index(kind).tolist()
        for k, column in enumerate(columns):
            source, target, _key = edges[column]
            if target == "atmosphere":
                total = total + q[offset + k] * x[net.node_index(source)]
            elif source == "atmosphere":
                total = total - q[offset + k] * x[net.node_index(source)]
        offset += len(columns)
    return total


def test_the_steady_state_has_a_zero_nodal_balance_at_every_street():
    model, state, _ = build_model(from_test_network(), pblh_floor=False)
    drivers = _drivers(model)
    solved = model.steady(state, drivers)
    residual = model.residuals(solved, drivers)["street"]
    layer = model.transport["street"]
    scale = float(
        (drivers["street.sources"][layer.interior_idx] / layer.capacity).abs().max()
    )
    # The residual is dC/dt; the natural scale is the emission's own contribution to it,
    # so this is a RELATIVE statement. Measured ratio: 2.1e-16.
    assert float(residual.abs().max()) <= 1e-10 * scale


def test_everything_emitted_leaves_through_the_atmosphere():
    """Steady state: every kilogram emitted crosses the atmosphere boundary.

    These three are conservation IDENTITIES of the advection operator, not
    misattribution checks: a flow written on the WRONG edge still balances, so the
    reference for that is `test_junction_elimination_equals_the_hand_written_dense_system`.
    """
    model, state, _ = build_model(from_test_network(), pblh_floor=False)
    drivers = _drivers(model)
    solved = model.steady(state, drivers)
    emitted = drivers["street.sources"].sum()
    torch.testing.assert_close(
        _atmosphere_balance(model, solved, drivers), emitted, rtol=1e-12, atol=0
    )


def test_the_balance_closes_with_a_zero_background_too():
    """The same balance with no incoming background, which removes the compensating inflow.

    These three are conservation IDENTITIES of the advection operator, not
    misattribution checks: a flow written on the WRONG edge still balances, so the
    reference for that is `test_junction_elimination_equals_the_hand_written_dense_system`.
    """
    model, state, _ = build_model(from_test_network(), pblh_floor=False)
    drivers = _drivers(model, background=0.0)
    solved = model.steady(state, drivers)
    torch.testing.assert_close(
        _atmosphere_balance(model, solved, drivers),
        drivers["street.sources"].sum(), rtol=1e-12, atol=0,
    )


def _hand_reference(sn, *, emission, background, u_ref=U_REF, theta_w=THETA_W,
                    h_abl=H_ABL):
    """The eliminated network written out by hand, as a dense linear system.

    Nothing here touches `routing.py`: the junction algebra -- classify by the sign of the
    flux, close the imbalance onto one side, mix in proportion to the outflow shares -- is
    spelled out again, so agreement is evidence about the elimination and not a tautology.
    """
    streets = sn.streets
    n = len(streets)
    width = torch.tensor([s.width for s in streets], dtype=DT)
    height = torch.tensor([s.height for s in streets], dtype=DT)
    length = torch.tensor([s.length for s in streets], dtype=DT)
    azimuth = torch.tensor(sn.azimuth, dtype=DT)
    layer = boundary_layer(height.mean(), torch.tensor(u_ref, dtype=DT),
                           torch.tensor(h_abl, dtype=DT), z_ref=30.0, kappa=KAPPA_IMPAQ)
    u = canyon_velocity(width, height, torch.tensor(theta_w, dtype=DT) - azimuth,
                        u_star=layer.u_star, form="soulhac", kappa=KAPPA_IMPAQ)
    sigma_w = 1.3 * layer.u_star * (1.0 - 0.8 * height / layer.h_abl)
    u_d = exchange_velocity(sigma_w, height, width, form="sirane")
    flux = u * width * height
    a = torch.zeros(n, n, dtype=DT)
    b = torch.zeros(n, dtype=DT)
    for i in range(n):
        a[i, i] -= u_d[i] * width[i] * length[i]
        b[i] += u_d[i] * width[i] * length[i] * background + emission[i]
    for junction in sn.junctions:
        members, signed = [], []
        for i, street in enumerate(streets):
            if street.u == junction:
                members.append(i)
                signed.append(-flux[i])
            elif street.v == junction:
                members.append(i)
                signed.append(flux[i])
        signed = torch.stack(signed)
        p_in = torch.clamp(signed, min=0.0)
        p_out = torch.clamp(-signed, min=0.0)
        imbalance = p_in.sum() - p_out.sum()
        to_atmosphere = torch.zeros_like(p_in)
        from_atmosphere = torch.zeros_like(p_out)
        if float(imbalance) > 0:
            to_atmosphere = imbalance / p_in.sum() * p_in
            p_in = p_in - to_atmosphere
        elif float(imbalance) < 0:
            from_atmosphere = -imbalance / p_out.sum() * p_out
            p_out = p_out - from_atmosphere
        total_out = p_out.sum()
        for k, i in enumerate(members):
            a[i, i] -= to_atmosphere[k] + p_in[k]
            b[i] += from_atmosphere[k] * background
            if float(total_out) > 0:
                for m, j in enumerate(members):
                    a[j, i] += p_in[k] * p_out[m] / total_out
    return torch.linalg.solve(a, -b)


def test_junction_elimination_equals_the_hand_written_dense_system():
    sn = from_test_network()
    model, state, _ = build_model(sn, routing="mixing", pblh_floor=False)
    emission = (1.0, 2.0, 3.0)
    solved = model.steady(state, _drivers(model, emission))
    hand = _hand_reference(sn, emission=torch.tensor(emission, dtype=DT),
                           background=BACKGROUND)
    torch.testing.assert_close(solved["street.x"], hand, rtol=1e-12, atol=0)


def test_the_exchange_edge_pair_is_exactly_the_conduction_term():
    """The deviation recorded in the plan's Conventions, pinned.

    A two-way pair of `exchange` edges carrying `u_d W L` gives the same answer as a
    `TransportLayer` built with `conduction_kind` and that same conductance -- which is why
    the roof exchange may be driver-prescribed rather than fixed at construction.
    """
    capacity = torch.tensor([100.0 * 20.0 * 20.0, 120.0 * 20.0 * 20.0], dtype=DT)
    conductance = torch.tensor([0.5 * 20.0 * 100.0, 0.4 * 20.0 * 120.0], dtype=DT)
    route_and_vent = torch.tensor([200.0, 0.0, 0.0, 200.0, 0.0, 0.0], dtype=DT)
    boundary = torch.tensor([3.0e-8], dtype=DT)
    results = []
    for as_flow in (False, True):
        net = Network(dtype=DT)
        for name in ("atmosphere", "s0", "s1"):
            net.add_node(name)
        net.add_edge("s0", "s1", kind="route")
        net.add_edge("s1", "s0", kind="route")
        net.add_edge("s0", "atmosphere", kind="vent")
        net.add_edge("s1", "atmosphere", kind="vent")
        net.add_edge("atmosphere", "s0", kind="vent")
        net.add_edge("atmosphere", "s1", kind="vent")
        net.add_edge("s0", "atmosphere", kind="exchange")
        net.add_edge("s1", "atmosphere", kind="exchange")
        if as_flow:
            net.add_edge("atmosphere", "s0", kind="exchange")
            net.add_edge("atmosphere", "s1", kind="exchange")
            layer = TransportLayer(net, "street", capacity=capacity,
                                   flow_kind=("route", "vent", "exchange"),
                                   boundary=["atmosphere"], scheme="implicit")
            q = torch.cat([route_and_vent, conductance, conductance])
        else:
            layer = TransportLayer(net, "street", capacity=capacity,
                                   flow_kind=("route", "vent"), boundary=["atmosphere"],
                                   conduction_kind="exchange", conductance=conductance,
                                   scheme="implicit")
            q = route_and_vent
        sources = torch.zeros(net.n, dtype=DT)
        sources[net.node_index("s0")] = 1.0e-3
        results.append(layer.steady(q, sources, boundary))
    torch.testing.assert_close(results[0], results[1], rtol=1e-12, atol=0)


def test_gaussian_averaging_collapses_to_a_single_sample_at_zero_spread():
    sn = from_test_network()
    plain, state, _ = build_model(sn, direction_averaging="none", pblh_floor=False)
    reference = plain.steady(state, _drivers(plain))["street.x"]
    for spread in (0.0, 1.0e-9):
        model, state_g, _ = build_model(
            sn, direction_averaging="gauss", n_theta=5, sigma_theta=spread,
            pblh_floor=False,
        )
        got = model.steady(state_g, _drivers(model))["street.x"]
        # Exactly equal: the Gauss-Hermite weights are normalised and every offset is
        # zero at zero spread, so the closure evaluates the identical flux matrix.
        torch.testing.assert_close(got, reference, rtol=0, atol=0)


def test_every_street_s_own_flux_is_fully_accounted_for_at_the_junction_it_enters():
    """Routing rows sum to one: the flux a street delivers to a junction leaves again,
    either into the other streets or through that junction's roof."""
    model, state, _ = build_model(from_test_network(), routing="sirane",
                                         pblh_floor=False)
    drivers = _drivers(model)
    resolved = model._apply_closures(state, drivers)
    q = model._kind_flows("street", state, resolved)
    u_canyon = resolved["street.u_canyon"]
    net = model.net
    layer = model.transport["street"]
    edges = net.edges
    delivered = {name: 0.0 for name in ("r1", "r2", "r3")}
    offset = 0
    for kind in layer.flow_kinds:
        columns = net.edge_index(kind).tolist()
        for k, column in enumerate(columns):
            source, target, key = edges[column]
            data = net.graph.edges[source, target, key]
            if kind == "route":
                delivered[source] += float(q[offset + k])
            elif kind == "vent" and data["direction"] == "out":
                delivered[source] += float(q[offset + k])
        offset += len(columns)
    geometry = {s.name: (s.width, s.height) for s in from_test_network().streets}
    for i, name in enumerate(("r1", "r2", "r3")):
        width, height = geometry[name]
        flux = abs(float(u_canyon[i])) * width * height
        assert abs(delivered[name] - flux) <= 1e-12 * flux


def test_gradients_match_central_differences_through_the_whole_model():
    """Spec section 7's gradient row, on emissions, background, wind speed and ABL height.

    Central differences are the limit here, not autograd: the measured disagreements are
    4e-10 (emissions), 2.7e-10 (`U_ref`), 1.8e-9 (background) and 1.3e-8 (`h_abl`, the
    weakest sensitivity of the four), so 1e-6 is a tolerance the finite differences can
    support rather than one autograd needs.
    """
    model, state, _ = build_model(from_test_network(), pblh_floor=False)
    for key in ("U_ref", "h_abl", "street.x_boundary"):
        drivers = _drivers(model)
        leaf = drivers[key].clone().requires_grad_(True)
        drivers[key] = leaf
        (grad,) = torch.autograd.grad(
            model.steady(state, drivers)["street.x"].sum(), leaf
        )
        step = 1e-6 * float(leaf.detach().abs().max())
        plus, minus = _drivers(model), _drivers(model)
        plus[key] = plus[key] + step
        minus[key] = minus[key] - step
        central = (
            model.steady(state, plus)["street.x"].sum()
            - model.steady(state, minus)["street.x"].sum()
        ) / (2.0 * step)
        assert abs(float(grad.sum()) / float(central) - 1.0) < 1e-6
    drivers = _drivers(model)
    leaf = drivers["street.sources"].clone().requires_grad_(True)
    drivers["street.sources"] = leaf
    (grad,) = torch.autograd.grad(model.steady(state, drivers)["street.x"].sum(), leaf)
    row = model.net.node_index("r1")
    step = 1e-6
    plus, minus = _drivers(model), _drivers(model)
    plus["street.sources"] = plus["street.sources"].clone()
    minus["street.sources"] = minus["street.sources"].clone()
    plus["street.sources"][row] += step
    minus["street.sources"][row] -= step
    central = (
        model.steady(state, plus)["street.x"].sum()
        - model.steady(state, minus)["street.x"].sum()
    ) / (2.0 * step)
    assert abs(float(grad[row]) / float(central) - 1.0) < 1e-6


def test_a_two_street_dead_end_pair_conserves_mass_in_both_wind_directions():
    """A three-street T with two dead ends, balanced at four wind directions.

    These three are conservation IDENTITIES of the advection operator, not
    misattribution checks: a flow written on the WRONG edge still balances, so the
    reference for that is `test_junction_elimination_equals_the_hand_written_dense_system`.
    """
    sn = StreetNetwork(
        streets=[Street("r1", "a", "b", 100.0, 20.0, 20.0),
                 Street("r2", "b", "c", 120.0, 20.0, 20.0),
                 Street("r3", "b", "d", 140.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 220.0, "d": 100.0},
        y={"a": 0.0, "b": 0.0, "c": 0.0, "d": 140.0},
    )
    model, state, _ = build_model(sn, routing="sirane", pblh_floor=False)
    for theta in (0.0, 0.5 * math.pi, math.pi, 1.3 * math.pi):
        drivers = _drivers(model, (1.0, 2.0, 3.0), theta_w=theta, background=0.0)
        solved = model.steady(state, drivers)
        torch.testing.assert_close(
            _atmosphere_balance(model, solved, drivers),
            drivers["street.sources"].sum(), rtol=1e-12, atol=0,
        )
        assert bool((solved["street.x"] >= 0).all())
