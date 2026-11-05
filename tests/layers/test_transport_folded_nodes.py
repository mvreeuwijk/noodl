"""Storage-free nodes joined only by conduction (a wall's surface node, a zone's radiant star
node) are folded into their neighbours: the star-delta / combined-surface-coefficient idea.
The layer steps the remaining nodes with equivalent conductances and folded sources, so every
scheme (including `exact`) is available, and recovers the folded nodes' values afterwards.

The references here are independent of the layer: a dense elimination over the whole network
written out in the test (`_dense_reference`) and closed forms for a wall."""

from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import gradcheck

from noodl.apps.building_physics.thermal import thermal_layer
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
SCHEMES = ["exact", "implicit", "trapezoidal"]


def _t(values, **kwargs):
    return torch.tensor(values, dtype=F64, **kwargs)


# ---------------------------------------------------------------- a wall
# ambient (boundary) | zone (stored) - s1 (surface) - w (wall mass) - s2 (surface) - ambient
H1, K1, K2, H2 = 3.0, 8.0, 6.0, 12.0          # W/K: film, wall half, wall half, film
CZ, CW = 5.0e4, 2.0e5                          # J/K
T_AMB = 270.0


def _wall_net():
    net = Network(dtype=F64)
    for n in ("ambient", "zone", "s1", "w", "s2"):
        net.add_node(n)
    net.add_edge("zone", "ambient", kind="air")
    net.add_edge("zone", "s1", kind="film")
    net.add_edge("s2", "ambient", kind="film")
    net.add_edge("s1", "w", kind="wall")
    net.add_edge("w", "s2", kind="wall")
    return net


def _wall(scheme="exact", g=(H1, H2, K1, K2), **kw):
    """Interior order: zone, s1, w, s2. Conductance order: film (zone-s1, s2-ambient), then
    wall (s1-w, w-s2)."""
    return TransportLayer(
        _wall_net(), "T", capacity=_t([CZ, 0.0, CW, 0.0]), flow_kind="air",
        boundary=["ambient"], conduction_kind=("film", "wall"), conductance=_t(list(g)),
        storage_free=["s1", "s2"], scheme=scheme, **kw,
    )


def _two_node_wall(scheme="exact"):
    """The same wall with the surfaces folded by hand: zone - w by 1/(1/h1 + 1/k1), w -
    ambient by 1/(1/k2 + 1/h2)."""
    net = Network(dtype=F64)
    for n in ("ambient", "zone", "w"):
        net.add_node(n)
    net.add_edge("zone", "ambient", kind="air")
    net.add_edge("zone", "w", kind="wall")
    net.add_edge("w", "ambient", kind="wall")
    u_in = 1.0 / (1.0 / H1 + 1.0 / K1)
    u_out = 1.0 / (1.0 / K2 + 1.0 / H2)
    return TransportLayer(
        net, "T", capacity=_t([CZ, CW]), flow_kind="air", boundary=["ambient"],
        conduction_kind="wall", conductance=_t([u_in, u_out]), scheme=scheme,
    )


X0_WALL = _t([295.0, 290.0, 285.0, 275.0])
Q0 = _t([0.0])
XB = _t([T_AMB])
SRC0 = torch.zeros(5, dtype=F64)


def test_exact_is_allowed_and_the_surfaces_are_folded():
    layer = _wall("exact")
    assert layer.storage_free_mask.tolist() == [False, True, False, True]
    assert layer.folded_mask.tolist() == [False, True, False, True]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_folded_wall_steps_like_the_hand_folded_two_node_wall(scheme):
    """The stored nodes follow the wall whose surface films are combined in series by hand,
    to rounding, under every scheme (the folding is exact, not an approximation)."""
    x = _wall(scheme).step(X0_WALL, Q0, SRC0, XB, 600.0)
    ref = _two_node_wall(scheme).step(X0_WALL[[0, 2]], Q0, torch.zeros(3, dtype=F64), XB,
                                      600.0)
    torch.testing.assert_close(x[[0, 2]], ref, rtol=1e-13, atol=0)


def test_the_surface_temperatures_are_recovered_from_their_neighbours():
    """A surface between two nodes takes the conductance-weighted mean of their end-of-step
    values (no heat stored at the surface)."""
    x = _wall("exact").step(X0_WALL, Q0, SRC0, XB, 600.0)
    zone, w = x[0], x[2]
    torch.testing.assert_close(x[1], (H1 * zone + K1 * w) / (H1 + K1), rtol=1e-14, atol=0)
    torch.testing.assert_close(x[3], (K2 * w + H2 * T_AMB) / (K2 + H2), rtol=1e-14, atol=0)


def test_exact_agrees_with_implicit_at_small_steps():
    """Backward Euler is first order: its error at step h is about twice the difference
    between its runs at h and h/2 (Richardson). The exact step over the same hour must fall
    within that error estimate of the implicit run, and close to the extrapolated value."""
    exact = _wall("exact").step(X0_WALL, Q0, SRC0, XB, 3600.0)

    def implicit(n):
        layer, x = _wall("implicit"), X0_WALL
        for _ in range(n):
            x = layer.step(x, Q0, SRC0, XB, 3600.0 / n)
        return x

    coarse, fine = implicit(200), implicit(400)
    error = 2.0 * (coarse - fine).abs()
    assert bool(((exact - coarse).abs() <= 1.1 * error + 1e-12).all())
    extrapolated = 2.0 * fine - coarse
    assert bool(((exact - extrapolated).abs() <= 0.05 * error + 1e-12).all())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_steady_wall_is_the_series_resistance_exactly(scheme):
    """With a heater P in the zone, all of P leaves through the series films and wall: the
    steady state and a long step both match the closed form."""
    P = 150.0
    src = _t([0.0, P, 0.0, 0.0, 0.0])
    r1, rk1, rk2, r2 = 1 / H1, 1 / K1, 1 / K2, 1 / H2
    zone = T_AMB + P * (r1 + rk1 + rk2 + r2)
    expected = _t([zone, zone - P * r1, zone - P * (r1 + rk1), zone - P * (r1 + rk1 + rk2)])
    layer = _wall(scheme)
    torch.testing.assert_close(layer.steady(Q0, src, XB), expected, rtol=1e-13, atol=0)
    if scheme == "exact":
        x = layer.step(X0_WALL, Q0, src, XB, 1.0e7)
        torch.testing.assert_close(x, expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- a dense reference
def _dense_reference(net, layer, g, cap, sources, xb, x0, dt):
    """The folded exact step written out over the whole network, densely: the conduction
    Laplacian `K`, the folded nodes' own block eliminated (`K_kk - K_ke K_ee^-1 K_ek`), the
    stored nodes stepped by the matrix exponential of the affine system, then
    `x_e = K_ee^-1 (s_e - K_ek x_k - K_eb x_b)`. Conduction only (zero flows)."""
    n = net.n
    Kfull = torch.zeros(n, n, dtype=F64)
    e = 0
    for _kind, src, tgt in layer.conduction_edges:
        for s, t in zip(src.tolist(), tgt.tolist(), strict=True):
            Kfull[s, s] += g[e]
            Kfull[t, t] += g[e]
            Kfull[s, t] -= g[e]
            Kfull[t, s] -= g[e]
            e += 1
    interior = layer.interior_idx
    folded = interior[layer.folded_mask]
    kept = interior[~layer.folded_mask]
    bnd = layer.boundary_idx
    Kee = Kfull[folded][:, folded]
    Kek = Kfull[folded][:, kept]
    Keb = Kfull[folded][:, bnd]
    Kke = Kfull[kept][:, folded]
    Kred = Kfull[kept][:, kept] - Kke @ torch.linalg.solve(Kee, Kek)
    Kredb = Kfull[kept][:, bnd] - Kke @ torch.linalg.solve(Kee, Keb)
    s_k = sources[kept] - Kke @ torch.linalg.solve(Kee, sources[folded])
    c = cap[~layer.folded_mask]
    A = -Kred / c[:, None]
    b = (-Kredb @ xb + s_k) / c
    m = A.shape[0]
    aug = torch.zeros(m + 1, m + 1, dtype=F64)
    aug[:m, :m] = A * dt
    aug[:m, m] = b * dt
    xk0 = x0[~layer.folded_mask]
    xk = (torch.linalg.matrix_exp(aug) @ torch.cat([xk0, _t([1.0])]))[:m]
    xe = torch.linalg.solve(Kee, sources[folded] - Kek @ xk - Keb @ xb)
    out = torch.zeros(layer.n_i, dtype=F64)
    out[~layer.folded_mask] = xk
    out[layer.folded_mask] = xe
    return out


# ---------------------------------------------------------------- a radiant star node
# zone air z; three interior surfaces s1..s3, each with a film to z, a radiative edge to the
# star node r and wall conduction to its own wall mass w1..w3, which conducts to ambient.
H_FILM = [2.5, 4.0, 1.5]
H_RAD = [5.0, 3.0, 6.5]
K_WALL = [9.0, 7.0, 11.0]
U_OUT = [1.2, 0.8, 2.0]


def _star_net():
    net = Network(dtype=F64)
    for n in ("ambient", "z", "r", "s1", "s2", "s3", "w1", "w2", "w3"):
        net.add_node(n)
    net.add_edge("z", "ambient", kind="air")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", "z", kind="film")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", "r", kind="rad")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", f"w{i}", kind="wall")
    for i in (1, 2, 3):
        net.add_edge(f"w{i}", "ambient", kind="wall")
    return net


STAR_G = H_FILM + H_RAD + K_WALL + U_OUT
# interior order: z, r, s1, s2, s3, w1, w2, w3
STAR_CAP = [6.0e4, 0.0, 0.0, 0.0, 0.0, 1.0e5, 2.0e5, 1.5e5]
STAR_X0 = _t([294.0, 291.0, 292.0, 290.0, 293.0, 288.0, 284.0, 286.0])


def _star(scheme="exact", g=STAR_G):
    net = _star_net()
    layer = TransportLayer(
        net, "T", capacity=_t(STAR_CAP), flow_kind="air", boundary=["ambient"],
        conduction_kind=("film", "rad", "wall"), conductance=_t(list(g)),
        storage_free=["r", "s1", "s2", "s3"], scheme=scheme,
    )
    return net, layer


def _star_sources(solar=(40.0, 0.0, 25.0), radiant=30.0, convective=100.0):
    """Full node order: ambient, z, r, s1, s2, s3, w1, w2, w3."""
    return _t([0.0, convective, radiant, *solar, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("with_sources", [False, True])
def test_a_star_node_with_three_surfaces_matches_the_dense_elimination(with_sources):
    net, layer = _star("exact")
    assert layer.folded_mask.tolist() == [False, True, True, True, True, False, False, False]
    src = _star_sources() if with_sources else torch.zeros(9, dtype=F64)
    x = layer.step(STAR_X0, Q0, src, XB, 900.0)
    ref = _dense_reference(net, layer, _t(STAR_G), _t(STAR_CAP), src, XB, STAR_X0, 900.0)
    torch.testing.assert_close(x, ref, rtol=1e-12, atol=0)


def test_sources_on_folded_nodes_reach_the_stored_nodes_and_conserve_energy():
    """Absorbed solar on the surfaces and a radiant gain at the star node: what the stored
    nodes gained plus what reached the ambient equals what the sources put in, under every
    scheme (the boundary transfer counts the heat leaving through the folded nodes)."""
    src = _star_sources()
    dt = 600.0
    stored = _t(STAR_CAP)
    for scheme in SCHEMES:
        _, layer = _star(scheme)
        out = layer.step_with_transfer(STAR_X0, Q0, src, XB, dt)
        gained = (stored * (out.x - STAR_X0)).sum()
        torch.testing.assert_close(gained + out.boundary_transfer.sum(), dt * src.sum(),
                                   rtol=1e-11, atol=1e-6)


def test_the_boundary_transfer_counts_a_path_between_two_boundaries():
    """An exterior surface between the wall mass, the ambient air and the sky (two
    boundaries): the heat that passes from the ambient to the sky through the surface is
    reported at both boundaries, as it is without folding (a tiny surface capacity)."""
    def build(folded):
        net = Network(dtype=F64)
        for n in ("ambient", "sky", "w", "s"):
            net.add_node(n)
        net.add_edge("w", "ambient", kind="air")
        net.add_edge("w", "s", kind="wall")
        net.add_edge("s", "ambient", kind="film")
        net.add_edge("s", "sky", kind="rad")
        return TransportLayer(
            net, "T", capacity=_t([1.0e5, 0.0 if folded else 1.0e-6]), flow_kind="air",
            boundary=["ambient", "sky"], conduction_kind=("wall", "film", "rad"),
            conductance=_t([10.0, 15.0, 5.0]), storage_free=["s"] if folded else (),
            scheme="implicit",
        )

    x0 = _t([280.0, 275.0])
    xb = _t([275.0, 250.0])
    src = _t([0.0, 0.0, 0.0, 20.0])          # absorbed solar on the surface
    folded = build(True).step_with_transfer(x0, Q0, src, xb, 60.0)
    stored = build(False).step_with_transfer(x0, Q0, src, xb, 60.0)
    torch.testing.assert_close(folded.boundary_transfer, stored.boundary_transfer,
                               rtol=1e-6, atol=1e-6)
    assert folded.boundary_transfer[1].item() > 0.0     # the sky gains


def test_a_folded_node_held_by_a_zero_conductance_keeps_its_value_and_drops_its_source():
    """Every edge of a folded node is zero this call: it is held at its value and its
    source dropped (counted), as a storage-free node nothing passes through always was."""
    net, layer = _star("exact")
    g = _t(STAR_G)
    g[[0, 3, 6]] = 0.0                       # s1's film, radiation and wall edges
    diagnostics: dict = {}
    src = _star_sources()
    x = layer.step(STAR_X0, Q0, src, XB, 900.0, conductance=g, diagnostics=diagnostics)
    assert x[2].item() == STAR_X0[2].item()
    assert diagnostics["held_sources_dropped"] == 1
    # the rest of the group folds as if s1 were not there
    bare = layer.step(STAR_X0, Q0, torch.cat([src[:3], _t([0.0]), src[4:]]), XB, 900.0,
                      conductance=g)
    torch.testing.assert_close(x, bare, rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match=r"nothing passes through storage-free.*'s1'"):
        layer.steady(Q0, src, XB, conductance=g)


def test_a_folded_group_cut_off_from_every_stored_node_is_refused_by_name():
    """The star and its surfaces joined only to each other this call (every film and wall
    edge zero): they have no value, and the step refuses them by name."""
    _, layer = _star("exact")
    g = _t(STAR_G)
    g[[0, 1, 2, 6, 7, 8]] = 0.0
    with pytest.raises(ValueError, match=r"\['r', 's1', 's2', 's3'\].*only to each other"):
        layer.step(STAR_X0, Q0, _star_sources(), XB, 900.0, conductance=g)


@pytest.mark.parametrize("call", ["step", "steady"])
def test_a_negative_per_call_conductance_on_a_folded_layer_is_refused_by_name(call):
    """The sign check a folded layer reads together with the folding's zero check still
    names the edge, on any batch instance, before anything is folded."""
    _, layer = _star("exact")
    g = torch.stack([_t(STAR_G), _t(STAR_G)])
    g[1, 4] = -1.0                                    # s2-r radiation, second instance
    with pytest.raises(ValueError, match=r"zero or positive; negative on conduction "
                                         r"edge\(s\) \[\('s2', 'r'\)\]"):
        if call == "step":
            layer.step(STAR_X0, Q0, _star_sources(), XB, 900.0, conductance=g)
        else:
            layer.steady(Q0, _star_sources(), XB, conductance=g)


def test_a_folded_group_with_no_neighbour_at_all_is_refused_at_construction():
    net = Network(dtype=F64)
    for n in ("ambient", "z", "a", "b"):
        net.add_node(n)
    net.add_edge("z", "ambient", kind="air")
    net.add_edge("a", "b", kind="wall")
    with pytest.raises(ValueError, match=r"\['a', 'b'\].*only to each other"):
        TransportLayer(net, "T", capacity=_t([1.0, 0.0, 0.0]), flow_kind="air",
                       boundary=["ambient"], conduction_kind="wall", conductance=_t([1.0]),
                       storage_free=["a", "b"])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_layer_of_folded_nodes_only_is_algebraic(scheme):
    """No stored node at all (a heat network without heat capacity): every step and the
    steady state are the folded node's balance at the end of the step, whatever the start
    value; the boundary transfer is that balance's heat over the step."""
    net = Network(dtype=F64)
    for n in ("ambient", "sky", "s"):
        net.add_node(n)
    net.add_edge("ambient", "sky", kind="air")
    net.add_edge("s", "ambient", kind="film")
    net.add_edge("s", "sky", kind="rad")
    layer = TransportLayer(net, "T", capacity=_t([0.0]), flow_kind="air",
                           boundary=["ambient", "sky"], conduction_kind=("film", "rad"),
                           conductance=_t([1.0, 2.0]), storage_free=["s"], scheme=scheme)
    q, xb, src = _t([0.0]), _t([270.0, 250.0]), _t([0.0, 0.0, 3.0])
    balance = (1.0 * 270.0 + 2.0 * 250.0 + 3.0) / 3.0
    assert layer.step(_t([300.0]), q, src, xb, 10.0).tolist() == pytest.approx([balance],
                                                                                 rel=1e-15)
    assert layer.steady(q, src, xb).tolist() == pytest.approx([balance], rel=1e-15)
    out = layer.step_with_transfer(_t([300.0]), q, src, xb, 10.0)
    torch.testing.assert_close(out.boundary_transfer,
                               10.0 * _t([1.0 * (balance - 270.0), 2.0 * (balance - 250.0)]))
    batched = layer.step(_t([[300.0], [200.0]]), q, src, xb, 10.0)
    torch.testing.assert_close(batched, torch.full((2, 1), balance, dtype=F64))


# ---------------------------------------------------------------- the maximum principle
@pytest.mark.parametrize("dt", [1.0, 600.0, 1.0e6])
def test_no_new_extrema_under_exact(dt):
    """Without sources every value after the step, the folded nodes' included, lies within
    the range of the start values and the boundary value."""
    gen = torch.Generator().manual_seed(7)
    for _ in range(5):
        g = 0.1 + 10.0 * torch.rand(12, generator=gen, dtype=F64)
        x0 = 280.0 + 20.0 * torch.rand(8, generator=gen, dtype=F64)
        xb = 280.0 + 20.0 * torch.rand(1, generator=gen, dtype=F64)
        _, layer = _star("exact", g=g.tolist())
        x = layer.step(x0, Q0, torch.zeros(9, dtype=F64), xb, dt)
        lo = min(x0.min().item(), xb.item())
        hi = max(x0.max().item(), xb.item())
        assert x.min().item() >= lo - 1e-9 and x.max().item() <= hi + 1e-9


# ---------------------------------------------------------------- batches and drivers
def test_a_batched_conductance_folds_each_instance_on_its_own():
    _, layer = _star("exact")
    g = torch.stack([_t(STAR_G), 1.7 * _t(STAR_G)])
    g[1, 4] = 0.3
    src = _star_sources()
    batched = layer.step(STAR_X0, Q0, src, XB, 900.0, conductance=g)
    for b in range(2):
        one = layer.step(STAR_X0, Q0, src, XB, 900.0, conductance=g[b])
        torch.testing.assert_close(batched[b], one, rtol=1e-13, atol=0)


def test_the_driver_equal_to_the_construction_conductance_changes_nothing():
    _, layer = _star("exact")
    src = _star_sources()
    a = layer.step(STAR_X0, Q0, src, XB, 900.0)
    b = layer.step(STAR_X0, Q0, src, XB, 900.0, conductance=_t(STAR_G))
    torch.testing.assert_close(a, b, rtol=1e-14, atol=0)


def test_several_species_fold_through_the_same_conductances():
    net = _wall_net()
    layer = TransportLayer(
        net, "T", capacity=_t([CZ, 0.0, CW, 0.0]), flow_kind="air", boundary=["ambient"],
        conduction_kind=("film", "wall"), conductance=_t([H1, H2, K1, K2]),
        storage_free=["s1", "s2"], n_species=2,
    )
    x0 = torch.stack([X0_WALL, X0_WALL - 10.0], -1)
    src = torch.zeros(5, 2, dtype=F64)
    src[2, 1] = 50.0
    x = layer.step(x0, Q0, src, torch.stack([XB, XB + 5.0], -1), 600.0)
    one = _wall("exact")
    torch.testing.assert_close(x[:, 0], one.step(X0_WALL, Q0, SRC0, XB, 600.0), rtol=1e-13,
                               atol=0)
    torch.testing.assert_close(x[:, 1], one.step(X0_WALL - 10.0, Q0, src[:, 1], XB + 5.0,
                                                 600.0), rtol=1e-13, atol=0)


@pytest.mark.parametrize("scheme", [*SCHEMES, "steady"])
def test_gradcheck_through_the_folding(scheme):
    """Gradients reach the conductances (a per-call driver), the sources (on a folded node
    and a stored one) and the start state, through the folding and the recovery."""
    _, layer = _star("exact" if scheme == "steady" else scheme)
    g = _t(STAR_G, requires_grad=True)
    src = _star_sources()[1:].requires_grad_(True)       # the interior rows
    x0 = STAR_X0.clone().requires_grad_(True)

    def run(g_, src_, x0_):
        src_ = torch.cat([_t([0.0]), src_])
        if scheme == "steady":
            return layer.steady(Q0, src_, XB, conductance=g_)
        return layer.step(x0_, Q0, src_, XB, 300.0, conductance=g_)

    assert gradcheck(run, (g, src, x0), eps=1e-6, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_transfer_gradient_with_respect_to_the_stored_capacities(scheme):
    """d(boundary transfer)/d(capacity of the stored nodes) through the folding, against a
    fourth-order central difference with steps relative to each capacity. (`gradcheck`'s
    absolute step of 1e-6 on capacities of 1e5 J/K moves a transfer of 1e5 J by about
    1e-10 J, at its rounding level, so it reports a mismatch under every scheme, folded or
    not, that is the difference quotient's own error.)"""
    _, layer = _star(scheme)
    src = _star_sources()

    def transfer(c):
        cap = torch.cat([c[:1], torch.zeros(4, dtype=F64), c[1:]])
        out = layer.step_with_transfer(STAR_X0, _t([0.2]), src, XB, 900.0, capacity=cap)
        return out.boundary_transfer.sum()

    c0 = _t([STAR_CAP[0], *STAR_CAP[5:]])
    c = c0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(transfer(c), c)
    numeric = torch.empty_like(c0)
    for i in range(c0.numel()):
        e = torch.zeros_like(c0)
        e[i] = 1e-3 * c0[i]
        numeric[i] = (-transfer(c0 + 2 * e) + 8 * transfer(c0 + e) - 8 * transfer(c0 - e)
                      + transfer(c0 - 2 * e)) / (12 * e[i])
    torch.testing.assert_close(grad, numeric, rtol=1e-6, atol=1e-6 * grad.abs().max().item())


def test_the_model_transfer_gradient_with_respect_to_a_capacity_driver():
    """The same gradient through `Model.step`: the stored capacities as the driver
    `"T.capacity"` (and the step-start state's), the transfer from
    `boundary_transfers`, under `exact`; the state's gradient too."""
    from noodl.model import Model

    net, layer = _star("exact")
    model = Model(net, {"T": layer})

    def outputs(c):
        cap = torch.cat([c[:1], torch.zeros(4, dtype=F64), c[1:]])
        drivers = {"T.x_boundary": XB, "T.q": _t([0.2]), "T.sources": _star_sources(),
                   "T.capacity": cap}
        diagnostics: dict = {}
        new = model.step({"T.x": STAR_X0, "T.capacity": cap}, drivers, 900.0,
                         diagnostics=diagnostics, boundary_transfers={"T"})
        return diagnostics["layers"]["T"]["boundary_transfer"].sum(), new["T.x"].sum()

    c0 = _t([STAR_CAP[0], *STAR_CAP[5:]])
    c = c0.clone().requires_grad_(True)
    for k, value in enumerate(outputs(c)):
        (grad,) = torch.autograd.grad(value, c, retain_graph=True)
        numeric = torch.empty_like(c0)
        for i in range(c0.numel()):
            e = torch.zeros_like(c0)
            e[i] = 1e-3 * c0[i]
            f = [outputs(c0 + s * e)[k] for s in (2, 1, -1, -2)]
            numeric[i] = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * e[i])
        torch.testing.assert_close(grad, numeric, rtol=1e-6,
                                   atol=1e-6 * grad.abs().max().item())


def test_gradient_reaches_a_construction_conductance_that_requires_grad():
    g = _t([H1, H2, K1, K2], requires_grad=True)
    layer = TransportLayer(
        _wall_net(), "T", capacity=_t([CZ, 0.0, CW, 0.0]), flow_kind="air",
        boundary=["ambient"], conduction_kind=("film", "wall"), conductance=g,
        storage_free=["s1", "s2"],
    )
    x = layer.step(X0_WALL, Q0, SRC0, XB, 600.0)
    x.sum().backward()
    assert g.grad is not None and bool((g.grad != 0).all())


# ---------------------------------------------------------------- what is not folded
def _plenum_and_surface(scheme, **kw):
    net = Network(dtype=F64)
    for n in ("ambient", "plenum", "room", "s"):
        net.add_node(n)
    net.add_edge("ambient", "plenum", kind="air")
    net.add_edge("plenum", "room", kind="air")
    net.add_edge("room", "ambient", kind="air")
    net.add_edge("room", "s", kind="film")
    net.add_edge("s", "ambient", kind="wall")
    return TransportLayer(
        net, "T", capacity=_t([0.0, 1.0e5, 0.0]), flow_kind="air", boundary=["ambient"],
        conduction_kind=("film", "wall"), conductance=_t([3.0, 1.0]),
        storage_free=["plenum", "s"], scheme=scheme, **kw,
    )


@pytest.mark.parametrize("scheme", ["exact", "trapezoidal"])
def test_a_plenum_with_throughflow_still_needs_implicit(scheme):
    with pytest.raises(ValueError, match=r"storage-free nodes \['plenum'\].*implicit"):
        _plenum_and_surface(scheme)


def test_a_plenum_and_a_surface_in_one_implicit_layer():
    """The surface is folded, the plenum keeps its algebraic mixing row."""
    layer = _plenum_and_surface("implicit")
    assert layer.folded_mask.tolist() == [False, False, True]
    q = _t([0.1, 0.1, 0.1])
    x0 = _t([280.0, 295.0, 285.0])
    x = layer.step(x0, q, torch.zeros(4, dtype=F64), XB, 300.0)
    assert x[0].item() == pytest.approx(T_AMB, rel=1e-13)          # the plenum mixes ambient
    torch.testing.assert_close(x[2], (3.0 * x[1] + 1.0 * T_AMB) / 4.0, rtol=1e-13, atol=0)


@pytest.mark.parametrize("term", ["removal", "kinetics"])
def test_a_storage_free_node_with_a_removal_or_reaction_is_not_folded(term):
    kw = ({"removal": _t([[0.0], [1e-3], [0.0], [0.0]])} if term == "removal"
          else {"kinetics": _t([[[0.0]], [[0.0]], [[0.0]], [[-1e-3]]])})
    layer = _wall("implicit", **kw)
    assert layer.folded_mask.tolist() == [False, term == "kinetics", False, term == "removal"]
    with pytest.raises(ValueError, match=r"storage-free nodes \['s(1|2)'\].*implicit"):
        _wall("exact", **kw)


@pytest.mark.parametrize("call", ["step", "step_with_transfer", "steady"])
def test_a_per_call_removal_at_a_folded_node_is_refused_by_name(call):
    """A folded node has no removal term (one given at construction keeps it unfolded); a
    per-call removal there would be dropped, so it is refused by name. A removal at the
    stored nodes only is taken as before."""
    layer = _wall("implicit")
    removal = _t([1e-4, 0.0, 2e-5, 0.0])

    def run(r):
        if call == "steady":
            return layer.steady(Q0, SRC0, XB, removal=r)
        return getattr(layer, call)(X0_WALL, Q0, SRC0, XB, 600.0, removal=r)

    run(removal)
    bad = torch.stack([removal, _t([1e-4, 0.0, 2e-5, 3e-4])])     # a batch, s2 on one
    with pytest.raises(ValueError, match=r"'T'.*removal.*folded.*\['s2'\]"):
        run(bad)


def test_a_layer_without_storage_free_nodes_has_no_folding():
    layer = _two_node_wall()
    assert layer.folded_mask.tolist() == [False, False]


def test_the_thermal_layer_keeps_exact_when_its_storage_free_nodes_only_conduct():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("room", volume=100.0)
    net.add_node("s", volume=0.0)
    net.add_edge("room", "ambient", kind="airpath", ua=0.0)
    net.add_edge("room", "s", kind="wall", ua=3.0)
    net.add_edge("s", "ambient", kind="wall", ua=1.0)
    layer = thermal_layer(net, storage_free=["s"])
    assert layer.scheme == "exact"
    assert layer.folded_mask.tolist() == [False, True]


def test_the_folded_step_is_the_limit_of_a_vanishing_surface_capacity():
    """An independent check of the recovery: a surface node given a tiny heat capacity
    (stored, not folded) tends to the folded surface under the implicit scheme."""
    def build(cap_s):
        return TransportLayer(
            _wall_net(), "T", capacity=_t([CZ, cap_s, CW, cap_s]), flow_kind="air",
            boundary=["ambient"], conduction_kind=("film", "wall"),
            conductance=_t([H1, H2, K1, K2]),
            storage_free=["s1", "s2"] if cap_s == 0.0 else (), scheme="implicit",
        )

    src = _t([0.0, 50.0, 20.0, 0.0, 10.0])
    folded = build(0.0).step(X0_WALL, Q0, src, XB, 600.0)
    tiny = build(1.0e-6).step(X0_WALL, Q0, src, XB, 600.0)
    assert math.isclose(folded[1].item(), tiny[1].item(), rel_tol=1e-9)
    torch.testing.assert_close(folded, tiny, rtol=1e-9, atol=0)


# ---------------------------------------------------------------- more shapes and paths
def _star_with_exterior_surfaces():
    """The star network with an exterior surface e_i between each wall mass and the
    ambient: groups of one node (e_i) and one group of four (r, s1..s3) side by side."""
    net = Network(dtype=F64)
    for n in ("ambient", "z", "r", "s1", "s2", "s3", "w1", "w2", "w3", "e1", "e2", "e3"):
        net.add_node(n)
    net.add_edge("z", "ambient", kind="air")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", "z", kind="film")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", "r", kind="rad")
    for i in (1, 2, 3):
        net.add_edge(f"s{i}", f"w{i}", kind="wall")
    for i in (1, 2, 3):
        net.add_edge(f"w{i}", f"e{i}", kind="wall")
    for i in (1, 2, 3):
        net.add_edge(f"e{i}", "ambient", kind="film")
    # conductance order: film edges (s-z, then e-ambient), rad, wall (s-w, then w-e)
    g = _t(H_FILM + [15.0, 18.0, 21.0] + H_RAD + K_WALL + [2.0, 3.0, 4.0])
    layer = TransportLayer(
        net, "T", capacity=_t(STAR_CAP + [0.0, 0.0, 0.0]), flow_kind="air",
        boundary=["ambient"], conduction_kind=("film", "rad", "wall"),
        conductance=g, storage_free=["r", "s1", "s2", "s3", "e1", "e2", "e3"],
    )
    assert len(layer._fold_plan.buckets) == 2
    return net, layer, g


def test_groups_of_different_sizes_fold_together():
    net, layer, g = _star_with_exterior_surfaces()
    src = _t([0.0, 100.0, 30.0, 40.0, 0.0, 25.0, 0.0, 0.0, 0.0, 60.0, 0.0, 10.0])
    x0 = torch.cat([STAR_X0, _t([275.0, 276.0, 277.0])])
    x = layer.step(x0, Q0, src, XB, 900.0)
    ref = _dense_reference(net, layer, g, _t(STAR_CAP + [0.0, 0.0, 0.0]), src, XB, x0, 900.0)
    torch.testing.assert_close(x, ref, rtol=1e-12, atol=0)


def test_batched_states_and_sources_with_one_conductance():
    _, layer = _star("exact")
    x0 = torch.stack([STAR_X0, STAR_X0 + 3.0])
    src = torch.stack([_star_sources(), _star_sources(solar=(0.0, 80.0, 5.0))])
    batched = layer.step(x0, Q0, src, XB, 900.0)
    for b in range(2):
        torch.testing.assert_close(batched[b], layer.step(x0[b], Q0, src[b], XB, 900.0),
                                   rtol=1e-13, atol=0)


def test_on_failure_return_gives_the_whole_state():
    _, layer = _star("implicit")
    src = _star_sources()
    result = layer.step(STAR_X0, Q0, src, XB, 900.0, on_failure="return")
    assert bool(result.converged.all())
    torch.testing.assert_close(result.x, layer.step(STAR_X0, Q0, src, XB, 900.0),
                               rtol=1e-13, atol=0)
    steady = layer.steady(Q0, src, XB, on_failure="return")
    torch.testing.assert_close(steady.x, layer.steady(Q0, src, XB), rtol=1e-13, atol=0)


def test_a_ramping_boundary_sets_the_surfaces_at_the_end_of_the_step():
    layer = _wall("exact")
    x = layer.step(X0_WALL, Q0, SRC0, XB, 600.0, x_boundary_end=XB - 10.0)
    torch.testing.assert_close(x[3], (K2 * x[2] + H2 * (T_AMB - 10.0)) / (K2 + H2),
                               rtol=1e-14, atol=0)


# ---------------------------------------------------------------- through a Model
SIGMA, EPS_A, P_HEAT, UA_S, H_S = 5.670374419e-8, 0.9 * 1.5, 120.0, 2.0, 4.0
T_A, T_ENV = 300.0, 250.0


class _SurfaceRadiation:
    """Closure: the surface's radiative coefficient to the environment from its previous
    temperature, written with the other conductances (film, wall, radiation order)."""

    def __call__(self, state, drivers):
        t_s = state["thermal.x"].reshape(-1)[1]
        h_r = SIGMA * EPS_A * (t_s**2 + T_ENV**2) * (t_s + T_ENV)
        return {"thermal.conductance": torch.stack(
            [torch.full_like(h_r, H_S), torch.full_like(h_r, UA_S), h_r])}


def _surface_model(scheme):
    from noodl.model import Model

    net = Network(dtype=F64)
    for n in ("amb", "env", "z", "s"):
        net.add_node(n)
    net.add_edge("z", "amb", kind="flow")
    net.add_edge("z", "s", kind="film")
    net.add_edge("s", "amb", kind="wall")
    net.add_edge("s", "env", kind="rad")
    layer = TransportLayer(
        net, "thermal", capacity=_t([1500.0, 0.0]), flow_kind="flow",
        boundary=["amb", "env"], conduction_kind=("film", "wall", "rad"),
        conductance=_t([H_S, UA_S, 1.0]), storage_free=["s"], scheme=scheme,
    )
    model = Model(net, {"thermal": layer}, closures=[_SurfaceRadiation()],
                  coupling="iterate", iterate_tol={"thermal": 1e-12}, iterate_max=200)
    drivers = {"thermal.x_boundary": _t([T_A, T_ENV]), "thermal.q": _t([0.0]),
               "thermal.sources": _t([0.0, 0.0, P_HEAT, 0.0])}
    return model, drivers


def test_a_folded_surface_with_a_radiative_coefficient_iterated_through_a_model():
    """The zone heater's power crosses the film to the surface, which loses it by
    conduction to the ambient and by radiation (sigma T^4) to the environment. The surface
    is folded under the exact scheme and its coefficient re-evaluated each pass."""
    from scipy.optimize import brentq

    def balance(t):
        return P_HEAT + UA_S * (T_A - t) + SIGMA * EPS_A * (T_ENV**4 - t**4)

    t_s = brentq(balance, 200.0, 600.0, xtol=1e-13, rtol=1e-15)
    model, drivers = _surface_model("exact")
    out = model.steady({"thermal.x": _t([T_A, T_A])}, drivers)
    torch.testing.assert_close(out["thermal.x"], _t([t_s + P_HEAT / H_S, t_s]), rtol=0,
                               atol=1e-7)
    state = {"thermal.x": _t([T_A, T_A])}
    for _ in range(20):                   # 20000 s: some 30 time constants of the zone
        state = model.step(state, drivers, 1000.0)
    torch.testing.assert_close(state["thermal.x"], _t([t_s + P_HEAT / H_S, t_s]), rtol=0,
                               atol=1e-6)


def test_attributes_reassigned_after_construction_reach_the_folded_step():
    """`capacity` (as the Modelica assembly reassigns it) and `carrier` edited on a built
    layer act on the folded step exactly as on the same layer built with them."""
    layer = _wall("exact")
    layer.step(X0_WALL, Q0, SRC0, XB, 600.0)            # the kept layer exists and was used
    layer.capacity = layer.capacity * 100.0
    built = TransportLayer(
        _wall_net(), "T", capacity=_t([100 * CZ, 0.0, 100 * CW, 0.0]), flow_kind="air",
        boundary=["ambient"], conduction_kind=("film", "wall"),
        conductance=_t([H1, H2, K1, K2]), storage_free=["s1", "s2"],
    )
    q = _t([0.05])
    torch.testing.assert_close(layer.step(X0_WALL, q, SRC0, XB, 600.0),
                               built.step(X0_WALL, q, SRC0, XB, 600.0), rtol=1e-14, atol=0)
    layer.carrier = torch.as_tensor(1005.0, dtype=F64)
    built.carrier = torch.as_tensor(1005.0, dtype=F64)
    hot = layer.step(X0_WALL, q, SRC0, XB, 600.0)
    torch.testing.assert_close(hot, built.step(X0_WALL, q, SRC0, XB, 600.0), rtol=1e-14,
                               atol=0)
    assert abs(hot[0].item() - X0_WALL[0].item()) > 1.0    # the carrier really acted


def test_moving_the_layer_drops_the_cached_folding():
    layer = _wall("exact")
    layer.step(X0_WALL, Q0, SRC0, XB, 600.0)
    assert layer._fold_cache is not None
    layer.to(dtype=torch.float32)
    assert layer._fold_cache is None
    x = layer.step(X0_WALL.float(), Q0.float(), SRC0.float(), XB.float(), 600.0)
    ref = _wall("exact").step(X0_WALL, Q0, SRC0, XB, 600.0)
    torch.testing.assert_close(x.double(), ref, rtol=1e-5, atol=0)


def test_an_in_place_edit_of_the_conductance_refolds():
    """The kept folding belongs to the conductance tensor and its version: an in-place edit
    of the construction-time conductance is folded again, as a new layer would be."""
    layer = _wall("exact")
    layer.step(X0_WALL, Q0, SRC0, XB, 600.0)
    layer.conductance[0] = 2.0 * H1
    torch.testing.assert_close(layer.step(X0_WALL, Q0, SRC0, XB, 600.0),
                               _wall("exact", g=(2.0 * H1, H2, K1, K2)).step(
                                   X0_WALL, Q0, SRC0, XB, 600.0), rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_a_layer_of_folded_nodes_only_returns_a_solve_result_under_on_failure_return(scheme):
    """No stored node, no linear solve: `on_failure="return"` still returns a
    `SolveResult` (converged, zero iterations), its `x` the recovered state, stacked."""
    net = Network(dtype=F64)
    for n in ("ambient", "sky", "s"):
        net.add_node(n)
    net.add_edge("ambient", "sky", kind="air")
    net.add_edge("s", "ambient", kind="film")
    layer = TransportLayer(net, "T", capacity=_t([0.0]), flow_kind="air",
                           boundary=["ambient", "sky"], conduction_kind="film",
                           conductance=_t([2.0]), storage_free=["s"], scheme=scheme)
    q, xb, src = _t([0.0]), _t([270.0, 250.0]), _t([0.0, 0.0, 4.0])
    for result, plain in ((layer.step(_t([300.0]), q, src, xb, 10.0, on_failure="return"),
                           layer.step(_t([300.0]), q, src, xb, 10.0)),
                          (layer.steady(q, src, xb, on_failure="return"),
                           layer.steady(q, src, xb))):
        assert bool(result.converged.all()) and int(result.iterations.sum()) == 0
        assert result.x.tolist() == plain.tolist()
        assert plain.tolist() == pytest.approx([270.0 + 4.0 / 2.0], rel=1e-15)


def test_the_kept_node_view_is_built_once_and_a_call_writes_nothing_on_it():
    """The kept-node layer is one object holding the full layer and its own structure; its
    per-node coefficients are the full layer's sliced on reading, and a step leaves it as
    it was (no per-call copy of the full layer's attributes)."""
    layer = _wall("trapezoidal")
    kept = layer._kept
    before = dict(vars(kept))
    diagnostics: dict = {}
    layer.step(X0_WALL, _t([0.05]), SRC0, XB, 600.0, diagnostics=diagnostics)
    assert layer._kept is kept
    assert vars(kept).keys() == before.keys()
    assert all(vars(kept)[k] is v for k, v in before.items())
    assert "capacity" not in vars(kept) and "carrier" not in vars(kept)
    assert kept.capacity.tolist() == [CZ, CW]
    assert kept.carrier is layer.carrier
    assert diagnostics["held_sources_dropped"] == 0


def test_a_folded_layer_survives_pickling():
    import pickle

    layer = _wall("implicit")
    copy = pickle.loads(pickle.dumps(layer))
    assert copy._kept._layer is copy
    torch.testing.assert_close(copy.step(X0_WALL, Q0, SRC0, XB, 600.0),
                               layer.step(X0_WALL, Q0, SRC0, XB, 600.0), rtol=0, atol=0)


def test_the_conductance_is_read_through_a_public_property():
    layer = _wall("exact")
    assert layer.conductance.tolist() == [H1, H2, K1, K2]
    assert _two_node_wall().conductance.shape == (2,)
    plain = TransportLayer(_wall_net(), "T", capacity=_t([CZ]), flow_kind="air",
                           boundary=["ambient"])
    assert plain.conductance is None


@pytest.mark.parametrize("call", ["step", "step_with_transfer", "steady"])
def test_a_per_call_removal_is_refused_on_a_layer_whose_every_node_is_folded(call):
    """With no stored node left there is no kept-node step, and a per-call removal at the
    folded node is still refused by name rather than dropped."""
    net = Network(dtype=F64)
    for n in ("inside", "s", "outside"):
        net.add_node(n)
    net.add_edge("inside", "s", kind="film")
    net.add_edge("s", "outside", kind="film")
    layer = TransportLayer(
        net, "T", capacity=_t([0.0]), flow_kind=(), boundary=["inside", "outside"],
        conduction_kind="film", conductance=_t([3.0, 5.0]), storage_free=["s"],
        scheme="exact",
    )
    xb = _t([293.0, 273.0])
    src = torch.zeros(3, dtype=F64)

    def run(r):
        if call == "steady":
            return layer.steady(_t([]), src, xb, removal=r)
        return getattr(layer, call)(_t([280.0]), _t([]), src, xb, 600.0, removal=r)

    run(_t([0.0]))
    with pytest.raises(ValueError, match=r"'T'.*removal.*folded.*\['s'\]"):
        run(_t([1e-4]))
