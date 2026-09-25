"""Intersection routing, node closure and direction averaging — spec sections 4.4, 4.5.

Every pinned number is from `.superpowers/munich-formulas.md` (T7, T8, T9).
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.canyon import KAPPA_IMPAQ, KAPPA_MUNICH
from noodl.apps.street_aq.routing import (
    MAX_N_THETA,
    MAX_SIGMA_THETA,
    StreetFlows,
    StreetGeometry,
    direction_offsets,
    n_theta_munich,
    node_closure,
    order_slots,
    routing_matrix,
    sigma_theta_munich,
)
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

DT = torch.float64


def test_sirane_routing_is_munich_s_worked_two_in_two_out_example():
    # `.superpowers/munich-formulas.md` T8: inflows [10, 4] counter-clockwise against
    # outflows [6, 8] clockwise, already balanced, give the greedy non-crossing fill.
    flux_in = torch.tensor([10.0, 4.0], dtype=DT)
    flux_out = torch.tensor([6.0, 8.0], dtype=DT)
    sirane = routing_matrix(flux_in, flux_out, model="sirane")
    torch.testing.assert_close(
        sirane, torch.tensor([[6.0, 4.0], [0.0, 4.0]], dtype=DT), rtol=0, atol=1e-14
    )
    mixing = routing_matrix(flux_in, flux_out, model="mixing")
    torch.testing.assert_close(
        mixing, torch.tensor([[30.0 / 7.0, 40.0 / 7.0], [12.0 / 7.0, 16.0 / 7.0]], dtype=DT),
        rtol=1e-14, atol=0,
    )
    # The two models are unambiguously different, which is the point of the test.
    assert float((sirane - mixing).abs().max()) > 1.0


@pytest.mark.parametrize("model", ["sirane", "mixing"])
def test_routing_matrix_reproduces_both_marginals_exactly(model):
    flux_in = torch.tensor([[3.0, 0.0, 5.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=DT)
    flux_out = torch.tensor([[4.0, 6.0, 0.0, 0.0], [2.0, 0.0, 2.0, 0.0]], dtype=DT)
    f = routing_matrix(flux_in, flux_out, model=model)
    torch.testing.assert_close(f.sum(-1), flux_in, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(f.sum(-2), flux_out, rtol=1e-14, atol=1e-15)
    assert bool((f >= 0).all())


def test_zero_rows_do_not_move_the_other_entries():
    """A zero entry anywhere in either ordered marginal leaves the rest of F alone."""
    a = routing_matrix(torch.tensor([10.0, 4.0], dtype=DT),
                       torch.tensor([6.0, 8.0], dtype=DT), model="sirane")
    b = routing_matrix(torch.tensor([10.0, 0.0, 4.0], dtype=DT),
                       torch.tensor([6.0, 0.0, 8.0], dtype=DT), model="sirane")
    torch.testing.assert_close(b[[0, 2]][:, [0, 2]], a, rtol=0, atol=1e-14)


def test_routing_matrix_is_differentiable_in_the_fluxes():
    # Deliberately NOT balanced. At a balanced node the two cumulative sums tie exactly at
    # the last entry, which is a kink of the piecewise-linear fill; what autograd returns
    # there is a legitimate average of the two one-sided slopes, not a value worth
    # asserting. Away from the tie, sum(F) = min(sum in, sum out) = 14, so it moves one
    # for one with each inflow and not at all with the outflows.
    flux_in = torch.tensor([10.0, 4.0], dtype=DT, requires_grad=True)
    flux_out = torch.tensor([6.0, 9.0], dtype=DT, requires_grad=True)
    f = routing_matrix(flux_in, flux_out, model="sirane")
    gi, go = torch.autograd.grad(f.sum(), (flux_in, flux_out))
    torch.testing.assert_close(gi, torch.ones(2, dtype=DT), rtol=0, atol=1e-12)
    torch.testing.assert_close(go, torch.zeros(2, dtype=DT), rtol=0, atol=1e-12)
    entry = routing_matrix(flux_in, flux_out, model="sirane")[0, 1]
    (grad,) = torch.autograd.grad(entry, flux_in)
    step = 1e-7
    for i in range(2):
        plus = flux_in.detach().clone()
        minus = flux_in.detach().clone()
        plus[i] += step
        minus[i] -= step
        central = (
            routing_matrix(plus, flux_out.detach(), model="sirane")[0, 1]
            - routing_matrix(minus, flux_out.detach(), model="sirane")[0, 1]
        ) / (2.0 * step)
        assert abs(float(grad[i]) - float(central)) < 1e-6


def test_node_closure_matches_munich_both_ways():
    # T9, exporting node: P_in = [10, 4], P_out = [6, 5] -> P0 = +3, alpha0 = 3/14.
    p_in, p_out, to_atm, from_atm = node_closure(
        torch.tensor([10.0, 4.0, -6.0, -5.0], dtype=DT)
    )
    torch.testing.assert_close(
        to_atm, torch.tensor([2.142857142857143, 0.8571428571428571, 0.0, 0.0], dtype=DT),
        rtol=1e-13, atol=0,
    )
    torch.testing.assert_close(
        p_in, torch.tensor([7.857142857142858, 3.142857142857143, 0.0, 0.0], dtype=DT),
        rtol=1e-13, atol=0,
    )
    assert float(from_atm.abs().max()) == 0.0
    torch.testing.assert_close(p_in.sum(), p_out.sum(), rtol=1e-14, atol=0)
    # T9 mirror, importing node: P_in = [4, 2], P_out = [6, 5] -> alpha0 = 5/11.
    p_in2, p_out2, to2, from2 = node_closure(
        torch.tensor([4.0, 2.0, -6.0, -5.0], dtype=DT)
    )
    torch.testing.assert_close(
        from2, torch.tensor([0.0, 0.0, 2.727272727272727, 2.2727272727272725], dtype=DT),
        rtol=1e-13, atol=0,
    )
    assert float(to2.abs().max()) == 0.0
    torch.testing.assert_close(p_in2.sum(), p_out2.sum(), rtol=1e-14, atol=0)


def test_node_closure_is_one_sided_and_survives_an_empty_junction():
    p_in, p_out, to_atm, from_atm = node_closure(torch.zeros(4, dtype=DT))
    for tensor in (p_in, p_out, to_atm, from_atm):
        assert float(tensor.abs().max()) == 0.0
    # A dead end: one inflow and nothing else. Its whole flux goes to the atmosphere.
    p_in, p_out, to_atm, from_atm = node_closure(torch.tensor([5.0, 0.0], dtype=DT))
    torch.testing.assert_close(to_atm, torch.tensor([5.0, 0.0], dtype=DT), rtol=0, atol=0)
    assert float(p_in.abs().max()) == 0.0
    # A dead end the other way: one outflow, fed entirely from the atmosphere.
    p_in, p_out, to_atm, from_atm = node_closure(torch.tensor([-5.0, 0.0], dtype=DT))
    torch.testing.assert_close(from_atm, torch.tensor([5.0, 0.0], dtype=DT), rtol=0, atol=0)
    assert float(p_out.abs().max()) == 0.0


@pytest.mark.parametrize(
    "n, expected",
    [(2, 0.431928), (3, 1.013848), (4, 0.995837), (5, 0.990866), (6, 0.986315),
     (7, 0.982560), (8, 0.979509), (9, 0.977016), (10, 0.974953)],
)
def test_munich_quadrature_weights_are_unnormalised_exactly_as_munich_leaves_them(
    n, expected
):
    """T7. The rectangle rule on a truncated +-2 sigma range with both endpoints at full
    weight does not sum to one, and reproducing that is the point: normalising it would put
    a uniform few-percent bias between this model and MUNICH."""
    sigma = torch.tensor([(n + 0.5) * math.pi / 180.0], dtype=DT)
    _, weights = direction_offsets("munich", sigma)
    assert weights.shape[-1] == n
    # The published sums are quoted to six decimal places, which is what 1e-6 reflects.
    assert abs(float(weights.sum()) - expected) < 1e-6


def test_munich_sample_count_and_sigma_theta():
    assert abs(MAX_SIGMA_THETA - math.pi / 18.0) < 1e-15
    assert MAX_N_THETA == 10
    sigma_v = torch.tensor([0.36, 0.36, 5.0], dtype=DT)
    u_ref = torch.tensor([5.0, 10.0, 1.0], dtype=DT)
    sigma = sigma_theta_munich(sigma_v, u_ref)
    torch.testing.assert_close(
        sigma, torch.tensor([0.072, 0.036, math.pi / 18.0], dtype=DT), rtol=1e-13, atol=0
    )
    # T7: u* = 0.3, PBLH = 500 -> sigma_v = 1.2 u* = 0.36; U = 5 gives 4.1253 deg (n = 4)
    # and U = 10 gives 2.0626 deg (n = 2).
    assert n_theta_munich(sigma).tolist() == [4, 2, 10]


def test_direction_offsets_handle_a_batch_with_different_sample_counts():
    sigma = torch.tensor([0.072, 0.036, 0.001], dtype=DT)
    offsets, weights = direction_offsets("munich", sigma)
    assert offsets.shape == weights.shape == (3, 4)
    # Row 1 needs two samples, so its two surplus slots carry exactly zero weight and its
    # sum is the one MUNICH computes for n = 2.
    assert abs(float(weights[1].sum()) - 0.431928) < 1e-6
    assert float(weights[1, 2]) == 0.0 and float(weights[1, 3]) == 0.0
    # Row 2 is below two degrees: MUNICH skips the averaging entirely.
    torch.testing.assert_close(weights[2], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DT),
                               rtol=0, atol=0)
    torch.testing.assert_close(offsets[2], torch.zeros(4, dtype=DT), rtol=0, atol=0)


def test_none_and_gauss_schemes():
    sigma = torch.tensor([0.1, 0.2], dtype=DT)
    offsets, weights = direction_offsets("none", sigma)
    assert offsets.shape == (2, 1)
    torch.testing.assert_close(offsets, torch.zeros(2, 1, dtype=DT), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.ones(2, 1, dtype=DT), rtol=0, atol=0)
    offsets, weights = direction_offsets("gauss", sigma, n_theta=5)
    assert offsets.shape == (2, 5)
    # Gauss-Hermite weights ARE normalised -- that is the whole difference from "munich".
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=DT),
                               rtol=1e-13, atol=0)
    torch.testing.assert_close(
        (weights * offsets**2).sum(-1), sigma**2, rtol=1e-12, atol=0
    )
    with pytest.raises(ValueError, match="scheme='gauss' needs n_theta"):
        direction_offsets("gauss", sigma)
    with pytest.raises(ValueError, match=r"'none', 'munich' or 'gauss'"):
        direction_offsets("rectangle", sigma)


def test_order_slots_sorts_and_rotates_at_most_once():
    angle = torch.tensor([[0.1, 1.0, 3.0, 5.0]], dtype=DT)
    active = torch.tensor([[True, True, True, False]])
    assert order_slots(angle, active, descending=True).tolist() == [[2, 1, 0, 3]]
    assert order_slots(angle, active, descending=False).tolist() == [[0, 1, 2, 3]]
    # The leading gap 4.0 - 0.2 = 3.8 exceeds pi, so one rotation settles it.
    wide = torch.tensor([[0.1, 0.2, 4.0, 0.0]], dtype=DT)
    assert order_slots(wide, active, descending=True).tolist() == [[1, 0, 2, 3]]
    assert order_slots(wide, active, descending=False).tolist() == [[2, 0, 1, 3]]


def test_routing_matrix_names_an_unknown_model():
    with pytest.raises(ValueError, match=r"routing_matrix.*'mixing' or 'sirane'.*'soulhac'"):
        routing_matrix(torch.ones(2, dtype=DT), torch.ones(2, dtype=DT), model="soulhac")


def _flows_fixture() -> tuple[Network, StreetGeometry]:
    """Two streets end to end (`s1`: a->b, `s2`: b->c): one real junction of degree two
    (`b`) and two dead ends (`a`, `c`). Built by hand, replicating `build_model`'s
    own edge-construction pattern (plan Task 5) -- `route`/`vent`/`exchange` edges carrying
    exactly the attributes `StreetFlows._read_edges` reads -- since `network.py` (Task 5)
    does not exist yet. Used only to construct `StreetFlows`, never called: this is the
    fixture the M3-R1 kappa-resolution test needs, not an end-to-end run.
    """
    names = ["s1", "s2"]
    graph = Network(dtype=torch.float64)
    graph.add_node("atmosphere")
    for name in names:
        graph.add_node(name)
    # Junction 0 = "a" (slot 0: s1 end u); junction 1 = "b" (slot 0: s1 end v, slot 1: s2
    # end u); junction 2 = "c" (slot 0: s2 end v) -- first-appearance order over (u, v)
    # per street, exactly as `StreetNetwork.junctions` walks it.
    graph.add_edge("s1", "s2", kind="route", junction=1, slot_a=0, slot_b=1)
    graph.add_edge("s2", "s1", kind="route", junction=1, slot_a=1, slot_b=0)
    graph.add_edge("s1", "atmosphere", kind="vent", junction=0, slot=0, street=0,
                   end="u", direction="out")
    graph.add_edge("atmosphere", "s1", kind="vent", junction=0, slot=0, street=0,
                   end="u", direction="in")
    graph.add_edge("s1", "atmosphere", kind="vent", junction=1, slot=0, street=0,
                   end="v", direction="out")
    graph.add_edge("atmosphere", "s1", kind="vent", junction=1, slot=0, street=0,
                   end="v", direction="in")
    graph.add_edge("s2", "atmosphere", kind="vent", junction=1, slot=1, street=1,
                   end="u", direction="out")
    graph.add_edge("atmosphere", "s2", kind="vent", junction=1, slot=1, street=1,
                   end="u", direction="in")
    graph.add_edge("s2", "atmosphere", kind="vent", junction=2, slot=0, street=1,
                   end="v", direction="out")
    graph.add_edge("atmosphere", "s2", kind="vent", junction=2, slot=0, street=1,
                   end="v", direction="in")
    graph.add_edge("s1", "atmosphere", kind="exchange", street=0, direction="out")
    graph.add_edge("atmosphere", "s1", kind="exchange", street=0, direction="in")
    graph.add_edge("s2", "atmosphere", kind="exchange", street=1, direction="out")
    graph.add_edge("atmosphere", "s2", kind="exchange", street=1, direction="in")
    geometry = StreetGeometry(
        names=names, u=["a", "b"], v=["b", "c"],
        length=torch.tensor([100.0, 100.0], dtype=DT),
        width=torch.tensor([20.0, 20.0], dtype=DT),
        height=torch.tensor([20.0, 20.0], dtype=DT),
        z0_b=torch.tensor([0.15, 0.15], dtype=DT),
        azimuth=torch.tensor([0.0, 0.0], dtype=DT),
    )
    return graph, geometry


def test_street_flows_resolves_kappa_by_formulation():
    """Ruling M3-R1: `kappa=None` resolves to MUNICH's 0.41 whenever a MUNICH-style form
    (`canyon_wind='exponential'`, `exchange='schulte'`, `roof_wind_form='macdonald'`) is
    selected, and to IMPAQ's 0.4 otherwise; an explicit float always wins."""
    net, geometry = _flows_fixture()

    def make(**kwargs) -> StreetFlows:
        return StreetFlows(net, None, geometry, **kwargs)

    assert make().kappa == KAPPA_IMPAQ
    assert make(canyon_wind="exponential").kappa == KAPPA_MUNICH
    assert make(exchange="schulte").kappa == KAPPA_MUNICH
    assert make(roof_wind_form="macdonald").kappa == KAPPA_MUNICH
    assert make(canyon_wind="exponential", kappa=0.38).kappa == 0.38


def _street_layer(net, kinds) -> TransportLayer:
    """The transport layer `build_model` builds, with `flow_kinds` as given."""
    return TransportLayer(
        net, "street", capacity=torch.tensor([40000.0, 40000.0], dtype=DT),
        flow_kind=kinds, boundary=["atmosphere"], scheme="implicit",
        quantity="concentration", unit="kg/m3",
    )


def test_street_flows_refuses_a_layer_whose_flow_kinds_are_in_another_order():
    """FR-3: the closure writes `q` as one concatenated block in the order
    `("route", "vent", "exchange")`, so a layer built with the same kinds in any other
    order would read the route flows as vent flows with no error anywhere. `layer=None`
    stays legal -- the fixtures above drive `_flows` without a Model at all."""
    net, geometry = _flows_fixture()
    good = StreetFlows(net, _street_layer(net, ("route", "vent", "exchange")), geometry)
    assert good.layer.flow_kinds == ("route", "vent", "exchange")
    swapped = _street_layer(net, ("vent", "route", "exchange"))
    with pytest.raises(ValueError, match=r"StreetFlows.*'street'.*vent.*route.*exchange"):
        StreetFlows(net, swapped, geometry)
