"""Model: several layers on one graph, stepped together."""

from __future__ import annotations

import sys
import types
import warnings

import pytest
import torch

from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.elements.base import ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.model import Model, Ports, StepContext
from noodl.nodesources import NodeSource
from noodl.topology import Network

F64 = torch.float64


def _net():
    """ambient -> z1 -> z2 -> ambient, all airpath; node order ambient, z1, z2."""
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    return net


def _build(*, learnable=False, closures=(), reactions=(), substeps=None, **model_kw):
    net = _net()
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65, learnable=learnable)
    air = KirchhoffLayer(
        net, "air", [el], drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"],
        quantity="pressure", unit="Pa",
    )
    species = TransportLayer(
        net, "species", capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", quantity="mass_fraction", unit="kg/kg",
    )
    model = Model(
        net, {"air": air, "species": species}, closures=closures, reactions=reactions,
        substeps=substeps, **model_kw,
    )
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "species.x_boundary": torch.tensor([1e-3], dtype=F64),
        "species.sources": torch.tensor([0.0, 2e-6, 0.0], dtype=F64),
    }
    state = {"species.x": torch.zeros(2, dtype=F64)}
    return net, model, state, drivers, el, species


def _build_multi():
    """The same network with a TWO-species transport layer: stacked (n_i, K) state layout."""
    net = _net()
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"],
    )
    gas = TransportLayer(
        net, "gas", capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], n_species=2, scheme="implicit",
    )
    model = Model(net, {"air": air, "gas": gas})
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "gas.x_boundary": torch.tensor([[1e-3, 2e-3]], dtype=F64),
    }
    state = {"gas.x": torch.zeros(2, 2, dtype=F64)}
    return model, state, drivers, gas


def test_step_produces_every_state_key_and_the_implicit_step_satisfies_the_balance():
    _, model, state, drivers, _, species = _build()
    new = model.step(state, drivers, 600.0)
    assert set(new) == {"air.phi", "air.q", "species.x"}
    assert new["air.phi"].shape == (3,) and new["air.q"].shape == (3,)
    q = model.kirchhoff["air"].flows_of_kind(new["air.q"], "airpath")
    rate = species.rate(new["species.x"], q, drivers["species.sources"],
                        drivers["species.x_boundary"])
    # backward Euler: (x_new - x_old) / dt == rate(x_new)
    torch.testing.assert_close(rate, (new["species.x"] - state["species.x"]) / 600.0,
                               rtol=1e-9, atol=1e-15)


def test_steady_is_the_limit_of_stepping_and_residuals_vanish_there():
    # The residual this test asserts on is the air layer's own Newton residual,
    # so the solve is asked for the accuracy the assertion needs (the default is the
    # dtype-derived sqrt(eps) ~ 1.5e-8, which leaves it at ~1e-9) rather than the assertion
    # being loosened. `**solve_kwargs` of step/steady reach the potential solves only.
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, _, _ = _build()
    ss = model.steady(state, drivers, **tight)
    x = state
    for _ in range(400):
        x = model.step(x, drivers, 3600.0, **tight)
    torch.testing.assert_close(x["species.x"], ss["species.x"], rtol=1e-6, atol=1e-12)
    res = model.residuals(ss, drivers)
    assert set(res) == {"air", "species"}
    assert res["air"].abs().max().item() < 1e-9
    assert res["species"].abs().max().item() < 1e-12


def test_closures_update_drivers_before_the_solve_and_may_not_write_state_keys():
    class Gust:
        def __call__(self, state, drivers):
            return {"wind": drivers["wind"] + torch.tensor([3.0, 0.0, 0.0], dtype=F64)}

    _, plain, state, drivers, _, _ = _build()
    _, gusty, _, _, _, _ = _build(closures=[Gust()])
    q_plain = plain.step(state, drivers, 60.0)["air.q"]
    q_gusty = gusty.step(state, drivers, 60.0)["air.q"]
    assert q_gusty[0] > q_plain[0]
    stronger = dict(drivers, wind=torch.tensor([8.0, 0.0, 0.0], dtype=F64))
    torch.testing.assert_close(q_gusty, plain.step(state, stronger, 60.0)["air.q"])

    class Rogue:
        def __call__(self, state, drivers):
            return {"air.phi": torch.zeros(3, dtype=F64)}

    _, rogue, _, _, _, _ = _build(closures=[Rogue()])
    with pytest.raises(ValueError, match=r"closure.*Rogue.*'air.phi'"):
        rogue.step(state, drivers, 60.0)


def test_pingpong_solves_with_the_state_at_the_start_of_the_step():
    class Feedback:
        def __call__(self, state, drivers):
            return {"wind": torch.stack([5.0 + 1e4 * state["species.x"][..., 0],
                                         torch.zeros_like(state["species.x"][..., 0]),
                                         torch.zeros_like(state["species.x"][..., 0])], dim=-1)}

    _, model, state, drivers, _, _ = _build(closures=[Feedback()])
    start = {"species.x": torch.tensor([2e-4, 0.0], dtype=F64)}
    new = model.step(start, drivers, 60.0)
    _, plain, _, _, _, _ = _build()
    same = plain.step(start, dict(drivers, wind=torch.tensor([7.0, 0.0, 0.0], dtype=F64)), 60.0)
    torch.testing.assert_close(new["air.q"], same["air.q"])


def test_a_missing_required_driver_is_named():
    _, model, state, drivers, _, _ = _build()
    del drivers["air.phi_boundary"]
    with pytest.raises(KeyError, match="air.phi_boundary"):
        model.step(state, drivers, 60.0)


def test_a_transport_layer_whose_kinds_no_kirchhoff_layer_provides_reads_a_driver():
    """The rule this replaces refused the model outright. A transport layer advecting on a
    kind the Kirchhoff layer does not provide is now driver-prescribed."""
    net = _net()
    net.add_edge("z1", "z2", kind="duct")
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        boundary=["ambient"],
    )
    heat = TransportLayer(net, "heat", capacity=torch.ones(2, dtype=F64),
                          flow_kind=("airpath", "duct"), boundary=["ambient"])
    model = Model(net, {"air": air, "heat": heat})
    assert model.flow_layer_of["heat"] is None
    assert model.flow_driver_of["heat"] == "heat.q"
    with pytest.raises(KeyError, match=r"heat\.q"):
        model.steady(
            {},
            {"air.phi_boundary": torch.zeros(1, dtype=F64),
             "heat.x_boundary": torch.zeros(1, dtype=F64)},
        )


def test_layer_dict_is_validated():
    net = _net()
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        boundary=["ambient"],
    )
    with pytest.raises(TypeError, match="'oops'"):
        Model(net, {"air": air, "oops": object()})
    with pytest.raises(ValueError, match="different Network"):
        Model(_net(), {"air": air})
    with pytest.raises(ValueError, match="coupling"):
        Model(net, {"air": air}, coupling="onion")


def test_substeps_match_a_uniformly_finer_step():
    _, coarse, state, drivers, _, _ = _build(substeps={"species": 4})
    _, fine, _, _, _, _ = _build()
    x_coarse = coarse.step(state, drivers, 2400.0)["species.x"]
    x = state
    for _ in range(4):
        x = fine.step(x, drivers, 600.0)
    torch.testing.assert_close(x_coarse, x["species.x"], rtol=1e-12, atol=1e-18)
    with pytest.raises(ValueError, match=r"substeps\['species'\]"):
        _build(substeps={"species": 0})
    with pytest.raises(KeyError, match="'nope'"):
        _build(substeps={"nope": 2})


def test_reactions_apply_after_the_transport_step():
    _, model, state, drivers, _, _ = _build(reactions=[("species", FirstOrderDecay(1e-3))])
    _, plain, _, _, _, _ = _build()
    x_r = model.step(state, drivers, 600.0)["species.x"]
    x_p = plain.step(state, drivers, 600.0)["species.x"]
    torch.testing.assert_close(x_r, x_p * torch.exp(torch.tensor(-1e-3 * 600.0, dtype=F64)))
    with pytest.raises(KeyError, match="'nope'"):
        _build(reactions=[("nope", FirstOrderDecay(1e-3))])


def test_reaction_order_before_transport_reacts_the_step_start_state_then_transports_it():
    # R then T: the returned state is the transport step of the REACTED step-start state,
    # the sample SWMM reports. Pinned against the plain model stepped from the reacted state
    # by hand, from a nonzero start so the reaction actually acts; the default order is
    # "after_transport", and anything else is refused by name.
    decay = FirstOrderDecay(1e-3)
    _, model, state, drivers, _, _ = _build(
        reactions=[("species", decay)], reaction_order="before_transport",
    )
    _, plain, _, _, _, _ = _build()
    assert plain.reaction_order == "after_transport"
    start = {"species.x": torch.tensor([4e-4, 7e-4], dtype=F64)}
    x_r = model.step(start, drivers, 600.0)["species.x"]
    reacted = {"species.x": decay.apply(start["species.x"], 600.0)}
    torch.testing.assert_close(x_r, plain.step(reacted, drivers, 600.0)["species.x"])
    # The two orders' fixed points differ by exactly one reaction: x_TR = R(x_RT).
    x_rt = x_tr = start
    _, after, _, _, _, _ = _build(reactions=[("species", decay)])
    for _ in range(400):
        x_rt = model.step(x_rt, drivers, 600.0)
        x_tr = after.step(x_tr, drivers, 600.0)
    torch.testing.assert_close(
        x_tr["species.x"], decay.apply(x_rt["species.x"], 600.0), rtol=1e-12, atol=1e-18,
    )
    with pytest.raises(ValueError, match="reaction_order"):
        _build(reaction_order="sideways")


def test_reaction_order_before_transport_keeps_the_gradient_and_the_iterated_coupling():
    decay = FirstOrderDecay(1e-3)
    _, model, state, drivers, el, _ = _build(
        learnable=True, reactions=[("species", decay)], reaction_order="before_transport",
    )
    start = {"species.x": torch.tensor([4e-4, 7e-4], dtype=F64)}

    def loss():
        return model.step(start, drivers, 600.0)["species.x"].sum()

    loss().backward()
    grad = el.C.grad[1].item()
    h = 1e-6
    with torch.no_grad():
        el.C[1] += h
        up = loss().item()
        el.C[1] -= 2 * h
        down = loss().item()
        el.C[1] += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-5)
    _, onion, _, _, _, _ = _build(
        reactions=[("species", decay)], reaction_order="before_transport",
        coupling="iterate", iterate_tol={"species": 1e-14},
    )
    torch.testing.assert_close(
        onion.step(start, drivers, 600.0)["species.x"],
        model.step(start, drivers, 600.0)["species.x"].detach(),
        rtol=1e-10, atol=1e-18,
    )


def test_reactions_are_a_splitting_of_step_only_and_stay_out_of_steady_and_residuals():
    # Pins the documented placement: a reaction is applied around a transport step, so it
    # is outside both `steady` and the balance `residuals` reports. A later change must not
    # "fix" this by folding the reaction into either.
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, _, _ = _build(reactions=[("species", FirstOrderDecay(1e-3))])
    _, plain, _, _, _, _ = _build()
    ss = model.steady(state, drivers, **tight)
    torch.testing.assert_close(ss["species.x"], plain.steady(state, drivers, **tight)["species.x"])
    assert model.residuals(ss, drivers)["species"].abs().max().item() < 1e-12


def test_diagnostics_carry_the_layers_status_and_the_pass_count():
    _, model, state, drivers, _, _ = _build()
    diag: dict = {}
    model.step(state, drivers, 60.0, diagnostics=diag)
    assert diag["passes"] == 1
    assert bool(diag["layers"]["air"]["converged"].all())
    assert diag["layers"]["species"]["substeps"] == 1
    # The default linear_solver="auto" resolves to sparse_direct at this small
    # (single-instance) batch size (see `benchmarks/transport_solver_bench.py`).
    assert diag["layers"]["species"]["linear"]["backend"] == "sparse_direct"


def test_ports_round_trip_and_air_boundary_flow_balances_the_interior():
    _, model, state, drivers, _, _ = _build()
    # The boundary flow this asserts on IS the air layer's interior
    # residual (A q sums to zero over all nodes), so the solve is asked for the accuracy the
    # bound needs instead of relying on two default-tolerance residuals cancelling.
    new = model.step(state, drivers, 60.0, atol=1e-14, rtol=1e-14)
    ports = model.ports(new)
    assert isinstance(ports, Ports)
    assert ports.boundary_nodes == {"air": ["ambient"], "species": ["ambient"]}
    assert ports.prescribed_keys == {"air": "air.phi_boundary", "species": "species.x_boundary"}
    assert ports.boundary_flows["air"].shape == (1,)
    assert abs(ports.boundary_flows["air"].item()) < 1e-10     # no interior air sources


def test_gradient_of_a_species_loss_wrt_leakage_matches_finite_differences():
    _, model, state, drivers, el, _ = _build(learnable=True)

    def loss():
        return model.step(state, drivers, 600.0)["species.x"].sum()

    value = loss()
    value.backward()
    grad = el.C.grad[1].item()
    h = 1e-6
    with torch.no_grad():
        el.C[1] += h
        up = loss().item()
        el.C[1] -= 2 * h
        down = loss().item()
        el.C[1] += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_optional_sources_default_to_zero_and_bad_dt_or_missing_state_is_named():
    # The "<layer>.sources" driver is optional; without it the layer is stepped (and solved
    # steady) with full-node zeros. Covers the two step-time refusals as well.
    _, model, state, drivers, _, _ = _build()
    bare = {k: v for k, v in drivers.items() if k != "species.sources"}
    zeroed = dict(bare, **{"species.sources": torch.zeros(3, dtype=F64)})
    torch.testing.assert_close(
        model.step(state, bare, 600.0)["species.x"], model.step(state, zeroed, 600.0)["species.x"]
    )
    torch.testing.assert_close(
        model.steady(state, bare)["species.x"], model.steady(state, zeroed)["species.x"]
    )
    at = model.step(state, bare, 600.0)
    torch.testing.assert_close(
        model.residuals(at, bare)["species"], model.residuals(at, zeroed)["species"]
    )
    with pytest.raises(ValueError, match="dt must be positive"):
        model.step(state, drivers, 0.0)
    with pytest.raises(KeyError, match=r"species\.x"):
        model.step({}, drivers, 600.0)


def test_steady_without_a_sources_driver_keeps_the_states_own_layout():
    # `TransportLayer.steady` reads its reduced/stacked flag off the SOURCES, so the zeros
    # `Model` defaults to must follow the STATE's layout, not `x_boundary`'s: a (n_b, K)
    # -shaped boundary value for a single-species layer would otherwise flip the returned
    # "<l>.x" from (..., n_i) to (..., n_i, 1).
    _, model, state, drivers, _, _ = _build()
    bare = {k: v for k, v in drivers.items() if k != "species.sources"}
    bare["species.x_boundary"] = torch.tensor([[1e-3]], dtype=F64)  # (n_b, K), not (n_b,)
    ss = model.steady(state, bare)
    explicit = model.steady(state, dict(bare, **{"species.sources": torch.zeros(3, dtype=F64)}))
    assert ss["species.x"].shape == explicit["species.x"].shape == state["species.x"].shape
    torch.testing.assert_close(ss["species.x"], explicit["species.x"])


def test_a_two_species_layer_steps_to_a_steady_state_in_the_stacked_layout():
    # The (n_i, K) half of the "<l>.x" layout promise, including the zero-sources default,
    # whose stacked branch no single-species test reaches. Both routes to the steady state
    # are checked and must agree: STEPPING (what the test's name promises) and
    # `Model.steady`. `TransportLayer.steady` is reliable for K > 1: a K-species operator
    # is block diagonal with K identical blocks, so its Krylov space is invariant after n_i
    # steps, and `gmres` must handle that near-breakdown. See
    # `tests/solvers/test_gmres_breakdown.py`.
    model, state, drivers, gas = _build_multi()
    zero = torch.zeros(3, 2, dtype=F64)
    new = model.step(state, drivers, 600.0)
    assert new["gas.x"].shape == (2, 2)
    q = model.kirchhoff["air"].flows_of_kind(new["air.q"], "airpath")
    torch.testing.assert_close(
        gas.rate(new["gas.x"], q, zero, drivers["gas.x_boundary"]),
        (new["gas.x"] - state["gas.x"]) / 600.0, rtol=1e-9, atol=1e-15,
    )
    x = state
    for _ in range(200):
        x = model.step(x, drivers, 3600.0, atol=1e-14, rtol=1e-14)
    assert x["gas.x"].shape == (2, 2)
    # Pure advection from the boundary with no sources: every interior node ends at the
    # boundary value of its own species, and the transport balance vanishes there.
    torch.testing.assert_close(
        x["gas.x"], drivers["gas.x_boundary"].expand(2, 2).contiguous(),
        rtol=1e-6, atol=1e-12,
    )
    res = model.residuals(x, drivers)["gas"]
    assert res.shape == (2, 2) and res.abs().max().item() < 1e-12

    # The same fixed point straight from `Model.steady` (i.e. through the linear solve that
    # used to fail at K > 1), in the same stacked layout.
    ss = model.steady(state, drivers)
    assert ss["gas.x"].shape == (2, 2)
    torch.testing.assert_close(ss["gas.x"], x["gas.x"], rtol=1e-6, atol=1e-9)
    assert model.residuals(ss, drivers)["gas"].abs().max().item() < 1e-12


def test_closures_and_reactions_are_checked_for_callability_at_construction():
    with pytest.raises(TypeError, match=r"closure 1 is a int"):
        _build(closures=[lambda state, drivers: {}, 7])
    with pytest.raises(TypeError, match=r"reaction for layer 'species' is a str"):
        _build(reactions=[("species", "decay")])
    with pytest.raises(TypeError, match=r"substeps\['species'\] must be an integer"):
        _build(substeps={"species": "two"})


def test_gradient_of_a_steady_species_loss_wrt_leakage_matches_finite_differences():
    # `steady` is the other differentiable entry point (the transport steady solve on the
    # Kirchhoff layer's adjoint); the tests above cover `step` only. Tight Newton
    # tolerances so the finite difference is not comparing solver noise.
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, el, _ = _build(learnable=True)

    def loss():
        return model.steady(state, drivers, **tight)["species.x"].sum()

    loss().backward()
    grad = el.C.grad[1].item()
    h = 1e-6
    with torch.no_grad():
        el.C[1] += h
        up = loss().item()
        el.C[1] -= 2 * h
        down = loss().item()
        el.C[1] += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_batched_drivers_run_as_a_batch_matching_the_single_instances():
    _, model, state, drivers, _, _ = _build()
    winds = torch.tensor([[5.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=F64)
    batched = model.step(state, dict(drivers, wind=winds), 600.0)
    assert batched["air.phi"].shape == (3, 3) and batched["species.x"].shape == (3, 2)
    for i in range(3):
        single = model.step(state, dict(drivers, wind=winds[i]), 600.0)
        torch.testing.assert_close(batched["species.x"][i], single["species.x"],
                                   rtol=1e-10, atol=1e-18)


class _Feedback:
    """Wind that grows with z1's concentration: a genuine flow <- transport feedback.

    The gain is ADDED to the wind driver rather than to a hard-coded 5.0, so that a batched
    `wind` survives the closure: with the fixtures' (5, 0, 0) this is exactly
    (5 + gain * x1, 0, 0), and with a batch of winds it is that per instance.
    """

    def __init__(self, gain: float) -> None:
        self.gain = gain

    def __call__(self, state, drivers):
        x1 = state["species.x"][..., 0]
        zero = torch.zeros_like(x1)
        bump = torch.stack([self.gain * x1, zero, zero], dim=-1)
        return {"wind": drivers["wind"] + bump}


def test_iterate_requires_tolerances_naming_the_layers():
    with pytest.raises(ValueError, match="iterate_tol"):
        _build(coupling="iterate")
    with pytest.raises(KeyError, match="'nope'"):
        _build(coupling="iterate", iterate_tol={"nope": 1e-9})


def test_iterate_reaches_a_fixed_point_of_the_coupled_pass():
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-12},
        iterate_max=60,
    )
    diag: dict = {}
    ss = model.steady(state, drivers, diagnostics=diag)
    assert diag["passes"] > 1 and bool(diag["converged"].all())
    # one more pass from the fixed point changes nothing beyond the tolerance
    again, _, _ = model._pass(ss, drivers, None, {})
    assert (again["species.x"] - ss["species.x"]).abs().max().item() < 1e-11
    assert (again["air.q"] - ss["air.q"]).abs().max().item() < 1e-9


def test_iterate_carries_on_past_a_pass_whose_potential_solve_stops_short():
    """A primal pass is one iterate: with one Newton iteration per pass the head solve
    stops short on the first passes, the iteration carries on from each pass's last Newton
    iterate (warm-starting the next), and lands on the same fixed point as a full solve."""
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-12},
        iterate_max=200,
    )
    full = model.step(state, drivers, 600.0)
    diag: dict = {}
    short = model.step(state, drivers, 600.0, max_iter=1, diagnostics=diag)
    assert bool(diag["converged"].all())
    assert bool(diag["layers"]["air"]["converged"].all())
    # the same fixed point, to the iteration's own resolution
    torch.testing.assert_close(short["species.x"], full["species.x"], rtol=1e-6, atol=0.0)


def test_iterate_never_accepts_a_pass_whose_potential_solve_stopped_short():
    """A pass whose head solve did not converge is not converged, whatever its change; the
    iteration's own failure then names it."""
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1.0},
        iterate_max=3,
    )
    with pytest.raises(RuntimeError, match="potential solve did not converge"):
        model.step(state, drivers, 600.0, max_iter=0)
    # a caller's own on_failure is passed through unchanged
    with pytest.raises(RuntimeError, match="failed to converge"):
        model.step(state, drivers, 600.0, max_iter=0, on_failure="raise")


def test_iterate_equals_pingpong_when_there_is_no_feedback():
    """The second pass warm-starts Newton from the first pass's phi, so the two passes can
    differ at solver tolerance; the tolerance here is above that and the step count is 2."""
    _, it, state, drivers, _, _ = _build(coupling="iterate", iterate_tol={"species": 1e-11})
    _, pp, _, _, _, _ = _build()
    diag: dict = {}
    a = it.step(state, drivers, 600.0, diagnostics=diag)
    b = pp.step(state, drivers, 600.0)
    torch.testing.assert_close(a["species.x"], b["species.x"], rtol=1e-9, atol=1e-15)
    assert diag["passes"] == 2          # the second pass only confirms


def test_iterate_reports_non_convergence_per_instance_or_raises():
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-15},
        iterate_max=2,
    )
    with pytest.raises(RuntimeError, match=r"iterate.*2 passes"):
        model.steady(state, drivers)
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, differentiable=False,
                       on_failure="return")
    assert "species.x" in out
    assert diag["passes"] == 2 and not bool(diag["converged"].all())
    assert diag["max_change"]["species"].item() > 1e-15


def test_iterate_gradient_is_the_fixed_points_derivative_from_a_cold_start():
    """The mechanism is the implicit interface adjoint, not unrolling; the check is a
    central-difference comparison, from the zero state."""
    _, model, state, drivers, el, _ = _build(
        learnable=True, closures=[_Feedback(2e3)], coupling="iterate",
        iterate_tol={"species": 1e-13}, iterate_max=80,
    )

    def loss():
        return model.steady(state, drivers)["species.x"].sum()

    loss().backward()
    grad = el.C.grad[0].item()
    h = 1e-6
    with torch.no_grad():
        el.C[0] += h
        up = loss().item()
        el.C[0] -= 2 * h
        down = loss().item()
        el.C[0] += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-4)


def test_iterate_batched_convergence_is_per_instance():
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-12},
        iterate_max=60,
    )
    winds = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    diag: dict = {}
    # At the default Newton tolerance the weaker-wind instance stagnates at the
    # solver's own noise floor (~2e-11, measured) and never reaches the 1e-12 asked of the
    # coupling; the SOLVE is asked for the accuracy the assertion needs instead.
    model.steady(state, dict(drivers, wind=winds), diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert diag["converged"].shape == (2,) and bool(diag["converged"].all())
    assert diag["max_change"]["species"].shape == (2,)


def test_iterate_non_convergence_names_only_the_failing_batch_instances():
    """The basic non-convergence test is unbatched, so it reports "all"; this one pins the
    per-instance naming.

    Tight Newton tolerances so that what instance 1 runs out of is the PASS
    budget and not the solver's noise floor: at pass 11 the strong-wind instance is at a
    change of ~6e-15 and the weak-wind one still at ~2e-11 (measured).
    """
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-12},
        iterate_max=11,
    )
    winds = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    drv = dict(drivers, wind=winds)
    with pytest.raises(RuntimeError, match=r"instances \[1\]"):
        model.steady(state, drv, **tight)
    diag: dict = {}
    model.steady(state, drv, diagnostics=diag, differentiable=False, on_failure="return",
                 **tight)
    assert diag["converged"].tolist() == [True, False]
    assert diag["max_change"]["species"][0].item() <= 1e-12 < diag["max_change"]["species"][1]


def test_iterate_refuses_a_pass_budget_below_two_at_construction():
    """One pass has no predecessor to compare against, so it can never be reported as
    converged: a budget below two is a configuration that would always raise, and zero would
    return the input state untouched first. Both are refused when the model is built."""
    for bad in (1, 0, -3):
        with pytest.raises(ValueError, match=rf"iterate_max >= 2, got {bad!r}"):
            _build(coupling="iterate", iterate_tol={"species": 1e-9}, iterate_max=bad)
    # `pingpong` never iterates, so the same value is none of its business
    _build(iterate_max=1)


def test_iterate_relaxation_moves_no_fixed_point():
    """`iterate_relaxation` (the newest pass's weight in the fed transport state) is refused
    outside (0, 1]; inside, it changes how the passes reach the step, not where: 0.5
    (Hensen's default) and 1 (plain successive substitution) land on the same state."""
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="iterate_relaxation"):
            _build(coupling="iterate", iterate_tol={"species": 1e-12}, iterate_relaxation=bad)
    out = []
    for w in (0.5, 1.0):
        _, model, state, drivers, _, _ = _build(
            closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-13},
            iterate_max=60, iterate_relaxation=w)
        out.append(model.step(state, drivers, 60.0, atol=1e-14, rtol=1e-14)["species.x"])
    assert torch.allclose(out[0], out[1], rtol=0, atol=1e-11)


def test_iterate_steps_every_transport_layer_but_tests_only_the_named_ones():
    """The motivating case: a species layer in kg/kg beside a thermal layer in kelvin.

    No single absolute tolerance can straddle the two, so `iterate_tol` names the species
    layer alone. The thermal layer is still stepped on EVERY pass -- it just gets no vote on
    convergence, and no entry in `max_change`. On the tight Newton tolerances:
    the assertions below are three orders inside the coupling tolerance, so the solve is
    asked for the accuracy they need.
    """
    tight = {"atol": 1e-14, "rtol": 1e-14}
    net = _net()
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"],
    )
    species = TransportLayer(
        net, "species", capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", quantity="mass_fraction", unit="kg/kg",
    )
    heat = TransportLayer(
        net, "heat", capacity=torch.tensor([5.0e4, 8.0e4], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", quantity="temperature", unit="K",
    )
    model = Model(
        net, {"air": air, "species": species, "heat": heat}, closures=[_Feedback(2e3)],
        coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
    )
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "species.x_boundary": torch.tensor([1e-3], dtype=F64),
        "species.sources": torch.tensor([0.0, 2e-6, 0.0], dtype=F64),
        "heat.x_boundary": torch.tensor([293.0], dtype=F64),
        "heat.sources": torch.tensor([0.0, 0.1, 0.0], dtype=F64),
    }
    state = {"species.x": torch.zeros(2, dtype=F64),
             "heat.x": torch.full((2,), 293.0, dtype=F64)}
    diag: dict = {}
    ss = model.steady(state, drivers, diagnostics=diag, **tight)
    assert diag["passes"] > 1 and bool(diag["converged"].all())
    # the untested layer is reported on by neither `max_change` nor the convergence verdict
    assert set(diag["max_change"]) == {"species"}
    assert set(diag["layers"]) == {"air", "species", "heat"}   # but it IS part of the pass

    # Stepped, not skipped: the thermal state is more than a kelvin away from what the FIRST
    # pass alone gives, because the flows kept moving under it while the species converged.
    first, _, _ = model._pass(state, drivers, None, tight)
    assert (ss["heat.x"] - first["heat.x"]).abs().max().item() > 0.5
    assert 293.0 < ss["heat.x"].min().item() < 300.0

    # And the reason it may not share the species layer's tolerance: at the fixed point one
    # further pass still moves it by ~1e-8 K, which is 1e4 times the 1e-12 the species layer
    # has converged to. Had it been named in `iterate_tol`, this model would never converge.
    again, _, _ = model._pass(ss, drivers, None, tight)
    moved_x = (again["species.x"] - ss["species.x"]).abs().max().item()
    moved_t = (again["heat.x"] - ss["heat.x"]).abs().max().item()
    assert moved_x < 1e-12 < moved_t and moved_t > 1e3 * moved_x


def _prescribed_flow_network():
    """Two interior nodes and one boundary node, joined by 'link' edges only.

    No Kirchhoff layer can own 'link' (no element is built for it), so the flows must
    come from the driver "conc.q". Edge order is the insertion order: a->b, b->amb.
    """
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "amb"):
        net.add_node(name)
    net.add_edge("a", "b", kind="link")
    net.add_edge("b", "amb", kind="link")
    layer = TransportLayer(
        net, "conc", capacity=torch.tensor([10.0, 20.0], dtype=torch.float64),
        flow_kind="link", boundary=["amb"], scheme="implicit",
        quantity="concentration", unit="kg/m3",
    )
    return net, layer


def test_model_reads_transport_flows_from_a_driver_when_no_kirchhoff_layer_owns_them():
    net, layer = _prescribed_flow_network()
    model = Model(net, {"conc": layer})
    assert model.flow_layer_of["conc"] is None
    assert model.flow_driver_of["conc"] == "conc.q"
    sources = torch.zeros(3, dtype=torch.float64)
    sources[net.node_index("a")] = 1.0
    drivers = {
        "conc.x_boundary": torch.zeros(1, dtype=torch.float64),
        "conc.sources": sources,
        "conc.q": torch.tensor([2.0, 2.0], dtype=torch.float64),
    }
    state = model.steady({}, drivers)
    # Steady: 1 kg/s in at a, carried a->b->amb by a 2 m3/s flow, so x_a = x_b = 0.5.
    torch.testing.assert_close(
        state["conc.x"], torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


def test_model_raises_naming_the_missing_flow_driver():
    net, layer = _prescribed_flow_network()
    model = Model(net, {"conc": layer})
    drivers = {
        "conc.x_boundary": torch.zeros(1, dtype=torch.float64),
        "conc.sources": torch.zeros(3, dtype=torch.float64),
    }
    with pytest.raises(KeyError, match=r"conc\.q"):
        model.steady({}, drivers)


def test_model_raises_when_the_flow_driver_has_the_wrong_length():
    net, layer = _prescribed_flow_network()
    model = Model(net, {"conc": layer})
    drivers = {
        "conc.x_boundary": torch.zeros(1, dtype=torch.float64),
        "conc.sources": torch.zeros(3, dtype=torch.float64),
        "conc.q": torch.tensor([2.0], dtype=torch.float64),
    }
    with pytest.raises(ValueError, match=r"conc\.q.*2 flow edges.*got 1"):
        model.steady({}, drivers)


def test_ports_reports_the_flow_driver_key_for_a_prescribed_flow_layer():
    net, layer = _prescribed_flow_network()
    model = Model(net, {"conc": layer})
    ports = model.ports({"conc.x": torch.zeros(2, dtype=torch.float64)})
    assert ports.flow_keys == {"conc": "conc.q"}
    assert ports.prescribed_keys["conc"] == "conc.x_boundary"


def test_a_closure_may_write_the_flow_driver_of_a_prescribed_flow_layer():
    net, layer = _prescribed_flow_network()

    class Flows:
        def __call__(self, state, drivers):
            return {"conc.q": torch.tensor([2.0, 2.0], dtype=torch.float64)}

    model = Model(net, {"conc": layer}, closures=[Flows()])
    sources = torch.zeros(3, dtype=torch.float64)
    sources[net.node_index("a")] = 1.0
    state = model.steady({}, {
        "conc.x_boundary": torch.zeros(1, dtype=torch.float64),
        "conc.sources": sources,
    })
    torch.testing.assert_close(
        state["conc.x"], torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=1e-12, atol=1e-15
    )


def test_a_closure_still_may_not_write_a_state_key():
    net, layer = _prescribed_flow_network()

    class WritesX:
        def __call__(self, state, drivers):
            return {"conc.x": torch.zeros(2, dtype=torch.float64)}

    bad = Model(net, {"conc": layer}, closures=[WritesX()])
    with pytest.raises(ValueError, match=r"conc\.x.*state key"):
        bad.steady({}, {"conc.x_boundary": torch.zeros(1, dtype=torch.float64),
                        "conc.sources": torch.zeros(3, dtype=torch.float64),
                        "conc.q": torch.ones(2, dtype=torch.float64)})


def test_model_raises_when_a_layer_has_both_a_kirchhoff_owner_and_a_flow_driver():
    """Uses `_build()`, this module's own fixture for a Model with one Kirchhoff layer
    ("air") and one transport layer ("species") the Kirchhoff layer owns."""
    net, model, state, drivers, _, layer = _build()
    drivers = dict(drivers)
    name = next(iter(model.transport))
    owner = model.flow_layer_of[name]
    assert owner is not None
    b = int(sum(len(model.net.edge_index(k)) for k in layer.flow_kinds))
    drivers[f"{name}.q"] = torch.zeros(b, dtype=torch.float64)
    with pytest.raises(ValueError, match=rf"{name}.*{owner}.*{name}\.q"):
        model.steady(state, drivers)


class _WritesSpeciesQ:
    def __call__(self, state, drivers):
        return {"species.q": torch.zeros(3, dtype=F64)}


def test_a_closure_may_not_write_the_flow_driver_of_a_layer_a_kirchhoff_layer_owns():
    """The `_apply_closures` carve-out has three conjuncts, and the third -- that NO
    Kirchhoff layer owns the transport layer's kinds -- is what keeps a closure from
    quietly overwriting flows the solve produced. "air" owns "species" here, so
    `"species.q"` is refused as a state key rather than accepted as a driver, and the
    layer goes on seeing the solved flows."""
    _net_, model, state, drivers, _el, _layer = _build(closures=[_WritesSpeciesQ()])
    assert model.flow_layer_of["species"] == "air"
    with pytest.raises(ValueError, match=r"species\.q.*state key"):
        model.steady(state, drivers)


class _CounterClosure:
    """A closure that carries its own integer state across steps."""

    state_keys = ("demo.count",)
    # Advances by a FIXED 1.0 every call, regardless of the step's own interval, so it
    # does not integrate (does not need a StepContext).
    integrates = False

    def __call__(self, state, drivers):
        return {"demo.count": state["demo.count"] + 1.0}


class _SilentClosure:
    state_keys = ("demo.count",)
    integrates = False  # never writes its state key at all; no interval involved either.

    def __call__(self, state, drivers):
        return {}


def _counter_model(closures):
    net = Network(dtype=torch.float64)
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="pipe")
    layer = TransportLayer(
        net, "c", capacity=torch.tensor([2.0], dtype=torch.float64), flow_kind="pipe",
        boundary=[net.nodes[1]], scheme="implicit",
    )
    return net, layer, Model(net, {"c": layer}, closures=closures)


def test_closure_state_is_carried_across_steps():
    _, _, model = _counter_model([_CounterClosure()])
    state = {
        "c.x": torch.zeros(1, dtype=torch.float64),
        "demo.count": torch.zeros((), dtype=torch.float64),
    }
    drivers = {
        "c.q": torch.tensor([1.0], dtype=torch.float64),
        "c.x_boundary": torch.zeros(1, dtype=torch.float64),
    }
    for _ in range(3):
        state = model.step(state, drivers, 1.0)
    assert float(state["demo.count"]) == 3.0


def test_closure_state_key_is_registered_on_the_model():
    closure = _CounterClosure()
    _, _, model = _counter_model([closure])
    assert model.closure_state_keys == {"demo.count": closure}


def test_two_closures_claiming_one_state_key_are_refused():
    with pytest.raises(ValueError, match="both declare the state key 'demo.count'"):
        _counter_model([_CounterClosure(), _CounterClosure()])


def test_a_closure_may_not_claim_a_layer_state_key():
    class _Bad:
        state_keys = ("c.x",)
        integrates = False  # declared so the state_keys/layer-collision check is reached.

        def __call__(self, state, drivers):
            return {}

    with pytest.raises(ValueError, match="which is layer 'c''s own state key"):
        _counter_model([_Bad()])


def test_a_declared_state_key_that_is_not_written_is_refused():
    _, _, model = _counter_model([_SilentClosure()])
    state = {
        "c.x": torch.zeros(1, dtype=torch.float64),
        "demo.count": torch.zeros((), dtype=torch.float64),
    }
    drivers = {
        "c.q": torch.tensor([1.0], dtype=torch.float64),
        "c.x_boundary": torch.zeros(1, dtype=torch.float64),
    }
    with pytest.raises(KeyError, match="did not return it"):
        model.step(state, drivers, 1.0)


def test_a_same_named_driver_does_not_satisfy_a_silent_closures_state_key():
    """`key not in drv` alone would be satisfied by a caller-supplied DRIVER of the same
    name, sitting in `drv` from its initial `dict(drivers)` copy -- even though no closure
    ever wrote it -- freezing the closure's carried state at the caller's value with no
    error. The check must be on what the closure itself returned."""
    _, _, model = _counter_model([_SilentClosure()])
    state = {
        "c.x": torch.zeros(1, dtype=torch.float64),
        "demo.count": torch.zeros((), dtype=torch.float64),
    }
    drivers = {
        "c.q": torch.tensor([1.0], dtype=torch.float64),
        "c.x_boundary": torch.zeros(1, dtype=torch.float64),
        "demo.count": torch.tensor(99.0, dtype=torch.float64),  # a DRIVER, same name
    }
    with pytest.raises(KeyError, match=r"_SilentClosure.*demo\.count.*did not return it"):
        model.step(state, drivers, 1.0)


def test_a_closure_may_declare_the_flow_driver_of_an_ownerless_layer_as_state():
    """The construction-time state_keys/layer-state collision check must mirror
    `_apply_closures`' own ownerless-"<layer>.q" carve-out -- "<layer>.q" for a transport
    layer no Kirchhoff layer owns is a DRIVER (what a flow closure writes), not that
    layer's state, so a closure may legitimately declare it as state it carries across
    steps (a flow closure that also remembers its own last-written flow)."""

    class _RememberedFlow:
        state_keys = ("c.q",)
        integrates = False  # passes its own last-written flow through unchanged.

        def __call__(self, state, drivers):
            return {"c.q": state["c.q"]}

    net = Network(dtype=torch.float64)
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="pipe")
    layer = TransportLayer(
        net, "c", capacity=torch.tensor([2.0], dtype=torch.float64), flow_kind="pipe",
        boundary=[net.nodes[1]], scheme="implicit",
    )
    # Construction must NOT raise "which is layer 'c''s own state key".
    model = Model(net, {"c": layer}, closures=[_RememberedFlow()])
    assert model.flow_layer_of["c"] is None
    state = {
        "c.x": torch.zeros(1, dtype=torch.float64),
        "c.q": torch.tensor([1.0], dtype=torch.float64),
    }
    drivers = {"c.x_boundary": torch.zeros(1, dtype=torch.float64)}
    new = model.step(state, drivers, 1.0)
    torch.testing.assert_close(new["c.q"], state["c.q"])


class _IntegratingCounter:
    """A closure that carries a scalar it advances by a FIXED amount every call, mimicking
    a stateful closure that integrates over time (a sewer manhole storage sweep, a tank
    level): the counter-closure for the step-start pinning rule.

    `integrates = False` here even though the docstring says "integrates" -- the
    increment is a constant fixed at CONSTRUCTION (the test passes it the step's own `dt`
    so the numbers line up), never read off a `StepContext`, so this closure does not
    consult the model's own clock and takes no `ctx`. Pass-vs-step (exercised here) and
    whose clock a closure reads are independent concerns.
    """

    state_keys = ("demo.n",)
    integrates = False

    def __init__(self, increment: float) -> None:
        self.increment = increment

    def __call__(self, state, drivers):
        return {"demo.n": state["demo.n"] + self.increment}


def test_closure_carried_state_advances_once_per_step_not_once_per_pass():
    """Under coupling='iterate', a closure-carried state key must be evaluated from the
    STEP-START state on EVERY pass, so one model.step(dt) advances it by exactly the
    closure's own per-call increment -- never by (number of passes) * increment, which is
    what `_iterate` fed pass k-1's own output back into pass k used to produce."""
    dt = 600.0
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3), _IntegratingCounter(dt)],
        coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
    )
    start = dict(state, **{"demo.n": torch.zeros((), dtype=F64)})
    diag: dict = {}
    new = model.step(start, drivers, dt, diagnostics=diag)
    assert diag["passes"] >= 2   # the bug requires more than one pass to be visible at all
    assert float(new["demo.n"]) == pytest.approx(dt, rel=0.0, abs=1e-9)


class _ParametrizedIntegratingCounter:
    """Like `_IntegratingCounter`, but the per-call increment is a differentiable PARAMETER
    rather than a fixed float, and the call tolerates its declared key being absent from the
    state it is handed: the edge case tested below is exactly a step-start state that never seeds
    "demo.n" at all, so `__call__` must not do the bare `state["demo.n"]` lookup
    `_IntegratingCounter` does."""

    state_keys = ("demo.n",)
    integrates = False

    def __init__(self, increment: torch.Tensor) -> None:
        self.increment = increment

    def __call__(self, state, drivers):
        n = state.get("demo.n")
        if n is None:
            n = torch.zeros((), dtype=F64)
        return {"demo.n": n + self.increment}


def _iterate_model_with_counter(param=None):
    """Shared builder for the missing-key tests (a closure-carried key missing vs. seeded from
    the step-start state) and the warning tests, all on the same `_Feedback` +
    `_ParametrizedIntegratingCounter` iterate model. `state` is SEEDED ("demo.n" zero): the
    warning's control test wants that, and its warning test derives the unseeded state by
    dropping the key from it, exactly as the gradient control test builds its own `start`.
    `param` defaults to a fixed, non-differentiable 0.5 for the warning tests, which never
    differentiate; the gradient tests each pass their own differentiable leaf so
    `torch.autograd.grad` can read it back afterward."""
    if param is None:
        param = torch.tensor(0.5, dtype=F64)
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3), _ParametrizedIntegratingCounter(param)],
        coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
    )
    state = dict(state, **{"demo.n": torch.zeros((), dtype=F64)})
    return model, state, drivers


def test_a_closure_state_key_missing_from_the_step_start_state_gets_a_truncated_gradient():
    """`_iterate`'s interface excludes every `closure_state_keys` entry
    UNCONDITIONALLY (`keys = [... if k not in self.closure_state_keys ...]`), whether or not
    the `if key in base:` pinning applies to it. When the key IS seeded (see the control
    below) that costs nothing: the pinning holds the closure's read to the step-start state on every
    pass, and the one differentiable re-run of the certified pass reads that same
    graph-carrying tensor directly (`step_from=state`), so the gradient survives untouched.

    When the key is ABSENT from the step-start state -- the one case that is not pinned,
    documented in `_iterate`'s own docstring ("a closure-carried key MISSING from the
    step-start state ... drops a path that only exists because that key was not seeded in the
    first place") -- the closure instead reads the PREVIOUS pass's own value on every pass, so
    "demo.n" genuinely COMPOUNDS the parameter once per PASS, and this fixture's true,
    central-difference sensitivity equals its own pass count. The one differentiable pass the
    adjoint runs sees that compounded history only as a plain, already-detached number and
    re-applies the closure ONCE more, so it can return only THAT one application's own local
    derivative (exactly 1.0 here) -- not the true, pass-count-sized sensitivity, and,
    measured here, not literally zero either: the single surviving local term is what this
    test pins, verified by directly running this fixture before writing the assertion.
    Either way this is the documented SAFE side of the trade
    (silently wrong-by-omission, never wrong-signed or blown up) and NOT a runtime refusal --
    the first step is allowed to lack the key by design (`_pass`'s `if key in base:` guard)."""
    param = torch.tensor(0.5, dtype=F64, requires_grad=True)
    model, state, drivers = _iterate_model_with_counter(param)
    unseeded = {k: v for k, v in state.items() if k != "demo.n"}
    diag: dict = {}
    with pytest.warns(RuntimeWarning, match=r"closure-carried state.*'demo\.n'.*once per pass"):
        new = model.step(unseeded, drivers, 600.0, diagnostics=diag)   # "demo.n" NOT seeded
    assert diag["passes"] >= 2      # the compounding needs more than one pass to be visible
    assert diag["adjoint"] == "implicit"
    (grad,) = torch.autograd.grad(new["demo.n"], (param,))
    assert grad.item() == pytest.approx(1.0, rel=0.0, abs=1e-9)   # the single local term only

    def _unseeded_value(pval: float) -> float:
        p = torch.tensor(pval, dtype=F64)
        _, m, s, d, _, _ = _build(
            closures=[_Feedback(2e3), _ParametrizedIntegratingCounter(p)],
            coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
        )
        with torch.no_grad():
            return m.step(s, d, 600.0)["demo.n"].item()

    h = 1e-6
    warn_match = r"closure-carried state.*'demo\.n'.*once per pass"
    with pytest.warns(RuntimeWarning, match=warn_match):
        up = _unseeded_value(0.5 + h)
    with pytest.warns(RuntimeWarning, match=warn_match):
        down = _unseeded_value(0.5 - h)
    cd = (up - down) / (2 * h)
    assert cd == pytest.approx(float(diag["passes"]), rel=1e-6)   # true sensitivity ~ pass count
    assert grad.item() < 0.2 * cd   # the adjoint drops nearly all of that true sensitivity


def test_control_the_same_gradient_matches_central_differences_once_seeded():
    """Control for the test above: the identical closure and parameter, seeded this time, so
    "demo.n" is pinned to the step-start state on every pass and the gradient the implicit
    adjoint returns is the fixed point's own, matching central differences the way every other
    `coupling="iterate"` gradient test in this module does."""
    param = torch.tensor(0.5, dtype=F64, requires_grad=True)
    model, start, drivers = _iterate_model_with_counter(param)
    diag: dict = {}
    new = model.step(start, drivers, 600.0, diagnostics=diag)
    assert diag["adjoint"] == "implicit"
    (grad,) = torch.autograd.grad(new["demo.n"], (param,))

    def _seeded_value(pval: float) -> float:
        p = torch.tensor(pval, dtype=F64)
        _, m, s, d, _, _ = _build(
            closures=[_Feedback(2e3), _ParametrizedIntegratingCounter(p)],
            coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
        )
        s = dict(s, **{"demo.n": torch.zeros((), dtype=F64)})
        with torch.no_grad():
            return m.step(s, d, 600.0)["demo.n"].item()

    h = 1e-6
    cd = (_seeded_value(0.5 + h) - _seeded_value(0.5 - h)) / (2 * h)
    assert grad.item() == pytest.approx(cd, rel=1e-6)


def test_iterate_warns_when_a_closure_carried_key_is_missing_from_the_step_start_state():
    """On such a step the key is not pinned to the step start (`if key in base`), so
    its closure integrates once per PASS and the adjoint omits the compounding path. The
    misconfiguration is named where it happens; it is not refused, because the first step is
    allowed to lack the key by design.

    `stacklevel` is pinned here too, not just the message text: the warning must be attributed
    to the CALLER of `step`/`steady` (this test module), not to a line inside `model.py` itself
    -- `warnings.warn`'s default filter dedupes on (message, category, module, lineno), so a
    warning permanently attributed to one internal line would let a second, later, genuinely
    different caller's identical warning go silently missing. Both `step` and
    `steady` reach `_iterate` through the same `_advance`, so one `stacklevel` serves both --
    checked on both calls below."""
    model, state, drivers = _iterate_model_with_counter()   # the missing-key tests' builder
    unseeded = {k: v for k, v in state.items() if k != "demo.n"}
    match = r"closure-carried state.*'demo\.n'.*once per pass"
    with pytest.warns(RuntimeWarning, match=match) as record:
        model.step(unseeded, drivers, 600.0)
    assert record[0].filename.endswith("test_model.py")
    with pytest.warns(RuntimeWarning, match=match) as record:
        model.steady(unseeded, drivers)
    assert record[0].filename.endswith("test_model.py")


def test_iterate_does_not_warn_when_the_key_is_seeded():
    model, state, drivers = _iterate_model_with_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.step(state, drivers, 600.0)


def test_current_flows_returns_closure_written_driver_prescribed_flow():
    """A transport layer with no owning Kirchhoff layer: its flow is written by a closure,
    only visible after closures run -- `current_flows` must run them and return it."""
    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    layer = TransportLayer(
        net, "x", capacity=torch.tensor([1.0], dtype=torch.float64), flow_kind="link",
        boundary=["b"],
    )

    def closure(state, drivers):
        return {"x.q": torch.tensor([2.5], dtype=torch.float64)}

    model = Model(net, {"x": layer}, closures=[closure])
    state = {"x.x": torch.tensor([0.0], dtype=torch.float64)}
    drivers = {"x.x_boundary": torch.zeros(1, dtype=torch.float64)}
    flows = model.current_flows("x", state, drivers)
    assert torch.equal(flows, torch.tensor([2.5], dtype=torch.float64))


def test_current_flows_returns_kirchhoff_owned_flow():
    """A transport layer whose flow kind is owned by a Kirchhoff layer: `current_flows`
    must read it from the SOLVED state, not require a driver."""
    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    from noodl.elements.conductance import Conductance

    air = KirchhoffLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
    species = TransportLayer(
        net, "x", capacity=torch.tensor([1.0], dtype=torch.float64), flow_kind="link",
        boundary=["b"],
    )
    model = Model(net, {"air": air, "x": species})
    solved_q = torch.tensor([0.7], dtype=torch.float64)
    state = {"x.x": torch.tensor([0.0], dtype=torch.float64), "air.q": solved_q}
    drivers = {"air.phi_boundary": torch.zeros(1, dtype=torch.float64),
               "x.x_boundary": torch.zeros(1, dtype=torch.float64)}
    flows = model.current_flows("x", state, drivers)
    assert torch.equal(flows, air.flows_of_kind(solved_q, species.flow_kinds))


def test_current_flows_solves_a_kirchhoff_owner_when_the_state_has_no_q_yet():
    """The first pass of a step: the start state carries no "<owner>.q" (project_to_model's
    initial state has only "species.x"), so current_flows must SOLVE the owner for these
    drivers, and agree with what a step would have solved."""
    from noodl.elements.conductance import Conductance

    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    air = KirchhoffLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
    species = TransportLayer(
        net, "x", capacity=torch.tensor([1.0], dtype=torch.float64), flow_kind="link",
        boundary=["b"],
    )
    model = Model(net, {"air": air, "x": species})
    state = {"x.x": torch.tensor([0.0], dtype=torch.float64)}
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
        "air.sources": torch.tensor([0.0, 0.3], dtype=torch.float64),  # 0.3 injected at "i"
        "x.x_boundary": torch.zeros(1, dtype=torch.float64),
    }
    flows = model.current_flows("x", state, drivers)
    stepped = model.step(state, drivers, 1.0)
    assert torch.allclose(flows, air.flows_of_kind(stepped["air.q"], species.flow_kinds))
    assert flows.item() == pytest.approx(0.3)  # the injected 0.3 must leave through the edge


def test_current_flows_refuses_a_flow_driver_beside_a_kirchhoff_owner_on_the_solve_path():
    """The two-sources-of-flows refusal must not depend on whether the state happens to carry
    a solved q yet: `_kind_flows` makes it on the read path, and the solve path makes it too."""
    from noodl.elements.conductance import Conductance

    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    air = KirchhoffLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
    species = TransportLayer(
        net, "x", capacity=torch.tensor([1.0], dtype=torch.float64), flow_kind="link",
        boundary=["b"],
    )
    model = Model(net, {"air": air, "x": species})
    state = {"x.x": torch.tensor([0.0], dtype=torch.float64)}  # no "air.q" -> solve path
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
        "x.x_boundary": torch.zeros(1, dtype=torch.float64),
        "x.q": torch.tensor([0.9], dtype=torch.float64),  # contradicts the Kirchhoff layer
    }
    with pytest.raises(ValueError, match="one source of flows, not two"):
        model.current_flows("x", state, drivers)


def test_iterate_gradient_from_a_near_fixed_point_start_matches_central_differences():
    """Start the step at the model's own steady state: the primal needs only the structural
    two passes, and the derivative it returns must still be the fixed point's.

    Two things this test needs that the fixture's own defaults do not give, both measured
    while it was written -- and neither of them a loosened assertion:

    * A FEEDBACK WORTH DIFFERENTIATING. With the fixture's 2e-6 kg/s source nearly all of
      z1's mass fraction is the boundary's own 1e-3, which no amount of wind moves, so the
      pass map contracts at dx_new/dx_fed = -0.004 and even a two-pass unrolling is right to
      1.2e-5 (~ that ratio squared): the bug would be invisible here whatever the tolerance.
      The source is raised to 5e-4 kg/s, where the loop gain is -0.23, the two-pass
      derivative is 3.2 % wrong and the implicit one is right to 3e-7.
    * Tight Newton tolerances. The CENTRAL DIFFERENCE is the reference,
      and at the default (~1.5e-8) each perturbed run chases the solver's own noise through
      several extra passes; the drift that leaves in (up - down) does not scale with h, so
      the measured "derivative" swings by 9 % between h=1e-6 and h=1e-5 and is no reference
      at all. Tight, it is stable to seven digits across h=1e-5..1e-7. The solve is asked
      for the accuracy the reference needs; the assertion below is untouched.
    """
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, el, _ = _build(
        learnable=True, closures=[_Feedback(2e3)], coupling="iterate",
        iterate_tol={"species": 1e-13}, iterate_max=80,
    )
    drivers = dict(drivers, **{"species.sources": torch.tensor([0.0, 5e-4, 0.0], dtype=F64)})
    with torch.no_grad():
        ss = model.steady(state, drivers, differentiable=False, **tight)
    start = {k: v.detach().clone() for k, v in ss.items()}

    def loss():
        return model.step(start, drivers, 600.0, **tight)["species.x"].sum()

    diag: dict = {}
    model.step(start, drivers, 600.0, diagnostics=diag, **tight)
    assert diag["passes"] == 2          # the structural floor: the start IS the fixed point
    assert diag["adjoint"] == "implicit"
    el.C.grad = None
    loss().backward()
    grad = el.C.grad[0].item()
    h = 1e-6
    with torch.no_grad():
        el.C[0] += h
        up = loss().item()
        el.C[0] -= 2 * h
        down = loss().item()
        el.C[0] += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-5)


def test_iterate_pays_for_the_adjoint_pass_only_when_a_gradient_is_wanted():
    """`passes` counts PRIMAL passes. A run nothing differentiable reaches costs exactly
    those and reports `adjoint is None`; a differentiable one costs ONE more -- the single
    pass the implicit adjoint is attached to -- and never one per pass."""
    tight = {"atol": 1e-14, "rtol": 1e-14}
    _, model, state, drivers, el, _ = _build(
        learnable=True, closures=[_Feedback(2e3)], coupling="iterate",
        iterate_tol={"species": 1e-13}, iterate_max=80,
    )
    calls: list[int] = []
    real = model._pass

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    model._pass = counting                      # shadows the bound method on this instance
    for kwargs, expect, extra in (
        ({}, None, 0),                          # under no_grad below: nothing to attach
        # Detached solves: nothing reaches the state on pass 1, but the learnable law IS a
        # tensor requiring grad, so one pass at the certified point confirms that nothing
        # reaches it there either.
        ({"differentiable": False}, None, 1),
    ):
        calls.clear()
        diag: dict = {}
        with torch.no_grad() if not kwargs else torch.enable_grad():
            model.steady(state, drivers, diagnostics=diag, **kwargs, **tight)
        assert diag["adjoint"] is expect
        assert len(calls) == diag["passes"] + extra
    calls.clear()
    diag = {}
    out = model.steady(state, drivers, diagnostics=diag, **tight)
    assert diag["adjoint"] == "implicit"
    assert len(calls) == diag["passes"] + 1
    assert out["species.x"].requires_grad
    out["species.x"].sum().backward()
    assert el.C.grad is not None


def test_iterate_gradient_is_per_instance_on_a_batched_model():
    """Two wind speeds through coupling="iterate" with a learnable
    element. Each instance's sensitivity must match its OWN central difference, so the adjoint
    solve keeps instances apart; the instances must also differ, or a batch-mixing error could
    hide. Sources at 5e-4 as in the near-fixed-point test, so the feedback gain is strong.
    Solver tolerances tightened for the central-difference reference."""
    _, model, state, drivers, el, _ = _build(
        learnable=True, closures=[_Feedback(2e3)], coupling="iterate",
        iterate_tol={"species": 1e-13}, iterate_max=80,
    )
    drv = dict(drivers)
    drv["wind"] = torch.tensor([[5.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=F64)
    drv["species.sources"] = torch.tensor([0.0, 5e-4, 0.0], dtype=F64)
    tight = {"atol": 1e-14, "rtol": 1e-14}

    def run():
        return model.step(state, drv, 600.0, **tight)["species.x"]      # (2, n_i)

    diag: dict = {}
    x = model.step(state, drv, 600.0, diagnostics=diag, **tight)["species.x"]
    assert x.shape == (2, 2)
    assert diag["adjoint"] == "implicit"
    assert diag["adjoint_batched"] is True
    assert diag["converged"].shape == (2,) and bool(diag["converged"].all())
    grads = []
    for i in range(2):
        el.C.grad = None
        x[i].sum().backward(retain_graph=True)
        grads.append(el.C.grad[0].item())
    h = 1e-6
    with torch.no_grad():
        el.C[0] += h
        up = run().sum(-1)
        el.C[0] -= 2 * h
        down = run().sum(-1)
        el.C[0] += h
    cd = ((up - down) / (2 * h)).tolist()
    for i in range(2):
        assert grads[i] == pytest.approx(cd[i], rel=1e-5), i
    assert grads[0] != pytest.approx(grads[1], rel=1e-3)


def test_a_non_converged_iteration_refuses_to_return_when_a_gradient_is_wanted():
    """on_failure="return" keeps its meaning for forward-only and differentiable=False runs
    (`test_iterate_reports_non_convergence_per_instance_or_raises`, unlearnable so
    `needs_adjoint` stays False there), but a non-converged iteration has no fixed point to
    differentiate, and returning the primal state there would hand the caller a graph-free
    tensor that differentiates to nothing, silently.

    Why not the obvious fixture: `model.steady(..., on_failure="return")` with a
    `learnable=True` element cannot reach `_iterate`'s own non-convergence branch with
    `needs_adjoint` True, because `solve_kwargs` (including `on_failure`) is forwarded to
    every Kirchhoff-layer solve too (module docstring), and `KirchhoffLayer.solve`'s
    differentiable path (the default) refuses `on_failure="return"` UNCONDITIONALLY
    (`solvers.implicit.implicit_solve`, not only on the potential solve's own
    non-convergence) -- verified by running: it raises `ValueError` from `implicit_solve`
    before `_iterate` is ever reached. `differentiable=False` keeps the potential solve on
    its non-differentiable path instead (which does accept `on_failure="return"` given
    `diagnostics`), and the gradient reaches "species.x" directly through a `requires_grad`
    `species.sources` driver, which the transport step differentiates in plain autograd,
    independently of the potential solve."""
    _, model, state, drivers, _el, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate",
        iterate_tol={"species": 1e-15}, iterate_max=2,
    )
    drv = dict(drivers)
    drv["species.sources"] = drivers["species.sources"].clone().requires_grad_(True)
    diag: dict = {}
    with pytest.raises(RuntimeError, match=r"did not converge.*no fixed point to differentiate"):
        model.steady(state, drv, diagnostics=diag, differentiable=False, on_failure="return")
    with torch.no_grad():
        out = model.steady(state, drv, diagnostics=diag, differentiable=False,
                            on_failure="return")
    assert "species.x" in out and not bool(diag["converged"].all())


class _DriverConductance(ConstitutiveLaw):
    """q = k * dp with k per edge from the driver "toy.k" (test-local)."""

    def __init__(self, kind):
        super().__init__(kind)

    def flow(self, dp, drivers=None):
        return drivers["toy.k"] * dp

    def dflow(self, dp, drivers=None):
        return drivers["toy.k"].expand_as(dp)


class _HeadDependentConductance:
    """Closure: k = k0 * (1 + phi_upstream / 10), read from the FED state -- a Picard
    coupling between a Kirchhoff layer and its own coefficients."""

    def __init__(self, src, k0):
        self.src, self.k0 = src, k0

    def __call__(self, state, drivers):
        phi = state["toy.phi"]
        return {"toy.k": self.k0 * (1.0 + phi[..., self.src] / 10.0)}


def _toy(k0=1.0, **model_kwargs):
    net = Network(dtype=torch.float64)
    for n in ("a", "b", "c"):
        net.add_node(n)
    net.add_edge("a", "b", kind="link")
    net.add_edge("b", "c", kind="link")
    layer = KirchhoffLayer(net, "toy", [_DriverConductance("link")], boundary=["c"])
    src, _ = net.endpoints("link")
    k0 = torch.as_tensor(k0, dtype=torch.float64)
    closure = _HeadDependentConductance(src, k0)
    model = Model(net, {"toy": layer}, closures=[closure], **model_kwargs)
    state = {"toy.phi": torch.zeros(3, dtype=torch.float64),
             "toy.q": torch.zeros(2, dtype=torch.float64)}
    drivers = {"toy.phi_boundary": torch.zeros(1, dtype=torch.float64),
               "toy.sources": torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)}
    return model, state, drivers, closure


@pytest.mark.parametrize("acceleration", [None, "anderson"])
def test_accepting_a_non_converged_iterate_refuses_a_gradient_alike_for_every_scheme(
        acceleration):
    """`iterate_on_limit="accept"` returns the last pass of a run that did not converge, but
    a gradient there is refused by name, with plain relaxation and Anderson mixing alike:
    the derivative returned is the implicit one at a fixed point, which a non-converged
    pass is not. Without a gradient, the same run returns its last pass."""
    def run(k0):
        model, state, drivers, _ = _toy(
            k0, coupling="iterate", iterate_tol={"toy": 1e-13}, iterate_max=3,
            iterate_acceleration=acceleration, iterate_memory=1,
            iterate_flow_tol={"toy": 1e-12}, iterate_on_limit="accept")
        drivers["toy.sources"] = torch.tensor([20.0, 0.0, 0.0], dtype=torch.float64)
        diag: dict = {}
        return model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14), diag

    k0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match=(
            r"did not converge within 3 passes.*no fixed point to differentiate, so "
            r"iterate_on_limit='accept' cannot return a state here")):
        run(k0)
    out, diag = run(torch.tensor(1.0, dtype=torch.float64))
    assert diag["passes"] == 3 and not bool(diag["converged"].all())
    assert diag["adjoint"] is None and not out["toy.phi"].requires_grad


def test_a_step_about_to_raise_stops_early_once_the_potential_solve_stalls():
    """The stall rule: one Newton iteration per pass (`max_iter=1`) fails the air solve on
    every pass, and with a tolerance loose enough that the second pass is fed a state within
    it of the first's, the next pass would only repeat the failure -- so a step that is going
    to raise does so after 2 passes, not `iterate_max`. A step that returns instead
    (`on_failure="return"` with diagnostics) is not cut short: it runs to the limit."""
    from noodl.solvers.errors import ConvergenceError

    def build():
        return _build(closures=[_Feedback(2e3)], coupling="iterate",
                      iterate_tol={"species": 1.0}, iterate_max=10)

    _, model, state, drivers, _, _ = build()
    with pytest.raises(ConvergenceError, match=(
            r"within 2 passes \(of at most 10; stopped there, as the potential solve failed "
            r"again")):
        model.steady(state, drivers, differentiable=False, max_iter=1)
    _, model, state, drivers, _, _ = build()
    diag: dict = {}
    model.steady(state, drivers, diagnostics=diag, differentiable=False, max_iter=1,
                 on_failure="return")
    assert diag["passes"] == 10 and not bool(diag["converged"].all())


def test_iterate_tol_may_name_a_kirchhoff_layer():
    model, state, drivers, closure = _toy(
        coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_max=200)
    diag: dict = {}
    # tighten the inner Newton solve below its default floor so the 1e-10 checks are fair
    out = model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert bool(diag["converged"].all()) and diag["passes"] > 2
    # the returned state satisfies the NONLINEAR law q = k(phi) * dp
    k = closure(out, drivers)["toy.k"]
    phi = out["toy.phi"]
    src, tgt = model.net.endpoints("link")
    torch.testing.assert_close(out["toy.q"], k * (phi[src] - phi[tgt]),
                               rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out["toy.q"], torch.full((2,), 2.0, dtype=torch.float64),
                               rtol=1e-10, atol=1e-12)


def test_iterate_tol_naming_an_unknown_layer_lists_both_kinds():
    with pytest.raises(KeyError, match="Kirchhoff"):
        _toy(coupling="iterate", iterate_tol={"nope": 1e-9})


def test_iterate_on_a_kirchhoff_layer_differentiates_the_fixed_point():
    k0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    model, state, drivers, _ = _toy(k0=k0, coupling="iterate",
                                    iterate_tol={"toy": 1e-13}, iterate_max=300)

    def loss(k):
        model.closures[0].k0 = k
        return model.steady(state, drivers)["toy.phi"][0]

    loss(k0).backward()
    h = 1e-6
    with torch.no_grad():
        fd = (loss(k0 + h) - loss(k0 - h)) / (2 * h)
    assert k0.grad.item() == pytest.approx(fd.item(), rel=1e-6)


def test_unnamed_kirchhoff_layers_are_not_relaxed():
    """A model naming only a transport layer must behave exactly as before."""
    _, it, state, drivers, _, _ = _build(coupling="iterate", iterate_tol={"species": 1e-11})
    diag: dict = {}
    it.step(state, drivers, 600.0, diagnostics=diag)
    assert set(diag["max_change"]) == {"species"}


class _Ground(NodeSource):
    """w = g * phi at the given nodes (test-local): a stiff tie of a node to zero."""

    def __init__(self, nodes, g):
        super().__init__(nodes)
        self.g = g

    def flow(self, phi_nodes, drivers=None):
        return self.g * phi_nodes

    def dflow(self, phi_nodes, drivers=None):
        return torch.full_like(phi_nodes, self.g)


class _BoundaryFromState:
    """Closure: the boundary potential is 1 + phi_b / 2 of the FED state (fixed point 2)."""

    def __init__(self, b):
        self.b = b

    def __call__(self, state, drivers):
        return {"toy.phi_boundary": 1.0 + 0.5 * state["toy.phi"][..., self.b:self.b + 1]}


def test_iterate_on_a_kirchhoff_layer_converges_its_boundary_too():
    """A closure may write the boundary potentials from the previous pass (the sewer's
    outfall heads). The interior here barely sees the boundary (a stiff tie to zero), so a
    test on the interior alone would stop with the boundary still ~1e-6 from its fixed
    point."""
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="link")
    layer = KirchhoffLayer(net, "toy", [PowerLaw(C=torch.tensor(1.0, dtype=torch.float64),
                                                     n=torch.tensor(1.0, dtype=torch.float64),
                                                     kind="link")],
                           boundary=["b"], node_sources=[_Ground([0], 1e6)])
    model = Model(net, {"toy": layer}, closures=[_BoundaryFromState(1)], coupling="iterate",
                  iterate_tol={"toy": 1e-12}, iterate_max=500)
    state = {"toy.phi": torch.zeros(2, dtype=torch.float64),
             "toy.q": torch.zeros(1, dtype=torch.float64)}
    drivers = {"toy.sources": torch.tensor([1.0, 0.0], dtype=torch.float64)}
    out = model.steady(state, drivers, atol=1e-14, rtol=1e-14)
    assert abs(out["toy.phi"][1].item() - 2.0) < 1e-10


# --- Anderson acceleration of the iterated coupling -------------------------------------


def _toy_steady(acceleration, sources=(2.0, 0.0, 0.0), **kwargs):
    model, state, drivers, closure = _toy(
        coupling="iterate", iterate_tol={"toy": 1e-13}, iterate_max=400,
        iterate_acceleration=acceleration, iterate_flow_tol={"toy": 1e-12}, **kwargs)
    drivers["toy.sources"] = torch.tensor(sources, dtype=F64)
    if drivers["toy.sources"].dim() == 2:
        batch = drivers["toy.sources"].shape[:1]
        state = {k: v.expand(batch + v.shape).clone() for k, v in state.items()}
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
    return out, diag


def test_iterate_acceleration_arguments_are_refused_by_name():
    with pytest.raises(ValueError, match="iterate_acceleration"):
        _toy(coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_acceleration="broyden")
    with pytest.raises(ValueError, match="iterate_memory"):
        _toy(coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_acceleration="anderson",
             iterate_memory=0)
    # acceleration belongs to the iterated coupling only
    with pytest.raises(ValueError, match="coupling='iterate'"):
        _toy(iterate_acceleration="anderson")
    # ... and to a Kirchhoff layer named in iterate_tol
    with pytest.raises(ValueError, match="Kirchhoff layer"):
        _build(coupling="iterate", iterate_tol={"species": 1e-11},
               iterate_acceleration="anderson")
    # ... whose flow tolerance it needs
    with pytest.raises(ValueError, match=r"iterate_flow_tol for the .* \['toy'\]"):
        _toy(coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_acceleration="anderson")
    with pytest.raises(KeyError, match="iterate_flow_tol names"):
        _toy(coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_flow_tol={"nope": 1e-9})
    with pytest.raises(ValueError, match="iterate_flow_tol must be positive"):
        _toy(coupling="iterate", iterate_tol={"toy": 1e-12}, iterate_flow_tol={"toy": 0.0})


def test_iterate_flow_tol_judges_the_flows_without_acceleration():
    """A flow tolerance alone (plain relaxation) adds the flows' fixed-point residual to the
    convergence test and reports it under "<layer>.q"."""
    model, state, drivers, _ = _toy(coupling="iterate", iterate_tol={"toy": 1e-13},
                                    iterate_max=400, iterate_flow_tol={"toy": 1e-12})
    diag: dict = {}
    model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert set(diag["max_change"]) == {"toy", "toy.q"}
    assert float(diag["max_change"]["toy.q"]) <= 1e-12


def test_anderson_reaches_the_relaxed_fixed_point_in_fewer_passes():
    """Anderson mixing changes only the sequence of fed states, not the fixed point."""
    plain, d_plain = _toy_steady(None, sources=(20.0, 0.0, 0.0))
    fast, d_fast = _toy_steady("anderson", sources=(20.0, 0.0, 0.0))
    assert bool(d_fast["converged"].all())
    assert d_fast["passes"] < d_plain["passes"]
    for key in ("toy.phi", "toy.q"):
        torch.testing.assert_close(fast[key], plain[key], rtol=0.0, atol=1e-12)


def test_anderson_leaves_the_fixed_point_gradient_unchanged():
    """The adjoint stays on the unrelaxed certified pass: same gradient either way."""
    grads = []
    for acceleration in (None, "anderson"):
        k0 = torch.tensor(1.0, dtype=F64, requires_grad=True)
        model, state, drivers, _ = _toy(k0=k0, coupling="iterate", iterate_tol={"toy": 1e-13},
                                        iterate_max=300, iterate_acceleration=acceleration,
                                        iterate_flow_tol={"toy": 1e-12})
        drivers["toy.sources"] = torch.tensor([20.0, 0.0, 0.0], dtype=F64)
        model.steady(state, drivers, atol=1e-14, rtol=1e-14)["toy.phi"][0].backward()
        grads.append(k0.grad.item())
    assert grads[1] == pytest.approx(grads[0], rel=1e-9)


def test_anderson_is_per_instance_under_batching():
    """Each instance lands where it lands alone, including one that converges passes before
    the other (it is held at its certified fed state, not mixed further)."""
    sources = [[0.5, 0.0, 0.0], [40.0, 0.0, 0.0]]
    batched, d_b = _toy_steady("anderson", sources=sources)
    passes = []
    for i, s in enumerate(sources):
        single, d_s = _toy_steady("anderson", sources=s)
        passes.append(d_s["passes"])
        for key in ("toy.phi", "toy.q"):
            torch.testing.assert_close(batched[key][i], single[key], rtol=0.0, atol=1e-12)
    assert passes[0] < passes[1]
    assert d_b["passes"] == max(passes)


def test_anderson_mix_drops_a_collinear_history_and_solves_a_linear_map():
    """An ill-conditioned least-squares problem (here: identical residual differences)
    keeps no column and says so, leaving the fallback to the caller; a well-conditioned
    history of a linear map is solved exactly once it spans the map."""
    from noodl.solvers.anderson import _anderson_mix

    x = [torch.tensor([0.0, 0.0], dtype=F64), torch.tensor([1.0, 1.0], dtype=F64)]
    g = [torch.tensor([1.0, 1.0], dtype=F64), torch.tensor([1.0, 1.0], dtype=F64)]
    r = [torch.tensor([1.0], dtype=F64), torch.tensor([1.0], dtype=F64)]   # dR = 0
    _, kept = _anderson_mix(x, g, r)
    assert not bool(kept)
    # G(x) = A x + b
    a = torch.tensor([[0.5, 0.2], [0.1, 0.3]], dtype=F64)
    b = torch.tensor([1.0, -1.0], dtype=F64)
    xs = [torch.zeros(2, dtype=F64)]
    gs, rs = [], []
    for _ in range(3):
        gs.append(a @ xs[-1] + b)
        rs.append(gs[-1] - xs[-1])
        xs.append(_anderson_mix(xs, gs, rs)[0])
    exact = torch.linalg.solve(torch.eye(2, dtype=F64) - a, b)
    torch.testing.assert_close(xs[-1], exact, rtol=0.0, atol=1e-12)


class _RootPair:
    """Closure: k = 1 / (0.2 + phi_a^2), so with a unit source at `a` and the boundary at 0
    the pass maps phi_a -> 0.2 + phi_a^2. Fixed points 0.276 (attracting: slope 0.55) and
    0.724 (repelling: slope 1.45)."""

    def __call__(self, state, drivers):
        phi_a = state["toy.phi"][..., :1]
        return {"toy.k": 1.0 / (0.2 + phi_a * phi_a)}


def test_anderson_keeps_the_fixed_point_the_relaxed_iteration_is_attracted_to():
    """From phi_a = 0.6 the relaxed iteration falls to the attracting root; an unguarded
    secant step overshoots past the repelling one. The safeguard (no step against an
    entry's own residual) keeps Anderson on the relaxed iteration's root."""
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("c")
    net.add_edge("a", "c", kind="link")
    root = (1.0 - 0.2 ** 0.5) / 2.0
    for acceleration in (None, "anderson"):
        layer = KirchhoffLayer(net, "toy", [_DriverConductance("link")], boundary=["c"])
        model = Model(net, {"toy": layer}, closures=[_RootPair()], coupling="iterate",
                      iterate_tol={"toy": 1e-13}, iterate_max=400,
                      iterate_acceleration=acceleration, iterate_flow_tol={"toy": 1e-12})
        state = {"toy.phi": torch.tensor([0.6, 0.0], dtype=F64),
                 "toy.q": torch.zeros(1, dtype=F64)}
        drivers = {"toy.phi_boundary": torch.zeros(1, dtype=F64),
                   "toy.sources": torch.tensor([1.0, 0.0], dtype=F64)}
        out = model.steady(state, drivers, atol=1e-14, rtol=1e-14)
        assert abs(out["toy.phi"][0].item() - root) < 1e-10, acceleration


class _BranchingSource:
    """Closure on a one-node transport layer with unit removal and no flow: the source is
    the constant 2 for x <= 1 and `theta + 0.1 x` above, so the steady state is
    x = theta / 0.9 (10/3 at theta = 3, on the parameter branch) with dx/dtheta = 10/9."""

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, state, drivers):
        x = state["c.x"].reshape(-1)[0]
        zero = torch.zeros((), dtype=F64)
        source = 2.0 + zero if float(x.detach()) <= 1.0 else self.theta + 0.1 * x
        return {"c.sources": torch.stack([zero, source]), "c.q": torch.zeros(1, dtype=F64)}


def _branching_steady(theta, x0):
    net = Network(dtype=F64)
    net.add_node("b")
    net.add_node("a")
    net.add_edge("a", "b", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([1.0], dtype=F64),
                           flow_kind="flow", boundary=["b"],
                           removal=torch.tensor([1.0], dtype=F64))
    model = Model(net, {"c": layer}, closures=[_BranchingSource(theta)], coupling="iterate",
                  iterate_tol={"c": 1e-13}, iterate_max=200)
    drivers = {"c.x_boundary": torch.zeros(1, dtype=F64),
               "c.sources": torch.zeros(2, dtype=F64)}
    return model.steady({"c.x": torch.tensor([x0], dtype=F64)}, drivers)["c.x"].sum()


@pytest.mark.parametrize("x0", [0.0, 2.0])
def test_iterate_gradient_does_not_depend_on_the_branch_the_first_pass_took(x0):
    """Whether the returned state carries a gradient is decided at the converged point, not
    by the branch a closure took on pass 1: from x0 = 0 the first pass takes the constant
    branch and reads no parameter at all, yet the fixed point lies on the parameter branch.
    Both initial guesses must return the same value and the same derivative."""
    theta = torch.tensor(3.0, dtype=F64, requires_grad=True)
    x = _branching_steady(theta, x0)
    assert x.item() == pytest.approx(10.0 / 3.0, rel=1e-10)
    assert x.requires_grad
    (g,) = torch.autograd.grad(x, (theta,))
    assert g.item() == pytest.approx(10.0 / 9.0, rel=1e-8)
    h = 1e-6
    with torch.no_grad():
        fd = (_branching_steady(torch.tensor(3.0 + h, dtype=F64), x0)
              - _branching_steady(torch.tensor(3.0 - h, dtype=F64), x0)) / (2 * h)
    assert g.item() == pytest.approx(fd.item(), rel=1e-6)


def _counting(model):
    calls: list[int] = []
    real = model._pass

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    model._pass = counting                      # shadows the bound method on this instance
    return calls


def test_iterate_pays_one_confirming_pass_in_grad_mode_and_none_under_no_grad():
    """Grad mode on (PyTorch's default) but nothing requires grad: pass 1 finds no
    dependency, so one uncounted probe pass at the certified point confirms it (no
    inspection can prove that nothing a pass reads requires grad). Under `torch.no_grad()`
    the run costs exactly its primal passes."""
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3)], coupling="iterate", iterate_tol={"species": 1e-13},
        iterate_max=80,
    )
    calls = _counting(model)
    diag: dict = {}
    assert torch.is_grad_enabled()
    model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert diag["adjoint"] is None and diag["passes"] >= 2
    assert len(calls) == diag["passes"] + 1
    calls.clear()
    diag = {}
    with torch.no_grad():
        model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert diag["adjoint"] is None and len(calls) == diag["passes"]


_BRANCH_SOURCE = """
import torch
F64 = torch.float64
def closure(state, drivers):
    x = state["c.x"].reshape(-1)[0]
    zero = torch.zeros((), dtype=F64)
    source = 2.0 + zero if float(x.detach()) <= 1.0 else {theta} + 0.1 * x
    return {{"c.sources": torch.stack([zero, source]), "c.q": torch.zeros(1, dtype=F64)}}
"""


def _closure_in_a_fresh_module(monkeypatch, name, theta_expr, **globals_):
    """`closure` defined by exec in a module with no `__file__`, registered in
    `sys.modules` for the test (as `__main__` is in a REPL, Jupyter or `python -c`), its
    globals seeded with `globals_`."""
    module = types.ModuleType(name)
    module.__dict__.update(globals_)
    monkeypatch.setitem(sys.modules, name, module)
    exec(_BRANCH_SOURCE.format(theta=theta_expr), module.__dict__)
    return module.closure


def _steady_with(closure, x0):
    net = Network(dtype=F64)
    net.add_node("b")
    net.add_node("a")
    net.add_edge("a", "b", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([1.0], dtype=F64),
                           flow_kind="flow", boundary=["b"],
                           removal=torch.tensor([1.0], dtype=F64))
    model = Model(net, {"c": layer}, closures=[closure], coupling="iterate",
                  iterate_tol={"c": 1e-13}, iterate_max=200)
    drivers = {"c.x_boundary": torch.zeros(1, dtype=F64),
               "c.sources": torch.zeros(2, dtype=F64)}
    return model.steady({"c.x": torch.tensor([x0], dtype=F64)}, drivers)["c.x"].sum()


@pytest.mark.parametrize("x0", [0.0, 2.0])
def test_r01_a_closure_defined_where_its_module_has_no_file_keeps_its_gradient(x0,
                                                                                 monkeypatch):
    """Regression: a closure exec'd in a module without `__file__` (REPL/Jupyter) reading a
    global parameter only on the branch the fixed point lies on."""
    theta = torch.tensor(3.0, dtype=F64, requires_grad=True)
    closure = _closure_in_a_fresh_module(monkeypatch, "noodl_r01_repl", "theta", theta=theta)
    x = _steady_with(closure, x0)
    assert x.item() == pytest.approx(10.0 / 3.0, rel=1e-10) and x.requires_grad
    (g,) = torch.autograd.grad(x, (theta,))
    assert g.item() == pytest.approx(10.0 / 9.0, rel=1e-8)


@pytest.mark.parametrize("x0", [0.0, 2.0])
def test_r01_a_closure_reading_a_parameter_from_a_user_module_keeps_its_gradient(
        x0, monkeypatch):
    """Regression: the parameter lives on a user module (`params.theta`), which the closure
    reads through a module attribute only on the branch the fixed point lies on."""
    params = types.ModuleType("noodl_r01_params")
    params.theta = torch.tensor(3.0, dtype=F64, requires_grad=True)
    closure = _closure_in_a_fresh_module(monkeypatch, "noodl_r01_user", "params.theta",
                                         params=params)
    x = _steady_with(closure, x0)
    assert x.item() == pytest.approx(10.0 / 3.0, rel=1e-10) and x.requires_grad
    (g,) = torch.autograd.grad(x, (params.theta,))
    assert g.item() == pytest.approx(10.0 / 9.0, rel=1e-8)


def test_iterate_pays_the_confirming_pass_when_a_parameter_is_unused_on_pass_one():
    """`theta` requires grad but pass 1 (x0 = 0, the constant branch) never reads it: one
    pass at the certified point finds it, then one more carries the adjoint."""
    theta = torch.tensor(3.0, dtype=F64, requires_grad=True)
    net = Network(dtype=F64)
    net.add_node("b")
    net.add_node("a")
    net.add_edge("a", "b", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([1.0], dtype=F64),
                           flow_kind="flow", boundary=["b"],
                           removal=torch.tensor([1.0], dtype=F64))
    model = Model(net, {"c": layer}, closures=[_BranchingSource(theta)], coupling="iterate",
                  iterate_tol={"c": 1e-13}, iterate_max=200)
    calls = _counting(model)
    diag: dict = {}
    drivers = {"c.x_boundary": torch.zeros(1, dtype=F64),
               "c.sources": torch.zeros(2, dtype=F64)}
    model.steady({"c.x": torch.tensor([0.0], dtype=F64)}, drivers, diagnostics=diag)
    assert diag["adjoint"] == "implicit"
    assert len(calls) == diag["passes"] + 2


# ---- S01: closure declarations are read once, at construction, and malformed ones refused ---


def _closure_with(**attrs):
    def closure(state, drivers):
        return {}

    for k, v in attrs.items():
        setattr(closure, k, v)
    return closure


@pytest.mark.parametrize("attr", ["inputs", "outputs", "state_keys", "edge_kinds"])
def test_a_bare_string_declaration_is_refused_at_construction(attr):
    extra = {"integrates": False} if attr == "state_keys" else {}
    with pytest.raises(TypeError, match=f"declares {attr} as the string 'rho_amb'"):
        _build(closures=(_closure_with(**{attr: "rho_amb"}, **extra),))


def test_a_non_bool_integrates_is_refused_at_construction():
    with pytest.raises(TypeError, match="integrates='yes'; it must be True or False"):
        _build(closures=(_closure_with(integrates="yes"),))


def test_a_declared_output_that_is_a_layer_state_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="declares output 'air.phi', which is a state key"):
        _build(closures=(_closure_with(outputs=("air.phi",)),))
    # the unowned transport "<layer>.q" carve-out stays allowed (no Kirchhoff layer owns it)


def test_species_columns_that_disagree_with_the_layer_are_refused_at_construction():
    net = _net()
    air = KirchhoffLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        drives=[ConstantDrive("airpath", "wind")], boundary=["ambient"],
    )
    gas = TransportLayer(
        net, "gas", capacity=torch.tensor([50.0, 80.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], n_species=2, species_names=("bod", "sulfide"), scheme="implicit",
    )
    good = _closure_with(species_columns={"gas": {"bod": 0, "sulfide": 1}})
    Model(net, {"air": air, "gas": gas}, closures=[good])
    bad = _closure_with(species_columns={"gas": {"bod": 1, "sulfide": 0}})
    with pytest.raises(ValueError, match=r"species_columns\['gas'\]\['bod'\] = 1"):
        Model(net, {"air": air, "gas": gas}, closures=[bad])


def test_a_closure_added_after_construction_is_dispatched_by_its_own_declarations():
    _, model, state, drivers, _, _ = _build()

    class Late:
        integrates = True

        def __call__(self, state, drivers, ctx):
            return {"late_dt": torch.tensor(ctx.dt, dtype=F64)}

    model.closures.append(Late())
    assert model._apply_closures(state, drivers, ctx=StepContext(dt=3.0))["late_dt"] == 3.0
    assert id(model.closures[0]) in model._integrating


def test_ownership_lists_writers_and_readers_of_declared_keys():
    class Writer:
        outputs = ("wind",)
        inputs = ("species.x",)

        def __call__(self, state, drivers):
            return {"wind": drivers["wind"]}

    _, model, *_ = _build(closures=(Writer(),))
    own = model.refs.ownership()
    assert own["species.x"].writers == ("layer species",)
    assert any(r.startswith("closure ") for r in own["species.x"].readers)
    assert any(w.startswith("closure ") for w in own["wind"].writers)
    assert own["species.sources"].readers == ("layer species",)
    assert own["species.sources"].writers == ("caller",)
    assert own["species.sources"].field.role == "driver"


def test_field_batch_shape_is_the_shape_without_the_fields_layout():
    _, model, *_ = _build()
    f = model.refs.species.x
    assert f.batch_shape(torch.zeros(4, 2, dtype=F64)) == (4,)
    assert f.batch_shape(torch.zeros(2, dtype=F64)) == ()


def test_check_reports_overwritten_undeclared_and_probed_outputs():
    class A:
        outputs = ("wind",)

        def __call__(self, state, drivers):
            return {"wind": drivers["wind"], "extra": drivers["wind"]}

    class B:
        outputs = ("wind",)

        def __call__(self, state, drivers):
            return {"wind": drivers["wind"]}

    def undeclared(state, drivers):
        return {}

    _, model, state, drivers, *_ = _build(closures=(A(), B(), undeclared))
    codes = {i.code for i in model.check(state, drivers, dt=10.0, probe=True).issues}
    assert {"overwritten-output", "undeclared-closure", "declared-vs-probed"} <= codes
