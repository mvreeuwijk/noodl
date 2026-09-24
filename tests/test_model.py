"""Model: several layers on one graph, stepped together (milestone 2 spec section 7)."""

from __future__ import annotations

import warnings

import pytest
import torch

from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.model import Model, Ports
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
    air = PotentialFlowLayer(
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
    air = PotentialFlowLayer(
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
    q = model.potential["air"].flows_of_kind(new["air.q"], "airpath")
    rate = species.rate(new["species.x"], q, drivers["species.sources"],
                        drivers["species.x_boundary"])
    # backward Euler: (x_new - x_old) / dt == rate(x_new)
    torch.testing.assert_close(rate, (new["species.x"] - state["species.x"]) / 600.0,
                               rtol=1e-9, atol=1e-15)


def test_steady_is_the_limit_of_stepping_and_residuals_vanish_there():
    # Ruling R21: the residual this test asserts on is the air layer's own Newton residual,
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


def test_a_transport_layer_whose_kinds_no_potential_layer_provides_reads_a_driver():
    """The rule this replaces refused the model outright. A transport layer advecting on a
    kind the potential layer does not provide is now driver-prescribed (spec section 5)."""
    net = _net()
    net.add_edge("z1", "z2", kind="duct")
    air = PotentialFlowLayer(
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
    air = PotentialFlowLayer(
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


def test_reactions_are_a_splitting_of_step_only_and_stay_out_of_steady_and_residuals():
    # Pins the plan-mandated placement: a reaction is applied AFTER a transport step, so it
    # is outside both `steady` and the balance `residuals` reports. A later task must not
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


def test_ports_round_trip_and_air_boundary_flow_balances_the_interior():
    _, model, state, drivers, _, _ = _build()
    # Ruling R21 again: the boundary flow this asserts on IS the air layer's interior
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
    # `Model.steady`. The comment that used to stand here said the second route had to be
    # avoided because `TransportLayer.steady` was unreliable for K > 1 -- true when it was
    # written, and fixed in Task 8b: a K-species operator is block diagonal with K
    # identical blocks, so its Krylov space is invariant after n_i steps, and `gmres` was
    # mishandling that near-breakdown. See `tests/solvers/test_gmres_breakdown.py`.
    model, state, drivers, gas = _build_multi()
    zero = torch.zeros(3, 2, dtype=F64)
    new = model.step(state, drivers, 600.0)
    assert new["gas.x"].shape == (2, 2)
    q = model.potential["air"].flows_of_kind(new["air.q"], "airpath")
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
    # potential layer's adjoint); the dictated tests cover `step` only. Tight Newton
    # tolerances (ruling R21) so the finite difference is not comparing solver noise.
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
    """Renamed in part 3 (P1-2): the mechanism is the implicit interface adjoint, not
    unrolling; the check is the same central-difference comparison, from the zero state."""
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
    # Ruling R21: at the default Newton tolerance the weaker-wind instance stagnates at the
    # solver's own noise floor (~2e-11, measured) and never reaches the 1e-12 asked of the
    # coupling; the SOLVE is asked for the accuracy the assertion needs instead.
    model.steady(state, dict(drivers, wind=winds), diagnostics=diag, atol=1e-14, rtol=1e-14)
    assert diag["converged"].shape == (2,) and bool(diag["converged"].all())
    assert diag["max_change"]["species"].shape == (2,)


def test_iterate_non_convergence_names_only_the_failing_batch_instances():
    """The dictated non-convergence test is unbatched, so it reports "all"; this one pins the
    per-instance naming the milestone asks for.

    Tight Newton tolerances (ruling R21) so that what instance 1 runs out of is the PASS
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


def test_iterate_steps_every_transport_layer_but_tests_only_the_named_ones():
    """Spec 7's motivating case: a species layer in kg/kg beside a thermal layer in kelvin.

    No single absolute tolerance can straddle the two, so `iterate_tol` names the species
    layer alone. The thermal layer is still stepped on EVERY pass -- it just gets no vote on
    convergence, and no entry in `max_change`. Ruling R21 for the tight Newton tolerances:
    the assertions below are three orders inside the coupling tolerance, so the solve is
    asked for the accuracy they need.
    """
    tight = {"atol": 1e-14, "rtol": 1e-14}
    net = _net()
    air = PotentialFlowLayer(
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

    No potential layer can own 'link' (no element is built for it), so the flows must
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


def test_model_reads_transport_flows_from_a_driver_when_no_potential_layer_owns_them():
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


def test_model_raises_when_a_layer_has_both_a_potential_owner_and_a_flow_driver():
    """Uses `_build()`, this module's own fixture for a Model with one potential layer
    ("air") and one transport layer ("species") the potential layer owns."""
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


def test_a_closure_may_not_write_the_flow_driver_of_a_layer_a_potential_layer_owns():
    """The `_apply_closures` carve-out has three conjuncts, and the third -- that NO
    potential layer owns the transport layer's kinds -- is what keeps a closure from
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
    # R6: advances by a FIXED 1.0 every call, regardless of the step's own interval, so it
    # does not integrate (does not need a StepContext).
    integrates = False

    def __call__(self, state, drivers):
        return {"demo.count": state["demo.count"] + 1.0}


class _SilentClosure:
    state_keys = ("demo.count",)
    integrates = False  # R6: never writes its state key at all; no interval involved either.

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
        integrates = False  # R6: declared so the state_keys/layer-collision check is reached.

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


def test_n14_a_same_named_driver_does_not_satisfy_a_silent_closures_state_key():
    """N14: `key not in drv` used to be satisfied by a caller-supplied DRIVER of the same
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


def test_fr2_a_closure_may_declare_the_flow_driver_of_an_ownerless_layer_as_state():
    """FR-2: the construction-time state_keys/layer-state collision check must mirror
    `_apply_closures`' own ownerless-"<layer>.q" carve-out -- "<layer>.q" for a transport
    layer no potential layer owns is a DRIVER (what a flow closure writes), not that
    layer's state, so a closure may legitimately declare it as state it carries across
    steps (a flow closure that also remembers its own last-written flow)."""

    class _RememberedFlow:
        state_keys = ("c.q",)
        integrates = False  # R6: passes its own last-written flow through unchanged.

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
    # Construction must NOT raise "which is layer 'c''s own state key" (pre-FR-2 behaviour).
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
    level): N1's counter-closure.

    R6: `integrates = False` here even though the docstring says "integrates" -- the
    increment is a constant fixed at CONSTRUCTION (the test passes it the step's own `dt`
    so the numbers line up), never read off a `StepContext`, so this closure does not
    consult the model's own clock and takes no `ctx`. N1 (pass-vs-step, exercised here) and
    R6 (whose clock a closure reads) are independent concerns.
    """

    state_keys = ("demo.n",)
    integrates = False

    def __init__(self, increment: float) -> None:
        self.increment = increment

    def __call__(self, state, drivers):
        return {"demo.n": state["demo.n"] + self.increment}


def test_n1_closure_carried_state_advances_once_per_step_not_once_per_pass():
    """N1: under coupling='iterate', a closure-carried state key must be evaluated from the
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
    state it is handed: I8-1's edge case is exactly a step-start state that never seeds
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
    """Shared builder for the I8-1 tests (a closure-carried key missing vs. seeded from the
    step-start state) and A3's warning tests, all on the same `_Feedback` +
    `_ParametrizedIntegratingCounter` iterate model. `state` is SEEDED ("demo.n" zero): A3's
    own control test wants that, and its warning test derives the unseeded state by dropping
    the key from it, exactly as the I8-1 control test already built its own `start`. `param`
    defaults to a fixed, non-differentiable 0.5 for A3's tests, which never differentiate;
    the I8-1 tests each pass their own differentiable leaf so `torch.autograd.grad` can read
    it back afterward."""
    if param is None:
        param = torch.tensor(0.5, dtype=F64)
    _, model, state, drivers, _, _ = _build(
        closures=[_Feedback(2e3), _ParametrizedIntegratingCounter(param)],
        coupling="iterate", iterate_tol={"species": 1e-12}, iterate_max=60,
    )
    state = dict(state, **{"demo.n": torch.zeros((), dtype=F64)})
    return model, state, drivers


def test_i8_1_a_closure_state_key_missing_from_the_step_start_state_gets_a_truncated_gradient():
    """I8-1 (Task 8 review): `_iterate`'s interface excludes every `closure_state_keys` entry
    UNCONDITIONALLY (`keys = [... if k not in self.closure_state_keys ...]`), whether or not
    N1's own `if key in base:` pinning applies to it. When the key IS seeded (see the control
    below) that costs nothing: N1 pins the closure's read to the step-start state on every
    pass, and the one differentiable re-run of the certified pass reads that same
    graph-carrying tensor directly (`step_from=state`), so the gradient survives untouched.

    When the key is ABSENT from the step-start state -- the one case N1 does not pin,
    documented in `_iterate`'s own docstring ("a closure-carried key MISSING from the
    step-start state ... drops a path that only exists because that key was not seeded in the
    first place") -- the closure instead reads the PREVIOUS pass's own value on every pass, so
    "demo.n" genuinely COMPOUNDS the parameter once per PASS, and this fixture's true,
    central-difference sensitivity equals its own pass count. The one differentiable pass the
    adjoint runs sees that compounded history only as a plain, already-detached number and
    re-applies the closure ONCE more, so it can return only THAT one application's own local
    derivative (exactly 1.0 here) -- not the true, pass-count-sized sensitivity, and,
    measured here, not literally zero either: a looser paraphrase of this finding elsewhere
    (the review ledger) says the adjoint "returns ZERO gradient through it", but the single
    surviving local term is what this test pins, verified by directly running this fixture
    before writing the assertion. Either way this is the documented SAFE side of the trade
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


def test_i8_1_control_the_same_gradient_matches_central_differences_once_seeded():
    """Control for the test above: the identical closure and parameter, seeded this time, so
    N1 pins "demo.n" to the step-start state on every pass and the gradient the implicit
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
    """A3: on such a step the key is not pinned to the step start (N1's `if key in base`), so
    its closure integrates once per PASS and the adjoint omits the compounding path. The
    misconfiguration is named where it happens; it is not refused, because the first step is
    allowed to lack the key by design.

    `stacklevel` is pinned here too, not just the message text: the warning must be attributed
    to the CALLER of `step`/`steady` (this test module), not to a line inside `model.py` itself
    -- `warnings.warn`'s default filter dedupes on (message, category, module, lineno), so a
    warning permanently attributed to one internal line would let a second, later, genuinely
    different caller's identical warning go silently missing (A3 fix round). Both `step` and
    `steady` reach `_iterate` through the same `_advance`, so one `stacklevel` serves both --
    checked on both calls below."""
    model, state, drivers = _iterate_model_with_counter()   # the I8-1 tests' builder
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
    """A transport layer with no owning potential layer: its flow is written by a closure,
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


def test_current_flows_returns_potential_owned_flow():
    """A transport layer whose flow kind is owned by a potential layer: `current_flows`
    must read it from the SOLVED state, not require a driver."""
    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    from noodl.elements.conductance import Conductance

    air = PotentialFlowLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
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


def test_current_flows_solves_a_potential_owner_when_the_state_has_no_q_yet():
    """The first pass of a step: the start state carries no "<owner>.q" (project_to_model's
    initial state has only "species.x"), so current_flows must SOLVE the owner for these
    drivers, and agree with what a step would have solved."""
    from noodl.elements.conductance import Conductance

    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    air = PotentialFlowLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
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


def test_current_flows_refuses_a_flow_driver_beside_a_potential_owner_on_the_solve_path():
    """The two-sources-of-flows refusal must not depend on whether the state happens to carry
    a solved q yet: `_kind_flows` makes it on the read path, and the solve path makes it too."""
    from noodl.elements.conductance import Conductance

    net = Network(dtype=torch.float64)
    net.add_node("b")
    net.add_node("i")
    net.add_edge("i", "b", kind="link")
    air = PotentialFlowLayer(net, "air", [Conductance(kind="link", g=1.0)], boundary=["b"])
    species = TransportLayer(
        net, "x", capacity=torch.tensor([1.0], dtype=torch.float64), flow_kind="link",
        boundary=["b"],
    )
    model = Model(net, {"air": air, "x": species})
    state = {"x.x": torch.tensor([0.0], dtype=torch.float64)}  # no "air.q" -> solve path
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=torch.float64),
        "x.x_boundary": torch.zeros(1, dtype=torch.float64),
        "x.q": torch.tensor([0.9], dtype=torch.float64),  # contradicts the potential layer
    }
    with pytest.raises(ValueError, match="one source of flows, not two"):
        model.current_flows("x", state, drivers)


def test_iterate_gradient_from_a_near_fixed_point_start_matches_central_differences():
    """Start the step at the model's own steady state: the primal needs only the structural
    two passes, and the derivative it returns must still be the fixed point's (P1-2).

    Two things this test needs that the fixture's own defaults do not give, both measured
    while it was written -- and neither of them a loosened assertion:

    * A FEEDBACK WORTH DIFFERENTIATING. With the fixture's 2e-6 kg/s source nearly all of
      z1's mass fraction is the boundary's own 1e-3, which no amount of wind moves, so the
      pass map contracts at dx_new/dx_fed = -0.004 and even a two-pass unrolling is right to
      1.2e-5 (~ that ratio squared): the bug would be invisible here whatever the tolerance.
      The source is raised to 5e-4 kg/s, where the loop gain is -0.23, the two-pass
      derivative is 3.2 % wrong and the implicit one is right to 3e-7.
    * RULING R21 for the tight Newton tolerances. The CENTRAL DIFFERENCE is the reference,
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
    pass the implicit adjoint is attached to -- and never one per pass (P1-2)."""
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
    for kwargs, expect in (
        ({}, None),                             # under no_grad below: nothing to attach
        ({"differentiable": False}, None),      # detached solves: nothing reaches the state
    ):
        calls.clear()
        diag: dict = {}
        with torch.no_grad() if not kwargs else torch.enable_grad():
            model.steady(state, drivers, diagnostics=diag, **kwargs, **tight)
        assert diag["adjoint"] is expect
        assert len(calls) == diag["passes"]
    calls.clear()
    diag = {}
    out = model.steady(state, drivers, diagnostics=diag, **tight)
    assert diag["adjoint"] == "implicit"
    assert len(calls) == diag["passes"] + 1
    assert out["species.x"].requires_grad
    out["species.x"].sum().backward()
    assert el.C.grad is not None


def test_iterate_gradient_is_per_instance_on_a_batched_model():
    """I8-2 (part 3 follow-up): two wind speeds through coupling="iterate" with a learnable
    element. Each instance's sensitivity must match its OWN central difference, so the adjoint
    solve keeps instances apart; the instances must also differ, or a batch-mixing error could
    hide. Sources at 5e-4 as in the near-fixed-point test, so the feedback gain is strong.
    Solver tolerances tightened for the central-difference reference (ruling R21)."""
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

    Adapted from the brief's literal fixture: `model.steady(..., on_failure="return")` with a
    `learnable=True` element cannot reach `_iterate`'s own non-convergence branch with
    `needs_adjoint` True, because `solve_kwargs` (including `on_failure`) is forwarded to
    every potential-layer solve too (module docstring), and `PotentialFlowLayer.solve`'s
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
