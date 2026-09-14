"""Model: several layers on one graph, stepped together (milestone 2 spec section 7)."""

from __future__ import annotations

import pytest
import torch

from tellegen.drives import ConstantDrive
from tellegen.elements import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.layers.reaction import FirstOrderDecay
from tellegen.layers.transport import TransportLayer
from tellegen.model import Model, Ports
from tellegen.topology import Network

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


def test_a_transport_layer_whose_kinds_no_potential_layer_provides_is_refused():
    net = _net()
    net.add_edge("z1", "z2", kind="duct")
    air = PotentialFlowLayer(
        net, "air", [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)],
        boundary=["ambient"],
    )
    heat = TransportLayer(net, "heat", capacity=torch.ones(2, dtype=F64),
                          flow_kind=("airpath", "duct"), boundary=["ambient"])
    with pytest.raises(ValueError, match=r"'heat'.*duct.*0 potential layers"):
        Model(net, {"air": air, "heat": heat})


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
    # whose stacked branch no single-species test reaches. The steady state is reached by
    # STEPPING, not by `Model.steady`: `TransportLayer.steady` is unreliable for K > 1
    # independently of `Model` (measured: its linear solve fails with MAX_ITER/SINGULAR for
    # 14 of 60 flow magnitudes in 0.005..0.035 kg/s at K=2 and 0 of 60 at K=1, for a system
    # whose condition number is ~1.0) -- a pre-existing defect of the transport layer,
    # reported with the numbers in this task's report rather than worked around here.
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
