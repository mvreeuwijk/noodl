"""Sparse-path tests for PotentialFlowLayer (Task 11): grounding via the per-instance
certificate from solvers.grounding, and linear_init/solve routed through
GraphLaplacianOperator + solvers.select.solve rather than a dense einsum Jacobian.
"""

import pytest
import torch

import tellegen.solvers.select as select_module
from tellegen.drives import ConstantDrive
from tellegen.elements import Conductance, FixedFlow, PowerLaw
from tellegen.elements.fan import FanCurve
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network


def _three_node_chain() -> Network:
    """ambient (boundary) -- z1 -- z2, both edges kind "conduction"."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="conduction")
    net.add_edge("z1", "z2", kind="conduction")
    return net


def test_batched_counterexample_raises_naming_instance_1_not_empty_list():
    # The verified defect (spec section 3.1): per-instance slopes [[1, 1], [0, 1]] on a
    # grounded 3-node chain. Instance 0 (slopes [1, 1]) is fully grounded (min eigenvalue
    # 0.382); instance 1 (slopes [0, 1]) has its ambient-z1 edge closed (slope 0), so {z1,
    # z2} floats as a group (min eigenvalue 0.0) even though z2 alone still "looks" fine
    # from its own nonzero z1-z2 edge. The OLD _floating_group_nodes ORs "is this edge
    # nonzero" across the WHOLE batch before checking connectivity, so it sees the edge as
    # present for BOTH instances (nonzero somewhere in the batch) and returns [] -- passing
    # a genuinely singular instance. spd_certificate is per-instance and must catch this.
    #
    # Naming instance 1 is not on its own enough to distinguish the fix from the defect:
    # before the fix the call DID raise here, but only downstream, from
    # solvers.linear.solve's own singular-system retry ("singular system at batch indices
    # [1]") -- after a dense (n_I, n_I) Jacobian had been assembled and factorised, with no
    # statement of WHY instance 1 is singular and no check at all on any path that does not
    # end in a dense factorisation. The assertions below therefore pin the message to the
    # grounding check itself and to the per-instance diagnosis (amendment A2) it must carry.
    net = _three_node_chain()
    g = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "chain", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )
    phi_b = torch.zeros(2, 1, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"\[1\]") as excinfo:
        layer.linear_init(phi_b, {}, None)

    message = str(excinfo.value)
    assert "PotentialFlowLayer 'chain'" in message  # errors name the offending layer
    assert "linear_init" in message
    assert "ungrounded interior nodes" in message
    assert "z1" in message and "z2" in message
    assert "singular system" not in message


def test_unbatched_floating_group_error_still_names_nodes():
    # Companion to the above: the existing (unbatched) tests in test_potential.py must keep
    # naming actual node identifiers, not a batch index -- this is the ndim==0 certificate
    # branch, checked directly here rather than only implicitly via the unchanged old tests.
    net = Network(dtype=torch.float64)
    for name in ("ambient", "z0", "f1", "f2"):
        net.add_node(name)
    net.add_edge("ambient", "z0", kind="conduction")
    net.add_edge("f1", "f2", kind="conduction")
    net.add_edge("z0", "f1", kind="duct")

    layer = PotentialFlowLayer(
        net,
        "grp",
        [
            Conductance(torch.tensor([1.0, 1.0], dtype=torch.float64), kind="conduction"),
            FixedFlow(torch.tensor([0.1], dtype=torch.float64), kind="duct"),
        ],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"f1.*f2|f2.*f1") as excinfo:
        layer.linear_init(phi_b, {}, None)

    assert "PotentialFlowLayer 'grp'" in str(excinfo.value)


def _shutoff_fan_layer() -> tuple[PotentialFlowLayer, torch.Tensor, torch.Tensor]:
    """ambient (boundary) -- z, one FanCurve edge, plus a source that pulls 0.2 through it.

    P(q) = 100 - 200 q on 0 <= q <= 1: shutoff pressure 100 at q = 0, -100 at q = q_max, so
    the fan is on the smooth part of its curve at dp = 0 (slope -1/P'(q) = 0.005 > 0) and the
    whole network is grounded there. Past its shutoff point (back-pressure -dp > 100, i.e.
    phi_z > 100) FanCurve.dflow is EXACTLY zero -- a physical fan that is shut contributes no
    slope at all -- so at such a point the only edge tying z to the boundary is inactive and
    the operator is genuinely singular.
    """
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="fan")
    element = FanCurve(
        torch.tensor([100.0, -200.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        kind="fan",
    )
    layer = PotentialFlowLayer(net, "fan", [element], boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, -0.2], dtype=torch.float64)
    return layer, phi_b, sources


def test_grounding_is_checked_on_actual_slopes_even_when_phi0_is_supplied():
    # The pre-Task-11 code checked grounding only inside linear_init, on linear_init's own
    # tangent-at-zero slopes, and skipped it entirely whenever a caller supplied phi0 (`if
    # phi0 is None: phi0 = self.linear_init(...)` never runs, and with it neither does its
    # check). FanCurve is what makes the two distinguishable: its slope is dp-dependent AND
    # exactly zero past shutoff, so the SAME network is grounded at its linear_init point and
    # ungrounded at a supplied phi0 beyond the fan's shutoff pressure.
    layer, phi_b, sources = _shutoff_fan_layer()

    # Non-vacuity: from its own linear_init the network is grounded and the solve succeeds.
    phi, q = layer.solve(phi_b, {}, sources, differentiable=False)
    torch.testing.assert_close(q, torch.tensor([0.2], dtype=torch.float64), atol=1e-9, rtol=0)

    phi0_supplied = torch.tensor([200.0], dtype=torch.float64)  # past shutoff: dflow == 0
    with pytest.raises(RuntimeError, match=r"solve: floating nodes") as excinfo:
        layer.solve(phi_b, {}, sources, phi0=phi0_supplied, differentiable=False)

    assert "PotentialFlowLayer 'fan'" in str(excinfo.value)


def test_grounding_at_a_supplied_phi0_is_checked_on_the_differentiable_path_too():
    # Same check, same place: solve() runs it once, before the differentiable/
    # non-differentiable branch, so neither path can be the one that skips it.
    layer, phi_b, sources = _shutoff_fan_layer()
    phi0_supplied = torch.tensor([200.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"solve: floating nodes"):
        layer.solve(phi_b, {}, sources, phi0=phi0_supplied, differentiable=True)


def _series_layer(linear_solver: str = "auto") -> tuple[PotentialFlowLayer, dict, torch.Tensor]:
    """Three-zone series network under a wind drive, matching the shape of the CONTAM
    verification cases: ambient_w -> z1 -> z2 -> ambient_l, one PowerLaw per airpath edge.

    Returns `(layer, drivers, phi_boundary)`.
    """
    net = Network(dtype=torch.float64)
    for name in ("ambient_w", "z1", "z2", "ambient_l"):
        net.add_node(name)
    net.add_edge("ambient_w", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient_l", kind="airpath")
    element = PowerLaw(
        torch.tensor([0.010, 0.008, 0.012], dtype=torch.float64), 0.65, dp_transition=1e-6
    )
    layer = PotentialFlowLayer(
        net,
        "series",
        [element],
        [ConstantDrive(kind="airpath", key="wind")],
        boundary=["ambient_w", "ambient_l"],
        linear_solver=linear_solver,
    )
    drivers = {"wind": torch.tensor([12.0, 0.0, 0.0], dtype=torch.float64)}
    phi_b = torch.zeros(2, dtype=torch.float64)
    return layer, drivers, phi_b


def test_sparse_and_dense_newton_paths_agree_on_a_contam_style_series_network():
    # Three-zone series network under a wind drive, matching the shape of the CONTAM
    # verification cases: this exercises solve()'s full non-differentiable path (which now
    # builds a GraphLaplacianOperator per Newton iteration) against layer.jacobian() (still
    # dense, unchanged, used here only as the independent reference via a hand-rolled
    # Newton loop) to confirm the two agree to solver-contract tolerance.
    layer, drivers, phi_b = _series_layer()

    phi_sparse, q_sparse = layer.solve(
        phi_b, drivers, None, differentiable=False, atol=1e-13, rtol=1e-13
    )

    # Independent dense reference: layer.jacobian() (unchanged, still a dense einsum) fed to
    # torch.linalg.solve directly, in a hand-rolled damped Newton loop mirroring newton()'s
    # own iteration exactly (same omega/switch_ratio schedule) but never touching the
    # operator contract at all.
    def residual_fn(x):
        return layer.residual(x, phi_b, drivers, None)

    x = layer.linear_init(phi_b, drivers, None)
    omega_i = torch.tensor(0.75, dtype=torch.float64)
    r = residual_fn(x)
    norm0 = r.abs().amax(dim=-1)
    tol = 1e-13 + 1e-13 * norm0
    for _ in range(50):
        if bool(norm0 < tol):
            break
        J = layer.jacobian(x, phi_b, drivers)
        dx = torch.linalg.solve(J, r)
        prev_norm = norm0
        x = x - omega_i * dx
        r = residual_fn(x)
        norm0 = r.abs().amax(dim=-1)
        if bool(norm0 / prev_norm.clamp_min(1e-300) < 0.5):
            omega_i = torch.tensor(1.0, dtype=torch.float64)
    phi_dense = layer.assemble(x, phi_b)
    q_dense = layer.flows(phi_dense, drivers)

    torch.testing.assert_close(phi_sparse, phi_dense, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(q_sparse, q_dense, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("differentiable", [False, True])
def test_solve_never_assembles_the_dense_einsum_jacobian(monkeypatch, differentiable):
    # The structural half of the sparse-vs-dense pair above: the numbers were already right
    # before Newton's per-iteration operator became a GraphLaplacianOperator (that refactor
    # is a no-op for correctness), so what has to be asserted is that the dense
    # (n_interior, n_interior) einsum is no longer built at all. layer.jacobian() itself is
    # deliberately untouched -- it remains the dense oracle other tests call directly -- so
    # the assertion is that solve() does not call it.
    layer, drivers, phi_b = _series_layer()

    def _no_dense_jacobian(self, *args, **kwargs):
        raise AssertionError("solve() must not assemble the dense einsum Jacobian")

    monkeypatch.setattr(PotentialFlowLayer, "jacobian", _no_dense_jacobian)
    phi, q = layer.solve(phi_b, drivers, None, differentiable=differentiable)
    assert torch.isfinite(phi).all() and torch.isfinite(q).all()


def test_direct_and_auto_linear_solvers_agree_on_a_contam_style_series_network():
    # linear_solver="direct" is the RETAINED milestone-1 numerics: the operator's explicit
    # A_I diag(g) A_I^T, LU-factorised. It must agree with the migrated (sparse, Krylov)
    # default to solver-contract tolerance, or the retained reference is not a reference.
    auto_layer, drivers, phi_b = _series_layer("auto")
    direct_layer, _, _ = _series_layer("direct")

    phi_auto, q_auto = auto_layer.solve(phi_b, drivers, None, differentiable=False)
    phi_direct, q_direct = direct_layer.solve(phi_b, drivers, None, differentiable=False)

    torch.testing.assert_close(phi_auto, phi_direct, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(q_auto, q_direct, rtol=1e-9, atol=1e-12)


def test_auto_goes_through_pcg_and_direct_goes_through_an_lu_factorisation(monkeypatch):
    # The numeric agreement above is only meaningful if the two are genuinely different code
    # paths; these spies are what establish that. `select.pcg` is the sparse path's own
    # inner solver, `torch.linalg.lu_factor_ex` the direct path's.
    pcg_calls, lu_calls = [], []
    real_pcg, real_lu = select_module.pcg, torch.linalg.lu_factor_ex

    def spy_pcg(*args, **kwargs):
        pcg_calls.append(1)
        return real_pcg(*args, **kwargs)

    def spy_lu(*args, **kwargs):
        lu_calls.append(1)
        return real_lu(*args, **kwargs)

    monkeypatch.setattr(select_module, "pcg", spy_pcg)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", spy_lu)

    auto_layer, drivers, phi_b = _series_layer("auto")
    auto_layer.solve(phi_b, drivers, None, differentiable=False)
    assert pcg_calls, "linear_solver 'auto' must go through pcg"
    pcg_after_auto = len(pcg_calls)

    direct_layer, _, _ = _series_layer("direct")
    direct_layer.solve(phi_b, drivers, None, differentiable=False)
    assert len(pcg_calls) == pcg_after_auto, "linear_solver 'direct' must not call pcg"
    assert lu_calls, "linear_solver 'direct' must LU-factorise the assembled operator"


def test_solve_fills_a_supplied_diagnostics_dict():
    layer, drivers, phi_b = _series_layer()
    diagnostics: dict = {}

    layer.solve(phi_b, drivers, None, differentiable=False, diagnostics=diagnostics)

    assert set(diagnostics) == {"newton_iterations", "linear_iterations", "method"}
    assert diagnostics["method"] == "auto"
    assert diagnostics["newton_iterations"] >= 1
    assert isinstance(diagnostics["linear_iterations"], torch.Tensor)
    assert bool(torch.all(diagnostics["linear_iterations"] >= 1))


def test_diagnostics_are_filled_on_the_differentiable_path_too():
    # Task 14 reads these alongside its timings, and its backward budget runs the
    # differentiable path -- so diagnostics must not be a non-differentiable-only feature.
    layer, drivers, phi_b = _series_layer()
    diagnostics: dict = {}

    layer.solve(phi_b, drivers, None, differentiable=True, diagnostics=diagnostics)

    assert set(diagnostics) == {"newton_iterations", "linear_iterations", "method"}
    assert diagnostics["newton_iterations"] >= 1
    assert isinstance(diagnostics["linear_iterations"], torch.Tensor)


def test_unknown_linear_solver_raises_value_error_naming_it():
    net = _three_node_chain()
    with pytest.raises(ValueError, match="banana"):
        PotentialFlowLayer(
            net,
            "chain",
            [Conductance(torch.ones(2, dtype=torch.float64), kind="conduction")],
            boundary=["ambient"],
            linear_solver="banana",
        )


def test_sparse_adjoint_matches_dense_jacobian_transpose_solve(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    el = PowerLaw(
        elements[0].C.detach().clone().requires_grad_(True), elements[0].n, learnable=True
    )
    layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi, q = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)
    phi_i = phi[layer.interior]
    grad_phi_i = torch.tensor([1.0, -2.0], dtype=torch.float64)

    lam_sparse = layer.adjoint(phi_i, phi_b, {"wind": wind}, grad_phi_i)

    J = layer.jacobian(phi_i, phi_b, {"wind": wind})
    lam_dense = torch.linalg.solve(J.T, grad_phi_i)

    torch.testing.assert_close(lam_sparse, lam_dense, rtol=1e-9, atol=1e-12)


def test_adjoint_routes_through_the_layers_own_linear_solver(monkeypatch, two_zone_layer):
    # Amendment A3.3 requires `linear_solver` to reach `adjoint` too, not only `linear_init`
    # and `solve`: a layer configured with the retained milestone-1 dense numerics must keep
    # them on the BACKWARD pass as well, or "direct" is only half a reference. The numbers
    # must agree with the migrated default (first assertion) AND the two must genuinely be
    # different code paths (the spies) -- either alone proves nothing.
    net, elements, drives, boundary = two_zone_layer
    phi_b = torch.zeros(1, dtype=torch.float64)
    drivers = {"wind": torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)}
    grad_phi_i = torch.tensor([1.0, -2.0], dtype=torch.float64)

    pcg_calls, lu_calls = [], []
    real_pcg, real_lu = select_module.pcg, torch.linalg.lu_factor_ex

    def spy_pcg(*args, **kwargs):
        pcg_calls.append(1)
        return real_pcg(*args, **kwargs)

    def spy_lu(*args, **kwargs):
        lu_calls.append(1)
        return real_lu(*args, **kwargs)

    monkeypatch.setattr(select_module, "pcg", spy_pcg)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", spy_lu)

    lam = {}
    for linear_solver in ("auto", "direct"):
        layer = PotentialFlowLayer(
            net,
            "zones",
            list(elements),
            drives=drives,
            boundary=boundary,
            linear_solver=linear_solver,
        )
        phi, _ = layer.solve(
            phi_b, drivers, None, differentiable=False, atol=1e-13, rtol=1e-13
        )
        phi_i = phi[layer.interior]
        # Cleared AFTER the forward solve so the counters below describe the adjoint call
        # alone; the forward path's own solver choice is already covered elsewhere.
        pcg_calls.clear()
        lu_calls.clear()
        lam[linear_solver] = layer.adjoint(phi_i, phi_b, drivers, grad_phi_i)
        if linear_solver == "auto":
            assert pcg_calls, "adjoint under linear_solver 'auto' must go through pcg"
            assert not lu_calls, "adjoint under linear_solver 'auto' must not LU-factorise"
        else:
            assert lu_calls, "adjoint under linear_solver 'direct' must LU-factorise"
            assert not pcg_calls, "adjoint under linear_solver 'direct' must not call pcg"

    torch.testing.assert_close(lam["auto"], lam["direct"], rtol=1e-9, atol=1e-12)
