"""UpstreamDensityPowerLaw: CONTAM's upstream-density orifice coefficient.

Every tensor here is built float64 explicitly: ``torch.get_default_dtype()`` is
float32 in this repository, and the reference values these tests compare against are
computed in Python floats, so a float32 element would agree with them only to ~1e-7.
"""

from __future__ import annotations

import pytest
import torch

from noodl.elements import PowerLaw, UpstreamDensityPowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
RHO_REF = 1.2041
# Node densities: 0 = "ambient" (cold, dense), 1 = "zone" (warm, light).
RHO = torch.tensor([1.2922, 1.2041], dtype=F64)


def _element(m, *, n=0.5, C=0.141421, learnable=False):
    """One edge, node 0 -> node 1, with the given density exponent."""
    return UpstreamDensityPowerLaw(
        torch.tensor([C], dtype=F64),
        torch.tensor([n], dtype=F64),
        src=torch.tensor([0]),
        tgt=torch.tensor([1]),
        m=m,
        rho_ref=RHO_REF,
        learnable=learnable,
    )


def _drivers():
    return {"rho": RHO}


# ------------------------------------------------------------------ the correction itself
@pytest.mark.parametrize("m", [0.0, 0.5, 1.0])
def test_each_family_exponent_scales_C_by_the_upstream_density_ratio(m):
    """m = 0 (mass), 1/2 (sqrt-rho orifice/leak/crack/doorway), 1 (volumetric) -- the three
    CONTAM power-law families, against an uncorrected PowerLaw with the same (C, n)."""
    el = _element(m)
    base = PowerLaw(el.C, el.n)
    dp = torch.tensor([9.0], dtype=F64)
    # dp > 0: flow runs src -> tgt, so the air entering the path is node 0's.
    forward = (RHO[0] / RHO_REF) ** m
    torch.testing.assert_close(el.flow(dp, _drivers()), forward * base.flow(dp))
    # dp < 0: flow runs tgt -> src, so node 1's density is the upstream one.
    backward = (RHO[1] / RHO_REF) ** m
    torch.testing.assert_close(el.flow(-dp, _drivers()), backward * base.flow(-dp))


def test_the_two_directions_use_different_coefficients():
    """The whole point of the correction: |q| is NOT odd in dp once the two endpoints sit at
    different densities, because the coefficient itself switches with the flow direction."""
    el = _element(0.5)
    dp = torch.tensor([9.0], dtype=F64)
    q_pos, q_neg = el.flow(dp, _drivers()), el.flow(-dp, _drivers())
    assert q_pos.item() > -q_neg.item()                      # denser air in: more mass flow
    ratio = (RHO[0] / RHO[1]).sqrt().item()
    assert q_pos.item() / -q_neg.item() == pytest.approx(ratio, rel=1e-12)


def test_an_isothermal_network_reproduces_the_plain_power_law_exactly():
    el = _element(0.5)
    uniform = {"rho": torch.full((2,), RHO_REF, dtype=F64)}
    dp = torch.tensor([-9.0, -1e-4, 1e-4, 9.0], dtype=F64).reshape(4, 1)
    base = PowerLaw(el.C, el.n)
    torch.testing.assert_close(el.flow(dp, uniform), base.flow(dp))
    torch.testing.assert_close(el.dflow(dp, uniform), base.dflow(dp))


def test_m_zero_is_the_plain_power_law_at_any_density():
    el = _element(0.0)
    dp = torch.tensor([-9.0, 0.0, 9.0], dtype=F64).reshape(3, 1)
    torch.testing.assert_close(el.flow(dp, _drivers()), PowerLaw(el.C, el.n).flow(dp))


def test_a_per_edge_exponent_mixes_families_within_one_element():
    el = UpstreamDensityPowerLaw(
        torch.tensor([1.0, 1.0], dtype=F64),
        torch.tensor([0.5, 0.5], dtype=F64),
        src=torch.tensor([0, 0]),
        tgt=torch.tensor([1, 1]),
        m=torch.tensor([0.0, 1.0], dtype=F64),
        rho_ref=RHO_REF,
    )
    dp = torch.tensor([4.0, 4.0], dtype=F64)
    q = el.flow(dp, _drivers())
    assert q[0].item() == pytest.approx(2.0, rel=1e-12)
    assert q[1].item() == pytest.approx(2.0 * RHO[0].item() / RHO_REF, rel=1e-12)


# ------------------------------------------------------------------ derivative and Jacobian
def test_dflow_is_the_exact_elementwise_derivative_on_both_sides():
    """`C` is piecewise constant in sign(dp), so the analytic dflow stays exact -- the
    Jacobian the Newton solve assembles is still A_I diag(dflow) A_I^T, unchanged in form."""
    el = _element(0.5)
    dp = torch.tensor([-20.0, -0.5, 0.5, 20.0], dtype=F64)
    h = 1e-7
    fd = (el.flow(dp + h, _drivers()) - el.flow(dp - h, _drivers())) / (2 * h)
    torch.testing.assert_close(el.dflow(dp, _drivers()), fd, rtol=1e-6, atol=1e-9)


def test_dflow_matches_the_autograd_default_including_at_zero_and_the_transition():
    from noodl.elements.base import Element

    el = _element(0.5)
    dp = torch.tensor([-2.0, -1e-3, -5e-4, 0.0, 5e-4, 1e-3, 2.0], dtype=F64)
    torch.testing.assert_close(
        el.dflow(dp, _drivers()), Element.dflow(el, dp, _drivers()), atol=1e-9, rtol=1e-9
    )


def test_dflow_is_strictly_positive_so_the_operator_stays_spd():
    el = _element(0.5)
    dp = torch.linspace(-5.0, 5.0, 101, dtype=F64)
    assert bool((el.dflow(dp, _drivers()) > 0).all())


def test_linear_init_is_the_mean_of_the_two_direction_slopes():
    el = _element(0.5)
    c, k = el.linear_init(_drivers())
    _, k_base = PowerLaw(el.C, el.n).linear_init()
    mean = 0.5 * ((RHO[0] / RHO_REF) ** 0.5 + (RHO[1] / RHO_REF) ** 0.5)
    torch.testing.assert_close(k, mean * k_base)
    assert bool((c == 0).all())


# ------------------------------------------------------------------ differentiability
def test_gradcheck_flow_wrt_dp_on_both_sides_and_across_the_laminar_transition():
    """dp values sit on both sides of 0 and just outside +-dp_transition (1e-3).

    Neither dp = 0 nor dp = +-dp_transition is included: BOTH are genuine derivative kinks
    of this law, where no finite difference can agree with either one-sided analytic slope.
    +-dp_transition is `PowerLaw`'s own kink (the blend promises continuity in VALUE only --
    see test_blend_is_continuous_in_value_at_the_transition); dp = 0 is this element's, where
    the coefficient switches endpoints. The two tests below pin dp = 0 exactly: the kink
    vanishes when the endpoints share a density (then gradcheck must and does pass), and
    where it does not vanish the backward pass must still be finite.
    """
    el = _element(0.5)
    drv = _drivers()
    dp = torch.tensor([-2.0, -1.1e-3, -5e-4, 5e-4, 1.1e-3, 2.0], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x, drv), (dp,), eps=1e-6, atol=1e-6)


def test_gradcheck_at_dp_zero_exactly_when_the_endpoints_share_a_density():
    """With a uniform density the coefficient no longer switches, so dp = 0 is an ordinary
    interior point of the laminar branch and gradcheck must pass there exactly."""
    el = _element(0.5)
    uniform = {"rho": torch.full((2,), RHO_REF, dtype=F64)}
    zero = torch.zeros(1, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: el.flow(x, uniform), (zero,), eps=1e-6, atol=1e-6
    )


def test_at_dp_zero_the_slope_jumps_but_the_backward_pass_stays_finite():
    """The correction makes dp = 0 a slope kink whenever the two endpoints differ in
    density, exactly as `Damper` does: the VALUE is still continuous (both branches vanish
    at the origin, so Newton sees a continuous residual), but the one-sided slopes differ by
    the density ratio. autograd takes the dp >= 0 branch and must return it finite -- never
    an inf * 0 = nan leaking out of the unselected `PowerLaw` branch.
    """
    el = _element(0.5)
    k = PowerLaw(el.C, el.n).linear_init()[1]
    zero = torch.zeros(1, dtype=F64, requires_grad=True)
    q = el.flow(zero, _drivers())
    assert q.item() == 0.0
    (grad,) = torch.autograd.grad(q.sum(), zero)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, (RHO[0] / RHO_REF) ** 0.5 * k)


def test_no_nan_reaches_backward_exactly_at_the_transition_boundaries():
    dpt = 1e-3
    el = UpstreamDensityPowerLaw(
        torch.tensor([0.141421], dtype=F64),
        torch.tensor([0.5], dtype=F64),
        src=torch.tensor([0]),
        tgt=torch.tensor([1]),
        m=0.5,
        rho_ref=RHO_REF,
        dp_transition=dpt,
    )
    dp = torch.tensor([-dpt, 0.0, dpt], dtype=F64, requires_grad=True)
    q = el.flow(dp, _drivers())
    assert torch.isfinite(q).all()
    (grad,) = torch.autograd.grad(q.sum(), dp)
    assert torch.isfinite(grad).all()
    assert el.dflow(dp.detach(), _drivers()).isfinite().all()


def test_gradcheck_flow_wrt_the_density_driver():
    """The coefficient reads `drivers['rho']`, so a gradient must reach the density -- that
    is what makes the correction differentiable rather than a frozen constant."""
    el = _element(0.5)
    rho = RHO.clone().requires_grad_(True)
    dp = torch.tensor([-2.0, 2.0], dtype=F64)
    assert torch.autograd.gradcheck(
        lambda r: el.flow(dp, {"rho": r}), (rho,), eps=1e-6, atol=1e-6
    )


def test_gradcheck_flow_wrt_a_learnable_C():
    n = torch.tensor([0.6], dtype=F64)
    C = torch.tensor([1.3], dtype=F64, requires_grad=True)
    dp = torch.tensor([-2.0, 2.0], dtype=F64).reshape(2, 1)

    def f(c):
        el = UpstreamDensityPowerLaw(
            c, n, src=torch.tensor([0]), tgt=torch.tensor([1]), m=0.5, rho_ref=RHO_REF
        )
        return el.flow(dp, _drivers())

    assert torch.autograd.gradcheck(f, (C,), eps=1e-6, atol=1e-6)


def test_learnable_registers_C_and_n_and_keeps_the_index_buffers_undifferentiable():
    el = _element(0.5, learnable=True)
    assert sorted(name for name, _ in el.named_parameters()) == ["C", "n"]
    buffers = dict(el.named_buffers())
    assert set(buffers) == {"src", "tgt", "m"}
    assert buffers["src"].dtype is torch.long and buffers["tgt"].dtype is torch.long
    assert not any(b.requires_grad for b in buffers.values())


# ------------------------------------------------------------------ errors name the offender
def test_missing_density_driver_raises_key_error_naming_the_key_and_the_kind():
    el = _element(0.5)
    with pytest.raises(KeyError, match=r"rho.*not found"):
        el.flow(torch.tensor([1.0], dtype=F64), {})
    with pytest.raises(KeyError, match=r"rho.*not found"):
        el.flow(torch.tensor([1.0], dtype=F64), None)


def test_mismatched_endpoint_shapes_raise_value_error():
    with pytest.raises(ValueError, match=r"src has shape .*tgt has shape"):
        UpstreamDensityPowerLaw(
            torch.tensor([1.0], dtype=F64),
            torch.tensor([0.5], dtype=F64),
            src=torch.tensor([0, 0]),
            tgt=torch.tensor([1]),
            m=0.5,
        )


def test_two_dimensional_endpoints_raise_value_error():
    with pytest.raises(ValueError, match=r"src/tgt must be 1-D"):
        UpstreamDensityPowerLaw(
            torch.tensor([1.0], dtype=F64),
            torch.tensor([0.5], dtype=F64),
            src=torch.tensor([[0]]),
            tgt=torch.tensor([[1]]),
            m=0.5,
        )


def test_a_negative_node_position_raises_value_error_showing_both_endpoint_lists():
    """-1 is CONTAM's own spelling of "ambient" in a .prj path record; it is NOT a node
    POSITION, and silently taken as one it would index the last node from the end."""
    with pytest.raises(ValueError, match=r"non-negative node positions.*\[-1\]"):
        UpstreamDensityPowerLaw(
            torch.tensor([1.0], dtype=F64),
            torch.tensor([0.5], dtype=F64),
            src=torch.tensor([-1]),
            tgt=torch.tensor([1]),
            m=0.5,
        )


def test_an_exponent_of_the_wrong_length_raises_value_error_naming_the_edge_count():
    with pytest.raises(ValueError, match=r"m has shape \(3,\).*per edge \(2\)"):
        UpstreamDensityPowerLaw(
            torch.tensor([1.0, 1.0], dtype=F64),
            torch.tensor([0.5, 0.5], dtype=F64),
            src=torch.tensor([0, 0]),
            tgt=torch.tensor([1, 1]),
            m=torch.tensor([0.0, 0.5, 1.0], dtype=F64),
        )


def test_a_non_positive_reference_density_raises_value_error():
    with pytest.raises(ValueError, match=r"rho_ref must be strictly positive"):
        _element_with_rho_ref(0.0)


def _element_with_rho_ref(rho_ref):
    return UpstreamDensityPowerLaw(
        torch.tensor([1.0], dtype=F64),
        torch.tensor([0.5], dtype=F64),
        src=torch.tensor([0]),
        tgt=torch.tensor([1]),
        m=0.5,
        rho_ref=rho_ref,
    )


def test_a_dp_of_the_wrong_width_raises_value_error_naming_both_widths():
    """A single-edge element broadcasts against any dp (PowerLaw's convention); a
    multi-edge one must not, or a width-1 dp would silently give every edge edge 0's
    endpoints."""
    el = UpstreamDensityPowerLaw(
        torch.tensor([1.0, 1.0], dtype=F64),
        torch.tensor([0.5, 0.5], dtype=F64),
        src=torch.tensor([0, 1]),
        tgt=torch.tensor([1, 0]),
        m=0.5,
        rho_ref=RHO_REF,
    )
    with pytest.raises(ValueError, match=r"dp has 3 columns.*covers 2 edges"):
        el.flow(torch.tensor([1.0, 2.0, 3.0], dtype=F64), _drivers())
    with pytest.raises(ValueError, match=r"dp has 1 columns.*covers 2 edges"):
        el.flow(torch.tensor([1.0], dtype=F64), _drivers())


def test_a_density_driver_too_short_for_the_endpoints_raises_value_error():
    el = _element(0.5)
    with pytest.raises(ValueError, match=r"'rho' has 1 node"):
        el.flow(torch.tensor([1.0], dtype=F64), {"rho": torch.tensor([1.2], dtype=F64)})


def test_a_non_positive_density_raises_value_error_naming_the_node_instead_of_nan():
    """Without this check `rho = [-1.0, 1.2]` silently produces `flow() == nan`: a negative
    density raised to `m = 1/2` is a complex number, and torch collapses that to `nan` with
    no exception anywhere. The offending node's own index must be named."""
    el = _element(0.5)
    bad = {"rho": torch.tensor([-1.0, 1.2], dtype=F64)}
    with pytest.raises(ValueError, match=r"not strictly positive at node index/indices \[0\]"):
        el.flow(torch.tensor([1.0], dtype=F64), bad)


def test_a_zero_density_at_the_target_node_also_raises_naming_it():
    el = _element(0.5)
    bad = {"rho": torch.tensor([1.2, 0.0], dtype=F64)}
    with pytest.raises(ValueError, match=r"not strictly positive at node index/indices \[1\]"):
        el.flow(torch.tensor([1.0], dtype=F64), bad)


# ------------------------------------------------------------------ in a layer
def test_a_buoyant_two_orifice_stack_beats_the_uncorrected_law_in_both_directions():
    """ambient (boundary) -> zone through two openings at different heights, the classic
    one-zone stack. With the ambient COLDER than the zone, the upstream density on the
    inflow path exceeds the reference, so the corrected mass flow exceeds the uncorrected
    one; warming the ambient past the zone reverses both the flow and the inequality.
    """
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    net.add_node("zone", z_ref=0.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=0.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=1.5)
    src, tgt = net.endpoints("airpath")
    C = torch.full((2,), 0.141421 * RHO_REF**0.5, dtype=F64)
    n = torch.full((2,), 0.5, dtype=F64)
    from noodl.drives import Stack

    drive = Stack.from_network(net, "airpath")
    plain = PotentialFlowLayer(net, "air", [PowerLaw(C, n)], drives=[drive],
                               boundary=["ambient"])
    corrected = PotentialFlowLayer(
        net,
        "air",
        [UpstreamDensityPowerLaw(C, n, src=src, tgt=tgt, m=0.5, rho_ref=RHO_REF)],
        drives=[drive],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=F64)
    for T_amb, expect_inflow in ((273.15, True), (313.15, False)):
        rho = 101325.0 / (287.055 * torch.tensor([T_amb, 293.15], dtype=F64))
        drv = {"rho": rho}
        _, q_plain = plain.solve(phi_b, drv, differentiable=False, atol=1e-12, rtol=1e-12)
        _, q_corr = corrected.solve(phi_b, drv, differentiable=False, atol=1e-12, rtol=1e-12)
        assert (q_plain[0].item() > 0) is expect_inflow
        assert torch.sign(q_corr).tolist() == torch.sign(q_plain).tolist()
        # The zone sits at 293.15 K, whose density IS rho_ref, so whichever opening carries
        # air out of the zone is unscaled and the whole effect comes from the ambient side:
        # cold (dense) ambient in gives more mass flow than the reference law, warm less.
        if expect_inflow:
            assert q_corr.abs().max().item() > q_plain.abs().max().item()
        else:
            assert q_corr.abs().max().item() < q_plain.abs().max().item()


def test_a_gradient_reaches_the_density_driver_through_a_differentiable_solve():
    """The binding contract: nothing on the path from `drivers['rho']` to the solved flow may
    be detached. The implicit-function backward reaches the density through BOTH the `Stack`
    drive and this element's coefficient, and the two contributions have opposite signs, so a
    detached coefficient would not merely lose a term -- it would change the sign of dq/drho
    on the warm side. Checked against a central difference on the solve itself.
    """
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    net.add_node("zone", z_ref=0.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=0.0)
    net.add_edge("ambient", "zone", kind="airpath", z_path=1.5)
    src, tgt = net.endpoints("airpath")
    from noodl.drives import Stack

    layer = PotentialFlowLayer(
        net,
        "air",
        [
            UpstreamDensityPowerLaw(
                torch.full((2,), 0.141421 * RHO_REF**0.5, dtype=F64),
                torch.full((2,), 0.5, dtype=F64),
                src=src,
                tgt=tgt,
                m=0.5,
                rho_ref=RHO_REF,
            )
        ],
        drives=[Stack.from_network(net, "airpath")],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=F64)
    rho0 = torch.tensor([1.2922, 1.2041], dtype=F64)

    def total(rho):
        _, q = layer.solve(phi_b, {"rho": rho}, atol=1e-13, rtol=1e-13)
        return q[0]

    rho = rho0.clone().requires_grad_(True)
    total(rho).backward()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    h = 1e-7
    for i in range(2):
        step = torch.zeros(2, dtype=F64)
        step[i] = h
        fd = (total(rho0 + step) - total(rho0 - step)).item() / (2 * h)
        assert rho.grad[i].item() == pytest.approx(fd, rel=1e-5, abs=1e-9)
