"""Conservation and gradients on the committed tree (rows C1, C2, C3)."""

import pytest
import torch

from noodl.apps.sewer.network import build_model, sewer_steady, tree_steady

F64 = torch.float64


def test_c1_the_air_layer_balances_at_its_solution():
    """Row C1: the nodal residual of the solved air layer and Tellegen's own power identity.
    The plan measured 6.3e-15 / 8.1e-13 on the A3 configuration with the headspace stack
    path at the mean invert; with the path at the pipe CROWN (ruling M4-R19) the stack head
    is larger and, with the leak's `Orifice`-built `C` still float32 (the repository's
    default dtype, N2), the measured residual was 1.352e-12 kg/s / power 1.855e-11 W -- an
    ARITHMETIC floor, not a stopping criterion (Newton asked for atol = 1e-14 stalls at
    exactly that value). N2 found that floor was MOSTLY the float32 leak coefficient: with
    the leak's `PowerLaw` built directly in float64 (never through `Orifice`, which casts
    with `torch.get_default_dtype()` regardless of its inputs' own dtype), the measured
    figures drop by almost three orders of magnitude to residual 4.518e-13 kg/s / power
    6.141e-12 W. Ruling M4-R20b's tolerances (1e-11 on the residual, 1e-10 on the power
    identity) stay unchanged -- both float64 figures still clear them comfortably, with
    considerably more margin than the float32 ones did."""
    model, state, drivers = build_model(tree_steady())
    new = model.step(state, drivers, 60.0)
    residuals = model.residuals(new, drivers)
    assert float(residuals["air"].abs().max()) < 1e-11
    layer = model.potential["air"]
    resolved = model._apply_closures(new, drivers)
    power = layer.power_residual(new["air.phi"], new["air.q"], resolved)
    assert float(power.abs().max()) < 1e-10


def test_c1_the_water_flows_balance_every_manhole():
    """Continuity on the pipe tree, node by node: exact, by construction."""
    model, state, drivers = build_model(tree_steady(), air=False, quality=False)
    resolved = model._apply_closures(state, drivers)
    q = resolved["sewer.q"]
    src, tgt = model.net.endpoints("pipe")
    net_out = torch.zeros(model.net.n, dtype=F64)
    net_out = net_out.index_add(-1, src, q).index_add(-1, tgt, -q)
    balance = net_out - drivers["inflow"]
    # only the outfall absorbs
    interior = [i for i, n in enumerate(model.net.nodes) if n != "Outfall"]
    assert float(balance[interior].abs().max()) < 1e-15


def test_c2_cross_phase_sulfide_is_conserved_node_by_node():
    """Row C2, 1e-12: moles of S removed from the water equal moles added to the air.

    FR-21 (b): `build_model`'s default `sulfide_in = 0` (this test's `drivers`,
    unmodified), so `LateralLoads`'s own sulfide column is exactly zero everywhere and this
    invariant is unaffected by its presence -- the equality below is between `H2STransfer`'s
    OWN two source terms (`water_quality.sources[..., 1]` and `air_quality.sources`), which
    `H2STransfer` computes from a driver ("water_quality.sources") that already carries
    `LateralLoads`'s (zero, here) contribution ADDED IN before this closure runs, so a
    nonzero lateral BOD load elsewhere in the vector cannot leak into this comparison
    either: it lands in the BOD column, never the sulfide one this test reads."""
    from noodl.apps.sewer.quality import M_H2S, M_S

    model, state, drivers = build_model(tree_steady())
    state = dict(state)
    state["water_quality.x"] = torch.full_like(state["water_quality.x"], 1e-3)
    state["air_quality.x"] = torch.full_like(state["air_quality.x"], 1e-6)
    resolved = model._apply_closures(state, drivers)
    water = resolved["water_quality.sources"][..., 1]
    air = resolved["air_quality.sources"]
    moles_out = -water / M_S
    moles_in = air / M_H2S
    assert torch.allclose(moles_out, moles_in, rtol=1e-12, atol=1e-30)
    assert float(moles_out.abs().max()) > 0.0


def test_c3_gradients_against_central_differences():
    """Row C3, 1e-6 relative, through the tree solve, the Manning inversion AND the air
    Newton solve, with respect to the inflows.

    N6: the spec/README/docstring text names `f_i`, `f_air`, `T_head` and the leak area as
    well, but until this fix only the inflows were ever DIFFERENCED here -- `f_i`, `f_air`
    and `leak_area` were passed as plain Python floats into a FRESH `build_model`
    call inside `loss`, so no gradient tape ever reached them; the assertions below only
    ever exercised `inflow`. Three drivers/parameters are genuinely differentiable and are
    now each pinned by their own test:

    * `T_head` (a full-node/scalar DRIVER, spec 4.2): `test_c3_gradient_reaches_t_head`.
    * the leak area, via the leak element's own `C` (float64 after N2):
      `test_c3_gradient_reaches_a_learnable_leak_area`.
    * `f_air`: already covered by `test_c3_gradient_reaches_a_learnable_headspace_friction`
      below (finite, correct sign; not a central-difference row -- `f_air` is a `Headspace`
      element parameter reached the SAME way `leak_area` now is).

    `f_i` is NOT one of them: it is a `Drag` DRIVE's plain attribute (never a registered
    `nn.Parameter`), and `PotentialFlowLayer`'s own differentiable-solve safety check
    refuses `requires_grad=True` on a Drive's own tensor attribute by name (a Drive is
    captured by closure inside the differentiable solve, not threaded through
    `Function.apply`, so its gradient would otherwise be silently absent) --
    STRUCTURALLY not differentiable by this framework's design, not an oversight here.
    """

    def loss(inflow, f_i, f_air, leak_area):
        model, state, drivers = build_model(
            tree_steady(), quality=False, f_i=float(f_i), f_air=float(f_air),
            leak_area=float(leak_area),
        )
        drivers = dict(drivers)
        drivers["inflow"] = inflow
        new = model.step(state, drivers, 60.0)
        return new["air.q"].abs().sum()

    inflow = torch.zeros(7, dtype=F64)
    inflow[:3] = torch.tensor([0.05, 0.08, 0.03], dtype=F64)
    inflow = inflow.clone().requires_grad_(True)
    value = loss(inflow, 7.49e-4, 0.02, 8e-4)
    value.backward()
    analytic = inflow.grad.clone()
    # MEASURED (ruling M4-R20a): the plain central difference cannot reach 1e-6 here at ANY
    # single step. The loss carries the air Newton solve's arithmetic floor (~4e-12, see C1),
    # so its differencing noise is ~4e-12 / (2 eps |f'|), i.e. 1.4e-6 relative at eps = 1e-5;
    # and its truncation error grows as eps^2 and is large (component 0: 4.9e-5 relative at
    # eps = 1e-4, 4.95e-3 at 1e-3), so at eps = 2e-5 component 0 still misses by 1.95e-6.
    # Richardson extrapolation (4 fd(eps) - fd(2 eps)) / 3 removes the eps^2 term: at
    # eps = 1e-4 the three components agree to 3.17e-8, 3.16e-9 and 1.16e-8 relative (5e-5:
    # 6.4e-8 / 2.0e-7 / 3.8e-7; 2e-4: 5.5e-7 / 8.1e-9 / 3.0e-8). The row's tolerance
    # (rel 1e-6) is unchanged; only the differencing scheme moved.
    eps = 1e-4

    def central(i, h):
        up = inflow.detach().clone()
        up[i] += h
        down = inflow.detach().clone()
        down[i] -= h
        return (float(loss(up, 7.49e-4, 0.02, 8e-4))
                - float(loss(down, 7.49e-4, 0.02, 8e-4))) / (2 * h)

    for i in range(3):
        fd = (4.0 * central(i, eps) - central(i, 2.0 * eps)) / 3.0
        assert float(analytic[i]) == pytest.approx(fd, rel=1e-6, abs=1e-14)


def test_c3_gradient_reaches_t_head():
    """Row C3 (N6): `T_head` is a full-node/scalar DRIVER (spec 4.2) read by the
    `SewerHydraulics` closure's `_densities` and, through `rho_air_nodes`, by the air
    layer's `Stack` buoyancy drive -- a legitimate driver path, unlike `f_i` above.
    MEASURED (eps = 0.01, Richardson (4 fd(eps) - fd(2 eps)) / 3): relative error 3.57e-8
    against the analytic adjoint gradient, comfortably inside the row's 1e-6."""

    def loss(t_head):
        model, state, drivers = build_model(tree_steady(), quality=False)
        drivers = dict(drivers)
        drivers["T_head"] = torch.tensor(float(t_head), dtype=F64)
        new = model.step(state, drivers, 60.0)
        return float(new["air.q"].abs().sum())

    base = 293.15
    model, state, drivers = build_model(tree_steady(), quality=False)
    drivers = dict(drivers)
    t_head = torch.tensor(base, dtype=F64, requires_grad=True)
    drivers["T_head"] = t_head
    new = model.step(state, drivers, 60.0)
    new["air.q"].abs().sum().backward()
    analytic = float(t_head.grad)

    eps = 1e-2

    def central(h):
        return (loss(base + h) - loss(base - h)) / (2 * h)

    fd = (4.0 * central(eps) - central(2.0 * eps)) / 3.0
    assert analytic == pytest.approx(fd, rel=1e-6, abs=1e-14)


def test_c3_gradient_reaches_a_learnable_leak_area():
    """Row C3 (N6): the leak element's `C` (float64 after N2) reached the SAME way
    `test_c3_gradient_reaches_a_learnable_headspace_friction` reaches `f_air` -- a
    non-learnable element's already-registered `nn.Parameter` with `requires_grad_(True)`
    set on it afterward, never a fresh tensor built with `requires_grad=True` and passed
    into a NEW `build_model` call (which `PotentialFlowLayer`'s own safety check
    would refuse: an unregistered grad-tracked element attribute).

    `C = leak_cd * leak_area * sqrt(2 / rho_air)`, uniform over the manholes, so
    `d(loss)/d(leak_area) = sum_i(d(loss)/d(C_i)) * leak_cd * sqrt(2 / rho_air)` by the
    chain rule; the Richardson check perturbs `leak_area` itself (a `build_model`
    keyword, rebuilding the whole model, exactly as the inflow/`T_head` checks do) and
    compares against that chain-ruled analytic gradient. MEASURED (eps = 1e-3 * leak_area):
    relative error 9.67e-9, comfortably inside the row's 1e-6."""
    import math

    from noodl.apps.sewer.air import RHO_AIR_REF

    base = 8e-4
    leak_cd = 0.6

    def loss(leak_area):
        model, state, drivers = build_model(
            tree_steady(), quality=False, leak_area=float(leak_area), leak_cd=leak_cd,
        )
        new = model.step(state, drivers, 60.0)
        return float(new["air.q"].abs().sum())

    model, state, drivers = build_model(
        tree_steady(), quality=False, leak_area=base, leak_cd=leak_cd,
    )
    leak = next(el for el in model.potential["air"]._elements if el.kind == "leak")
    leak.C.requires_grad_(True)
    new = model.step(state, drivers, 60.0)
    new["air.q"].abs().sum().backward()
    k = leak_cd * math.sqrt(2.0 / RHO_AIR_REF)
    analytic = float(leak.C.grad.sum()) * k

    eps = 1e-3 * base

    def central(h):
        return (loss(base + h) - loss(base - h)) / (2 * h)

    fd = (4.0 * central(eps) - central(2.0 * eps)) / 3.0
    assert analytic == pytest.approx(fd, rel=1e-6, abs=1e-14)


def test_c3_gradient_reaches_a_learnable_headspace_friction():
    model, state, drivers = build_model(tree_steady(), quality=False)
    element = next(
        el for el in model.potential["air"]._elements if el.kind == "headspace"
    )
    element.f_air.requires_grad_(True)
    new = model.step(state, drivers, 60.0)
    new["air.q"].abs().sum().backward()
    assert element.f_air.grad is not None
    assert torch.isfinite(element.f_air.grad).all()


def test_the_golden_case_is_reproduced():
    """Row G1, 1e-10, against `tests/golden/sewer_tree.json`."""
    from tests.golden import load_golden

    golden = load_golden("sewer_tree")
    model, state, drivers = build_model(tree_steady())
    final = sewer_steady(model, state, drivers)
    for key, expected in golden.items():
        actual = final[key] if key in final else None
        assert actual is not None, key
        assert actual.flatten().tolist() == pytest.approx(expected, rel=1e-10, abs=1e-18)
