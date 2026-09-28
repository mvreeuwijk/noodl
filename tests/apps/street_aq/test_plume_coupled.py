"""The coupled street / above-roof solve (`above_roof.street_steady_with_plume`).

The street solve is linear in its per-street background, `x = A c + b`, and the fluxes are
linear in `(x, c)`, so the converged `C_ext` has a closed form: every test that checks a
value builds `A`, `b` and the flux matrices from the model itself and solves the linear
system directly.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from noodl.apps.street_aq.above_roof import (
    roof_fluxes,
    roof_wiring,
    street_steady_with_plume,
)
from noodl.apps.street_aq.network import (
    Street,
    StreetNetwork,
    build_model,
    munich_idealised,
    street_index,
)
from noodl.apps.street_aq.plume import junction_kernel, plume_table, street_kernel

DT = torch.float64
H = 20.0


def _drivers(model, *, theta=0.3, u_star=0.5, bg=2e-8, batch=()):
    net = model.net
    n = len(street_index(model))
    sources = torch.zeros(*batch, net.n, dtype=DT)
    for i, name in enumerate(street_index(model)):
        sources[..., net.node_index(name)] = 1e-6 * (i + 1)
    return {
        "street.x_boundary": torch.full((*batch, n), bg, dtype=DT),
        "street.sources": sources,
        "u_star": torch.as_tensor(u_star, dtype=DT),
        "theta_w": torch.as_tensor(theta, dtype=DT),
        "h_abl": torch.tensor(800.0, dtype=DT),
    }


def _kernels(net, *, theta=0.3, u_star=0.5):
    table = plume_table(u_star=u_star, h_abl=800.0, lmo=math.inf, h_canopy=H, z0=1.0,
                        d=13.0, sigma_theta=math.radians(5.0))
    return (street_kernel(net, table, theta_w=theta),
            junction_kernel(net, table, theta_w=theta))


def _linear_parts(model, state, drivers):
    """`(A, b, D, V_out, V_in)`: `x = A c + b`, `F_s = D (x - c)`, `F_j = V_out x - V_in c`,
    all from the model (one steady solve per street for `A`)."""
    n = len(street_index(model))
    zero = torch.zeros(n, dtype=DT)
    b = model.steady(state, {**drivers, "street.x_boundary": zero})["street.x"]
    no_src = {**drivers, "street.sources": torch.zeros_like(drivers["street.sources"])}
    a = torch.stack([model.steady(state, {**no_src, "street.x_boundary": torch.eye(
        n, dtype=DT)[j]})["street.x"] for j in range(n)], dim=-1)
    w = roof_wiring(model)
    q = model.current_flows("street", state, drivers)
    d = torch.diag(q[w.exchange_col])
    v_out = torch.zeros(w.n_junctions, n, dtype=DT)
    v_out.index_put_((w.vent_out_junction, w.vent_out_street), q[w.vent_out_col],
                     accumulate=True)
    v_in = torch.zeros(w.n_junctions, n, dtype=DT)
    v_in.index_put_((w.vent_in_junction, w.vent_in_street), q[w.vent_in_col],
                    accumulate=True)
    return a, b, d, v_out, v_in


def _closed_form(model, state, drivers, k_s, k_j, *, net_junction=False):
    a, b, d, v_out, v_in = _linear_parts(model, state, drivers)
    if not net_junction:
        v_in = torch.zeros_like(v_in)
    n = a.shape[0]
    eye = torch.eye(n, dtype=DT)
    g = k_s @ d @ (a - eye)
    bg = drivers["street.x_boundary"]
    rhs = bg + k_s @ d @ b
    if k_j is not None:
        # Junction sources are the excess over background: F_up = V_out (x - bg),
        # F_down = V_in (c - bg).
        g = g + k_j @ (v_out @ a - v_in)
        rhs = rhs + k_j @ v_out @ (b - bg) + k_j @ v_in @ bg
    c = torch.linalg.solve(eye - g, rhs)
    return c, a @ c + b, g


def _mid_ratios(changes: list[float]) -> list[float]:
    """Pass-to-pass ratios of the relative change, dropping the first three passes (the
    nilpotent transient) and every change below 1e-13 (round-off)."""
    kept = [c for c in changes[3:] if c > 1e-13]
    assert len(kept) >= 3
    return [b / a for a, b in zip(kept, kept[1:], strict=False)]


def _lattice(side: int, *, block: float = 100.0) -> StreetNetwork:
    """A `side` x `side` junction lattice, streets of `block` m, W = H = 20 m."""
    x, y, streets = {}, {}, []
    for i in range(side):
        for j in range(side):
            x[f"{i},{j}"], y[f"{i},{j}"] = i * block, j * block
    for i in range(side):
        for j in range(side):
            for di, dj in ((1, 0), (0, 1)):
                a, b = f"{i},{j}", f"{i + di},{j + dj}"
                if b in x:
                    streets.append(Street(f"{a}-{b}", a, b, block, 20.0, H))
    return StreetNetwork(streets, x, y)


@pytest.fixture(scope="module")
def lattice():
    net = _lattice(4)
    model, state, _ = build_model(net, background="per_street")
    return net, model, state


@pytest.fixture(scope="module")
def munich():
    net, _ = munich_idealised(L=100.0, W=20.0, H=H)
    model, state, _ = build_model(net, background="per_street")
    return net, model, state


# ------------------------------------------------------------------------ the checks


def test_zero_kernel_is_the_plain_per_street_solve_exactly(munich):
    net, model, state = munich
    d = _drivers(model)
    n = len(net.streets)
    out = street_steady_with_plume(model, state, d, kernel=torch.zeros(n, n, dtype=DT),
                                   junction_kernel=torch.zeros(n, 12, dtype=DT))
    plain = model.steady(state, d)["street.x"]
    assert torch.equal(out["street.x"], plain)
    assert torch.equal(out["street.c_ext"], d["street.x_boundary"])
    # Nothing required grad, so nothing carries a graph (no adjoint was attached).
    assert not any(v.requires_grad for v in out.values())


def test_two_streets_single_emitter_match_the_closed_form_2x2_solution():
    x = {"a": 0.0, "b": 100.0, "c": 200.0}
    y = {"a": 0.0, "b": 0.0, "c": 0.0}
    net = StreetNetwork([Street("s0", "a", "b", 100.0, 20.0, H),
                         Street("s1", "b", "c", 100.0, 20.0, H)], x, y)
    model, state, _ = build_model(net, background="per_street")
    d = _drivers(model, theta=0.0)
    d["street.sources"] = torch.zeros(model.net.n, dtype=DT)
    d["street.sources"][model.net.node_index("s0")] = 1e-5
    k_s = torch.tensor([[0.0, 0.0], [3e-3, 0.0]], dtype=DT)  # s0 upwind of s1
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=None,
                                   tol=1e-15)
    # By hand: x = A c + b, F = D (x - c), c = c_bg + K F  ->  (I - K D (A - I)) c = ...
    c, xs, _ = _closed_form(model, state, d, k_s, None)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-12, atol=0)
    torch.testing.assert_close(out["street.x"], xs, rtol=1e-12, atol=0)
    c_ext = out["street.c_ext"].detach()
    assert float(c_ext[1]) > float(c_ext[0]) == 2e-8


def test_real_kernels_match_the_closed_form_with_junction_fluxes(munich):
    net, model, state = munich
    d = _drivers(model)
    k_s, k_j = _kernels(net)
    diag: dict = {}
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   tol=1e-15, diagnostics=diag)
    c, xs, g = _closed_form(model, state, d, k_s, k_j)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-12, atol=0)
    torch.testing.assert_close(out["street.x"], xs, rtol=1e-12, atol=0)
    assert bool((c > d["street.x_boundary"]).any())
    # Every plume link points downwind: the coupling is nilpotent here, and the iteration
    # stops after as many passes as there are downwind levels (plus the one that sees no
    # change).
    rho = float(torch.linalg.eigvals(g).abs().max())
    print(f"\nmunich_idealised, upward junction source: {diag['passes']} passes, "
          f"spectral radius {rho:.1e}, changes {diag['changes']}")
    assert rho < 1e-12
    assert diag["passes"] <= 8


def test_net_junction_source_converges_at_the_spectral_radius(munich):
    """`junction_source="net"` subtracts the vent inflow from the junction's plume source;
    that sink feeds back on the receiving street's own C_ext, and the iteration then
    contracts at exactly the spectral radius of the coupling."""
    net, model, state = munich
    d = _drivers(model, theta=1.0)
    k_s, k_j = _kernels(net, theta=1.0)
    diag: dict = {}
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   junction_source="net", tol=1e-15, diagnostics=diag)
    c, _, g = _closed_form(model, state, d, k_s, k_j, net_junction=True)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-12, atol=0)
    rho = float(torch.linalg.eigvals(g).abs().max())
    print(f"\nmunich_idealised, net junction source, 1 rad: {diag['passes']} passes, "
          f"measured contraction {diag['contraction']:.4e}, spectral radius {rho:.4e}")
    assert 0.01 < rho < 0.1
    # Past the nilpotent transient and before round-off, the change shrinks by rho per
    # pass (to within 3 %: the double eigenvalue is defective, a 2x2 Jordan block from the
    # two mirror-image junctions, which adds the slowly fading factor (k + 1) / k).
    assert statistics.median(_mid_ratios(diag["changes"])) == pytest.approx(rho, rel=0.03)


@pytest.mark.parametrize("theta, rho", [(0.3, 0.2002), (0.0, 0.3081)])
def test_net_junction_source_spectral_radius_on_munich(munich, theta, rho):
    """The figures in `above_roof`'s docstring: with the net junction source the coupling on
    `munich_idealised` has spectral radius 0.20 at 0.3 rad and 0.31 at 0 rad; with the
    upward source it is exactly 0."""
    net, model, state = munich
    d = _drivers(model, theta=theta)
    k_s, k_j = _kernels(net, theta=theta)
    _, _, g = _closed_form(model, state, d, k_s, k_j, net_junction=True)
    assert float(torch.linalg.eigvals(g).abs().max()) == pytest.approx(rho, abs=5e-4)
    _, _, g_up = _closed_form(model, state, d, k_s, k_j)
    assert float(torch.linalg.eigvals(g_up).abs().max()) < 1e-12


def test_a_coupling_above_one_is_refused_by_name(lattice):
    """A coupling whose spectral radius exceeds one (here the head-on net source with the
    junction kernel scaled by 4: eigenvalue -1.23) diverges under plain substitution and is
    refused by name; under-relaxation 0.5 (relaxed eigenvalue |1 - 0.5 (1 + 1.23)| = 0.12)
    converges to the closed form."""
    net, model, state = lattice
    d = _drivers(model, theta=0.0)
    k_s, k_j = _kernels(net, theta=0.0)
    k_j = 4.0 * k_j
    _, _, g = _closed_form(model, state, d, k_s, k_j, net_junction=True)
    assert float(torch.linalg.eigvals(g).real.min()) == pytest.approx(-1.2326, abs=1e-3)
    with pytest.raises(RuntimeError, match="did not converge within 40 passes"):
        street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                 junction_source="net", max_iter=40)
    c, _, _ = _closed_form(model, state, d, k_s, k_j, net_junction=True)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   junction_source="net", relaxation=0.5, tol=1e-14)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-11, atol=0)


def test_net_junction_source_head_on_contracts_and_relaxation_keeps_the_fixed_point(
        lattice):
    """Wind along the 4 x 4 lattice (0 rad): the net source's self-feedback is its most
    negative eigenvalue, -0.31 (the junction plume's `P_z` is capped at `1/H`, which keeps
    the loop gain of the sink below one), so plain substitution converges; under-relaxation
    0.5 converges to the same fixed point at the rate of the relaxed map."""
    net, model, state = lattice
    d = _drivers(model, theta=0.0)
    k_s, k_j = _kernels(net, theta=0.0)
    c, _, g = _closed_form(model, state, d, k_s, k_j, net_junction=True)
    lam = torch.linalg.eigvals(g)
    assert float(lam.real.min()) == pytest.approx(-0.3081, abs=1e-3)
    plain = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                     junction_source="net", tol=1e-14)
    torch.testing.assert_close(plain["street.c_ext"], c, rtol=1e-11, atol=0)
    diag: dict = {}
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   junction_source="net", relaxation=0.5, tol=1e-14,
                                   diagnostics=diag)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-11, atol=0)
    # The relaxed map's eigenvalues are 1 - w + w lambda: 0.5 from every zero eigenvalue
    # and 0.35 from the sink; the rate is the larger (the median ratio, to 5 %: the 0.5
    # sits in Jordan blocks, the images of the nilpotent downwind chains, whose
    # (k + 1) / k factor lifts the late ratios slightly).
    relaxed = float((1 - 0.5 + 0.5 * lam).abs().max())
    assert relaxed == pytest.approx(0.5, rel=1e-9)
    assert statistics.median(_mid_ratios(diag["changes"])) == pytest.approx(relaxed,
                                                                            rel=0.05)


def test_the_relation_holds_to_tol_at_return(munich):
    net, model, state = munich
    d = _drivers(model)
    k_s, k_j = _kernels(net)
    for tol in (1e-6, 1e-12):
        out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                       tol=tol)
        w = roof_wiring(model)
        q = model.current_flows("street", state, d)
        c = out["street.c_ext"]
        f_s, up, _down = roof_fluxes(w, q, out["street.x"][:, None], c[:, None],
                                     d["street.x_boundary"][:, None])
        torch.testing.assert_close(f_s[:, 0], out["street.roof_flux"], rtol=1e-6, atol=0)
        image = d["street.x_boundary"] + k_s @ f_s[:, 0] + k_j @ up[:, 0]
        assert float((image - c).abs().max()) <= 2 * tol * float(c.abs().max())


def test_canopy_mass_balance_emissions_leave_through_roofs_and_junctions(munich):
    net, model, state = munich
    d = _drivers(model)
    k_s, k_j = _kernels(net)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j)
    emitted = float(d["street.sources"].sum())
    left = float(out["street.roof_flux"].sum() + out["street.junction_flux"].sum()
                 - out["street.junction_downflux"].sum())
    assert left == pytest.approx(emitted, rel=1e-10)
    assert bool((out["street.junction_flux"] >= 0).all())
    assert float(out["street.junction_downflux"].sum()) > 0


def test_gradient_wrt_an_emission_matches_finite_differences(munich):
    net, model, state = munich
    k_s, k_j = _kernels(net)
    base = _drivers(model)
    node = model.net.node_index("11")

    def objective(e11: torch.Tensor) -> torch.Tensor:
        d = dict(base)
        d["street.sources"] = base["street.sources"].clone()
        d["street.sources"] = d["street.sources"].index_put((torch.tensor([node]),), e11)
        out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                       tol=1e-15)
        return out["street.x"].sum() + 1e3 * out["street.c_ext"].sum()

    e = torch.tensor([2e-5], dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(e), e)
    h = 1e-7
    with torch.no_grad():
        fd = (objective(e + h) - objective(e - h)) / (2 * h)
    assert bool(torch.isfinite(grad).all())
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_gradient_wrt_friction_velocity_through_flows_and_kernels(munich):
    """u* moves the exchange velocity, the canyon winds AND the kernels: the implicit adjoint
    must carry all three through the converged fixed point."""
    net, model, state = munich

    def objective(u_star: torch.Tensor) -> torch.Tensor:
        k_s, k_j = _kernels(net, u_star=u_star)
        d = _drivers(model, u_star=u_star)
        out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                       tol=1e-15)
        return 1e8 * out["street.c_ext"].sum()

    u = torch.tensor(0.5, dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(u), u)
    h = 1e-5
    with torch.no_grad():
        fd = (objective(u + h) - objective(u - h)) / (2 * h)
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


@pytest.mark.parametrize("junction_source", ["upward", "net"])
def test_gradient_wrt_the_background_matches_finite_differences(munich, junction_source):
    """The background enters both the affine term `C_bg` of `C_ext = C_bg + K F` and the
    excess-over-background fluxes, so its adjoint path differs from an emission's."""
    net, model, state = munich
    k_s, k_j = _kernels(net, theta=1.0)
    base = _drivers(model, theta=1.0)
    n = len(street_index(model))
    profile = 1.0 + 0.1 * torch.arange(n, dtype=DT)

    def objective(bg: torch.Tensor) -> torch.Tensor:
        d = {**base, "street.x_boundary": bg * profile}
        out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                       junction_source=junction_source, tol=1e-15)
        return 1e8 * (out["street.x"].sum() + out["street.c_ext"].sum())

    bg = torch.tensor(2e-8, dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(bg), bg)
    h = 1e-10
    with torch.no_grad():
        fd = (objective(bg + h) - objective(bg - h)) / (2 * h)
    assert abs(float(grad)) > 0
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_gradient_with_the_net_junction_source_under_relaxation(lattice):
    """`junction_source="net"` on the head-on lattice, where only under-relaxation
    converges: the implicit adjoint of the relaxed iteration matches finite differences."""
    net, model, state = lattice
    k_s, k_j = _kernels(net, theta=0.0)
    base = _drivers(model, theta=0.0)

    def objective(scale: torch.Tensor) -> torch.Tensor:
        d = {**base, "street.sources": scale * base["street.sources"]}
        out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                       junction_source="net", relaxation=0.5, tol=1e-15)
        return 1e8 * out["street.c_ext"].sum()

    s = torch.tensor(1.0, dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(s), s)
    h = 1e-4
    with torch.no_grad():
        fd = (objective(s + h) - objective(s - h)) / (2 * h)
    assert abs(float(grad)) > 0
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_hours_batch_like_model_steady(munich):
    net, model, state = munich
    thetas = torch.tensor([0.3, 2.0], dtype=DT)
    u_stars = torch.tensor([0.5, 0.3], dtype=DT)
    d = _drivers(model, theta=thetas, u_star=u_stars, batch=(2,))
    k_s, k_j = _kernels(net, theta=thetas, u_star=u_stars)
    assert k_s.shape == (2, 12, 12)
    diag: dict = {}
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   tol=1e-14, diagnostics=diag)
    assert out["street.x"].shape == (2, 12) and out["street.c_ext"].shape == (2, 12)
    assert out["street.junction_flux"].shape == (2, 12)
    for h in range(2):
        dh = _drivers(model, theta=float(thetas[h]), u_star=float(u_stars[h]))
        one = street_steady_with_plume(model, state, dh, kernel=k_s[h],
                                       junction_kernel=k_j[h], tol=1e-14)
        torch.testing.assert_close(out["street.c_ext"][h], one["street.c_ext"],
                                   rtol=1e-12, atol=0)
    # A batched gradient runs the per-hour adjoint.
    src = d["street.sources"].clone().requires_grad_(True)
    res = street_steady_with_plume(model, state, {**d, "street.sources": src}, kernel=k_s,
                                   junction_kernel=k_j, tol=1e-14, diagnostics=diag)
    (g,) = torch.autograd.grad(res["street.c_ext"].sum(), src)
    assert diag["batched_adjoint"] and bool(torch.isfinite(g).all())


def test_species_axis(munich):
    net, _, _ = munich
    model, state, _ = build_model(net, background="per_street", species=("no", "no2"))
    d = _drivers(model)
    d["street.sources"] = torch.stack([d["street.sources"], 2 * d["street.sources"]], -1)
    d["street.x_boundary"] = torch.full((12, 2), 2e-8, dtype=DT)
    k_s, k_j = _kernels(net)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j)
    assert out["street.c_ext"].shape == (12, 2)
    single, s1, _ = build_model(net, background="per_street")
    d1 = _drivers(single)
    one = street_steady_with_plume(single, s1, d1, kernel=k_s, junction_kernel=k_j)
    torch.testing.assert_close(out["street.c_ext"][:, 0], one["street.c_ext"],
                               rtol=1e-12, atol=0)


def test_refusals_are_named(munich):
    net, model, state = munich
    uniform, us, _ = build_model(net)
    k_s, k_j = _kernels(net)
    with pytest.raises(ValueError, match="background='per_street'"):
        street_steady_with_plume(uniform, us, _drivers(uniform), kernel=k_s,
                                 junction_kernel=k_j)
    d = _drivers(model)
    with pytest.raises(ValueError, match="kernel must end in"):
        street_steady_with_plume(model, state, d, kernel=k_s[:3, :3], junction_kernel=k_j)
    with pytest.raises(ValueError, match="junction_kernel must end in"):
        street_steady_with_plume(model, state, d, kernel=k_s,
                                 junction_kernel=k_s[:, :5])
    with pytest.raises(ValueError, match="relaxation"):
        street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                 relaxation=0.0)
    with pytest.raises(ValueError, match="junction_source"):
        street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                 junction_source="down")
    with pytest.raises(TypeError):
        street_steady_with_plume(model, state, d, kernel=k_s)  # junction_kernel required


# ------------------------------------------------------- neutrality to the background


@pytest.mark.parametrize("junction_source", ["upward", "net"])
@pytest.mark.parametrize("case,theta", [("lattice", 0.3), ("lattice", 0.0),
                                        ("munich", 0.3), ("munich", 5.0)])
def test_no_emission_leaves_c_ext_at_a_uniform_background(case, theta, junction_source):
    """Junction sources are the EXCESS over background: with nothing emitted and a uniform
    background every street sits at C_bg and C_ext must stay there. A junction source of
    `sum_out q C_street` (not the excess) would not: it gives max C_ext / C_bg = 1.095 on
    the 7 x 7 lattice at 0.3 rad (asserted below; the figure in `above_roof`)."""
    net = _lattice(7) if case == "lattice" else munich_idealised(L=100.0, W=20.0, H=H)[0]
    model, state, _ = build_model(net, background="per_street")
    d = _drivers(model, theta=theta)
    d["street.sources"] = torch.zeros_like(d["street.sources"])
    k_s, k_j = _kernels(net, theta=theta)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   junction_source=junction_source)
    torch.testing.assert_close(out["street.c_ext"], d["street.x_boundary"], rtol=1e-12,
                               atol=0)
    torch.testing.assert_close(out["street.x"], d["street.x_boundary"], rtol=1e-12, atol=0)
    if (case, theta, junction_source) == ("lattice", 0.3, "upward"):
        _, _, _, v_out, _ = _linear_parts(model, state, d)
        ones = torch.ones(len(net.streets), dtype=DT)
        assert float((1.0 + k_j @ v_out @ ones).max()) == pytest.approx(1.095, abs=5e-4)


# ------------------------------------------------------- the upward junction path


def test_upward_junction_plumes_reach_streets_and_match_the_closed_form(lattice):
    net, model, state = lattice
    d = _drivers(model, theta=0.3)
    k_s, k_j = _kernels(net, theta=0.3)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   tol=1e-15)
    c, xs, _ = _closed_form(model, state, d, k_s, k_j)
    torch.testing.assert_close(out["street.c_ext"], c, rtol=1e-12, atol=0)
    torch.testing.assert_close(out["street.x"], xs, rtol=1e-12, atol=0)
    # The junction term is really there: a sizeable share of the plume rise.
    from_junctions = (k_j @ out["street.junction_flux"]).detach()
    rise = (out["street.c_ext"] - d["street.x_boundary"]).detach()
    share = float(from_junctions.abs().max() / rise.abs().max())
    print(f"\n4x4 lattice, 0.3 rad: junction share of the largest C_ext rise {share:.3f}")
    assert share > 0.05
    assert bool((out["street.junction_flux"] > 0).any())


def test_gradient_through_the_junction_kernel_matches_finite_differences(lattice):
    net, model, state = lattice
    d = _drivers(model, theta=0.3)
    k_s, k_j = _kernels(net, theta=0.3)

    def objective(scale: torch.Tensor) -> torch.Tensor:
        out = street_steady_with_plume(model, state, d, kernel=k_s,
                                       junction_kernel=scale * k_j, tol=1e-15)
        return 1e8 * out["street.c_ext"].sum()

    s = torch.tensor(1.0, dtype=DT, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(s), s)
    h = 1e-4
    with torch.no_grad():
        fd = (objective(s + h) - objective(s - h)) / (2 * h)
    assert abs(float(grad)) > 0
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_convergence_is_judged_per_hour(munich):
    """A clean hour (background 1e-3 of the other's) must converge on its own scale."""
    net, model, state = munich
    thetas = torch.tensor([0.3, 2.0], dtype=DT)
    u_stars = torch.tensor([0.5, 0.3], dtype=DT)
    d = _drivers(model, theta=thetas, u_star=u_stars, batch=(2,))
    d["street.x_boundary"] = torch.stack([torch.full((12,), 2e-8, dtype=DT),
                                          torch.full((12,), 2e-11, dtype=DT)])
    d["street.sources"] = d["street.sources"] * torch.tensor([[1.0], [1e-3]], dtype=DT)
    k_s, k_j = _kernels(net, theta=thetas, u_star=u_stars)
    out = street_steady_with_plume(model, state, d, kernel=k_s, junction_kernel=k_j,
                                   tol=1e-13)
    dh = _drivers(model, theta=2.0, u_star=0.3, bg=2e-11)
    dh["street.sources"] = dh["street.sources"] * 1e-3
    one = street_steady_with_plume(model, state, dh, kernel=k_s[1], junction_kernel=k_j[1],
                                   tol=1e-15)
    torch.testing.assert_close(out["street.c_ext"][1], one["street.c_ext"], rtol=1e-12,
                               atol=0)
