"""`Model.fixed_point`: the repeated pass, its judge and its implicit gradient, and the three
application loops (`street_steady`, `street_steady_with_plume`, `sewer_steady`) that run on it.

The application tests carry the OLD loop inline and compare: under `no_grad` the ported
function returns the same state after the same number of passes.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.sewer.network import build_model as sewer_build_model
from noodl.apps.sewer.network import sewer_steady, tree_steady
from noodl.apps.street_aq.above_roof import (
    roof_fluxes,
    roof_wiring,
    street_steady_with_plume,
)
from noodl.apps.street_aq.chemistry import photostationary_for_streets, street_steady
from noodl.apps.street_aq.network import (
    Street,
    StreetNetwork,
    street_index,
    twelve_street_grid,
)
from noodl.apps.street_aq.network import build_model as street_build_model
from noodl.apps.street_aq.plume import junction_kernel, plume_table, street_kernel

F64 = torch.float64


@pytest.fixture(scope="module")
def model():
    """Any model: the generic tests only use its adjoint tolerance."""
    return _street_case()[0]


# ------------------------------------------------------------------ the generic method


A = torch.tensor([[0.5, 0.2], [-0.1, 0.3]], dtype=F64)


def _linear(b):
    return lambda s: {"z": A @ s["z"] + b}


def test_linear_contraction_has_the_closed_form_fixed_point_and_gradient(model):
    b = torch.tensor([1.0, -2.0], dtype=F64, requires_grad=True)
    w = torch.tensor([0.7, 1.3], dtype=F64)
    diag: dict = {}
    out = model.fixed_point(_linear(b), {"z": torch.zeros(2, dtype=F64)}, watch=["z"],
                            tol=1e-15, max_passes=200, diagnostics=diag)
    inv = torch.linalg.inv(torch.eye(2, dtype=F64) - A)
    torch.testing.assert_close(out["z"].detach(), inv @ b.detach(), rtol=0, atol=1e-14)
    (g,) = torch.autograd.grad((out["z"] * w).sum(), b)
    torch.testing.assert_close(g, inv.T @ w, rtol=0, atol=1e-12)
    assert diag["adjoint"] == "implicit" and diag["passes"] == len(diag["changes"])
    assert 0.0 < diag["contraction"] < 1.0


def test_relaxation_gives_the_same_fixed_point_and_gradient(model):
    w = torch.tensor([0.7, 1.3], dtype=F64)
    results = []
    for relaxation in (1.0, 0.5):
        b = torch.tensor([1.0, -2.0], dtype=F64, requires_grad=True)
        out = model.fixed_point(_linear(b), {"z": torch.zeros(2, dtype=F64)}, watch=["z"],
                                tol=1e-15, max_passes=400, relaxation=relaxation)
        (g,) = torch.autograd.grad((out["z"] * w).sum(), b)
        results.append((out["z"].detach(), g))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=1e-13)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=0, atol=1e-12)


def test_relaxation_moves_only_the_watched_keys_part_of_the_way(model):
    fed: list[dict] = []

    def pass_fn(s):
        fed.append(s)
        return {"z": s["z"] * 0 + 4.0, "other": s.get("other", 0 * s["z"]) + 1.0}

    model.fixed_point(pass_fn, {"z": torch.zeros(1, dtype=F64)}, watch=["z"], tol=1e-12,
                      max_passes=60, relaxation=0.5)
    assert float(fed[1]["z"]) == 2.0 and float(fed[2]["z"]) == 3.0
    # The unwatched key is the pass's own output, not a relaxed one.
    assert float(fed[1]["other"]) == 1.0 and float(fed[2]["other"]) == 2.0


def test_non_convergence_raises_naming_where_the_passes_and_the_change(model):
    with pytest.raises(RuntimeError, match=r"my_loop.*did not converge within 3 passes.*"
                                           r"last pass was 8\.0"):
        model.fixed_point(lambda s: {"z": 2.0 * s["z"] + 1.0},
                          {"z": torch.ones(1, dtype=F64)}, watch=["z"], tol=1e-9,
                          max_passes=3, where="my_loop")


def test_a_missing_watched_key_forces_a_second_pass(model):
    diag: dict = {}
    out = model.fixed_point(lambda s: {"z": torch.full((1,), 3.0, dtype=F64)}, {},
                            watch=["z"], tol=0.0, max_passes=5, diagnostics=diag)
    assert diag["passes"] == 2 and diag["changes"] == [math.inf, 0.0]
    assert float(out["z"]) == 3.0


def test_a_pass_that_drops_a_watched_key_is_named(model):
    with pytest.raises(KeyError, match=r"the pass did not return the watched key"):
        model.fixed_point(lambda s: {}, {"z": torch.zeros(1, dtype=F64)}, watch=["z"],
                          tol=1e-9)


def test_each_instance_is_judged_on_its_own_measure(model):
    r = torch.tensor([0.0, 0.15], dtype=F64)
    target = torch.tensor([2.0, 3.0], dtype=F64)

    def pass_fn(s):
        return {"z": target * (1 - r) + r * s["z"]}

    def per_instance(fed, out):
        return (out["z"] - fed["z"]).abs()

    diag: dict = {}
    out = model.fixed_point(pass_fn, {"z": torch.zeros(2, dtype=F64)}, watch=["z"],
                            tol=2e-3, max_passes=20, measure=per_instance,
                            diagnostics=diag)
    # Instance 0 converges on pass 2 (pass 1 moves it all the way), instance 1 on pass 5.
    assert diag["passes"] == 5
    torch.testing.assert_close(out["z"], target, rtol=0, atol=1e-3)
    # The largest change over the instances: instance 1's, 3 * 0.85 * 0.15^(k-1) (instance 0
    # moves 2 on pass 1 and nothing after it).
    want = [3.0 * 0.85 * 0.15 ** k for k in range(5)]
    torch.testing.assert_close(torch.tensor(diag["changes"], dtype=F64),
                               torch.tensor(want, dtype=F64), rtol=1e-12, atol=0)


def test_a_batch_takes_the_per_instance_adjoint(model):
    r = torch.tensor([[0.2], [0.5]], dtype=F64)
    b = torch.tensor([[1.0], [2.0]], dtype=F64, requires_grad=True)
    diag: dict = {}
    out = model.fixed_point(
        lambda s: {"z": r * s["z"] + b}, {"z": torch.zeros(2, 1, dtype=F64)}, watch=["z"],
        tol=1e-14, max_passes=200, measure=lambda f, o: (o["z"] - f["z"]).abs().amax(-1),
        diagnostics=diag)
    (g,) = torch.autograd.grad(out["z"].sum(), b)
    torch.testing.assert_close(g, 1.0 / (1.0 - r), rtol=0, atol=1e-12)
    assert diag["adjoint_batched"] is True


def test_the_differentiable_pass_runs_only_when_something_requires_grad(model):
    b = torch.tensor([1.0, -2.0], dtype=F64, requires_grad=True)
    calls = []

    def counting(b):
        def pass_fn(s):
            calls.append(torch.is_grad_enabled())
            return {"z": A @ s["z"] + b}
        return pass_fn

    z0 = {"z": torch.zeros(2, dtype=F64)}
    diag: dict = {}
    with torch.no_grad():
        model.fixed_point(counting(b), z0, watch=["z"], tol=1e-12, diagnostics=diag)
    assert len(calls) == diag["passes"] and not any(calls)
    assert diag["adjoint"] is None
    calls.clear()
    # Grad mode with nothing differentiable: pass 1 asks, nothing more (no confirmation).
    model.fixed_point(counting(b.detach()), z0, watch=["z"], tol=1e-12, diagnostics=diag)
    assert len(calls) == diag["passes"] and calls[0] and not any(calls[1:])
    assert diag["adjoint"] is None
    calls.clear()
    # Differentiable: one extra pass, under grad, at the certified state.
    model.fixed_point(counting(b), z0, watch=["z"], tol=1e-12, diagnostics=diag)
    assert len(calls) == diag["passes"] + 1 and calls[0] and calls[-1]
    assert not any(calls[1:-1])
    assert diag["adjoint"] == "implicit"


def test_a_pass_that_echoes_its_watched_key_is_refused_by_name(model):
    b = torch.tensor([1.0], dtype=F64, requires_grad=True)
    with pytest.raises(RuntimeError, match=r"hands the watched key\(s\) \['z'\] back"):
        model.fixed_point(lambda s: {"z": s["z"], "y": b * 2.0},
                          {"z": torch.zeros(1, dtype=F64)}, watch=["z"], tol=1e-9)


def test_an_unwatched_key_that_feeds_back_is_an_unknown_of_the_adjoint(model):
    """a* = 2 theta, b* = 2 a* = 4 theta; only b is watched. `a` is recomputed and fed back,
    so holding it at its certified value would give db*/dtheta = 2, not 4."""
    theta = torch.tensor(1.0, dtype=F64, requires_grad=True)

    def pass_fn(z):
        a = 0.5 * z["a"] + theta
        return {"a": a, "b": 0.5 * z["b"] + a}

    z0 = {"a": torch.zeros((), dtype=F64), "b": torch.zeros((), dtype=F64)}
    diag: dict = {}
    out = model.fixed_point(pass_fn, z0, watch=["b"], tol=1e-14, max_passes=200,
                            adjoint_rtol=1e-13, diagnostics=diag)
    (g,) = torch.autograd.grad(out["b"], theta)
    assert g.item() == pytest.approx(4.0, rel=1e-12)
    assert diag["adjoint_keys"] == ["b", "a"]


def test_keys_the_pass_does_not_read_stay_out_of_the_adjoint(model):
    b = torch.tensor([1.0, -2.0], dtype=F64, requires_grad=True)

    def pass_fn(s):
        z = A @ s["z"] + b
        return {"z": z, "twice": 2.0 * z, "copied": s["copied"]}

    z0 = {"z": torch.zeros(2, dtype=F64), "twice": torch.zeros(2, dtype=F64),
          "copied": torch.ones(3, dtype=F64)}
    diag: dict = {}
    out = model.fixed_point(pass_fn, z0, watch=["z"], tol=1e-15, max_passes=200,
                            diagnostics=diag)
    assert diag["adjoint_keys"] == ["z"]
    inv = torch.linalg.inv(torch.eye(2, dtype=F64) - A)
    (g,) = torch.autograd.grad(out["twice"].sum(), b)
    torch.testing.assert_close(g, 2.0 * inv.T @ torch.ones(2, dtype=F64), rtol=0, atol=1e-12)
    assert torch.equal(out["copied"], z0["copied"]) and not out["copied"].requires_grad


def test_bad_arguments_are_refused(model):
    z0 = {"z": torch.zeros(1, dtype=F64)}
    with pytest.raises(ValueError, match="watch must name"):
        model.fixed_point(lambda s: s, z0, watch=[], tol=1e-9)
    with pytest.raises(ValueError, match="max_passes must be >= 1"):
        model.fixed_point(lambda s: s, z0, watch=["z"], tol=1e-9, max_passes=0)
    with pytest.raises(ValueError, match=r"relaxation must be in \(0, 1\]"):
        model.fixed_point(lambda s: s, z0, watch=["z"], tol=1e-9, relaxation=1.5)


# ----------------------------------------------------------------- street_steady


def _street_case():
    net = StreetNetwork(
        streets=[Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 Street("s2", "b", "c", 100.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0}, y={"a": 0.0, "b": 0.0, "c": 0.0},
    )
    reaction = photostationary_for_streets(("no", "no2", "o3"), preset="munich")
    model, state, _ = street_build_model(net, species=("no", "no2", "o3"),
                                         chemistry=reaction, pblh_floor=False)
    sources = torch.zeros(model.net.n, 3, dtype=F64)
    sources[model.net.node_index("s1"), 0] = 2.0e-3
    sources[model.net.node_index("s1"), 1] = 4.0e-4
    background = torch.zeros(1, 3, dtype=F64)
    background[0, 2] = 8.0e-8
    drivers = {
        "street.x_boundary": background, "street.sources": sources,
        "U_ref": torch.tensor(3.0, dtype=F64), "theta_w": torch.tensor(0.0, dtype=F64),
        "h_abl": torch.tensor(800.0, dtype=F64), "J_NO2": torch.tensor(5.0e-3, dtype=F64),
        "temperature": torch.tensor(298.0, dtype=F64),
    }
    return model, state, drivers, reaction


def _old_street_steady(model, state, drivers, *, reaction, tol, max_iter):
    """The loop `street_steady` ran before `Model.fixed_point`; returns `(state, passes)`."""
    key = "street.x"
    current = dict(state)
    change = float("inf")
    for passes in range(1, int(max_iter) + 1):
        solved = model.steady(current, drivers)
        relaxed = dict(solved)
        relaxed[key] = reaction.apply(solved[key], None, drivers)
        with torch.no_grad():
            if key in current:
                change = float((relaxed[key] - current[key]).abs().max())
        current = relaxed
        if change <= tol:
            return current, passes
    raise AssertionError("the old loop did not converge")


def test_street_steady_equals_the_old_loop_under_no_grad():
    model, state, drivers, reaction = _street_case()
    with torch.no_grad():
        want, passes = _old_street_steady(model, state, drivers, reaction=reaction,
                                          tol=1e-14, max_iter=50)
        calls = []
        real = model.steady
        model.steady = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            got = street_steady(model, state, drivers, reaction=reaction)
        finally:
            del model.steady
    assert len(calls) == passes and passes > 1
    assert got.keys() == want.keys()
    for k in want:
        assert torch.equal(got[k], want[k]), k


def test_street_steady_gradient_is_the_fixed_points_not_the_unrolled_passes():
    model, state, drivers, reaction = _street_case()
    grads = []
    for loop in ("old", "new"):
        src = drivers["street.sources"].clone().requires_grad_(True)
        d = {**drivers, "street.sources": src}
        if loop == "old":
            out, _ = _old_street_steady(model, state, d, reaction=reaction, tol=1e-14,
                                        max_iter=50)
        else:
            out = street_steady(model, state, d, reaction=reaction)
        (g,) = torch.autograd.grad(out["street.x"].sum(), src)
        grads.append((out["street.x"].detach(), g))
    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-11, atol=0)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-8, atol=0)


# ------------------------------------------------------------------- sewer_steady


def test_sewer_steady_equals_the_old_loop_under_no_grad():
    model, state, drivers = sewer_build_model(tree_steady())
    tol, dt, max_iter = 1e-12, 60.0, 500
    with torch.no_grad():
        current = dict(state)
        old_passes = 0
        for old_passes in range(1, max_iter + 1):  # noqa: B007
            new = model.step(current, drivers, dt)
            worst = 0.0
            for name in model.transport:
                key = f"{name}.x"
                worst = max(worst, float((new[key] - current[key]).abs().max()))
            current = new
            if worst < tol:
                break
        else:
            raise AssertionError("the old loop did not converge")
        calls = []
        real = model.step
        model.step = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            got = sewer_steady(model, state, drivers)
        finally:
            del model.step
    assert len(calls) == old_passes
    assert got.keys() == current.keys()
    for k in current:
        assert torch.equal(got[k], current[k]), k


@pytest.mark.parametrize("storage", [False, True])
def test_sewer_steady_gradient_matches_central_differences(storage):
    """With loads on the inflows (BOD and sulfide well above zero, away from the reaction's
    clamp at zero), the fixed point's gradient against the loads is the central difference
    of the converged states. With manhole storage the closure-carried `sewer.H` is
    recomputed by every step, so it is an unknown of the adjoint although it is not watched."""
    dt = 600.0
    kw = {"storage": True, "dt_storage": dt} if storage else {}
    model, state, drivers = sewer_build_model(tree_steady(), **kw)
    bod0 = torch.full_like(drivers["bod_in"], 0.2)
    sul0 = torch.full_like(drivers["sulfide_in"], 0.002)
    weights = torch.tensor([[1.0, 300.0], [2.0, 200.0], [1.5, 100.0], [0.5, 400.0],
                            [3.0, 250.0]], dtype=F64)
    spy: dict = {}
    real = model.fixed_point

    def recording(*args, **kwargs):
        return real(*args, **{**kwargs, "diagnostics": spy})

    def loss(bod, sul, record=False):
        model.fixed_point = recording if record else real
        try:
            out = sewer_steady(model, state, {**drivers, "bod_in": bod, "sulfide_in": sul},
                               dt=dt, tol=1e-11)
        finally:
            del model.fixed_point
        assert float(out["water_quality.x"].detach().min()) > 1e-4
        return (out["water_quality.x"] * weights).sum() + 1e3 * out["air_quality.x"].sum()

    bod = bod0.clone().requires_grad_(True)
    sul = sul0.clone().requires_grad_(True)
    g_bod, g_sul = torch.autograd.grad(loss(bod, sul, record=True), [bod, sul])
    assert ("sewer.H" in spy["adjoint_keys"]) == storage
    v_bod = torch.linspace(0.5, 1.5, bod0.numel(), dtype=F64)
    v_sul = torch.linspace(1.0, -1.0, sul0.numel(), dtype=F64)
    h = 1e-4
    with torch.no_grad():
        up = loss(bod0 + h * 0.2 * v_bod, sul0 + h * 0.002 * v_sul)
        dn = loss(bod0 - h * 0.2 * v_bod, sul0 - h * 0.002 * v_sul)
    fd = float(up - dn) / (2 * h)
    ad = float(0.2 * (g_bod * v_bod).sum() + 0.002 * (g_sul * v_sul).sum())
    assert ad == pytest.approx(fd, rel=1e-6)


# ------------------------------------------------------------ street_steady_with_plume


def _plume_case():
    net, _ = twelve_street_grid(L=100.0, W=20.0, H=20.0)
    model, state, _ = street_build_model(net, background="per_street")
    n = len(street_index(model))
    sources = torch.zeros(model.net.n, dtype=F64)
    for i, name in enumerate(street_index(model)):
        sources[model.net.node_index(name)] = 1e-6 * (i + 1)
    drivers = {
        "street.x_boundary": torch.full((n,), 2e-8, dtype=F64), "street.sources": sources,
        "u_star": torch.tensor(0.5, dtype=F64), "theta_w": torch.tensor(0.3, dtype=F64),
        "h_abl": torch.tensor(800.0, dtype=F64),
    }
    table = plume_table(u_star=0.5, h_abl=800.0, lmo=math.inf, h_canopy=20.0, z0=1.0,
                        d=13.0, sigma_theta=math.radians(5.0))
    return (model, state, drivers, street_kernel(net, table, theta_w=0.3),
            junction_kernel(net, table, theta_w=0.3))


def _old_plume_passes(model, state, drivers, k_s, k_j, *, tol, relaxation):
    """The forward iteration `street_steady_with_plume` ran before `Model.fixed_point`:
    `(c_ext at return, relative change per pass)`."""
    wiring = roof_wiring(model)
    c_bg = drivers["street.x_boundary"]
    q = model.current_flows("street", state, drivers)

    def image(c_ext):
        x = model.steady(state, {**drivers, "street.x_boundary": c_ext})["street.x"]
        f_s, up, _ = roof_fluxes(wiring, q, x.unsqueeze(-1), c_ext.unsqueeze(-1),
                                 c_bg.unsqueeze(-1))
        return c_bg + (k_s @ f_s).squeeze(-1) + (k_j @ up).squeeze(-1)

    changes = []
    with torch.no_grad():
        c_ext = c_bg
        for _ in range(100):
            c_new = image(c_ext)
            step = c_new - c_ext
            changes.append(float(step.abs().amax() / c_new.abs().amax()))
            c_ext = c_ext + relaxation * step
            if changes[-1] <= tol:
                break
    return c_ext, changes


@pytest.mark.parametrize("relaxation", [1.0, 0.7])
def test_street_steady_with_plume_runs_the_old_passes(relaxation):
    model, state, drivers, k_s, k_j = _plume_case()
    c_old, changes = _old_plume_passes(model, state, drivers, k_s, k_j, tol=1e-12,
                                       relaxation=relaxation)
    diag: dict = {}
    with torch.no_grad():
        out = street_steady_with_plume(model, state, drivers, kernel=k_s,
                                       junction_kernel=k_j, relaxation=relaxation,
                                       diagnostics=diag)
    assert diag["passes"] == len(changes)
    torch.testing.assert_close(torch.tensor(diag["changes"], dtype=F64),
                               torch.tensor(changes, dtype=F64), rtol=1e-9, atol=0)
    # The old loop returned one more image, at the updated iterate; the new one returns the
    # last pass's own: the two differ by the last change, which is within the tolerance.
    torch.testing.assert_close(out["street.c_ext"], c_old, rtol=1e-11, atol=0)
