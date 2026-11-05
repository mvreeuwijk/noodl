"""The puff closure inside the street model (`noodl.apps.street_aq.puff.PuffClosure`)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from noodl.apps.street_aq.network import (
    build_model,
    initial_state,
    street_index,
    twelve_street_grid,
)
from noodl.apps.street_aq.puff import PuffClosure, PuffOptions, PuffState, puff_mass
from noodl.contracts import StepContext
from noodl.model import Model

DT = torch.float64
OPTIONS = PuffOptions(sigma_theta=math.radians(6.0))


def _meteorology(*, theta=0.0, u_star=0.5):
    return {"u_star": torch.tensor(u_star, dtype=DT), "theta_w": torch.tensor(theta, dtype=DT),
            "h_abl": torch.tensor(800.0, dtype=DT)}


def _drivers(model, drivers, *, bg=2.0**-26, theta=0.0, sources=None):
    n = len(street_index(model))
    out = {**drivers, **_meteorology(theta=theta),
           "street.x_boundary": torch.full((n,), bg, dtype=DT)}
    if sources is not None:
        out["street.sources"] = sources
    return out


def _street_sources(model, values: dict[str, float]):
    s = torch.zeros(model.net.n, dtype=DT)
    for name, value in values.items():
        s[model.net.node_index(name)] = value
    return s


def test_without_puffs_build_model_is_unchanged():
    net, names = twelve_street_grid()
    plain, s0, d0 = build_model(net, background="per_street")
    none, s1, d1 = build_model(net, background="per_street", puffs=None)
    assert [type(c) for c in none.closures] == [type(c) for c in plain.closures]
    assert set(s1) == set(s0) == {"street.x"}
    assert set(initial_state(none)) == {"street.x"}
    assert set(d1) == set(d0)
    src = _street_sources(plain, {names[3]: 1e-6, names[10]: 3e-6})
    a, b = dict(s0), dict(s1)
    for _ in range(5):
        a = plain.step(a, _drivers(plain, d0, sources=src), 10.0)
        b = none.step(b, _drivers(none, d1, sources=src), 10.0)
    assert torch.equal(a["street.x"], b["street.x"])


def _puff_model(**kwargs):
    net, names = twelve_street_grid(**kwargs.pop("grid", {}))
    model, state, drivers = build_model(net, background="per_street",
                                        puffs=kwargs.pop("puffs", OPTIONS), **kwargs)
    return net, names, model, state, drivers


def test_build_model_appends_the_closure_and_seeds_the_state():
    _, _, model, state, _ = _puff_model()
    assert isinstance(model.closures[-1], PuffClosure)
    assert len(model.closures) == 2
    seeded = initial_state(model)
    assert set(seeded) == set(state)
    puff = PuffState.from_state(seeded)
    assert puff.mass.shape == (1, OPTIONS.capacity) and not bool(puff.live.any())
    assert torch.equal(seeded["puff.c_ext"], torch.zeros(12, dtype=DT))
    assert seeded["puff.q"].shape == model.current_flows(
        "street", seeded, _drivers(model, model.driver_template)).shape
    assert seeded["puff.time"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("junction_source", ["upward", "net"])
def test_no_emission_keeps_c_ext_at_background(junction_source):
    bg = 2.0**-26
    _, _, model, state, drivers = _puff_model(
        puffs=PuffOptions(sigma_theta=math.radians(6.0), junction_source=junction_source))
    state = {**state, "street.x": torch.full((12,), bg, dtype=DT)}
    drv = _drivers(model, drivers, bg=bg, theta=0.3)
    for k in range(50):
        state = model.step(state, drv, 5.0, t=5.0 * k)
        assert torch.equal(state["puff.c_ext"], torch.full((12,), bg, dtype=DT)), k
    assert int(state["puff.live"].sum()) == 0


def test_cloud_reaches_downwind_street_after_travel_time():
    # Wind toward east along the grid's middle row: street 3 (E3-A), 4 (A-B), 5 (B-E5). The
    # nearest puff source to street 5's midpoint is junction B, half a street upwind.
    length = 200.0
    _, names, model, state, drivers = _puff_model(grid={"L": length})
    closure = model.closures[-1]
    dt, bg = 5.0, 2.0**-26
    pulse = _street_sources(model, {names[2]: 1e-3})
    zero = torch.zeros_like(pulse)
    target = names.index("5")
    state = {**state, "street.x": torch.full((12,), bg, dtype=DT)}
    seen = []
    for k in range(40):
        drv = _drivers(model, drivers, bg=bg, sources=pulse if k == 0 else zero)
        state = model.step(state, drv, dt, t=dt * k)
        seen.append(float(state["puff.c_ext"][target]) - bg)
    u_roof = float(closure.tables.u[0, 0, 1])
    travel = (length / 2.0) / u_roof
    # nothing before half the travel time (the cloud's front runs cutoff_sigma spreads
    # ahead of its slug), material from the travel time on, rising to a later peak
    assert all(v == 0.0 for k, v in enumerate(seen) if dt * (k + 1) < 0.5 * travel)
    assert all(v > 0.0 for k, v in enumerate(seen) if dt * k >= travel)
    peak = seen.index(max(seen))
    assert dt * peak > travel


def test_query_returns_c_ext_without_advancing():
    _, names, model, state, drivers = _puff_model()
    src = _street_sources(model, {names[2]: 1e-3})
    drv = _drivers(model, drivers, sources=src)
    for k in range(4):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    closure = model.closures[-1]
    before = PuffState.from_state(state)
    out = closure(state, drv, None)
    assert torch.allclose(out["street.x_boundary"], state["puff.c_ext"], rtol=1e-14, atol=0)
    assert set(out) == {"street.x_boundary"}
    after = PuffState.from_state(state)
    assert torch.equal(before.age, after.age)
    # the model's queries see the same boundary
    res = model.residuals(state, drv)
    assert torch.isfinite(res["street"]).all()


def test_closure_keeps_caller_background_and_reports_diagnostics():
    _, names, model, state, drivers = _puff_model()
    src = _street_sources(model, {names[2]: 1e-3})
    drv = _drivers(model, drivers, sources=src)
    closure = model.closures[-1]
    state = model.step(state, drv, 5.0, t=0.0)
    assert closure.last["rebuilt"] is True
    state = model.step(state, drv, 5.0, t=5.0)
    assert closure.last["rebuilt"] is False
    assert int(closure.last["live"].sum()) > 0
    assert closure.last["evicted"].shape == (1,)
    # the caller's background is not replaced by the closure's own C_ext
    assert torch.equal(drv["street.x_boundary"], torch.full((12,), 2.0**-26, dtype=DT))
    # a new tensor of the same value rebuilds (identity rule); a changed value too
    drv2 = {**drv, "u_star": drv["u_star"].clone()}
    model.step(state, drv2, 5.0, t=10.0)
    assert closure.last["rebuilt"] is True


def test_batched_drivers_expand_the_unbatched_seed():
    _, names, model, state, drivers = _puff_model()
    src = _street_sources(model, {names[2]: 1e-3})
    drv = _drivers(model, drivers, sources=src)
    # (StreetFlows needs every meteorology driver batched alike)
    drv.update({"u_star": torch.tensor([0.4, 0.5, 0.6], dtype=DT),
                "theta_w": torch.zeros(3, dtype=DT), "h_abl": torch.full((3,), 800.0, dtype=DT)})
    for k in range(3):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    assert state["puff.mass"].shape == (3, OPTIONS.capacity)
    assert state["puff.c_ext"].shape == (3, 12)
    closure = model.closures[-1]
    drv = {**drv, "street.q": state["puff.q"], "u_star": torch.full((2, 3), 0.5, dtype=DT)}
    with pytest.raises(ValueError, match=r"\(\) or \(B,\)"):
        closure(state, drv, StepContext(dt=5.0, t=15.0))


def test_released_puff_reaches_the_streets():
    _, _, model, state, drivers = _puff_model(
        puffs=PuffOptions(sigma_theta=math.radians(6.0), n_releases=1))
    drv = _drivers(model, drivers, bg=0.0)
    drv.update({"puff.release_xy": torch.tensor([[[-150.0, 100.0]]], dtype=DT),
                "puff.release_z": torch.tensor([[30.0]], dtype=DT),
                "puff.release_rate": torch.tensor([[1e-3]], dtype=DT),
                "puff.release_start": torch.tensor([[0.0]], dtype=DT),
                "puff.release_duration": torch.tensor([[10.0]], dtype=DT)})
    # the release over [0, 10] s is emitted at the starts of the next two steps
    for k in range(3):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    puff = PuffState.from_state(state)
    released = puff.live & (puff.table == 1)
    assert int(released.sum()) == 2
    assert float(puff.mass[released].sum()) == pytest.approx(1e-2, rel=1e-12)
    for k in range(3, 30):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    assert float(state["street.x"].max()) > 0.0
    assert model.closures[-1].tables.u.shape[0] == 2
    with pytest.raises(ValueError, match=r"Model\.step\(t=\.\.\.\)"):
        model.step(state, drv, 5.0)


def test_refusals_are_named():
    net, _ = twelve_street_grid()
    with pytest.raises(ValueError, match="background='per_street'"):
        build_model(net, puffs=OPTIONS)
    with pytest.raises(ValueError, match="scheme='implicit'"):
        build_model(net, background="per_street", scheme="exact", puffs=OPTIONS)
    with pytest.raises(ValueError, match="one species"):
        build_model(net, background="per_street", species=("nox", "o3"), puffs=OPTIONS)
    with pytest.raises(ValueError, match="junction_source"):
        build_model(net, background="per_street",
                    puffs=PuffOptions(sigma_theta=0.1, junction_source="both"))
    with pytest.raises(NotImplementedError, match="shear"):
        build_model(net, background="per_street", puffs=PuffOptions(shear=True))

    model, state, drivers = build_model(net, background="per_street", puffs=OPTIONS)
    drv = _drivers(model, drivers)
    closure = model.closures[-1]
    iterated = Model(model.net, model.layers, closures=model.closures, coupling="iterate",
                     iterate_tol={"street": 1e-12})
    closure.bind(iterated)
    with pytest.raises(ValueError, match="coupling='pingpong'"):
        iterated.step(state, drv, 5.0, t=0.0)
    sub = Model(model.net, model.layers, closures=model.closures, substeps={"street": 2})
    closure.bind(sub)
    with pytest.raises(ValueError, match="substeps"):
        sub.step(state, drv, 5.0, t=0.0)
    closure.bind(model)
    removal = {**drv, "street.removal": torch.full((12,), 1e-3, dtype=DT)}
    with pytest.raises(ValueError, match="removal"):
        model.step(state, removal, 5.0, t=0.0)
    no_u_star = {k: v for k, v in drv.items() if k != "u_star"}
    no_u_star["U_ref"] = torch.tensor(3.0, dtype=DT)
    with pytest.raises(KeyError, match="u_star"):
        model.step(state, no_u_star, 5.0, t=0.0)
    model.step(state, drv, 5.0, t=0.0)


def _grad_run(model, state, drv, u_star):
    for k in range(4):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    return state["street.x"].sum()


def test_a_new_run_rebuilds_the_tables_for_its_own_graph():
    # the same driver tensors in every run, so only the run boundary can tell the runs apart
    _, names, model, state, drivers = _puff_model()
    closure = model.closures[-1]
    u_star = torch.tensor(0.5, dtype=DT, requires_grad=True)
    src = _street_sources(model, {names[2]: 1e-3})
    drv = {**_drivers(model, drivers, sources=src), "u_star": u_star}
    (reference,) = torch.autograd.grad(_grad_run(model, state, drv, u_star), u_star)
    # a warm-up without grad must not leave graph-less tables behind
    with torch.no_grad():
        _grad_run(model, state, drv, u_star)
    (after,) = torch.autograd.grad(_grad_run(model, state, drv, u_star), u_star)
    assert torch.equal(after, reference)
    # two differentiated runs in a row on the same leaf: no freed graph is reused
    (again,) = torch.autograd.grad(_grad_run(model, state, drv, u_star), u_star)
    assert torch.equal(again, reference)
    assert closure.last["rebuilt"] is False


def test_a_grad_run_after_a_no_grad_warm_up_has_the_full_gradient():
    grads = []
    for warm_up in (False, True):
        _, names, model, state, drivers = _puff_model()
        u_star = torch.tensor(0.5, dtype=DT, requires_grad=True)
        src = _street_sources(model, {names[2]: 1e-3})
        drv = {**_drivers(model, drivers, sources=src), "u_star": u_star}
        if warm_up:
            with torch.no_grad():
                _grad_run(model, state, drv, u_star)
        grads.append(torch.autograd.grad(_grad_run(model, state, drv, u_star), u_star)[0])
    assert torch.equal(grads[1], grads[0])


# ------------------------------------------------------------------ the mass budget (Test 3)


def _budget_run(junction_source, thetas, capacity, *, n_steps=60, dt=5.0):
    """A pulse in street 3 (canopy) and an above-roof pulse, stepped `n_steps` times with
    the wind direction `thetas[k]` at step `k`; yields `(k, state, terms)` after every
    step, `terms` the budget's sides (`_budget_sides`)."""
    from noodl.apps.street_aq.puff import mass_in_streets, pending_roof_mass, pulse_mass

    bg = 2.0e-8
    options = PuffOptions(sigma_theta=math.radians(6.0), junction_source=junction_source,
                          capacity=capacity, n_releases=1, x_max=300.0)
    _, names, model, state, drivers = _puff_model(puffs=options)
    state = {**state, "street.x": torch.full((12,), bg, dtype=DT)}
    release = {"puff.release_xy": torch.tensor([[[-60.0, 90.0]]], dtype=DT),
               "puff.release_z": torch.tensor([[30.0]], dtype=DT),
               "puff.release_rate": torch.tensor([[3e-3]], dtype=DT),
               "puff.release_start": torch.tensor([[7.0]], dtype=DT),
               "puff.release_duration": torch.tensor([[23.0]], dtype=DT)}
    pulse = _street_sources(model, {names[2]: 1.7e-3, names[8]: 0.3e-3})
    steady = _street_sources(model, {names[5]: 2e-6})
    inputs = 0.0
    for k in range(n_steps):
        t = dt * k
        sources = steady + (pulse if t < 20.0 else torch.zeros_like(pulse))
        drv = {**_drivers(model, drivers, bg=bg, theta=thetas[k], sources=sources), **release}
        state = model.step(state, drv, dt, t=t)
        # what entered over this step: the canopy sources (emission_scale 1) and the release
        inputs = inputs + sources.sum() * dt + pulse_mass(
            t, dt, release["puff.release_start"], release["puff.release_duration"],
            release["puff.release_rate"]).sum()
        puff = PuffState.from_state(state)
        c_bg = drv["street.x_boundary"]
        emit_next, drawn_next = pending_roof_mass(model, state, c_bg,
                                                  junction_source=junction_source)
        # the release's mass of this step is emitted by the next step too
        emit_next = emit_next + pulse_mass(
            t, dt, release["puff.release_start"], release["puff.release_duration"],
            release["puff.release_rate"]).sum(-1)
        held = (mass_in_streets(model, state, c_bg) + puff_mass(puff) + puff.mass_out
                + puff.mass_dropped - puff.mass_drawn_down + emit_next - drawn_next)
        yield k, model, state, held, inputs


def _assert_budget(junction_source, thetas, capacity):
    """Assert the identity after every step; return the final state and what the run
    exercised (the largest `mass_out` and `mass_dropped`, the puffs evicted)."""
    seen = {"out": 0.0, "dropped": 0.0, "evicted": 0}
    for k, model, state, held, inputs in _budget_run(junction_source, thetas, capacity):
        assert held.shape == (1,)
        assert abs(float(held[0]) - float(inputs)) <= 1e-12 * float(inputs), (
            k, float(held[0]), float(inputs))
        puff = PuffState.from_state(state)
        seen["out"] = max(seen["out"], float(puff.mass_out[0]))
        seen["dropped"] = max(seen["dropped"], float(puff.mass_dropped[0]))
        seen["evicted"] += int(model.closures[-1].last["evicted"].sum())
    return state, seen


@pytest.mark.parametrize("capacity", [30, 4096])
@pytest.mark.parametrize("junction_source", ["upward", "net"])
def test_budget_exact_every_step(junction_source, capacity):
    """Design Sect. 4.8, at every step, with the step's flux not yet emitted counted:

        sum_s V_s (x_s - C_bg,s) + sum_live m + mass_out + mass_dropped - mass_drawn_down
            + E_pending - D_pending = sum_steps (sum_s Q_s dt + release mass)

    `E_pending` is what the next step will emit (`pending_roof_mass`: the last step's roof
    and junction fluxes times its length, plus the release's mass of the last step) and
    `D_pending` what it will add to `mass_drawn_down` (zero under `"net"`), to 1e-12
    relative to the input. With 30 slots (25 sources) puffs are evicted for capacity;
    with the default capacity they leave the 300 m domain and fall below the faint
    threshold."""
    state, seen = _assert_budget(junction_source, [0.3] * 60, capacity)
    if capacity == 30:
        assert seen["evicted"] > 0 and seen["dropped"] != 0.0
    else:
        assert seen["evicted"] == 0 and seen["out"] != 0.0 and seen["dropped"] != 0.0
    if junction_source == "net":
        assert float(state["puff.mass_drawn_down"][0]) == 0.0
    else:
        assert float(state["puff.mass_drawn_down"][0]) > 0.0


def test_budget_exact_when_wind_turns():
    """The identity of `test_budget_exact_every_step` while the wind direction turns
    between steps (the closure rebuilds its source field; live puffs turn with the wind)."""
    thetas = [0.3 + 0.05 * k if k < 30 else 1.8 - 0.04 * (k - 30) for k in range(60)]
    _, seen = _assert_budget("upward", thetas, 4096)
    assert seen["out"] != 0.0


def test_mass_in_streets_is_the_volume_weighted_excess():
    from noodl.apps.street_aq.puff import mass_in_streets

    _, _, model, state, _ = _puff_model()
    x = torch.linspace(1.0, 2.0, 12, dtype=DT)
    bg = torch.full((12,), 0.5, dtype=DT)
    got = mass_in_streets(model, {**state, "street.x": x}, bg)
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(float((40000.0 * (x - 0.5)).sum()), rel=1e-15)
    batched = mass_in_streets(model, {**state, "street.x": torch.stack([x, 2 * x])}, bg)
    assert batched.shape == (2,)
    assert torch.allclose(batched[1], (40000.0 * (2 * x - 0.5)).sum(), rtol=1e-15, atol=0)


# --- Test 2: the steady limit (design Sect. 2, 4.3) ------------------------------------------

_STEADY_THETA = 0.3
_STEADY_MET = {"u_star": 0.5, "h_abl": 800.0, "lmo": math.inf, "h_canopy": 20.0, "z0": 1.0,
               "d": 13.0, "sigma_theta": math.radians(6.0)}


def _steady_setup(along_wind):
    from noodl.apps.street_aq.plume import (
        junction_kernel,
        plume_table,
        street_kernel,
        street_midpoints,
    )
    from noodl.apps.street_aq.puff import PuffField, puff_tables

    net, _ = twelve_street_grid()
    table = plume_table(**_STEADY_MET)
    kernel = torch.cat([street_kernel(net, table, theta_w=_STEADY_THETA),
                        junction_kernel(net, table, theta_w=_STEADY_THETA)], dim=1)
    tables = puff_tables(centres=torch.tensor([10.0], dtype=DT), **_STEADY_MET)
    field = PuffField.for_network(net, theta_w=_STEADY_THETA, along_wind=along_wind)
    return net, kernel, tables, field, street_midpoints(net)


def _steady_train(field, tables, dt):
    """The puffs of a unit continuous source in every street and junction (one batch
    instance per source) once steady: ages dt, 2 dt, ... while x(age) <= x_max, mass dt,
    displaced x(age) along the wind -- what emit / advance / housekeep leave behind
    (`test_steady_train_is_what_the_step_loop_leaves`)."""
    zero = torch.zeros((), dtype=torch.long)
    n_src = field.n_sources
    ages = torch.arange(1, int(float(tables.t[0, 0, -1]) / dt) + 2, dtype=DT) * dt
    x = tables.at_age(ages, zero, zero)[0]
    ages, x = ages[x <= tables.x_max], x[x <= tables.x_max]
    k = ages.numel()
    e = torch.tensor([math.cos(_STEADY_THETA), math.sin(_STEADY_THETA)], dtype=DT)
    empty = PuffState.empty((n_src,), k)
    return replace(empty, mass=torch.full((n_src, k), dt, dtype=DT),
                   age=ages.expand(n_src, k).clone(),
                   disp=(x[:, None] * e).expand(n_src, k, 2).clone(),
                   source=torch.arange(n_src)[:, None].expand(n_src, k).clone(),
                   live=torch.ones(n_src, k, dtype=torch.bool))


def _kernel_error(field, tables, receptors, kernel, state, dt):
    """max |K_puff - K| / max K, `K_puff[i, j]` the puffs of source `j` at street `i`'s
    midpoint, its own street's puffs excluded (as `street_kernel`)."""
    from noodl.apps.street_aq.puff import roof_concentration

    n = receptors.shape[0]
    k_puff = roof_concentration(field, state, tables, receptors, _STEADY_THETA, dt,
                                receptor_owner=torch.arange(n)).T
    return float((k_puff - kernel).abs().max() / kernel.abs().max())


def test_steady_train_is_what_the_step_loop_leaves():
    from noodl.apps.street_aq.puff import advance, emit, housekeep

    _, kernel, tables, field, receptors = _steady_setup("none")
    dt, n_src = 10.0, field.n_sources
    state = PuffState.empty((n_src,), 200)
    for _ in range(int(float(tables.t[0, 0, -1]) / dt) + 3):
        state, _ = emit(field, state, torch.eye(n_src, dtype=DT) * dt,
                        torch.zeros(n_src, dtype=torch.long), tables=tables)
        state = advance(state, tables, dt, _STEADY_THETA)
        state = housekeep(field, state, tables, drop_below=0.0)
    direct = _steady_train(field, tables, dt)
    order = torch.argsort(state.age.masked_fill(~state.live, math.inf), dim=1)
    live = int(state.live[0].sum())
    assert live == direct.age.shape[1]
    torch.testing.assert_close(state.age.gather(1, order)[:, :live], direct.age,
                               rtol=0, atol=0)
    disp = state.disp.gather(1, order[..., None].expand(-1, -1, 2))[:, :live]
    torch.testing.assert_close(disp, direct.disp, rtol=1e-13, atol=1e-11)
    looped = _kernel_error(field, tables, receptors, kernel, state, dt)
    assert looped == pytest.approx(_kernel_error(field, tables, receptors, kernel, direct,
                                                 dt), abs=1e-14)


# Measured (measurements.md, Round 3): e(5 s) = 5.6e-16, the steady kernel to rounding;
# rounded up to one significant figure.
_TOL_5S = 6e-16
# Measured max e(dt) / dt over the steps below: 6.0e-16 /s (e at rounding level); rounded up.
_C_PER_S = 7e-16


def test_steady_limit_without_along_wind_spread_is_the_kernel():
    """Test 2a, kernel level: the puff train of a unit continuous source per street and
    junction, at the street midpoints, against `street_kernel` / `junction_kernel` built
    from the same `plume_table` inputs. With the slug a segment of the steady plume
    (design Sect. 4.3) the error is rounding from dt = 10 s down (no pair falls in the
    first slug's emission-lag gap here)."""
    _, kernel, tables, field, receptors = _steady_setup("none")
    errors = {dt: _kernel_error(field, tables, receptors, kernel,
                                _steady_train(field, tables, dt), dt)
              for dt in (10.0, 5.0, 2.5, 1.25, 0.625)}
    print("steady limit, along_wind='none', e(dt):", errors)
    assert errors[5.0] <= _TOL_5S
    assert all(e <= 1e-8 for dt, e in errors.items() if dt <= 1.25)
    assert all(e <= _C_PER_S * dt for dt, e in errors.items())


# Measured (measurements.md, Round 3): e = 7.9e-12 at 10 s and 1.6e-11 at 5 s, the level
# of the stopping rule (C_ext changes < 1e-12 relative per step); one significant
# figure, rounded up.
_COUPLED_TOL = {10.0: 8e-12, 5.0: 2e-11}


@pytest.mark.slow
@pytest.mark.parametrize("dt", [10.0, 5.0])
def test_coupled_steady_limit_without_along_wind_spread_is_the_steady_model(dt):
    """Test 2a, coupled: constant sources in every street, the puff model stepped until
    C_ext changes < 1e-12 relative, against `street_steady_with_plume` with the kernels of
    the same `plume_table` inputs (junction_source "upward"). 250 / 470 steps: 40 / 80 s."""
    from noodl.apps.street_aq.above_roof import street_steady_with_plume
    from noodl.apps.street_aq.plume import junction_kernel, plume_table, street_kernel

    bg = 2e-8
    net, _ = twelve_street_grid()
    plain, s0, d0 = build_model(net, background="per_street")
    src = torch.zeros(plain.net.n, dtype=DT)
    for i, name in enumerate(street_index(plain)):
        src[plain.net.node_index(name)] = 1e-6 * (i + 1)
    met = {"u_star": torch.tensor(_STEADY_MET["u_star"], dtype=DT),
           "theta_w": torch.tensor(_STEADY_THETA, dtype=DT),
           "h_abl": torch.tensor(_STEADY_MET["h_abl"], dtype=DT)}
    drv = {**d0, **met, "street.x_boundary": torch.full((12,), bg, dtype=DT),
           "street.sources": src}
    table = plume_table(**_STEADY_MET)
    ref = street_steady_with_plume(
        plain, s0, drv, kernel=street_kernel(net, table, theta_w=_STEADY_THETA),
        junction_kernel=junction_kernel(net, table, theta_w=_STEADY_THETA),
        junction_source="upward")["street.c_ext"]
    model, state, d1 = build_model(net, background="per_street", puffs=PuffOptions(
        sigma_theta=_STEADY_MET["sigma_theta"], along_wind="none", junction_source="upward",
        drop_below=0.0, capacity=2000))
    drv1 = {**d1, **met, "street.x_boundary": drv["street.x_boundary"],
            "street.sources": src}
    prev = None
    with torch.no_grad():
        for k in range(2000):
            state = model.step(state, drv1, dt, t=dt * k)
            c = state["puff.c_ext"]
            if prev is not None and float((c - prev).abs().max()
                                          / (c - bg).abs().max()) < 1e-12:
                break
            prev = c
    e = float((c - ref).abs().max() / (ref - bg).abs().max())
    print(f"coupled steady limit, dt = {dt} s: e = {e:.3e} after {k} steps")
    assert e <= _COUPLED_TOL[dt]
