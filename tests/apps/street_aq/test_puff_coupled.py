"""The puff closure inside the street model (`noodl.apps.street_aq.puff.PuffClosure`)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

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
    _, names, model, state, drivers = _puff_model(puffs=replace(OPTIONS, puff_dt=30.0))
    src = _street_sources(model, {names[2]: 1e-3})
    drv = _drivers(model, drivers, sources=src)
    closure = model.closures[-1]
    state = model.step(state, drv, 5.0, t=0.0)
    assert closure.last["rebuilt"] is True
    state = model.step(state, drv, 5.0, t=5.0)
    assert closure.last["rebuilt"] is False
    assert int(closure.last["live"].sum()) == 0  # accumulating towards puff_dt = 30 s
    for k in range(2, 7):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    assert closure.last["rebuilt"] is False
    assert int(closure.last["live"].sum()) > 0
    assert closure.last["evicted"].shape == (1,)
    # the caller's background is not replaced by the closure's own C_ext
    assert torch.equal(drv["street.x_boundary"], torch.full((12,), 2.0**-26, dtype=DT))
    # a new tensor of the same value rebuilds (identity rule); a changed value too
    drv2 = {**drv, "u_star": drv["u_star"].clone()}
    model.step(state, drv2, 5.0, t=35.0)
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
        puffs=PuffOptions(sigma_theta=math.radians(6.0), n_releases=1, puff_dt=30.0))
    drv = _drivers(model, drivers, bg=0.0)
    drv.update({"puff.release_xy": torch.tensor([[[-150.0, 100.0]]], dtype=DT),
                "puff.release_z": torch.tensor([[30.0]], dtype=DT),
                "puff.release_rate": torch.tensor([[1e-3]], dtype=DT),
                "puff.release_start": torch.tensor([[0.0]], dtype=DT),
                "puff.release_duration": torch.tensor([[10.0]], dtype=DT)})
    # the release over [0, 10] s is accumulated over the first puff interval (30 s, six
    # steps) and emitted as one puff at the start of the seventh step
    for k in range(6):
        state = model.step(state, drv, 5.0, t=5.0 * k)
    assert not bool(PuffState.from_state(state).live.any())
    assert float(state["puff.pending"][0, -1]) == pytest.approx(1e-2, rel=1e-12)
    state = model.step(state, drv, 5.0, t=30.0)
    puff = PuffState.from_state(state)
    released = puff.live & (puff.table == 1)
    assert int(released.sum()) == 1
    assert float(puff.mass[released].sum()) == pytest.approx(1e-2, rel=1e-12)
    # emitted at the age puff_dt/2 - dt/2 and advanced one step: the interval's middle
    assert float(puff.age[released][0]) == 17.5
    assert float(state["puff.pending"].abs().max()) == 0.0
    for k in range(7, 30):
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
    with pytest.raises(ValueError, match="shear=True needs along_wind='sigma_y'"):
        build_model(net, background="per_street",
                    puffs=PuffOptions(sigma_theta=0.1, shear=True, along_wind="none"))
    for bad in (0.0, -30.0, math.inf):
        with pytest.raises(ValueError, match="puff_dt"):
            build_model(net, background="per_street",
                        puffs=PuffOptions(sigma_theta=0.1, puff_dt=bad))

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
    # a street step must divide puff_dt (30 s), and the steps of an interval fill it
    model, state, drivers = build_model(net, background="per_street",
                                        puffs=replace(OPTIONS, puff_dt=30.0))
    drv = _drivers(model, drivers)
    with pytest.raises(ValueError, match=r"does not divide PuffOptions\.puff_dt"):
        model.step(state, drv, 7.0, t=0.0)
    s = state
    for t, dt in ((0.0, 10.0), (10.0, 10.0), (20.0, 5.0)):
        s = model.step(s, drv, dt, t=t)
    with pytest.raises(ValueError, match="past PuffOptions.puff_dt"):
        model.step(model.step(s, drv, 10.0, t=25.0), drv, 10.0, t=35.0)


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


def _budget_run(junction_source, thetas, capacity, *, n_steps=60, dt=5.0, puff_dt=30.0,
                shear=False):
    """A pulse in street 3 (canopy) and an above-roof pulse, stepped `n_steps` times with
    the wind direction `thetas[k]` at step `k`; yields `(k, state, terms)` after every
    step, `terms` the budget's sides (`_budget_sides`)."""
    from noodl.apps.street_aq.puff import mass_in_streets, pending_roof_mass, pulse_mass

    bg = 2.0e-8
    options = PuffOptions(sigma_theta=math.radians(6.0), junction_source=junction_source,
                          capacity=capacity, n_releases=1, x_max=300.0, puff_dt=puff_dt,
                          shear=shear)
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


def _assert_budget(junction_source, thetas, capacity, puff_dt=30.0, shear=False):
    """Assert the identity after every step; return the final state and what the run
    exercised (the largest `mass_out`, `mass_dropped` and accumulated `"puff.pending"`,
    the puffs evicted)."""
    seen = {"out": 0.0, "dropped": 0.0, "evicted": 0, "pending": 0.0, "sxz": 0.0}
    for k, model, state, held, inputs in _budget_run(junction_source, thetas, capacity,
                                                     puff_dt=puff_dt, shear=shear):
        assert held.shape == (1,)
        assert abs(float(held[0]) - float(inputs)) <= 1e-12 * float(inputs), (
            k, float(held[0]), float(inputs))
        puff = PuffState.from_state(state)
        seen["out"] = max(seen["out"], float(puff.mass_out[0]))
        seen["dropped"] = max(seen["dropped"], float(puff.mass_dropped[0]))
        seen["evicted"] += int(model.closures[-1].last["evicted"].sum())
        seen["pending"] = max(seen["pending"], float(state["puff.pending"].abs().sum()))
        seen["sxz"] = max(seen["sxz"], float(puff.sxz.abs().max()))
    return state, seen


@pytest.mark.parametrize("puff_dt", [None, 30.0])
@pytest.mark.parametrize("capacity", [30, 4096])
@pytest.mark.parametrize("junction_source", ["upward", "net"])
def test_budget_exact_every_step(junction_source, capacity, puff_dt):
    """Design Sect. 4.8, at every step, with the step's flux not yet emitted counted:

        sum_s V_s (x_s - C_bg,s) + sum_live m + mass_out + mass_dropped - mass_drawn_down
            + E_pending - D_pending = sum_steps (sum_s Q_s dt + release mass)

    `E_pending` is what the next step will emit (`pending_roof_mass`: the last step's roof
    and junction fluxes times its length, plus the release's mass of the last step) and
    `D_pending` what it will add to `mass_drawn_down` (zero under `"net"`), to 1e-12
    relative to the input. With 30 slots (25 sources) puffs are evicted for capacity;
    with the default capacity they leave the 300 m domain and fall below the faint
    threshold. Under `puff_dt` = 30 s (six 5 s steps a puff) `E_pending` also holds the
    mass accumulated towards the next emission (`"puff.pending"`)."""
    state, seen = _assert_budget(junction_source, [0.3] * 60, capacity, puff_dt)
    assert (seen["pending"] > 0.0) == (puff_dt is not None)
    if capacity == 30:
        assert seen["evicted"] > 0 and seen["dropped"] != 0.0
    else:
        assert seen["evicted"] == 0 and seen["out"] != 0.0 and seen["dropped"] != 0.0
    if junction_source == "net":
        assert float(state["puff.mass_drawn_down"][0]) == 0.0
    else:
        assert float(state["puff.mass_drawn_down"][0]) > 0.0


@pytest.mark.parametrize("puff_dt", [None, 30.0])
def test_budget_exact_with_shear(puff_dt):
    """The identity of `test_budget_exact_every_step` with the wind-shear stretching on,
    with the wind turning: the shear moves puff material along the wind, never in or out
    of the account."""
    thetas = [0.3 + 0.05 * k if k < 30 else 1.8 - 0.04 * (k - 30) for k in range(60)]
    _, seen = _assert_budget("upward", thetas, 4096, puff_dt, shear=True)
    assert seen["sxz"] > 0.0 and seen["out"] != 0.0


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


def _steady_setup(along_wind, met=None):
    from noodl.apps.street_aq.plume import (
        junction_kernel,
        plume_table,
        street_kernel,
        street_midpoints,
    )
    from noodl.apps.street_aq.puff import PuffField, puff_tables

    met = _STEADY_MET if met is None else met
    net, _ = twelve_street_grid()
    table = plume_table(**met)
    kernel = torch.cat([street_kernel(net, table, theta_w=_STEADY_THETA),
                        junction_kernel(net, table, theta_w=_STEADY_THETA)], dim=1)
    tables = puff_tables(centres=torch.tensor([10.0], dtype=DT), **met)
    field = PuffField.for_network(net, theta_w=_STEADY_THETA, along_wind=along_wind)
    return net, kernel, tables, field, street_midpoints(net)


def _steady_train(field, tables, dt, puff_dt=None):
    """The puffs of a unit continuous source in every street and junction (one batch
    instance per source) once steady: ages dt, 2 dt, ... while x(age) <= x_max, mass dt,
    displaced x(age) along the wind -- what emit / advance / housekeep leave behind
    (`test_steady_train_is_what_the_step_loop_leaves`). With an emission interval
    `puff_dt` (the street step `dt` dividing it), the train at the step right after an
    emission: ages `puff_dt/2 + dt/2 + j puff_dt`, mass `puff_dt`."""
    zero = torch.zeros((), dtype=torch.long)
    n_src = field.n_sources
    if puff_dt is None:
        ages = torch.arange(1, int(float(tables.t[0, 0, -1]) / dt) + 2, dtype=DT) * dt
    else:
        ages = (puff_dt / 2.0 + dt / 2.0 + torch.arange(
            int(float(tables.t[0, 0, -1]) / puff_dt) + 2, dtype=DT) * puff_dt)
        dt = puff_dt
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


def test_steady_train_with_a_puff_interval_is_what_the_step_loop_leaves():
    """Under an emission interval (`puff_dt` = 30 s, street step 5 s) the puffs emitted
    every sixth step at the age `puff_dt/2 - dt/2`, displaced `x` of it, are the direct
    train of `_steady_train(..., puff_dt)` at the step after an emission."""
    from noodl.apps.street_aq.puff import advance, emit, housekeep

    _, kernel, tables, field, receptors = _steady_setup("none")
    dt, puff_dt, n_src = 5.0, 30.0, field.n_sources
    state = PuffState.empty((n_src,), 60)
    n_emit = int(float(tables.t[0, 0, -1]) / puff_dt) + 3
    for k in range(6 * n_emit + 1):
        if k % 6 == 0:
            state, _ = emit(field, state, torch.eye(n_src, dtype=DT) * puff_dt,
                            torch.zeros(n_src, dtype=torch.long), tables=tables,
                            age=puff_dt / 2.0 - dt / 2.0, theta_w=_STEADY_THETA)
        state = advance(state, tables, dt, _STEADY_THETA)
        state = housekeep(field, state, tables, drop_below=0.0)
    direct = _steady_train(field, tables, dt, puff_dt)
    order = torch.argsort(state.age.masked_fill(~state.live, math.inf), dim=1)
    live = int(state.live[0].sum())
    assert live == direct.age.shape[1]
    torch.testing.assert_close(state.age.gather(1, order)[:, :live], direct.age,
                               rtol=0, atol=0)
    disp = state.disp.gather(1, order[..., None].expand(-1, -1, 2))[:, :live]
    torch.testing.assert_close(disp, direct.disp, rtol=1e-13, atol=1e-11)
    looped = _kernel_error(field, tables, receptors, kernel, state, puff_dt)
    assert looped == pytest.approx(_kernel_error(field, tables, receptors, kernel, direct,
                                                 puff_dt), abs=1e-14)


def _train_disp(tables, ages):
    """`(n, k, 2)`: the displacement `x(age)` along `_STEADY_THETA` of `ages` `(n, k)`."""
    zero = torch.zeros((), dtype=torch.long)
    e = torch.tensor([math.cos(_STEADY_THETA), math.sin(_STEADY_THETA)], dtype=DT)
    return tables.at_age(ages, zero, zero)[0][..., None] * e


def test_steady_limit_with_a_puff_interval_is_the_kernel():
    """Test 2a with an emission interval: the train of `puff_dt` = 30 s puffs, each a slug
    spanning its interval, at the 5 s street step right after an emission (ages counted
    from the intervals' middles) is the kernel to rounding, as one puff per step is. At
    the five later steps of the interval the newest material is still accumulating (the
    lag gap `[0, x(dt/2 + j dt))`, j = 1..5): the error there is reported, not bounded."""
    _, kernel, tables, field, receptors = _steady_setup("none")
    dt, puff_dt = 5.0, 30.0
    train = _steady_train(field, tables, dt, puff_dt)
    e = _kernel_error(field, tables, receptors, kernel, train, puff_dt)
    later = []
    for j in range(1, 6):
        age = train.age + j * dt
        later.append(_kernel_error(field, tables, receptors, kernel,
                                   replace(train, age=age, disp=_train_disp(tables, age)),
                                   puff_dt))
    print(f"steady limit, puff_dt = 30 s, dt = 5 s: e = {e:.2e} after an emission; at the "
          f"later steps {[f'{v:.2e}' for v in later]}")
    assert e <= 1e-12


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
    # measured (measurements.md, Round 3): 3.8e-16 to 7.5e-16 at every dt, i.e. rounding
    assert all(e <= 1e-12 for e in errors.values())


def test_steady_limit_in_a_weakly_unstable_hour_is_the_kernel():
    """Test 2a in a weakly unstable hour (L = -2000 m, h = 330 m: h / |L| = 0.165, tabulated
    as neutral turbulence under the unstable wind profile, as Paris's first hour): the puff
    tables and the kernel come from the same inputs, so the train is the kernel to
    rounding."""
    met = {**_STEADY_MET, "u_star": 0.3, "h_abl": 330.0, "lmo": -2000.0}
    _, kernel, tables, field, receptors = _steady_setup("none", met)
    errors = {dt: _kernel_error(field, tables, receptors, kernel,
                                _steady_train(field, tables, dt), dt)
              for dt in (10.0, 2.5)}
    print("steady limit, weakly unstable, along_wind='none', e(dt):", errors)
    assert all(e <= 1e-12 for e in errors.values())


def test_steady_limit_in_a_weakly_stable_hour_is_the_kernel():
    """Test 2a in an hour of the band 0 < h / L <= 1 (L = 660 m, h = 330 m), tabulated as
    neutral turbulence under the stable wind profile: the train is the kernel to
    rounding."""
    met = {**_STEADY_MET, "u_star": 0.3, "h_abl": 330.0, "lmo": 660.0}
    _, kernel, tables, field, receptors = _steady_setup("none", met)
    errors = {dt: _kernel_error(field, tables, receptors, kernel,
                                _steady_train(field, tables, dt), dt)
              for dt in (10.0, 2.5)}
    print("steady limit, weakly stable, along_wind='none', e(dt):", errors)
    assert all(e <= 1e-12 for e in errors.values())


def test_steady_limit_with_along_wind_spread_has_a_converged_floor():
    """Test 2b, kernel level: with `along_wind="sigma_y"` the steady limit is the kernel
    smoothed along the wind, a property of the slender-plume steady kernel, not an error;
    its distance to the PLAIN kernel (max |dK| / max K) is the floor. Measured
    (measurements.md, Round 3): 0.117 (10 s), 0.0699 (5 s), 0.0577 (2.5 s), 0.0457
    (1.25 s), 0.0468 (0.625 s): below 0.10 from 5 s, and converged at O(dt) because a
    receptor reads the table at its own distance clamped to the slug's ends."""
    _, kernel, tables, field, receptors = _steady_setup("sigma_y")
    floor = {dt: _kernel_error(field, tables, receptors, kernel,
                               _steady_train(field, tables, dt), dt)
             for dt in (5.0, 2.5, 1.25, 0.625)}
    print("steady limit, along_wind='sigma_y', floor against the plain kernel:", floor)
    assert all(f < 0.10 for f in floor.values())
    assert abs(floor[0.625] - floor[1.25]) < 0.10 * floor[1.25]


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
        drop_below=0.0, capacity=2000, puff_dt=None))
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


# --- Tests 4 and 5: gradients, batching, checkpointing (design Sect. 4.5, 4.9) ---------------

# Ruling P3: the wind is oblique (0.3 rad) and the faint threshold is off (drop_below=0: a
# relative drop decision is a cut-off like the spatial ones), so no (puff, receptor) pair
# and no drop decision sits within the finite-difference step of a cut-off; the pulse and
# the release end inside a step (28 s, 29 s), away from the overlap's kinks.
# They run one puff per step (puff_dt=None): under puff_dt = 30 s the release [12, 29] s
# lies inside one emission interval, so its start has no effect on the puffs.
_GRAD_OPTIONS = PuffOptions(sigma_theta=math.radians(6.0), n_releases=1, drop_below=0.0,
                            capacity=1024, puff_dt=None)
_GRAD_TARGET = 6  # street 7 (D-B), downwind of the pulse street 8 (E8-C) along 0.3 rad
_GRAD_PULSE_STREET = 7
_GRAD_PARAMS = {"pulse.rate": 1e-3, "pulse.start": 7.0, "puff.release_rate": 3e-3,
                "puff.release_start": 12.0, "puff.release_duration": 17.0,
                "puff.release_xy": (-150.0, 20.0), "u_star": 0.5, "theta_w": 0.3}


def _pulse_model(options=_GRAD_OPTIONS):
    """The street model with puffs and a `PulseSource` (run first, so its sources reach
    the street step), the puff closure bound to it."""
    from noodl.apps.street_aq.puff import PulseSource

    net, names = twelve_street_grid()
    model, state, drivers = build_model(net, background="per_street", puffs=options)
    pulsed = Model(model.net, model.layers,
                   closures=[PulseSource(model.net, names), *model.closures])
    model.closures[-1].bind(pulsed)
    return pulsed, state, drivers


def _release_drivers(drivers, p, street):
    """The drivers of a batch of `B` releases: `p` maps the names of `_GRAD_PARAMS` to
    `(B,)` tensors (`(B, 2)` for `"puff.release_xy"`), `street` `(B,)` is the pulse's
    street; every meteorology driver carries the batch."""
    n = street.shape[0]

    def full(*shape, value):
        return torch.full(shape, value, dtype=DT)

    return {**drivers, "u_star": p["u_star"], "theta_w": p["theta_w"],
            "h_abl": full(n, value=800.0), "street.x_boundary": full(12, value=2.0**-26),
            "pulse.weights": torch.nn.functional.one_hot(street, 12).to(DT),
            "pulse.rate": p["pulse.rate"], "pulse.start": p["pulse.start"],
            "pulse.duration": full(n, value=21.0),
            "puff.release_xy": p["puff.release_xy"][:, None, :],
            "puff.release_z": full(n, 1, value=30.0),
            "puff.release_rate": p["puff.release_rate"][:, None],
            "puff.release_start": p["puff.release_start"][:, None],
            "puff.release_duration": p["puff.release_duration"][:, None]}


def _integrated_c_ext(model, state, drv, *, steps=40, dt=5.0):
    """`(B, 12)`: each street's `C_ext` integrated over `steps` steps of `dt`, kg s/m3."""
    total = 0.0
    for k in range(steps):
        state = model.step(state, drv, dt, t=dt * k)
        total = total + state["puff.c_ext"] * dt
    return total


def _grad_leaves(n=1, requires_grad=True):
    out = {}
    for name, value in _GRAD_PARAMS.items():
        v = torch.tensor(value, dtype=DT)
        out[name] = v.expand(n, *v.shape).clone().requires_grad_(requires_grad)
    return out


def _gradient_check(options=_GRAD_OPTIONS):
    """`{(name, component): (autograd, central difference)}` of the integrated `C_ext` of
    the downwind street: autograd from one run, the central differences (relative step
    1e-6) from one batched run of every perturbation (the batch is exact, Test 5)."""
    model, state, drivers = _pulse_model(options)
    leaves = _grad_leaves()
    street = torch.tensor([_GRAD_PULSE_STREET])
    loss = _integrated_c_ext(model, state, _release_drivers(drivers, leaves, street))
    grads = torch.autograd.grad(loss[0, _GRAD_TARGET], list(leaves.values()))
    cases = [(name, c) for name in _GRAD_PARAMS
             for c in ((0, 1) if name == "puff.release_xy" else (None,))]
    p = _grad_leaves(2 * len(cases), requires_grad=False)
    steps = []
    for j, (name, c) in enumerate(cases):
        at = (slice(2 * j, 2 * j + 2),) if c is None else (slice(2 * j, 2 * j + 2), c)
        v = p[name][at][0].clone()
        h = 1e-6 * abs(float(v))
        p[name][at] = torch.stack([v + h, v - h])
        steps.append(float(p[name][at][0] - p[name][at][1]))
    with torch.no_grad():
        values = _integrated_c_ext(
            model, state, _release_drivers(drivers, p, street.expand(2 * len(cases))))
    values = values[:, _GRAD_TARGET]
    out = {}
    for j, (name, c) in enumerate(cases):
        g = grads[list(_GRAD_PARAMS).index(name)][0]
        auto = float(g if c is None else g[c])
        out[name, c] = (auto, float(values[2 * j] - values[2 * j + 1]) / steps[j])
    return out


@pytest.fixture(scope="module")
def gradient_check():
    return _gradient_check()


@pytest.mark.parametrize("param", ["pulse.rate", "pulse.start", "puff.release_rate",
                                   "puff.release_start", "puff.release_duration",
                                   "puff.release_xy", "u_star", "theta_w"])
def test_gradient_matches_central_differences(param, gradient_check):
    """Test 4: the gradient of the downwind street's integrated `C_ext` over 40 steps at
    dt = 5 s to the release, the pulse and the meteorology, against central differences
    (relative step 1e-6), to 1e-6 relative (both components of the release position)."""
    checked = {c: v for (name, c), v in gradient_check.items() if name == param}
    assert checked
    for c, (auto, fd) in checked.items():
        assert auto != 0.0, (param, c)
        assert abs(auto - fd) <= 1e-6 * abs(auto), (param, c, auto, fd)


@pytest.mark.slow
def test_sheared_gradient_matches_central_differences(gradient_check):
    """Test 4 with the wind-shear stretching on (design Sect. 4.7), every parameter: the
    shear rate's dependence on `u_star` included (and the shear changes that gradient)."""
    checked = _gradient_check(replace(_GRAD_OPTIONS, shear=True))
    for (param, c), (auto, fd) in checked.items():
        assert auto != 0.0, (param, c)
        assert abs(auto - fd) <= 1e-6 * abs(auto), (param, c, auto, fd)
    assert checked["u_star", None][0] != gradient_check["u_star", None][0]


def test_batch_of_three_releases_equals_three_runs():
    """Test 5: three releases (different street, start time and u*) in one batch give each
    separate run's values (C_ext at every step and the street concentrations, 1e-13
    relative) and gradients (to u*, the start and the rate, 1e-12 relative)."""
    model, state, drivers = _pulse_model()
    streets = torch.tensor([7, 2, 5])
    starts, u_stars = [7.0, 3.0, 11.0], [0.5, 0.4, 0.6]
    wrt = ("u_star", "pulse.start", "pulse.rate")

    def run(index):
        p = _grad_leaves(len(index))
        with torch.no_grad():
            p["pulse.start"].copy_(torch.tensor([starts[i] for i in index], dtype=DT))
            p["u_star"].copy_(torch.tensor([u_stars[i] for i in index], dtype=DT))
        drv = _release_drivers(drivers, p, streets[list(index)])
        s, history = state, []
        for k in range(40):
            s = model.step(s, drv, 5.0, t=5.0 * k)
            history.append(torch.cat([s["puff.c_ext"], s["street.x"]], dim=-1))
        history = torch.stack(history, dim=1)          # (B, steps, 24)
        loss = history[..., :12].sum()
        return history.detach(), torch.autograd.grad(loss, [p[name] for name in wrt])

    batch, batch_grads = run([0, 1, 2])
    for i in range(3):
        alone, grads = run([i])
        scale = alone.abs().max()
        assert (batch[i] - alone[0]).abs().max() <= 1e-13 * scale, i
        for name, g_b, g_1 in zip(wrt, batch_grads, grads, strict=True):
            assert abs(float(g_b[i] - g_1[0])) <= 1e-12 * abs(float(g_1[0])), (i, name)


def test_second_order_is_refused_by_name():
    """A `create_graph=True` gradient through the coupled run is refused by name: through
    the tables (u*) and through the street step (the pulse rate)."""
    model, state, drivers = _pulse_model()
    street = torch.tensor([_GRAD_PULSE_STREET])
    for name in ("u_star", "pulse.rate"):
        leaves = _grad_leaves()
        loss = _integrated_c_ext(model, state, _release_drivers(drivers, leaves, street),
                                 steps=8)[0, _GRAD_TARGET]
        with pytest.raises(RuntimeError, match="second-order differentiation"):
            torch.autograd.grad(loss, leaves[name], create_graph=True)


def _turning_run(model, state, drivers, leaves, steps):
    """`(C_ext (steps, 12), street x (12,))` of a run whose wind turns every step (the
    closure rebuilds its field each step, so a checkpoint replayed in the backward must use
    its own step's field)."""
    street = torch.tensor([_GRAD_PULSE_STREET])
    drv = _release_drivers(drivers, leaves, street)
    s, history = state, []
    for k in range(steps):
        s = model.step(s, {**drv, "theta_w": leaves["theta_w"] + 0.02 * k}, 5.0, t=5.0 * k)
        history.append(s["puff.c_ext"][0])
    return torch.stack(history), s["street.x"][0]


@pytest.mark.parametrize("puff_dt", [None, 30.0])
def test_checkpoint_leaves_values_and_gradients_bit_identical(monkeypatch, puff_dt):
    """The per-step checkpoint (design Sect. 4.5) changes no value and no gradient, bit
    for bit, also across two differentiated runs on the same closure (its per-run cache)
    and with the wind turning every step; under `puff_dt` the mass accumulated between
    emissions carries the graph from step to step outside the checkpoint."""
    from noodl.apps.street_aq import puff

    out = {}
    for checkpointed in (True, False):
        monkeypatch.setattr(puff, "_CHECKPOINT", checkpointed)
        model, state, drivers = _pulse_model(replace(_GRAD_OPTIONS, puff_dt=puff_dt))
        for run in range(2):
            leaves = _grad_leaves()
            c_ext, x = _turning_run(model, state, drivers, leaves, 20)
            loss = c_ext[:, _GRAD_TARGET].sum() + 1e3 * x.sum()
            grads = torch.autograd.grad(loss, list(leaves.values()))
            out[checkpointed, run] = (c_ext.detach(), x.detach(), grads)
    for run in range(2):
        a, b = out[True, run], out[False, run]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for g_a, g_b in zip(a[2], b[2], strict=True):
            assert torch.equal(g_a, g_b)
    # the release [12, 29] s inside one 30 s interval: its start moves no puff
    moot = {"puff.release_start"} if puff_dt else set()
    assert all(bool((g != 0).any()) == (name not in moot)
               for name, g in zip(_GRAD_PARAMS, out[True, 0][2], strict=True))


def _saved_bytes_per_step(steps=40):
    """`(live puffs, autograd-saved bytes)` after every step of a differentiated run, the
    bytes summed over every tensor autograd saves (`saved_tensors_hooks`)."""
    model, state, drivers = _pulse_model()
    leaves = _grad_leaves()
    drv = _release_drivers(drivers, leaves, torch.tensor([_GRAD_PULSE_STREET]))
    total, rows = [0], []

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        s = state
        for k in range(steps):
            s = model.step(s, drv, 5.0, t=5.0 * k)
            rows.append((int(s["puff.live"].sum()), total[0]))
    return torch.tensor(rows, dtype=DT)


def test_saved_bytes_grow_linearly_with_the_live_puffs(monkeypatch):
    """Under the per-step checkpoint the graph holds what each step saves outside the
    checkpointed puff step (the street step's flows and solve), the same every step, and,
    by reference, each step's input puff state ((B, K) slot arrays, which the hooks do not
    see: proportional to the capacity). So the saved bytes after step k are linear
    in the live puff count (which grows linearly here, ~16 puffs a step): a straight-line
    fit leaves < 1 % of the final total. Without the checkpoint every step also saves its
    (puff, receptor, sub-source) products, which grow with the live puffs, so the total
    grows quadratically (measured: 0.86 MB with, 605 MB without, at 40 steps)."""
    from noodl.apps.street_aq import puff

    def straight_line_residual(rows):
        live, saved = rows[5:, 0], rows[5:, 1]   # past the first puffs' arrival
        a = torch.stack([torch.ones_like(live), live], dim=1)
        fit = torch.linalg.lstsq(a, saved[:, None]).solution
        return float((a @ fit - saved[:, None]).abs().max() / saved[-1])

    rows = _saved_bytes_per_step()
    assert float(rows[-1, 0]) > 5 * float(rows[5, 0])
    assert straight_line_residual(rows) < 0.01
    monkeypatch.setattr(puff, "_CHECKPOINT", False)
    plain = _saved_bytes_per_step()
    assert straight_line_residual(plain) > 0.05
    assert float(plain[-1, 1]) > 100 * float(rows[-1, 1])


def test_gradient_with_a_puff_interval_matches_central_differences():
    """Test 4 under `puff_dt` = 30 s (the mass accumulated between emissions carries the
    graph): the gradient of the downwind street's integrated `C_ext` to the pulse and
    release rates and the meteorology against central differences (relative step 1e-6),
    to 1e-6 relative."""
    model, state, drivers = _pulse_model(replace(_GRAD_OPTIONS, puff_dt=30.0))
    names = ("pulse.rate", "puff.release_rate", "u_star", "theta_w")
    leaves = _grad_leaves()
    street = torch.tensor([_GRAD_PULSE_STREET])
    loss = _integrated_c_ext(model, state, _release_drivers(drivers, leaves, street))
    grads = torch.autograd.grad(loss[0, _GRAD_TARGET], [leaves[n] for n in names])
    p = _grad_leaves(2 * len(names), requires_grad=False)
    steps = []
    for j, name in enumerate(names):
        v = p[name][2 * j].clone()
        h = 1e-6 * abs(float(v))
        p[name][2 * j:2 * j + 2] = torch.stack([v + h, v - h])
        steps.append(float(p[name][2 * j] - p[name][2 * j + 1]))
    with torch.no_grad():
        values = _integrated_c_ext(
            model, state, _release_drivers(drivers, p, street.expand(2 * len(names))))
    values = values[:, _GRAD_TARGET]
    for j, name in enumerate(names):
        auto = float(grads[j][0])
        fd = float(values[2 * j] - values[2 * j + 1]) / steps[j]
        assert auto != 0.0, name
        assert abs(auto - fd) <= 1e-6 * abs(auto), (name, auto, fd)


# ------------------------------------------------------------------ the emission interval

_REGRESSION = (Path(__file__).parents[2] / "data" / "street"
               / "puff_one_per_step_regression.pt")


def _regression_run(options):
    """The run recorded in `_REGRESSION` before `puff_dt` existed (commit e90355c): a pulse
    in streets 3 and 9 over [0, 20) s, a steady source in street 6, an above-roof release of
    3e-3 kg/s over [7, 30] s at (-60, 90) m, z = 30 m, background 2e-8, the wind turning
    0.01 rad a step from 0.3 rad, 40 steps of 5 s; `(C_ext (40, 12), street x (40, 12),
    final state)`."""
    net, names = twelve_street_grid()
    model, state, drivers = build_model(net, background="per_street", puffs=options)
    bg = 2.0e-8
    state = {**state, "street.x": torch.full((12,), bg, dtype=DT)}
    pulse = _street_sources(model, {names[2]: 1.7e-3, names[8]: 0.3e-3})
    steady = _street_sources(model, {names[5]: 2e-6})
    release = {"puff.release_xy": torch.tensor([[[-60.0, 90.0]]], dtype=DT),
               "puff.release_z": torch.tensor([[30.0]], dtype=DT),
               "puff.release_rate": torch.tensor([[3e-3]], dtype=DT),
               "puff.release_start": torch.tensor([[7.0]], dtype=DT),
               "puff.release_duration": torch.tensor([[23.0]], dtype=DT)}
    c, x = [], []
    for k in range(40):
        t = 5.0 * k
        drv = {**drivers, "u_star": torch.tensor(0.5, dtype=DT),
               "theta_w": torch.tensor(0.3 + 0.01 * k, dtype=DT),
               "h_abl": torch.tensor(800.0, dtype=DT),
               "street.x_boundary": torch.full((12,), bg, dtype=DT),
               "street.sources": steady + (pulse if t < 20.0 else torch.zeros_like(pulse)),
               **release}
        state = model.step(state, drv, 5.0, t=t)
        c.append(state["puff.c_ext"].clone())
        x.append(state["street.x"].clone())
    return torch.stack(c), torch.stack(x), state


def test_puff_dt_none_is_bit_identical_to_one_puff_per_step():
    """`puff_dt=None` is the model as it was before the emission interval, bit for bit
    (the run recorded at e90355c: C_ext and the street concentrations at every step, the
    budget counters, the live puffs), and `puff_dt` equal to the street step is the same
    model too."""
    ref = torch.load(_REGRESSION, weights_only=True)
    options = PuffOptions(sigma_theta=math.radians(6.0), n_releases=1, drop_below=1e-6,
                          capacity=4096, puff_dt=None)
    c, x, state = _regression_run(options)
    assert torch.equal(c, ref["c_ext"]) and torch.equal(x, ref["street_x"])
    puff = PuffState.from_state(state)
    for name, value in ref["counters"].items():
        assert torch.equal(getattr(puff, name), value), name
    assert torch.equal(puff_mass(puff), ref["live_mass"])
    assert torch.equal(puff.live.sum(1), ref["live"])
    c5, x5, _ = _regression_run(replace(options, puff_dt=5.0))
    assert torch.equal(c5, c) and torch.equal(x5, x)


def _release_series(options, steps=80):
    """`(street x - C_bg (steps, 12), C_ext - C_bg (steps, 12))` of an above-roof release
    of 3e-3 kg/s over [12, 29] s at (-150, 20) m, z = 30 m (the Test 4 release), wind 0.3
    rad, u* = 0.5 m/s, streets starting at the background, dt = 5 s."""
    model, state, drivers = _pulse_model(options)
    leaves = _grad_leaves(requires_grad=False)
    leaves["pulse.rate"] = torch.zeros(1, dtype=DT)
    drv = _release_drivers(drivers, leaves, torch.tensor([_GRAD_PULSE_STREET]))
    bg = drv["street.x_boundary"]
    state = {**state, "street.x": bg.clone()}
    xs, cs = [], []
    with torch.no_grad():
        for k in range(steps):
            state = model.step(state, drv, 5.0, t=5.0 * k)
            xs.append(state["street.x"][0] - bg)
            cs.append(state["puff.c_ext"][0] - bg)
    return torch.stack(xs), torch.stack(cs)


@pytest.fixture(scope="module")
def release_series():
    """The release run under (puff_dt, drop_below) = (None, 1e-6), (30 s, 1e-6) and
    (None, 1e-3)."""
    base = replace(_GRAD_OPTIONS, capacity=4096)
    return {(puff_dt, drop): _release_series(replace(base, puff_dt=puff_dt, drop_below=drop))
            for puff_dt, drop in ((None, 1e-6), (30.0, 1e-6), (None, 1e-3))}


# the street the release reaches most (street 3, index 2: peak excess 3.7e-8 kg/m3)
_RELEASE_TARGET = 2


def _series_error(got, ref):
    """max_t |got - ref| / max_t |ref| of the target street, and the relative error of its
    time integral (the dose)."""
    g, r = got[:, _RELEASE_TARGET], ref[:, _RELEASE_TARGET]
    return float((g - r).abs().max() / r.abs().max()), float((g.sum() - r.sum()) / r.sum())


def test_puff_interval_accuracy_against_one_puff_per_step(release_series):
    """The accuracy cost of puff_dt = 30 s against one puff per 5 s step (drop_below 1e-6
    both), for a 17 s release: the 30 s slug spreads it over its interval, centred at the
    interval's middle (15 s; the release's own is 20.5 s). Measured (measurements.md, Task
    12) at the most-hit street: street concentration 0.376 of its peak, C_ext 0.570, dose
    +2.5e-2; bounds one significant figure, rounded up."""
    ref = release_series[None, 1e-6]
    x_err, dose = _series_error(release_series[30.0, 1e-6][0], ref[0])
    c_err, _ = _series_error(release_series[30.0, 1e-6][1], ref[1])
    print(f"puff_dt 30 vs None: x {x_err:.3e}, C_ext {c_err:.3e}, dose {dose:+.3e}")
    assert x_err <= 0.4 and c_err <= 0.6 and abs(dose) <= 3e-2


def test_faint_threshold_accuracy_against_1e_6(release_series):
    """The accuracy cost of drop_below = 1e-3 against 1e-6 (one puff per step). Measured
    (measurements.md, Task 12) at the most-hit street: street concentration 1.26e-4 of its
    peak, C_ext 1.06e-4, dose -5.5e-5; bounds one significant figure, rounded up."""
    ref = release_series[None, 1e-6]
    x_err, dose = _series_error(release_series[None, 1e-3][0], ref[0])
    c_err, _ = _series_error(release_series[None, 1e-3][1], ref[1])
    print(f"drop_below 1e-3 vs 1e-6: x {x_err:.3e}, C_ext {c_err:.3e}, dose {dose:+.3e}")
    assert x_err <= 2e-4 and c_err <= 2e-4 and abs(dose) <= 6e-5


# ------------------------------------------------------------------ union with a building


def test_puff_street_couples_to_a_building_through_union():
    """The puff street model and a small vented building join through `couple.union`
    (unchanged) with a two-way `ValueLink`: an above-roof pulse upwind of the grid reaches
    the street the building reads after the travel time, and the building's indoor
    concentration is zero until then and positive after."""
    from noodl.couple import CONCENTRATION_TO_MASS_FRACTION, ValueLink, union
    from tests.test_couple import _tiny_building_model

    net, names = twelve_street_grid(L=200.0)
    model, state, drivers = build_model(
        net, background="per_street", puffs=PuffOptions(sigma_theta=math.radians(6.0),
                                                        n_releases=1))
    target = names.index("4")
    drv = {**_drivers(model, drivers, bg=0.0),
           "puff.release_xy": torch.tensor([[-150.0, 200.0]], dtype=DT),
           "puff.release_z": torch.tensor([30.0], dtype=DT),
           "puff.release_rate": torch.tensor([1e-3], dtype=DT),
           "puff.release_start": torch.tensor([0.0], dtype=DT),
           "puff.release_duration": torch.tensor([10.0], dtype=DT)}
    state = {**state, "street.x": torch.zeros(12, dtype=DT)}
    b_model, b_state, b_drivers = _tiny_building_model()
    b_state = {"species.x": torch.zeros(1, dtype=DT)}
    link = ValueLink(from_model="street", from_key="street.x", from_index=target,
                     to_model="building", to_key="species.x_boundary", to_index=0,
                     convert=CONCENTRATION_TO_MASS_FRACTION, two_way=True)
    city, s, d = union({"street": (model, state, drv), "building": (b_model, b_state, b_drivers)},
                       shared=[link])
    indoor, outdoor = [], []
    for k in range(30):
        s = city.step(s, d, 10.0, t=10.0 * k)
        indoor.append(float(s["building"]["species.x"][0]))
        outdoor.append(float(s["street"]["street.x"][target]))
    first = next(k for k, v in enumerate(outdoor) if v > 0.0)
    assert first >= 1
    assert all(v == 0.0 for v in indoor[:first])
    assert indoor[-1] > 0.0
