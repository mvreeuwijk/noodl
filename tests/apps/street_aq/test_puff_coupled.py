"""The puff closure inside the street model (`noodl.apps.street_aq.puff.PuffClosure`)."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.network import (
    build_model,
    initial_state,
    street_index,
    twelve_street_grid,
)
from noodl.apps.street_aq.puff import PuffClosure, PuffOptions, PuffState
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
    assert model.closures[-1].tables.x.shape[0] == 2
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
