"""Volume mass storage (`noodl.apps.building_physics.modelica.storage`).

Independent references: the backward-Euler and exponential relaxation of a volume through a
linear element (closed form), the conservation of total mass and energy of MBL's closed
heated rooms (`Examples/ClosedDoors.mo`, formulas from `PerfectGas.mo` written out here), the
MBL initial equations (`ConservationEquation.mo:219-227`), and finite differences.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import storage
from noodl.apps.building_physics.modelica.run import simulate
from noodl.elements import medium
from noodl.elements.conductance import Conductance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parents[4] / "tests" / "data" / "modelica"

# MSL SingleGasesData.mo:5,49,9187 (R_NASA_2002, Air.MM, H2O.MM); Psychrometrics/Constants.mo
# (cpAir, cpSte, h_fg); PerfectGas.mo:10 (reference_T).
R_AIR = 8.314510 / 0.0289651159
R_H2O = 8.314510 / 0.01801528
CP_AIR, CP_STE, H_FG, T_REF = 1006.0, 1860.0, 2501014.5, 273.15


def _t(values):
    return torch.tensor(values, dtype=F64)


def _one_zone(med_name: str, g: float, V: float, p_ref: float = 101325.0):
    """Boundary `out` (gauge 0) -- linear element `g` -- zone `z` with storage."""
    net = Network(dtype=F64)
    net.add_node("z")
    net.add_node("out")
    net.add_edge("z", "out", kind="c")
    th = storage.thermo(medium(med_name))
    layer = KirchhoffLayer(net, "air", [Conductance(_t([g]), kind="c")], boundary=["out"],
                           storage=[storage.ZoneStorage([0], _t([V]), p_ref, th)],
                           scheme="backward_euler")
    return layer, th


def _drivers(th, T, *, gain=0.0, w_fed=0.0):
    k = th.k(_t([T]), _t([0.0]))
    return {"T": _t([T, T]), "X_w": _t([0.0, 0.0]), "air.storage_k_prev": k,
            "air.storage_coupling_gain": _t([gain]), "air.storage_inflow": _t([w_fed])}


def _step(phi_prev, dt):
    """The step keywords from the zone's start pressure `phi_prev` (the boundary at 0)."""
    phi_prev = torch.as_tensor(phi_prev, dtype=F64).reshape(1)
    return {"dt": dt, "history": (torch.cat([phi_prev, _t([0.0])]), None)}


@pytest.mark.parametrize("med_name", ["Buildings.Media.Air", "Modelica.Media.Air.SimpleAir"])
def test_a_volume_relaxes_with_its_compressibility_time_constant(med_name):
    """dm/dt = V k dp/dt = -g p (gauge), so p decays with tau = V k / g: one backward-Euler
    step gives p0 / (1 + dt/tau) exactly, and steps of dt -> 0 converge on e^{-t/tau} at
    first order."""
    g, V, T = 2e-3, 62.5, 293.15
    layer, th = _one_zone(med_name, g, V)
    tau = V * float(th.k(_t([T]), _t([0.0]))[0]) / g
    dt = 0.3 * tau
    phi, _ = layer.solve(_t([0.0]), _drivers(th, T), None, phi0=_t([100.0]),
                         atol=1e-14, rtol=1e-14, **_step(100.0, dt))
    assert float(phi[0]) == pytest.approx(100.0 / (1.0 + dt / tau), rel=1e-12)
    errors = []
    for n in (20, 40):
        p = 100.0
        for _ in range(n):
            phi, _ = layer.solve(_t([0.0]), _drivers(th, T), None, phi0=_t([p]),
                                 atol=1e-14, rtol=1e-14, **_step(p, tau / n))
            p = float(phi[0])
        errors.append(abs(p - 100.0 * math.exp(-1.0)))
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)


def test_the_coupling_gain_does_not_move_the_converged_storage():
    """`ZoneStorage`'s blend `(r + a w_fed)/(1 + a)` is the step rate `r` itself when `w_fed`
    is the converged storage rate, whatever the gain `a`."""
    layer, th = _one_zone("Modelica.Media.Air.SimpleAir", 2e-3, 62.5)
    dt = 5.0
    phi0, _ = layer.solve(_t([0.0]), _drivers(th, 300.0), None, phi0=_t([100.0]),
                          atol=1e-15, rtol=1e-15, **_step(100.0, dt))
    w = -2e-3 * float(phi0[0])          # the element's inflow = the storage rate
    phi1, _ = layer.solve(_t([0.0]), _drivers(th, 300.0, gain=0.4, w_fed=w), None,
                          phi0=_t([50.0]), atol=1e-15, rtol=1e-15, **_step(100.0, dt))
    assert float(phi1[0]) == pytest.approx(float(phi0[0]), rel=1e-12)


def test_storage_gradcheck_through_the_airflow_solve():
    """The differentiable airflow solve with storage: gradients of the zone pressure in the
    zone temperature (through `k(T)` of the ideal gas) and in the start pressure."""
    layer, th = _one_zone("Modelica.Media.Air.SimpleAir", 2e-3, 62.5)

    def f(T, phi_prev):
        k_prev = th.k(_t([293.15]), _t([0.0]))
        drv = {"T": torch.cat([T, T]), "X_w": _t([0.0, 0.0]), "air.storage_k_prev": k_prev,
               "air.storage_coupling_gain": _t([0.0]), "air.storage_inflow": _t([0.0])}
        phi, _ = layer.solve(_t([0.0]), drv, None, phi0=_t([50.0]), atol=1e-14, rtol=1e-14,
                             **_step(phi_prev, 5.0))
        return phi

    T = _t([300.0]).requires_grad_(True)
    p = _t([80.0]).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (T, p), eps=1e-5, atol=1e-7)


@pytest.mark.parametrize("form", ["step_rate", "bdf2", "gain"])
def test_the_rate_slope_is_the_derivative_of_the_rate(form):
    """`ZoneStorage.rate_slope` is d rate / d phi (it keeps `exact_slope`), for the layer's
    own step rate, the BDF2 drivers of `run._Midpoint` and with a coupling gain; and the
    rate is the BDF2 formula `factor (m - m_prev) - offset` when those drivers are given."""
    from noodl.elements.base import LawContext

    th = storage.thermo(medium("Modelica.Media.Air.SimpleAir"))
    st = storage.ZoneStorage([0, 1], _t([62.5, 30.0]), 101325.0, th)
    drv = {"T": _t([300.0, 290.0]), "X_w": _t([0.0, 0.0]),
           "air.storage_k_prev": th.k(_t([295.0, 291.0]), _t([0.0, 0.0]))}
    if form == "bdf2":
        drv.update({"air.storage_bdf2_factor": _t([0.3, 0.3]),
                    "air.storage_bdf2_offset": _t([1e-3, -2e-3])})
    if form == "gain":
        drv.update({"air.storage_coupling_gain": _t([0.4, 0.1]),
                    "air.storage_inflow": _t([2e-3, -1e-3])})
    ctx = LawContext(mode="transient", dt=5.0, scheme="backward_euler")
    phi_start = _t([80.0, -3.0])
    phi = _t([60.0, 1.0]).requires_grad_(True)
    rate = st.rate(phi, phi_start, None, ctx, drv)
    (grad,) = torch.autograd.grad(rate.sum(), phi)
    slope = st.rate_slope(phi.detach(), phi_start, None, ctx, drv)
    assert torch.allclose(slope, grad, rtol=1e-14, atol=0.0)
    dm = storage.mass_change(st.V, th.k(drv["T"], drv["X_w"]), phi.detach(),
                             drv["air.storage_k_prev"], phi_start, 101325.0)
    if form == "step_rate":
        expected = dm / 5.0
    elif form == "bdf2":
        expected = 0.3 * dm - drv["air.storage_bdf2_offset"]
    else:
        a = drv["air.storage_coupling_gain"]
        expected = (dm / 5.0 + a * drv["air.storage_inflow"]) / (1 + a)
    assert torch.allclose(rate.detach(), expected, rtol=1e-14, atol=0.0)


def test_a_trapezoidal_step_is_refused_without_the_bdf2_drivers():
    """The closure's storage terms are backward-Euler rates, so `ZoneStorage` refuses the
    layer's trapezoidal scheme by name (the BDF2 drivers of `simulate(scheme="midpoint")`
    are the second-order route), for both the rate and its slope."""
    from noodl.elements.base import LawContext

    th = storage.thermo(medium("Modelica.Media.Air.SimpleAir"))
    st = storage.ZoneStorage([0], _t([62.5]), 101325.0, th)
    drv = {"T": _t([300.0]), "X_w": _t([0.0]), "air.storage_k_prev": th.k(_t([300.0]),
                                                                           _t([0.0]))}
    ctx = LawContext(mode="transient", dt=5.0, scheme="trapezoidal")
    for method in (st.rate, st.rate_slope):
        with pytest.raises(ValueError, match="backward Euler.*midpoint"):
            method(_t([1.0]), _t([0.0]), _t([0.0]), ctx, drv)
    # With the BDF2 drivers the layer's scheme is not used.
    drv.update({"air.storage_bdf2_factor": _t([0.2]), "air.storage_bdf2_offset": _t([0.0])})
    assert torch.isfinite(st.rate(_t([1.0]), _t([0.0]), _t([0.0]), ctx, drv)).all()


def _perfectgas_mass_energy(doc, names, out):
    """Total fluid mass and internal energy `sum m u` of MBL's volumes, from the history,
    with PerfectGas.mo's own formulas: `m = V p / (R(X_default) T(h, X_default))`
    (ConservationEquation.mo:246-253, PerfectGas.mo:494), `u = h - R(X) T` (:60-66)."""
    X_d = 0.01
    cp_d = CP_AIR * (1 - X_d) + CP_STE * X_d
    R_d = R_AIR * (1 - X_d) + R_H2O * X_d
    mass = energy = 0.0
    for c in doc["components"]:
        if not c["class"].endswith("MixingVolume"):
            continue
        i = names.nodes[c["name"]]
        p, T, X = (out[k][:, i].numpy() for k in ("p", "T", "X_w"))
        h = (T - T_REF) * (CP_AIR * (1 - X) + CP_STE * X) + H_FG * X
        T_d = T_REF + (h - H_FG * X_d) / cp_d
        m = c["parameters"]["V"] * p / (R_d * T_d)
        mass = mass + m
        energy = energy + m * (h - (R_AIR * (1 - X) + R_H2O * X) * T)
    return mass, energy


@pytest.mark.slow
def test_closed_heated_rooms_keep_their_mass_and_gain_the_heat():
    """ClosedDoors: three sealed rooms (doors closed) of PerfectGas heated by
    `Q = 100 sin(2 pi t/3600)` W. The total mass is constant and the total internal energy
    rises by `int Q` (no enthalpy leaves the building): measured 8.8e-13 and 4.8e-6 of the peak
    over the first 40 steps (the energy balance is exact only at convergence of the step's
    storage terms and to first order in the step's frozen capacity, hence the 1e-5 bound)."""
    doc = json.loads((DATA / "ClosedDoors.json").read_text())
    model, state, drivers, names = read_modelica(DATA / "ClosedDoors.json", return_names=True)
    assert names.storage == ("volA", "volB", "volC") and names.air_references == ()
    out = simulate(model, state, drivers, names.times[:41])
    mass, energy = _perfectgas_mass_energy(doc, names, out)
    t = out["time"].numpy()
    w = 2 * math.pi / 3600.0
    int_q = 100.0 * (1 - np.cos(w * t)) / w
    assert np.abs(mass / mass[0] - 1).max() < 1e-11
    assert np.abs((energy - energy[0]) - int_q).max() < 1e-5 * np.abs(int_q).max()


def test_the_start_row_holds_every_volume_at_its_start_pressure():
    """ReverseBuoyancy (`FixedInitial`, `initialize_p`): at t = StartTime MBL's volumes are
    at `p_start` = 101325 Pa against a 100000 Pa outside (ConservationEquation.mo:219-224),
    and the flows are the element laws there, not a balanced airflow."""
    model, state, drivers, names = read_modelica(DATA / "ReverseBuoyancy.json",
                                                 return_names=True)
    out = simulate(model, state, drivers, names.times[:1])
    for z in names.storage:
        assert float(out["p"][0, names.nodes[z]]) == pytest.approx(101325.0, abs=1e-9)
    (c, s), = names.edges["oriOutBot"]
    assert abs(float(out["air.q"][0, c])) > 0.1   # kg/s: the start imbalance, released later


def test_mass_storage_false_is_the_quasi_steady_route():
    model, state, _, names = read_modelica(DATA / "ClosedDoors.json", return_names=True,
                                           mass_storage=False)
    assert names.storage == () and "air.storage" not in state
    assert names.air_references == ("volA",)
    assert model.kirchhoff["air"]._withdrawals == []
