"""`Buildings.Fluid.Movers.FlowControlled_dp` read from hand-written `noodl-modelica/1` models.

The mover is an implicit law (`port_b.p - port_a.p = dp`, the flow an unknown), so the air layer
of such a model is solved by Modified Nodal Analysis. Reference values are the MBL power law of
`Orifice.mo` (as in `test_assemble.py`), never computed by the elements under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics import read_modelica as _read_modelica
from noodl.apps.building_physics.modelica import ModelicaImportError
from noodl.apps.building_physics.modelica.run import (
    _balanced,
    _closure,
    _initial_air,
    _storage,
    extrapolate,
    simulate,
    step_drivers,
)

FIX = Path(__file__).parent / "fixtures"
P0 = 101325.0
_BOU = "Buildings.Fluid.Sources.Boundary_pT"
_ORI = "Buildings.Airflow.Multizone.Orifice"
_MOVER = "Buildings.Fluid.Movers.FlowControlled_dp"
_ORIFICE = {"A": 0.01, "CD": 0.6, "m": 0.5, "dp_turbulent": 0.1}
_OK = {"inputType": "Buildings.Fluid.Types.InputType.Constant", "constantHead": 5.0,
       "use_riseTime": False, "addPowerToMedium": False,
       "energyDynamics": "Modelica.Fluid.Types.Dynamics.SteadyState"}


def read(path, **kwargs):
    kwargs.setdefault("mass_storage", False)
    return _read_modelica(path, **kwargs)


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(doc))
    return path


def _model(mover_params: dict, signals=()) -> dict:
    """bouA -> mover -> orifice -> bouB, both boundaries at 101325 Pa."""
    doc = json.loads((FIX / "two_zones_orifice.json").read_text())
    doc["experiment"]["StopTime"] = 60.0
    doc["components"] = [
        {"name": "bouA", "class": _BOU, "parameters": {"p": P0, "T": 293.15, "nPorts": 1}},
        {"name": "bouB", "class": _BOU, "parameters": {"p": P0, "T": 293.15, "nPorts": 1}},
        {"name": "ori", "class": _ORI, "parameters": dict(_ORIFICE)},
        {"name": "fan", "class": _MOVER, "parameters": dict(mover_params)},
    ]
    doc["connections"] = [["bouA.ports[1]", "fan.port_a"], ["fan.port_b", "ori.port_a"],
                          ["ori.port_b", "bouB.ports[1]"]]
    doc["signals"] = list(signals)
    return doc


def _power_law(dp: float) -> float:
    """`Orifice.mo:3-5`, `Coefficient_V_flow.mo:4` for `|dp| > dp_turbulent`, rho_default of
    `Buildings.Media.Specialized.Air.PerfectGas`."""
    r_air = 8.314510 / 0.0289651159
    r_h2o = 8.314510 / 0.01801528
    rho = P0 / ((r_air * 0.99 + r_h2o * 0.01) * 293.15)
    return rho * 0.6 * 0.01 * math.sqrt(2.0 / rho) * dp**0.5


def _flow(hist, names, inst: str):
    ((col, sign),) = names.edges[inst]
    return sign * hist["air.q"][:, col]


def test_constant_mover_drives_the_orifice_at_its_pressure_rise(tmp_path):
    model, state, drivers, names = read(_write(tmp_path, _model(_OK)), return_names=True)
    assert not model.transport  # no volumes: algebraic
    hist = simulate(model, state, drivers, names.times[:3])
    expected = torch.full((3,), _power_law(5.0), dtype=torch.float64)
    for inst in ("fan", "ori"):  # one flow through the junction between them
        assert torch.allclose(_flow(hist, names, inst), expected, rtol=1e-12, atol=0.0)


def test_constant_head_defaults_to_dp_nominal(tmp_path):
    params = {k: v for k, v in _OK.items() if k != "constantHead"} | {"dp_nominal": 5.0}
    model, state, drivers, names = read(_write(tmp_path, _model(params)), return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    assert float(_flow(hist, names, "ori")[-1]) == pytest.approx(_power_law(5.0), rel=1e-12)


def test_continuous_input_follows_the_signal(tmp_path):
    params = dict(_OK, inputType="Buildings.Fluid.Types.InputType.Continuous")
    params.pop("constantHead")
    ramp = {"name": "ramp", "class": "Modelica.Blocks.Sources.Ramp", "drives": "fan.dp_in",
            "parameters": {"height": 20.0, "duration": 20.0, "offset": 0.0, "startTime": 10.0}}
    model, state, drivers, names = read(_write(tmp_path, _model(params, [ramp])),
                                        return_names=True)
    hist = simulate(model, state, drivers, names.times)
    fan, ori = _flow(hist, names, "fan"), _flow(hist, names, "ori")
    for t in (15.0, 20.0, 30.0, 60.0):  # dp = 5, 10, 20, 20 Pa, all above dp_turbulent
        dp = min(max(t - 10.0, 0.0), 20.0)
        assert float(fan[int(t)]) == pytest.approx(_power_law(dp), rel=1e-12)
        assert float(ori[int(t)]) == pytest.approx(_power_law(dp), rel=1e-12)
    assert float(fan[0]) == pytest.approx(0.0, abs=1e-14)  # dp = 0 before the ramp


def test_continuous_input_without_a_signal_is_refused(tmp_path):
    params = dict(_OK, inputType="Buildings.Fluid.Types.InputType.Continuous")
    with pytest.raises(ModelicaImportError, match=r"fan.*dp_in"):
        read(_write(tmp_path, _model(params)))


@pytest.mark.parametrize(("override", "needle"), [
    ({"use_riseTime": True}, "use_riseTime"),
    ({"use_riseTime": None}, "use_riseTime"),  # absent: MBL's default is true
    ({"addPowerToMedium": True}, "addPowerToMedium"),
    ({"addPowerToMedium": None}, "addPowerToMedium"),
    ({"energyDynamics": "Modelica.Fluid.Types.Dynamics.DynamicFreeInitial"}, "energyDynamics"),
    ({"energyDynamics": None}, "energyDynamics"),
    ({"prescribeSystemPressure": True}, "prescribeSystemPressure"),
    ({"inputType": "Buildings.Fluid.Types.InputType.Stages"}, "Stages"),
])
def test_unreproduced_mover_features_are_refused_by_name(tmp_path, override, needle):
    params = {k: v for k, v in {**_OK, **override}.items() if v is not None}
    with pytest.raises(ModelicaImportError) as exc:
        read(_write(tmp_path, _model(params)))
    msg = str(exc.value)
    assert "fan" in msg and _MOVER in msg and needle in msg


def _two_zone_model() -> dict:
    """bouA -> oriA -> volA -> fan -> volB -> oriB -> bouB, both boundaries at 101325 Pa."""
    doc = _model(_OK)
    zone = {"V": 50.0, "nPorts": 2, "T_start": 293.15}
    vol = "Buildings.Fluid.MixingVolumes.MixingVolume"
    doc["components"] = [
        {"name": "bouA", "class": _BOU, "parameters": {"p": P0, "T": 293.15, "nPorts": 1}},
        {"name": "bouB", "class": _BOU, "parameters": {"p": P0, "T": 293.15, "nPorts": 1}},
        {"name": "oriA", "class": _ORI, "parameters": dict(_ORIFICE)},
        {"name": "oriB", "class": _ORI, "parameters": dict(_ORIFICE)},
        {"name": "volA", "class": vol, "parameters": dict(zone)},
        {"name": "volB", "class": vol, "parameters": dict(zone)},
        {"name": "fan", "class": _MOVER, "parameters": dict(_OK)},
    ]
    doc["connections"] = [
        ["bouA.ports[1]", "oriA.port_a"], ["oriA.port_b", "volA.ports[1]"],
        ["volA.ports[2]", "fan.port_a"], ["fan.port_b", "volB.ports[1]"],
        ["volB.ports[2]", "oriB.port_a"], ["oriB.port_b", "bouB.ports[1]"],
    ]
    return doc


def test_mover_between_two_zones_splits_the_rise_across_both_orifices(tmp_path):
    model, state, drivers, names = read(_write(tmp_path, _two_zone_model()), return_names=True)
    hist = simulate(model, state, drivers, names.times[:3])
    expected = _power_law(2.5)  # each orifice sees half of the 5 Pa rise
    for inst in ("oriA", "fan", "oriB"):
        got = _flow(hist, names, inst)
        assert torch.allclose(got, torch.full_like(got, expected), rtol=1e-12, atol=0.0)
    pA, pB = (hist["p"][:, names.nodes[z]] for z in ("volA", "volB"))
    assert torch.allclose(pB - pA, torch.full_like(pA, 5.0), rtol=1e-12, atol=0.0)


# ------------------------------------------------------------ movers with volume mass storage
# `Buildings.Media.Air`: the density is `dStp p / pStp`, so a volume's mass is `V k p` with `k`
# constant and the orifices use the default density: the airflow is decoupled from the zone
# temperatures, and the two-zone model below has a closed-form solution. The rise ties the two
# volume pressures (`p_B = p_A + 5`), so with `x = P0 - p_A` the whole state is the total mass:
#     (V_A + V_B) k dx/dt = -(q_A - q_B),   q_A = c sqrt(x),  q_B = c sqrt(5 - x),
# and the mover's flow is the one that keeps `der(p_A) = der(p_B)` (MBL's index reduction):
#     m_fan = (V_B q_A + V_A q_B) / (V_A + V_B).
# Separating variables (multiply by the conjugate; `u = sqrt(s)`, `w = sqrt(5 - s)`,
# `a = sqrt(5/2)`): t(x) = -(V_A + V_B) k / c [F(x) - F(x0)],
#     F(s) = u + 5/(4a) ln|(u - a)/(u + a)| + w - 5/(4a) ln|(a + w)/(a - w)|.
_VA, _VB, _X0 = 1000.0, 3000.0, 4.0
_RHO_AIR, _K_AIR = 1.2, 1.2 / 101325.0   # Buildings.Media.Air: dStp, dStp/pStp


def _storage_model(p_start_a: float, p_start_b: float, medium="Buildings.Media.Air") -> dict:
    doc = _two_zone_model()
    doc["medium"]["class"] = medium
    for c in doc["components"]:
        if c["class"].endswith("MixingVolume"):
            a = c["name"] == "volA"
            c["parameters"].update(
                massDynamics="Modelica.Fluid.Types.Dynamics.FixedInitial",
                V=_VA if a else _VB, p_start=p_start_a if a else p_start_b)
    return doc


def _x_closed_form(t: float) -> float:
    c = _RHO_AIR * 0.6 * 0.01 * math.sqrt(2.0 / _RHO_AIR)
    a = math.sqrt(2.5)

    def F(s):
        u, w = math.sqrt(s), math.sqrt(5.0 - s)
        return (u + 5 / (4 * a) * math.log(abs((u - a) / (u + a))) + w
                - 5 / (4 * a) * math.log(abs((a + w) / (a - w))))

    def t_of(x):
        return -(_VA + _VB) * _K_AIR / c * (F(x) - F(_X0))

    lo, hi = 2.5 + 1e-12, _X0  # t_of falls monotonically from inf at 2.5 to 0 at x0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_of(mid) > t else (lo, mid)
    return 0.5 * (lo + hi)


def _fan_flow_closed_form(x: float) -> float:
    c = _RHO_AIR * 0.6 * 0.01 * math.sqrt(2.0 / _RHO_AIR)
    return (_VB * c * math.sqrt(x) + _VA * c * math.sqrt(5.0 - x)) / (_VA + _VB)


def test_mover_with_volume_mass_storage_matches_the_closed_form(tmp_path):
    path = _write(tmp_path, _storage_model(P0 - _X0, P0 - _X0 + 5.0))  # the rise is met
    runs = []
    for r in (1, 2):
        model, state, drivers, names = _read_modelica(path, return_names=True, substeps=r)
        assert names.storage == ("volA", "volB") and model.kirchhoff["air"].mna
        runs.append(simulate(model, state, drivers, names.times[:11], scheme="midpoint"))
    hist = extrapolate(*runs)
    pA, pB = (hist["p"][:, names.nodes[z]] for z in ("volA", "volB"))
    fan = _flow(hist, names, "fan")
    assert torch.allclose(pB - pA, torch.full_like(pA, 5.0), rtol=0.0, atol=1e-9)
    # Row 0: the start pressures and the index-reduced mover flow, exactly.
    assert float(fan[0]) == pytest.approx(_fan_flow_closed_form(_X0), rel=1e-12)
    for i, t in enumerate(hist["time"].tolist()):
        x = _x_closed_form(t)
        # The parity tolerance (`test_modelica_parity`): 1e-6 relative (5e-7 at 10 s with
        # (1, 2) substeps, 1e-7 with (2, 4): the step-size error, measured).
        assert float(P0 - pA[i]) == pytest.approx(x, rel=1e-6), t
        assert float(fan[i]) == pytest.approx(_fan_flow_closed_form(x), rel=1e-6), t


def test_mover_between_storing_zones_with_a_junction_elsewhere(tmp_path):
    # oriB -> oriC in series: their junction's pressure is solved at t = StartTime with the
    # storing zones held (`init_layer`), where the mover joins two held nodes; its flow is
    # still the index-reduced one, from the orifice flows at the start pressures.
    doc = _storage_model(P0 - _X0, P0 - _X0 + 5.0)
    doc["components"].append({"name": "oriC", "class": _ORI, "parameters": dict(_ORIFICE)})
    doc["connections"].remove(["oriB.port_b", "bouB.ports[1]"])
    doc["connections"] += [["oriB.port_b", "oriC.port_a"], ["oriC.port_b", "bouB.ports[1]"]]
    model, state, drivers, names = _read_modelica(_write(tmp_path, doc), return_names=True)
    assert names.junctions
    hist = simulate(model, state, drivers, names.times[:3])
    qa, qb, fan = (_flow(hist, names, i) for i in ("oriA", "oriB", "fan"))
    assert torch.allclose(qb, _flow(hist, names, "oriC"), rtol=1e-12, atol=0.0)
    c = _RHO_AIR * 0.6 * 0.01 * math.sqrt(2.0 / _RHO_AIR)
    assert float(qa[0]) == pytest.approx(c * math.sqrt(_X0), rel=1e-12)
    assert float(fan[0]) == pytest.approx(
        (_VB * float(qa[0]) + _VA * float(qb[0])) / (_VA + _VB), rel=1e-12)
    pA, pB = (hist["p"][:, names.nodes[z]] for z in ("volA", "volB"))
    assert torch.allclose(pB - pA, torch.full_like(pA, 5.0), rtol=0.0, atol=1e-9)


def _at_start(drivers, names):
    return step_drivers(drivers, drivers.get("series:time", names.times), float(names.times[0]))


# Exactly, up to the airflow solve's tolerance; the stored mass of a `PerfectGas` volume
# depends on its temperature (`k = 1/(R T)`), which the step's coupling passes settle to
# their `iterate_tol` only (storage.py, "Coupling").
@pytest.mark.parametrize(("medium", "rel"), [("Buildings.Media.Air", 0.0),
                                             ("Buildings.Media.Specialized.Air.PerfectGas", 1e-6)])
def test_mover_with_volume_mass_storage_conserves_mass(tmp_path, medium, rel):
    # Both volumes start at P0, which the 5 Pa rise violates: the first step moves the
    # pressures onto it through the mover. Every backward-Euler step, that one included,
    # changes the stored mass by exactly what the two orifices carry in and out over it.
    path = _write(tmp_path, _storage_model(P0, P0, medium))
    model, state, drivers, names = _read_modelica(path, return_names=True)
    store = _storage(model)
    d = _at_start(drivers, names)
    state = _initial_air(model, store, state, d)
    zones = [names.nodes["volA"], names.nodes["volB"]]

    def mass(s):
        phi, k = s["air.storage"][zones, 0], s["air.storage"][zones, 1]
        return store.V[zones] * k * (phi + store.p_ref)

    def through(s, inst):
        ((col, sign),) = names.edges[inst]
        return sign * float(s["air.q"][col])

    h, t = 0.05, 0.0
    for _ in range(40):
        new = model.step(state, d, h, t=t, atol=1e-11, rtol=1e-12)
        new = {**new, "air.storage": store(new, {**d, **_closure_drivers(model, new, d)})[
            "air.storage"]}
        stored = float((mass(new) - mass(state)).sum())
        carried = h * (through(new, "oriA") - through(new, "oriB"))
        # The airflow solve's tolerance (atol 1e-11 kg/s per zone) over the step.
        assert stored == pytest.approx(carried, rel=rel, abs=2 * h * 1e-11)
        state, t = new, t + h
    pA, pB = (float(state["air.phi"][n]) for n in zones)
    assert pB - pA == pytest.approx(5.0, abs=1e-8)


def test_mover_with_volume_mass_storage_starts_out_of_balance_when_the_rise_is_not_met(
        tmp_path):
    # Equal volumes and orifices: with the rise met at p_A = P0 - 2.5 nothing moves; at
    # p_A = p_B = P0 the flows balance too (all zero), but the rise is not met.
    for (pa, pb), balanced in (((P0, P0), False), ((P0 - 2.5, P0 + 2.5), True)):
        doc = _storage_model(pa, pb)
        for c in doc["components"]:
            if c["class"].endswith("MixingVolume"):
                c["parameters"]["V"] = 50.0
        model, state, drivers, names = _read_modelica(_write(tmp_path, doc), return_names=True)
        store = _storage(model)
        d = _at_start(drivers, names)
        state = _initial_air(model, store, state, d)
        assert _balanced(model, store, state, d) is balanced


def _closure_drivers(model, state, d):
    return _closure(model)(state, dict(d))
