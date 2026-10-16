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
from noodl.apps.building_physics.modelica.run import simulate

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


def test_mover_with_volume_mass_storage_is_refused(tmp_path):
    doc = _two_zone_model()
    for c in doc["components"]:
        if c["class"].endswith("MixingVolume"):
            c["parameters"]["massDynamics"] = "Modelica.Fluid.Types.Dynamics.DynamicFreeInitial"
    path = _write(tmp_path, doc)
    with pytest.raises(ModelicaImportError, match=r"fan.*mass storage"):
        _read_modelica(path, mass_storage=None)
    read(path)  # the quasi-steady airflow is fine
