"""Feedback control in the Modelica route (MSL 4.1.0 LimPID as a P controller; MBL v13.0.0's
ChimneyShaft and TrickleVent examples)."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.controls import (
    Affine,
    ControlNetwork,
    DriverValue,
    LimitedPID,
    Sensor,
)
from noodl.apps.building_physics.modelica import expand, feedback, graph, read_modelica, schema
from noodl.apps.building_physics.modelica.run import _Switches, simulate
from noodl.apps.building_physics.modelica.schema import ModelicaImportError

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica"
WITH_VOLUME = DATA / "ChimneyShaftWithVolume.json"
F64 = torch.float64


def _control(model) -> feedback.FeedbackHeat:
    (fb,) = [c for c in model.closures if isinstance(c, feedback.FeedbackHeat)]
    return fb


def _room(model, names, zone: str = "roo") -> tuple[int, int]:
    """The zone's node position and its position among the thermal layer's interior nodes."""
    node = names.nodes[zone]
    th = model.transport["thermal"]
    return node, int((th.interior_idx == node).nonzero()[0, 0])


def _cooled(state, room: int, by: float):
    return {**state, "thermal.x": state["thermal.x"].clone().index_add_(
        0, torch.tensor([room]), torch.tensor([-by], dtype=F64))}


def test_the_closed_loop_is_the_controller_and_the_gain():
    doc = expand.expand(schema.load(WITH_VOLUME))
    assert feedback.closed_loop(doc, graph.build(doc)) == {"con", "gain"}


def test_the_chimney_controller_is_a_p_limited_pid_heating_the_room():
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    fb = _control(model)
    assert model.closures[-1] is fb
    network = fb.network
    assert isinstance(network, ControlNetwork) and not network.sampled
    assert network.actuators == []        # the heat acts in the zone balance, not as a write
    assert [type(b) for b in network.blocks] == [DriverValue, Sensor, LimitedPID, Affine]
    pid = network.blocks[2]
    assert (pid.gain, pid.y_max, pid.y_min, pid.integral_time) == (5.0, 1.0, -1.0, None)
    assert network.blocks[3].gain == 3000.0
    assert fb.heat == ((3, names.nodes["roo"]),)
    assert drivers["controls.input"].tolist() == [293.15]
    assert names.controls == {"con": 2, "gain": 3}


def test_the_dynamic_column_volume_is_named_by_its_modelica_path():
    _, _, _, names = read_modelica(WITH_VOLUME, return_names=True)
    assert "sha.vol" in names.nodes and "sha.vol" in names.storage


def test_the_room_heat_is_an_affine_source_on_the_controllers_piece():
    """At the start roo is at its set point (293.15 K): y = 0. 0.1 K below it y = 0.5
    (k = 5) and 1500 W enter the room, written as the conductance k gain = 15000 W/K towards
    the set point: a removal rate 15000/C on the room and a source 15000 x 293.15 W. 1 K
    below it the limiter holds y = 1: 3000 W and no conductance."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    fb = _control(model)
    node, room = _room(model, names)
    assert fb.network.evaluate(state, drivers)[2].item() == pytest.approx(0.0, abs=1e-12)
    cap = model.transport["thermal"].capacity[room]
    for by, y, conductance in ((0.1, 0.5, 15000.0), (1.0, 1.0, 0.0)):
        cold = _cooled(state, room, by)
        assert fb.network.evaluate(cold, drivers)[2].item() == pytest.approx(y)
        writes = fb(cold, drivers)
        removal = writes["thermal.removal"]
        assert removal.shape == (model.transport["thermal"].n_i,)
        assert removal[room].item() == pytest.approx(conductance / float(cap), rel=1e-14)
        assert float(removal.abs().sum() - removal[room].abs()) == 0.0
        T = float(cold["thermal.x"][room])
        added = writes["thermal.sources"] - drivers["thermal.sources"]  # less the inflow's
        q = float(added[node]) - float(removal[room] * cap) * T
        assert q == pytest.approx(3000.0 * y, rel=1e-12)
        heat = fb.heat_flows(cold, drivers)
        assert heat[node].item() == pytest.approx(3000.0 * y, rel=1e-12)


def test_the_piece_of_a_step_is_the_one_its_clear_end_lies_on():
    """A step starting at the limiter's corner (located there) takes the piece of the end
    that is clear of it, whichever end that is."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    fb = _control(model)
    _, room = _room(model, names)
    corner = _cooled(state, room, 0.2 - 1e-12)          # v = 1 - 5e-12: just unsaturated
    deep = _cooled(state, room, 0.5)                    # saturated
    inside = _cooled(state, room, 0.1)                  # unsaturated
    assert fb.pieces(corner, drivers).tolist() == [0.0]
    assert fb.pieces_between(corner, drivers, deep, drivers).tolist() == [1.0]
    assert fb.pieces_between(deep, drivers, corner, drivers).tolist() == [1.0]
    assert fb.pieces_between(corner, drivers, inside, drivers).tolist() == [0.0]


def test_the_limiter_corners_are_switches():
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    from noodl.apps.building_physics.modelica.run import _closure

    closure = _closure(model)
    plain = _Switches(model, closure, 1e-3)
    air = model.potential["air"]
    state = {**state, "air.q": air.flows(state["air.phi"], {**drivers, **closure(state, drivers)})}
    vals = plain(state, drivers)
    _, room = _room(model, names)
    cold = _cooled(state, room, 0.3)                    # v = 1.5: beyond the upper corner
    upper = plain(cold, drivers)[-2:]
    assert upper.tolist() == pytest.approx([0.25, 1.25])
    assert vals[-2:].tolist() == pytest.approx([-0.5, 0.5])


@pytest.mark.parametrize("scheme", ["implicit", "midpoint"])
def test_simulate_records_the_controller_outputs(scheme):
    """Quasi-steady airflow (no graded start: a short run). The controller holds the room
    within 0.04 K of its set point (|y| < 0.2) while the shaft draws air through it, and the
    heat it delivers is the gain times its output."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True,
                                                 mass_storage=False)
    out = simulate(model, state, drivers, names.times[:4], scheme=scheme)
    assert out["controls"].shape == (4, 4)
    assert out["controls"][0, 2].item() == pytest.approx(0.0, abs=1e-12)
    assert out["controls"][1:, 2].abs().max().item() < 0.2
    torch.testing.assert_close(out["controls"][:, 3], 3000.0 * out["controls"][:, 2])
    T = out["T"][:, names.nodes["roo"]]
    torch.testing.assert_close(out["controls"][:, 2], 5.0 * (293.15 - T))


def _edited(tmp_path, edit):
    """A copy of the WithVolume JSON, edited by `edit(raw)`, under its own name."""
    raw = json.loads(WITH_VOLUME.read_text())
    edit(raw)
    out = tmp_path / WITH_VOLUME.name
    out.write_text(json.dumps(raw))
    return out


@pytest.mark.parametrize(("pid", "message"), [
    ({"controllerType": "Modelica.Blocks.Types.SimpleController.PID"},
     "controllerType PID is not supported"),
    ({"controllerType": "Modelica.Blocks.Types.SimpleController.PI"},
     "controllerType PI is not supported"),
    ({"withFeedForward": True}, "feed-forward is not supported"),
    ({"initType": "Modelica.Blocks.Types.Init.SteadyState"}, "initType SteadyState"),
    ({"initType": "Modelica.Blocks.Types.Init.InitialOutput"}, "initType InitialOutput"),
])
def test_unsupported_controller_settings_are_refused_by_name(pid, message, tmp_path):
    def edit(raw):
        for sig in raw["signals"]:
            if sig["name"] == "con":
                sig["parameters"].update(pid)

    with pytest.raises(ModelicaImportError,
                       match=rf"con \(Modelica.Blocks.Continuous.LimPID\): {message}"):
        read_modelica(_edited(tmp_path, edit))


def test_a_feedback_signal_into_anything_but_a_heat_flow_is_refused(tmp_path):
    def edit(raw):
        for sig in raw["signals"]:
            if sig["name"] == "gain":
                sig["drives"] = "boundary.m_flow_in"
            if sig["name"] == "mRoo_flow":
                sig["drives"] = "preHea.Q_flow"

    with pytest.raises(ModelicaImportError, match=r"gain .*drives boundary\.m_flow_in; a "
                                                  r"feedback signal may drive only"):
        read_modelica(_edited(tmp_path, edit))


def test_a_product_of_two_feedback_signals_is_refused(tmp_path):
    """The heat flow must be affine in the zone temperature on every piece of the limiter."""
    def edit(raw):
        for sig in raw["signals"]:
            if sig["name"] == "con":
                sig["drives"] = ["prod.u1", "prod.u2"]
        raw["signals"].append({"name": "prod", "class": "Modelica.Blocks.Math.Product",
                               "parameters": {}, "drives": "gain.u"})

    with pytest.raises(ModelicaImportError,
                       match=r"prod \(Modelica.Blocks.Math.Product\): multiplies two "
                             r"feedback signals"):
        read_modelica(_edited(tmp_path, edit))


def test_a_heat_flow_driven_by_another_zones_temperature_is_refused():
    doc = expand.expand(schema.load(WITH_VOLUME))
    g = graph.build(doc)
    ((sensor, _),) = g.heat_sensors
    shaft = next(z for z in g.zones if z.startswith("sha"))
    g = dataclasses.replace(g, heat_sensors=((sensor, shaft),))
    errors: list[str] = []
    feedback.feedback_network(doc, g, zone_index={z: i for i, z in enumerate(g.zones)},
                              zone_temperature=lambda s, d: torch.zeros(2), pinned=set(),
                              errors=errors)
    assert any("gain (Modelica.Blocks.Math.Gain): drives preHea.Q_flow, a heat flow into "
               "zone roo, from the temperature of zone sha.vol" in e for e in errors), errors
