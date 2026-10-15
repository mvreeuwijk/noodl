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


def test_the_room_heat_removal_adds_to_a_removal_already_in_the_drivers():
    """A removal rate another closure wrote before this one is kept: the feedback's is added
    to it, as its source is added to `"thermal.sources"`."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    fb = _control(model)
    _, room = _room(model, names)
    cold = _cooled(state, room, 0.1)
    alone = fb(cold, drivers)["thermal.removal"]
    before = torch.linspace(1e-4, 2e-4, model.transport["thermal"].n_i, dtype=F64)
    both = fb(cold, {**drivers, "thermal.removal": before})["thermal.removal"]
    torch.testing.assert_close(both, before + alone, rtol=0.0, atol=0.0)
    assert alone[room].item() > 0.0


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
    key = _control(model).network.outputs_key  # the key the Python `simulate` uses too
    assert key == "controls.outputs" and "controls" not in out
    y = out[key]
    assert y.shape == (4, 4)
    assert y[0, 2].item() == pytest.approx(0.0, abs=1e-12)
    assert y[1:, 2].abs().max().item() < 0.2
    torch.testing.assert_close(y[:, 3], 3000.0 * y[:, 2])
    T = out["T"][:, names.nodes["roo"]]
    torch.testing.assert_close(y[:, 2], 5.0 * (293.15 - T))


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


def test_the_midpoint_run_converges_across_the_extraction_onset():
    """Quasi-steady airflow, the 0.05 kg/s extraction starting at 1800-1801 s: room T at
    1807.2 s from 1785.6 s at 1, 2 and 4 substeps. The drawn air's sink at a step's start
    takes that step's mean extraction, as the rest of the step does, so the second-order
    scheme's error falls about fourfold per halving (or faster: 8.3 measured; 1.15 when the
    start took the previous step's mean)."""
    T = {}
    for r in (1, 2, 4):
        model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True,
                                                     mass_storage=False, substeps=r)
        out = simulate(model, state, drivers, names.times[248:252], scheme="midpoint")
        assert float(names.times[251]) == pytest.approx(1807.2)
        T[r] = float(out["T"][-1, names.nodes["roo"]])
    assert abs(T[1] - T[4]) / abs(T[2] - T[4]) >= 3.5


def _signal(raw, name):
    return next(s for s in raw["signals"] if s["name"] == name)


def _division_by_feedback(raw):
    _signal(raw, "con")["drives"] = "div.u2"
    raw["signals"] += [
        {"name": "one", "class": "Modelica.Blocks.Sources.Constant", "parameters": {"k": 1.0},
         "drives": "div.u1"},
        {"name": "div", "class": "Modelica.Blocks.Math.Division", "parameters": {},
         "drives": "gain.u"}]


def _feedback_loop(raw):
    _signal(raw, "con")["drives"] = "fb.u1"
    raw["signals"].append({"name": "fb", "class": "Modelica.Blocks.Math.Feedback",
                           "parameters": {}, "drives": ["gain.u", "fb.u2"]})


def _undriven_input(raw):
    _signal(raw, "con")["drives"] = "add.u1"
    raw["signals"].append({"name": "add", "class": "Modelica.Blocks.Math.Add",
                           "parameters": {}, "drives": "gain.u"})


def _without(key):
    def edit(raw):
        del _signal(raw, "con")["parameters"][key]
    return edit


def _pd(raw):
    _signal(raw, "con")["parameters"]["controllerType"] = (
        "Modelica.Blocks.Types.SimpleController.PD")


@pytest.mark.parametrize(("edit", "message"), [
    (_division_by_feedback, r"div \(Modelica.Blocks.Math.Division\): divides by a feedback "
                            r"signal"),
    (_feedback_loop, r"fb \(Modelica.Blocks.Math.Feedback\): an algebraic loop of feedback "
                     r"blocks is not supported"),
    (_undriven_input, r"add\.u2: input of a feedback block that no supported signal drives"),
    (_pd, r"con \(Modelica.Blocks.Continuous.LimPID\): controllerType PD is not supported"),
    (_without("yMax"), r"con \(.*LimPID\): parameter 'yMax' is required"),
    (_without("k"), r"con \(.*LimPID\): parameter 'k' is required"),
])
def test_feedback_networks_the_heat_balance_cannot_take_are_refused(edit, message, tmp_path):
    with pytest.raises(ModelicaImportError, match=message):
        read_modelica(_edited(tmp_path, edit))


def test_a_feedback_heat_flow_into_a_pinned_zone_is_refused():
    doc = expand.expand(schema.load(WITH_VOLUME))
    g = graph.build(doc)
    errors: list[str] = []
    fb = feedback.feedback_network(doc, g, zone_index={z: i for i, z in enumerate(g.zones)},
                                   zone_temperature=lambda s, d: torch.zeros(2),
                                   pinned={"roo"}, errors=errors)
    assert errors == ["gain (Modelica.Blocks.Math.Gain): drives preHea.Q_flow, a heat flow "
                      "into zone roo, which a fixed temperature holds"]
    # A network that drives no heat flow writes nothing (its outputs are still recorded).
    assert fb.heat == ()
    closure = feedback.FeedbackHeat(fb, zone_index={z: i for i, z in enumerate(g.zones)},
                                    zone_temperature=lambda s, d: torch.zeros(2), layer=None)
    assert closure({}, {}) == {}


def _through(block):
    """`con -> <block>.u1`, a constant 2 into its other input, `<block> -> gain.u`."""
    cls, params, factor = block

    def edit(raw):
        _signal(raw, "con")["drives"] = "m.u1" if cls != "Sum" else "m.u[1]"
        raw["signals"] += [
            {"name": "two", "class": "Modelica.Blocks.Sources.Constant",
             "parameters": {"k": 2.0}, "drives": "m.u2" if cls != "Sum" else "m.u[2]"},
            {"name": "m", "class": f"Modelica.Blocks.Math.{cls}", "parameters": params,
             "drives": "gain.u"}]
    return edit, factor


@pytest.mark.parametrize("block", [
    ("Product", {}, lambda y: 2.0 * y),
    ("Add", {"k1": 3.0, "k2": -1.0}, lambda y: 3.0 * y - 2.0),
    ("Sum", {"nin": 2, "k": [3.0, 1.0]}, lambda y: 3.0 * y + 2.0),
    ("Feedback", {}, lambda y: y - 2.0),
], ids=lambda b: b[0])
def test_math_blocks_between_the_controller_and_the_heat_flow(block, tmp_path):
    """0.1 K below the set point the controller gives y = 0.5; the heat into the room is
    3000 W/(unit) times the Math block's output, affine in the room temperature."""
    edit, factor = _through(block)
    model, state, drivers, names = read_modelica(_edited(tmp_path, edit), return_names=True)
    fb = _control(model)
    node, room = _room(model, names)
    heat = fb.heat_flows(_cooled(state, room, 0.1), drivers)
    assert heat[node].item() == pytest.approx(3000.0 * factor(0.5), abs=1e-6)  # A + B T


def test_the_controller_gain_and_limit_carry_gradients_into_the_room_heat():
    """The LimitedPID's gain and upper limit set as tensors: 0.1 K below the set point the
    heat is 3000 k 0.1 W (dQ/dk = 300 W), its conductance 3000 k W/K (d/dk = 3000 W/K); 1 K
    below it the limit holds, Q = 3000 y_max (dQ/dy_max = 3000 W, dQ/dk = 0)."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    fb = _control(model)
    node, room = _room(model, names)
    pid = fb.network.blocks[2]
    k = torch.tensor(5.0, dtype=F64, requires_grad=True)
    y_max = torch.tensor(1.0, dtype=F64, requires_grad=True)
    pid.gain, pid.y_max = k, y_max
    cold = _cooled(state, room, 0.1)
    heat = fb.heat_flows(cold, drivers)[node]
    assert heat.item() == pytest.approx(1500.0, rel=1e-12)
    assert torch.autograd.grad(heat, k)[0].item() == pytest.approx(300.0, rel=1e-10)
    removal = fb(cold, drivers)["thermal.removal"][room]
    cap = float(model.transport["thermal"].capacity[room])
    assert torch.autograd.grad(removal, k)[0].item() == pytest.approx(3000.0 / cap, rel=1e-12)
    colder = _cooled(state, room, 1.0)
    heat = fb.heat_flows(colder, drivers)[node]
    assert heat.item() == pytest.approx(3000.0, rel=1e-12)
    grads = torch.autograd.grad(heat, (k, y_max), allow_unused=True)
    assert grads[0] is None or grads[0].item() == 0.0
    assert grads[1].item() == pytest.approx(3000.0, rel=1e-12)


def _room_after_one_step(gain):
    """The room 0.1 K below its set point, one 1 s `Model.step` with the controller's gain
    `gain`: the room temperature (K) at its end."""
    model, state, drivers, names = read_modelica(WITH_VOLUME, return_names=True)
    _control(model).network.blocks[2].gain = gain
    _, room = _room(model, names)
    new = model.step(_cooled(state, room, 0.1), drivers, 1.0, t=0.0)
    return new["thermal.x"][room]


def test_the_controller_gain_has_the_finite_difference_gradient_through_a_step():
    """d(room temperature after a step)/d(gain) by autograd through the implicit heat
    balance against a central difference: they agree to 7e-8, measured; 2e-6 is FD noise
    room. A step, not a run: `simulate` on this route is slow under autograd."""
    k = torch.tensor(5.0, dtype=F64, requires_grad=True)
    (grad,) = torch.autograd.grad(_room_after_one_step(k), k)
    h = 1e-4
    with torch.no_grad():
        fd = (_room_after_one_step(5.0 + h) - _room_after_one_step(5.0 - h)) / (2 * h)
    assert grad.item() > 0.0             # a stronger controller warms the cold room faster
    assert grad.item() == pytest.approx(fd.item(), rel=2e-6)
