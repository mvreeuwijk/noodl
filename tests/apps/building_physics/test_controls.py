"""Control blocks and networks (`noodl.apps.building_physics.controls`).

The expected values are ContamX 3.4.1.7's (probes `d3`, `d4`, `d8` of the controls work):
float32 in ContamX, so they are compared to 5e-8 where ContamX's own rounding shows.
"""

import csv
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.controls import (
    AboveLimit,
    Absolute,
    Actuator,
    Affine,
    And,
    Average,
    BelowLimit,
    Block,
    Complement,
    Constant,
    ControlNetwork,
    Divide,
    DriverReading,
    DriverValue,
    Excess,
    ExclusiveOr,
    FunctionReading,
    Hysteresis,
    Identity,
    IncrementalPI,
    LimitedPID,
    LowerBandSwitch,
    Maximum,
    Minimum,
    Or,
    Positive,
    Product,
    ProportionalControl,
    Sensor,
    Shortfall,
    StateReading,
    Sum,
    UpperBandSwitch,
    edge_flow,
    edge_pressure_difference,
    node_mass_fraction,
    node_pressure,
    node_temperature,
)
from noodl.model import StepContext

F64 = torch.float64
STEP = StepContext(dt=300.0, t=0.0)


def advance(network, state, drivers, ctx=STEP):
    """One evaluation; the state carries the network's outputs, memory and evaluated flag
    forward."""
    out = network(state, drivers, ctx)
    new = {**state, **{k: out[k] for k in network.state_keys}}
    return out, new


def values(*v):
    return {"controls.input": torch.tensor(v, dtype=F64)}


def logic_network():
    blocks = [DriverValue("controls.input", 0), DriverValue("controls.input", 1),
              DriverValue("controls.input", 2),
              Absolute(0), Positive(0), Complement(0), And(0, 1), Or(0, 1),
              ExclusiveOr(0, 1), Sum(0, 1), Sum(0, 1, weights=(1.0, -1.0)), Product(0, 1),
              Sum(0, 1, 2), Average(0, 1, 2), Maximum(0, 1, 2), Minimum(0, 1, 2)]
    return ControlNetwork(blocks)


@pytest.mark.parametrize(("in1", "in3", "expected"), [
    # abs, bin, inv, and, or, xor, add, sub, mul, sum, avg, max, min (ContamX rows 3, 4, 5)
    (-0.25, 0.3, [0.25, 0, 1.25, 0, 1, 1, 0.25, -0.75, -0.125, 0.55, 0.55 / 3, 0.5, -0.25]),
    (0.0, 0.4, [0.0, 0, 1.0, 0, 1, 1, 0.5, -0.5, 0.0, 0.9, 0.3, 0.5, 0.0]),
    (0.25, 0.5, [0.25, 1, 0.75, 1, 1, 0, 0.75, -0.25, 0.125, 1.25, 1.25 / 3, 0.5, 0.25]),
])
def test_arithmetic_and_logic_follow_contamx(in1, in3, expected):
    """`inv` is 1 - u (not a negation); `bin`, `and`, `or` and `xor` test u > 0, so 0 is
    off; `sub` is the first input minus the second."""
    net = logic_network()
    out, _ = advance(net, net.initial_state(), values(in1, 0.5, in3))
    got = out[net.outputs_key][3:]
    torch.testing.assert_close(got, torch.tensor(expected, dtype=F64), rtol=0, atol=1e-12)


def test_limit_switches_and_limit_controls_follow_contamx():
    """lls/uls are strict; llc = max(0, u2 - u1), ulc = max(0, u1 - u2), not clipped above
    1; log = (u - offset) * scale. ContamX rows at in1 = -1, 0.5, 0.75 with in2 = 0.5."""
    blocks = [DriverValue("controls.input", 0), DriverValue("controls.input", 1),
              BelowLimit(0, 1), AboveLimit(0, 1), Shortfall(0, 1), Excess(0, 1),
              Affine(0, offset=0.1, gain=2.0)]
    net = ControlNetwork(blocks)
    rows = {-1.0: [1, 0, 1.5, 0, -2.2], 0.5: [0, 0, 0, 0, 0.8], 0.75: [0, 1, 0, 0.25, 1.3]}
    for in1, expected in rows.items():
        out, _ = advance(net, net.initial_state(), values(in1, 0.5))
        torch.testing.assert_close(out[net.outputs_key][2:], torch.tensor(expected, dtype=F64),
                                   rtol=0, atol=1e-12)


def test_an_input_later_in_the_order_is_read_from_the_previous_evaluation():
    """ContamX evaluates in the file's sequence order; a pass node sequenced before its
    sensor reads the sensor's previous value, 0 before the first (measured: the fan's
    multiplier 0 at t = 0, then one more step behind)."""
    late = ControlNetwork([Identity(1), DriverValue("controls.input", 0)])
    early = ControlNetwork([DriverValue("controls.input", 0), Identity(0)])
    s_late, s_early = late.initial_state(), early.initial_state()
    got_late, got_early = [], []
    for a in (0.5, 0.48, 0.47):
        o, s_late = advance(late, s_late, values(a))
        got_late.append(float(o[late.outputs_key][0]))
        o, s_early = advance(early, s_early, values(a))
        got_early.append(float(o[early.outputs_key][1]))
    assert got_late == [0.0, 0.5, 0.48]
    assert got_early == [0.5, 0.48, 0.47]


def test_a_zero_divisor_is_refused_naming_the_block():
    net = ControlNetwork([DriverValue("controls.input", 0), DriverValue("controls.input", 1),
                          Divide(0, 1, name="control node 22 (div)")])
    assert float(net(net.initial_state(), values(3.0, 0.5), STEP)[net.outputs_key][2]) == 6.0
    with pytest.raises(ValueError, match=r"control node 22 \(div\).*divides by zero"):
        net(net.initial_state(), values(1.0, 0.0), STEP)


def test_a_flow_sensor_reads_zero_before_the_first_airflow_solve():
    net = ControlNetwork([Sensor(StateReading("air.q", 2))])
    out, _ = advance(net, net.initial_state(), {})
    assert float(out[net.outputs_key][0]) == 0.0
    state = {"air.q": torch.tensor([0.1, 0.2, 0.3], dtype=F64)}
    out, _ = advance(net, {**net.initial_state(), **state}, {})
    assert float(out[net.outputs_key][0]) == pytest.approx(0.3, abs=0)


def test_sensor_offset_scale_and_time_constant_follow_contamx():
    """Flags 1 and 2: x = (c - 1e-4) * 500 lagged with tau = 600 s at dt = 300 s; the
    first evaluation takes x itself. ContamX: 0.450000018, 0.437883079, 0.419348896 from
    zone-one mass fractions 1.000000047e-3, 9.3840974e-4, 8.81557118e-4."""
    net = ControlNetwork([Sensor(StateReading("species.x", (0, 0)), offset=1e-4, scale=500.0,
                                 time_constant=600.0)])
    state = net.initial_state()
    got = []
    for c in (1.000000047e-3, 1.000000047e-3, 9.3840974e-4, 8.81557118e-4):
        state["species.x"] = torch.tensor([[c], [0.0]], dtype=F64)
        out, state = advance(net, state, {})
        got.append(float(out[net.outputs_key][0]))
    assert got[1:] == pytest.approx([0.450000018, 0.437883079, 0.419348896], abs=5e-8)


def test_a_driver_sensor_reads_the_value_the_previous_step_used():
    """A temperature schedule switching 293.15 -> 303.15 K: the sensor sees the new value
    one step later (measured); a temperature the network itself sets is seen likewise."""
    net = ControlNetwork([Sensor(DriverReading("T", 1)), DriverValue("controls.input", 0)],
                         [Actuator(1, "T", (2,))])
    state = net.initial_state()
    seen = []
    for t_zone, t_set in ((293.15, 300.0), (303.15, 301.0), (303.15, 302.0)):
        drivers = {"T": torch.tensor([290.0, t_zone, 0.0], dtype=F64), **values(t_set)}
        out, state = advance(net, state, drivers)
        seen.append(float(out[net.outputs_key][0]))
        assert float(out["T"][2]) == t_set
    assert seen == [293.15, 293.15, 303.15]
    sense_set = ControlNetwork([Sensor(DriverReading("T", 2)), DriverValue("controls.input", 0)],
                               [Actuator(1, "T", (2,))])
    state, seen = sense_set.initial_state(), []
    for t_set in (300.0, 301.0, 302.0):
        drivers = {"T": torch.tensor([290.0, 293.15, 293.15], dtype=F64), **values(t_set)}
        out, state = advance(sense_set, state, drivers)
        seen.append(float(out[sense_set.outputs_key][0]))
    assert seen == [293.15, 300.0, 301.0]


def test_actuators_set_and_add_and_refuse_a_missing_driver():
    net = ControlNetwork([Constant(0.5), Constant(250.0)],
                         [Actuator(0, "flow_multiplier.fan", (1,)),
                          Actuator(1, "thermal.sources", (0,), mode="add")])
    drivers = {"flow_multiplier.fan": torch.ones(3, dtype=F64),
               "thermal.sources": torch.tensor([100.0, 0.0], dtype=F64)}
    out = net(net.initial_state(), drivers, STEP)
    assert out["flow_multiplier.fan"].tolist() == [1.0, 0.5, 1.0]
    assert out["thermal.sources"].tolist() == [350.0, 0.0]
    assert drivers["flow_multiplier.fan"].tolist() == [1.0, 1.0, 1.0]   # not modified
    with pytest.raises(KeyError, match=r"flow_multiplier.fan.*does not have"):
        net(net.initial_state(), {"thermal.sources": torch.zeros(2, dtype=F64)}, STEP)


def test_a_query_outside_a_step_writes_the_stored_outputs():
    """The initial airflow and every query between steps see what the last evaluation
    decided; only a call with a step context evaluates."""
    net = ControlNetwork([DriverValue("controls.input", 0)],
                         [Actuator(0, "flow_multiplier.fan", (0,))])
    drivers = {"flow_multiplier.fan": torch.ones(1, dtype=F64), **values(0.25)}
    _, state = advance(net, net.initial_state(), drivers)
    later = {**drivers, **values(0.75)}
    assert float(net(state, later)["flow_multiplier.fan"][0]) == 0.25
    assert float(net(state, later, STEP)["flow_multiplier.fan"][0]) == 0.75


def test_a_batched_network_keeps_the_batch_in_outputs_memory_and_writes():
    net = ControlNetwork([DriverValue("controls.input", 0),
                          Sensor(StateReading("species.x", (0, 0)), time_constant=600.0)],
                         [Actuator(0, "flow_multiplier.fan", (0,))])
    drivers = {"controls.input": torch.tensor([[0.2], [0.7]], dtype=F64),
               "flow_multiplier.fan": torch.ones(2, dtype=F64)}
    x = torch.tensor([[[1e-3], [0.0]], [[2e-3], [0.0]]], dtype=F64)   # (B, n_interior, K)
    state = {**net.initial_state(), "species.x": x}
    out, state = advance(net, state, drivers)
    assert out[net.outputs_key].shape == (2, 2)
    assert out[net.memory_key].shape == (2, 1)
    assert out["flow_multiplier.fan"].tolist() == [[0.2, 1.0], [0.7, 1.0]]
    out, _ = advance(net, state, drivers)
    assert out[net.outputs_key].shape == (2, 2)


def test_the_network_refuses_bad_positions_and_the_wrong_mode():
    with pytest.raises(ValueError, match=r"block 0 reads blocks \[3\]"):
        ControlNetwork([Identity(3)])
    with pytest.raises(ValueError, match=r"actuator.*block 5"):
        ControlNetwork([Constant(1.0)], [Actuator(5, "T", (0,))])
    with pytest.raises(ValueError, match=r"needs a sampled network"):
        ControlNetwork([Sensor(StateReading("air.q", 0), time_constant=60.0)], sampled=False)


def test_the_network_is_a_model_closure_that_carries_its_state():
    """`Model` accepts it as an integrating closure with its own state keys, and its
    declared inputs and outputs name the keys it reads and writes."""
    net = ControlNetwork([DriverValue("controls.input", 0)],
                         [Actuator(0, "flow_multiplier.fan", (0,))])
    assert net.integrates is True
    assert net.state_keys == ("controls.outputs", "controls.memory", "controls.evaluated")
    assert "controls.input" in net.inputs
    assert "flow_multiplier.fan" in net.outputs
    assert math.isnan(Sensor(DriverReading("T", 0)).initial_memory()[0])


class _Integrator(Block):
    """A test block: memory m with dm/dt = u, output m, a corner where m crosses 1."""

    continuous_only = True
    memory_size = 1

    def __init__(self, a):
        self.inputs = (a,)

    def initial_memory(self):
        return [0.0]

    def step(self, u, memory, ev):
        return memory[..., 0], memory

    def rate(self, u, memory, ev):
        return u[0].unsqueeze(-1).expand(memory.shape)

    def switching(self, u, memory, ev):
        return (memory[..., 0] - 1.0).unsqueeze(-1)


def test_a_continuous_network_is_algebraic_and_advances_its_memory_by_euler():
    """Output at every call from the state; with a step context the memory takes one
    explicit Euler step of `rates` at the step-start state."""
    reading = FunctionReading(lambda state, drivers: 2.0 * state["air.q"][..., 0])
    net = ControlNetwork([Sensor(reading), _Integrator(0)],
                         [Actuator(1, "thermal.sources", None)], sampled=False)
    state = {**net.initial_state(), "air.q": torch.tensor([0.25], dtype=F64)}
    drivers = {"thermal.sources": torch.zeros(2, dtype=F64)}
    assert float(net.evaluate(state, drivers)[0]) == 0.5
    assert net.rates(state, drivers).tolist() == [0.5]
    assert net.switching(state, drivers).tolist() == [-1.0]
    out = net(state, drivers, StepContext(dt=4.0, t=0.0))
    assert out[net.memory_key].tolist() == [2.0]
    assert out["thermal.sources"].tolist() == [2.0, 2.0]
    assert net(state, drivers)[net.memory_key].tolist() == [0.0]      # a query advances nothing
    assert ControlNetwork([Constant(1.0)]).switching({}, {}) is None
    with pytest.raises(ValueError, match=r"integrates in continuous time; it needs sampled"):
        ControlNetwork([Constant(0.0), _Integrator(0)])


def test_the_network_declares_its_input_driver_and_the_state_it_reads():
    net = ControlNetwork([DriverValue("controls.input", 2), Sensor(StateReading("air.dp", 1))])
    assert net.input_specs["controls.input"]["unit"] == "-"
    assert net.initial_drivers()["controls.input"].tolist() == [0.0, 0.0, 0.0]
    assert net.reads_state("air.dp") and not net.reads_state("air.q")
    assert "2 blocks, 0 actuators, sampled" in repr(net)
    plain = ControlNetwork([Constant(1.0)])
    assert plain.input_specs == {} and plain.initial_drivers() == {}


def test_bad_blocks_and_missing_drivers_are_refused_by_name():
    with pytest.raises(ValueError, match="time_constant must be >= 0"):
        Sensor(StateReading("air.q", 0), time_constant=-1.0)
    with pytest.raises(ValueError, match="at least one input"):
        Maximum()
    with pytest.raises(ValueError, match="1 weights for 2 inputs"):
        Sum(0, 1, weights=(1.0,))
    with pytest.raises(ValueError, match="no blocks"):
        ControlNetwork([])
    with pytest.raises(ValueError, match="'add' needs positions"):
        ControlNetwork([Constant(1.0)], [Actuator(0, "thermal.sources", None, mode="add")])
    with pytest.raises(NotImplementedError):
        Block().step([], torch.zeros(0, dtype=F64), None)
    net = ControlNetwork([DriverValue("controls.input", 0, name="set point")])
    with pytest.raises(KeyError, match="set point reads driver 'controls.input'"):
        net(net.initial_state(), {}, STEP)
    sensing = ControlNetwork([Sensor(DriverReading("T", 0))])
    with pytest.raises(KeyError, match="a sensor reads driver 'T'"):
        sensing(sensing.initial_state(), {}, STEP)


@pytest.mark.parametrize("entries", [3, 2])
def test_a_batched_value_fills_every_entry_of_each_batch_member(entries):
    """Batch (2,) against a driver of 3 entries and of 2 (the batch size): a whole-driver
    actuator writes each member's value to all entries; an added value likewise."""
    net = ControlNetwork([DriverValue("controls.input", 0), DriverValue("controls.input", 1)],
                         [Actuator(0, "T", None),
                          Actuator(1, "thermal.sources", tuple(range(entries)), mode="add")])
    drivers = {"controls.input": torch.tensor([[0.2, 10.0], [0.7, 20.0]], dtype=F64),
               "T": torch.ones(entries, dtype=F64),
               "thermal.sources": torch.ones(entries, dtype=F64)}
    out = net(net.initial_state(), drivers, STEP)
    assert out["T"].tolist() == [[0.2] * entries, [0.7] * entries]
    assert out["thermal.sources"].tolist() == [[11.0] * entries, [21.0] * entries]


def test_an_added_value_listed_twice_is_added_twice():
    net = ControlNetwork([Constant(5.0)], [Actuator(0, "thermal.sources", (0, 0), mode="add")])
    out = net(net.initial_state(), {"thermal.sources": torch.tensor([1.0, 0.0], dtype=F64)},
              STEP)
    assert out["thermal.sources"].tolist() == [11.0, 0.0]
    batched = {"thermal.sources": torch.zeros(2, 2, dtype=F64)}
    assert net(net.initial_state(), batched, STEP)["thermal.sources"].tolist() == [
        [10.0, 0.0], [10.0, 0.0]]


def test_a_batched_driver_sensor_keeps_each_members_previous_value():
    """The unbatched initial memory becomes batched on the first step; each member then
    sees its own previous-step temperature."""
    net = ControlNetwork([Sensor(DriverReading("T", 0), time_constant=600.0)])
    state = net.initial_state()
    assert state[net.memory_key].shape == (2,)
    seen = []
    # The one-step delay is intended: the network evaluated at a step's start reads the
    # driver value the previous step used (ContamX's sensors, measured).
    for t in ((290.0, 300.0), (291.0, 302.0), (292.0, 304.0)):
        drivers = {"T": torch.tensor([[t[0]], [t[1]]], dtype=F64)}
        out, state = advance(net, state, drivers)
        assert state[net.memory_key].shape == (2, 2)
        seen.append(out[net.outputs_key][:, 0])
    decay = math.exp(-0.5)
    torch.testing.assert_close(seen[0], torch.tensor([290.0, 300.0], dtype=F64))
    expect = torch.tensor([290.0, 300.0], dtype=F64)     # step 2 reads step 1's values
    torch.testing.assert_close(seen[1], expect + (seen[0] - expect) * decay)
    prev = torch.tensor([291.0, 302.0], dtype=F64)       # step 3 reads step 2's values
    torch.testing.assert_close(seen[2], prev + (seen[1] - prev) * decay)


@pytest.mark.parametrize("batched", [False, True])
def test_two_whole_driver_writes_to_a_missing_driver_keep_one_entry(batched):
    """A driver that does not exist yet gets one entry per batch member; a second actuator
    writing all of it keeps that shape."""
    net = ControlNetwork([DriverValue("controls.input", 0), DriverValue("controls.input", 1)],
                         [Actuator(0, "fan_speed.fan", None), Actuator(1, "fan_speed.fan", None)])
    inputs = torch.tensor([[0.2, 0.4], [0.7, 0.9]] if batched else [0.2, 0.4], dtype=F64)
    out = net(net.initial_state(), {"controls.input": inputs}, STEP)
    if batched:
        assert out["fan_speed.fan"].shape == (2, 1)
        assert out["fan_speed.fan"].tolist() == [[0.4], [0.9]]
    else:
        assert out["fan_speed.fan"].shape == () and float(out["fan_speed.fan"]) == 0.4


def test_setting_a_position_listed_twice_is_refused_by_name():
    with pytest.raises(ValueError, match=r"actuator 'damper' sets positions \[1, 1\].*twice"):
        ControlNetwork([Constant(0.5)],
                       [Actuator(0, "flow_multiplier.fan", (1, 1), name="damper")])


@pytest.mark.parametrize("mode", ["set", "add"])
def test_writing_past_the_end_of_a_driver_is_refused_naming_actuator_key_and_size(mode):
    net = ControlNetwork([Constant(0.5)],
                         [Actuator(0, "flow_multiplier.fan", (0, 2), mode=mode, name="damper")])
    with pytest.raises(ValueError, match=r"damper.*positions \[0, 2\].*'flow_multiplier.fan'"
                                         r".*1 entr"):
        net(net.initial_state(), {"flow_multiplier.fan": torch.ones(1, dtype=F64)}, STEP)


RAMP = [-1.0 + 0.25 * k for k in range(13)] + [2.0 - 0.25 * k for k in range(1, 31)]


def _sequence(net, inputs, position):
    state, got = net.initial_state(), []
    for v in inputs:
        out, state = advance(net, state, values(*v))
        got.append(float(out[net.outputs_key][position]))
    return got


def test_band_switches_follow_contamx():
    """in1 ramps -1 -> 2 -> -1 by 0.25 per step, in2 = 0.5, band 0.4. ContamX: the lower
    band switch is on from in1 = 1.0 (> 0.9) until in1 drops below 0.5 (still on at 0.5);
    the upper one is on until in1 rises above 0.5 and again from in1 = 0 (< 0.1)."""
    net = ControlNetwork([DriverValue("controls.input", 0), DriverValue("controls.input", 1),
                          LowerBandSwitch(0, 1, band=0.4), UpperBandSwitch(0, 1, band=0.4)])
    rows = [(v, 0.5) for v in RAMP]
    lower = _sequence(net, rows, 2)
    upper = _sequence(net, rows, 3)
    assert lower == [0.0] * 8 + [1.0] * 11 + [0.0] * 24
    assert upper == [1.0] * 7 + [0.0] * 13 + [1.0] * 23
    equal = _sequence(ControlNetwork([DriverValue("controls.input", 0),
                                      DriverValue("controls.input", 1),
                                      LowerBandSwitch(0, 1, band=0.0),
                                      UpperBandSwitch(0, 1, band=0.0)]), [(0.5, 0.5)], 2)
    assert equal == [0.0]


HYS_IN = ([0.1 * k for k in range(11)] + [0.92, 0.84, 0.76, 0.68, 0.6, 0.52, 0.44, 0.36,
          0.28, 0.2, 0.27, 0.34, 0.41, 0.48, 0.55, 0.62, 0.69, 0.76, 0.83, 0.9, 0.81, 0.72,
          0.63, 0.54, 0.45, 0.36, 0.27, 0.18, 0.09, 0.0])
HYS_OUT = [0, 0, 0, 0, 0.14286, 0.28571, 0.42857, 0.57143, 0.71429, 0.85714, 1, 1, 1, 1,
           0.97143, 0.85714, 0.74286, 0.62857, 0.51429, 0.4, 0.28571, 0.28571, 0.28571,
           0.28571, 0.28571, 0.35714, 0.45714, 0.55714, 0.65714, 0.75714, 0.85714, 0.85714,
           0.85714, 0.85714, 0.77143, 0.64286, 0.51429, 0.38571, 0.25714, 0.12857, 0]


def test_hysteresis_follows_contamx_over_a_ramp_up_down_and_up():
    """Slack 0.3, slope 1.42857 (the file's field), initial 0.1: ContamX's 41 outputs."""
    net = ControlNetwork([DriverValue("controls.input", 0),
                          Hysteresis(0, slack=0.3, slope=1.42857, initial=0.1)])
    got = _sequence(net, [(v,) for v in HYS_IN], 1)
    assert got == pytest.approx(HYS_OUT, abs=1e-5)


@pytest.mark.parametrize(("u", "slope", "initial", "expected"), [
    (0.5, 1.42857, 0.4, 0.571428), (0.5, 1.42857, 0.9, 0.714285), (0.5, 2.0, 0.4, 0.8),
    (1.8, 1.42857, 0.0, 0.999999), (-0.5, 1.42857, 0.5, 0.0),
])
def test_hysteresis_starts_from_the_initial_signal_times_the_slope(u, slope, initial,
                                                                   expected):
    """ContamX: the previous output starts at oldsig * slope; the input is clipped to
    0..1 first; the slope is the file's field, not 1/(1 - slack)."""
    net = ControlNetwork([Constant(u), Hysteresis(0, slack=0.3, slope=slope, initial=initial)])
    assert _sequence(net, [()], 1) == pytest.approx([expected], abs=1e-6)


def test_proportional_control_is_clipped_to_zero_and_one():
    net = ControlNetwork([DriverValue("controls.input", 0), ProportionalControl(0, gain=0.5)])
    assert _sequence(net, [(0.2,), (3.0,), (-1.0,)], 1) == pytest.approx([0.1, 1.0, 0.0])


def test_incremental_pi_follows_contamx_from_its_initial_values():
    """Kp 0.5, Ki 1e-4 1/s, dt 300 s, oldsig 0.1, olderr 0.05; the error ContamX sensed at
    each step (float32). The t = 0 evaluation is itself an update."""
    net = ControlNetwork([DriverValue("controls.input", 0),
                          IncrementalPI(0, kp=0.5, ki=1e-4, initial_output=0.1,
                                        initial_error=0.05)])
    errors = [0.199999988, 0.199999988, 0.201195657, 0.201829255, 0.201929986]
    got = _sequence(net, [(e,) for e in errors], 1)
    expected = [0.178749993, 0.184749991, 0.191365764, 0.197727934, 0.203834683]
    assert got == pytest.approx(expected, abs=5e-8)


def test_incremental_pi_winds_up_no_further_than_its_clip():
    """Kp 0.5, Ki 0.05: saturated at 1 for six steps; the stored output is the clipped one,
    so it leaves saturation at the first negative increment (ContamX: 0.351799011)."""
    net = ControlNetwork([DriverValue("controls.input", 0),
                          IncrementalPI(0, kp=0.5, ki=0.05, initial_output=0.1,
                                        initial_error=0.05)])
    errors = [0.199999988, 0.199999988, 0.138409734, 0.081557095, 0.029077709,
              -0.019364774, -0.064080954, -0.0689044, -0.052294433]
    got = _sequence(net, [(e,) for e in errors], 1)
    assert got == pytest.approx([1, 1, 1, 1, 1, 1, 0.351799011, 0, 0], abs=2e-7)


def test_memory_blocks_need_a_sampled_network():
    for block in (Hysteresis(0, slack=0.3), IncrementalPI(0, kp=1.0, ki=0.0),
                  LowerBandSwitch(0, 0, band=0.1)):
        with pytest.raises(ValueError, match="needs a sampled network"):
            ControlNetwork([Constant(0.0), block], sampled=False)


def _sequence_rows(net, inputs):
    state, rows = net.initial_state(), []
    for v in inputs:
        out, state = advance(net, state, values(*v))
        rows.append(out[net.outputs_key].tolist())
    return rows


def test_the_band_switches_hysteresis_and_controllers_keep_a_batch_member_by_member():
    """Two members whose inputs cross the thresholds at different steps: each member's
    outputs are those of its own unbatched run, and the memory is (2, m)."""
    def network():
        return ControlNetwork([
            DriverValue("controls.input", 0), DriverValue("controls.input", 1),
            LowerBandSwitch(0, 1, band=0.4), UpperBandSwitch(0, 1, band=0.4),
            Hysteresis(0, slack=0.3, slope=1.42857, initial=0.1),
            ProportionalControl(0, gain=0.5),
            IncrementalPI(0, kp=0.5, ki=1e-3, initial_output=0.1, initial_error=0.05)])

    first = [-0.5, 0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, -0.5]
    second = [1.5, 1.0, 0.5, 0.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    alone = [_sequence_rows(network(), [(v, 0.5) for v in series])
             for series in (first, second)]
    net = network()
    state = net.initial_state()
    for k, (a, b) in enumerate(zip(first, second, strict=True)):
        out, state = advance(net, state, {"controls.input": torch.tensor(
            [[a, 0.5], [b, 0.5]], dtype=F64)})
        assert out[net.memory_key].shape == (2, net.memory_size) == (2, 5)
        assert out[net.outputs_key].tolist() == [alone[0][k], alone[1][k]]
    for column in (2, 3, 4, 6):   # each block's outputs differ between the members
        assert [r[column] for r in alone[0]] != [r[column] for r in alone[1]]


@pytest.mark.parametrize("slack", [-0.1, 1.0, 1.5])
def test_hysteresis_refuses_a_slack_outside_zero_to_one(slack):
    with pytest.raises(ValueError, match=r"Hysteresis 'h': slack must be in \[0, 1\)"):
        Hysteresis(0, slack=slack, name="h")


def test_writing_positions_of_a_single_valued_driver_is_refused_by_name():
    net = ControlNetwork([Constant(0.5)],
                         [Actuator(0, "flow_multiplier.fan", (0,), name="damper")])
    with pytest.raises(ValueError, match=r"damper.*positions \[0\].*'flow_multiplier.fan'"
                                         r".*a single value"):
        net(net.initial_state(), {"flow_multiplier.fan": torch.tensor(1.0, dtype=F64)}, STEP)


PROBE = Path(__file__).resolve().parents[2] / "data" / "modelica" / "controls" / "LimPIDProbe.csv"


def _probe():
    lines = [ln for ln in PROBE.read_text().splitlines() if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def test_limited_pid_matches_openmodelica_on_a_heated_room():
    """MSL 4.1.0's LimPID (PI with anti-windup, limits +-1) heating a first-order room,
    integrated here to 1e-12 from the network's own outputs and rates: T and the integral
    state follow OpenModelica's run to 1e-8 through saturation and its end at 78 s."""
    integrate = pytest.importorskip("scipy.integrate")
    ref = _probe()
    net = ControlNetwork([DriverValue("controls.input", 0), Sensor(StateReading("T", 0)),
                          LimitedPID(0, 1, gain=5.0, y_max=1.0, y_min=-1.0,
                                     integral_time=60.0, windup_ratio=0.9,
                                     initial_integral=0.1)], sampled=False)
    drivers = {"controls.input": torch.tensor([293.15], dtype=F64)}

    def rhs(t, z):
        state = {"T": torch.tensor([z[0]], dtype=F64),
                 net.memory_key: torch.tensor([z[1]], dtype=F64)}
        y = float(net.evaluate(state, drivers)[2])
        return [(3000.0 * y - 100.0 * (z[0] - 273.15)) / 75000.0,
                float(net.rates(state, drivers)[0])]

    sol = integrate.solve_ivp(rhs, (0.0, 1200.0), [292.0, 0.1], t_eval=ref["time"],
                              rtol=1e-12, atol=1e-12, method="LSODA")
    assert max(abs(a - b) for a, b in zip(sol.y[0], ref["T"], strict=True)) < 1e-8
    assert max(abs(a - b) for a, b in zip(sol.y[1], ref["con.I.y"], strict=True)) < 1e-8


def test_a_p_only_limited_pid_is_algebraic_and_reports_its_limits():
    """MBL's examples: P, k = 5, limits +-1 -> y = clamp(5 (T_set - T), -1, 1); the
    switching values cross zero at the limits."""
    net = ControlNetwork([Constant(293.15), Sensor(StateReading("T", 0)),
                          LimitedPID(0, 1, gain=5.0, y_max=1.0, y_min=-1.0)], sampled=False)
    assert net.memory_size == 0
    for T, y in ((293.0, 0.75), (292.0, 1.0), (294.0, -1.0)):
        state = {"T": torch.tensor([T], dtype=F64)}
        assert float(net.evaluate(state, {})[2]) == pytest.approx(y)
    s = net.switching({"T": torch.tensor([292.95], dtype=F64)}, {})
    assert s.tolist() == pytest.approx([0.0, 1.0], abs=1e-9)   # (v - 1)/2, (v + 1)/2 at v = 1


def test_a_batched_limited_pid_keeps_each_members_integral():
    """Two set points: one member saturates high, the other low; the integral of each
    advances by its own anti-windup rate (one Euler step of 1 s)."""
    net = ControlNetwork([DriverValue("controls.input", 0), Sensor(StateReading("T", 0)),
                          LimitedPID(0, 1, gain=5.0, y_max=1.0, integral_time=60.0,
                                     initial_integral=0.1)], sampled=False)
    drivers = {"controls.input": torch.tensor([[293.15], [292.0]], dtype=F64)}
    state = {**net.initial_state(), "T": torch.tensor([292.9], dtype=F64)}
    out = net(state, drivers, StepContext(dt=1.0, t=0.0))
    # v = 5 (0.25 + 0.1) = 1.75 -> y = 1; v = 5 (-0.9 + 0.1) = -4 -> y = -1; k Ni = 4.5
    rate = [(0.25 + (1.0 - 1.75) / 4.5) / 60.0, (-0.9 + (-1.0 + 4.0) / 4.5) / 60.0]
    assert out[net.memory_key].shape == (2, 1)
    assert out[net.memory_key][:, 0].tolist() == pytest.approx([0.1 + r for r in rate])
    xi = torch.tensor([0.1 + r for r in rate], dtype=F64)
    v = 5.0 * (torch.tensor([0.25, -0.9], dtype=F64) + xi)
    assert out[net.outputs_key][:, 2].tolist() == pytest.approx(v.clamp(-1, 1).tolist())


def test_limited_pid_refuses_bad_parameters_and_a_sampled_network():
    with pytest.raises(ValueError, match="y_max"):
        LimitedPID(0, 1, gain=1.0, y_max=-1.0, y_min=1.0)
    with pytest.raises(ValueError, match="gain"):
        LimitedPID(0, 1, gain=0.0, y_max=1.0)
    with pytest.raises(ValueError, match="integral_time"):
        LimitedPID(0, 1, gain=1.0, y_max=1.0, integral_time=0.0)
    with pytest.raises(ValueError, match="windup_ratio"):
        LimitedPID(0, 1, gain=1.0, y_max=1.0, integral_time=1.0, windup_ratio=0.0)
    with pytest.raises(ValueError, match="setpoint_weight must be >= 0"):
        LimitedPID(0, 1, gain=1.0, y_max=1.0, setpoint_weight=-0.5)
    with pytest.raises(ValueError, match="sampled=False"):
        ControlNetwork([Constant(0.0), Constant(0.0), LimitedPID(0, 1, gain=1.0, y_max=1.0)])


def _two_rooms(**kwargs):
    """The application page's Python model, smaller: rooms A and B in series between
    two orifices to the ambient, with one species unless `species=` says otherwise."""
    from noodl.apps.building_physics import (
        Zone,
        add_zone,
        build_model,
        orifice_elements_from_edges,
    )
    from noodl.drives import Stack
    from noodl.topology import Network

    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=293.15))
    add_zone(net, Zone("B", volume=60.0, T0=293.15))
    net.add_edge("ambient", "A", kind="airpath", z_path=0.5, Cd=0.6, area=0.05)
    net.add_edge("A", "B", kind="airpath", z_path=1.0, Cd=0.6, area=0.05)
    net.add_edge("B", "ambient", kind="airpath", z_path=1.5, Cd=0.6, area=0.05)
    el = orifice_elements_from_edges(net, "airpath")
    kwargs.setdefault("species", ("CO2",))
    return build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                       return_inputs=True, **kwargs)


def _interior(model, layer, node):
    return model.transport[layer].interior_idx.tolist().index(model.net.node_index(node))


def _sensing(*readings, **kwargs):
    """`_two_rooms` with a network of one sensor per reading."""
    network = ControlNetwork([Sensor(r) for r in readings])
    model, state, drivers = _two_rooms(controls=network, **kwargs)
    return network, model, state, drivers


def test_sensors_by_name_resolve_against_the_model_they_are_built_into():
    """One species is stored reduced, (n_interior,): the reading follows the layer's own
    layout. A boundary node's temperature is its boundary driver, read one step late."""
    network, model, state, drivers = _sensing(
        node_mass_fraction("B", "CO2"), node_temperature("A"), node_pressure("A"),
        edge_flow("airpath", 1), edge_pressure_difference("airpath", 1),
        node_temperature("ambient"))
    co2, temperature, pressure, flow, dp, outdoor = (b.reading.resolved
                                                     for b in network.blocks)
    x = torch.zeros(2, dtype=F64)
    x[_interior(model, "species", "B")] = 6e-4
    assert float(co2({**state, "species.x": x}, drivers)) == 6e-4
    assert isinstance(temperature, StateReading)
    assert (temperature.key, temperature.index) == ("thermal.x",
                                                    (_interior(model, "thermal", "A"),))
    assert (pressure.key, pressure.index) == ("air.phi", (model.net.node_index("A"),))
    column = model.potential["air"].kind_slice("airpath").start + 1
    assert (flow.key, flow.index) == ("air.q", (column,))
    assert (dp.key, dp.index) == ("air.dp", (column,))
    assert isinstance(outdoor, DriverReading)
    assert (outdoor.key, outdoor.index) == ("thermal.x_boundary", (0,))
    assert network.memory_size == 1 and network.blocks[5].memory_size == 1
    assert "thermal.x_boundary" in network.inputs
    assert state[network.memory_key].shape == (1,)
    assert network.reads_state("species.x") and network.reads_state("air.dp")
    assert not network.reads_state("thermal.x_boundary")


@pytest.mark.parametrize(("reading", "error", "match"), [
    (node_mass_fraction("B", "H2O"), KeyError, "no species 'H2O'"),
    (node_mass_fraction("B", 2), IndexError, "1 species, not position 2"),
    (node_mass_fraction("ambient"), KeyError, "'ambient' is not an interior node"),
    (edge_flow("airpath", 3), IndexError, "3 edges, not position 3"),
    (edge_pressure_difference("door"), KeyError, "no element of kind 'door'"),
])
def test_a_sensor_that_names_what_the_model_lacks_is_refused_when_the_model_is_built(
        reading, error, match):
    with pytest.raises(error, match=match):
        _sensing(reading)


def test_an_unresolved_named_reading_is_refused_naming_the_sensor():
    net = ControlNetwork([Sensor(node_pressure("A"), name="lobby pressure")])
    with pytest.raises(ValueError, match=r"lobby pressure reads node_pressure\('A'\), which "
                                         r"is not resolved"):
        net(net.initial_state(), {}, STEP)
    with pytest.raises(ValueError, match="not resolved"):
        node_pressure("A")({}, {})


def test_a_sensor_finds_its_row_and_column_whatever_the_layer_order():
    """Zone C is reached only by a kind the heat and species layers do not carry, and the
    zones and elements are not in a naive order: the rows and the column are the model's
    own, not node_index - 1 or the airpath's position."""
    from noodl.apps.building_physics import (
        Zone,
        add_zone,
        build_model,
        orifice_elements_from_edges,
    )
    from noodl.drives import Stack
    from noodl.topology import Network

    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    for name in ("C", "B", "A"):
        add_zone(net, Zone(name, volume=60.0, T0=293.15))
    net.add_edge("ambient", "C", kind="crack", z_path=0.5, Cd=0.6, area=0.01)
    net.add_edge("ambient", "A", kind="airpath", z_path=0.5, Cd=0.6, area=0.05)
    net.add_edge("A", "B", kind="airpath", z_path=1.0, Cd=0.6, area=0.05)
    net.add_edge("B", "ambient", kind="airpath", z_path=1.5, Cd=0.6, area=0.05)
    elements = [orifice_elements_from_edges(net, "crack"),
                orifice_elements_from_edges(net, "airpath")]
    drives = [Stack.from_network(net, "crack"), Stack.from_network(net, "airpath")]
    network = ControlNetwork([Sensor(node_mass_fraction("A")), Sensor(node_temperature("A")),
                              Sensor(edge_flow("airpath", 1))])
    model, state, drivers = build_model(net, air_elements=elements, drives=drives,
                                        flow_kinds=("airpath",), species=1,
                                        controls=network, return_inputs=True)
    assert model.transport["species"].interior_idx.tolist() == [2, 3]   # B, A; not C
    state = {**state, "species.x": torch.tensor([1e-4, 3e-4], dtype=F64),
             "thermal.x": torch.tensor([290.0, 300.0], dtype=F64),
             "air.q": torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=F64)}
    got = network.evaluate(state, drivers).tolist()
    assert got == [3e-4, 300.0, 20.0]     # A's rows (1, not 2), airpath edge 1 (column 2)


def test_a_mass_fraction_sensor_reads_two_species_batched_or_not():
    """Two species are stored (n_interior, K); a batch goes in front."""
    network, model, state, drivers = _sensing(node_mass_fraction("B", "H2O"),
                                              node_mass_fraction("B", 0),
                                              species=("CO2", "H2O"))
    water, co2 = (b.reading for b in network.blocks)
    x = torch.zeros(2, 2, dtype=F64)
    x[_interior(model, "species", "B")] = torch.tensor([6e-4, 9e-3], dtype=F64)
    assert float(water({**state, "species.x": x}, drivers)) == 9e-3
    assert float(co2({**state, "species.x": x}, drivers)) == 6e-4
    batched = torch.stack([x, 2.0 * x])
    assert water({**state, "species.x": batched}, drivers).tolist() == [9e-3, 1.8e-2]


def test_a_temperature_sensor_needs_a_thermal_layer_or_a_prescribed_temperature():
    """Without a thermal layer and a "T" driver the sensor is refused; a .prj model declares
    "T", and the sensor reads it (one step late)."""
    with pytest.raises(ValueError, match=r"no thermal layer and declares no prescribed "
                                         r"temperature driver 'T'"):
        _sensing(node_temperature("B"), thermal=False)
    from noodl.apps.building_physics import project_to_model, read_prj

    project = read_prj(Path(__file__).resolve().parents[2] / "data" / "contam"
                       / "valThreeZonesWthCtm-UseApi.prj")
    network = ControlNetwork([Sensor(node_temperature("two"))])
    project.closures.append(network)
    model, state, drivers = project_to_model(project)
    reading = network.blocks[0].reading.resolved
    assert isinstance(reading, DriverReading)
    assert (reading.key, reading.index) == ("T", (model.net.node_index("two"),))
    assert network.inputs == ("T",) and network.memory_size == 1


def test_build_model_puts_the_network_first_and_seeds_its_state():
    """A network adding a heat flow to room A from a value written each step."""
    network = ControlNetwork([DriverValue("controls.input", 0)],
                             [Actuator(0, "thermal.sources", (1,), mode="add")])
    model, state, drivers = _two_rooms(controls=network)
    assert model.closures[0] is network
    assert torch.isnan(state["controls.outputs"]).all()          # not evaluated yet
    assert drivers["controls.input"].tolist() == [0.0]
    a = model.net.node_index("A")
    assert a == 1   # node order: ambient, A, B
    sources = torch.zeros(len(model.net.nodes), dtype=F64)
    new = model.step(state, {**drivers, "thermal.sources": sources,
                             "controls.input": torch.tensor([500.0], dtype=F64)},
                     60.0, t=0.0)
    assert new["controls.outputs"].tolist() == [500.0]
    unheated = model.step(state, {**drivers, "thermal.sources": sources}, 60.0, t=0.0)
    i = _interior(model, "thermal", "A")
    assert float(new["thermal.x"][i]) > float(unheated["thermal.x"][i])


def test_a_network_writes_nothing_before_its_first_evaluation():
    """The initial state's outputs are NaN: a query writes nothing (a zone temperature
    control never writes 0 K), and the first evaluation reads them as 0."""
    net = ControlNetwork([Identity(1), DriverValue("controls.input", 0)],
                         [Actuator(1, "T", (1,))])
    drivers = {"T": torch.tensor([280.0, 293.15], dtype=F64), **values(303.15)}
    queried = net(net.initial_state(), drivers)
    assert "T" not in queried and torch.isnan(queried[net.outputs_key]).all()
    out, state = advance(net, net.initial_state(), drivers)
    assert out[net.outputs_key].tolist() == [0.0, 303.15]           # the later block: 0
    assert out["T"].tolist() == [280.0, 303.15]
    assert net(state, drivers)["T"].tolist() == [280.0, 303.15]


def test_a_held_network_writes_its_outputs_and_records_the_drivers_the_step_uses():
    """`held` stops a step from evaluating the network again; a driver sensor still
    records the value the step used."""
    net = ControlNetwork([Sensor(DriverReading("T", 0)), DriverValue("controls.input", 0)])
    drivers = {"T": torch.tensor([280.0], dtype=F64), **values(0.5)}
    _, state = advance(net, net.initial_state(), drivers)
    later = {"T": torch.tensor([290.0], dtype=F64), **values(0.7)}
    with net.held():
        out, held = advance(net, state, later)
    assert out[net.outputs_key].tolist() == [280.0, 0.5]
    assert held[net.memory_key].tolist() == [290.0]
    assert advance(net, held, later)[0][net.outputs_key].tolist() == [290.0, 0.7]


def test_a_network_resolved_for_one_model_is_refused_by_another():
    network = ControlNetwork([Sensor(node_pressure("A"))])
    _two_rooms(controls=network)
    with pytest.raises(ValueError, match=r"'controls' is already resolved for another model; "
                                         r"one network per model"):
        _two_rooms(controls=network)


def test_a_controlled_project_is_seeded_and_checks_without_warnings():
    """A zone temperature control in a project: the state carries the network's outputs
    and memory, the capacities are finite, and `Model.check` warns of nothing (the network
    writes nothing before `simulate` evaluates it)."""
    from noodl.apps.building_physics import project_to_model, read_prj
    from noodl.apps.building_physics.simulation import simulate

    project = read_prj(Path(__file__).resolve().parents[2] / "data" / "contam"
                       / "valThreeZonesWthCtm-UseApi.prj")
    network = ControlNetwork([Constant(303.15)], [Actuator(0, "T", (1,))])
    project.closures.append(network)
    model, state, drivers = project_to_model(project)
    assert network.state_keys[0] in state and network.state_keys[1] in state
    assert torch.isfinite(state["species.capacity"]).all()
    report = model.check(state, drivers)
    assert not report.errors and not report.warnings, str(report)
    out = simulate(model, state, drivers, [0.0, 300.0], atol=1e-14, rtol=1e-14)
    assert out["controls.outputs"][:, 0].tolist() == [303.15, 303.15]


def test_a_constant_later_in_the_order_is_read_before_its_first_evaluation():
    """ContamX presets its constant nodes: a node reading a constant later in the sequence
    sees its value at the first evaluation (measured: set 1 / later set 2 gives 0.5 from
    t = 0), whereas any other later block reads 0 there."""
    net = ControlNetwork([Constant(1.0), Divide(0, 2), Constant(2.0),
                          Identity(4), DriverValue("controls.input", 0)])
    out, state = advance(net, net.initial_state(), values(0.7))
    assert out[net.outputs_key].tolist() == [1.0, 0.5, 2.0, 0.0, 0.7]
    assert advance(net, state, values(0.7))[0][net.outputs_key].tolist()[3] == 0.7


def test_the_evaluated_flag_not_the_outputs_decides_what_is_written():
    """A state whose outputs are numbers but whose flag says "not evaluated" writes
    nothing; a batch whose second member is not evaluated yet keeps that member's driver."""
    net = ControlNetwork([DriverValue("controls.input", 0)], [Actuator(0, "T", (1,))])
    drivers = {"T": torch.tensor([280.0, 293.15], dtype=F64)}
    unevaluated = {**net.initial_state(),
                   net.outputs_key: torch.tensor([303.15], dtype=F64)}
    assert "T" not in net(unevaluated, drivers)
    batch = {net.outputs_key: torch.tensor([[303.15], [math.nan]], dtype=F64),
             net.memory_key: torch.zeros(2, 0, dtype=F64),
             net.evaluated_key: torch.tensor([1.0, 0.0], dtype=F64)}
    written = net(batch, drivers)["T"]
    assert written.tolist() == [[280.0, 303.15], [280.0, 293.15]]
    whole = ControlNetwork([DriverValue("controls.input", 0)], [Actuator(0, "P_ref")])
    with pytest.raises(ValueError, match=r"actuator 'P_ref'.*some members are not "
                                         r"evaluated"):
        whole({**batch, whole.outputs_key: batch[net.outputs_key]}, {})


def test_a_nested_hold_keeps_the_outer_one():
    net = ControlNetwork([DriverValue("controls.input", 0)])
    _, state = advance(net, net.initial_state(), values(0.5))
    with net.held():
        with net.held():
            pass
        out, _ = advance(net, state, values(0.9))
    assert out[net.outputs_key].tolist() == [0.5]           # still held: not evaluated again
    assert advance(net, state, values(0.9))[0][net.outputs_key].tolist() == [0.9]
