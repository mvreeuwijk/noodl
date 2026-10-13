"""Control blocks and networks (`noodl.apps.building_physics.controls`).

The expected values are ContamX 3.4.1.7's (probes `d3`, `d4`, `d8` of the controls work):
float32 in ContamX, so they are compared to 5e-8 where ContamX's own rounding shows.
"""

import math

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
)
from noodl.model import StepContext

F64 = torch.float64
STEP = StepContext(dt=300.0, t=0.0)


def advance(network, state, drivers, ctx=STEP):
    """One evaluation; the state carries the network's outputs and memory forward."""
    out = network(state, drivers, ctx)
    new = {**state, network.outputs_key: out[network.outputs_key],
           network.memory_key: out[network.memory_key]}
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
    assert net.state_keys == ("controls.outputs", "controls.memory")
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
