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
    Identity,
    Maximum,
    Minimum,
    Or,
    Positive,
    Product,
    Sensor,
    Shortfall,
    StateReading,
    Sum,
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
