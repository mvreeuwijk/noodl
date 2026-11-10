"""Pump energy and cost (`noodl.apps.water.energy`, `water_run(...).energy`): EPANET 2.2's
`addenergy`, `getenergy` and `saveenergy`. Against EPANET itself: `tests/verification/
test_water_parity.py` (energy section)."""

import dataclasses

import pytest
import torch

from noodl.apps.water.controls import CLOSED, OPEN, Control
from noodl.apps.water.energy import KW_PER_HP, interpolate_curves
from noodl.apps.water.network import (
    EnergyOptions,
    Junction,
    Pump,
    Reservoir,
    WaterNetwork,
    WaterOptions,
    WaterPipe,
    build_model,
)
from noodl.apps.water.simulate import water_run

F64 = torch.float64
RHO_G = 1000.0 * 9.80665          # N/m3


def _constant_pump(**changes):
    """R (20 m) -> pump of 5 kW -> J, withdrawing 0.01 m3/s; two hours in hourly steps. The
    network is an LPS one, so that the pump's power is EPANET's (0.7457 kW per hp, rounded
    cfs per L/s): a hand-built network takes the exact factors."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01),), reservoirs=(Reservoir("R", 20.0),),
        pumps=(Pump("PU", "R", "J", None, power=5.0),), duration=7200.0,
        options=WaterOptions(flow_units="LPS"),
        energy=EnergyOptions(global_efficiency=75.0, global_price=0.1, demand_charge=3.0))
    return dataclasses.replace(net, **changes)


def _run(net, **kwargs):
    model, _, _ = build_model(net)
    return model, water_run(model, **kwargs)


def test_energy_of_a_constant_pump_by_hand():
    """One pump at constant power, constant flow and head: EPANET's energy is the solved
    head gain times the flow (`getenergy`, `dh q SpGrav / 8.814 / e KWperHP`), which for a
    POWER pump is its power over the efficiency (`input2.c`, `updatepumpparams`:
    `R = -8.814 P`), and physically `rho g q h / eff` (EPANET's 62.4 lbf/ft3 is 0.04 % off
    9806.65 N/m3)."""
    model, out = _run(_constant_pump())
    energy = out.energy
    assert energy.pump_names == ["PU"]
    q = 0.01
    nodes = model.node_names
    h = float(out.head[0, nodes.index("J")] - out.head[0, nodes.index("R")])
    assert h > 0
    kw = 5.0 / 0.75
    assert float(energy.average_power) == pytest.approx(kw, rel=1e-10)
    assert float(energy.peak_power) == pytest.approx(kw, rel=1e-10)
    assert float(energy.average_power) == pytest.approx(RHO_G * q * h / 1000.0 / 0.75,
                                                        rel=1e-3)
    assert float(energy.utilisation) == pytest.approx(100.0, rel=1e-12)
    assert float(energy.efficiency) == pytest.approx(75.0, rel=1e-12)
    assert float(energy.energy_per_volume) == pytest.approx(kw / (q * 3600.0), rel=1e-10)
    # kWh over the two hours, as the product of the average power and the hours
    assert float(energy.average_power) * 2.0 == pytest.approx(2 * kw, rel=1e-10)
    assert float(energy.cost) == pytest.approx(0.1 * kw * 24.0, rel=1e-10)
    assert float(energy.demand_charge) == pytest.approx(3.0 * kw, rel=1e-10)
    assert KW_PER_HP == 0.7457


def test_a_network_without_pumps_has_no_energy():
    net = WaterNetwork(junctions=(Junction("J", 0.0, 0.01),),
                       reservoirs=(Reservoir("R", 20.0),),
                       pipes=(WaterPipe("P", "R", "J", 100.0, 0.2, 130.0),))
    _, out = _run(net)
    assert out.energy is None


def test_price_patterns_with_pattern_start():
    """The price is the pump's own (when positive) else the global one, times the pump's
    price pattern else the global price pattern, at the period `(t + Pstart) / Pstep`
    (`addenergy`). Here Pstart is one period: the three hourly steps are periods 1, 2, 3."""
    patterns = {"G": (1.0, 2.0, 3.0), "P": (10.0, 20.0, 30.0, 40.0)}
    kw = 5.0 / 0.75
    base = _constant_pump(duration=10800.0, pattern_start=3600.0, patterns=patterns,
                          energy=EnergyOptions(75.0, 0.1, "G", 0.0))
    _, out = _run(base)
    # factors 2, 3, 1 of the global pattern over three hours, cost per day = sum * 24 / 3
    assert float(out.energy.cost) == pytest.approx(0.1 * kw * (2 + 3 + 1) * 24.0 / 3.0,
                                                   rel=1e-10)
    own = dataclasses.replace(base, pumps=(Pump("PU", "R", "J", None, power=5.0, price=0.5,
                                                price_pattern="P"),))
    _, out = _run(own)       # the pump's price and pattern replace the global ones: 20, 30, 40
    assert float(out.energy.cost) == pytest.approx(0.5 * kw * (20 + 30 + 40) * 24.0 / 3.0,
                                                   rel=1e-10)
    free = dataclasses.replace(base, pumps=(Pump("PU", "R", "J", None, power=5.0, price=0.0),))
    _, out = _run(free)      # a zero pump price is no price: the global one applies
    assert float(out.energy.cost) == pytest.approx(0.1 * kw * 6 * 24.0 / 3.0, rel=1e-10)


def test_energy_counts_every_step_to_the_end_of_an_unaligned_duration():
    """EPANET adds the energy of every step that starts before the duration, the last one
    whole (`nexthyd`: `addenergy(hydstep)`, the step not shortened to the duration): at
    5400 s the second step counts one hour, so the pump is on line 2 h of 1.5 h, and the
    cost per day is that of 2 h scaled to 1.5 h. The reported times are unchanged."""
    model, out = _run(_constant_pump(duration=5400.0))
    assert out.time.tolist() == [0.0, 3600.0]
    kw = 5.0 / 0.75
    assert float(out.energy.utilisation) == pytest.approx(200.0 / 1.5, rel=1e-12)
    assert float(out.energy.cost) == pytest.approx(0.1 * kw * 2.0 * 24.0 / 1.5, rel=1e-10)
    assert float(out.energy.average_power) == pytest.approx(kw, rel=1e-10)


def test_energy_of_a_zero_duration_run_is_one_hour():
    """EPANET's `addenergy` takes an hour for a run of duration 0 and `saveenergy` then
    reports the cost per day of that hour (times 24) and the unscaled sums."""
    _, out = _run(_constant_pump(duration=0.0))
    kw = 5.0 / 0.75
    assert float(out.energy.utilisation) == pytest.approx(100.0, rel=1e-12)
    assert float(out.energy.cost) == pytest.approx(0.1 * kw * 24.0, rel=1e-10)
    assert float(out.energy.average_power) == pytest.approx(kw, rel=1e-10)


def test_a_pump_closed_by_a_control_draws_no_energy():
    """A pump closed by a control is at speed 0, as in EPANET (a pump closed otherwise keeps
    its speed): the energy reads the status. The pump here is closed at 3600 s of a 7200 s
    run, with a second pump keeping the junction supplied; it is on line half the time."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01),), reservoirs=(Reservoir("R", 20.0),),
        pumps=(Pump("PU", "R", "J", None, power=5.0), Pump("PV", "R", "J", None, power=5.0)),
        controls=(Control("PV", CLOSED, None, "time", value=3600.0),), duration=7200.0)
    _, out = _run(net)
    on = out.energy.utilisation
    assert on[0] == pytest.approx(100.0, rel=1e-12)
    assert on[1] == pytest.approx(50.0, rel=1e-12)
    assert float(out.setting[1, 1]) == 0.0 and float(out.setting[0, 1]) == 1.0
    # the pump closed by the control draws no energy: its average is over the time on line
    assert float(out.energy.average_power[1]) == pytest.approx(
        float(out.energy.average_power[0]), rel=1e-3)
    assert float(out.energy.peak_power[1]) > 0


def test_a_pump_opened_by_a_control_reads_as_open_and_draws_energy():
    """A pump closed at the start and opened by a control at 3600 s is OPEN in the results
    from then on (its state was read with the drivers' stale status, CLOSED, before), and
    draws energy for the second hour only."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01),), reservoirs=(Reservoir("R", 20.0),),
        pumps=(Pump("PU", "R", "J", None, power=5.0), Pump("PV", "R", "J", None, power=5.0)),
        controls=(Control("PV", OPEN, None, "time", value=3600.0),), duration=7200.0,
        initial_status={"PV": "CLOSED"}, options=WaterOptions(flow_units="LPS"))
    _, out = _run(net)
    assert out.status[:, 1].tolist() == [CLOSED, OPEN, OPEN]
    assert float(out.energy.utilisation[1]) == pytest.approx(50.0, rel=1e-12)


def test_a_pump_shut_by_the_solver_at_shut_off_draws_no_energy():
    """The pump pushes against a higher reservoir than its shut-off head: the solve shuts
    it (status CLOSED) and no energy is used."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 5.0), Reservoir("R2", 100.0)),
        pipes=(WaterPipe("P2", "J", "R2", 100.0, 0.3, 130.0),),
        pumps=(Pump("PU", "R1", "J", "C"),), curves={"C": ((0.0, 30.0),) + ((0.01, 28.0),
                                                                          (0.02, 20.0))},
        duration=3600.0)
    _, out = _run(net)
    assert int(out.status[0, 1]) == CLOSED
    assert float(out.energy.utilisation) == 0.0
    assert float(out.energy.average_power) == 0.0
    assert float(out.energy.cost) == 0.0
    assert float(out.energy.efficiency) == 0.0


def test_energy_follows_the_efficiency_curve_and_the_speed():
    """The efficiency of a pump with a curve is `interp` at `q / speed`, adjusted for the
    speed as `100 - (100 - e) (1 / speed)^0.1` (`getenergy`)."""
    x = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 2.0]], dtype=F64)
    y = torch.tensor([[50.0, 70.0, 60.0], [10.0, 90.0, 90.0]], dtype=F64)
    xx = torch.tensor([[-1.0, 0.5, 1.0, 2.0, 9.0], [1.0, 1.0, 2.0, 5.0, 0.0]], dtype=F64)
    got = interpolate_curves(x, y, xx)
    expected = torch.tensor([[50.0, 60.0, 70.0, 65.0, 60.0], [50.0, 50.0, 90.0, 90.0, 10.0]],
                            dtype=F64)
    torch.testing.assert_close(got, expected, rtol=1e-14, atol=0)
    net = _constant_pump(
        curves={"E": ((0.0, 50.0), (0.01, 80.0), (0.02, 60.0))},
        pumps=(Pump("PU", "R", "J", None, power=5.0, efficiency_curve="E"),))
    _, out = _run(net)
    assert float(out.energy.efficiency) == pytest.approx(80.0, rel=1e-12)
    assert float(out.energy.average_power) == pytest.approx(5.0 / 0.8, rel=1e-10)
    slow = dataclasses.replace(net, pumps=(Pump("PU", "R", "J", None, power=5.0, speed=0.8,
                                                efficiency_curve="E"),))
    _, out = _run(slow)
    q4eff = 0.01 / 0.8                         # the flow at nominal speed
    e = 50.0 + (80.0 - 50.0) * (q4eff / 0.01) if q4eff <= 0.01 else \
        80.0 - (80.0 - 60.0) * (q4eff - 0.01) / 0.01
    e = 100.0 - (100.0 - e) * (1.0 / 0.8) ** 0.1
    assert float(out.energy.efficiency) == pytest.approx(e, rel=1e-10)


def test_energy_of_a_batch_equals_the_single_runs():
    """A batch of two demands gives each instance its own energy."""
    net = _constant_pump()
    model, _, drivers = build_model(net)
    base = drivers["water.sources"]
    batch = water_run(model, drivers={**drivers, "water.sources": torch.stack(
        [base, base * 0.5])})
    assert batch.energy.average_power.shape == (2, 1)
    assert batch.energy.demand_charge.shape == (2,)
    for k, m in enumerate((1.0, 0.5)):
        alone = water_run(model, drivers={**drivers, "water.sources": base * m})
        for field in ("utilisation", "efficiency", "energy_per_volume", "average_power",
                      "peak_power", "cost"):
            torch.testing.assert_close(getattr(batch.energy, field)[k],
                                       getattr(alone.energy, field), rtol=1e-12, atol=0)
        torch.testing.assert_close(batch.energy.demand_charge[k], alone.energy.demand_charge,
                                   rtol=1e-12, atol=0)


def test_energy_of_a_batch_with_a_pump_on_in_one_instance_only():
    """Two instances, the second pump at speed 0 (closed) in the second one only: the
    time on line, the average and peak power, the cost and the network's peak total power
    are each the single run's, and differ between the instances."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01),), reservoirs=(Reservoir("R", 20.0),),
        pumps=(Pump("PU", "R", "J", None, power=5.0), Pump("PV", "R", "J", None, power=5.0)),
        duration=7200.0, options=WaterOptions(flow_units="LPS"),
        energy=EnergyOptions(global_efficiency=75.0, global_price=0.1, demand_charge=3.0))
    model, _, drivers = build_model(net)
    both = drivers["water.setting"].clone()
    one = both.clone()
    one[model.link_names.index("PV")] = 0.0
    batch = water_run(model, drivers={**drivers, "water.setting": torch.stack([both, one])})
    e = batch.energy
    assert e.utilisation.tolist() == [[100.0, 100.0], [100.0, 0.0]]
    assert float(e.peak_power[1, 1]) == 0.0 and float(e.peak_power[0, 1]) > 0.0
    assert float(e.demand_charge[0]) != float(e.demand_charge[1])
    for k, setting in enumerate((both, one)):
        alone = water_run(model, drivers={**drivers, "water.setting": setting}).energy
        for field in ("utilisation", "efficiency", "energy_per_volume", "average_power",
                      "peak_power", "cost", "demand_charge"):
            torch.testing.assert_close(getattr(e, field)[k], getattr(alone, field),
                                       rtol=1e-12, atol=0)


def test_an_undefined_default_pattern_is_a_constant_one():
    """EPANET looks `[OPTIONS] PATTERN` (default "1") up among the patterns and uses none
    when it is not there (`input1.c`, `adjustdata`: `DefPat = findpattern(...)` is 0): the
    demands are their base values, where the run used to raise a KeyError."""
    net = _constant_pump(demand_pattern="1")
    assert "1" not in net.patterns
    _, out = _run(net)
    assert out.head.shape[0] == 3
    torch.testing.assert_close(out.head[0], out.head[2], rtol=1e-12, atol=0)


def test_efficiency_curve_tiny_is_in_the_files_flow_units():
    """EPANET's `interp` treats a step in x below `TINY = 1e-6` as flat, in the file's
    flow units (it interpolates `q * Ucf`): a 5e-10 m3/s step is 5e-7 L/s, flat, in an LPS
    network, but a real step in m3/s."""
    x = torch.tensor([[0.0, 5e-10]], dtype=F64)
    y = torch.tensor([[10.0, 20.0]], dtype=F64)
    xx = torch.tensor([[2.5e-10]], dtype=F64)
    flat = interpolate_curves(x, y, xx, tiny=1e-6 * 1e-3)
    real = interpolate_curves(x, y, xx, tiny=1e-12)
    assert float(flat) == 20.0
    assert float(real) == pytest.approx(15.0, rel=1e-12)
    net = _constant_pump(curves={"E": ((0.0, 50.0), (0.01, 80.0))},
                         pumps=(Pump("PU", "R", "J", None, power=5.0, efficiency_curve="E"),))
    model, _ = _run(net)
    from noodl.apps.water.energy import EnergyMeter

    assert EnergyMeter(model, 0).tiny == pytest.approx(1e-9, rel=1e-12)
