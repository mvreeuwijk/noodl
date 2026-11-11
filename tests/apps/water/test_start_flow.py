"""The water laws' start flows (`ImplicitLaw.start_flow`), where the linear guess takes them:
EPANET 2.2's first iterate, `initlinkflow` (`hydraul.c`)."""

from pathlib import Path

import pytest
import torch

from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN
from noodl.apps.water.elements import (
    ConstantPowerPump,
    FlowControlledPipe,
    FlowControlledPump,
    FlowControlValve,
    HazenWilliams,
    PiecewisePumpCurve,
    PressureReducingValve,
    PumpCurve,
    ThrottleValve,
)
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model

F64 = torch.float64
NET1 = Path(__file__).resolve().parents[2] / "data" / "water" / "Net1.inp"
FT = 0.3048


def _status(*values):
    return {"water.status": torch.tensor(values, dtype=F64)}


def test_a_pipe_starts_at_one_foot_per_second_and_a_closed_one_at_zero():
    d = torch.tensor([0.3, 0.2], dtype=F64)
    pipe = FlowControlledPipe(HazenWilliams(torch.tensor([100.0, 50.0], dtype=F64), d,
                                            torch.tensor([100.0, 120.0], dtype=F64)))
    area = torch.pi * d**2 / 4.0 * FT
    torch.testing.assert_close(pipe.start_flow(), area, rtol=1e-15, atol=0)
    with_links = FlowControlledPipe(pipe.pipe, links=slice(1, 3))
    q = with_links.start_flow(_status(OPEN, CLOSED, OPEN))
    assert q.tolist() == [0.0, float(area[1])]


def test_a_valve_starts_at_one_foot_per_second_with_its_diameter_and_at_zero_without():
    d = torch.tensor([0.25], dtype=F64)
    area = float(torch.pi * d**2 / 4.0 * FT)
    tcv = ThrottleValve(d, setting=torch.tensor([2.0], dtype=F64))
    assert float(tcv.start_flow()) == pytest.approx(area, rel=1e-15)
    assert FlowControlValve(setting=torch.tensor([0.01], dtype=F64)).start_flow() is None
    prv = PressureReducingValve(torch.tensor([10.0], dtype=F64), 0.0, kappa=1.0, diameter=d,
                                links=slice(0, 1))
    assert float(prv.start_flow(_status(ACTIVE))) == pytest.approx(area, rel=1e-15)
    assert float(prv.start_flow(_status(CLOSED))) == 0.0


def test_a_pump_starts_at_its_design_flow_times_its_speed():
    power = PumpCurve(torch.tensor([101.6, 30.0], dtype=F64),
                      torch.tensor([2836.0, 40.0], dtype=F64),
                      design_flow=torch.tensor([0.1, 0.02], dtype=F64))
    segments = PiecewisePumpCurve(torch.tensor([[0.01, 0.1, 0.2, torch.inf]], dtype=F64),
                                  torch.tensor([[90.0, 88.0, 80.0, 0.0]], dtype=F64), [3])
    constant = ConstantPowerPump(torch.tensor([10.0], dtype=F64))
    # a custom curve's design flow is the mean of its first and last points'; a constant
    # power pump's is 1 cfs
    assert float(segments.design_flow) == pytest.approx(0.105, rel=1e-15)
    assert float(constant.design_flow) == pytest.approx(FT**3, rel=1e-15)
    law = FlowControlledPump([(power, [0, 3]), (segments, [1]), (constant, [2])],
                             links=slice(0, 4))
    drivers = {"water.status": torch.tensor([OPEN, OPEN, OPEN, CLOSED], dtype=F64),
               "water.setting": torch.tensor([0.9, 0.8, 0.0, 1.0], dtype=F64)}
    q = law.start_flow(drivers)
    # pump 2 at speed 0 and pump 3 CLOSED start at zero
    assert q.tolist() == pytest.approx([0.09, 0.8 * 0.105, 0.0, 0.0], rel=1e-15)
    # a power function built without a design flow starts at zero
    bare = PumpCurve(torch.tensor([101.6], dtype=F64), torch.tensor([2836.0], dtype=F64))
    assert FlowControlledPump(bare).start_flow() is None


def test_build_model_gives_net1_epanets_first_iterate():
    """Net1's start flows are `build_model`'s seed of the Newton solve, `initial_flows`."""
    model, _, drivers = build_model(read_epanet_inp(NET1))
    layer = model.kirchhoff["water"]
    for kind in ("pipe", "pump"):
        law, cols = layer.law_for(kind)
        torch.testing.assert_close(law.start_flow(drivers), model.initial_flows[cols],
                                   rtol=1e-15, atol=0)
