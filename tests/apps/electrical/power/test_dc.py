"""DC power flow: plain nodal analysis on the classic potential layer."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.electrical.power import (
    DC_LAYER,
    Bus,
    DcPowerFlowSolution,
    Generator,
    Load,
    PowerNetwork,
    dc_power_flow,
    dc_power_flow_solution,
    read_matpower,
    solve_power_flow,
)
from noodl.elements import Conductance

DATA = Path(__file__).resolve().parents[3] / "data" / "power_flow"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _dc(net):
    model, state, drivers = dc_power_flow(net)
    return model, dc_power_flow_solution(model, solve_power_flow(model, state, drivers))


def test_case9_is_nodal_analysis_on_the_classic_layer():
    model, sol = _dc(read_matpower(DATA / "case9.m"))
    layer = model.potential[DC_LAYER]
    assert layer.mna is False
    assert list(layer.kinds) == ["branch"]
    assert type(layer.element_for("branch")[0]) is Conductance
    assert isinstance(sol, DcPowerFlowSolution)
    # PYPOWER 5.1.21 rundcpf on case9: bus 2 angle and branch 1-4 flow, slack generation
    assert abs(sol.va[1] - 0.17097278260869575) < 1e-12
    assert abs(sol.pf[0] - 0.6699999999999996) < 1e-12
    assert abs(sol.pg[0] - 0.67) < 1e-12


def test_build_rules():
    net = read_matpower(DATA / "case9_features.m")
    model, state, drivers = dc_power_flow(net)
    assert state == {}
    layer = model.potential[DC_LAYER]
    rows = model.power_index.branch_rows
    branches = [net.branches[k] for k in rows]
    g = layer.element_for("branch")[0].g
    assert g.dtype == torch.float64
    expected = [1.0 / (br.x * (br.ratio or 1.0)) for br in branches]
    assert torch.allclose(g, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    shift = drivers[f"{DC_LAYER}.shift"]
    assert shift.dtype == torch.float64
    assert shift.tolist() == [-math.radians(br.shift_deg) for br in branches]
    slack = model.power_index.slack_bus
    va = {b.id: b.va_deg for b in net.buses}[slack]
    assert drivers[f"{DC_LAYER}.phi_boundary"].tolist() == [math.radians(va)]
    assert model.power_network is net


def test_phase_shifter_enters_as_a_drive():
    # case9_features: branch 1-4 has ratio 1.02 and shift -3 deg; its DC flow is 0.97 p.u.
    model, sol = _dc(read_matpower(DATA / "case9_features.m"))
    assert abs(sol.pf[0] - 0.9699999999999992) < 1e-12
    assert len(sol.pf) == 9                     # the out-of-service branch 9-4 is dropped


def test_injections_balance_the_branch_flows():
    net = read_matpower(DATA / "case14.m")
    model, sol = _dc(net)
    # total generation equals total demand plus shunt conductance: the DC network is lossless
    demand = sum(ld.p_mw for ld in net.loads) + sum(sh.g_mw for sh in net.shunts)
    assert abs(float(sol.pg.sum()) - demand / net.base_mva) < 1e-12


def test_batched_drivers():
    net = read_matpower(DATA / "case9.m")
    model, state, drivers = dc_power_flow(net)
    batched = {k: torch.stack([v, 2 * v]) for k, v in drivers.items()}
    sol = dc_power_flow_solution(model, solve_power_flow(model, state, batched))
    single = dc_power_flow_solution(model, solve_power_flow(model, state, drivers))
    assert sol.pf.shape == (2, 9) and sol.pg.shape == (2, 3)
    assert torch.allclose(sol.pf[1], 2 * single.pf, atol=1e-12)
    # only the slack's generation follows the injections; the others are fixed setpoints
    assert torch.allclose(sol.va[1], 2 * single.va, atol=1e-12)


def test_zero_reactance_and_islands_are_refused():
    net = read_matpower(DATA / "case9.m")
    br = list(net.branches)
    br[1] = dataclasses.replace(br[1], x=0.0)   # r = 0.017 keeps the AC model valid
    with pytest.raises(ValueError, match=r"row 2 of the branch table \(from bus 4 to bus 5\)"):
        dc_power_flow(dataclasses.replace(net, branches=tuple(br)))
    cut = [dataclasses.replace(b, in_service=False) if k in (7, 8) else b
           for k, b in enumerate(net.branches)]
    with pytest.raises(ValueError, match=r"not connected to the slack bus.*\[9\]"):
        dc_power_flow(dataclasses.replace(net, branches=tuple(cut)))


def test_a_network_without_branches_is_refused():
    net = PowerNetwork(buses=(Bus(1, "slack"),), branches=(),
                       generators=(Generator(1, 50.0),), loads=(Load(1, 50.0, 0.0),))
    with pytest.raises(ValueError, match="at least one in-service branch"):
        dc_power_flow(net)
