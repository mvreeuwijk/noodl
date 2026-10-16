"""DC power flow: plain nodal analysis on the classic Kirchhoff layer."""

import dataclasses
import math

import pytest
import torch

from noodl.apps.electrical.power import (
    DC_LAYER,
    Bus,
    DcPowerFlowSolution,
    FixedShunt,
    Generator,
    Load,
    PowerNetwork,
    build_power_flow,
    dc_power_flow,
    dc_power_flow_solution,
    read_matpower,
    solve_power_flow,
)
from noodl.elements import Conductance
from tests._power_cases import four_bus, power_case


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _dc(net):
    model, state, drivers = dc_power_flow(net)
    return model, dc_power_flow_solution(model, solve_power_flow(model, state, drivers))


@pytest.mark.external
def test_case9_is_nodal_analysis_on_the_classic_layer():
    model, sol = _dc(read_matpower(power_case("case9")))
    layer = model.kirchhoff[DC_LAYER]
    assert layer.mna is False
    assert list(layer.kinds) == ["branch"]
    assert type(layer.law_for("branch")[0]) is Conductance
    assert isinstance(sol, DcPowerFlowSolution)
    # PYPOWER 5.1.21 rundcpf on case9: bus 2 angle and branch 1-4 flow, slack generation
    assert abs(sol.va[1] - 0.17097278260869575) < 1e-12
    assert abs(sol.pf[0] - 0.6699999999999996) < 1e-12
    assert abs(sol.pg[0] - 0.67) < 1e-12


def _shifted():
    """The four-bus network with a phase shift of -3 degrees on its transformer 3-4, a slack
    angle of 10 degrees and a shunt conductance at bus 4."""
    net = four_bus()
    br = list(net.branches)
    br[3] = dataclasses.replace(br[3], shift_deg=-3.0)
    buses = (dataclasses.replace(net.buses[0], va_deg=10.0), *net.buses[1:])
    return dataclasses.replace(net, buses=buses, branches=tuple(br),
                               shunts=(FixedShunt(4, 2.0, 8.0),))


def test_build_rules():
    net = _shifted()
    model, state, drivers = dc_power_flow(net)
    assert state == {}
    layer = model.kirchhoff[DC_LAYER]
    rows = model.power_index.branch_rows
    branches = [net.branches[k] for k in rows]
    g = layer.law_for("branch")[0].g
    assert g.dtype == torch.float64
    expected = [1.0 / (br.x * (br.ratio or 1.0)) for br in branches]
    assert torch.allclose(g, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    shift = drivers[f"{DC_LAYER}.shift"]
    assert shift.dtype == torch.float64
    assert shift.tolist() == [-math.radians(br.shift_deg) for br in branches]
    assert shift.tolist()[3] == math.radians(3.0)
    slack = model.power_index.slack_bus
    va = {b.id: b.va_deg for b in net.buses}[slack]
    assert drivers[f"{DC_LAYER}.phi_boundary"].tolist() == [math.radians(10.0)] == [
        math.radians(va)]
    assert model.power_network is net


@pytest.mark.external
def test_phase_shifter_enters_as_a_drive():
    # case9_features: branch 1-4 has ratio 1.02 and shift -3 deg; its DC flow is 0.97 p.u.
    model, sol = _dc(read_matpower(power_case("case9_features")))
    assert abs(sol.pf[0] - 0.9699999999999992) < 1e-12
    assert len(sol.pf) == 9                     # the out-of-service branch 9-4 is dropped


def test_injections_balance_the_branch_flows():
    net = _shifted()
    model, sol = _dc(net)
    # total generation equals total demand plus shunt conductance: the DC network is lossless
    demand = sum(ld.p_mw for ld in net.loads) + sum(sh.g_mw for sh in net.shunts)
    assert abs(float(sol.pg.sum()) - demand / net.base_mva) < 1e-12
    # bus 4 is fed by the transformer alone: its flow is bus 4's demand and conductance
    assert abs(float(sol.pf[3]) - (45.0 + 2.0) / 100) < 1e-12


def test_batched_drivers():
    net = four_bus()
    model, state, drivers = dc_power_flow(net)
    batched = {k: torch.stack([v, 2 * v]) for k, v in drivers.items()}
    sol = dc_power_flow_solution(model, solve_power_flow(model, state, batched))
    single = dc_power_flow_solution(model, solve_power_flow(model, state, drivers))
    assert sol.pf.shape == (2, 4) and sol.pg.shape == (2, 2)
    assert torch.allclose(sol.pf[1], 2 * single.pf, atol=1e-12)
    assert torch.allclose(sol.va[1], 2 * single.va, atol=1e-12)
    # only the slack's generation follows the injections; the others are the network's
    # setpoints, not read from the batched sources driver
    assert torch.equal(sol.pg[0], single.pg)
    assert sol.gen_buses == (1, 2) and sol.pg[1, 1] == single.pg[1] == 0.5
    # bus 1, the slack, has no load or shunt: its generation doubles with the injections
    assert torch.allclose(sol.pg[1, 0], 2 * single.pg[0], atol=1e-12)


def test_a_layer_error_other_than_a_failed_solve_is_not_wrapped():
    # a layer error that is neither a failed Newton solve nor a singular system (here: a
    # negative conductance written into the built model) passes through unchanged
    model, state, drivers = dc_power_flow(four_bus())
    g = model.kirchhoff[DC_LAYER].law_for("branch")[0].g
    with torch.no_grad():
        g[1] = -g[1]
    with pytest.raises(RuntimeError, match="negative slope") as failure:
        solve_power_flow(model, state, drivers)
    assert "power flow: no solution found" not in str(failure.value)


@pytest.mark.parametrize("change, row", [
    (dict(x=-0.05), 2),                      # series compensation: negative reactance
    (dict(ratio=-1.0), 2),                   # x tau < 0 through the tap ratio
])
def test_nonpositive_branch_susceptance_is_refused(change, row):
    net = four_bus()
    br = list(net.branches)
    br[row - 1] = dataclasses.replace(br[row - 1], **change)
    changed = dataclasses.replace(net, branches=tuple(br))
    with pytest.raises(ValueError, match=(
            rf"row {row} of the branch table \(from bus 1 to bus 3\).*the DC power flow on the "
            r"classic layer needs positive branch susceptance")):
        dc_power_flow(changed)
    build_power_flow(changed)               # the AC power flow takes a negative reactance


def test_zero_reactance_and_islands_are_refused():
    net = four_bus()
    br = list(net.branches)
    br[0] = dataclasses.replace(br[0], x=0.0)   # r = 0.01 keeps the AC model valid
    with pytest.raises(ValueError, match=r"row 1 of the branch table \(from bus 1 to bus 2\)"):
        dc_power_flow(dataclasses.replace(net, branches=tuple(br)))
    cut = [dataclasses.replace(b, in_service=False) if k == 3 else b
           for k, b in enumerate(net.branches)]
    with pytest.raises(ValueError, match=r"not connected to the slack bus.*\[4\]"):
        dc_power_flow(dataclasses.replace(net, branches=tuple(cut)))


def test_a_network_without_branches_is_refused():
    net = PowerNetwork(buses=(Bus(1, "slack"),), branches=(),
                       generators=(Generator(1, 50.0),), loads=(Load(1, 50.0, 0.0),))
    with pytest.raises(ValueError, match="at least one in-service branch"):
        dc_power_flow(net)
