"""AC power flow: model building, MATPOWER semantics, refusals and gradients."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.electrical.power import (
    POWER_LAYER,
    build_power_flow,
    check_power_network,
    power_flow_solution,
    read_matpower,
    solve_power_flow,
)

DATA = Path(__file__).resolve().parents[3] / "data" / "power_flow"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _solve(net, **kw):
    model, state, drivers = build_power_flow(net, **kw)
    return model, power_flow_solution(model, solve_power_flow(model, state, drivers))


def test_case9_spot_values():
    # PYPOWER 5.1.21 on MATPOWER 8.1's case9 (flat start): the slack magnitude is its
    # generator's Vg (1.04), not the bus table's Vm (1.0).
    _, sol = _solve(read_matpower(DATA / "case9.m"))
    assert abs(sol.vm[0] - 1.04) < 1e-12
    assert abs(sol.vm[8] - 0.9956308580482949) < 1e-9
    assert abs(sol.va[1] - 0.161966650257789) < 1e-9
    assert abs(sol.pg[0] - 0.716410214744823) < 1e-8
    assert abs(sol.qg[2] - (-0.10859709070988515)) < 1e-8


def test_layer_layout_flat_start_and_complex_tellegen():
    model, state, drivers = build_power_flow(read_matpower(DATA / "case9.m"))
    layer = model.potential[POWER_LAYER]
    assert layer.phasor and layer.kinds == ["branch", "load", "generator"]
    q0 = state["power.q"]
    assert q0.dtype == torch.complex128 and q0.shape == (2 * 9 + 3 + 2,)  # branches, loads, gens
    # flat start: |V| = Vg at regulated buses, 1 elsewhere, 0 at ground
    start = state["power.phi"].abs()
    assert start.tolist() == [1.04, 1.025, 1.025] + [1.0] * 6 + [0.0]
    solved = solve_power_flow(model, state, drivers)
    phi, q = solved["power.phi"], solved["power.q"]
    assert abs(layer.power_residual(phi, q, {})) < 1e-10


def test_enforce_q_limits_is_refused():
    with pytest.raises(ValueError, match="reactive-power limits"):
        build_power_flow(read_matpower(DATA / "case9.m"), enforce_q_limits=True)


def test_island_is_refused_naming_its_buses():
    net = read_matpower(DATA / "case9.m")
    # bus 9 hangs on branches 8-9 (row 7) and 9-4 (row 8); switch both out
    cut = [dataclasses.replace(b, in_service=False) if k in (7, 8) else b
           for k, b in enumerate(net.branches)]
    with pytest.raises(ValueError, match=r"not connected to the slack bus.*\[9\]"):
        build_power_flow(dataclasses.replace(net, branches=tuple(cut)))


def test_conflicting_voltage_setpoints_are_refused():
    net = read_matpower(DATA / "case9_features.m")
    gens = list(net.generators)
    gens[2] = dataclasses.replace(gens[2], vm_set=1.03)     # second generator on bus 2
    with pytest.raises(ValueError, match=r"bus 2.*1\.025.*1\.03"):
        build_power_flow(dataclasses.replace(net, generators=tuple(gens)))


def test_zero_impedance_branch_is_refused():
    net = read_matpower(DATA / "case9.m")
    br = list(net.branches)
    br[0] = dataclasses.replace(br[0], x=0.0)               # r is already 0 on 1-4
    with pytest.raises(ValueError, match=r"branch 0 \(1-4\)"):
        build_power_flow(dataclasses.replace(net, branches=tuple(br)))


def test_infeasible_loading_raises_naming_the_layer():
    net = read_matpower(DATA / "case9.m")
    heavy = tuple(dataclasses.replace(ld, p_mw=10 * ld.p_mw, q_mvar=10 * ld.q_mvar)
                  for ld in net.loads)
    model, state, drivers = build_power_flow(dataclasses.replace(net, loads=heavy))
    with pytest.raises(RuntimeError, match="PotentialFlowLayer 'power'"):
        solve_power_flow(model, state, drivers)


def test_voltage_sensitivity_to_load_matches_finite_differences():
    # d|V_k|/dP_load at bus 5 of case9, by the adjoint of the converged solve, against a
    # central difference of two plain solves (h = 1e-5 p.u.; truncation ~1e-10).
    net = read_matpower(DATA / "case9.m")
    model, state, drivers = build_power_flow(net, learnable=True)
    load, _ = model.potential[POWER_LAYER].element_for("load")
    sol = power_flow_solution(model, solve_power_flow(model, state, drivers))
    grads = [torch.autograd.grad(sol.vm[k], load.P, retain_graph=True)[0][0] for k in (4, 8)]

    def vm_at(dp):
        loads = tuple(dataclasses.replace(ld, p_mw=ld.p_mw + (dp * 100 if ld.bus == 5 else 0))
                      for ld in net.loads)
        m, s, d = build_power_flow(dataclasses.replace(net, loads=loads))
        out = power_flow_solution(m, solve_power_flow(m, s, d, differentiable=False))
        return out.vm[[4, 8]]

    h = 1e-5
    fd = (vm_at(h) - vm_at(-h)) / (2 * h)
    for g, f in zip(grads, fd, strict=True):
        assert math.isclose(float(g), float(f), rel_tol=1e-6)


def test_one_slack_bus_with_a_generator_is_required():
    net = read_matpower(DATA / "case9.m")
    two = tuple(dataclasses.replace(b, kind="slack") if b.id == 2 else b for b in net.buses)
    with pytest.raises(ValueError, match=r"exactly one slack bus, found 2: \[1, 2\]"):
        build_power_flow(dataclasses.replace(net, buses=two))
    off = tuple(dataclasses.replace(g, in_service=False) if g.bus == 1 else g
                for g in net.generators)
    with pytest.raises(ValueError, match="slack bus 1 has no in-service generator"):
        build_power_flow(dataclasses.replace(net, generators=off))


def test_zero_reactance_is_refused_only_by_the_dc_check():
    net = read_matpower(DATA / "case9.m")
    br = list(net.branches)
    br[1] = dataclasses.replace(br[1], x=0.0)               # 4-5 keeps r = 0.017
    net = dataclasses.replace(net, branches=tuple(br))
    assert check_power_network(net).branch_rows == tuple(range(9))
    with pytest.raises(ValueError, match=r"branch 1 \(4-5\) has zero reactance"):
        check_power_network(net, dc=True)


def test_feature_case_index_follows_matpower_bus_types():
    index = check_power_network(read_matpower(DATA / "case9_features.m"))
    # bus 3's only generator is out of service: solved as PQ; bus 7's generator is fixed
    assert index.pv_buses == (2,) and index.gen_buses == (1, 2, 7)
    assert index.fixed_generation == {7: complex(0.6, 0.15)}
    assert index.branch_rows == tuple(range(9)) and index.shunt_buses == (5,)
    assert index.load_buses == (5, 7, 9)


def test_branch_end_powers_close_the_power_balance():
    net = read_matpower(DATA / "case9_features.m")
    _, sol = _solve(net)
    losses = (sol.sf + sol.st).sum()
    shunt = complex(0.05, -0.10) * sol.vm[4] ** 2           # Gs = 5 MW, Bs = 10 MVAr at bus 5
    load = complex(sum(ld.p_mw for ld in net.loads), sum(ld.q_mvar for ld in net.loads)) / 100
    generation = complex(sol.pg.sum(), sol.qg.sum())
    assert abs(generation - load - shunt - complex(losses)) < 1e-10
    assert float(losses.real) > 0


def test_coefficients_are_float64_whatever_the_default_dtype():
    torch.set_default_dtype(torch.float32)      # the autouse fixture restores it
    model, sol = _solve(read_matpower(DATA / "case9.m"))
    branch, _ = model.potential[POWER_LAYER].element_for("branch")
    assert branch.r.dtype == torch.float64 and sol.vm.dtype == torch.float64
    assert abs(sol.vm[8] - 0.9956308580482949) < 1e-9
