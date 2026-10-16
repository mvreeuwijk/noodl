"""AC power flow: model building, MATPOWER semantics, refusals and gradients."""

import dataclasses
import math

import pytest
import torch

from noodl.apps.electrical.power import (
    POWER_LAYER,
    Generator,
    build_power_flow,
    check_power_network,
    power_flow_solution,
    read_matpower,
    solve_power_flow,
)
from tests._power_cases import four_bus, power_case


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _solve(net, **kw):
    model, state, drivers = build_power_flow(net, **kw)
    return model, power_flow_solution(model, solve_power_flow(model, state, drivers))


@pytest.mark.external
def test_case9_spot_values():
    # PYPOWER 5.1.21 on MATPOWER 8.1's case9 (flat start): the slack magnitude is its
    # generator's Vg (1.04), not the bus table's Vm (1.0).
    _, sol = _solve(read_matpower(power_case("case9")))
    assert abs(sol.vm[0] - 1.04) < 1e-12
    assert abs(sol.vm[8] - 0.9956308580482949) < 1e-9
    assert abs(sol.va[1] - 0.161966650257789) < 1e-9
    assert abs(sol.pg[0] - 0.716410214744823) < 1e-8
    assert abs(sol.qg[2] - (-0.10859709070988515)) < 1e-8


def test_layer_layout_flat_start_and_complex_tellegen():
    model, state, drivers = build_power_flow(four_bus())
    layer = model.kirchhoff[POWER_LAYER]
    assert layer.phasor and layer.kinds == ["branch", "load", "shunt", "generator"]
    q0 = state["power.q"]
    # branch ends, loads, shunt, the PV generator (the slack is a boundary node, not an edge)
    assert q0.dtype == torch.complex128 and q0.shape == (2 * 4 + 3 + 1 + 1,)
    # flat start: |V| = Vg at regulated buses, 1 elsewhere, 0 at ground
    start = state["power.phi"].abs()
    assert start.tolist() == [1.02, 1.01, 1.0, 1.0, 0.0]
    solved = solve_power_flow(model, state, drivers)
    phi, q = solved["power.phi"], solved["power.q"]
    assert abs(layer.power_residual(phi, q, {})) < 1e-10


def test_enforce_q_limits_is_refused():
    with pytest.raises(ValueError, match="reactive-power limits"):
        build_power_flow(four_bus(), enforce_q_limits=True)


def test_island_is_refused_naming_its_buses():
    net = four_bus()
    # bus 4 hangs on its transformer 3-4 (row 3) alone
    cut = [dataclasses.replace(b, in_service=False) if k == 3 else b
           for k, b in enumerate(net.branches)]
    with pytest.raises(ValueError, match=r"not connected to the slack bus.*\[4\]"):
        build_power_flow(dataclasses.replace(net, branches=tuple(cut)))


def test_conflicting_voltage_setpoints_are_refused():
    net = four_bus()
    second = Generator(2, 10.0, vm_set=1.03)                # a second generator on bus 2
    with pytest.raises(ValueError, match=r"bus 2.*1\.01.*1\.03"):
        build_power_flow(dataclasses.replace(net, generators=(*net.generators, second)))


def test_zero_impedance_branch_is_refused():
    net = four_bus()
    br = list(net.branches)
    br[3] = dataclasses.replace(br[3], x=0.0)               # r is already 0 on 3-4
    with pytest.raises(ValueError, match=r"row 4 of the branch table \(from bus 3 to bus 4\)"):
        build_power_flow(dataclasses.replace(net, branches=tuple(br)))


def test_infeasible_loading_raises_naming_the_layer():
    net = four_bus()
    heavy = tuple(dataclasses.replace(ld, p_mw=10 * ld.p_mw, q_mvar=10 * ld.q_mvar)
                  for ld in net.loads)
    model, state, drivers = build_power_flow(dataclasses.replace(net, loads=heavy))
    with pytest.raises(RuntimeError, match="KirchhoffLayer 'power'") as failure:
        solve_power_flow(model, state, drivers)
    assert "power flow: no solution found" in str(failure.value)
    assert "exceed what the network can transfer" in str(failure.value)
    assert isinstance(failure.value.__cause__, RuntimeError)


@pytest.mark.parametrize("bus, vm_set", [(1, 0.0), (2, -1.01)])
def test_nonpositive_voltage_setpoint_is_refused(bus, vm_set):
    net = four_bus()
    gens = tuple(dataclasses.replace(g, vm_set=vm_set) if g.bus == bus else g
                 for g in net.generators)
    with pytest.raises(ValueError, match=rf"bus {bus}: voltage setpoint {vm_set}.*positive"):
        build_power_flow(dataclasses.replace(net, generators=gens))


def _failing_steady(model, error):
    def steady(*args, **kwargs):
        raise error

    model.steady = steady


def test_a_singular_power_flow_system_gets_the_advice():
    model, state, drivers = build_power_flow(four_bus())
    error = RuntimeError("KirchhoffLayer 'power' solve: singular system at batch indices [0]")
    _failing_steady(model, error)
    with pytest.raises(RuntimeError, match="power flow: no solution found") as failure:
        solve_power_flow(model, state, drivers)
    assert failure.value.__cause__ is error


@pytest.mark.parametrize("message", [
    "KirchhoffLayer 'power': law 0 of kind 'load' declares dp_independent = True",
    "KirchhoffLayer 'power_dc' solve: batch indices [0] failed to converge",
    "solve: batch indices [0] failed to converge after 30 iterations",
])
def test_other_runtime_errors_pass_through_unchanged(message):
    model, state, drivers = build_power_flow(four_bus())
    error = RuntimeError(message)
    _failing_steady(model, error)
    with pytest.raises(RuntimeError) as failure:
        solve_power_flow(model, state, drivers)
    assert failure.value is error


def test_voltage_sensitivity_to_load_matches_finite_differences():
    # d|V_k|/dP_load at bus 3 of the four-bus network for the PQ buses 3 and 4 (sensitivities
    # 1.06e-2 and 1.12e-2, far above the difference's noise ~1e-11), by the adjoint of the
    # converged solve, against a central difference of two plain solves (h = 1e-5 p.u.).
    net = four_bus()
    model, state, drivers = build_power_flow(net, learnable=True)
    load, _ = model.kirchhoff[POWER_LAYER].law_for("load")
    sol = power_flow_solution(model, solve_power_flow(model, state, drivers))
    grads = [torch.autograd.grad(sol.vm[k], load.P, retain_graph=True)[0][1] for k in (2, 3)]

    def vm_at(dp):
        extra = dp * net.base_mva
        loads = tuple(dataclasses.replace(ld, p_mw=ld.p_mw + (extra if ld.bus == 3 else 0))
                      for ld in net.loads)
        m, s, d = build_power_flow(dataclasses.replace(net, loads=loads))
        out = power_flow_solution(m, solve_power_flow(m, s, d, differentiable=False))
        return out.vm[[2, 3]]

    h = 1e-5
    fd = (vm_at(h) - vm_at(-h)) / (2 * h)
    for g, f in zip(grads, fd, strict=True):
        assert abs(float(g)) > 1e-3
        assert math.isclose(float(g), float(f), rel_tol=1e-6)


def test_one_slack_bus_with_a_generator_is_required():
    net = four_bus()
    two = tuple(dataclasses.replace(b, kind="slack") if b.id == 2 else b for b in net.buses)
    with pytest.raises(ValueError, match=r"exactly one slack bus, found 2: \[1, 2\]"):
        build_power_flow(dataclasses.replace(net, buses=two))
    off = tuple(dataclasses.replace(g, in_service=False) if g.bus == 1 else g
                for g in net.generators)
    with pytest.raises(ValueError, match="slack bus 1 has no in-service generator"):
        build_power_flow(dataclasses.replace(net, generators=off))


def test_zero_reactance_is_refused_only_by_the_dc_check():
    net = four_bus()
    br = list(net.branches)
    br[0] = dataclasses.replace(br[0], x=0.0)               # 1-2 keeps r = 0.01
    net = dataclasses.replace(net, branches=tuple(br))
    assert check_power_network(net).branch_rows == tuple(range(4))
    with pytest.raises(ValueError,
                       match=r"row 1 of the branch table \(from bus 1 to bus 2\) "
                             r"has zero reactance"):
        check_power_network(net, dc=True)


@pytest.mark.external
def test_feature_case_index_follows_matpower_bus_types():
    index = check_power_network(read_matpower(power_case("case9_features")))
    # bus 3's only generator is out of service: solved as PQ; bus 7's generator is fixed
    assert index.pv_buses == (2,) and index.gen_buses == (1, 2, 7)
    assert index.fixed_generation == {7: complex(0.6, 0.15)}
    assert index.branch_rows == tuple(range(9)) and index.shunt_buses == (5,)
    assert index.load_buses == (5, 7, 9)


@pytest.mark.external
def test_branch_end_powers_close_the_power_balance():
    net = read_matpower(power_case("case9_features"))
    _, sol = _solve(net)
    losses = (sol.sf + sol.st).sum()
    shunt = complex(0.05, -0.10) * sol.vm[4] ** 2           # Gs = 5 MW, Bs = 10 MVAr at bus 5
    load = complex(sum(ld.p_mw for ld in net.loads), sum(ld.q_mvar for ld in net.loads)) / 100
    generation = complex(sol.pg.sum(), sol.qg.sum())
    assert abs(generation - load - shunt - complex(losses)) < 1e-10
    assert float(losses.real) > 0


def test_coefficients_are_float64_whatever_the_default_dtype():
    torch.set_default_dtype(torch.float32)      # the autouse fixture restores it
    model, sol = _solve(four_bus())
    branch, _ = model.kirchhoff[POWER_LAYER].law_for("branch")
    assert branch.r.dtype == torch.float64 and sol.vm.dtype == torch.float64
    assert abs(sol.vm[3] - 1.0129223498697453) < 1e-9      # its value under float64


def test_cut_off_slack_bus_leaves_the_rest_as_an_island():
    net = four_bus()
    # the slack bus 1 hangs on the lines 1-2 and 1-3 (rows 0 and 1)
    cut = [dataclasses.replace(b, in_service=False) if k in (0, 1) else b
           for k, b in enumerate(net.branches)]
    with pytest.raises(ValueError, match=r"slack bus 1 .*\[2, 3, 4\]"):
        build_power_flow(dataclasses.replace(net, branches=tuple(cut)))


def test_solution_keeps_leading_batch_dimensions():
    model, state, drivers = build_power_flow(four_bus())
    slack = drivers["power.phi_boundary"]
    batched = dict(drivers, **{"power.phi_boundary": torch.stack([slack, 1.01 * slack])})
    sol = power_flow_solution(model, solve_power_flow(model, state, batched))
    assert sol.vm.shape == (2, 4) and sol.pg.shape == (2, 2) and sol.sf.shape == (2, 4)
    _, single = _solve(four_bus())
    assert torch.allclose(sol.vm[0], single.vm, atol=1e-12, rtol=0)
    assert torch.allclose(sol.qg[0], single.qg, atol=1e-12, rtol=0)
    assert abs(float(sol.vm[1, 0]) - 1.01 * 1.02) < 1e-12
