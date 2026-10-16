"""AC and DC power flow against pandapower on MATPOWER cases (stored reference output).

The fixtures were computed once by `scripts/power_flow_reference.py` in a scratch virtual
environment (pandapower 3.5.6, checked against PYPOWER 5.1.21); see
`tests/data/power_flow/NOTICE.md`. Tolerances are absolute: 1e-8 p.u. on |V|, 1e-8 rad on
angles, 1e-6 p.u. on generator powers and DC branch flows. The case files are not in the
repository: `matpower_case` fetches them on first use (MATPOWER 8.1, SHA-256 checked) and the
comparisons skip without a network connection.
"""

import json
from pathlib import Path

import pytest
import torch

from noodl.apps.electrical.power import (
    MATPOWER_CASES,
    build_power_flow,
    dc_power_flow,
    dc_power_flow_solution,
    power_flow_solution,
    read_matpower,
    solve_power_flow,
)
from tests._power_cases import power_case

DATA = Path(__file__).resolve().parents[1] / "data" / "power_flow"
CASES = ["case9", "case14", "case30", "case118", "case9_features"]


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _fixture(case):
    return json.loads((DATA / f"{case}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_fixture_belongs_to_the_pinned_case_file(case):
    # `matpower_case` refuses any file whose LF-normalised SHA-256 is not MATPOWER_CASES[case],
    # so the reference result and the case the comparisons below read are the same file.
    ref = _fixture(case)
    assert ref["format"] == "noodl-power-flow-reference/1" and ref["case"] == case
    assert ref["case_sha256_lf"] == MATPOWER_CASES[case]
    assert ref["engines"]["pandapower"] == "3.5.6" and ref["engines"]["PYPOWER"] == "5.1.21"
    assert max(ref["pandapower_vs_pypower_max_abs"].values()) < 1e-10
    # one entry per bus, per generator bus and per in-service branch, consistently
    ac, dc = ref["ac"], ref["dc"]
    n_bus, n_gen = len(ac["bus"]), len(ac["gen_bus"])
    assert len(ac["vm"]) == len(ac["va_rad"]) == len(dc["va_rad"]) == n_bus
    assert len(ac["pg_pu"]) == len(ac["qg_pu"]) == len(dc["pg_pu"]) == n_gen
    assert dc["gen_bus"] == ac["gen_bus"]


@pytest.mark.external
@pytest.mark.parametrize("case", CASES)
def test_fixture_sizes_match_the_case_file(case):
    ref = _fixture(case)
    net = read_matpower(power_case(case))
    assert ref["ac"]["bus"] == list(net.bus_ids())
    gen_buses = [b for b in net.bus_ids()
                 if any(g.in_service and g.bus == b for g in net.generators)]
    assert ref["ac"]["gen_bus"] == gen_buses
    assert len(ref["dc"]["branch_pf_pu"]) == sum(br.in_service for br in net.branches)


def _max_abs(a, b):
    a, b = torch.as_tensor(a), torch.tensor(b)
    assert a.shape == b.shape, f"shape {tuple(a.shape)} against the reference's {tuple(b.shape)}"
    return float((a - b).abs().max())


@pytest.mark.external
@pytest.mark.parametrize("case", CASES)
def test_ac_power_flow_matches_pandapower(case):
    ref = _fixture(case)["ac"]
    model, state, drivers = build_power_flow(read_matpower(power_case(case)))
    sol = power_flow_solution(model, solve_power_flow(model, state, drivers,
                                                      differentiable=False))
    assert list(sol.bus_ids) == ref["bus"] and list(sol.gen_buses) == ref["gen_bus"]
    assert _max_abs(sol.vm, ref["vm"]) < 1e-8
    assert _max_abs(sol.va, ref["va_rad"]) < 1e-8
    assert _max_abs(sol.pg, ref["pg_pu"]) < 1e-6
    assert _max_abs(sol.qg, ref["qg_pu"]) < 1e-6


@pytest.mark.external
@pytest.mark.parametrize("case", CASES)
def test_dc_power_flow_matches_pandapower(case):
    ref = _fixture(case)["dc"]
    model, state, drivers = dc_power_flow(read_matpower(power_case(case)))
    sol = dc_power_flow_solution(model, solve_power_flow(model, state, drivers,
                                                         differentiable=False))
    assert list(sol.bus_ids) == _fixture(case)["ac"]["bus"]
    assert list(sol.gen_buses) == ref["gen_bus"]
    assert len(sol.branch_rows) == len(ref["branch_pf_pu"])
    assert _max_abs(sol.va, ref["va_rad"]) < 1e-8
    assert _max_abs(sol.pg, ref["pg_pu"]) < 1e-6
    assert _max_abs(sol.pf, ref["branch_pf_pu"]) < 1e-6
