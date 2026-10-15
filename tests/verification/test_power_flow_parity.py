"""AC and DC power flow against pandapower on MATPOWER cases (stored reference output).

The fixtures were computed once by `scripts/power_flow_reference.py` in a scratch virtual
environment (pandapower 3.5.6, checked against PYPOWER 5.1.21); see
`tests/data/power_flow/NOTICE.md`. Tolerances are absolute: 1e-8 p.u. on |V|, 1e-8 rad on
angles, 1e-6 p.u. on generator powers and DC branch flows.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from noodl.apps.electrical.power import (
    build_power_flow,
    power_flow_solution,
    read_matpower,
    solve_power_flow,
)

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
def test_fixture_belongs_to_the_committed_case_file(case):
    ref = _fixture(case)
    raw = (DATA / f"{case}.m").read_bytes().replace(b"\r\n", b"\n")
    assert ref["format"] == "noodl-power-flow-reference/1" and ref["case"] == case
    assert ref["case_sha256_lf"] == hashlib.sha256(raw).hexdigest()
    assert ref["engines"]["pandapower"] == "3.5.6" and ref["engines"]["PYPOWER"] == "5.1.21"
    assert max(ref["pandapower_vs_pypower_max_abs"].values()) < 1e-10


def _max_abs(a, b):
    return float((torch.as_tensor(a) - torch.tensor(b)).abs().max())


@pytest.mark.parametrize("case", CASES)
def test_ac_power_flow_matches_pandapower(case):
    ref = _fixture(case)["ac"]
    model, state, drivers = build_power_flow(read_matpower(DATA / f"{case}.m"))
    sol = power_flow_solution(model, solve_power_flow(model, state, drivers,
                                                      differentiable=False))
    assert list(sol.bus_ids) == ref["bus"] and list(sol.gen_buses) == ref["gen_bus"]
    assert _max_abs(sol.vm, ref["vm"]) < 1e-8
    assert _max_abs(sol.va, ref["va_rad"]) < 1e-8
    assert _max_abs(sol.pg, ref["pg_pu"]) < 1e-6
    assert _max_abs(sol.qg, ref["qg_pu"]) < 1e-6
