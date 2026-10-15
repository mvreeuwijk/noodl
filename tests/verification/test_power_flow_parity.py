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

DATA = Path(__file__).resolve().parents[1] / "data" / "power_flow"
CASES = ["case9", "case14", "case30", "case118", "case9_features"]


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
