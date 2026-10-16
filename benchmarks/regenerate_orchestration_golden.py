"""Regenerate `tests/golden/orchestration.json`, the K0 characterization of the
Model / CoupledModel orchestration and of `KirchhoffLayer.solve`'s policy matrix.

Run it from the repository root, on code whose behaviour is the reference:

    python benchmarks/regenerate_orchestration_golden.py

and commit the result only for a deliberate change of behaviour; a refactor must leave the
file untouched. The cases live in `tests/test_golden_orchestration.py` and
`tests/layers/test_kirchhoff_solve_policy.py`, which compare against this file exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import test_golden_orchestration as orchestration  # noqa: E402
from tests.golden import save_golden  # noqa: E402
from tests.layers import test_kirchhoff_solve_policy as policy  # noqa: E402


def main() -> None:
    data = orchestration.build_all()
    data["solve_policy"] = policy.build_all()
    # the same exact round trip the tests compare against
    save_golden("orchestration", json.loads(json.dumps(data)))
    print(f"wrote {len(data) - 1} orchestration cases and "
          f"{sum(len(v) for v in data['solve_policy'].values())} solve-policy combinations")


if __name__ == "__main__":
    main()
