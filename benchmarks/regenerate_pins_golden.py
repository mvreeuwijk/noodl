"""Regenerate `tests/golden/potential_layer.json`, the bitwise pin of both `KirchhoffLayer`
solve paths (the fast path and the general path).

Run it from the repository root, on code whose behaviour is the reference:

    python benchmarks/regenerate_pins_golden.py

and commit the result only for a deliberate change of behaviour, with the controller's
approval; a refactor must leave the file untouched. The cases live in
`tests/layers/test_potential_layer_pins.py`, which compares against this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.golden import save_golden  # noqa: E402
from tests.layers import test_potential_layer_pins as pins  # noqa: E402


def main() -> None:
    cases = pins.build_all()
    meta = {
        "exact": sorted(cases),
        "tolerance_1e-12": [],
        "movers": list(pins.MOVERS),
    }
    # the same exact round trip the tests compare against
    save_golden("potential_layer", json.loads(json.dumps({"cases": cases, "meta": meta})))
    errors = [n for n, v in cases.items() if "error" in v]
    print(f"wrote {len(cases)} cases; error outcomes: {errors}")
    for n in errors:
        print(n, cases[n]["error"])


if __name__ == "__main__":
    main()
