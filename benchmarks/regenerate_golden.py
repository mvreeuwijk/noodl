"""Regenerate the stored golden results from the current implementation.

Writes the stored golden results under `tests/golden/`, among them
`natural_ventilation.json` (the coupled airflow-heat demo trajectory).

This is an explicit action, not run automatically: run it only after an intentional change
to element laws, coupling or solver defaults that is expected to change the reference
numbers.

    .venv/Scripts/python benchmarks/regenerate_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
# `tests` for `golden`, and the repository root so `benchmarks.natural_ventilation` imports
# whether this file is run as a path (sys.path[0] is `benchmarks/`) or as `-m`.
sys.path.insert(0, str(_REPO_ROOT / "tests"))
sys.path.insert(0, str(_REPO_ROOT))

from golden import save_golden  # noqa: E402

from benchmarks.natural_ventilation import run as run_natural_ventilation  # noqa: E402

DTYPE = torch.float64


def _natural_ventilation_case() -> dict:
    """Two simulated hours (12 steps of 600 s) of the coupled airflow-heat demo, `iterate`.

    `benchmarks.natural_ventilation.run` is deterministic -- no RNG anywhere in it -- so this
    is a genuine bit-for-bit regression reference for the whole building stack at once:
    `build_model`, the stack drive, the density closure, the large opening, and the iterated
    coupling of the air and thermal layers, step by step.
    """
    result = run_natural_ventilation("iterate", 600.0, hours=2.0)
    return {key: result[key] for key in ("T_A", "T_B", "door_kg_s")}


def _munich_idealised_case() -> dict:
    """The model's own 12x6 answer on the published idealised case, so that a change in
    the routing, the direction averaging or the canyon wind shows up as a diff."""
    from tests.verification.test_munich import _fixture, _run

    out, _names = _run(_fixture())
    return {panel: {street: float(value) for street, value in row.items()}
            for panel, row in out.items()}


def _sewer_tree_case() -> dict:
    """Steady water, air and quality on the committed sewer fixture."""
    from noodl.apps.sewer import network as sewer_network
    from noodl.apps.sewer.network import sewer_steady

    model, state, drivers = sewer_network.build_model(sewer_network.tree_steady())
    final = sewer_steady(model, state, drivers)
    return {key: value.flatten().tolist() for key, value in sorted(final.items())}


def _water_twoloop_case() -> dict:
    """Steady heads and flows on the committed two-loop water fixture.

    Built from `twoloop()` rather than from the `.inp`, so regenerating the golden does not
    depend on the reader; `test_water_parity.py` pins that the two agree.
    """
    from noodl.apps.water import network as water_network
    from noodl.apps.water.network import water_steady

    model, state, drivers = water_network.build_model(water_network.twoloop())
    final = water_steady(model, state, drivers)
    return {key: value.flatten().tolist() for key, value in sorted(final.items())}


def main() -> None:
    save_golden("natural_ventilation", _natural_ventilation_case())
    print("wrote tests/golden/natural_ventilation.json")
    save_golden("munich_idealised", _munich_idealised_case())
    print("wrote tests/golden/munich_idealised.json")
    save_golden("water_twoloop", _water_twoloop_case())
    print("wrote tests/golden/water_twoloop.json")
    save_golden("sewer_tree", _sewer_tree_case())
    print("wrote tests/golden/sewer_tree.json")


if __name__ == "__main__":
    main()
