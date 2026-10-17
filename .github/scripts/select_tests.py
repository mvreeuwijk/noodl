"""The test paths a branch push runs: the core tests plus every application the push touched.

Reads the changed files (one per line, repository-relative) on stdin and prints the pytest
paths, space-separated. The applications and their test paths are the table in AGENTS.md;
keep the two in step. A change to `pyproject.toml` (dependencies, pytest options) selects
every application. Shared core code selects only the core tests: an application it breaks
is caught by the full run on the pull request.

    git diff --name-only origin/main...HEAD | python .github/scripts/select_tests.py
"""

import sys

CORE = [
    "tests/test_*.py", "tests/layers", "tests/solvers", "tests/operators", "tests/elements",
    "tests/components", "tests/apps/test_*.py",
]

#: Application: (prefixes of the files that belong to it, its test paths).
APPS = {
    "building": (
        ["src/noodl/apps/building_physics/", "src/noodl/modelica/",
         "tests/apps/building_physics/", "tests/data/contam/", "tests/data/modelica/",
         "tests/verification/test_contam_parity.py", "tests/verification/test_modelica_parity.py",
         "tests/verification/test_airflow_closed_forms.py",
         "tests/verification/test_natural_ventilation.py"],
        ["tests/apps/building_physics", "tests/verification/test_contam_parity.py",
         "tests/verification/test_modelica_parity.py",
         "tests/verification/test_airflow_closed_forms.py",
         "tests/verification/test_natural_ventilation.py"],
    ),
    "sewer": (
        ["src/noodl/apps/sewer/", "src/noodl/apps/inpfile.py", "tests/apps/sewer/",
         "tests/apps/inp/", "tests/data/sewer/", "tests/verification/test_sewer_",
         "tests/benchmarks/test_swmm_corpus.py"],
        ["tests/apps/sewer", "tests/apps/inp", "tests/verification/test_sewer_*.py",
         "tests/benchmarks/test_swmm_corpus.py"],
    ),
    "water": (
        ["src/noodl/apps/water/", "src/noodl/apps/wsimod/", "src/noodl/apps/inpfile.py",
         "tests/apps/water/", "tests/apps/inp/", "tests/data/water/", "tests/data/wsimod/",
         "tests/verification/test_water_parity.py", "tests/verification/test_wsimod_",
         "tests/verification/_wsimod_reference.py"],
        ["tests/apps/water", "tests/apps/inp", "tests/verification/test_water_parity.py",
         "tests/verification/test_wsimod_*.py"],
    ),
    "electrical": (
        ["src/noodl/apps/electrical/", "src/noodl/modelica/", "tests/apps/electrical/",
         "tests/data/modelica_electrical/", "tests/data/power_flow/",
         "tests/verification/test_power_flow_parity.py",
         "tests/verification/test_modelica_electrical_parity.py"],
        ["tests/apps/electrical", "tests/verification/test_power_flow_parity.py",
         "tests/verification/test_modelica_electrical_parity.py"],
    ),
    "street": (
        ["src/noodl/apps/street_aq/", "tests/apps/street_aq/", "tests/data/street/",
         "tests/verification/test_munich.py", "tests/verification/test_sirane"],
        ["tests/apps/street_aq", "tests/verification/test_munich.py",
         "tests/verification/test_sirane*.py"],
    ),
    "coupling": (
        ["src/noodl/couple.py", "benchmarks/", "tests/verification/test_coupling_",
         "tests/verification/test_composed_"],
        ["tests/verification/test_coupling_*.py", "tests/verification/test_composed_*.py"],
    ),
}


def select(changed: list[str]) -> list[str]:
    """The core paths plus the test paths of every application a changed file belongs to."""
    everything = "pyproject.toml" in changed
    paths = list(CORE)
    for prefixes, tests in APPS.values():
        if everything or any(f.startswith(p) for f in changed for p in prefixes):
            paths += [t for t in tests if t not in paths]
    return paths


if __name__ == "__main__":
    print(" ".join(select([line.strip() for line in sys.stdin if line.strip()])))
