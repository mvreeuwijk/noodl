# Notes for coding agents

Follow [CONTRIBUTING.md](CONTRIBUTING.md): code, tests and documentation change together, and
`ruff check .` runs before every commit.

## Running the tests

The full suite takes about 45 minutes. The applications (`building_physics`, `sewer`,
`street_aq`, `water`) are large and independent of each other. While developing, run the core
tests plus the application you are changing. Run the full suite, with coverage, once, just
before opening a pull request.

Core (always include):

```bash
tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py tests/verification/test_newton_batched.py
```

Add the application you are changing:

| Application | Paths |
| --- | --- |
| building | `tests/apps/building_physics tests/verification/test_contam_parity.py tests/verification/test_modelica_parity.py tests/verification/test_airflow_closed_forms.py tests/verification/test_natural_ventilation.py` |
| sewer | `tests/apps/sewer tests/apps/inp tests/verification/test_sewer_*.py tests/benchmarks/test_swmm_corpus.py` |
| water | `tests/apps/water tests/apps/inp tests/verification/test_water_parity.py tests/verification/test_wsimod_*.py` |
| street air quality | `tests/apps/street_aq tests/verification/test_munich.py tests/verification/test_sirane*.py` |
| coupling | `tests/verification/test_coupling_*.py tests/verification/test_composed_*.py` |

For example, for the building application:

```bash
NOODL_PARITY_WORKERS=4 python -m pytest -q -n 2 --no-cov tests/test_*.py tests/layers tests/solvers tests/operators \
  tests/elements tests/components tests/apps/test_*.py tests/apps/building_physics \
  tests/verification/test_contam_parity.py tests/verification/test_modelica_parity.py \
  tests/verification/test_airflow_closed_forms.py tests/verification/test_natural_ventilation.py
```

- Pass `--no-cov` on a subset: the coverage threshold (90 %) applies to the whole suite only.
- Keep the number of test processes at 8 or below. The reference-model comparisons start their
  own worker pool inside each pytest worker, so the total is up to `-n` × `NOODL_PARITY_WORKERS`:
  use `-n 2` with `NOODL_PARITY_WORKERS=4`.
- Tests marked `slow` (full reference-model runs) are deselected by default; select them with
  `-m slow` when you change what they compare.
- Before the pull request: `ruff check .`, the full `pytest`, and `python -m mkdocs build --strict`.
