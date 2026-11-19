# Testing and building the docs

## Running the tests

From a source checkout:

```bash
pip install -e ".[dev]"
pytest -q
```

The full suite takes about 45 minutes, because the applications are large. While developing, run
the core tests plus the application you are changing. The core tests are:

```bash
tests/test_*.py tests/layers tests/solvers tests/operators tests/elements tests/components tests/apps/test_*.py
```

Add the paths of your application (the same table drives CI's `branch` job, in
`.github/scripts/select_tests.py`, so keep the two in step):

| Application | Paths |
| --- | --- |
| building | `tests/apps/building_physics` and the CONTAM, Modelica, airflow and natural-ventilation verification files |
| sewer | `tests/apps/sewer`, `tests/apps/inp`, `tests/verification/test_sewer_*.py`, `tests/benchmarks/test_swmm_corpus.py` |
| water | `tests/apps/water`, `tests/apps/inp`, and the water parity, network and quality verification files |
| wsimod | `tests/apps/wsimod` and the allocation and WSIMOD verification files |
| street air quality | `tests/apps/street_aq` and the MUNICH and SIRANE verification files |
| electrical | `tests/apps/electrical`, the MNA layer, model and operator tests, and the power-flow and Modelica electrical verification files |
| coupling | `tests/verification/test_coupling_*.py`, `tests/verification/test_composed_*.py` |

For example, for the sewer application:

```bash
python -m pytest -q -n 4 --no-cov tests/test_*.py tests/layers tests/solvers tests/operators \
  tests/elements tests/components tests/apps/test_*.py tests/apps/sewer tests/apps/inp \
  tests/verification/test_sewer_*.py tests/benchmarks/test_swmm_corpus.py
```

- Pass `--no-cov` on a subset: the 90 % coverage threshold applies to the whole suite only.
- Tests marked `slow` (full reference-model runs) are deselected by default; select them with
  `-m slow` when you change what they compare.
- Tests needing an external engine carry the `external` marker and skip themselves when the
  engine is not installed.
- No test asserts a wall-clock time or a memory figure. Performance budgets and scaling checks
  live in `benchmarks/` (`python -m benchmarks.report_composed_scaling`, `--quick` for a short
  run).

## What CI runs

| Job | Runs on | What it runs |
| --- | --- | --- |
| `branch` | every push to a branch other than `main` | `ruff check .`, the core tests and the tests of each application the branch changed; no coverage |
| `test` | pull requests, pushes to `main` | `ruff check .` and the full default suite with coverage |
| `minimum-versions` | pull requests, pushes to `main` | the core tests on the oldest supported Python, torch and NumPy |
| `wheel-smoke` | pull requests, pushes to `main` | builds the wheel and imports every application from a clean install |
| `windows`, `macos` | pull requests, pushes to `main` | the core tests; on Windows also the CONTAM comparisons, which need the Windows-only `contamxpy` wheel |
| `slow` | weekly, manually, and on pull requests labelled `run-slow` | `pytest -m slow` over all tests |

Before opening a pull request run `ruff check .`, the full `pytest` and
`python -m mkdocs build --strict`. See [Contributing](contributing.md).

## Documentation examples

Every `python` block on the pages listed in `tests/test_docs_usage.py` is executed, in order, and
must print exactly the `text` block that follows it. Use `py` for code that is illustrative. A
page with executed blocks is added to `PAGES` in that file; fixture files the examples read go
under `tests/data/` and are matched by `FIXTURES`.

## Building the documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

The API reference is generated from the package's docstrings, so noodl physics must be installed
in the same environment. `python -m mkdocs build --strict` fails on any broken link.
