# Contributing to noodl physics

Thank you for improving noodl physics. A change is complete when its code, tests **and
documentation** are complete; the test suite enforces all three.

## Every change that adds or changes functionality updates the documentation

- **Docstrings.** Every public class, function and option has a docstring whose first
  paragraph says what it does. The [catalogue](docs/catalogue/core.md) is generated from
  them: run `python scripts/gen_catalogue.py` and commit the result. A parameter with a
  physical unit states it in the docstring; the catalogue is where users look it up.
- **Pages.** Update the application page (`docs/applications/`), and
  [Using noodl](docs/usage.md) if the common workflow changes. Examples are executed by
  `tests/test_docs_usage.py`; mark a snippet that is not meant to run on its own as `py`.
- **Names.** Identifiers say what they do physically (`temperature`, `roof_exchange`), not
  which tool they come from. When renaming, keep the old name as an alias so existing code
  keeps working.

`tests/test_docs_catalogue.py` fails when a public name has no docstring or the catalogue
is stale; `tests/test_docs_usage.py` fails when a documented example no longer runs or
prints something else.

## Before you push

```bash
ruff check .
pytest
python -m mkdocs build --strict
```

## Continuous integration

A push to a branch runs only the `branch` job, which takes a few minutes. A pull request, a push to `main` and a manual run run the other `tests` jobs. A newer push cancels the run still in progress on the same branch or pull request.

| Workflow / job | Runs on | What it runs |
| --- | --- | --- |
| `tests` / `branch` | every push to a branch other than `main`; Linux, Python 3.12 | `ruff check .`, the core tests, and the tests of every application the branch changed relative to `main` (`.github/scripts/select_tests.py`, the table in `AGENTS.md`); no coverage |
| `tests` / `test` | pull requests, pushes to `main`, manual runs; Linux, Python 3.12 | `ruff check .` and the full default suite (`-m "not slow"`) with coverage |
| `tests` / `minimum-versions` | pull requests, pushes to `main`; Linux, Python 3.11, torch 2.2, numpy 1.26 | the core subset from `AGENTS.md`, installed with `.[test]` (not `.[dev]`: the `reference` extra's `wntr` needs numpy>=2.2.6) |
| `tests` / `wheel-smoke` | pull requests, pushes to `main`; Linux | builds the wheel, installs it in a clean venv and imports every application outside the source tree |
| `tests` / `windows` | pull requests, pushes to `main`; Windows, Python 3.12 | the core subset, then every test in `tests/verification/test_contam_parity.py`, `slow` ones included, with `pip install .[contam]` (the `contamxpy` wheel exists for Windows only, so this is the only place the CONTAM parity tests run) |
| `tests` / `macos` | pull requests, pushes to `main`; macOS, Python 3.12 | the core subset |
| `slow-reference` / `slow` | weekly (Monday 03:00 UTC), manual `workflow_dispatch`, and pull requests labelled `run-slow` | `pytest -m slow` over all of `tests/` (full reference-model comparisons and long simulations) |

Test extras: `pip install -e ".[dev]"` installs everything (`test` plus `reference`). Floats in the
golden and pinned-value tests (`tests/test_golden_orchestration.py`, `test_pcg_results_are_pinned`)
are compared bit-exactly when `NOODL_GOLDEN_EXACT=1` and to rtol 1e-12 otherwise, because BLAS and
vectorised reductions reassociate sums differently per platform (the golden file was recorded on a
machine that none of the CI runners reproduces bit for bit). CI leaves it unset; set it locally, on
the recording machine, to check that a refactor changed no number.

Skips are visible: every job runs pytest with `-rs`, writes a pass/fail/skip table with the skip
reasons to the job summary (`.github/scripts/junit_summary.py`), and the `slow-reference` job
uploads its junit XML and skip list as the `slow-reference-report` artifact. MATPOWER cases are
cached in `~/.cache/noodl`. To run the slow comparisons for your pull request, add the
`run-slow` label.

A new application also follows the checklist in
[Extending noodl](docs/development/extending.md#what-a-real-application-must-ship).
