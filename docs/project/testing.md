# Testing and building the docs

## Running the test suite

To run the test suite from a source checkout:

```bash
pip install -e ".[dev]"
pytest -q
```

Slow tests — the long reference comparisons — are excluded by default and run with
`pytest -m slow`. No test asserts a wall-clock time or a memory figure, since both depend on the
machine; the performance budgets and scaling checks are reported by the scripts in `benchmarks/`
(`python -m benchmarks.report_composed_scaling`, `--quick` for a short run). Tests needing an external engine carry the `external` marker and skip
themselves when that engine is unavailable.

## Building the documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

The API reference is generated from the package's own docstrings, so noodl physics must be installed in
the same environment for that section to build.
