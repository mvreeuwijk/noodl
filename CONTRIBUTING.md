# Contributing to noodl physics

Thank you for improving noodl physics. A change is complete when its code, tests **and
documentation** are complete; the test suite enforces all three.

## Every change that adds or changes functionality updates the documentation

- **Docstrings.** Every public class, function and option has a docstring whose first
  paragraph says what it does. The [catalogue](docs/catalogue/core.md) is generated from
  them: run `python scripts/gen_catalogue.py` and commit the result.
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

A new application also follows the checklist in
[Extending noodl](docs/development/extending.md#what-a-real-application-must-ship).
