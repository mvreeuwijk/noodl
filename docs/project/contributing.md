# Contributing

Contributions are welcome. The full rules are in
[CONTRIBUTING.md](https://github.com/mvreeuwijk/noodl/blob/main/CONTRIBUTING.md); this page
summarises them.

A change is complete when its code, tests and documentation are complete:

- **Docstrings.** Every public class, function and option has a docstring whose first paragraph
  says what it does, with units for physical parameters. The
  [catalogue](../reference/catalogue/index.md) is generated from them with
  `python scripts/gen_catalogue.py`; commit the result.
- **Pages.** Update the application page, and the [user guide](../guide/workflow.md) if the
  common workflow changes. Examples on the pages are executed by the test suite: mark a snippet
  that is not meant to run as `py`.
- **Names.** Identifiers say what they do physically, not which tool they come from. While
  noodl is pre-1.0, a rename changes every use rather than keeping an alias. The exception is the
  name of an external reference relation (for example `MBLPowerLaw`), kept as a secondary alias
  of the physical name.
- **Before you push.** Run `ruff check .`, `pytest` and `python -m mkdocs build --strict`.

See [Testing](testing.md) for what CI runs and how to run a subset, and
[Extending noodl](extending/index.md) for adding a law, an application or a reader, including
what a new application must ship.
