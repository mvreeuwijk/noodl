"""The component catalogue (`docs/catalogue/`) cannot fall behind the code.

Adding or changing a public class, function or option means updating its docstring and
re-running `python scripts/gen_catalogue.py`; these tests fail until both are done.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_catalogue", ROOT / "scripts" / "gen_catalogue.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_every_public_name_and_option_is_documented():
    assert gen.undocumented() == [], (
        "public names without a docstring (or options without a description) -- document "
        "them, then run `python scripts/gen_catalogue.py`"
    )


def test_the_committed_catalogue_matches_the_code():
    stale = [
        path.name for path, text in gen.generate().items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    assert stale == [], (
        f"docs/catalogue is out of date for {stale}; run `python scripts/gen_catalogue.py`"
    )


def test_every_catalogue_page_is_in_the_navigation():
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    missing = [p.name for p in gen.generate() if f"catalogue/{p.name}" not in nav]
    assert missing == []
