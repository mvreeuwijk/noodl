"""The documentation's examples run: every `python` block of the pages below executes, in
order, in one namespace per page, and each prints exactly the `text` block that follows it.

Illustrative code that is not meant to run on its own (a signature, a loop outline) is
marked `py` instead of `python` and is not executed. Each page runs in a temporary
directory holding the fixture files its examples name (`tree_steady.inp`,
`twoloop_si.inp`, `quickstart_topology.json`, ...), so an example may read or write files
by bare name, as a user would.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "usage.md",
    "concepts/components.md",
    "applications/building_physics.md",
    "applications/street_aq.md",
    "applications/sewer.md",
    "applications/water.md",
    "applications/allocation.md",
]
FIXTURES = ["sewer/*.inp", "water/*.inp", "wsimod/*.json"]
BLOCK = re.compile(r"```(python|text)\n(.*?)```", re.S)


@pytest.mark.parametrize("page", PAGES)
def test_the_page_runs_and_prints_what_it_shows(page, tmp_path, monkeypatch):
    for pattern in FIXTURES:
        for f in (ROOT / "tests" / "data").glob(pattern):
            shutil.copy(f, tmp_path / f.name)
    monkeypatch.chdir(tmp_path)
    path = ROOT / "docs" / page
    blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
    assert blocks, f"{page} has no python blocks"
    namespace: dict = {}
    pending: str | None = None
    for kind, body in blocks:
        if kind == "python":
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(body, str(path), "exec"), namespace)  # noqa: S102 - our own docs
            pending = out.getvalue()
        elif pending is not None:
            assert pending.strip() == body.strip(), f"{page}: printed output differs"
            pending = None
