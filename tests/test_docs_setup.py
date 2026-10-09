"""`docs/concepts/setup.md` is runnable: its Python blocks execute in order, in one
namespace, and each prints exactly the `text` block that follows it."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "docs" / "concepts" / "setup.md"
BLOCK = re.compile(r"```(python|text)\n(.*?)```", re.S)


def test_the_setup_page_runs_and_prints_what_it_shows():
    blocks = BLOCK.findall(PAGE.read_text(encoding="utf-8"))
    assert blocks and blocks[0][0] == "python"
    namespace: dict = {}
    pending: str | None = None
    for kind, body in blocks:
        if kind == "python":
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(body, str(PAGE), "exec"), namespace)  # noqa: S102 - our own docs
            pending = out.getvalue()
        else:
            assert pending is not None, "a text block must follow a python block"
            assert pending.strip() == body.strip()
            pending = None
