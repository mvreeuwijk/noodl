"""User-facing documentation carries no internal tracking vocabulary.

Every page under `docs/` except `plans/`, `superpowers/`, `project/` (maintainer notes) and
`reference/verification/` (evidence pages, which may name test files) is scanned in its prose:
fenced code blocks, inline code and `$...$` / `$$...$$` maths are removed first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
EXCLUDED = ("plans", "superpowers", "project", "reference/verification")

RULES = {
    "tracking code": re.compile(r"\b[DWXYLS]\d{1,3}\b"),
    "milestone name": re.compile(r"\b(?:sub-projects?|3a|3b|3c)\b", re.IGNORECASE),
    "issue or PR number": re.compile(r"\b(?:issues?|PRs?|pull requests?) #\d+", re.IGNORECASE),
    "tests path": re.compile(r"\btests/"),
    "revision note": re.compile(r"under revision", re.IGNORECASE),
    "machine note": re.compile(r"development machine", re.IGNORECASE),
}


INLINE_CODE = re.compile(r"(`+).+?(?:\1)")


def _pages()-> list[Path]:
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS).as_posix()
        if not any(rel == e or rel.startswith(e + "/") for e in EXCLUDED):
            pages.append(path)
    return pages


def _prose(text: str) -> list[tuple[int, str]]:
    """The (line number, line) pairs of prose, with code and maths blanked out."""
    out = []
    fence = None
    in_math = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith("$$"):
            if not (stripped.endswith("$$") and len(stripped) > 2):
                in_math = not in_math
            continue
        if in_math:
            continue
        line = INLINE_CODE.sub("", line)
        line = re.sub(r"\$[^$]*\$", "", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)  # link targets are paths, not prose
        out.append((number, line))
    return out


@pytest.mark.parametrize("path", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_page_has_no_internal_vocabulary(path: Path) -> None:
    found = []
    for number, line in _prose(path.read_text(encoding="utf-8")):
        for name, pattern in RULES.items():
            match = pattern.search(line)
            if match:
                where = f"{path.relative_to(DOCS).as_posix()}:{number}"
                found.append(f"{where}: {name}: {match.group(0)!r}")
    assert not found, "\n".join(found)
