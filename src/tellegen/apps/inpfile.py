"""Section-keyed tokenizer shared by the SWMM and EPANET `.inp` readers.

Both formats share exactly one lexical layer -- `[SECTION]` headers, `;` comments,
whitespace-separated fields, blank lines -- and nothing above it. This module is that layer
and no more: it knows no section names, no column layouts and no units. `apps/sewer/inp.py`
and `apps/water/inp.py` each own their own.

Every `InpLine` carries the file's OWN 1-based line number, so a reader's refusal can name
the line the way `apps/building/prj.py` names a CONTAM record. Section names are upper-cased
and a repeated section is MERGED (EPANET's own example files split `[REACTIONS]` in two).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InpLine:
    """One content line of a section: where it came from and what it holds."""

    section: str
    number: int
    fields: tuple[str, ...]
    raw: str


def read_sections(path, *, comment: str = ";") -> dict[str, list[InpLine]]:
    """`{SECTION: [InpLine, ...]}` for the file at `path`, comments and blanks removed.

    A line before the first `[SECTION]` header is REFUSED naming the line: it would
    otherwise be dropped silently, and a stray record is exactly the kind of thing a
    hand-edited `.inp` grows.
    """
    path = Path(path)
    sections: dict[str, list[InpLine]] = {}
    current: str | None = None
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.split(comment, 1)[0].strip()
        if not text:
            continue
        if text.startswith("["):
            # FR-4: exactly one bracket pair -- `text.endswith("]")` alone let a stray
            # inner `[` through (`[JUNC[TIONS]` read as the section name `JUNC[TIONS`,
            # surfacing later as a confusing "unknown section" instead of naming the
            # malformed header itself).
            if (
                not text.endswith("]") or len(text) < 3
                or text.count("[") != 1 or text.count("]") != 1
            ):
                raise ValueError(
                    f"{path}: line {number}: a section header must read '[NAME]' (exactly "
                    f"one bracket pair), got {raw.strip()!r}"
                )
            current = text[1:-1].strip().upper()
            sections.setdefault(current, [])
            continue
        if current is None:
            # FR-5: punctuation after the line number, matching every other refusal in this
            # module (`require_fields`, `as_float`).
            raise ValueError(
                f"{path}: line {number}: content {raw.strip()!r} appears before any "
                f"[section] header"
            )
        sections[current].append(
            InpLine(section=current, number=number, fields=tuple(text.split()), raw=raw)
        )
    return sections


def require_fields(line: InpLine, n: int, what: str, path) -> None:
    """Refuse a short record, naming the file, the line and what was expected."""
    if len(line.fields) < n:
        raise ValueError(
            f"{path}: line {line.number} of [{line.section}]: {what} needs {n} fields, "
            f"got {len(line.fields)} ({line.raw.strip()!r})"
        )


def as_float(line: InpLine, i: int, what: str, path) -> float:
    """Field `i` of `line` as a float, naming the file, the line and the field on failure."""
    try:
        return float(line.fields[i])
    except (IndexError, ValueError) as exc:
        got = line.fields[i] if i < len(line.fields) else "<missing>"
        raise ValueError(
            f"{path}: line {line.number} of [{line.section}]: {what} must be a number, "
            f"got {got!r}"
        ) from exc
