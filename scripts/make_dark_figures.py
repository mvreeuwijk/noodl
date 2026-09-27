"""Generates the dark-theme twin of every documentation figure under docs/assets/.

Run manually after editing a figure: `.venv/Scripts/python scripts/make_dark_figures.py`.
Pass `--check` to report stale or missing twins without writing (the test suite does this).

The hand-written SVG figures share one light palette. Each `name.svg` gets a `name-dark.svg`
with every palette colour swapped for its dark-theme counterpart, and nothing else changed.
`docs/javascripts/theme-figures.js` shows the twin whenever the slate (dark) scheme is active.
A figure that uses a colour outside the palette is an error, so a new colour cannot slip
through unmapped. The logo files (noodl-physics-*) are not figures and are left alone.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

# Light colour -> dark colour. Keys are upper-case, as written in the figures.
PALETTE = {
    "#F7F9FA": "#262A33",  # figure background
    "#FFFFFF": "#2E333D",  # box fill
    "#DCE3E8": "#4A525E",  # hairlines, neutral box borders
    "#D8F0ED": "#1E4541",  # tinted teal fill
    "#1E3A52": "#E3EAF0",  # primary text and lines (navy)
    "#7C8A99": "#A3AFBC",  # secondary text and lines
    "#17A398": "#2EC4B6",  # accent (teal)
    "#C2703D": "#E68F57",  # second accent (orange)
}

_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def light_figures() -> list[Path]:
    # The noodl-physics-* logo files ship their own light and dark versions.
    return sorted(
        p
        for p in ASSETS.glob("*.svg")
        if not p.stem.endswith("-dark") and not p.name.startswith("noodl-physics")
    )


def darken(svg: str, name: str) -> str:
    def swap(match: re.Match[str]) -> str:
        colour = match.group(0).upper()
        if colour not in PALETTE:
            raise ValueError(f"{name}: colour {colour} is not in the figure palette")
        return PALETTE[colour]

    return _HEX.sub(swap, svg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale twins, write nothing")
    args = parser.parse_args()

    stale = []
    for light in light_figures():
        dark = light.with_name(f"{light.stem}-dark.svg")
        text = darken(light.read_text(encoding="utf-8"), light.name)
        if dark.exists() and dark.read_text(encoding="utf-8") == text:
            continue
        stale.append(dark.name)
        if not args.check:
            dark.write_text(text, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("stale or missing dark figures: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(f"{'checked' if args.check else 'wrote'} {len(stale)} dark figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
