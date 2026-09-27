"""Every documentation figure has an up-to-date dark-theme twin (scripts/make_dark_figures.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_dark_figures.py"


def test_dark_figures_are_current():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
