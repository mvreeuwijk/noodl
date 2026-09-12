"""Golden-file helpers: JSON storage of reference numeric results for regression tests."""

import json
from pathlib import Path
from typing import Any

_GOLDEN_DIR = Path(__file__).parent


def _path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict[str, Any]:
    """Load the golden data stored under `name`; raises FileNotFoundError if absent."""
    with _path(name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_golden(name: str, data: dict[str, Any]) -> None:
    """Overwrite the golden data stored under `name` with `data`."""
    with _path(name).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
