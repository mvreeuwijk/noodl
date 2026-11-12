"""WSIMOD 0.8.1 constants used by the core port (``wsimod/core/constants.py``)."""

from __future__ import annotations

MAXITER = 5  # constants.py:57, maximum iterations of a while loop
UNBOUNDED_CAPACITY = 1e15  # constants.py:18
DT_DAYS = 1  # constants.py:20
FLOAT_ACCURACY = 1e-11  # constants.py:17, default; per model in the port

# constants.py:21-57: WSIMOD 0.8.1's default pollutant lists.
POLLUTANTS: tuple[str, ...] = (
    "do",
    "org-phosphorus",
    "phosphate",
    "ammonia",
    "solids",
    "bod",
    "cod",
    "ph",
    "temperature",
    "nitrate",
    "nitrite",
    "org-nitrogen",
)
NON_ADDITIVE_POLLUTANTS: tuple[str, ...] = ("do", "temperature", "ph")
ADDITIVE_POLLUTANTS: tuple[str, ...] = (
    "org-phosphorus",
    "phosphate",
    "ammonia",
    "solids",
    "bod",
    "cod",
    "nitrate",
    "nitrite",
    "org-nitrogen",
)
