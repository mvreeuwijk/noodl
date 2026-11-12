"""WSIMOD 0.8.1 constants used by the core port (``wsimod/core/constants.py``)."""

from __future__ import annotations

MAXITER = 5  # constants.py:57, maximum iterations of a while loop
UNBOUNDED_CAPACITY = 1e15  # constants.py:18
DT_DAYS = 1  # constants.py:20
DECAY_REFERENCE_TEMPERATURE = 20  # constants.py:66, degC: the exponents are relative to it
FLOAT_ACCURACY = 1e-11  # constants.py:17, default; per model in the port
MG_L_TO_KG_M3 = 1e-3  # constants.py:61

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

# land.py:375-380, :540-576: the surface data the deposition functions read (D8)
SURFACE_DEPOSITION = ("nhx-dry", "noy-dry", "srp-dry", "nhx-wet", "noy-wet", "srp-wet")
# land.py:1167-1172, :1649-1710: the fertiliser and manure data a growing surface reads (3c, later)
SURFACE_INPUTS = (
    "nhx-fertiliser",
    "noy-fertiliser",
    "srp-fertiliser",
    "nhx-manure",
    "noy-manure",
    "srp-manure",
)
# land.py:545-548, :573-576: the pollutant each deposition variable feeds
DEPOSITION_SPECIES = {"nhx": "ammonia", "noy": "nitrate", "srp": "phosphate"}
