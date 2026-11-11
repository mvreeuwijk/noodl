"""EPANET 2.2's unit constants and conversions, shared by the reader, the laws, the tanks and
the energy. A leaf module: it imports nothing of the application.

EPANET computes in feet and cfs whatever the file says, converting at the boundary
(`input1.c`, `convertunits`; `types.h`); this application computes in SI, and these are the
numbers it reproduces EPANET's own conversions with.
"""

from __future__ import annotations

#: EPANET's ``MperFT`` (`types.h`): metres per foot, exact.
M_PER_FT = 0.3048

#: EPANET's kilowatts per horsepower (`types.h`, ``KWperHP``), by which it divides the power
#: of an SI file (`input1.c`, `convertunits`) and multiplies the energy (`getenergy`).
KW_PER_HP = 0.7457

#: EPANET's constant of a constant-power pump and of `getenergy`: the head (ft) one
#: horsepower lifts one cfs, ``550 / 62.4`` (`input2.c`, `updatepumpparams`; `hydraul.c`,
#: `getenergy`, horsepower = ``dh q SpGrav / 8.814``).
HP_HEAD_FT_CFS = 8.814

#: EPANET's `TINY` (`types.h`): the tolerance of its pump-curve checks (`powercurve`) and the
#: width below which `interp` (`project.c`) takes a curve segment's end value. EPANET applies
#: it in the file's units; each user says in which units it applies it here.
TINY = 1e-6

#: `.inp` flow units -> m3/s. GPM is one US gallon per minute; the rest follow the manual's
#: Appendix C.2.11 list. `GPM` is exactly `3.785411784e-3 / 60 = 6.309019640343977e-05`,
#: which reproduces wntr's own conversion of Net1's 1500 GPM design point to
#: 0.0946352946 m3/s (measured).
FLOW_UNITS = {
    "CFS": 0.028316846592,
    "GPM": 3.785411784e-3 / 60.0,
    "MGD": 3.785411784e3 / 86400.0,
    "IMGD": 4.54609e3 / 86400.0,
    "AFD": 1233.48183754752 / 86400.0,
    "LPS": 1e-3,
    "LPM": 1e-3 / 60.0,
    "MLD": 1e3 / 86400.0,
    "CMH": 1.0 / 3600.0,
    "CMD": 1.0 / 86400.0,
}
#: EPANET's own flow-unit factors, file units per cfs (`types.h`, ``GPMperCFS`` ...).
#: Several are rounded (``LPSperCFS = 28.317`` against the exact 28.3168466), and EPANET
#: divides a file's flows by exactly these numbers on the way in.
EPANET_QCF = {
    "CFS": 1.0,
    "GPM": 448.831,
    "MGD": 0.64632,
    "IMGD": 0.5382,
    "AFD": 1.9837,
    "LPS": 28.317,
    "LPM": 1699.0,
    "MLD": 2.4466,
    "CMH": 101.94,
    "CMD": 2446.6,
}


def rounded_cfs_per_m3s(flow_units: str | None) -> float | None:
    """EPANET's internal cfs per m3/s for a file in `flow_units`: the file's flow unit in
    m3/s (`FLOW_UNITS`) divided by EPANET's own, rounded, file units per cfs (`EPANET_QCF`);
    `None` for no file unit (the exact ``1 / 0.3048^3`` then)."""
    if flow_units is None:
        return None
    return 1.0 / (FLOW_UNITS[flow_units] * EPANET_QCF[flow_units])
