"""`PowerNetwork`: a balanced power-flow network in MATPOWER's file units.

Buses, branches, generators, loads and fixed shunts are frozen records in the units a case
file states them in: real power in MW, reactive power in MVAr, angles in degrees, voltage
magnitudes in p.u., and branch series resistance `r`, reactance `x` and total line charging
`b` in p.u. on the network's `base_mva`. The power-flow builders convert to per unit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

#: Bus kinds: a load bus (`"pq"`, real and reactive injection given), a voltage-controlled
#: bus (`"pv"`, real injection and voltage magnitude given) and the reference bus
#: (`"slack"`, voltage magnitude and angle given).
BUS_KINDS = ("pq", "pv", "slack")


@dataclass(frozen=True)
class Bus:
    """A bus: its id, kind (one of `BUS_KINDS`), voltage magnitude `vm` (p.u.) and angle
    `va_deg` (degrees) as stated in the file (the slack bus's angle is its setpoint), and
    its base voltage `base_kv` (kV, informative)."""

    id: int
    kind: str
    vm: float = 1.0
    va_deg: float = 0.0
    base_kv: float = 0.0


@dataclass(frozen=True)
class Branch:
    """A line or transformer from `from_bus` to `to_bus`: series resistance `r`, reactance
    `x` and total line charging susceptance `b` (p.u. on `base_mva`), off-nominal turns
    ratio `ratio` at the from end (0 means 1) and phase shift `shift_deg` (degrees)."""

    from_bus: int
    to_bus: int
    r: float
    x: float
    b: float
    ratio: float = 0.0
    shift_deg: float = 0.0
    in_service: bool = True


@dataclass(frozen=True)
class Generator:
    """A generator on `bus`: real output `p_mw` (MW), reactive output `q_mvar` (MVAr, a
    starting value), voltage magnitude setpoint `vm_set` (p.u.) and reactive limits
    (MVAr)."""

    bus: int
    p_mw: float
    q_mvar: float = 0.0
    vm_set: float = 1.0
    in_service: bool = True
    q_max_mvar: float = math.inf
    q_min_mvar: float = -math.inf


@dataclass(frozen=True)
class Load:
    """A constant-power load on `bus`, absorbing `p_mw` (MW) and `q_mvar` (MVAr)."""

    bus: int
    p_mw: float
    q_mvar: float


@dataclass(frozen=True)
class FixedShunt:
    """A fixed admittance from `bus` to ground, stated as the real power `g_mw` (MW) it
    consumes and the reactive power `b_mvar` (MVAr) it injects at a bus voltage of 1 p.u."""

    bus: int
    g_mw: float
    b_mvar: float


@dataclass(frozen=True)
class PowerNetwork:
    """A power-flow network: buses, branches, generators, loads and fixed shunts in file
    order, the system base `base_mva` (MVA), a `name` and free-form `notes` (a reader
    records what it skipped there). Construction refuses duplicate bus ids, an unknown bus
    kind, a nonpositive base and any element on a bus that does not exist."""

    buses: tuple[Bus, ...]
    branches: tuple[Branch, ...]
    generators: tuple[Generator, ...]
    loads: tuple[Load, ...] = ()
    shunts: tuple[FixedShunt, ...] = ()
    base_mva: float = 100.0
    name: str = ""
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_mva > 0:
            raise ValueError(f"base_mva must be positive, got {self.base_mva!r}")
        seen: set[int] = set()
        for k, bus in enumerate(self.buses):
            if bus.id in seen:
                raise ValueError(f"duplicate bus id {bus.id} (bus {k})")
            seen.add(bus.id)
            if bus.kind not in BUS_KINDS:
                raise ValueError(
                    f"bus {bus.id}: kind {bus.kind!r} is not one of {BUS_KINDS}"
                )
        for k, br in enumerate(self.branches):
            for end in (br.from_bus, br.to_bus):
                if end not in seen:
                    raise ValueError(
                        f"branch {k} (from bus {br.from_bus} to bus {br.to_bus}): "
                        f"bus {end} does not exist"
                    )
        for label, items in (
            ("generator", self.generators),
            ("load", self.loads),
            ("shunt", self.shunts),
        ):
            for k, item in enumerate(items):
                if item.bus not in seen:
                    raise ValueError(f"{label} {k}: bus {item.bus} does not exist")

    def bus_ids(self) -> tuple[int, ...]:
        """The bus ids in file order."""
        return tuple(bus.id for bus in self.buses)
