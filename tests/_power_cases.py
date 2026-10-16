"""Power-flow networks for tests.

`power_case` returns one of MATPOWER's standard cases, fetched on first use; the calling test
is skipped without a network connection and carries the `external` marker. `four_bus` is the
small network of the power-flow docs, written for noodl physics, for tests that need some
network rather than a MATPOWER case.
"""

from pathlib import Path

import pytest

from noodl.apps.electrical.power import (
    Branch,
    Bus,
    FixedShunt,
    Generator,
    Load,
    MatpowerCaseUnavailableError,
    PowerNetwork,
    matpower_case,
)


def power_case(name: str) -> Path:
    """The cached, verified MATPOWER case `name`; skips the calling test without a network."""
    try:
        return matpower_case(name)
    except MatpowerCaseUnavailableError as exc:
        pytest.skip(f"MATPOWER case not available: {exc}")


def four_bus() -> PowerNetwork:
    """A 132 kV triangle (buses 1-3; 1 the slack at 1.02 p.u., 2 a PV bus at 1.01 p.u. making
    50 MW) feeding the 33 kV bus 4 through a transformer with tap 0.975 (branch row 4, r = 0),
    with loads on buses 2-4 and an 8 MVAr capacitor bank on bus 4."""
    return PowerNetwork(
        buses=(Bus(1, "slack"), Bus(2, "pv"), Bus(3, "pq"), Bus(4, "pq")),
        branches=(Branch(1, 2, r=0.01, x=0.06, b=0.03), Branch(1, 3, r=0.02, x=0.08, b=0.04),
                  Branch(2, 3, r=0.015, x=0.07, b=0.02),
                  Branch(3, 4, r=0.0, x=0.1, b=0.0, ratio=0.975)),
        generators=(Generator(1, 0.0, vm_set=1.02), Generator(2, 50.0, vm_set=1.01)),
        loads=(Load(2, 20.0, 5.0), Load(3, 60.0, 20.0), Load(4, 45.0, 15.0)),
        shunts=(FixedShunt(4, 0.0, 8.0),),
        base_mva=100.0,
        name="four_bus",
    )
