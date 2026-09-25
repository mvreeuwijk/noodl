"""Building element helpers on top of the core elements. Mass-flow convention (kg/s)."""

from __future__ import annotations

import math

import torch

from noodl.elements import PowerLaw
from noodl.topology import Network

RHO_0 = 1.2041

# The building application is float64 THROUGHOUT: a stack head is a
# difference of two ~1e5 Pa hydrostatic terms, so a float32 element coefficient would cost
# the very digits the natural-ventilation cases are asserted on. `Element._param` converts a
# bare Python float at `torch.get_default_dtype()` (float32 in this project), so every float
# that becomes an element parameter is converted HERE, explicitly, before it gets there.
# Coefficients derived from a `Network` instead take that network's own dtype.
_DTYPE = torch.float64


def mass_orifice(Cd, A, *, rho: float = RHO_0, dp_transition: float = 1e-3,
                 kind: str = "airpath", learnable: bool = False) -> PowerLaw:
    """F = Cd A sqrt(2 rho dp) [kg/s]: `PowerLaw(C = Cd A sqrt(2 rho), n = 0.5)`."""
    C = (torch.as_tensor(Cd, dtype=_DTYPE) * torch.as_tensor(A, dtype=_DTYPE)
         * math.sqrt(2.0 * rho))
    n = torch.as_tensor(0.5, dtype=C.dtype)
    return PowerLaw(C, n, dp_transition=dp_transition, kind=kind, learnable=learnable)


def add_large_opening(net: Network, a, b, *, H: float, W: float, z_mid: float,
                      Cd: float = 0.78, kind: str = "airpath"):
    """CONTAM's two-opening doorway (DR_PL2, TN 1887r1 eq. 69-70): two orifices of area
    W H / 2 at z_mid -/+ 2H/9, both oriented a -> b. Exact for a mid-height neutral plane.
    Returns the two edge keys (low, high). Edge attributes: z_path, area, Cd, opening=1."""
    if not H > 0 or not W > 0:
        raise ValueError(f"large opening {a!r}->{b!r}: H and W must be positive, got H={H}, W={W}")
    area = W * H / 2.0
    lo = net.add_edge(a, b, kind=kind, z_path=z_mid - 2.0 * H / 9.0, area=area, Cd=Cd, opening=1.0)
    hi = net.add_edge(a, b, kind=kind, z_path=z_mid + 2.0 * H / 9.0, area=area, Cd=Cd, opening=1.0)
    return lo, hi


def orifice_elements_from_edges(net: Network, kind: str, *, rho: float = RHO_0,
                                dp_transition: float = 1e-3, learnable: bool = False) -> PowerLaw:
    """One `PowerLaw` over every edge of `kind`, C_e = Cd_e A_e sqrt(2 rho), from the edge
    attributes `Cd` and `area` (KeyError names a missing one)."""
    C = net.edge_attr("Cd", kind) * net.edge_attr("area", kind) * math.sqrt(2.0 * rho)
    n = torch.as_tensor(0.5, dtype=C.dtype)
    return PowerLaw(C, n, dp_transition=dp_transition, kind=kind, learnable=learnable)
