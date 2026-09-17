"""Local per-node reactions applied after a transport step by operator splitting, some
rate-based and others instantaneous equilibria."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class Reaction:
    """A local, per-node nonlinear map applied to the state after a transport step."""

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class FirstOrderDecay(Reaction):
    """``x <- x * exp(-rate * dt)``; ``rate`` broadcasts against ``x``."""

    def __init__(self, rate: torch.Tensor | float) -> None:
        self.rate = torch.as_tensor(rate, dtype=torch.float64)

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        rate = self.rate.to(x.dtype)
        return x * torch.exp(-rate * dt)


# Leighton, as MUNICH implements it: include/modules/chemistry/Leighton/reactions,
#   R1  NO2 + hv -> NO + O        J_NO2 [1/s], a driver
#   R2  O + O2 + M -> O3 + M      effectively instantaneous
#   R3  O3 + NO -> NO2 + O2       k3 = 3.0e-12 exp(-1500/T) cm3 molecule^-1 s^-1
#                                 (SPACK `ARR2 A B` == A exp(-B/T); NASA/JPL 2003)
# See `.superpowers/munich-formulas.md` section 6 and its T11.
K_NO_O3_298 = 1.9546779094727322e-14
"""k3 at 298 K, cm3 molecule^-1 s^-1."""

MOLAR_MASS = {"no": 30.0e-3, "no2": 46.0e-3, "o3": 48.0e-3}
"""kg/mol, from MUNICH's `species-leighton.dat` (NO 30., NO2 46., O3 48.)."""

# k' for a kg/m3 state, from dC_NO/dt = -k' C_NO C_O3:
#   k' = k3 * N_A / (1e6 * M_O3[kg/mol]) = k3 * 1.2546126583333333e19  [m3 kg^-1 s^-1]
# It is M_O3 and not M_NO because the O3 partner is the one converted from mass to number
# density. The published value is 2.45236e5, to six significant figures.
K_NO_O3 = K_NO_O3_298 * (6.02214076e23 / (1e6 * 48.0e-3))
"""k3(298 K) in m3 kg^-1 s^-1 for a kg/m3 state: 245236.36481890274."""


class Photostationary(Reaction):
    """NO/NO2/O3 relaxed to the Leighton photostationary state, in closed form.

    ``apply`` replaces the three named species columns of ``x`` (kg/m3, trailing shape
    ``(..., n, K)``) by the state satisfying ``J [NO2] = k [NO][O3]`` at the SAME NOx and
    Ox as the input -- conserved in MOLAR terms, because one NO2 becomes one NO and one O3
    per photolysis event and the two nitrogen species do not share a molar mass. It is an
    equilibrium, not a rate: ``dt`` is accepted for the ``Reaction`` contract and ignored,
    and applying it twice changes nothing (pinned by a test).

    The quadratic ``k z^2 - (k (P+Q) + J) z + k P Q = 0`` in ``z = [NO2]`` has its physical
    root at the MINUS sign; it is evaluated as ``2 k P Q / (S + sqrt(S^2 - 4 k^2 P Q))``
    with ``S = k (P+Q) + J``, the same root written without the catastrophic cancellation
    the direct form suffers whenever ``J >> k (P+Q)`` -- the ordinary daytime case. The
    discriminant equals ``(k (P-Q))^2 + 2 J k (P+Q) + J^2``, a sum of non-negative terms,
    so it is never clamped: a negative value there would be a bug worth raising on, not a
    value worth hiding.
    """

    def __init__(
        self,
        no: int,
        no2: int,
        o3: int,
        *,
        j_key: str = "J_NO2",
        k_no_o3: float = K_NO_O3,
        m_no: float = MOLAR_MASS["no"],
        m_no2: float = MOLAR_MASS["no2"],
        m_o3: float = MOLAR_MASS["o3"],
    ) -> None:
        self.columns = (int(no), int(no2), int(o3))
        self.j_key = str(j_key)
        self.masses = (float(m_no), float(m_no2), float(m_o3))
        # The molar rate constant the closed form uses: k_no_o3 acts on kg/m3, so
        # dc_NO/dt = -(k' M_O3) c_NO c_O3 once both partners are molar.
        self.k_molar = float(k_no_o3) * float(m_o3)

    def apply(
        self,
        x: torch.Tensor,
        dt: float | None = None,
        drivers: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(
                f"Photostationary: x must have trailing shape (n, K) with at least three "
                f"species columns; got {tuple(x.shape)}"
            )
        n_species = x.shape[-1]
        for column in self.columns:
            if not 0 <= column < n_species:
                raise ValueError(
                    f"Photostationary: column {column} is outside the state's "
                    f"{n_species} species columns (asked for {self.columns})"
                )
        if drivers is None or self.j_key not in drivers:
            raise KeyError(
                f"Photostationary: driver {self.j_key!r} (the NO2 photolysis rate, 1/s) is "
                f"required and was not given"
            )
        dtype = x.dtype
        i_no, i_no2, i_o3 = self.columns
        m_no, m_no2, m_o3 = (torch.tensor(m, dtype=dtype) for m in self.masses)
        c_no = x[..., i_no] / m_no
        c_no2 = x[..., i_no2] / m_no2
        c_o3 = x[..., i_o3] / m_o3
        j = torch.as_tensor(drivers[self.j_key], dtype=dtype)
        k = torch.tensor(self.k_molar, dtype=dtype)
        p = c_no + c_no2                      # NOx, mol/m3
        q = c_no2 + c_o3                      # Ox,  mol/m3
        s = k * (p + q) + j
        disc = s * s - 4.0 * k * k * p * q    # = (k(p-q))^2 + 2 j k (p+q) + j^2 >= 0
        denominator = s + torch.sqrt(disc)
        # denominator == 0 only when j == 0 AND p + q == 0: an empty cell with no
        # photolysis, whose answer is 0. The guard keeps that 0/0 off the autograd graph.
        safe = denominator > 0
        ones = torch.ones_like(denominator)
        z = torch.where(
            safe,
            2.0 * k * p * q / torch.where(safe, denominator, ones),
            torch.zeros_like(denominator),
        )
        out = x.clone()
        out[..., i_no] = (p - z) * m_no
        out[..., i_no2] = z * m_no2
        out[..., i_o3] = (q - z) * m_o3
        return out
