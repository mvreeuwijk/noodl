"""Local per-node reactions applied after a transport step by operator splitting, some
rate-based and others instantaneous equilibria."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

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
# `Photostationary` evaluates k3 at the temperature it is given (`k_no_o3_munich`); the
# K_NO_O3 constants below are its value at 298 K.
AVOGADRO = 6.02214076e23
"""1/mol (SI, exact)."""

R_GAS = 8.314462618
"""J/(mol K) (CODATA 2018)."""

P_STANDARD = 101325.0
"""Pa: the pressure `molar_volume` assumes when none is given."""

K_NO_O3_298 = 1.9546779094727322e-14
"""k3 at 298 K, cm3 molecule^-1 s^-1."""

MOLAR_MASS = {"no": 30.0e-3, "no2": 46.0e-3, "o3": 48.0e-3}
"""kg/mol, from MUNICH's `species-leighton.dat` (NO 30., NO2 46., O3 48.)."""

# k' for a kg/m3 state, from dC_NO/dt = -k' C_NO C_O3:
#   k' = k3 * N_A / (1e6 * M_O3[kg/mol]) = k3 * 1.2546126583333333e19  [m3 kg^-1 s^-1]
# It is M_O3 and not M_NO because the O3 partner is the one converted from mass to number
# density. The published value is 2.45236e5, to six significant figures.
K_NO_O3 = K_NO_O3_298 * (AVOGADRO / (1e6 * 48.0e-3))
"""k3(298 K) in m3 kg^-1 s^-1 for a kg/m3 state: 245236.36481890274."""


def k_no_o3_munich(temperature) -> torch.Tensor:
    """k(NO + O3) = 3.0e-12 exp(-1500/T) cm3 molecule^-1 s^-1, returned in m3 mol^-1 s^-1.

    MUNICH's Leighton rate (NASA/JPL 2003) at the absolute temperature `temperature` (K),
    times `N_A * 1e-6`. It is `Photostationary`'s default rate.
    """
    t = torch.as_tensor(temperature, dtype=torch.float64)
    return 3.0e-12 * AVOGADRO * 1e-6 * torch.exp(-1500.0 / t)


def molar_volume(temperature, pressure=P_STANDARD) -> torch.Tensor:
    """The ideal-gas molar volume `R T / P`, m3/mol, at `temperature` (K) and `pressure` (Pa).

    One ppb of a species is `1e-9 / molar_volume` mol/m3.
    """
    t = torch.as_tensor(temperature, dtype=torch.float64)
    return R_GAS * t / torch.as_tensor(pressure, dtype=torch.float64)


class Photostationary(Reaction):
    """NO/NO2/O3 relaxed to the Leighton photostationary state, in closed form.

    ``apply`` replaces the three named species columns of ``x`` (kg/m3, trailing shape
    ``(..., n, K)``) by the state satisfying ``J [NO2] = k [NO][O3]`` at the SAME NOx and
    Ox as the input -- conserved in MOLAR terms, because one NO2 becomes one NO and one O3
    per photolysis event and the two nitrogen species do not share a molar mass. It is an
    equilibrium, not a rate: ``dt`` is accepted for the ``Reaction`` contract and ignored,
    and applying it twice changes nothing (pinned by a test).

    The rate ``k`` of NO + O3 is ``rate(T)`` in m3 mol^-1 s^-1, evaluated at the driver
    ``temperature_key`` (absolute temperature, K; it broadcasts against the state's leading
    shape, so it may be one value or one per node). Without a ``rate`` it is MUNICH's
    ``k_no_o3_munich``. A constant ``k_no_o3`` (m3 kg^-1 s^-1 for the kg/m3 state, e.g.
    ``K_NO_O3``, the 298 K value) is used instead of a rate and makes the temperature
    unnecessary; giving both ``rate`` and ``k_no_o3`` raises.

    ``floor_ppb`` bounds the ratio ``J/k`` from below, ``K = max(J/k, floor)``, with the
    floor in ppb converted to mol/m3 by ``1e-9 / V_m``. ``V_m`` (m3/mol) is the driver
    ``molar_volume_key`` if given, else ``molar_volume(T)`` at 101325 Pa. The molar volume
    enters nowhere else: without a floor (``floor_ppb = 0``, the default) the equilibrium
    is the same whatever unit it is solved in. Reproducing SIRANE exactly needs the
    molar-volume driver (SIRANE's ground-level V_m), since the fallback uses T and
    101325 Pa. The floor is a clip, so the gradient with respect to ``J`` is zero
    wherever it is active.

    The quadratic ``k z^2 - (k (P+Q) + J) z + k P Q = 0`` in ``z = [NO2]`` has its physical
    root at the MINUS sign; it is evaluated as ``2 k P Q / (S + sqrt(S^2 - 4 k^2 P Q))``
    with ``S = k (P+Q) + J``, the same root written without the catastrophic cancellation
    the direct form suffers whenever ``J >> k (P+Q)`` -- the ordinary daytime case. The
    discriminant is evaluated directly in that sum-of-squares form,
    ``(k (P-Q))^2 + 2 J k (P+Q) + J^2``, and not via the algebraically equal expansion
    ``S^2 - 4 k^2 P Q``: the expansion cancels catastrophically whenever ``P ~= Q`` and
    ``J`` is small (the ordinary night-time near-titration state, NOx ~= Ox), producing a
    spuriously negative value and a NaN square root. Every term of the sum-of-squares form
    is a square or a product of non-negatives, so it is never negative in floating point,
    not only in exact arithmetic, and is therefore never clamped: a negative value there
    would be a bug worth raising on, not a value worth hiding. With a floor, ``J`` in these
    expressions is ``max(J, k floor)``.
    """

    def __init__(
        self,
        no: int,
        no2: int,
        o3: int,
        *,
        j_key: str = "J_NO2",
        rate: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature_key: str = "temperature",
        k_no_o3: float | None = None,
        floor_ppb: float = 0.0,
        molar_volume_key: str = "molar_volume",
        m_no: float = MOLAR_MASS["no"],
        m_no2: float = MOLAR_MASS["no2"],
        m_o3: float = MOLAR_MASS["o3"],
    ) -> None:
        if rate is not None and k_no_o3 is not None:
            raise ValueError(
                "Photostationary: give either rate (a function of temperature) or "
                "k_no_o3 (a constant), not both"
            )
        self.columns = (int(no), int(no2), int(o3))
        self.j_key = str(j_key)
        self.rate = k_no_o3_munich if rate is None else rate
        self.temperature_key = str(temperature_key)
        self.molar_volume_key = str(molar_volume_key)
        self.masses = (float(m_no), float(m_no2), float(m_o3))
        floor = float(floor_ppb)
        if not (math.isfinite(floor) and floor >= 0.0):
            raise ValueError(
                f"Photostationary: floor_ppb must be finite and >= 0, got {floor_ppb!r}"
            )
        self.floor_ppb = floor
        # A constant rate, in the molar form the closed form uses: k_no_o3 acts on kg/m3,
        # so dc_NO/dt = -(k' M_O3) c_NO c_O3 once both partners are molar.
        self.k_molar = None if k_no_o3 is None else float(k_no_o3) * float(m_o3)

    def _temperature(self, drivers: Mapping[str, torch.Tensor], dtype) -> torch.Tensor:
        if self.temperature_key not in drivers:
            raise KeyError(
                f"Photostationary: driver {self.temperature_key!r} (the air temperature, K, "
                f"at which the NO + O3 rate is evaluated) is required and was not given"
            )
        t = torch.as_tensor(drivers[self.temperature_key], dtype=dtype)
        with torch.no_grad():
            if not bool(torch.isfinite(t).all()) or bool((t <= 0).any()):
                raise ValueError(
                    f"Photostationary: driver {self.temperature_key!r} must be an absolute "
                    f"temperature (finite, > 0 K); got {t.detach().flatten()[:4].tolist()}"
                )
        return t

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
        if self.k_molar is None:
            k = torch.as_tensor(self.rate(self._temperature(drivers, dtype)), dtype=dtype)
        else:
            k = torch.tensor(self.k_molar, dtype=dtype)
        if self.floor_ppb > 0.0:
            if self.molar_volume_key in drivers:
                v_m = torch.as_tensor(drivers[self.molar_volume_key], dtype=dtype)
            else:
                v_m = molar_volume(self._temperature(drivers, dtype)).to(dtype)
            # K = max(J/k, floor)  <=>  J_eff = max(J, k * floor), the floor in mol/m3.
            j = torch.maximum(j, k * (self.floor_ppb * 1e-9) / v_m)
        p = c_no + c_no2                      # NOx, mol/m3
        q = c_no2 + c_o3                      # Ox,  mol/m3
        s = k * (p + q) + j
        # Sum-of-squares form: every term is a square or a product of non-negatives, so
        # this is >= 0 in floating point, not only in exact arithmetic. The algebraically
        # equal expanded form s*s - 4*k*k*p*q cancels catastrophically whenever p ~= q
        # and j is small (the ordinary night-time near-titration state, NOx ~= Ox),
        # producing a spuriously negative disc, a NaN sqrt, and a silently unchanged
        # output.
        disc = (k * (p - q)) ** 2 + 2.0 * j * k * (p + q) + j * j
        # disc == 0 exactly when p == q bitwise AND j == 0 -- the exactly titrated cell
        # with no photolysis. sqrt has infinite slope there, so the raw sqrt puts NaN into
        # the gradient w.r.t. every input (measured: [nan, nan, nan] at the point,
        # [1, 1, 1/3] one ulp away). The guard substitutes a safe 1.0 under the sqrt on the
        # degenerate branch and returns a hard 0 there, which is the forward value already.
        positive = disc > 0
        root = torch.where(
            positive,
            torch.sqrt(torch.where(positive, disc, torch.ones_like(disc))),
            torch.zeros_like(disc),
        )
        denominator = s + root
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
