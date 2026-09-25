"""Water quality and H2S: sulfide generation, BOD decay and two-film transfer.

Formula set and verification status from the milestone 4 literature review, recorded in the
spec's coefficient register:

* Henry's law, H2S, dimensionless gas-over-liquid
  ``H(T) = 1 / (H_cp(T) R T)`` with ``H_cp(298.15) = 1.0e-3 mol m^-3 Pa^-1`` and van 't Hoff
  slope 2100 K (Sander 2023, ACP 23:10901 -- VERIFIED). Measured: ``H(293.15) = 0.363854``,
  ``H(298.15) = 0.403418``.
* free-sulfide fraction ``f = 1 / (1 + 10^(pH - pKa))``, ``pKa = 7.0`` (VERIFIED value;
  its temperature dependence is approximated as constant, stated).
* transfer coefficient ``K_L a = a (1 + b Fr^2) (S0 V)^(3/8) / d_m`` per hour, ``a = 0.86``,
  ``b = 0.20`` (form corroborated, CONSTANTS UNVERIFIED -- Yongsiri et al. 2004, paywalled).
* Pomeroy-Parkhurst sulfide balance
  ``d[S]/dt = M' EBOD / R_h - m [S] (S0 V)^(3/8) / d_m`` with ``EBOD = BOD 1.07^(T-20)``,
  ``M' = 0.32e-3 m/h``, ``m = 0.64 m/h (m/s)^(-3/8)`` (vendor defaults, UNVERIFIED against
  Pomeroy and Parkhurst 1977).
* bulk BOD decay ``k_BOD = 0.2 /day`` with the same ``1.07^(T-20)`` (generic, Metcalf &
  Eddy).
"""

from __future__ import annotations

import torch

from noodl.apps.sewer.hydraulics import resolve_nodal_driver
from noodl.layers.reaction import Reaction

Tensor = torch.Tensor
F64 = torch.float64

R_GAS = 8.314
HCP_298 = 1.0e-3
VANT_HOFF = 2100.0
PKA_H2S = 7.0
KLA_A = 0.86
KLA_B = 0.20
M_PRIME = 0.32e-3
M_LOSS = 0.64
K_BOD = 0.2
THETA = 1.07
G = 9.80665

#: Molar masses (kg/mol). Total sulfide is carried as S, headspace H2S as H2S, so a flux in
#: moles of S becomes ``M_H2S / M_S`` times the mass on the gas side.
M_S = 32.06e-3
M_H2S = 34.08e-3


def henry_h2s(temperature: Tensor) -> Tensor:
    """Dimensionless Henry constant ``H = C_gas / C_liquid`` at absolute temperature (K)."""
    hcp = HCP_298 * torch.exp(VANT_HOFF * (1.0 / temperature - 1.0 / 298.15))
    return 1.0 / (hcp * R_GAS * temperature)


def free_fraction(ph: Tensor, pka: float = PKA_H2S) -> Tensor:
    """Fraction of total dissolved sulfide present as volatile H2S."""
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def kla_h2s(slope: Tensor, velocity: Tensor, depth: Tensor, *, a=KLA_A, b=KLA_B) -> Tensor:
    """``K_L a`` in 1/s from the per-hour correlation ``a (1 + b Fr^2)(S0 V)^(3/8)/d_m``.

    ``Fr = V / sqrt(g d_m)``. ``d_m`` and ``S0 V`` are floored on BOTH branches of their
    ``where`` so a dry pipe (``d_m = 0``, ``V = 0``) returns exactly zero instead of a
    ``nan`` gradient.
    """
    dry = depth <= 0
    depth_safe = torch.where(dry, torch.ones_like(depth), depth)
    froude2 = velocity**2 / (G * depth_safe)
    sv = torch.clamp(slope * velocity, min=0.0)
    sv_safe = torch.where(sv <= 0, torch.ones_like(sv), sv)
    value = a * (1.0 + b * froude2) * sv_safe ** (3.0 / 8.0) / depth_safe / 3600.0
    return torch.where(dry | (sv <= 0), torch.zeros_like(value), value)


def sulfide_rate(
    bod: Tensor,
    sulfide: Tensor,
    t_water_c: Tensor,
    radius: Tensor,
    slope: Tensor,
    velocity: Tensor,
    depth: Tensor,
    *,
    m_prime=M_PRIME,
    m_loss=M_LOSS,
) -> Tensor:
    """Pomeroy-Parkhurst ``d[S]/dt`` in kg m^-3 s^-1 (inputs in kg/m^3, SI, T in Celsius).

    ``M' EBOD / R_h - m [S] (S0 V)^(3/8) / d_m``, both coefficients in m/h, so the whole
    expression is divided by 3600 to reach per-second. ``R_h`` and ``d_m`` are floored on
    BOTH branches of their ``where``: a dry pipe generates and loses nothing.
    """
    theta = THETA ** (t_water_c - 20.0)
    dry = (radius <= 0) | (depth <= 0)
    radius_safe = torch.where(dry, torch.ones_like(radius), radius)
    depth_safe = torch.where(dry, torch.ones_like(depth), depth)
    sv = torch.clamp(slope * velocity, min=0.0)
    sv_safe = torch.where(sv <= 0, torch.zeros_like(sv) + 1e-300, sv)
    gen = m_prime * bod * theta / radius_safe
    loss = m_loss * sulfide * theta * sv_safe ** (3.0 / 8.0) / depth_safe
    rate = (gen - loss) / 3600.0
    return torch.where(dry, torch.zeros_like(rate), rate)


def bod_rate(bod: Tensor, t_water_c: Tensor, *, k_bod=K_BOD) -> Tensor:
    """First-order bulk BOD decay ``-k_BOD 1.07^(T-20) [BOD]`` in kg m^-3 s^-1."""
    return -k_bod * THETA ** (t_water_c - 20.0) * bod / 86400.0


def two_film_flux(
    sulfide: Tensor,
    gas: Tensor,
    volume: Tensor,
    kla: Tensor,
    free: Tensor,
    henry: Tensor,
) -> Tensor:
    """Water-to-air H2S flux in kg S/s: ``K_L a V (f C_S - C_G M_S / (M_H2S H))``.

    ``sulfide`` is total dissolved sulfide as S (kg/m^3); ``gas`` is headspace H2S as H2S
    (kg/m^3). The gas concentration is converted to S-equivalent by ``M_S / M_H2S`` before
    the Henry partition, so the returned flux is in kg of S per second on BOTH sides and the
    two source terms are exactly opposite in moles of S.
    """
    return kla * volume * (free * sulfide - gas * (M_S / M_H2S) / henry)


def ppm_from_concentration(
    concentration: Tensor, temperature: Tensor, pressure: float = 101325.0
) -> Tensor:
    """kg/m^3 of H2S -> parts per million by volume at the headspace temperature."""
    molar_volume = R_GAS * temperature / pressure
    return concentration / M_H2S * molar_volume * 1e6


def _gather(value: Tensor, out_pipe: Tensor | None) -> Tensor:
    """Gather a per-PIPE driver into manhole order through ``out_pipe`` (see the two
    classes' docstrings for the two orders); returns ``value`` unchanged when ``out_pipe``
    is ``None`` (the dictated unit tests hand per-manhole vectors directly)."""
    if out_pipe is None:
        return value
    return value.index_select(-1, out_pipe)


class LateralLoads:
    """Lateral inflow-concentration loads (spec 3.4/4.2), as a `Model` closure.

    `bod_in`/`sulfide_in` (kg/m3, full-node, spec 4.2) times `inflow` (m3/s, full-node)
    gives the load `s_j C_in,j` in kg/s per species (spec 3.4's water_quality source term),
    written to `"<water_layer>.sources"` in FULL node order. FR-21: before this closure
    existed, `bod_in`/`sulfide_in` were created by the builder and read by nothing, so no
    lateral load ever reached the water-quality layer.

    Registered BEFORE `H2STransfer` in `build_model`'s closure list: `H2STransfer`
    reads `drivers.get("<water_layer>.sources")` and ADDS its own transfer term to it
    (rather than overwriting), so this closure's load survives regardless of whether
    `H2STransfer` also runs (it is registered whenever `quality=True`, with or without
    `air=True`).

    `columns` maps a species column index to the driver key carrying that species' own
    inflow concentration; a species column with no entry gets a zero load. The default the
    builder uses, for `species=("bod", "sulfide")`, is `{0: "bod_in", 1: "sulfide_in"}`.
    """

    def __init__(
        self,
        n_nodes: int,
        *,
        columns: dict[int, str],
        n_species: int,
        water_layer: str = "water_quality",
    ) -> None:
        self.n_nodes = int(n_nodes)
        self.columns = dict(columns)
        self.n_species = int(n_species)
        self.water_layer = water_layer

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        inflow = _need(drivers, "inflow")
        if inflow.shape[-1] != self.n_nodes:
            raise ValueError(
                f"LateralLoads: driver 'inflow' must be full-node, trailing shape "
                f"({self.n_nodes},), got {tuple(inflow.shape)}"
            )
        parts = []
        for col in range(self.n_species):
            key = self.columns.get(col)
            if key is None:
                parts.append(torch.zeros_like(inflow))
                continue
            c_in = _need(drivers, key)
            if c_in.shape[-1] != self.n_nodes:
                raise ValueError(
                    f"LateralLoads: driver {key!r} must be full-node, trailing shape "
                    f"({self.n_nodes},), got {tuple(c_in.shape)}"
                )
            parts.append(inflow * c_in)
        sources = parts[0] if self.n_species == 1 else torch.stack(parts, dim=-1)
        return {f"{self.water_layer}.sources": sources}


class SulfideGeneration(Reaction):
    """Operator-split Pomeroy-Parkhurst sulfide generation and bulk BOD decay.

    Applied per manhole (equivalently per outgoing pipe) after the transport step, exactly
    as the framework applies every `Reaction`. `x` carries the species in the layer's own
    column order, defaulting to ``("bod", "sulfide")``; the per-pipe hydraulic drivers
    ``"sewer.R_h"``, ``"sewer.v"`` and ``"sewer.d_m"`` are gathered from PIPE order into
    MANHOLE order (the layer's own interior order) through ``out_pipe`` when it is given
    (M4-R4 amendment): ``value.index_select(-1, out_pipe)``, where ``out_pipe[i]`` is the
    position, in the per-pipe driver vectors, of manhole ``i``'s outgoing pipe.
    ``"sewer.q_slope"`` is ALREADY per manhole (the builder supplies it as a constant
    driver) and is NEVER gathered. When ``out_pipe`` is ``None`` every driver is used
    exactly as given -- the per-manhole vectors the dictated unit tests hand in directly.
    """

    def __init__(
        self,
        *,
        bod: int = 0,
        sulfide: int = 1,
        m_prime: float = M_PRIME,
        m_loss: float = M_LOSS,
        k_bod: float = K_BOD,
        out_pipe: Tensor | None = None,
        manhole_idx: Tensor | None = None,
        n_nodes: int | None = None,
    ) -> None:
        self.bod = int(bod)
        self.sulfide = int(sulfide)
        self.m_prime = float(m_prime)
        self.m_loss = float(m_loss)
        self.k_bod = float(k_bod)
        self.out_pipe = (
            None if out_pipe is None else torch.as_tensor(out_pipe, dtype=torch.long)
        )
        # FR-12/N3: when given (the `build_model` builder passes both), `T_water` is
        # resolved through the shared `resolve_nodal_driver` helper (0-d / full-node /
        # trailing-singleton, spec 4.2); when either is omitted (the dictated unit tests'
        # own convention) `T_water` is used exactly as given, already per-manhole or scalar.
        self.manhole_idx = (
            None if manhole_idx is None else torch.as_tensor(manhole_idx, dtype=torch.long)
        )
        self.n_nodes = None if n_nodes is None else int(n_nodes)

    def apply(self, x: Tensor, dt: float | None, drivers=None) -> Tensor:
        if dt is None:
            return x
        drivers = drivers or {}
        n_species = x.shape[-1]
        if not (0 <= self.bod < n_species) or not (0 <= self.sulfide < n_species):
            raise ValueError(
                f"SulfideGeneration: species index bod={self.bod} sulfide={self.sulfide} "
                f"out of range for a water_quality state with {n_species} species"
            )
        radius = _gather(_need(drivers, "sewer.R_h"), self.out_pipe)
        slope = _need(drivers, "sewer.q_slope")
        velocity = _gather(_need(drivers, "sewer.v"), self.out_pipe)
        depth = _gather(_need(drivers, "sewer.d_m"), self.out_pipe)
        t_water = _need(drivers, "T_water")
        if self.manhole_idx is not None and self.n_nodes is not None:
            t_water = resolve_nodal_driver(
                t_water, self.manhole_idx, self.n_nodes,
                key="T_water", name="apps.sewer.SulfideGeneration",
            )
        bod = x[..., self.bod]
        sulfide = x[..., self.sulfide]
        d_sulfide = sulfide_rate(
            bod, sulfide, t_water, radius, slope, velocity, depth,
            m_prime=self.m_prime, m_loss=self.m_loss,
        )
        d_bod = bod_rate(bod, t_water, k_bod=self.k_bod)
        out = x.clone()
        out[..., self.sulfide] = torch.clamp(sulfide + dt * d_sulfide, min=0.0)
        out[..., self.bod] = torch.clamp(bod + dt * d_bod, min=0.0)
        return out


class H2STransfer:
    """Two-film water-to-air H2S transfer, as a `Model` closure.

    Reads the PREVIOUS pass's concentrations (the framework's lagged cross-layer coupling)
    and writes the two layers' `sources` drivers. Both are in FULL node order and both are
    exactly opposite in MOLES of sulfur, which row C2 asserts node by node.

    ADDS to an existing `"<water_layer>.sources"`/`"<air_layer>.sources"` driver rather than
    overwriting it (FR-21): registered AFTER `LateralLoads` in `build_model`'s closure
    list, so `drivers` here already carries `LateralLoads`'s inflow-concentration term (spec
    3.4's `s_j C_in,j`) when quality is built with lateral loads, and this closure's own
    transfer term is added on top rather than silently discarding it.

    The per-pipe hydraulic drivers ``"sewer.v"``, ``"sewer.d_m"`` and ``"sewer.V_wet"`` are
    gathered from PIPE order into MANHOLE order
    (this closure's own interior order, ``manhole_idx``) through ``out_pipe`` when it is
    given (M4-R4 amendment): ``value.index_select(-1, out_pipe)``, where ``out_pipe[i]`` is
    the position, in the per-pipe driver vectors, of manhole ``i``'s outgoing pipe.
    ``"sewer.q_slope"`` is ALREADY per manhole (the builder supplies it as a constant
    driver) and is NEVER gathered. When ``out_pipe`` is ``None`` every per-pipe driver is
    used exactly as given -- the per-manhole vectors the dictated unit tests hand in
    directly.

    The closure never returns a layer's own state key, so nothing here needs spec 4.6a.
    """

    def __init__(
        self,
        n_nodes: int,
        manhole_idx: Tensor,
        *,
        sulfide: int = 1,
        water_layer: str = "water_quality",
        air_layer: str = "air_quality",
        out_pipe: Tensor | None = None,
    ) -> None:
        self.n_nodes = int(n_nodes)
        self.manhole_idx = torch.as_tensor(manhole_idx, dtype=torch.long)
        self.sulfide = int(sulfide)
        self.water_layer = water_layer
        self.air_layer = air_layer
        self.out_pipe = (
            None if out_pipe is None else torch.as_tensor(out_pipe, dtype=torch.long)
        )
        self.notes: dict[str, str] = {
            "pka": "pKa = 7.0 is held constant; its temperature dependence is approximated",
            "species": "dissolved sulfide is carried as S and headspace H2S as H2S; the "
            "flux is converted by the molar masses 32.06 and 34.08 g/mol",
        }

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        water = state.get(f"{self.water_layer}.x")
        gas = state.get(f"{self.air_layer}.x")
        if water is None or gas is None:
            raise KeyError(
                f"H2STransfer: states {self.water_layer + '.x'!r} and "
                f"{self.air_layer + '.x'!r} are both required; build them with "
                f"initial_state(model)"
            )
        n_species = water.shape[-1]
        if not (0 <= self.sulfide < n_species):
            raise ValueError(
                f"H2STransfer: species index sulfide={self.sulfide} out of range for a "
                f"water_quality state with {n_species} species"
            )
        volume = _gather(_need(drivers, "sewer.V_wet"), self.out_pipe)
        slope = _need(drivers, "sewer.q_slope")
        velocity = _gather(_need(drivers, "sewer.v"), self.out_pipe)
        depth = _gather(_need(drivers, "sewer.d_m"), self.out_pipe)
        t_head = resolve_nodal_driver(
            _need(drivers, "T_head"), self.manhole_idx, self.n_nodes,
            key="T_head", name="H2STransfer",
        )
        ph = resolve_nodal_driver(
            _need(drivers, "pH"), self.manhole_idx, self.n_nodes,
            key="pH", name="H2STransfer",
        )
        kla = kla_h2s(slope, velocity, depth)
        free = free_fraction(ph)
        henry = henry_h2s(t_head)
        flux = two_film_flux(
            water[..., self.sulfide], gas, volume, kla, free, henry
        )
        batch = flux.shape[:-1]
        water_sources = torch.zeros(batch + (self.n_nodes, n_species), dtype=F64)
        water_sources = water_sources.index_add(
            -2,
            self.manhole_idx,
            torch.nn.functional.one_hot(
                torch.tensor(self.sulfide), n_species
            ).to(F64) * (-flux).unsqueeze(-1),
        )
        air_sources = torch.zeros(batch + (self.n_nodes,), dtype=F64)
        air_sources = air_sources.index_add(
            -1, self.manhole_idx, flux * (M_H2S / M_S)
        )
        existing_water = drivers.get(f"{self.water_layer}.sources")
        if existing_water is not None:
            water_sources = water_sources + torch.as_tensor(existing_water, dtype=F64)
        existing_air = drivers.get(f"{self.air_layer}.sources")
        if existing_air is not None:
            air_sources = air_sources + torch.as_tensor(existing_air, dtype=F64)
        return {
            f"{self.water_layer}.sources": water_sources,
            f"{self.air_layer}.sources": air_sources,
        }


def _need(drivers, key: str) -> Tensor:
    try:
        return torch.as_tensor(drivers[key], dtype=F64)
    except KeyError as exc:
        raise KeyError(
            f"apps.sewer.quality: driver {key!r} is required and was not given; it is "
            f"written by the SewerHydraulics closure in the same pass"
        ) from exc
