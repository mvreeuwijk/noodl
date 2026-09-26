"""The format-neutral street case: ONE `StreetCase` structure that every street-model
source (MUNICH's own files, and, in a later plan, SIRANE's decks) is read into and written
from -- no MUNICH- or SIRANE-specific element belongs on `StreetCase` itself. MUNICH's own
file format (its `[section]`/`key: value` dialect, the semicolon street/intersection
tables, the binaries) lives in the private `_munich_files` module; this module knows only
plain arrays and the `StreetNetwork` they describe.

`UG_PER_KG` and `EARTH_RADIUS_M` are MUNICH-specific constants, re-exported here (from
`_munich_files`, which owns and documents them) because callers reading/writing MUNICH data
through this module's own units and geometry need them without reaching into the private
module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from noodl.apps.street_aq import _munich_files
from noodl.apps.street_aq._munich_files import (
    EARTH_RADIUS_M,
    UG_PER_KG,
    deg_from_to_munich_rad,
    munich_rad_to_deg_from,
)
from noodl.apps.street_aq.network import StreetNetwork
from noodl.apps.street_aq.routing import StreetFlows
from noodl.couple import CONTAM_DEG_TO_STREET_RAD, apply_conversion
from noodl.model import Model

F64 = torch.float64

__all__ = [
    "EARTH_RADIUS_M",
    "UG_PER_KG",
    "StreetCase",
    "deg_from_to_munich_rad",
    "drivers_at",
    "munich_rad_to_deg_from",
    "read_case",
    "write_case",
]

_JUNCTION_CASE_KEY = {"theta_w": "wind_dir_from_deg", "U_ref": "wind_speed"}
"""`drivers_at`'s driver key -> the `StreetCase.meteo`/`meteo_junction` key it reads from,
for the two keys whose names differ (`theta_w` is stored as `wind_dir_from_deg`, in
degrees, until converted; `U_ref` is stored as `wind_speed`). Every other key
(`u_star`, `h_abl`, `lmo`) is spelled the same on both sides."""


@dataclass(frozen=True)
class StreetCase:
    """One street-network case: the network plus every array a `Model` built on it needs
    to run, independent of which source model (`source`) it was read from.

    Attributes:
        source: `"munich"` or `"sirane"` -- which reader produced this case (and which
            `model_options` resolves against).
        network: The case's `StreetNetwork` (metres).
        times: `(n_hours,)`, seconds since the case's own start (not necessarily an
            absolute date).
        street_ids: `network.streets`' names, in the order `emissions`'/`background`'s
            street axis uses (== `network.streets` order).
        junction_ids: The source model's OWN node ids for `network.junctions`, in that
            same order (MUNICH: the intersection ids from `intersection.dat`).
        species: The case's species names, in the order `emissions`'/`background`'s last
            axis uses.
        meteo: One `(n_hours, n_streets)` array per key: `wind_dir_from_deg` (degrees
            clockwise from north, the direction the wind blows FROM) and `wind_speed`
            always; `h_abl`, `u_star`, `lmo`, `temperature` wherever the source provides
            them.
        meteo_junction: The same keys, `(n_hours, n_junctions)`, in `network.junctions`
            order -- may be empty (or missing some keys) when the source has no genuine
            per-junction meteorology (MUNICH: the `...Inter` fields).
        emissions: `(n_hours, n_streets, n_species)`, **kg/s** per street (MUNICH's own
            files are micrograms/s; converted at read time -- see `UG_PER_KG`).
        background: `(n_hours, n_streets, n_species)`, **kg/m3**, per street (kept per
            street, unlike an earlier version of this reader that reduced it to one
            network-wide value at read time).
        native: The source model's own options, as read (MUNICH: one dict per `munich.cfg`
            section) -- `model_options` reads this to translate them into `build_model`
            keywords.
    """

    source: str
    network: StreetNetwork
    times: list[float]
    street_ids: list[str]
    junction_ids: list[str]
    species: list[str]
    meteo: dict[str, np.ndarray]
    meteo_junction: dict[str, np.ndarray]
    emissions: np.ndarray
    background: np.ndarray
    native: dict

    def model_options(self) -> dict:
        """The `build_model`/`StreetFlows` keywords this case's own source model implies --
        e.g. MUNICH's `Mean_wind_speed_parameterization: Exponential` becomes
        `canyon_wind="exponential"`.

        Only `source="munich"` is implemented; any other source raises `NotImplementedError`
        naming it (SIRANE's own options are Plan 2's job).
        """
        if self.source != "munich":
            raise NotImplementedError(
                f"StreetCase.model_options: source {self.source!r} is not implemented; "
                f"only 'munich' is"
            )
        street = self.native.get("street", {})
        canyon_wind_min = float(street.get("Minimum_Street_Wind_Speed", 0.1))
        return dict(
            canyon_wind="exponential", exchange="schulte", stability="munich",
            direction_averaging="munich", roof_wind_form="sirane",
            canyon_wind_min=canyon_wind_min, u_d_min=0.001,
        )


def read_case(path: Path) -> StreetCase:
    """`StreetCase` from `path`: a directory holding `munich.cfg` is read as a MUNICH case;
    a SIRANE master `.dat` file will be read as a SIRANE case once Plan 2 implements it.
    Anything else raises `ValueError` naming both expectations.
    """
    path = Path(path)
    if path.is_dir() and (path / "munich.cfg").is_file():
        raw = _munich_files.read_munich_case(path)
        return StreetCase(source="munich", **raw)
    raise ValueError(
        f"read_case: {path} is neither a directory holding 'munich.cfg' (a MUNICH case) "
        f"nor a SIRANE master '.dat' file (a SIRANE case)"
    )


def write_case(
    out_dir: Path,
    network: StreetNetwork,
    *,
    format: str = "munich",
    species: Sequence[str],
    date_min: str,
    n_hours: int,
    meteo: Mapping[str, float | np.ndarray],
    emissions_kg_s: float | np.ndarray,
    background_kg_m3: float | np.ndarray,
    options: Mapping[str, str] | None = None,
    lat0_deg: float = 48.85,
    lon0_deg: float = 2.35,
) -> Path:
    """Writes `network` as a case in `format` under `out_dir`, and returns `out_dir`. Only
    `format="munich"` is implemented -- see `_munich_files.write_munich_case` for its
    argument semantics (`meteo`'s keys are MUNICH's own field names)."""
    if format != "munich":
        raise NotImplementedError(
            f"write_case: format {format!r} is not implemented; only 'munich' is"
        )
    return _munich_files.write_munich_case(
        out_dir, network, species=species, date_min=date_min, n_hours=n_hours, meteo=meteo,
        emissions_kg_s=emissions_kg_s, background_kg_m3=background_kg_m3, options=options,
        lat0_deg=lat0_deg, lon0_deg=lon0_deg,
    )


def _circular_mean(rad: np.ndarray, axis: int) -> np.ndarray:
    """The mean DIRECTION of angles in radians -- invariant to where the angles wrap
    (an arithmetic mean of angles straddling the wrap lands on the opposite side)."""
    return np.arctan2(np.sin(rad).mean(axis=axis), np.cos(rad).mean(axis=axis)) % (2.0 * np.pi)


def _reciprocal_mean(lmo: np.ndarray, axis: int) -> np.ndarray:
    """The mean of the Obukhov length through its reciprocal, `1 / mean(1 / L)`: the
    stability branches depend continuously on `1/L`, and a plain mean of `L` across values
    that straddle zero (stable next to unstable) can land on the wrong sign."""
    return 1.0 / np.mean(1.0 / lmo, axis=axis)


def drivers_at(
    case: StreetCase, model: Model, k: int, *, species: Sequence[str] | None = None
) -> dict[str, torch.Tensor]:
    """The driver mapping at time index `k` for `model`, built from `case`.

    Reads `model`'s `StreetFlows` closure to decide the shape: `meteo="per_street"` gives
    every driver a trailing street axis straight from `case`'s own per-street arrays (plus
    `"<key>_junction"`, from `case.meteo_junction` when the source has it, else the same
    street-to-junction reduction `StreetFlows.junction_values` itself falls back to);
    `meteo="uniform"` reduces every driver to one network-wide value first (circular mean
    for direction, the reciprocal mean for the Obukhov length, a plain mean otherwise).
    Similarly, the transport layer's own boundary count decides whether `"<layer>.x_boundary"`
    is per-street or a single network-wide mean.

    `u_star` is supplied whenever `case` has it (MUNICH's `UST`/`USTInter`); `U_ref` is
    `case`'s own wind speed (not derived through noodl's log law -- `u_star`, when present,
    is what actually sets the friction velocity; `U_ref` is needed only for the direction
    spread `sigma_theta = sigma_v / U_ref`).
    """
    flows = next(c for c in model.closures if isinstance(c, StreetFlows))
    layer = model.transport[flows.layer_name]
    n_streets = len(case.street_ids)

    species_names = list(species) if species is not None else list(case.species)
    s_idx = [case.species.index(s) for s in species_names]
    n_species = len(species_names)

    def pick_species(row: np.ndarray) -> np.ndarray:
        picked = row[:, s_idx]
        return picked[:, 0] if n_species == 1 else picked

    emissions_row = pick_species(case.emissions[k])
    background_row = pick_species(case.background[k])

    sources = torch.zeros(
        (model.net.n,) if n_species == 1 else (model.net.n, n_species), dtype=F64
    )
    for i, street_id in enumerate(case.street_ids):
        sources[model.net.node_index(street_id)] = torch.as_tensor(emissions_row[i], dtype=F64)

    n_b = layer.n_b
    if n_b == n_streets:
        boundary_value = background_row
    elif n_b == 1:
        mean = background_row.mean(axis=0)
        boundary_value = np.reshape(mean, (1,) if n_species == 1 else (1, n_species))
    else:
        raise ValueError(
            f"drivers_at: the model's boundary count is {n_b}, neither 1 "
            f"(background='uniform') nor {n_streets} (background='per_street')"
        )

    out: dict[str, torch.Tensor] = {
        f"{flows.layer_name}.sources": sources,
        f"{flows.layer_name}.x_boundary": torch.as_tensor(boundary_value, dtype=F64),
    }

    theta_deg = case.meteo["wind_dir_from_deg"][k]
    wind_speed = case.meteo["wind_speed"][k]
    u_star = case.meteo["u_star"][k] if "u_star" in case.meteo else None
    h_abl = case.meteo["h_abl"][k] if "h_abl" in case.meteo else None
    lmo = case.meteo["lmo"][k] if "lmo" in case.meteo else None
    theta_w_all = apply_conversion(
        CONTAM_DEG_TO_STREET_RAD, torch.as_tensor(theta_deg, dtype=F64), {}
    )

    if flows.meteo == "per_street":
        out["theta_w"] = theta_w_all
        out["U_ref"] = torch.as_tensor(wind_speed, dtype=F64)
        if u_star is not None:
            out["u_star"] = torch.as_tensor(u_star, dtype=F64)
        if h_abl is not None:
            out["h_abl"] = torch.as_tensor(h_abl, dtype=F64)
        if lmo is not None:
            out["lmo"] = torch.as_tensor(lmo, dtype=F64)

        junction_drivers = dict(out)
        for key in ("theta_w", "U_ref", "u_star", "h_abl", "lmo"):
            case_key = _JUNCTION_CASE_KEY.get(key, key)
            if case_key not in case.meteo_junction:
                continue
            values = case.meteo_junction[case_key][k]
            if key == "theta_w":
                junction_drivers[f"{key}_junction"] = apply_conversion(
                    CONTAM_DEG_TO_STREET_RAD, torch.as_tensor(values, dtype=F64), {}
                )
            else:
                junction_drivers[f"{key}_junction"] = torch.as_tensor(values, dtype=F64)
        for key, value in flows.junction_values(junction_drivers).items():
            if value is not None:
                out[f"{key}_junction"] = value
    else:
        out["theta_w"] = torch.tensor(
            float(_circular_mean(theta_w_all.numpy(), axis=0)), dtype=F64
        )
        out["U_ref"] = torch.tensor(float(np.mean(wind_speed)), dtype=F64)
        if u_star is not None:
            out["u_star"] = torch.tensor(float(np.mean(u_star)), dtype=F64)
        if h_abl is not None:
            out["h_abl"] = torch.tensor(float(np.mean(h_abl)), dtype=F64)
        if lmo is not None:
            out["lmo"] = torch.tensor(float(_reciprocal_mean(lmo, axis=0)), dtype=F64)

    return out
