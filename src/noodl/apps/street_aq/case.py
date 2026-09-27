"""The format-neutral street case: ONE `StreetCase` structure that every street-model
source (MUNICH's own files, and, in a later plan, SIRANE's decks) is read into and written
from -- no MUNICH- or SIRANE-specific element belongs on `StreetCase` itself. Each source
model's own file format (names, units, direction conventions) lives in a private module
(`_munich_files` for MUNICH); this module knows only plain SI arrays in the neutral
vocabulary and the `StreetNetwork` they describe.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from noodl.apps.street_aq import _munich_files
from noodl.apps.street_aq.network import StreetNetwork, street_index
from noodl.apps.street_aq.routing import StreetFlows
from noodl.couple import CONTAM_DEG_TO_STREET_RAD, apply_conversion
from noodl.model import Model

F64 = torch.float64

__all__ = [
    "METEO_KEYS",
    "StreetCase",
    "drivers_at",
    "read_case",
    "write_case",
]

METEO_KEYS = ("wind_dir_from_deg", "wind_speed", "h_abl", "u_star", "lmo", "temperature")
"""The neutral meteorology keys a `StreetCase` may carry: the wind direction in degrees
clockwise from north, the direction the wind blows FROM; the reference wind speed (m/s);
the boundary-layer height (m); the friction velocity (m/s); the Obukhov length (m); the
surface temperature (K). The first two are required."""

_REQUIRED_METEO = ("wind_dir_from_deg", "wind_speed")

_JUNCTION_CASE_KEY = {"theta_w": "wind_dir_from_deg", "U_ref": "wind_speed"}
"""`drivers_at`'s driver key -> the `StreetCase.meteo`/`meteo_junction` key it reads from,
for the two keys whose names differ (`theta_w` is stored as `wind_dir_from_deg`, in
degrees, until converted; `U_ref` is stored as `wind_speed`). Every other key
(`u_star`, `h_abl`, `lmo`) is spelled the same on both sides."""

_MUNICH_OPTION_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "Mean_wind_speed_parameterization": (
        "canyon_wind", {"exponential": "exponential", "sirane": "soulhac"}),
    "Transfer_parameterization": ("exchange", {"schulte": "schulte", "sirane": "sirane"}),
    "Building_height_wind_speed_parameterization": (
        "roof_wind_form", {"sirane": "sirane", "macdonald": "macdonald"}),
    "With_horizontal_fluctuation": ("direction_averaging", {"yes": "munich", "no": "none"}),
}
"""MUNICH `[street]` key -> (the `build_model` keyword it sets, MUNICH value (lower case) ->
noodl value). A value missing from its map (e.g. MUNICH's `Wang` transfer) has no noodl
counterpart."""

_MUNICH_FLOAT_MAP = {"Zref": "z_ref", "Minimum_Street_Wind_Speed": "canyon_wind_min"}
"""MUNICH `[street]` key -> the float `build_model` keyword it sets."""

_MUNICH_U_D_MIN = 0.001
"""MUNICH's hard-coded minimum exchange velocity, `min_velocity`
(`StreetNetworkTransport.cxx:206` and `:3295`): not a configuration key, so every MUNICH
case implies it."""

_MUNICH_FLOAT_DEFAULTS = {"Minimum_Street_Wind_Speed": 0.1}
"""`_MUNICH_FLOAT_MAP` keys MUNICH itself defaults when `[street]` lacks them, instead of
`model_options` raising: `Minimum_Street_Wind_Speed` -> `ustreet_min = 0.1`
(`StreetNetworkTransport.cxx:164-168`). `Zref` has no such default and still raises when
absent."""


def _hours_by(value, n_hours: int, tail: tuple[int, ...], label: str) -> np.ndarray:
    """`value` broadcast to `(n_hours, *tail)`: a scalar, a `(n_hours,)` time series, or
    an array whose leading axes are `(n_hours, *tail[:k])` (the rest broadcast)."""
    arr = np.asarray(value, dtype=np.float64)
    full = (n_hours, *tail)
    if arr.ndim == 0:
        return np.full(full, float(arr))
    if arr.ndim > len(full) or arr.shape != full[: arr.ndim]:
        raise ValueError(
            f"StreetCase.synthetic: {label} must be a scalar or have leading shape of "
            f"{full} (n_hours first), got {arr.shape}"
        )
    return np.broadcast_to(arr.reshape(arr.shape + (1,) * (len(full) - arr.ndim)),
                           full).copy()


@dataclass(frozen=True)
class StreetCase:
    """One street-network case: the network plus every array a `Model` built on it needs
    to run, independent of which source model (`source`) it came from.

    Attributes:
        source: Which reader produced this case -- `"munich"` (and, later, `"sirane"`), or
            `"synthetic"` for one built in Python with `StreetCase.synthetic`.
        network: The case's `StreetNetwork` (metres).
        times: `(n_hours,)`, seconds since `start`.
        street_ids: `network.streets`' names, in the order `emissions`'/`background`'s
            street axis uses (== `network.streets` order).
        junction_ids: The source model's OWN node ids for `network.junctions`, in that
            same order (MUNICH: the intersection ids from `intersection.dat`).
        species: The case's species names, in the order `emissions`'/`background`'s last
            axis uses.
        meteo: One `(n_hours, n_streets)` array per `METEO_KEYS` key: `wind_dir_from_deg`
            (degrees clockwise from north, the direction the wind blows FROM) and
            `wind_speed` always; `h_abl`, `u_star`, `lmo`, `temperature` wherever the source
            provides them.
        meteo_junction: The same keys, `(n_hours, n_junctions)`, in `network.junctions`
            order -- may be empty (or missing some keys) when the source has no genuine
            per-junction meteorology.
        emissions: `(n_hours, n_streets, n_species)`, **kg/s** per street.
        background: `(n_hours, n_streets, n_species)`, **kg/m3**, per street.
        native: The source model's own options, as read (MUNICH: one dict per `munich.cfg`
            section, plus the lon/lat projection the reader used) -- `model_options`
            translates them into `build_model` keywords, and `write_case` writes them back
            when the format matches.
        start: The absolute date and time of `times[0]`, or `None` when the case has none
            (a synthetic case built without one; writing it to a file format that needs a
            date then raises).
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
    start: datetime | None = None

    @classmethod
    def synthetic(
        cls,
        network: StreetNetwork,
        *,
        species: Sequence[str],
        times: Sequence[float],
        meteo: Mapping[str, float | np.ndarray],
        emissions: float | np.ndarray,
        background: float | np.ndarray,
        meteo_junction: Mapping[str, float | np.ndarray] | None = None,
        start: datetime | None = None,
    ) -> StreetCase:
        """A case built in Python (`source="synthetic"`), e.g. an idealised network to
        write out for another street model.

        `meteo` takes `METEO_KEYS` keys (`wind_dir_from_deg` and `wind_speed` required);
        each value is a scalar, a `(n_hours,)` series (the same at every street) or a
        `(n_hours, n_streets)` array. `meteo_junction` likewise with `n_junctions`, in
        `network.junctions` order (any subset of `METEO_KEYS`, or none at all). `emissions`
        (kg/s per street) and `background` (kg/m3) are a scalar, `(n_hours,)`,
        `(n_hours, n_streets)` or `(n_hours, n_streets, n_species)`. The junction ids are
        `network.junctions`' own names.

        Writing the case out with `write_case(format="munich")` needs `meteo` to also carry
        `h_abl`, `u_star` and `lmo`: MUNICH requires their `PBLH`, `UST`, `LMO` fields (and
        the `...Inter` junction counterparts derived from them) whenever transport is on,
        which `write_case` always turns on.
        """
        species = list(species)
        times = [float(t) for t in times]
        n_hours, n_streets = len(times), len(network.streets)
        n_junctions, n_species = len(network.junctions), len(species)
        missing = [k for k in _REQUIRED_METEO if k not in meteo]
        if missing:
            raise ValueError(
                f"StreetCase.synthetic: meteo needs {list(_REQUIRED_METEO)}; missing "
                f"{missing}"
            )
        tables = {}
        for label, table, n_columns in (("meteo", meteo, n_streets),
                                        ("meteo_junction", meteo_junction or {}, n_junctions)):
            unknown = sorted(set(table) - set(METEO_KEYS))
            if unknown:
                raise ValueError(
                    f"StreetCase.synthetic: {label} keys {unknown} are not among "
                    f"{list(METEO_KEYS)}"
                )
            tables[label] = {k: _hours_by(v, n_hours, (n_columns,), f"{label}[{k!r}]")
                             for k, v in table.items()}
        return cls(
            source="synthetic", network=network, times=times,
            street_ids=[s.name for s in network.streets],
            junction_ids=list(network.junctions), species=species,
            meteo=tables["meteo"], meteo_junction=tables["meteo_junction"],
            emissions=_hours_by(emissions, n_hours, (n_streets, n_species), "emissions"),
            background=_hours_by(background, n_hours, (n_streets, n_species), "background"),
            native={}, start=start,
        )

    def model_options(self) -> dict:
        """The `build_model` keywords this case's own source model implies, translated from
        its native options.

        MUNICH (`source="munich"`), from `munich.cfg`'s `[street]` section:
        `Mean_wind_speed_parameterization` (`Exponential` -> `canyon_wind="exponential"`,
        `Sirane` -> `"soulhac"`), `Transfer_parameterization` (`Schulte` ->
        `exchange="schulte"`, `Sirane` -> `"sirane"`),
        `Building_height_wind_speed_parameterization` (`Sirane`/`Macdonald` ->
        `roof_wind_form`), `With_horizontal_fluctuation` (`yes` ->
        `direction_averaging="munich"`, `no` -> `"none"`), `Zref` -> `z_ref`,
        `Minimum_Street_Wind_Speed` -> `canyon_wind_min`; always `stability="munich"` and
        MUNICH's hard-coded `u_d_min=0.001`. A value with no noodl counterpart (e.g. the
        `Wang` transfer) raises `NotImplementedError` naming the key and value; a missing
        key raises `ValueError`, except `Minimum_Street_Wind_Speed`, which MUNICH itself
        defaults to `0.1` when absent (see `_MUNICH_FLOAT_DEFAULTS`) -- an unparsable value
        still raises.

        Any other source raises `NotImplementedError`: a synthetic case carries no source
        model's options (pass `build_model`'s keywords directly), and SIRANE's are a later
        plan's job.
        """
        if self.source != "munich":
            raise NotImplementedError(
                f"StreetCase.model_options: source {self.source!r} carries no source-model "
                f"options this function can translate; only 'munich' is implemented"
            )
        street = self.native.get("street", {})

        def value(key: str) -> str:
            if key not in street:
                raise ValueError(
                    f"StreetCase.model_options: munich.cfg [street] has no {key!r}"
                )
            return street[key]

        options: dict = {}
        for key, (keyword, mapping) in _MUNICH_OPTION_MAP.items():
            raw = value(key)
            if raw.strip().lower() not in mapping:
                raise NotImplementedError(
                    f"StreetCase.model_options: MUNICH {key}: {raw} has no noodl "
                    f"counterpart; supported: {sorted(mapping)}"
                )
            options[keyword] = mapping[raw.strip().lower()]
        for key, keyword in _MUNICH_FLOAT_MAP.items():
            if key not in street and key in _MUNICH_FLOAT_DEFAULTS:
                options[keyword] = _MUNICH_FLOAT_DEFAULTS[key]
            else:
                options[keyword] = float(value(key))
        options["stability"] = "munich"
        options["u_d_min"] = _MUNICH_U_D_MIN
        return options


def read_case(path: Path) -> StreetCase:
    """`StreetCase` from `path`: a directory holding `munich.cfg` is read as a MUNICH case;
    a SIRANE master `.dat` file will be read as a SIRANE case once a later plan implements
    it. Anything else raises `ValueError` naming the expectation.
    """
    path = Path(path)
    if path.is_dir() and (path / "munich.cfg").is_file():
        raw = _munich_files.read_munich_case(path)
        return StreetCase(source="munich", **raw)
    raise ValueError(
        f"read_case: {path} is not a directory holding 'munich.cfg' (a MUNICH case)"
    )


def write_case(
    out_dir: Path,
    case: StreetCase,
    *,
    format: str = "munich",
    options: Mapping[str, object] | None = None,
) -> Path:
    """Writes `case` under `out_dir` in `format`, and returns `out_dir`. Only
    `format="munich"` is implemented.

    `case.start` must be set (MUNICH dates every input), and `case.meteo` must have `h_abl`,
    `u_star` and `lmo`: MUNICH needs their `PBLH`, `UST`, `LMO` fields (and the `...Inter`
    junction counterparts derived from them) whenever transport is on, which this writer
    always turns on; a `ValueError` names whichever of the three is missing.

    A case read from MUNICH files writes its own `[street]` section and projection back
    (`read_case` -> `write_case` round-trips those two); `[options]` itself always turns
    chemistry, photolysis, deposition and scavenging off, and the six `[meteo]` fields
    MUNICH always requires (`Rain`, `SolarRadiation`, `SpecificHumidity`, `SurfacePressure`,
    `SurfaceTemperature`, `Attenuation` -- see `_munich_files._REQUIRED_METEO_DEFAULTS`) come
    from the case where it has one (only `SurfaceTemperature`, from `meteo["temperature"]`),
    else a constant default -- either way, `options` overrides them.

    `options` are the format's own overrides -- for MUNICH, any `munich.cfg` `[street]` key;
    any of the six mandatory `[meteo]` fields; plus `lat0_deg`/`lon0_deg`, the lon/lat of a
    synthetic network's `(0, 0)`. `options` overrides win even over a value the case itself
    supplies. Missing junction (`*Inter`) meteorology is derived from the streets meeting at
    each junction: circular mean for the direction, `1/L` for the Obukhov length, arithmetic
    mean otherwise.
    """
    if format != "munich":
        raise NotImplementedError(
            f"write_case: format {format!r} is not implemented; only 'munich' is"
        )
    if case.start is None:
        raise ValueError(
            "write_case: case.start is None; MUNICH needs an absolute start date "
            "(set StreetCase.start)"
        )
    return _munich_files.write_munich_case(
        out_dir, network=case.network, times=case.times, start=case.start,
        junction_ids=case.junction_ids,
        species=case.species, meteo=case.meteo, meteo_junction=case.meteo_junction,
        emissions=case.emissions, background=case.background,
        native=case.native if case.source == "munich" else None, options=options,
    )


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

    The model's street order (`street_index(model)`) must be `case.street_ids`: the
    per-street arrays are placed by position. `species` (default `case.species`) selects
    and orders the species.

    `u_star` is supplied whenever `case` has it; `U_ref` is `case`'s own wind speed (not
    derived through noodl's log law -- `u_star`, when present, is what actually sets the
    friction velocity; `U_ref` is needed only for the direction spread
    `sigma_theta = sigma_v / U_ref`).
    """
    flows = next(c for c in model.closures if isinstance(c, StreetFlows))
    layer = model.transport[flows.layer_name]
    n_streets = len(case.street_ids)

    model_streets = list(street_index(model, layer_name=flows.layer_name))
    if model_streets != list(case.street_ids):
        raise ValueError(
            f"drivers_at: the model's street order {model_streets} is not the case's "
            f"street_ids {list(case.street_ids)}; build the model on case.network"
        )
    missing = [key for key in _REQUIRED_METEO if key not in case.meteo]
    if missing:
        raise KeyError(
            f"drivers_at: case.meteo has no {missing}; every StreetCase needs "
            f"{list(_REQUIRED_METEO)}"
        )

    species_names = list(species) if species is not None else list(case.species)
    unknown = [s for s in species_names if s not in case.species]
    if unknown:
        raise ValueError(
            f"drivers_at: species {unknown} are not in the case's species {case.species}"
        )
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
            float(_munich_files.circular_mean_rad(theta_w_all.numpy(), axis=0)), dtype=F64
        )
        out["U_ref"] = torch.tensor(float(np.mean(wind_speed)), dtype=F64)
        if u_star is not None:
            out["u_star"] = torch.tensor(float(np.mean(u_star)), dtype=F64)
        if h_abl is not None:
            out["h_abl"] = torch.tensor(float(np.mean(h_abl)), dtype=F64)
        if lmo is not None:
            out["lmo"] = torch.tensor(
                float(_munich_files.reciprocal_mean(lmo, axis=0)), dtype=F64
            )

    return out
