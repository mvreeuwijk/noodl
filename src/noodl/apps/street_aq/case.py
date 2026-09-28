"""The format-neutral street case: ONE `StreetCase` structure that every street-model
source (MUNICH's own files, SIRANE's decks) is read into and written from -- no MUNICH- or
SIRANE-specific element belongs on `StreetCase` itself. Each source model's own file format
(names, units, direction conventions) lives in a private module (`_munich_files` for MUNICH,
`_sirane_files` for SIRANE); this module knows only plain SI arrays in the neutral
vocabulary and the `StreetNetwork` they describe.
"""
from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from noodl.apps.street_aq import _munich_files, _sirane_files
from noodl.apps.street_aq.network import StreetNetwork, street_index
from noodl.apps.street_aq.routing import StreetFlows
from noodl.couple import CONTAM_DEG_TO_STREET_RAD, apply_conversion
from noodl.model import Model

F64 = torch.float64

__all__ = [
    "METEO_KEYS",
    "StreetCase",
    "StreetResults",
    "drivers_at",
    "read_case",
    "read_results",
    "write_case",
    "write_sweep",
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
        source: Which reader produced this case -- `"munich"` or `"sirane"`, or
            `"synthetic"` for one built in Python with `StreetCase.synthetic`.
        network: The case's `StreetNetwork` (metres).
        times: `(n_hours,)`, seconds since `start`.
        street_ids: `network.streets`' names, in the order `emissions`'/`background`'s
            street axis uses (== `network.streets` order).
        junction_ids: The source model's OWN node ids for `network.junctions`, in that
            same order (MUNICH: the intersection ids from `intersection.dat`; SIRANE: the
            node ids of the network's `NDDEB`/`NDFIN` fields, which are also the junction
            names).
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
            section, plus the lon/lat projection the reader used; SIRANE: the master file's
            options by SIRANE keyword, the two site files, each street's network fields,
            the one-sided streets and the raw meteo columns -- see
            `_sirane_files.read_sirane_case`) -- `model_options`
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

        SIRANE (`source="sirane"`): SIRANE's closure set, which its master file does not
        switch -- `canyon_wind="soulhac"` (the in-canyon velocity of Soulhac et al. 2008),
        `exchange="sirane"` (`u_d = sigma_w / (sqrt(2) pi)`), `routing="sirane"` (the
        non-crossing-streamline junction exchange), `direction_averaging="sirane"` (SIRANE's
        normalised Gaussian integral of the junction exchange over the direction spread,
        Soulhac et al. 2011, Eq. 7, evaluated exactly: the integrand is piecewise constant
        in the direction; SIRANE's spread varies hourly, so it is supplied as the
        `sigma_theta` driver rather than fixed here), `stability="munich"` (SIRANE's
        three-branch stable/neutral/unstable `sigma_w`; noodl physics' `"munich"` form is
        the closest it has). `z_ref` is left at `build_model`'s default: noodl physics has no
        SIRANE meteorological preprocessor (SIRANE derives u*, the boundary-layer height,
        the Obukhov length and sigma_theta from the meteo site's wind, temperature and cloud
        cover, over that site's own roughness), so a SIRANE case is driven with SIRANE's own
        u* -- e.g. from its `Resul_Meteo.dat` -- through the `u_star` driver, never through
        noodl's log law from the measured `wind_speed`. Likewise the dispersion site's `Z0D`
        and `ZDISPL` are recorded in `native["site_disp"]` but not used: noodl physics'
        canopy takes `d = 2 h_mean / 3` and `z0 = h_mean / 10` from the network.

        Any other source raises `NotImplementedError`: a synthetic case carries no source
        model's options (pass `build_model`'s keywords directly).
        """
        if self.source == "sirane":
            return {
                "canyon_wind": "soulhac",
                "exchange": "sirane",
                "routing": "sirane",
                "direction_averaging": "sirane",
                "stability": "munich",
            }
        if self.source != "munich":
            raise NotImplementedError(
                f"StreetCase.model_options: source {self.source!r} carries no source-model "
                f"options this function can translate; only 'munich' and 'sirane' are "
                f"implemented"
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


@dataclass(frozen=True)
class StreetResults:
    """One street model's own results, read format-neutral by `read_results` from a SIRANE
    result directory or a MUNICH `results/` directory.

    Attributes:
        source: `"sirane"` or `"munich"` (which reader produced it).
        times: The absolute time of each output hour.
        street_ids: What `c_in`/`c_above`/`u_canyon`/`sigma_w_roof`/`u_exchange`'s street
            axis indexes, in that order (SIRANE: its own record-order ids -- matching
            `StreetCase.street_ids` for a case read from the same deck; MUNICH:
            `case.street_ids`).
        species: `c_in`/`c_above`'s dict keys, in their in-file order.
        c_in: One `(n_hours, n_streets)` array per species, kg/m3 -- the in-canyon
            concentration (SIRANE `Cint`; MUNICH's own street concentration).
        c_above: The same, kg/m3, for the concentration just above the canyon (SIRANE
            `Cext`) -- `{}` for MUNICH, which has no such output.
        u_canyon: `(n_hours, n_streets)`, m/s, the mean in-canyon velocity (SIRANE
            `U_moy`) -- `None` for MUNICH.
        sigma_w_roof: `(n_hours, n_streets)`, m/s, the vertical-velocity fluctuation at
            roof height (SIRANE `Sigma_wH`) -- `None` for MUNICH.
        u_exchange: `(n_hours, n_streets)`, m/s, the roof-level exchange velocity, read AS
            PRINTED (SIRANE `u_d`), not recomputed -- `None` for MUNICH. Pinned against
            SIRANE's own output: `u_exchange == sigma_w_roof * SIRANE_EXCHANGE`
            (`sigma_w_roof / (sqrt(2) pi)`, `noodl.apps.street_aq.canyon`), to the two
            decimals SIRANE prints.
        meteo: `u_star`, `sigma_theta` (radians -- SIRANE's own `SigmaTheta` is degrees),
            `h_abl`, `lmo`, `wind_speed`, `wind_dir_from_deg`, `temperature` (K), each
            `(n_hours, n_streets)`, network-wide (SIRANE's `Resul_Meteo.dat`, one
            meteorological station for the whole case) -- `{}` for MUNICH, whose own
            meteorology is the case's `meteo`, not a result.
    """

    source: str
    times: list[datetime]
    street_ids: list[str]
    species: list[str]
    c_in: dict[str, np.ndarray]
    c_above: dict[str, np.ndarray]
    u_canyon: np.ndarray | None
    sigma_w_roof: np.ndarray | None
    u_exchange: np.ndarray | None
    meteo: dict[str, np.ndarray]


def read_results(
    path: Path, *, case: StreetCase | None = None, hours: str = "case"
) -> StreetResults:
    """`StreetResults` from `path`: a SIRANE result directory (holding `RUES_PAR_HEURE/` and
    `METEO/Resul_Meteo.dat`) or a MUNICH `results/` directory of `<species>.bin` files (see
    `_sirane_files.read_sirane_results`, `_munich_files.read_munich_results` for what each
    reads).

    A MUNICH results directory always needs `case`: its binaries carry no street order,
    species or absolute time of their own. A SIRANE result directory needs it only for
    `hours="case"` (the default): the hours inside `case`'s own period (`case.start` +
    `case.times`) -- an archived result directory can mix hours from more than one run (see
    `tests/data/street/sirane_south_kensington`'s NOTICE.md), so this is the safer default;
    a case hour missing from the directory raises `ValueError` naming it. `hours="all"`
    returns every hour the directory holds, in time order, with or without a `case`.
    `hours` applies to a SIRANE result directory only: a MUNICH result is the case's own
    period, so any other `hours` than `"case"` is refused for it by name.

    A path with no `RUES_PAR_HEURE/` and no `case` is refused, naming both expectations.
    """
    path = Path(path)
    if (path / "RUES_PAR_HEURE").is_dir():
        raw = _sirane_files.read_sirane_results(path, case=case, hours=hours)
        return StreetResults(source="sirane", **raw)
    if case is None:
        raise ValueError(
            f"read_results: {path} has no RUES_PAR_HEURE (not a SIRANE result directory) "
            f"and no case was given to read it as a MUNICH results/ directory"
        )
    if hours != "case":
        raise ValueError(
            f"read_results: hours={hours!r} applies to a SIRANE result directory only; "
            f"{path} is read as a MUNICH results/ directory, which covers the case's own "
            f"period (hours='case')"
        )
    raw = _munich_files.read_munich_results(path, case=case)
    return StreetResults(source="munich", **raw)


def read_case(path: Path) -> StreetCase:
    """`StreetCase` from `path`: a directory holding `munich.cfg` is read as a MUNICH case;
    a `.dat` file is read as a SIRANE master file (`Donnees_*.dat`) with the deck it names
    (see `_sirane_files.read_sirane_case` for what is read and what is refused). Anything
    else raises `ValueError` naming both expectations.
    """
    path = Path(path)
    if path.is_dir() and (path / "munich.cfg").is_file():
        raw = _munich_files.read_munich_case(path)
        return StreetCase(source="munich", **raw)
    if path.is_file() and path.suffix.lower() == ".dat":
        raw = _sirane_files.read_sirane_case(path)
        return StreetCase(source="sirane", **raw)
    raise ValueError(
        f"read_case: {path} is neither a directory holding 'munich.cfg' (a MUNICH case) "
        f"nor a SIRANE master .dat file"
    )


def write_case(
    out_dir: Path,
    case: StreetCase,
    *,
    format: str = "munich",
    options: Mapping[str, object] | None = None,
) -> Path:
    """Writes `case` under `out_dir` in `format` (`"munich"` or `"sirane"`), and returns
    `out_dir`. `case.start` must be set: both formats date every input.

    **`format="munich"`**: `case.meteo` must have `h_abl`, `u_star` and `lmo`: MUNICH needs
    their `PBLH`, `UST`, `LMO` fields (and the `...Inter` junction counterparts derived from
    them) whenever transport is on, which this writer always turns on; a `ValueError` names
    whichever of the three is missing.

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

    **`format="sirane"`**: a SIRANE v2.1 deck, master file `out_dir / "Donnees.dat"` in
    French labels (see `_sirane_files.write_sirane_case` for every file and default).
    `times` must be whole consecutive hours; meteorology, background and the streets'
    `z0_b` must each be one value for the network (SIRANE has one meteo station, one
    background and one building roughness); the wind speed a multiple of 0.1 m/s and the
    direction whole degrees (the meteo file's format); the species SIRANE species (a
    passive tracer is written as an existing one, e.g. NO2, with chemistry and deposition
    off). Masses are written in SIRANE's units (g/s, micrograms/m3). The deck's input folder is
    `out_dir` relative to its parent (SIRANE's working directory) and its results folder
    `<out_dir>/RESULT` (created in advance, with SIRANE's result subfolders).
    Every numeric setting is checked against SIRANE's own range and refused by name
    outside it. A case read from a SIRANE deck writes its own physics and
    numerical settings, site files, street fields and species flags back. `options`:
    `chapman` (0/1, SIRANE's Chapman NO-NO2-O3 chemistry; default 0), `plume` (0/1,
    SIRANE's street-plume model above the roofs; default 1), `deposition` (0/1; default
    0), `latitude` (deg; default 51.5), `measurement_height` (m, the height of the wind
    speed; default 10), `input_dir`/`result_dir` (SIRANE's two folders, relative to its
    working directory), and SIRANE's own keyword for any physics or numerical setting it
    writes back (e.g. `U_MIN`).
    """
    if format not in ("munich", "sirane"):
        raise NotImplementedError(
            f"write_case: format {format!r} is not implemented; only 'munich' and 'sirane' "
            f"are"
        )
    if case.start is None:
        raise ValueError(
            f"write_case: case.start is None; {format} needs an absolute start date "
            f"(set StreetCase.start)"
        )
    if format == "sirane":
        _sirane_files.write_sirane_case(
            out_dir, network=case.network, times=case.times, start=case.start,
            junction_ids=case.junction_ids, species=case.species, meteo=case.meteo,
            emissions=case.emissions, background=case.background,
            native=case.native if case.source == "sirane" else None,
            options=dict(options or {}),
        )
        return Path(out_dir)
    return _munich_files.write_munich_case(
        out_dir, network=case.network, times=case.times, start=case.start,
        junction_ids=case.junction_ids,
        species=case.species, meteo=case.meteo, meteo_junction=case.meteo_junction,
        emissions=case.emissions, background=case.background,
        native=case.native if case.source == "munich" else None, options=options,
    )


_SWEEP_START = datetime(2014, 1, 7)
"""The sweep's default start (00:00, 7 January 2014): the archived South Kensington SIRANE
results' own night-time hour. SIRANE's meteorological preprocessor sets the stability from the solar
elevation, so the hour of day matters."""

_RUN_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def write_sweep(
    out_dir: Path,
    base_case: StreetCase,
    *,
    directions_deg: Sequence[float],
    speeds: Sequence[float],
    sources: str | Sequence[str] = "unit_impulse",
    species: str = "NO2",
    variants: Mapping[str, Mapping[str, object]] | None = None,
    chapman: int = 0,
) -> Path:
    """Writes a sweep of SIRANE decks under `out_dir` -- one per (direction, speed, source
    street, variant) -- plus `runs.csv`, a manifest of each deck's parameters; returns
    `out_dir`.

    Each deck follows the South Kensington deck: `base_case.network`, two hours from
    `base_case.start` (default `_SWEEP_START`) -- the first a warm-up -- with
    the wind from `direction` (degrees clockwise from north, whole degrees) at
    `speed` (m/s at the meteo site's height, a multiple of 0.1), the
    temperature of `base_case`'s first hour when it has one, zero background, and a unit
    emission (1 g/s, i.e. 1e-3 kg/s) of `species` on the source street only.
    `sources="unit_impulse"` takes every street in turn; a sequence of street ids
    (`base_case.street_ids`) takes those. With `chapman=0` (default) the case has
    `species` alone -- a passive tracer written as that SIRANE species, chemistry off; with
    `chapman=1` it has NO2, NO and O3 (`species` among them) and SIRANE's chemistry on.
    Deposition is off unless a variant sets `deposition`. A `base_case` read from a SIRANE
    deck keeps its site files, street fields and physics settings.

    Every argument, every variant's options and the base case itself (by writing the
    first deck to a temporary directory) are checked before anything is written under
    `out_dir`. Writing into an existing `out_dir` leaves any older decks there in place;
    `runs.csv` lists only this sweep's decks.

    `variants`: name -> `write_case` options for that variant (default
    `{"plume_on": {"plume": 1}, "plume_off": {"plume": 0}}`); `chapman` applies to every
    variant that does not set its own. Names must be letters, digits, `_` or `-`.

    Layout: `decks/<run-id>/` (each deck, with its results folder `decks/<run-id>/RESULT/`,
    whose SIRANE output reads back with `read_results`). The run id is
    `d<direction>_u<speed>_s<street index>_<variant>`, e.g. `d045_u5p0_s07_plume_on` (the street
    index is SIRANE's own street number, the record order; `runs.csv` maps it to the case's
    street id).
    """
    out_dir = Path(out_dir)
    variants = dict(variants if variants is not None
                    else {"plume_on": {"plume": 1}, "plume_off": {"plume": 0}})
    if not variants:
        raise ValueError("write_sweep: variants is empty; give at least one")
    bad = [name for name in variants if not _RUN_NAME.match(name)]
    if bad:
        raise ValueError(
            f"write_sweep: variant name(s) {bad} must be letters, digits, '_' or '-' (they "
            f"become file names)"
        )
    if chapman not in (0, 1):
        raise ValueError(f"write_sweep: chapman must be 0 or 1, got {chapman!r}")
    if chapman == 1:
        if species not in ("NO2", "NO", "O3"):
            raise ValueError(
                f"write_sweep: with chapman=1 the emitted species must be NO2, NO or O3, got "
                f"{species!r}"
            )
        case_species = ["NO2", "NO", "O3"]
    else:
        case_species = [species]
    def run_options(variant_options: Mapping[str, object]) -> dict:
        own_chapman = {"chapman", "CHAPMAN"} & set(variant_options)
        return dict({} if own_chapman else {"chapman": chapman}, **variant_options)

    for variant_options in variants.values():
        forbidden = sorted({"input_dir", "result_dir"} & set(variant_options))
        if forbidden:
            raise ValueError(
                f"write_sweep: a variant may not set {forbidden}; the sweep lays out its "
                f"own decks and results"
            )
        _sirane_files.check_sirane_options(run_options(variant_options), case_species,
                                           who="write_sweep")
    directions = [float(d) for d in directions_deg]
    speed_values = [float(u) for u in speeds]
    if not directions or not speed_values:
        raise ValueError("write_sweep: directions_deg and speeds must each be non-empty")
    for d in directions:
        if abs(d - round(d)) > 1e-9:
            raise ValueError(
                f"write_sweep: direction {d} deg is not whole degrees (SIRANE's meteo file "
                f"format)"
            )
    for u in speed_values:
        if not u > 0 or abs(u * 10 - round(u * 10)) > 1e-9:
            raise ValueError(
                f"write_sweep: speed {u} m/s must be positive and a multiple of 0.1 m/s "
                f"(SIRANE's meteo file format)"
            )
    street_ids = list(base_case.street_ids)
    if isinstance(sources, str):
        if sources != "unit_impulse":
            raise ValueError(
                f"write_sweep: sources must be 'unit_impulse' or a sequence of street ids, "
                f"got {sources!r}"
            )
        source_streets = street_ids
    else:
        source_streets = list(sources)
        unknown = [s for s in source_streets if s not in street_ids]
        if unknown:
            raise ValueError(
                f"write_sweep: source street(s) {unknown} are not in base_case.street_ids"
            )
    start = base_case.start or _SWEEP_START
    n_streets = len(street_ids)
    width = len(str(n_streets - 1))
    meteo_base = {}
    if "temperature" in base_case.meteo:
        meteo_base["temperature"] = float(np.mean(base_case.meteo["temperature"][0]))
    native: dict = {}
    if base_case.source == "sirane":
        native = {k: v for k, v in base_case.native.items()
                  if k in ("options", "site_disp", "site_meteo", "streets")}
        raw = base_case.native.get("meteo_raw", {})
        native["meteo_raw"] = {k: [v[0], v[0]] for k, v in raw.items() if v}

    runs = []
    for d in directions:
        for u in speed_values:
            for source in source_streets:
                index = street_ids.index(source)
                for variant, variant_options in variants.items():
                    run_id = (f"d{int(round(d)) % 360:03d}_u{u:.1f}".replace(".", "p")
                              + f"_s{index:0{width}d}_{variant}")
                    runs.append({"run_id": run_id, "direction_deg": d, "speed": u,
                                 "source_street": source, "source_index": index,
                                 "variant": variant, "options": run_options(variant_options)})
    ids = [r["run_id"] for r in runs]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"write_sweep: run ids {duplicated[:5]} repeat (a direction, speed or source "
            f"given twice, or two directions equal modulo 360?)"
        )

    s_index = case_species.index(species)
    zeros = np.zeros((2, n_streets, len(case_species)), dtype=np.float64)

    def write_run(root: Path, run: dict) -> None:
        emissions = zeros.copy()
        emissions[:, run["source_index"], s_index] = 1e-3
        meteo = {key: np.full((2, n_streets), value, dtype=np.float64)
                 for key, value in dict(meteo_base, wind_dir_from_deg=run["direction_deg"],
                                        wind_speed=run["speed"]).items()}
        case = StreetCase(
            source="sirane" if native else "synthetic", network=base_case.network,
            times=[0.0, 3600.0], street_ids=street_ids,
            junction_ids=list(base_case.junction_ids), species=case_species, meteo=meteo,
            meteo_junction={}, emissions=emissions, background=zeros.copy(),
            native=native, start=start,
        )
        run_id = run["run_id"]
        options = dict(run["options"], input_dir=f"decks/{run_id}",
                       result_dir=f"decks/{run_id}/{_sirane_files.RESULT_SUBDIR}")
        write_case(root / "decks" / run_id, case, format="sirane", options=options)

    # Every deck-level check (the network, the base case's own settings) runs on a throwaway
    # copy of the first deck before anything is written under out_dir.
    with tempfile.TemporaryDirectory() as scratch:
        write_run(Path(scratch), runs[0])
    for run in runs:
        write_run(out_dir, run)

    _sirane_files.write_sweep_manifest(out_dir, runs=runs)
    return out_dir


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
    `sigma_theta = sigma_v / U_ref`). `temperature` (K) is supplied whenever `case` has it;
    the photostationary chemistry evaluates its rate there.
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
    temperature = case.meteo["temperature"][k] if "temperature" in case.meteo else None
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
        if temperature is not None:
            out["temperature"] = torch.as_tensor(temperature, dtype=F64)

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
        if temperature is not None:
            out["temperature"] = torch.tensor(float(np.mean(temperature)), dtype=F64)

    return out
